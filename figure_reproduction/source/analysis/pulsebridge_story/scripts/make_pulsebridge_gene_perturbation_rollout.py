#!/usr/bin/env python3
"""PulseBridge in-model gene/program perturbation followed by rollout.

This is the explicit perturbation analysis:
day0 latent state + gene-program PCA push -> trained PulseBridge ODE rollout
-> day3 gene/program readout through observed latent nearest neighbors.

The readout intentionally avoids hand-defined state bins.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cellcompass_pulsebridge_gene_perturb")
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/cellcompass_pulsebridge_gene_perturb_numba")

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D
from scipy import sparse
from sklearn.neighbors import NearestNeighbors


REPO = Path(__file__).resolve().parents[3]
BUNDLE = REPO / "downloads" / "pulsebridge_welltemp_hct116_20260627"
OUT = REPO / "analysis" / "pulsebridge_story" / "outputs"
FIG = OUT / "figures"
TAB = OUT / "tables"
ADATA_PATH = (
    BUNDLE
    / "home"
    / "ubuntu"
    / ".cellcompass"
    / "algorithm_benchmarks"
    / "datasets"
    / "pcib_welltemp_hct116_raw_pool_20k"
    / "data.h5ad"
)
MODEL_STATE_PATH = (
    BUNDLE
    / "data"
    / "cytobridge"
    / "runs"
    / "night_wave_20260621_0140"
    / "existing_deep_raw_a"
    / "training_runs"
    / "20260620T234044691964Z__pilot__pulse_control_identifiability_bridge__campaign_20260-trial_20260620-pcib_welltem"
    / "artifacts"
    / "model_state.pt"
)

BLUE = "#2F5D8C"
TEAL = "#1B8A8F"
GOLD = "#A17C00"
PURPLE = "#8A4F9E"
GRAY = "#6B7280"
RED = "#B23A48"

PERTURBATION_MODULES = {
    "reactivation": {
        "label": "DKK1/GAS5\nreactivation",
        "genes": ["DKK1", "GAS5"],
        "target_program": "reactivation",
        "color": BLUE,
    },
    "immune_surface": {
        "label": "B2M/HLA/IFITM3\nimmune surface",
        "genes": ["B2M", "HLA-A", "HLA-B", "HLA-C", "IFITM3"],
        "target_program": "immune_surface",
        "color": TEAL,
    },
    "proliferation_genes": {
        "label": "EREG/PCNA\nrepression readout",
        "genes": ["EREG", "PCNA"],
        "target_program": "proliferation_repression",
        "color": RED,
    },
    "adaptive_metabolic": {
        "label": "SAT1/LGALS1/MT\nadaptive",
        "genes": ["SAT1", "LGALS1", "MT-CO1", "MT-CO2", "MT-CO3", "MT-ND4", "MT-CYB"],
        "target_program": "adaptive_metabolic",
        "color": GOLD,
    },
}

READOUT_PROGRAMS = {
    "source_aligned_antitumor": {
        "label": "anti-tumor-like",
        "up": ["DKK1", "GAS5", "B2M", "HLA-A", "HLA-B", "HLA-C", "IFITM3"],
        "down": ["EREG", "PCNA"],
        "color": BLUE,
    },
    "reactivation": {
        "label": "DKK1/GAS5",
        "up": ["DKK1", "GAS5"],
        "down": [],
        "color": BLUE,
    },
    "immune_surface": {
        "label": "B2M/HLA/IFITM3",
        "up": ["B2M", "HLA-A", "HLA-B", "HLA-C", "IFITM3"],
        "down": [],
        "color": TEAL,
    },
    "proliferation_repression": {
        "label": "EREG/PCNA down",
        "up": [],
        "down": ["EREG", "PCNA"],
        "color": RED,
    },
    "adaptive_metabolic": {
        "label": "SAT1/LGALS1/MT",
        "up": ["SAT1", "LGALS1", "MT-CO1", "MT-CO2", "MT-CO3", "MT-ND4", "MT-CYB"],
        "down": [],
        "color": GOLD,
    },
    "therapeutic_margin": {
        "label": "anti-tumor minus adaptive",
        "derived": ("source_aligned_antitumor", "adaptive_metabolic"),
        "color": PURPLE,
    },
}

GENE_READOUT_ORDER = ["SAT1", "LGALS1", "MT-CO1", "B2M", "DKK1", "PCNA", "EREG"]
GENE_COLORS = {
    "SAT1": GOLD,
    "LGALS1": PURPLE,
    "MT-CO1": "#6B7F2A",
    "B2M": TEAL,
    "DKK1": BLUE,
    "PCNA": RED,
    "EREG": RED,
}


def dense(x):
    return x.toarray() if sparse.issparse(x) else np.asarray(x)


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.linewidth": 0.9,
            "axes.edgecolor": "#111827",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 9.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 8.2,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_model(latent_dim: int):
    cyto_path = str(REPO / "CytoBridge-main")
    if cyto_path not in sys.path:
        sys.path.insert(0, cyto_path)
    from CytoBridge.tl.models import DynamicalModel

    payload = torch.load(MODEL_STATE_PATH, map_location="cpu", weights_only=False)
    record = payload["all_model"]
    model = DynamicalModel(int(latent_dim), record["model_config"])
    state = {k: torch.as_tensor(v) for k, v in record["model_state_dict"].items()}
    model.load_state_dict(state)
    model.eval()
    return model, record


def euler_rollout(
    model,
    x0: np.ndarray,
    t0: float = 0.0,
    t1: float = 3.0,
    n_steps: int = 60,
    batch_size: int = 1024,
) -> np.ndarray:
    device = torch.device("cpu")
    model = model.to(device)
    out = np.empty_like(x0, dtype=np.float32)
    dt = (t1 - t0) / float(n_steps)
    for start in range(0, x0.shape[0], batch_size):
        stop = min(start + batch_size, x0.shape[0])
        z = torch.as_tensor(x0[start:stop], dtype=torch.float32, device=device)
        with torch.no_grad():
            for step in range(n_steps):
                t = torch.full((z.shape[0], 1), t0 + step * dt, dtype=torch.float32, device=device)
                lnw = torch.zeros((z.shape[0], 1), dtype=torch.float32, device=device)
                vel = model(t, z, lnw, except_interaction=True)["velocity"]
                z = z + float(dt) * vel
        out[start:stop] = z.detach().cpu().numpy()
    return out


def program_delta(adata: ad.AnnData, genes: list[str], strength: float) -> tuple[np.ndarray, list[str]]:
    var_to_idx = {str(g): i for i, g in enumerate(adata.var_names)}
    used = [g for g in genes if g in var_to_idx]
    if not used:
        raise ValueError(f"No genes found for perturbation program: {genes}")
    pcs = np.asarray(adata.varm["PCs"], dtype=np.float32)
    latent_dim = int(adata.obsm["X_latent"].shape[1])
    pcs = pcs[:, :latent_dim]
    idx = np.asarray([var_to_idx[g] for g in used], dtype=int)
    return float(strength) * pcs[idx].mean(axis=0).astype(np.float32), used


def selected_expression(adata: ad.AnnData) -> tuple[list[str], np.ndarray, np.ndarray]:
    genes = sorted(
        {
            g
            for meta in READOUT_PROGRAMS.values()
            for key in ("up", "down")
            for g in meta.get(key, [])
        }
        | set(GENE_READOUT_ORDER)
    )
    present = [g for g in genes if g in set(adata.var_names)]
    idx = [int(adata.var_names.get_loc(g)) for g in present]
    layer = adata.layers["log1p_norm"] if "log1p_norm" in adata.layers else adata.X
    expr = dense(layer[:, idx]).astype(np.float32)
    mu = expr.mean(axis=0, keepdims=True)
    sd = expr.std(axis=0, keepdims=True)
    zexpr = (expr - mu) / np.maximum(sd, 1e-6)
    return present, expr, zexpr


def neighbor_gene_readout(
    nn: NearestNeighbors,
    points: np.ndarray,
    expr_ref: np.ndarray,
    zexpr_ref: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    _dist, ind = nn.kneighbors(points)
    return expr_ref[ind].mean(axis=1), zexpr_ref[ind].mean(axis=1)


def program_scores(zreadout: np.ndarray, gene_names: list[str]) -> dict[str, np.ndarray]:
    gene_to_col = {g: i for i, g in enumerate(gene_names)}
    scores: dict[str, np.ndarray] = {}
    for program, meta in READOUT_PROGRAMS.items():
        if "derived" in meta:
            continue
        up = [gene_to_col[g] for g in meta.get("up", []) if g in gene_to_col]
        down = [gene_to_col[g] for g in meta.get("down", []) if g in gene_to_col]
        vals = np.zeros(zreadout.shape[0], dtype=np.float32)
        pieces = 0
        if up:
            vals += zreadout[:, up].mean(axis=1)
            pieces += 1
        if down:
            vals -= zreadout[:, down].mean(axis=1)
            pieces += 1
        if pieces > 1:
            vals /= float(pieces)
        scores[program] = vals
    scores["therapeutic_margin"] = scores["source_aligned_antitumor"] - scores["adaptive_metabolic"]
    return scores


def compute() -> dict[str, pd.DataFrame]:
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    adata = ad.read_h5ad(ADATA_PATH)
    latent = np.asarray(adata.obsm["X_latent"], dtype=np.float32)
    times = adata.obs["time_point_processed"].astype(float).to_numpy()
    day0_idx = np.where(times == float(np.min(times)))[0]
    day3_idx = np.where(times == float(np.max(times)))[0]
    x0 = latent[day0_idx]

    model, record = load_model(latent.shape[1])
    gene_names, expr, zexpr = selected_expression(adata)
    expr_day0 = expr[day0_idx]
    zexpr_day0 = zexpr[day0_idx]
    expr_day3 = expr[day3_idx]
    zexpr_day3 = zexpr[day3_idx]
    nn_day0 = NearestNeighbors(n_neighbors=30).fit(latent[day0_idx])
    nn_day3 = NearestNeighbors(n_neighbors=30).fit(latent[day3_idx])

    scenarios = [("baseline", "baseline", "baseline", 0.0, np.zeros(latent.shape[1], dtype=np.float32), "")]
    for module, meta in PERTURBATION_MODULES.items():
        for strength in (-5.0, 5.0):
            delta, used = program_delta(adata, meta["genes"], strength)
            scenarios.append((f"{module} {strength:+.0f}z", module, meta["target_program"], strength, delta, ",".join(used)))

    program_rows = []
    initial_terminal_rows = []
    gene_rows = []
    trajectory_rows = []

    for scenario, module, target_program, strength, delta, genes_used in scenarios:
        init_points = x0 + delta[None, :]
        _expr_init, z_init = neighbor_gene_readout(nn_day0, init_points, expr_day0, zexpr_day0)
        init_scores = program_scores(z_init, gene_names)
        terminal = euler_rollout(model, init_points)
        expr_read, z_read = neighbor_gene_readout(nn_day3, terminal, expr_day3, zexpr_day3)
        term_scores = program_scores(z_read, gene_names)
        for program, vals in term_scores.items():
            meta = READOUT_PROGRAMS[program]
            program_rows.append(
                {
                    "scenario": scenario,
                    "perturbation_module": module,
                    "target_program": target_program,
                    "strength_z": float(strength),
                    "program": program,
                    "program_label": meta["label"],
                    "mean_score": float(vals.mean()),
                    "sem": float(vals.std(ddof=1) / np.sqrt(vals.shape[0])),
                    "genes_used": genes_used,
                    "n_day0_cells": int(x0.shape[0]),
                }
            )
            initial_terminal_rows.append(
                {
                    "scenario": scenario,
                    "perturbation_module": module,
                    "target_program": target_program,
                    "strength_z": float(strength),
                    "program": program,
                    "program_label": meta["label"],
                    "initial_mean_score": float(init_scores[program].mean()),
                    "terminal_mean_score": float(vals.mean()),
                    "genes_used": genes_used,
                    "n_day0_cells": int(x0.shape[0]),
                }
            )
        gene_to_col = {g: i for i, g in enumerate(gene_names)}
        for gene in GENE_READOUT_ORDER:
            if gene not in gene_to_col:
                continue
            col = gene_to_col[gene]
            gene_rows.append(
                {
                    "scenario": scenario,
                    "perturbation_module": module,
                    "target_program": target_program,
                    "strength_z": float(strength),
                    "gene": gene,
                    "mean_log1p_norm_readout": float(expr_read[:, col].mean()),
                    "mean_z_readout": float(z_read[:, col].mean()),
                    "genes_used": genes_used,
                    "n_day0_cells": int(x0.shape[0]),
                }
            )

        # Keep a compact trajectory trace for reproducibility without storing all cells.
        if scenario in {"baseline", "adaptive_metabolic -5z", "adaptive_metabolic +5z"}:
            for t in (0.0, 1.0, 2.0, 3.0):
                n = int(round(60 * t / 3.0))
                point = euler_rollout(model, x0[:800] + delta[None, :], t0=0.0, t1=t, n_steps=max(n, 1)) if t > 0 else x0[:800] + delta[None, :]
                expr_t, z_t = neighbor_gene_readout(nn_day3, point, expr_day3, zexpr_day3)
                score_t = program_scores(z_t, gene_names)
                trajectory_rows.append(
                    {
                        "scenario": scenario,
                        "perturbation_module": module,
                        "strength_z": float(strength),
                        "time": float(t),
                        "adaptive_metabolic": float(score_t["adaptive_metabolic"].mean()),
                        "source_aligned_antitumor": float(score_t["source_aligned_antitumor"].mean()),
                        "therapeutic_margin": float(score_t["therapeutic_margin"].mean()),
                        "n_trace_cells": 800,
                    }
                )

    programs = pd.DataFrame(program_rows)
    initial_terminal = pd.DataFrame(initial_terminal_rows)
    genes = pd.DataFrame(gene_rows)
    traces = pd.DataFrame(trajectory_rows)

    base_program = programs[programs["scenario"].eq("baseline")].set_index("program")["mean_score"]
    programs["delta_from_baseline"] = programs.apply(
        lambda r: float(r["mean_score"] - base_program.loc[r["program"]]), axis=1
    )
    base_it = initial_terminal[initial_terminal["scenario"].eq("baseline")].set_index("program")
    initial_terminal["initial_delta_from_baseline"] = initial_terminal.apply(
        lambda r: float(r["initial_mean_score"] - base_it.loc[r["program"], "initial_mean_score"]),
        axis=1,
    )
    initial_terminal["terminal_delta_from_baseline"] = initial_terminal.apply(
        lambda r: float(r["terminal_mean_score"] - base_it.loc[r["program"], "terminal_mean_score"]),
        axis=1,
    )
    initial_terminal["dynamic_gain_over_initial"] = (
        initial_terminal["terminal_delta_from_baseline"] - initial_terminal["initial_delta_from_baseline"]
    )
    base_gene = genes[genes["scenario"].eq("baseline")].set_index("gene")["mean_z_readout"]
    genes["delta_z_from_baseline"] = genes.apply(lambda r: float(r["mean_z_readout"] - base_gene.loc[r["gene"]]), axis=1)

    target_rows = []
    for module, meta in PERTURBATION_MODULES.items():
        target = meta["target_program"]
        base = programs[programs["scenario"].eq("baseline") & programs["program"].eq(target)].iloc[0]
        target_rows.append(
            {
                "perturbation_module": module,
                "target_program": target,
                "strength_z": 0.0,
                "target_score": float(base["mean_score"]),
                "delta_from_baseline": 0.0,
                "program_label": READOUT_PROGRAMS[target]["label"],
                "perturbation_label": meta["label"].replace("\n", " "),
                "genes_used": "",
            }
        )
        for strength in (-5.0, 5.0):
            row = programs[
                programs["perturbation_module"].eq(module)
                & programs["strength_z"].eq(strength)
                & programs["program"].eq(target)
            ].iloc[0]
            target_rows.append(
                {
                    "perturbation_module": module,
                    "target_program": target,
                    "strength_z": float(strength),
                    "target_score": float(row["mean_score"]),
                    "delta_from_baseline": float(row["delta_from_baseline"]),
                    "program_label": READOUT_PROGRAMS[target]["label"],
                    "perturbation_label": meta["label"].replace("\n", " "),
                    "genes_used": row["genes_used"],
                }
            )
    target_response = pd.DataFrame(target_rows)

    programs.to_csv(TAB / "pulsebridge_gene_perturbation_program_scores.csv", index=False)
    initial_terminal.to_csv(TAB / "pulsebridge_gene_perturbation_initial_terminal_scores.csv", index=False)
    genes.to_csv(TAB / "pulsebridge_gene_perturbation_gene_readout.csv", index=False)
    target_response.to_csv(TAB / "pulsebridge_gene_perturbation_target_response.csv", index=False)
    traces.to_csv(TAB / "pulsebridge_gene_perturbation_trace.csv", index=False)
    manifest = {
        "analysis": "pulsebridge_gene_perturbation_rollout",
        "contract": "day0 latent + gene-program PCA direction push -> trained PulseBridge ODE rollout -> day3 observed latent kNN gene/program readout",
        "does_not_use": ["hand-defined state bins", "condition-label swap counterfactual", "de novo RNA decoder"],
        "model_state_path": str(MODEL_STATE_PATH.relative_to(REPO)),
        "adata_path": str(ADATA_PATH.relative_to(REPO)),
        "n_day0_cells": int(x0.shape[0]),
        "n_day3_reference_cells": int(day3_idx.shape[0]),
        "ode_steps": 60,
        "knn_readout_k": 30,
        "model_config": record["model_config"],
    }
    (TAB / "pulsebridge_gene_perturbation_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {
        "programs": programs,
        "initial_terminal": initial_terminal,
        "genes": genes,
        "target_response": target_response,
        "traces": traces,
    }


def panel_initial_vs_terminal(ax: plt.Axes, initial_terminal: pd.DataFrame) -> None:
    rows = []
    for module, meta in PERTURBATION_MODULES.items():
        target = meta["target_program"]
        for strength in (-5.0, 5.0):
            row = initial_terminal[
                initial_terminal["perturbation_module"].eq(module)
                & initial_terminal["strength_z"].eq(strength)
                & initial_terminal["program"].eq(target)
            ].iloc[0]
            rows.append(row)
    df = pd.DataFrame(rows)
    y = np.arange(len(df))
    for i, row in enumerate(df.itertuples()):
        meta = PERTURBATION_MODULES[row.perturbation_module]
        x0 = float(row.initial_delta_from_baseline)
        x1 = float(row.terminal_delta_from_baseline)
        ax.plot([x0, x1], [i, i], color=meta["color"], lw=1.6, alpha=0.75)
        ax.scatter(x0, i, s=42, facecolor="white", edgecolor=meta["color"], linewidth=1.4, zorder=4)
        ax.scatter(x1, i, s=48, facecolor=meta["color"], edgecolor="white", linewidth=0.7, zorder=5)
    labels = [
        f"{PERTURBATION_MODULES[r.perturbation_module]['label'].replace(chr(10), ' ')} {r.strength_z:+.0f}z"
        for r in df.itertuples()
    ]
    ax.axvline(0, color="#111827", lw=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6.9)
    ax.invert_yaxis()
    ax.set_xlabel("target-program delta from baseline")
    ax.set_title("Initial push vs terminal effect", loc="left")
    ax.grid(axis="x", color="#E5E7EB", lw=0.7)
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="#111827", markerfacecolor="white", lw=0, label="initial kNN"),
            Line2D([0], [0], marker="o", color="#111827", markerfacecolor="#111827", lw=0, label="terminal rollout"),
        ],
        frameon=False,
        fontsize=6.8,
        loc="upper right",
    )


def panel_target_response(ax: plt.Axes, target_response: pd.DataFrame) -> None:
    for module, meta in PERTURBATION_MODULES.items():
        d = target_response[target_response["perturbation_module"].eq(module)].sort_values("strength_z")
        ax.plot(
            d["strength_z"],
            d["target_score"],
            color=meta["color"],
            marker="o",
            ms=5.2,
            lw=2.0,
            label=meta["label"].replace("\n", " "),
        )
    ax.axvline(0, color="#CBD5E1", lw=0.9)
    ax.set_xticks([-5, 0, 5])
    ax.set_xticklabels(["-5z", "baseline", "+5z"])
    ax.set_ylabel("terminal target program score")
    ax.set_title("Terminal response to gene push", loc="left")
    ax.grid(axis="y", color="#E5E7EB", lw=0.7)
    ax.legend(frameon=False, fontsize=6.8, loc="upper left", bbox_to_anchor=(0.0, -0.15), ncol=2)


def panel_tradeoff(ax: plt.Axes, programs: pd.DataFrame) -> None:
    wide = (
        programs[programs["program"].isin(["source_aligned_antitumor", "adaptive_metabolic", "therapeutic_margin"])]
        .pivot_table(
            index=["scenario", "perturbation_module", "strength_z"],
            columns="program",
            values="delta_from_baseline",
            aggfunc="first",
        )
        .reset_index()
    )
    d = wide[~wide["scenario"].eq("baseline")].copy()
    for module, meta in PERTURBATION_MODULES.items():
        sub = d[d["perturbation_module"].eq(module)].sort_values("strength_z")
        ax.plot(
            sub["adaptive_metabolic"],
            sub["source_aligned_antitumor"],
            color=meta["color"],
            lw=1.5,
            alpha=0.85,
        )
        for _, row in sub.iterrows():
            marker = "<" if row["strength_z"] < 0 else ">"
            ax.scatter(
                row["adaptive_metabolic"],
                row["source_aligned_antitumor"],
                color=meta["color"],
                marker=marker,
                s=56,
                edgecolor="white",
                linewidth=0.7,
                label=meta["label"].replace("\n", " ") if row["strength_z"] < 0 else None,
                zorder=5,
            )
            ax.text(
                row["adaptive_metabolic"] + 0.006,
                row["source_aligned_antitumor"],
                f"{row['strength_z']:+.0f}z",
                fontsize=6.6,
                va="center",
                color="#111827",
            )
    ax.axhline(0, color="#111827", lw=0.8)
    ax.axvline(0, color="#111827", lw=0.8)
    ax.set_xlabel("delta adaptive/metabolic")
    ax.set_ylabel("delta anti-tumor-like")
    ax.set_title("Response-adaptation tradeoff", loc="left")
    ax.grid(color="#E5E7EB", lw=0.7)
    ax.legend(frameon=False, fontsize=6.6, loc="best")


def panel_adaptive_crossprogram(ax: plt.Axes, programs: pd.DataFrame) -> None:
    program_order = [
        "source_aligned_antitumor",
        "reactivation",
        "immune_surface",
        "proliferation_repression",
        "adaptive_metabolic",
        "therapeutic_margin",
    ]
    labels = {
        "source_aligned_antitumor": "anti-tumor-like",
        "reactivation": "DKK1/GAS5",
        "immune_surface": "B2M/HLA/IFITM3",
        "proliferation_repression": "EREG/PCNA down",
        "adaptive_metabolic": "SAT1/LGALS1/MT",
        "therapeutic_margin": "margin",
    }
    keep = programs[
        programs["perturbation_module"].eq("adaptive_metabolic")
        & programs["strength_z"].isin([-5.0, 5.0])
        & programs["program"].isin(program_order)
    ].copy()
    y = np.arange(len(program_order))
    width = 0.34
    for offset, strength, label, color in [
        (-width / 2, -5.0, "-5z adaptive genes", "#8FB6C8"),
        (width / 2, 5.0, "+5z adaptive genes", GOLD),
    ]:
        sub = keep[keep["strength_z"].eq(strength)].set_index("program").reindex(program_order)
        ax.barh(
            y + offset,
            sub["delta_from_baseline"],
            height=width,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            label=label,
    )
    ax.axvline(0, color="#111827", lw=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels([labels[p] for p in program_order])
    ax.invert_yaxis()
    ax.set_xlabel("terminal program delta")
    ax.set_title("Adaptive push propagates", loc="left")
    ax.grid(axis="x", color="#E5E7EB", lw=0.7)
    ax.legend(frameon=False, fontsize=6.8, loc="upper left", bbox_to_anchor=(0.0, -0.15), ncol=2)


def plot(results: dict[str, pd.DataFrame]) -> None:
    setup_style()
    fig = plt.figure(figsize=(14.2, 4.8), dpi=220)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.0, 1.08], wspace=0.52)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    panel_initial_vs_terminal(ax_a, results["initial_terminal"])
    panel_tradeoff(ax_b, results["programs"])
    panel_adaptive_crossprogram(ax_c, results["programs"])
    for ax, letter in [(ax_a, "A"), (ax_b, "B"), (ax_c, "C")]:
        ax.text(-0.13, 1.08, letter, transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")
    fig.savefig(FIG / "story_pulsebridge_gene_perturbation_rollout.png", bbox_inches="tight")
    fig.savefig(FIG / "story_pulsebridge_gene_perturbation_rollout.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    results = compute()
    plot(results)
    print(FIG / "story_pulsebridge_gene_perturbation_rollout.png")
    print(FIG / "story_pulsebridge_gene_perturbation_rollout.pdf")
    print(TAB / "pulsebridge_gene_perturbation_program_scores.csv")
    print(TAB / "pulsebridge_gene_perturbation_initial_terminal_scores.csv")
    print(TAB / "pulsebridge_gene_perturbation_target_response.csv")
    print(TAB / "pulsebridge_gene_perturbation_gene_readout.csv")


if __name__ == "__main__":
    main()
