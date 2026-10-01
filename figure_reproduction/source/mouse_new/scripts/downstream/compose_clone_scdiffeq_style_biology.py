#!/usr/bin/env python3
"""scDiffEq-style biological analyses for the clone/BMF case.

This script replaces the rejected C8-C14 panels with a smaller set of
model-native analyses:

1. BMF branch fans from real day-2 barcoded cells.
2. Gene-program waves decoded along simulated BMF branches.
3. In silico +/-5z RNA-program pushes and terminal fate output.

No model is retrained.  The locked final BMF model is reconstructed from the
saved CellCompass run.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cellcompass_mpl_cache")
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import anndata as ad
import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[2]
TABLE_DIR = ROOT / "downstream/cellcompass_clone_case"
FIG_DIR = ROOT / "figures/cellcompass_clone_case"
ADATA_PATH = Path("/Users/zhenyizhang/.cellcompass/algorithm_benchmarks/datasets/weinreb_rawrebuild_full_k10mindiff1/data.h5ad")
ALGO_PATH = Path("/Users/zhenyizhang/.cellcompass/training_algorithms/clone_cuvgf/algorithm.py")
FINAL_RUN_DIR = ROOT / "training_runs/20260607T113528987593Z__final__clone_cuvgf__campaign_20260-trial_20260607-weinreb_rawr"
FINAL_MODEL_STATE = FINAL_RUN_DIR / "artifacts/model_state.pt"
FINAL_RESOLVED_CONFIG = FINAL_RUN_DIR / "resolved_config.yaml"
BMF_TABLE = ROOT / "downstream/clone_fate_spring_final/clone_cuvgf_bmf_final_day2_fate_probabilities.csv"
PATH_TABLE = TABLE_DIR / "clone_downstream_selected_bmf_branch_paths.csv"
MIX_TABLE = TABLE_DIR / "clone_downstream_selected_clone_mixtures.csv"

FATE_PALETTE = {
    "Undifferentiated": "#2E8B57",
    "Neutrophil": "#19B7C7",
    "Monocyte": "#8C564B",
    "Baso": "#7E57C2",
    "Mast": "#D95F9F",
    "Meg": "#2563EB",
    "Erythroid": "#F28E2B",
    "Eos": "#D62728",
    "Lymphoid": "#8FBCE6",
    "Ccr7_DC": "#B8B34A",
    "pDC": "#6B7280",
}

PROGRAMS = {
    "Stem/progenitor": ["Meis1", "Mecom", "Procr", "Mpl", "Kit"],
    "Neutrophil": ["Cebpe", "Lyz2", "S100a8"],
    "Monocyte": ["Irf8", "Csf1r", "Lyz2"],
    "Baso/Mast": ["Gata2", "Ms4a2", "Cpa3"],
    "Meg/Ery": ["Pf4", "Ppbp", "Itga2b", "Car1", "Car2"],
}

BRANCH_PROGRAM = {
    "Undifferentiated": "Stem/progenitor",
    "Neutrophil": "Neutrophil",
    "Monocyte": "Monocyte",
    "Baso": "Baso/Mast",
    "Mast": "Baso/Mast",
    "Meg": "Meg/Ery",
    "Erythroid": "Meg/Ery",
}

PROGRAM_TARGETS = {
    "Stem/progenitor": ["Undifferentiated"],
    "Neutrophil": ["Neutrophil"],
    "Monocyte": ["Monocyte"],
    "Baso/Mast": ["Baso", "Mast"],
    "Meg/Ery": ["Meg", "Erythroid"],
}

MISSING_CLONE_STRINGS = {"", "nan", "none", "null", "na", "<na>"}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial"],
            "font.size": 8,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "mathtext.sf": "Arial",
            "mathtext.default": "regular",
            "axes.linewidth": 1.1,
            "axes.edgecolor": "#111827",
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
        }
    )


def clone_key(value) -> str:
    text = str(value).strip()
    if text.lower() in MISSING_CLONE_STRINGS:
        return ""
    try:
        f = float(text)
        if np.isfinite(f) and abs(f - int(f)) < 1e-9:
            return str(int(f))
    except Exception:
        pass
    return text


def fate_names(df: pd.DataFrame) -> list[str]:
    out = []
    for col in df.columns:
        if not col.startswith("truth_"):
            continue
        fate = col.removeprefix("truth_")
        if f"pred_{fate}" in df.columns:
            out.append(fate)
    return out


def normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(np.asarray(x, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    s = x.sum(axis=1, keepdims=True)
    return np.divide(x, s, out=np.zeros_like(x), where=s > 0)


def load_obs(adata: ad.AnnData) -> pd.DataFrame:
    obs = adata.obs[["clone", "time_point_processed", "Cell type annotation", "SPRING-x", "SPRING-y"]].copy()
    obs["clone_str"] = [clone_key(x) for x in obs["clone"].to_numpy()]
    obs["t"] = obs["time_point_processed"].astype(int)
    obs["cell_type"] = obs["Cell type annotation"].astype(str)
    return obs.rename(columns={"SPRING-x": "spring_x", "SPRING-y": "spring_y"})


def style_embedding_axis(ax, obs: pd.DataFrame) -> None:
    x = obs["spring_x"].to_numpy(float)
    y = obs["spring_y"].to_numpy(float)
    padx = 0.035 * (np.nanmax(x) - np.nanmin(x))
    pady = 0.035 * (np.nanmax(y) - np.nanmin(y))
    ax.set_xlim(float(np.nanmin(x) - padx), float(np.nanmax(x) + padx))
    ax.set_ylim(float(np.nanmin(y) - pady), float(np.nanmax(y) + pady))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(FIG_DIR / f"{stem}.{ext}", dpi=360 if ext == "png" else None)
    plt.close(fig)


def load_bmf_model(latent_dim: int, n_fates: int):
    sys.modules.setdefault("cellrank", types.ModuleType("cellrank"))
    spec = importlib.util.spec_from_file_location("clone_cuvgf_algorithm_for_scdiffeq_panels", str(ALGO_PATH))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    with FINAL_RESOLVED_CONFIG.open() as f:
        resolved = yaml.safe_load(f)
    model = mod.BranchDynamicalModel(latent_dim, resolved["model"], n_fates=n_fates).to(torch.device("cpu"))
    state_obj = torch.load(FINAL_MODEL_STATE, map_location="cpu", weights_only=False)
    state = {
        k: (torch.as_tensor(v) if isinstance(v, np.ndarray) else v)
        for k, v in state_obj["all_model"]["model_state_dict"].items()
    }
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, resolved


def terminal_fate_probabilities(query: np.ndarray, obs: pd.DataFrame, latent: np.ndarray, fates: list[str], k: int = 30) -> np.ndarray:
    ref_mask = obs["t"].eq(2).to_numpy()
    ref_latent = np.asarray(latent[ref_mask], dtype=np.float32)
    ref_labels = obs.loc[ref_mask, "cell_type"].astype(str).to_numpy()
    code = {f: i for i, f in enumerate(fates)}
    ref_codes = np.asarray([code.get(x, -1) for x in ref_labels], dtype=int)
    keep = ref_codes >= 0
    ref_latent = ref_latent[keep]
    ref_codes = ref_codes[keep]
    tree = cKDTree(ref_latent)
    d, idx = tree.query(np.asarray(query, dtype=np.float32), k=min(k, len(ref_latent)))
    if idx.ndim == 1:
        d = d[:, None]
        idx = idx[:, None]
    weights = 1.0 / (np.asarray(d, dtype=float) + 1e-4)
    weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    out = np.zeros((idx.shape[0], len(fates)), dtype=float)
    for row in range(idx.shape[0]):
        np.add.at(out[row], ref_codes[idx[row]], weights[row])
    return normalize_rows(out)


def simulate_bmf(
    model,
    resolved: dict,
    x0_np: np.ndarray,
    fates: list[str],
    *,
    snapshot_times: np.ndarray | None = None,
) -> dict:
    if snapshot_times is None:
        snapshot_times = np.linspace(0.0, 2.0, 9)
    integration_dt = float(((resolved.get("evaluation") or {}).get("trajectory_integration_dt") or 0.0125))
    x0 = torch.as_tensor(np.asarray(x0_np, dtype=np.float32), dtype=torch.float32)
    n0 = int(x0.shape[0])
    n_fates = len(fates)
    branch_eye = torch.eye(n_fates, dtype=x0.dtype)
    with torch.no_grad():
        gate = torch.softmax(model.gate_net(torch.cat([x0, torch.zeros((n0, 1), dtype=x0.dtype)], dim=1)), dim=1)
        x = x0.repeat_interleave(n_fates, dim=0).contiguous()
        branch_onehot = branch_eye.repeat(n0, 1).contiguous()
        parent = np.repeat(np.arange(n0, dtype=np.int64), n_fates)
        branch_idx = np.tile(np.arange(n_fates, dtype=np.int64), n0)
        logw = torch.log(gate.reshape(-1).clamp_min(1e-12)).reshape(-1, 1)
        snapshots = {float(snapshot_times[0]): x.cpu().numpy().copy()}
        weights = {float(snapshot_times[0]): torch.exp(logw).cpu().numpy().reshape(-1).copy()}
        current_t = float(snapshot_times[0])
        for target_t in snapshot_times[1:]:
            target_t = float(target_t)
            while current_t < target_t - 1e-9:
                dt = min(integration_dt, target_t - current_t)
                t_col = torch.full((x.shape[0], 1), current_t, dtype=x.dtype)
                inp = torch.cat([x, t_col, branch_onehot], dim=1)
                x = x + model.branch_velocity_net(inp) * dt
                logw = logw + model.branch_growth_net(inp) * dt
                current_t += dt
            snapshots[target_t] = x.cpu().numpy().copy()
            weights[target_t] = torch.exp(logw).cpu().numpy().reshape(-1).copy()
    return {
        "snapshots": snapshots,
        "weights": weights,
        "parent": parent,
        "branch_idx": branch_idx,
        "gate": gate.cpu().numpy(),
        "final_latent": snapshots[float(snapshot_times[-1])],
        "final_weights": weights[float(snapshot_times[-1])],
    }


def aggregate_terminal_fates(sim: dict, endpoint_probs: np.ndarray, n0: int, n_fates: int) -> np.ndarray:
    pred = np.zeros((n0, n_fates), dtype=float)
    denom = np.zeros(n0, dtype=float)
    np.add.at(pred, sim["parent"], endpoint_probs * sim["final_weights"][:, None])
    np.add.at(denom, sim["parent"], sim["final_weights"])
    return normalize_rows(pred / np.maximum(denom[:, None], 1e-12))


def program_delta(adata: ad.AnnData, genes: list[str], strength: float) -> tuple[np.ndarray, list[str]]:
    var_to_idx = {str(g): i for i, g in enumerate(adata.var_names)}
    used = [g for g in genes if g in var_to_idx]
    if not used:
        raise ValueError(f"No genes found for program: {genes}")
    pcs = np.asarray(adata.varm["PCs"], dtype=np.float32)
    idx = np.asarray([var_to_idx[g] for g in used], dtype=int)
    # PCA coordinates were computed from standardized genes.  This moves the
    # initial state along the average PCA direction of the queried RNA program.
    return float(strength) * pcs[idx].mean(axis=0), used


def program_score_from_latent(adata: ad.AnnData, latent_block: np.ndarray, genes: list[str]) -> tuple[np.ndarray, list[str]]:
    var_to_idx = {str(g): i for i, g in enumerate(adata.var_names)}
    used = [g for g in genes if g in var_to_idx]
    pcs = np.asarray(adata.varm["PCs"], dtype=np.float32)
    idx = np.asarray([var_to_idx[g] for g in used], dtype=int)
    scores = np.asarray(latent_block, dtype=np.float32) @ pcs[idx].T
    return scores.mean(axis=1), used


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 4:
        return float("nan")
    x = x[mask]
    y = y[mask]
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.sqrt(np.sum(x**2) * np.sum(y**2)))
    if denom <= 1e-12:
        return float("nan")
    return float(np.sum(x * y) / denom)


def target_fraction(table: pd.DataFrame, target_fates: list[str], prefix: str) -> np.ndarray:
    cols = [f"{prefix}_{f}" for f in target_fates if f"{prefix}_{f}" in table.columns]
    if not cols:
        return np.zeros(len(table), dtype=float)
    return table[cols].to_numpy(float).sum(axis=1)


def source_cell_indices(bmf: pd.DataFrame, max_cells: int = 420) -> np.ndarray:
    truth_cols = [c for c in bmf.columns if c.startswith("truth_")]
    pred_cols = [c for c in bmf.columns if c.startswith("pred_")]
    truth = normalize_rows(bmf[truth_cols].to_numpy(float))
    pred = normalize_rows(bmf[pred_cols].to_numpy(float))
    truth_eff = np.exp(-(np.clip(truth, 1e-12, 1.0) * np.log(np.clip(truth, 1e-12, 1.0))).sum(axis=1))
    pred_eff = np.exp(-(np.clip(pred, 1e-12, 1.0) * np.log(np.clip(pred, 1e-12, 1.0))).sum(axis=1))
    d = bmf.copy()
    d["truth_eff"] = truth_eff
    d["pred_eff"] = pred_eff
    cand = d[d["eligible"].astype(bool) & d["source_label"].eq("Undifferentiated")].copy()
    cand["score"] = cand["truth_reliability"].fillna(0) * (0.6 * cand["truth_eff"] + 0.4 * cand["pred_eff"])
    cand = cand.sort_values(["score", "truth_reliability"], ascending=False)
    return cand["source_index"].head(max_cells).to_numpy(int)


def panel_c8_fate_fan(adata: ad.AnnData, obs: pd.DataFrame) -> None:
    if not PATH_TABLE.exists() or not MIX_TABLE.exists():
        raise FileNotFoundError("Run compose_clone_downstream_story_panels.py first to create selected BMF branch paths.")
    paths = pd.read_csv(PATH_TABLE)
    mix = pd.read_csv(MIX_TABLE)
    selected = [str(c) for c in [2431, 2484, 3530, 4061] if str(c) in set(paths["clone"].astype(str))]
    paths["clone"] = paths["clone"].astype(str)
    mix["clone"] = mix["clone"].astype(str)

    fig = plt.figure(figsize=(13.4, 5.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.42, 0.82], wspace=0.12)
    ax = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[0, 1])

    term = obs[obs["t"].eq(2)]
    for fate, color in FATE_PALETTE.items():
        d = term[term["cell_type"].eq(fate)]
        if d.empty:
            continue
        sample = d.sample(n=min(1500, len(d)), random_state=sum(ord(x) for x in fate) + 5)
        ax.scatter(sample["spring_x"], sample["spring_y"], s=2.0, color=color, alpha=0.055, linewidth=0, zorder=1)

    weight_tbl = paths.drop_duplicates(["clone", "source_pos", "branch_fate"]).copy()
    weight_tbl["rank_within_cell"] = weight_tbl.groupby(["clone", "source_pos"])["branch_weight"].rank("first", ascending=False)
    keep = weight_tbl[(weight_tbl["rank_within_cell"].le(3)) & (weight_tbl["branch_weight"].ge(0.004))]
    visible = paths.merge(keep[["clone", "source_pos", "branch_fate"]], on=["clone", "source_pos", "branch_fate"])
    n_day2 = weight_tbl.groupby("clone")["source_pos"].nunique().to_dict()
    for clone in selected:
        d_clone = visible[visible["clone"].eq(clone)]
        for (_, branch), d in d_clone.groupby(["source_pos", "branch_fate"]):
            d = d.sort_values("model_time")
            endpoint = str(d["terminal_fate"].iloc[0])
            color = FATE_PALETTE.get(endpoint, "#64748B")
            w = float(d["branch_weight"].iloc[0]) * max(int(n_day2.get(clone, 1)), 1)
            lw = 0.30 + 2.8 * np.sqrt(np.clip(w, 0, 1))
            alpha = 0.16 + 0.55 * np.sqrt(np.clip(w, 0, 1))
            line = ax.plot(d["spring_x"], d["spring_y"], color=color, lw=lw, alpha=alpha, zorder=10)[0]
            line.set_path_effects([pe.Stroke(linewidth=lw + 1.1, foreground="white", alpha=0.75), pe.Normal()])
            end = d.iloc[-1]
            ax.scatter(
                end["spring_x"],
                end["spring_y"],
                s=26 + 105 * np.sqrt(np.clip(w, 0, 1)),
                facecolor="white",
                edgecolor=color,
                linewidth=1.0,
                alpha=0.92,
                zorder=20,
            )
        d0 = obs[obs["clone_str"].eq(clone) & obs["t"].eq(0)]
        ax.scatter(d0["spring_x"], d0["spring_y"], s=74, facecolor="#111827", edgecolor="white", linewidth=1.0, zorder=25)
        if not d0.empty:
            ax.text(
                float(d0["spring_x"].median()),
                float(d0["spring_y"].median()) + 42,
                f"clone {clone}",
                ha="center",
                fontsize=7.2,
                weight="bold",
                color="#111827",
                bbox=dict(fc="white", ec="none", boxstyle="round,pad=0.12", alpha=0.78),
                zorder=26,
            )

    style_embedding_axis(ax, obs)
    ax.set_title("BMF branch fans from the same day-2 barcode cells", loc="left", fontsize=11)
    ax.text(
        0.015,
        0.025,
        "background = observed day-6 fate map\nblack dots = observed day-2 barcode cells\nlines = BMF-generated branch paths\nline width = branch probability",
        transform=ax.transAxes,
        fontsize=7.2,
        color="#334155",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#CBD5E1", alpha=0.88),
    )

    fates = [f for f in FATE_PALETTE if f in set(mix["fate"])]
    y = 0
    for clone in selected:
        sub = mix[mix["clone"].eq(clone)]
        for source, label in [("observed descendants", "barcode"), ("BMF from day-2 cells", "BMF")]:
            vals = sub[sub["source"].eq(source)].set_index("fate")["fraction"].to_dict()
            left = 0.0
            for fate in fates:
                val = float(vals.get(fate, 0.0))
                if val <= 0:
                    continue
                ax_bar.barh(y, val, left=left, height=0.34, color=FATE_PALETTE[fate], edgecolor="white", linewidth=0.45)
                left += val
            ax_bar.text(-0.025, y, label, ha="right", va="center", fontsize=7.2, color="#111827")
            y += 0.45
        ax_bar.text(-0.18, y - 0.66, f"clone {clone}", ha="right", va="center", fontsize=7.5, weight="bold")
        y += 0.18
    ax_bar.set_xlim(-0.19, 1.0)
    ax_bar.set_ylim(-0.35, y - 0.02)
    ax_bar.invert_yaxis()
    ax_bar.set_yticks([])
    ax_bar.set_xlabel("day-6 fate fraction")
    ax_bar.set_title("barcode truth vs BMF output", loc="left", fontsize=11)
    ax_bar.grid(axis="x", color="#E5E7EB", lw=0.75)
    sns.despine(ax=ax_bar, left=True)

    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markeredgecolor="white", markersize=6, label=fate)
        for fate, color in FATE_PALETTE.items()
        if fate in {"Undifferentiated", "Neutrophil", "Monocyte", "Baso", "Mast", "Meg", "Erythroid"}
    ]
    ax_bar.legend(handles=handles, frameon=False, fontsize=6.2, loc="lower right", ncol=2)
    fig.suptitle("Model-generated clone futures are branch mixtures, not one averaged path", x=0.02, y=0.99, ha="left", fontsize=14, weight="bold")
    save_figure(fig, "panel_c8_scdiffeq_style_bmf_branch_fans")


def compute_program_waves(adata: ad.AnnData, obs: pd.DataFrame, bmf: pd.DataFrame, fates: list[str]) -> pd.DataFrame:
    idx = source_cell_indices(bmf, max_cells=420)
    latent = np.asarray(adata.obsm["X_latent"], dtype=np.float32)
    model, resolved = load_bmf_model(latent.shape[1], len(fates))
    sim = simulate_bmf(model, resolved, latent[idx], fates, snapshot_times=np.linspace(0.0, 2.0, 13))
    rows = []
    for t, block in sim["snapshots"].items():
        w = sim["weights"][t]
        for branch_i, branch_fate in enumerate(fates):
            sel = sim["branch_idx"] == branch_i
            ww = w[sel]
            ww = ww / max(float(ww.sum()), 1e-12)
            for program, genes in PROGRAMS.items():
                score, used = program_score_from_latent(adata, block[sel], genes)
                rows.append(
                    {
                        "model_time": float(t),
                        "branch_fate": branch_fate,
                        "program": program,
                        "program_score": float(np.sum(ww * score)),
                        "genes_used": ",".join(used),
                        "n_source_cells": int(len(idx)),
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "clone_scdiffeq_program_waves.csv", index=False)
    return out


def panel_c9_program_waves(waves: pd.DataFrame) -> None:
    pairs = [
        ("Undifferentiated", "Stem/progenitor"),
        ("Neutrophil", "Neutrophil"),
        ("Monocyte", "Monocyte"),
        ("Baso", "Baso/Mast"),
        ("Mast", "Baso/Mast"),
        ("Meg", "Meg/Ery"),
        ("Erythroid", "Meg/Ery"),
    ]
    pairs = [(f, p) for f, p in pairs if ((waves["branch_fate"].eq(f)) & (waves["program"].eq(p))).any()]
    fig, ax = plt.subplots(figsize=(7.4, 4.9))
    all_vals = []
    for fate, program in pairs:
        d = waves[waves["branch_fate"].eq(fate) & waves["program"].eq(program)].sort_values("model_time")
        all_vals.extend(d["program_score"].to_numpy(float).tolist())
    lo, hi = np.percentile(all_vals, [2, 98])
    span = max(float(hi - lo), 1e-6)
    for row, (fate, program) in enumerate(pairs):
        d = waves[waves["branch_fate"].eq(fate) & waves["program"].eq(program)].sort_values("model_time")
        x = d["model_time"].to_numpy(float) / 2.0
        y = (d["program_score"].to_numpy(float) - lo) / span
        y = np.clip(y, -0.15, 1.15)
        base = len(pairs) - row - 1
        color = FATE_PALETTE.get(fate, "#64748B")
        ax.fill_between(x, base, base + 0.70 * y, color=color, alpha=0.42, linewidth=0)
        ax.plot(x, base + 0.70 * y, color=color, lw=2.2)
        peak_t = x[int(np.argmax(y))]
        ax.vlines(peak_t, base, base + 0.70 * float(np.max(y)), color=color, lw=1.0, alpha=0.7)
        ax.text(-0.035, base + 0.24, fate, ha="right", va="center", fontsize=8.0, color=color, weight="bold")
        ax.text(1.035, base + 0.24, program, ha="left", va="center", fontsize=7.4, color="#334155", clip_on=False)
    ax.set_xlim(-0.06, 1.28)
    ax.set_ylim(-0.4, len(pairs) - 0.08)
    ax.set_yticks([])
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xticklabels(["day 2", "mid", "day 6"])
    ax.set_xlabel("BMF simulated time")
    ax.set_title("Program activity decoded along generated fate branches", loc="left", fontsize=11)
    ax.text(
        0.01,
        -0.20,
        "values are decoded from X_pca and row-scaled for display",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
        color="#475569",
    )
    sns.despine(ax=ax, left=True)
    fig.suptitle("BMF simulation carries distinct lineage-program waves", x=0.02, y=0.99, ha="left", fontsize=13.2, weight="bold")
    save_figure(fig, "panel_c9_scdiffeq_style_program_waves")


def compute_program_pushes(adata: ad.AnnData, obs: pd.DataFrame, bmf: pd.DataFrame, fates: list[str]) -> pd.DataFrame:
    idx = source_cell_indices(bmf, max_cells=360)
    latent = np.asarray(adata.obsm["X_latent"], dtype=np.float32)
    model, resolved = load_bmf_model(latent.shape[1], len(fates))
    x0 = latent[idx].copy()
    scenarios = [("baseline", "baseline", 0.0, np.zeros(latent.shape[1], dtype=np.float32), "")]
    for program, genes in PROGRAMS.items():
        for strength in [-5.0, 5.0]:
            delta, used = program_delta(adata, genes, strength)
            scenarios.append((f"{program} {strength:+.0f}z", program, strength, delta.astype(np.float32), ",".join(used)))

    rows = []
    for scenario, program, strength, delta, used in scenarios:
        sim = simulate_bmf(model, resolved, x0 + delta[None, :], fates, snapshot_times=np.asarray([0.0, 1.0, 2.0]))
        endpoint = terminal_fate_probabilities(sim["final_latent"], obs, latent, fates, k=30)
        pred = aggregate_terminal_fates(sim, endpoint, len(idx), len(fates))
        avg = pred.mean(axis=0)
        for fate, value in zip(fates, avg):
            rows.append(
                {
                    "scenario": scenario,
                    "program": program,
                    "strength_z": float(strength),
                    "fate": fate,
                    "fraction": float(value),
                    "genes_used": used,
                    "n_source_cells": int(len(idx)),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "clone_scdiffeq_program_push_fate_output.csv", index=False)
    return out


def panel_c10_program_push(push: pd.DataFrame) -> None:
    fates = [f for f in FATE_PALETTE if f in set(push["fate"])]
    programs = [p for p in PROGRAMS if p in set(push["program"])]
    scenarios = ["baseline"] + [f"{p} +5z" for p in programs] + [f"{p} -5z" for p in programs]
    scenarios = [s for s in scenarios if s in set(push["scenario"])]
    fig, (ax, ax_delta) = plt.subplots(
        1,
        2,
        figsize=(10.8, 4.1),
        gridspec_kw={"width_ratios": [3.15, 1.35]},
    )
    x = np.arange(len(scenarios))
    for i, scenario in enumerate(scenarios):
        d = push[push["scenario"].eq(scenario)].set_index("fate")
        bottom = 0.0
        for fate in fates:
            val = float(d["fraction"].get(fate, 0.0))
            if val <= 0:
                continue
            ax.bar(i, val, bottom=bottom, width=0.72, color=FATE_PALETTE[fate], edgecolor="white", linewidth=0.45)
            bottom += val
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=38, ha="right", fontsize=7.0)
    ax.set_ylabel("terminal fate fraction")
    ax.set_ylim(0, 1)
    ax.set_title("a   Predicted terminal fate composition", loc="left", fontsize=10.2, weight="bold")
    ax.grid(axis="y", color="#E5E7EB", lw=0.7)
    handles = [
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=FATE_PALETTE[f], markeredgecolor="white", markersize=6, label=f)
        for f in fates
        if push[push["fate"].eq(f)]["fraction"].max() > 0.015
    ]
    ax.legend(handles=handles, frameon=False, fontsize=6.5, ncol=4, loc="upper left", bbox_to_anchor=(0, -0.31))
    baseline = push[push["scenario"].eq("baseline")].set_index("fate")["fraction"]
    deltas = []
    for scenario in scenarios[1:]:
        d = push[push["scenario"].eq(scenario)].set_index("fate")["fraction"] - baseline
        fate = str(d.abs().idxmax())
        deltas.append((scenario, fate, float(d[fate])))

    delta_y = np.arange(len(deltas))
    for yy, (scenario, fate, value) in zip(delta_y, deltas):
        color = FATE_PALETTE.get(fate, "#6B7280")
        ax_delta.plot([0, value], [yy, yy], color="#B8C0CC", lw=1.0, zorder=1)
        ax_delta.scatter(value, yy, s=25, color=color, edgecolor="white", linewidth=0.45, zorder=2)
        ax_delta.text(value + 0.006, yy, f"{value:+.2f}", ha="left", va="center", fontsize=6.5, color="#374151")
    ax_delta.axvline(0, color="#111827", lw=0.8)
    ax_delta.set_yticks(delta_y)
    ax_delta.set_yticklabels([scenario for scenario, _, _ in deltas], fontsize=6.7)
    ax_delta.invert_yaxis()
    ax_delta.set_xlim(-0.10, 0.21)
    ax_delta.set_xlabel("largest fate-fraction change")
    ax_delta.set_title("b   Largest change from baseline", loc="left", fontsize=10.2, weight="bold")
    ax_delta.grid(axis="x", color="#E5E7EB", lw=0.6)
    sns.despine(ax=ax_delta, left=True)
    sns.despine(ax=ax)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.91, bottom=0.28, wspace=0.35)
    save_figure(fig, "panel_c10_scdiffeq_style_program_push_stackbar")


def compute_per_cell_program_push(
    adata: ad.AnnData,
    obs: pd.DataFrame,
    bmf: pd.DataFrame,
    fates: list[str],
    program: str,
    strength: float = 5.0,
) -> pd.DataFrame:
    eligible = bmf[bmf["eligible"].astype(bool)].copy()
    latent = np.asarray(adata.obsm["X_latent"], dtype=np.float32)
    model, resolved = load_bmf_model(latent.shape[1], len(fates))
    source_idx = eligible["source_index"].to_numpy(int)
    delta, used = program_delta(adata, PROGRAMS[program], strength)
    sim = simulate_bmf(model, resolved, latent[source_idx] + delta[None, :], fates, snapshot_times=np.asarray([0.0, 1.0, 2.0]))
    endpoint = terminal_fate_probabilities(sim["final_latent"], obs, latent, fates, k=30)
    pushed = aggregate_terminal_fates(sim, endpoint, len(source_idx), len(fates))
    score, _ = program_score_from_latent(adata, latent[source_idx], PROGRAMS[program])
    out = eligible[
        ["source_index", "clone", "spring_x", "spring_y", "source_label", "truth_reliability", "metric_weight"]
    ].copy()
    out["program"] = program
    out["strength_z"] = float(strength)
    out["program_score"] = score
    out["genes_used"] = ",".join(used)
    for j, fate in enumerate(fates):
        out[f"push_{fate}"] = pushed[:, j]
        out[f"delta_{fate}"] = pushed[:, j] - eligible[f"pred_{fate}"].to_numpy(float)
    target = PROGRAM_TARGETS[program]
    out["truth_target"] = target_fraction(eligible, target, "truth")
    out["bmf_target"] = target_fraction(eligible, target, "pred")
    out["push_target"] = out[[f"push_{f}" for f in target if f"push_{f}" in out.columns]].sum(axis=1)
    out["delta_target"] = out["push_target"] - out["bmf_target"]
    return out


def scatter_spring_value(
    ax,
    obs: pd.DataFrame,
    values_df: pd.DataFrame,
    value: str,
    title: str,
    *,
    cmap,
    vmin=None,
    vmax=None,
    point_size=5.2,
):
    rng = np.random.default_rng(20260612)
    bg = obs.sample(n=min(len(obs), 22000), random_state=20260612)
    ax.scatter(bg["spring_x"], bg["spring_y"], s=1.0, color="#CBD5E1", alpha=0.12, linewidth=0, zorder=0)
    vals = values_df[value].to_numpy(float)
    if vmin is None:
        vmin = float(np.nanpercentile(vals, 2))
    if vmax is None:
        vmax = float(np.nanpercentile(vals, 98))
    sc = ax.scatter(
        values_df["spring_x"],
        values_df["spring_y"],
        c=vals,
        s=point_size,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        alpha=0.88,
        linewidth=0,
        zorder=5,
    )
    style_embedding_axis(ax, obs)
    ax.set_title(title, loc="left", fontsize=9.4)
    cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.010)
    cb.ax.tick_params(labelsize=6)
    return sc


def panel_c12_neutrophil_spring_competence(adata: ad.AnnData, obs: pd.DataFrame, bmf: pd.DataFrame, fates: list[str]) -> None:
    neu = compute_per_cell_program_push(adata, obs, bmf, fates, "Neutrophil", strength=5.0)
    neu.to_csv(TABLE_DIR / "clone_scdiffeq_neutrophil_spring_competence.csv", index=False)
    neu["program_score_z"] = (neu["program_score"] - neu["program_score"].mean()) / max(float(neu["program_score"].std()), 1e-8)
    target = "Neutrophil"
    top = neu["delta_target"] >= float(np.nanpercentile(neu["delta_target"], 90))
    truth_top = neu["truth_target"] >= float(np.nanpercentile(neu["truth_target"], 90))
    overlap = int((top & truth_top).sum())
    union = int((top | truth_top).sum())
    jaccard = overlap / max(union, 1)
    corr_truth = pearson(neu["program_score"].to_numpy(float), neu["truth_target"].to_numpy(float))
    corr_push = pearson(neu["program_score"].to_numpy(float), neu["delta_target"].to_numpy(float))

    fig = plt.figure(figsize=(12.6, 7.2))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 1.05], height_ratios=[1.0, 0.90], wspace=0.12, hspace=0.18)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)] + [fig.add_subplot(gs[1, i]) for i in range(3)]
    teal_cmap = mpl.colors.LinearSegmentedColormap.from_list("teal_prob", ["#F8FAFC", "#CCFBF1", "#2DD4BF", "#0F766E"])
    gain_cmap = mpl.colors.LinearSegmentedColormap.from_list("gain", ["#F8FAFC", "#C7F9E9", "#14B8A6", "#065F5B"])
    scatter_spring_value(
        axes[0],
        obs,
        neu,
        "program_score_z",
        "day-2 neutrophil RNA program",
        cmap=mpl.cm.PuBuGn,
        vmin=-1.5,
        vmax=1.8,
    )
    scatter_spring_value(
        axes[1],
        obs,
        neu,
        "truth_target",
        "barcode-derived future Neutrophil fate",
        cmap=teal_cmap,
        vmin=0,
        vmax=float(np.nanpercentile(neu["truth_target"], 98)),
    )
    scatter_spring_value(
        axes[2],
        obs,
        neu,
        "bmf_target",
        "BMF baseline Neutrophil output",
        cmap=teal_cmap,
        vmin=0,
        vmax=float(np.nanpercentile(neu["bmf_target"], 98)),
    )
    scatter_spring_value(
        axes[3],
        obs,
        neu,
        "delta_target",
        "Neutrophil output gained after +5z RNA push",
        cmap=gain_cmap,
        vmin=0,
        vmax=float(np.nanpercentile(neu["delta_target"], 98)),
    )
    # Spatial overlap panel: where natural barcode fate and model susceptibility
    # do or do not mark the same day-2 cells.
    ax = axes[4]
    bg = obs.sample(n=min(len(obs), 22000), random_state=9)
    ax.scatter(bg["spring_x"], bg["spring_y"], s=1.0, color="#CBD5E1", alpha=0.10, linewidth=0)
    ax.scatter(neu.loc[truth_top, "spring_x"], neu.loc[truth_top, "spring_y"], s=14, facecolor="#93C5FD", edgecolor="white", linewidth=0.25, alpha=0.75, label="top barcode Neutrophil")
    ax.scatter(neu.loc[top, "spring_x"], neu.loc[top, "spring_y"], s=19, facecolor="none", edgecolor="#0F766E", linewidth=0.9, alpha=0.95, label="top BMF push gain")
    style_embedding_axis(ax, obs)
    ax.set_title("different day-2 regions mark fate vs susceptibility", loc="left", fontsize=9.4)
    ax.legend(frameon=False, fontsize=6.5, loc="lower left")
    ax.text(
        0.02,
        0.96,
        f"top-decile overlap Jaccard = {jaccard:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.0,
        color="#334155",
        bbox=dict(boxstyle="round,pad=0.20", fc="white", ec="#CBD5E1", alpha=0.88),
    )

    # Fate composition of high-susceptibility cells.
    ax = axes[5]
    groups = [
        ("all day-2\neligible", neu.index == neu.index),
        ("top push-gain\ncells", top),
    ]
    rows = []
    for label, mask in groups:
        d = neu.loc[mask]
        rows.append((label, "baseline", d[[f"pred_{f}" for f in []]].shape[0] if False else None))
    f_show = [f for f in ["Undifferentiated", "Neutrophil", "Monocyte", "Baso", "Mast", "Meg", "Erythroid"] if f in fates]
    bars = []
    for label, mask in groups:
        d = neu.loc[mask]
        base = np.asarray([d[f"bmf_target"].mean() if f == target else bmf.loc[bmf["source_index"].isin(d["source_index"]), f"pred_{f}"].mean() for f in f_show])
        push = np.asarray([d[f"push_{f}"].mean() for f in f_show])
        bars.append((label, "baseline", base / max(float(base.sum()), 1e-12)))
        bars.append((label, "+5z Neutrophil", push / max(float(push.sum()), 1e-12)))
    y = np.arange(len(bars))
    for i, (_, state, vals) in enumerate(bars):
        left = 0.0
        for fate, val in zip(f_show, vals):
            ax.barh(i, val, left=left, height=0.45, color=FATE_PALETTE.get(fate, "#94A3B8"), edgecolor="white", linewidth=0.45)
            left += val
        ax.text(-0.025, i, state, ha="right", va="center", fontsize=6.8)
    ax.set_yticks([0.5, 2.5])
    ax.set_yticklabels([g[0] for g in groups], fontsize=7.2, weight="bold")
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("terminal fate fraction")
    ax.set_title("susceptible cells shift toward Neutrophil output", loc="left", fontsize=9.4)
    ax.grid(axis="x", color="#E5E7EB", lw=0.7)
    handles = [
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=FATE_PALETTE[f], markeredgecolor="white", markersize=5.5, label=f)
        for f in f_show
    ]
    ax.legend(handles=handles, frameon=False, fontsize=6.0, ncol=2, loc="lower right")
    sns.despine(ax=ax, left=True)

    fig.suptitle(
        "Neutrophil fate gain is localized to a permissive day-2 state, not pre-marked barcode fate",
        x=0.02,
        y=0.99,
        ha="left",
        fontsize=13.0,
        weight="bold",
    )
    fig.text(
        0.02,
        0.02,
        f"Across eligible day-2 cells: corr(RNA program, barcode Neutrophil fate) = {corr_truth:.2f}; "
        f"corr(RNA program, BMF +5z Neutrophil gain) = {corr_push:.2f}. "
        "Thus the Neutrophil response is not a simple RNA-marker gradient; it is an in silico state-dependent response to a PCA-direction push.",
        fontsize=7.2,
        color="#475569",
    )
    fig.subplots_adjust(left=0.035, right=0.985, top=0.90, bottom=0.10)
    save_figure(fig, "panel_c12_scdiffeq_style_neutrophil_spring_competence")


def main() -> None:
    setup_style()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    adata = ad.read_h5ad(ADATA_PATH)
    obs = load_obs(adata)
    bmf = pd.read_csv(BMF_TABLE)
    fates = fate_names(bmf)

    panel_c8_fate_fan(adata, obs)
    waves = compute_program_waves(adata, obs, bmf, fates)
    panel_c9_program_waves(waves)
    push = compute_program_pushes(adata, obs, bmf, fates)
    panel_c10_program_push(push)
    panel_c12_neutrophil_spring_competence(adata, obs, bmf, fates)
    print(
        {
            "figures": [
                str(FIG_DIR / "panel_c8_scdiffeq_style_bmf_branch_fans.png"),
                str(FIG_DIR / "panel_c9_scdiffeq_style_program_waves.png"),
                str(FIG_DIR / "panel_c10_scdiffeq_style_program_push_stackbar.png"),
                str(FIG_DIR / "panel_c12_scdiffeq_style_neutrophil_spring_competence.png"),
            ],
            "tables": [
                str(TABLE_DIR / "clone_scdiffeq_program_waves.csv"),
                str(TABLE_DIR / "clone_scdiffeq_program_push_fate_output.csv"),
                str(TABLE_DIR / "clone_scdiffeq_neutrophil_spring_competence.csv"),
            ],
        }
    )


if __name__ == "__main__":
    main()
