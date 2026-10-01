#!/usr/bin/env python3
"""Fig06J: RNA-program push changes terminal protein phenotype composition.

This is a model-query panel, not an observed perturbation experiment.  The
analysis starts from the locked t0 RNA seed states, shifts each seed along a
gene-program latent direction, rolls the RNA state through the CITE-PCFM RNA
ODE, reads out terminal protein PCA coordinates with h_eta(r, t), and then
summarizes terminal protein phenotype probabilities with the shared MLP
protein phenotype readout.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/cytobridge_mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/cytobridge_xdg_cache")
os.environ.setdefault("NUMBA_CACHE_DIR", "/private/tmp/numba_cache")
os.environ.setdefault("NUMBA_DISABLE_CACHE", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")
for key in ["MPLCONFIGDIR", "XDG_CACHE_HOME", "NUMBA_CACHE_DIR"]:
    Path(os.environ[key]).mkdir(parents=True, exist_ok=True)

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import Patch

from fig06_rna_protein_regulatory_map import (
    DATA_PATH,
    FINITE_DIFF_EPS,
    OUT_DIR,
    PHENO_ORDER,
    PROTEIN_DIM,
    RANDOM_STATE,
    RNA_DIM,
    RNA_PROGRAMS,
    TRAINED_MODEL_H5AD,
    build_program_directions,
    ensure_import_paths,
    load_readout_input,
    projection_audit_and_scales,
)
from multistageot_style import PHENO_COLORS, clean_metric_axis, setup
from protein_phenotype_readout import train_mlp_protein_phenotype_readout


ROOT = Path("/Users/zhenyizhang/Documents/GitHub/CytoBridge-agent-runtime-v2-20260309/multi-omics")
REPO_ROOT = ROOT.parent
CYTOBRIDGE_SRC = REPO_ROOT / "CytoBridge-main"
TRAJECTORY_PATH = ROOT / "training_runs/20260526T174204321821Z__final__cite_paired_coord_fm__campaign_20260-trial_20260526-gse305370_ci/artifacts/evaluation_trajectory.npz"
FIG_DIR = ROOT / "figures/cite_pcfm_deep_story/biology_landscape_discovery"

PROGRAM_ORDER = ["HSC", "Erythroid", "Mk", "Myeloid"]
PUSH_EPS = 5.0
BATCH_SIZE = 512
STACK_COLORS = PHENO_COLORS


def load_trajectory_seed() -> tuple[np.ndarray, np.ndarray]:
    traj = np.load(TRAJECTORY_PATH, allow_pickle=True)
    time_points = np.asarray(traj["time_points"], dtype=np.float32)
    z0 = np.asarray(traj["points_000000"], dtype=np.float32)
    return time_points, z0[:, :RNA_DIM].copy()


def rollout_terminal_q(
    model,
    readout_input,
    r0: np.ndarray,
    time_points: np.ndarray,
    *,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """Roll RNA ODE from t0 to terminal time and return terminal protein q."""

    model = model.cpu().eval()
    n_fourier = int(getattr(model, "cite_pcfm_time_fourier_features", 0))
    time_fourier_scale = float(getattr(model, "cite_pcfm_time_fourier_scale", 1.0))
    q_parts: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, r0.shape[0], batch_size):
            end = min(start + batch_size, r0.shape[0])
            r = torch.as_tensor(r0[start:end], dtype=torch.float32)
            prev_t = float(time_points[0])
            for t in time_points[1:]:
                t_float = float(t)
                dt = t_float - prev_t
                if dt > 0:
                    t_mid = torch.full((end - start, 1), prev_t + 0.5 * dt, dtype=r.dtype)
                    zero_q = torch.zeros((end - start, PROTEIN_DIM), dtype=r.dtype)
                    vel = model.velocity_net(torch.cat([r, zero_q, t_mid], dim=1))[:, :RNA_DIM]
                    r = (r + vel * dt).detach()
                prev_t = t_float
            t_col = torch.full((end - start, 1), float(time_points[-1]), dtype=r.dtype)
            q = model.protein_readout(readout_input(r, t_col, n_fourier, time_fourier_scale))
            q_parts.append(q.detach().cpu().numpy().astype(np.float32))
    return np.vstack(q_parts)


def condition_rna(r0: np.ndarray, direction: np.ndarray | None, scale: float) -> np.ndarray:
    if direction is None or abs(scale) < 1e-12:
        return r0.copy()
    return r0 + float(scale) * direction.reshape(1, -1).astype(np.float32)


def summarize_conditions(readout, terminal_q_by_condition: dict[tuple[str, str], np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    particle_rows = []
    for (program, condition), q in terminal_q_by_condition.items():
        proba = readout.predict_proba_ordered(q)
        pred = readout.predict(q)
        for j, phenotype in enumerate(PHENO_ORDER):
            rows.append(
                {
                    "program": program,
                    "condition": condition,
                    "phenotype": phenotype,
                    "mean_probability": float(np.mean(proba[:, j])),
                    "hard_fraction": float(np.mean(pred.astype(str) == phenotype)),
                    "n_particles": int(q.shape[0]),
                    "push_eps": float(PUSH_EPS),
                }
            )
        for i in range(q.shape[0]):
            particle_rows.append(
                {
                    "program": program,
                    "condition": condition,
                    "particle_id": int(i),
                    "predicted_phenotype": str(pred[i]),
                    **{f"p_{phenotype}": float(proba[i, j]) for j, phenotype in enumerate(PHENO_ORDER)},
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(particle_rows)


def build_delta_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for program in PROGRAM_ORDER:
        pivot = (
            summary_df[summary_df["program"] == program]
            .pivot(index="phenotype", columns="condition", values="mean_probability")
            .reindex(PHENO_ORDER)
        )
        for phenotype in PHENO_ORDER:
            minus = float(pivot.loc[phenotype, "-program"])
            base = float(pivot.loc[phenotype, "baseline"])
            plus = float(pivot.loc[phenotype, "+program"])
            rows.append(
                {
                    "program": program,
                    "phenotype": phenotype,
                    "minus_probability": minus,
                    "baseline_probability": base,
                    "plus_probability": plus,
                    "plus_minus_baseline_pp": 100.0 * (plus - base),
                    "plus_minus_minus_pp": 100.0 * (plus - minus),
                    "minus_minus_baseline_pp": 100.0 * (minus - base),
                }
            )
    return pd.DataFrame(rows)


def plot_stacked_bar(summary_df: pd.DataFrame, delta_df: pd.DataFrame, out_png: Path) -> None:
    setup()
    fig, axes = plt.subplots(
        2,
        4,
        figsize=(15.4, 7.1),
        sharey="row",
        gridspec_kw={"height_ratios": [3.3, 1.35]},
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.055, right=0.985, top=0.795, bottom=0.155, hspace=0.50, wspace=0.18)
    condition_order = ["-program", "baseline", "+program"]
    eps_label = f"{PUSH_EPS:g}z"
    condition_labels = [f"-{eps_label}", "0", f"+{eps_label}"]

    for col, program in enumerate(PROGRAM_ORDER):
        ax = axes[0, col]
        sub = summary_df[summary_df["program"] == program]
        x = np.arange(len(condition_order), dtype=float)
        bottoms = np.zeros(len(condition_order), dtype=float)
        for phenotype in PHENO_ORDER:
            vals = (
                sub[sub["phenotype"] == phenotype]
                .set_index("condition")
                .reindex(condition_order)["mean_probability"]
                .to_numpy(dtype=float)
            )
            ax.bar(
                x,
                vals,
                bottom=bottoms,
                width=0.66,
                color=STACK_COLORS.get(phenotype, "#94A3B8"),
                edgecolor="white",
                linewidth=0.55,
                alpha=0.95,
            )
            bottoms += vals

        ax.set_xticks(x)
        ax.set_xticklabels(condition_labels, fontsize=8.2)
        ax.set_ylim(0, 1.0)
        ax.set_xlim(-0.55, len(condition_order) - 0.45)
        ax.set_title(f"{program} RNA program", color=RNA_PROGRAMS[program]["color"], fontsize=11.4, weight="bold")
        ax.set_xlabel("terminal rollout query", fontsize=8.2)
        clean_metric_axis(ax, grid_axis="y")

        d = delta_df[delta_df["program"] == program].copy()
        top = d.iloc[np.argmax(np.abs(d["plus_minus_minus_pp"].to_numpy(dtype=float)))]
        ax.text(
            0.02,
            1.018,
            f"max shift: {top.phenotype} {top.plus_minus_minus_pp:+.1f} pp",
            transform=ax.transAxes,
            fontsize=7.8,
            color="#334155",
            va="bottom",
        )
        if col == 0:
            ax.set_ylabel("terminal phenotype probability", fontsize=9.0)

        ax_delta = axes[1, col]
        delta = d.set_index("phenotype").reindex(PHENO_ORDER)["plus_minus_minus_pp"].to_numpy(dtype=float)
        y = np.arange(len(PHENO_ORDER))
        colors = [STACK_COLORS[p] for p in PHENO_ORDER]
        ax_delta.axvline(0, color="#94A3B8", lw=0.9)
        ax_delta.barh(y, delta, color=colors, alpha=0.88, height=0.55, edgecolor="white", linewidth=0.4)
        ax_delta.set_yticks(y)
        ax_delta.set_yticklabels(PHENO_ORDER, fontsize=7.0 if col == 0 else 0)
        if col != 0:
            ax_delta.set_yticklabels([])
        ax_delta.set_xlabel("Delta + vs -\npercentage points", fontsize=7.4)
        ax_delta.set_xlim(-9.0, 9.0)
        ax_delta.tick_params(axis="x", labelsize=7.0)
        clean_metric_axis(ax_delta, grid_axis="x")

    handles = [Patch(facecolor=STACK_COLORS[p], edgecolor="white", label=p) for p in PHENO_ORDER]
    fig.legend(handles=handles, frameon=False, ncol=7, loc="lower center", bbox_to_anchor=(0.5, -0.020), fontsize=8.2)
    fig.suptitle(
        "Fig06J. Counterfactual RNA-program push changes terminal protein phenotype composition",
        x=0.01,
        y=0.975,
        ha="left",
        fontsize=13.2,
        weight="bold",
    )
    fig.text(
        0.01,
        0.925,
        "Same t0 RNA seeds are shifted along a gene-program direction, rolled through CITE-PCFM RNA dynamics, and read as final protein phenotype probabilities.",
        ha="left",
        va="top",
        fontsize=9.0,
        color="#334155",
    )
    fig.savefig(out_png, dpi=330, bbox_inches="tight", facecolor="white")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> dict[str, object]:
    ensure_import_paths()
    if CYTOBRIDGE_SRC.exists() and str(CYTOBRIDGE_SRC) not in sys.path:
        sys.path.insert(0, str(CYTOBRIDGE_SRC))
    from CytoBridge.utils import load_model_from_adata

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(DATA_PATH)
    model_adata = ad.read_h5ad(TRAINED_MODEL_H5AD)
    _, rna_component_std, _ = projection_audit_and_scales(adata)
    _, _, directions, direction_df = build_program_directions(adata, rna_component_std)

    labels = adata.obs["cell_type"].astype(str).to_numpy()
    protein_latent = np.asarray(adata.obsm["X_protein_pca"], dtype=np.float32)[:, :PROTEIN_DIM]
    phenotype_readout, readout_metrics, confusion_df = train_mlp_protein_phenotype_readout(
        protein_latent,
        labels,
        PHENO_ORDER,
        RANDOM_STATE,
    )

    model = load_model_from_adata(model_adata)
    readout_input = load_readout_input()
    time_points, r0 = load_trajectory_seed()

    terminal_q_by_condition: dict[tuple[str, str], np.ndarray] = {}
    baseline_q = rollout_terminal_q(model, readout_input, r0, time_points)
    for program in PROGRAM_ORDER:
        terminal_q_by_condition[(program, "baseline")] = baseline_q
        direction = directions[program]
        for condition, scale in [("-program", -PUSH_EPS), ("+program", PUSH_EPS)]:
            terminal_q_by_condition[(program, condition)] = rollout_terminal_q(
                model,
                readout_input,
                condition_rna(r0, direction, scale),
                time_points,
            )

    summary_df, particle_df = summarize_conditions(phenotype_readout, terminal_q_by_condition)
    delta_df = build_delta_table(summary_df)

    summary_path = OUT_DIR / "fig06j_program_push_terminal_composition.csv"
    particle_path = OUT_DIR / "fig06j_program_push_terminal_particle_probabilities.csv"
    delta_path = OUT_DIR / "fig06j_program_push_terminal_composition_deltas.csv"
    confusion_path = OUT_DIR / "fig06j_terminal_composition_readout_confusion.csv"
    direction_path = OUT_DIR / "fig06j_rna_program_latent_directions.csv"
    summary_df.to_csv(summary_path, index=False)
    particle_df.to_csv(particle_path, index=False)
    delta_df.to_csv(delta_path, index=False)
    confusion_df.to_csv(confusion_path)
    direction_df.to_csv(direction_path, index=False)

    fig_path = FIG_DIR / "fig06j_program_push_terminal_composition_stackbar.png"
    plot_stacked_bar(summary_df, delta_df, fig_path)

    manifest = {
        "analysis": "fig06j_program_push_terminal_composition",
        "evidence_type": "model-derived counterfactual RNA-program push on CITE-PCFM terminal protein readout",
        "figure": str(fig_path),
        "figure_pdf": str(fig_path.with_suffix(".pdf")),
        "tables": {
            "terminal_composition": str(summary_path),
            "terminal_particle_probabilities": str(particle_path),
            "terminal_composition_deltas": str(delta_path),
            "readout_confusion": str(confusion_path),
            "rna_program_latent_directions": str(direction_path),
        },
        "data_h5ad": str(DATA_PATH),
        "trained_model_h5ad": str(TRAINED_MODEL_H5AD),
        "trajectory_seed_source": str(TRAJECTORY_PATH),
        "model_equation": "RNA ODE dr/dt=f_theta(r,t); terminal protein q=h_eta(r_T,T)",
        "condition_contract": {
            "baseline": "locked t0 RNA seed states rolled to terminal time",
            "+program": f"t0 RNA seed plus {PUSH_EPS:g} times the RNA-program latent direction, then rolled to terminal time",
            "-program": f"t0 RNA seed minus {PUSH_EPS:g} times the RNA-program latent direction, then rolled to terminal time",
            "rna_program_direction": "mean latent response to +1 z-score in available program genes using RNA SVD loadings",
        },
        "readout_contract": {
            "phenotype_readout": "StandardScaler + MLPClassifier trained on observed protein PCA and observed cell_type labels",
            "plotted_value": "mean terminal phenotype probability, not hard-count fraction",
            "classifier_metrics": readout_metrics,
        },
        "finite_difference_eps_from_fig06e": FINITE_DIFF_EPS,
        "display_push_eps": PUSH_EPS,
        "n_t0_seeds": int(r0.shape[0]),
        "n_rollout_timepoints": int(time_points.shape[0]),
        "semantic_warning": "This is an in-silico model query, not an experimental causal perturbation.",
    }
    manifest_path = OUT_DIR / "fig06j_program_push_terminal_composition_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    main()
