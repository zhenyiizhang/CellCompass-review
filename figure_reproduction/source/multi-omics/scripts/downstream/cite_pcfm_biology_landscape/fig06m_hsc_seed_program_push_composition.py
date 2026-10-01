#!/usr/bin/env python3
"""Fig06M: RNA-program counterfactuals starting only from HSC-labeled t0 cells."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/cytobridge_mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/cytobridge_xdg_cache")
for key in ["MPLCONFIGDIR", "XDG_CACHE_HOME"]:
    Path(os.environ[key]).mkdir(parents=True, exist_ok=True)

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from fig06_rna_protein_regulatory_map import DATA_PATH, PHENO_ORDER, RNA_DIM, RNA_PROGRAMS
from multistageot_style import PHENO_COLORS, clean_metric_axis, setup


ROOT = Path("/Users/zhenyizhang/Documents/GitHub/CytoBridge-agent-runtime-v2-20260309/multi-omics")
FIG_DIR = ROOT / "figures/cite_pcfm_deep_story/biology_landscape_discovery"
OUT_DIR = ROOT / "downstream/cite_pcfm_deep_story/biology_landscape_discovery_rna_protein_regulatory_map"
PARTICLE_PATH = OUT_DIR / "fig06j_program_push_terminal_particle_probabilities.csv"
TRAJECTORY_PATH = ROOT / "training_runs/20260526T174204321821Z__final__cite_paired_coord_fm__campaign_20260-trial_20260526-gse305370_ci/artifacts/evaluation_trajectory.npz"

PROGRAM_ORDER = ["HSC", "Erythroid", "Mk", "Myeloid"]
TARGET_BASIN = {
    "HSC": "HSC",
    "Erythroid": "EryP",
    "Mk": "MkP",
    "Myeloid": "NeuP",
}
CONDITION_ORDER = ["-program", "baseline", "+program"]
CONDITION_LABELS = ["-5z", "0", "+5z"]
STACK_COLORS = PHENO_COLORS


def hsc_t0_particle_ids(adata: ad.AnnData) -> tuple[np.ndarray, dict[str, object]]:
    times = np.asarray(adata.obs["time_point_processed"], dtype=float)
    labels = adata.obs["cell_type"].astype(str).to_numpy()
    t0_idx = np.where(np.isclose(times, 0.0))[0]
    traj = np.load(TRAJECTORY_PATH)
    z0 = np.asarray(traj["points_000000"], dtype=np.float32)
    rna = np.asarray(adata.obsm["X_rna_svd"], dtype=np.float32)[:, :RNA_DIM]
    if t0_idx.shape[0] != z0.shape[0]:
        raise ValueError(f"t0 observed count {t0_idx.shape[0]} != trajectory particles {z0.shape[0]}")
    ids = np.where(labels[t0_idx] == "HSC")[0]
    order_check = {
        "t0_observed_count": int(t0_idx.shape[0]),
        "trajectory_particle_count": int(z0.shape[0]),
        "hsc_t0_particle_count": int(ids.shape[0]),
        "hsc_t0_fraction": float(ids.shape[0] / max(1, t0_idx.shape[0])),
        "rna_mean_abs_diff_points0_vs_observed_t0": float(np.mean(np.abs(z0[:, :RNA_DIM] - rna[t0_idx]))),
        "rna_max_abs_diff_points0_vs_observed_t0": float(np.max(np.abs(z0[:, :RNA_DIM] - rna[t0_idx]))),
    }
    return ids.astype(int), order_check


def summarize_hsc_subset(particle_df: pd.DataFrame, hsc_ids: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    delta_rows = []
    for program in PROGRAM_ORDER:
        sub = particle_df[(particle_df["program"] == program) & (particle_df["particle_id"].isin(hsc_ids))]
        for condition in CONDITION_ORDER:
            cond = sub[sub["condition"] == condition]
            for phenotype in PHENO_ORDER:
                summary_rows.append(
                    {
                        "program": program,
                        "condition": condition,
                        "phenotype": phenotype,
                        "mean_probability": float(cond[f"p_{phenotype}"].mean()),
                        "n_hsc_t0_particles": int(cond.shape[0]),
                        "target_basin": TARGET_BASIN[program],
                    }
                )
        for phenotype in PHENO_ORDER:
            minus = float(sub[sub["condition"] == "-program"][f"p_{phenotype}"].mean())
            baseline = float(sub[sub["condition"] == "baseline"][f"p_{phenotype}"].mean())
            plus = float(sub[sub["condition"] == "+program"][f"p_{phenotype}"].mean())
            delta_rows.append(
                {
                    "program": program,
                    "phenotype": phenotype,
                    "minus_probability": minus,
                    "baseline_probability": baseline,
                    "plus_probability": plus,
                    "plus_minus_baseline_pp": 100.0 * (plus - baseline),
                    "plus_minus_minus_pp": 100.0 * (plus - minus),
                    "minus_minus_baseline_pp": 100.0 * (minus - baseline),
                    "target_basin": TARGET_BASIN[program],
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(delta_rows)


def plot_hsc_stackbars(summary_df: pd.DataFrame, delta_df: pd.DataFrame, out_png: Path) -> None:
    setup()
    fig, axes = plt.subplots(1, 4, figsize=(15.2, 4.9), sharey=True)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.76, bottom=0.23, wspace=0.18)
    for ax, program in zip(axes, PROGRAM_ORDER):
        sub = summary_df[summary_df["program"] == program]
        x = np.arange(len(CONDITION_ORDER), dtype=float)
        bottom = np.zeros(len(CONDITION_ORDER), dtype=float)
        for phenotype in PHENO_ORDER:
            vals = (
                sub[sub["phenotype"] == phenotype]
                .set_index("condition")
                .reindex(CONDITION_ORDER)["mean_probability"]
                .to_numpy(dtype=float)
            )
            ax.bar(
                x,
                vals,
                bottom=bottom,
                width=0.66,
                color=STACK_COLORS[phenotype],
                edgecolor="white",
                linewidth=0.55,
                alpha=0.96,
            )
            bottom += vals
        target = TARGET_BASIN[program]
        d = delta_df[(delta_df["program"] == program) & (delta_df["phenotype"] == target)].iloc[0]
        ax.set_title(f"{program} -> {target}", color=RNA_PROGRAMS[program]["color"], fontsize=11.6, weight="bold")
        ax.text(
            0.03,
            0.985,
            f"HSC-start Delta target {d.plus_minus_minus_pp:+.1f} pp",
            transform=ax.transAxes,
            fontsize=8.0,
            color="#334155",
            va="top",
            bbox={"boxstyle": "round,pad=0.16", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
        )
        ax.set_xticks(x)
        ax.set_xticklabels(CONDITION_LABELS, fontsize=8.4)
        ax.set_ylim(0, 1.0)
        ax.set_xlabel("RNA-program push", fontsize=8.6)
        clean_metric_axis(ax, grid_axis="y")
    axes[0].set_ylabel("terminal phenotype probability\nHSC-labeled t0 seeds only", fontsize=9.0)
    handles = [Patch(facecolor=STACK_COLORS[p], edgecolor="white", label=p) for p in PHENO_ORDER]
    fig.legend(handles=handles, frameon=False, ncol=7, loc="lower center", bbox_to_anchor=(0.5, 0.04), fontsize=8.2)
    fig.suptitle(
        "Fig06M. RNA-program counterfactuals starting from HSC-labeled t0 cells",
        x=0.01,
        y=0.97,
        ha="left",
        fontsize=13.2,
        weight="bold",
    )
    fig.text(
        0.01,
        0.90,
        "The 0 condition is the CITE-PCFM terminal rollout from the same HSC t0 seeds with no RNA-program shift; -5z/+5z are in-silico RNA-program pushes.",
        ha="left",
        va="top",
        fontsize=9.0,
        color="#334155",
    )
    fig.savefig(out_png, dpi=330, bbox_inches="tight", facecolor="white")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> dict[str, object]:
    if not PARTICLE_PATH.exists():
        raise FileNotFoundError(f"Run fig06j first; missing {PARTICLE_PATH}")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    adata = ad.read_h5ad(DATA_PATH)
    particle_df = pd.read_csv(PARTICLE_PATH)
    hsc_ids, order_check = hsc_t0_particle_ids(adata)
    summary_df, delta_df = summarize_hsc_subset(particle_df, hsc_ids)

    summary_path = OUT_DIR / "fig06m_hsc_seed_program_push_terminal_composition.csv"
    delta_path = OUT_DIR / "fig06m_hsc_seed_program_push_terminal_composition_deltas.csv"
    summary_df.to_csv(summary_path, index=False)
    delta_df.to_csv(delta_path, index=False)

    fig_path = FIG_DIR / "fig06m_hsc_seed_program_push_terminal_composition_stackbar.png"
    plot_hsc_stackbars(summary_df, delta_df, fig_path)

    target_delta = {}
    for program in PROGRAM_ORDER:
        target = TARGET_BASIN[program]
        row = delta_df[(delta_df["program"] == program) & (delta_df["phenotype"] == target)].iloc[0]
        target_delta[program] = {
            "target_basin": target,
            "minus_probability": float(row.minus_probability),
            "baseline_probability": float(row.baseline_probability),
            "plus_probability": float(row.plus_probability),
            "plus_minus_minus_pp": float(row.plus_minus_minus_pp),
        }
    manifest = {
        "analysis": "fig06m_hsc_seed_program_push_terminal_composition",
        "figure": str(fig_path),
        "figure_pdf": str(fig_path.with_suffix(".pdf")),
        "tables": {
            "hsc_seed_composition": str(summary_path),
            "hsc_seed_deltas": str(delta_path),
        },
        "input_particle_probabilities": str(PARTICLE_PATH),
        "seed_definition": "observed t0 cells with cell_type == HSC; same particle ordering as evaluation_trajectory points_000000",
        "condition_semantics": {
            "baseline": "CITE-PCFM terminal rollout from HSC-labeled t0 RNA seeds with no RNA-program shift",
            "-program/+program": "same HSC-labeled t0 RNA seeds shifted by -5z/+5z along the RNA-program latent direction before rollout",
        },
        "target_basin": TARGET_BASIN,
        "target_delta": target_delta,
        "source_order_check": order_check,
    }
    manifest_path = OUT_DIR / "fig06m_hsc_seed_program_push_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    main()
