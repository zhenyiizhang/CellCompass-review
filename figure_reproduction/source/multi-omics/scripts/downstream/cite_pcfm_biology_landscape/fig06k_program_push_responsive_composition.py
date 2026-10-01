#!/usr/bin/env python3
"""Fig06K: state-dependent RNA-program-to-protein susceptibility.

Reads the per-particle probabilities from Fig06J and asks which t0 RNA seeds
are actually responsive to a +5z versus -5z RNA-program push.  This avoids
burying a branch-specific transfer signal under a global average over all
initial states.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/cytobridge_mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/cytobridge_xdg_cache")
for key in ["MPLCONFIGDIR", "XDG_CACHE_HOME"]:
    Path(os.environ[key]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from multistageot_style import PHENO_COLORS, clean_metric_axis, setup
from fig06_rna_protein_regulatory_map import PHENO_ORDER, RNA_PROGRAMS


ROOT = Path("/Users/zhenyizhang/Documents/GitHub/CytoBridge-agent-runtime-v2-20260309/multi-omics")
FIG_DIR = ROOT / "figures/cite_pcfm_deep_story/biology_landscape_discovery"
OUT_DIR = ROOT / "downstream/cite_pcfm_deep_story/biology_landscape_discovery_rna_protein_regulatory_map"

PARTICLE_PATH = OUT_DIR / "fig06j_program_push_terminal_particle_probabilities.csv"
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


def responsive_particle_ids(df: pd.DataFrame, program: str) -> tuple[np.ndarray, pd.DataFrame]:
    target = TARGET_BASIN[program]
    sub = df[df["program"] == program]
    plus = sub[sub["condition"] == "+program"].set_index("particle_id")[f"p_{target}"]
    minus = sub[sub["condition"] == "-program"].set_index("particle_id")[f"p_{target}"]
    delta = (plus - minus).dropna()
    threshold = float(delta.quantile(0.75))
    responsive = delta[delta >= threshold]
    summary = pd.DataFrame(
        {
            "program": [program],
            "target_basin": [target],
            "n_particles": [int(delta.shape[0])],
            "n_responsive_top_quartile": [int(responsive.shape[0])],
            "delta_threshold_pp": [100.0 * threshold],
            "mean_delta_all_pp": [100.0 * float(delta.mean())],
            "mean_delta_responsive_pp": [100.0 * float(responsive.mean())],
            "max_delta_pp": [100.0 * float(delta.max())],
        }
    )
    return responsive.index.to_numpy(dtype=int), summary


def summarize_subset(df: pd.DataFrame, program: str, particle_ids: np.ndarray) -> pd.DataFrame:
    rows = []
    sub = df[(df["program"] == program) & (df["particle_id"].isin(particle_ids))]
    for condition in CONDITION_ORDER:
        cond = sub[sub["condition"] == condition]
        for phenotype in PHENO_ORDER:
            rows.append(
                {
                    "program": program,
                    "condition": condition,
                    "phenotype": phenotype,
                    "mean_probability": float(cond[f"p_{phenotype}"].mean()),
                    "n_particles": int(cond.shape[0]),
                    "target_basin": TARGET_BASIN[program],
                }
            )
    return pd.DataFrame(rows)


def plot_responsive_stackbars(summary_df: pd.DataFrame, stats_df: pd.DataFrame, out_png: Path) -> None:
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

        stat = stats_df[stats_df["program"] == program].iloc[0]
        target = str(stat["target_basin"])
        ax.set_title(
            f"{program} -> {target}",
            color=RNA_PROGRAMS[program]["color"],
            fontsize=11.6,
            weight="bold",
        )
        ax.text(
            0.03,
            0.985,
            f"top 25% Delta target {stat.mean_delta_responsive_pp:+.1f} pp",
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
    axes[0].set_ylabel("terminal phenotype probability\nresponsive t0 seeds only", fontsize=9.0)

    handles = [Patch(facecolor=STACK_COLORS[p], edgecolor="white", label=p) for p in PHENO_ORDER]
    fig.legend(handles=handles, frameon=False, ncol=7, loc="lower center", bbox_to_anchor=(0.5, 0.04), fontsize=8.2)
    fig.suptitle(
        "Fig06K. RNA-to-protein transfer is state-dependent in susceptible t0 seeds",
        x=0.01,
        y=0.97,
        ha="left",
        fontsize=13.2,
        weight="bold",
    )
    fig.text(
        0.01,
        0.90,
        "For each program, responsive seeds are the top quartile by terminal target-basin probability increase under +5z versus -5z program push.",
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
    particle_df = pd.read_csv(PARTICLE_PATH)

    summaries = []
    stats = []
    for program in PROGRAM_ORDER:
        ids, stat = responsive_particle_ids(particle_df, program)
        summaries.append(summarize_subset(particle_df, program, ids))
        stats.append(stat)
    summary_df = pd.concat(summaries, ignore_index=True)
    stats_df = pd.concat(stats, ignore_index=True)

    summary_path = OUT_DIR / "fig06k_responsive_program_push_terminal_composition.csv"
    stats_path = OUT_DIR / "fig06k_responsive_program_push_stats.csv"
    summary_df.to_csv(summary_path, index=False)
    stats_df.to_csv(stats_path, index=False)

    fig_path = FIG_DIR / "fig06k_program_push_responsive_terminal_composition_stackbar.png"
    plot_responsive_stackbars(summary_df, stats_df, fig_path)

    manifest = {
        "analysis": "fig06k_responsive_program_push_terminal_composition",
        "input_particle_probabilities": str(PARTICLE_PATH),
        "figure": str(fig_path),
        "figure_pdf": str(fig_path.with_suffix(".pdf")),
        "tables": {
            "responsive_composition": str(summary_path),
            "responsive_stats": str(stats_path),
        },
        "responsive_definition": "top quartile of particles by target-basin probability increase under +5z versus -5z RNA-program push",
        "target_basin": TARGET_BASIN,
        "semantic_warning": "Responsive seeds are selected using model-derived counterfactual target-basin shifts; this is a state-dependent model query, not an experimental perturbation.",
    }
    manifest_path = OUT_DIR / "fig06k_responsive_program_push_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    main()
