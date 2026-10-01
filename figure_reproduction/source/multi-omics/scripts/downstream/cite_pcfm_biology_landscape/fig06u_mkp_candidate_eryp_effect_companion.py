#!/usr/bin/env python3
"""Companion check for Fig C3: how the same Mk/Ery-Mk RNA queries affect EryP.

This plot does not run new model perturbations. It visualizes the existing
fig06s single-gene +5z HSC-start CITE-PCFM query table, using the same 16
available genes as the MkP panel.
"""

from __future__ import annotations

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


ROOT = Path("/Users/zhenyizhang/Documents/GitHub/CytoBridge-agent-runtime-v2-20260309/multi-omics")
TABLE_DIR = ROOT / "downstream/cite_pcfm_deep_story/biology_landscape_discovery_rna_protein_regulatory_map"
FIG_DIR = ROOT / "figures/cite_pcfm_deep_story/biology_landscape_discovery"
OUT_BASE = FIG_DIR / "fig06u_mkp_candidate_eryp_effect_companion"

GENE_CLASS_COLORS = {
    "Mk effector": "#0F766E",
    "Mk marker": "#DB2777",
    "Mk regulator": "#7C3AED",
    "Ery-Mk regulator": "#EA580C",
}
CLASS_LABELS = {
    "Mk effector": "Mk effector",
    "Mk marker": "Mk marker",
    "Mk regulator": "Mk regulator",
    "Ery-Mk regulator": "Ery/Mk TF",
}


def clean_axis(ax: plt.Axes) -> None:
    ax.axvline(0, color="#CBD5E1", lw=1.0, zorder=0)
    ax.grid(axis="x", color="#E5E7EB", lw=0.7, alpha=0.85)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=8)
    ax.set_axisbelow(True)


def draw_lollipop(ax: plt.Axes, df: pd.DataFrame, y: np.ndarray, value_col: str, title: str, xlim: tuple[float, float]) -> None:
    values = df[value_col].to_numpy(float)
    for yi, (_, row), value in zip(y, df.iterrows(), values):
        color = GENE_CLASS_COLORS.get(str(row["gene_class"]), "#64748B")
        ax.hlines(yi, 0, value, color=color, lw=3.0, alpha=0.34, zorder=1)
        ax.scatter(value, yi, s=54, color=color, edgecolor="white", linewidth=0.8, zorder=3)
        ax.text(
            value + (0.11 if value >= 0 else -0.11),
            yi,
            f"{value:+.1f}",
            ha="left" if value >= 0 else "right",
            va="center",
            fontsize=7.1,
            color="#475569",
        )
    clean_axis(ax)
    ax.set_title(title, fontsize=11.2, weight="bold", color="#111827", pad=8)
    ax.set_xlim(*xlim)
    ax.set_xlabel("change in terminal phenotype probability (percentage points)", fontsize=8.4)


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )

    df = pd.read_csv(TABLE_DIR / "fig06s_mkp_hsc_gene_push_deltas.csv").copy()
    df = df.sort_values("plus_minus_baseline_p_MkP_pp", ascending=True).reset_index(drop=True)
    y = np.arange(df.shape[0])

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(8.8, 6.4),
        sharey=True,
        gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.18},
    )
    fig.subplots_adjust(left=0.19, right=0.97, top=0.82, bottom=0.14)

    draw_lollipop(
        axes[0],
        df,
        y,
        "plus_minus_baseline_p_MkP_pp",
        "MkP gain",
        (-2.0, 6.8),
    )
    draw_lollipop(
        axes[1],
        df,
        y,
        "delta_p_EryP_pp",
        "EryP gain from the same RNA query",
        (-3.4, 3.4),
    )

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(df["gene"], fontsize=8.5)
    for tick, (_, row) in zip(axes[0].get_yticklabels(), df.iterrows()):
        tick.set_color(GENE_CLASS_COLORS.get(str(row["gene_class"]), "#334155"))
        tick.set_fontweight("bold" if row["gene"] in {"THBS1", "TUBB1", "GP9", "GP1BA", "KLF1"} else "normal")
    axes[1].tick_params(axis="y", labelleft=False)

    fig.text(
        0.02,
        0.95,
        "Same RNA-gene queries show Mk gain is not simply erythroid gain",
        fontsize=16,
        weight="bold",
        ha="left",
        color="#111827",
    )
    fig.text(
        0.02,
        0.895,
        "Same 16 biology-selected genes; each query starts from 972 HSC t0 RNA seeds and adds +5z in one RNA-gene direction.",
        fontsize=9.3,
        ha="left",
        color="#475569",
    )

    handles = [
        Patch(facecolor=color, edgecolor="white", label=CLASS_LABELS.get(label, label))
        for label, color in GENE_CLASS_COLORS.items()
    ]
    fig.legend(handles=handles, frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.57, 0.035), fontsize=8.0)

    top_mk = df.sort_values("plus_minus_baseline_p_MkP_pp", ascending=False).head(6)
    anti_ery = int((top_mk["delta_p_EryP_pp"] <= 0).sum())
    fig.text(
        0.58,
        0.845,
        f"Among top 6 MkP-gain genes, {anti_ery}/6 do not increase EryP.",
        fontsize=9.0,
        ha="left",
        color="#334155",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#CBD5E1", "linewidth": 0.8},
    )

    for suffix in [".png", ".pdf", ".svg"]:
        fig.savefig(OUT_BASE.with_suffix(suffix), bbox_inches="tight", dpi=360)
    plt.close(fig)
    print(OUT_BASE.with_suffix(".png"))
    print(OUT_BASE.with_suffix(".pdf"))
    print(OUT_BASE.with_suffix(".svg"))


if __name__ == "__main__":
    main()
