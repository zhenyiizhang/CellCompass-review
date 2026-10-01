#!/usr/bin/env python3
"""Pooled cell-type sanity check for BMF growth.

This complements the time-split cell-type growth plot.  Because BMF growth has
a strong global time trend, the pooled plot shows two views:

1. Raw pooled BMF growth by cell type.
2. Within-day rank score pooled by cell type, which removes the day2/day4/day6
   global shift before comparing cell types.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cellcompass_mpl_cache")
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compose_clone_scdiffeq_style_biology as base  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
TABLE_DIR = ROOT / "downstream/cellcompass_clone_case"
FIG_DIR = ROOT / "figures/cellcompass_clone_case"
GROWTH_TABLE = TABLE_DIR / "clone_bmf_observed_cell_growth_spring.csv"

CELLTYPE_ORDER = ["Undifferentiated", "Neutrophil", "Monocyte", "Baso", "Mast", "Meg", "Erythroid"]
DISPLAY_LABELS = {
    "Undifferentiated": "Undiff.",
    "Neutrophil": "Neutrophil",
    "Monocyte": "Monocyte",
    "Baso": "Baso",
    "Mast": "Mast",
    "Meg": "Meg",
    "Erythroid": "Erythroid",
}


def setup_style() -> None:
    base.setup_style()
    mpl.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9.0,
            "axes.titleweight": "bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def add_within_day_rank(growth: pd.DataFrame) -> pd.DataFrame:
    out = growth.copy()
    out["growth_rank_within_day"] = np.nan
    out["growth_rank_score_within_day"] = np.nan
    for _, idx in out.groupby("t").groups.items():
        ranks = out.loc[idx, "gate_weighted_growth"].rank(method="average", pct=True).to_numpy(float)
        out.loc[idx, "growth_rank_within_day"] = ranks
        out.loc[idx, "growth_rank_score_within_day"] = 2.0 * (ranks - 0.5)
    return out


def summarize(growth: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cell_type, d in growth.groupby("cell_type"):
        raw = d["gate_weighted_growth"].to_numpy(float)
        rank = d["growth_rank_score_within_day"].to_numpy(float)
        rows.append(
            {
                "cell_type": str(cell_type),
                "n_cells": int(len(d)),
                "raw_median_growth": float(np.median(raw)),
                "raw_mean_growth": float(np.mean(raw)),
                "within_day_rank_median": float(np.median(rank)),
                "within_day_rank_mean": float(np.mean(rank)),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "clone_bmf_growth_by_celltype_pooled_summary.csv", index=False)
    return out


def plot(growth: pd.DataFrame, summary: pd.DataFrame) -> None:
    plot_df = growth[growth["cell_type"].isin(CELLTYPE_ORDER)].copy()
    plot_df["cell_type_display"] = plot_df["cell_type"].map(DISPLAY_LABELS)
    order_display = [DISPLAY_LABELS[x] for x in CELLTYPE_ORDER]
    palette = {DISPLAY_LABELS[k]: base.FATE_PALETTE.get(k, "#64748B") for k in CELLTYPE_ORDER}

    rng = np.random.default_rng(20260610)
    sampled = []
    for _, d in plot_df.groupby("cell_type"):
        n = min(len(d), 900)
        sampled.append(d.iloc[rng.choice(len(d), size=n, replace=False)] if len(d) > n else d)
    sample_df = pd.concat(sampled, ignore_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.45), gridspec_kw={"wspace": 0.22})
    specs = [
        ("gate_weighted_growth", "Raw pooled BMF growth", "BMF gate-weighted growth"),
        ("growth_rank_score_within_day", "Pooled after within-day ranking", "within-day growth rank score"),
    ]
    for ax, (y, title, ylabel) in zip(axes, specs, strict=True):
        sns.boxplot(
            data=plot_df,
            x="cell_type_display",
            y=y,
            order=order_display,
            hue="cell_type_display",
            hue_order=order_display,
            palette=palette,
            legend=False,
            width=0.62,
            fliersize=0,
            linewidth=0.85,
            ax=ax,
        )
        sns.stripplot(
            data=sample_df,
            x="cell_type_display",
            y=y,
            order=order_display,
            color="#111827",
            size=1.15,
            jitter=0.22,
            alpha=0.13,
            linewidth=0,
            ax=ax,
        )
        if y == "growth_rank_score_within_day":
            ax.axhline(0, color="#CBD5E1", lw=0.9)
            ax.set_ylim(-1.05, 1.05)
        ax.set_title(title, loc="left")
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", labelrotation=38, labelsize=6.8)
        ax.grid(axis="y", color="#E5E7EB", lw=0.75)
        sns.despine(ax=ax)

    undiff = summary[summary["cell_type"].eq("Undifferentiated")]
    if not undiff.empty:
        u = undiff.iloc[0]
        axes[1].text(
            0.02,
            0.96,
            f"Undiff. pooled rank median = {u['within_day_rank_median']:.2f}\n"
            f"Undiff. pooled rank mean = {u['within_day_rank_mean']:.2f}",
            transform=axes[1].transAxes,
            ha="left",
            va="top",
            fontsize=6.9,
            color="#334155",
            bbox=dict(boxstyle="round,pad=0.20", fc="white", ec="#CBD5E1", alpha=0.92),
        )

    fig.suptitle("BMF-inferred growth by cell type, pooled across all time points", x=0.02, y=0.995, ha="left", fontsize=12.8, weight="bold")
    fig.text(
        0.02,
        0.025,
        "Left mixes biological time and cell type composition. Right compares cells after ranking growth within each day, so it asks whether a cell type is high-growth relative to cells from the same time point.",
        fontsize=7.1,
        color="#475569",
    )
    fig.subplots_adjust(left=0.07, right=0.985, top=0.84, bottom=0.22)
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(FIG_DIR / f"panel_c_growthrate05_growth_by_celltype_pooled_sanity.{ext}", dpi=420 if ext == "png" else None)
    plt.close(fig)


def main() -> None:
    setup_style()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    if not GROWTH_TABLE.exists():
        raise FileNotFoundError(f"Missing {GROWTH_TABLE}. Run compose_clone_observed_growth_spring.py first.")
    growth = pd.read_csv(GROWTH_TABLE)
    growth = add_within_day_rank(growth)
    summary = summarize(growth)
    plot(growth, summary)
    print(
        {
            "figure": str(FIG_DIR / "panel_c_growthrate05_growth_by_celltype_pooled_sanity.png"),
            "summary_table": str(TABLE_DIR / "clone_bmf_growth_by_celltype_pooled_summary.csv"),
            "undifferentiated": summary[summary["cell_type"].eq("Undifferentiated")].to_dict(orient="records"),
        }
    )


if __name__ == "__main__":
    main()
