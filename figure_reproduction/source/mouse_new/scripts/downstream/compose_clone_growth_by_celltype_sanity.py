#!/usr/bin/env python3
"""Sanity check: BMF growth rate grouped by observed cell type.

Input is the model-derived table from compose_clone_observed_growth_spring.py:
each observed cell stays at its measured SPRING/latent state, and BMF
gate-weighted growth is evaluated at that state and time.

This script asks a simple question:
    Are immature/undifferentiated cells inferred to have higher local growth?
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

TIME_LABELS = {0: "day 2", 1: "day 4", 2: "day 6"}
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


def summarize_growth(growth: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (t, cell_type), d in growth.groupby(["t", "cell_type"]):
        vals = d["gate_weighted_growth"].to_numpy(float)
        if vals.size == 0:
            continue
        rows.append(
            {
                "t": int(t),
                "time_label": TIME_LABELS.get(int(t), str(t)),
                "cell_type": str(cell_type),
                "n_cells": int(vals.size),
                "mean_growth": float(np.mean(vals)),
                "median_growth": float(np.median(vals)),
                "q25_growth": float(np.percentile(vals, 25)),
                "q75_growth": float(np.percentile(vals, 75)),
                "q10_growth": float(np.percentile(vals, 10)),
                "q90_growth": float(np.percentile(vals, 90)),
            }
        )
    summary = pd.DataFrame(rows)
    summary["median_rank_within_day"] = np.nan
    summary["median_rank_score_within_day"] = np.nan
    for t, idx in summary.groupby("t").groups.items():
        d = summary.loc[idx].copy()
        rank = d["median_growth"].rank(method="average", pct=True)
        summary.loc[idx, "median_rank_within_day"] = rank.to_numpy(float)
        summary.loc[idx, "median_rank_score_within_day"] = 2.0 * (rank.to_numpy(float) - 0.5)
    summary.to_csv(TABLE_DIR / "clone_bmf_growth_by_celltype_summary.csv", index=False)
    return summary


def plot_growth_by_celltype(growth: pd.DataFrame, summary: pd.DataFrame) -> None:
    plot_df = growth[growth["cell_type"].isin(CELLTYPE_ORDER)].copy()
    plot_df["time_label"] = plot_df["t"].map(TIME_LABELS)
    plot_df["cell_type_display"] = plot_df["cell_type"].map(DISPLAY_LABELS)
    order_display = [DISPLAY_LABELS[x] for x in CELLTYPE_ORDER]
    palette = {DISPLAY_LABELS[k]: base.FATE_PALETTE.get(k, "#64748B") for k in CELLTYPE_ORDER}

    # Downsample only for scatter overlay; box statistics use all cells.
    rng = np.random.default_rng(20260610)
    scatter_parts = []
    for _, d in plot_df.groupby(["t", "cell_type"]):
        n = min(len(d), 420)
        scatter_parts.append(d.iloc[rng.choice(len(d), size=n, replace=False)] if len(d) > n else d)
    scatter_df = pd.concat(scatter_parts, ignore_index=True)

    fig = plt.figure(figsize=(12.0, 6.25))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 0.9], width_ratios=[1, 1, 1], hspace=0.42, wspace=0.22)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    for ax, t in zip(axes, [0, 1, 2], strict=True):
        d = plot_df[plot_df["t"].eq(t)]
        s = scatter_df[scatter_df["t"].eq(t)]
        sns.boxplot(
            data=d,
            x="cell_type_display",
            y="gate_weighted_growth",
            order=order_display,
            palette=palette,
            width=0.62,
            fliersize=0,
            linewidth=0.85,
            ax=ax,
        )
        sns.stripplot(
            data=s,
            x="cell_type_display",
            y="gate_weighted_growth",
            order=order_display,
            color="#111827",
            size=1.15,
            jitter=0.24,
            alpha=0.16,
            linewidth=0,
            ax=ax,
        )
        day_summary = summary[summary["t"].eq(t) & summary["cell_type"].isin(CELLTYPE_ORDER)].copy()
        undiff = day_summary[day_summary["cell_type"].eq("Undifferentiated")]
        if not undiff.empty:
            u = undiff.iloc[0]
            ax.text(
                0.02,
                0.96,
                f"Undiff. median = {u['median_growth']:.3f}\nrank score = {u['median_rank_score_within_day']:.2f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=6.7,
                color="#334155",
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#CBD5E1", alpha=0.92),
            )
        ax.set_title(f"{TIME_LABELS[t]} observed cells", loc="left")
        ax.set_xlabel("")
        ax.set_ylabel("BMF gate-weighted growth" if t == 0 else "")
        ax.tick_params(axis="x", labelrotation=42, labelsize=6.7)
        ax.grid(axis="y", color="#E5E7EB", lw=0.75)
        sns.despine(ax=ax)

    ax_line = fig.add_subplot(gs[1, :2])
    line_summary = summary[summary["cell_type"].isin(CELLTYPE_ORDER)].copy()
    for cell_type in CELLTYPE_ORDER:
        d = line_summary[line_summary["cell_type"].eq(cell_type)].sort_values("t")
        if d.empty:
            continue
        ax_line.plot(
            d["t"],
            d["median_growth"],
            marker="o",
            lw=2.0 if cell_type == "Undifferentiated" else 1.35,
            ms=4.2 if cell_type == "Undifferentiated" else 3.4,
            color=base.FATE_PALETTE.get(cell_type, "#64748B"),
            alpha=1.0 if cell_type == "Undifferentiated" else 0.72,
            label=DISPLAY_LABELS[cell_type],
        )
    ax_line.set_xticks([0, 1, 2])
    ax_line.set_xticklabels(["day 2", "day 4", "day 6"])
    ax_line.set_ylabel("median BMF growth")
    ax_line.set_title("median growth over biological time", loc="left")
    ax_line.grid(axis="y", color="#E5E7EB", lw=0.75)
    ax_line.legend(frameon=False, fontsize=6.5, ncol=4, loc="upper right")
    sns.despine(ax=ax_line)

    ax_rank = fig.add_subplot(gs[1, 2])
    rank_summary = summary[summary["cell_type"].eq("Undifferentiated")].sort_values("t")
    bars = ax_rank.bar(
        rank_summary["time_label"],
        rank_summary["median_rank_score_within_day"],
        color=base.FATE_PALETTE.get("Undifferentiated", "#2E8B57"),
        width=0.55,
    )
    ax_rank.axhline(0, color="#CBD5E1", lw=0.9)
    ax_rank.set_ylim(-1.05, 1.05)
    ax_rank.set_ylabel("within-day rank score")
    ax_rank.set_title("Undiff. relative rank", loc="left")
    ax_rank.grid(axis="y", color="#E5E7EB", lw=0.75)
    for bar, val in zip(bars, rank_summary["median_rank_score_within_day"], strict=False):
        ax_rank.text(bar.get_x() + bar.get_width() / 2, float(val) + (0.06 if val >= 0 else -0.09), f"{val:.2f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=7.0)
    sns.despine(ax=ax_rank)

    fig.suptitle("BMF-inferred growth by observed cell type", x=0.02, y=0.995, ha="left", fontsize=13.0, weight="bold")
    fig.text(
        0.02,
        0.02,
        "Each observed cell is evaluated by the final BMF model at its measured latent state and time. Growth is gate-weighted d log(mass)/dt; no rollout or clone counts are used in this plot.",
        fontsize=7.2,
        color="#475569",
    )
    fig.subplots_adjust(left=0.065, right=0.985, top=0.88, bottom=0.15)
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(FIG_DIR / f"panel_c_growthrate04_growth_by_celltype_sanity.{ext}", dpi=420 if ext == "png" else None)
    plt.close(fig)


def main() -> None:
    setup_style()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    if not GROWTH_TABLE.exists():
        raise FileNotFoundError(f"Missing {GROWTH_TABLE}. Run compose_clone_observed_growth_spring.py first.")
    growth = pd.read_csv(GROWTH_TABLE)
    summary = summarize_growth(growth)
    plot_growth_by_celltype(growth, summary)
    undiff = summary[summary["cell_type"].eq("Undifferentiated")][
        ["time_label", "n_cells", "median_growth", "median_rank_score_within_day"]
    ].to_dict(orient="records")
    print(
        {
            "figure": str(FIG_DIR / "panel_c_growthrate04_growth_by_celltype_sanity.png"),
            "summary_table": str(TABLE_DIR / "clone_bmf_growth_by_celltype_summary.csv"),
            "undifferentiated_summary": undiff,
        }
    )


if __name__ == "__main__":
    main()
