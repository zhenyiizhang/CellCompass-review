#!/usr/bin/env python3
"""Result-only DynBench figure block: five panels c-g to place after panels a/b."""

from pathlib import Path
import os

OUT_DIR = Path(__file__).resolve().parent
MPL_DIR = OUT_DIR / ".mplconfig"
MPL_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

import matplotlib as mpl
import matplotlib.pyplot as plt

from plot_dynbench_extended_7panel import (
    load_data,
    draw_panel_c,
    draw_panel_d_heatmap,
    draw_panel_e_difficulty,
    draw_panel_f_algorithms,
    draw_panel_g_delta,
)


mpl.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 8,
        "axes.titlesize": 10,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "legend.fontsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def main():
    baseline, diff, alg_diff, algorithms = load_data()

    fig = plt.figure(figsize=(13.2, 7.3), facecolor="white")
    gs = fig.add_gridspec(
        2,
        6,
        height_ratios=[1.02, 1.0],
        hspace=0.54,
        wspace=0.72,
        left=0.095,
        right=0.985,
        top=0.93,
        bottom=0.10,
    )

    ax_c = fig.add_subplot(gs[0, 0:3])
    ax_d = fig.add_subplot(gs[0, 3:6])
    ax_e = fig.add_subplot(gs[1, 0:2])
    ax_f = fig.add_subplot(gs[1, 2:4])
    ax_g = fig.add_subplot(gs[1, 4:6])

    draw_panel_c(ax_c, baseline)
    draw_panel_d_heatmap(ax_d, baseline)
    draw_panel_e_difficulty(ax_e, diff, alg_diff)
    draw_panel_f_algorithms(ax_f, algorithms)
    draw_panel_g_delta(ax_g, algorithms)

    for suffix in ["png", "pdf", "svg"]:
        output = OUT_DIR / f"dynbench_results_5panel_draft.{suffix}"
        fig.savefig(output, dpi=450 if suffix == "png" else None)
        print(output)


if __name__ == "__main__":
    main()
