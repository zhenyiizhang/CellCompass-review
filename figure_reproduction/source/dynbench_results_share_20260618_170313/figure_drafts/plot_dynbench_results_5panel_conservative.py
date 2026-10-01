#!/usr/bin/env python3
"""Conservative DynBench result block: benchmark results plus one integrated method example."""

from pathlib import Path
import os

OUT_DIR = Path(__file__).resolve().parent
MPL_DIR = OUT_DIR / ".mplconfig"
MPL_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.colors import LinearSegmentedColormap

from plot_dynbench_extended_7panel import (
    COLORS,
    METRIC_COLS,
    DIFFICULTY_ORDER,
    load_data,
    draw_panel_c,
    panel_label,
    pretty_agent,
    style_axis,
    get_baseline_diff_series,
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


def draw_panel_d_clean(ax, baseline):
    panel_label(ax, "d", "Six-task performance profile")
    matrix = baseline[METRIC_COLS].to_numpy()
    cmap = LinearSegmentedColormap.from_list("dynbench_score", ["#F8F9FA", "#BFE3EC", COLORS["teal"]])
    im = ax.imshow(matrix, aspect="auto", vmin=0.2, vmax=1.0, cmap=cmap)
    ax.set_yticks(np.arange(len(baseline)))
    ax.set_yticklabels([pretty_agent(x) for x in baseline["label"]], linespacing=0.92)
    ax.set_xticks(np.arange(len(METRIC_COLS)))
    ax.set_xticklabels(["velocity", "growth", "holdout", "fate", "perturb.", "GRN"], rotation=35, ha="right")
    ax.tick_params(length=0)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            txt_color = "white" if matrix[i, j] > 0.72 else COLORS["dark"]
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=6.3, color=txt_color)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cax = ax.inset_axes([0.90, 0.03, 0.035, 0.30])
    cb = plt.colorbar(im, cax=cax, ticks=[0.2, 0.6, 1.0])
    cb.ax.tick_params(labelsize=5.8, length=2)
    cb.outline.set_linewidth(0.4)


def draw_panel_e_baseline_difficulty(ax, diff):
    panel_label(ax, "e", "Difficulty scaling")
    x = np.arange(len(DIFFICULTY_ORDER))
    series = [
        ("Ours (GPT-5.5 high)", "Our CytoBridge Codex55 high", COLORS["blue"]),
        ("Ours (Mimo-v2.5-pro)", "Our CytoBridge Xiaomi high", COLORS["teal"]),
        ("Biomni", "Biomni", COLORS["gray"]),
        ("Claude Code (DeepSeek)", "DeepSeek v4 pro + Claude Code", COLORS["purple"]),
    ]
    for label, source, color in series:
        values = get_baseline_diff_series(diff, source)
        ax.plot(x, values, marker="o", color=color, lw=1.6, ms=4.2, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(["easy", "medium", "hard", "adv."])
    ax.set_ylabel("Total score")
    ax.set_ylim(0.42, 0.83)
    ax.set_xlim(-0.15, 3.15)
    style_axis(ax, "y")
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 0.0), handlelength=1.6)


def draw_panel_f_brgu_profile(ax, baseline, algorithms):
    panel_label(ax, "f", "Integrated method example")
    best = baseline.iloc[0]
    brgu = algorithms[algorithms["method"].eq("BRGUSFM")].iloc[0]
    x = np.arange(len(METRIC_COLS))
    best_vals = [best[m] for m in METRIC_COLS]
    brgu_vals = [brgu[m] for m in METRIC_COLS]
    ax.plot(x, best_vals, color=COLORS["blue"], marker="o", lw=1.8, ms=4.5, label="Best baseline")
    ax.plot(x, brgu_vals, color=COLORS["orange"], marker="o", lw=1.8, ms=4.5, label="BRGU-SFM")
    ax.set_xticks(x)
    ax.set_xticklabels(["velocity", "growth", "holdout", "fate", "perturb.", "GRN"], rotation=35, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0.2, 1.0)
    style_axis(ax, "y")
    ax.legend(frameon=False, loc="lower left")
    ax.text(
        0.98,
        0.96,
        "six-task objective",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=COLORS["mid"],
        fontsize=7,
    )


def draw_panel_g_brgu_schematic(ax):
    panel_label(ax, "g", "BRGU-SFM design")
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    def box(x, y, w, h, text, fc="#F7F9FB", ec=COLORS["dark"]):
        ax.add_patch(
            patches.FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.012,rounding_size=0.018",
                facecolor=fc,
                edgecolor=ec,
                lw=1.0,
            )
        )
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.5, color=COLORS["dark"])

    def arr(x0, y0, x1, y1):
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(arrowstyle="-|>", lw=1.1, color=COLORS["mid"], shrinkA=2, shrinkB=2),
        )

    box(0.05, 0.64, 0.26, 0.16, "time-course\nsnapshots", fc="#F3F6F9")
    box(0.05, 0.26, 0.26, 0.16, "task\nmetadata", fc="#F3F6F9")
    box(0.39, 0.54, 0.28, 0.18, "unbalanced\nstochastic bridge", fc="#FFF4E5", ec=COLORS["orange"])
    box(0.39, 0.29, 0.28, 0.16, "branch posterior\n+ regulatory drift", fc="#FFF4E5", ec=COLORS["orange"])
    box(0.76, 0.44, 0.18, 0.22, "six\nmodel-native\noutputs", fc="#EAF4F4", ec=COLORS["teal"])

    arr(0.31, 0.72, 0.39, 0.64)
    arr(0.31, 0.34, 0.39, 0.37)
    arr(0.53, 0.54, 0.53, 0.45)
    arr(0.67, 0.62, 0.76, 0.56)
    arr(0.67, 0.37, 0.76, 0.50)

    outputs = ["velocity", "growth", "holdout", "fate", "perturbation", "GRN"]
    for i, label in enumerate(outputs):
        ax.text(0.82, 0.35 - i * 0.045, label, ha="center", va="center", fontsize=6.8, color=COLORS["mid"])


def main():
    baseline, diff, _alg_diff, algorithms = load_data()

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
    draw_panel_d_clean(ax_d, baseline)
    draw_panel_e_baseline_difficulty(ax_e, diff)
    draw_panel_f_brgu_profile(ax_f, baseline, algorithms)
    draw_panel_g_brgu_schematic(ax_g)

    for suffix in ["png", "pdf", "svg"]:
        output = OUT_DIR / f"dynbench_results_5panel_conservative.{suffix}"
        fig.savefig(output, dpi=450 if suffix == "png" else None)
        print(output)


if __name__ == "__main__":
    main()
