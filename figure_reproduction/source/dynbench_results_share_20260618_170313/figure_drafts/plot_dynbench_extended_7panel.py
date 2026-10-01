#!/usr/bin/env python3
"""Draft layout with existing design panels a/b plus five result panels c-g."""

from pathlib import Path
import os

OUT_DIR = Path(__file__).resolve().parent
RESULT_DIR = OUT_DIR.parent
MPL_DIR = OUT_DIR / ".mplconfig"
MPL_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap

from plot_dynbench_main_figure import (
    COLORS,
    BASELINE_NAME_MAP,
    METRIC_COLS,
    METRIC_SHORT,
    DIFFICULTY_ORDER,
    draw_panel_b,
    arrow,
    draw_nodes,
    panel_label,
)


mpl.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 7.5,
        "axes.titlesize": 9,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def pretty_agent(label):
    return label.replace(" (", "\n(")


def panel_label(ax, letter, title):
    ax.text(
        0.0,
        1.04,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=14,
        fontweight="bold",
        color=COLORS["dark"],
    )
    ax.text(
        0.075,
        1.05,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        color=COLORS["dark"],
    )


def load_data():
    overview = pd.read_csv(RESULT_DIR / "00_overview" / "main_metrics_overview.csv")
    baseline = overview[overview["kind"].eq("baseline_main100")].copy()
    baseline["label"] = baseline["name"].map(BASELINE_NAME_MAP)
    baseline = baseline.sort_values("total_score", ascending=False)

    diff = pd.read_csv(
        RESULT_DIR
        / "01_baseline_raw_rankings"
        / "benchmark_rankings_20260608_0142_corrected_deepseek_selected20"
        / "main100_repeat1_by_difficulty.csv"
    )
    alg_diff = pd.read_csv(
        RESULT_DIR
        / "05_m2_growth_algorithm_grjwfrfm"
        / "100case_external_eval_final_20260608_0842"
        / "means_by_prefix.csv"
    )
    alg_diff["difficulty"] = alg_diff["prefix"].str.replace("DB100_", "", regex=False)

    grj = pd.read_csv(
        RESULT_DIR
        / "05_m2_growth_algorithm_grjwfrfm"
        / "100case_external_eval_final_20260608_0842"
        / "overall_means.csv"
    ).iloc[0]
    brg = pd.read_csv(
        RESULT_DIR
        / "06_total_score_algorithm_brgusfm"
        / "100case_external_eval_final_locked"
        / "summary.csv"
    )[["total_score", *METRIC_COLS]].mean()
    spf = pd.read_csv(
        RESULT_DIR
        / "07_extra_m4_fate_algorithm_spftcfm"
        / "100case_external_eval_final"
        / "summary.csv"
    )[["total_score", *METRIC_COLS]].mean()

    algorithms = pd.DataFrame(
        [
            {"method": "Best baseline", **baseline.iloc[0][["total_score", *METRIC_COLS]].to_dict()},
            {"method": "GRJ-VGFM", **grj[["total_score", *METRIC_COLS]].to_dict()},
            {"method": "SPFT-CFM", **spf.to_dict()},
            {"method": "BRGUSFM", **brg.to_dict()},
        ]
    )
    return baseline, diff, alg_diff, algorithms


def draw_panel_a_wide(ax):
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    panel_label(ax, "a", "DynBench workflow")

    ax.text(0.14, 0.89, "Scenario design", ha="center", va="center", fontsize=10)
    ax.text(0.49, 0.89, "Simulated time course\nand model input", ha="center", va="center", fontsize=10)
    ax.text(0.82, 0.89, "Predicted dynamics\nand evaluation", ha="center", va="center", fontsize=10)

    # GRN icon.
    grn = [(0.055, 0.63), (0.10, 0.72), (0.145, 0.63), (0.10, 0.54)]
    for p0, p1 in zip(grn, grn[1:] + grn[:1]):
        ax.add_line(Line2D([p0[0], p1[0]], [p0[1], p1[1]], transform=ax.transAxes, color=COLORS["dark"], lw=1.2))
    draw_nodes(ax, grn, COLORS["grid"], size=0.013)
    ax.text(0.10, 0.45, "GRN", ha="center", va="center")

    # Fate tree icon.
    root = (0.25, 0.72)
    nodes1 = [(0.21, 0.61), (0.29, 0.61)]
    leaves = [(0.19, 0.53), (0.23, 0.53), (0.27, 0.53), (0.31, 0.53)]
    for p in nodes1:
        ax.plot([root[0], p[0]], [root[1], p[1]], color=COLORS["dark"], lw=1.1, transform=ax.transAxes)
    for parent, pair in zip(nodes1, [leaves[:2], leaves[2:]]):
        for leaf in pair:
            ax.plot([parent[0], leaf[0]], [parent[1], leaf[1]], color=COLORS["dark"], lw=1.1, transform=ax.transAxes)
    draw_nodes(ax, [root], COLORS["blue"], size=0.012)
    draw_nodes(ax, nodes1, COLORS["green"], size=0.010)
    draw_nodes(ax, leaves, COLORS["orange"], size=0.009)
    ax.text(0.25, 0.45, "fate tree", ha="center", va="center")

    # Growth/death icon.
    xs = np.linspace(0.055, 0.15, 8)
    ys = 0.24 + 0.055 * np.sin(np.linspace(0.2, 2.4, 8))
    ax.plot(xs, ys, color=COLORS["dark"], lw=1.1, transform=ax.transAxes)
    ax.axvline(0.103, ymin=0.19, ymax=0.34, color=COLORS["dark"], lw=1.0)
    draw_nodes(ax, list(zip(xs[:4], ys[:4] + 0.04)), COLORS["blue"], size=0.0055)
    draw_nodes(ax, list(zip(xs[4:], ys[4:] + 0.02)), COLORS["gray"], size=0.0055)
    ax.text(0.103, 0.11, "growth/death", ha="center", va="center")

    # Perturbation icon.
    ax.scatter([0.235, 0.305], [0.25, 0.25], s=[85, 110], c=[COLORS["blue"], COLORS["purple"]], transform=ax.transAxes, zorder=3)
    arrow(ax, (0.248, 0.25), (0.292, 0.25), lw=1.3)
    ax.text(0.27, 0.11, "perturbation", ha="center", va="center")

    arrow(ax, (0.35, 0.50), (0.41, 0.50))

    # Model-input strip.
    x0, width = 0.41, 0.23
    for i in range(6):
        xpos = x0 + i * width / 5
        if i in [2, 4]:
            ax.add_patch(
                patches.Rectangle(
                    (xpos - 0.017, 0.36),
                    0.034,
                    0.17,
                    fill=False,
                    ec=COLORS["dark"],
                    lw=1.0,
                    ls=(0, (3, 2)),
                    transform=ax.transAxes,
                )
            )
        else:
            col = [COLORS["blue"], COLORS["green"], COLORS["orange"], COLORS["purple"]][min(i, 3) % 4]
            pts = [(xpos + dx, 0.41 + dy) for dx in [-0.010, 0, 0.010] for dy in [0, 0.035, 0.070]]
            draw_nodes(ax, pts, col, size=0.0045)
        ax.text(xpos, 0.30, str(i), ha="center", va="center", fontsize=7)
    ax.plot([x0 - 0.02, x0 + width + 0.02], [0.33, 0.33], color=COLORS["dark"], lw=1.0, transform=ax.transAxes)
    arrow(ax, (x0 + width + 0.012, 0.33), (x0 + width + 0.04, 0.33), lw=1.0)
    ax.text(0.50, 0.61, "training data", ha="center", va="center", fontsize=8)
    ax.text(0.52, 0.19, "withheld time points", ha="center", va="center", fontsize=8, color=COLORS["mid"])

    doc_x, doc_y = 0.58, 0.62
    ax.add_patch(patches.Rectangle((doc_x, doc_y), 0.050, 0.13, fill=False, ec=COLORS["dark"], lw=1.1, transform=ax.transAxes))
    for k in range(4):
        ax.plot([doc_x + 0.012, doc_x + 0.040], [doc_y + 0.100 - 0.024 * k] * 2, color=COLORS["mid"], lw=0.8, transform=ax.transAxes)
    ax.text(doc_x + 0.025, doc_y - 0.045, "task", ha="center", va="center")

    arrow(ax, (0.66, 0.50), (0.71, 0.50))

    # Square inset avoids distorting the wheel into an ellipse.
    wheel = ax.inset_axes([0.70, 0.15, 0.25, 0.68])
    wheel.set_axis_off()
    wheel.set_aspect("equal")
    cx, cy, r = 0.5, 0.5, 0.48
    labels = [
        "M1\nvelocity",
        "M2\ngrowth\ndrivers",
        "M3\nholdout\ndistribution",
        "M4\nfate\nprobability",
        "M5\nperturbation\nresponse",
        "M6\nGRN\nrecovery",
    ]
    for i, lab in enumerate(labels):
        theta1 = 90 - i * 60
        theta2 = 90 - (i + 1) * 60
        wheel.add_patch(
            patches.Wedge((cx, cy), r, theta2, theta1, width=0.21, facecolor="white", edgecolor=COLORS["dark"], lw=1.0)
        )
        ang = np.deg2rad((theta1 + theta2) / 2)
        wheel.text(cx + 0.36 * np.cos(ang), cy + 0.36 * np.sin(ang), lab, ha="center", va="center", fontsize=6.6)
    wheel.add_patch(patches.Circle((cx, cy), 0.18, facecolor="white", edgecolor=COLORS["dark"], lw=1.0))
    wheel.text(cx, cy, "DynBench", ha="center", va="center", fontsize=9)


def style_axis(ax, grid_axis="x"):
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(COLORS["dark"])
    ax.spines["bottom"].set_color(COLORS["dark"])
    ax.tick_params(colors=COLORS["dark"], length=2.5, width=0.8)
    if grid_axis:
        ax.grid(axis=grid_axis, color=COLORS["grid"], lw=0.55, alpha=0.9)


def draw_panel_c(ax, baseline):
    panel_label(ax, "c", "Baseline agents")
    y = np.arange(len(baseline))
    colors = [COLORS["teal"], COLORS["blue"]] + [COLORS["gray"]] * (len(baseline) - 2)
    ax.barh(y, baseline["total_score"], color=colors, height=0.62, edgecolor="none")
    ax.set_yticks(y)
    ax.set_yticklabels([pretty_agent(x) for x in baseline["label"]], linespacing=0.92)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.78)
    ax.set_xlabel("Total score")
    style_axis(ax, "x")
    ax.grid(axis="y", visible=False)
    for yi, score in zip(y, baseline["total_score"]):
        ax.text(score + 0.01, yi, f"{score:.3f}", ha="left", va="center", fontsize=7)


def draw_panel_d_heatmap(ax, baseline):
    panel_label(ax, "d", "Six-task performance profile")
    matrix = baseline[METRIC_COLS].to_numpy()
    cmap = LinearSegmentedColormap.from_list("dynbench_score", ["#F8F9FA", "#BFE3EC", COLORS["teal"]])
    im = ax.imshow(matrix, aspect="auto", vmin=0.2, vmax=1.0, cmap=cmap)
    ax.set_yticks(np.arange(len(baseline)))
    ax.set_yticklabels([pretty_agent(x) for x in baseline["label"]], linespacing=0.92)
    ax.set_xticks(np.arange(len(METRIC_SHORT)))
    ax.set_xticklabels(["velocity", "growth", "holdout", "fate", "perturb.", "GRN"], rotation=35, ha="right")
    ax.tick_params(length=0)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            txt_color = "white" if matrix[i, j] > 0.72 else COLORS["dark"]
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=6.3, color=txt_color)

    ax.add_patch(
        patches.Rectangle(
            (0.5, -0.5),
            1.0,
            len(baseline),
            fill=False,
            ec=COLORS["red"],
            lw=1.3,
            clip_on=False,
        )
    )
    ax.text(1, -0.70, "bottleneck", ha="center", va="center", fontsize=6.5, color=COLORS["red"])
    for spine in ax.spines.values():
        spine.set_visible(False)
    cax = ax.inset_axes([0.90, 0.03, 0.035, 0.30])
    cb = plt.colorbar(im, cax=cax, ticks=[0.2, 0.6, 1.0])
    cb.ax.tick_params(labelsize=5.8, length=2)
    cb.outline.set_linewidth(0.4)


def draw_panel_f_algorithms(ax, algorithms):
    panel_label(ax, "f", "Benchmark-guided methods")
    colors = [COLORS["blue"], COLORS["teal"], COLORS["purple"], COLORS["orange"]]
    x = np.arange(len(algorithms))
    ax.bar(x, algorithms["total_score"], color=colors, edgecolor="none", width=0.62)
    ax.set_xticks(x)
    ax.set_xticklabels(["Best\nbaseline", "GRJ-\nVGFM", "SPFT-\nCFM", "BRGUSFM"])
    ax.set_ylim(0, 0.86)
    ax.set_ylabel("Total score")
    style_axis(ax, "y")
    for xi, val in zip(x, algorithms["total_score"]):
        ax.text(xi, val + 0.018, f"{val:.2f}", ha="center", va="bottom", fontsize=7)
    ax.text(0.98, 0.96, "DB100", transform=ax.transAxes, ha="right", va="top", color=COLORS["mid"], fontsize=6.8)


def get_baseline_diff_series(diff, agent_name):
    sub = diff[diff["agent_run"].eq(agent_name)].set_index("difficulty")
    return sub.loc[DIFFICULTY_ORDER, "total_score"].values


def draw_panel_e_difficulty(ax, diff, alg_diff):
    panel_label(ax, "e", "Difficulty scaling")
    x = np.arange(len(DIFFICULTY_ORDER))
    series = [
        (
            "GRJ-VGFM",
            alg_diff.set_index("difficulty").loc[DIFFICULTY_ORDER, "total_score"].values,
            COLORS["teal"],
            2.1,
            5.2,
        ),
        (
            "Ours (GPT-5.5 high)",
            get_baseline_diff_series(diff, "Our CytoBridge Codex55 high"),
            COLORS["blue"],
            1.6,
            4.2,
        ),
        (
            "Biomni",
            get_baseline_diff_series(diff, "Biomni"),
            COLORS["gray"],
            1.3,
            3.6,
        ),
        (
            "Claude Code (DeepSeek)",
            get_baseline_diff_series(diff, "DeepSeek v4 pro + Claude Code"),
            COLORS["purple"],
            1.3,
            3.6,
        ),
    ]
    for label, values, color, lw, ms in series:
        ax.plot(x, values, marker="o", color=color, lw=lw, ms=ms, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(["easy", "medium", "hard", "adv."])
    ax.set_ylabel("Total score")
    ax.set_ylim(0.42, 0.88)
    ax.set_xlim(-0.15, 3.15)
    style_axis(ax, "y")
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 0.0), handlelength=1.6)


def draw_panel_g_delta(ax, algorithms):
    panel_label(ax, "g", "GRJ-VGFM metric shifts")
    base = algorithms[algorithms["method"].eq("Best baseline")].iloc[0]
    grj = algorithms[algorithms["method"].eq("GRJ-VGFM")].iloc[0]
    deltas = np.array([grj[m] - base[m] for m in METRIC_COLS])
    y = np.arange(len(METRIC_COLS))[::-1]
    names = ["Velocity", "Growth drivers", "Holdout dist.", "Fate prob.", "Perturbation", "GRN recovery"]
    ax.axvline(0, color=COLORS["dark"], lw=0.8)
    for yi, d in zip(y, deltas):
        color = COLORS["teal"] if d >= 0 else COLORS["red"]
        ax.plot([0, d], [yi, yi], color=color, lw=2.0)
        ax.scatter([d], [yi], color=color, s=28, zorder=3)
        ax.text(
            d + (0.025 if d >= 0 else -0.025),
            yi,
            f"{d:+.2f}",
            ha="left" if d >= 0 else "right",
            va="center",
            fontsize=6.6,
            color=color,
            clip_on=False,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlim(-0.32, 0.72)
    ax.set_xlabel("Score change vs best baseline")
    style_axis(ax, "x")
    ax.grid(axis="y", visible=False)


def main():
    baseline, diff, alg_diff, algorithms = load_data()

    fig = plt.figure(figsize=(13.2, 12.4), facecolor="white")
    gs = fig.add_gridspec(
        4,
        6,
        height_ratios=[1.28, 0.82, 1.06, 1.00],
        hspace=0.50,
        wspace=0.70,
        left=0.095,
        right=0.985,
        top=0.965,
        bottom=0.065,
    )

    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, :])
    ax_c = fig.add_subplot(gs[2, 0:3])
    ax_d = fig.add_subplot(gs[2, 3:6])
    ax_e = fig.add_subplot(gs[3, 0:2])
    ax_f = fig.add_subplot(gs[3, 2:4])
    ax_g = fig.add_subplot(gs[3, 4:6])

    draw_panel_a_wide(ax_a)
    draw_panel_b(ax_b)
    draw_panel_c(ax_c, baseline)
    draw_panel_d_heatmap(ax_d, baseline)
    draw_panel_e_difficulty(ax_e, diff, alg_diff)
    draw_panel_f_algorithms(ax_f, algorithms)
    draw_panel_g_delta(ax_g, algorithms)

    for suffix in ["png", "pdf", "svg"]:
        output = OUT_DIR / f"dynbench_extended_7panel_draft.{suffix}"
        fig.savefig(output, dpi=450 if suffix == "png" else None)
        print(output)


if __name__ == "__main__":
    main()
