#!/usr/bin/env python3
"""Draw a five-panel DynBench main figure draft from shared result tables."""

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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


mpl.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 8,
        "axes.titlesize": 10,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


COLORS = {
    "dark": "#202124",
    "mid": "#5F6368",
    "grid": "#DADCE0",
    "pale": "#F7F9FB",
    "blue": "#4C9BE8",
    "green": "#88C38A",
    "orange": "#F28E2B",
    "red": "#E65A5A",
    "teal": "#1F9A9B",
    "purple": "#7E57C2",
    "gray": "#9AA0A6",
}

DIFFICULTY_COLORS = {
    "easy": COLORS["blue"],
    "medium": COLORS["green"],
    "hard": COLORS["orange"],
    "adversarial": COLORS["red"],
}

BASELINE_NAME_MAP = {
    "Our CytoBridge Codex55 high": "Ours (GPT-5.5 high)",
    "Our CytoBridge Xiaomi high": "Ours (Mimo-v2.5-pro)",
    "Zhenyi Codex55 high": "Codex (GPT-5.5 high)",
    "Biomni": "Biomni (GPT-5.5 high)",
    "Zhenyi Codex55 medium": "Codex (GPT-5.5 medium)",
    "DeepSeek v4 pro + Claude Code": "Claude Code (DeepSeek-v4-pro)",
    "Xiaomi Mimo + Claude Code": "Claude Code (Mimo-v2.5-pro)",
}

METRIC_COLS = [
    "M1_velocity",
    "M2_growth",
    "M3_distribution",
    "M4_fate",
    "M5_perturbation",
    "M6_grn",
]
METRIC_SHORT = ["M1", "M2", "M3", "M4", "M5", "M6"]
METRIC_NAMES = [
    "velocity",
    "growth\ndrivers",
    "holdout\ndistribution",
    "fate\nprobability",
    "perturbation\nresponse",
    "GRN\nrecovery",
]
DIFFICULTY_ORDER = ["easy", "medium", "hard", "adversarial"]


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

    alg_overall = pd.read_csv(
        RESULT_DIR
        / "05_m2_growth_algorithm_grjwfrfm"
        / "100case_external_eval_final_20260608_0842"
        / "overall_means.csv"
    ).iloc[0]

    alg_diff = pd.read_csv(
        RESULT_DIR
        / "05_m2_growth_algorithm_grjwfrfm"
        / "100case_external_eval_final_20260608_0842"
        / "means_by_prefix.csv"
    )
    alg_diff["difficulty"] = alg_diff["prefix"].str.replace("DB100_", "", regex=False)

    return baseline, diff, alg_overall, alg_diff


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
        0.035,
        1.045,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        color=COLORS["dark"],
    )


def arrow(ax, start, end, color=COLORS["mid"], lw=1.8):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", lw=lw, color=color, shrinkA=0, shrinkB=0),
    )


def draw_nodes(ax, pts, color=COLORS["gray"], size=0.018):
    for x, y in pts:
        ax.add_patch(
            patches.Circle(
                (x, y),
                size,
                facecolor=color,
                edgecolor="white",
                lw=0.7,
                transform=ax.transAxes,
                clip_on=False,
            )
        )


def draw_panel_a(ax):
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    panel_label(ax, "a", "DynBench workflow")

    ax.text(0.135, 0.92, "Scenario design", ha="center", va="center", fontsize=10)
    ax.text(0.49, 0.92, "Simulated time course\nand model input", ha="center", va="center", fontsize=10)
    ax.text(0.82, 0.92, "Predicted dynamics\nand evaluation", ha="center", va="center", fontsize=10)

    # Scenario design: GRN.
    grn_pts = [(0.06, 0.68), (0.11, 0.76), (0.16, 0.68), (0.11, 0.60)]
    for p0, p1 in zip(grn_pts, grn_pts[1:] + grn_pts[:1]):
        ax.add_line(Line2D([p0[0], p1[0]], [p0[1], p1[1]], transform=ax.transAxes, color=COLORS["dark"], lw=1.2))
    draw_nodes(ax, grn_pts, COLORS["grid"], size=0.014)
    ax.text(0.11, 0.52, "GRN", ha="center", va="center")

    # Scenario design: fate tree.
    root = (0.25, 0.76)
    children = [(0.22, 0.66), (0.28, 0.66)]
    leaves = [(0.20, 0.58), (0.24, 0.58), (0.27, 0.58), (0.31, 0.58)]
    for child in children:
        ax.add_line(Line2D([root[0], child[0]], [root[1], child[1]], transform=ax.transAxes, color=COLORS["dark"], lw=1.1))
    for child, pair in zip(children, [leaves[:2], leaves[2:]]):
        for leaf in pair:
            ax.add_line(Line2D([child[0], leaf[0]], [child[1], leaf[1]], transform=ax.transAxes, color=COLORS["dark"], lw=1.1))
    draw_nodes(ax, [root], COLORS["blue"], size=0.014)
    draw_nodes(ax, children, COLORS["green"], size=0.012)
    draw_nodes(ax, leaves, COLORS["orange"], size=0.012)
    ax.text(0.255, 0.52, "fate tree", ha="center", va="center")

    # Scenario design: growth/death.
    x = np.linspace(0.05, 0.18, 8)
    y = 0.25 + 0.08 * np.sin(np.linspace(0.2, 2.4, 8))
    ax.plot(x, y, color=COLORS["dark"], lw=1.1, transform=ax.transAxes)
    ax.axvline(0.115, ymin=0.18, ymax=0.34, color=COLORS["dark"], lw=1.0)
    draw_nodes(ax, list(zip(x[:4], y[:4] + 0.05)), COLORS["blue"], size=0.006)
    draw_nodes(ax, list(zip(x[4:], y[4:] + 0.02)), COLORS["gray"], size=0.006)
    ax.text(0.115, 0.14, "growth/death", ha="center", va="center")

    # Scenario design: perturbation.
    ax.add_patch(
        patches.FancyArrowPatch(
            (0.245, 0.29),
            (0.315, 0.29),
            arrowstyle="-|>",
            mutation_scale=12,
            lw=1.2,
            color=COLORS["dark"],
            transform=ax.transAxes,
        )
    )
    ax.scatter([0.225, 0.325], [0.29, 0.29], s=[120, 150], c=[COLORS["blue"], COLORS["purple"]], transform=ax.transAxes, zorder=3)
    ax.text(0.237, 0.38, "perturbation", ha="left", va="center")
    ax.text(0.255, 0.14, "intervention", ha="center", va="center")

    arrow(ax, (0.34, 0.50), (0.405, 0.50))

    # Simulated input: training points and holdout gaps.
    times = np.arange(6)
    x0, width = 0.40, 0.25
    for i, t in enumerate(times):
        xpos = x0 + i * width / 5.0
        if t in [2, 4]:
            ax.add_patch(
                patches.Rectangle(
                    (xpos - 0.018, 0.34),
                    0.036,
                    0.18,
                    fill=False,
                    ec=COLORS["dark"],
                    lw=1.0,
                    ls=(0, (3, 2)),
                    transform=ax.transAxes,
                )
            )
        else:
            col = [COLORS["blue"], COLORS["green"], COLORS["orange"], COLORS["purple"]][min(i, 3) % 4]
            pts = [(xpos + dx, 0.40 + dy) for dx in [-0.012, 0, 0.012] for dy in [0, 0.04, 0.08]]
            draw_nodes(ax, pts, col, size=0.0048)
        ax.text(xpos, 0.29, str(t), ha="center", va="center", fontsize=7)
    ax.add_line(Line2D([x0 - 0.018, x0 + width + 0.018], [0.32, 0.32], transform=ax.transAxes, color=COLORS["dark"], lw=1.0))
    arrow(ax, (x0 + width + 0.008, 0.32), (x0 + width + 0.04, 0.32), lw=1.0)
    ax.text(x0 + width + 0.045, 0.32, "time", ha="left", va="center", fontsize=7)
    ax.text(0.49, 0.61, "training data", ha="center", va="center", fontsize=8)
    ax.text(0.515, 0.20, "withheld time points", ha="center", va="center", fontsize=8, color=COLORS["mid"])

    # Minimal document icon.
    doc_x, doc_y = 0.58, 0.62
    ax.add_patch(patches.Rectangle((doc_x, doc_y), 0.055, 0.15, fill=False, ec=COLORS["dark"], lw=1.2, transform=ax.transAxes))
    ax.add_patch(patches.Polygon([(doc_x + 0.04, doc_y + 0.15), (doc_x + 0.055, doc_y + 0.135), (doc_x + 0.055, doc_y + 0.15)], fill=False, ec=COLORS["dark"], lw=1.0, transform=ax.transAxes))
    for k in range(4):
        ax.add_line(Line2D([doc_x + 0.014, doc_x + 0.043], [doc_y + 0.115 - 0.025 * k] * 2, transform=ax.transAxes, color=COLORS["mid"], lw=0.8))
    ax.text(doc_x + 0.028, doc_y - 0.05, "task", ha="center", va="center")

    arrow(ax, (0.66, 0.50), (0.72, 0.50))

    # Evaluation wheel.
    cx, cy, r = 0.835, 0.48, 0.225
    for i in range(6):
        theta1 = 90 - i * 60
        theta2 = 90 - (i + 1) * 60
        ax.add_patch(
            patches.Wedge(
                (cx, cy),
                r,
                theta2,
                theta1,
                width=0.095,
                facecolor="white",
                edgecolor=COLORS["dark"],
                lw=1.0,
                transform=ax.transAxes,
            )
        )
        angle = np.deg2rad((theta1 + theta2) / 2.0)
        lx = cx + 0.17 * np.cos(angle)
        ly = cy + 0.17 * np.sin(angle)
        ax.text(lx, ly, f"{METRIC_SHORT[i]}\n{METRIC_NAMES[i]}", ha="center", va="center", fontsize=6.8)
    ax.add_patch(patches.Circle((cx, cy), 0.08, facecolor="white", edgecolor=COLORS["dark"], lw=1.0, transform=ax.transAxes))
    ax.text(cx, cy, "DynBench", ha="center", va="center", fontsize=10)


def draw_dna_icon(ax, x, y, scale=1.0):
    yy = np.linspace(-0.04, 0.04, 60) * scale + y
    xx1 = x + 0.018 * scale * np.sin(np.linspace(0, 2 * np.pi, 60))
    xx2 = x - 0.018 * scale * np.sin(np.linspace(0, 2 * np.pi, 60))
    ax.plot(xx1, yy, color=COLORS["teal"], lw=1.5, transform=ax.transAxes)
    ax.plot(xx2, yy, color=COLORS["teal"], lw=1.5, transform=ax.transAxes)
    for idx in range(8, 56, 12):
        ax.plot([xx1[idx], xx2[idx]], [yy[idx], yy[idx]], color=COLORS["teal"], lw=1.0, transform=ax.transAxes)


def draw_panel_b(ax):
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    panel_label(ax, "b", "Task composition across DB100")

    ax.text(0.02, 0.86, "Difficulty groups", ha="left", va="center", color=COLORS["mid"])
    x = 0.04
    total_w = 0.72
    segments = [("easy", 20), ("medium", 40), ("hard", 25), ("adversarial", 15)]
    for label, n in segments:
        w = total_w * n / 100
        ax.add_patch(patches.Rectangle((x, 0.72), w, 0.11, fc=DIFFICULTY_COLORS[label], ec="white", lw=0.6, transform=ax.transAxes))
        ax.text(x + w / 2, 0.775, str(n), ha="center", va="center", color="white", fontsize=11, fontweight="bold")
        ax.text(x + w / 2, 0.66, label, ha="center", va="center", fontsize=8)
        x += w
    ax.text(0.80, 0.775, "100\nscenarios", ha="center", va="center", fontsize=10, color=COLORS["dark"])

    items = [
        ("genes\n200-500", "dna"),
        ("cells\n1k-5k", "cells"),
        ("fates\n2-4", "tree"),
        ("time points\n6-12", "time"),
        ("withheld\nfuture gaps", "gap"),
        ("confounders\n+ background", "conf"),
    ]
    xs = np.linspace(0.10, 0.88, len(items))
    for xi, (label, kind) in zip(xs, items):
        if kind == "dna":
            draw_dna_icon(ax, xi, 0.39, 1.05)
        elif kind == "cells":
            rng = np.random.default_rng(2)
            pts = [(xi + rng.normal(0, 0.025), 0.39 + rng.normal(0, 0.028)) for _ in range(14)]
            draw_nodes(ax, pts, COLORS["blue"], size=0.007)
        elif kind == "tree":
            root = (xi, 0.45)
            children = [(xi - 0.035, 0.37), (xi + 0.035, 0.37)]
            leaves = [(xi - 0.05, 0.30), (xi - 0.02, 0.30), (xi + 0.02, 0.30), (xi + 0.05, 0.30)]
            for p in children:
                ax.plot([root[0], p[0]], [root[1], p[1]], color=COLORS["dark"], lw=1.0, transform=ax.transAxes)
            for parent, child_pair in zip(children, [leaves[:2], leaves[2:]]):
                for leaf in child_pair:
                    ax.plot([parent[0], leaf[0]], [parent[1], leaf[1]], color=COLORS["dark"], lw=1.0, transform=ax.transAxes)
            draw_nodes(ax, [root], COLORS["blue"], size=0.012)
            draw_nodes(ax, children, COLORS["green"], size=0.010)
            draw_nodes(ax, leaves, COLORS["orange"], size=0.009)
        elif kind == "time":
            ax.plot([xi - 0.055, xi + 0.055], [0.38, 0.38], color=COLORS["dark"], lw=1.1, transform=ax.transAxes)
            for k in range(5):
                ax.add_patch(patches.Circle((xi - 0.045 + 0.022 * k, 0.38), 0.009, fc="white", ec=COLORS["dark"], lw=1.1, transform=ax.transAxes))
        elif kind == "gap":
            ax.plot([xi - 0.055, xi + 0.055], [0.38, 0.38], color=COLORS["dark"], lw=1.1, transform=ax.transAxes)
            ax.add_patch(patches.Rectangle((xi - 0.012, 0.345), 0.034, 0.07, fill=False, ec=COLORS["dark"], lw=1.0, ls=(0, (3, 2)), transform=ax.transAxes))
        elif kind == "conf":
            pts = [(xi + dx, 0.34 + dy) for dx in [-0.035, 0, 0.035] for dy in [0, 0.04, 0.08]]
            draw_nodes(ax, pts, COLORS["red"], size=0.009)
        ax.text(xi, 0.16, label, ha="center", va="center", fontsize=8)


def style_score_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLORS["dark"])
    ax.spines["bottom"].set_color(COLORS["dark"])
    ax.grid(axis="x", color=COLORS["grid"], lw=0.6, alpha=0.8)
    ax.tick_params(colors=COLORS["dark"], length=3, width=0.8)


def draw_panel_c(ax, baseline):
    panel_label(ax, "c", "Baseline runs on DB100")
    y = np.arange(len(baseline))
    colors = [COLORS["teal"], COLORS["blue"]] + [COLORS["gray"]] * (len(baseline) - 2)
    ax.barh(y, baseline["total_score"], color=colors, edgecolor="none", height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels(baseline["label"])
    ax.invert_yaxis()
    ax.set_xlim(0, 0.76)
    ax.set_xlabel("Total score")
    style_score_axis(ax)
    ax.grid(axis="y", visible=False)
    for yi, score in zip(y, baseline["total_score"]):
        ax.text(score + 0.008, yi, f"{score:.2f}", va="center", ha="left", fontsize=7, color=COLORS["dark"])

    inset = inset_axes(ax, width="39%", height="42%", loc="lower right", borderpad=0.6)
    means = baseline[METRIC_COLS].mean()
    bar_cols = [COLORS["gray"]] * 6
    bar_cols[1] = COLORS["red"]
    inset.bar(np.arange(6), means.values, color=bar_cols, edgecolor="none", width=0.7)
    inset.set_ylim(0, 1.0)
    inset.set_xticks(np.arange(6))
    inset.set_xticklabels(METRIC_SHORT, fontsize=6)
    inset.set_yticks([0, 0.5, 1.0])
    inset.set_title("Mean by metric", fontsize=7)
    inset.grid(axis="y", color=COLORS["grid"], lw=0.5)
    for spine in ["top", "right"]:
        inset.spines[spine].set_visible(False)
    inset.tick_params(length=2, width=0.6, labelsize=6)
    inset.text(1, means.iloc[1] + 0.06, "bottleneck", ha="center", va="bottom", fontsize=6, color=COLORS["red"])


def get_baseline_diff_series(diff, agent_name):
    sub = diff[diff["agent_run"].eq(agent_name)].set_index("difficulty")
    return sub.loc[DIFFICULTY_ORDER, "total_score"].values


def draw_panel_d(ax, diff, alg_diff):
    panel_label(ax, "d", "Performance decreases with task difficulty")
    x = np.arange(len(DIFFICULTY_ORDER))

    series = [
        (
            "GRJ-VGFM",
            alg_diff.set_index("difficulty").loc[DIFFICULTY_ORDER, "total_score"].values,
            COLORS["teal"],
            2.3,
            6.0,
        ),
        (
            "Ours (GPT-5.5 high)",
            get_baseline_diff_series(diff, "Our CytoBridge Codex55 high"),
            COLORS["blue"],
            1.7,
            4.5,
        ),
        (
            "Biomni",
            get_baseline_diff_series(diff, "Biomni"),
            COLORS["gray"],
            1.4,
            4.0,
        ),
        (
            "Claude Code (DeepSeek-v4-pro)",
            get_baseline_diff_series(diff, "DeepSeek v4 pro + Claude Code"),
            COLORS["purple"],
            1.4,
            4.0,
        ),
    ]
    for label, values, color, lw, ms in series:
        ax.plot(x, values, marker="o", color=color, lw=lw, ms=ms, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(["easy", "medium", "hard", "adversarial"])
    ax.set_ylabel("Total score")
    ax.set_ylim(0.42, 0.88)
    ax.set_xlim(-0.15, 3.15)
    ax.grid(axis="y", color=COLORS["grid"], lw=0.6)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.01, 0.01), handlelength=1.8)


def draw_panel_e(ax, baseline, alg_overall):
    panel_label(ax, "e", "A targeted method resolves the growth-driver bottleneck")
    best = baseline.iloc[0]
    metrics = ["Total score", "Growth drivers (M2)"]
    base_vals = [best["total_score"], best["M2_growth"]]
    alg_vals = [alg_overall["total_score"], alg_overall["M2_growth"]]

    x = np.arange(len(metrics))
    w = 0.30
    ax.bar(x - w / 2, base_vals, width=w, color=COLORS["blue"], edgecolor="none", label="Best baseline")
    ax.bar(x + w / 2, alg_vals, width=w, color=COLORS["teal"], edgecolor="none", label="GRJ-VGFM")
    for xi, val in zip(x - w / 2, base_vals):
        ax.text(xi, val + 0.025, f"{val:.2f}", ha="center", va="bottom", fontsize=7, color=COLORS["dark"])
    for xi, val in zip(x + w / 2, alg_vals):
        ax.text(xi, val + 0.025, f"{val:.2f}", ha="center", va="bottom", fontsize=7, color=COLORS["dark"])
    for xi, delta in zip(x, np.array(alg_vals) - np.array(base_vals)):
        color = COLORS["teal"] if delta >= 0 else COLORS["red"]
        ax.text(xi, 0.96, f"+{delta:.2f}" if delta >= 0 else f"{delta:.2f}", ha="center", va="top", fontsize=8, color=color, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["Total\nscore", "Growth\ndrivers (M2)"])
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Score")
    ax.grid(axis="y", color=COLORS["grid"], lw=0.6)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    ax.text(
        0.03,
        0.78,
        "Selected example:\nrank-weighted growth\nand velocity features",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        color=COLORS["mid"],
    )


def main():
    baseline, diff, alg_overall, alg_diff = load_data()

    fig = plt.figure(figsize=(12.2, 10.4), facecolor="white")
    gs = fig.add_gridspec(
        3,
        2,
        height_ratios=[1.15, 1.0, 1.0],
        width_ratios=[1.0, 1.0],
        hspace=0.45,
        wspace=0.26,
        left=0.055,
        right=0.985,
        top=0.965,
        bottom=0.065,
    )

    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[2, 0])
    ax_e = fig.add_subplot(gs[2, 1])

    draw_panel_a(ax_a)
    draw_panel_b(ax_b)
    draw_panel_c(ax_c, baseline)
    draw_panel_d(ax_d, diff, alg_diff)
    draw_panel_e(ax_e, baseline, alg_overall)

    for suffix in ["png", "pdf", "svg"]:
        output = OUT_DIR / f"dynbench_main_figure_5panel_draft.{suffix}"
        fig.savefig(output, dpi=450 if suffix == "png" else None)
        print(output)


if __name__ == "__main__":
    main()
