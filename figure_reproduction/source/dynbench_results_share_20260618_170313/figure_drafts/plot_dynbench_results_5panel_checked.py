#!/usr/bin/env python3
"""Result-only c-g panels after checking tuned-method metric-hacking risk."""

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


def draw_panel_e_baselines(ax, diff):
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


def draw_panel_f_brgu_metrics(ax, baseline, algorithms):
    panel_label(ax, "f", "Integrated method trade-off")
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


def brgu_case_level():
    base = pd.read_csv(
        RESULT_DIR
        / "01_baseline_raw_rankings"
        / "benchmark_rankings_20260608_0142_corrected_deepseek_selected20"
        / "main100_repeat1_case_rows.csv"
    )
    base = base[base["agent_run"].eq("Our CytoBridge Codex55 high")][["scenario", "difficulty", "total_score"]]
    brgu = pd.read_csv(
        RESULT_DIR
        / "06_total_score_algorithm_brgusfm"
        / "100case_external_eval_final_locked"
        / "summary.csv"
    )[["case_id", "total_score"]]
    merged = base.merge(brgu, left_on="scenario", right_on="case_id", suffixes=("_baseline", "_brgu"))
    return merged


def draw_panel_g_brgu_difficulty(ax):
    panel_label(ax, "g", "Integrated method across DB100")
    merged = brgu_case_level()
    means = merged.groupby("difficulty")[["total_score_baseline", "total_score_brgu"]].mean().loc[DIFFICULTY_ORDER]
    wins = merged.groupby("difficulty").apply(lambda d: int((d["total_score_brgu"] > d["total_score_baseline"]).sum())).loc[DIFFICULTY_ORDER]
    n = merged.groupby("difficulty").size().loc[DIFFICULTY_ORDER]
    x = np.arange(len(DIFFICULTY_ORDER))
    ax.plot(x, means["total_score_baseline"], color=COLORS["blue"], marker="o", lw=1.8, ms=4.5, label="Best baseline")
    ax.plot(x, means["total_score_brgu"], color=COLORS["orange"], marker="o", lw=1.8, ms=4.5, label="BRGU-SFM")
    for xi, w, nn, y in zip(x, wins, n, means["total_score_brgu"]):
        ax.text(xi, y + 0.025, f"{w}/{nn}", ha="center", va="bottom", fontsize=6.8, color=COLORS["mid"])
    ax.set_xticks(x)
    ax.set_xticklabels(["easy", "medium", "hard", "adv."])
    ax.set_ylabel("Total score")
    ax.set_ylim(0.55, 0.82)
    ax.set_xlim(-0.15, 3.15)
    style_axis(ax, "y")
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 0.0), handlelength=1.6)
    ax.text(0.98, 0.95, "wins/cases", transform=ax.transAxes, ha="right", va="top", fontsize=6.8, color=COLORS["mid"])


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
    draw_panel_e_baselines(ax_e, diff)
    draw_panel_f_brgu_metrics(ax_f, baseline, algorithms)
    draw_panel_g_brgu_difficulty(ax_g)

    for suffix in ["png", "pdf", "svg"]:
        output = OUT_DIR / f"dynbench_results_5panel_checked.{suffix}"
        fig.savefig(output, dpi=450 if suffix == "png" else None)
        print(output)


if __name__ == "__main__":
    main()
