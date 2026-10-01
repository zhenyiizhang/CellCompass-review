#!/usr/bin/env python3
"""Result-only c-g panels with method-audit and selected20 stability panels."""

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
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

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


METHOD_LABELS = {
    "GRJ-VGFM": "GRJ-VGFM\n(growth-targeted)",
    "SPFT-CFM": "SPFT-CFM\n(fate-targeted)",
    "BRGUSFM": "BRGU-SFM\n(integrated)",
}


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


def algorithm_rows():
    metrics = ["total_score", *METRIC_COLS]
    paths = {
        "GRJ-VGFM": RESULT_DIR / "05_m2_growth_algorithm_grjwfrfm" / "100case_external_eval_final_20260608_0842" / "overall_means.csv",
        "SPFT-CFM": RESULT_DIR / "07_extra_m4_fate_algorithm_spftcfm" / "100case_external_eval_final" / "summary.csv",
        "BRGUSFM": RESULT_DIR / "06_total_score_algorithm_brgusfm" / "100case_external_eval_final_locked" / "summary.csv",
    }
    rows = []
    for name, path in paths.items():
        df = pd.read_csv(path)
        vals = df[metrics].mean() if len(df) > 1 else df.iloc[0][metrics]
        row = {"method": name}
        row.update({m: float(vals[m]) for m in metrics})
        rows.append(row)
    return pd.DataFrame(rows)


def draw_panel_f_delta_audit(ax, baseline):
    panel_label(ax, "f", "Component-level method audit")
    best = baseline.iloc[0]
    alg = algorithm_rows().set_index("method")
    order = ["GRJ-VGFM", "SPFT-CFM", "BRGUSFM"]
    delta = alg.loc[order, METRIC_COLS].to_numpy() - best[METRIC_COLS].astype(float).to_numpy()[None, :]

    cmap = LinearSegmentedColormap.from_list("delta", ["#D65F5F", "#F7F9FB", COLORS["teal"]])
    norm = TwoSlopeNorm(vmin=-0.25, vcenter=0.0, vmax=0.70)
    im = ax.imshow(delta, aspect="auto", cmap=cmap, norm=norm)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([METHOD_LABELS[x] for x in order], linespacing=0.95)
    ax.set_xticks(np.arange(len(METRIC_COLS)))
    ax.set_xticklabels(["velocity", "growth", "holdout", "fate", "perturb.", "GRN"], rotation=35, ha="right")
    ax.tick_params(length=0)
    for i in range(delta.shape[0]):
        for j in range(delta.shape[1]):
            val = delta[i, j]
            txt_color = "white" if abs(val) > 0.32 else COLORS["dark"]
            ax.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=6.3, color=txt_color)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cax = ax.inset_axes([0.90, 0.05, 0.035, 0.34])
    cb = plt.colorbar(im, cax=cax, ticks=[-0.2, 0.0, 0.4, 0.7])
    cb.ax.tick_params(labelsize=5.8, length=2)
    cb.outline.set_linewidth(0.4)


def selected20_repeat_rows():
    rows = []
    attempt = pd.read_csv(
        RESULT_DIR
        / "03_our_agent_combined_summaries"
        / "our_agent_cytobridge_results_20260608_0050"
        / "our_agent_provider_compare_attempt_rows.csv"
    )
    sel = attempt[(attempt["scope"].eq("selected20_3repeat_history")) & (attempt["is_ok"].eq(True))].copy()
    name_map = {
        "codex55_high": "Ours\n(GPT-5.5 high)",
        "xiaomi_high": "Ours\n(Mimo-v2.5-pro)",
    }
    for provider, label in name_map.items():
        vals = sel[sel["provider_name"].eq(provider)]["total_score"].dropna().astype(float).to_numpy()
        if vals.size:
            rows.append({"method": label, "mean": float(vals.mean()), "sd": float(vals.std(ddof=1)), "n": int(vals.size)})

    deep = pd.read_csv(
        RESULT_DIR
        / "01_baseline_raw_rankings"
        / "benchmark_rankings_20260608_0142_corrected_deepseek_selected20"
        / "deepseek_cc_selected20_3repeat_corrected_case_rows.csv"
    )
    vals = deep[deep["status"].eq("ok")]["total_score"].dropna().astype(float).to_numpy()
    if vals.size:
        rows.append({"method": "Claude Code\n(DeepSeek)", "mean": float(vals.mean()), "sd": float(vals.std(ddof=1)), "n": int(vals.size)})

    return pd.DataFrame(rows)


def draw_panel_g_selected20(ax):
    panel_label(ax, "g", "Selected20 repeat stability")
    stats = selected20_repeat_rows()
    colors = [COLORS["blue"], COLORS["teal"], COLORS["purple"]]
    x = np.arange(len(stats))
    ax.bar(x, stats["mean"], yerr=stats["sd"], capsize=3.0, color=colors[: len(stats)], edgecolor="none", width=0.62)
    for xi, mean, sd, n in zip(x, stats["mean"], stats["sd"], stats["n"]):
        ax.text(xi, mean + sd + 0.025, f"n={n}", ha="center", va="bottom", fontsize=6.6, color=COLORS["mid"])
    ax.set_xticks(x)
    ax.set_xticklabels(stats["method"], linespacing=0.92)
    ax.set_ylabel("Total score")
    ax.set_ylim(0.35, 0.78)
    style_axis(ax, "y")
    ax.grid(axis="x", visible=False)
    ax.text(
        0.98,
        0.95,
        "mean +/- s.d.",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.8,
        color=COLORS["mid"],
    )


def main():
    baseline, diff, _alg_diff, _algorithms = load_data()

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
    draw_panel_f_delta_audit(ax_f, baseline)
    draw_panel_g_selected20(ax_g)

    for suffix in ["png", "pdf", "svg"]:
        output = OUT_DIR / f"dynbench_results_5panel_audit.{suffix}"
        fig.savefig(output, dpi=450 if suffix == "png" else None)
        print(output)


if __name__ == "__main__":
    main()
