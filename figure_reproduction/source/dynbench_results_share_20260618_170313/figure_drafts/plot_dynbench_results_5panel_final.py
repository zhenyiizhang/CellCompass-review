#!/usr/bin/env python3
"""Final result-only c-g draft: BRGU-SFM result plus selected20 stability."""

from pathlib import Path
import os
import zipfile

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
from scipy.stats import wilcoxon

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
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


SELECTED20_LABELS = {
    "Our CytoBridge Codex55 high": "Ours\n(GPT-5.5 high)",
    "Our CytoBridge Xiaomi high": "Ours\n(Mimo-v2.5-pro)",
    "Zhenyi Codex55 high": "Codex\n(GPT-5.5 high)",
    "DeepSeek v4 pro + Claude Code": "Claude Code\n(DeepSeek)",
    "Biomni": "Biomni\n(GPT-5.5 high)",
    "Xiaomi Mimo + Claude Code": "Claude Code\n(Mimo-v2.5-pro)",
}


def _bool_mask(series):
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def selected20_raw_scores():
    raw = []

    attempt = pd.read_csv(
        RESULT_DIR
        / "03_our_agent_combined_summaries"
        / "our_agent_cytobridge_results_20260608_0050"
        / "our_agent_provider_compare_attempt_rows.csv"
    )
    sel = attempt[(attempt["scope"].eq("selected20_3repeat_history")) & (attempt["is_ok"].eq(True))].copy()
    for provider, agent in [
        ("codex55_high", "Our CytoBridge Codex55 high"),
        ("xiaomi_high", "Our CytoBridge Xiaomi high"),
    ]:
        vals = sel[sel["provider_name"].eq(provider)][["scenario", "total_score"]].copy()
        vals["agent_run"] = agent
        raw.append(vals)

    with zipfile.ZipFile(RESULT_DIR / "zhenyi_dynbench_results_20260607.zip") as zf:
        with zf.open(
            "zhenyi_dynbench_results_20260607/selected20_repeats/selected20_repeatability_summary.csv"
        ) as f:
            zhenyi = pd.read_csv(f)
    zhenyi_vals = zhenyi.melt(
        id_vars="scenario",
        value_vars=["repeat1", "repeat2", "repeat3"],
        value_name="total_score",
    )[["scenario", "total_score"]]
    zhenyi_vals["agent_run"] = "Zhenyi Codex55 high"
    raw.append(zhenyi_vals)

    deep = pd.read_csv(
        RESULT_DIR
        / "01_baseline_raw_rankings"
        / "benchmark_rankings_20260608_0142_corrected_deepseek_selected20"
        / "deepseek_cc_selected20_3repeat_corrected_case_rows.csv"
    )
    deep_vals = deep[deep["status"].eq("ok")][["scenario", "total_score"]].copy()
    deep_vals["agent_run"] = "DeepSeek v4 pro + Claude Code"
    raw.append(deep_vals)

    biomni = pd.read_csv(RESULT_DIR / "biomni_case_scores.csv")
    counts = biomni.groupby("scenario").size()
    repeated_scenarios = counts[counts.eq(3)].index
    biomni_repeats = biomni[biomni["scenario"].isin(repeated_scenarios)].copy()
    biomni_vals = biomni_repeats[biomni_repeats["success"].eq(True)][["scenario", "total_score"]].copy()
    biomni_vals["agent_run"] = "Biomni"
    raw.append(biomni_vals)

    with zipfile.ZipFile(RESULT_DIR / "mimo_cc_20260607.zip") as zf:
        with zf.open("mimo_cc_20260607/selected20_repeats/selected20_repeat_details_all.csv") as f:
            mimo_cc = pd.read_csv(f)
    mimo_mask = mimo_cc["status"].eq("ok")
    if "include_in_stats" in mimo_cc.columns:
        mimo_mask &= _bool_mask(mimo_cc["include_in_stats"])
    mimo_vals = mimo_cc[mimo_mask][["scenario", "total_score"]].copy()
    mimo_vals["agent_run"] = "Xiaomi Mimo + Claude Code"
    raw.append(mimo_vals)

    out = pd.concat(raw, ignore_index=True)
    out["total_score"] = pd.to_numeric(out["total_score"], errors="coerce")
    return out.dropna(subset=["scenario", "total_score", "agent_run"])


def selected20_pairwise_tests():
    raw = selected20_raw_scores()
    scenario_mean = raw.groupby(["agent_run", "scenario"], as_index=False)["total_score"].mean()
    wide = scenario_mean.pivot(index="scenario", columns="agent_run", values="total_score")
    ordered = wide.mean().sort_values(ascending=False).index.tolist()
    top = ordered[0]
    rows = []
    for other in ordered[1:]:
        paired = wide[[top, other]].dropna()
        diff = paired[top] - paired[other]
        rows.append(
            {
                "top_agent_run": top,
                "other_agent_run": other,
                "n_scenarios": int(paired.shape[0]),
                "mean_difference": float(diff.mean()),
                "median_difference": float(np.median(diff)),
                "wilcoxon_two_sided_p": float(
                    wilcoxon(paired[top], paired[other], alternative="two-sided", zero_method="wilcox").pvalue
                ),
            }
        )

    tests = pd.DataFrame(rows)
    order = np.argsort(tests["wilcoxon_two_sided_p"].to_numpy())
    adjusted = np.empty(len(tests))
    running = 0.0
    for rank, idx in enumerate(order):
        value = (len(tests) - rank) * tests.loc[idx, "wilcoxon_two_sided_p"]
        running = max(running, value)
        adjusted[idx] = min(running, 1.0)
    tests["holm_adjusted_p"] = adjusted
    return tests


def format_pvalue(pval):
    if pval < 0.001:
        return f"{pval:.1e}".replace("e-0", "e-").replace("e+0", "e+")
    return f"{pval:.3f}"


def significance_stars(pval):
    if pval < 0.001:
        return "***"
    if pval < 0.01:
        return "**"
    if pval < 0.05:
        return "*"
    return "n.s."


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
        ("Ours GPT-5.5 high", "Our CytoBridge Codex55 high", COLORS["blue"], "o", "-", 1.8, 1.00),
        ("Ours Mimo", "Our CytoBridge Xiaomi high", COLORS["teal"], "o", "-", 1.8, 1.00),
        ("Codex high", "Zhenyi Codex55 high", "#7C858C", "s", "-", 1.25, 0.85),
        ("Biomni", "Biomni", "#A0A6AB", "^", "-", 1.25, 0.85),
        ("Codex medium", "Zhenyi Codex55 medium", "#B1B6BA", "D", "--", 1.05, 0.82),
        ("Claude DeepSeek", "DeepSeek v4 pro + Claude Code", COLORS["purple"], "v", "-", 1.25, 0.90),
        ("Claude Mimo", "Xiaomi Mimo + Claude Code", "#8C9399", "P", "--", 1.05, 0.82),
    ]
    for label, source, color, marker, linestyle, linewidth, alpha in series:
        values = get_baseline_diff_series(diff, source)
        ax.plot(
            x,
            values,
            marker=marker,
            linestyle=linestyle,
            color=color,
            lw=linewidth,
            ms=3.5,
            alpha=alpha,
            label=label,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(["easy", "medium", "hard", "adv."])
    ax.set_ylabel("Total score")
    ax.set_ylim(0.42, 0.83)
    ax.set_xlim(-0.15, 3.15)
    style_axis(ax, "y")
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(-0.01, -0.01), handlelength=1.4, ncol=2, columnspacing=0.8)


def draw_panel_f_brgu_metrics(ax, baseline, algorithms):
    panel_label(ax, "f", "BRGU-SFM overall and components")
    best = baseline.iloc[0]
    brgu = algorithms[algorithms["method"].eq("BRGUSFM")].iloc[0]
    cols = ["total_score", *METRIC_COLS]
    x = np.arange(len(cols))
    best_vals = [best[m] for m in cols]
    brgu_vals = [brgu[m] for m in cols]
    ax.plot(x, best_vals, color=COLORS["blue"], marker="o", lw=1.8, ms=4.5, label="Best baseline")
    ax.plot(x, brgu_vals, color=COLORS["orange"], marker="o", lw=1.8, ms=4.5, label="BRGU-SFM")
    ax.scatter([0], [best_vals[0]], color=COLORS["blue"], s=42, zorder=4)
    ax.scatter([0], [brgu_vals[0]], color=COLORS["orange"], s=42, zorder=4)
    ax.axvline(0.5, color=COLORS["grid"], lw=0.8)
    ax.text(
        0,
        max(best_vals[0], brgu_vals[0]) + 0.055,
        f"+{brgu_vals[0] - best_vals[0]:.3f}",
        ha="center",
        va="bottom",
        fontsize=6.8,
        color=COLORS["orange"],
    )
    ax.set_xticks(x)
    ax.set_xticklabels(["overall", "velocity", "growth", "holdout", "fate", "perturb.", "GRN"], rotation=35, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0.2, 1.0)
    style_axis(ax, "y")
    ax.legend(frameon=False, loc="lower left")


def selected20_stats():
    rank = pd.read_csv(
        RESULT_DIR
        / "01_baseline_raw_rankings"
        / "benchmark_rankings_20260608_0142_corrected_deepseek_selected20"
        / "selected20_repeat_ranking.csv"
    )
    rank["label"] = rank["agent_run"].map(SELECTED20_LABELS)
    rank["sd"] = np.nan
    rank["raw_n"] = np.nan

    raw = selected20_raw_scores()
    raw_map = {agent: vals["total_score"].astype(float) for agent, vals in raw.groupby("agent_run")}

    for agent, vals in raw_map.items():
        mask = rank["agent_run"].eq(agent)
        if vals.size:
            rank.loc[mask, "sd"] = float(vals.std(ddof=1))
            rank.loc[mask, "raw_n"] = int(vals.size)
    return rank


def draw_panel_g_selected20(ax):
    panel_label(ax, "g", "Selected20 repeat stability")
    stats = selected20_stats().copy()
    stats = stats.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(stats))
    colors = [COLORS["blue"], COLORS["teal"], COLORS["gray"], COLORS["purple"], "#B0B5BA", "#8A8F94"]
    colors = list(reversed(colors[: len(stats)]))
    ax.barh(y, stats["total_score"], color=colors, edgecolor="none", height=0.62)

    for yi, mean, sd, raw_n in zip(y, stats["total_score"], stats["sd"], stats["raw_n"]):
        if np.isfinite(sd):
            ax.errorbar(mean, yi, xerr=sd, color=COLORS["dark"], lw=1.0, capsize=3.0, fmt="none")
        ax.text(mean + 0.012, yi, f"{mean:.2f}", ha="left", va="center", fontsize=6.5, color=COLORS["dark"])

    tests = selected20_pairwise_tests()
    y_top = int(stats.index[stats["agent_run"].eq(tests.iloc[0]["top_agent_run"])][0])
    y_others = [
        int(stats.index[stats["agent_run"].eq(agent)][0])
        for agent in tests["other_agent_run"]
    ]
    y_low = min(y_others)
    bracket_x = 0.82
    tick = 0.018
    ax.plot([bracket_x, bracket_x], [y_low, y_top], color=COLORS["dark"], lw=0.9, clip_on=False)
    ax.plot([bracket_x - tick, bracket_x], [y_low, y_low], color=COLORS["dark"], lw=0.9, clip_on=False)
    ax.plot([bracket_x - tick, bracket_x], [y_top, y_top], color=COLORS["dark"], lw=0.9, clip_on=False)
    for yi in y_others:
        ax.plot([bracket_x - tick * 0.65, bracket_x], [yi, yi], color=COLORS["dark"], lw=0.65, clip_on=False)
    p_for_group = tests["holm_adjusted_p"].max()
    ax.text(
        bracket_x + 0.012,
        (y_top + y_low) / 2,
        f"vs all others\n{significance_stars(p_for_group)}",
        ha="left",
        va="center",
        fontsize=6.6,
        color=COLORS["dark"],
    )

    ax.set_yticks(y)
    ax.set_yticklabels(stats["label"], linespacing=0.90)
    ax.set_xlabel("Total score, mean +/- s.d. (20 scenarios x 3 runs)")
    ax.set_xlim(0.0, 0.98)
    style_axis(ax, "x")
    ax.grid(axis="y", visible=False)


def main():
    baseline, diff, _alg_diff, algorithms = load_data()
    sig_output = OUT_DIR / "selected20_pairwise_significance.csv"
    selected20_pairwise_tests().to_csv(sig_output, index=False)

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
    draw_panel_g_selected20(ax_g)

    for suffix in ["png", "pdf", "svg"]:
        output = OUT_DIR / f"dynbench_results_5panel_final.{suffix}"
        fig.savefig(output, dpi=450 if suffix == "png" else None)
        print(output)
    print(sig_output)


if __name__ == "__main__":
    main()
