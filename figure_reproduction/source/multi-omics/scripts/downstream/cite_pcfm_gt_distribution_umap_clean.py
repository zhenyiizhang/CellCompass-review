#!/usr/bin/env python3
"""Clean UMAP GT-vs-generated distribution panel for CITE methods.

This is a less colorful replacement for the first comparison figure: matched
GT protein UMAP is shown as a grey terrain, while each algorithm is represented
only by its colored generated-density contours.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/cytobridge_mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/cytobridge_xdg_cache")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
from scipy.ndimage import gaussian_filter


sns.set_theme(style="white", context="talk")

ROOT = Path("/Users/zhenyizhang/Documents/GitHub/CytoBridge-agent-runtime-v2-20260309/multi-omics")
EMB_TABLE_DIR = ROOT / "downstream/cite_pcfm_gt_protein_comparison/tables"
METRIC_TABLE_DIR = ROOT / "downstream/cite_pcfm_rna_to_protein_gt/tables"
FIG_DIR = ROOT / "figures/downstream_cite_pcfm_rna_to_protein_gt"
MANIFEST_PATH = ROOT / "downstream/cite_pcfm_rna_to_protein_gt/rna_to_protein_gt_manifest.json"
INTERPRETATION_PATH = ROOT / "downstream/cite_pcfm_rna_to_protein_gt/interpretation.md"

METHODS = {
    "CITE-PCFM": "#E11D48",
    "CRUFM": "#2563EB",
    "balanced OT-CFM": "#F59E0B",
}
TIMES = [1.0, 2.0, 3.0, 4.0]


def density_grid(xy: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray, sigma: float) -> np.ndarray:
    hist, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=[x_edges, y_edges])
    hist = gaussian_filter(hist.T.astype(float), sigma=sigma)
    max_value = float(np.nanmax(hist))
    return hist / max_value if max_value > 0 else hist


def draw_panel(ax, obs_xy, gen_xy, x_edges, y_edges, xx, yy, color: str, pe: float) -> None:
    obs_den = density_grid(obs_xy, x_edges, y_edges, sigma=1.05)
    gen_den = density_grid(gen_xy, x_edges, y_edges, sigma=1.20)
    ax.contourf(xx, yy, obs_den, levels=[0.05, 0.12, 0.22, 0.36, 0.55, 1.01], colors=["#FFFFFF", "#F8FAFC", "#EEF2F7", "#DCE4EE", "#B7C4D5"], alpha=0.95)
    ax.contour(xx, yy, obs_den, levels=[0.16, 0.34, 0.56], colors="#7C8798", linewidths=[0.65, 0.85, 1.05], alpha=0.62)
    ax.contourf(xx, yy, gen_den, levels=[0.16, 0.32, 0.52, 0.76, 1.01], colors=[color] * 4, alpha=0.10)
    ax.contour(xx, yy, gen_den, levels=[0.18, 0.38, 0.62], colors=color, linewidths=[1.25, 1.85, 2.55])
    ax.text(
        0.965,
        0.92,
        f"PE {pe:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.6,
        color="#111827",
        bbox={"boxstyle": "round,pad=0.18,rounding_size=0.08", "facecolor": "white", "edgecolor": "#D1D9E6", "alpha": 0.95},
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#D1D9E6")
        spine.set_linewidth(0.9)


def metric_card(ax, x, title, value, color, subtitle) -> None:
    ax.add_patch(FancyBboxPatch((x, 0.10), 0.235, 0.78, boxstyle="round,pad=0.012,rounding_size=0.028", facecolor="#F8FAFC", edgecolor="#CBD5E1", linewidth=1.0))
    ax.text(x + 0.026, 0.76, title, fontsize=8.2, color="#475569", ha="left", va="top")
    ax.text(x + 0.026, 0.58, value, fontsize=14, color=color, weight="bold", ha="left", va="top")
    ax.text(x + 0.026, 0.18, subtitle, fontsize=7.2, color="#64748B", ha="left", va="bottom")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    obs = pd.read_csv(EMB_TABLE_DIR / "observed_protein_embedding_points.csv")
    gen = pd.read_csv(EMB_TABLE_DIR / "generated_protein_embedding_points.csv")
    per_time = pd.read_csv(METRIC_TABLE_DIR / "per_time.csv")
    score_raw = pd.read_csv(METRIC_TABLE_DIR / "summary_scorecard_raw.csv")

    plot_obs = obs[obs["time"].isin(TIMES)].copy()
    plot_gen = gen[gen["time"].isin(TIMES)].copy()
    all_xy = np.vstack([plot_obs[["umap1", "umap2"]].to_numpy(), plot_gen[["umap1", "umap2"]].to_numpy()])
    x0, x1 = np.nanpercentile(all_xy[:, 0], [0.4, 99.6])
    y0, y1 = np.nanpercentile(all_xy[:, 1], [0.4, 99.6])
    xlim = (x0 - 0.055 * (x1 - x0), x1 + 0.055 * (x1 - x0))
    ylim = (y0 - 0.075 * (y1 - y0), y1 + 0.075 * (y1 - y0))
    x_edges = np.linspace(xlim[0], xlim[1], 100)
    y_edges = np.linspace(ylim[0], ylim[1], 100)
    xx, yy = np.meshgrid(0.5 * (x_edges[:-1] + x_edges[1:]), 0.5 * (y_edges[:-1] + y_edges[1:]))

    fig = plt.figure(figsize=(17.6, 9.8))
    title_ax = fig.add_axes([0.05, 0.91, 0.90, 0.075])
    title_ax.axis("off")
    title_ax.text(0, 0.78, "Which algorithm lands on the GT protein landscape?", fontsize=24, weight="bold", color="#0F172A")
    title_ax.text(0, 0.25, "Grey terrain is matched observed protein UMAP; colored contours are generated protein distributions. Less color here is intentional: the readout is overlap with GT.", fontsize=10.8, color="#475569")

    methods = list(METHODS)
    grid_left, grid_bottom = 0.085, 0.305
    panel_w, panel_h = 0.158, 0.170
    x_gap, y_gap = 0.019, 0.042
    for col, time in enumerate(TIMES):
        fig.text(grid_left + col * (panel_w + x_gap) + 0.006, grid_bottom + (len(methods) - 1) * (panel_h + y_gap) + panel_h + 0.010, f"t{int(time)}", fontsize=12.4, weight="bold", color="#111827")
    for row, method in enumerate(methods):
        y = grid_bottom + (len(methods) - 1 - row) * (panel_h + y_gap)
        color = METHODS[method]
        future_pe = float(score_raw.loc[score_raw["method"].eq(method), "protein_energy"].iloc[0])
        fig.text(grid_left - 0.020, y + panel_h / 2, f"{method}\nfuture PE {future_pe:.3f}", fontsize=10.2, weight="bold", color=color, ha="right", va="center")
        for col, time in enumerate(TIMES):
            x = grid_left + col * (panel_w + x_gap)
            ax = fig.add_axes([x, y, panel_w, panel_h])
            obs_xy = plot_obs[plot_obs["time"].eq(time)][["umap1", "umap2"]].to_numpy()
            gen_xy = plot_gen[plot_gen["method"].eq(method) & plot_gen["time"].eq(time)][["umap1", "umap2"]].to_numpy()
            pe_row = per_time[per_time["method"].eq(method) & np.isclose(per_time["time"], time)]["protein_energy"]
            pe = float(pe_row.iloc[0]) if len(pe_row) else np.nan
            draw_panel(ax, obs_xy, gen_xy, x_edges, y_edges, xx, yy, color, pe)
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)

    line_ax = fig.add_axes([0.815, 0.600, 0.155, 0.255])
    for method, color in METHODS.items():
        d = per_time[per_time["method"].eq(method) & per_time["time"].isin(TIMES)].sort_values("time")
        line_ax.plot(d["time"], d["protein_energy"], marker="o", lw=2.2, ms=4.4, color=color, label=method)
    line_ax.set_yscale("log")
    line_ax.set_title("matched-time PE", loc="left", fontsize=12.2, weight="bold")
    line_ax.set_xlabel("time", fontsize=9)
    line_ax.set_ylabel("protein energy, log", fontsize=9)
    line_ax.tick_params(labelsize=8)
    line_ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    line_ax.spines[["top", "right"]].set_visible(False)

    score = score_raw.copy()
    metric_cols = ["protein_energy", "covariance_error", "nearest_distance", "variance_error"]
    for col in metric_cols:
        lo, hi = float(score[col].min()), float(score[col].max())
        score[f"{col}_norm"] = 0.0 if hi == lo else (score[col] - lo) / (hi - lo)
    score["aggregate_mismatch"] = score[[f"{col}_norm" for col in metric_cols]].mean(axis=1)
    score = score.sort_values("aggregate_mismatch")
    bar_ax = fig.add_axes([0.815, 0.335, 0.155, 0.205])
    y_pos = np.arange(len(score))
    bar_ax.barh(y_pos, score["aggregate_mismatch"], color=[METHODS[m] for m in score["method"]], alpha=0.86, height=0.58)
    bar_ax.set_yticks(y_pos)
    bar_ax.set_yticklabels(score["method"], fontsize=8)
    bar_ax.invert_yaxis()
    bar_ax.set_xlim(0, 1)
    bar_ax.set_xlabel("normalized error", fontsize=9)
    bar_ax.set_title("future-time aggregate", loc="left", fontsize=12.2, weight="bold")
    bar_ax.tick_params(axis="x", labelsize=8)
    bar_ax.spines[["top", "right", "left"]].set_visible(False)
    for i, (_, row) in enumerate(score.iterrows()):
        bar_ax.text(row["aggregate_mismatch"] + 0.025, i, f"{row['aggregate_mismatch']:.2f}", va="center", fontsize=8.3, color="#111827", weight="bold")

    card_ax = fig.add_axes([0.085, 0.055, 0.885, 0.145])
    card_ax.axis("off")
    pcfm = score_raw[score_raw["method"].eq("CITE-PCFM")].iloc[0]
    crufm = score_raw[score_raw["method"].eq("CRUFM")].iloc[0]
    bot = score_raw[score_raw["method"].eq("balanced OT-CFM")].iloc[0]
    best = score.iloc[0]
    metric_card(card_ax, 0.000, "Best overall", str(best["method"]), METHODS[str(best["method"])], f"aggregate normalized error {best['aggregate_mismatch']:.2f}")
    metric_card(card_ax, 0.255, "Future PE", f"{pcfm['protein_energy']:.3f}", METHODS["CITE-PCFM"], f"CRUFM {crufm['protein_energy']:.3f}; balanced {bot['protein_energy']:.3f}")
    metric_card(card_ax, 0.510, "Covariance error", f"{pcfm['covariance_error']:.3f}", METHODS["CITE-PCFM"], f"CRUFM {crufm['covariance_error']:.3f}; balanced {bot['covariance_error']:.3f}")
    metric_card(card_ax, 0.765, "Variance error", f"{pcfm['variance_error']:.3f}", METHODS["CITE-PCFM"], f"CRUFM {crufm['variance_error']:.3f}; balanced {bot['variance_error']:.3f}")

    fig.text(grid_left, 0.240, "GT density contours are grey; generated density contours are colored by method.", fontsize=8.6, color="#64748B")
    out = FIG_DIR / "fig01d_clean_umap_gt_distribution_panel.png"
    fig.savefig(out, dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text())
        figures = manifest.setdefault("figures", [])
        for path in [str(out), str(out.with_suffix(".pdf"))]:
            if path not in figures:
                figures.insert(0, path)
        MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if INTERPRETATION_PATH.exists():
        text = INTERPRETATION_PATH.read_text()
        line = "- Fig. 01d is the clean preferred result panel: matched GT is reduced to a grey UMAP terrain and each algorithm is judged by whether its colored density contours cover that terrain.\n"
        if "Fig. 01d is the clean preferred result panel" not in text:
            text = text.replace("## Figure guide\n\n", "## Figure guide\n\n" + line)
            INTERPRETATION_PATH.write_text(text)
    print(json.dumps({"figure": str(out), "pdf": str(out.with_suffix(".pdf"))}, indent=2))


if __name__ == "__main__":
    main()
