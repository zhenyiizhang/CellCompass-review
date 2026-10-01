#!/usr/bin/env python3
"""Protein-UMAP GT-versus-generated distribution panel for CITE methods."""

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
from matplotlib.lines import Line2D
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
    "CITE-PCFM": {"color": "#E11D48", "label": "CITE-PCFM"},
    "CRUFM": {"color": "#2563EB", "label": "CRUFM"},
    "balanced OT-CFM": {"color": "#F59E0B", "label": "balanced OT-CFM"},
}
TIMES = [1.0, 2.0, 3.0, 4.0]
PHENO_COLORS = {
    "HSC": "#2F855A",
    "MasP": "#D62728",
    "EryP": "#F28E2B",
    "MoP": "#8C564B",
    "MkP": "#7B61B3",
    "NeuP": "#E377C2",
    "BP": "#1F77B4",
}


def density_grid(xy: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray, sigma: float = 1.15) -> np.ndarray:
    hist, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=[x_edges, y_edges])
    hist = gaussian_filter(hist.T.astype(float), sigma=sigma)
    max_value = float(np.nanmax(hist))
    return hist / max_value if max_value > 0 else hist


def metric_card(ax, x: float, y: float, w: float, h: float, title: str, value: str, color: str, subtitle: str) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.028",
            facecolor="#F8FAFC",
            edgecolor="#CBD5E1",
            linewidth=1.0,
        )
    )
    ax.text(x + 0.028, y + h - 0.08, title, fontsize=8.4, color="#475569", ha="left", va="top")
    ax.text(x + 0.028, y + h - 0.21, value, fontsize=14.2, color=color, weight="bold", ha="left", va="top")
    ax.text(x + 0.028, y + 0.055, subtitle, fontsize=7.4, color="#64748B", ha="left", va="bottom")


def draw_panel(ax, obs_t: pd.DataFrame, gen_t: pd.DataFrame, x_edges, y_edges, xx, yy, method_color: str, pe_value: float) -> None:
    obs_xy = obs_t[["umap1", "umap2"]].to_numpy()
    gen_xy = gen_t[["umap1", "umap2"]].to_numpy()
    obs_den = density_grid(obs_xy, x_edges, y_edges, sigma=1.05)
    gen_den = density_grid(gen_xy, x_edges, y_edges, sigma=1.20)

    ax.contour(xx, yy, obs_den, levels=[0.13, 0.27, 0.46, 0.68], colors="#64748B", linewidths=[0.5, 0.65, 0.8, 1.0], alpha=0.50)
    for pheno, color in PHENO_COLORS.items():
        sub = obs_t[obs_t["cell_type"].eq(pheno)]
        if len(sub):
            ax.scatter(sub["umap1"], sub["umap2"], s=5.6, color=color, alpha=0.34, linewidths=0, rasterized=True)
    ax.contourf(xx, yy, gen_den, levels=[0.10, 0.22, 0.38, 0.58, 0.80, 1.01], colors=[method_color] * 5, alpha=0.08)
    ax.contour(xx, yy, gen_den, levels=[0.18, 0.36, 0.58], colors=method_color, linewidths=[1.15, 1.75, 2.35])
    ax.scatter(gen_t["umap1"], gen_t["umap2"], s=4.2, color=method_color, alpha=0.12, linewidths=0, rasterized=True)
    ax.text(
        0.98,
        0.94,
        f"PE {pe_value:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.8,
        color="#0F172A",
        bbox={"boxstyle": "round,pad=0.20,rounding_size=0.08", "facecolor": "white", "edgecolor": "#CBD5E1", "alpha": 0.94},
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#CBD5E1")
        spine.set_linewidth(0.9)


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
    x_pad = 0.055 * (x1 - x0)
    y_pad = 0.075 * (y1 - y0)
    xlim = (x0 - x_pad, x1 + x_pad)
    ylim = (y0 - y_pad, y1 + y_pad)
    x_edges = np.linspace(xlim[0], xlim[1], 100)
    y_edges = np.linspace(ylim[0], ylim[1], 100)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    xx, yy = np.meshgrid(x_centers, y_centers)

    fig = plt.figure(figsize=(18.8, 10.8))
    title_ax = fig.add_axes([0.045, 0.925, 0.91, 0.06])
    title_ax.axis("off")
    title_ax.text(0, 0.75, "Generated protein distributions on matched GT UMAP", fontsize=24, weight="bold", color="#0F172A", ha="left")
    title_ax.text(
        0,
        0.20,
        "Each panel shows the matched observed protein UMAP terrain as phenotype-colored GT cells; colored contours/points show one algorithm's generated protein distribution.",
        fontsize=10.7,
        color="#475569",
        ha="left",
    )
    title_ax.text(1.0, 0.20, "UMAP is fit in protein space; PE is computed in protein PCA space", fontsize=8.8, color="#64748B", ha="right")

    methods = list(METHODS.keys())
    grid_left, grid_bottom = 0.080, 0.320
    panel_w, panel_h = 0.155, 0.168
    x_gap, y_gap = 0.018, 0.040
    col_label_y = grid_bottom + (len(methods) - 1) * (panel_h + y_gap) + panel_h + 0.007
    for col, time in enumerate(TIMES):
        fig.text(grid_left + col * (panel_w + x_gap) + 0.005, col_label_y, f"t{int(time)}", fontsize=12.5, weight="bold", color="#0F172A")

    for row, method in enumerate(methods):
        y = grid_bottom + (len(methods) - 1 - row) * (panel_h + y_gap)
        color = METHODS[method]["color"]
        future_pe = score_raw.loc[score_raw["method"].eq(method), "protein_energy"].iloc[0]
        fig.text(grid_left - 0.018, y + panel_h / 2, f"{method}\nfuture PE {future_pe:.3f}", fontsize=10, weight="bold", color=color, ha="right", va="center")
        for col, time in enumerate(TIMES):
            x = grid_left + col * (panel_w + x_gap)
            ax = fig.add_axes([x, y, panel_w, panel_h])
            obs_t = plot_obs[plot_obs["time"].eq(time)]
            gen_t = plot_gen[plot_gen["method"].eq(method) & plot_gen["time"].eq(time)]
            metric = per_time[per_time["method"].eq(method) & np.isclose(per_time["time"], time)]["protein_energy"]
            pe = float(metric.iloc[0]) if len(metric) else np.nan
            draw_panel(ax, obs_t, gen_t, x_edges, y_edges, xx, yy, color, pe)
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)

    line_ax = fig.add_axes([0.825, 0.615, 0.145, 0.260])
    for method, spec in METHODS.items():
        d = per_time[per_time["method"].eq(method) & per_time["time"].isin(TIMES)].sort_values("time")
        line_ax.plot(d["time"], d["protein_energy"], marker="o", lw=2.2, ms=4.6, color=spec["color"], label=method)
    line_ax.set_yscale("log")
    line_ax.set_title("protein energy by time", loc="left", fontsize=12.2, weight="bold")
    line_ax.set_xlabel("time", fontsize=9)
    line_ax.set_ylabel("PE, log scale", fontsize=9)
    line_ax.tick_params(labelsize=8)
    line_ax.spines[["top", "right"]].set_visible(False)
    line_ax.legend(frameon=False, fontsize=7.6, loc="upper left")

    score = score_raw.copy()
    metric_cols = ["protein_energy", "covariance_error", "nearest_distance", "variance_error"]
    for col in metric_cols:
        lo, hi = float(score[col].min()), float(score[col].max())
        score[f"{col}_norm"] = 0.0 if hi == lo else (score[col] - lo) / (hi - lo)
    score["aggregate_mismatch"] = score[[f"{col}_norm" for col in metric_cols]].mean(axis=1)
    score = score.sort_values("aggregate_mismatch", ascending=True)

    bar_ax = fig.add_axes([0.825, 0.335, 0.145, 0.205])
    y_pos = np.arange(score.shape[0])
    colors = [METHODS[m]["color"] for m in score["method"]]
    bar_ax.barh(y_pos, score["aggregate_mismatch"], color=colors, alpha=0.86, height=0.58)
    bar_ax.set_yticks(y_pos)
    bar_ax.set_yticklabels(score["method"], fontsize=8)
    bar_ax.invert_yaxis()
    bar_ax.set_xlim(0, 1.0)
    bar_ax.set_xlabel("normalized error", fontsize=9)
    bar_ax.set_title("future-time aggregate", loc="left", fontsize=12.2, weight="bold")
    bar_ax.tick_params(axis="x", labelsize=8)
    bar_ax.spines[["top", "right", "left"]].set_visible(False)
    for y_i, (_, row_data) in enumerate(score.iterrows()):
        bar_ax.text(row_data["aggregate_mismatch"] + 0.025, y_i, f"{row_data['aggregate_mismatch']:.2f}", va="center", fontsize=8.3, color="#0F172A", weight="bold")

    legend_ax = fig.add_axes([0.080, 0.228, 0.665, 0.045])
    legend_ax.axis("off")
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markeredgecolor="none", markersize=6, alpha=0.75, label=pheno)
        for pheno, color in PHENO_COLORS.items()
    ]
    handles.extend(
        [
            Line2D([0], [0], color="#64748B", lw=1.2, label="GT density contour"),
            Line2D([0], [0], color="#0F172A", lw=2.3, label="generated density contour"),
        ]
    )
    legend_ax.legend(handles=handles, ncol=9, frameon=False, fontsize=7.7, loc="center left", handlelength=1.4, columnspacing=1.1)

    card_ax = fig.add_axes([0.080, 0.055, 0.890, 0.145])
    card_ax.axis("off")
    best = score.iloc[0]
    metric_card(card_ax, 0.000, 0.06, 0.235, 0.78, "Best overall distribution match", str(best["method"]), METHODS[str(best["method"])]["color"], f"aggregate normalized error {best['aggregate_mismatch']:.2f}")
    pcfm = score_raw[score_raw["method"].eq("CITE-PCFM")].iloc[0]
    crufm = score_raw[score_raw["method"].eq("CRUFM")].iloc[0]
    bot = score_raw[score_raw["method"].eq("balanced OT-CFM")].iloc[0]
    metric_card(card_ax, 0.255, 0.06, 0.215, 0.78, "Future PE", f"{pcfm['protein_energy']:.3f}", METHODS["CITE-PCFM"]["color"], f"CRUFM {crufm['protein_energy']:.3f}; balanced {bot['protein_energy']:.3f}")
    metric_card(card_ax, 0.490, 0.06, 0.215, 0.78, "RNA-protein covariance", f"{pcfm['covariance_error']:.3f}", METHODS["CITE-PCFM"]["color"], f"CRUFM {crufm['covariance_error']:.3f}; balanced {bot['covariance_error']:.3f}")
    metric_card(card_ax, 0.725, 0.06, 0.255, 0.78, "Diversity preservation", f"{pcfm['variance_error']:.3f}", METHODS["CITE-PCFM"]["color"], f"CRUFM {crufm['variance_error']:.3f}; balanced {bot['variance_error']:.3f}")

    out = FIG_DIR / "fig01c_umap_distribution_gt_metric_panel.png"
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
        line = "- Fig. 01c is the preferred UMAP main panel: matched GT protein UMAP cells remain visible in every panel, with algorithm-generated distributions overlaid as colored contours and points.\n"
        if "Fig. 01c is the preferred UMAP main panel" not in text:
            text = text.replace("## Figure guide\n\n", "## Figure guide\n\n" + line)
            INTERPRETATION_PATH.write_text(text)

    print(json.dumps({"figure": str(out), "pdf": str(out.with_suffix(".pdf"))}, indent=2))


if __name__ == "__main__":
    main()
