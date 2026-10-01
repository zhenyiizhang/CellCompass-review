#!/usr/bin/env python3
"""Fancy GT-versus-generated protein distribution panel for CITE methods."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/cytobridge_mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/cytobridge_xdg_cache")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
from scipy.ndimage import gaussian_filter


sns.set_theme(style="white", context="talk")

ROOT = Path("/Users/zhenyizhang/Documents/GitHub/CytoBridge-agent-runtime-v2-20260309/multi-omics")
DATA_PATH = Path("/Users/zhenyizhang/.cellcompass/algorithm_benchmarks/datasets/gse305370_citeseq_stage2_10k/data.h5ad")
FIG_DIR = ROOT / "figures/downstream_cite_pcfm_rna_to_protein_gt"
TABLE_DIR = ROOT / "downstream/cite_pcfm_rna_to_protein_gt/tables"
MANIFEST_PATH = ROOT / "downstream/cite_pcfm_rna_to_protein_gt/rna_to_protein_gt_manifest.json"

RNA_DIM = 50
PROTEIN_DIM = 30
TIMES = [0.0, 1.0, 2.0, 3.0, 4.0]
METHODS = {
    "CITE-PCFM": {
        "trajectory": ROOT / "training_runs/20260526T174204321821Z__final__cite_paired_coord_fm__campaign_20260-trial_20260526-gse305370_ci/artifacts/evaluation_trajectory.npz",
        "color": "#E11D48",
        "label": "CITE-PCFM",
    },
    "CRUFM": {
        "trajectory": ROOT / "training_runs/20260525T220649168798Z__pilot__crufm__campaign_20260-stage2_c-baseline-crufm-gse305370_ci/artifacts/evaluation_trajectory.npz",
        "color": "#2563EB",
        "label": "CRUFM",
    },
    "balanced OT-CFM": {
        "trajectory": ROOT / "training_runs/20260525T215826526439Z__pilot__balanced_ot_cfm__campaign_20260-stage2_c-baseline-balanced_ot_-gse305370_ci/artifacts/evaluation_trajectory.npz",
        "color": "#F59E0B",
        "label": "balanced OT-CFM",
    },
}


def load_trajectory(path: Path) -> dict:
    z = np.load(path)
    time_points = np.asarray(z["time_points"], dtype=float)
    points = {}
    weights = {}
    for i, time in enumerate(time_points):
        pk = f"points_{i:06d}"
        wk = f"weights_{i:06d}"
        if pk in z.files:
            points[float(time)] = np.asarray(z[pk], dtype=np.float32)
        if wk in z.files:
            weights[float(time)] = np.asarray(z[wk], dtype=np.float32)
    return {"time_points": time_points, "points": points, "weights": weights}


def nearest_time(times, target: float) -> float:
    arr = np.asarray(list(times), dtype=float)
    return float(arr[np.argmin(np.abs(arr - float(target)))])


def density_grid(xy: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray, weights: np.ndarray | None = None, sigma: float = 1.15) -> np.ndarray:
    w = None
    if weights is not None:
        ww = np.clip(np.asarray(weights, dtype=float).reshape(-1), 0, np.inf)
        if ww.shape[0] == xy.shape[0] and np.isfinite(ww).all() and ww.sum() > 0:
            w = ww / ww.sum()
    hist, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=[x_edges, y_edges], weights=w)
    hist = gaussian_filter(hist.T, sigma=sigma)
    max_value = float(np.nanmax(hist))
    return hist / max_value if max_value > 0 else hist


def weighted_mean(xy: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    if weights is None:
        return np.nanmean(xy, axis=0)
    ww = np.clip(np.asarray(weights, dtype=float).reshape(-1), 0, np.inf)
    if ww.shape[0] != xy.shape[0] or not np.isfinite(ww).all() or ww.sum() <= 0:
        return np.nanmean(xy, axis=0)
    return np.average(xy, axis=0, weights=ww / ww.sum())


def draw_distribution_panel(ax, xx, yy, obs_density, gen_density, obs_centroid, gen_centroid, method_color, pe_value: float) -> None:
    obs_levels = [0.06, 0.14, 0.26, 0.42, 0.64, 1.01]
    gen_levels = [0.12, 0.24, 0.40, 0.60, 0.82, 1.01]
    ax.contourf(xx, yy, obs_density, levels=obs_levels, colors=["#F8FAFC", "#F1F5F9", "#E2E8F0", "#CBD5E1", "#94A3B8"], alpha=0.76)
    ax.contour(xx, yy, obs_density, levels=[0.18, 0.36, 0.58], colors="#94A3B8", linewidths=0.7, alpha=0.72)
    ax.contourf(xx, yy, gen_density, levels=gen_levels, colors=[method_color] * 5, alpha=0.13)
    ax.contour(xx, yy, gen_density, levels=[0.20, 0.42, 0.68], colors=method_color, linewidths=[1.2, 1.7, 2.4])
    ax.plot([obs_centroid[0], gen_centroid[0]], [obs_centroid[1], gen_centroid[1]], color=method_color, lw=1.3, alpha=0.75)
    ax.scatter([obs_centroid[0]], [obs_centroid[1]], s=30, color="white", edgecolor="#475569", linewidth=1.1, zorder=8)
    ax.scatter([gen_centroid[0]], [gen_centroid[1]], s=46, color=method_color, edgecolor="white", linewidth=1.2, zorder=9)
    ax.text(
        0.98,
        0.94,
        f"PE {pe_value:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.6,
        color="#0F172A",
        bbox={"boxstyle": "round,pad=0.22,rounding_size=0.08", "facecolor": "white", "edgecolor": "#E2E8F0", "alpha": 0.92},
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#CBD5E1")
        spine.set_linewidth(0.9)


def metric_card(ax, x: float, y: float, w: float, h: float, title: str, value: str, color: str, subtitle: str) -> None:
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.035", facecolor="#F8FAFC", edgecolor="#CBD5E1", linewidth=1.0))
    ax.text(x + 0.035, y + h - 0.08, title, fontsize=8.8, color="#475569", ha="left", va="top")
    ax.text(x + 0.035, y + h - 0.22, value, fontsize=15, color=color, weight="bold", ha="left", va="top")
    ax.text(x + 0.035, y + 0.06, subtitle, fontsize=7.6, color="#64748B", ha="left", va="bottom")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    adata = ad.read_h5ad(DATA_PATH)
    obs = adata.obs.copy()
    obs["time_point_processed"] = obs["time_point_processed"].astype(float)
    protein = np.asarray(adata.obsm["X_protein_pca"], dtype=np.float32)
    trajectories = {method: load_trajectory(spec["trajectory"]) for method, spec in METHODS.items()}
    per_time = pd.read_csv(TABLE_DIR / "per_time.csv")
    score_raw = pd.read_csv(TABLE_DIR / "summary_scorecard_raw.csv")

    focus_blocks = [protein[:, :2]]
    for method, traj in trajectories.items():
        for time in TIMES:
            tt = nearest_time(traj["points"].keys(), time)
            focus_blocks.append(traj["points"][tt][:, RNA_DIM : RNA_DIM + 2])
    focus = np.vstack(focus_blocks)
    x0, x1 = np.nanpercentile(focus[:, 0], [0.35, 99.55])
    y0, y1 = np.nanpercentile(focus[:, 1], [0.35, 99.55])
    x_pad = 0.06 * (x1 - x0)
    y_pad = 0.08 * (y1 - y0)
    xlim = (x0 - x_pad, x1 + x_pad)
    ylim = (y0 - y_pad, y1 + y_pad)
    x_edges = np.linspace(xlim[0], xlim[1], 92)
    y_edges = np.linspace(ylim[0], ylim[1], 92)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    xx, yy = np.meshgrid(x_centers, y_centers)

    obs_densities = {}
    obs_centroids = {}
    for time in TIMES:
        mask = obs["time_point_processed"].to_numpy(dtype=float) == time
        obs_xy = protein[mask, :2]
        obs_densities[time] = density_grid(obs_xy, x_edges, y_edges, sigma=1.2)
        obs_centroids[time] = np.nanmean(obs_xy, axis=0)

    gen_densities = {}
    gen_centroids = {}
    for method, traj in trajectories.items():
        for time in TIMES:
            tt = nearest_time(traj["points"].keys(), time)
            pts = traj["points"][tt]
            w = traj["weights"].get(tt)
            xy = pts[:, RNA_DIM : RNA_DIM + 2]
            gen_densities[(method, time)] = density_grid(xy, x_edges, y_edges, w, sigma=1.3)
            gen_centroids[(method, time)] = weighted_mean(xy, w)

    fig = plt.figure(figsize=(18.8, 11.2))
    title_ax = fig.add_axes([0.045, 0.925, 0.91, 0.06])
    title_ax.axis("off")
    title_ax.text(0, 0.75, "Generated protein distributions against matched GT", fontsize=24, weight="bold", color="#0F172A", ha="left")
    title_ax.text(
        0,
        0.18,
        "Grey density is the observed target CITE protein distribution at the same time; colored contours are generated distributions from each algorithm.",
        fontsize=11.2,
        color="#475569",
        ha="left",
    )
    title_ax.text(1.0, 0.18, "lower PE and lower aggregate mismatch are better", fontsize=9.2, color="#475569", ha="right")

    grid_left, grid_bottom = 0.080, 0.310
    panel_w, panel_h = 0.132, 0.162
    x_gap, y_gap = 0.014, 0.035
    methods = list(METHODS.keys())
    for col, time in enumerate(TIMES):
        fig.text(grid_left + col * (panel_w + x_gap) + 0.006, grid_bottom + len(methods) * (panel_h + y_gap) - 0.014, f"t{int(time)}", fontsize=12.5, weight="bold", color="#0F172A")
    for row, method in enumerate(methods):
        y = grid_bottom + (len(methods) - 1 - row) * (panel_h + y_gap)
        spec = METHODS[method]
        future_pe = score_raw.loc[score_raw["method"].eq(method), "protein_energy"].iloc[0]
        fig.text(grid_left - 0.018, y + panel_h / 2, f"{spec['label']}\nfuture PE {future_pe:.3f}", fontsize=10, weight="bold", color=spec["color"], ha="right", va="center")
        for col, time in enumerate(TIMES):
            x = grid_left + col * (panel_w + x_gap)
            ax = fig.add_axes([x, y, panel_w, panel_h])
            metric = per_time[(per_time["method"].eq(method)) & np.isclose(per_time["time"], time)]["protein_energy"]
            pe = float(metric.iloc[0]) if len(metric) else np.nan
            draw_distribution_panel(
                ax,
                xx,
                yy,
                obs_densities[time],
                gen_densities[(method, time)],
                obs_centroids[time],
                gen_centroids[(method, time)],
                spec["color"],
                pe,
            )
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
    fig.text(grid_left, grid_bottom - 0.03, "white point: GT centroid; colored point: generated centroid; connecting segment: centroid displacement", fontsize=8.5, color="#64748B")

    line_ax = fig.add_axes([0.835, 0.595, 0.135, 0.285])
    for method, spec in METHODS.items():
        d = per_time[per_time["method"].eq(method)].sort_values("time")
        line_ax.plot(d["time"], d["protein_energy"], marker="o", lw=2.2, ms=4.6, color=spec["color"], label=spec["label"])
    line_ax.set_yscale("log")
    line_ax.set_title("protein energy by time", loc="left", fontsize=12.3, weight="bold")
    line_ax.set_xlabel("time", fontsize=9)
    line_ax.set_ylabel("PE, log scale", fontsize=9)
    line_ax.tick_params(labelsize=8)
    line_ax.spines[["top", "right"]].set_visible(False)
    line_ax.legend(frameon=False, fontsize=7.8, loc="upper left")

    score = score_raw.copy()
    metric_cols = ["protein_energy", "covariance_error", "nearest_distance", "variance_error"]
    for col in metric_cols:
        lo, hi = float(score[col].min()), float(score[col].max())
        score[f"{col}_norm"] = 0.0 if hi == lo else (score[col] - lo) / (hi - lo)
    score["aggregate_mismatch"] = score[[f"{col}_norm" for col in metric_cols]].mean(axis=1)
    score = score.sort_values("aggregate_mismatch", ascending=True)

    bar_ax = fig.add_axes([0.835, 0.300, 0.135, 0.205])
    y_pos = np.arange(score.shape[0])
    colors = [METHODS[m]["color"] for m in score["method"]]
    bar_ax.barh(y_pos, score["aggregate_mismatch"], color=colors, alpha=0.86, height=0.58)
    bar_ax.set_yticks(y_pos)
    bar_ax.set_yticklabels(score["method"], fontsize=8)
    bar_ax.invert_yaxis()
    bar_ax.set_xlim(0, 1.0)
    bar_ax.set_xlabel("normalized error", fontsize=9)
    bar_ax.set_title("future-time aggregate", loc="left", fontsize=12.3, weight="bold")
    bar_ax.tick_params(axis="x", labelsize=8)
    bar_ax.spines[["top", "right", "left"]].set_visible(False)
    for y_i, (_, row) in enumerate(score.iterrows()):
        bar_ax.text(row["aggregate_mismatch"] + 0.025, y_i, f"{row['aggregate_mismatch']:.2f}", va="center", fontsize=8.5, color="#0F172A", weight="bold")

    card_ax = fig.add_axes([0.055, 0.055, 0.915, 0.16])
    card_ax.axis("off")
    best = score.iloc[0]
    metric_card(card_ax, 0.000, 0.06, 0.235, 0.78, "Best overall distribution match", str(best["method"]), METHODS[str(best["method"])]["color"], f"aggregate normalized error {best['aggregate_mismatch']:.2f}")
    pcfm = score_raw[score_raw["method"].eq("CITE-PCFM")].iloc[0]
    crufm = score_raw[score_raw["method"].eq("CRUFM")].iloc[0]
    bot = score_raw[score_raw["method"].eq("balanced OT-CFM")].iloc[0]
    metric_card(card_ax, 0.255, 0.06, 0.215, 0.78, "Future PE", f"{pcfm['protein_energy']:.3f}", METHODS["CITE-PCFM"]["color"], f"CRUFM {crufm['protein_energy']:.3f}; balanced {bot['protein_energy']:.3f}")
    metric_card(card_ax, 0.490, 0.06, 0.215, 0.78, "RNA-protein covariance", f"{pcfm['covariance_error']:.3f}", METHODS["CITE-PCFM"]["color"], f"CRUFM {crufm['covariance_error']:.3f}; balanced {bot['covariance_error']:.3f}")
    metric_card(card_ax, 0.725, 0.06, 0.255, 0.78, "Diversity preservation", f"{pcfm['variance_error']:.3f}", METHODS["CITE-PCFM"]["color"], f"CRUFM {crufm['variance_error']:.3f}; balanced {bot['variance_error']:.3f}")

    out = FIG_DIR / "fig01b_fancy_distribution_gt_metric_panel.png"
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

    print(json.dumps({"figure": str(out), "pdf": str(out.with_suffix(".pdf"))}, indent=2))


if __name__ == "__main__":
    main()
