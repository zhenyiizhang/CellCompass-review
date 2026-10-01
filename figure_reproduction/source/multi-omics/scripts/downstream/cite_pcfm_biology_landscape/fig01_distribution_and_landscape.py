#!/usr/bin/env python3
"""Observed versus generated CITE-seq surface-protein distributions.

This script is intentionally narrow and evidence-driven. It uses the locked
evaluation trajectories as model-native generated predictions and uses observed
protein PCA only to define the visualization manifold and validation targets.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/cytobridge_mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/cytobridge_xdg_cache")
os.environ.setdefault("NUMBA_CACHE_DIR", "/private/tmp/numba_cache")
os.environ.setdefault("NUMBA_DISABLE_CACHE", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")
for key in ["MPLCONFIGDIR", "XDG_CACHE_HOME", "NUMBA_CACHE_DIR"]:
    Path(os.environ[key]).mkdir(parents=True, exist_ok=True)

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from multistageot_style import (
    PHENO_COLORS,
    PHENO_ORDER,
    TIME_COLORS,
    add_median_labels,
    clean_embedding_axis,
    scatter_background,
    scatter_groups,
    setup,
)

try:
    import umap  # type: ignore

    HAVE_UMAP = True
except Exception:
    HAVE_UMAP = False


ROOT = Path("/Users/zhenyizhang/Documents/GitHub/CytoBridge-agent-runtime-v2-20260309/multi-omics")
REPO_ROOT = ROOT.parent
DATA_PATH = Path("/Users/zhenyizhang/.cellcompass/algorithm_benchmarks/datasets/gse305370_citeseq_stage2_10k/data.h5ad")
TRAINED_MODEL_H5AD = ROOT / "training_runs/20260526T174204321821Z__final__cite_paired_coord_fm__campaign_20260-trial_20260526-gse305370_ci/artifacts/trained_model.h5ad"
CYTOBRIDGE_SRC = REPO_ROOT / "CytoBridge-main"
OUT_ROOT = ROOT / "downstream/cite_pcfm_deep_story"
FIG_DIR = ROOT / "figures/cite_pcfm_deep_story/biology_landscape_discovery"
TABLE_DIR = OUT_ROOT / "biology_landscape_discovery_tables"
STREAM_DIR = OUT_ROOT / "biology_landscape_discovery_velocity_stream"
MANIFEST_PATH = OUT_ROOT / "biology_landscape_discovery_manifest.json"
STORY_PATH = OUT_ROOT / "biology_landscape_discovery_story.md"

RNA_DIM = 50
PROTEIN_DIM = 30
TARGET_TIME = 4.0
VALIDATION_TIMES = [1.0, 2.0, 3.0, 4.0]
RNG = np.random.default_rng(20260528)

METHODS = {
    "ProteinBridge": {
        "trajectory": ROOT / "training_runs/20260526T174204321821Z__final__cite_paired_coord_fm__campaign_20260-trial_20260526-gse305370_ci/artifacts/evaluation_trajectory.npz",
        "metrics": ROOT / "training_runs/20260526T174204321821Z__final__cite_paired_coord_fm__campaign_20260-trial_20260526-gse305370_ci/artifacts/metrics.json",
        "color": "#E11D48",
        "role": "RNA-conditioned protein readout",
    },
    "CRUFM": {
        "trajectory": ROOT / "training_runs/20260525T220649168798Z__pilot__crufm__campaign_20260-stage2_c-baseline-crufm-gse305370_ci/artifacts/evaluation_trajectory.npz",
        "metrics": ROOT / "training_runs/20260525T220649168798Z__pilot__crufm__campaign_20260-stage2_c-baseline-crufm-gse305370_ci/artifacts/metrics.json",
        "color": "#2563EB",
        "role": "baseline",
    },
    "balanced OT-CFM": {
        "trajectory": ROOT / "training_runs/20260525T215826526439Z__pilot__balanced_ot_cfm__campaign_20260-stage2_c-baseline-balanced_ot_-gse305370_ci/artifacts/evaluation_trajectory.npz",
        "metrics": ROOT / "training_runs/20260525T215826526439Z__pilot__balanced_ot_cfm__campaign_20260-stage2_c-baseline-balanced_ot_-gse305370_ci/artifacts/metrics.json",
        "color": "#F59E0B",
        "role": "baseline",
    },
}
TIME_COLORS = {
    0.0: "#2563EB",
    1.0: "#06B6D4",
    2.0: "#22C55E",
    3.0: "#F59E0B",
    4.0: "#E11D48",
}
setup()
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "mathtext.sf": "Arial",
        "mathtext.cal": "Arial",
        "mathtext.tt": "Arial",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    STREAM_DIR.mkdir(parents=True, exist_ok=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)


def load_trajectory(path: Path) -> dict:
    z = np.load(path)
    times = np.asarray(z["time_points"], dtype=float)
    points: dict[float, np.ndarray] = {}
    weights: dict[float, np.ndarray] = {}
    for i, time in enumerate(times):
        pkey = f"points_{i:06d}"
        wkey = f"weights_{i:06d}"
        if pkey in z.files:
            points[float(time)] = np.asarray(z[pkey], dtype=np.float32)
        if wkey in z.files:
            weights[float(time)] = np.asarray(z[wkey], dtype=np.float32)
    return {"time_points": times, "points": points, "weights": weights}


def nearest_time(points: dict[float, np.ndarray], target: float) -> float:
    times = np.asarray(list(points.keys()), dtype=float)
    return float(times[np.argmin(np.abs(times - target))])


def sample_idx(n: int, max_n: int, weights: np.ndarray | None = None) -> np.ndarray:
    if n <= max_n:
        return np.arange(n, dtype=int)
    prob = None
    if weights is not None:
        w = np.clip(np.asarray(weights, dtype=float).reshape(-1), 0, np.inf)
        if w.shape[0] == n and np.isfinite(w).all() and w.sum() > 0:
            prob = w / w.sum()
    return np.sort(RNG.choice(n, size=max_n, replace=False, p=prob))


def fit_gt_embedding(x_gt: np.ndarray) -> tuple[str, StandardScaler, object, np.ndarray]:
    scaler = StandardScaler().fit(x_gt)
    x_scaled = scaler.transform(x_gt)
    if HAVE_UMAP:
        reducer = umap.UMAP(
            n_neighbors=35,
            min_dist=0.22,
            metric="euclidean",
            random_state=20260528,
            low_memory=True,
        )
        emb = reducer.fit_transform(x_scaled)
        return "UMAP", scaler, reducer, emb
    reducer = PCA(n_components=2, random_state=20260528)
    emb = reducer.fit_transform(x_scaled)
    return "PCA fallback", scaler, reducer, emb


def transform_embedding(scaler: StandardScaler, reducer: object, x: np.ndarray) -> np.ndarray:
    return reducer.transform(scaler.transform(x))


def _set_collection_rasterization(fig: plt.Figure, rasterized: bool) -> None:
    for ax in fig.axes:
        for coll in ax.collections:
            try:
                coll.set_rasterized(rasterized)
            except Exception:
                pass


def save_figure_outputs(
    fig: plt.Figure,
    png: Path,
    *,
    dpi: int = 320,
    vector_pdf: bool = False,
) -> tuple[str, str]:
    pdf = png.with_suffix(".pdf")
    fig.savefig(png, dpi=dpi, bbox_inches="tight", facecolor="white")
    if vector_pdf:
        _set_collection_rasterization(fig, False)
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    return str(png), str(pdf)


def load_locked_protein_energy(
    method: str,
    metrics_path: Path,
    trajectory: dict,
) -> tuple[list[dict], float]:
    """Read the protein energy values sealed by the model evaluator."""
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    custom_metrics = payload["custom_metrics"]
    per_time = custom_metrics["protein_energy_per_time_json"]
    if isinstance(per_time, str):
        per_time = json.loads(per_time)

    rows = []
    for target_time in VALIDATION_TIMES:
        matched = [
            record
            for record in per_time
            if np.isclose(float(record["time_point"]), target_time)
        ]
        if len(matched) != 1:
            raise ValueError(
                f"Expected one locked protein energy value for {method} at "
                f"t={target_time}, found {len(matched)} in {metrics_path}"
            )
        rows.append(
            {
                "method": method,
                "target_time": target_time,
                "trajectory_time": nearest_time(trajectory["points"], target_time),
                "protein_energy": float(matched[0]["protein_energy"]),
            }
        )

    locked_mean = float(custom_metrics["protein_energy"])
    per_time_mean = float(np.mean([row["protein_energy"] for row in rows]))
    if not np.isclose(locked_mean, per_time_mean, atol=1e-10, rtol=1e-8):
        raise ValueError(
            f"Locked mean protein energy for {method} ({locked_mean}) does not "
            f"match its per-time values ({per_time_mean})"
        )
    return rows, locked_mean


def density_grid(xy: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray, sigma: float = 1.1) -> np.ndarray:
    hist, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=[x_edges, y_edges])
    den = gaussian_filter(hist.T.astype(float), sigma=sigma)
    m = float(np.nanmax(den))
    return den / m if m > 0 else den


def draw_gt_terrain(ax: plt.Axes, gt_xy: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray, label_centers: dict[str, tuple[float, float]] | None = None) -> None:
    xx, yy = np.meshgrid(0.5 * (x_edges[:-1] + x_edges[1:]), 0.5 * (y_edges[:-1] + y_edges[1:]))
    gt_den = density_grid(gt_xy, x_edges, y_edges, sigma=1.05)
    ax.contourf(
        xx,
        yy,
        gt_den,
        levels=[0.04, 0.11, 0.22, 0.40, 0.62, 1.01],
        colors=["#F8FAFC", "#F1F5F9", "#E2E8F0", "#CBD5E1", "#94A3B8"],
        alpha=0.90,
    )
    ax.contour(xx, yy, gt_den, levels=[0.18, 0.38, 0.62], colors="#64748B", linewidths=[0.5, 0.75, 1.0], alpha=0.55)
    if label_centers:
        for label, (x, y) in label_centers.items():
            ax.text(
                x,
                y,
                label,
                fontsize=8,
                weight="bold",
                color=PHENO_COLORS.get(label, "#334155"),
                ha="center",
                va="center",
                bbox={"boxstyle": "round,pad=0.10", "facecolor": "white", "edgecolor": "none", "alpha": 0.70},
            )


def draw_generated_overlay(ax: plt.Axes, gen_xy: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray, color: str) -> None:
    xx, yy = np.meshgrid(0.5 * (x_edges[:-1] + x_edges[1:]), 0.5 * (y_edges[:-1] + y_edges[1:]))
    gen_den = density_grid(gen_xy, x_edges, y_edges, sigma=1.15)
    ax.scatter(gen_xy[:, 0], gen_xy[:, 1], s=4, color=color, alpha=0.08, linewidths=0)
    ax.contourf(xx, yy, gen_den, levels=[0.16, 0.30, 0.48, 0.68, 0.86, 1.01], colors=[color] * 5, alpha=0.10)
    ax.contour(xx, yy, gen_den, levels=[0.20, 0.40, 0.62], colors=color, linewidths=[1.2, 1.9, 2.7])


def scatter_cell_types(
    ax: plt.Axes,
    xy: np.ndarray,
    labels: np.ndarray,
    *,
    s: float = 9.0,
    alpha: float = 0.78,
    annotate: bool = True,
) -> None:
    scatter_background(ax, xy, s=max(7.5, s * 0.95), alpha=0.92, zorder=0)
    scatter_groups(ax, xy, labels, PHENO_COLORS, order=PHENO_ORDER, s=s, alpha=alpha, zorder=2)
    if annotate:
        add_median_labels(ax, xy, labels, order=PHENO_ORDER, min_cells=10, color_map=PHENO_COLORS, fontsize=8.0)


def scatter_times(ax: plt.Axes, xy: np.ndarray, time: np.ndarray, *, s: float = 7.5, alpha: float = 0.72) -> None:
    scatter_background(ax, xy, s=max(7.0, s * 0.95), alpha=0.88, zorder=0)
    for t in sorted(np.unique(time[np.isfinite(time)])):
        mask = np.isclose(time, t)
        ax.scatter(
            xy[mask, 0],
            xy[mask, 1],
            s=s,
            color=TIME_COLORS.get(float(t), "#64748B"),
            alpha=alpha,
            linewidths=0,
            rasterized=True,
            label=f"t{int(t)}" if float(t).is_integer() else f"t{t:g}",
        )


def style_embedding_axis(ax: plt.Axes, xlabel: str, ylabel: str) -> None:
    clean_embedding_axis(ax)


def save_real_landscape_reference(
    rna_emb: np.ndarray,
    protein_emb: np.ndarray,
    labels: np.ndarray,
    time: np.ndarray,
    rna_emb_name: str,
    protein_emb_name: str,
) -> tuple[str, str]:
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.4), constrained_layout=True)
    scatter_cell_types(axes[0, 0], rna_emb, labels, s=9.0, alpha=0.78, annotate=True)
    axes[0, 0].set_title("Observed RNA landscape by cell type", loc="left", fontsize=12, weight="bold")
    scatter_times(axes[0, 1], rna_emb, time, s=7.5, alpha=0.72)
    axes[0, 1].set_title("Observed RNA landscape by time", loc="left", fontsize=12, weight="bold")
    scatter_cell_types(axes[1, 0], protein_emb, labels, s=9.0, alpha=0.78, annotate=True)
    axes[1, 0].set_title("Observed protein landscape by cell type", loc="left", fontsize=12, weight="bold")
    scatter_times(axes[1, 1], protein_emb, time, s=7.5, alpha=0.72)
    axes[1, 1].set_title("Observed protein landscape by time", loc="left", fontsize=12, weight="bold")
    for ax in axes[0, :]:
        style_embedding_axis(ax, f"RNA {rna_emb_name} 1", f"RNA {rna_emb_name} 2")
    for ax in axes[1, :]:
        style_embedding_axis(ax, f"protein {protein_emb_name} 1", f"protein {protein_emb_name} 2")
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=TIME_COLORS[t], markersize=6, label=f"t{int(t)}") for t in sorted(TIME_COLORS)]
    axes[0, 1].legend(handles=handles, frameon=False, fontsize=8, ncol=5, loc="lower left")
    axes[1, 1].legend(handles=handles, frameon=False, fontsize=8, ncol=5, loc="lower left")
    fig.suptitle("Observed paired CITE-seq landscapes", fontsize=17, weight="bold")
    png = FIG_DIR / "fig01a_observed_rna_protein_landscapes_by_celltype_and_time.png"
    png_str, pdf_str = save_figure_outputs(fig, png, dpi=320, vector_pdf=True)
    plt.close(fig)
    return png_str, pdf_str


def save_generated_timecourse(
    trajectories: dict[str, dict],
    metric_df: pd.DataFrame,
    obs_emb: np.ndarray,
    protein: np.ndarray,
    time: np.ndarray,
    scaler: StandardScaler,
    reducer: object,
    emb_name: str,
) -> tuple[str, str, str]:
    time_grid = [0.0, 1.0, 2.0, 3.0, 4.0]
    generated: dict[tuple[str, float], np.ndarray] = {}
    all_xy = [obs_emb]
    rows = []
    for method, traj in trajectories.items():
        for t in time_grid:
            tt = nearest_time(traj["points"], t)
            pts = traj["points"][tt]
            w = traj["weights"].get(tt)
            idx = sample_idx(pts.shape[0], 520, w)
            pred = pts[idx, RNA_DIM : RNA_DIM + PROTEIN_DIM]
            pred_xy = transform_embedding(scaler, reducer, pred)
            generated[(method, t)] = pred_xy
            all_xy.append(pred_xy)
            for particle, xy in zip(idx, pred_xy):
                rows.append(
                    {
                        "method": method,
                        "time": t,
                        "trajectory_time": tt,
                        "particle": int(particle),
                        "umap1": float(xy[0]),
                        "umap2": float(xy[1]),
                    }
                )
    out_table = TABLE_DIR / "fig01b_generated_protein_umap_all_times.csv"
    pd.DataFrame(rows).to_csv(out_table, index=False)

    all_xy_arr = np.vstack(all_xy)
    xlim = tuple(np.nanpercentile(all_xy_arr[:, 0], [0.4, 99.6]))
    ylim = tuple(np.nanpercentile(all_xy_arr[:, 1], [0.4, 99.6]))
    x_pad = 0.05 * (xlim[1] - xlim[0])
    y_pad = 0.05 * (ylim[1] - ylim[0])
    xlim = (xlim[0] - x_pad, xlim[1] + x_pad)
    ylim = (ylim[0] - y_pad, ylim[1] + y_pad)

    methods = list(METHODS)
    fig, axes = plt.subplots(len(methods), len(time_grid), figsize=(17.8, 9.2), constrained_layout=True)
    for row, method in enumerate(methods):
        for col, t in enumerate(time_grid):
            ax = axes[row, col]
            mask = np.isclose(time, t)
            ax.scatter(obs_emb[mask, 0], obs_emb[mask, 1], s=5, color="#CBD5E1", alpha=0.34, linewidths=0, rasterized=True)
            gen_xy = generated[(method, t)]
            ax.scatter(gen_xy[:, 0], gen_xy[:, 1], s=8, color=METHODS[method]["color"], alpha=0.32, linewidths=0, rasterized=True)
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            style_embedding_axis(ax, "", "")
            if row == 0:
                ax.set_title(f"t{int(t)}", fontsize=11.5, weight="bold")
            if col == 0:
                ax.text(-0.12, 0.5, method, transform=ax.transAxes, rotation=90, ha="center", va="center", fontsize=11, weight="bold", color=METHODS[method]["color"])
            if t > 0:
                pe = float(metric_df[(metric_df["method"] == method) & np.isclose(metric_df["target_time"], t)]["protein_energy"].iloc[0])
                ax.text(0.96, 0.05, f"distance {pe:.3f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color="#0F172A", bbox={"boxstyle": "round,pad=0.14", "facecolor": "white", "edgecolor": "#CBD5E1", "alpha": 0.85})
            else:
                ax.text(0.96, 0.05, "source", transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color="#0F172A", bbox={"boxstyle": "round,pad=0.14", "facecolor": "white", "edgecolor": "#CBD5E1", "alpha": 0.85})
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#CBD5E1", markersize=6, label="matched observed surface-protein states"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#64748B", markersize=6, label="coloured dots = generated surface-protein states"),
    ]
    fig.legend(handles=handles, frameon=False, fontsize=9, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Generated surface-protein distributions on the observed surface-protein UMAP", fontsize=17, weight="bold")
    png = FIG_DIR / "fig01b_generated_protein_umap_timecourse_by_method.png"
    png_str, pdf_str = save_figure_outputs(fig, png, dpi=320, vector_pdf=True)
    plt.close(fig)
    return png_str, pdf_str, str(out_table)


def save_generated_pca_timecourse(
    trajectories: dict[str, dict],
    metric_df: pd.DataFrame,
    protein: np.ndarray,
    time: np.ndarray,
) -> tuple[str, str, str]:
    """Generated protein distributions over time in direct protein PC1/PC2 space."""
    time_grid = [0.0, 1.0, 2.0, 3.0, 4.0]
    generated: dict[tuple[str, float], np.ndarray] = {}
    all_xy = [protein[:, :2]]
    rows = []
    for method, traj in trajectories.items():
        for t in time_grid:
            tt = nearest_time(traj["points"], t)
            pts = traj["points"][tt]
            w = traj["weights"].get(tt)
            idx = sample_idx(pts.shape[0], 520, w)
            pred_xy = pts[idx, RNA_DIM : RNA_DIM + 2]
            generated[(method, t)] = pred_xy
            all_xy.append(pred_xy)
            for particle, xy in zip(idx, pred_xy):
                rows.append(
                    {
                        "method": method,
                        "time": t,
                        "trajectory_time": tt,
                        "particle": int(particle),
                        "protein_pc1": float(xy[0]),
                        "protein_pc2": float(xy[1]),
                    }
                )

    out_table = TABLE_DIR / "fig01d_generated_protein_pca_all_times.csv"
    pd.DataFrame(rows).to_csv(out_table, index=False)

    all_xy_arr = np.vstack(all_xy)
    xlim = tuple(np.nanpercentile(all_xy_arr[:, 0], [0.4, 99.6]))
    ylim = tuple(np.nanpercentile(all_xy_arr[:, 1], [0.4, 99.6]))
    x_pad = 0.06 * (xlim[1] - xlim[0])
    y_pad = 0.06 * (ylim[1] - ylim[0])
    xlim = (xlim[0] - x_pad, xlim[1] + x_pad)
    ylim = (ylim[0] - y_pad, ylim[1] + y_pad)

    methods = list(METHODS)
    fig, axes = plt.subplots(len(methods), len(time_grid), figsize=(17.8, 9.2), constrained_layout=True)
    for row, method in enumerate(methods):
        for col, t in enumerate(time_grid):
            ax = axes[row, col]
            mask = np.isclose(time, t)
            ax.scatter(protein[mask, 0], protein[mask, 1], s=5, color="#CBD5E1", alpha=0.35, linewidths=0, rasterized=True)
            pred_xy = generated[(method, t)]
            ax.scatter(pred_xy[:, 0], pred_xy[:, 1], s=8, color=METHODS[method]["color"], alpha=0.32, linewidths=0, rasterized=True)
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            style_embedding_axis(ax, "", "")
            if row == 0:
                ax.set_title(f"t{int(t)}", fontsize=11.5, weight="bold")
            if col == 0:
                ax.text(-0.12, 0.5, method, transform=ax.transAxes, rotation=90, ha="center", va="center", fontsize=11, weight="bold", color=METHODS[method]["color"])
            if t > 0:
                pe = float(metric_df[(metric_df["method"] == method) & np.isclose(metric_df["target_time"], t)]["protein_energy"].iloc[0])
                ax.text(0.96, 0.05, f"distance {pe:.3f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color="#0F172A", bbox={"boxstyle": "round,pad=0.14", "facecolor": "white", "edgecolor": "#CBD5E1", "alpha": 0.85})
            else:
                ax.text(0.96, 0.05, "source", transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color="#0F172A", bbox={"boxstyle": "round,pad=0.14", "facecolor": "white", "edgecolor": "#CBD5E1", "alpha": 0.85})

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#CBD5E1", markersize=6, label="matched observed surface-protein states"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#64748B", markersize=6, label="coloured dots = generated surface-protein states"),
    ]
    fig.legend(handles=handles, frameon=False, fontsize=9, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Generated surface-protein distributions in the observed protein PC1/PC2 space", fontsize=17, weight="bold")
    png = FIG_DIR / "fig01d_generated_protein_pca_timecourse_by_method.png"
    png_str, pdf_str = save_figure_outputs(fig, png, dpi=320, vector_pdf=True)
    plt.close(fig)
    return png_str, pdf_str, str(out_table)


def save_gt_pca_comparison(
    trajectories: dict[str, dict],
    metric_df: pd.DataFrame,
    protein: np.ndarray,
    labels: np.ndarray,
    time: np.ndarray,
) -> tuple[str, str, str]:
    """Direct PC1/PC2 comparison in the shared GT protein PCA coordinate system."""
    target_mask = np.isclose(time, TARGET_TIME)
    gt_xy = protein[target_mask, :2]
    gt_labels = labels[target_mask]
    rows = []
    generated_xy_by_method = {}
    for method, traj in trajectories.items():
        tt = nearest_time(traj["points"], TARGET_TIME)
        pts = traj["points"][tt]
        w = traj["weights"].get(tt)
        idx = sample_idx(pts.shape[0], 1200, w)
        pred = pts[idx, RNA_DIM : RNA_DIM + PROTEIN_DIM]
        pred_xy = pred[:, :2]
        generated_xy_by_method[method] = pred_xy
        for particle, xy in zip(idx, pred_xy):
            rows.append(
                {
                    "method": method,
                    "target_time": TARGET_TIME,
                    "trajectory_time": tt,
                    "particle": int(particle),
                    "protein_pc1": float(xy[0]),
                    "protein_pc2": float(xy[1]),
                }
            )

    out_table = TABLE_DIR / "fig01c_generated_protein_gt_pca_target_time.csv"
    pd.DataFrame(rows).to_csv(out_table, index=False)

    all_xy = np.vstack([gt_xy] + list(generated_xy_by_method.values()))
    xlim = tuple(np.nanpercentile(all_xy[:, 0], [0.4, 99.6]))
    ylim = tuple(np.nanpercentile(all_xy[:, 1], [0.4, 99.6]))
    x_pad = 0.06 * (xlim[1] - xlim[0])
    y_pad = 0.06 * (ylim[1] - ylim[0])
    xlim = (xlim[0] - x_pad, xlim[1] + x_pad)
    ylim = (ylim[0] - y_pad, ylim[1] + y_pad)

    methods = list(METHODS)
    fig = plt.figure(figsize=(16.4, 5.1))
    ax_title = fig.add_axes([0.045, 0.865, 0.91, 0.105])
    ax_title.axis("off")
    ax_title.text(0, 0.72, "Linear sanity check in the GT protein PCA coordinate system", fontsize=19.5, weight="bold", color="#0F172A")
    ax_title.text(
        0,
        0.20,
        "No nonlinear embedding is used here: GT cells use observed protein PC1/PC2, and generated particles use the same trajectory protein block dimensions.",
        fontsize=10.2,
        color="#475569",
    )

    panel_y, panel_h = 0.190, 0.585
    panel_w, gap = 0.210, 0.026
    left = 0.055
    ax_gt = fig.add_axes([left, panel_y, panel_w, panel_h])
    scatter_cell_types(ax_gt, gt_xy, gt_labels, s=9.0, alpha=0.78, annotate=True)
    ax_gt.set_title(f"GT protein t{int(TARGET_TIME)} on PC1/PC2", loc="left", fontsize=11.5, weight="bold")
    ax_gt.text(0.02, 0.04, f"observed target cells\nn={int(target_mask.sum())}", transform=ax_gt.transAxes, fontsize=8.3, color="#334155")

    axes = [ax_gt]
    for i, method in enumerate(methods, start=1):
        ax = fig.add_axes([left + i * (panel_w + gap), panel_y, panel_w, panel_h])
        ax.scatter(gt_xy[:, 0], gt_xy[:, 1], s=7, color="#CBD5E1", alpha=0.40, linewidths=0, rasterized=True)
        pred_xy = generated_xy_by_method[method]
        ax.scatter(pred_xy[:, 0], pred_xy[:, 1], s=9, color=METHODS[method]["color"], alpha=0.30, linewidths=0, rasterized=True)
        pe_t = float(metric_df[(metric_df["method"] == method) & np.isclose(metric_df["target_time"], TARGET_TIME)]["protein_energy"].iloc[0])
        ax.set_title(method, loc="left", fontsize=11.5, weight="bold", color=METHODS[method]["color"])
        ax.text(
            0.98,
            0.05,
            f"t4 distance {pe_t:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8.3,
            color="#0F172A",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#CBD5E1", "alpha": 0.90},
        )
        axes.append(ax)

    for ax in axes:
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        style_embedding_axis(ax, "protein PC1", "protein PC2")

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#CBD5E1", markersize=6, label="GT observed t4 protein cells"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#64748B", markersize=6, label="coloured dots = generated surface-protein states"),
    ]
    fig.legend(handles=handles, frameon=False, fontsize=8.5, loc="lower left", bbox_to_anchor=(0.055, 0.045))
    png = FIG_DIR / "fig01c_gt_pca_vs_generated_protein_distribution.png"
    png_str, pdf_str = save_figure_outputs(fig, png, dpi=320, vector_pdf=True)
    plt.close(fig)
    return png_str, pdf_str, str(out_table)


def save_rna_scvelo_streams(rna_emb: np.ndarray) -> dict:
    """Package-native scVelo stream plots on the observed RNA UMAP basis."""
    if CYTOBRIDGE_SRC.exists() and str(CYTOBRIDGE_SRC) not in sys.path:
        sys.path.insert(0, str(CYTOBRIDGE_SRC))

    import torch
    from CytoBridge.utils import load_model_from_adata
    from CytoBridge.tl.downstream import build_velocity_graph_bundle
    from CytoBridge.pl.downstream import plot_velocity_stream_bundle

    model_adata = ad.read_h5ad(TRAINED_MODEL_H5AD)
    warnings: list[str] = []
    artifacts: dict[str, object] = {}

    if "X_rna_svd" not in model_adata.obsm:
        raise KeyError("trained_model.h5ad is missing obsm['X_rna_svd']")
    if "time_point_processed" not in model_adata.obs:
        raise KeyError("trained_model.h5ad is missing obs['time_point_processed']")
    if np.asarray(model_adata.obsm["X_rna_svd"]).shape[0] != rna_emb.shape[0]:
        raise ValueError("RNA UMAP row count does not match trained_model.h5ad")

    rna_state = np.asarray(model_adata.obsm["X_rna_svd"], dtype=np.float32)[:, :RNA_DIM]
    time_col = np.asarray(model_adata.obs["time_point_processed"], dtype=np.float32).reshape(-1, 1)

    # ProteinBridge's trusted runtime is a 50D RNA ODE plus protein readout h_eta(r,t).
    # The generic trained_model.h5ad may contain an 80D velocity_latent interface
    # field, but the protein block is a zero-velocity placeholder and should not
    # be treated as protein dynamics.
    model = load_model_from_adata(model_adata).cpu().eval()
    with torch.no_grad():
        r_tensor = torch.as_tensor(rna_state, dtype=torch.float32)
        zero_q = torch.zeros((rna_state.shape[0], PROTEIN_DIM), dtype=torch.float32)
        t_tensor = torch.as_tensor(time_col, dtype=torch.float32)
        full_velocity = model.velocity_net(torch.cat([r_tensor, zero_q, t_tensor], dim=1)).detach().cpu().numpy().astype(np.float32)
    rna_velocity = full_velocity[:, :RNA_DIM]
    protein_velocity_rms = float(np.sqrt(np.mean(full_velocity[:, RNA_DIM : RNA_DIM + PROTEIN_DIM] ** 2)))
    stored_velocity_diagnostic = None
    if "velocity_latent" in model_adata.obsm:
        stored = np.asarray(model_adata.obsm["velocity_latent"], dtype=np.float32)
        stored_velocity_diagnostic = {
            "stored_shape": list(stored.shape),
            "mean_abs_diff_runtime_rna_vs_stored_first50": float(np.mean(np.abs(rna_velocity - stored[:, :RNA_DIM]))),
            "stored_protein_block_rms": float(np.sqrt(np.mean(stored[:, RNA_DIM : RNA_DIM + PROTEIN_DIM] ** 2))),
        }

    rna_adata = ad.AnnData(
        X=rna_state,
        obs=model_adata.obs.copy(),
    )
    rna_adata.obs_names = model_adata.obs_names.copy()
    present_cell_types = [label for label in PHENO_ORDER if label in set(rna_adata.obs["cell_type"].astype(str))]
    rna_adata.obs["cell_type"] = pd.Categorical(
        rna_adata.obs["cell_type"].astype(str),
        categories=present_cell_types,
        ordered=True,
    )
    rna_adata.uns["cell_type_colors"] = [PHENO_COLORS[label] for label in present_cell_types]
    rna_adata.obsm["X_latent"] = rna_state
    rna_adata.obsm["velocity_latent"] = rna_velocity
    rna_adata.obsm["X_umap"] = np.asarray(rna_emb, dtype=np.float32)
    if "time_point_processed" in rna_adata.obs:
        time_values = pd.to_numeric(rna_adata.obs["time_point_processed"], errors="coerce").to_numpy(dtype=float)
        time_categories = []
        for t in sorted(np.unique(time_values[np.isfinite(time_values)])):
            time_categories.append(f"t{int(t)}" if float(t).is_integer() else f"t{t:g}")
        rna_adata.obs["time_label"] = pd.Categorical(
            [f"t{int(float(t))}" if float(t).is_integer() else f"t{float(t):g}" for t in time_values],
            categories=time_categories,
            ordered=True,
        )
        rna_adata.uns["time_label_colors"] = [
            TIME_COLORS.get(float(label[1:]), "#64748B") if label.startswith("t") else "#64748B"
            for label in time_categories
        ]

    graph_dir = STREAM_DIR / "rna_umap_graph"
    graph_dir.mkdir(parents=True, exist_ok=True)
    graph = build_velocity_graph_bundle(
        rna_adata,
        output_dir=str(graph_dir),
        n_neighbors=30,
        n_jobs=1,
        reuse_neighbors=False,
        vkey="velocity",
        preferred_basis="umap",
        strict_preferred=True,
    )
    warnings.extend(graph.get("warnings", []))
    artifacts["graph"] = graph.get("artifacts", {})

    plot_outputs = {}
    for color_key, suffix in [("cell_type", "celltype"), ("time_label", "time")]:
        out_dir = STREAM_DIR / f"rna_umap_stream_{suffix}"
        out_dir.mkdir(parents=True, exist_ok=True)
        plot = plot_velocity_stream_bundle(
            rna_adata,
            output_dir=str(out_dir),
            dim_reduction="umap",
            color_key=color_key,
            vkey="velocity",
        )
        warnings.extend(plot.get("warnings", []))
        src = Path(plot.get("artifacts", {}).get("velocity_stream", out_dir / "velocity_stream_umap.png"))
        dst_png = FIG_DIR / f"fig01e_rna_umap_scvelo_stream_by_{suffix}.png"
        if src.exists():
            shutil.copyfile(src, dst_png)
        plot_outputs[suffix] = {
            "color_key": color_key,
            "source_png": str(src),
            "png": str(dst_png),
            "plot_payload": plot,
        }

    summary = {
        "analysis": "rna_umap_scvelo_stream",
        "skill": "downstream-scvelo-stream",
        "evidence_type": "local flow diagnostic",
        "trained_model_h5ad": str(TRAINED_MODEL_H5AD),
        "basis": "observed RNA UMAP from X_rna_svd",
        "state_space": "X_rna_svd",
        "velocity_space": "runtime v_theta([X_rna_svd, zero_protein_block, time])[:, :50]",
        "full_velocity_shape": list(full_velocity.shape),
        "trusted_rna_velocity_shape": list(rna_velocity.shape),
        "runtime_protein_velocity_block_rms": protein_velocity_rms,
        "stored_velocity_latent_diagnostic": stored_velocity_diagnostic,
        "graph_payload": graph,
        "plots": plot_outputs,
        "warnings": warnings
        + [
            "This stream plot is a local RNA-space flow diagnostic, not continuous protein trajectory evidence.",
            "The basis is the observed RNA UMAP used in Fig. 1 reference panels.",
            "ProteinBridge rollout integrates RNA only; protein states are generated by h_eta(r,t), not by the protein block of velocity_latent.",
        ],
    }
    summary_path = STREAM_DIR / "rna_umap_scvelo_stream_manifest.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return {
        "celltype_png": plot_outputs["celltype"]["png"],
        "time_png": plot_outputs["time"]["png"],
        "manifest": str(summary_path),
        "warnings": summary["warnings"],
    }


def build_figure() -> dict:
    ensure_dirs()
    adata = ad.read_h5ad(DATA_PATH)
    obs = adata.obs.copy()
    time = pd.to_numeric(obs["time_point_processed"], errors="coerce").to_numpy(dtype=float)
    labels = obs["cell_type"].astype(str).to_numpy()
    protein = np.asarray(adata.obsm["X_protein_pca"], dtype=np.float32)
    rna = np.asarray(adata.obsm["X_rna_svd"], dtype=np.float32)

    emb_name, scaler, reducer, obs_emb = fit_gt_embedding(protein)
    rna_emb_name, _, _, rna_emb = fit_gt_embedding(rna)
    obs_df = pd.DataFrame(
        {
            "umap1": obs_emb[:, 0],
            "umap2": obs_emb[:, 1],
            "time": time,
            "cell_type": labels,
        }
    )

    target_mask = np.isclose(time, TARGET_TIME)
    gt_target = protein[target_mask]
    gt_target_xy = obs_emb[target_mask]
    label_centers: dict[str, tuple[float, float]] = {}
    for label in PHENO_ORDER:
        mask = target_mask & (labels == label)
        if int(mask.sum()) >= 10:
            center = np.nanmedian(obs_emb[mask], axis=0)
            label_centers[label] = (float(center[0]), float(center[1]))

    trajectories = {name: load_trajectory(spec["trajectory"]) for name, spec in METHODS.items()}
    metric_rows = []
    locked_metric_means = {}
    generated_rows = []
    generated_xy_by_method = {}
    for method, traj in trajectories.items():
        rows, locked_mean = load_locked_protein_energy(
            method,
            METHODS[method]["metrics"],
            traj,
        )
        metric_rows.extend(rows)
        locked_metric_means[method] = locked_mean
        tt = nearest_time(traj["points"], TARGET_TIME)
        pts = traj["points"][tt]
        w = traj["weights"].get(tt)
        idx = sample_idx(pts.shape[0], 1200, w)
        pred = pts[idx, RNA_DIM : RNA_DIM + PROTEIN_DIM]
        pred_xy = transform_embedding(scaler, reducer, pred)
        generated_xy_by_method[method] = pred_xy
        for i, (particle, xy) in enumerate(zip(idx, pred_xy)):
            generated_rows.append(
                {
                    "method": method,
                    "target_time": TARGET_TIME,
                    "particle": int(particle),
                    "umap1": float(xy[0]),
                    "umap2": float(xy[1]),
                    "weight": float(w[particle]) if w is not None and len(w) > int(particle) else np.nan,
                }
            )

    metric_df = pd.DataFrame(metric_rows)
    obs_target_df = obs_df[target_mask].copy()
    gen_df = pd.DataFrame(generated_rows)
    metric_df.to_csv(TABLE_DIR / "fig01_protein_energy_by_method_time.csv", index=False)
    obs_target_df.to_csv(TABLE_DIR / "fig01_gt_protein_umap_target_time.csv", index=False)
    gen_df.to_csv(TABLE_DIR / "fig01_generated_protein_umap_target_time.csv", index=False)

    all_xy = np.vstack([obs_emb] + list(generated_xy_by_method.values()))
    xlim = tuple(np.nanpercentile(all_xy[:, 0], [0.4, 99.6]))
    ylim = tuple(np.nanpercentile(all_xy[:, 1], [0.4, 99.6]))
    x_pad = 0.05 * (xlim[1] - xlim[0])
    y_pad = 0.05 * (ylim[1] - ylim[0])
    xlim = (xlim[0] - x_pad, xlim[1] + x_pad)
    ylim = (ylim[0] - y_pad, ylim[1] + y_pad)
    methods = list(METHODS)
    # Publication-style small multiples.  Figure-level interpretation belongs
    # in the caption, leaving the graphic to show data, coordinates and metrics.
    fig = plt.figure(figsize=(15.4, 7.2))
    panel_w, gap = 0.275, 0.040
    left = 0.060
    top_y, top_h = 0.585, 0.335
    ax_rna = fig.add_axes([left, top_y, panel_w, top_h])
    ax_protein = fig.add_axes([left + panel_w + gap, top_y, panel_w, top_h])
    ax_gt = fig.add_axes([left + 2 * (panel_w + gap), top_y, panel_w, top_h])

    scatter_cell_types(ax_rna, rna_emb, labels, s=8.0, alpha=0.70, annotate=False)
    ax_rna.set_title("Observed RNA", loc="left", fontsize=10.0, weight="bold", color="#111827")

    scatter_cell_types(ax_protein, obs_emb, labels, s=8.0, alpha=0.70, annotate=False)
    ax_protein.set_title("Observed surface proteins", loc="left", fontsize=10.0, weight="bold", color="#111827")

    scatter_cell_types(ax_gt, gt_target_xy, labels[target_mask], s=8.0, alpha=0.70, annotate=False)
    ax_gt.set_title(f"Observed target, t{int(TARGET_TIME)} (n={int(target_mask.sum()):,})", loc="left", fontsize=10.0, weight="bold", color="#111827")

    bottom_y, bottom_h = 0.115, 0.335
    method_axes = []
    for i, method in enumerate(methods):
        ax = fig.add_axes([left + i * (panel_w + gap), bottom_y, panel_w, bottom_h])
        ax.scatter(gt_target_xy[:, 0], gt_target_xy[:, 1], s=6.5, color="#B8C0CC", alpha=0.38, linewidths=0, rasterized=True)
        gen_xy = generated_xy_by_method[method]
        ax.scatter(gen_xy[:, 0], gen_xy[:, 1], s=8.0, color=METHODS[method]["color"], alpha=0.34, linewidths=0, rasterized=True)
        ax.set_title(method, loc="left", fontsize=10.0, weight="bold", color="#111827")
        method_axes.append(ax)

    for ax in [ax_protein, ax_gt] + method_axes:
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
    for ax in [ax_rna, ax_protein, ax_gt] + method_axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#9CA3AF")
            spine.set_linewidth(0.7)
    ax_rna.set_xlabel(f"RNA {rna_emb_name} 1", fontsize=8.5)
    ax_rna.set_ylabel(f"RNA {rna_emb_name} 2", fontsize=8.5)
    for ax in [ax_protein, ax_gt] + method_axes:
        ax.set_xlabel(f"surface-protein {emb_name} 1", fontsize=8.5)
        ax.set_ylabel(f"surface-protein {emb_name} 2", fontsize=8.5)

    panel_letters = ["a", "b", "c", "d", "e", "f"]
    for letter, ax in zip(panel_letters, [ax_rna, ax_protein, ax_gt] + method_axes):
        ax.text(-0.09, 1.06, letter, transform=ax.transAxes, fontsize=11.0, weight="bold", ha="left", va="bottom", color="#111827")

    present_cell_types = [label for label in PHENO_ORDER if label in set(labels)]
    celltype_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PHENO_COLORS[label], markeredgecolor="none", markersize=5.2, label=label)
        for label in present_cell_types
    ]
    fig.legend(handles=celltype_handles, frameon=False, fontsize=7.2, ncol=len(celltype_handles), loc="upper center", bbox_to_anchor=(0.5, 0.545))

    png = FIG_DIR / "fig01_gt_vs_generated_protein_umap_distribution.png"
    png_str, pdf_str = save_figure_outputs(fig, png, dpi=320, vector_pdf=True)
    plt.close(fig)

    # Keep the quantitative comparisons as a separate figure so that the
    # manifold panels remain large enough to inspect at manuscript scale.
    metric_fig, (ax_metric, ax_line) = plt.subplots(
        1,
        2,
        figsize=(10.8, 3.7),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [0.92, 1.08]},
    )
    mean_metric = pd.Series(locked_metric_means, dtype=float).reindex(methods)
    ordered_methods = list(mean_metric.sort_values().index)
    y = np.arange(len(ordered_methods))
    for yy, method in zip(y, ordered_methods):
        value = float(mean_metric.loc[method])
        ax_metric.plot([1e-2, value], [yy, yy], color="#D1D5DB", lw=1.0, zorder=1)
        ax_metric.scatter(value, yy, s=32, color=METHODS[method]["color"], edgecolor="white", linewidth=0.5, zorder=2)
        ax_metric.text(value * 1.08, yy, f"{value:.3f}", va="center", fontsize=7.4, color="#374151")
    ax_metric.set_yticks(y)
    ax_metric.set_yticklabels(ordered_methods, fontsize=7.8)
    ax_metric.set_xscale("log")
    ax_metric.set_xlim(1e-2, 3e-1)
    ax_metric.set_xlabel("mean protein energy distance, t1-t4", fontsize=8.5)
    ax_metric.set_title("Mean distribution error", loc="left", fontsize=10.0, weight="bold", pad=5)
    ax_metric.grid(axis="x", color="#E5E7EB", lw=0.55)
    ax_metric.spines[["top", "right", "left"]].set_visible(False)
    ax_metric.text(-0.13, 1.08, "a", transform=ax_metric.transAxes, fontsize=11.0, weight="bold", ha="left", va="bottom")

    for method in methods:
        d = metric_df[metric_df["method"] == method].sort_values("target_time")
        ax_line.plot(d["target_time"], d["protein_energy"], marker="o", ms=4.0, lw=1.6, color=METHODS[method]["color"], label=method)
    ax_line.set_yscale("log")
    ax_line.set_xticks(VALIDATION_TIMES)
    ax_line.set_xlabel("target time", fontsize=8.5)
    ax_line.set_ylabel("protein energy distance", fontsize=8.5)
    ax_line.set_title("Distribution error by target time", loc="left", fontsize=10.0, weight="bold", pad=5)
    ax_line.grid(axis="y", color="#E5E7EB", lw=0.55)
    ax_line.spines[["top", "right"]].set_visible(False)
    ax_line.legend(frameon=False, fontsize=7.4, ncol=1, loc="upper left")
    ax_line.text(-0.13, 1.08, "b", transform=ax_line.transAxes, fontsize=11.0, weight="bold", ha="left", va="bottom")

    metric_png = FIG_DIR / "fig01_protein_distribution_error_summary.png"
    metric_png_str, metric_pdf_str = save_figure_outputs(metric_fig, metric_png, dpi=320, vector_pdf=True)
    plt.close(metric_fig)

    real_png, real_pdf = save_real_landscape_reference(
        rna_emb=rna_emb,
        protein_emb=obs_emb,
        labels=labels,
        time=time,
        rna_emb_name=rna_emb_name,
        protein_emb_name=emb_name,
    )
    timecourse_png, timecourse_pdf, timecourse_table = save_generated_timecourse(
        trajectories=trajectories,
        metric_df=metric_df,
        obs_emb=obs_emb,
        protein=protein,
        time=time,
        scaler=scaler,
        reducer=reducer,
        emb_name=emb_name,
    )
    pca_png, pca_pdf, pca_table = save_gt_pca_comparison(
        trajectories=trajectories,
        metric_df=metric_df,
        protein=protein,
        labels=labels,
        time=time,
    )
    pca_timecourse_png, pca_timecourse_pdf, pca_timecourse_table = save_generated_pca_timecourse(
        trajectories=trajectories,
        metric_df=metric_df,
        protein=protein,
        time=time,
    )
    rna_stream = save_rna_scvelo_streams(rna_emb)

    manifest = {
        "analysis": "fig01_gt_vs_generated_protein_umap_distribution",
        "question": "Do RNA-generated protein states land on the observed CITE protein manifold?",
        "evidence_skill": "downstream-prediction",
        "data": str(DATA_PATH),
        "target_time": TARGET_TIME,
        "validation_times": VALIDATION_TIMES,
        "embedding": {
            "protein_embedding": emb_name,
            "rna_embedding": rna_emb_name,
            "fit_on": "observed GT protein PCA for protein panels; observed GT RNA SVD for the RNA reference panel",
            "generated_projection": "UMAP panels use reducer.transform on generated locked-rollout protein block; PCA panel directly plots protein block PC1/PC2",
            "umap_transform_note": "UMAP out-of-sample transform places generated points relative to the GT-fitted neighbor graph and is nonlinear; distances are interpreted only qualitatively.",
        },
        "feature_space": {
            "observed_gt": "adata.obsm['X_protein_pca']",
            "generated": "evaluation_trajectory points[:, 50:80]",
            "rna_dim": RNA_DIM,
            "protein_dim": PROTEIN_DIM,
        },
        "methods": {
            m: {
                "trajectory": str(spec["trajectory"]),
                "metrics": str(spec["metrics"]),
                "role": spec["role"],
            }
            for m, spec in METHODS.items()
        },
        "metrics": {
            "table": str(TABLE_DIR / "fig01_protein_energy_by_method_time.csv"),
            "definition": "energy distance in protein PCA space against matched observed target time; lower is better",
        },
        "figures": {
            "main_gt_vs_generated_t4": {"png": png_str, "pdf": pdf_str},
            "protein_distribution_error_summary": {"png": metric_png_str, "pdf": metric_pdf_str},
            "observed_rna_protein_reference": {"png": real_png, "pdf": real_pdf},
            "generated_timecourse_by_method": {"png": timecourse_png, "pdf": timecourse_pdf},
            "linear_gt_pca_comparison": {"png": pca_png, "pdf": pca_pdf},
            "linear_gt_pca_timecourse_by_method": {"png": pca_timecourse_png, "pdf": pca_timecourse_pdf},
            "rna_umap_scvelo_stream_by_celltype": {"png": rna_stream["celltype_png"]},
            "rna_umap_scvelo_stream_by_time": {"png": rna_stream["time_png"]},
        },
        "tables": {
            "gt_umap_target": str(TABLE_DIR / "fig01_gt_protein_umap_target_time.csv"),
            "generated_umap_target": str(TABLE_DIR / "fig01_generated_protein_umap_target_time.csv"),
            "generated_umap_all_times": timecourse_table,
            "generated_gt_pca_target": pca_table,
            "generated_gt_pca_all_times": pca_timecourse_table,
        },
        "velocity_stream": {
            "manifest": rna_stream["manifest"],
            "basis": "observed RNA UMAP",
            "state_space": "X_rna_svd",
            "velocity_space": "runtime v_theta([X_rna_svd, zero_protein_block, time])[:, :50]",
            "evidence_type": "local flow diagnostic",
        },
        "warnings": [
            "UMAP is only a visualization coordinate and is not used to train or alter generated predictions.",
            "UMAP transform is nonlinear out-of-sample placement, not a linear projection or metric-preserving map.",
            "The PCA comparison panel directly uses the shared protein PCA coordinate system and should be used as the transparent sanity check.",
            "RNA UMAP stream plots are scVelo-style local flow diagnostics from the ProteinBridge runtime RNA velocity and are not continuous trajectory evidence.",
            "ProteinBridge rollout integrates 50D RNA states; protein states are generated by h_eta(r,t), not by a 30D protein velocity field.",
            "GT target snapshots are used for validation and manifold visualization, not for constructing the generated protein states.",
            "This figure validates distribution realism; it is not by itself a biological mechanism claim.",
        ]
        + list(rna_stream.get("warnings", [])),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    best = mean_metric.idxmin()
    STORY_PATH.write_text(
        "\n".join(
            [
                "# ProteinBridge Biology Landscape Discovery",
                "",
                "## Fig. 1: GT Protein UMAP Validation",
                "",
                "Question: do RNA-generated protein states land on the observed CITE protein manifold?",
                "",
                "How it was made:",
                "",
                "1. Fit the 2D protein UMAP using only observed GT `X_protein_pca`.",
                "2. Separately fit an observed RNA UMAP from GT `X_rna_svd`, only to show what the real paired RNA landscape looks like.",
                "3. Read each method's locked `evaluation_trajectory.npz`.",
                "4. Extract the generated protein block `points[:, 50:80]` at matched target times.",
                "5. Project generated protein states into the GT-trained protein UMAP.",
                "6. In the comparison panels, gray dots are observed GT t4 protein cells and colored dots are generated protein particles.",
                "7. Also draw a linear PC1/PC2 sanity-check panel in the original protein PCA coordinate system.",
                "8. Compare generated and observed distributions with protein energy distance in protein PCA space.",
                "",
                f"At t4, the main figure overlays each generated distribution on the same observed surface-protein map. Across t1-t4, the lowest mean protein energy distance is `{best}`.",
                "",
                "Additional split panels:",
                "",
                f"- Observed RNA/protein landscapes by cell type and time: `{real_png}`",
                f"- Generated protein distributions at t0-t4 for each method: `{timecourse_png}`",
                f"- Direct GT protein PCA comparison: `{pca_png}`",
                f"- Direct GT protein PCA timecourse: `{pca_timecourse_png}`",
                f"- RNA UMAP scVelo stream by cell type: `{rna_stream['celltype_png']}`",
                f"- RNA UMAP scVelo stream by time: `{rna_stream['time_png']}`",
                "",
                "Velocity provenance note:",
                "",
                "The RNA UMAP scVelo stream uses the ProteinBridge runtime RNA velocity `v_theta([X_rna_svd, zero_protein_block, time])[:, :50]`. ProteinBridge rollout integrates 50D RNA states and generates protein states with `h_eta(r,t)`; the 30D protein block of the generic `velocity_latent` interface is not a protein velocity field.",
                "",
                f"Figure: `{png}`",
                f"Manifest: `{MANIFEST_PATH}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "png": str(png),
        "pdf": pdf_str,
        "observed_reference_png": real_png,
        "timecourse_png": timecourse_png,
        "pca_comparison_png": pca_png,
        "pca_timecourse_png": pca_timecourse_png,
        "rna_stream_celltype_png": rna_stream["celltype_png"],
        "rna_stream_time_png": rna_stream["time_png"],
        "metric_table": str(TABLE_DIR / "fig01_protein_energy_by_method_time.csv"),
        "manifest": str(MANIFEST_PATH),
        "story": str(STORY_PATH),
    }


def main() -> None:
    result = build_figure()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
