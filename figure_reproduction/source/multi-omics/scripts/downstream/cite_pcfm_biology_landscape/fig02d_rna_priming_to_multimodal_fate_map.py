#!/usr/bin/env python3
"""RNA priming map colored by future multimodal cell-state fate.

This is a companion to the protein-only Fig02 fate map. It uses the full
generated CITE-PCFM state [RNA(50), protein(30)] at terminal time and assigns
fate by kNN in observed joint latent space. Use this when the question is
"future cell state" rather than strictly "future surface-protein phenotype".
"""

from __future__ import annotations

import json
import os
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
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from multistageot_style import (
    PHENO_COLORS,
    PHENO_ORDER,
    add_median_labels,
    clean_embedding_axis,
    note_box,
    scatter_background,
    scatter_groups,
    setup,
    title_left,
)

try:
    import umap  # type: ignore

    HAVE_UMAP = True
except Exception:
    HAVE_UMAP = False


ROOT = Path("/Users/zhenyizhang/Documents/GitHub/CytoBridge-agent-runtime-v2-20260309/multi-omics")
DATA_PATH = Path("/Users/zhenyizhang/.cellcompass/algorithm_benchmarks/datasets/gse305370_citeseq_stage2_10k/data.h5ad")
TRAJECTORY_PATH = ROOT / "training_runs/20260526T174204321821Z__final__cite_paired_coord_fm__campaign_20260-trial_20260526-gse305370_ci/artifacts/evaluation_trajectory.npz"
FIG_DIR = ROOT / "figures/cite_pcfm_deep_story/biology_landscape_discovery"
OUT_DIR = ROOT / "downstream/cite_pcfm_deep_story/biology_landscape_discovery_rna_priming_multimodal_fate"

RNA_DIM = 50
PROTEIN_DIM = 30
JOINT_DIM = RNA_DIM + PROTEIN_DIM
KNN_K = 25
RANDOM_STATE = 20260528
setup()


def fit_rna_embedding(rna: np.ndarray) -> tuple[str, np.ndarray]:
    x = StandardScaler().fit_transform(rna)
    if HAVE_UMAP:
        reducer = umap.UMAP(
            n_neighbors=40,
            min_dist=0.28,
            metric="euclidean",
            random_state=RANDOM_STATE,
            low_memory=True,
        )
        return "UMAP", np.asarray(reducer.fit_transform(x), dtype=np.float32)
    reducer = PCA(n_components=2, random_state=RANDOM_STATE)
    return "PCA fallback", np.asarray(reducer.fit_transform(x), dtype=np.float32)


def load_points(path: Path) -> tuple[np.ndarray, list[np.ndarray]]:
    z = np.load(path)
    time_points = np.asarray(z["time_points"], dtype=float)
    points = [np.asarray(z[f"points_{i:06d}"], dtype=np.float32) for i in range(time_points.shape[0])]
    return time_points, points


def knn_probabilities(generated: np.ndarray, observed_ref: np.ndarray, labels_ref: np.ndarray, k: int) -> pd.DataFrame:
    scaler = StandardScaler().fit(observed_ref)
    ref = scaler.transform(observed_ref)
    gen = scaler.transform(generated)
    k_eff = int(min(k, ref.shape[0]))
    nn = NearestNeighbors(n_neighbors=k_eff, metric="euclidean").fit(ref)
    _, idx = nn.kneighbors(gen)
    probs = np.zeros((generated.shape[0], len(PHENO_ORDER)), dtype=np.float32)
    label_to_col = {label: i for i, label in enumerate(PHENO_ORDER)}
    for i, row in enumerate(idx):
        vals, counts = np.unique(labels_ref[row].astype(str), return_counts=True)
        for val, count in zip(vals, counts):
            if val in label_to_col:
                probs[i, label_to_col[val]] = float(count) / float(k_eff)
    top = probs.max(axis=1)
    sorted_probs = np.sort(probs, axis=1)
    eps = 1e-8
    out = pd.DataFrame(probs, columns=[f"future_{label}_prob" for label in PHENO_ORDER])
    out["terminal_multimodal_basin"] = np.asarray(PHENO_ORDER, dtype=object)[np.argmax(probs, axis=1)]
    out["terminal_multimodal_basin_prob"] = top
    out["terminal_multimodal_basin_margin"] = sorted_probs[:, -1] - sorted_probs[:, -2]
    out["terminal_multimodal_fate_entropy"] = -np.sum(probs * np.log(probs + eps), axis=1) / np.log(float(len(PHENO_ORDER)))
    return out


def plot_joint_map(obs_xy: np.ndarray, labels: np.ndarray, t0_xy: np.ndarray, fate: pd.DataFrame, emb_name: str, out_png: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.35), constrained_layout=True)

    ax = axes[0]
    scatter_background(ax, obs_xy, s=8.5, alpha=0.92, zorder=0)
    scatter_groups(ax, obs_xy, labels, PHENO_COLORS, order=PHENO_ORDER, s=9.0, alpha=0.78, zorder=2)
    add_median_labels(ax, obs_xy, labels, order=PHENO_ORDER, color_map=PHENO_COLORS, fontsize=8.2)
    title_left(ax, "observed RNA landscape", fontsize=12.5)
    note_box(ax, "source cell_type annotation")
    clean_embedding_axis(ax)

    ax = axes[1]
    scatter_background(ax, obs_xy, s=7.5, alpha=0.88, zorder=0)
    terminal = fate["terminal_multimodal_basin"].astype(str).to_numpy()
    scatter_groups(ax, t0_xy, terminal, PHENO_COLORS, order=PHENO_ORDER, s=16.0, alpha=0.86, zorder=3)
    add_median_labels(ax, obs_xy, labels, order=PHENO_ORDER, color_map=PHENO_COLORS, fontsize=8.2)
    title_left(ax, "t0 RNA states colored by future multimodal state", fontsize=12.5)
    note_box(ax, "color = generated t4 [RNA, protein] kNN basin")
    handles = [Line2D([0], [0], marker="o", linestyle="None", color="none", markerfacecolor=PHENO_COLORS[label], markersize=6, label=label) for label in PHENO_ORDER if np.any(terminal == label)]
    ax.legend(handles=handles, frameon=False, fontsize=8, ncol=2, loc="lower left")
    clean_embedding_axis(ax)

    ax = axes[2]
    scatter_background(ax, obs_xy, s=7.5, alpha=0.86, zorder=0)
    sc = ax.scatter(
        t0_xy[:, 0],
        t0_xy[:, 1],
        c=fate["terminal_multimodal_basin_prob"].to_numpy(),
        cmap="magma",
        vmin=0,
        vmax=1,
        s=15,
        alpha=0.88,
        linewidths=0,
        rasterized=True,
    )
    add_median_labels(ax, obs_xy, labels, order=PHENO_ORDER, color_map=PHENO_COLORS, fontsize=8.2)
    title_left(ax, "future multimodal fate certainty from t0 RNA", fontsize=12.5)
    note_box(ax, "color = top joint-basin vote fraction")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.042, pad=0.012)
    cbar.set_label("terminal multimodal-basin probability", fontsize=8.5)
    cbar.ax.tick_params(labelsize=8)
    clean_embedding_axis(ax)

    fig.suptitle("RNA priming to future multimodal cell state", x=0.01, ha="left", fontsize=15.5, weight="bold")
    fig.savefig(out_png, dpi=330, bbox_inches="tight", facecolor="white")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> dict:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(DATA_PATH)
    rna = np.asarray(adata.obsm["X_rna_svd"], dtype=np.float32)[:, :RNA_DIM]
    latent = np.asarray(adata.obsm["X_latent"], dtype=np.float32)[:, :JOINT_DIM]
    labels = adata.obs["cell_type"].astype(str).to_numpy()
    times = np.asarray(adata.obs["time_point_processed"], dtype=float)

    time_points, points_by_time = load_points(TRAJECTORY_PATH)
    t0 = float(time_points[0])
    t_terminal = float(time_points[-1])
    t0_idx = np.where(np.isclose(times, t0))[0]
    terminal_idx = np.where(np.isclose(times, t_terminal))[0]
    if t0_idx.shape[0] != points_by_time[0].shape[0]:
        raise ValueError(f"t0 observed count {t0_idx.shape[0]} != trajectory particles {points_by_time[0].shape[0]}")

    emb_name, obs_xy = fit_rna_embedding(rna)
    t0_xy = obs_xy[t0_idx]
    fate = knn_probabilities(points_by_time[-1][:, :JOINT_DIM], latent[terminal_idx], labels[terminal_idx], KNN_K)
    fate.insert(0, "particle_id", np.arange(fate.shape[0], dtype=int))
    fate.insert(1, "init_obs_name", np.asarray(adata.obs_names[t0_idx].astype(str)))
    fate.insert(2, "initial_source_cell_type", labels[t0_idx])
    fate.insert(3, "rna_umap_1", t0_xy[:, 0])
    fate.insert(4, "rna_umap_2", t0_xy[:, 1])

    particle_table = OUT_DIR / "rna_priming_future_multimodal_fate_particles.csv"
    fate.to_csv(particle_table, index=False)
    counts = fate["terminal_multimodal_basin"].value_counts().reindex(PHENO_ORDER).fillna(0).astype(int)
    summary = pd.DataFrame(
        [
            {
                "terminal_multimodal_basin": label,
                "n_particles": int(counts[label]),
                "fraction_particles": float(counts[label] / fate.shape[0]),
                "mean_probability": float(fate[f"future_{label}_prob"].mean()),
                "max_probability": float(fate[f"future_{label}_prob"].max()),
            }
            for label in PHENO_ORDER
        ]
    )
    summary_table = OUT_DIR / "rna_priming_future_multimodal_fate_summary.csv"
    summary.to_csv(summary_table, index=False)

    out_png = FIG_DIR / "fig02d_rna_priming_future_multimodal_fate_map.png"
    plot_joint_map(obs_xy, labels, t0_xy, fate, emb_name, out_png)

    manifest = {
        "analysis": "rna_priming_to_future_multimodal_fate_map",
        "data_h5ad": str(DATA_PATH),
        "trajectory_path": str(TRAJECTORY_PATH),
        "fate_assignment": {
            "generated_terminal_state": "evaluation_trajectory terminal points[:, 0:80]",
            "reference": "observed terminal-time cells in adata.obsm['X_latent']",
            "method": "kNN vote probabilities in observed joint [RNA, protein] latent space",
            "k": KNN_K,
            "classes": PHENO_ORDER,
            "terminal_time": t_terminal,
            "terminal_counts": counts.to_dict(),
        },
        "source_order_check": {
            "t0_observed_count": int(t0_idx.shape[0]),
            "t0_trajectory_particle_count": int(points_by_time[0].shape[0]),
            "rna_block_mean_abs_diff_points0_vs_observed_t0": float(np.mean(np.abs(points_by_time[0][:, :RNA_DIM] - rna[t0_idx]))),
            "rna_block_max_abs_diff_points0_vs_observed_t0": float(np.max(np.abs(points_by_time[0][:, :RNA_DIM] - rna[t0_idx]))),
        },
        "figures": {"joint_fate_map_png": str(out_png), "joint_fate_map_pdf": str(out_png.with_suffix(".pdf"))},
        "tables": {"particle_fate_probabilities": str(particle_table), "fate_summary": str(summary_table)},
        "interpretation_boundary": [
            "This is a multimodal cell-state readout, not a strict protein phenotype readout.",
            "It should be used for cell-type/state proportions when source labels are RNA-derived or multimodal in practice.",
            "The protein-only Fig02 remains the stricter test of generated surface phenotype.",
        ],
    }
    manifest_path = OUT_DIR / "rna_priming_future_multimodal_fate_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(json.dumps(
        {
            "joint_fate_map_png": str(out_png),
            "particle_table": str(particle_table),
            "summary_table": str(summary_table),
            "manifest": str(manifest_path),
            "terminal_counts": counts.to_dict(),
            "source_order_check": manifest["source_order_check"],
        },
        indent=2,
    ))
    return manifest


if __name__ == "__main__":
    main()
