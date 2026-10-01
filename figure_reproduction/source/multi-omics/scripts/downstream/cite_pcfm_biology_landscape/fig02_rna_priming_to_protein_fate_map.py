#!/usr/bin/env python3
"""Map initial RNA states to future CITE-PCFM protein fate probabilities.

This is a focused downstream analysis for the CITE-PCFM biological story:
given the same t0 RNA-initialized particles, assign their generated terminal
protein states to observed protein-space basins, then paint those future
protein fates back onto the t0 RNA landscape.
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
OUT_DIR = ROOT / "downstream/cite_pcfm_deep_story/biology_landscape_discovery_rna_priming_fate_map"

RNA_DIM = 50
PROTEIN_DIM = 30
KNN_K = 25
RANDOM_STATE = 20260528
setup()


def fit_embedding(x: np.ndarray, *, n_neighbors: int, min_dist: float, seed: int) -> tuple[str, np.ndarray]:
    scaler = StandardScaler().fit(x)
    x_scaled = scaler.transform(x)
    if HAVE_UMAP:
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric="euclidean",
            random_state=seed,
            low_memory=True,
        )
        emb = reducer.fit_transform(x_scaled)
        return "UMAP", np.asarray(emb, dtype=np.float32)
    reducer = PCA(n_components=2, random_state=seed)
    emb = reducer.fit_transform(x_scaled)
    return "PCA fallback", np.asarray(emb, dtype=np.float32)


def load_points(path: Path) -> tuple[np.ndarray, list[np.ndarray]]:
    z = np.load(path)
    time_points = np.asarray(z["time_points"], dtype=float)
    points = [np.asarray(z[f"points_{i:06d}"], dtype=np.float32) for i in range(time_points.shape[0])]
    return time_points, points


def protein_knn_fate_probabilities(
    generated_terminal_protein: np.ndarray,
    observed_terminal_protein: np.ndarray,
    observed_terminal_labels: np.ndarray,
    *,
    k: int,
) -> pd.DataFrame:
    k_eff = int(min(k, observed_terminal_protein.shape[0]))
    nn = NearestNeighbors(n_neighbors=k_eff, metric="euclidean").fit(observed_terminal_protein)
    _, idx = nn.kneighbors(generated_terminal_protein)
    label_to_col = {label: i for i, label in enumerate(PHENO_ORDER)}
    probs = np.zeros((generated_terminal_protein.shape[0], len(PHENO_ORDER)), dtype=np.float32)
    for row_i, neighbors in enumerate(idx):
        vals, counts = np.unique(observed_terminal_labels[neighbors].astype(str), return_counts=True)
        for val, count in zip(vals, counts):
            if val in label_to_col:
                probs[row_i, label_to_col[val]] = float(count) / float(k_eff)
    max_prob = probs.max(axis=1)
    sorted_probs = np.sort(probs, axis=1)
    margin = sorted_probs[:, -1] - sorted_probs[:, -2]
    eps = 1e-8
    entropy = -np.sum(probs * np.log(probs + eps), axis=1) / np.log(float(len(PHENO_ORDER)))
    terminal = np.asarray(PHENO_ORDER, dtype=object)[np.argmax(probs, axis=1)]
    out = pd.DataFrame(probs, columns=[f"future_{label}_prob" for label in PHENO_ORDER])
    out["terminal_protein_basin"] = terminal
    out["terminal_protein_basin_prob"] = max_prob
    out["terminal_protein_basin_margin"] = margin
    out["terminal_protein_fate_entropy"] = entropy
    return out


def plot_main_map(
    obs_xy: np.ndarray,
    obs_labels: np.ndarray,
    obs_time: np.ndarray,
    t0_xy: np.ndarray,
    fate_df: pd.DataFrame,
    emb_name: str,
    out_png: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.35), constrained_layout=True)

    ax = axes[0]
    scatter_background(ax, obs_xy, s=8.5, alpha=0.92, zorder=0)
    scatter_groups(ax, obs_xy, obs_labels, PHENO_COLORS, order=PHENO_ORDER, s=9.0, alpha=0.78, zorder=2)
    add_median_labels(ax, obs_xy, obs_labels, order=PHENO_ORDER, color_map=PHENO_COLORS, fontsize=8.2)
    title_left(ax, "observed RNA landscape", fontsize=12.5)
    note_box(ax, "source cell_type annotation")
    clean_embedding_axis(ax)

    ax = axes[1]
    scatter_background(ax, obs_xy, s=7.5, alpha=0.88, zorder=0)
    terminal = fate_df["terminal_protein_basin"].astype(str).to_numpy()
    scatter_groups(ax, t0_xy, terminal, PHENO_COLORS, order=PHENO_ORDER, s=16.0, alpha=0.86, zorder=3)
    add_median_labels(ax, obs_xy, obs_labels, order=PHENO_ORDER, color_map=PHENO_COLORS, fontsize=8.2)
    title_left(ax, "t0 RNA states colored by future protein basin", fontsize=12.5)
    note_box(ax, "color = generated t4 protein kNN basin")
    handles = [
        Line2D([0], [0], marker="o", linestyle="None", color="none", markerfacecolor=PHENO_COLORS[label], markersize=6, label=label)
        for label in PHENO_ORDER
        if np.any(terminal == label)
    ]
    ax.legend(handles=handles, frameon=False, fontsize=8, ncol=2, loc="lower left")
    clean_embedding_axis(ax)

    ax = axes[2]
    scatter_background(ax, obs_xy, s=7.5, alpha=0.86, zorder=0)
    sc = ax.scatter(
        t0_xy[:, 0],
        t0_xy[:, 1],
        c=fate_df["terminal_protein_basin_prob"].to_numpy(),
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
        s=15.0,
        alpha=0.88,
        linewidths=0,
        rasterized=True,
    )
    add_median_labels(ax, obs_xy, obs_labels, order=PHENO_ORDER, color_map=PHENO_COLORS, fontsize=8.2)
    title_left(ax, "future protein fate certainty from t0 RNA", fontsize=12.5)
    note_box(ax, "color = top protein-basin vote fraction")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.042, pad=0.012)
    cbar.set_label("terminal protein-basin probability", fontsize=8.5)
    cbar.ax.tick_params(labelsize=8)
    clean_embedding_axis(ax)

    fig.suptitle(
        "RNA priming to future protein fate",
        x=0.01,
        ha="left",
        fontsize=15.5,
        weight="bold",
    )
    fig.savefig(out_png, dpi=330, bbox_inches="tight", facecolor="white")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_probability_facets(
    obs_xy: np.ndarray,
    obs_labels: np.ndarray,
    t0_xy: np.ndarray,
    fate_df: pd.DataFrame,
    emb_name: str,
    out_png: Path,
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(15.2, 7.4), constrained_layout=True)
    axes_flat = axes.ravel()
    last_sc = None
    for ax, label in zip(axes_flat, PHENO_ORDER):
        scatter_background(ax, obs_xy, s=7.0, alpha=0.84, zorder=0)
        prob = fate_df[f"future_{label}_prob"].to_numpy()
        last_sc = ax.scatter(
            t0_xy[:, 0],
            t0_xy[:, 1],
            c=prob,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            s=13.0,
            alpha=0.90,
            linewidths=0,
            rasterized=True,
        )
        add_median_labels(ax, obs_xy, obs_labels, order=PHENO_ORDER, min_cells=80, color_map=PHENO_COLORS, fontsize=7.8)
        title_left(ax, f"future {label} protein probability", fontsize=10.5)
        note_box(ax, f"mean {prob.mean():.2f} | max {prob.max():.2f}", x=0.02, y=0.96)
        clean_embedding_axis(ax)
    axes_flat[-1].axis("off")
    if last_sc is not None:
        cbar = fig.colorbar(last_sc, ax=axes_flat[:-1], fraction=0.022, pad=0.010)
        cbar.set_label("protein-space kNN vote probability", fontsize=8.5)
        cbar.ax.tick_params(labelsize=8)
    fig.suptitle("Future protein fate probabilities are already structured on the initial RNA map", fontsize=15, weight="bold")
    fig.savefig(out_png, dpi=330, bbox_inches="tight", facecolor="white")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_initial_label_to_terminal_basin(
    fate_df: pd.DataFrame,
    out_png: Path,
) -> None:
    tab = pd.crosstab(fate_df["initial_source_cell_type"], fate_df["terminal_protein_basin"])
    tab = tab.reindex(index=[label for label in PHENO_ORDER if label in tab.index], columns=PHENO_ORDER, fill_value=0)
    frac = tab.div(tab.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    fig, ax = plt.subplots(figsize=(8.8, 4.9), constrained_layout=True)
    y = np.arange(frac.shape[0])
    left = np.zeros(frac.shape[0], dtype=float)
    for label in PHENO_ORDER:
        values = frac[label].to_numpy()
        ax.barh(y, values, left=left, color=PHENO_COLORS.get(label, "#64748B"), height=0.72, label=label)
        left += values
    for i, source_label in enumerate(frac.index):
        ax.text(1.01, i, f"n={int(tab.loc[source_label].sum())}", va="center", fontsize=8.5, color="#475569")
    ax.set_yticks(y)
    ax.set_yticklabels(frac.index)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.13)
    ax.set_xlabel("fraction of t0 cells")
    ax.set_title("RNA-derived t0 label versus future protein basin", loc="left", fontsize=12.5, weight="bold")
    ax.text(
        0.012,
        0.985,
        "Rows use source cell_type at t0; colors use generated terminal protein-space basin.",
        transform=ax.transAxes,
        va="top",
        fontsize=8.7,
        color="#334155",
        bbox={"boxstyle": "round,pad=0.14", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="lower center", bbox_to_anchor=(0.50, -0.30))
    fig.savefig(out_png, dpi=330, bbox_inches="tight", facecolor="white")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> dict:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(DATA_PATH)
    rna = np.asarray(adata.obsm["X_rna_svd"], dtype=np.float32)[:, :RNA_DIM]
    protein = np.asarray(adata.obsm["X_protein_pca"], dtype=np.float32)[:, :PROTEIN_DIM]
    labels = adata.obs["cell_type"].astype(str).to_numpy()
    times = np.asarray(adata.obs["time_point_processed"], dtype=float)

    time_points, points_by_time = load_points(TRAJECTORY_PATH)
    t0 = float(time_points[0])
    t_terminal = float(time_points[-1])
    t0_idx = np.where(np.isclose(times, t0))[0]
    terminal_idx = np.where(np.isclose(times, t_terminal))[0]
    if t0_idx.shape[0] != points_by_time[0].shape[0]:
        raise ValueError(f"t0 observed count {t0_idx.shape[0]} != trajectory particles {points_by_time[0].shape[0]}")
    if terminal_idx.shape[0] == 0:
        raise ValueError(f"No observed cells found at terminal time {t_terminal}")

    t0_rna_order_mean_abs_diff = float(np.mean(np.abs(points_by_time[0][:, :RNA_DIM] - rna[t0_idx])))
    t0_rna_order_max_abs_diff = float(np.max(np.abs(points_by_time[0][:, :RNA_DIM] - rna[t0_idx])))

    emb_name, obs_rna_xy = fit_embedding(rna, n_neighbors=40, min_dist=0.28, seed=RANDOM_STATE)
    t0_xy = obs_rna_xy[t0_idx]

    terminal_protein = points_by_time[-1][:, RNA_DIM : RNA_DIM + PROTEIN_DIM]
    fate_prob = protein_knn_fate_probabilities(
        terminal_protein,
        protein[terminal_idx],
        labels[terminal_idx],
        k=KNN_K,
    )
    fate_prob.insert(0, "particle_id", np.arange(fate_prob.shape[0], dtype=int))
    fate_prob.insert(1, "init_obs_name", np.asarray(adata.obs_names[t0_idx].astype(str)))
    fate_prob.insert(2, "initial_source_cell_type", labels[t0_idx])
    fate_prob.insert(3, "rna_umap_1", t0_xy[:, 0])
    fate_prob.insert(4, "rna_umap_2", t0_xy[:, 1])

    particle_table = OUT_DIR / "rna_priming_future_protein_fate_particles.csv"
    fate_prob.to_csv(particle_table, index=False)

    summary_rows = []
    terminal_counts = fate_prob["terminal_protein_basin"].value_counts().reindex(PHENO_ORDER).fillna(0).astype(int)
    for label in PHENO_ORDER:
        summary_rows.append(
            {
                "terminal_protein_basin": label,
                "n_particles": int(terminal_counts[label]),
                "fraction_particles": float(terminal_counts[label] / fate_prob.shape[0]),
                "mean_probability": float(fate_prob[f"future_{label}_prob"].mean()),
                "max_probability": float(fate_prob[f"future_{label}_prob"].max()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary_table = OUT_DIR / "rna_priming_future_protein_fate_summary.csv"
    summary.to_csv(summary_table, index=False)

    cross_tab_table = OUT_DIR / "rna_initial_label_to_terminal_protein_basin_counts.csv"
    pd.crosstab(fate_prob["initial_source_cell_type"], fate_prob["terminal_protein_basin"]).reindex(
        index=[label for label in PHENO_ORDER if label in set(fate_prob["initial_source_cell_type"])],
        columns=PHENO_ORDER,
        fill_value=0,
    ).to_csv(cross_tab_table)

    main_png = FIG_DIR / "fig02a_rna_priming_future_protein_fate_map.png"
    facets_png = FIG_DIR / "fig02b_rna_priming_future_protein_fate_probabilities.png"
    stacked_png = FIG_DIR / "fig02c_rna_initial_label_to_terminal_protein_basin.png"
    plot_main_map(obs_rna_xy, labels, times, t0_xy, fate_prob, emb_name, main_png)
    plot_probability_facets(obs_rna_xy, labels, t0_xy, fate_prob, emb_name, facets_png)
    plot_initial_label_to_terminal_basin(fate_prob, stacked_png)

    manifest = {
        "analysis": "rna_priming_to_future_protein_fate_map",
        "skill": "downstream-prediction / trajectory-fate style",
        "evidence_type": "locked CITE-PCFM generated terminal protein states mapped back to t0 RNA states",
        "data_h5ad": str(DATA_PATH),
        "trajectory_path": str(TRAJECTORY_PATH),
        "rna_display_basis": f"observed RNA {emb_name} fitted from adata.obsm['X_rna_svd']",
        "fate_assignment": {
            "generated_terminal_state": "evaluation_trajectory terminal points[:, 50:80]",
            "reference": "observed terminal-time cells in adata.obsm['X_protein_pca']",
            "method": "kNN vote probabilities in observed protein PCA space",
            "k": KNN_K,
            "classes": PHENO_ORDER,
            "terminal_time": t_terminal,
            "terminal_counts": terminal_counts.to_dict(),
        },
        "source_order_check": {
            "t0_observed_count": int(t0_idx.shape[0]),
            "t0_trajectory_particle_count": int(points_by_time[0].shape[0]),
            "rna_block_mean_abs_diff_points0_vs_observed_t0": t0_rna_order_mean_abs_diff,
            "rna_block_max_abs_diff_points0_vs_observed_t0": t0_rna_order_max_abs_diff,
        },
        "figures": {
            "main_map_png": str(main_png),
            "main_map_pdf": str(main_png.with_suffix(".pdf")),
            "probability_facets_png": str(facets_png),
            "probability_facets_pdf": str(facets_png.with_suffix(".pdf")),
            "initial_label_to_terminal_basin_png": str(stacked_png),
            "initial_label_to_terminal_basin_pdf": str(stacked_png.with_suffix(".pdf")),
        },
        "tables": {
            "particle_fate_probabilities": str(particle_table),
            "fate_summary": str(summary_table),
            "initial_label_to_terminal_basin_counts": str(cross_tab_table),
        },
        "interpretation_boundary": [
            "The RNA UMAP is a display coordinate for t0 states in the observed RNA landscape; fate probabilities are computed in protein PCA space, not in UMAP space.",
            "Source cell_type labels are dataset annotations used to name observed protein-space basins; terminal protein fate is assigned from generated protein vectors by kNN against observed terminal protein cells.",
            "Observed terminal protein cells label the generated terminal states but do not alter the CITE-PCFM rollout trajectory.",
        ],
    }
    manifest_path = OUT_DIR / "rna_priming_future_protein_fate_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    print(json.dumps(
        {
            "main_map_png": str(main_png),
            "probability_facets_png": str(facets_png),
            "initial_label_to_terminal_basin_png": str(stacked_png),
            "particle_table": str(particle_table),
            "summary_table": str(summary_table),
            "manifest": str(manifest_path),
            "source_order_check": manifest["source_order_check"],
            "terminal_counts": terminal_counts.to_dict(),
        },
        indent=2,
    ))
    return manifest


if __name__ == "__main__":
    main()
