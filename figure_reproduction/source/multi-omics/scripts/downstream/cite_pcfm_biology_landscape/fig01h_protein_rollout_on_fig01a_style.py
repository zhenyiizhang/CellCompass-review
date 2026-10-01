#!/usr/bin/env python3
"""Panel-scale CITE-PCFM protein rollout on the Fig01A observed protein UMAP style.

This redraws the trajectory panel instead of cropping the older Fig01G. The
display basis is the observed protein PCA UMAP used for Fig01A. Generated
CITE-PCFM sampled particle IDs and terminal labels are read from the audited
Fig01G coordinate table, then the corresponding generated protein states are
transformed into the freshly fitted observed protein UMAP. Only terminal basin
median paths plus a small number of faint particle paths are shown, so the
observed protein phenotype landscape remains readable.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/cytobridge_mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/cytobridge_xdg_cache")
os.environ.setdefault("NUMBA_CACHE_DIR", "/private/tmp/numba_cache")
os.environ.setdefault("NUMBA_DISABLE_CACHE", "1")
for key in ["MPLCONFIGDIR", "XDG_CACHE_HOME", "NUMBA_CACHE_DIR"]:
    Path(os.environ[key]).mkdir(parents=True, exist_ok=True)

import anndata as ad
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from multistageot_style import PHENO_COLORS, PHENO_ORDER, add_median_labels, clean_embedding_axis, setup

try:
    import umap  # type: ignore

    HAVE_UMAP = True
except Exception:
    HAVE_UMAP = False


ROOT = Path("/Users/zhenyizhang/Documents/GitHub/CytoBridge-agent-runtime-v2-20260309/multi-omics")
DATA_PATH = Path("/Users/zhenyizhang/.cellcompass/algorithm_benchmarks/datasets/gse305370_citeseq_stage2_10k/data.h5ad")
TRAJECTORY_PATH = ROOT / "training_runs/20260526T174204321821Z__final__cite_paired_coord_fm__campaign_20260-trial_20260526-gse305370_ci/artifacts/evaluation_trajectory.npz"
FIG_DIR = ROOT / "figures/cite_pcfm_deep_story/biology_landscape_discovery"
OUT_DIR = ROOT / "downstream/cite_pcfm_deep_story/biology_landscape_discovery_protein_rollout_trajectory"
COORD_TABLE = OUT_DIR / "cite_pcfm_rollout_trajectory_protein_umap_coordinates.csv"

PROTEIN_DIM = 30
RANDOM_STATE = 20260528
RNG = np.random.default_rng(RANDOM_STATE)
MAX_FAINT_PER_BASIN = 6


def fit_observed_protein_umap(protein: np.ndarray) -> tuple[str, StandardScaler, object, np.ndarray]:
    scaler = StandardScaler().fit(protein)
    x_scaled = scaler.transform(protein)
    if HAVE_UMAP:
        reducer = umap.UMAP(
            n_neighbors=35,
            min_dist=0.22,
            metric="euclidean",
            random_state=RANDOM_STATE,
            low_memory=True,
        )
        emb = reducer.fit_transform(x_scaled)
        return "UMAP", scaler, reducer, emb.astype(np.float32)
    reducer = PCA(n_components=2, random_state=RANDOM_STATE)
    emb = reducer.fit_transform(x_scaled)
    return "PCA fallback", scaler, reducer, emb.astype(np.float32)


def draw_observed_fig01a_style(
    ax: plt.Axes,
    xy: np.ndarray,
    labels: np.ndarray,
    *,
    label_order: tuple[str, ...] | None = None,
) -> None:
    """Draw observed protein cells with Fig01A-like colored points and black rims."""
    for label in PHENO_ORDER:
        mask = labels.astype(str) == label
        if not np.any(mask):
            continue
        color = PHENO_COLORS.get(label, "#64748B")
        ax.scatter(
            xy[mask, 0],
            xy[mask, 1],
            s=8.6,
            facecolor=color,
            edgecolor="#111111",
            linewidth=0.28,
            alpha=0.54,
            zorder=1,
        )
    add_median_labels(ax, xy, labels, order=label_order or PHENO_ORDER, min_cells=24, color_map=PHENO_COLORS, fontsize=8.1)


def load_sampled_particle_ids(coord_table: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(coord_table)
    particle_ids = np.asarray(sorted(df["particle_id"].unique()), dtype=int)
    terminal = np.empty(particle_ids.shape[0], dtype=object)
    first_rows = df.sort_values("rollout_time").drop_duplicates("particle_id").set_index("particle_id")
    for col, pid in enumerate(particle_ids):
        terminal[col] = str(first_rows.loc[int(pid), "terminal_basin"])
    return particle_ids, terminal


def segment_array(path_xy: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [np.stack([path_xy[i], path_xy[i + 1]], axis=1) for i in range(path_xy.shape[0] - 1)],
        axis=0,
    )


def transform_embedding(scaler: StandardScaler, reducer: object, x: np.ndarray) -> np.ndarray:
    return reducer.transform(scaler.transform(x)).astype(np.float32)


def draw_rollout_paths(
    ax: plt.Axes,
    path_xy: np.ndarray,
    time_points: np.ndarray,
    terminal_basin: np.ndarray,
    *,
    include_labels: tuple[str, ...] | None = None,
    skip_labels: tuple[str, ...] = (),
    show_arrows: bool = True,
    max_faint_per_basin: int = MAX_FAINT_PER_BASIN,
    show_faint: bool = True,
    marker_time_values: tuple[float, ...] = (0, 1, 2, 3, 4),
    marker_size: float = 30,
) -> dict[str, int]:
    counts = {}
    integer_idx = [int(np.argmin(np.abs(time_points - t))) for t in marker_time_values]
    draw_order = include_labels if include_labels is not None else tuple(PHENO_ORDER)
    for label in draw_order:
        if label in skip_labels:
            continue
        ids = np.where(terminal_basin == label)[0]
        if ids.size == 0:
            continue
        counts[label] = int(ids.size)
        color = PHENO_COLORS.get(label, "#64748B")

        if show_faint and max_faint_per_basin > 0:
            take = min(max_faint_per_basin, ids.size)
            faint_ids = np.sort(RNG.choice(ids, size=take, replace=False))
            faint_segments = segment_array(path_xy[:, faint_ids, :])
            ax.add_collection(LineCollection(faint_segments, colors=color, linewidths=0.55, alpha=0.075, zorder=3))

        median_path = np.nanmedian(path_xy[:, ids, :], axis=1)
        median_path = gaussian_filter1d(median_path, sigma=1.15, axis=0, mode="nearest")
        line = ax.plot(
            median_path[:, 0],
            median_path[:, 1],
            color=color,
            lw=2.55,
            alpha=0.96,
            solid_capstyle="round",
            zorder=7,
        )[0]
        line.set_path_effects([pe.Stroke(linewidth=4.4, foreground="white", alpha=0.86), pe.Normal()])

        ax.scatter(
            median_path[integer_idx, 0],
            median_path[integer_idx, 1],
            s=marker_size,
            facecolor="white",
            edgecolor=color,
            linewidth=1.25,
            zorder=8,
        )
        # Direction cue near the late path, kept small to avoid competing with labels.
        if show_arrows and median_path.shape[0] > 8:
            j = int(round(median_path.shape[0] * 0.70))
            ax.annotate(
                "",
                xy=median_path[min(j + 2, median_path.shape[0] - 1)],
                xytext=median_path[max(j - 2, 0)],
                arrowprops={"arrowstyle": "->", "lw": 1.25, "color": color, "shrinkA": 0, "shrinkB": 0},
                zorder=9,
            )
    return counts


def main() -> None:
    setup()
    plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42})
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(DATA_PATH)
    protein = np.asarray(adata.obsm["X_protein_pca"], dtype=np.float32)[:, :PROTEIN_DIM]
    labels = adata.obs["cell_type"].astype(str).to_numpy()

    emb_name, scaler, reducer, obs_emb = fit_observed_protein_umap(protein)
    particle_ids, terminal_basin = load_sampled_particle_ids(COORD_TABLE)

    traj = np.load(TRAJECTORY_PATH)
    time_points = np.asarray(traj["time_points"], dtype=float)
    selected_proteins = np.concatenate(
        [np.asarray(traj[f"points_{i:06d}"], dtype=np.float32)[particle_ids, 50 : 50 + PROTEIN_DIM] for i in range(time_points.shape[0])],
        axis=0,
    )
    selected_xy = transform_embedding(scaler, reducer, selected_proteins)
    path_xy = selected_xy.reshape(time_points.shape[0], particle_ids.shape[0], 2)

    fig, ax = plt.subplots(figsize=(6.6, 6.35))
    draw_observed_fig01a_style(ax, obs_emb, labels)
    counts = draw_rollout_paths(ax, path_xy, time_points, terminal_basin)
    clean_embedding_axis(ax)

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#D1D5DB", markeredgecolor="#111111", markersize=5.2, label="observed protein cells"),
        Line2D([0], [0], color="#111827", lw=2.1, label="median generated path"),
        Line2D([0], [0], marker="o", color="#111827", markerfacecolor="white", markeredgewidth=1.2, linestyle="None", markersize=5.2, label="t0-t4"),
    ]
    ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=7.2, handlelength=1.6)

    out_png = FIG_DIR / "fig01h_protein_umap_rollout_fig01a_style_clean.png"
    fig.savefig(out_png, dpi=420, bbox_inches="tight", facecolor="white")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(out_png.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

    manifest = {
        "analysis": "fig01h_protein_umap_rollout_fig01a_style_clean",
        "basis": f"observed protein {emb_name} fitted from X_protein_pca, same display logic as Fig01A",
        "trajectory_path": str(TRAJECTORY_PATH),
        "sampled_particle_id_table": str(COORD_TABLE),
        "projection": "Uses Fig01G sampled particle IDs; selected generated protein states are transformed with the current observed protein UMAP reducer.",
        "terminal_basin_assignment": "Inherited from Fig01G: 25-NN majority vote in observed final-day protein PCA space.",
        "n_particles": int(path_xy.shape[1]),
        "n_timepoints": int(time_points.shape[0]),
        "terminal_basin_counts": counts,
        "figures": {
            "png": str(out_png),
            "pdf": str(out_png.with_suffix(".pdf")),
            "svg": str(out_png.with_suffix(".svg")),
        },
    }
    manifest_path = OUT_DIR / "fig01h_protein_umap_rollout_fig01a_style_clean_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"figure": str(out_png), "manifest": str(manifest_path), "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
