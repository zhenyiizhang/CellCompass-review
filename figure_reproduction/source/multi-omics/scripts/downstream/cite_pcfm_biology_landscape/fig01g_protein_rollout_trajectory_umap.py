#!/usr/bin/env python3
"""Plot CITE-PCFM rollout trajectories on the observed protein UMAP.

The rollout is read from the locked CITE-PCFM evaluation trajectory artifact.
Observed protein PCA is used only to fit the UMAP display coordinate. Generated
protein states are projected into that fixed coordinate system with UMAP
transform; no observed future protein values are used to modify the rollout.
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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
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
OUT_DIR = ROOT / "downstream/cite_pcfm_deep_story/biology_landscape_discovery_protein_rollout_trajectory"

RNA_DIM = 50
PROTEIN_DIM = 30
MAX_TRAJECTORIES = 168
MAX_PER_TERMINAL_BASIN = 28
RANDOM_STATE = 20260528
RNG = np.random.default_rng(RANDOM_STATE)
PHENO_ABBR = {
    "HSC": "H",
    "MasP": "Ma",
    "NeuP": "Ne",
    "EryP": "Er",
    "MkP": "Mk",
    "MoP": "Mo",
    "BP": "B",
}
setup()
TIME_CMAP = LinearSegmentedColormap.from_list(
    "cite_pcfm_rollout_time",
    ["#006FFC", "#1EBFC1", "#91FE46", "#FEA840", "#D00426"],
)


def scatter_celltype_background(
    ax: plt.Axes,
    xy: np.ndarray,
    labels: np.ndarray,
    *,
    s: float = 4.8,
    alpha: float = 0.16,
) -> None:
    scatter_groups(ax, xy, labels, PHENO_COLORS, order=PHENO_ORDER, s=s, alpha=alpha, zorder=1)
    add_median_labels(ax, xy, labels, order=PHENO_ORDER, min_cells=24, color_map=PHENO_COLORS, fontsize=7.6)


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


def transform_embedding(scaler: StandardScaler, reducer: object, x: np.ndarray, batch_size: int = 50000) -> np.ndarray:
    parts = []
    for start in range(0, x.shape[0], batch_size):
        end = min(start + batch_size, x.shape[0])
        parts.append(reducer.transform(scaler.transform(x[start:end])).astype(np.float32))
    return np.vstack(parts)


def knn_majority_labels(generated: np.ndarray, observed_ref: np.ndarray, observed_ref_labels: np.ndarray, k: int = 25) -> np.ndarray:
    k_eff = int(min(k, observed_ref.shape[0]))
    nn = NearestNeighbors(n_neighbors=k_eff, metric="euclidean").fit(observed_ref)
    _, idx = nn.kneighbors(generated)
    out = []
    for row in idx:
        vals, counts = np.unique(observed_ref_labels[row].astype(str), return_counts=True)
        out.append(str(vals[np.argmax(counts)]))
    return np.asarray(out, dtype=object)


def nearest_terminal_basin(generated_terminal: np.ndarray, observed_final: np.ndarray, observed_final_labels: np.ndarray) -> np.ndarray:
    return knn_majority_labels(generated_terminal, observed_final, observed_final_labels, k=25)


def classify_integer_time_current_types(
    points_by_time: list[np.ndarray],
    time_points: np.ndarray,
    selected_ids: np.ndarray,
    protein: np.ndarray,
    labels: np.ndarray,
    times_obs: np.ndarray,
) -> tuple[dict[int, np.ndarray], dict[int, float]]:
    """Classify selected generated states at t0-t4 against same-time observed protein cells."""
    current: dict[int, np.ndarray] = {}
    time_by_idx: dict[int, float] = {}
    for target_t in [0.0, 1.0, 2.0, 3.0, 4.0]:
        ti = int(np.argmin(np.abs(time_points - target_t)))
        ref_mask = np.isclose(times_obs, target_t)
        if not np.any(ref_mask):
            continue
        generated = points_by_time[ti][selected_ids, RNA_DIM : RNA_DIM + PROTEIN_DIM]
        current[ti] = knn_majority_labels(generated, protein[ref_mask], labels[ref_mask], k=25)
        time_by_idx[ti] = float(target_t)
    return current, time_by_idx


def choose_balanced_trajectory_ids(terminal_basin: np.ndarray) -> np.ndarray:
    picked: list[int] = []
    for label in PHENO_ORDER:
        idx = np.where(terminal_basin == label)[0]
        if idx.size == 0:
            continue
        take = min(MAX_PER_TERMINAL_BASIN, idx.size)
        picked.extend(RNG.choice(idx, size=take, replace=False).tolist())
    if len(picked) < min(MAX_TRAJECTORIES, terminal_basin.shape[0]):
        remaining = np.setdiff1d(np.arange(terminal_basin.shape[0]), np.asarray(picked, dtype=int), assume_unique=False)
        take = min(MAX_TRAJECTORIES - len(picked), remaining.size)
        if take > 0:
            picked.extend(RNG.choice(remaining, size=take, replace=False).tolist())
    picked = picked[: min(MAX_TRAJECTORIES, terminal_basin.shape[0])]
    return np.asarray(sorted(picked), dtype=int)


def build_segments(path_xy: np.ndarray) -> np.ndarray:
    # path_xy shape: n_time x n_selected x 2
    return np.concatenate(
        [np.stack([path_xy[i], path_xy[i + 1]], axis=1) for i in range(path_xy.shape[0] - 1)],
        axis=0,
    )


def majority_current_type_by_terminal(
    selected_terminal_basin: np.ndarray,
    current_labels_by_time_idx: dict[int, np.ndarray],
    current_label_time_by_idx: dict[int, float],
) -> dict[str, dict[str, str]]:
    summary: dict[str, dict[str, str]] = {}
    sorted_time_idx = sorted(current_labels_by_time_idx, key=lambda i: current_label_time_by_idx.get(i, float(i)))
    for label in PHENO_ORDER:
        mask = selected_terminal_basin == label
        if not np.any(mask):
            continue
        summary[label] = {}
        for ti in sorted_time_idx:
            current_labels = current_labels_by_time_idx[ti]
            vals, counts = np.unique(current_labels[mask].astype(str), return_counts=True)
            marker_label = str(vals[np.argmax(counts)])
            summary[label][f"t{int(current_label_time_by_idx[ti])}"] = marker_label
    return summary


def add_identity_ribbon(
    ax: plt.Axes,
    current_marker_summary: dict[str, dict[str, str]],
) -> None:
    rows = [label for label in PHENO_ORDER if label in current_marker_summary]
    if not rows:
        return
    columns = ["t0", "t1", "t2", "t3", "t4"]
    inset = ax.inset_axes([0.655, 0.705, 0.315, 0.245])
    inset.set_facecolor((1.0, 1.0, 1.0, 0.93))
    for spine in inset.spines.values():
        spine.set_color("#CBD5E1")
        spine.set_linewidth(0.8)
    inset.set_xlim(-1.35, len(columns) - 0.25)
    inset.set_ylim(len(rows) - 0.35, -1.05)
    inset.set_xticks([])
    inset.set_yticks([])
    inset.tick_params(length=0)
    inset.text(-1.24, -0.70, "terminal", fontsize=6.8, color="#475569", weight="bold", ha="left", va="center")
    inset.text(1.92, -0.70, "current kNN type", fontsize=6.8, color="#475569", weight="bold", ha="center", va="center")
    for col, name in enumerate(columns):
        inset.text(col, -0.28, name, fontsize=6.5, color="#475569", ha="center", va="center")
    for row_i, terminal_label in enumerate(rows):
        y = float(row_i)
        terminal_color = PHENO_COLORS.get(terminal_label, "#64748B")
        inset.plot([-1.22, -0.88], [y, y], color=terminal_color, lw=2.8, solid_capstyle="round")
        inset.text(-0.78, y, terminal_label, fontsize=7.2, color="#0F172A", ha="left", va="center")
        for col, time_label in enumerate(columns):
            current_label = current_marker_summary[terminal_label].get(time_label, "")
            current_color = PHENO_COLORS.get(current_label, "#CBD5E1")
            inset.scatter(col, y, s=58, color=current_color, edgecolor="white", linewidth=0.7, zorder=3)
            inset.text(
                col,
                y,
                PHENO_ABBR.get(current_label, ""),
                fontsize=5.4,
                color="#0F172A",
                weight="bold",
                ha="center",
                va="center",
                zorder=4,
            )


def plot_time_colored_trajectories(
    obs_emb: np.ndarray,
    obs_labels: np.ndarray,
    path_xy: np.ndarray,
    time_points: np.ndarray,
    selected_ids: np.ndarray,
    out_png: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(13.2, 9.4))
    scatter_celltype_background(ax, obs_emb, obs_labels, s=5.2, alpha=0.24)

    segments = build_segments(path_xy)
    seg_times = np.repeat(0.5 * (time_points[:-1] + time_points[1:]), selected_ids.shape[0])
    norm = Normalize(vmin=float(time_points.min()), vmax=float(time_points.max()))
    lc = LineCollection(segments, cmap=TIME_CMAP, norm=norm, linewidths=0.72, alpha=0.12, zorder=3)
    lc.set_array(seg_times)
    ax.add_collection(lc)

    integer_times = [0.0, 1.0, 2.0, 3.0, 4.0]
    for t in integer_times:
        idx = int(np.argmin(np.abs(time_points - t)))
        color = TIME_CMAP(norm(float(time_points[idx])))
        size = 13 if t in (0.0, 4.0) else 7
        alpha = 0.66 if t in (0.0, 4.0) else 0.24
        ax.scatter(path_xy[idx, :, 0], path_xy[idx, :, 1], s=size, color=color, alpha=alpha, linewidths=0, rasterized=True, zorder=4)
        center = np.nanmedian(path_xy[idx], axis=0)
        ax.text(
            center[0],
            center[1],
            f"t{int(t)}",
            fontsize=9.2,
            weight="bold",
            color="#0F172A",
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.13", "facecolor": "white", "edgecolor": "none", "alpha": 0.78},
            zorder=9,
        )

    title_left(ax, "CITE-PCFM rollout trajectories on the observed protein UMAP", fontsize=14.5)
    note_box(ax, "Thin lines are sampled RNA-initialized particles; points mark generated protein states at t0-t4.")
    cbar = fig.colorbar(lc, ax=ax, fraction=0.028, pad=0.012)
    cbar.set_label("rollout time", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    clean_embedding_axis(ax)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#CBD5E1", markersize=6, label="observed protein cells"),
        Line2D([0], [0], color="#334155", lw=1.2, alpha=0.55, label="CITE-PCFM rollout path"),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower left", fontsize=8.5)
    fig.savefig(out_png, dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_png


def plot_terminal_basin_trajectories(
    obs_emb: np.ndarray,
    obs_labels: np.ndarray,
    path_xy: np.ndarray,
    selected_terminal_basin: np.ndarray,
    current_labels_by_time_idx: dict[int, np.ndarray],
    current_label_time_by_idx: dict[int, float],
    out_png: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(13.2, 9.4))
    scatter_celltype_background(ax, obs_emb, obs_labels, s=5.2, alpha=0.24)

    for label in PHENO_ORDER:
        mask = selected_terminal_basin == label
        if not np.any(mask):
            continue
        color = PHENO_COLORS.get(label, "#64748B")
        seg = build_segments(path_xy[:, mask, :])
        ax.add_collection(LineCollection(seg, colors=color, linewidths=0.55, alpha=0.035, zorder=3))
        median_path = np.nanmedian(path_xy[:, mask, :], axis=1)
        ax.plot(
            median_path[:, 0],
            median_path[:, 1],
            color=color,
            lw=2.8,
            alpha=0.92,
            zorder=6,
            solid_capstyle="round",
        )
        integer_idx = [0, 10, 20, 30, 40]
        integer_idx = [i for i in integer_idx if i < median_path.shape[0]]
        ax.scatter(
            median_path[integer_idx, 0],
            median_path[integer_idx, 1],
            s=38,
            facecolors="white",
            edgecolor=color,
            linewidth=1.4,
            zorder=7,
        )
        ax.scatter(
            path_xy[-1, mask, 0],
            path_xy[-1, mask, 1],
            s=8,
            color=color,
            alpha=0.18,
            linewidths=0,
            rasterized=True,
            zorder=5,
        )

    title_left(ax, "CITE-PCFM rollout colored by terminal protein basin", fontsize=14.5)
    note_box(
        ax,
        "Line color denotes terminal protein basin; white markers indicate t0-t4 along each median rollout.\n"
        "Inset summarizes current kNN protein identity at each marked time.",
    )
    current_marker_summary = majority_current_type_by_terminal(
        selected_terminal_basin,
        current_labels_by_time_idx,
        current_label_time_by_idx,
    )
    add_identity_ribbon(ax, current_marker_summary)
    clean_embedding_axis(ax)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#CBD5E1", markersize=6, label="observed protein cells"),
        Line2D([0], [0], marker="o", color="#334155", markerfacecolor="white", markeredgewidth=1.2, markersize=6, linestyle="None", label="t0-t4 markers"),
    ] + [
        Line2D([0], [0], color=PHENO_COLORS[label], lw=2, label=label)
        for label in PHENO_ORDER
        if np.any(selected_terminal_basin == label)
    ]
    ax.legend(handles=handles, frameon=False, loc="lower left", fontsize=8.5, ncol=4)
    fig.savefig(out_png, dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_png


def main() -> dict:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(DATA_PATH)
    protein = np.asarray(adata.obsm["X_protein_pca"], dtype=np.float32)[:, :PROTEIN_DIM]
    latent = np.asarray(adata.obsm["X_latent"], dtype=np.float32)
    labels = adata.obs["cell_type"].astype(str).to_numpy()
    times_obs = np.asarray(adata.obs["time_point_processed"], dtype=float)

    emb_name, scaler, reducer, obs_emb = fit_observed_protein_umap(protein)

    traj = np.load(TRAJECTORY_PATH)
    time_points = np.asarray(traj["time_points"], dtype=float)
    points_by_time = [np.asarray(traj[f"points_{i:06d}"], dtype=np.float32) for i in range(time_points.shape[0])]
    n_particles = points_by_time[0].shape[0]

    t0_idx = np.where(np.isclose(times_obs, float(time_points[0])))[0]
    if t0_idx.shape[0] != n_particles:
        raise ValueError(f"t0 observed count {t0_idx.shape[0]} does not match trajectory particles {n_particles}")
    t0_rna_order_mean_abs_diff = float(np.mean(np.abs(points_by_time[0][:, :RNA_DIM] - latent[t0_idx, :RNA_DIM])))
    t0_rna_order_max_abs_diff = float(np.max(np.abs(points_by_time[0][:, :RNA_DIM] - latent[t0_idx, :RNA_DIM])))

    final_mask = np.isclose(times_obs, float(time_points[-1]))
    terminal_basin = nearest_terminal_basin(
        points_by_time[-1][:, RNA_DIM : RNA_DIM + PROTEIN_DIM],
        protein[final_mask],
        labels[final_mask],
    )
    selected_ids = choose_balanced_trajectory_ids(terminal_basin)
    current_labels_by_time_idx, current_label_time_by_idx = classify_integer_time_current_types(
        points_by_time,
        time_points,
        selected_ids,
        protein,
        labels,
        times_obs,
    )

    # Transform every sampled continuous rollout state into the fixed observed protein UMAP.
    selected_protein_states = np.concatenate(
        [points[:, RNA_DIM : RNA_DIM + PROTEIN_DIM][selected_ids] for points in points_by_time],
        axis=0,
    )
    selected_xy_flat = transform_embedding(scaler, reducer, selected_protein_states)
    path_xy = selected_xy_flat.reshape(time_points.shape[0], selected_ids.shape[0], 2)

    time_png = FIG_DIR / "fig01g_protein_umap_cite_pcfm_rollout_trajectory_time.png"
    basin_png = FIG_DIR / "fig01g_protein_umap_cite_pcfm_rollout_trajectory_terminal_basin.png"
    plot_time_colored_trajectories(obs_emb, labels, path_xy, time_points, selected_ids, time_png)
    plot_terminal_basin_trajectories(
        obs_emb,
        labels,
        path_xy,
        terminal_basin[selected_ids],
        current_labels_by_time_idx,
        current_label_time_by_idx,
        basin_png,
    )

    rows = []
    init_obs_names = adata.obs_names[t0_idx].astype(str).to_numpy()
    init_observed_cell_types = labels[t0_idx]
    for ti, t in enumerate(time_points):
        xy = path_xy[ti]
        current_labels = current_labels_by_time_idx.get(ti)
        for col, pid in enumerate(selected_ids):
            rows.append(
                {
                    "particle_id": int(pid),
                    "init_obs_name": str(init_obs_names[pid]),
                    "init_observed_cell_type": str(init_observed_cell_types[pid]),
                    "rollout_time": float(t),
                    "protein_umap_1": float(xy[col, 0]),
                    "protein_umap_2": float(xy[col, 1]),
                    "terminal_basin": str(terminal_basin[pid]),
                    "current_knn_cell_type": str(current_labels[col]) if current_labels is not None else "",
                }
            )
    coord_table = OUT_DIR / "cite_pcfm_rollout_trajectory_protein_umap_coordinates.csv"
    pd.DataFrame(rows).to_csv(coord_table, index=False)

    terminal_counts = pd.Series(terminal_basin).value_counts().reindex(PHENO_ORDER).fillna(0).astype(int).to_dict()
    selected_terminal_counts = pd.Series(terminal_basin[selected_ids]).value_counts().reindex(PHENO_ORDER).fillna(0).astype(int).to_dict()
    selected_terminal_basin = terminal_basin[selected_ids]
    current_marker_summary = majority_current_type_by_terminal(
        selected_terminal_basin,
        current_labels_by_time_idx,
        current_label_time_by_idx,
    )

    manifest = {
        "analysis": "cite_pcfm_rollout_trajectory_on_observed_protein_umap",
        "skill": "downstream-trajectory-fate",
        "evidence_type": "locked model rollout trajectory projected to observed protein UMAP",
        "data_h5ad": str(DATA_PATH),
        "trajectory_path": str(TRAJECTORY_PATH),
        "basis": f"observed protein {emb_name} fitted from X_protein_pca",
        "projection": "Generated protein blocks points[:, 50:80] were transformed with the GT-fitted protein UMAP reducer.",
        "n_particles_total": int(n_particles),
        "n_particles_plotted": int(selected_ids.shape[0]),
        "n_rollout_timepoints": int(time_points.shape[0]),
        "rollout_time_min": float(time_points.min()),
        "rollout_time_max": float(time_points.max()),
        "source_order_check": {
            "t0_observed_count": int(t0_idx.shape[0]),
            "t0_trajectory_particle_count": int(n_particles),
            "rna_block_mean_abs_diff_points0_vs_observed_t0": t0_rna_order_mean_abs_diff,
            "rna_block_max_abs_diff_points0_vs_observed_t0": t0_rna_order_max_abs_diff,
        },
        "terminal_basin_assignment": {
            "method": "kNN majority vote in protein PCA space",
            "reference": "observed t4 protein cells",
            "k": 25,
            "all_particle_counts": terminal_counts,
            "plotted_particle_counts": selected_terminal_counts,
        },
        "current_marker_cell_type_assignment": {
            "method": "For displayed t0-t4 markers, generated protein states are classified by kNN majority vote against observed protein cells from the same time point.",
            "k": 25,
            "display_semantics": "main UMAP line color is terminal protein basin; white markers indicate t0-t4; inset dots show current kNN protein cell type at those times",
            "majority_current_type_by_terminal_basin": current_marker_summary,
        },
        "figures": {
            "time_colored_png": str(time_png),
            "time_colored_pdf": str(time_png.with_suffix(".pdf")),
            "terminal_basin_png": str(basin_png),
            "terminal_basin_pdf": str(basin_png.with_suffix(".pdf")),
        },
        "tables": {
            "trajectory_coordinates": str(coord_table),
        },
        "warnings": [
            "Continuous trajectories come from the locked CITE-PCFM evaluation_trajectory.npz artifact, not from a local velocity-field interpolation.",
            "UMAP transform is nonlinear and only provides a qualitative display coordinate; the true generated states remain in 30D protein PCA space.",
            "Observed protein cells define the display manifold, terminal kNN labels, and same-time marker kNN labels; they do not modify the generated rollout trajectory.",
        ],
    }
    manifest_path = OUT_DIR / "cite_pcfm_rollout_trajectory_protein_umap_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    print(json.dumps(
        {
            "time_colored_png": str(time_png),
            "terminal_basin_png": str(basin_png),
            "manifest": str(manifest_path),
            "coordinate_table": str(coord_table),
            "n_particles_plotted": int(selected_ids.shape[0]),
            "source_order_check": manifest["source_order_check"],
            "selected_terminal_counts": selected_terminal_counts,
        },
        indent=2,
    ))
    return manifest


if __name__ == "__main__":
    main()
