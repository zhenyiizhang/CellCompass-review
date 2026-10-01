#!/usr/bin/env python3
"""Plot all-cell BMF velocity stream and growth rate on the LARRY SPRING map."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cellcompass_mpl_cache")
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import anndata as ad
import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "downstream/cellcompass_clone_case/scdiffeq_1to1_bmf"
FIG_DIR = ROOT / "figures/cellcompass_clone_case/scdiffeq_1to1_bmf"
VELOCITY_INPUT = OUT_DIR / "fig2b_scdiffeq_velocity_input.h5ad"

FATE_PALETTE = {
    "Undifferentiated": "#2E8B57",
    "Neutrophil": "#19B7C7",
    "Monocyte": "#8C564B",
    "Baso": "#7E57C2",
    "Mast": "#D95F9F",
    "Meg": "#2563EB",
    "Erythroid": "#F28E2B",
    "Eos": "#D62728",
    "Lymphoid": "#8FBCE6",
    "Ccr7_DC": "#B8B34A",
    "pDC": "#6B7280",
}

LABEL_OFFSETS = {
    "Undifferentiated": (-20, 20),
    "Neutrophil": (0, -35),
    "Monocyte": (20, 20),
    "Baso": (-25, -15),
    "Mast": (-15, -25),
    "Meg": (-25, 20),
    "Erythroid": (20, -15),
    "Eos": (0, 10),
    "Lymphoid": (0, 10),
    "Ccr7_DC": (0, 10),
    "pDC": (0, -10),
}

GROWTH_CMAP = LinearSegmentedColormap.from_list(
    "bmf_growth",
    ["#D9E2EC", "#A7F3D0", "#FDE68A", "#EF4444"],
)


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.linewidth": 1.1,
            "axes.edgecolor": "#111827",
            "axes.titlesize": 9,
            "axes.titleweight": "bold",
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
        }
    )


def style_spring(ax: plt.Axes, xy: np.ndarray) -> None:
    x = xy[:, 0]
    y = xy[:, 1]
    padx = 0.035 * (np.nanmax(x) - np.nanmin(x))
    pady = 0.035 * (np.nanmax(y) - np.nanmin(y))
    ax.set_xlim(float(np.nanmin(x) - padx), float(np.nanmax(x) + padx))
    ax.set_ylim(float(np.nanmin(y) - pady), float(np.nanmax(y) + pady))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def add_fate_labels(ax: plt.Axes, obs: pd.DataFrame) -> None:
    for fate, color in FATE_PALETTE.items():
        d = obs[obs["cell_type"].astype(str).eq(fate)]
        if len(d) < 25:
            continue
        dx, dy = LABEL_OFFSETS.get(fate, (0, 0))
        ax.text(
            float(d["spring_x"].median()) + dx,
            float(d["spring_y"].median()) + dy,
            fate.replace("Undifferentiated", "Undiff."),
            color=color,
            fontsize=7,
            weight="bold",
            ha="center",
            va="center",
            path_effects=[pe.Stroke(linewidth=2.0, foreground="white", alpha=0.95), pe.Normal()],
            zorder=50,
        )


def draw_fate_background(ax: plt.Axes, obs: pd.DataFrame, alpha: float = 0.09) -> None:
    for fate, color in FATE_PALETTE.items():
        d = obs[obs["cell_type"].astype(str).eq(fate)]
        if d.empty:
            continue
        ax.scatter(d["spring_x"], d["spring_y"], s=0.75, c=color, alpha=alpha, linewidth=0, rasterized=True, zorder=0)


def project_velocity_to_spring(x_latent: np.ndarray, v_latent: np.ndarray, xy: np.ndarray, k: int = 30) -> np.ndarray:
    x_use = np.asarray(x_latent[:, :10], dtype=np.float32)
    v_use = np.asarray(v_latent[:, :10], dtype=np.float32)
    nn = NearestNeighbors(n_neighbors=2, n_jobs=-1).fit(x_use)
    d, _ = nn.kneighbors(x_use)
    local_step = float(np.nanmedian(d[:, 1]))
    v_norm = np.linalg.norm(v_use, axis=1)
    scale = 0.65 * local_step / max(float(np.nanmedian(v_norm[v_norm > 1e-12])), 1e-12)
    query = x_use + scale * v_use
    tree = cKDTree(x_use)
    dist, idx = tree.query(query, k=min(k, len(x_use)))
    if idx.ndim == 1:
        dist = dist[:, None]
        idx = idx[:, None]
    weights = 1.0 / (np.asarray(dist, dtype=float) + 1e-4)
    weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    xy_next = np.einsum("ij,ijk->ik", weights, xy[np.asarray(idx, dtype=int)])
    return xy_next - xy


def grid_velocity(xy: np.ndarray, vxy: np.ndarray, density: float = 1.0, smooth: float = 0.75) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(xy).all(axis=1) & np.isfinite(vxy).all(axis=1)
    xy = xy[finite]
    vxy = vxy[finite]
    grids = []
    for dim in range(2):
        lo = float(np.min(xy[:, dim]))
        hi = float(np.max(xy[:, dim]))
        pad = 0.01 * abs(hi - lo)
        grids.append(np.linspace(lo - pad, hi + pad, int(55 * density)))
    grid = np.vstack([g.ravel() for g in np.meshgrid(*grids)]).T
    n_neighbors = int(max(20, min(260, xy.shape[0] / 80)))
    nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1).fit(xy)
    dists, idx = nn.kneighbors(grid)
    step = float(np.mean([g[1] - g[0] for g in grids]))
    weights = scipy.stats.norm.pdf(dists, scale=max(step * smooth, 1e-6))
    mass = weights.sum(axis=1)
    vel = (vxy[idx] * weights[:, :, None]).sum(axis=1)
    vel = np.divide(vel, np.maximum(mass, 1e-12)[:, None])
    ns = int(np.sqrt(len(vel)))
    stream = vel.T.reshape(2, ns, ns)
    speed = np.sqrt((stream**2).sum(axis=0))
    mass_grid = mass.reshape(ns, ns)
    keep = (mass_grid >= np.nanpercentile(mass_grid, 56)) & (speed >= np.nanpercentile(speed, 35))
    stream[0][~keep] = np.nan
    stream[1][~keep] = np.nan
    return np.stack([np.unique(grid[:, 0]), np.unique(grid[:, 1])]), stream


def main() -> None:
    setup_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    adata = ad.read_h5ad(VELOCITY_INPUT)
    obs = adata.obs.copy()
    obs[["spring_x", "spring_y"]] = adata.obsm["X_umap"]
    xy = np.asarray(adata.obsm["X_umap"], dtype=float)
    latent = np.asarray(adata.obsm["X_pca"], dtype=np.float32)
    velocity = np.asarray(adata.obsm["X_drift"], dtype=np.float32)
    growth = obs["bmf_growth_dlogmass_dt"].to_numpy(float)

    vxy = project_velocity_to_spring(latent, velocity, xy, k=30)
    grid, stream = grid_velocity(xy, vxy, density=1.0, smooth=0.75)

    vmin, vmax = np.nanpercentile(growth, [4, 96])
    order = np.argsort(growth)

    fig, axes = plt.subplots(1, 2, figsize=(8.7, 4.1), gridspec_kw={"wspace": 0.08})
    ax_stream, ax_growth = axes

    draw_fate_background(ax_stream, obs, alpha=0.12)
    ax_stream.streamplot(
        grid[0],
        grid[1],
        stream[0],
        stream[1],
        color="#111827",
        linewidth=0.55,
        density=1.95,
        arrowsize=0.78,
        maxlength=4.0,
        integration_direction="both",
        zorder=20,
    )
    add_fate_labels(ax_stream, obs)
    style_spring(ax_stream, xy)
    ax_stream.set_title("BMF velocity stream on all observed cells", loc="left")

    sc = ax_growth.scatter(
        xy[:, 0][order],
        xy[:, 1][order],
        c=growth[order],
        cmap=GROWTH_CMAP,
        vmin=vmin,
        vmax=vmax,
        s=1.5,
        alpha=0.68,
        linewidth=0,
        rasterized=True,
    )
    add_fate_labels(ax_growth, obs)
    style_spring(ax_growth, xy)
    ax_growth.set_title("BMF growth rate per observed cell", loc="left")
    cb = fig.colorbar(sc, ax=ax_growth, fraction=0.045, pad=0.02)
    cb.set_label("BMF d log mass / dt", fontsize=7)

    fig.suptitle("BMF velocity and growth on the LARRY SPRING landscape", x=0.02, y=0.99, ha="left", weight="bold")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIG_DIR / f"fig2b_fig4ef_bmf_velocity_growth_spring.{ext}", dpi=360 if ext == "png" else None)
    plt.close(fig)
    print(f"Wrote {FIG_DIR / 'fig2b_fig4ef_bmf_velocity_growth_spring.png'}")


if __name__ == "__main__":
    main()
