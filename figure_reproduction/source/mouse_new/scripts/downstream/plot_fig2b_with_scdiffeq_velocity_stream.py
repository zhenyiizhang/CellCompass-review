#!/usr/bin/env python3
"""Plot Fig2B with scDiffEq's native velocity_stream function."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cellcompass_mpl_cache")
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import anndata as ad
import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import scdiffeq as sdq


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "downstream/cellcompass_clone_case/scdiffeq_1to1_bmf"
FIG_DIR = ROOT / "figures/cellcompass_clone_case/scdiffeq_1to1_bmf"
VELOCITY_INPUT = OUT_DIR / "fig2b_scdiffeq_velocity_input.h5ad"
GROWTH_TABLE = OUT_DIR / "fig2b_fig4ef_bmf_growth_on_observed_timepoints.csv"

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

GROWTH_CMAP = LinearSegmentedColormap.from_list(
    "bmf_growth",
    ["#E5E7EB", "#A7F3D0", "#FDE68A", "#EF4444"],
)

LABEL_OFFSETS = {
    "Undifferentiated": (-20, 20),
    "Neutrophil": (0, -35),
    "Monocyte": (20, 20),
    "Baso": (-25, -15),
    "Mast": (-15, -25),
    "Meg": (-25, 20),
    "Erythroid": (20, -15),
}


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


def style_spring(ax: plt.Axes, x: np.ndarray, y: np.ndarray) -> None:
    padx = 0.035 * (np.nanmax(x) - np.nanmin(x))
    pady = 0.035 * (np.nanmax(y) - np.nanmin(y))
    ax.set_xlim(float(np.nanmin(x) - padx), float(np.nanmax(x) + padx))
    ax.set_ylim(float(np.nanmin(y) - pady), float(np.nanmax(y) + pady))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def add_fate_labels(ax: plt.Axes, frame: pd.DataFrame) -> None:
    for fate, color in FATE_PALETTE.items():
        d = frame[frame["cell_type"].astype(str).eq(fate)]
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
            zorder=500,
        )


def draw_light_background(ax: plt.Axes, frame: pd.DataFrame, alpha: float = 0.07) -> None:
    for fate, color in FATE_PALETTE.items():
        d = frame[frame["cell_type"].astype(str).eq(fate)]
        if d.empty:
            continue
        ax.scatter(d["spring_x"], d["spring_y"], s=0.9, c=color, alpha=alpha, linewidth=0, zorder=0)


def main() -> None:
    setup_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    adata = ad.read_h5ad(VELOCITY_INPUT)

    # This is the actual scDiffEq velocity graph calculation.
    sdq.tl.velocity_graph(
        adata,
        state_key="X_pca",
        velocity_key="X_drift",
        velocity_key_added="velocity",
        n_pcs=10,
        split_negative=True,
        silent=True,
        neighbor_kwargs={"n_neighbors": 15, "use_rep": "X_pca", "n_pcs": 10, "force": True},
    )

    all_cells = adata.obs.copy()
    all_cells[["spring_x", "spring_y"]] = adata.obsm["X_umap"]
    adata.obs["Cell type annotation"] = pd.Categorical(adata.obs["cell_type"].astype(str))

    growth = pd.read_csv(GROWTH_TABLE)
    bg_obs = growth[["cell_type", "spring_x", "spring_y"]].copy()
    all_growth = growth["bmf_growth_dlogmass_dt"].to_numpy(float)
    vmin, vmax = np.nanpercentile(all_growth, [2, 98])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        vmin, vmax = float(np.nanmin(all_growth)), float(np.nanmax(all_growth))

    fig, axes = plt.subplots(1, 2, figsize=(8.7, 4.1), gridspec_kw={"wspace": 0.08})
    ax_stream, ax_growth = axes

    draw_light_background(ax_stream, bg_obs, alpha=0.13)
    sdq.pl.velocity_stream(
        adata,
        ax=ax_stream,
        velocity_key="velocity",
        density=1.0,
        smooth=0.5,
        min_mass=1,
        cutoff_percentile=0.08,
        stream_color="#111827",
        linewidth=0.55,
        stream_density=2.35,
        arrowsize=0.8,
        maxlength=4.0,
        integration_direction="both",
        disable_scatter=True,
        disable_cbar=True,
        scatter_kwargs={"rasterized": True, "s": 4.5, "alpha": 0.42, "linewidth": 0},
        rasterized=True,
    )
    add_fate_labels(ax_stream, bg_obs)
    style_spring(ax_stream, bg_obs["spring_x"].to_numpy(float), bg_obs["spring_y"].to_numpy(float))
    ax_stream.set_title("BMF velocity stream on all observed cells", loc="left")

    order = np.argsort(growth["bmf_growth_dlogmass_dt"].to_numpy(float))
    last_sc = ax_growth.scatter(
        growth["spring_x"].to_numpy(float)[order],
        growth["spring_y"].to_numpy(float)[order],
        c=growth["bmf_growth_dlogmass_dt"].to_numpy(float)[order],
        cmap=GROWTH_CMAP,
        vmin=vmin,
        vmax=vmax,
        s=1.9,
        alpha=0.72,
        linewidth=0,
        rasterized=True,
    )
    add_fate_labels(ax_growth, growth)
    style_spring(ax_growth, growth["spring_x"].to_numpy(float), growth["spring_y"].to_numpy(float))
    ax_growth.set_title("BMF growth on observed cells", loc="left")
    cb = fig.colorbar(last_sc, ax=ax_growth, fraction=0.045, pad=0.02)
    cb.set_label("BMF d log mass / dt", fontsize=7)

    fig.suptitle("BMF velocity and growth on the LARRY SPRING landscape", x=0.02, y=0.99, ha="left", weight="bold")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIG_DIR / f"fig2b_fig4ef_bmf_velocity_growth_spring.{ext}", dpi=360 if ext == "png" else None)
    plt.close(fig)

    fig_nb, ax_nb = plt.subplots(figsize=(4.2, 4.0))
    draw_light_background(ax_nb, bg_obs, alpha=0.15)
    sdq.pl.velocity_stream(
        adata,
        cutoff_percentile=0.1,
        ax=ax_nb,
        disable_scatter=True,
        disable_cbar=True,
        velocity_key="velocity",
    )
    add_fate_labels(ax_nb, bg_obs)
    style_spring(ax_nb, bg_obs["spring_x"].to_numpy(float), bg_obs["spring_y"].to_numpy(float))
    ax_nb.set_title("BMF velocity stream on all observed cells", loc="left")
    for ext in ("png", "pdf", "svg"):
        fig_nb.savefig(FIG_DIR / f"fig2b_bmf_velocity_stream_scdiffeq_notebook_style.{ext}", dpi=500 if ext == "png" else None)
    plt.close(fig_nb)
    print(f"Wrote {FIG_DIR / 'fig2b_fig4ef_bmf_velocity_growth_spring.png'}")
    print(f"Wrote {FIG_DIR / 'fig2b_bmf_velocity_stream_scdiffeq_notebook_style.png'}")


if __name__ == "__main__":
    main()
