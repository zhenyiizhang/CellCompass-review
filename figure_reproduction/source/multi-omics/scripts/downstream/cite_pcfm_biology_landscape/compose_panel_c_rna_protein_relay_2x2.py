#!/usr/bin/env python3
"""Assemble a 2x2 panel C for RNA-to-protein CITE-PCFM biology.

The left column draws the same CITE-PCFM particles in two observed manifolds:
RNA space on top, protein space below. The right column shows the branch-level
RNA-to-ADT timing waves and the Mk-directed RNA query readout.
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
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    import umap  # type: ignore

    HAVE_UMAP = True
except Exception:
    HAVE_UMAP = False

from compose_panel_c_rna_protein_relay import draw_mkp_push, draw_relay
from fig01h_protein_rollout_on_fig01a_style import (
    COORD_TABLE,
    DATA_PATH,
    PROTEIN_DIM,
    TRAJECTORY_PATH,
    draw_rollout_paths,
    load_sampled_particle_ids,
    transform_embedding,
)
from multistageot_style import PHENO_COLORS, PHENO_ORDER, add_median_labels, setup


ROOT = Path("/Users/zhenyizhang/Documents/GitHub/CytoBridge-agent-runtime-v2-20260309/multi-omics")
FIG_DIR = ROOT / "figures/cite_pcfm_deep_story/biology_landscape_discovery"
OUT_DIR = ROOT / "downstream/cite_pcfm_deep_story/biology_landscape_discovery_protein_rollout_trajectory"
OUT_BASE = FIG_DIR / "panel_c_rna_to_surface_protein_relay_2x2"

RNA_DIM = 50
RANDOM_STATE = 20260528


def lighten(hex_color: str, amount: float = 0.18) -> str:
    rgb = np.asarray(to_rgb(hex_color))
    out = rgb + (1.0 - rgb) * amount
    return "#{:02x}{:02x}{:02x}".format(*(np.clip(out * 255, 0, 255).astype(int)))


LIGHT_PHENO_COLORS = {label: lighten(color, 0.16) for label, color in PHENO_COLORS.items()}


def fit_observed_embedding(x: np.ndarray, *, n_neighbors: int, min_dist: float) -> tuple[str, StandardScaler, object, np.ndarray]:
    scaler = StandardScaler().fit(x)
    x_scaled = scaler.transform(x)
    if HAVE_UMAP:
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric="euclidean",
            random_state=RANDOM_STATE,
            low_memory=True,
        )
        emb = reducer.fit_transform(x_scaled).astype(np.float32)
        return "UMAP", scaler, reducer, emb
    reducer = PCA(n_components=2, random_state=RANDOM_STATE)
    emb = reducer.fit_transform(x_scaled).astype(np.float32)
    return "PCA fallback", scaler, reducer, emb


def clean_embedding_axis(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_aspect("equal", adjustable="box")
    for spine in ax.spines.values():
        spine.set_visible(False)


def add_compact_panel_title(ax: plt.Axes, title: str, label: str) -> None:
    ax.text(
        0.0,
        1.035,
        label,
        transform=ax.transAxes,
        fontsize=11.5,
        weight="bold",
        ha="left",
        va="bottom",
        color="#111827",
    )
    ax.text(
        0.070,
        1.035,
        title,
        transform=ax.transAxes,
        fontsize=11.0,
        weight="bold",
        ha="left",
        va="bottom",
        color="#111827",
    )


def restyle_panel_title(texts: list[plt.Text], label: str, title: str, *, title_at_end: bool) -> None:
    label_text, title_text = (texts[-2], texts[-1]) if title_at_end else (texts[0], texts[1])
    label_text.set_text(label)
    label_text.set_fontsize(11.5)
    label_text.set_color("#111827")
    title_text.set_text(title)
    title_text.set_fontsize(11.0)
    title_text.set_color("#111827")


def draw_light_observed_cells(ax: plt.Axes, xy: np.ndarray, labels: np.ndarray) -> None:
    """Draw compact UMAP backgrounds that stay readable after panel scaling."""
    for label in PHENO_ORDER:
        mask = labels.astype(str) == label
        if not np.any(mask):
            continue
        ax.scatter(
            xy[mask, 0],
            xy[mask, 1],
            s=6.0,
            facecolor=LIGHT_PHENO_COLORS.get(label, "#94A3B8"),
            edgecolor="none",
            linewidth=0,
            alpha=0.42,
            zorder=1,
        )
    add_median_labels(ax, xy, labels, order=PHENO_ORDER, min_cells=24, color_map=PHENO_COLORS, fontsize=8.2)


def load_rollout_data() -> dict:
    adata = ad.read_h5ad(DATA_PATH)
    labels = adata.obs["cell_type"].astype(str).to_numpy()
    rna = np.asarray(adata.obsm["X_rna_svd"], dtype=np.float32)[:, :RNA_DIM]
    protein = np.asarray(adata.obsm["X_protein_pca"], dtype=np.float32)[:, :PROTEIN_DIM]

    particle_ids, terminal_basin = load_sampled_particle_ids(COORD_TABLE)
    traj = np.load(TRAJECTORY_PATH)
    time_points = np.asarray(traj["time_points"], dtype=float)

    selected_rna = np.concatenate(
        [np.asarray(traj[f"points_{i:06d}"], dtype=np.float32)[particle_ids, :RNA_DIM] for i in range(time_points.shape[0])],
        axis=0,
    )
    selected_protein = np.concatenate(
        [
            np.asarray(traj[f"points_{i:06d}"], dtype=np.float32)[particle_ids, RNA_DIM : RNA_DIM + PROTEIN_DIM]
            for i in range(time_points.shape[0])
        ],
        axis=0,
    )

    rna_basis, rna_scaler, rna_reducer, rna_obs_xy = fit_observed_embedding(rna, n_neighbors=40, min_dist=0.28)
    protein_basis, protein_scaler, protein_reducer, protein_obs_xy = fit_observed_embedding(
        protein,
        n_neighbors=35,
        min_dist=0.22,
    )
    rna_path_xy = transform_embedding(rna_scaler, rna_reducer, selected_rna).reshape(
        time_points.shape[0],
        particle_ids.shape[0],
        2,
    )
    protein_path_xy = transform_embedding(protein_scaler, protein_reducer, selected_protein).reshape(
        time_points.shape[0],
        particle_ids.shape[0],
        2,
    )
    return {
        "labels": labels,
        "time_points": time_points,
        "terminal_basin": terminal_basin,
        "rna_basis": rna_basis,
        "protein_basis": protein_basis,
        "rna_obs_xy": rna_obs_xy,
        "protein_obs_xy": protein_obs_xy,
        "rna_path_xy": rna_path_xy,
        "protein_path_xy": protein_path_xy,
    }


def draw_umap_panel(
    ax: plt.Axes,
    *,
    xy: np.ndarray,
    labels: np.ndarray,
    path_xy: np.ndarray,
    time_points: np.ndarray,
    terminal_basin: np.ndarray,
    title: str,
    label: str,
    note: str,
) -> dict[str, int]:
    draw_light_observed_cells(ax, xy, labels)
    counts = draw_rollout_paths(ax, path_xy, time_points, terminal_basin)
    clean_embedding_axis(ax)
    add_compact_panel_title(ax, title, label)
    ax.text(
        0.012,
        0.965,
        note,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.0,
        color="#334155",
        zorder=30,
    ).set_path_effects([pe.withStroke(linewidth=2.7, foreground="white", alpha=0.95)])
    return counts


def draw_shared_legend(ax: plt.Axes) -> None:
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#CBD5E1",
            markeredgecolor="none",
            markersize=4.8,
            label="observed cells",
        ),
        Line2D([0], [0], color="#111827", lw=2.0, label="median CITE-PCFM path"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="#111827",
            markerfacecolor="white",
            markeredgewidth=1.0,
            linestyle="None",
            markersize=4.7,
            label="t0-t4",
        ),
    ]
    ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=6.9, handlelength=1.55)


def main() -> None:
    setup()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_rollout_data()

    fig = plt.figure(figsize=(14.2, 11.15), dpi=260)
    gs = fig.add_gridspec(
        3,
        2,
        height_ratios=[0.13, 0.435, 0.435],
        width_ratios=[1.0, 1.18],
        left=0.035,
        right=0.986,
        top=0.965,
        bottom=0.070,
        wspace=0.235,
        hspace=0.290,
    )

    title_ax = fig.add_subplot(gs[0, :])
    title_ax.set_axis_off()
    title_ax.text(0.0, 0.68, "c", fontsize=24, weight="bold", ha="left", va="center", color="#111827")
    title_ax.text(
        0.038,
        0.68,
        "RNA-to-surface-protein relay during hematopoietic commitment",
        fontsize=22,
        weight="bold",
        ha="left",
        va="center",
        color="#111827",
    )
    title_ax.text(
        0.038,
        0.24,
        "The same RNA-started particles are followed in RNA space and decoded into time-resolved CITE surface-protein states.",
        fontsize=11.2,
        ha="left",
        va="center",
        color="#475569",
    )

    ax_rna = fig.add_subplot(gs[1, 0])
    ax_relay = fig.add_subplot(gs[1, 1])
    ax_protein = fig.add_subplot(gs[2, 0])
    ax_mk = fig.add_subplot(gs[2, 1])

    rna_counts = draw_umap_panel(
        ax_rna,
        xy=data["rna_obs_xy"],
        labels=data["labels"],
        path_xy=data["rna_path_xy"],
        time_points=data["time_points"],
        terminal_basin=data["terminal_basin"],
        title="RNA paths from the starting cells",
        label="1",
        note="observed RNA manifold; paths colored by final protein basin",
    )
    draw_shared_legend(ax_rna)

    protein_counts = draw_umap_panel(
        ax_protein,
        xy=data["protein_obs_xy"],
        labels=data["labels"],
        path_xy=data["protein_path_xy"],
        time_points=data["time_points"],
        terminal_basin=data["terminal_basin"],
        title="Protein paths decoded from RNA",
        label="2",
        note="observed CITE protein manifold; generated protein states over time",
    )

    draw_relay(ax_relay)
    restyle_panel_title(ax_relay.texts, "3", "RNA-to-ADT timing waves", title_at_end=True)

    draw_mkp_push(ax_mk)
    restyle_panel_title(ax_mk.texts, "4", "RNA genes ranked by Mk protein gain", title_at_end=False)

    fig.text(
        0.035,
        0.022,
        "UMAP panels are drawn directly from observed RNA/protein manifolds and CITE-PCFM rollout states; trajectory colors denote terminal protein basins.",
        fontsize=7.8,
        color="#64748B",
        ha="left",
    )

    for suffix in [".png", ".pdf", ".svg"]:
        fig.savefig(OUT_BASE.with_suffix(suffix), bbox_inches="tight")
    plt.close(fig)

    manifest = {
        "analysis": "panel_c_rna_to_surface_protein_relay_2x2",
        "rna_panel": f"observed RNA {data['rna_basis']} fitted from X_rna_svd; generated RNA rollout states transformed into the same reducer",
        "protein_panel": f"observed protein {data['protein_basis']} fitted from X_protein_pca; generated protein readout states transformed into the same reducer",
        "trajectory_path": str(TRAJECTORY_PATH),
        "sampled_particle_table": str(COORD_TABLE),
        "terminal_basin_counts_rna_panel": rna_counts,
        "terminal_basin_counts_protein_panel": protein_counts,
        "figures": {
            "png": str(OUT_BASE.with_suffix(".png")),
            "pdf": str(OUT_BASE.with_suffix(".pdf")),
            "svg": str(OUT_BASE.with_suffix(".svg")),
        },
    }
    manifest_path = OUT_DIR / "panel_c_rna_to_surface_protein_relay_2x2_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(OUT_BASE.with_suffix(".png"))
    print(OUT_BASE.with_suffix(".pdf"))
    print(OUT_BASE.with_suffix(".svg"))
    print(manifest_path)


if __name__ == "__main__":
    main()
