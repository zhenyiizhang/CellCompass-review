#!/usr/bin/env python3
"""Assemble a draft NBT-style panel C for CITE-PCFM biology.

This is a story/layout composite. The trajectory, relay and Mk-gene panels are
drawn from source data/tables so PDF/SVG exports remain editable vector art.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/cytobridge_mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/cytobridge_xdg_cache")
for key in ["MPLCONFIGDIR", "XDG_CACHE_HOME"]:
    Path(os.environ[key]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import anndata as ad

from fig01h_protein_rollout_on_fig01a_style import (
    COORD_TABLE as C1_COORD_TABLE,
    DATA_PATH as C1_DATA_PATH,
    PROTEIN_DIM as C1_PROTEIN_DIM,
    TRAJECTORY_PATH as C1_TRAJECTORY_PATH,
    draw_observed_fig01a_style,
    draw_rollout_paths,
    fit_observed_protein_umap,
    load_sampled_particle_ids,
    transform_embedding,
)


ROOT = Path("/Users/zhenyizhang/Documents/GitHub/CytoBridge-agent-runtime-v2-20260309/multi-omics")
FIG_DIR = ROOT / "figures/cite_pcfm_deep_story/biology_landscape_discovery"
TABLE_DIR = ROOT / "downstream/cite_pcfm_deep_story/biology_landscape_discovery_rna_protein_regulatory_map"
OUT_BASE = FIG_DIR / "panel_c_rna_to_surface_protein_relay_draft"
COMPACT_OUT_BASE = FIG_DIR / "panel_c_rna_to_surface_protein_relay_compact_a4"

PHENO_COLORS = {
    "HSC": "#2F855A",
    "NeuP": "#28BFC4",
    "MasP": "#D62728",
    "EryP": "#F28E2B",
    "MkP": "#7B61B3",
    "MoP": "#8C564B",
    "BP": "#1F77B4",
}
BRANCH_TITLES = {
    "HSC": "HSC retention",
    "NeuP": "NeuP acquisition",
    "MasP": "MasP / myeloid surface",
    "EryP": "EryP acquisition",
    "MkP": "MkP acquisition",
}
BRANCH_ORDER = ["HSC", "NeuP", "MasP", "EryP", "MkP"]
REPRESENTATIVE_PATHS = ("MoP", "MasP", "EryP", "MkP")
OBSERVED_LABELS_FOR_C1 = ("HSC", "NeuP", "MasP", "EryP", "MkP", "MoP")
GENE_CLASS_COLORS = {
    "Mk effector": "#0F766E",
    "Mk marker": "#DB2777",
    "Mk regulator": "#7C3AED",
    "Ery-Mk regulator": "#EA580C",
}


def add_panel_title(ax: plt.Axes, title: str, label: str) -> None:
    ax.text(
        0.0,
        1.035,
        label,
        transform=ax.transAxes,
        fontsize=13,
        weight="bold",
        ha="left",
        va="bottom",
        color="#111827",
    )
    ax.text(
        0.075,
        1.035,
        title,
        transform=ax.transAxes,
        fontsize=12.2,
        weight="bold",
        ha="left",
        va="bottom",
        color="#111827",
    )


def add_compact_number(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.0,
        1.01,
        label,
        transform=ax.transAxes,
        fontsize=8.3,
        weight="bold",
        ha="left",
        va="bottom",
        color="#111827",
    )


def draw_observed_points_light(ax: plt.Axes, xy: np.ndarray, labels: np.ndarray) -> None:
    for label, color in PHENO_COLORS.items():
        mask = labels.astype(str) == label
        if not np.any(mask):
            continue
        ax.scatter(
            xy[mask, 0],
            xy[mask, 1],
            s=3.0,
            facecolor=color,
            edgecolor="none",
            alpha=0.18,
            zorder=1,
        )


def add_start_zoom_inset(
    ax: plt.Axes,
    *,
    obs_emb: np.ndarray,
    labels: np.ndarray,
    path_xy: np.ndarray,
    time_points: np.ndarray,
    terminal_basin: np.ndarray,
) -> None:
    """Add a small zoom for the crowded shared-start portion of the paths."""
    early_idx = np.where(time_points <= 0.8)[0]
    if early_idx.size == 0:
        early_idx = np.arange(min(3, time_points.shape[0]))

    early_medians = []
    for label in REPRESENTATIVE_PATHS:
        ids = np.where(terminal_basin == label)[0]
        if ids.size == 0:
            continue
        early_medians.append(np.nanmedian(path_xy[early_idx][:, ids, :], axis=1))
    if not early_medians:
        return
    early_xy = np.concatenate(early_medians, axis=0)
    early_xy = early_xy[np.isfinite(early_xy).all(axis=1)]
    if early_xy.size == 0:
        return

    lo = np.nanmin(early_xy, axis=0)
    hi = np.nanmax(early_xy, axis=0)
    span = np.maximum(hi - lo, 0.22)
    pad = span * 0.58
    xlim = (float(lo[0] - pad[0]), float(hi[0] + pad[0]))
    ylim = (float(lo[1] - pad[1]), float(hi[1] + pad[1]))

    axins = ax.inset_axes([0.615, 0.742, 0.29, 0.195])
    axins.set_facecolor((1, 1, 1, 0.92))
    draw_observed_points_light(axins, obs_emb, labels)
    draw_rollout_paths(
        axins,
        path_xy,
        time_points,
        terminal_basin,
        include_labels=REPRESENTATIVE_PATHS,
        show_arrows=False,
        show_faint=False,
        marker_time_values=(0, 0.4, 0.8),
        marker_size=18,
    )
    axins.set_xlim(*xlim)
    axins.set_ylim(*ylim)
    axins.set_xticks([])
    axins.set_yticks([])
    axins.set_aspect("equal", adjustable="box")
    for spine in axins.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.7)
        spine.set_edgecolor("#CBD5E1")


def draw_trajectory(ax: plt.Axes, *, show_title: bool = True, compact: bool = False) -> None:
    adata = ad.read_h5ad(C1_DATA_PATH)
    protein = np.asarray(adata.obsm["X_protein_pca"], dtype=np.float32)[:, :C1_PROTEIN_DIM]
    labels = adata.obs["cell_type"].astype(str).to_numpy()
    _, scaler, reducer, obs_emb = fit_observed_protein_umap(protein)

    particle_ids, terminal_basin = load_sampled_particle_ids(C1_COORD_TABLE)
    traj = np.load(C1_TRAJECTORY_PATH)
    time_points = np.asarray(traj["time_points"], dtype=float)
    selected_proteins = np.concatenate(
        [
            np.asarray(traj[f"points_{i:06d}"], dtype=np.float32)[particle_ids, 50 : 50 + C1_PROTEIN_DIM]
            for i in range(time_points.shape[0])
        ],
        axis=0,
    )
    selected_xy = transform_embedding(scaler, reducer, selected_proteins)
    path_xy = selected_xy.reshape(time_points.shape[0], particle_ids.shape[0], 2)

    draw_observed_fig01a_style(ax, obs_emb, labels, label_order=OBSERVED_LABELS_FOR_C1 if compact else None)
    draw_rollout_paths(
        ax,
        path_xy,
        time_points,
        terminal_basin,
        include_labels=REPRESENTATIVE_PATHS if compact else None,
        skip_labels=(),
        show_arrows=not compact,
        max_faint_per_basin=3 if compact else 6,
        marker_size=23 if compact else 30,
    )
    if compact:
        add_start_zoom_inset(
            ax,
            obs_emb=obs_emb,
            labels=labels,
            path_xy=path_xy,
            time_points=time_points,
            terminal_basin=terminal_basin,
        )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")
    for spine in ax.spines.values():
        spine.set_visible(False)
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#D1D5DB",
            markeredgecolor="#111111",
            markersize=4.5,
            label="observed protein cells",
        ),
        Line2D([0], [0], color="#111827", lw=1.9, label="representative median paths" if compact else "median generated path"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="#111827",
            markerfacecolor="white",
            markeredgewidth=1.05,
            linestyle="None",
            markersize=4.4,
            label="t0-t4",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="lower left",
        frameon=False,
        fontsize=5.6 if compact else 6.6,
        handlelength=1.25 if compact else 1.45,
        borderpad=0.1,
        labelspacing=0.25,
    )
    if show_title:
        add_panel_title(ax, "RNA-initialized trajectories on the protein landscape", "1")
    elif compact:
        add_compact_number(ax, "1")
        ax.text(
            0.055,
            0.985,
            "observed CITE protein UMAP\npaths start from day-2 RNA only",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=5.8,
            color="#475569",
            linespacing=0.92,
            zorder=40,
        )


def draw_relay(ax: plt.Axes, *, show_title: bool = True, compact: bool = False) -> None:
    waves = pd.read_csv(TABLE_DIR / "fig06q_relay_program_ridgeline_waves.csv")
    counts = pd.read_csv(TABLE_DIR / "fig06n_hsc_seed_terminal_branch_counts.csv").set_index("terminal_branch")

    row_specs: list[tuple[str, str]] = []
    for branch in BRANCH_ORDER:
        for feature_type in ["RNA", "ADT"]:
            if ((waves["terminal_branch"] == branch) & (waves["feature_type"] == feature_type)).any():
                row_specs.append((branch, feature_type))

    y_gap = 0.68 if compact else 0.74
    amp = 0.49 if compact else 0.55
    y_positions = np.arange(len(row_specs))[::-1] * y_gap

    for y0, (branch, feature_type) in zip(y_positions, row_specs):
        sub = waves[(waves["terminal_branch"] == branch) & (waves["feature_type"] == feature_type)].sort_values("rollout_time")
        x = sub["rollout_time"].to_numpy(float)
        y = sub["program_wave"].to_numpy(float)
        color = PHENO_COLORS[branch]
        alpha = 0.76 if feature_type == "RNA" else 0.36
        edge_alpha = 0.96 if feature_type == "RNA" else 0.72
        ax.hlines(y0, 0, 4, color="#111827", lw=0.55, alpha=0.75)
        ax.fill_between(x, y0, y0 + amp * y, color=color, alpha=alpha, lw=0)
        ax.plot(x, y0 + amp * y, color=color, lw=1.35, alpha=edge_alpha)
        peak_t = float(sub["peak_time"].iloc[0])
        ax.plot([peak_t, peak_t], [y0, y0 + amp * np.nanmax(y)], color=color, lw=0.75, alpha=0.55)

        if feature_type == "RNA":
            n = int(counts.loc[branch, "n_hsc_seed_particles"]) if branch in counts.index else 0
            ax.text(
                -0.42 if compact else -0.50,
                y0 + 0.18,
                f"{branch}\nn={n}" if compact else f"{BRANCH_TITLES[branch]}\n{branch}, n={n}",
                ha="right",
                va="center",
                color=color,
                fontsize=5.7 if compact else 8.35,
                weight="bold",
                linespacing=0.98,
            )
        ax.text(
            -0.08,
            y0 + 0.18,
            feature_type,
            ha="right",
            va="center",
            color="#2563EB" if feature_type == "RNA" else "#111827",
            fontsize=5.7 if compact else 7.7,
            weight="bold",
        )

    ax.set_xlim((-0.54, 4.42) if compact else (-0.65, 4.55))
    ax.set_ylim(-0.35, y_positions.max() + 0.85)
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_xticklabels(["t0", "t1", "t2", "t3", "t4"], fontsize=5.9 if compact else 8.3)
    ax.set_yticks([])
    ax.set_xlabel("CITE-PCFM pseudotime", fontsize=6.4 if compact else 8.8, labelpad=1.5)
    ax.annotate(
        "",
        xy=(4.0, -0.24),
        xytext=(0.0, -0.24),
        arrowprops={"arrowstyle": "->", "lw": 1.2, "color": "#111827"},
        annotation_clip=False,
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)
    # The ridges are independently scaled per feature row before branch-level
    # averaging; add a visible scale cue so the panel is not read as absolute
    # abundance.
    x0 = 4.12 if compact else 4.24
    y0, y1 = y_positions.max() - 0.02, y_positions.max() + 0.55
    ax.plot([x0, x0], [y0, y1], color="#475569", lw=0.8, clip_on=False)
    ax.plot([x0 - 0.05, x0], [y0, y0], color="#475569", lw=0.8, clip_on=False)
    ax.plot([x0 - 0.05, x0], [y1, y1], color="#475569", lw=0.8, clip_on=False)
    ax.text(x0 + 0.04, y0, "0", fontsize=4.9 if compact else 6.6, va="center", ha="left", color="#475569")
    ax.text(x0 + 0.04, y1, "1", fontsize=4.9 if compact else 6.6, va="center", ha="left", color="#475569")
    ax.text(
        x0 + (0.12 if compact else 0.18),
        (y0 + y1) * 0.5,
        "relative\nscale",
        fontsize=4.9 if compact else 6.7,
        va="center",
        ha="left",
        color="#475569",
        linespacing=0.9,
    )
    ax.text(
        0.56,
        0.985,
        "row-scaled relative timing",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.5 if compact else 7.3,
        color="#475569",
    )
    if show_title:
        add_panel_title(ax, "Relative RNA-to-ADT timing waves", "2")
    elif compact:
        add_compact_number(ax, "2")


def draw_mkp_push(ax: plt.Axes, *, show_title: bool = True, compact: bool = False) -> None:
    df = pd.read_csv(TABLE_DIR / "fig06s_mkp_hsc_gene_push_deltas.csv").copy()
    df = df.sort_values("plus_minus_baseline_p_MkP_pp", ascending=True)
    y = np.arange(df.shape[0], dtype=float)

    ax.set_axis_off()
    if show_title:
        add_panel_title(ax, "RNA genes ranked by final Mk protein-phenotype gain", "3")
        ax.text(
            0.075,
            0.985,
            "HSC-start +5z RNA query; final CITE-PCFM protein phenotype",
            transform=ax.transAxes,
            fontsize=7.4,
            color="#475569",
            ha="left",
            va="top",
        )
    elif compact:
        add_compact_number(ax, "3")

    ax_gain = ax.inset_axes([0.14, 0.13, 0.80, 0.70] if compact else [0.11, 0.10, 0.82, 0.78])
    values = df["plus_minus_baseline_p_MkP_pp"].astype(float)
    negative_reference = {"KLF1"}
    mk_color = "#0F766E"
    neg_color = "#EA580C"

    ax_gain.axvline(0, color="#CBD5E1", lw=0.9, zorder=0)
    for yi, (_, row) in zip(y, df.iterrows()):
        gene = str(row["gene"])
        value = float(values.loc[row.name])
        if gene in negative_reference:
            color = neg_color
            alpha = 0.42
            size = 22 if compact else 46
            lw = 2.0 if compact else 3.0
            z = 4
        else:
            color = mk_color
            alpha = 0.42
            size = 18 if compact else 38
            lw = 1.65 if compact else 2.65
            z = 3
        ax_gain.hlines(yi, 0, value, color=color, lw=lw, alpha=alpha, zorder=1)
        ax_gain.scatter(value, yi, s=size, color=color, edgecolor="white", linewidth=0.45 if compact else 0.75, zorder=z)

    ax_gain.set_xlim(-2.0, 6.8)
    ax_gain.set_ylim(-0.6, df.shape[0] - 0.4)
    ax_gain.set_title(
        "Final Mk protein phenotype gain",
        fontsize=7.0 if compact else 8.7,
        weight="bold",
        color="#111827",
        pad=3 if compact else 5,
    )
    ax_gain.set_xlabel("change in terminal MkP probability (percentage points)", fontsize=5.6 if compact else 7.3, labelpad=1.5)
    ax_gain.set_yticks(y)
    ax_gain.set_yticklabels(df["gene"], fontsize=5.6 if compact else 7.3)
    for tick, (_, row) in zip(ax_gain.get_yticklabels(), df.iterrows()):
        gene = str(row["gene"])
        if gene in negative_reference:
            tick.set_color(neg_color)
            tick.set_fontweight("bold")
        else:
            tick.set_color(mk_color)
            tick.set_fontweight("bold" if gene in {"THBS1", "TUBB1", "GP9", "GP1BA", "VWF", "ITGA2B", "NFE2"} else "normal")

    ax_gain.grid(axis="x", color="#E5E7EB", lw=0.65)
    ax_gain.spines[["top", "right", "left"]].set_visible(False)
    ax_gain.tick_params(axis="x", labelsize=5.2 if compact else 6.8, length=2)
    ax_gain.tick_params(axis="y", length=0)
    ax_gain.set_axisbelow(True)
    handles = [
        Line2D([0], [0], marker="o", color=mk_color, markerfacecolor=mk_color, linestyle="None", markersize=4.3, label="candidate Mk-associated RNA genes"),
        Line2D([0], [0], marker="o", color=neg_color, markerfacecolor=neg_color, linestyle="None", markersize=4.3, label="KLF1 erythroid-biased TF"),
    ]
    ax_gain.legend(
        handles=handles,
        frameon=False,
        fontsize=5.1 if compact else 6.8,
        loc="lower right",
        handlelength=0.8,
        handletextpad=0.35,
        borderpad=0.1,
        labelspacing=0.18,
    )


def main() -> None:
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

    fig = plt.figure(figsize=(8.27, 4.75), dpi=360)
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.02, 1.16],
        height_ratios=[0.50, 0.50],
        left=0.035,
        right=0.99,
        top=0.895,
        bottom=0.082,
        wspace=0.10,
        hspace=0.22,
    )

    fig.text(0.035, 0.953, "c", fontsize=15, weight="bold", ha="left", va="center", color="#111827")
    fig.text(
        0.066,
        0.953,
        "RNA-to-surface-protein relay",
        fontsize=14,
        weight="bold",
        ha="left",
        va="center",
        color="#111827",
    )

    draw_trajectory(fig.add_subplot(gs[:, 0]), show_title=False, compact=True)
    draw_relay(fig.add_subplot(gs[0, 1]), show_title=False, compact=True)
    draw_mkp_push(fig.add_subplot(gs[1, 1]), show_title=False, compact=True)

    for suffix in [".png", ".pdf", ".svg"]:
        fig.savefig(COMPACT_OUT_BASE.with_suffix(suffix), bbox_inches="tight")
    plt.close(fig)
    print(COMPACT_OUT_BASE.with_suffix(".png"))
    print(COMPACT_OUT_BASE.with_suffix(".pdf"))
    print(COMPACT_OUT_BASE.with_suffix(".svg"))


if __name__ == "__main__":
    main()
