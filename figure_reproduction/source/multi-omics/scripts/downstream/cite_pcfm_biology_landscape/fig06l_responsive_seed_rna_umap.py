#!/usr/bin/env python3
"""Fig06L: where responsive RNA-program seeds live on the t0 RNA landscape."""

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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from fig06_rna_protein_regulatory_map import DATA_PATH, PHENO_ORDER, RNA_DIM, RNA_PROGRAMS
from multistageot_style import PHENO_COLORS, add_median_labels, clean_embedding_axis, clean_metric_axis, setup

try:
    import umap  # type: ignore

    HAVE_UMAP = True
except Exception:
    HAVE_UMAP = False


ROOT = Path("/Users/zhenyizhang/Documents/GitHub/CytoBridge-agent-runtime-v2-20260309/multi-omics")
FIG_DIR = ROOT / "figures/cite_pcfm_deep_story/biology_landscape_discovery"
OUT_DIR = ROOT / "downstream/cite_pcfm_deep_story/biology_landscape_discovery_rna_protein_regulatory_map"
PARTICLE_PATH = OUT_DIR / "fig06j_program_push_terminal_particle_probabilities.csv"
TRAJECTORY_PATH = ROOT / "training_runs/20260526T174204321821Z__final__cite_paired_coord_fm__campaign_20260-trial_20260526-gse305370_ci/artifacts/evaluation_trajectory.npz"

PROGRAM_ORDER = ["HSC", "Erythroid", "Mk", "Myeloid"]
TARGET_BASIN = {
    "HSC": "HSC",
    "Erythroid": "EryP",
    "Mk": "MkP",
    "Myeloid": "NeuP",
}
RANDOM_STATE = 20260528


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


def responsive_ids(particle_df: pd.DataFrame, program: str) -> tuple[np.ndarray, pd.Series]:
    target = TARGET_BASIN[program]
    sub = particle_df[particle_df["program"] == program]
    plus = sub[sub["condition"] == "+program"].set_index("particle_id")[f"p_{target}"]
    minus = sub[sub["condition"] == "-program"].set_index("particle_id")[f"p_{target}"]
    delta = (plus - minus).dropna()
    threshold = float(delta.quantile(0.75))
    keep = delta[delta >= threshold]
    return keep.index.to_numpy(dtype=int), keep


def build_tables(
    adata: ad.AnnData,
    particle_df: pd.DataFrame,
    obs_xy: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    times = np.asarray(adata.obs["time_point_processed"], dtype=float)
    t0_idx = np.where(np.isclose(times, 0.0))[0]
    traj = np.load(TRAJECTORY_PATH)
    z0 = np.asarray(traj["points_000000"], dtype=np.float32)
    rna = np.asarray(adata.obsm["X_rna_svd"], dtype=np.float32)[:, :RNA_DIM]
    if t0_idx.shape[0] != z0.shape[0]:
        raise ValueError(f"t0 observed count {t0_idx.shape[0]} != trajectory particles {z0.shape[0]}")
    order_check = {
        "t0_observed_count": int(t0_idx.shape[0]),
        "trajectory_particle_count": int(z0.shape[0]),
        "rna_mean_abs_diff_points0_vs_observed_t0": float(np.mean(np.abs(z0[:, :RNA_DIM] - rna[t0_idx]))),
        "rna_max_abs_diff_points0_vs_observed_t0": float(np.max(np.abs(z0[:, :RNA_DIM] - rna[t0_idx]))),
    }

    t0_labels = adata.obs["cell_type"].astype(str).to_numpy()[t0_idx]
    t0_obs_names = adata.obs_names.astype(str).to_numpy()[t0_idx]
    t0_xy = obs_xy[t0_idx]
    particle_rows = []
    composition_rows = []

    all_counts = pd.Series(t0_labels).value_counts().reindex(PHENO_ORDER).fillna(0).astype(int)
    for program in PROGRAM_ORDER:
        ids, delta = responsive_ids(particle_df, program)
        responsive_mask = np.zeros(t0_idx.shape[0], dtype=bool)
        responsive_mask[ids] = True
        responsive_labels = t0_labels[responsive_mask]
        responsive_counts = pd.Series(responsive_labels).value_counts().reindex(PHENO_ORDER).fillna(0).astype(int)
        for label in PHENO_ORDER:
            composition_rows.append(
                {
                    "program": program,
                    "target_basin": TARGET_BASIN[program],
                    "cell_type": label,
                    "all_t0_count": int(all_counts[label]),
                    "all_t0_fraction": float(all_counts[label] / max(1, t0_labels.shape[0])),
                    "responsive_count": int(responsive_counts[label]),
                    "responsive_fraction": float(responsive_counts[label] / max(1, responsive_labels.shape[0])),
                    "enrichment_over_all_t0": float(
                        (responsive_counts[label] / max(1, responsive_labels.shape[0]))
                        / max(all_counts[label] / max(1, t0_labels.shape[0]), 1e-8)
                    ),
                }
            )
        for pid in ids:
            particle_rows.append(
                {
                    "program": program,
                    "target_basin": TARGET_BASIN[program],
                    "particle_id": int(pid),
                    "obs_name": str(t0_obs_names[pid]),
                    "initial_cell_type": str(t0_labels[pid]),
                    "rna_umap_1": float(t0_xy[pid, 0]),
                    "rna_umap_2": float(t0_xy[pid, 1]),
                    "target_probability_delta_plus_vs_minus": float(delta.loc[pid]),
                }
            )
    return pd.DataFrame(particle_rows), pd.DataFrame(composition_rows), order_check


def top_composition_text(comp_df: pd.DataFrame, program: str) -> str:
    sub = comp_df[comp_df["program"] == program].sort_values("responsive_fraction", ascending=False)
    parts = []
    for row in sub.head(2).itertuples():
        parts.append(f"{row.cell_type} {100.0 * row.responsive_fraction:.0f}%")
    return " | ".join(parts)


def plot_figure(
    obs_xy: np.ndarray,
    obs_labels: np.ndarray,
    responsive_df: pd.DataFrame,
    comp_df: pd.DataFrame,
    emb_name: str,
    out_png: Path,
) -> None:
    setup()
    fig = plt.figure(figsize=(14.8, 11.2))
    gs = fig.add_gridspec(
        4,
        2,
        width_ratios=[2.25, 1.0],
        left=0.045,
        right=0.985,
        top=0.90,
        bottom=0.075,
        hspace=0.20,
        wspace=0.12,
    )
    all_t0_comp = (
        comp_df[comp_df["program"] == PROGRAM_ORDER[0]]
        .set_index("cell_type")
        .reindex(PHENO_ORDER)["all_t0_fraction"]
        .to_numpy(dtype=float)
    )

    for row_i, program in enumerate(PROGRAM_ORDER):
        ax = fig.add_subplot(gs[row_i, 0])
        ax.scatter(
            obs_xy[:, 0],
            obs_xy[:, 1],
            s=5.5,
            color="#D1D5DB",
            alpha=0.55,
            edgecolor="none",
            rasterized=True,
            zorder=0,
        )
        add_median_labels(ax, obs_xy, obs_labels, order=PHENO_ORDER, min_cells=70, color_map=PHENO_COLORS, fontsize=7.4)
        sub = responsive_df[responsive_df["program"] == program]
        for label in PHENO_ORDER:
            s = sub[sub["initial_cell_type"] == label]
            if s.empty:
                continue
            ax.scatter(
                s["rna_umap_1"],
                s["rna_umap_2"],
                s=22,
                color=PHENO_COLORS.get(label, "#64748B"),
                alpha=0.88,
                edgecolor="white",
                linewidth=0.35,
                rasterized=True,
                zorder=5,
            )
        ax.set_title(
            f"{program} responsive seeds on t0 RNA {emb_name}",
            loc="left",
            fontsize=11.2,
            weight="bold",
            color=RNA_PROGRAMS[program]["color"],
        )
        ax.text(
            0.01,
            0.98,
            f"target: {TARGET_BASIN[program]} | {top_composition_text(comp_df, program)}",
            transform=ax.transAxes,
            va="top",
            fontsize=8.0,
            color="#334155",
            bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
        )
        clean_embedding_axis(ax)

        ax_bar = fig.add_subplot(gs[row_i, 1])
        sub_comp = comp_df[comp_df["program"] == program].set_index("cell_type").reindex(PHENO_ORDER)
        responsive = sub_comp["responsive_fraction"].to_numpy(dtype=float)
        x = np.arange(2)
        bottom = np.zeros(2, dtype=float)
        for j, label in enumerate(PHENO_ORDER):
            vals = np.asarray([all_t0_comp[j], responsive[j]])
            ax_bar.bar(
                x,
                vals,
                bottom=bottom,
                color=PHENO_COLORS.get(label, "#94A3B8"),
                edgecolor="white",
                linewidth=0.55,
                width=0.62,
                alpha=0.96,
            )
            bottom += vals
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(["all t0", "responsive"], fontsize=8.2)
        ax_bar.set_ylim(0, 1.0)
        ax_bar.set_ylabel("initial cell-type fraction", fontsize=8.2)
        ax_bar.set_title("source identity", loc="left", fontsize=10.2, weight="bold")
        clean_metric_axis(ax_bar, grid_axis="y")

    handles = [Patch(facecolor=PHENO_COLORS[p], edgecolor="white", label=p) for p in PHENO_ORDER]
    fig.legend(handles=handles, frameon=False, ncol=7, loc="lower center", bbox_to_anchor=(0.5, 0.02), fontsize=8.0)
    fig.suptitle(
        "Fig06L. Responsive RNA-program seeds map to early t0 RNA neighborhoods, not a single annotated class",
        x=0.01,
        y=0.985,
        ha="left",
        fontsize=13.4,
        weight="bold",
    )
    fig.text(
        0.01,
        0.945,
        "Responsive seeds are the Fig06K top-quartile cells by target-basin probability gain under +5z versus -5z RNA-program push.",
        ha="left",
        va="top",
        fontsize=9.0,
        color="#334155",
    )
    fig.savefig(out_png, dpi=330, bbox_inches="tight", facecolor="white")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> dict[str, object]:
    if not PARTICLE_PATH.exists():
        raise FileNotFoundError(f"Run fig06j first; missing {PARTICLE_PATH}")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    adata = ad.read_h5ad(DATA_PATH)
    rna = np.asarray(adata.obsm["X_rna_svd"], dtype=np.float32)[:, :RNA_DIM]
    labels = adata.obs["cell_type"].astype(str).to_numpy()
    emb_name, obs_xy = fit_rna_embedding(rna)
    particle_df = pd.read_csv(PARTICLE_PATH)
    responsive_df, comp_df, order_check = build_tables(adata, particle_df, obs_xy)

    responsive_path = OUT_DIR / "fig06l_responsive_seed_rna_umap_particles.csv"
    comp_path = OUT_DIR / "fig06l_responsive_seed_initial_celltype_composition.csv"
    responsive_df.to_csv(responsive_path, index=False)
    comp_df.to_csv(comp_path, index=False)

    fig_path = FIG_DIR / "fig06l_responsive_seed_rna_umap_and_celltype_composition.png"
    plot_figure(obs_xy, labels, responsive_df, comp_df, emb_name, fig_path)

    program_summary = {}
    for program in PROGRAM_ORDER:
        sub = comp_df[comp_df["program"] == program].sort_values("responsive_fraction", ascending=False)
        program_summary[program] = [
            {
                "cell_type": str(row.cell_type),
                "responsive_fraction": float(row.responsive_fraction),
                "all_t0_fraction": float(row.all_t0_fraction),
                "enrichment_over_all_t0": float(row.enrichment_over_all_t0),
            }
            for row in sub.head(3).itertuples()
        ]

    manifest = {
        "analysis": "fig06l_responsive_seed_rna_umap",
        "figure": str(fig_path),
        "figure_pdf": str(fig_path.with_suffix(".pdf")),
        "tables": {
            "responsive_particles": str(responsive_path),
            "celltype_composition": str(comp_path),
        },
        "input_particle_probabilities": str(PARTICLE_PATH),
        "rna_display_basis": f"observed RNA {emb_name} fitted from X_rna_svd",
        "responsive_definition": "top quartile of particles by target-basin probability increase under +5z versus -5z RNA-program push",
        "target_basin": TARGET_BASIN,
        "source_order_check": order_check,
        "top_responsive_celltype_summary": program_summary,
    }
    manifest_path = OUT_DIR / "fig06l_responsive_seed_rna_umap_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    main()
