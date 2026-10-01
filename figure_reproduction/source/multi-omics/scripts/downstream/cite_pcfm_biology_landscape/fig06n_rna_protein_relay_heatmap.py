#!/usr/bin/env python3
"""Fig06N: natural RNA-to-protein relay heatmap along CITE-PCFM rollout.

This figure is deliberately not a perturbation analysis.  It mirrors a
pseudotime-bin biology panel: HSC-labeled t0 RNA seeds are rolled forward by
the locked CITE-PCFM trajectory, grouped by generated terminal protein branch,
and selected RNA genes plus ADT markers are decoded at each rollout time bin.

Interpretation:
    natural RNA rollout r(t) -> protein readout q(t)=h_eta(r(t),t)
    -> decoded RNA-gene and ADT-marker trajectories

The RNA gene rows are approximate because the benchmark uses a 50D RNA SVD
representation; per-feature holdout R2 values are saved for auditing.
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
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from fig06_rna_protein_regulatory_map import (
    DATA_PATH,
    OUT_DIR,
    PHENO_ORDER,
    PROTEIN_DIM,
    RANDOM_STATE,
    RNA_DIM,
    clean_marker,
    feature_masks,
)
from protein_phenotype_readout import train_mlp_protein_phenotype_readout
from multistageot_style import PHENO_COLORS, setup


ROOT = Path("/Users/zhenyizhang/Documents/GitHub/CytoBridge-agent-runtime-v2-20260309/multi-omics")
FIG_DIR = ROOT / "figures/cite_pcfm_deep_story/biology_landscape_discovery"
TRAJECTORY_PATH = ROOT / "training_runs/20260526T174204321821Z__final__cite_paired_coord_fm__campaign_20260-trial_20260526-gse305370_ci/artifacts/evaluation_trajectory.npz"

RIDGE_ALPHA = 10.0
MIN_BRANCH_CELLS = 35
BRANCH_ORDER = ["HSC", "NeuP", "MasP", "EryP", "MkP"]
BRANCH_SPECS = {
    "HSC": {
        "title": "HSC retention",
        "rna": ["CD34", "MEIS1", "MECOM", "HOXA9", "PROM1", "SPINK2", "PROCR"],
        "adt": ["CD34", "CD49f", "CD123", "CD45RA", "HLA-DR", "CD62L", "CD31"],
        "color": PHENO_COLORS["HSC"],
    },
    "NeuP": {
        "title": "NeuP acquisition",
        "rna": ["MPO", "CTSG", "AZU1", "ELANE", "PRTN3", "CEBPA", "SPI1", "LYZ", "S100A8", "S100A9"],
        "adt": ["CD11b", "CD11c", "CD13", "CD16", "CD64", "CD115", "CD172a", "CD88", "HLA-DR"],
        "color": PHENO_COLORS["NeuP"],
    },
    "MasP": {
        "title": "MasP / myeloid surface",
        "rna": ["MPO", "CTSG", "AZU1", "ELANE", "PRTN3", "CEBPA", "SPI1", "LYZ", "LST1"],
        "adt": ["CD11b", "CD11c", "CD13", "CD64", "CD115", "CD172a", "HLA-DR", "CD123"],
        "color": PHENO_COLORS["MasP"],
    },
    "EryP": {
        "title": "EryP acquisition",
        "rna": ["GATA1", "KLF1", "HBA1", "HBA2", "HBB", "ALAS2", "GYPB"],
        "adt": ["CD71", "CD41", "CD42b", "CD105", "CD49b", "CD235a", "GPA"],
        "color": PHENO_COLORS["EryP"],
    },
    "MkP": {
        "title": "MkP acquisition",
        "rna": ["ITGA2B", "GP9", "GP1BA", "VWF", "NFE2", "FLI1", "TUBB1", "GFI1B", "RUNX1"],
        "adt": ["CD41", "CD42b", "CD62P", "CD49b", "CD61", "CD36"],
        "color": PHENO_COLORS["MkP"],
    },
}


def dense(x) -> np.ndarray:
    return x.toarray() if sparse.issparse(x) else np.asarray(x)


def load_trajectory() -> tuple[np.ndarray, list[np.ndarray]]:
    z = np.load(TRAJECTORY_PATH, allow_pickle=True)
    time_points = np.asarray(z["time_points"], dtype=np.float32)
    points = [np.asarray(z[f"points_{i:06d}"], dtype=np.float32) for i in range(time_points.shape[0])]
    return time_points, points


def hsc_seed_ids(adata: ad.AnnData, points0: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    times = np.asarray(adata.obs["time_point_processed"], dtype=float)
    labels = adata.obs["cell_type"].astype(str).to_numpy()
    t0_idx = np.where(np.isclose(times, 0.0))[0]
    rna = np.asarray(adata.obsm["X_rna_svd"], dtype=np.float32)[:, :RNA_DIM]
    if t0_idx.shape[0] != points0.shape[0]:
        raise ValueError(f"t0 observed count {t0_idx.shape[0]} != trajectory particles {points0.shape[0]}")
    ids = np.where(labels[t0_idx] == "HSC")[0]
    return ids.astype(int), {
        "t0_observed_count": int(t0_idx.shape[0]),
        "trajectory_particle_count": int(points0.shape[0]),
        "hsc_t0_particle_count": int(ids.shape[0]),
        "hsc_t0_fraction": float(ids.shape[0] / max(1, t0_idx.shape[0])),
        "rna_mean_abs_diff_points0_vs_observed_t0": float(np.mean(np.abs(points0[:, :RNA_DIM] - rna[t0_idx]))),
        "rna_max_abs_diff_points0_vs_observed_t0": float(np.max(np.abs(points0[:, :RNA_DIM] - rna[t0_idx]))),
    }


def available_features(adata: ad.AnnData) -> tuple[list[str], list[str], np.ndarray, np.ndarray, dict[str, dict[str, list[str]]]]:
    var_names, rna_mask, protein_mask = feature_masks(adata)
    rna_names_all = [clean_marker(v) for v in var_names[rna_mask]]
    adt_names_all = [clean_marker(v) for v in var_names[protein_mask]]
    rna_lookup = {name: i for i, name in enumerate(rna_names_all)}
    adt_lookup = {name: i for i, name in enumerate(adt_names_all)}

    selected: dict[str, dict[str, list[str]]] = {}
    rna_union: list[str] = []
    adt_union: list[str] = []
    for branch, spec in BRANCH_SPECS.items():
        rna = [g for g in spec["rna"] if g in rna_lookup]
        adt = [m for m in spec["adt"] if m in adt_lookup]
        selected[branch] = {"RNA": rna, "ADT": adt}
        rna_union.extend(rna)
        adt_union.extend(adt)
    rna_union = list(dict.fromkeys(rna_union))
    adt_union = list(dict.fromkeys(adt_union))
    rna_idx = np.asarray([rna_lookup[g] for g in rna_union], dtype=int)
    adt_idx = np.asarray([adt_lookup[m] for m in adt_union], dtype=int)
    return rna_union, adt_union, rna_idx, adt_idx, selected


def fit_feature_decoder(
    latent: np.ndarray,
    features: np.ndarray,
    names: list[str],
    *,
    feature_type: str,
) -> tuple[Ridge, pd.DataFrame]:
    scaler = StandardScaler().fit(features)
    y = scaler.transform(features)
    idx = np.arange(latent.shape[0])
    train_idx, test_idx = train_test_split(idx, test_size=0.25, random_state=RANDOM_STATE)
    audit_decoder = Ridge(alpha=RIDGE_ALPHA)
    audit_decoder.fit(latent[train_idx], y[train_idx])
    pred = audit_decoder.predict(latent[test_idx])
    rows = []
    for j, name in enumerate(names):
        rows.append(
            {
                "feature_type": feature_type,
                "feature": name,
                "decoder_holdout_r2": float(r2_score(y[test_idx, j], pred[:, j])),
                "feature_mean": float(scaler.mean_[j]),
                "feature_scale": float(scaler.scale_[j]),
            }
        )
    decoder = Ridge(alpha=RIDGE_ALPHA)
    decoder.fit(latent, y)
    return decoder, pd.DataFrame(rows)


def terminal_branches(adata: ad.AnnData, terminal_q: np.ndarray) -> tuple[np.ndarray, dict[str, object], pd.DataFrame]:
    labels = adata.obs["cell_type"].astype(str).to_numpy()
    protein = np.asarray(adata.obsm["X_protein_pca"], dtype=np.float32)[:, :PROTEIN_DIM]
    readout, metrics, confusion = train_mlp_protein_phenotype_readout(
        protein,
        labels,
        PHENO_ORDER,
        RANDOM_STATE,
    )
    pred = readout.predict(terminal_q)
    return pred.astype(str), metrics, confusion


def decode_branch_timecourse(
    points: list[np.ndarray],
    hsc_ids: np.ndarray,
    branch_labels: np.ndarray,
    time_points: np.ndarray,
    gene_decoder: Ridge,
    adt_decoder: Ridge,
    rna_names: list[str],
    adt_names: list[str],
    selected: dict[str, dict[str, list[str]]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    branch_rows = []
    branch_counts = pd.Series(branch_labels).value_counts().reindex(PHENO_ORDER).fillna(0).astype(int)
    for branch in PHENO_ORDER:
        branch_rows.append({"terminal_branch": branch, "n_hsc_seed_particles": int(branch_counts[branch])})

    for branch in BRANCH_ORDER:
        branch_mask = branch_labels == branch
        n_branch = int(branch_mask.sum())
        if n_branch < MIN_BRANCH_CELLS:
            continue
        features = [("RNA", f) for f in selected[branch]["RNA"]] + [("ADT", f) for f in selected[branch]["ADT"]]
        for ti, (t, z) in enumerate(zip(time_points, points)):
            r = z[hsc_ids, :RNA_DIM][branch_mask]
            q = z[hsc_ids, RNA_DIM : RNA_DIM + PROTEIN_DIM][branch_mask]
            gene_pred = gene_decoder.predict(r)
            adt_pred = adt_decoder.predict(q)
            gene_map = {name: float(gene_pred[:, j].mean()) for j, name in enumerate(rna_names)}
            adt_map = {name: float(adt_pred[:, j].mean()) for j, name in enumerate(adt_names)}
            for feature_type, feature in features:
                val = gene_map[feature] if feature_type == "RNA" else adt_map[feature]
                rows.append(
                    {
                        "terminal_branch": branch,
                        "branch_title": BRANCH_SPECS[branch]["title"],
                        "n_hsc_seed_particles": n_branch,
                        "rollout_time": float(t),
                        "time_index": int(ti),
                        "feature_type": feature_type,
                        "feature": feature,
                        "mean_decoded_z": val,
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(branch_rows)


def add_row_scaled_values(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for _, sub in df.groupby(["terminal_branch", "feature_type", "feature"], sort=False):
        vals = sub["mean_decoded_z"].to_numpy(dtype=float)
        lo = float(np.nanmin(vals))
        hi = float(np.nanmax(vals))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-8:
            scaled = np.full(vals.shape, 0.5, dtype=float)
        else:
            scaled = (vals - lo) / (hi - lo)
        s = sub.copy()
        s["row_minmax_scaled"] = scaled
        s["row_dynamic_range_z"] = hi - lo
        parts.append(s)
    return pd.concat(parts, ignore_index=True)


def relay_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "cite_relay_magma_soft",
        ["#180F3D", "#5F187F", "#B5367A", "#EF7D4F", "#FECF78", "#FFF7BC"],
        N=256,
    )


def plot_relay_heatmap(df: pd.DataFrame, branch_counts: pd.DataFrame, decoder_r2: pd.DataFrame, out_png: Path) -> None:
    setup()
    available_branches = [b for b in BRANCH_ORDER if b in set(df["terminal_branch"])]
    row_blocks: list[dict[str, object]] = []
    mats: list[np.ndarray] = []
    row_labels: list[str] = []
    branch_spans: list[tuple[str, int, int]] = []
    modality_spans: list[tuple[str, str, int, int]] = []
    start = 0
    cmap = relay_cmap()

    for branch in available_branches:
        branch_start = start
        for feature_type in ["RNA", "ADT"]:
            sub = df[(df["terminal_branch"] == branch) & (df["feature_type"] == feature_type)].copy()
            if sub.empty:
                continue
            feature_order = sub[["feature_type", "feature"]].drop_duplicates()
            idx = pd.MultiIndex.from_frame(feature_order)
            mat = (
                sub.pivot_table(index=["feature_type", "feature"], columns="time_index", values="row_minmax_scaled")
                .reindex(idx)
                .to_numpy(dtype=float)
            )
            mats.append(mat)
            row_labels.extend(feature_order["feature"].astype(str).tolist())
            mod_start = start
            start += mat.shape[0]
            modality_spans.append((branch, feature_type, mod_start, start - 1))
        branch_spans.append((branch, branch_start, start - 1))

    full_mat = np.vstack(mats)
    fig, ax = plt.subplots(figsize=(9.7, 13.4))
    fig.subplots_adjust(left=0.25, right=0.79, top=0.85, bottom=0.07)
    im = ax.imshow(full_mat, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto", interpolation="nearest")

    ax.set_xticks([0, 10, 20, 30, 40])
    ax.set_xticklabels(["t0", "t1", "t2", "t3", "t4"], fontsize=8.8)
    ax.set_xlabel("CITE-PCFM rollout pseudotime bins", fontsize=9.0)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=6.1)
    ax.yaxis.tick_right()
    ax.tick_params(axis="both", length=0, pad=2)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for branch, span_start, span_end in branch_spans:
        ax.axhline(span_start - 0.5, color="white", lw=1.15)
        n_branch = int(branch_counts.set_index("terminal_branch").loc[branch, "n_hsc_seed_particles"])
        center = 0.5 * (span_start + span_end)
        ax.add_patch(
            Rectangle(
                (-4.0, span_start - 0.5),
                0.5,
                span_end - span_start + 1,
                color=BRANCH_SPECS[branch]["color"],
                clip_on=False,
                lw=0,
            )
        )
        ax.text(
            -5.2,
            center,
            f"{branch}\nn={n_branch}",
            ha="right",
            va="center",
            fontsize=8.4,
            weight="bold",
            color=BRANCH_SPECS[branch]["color"],
            clip_on=False,
            linespacing=1.05,
        )
    ax.axhline(full_mat.shape[0] - 0.5, color="white", lw=1.15)

    for branch, feature_type, span_start, span_end in modality_spans:
        color = "#2563EB" if feature_type == "RNA" else "#111827"
        ax.add_patch(
            Rectangle(
                (-3.25, span_start - 0.5),
                0.26,
                span_end - span_start + 1,
                color=color,
                clip_on=False,
                lw=0,
            )
        )
        if span_end - span_start >= 3:
            ax.text(
                -2.72,
                0.5 * (span_start + span_end),
                feature_type,
                ha="left",
                va="center",
                fontsize=6.9,
                color=color,
                weight="bold",
                clip_on=False,
            )

    cax = fig.add_axes([0.875, 0.27, 0.017, 0.40])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("row-scaled decoded abundance", fontsize=8.5)
    cbar.ax.tick_params(labelsize=7.5)
    med_rna = float(decoder_r2[decoder_r2["feature_type"] == "RNA"]["decoder_holdout_r2"].median())
    med_adt = float(decoder_r2[decoder_r2["feature_type"] == "ADT"]["decoder_holdout_r2"].median())
    fig.suptitle(
        "Fig06N. Natural CITE-PCFM rollout resolves branch-specific RNA-to-protein relay dynamics",
        x=0.01,
        y=0.975,
        ha="left",
        fontsize=13.5,
        weight="bold",
    )
    fig.text(
        0.01,
        0.925,
        (
            "HSC-labeled t0 RNA seeds are grouped by generated terminal protein branch. "
            "Rows show decoded RNA genes and ADT markers over natural rollout time; left bars annotate terminal branch and modality. "
            f"Median decoder R2: RNA {med_rna:.2f}, ADT {med_adt:.2f}."
        ),
        ha="left",
        va="top",
        fontsize=8.8,
        color="#334155",
    )
    fig.savefig(out_png, dpi=330, bbox_inches="tight", facecolor="white")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> dict[str, object]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    adata = ad.read_h5ad(DATA_PATH)
    time_points, points = load_trajectory()
    hsc_ids, seed_check = hsc_seed_ids(adata, points[0])

    rna_names, adt_names, rna_idx, adt_idx, selected = available_features(adata)
    var_names, rna_mask, protein_mask = feature_masks(adata)
    x_rna = dense(adata.X[:, rna_mask][:, rna_idx]).astype(np.float32)
    x_adt = dense(adata.X[:, protein_mask][:, adt_idx]).astype(np.float32)
    rna_latent = np.asarray(adata.obsm["X_rna_svd"], dtype=np.float32)[:, :RNA_DIM]
    protein_latent = np.asarray(adata.obsm["X_protein_pca"], dtype=np.float32)[:, :PROTEIN_DIM]
    gene_decoder, gene_r2 = fit_feature_decoder(rna_latent, x_rna, rna_names, feature_type="RNA")
    adt_decoder, adt_r2 = fit_feature_decoder(protein_latent, x_adt, adt_names, feature_type="ADT")
    decoder_r2 = pd.concat([gene_r2, adt_r2], ignore_index=True)

    terminal_q = points[-1][hsc_ids, RNA_DIM : RNA_DIM + PROTEIN_DIM]
    branch_labels, classifier_metrics, confusion = terminal_branches(adata, terminal_q)
    timecourse_df, branch_counts = decode_branch_timecourse(
        points,
        hsc_ids,
        branch_labels,
        time_points,
        gene_decoder,
        adt_decoder,
        rna_names,
        adt_names,
        selected,
    )
    timecourse_df = add_row_scaled_values(timecourse_df)

    timecourse_path = OUT_DIR / "fig06n_hsc_seed_natural_rna_protein_relay_timecourse.csv"
    r2_path = OUT_DIR / "fig06n_natural_relay_feature_decoder_r2.csv"
    branch_path = OUT_DIR / "fig06n_hsc_seed_terminal_branch_counts.csv"
    confusion_path = OUT_DIR / "fig06n_terminal_branch_readout_confusion.csv"
    timecourse_df.to_csv(timecourse_path, index=False)
    decoder_r2.to_csv(r2_path, index=False)
    branch_counts.to_csv(branch_path, index=False)
    confusion.to_csv(confusion_path)

    fig_path = FIG_DIR / "fig06n_hsc_seed_natural_rna_protein_relay_heatmap.png"
    plot_relay_heatmap(timecourse_df, branch_counts, decoder_r2, fig_path)

    manifest = {
        "analysis": "fig06n_hsc_seed_natural_rna_protein_relay_heatmap",
        "figure": str(fig_path),
        "figure_pdf": str(fig_path.with_suffix(".pdf")),
        "tables": {
            "timecourse": str(timecourse_path),
            "feature_decoder_r2": str(r2_path),
            "terminal_branch_counts": str(branch_path),
            "terminal_branch_readout_confusion": str(confusion_path),
        },
        "data_h5ad": str(DATA_PATH),
        "trajectory_path": str(TRAJECTORY_PATH),
        "seed_definition": "observed t0 cells with cell_type == HSC; same particle order as evaluation_trajectory points_000000",
        "condition": "natural CITE-PCFM rollout, no RNA-program perturbation",
        "branch_definition": "terminal protein branch predicted from generated terminal protein PCA using shared observed-protein MLP phenotype readout",
        "decoder_contract": {
            "RNA": "Ridge decoder from observed X_rna_svd to selected RNA-gene z-scores; 50D RNA SVD means gene trends are approximate",
            "ADT": "Ridge decoder from observed X_protein_pca to selected ADT-marker z-scores",
            "ridge_alpha": RIDGE_ALPHA,
            "row_scaling": "heatmap displays row-wise min-max scaling over rollout time within each terminal branch",
        },
        "source_order_check": seed_check,
        "classifier_metrics": classifier_metrics,
        "selected_features": selected,
        "decoder_summary": {
            "median_rna_feature_r2": float(gene_r2["decoder_holdout_r2"].median()),
            "median_adt_feature_r2": float(adt_r2["decoder_holdout_r2"].median()),
            "mean_rna_feature_r2": float(gene_r2["decoder_holdout_r2"].mean()),
            "mean_adt_feature_r2": float(adt_r2["decoder_holdout_r2"].mean()),
        },
        "semantic_warning": "This is a model-derived decoded trajectory heatmap, not measured single-cell time-lapse expression.",
    }
    manifest_path = OUT_DIR / "fig06n_natural_rna_protein_relay_heatmap_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    main()
