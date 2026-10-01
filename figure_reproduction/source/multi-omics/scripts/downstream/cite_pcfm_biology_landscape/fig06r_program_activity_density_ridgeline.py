#!/usr/bin/env python3
"""Fig06R: program-activity density along CITE-PCFM pseudotime.

This is a true density-style ridgeline rather than a mean-program curve. For
each HSC-start terminal branch and modality, the script scores individual
rollout particles for directional RNA/ADT marker programs, then plots the
fraction of high-program-activity particles over rollout pseudotime.

No observed phenotype trajectory is used here. The plot is a model-derived
landscape summary of the CITE-PCFM natural rollout.
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
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy import sparse

from fig06_rna_protein_regulatory_map import DATA_PATH, OUT_DIR, PROTEIN_DIM, RNA_DIM, feature_masks
from fig06n_rna_protein_relay_heatmap import (
    BRANCH_ORDER,
    BRANCH_SPECS,
    FIG_DIR,
    available_features,
    fit_feature_decoder,
    hsc_seed_ids,
    load_trajectory,
    terminal_branches,
)
from multistageot_style import setup


TIMECOURSE_PATH = OUT_DIR / "fig06n_hsc_seed_natural_rna_protein_relay_timecourse.csv"
BRANCH_COUNTS_PATH = OUT_DIR / "fig06n_hsc_seed_terminal_branch_counts.csv"
MIN_DELTA = 0.15
ACTIVE_QUANTILE = 0.70


def dense(x) -> np.ndarray:
    return x.toarray() if sparse.issparse(x) else np.asarray(x)


def directional_features(timecourse: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for branch in BRANCH_ORDER:
        for feature_type in ["RNA", "ADT"]:
            sub = timecourse[(timecourse["terminal_branch"] == branch) & (timecourse["feature_type"] == feature_type)]
            scored = []
            for feature, fsub in sub.groupby("feature", sort=False):
                fsub = fsub.sort_values("time_index")
                delta = float(fsub.iloc[-1]["mean_decoded_z"] - fsub.iloc[0]["mean_decoded_z"])
                scored.append((feature, delta))
            if not scored:
                continue
            if branch == "HSC":
                selected = [(f, d) for f, d in scored if d < -MIN_DELTA]
                selected = sorted(selected, key=lambda x: x[1])
                program_direction = "retention-high-early"
            else:
                selected = [(f, d) for f, d in scored if d > MIN_DELTA]
                selected = sorted(selected, key=lambda x: x[1], reverse=True)
                program_direction = "acquisition-high-late"
            if len(selected) < 2:
                selected = sorted(scored, key=lambda x: x[1], reverse=(branch != "HSC"))[:2]
                program_direction += "-weak"
            for rank, (feature, delta) in enumerate(selected, start=1):
                rows.append(
                    {
                        "terminal_branch": branch,
                        "feature_type": feature_type,
                        "feature": feature,
                        "rank": int(rank),
                        "terminal_delta_z": float(delta),
                        "program_direction": program_direction,
                    }
                )
    return pd.DataFrame(rows)


def compute_density_waves() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    adata = ad.read_h5ad(DATA_PATH)
    time_points, points = load_trajectory()
    hsc_ids, seed_check = hsc_seed_ids(adata, points[0])

    rna_names, adt_names, rna_idx, adt_idx, _selected = available_features(adata)
    _var_names, rna_mask, protein_mask = feature_masks(adata)
    x_rna = dense(adata.X[:, rna_mask][:, rna_idx]).astype(np.float32)
    x_adt = dense(adata.X[:, protein_mask][:, adt_idx]).astype(np.float32)
    rna_latent = np.asarray(adata.obsm["X_rna_svd"], dtype=np.float32)[:, :RNA_DIM]
    protein_latent = np.asarray(adata.obsm["X_protein_pca"], dtype=np.float32)[:, :PROTEIN_DIM]
    gene_decoder, gene_r2 = fit_feature_decoder(rna_latent, x_rna, rna_names, feature_type="RNA")
    adt_decoder, adt_r2 = fit_feature_decoder(protein_latent, x_adt, adt_names, feature_type="ADT")
    decoder_r2 = pd.concat([gene_r2, adt_r2], ignore_index=True)

    terminal_q = points[-1][hsc_ids, RNA_DIM : RNA_DIM + PROTEIN_DIM]
    branch_labels, classifier_metrics, _confusion = terminal_branches(adata, terminal_q)

    timecourse = pd.read_csv(TIMECOURSE_PATH)
    selected = directional_features(timecourse)
    idx_lookup = {
        "RNA": {name: i for i, name in enumerate(rna_names)},
        "ADT": {name: i for i, name in enumerate(adt_names)},
    }

    rows = []
    score_rows = []
    for branch in BRANCH_ORDER:
        branch_mask = branch_labels == branch
        if int(branch_mask.sum()) == 0:
            continue
        for feature_type in ["RNA", "ADT"]:
            fsub = selected[(selected["terminal_branch"] == branch) & (selected["feature_type"] == feature_type)]
            features = [f for f in fsub["feature"].tolist() if f in idx_lookup[feature_type]]
            if not features:
                continue
            feature_idx = np.asarray([idx_lookup[feature_type][f] for f in features], dtype=int)
            all_scores = []
            per_time_scores = []
            for z in points:
                if feature_type == "RNA":
                    pred = gene_decoder.predict(z[hsc_ids, :RNA_DIM][branch_mask])
                else:
                    pred = adt_decoder.predict(z[hsc_ids, RNA_DIM : RNA_DIM + PROTEIN_DIM][branch_mask])
                score = pred[:, feature_idx].mean(axis=1)
                per_time_scores.append(score)
                all_scores.append(score)
            all_scores_vec = np.concatenate(all_scores)
            threshold = float(np.nanquantile(all_scores_vec, ACTIVE_QUANTILE))
            active_fraction = []
            mean_score = []
            for score in per_time_scores:
                active_fraction.append(float(np.nanmean(score >= threshold)))
                mean_score.append(float(np.nanmean(score)))
            density = gaussian_filter1d(np.asarray(active_fraction, dtype=float), sigma=1.15, mode="nearest")
            if float(np.nanmax(density)) > 1e-8:
                density = density / float(np.nanmax(density))
            for ti, (t, frac, dens, mean_val) in enumerate(zip(time_points, active_fraction, density, mean_score)):
                rows.append(
                    {
                        "terminal_branch": branch,
                        "feature_type": feature_type,
                        "time_index": int(ti),
                        "rollout_time": float(t),
                        "active_fraction": float(frac),
                        "density_scaled": float(dens),
                        "mean_program_score_z": float(mean_val),
                        "active_threshold_z": threshold,
                        "n_branch_particles": int(branch_mask.sum()),
                        "n_features": int(len(features)),
                        "features": ";".join(features),
                    }
                )
            score_rows.append(
                {
                    "terminal_branch": branch,
                    "feature_type": feature_type,
                    "active_threshold_z": threshold,
                    "n_branch_particles": int(branch_mask.sum()),
                    "n_features": int(len(features)),
                    "features": ";".join(features),
                    "peak_time": float(time_points[int(np.nanargmax(density))]),
                    "start_active_fraction": float(active_fraction[0]),
                    "end_active_fraction": float(active_fraction[-1]),
                }
            )
    manifest_bits = {
        "source_order_check": seed_check,
        "classifier_metrics": classifier_metrics,
        "density_definition": f"Per time point, fraction of branch particles with directional program score above its within-branch {ACTIVE_QUANTILE:.2f} quantile over all rollout times; smoothed and max-scaled for ridgeline display.",
    }
    return pd.DataFrame(rows), selected, pd.DataFrame(score_rows), manifest_bits


def plot_density_ridges(waves: pd.DataFrame, score_summary: pd.DataFrame, out_png: Path) -> None:
    setup()
    available_branches = [
        branch
        for branch in BRANCH_ORDER
        if ((waves["terminal_branch"] == branch) & waves["feature_type"].isin(["RNA", "ADT"])).any()
    ]

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 5.8), sharey=True)
    fig.subplots_adjust(left=0.22, right=0.965, top=0.78, bottom=0.17, wspace=0.13)
    gap = 0.86
    amp = 0.58
    y_positions = np.arange(len(available_branches))[::-1] * gap

    for ax, feature_type in zip(axes, ["RNA", "ADT"]):
        ax.set_title(
            "RNA program density" if feature_type == "RNA" else "ADT surface-program density",
            loc="left",
            fontsize=10.4,
            weight="bold",
            color="#2563EB" if feature_type == "RNA" else "#111827",
        )
        for y0, branch in zip(y_positions, available_branches):
            sub = waves[(waves["terminal_branch"] == branch) & (waves["feature_type"] == feature_type)].sort_values("rollout_time")
            if sub.empty:
                continue
            x = sub["rollout_time"].to_numpy(dtype=float)
            y = sub["density_scaled"].to_numpy(dtype=float)
            color = BRANCH_SPECS[branch]["color"]
            fill_alpha = 0.70 if feature_type == "RNA" else 0.46
            z = 20 + int(y0 * 10)
            ax.hlines(y0, 0, 4, color="#111827", lw=0.72, alpha=0.78, zorder=z)
            poly = ax.fill_between(x, y0, y0 + amp * y, color=color, alpha=fill_alpha, lw=0, zorder=z + 1)
            poly.set_path_effects([pe.SimplePatchShadow(offset=(0.0, -0.7), alpha=0.08), pe.Normal()])
            ax.plot(x, y0 + amp * y, color="#111827", lw=0.65, alpha=0.62, zorder=z + 3)
            ax.plot(x, y0 + amp * y, color=color, lw=1.28, alpha=0.92, zorder=z + 4)

            peak_t = float(
                score_summary[
                    (score_summary["terminal_branch"] == branch)
                    & (score_summary["feature_type"] == feature_type)
                ]["peak_time"].iloc[0]
            )
            ax.plot([peak_t, peak_t], [y0, y0 + amp * float(np.nanmax(y))], color=color, lw=0.75, alpha=0.48, zorder=z + 5)

            if feature_type == "RNA":
                n = int(sub["n_branch_particles"].iloc[0])
                ax.text(
                    -0.16,
                    y0 + 0.18,
                    f"{branch}\nn={n}",
                    transform=ax.get_yaxis_transform(),
                    ha="right",
                    va="center",
                    fontsize=8.5,
                    weight="bold",
                    color=color,
                    linespacing=0.95,
                )

        ax.set_xlim(0, 4)
        ax.set_ylim(-0.35, y_positions.max() + 0.72)
        ax.set_xticks([0, 1, 2, 3, 4])
        ax.set_xticklabels(["t0", "t1", "t2", "t3", "t4"], fontsize=9)
        ax.set_yticks([])
        ax.set_xlabel("CITE-PCFM rollout pseudotime", fontsize=9.0)
        ax.annotate(
            "",
            xy=(4.0, -0.23),
            xytext=(0.0, -0.23),
            arrowprops={"arrowstyle": "->", "lw": 1.18, "color": "#111827"},
            annotation_clip=False,
        )
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(False)

    fig.text(
        0.60,
        0.07,
        "ridge height = density of high program-activity particles",
        ha="center",
        va="center",
        fontsize=8.3,
        color="#334155",
    )

    fig.suptitle(
        "Fig06R. Program-activity density along RNA-initialized CITE-PCFM pseudotime",
        x=0.01,
        y=0.965,
        ha="left",
        fontsize=12.6,
        weight="bold",
    )
    fig.text(
        0.01,
        0.895,
        (
            "Each ridge shows the fraction of HSC-start rollout particles with high directional RNA or ADT program activity, "
            "grouped by generated terminal protein branch. This is model-derived, not observed lineage tracing."
        ),
        ha="left",
        va="top",
        fontsize=8.45,
        color="#334155",
    )
    fig.savefig(out_png, dpi=380, bbox_inches="tight", facecolor="white")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> dict[str, object]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    waves, selected, score_summary, manifest_bits = compute_density_waves()
    waves_path = OUT_DIR / "fig06r_program_activity_density_waves.csv"
    selected_path = OUT_DIR / "fig06r_directional_program_features.csv"
    summary_path = OUT_DIR / "fig06r_program_activity_density_summary.csv"
    waves.to_csv(waves_path, index=False)
    selected.to_csv(selected_path, index=False)
    score_summary.to_csv(summary_path, index=False)
    fig_path = FIG_DIR / "fig06r_program_activity_density_ridgeline.png"
    plot_density_ridges(waves, score_summary, fig_path)
    manifest = {
        "analysis": "fig06r_program_activity_density_ridgeline",
        "figure": str(fig_path),
        "figure_pdf": str(fig_path.with_suffix(".pdf")),
        "tables": {
            "density_waves": str(waves_path),
            "directional_program_features": str(selected_path),
            "density_summary": str(summary_path),
        },
        **manifest_bits,
    }
    manifest_path = OUT_DIR / "fig06r_program_activity_density_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    main()
