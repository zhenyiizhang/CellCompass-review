#!/usr/bin/env python3
"""Fig06P: global observed-vs-generated trend sanity without phenotype labels.

This figure deliberately avoids terminal phenotype classifiers and observed
cell-type labels. It compares the global CITE-PCFM natural rollout trend from
all t0 particles against the global observed CITE time-course trend from all
cells at the five measured time anchors.

The goal is a simple sanity check:
    do generated RNA-gene and ADT-marker trends move in the same directions as
    the real sampled CITE time course, without relying on phenotype assignment?
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
from matplotlib.patches import Rectangle
from scipy import sparse
from scipy.stats import pearsonr, spearmanr

from fig06_rna_protein_regulatory_map import DATA_PATH, OUT_DIR, RNA_DIM, PROTEIN_DIM, feature_masks
from fig06n_rna_protein_relay_heatmap import (
    FIG_DIR,
    available_features,
    fit_feature_decoder,
    load_trajectory,
    relay_cmap,
)
from multistageot_style import clean_metric_axis, setup


def dense(x) -> np.ndarray:
    return x.toarray() if sparse.issparse(x) else np.asarray(x)


def feature_z_lookup(decoder_r2: pd.DataFrame) -> dict[tuple[str, str], tuple[float, float]]:
    return {
        (row.feature_type, row.feature): (float(row.feature_mean), float(row.feature_scale))
        for row in decoder_r2.itertuples()
    }


def zscore_temporal(values: np.ndarray) -> np.ndarray | None:
    values = np.asarray(values, dtype=float)
    if np.any(~np.isfinite(values)):
        return None
    sd = float(np.std(values))
    if sd < 1e-8:
        return None
    return (values - float(np.mean(values))) / sd


def build_global_trends() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    adata = ad.read_h5ad(DATA_PATH)
    time_points, points = load_trajectory()
    rna_names, adt_names, rna_idx, adt_idx, _selected = available_features(adata)
    var_names, rna_mask, protein_mask = feature_masks(adata)

    x_rna = dense(adata.X[:, rna_mask][:, rna_idx]).astype(np.float32)
    x_adt = dense(adata.X[:, protein_mask][:, adt_idx]).astype(np.float32)
    rna_latent = np.asarray(adata.obsm["X_rna_svd"], dtype=np.float32)[:, :RNA_DIM]
    protein_latent = np.asarray(adata.obsm["X_protein_pca"], dtype=np.float32)[:, :PROTEIN_DIM]
    gene_decoder, gene_r2 = fit_feature_decoder(rna_latent, x_rna, rna_names, feature_type="RNA")
    adt_decoder, adt_r2 = fit_feature_decoder(protein_latent, x_adt, adt_names, feature_type="ADT")
    decoder_r2 = pd.concat([gene_r2, adt_r2], ignore_index=True)
    z_lookup = feature_z_lookup(decoder_r2)

    observed_times = sorted(np.asarray(adata.obs["time_point_processed"], dtype=float))
    anchors = sorted(float(t) for t in np.unique(observed_times))
    available = pd.DataFrame(
        {"time_index": np.arange(len(time_points), dtype=int), "rollout_time": np.asarray(time_points, dtype=float)}
    )
    time_to_index = {
        float(t): int(available.iloc[np.argmin(np.abs(available["rollout_time"].to_numpy(dtype=float) - float(t)))]["time_index"])
        for t in anchors
    }

    rows = []
    for t in anchors:
        obs_mask = np.isclose(np.asarray(adata.obs["time_point_processed"], dtype=float), t)
        model_z = points[time_to_index[t]]
        model_r = model_z[:, :RNA_DIM]
        model_q = model_z[:, RNA_DIM : RNA_DIM + PROTEIN_DIM]
        model_gene = gene_decoder.predict(model_r)
        model_adt = adt_decoder.predict(model_q)

        for j, feature in enumerate(rna_names):
            mean, scale = z_lookup[("RNA", feature)]
            scale = scale if scale > 1e-8 else 1.0
            observed = (x_rna[obs_mask, j] - mean) / scale
            rows.append(
                {
                    "feature_type": "RNA",
                    "feature": feature,
                    "time_anchor": float(t),
                    "model_mean_z": float(np.nanmean(model_gene[:, j])),
                    "observed_mean_z": float(np.nanmean(observed)),
                    "observed_n_cells": int(obs_mask.sum()),
                }
            )
        for j, feature in enumerate(adt_names):
            mean, scale = z_lookup[("ADT", feature)]
            scale = scale if scale > 1e-8 else 1.0
            observed = (x_adt[obs_mask, j] - mean) / scale
            rows.append(
                {
                    "feature_type": "ADT",
                    "feature": feature,
                    "time_anchor": float(t),
                    "model_mean_z": float(np.nanmean(model_adt[:, j])),
                    "observed_mean_z": float(np.nanmean(observed)),
                    "observed_n_cells": int(obs_mask.sum()),
                }
            )

    trend = pd.DataFrame(rows)
    summary_rows = []
    delta_rows = []
    for feature_type in ["RNA", "ADT", "all"]:
        sub = trend if feature_type == "all" else trend[trend["feature_type"] == feature_type]
        model_shape = []
        observed_shape = []
        for feature, fsub in sub.groupby(["feature_type", "feature"], sort=False):
            fsub = fsub.sort_values("time_anchor")
            m = zscore_temporal(fsub["model_mean_z"].to_numpy(dtype=float))
            o = zscore_temporal(fsub["observed_mean_z"].to_numpy(dtype=float))
            if m is not None and o is not None:
                model_shape.extend(m.tolist())
                observed_shape.extend(o.tolist())
            first = fsub.iloc[0]
            last = fsub.iloc[-1]
            delta_rows.append(
                {
                    "feature_type": first["feature_type"],
                    "feature": first["feature"],
                    "model_delta_t4_t0": float(last["model_mean_z"] - first["model_mean_z"]),
                    "observed_delta_t4_t0": float(last["observed_mean_z"] - first["observed_mean_z"]),
                }
            )
        dsub = pd.DataFrame(delta_rows)
        if feature_type != "all":
            dsub = dsub[dsub["feature_type"] == feature_type]
        summary_rows.append(
            {
                "feature_type": feature_type,
                "n_features": int(sub[["feature_type", "feature"]].drop_duplicates().shape[0]),
                "temporal_shape_pearson": float(pearsonr(observed_shape, model_shape).statistic),
                "temporal_shape_spearman": float(spearmanr(observed_shape, model_shape).statistic),
                "terminal_delta_pearson": float(pearsonr(dsub["observed_delta_t4_t0"], dsub["model_delta_t4_t0"]).statistic),
                "terminal_delta_spearman": float(spearmanr(dsub["observed_delta_t4_t0"], dsub["model_delta_t4_t0"]).statistic),
            }
        )
    delta = pd.DataFrame(delta_rows).drop_duplicates(["feature_type", "feature"])
    return trend, delta, pd.DataFrame(summary_rows)


def row_scale(mat: np.ndarray) -> np.ndarray:
    out = np.zeros_like(mat, dtype=float)
    for i in range(mat.shape[0]):
        row = mat[i]
        lo = float(np.nanmin(row))
        hi = float(np.nanmax(row))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-8:
            out[i] = 0.5
        else:
            out[i] = (row - lo) / (hi - lo)
    return out


def plot_global_sanity(trend: pd.DataFrame, delta: pd.DataFrame, summary: pd.DataFrame, out_png: Path) -> None:
    setup()
    feature_order = trend[["feature_type", "feature"]].drop_duplicates()
    mat_rows = []
    labels = []
    for row in feature_order.itertuples(index=False):
        sub = trend[(trend["feature_type"] == row.feature_type) & (trend["feature"] == row.feature)].sort_values("time_anchor")
        mat_rows.append(np.r_[sub["observed_mean_z"].to_numpy(dtype=float), sub["model_mean_z"].to_numpy(dtype=float)])
        labels.append(row.feature)
    mat = row_scale(np.vstack(mat_rows))

    fig = plt.figure(figsize=(12.8, 10.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.18, 0.92], height_ratios=[0.52, 0.48], left=0.15, right=0.965, top=0.82, bottom=0.08, wspace=0.34, hspace=0.34)
    ax_hm = fig.add_subplot(gs[:, 0])
    ax_bar = fig.add_subplot(gs[0, 1])
    ax_sc = fig.add_subplot(gs[1, 1])

    im = ax_hm.imshow(mat, cmap=relay_cmap(), vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    ax_hm.axvline(4.5, color="white", lw=1.3)
    ax_hm.set_xticks(np.arange(10))
    ax_hm.set_xticklabels(["obs\nt0", "obs\nt1", "obs\nt2", "obs\nt3", "obs\nt4", "PCFM\nt0", "PCFM\nt1", "PCFM\nt2", "PCFM\nt3", "PCFM\nt4"], fontsize=7.4)
    ax_hm.set_yticks(np.arange(len(labels)))
    ax_hm.set_yticklabels(labels, fontsize=5.7)
    ax_hm.yaxis.tick_right()
    ax_hm.tick_params(length=0, pad=2)
    for spine in ax_hm.spines.values():
        spine.set_visible(False)
    ax_hm.set_title("Global observed and generated marker trends", loc="left", fontsize=10.8, weight="bold")

    n_rna = int((feature_order["feature_type"] == "RNA").sum())
    ax_hm.axhline(n_rna - 0.5, color="white", lw=1.15)
    ax_hm.add_patch(Rectangle((-1.15, -0.5), 0.32, n_rna, color="#2563EB", clip_on=False, lw=0))
    ax_hm.add_patch(Rectangle((-1.15, n_rna - 0.5), 0.32, mat.shape[0] - n_rna, color="#111827", clip_on=False, lw=0))
    ax_hm.text(-1.55, n_rna / 2, "RNA", ha="right", va="center", color="#2563EB", weight="bold", fontsize=8.2, clip_on=False)
    ax_hm.text(-1.55, n_rna + (mat.shape[0] - n_rna) / 2, "ADT", ha="right", va="center", color="#111827", weight="bold", fontsize=8.2, clip_on=False)

    cbar = fig.colorbar(im, ax=ax_hm, fraction=0.025, pad=0.09)
    cbar.set_label("row-scaled mean abundance", fontsize=8.0)
    cbar.ax.tick_params(labelsize=7.5)

    metrics = []
    for metric in ["temporal_shape_pearson", "terminal_delta_pearson"]:
        for feature_type in ["RNA", "ADT", "all"]:
            hit = summary[summary["feature_type"] == feature_type].iloc[0]
            metrics.append({"feature_type": feature_type, "metric": metric.replace("_pearson", "").replace("_", " "), "r": float(hit[metric])})
    mdf = pd.DataFrame(metrics)
    x = np.arange(mdf.shape[0])
    colors = {"RNA": "#2563EB", "ADT": "#111827", "all": "#64748B"}
    ax_bar.bar(x, mdf["r"], color=[colors[t] for t in mdf["feature_type"]], width=0.72)
    ax_bar.axhline(0, color="#CBD5E1", lw=0.8)
    ax_bar.set_ylim(-1, 1)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([f"{r.feature_type}\n{r.metric.split()[0]}" for r in mdf.itertuples()], fontsize=7.1)
    ax_bar.set_ylabel("Pearson r")
    ax_bar.set_title("Overall trend agreement", loc="left", fontsize=10.8, weight="bold")
    clean_metric_axis(ax_bar)

    for feature_type, color in [("RNA", "#2563EB"), ("ADT", "#111827")]:
        sub = delta[delta["feature_type"] == feature_type]
        ax_sc.scatter(
            sub["observed_delta_t4_t0"],
            sub["model_delta_t4_t0"],
            s=28 if feature_type == "RNA" else 34,
            color=color,
            alpha=0.72,
            edgecolor="white",
            linewidth=0.45,
            label=feature_type,
        )
    finite = delta[np.isfinite(delta["observed_delta_t4_t0"]) & np.isfinite(delta["model_delta_t4_t0"])]
    lo = float(np.nanmin([finite["observed_delta_t4_t0"].min(), finite["model_delta_t4_t0"].min(), -0.1]))
    hi = float(np.nanmax([finite["observed_delta_t4_t0"].max(), finite["model_delta_t4_t0"].max(), 0.1]))
    pad = 0.08 * (hi - lo + 1e-8)
    ax_sc.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#94A3B8", lw=1.0, ls="--")
    ax_sc.axhline(0, color="#CBD5E1", lw=0.8)
    ax_sc.axvline(0, color="#CBD5E1", lw=0.8)
    ax_sc.set_xlim(lo - pad, hi + pad)
    ax_sc.set_ylim(lo - pad, hi + pad)
    r_all = summary[summary["feature_type"] == "all"].iloc[0]["terminal_delta_pearson"]
    ax_sc.set_title(f"Global terminal shifts, all markers r={r_all:.2f}", loc="left", fontsize=10.8, weight="bold")
    ax_sc.set_xlabel("observed all-cell trend, t4 - t0 (z)")
    ax_sc.set_ylabel("CITE-PCFM all-particle trend, t4 - t0 (z)")
    ax_sc.legend(frameon=False, loc="lower right")
    clean_metric_axis(ax_sc)

    fig.suptitle(
        "Fig06P. Global RNA/ADT trends match observed CITE time course without phenotype labels",
        x=0.01,
        y=0.965,
        ha="left",
        fontsize=13.2,
        weight="bold",
    )
    fig.text(
        0.01,
        0.895,
        "No terminal phenotype classifier is used: model means are computed over all rollout particles; observed means are computed over all real cells at each measured time.",
        ha="left",
        va="top",
        fontsize=8.7,
        color="#334155",
    )
    fig.savefig(out_png, dpi=360, bbox_inches="tight", facecolor="white")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> dict[str, object]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    trend, delta, summary = build_global_trends()
    trend_path = OUT_DIR / "fig06p_global_model_observed_anchor_trends.csv"
    delta_path = OUT_DIR / "fig06p_global_model_observed_terminal_delta.csv"
    summary_path = OUT_DIR / "fig06p_global_observed_trend_correlation_summary.csv"
    trend.to_csv(trend_path, index=False)
    delta.to_csv(delta_path, index=False)
    summary.to_csv(summary_path, index=False)
    fig_path = FIG_DIR / "fig06p_global_relay_observed_trend_match.png"
    plot_global_sanity(trend, delta, summary, fig_path)
    manifest = {
        "analysis": "fig06p_global_relay_observed_trend_match",
        "figure": str(fig_path),
        "figure_pdf": str(fig_path.with_suffix(".pdf")),
        "tables": {
            "global_anchor_trends": str(trend_path),
            "global_terminal_delta": str(delta_path),
            "global_correlation_summary": str(summary_path),
        },
        "comparison_contract": {
            "model": "all t0 particles rolled forward by natural CITE-PCFM trajectory; no terminal phenotype classifier",
            "observed": "all observed CITE cells at each measured time anchor; no observed phenotype labels",
            "features": "same selected RNA genes and ADT markers as Fig06N",
        },
        "summary": summary.to_dict(orient="records"),
    }
    manifest_path = OUT_DIR / "fig06p_global_relay_observed_trend_match_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    main()
