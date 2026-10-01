#!/usr/bin/env python3
"""Protein phenotype river for CITE-PCFM generated protein dynamics.

This figure deliberately has one semantic contract:

1. The observed panel shows source-annotated phenotype fractions in the real
   CITE-seq cells at the five measured culture times.
2. The generated panel shows CITE-PCFM protein states classified in observed
   protein PCA space by an audited MLP probability readout. These are protein phenotype / surface
   basin proportions, not multimodal cell-type proportions.

Only the five biological timepoints are plotted to avoid over-interpreting
classifier jitter from dense rollout steps.
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
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from multistageot_style import PHENO_COLORS, PHENO_ORDER
from protein_phenotype_readout import ProteinPhenotypeReadout, train_mlp_protein_phenotype_readout


ROOT = Path("/Users/zhenyizhang/Documents/GitHub/CytoBridge-agent-runtime-v2-20260309/multi-omics")
DATA_PATH = Path("/Users/zhenyizhang/.cellcompass/algorithm_benchmarks/datasets/gse305370_citeseq_stage2_10k/data.h5ad")
TRAJECTORY_PATH = ROOT / "training_runs/20260526T174204321821Z__final__cite_paired_coord_fm__campaign_20260-trial_20260526-gse305370_ci/artifacts/evaluation_trajectory.npz"

FIG_DIR = ROOT / "figures/cite_pcfm_deep_story/biology_landscape_discovery"
OUT_DIR = ROOT / "downstream/cite_pcfm_deep_story/biology_landscape_discovery_protein_phenotype_river"

RNA_DIM = 50
PROTEIN_DIM = 30
RANDOM_STATE = 20260528
BIOLOGICAL_TIMES = [0.0, 1.0, 2.0, 3.0, 4.0]
MIDPOINT_GENERATED_TIMES = [0.5, 1.5, 2.5, 3.5]
ANCHOR_AND_MIDPOINT_GENERATED_TIMES = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]


def load_trajectory(path: Path) -> dict[str, object]:
    z = np.load(path)
    times = np.asarray(z["time_points"], dtype=float)
    points: dict[float, np.ndarray] = {}
    weights: dict[float, np.ndarray] = {}
    for i, time in enumerate(times):
        pkey = f"points_{i:06d}"
        wkey = f"weights_{i:06d}"
        if pkey in z.files:
            points[float(time)] = np.asarray(z[pkey], dtype=np.float32)
        if wkey in z.files:
            weights[float(time)] = np.asarray(z[wkey], dtype=np.float64)
    return {"time_points": times, "points": points, "weights": weights}


def nearest_time(available: np.ndarray, target: float) -> float:
    available = np.asarray(available, dtype=float)
    return float(available[np.argmin(np.abs(available - target))])


def normalized_weights(weights: np.ndarray | None, n: int) -> np.ndarray:
    if weights is None:
        return np.full(n, 1.0 / n, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if w.shape[0] != n or not np.isfinite(w).all() or float(w.sum()) <= 0:
        return np.full(n, 1.0 / n, dtype=np.float64)
    w = np.clip(w, 0.0, np.inf)
    total = float(w.sum())
    return w / total if total > 0 else np.full(n, 1.0 / n, dtype=np.float64)


def class_probability_matrix(readout: ProteinPhenotypeReadout, protein: np.ndarray) -> np.ndarray:
    return readout.predict_proba_ordered(protein)


def observed_fraction_table(labels: np.ndarray, times: np.ndarray) -> pd.DataFrame:
    rows = []
    for time in BIOLOGICAL_TIMES:
        mask = np.isclose(times, time)
        labels_t = labels[mask]
        denom = float(labels_t.shape[0])
        for label in PHENO_ORDER:
            frac = float(np.mean(labels_t == label)) if denom > 0 else np.nan
            rows.append(
                {
                    "panel": "observed",
                    "readout": "source_annotation_fraction",
                    "time": time,
                    "phenotype": label,
                    "fraction": frac,
                    "n_cells": int(denom),
                }
            )
    return pd.DataFrame(rows)


def generated_fraction_table(
    trajectory: dict[str, object],
    readout: ProteinPhenotypeReadout,
    display_times: list[float],
) -> pd.DataFrame:
    time_points = np.asarray(trajectory["time_points"], dtype=float)
    points_by_time: dict[float, np.ndarray] = trajectory["points"]  # type: ignore[assignment]
    weights_by_time: dict[float, np.ndarray] = trajectory["weights"]  # type: ignore[assignment]
    rows = []
    for target_time in display_times:
        t = nearest_time(time_points, target_time)
        state = np.asarray(points_by_time[t], dtype=np.float32)
        protein = state[:, RNA_DIM : RNA_DIM + PROTEIN_DIM]
        probs = class_probability_matrix(readout, protein)
        hard = np.asarray(PHENO_ORDER, dtype=object)[np.argmax(probs, axis=1)]
        w = normalized_weights(weights_by_time.get(t), protein.shape[0])
        for j, label in enumerate(PHENO_ORDER):
            soft_fraction = float(np.sum(w * probs[:, j]))
            hard_fraction = float(np.sum(w[hard == label]))
            rows.append(
                {
                    "panel": "generated",
                    "readout": "protein_pca_mlp_probability",
                    "time": target_time,
                    "trajectory_time": t,
                    "phenotype": label,
                    "fraction": soft_fraction,
                    "hard_fraction": hard_fraction,
                    "n_particles": int(protein.shape[0]),
                }
            )
    return pd.DataFrame(rows)


def wide_fraction(df: pd.DataFrame, display_times: list[float]) -> np.ndarray:
    wide = (
        df.pivot_table(index="time", columns="phenotype", values="fraction", aggfunc="mean")
        .reindex(index=display_times, columns=PHENO_ORDER)
        .fillna(0.0)
    )
    values = wide.to_numpy(dtype=np.float64).T
    col_sum = values.sum(axis=0, keepdims=True)
    np.divide(values, col_sum, out=values, where=col_sum > 0)
    return values


def time_labels(adata: ad.AnnData, times: np.ndarray) -> list[str]:
    labels = []
    obs_time = adata.obs["time"].astype(str).to_numpy() if "time" in adata.obs else None
    for time in BIOLOGICAL_TIMES:
        label = f"t{int(time)}"
        if obs_time is not None:
            vals = pd.Series(obs_time[np.isclose(times, time)]).dropna()
            if not vals.empty:
                label = str(vals.mode().iloc[0])
        labels.append(label)
    return labels


def plot_river(
    observed_df: pd.DataFrame,
    generated_df: pd.DataFrame,
    labels: list[str],
    classifier_metrics: dict[str, object],
    out_png: Path,
) -> None:
    x = np.asarray(BIOLOGICAL_TIMES, dtype=float)
    observed = wide_fraction(observed_df, BIOLOGICAL_TIMES)
    generated = wide_fraction(generated_df, BIOLOGICAL_TIMES)
    colors = [PHENO_COLORS[label] for label in PHENO_ORDER]

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.8), sharey=True, constrained_layout=True)
    panels = [
        (
            axes[0],
            observed,
            "Observed CITE cells",
            "source annotation fractions at measured culture times",
        ),
        (
            axes[1],
            generated,
            "CITE-PCFM generated proteins",
            "protein PCA MLP probability phenotype mass",
        ),
    ]
    for ax, matrix, title, subtitle in panels:
        ax.stackplot(x, matrix, colors=colors, alpha=0.88, linewidth=0.0)
        ax.plot(x, np.ones_like(x), color="#0F172A", lw=0.8, alpha=0.55)
        ax.set_title(title, loc="left", fontsize=12.5, weight="bold")
        ax.text(
            0.012,
            0.98,
            subtitle,
            transform=ax.transAxes,
            va="top",
            fontsize=8.6,
            color="#334155",
            bbox={"boxstyle": "round,pad=0.16", "facecolor": "white", "edgecolor": "none", "alpha": 0.78},
        )
        ax.set_xlim(float(x.min()), float(x.max()))
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_xlabel("culture time", fontsize=9)
        ax.grid(axis="y", color="#E2E8F0", lw=0.7, alpha=0.85)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#CBD5E1")
        ax.tick_params(axis="y", labelsize=8)
    axes[0].set_ylabel("phenotype fraction", fontsize=9)

    handles = [Patch(facecolor=PHENO_COLORS[label], edgecolor="none", label=label) for label in PHENO_ORDER]
    axes[1].legend(handles=handles, title="Protein phenotype", frameon=False, fontsize=8.3, title_fontsize=8.7, loc="center left", bbox_to_anchor=(1.01, 0.5))

    acc = classifier_metrics["accuracy"]
    bal_acc = classifier_metrics["balanced_accuracy"]
    fig.suptitle("Protein phenotype river from RNA-initialized CITE-PCFM rollout", x=0.01, ha="left", fontsize=15.5, weight="bold")
    fig.text(
        0.01,
        -0.02,
        f"Generated panel uses a fixed observed-protein-PCA MLP probability readout; holdout accuracy={acc:.3f}, balanced accuracy={bal_acc:.3f}.",
        fontsize=8.8,
        color="#475569",
    )
    fig.savefig(out_png, dpi=330, bbox_inches="tight", facecolor="white")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_midpoint_stackbar(
    observed_df: pd.DataFrame,
    midpoint_generated_df: pd.DataFrame,
    biological_labels: list[str],
    classifier_metrics: dict[str, object],
    out_png: Path,
) -> None:
    observed = wide_fraction(observed_df, BIOLOGICAL_TIMES)
    generated = wide_fraction(midpoint_generated_df, MIDPOINT_GENERATED_TIMES)
    colors = [PHENO_COLORS[label] for label in PHENO_ORDER]

    bar_specs: list[tuple[str, float, str, np.ndarray]] = []
    for i, time in enumerate(BIOLOGICAL_TIMES):
        bar_specs.append(("observed", time, biological_labels[i], observed[:, i]))
        if i < len(MIDPOINT_GENERATED_TIMES):
            midpoint = MIDPOINT_GENERATED_TIMES[i]
            bar_specs.append(("generated", midpoint, f"t={midpoint:.1f}", generated[:, i]))

    x = np.arange(len(bar_specs), dtype=float)
    fig, ax = plt.subplots(figsize=(12.6, 5.8), constrained_layout=True)
    bottom = np.zeros(len(bar_specs), dtype=float)
    for j, label in enumerate(PHENO_ORDER):
        heights = np.asarray([spec[3][j] for spec in bar_specs], dtype=float)
        ax.bar(
            x,
            heights,
            width=0.76,
            bottom=bottom,
            color=colors[j],
            edgecolor="white",
            linewidth=0.35,
            label=label,
        )
        bottom += heights

    for i, spec in enumerate(bar_specs):
        source, _, _, _ = spec
        if source == "observed":
            ax.bar(i, 1.0, width=0.82, bottom=0, facecolor="none", edgecolor="#0F172A", linewidth=1.1)
        else:
            ax.bar(i, 1.0, width=0.82, bottom=0, facecolor="none", edgecolor="#64748B", linewidth=0.9, linestyle="--")

    for boundary in [1.5, 3.5, 5.5, 7.5]:
        ax.axvline(boundary, color="#CBD5E1", lw=0.8, alpha=0.85, zorder=0)

    tick_labels = [spec[2] for spec in bar_specs]
    source_labels = ["Obs" if spec[0] == "observed" else "Gen" for spec in bar_specs]
    ax.set_xticks(x)
    ax.set_xticklabels([f"{label}\n{source}" for label, source in zip(tick_labels, source_labels)], fontsize=8.4)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(-0.65, len(bar_specs) - 0.35)
    ax.set_ylabel("protein phenotype fraction", fontsize=9)
    ax.set_xlabel("observed sampling days and CITE-PCFM generated midpoint rollout states", fontsize=9)
    ax.grid(axis="y", color="#E2E8F0", lw=0.7, alpha=0.85)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#CBD5E1")
    ax.tick_params(axis="y", labelsize=8)
    ax.set_title("Observed anchors with generated midpoint protein phenotypes", loc="left", fontsize=13.0, weight="bold")
    ax.text(
        0.012,
        0.985,
        "solid-border bars = observed CITE sampling days; dashed-border bars = generated rollout midpoints",
        transform=ax.transAxes,
        va="top",
        fontsize=8.7,
        color="#334155",
        bbox={"boxstyle": "round,pad=0.16", "facecolor": "white", "edgecolor": "none", "alpha": 0.80},
    )

    handles = [Patch(facecolor=PHENO_COLORS[label], edgecolor="none", label=label) for label in PHENO_ORDER]
    handles += [
        Patch(facecolor="white", edgecolor="#0F172A", linewidth=1.1, label="observed anchor"),
        Patch(facecolor="white", edgecolor="#64748B", linewidth=0.9, linestyle="--", label="generated midpoint"),
    ]
    ax.legend(handles=handles, title="Readout", frameon=False, fontsize=8.2, title_fontsize=8.7, loc="center left", bbox_to_anchor=(1.01, 0.5))

    acc = classifier_metrics["accuracy"]
    bal_acc = classifier_metrics["balanced_accuracy"]
    fig.suptitle("Protein phenotype stacked bars from RNA-initialized CITE-PCFM rollout", x=0.01, ha="left", fontsize=15.2, weight="bold")
    fig.text(
        0.01,
        -0.015,
        f"Midpoint bars are generated protein-PCA MLP probability masses, not interpolation of observed fractions; holdout accuracy={acc:.3f}, balanced accuracy={bal_acc:.3f}.",
        fontsize=8.7,
        color="#475569",
    )
    fig.savefig(out_png, dpi=330, bbox_inches="tight", facecolor="white")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_anchor_midpoint_stackplot(
    observed_df: pd.DataFrame,
    generated_df: pd.DataFrame,
    biological_labels: list[str],
    classifier_metrics: dict[str, object],
    out_png: Path,
) -> None:
    observed_x = np.asarray(BIOLOGICAL_TIMES, dtype=float)
    generated_x = np.asarray(ANCHOR_AND_MIDPOINT_GENERATED_TIMES, dtype=float)
    observed = wide_fraction(observed_df, BIOLOGICAL_TIMES)
    generated = wide_fraction(generated_df, ANCHOR_AND_MIDPOINT_GENERATED_TIMES)
    colors = [PHENO_COLORS[label] for label in PHENO_ORDER]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11.8, 7.2),
        sharex=True,
        sharey=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [0.82, 1.18]},
    )

    axes[0].stackplot(observed_x, observed, colors=colors, alpha=0.86, linewidth=0.0)
    axes[0].set_title("Observed protein phenotype anchors", loc="left", fontsize=12.4, weight="bold")
    axes[0].text(
        0.012,
        0.95,
        "real CITE cells are observed only at the five sampling days",
        transform=axes[0].transAxes,
        va="top",
        fontsize=8.7,
        color="#334155",
        bbox={"boxstyle": "round,pad=0.16", "facecolor": "white", "edgecolor": "none", "alpha": 0.78},
    )

    axes[1].stackplot(generated_x, generated, colors=colors, alpha=0.88, linewidth=0.0)
    axes[1].set_title("Generated protein phenotype trajectory with midpoint rollout states", loc="left", fontsize=12.4, weight="bold")
    axes[1].text(
        0.012,
        0.95,
        "generated curve uses model rollout states at anchors and midpoints, not observed-fraction interpolation",
        transform=axes[1].transAxes,
        va="top",
        fontsize=8.7,
        color="#334155",
        bbox={"boxstyle": "round,pad=0.16", "facecolor": "white", "edgecolor": "none", "alpha": 0.78},
    )

    for ax in axes:
        for t in BIOLOGICAL_TIMES:
            ax.axvline(t, color="#0F172A", lw=0.65, alpha=0.25, zorder=3)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlim(float(generated_x.min()), float(generated_x.max()))
        ax.grid(axis="y", color="#E2E8F0", lw=0.7, alpha=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#CBD5E1")
        ax.tick_params(axis="y", labelsize=8)
        ax.set_ylabel("phenotype fraction", fontsize=9)

    label_by_time = dict(zip(BIOLOGICAL_TIMES, biological_labels))
    tick_labels = [label_by_time[t] if t in label_by_time else f"t={t:.1f}" for t in ANCHOR_AND_MIDPOINT_GENERATED_TIMES]
    axes[1].set_xticks(generated_x)
    axes[1].set_xticklabels(tick_labels, fontsize=8.4)
    axes[1].set_xlabel("observed anchors plus CITE-PCFM generated midpoint times", fontsize=9)

    handles = [Patch(facecolor=PHENO_COLORS[label], edgecolor="none", label=label) for label in PHENO_ORDER]
    axes[1].legend(handles=handles, title="Protein phenotype", frameon=False, fontsize=8.2, title_fontsize=8.7, loc="center left", bbox_to_anchor=(1.01, 0.5))

    acc = classifier_metrics["accuracy"]
    bal_acc = classifier_metrics["balanced_accuracy"]
    fig.suptitle("Protein phenotype stackplot with generated midpoint rollout states", x=0.01, ha="left", fontsize=15.2, weight="bold")
    fig.text(
        0.01,
        -0.015,
        f"Generated fractions are protein-PCA MLP probability masses; holdout accuracy={acc:.3f}, balanced accuracy={bal_acc:.3f}.",
        fontsize=8.7,
        color="#475569",
    )
    fig.savefig(out_png, dpi=330, bbox_inches="tight", facecolor="white")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> dict[str, object]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(DATA_PATH)
    protein = np.asarray(adata.obsm["X_protein_pca"], dtype=np.float32)[:, :PROTEIN_DIM]
    labels = adata.obs["cell_type"].astype(str).to_numpy()
    times = np.asarray(adata.obs["time_point_processed"], dtype=float)

    readout, classifier_metrics, confusion_df = train_mlp_protein_phenotype_readout(
        protein, labels, PHENO_ORDER, RANDOM_STATE
    )
    trajectory = load_trajectory(TRAJECTORY_PATH)

    observed_df = observed_fraction_table(labels, times)
    generated_df = generated_fraction_table(trajectory, readout, BIOLOGICAL_TIMES)
    generated_midpoint_df = generated_fraction_table(trajectory, readout, MIDPOINT_GENERATED_TIMES)
    generated_anchor_midpoint_df = generated_fraction_table(trajectory, readout, ANCHOR_AND_MIDPOINT_GENERATED_TIMES)
    fraction_df = pd.concat([observed_df, generated_df], ignore_index=True, sort=False)
    fraction_path = OUT_DIR / "protein_phenotype_river_fractions.csv"
    fraction_df.to_csv(fraction_path, index=False)
    midpoint_fraction_path = OUT_DIR / "protein_phenotype_midpoint_generated_fractions.csv"
    generated_midpoint_df.to_csv(midpoint_fraction_path, index=False)
    anchor_midpoint_fraction_path = OUT_DIR / "protein_phenotype_anchor_midpoint_generated_fractions.csv"
    generated_anchor_midpoint_df.to_csv(anchor_midpoint_fraction_path, index=False)
    confusion_path = OUT_DIR / "protein_phenotype_readout_holdout_confusion.csv"
    confusion_df.to_csv(confusion_path)

    out_png = FIG_DIR / "fig03_protein_phenotype_river.png"
    labels_by_time = time_labels(adata, times)
    plot_river(observed_df, generated_df, labels_by_time, classifier_metrics, out_png)
    midpoint_bar_png = FIG_DIR / "fig03b_protein_phenotype_midpoint_stackbar.png"
    plot_midpoint_stackbar(observed_df, generated_midpoint_df, labels_by_time, classifier_metrics, midpoint_bar_png)
    anchor_midpoint_stack_png = FIG_DIR / "fig03c_protein_phenotype_anchor_midpoint_stackplot.png"
    plot_anchor_midpoint_stackplot(observed_df, generated_anchor_midpoint_df, labels_by_time, classifier_metrics, anchor_midpoint_stack_png)

    generated_terminal = (
        generated_df[np.isclose(generated_df["time"].astype(float), BIOLOGICAL_TIMES[-1])]
        .set_index("phenotype")["fraction"]
        .reindex(PHENO_ORDER)
        .fillna(0.0)
        .to_dict()
    )
    observed_terminal = (
        observed_df[np.isclose(observed_df["time"].astype(float), BIOLOGICAL_TIMES[-1])]
        .set_index("phenotype")["fraction"]
        .reindex(PHENO_ORDER)
        .fillna(0.0)
        .to_dict()
    )
    manifest = {
        "analysis": "protein_phenotype_river",
        "data_h5ad": str(DATA_PATH),
        "trajectory_path": str(TRAJECTORY_PATH),
        "figure": str(out_png),
        "midpoint_stackbar_figure": str(midpoint_bar_png),
        "anchor_midpoint_stackplot_figure": str(anchor_midpoint_stack_png),
        "tables": {
            "fractions": str(fraction_path),
            "midpoint_generated_fractions": str(midpoint_fraction_path),
            "anchor_midpoint_generated_fractions": str(anchor_midpoint_fraction_path),
            "holdout_confusion": str(confusion_path),
        },
        "semantic_contract": {
            "observed_panel": "source annotation fractions in observed CITE cells at five measured culture times",
            "generated_panel": "CITE-PCFM generated protein block classified by fixed observed-protein-PCA MLP probability readout",
            "not_claimed": "not a multimodal cell-state proportion figure",
        },
        "readout": {
            "feature_space": "adata.obsm['X_protein_pca'] for observed cells; trajectory state[:, 50:80] for generated cells",
            "classifier": "StandardScaler + MLPClassifier",
            "hidden_layer_sizes": [64, 32],
            "fraction": "weighted mean of MLP class probabilities across generated particles",
            "timepoints": BIOLOGICAL_TIMES,
            "midpoint_generated_timepoints": MIDPOINT_GENERATED_TIMES,
            "anchor_and_midpoint_generated_timepoints": ANCHOR_AND_MIDPOINT_GENERATED_TIMES,
        },
        "classifier_holdout_metrics": classifier_metrics,
        "terminal_observed_fraction": observed_terminal,
        "terminal_generated_soft_fraction": generated_terminal,
    }
    manifest_path = OUT_DIR / "protein_phenotype_river_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    main()
