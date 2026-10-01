#!/usr/bin/env python3
"""Commitment-time analysis for CITE-PCFM generated protein dynamics.

This analysis asks when generated protein fate starts to sharpen. It uses the
dense CITE-PCFM rollout, but uses only soft protein-basin probabilities:

- terminal-basin probability: probability assigned at time t to the protein
  basin that the same particle occupies at terminal time;
- protein-fate entropy: normalized entropy of the protein-basin probability
  vector at time t;
- RNA neighborhoods: KMeans clusters of the t0 RNA latent state, used only for
  stratifying initial transcriptional priming.

The result is a model-derived protein commitment-time landscape, not an
observed lineage-tracing assay.
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
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from multistageot_style import PHENO_COLORS, PHENO_ORDER
from protein_phenotype_readout import ProteinPhenotypeReadout, train_mlp_protein_phenotype_readout


ROOT = Path("/Users/zhenyizhang/Documents/GitHub/CytoBridge-agent-runtime-v2-20260309/multi-omics")
DATA_PATH = Path("/Users/zhenyizhang/.cellcompass/algorithm_benchmarks/datasets/gse305370_citeseq_stage2_10k/data.h5ad")
TRAJECTORY_PATH = ROOT / "training_runs/20260526T174204321821Z__final__cite_paired_coord_fm__campaign_20260-trial_20260526-gse305370_ci/artifacts/evaluation_trajectory.npz"

FIG_DIR = ROOT / "figures/cite_pcfm_deep_story/biology_landscape_discovery"
OUT_DIR = ROOT / "downstream/cite_pcfm_deep_story/biology_landscape_discovery_commitment_time"

RNA_DIM = 50
PROTEIN_DIM = 30
N_RNA_NEIGHBORHOODS = 6
RANDOM_STATE = 20260528
BIOLOGICAL_TIMES = [0.0, 1.0, 2.0, 3.0, 4.0]


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


def normalized_weights(weights: np.ndarray | None, n: int) -> np.ndarray:
    if weights is None:
        return np.full(n, 1.0 / n, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if w.shape[0] != n or not np.isfinite(w).all():
        return np.full(n, 1.0 / n, dtype=np.float64)
    w = np.clip(w, 0.0, np.inf)
    total = float(w.sum())
    return w / total if total > 0 else np.full(n, 1.0 / n, dtype=np.float64)


def class_probability_matrix(readout: ProteinPhenotypeReadout, protein: np.ndarray) -> np.ndarray:
    return readout.predict_proba_ordered(protein)


def entropy(probs: np.ndarray) -> np.ndarray:
    eps = 1e-8
    return -np.sum(probs * np.log(probs + eps), axis=1) / np.log(float(len(PHENO_ORDER)))


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    denom = float(w.sum())
    return float(np.sum(w * v) / denom) if denom > 0 else float(np.nan)


def commitment_time(time: np.ndarray, values: np.ndarray, fraction: float) -> float:
    """First time where values reach fraction of their terminal gain."""
    t = np.asarray(time, dtype=float)
    y = np.asarray(values, dtype=float)
    if y.shape[0] == 0 or not np.isfinite(y).any():
        return float("nan")
    start = float(y[0])
    terminal = float(y[-1])
    gain = terminal - start
    if abs(gain) < 1e-8:
        return float(t[0])
    threshold = start + fraction * gain
    if gain > 0:
        hit = np.where(y >= threshold)[0]
    else:
        hit = np.where(y <= threshold)[0]
    return float(t[int(hit[0])]) if hit.size else float(t[-1])


def fastest_gain_interval(time: np.ndarray, values: np.ndarray) -> tuple[float, float, float]:
    t = np.asarray(time, dtype=float)
    y = np.asarray(values, dtype=float)
    if t.shape[0] < 2:
        return float("nan"), float("nan"), float("nan")
    slope = np.diff(y) / np.diff(t)
    idx = int(np.nanargmax(slope))
    return float(t[idx]), float(t[idx + 1]), float(slope[idx])


def time_labels(adata: ad.AnnData, times: np.ndarray) -> dict[float, str]:
    labels = {}
    obs_time = adata.obs["time"].astype(str).to_numpy() if "time" in adata.obs else None
    for time in BIOLOGICAL_TIMES:
        label = f"t{int(time)}"
        if obs_time is not None:
            vals = pd.Series(obs_time[np.isclose(times, time)]).dropna()
            if not vals.empty:
                label = str(vals.mode().iloc[0])
        labels[time] = label
    return labels


def assign_rna_neighborhoods(t0_rna: np.ndarray, terminal_probs: np.ndarray) -> tuple[np.ndarray, pd.DataFrame]:
    x = StandardScaler().fit_transform(t0_rna)
    raw = KMeans(n_clusters=N_RNA_NEIGHBORHOODS, n_init=30, random_state=RANDOM_STATE).fit_predict(x)
    rows = []
    terminal_hard = np.asarray(PHENO_ORDER, dtype=object)[np.argmax(terminal_probs, axis=1)]
    for cluster in sorted(np.unique(raw)):
        mask = raw == cluster
        masses = {label: float(np.mean(terminal_hard[mask] == label)) for label in PHENO_ORDER}
        dominant = max(masses, key=masses.get)
        rows.append(
            {
                "raw_cluster": int(cluster),
                "dominant_terminal_phenotype": dominant,
                "dominant_terminal_fraction": masses[dominant],
                "sort_key": PHENO_ORDER.index(dominant),
            }
        )
    order_df = pd.DataFrame(rows).sort_values(["sort_key", "dominant_terminal_fraction", "raw_cluster"], ascending=[True, False, True])
    rename = {int(row.raw_cluster): f"N{i}" for i, row in enumerate(order_df.itertuples(index=False))}
    neighborhoods = np.asarray([rename[int(v)] for v in raw], dtype=object)
    mapping = order_df.copy()
    mapping["neighborhood"] = mapping["raw_cluster"].map(rename)
    return neighborhoods, mapping.drop(columns=["sort_key"])


def compute_probability_cube(
    trajectory: dict[str, object],
    readout: ProteinPhenotypeReadout,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    time_points = np.asarray(trajectory["time_points"], dtype=float)
    points_by_time: dict[float, np.ndarray] = trajectory["points"]  # type: ignore[assignment]
    weights_by_time: dict[float, np.ndarray] = trajectory["weights"]  # type: ignore[assignment]
    first_state = points_by_time[float(time_points[0])]
    n_particles = first_state.shape[0]
    base_weights = normalized_weights(weights_by_time.get(float(time_points[0])), n_particles)
    probs_by_time: list[np.ndarray] = []
    for time in time_points:
        state = np.asarray(points_by_time[float(time)], dtype=np.float32)
        if state.shape[0] != n_particles:
            raise ValueError(f"particle count changed at time {time}: {state.shape[0]} != {n_particles}")
        protein = state[:, RNA_DIM : RNA_DIM + PROTEIN_DIM]
        probs_by_time.append(class_probability_matrix(readout, protein))
    return time_points, probs_by_time, base_weights


def build_tables(
    time_points: np.ndarray,
    probs_by_time: list[np.ndarray],
    weights: np.ndarray,
    neighborhoods: np.ndarray,
    initial_labels: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    terminal_probs = probs_by_time[-1]
    terminal_idx = np.argmax(terminal_probs, axis=1)
    terminal_labels = np.asarray(PHENO_ORDER, dtype=object)[terminal_idx]
    label_to_col = {label: i for i, label in enumerate(PHENO_ORDER)}

    particle_rows = []
    global_rows = []
    terminal_rows = []
    neighborhood_rows = []
    for time, probs in zip(time_points, probs_by_time):
        ent = entropy(probs)
        top = probs.max(axis=1)
        terminal_prob = probs[np.arange(probs.shape[0]), terminal_idx]
        hard = np.asarray(PHENO_ORDER, dtype=object)[np.argmax(probs, axis=1)]
        global_row = {
            "time": float(time),
            "mean_entropy": weighted_mean(ent, weights),
            "mean_top_probability": weighted_mean(top, weights),
            "mean_terminal_basin_probability": weighted_mean(terminal_prob, weights),
            "mean_HSC_probability": weighted_mean(probs[:, label_to_col["HSC"]], weights),
        }
        for label in PHENO_ORDER:
            global_row[f"soft_mass_{label}"] = weighted_mean(probs[:, label_to_col[label]], weights)
        global_rows.append(global_row)

        for label in PHENO_ORDER:
            mask = terminal_labels == label
            if not np.any(mask):
                continue
            w = weights[mask]
            terminal_rows.append(
                {
                    "time": float(time),
                    "terminal_phenotype": label,
                    "n_particles": int(mask.sum()),
                    "weight_fraction": float(weights[mask].sum()),
                    "mean_entropy": weighted_mean(ent[mask], w),
                    "mean_terminal_basin_probability": weighted_mean(terminal_prob[mask], w),
                    "mean_top_probability": weighted_mean(top[mask], w),
                }
            )

        for neighborhood in sorted(np.unique(neighborhoods)):
            mask = neighborhoods == neighborhood
            w = weights[mask]
            counts = pd.Series(terminal_labels[mask]).value_counts().reindex(PHENO_ORDER).fillna(0.0)
            dominant_terminal = str(counts.idxmax())
            neighborhood_rows.append(
                {
                    "time": float(time),
                    "neighborhood": str(neighborhood),
                    "n_particles": int(mask.sum()),
                    "weight_fraction": float(weights[mask].sum()),
                    "dominant_terminal_phenotype": dominant_terminal,
                    "dominant_terminal_fraction": float(counts.max() / counts.sum()) if float(counts.sum()) > 0 else 0.0,
                    "mean_entropy": weighted_mean(ent[mask], w),
                    "mean_terminal_basin_probability": weighted_mean(terminal_prob[mask], w),
                    "mean_top_probability": weighted_mean(top[mask], w),
                    "mean_HSC_probability": weighted_mean(probs[mask, label_to_col["HSC"]], w),
                }
            )

        # Particle-level table keeps only compact diagnostics, not all class probs.
        if np.isclose(time % 0.5, 0.0, atol=1e-8) or np.isclose(time, time_points[-1]):
            for particle_id in range(probs.shape[0]):
                particle_rows.append(
                    {
                        "particle_id": particle_id,
                        "time": float(time),
                        "initial_source_cell_type": str(initial_labels[particle_id]),
                        "rna_neighborhood": str(neighborhoods[particle_id]),
                        "current_top_phenotype": str(hard[particle_id]),
                        "terminal_phenotype": str(terminal_labels[particle_id]),
                        "entropy": float(ent[particle_id]),
                        "top_probability": float(top[particle_id]),
                        "terminal_basin_probability": float(terminal_prob[particle_id]),
                        "base_weight": float(weights[particle_id]),
                    }
                )

    global_df = pd.DataFrame(global_rows)
    terminal_df = pd.DataFrame(terminal_rows)
    neighborhood_df = pd.DataFrame(neighborhood_rows)
    particle_df = pd.DataFrame(particle_rows)
    return global_df, terminal_df, neighborhood_df, particle_df


def summarize_commitment(global_df: pd.DataFrame, terminal_df: pd.DataFrame, neighborhood_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    time = global_df["time"].to_numpy(dtype=float)
    global_prob = global_df["mean_terminal_basin_probability"].to_numpy(dtype=float)
    t0, t1, slope = fastest_gain_interval(time, global_prob)
    global_summary = pd.DataFrame(
        [
            {
                "group": "all_particles",
                "commitment_time_50pct_terminal_prob_gain": commitment_time(time, global_prob, 0.50),
                "commitment_time_90pct_terminal_prob_gain": commitment_time(time, global_prob, 0.90),
                "fastest_gain_interval_start": t0,
                "fastest_gain_interval_end": t1,
                "fastest_gain_slope": slope,
                "initial_terminal_basin_probability": float(global_prob[0]),
                "final_terminal_basin_probability": float(global_prob[-1]),
                "initial_entropy": float(global_df["mean_entropy"].iloc[0]),
                "final_entropy": float(global_df["mean_entropy"].iloc[-1]),
            }
        ]
    )

    rows = []
    for name, df, id_col in [
        ("terminal", terminal_df, "terminal_phenotype"),
        ("neighborhood", neighborhood_df, "neighborhood"),
    ]:
        for group, sub in df.groupby(id_col, sort=False):
            sub = sub.sort_values("time")
            t = sub["time"].to_numpy(dtype=float)
            p = sub["mean_terminal_basin_probability"].to_numpy(dtype=float)
            i0, i1, s = fastest_gain_interval(t, p)
            row = {
                "summary_type": name,
                "group": str(group),
                "n_particles": int(sub["n_particles"].iloc[0]),
                "weight_fraction": float(sub["weight_fraction"].iloc[0]),
                "commitment_time_50pct_terminal_prob_gain": commitment_time(t, p, 0.50),
                "commitment_time_90pct_terminal_prob_gain": commitment_time(t, p, 0.90),
                "fastest_gain_interval_start": i0,
                "fastest_gain_interval_end": i1,
                "fastest_gain_slope": s,
                "initial_terminal_basin_probability": float(p[0]),
                "final_terminal_basin_probability": float(p[-1]),
                "initial_entropy": float(sub["mean_entropy"].iloc[0]),
                "final_entropy": float(sub["mean_entropy"].iloc[-1]),
            }
            if name == "neighborhood":
                row["dominant_terminal_phenotype"] = str(sub["dominant_terminal_phenotype"].iloc[-1])
                row["dominant_terminal_fraction"] = float(sub["dominant_terminal_fraction"].iloc[-1])
            rows.append(row)
    group_summary = pd.DataFrame(rows)
    return global_summary, group_summary


def draw_time_anchors(ax: plt.Axes, labels: dict[float, str]) -> None:
    for time in BIOLOGICAL_TIMES:
        ax.axvline(time, color="#CBD5E1", lw=0.8, alpha=0.8, zorder=0)
    ax.set_xticks(BIOLOGICAL_TIMES)
    ax.set_xticklabels([labels[t] for t in BIOLOGICAL_TIMES], fontsize=8.4)


def plot_commitment(
    global_df: pd.DataFrame,
    terminal_df: pd.DataFrame,
    neighborhood_df: pd.DataFrame,
    group_summary: pd.DataFrame,
    labels: dict[float, str],
    classifier_metrics: dict[str, object],
    out_png: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14.8, 9.0), constrained_layout=True)
    ax0, ax1, ax2, ax3 = axes.ravel()

    time = global_df["time"].to_numpy(dtype=float)
    ax0.plot(time, global_df["mean_terminal_basin_probability"], color="#E11D48", lw=2.4, label="future terminal-basin probability")
    ax0.plot(time, global_df["mean_top_probability"], color="#7C3AED", lw=2.0, label="top protein-basin probability")
    fast_t0, fast_t1, _ = fastest_gain_interval(time, global_df["mean_terminal_basin_probability"].to_numpy(dtype=float))
    ax0.plot(time, global_df["mean_entropy"], color="#2563EB", lw=2.0, label="current protein-basin entropy")
    draw_time_anchors(ax0, labels)
    ax0.set_ylim(0.0, 1.0)
    ax0.set_title("Global protein commitment timing", loc="left", fontsize=12.8, weight="bold")
    ax0.set_ylabel("weighted mean score", fontsize=9)
    ax0.legend(frameon=False, fontsize=8.2, loc="center left", bbox_to_anchor=(1.01, 0.5))

    for label in PHENO_ORDER:
        sub = terminal_df[terminal_df["terminal_phenotype"] == label].sort_values("time")
        if sub.empty:
            continue
        ax1.plot(
            sub["time"],
            sub["mean_terminal_basin_probability"],
            color=PHENO_COLORS[label],
            lw=2.0,
            label=f"{label} ({sub['weight_fraction'].iloc[0] * 100:.0f}%)",
        )
    draw_time_anchors(ax1, labels)
    ax1.set_ylim(0.0, 1.0)
    ax1.set_title("Commitment by terminal protein basin", loc="left", fontsize=12.8, weight="bold")
    ax1.set_ylabel("mean probability of own terminal basin", fontsize=9)
    ax1.legend(frameon=False, fontsize=8.0, loc="center left", bbox_to_anchor=(1.01, 0.5))

    neighborhood_ids = sorted(neighborhood_df["neighborhood"].unique(), key=lambda x: int(str(x).replace("N", "")))
    for nid in neighborhood_ids:
        sub = neighborhood_df[neighborhood_df["neighborhood"] == nid].sort_values("time")
        dominant = str(sub["dominant_terminal_phenotype"].iloc[-1])
        ax2.plot(
            sub["time"],
            sub["mean_terminal_basin_probability"],
            color=PHENO_COLORS[dominant],
            lw=1.8,
            alpha=0.82,
            marker="o",
            markersize=2.8,
            label=f"{nid}->{dominant}",
        )
    draw_time_anchors(ax2, labels)
    ax2.set_ylim(0.0, 1.0)
    ax2.set_title("Initial RNA-neighborhood commitment curves", loc="left", fontsize=12.8, weight="bold")
    ax2.set_ylabel("mean probability of own terminal basin", fontsize=9)
    ax2.legend(frameon=False, fontsize=7.4, ncol=2, loc="lower right")

    summary = group_summary[group_summary["summary_type"] == "neighborhood"].copy()
    summary["neighborhood_order"] = summary["group"].str.replace("N", "", regex=False).astype(int)
    summary = summary.sort_values("neighborhood_order")
    y = np.arange(summary.shape[0])
    colors = [PHENO_COLORS[str(v)] for v in summary["dominant_terminal_phenotype"]]
    ax3.hlines(y, summary["commitment_time_50pct_terminal_prob_gain"], summary["commitment_time_90pct_terminal_prob_gain"], color=colors, lw=4, alpha=0.48)
    ax3.scatter(summary["commitment_time_50pct_terminal_prob_gain"], y, s=42, color="white", edgecolor=colors, linewidth=1.6, zorder=3, label="50% terminal-prob gain")
    ax3.scatter(summary["commitment_time_90pct_terminal_prob_gain"], y, s=54, color=colors, edgecolor="white", linewidth=0.8, zorder=4, label="90% terminal-prob gain")
    draw_time_anchors(ax3, labels)
    ax3.set_yticks(y)
    ax3.set_yticklabels(
        [
            f"{row.group} -> {row.dominant_terminal_phenotype} ({row.dominant_terminal_fraction * 100:.0f}%)"
            for row in summary.itertuples(index=False)
        ],
        fontsize=8.0,
    )
    ax3.set_xlim(time.min(), time.max())
    ax3.set_title("RNA-neighborhood commitment windows", loc="left", fontsize=12.8, weight="bold")
    ax3.set_xlabel("model time aligned to observed sampling anchors", fontsize=9)
    ax3.legend(frameon=False, fontsize=7.8, loc="lower right")

    for ax in [ax0, ax1, ax2, ax3]:
        ax.axvspan(fast_t0, fast_t1, color="#E11D48", alpha=0.075, zorder=0)
        ax.set_xlim(time.min(), time.max())
        ax.grid(axis="y", color="#E2E8F0", lw=0.7, alpha=0.85)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#CBD5E1")
        ax.tick_params(axis="both", labelsize=8)
        if ax is not ax3:
            ax.set_xlabel("model time", fontsize=9)
    ax0.text(
        fast_t0,
        0.965,
        "fastest global\ncommitment gain",
        fontsize=7.8,
        color="#BE123C",
        ha="left",
        va="top",
        bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.75},
    )

    acc = classifier_metrics["accuracy"]
    bal_acc = classifier_metrics["balanced_accuracy"]
    fig.suptitle("Protein commitment time from RNA-initialized CITE-PCFM dynamics", x=0.01, ha="left", fontsize=15.8, weight="bold")
    fig.text(
        0.01,
        -0.012,
        f"Commitment is quantified from soft protein-PCA MLP basin probabilities. Readout holdout accuracy={acc:.3f}, balanced accuracy={bal_acc:.3f}.",
        fontsize=8.8,
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
    obs_times = np.asarray(adata.obs["time_point_processed"], dtype=float)
    t0_idx = np.where(np.isclose(obs_times, BIOLOGICAL_TIMES[0]))[0]

    readout, classifier_metrics, confusion_df = train_mlp_protein_phenotype_readout(
        protein, labels, PHENO_ORDER, RANDOM_STATE
    )

    trajectory = load_trajectory(TRAJECTORY_PATH)
    time_points, probs_by_time, weights = compute_probability_cube(trajectory, readout)
    points_by_time: dict[float, np.ndarray] = trajectory["points"]  # type: ignore[assignment]
    t0_state = np.asarray(points_by_time[float(time_points[0])], dtype=np.float32)
    t0_rna = t0_state[:, :RNA_DIM]
    initial_labels = labels[t0_idx] if t0_idx.shape[0] == t0_state.shape[0] else np.repeat("unknown", t0_state.shape[0])
    terminal_probs = probs_by_time[-1]
    neighborhoods, neighborhood_mapping = assign_rna_neighborhoods(t0_rna, terminal_probs)

    global_df, terminal_df, neighborhood_df, particle_df = build_tables(
        time_points,
        probs_by_time,
        weights,
        neighborhoods,
        initial_labels,
    )
    global_summary, group_summary = summarize_commitment(global_df, terminal_df, neighborhood_df)

    global_path = OUT_DIR / "commitment_time_global_metrics.csv"
    terminal_path = OUT_DIR / "commitment_time_by_terminal_basin.csv"
    neighborhood_path = OUT_DIR / "commitment_time_by_rna_neighborhood.csv"
    particle_path = OUT_DIR / "commitment_time_particle_diagnostics_sparse.csv"
    global_summary_path = OUT_DIR / "commitment_time_global_summary.csv"
    group_summary_path = OUT_DIR / "commitment_time_group_summary.csv"
    mapping_path = OUT_DIR / "commitment_time_rna_neighborhood_mapping.csv"
    confusion_path = OUT_DIR / "commitment_time_readout_holdout_confusion.csv"
    global_df.to_csv(global_path, index=False)
    terminal_df.to_csv(terminal_path, index=False)
    neighborhood_df.to_csv(neighborhood_path, index=False)
    particle_df.to_csv(particle_path, index=False)
    global_summary.to_csv(global_summary_path, index=False)
    group_summary.to_csv(group_summary_path, index=False)
    neighborhood_mapping.to_csv(mapping_path, index=False)
    confusion_df.to_csv(confusion_path)

    out_png = FIG_DIR / "fig05_commitment_time_analysis.png"
    plot_commitment(global_df, terminal_df, neighborhood_df, group_summary, time_labels(adata, obs_times), classifier_metrics, out_png)

    source_order_check = {}
    if t0_idx.shape[0] == t0_state.shape[0] and "X_rna_svd" in adata.obsm:
        obs_t0_rna = np.asarray(adata.obsm["X_rna_svd"], dtype=np.float32)[t0_idx, :RNA_DIM]
        diff = np.linalg.norm(obs_t0_rna - t0_rna, axis=1)
        source_order_check = {
            "t0_observed_count": int(t0_idx.shape[0]),
            "t0_particle_count": int(t0_state.shape[0]),
            "mean_l2_diff_observed_t0_rna_vs_particle_t0_rna": float(np.mean(diff)),
            "max_l2_diff_observed_t0_rna_vs_particle_t0_rna": float(np.max(diff)),
        }
    else:
        source_order_check = {
            "t0_observed_count": int(t0_idx.shape[0]),
            "t0_particle_count": int(t0_state.shape[0]),
            "warning": "could not verify observed t0 order against trajectory t0 RNA",
        }

    manifest = {
        "analysis": "commitment_time_analysis",
        "data_h5ad": str(DATA_PATH),
        "trajectory_path": str(TRAJECTORY_PATH),
        "figure": str(out_png),
        "tables": {
            "global_metrics": str(global_path),
            "terminal_basin_metrics": str(terminal_path),
            "rna_neighborhood_metrics": str(neighborhood_path),
            "particle_diagnostics_sparse": str(particle_path),
            "global_summary": str(global_summary_path),
            "group_summary": str(group_summary_path),
            "rna_neighborhood_mapping": str(mapping_path),
            "holdout_confusion": str(confusion_path),
        },
        "semantic_contract": {
            "commitment_metric": "soft probability assigned to each particle's terminal generated protein basin over time",
            "entropy_metric": "normalized entropy of the current protein-basin probability vector over time",
            "rna_neighborhoods": f"KMeans(n={N_RNA_NEIGHBORHOODS}) on t0 generated RNA latent state",
            "not_claimed": "not observed lineage tracing and not multimodal cell-state commitment",
        },
        "readout": {
            "feature_space": "trajectory state[:, 50:80] classified against observed adata.obsm['X_protein_pca']",
            "classifier": "StandardScaler + MLPClassifier",
            "hidden_layer_sizes": [64, 32],
            "timepoints": [float(t) for t in time_points],
            "classes": PHENO_ORDER,
        },
        "classifier_holdout_metrics": classifier_metrics,
        "global_summary": global_summary.iloc[0].to_dict(),
        "source_order_check": source_order_check,
    }
    manifest_path = OUT_DIR / "commitment_time_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    main()
