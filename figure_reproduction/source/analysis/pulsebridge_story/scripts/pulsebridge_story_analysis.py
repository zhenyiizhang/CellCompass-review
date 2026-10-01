#!/usr/bin/env python3
"""Rebuild and extend the PulseBridge HCT116 Well-TEMP-seq story figures.

The script is intentionally data-driven from the downloaded remote bundle:
  downloads/pulsebridge_welltemp_hct116_20260627

Outputs are written under:
  analysis/pulsebridge_story/outputs
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cellcompass_matplotlib_cache")
# scVelo imports scanpy, whose numba cache can fail in the bundled desktop env.
# Disabling JIT is acceptable here because the stream plot uses a downsampled
# diagnostic subset, not a performance-critical training path.
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/cellcompass_numba_cache")

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


REPO = Path(__file__).resolve().parents[3]
BUNDLE = REPO / "downloads" / "pulsebridge_welltemp_hct116_20260627"
PAPER_PKG = (
    REPO
    / "downloads"
    / "cellcompass_cancer_papers_20260627"
    / "cellcompass_cancer_papers_20260627_1428"
    / "completed_packages"
    / "pulse_control_identifiability_bridge"
)
OUT = REPO / "analysis" / "pulsebridge_story" / "outputs"
FIG = OUT / "figures"
TAB = OUT / "tables"

ADATA_PATH = (
    BUNDLE
    / "home"
    / "ubuntu"
    / ".cellcompass"
    / "algorithm_benchmarks"
    / "datasets"
    / "pcib_welltemp_hct116_raw_pool_20k"
    / "data.h5ad"
)
PREPARED_H5AD_PATH = (
    BUNDLE
    / "data"
    / "cytobridge"
    / "runs"
    / "night_wave_20260621_0140"
    / "existing_deep_raw_a"
    / "paper"
    / "preprocessing"
    / "welltemp_hct116_raw_pool_pulse_identifiability_prepared.h5ad"
)
TRAJ_PATH = (
    BUNDLE
    / "data"
    / "cytobridge"
    / "runs"
    / "night_wave_20260621_0140"
    / "existing_deep_raw_a"
    / "training_runs"
    / "20260620T234044691964Z__pilot__pulse_control_identifiability_bridge__campaign_20260-trial_20260620-pcib_welltem"
    / "artifacts"
    / "evaluation_trajectory.npz"
)
MODEL_STATE_PATH = TRAJ_PATH.parent / "model_state.pt"
MODEL_ARTIFACT_PATH = TRAJ_PATH.parent / "model_artifact.json"
METRICS_PATH = TRAJ_PATH.parent / "metrics.json"
DOWNSTREAM = (
    BUNDLE
    / "data"
    / "cytobridge"
    / "runs"
    / "night_wave_20260621_0140"
    / "existing_deep_raw_a"
    / "paper"
    / "downstream"
    / "pcib_locked_biology"
)
DOWNSTREAM_TABLES = DOWNSTREAM / "tables"
BASELINE_TABLE = PAPER_PKG / "tables" / "baseline_metrics.csv"


def ensure_inputs() -> None:
    required = [
        ADATA_PATH if ADATA_PATH.exists() else PREPARED_H5AD_PATH,
        TRAJ_PATH,
        MODEL_STATE_PATH,
        METRICS_PATH,
        DOWNSTREAM_TABLES / "heldout_pulse_alignment_by_slice.csv",
        DOWNSTREAM_TABLES / "trajectory_time_readout_summary.csv",
        DOWNSTREAM_TABLES / "pcib_deconfounded_velocity_gene_projection_top80.csv",
        BASELINE_TABLE,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def dense_slice(x):
    return x.toarray() if sparse.issparse(x) else np.asarray(x)


def sample_balanced_by_time(times: np.ndarray, max_cells: int, random_state: int = 0) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    n = len(times)
    if n <= max_cells:
        return np.arange(n, dtype=int)
    finite = np.isfinite(times)
    unique_times = np.unique(times[finite])
    if unique_times.size == 0:
        return np.sort(rng.choice(np.arange(n), size=max_cells, replace=False))
    per_time = max(64, int(max_cells // unique_times.size))
    picked: list[int] = []
    for t in unique_times:
        idx = np.where(np.isclose(times, t))[0]
        if idx.size == 0:
            continue
        picked.extend(rng.choice(idx, size=min(per_time, idx.size), replace=False).tolist())
    if len(picked) < max_cells:
        remaining = np.setdiff1d(np.arange(n), np.asarray(picked, dtype=int), assume_unique=False)
        extra = min(max_cells - len(picked), remaining.size)
        if extra > 0:
            picked.extend(rng.choice(remaining, size=extra, replace=False).tolist())
    return np.sort(np.asarray(picked[:max_cells], dtype=int))


def load_trajectory_points():
    z = np.load(TRAJ_PATH)
    time_points = z["time_points"]
    observed_indices = z["observed_time_indices"]
    observed_time_points = z["observed_time_points"]
    points = {float(t): z[f"points_{int(i):06d}"] for t, i in zip(observed_time_points, observed_indices)}
    return z, time_points, points


def load_all_trajectory_points():
    z = np.load(TRAJ_PATH)
    time_points = z["time_points"]
    points = [z[f"points_{i:06d}"] for i in range(len(time_points))]
    return time_points, points


def build_response_axis(adata) -> dict:
    """Build a simple drug-response axis from latent PCs associated with time.

    This is a descriptive axis, not a new trained model. It gives a common
    coordinate for endpoint-vs-trajectory and state-level summaries.
    """
    x_latent = np.asarray(adata.obsm["X_latent"])
    times = adata.obs["time_point_processed"].astype(float).to_numpy()
    pca = PCA(n_components=5, random_state=7)
    scores = pca.fit_transform(x_latent)
    cors = []
    for j in range(scores.shape[1]):
        c = np.corrcoef(scores[:, j], times)[0, 1]
        cors.append(0.0 if np.isnan(c) else c)
    axis_idx = int(np.argmax(np.abs(cors)))
    sign = -1.0 if cors[axis_idx] < 0 else 1.0
    axis_values = sign * scores[:, axis_idx]
    axis_vector = sign * pca.components_[axis_idx]
    edges = np.quantile(axis_values, np.linspace(0, 1, 7))
    edges[0] -= 1e-6
    edges[-1] += 1e-6
    labels = np.clip(np.digitize(axis_values, edges[1:-1], right=False), 0, 5)
    return {
        "pca": pca,
        "axis_idx": axis_idx,
        "axis_sign": sign,
        "axis_vector": axis_vector,
        "axis_values": axis_values,
        "axis_time_correlation": abs(cors[axis_idx]),
        "edges": edges,
        "labels": labels,
        "state_names": [f"state_{i+1}" for i in range(6)],
    }


def plot_existing_metrics() -> pd.DataFrame:
    baseline = pd.read_csv(BASELINE_TABLE)
    metric_rows = []
    for metric, label in [("w1_mean", "Mean W1"), ("pcie", "PCIE")]:
        df = baseline.dropna(subset=[metric]).copy()
        df = df.sort_values(metric)
        fig_h = max(3.5, 0.35 * len(df) + 1.0)
        plt.figure(figsize=(7.2, fig_h))
        colors = ["#146C94" if a == "PulseBridge" else "#9AA6B2" for a in df["algorithm"]]
        plt.barh(df["algorithm"], df[metric], color=colors)
        plt.xlabel(label + " (lower is better)")
        plt.title(f"Existing result rebuilt: {label}")
        for y, value in enumerate(df[metric]):
            plt.text(value, y, f" {value:.3f}", va="center", fontsize=8)
        savefig(FIG / f"existing_metric_{metric}.png")
        for _, row in df.iterrows():
            metric_rows.append({"metric": metric, "algorithm": row["algorithm"], "value": row[metric]})
    pd.DataFrame(metric_rows).to_csv(TAB / "existing_metric_summary_long.csv", index=False)
    return baseline


def plot_existing_temporal_clouds(adata, traj_points: dict[float, np.ndarray]) -> None:
    rng = np.random.default_rng(7)
    x_latent = np.asarray(adata.obsm["X_latent"])
    times = adata.obs["time_point_processed"].astype(float).to_numpy()
    pca = PCA(n_components=2, random_state=7)
    obs_2d = pca.fit_transform(x_latent)

    obs_rows = []
    gen_rows = []
    for t in sorted(np.unique(times)):
        idx = np.where(times == t)[0]
        keep = rng.choice(idx, size=min(900, len(idx)), replace=False)
        for xy in obs_2d[keep]:
            obs_rows.append({"time": t, "x": xy[0], "y": xy[1], "source": "observed"})
    for t, pts in traj_points.items():
        pts2 = pca.transform(pts)
        keep = rng.choice(np.arange(pts2.shape[0]), size=min(900, pts2.shape[0]), replace=False)
        for xy in pts2[keep]:
            gen_rows.append({"time": t, "x": xy[0], "y": xy[1], "source": "generated"})

    obs_df = pd.DataFrame(obs_rows)
    gen_df = pd.DataFrame(gen_rows)
    combined = pd.concat([obs_df, gen_df], ignore_index=True)
    combined.to_csv(TAB / "existing_temporal_cloud_projection_points.csv", index=False)

    palette = {0.0: "#31572C", 1.0: "#4F86C6", 2.0: "#F28E2B", 3.0: "#B23A48"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharex=True, sharey=True)
    for ax, df, title, marker, alpha in [
        (axes[0], obs_df, "Observed HCT116 states", "o", 0.28),
        (axes[1], gen_df, "PulseBridge generated slices", "x", 0.55),
    ]:
        for t, sub in df.groupby("time"):
            ax.scatter(sub["x"], sub["y"], s=7, alpha=alpha, marker=marker, color=palette.get(t), label=f"day {t:g}")
        ax.set_title(title)
        ax.set_xlabel("latent PC1")
        ax.set_ylabel("latent PC2")
        ax.legend(markerscale=2, fontsize=8, frameon=False)
    savefig(FIG / "existing_temporal_clouds_rebuilt.png")


def plot_existing_time_readout() -> pd.DataFrame:
    readout = pd.read_csv(DOWNSTREAM_TABLES / "trajectory_time_readout_summary.csv")
    prob_cols = [c for c in readout.columns if c.startswith("prob_day_")]
    prob_labels = [c.replace("prob_day_", "day ") for c in prob_cols]
    plt.figure(figsize=(8, 4.6))
    plt.stackplot(readout["time"], [readout[c] for c in prob_cols], labels=prob_labels, alpha=0.9)
    plt.xlabel("generated trajectory time (days)")
    plt.ylabel("time-readout probability")
    plt.title("Existing result rebuilt: generated trajectory progresses through treatment days")
    plt.legend(ncol=4, frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    savefig(FIG / "existing_trajectory_time_readout_rebuilt.png")
    return readout


def plot_existing_pulse_alignment() -> pd.DataFrame:
    align = pd.read_csv(DOWNSTREAM_TABLES / "heldout_pulse_alignment_by_slice.csv")
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(align["time"], align["mean_deconfounded_cosine"], color="#146C94", lw=2, label="mean cosine")
    axes[0].plot(align["time"], align["frac_positive_cosine"], color="#789461", lw=2, label="fraction positive")
    axes[0].axhline(0, color="#555555", lw=0.8)
    axes[0].set_ylabel("pulse alignment")
    axes[0].set_title("Existing result rebuilt: heldout pulse alignment")
    axes[0].legend(frameon=False)
    axes[1].plot(align["time"], align["mean_nuisance_leakage_fraction"], color="#B23A48", lw=2)
    axes[1].set_ylabel("nuisance leakage fraction")
    axes[1].set_xlabel("generated trajectory time (days)")
    savefig(FIG / "existing_pulse_alignment_and_leakage_rebuilt.png")
    return align


def plot_existing_gene_projection() -> pd.DataFrame:
    genes = pd.read_csv(DOWNSTREAM_TABLES / "pcib_deconfounded_velocity_gene_projection_top80.csv")
    top = genes.head(25).copy()
    top = top.sort_values("signed_score")
    plt.figure(figsize=(7.5, 7.0))
    colors = np.where(top["signed_score"] >= 0, "#146C94", "#B23A48")
    plt.barh(top["gene"], top["signed_score"], color=colors)
    plt.axvline(0, color="#555555", lw=0.8)
    plt.xlabel("PCA-loading projected deconfounded velocity score")
    plt.title("Existing result rebuilt: top gene readouts")
    savefig(FIG / "existing_gene_projection_top25_rebuilt.png")
    return genes


def summarize_response_phases(readout: pd.DataFrame, align: pd.DataFrame) -> pd.DataFrame:
    phase_defs = [
        ("initial_0_to_0.5", 0.0, 0.5),
        ("transition_0.5_to_1", 0.5, 1.0),
        ("middle_1_to_2", 1.0, 2.0),
        ("late_2_to_3", 2.0, 3.0),
    ]
    merged = pd.merge(readout, align, on="time", how="left", suffixes=("_readout", "_align"))
    rows = []
    for phase, lo, hi in phase_defs:
        sub = merged[(merged["time"] >= lo) & (merged["time"] < hi)].copy()
        day_prob_cols = [c for c in sub.columns if c.startswith("prob_day_")]
        dominant_prob = sub[day_prob_cols].max(axis=1).mean()
        rows.append(
            {
                "phase": phase,
                "time_start": lo,
                "time_end": hi,
                "mean_speed": sub["mean_speed"].mean(),
                "mean_deconfounded_cosine": sub["mean_deconfounded_cosine"].mean(),
                "frac_positive_cosine": sub["frac_positive_cosine"].mean(),
                "mean_nuisance_leakage_fraction": sub["mean_nuisance_leakage_fraction"].mean(),
                "mean_dominant_time_readout_prob": dominant_prob,
            }
        )
    phase = pd.DataFrame(rows)
    phase.to_csv(TAB / "new_response_phase_summary.csv", index=False)

    metrics = [
        ("mean_speed", "mean generated speed"),
        ("mean_deconfounded_cosine", "deconfounded pulse cosine"),
        ("frac_positive_cosine", "fraction positive cosine"),
        ("mean_nuisance_leakage_fraction", "nuisance leakage"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.2))
    for ax, (col, label) in zip(axes.flat, metrics):
        ax.bar(phase["phase"], phase[col], color=["#31572C", "#4F86C6", "#F28E2B", "#7A4EAB"])
        ax.set_title(label)
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("New analysis: drug-response trajectory phases", y=1.02)
    savefig(FIG / "new_response_phase_summary.png")
    return phase


def endpoint_vs_dynamic_analysis(adata, axis_info: dict, readout: pd.DataFrame, align: pd.DataFrame) -> pd.DataFrame:
    """Quantify what a static endpoint comparison misses."""
    times = adata.obs["time_point_processed"].astype(float).to_numpy()
    axis_values = axis_info["axis_values"]
    time_points, traj = load_all_trajectory_points()
    axis_vector = axis_info["axis_vector"]
    gen_axis = np.vstack([pts @ axis_vector for pts in traj])
    gen_axis_mean = gen_axis.mean(axis=1)
    gen_axis_std = gen_axis.std(axis=1)
    obs_axis_rows = []
    for t in sorted(np.unique(times)):
        vals = axis_values[times == t]
        obs_axis_rows.append({"time": t, "mean_axis": vals.mean(), "std_axis": vals.std(), "source": "observed"})
    gen_axis_df = pd.DataFrame(
        {
            "time": time_points,
            "mean_axis": gen_axis_mean,
            "std_axis": gen_axis_std,
            "source": "generated",
        }
    )
    axis_df = pd.concat([pd.DataFrame(obs_axis_rows), gen_axis_df], ignore_index=True)
    axis_df.to_csv(TAB / "new_endpoint_vs_dynamic_axis_summary.csv", index=False)

    diffs = np.diff(np.stack(traj, axis=0), axis=0)
    step_lengths = np.linalg.norm(diffs, axis=2)
    path_len = step_lengths.sum(axis=0)
    endpoint_len = np.linalg.norm(traj[-1] - traj[0], axis=1)
    tortuosity = path_len / np.maximum(endpoint_len, 1e-8)
    dyn_summary = pd.DataFrame(
        [
            {"quantity": "mean_generated_path_length", "value": float(path_len.mean())},
            {"quantity": "mean_generated_endpoint_displacement", "value": float(endpoint_len.mean())},
            {"quantity": "mean_path_to_endpoint_ratio", "value": float(tortuosity.mean())},
            {
                "quantity": "endpoint_only_day0_to_day3_observed_axis_shift",
                "value": float(
                    axis_values[times == 3.0].mean() - axis_values[times == 0.0].mean()
                ),
            },
        ]
    )
    dyn_summary.to_csv(TAB / "new_endpoint_vs_dynamic_summary.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2))
    axes[0, 0].bar(
        ["path length", "endpoint displacement"],
        [path_len.mean(), endpoint_len.mean()],
        color=["#146C94", "#9AA6B2"],
    )
    axes[0, 0].set_title("Generated route length vs endpoint displacement")
    axes[0, 0].set_ylabel("latent distance")

    axes[0, 1].plot(time_points, gen_axis_mean, color="#146C94", lw=2, label="generated trajectory")
    axes[0, 1].fill_between(time_points, gen_axis_mean - gen_axis_std, gen_axis_mean + gen_axis_std, color="#146C94", alpha=0.12)
    obs_df = pd.DataFrame(obs_axis_rows)
    axes[0, 1].scatter(obs_df["time"], obs_df["mean_axis"], color="#B23A48", s=35, label="observed days")
    endpoint_line = np.interp(time_points, [0, 3], [obs_df.loc[obs_df["time"] == 0.0, "mean_axis"].iloc[0], obs_df.loc[obs_df["time"] == 3.0, "mean_axis"].iloc[0]])
    axes[0, 1].plot(time_points, endpoint_line, color="#555555", lw=1.5, ls="--", label="endpoint-only straight summary")
    axes[0, 1].set_title("Response-axis movement is continuous, not just endpoint")
    axes[0, 1].set_xlabel("time")
    axes[0, 1].set_ylabel("response-axis coordinate")
    axes[0, 1].legend(frameon=False, fontsize=8)

    axes[1, 0].plot(readout["time"], readout["mean_speed"], color="#31572C", lw=2)
    axes[1, 0].set_title("Generated speed changes over time")
    axes[1, 0].set_xlabel("time")
    axes[1, 0].set_ylabel("mean speed")

    axes[1, 1].plot(align["time"], align["mean_deconfounded_cosine"], color="#146C94", lw=2, label="pulse cosine")
    axes[1, 1].plot(align["time"], align["mean_nuisance_leakage_fraction"], color="#B23A48", lw=2, label="nuisance leakage")
    axes[1, 1].axhline(0, color="#555555", lw=0.8)
    axes[1, 1].set_title("Directional validation varies along the path")
    axes[1, 1].set_xlabel("time")
    axes[1, 1].legend(frameon=False, fontsize=8)
    fig.suptitle("New analysis: endpoint-only comparison misses dynamic response structure", y=1.02)
    savefig(FIG / "new_endpoint_vs_dynamic_response.png")
    return dyn_summary


def state_heterogeneity_analysis(adata, traj_points: dict[float, np.ndarray], axis_info: dict) -> pd.DataFrame:
    times = adata.obs["time_point_processed"].astype(float).to_numpy()
    labels = axis_info["labels"]
    state_names = axis_info["state_names"]
    pca = axis_info["pca"]
    axis_idx = axis_info["axis_idx"]
    sign = axis_info["axis_sign"]
    edges = axis_info["edges"]

    rows = []
    for t in sorted(np.unique(times)):
        sub = labels[times == t]
        counts = np.bincount(sub, minlength=6)
        for i, count in enumerate(counts):
            rows.append(
                {
                    "source": "observed",
                    "time": t,
                    "state": state_names[i],
                    "count": int(count),
                    "fraction": float(count / counts.sum()),
                }
            )
    for t, pts in traj_points.items():
        pred_scores = pca.transform(pts)[:, axis_idx]
        pred_scores = sign * pred_scores
        pred = np.clip(np.digitize(pred_scores, edges[1:-1], right=False), 0, 5)
        counts = np.bincount(pred, minlength=6)
        for i, count in enumerate(counts):
            rows.append(
                {
                    "source": "generated",
                    "time": t,
                    "state": state_names[i],
                    "count": int(count),
                    "fraction": float(count / counts.sum()),
                }
            )
    state_df = pd.DataFrame(rows)
    state_df.to_csv(TAB / "new_state_distribution_observed_vs_generated.csv", index=False)

    colors = sns.color_palette("Spectral", n_colors=6)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, source in zip(axes, ["observed", "generated"]):
        pivot = (
            state_df[state_df["source"] == source]
            .pivot(index="time", columns="state", values="fraction")
            .fillna(0.0)
            .sort_index()
        )
        bottom = np.zeros(len(pivot))
        x = np.arange(len(pivot))
        for i, state in enumerate(state_names):
            vals = pivot[state].to_numpy()
            ax.bar(x, vals, bottom=bottom, color=colors[i], label=state)
            bottom += vals
        ax.set_xticks(x)
        ax.set_xticklabels([f"day {t:g}" for t in pivot.index])
        ax.set_title(f"{source.capitalize()} state composition")
        ax.set_ylabel("fraction")
        ax.set_ylim(0, 1.0)
    axes[1].legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle("New analysis: cell-state composition across the drug-response trajectory", y=1.02)
    savefig(FIG / "new_state_distribution_observed_vs_generated.png")

    state_meta = pd.DataFrame(
        {
            "state": state_names,
            "response_axis_component": axis_idx,
            "response_axis_time_correlation": axis_info["axis_time_correlation"],
            "axis_bin_left": edges[:-1],
            "axis_bin_right": edges[1:],
        }
    )
    state_meta.to_csv(TAB / "new_state_cluster_ordering.csv", index=False)
    return state_df


def state_specific_response_analysis(axis_info: dict) -> pd.DataFrame:
    time_points, traj = load_all_trajectory_points()
    axis_vector = axis_info["axis_vector"]
    edges = axis_info["edges"]
    state_names = axis_info["state_names"]
    rows = []
    for i in range(len(time_points) - 1):
        t = float(time_points[i])
        next_t = float(time_points[i + 1])
        start = traj[i]
        delta = traj[i + 1] - traj[i]
        start_axis = start @ axis_vector
        delta_axis = delta @ axis_vector
        speed = np.linalg.norm(delta, axis=1) / max(next_t - t, 1e-8)
        states = np.clip(np.digitize(start_axis, edges[1:-1], right=False), 0, 5)
        if t < 0.5:
            phase = "initial_0_to_0.5"
        elif t < 1.0:
            phase = "transition_0.5_to_1"
        elif t < 2.0:
            phase = "middle_1_to_2"
        else:
            phase = "late_2_to_3"
        for s_idx, state in enumerate(state_names):
            mask = states == s_idx
            if not mask.any():
                continue
            rows.append(
                {
                    "phase": phase,
                    "time": t,
                    "state": state,
                    "count": int(mask.sum()),
                    "mean_axis_velocity": float(delta_axis[mask].mean() / max(next_t - t, 1e-8)),
                    "forward_fraction": float((delta_axis[mask] > 0).mean()),
                    "mean_speed": float(speed[mask].mean()),
                }
            )
    state_resp = pd.DataFrame(rows)
    state_resp.to_csv(TAB / "new_state_specific_generated_response.csv", index=False)
    phase_state = (
        state_resp.groupby(["phase", "state"], as_index=False)
        .agg(
            mean_axis_velocity=("mean_axis_velocity", "mean"),
            forward_fraction=("forward_fraction", "mean"),
            mean_speed=("mean_speed", "mean"),
        )
    )
    phase_state.to_csv(TAB / "new_state_specific_generated_response_by_phase.csv", index=False)

    phase_order = ["initial_0_to_0.5", "transition_0.5_to_1", "middle_1_to_2", "late_2_to_3"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharey=True)
    for ax, value, title, cmap in [
        (axes[0], "mean_axis_velocity", "axis velocity", "vlag"),
        (axes[1], "forward_fraction", "forward fraction", "YlGnBu"),
        (axes[2], "mean_speed", "speed", "mako"),
    ]:
        pivot = (
            phase_state.pivot(index="state", columns="phase", values=value)
            .reindex(index=state_names, columns=phase_order)
        )
        sns.heatmap(pivot, ax=ax, cmap=cmap, center=0 if value == "mean_axis_velocity" else None, annot=True, fmt=".2f", cbar=True)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("")
    fig.suptitle("New analysis: which generated cell states are pushed, and when", y=1.05)
    savefig(FIG / "new_state_specific_response_heatmap.png")
    return phase_state


def metabolic_state_direction_analysis(adata, axis_info: dict) -> pd.DataFrame:
    times = adata.obs["time_point_processed"].astype(float).to_numpy()
    splits = adata.obs["mvab_anchor_split"].astype(str).to_numpy()
    conf = adata.obs["mvab_anchor_confidence"].astype(float).to_numpy()
    labels = axis_info["labels"]
    state_names = axis_info["state_names"]
    axis_vector = axis_info["axis_vector"]
    metabolic_v = np.asarray(adata.obsm["X_metabolic_velocity"])
    metabolic_axis = metabolic_v @ axis_vector
    rows = []
    for split in ["train", "heldout"]:
        for t in sorted(np.unique(times)):
            for s_idx, state in enumerate(state_names):
                mask = (splits == split) & (times == t) & (labels == s_idx) & (conf > 0)
                if mask.sum() == 0:
                    continue
                w = conf[mask]
                vals = metabolic_axis[mask]
                rows.append(
                    {
                        "split": split,
                        "time": t,
                        "state": state,
                        "anchor_count": int(mask.sum()),
                        "confidence_weighted_metabolic_axis": float(np.average(vals, weights=w)),
                        "forward_fraction": float((vals > 0).mean()),
                        "mean_confidence": float(w.mean()),
                    }
                )
    meta = pd.DataFrame(rows)
    meta.to_csv(TAB / "new_state_specific_metabolic_direction.csv", index=False)
    held = meta[meta["split"] == "heldout"].copy()
    pivot = held.pivot_table(index="state", columns="time", values="confidence_weighted_metabolic_axis", aggfunc="mean")
    pivot = pivot.reindex(index=state_names)
    plt.figure(figsize=(7.2, 4.8))
    sns.heatmap(pivot, cmap="vlag", center=0, annot=True, fmt=".2f", cbar_kws={"label": "heldout metabolic velocity along response axis"})
    plt.title("New analysis: heldout old/new RNA direction by cell state")
    plt.xlabel("treatment day")
    plt.ylabel("response-axis state")
    savefig(FIG / "new_state_metabolic_direction_heatmap.png")
    return meta


def gene_expression_dynamics(adata, genes: pd.DataFrame) -> pd.DataFrame:
    top_positive = genes[genes["direction"] == "positive"].head(5)["gene"].tolist()
    top_negative = genes[genes["direction"] == "negative"].head(5)["gene"].tolist()
    selected = []
    for g in top_positive + top_negative:
        if g not in selected and g in adata.var_names:
            selected.append(g)
    times = adata.obs["time_point_processed"].astype(float).to_numpy()
    layer = adata.layers["log1p_norm"] if "log1p_norm" in adata.layers else adata.X
    rows = []
    for gene in selected:
        j = int(adata.var_names.get_loc(gene))
        vec = dense_slice(layer[:, j]).reshape(-1)
        score_row = genes[genes["gene"] == gene].iloc[0]
        for t in sorted(np.unique(times)):
            vals = vec[times == t]
            rows.append(
                {
                    "gene": gene,
                    "time": t,
                    "mean_log1p_norm": float(np.mean(vals)),
                    "direction": score_row["direction"],
                    "signed_score": float(score_row["signed_score"]),
                }
            )
    expr = pd.DataFrame(rows)
    expr.to_csv(TAB / "new_top_projected_gene_expression_dynamics.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True)
    for ax, direction, color, title in [
        (axes[0], "positive", "#146C94", "positive projected genes"),
        (axes[1], "negative", "#B23A48", "negative projected genes"),
    ]:
        sub_expr = expr[expr["direction"] == direction]
        for gene, sub in sub_expr.groupby("gene"):
            ax.plot(sub["time"], sub["mean_log1p_norm"], marker="o", lw=2, alpha=0.86, color=color, label=gene)
        ax.set_title(title)
        ax.set_xlabel("treatment day")
        ax.set_ylabel("mean log1p normalized expression")
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle("New analysis: expression dynamics of top PulseBridge velocity-projection genes", y=1.03)
    savefig(FIG / "new_top_projected_gene_expression_dynamics.png")
    return expr


def gene_program_readout(adata, genes: pd.DataFrame) -> pd.DataFrame:
    top_genes = genes.head(80).copy()
    modules = {
        "polyamine_stress": ["SAT1"],
        "adhesion_stress": ["LGALS1"],
        "proteostasis_heat_shock": [g for g in top_genes["gene"] if g.startswith("HSP")],
        "mitochondrial_readout": [g for g in top_genes["gene"] if g.startswith("MT-")],
        "ribosome_translation": [g for g in top_genes["gene"] if g.startswith("RPL") or g.startswith("RPS") or g == "NCL"],
        "immune_surface_stress": ["B2M"],
        "iron_storage": ["FTL"],
    }
    modules = {
        name: [g for g in dict.fromkeys(gs) if g in adata.var_names]
        for name, gs in modules.items()
    }
    modules = {name: gs for name, gs in modules.items() if gs}
    layer = adata.layers["log1p_norm"] if "log1p_norm" in adata.layers else adata.X
    times = adata.obs["time_point_processed"].astype(float).to_numpy()
    rows = []
    for module, module_genes in modules.items():
        idx = [int(adata.var_names.get_loc(g)) for g in module_genes]
        mat = dense_slice(layer[:, idx])
        score = mat.mean(axis=1)
        for t in sorted(np.unique(times)):
            vals = score[times == t]
            rows.append(
                {
                    "module": module,
                    "time": t,
                    "gene_count": len(module_genes),
                    "genes": ";".join(module_genes),
                    "mean_score": float(vals.mean()),
                    "z_score": np.nan,
                }
            )
    module_df = pd.DataFrame(rows)
    for module, sub_idx in module_df.groupby("module").groups.items():
        vals = module_df.loc[sub_idx, "mean_score"].to_numpy()
        sd = vals.std()
        z = (vals - vals.mean()) / (sd if sd > 1e-8 else 1.0)
        module_df.loc[sub_idx, "z_score"] = z
    module_df.to_csv(TAB / "new_gene_program_readout_by_day.csv", index=False)
    pivot = module_df.pivot(index="module", columns="time", values="z_score")
    plt.figure(figsize=(7.2, max(4.2, 0.45 * len(pivot) + 1.8)))
    sns.heatmap(pivot, cmap="vlag", center=0, annot=True, fmt=".2f", cbar_kws={"label": "within-module z-score"})
    plt.title("New analysis: pathway-like readouts from PulseBridge projected genes")
    plt.xlabel("treatment day")
    plt.ylabel("gene readout group")
    savefig(FIG / "new_gene_program_readout_heatmap.png")
    return module_df


def original_paper_marker_readout(adata) -> pd.DataFrame:
    """Track marker genes highlighted by the source Well-TEMP-seq paper."""
    markers = [
        ("DKK1", "reported_tumor_suppressor_up"),
        ("GAS5", "reported_tumor_suppressor_up"),
        ("EREG", "reported_oncogene_down"),
        ("MALAT1", "reported_oncogene_down"),
        ("STAT1", "reported_regulon"),
        ("HEYL", "reported_regulon"),
        ("PITX1", "reported_regulon"),
    ]
    markers = [(g, role) for g, role in markers if g in adata.var_names]
    times = adata.obs["time_point_processed"].astype(float).to_numpy()
    layer_names = [name for name in ["log1p_norm", "new", "old"] if name in adata.layers]
    rows = []
    for layer_name in layer_names:
        layer = adata.layers[layer_name]
        for gene, paper_role in markers:
            j = int(adata.var_names.get_loc(gene))
            vec = dense_slice(layer[:, j]).reshape(-1)
            day_means = {}
            for t in sorted(np.unique(times)):
                vals = vec[times == t]
                day_means[float(t)] = float(vals.mean())
            baseline = day_means.get(0.0, np.nan)
            for t, mean_value in day_means.items():
                rows.append(
                    {
                        "gene": gene,
                        "paper_role": paper_role,
                        "layer": layer_name,
                        "time": t,
                        "mean_value": mean_value,
                        "delta_vs_day0": mean_value - baseline,
                    }
                )
    marker_df = pd.DataFrame(rows)
    marker_df.to_csv(TAB / "new_original_paper_marker_readout_by_layer.csv", index=False)

    fig, axes = plt.subplots(1, len(layer_names), figsize=(4.6 * len(layer_names), 5.0), sharey=True)
    if len(layer_names) == 1:
        axes = [axes]
    gene_order = [g for g, _ in markers]
    for ax, layer_name in zip(axes, layer_names):
        sub = marker_df[marker_df["layer"] == layer_name].copy()
        z_rows = []
        for gene, gsub in sub.groupby("gene"):
            vals = gsub.sort_values("time")["mean_value"].to_numpy()
            sd = vals.std()
            z = (vals - vals.mean()) / (sd if sd > 1e-8 else 1.0)
            for time, value in zip(gsub.sort_values("time")["time"], z):
                z_rows.append({"gene": gene, "time": time, "z": value})
        z_df = pd.DataFrame(z_rows)
        pivot = z_df.pivot(index="gene", columns="time", values="z").reindex(gene_order)
        sns.heatmap(pivot, ax=ax, cmap="vlag", center=0, annot=True, fmt=".2f", cbar=True)
        ax.set_title(layer_name)
        ax.set_xlabel("treatment day")
        ax.set_ylabel("source-paper marker")
    fig.suptitle("New analysis: source Well-TEMP-seq marker readouts in local HCT116 data", y=1.04)
    savefig(FIG / "new_original_paper_marker_readout.png")
    return marker_df


def targeted_drug_response_support_analysis(adata, axis_info: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Quantify whether the cancer/drug/gene story is supported by measured data."""
    times = adata.obs["time_point_processed"].astype(float).to_numpy()
    labels = axis_info["labels"]
    state_names = axis_info["state_names"]
    layers = [name for name in ["log1p_norm", "new", "old"] if name in adata.layers]

    gene_roles = {
        "DKK1": ("tumor_suppressor_reactivation", "expected_up"),
        "GAS5": ("tumor_suppressor_reactivation", "expected_up"),
        "EREG": ("oncogene_growth_repression", "expected_down"),
        "MALAT1": ("oncogene_caution", "expected_down_not_reproduced"),
        "STAT1": ("source_interferon_regulon", "weak_local_gene_readout"),
        "B2M": ("immune_surface_interferon_like", "expected_up"),
        "HLA-A": ("immune_surface_interferon_like", "expected_up"),
        "HLA-B": ("immune_surface_interferon_like", "expected_up"),
        "HLA-C": ("immune_surface_interferon_like", "expected_up"),
        "IFITM3": ("immune_surface_interferon_like", "expected_up"),
        "SAT1": ("polyamine_metabolic_stress", "expected_up_hypothesis"),
        "LGALS1": ("adaptive_adhesion_caution", "expected_up_caution"),
        "PCNA": ("proliferation_readout", "expected_down"),
        "HSP90AB1": ("proteostasis_heat_shock", "expected_down"),
        "HSP90AA1": ("proteostasis_heat_shock", "expected_down"),
        "HSPD1": ("proteostasis_heat_shock", "expected_down"),
        "FTL": ("iron_redox_readout", "expected_down"),
        "MT-CO1": ("mitochondrial_stress_readout", "expected_up_caution"),
        "MT-CO2": ("mitochondrial_stress_readout", "expected_up_caution"),
        "MT-CO3": ("mitochondrial_stress_readout", "expected_up_caution"),
        "MT-ND4": ("mitochondrial_stress_readout", "expected_up_caution"),
        "MT-CYB": ("mitochondrial_stress_readout", "expected_up_caution"),
    }
    selected_genes = [g for g in gene_roles if g in adata.var_names]

    gene_rows = []
    for layer_name in layers:
        layer = adata.layers[layer_name]
        for gene in selected_genes:
            j = int(adata.var_names.get_loc(gene))
            vec = dense_slice(layer[:, j]).reshape(-1)
            day_means = {}
            day_stats = {}
            for t in sorted(np.unique(times)):
                vals = vec[times == t]
                day_means[float(t)] = float(vals.mean())
                day_stats[float(t)] = {
                    "cell_count": int(vals.size),
                    "std": float(vals.std()),
                    "sem": float(vals.std() / np.sqrt(max(vals.size, 1))),
                }
            baseline = day_means.get(0.0, np.nan)
            for t, value in day_means.items():
                role, expected = gene_roles[gene]
                gene_rows.append(
                    {
                        "gene": gene,
                        "role": role,
                        "expected_direction": expected,
                        "layer": layer_name,
                        "time": t,
                        "mean_value": value,
                        "delta_vs_day0": value - baseline,
                        "cell_count": day_stats[t]["cell_count"],
                        "std": day_stats[t]["std"],
                        "sem": day_stats[t]["sem"],
                    }
                )
    gene_day = pd.DataFrame(gene_rows)
    gene_day.to_csv(TAB / "targeted_drug_response_gene_by_day_layer.csv", index=False)

    module_defs = [
        {
            "module": "tumor_suppressor_reactivation",
            "label": "DKK1/GAS5 reactivation",
            "genes": ["DKK1", "GAS5"],
            "expected": "up",
            "claim_class": "desired_anti_tumor_like",
        },
        {
            "module": "oncogene_proliferation_repression",
            "label": "EREG/PCNA repression",
            "genes": ["EREG", "PCNA"],
            "expected": "down",
            "claim_class": "desired_anti_tumor_like",
        },
        {
            "module": "immune_surface_interferon_like",
            "label": "B2M/HLA/IFITM3 immune surface",
            "genes": ["B2M", "HLA-A", "HLA-B", "HLA-C", "IFITM3"],
            "expected": "up",
            "claim_class": "desired_immune_like",
        },
        {
            "module": "polyamine_mito_adaptive_stress",
            "label": "SAT1/LGALS1/mitochondrial stress",
            "genes": ["SAT1", "LGALS1", "MT-CO1", "MT-CO2", "MT-CO3", "MT-ND4", "MT-CYB"],
            "expected": "up",
            "claim_class": "adaptive_stress_caution",
        },
        {
            "module": "proteostasis_iron_reduction",
            "label": "HSP/FTL reduction",
            "genes": ["HSP90AB1", "HSP90AA1", "HSPD1", "FTL"],
            "expected": "down",
            "claim_class": "remodeling_support",
        },
        {
            "module": "malat1_caution",
            "label": "MALAT1 caution",
            "genes": ["MALAT1"],
            "expected": "down",
            "claim_class": "not_reproduced_caution",
        },
    ]

    module_rows = []
    state_rows = []
    layer = adata.layers["log1p_norm"] if "log1p_norm" in adata.layers else adata.X
    for module in module_defs:
        module_genes = [g for g in module["genes"] if g in adata.var_names]
        if not module_genes:
            continue
        idx = [int(adata.var_names.get_loc(g)) for g in module_genes]
        mat = dense_slice(layer[:, idx])
        score = mat.mean(axis=1)
        day_values = {}
        day_stats = {}
        for t in sorted(np.unique(times)):
            mask_t = times == t
            vals = score[mask_t]
            day_values[float(t)] = float(vals.mean())
            day_stats[float(t)] = {
                "cell_count": int(vals.size),
                "std": float(vals.std()),
                "sem": float(vals.std() / np.sqrt(max(vals.size, 1))),
            }
            module_rows.append(
                {
                    "module": module["module"],
                    "label": module["label"],
                    "claim_class": module["claim_class"],
                    "expected": module["expected"],
                    "genes": ";".join(module_genes),
                    "gene_count": len(module_genes),
                    "time": float(t),
                    "mean_score": day_values[float(t)],
                    "delta_vs_day0": np.nan,
                    "z_score": np.nan,
                    "cell_count": day_stats[float(t)]["cell_count"],
                    "std": day_stats[float(t)]["std"],
                    "sem": day_stats[float(t)]["sem"],
                }
            )
            for s_idx, state in enumerate(state_names):
                mask = mask_t & (labels == s_idx)
                if not mask.any():
                    continue
                state_rows.append(
                    {
                        "module": module["module"],
                        "label": module["label"],
                        "claim_class": module["claim_class"],
                        "expected": module["expected"],
                        "genes": ";".join(module_genes),
                        "time": float(t),
                        "state": state,
                        "cell_count": int(mask.sum()),
                        "mean_score": float(score[mask].mean()),
                    }
                )
        baseline = day_values.get(0.0, np.nan)
        values = np.asarray([row["mean_score"] for row in module_rows if row["module"] == module["module"]], dtype=float)
        sd = values.std()
        module_indices = [i for i, row in enumerate(module_rows) if row["module"] == module["module"]]
        for i in module_indices:
            module_rows[i]["delta_vs_day0"] = module_rows[i]["mean_score"] - baseline
            module_rows[i]["z_score"] = (module_rows[i]["mean_score"] - values.mean()) / (sd if sd > 1e-8 else 1.0)

    module_day = pd.DataFrame(module_rows)
    module_day.to_csv(TAB / "targeted_drug_response_module_by_day.csv", index=False)
    state_module = pd.DataFrame(state_rows)
    state_module.to_csv(TAB / "targeted_drug_response_module_by_state_day.csv", index=False)

    summary_rows = []
    for module, sub in module_day.groupby("module"):
        first = sub.iloc[0]
        state_sub = state_module[state_module["module"] == module]
        day0 = float(sub.loc[sub["time"] == 0.0, "mean_score"].iloc[0]) if (sub["time"] == 0.0).any() else np.nan
        day3 = float(sub.loc[sub["time"] == 3.0, "mean_score"].iloc[0]) if (sub["time"] == 3.0).any() else np.nan
        early_mask = (state_sub["time"] == 0.0) & (state_sub["state"].isin(["state_1", "state_2"]))
        late3_mask = (state_sub["time"] == 3.0) & (state_sub["state"].isin(["state_5", "state_6"]))
        late2_mask = (state_sub["time"] == 2.0) & (state_sub["state"].isin(["state_5", "state_6"]))
        day0_early = np.average(
            state_sub.loc[early_mask, "mean_score"],
            weights=state_sub.loc[early_mask, "cell_count"],
        ) if early_mask.any() else np.nan
        day3_late = np.average(
            state_sub.loc[late3_mask, "mean_score"],
            weights=state_sub.loc[late3_mask, "cell_count"],
        ) if late3_mask.any() else np.nan
        day2_late = np.average(
            state_sub.loc[late2_mask, "mean_score"],
            weights=state_sub.loc[late2_mask, "cell_count"],
        ) if late2_mask.any() else np.nan
        delta_day3 = day3 - day0
        expected = first["expected"]
        if expected == "up":
            status = "supports_expected_direction" if delta_day3 > 0 else "does_not_support_expected_direction"
        elif expected == "down":
            status = "supports_expected_direction" if delta_day3 < 0 else "does_not_support_expected_direction"
        else:
            status = "descriptive"
        summary_rows.append(
            {
                "module": module,
                "label": first["label"],
                "claim_class": first["claim_class"],
                "genes": first["genes"],
                "expected": expected,
                "day0_mean": day0,
                "day3_mean": day3,
                "day3_minus_day0": delta_day3,
                "day0_early_state12": float(day0_early),
                "day2_late_state56": float(day2_late),
                "day3_late_state56": float(day3_late),
                "day3_late_minus_day0_early": float(day3_late - day0_early),
                "support_status": status,
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(TAB / "targeted_drug_response_support_summary.csv", index=False)

    plot_targeted_drug_response_gene_panel(gene_day, module_day, state_module, summary)
    return gene_day, module_day, state_module, summary


def plot_targeted_drug_response_gene_panel(
    gene_day: pd.DataFrame,
    module_day: pd.DataFrame,
    state_module: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    """Draw the result panel that backs the drug/cancer/gene interpretation."""
    fig = plt.figure(figsize=(15.2, 9.8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.10, 1.15], height_ratios=[1.05, 1.0], hspace=0.46, wspace=0.52)
    ax_marker = fig.add_subplot(gs[0, 0])
    ax_module = fig.add_subplot(gs[0, 1])
    ax_state = fig.add_subplot(gs[1, 0])
    ax_summary = fig.add_subplot(gs[1, 1])

    marker_order = ["DKK1", "GAS5", "EREG", "MALAT1", "STAT1", "B2M", "HLA-A", "HLA-B", "IFITM3", "SAT1", "LGALS1", "PCNA"]
    marker_sub = gene_day[(gene_day["time"] == 3.0) & (gene_day["gene"].isin(marker_order))].copy()
    marker_pivot = (
        marker_sub.pivot_table(index="gene", columns="layer", values="delta_vs_day0", aggfunc="mean")
        .reindex(index=[g for g in marker_order if g in set(marker_sub["gene"])], columns=["log1p_norm", "new", "old"])
    )
    sns.heatmap(
        marker_pivot,
        ax=ax_marker,
        cmap="vlag",
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "day3 - day0 mean"},
    )
    ax_marker.set_title("A. Targeted genes by RNA layer")
    ax_marker.set_xlabel("")
    ax_marker.set_ylabel("")
    ax_marker.set_xticklabels(["total/log", "new", "old"], rotation=0)

    plot_modules = [
        "tumor_suppressor_reactivation",
        "oncogene_proliferation_repression",
        "immune_surface_interferon_like",
        "polyamine_mito_adaptive_stress",
        "proteostasis_iron_reduction",
    ]
    palette = {
        "tumor_suppressor_reactivation": "#31572C",
        "oncogene_proliferation_repression": "#B23A48",
        "immune_surface_interferon_like": "#146C94",
        "polyamine_mito_adaptive_stress": "#7A4EAB",
        "proteostasis_iron_reduction": "#6E6259",
    }
    for module in plot_modules:
        sub = module_day[module_day["module"] == module].sort_values("time")
        if sub.empty:
            continue
        label = str(sub["label"].iloc[0])
        ax_module.plot(sub["time"], sub["z_score"], marker="o", lw=2.2, color=palette[module], label=label)
    ax_module.axhline(0, color="#555555", lw=0.8)
    ax_module.set_title("B. Drug-response module trajectories")
    ax_module.set_xlabel("treatment day")
    ax_module.set_ylabel("within-module z-score")
    ax_module.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)

    state_plot_rows = []
    for module in plot_modules:
        sub = state_module[state_module["module"] == module]
        if sub.empty:
            continue
        label = str(sub["label"].iloc[0])
        for group_label, time, states in [
            ("day0 early states", 0.0, ["state_1", "state_2"]),
            ("day2 late states", 2.0, ["state_5", "state_6"]),
            ("day3 late states", 3.0, ["state_5", "state_6"]),
        ]:
            mask = (sub["time"] == time) & (sub["state"].isin(states))
            if not mask.any():
                continue
            mean_value = np.average(sub.loc[mask, "mean_score"], weights=sub.loc[mask, "cell_count"])
            state_plot_rows.append({"module": module, "label": label, "state_group": group_label, "mean_score": mean_value})
    state_plot = pd.DataFrame(state_plot_rows)
    pivot = state_plot.pivot(index="label", columns="state_group", values="mean_score")
    order_labels = [summary.loc[summary["module"] == m, "label"].iloc[0] for m in plot_modules if (summary["module"] == m).any()]
    pivot = pivot.reindex(index=order_labels, columns=["day0 early states", "day2 late states", "day3 late states"])
    row_z = pivot.copy()
    for idx in row_z.index:
        vals = row_z.loc[idx].to_numpy(dtype=float)
        sd = np.nanstd(vals)
        row_z.loc[idx] = (vals - np.nanmean(vals)) / (sd if sd > 1e-8 else 1.0)
    sns.heatmap(
        row_z,
        ax=ax_state,
        cmap="vlag",
        center=0,
        annot=pivot,
        fmt=".2f",
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "row-scaled state score"},
    )
    ax_state.set_title("C. Do late response states carry the signal?")
    ax_state.set_xlabel("")
    ax_state.set_ylabel("")
    ax_state.set_xticklabels(ax_state.get_xticklabels(), rotation=18, ha="right")

    summary_plot = summary[summary["module"].isin(plot_modules)].copy()
    summary_plot["expected_signed_delta"] = np.where(
        summary_plot["expected"] == "down",
        -summary_plot["day3_minus_day0"],
        summary_plot["day3_minus_day0"],
    )
    summary_plot = summary_plot.set_index("label").reindex(order_labels).reset_index()
    colors = ["#31572C" if v > 0 else "#B23A48" for v in summary_plot["expected_signed_delta"]]
    y = np.arange(len(summary_plot))
    ax_summary.barh(y, summary_plot["expected_signed_delta"], color=colors, alpha=0.88)
    ax_summary.axvline(0, color="#333333", lw=0.8)
    ax_summary.set_yticks(y)
    ax_summary.set_yticklabels(summary_plot["label"], fontsize=8)
    ax_summary.invert_yaxis()
    xmax = max(0.05, float(summary_plot["expected_signed_delta"].abs().max()) * 1.18)
    ax_summary.set_xlim(-0.05 * xmax, xmax)
    ax_summary.set_xlabel("signed support for expected direction")
    ax_summary.set_title("D. Directional support summary")
    ax_summary.text(
        0.98,
        0.04,
        "positive = matches expected direction",
        transform=ax_summary.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#333333",
    )

    fig.suptitle("Targeted HCT116 5-AZA-CdR gene evidence for the PulseBridge story", y=0.99)
    savefig(FIG / "story_targeted_drug_response_gene_panel.png")


def plot_story_dynamic_drug_response_route(
    state_df: pd.DataFrame,
    module_day: pd.DataFrame,
    summary: pd.DataFrame,
    gene_day: pd.DataFrame,
) -> None:
    """Draw a non-heatmap route figure for the dynamic cancer drug-response claim."""
    from matplotlib.patches import Circle, FancyArrowPatch, Wedge

    obs = state_df[state_df["source"] == "observed"].copy()
    obs = obs.set_index(["time", "state"])["fraction"]

    def frac(time: float, states: list[str]) -> float:
        return float(sum(obs.get((time, state), 0.0) for state in states))

    def row(module: str) -> pd.Series:
        return summary[summary["module"] == module].iloc[0]

    def delta_text(module: str) -> tuple[str, str]:
        r = row(module)
        day_delta = float(r["day3_minus_day0"])
        state_delta = float(r["day3_late_minus_day0_early"])
        if r["expected"] == "down":
            main = f"day3-day0 {day_delta:+.2f}"
            state = f"late states {state_delta:+.2f}"
        else:
            main = f"day3-day0 {day_delta:+.2f}"
            state = f"late states {state_delta:+.2f}"
        return main, state

    def gene_delta(gene: str, layer: str) -> float:
        sub = gene_day[(gene_day["gene"] == gene) & (gene_day["layer"] == layer) & (gene_day["time"] == 3.0)]
        return float(sub["delta_vs_day0"].iloc[0]) if len(sub) else float("nan")

    fig, ax = plt.subplots(figsize=(15.2, 8.6))
    ax.set_xlim(-0.45, 3.65)
    ax.set_ylim(-2.12, 2.48)
    ax.axis("off")

    route_x = np.array([0.0, 1.0, 2.0, 3.0])
    route_y = np.array([0.00, 0.24, 0.62, 0.43])
    day_colors = {0.0: "#31572C", 1.0: "#4F86C6", 2.0: "#F28E2B", 3.0: "#B23A48"}

    ax.plot(route_x, route_y, color="#20323F", lw=4.2, solid_capstyle="round", zorder=2)
    for x0, y0, x1, y1 in zip(route_x[:-1], route_y[:-1], route_x[1:], route_y[1:]):
        ax.add_patch(
            FancyArrowPatch(
                (x0, y0),
                (x1, y1),
                arrowstyle="-|>",
                mutation_scale=16,
                lw=0,
                color="#20323F",
                zorder=3,
            )
        )

    shortcut_y = -0.58
    ax.plot([0, 3], [shortcut_y, shortcut_y], color="#8A8A8A", lw=2.0, ls="--", zorder=1)
    ax.add_patch(
        FancyArrowPatch(
            (0, shortcut_y),
            (3, shortcut_y),
            arrowstyle="-|>",
            mutation_scale=13,
            lw=1.7,
            color="#8A8A8A",
            linestyle="--",
        )
    )
    ax.text(
        1.5,
        shortcut_y - 0.16,
        "endpoint-only view: one day0-to-day3 difference",
        ha="center",
        va="top",
        fontsize=10,
        color="#555555",
    )

    def draw_state_station(x: float, y: float, time: float) -> None:
        early = frac(time, ["state_1", "state_2"])
        late = frac(time, ["state_5", "state_6"])
        other = max(0.0, 1.0 - early - late)
        start = 90.0
        radius = 0.155
        for value, color in [(early, "#85A87A"), (other, "#D8DEE5"), (late, "#C66C4E")]:
            if value <= 0:
                continue
            theta2 = start + 360.0 * value
            ax.add_patch(Wedge((x, y), radius, start, theta2, facecolor=color, edgecolor="white", lw=0.7, zorder=6))
            start = theta2
        ax.add_patch(Circle((x, y), radius, facecolor="none", edgecolor="#20323F", lw=1.2, zorder=7))
        ax.text(x, y + 0.27, f"day {time:g}", ha="center", va="bottom", fontsize=10, color=day_colors[time], fontweight="bold")
        ax.text(
            x,
            y - 0.30,
            f"early {early:.0%}\nlate {late:.0%}",
            ha="center",
            va="top",
            fontsize=8.5,
            color="#333333",
        )

    for t, x, y in zip([0.0, 1.0, 2.0, 3.0], route_x, route_y):
        draw_state_station(x, y, t)

    ax.text(
        -0.38,
        0.91,
        "CellCompass route\n+ state timing",
        ha="left",
        va="center",
        fontsize=11.5,
        fontweight="bold",
        color="#20323F",
    )
    ax.text(
        3.46,
        0.56,
        "late response\nstates remain high",
        ha="right",
        va="center",
        fontsize=10,
        color="#333333",
    )

    event_specs = [
        (
            "tumor_suppressor_reactivation",
            "DKK1/GAS5 reactivation",
            "demethylation response",
            1.74,
            "#31572C",
            "desired anti-tumor-like",
        ),
        (
            "oncogene_proliferation_repression",
            "EREG/PCNA repression",
            "growth/proliferation readout",
            1.38,
            "#B23A48",
            "desired anti-tumor-like",
        ),
        (
            "immune_surface_interferon_like",
            "B2M/HLA/IFITM3 up",
            "immune-surface / viral-mimicry-like",
            1.02,
            "#146C94",
            "desired immune-like",
        ),
        (
            "polyamine_mito_adaptive_stress",
            "SAT1/LGALS1/mitochondrial up",
            "adaptive stress candidate",
            -1.10,
            "#7A4EAB",
            "follow-up branch",
        ),
        (
            "proteostasis_iron_reduction",
            "HSP/FTL down",
            "proteostasis/iron remodeling",
            -1.46,
            "#6E6259",
            "supporting remodeling",
        ),
    ]

    ax.text(-0.38, 2.07, "desired response components", ha="left", va="center", fontsize=11, color="#31572C", fontweight="bold")
    ax.text(-0.38, -0.88, "adaptive/remodeling components", ha="left", va="center", fontsize=11, color="#7A4EAB", fontweight="bold")

    for module, label, subtitle, y, color, group in event_specs:
        main, state = delta_text(module)
        lw = 2.0 + min(2.4, abs(float(row(module)["day3_minus_day0"])) * 1.2)
        ax.add_patch(
            FancyArrowPatch(
                (0.18, y),
                (2.86, y),
                arrowstyle="-|>",
                mutation_scale=16,
                lw=lw,
                color=color,
                alpha=0.88,
                zorder=4,
            )
        )
        ax.scatter([0.18], [y], s=58, facecolor="white", edgecolor=color, linewidth=2.0, zorder=5)
        ax.scatter([2.86], [y], s=70, facecolor=color, edgecolor="white", linewidth=0.8, zorder=5)
        ax.text(0.20, y + 0.085, label, ha="left", va="bottom", fontsize=10.4, fontweight="bold", color=color)
        ax.text(0.20, y - 0.09, subtitle, ha="left", va="top", fontsize=8.5, color="#333333")
        ax.text(
            2.93,
            y,
            f"{main}; {state}",
            ha="left",
            va="center",
            fontsize=9.2,
            color="#333333",
        )

    ax.text(
        0.02,
        -1.86,
        (
            f"MALAT1 caution: local total RNA is {gene_delta('MALAT1', 'log1p_norm'):+.2f} "
            f"and old RNA is {gene_delta('MALAT1', 'old'):+.2f}; do not use it as support for repression."
        ),
        ha="left",
        va="center",
        fontsize=9.2,
        color="#5F6368",
    )
    ax.plot([0.0, 3.0], [-1.72, -1.72], color="#B0B0B0", lw=1.2, ls=":")

    ax.text(
        1.5,
        2.40,
        "Dynamic cancer-drug profiling: the same endpoint distance can hide different response routes",
        ha="center",
        va="top",
        fontsize=15,
        fontweight="bold",
        color="#111111",
    )
    ax.text(
        1.5,
        2.23,
        "HCT116 5-AZA-CdR response is resolved into state timing plus drug-relevant gene events, not a generic endpoint heatmap.",
        ha="center",
        va="top",
        fontsize=10.5,
        color="#333333",
    )

    legend_x = 3.05
    ax.add_patch(Wedge((legend_x, -0.05), 0.055, 90, 260, facecolor="#85A87A", edgecolor="white", lw=0.4))
    ax.add_patch(Wedge((legend_x, -0.05), 0.055, 260, 450, facecolor="#C66C4E", edgecolor="white", lw=0.4))
    ax.add_patch(Circle((legend_x, -0.05), 0.055, facecolor="none", edgecolor="#20323F", lw=0.7))
    ax.text(legend_x + 0.08, -0.05, "station pies: early vs late state fraction", ha="left", va="center", fontsize=8.5, color="#333333")

    savefig(FIG / "story_dynamic_drug_response_route.png")


def plot_story_drug_response_evidence_panel(
    adata,
    axis_info: dict,
    state_df: pd.DataFrame,
    module_day: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    """Evidence-first non-heatmap figure for the drug/cancer/gene story."""
    times = adata.obs["time_point_processed"].astype(float).to_numpy()
    axis_values = np.asarray(axis_info["axis_values"], dtype=float)
    axis_values = axis_values - axis_values[times == 0.0].mean()
    axis_vector = axis_info["axis_vector"]
    z = np.load(TRAJ_PATH)
    rng = np.random.default_rng(29)

    fig = plt.figure(figsize=(14.8, 10.0))
    gs = fig.add_gridspec(2, 2, hspace=0.40, wspace=0.32)
    ax_dist = fig.add_subplot(gs[0, 0])
    ax_state = fig.add_subplot(gs[0, 1])
    ax_modules = fig.add_subplot(gs[1, 0])
    ax_support = fig.add_subplot(gs[1, 1])

    # A. Observed single-cell response-axis distributions with generated slices overlaid.
    sample_rows = []
    for t in sorted(np.unique(times)):
        idx = np.where(times == t)[0]
        keep = rng.choice(idx, size=min(1300, idx.size), replace=False)
        sample_rows.extend({"day": f"day {t:g}", "time": float(t), "axis": float(axis_values[i])} for i in keep)
    sample_df = pd.DataFrame(sample_rows)
    day_order = [f"day {t:g}" for t in sorted(np.unique(times))]
    day_palette = ["#31572C", "#4F86C6", "#F28E2B", "#B23A48"]
    sns.violinplot(
        data=sample_df,
        x="day",
        y="axis",
        order=day_order,
        palette=day_palette,
        inner=None,
        cut=0,
        linewidth=0.8,
        saturation=0.7,
        ax=ax_dist,
    )
    sns.boxplot(
        data=sample_df,
        x="day",
        y="axis",
        order=day_order,
        width=0.18,
        showcaps=False,
        boxprops={"facecolor": "white", "edgecolor": "#222222", "linewidth": 0.8},
        whiskerprops={"linewidth": 0.8, "color": "#222222"},
        medianprops={"linewidth": 1.0, "color": "#222222"},
        showfliers=False,
        ax=ax_dist,
    )
    generated_means = []
    generated_times = []
    base = float((z[f"points_{int(z['observed_time_indices'][0]):06d}"] @ axis_vector).mean())
    for t, idx in zip(z["observed_time_points"], z["observed_time_indices"]):
        pts = z[f"points_{int(idx):06d}"]
        generated_times.append(float(t))
        generated_means.append(float((pts @ axis_vector).mean() - base))
    ax_dist.plot(np.arange(len(generated_means)), generated_means, color="#111111", lw=1.8, marker="D", ms=5, label="generated slice mean")
    ax_dist.axhline(0, color="#777777", lw=0.8, ls=":")
    ax_dist.set_title("A. Observed response-axis distributions and generated slices")
    ax_dist.set_xlabel("")
    ax_dist.set_ylabel("response-axis coordinate\n(centered at day0)")
    ax_dist.legend(frameon=False, fontsize=8, loc="upper left")

    # B. State carrying evidence: early and late state fractions over days.
    state_groups = {
        "early states 1-2": ["state_1", "state_2"],
        "late states 5-6": ["state_5", "state_6"],
    }
    for source, linestyle, marker, alpha in [("observed", "-", "o", 0.95), ("generated", "--", "s", 0.82)]:
        sub = state_df[state_df["source"] == source]
        for label, states in state_groups.items():
            rows = []
            for t, g in sub.groupby("time"):
                rows.append({"time": float(t), "fraction": float(g[g["state"].isin(states)]["fraction"].sum())})
            rows = pd.DataFrame(rows).sort_values("time")
            color = "#31572C" if "early" in label else "#B23A48"
            ax_state.plot(
                rows["time"],
                rows["fraction"],
                linestyle=linestyle,
                marker=marker,
                lw=2.2,
                color=color,
                alpha=alpha,
                label=f"{source} {label}",
            )
    ax_state.set_ylim(-0.03, 1.03)
    ax_state.set_xticks([0, 1, 2, 3])
    ax_state.set_title("B. Response is carried by a state-composition shift")
    ax_state.set_xlabel("treatment day")
    ax_state.set_ylabel("cell fraction")
    ax_state.legend(frameon=False, fontsize=8, loc="center right")

    # C. Module trajectories in expected-direction coordinates.
    modules = [
        "tumor_suppressor_reactivation",
        "oncogene_proliferation_repression",
        "immune_surface_interferon_like",
        "polyamine_mito_adaptive_stress",
        "proteostasis_iron_reduction",
        "malat1_caution",
    ]
    colors = {
        "tumor_suppressor_reactivation": "#31572C",
        "oncogene_proliferation_repression": "#B23A48",
        "immune_surface_interferon_like": "#146C94",
        "polyamine_mito_adaptive_stress": "#7A4EAB",
        "proteostasis_iron_reduction": "#6E6259",
        "malat1_caution": "#9AA0A6",
    }
    short_labels = {
        "tumor_suppressor_reactivation": "DKK1/GAS5",
        "oncogene_proliferation_repression": "EREG/PCNA",
        "immune_surface_interferon_like": "B2M/HLA",
        "polyamine_mito_adaptive_stress": "SAT1/LGALS1/MT",
        "proteostasis_iron_reduction": "HSP/FTL",
        "malat1_caution": "MALAT1",
    }
    label_y = {
        "tumor_suppressor_reactivation": 0.20,
        "oncogene_proliferation_repression": 0.38,
        "immune_surface_interferon_like": 0.56,
        "polyamine_mito_adaptive_stress": 1.02,
        "proteostasis_iron_reduction": 0.78,
        "malat1_caution": -0.46,
    }
    for module in modules:
        sub = module_day[module_day["module"] == module].sort_values("time")
        if sub.empty:
            continue
        sign = -1.0 if str(sub["expected"].iloc[0]) == "down" else 1.0
        y = sign * sub["delta_vs_day0"].to_numpy(dtype=float)
        yerr = sub["sem"].to_numpy(dtype=float)
        label = str(sub["label"].iloc[0])
        ax_modules.errorbar(
            sub["time"],
            y,
            yerr=yerr,
            marker="o",
            lw=2.0,
            ms=4.5,
            capsize=2.2,
            color=colors.get(module, "#555555"),
            alpha=0.94 if module != "malat1_caution" else 0.72,
        )
        ax_modules.text(
            3.04,
            label_y.get(module, y[-1]),
            short_labels.get(module, label),
            va="center",
            ha="left",
            fontsize=8,
            color=colors.get(module, "#555555"),
        )
        if abs(label_y.get(module, y[-1]) - y[-1]) > 0.025:
            ax_modules.plot([3.0, 3.035], [y[-1], label_y[module]], color=colors.get(module, "#555555"), lw=0.8, alpha=0.65)
    ax_modules.axhline(0, color="#333333", lw=0.8)
    ax_modules.set_xticks([0, 1, 2, 3])
    ax_modules.set_xlim(-0.15, 4.10)
    ax_modules.set_title("C. Module trajectories support specific drug-response axes")
    ax_modules.set_xlabel("treatment day")
    ax_modules.set_ylabel("signed change from day0\n(positive = expected direction)")

    # D. Paired dot evidence: temporal support and late-state carrying.
    support_rows = []
    for module in modules:
        rsub = summary[summary["module"] == module]
        if rsub.empty:
            continue
        r = rsub.iloc[0]
        sign = -1.0 if str(r["expected"]) == "down" else 1.0
        support_rows.append(
            {
                "module": module,
                "label": str(r["label"]),
                "day_support": sign * float(r["day3_minus_day0"]),
                "state_support": sign * float(r["day3_late_minus_day0_early"]),
            }
        )
    support_df = pd.DataFrame(support_rows)
    support_df["order"] = support_df["module"].map({m: i for i, m in enumerate(modules)})
    support_df = support_df.sort_values("order", ascending=False)
    y_pos = np.arange(len(support_df))
    for y, (_, row) in zip(y_pos, support_df.iterrows()):
        color = colors.get(row["module"], "#555555")
        ax_support.plot([row["day_support"], row["state_support"]], [y, y], color=color, lw=1.5, alpha=0.45)
        ax_support.scatter(row["day_support"], y, s=58, color=color, marker="o", label="day3-day0" if y == 0 else None, zorder=3)
        ax_support.scatter(row["state_support"], y, s=58, facecolor="white", edgecolor=color, marker="s", linewidth=1.7, label="late state - day0 early" if y == 0 else None, zorder=4)
    ax_support.axvline(0, color="#333333", lw=0.9)
    ax_support.set_yticks(y_pos)
    ax_support.set_yticklabels(support_df["label"], fontsize=8)
    ax_support.set_xlabel("signed support\n(positive = supports expected direction)")
    ax_support.set_title("D. Temporal and state-specific support are both measured")
    ax_support.legend(frameon=False, fontsize=8, loc="lower right")
    ax_support.grid(axis="x", color="#D9D9D9", lw=0.8)

    fig.suptitle("Evidence for dynamic HCT116 5-AZA-CdR response resolved by CellCompass", y=0.995, fontsize=15, fontweight="bold")
    savefig(FIG / "story_drug_response_evidence_panel.png")


def plot_story_pulsebridge_unique_insight_panel(
    state_df: pd.DataFrame,
    state_response: pd.DataFrame,
    baseline: pd.DataFrame,
) -> None:
    """Evidence for what PulseBridge adds beyond observed-day expression analysis."""
    time_points, traj = load_all_trajectory_points()
    traj_arr = np.stack(traj, axis=0)
    step_lengths = np.linalg.norm(np.diff(traj_arr, axis=0), axis=2)
    path_len = step_lengths.sum(axis=0)
    endpoint_len = np.linalg.norm(traj_arr[-1] - traj_arr[0], axis=1)
    ratio = path_len / np.maximum(endpoint_len, 1e-8)

    fig = plt.figure(figsize=(14.6, 10.0))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.32)
    ax_path = fig.add_subplot(gs[0, 0])
    ax_river = fig.add_subplot(gs[0, 1])
    ax_velocity = fig.add_subplot(gs[1, 0])
    ax_pcie = fig.add_subplot(gs[1, 1])

    rng = np.random.default_rng(41)
    sample = rng.choice(np.arange(path_len.size), size=min(350, path_len.size), replace=False)
    for idx in sample:
        ax_path.plot([0, 1], [endpoint_len[idx], path_len[idx]], color="#B8BEC6", lw=0.45, alpha=0.20, zorder=1)
    parts = ax_path.violinplot([endpoint_len, path_len], positions=[0, 1], widths=0.42, showmeans=False, showextrema=False)
    for body, color in zip(parts["bodies"], ["#9AA6B2", "#146C94"]):
        body.set_facecolor(color)
        body.set_edgecolor("none")
        body.set_alpha(0.55)
    ax_path.scatter([0, 1], [endpoint_len.mean(), path_len.mean()], s=85, color=["#555555", "#146C94"], zorder=4)
    ax_path.plot([0, 1], [endpoint_len.mean(), path_len.mean()], color="#333333", lw=1.8, zorder=3)
    ax_path.set_xticks([0, 1])
    ax_path.set_xticklabels(["endpoint\nshortcut", "PulseBridge\nrollout path"])
    ax_path.set_ylabel("latent distance per generated particle")
    ax_path.set_title("A. PulseBridge estimates a route, not only an endpoint")
    ax_path.text(
        0.5,
        0.95,
        f"mean path / endpoint = {ratio.mean():.2f}x",
        transform=ax_path.transAxes,
        ha="center",
        va="top",
        fontsize=11,
        fontweight="bold",
        color="#146C94",
    )

    river = state_df[state_df["source"] == "generated"].copy()
    pivot = (
        river.pivot(index="time", columns="state", values="fraction")
        .fillna(0.0)
        .sort_index()
    )
    state_names = [c for c in [f"state_{i}" for i in range(1, 7)] if c in pivot.columns]
    colors = sns.color_palette("Spectral", n_colors=len(state_names))
    ax_river.stackplot(
        pivot.index.to_numpy(dtype=float),
        [pivot[s].to_numpy(dtype=float) for s in state_names],
        colors=colors,
        alpha=0.92,
        labels=state_names,
    )
    obs = state_df[state_df["source"] == "observed"].copy()
    for t, sub in obs.groupby("time"):
        sub = sub.set_index("state").reindex(state_names).fillna(0.0)
        centers = sub["fraction"].cumsum() - 0.5 * sub["fraction"]
        ax_river.scatter(
            np.full(len(state_names), float(t)),
            centers,
            marker="_",
            s=78,
            color="#111111",
            linewidths=1.3,
            zorder=5,
        )
    ax_river.set_xlim(float(pivot.index.min()), float(pivot.index.max()))
    ax_river.set_ylim(0, 1)
    ax_river.set_xlabel("generated continuous time")
    ax_river.set_ylabel("generated state fraction")
    ax_river.set_title("B. The rollout resolves continuous state occupancy")
    ax_river.text(
        0.02,
        0.98,
        "black ticks: observed day compositions",
        transform=ax_river.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color="#333333",
    )
    ax_river.legend(frameon=False, fontsize=7, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.25))

    phase_order = ["initial_0_to_0.5", "transition_0.5_to_1", "middle_1_to_2", "late_2_to_3"]
    phase_labels = ["0-0.5", "0.5-1", "1-2", "2-3"]
    groups = {
        "early states 1-2": ["state_1", "state_2"],
        "middle states 3-4": ["state_3", "state_4"],
        "late states 5-6": ["state_5", "state_6"],
    }
    group_colors = {
        "early states 1-2": "#31572C",
        "middle states 3-4": "#4F86C6",
        "late states 5-6": "#B23A48",
    }
    for group, states in groups.items():
        rows = []
        for phase in phase_order:
            sub = state_response[(state_response["phase"] == phase) & (state_response["state"].isin(states))]
            if sub.empty:
                rows.append({"phase": phase, "velocity": np.nan, "forward": np.nan})
                continue
            weights = sub["mean_speed"].to_numpy(dtype=float)
            weights = weights / weights.sum() if weights.sum() > 0 else None
            rows.append(
                {
                    "phase": phase,
                    "velocity": float(np.average(sub["mean_axis_velocity"], weights=weights)),
                    "forward": float(np.average(sub["forward_fraction"], weights=weights)),
                }
            )
        df = pd.DataFrame(rows)
        x = np.arange(len(phase_order))
        sizes = 35 + 110 * df["forward"].fillna(0.0).to_numpy(dtype=float)
        ax_velocity.plot(x, df["velocity"], color=group_colors[group], lw=2.1, alpha=0.92)
        ax_velocity.scatter(x, df["velocity"], s=sizes, color=group_colors[group], edgecolor="white", linewidth=0.8, label=group, zorder=3)
    ax_velocity.axhline(0, color="#333333", lw=0.9)
    ax_velocity.set_xticks(np.arange(len(phase_order)))
    ax_velocity.set_xticklabels(phase_labels)
    ax_velocity.set_xlabel("generated phase")
    ax_velocity.set_ylabel("generated response-axis velocity")
    ax_velocity.set_title("C. PulseBridge localizes when each state group is pushed")
    ax_velocity.legend(frameon=False, fontsize=8, loc="lower left")
    ax_velocity.text(
        0.99,
        0.05,
        "dot size: forward fraction",
        transform=ax_velocity.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#333333",
    )

    pcie = baseline.dropna(subset=["pcie"]).sort_values("pcie").head(8).copy()
    colors = ["#146C94" if alg == "PulseBridge" else "#9AA6B2" for alg in pcie["algorithm"]]
    y = np.arange(len(pcie))
    ax_pcie.barh(y, pcie["pcie"], color=colors, alpha=0.92)
    ax_pcie.set_yticks(y)
    ax_pcie.set_yticklabels(pcie["algorithm"], fontsize=8)
    ax_pcie.invert_yaxis()
    ax_pcie.set_xlabel("PCIE, lower is better")
    ax_pcie.set_title("D. Heldout old/new RNA direction favors PulseBridge")
    pulse = float(pcie.loc[pcie["algorithm"] == "PulseBridge", "pcie"].iloc[0])
    next_best = float(pcie.loc[pcie["algorithm"] != "PulseBridge", "pcie"].iloc[0])
    ax_pcie.text(
        0.98,
        0.08,
        f"PulseBridge {pulse:.3f}\nnext best {next_best:.3f}",
        transform=ax_pcie.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#333333",
    )

    fig.suptitle("PulseBridge-specific evidence beyond observed-day expression analysis", y=0.995, fontsize=15, fontweight="bold")
    savefig(FIG / "story_pulsebridge_unique_insight_panel.png")


def pulsebridge_rollout_biology_timing_analysis(adata) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Project measured biology onto the continuous PulseBridge rollout through latent neighbors."""
    module_defs = [
        {
            "module": "tumor_suppressor_reactivation",
            "label": "DKK1/GAS5 reactivation",
            "genes": ["DKK1", "GAS5"],
            "expected": "up",
            "class": "desired",
        },
        {
            "module": "oncogene_proliferation_repression",
            "label": "EREG/PCNA repression",
            "genes": ["EREG", "PCNA"],
            "expected": "down",
            "class": "desired",
        },
        {
            "module": "antigen_presentation_surface",
            "label": "B2M/HLA antigen presentation",
            "genes": ["B2M", "HLA-A", "HLA-B", "HLA-C"],
            "expected": "up",
            "class": "immune",
        },
        {
            "module": "ifitm3_antiviral_membrane",
            "label": "IFITM3 antiviral membrane",
            "genes": ["IFITM3"],
            "expected": "up",
            "class": "immune",
        },
        {
            "module": "sat1_polyamine_catabolism",
            "label": "SAT1 polyamine catabolism",
            "genes": ["SAT1"],
            "expected": "up",
            "class": "metabolic",
        },
        {
            "module": "lgals1_adhesion_escape",
            "label": "LGALS1 adhesion/escape candidate",
            "genes": ["LGALS1"],
            "expected": "up",
            "class": "adaptive",
        },
        {
            "module": "mitochondrial_remodeling",
            "label": "mitochondrial remodeling",
            "genes": ["MT-CO1", "MT-CO2", "MT-CO3", "MT-ND4", "MT-CYB"],
            "expected": "up",
            "class": "metabolic",
        },
        {
            "module": "proteostasis_iron_reduction",
            "label": "HSP/FTL proteostasis/iron reduction",
            "genes": ["HSP90AB1", "HSP90AA1", "HSPD1", "FTL"],
            "expected": "down",
            "class": "remodeling",
        },
    ]
    layer = adata.layers["log1p_norm"] if "log1p_norm" in adata.layers else adata.X
    x_latent = np.asarray(adata.obsm["X_latent"])
    observed_scores = {}
    valid_defs = []
    for module in module_defs:
        genes = [g for g in module["genes"] if g in adata.var_names]
        if not genes:
            continue
        idx = [int(adata.var_names.get_loc(g)) for g in genes]
        score = dense_slice(layer[:, idx]).mean(axis=1)
        observed_scores[module["module"]] = np.asarray(score, dtype=float)
        module = dict(module)
        module["genes"] = genes
        valid_defs.append(module)

    nn = NearestNeighbors(n_neighbors=15, metric="euclidean")
    nn.fit(x_latent)
    time_points, traj = load_all_trajectory_points()
    baseline_neighbors = nn.kneighbors(traj[0], return_distance=False)

    baseline = {}
    for module in valid_defs:
        scores = observed_scores[module["module"]][baseline_neighbors].mean(axis=1)
        baseline[module["module"]] = float(scores.mean())

    profile_rows = []
    for t, pts in zip(time_points, traj):
        neigh = nn.kneighbors(pts, return_distance=False)
        for module in valid_defs:
            particle_scores = observed_scores[module["module"]][neigh].mean(axis=1)
            mean_score = float(particle_scores.mean())
            sem = float(particle_scores.std() / np.sqrt(max(particle_scores.size, 1)))
            raw_delta = mean_score - baseline[module["module"]]
            sign = -1.0 if module["expected"] == "down" else 1.0
            profile_rows.append(
                {
                    "time": float(t),
                    "module": module["module"],
                    "label": module["label"],
                    "class": module["class"],
                    "genes": ";".join(module["genes"]),
                    "expected": module["expected"],
                    "mean_neighbor_score": mean_score,
                    "sem": sem,
                    "raw_delta_from_t0": raw_delta,
                    "signed_delta_from_t0": sign * raw_delta,
                }
            )
    profile = pd.DataFrame(profile_rows)
    profile.to_csv(TAB / "pulsebridge_rollout_neighbor_biology_profile.csv", index=False)

    timing_rows = []
    for module, sub in profile.groupby("module"):
        sub = sub.sort_values("time").copy()
        label = str(sub["label"].iloc[0])
        module_class = str(sub["class"].iloc[0])
        expected = str(sub["expected"].iloc[0])
        signed = sub["signed_delta_from_t0"].to_numpy(dtype=float)
        times_arr = sub["time"].to_numpy(dtype=float)
        max_idx = int(np.nanargmax(signed))
        max_value = float(signed[max_idx])
        peak_time = float(times_arr[max_idx])
        half = 0.5 * max_value
        crossing = times_arr[signed >= half]
        first_half_time = float(crossing[0]) if crossing.size and max_value > 0 else np.nan
        end_value = float(signed[-1])
        timing_rows.append(
            {
                "module": module,
                "label": label,
                "class": module_class,
                "expected": expected,
                "peak_time": peak_time,
                "peak_signed_delta": max_value,
                "first_half_peak_time": first_half_time,
                "end_signed_delta": end_value,
                "retained_fraction_at_day3": end_value / max_value if abs(max_value) > 1e-8 else np.nan,
            }
        )
    timing = pd.DataFrame(timing_rows)
    timing.to_csv(TAB / "pulsebridge_rollout_biology_event_timing.csv", index=False)
    return profile, timing


def plot_story_unique_biology_insight_panel(
    state_df: pd.DataFrame,
    baseline: pd.DataFrame,
    rollout_profile: pd.DataFrame,
    rollout_timing: pd.DataFrame,
) -> None:
    """Plot the unique biological insight from the PulseBridge continuous rollout."""
    fig = plt.figure(figsize=(14.8, 10.2))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.34)
    ax_profile = fig.add_subplot(gs[0, 0])
    ax_events = fig.add_subplot(gs[0, 1])
    ax_state = fig.add_subplot(gs[1, 0])
    ax_pcie = fig.add_subplot(gs[1, 1])

    module_order = [
        "tumor_suppressor_reactivation",
        "oncogene_proliferation_repression",
        "antigen_presentation_surface",
        "ifitm3_antiviral_membrane",
        "sat1_polyamine_catabolism",
        "lgals1_adhesion_escape",
        "mitochondrial_remodeling",
        "proteostasis_iron_reduction",
    ]
    colors = {
        "tumor_suppressor_reactivation": "#31572C",
        "oncogene_proliferation_repression": "#B23A48",
        "antigen_presentation_surface": "#146C94",
        "ifitm3_antiviral_membrane": "#1B8A8F",
        "sat1_polyamine_catabolism": "#7A4EAB",
        "lgals1_adhesion_escape": "#C17C0E",
        "mitochondrial_remodeling": "#8F6A56",
        "proteostasis_iron_reduction": "#6E6259",
    }
    labels = {
        "tumor_suppressor_reactivation": "DKK1/GAS5",
        "oncogene_proliferation_repression": "EREG/PCNA",
        "antigen_presentation_surface": "B2M/HLA",
        "ifitm3_antiviral_membrane": "IFITM3",
        "sat1_polyamine_catabolism": "SAT1",
        "lgals1_adhesion_escape": "LGALS1",
        "mitochondrial_remodeling": "mitochondrial",
        "proteostasis_iron_reduction": "HSP/FTL",
    }
    profile_label_y = {
        "tumor_suppressor_reactivation": 0.22,
        "antigen_presentation_surface": 0.12,
        "ifitm3_antiviral_membrane": 0.34,
        "sat1_polyamine_catabolism": 0.72,
        "lgals1_adhesion_escape": 0.56,
        "mitochondrial_remodeling": 0.46,
    }

    for module in module_order:
        sub = rollout_profile[rollout_profile["module"] == module].sort_values("time")
        if sub.empty:
            continue
        ax_profile.plot(sub["time"], sub["signed_delta_from_t0"], lw=2.0, color=colors[module], alpha=0.94)
        ax_profile.fill_between(
            sub["time"],
            sub["signed_delta_from_t0"] - sub["sem"],
            sub["signed_delta_from_t0"] + sub["sem"],
            color=colors[module],
            alpha=0.10,
            linewidth=0,
        )
        ax_profile.text(
            float(sub["time"].iloc[-1]) + 0.04,
            profile_label_y.get(module, float(sub["signed_delta_from_t0"].iloc[-1])),
            labels[module],
            va="center",
            ha="left",
            fontsize=8,
            color=colors[module],
        )
        if module in profile_label_y:
            ax_profile.plot(
                [float(sub["time"].iloc[-1]), float(sub["time"].iloc[-1]) + 0.035],
                [float(sub["signed_delta_from_t0"].iloc[-1]), profile_label_y[module]],
                color=colors[module],
                lw=0.7,
                alpha=0.7,
            )
    ax_profile.axhline(0, color="#333333", lw=0.8)
    ax_profile.set_xlim(-0.05, 3.62)
    ax_profile.set_xlabel("PulseBridge generated time")
    ax_profile.set_ylabel("latent-neighbor biology score\nsigned to expected direction")
    ax_profile.set_title("A. PulseBridge orders biological programs along a continuous route")

    timing = rollout_timing.set_index("module").reindex(module_order).dropna(subset=["peak_time"], how="all").reset_index()
    y = np.arange(len(timing))
    for yi, (_, row) in enumerate(timing.iterrows()):
        module = row["module"]
        color = colors.get(module, "#555555")
        ax_events.plot([row["first_half_peak_time"], row["peak_time"]], [yi, yi], color=color, lw=2.2, alpha=0.75)
        ax_events.scatter(row["first_half_peak_time"], yi, s=55, facecolor="white", edgecolor=color, linewidth=1.8, marker="o", zorder=3)
        ax_events.scatter(row["peak_time"], yi, s=70, color=color, marker="D", zorder=4)
        ax_events.text(
            min(3.04, float(row["peak_time"]) + 0.06),
            yi,
            f"peak {row['peak_time']:.1f}; retained {row['retained_fraction_at_day3']:.0%}",
            va="center",
            ha="left",
            fontsize=7.8,
            color="#333333",
        )
    ax_events.set_yticks(y)
    ax_events.set_yticklabels([labels.get(m, m) for m in timing["module"]], fontsize=8)
    ax_events.set_xlim(-0.05, 3.55)
    ax_events.set_xlabel("PulseBridge generated time")
    ax_events.set_title("B. Event timing separates early response, immune remodeling, and adaptive stress")
    ax_events.grid(axis="x", color="#D9D9D9", lw=0.8)
    ax_events.invert_yaxis()

    generated = state_df[state_df["source"] == "generated"].copy()
    state_groups = {
        "early states 1-2": ["state_1", "state_2"],
        "late states 5-6": ["state_5", "state_6"],
    }
    for label, states in state_groups.items():
        rows = []
        for t, sub in generated.groupby("time"):
            rows.append({"time": float(t), "fraction": float(sub[sub["state"].isin(states)]["fraction"].sum())})
        rows = pd.DataFrame(rows).sort_values("time")
        color = "#31572C" if "early" in label else "#B23A48"
        ax_state.plot(rows["time"], rows["fraction"], lw=2.6, color=color, label=label)
    for module in ["tumor_suppressor_reactivation", "sat1_polyamine_catabolism", "antigen_presentation_surface", "lgals1_adhesion_escape"]:
        t = float(rollout_timing.loc[rollout_timing["module"] == module, "peak_time"].iloc[0])
        ax_state.axvline(t, color=colors[module], lw=1.4, ls=":", alpha=0.8)
        ax_state.text(t + 0.03, 0.96, labels[module], rotation=90, va="top", ha="left", fontsize=7.5, color=colors[module])
    ax_state.set_ylim(-0.03, 1.03)
    ax_state.set_xlim(-0.05, 3.05)
    ax_state.set_xlabel("PulseBridge generated time")
    ax_state.set_ylabel("generated state fraction")
    ax_state.set_title("C. Biological events are placed on the generated state transition")
    ax_state.legend(frameon=False, fontsize=8, loc="center right")

    pcie = baseline.dropna(subset=["pcie"]).sort_values("pcie").head(8).copy()
    bar_colors = ["#146C94" if alg == "PulseBridge" else "#9AA6B2" for alg in pcie["algorithm"]]
    yy = np.arange(len(pcie))
    ax_pcie.barh(yy, pcie["pcie"], color=bar_colors, alpha=0.92)
    ax_pcie.set_yticks(yy)
    ax_pcie.set_yticklabels(pcie["algorithm"], fontsize=8)
    ax_pcie.invert_yaxis()
    ax_pcie.set_xlabel("PCIE, lower is better")
    ax_pcie.set_title("D. Metrics support using this PulseBridge route")
    pulse = float(pcie.loc[pcie["algorithm"] == "PulseBridge", "pcie"].iloc[0])
    next_best = float(pcie.loc[pcie["algorithm"] != "PulseBridge", "pcie"].iloc[0])
    ax_pcie.text(
        0.98,
        0.08,
        f"PulseBridge {pulse:.3f}\nnext best {next_best:.3f}",
        transform=ax_pcie.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#333333",
    )

    fig.suptitle("PulseBridge-specific biology: a validated route reveals timing and state placement of drug-response programs", y=0.995, fontsize=14.5, fontweight="bold")
    savefig(FIG / "story_unique_biology_insight_panel.png")


def plot_story_source_aligned_pulsebridge_map(
    state_df: pd.DataFrame,
    baseline: pd.DataFrame,
    rollout_profile: pd.DataFrame,
    rollout_timing: pd.DataFrame,
    targeted_support: pd.DataFrame,
) -> None:
    """Compact source-paper biology map focused on PulseBridge-specific added value."""
    fig = plt.figure(figsize=(14.8, 10.0))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.36)
    ax_support = fig.add_subplot(gs[0, 0])
    ax_events = fig.add_subplot(gs[0, 1])
    ax_arms = fig.add_subplot(gs[1, 0])
    ax_state = fig.add_subplot(gs[1, 1])

    colors = {
        "tumor_suppressor_reactivation": "#31572C",
        "oncogene_proliferation_repression": "#B23A48",
        "immune_surface_interferon_like": "#146C94",
        "antigen_presentation_surface": "#146C94",
        "ifitm3_antiviral_membrane": "#1B8A8F",
        "sat1_polyamine_catabolism": "#7A4EAB",
        "lgals1_adhesion_escape": "#C17C0E",
        "mitochondrial_remodeling": "#8F6A56",
        "proteostasis_iron_reduction": "#6E6259",
    }

    support_order = [
        "tumor_suppressor_reactivation",
        "oncogene_proliferation_repression",
        "immune_surface_interferon_like",
        "polyamine_mito_adaptive_stress",
        "proteostasis_iron_reduction",
    ]
    support_labels = {
        "tumor_suppressor_reactivation": "DKK1/GAS5\nTSG reactivation",
        "oncogene_proliferation_repression": "EREG/PCNA\noncogene/proliferation down",
        "immune_surface_interferon_like": "B2M/HLA/IFITM3\nviral-mimicry-like surface",
        "polyamine_mito_adaptive_stress": "SAT1/LGALS1/MT\nadaptive stress candidate",
        "proteostasis_iron_reduction": "HSP/FTL\nremodeling down",
    }
    support_color = {
        "tumor_suppressor_reactivation": "#31572C",
        "oncogene_proliferation_repression": "#B23A48",
        "immune_surface_interferon_like": "#146C94",
        "polyamine_mito_adaptive_stress": "#7A4EAB",
        "proteostasis_iron_reduction": "#6E6259",
    }
    support = targeted_support[targeted_support["module"].isin(support_order)].copy()
    support["order"] = support["module"].map({m: i for i, m in enumerate(support_order)})
    support = support.sort_values("order", ascending=False)
    y_pos = np.arange(len(support))
    for y, (_, row) in zip(y_pos, support.iterrows()):
        module = str(row["module"])
        sign = -1.0 if str(row["expected"]) == "down" else 1.0
        day_support = sign * float(row["day3_minus_day0"])
        state_support = sign * float(row["day3_late_minus_day0_early"])
        color = support_color[module]
        ax_support.plot([day_support, state_support], [y, y], color=color, lw=1.8, alpha=0.46)
        ax_support.scatter(day_support, y, s=68, color=color, marker="o", label="day3-day0" if y == 0 else None, zorder=3)
        ax_support.scatter(
            state_support,
            y,
            s=72,
            facecolor="white",
            edgecolor=color,
            marker="s",
            linewidth=1.8,
            label="late state - early state" if y == 0 else None,
            zorder=4,
        )
    ax_support.axvline(0, color="#333333", lw=0.9)
    ax_support.set_yticks(y_pos)
    ax_support.set_yticklabels([support_labels[m] for m in support["module"]], fontsize=8.4)
    ax_support.set_xlabel("signed support, positive = supports the axis")
    ax_support.set_title("A. Article biology axes are anchored before interpretation")
    ax_support.grid(axis="x", color="#D9D9D9", lw=0.8)
    ax_support.legend(frameon=False, fontsize=8, loc="lower right")

    event_order = [
        "tumor_suppressor_reactivation",
        "ifitm3_antiviral_membrane",
        "sat1_polyamine_catabolism",
        "proteostasis_iron_reduction",
        "mitochondrial_remodeling",
        "antigen_presentation_surface",
        "lgals1_adhesion_escape",
        "oncogene_proliferation_repression",
    ]
    event_labels = {
        "tumor_suppressor_reactivation": "DKK1/GAS5\nreactivation",
        "ifitm3_antiviral_membrane": "IFITM3\nantiviral",
        "sat1_polyamine_catabolism": "SAT1\npolyamine",
        "proteostasis_iron_reduction": "HSP/FTL\nreduction",
        "mitochondrial_remodeling": "mitochondrial\nremodeling",
        "antigen_presentation_surface": "B2M/HLA\nsurface",
        "lgals1_adhesion_escape": "LGALS1\nadaptive",
        "oncogene_proliferation_repression": "EREG/PCNA\nrepression",
    }
    timing = rollout_timing.set_index("module").reindex(event_order).dropna(subset=["peak_time"], how="all").reset_index()
    event_y = np.arange(len(timing))
    for yi, (_, row) in enumerate(timing.iterrows()):
        module = str(row["module"])
        color = colors.get(module, "#555555")
        half_time = float(row["first_half_peak_time"])
        peak_time = float(row["peak_time"])
        ax_events.plot([half_time, peak_time], [yi, yi], color=color, lw=2.3, alpha=0.78)
        ax_events.scatter(half_time, yi, s=58, facecolor="white", edgecolor=color, linewidth=1.8, marker="o", zorder=3)
        ax_events.scatter(peak_time, yi, s=76, color=color, marker="D", zorder=4)
        ax_events.text(
            3.18,
            yi,
            f"peak {peak_time:.1f}; kept {float(row['retained_fraction_at_day3']):.0%}",
            va="center",
            ha="left",
            fontsize=7.8,
            color="#333333",
        )
    ax_events.set_yticks(event_y)
    ax_events.set_yticklabels([event_labels.get(m, m) for m in timing["module"]], fontsize=8.2)
    ax_events.set_xlim(-0.05, 4.12)
    ax_events.set_xlabel("PulseBridge generated time")
    ax_events.set_title("B. PulseBridge turns endpoint markers into an event order")
    ax_events.grid(axis="x", color="#D9D9D9", lw=0.8)
    ax_events.invert_yaxis()

    arm_defs = {
        "reactivation/repression": ["tumor_suppressor_reactivation", "oncogene_proliferation_repression", "proteostasis_iron_reduction"],
        "viral-mimicry-like surface": ["antigen_presentation_surface", "ifitm3_antiviral_membrane"],
        "adaptive/metabolic caution": ["sat1_polyamine_catabolism", "lgals1_adhesion_escape", "mitochondrial_remodeling"],
    }
    arm_colors = {
        "reactivation/repression": "#31572C",
        "viral-mimicry-like surface": "#146C94",
        "adaptive/metabolic caution": "#7A4EAB",
    }
    peak_map = rollout_timing.set_index("module")["peak_signed_delta"].to_dict()
    arm_rows = []
    for arm, modules in arm_defs.items():
        parts = []
        for module in modules:
            sub = rollout_profile[rollout_profile["module"] == module].sort_values("time")
            if sub.empty:
                continue
            peak = max(abs(float(peak_map.get(module, 1.0))), 1e-8)
            tmp = sub[["time", "signed_delta_from_t0"]].copy()
            tmp["norm"] = tmp["signed_delta_from_t0"] / peak
            tmp["module"] = module
            parts.append(tmp)
        if not parts:
            continue
        merged = pd.concat(parts, ignore_index=True)
        mean = merged.groupby("time", as_index=False)["norm"].mean()
        mean["arm"] = arm
        arm_rows.append(mean)
    arms = pd.concat(arm_rows, ignore_index=True)
    arm_label_y = {
        "reactivation/repression": 0.82,
        "adaptive/metabolic caution": 0.75,
        "viral-mimicry-like surface": 0.52,
    }
    for arm, sub in arms.groupby("arm"):
        sub = sub.sort_values("time")
        ax_arms.plot(sub["time"], sub["norm"], lw=2.6, color=arm_colors[arm], label=arm)
        label_y = arm_label_y.get(arm, float(sub["norm"].iloc[-1]))
        ax_arms.text(float(sub["time"].iloc[-1]) + 0.05, label_y, arm, ha="left", va="center", fontsize=8.2, color=arm_colors[arm])
        if abs(label_y - float(sub["norm"].iloc[-1])) > 0.03:
            ax_arms.plot(
                [float(sub["time"].iloc[-1]), float(sub["time"].iloc[-1]) + 0.04],
                [float(sub["norm"].iloc[-1]), label_y],
                color=arm_colors[arm],
                lw=0.8,
                alpha=0.7,
            )
    ax_arms.axhline(0, color="#333333", lw=0.8)
    ax_arms.set_xlim(-0.05, 3.82)
    ax_arms.set_ylim(-0.12, 1.12)
    ax_arms.set_xlabel("PulseBridge generated time")
    ax_arms.set_ylabel("mean normalized trajectory signal")
    ax_arms.set_title("C. The route separates desired and cautionary response arms")

    generated = state_df[state_df["source"] == "generated"].copy()
    state_groups = {
        "early states 1-2": ["state_1", "state_2"],
        "late states 5-6": ["state_5", "state_6"],
    }
    for label, states in state_groups.items():
        rows = []
        for t, sub in generated.groupby("time"):
            rows.append({"time": float(t), "fraction": float(sub[sub["state"].isin(states)]["fraction"].sum())})
        rows = pd.DataFrame(rows).sort_values("time")
        color = "#31572C" if "early" in label else "#B23A48"
        ax_state.plot(rows["time"], rows["fraction"], lw=2.6, color=color, marker="o", ms=4.5, label=label)
    for module in ["tumor_suppressor_reactivation", "sat1_polyamine_catabolism", "antigen_presentation_surface", "lgals1_adhesion_escape"]:
        if module not in set(rollout_timing["module"]):
            continue
        t = float(rollout_timing.loc[rollout_timing["module"] == module, "peak_time"].iloc[0])
        ax_state.axvline(t, color=colors[module], lw=1.3, ls=":", alpha=0.82)
        ax_state.text(t + 0.03, 0.97, event_labels[module].replace("\n", "/"), rotation=90, va="top", ha="left", fontsize=7.2, color=colors[module])
    pcie = baseline.dropna(subset=["pcie"]).sort_values("pcie").head(2).copy()
    pulse = float(pcie.loc[pcie["algorithm"] == "PulseBridge", "pcie"].iloc[0])
    next_best = float(pcie.loc[pcie["algorithm"] != "PulseBridge", "pcie"].iloc[0])
    ax_state.text(
        0.98,
        0.08,
        f"validated route\nPCIE {pulse:.3f} vs {next_best:.3f}",
        transform=ax_state.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        color="#333333",
    )
    ax_state.set_ylim(-0.03, 1.04)
    ax_state.set_xlim(-0.05, 3.05)
    ax_state.set_xlabel("PulseBridge generated time")
    ax_state.set_ylabel("generated state fraction")
    ax_state.set_title("D. Events are placed on the generated state transition")
    ax_state.legend(frameon=False, fontsize=8, loc="center right")

    fig.suptitle(
        "Source-paper biology resolved by PulseBridge as a continuous, directionally validated drug-response route",
        y=0.995,
        fontsize=14.5,
        fontweight="bold",
    )
    savefig(FIG / "story_source_aligned_pulsebridge_biology_map.png")


def plot_story_anchor_pulse_diagnostic(align: pd.DataFrame) -> None:
    """Plot pulse alignment as an anchor-based diagnostic, not a smooth biology curve."""
    anchor_palette = {
        0.0: "#31572C",
        1.0: "#4F86C6",
        2.0: "#F28E2B",
        3.0: "#B23A48",
    }
    df = align.copy()
    colors = [anchor_palette.get(float(a), "#777777") for a in df["nearest_anchor_time"]]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(9.8, 6.2),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]},
    )
    for ax in axes:
        for anchor, sub in df.groupby("nearest_anchor_time"):
            lo = max(0.0, float(sub["time"].min()) - 0.05)
            hi = min(3.0, float(sub["time"].max()) + 0.10)
            ax.axvspan(lo, hi, color=anchor_palette.get(float(anchor), "#AAAAAA"), alpha=0.08, lw=0)
        for switch in [0.5, 1.5, 2.5]:
            ax.axvline(switch, color="#777777", lw=0.8, ls=":", alpha=0.65)

    axes[0].axhline(0, color="#333333", lw=0.8)
    axes[0].vlines(
        df["time"],
        0,
        df["mean_deconfounded_cosine"],
        color=colors,
        lw=1.8,
        alpha=0.82,
    )
    axes[0].scatter(
        df["time"],
        df["mean_deconfounded_cosine"],
        s=18 + 115 * df["frac_positive_cosine"],
        c=colors,
        edgecolor="white",
        linewidth=0.7,
        zorder=3,
    )
    axes[0].set_ylabel("directional cosine")
    axes[0].set_title("Anchor-based heldout pulse diagnostic")
    axes[0].text(
        0.01,
        0.98,
        "Points compare local PulseBridge tangent with the nearest heldout old/new RNA anchor.\n"
        "Background bands show the anchor used; this is not a biological oscillation curve.",
        transform=axes[0].transAxes,
        va="top",
        ha="left",
        fontsize=8.5,
        color="#333333",
    )

    axes[1].bar(
        df["time"],
        df["mean_nuisance_leakage_fraction"],
        width=0.07,
        color=colors,
        alpha=0.72,
    )
    axes[1].axhline(0.5, color="#B23A48", lw=1.0, ls="--", label="nuisance budget")
    axes[1].set_ylabel("nuisance\nleakage")
    axes[1].set_xlabel("generated trajectory time (days)")
    axes[1].set_ylim(0, max(0.55, df["mean_nuisance_leakage_fraction"].max() + 0.08))
    axes[1].legend(frameon=False, fontsize=8, loc="upper right")

    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markeredgecolor="white", label=f"anchor day {anchor:g}")
        for anchor, color in anchor_palette.items()
    ]
    axes[0].legend(handles=handles, frameon=False, ncol=4, fontsize=8, loc="lower left")
    savefig(FIG / "story_anchor_pulse_diagnostic.png")


def plot_story_generated_state_river(axis_info: dict, state_df: pd.DataFrame) -> pd.DataFrame:
    """Show generated state composition as a continuous river along the trajectory."""
    time_points, traj = load_all_trajectory_points()
    pca = axis_info["pca"]
    axis_idx = axis_info["axis_idx"]
    sign = axis_info["axis_sign"]
    edges = axis_info["edges"]
    state_names = axis_info["state_names"]
    rows = []
    for t, pts in zip(time_points, traj):
        scores = sign * pca.transform(pts)[:, axis_idx]
        labels = np.clip(np.digitize(scores, edges[1:-1], right=False), 0, len(state_names) - 1)
        counts = np.bincount(labels, minlength=len(state_names))
        for i, state in enumerate(state_names):
            rows.append(
                {
                    "time": float(t),
                    "state": state,
                    "count": int(counts[i]),
                    "fraction": float(counts[i] / counts.sum()),
                }
            )
    river = pd.DataFrame(rows)
    river.to_csv(TAB / "story_generated_state_river.csv", index=False)

    pivot = (
        river.pivot(index="time", columns="state", values="fraction")
        .fillna(0.0)
        .reindex(columns=state_names)
    )
    colors = sns.color_palette("Spectral", n_colors=len(state_names))
    fig, ax = plt.subplots(figsize=(10.2, 4.8))
    ax.stackplot(pivot.index.to_numpy(), [pivot[s].to_numpy() for s in state_names], colors=colors, alpha=0.92)

    obs = state_df[state_df["source"] == "observed"].copy()
    for t, sub in obs.groupby("time"):
        sub = sub.set_index("state").reindex(state_names).fillna(0.0)
        bottoms = sub["fraction"].cumsum() - sub["fraction"]
        centers = bottoms + 0.5 * sub["fraction"]
        ax.scatter(
            np.full(len(state_names), float(t)),
            centers,
            marker="_",
            s=85,
            color="#111111",
            linewidths=1.4,
            zorder=4,
        )

    cumulative = np.zeros(len(pivot))
    final_x = float(pivot.index.max()) + 0.04
    for color, state in zip(colors, state_names):
        values = pivot[state].to_numpy()
        center = cumulative[-1] + 0.5 * values[-1]
        ax.text(final_x, center, state, va="center", ha="left", fontsize=8, color=color)
        cumulative += values
    ax.set_xlim(float(pivot.index.min()), float(pivot.index.max()) + 0.32)
    ax.set_ylim(0, 1)
    ax.set_xlabel("generated trajectory time (days)")
    ax.set_ylabel("cell-state fraction")
    ax.set_title("PulseBridge moves the population through response-axis states")
    ax.text(
        0.01,
        0.98,
        "Black horizontal ticks mark observed day compositions.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="#333333",
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#DDDDDD", "alpha": 0.9},
    )
    savefig(FIG / "story_generated_state_river.png")
    return river


def plot_story_gene_program_timeline(gene_programs: pd.DataFrame) -> None:
    """Use a line timeline instead of a heatmap for program-level biology."""
    df = gene_programs.copy()
    modules = [
        "polyamine_stress",
        "adhesion_stress",
        "mitochondrial_readout",
        "immune_surface_stress",
        "ribosome_translation",
        "proteostasis_heat_shock",
        "iron_storage",
    ]
    modules = [m for m in modules if m in set(df["module"])]
    palette = dict(zip(modules, sns.color_palette("tab10", n_colors=len(modules))))
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    for module in modules:
        sub = df[df["module"] == module].sort_values("time")
        x = sub["time"].to_numpy()
        y = sub["z_score"].to_numpy()
        ax.plot(x, y, marker="o", lw=2.0, color=palette[module], alpha=0.92)
        ax.annotate(
            "",
            xy=(x[-1], y[-1]),
            xytext=(x[-2], y[-2]),
            arrowprops={"arrowstyle": "->", "lw": 1.5, "color": palette[module]},
        )
        ax.text(x[-1] + 0.05, y[-1], module.replace("_", " "), va="center", fontsize=8, color=palette[module])
    ax.axhline(0, color="#555555", lw=0.8)
    ax.set_xlim(-0.05, 3.95)
    ax.set_xlabel("treatment day")
    ax.set_ylabel("within-program z-score")
    ax.set_title("Gene-program trajectories along the PulseBridge drug-response direction")
    ax.text(
        0.01,
        0.98,
        "Positive-trending programs mark late response direction; negative-trending programs mark baseline/early programs.",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8.5,
        color="#333333",
    )
    savefig(FIG / "story_gene_program_timeline.png")


def plot_story_endpoint_compass(endpoint_dynamic: pd.DataFrame) -> None:
    """A compact visual for why endpoint-only analysis is insufficient."""
    summary = endpoint_dynamic.set_index("quantity")["value"]
    path_length = float(summary["mean_generated_path_length"])
    endpoint = float(summary["mean_generated_endpoint_displacement"])
    ratio = float(summary["mean_path_to_endpoint_ratio"])
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-0.35, 3.65)
    ax.set_ylim(-1.12, 1.52)

    start = np.array([0.0, 0.0])
    end = np.array([3.0, 0.35])
    curve = np.array(
        [
            [0.0, 0.0],
            [0.65, 0.88],
            [1.35, -0.28],
            [2.20, 1.02],
            [3.0, 0.35],
        ]
    )
    ax.plot([start[0], end[0]], [start[1], end[1]], color="#777777", lw=2.2, ls="--")
    ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 2.2, "color": "#777777", "linestyle": "--"})
    ax.plot(curve[:, 0], curve[:, 1], color="#146C94", lw=3.2)
    for p0, p1 in zip(curve[:-1], curve[1:]):
        ax.annotate("", xy=p1, xytext=p0, arrowprops={"arrowstyle": "->", "lw": 2.2, "color": "#146C94"})
    ax.scatter([start[0], end[0]], [start[1], end[1]], s=90, color=["#31572C", "#B23A48"], zorder=4)
    ax.text(start[0] - 0.10, start[1] - 0.22, "day 0", ha="right", va="top", fontsize=10)
    ax.text(end[0] + 0.10, end[1], "day 3", ha="left", va="center", fontsize=10)
    ax.text(1.5, -0.06, f"endpoint displacement: {endpoint:.2f}", ha="center", va="top", color="#555555", fontsize=10)
    ax.text(1.55, 1.22, f"PulseBridge path length: {path_length:.2f}", ha="center", va="bottom", color="#146C94", fontsize=10)
    ax.text(
        1.55,
        -0.92,
        f"route is {ratio:.2f}x longer\nthan endpoint summary",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color="#111111",
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#DDDDDD", "alpha": 0.96},
    )
    ax.text(1.55, 1.46, "Endpoint difference compresses a longer drug-response path", ha="center", va="top", fontsize=14)
    savefig(FIG / "story_endpoint_compass.png")


def plot_story_data_space_trajectory(adata) -> None:
    """Map the generated trajectory onto the observed latent data space."""
    rng = np.random.default_rng(13)
    x_latent = np.asarray(adata.obsm["X_latent"])
    times = adata.obs["time_point_processed"].astype(float).to_numpy()
    proj = PCA(n_components=2, random_state=13)
    obs_2d = proj.fit_transform(x_latent)
    time_points, traj = load_all_trajectory_points()
    gen_centers = np.vstack([proj.transform(pts).mean(axis=0) for pts in traj])

    day_palette = {0.0: "#31572C", 1.0: "#4F86C6", 2.0: "#F28E2B", 3.0: "#B23A48"}
    keep = rng.choice(np.arange(obs_2d.shape[0]), size=min(14000, obs_2d.shape[0]), replace=False)

    fig, ax = plt.subplots(figsize=(8.2, 7.0))
    ax.scatter(obs_2d[keep, 0], obs_2d[keep, 1], s=3, color="#C9D1D9", alpha=0.22, linewidths=0, rasterized=True)
    day_centers = {}
    label_offsets = {
        0.0: (0.14, 0.14),
        1.0: (0.14, -0.15),
        2.0: (-0.58, 0.13),
        3.0: (0.14, 0.10),
    }
    for t in sorted(np.unique(times)):
        mask = times == t
        center = obs_2d[mask].mean(axis=0)
        day_centers[float(t)] = center
        ax.scatter(center[0], center[1], s=140, color=day_palette.get(float(t), "#555555"), edgecolor="white", linewidth=1.2, zorder=4)
        dx, dy = label_offsets.get(float(t), (0.16, 0.0))
        ax.text(center[0] + dx, center[1] + dy, f"day {t:g}", color=day_palette.get(float(t), "#555555"), va="center", fontsize=10, fontweight="bold")

    start = day_centers[min(day_centers)]
    end = day_centers[max(day_centers)]
    ax.plot([start[0], end[0]], [start[1], end[1]], color="#7A7A7A", lw=1.8, ls="--", alpha=0.85, label="endpoint-only shortcut")
    ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 1.8, "color": "#7A7A7A", "linestyle": "--"})

    ax.plot(gen_centers[:, 0], gen_centers[:, 1], color="#0E6F8E", lw=3.2, label="PulseBridge generated path", zorder=5)
    for i in range(0, len(gen_centers) - 1, 4):
        ax.annotate(
            "",
            xy=gen_centers[i + 1],
            xytext=gen_centers[i],
            arrowprops={"arrowstyle": "->", "lw": 2.2, "color": "#0E6F8E"},
            zorder=6,
        )
    scatter = ax.scatter(
        gen_centers[:, 0],
        gen_centers[:, 1],
        c=time_points,
        cmap="viridis",
        s=26,
        edgecolor="white",
        linewidth=0.5,
        zorder=7,
    )
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("generated time")

    ax.set_title("PulseBridge trajectory in the observed HCT116 state space")
    ax.set_xlabel("latent data-space PC1")
    ax.set_ylabel("latent data-space PC2")
    ax.legend(frameon=False, loc="best")
    ax.grid(False)
    focus = np.vstack([gen_centers, np.vstack(list(day_centers.values()))])
    xmin, ymin = focus.min(axis=0)
    xmax, ymax = focus.max(axis=0)
    xpad = max(0.55, 0.42 * (xmax - xmin))
    ypad = max(0.45, 0.75 * (ymax - ymin))
    ax.set_xlim(xmin - xpad, xmax + xpad)
    ax.set_ylim(ymin - ypad, ymax + ypad)
    ax.text(
        0.01,
        0.02,
        "Grey points are observed cells; colored circles are observed day centroids.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        color="#333333",
    )
    savefig(FIG / "story_data_space_trajectory.png")


def plot_story_observed_generated_match(adata, axis_info: dict, state_df: pd.DataFrame) -> pd.DataFrame:
    """Directly check observed-vs-generated agreement at observed days."""
    x_latent = np.asarray(adata.obsm["X_latent"])
    times = adata.obs["time_point_processed"].astype(float).to_numpy()
    axis_values = axis_info["axis_values"]
    axis_vector = axis_info["axis_vector"]
    z = np.load(TRAJ_PATH)
    rows = []
    for t, idx in zip(z["observed_time_points"], z["observed_time_indices"]):
        t = float(t)
        idx = int(idx)
        obs_mask = times == t
        obs = x_latent[obs_mask]
        gen = z[f"points_{idx:06d}"]
        obs_axis = axis_values[obs_mask]
        gen_axis = gen @ axis_vector
        obs_center = obs.mean(axis=0)
        gen_center = gen.mean(axis=0)
        obs_radius = np.linalg.norm(obs - obs_center, axis=1).mean()

        obs_state = (
            state_df[(state_df["source"] == "observed") & (state_df["time"] == t)]
            .set_index("state")["fraction"]
        )
        gen_state = (
            state_df[(state_df["source"] == "generated") & (state_df["time"] == t)]
            .set_index("state")["fraction"]
        )
        states = sorted(set(obs_state.index) | set(gen_state.index), key=lambda s: int(s.split("_")[1]))
        tvd = 0.5 * sum(abs(float(obs_state.get(s, 0.0)) - float(gen_state.get(s, 0.0))) for s in states)

        rows.append(
            {
                "time": t,
                "observed_axis_mean": float(obs_axis.mean()),
                "generated_axis_mean": float(gen_axis.mean()),
                "observed_axis_std": float(obs_axis.std()),
                "generated_axis_std": float(gen_axis.std()),
                "latent_centroid_distance": float(np.linalg.norm(gen_center - obs_center)),
                "observed_latent_radius": float(obs_radius),
                "centroid_distance_over_radius": float(np.linalg.norm(gen_center - obs_center) / max(obs_radius, 1e-8)),
                "state_fraction_tvd": float(tvd),
            }
        )
    match = pd.DataFrame(rows)
    match.to_csv(TAB / "story_observed_generated_match_by_day.csv", index=False)

    day_colors = {0.0: "#31572C", 1.0: "#4F86C6", 2.0: "#F28E2B", 3.0: "#B23A48"}
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    ax = axes[0]
    ax.errorbar(
        match["time"] - 0.035,
        match["observed_axis_mean"],
        yerr=0.18 * match["observed_axis_std"],
        fmt="o",
        color="#111111",
        ecolor="#555555",
        elinewidth=1.2,
        capsize=3,
        label="observed",
    )
    ax.errorbar(
        match["time"] + 0.035,
        match["generated_axis_mean"],
        yerr=0.18 * match["generated_axis_std"],
        fmt="o",
        color="#146C94",
        ecolor="#146C94",
        elinewidth=1.2,
        capsize=3,
        label="generated",
    )
    for _, row in match.iterrows():
        ax.plot(
            [row["time"] - 0.035, row["time"] + 0.035],
            [row["observed_axis_mean"], row["generated_axis_mean"]],
            color="#A0A0A0",
            lw=1.0,
            zorder=0,
        )
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xlabel("treatment day")
    ax.set_ylabel("response-axis coordinate")
    ax.set_title("Observed and generated days align on response axis")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    ax.bar(
        match["time"],
        match["centroid_distance_over_radius"],
        color=[day_colors[float(t)] for t in match["time"]],
        width=0.58,
    )
    ax.axhline(0.1, color="#777777", ls="--", lw=1.0)
    for _, row in match.iterrows():
        ax.text(row["time"], row["centroid_distance_over_radius"] + 0.006, f"{row['centroid_distance_over_radius']:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xlabel("treatment day")
    ax.set_ylabel("centroid error / observed radius")
    ax.set_ylim(0, max(0.13, float(match["centroid_distance_over_radius"].max()) + 0.04))
    ax.set_title("Latent centroid error is small relative to population spread")

    ax = axes[2]
    ax.bar(
        match["time"],
        match["state_fraction_tvd"],
        color=[day_colors[float(t)] for t in match["time"]],
        width=0.58,
    )
    for _, row in match.iterrows():
        ax.text(row["time"], row["state_fraction_tvd"] + 0.004, f"{row['state_fraction_tvd']:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xlabel("treatment day")
    ax.set_ylabel("state-fraction distance")
    ax.set_ylim(0, max(0.08, float(match["state_fraction_tvd"].max()) + 0.03))
    ax.set_title("Observed/generated state composition remains close")

    fig.suptitle("Observed-day checks show generated slices closely track measured time points", y=1.04)
    savefig(FIG / "story_observed_generated_match.png")
    return match


def plot_story_state_occupancy_bubbles(state_df: pd.DataFrame) -> None:
    """Show which response-axis states dominate each day without a heatmap."""
    state_names = sorted(state_df["state"].unique(), key=lambda s: int(s.split("_")[1]))
    state_to_y = {state: i + 1 for i, state in enumerate(state_names)}
    colors = dict(zip(state_names, sns.color_palette("Spectral", n_colors=len(state_names))))

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), sharey=True)
    for ax, source in zip(axes, ["observed", "generated"]):
        sub = state_df[state_df["source"] == source].copy()
        for _, row in sub.iterrows():
            ax.scatter(
                row["time"],
                state_to_y[row["state"]],
                s=35 + 2550 * row["fraction"],
                color=colors[row["state"]],
                alpha=0.82,
                edgecolor="white",
                linewidth=0.8,
            )
        ax.set_title(source.capitalize())
        ax.set_xlabel("treatment day")
        ax.set_xticks([0, 1, 2, 3])
        ax.grid(axis="x", alpha=0.22)
        ax.set_xlim(-0.25, 3.25)
    axes[0].set_yticks(list(state_to_y.values()))
    axes[0].set_yticklabels(state_names)
    axes[0].set_ylabel("response-axis state")
    axes[1].text(
        0.02,
        0.98,
        "Bubble size = cell fraction",
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color="#333333",
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#DDDDDD", "alpha": 0.9},
    )
    fig.suptitle("Drug response shifts cells from early states toward late response states", y=1.03)
    savefig(FIG / "story_state_occupancy_bubbles.png")


def plot_story_gene_program_direction_arrows(gene_programs: pd.DataFrame) -> None:
    """Dumbbell/arrow plot for program changes from day 0 to day 3."""
    df = gene_programs.copy()
    rows = []
    for module, sub in df.groupby("module"):
        sub = sub.sort_values("time")
        if not {0.0, 3.0}.issubset(set(sub["time"].astype(float))):
            continue
        z0 = float(sub.loc[sub["time"] == 0.0, "z_score"].iloc[0])
        z3 = float(sub.loc[sub["time"] == 3.0, "z_score"].iloc[0])
        rows.append({"module": module, "z0": z0, "z3": z3, "delta": z3 - z0})
    arrow_df = pd.DataFrame(rows).sort_values("delta", ascending=True)
    arrow_df.to_csv(TAB / "story_gene_program_day0_to_day3_arrows.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.2, max(4.6, 0.52 * len(arrow_df) + 1.4)))
    y_positions = np.arange(len(arrow_df))
    for y, row in zip(y_positions, arrow_df.itertuples(index=False)):
        color = "#146C94" if row.delta > 0 else "#B23A48"
        ax.annotate(
            "",
            xy=(row.z3, y),
            xytext=(row.z0, y),
            arrowprops={"arrowstyle": "->", "lw": 2.4, "color": color, "shrinkA": 0, "shrinkB": 0},
        )
        ax.scatter(row.z0, y, s=58, color="#FFFFFF", edgecolor=color, linewidth=1.5, zorder=3)
        ax.scatter(row.z3, y, s=72, color=color, edgecolor="white", linewidth=0.8, zorder=4)
        ax.text(
            max(row.z0, row.z3) + 0.08,
            y,
            f"{row.delta:+.2f}",
            va="center",
            ha="left",
            fontsize=8.5,
            color=color,
        )
    ax.axvline(0, color="#555555", lw=0.8)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([m.replace("_", " ") for m in arrow_df["module"]])
    ax.set_xlabel("within-program z-score")
    ax.set_title("Gene programs aligned with the PulseBridge response direction")
    savefig(FIG / "story_gene_program_direction_arrows.png")


def plot_story_directional_validation_scorecard(baseline: pd.DataFrame, align: pd.DataFrame) -> None:
    """Clean scorecard for directional validation without a misleading cosine curve."""
    focus = baseline[baseline["algorithm"].isin(["PulseBridge", "WFR-FM"])].copy()
    focus = focus.sort_values("pcie")
    pulse = focus[focus["algorithm"] == "PulseBridge"].iloc[0]
    wfr = focus[focus["algorithm"] == "WFR-FM"].iloc[0]
    improvement = 100.0 * (1.0 - float(pulse["pcie"]) / float(wfr["pcie"]))
    mean_leak = float(align["mean_nuisance_leakage_fraction"].mean())
    max_leak = float(align["mean_nuisance_leakage_fraction"].max())

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2), gridspec_kw={"width_ratios": [1.1, 1.0]})
    ax = axes[0]
    colors = ["#146C94" if a == "PulseBridge" else "#A5ADB5" for a in focus["algorithm"]]
    ax.bar(focus["algorithm"], focus["pcie"], color=colors, width=0.58)
    ax.set_ylabel("PCIE (lower is better)")
    ax.set_title("Heldout pulse direction agreement")
    for i, (_, row) in enumerate(focus.iterrows()):
        ax.text(i, row["pcie"] + 0.025, f"{row['pcie']:.3f}", ha="center", va="bottom", fontsize=9)
    ax.text(
        0.5,
        0.78,
        f"PulseBridge\n{improvement:.1f}% lower PCIE\nthan WFR-FM",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="#146C94",
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#D9E8EF", "alpha": 0.96},
    )

    ax = axes[1]
    x = np.arange(len(align))
    ax.scatter(x, align["mean_nuisance_leakage_fraction"], s=28, color="#B23A48", alpha=0.75, edgecolor="white", linewidth=0.5)
    ax.axhline(0.5, color="#B23A48", lw=1.2, ls="--", label="budget 0.5")
    ax.axhline(mean_leak, color="#333333", lw=1.3, label=f"mean {mean_leak:.2f}")
    ax.set_ylim(0, 0.58)
    ax.set_xticks([])
    ax.set_ylabel("nuisance leakage fraction")
    ax.set_title("Baseline-labeling leakage stays controlled")
    ax.text(
        0.04,
        0.08,
        f"max {max_leak:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color="#333333",
    )
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("Heldout metabolic-pulse validation and nuisance control", y=1.04)
    savefig(FIG / "story_directional_validation_scorecard.png")


def load_pulsebridge_model_for_downstream(latent_dim: int):
    """Reconstruct the trained PulseBridge velocity model from the synced artifact."""
    cyto_path = str(REPO / "CytoBridge-main")
    if cyto_path not in sys.path:
        sys.path.insert(0, cyto_path)
    import torch
    from CytoBridge.tl.models import DynamicalModel

    payload = torch.load(MODEL_STATE_PATH, map_location="cpu", weights_only=False)
    record = payload["all_model"]
    model = DynamicalModel(int(latent_dim), record["model_config"])
    state = {k: torch.as_tensor(v) for k, v in record["model_state_dict"].items()}
    model.load_state_dict(state)
    model.eval()
    return model, record


def plot_story_scvelo_velocity_stream(adata, max_cells: int = 5000) -> dict:
    """Use the downstream scVelo stream skill as a local flow diagnostic."""
    stream_dir = OUT / "downstream" / "velocity_stream"
    stream_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    artifacts: dict[str, object] = {}
    manifest_path = stream_dir / "manifest.json"

    cyto_path = str(REPO / "CytoBridge-main")
    if cyto_path not in sys.path:
        sys.path.insert(0, cyto_path)
    try:
        import scanpy as sc
        from CytoBridge.tl.downstream import compute_velocity_bundle, build_velocity_graph_bundle
        from CytoBridge.pl.downstream import plot_velocity_stream_bundle
    except Exception as e:
        payload = {
            "analysis": "downstream-scvelo-stream",
            "evidence_type": "local flow diagnostic",
            "artifacts": {},
            "warnings": [f"scVelo/CytoBridge downstream import failed: {e}"],
        }
        manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return payload

    try:
        x_latent = np.asarray(adata.obsm["X_latent"], dtype=np.float32)
        times = adata.obs["time_point_processed"].astype(float).to_numpy()
        keep = sample_balanced_by_time(times, max_cells=max_cells, random_state=31)

        obs = adata.obs.iloc[keep].copy()
        sub = ad.AnnData(X=np.zeros((len(keep), 1), dtype=np.float32), obs=obs)
        sub.var_names = ["placeholder"]
        sub.obsm["X_latent"] = x_latent[keep].copy()
        sub.obs["time_point_processed"] = pd.to_numeric(sub.obs["time_point_processed"], errors="coerce").astype(float)
        sub.obs["treatment_day"] = pd.Categorical(
            [f"day {int(round(float(x)))}" for x in sub.obs["time_point_processed"]],
            categories=["day 0", "day 1", "day 2", "day 3"],
            ordered=True,
        )
        sc.pp.neighbors(sub, n_neighbors=35, use_rep="X_latent", random_state=31)
        sc.tl.umap(sub, random_state=31, min_dist=0.35)
        sub.obsm["X_pca"] = PCA(n_components=2, random_state=31).fit_transform(sub.obsm["X_latent"]).astype(np.float32)

        model, record = load_pulsebridge_model_for_downstream(latent_dim=x_latent.shape[1])
        vel = compute_velocity_bundle(adata=sub, model=model, device="cpu", output_dir=str(stream_dir))
        graph_umap = build_velocity_graph_bundle(
            adata=sub,
            output_dir=str(stream_dir),
            n_neighbors=35,
            n_jobs=1,
            reuse_neighbors=True,
            preferred_basis="umap",
            strict_preferred=True,
        )
        graph_summary = stream_dir / "velocity_graph_summary.json"
        graph_umap_summary = stream_dir / "velocity_graph_summary_umap.json"
        if graph_summary.exists():
            graph_umap_summary.write_bytes(graph_summary.read_bytes())

        plot_umap = plot_velocity_stream_bundle(
            adata=sub,
            output_dir=str(stream_dir),
            dim_reduction="umap",
            color_key="treatment_day",
            vkey="velocity",
        )
        graph_pca = build_velocity_graph_bundle(
            adata=sub,
            output_dir=str(stream_dir),
            n_neighbors=35,
            n_jobs=1,
            reuse_neighbors=True,
            preferred_basis="pca",
            strict_preferred=True,
        )
        graph_pca_summary = stream_dir / "velocity_graph_summary_pca.json"
        if graph_summary.exists():
            graph_pca_summary.write_bytes(graph_summary.read_bytes())

        plot_pca = plot_velocity_stream_bundle(
            adata=sub,
            output_dir=str(stream_dir),
            dim_reduction="pca",
            color_key="treatment_day",
            vkey="velocity",
        )
        warnings.extend(vel.get("warnings", []))
        warnings.extend(graph_umap.get("warnings", []))
        warnings.extend(plot_umap.get("warnings", []))
        warnings.extend(graph_pca.get("warnings", []))
        warnings.extend(plot_pca.get("warnings", []))
        artifacts["package_velocity"] = vel.get("artifacts", {})
        artifacts["package_graph_umap"] = graph_umap.get("artifacts", {})
        artifacts["package_plot_umap"] = plot_umap.get("artifacts", {})
        artifacts["package_graph_pca"] = graph_pca.get("artifacts", {})
        artifacts["package_plot_pca"] = plot_pca.get("artifacts", {})
        if graph_umap_summary.exists():
            artifacts["graph_summary_umap"] = str(graph_umap_summary)
        if graph_pca_summary.exists():
            artifacts["graph_summary_pca"] = str(graph_pca_summary)

        umap_table = stream_dir / "velocity_stream_umap_cells.csv"
        pd.DataFrame(
            {
                "cell": sub.obs_names.to_numpy(),
                "umap_1": sub.obsm["X_umap"][:, 0],
                "umap_2": sub.obsm["X_umap"][:, 1],
                "time_point_processed": sub.obs["time_point_processed"].to_numpy(),
                "treatment_day": sub.obs["treatment_day"].astype(str).to_numpy(),
            }
        ).to_csv(umap_table, index=False)
        artifacts["umap_cells_table"] = str(umap_table)
        pca_table = stream_dir / "velocity_stream_pca_cells.csv"
        pd.DataFrame(
            {
                "cell": sub.obs_names.to_numpy(),
                "pca_1": sub.obsm["X_pca"][:, 0],
                "pca_2": sub.obsm["X_pca"][:, 1],
                "time_point_processed": sub.obs["time_point_processed"].to_numpy(),
                "treatment_day": sub.obs["treatment_day"].astype(str).to_numpy(),
            }
        ).to_csv(pca_table, index=False)
        artifacts["pca_cells_table"] = str(pca_table)

        package_stream = Path(plot_umap.get("artifacts", {}).get("velocity_stream", ""))
        story_path = FIG / "story_scvelo_velocity_stream.png"
        if package_stream.exists():
            story_path.write_bytes(package_stream.read_bytes())
            artifacts["story_stream"] = str(story_path)
        package_stream_pca = Path(plot_pca.get("artifacts", {}).get("velocity_stream", ""))
        story_path_pca = FIG / "story_scvelo_velocity_stream_pca.png"
        if package_stream_pca.exists():
            story_path_pca.write_bytes(package_stream_pca.read_bytes())
            artifacts["story_stream_pca"] = str(story_path_pca)

        payload = {
            "analysis": "downstream-scvelo-stream",
            "evidence_type": "local flow diagnostic from trained CellCompass velocity field",
            "core_rule": "stream plots are not continuous trajectory evidence by themselves",
            "basis": ["umap_from_X_latent", "pca_from_X_latent"],
            "velocity_key": "velocity",
            "color_key": "treatment_day",
            "strict_skill_api": True,
            "api_sequence": [
                "scanpy.pp.neighbors(use_rep='X_latent')",
                "scanpy.tl.umap",
                "sklearn.PCA(n_components=2).fit_transform(X_latent)",
                "compute_velocity_bundle",
                "build_velocity_graph_bundle(preferred_basis='umap')",
                "plot_velocity_stream_bundle(dim_reduction='umap')",
                "build_velocity_graph_bundle(preferred_basis='pca')",
                "plot_velocity_stream_bundle(dim_reduction='pca')",
            ],
            "n_cells_stream": int(sub.n_obs),
            "n_cells_total": int(adata.n_obs),
            "model_state_path": str(MODEL_STATE_PATH.relative_to(REPO)),
            "model_artifact_path": str(MODEL_ARTIFACT_PATH.relative_to(REPO)),
            "input_h5ad_path": str(ADATA_PATH.relative_to(REPO)),
            "algorithm_id": str(record.get("algorithm_id", "")),
            "stage": str(record.get("stage", "")),
            "artifacts": artifacts,
            "warnings": warnings,
        }
        manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return payload
    except Exception as e:
        payload = {
            "analysis": "downstream-scvelo-stream",
            "evidence_type": "local flow diagnostic",
            "artifacts": artifacts,
            "warnings": warnings + [f"stream analysis failed: {type(e).__name__}: {e}"],
            "model_state_path": str(MODEL_STATE_PATH.relative_to(REPO)),
            "input_h5ad_path": str(ADATA_PATH.relative_to(REPO)),
        }
        manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[pulsebridge] scVelo stream failed: {e}", flush=True)
        return payload


def plot_story_scvelo_velocity_stream_pca_only(adata, max_cells: int = 5000) -> dict:
    """Strict scVelo stream plot on PCA coordinates from X_latent."""
    stream_dir = OUT / "downstream" / "velocity_stream"
    stream_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = stream_dir / "manifest_pca.json"
    warnings: list[str] = []
    artifacts: dict[str, object] = {}

    cyto_path = str(REPO / "CytoBridge-main")
    if cyto_path not in sys.path:
        sys.path.insert(0, cyto_path)
    try:
        from CytoBridge.tl.downstream import compute_velocity_bundle, build_velocity_graph_bundle
        from CytoBridge.pl.downstream import plot_velocity_stream_bundle
    except Exception as e:
        payload = {
            "analysis": "downstream-scvelo-stream-pca",
            "evidence_type": "local flow diagnostic",
            "artifacts": {},
            "warnings": [f"scVelo/CytoBridge downstream import failed: {e}"],
        }
        manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return payload

    try:
        x_latent = np.asarray(adata.obsm["X_latent"], dtype=np.float32)
        times = adata.obs["time_point_processed"].astype(float).to_numpy()
        keep = sample_balanced_by_time(times, max_cells=max_cells, random_state=31)

        obs = adata.obs.iloc[keep].copy()
        sub = ad.AnnData(X=np.zeros((len(keep), 1), dtype=np.float32), obs=obs)
        sub.var_names = ["placeholder"]
        sub.obsm["X_latent"] = x_latent[keep].copy()
        sub.obsm["X_pca"] = PCA(n_components=2, random_state=31).fit_transform(sub.obsm["X_latent"]).astype(np.float32)
        sub.obs["time_point_processed"] = pd.to_numeric(sub.obs["time_point_processed"], errors="coerce").astype(float)
        sub.obs["treatment_day"] = pd.Categorical(
            [f"day {int(round(float(x)))}" for x in sub.obs["time_point_processed"]],
            categories=["day 0", "day 1", "day 2", "day 3"],
            ordered=True,
        )

        model, record = load_pulsebridge_model_for_downstream(latent_dim=x_latent.shape[1])
        vel = compute_velocity_bundle(adata=sub, model=model, device="cpu", output_dir=str(stream_dir))
        graph = build_velocity_graph_bundle(
            adata=sub,
            output_dir=str(stream_dir),
            n_neighbors=35,
            n_jobs=1,
            reuse_neighbors=False,
            preferred_basis="pca",
            strict_preferred=True,
        )
        graph_summary = stream_dir / "velocity_graph_summary.json"
        graph_pca_summary = stream_dir / "velocity_graph_summary_pca.json"
        if graph_summary.exists():
            graph_pca_summary.write_bytes(graph_summary.read_bytes())

        plot = plot_velocity_stream_bundle(
            adata=sub,
            output_dir=str(stream_dir),
            dim_reduction="pca",
            color_key="treatment_day",
            vkey="velocity",
        )
        warnings.extend(vel.get("warnings", []))
        warnings.extend(graph.get("warnings", []))
        warnings.extend(plot.get("warnings", []))
        artifacts["package_velocity"] = vel.get("artifacts", {})
        artifacts["package_graph_pca"] = graph.get("artifacts", {})
        artifacts["package_plot_pca"] = plot.get("artifacts", {})
        if graph_pca_summary.exists():
            artifacts["graph_summary_pca"] = str(graph_pca_summary)

        pca_table = stream_dir / "velocity_stream_pca_cells.csv"
        pd.DataFrame(
            {
                "cell": sub.obs_names.to_numpy(),
                "pca_1": sub.obsm["X_pca"][:, 0],
                "pca_2": sub.obsm["X_pca"][:, 1],
                "time_point_processed": sub.obs["time_point_processed"].to_numpy(),
                "treatment_day": sub.obs["treatment_day"].astype(str).to_numpy(),
            }
        ).to_csv(pca_table, index=False)
        artifacts["pca_cells_table"] = str(pca_table)

        package_stream = Path(plot.get("artifacts", {}).get("velocity_stream", ""))
        story_path = FIG / "story_scvelo_velocity_stream_pca.png"
        if package_stream.exists():
            story_path.write_bytes(package_stream.read_bytes())
            artifacts["story_stream_pca"] = str(story_path)

        payload = {
            "analysis": "downstream-scvelo-stream-pca",
            "evidence_type": "local flow diagnostic from trained CellCompass velocity field",
            "core_rule": "stream plots are not continuous trajectory evidence by themselves",
            "basis": "pca_from_X_latent",
            "velocity_key": "velocity",
            "color_key": "treatment_day",
            "strict_skill_api": True,
            "api_sequence": [
                "sklearn.PCA(n_components=2).fit_transform(X_latent)",
                "compute_velocity_bundle",
                "build_velocity_graph_bundle(preferred_basis='pca', strict_preferred=True)",
                "plot_velocity_stream_bundle(dim_reduction='pca')",
                "scv.pl.velocity_embedding_stream",
            ],
            "n_cells_stream": int(sub.n_obs),
            "n_cells_total": int(adata.n_obs),
            "model_state_path": str(MODEL_STATE_PATH.relative_to(REPO)),
            "model_artifact_path": str(MODEL_ARTIFACT_PATH.relative_to(REPO)),
            "input_h5ad_path": str(ADATA_PATH.relative_to(REPO)),
            "algorithm_id": str(record.get("algorithm_id", "")),
            "stage": str(record.get("stage", "")),
            "artifacts": artifacts,
            "warnings": warnings,
        }
        manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return payload
    except Exception as e:
        payload = {
            "analysis": "downstream-scvelo-stream-pca",
            "evidence_type": "local flow diagnostic",
            "artifacts": artifacts,
            "warnings": warnings + [f"PCA stream analysis failed: {type(e).__name__}: {e}"],
            "model_state_path": str(MODEL_STATE_PATH.relative_to(REPO)),
            "input_h5ad_path": str(ADATA_PATH.relative_to(REPO)),
        }
        manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[pulsebridge] PCA scVelo stream failed: {e}", flush=True)
        return payload


def plot_story_main_claim_figure(
    endpoint_dynamic: pd.DataFrame,
    baseline: pd.DataFrame,
    align: pd.DataFrame,
    match_by_day: pd.DataFrame,
    state_df: pd.DataFrame,
    gene_programs: pd.DataFrame,
) -> None:
    """A claim-driven main figure draft that avoids diagnostic cosine/time projections."""
    axis_df = pd.read_csv(TAB / "new_endpoint_vs_dynamic_axis_summary.csv")
    summary = endpoint_dynamic.set_index("quantity")["value"]
    path_ratio = float(summary["mean_path_to_endpoint_ratio"])
    focus = baseline[baseline["algorithm"].isin(["PulseBridge", "WFR-FM"])].copy()
    focus = focus.set_index("algorithm").loc[["PulseBridge", "WFR-FM"]].reset_index()
    improvement = 100.0 * (
        1.0
        - float(focus.loc[focus["algorithm"] == "PulseBridge", "pcie"].iloc[0])
        / float(focus.loc[focus["algorithm"] == "WFR-FM", "pcie"].iloc[0])
    )
    mean_leak = float(align["mean_nuisance_leakage_fraction"].mean())
    max_leak = float(align["mean_nuisance_leakage_fraction"].max())

    day_colors = {0.0: "#31572C", 1.0: "#4F86C6", 2.0: "#F28E2B", 3.0: "#B23A48"}
    teal = "#146C94"
    red = "#B23A48"
    gray = "#707070"

    fig = plt.figure(figsize=(14.2, 8.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.05], width_ratios=[1.2, 1.0, 0.95])
    ax_path = fig.add_subplot(gs[0, 0])
    ax_match = fig.add_subplot(gs[0, 1])
    ax_pcie = fig.add_subplot(gs[0, 2])
    ax_state = fig.add_subplot(gs[1, 0])
    ax_gene = fig.add_subplot(gs[1, 1:3])

    def panel_label(ax, label: str) -> None:
        ax.text(
            -0.12,
            1.10,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=13,
            fontweight="bold",
            color="#111111",
        )

    # A. Endpoint-vs-path response-axis view. This is a descriptive axis readout,
    # while the path ratio annotation comes from the full latent trajectory.
    gen = axis_df[axis_df["source"] == "generated"].sort_values("time")
    obs = axis_df[axis_df["source"] == "observed"].sort_values("time")
    ax_path.plot(gen["time"], gen["mean_axis"], color=teal, lw=2.6, label="PulseBridge path")
    ax_path.scatter(
        obs["time"],
        obs["mean_axis"],
        s=54,
        color=[day_colors[float(t)] for t in obs["time"]],
        edgecolor="white",
        linewidth=0.9,
        zorder=4,
        label="observed days",
    )
    endpoint_line = np.interp(gen["time"], [0, 3], [obs.iloc[0]["mean_axis"], obs.iloc[-1]["mean_axis"]])
    ax_path.plot(gen["time"], endpoint_line, color=gray, ls="--", lw=1.5, label="endpoint shortcut")
    ax_path.set_title("Endpoint summary compresses the route", fontsize=11)
    ax_path.set_xlabel("treatment day")
    ax_path.set_ylabel("response-axis coordinate")
    ax_path.set_xticks([0, 1, 2, 3])
    ax_path.text(
        0.05,
        0.92,
        f"latent route / endpoint = {path_ratio:.2f}x",
        transform=ax_path.transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
        color="#111111",
        bbox={"boxstyle": "round,pad=0.28", "fc": "white", "ec": "#D8D8D8", "alpha": 0.95},
    )
    ax_path.legend(frameon=False, fontsize=8, loc="lower right")
    panel_label(ax_path, "A")

    # B. Match should be judged in the model's response/latent checks, not by a 2D PCA path.
    min_val = float(min(match_by_day["observed_axis_mean"].min(), match_by_day["generated_axis_mean"].min()))
    max_val = float(max(match_by_day["observed_axis_mean"].max(), match_by_day["generated_axis_mean"].max()))
    pad = 0.14 * max(1.0, max_val - min_val)
    lims = [min_val - pad, max_val + pad]
    ax_match.plot(lims, lims, color="#8C8C8C", lw=1.2, ls="--", zorder=1)
    for _, row in match_by_day.iterrows():
        t = float(row["time"])
        ax_match.scatter(
            row["observed_axis_mean"],
            row["generated_axis_mean"],
            s=62,
            color=day_colors[t],
            edgecolor="white",
            linewidth=0.9,
            zorder=3,
        )
        ax_match.text(
            row["observed_axis_mean"] + 0.025,
            row["generated_axis_mean"] + 0.025,
            f"d{int(t)}",
            fontsize=8.5,
            color=day_colors[t],
            ha="left",
            va="bottom",
        )
    ax_match.set_xlim(lims)
    ax_match.set_ylim(lims)
    ax_match.set_aspect("equal", adjustable="box")
    ax_match.set_title("Generated slices match observed days", fontsize=11)
    ax_match.set_xlabel("observed response-axis mean")
    ax_match.set_ylabel("generated response-axis mean")
    ax_match.text(
        0.04,
        0.96,
        "max centroid error/radius\n"
        f"{match_by_day['centroid_distance_over_radius'].max():.2f}\n"
        "max state-distance\n"
        f"{match_by_day['state_fraction_tvd'].max():.2f}",
        transform=ax_match.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color="#222222",
        bbox={"boxstyle": "round,pad=0.28", "fc": "white", "ec": "#D8D8D8", "alpha": 0.95},
    )
    panel_label(ax_match, "B")

    # C. Scorecard summary: direction agreement plus leakage boundary, not a cosine time curve.
    bar_colors = [teal, "#A5ADB5"]
    ax_pcie.bar(focus["algorithm"], focus["pcie"], color=bar_colors, width=0.62)
    for i, (_, row) in enumerate(focus.iterrows()):
        ax_pcie.text(i, float(row["pcie"]) + 0.025, f"{float(row['pcie']):.3f}", ha="center", va="bottom", fontsize=9)
    ax_pcie.set_ylim(0, max(1.08, float(focus["pcie"].max()) + 0.12))
    ax_pcie.set_ylabel("PCIE (lower is better)")
    ax_pcie.set_title("Heldout old/new RNA checks direction", fontsize=11)
    ax_pcie.text(
        0.5,
        0.70,
        f"{improvement:.1f}% lower PCIE\nthan WFR-FM",
        transform=ax_pcie.transAxes,
        ha="center",
        va="center",
        fontsize=10,
        color=teal,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.28", "fc": "white", "ec": "#D9E8EF", "alpha": 0.96},
    )
    ax_pcie.text(
        0.5,
        0.20,
        f"baseline-labeling leakage\nmean {mean_leak:.2f}, max {max_leak:.2f} < budget 0.50",
        transform=ax_pcie.transAxes,
        ha="center",
        va="center",
        fontsize=8.5,
        color="#333333",
    )
    panel_label(ax_pcie, "C")

    # D. Collapse six response-axis bins into an interpretable early-vs-late state readout.
    state_summary_rows = []
    for (source, time), sub in state_df.groupby(["source", "time"]):
        frac = sub.set_index("state")["fraction"]
        state_summary_rows.append(
            {
                "source": source,
                "time": float(time),
                "early": float(frac.get("state_1", 0.0) + frac.get("state_2", 0.0)),
                "late": float(frac.get("state_5", 0.0) + frac.get("state_6", 0.0)),
            }
        )
    state_summary = pd.DataFrame(state_summary_rows).sort_values(["source", "time"])
    state_summary.to_csv(TAB / "story_early_late_state_summary.csv", index=False)
    style = {"observed": "-", "generated": "--"}
    marker = {"observed": "o", "generated": "s"}
    for source in ["observed", "generated"]:
        sub = state_summary[state_summary["source"] == source]
        ax_state.plot(
            sub["time"],
            sub["early"],
            color="#C65A36",
            ls=style[source],
            marker=marker[source],
            lw=2.0,
            label=f"early {source}",
        )
        ax_state.plot(
            sub["time"],
            sub["late"],
            color=teal,
            ls=style[source],
            marker=marker[source],
            lw=2.0,
            label=f"late {source}",
        )
    ax_state.set_title("Cells move from early to late response states", fontsize=11)
    ax_state.set_xlabel("treatment day")
    ax_state.set_ylabel("cell fraction")
    ax_state.set_xticks([0, 1, 2, 3])
    ax_state.set_ylim(0, max(0.70, float(state_summary[["early", "late"]].to_numpy().max()) + 0.08))
    ax_state.legend(frameon=False, fontsize=7.5, loc="best", ncol=2)
    panel_label(ax_state, "D")

    # E. Program arrows are biology readouts along the trajectory, not causal mechanism claims.
    selected_modules = [
        "immune_surface_stress",
        "adhesion_stress",
        "mitochondrial_readout",
        "polyamine_stress",
        "proteostasis_heat_shock",
        "iron_storage",
    ]
    arrow_rows = []
    for module in selected_modules:
        sub = gene_programs[gene_programs["module"] == module].sort_values("time")
        if sub.empty or not {0.0, 3.0}.issubset(set(sub["time"].astype(float))):
            continue
        z0 = float(sub.loc[sub["time"] == 0.0, "z_score"].iloc[0])
        z3 = float(sub.loc[sub["time"] == 3.0, "z_score"].iloc[0])
        arrow_rows.append({"module": module, "z0": z0, "z3": z3, "delta": z3 - z0})
    arrow_df = pd.DataFrame(arrow_rows)
    arrow_df["abs_delta"] = arrow_df["delta"].abs()
    arrow_df = arrow_df.sort_values(["delta", "abs_delta"], ascending=[True, False]).reset_index(drop=True)
    y_positions = np.arange(len(arrow_df))
    for y, row in zip(y_positions, arrow_df.itertuples(index=False)):
        color = teal if row.delta > 0 else red
        ax_gene.annotate(
            "",
            xy=(row.z3, y),
            xytext=(row.z0, y),
            arrowprops={"arrowstyle": "->", "lw": 2.2, "color": color, "shrinkA": 0, "shrinkB": 0},
        )
        ax_gene.scatter(row.z0, y, s=55, color="#FFFFFF", edgecolor=color, linewidth=1.4, zorder=3)
        ax_gene.scatter(row.z3, y, s=66, color=color, edgecolor="white", linewidth=0.8, zorder=4)
        ax_gene.text(
            max(row.z0, row.z3) + 0.07,
            y,
            f"{row.delta:+.2f}",
            va="center",
            ha="left",
            fontsize=8.5,
            color=color,
        )
    ax_gene.axvline(0, color="#555555", lw=0.8)
    ax_gene.set_yticks(y_positions)
    ax_gene.set_yticklabels([m.replace("_", " ") for m in arrow_df["module"]], fontsize=9)
    ax_gene.set_xlabel("within-program z-score")
    ax_gene.set_title("Trajectory-linked biological readouts shift from day 0 to day 3", fontsize=11)
    xmin = float(arrow_df[["z0", "z3"]].min().min()) - 0.25
    xmax = float(arrow_df[["z0", "z3"]].max().max()) + 0.42
    ax_gene.set_xlim(xmin, xmax)
    ax_gene.text(
        0.99,
        0.06,
        "open: day 0   filled: day 3\nreadouts, not causal proof",
        transform=ax_gene.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        color="#333333",
    )
    panel_label(ax_gene, "E")

    fig.suptitle(
        "PulseBridge models HCT116 5-AZA-CdR response as a directionally validated cell-state trajectory",
        y=0.995,
        fontsize=14,
    )
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.965])
    plt.savefig(FIG / "story_main_claim_figure_v1.png", dpi=240, bbox_inches="tight")
    plt.close()


def plot_story_manuscript_overview(
    endpoint_dynamic: pd.DataFrame,
    align: pd.DataFrame,
    state_river: pd.DataFrame,
    gene_programs: pd.DataFrame,
) -> None:
    """One compact figure that maps the central manuscript claim to evidence."""
    axis_df = pd.read_csv(TAB / "new_endpoint_vs_dynamic_axis_summary.csv")
    summary = endpoint_dynamic.set_index("quantity")["value"]
    path_ratio = float(summary["mean_path_to_endpoint_ratio"])
    state_names = sorted(state_river["state"].unique(), key=lambda s: int(s.split("_")[1]))
    state_pivot = (
        state_river.pivot(index="time", columns="state", values="fraction")
        .fillna(0.0)
        .reindex(columns=state_names)
    )
    state_colors = sns.color_palette("Spectral", n_colors=len(state_names))
    anchor_palette = {0.0: "#31572C", 1.0: "#4F86C6", 2.0: "#F28E2B", 3.0: "#B23A48"}

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.2))
    ax = axes[0, 0]
    gen = axis_df[axis_df["source"] == "generated"].sort_values("time")
    obs = axis_df[axis_df["source"] == "observed"].sort_values("time")
    ax.plot(gen["time"], gen["mean_axis"], color="#146C94", lw=2.5, label="PulseBridge trajectory")
    ax.fill_between(
        gen["time"],
        gen["mean_axis"] - 0.25 * gen["std_axis"],
        gen["mean_axis"] + 0.25 * gen["std_axis"],
        color="#146C94",
        alpha=0.12,
        lw=0,
    )
    ax.scatter(obs["time"], obs["mean_axis"], color="#B23A48", s=42, label="observed days", zorder=3)
    endpoint_line = np.interp(gen["time"], [0, 3], [obs.iloc[0]["mean_axis"], obs.iloc[-1]["mean_axis"]])
    ax.plot(gen["time"], endpoint_line, color="#777777", ls="--", lw=1.5, label="endpoint-only summary")
    ax.set_title(f"1. Dynamic path, not endpoint only ({path_ratio:.2f}x)")
    ax.set_xlabel("treatment day")
    ax.set_ylabel("response-axis coordinate")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    ax.stackplot(state_pivot.index.to_numpy(), [state_pivot[s].to_numpy() for s in state_names], colors=state_colors, alpha=0.92)
    ax.set_ylim(0, 1)
    ax.set_title("2. Population shifts through response states")
    ax.set_xlabel("generated trajectory time")
    ax.set_ylabel("state fraction")
    ax.text(3.05, 0.88, "late-like\nstates expand", fontsize=8, ha="left", va="center")

    ax = axes[1, 0]
    for anchor, sub in align.groupby("nearest_anchor_time"):
        color = anchor_palette.get(float(anchor), "#777777")
        ax.axvspan(max(0, sub["time"].min() - 0.05), min(3, sub["time"].max() + 0.10), color=color, alpha=0.08)
        ax.scatter(sub["time"], sub["mean_deconfounded_cosine"], s=26 + 80 * sub["frac_positive_cosine"], color=color, edgecolor="white", lw=0.6)
        ax.vlines(sub["time"], 0, sub["mean_deconfounded_cosine"], color=color, lw=1.2, alpha=0.75)
    for switch in [0.5, 1.5, 2.5]:
        ax.axvline(switch, color="#777777", lw=0.8, ls=":")
    ax.axhline(0, color="#333333", lw=0.8)
    ax.set_title("3. Direction checked against heldout pulse anchors")
    ax.set_xlabel("generated trajectory time")
    ax.set_ylabel("directional cosine")

    ax = axes[1, 1]
    programs = [
        "polyamine_stress",
        "adhesion_stress",
        "mitochondrial_readout",
        "ribosome_translation",
        "proteostasis_heat_shock",
    ]
    programs = [p for p in programs if p in set(gene_programs["module"])]
    colors = dict(zip(programs, sns.color_palette("tab10", n_colors=len(programs))))
    for program in programs:
        sub = gene_programs[gene_programs["module"] == program].sort_values("time")
        ax.plot(sub["time"], sub["z_score"], marker="o", lw=2, color=colors[program], label=program.replace("_", " "))
    ax.axhline(0, color="#555555", lw=0.8)
    ax.set_title("4. Gene programs mark trajectory biology")
    ax.set_xlabel("treatment day")
    ax.set_ylabel("within-program z-score")
    ax.legend(frameon=False, fontsize=7, ncol=1, loc="best")

    fig.suptitle("PulseBridge turns HCT116 5-AZA-CdR response into a directionally validated trajectory", y=1.02)
    savefig(FIG / "story_manuscript_overview.png")


def write_story(
    baseline: pd.DataFrame,
    phase: pd.DataFrame,
    state_df: pd.DataFrame,
    gene_expr: pd.DataFrame,
    genes: pd.DataFrame,
    endpoint_dynamic: pd.DataFrame,
    state_response: pd.DataFrame,
    metabolic_state: pd.DataFrame,
    gene_programs: pd.DataFrame,
    original_markers: pd.DataFrame,
) -> None:
    pulse = baseline[baseline["algorithm"] == "PulseBridge"].iloc[0]
    wfr = baseline[baseline["algorithm"] == "WFR-FM"].iloc[0]
    phase_best = phase.sort_values("mean_deconfounded_cosine", ascending=False).iloc[0]
    phase_late = phase[phase["phase"] == "late_2_to_3"].iloc[0]
    phase_transition = phase[phase["phase"] == "transition_0.5_to_1"].iloc[0]
    sat1 = genes[genes["gene"] == "SAT1"].iloc[0] if (genes["gene"] == "SAT1").any() else None
    path_ratio = endpoint_dynamic.loc[endpoint_dynamic["quantity"] == "mean_path_to_endpoint_ratio", "value"].iloc[0]
    endpoint_axis_shift = endpoint_dynamic.loc[endpoint_dynamic["quantity"] == "endpoint_only_day0_to_day3_observed_axis_shift", "value"].iloc[0]
    strongest_state = state_response.sort_values("mean_axis_velocity", ascending=False).iloc[0]
    heldout_meta = metabolic_state[metabolic_state["split"] == "heldout"]
    strongest_meta = heldout_meta.sort_values("confidence_weighted_metabolic_axis", ascending=False).iloc[0] if len(heldout_meta) else None
    marker_total = original_markers[original_markers["layer"] == "log1p_norm"].copy()

    def marker_delta(gene: str) -> float:
        sub = marker_total[marker_total["gene"] == gene]
        if sub.empty:
            return float("nan")
        day0 = sub.loc[sub["time"] == 0.0, "mean_value"].iloc[0]
        day3 = sub.loc[sub["time"] == 3.0, "mean_value"].iloc[0]
        return float(day3 - day0)

    marker_deltas = {
        gene: marker_delta(gene)
        for gene in ["DKK1", "GAS5", "EREG", "MALAT1", "STAT1", "HEYL", "PITX1"]
        if gene in set(marker_total["gene"])
    }

    lines = [
        "# PulseBridge HCT116 Well-TEMP-seq Biological Story",
        "",
        "## One-sentence story",
        "",
        (
            "PulseBridge turns the HCT116 5-AZA-CdR Well-TEMP-seq experiment into a "
            "directionally checked drug-response trajectory: it fits the day 0-3 cell-state "
            "sequence while using old/new RNA information to favor treatment-response motion "
            "after subtracting baseline labeling/control motion."
        ),
        "",
        "## How this relates to the original Well-TEMP-seq paper",
        "",
        (
            "The original Well-TEMP-seq study established the experimental substrate: HCT116 "
            "cells treated with 5-AZA-CdR were profiled across control, day 0, day 1, day 2, "
            "and day 3, and the assay separates old and newly synthesized RNA. PulseBridge "
            "does not claim to re-prove the paper's full molecular mechanism. It uses the same "
            "kind of time-resolved old/new RNA signal as directional side information inside "
            "CellCompass, then asks whether the generated trajectory remains aligned with heldout "
            "pulse directions after removing control-vs-day0 labeling nuisance."
        ),
        "",
        "## Existing paper analyses rebuilt locally",
        "",
        f"- Metrics: PulseBridge W1 {pulse['w1_mean']:.4f}, PCIE {pulse['pcie']:.4f}; WFR-FM W1 {wfr['w1_mean']:.4f}, PCIE {wfr['pcie']:.4f}.",
        "- Time-readout: generated slices progress from day 0-like to later treatment-day-like states.",
        "- Pulse alignment: generated finite-difference tangents retain positive heldout pulse alignment over early/post-day1 regions while monitoring nuisance leakage.",
        "- Gene projection: the deconfounded velocity projection nominates SAT1 and stress/metabolism-associated readouts as hypotheses, not causal regulators.",
        "",
        "Figures:",
        "- `outputs/figures/existing_metric_w1_mean.png`",
        "- `outputs/figures/existing_metric_pcie.png`",
        "- `outputs/figures/existing_temporal_clouds_rebuilt.png`",
        "- `outputs/figures/existing_trajectory_time_readout_rebuilt.png`",
        "- `outputs/figures/existing_pulse_alignment_and_leakage_rebuilt.png`",
        "- `outputs/figures/existing_gene_projection_top25_rebuilt.png`",
        "",
        "## New analysis added for a cancer drug-response story",
        "",
        (
            f"0. Endpoint-only versus trajectory: observed day0-to-day3 endpoint shift along the response axis is "
            f"{endpoint_axis_shift:.3f}, but the generated route has a mean path-to-endpoint ratio of "
            f"{path_ratio:.2f}. This is the core reason to analyze drug response as a path rather than only "
            "as a final treated-versus-baseline difference."
        ),
        (
            f"1. Phase readout: the strongest mean deconfounded pulse cosine occurs in "
            f"`{phase_best['phase']}` (mean cosine {phase_best['mean_deconfounded_cosine']:.3f}). "
            f"The late `2_to_3` phase is also positive (mean cosine "
            f"{phase_late['mean_deconfounded_cosine']:.3f}), whereas the 0.5-1.0 day transition is "
            f"weaker/reversed (mean cosine {phase_transition['mean_deconfounded_cosine']:.3f}). "
            "This frames the response as a staged trajectory rather than a static endpoint comparison."
        ),
        (
            "2. Cell-state heterogeneity: six latent states ordered by observed treatment time show "
            "how the population composition shifts across days. Comparing observed and generated "
            "state fractions tests whether PulseBridge moves cells through similar coarse response states."
        ),
        (
            f"   The strongest generated forward push is in `{strongest_state['state']}` during "
            f"`{strongest_state['phase']}` (axis velocity {strongest_state['mean_axis_velocity']:.3f}, "
            f"forward fraction {strongest_state['forward_fraction']:.2f})."
        ),
        (
            "3. Metabolic side-information by state: heldout old/new RNA sidecars provide an independent "
            "directional readout over the same response-axis states. This does not replace PCIE, but it "
            "helps identify which coarse states carry pulse-supported response direction."
        ),
        (
            "4. Gene readout dynamics: top velocity-projection genes are plotted across treatment days, "
            "so the story can connect model-derived motion to measured expression trends while staying "
            "clear that this is hypothesis-generating."
        ),
    ]
    if strongest_meta is not None:
        lines.append(
            f"   The strongest heldout metabolic-axis readout is `{strongest_meta['state']}` at day "
            f"{float(strongest_meta['time']):.1f} (weighted axis value "
            f"{strongest_meta['confidence_weighted_metabolic_axis']:.3f})."
        )
    if sat1 is not None:
        lines.append(
            f"   SAT1 is the top positive projected gene in the downloaded analysis "
            f"(signed score {float(sat1['signed_score']):.3f}; day3-day0 mean expression delta "
            f"{float(sat1['expr_delta_t3_minus_t0']):.3f})."
        )
    lines += [
        "",
        "New figures:",
        "- `outputs/figures/new_endpoint_vs_dynamic_response.png`",
        "- `outputs/figures/new_response_phase_summary.png`",
        "- `outputs/figures/new_state_distribution_observed_vs_generated.png`",
        "- `outputs/figures/new_state_specific_response_heatmap.png`",
        "- `outputs/figures/new_state_metabolic_direction_heatmap.png`",
        "- `outputs/figures/new_top_projected_gene_expression_dynamics.png`",
        "- `outputs/figures/new_gene_program_readout_heatmap.png`",
        "- `outputs/figures/new_original_paper_marker_readout.png`",
        "- `outputs/figures/story_unique_biology_insight_panel.png`",
        "- `outputs/figures/story_source_aligned_pulsebridge_biology_map.png`",
        "- `outputs/figures/story_pulsebridge_unique_insight_panel.png`",
        "- `outputs/figures/story_drug_response_evidence_panel.png`",
        "- `outputs/figures/story_targeted_drug_response_gene_panel.png`",
        "- `outputs/figures/story_dynamic_drug_response_route.png`",
        "",
        "Recommended story figures for manuscript use:",
        "- `outputs/figures/story_unique_biology_insight_panel.png`: primary PulseBridge-specific biology figure; projects measured drug/cancer modules onto the continuous generated rollout to reveal event timing and state placement.",
        "- `outputs/figures/story_source_aligned_pulsebridge_biology_map.png`: source-paper-aligned biology map; anchors the 5-AZA-CdR mechanisms, removes MALAT1 as a main claim, and shows the added PulseBridge event ordering.",
        "- `outputs/figures/story_pulsebridge_unique_insight_panel.png`: model-specific support figure showing route-vs-endpoint geometry, continuous generated state occupancy, state-specific generated velocity, and PCIE advantage.",
        "- `outputs/figures/story_main_claim_figure_v1.png`: current draft main figure organized around the dynamic cancer-screening proof-of-concept claim.",
        "- `outputs/figures/story_drug_response_evidence_panel.png`: evidence-first, non-heatmap figure using real response-axis distributions, state fractions, module trajectories with SEM, and temporal/state support dot plots.",
        "- `outputs/figures/story_dynamic_drug_response_route.png`: graphical summary only; useful for talks, not the primary evidence figure.",
        "- `outputs/figures/story_targeted_drug_response_gene_panel.png`: targeted drug/cancer/gene support panel for DKK1/GAS5, EREG/PCNA, B2M/HLA/IFITM3, SAT1/LGALS1/mitochondrial stress, and HSP/FTL remodeling.",
        "- `outputs/figures/story_observed_generated_match.png`: direct observed-day match check on response axis, latent centroid error, and state composition distance.",
        "- `outputs/figures/story_directional_validation_scorecard.png`: clean PCIE/nuisance scorecard, avoiding a misleading cosine time-curve as the main display.",
        "- `outputs/figures/story_state_occupancy_bubbles.png`: bubble view of which response-axis states dominate each observed/generated day.",
        "- `outputs/figures/story_gene_program_direction_arrows.png`: day0-to-day3 program arrows for the biological readouts aligned with the inferred trajectory.",
        "- `outputs/figures/story_scvelo_velocity_stream.png`: scVelo-style UMAP stream from the trained CellCompass velocity field, generated by the downstream stream API; use as a supporting diagnostic, not standalone trajectory evidence.",
        "- `outputs/figures/story_scvelo_velocity_stream_pca.png`: scVelo-style PCA stream from the same trained CellCompass velocity field and downstream stream API; useful when UMAP geometry is too nonlinear.",
        "- `outputs/figures/story_endpoint_compass.png`: schematic visual argument that endpoint-only analysis compresses a longer response path.",
        "- `outputs/figures/story_data_space_trajectory.png`: exploratory 2D projection of the path in observed latent space; do not use alone to judge match quality.",
        "- `outputs/figures/story_anchor_pulse_diagnostic.png`: supplementary anchor-aware diagnostic only; do not use as a main biological response curve.",
        "",
        "## 中文文章故事线",
        "",
        (
            "这个结果和原 Well-TEMP-seq 文章可以对上，但对上的层级要说准确。原文证明的是："
            "HCT116 癌细胞在 5-AZA-CdR 处理 0、1、2、3 天后，old/new RNA 代谢标记能够捕捉"
            "药物反应中的时间性 RNA 动态。我们的 PulseBridge 结果不是重新证明 5-AZA-CdR 的完整"
            "分子机制，而是把原文的 old/new RNA 方向信息放进 CellCompass 框架，检验模型生成的细胞"
            "状态轨迹是不是沿着 heldout 代谢 pulse 指向的方向移动，并且把 control/day0 的标记背景"
            "影响扣掉。"
        ),
        "",
        (
            f"最适合写进 CellCompass significance 的结论是：只看 day0 到 day3 的 endpoint 会漏掉药物"
            f"反应路径本身。这里 observed day0-to-day3 在 response axis 上只移动 {endpoint_axis_shift:.3f}，"
            f"但 PulseBridge 生成轨迹的平均路径长度相对 endpoint 位移是 {path_ratio:.2f} 倍，说明细胞不是"
            "简单从起点直线到终点，而是经历了可分阶段的状态转移。"
        ),
        "",
        (
            f"阶段上，0-0.5 天和 2-3 天的 deconfounded pulse alignment 是正的，分别为 "
            f"{phase_best['mean_deconfounded_cosine']:.3f} 和 {phase_late['mean_deconfounded_cosine']:.3f}；"
            f"0.5-1 天阶段较弱/反向，为 {phase_transition['mean_deconfounded_cosine']:.3f}。这可以被写成："
            "CellCompass 不只是告诉我们药物最终有没有改变细胞，而是告诉我们反应什么时候出现、"
            "中间是否有转折、哪些细胞状态承担主要变化。"
        ),
        "",
        (
            f"细胞状态层面，最强的生成 forward push 出现在 `{strongest_state['phase']}` 的 "
            f"`{strongest_state['state']}`，axis velocity 为 {strongest_state['mean_axis_velocity']:.3f}。"
            "结合 state composition 图，可以把它解释为：同一个 drug response population 里面，"
            "不同粗粒度状态对药物轨迹的贡献不同，这正是动态筛药相比 endpoint DE 分析更有信息量的地方。"
        ),
        "",
        (
            "基因/药物反应层面，不能只写成 SAT1/polyamine stress、LGALS1/adhesion-stress 这种泛 pathway 标签。"
            "更强的讲法是把它拆成四条癌症药物反应轴：第一，DKK1/GAS5 的 new RNA 和 total RNA 上升，"
            "支持 5-AZA-CdR 作为 demethylating drug 诱导 tumor-suppressor reactivation；第二，EREG 下降、"
            "PCNA 下降，支持 oncogenic-growth/proliferation-associated readout 被压低，但不能说已经证明细胞死亡"
            "或增殖被抑制；第三，B2M/HLA/IFITM3 和原文 STAT1/viral-mimicry 机制相连，说明 late response state "
            "有 immune-surface/interferon-like remodeling；第四，SAT1、mitochondrial readout、LGALS1 上升提示 "
            "polyamine/metabolic stress 和 adhesion/immune-evasion-like adaptive response，其中 LGALS1 不应该写成"
            "“好”的抗癌 marker，而应该写成后续要验证的 compensatory candidate。"
        ),
        "",
        (
            "图形表达上，主线应该围绕 `story_main_claim_figure_v1.png` 组织：endpoint 不够、生成轨迹和"
            "观测天数吻合、old/new RNA 方向验证支持、细胞从 early response states 转到 late response states、"
            "并给出可跟进的 gene-program readouts。`story_scvelo_velocity_stream.png` 可以作为一个直观辅助图："
            "它按 downstream-scvelo-stream workflow 在 UMAP 上显示 trained CellCompass velocity field 的局部方向；但它仍然"
            "是 local flow diagnostic，不能单独替代 rollout 和 PCIE 证据。`story_data_space_trajectory.png` 只是"
            "二维投影探索图，不能单独用来判断匹配好坏；`story_anchor_pulse_diagnostic.png` 只适合作为补充诊断图，"
            "不要把 cosine 画成主文里的连续生物响应曲线。"
        ),
        "",
        (
            "和原文更具体的 marker 结论相比，本地数据支持的是“部分分子趋势可对齐，但不是逐条完全复现”："
            f"DKK1 day3-day0 总表达变化为 {marker_deltas.get('DKK1', float('nan')):.3f}，"
            f"GAS5 为 {marker_deltas.get('GAS5', float('nan')):.3f}，"
            f"EREG 为 {marker_deltas.get('EREG', float('nan')):.3f}，"
            f"MALAT1 为 {marker_deltas.get('MALAT1', float('nan')):.3f}。"
            "因此文章里更稳的写法是：PulseBridge 利用原文这种时间 RNA 代谢信息验证癌症药物反应轨迹；"
            "原文 marker 可作为后续生物应用分析入口，而不是把 PulseBridge 结果写成完整复现原文机制。"
        ),
        "",
        (
            "更具体地说，CellCompass 在这里的 significance 不是“发现了一个新的 5-AZA-CdR 分子机制”，"
            "而是把同一个药物反应拆成可比较的动态组件：哪些细胞先进入 response state，哪些 late state 保留，"
            "哪些 readout 看起来像 desired anti-tumor-like response，哪些 readout 反而像 adaptive/stress response。"
            "这个逻辑可以直接连接到动态筛药：未来比较两个药时，不只比较 day3 endpoint 距离，还可以比较是否进入 "
            "DKK1/GAS5/B2M 高、PCNA/EREG 低的 response path，还是主要进入 LGALS1/mitochondrial stress 高的"
            "可能补偿路径。详细 gene-level 解释见 `analysis/pulsebridge_story/DRUG_CANCER_GENE_INTERPRETATION.md`。"
        ),
        "",
        "## Evidence-to-claim map",
        "",
        "| Claim for manuscript | Local evidence | Main file |",
        "| --- | --- | --- |",
        (
            f"| Endpoint-only analysis misses dynamic response | path-to-endpoint ratio {path_ratio:.2f}; "
            f"endpoint axis shift {endpoint_axis_shift:.3f} | `outputs/figures/story_observed_generated_match.png`; `outputs/figures/story_endpoint_compass.png`; `outputs/figures/new_endpoint_vs_dynamic_response.png` |"
        ),
        (
            f"| Drug response is staged | early cosine {phase_best['mean_deconfounded_cosine']:.3f}, "
            f"late cosine {phase_late['mean_deconfounded_cosine']:.3f}, transition cosine "
            f"{phase_transition['mean_deconfounded_cosine']:.3f} | `outputs/tables/new_response_phase_summary.csv` |"
        ),
        (
            "| Response is cell-state heterogeneous | observed vs generated state fractions and state-specific "
            "axis velocities | `outputs/figures/story_state_occupancy_bubbles.png`; `outputs/figures/new_state_distribution_observed_vs_generated.png`; "
            "`outputs/figures/new_state_specific_response_heatmap.png` |"
        ),
        (
            f"| Old/new RNA provides directional validation | PulseBridge PCIE {pulse['pcie']:.4f} versus "
            f"WFR-FM {wfr['pcie']:.4f}; heldout state metabolic direction table | "
            "`outputs/figures/story_directional_validation_scorecard.png`; `outputs/figures/story_anchor_pulse_diagnostic.png`; `outputs/figures/existing_metric_pcie.png` |"
        ),
        (
            "| Local state-space direction supports the rollout | trained CellCompass velocity field rendered with the downstream scVelo-style UMAP stream workflow | "
            "`outputs/figures/story_scvelo_velocity_stream.png`; `outputs/figures/story_scvelo_velocity_stream_pca.png`; `outputs/downstream/velocity_stream/manifest.json` |"
        ),
        (
            "| Drug/cancer/gene application entry points | DKK1/GAS5 reactivation, EREG/PCNA repression, "
            "B2M/HLA/IFITM3 immune-surface remodeling, SAT1/LGALS1/mitochondrial adaptive stress readouts | "
            "`outputs/figures/story_drug_response_evidence_panel.png`; `outputs/figures/story_targeted_drug_response_gene_panel.png`; "
            "`outputs/tables/targeted_drug_response_support_summary.csv`; "
            "`outputs/tables/targeted_drug_response_gene_by_day_layer.csv`; `analysis/pulsebridge_story/DRUG_CANCER_GENE_INTERPRETATION.md` |"
        ),
        (
            "| Alignment to source-paper biology | source-paper markers DKK1/GAS5/EREG/MALAT1/STAT1/HEYL/PITX1 "
            "tracked across total/new/old RNA layers | `outputs/figures/new_original_paper_marker_readout.png`; "
            "`outputs/tables/new_original_paper_marker_readout_by_layer.csv` |"
        ),
        (
            "| Boundary of the claim | one cell line, one drug, balanced trajectory, gene readouts are hypotheses | "
            "`outputs/BIOLOGICAL_STORY.md` |"
        ),
        "",
        "## Data checklist for follow-up lineage/multi-omics-style figures",
        "",
        (
            "- Raw GEO archive: "
            "`downloads/pulsebridge_welltemp_hct116_20260627/data/cytobridge/runs/app_full_wave_20260617_214104/"
            "metabolic_velocity_anchor_bridge/paper_intake/downloads/GSE194357_RAW.tar`"
        ),
        (
            "- GEO metadata matrix: "
            "`downloads/pulsebridge_welltemp_hct116_20260627/data/cytobridge/runs/app_full_wave_20260617_214104/"
            "metabolic_velocity_anchor_bridge/paper_intake/downloads/GSE194357_series_matrix.txt.gz`"
        ),
        f"- Prepared AnnData with expression, old/new layers, metabolic sidecars: `{ADATA_PATH.relative_to(REPO)}`",
        f"- Duplicate prepared h5ad from paper run: `{PREPARED_H5AD_PATH.relative_to(REPO)}`",
        f"- Generated PulseBridge trajectory npz: `{TRAJ_PATH.relative_to(REPO)}`",
        f"- Training metrics: `{METRICS_PATH.relative_to(REPO)}`",
        f"- Locked downstream biology tables: `{DOWNSTREAM_TABLES.relative_to(REPO)}`",
        f"- scVelo stream manifest: `{(OUT / 'downstream' / 'velocity_stream' / 'manifest.json').relative_to(REPO)}`",
        "- Reproducible story script: `analysis/pulsebridge_story/scripts/pulsebridge_story_analysis.py`",
        "",
        "## Significance language for CellCompass",
        "",
        (
            "A safe manuscript-level significance statement is: Traditional cancer drug-response analysis "
            "often asks how different treated cells are from controls at a fixed endpoint. PulseBridge shows "
            "that CellCompass can instead model response as a continuous, directionally validated cell-state "
            "trajectory. In HCT116 cells treated with 5-AZA-CdR, CellCompass integrates temporal single-cell "
            "states with metabolic RNA-labeling information, subtracts baseline labeling-related nuisance, "
            "and recovers a trajectory whose generated motion better matches heldout metabolic pulse direction. "
            "This is a proof of concept for dynamic cancer drug-response profiling: future screens could rank "
            "not only whether a drug changes endpoint states, but when the response emerges, which cell states "
            "carry it, and how the drug redirects state transitions."
        ),
        "",
        "## Boundaries",
        "",
        "- This is one cell line and one drug panel, not a full screening platform yet.",
        "- PulseBridge is balanced-only; it does not infer growth, death, or cell-count changes.",
        "- Gene projections are hypotheses for follow-up validation, not causal mechanism proof.",
        "- Full negative-control validation still needs shuffled-pulse and no-instrument/shuffled-baseline runs.",
        "",
        "## Data now available locally",
        "",
        f"- Prepared h5ad: `{ADATA_PATH.relative_to(REPO)}`",
        f"- Generated trajectory: `{TRAJ_PATH.relative_to(REPO)}`",
        f"- Downstream tables: `{DOWNSTREAM_TABLES.relative_to(REPO)}`",
        f"- Original GEO tar: `downloads/pulsebridge_welltemp_hct116_20260627/data/cytobridge/runs/app_full_wave_20260617_214104/metabolic_velocity_anchor_bridge/paper_intake/downloads/GSE194357_RAW.tar`",
    ]
    (OUT / "BIOLOGICAL_STORY.md").write_text("\n".join(lines) + "\n")


def write_data_inventory(adata) -> None:
    geo_dir = (
        BUNDLE
        / "data"
        / "cytobridge"
        / "runs"
        / "app_full_wave_20260617_214104"
        / "metabolic_velocity_anchor_bridge"
        / "paper_intake"
        / "downloads"
    )
    raw_geo = geo_dir / "GSE194357_RAW.tar"
    series_matrix = geo_dir / "GSE194357_series_matrix.txt.gz"
    lines = [
        "# PulseBridge HCT116 Data Inventory",
        "",
        "This inventory records the local files needed to reproduce the current PulseBridge story figures and to build follow-up lineage/multi-omics-style figures.",
        "",
        "## Dataset Shape",
        "",
        f"- AnnData shape: {adata.n_obs} cells x {adata.n_vars} genes",
        f"- Observation columns: `{', '.join(map(str, adata.obs.columns))}`",
        f"- Layers: `{', '.join(map(str, adata.layers.keys()))}`",
        f"- Embeddings/sidecars: `{', '.join(map(str, adata.obsm.keys()))}`",
        "",
        "## Source Data",
        "",
        f"- GEO raw archive: `{raw_geo.relative_to(REPO)}`",
        f"- GEO series metadata: `{series_matrix.relative_to(REPO)}`",
        f"- GEO accession: GSE194357",
        f"- Source paper: Lin et al. 2023, Well-TEMP-seq, Nature Communications, DOI 10.1038/s41467-023-36902-5",
        "",
        "## Prepared Analysis Data",
        "",
        f"- Primary h5ad: `{ADATA_PATH.relative_to(REPO)}`",
        f"- Duplicate h5ad from paper run: `{PREPARED_H5AD_PATH.relative_to(REPO)}`",
        "- Key layers: `log1p_norm` for total expression-like readouts, `new` and `old` for metabolic RNA labeling readouts.",
        "- Key sidecars: `X_latent` for generated trajectory comparison; `X_metabolic_velocity`, `X_metabolic_decay`, and `X_metabolic_synthesis` for old/new RNA direction analyses.",
        "",
        "## PulseBridge Run Artifacts",
        "",
        f"- Generated trajectory: `{TRAJ_PATH.relative_to(REPO)}`",
        f"- Metrics JSON: `{METRICS_PATH.relative_to(REPO)}`",
        f"- Locked downstream tables: `{DOWNSTREAM_TABLES.relative_to(REPO)}`",
        f"- Baseline comparison table: `{BASELINE_TABLE.relative_to(REPO)}`",
        "",
        "## Reproducible Story Outputs",
        "",
        "- Script: `analysis/pulsebridge_story/scripts/pulsebridge_story_analysis.py`",
        "- Story: `analysis/pulsebridge_story/outputs/BIOLOGICAL_STORY.md`",
        "- Drug/cancer/gene interpretation note: `analysis/pulsebridge_story/DRUG_CANCER_GENE_INTERPRETATION.md`",
        "- Manifest: `analysis/pulsebridge_story/outputs/analysis_manifest.json`",
        "- Figures: `analysis/pulsebridge_story/outputs/figures/`",
        "- Tables: `analysis/pulsebridge_story/outputs/tables/`",
        "",
        "## Follow-up Figure Entry Points",
        "",
        "- Main claim draft figure: `outputs/figures/story_main_claim_figure_v1.png`",
        "- scVelo-style trained velocity stream: `outputs/figures/story_scvelo_velocity_stream.png`",
        "- scVelo-style trained velocity stream in PCA space: `outputs/figures/story_scvelo_velocity_stream_pca.png`",
        "- scVelo stream manifest: `outputs/downstream/velocity_stream/manifest.json`",
        "- Main observed/generated match check: `outputs/figures/story_observed_generated_match.png`",
        "- Directional validation scorecard: `outputs/figures/story_directional_validation_scorecard.png`",
        "- State occupancy bubbles: `outputs/figures/story_state_occupancy_bubbles.png`",
        "- Primary PulseBridge-specific biology insight panel: `outputs/figures/story_unique_biology_insight_panel.png`",
        "- Source-paper-aligned PulseBridge biology map: `outputs/figures/story_source_aligned_pulsebridge_biology_map.png`",
        "- PulseBridge-specific model evidence panel: `outputs/figures/story_pulsebridge_unique_insight_panel.png`",
        "- Rollout biology profile table: `outputs/tables/pulsebridge_rollout_neighbor_biology_profile.csv`",
        "- Rollout biology event timing table: `outputs/tables/pulsebridge_rollout_biology_event_timing.csv`",
        "- Evidence-first drug response panel: `outputs/figures/story_drug_response_evidence_panel.png`",
        "- Dynamic drug-response route figure: `outputs/figures/story_dynamic_drug_response_route.png`",
        "- Targeted drug/cancer/gene support panel: `outputs/figures/story_targeted_drug_response_gene_panel.png`",
        "- Targeted support summary table: `outputs/tables/targeted_drug_response_support_summary.csv`",
        "- Targeted gene-by-layer table: `outputs/tables/targeted_drug_response_gene_by_day_layer.csv`",
        "- Targeted state/day module table: `outputs/tables/targeted_drug_response_module_by_state_day.csv`",
        "- Gene-program direction arrows: `outputs/figures/story_gene_program_direction_arrows.png`",
        "- Endpoint compass: `outputs/figures/story_endpoint_compass.png`",
        "- Exploratory data-space trajectory: `outputs/figures/story_data_space_trajectory.png`",
        "- Supplement-only anchor-aware pulse diagnostic: `outputs/figures/story_anchor_pulse_diagnostic.png`",
        "- Endpoint-vs-trajectory figure: `outputs/figures/new_endpoint_vs_dynamic_response.png`",
        "- State transition/composition figure: `outputs/figures/new_state_distribution_observed_vs_generated.png`",
        "- State-specific response heatmap: `outputs/figures/new_state_specific_response_heatmap.png`",
        "- Old/new RNA marker readout: `outputs/figures/new_original_paper_marker_readout.png`",
        "- Gene-program readout: `outputs/figures/new_gene_program_readout_heatmap.png`",
        "",
    ]
    (OUT / "DATA_INVENTORY.md").write_text("\n".join(lines))


def main() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    ensure_inputs()
    print("[pulsebridge] loading h5ad in backed mode", flush=True)
    adata = ad.read_h5ad(ADATA_PATH if ADATA_PATH.exists() else PREPARED_H5AD_PATH, backed="r")
    print("[pulsebridge] loading trajectory", flush=True)
    _, _, traj_points = load_trajectory_points()
    print("[pulsebridge] building response axis", flush=True)
    axis_info = build_response_axis(adata)

    print("[pulsebridge] rebuilding existing metrics", flush=True)
    baseline = plot_existing_metrics()
    print("[pulsebridge] rebuilding temporal clouds", flush=True)
    plot_existing_temporal_clouds(adata, traj_points)
    print("[pulsebridge] rebuilding time readout", flush=True)
    readout = plot_existing_time_readout()
    print("[pulsebridge] rebuilding pulse alignment", flush=True)
    align = plot_existing_pulse_alignment()
    print("[pulsebridge] rebuilding gene projection", flush=True)
    genes = plot_existing_gene_projection()

    print("[pulsebridge] summarizing response phases", flush=True)
    phase = summarize_response_phases(readout, align)
    print("[pulsebridge] contrasting endpoint and dynamic response", flush=True)
    endpoint_dynamic = endpoint_vs_dynamic_analysis(adata, axis_info, readout, align)
    print("[pulsebridge] analyzing state heterogeneity", flush=True)
    state_df = state_heterogeneity_analysis(adata, traj_points, axis_info)
    print("[pulsebridge] analyzing state-specific generated response", flush=True)
    state_response = state_specific_response_analysis(axis_info)
    print("[pulsebridge] analyzing state-specific metabolic direction", flush=True)
    metabolic_state = metabolic_state_direction_analysis(adata, axis_info)
    print("[pulsebridge] analyzing gene expression dynamics", flush=True)
    gene_expr = gene_expression_dynamics(adata, genes)
    print("[pulsebridge] analyzing gene-program readouts", flush=True)
    gene_programs = gene_program_readout(adata, genes)
    print("[pulsebridge] analyzing source-paper markers", flush=True)
    original_markers = original_paper_marker_readout(adata)
    print("[pulsebridge] analyzing targeted drug/cancer/gene support", flush=True)
    targeted_gene_day, targeted_module_day, targeted_state_module, targeted_support = targeted_drug_response_support_analysis(adata, axis_info)
    print("[pulsebridge] projecting measured biology onto PulseBridge rollout", flush=True)
    rollout_biology, rollout_timing = pulsebridge_rollout_biology_timing_analysis(adata)
    print("[pulsebridge] drawing story-oriented figures", flush=True)
    plot_story_pulsebridge_unique_insight_panel(state_df, state_response, baseline)
    plot_story_unique_biology_insight_panel(state_df, baseline, rollout_biology, rollout_timing)
    plot_story_source_aligned_pulsebridge_map(state_df, baseline, rollout_biology, rollout_timing, targeted_support)
    plot_story_drug_response_evidence_panel(adata, axis_info, state_df, targeted_module_day, targeted_support)
    plot_story_dynamic_drug_response_route(state_df, targeted_module_day, targeted_support, targeted_gene_day)
    plot_story_anchor_pulse_diagnostic(align)
    plot_story_endpoint_compass(endpoint_dynamic)
    plot_story_data_space_trajectory(adata)
    match_by_day = plot_story_observed_generated_match(adata, axis_info, state_df)
    state_river = plot_story_generated_state_river(axis_info, state_df)
    plot_story_state_occupancy_bubbles(state_df)
    plot_story_gene_program_timeline(gene_programs)
    plot_story_gene_program_direction_arrows(gene_programs)
    plot_story_directional_validation_scorecard(baseline, align)
    if os.environ.get("PULSEBRIDGE_SKIP_SCVELO", "0") == "1":
        stream_manifest = {"warnings": ["Skipped by PULSEBRIDGE_SKIP_SCVELO=1; existing stream output was not regenerated."]}
    else:
        stream_manifest = plot_story_scvelo_velocity_stream(adata)
    if stream_manifest.get("warnings"):
        print("[pulsebridge] scVelo stream warnings:", stream_manifest["warnings"], flush=True)
    plot_story_main_claim_figure(endpoint_dynamic, baseline, align, match_by_day, state_df, gene_programs)
    plot_story_manuscript_overview(endpoint_dynamic, align, state_river, gene_programs)

    print("[pulsebridge] writing story", flush=True)
    write_story(
        baseline,
        phase,
        state_df,
        gene_expr,
        genes,
        endpoint_dynamic,
        state_response,
        metabolic_state,
        gene_programs,
        original_markers,
    )
    print("[pulsebridge] writing data inventory", flush=True)
    write_data_inventory(adata)
    manifest = {
        "adata_path": str(ADATA_PATH),
        "trajectory_path": str(TRAJ_PATH),
        "metrics_path": str(METRICS_PATH),
        "downstream_tables": str(DOWNSTREAM_TABLES),
        "figures": sorted(str(p.relative_to(OUT)) for p in FIG.glob("*.png")),
        "tables": sorted(str(p.relative_to(OUT)) for p in TAB.glob("*.csv")),
        "documents": sorted(str(p.relative_to(OUT)) for p in OUT.glob("*.md")),
    }
    (OUT / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
