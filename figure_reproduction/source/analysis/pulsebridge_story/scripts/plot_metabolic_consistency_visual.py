#!/usr/bin/env python3
"""Visualize PulseBridge consistency with heldout metabolic pulse directions."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cellcompass_metabolic_vis")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/cellcompass_metabolic_numba")
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
METRICS_PATH = TRAJ_PATH.parent / "metrics.json"
BASELINE_TABLE = PAPER_PKG / "tables" / "baseline_metrics.csv"
UMAP_H5AD = OUT / "downstream" / "dynamo_stream" / "pulsebridge_full_umap_velocity_for_dynamo.h5ad"


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return np.sum(a * b, axis=1) / np.maximum(denom, 1e-8)


def build_nuisance_basis(latent: np.ndarray, obs: pd.DataFrame, rank: int, rng: np.random.Generator) -> np.ndarray:
    times = obs["time_point_processed"].astype(float).to_numpy()
    sample = obs["sample_name"].astype(str).to_numpy()
    t0 = times == 0.0
    control = t0 & (sample == "HCT116_control")
    baseline = t0 & (sample == "HCT116_0d")
    x_control, x_baseline = latent[control], latent[baseline]
    n = min(4000, len(x_control), len(x_baseline))
    if n < 10:
        return np.zeros((latent.shape[1], 0), dtype=np.float32)
    control_idx = rng.choice(len(x_control), n, replace=False)
    baseline_idx = rng.choice(len(x_baseline), n, replace=False)
    delta = x_baseline[baseline_idx] - x_control[control_idx]
    delta = delta - delta.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(delta, full_matrices=False)
    q, _ = np.linalg.qr(vh[:rank].T.astype(np.float32))
    return q[:, :rank].astype(np.float32)


def project_perp(x: np.ndarray, basis: np.ndarray) -> np.ndarray:
    if basis.size == 0:
        return x
    return x - (x @ basis) @ basis.T


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    weights = np.maximum(np.asarray(weights, dtype=float), 0.0)
    if float(weights.sum()) <= 0:
        return float(np.nanmean(values))
    return float(np.nansum(values * weights) / weights.sum())


def compute_anchor_pairs() -> tuple[pd.DataFrame, dict[float, dict[str, np.ndarray]], dict]:
    rng = np.random.default_rng(1729)
    adata = ad.read_h5ad(ADATA_PATH)
    latent = np.asarray(adata.obsm["X_latent"], dtype=np.float32)
    metabolic = np.asarray(adata.obsm["X_metabolic_velocity"], dtype=np.float32)
    obs = adata.obs.copy()
    obs_times = obs["time_point_processed"].astype(float).to_numpy()
    split = obs["mvab_anchor_split"].astype(str).to_numpy()
    confidence = obs["mvab_anchor_confidence"].astype(float).to_numpy().astype(np.float32)

    with METRICS_PATH.open() as f:
        metrics = json.load(f)
    custom = metrics.get("custom_metrics", {})
    nuisance_rank = int(custom.get("pcib_nuisance_rank", 3))
    nuisance_budget = float(custom.get("pcib_nuisance_budget", 0.5))
    basis = build_nuisance_basis(latent, obs, rank=nuisance_rank, rng=rng)

    z = np.load(TRAJ_PATH)
    time_points = np.asarray(z["time_points"], dtype=float)
    observed_indices = np.asarray(z["observed_time_indices"], dtype=int)
    observed_times = np.asarray(z["observed_time_points"], dtype=float)

    rows: list[pd.DataFrame] = []
    anchor_payload: dict[float, dict[str, np.ndarray]] = {}
    for time_i, traj_idx in enumerate(observed_indices):
        if int(traj_idx) + 1 >= len(time_points):
            continue
        anchor_time = float(observed_times[time_i])
        anchor_rows = np.where((obs_times == anchor_time) & (split == "heldout") & (confidence > 0))[0]
        if anchor_rows.size == 0:
            continue
        anchor_x = latent[anchor_rows]
        anchor_pulse = project_perp(metabolic[anchor_rows], basis)
        anchor_conf = confidence[anchor_rows]
        valid = (np.linalg.norm(anchor_pulse, axis=1) > 1e-8) & (anchor_conf > 0)
        anchor_rows = anchor_rows[valid]
        anchor_x = anchor_x[valid]
        anchor_pulse = anchor_pulse[valid]
        anchor_conf = anchor_conf[valid]
        if anchor_rows.size == 0:
            continue

        cur = np.asarray(z[f"points_{int(traj_idx):06d}"], dtype=np.float32)
        nxt = np.asarray(z[f"points_{int(traj_idx) + 1:06d}"], dtype=np.float32)
        n = min(len(cur), len(nxt))
        cur, nxt = cur[:n], nxt[:n]
        if n > 4096:
            particle_idx = rng.choice(np.arange(n), size=4096, replace=False)
            cur_ref = cur[particle_idx]
            nxt_ref = nxt[particle_idx]
        else:
            cur_ref, nxt_ref = cur, nxt
        dt = max(abs(float(time_points[int(traj_idx) + 1] - time_points[int(traj_idx)])), 1e-8)
        pred_velocity = (nxt_ref - cur_ref) / dt

        nn = NearestNeighbors(n_neighbors=1).fit(cur_ref)
        dist, ind = nn.kneighbors(anchor_x)
        pred_velocity = pred_velocity[ind[:, 0]]
        pred_perp = project_perp(pred_velocity, basis)
        cos = cosine(pred_perp, anchor_pulse)
        cos_error = np.clip(1.0 - cos, 0.0, 2.0)
        nuisance_energy = np.sum((pred_velocity @ basis) ** 2, axis=1) if basis.size else np.zeros(len(pred_velocity))
        velocity_energy = np.maximum(np.sum(pred_velocity * pred_velocity, axis=1), 1e-8)
        leakage = nuisance_energy / velocity_energy
        leakage_over_budget = np.maximum(leakage - nuisance_budget, 0.0)

        rows.append(
            pd.DataFrame(
                {
                    "anchor_time": anchor_time,
                    "anchor_index": anchor_rows,
                    "anchor_distance": dist[:, 0],
                    "confidence": anchor_conf,
                    "cosine": cos,
                    "cosine_error": cos_error,
                    "nuisance_leakage": leakage,
                    "leakage_over_budget": leakage_over_budget,
                    "positive_cosine": cos > 0,
                }
            )
        )
        anchor_payload[anchor_time] = {
            "anchor_x": anchor_x,
            "pred_perp": pred_perp,
            "pulse_perp": anchor_pulse,
            "cosine": cos,
            "confidence": anchor_conf,
        }

    pair_df = pd.concat(rows, ignore_index=True)
    summary = {
        "locked_pcie": float(custom.get("PCIE", custom.get("pcib_pulse_control_identifiability_error", np.nan))),
        "locked_cosine_error": float(custom.get("pcib_pcie_cosine_error", np.nan)),
        "locked_leakage_over_budget": float(custom.get("pcib_pcie_nuisance_leakage_above_budget", np.nan)),
        "visual_recomputed_cosine_error": weighted_mean(pair_df["cosine_error"].to_numpy(), pair_df["confidence"].to_numpy()),
        "visual_recomputed_leakage_over_budget": weighted_mean(pair_df["leakage_over_budget"].to_numpy(), pair_df["confidence"].to_numpy()),
        "nuisance_budget": nuisance_budget,
        "nuisance_rank": nuisance_rank,
        "heldout_anchor_pairs": int(len(pair_df)),
    }
    return pair_df, anchor_payload, summary


def draw_agreement_map(
    ax,
    umap: np.ndarray,
    times: np.ndarray,
    pair_df: pd.DataFrame,
    day: float,
    title: str,
) -> plt.Collection:
    bg_idx = np.arange(umap.shape[0])
    rng = np.random.default_rng(int(100 + day * 10))
    if bg_idx.size > 7000:
        bg_idx = rng.choice(bg_idx, size=7000, replace=False)
    ax.scatter(
        umap[bg_idx, 0],
        umap[bg_idx, 1],
        c=times[bg_idx],
        cmap="viridis",
        s=4,
        alpha=0.11,
        linewidths=0,
        rasterized=True,
        zorder=1,
    )
    sub = pair_df[pair_df["anchor_time"] == day].copy()
    anchor_idx = sub["anchor_index"].to_numpy(dtype=int)
    sizes = 10 + 28 * np.clip(sub["confidence"].to_numpy(dtype=float), 0, 1)
    sc = ax.scatter(
        umap[anchor_idx, 0],
        umap[anchor_idx, 1],
        c=sub["cosine"].to_numpy(dtype=float),
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        s=sizes,
        alpha=0.86,
        linewidths=0.15,
        edgecolors="white",
        rasterized=True,
        zorder=3,
    )
    ax.text(
        0.02,
        0.04,
        f"weighted mean cosine {weighted_mean(sub['cosine'].to_numpy(), sub['confidence'].to_numpy()):.2f}\n"
        f"{(sub['cosine'] > 0).mean():.0%} positive anchors",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        color="#222222",
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "none", "alpha": 0.72},
    )
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return sc


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    pair_df, _anchor_payload, summary = compute_anchor_pairs()
    pair_path = TAB / "story_metabolic_consistency_anchor_pairs.csv"
    pair_df.to_csv(pair_path, index=False)

    baseline = pd.read_csv(BASELINE_TABLE)
    focus = baseline[baseline["algorithm"].isin(["PulseBridge", "WFR-FM"])].copy()
    focus = focus.set_index("algorithm").loc[["PulseBridge", "WFR-FM"]].reset_index()
    pulse_pcie = float(focus.loc[focus["algorithm"] == "PulseBridge", "pcie"].iloc[0])
    wfr_pcie = float(focus.loc[focus["algorithm"] == "WFR-FM", "pcie"].iloc[0])
    improvement = 100.0 * (1.0 - pulse_pcie / wfr_pcie)

    if not UMAP_H5AD.exists():
        raise FileNotFoundError(f"Missing UMAP cache: {UMAP_H5AD}")
    adata_umap = ad.read_h5ad(UMAP_H5AD, backed="r")
    umap = np.asarray(adata_umap.obsm["X_umap"], dtype=np.float32)
    times = adata_umap.obs["time_point_processed"].astype(float).to_numpy()
    rng = np.random.default_rng(42)

    fig = plt.figure(figsize=(12.8, 9.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[0.9, 1.15])
    ax_pcie = fig.add_subplot(gs[0, 0])
    ax_cos = fig.add_subplot(gs[0, 1])
    ax_a = fig.add_subplot(gs[1, 0])
    ax_b = fig.add_subplot(gs[1, 1])

    colors = ["#146C94", "#AAB2BA"]
    ax_pcie.bar(focus["algorithm"], focus["pcie"], color=colors, width=0.58)
    for i, row in focus.iterrows():
        ax_pcie.text(i, row["pcie"] + 0.025, f"{row['pcie']:.3f}", ha="center", va="bottom", fontsize=9)
    ax_pcie.set_ylabel("PCIE, lower is better")
    ax_pcie.set_title("A. Heldout metabolic-pulse score")
    ax_pcie.set_ylim(0, max(1.15, float(focus["pcie"].max()) + 0.18))
    ax_pcie.text(
        0.04,
        0.96,
        f"{improvement:.1f}% lower than WFR-FM\n"
        f"locked cosine error {summary['locked_cosine_error']:.3f}\n"
        f"leakage over budget {summary['locked_leakage_over_budget']:.4f}",
        transform=ax_pcie.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="#222222",
    )
    ax_pcie.spines[["top", "right"]].set_visible(False)

    day_colors = {0.0: "#31572C", 1.0: "#4F86C6", 2.0: "#F28E2B"}
    for x, day in enumerate(sorted(pair_df["anchor_time"].unique())):
        sub = pair_df[pair_df["anchor_time"] == day]
        jitter = rng.normal(0, 0.055, len(sub))
        ax_cos.scatter(
            np.full(len(sub), x) + jitter,
            sub["cosine"],
            s=8,
            alpha=0.28,
            color=day_colors.get(float(day), "#777777"),
            linewidths=0,
            rasterized=True,
        )
        wm = weighted_mean(sub["cosine"].to_numpy(), sub["confidence"].to_numpy())
        ax_cos.scatter([x], [wm], marker="D", s=58, color="black", zorder=5)
        ax_cos.text(
            x,
            -0.93,
            f"{(sub['cosine'] > 0).mean():.0%} > 0",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax_cos.axhline(0, color="#333333", lw=0.8)
    ax_cos.set_xticks(range(len(sorted(pair_df["anchor_time"].unique()))))
    ax_cos.set_xticklabels([f"day {d:g}" for d in sorted(pair_df["anchor_time"].unique())])
    ax_cos.set_ylim(-1.0, 1.0)
    ax_cos.set_ylabel("cosine(model tangent, heldout pulse)")
    ax_cos.set_title("B. Most heldout anchors point the same way")
    ax_cos.spines[["top", "right"]].set_visible(False)

    sc = draw_agreement_map(
        ax_a,
        umap,
        times,
        pair_df,
        1.0,
        "C. Day-1 anchors on UMAP: red means direction agreement",
    )
    draw_agreement_map(
        ax_b,
        umap,
        times,
        pair_df,
        2.0,
        "D. Day-2 anchors on UMAP: agreement is spatially broad",
    )
    cbar = fig.colorbar(sc, ax=[ax_a, ax_b], fraction=0.025, pad=0.012)
    cbar.set_label("directional cosine to heldout metabolic pulse", fontsize=9)
    fig.suptitle(
        "Why the PCIE is lower: PulseBridge tangents preserve heldout metabolic-pulse direction",
        fontsize=14.2,
        y=1.02,
    )
    out_png = FIG / "story_metabolic_consistency_visual.png"
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    plt.close(fig)

    manifest = {
        "analysis": "metabolic-consistency-visual",
        "figure": str(out_png),
        "anchor_pair_table": str(pair_path),
        "locked_metrics": {
            "pulsebridge_pcie": pulse_pcie,
            "wfr_fm_pcie": wfr_pcie,
            "improvement_percent_vs_wfr_fm": improvement,
            "locked_cosine_error": summary["locked_cosine_error"],
            "locked_leakage_over_budget": summary["locked_leakage_over_budget"],
        },
        "visual_recomputation": summary,
        "notes": [
            "Cosines use heldout metabolic anchors after nuisance-basis projection.",
            "UMAP panels map each heldout anchor's high-dimensional cosine agreement back onto its observed cell coordinate.",
            "Locked PCIE values come from baseline_metrics.csv and metrics.json.",
        ],
    }
    manifest_path = OUT / "metabolic_consistency_visual_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
