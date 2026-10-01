#!/usr/bin/env python3
"""Gene-space decomposition of the locked PulseBridge PCIE calculation.

This script follows the same evaluation logic used by PCIE:

1. Load the generated trajectory.
2. Compute finite-difference generated tangents.
3. Match each heldout metabolic-pulse anchor to the nearest generated
   particle at the same observed anchor time.
4. Project generated tangents and heldout pulse anchors away from the
   control-vs-day0 nuisance basis.
5. Decompose the resulting cosine into PCA-supported gene contributions.

The output is therefore a gene-level explanation of the PCIE direction score,
not an observed-cell local velocity diagnostic.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cellcompass_pcie_gene_decomp")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/cellcompass_pcie_gene_numba")
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
DOWNSTREAM = OUT / "downstream" / "pcie_gene_space"
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


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return np.sum(a * b, axis=1) / np.maximum(denom, 1e-8)


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    weights = np.maximum(np.asarray(weights, dtype=float), 0.0)
    if float(weights.sum()) <= 0:
        return float(np.nanmean(values))
    return float(np.nansum(np.asarray(values, dtype=float) * weights) / weights.sum())


def build_nuisance_basis(latent: np.ndarray, obs: pd.DataFrame, rank: int, rng: np.random.Generator) -> np.ndarray:
    times = obs["time_point_processed"].astype(float).to_numpy()
    sample = obs["sample_name"].astype(str).to_numpy()
    control = (times == 0.0) & (sample == "HCT116_control")
    baseline = (times == 0.0) & (sample == "HCT116_0d")
    x_control = latent[control]
    x_baseline = latent[baseline]
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
        return x.astype(np.float32, copy=False)
    return (x - (x @ basis) @ basis.T).astype(np.float32, copy=False)


def load_inputs():
    adata = ad.read_h5ad(ADATA_PATH)
    with METRICS_PATH.open() as f:
        metrics = json.load(f)
    baseline = pd.read_csv(BASELINE_TABLE)
    return adata, metrics, baseline


def compute_pcie_pairs_and_gene_contrib():
    rng = np.random.default_rng(1729)
    adata, metrics, baseline = load_inputs()
    custom = metrics.get("custom_metrics", {})
    nuisance_rank = int(custom.get("pcib_nuisance_rank", 3))
    nuisance_budget = float(custom.get("pcib_nuisance_budget", 0.5))
    eta = float(metrics.get("config", {}).get("pcib", {}).get("metric_leakage_weight", 0.25))

    latent = np.asarray(adata.obsm["X_latent"], dtype=np.float32)
    metabolic = np.asarray(adata.obsm["X_metabolic_velocity"], dtype=np.float32)
    obs = adata.obs.copy()
    obs_times = obs["time_point_processed"].astype(float).to_numpy()
    split = obs["mvab_anchor_split"].astype(str).to_numpy()
    confidence = obs["mvab_anchor_confidence"].astype(float).to_numpy().astype(np.float32)

    pcs = np.asarray(adata.varm["PCs"], dtype=np.float32)
    hvg_mask = np.linalg.norm(pcs[:, : latent.shape[1]], axis=1) > 1e-10
    hvg_idx = np.flatnonzero(hvg_mask)
    hvg_genes = adata.var_names.to_numpy()[hvg_idx]
    pcs_hvg = pcs[hvg_idx, : latent.shape[1]].astype(np.float32)
    pd.DataFrame({"gene": hvg_genes}).to_csv(DOWNSTREAM / "pcie_gene_space_hvg_genes.csv", index=False)

    basis = build_nuisance_basis(latent, obs, rank=nuisance_rank, rng=rng)
    z = np.load(TRAJ_PATH)
    time_points = np.asarray(z["time_points"], dtype=float)
    observed_indices = np.asarray(z["observed_time_indices"], dtype=int)
    observed_times = np.asarray(z["observed_time_points"], dtype=float)

    pair_rows: list[dict[str, float | int | str | bool]] = []
    gene_accum = {
        "unit": np.zeros(len(hvg_idx), dtype=np.float64),
        "abs_unit": np.zeros(len(hvg_idx), dtype=np.float64),
        "gen_unit": np.zeros(len(hvg_idx), dtype=np.float64),
        "pulse_unit": np.zeros(len(hvg_idx), dtype=np.float64),
        "gen_raw": np.zeros(len(hvg_idx), dtype=np.float64),
        "pulse_raw": np.zeros(len(hvg_idx), dtype=np.float64),
        "same_sign": np.zeros(len(hvg_idx), dtype=np.float64),
    }
    time_gene_rows = []
    total_weight = 0.0

    for time_i, traj_idx in enumerate(observed_indices):
        traj_idx = int(traj_idx)
        if traj_idx + 1 >= len(time_points):
            continue
        anchor_time = float(observed_times[time_i])
        anchor_rows = np.where((obs_times == anchor_time) & (split == "heldout") & (confidence > 0))[0]
        if anchor_rows.size == 0:
            continue

        anchor_x = latent[anchor_rows]
        pulse_perp = project_perp(metabolic[anchor_rows], basis)
        anchor_conf = confidence[anchor_rows]
        valid = (np.linalg.norm(pulse_perp, axis=1) > 1e-8) & (anchor_conf > 0)
        anchor_rows = anchor_rows[valid]
        anchor_x = anchor_x[valid]
        pulse_perp = pulse_perp[valid]
        anchor_conf = anchor_conf[valid]
        if anchor_rows.size == 0:
            continue

        cur = np.asarray(z[f"points_{traj_idx:06d}"], dtype=np.float32)
        nxt = np.asarray(z[f"points_{traj_idx + 1:06d}"], dtype=np.float32)
        n = min(len(cur), len(nxt))
        cur, nxt = cur[:n], nxt[:n]
        if n > 4096:
            particle_idx = rng.choice(np.arange(n), size=4096, replace=False)
            cur_ref = cur[particle_idx]
            nxt_ref = nxt[particle_idx]
        else:
            particle_idx = np.arange(n, dtype=int)
            cur_ref = cur
            nxt_ref = nxt

        dt = max(abs(float(time_points[traj_idx + 1] - time_points[traj_idx])), 1e-8)
        gen_velocity = (nxt_ref - cur_ref) / dt
        nn = NearestNeighbors(n_neighbors=1).fit(cur_ref)
        dist, ind = nn.kneighbors(anchor_x)
        matched_local = ind[:, 0]
        matched_particle = particle_idx[matched_local]
        matched_gen_velocity = gen_velocity[matched_local]
        gen_perp = project_perp(matched_gen_velocity, basis)

        cos = cosine_rows(gen_perp, pulse_perp)
        cos_error = np.clip(1.0 - cos, 0.0, 2.0)
        nuisance_energy = np.sum((matched_gen_velocity @ basis) ** 2, axis=1) if basis.size else np.zeros(len(gen_perp))
        velocity_energy = np.maximum(np.sum(matched_gen_velocity * matched_gen_velocity, axis=1), 1e-8)
        leakage = nuisance_energy / velocity_energy
        leakage_over_budget = np.maximum(leakage - nuisance_budget, 0.0)

        gen_gene = gen_perp @ pcs_hvg.T
        pulse_gene = pulse_perp @ pcs_hvg.T
        gen_norm = np.linalg.norm(gen_gene, axis=1, keepdims=True)
        pulse_norm = np.linalg.norm(pulse_gene, axis=1, keepdims=True)
        gen_unit = gen_gene / np.maximum(gen_norm, 1e-8)
        pulse_unit = pulse_gene / np.maximum(pulse_norm, 1e-8)
        unit_contrib = gen_unit * pulse_unit
        weights = anchor_conf.astype(np.float64)
        w_col = weights.reshape(-1, 1)

        gene_accum["unit"] += np.sum(unit_contrib * w_col, axis=0)
        gene_accum["abs_unit"] += np.sum(np.abs(unit_contrib) * w_col, axis=0)
        gene_accum["gen_unit"] += np.sum(gen_unit * w_col, axis=0)
        gene_accum["pulse_unit"] += np.sum(pulse_unit * w_col, axis=0)
        gene_accum["gen_raw"] += np.sum(gen_gene * w_col, axis=0)
        gene_accum["pulse_raw"] += np.sum(pulse_gene * w_col, axis=0)
        gene_accum["same_sign"] += np.sum((np.sign(gen_gene) == np.sign(pulse_gene)) * w_col, axis=0)
        total_weight += float(weights.sum())

        denom_t = max(float(weights.sum()), 1e-8)
        time_contrib = np.sum(unit_contrib * w_col, axis=0) / denom_t
        time_abs = np.sum(np.abs(unit_contrib) * w_col, axis=0) / denom_t
        for local_j in np.argsort(-time_abs)[:80]:
            time_gene_rows.append(
                {
                    "anchor_time": anchor_time,
                    "gene": str(hvg_genes[local_j]),
                    "mean_unit_contribution": float(time_contrib[local_j]),
                    "mean_abs_unit_contribution": float(time_abs[local_j]),
                }
            )

        for j, cell_idx in enumerate(anchor_rows):
            pair_rows.append(
                {
                    "anchor_time": anchor_time,
                    "anchor_index": int(cell_idx),
                    "anchor_cell": str(adata.obs_names[cell_idx]),
                    "matched_particle_index": int(matched_particle[j]),
                    "anchor_distance": float(dist[j, 0]),
                    "confidence": float(anchor_conf[j]),
                    "cosine": float(cos[j]),
                    "cosine_error": float(cos_error[j]),
                    "nuisance_leakage": float(leakage[j]),
                    "leakage_over_budget": float(leakage_over_budget[j]),
                    "positive_cosine": bool(cos[j] > 0),
                    "gen_gene_norm": float(gen_norm[j, 0]),
                    "pulse_gene_norm": float(pulse_norm[j, 0]),
                }
            )
        print(f"[pcie] anchor_time={anchor_time:g} pairs={len(anchor_rows)}", flush=True)

    pair_df = pd.DataFrame(pair_rows)
    denom = max(total_weight, 1e-8)
    gene_df = pd.DataFrame(
        {
            "gene": hvg_genes,
            "mean_unit_contribution": gene_accum["unit"] / denom,
            "mean_abs_unit_contribution": gene_accum["abs_unit"] / denom,
            "mean_generated_unit_velocity": gene_accum["gen_unit"] / denom,
            "mean_pulse_unit_velocity": gene_accum["pulse_unit"] / denom,
            "mean_generated_gene_velocity": gene_accum["gen_raw"] / denom,
            "mean_pulse_gene_velocity": gene_accum["pulse_raw"] / denom,
            "same_sign_fraction": gene_accum["same_sign"] / denom,
        }
    )
    gene_df["direction_class"] = np.where(gene_df["mean_unit_contribution"] >= 0, "supports_alignment", "opposes_alignment")
    gene_df["biology_note"] = [biology_note(g) for g in gene_df["gene"]]
    gene_df = gene_df.sort_values("mean_abs_unit_contribution", ascending=False)
    time_gene_df = pd.DataFrame(time_gene_rows).sort_values(["anchor_time", "mean_abs_unit_contribution"], ascending=[True, False])

    weighted_cosine_error = weighted_mean(pair_df["cosine_error"].to_numpy(), pair_df["confidence"].to_numpy())
    weighted_leakage = weighted_mean(pair_df["leakage_over_budget"].to_numpy(), pair_df["confidence"].to_numpy())
    recomputed_pcie = weighted_cosine_error + eta * weighted_leakage
    summary = {
        "locked_pcie": float(custom.get("PCIE", custom.get("pcib_pulse_control_identifiability_error", np.nan))),
        "locked_cosine_error": float(custom.get("pcib_pcie_cosine_error", np.nan)),
        "locked_leakage_over_budget": float(custom.get("pcib_pcie_nuisance_leakage_above_budget", np.nan)),
        "recomputed_gene_decomp_cosine_error": float(weighted_cosine_error),
        "recomputed_gene_decomp_leakage_over_budget": float(weighted_leakage),
        "recomputed_gene_decomp_pcie": float(recomputed_pcie),
        "eta": eta,
        "nuisance_budget": nuisance_budget,
        "nuisance_rank": nuisance_rank,
        "heldout_anchor_pairs": int(len(pair_df)),
        "anchor_times_evaluated": sorted(float(x) for x in pair_df["anchor_time"].unique()),
        "pca_supported_gene_count": int(len(hvg_idx)),
        "pulsebridge_pcie": float(baseline.loc[baseline["algorithm"] == "PulseBridge", "pcie"].iloc[0]),
        "wfr_fm_pcie": float(baseline.loc[baseline["algorithm"] == "WFR-FM", "pcie"].iloc[0]),
    }
    return pair_df, gene_df, time_gene_df, summary


def biology_note(gene: str) -> str:
    marker_roles = {
        "SAT1": "polyamine catabolism / metabolic stress",
        "B2M": "antigen presentation / immune surface",
        "HLA-A": "antigen presentation",
        "HLA-B": "antigen presentation",
        "IFITM3": "interferon/viral-mimicry membrane readout",
        "LGALS1": "adhesion/adaptive stress",
        "PCNA": "DNA replication/proliferation",
        "DKK1": "tumor-suppressor-like source-paper marker",
        "GAS5": "tumor-suppressor-like source-paper marker",
        "EREG": "EGFR ligand / oncogene-like source-paper marker",
        "AREG": "EGFR ligand / growth-factor signaling",
        "MALAT1": "lncRNA source-paper marker",
    }
    gene = str(gene)
    if gene in marker_roles:
        return marker_roles[gene]
    if gene.startswith("MT-"):
        return "mitochondrial transcript / metabolic remodeling"
    if gene.startswith("HSP"):
        return "proteostasis / heat-shock stress"
    if gene.startswith("RPL") or gene.startswith("RPS"):
        return "ribosome / translation"
    return ""


def draw_figure(pair_df: pd.DataFrame, gene_df: pd.DataFrame, summary: dict) -> None:
    fig = plt.figure(figsize=(15.5, 11.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.1], width_ratios=[1.05, 1.0], hspace=0.34, wspace=0.28)
    ax_hist = fig.add_subplot(gs[0, 0])
    ax_day = fig.add_subplot(gs[0, 1])
    ax_bar = fig.add_subplot(gs[1, 0])
    ax_scatter = fig.add_subplot(gs[1, 1])

    palette = {0.0: "#31572C", 1.0: "#4F86C6", 2.0: "#F28E2B", 3.0: "#B23A48"}
    bins = np.linspace(-1, 1, 45)
    for t, sub in pair_df.groupby("anchor_time"):
        ax_hist.hist(
            sub["cosine"],
            bins=bins,
            density=True,
            histtype="step",
            lw=2.0,
            color=palette.get(float(t), "#777777"),
            label=f"day {int(t)} anchors",
        )
    ax_hist.axvline(0, color="#333333", lw=0.9)
    ax_hist.set_xlabel("cosine: generated tangent vs heldout pulse")
    ax_hist.set_ylabel("density")
    ax_hist.set_title("A. PCIE-matched heldout anchors")
    ax_hist.legend(frameon=False, fontsize=8)
    ax_hist.text(
        0.02,
        0.96,
        f"weighted mean cosine {1.0 - summary['recomputed_gene_decomp_cosine_error']:.3f}\n"
        f"PCIE {summary['locked_pcie']:.3f}",
        transform=ax_hist.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="#333333",
    )

    by_day = (
        pair_df.groupby("anchor_time")
        .agg(
            mean_cosine=("cosine", "mean"),
            median_cosine=("cosine", "median"),
            positive_fraction=("cosine", lambda s: float((s > 0).mean())),
            anchor_count=("cosine", "size"),
            mean_error=("cosine_error", "mean"),
        )
        .reset_index()
        .sort_values("anchor_time")
    )
    ax_day.axhline(0, color="#333333", lw=0.8)
    ax_day.plot(by_day["anchor_time"], by_day["mean_cosine"], marker="o", lw=2.2, color="#146C94")
    for _, row in by_day.iterrows():
        ax_day.text(
            row["anchor_time"],
            row["mean_cosine"] + 0.035,
            f"{row['positive_fraction']:.0%}+",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#146C94",
        )
    ax_day.set_xticks([0, 1, 2])
    ax_day.set_ylim(-0.05, max(0.52, float(by_day["mean_cosine"].max()) + 0.12))
    ax_day.set_xlabel("heldout anchor day used by PCIE")
    ax_day.set_ylabel("mean cosine")
    ax_day.set_title("B. PCIE does not use a day-3 forward anchor")
    ax_day.text(
        0.03,
        0.94,
        "PCIE compares generated t->t+dt tangents to heldout anchors.\n"
        "The last observed day has no forward finite-difference target.",
        transform=ax_day.transAxes,
        ha="left",
        va="top",
        fontsize=8.4,
        color="#333333",
        bbox={"boxstyle": "round,pad=0.26", "fc": "white", "ec": "#DDDDDD", "alpha": 0.95},
    )

    top_bar = gene_df.head(20).iloc[::-1].copy()
    colors = np.where(top_bar["mean_unit_contribution"] >= 0, "#146C94", "#B23A48")
    ax_bar.barh(top_bar["gene"], top_bar["mean_unit_contribution"], color=colors)
    ax_bar.axvline(0, color="#333333", lw=0.8)
    ax_bar.set_xlabel("mean contribution to PCIE cosine")
    ax_bar.set_title("C. Gene-level decomposition of the PCIE direction")

    top_scatter = gene_df.head(90).copy()
    point_colors = np.where(top_scatter["mean_unit_contribution"] >= 0, "#146C94", "#B23A48")
    sizes = 32 + 2600 * top_scatter["mean_abs_unit_contribution"] / max(float(top_scatter["mean_abs_unit_contribution"].max()), 1e-8)
    ax_scatter.scatter(
        top_scatter["mean_pulse_unit_velocity"],
        top_scatter["mean_generated_unit_velocity"],
        s=sizes,
        c=point_colors,
        alpha=0.78,
        edgecolor="white",
        linewidth=0.7,
    )
    lim = float(
        np.nanmax(
            np.abs(
                np.r_[
                    top_scatter["mean_pulse_unit_velocity"].to_numpy(),
                    top_scatter["mean_generated_unit_velocity"].to_numpy(),
                ]
            )
        )
    )
    lim = max(lim * 1.1, 0.002)
    ax_scatter.plot([-lim, lim], [-lim, lim], color="#777777", lw=0.9, ls="--")
    ax_scatter.axhline(0, color="#DDDDDD", lw=0.8)
    ax_scatter.axvline(0, color="#DDDDDD", lw=0.8)
    ax_scatter.set_xlim(-lim, lim)
    ax_scatter.set_ylim(-lim, lim)
    ax_scatter.set_xlabel("mean heldout pulse unit velocity")
    ax_scatter.set_ylabel("mean generated tangent unit velocity")
    ax_scatter.set_title("D. Matched generated-pulse gene directions")
    label_genes = {"SAT1", "MT-RNR2", "MALAT1", "RPL13", "RPS2", "MT-CO1", "MT-CO2", "ACTG1", "B2M", "LGALS1"}
    for _, row in top_scatter[top_scatter["gene"].isin(label_genes)].iterrows():
        ax_scatter.text(
            row["mean_pulse_unit_velocity"],
            row["mean_generated_unit_velocity"],
            str(row["gene"]),
            fontsize=7.6,
            ha="center",
            va="center",
            color="#222222",
        )

    fig.suptitle(
        "Gene-space explanation of the locked PulseBridge PCIE calculation",
        fontsize=15,
        fontweight="bold",
        y=0.985,
    )
    path = FIG / "story_pcie_gene_space_decomposition.png"
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    DOWNSTREAM.mkdir(parents=True, exist_ok=True)
    pair_df, gene_df, time_gene_df, summary = compute_pcie_pairs_and_gene_contrib()

    pair_table = TAB / "pcie_gene_space_anchor_pairs.csv"
    gene_table = TAB / "pcie_gene_space_top_genes.csv"
    time_gene_table = TAB / "pcie_gene_space_top_genes_by_anchor_time.csv"
    manifest_path = OUT / "pcie_gene_space_manifest.json"
    pair_df.to_csv(pair_table, index=False)
    gene_df.to_csv(gene_table, index=False)
    time_gene_df.to_csv(time_gene_table, index=False)
    draw_figure(pair_df, gene_df, summary)

    summary["outputs"] = {
        "pair_table": str(pair_table),
        "gene_table": str(gene_table),
        "time_gene_table": str(time_gene_table),
        "figure": str(FIG / "story_pcie_gene_space_decomposition.png"),
        "hvg_genes": str(DOWNSTREAM / "pcie_gene_space_hvg_genes.csv"),
    }
    summary["method"] = {
        "generated_tangent": "(points[t+dt] - points[t]) / dt from evaluation_trajectory.npz",
        "matching": "nearest generated particle to each heldout metabolic anchor at the same observed anchor time",
        "projection": "both generated tangent and metabolic pulse are projected away from the train-derived nuisance basis",
        "gene_decomposition": "project nuisance-removed latent vectors to PCA-supported HVG gene coefficients and decompose the per-anchor cosine as a sum of normalized gene products",
    }
    manifest_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
