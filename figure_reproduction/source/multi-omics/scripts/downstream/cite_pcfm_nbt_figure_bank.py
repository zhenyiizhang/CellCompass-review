#!/usr/bin/env python3
"""Generate an NBT-style figure bank for the CITE-PCFM case study.

This script intentionally makes many article-thinking drafts rather than one
single summary panel. It has two tracks:

1. biology_deep_dive: downstream biological analyses for CITE-PCFM.
2. agent_lifecycle: how CellCompass searched, failed, revised and locked the
   final algorithm.

The figures are meant for selection and refinement. They avoid using heatmaps
as the main visual device.
"""

from __future__ import annotations

import json
import math
import re
import textwrap
from collections import Counter
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.path import Path as MplPath
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, PathPatch, Rectangle, Wedge
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


sns.set_theme(style="white", context="talk")
RNG = np.random.default_rng(20260527)

ROOT = Path("/Users/zhenyizhang/Documents/GitHub/CytoBridge-agent-runtime-v2-20260309/multi-omics")
DATA_PATH = Path("/Users/zhenyizhang/.cellcompass/algorithm_benchmarks/datasets/gse305370_citeseq_stage2_10k/data.h5ad")
CELL_ROOT = Path("/Users/zhenyizhang/.cellcompass")
GT_DIR = ROOT / "downstream/cite_pcfm_rna_to_protein_gt"
TABLE_DIR = GT_DIR / "tables"
OUT_ROOT = ROOT / "figures/nbt_cite_pcfm_figure_bank"
BIO_DIR = OUT_ROOT / "biology_deep_dive"
LIFE_DIR = OUT_ROOT / "agent_lifecycle"
NOTE_DIR = ROOT / "downstream/nbt_cite_pcfm_figure_bank"
CATALOG_PATH = NOTE_DIR / "figure_catalog.md"
MANIFEST_PATH = NOTE_DIR / "manifest.json"

RNA_DIM = 50
PROT_DIM = 30

METHOD_COLORS = {
    "CITE-PCFM": "#B73A3A",
    "CRUFM": "#2C7FB8",
    "balanced OT-CFM": "#41AB5D",
}
PHENO_COLORS = {
    "HSC": "#3C5488",
    "MasP": "#E64B35",
    "EryP": "#F39B2F",
    "MoP": "#00A087",
    "MkP": "#7E6148",
    "NeuP": "#8491B4",
}
ALG_COLORS = {
    "cite_vgfm": "#94A3B8",
    "cite_bfm": "#64748B",
    "cite_bfm_balanced": "#64748B",
    "cite_factorized_fm": "#A78BFA",
    "cite_residual_sb_fm": "#F59E0B",
    "cite_paired_coord_fm": "#B73A3A",
}

TRAJECTORIES = {
    "CITE-PCFM": ROOT / "training_runs/20260526T174204321821Z__final__cite_paired_coord_fm__campaign_20260-trial_20260526-gse305370_ci/artifacts/evaluation_trajectory.npz",
    "CRUFM": ROOT / "training_runs/20260525T220649168798Z__pilot__crufm__campaign_20260-stage2_c-baseline-crufm-gse305370_ci/artifacts/evaluation_trajectory.npz",
    "balanced OT-CFM": ROOT / "training_runs/20260525T215826526439Z__pilot__balanced_ot_cfm__campaign_20260-stage2_c-baseline-balanced_ot_-gse305370_ci/artifacts/evaluation_trajectory.npz",
}

FIGURES: list[dict] = []


def ensure_dirs() -> None:
    for path in [BIO_DIR, LIFE_DIR, NOTE_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def savefig(fig: plt.Figure, out_dir: Path, name: str, title: str, track: str, question: str) -> Path:
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    FIGURES.append(
        {
            "track": track,
            "name": name,
            "title": title,
            "question": question,
            "png": str(path),
            "pdf": str(path.with_suffix(".pdf")),
        }
    )
    return path


def add_panel(ax, label: str) -> None:
    ax.text(-0.08, 1.08, label, transform=ax.transAxes, fontsize=18, weight="bold", va="top")


def rounded_box(ax, xy, w, h, text, fc="#F8FAFC", ec="#CBD5E1", size=10, weight="normal", color="#111827"):
    box = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.035",
        linewidth=1.2,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=size, weight=weight, color=color, wrap=True)
    return box


def curved_arrow(ax, p0, p1, color="#64748B", lw=1.8, rad=0.15, alpha=1.0):
    patch = FancyArrowPatch(
        p0,
        p1,
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle="Simple,head_width=7,head_length=9",
        linewidth=lw,
        color=color,
        alpha=alpha,
    )
    ax.add_patch(patch)
    return patch


def safe_json(path: Path) -> dict:
    try:
        with path.open() as f:
            return json.load(f)
    except Exception:
        return {}


def load_tables() -> dict[str, pd.DataFrame]:
    return {p.stem: pd.read_csv(p) for p in sorted(TABLE_DIR.glob("*.csv"))}


def load_traj(path: Path) -> dict:
    z = np.load(path)
    time_points = np.asarray(z["time_points"], dtype=float)
    observed_times = np.asarray(z.get("observed_time_points", np.array([0, 1, 2, 3, 4], dtype=float)), dtype=float)
    pts = {}
    weights = {}
    for i, t in enumerate(time_points):
        pk = f"points_{i:06d}"
        wk = f"weights_{i:06d}"
        if pk in z.files:
            pts[float(t)] = np.asarray(z[pk], dtype=np.float32)
        if wk in z.files:
            weights[float(t)] = np.asarray(z[wk], dtype=np.float32)
    return {"time_points": time_points, "observed_times": observed_times, "points": pts, "weights": weights}


def nearest_key(keys, t: float) -> float:
    arr = np.asarray(list(keys), dtype=float)
    return float(arr[np.argmin(np.abs(arr - float(t)))])


def weighted_sample(n: int, max_n: int, weights=None) -> np.ndarray:
    if n <= max_n:
        return np.arange(n)
    p = None
    if weights is not None:
        w = np.asarray(weights, dtype=float).reshape(-1)
        w = np.clip(w, 0, np.inf)
        if len(w) == n and np.isfinite(w).all() and w.sum() > 0:
            p = w / w.sum()
    return np.sort(RNG.choice(n, size=max_n, replace=False, p=p))


def normalize_higher_better(df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in value_cols:
        lo, hi = out[col].min(), out[col].max()
        out[col] = 0.5 if hi == lo else 1.0 - (out[col] - lo) / (hi - lo)
    return out


def collect_lifecycle() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    algs = ["cite_vgfm", "cite_bfm", "cite_bfm_balanced", "cite_factorized_fm", "cite_residual_sb_fm", "cite_paired_coord_fm"]
    trial_rows = []
    run_rows = []
    decision_rows = []

    for alg in algs:
        reg = CELL_ROOT / "training_algorithms" / alg / "registry"
        dec_path = reg / "decisions.jsonl"
        if dec_path.exists():
            for line in dec_path.read_text(errors="ignore").splitlines():
                if not line.strip():
                    continue
                d = json.loads(line)
                decision_rows.append(
                    {
                        "algorithm": alg,
                        "at": d.get("at", ""),
                        "phase": d.get("phase", ""),
                        "decision": d.get("decision", ""),
                        "status": d.get("status", ""),
                        "rationale": d.get("rationale", ""),
                    }
                )

        for p in sorted((CELL_ROOT / "algorithm_campaigns" / alg).glob("campaign_*/trials/*.json")):
            d = safe_json(p)
            m = d.get("metrics_summary", {}) or {}
            cm = m.get("custom_metrics", {}) or {}
            primary_metric = m.get("primary_metric", "")
            primary_value = m.get("primary_value", np.nan)
            protein_energy = cm.get("protein_energy", np.nan)
            if pd.isna(protein_energy) and primary_metric == "protein_energy":
                protein_energy = primary_value
            trial_rows.append(
                {
                    "algorithm": alg,
                    "trial_id": d.get("trial_id", p.stem),
                    "campaign_id": d.get("campaign_id", ""),
                    "stage": d.get("stage", ""),
                    "decision": d.get("decision", "unknown"),
                    "created_at": d.get("created_at", ""),
                    "completed_at": d.get("completed_at", d.get("updated_at", "")),
                    "primary_metric": primary_metric,
                    "primary_value": primary_value,
                    "secondary_metric": m.get("secondary_metric", ""),
                    "secondary_value": m.get("secondary_value", np.nan),
                    "protein_energy": protein_energy,
                    "w1_mean": m.get("w1_mean", np.nan),
                    "tmv_mean": m.get("tmv_mean", np.nan),
                    "decision_reasons": "; ".join(d.get("decision_reasons", []) or []),
                    "run_id": (d.get("run_ids") or [""])[0],
                }
            )

        for p in sorted((reg / "runs").glob("*.json")):
            d = safe_json(p)
            data_path = d.get("data_path", "")
            dataset_id = Path(data_path).parent.name if data_path else ""
            metrics_path = Path(d.get("metrics_path", ""))
            md = safe_json(metrics_path) if metrics_path.exists() else {}
            cm = md.get("custom_metrics", {}) or {}
            cfg = md.get("config", {}) or {}
            cite_cfg = cfg.get("cite_pcfm", {}) or {}
            train = md.get("training_time_budget", {}) or md.get("training_summary", {}).get("time_budget", {}) or {}
            run_rows.append(
                {
                    "algorithm": alg,
                    "run_id": d.get("run_id", p.stem),
                    "decision": d.get("decision", d.get("campaign_trial_decision", "")),
                    "decision_reason": d.get("decision_reason", ""),
                    "created_at": d.get("created_at", ""),
                    "completed_at": d.get("completed_at", ""),
                    "stage": d.get("stage", ""),
                    "dataset_id": dataset_id,
                    "w1_mean": d.get("w1_mean", np.nan),
                    "tmv_mean": d.get("tmv_mean", np.nan),
                    "protein_energy": cm.get("protein_energy", np.nan),
                    "rna_energy_guardrail": cm.get("rna_energy_guardrail", np.nan),
                    "paired_cov_error": cm.get("paired_cov_error", np.nan),
                    "runtime_sec": md.get("runtime_sec", np.nan),
                    "elapsed_sec": train.get("elapsed_sec", np.nan),
                    "completed_epochs": train.get("completed_epochs", np.nan),
                    "lambda_p": cite_cfg.get("lambda_p", np.nan),
                    "alpha_protein": cite_cfg.get("alpha_protein", np.nan),
                    "endpoint_protein_weight": cite_cfg.get("endpoint_protein_weight", np.nan),
                    "protein_head_hidden_dim": cite_cfg.get("protein_head_hidden_dim", np.nan),
                    "protein_time_weight_width": cite_cfg.get("protein_time_weight_width", np.nan),
                }
            )

    trial_df = pd.DataFrame(trial_rows)
    run_df = pd.DataFrame(run_rows)
    decision_df = pd.DataFrame(decision_rows)
    for df in [trial_df, run_df, decision_df]:
        for col in ["created_at", "completed_at", "at"]:
            if col in df:
                df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    if len(trial_df):
        trial_df = trial_df.sort_values("created_at").reset_index(drop=True)
        trial_df["trial_index"] = np.arange(1, len(trial_df) + 1)
    if len(run_df):
        run_df = run_df.sort_values("created_at").reset_index(drop=True)
        run_df["run_index"] = np.arange(1, len(run_df) + 1)
    return trial_df, run_df, decision_df


def biology_01_mechanism_card() -> None:
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.02, 0.96, "CITE-PCFM", fontsize=22, weight="bold", color="#7F1D1D")
    ax.text(0.02, 0.90, "RNA trajectory with a learned CITE protein phenotype readout", fontsize=16, weight="bold")
    rounded_box(ax, (0.05, 0.58), 0.20, 0.16, "RNA temporal\nsnapshots", "#E8F2FF", "#93C5FD", 13, "bold")
    rounded_box(ax, (0.05, 0.32), 0.20, 0.16, "paired CITE\nprotein PCs", "#FFF7ED", "#FDBA74", 13, "bold")
    rounded_box(ax, (0.36, 0.55), 0.24, 0.20, "paired-coordinate\ntransport geometry", "#F8FAFC", "#64748B", 13, "bold")
    rounded_box(ax, (0.70, 0.60), 0.22, 0.14, "RNA state\ntrajectory", "#FEE2E2", "#B73A3A", 13, "bold")
    rounded_box(ax, (0.70, 0.34), 0.22, 0.16, "protein readout\n$h_{\\eta}(R_t,t)$", "#FFF7ED", "#F97316", 13, "bold")
    for p0, p1 in [((0.25, 0.66), (0.36, 0.66)), ((0.25, 0.40), (0.36, 0.61)), ((0.60, 0.65), (0.70, 0.67)), ((0.81, 0.60), (0.81, 0.50))]:
        curved_arrow(ax, p0, p1)
    ax.text(0.36, 0.26, "Article question: does protein-paired geometry produce better target protein dynamics?", fontsize=13, weight="bold")
    ax.text(0.36, 0.20, "The figures below test this against observed target distributions and joint RNA-protein structure.", fontsize=11, color="#475569")
    savefig(fig, BIO_DIR, "bio01_mechanism_card", "Mechanism card", "biology_deep_dive", "What biological capability is being claimed?")


def biology_02_to_12(tables: dict[str, pd.DataFrame]) -> None:
    per = tables["per_time"].copy()
    pheno = tables["phenotype"].copy()
    centroid = tables["centroid"].copy()
    nearest = tables["nearest"].copy()
    variance = tables["variance"].copy()
    score = tables["summary_scorecard_raw"].copy()

    # 02 protein energy time trace
    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    for method, color in METHOD_COLORS.items():
        d = per[per.method == method]
        ax.plot(d.time, d.protein_energy, marker="o", lw=2.5, color=color, label=method)
    ax.set_title("Protein-coordinate energy over target time")
    ax.set_xlabel("target time")
    ax.set_ylabel("energy distance, lower is better")
    ax.legend(frameon=False)
    savefig(fig, BIO_DIR, "bio02_protein_energy_temporal_trace", "Protein energy over time", "biology_deep_dive", "Where does the protein advantage occur in time?")

    # 03 RNA guardrail
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for method, color in METHOD_COLORS.items():
        d = per[per.method == method]
        ax.scatter(d.rna_energy, d.protein_energy, s=120, color=color, label=method, alpha=0.8, edgecolor="white")
        for _, r in d.iterrows():
            ax.text(r.rna_energy, r.protein_energy, f"t{int(r.time)}", fontsize=8)
    ax.set_title("Protein gain versus RNA-state guardrail")
    ax.set_xlabel("RNA energy")
    ax.set_ylabel("protein energy")
    ax.legend(frameon=False)
    savefig(fig, BIO_DIR, "bio03_rna_guardrail_pareto", "RNA guardrail Pareto", "biology_deep_dive", "Is protein improvement bought by breaking RNA dynamics?")

    # 04 covariance trace
    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    for method, color in METHOD_COLORS.items():
        d = per[per.method == method]
        ax.plot(d.time, d.rna_protein_cov_error, marker="o", lw=2.3, color=color, label=method)
    ax.set_title("RNA-protein covariance preservation")
    ax.set_xlabel("target time")
    ax.set_ylabel("relative covariance error")
    ax.legend(frameon=False)
    savefig(fig, BIO_DIR, "bio04_covariance_preservation_trace", "Covariance preservation", "biology_deep_dive", "Does the method preserve joint RNA-protein structure?")

    # 05 nearest ECDF
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for method, color in METHOD_COLORS.items():
        vals = nearest[(nearest.method == method) & (nearest.time > 0)]["nearest_distance"].dropna().to_numpy()
        vals = np.sort(vals)
        y = np.arange(1, len(vals) + 1) / max(1, len(vals))
        ax.plot(vals, y, lw=2.4, color=color, label=method)
    ax.set_title("Generated protein states near observed target manifold")
    ax.set_xlabel("nearest observed protein distance")
    ax.set_ylabel("cumulative fraction")
    ax.legend(frameon=False)
    savefig(fig, BIO_DIR, "bio05_nearest_target_ecdf", "Nearest target ECDF", "biology_deep_dive", "Do generated protein states sit on the observed protein manifold?")

    # 06 phenotype river
    phenotypes = [p for p in ["HSC", "MasP", "EryP", "MoP", "MkP", "NeuP"] if p in set(pheno.phenotype)]
    panels = ["Observed target", "CITE-PCFM", "CRUFM", "balanced OT-CFM"]
    fig, axes = plt.subplots(4, 1, figsize=(12, 8.2), sharex=True)
    for ax, panel in zip(axes, panels):
        times = sorted(pheno.time.unique())
        vals = []
        for phenotype in phenotypes:
            d = pheno[pheno.phenotype == phenotype]
            if panel == "Observed target":
                s = d.groupby("time")["observed_fraction"].mean().reindex(times).fillna(0).to_numpy()
            else:
                s = d[d.method == panel].set_index("time")["generated_fraction"].reindex(times).fillna(0).to_numpy()
            vals.append(s)
        ax.stackplot(times, vals, colors=[PHENO_COLORS[p] for p in phenotypes], labels=phenotypes, alpha=0.92)
        ax.text(-0.08, 0.5, panel, transform=ax.transAxes, ha="right", va="center", fontsize=11, weight="bold")
        ax.set_yticks([])
        ax.spines[["top", "right", "left"]].set_visible(False)
    axes[0].legend(ncol=6, frameon=False, bbox_to_anchor=(1.0, 1.35), loc="upper right", fontsize=9)
    axes[-1].set_xlabel("time")
    fig.suptitle("Protein phenotype river against observed targets", fontsize=17, weight="bold", y=1.02)
    savefig(fig, BIO_DIR, "bio06_protein_phenotype_river", "Phenotype river", "biology_deep_dive", "Does the generated trajectory reproduce target protein phenotype flow?")

    # 07 phenotype error lollipops
    d = pheno[pheno.time > 0].groupby(["method", "phenotype"], as_index=False)["fraction_abs_error"].mean()
    fig, ax = plt.subplots(figsize=(9, 5.2))
    y_positions = np.arange(len(phenotypes))
    for i, method in enumerate(METHOD_COLORS):
        sub = d[d.method == method].set_index("phenotype").reindex(phenotypes)
        x = sub["fraction_abs_error"].to_numpy()
        ax.scatter(x, y_positions + (i - 1) * 0.17, s=80, color=METHOD_COLORS[method], label=method)
        for xx, yy in zip(x, y_positions + (i - 1) * 0.17):
            ax.plot([0, xx], [yy, yy], color=METHOD_COLORS[method], alpha=0.35, lw=2)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(phenotypes)
    ax.set_xlabel("mean absolute phenotype-fraction error")
    ax.set_title("Which protein phenotypes are matched or missed?")
    ax.legend(frameon=False)
    savefig(fig, BIO_DIR, "bio07_phenotype_error_lollipops", "Phenotype error lollipops", "biology_deep_dive", "Which biological phenotypes still expose limitations?")

    # 08 centroid displacement constellation
    d = centroid[centroid.time > 0].groupby(["method", "phenotype"], as_index=False)["centroid_displacement"].mean()
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    for method, color in METHOD_COLORS.items():
        sub = d[d.method == method]
        ax.scatter(sub.phenotype, sub.centroid_displacement, s=120, color=color, alpha=0.8, label=method, edgecolor="white")
        ax.plot(sub.phenotype, sub.centroid_displacement, color=color, alpha=0.45, lw=2)
    ax.set_title("Protein phenotype centroids relative to observed targets")
    ax.set_ylabel("mean centroid displacement")
    ax.set_xlabel("")
    ax.legend(frameon=False)
    savefig(fig, BIO_DIR, "bio08_centroid_constellation", "Centroid constellation", "biology_deep_dive", "Which protein phenotype centers are geometrically aligned?")

    # 09 variance trace
    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    for method, color in METHOD_COLORS.items():
        d = variance[variance.method == method]
        ax.plot(d.time, d.protein_variance_relative_l1, marker="o", lw=2.4, color=color, label=method)
    ax.set_title("Protein diversity preservation")
    ax.set_xlabel("target time")
    ax.set_ylabel("relative variance error")
    ax.legend(frameon=False)
    savefig(fig, BIO_DIR, "bio09_protein_diversity_trace", "Protein diversity trace", "biology_deep_dive", "Does the method collapse protein heterogeneity?")

    # 10 raw score small multiples as ranked dots
    metrics = ["protein_energy", "phenotype_fraction_error", "centroid_displacement", "covariance_error", "nearest_distance", "variance_error"]
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.0))
    for ax, metric in zip(axes.flat, metrics):
        s = score.sort_values(metric)
        ax.scatter(s[metric], s.method, s=90, c=[METHOD_COLORS[m] for m in s.method], edgecolor="white")
        ax.set_title(metric.replace("_", " "))
        ax.set_xlabel("lower is better")
        ax.set_ylabel("")
    fig.suptitle("Six biological score views, without hiding tradeoffs", y=1.02, weight="bold")
    savefig(fig, BIO_DIR, "bio10_ranked_score_views", "Ranked score views", "biology_deep_dive", "Which evidence dimensions support or weaken the claim?")

    # 11 improvement waterfall
    base = score.set_index("method")
    metrics2 = ["protein_energy", "covariance_error", "nearest_distance", "variance_error", "phenotype_fraction_error"]
    vals = []
    for metric in metrics2:
        ref = float(base.loc["CRUFM", metric])
        cur = float(base.loc["CITE-PCFM", metric])
        vals.append(100 * (ref - cur) / max(ref, 1e-8))
    fig, ax = plt.subplots(figsize=(9.0, 4.5))
    colors = ["#B73A3A" if v >= 0 else "#64748B" for v in vals]
    ax.bar(np.arange(len(metrics2)), vals, color=colors)
    ax.axhline(0, color="#111827", lw=1)
    ax.set_xticks(np.arange(len(metrics2)))
    ax.set_xticklabels([m.replace("_", "\n") for m in metrics2], fontsize=9)
    ax.set_ylabel("% improvement vs CRUFM")
    ax.set_title("CITE-PCFM gains and regressions versus a strong baseline")
    savefig(fig, BIO_DIR, "bio11_crufm_improvement_waterfall", "Improvement waterfall", "biology_deep_dive", "Where is the claim strongest relative to CRUFM?")

    # 12 timepoint winners
    d = per[per.time > 0].copy()
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    for i, t in enumerate(sorted(d.time.unique())):
        sub = d[d.time == t].sort_values("protein_energy")
        for j, (_, r) in enumerate(sub.iterrows()):
            ax.scatter(t, j, s=200 - 40 * j, color=METHOD_COLORS[r.method], edgecolor="white")
            ax.text(t + 0.04, j, r.method, va="center", fontsize=9)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["best", "second", "third"])
    ax.set_xlabel("target time")
    ax.set_title("Protein-energy winner at each future time")
    savefig(fig, BIO_DIR, "bio12_timepoint_winners", "Timepoint winners", "biology_deep_dive", "At which time points is CITE-PCFM the leading method?")


def biology_landscape_figures(tables: dict[str, pd.DataFrame]) -> None:
    adata = ad.read_h5ad(DATA_PATH)
    obs = adata.obs.copy()
    obs["time_point_processed"] = obs["time_point_processed"].astype(float)
    protein = np.asarray(adata.obsm["X_protein_pca"], dtype=np.float32)
    trajectories = {m: load_traj(p) for m, p in TRAJECTORIES.items()}

    # 13 target landscape overlays
    times = [1.0, 2.0, 3.0, 4.0]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2), sharex=True, sharey=True)
    for ax, t in zip(axes, times):
        mask = np.asarray(obs["time_point_processed"] == t)
        oi = weighted_sample(int(mask.sum()), 900)
        obs_prot = protein[mask][oi, :2]
        sns.kdeplot(x=obs_prot[:, 0], y=obs_prot[:, 1], levels=7, color="#111827", linewidths=1.0, ax=ax, alpha=0.45)
        ax.scatter(obs_prot[:, 0], obs_prot[:, 1], s=5, color="#94A3B8", alpha=0.10, linewidths=0)
        for method, traj in trajectories.items():
            tt = nearest_key(traj["points"].keys(), t)
            pts = traj["points"][tt]
            idx = weighted_sample(pts.shape[0], 350, traj["weights"].get(tt))
            gp = pts[idx, RNA_DIM : RNA_DIM + 2]
            ax.scatter(gp[:, 0], gp[:, 1], s=8, color=METHOD_COLORS[method], alpha=0.28, linewidths=0, label=method if t == 1.0 else None)
            ax.scatter(gp[:, 0].mean(), gp[:, 1].mean(), s=70, color=METHOD_COLORS[method], edgecolor="white", linewidth=1.2)
        ax.set_title(f"target time {int(t)}")
        ax.set_xlabel("protein PC1")
    axes[0].set_ylabel("protein PC2")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Generated protein states against observed target landscapes", y=1.05, weight="bold")
    savefig(fig, BIO_DIR, "bio13_observed_target_landscape_overlays", "Target landscape overlays", "biology_deep_dive", "Do generated protein states occupy observed target landscapes?")

    # 14 protein drift vectors
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    for method, traj in trajectories.items():
        means = []
        for t in [0.0, 1.0, 2.0, 3.0, 4.0]:
            tt = nearest_key(traj["points"].keys(), t)
            pts = traj["points"][tt]
            idx = weighted_sample(pts.shape[0], 700, traj["weights"].get(tt))
            means.append(pts[idx, RNA_DIM : RNA_DIM + 2].mean(axis=0))
        means = np.asarray(means)
        ax.plot(means[:, 0], means[:, 1], lw=2.5, marker="o", color=METHOD_COLORS[method], label=method)
        for a, b in zip(means[:-1], means[1:]):
            ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", lw=1.8, color=METHOD_COLORS[method]))
    obs_means = []
    for t in [0, 1, 2, 3, 4]:
        mask = np.asarray(obs["time_point_processed"] == float(t))
        obs_means.append(protein[mask, :2].mean(axis=0))
    obs_means = np.asarray(obs_means)
    ax.plot(obs_means[:, 0], obs_means[:, 1], lw=3, marker="s", color="#111827", label="observed target")
    ax.set_xlabel("protein PC1")
    ax.set_ylabel("protein PC2")
    ax.set_title("Mean protein-state drift through time")
    ax.legend(frameon=False)
    savefig(fig, BIO_DIR, "bio14_mean_protein_drift_vectors", "Mean protein drift", "biology_deep_dive", "Does the generated protein trajectory drift in the same direction as observed targets?")

    # 15 target ring density
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    per = tables["per_time"]
    for method, color in METHOD_COLORS.items():
        d = per[(per.method == method) & (per.time > 0)]
        ax.plot(d.time, d.nearest_observed_protein_distance_mean, lw=2.4, marker="o", color=color, label=method)
        ax.fill_between(d.time, d.nearest_observed_protein_distance_median, d.nearest_observed_protein_distance_mean, color=color, alpha=0.12)
    ax.set_title("Distance from generated states to observed target protein manifold")
    ax.set_xlabel("target time")
    ax.set_ylabel("nearest observed protein distance")
    ax.legend(frameon=False)
    savefig(fig, BIO_DIR, "bio15_target_manifold_distance_trace", "Target manifold distance", "biology_deep_dive", "How close are generated cells to the observed target protein manifold?")


def draw_chord(ax, theta0, theta1, r0, r1, color, width=1.0, alpha=0.45):
    p0 = np.array([r0 * np.cos(theta0), r0 * np.sin(theta0)])
    p1 = np.array([r1 * np.cos(theta1), r1 * np.sin(theta1)])
    c0 = p0 * 0.25
    c1 = p1 * 0.25
    path = MplPath([p0, c0, c1, p1], [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
    ax.add_patch(PathPatch(path, facecolor="none", edgecolor=color, lw=width, alpha=alpha, capstyle="round"))


def biology_16_to_20(tables: dict[str, pd.DataFrame]) -> None:
    terminal = tables["initial_rna_neighborhood_terminal_protein_fates"].copy()
    residual = tables.get("rna_local_masp_residuals", pd.DataFrame()).copy()
    score = tables["summary_scorecard_raw"].copy()

    # 16 chord
    d = terminal[terminal.method == "CITE-PCFM"].copy()
    neighborhoods = sorted(d.initial_rna_neighborhood.unique())
    phenotypes = [p for p in ["HSC", "MasP", "EryP", "MoP"] if p in set(d.terminal_phenotype)]
    fig, ax = plt.subplots(figsize=(8, 7.5))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-1.25, 1.35)
    ax.set_ylim(-1.25, 1.25)
    left = np.linspace(np.deg2rad(140), np.deg2rad(220), len(neighborhoods))
    right = np.linspace(np.deg2rad(-45), np.deg2rad(45), len(phenotypes))
    for n, th in zip(neighborhoods, left):
        ax.add_patch(Circle((np.cos(th), np.sin(th)), 0.045, fc="#E0F2FE", ec="#0284C7", lw=1.3))
        ax.text(1.10 * np.cos(th), 1.10 * np.sin(th), f"N{int(n)}", ha="center", va="center", fontsize=10)
    for p, th in zip(phenotypes, right):
        ax.add_patch(Wedge((0, 0), 1.04, np.rad2deg(th) - 6, np.rad2deg(th) + 6, width=0.11, fc=PHENO_COLORS[p], ec="white", alpha=0.95))
        ax.text(1.18 * np.cos(th), 1.18 * np.sin(th), p, ha="center", va="center", fontsize=11, color=PHENO_COLORS[p], weight="bold")
    for _, r in d.iterrows():
        if r.terminal_phenotype not in phenotypes:
            continue
        draw_chord(
            ax,
            left[neighborhoods.index(r.initial_rna_neighborhood)],
            right[phenotypes.index(r.terminal_phenotype)],
            0.98,
            0.98,
            PHENO_COLORS[r.terminal_phenotype],
            width=max(0.5, 11 * float(r.fraction)),
        )
    ax.set_title("RNA neighborhoods resolve terminal protein fates", weight="bold")
    savefig(fig, BIO_DIR, "bio16_rna_neighborhood_terminal_chord", "RNA neighborhood chord", "biology_deep_dive", "Can t0 RNA neighborhoods be linked to terminal protein outcomes?")

    # 17 terminal odds point plot
    fig, ax = plt.subplots(figsize=(9, 5.0))
    d2 = d[d.terminal_phenotype.isin(phenotypes)]
    for p, color in PHENO_COLORS.items():
        sub = d2[d2.terminal_phenotype == p]
        if len(sub):
            ax.plot(sub.initial_rna_neighborhood, sub.fraction, marker="o", lw=2, color=color, label=p)
    ax.set_xlabel("initial RNA neighborhood")
    ax.set_ylabel("terminal phenotype fraction")
    ax.set_title("Terminal protein fate probabilities by initial RNA neighborhood")
    ax.legend(frameon=False, ncol=4)
    savefig(fig, BIO_DIR, "bio17_terminal_fate_odds_by_neighborhood", "Terminal fate odds", "biology_deep_dive", "Which initial RNA neighborhoods create distinct protein futures?")

    # 18 MasP residual landscape
    if len(residual):
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.1), sharex=True, sharey=True)
        vmax = float(np.nanpercentile(np.abs(residual.residual), 98))
        for ax, method in zip(axes, METHOD_COLORS):
            sub = residual[(residual.method == method) & (residual.time > 0)]
            ax.scatter(sub.map1, sub.map2, c=sub.residual, s=10, cmap="coolwarm", vmin=-vmax, vmax=vmax, alpha=0.65, linewidths=0)
            ax.set_title(method)
            ax.set_xlabel("RNA map 1")
        axes[0].set_ylabel("RNA map 2")
        fig.suptitle("Local MasP protein affinity residuals over RNA-state neighborhoods", y=1.03, weight="bold")
        savefig(fig, BIO_DIR, "bio18_masp_local_residual_landscape", "MasP residual landscape", "biology_deep_dive", "Where in RNA state space is protein readout locally high or low?")

        fig, ax = plt.subplots(figsize=(8.5, 4.5))
        sns.violinplot(data=residual[residual.time > 0], x="method", y="residual", palette=METHOD_COLORS, inner="quartile", ax=ax, cut=0)
        ax.axhline(0, color="#111827", lw=1)
        ax.set_xlabel("")
        ax.set_ylabel("generated minus local observed MasP affinity")
        ax.set_title("Distribution of local MasP readout residuals")
        savefig(fig, BIO_DIR, "bio19_masp_residual_distribution", "MasP residual distribution", "biology_deep_dive", "Is the protein readout locally calibrated or biased?")

    # 20 assembly board
    fig, ax = plt.subplots(figsize=(11.5, 6.0))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    title = "Candidate NBT biology story"
    ax.text(0.04, 0.92, title, fontsize=20, weight="bold")
    cards = [
        ("Mechanism", "paired CITE geometry\nshapes RNA dynamics"),
        ("Landscape", "generated proteins\noccupy target states"),
        ("Phenotypes", "river and terminal fate\nviews expose biology"),
        ("Tradeoff", "protein geometry improves;\ncoarse fractions still caveat"),
    ]
    xs = [0.06, 0.30, 0.54, 0.78]
    for x, (h, b) in zip(xs, cards):
        rounded_box(ax, (x, 0.48), 0.17, 0.24, f"{h}\n\n{b}", "#F8FAFC", "#94A3B8", 11, "bold")
    for x0, x1 in zip(xs[:-1], xs[1:]):
        curved_arrow(ax, (x0 + 0.17, 0.60), (x1, 0.60), "#64748B")
    ax.text(0.06, 0.28, "Use as main figure logic: A mechanism panel, one observed-vs-generated landscape, one fate/phenotype panel, and one scorecard.", fontsize=12)
    ax.text(0.06, 0.20, "Keep phenotype-fraction limits explicit. The strongest evidence is protein-coordinate dynamics and joint RNA-protein structure.", fontsize=12, color="#475569")
    savefig(fig, BIO_DIR, "bio20_article_assembly_board", "Article assembly board", "biology_deep_dive", "How should the biology panels be assembled into an article figure?")


def lifecycle_01_to_20(trial_df: pd.DataFrame, run_df: pd.DataFrame, decision_df: pd.DataFrame) -> None:
    # 01 genealogy
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    nodes = [
        ("cite_vgfm", "unbalanced\nmulti-omic start", "Stage 1 failed"),
        ("cite_bfm_balanced", "balanced\nrevision", "Stage 2 weak"),
        ("cite_factorized_fm", "factorized\nRNA/protein", "provenance blocked"),
        ("cite_residual_sb_fm", "residual\nbridge", "baseline gap"),
        ("cite_paired_coord_fm", "paired-coordinate\nFM", "final locked"),
    ]
    xs = np.linspace(0.10, 0.90, len(nodes))
    ys = [0.52, 0.63, 0.48, 0.62, 0.52]
    for i in range(len(nodes) - 1):
        curved_arrow(ax, (xs[i] + 0.07, ys[i]), (xs[i + 1] - 0.07, ys[i + 1]), "#94A3B8")
    for x, y, (alg, label, note) in zip(xs, ys, nodes):
        color = ALG_COLORS.get(alg, "#64748B")
        ax.add_patch(Circle((x, y), 0.07, fc="#F8FAFC" if alg != "cite_paired_coord_fm" else "#FEE2E2", ec=color, lw=2.2))
        ax.text(x, y + 0.01, label, ha="center", va="center", fontsize=10.5, weight="bold")
        ax.text(x, y - 0.13, note, ha="center", fontsize=9.5, color=color, weight="bold")
    ax.text(0.04, 0.92, "CellCompass design genealogy", fontsize=21, weight="bold")
    ax.text(0.04, 0.86, "The successful algorithm emerged after several falsified or blocked designs.", fontsize=12, color="#475569")
    savefig(fig, LIFE_DIR, "life01_algorithm_genealogy", "Algorithm genealogy", "agent_lifecycle", "What sequence of algorithm ideas led to CITE-PCFM?")

    # 02 stage timeline
    if len(trial_df):
        df = trial_df[trial_df.algorithm == "cite_paired_coord_fm"].copy()
        df = df.sort_values("created_at").reset_index(drop=True)
        df["local_trial_index"] = np.arange(1, len(df) + 1)
        fig, ax = plt.subplots(figsize=(12, 4.8))
        stages = list(dict.fromkeys(df.stage.dropna()))
        ymap = {s: i for i, s in enumerate(stages)}
        for _, r in df.iterrows():
            color = "#B73A3A" if r.decision == "promote" else "#94A3B8"
            ax.scatter(r.created_at, ymap.get(r.stage, 0), s=70, color=color, alpha=0.8, edgecolor="white")
        ax.set_yticks(range(len(stages)))
        ax.set_yticklabels([s.replace("_", "\n") for s in stages])
        ax.set_title("CITE-PCFM campaign stage timeline")
        ax.set_xlabel("time")
        savefig(fig, LIFE_DIR, "life02_stage_timeline", "Campaign timeline", "agent_lifecycle", "How did the search move through feasibility, claim validation, tuning and final regression?")

        # 03 cumulative best protein energy
        df2 = df.dropna(subset=["protein_energy"]).copy()
        if len(df2):
            df2["best_so_far"] = df2["protein_energy"].cummin()
            fig, ax = plt.subplots(figsize=(10, 4.8))
            ax.plot(df2.local_trial_index, df2.protein_energy, color="#CBD5E1", lw=1.5, marker="o", label="trial")
            ax.step(df2.local_trial_index, df2.best_so_far, where="post", color="#B73A3A", lw=3, label="best so far")
            ax.set_xlabel("CITE-PCFM trial order")
            ax.set_ylabel("protein energy")
            ax.set_title("Primary claim metric improved by staged search")
            ax.legend(frameon=False)
            savefig(fig, LIFE_DIR, "life03_primary_metric_active_best", "Active best metric", "agent_lifecycle", "How did the primary protein metric improve over trials?")

        # 04 trial scatter
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        for dec, sub in df.groupby("decision"):
            color = "#B73A3A" if dec == "promote" else "#94A3B8"
            ax.scatter(sub.w1_mean, sub.protein_energy, s=90, color=color, alpha=0.75, label=dec, edgecolor="white")
        ax.set_xlabel("W1 guardrail")
        ax.set_ylabel("protein energy")
        ax.set_title("Trial-level tradeoff: claim metric versus RNA guardrail")
        ax.legend(frameon=False)
        savefig(fig, LIFE_DIR, "life04_trial_pareto_cloud", "Trial Pareto cloud", "agent_lifecycle", "Which trials improved protein dynamics without breaking RNA guardrails?")

        # 05 promote/reject clock
        fig, ax = plt.subplots(figsize=(11, 3.8))
        colors = ["#B73A3A" if d == "promote" else "#CBD5E1" for d in df.decision]
        ax.bar(df.local_trial_index, np.ones(len(df)), color=colors, width=0.9)
        ax.set_yticks([])
        ax.set_xlabel("CITE-PCFM trial order")
        ax.set_title("Promote and reject decisions across the campaign")
        savefig(fig, LIFE_DIR, "life05_promote_reject_clock", "Promote reject clock", "agent_lifecycle", "Where did the campaign accept or reject candidate changes?")

        # 06 stage outcome counts
        counts = df.groupby(["stage", "decision"]).size().reset_index(name="n")
        fig, ax = plt.subplots(figsize=(9.5, 5.2))
        bottom = Counter()
        for dec, color in [("promote", "#B73A3A"), ("reject", "#CBD5E1"), ("locked", "#7F1D1D")]:
            sub = counts[counts.decision == dec]
            xs = list(sub.stage)
            vals = list(sub.n)
            ax.bar(xs, vals, bottom=[bottom[x] for x in xs], color=color, label=dec)
            for x, v in zip(xs, vals):
                bottom[x] += v
        ax.set_title("Stage-level decision burden")
        ax.set_ylabel("number of trials")
        ax.set_xticklabels([x.get_text().replace("_", "\n") for x in ax.get_xticklabels()], fontsize=9)
        ax.legend(frameon=False)
        savefig(fig, LIFE_DIR, "life06_stage_outcome_counts", "Stage outcome counts", "agent_lifecycle", "Which campaign stages consumed the most exploration?")

    # 07 failure reason bars
    if len(trial_df):
        reasons = []
        for txt in trial_df.decision_reasons.fillna(""):
            low = txt.lower()
            if "wall-clock" in low or "timeout" in low:
                reasons.append("training time")
            elif "primary metric" in low or "active best" in low:
                reasons.append("no metric gain")
            elif "secondary metric" in low or "tolerance" in low:
                reasons.append("guardrail regression")
            elif "provenance" in low:
                reasons.append("provenance")
            elif "tmv" in low:
                reasons.append("mass diagnostic")
            else:
                reasons.append("other")
        tmp = pd.Series(reasons).value_counts()
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        ax.barh(tmp.index[::-1], tmp.values[::-1], color="#94A3B8")
        ax.set_title("Failure modes recorded by the campaign")
        ax.set_xlabel("events")
        savefig(fig, LIFE_DIR, "life07_failure_reason_bars", "Failure reason bars", "agent_lifecycle", "Why did candidate trials fail or get rolled back?")

    # 08 to 11 config traces
    cdf = run_df[run_df.algorithm == "cite_paired_coord_fm"].copy()
    if len(cdf):
        config_cols = ["lambda_p", "alpha_protein", "endpoint_protein_weight", "protein_time_weight_width"]
        for idx, col in enumerate(config_cols, start=8):
            if cdf[col].notna().sum() == 0:
                continue
            fig, ax = plt.subplots(figsize=(10, 4.2))
            ax.plot(cdf.run_index, cdf[col], lw=2.0, marker="o", color="#B73A3A")
            ax.scatter(cdf.loc[cdf.decision == "promote", "run_index"], cdf.loc[cdf.decision == "promote", col], s=110, color="#7F1D1D", edgecolor="white", label="promote")
            ax.set_xlabel("run order")
            ax.set_ylabel(col)
            ax.set_title(f"Configuration mutation trace: {col}")
            ax.legend(frameon=False)
            savefig(fig, LIFE_DIR, f"life{idx:02d}_{col}_trace", f"{col} trace", "agent_lifecycle", f"How did the agent mutate {col} during the search?")

        # 12 runtime versus metric
        fig, ax = plt.subplots(figsize=(7.5, 5.2))
        ax.scatter(cdf.elapsed_sec, cdf.protein_energy, c=cdf.w1_mean, cmap="viridis", s=90, edgecolor="white")
        ax.set_xlabel("elapsed seconds")
        ax.set_ylabel("protein energy")
        ax.set_title("Runtime cost versus protein claim metric")
        cb = fig.colorbar(ax.collections[0], ax=ax)
        cb.set_label("W1")
        savefig(fig, LIFE_DIR, "life12_runtime_vs_metric", "Runtime versus metric", "agent_lifecycle", "Did better candidates require more compute?")

        # 13 decision embedding
        features = ["w1_mean", "protein_energy", "runtime_sec", "lambda_p", "alpha_protein", "endpoint_protein_weight", "protein_time_weight_width"]
        embdf = cdf.dropna(subset=["w1_mean", "protein_energy"]).copy().reset_index(drop=True)
        use = [f for f in features if embdf[f].notna().sum() > 3]
        if len(embdf) > 4 and len(use) >= 2:
            X = embdf[use].copy()
            X = X.fillna(X.median(numeric_only=True))
            emb = PCA(n_components=2, random_state=20260527).fit_transform(StandardScaler().fit_transform(X))
            fig, ax = plt.subplots(figsize=(7.5, 5.7))
            for dec, sub_idx in embdf.groupby("decision").groups.items():
                color = "#B73A3A" if dec == "promote" else "#94A3B8"
                pos = emb[list(sub_idx), :]
                ax.scatter(pos[:, 0], pos[:, 1], s=85, color=color, label=dec, alpha=0.8, edgecolor="white")
            ax.set_title("Trial design embedding")
            ax.set_xlabel("design PC1")
            ax.set_ylabel("design PC2")
            ax.legend(frameon=False)
            savefig(fig, LIFE_DIR, "life13_decision_embedding", "Decision embedding", "agent_lifecycle", "Can trial designs be embedded to show productive regions of search?")

    # 14 algorithm family best scores
    allrun = run_df.dropna(subset=["protein_energy"]).copy()
    if len(allrun):
        best = allrun.sort_values("protein_energy").groupby("algorithm", as_index=False).first()
        fig, ax = plt.subplots(figsize=(9, 4.8))
        best = best.sort_values("protein_energy")
        ax.scatter(best.protein_energy, best.algorithm, s=130, c=[ALG_COLORS.get(a, "#64748B") for a in best.algorithm], edgecolor="white")
        for _, r in best.iterrows():
            ax.plot([0, r.protein_energy], [r.algorithm, r.algorithm], color=ALG_COLORS.get(r.algorithm, "#64748B"), alpha=0.35, lw=3)
        ax.set_xlabel("best recorded protein energy, lower is better")
        ax.set_title("Best recorded protein metric by design family")
        ax.text(0.01, -0.18, "Dataset contexts differ for early designs; use this as process evidence, not a direct benchmark.", transform=ax.transAxes, fontsize=9, color="#475569")
        savefig(fig, LIFE_DIR, "life14_algorithm_family_best_scores", "Algorithm family best scores", "agent_lifecycle", "Which algorithm family ultimately produced the strongest metric?")

    # 15 proposal review gate
    if len(decision_df):
        dd = decision_df[decision_df.algorithm == "cite_paired_coord_fm"].copy()
        dd = dd[dd.phase.isin(["proposal", "authoring"])]
        fig, ax = plt.subplots(figsize=(10.5, 4.5))
        colors = ["#B73A3A" if "approve" in str(d) else "#94A3B8" if "revise" in str(d) else "#CBD5E1" for d in dd.decision]
        ax.scatter(dd["at"], np.arange(len(dd)), s=110, color=colors, edgecolor="white")
        for i, (_, r) in enumerate(dd.iterrows()):
            ax.text(r["at"], i + 0.15, r.decision.replace("_", " "), fontsize=8, rotation=20)
        ax.set_yticks([])
        ax.set_title("Proposal review gate before implementation")
        ax.set_xlabel("time")
        savefig(fig, LIFE_DIR, "life15_proposal_review_gate", "Proposal review gate", "agent_lifecycle", "How was the algorithm proposal revised before training?")

    # 16 metric waterfall across designs
    if len(allrun):
        best = allrun.sort_values("protein_energy").groupby("algorithm", as_index=False).first()
        order = ["cite_vgfm", "cite_bfm_balanced", "cite_factorized_fm", "cite_residual_sb_fm", "cite_paired_coord_fm"]
        best = best.set_index("algorithm").reindex([o for o in order if o in set(best.algorithm)]).reset_index()
        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.plot(best.algorithm, best.protein_energy, lw=3, marker="o", color="#B73A3A")
        ax.set_xticklabels([x.get_text().replace("cite_", "").replace("_", "\n") for x in ax.get_xticklabels()], fontsize=9)
        ax.set_ylabel("best recorded protein energy")
        ax.set_title("Metric records across design families")
        ax.text(0.01, -0.20, "Context changes across early debug, simulation and final GSE runs; read as design history, not a head-to-head score.", transform=ax.transAxes, fontsize=9, color="#475569")
        savefig(fig, LIFE_DIR, "life16_metric_waterfall_across_algorithms", "Metric waterfall across algorithms", "agent_lifecycle", "How did successive design families change the best metric?")

    # 17 design loop schematic
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    loop = [
        (0.20, 0.72, "hypothesis"),
        (0.55, 0.72, "proposal gate"),
        (0.76, 0.45, "training trial"),
        (0.55, 0.18, "metric gate"),
        (0.20, 0.18, "revision"),
    ]
    for x, y, label in loop:
        rounded_box(ax, (x - 0.08, y - 0.055), 0.16, 0.11, label, "#F8FAFC", "#94A3B8", 11, "bold")
    for (x0, y0, _), (x1, y1, _) in zip(loop, loop[1:] + loop[:1]):
        curved_arrow(ax, (x0, y0 - 0.06), (x1, y1 + 0.06), "#64748B", rad=0.15)
    ax.text(0.08, 0.92, "Lifecycle mechanism: fail fast, keep evidence-compatible assumptions", fontsize=18, weight="bold")
    ax.text(0.18, 0.42, "This is the story to show readers: not a one-shot prompt,\nbut an evidence-filtered model design process.", fontsize=13, color="#475569")
    savefig(fig, LIFE_DIR, "life17_agent_design_loop_schematic", "Agent design loop", "agent_lifecycle", "What recurring loop generated and filtered algorithms?")

    # 18 snapshot and rollback marks
    if len(decision_df):
        dd = decision_df[decision_df.algorithm == "cite_paired_coord_fm"].copy()
        masks = dd.decision.str.contains("rollback|promote|reject|approve|locked", case=False, na=False)
        dd = dd[masks]
        fig, ax = plt.subplots(figsize=(11, 4.5))
        categories = ["approve", "promote", "reject", "rollback", "locked"]
        ymap = {c: i for i, c in enumerate(categories)}
        for _, r in dd.iterrows():
            low = str(r.decision).lower()
            cat = next((c for c in categories if c in low), "reject")
            color = "#B73A3A" if cat in ["approve", "promote", "locked"] else "#94A3B8"
            ax.scatter(r["at"], ymap[cat], s=90, color=color, edgecolor="white")
        ax.set_yticks(range(len(categories)))
        ax.set_yticklabels(categories)
        ax.set_title("Key campaign decisions and rollback events")
        ax.set_xlabel("time")
        savefig(fig, LIFE_DIR, "life18_snapshot_rollback_events", "Rollback events", "agent_lifecycle", "Where did the system keep or discard trial work?")

    # 19 final lock card
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    rounded_box(ax, (0.08, 0.56), 0.22, 0.18, "final model\nCITE-PCFM", "#FEE2E2", "#B73A3A", 15, "bold")
    rounded_box(ax, (0.39, 0.64), 0.22, 0.12, "protein energy\n0.0222", "#FFF7ED", "#F97316", 13, "bold")
    rounded_box(ax, (0.39, 0.43), 0.22, 0.12, "RNA guardrail\nretained", "#E8F2FF", "#60A5FA", 13, "bold")
    rounded_box(ax, (0.70, 0.52), 0.22, 0.18, "locked final\nregression", "#F0FDF4", "#16A34A", 14, "bold")
    curved_arrow(ax, (0.30, 0.65), (0.39, 0.70), "#64748B")
    curved_arrow(ax, (0.61, 0.70), (0.70, 0.62), "#64748B")
    curved_arrow(ax, (0.61, 0.49), (0.70, 0.58), "#64748B")
    ax.text(0.08, 0.86, "Final-lock evidence card", fontsize=18, weight="bold")
    ax.text(0.08, 0.30, "Use this as an inset, not as a replacement for biological validation panels.", fontsize=12, color="#475569")
    savefig(fig, LIFE_DIR, "life19_final_lock_evidence_card", "Final lock evidence card", "agent_lifecycle", "What evidence supported declaring this design locked?")

    # 20 lifecycle assembly board
    fig, ax = plt.subplots(figsize=(11.5, 6.0))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.04, 0.92, "Candidate NBT lifecycle story", fontsize=20, weight="bold")
    cards = [
        ("Search space", "several CITE\nalgorithm families"),
        ("Gates", "proposal, metric,\nand guardrail filters"),
        ("Mutation", "config and mechanism\nchanges over trials"),
        ("Lock", "paired-coordinate\nFM retained"),
    ]
    xs = [0.06, 0.30, 0.54, 0.78]
    for x, (h, b) in zip(xs, cards):
        rounded_box(ax, (x, 0.48), 0.17, 0.24, f"{h}\n\n{b}", "#F8FAFC", "#94A3B8", 11, "bold")
    for x0, x1 in zip(xs[:-1], xs[1:]):
        curved_arrow(ax, (x0 + 0.17, 0.60), (x1, 0.60), "#64748B")
    ax.text(0.06, 0.28, "Use as a separate process figure or extended-data figure.", fontsize=12)
    ax.text(0.06, 0.20, "The message is gradual convergence through falsification, not a single lucky algorithm proposal.", fontsize=12, color="#475569")
    savefig(fig, LIFE_DIR, "life20_lifecycle_assembly_board", "Lifecycle assembly board", "agent_lifecycle", "How should the agent-process panels be assembled into an article figure?")


def write_catalog() -> None:
    rows = []
    rows.append("# NBT-style CITE-PCFM Figure Bank\n")
    rows.append("This is a selection bank, not final publication art. The goal is to support two article stories:\n")
    rows.append("1. CITE-PCFM biology: RNA-initialized protein-coordinate dynamics against observed CITE targets.\n")
    rows.append("2. CellCompass lifecycle: several failed or revised designs converged to the locked CITE-PCFM algorithm.\n")
    rows.append("\n## Design notes\n")
    rows.append("- Avoid using a heatmap as the main visual argument.\n")
    rows.append("- Prefer observed-target overlays, trajectories, rivers, chords, scorecards and staged timelines.\n")
    rows.append("- Do not overclaim: CITE-PCFM is strongest on protein-coordinate dynamics and joint structure; phenotype fractions remain a caveat.\n")
    rows.append("- In the main paper, do not call CRUFM or balanced OT-CFM RNA-only baselines unless a separate audit proves their state excludes protein coordinates.\n")
    for track in ["biology_deep_dive", "agent_lifecycle"]:
        rows.append(f"\n## {track}\n")
        for fig in [f for f in FIGURES if f["track"] == track]:
            rel = Path(fig["png"])
            rows.append(f"- **{fig['name']}**: {fig['title']}. Question: {fig['question']}. File: `{rel}`\n")
    CATALOG_PATH.write_text("".join(rows))
    MANIFEST_PATH.write_text(json.dumps({"figures": FIGURES}, indent=2))


def main() -> None:
    ensure_dirs()
    tables = load_tables()
    biology_01_mechanism_card()
    biology_02_to_12(tables)
    biology_landscape_figures(tables)
    biology_16_to_20(tables)
    trial_df, run_df, decision_df = collect_lifecycle()
    trial_df.to_csv(NOTE_DIR / "lifecycle_trials.csv", index=False)
    run_df.to_csv(NOTE_DIR / "lifecycle_runs.csv", index=False)
    decision_df.to_csv(NOTE_DIR / "lifecycle_decisions.csv", index=False)
    lifecycle_01_to_20(trial_df, run_df, decision_df)
    write_catalog()
    print(json.dumps({"figure_count": len(FIGURES), "catalog": str(CATALOG_PATH), "out_root": str(OUT_ROOT)}, indent=2))


if __name__ == "__main__":
    main()
