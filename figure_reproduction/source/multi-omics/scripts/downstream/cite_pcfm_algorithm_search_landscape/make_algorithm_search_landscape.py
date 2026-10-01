#!/usr/bin/env python3
"""Quantitative algorithm-search landscapes from ~/.cellcompass registries."""

from __future__ import annotations

import json
import math
import os
import textwrap
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/cytobridge_mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/cytobridge_xdg_cache")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)

import matplotlib
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler


sns.set_theme(style="white", context="talk")

WORK_ROOT = Path("/Users/zhenyizhang/Documents/GitHub/CytoBridge-agent-runtime-v2-20260309/multi-omics")
CELL_ROOT = Path("/Users/zhenyizhang/.cellcompass")
OUT_ROOT = WORK_ROOT / "figures/cite_pcfm_deep_story/algorithm_search_landscape"
NOTE_DIR = WORK_ROOT / "downstream/cite_pcfm_deep_story"
TABLE_DIR = NOTE_DIR / "tables"
STORY_PATH = NOTE_DIR / "storyboard.md"
SEARCH_NOTE_PATH = NOTE_DIR / "algorithm_search_landscape.md"
MANIFEST_PATH = NOTE_DIR / "algorithm_search_manifest.json"

FINAL_DATASET = "gse305370_citeseq_stage2_10k"
CITE_ALGS = [
    "cite_vgfm",
    "cite_bfm",
    "cite_bfm_balanced",
    "cite_factorized_fm",
    "cite_residual_sb_fm",
    "cite_paired_coord_fm",
]
ALG_COLORS = {
    "cite_vgfm": "#94A3B8",
    "cite_bfm": "#64748B",
    "cite_bfm_balanced": "#475569",
    "cite_factorized_fm": "#A78BFA",
    "cite_residual_sb_fm": "#F59E0B",
    "cite_paired_coord_fm": "#B73A3A",
}
DISPLAY_METHOD_LABELS = {
    "VGFM": "CITE-VGFM\nproduct-space VGFM",
    "BFM": "CITE-BFM-Balanced\nbalanced CITE FM",
    "FFM": "CITE-FFM\nFactorized FM",
    "RSB-FM": "CITE-RSB-FM\nResidual Stochastic-Interpolant FM",
    "PCFM": "CITE-PCFM\nPaired-Coordinate FM",
}
POINT_METHOD_LABELS = {
    "VGFM": "CITE-VGFM",
    "BFM": "CITE-BFM-Balanced",
    "FFM": "CITE-FFM",
    "RSB-FM": "CITE-RSB-FM",
    "PCFM": "CITE-PCFM",
}

FIGURES: list[dict] = []


def ensure_dirs() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    NOTE_DIR.mkdir(parents=True, exist_ok=True)


def safe_json(path: Path) -> dict:
    try:
        with path.open() as f:
            return json.load(f)
    except Exception:
        return {}


def flatten_numbers(obj, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    if isinstance(obj, dict):
        for key, val in obj.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            out.update(flatten_numbers(val, name))
    elif isinstance(obj, list):
        for i, val in enumerate(obj[:3]):
            name = f"{prefix}.{i}" if prefix else str(i)
            out.update(flatten_numbers(val, name))
    elif isinstance(obj, (int, float, np.integer, np.floating)) and not isinstance(obj, bool):
        if np.isfinite(float(obj)):
            out[prefix] = float(obj)
    return out


def dataset_id_from_path(path: str) -> str:
    if not path:
        return ""
    try:
        return Path(path).parent.name
    except Exception:
        return ""


def safe_nanmean(values) -> float:
    if values is None:
        return float("nan")
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.isnan(arr).all():
        return float("nan")
    return float(np.nanmean(arr))


def savefig(fig: plt.Figure, name: str, title: str, question: str) -> Path:
    path = OUT_ROOT / f"{name}.png"
    fig.savefig(path, dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    FIGURES.append({"name": name, "title": title, "question": question, "png": str(path), "pdf": str(path.with_suffix(".pdf")), "svg": str(path.with_suffix(".svg"))})
    return path


def load_runs() -> pd.DataFrame:
    rows = []
    for alg in CITE_ALGS:
        run_dir = CELL_ROOT / "training_algorithms" / alg / "registry" / "runs"
        for p in sorted(run_dir.glob("*.json")):
            d = safe_json(p)
            metrics_path = Path(d.get("metrics_path", ""))
            md = safe_json(metrics_path) if metrics_path.exists() else {}
            cm = md.get("custom_metrics", {}) or {}
            cfg = md.get("config", {}) or {}
            flat = flatten_numbers(cfg)

            row = {
                "algorithm": alg,
                "run_id": d.get("run_id", p.stem),
                "trial_id": d.get("trial_id", ""),
                "campaign_id": d.get("campaign_id", ""),
                "stage": d.get("stage", ""),
                "decision": d.get("decision", d.get("campaign_trial_decision", "")),
                "decision_reason": d.get("decision_reason", ""),
                "status": d.get("status", ""),
                "created_at": d.get("created_at", ""),
                "completed_at": d.get("completed_at", ""),
                "dataset_id": dataset_id_from_path(d.get("data_path", md.get("adata_path", ""))),
                "w1_mean": d.get("w1_mean", md.get("w1_mean", np.nan)),
                "tmv_mean": d.get("tmv_mean", md.get("tmv_mean", np.nan)),
                "protein_energy": cm.get("protein_energy", np.nan),
                "rna_energy_guardrail": cm.get("rna_energy_guardrail", np.nan),
                "paired_cov_error": cm.get("paired_cov_error", np.nan),
                "residual_energy": cm.get("residual_energy", np.nan),
                "runtime_sec": md.get("runtime_sec", np.nan),
                "elapsed_sec": (md.get("training_time_budget", {}) or {}).get("elapsed_sec", np.nan),
                "completed_epochs": (md.get("training_time_budget", {}) or {}).get("completed_epochs", md.get("completed_epochs", np.nan)),
            }
            interesting = [
                "cite_pcfm.lambda_p",
                "cite_pcfm.alpha_protein",
                "cite_pcfm.endpoint_protein_weight",
                "cite_pcfm.protein_head_hidden_dim",
                "cite_pcfm.time_fourier_features",
                "cite_pcfm.time_fourier_scale",
                "cite_pcfm.protein_time_weight_strength",
                "cite_pcfm.protein_time_weight_width",
                "training.plan.0.lr",
                "training.plan.0.batch_size",
                "training.plan.0.cite_pcfm.protein_time_weight_width",
                "training.plan.0.flow_matching.coupling.reg",
                "flow_matching.cite_rsb_fm.residual_sigma",
                "flow_matching.cite_rsb_fm.alpha_e",
                "flow_matching.cite_rsb_fm.lambda_z",
                "training.plan.0.flow_matching.cite_rsb_fm.residual_sigma",
                "training.plan.0.flow_matching.cite_rsb_fm.alpha_e",
                "training.plan.0.flow_matching.cite_rsb_fm.lambda_z",
            ]
            for key in interesting:
                row[key.replace(".", "__")] = flat.get(key, np.nan)
            rows.append(row)

    df = pd.DataFrame(rows)
    for col in ["created_at", "completed_at"]:
        df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    df = df.sort_values("created_at").reset_index(drop=True)
    df["global_order"] = np.arange(1, len(df) + 1)
    df["is_final_dataset"] = df["dataset_id"].eq(FINAL_DATASET)
    df["is_promoted"] = df["decision"].astype(str).str.contains("promote|locked|approve", case=False, na=False)
    df["is_final_lock"] = df["run_id"].astype(str).str.contains("__final__cite_paired_coord_fm", na=False)
    return df


def load_trials() -> pd.DataFrame:
    rows = []
    for alg in CITE_ALGS:
        base = CELL_ROOT / "algorithm_campaigns" / alg
        for p in sorted(base.glob("campaign_*/trials/*.json")):
            d = safe_json(p)
            ms = d.get("metrics_summary", {}) or {}
            cm = ms.get("custom_metrics", {}) or {}
            rows.append(
                {
                    "algorithm": alg,
                    "trial_id": d.get("trial_id", p.stem),
                    "campaign_id": d.get("campaign_id", ""),
                    "stage": d.get("stage", ""),
                    "decision": d.get("decision", ""),
                    "created_at": d.get("created_at", ""),
                    "completed_at": d.get("completed_at", d.get("updated_at", "")),
                    "primary_metric": ms.get("primary_metric", ""),
                    "primary_value": ms.get("primary_value", np.nan),
                    "protein_energy": cm.get("protein_energy", ms.get("primary_value", np.nan) if ms.get("primary_metric") == "protein_energy" else np.nan),
                    "w1_mean": ms.get("w1_mean", np.nan),
                    "tmv_mean": ms.get("tmv_mean", np.nan),
                    "secondary_metric": ms.get("secondary_metric", ""),
                    "secondary_value": ms.get("secondary_value", np.nan),
                    "decision_reasons": "; ".join(d.get("decision_reasons", []) or []),
                    "run_id": (d.get("run_ids") or [""])[0],
                }
            )
    df = pd.DataFrame(rows)
    for col in ["created_at", "completed_at"]:
        df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    df = df.sort_values("created_at").reset_index(drop=True)
    df["global_trial_order"] = np.arange(1, len(df) + 1)
    return df


def load_builtin_baselines(dataset_id: str) -> pd.DataFrame:
    rows = []
    base = CELL_ROOT / "algorithm_benchmarks" / "datasets" / dataset_id / "baseline_artifacts"
    for p in sorted(base.glob("builtin_*/metrics.json")):
        md = safe_json(p)
        cm = md.get("custom_metrics", {}) or {}
        w1_scores = md.get("w1_scores")
        tmv_scores = md.get("tmv_scores")
        rows.append(
            {
                "algorithm": p.parent.name.replace("builtin_", ""),
                "dataset_id": dataset_id,
                "protein_energy": cm.get("protein_energy", np.nan),
                "w1_mean": safe_nanmean(w1_scores),
                "tmv_mean": safe_nanmean(tmv_scores),
                "rna_energy_guardrail": cm.get("rna_energy_guardrail", np.nan),
                "paired_cov_error": cm.get("paired_cov_error", np.nan),
            }
        )
    return pd.DataFrame(rows)


def rounded_box(ax, xy, w, h, text, fc="#F8FAFC", ec="#CBD5E1", size=9.5, weight="normal"):
    patch = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.014,rounding_size=0.025", linewidth=1.1, edgecolor=ec, facecolor=fc)
    ax.add_patch(patch)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=size, weight=weight, wrap=True)
    return patch


def add_arrow(ax, p0, p1, color="#64748B", rad=0.08, lw=1.5):
    patch = FancyArrowPatch(p0, p1, connectionstyle=f"arc3,rad={rad}", arrowstyle="Simple,head_width=6,head_length=8", color=color, linewidth=lw, alpha=0.85)
    ax.add_patch(patch)


def lineage_rows(baselines: pd.DataFrame) -> pd.DataFrame:
    crufm = baselines[baselines["algorithm"].eq("crufm")]
    crufm_energy = float(crufm["protein_energy"].iloc[0]) if len(crufm) and pd.notna(crufm["protein_energy"].iloc[0]) else np.nan
    rows = [
        {
            "order": 1,
            "family": "1. cite_vgfm",
            "hypothesis": "Start from a VGFM-style continuous rollout and let protein fidelity emerge from generic latent transport.",
            "evidence": "debug/proportional best protein_energy 0.124831; final-panel builtin VGFM reference 0.028261.",
            "failure_or_limit": "Unbalanced/mass-style assumptions made TMV/W1 gates fragile and did not create a CITE-specific protein readout.",
            "lesson_to_next": "Protein fidelity cannot be treated as a generic latent side effect; introduce paired CITE geometry.",
            "status": "failed",
            "metric": 0.12483108043670654,
        },
        {
            "order": 2,
            "family": "2. cite_bfm / balanced",
            "hypothesis": "Use balanced OT-CFM structure with explicit RNA/protein coordinates and endpoint-aware protein cost.",
            "evidence": "debug/proportional best protein_energy 0.083874; W1 9.297.",
            "failure_or_limit": "Endpoint/coupling plausibility improved, but sealed rollout protein biology remained too weak.",
            "lesson_to_next": "Factorize the problem so RNA transport and protein residual structure can be inspected separately.",
            "status": "failed",
            "metric": 0.08387410640716553,
        },
        {
            "order": 3,
            "family": "3. cite_factorized_fm",
            "hypothesis": "Separate RNA dynamics from protein residual readout to avoid endpoint-cost shortcuts.",
            "evidence": "Stage 1 feasibility passed with W1 9.8755 versus comparator 11.1421.",
            "failure_or_limit": "Stage 2 could not produce trusted evidence because evaluator provenance and sibling-campaign arbitration conflicted.",
            "lesson_to_next": "The lifecycle must lock evaluator provenance before claim validation; scientific potential is not enough.",
            "status": "blocked",
            "metric": np.nan,
        },
        {
            "order": 4,
            "family": "4. cite_residual_sb_fm",
            "hypothesis": "Model protein residual dynamics with a stochastic bridge so missing protein motion is not collapsed to endpoints.",
            "evidence": "final GSE305370 protein_energy 0.031822; controlled residual simulation best 0.005369.",
            "failure_or_limit": f"Real-data claim missed CRUFM reference {crufm_energy:.6f}; residual stochasticity increased protein/covariance drift.",
            "lesson_to_next": "Remove unsupported stochastic residual machinery and keep a deterministic paired-coordinate rule.",
            "status": "failed",
            "metric": 0.031821608543395996,
        },
        {
            "order": 5,
            "family": "5. cite_paired_coord_fm",
            "hypothesis": "Keep RNA runtime deterministic, train protein as path-coupled paired-coordinate readout, and weight local biological time windows.",
            "evidence": "final locked protein_energy 0.022198; raw same-panel minimum 0.021899 shown separately.",
            "failure_or_limit": "Not W1-SOTA and not a mass model; claim is specifically rollout-derived CITE protein fidelity.",
            "lesson_to_next": "Lock the reproducible low-energy basin instead of chasing a single unreproduced scalar minimum.",
            "status": "locked",
            "metric": 0.022197961807250977,
        },
    ]
    return pd.DataFrame(rows)


def fig_search00_five_algorithm_lineage(baselines: pd.DataFrame) -> None:
    df = lineage_rows(baselines)
    df.to_csv(TABLE_DIR / "five_algorithm_lineage_map.csv", index=False)

    status_colors = {
        "failed": "#FEE2E2",
        "blocked": "#FEF3C7",
        "locked": "#DCFCE7",
    }
    edge_colors = {
        "failed": "#DC2626",
        "blocked": "#D97706",
        "locked": "#16A34A",
    }

    fig, ax = plt.subplots(figsize=(17, 8.2))
    ax.set_axis_off()
    ax.set_xlim(-0.035, 1.035)
    ax.set_ylim(0, 1)

    xs = np.linspace(0.08, 0.92, len(df))
    y = 0.62
    box_w = 0.16
    box_h = 0.42

    for i, (_, row) in enumerate(df.iterrows()):
        x = xs[i]
        status = row["status"]
        box = FancyBboxPatch(
            (x - box_w / 2, y - box_h / 2),
            box_w,
            box_h,
            boxstyle="round,pad=0.018,rounding_size=0.02",
            linewidth=1.4,
            edgecolor=edge_colors[status],
            facecolor=status_colors[status],
        )
        ax.add_patch(box)
        metric = "no trusted Stage 2" if pd.isna(row["metric"]) else f"protein_energy {row['metric']:.4f}"
        ax.text(x, y + box_h * 0.41, row["family"], ha="center", va="top", fontsize=10.5, weight="bold", color="#111827")
        ax.text(x, y + box_h * 0.29, f"{row['status'].upper()} | {metric}", ha="center", va="top", fontsize=8.5, color=edge_colors[status], weight="bold")
        ax.text(x, y + box_h * 0.12, textwrap.fill(row["hypothesis"], 26), ha="center", va="top", fontsize=8.0, color="#1F2937")
        ax.text(x, y - box_h * 0.18, textwrap.fill("Learned: " + row["lesson_to_next"], 27), ha="center", va="top", fontsize=7.8, color="#374151")
        if i < len(df) - 1:
            add_arrow(ax, (x + box_w / 2 + 0.005, y), (xs[i + 1] - box_w / 2 - 0.005, y), color="#64748B", rad=0.0, lw=1.1)

    ax.text(0.5, 0.95, "Five-algorithm CITE lifecycle: what each generation falsified and retained", ha="center", va="center", fontsize=19, weight="bold")
    ax.text(
        0.5,
        0.90,
        "The evolution is not a leaderboard: each algorithm removed one unsupported assumption before PCFM locked a reproducible protein-fidelity rule.",
        ha="center",
        va="center",
        fontsize=11,
        color="#475569",
    )

    failures = [
        ("VGFM", "generic latent transport was not enough"),
        ("BFM", "endpoint/coupling repair did not explain sealed rollout protein biology"),
        ("FFM", "trusted Stage 2 evidence was blocked by evaluator provenance"),
        ("RSB-FM", "stochastic residual bridge failed real-data protein fidelity"),
        ("PCFM", "time-local paired-coordinate rule passed and locked"),
    ]
    for j, (name, reason) in enumerate(failures):
        ax.text(0.05, 0.18 - 0.035 * j, f"{name}: {reason}", ha="left", va="center", fontsize=9.2, color="#334155")

    ax.text(0.63, 0.18, "Final metric context", fontsize=11, weight="bold", color="#111827")
    ax.text(0.63, 0.13, "Residual SB-FM real panel: 0.031822", fontsize=9.2, color="#334155")
    ax.text(0.63, 0.095, "CRUFM reference: 0.025575", fontsize=9.2, color="#334155")
    ax.text(0.63, 0.06, "Final CITE-PCFM lock: 0.022198", fontsize=9.2, color="#166534", weight="bold")

    savefig(
        fig,
        "search_landscape00_five_algorithm_lineage",
        "Five-algorithm lineage",
        "Can a reader see the full five-algorithm evolution, including what each generation falsified and what PCFM retained?",
    )


def fig_search01_final_panel_curve(runs: pd.DataFrame, baselines: pd.DataFrame) -> None:
    d = runs[runs["dataset_id"].eq(FINAL_DATASET) & runs["protein_energy"].notna()].copy()
    d = d.sort_values("created_at").reset_index(drop=True)
    d["final_panel_order"] = np.arange(1, len(d) + 1)
    promoted = d[d["is_promoted"]].copy()
    promoted["promoted_best_so_far"] = promoted["protein_energy"].cummin()

    fig, ax = plt.subplots(figsize=(13.2, 6.5))
    for alg, sub in d.groupby("algorithm"):
        ax.scatter(sub["final_panel_order"], sub["protein_energy"], s=55, color=ALG_COLORS.get(alg, "#94A3B8"), alpha=0.32, label=alg.replace("cite_", ""), edgecolor="white")
    ax.plot(promoted["final_panel_order"], promoted["promoted_best_so_far"], color="#B73A3A", lw=3.4, label="promoted best so far")
    raw_best = d.loc[d["protein_energy"].idxmin()]
    final = d[d["is_final_lock"]].tail(1)
    if len(final):
        ax.scatter(final["final_panel_order"], final["protein_energy"], s=260, marker="*", color="#16A34A", edgecolor="white", zorder=6, label="final locked")
    ax.scatter([raw_best["final_panel_order"]], [raw_best["protein_energy"]], s=160, marker="X", color="#111827", edgecolor="white", zorder=5, label="raw metric minimum")

    for _, r in baselines.dropna(subset=["protein_energy"]).iterrows():
        if r["algorithm"] in ["crufm", "vgfm", "sf2m", "wfrfm"]:
            ax.axhline(r["protein_energy"], color="#64748B", alpha=0.35, lw=1.1)
            ax.text(d["final_panel_order"].max() + 0.6, r["protein_energy"], r["algorithm"], fontsize=8, va="center", color="#475569")
    ax.axhline(0.022906601428985596, color="#111827", ls="--", lw=1.2)
    ax.text(1, 0.0234, "Stage 2 pass threshold", fontsize=9, color="#111827")
    ax.set_yscale("log")
    ax.set_xlabel("same-panel run order in ~/.cellcompass registry")
    ax.set_ylabel("protein_energy, log scale")
    ax.set_title("Same GSE305370 panel: quantitative algorithm search curve", weight="bold")
    ax.legend(frameon=False, ncol=3, fontsize=8)
    savefig(fig, "search_landscape01_same_panel_metric_curve", "Same-panel metric search curve", "How did the protein metric evolve on the final CITE-seq panel, and where is the final locked run relative to baselines and rejected minima?")


def fig_search02_family_records(runs: pd.DataFrame) -> None:
    d = runs[runs["protein_energy"].notna()].copy()
    d["context"] = np.where(d["dataset_id"].eq(FINAL_DATASET), "final GSE305370", np.where(d["dataset_id"].str.contains("sim", na=False), "controlled simulation", "debug/proportional"))
    records = []
    for (context, alg), sub in d.groupby(["context", "algorithm"]):
        best = sub.sort_values("protein_energy").iloc[0]
        records.append(best[["context", "algorithm", "protein_energy", "w1_mean", "decision", "run_id"]])
    rec = pd.DataFrame(records).sort_values(["context", "protein_energy"])
    rec.to_csv(TABLE_DIR / "algorithm_family_best_records_by_context.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), sharey=False)
    contexts = ["debug/proportional", "controlled simulation", "final GSE305370"]
    for ax, context in zip(axes, contexts):
        sub = rec[rec["context"] == context].sort_values("protein_energy", ascending=True)
        if sub.empty:
            ax.axis("off")
            continue
        ax.scatter(sub["protein_energy"], sub["algorithm"], s=140, c=[ALG_COLORS.get(a, "#94A3B8") for a in sub["algorithm"]], edgecolor="white")
        for _, r in sub.iterrows():
            ax.plot([0, r["protein_energy"]], [r["algorithm"], r["algorithm"]], color=ALG_COLORS.get(r["algorithm"], "#94A3B8"), alpha=0.35, lw=3)
        ax.set_xscale("log")
        ax.set_title(context, fontsize=12, weight="bold")
        ax.set_xlabel("best protein_energy")
        ax.tick_params(axis="y", labelsize=8)
    fig.suptitle("Algorithm-family records by search context", y=1.03, weight="bold")
    savefig(fig, "search_landscape02_family_records_by_context", "Algorithm family records", "Which algorithm families produced metric records in each search context without pretending all contexts are head-to-head benchmarks?")


def build_embedding(df: pd.DataFrame, numeric_cols: list[str], cat_cols: list[str], n_components: int = 2) -> tuple[np.ndarray, list[str]]:
    use = df.copy()
    Xnum = use[numeric_cols].copy()
    for col in numeric_cols:
        if col not in Xnum:
            Xnum[col] = np.nan
        vals = Xnum[col].astype(float)
        if vals.notna().sum() == 0:
            Xnum[col] = 0.0
        else:
            Xnum[col] = vals.fillna(vals.median())
    Xnum = np.log10(np.clip(Xnum.astype(float), 1e-8, None))
    Xnum = StandardScaler().fit_transform(Xnum)
    parts = [Xnum]
    names = list(numeric_cols)
    if cat_cols:
        enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        Xcat = enc.fit_transform(use[cat_cols].fillna("missing").astype(str))
        parts.append(Xcat)
        names.extend(list(enc.get_feature_names_out(cat_cols)))
    X = np.hstack(parts)
    emb = PCA(n_components=n_components, random_state=20260527).fit_transform(X)
    return emb, names


def fig_search03_pcfm_embedding(runs: pd.DataFrame) -> None:
    d = runs[runs["algorithm"].eq("cite_paired_coord_fm") & runs["dataset_id"].eq(FINAL_DATASET) & runs["protein_energy"].notna()].copy()
    d = d.sort_values("created_at").reset_index(drop=True)
    d["local_order"] = np.arange(1, len(d) + 1)
    numeric = [
        "protein_energy",
        "w1_mean",
        "rna_energy_guardrail",
        "paired_cov_error",
        "runtime_sec",
        "elapsed_sec",
        "cite_pcfm__lambda_p",
        "cite_pcfm__alpha_protein",
        "cite_pcfm__endpoint_protein_weight",
        "cite_pcfm__protein_head_hidden_dim",
        "cite_pcfm__time_fourier_features",
        "cite_pcfm__time_fourier_scale",
        "cite_pcfm__protein_time_weight_width",
        "training__plan__0__cite_pcfm__protein_time_weight_width",
        "training__plan__0__lr",
        "training__plan__0__flow_matching__coupling__reg",
    ]
    numeric = [c for c in numeric if c in d.columns]
    emb, _ = build_embedding(d, numeric, ["stage", "decision"], 2)
    d["embed1"], d["embed2"] = emb[:, 0], emb[:, 1]
    d.to_csv(TABLE_DIR / "pcfm_final_panel_design_embedding.csv", index=False)

    fig, ax = plt.subplots(figsize=(9.6, 7.4))
    sc = ax.scatter(d["embed1"], d["embed2"], c=d["protein_energy"], cmap="magma_r", s=np.clip(1500 / np.clip(d["w1_mean"], 1, None), 80, 210), edgecolor="white", alpha=0.9)
    for i in range(len(d) - 1):
        if i % 3 == 0:
            add_arrow(ax, (d.loc[i, "embed1"], d.loc[i, "embed2"]), (d.loc[i + 1, "embed1"], d.loc[i + 1, "embed2"]), color="#94A3B8", lw=1.0)
    promoted = d[d["is_promoted"]]
    ax.scatter(promoted["embed1"], promoted["embed2"], s=70, facecolors="none", edgecolors="#111827", linewidths=1.2, label="promoted")
    final = d[d["is_final_lock"]]
    if len(final):
        ax.scatter(final["embed1"], final["embed2"], s=280, marker="*", color="#16A34A", edgecolor="white", zorder=7, label="final locked")
    best_raw = d.loc[d["protein_energy"].idxmin()]
    ax.scatter([best_raw["embed1"]], [best_raw["embed2"]], s=180, marker="X", color="#111827", edgecolor="white", zorder=6, label="raw min")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("protein_energy")
    ax.set_title("CITE-PCFM design embedding: the search moved into a low-energy basin", weight="bold")
    ax.set_xlabel("design embedding PC1")
    ax.set_ylabel("design embedding PC2")
    ax.legend(frameon=False, fontsize=9)
    savefig(fig, "search_landscape03_pcfm_design_embedding", "PCFM design embedding", "Can the CITE-PCFM trials be viewed as a quantitative design-space trajectory ending in a low-energy basin?")


def fig_search04_global_embedding(runs: pd.DataFrame) -> None:
    d = runs[runs["protein_energy"].notna()].copy().reset_index(drop=True)
    numeric = [
        "protein_energy",
        "w1_mean",
        "tmv_mean",
        "rna_energy_guardrail",
        "paired_cov_error",
        "residual_energy",
        "runtime_sec",
        "elapsed_sec",
        "completed_epochs",
    ]
    numeric = [c for c in numeric if c in d.columns]
    emb, _ = build_embedding(d, numeric, ["algorithm", "dataset_id", "stage"], 2)
    d["embed1"], d["embed2"] = emb[:, 0], emb[:, 1]
    d.to_csv(TABLE_DIR / "global_cite_algorithm_metric_embedding.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 7.6))
    for alg, sub in d.groupby("algorithm"):
        ax.scatter(sub["embed1"], sub["embed2"], s=70, color=ALG_COLORS.get(alg, "#94A3B8"), alpha=0.68, edgecolor="white", label=alg.replace("cite_", ""))
    final = d[d["is_final_lock"]]
    if len(final):
        ax.scatter(final["embed1"], final["embed2"], s=300, marker="*", color="#16A34A", edgecolor="white", zorder=7, label="final locked")
        ax.text(float(final["embed1"].iloc[0]) + 0.15, float(final["embed2"].iloc[0]), "final PCFM", fontsize=10, weight="bold", color="#166534")
    ax.set_title("Global CITE algorithm metric landscape from ~/.cellcompass", weight="bold")
    ax.set_xlabel("algorithm landscape PC1")
    ax.set_ylabel("algorithm landscape PC2")
    ax.legend(frameon=False, ncol=2, fontsize=8)
    savefig(fig, "search_landscape04_global_algorithm_embedding", "Global algorithm embedding", "Where does the final CITE-PCFM run sit relative to prior CITE algorithm families in a metric/context embedding?")


def fig_search05_mechanism_waterfall() -> None:
    path = TABLE_DIR / "lifecycle_milestones.csv"
    if path.exists():
        milestones = pd.read_csv(path)
    else:
        milestones = pd.DataFrame(
            [
                ("Residual SB-FM", 0, 0.031822),
                ("PCFM initial", 2, 0.534965),
                ("endpoint/capacity", 8, 0.039508),
                ("Fourier time", 32, 0.023627),
                ("time-local weights", 39, 0.022801),
                ("width tuning", 48, 0.022218),
                ("final lock", 70, 0.022198),
            ],
            columns=["step", "trial", "metric"],
        )
    m = milestones[milestones["trial"] > 0].copy()
    m["quality"] = -np.log10(m["metric"])
    m["delta_quality"] = m["quality"].diff().fillna(0)

    fig, ax = plt.subplots(figsize=(11.5, 5.5))
    x = np.arange(len(m))
    colors = ["#B73A3A" if v > 0 else "#94A3B8" for v in m["delta_quality"]]
    ax.bar(x, m["delta_quality"], color=colors, width=0.72)
    ax.axhline(0, color="#111827", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([str(s).replace(" ", "\n") for s in m["step"]], fontsize=8)
    ax.set_ylabel("change in -log10(protein_energy)")
    ax.set_title("Mechanism-level search gains: which repairs actually moved the metric?", weight="bold")
    for xi, (_, r) in zip(x, m.iterrows()):
        ax.text(xi, r["delta_quality"] + (0.03 if r["delta_quality"] >= 0 else -0.06), f"{r['metric']:.4f}", ha="center", va="bottom" if r["delta_quality"] >= 0 else "top", fontsize=8)
    savefig(fig, "search_landscape05_mechanism_gain_waterfall", "Mechanism gain waterfall", "Which conceptual repair caused each quantitative jump in protein-energy quality?")


def fig_search06_parameter_response(runs: pd.DataFrame) -> None:
    d = runs[runs["algorithm"].eq("cite_paired_coord_fm") & runs["dataset_id"].eq(FINAL_DATASET) & runs["protein_energy"].notna()].copy()
    d["width"] = d["training__plan__0__cite_pcfm__protein_time_weight_width"].fillna(d.get("cite_pcfm__protein_time_weight_width", np.nan))
    d["fourier"] = d["cite_pcfm__time_fourier_features"]
    d["endpoint"] = d["cite_pcfm__endpoint_protein_weight"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True)
    panels = [("endpoint", "endpoint protein weight"), ("fourier", "time Fourier features"), ("width", "time-local width")]
    for ax, (col, label) in zip(axes, panels):
        sub = d[d[col].notna()].copy()
        if sub.empty:
            ax.axis("off")
            continue
        sc = ax.scatter(sub[col], sub["protein_energy"], c=sub["w1_mean"], cmap="viridis", s=90, edgecolor="white", alpha=0.85)
        promoted = sub[sub["is_promoted"]]
        ax.scatter(promoted[col], promoted["protein_energy"], s=120, facecolors="none", edgecolors="#111827", linewidths=1.2)
        ax.set_yscale("log")
        ax.axhline(0.022906601428985596, color="#111827", ls="--", lw=1)
        ax.set_xlabel(label)
        ax.set_title(label, fontsize=11, weight="bold")
    axes[0].set_ylabel("protein_energy")
    cb = fig.colorbar(sc, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("W1 guardrail")
    fig.suptitle("Parameter response surfaces show why final tuning became local", y=1.03, weight="bold")
    savefig(fig, "search_landscape06_parameter_response_surfaces", "Parameter response surfaces", "Which parameter axes produced real gains and which became saturated local tuning?")


def value_at_min(sub: pd.DataFrame, metric_col: str, value_col: str) -> float:
    d = sub[sub[metric_col].notna()].copy()
    if d.empty or value_col not in d:
        return float("nan")
    idx = d[metric_col].idxmin()
    val = d.loc[idx, value_col]
    return float(val) if pd.notna(val) else float("nan")


def finite_min(values: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce")
    return float(vals.min()) if vals.notna().any() else float("nan")


def frac_contains(texts: pd.Series, pattern: str) -> float:
    if texts.empty:
        return 0.0
    return float(texts.fillna("").astype(str).str.contains(pattern, case=False, regex=True).mean())


def quality(value: float, missing_value: float = 0.0) -> float:
    if value is None or not np.isfinite(value) or value <= 0:
        return missing_value
    return float(-np.log10(value))


def build_family_waddington_features(runs: pd.DataFrame, trials: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
    lineages = lineage_rows(baselines)
    crufm = baselines[baselines["algorithm"].eq("crufm")]
    crufm_energy = float(crufm["protein_energy"].iloc[0]) if len(crufm) and pd.notna(crufm["protein_energy"].iloc[0]) else float("nan")
    family_specs = [
        {
            "family_key": "VGFM",
            "family_short": "VGFM",
            "algorithms": ["cite_vgfm"],
            "design_prior": "generic transport",
            "retained_lesson": "pair CITE coordinates",
            "paired_geometry": 0.0,
            "factorized_readout": 0.0,
            "stochastic_residual": 0.0,
            "time_local_readout": 0.0,
        },
        {
            "family_key": "BFM",
            "family_short": "BFM",
            "algorithms": ["cite_bfm", "cite_bfm_balanced"],
            "design_prior": "balanced endpoint repair",
            "retained_lesson": "separate protein readout",
            "paired_geometry": 1.0,
            "factorized_readout": 0.0,
            "stochastic_residual": 0.0,
            "time_local_readout": 0.0,
        },
        {
            "family_key": "FFM",
            "family_short": "FFM",
            "algorithms": ["cite_factorized_fm"],
            "design_prior": "factorized readout",
            "retained_lesson": "lock evaluator provenance",
            "paired_geometry": 1.0,
            "factorized_readout": 1.0,
            "stochastic_residual": 0.0,
            "time_local_readout": 0.0,
        },
        {
            "family_key": "RSB-FM",
            "family_short": "RSB-FM",
            "algorithms": ["cite_residual_sb_fm"],
            "design_prior": "stochastic residual bridge",
            "retained_lesson": "drop stochastic residual",
            "paired_geometry": 1.0,
            "factorized_readout": 1.0,
            "stochastic_residual": 1.0,
            "time_local_readout": 0.0,
        },
        {
            "family_key": "PCFM",
            "family_short": "PCFM",
            "algorithms": ["cite_paired_coord_fm"],
            "design_prior": "time-local paired coordinates",
            "retained_lesson": "lock reproducible basin",
            "paired_geometry": 1.0,
            "factorized_readout": 1.0,
            "stochastic_residual": 0.0,
            "time_local_readout": 1.0,
        },
    ]
    rows = []
    for order, spec in enumerate(family_specs, start=1):
        rsub = runs[runs["algorithm"].isin(spec["algorithms"])].copy()
        tsub = trials[trials["algorithm"].isin(spec["algorithms"])].copy()
        reasons = pd.concat(
            [
                rsub.get("decision_reason", pd.Series(dtype=str)).fillna("").astype(str),
                tsub.get("decision_reasons", pd.Series(dtype=str)).fillna("").astype(str),
            ],
            ignore_index=True,
        )
        metric_rows = rsub[rsub["protein_energy"].notna()].copy()
        final_rows = metric_rows[metric_rows["dataset_id"].eq(FINAL_DATASET)].copy()
        simulation_rows = metric_rows[metric_rows["dataset_id"].fillna("").str.contains("sim", case=False, na=False)].copy()
        debug_rows = metric_rows[~metric_rows["dataset_id"].eq(FINAL_DATASET) & ~metric_rows["dataset_id"].fillna("").str.contains("sim", case=False, na=False)].copy()
        lineage = lineages[lineages["family"].str.contains(spec["family_key"].replace("-", "_"), case=False, regex=False)]
        if spec["family_key"] == "BFM":
            lineage = lineages[lineages["family"].str.contains("bfm", case=False, regex=False)]
        elif spec["family_key"] == "RSB-FM":
            lineage = lineages[lineages["family"].str.contains("residual", case=False, regex=False)]
        elif spec["family_key"] == "PCFM":
            lineage = lineages[lineages["family"].str.contains("paired_coord", case=False, regex=False)]
        evidence_metric = float(lineage["metric"].iloc[0]) if len(lineage) and pd.notna(lineage["metric"].iloc[0]) else float("nan")
        best_any = finite_min(metric_rows["protein_energy"]) if not metric_rows.empty else float("nan")
        best_final = finite_min(final_rows["protein_energy"]) if not final_rows.empty else float("nan")
        best_sim = finite_min(simulation_rows["protein_energy"]) if not simulation_rows.empty else float("nan")
        best_debug = finite_min(debug_rows["protein_energy"]) if not debug_rows.empty else float("nan")
        best_w1 = value_at_min(metric_rows, "protein_energy", "w1_mean")
        best_tmv = value_at_min(metric_rows, "protein_energy", "tmv_mean")
        total_decisions = max(len(rsub) + len(tsub), 1)
        decisions = pd.concat([rsub.get("decision", pd.Series(dtype=str)), tsub.get("decision", pd.Series(dtype=str))], ignore_index=True).fillna("").astype(str)
        provenance_block = 1.0 if spec["family_key"] == "FFM" else frac_contains(reasons, "provenance|evaluator|required|sibling|arbitration")
        no_trusted_stage2 = 1.0 if spec["family_key"] == "FFM" else 0.0
        comparator_miss = 1.0 if np.isfinite(best_final) and np.isfinite(crufm_energy) and best_final > crufm_energy else 0.0
        locked_pass = 1.0 if bool(rsub.get("is_final_lock", pd.Series(dtype=bool)).any()) else 0.0
        pass_crufm = 1.0 if np.isfinite(best_final) and np.isfinite(crufm_energy) and best_final < crufm_energy else 0.0
        metric_quality = quality(evidence_metric, missing_value=0.0)
        lifecycle_penalty = 0.45 * provenance_block + 0.35 * comparator_miss + 0.25 * frac_contains(reasons, "tmv hard gate failed") + 0.2 * no_trusted_stage2
        lifecycle_bonus = 0.35 * locked_pass + 0.2 * pass_crufm
        evidence_depth = max(0.12, metric_quality - lifecycle_penalty + lifecycle_bonus)
        row = {
            "order": order,
            "family_key": spec["family_key"],
            "family_short": spec["family_short"],
            "algorithms": "+".join(spec["algorithms"]),
            "design_prior": spec["design_prior"],
            "retained_lesson": spec["retained_lesson"],
            "n_runs": float(len(rsub)),
            "n_trials": float(len(tsub)),
            "promote_rate": float(decisions.str.contains("promote", case=False, na=False).sum() / total_decisions),
            "reject_rate": float(decisions.str.contains("reject", case=False, na=False).sum() / total_decisions),
            "final_panel_frac": float(rsub["dataset_id"].eq(FINAL_DATASET).mean()) if len(rsub) else 0.0,
            "simulation_frac": float(rsub["dataset_id"].fillna("").str.contains("sim", case=False, na=False).mean()) if len(rsub) else 0.0,
            "best_any_protein_energy": best_any,
            "best_debug_protein_energy": best_debug,
            "best_simulation_protein_energy": best_sim,
            "best_final_panel_protein_energy": best_final,
            "evidence_protein_energy": evidence_metric,
            "best_w1_at_best_energy": best_w1,
            "best_tmv_at_best_energy": best_tmv,
            "metric_quality": metric_quality,
            "final_panel_quality": quality(best_final, missing_value=0.0),
            "w1_quality": quality(best_w1, missing_value=0.0),
            "tmv_quality": quality((best_tmv + 1e-4) if np.isfinite(best_tmv) else float("nan"), missing_value=0.0),
            "tmv_gate_fail_frac": frac_contains(reasons, "tmv hard gate failed"),
            "no_improvement_frac": frac_contains(reasons, "did not improve|not improve|regression tolerance"),
            "training_failed_frac": frac_contains(reasons, "training failed|returned an error|no usable metrics|missing"),
            "provenance_block": provenance_block,
            "no_trusted_stage2": no_trusted_stage2,
            "comparator_miss": comparator_miss,
            "locked_pass": locked_pass,
            "pass_crufm": pass_crufm,
            "evidence_depth": evidence_depth,
            "paired_geometry": spec["paired_geometry"],
            "factorized_readout": spec["factorized_readout"],
            "stochastic_residual": spec["stochastic_residual"],
            "time_local_readout": spec["time_local_readout"],
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    feature_cols = [
        "metric_quality",
        "final_panel_quality",
        "w1_quality",
        "tmv_quality",
        "promote_rate",
        "reject_rate",
        "final_panel_frac",
        "simulation_frac",
        "tmv_gate_fail_frac",
        "no_improvement_frac",
        "training_failed_frac",
        "provenance_block",
        "no_trusted_stage2",
        "comparator_miss",
        "locked_pass",
        "pass_crufm",
        "paired_geometry",
        "factorized_readout",
        "stochastic_residual",
        "time_local_readout",
    ]
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    X = StandardScaler().fit_transform(X)
    emb = PCA(n_components=2, random_state=20260527).fit_transform(X)
    if np.corrcoef(emb[:, 0], df["order"].to_numpy(dtype=float))[0, 1] < 0:
        emb[:, 0] *= -1
    df["embed1"] = emb[:, 0]
    df["embed2"] = emb[:, 1]
    df.to_csv(TABLE_DIR / "family_waddington_embedding.csv", index=False)
    return df


def fig_search07_family_waddington_embedding(runs: pd.DataFrame, trials: pd.DataFrame, baselines: pd.DataFrame) -> None:
    df = build_family_waddington_features(runs, trials, baselines)
    fig = plt.figure(figsize=(16.5, 9.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.42, 1.0], height_ratios=[1.0, 0.52], wspace=0.28, hspace=0.35)
    ax = fig.add_subplot(gs[0, 0])
    ax_metric = fig.add_subplot(gs[1, 0])
    ax_heat = fig.add_subplot(gs[:, 1])

    pad_x = max(1.0, 0.18 * (df["embed1"].max() - df["embed1"].min()))
    pad_y = max(1.0, 0.2 * (df["embed2"].max() - df["embed2"].min()))
    xs = np.linspace(df["embed1"].min() - pad_x, df["embed1"].max() + pad_x, 260)
    ys = np.linspace(df["embed2"].min() - pad_y, df["embed2"].max() + pad_y, 220)
    xx, yy = np.meshgrid(xs, ys)
    sigma_x = max(0.55, 0.16 * (xs.max() - xs.min()))
    sigma_y = max(0.55, 0.18 * (ys.max() - ys.min()))
    potential = np.zeros_like(xx, dtype=float)
    depths = df["evidence_depth"].to_numpy(dtype=float)
    depths = depths / np.nanmax(depths)
    for (_, row), depth in zip(df.iterrows(), depths):
        basin = np.exp(-(((xx - row["embed1"]) ** 2) / (2 * sigma_x**2) + ((yy - row["embed2"]) ** 2) / (2 * sigma_y**2)))
        potential -= depth * basin
    potential += 0.06 * (xx - xs.min()) / (xs.max() - xs.min())
    levels = np.linspace(float(np.nanmin(potential)), float(np.nanmax(potential)), 18)
    cf = ax.contourf(xx, yy, potential, levels=levels, cmap="viridis_r", alpha=0.82)
    ax.contour(xx, yy, potential, levels=levels[::2], colors="#334155", linewidths=0.35, alpha=0.35)

    for i in range(len(df) - 1):
        p0 = (df.loc[i, "embed1"], df.loc[i, "embed2"])
        p1 = (df.loc[i + 1, "embed1"], df.loc[i + 1, "embed2"])
        add_arrow(ax, p0, p1, color="#111827", rad=0.10, lw=1.35)
        mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
        ax.text(mx, my + 0.18, df.loc[i, "retained_lesson"], fontsize=8.2, color="#111827", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#CBD5E1", alpha=0.86))

    status_edges = {"VGFM": "#DC2626", "BFM": "#DC2626", "FFM": "#D97706", "RSB-FM": "#DC2626", "PCFM": "#16A34A"}
    for _, row in df.iterrows():
        color = ALG_COLORS.get("cite_paired_coord_fm" if row["family_key"] == "PCFM" else "cite_residual_sb_fm" if row["family_key"] == "RSB-FM" else "cite_factorized_fm" if row["family_key"] == "FFM" else "cite_bfm_balanced" if row["family_key"] == "BFM" else "cite_vgfm", "#64748B")
        size = 430 + 360 * row["promote_rate"]
        ax.scatter(row["embed1"], row["embed2"], s=size, color=color, edgecolor=status_edges[row["family_key"]], linewidth=2.2, zorder=6)
        metric = "no trusted S2" if not np.isfinite(row["evidence_protein_energy"]) else f"{row['evidence_protein_energy']:.4f}"
        ax.text(row["embed1"], row["embed2"] + 0.34, f"{row['family_short']}\n{metric}", ha="center", va="bottom", fontsize=9.2, weight="bold", color="#0F172A", zorder=7)
        ax.text(row["embed1"], row["embed2"] - 0.38, row["design_prior"], ha="center", va="top", fontsize=7.8, color="#334155", zorder=7)

    cb = fig.colorbar(cf, ax=ax, fraction=0.036, pad=0.02)
    cb.set_label("algorithm potential: lower = deeper validated basin", fontsize=8.5)
    cb.ax.tick_params(labelsize=8)
    ax.set_title("Quantitative Waddington landscape of the five-algorithm CITE evolution", weight="bold", fontsize=13.5)
    ax.set_xlabel("PCA axis 1 from metric + failure-mode features", fontsize=10.5)
    ax.set_ylabel("PCA axis 2 from metric + failure-mode features", fontsize=10.5)
    ax.tick_params(labelsize=9)

    metric_df = df.copy()
    finite = metric_df["evidence_protein_energy"].notna()
    finite_df = metric_df.loc[finite, ["order", "evidence_protein_energy"]].copy()
    segment = []
    last_order = None
    for _, point in finite_df.iterrows():
        if last_order is None or point["order"] == last_order + 1:
            segment.append(point)
        else:
            seg_df = pd.DataFrame(segment)
            ax_metric.plot(seg_df["order"], seg_df["evidence_protein_energy"], color="#111827", lw=2.2, marker="o", markersize=7)
            segment = [point]
        last_order = point["order"]
    if segment:
        seg_df = pd.DataFrame(segment)
        ax_metric.plot(seg_df["order"], seg_df["evidence_protein_energy"], color="#111827", lw=2.2, marker="o", markersize=7)
    ax_metric.scatter(metric_df.loc[~finite, "order"], [0.055] * int((~finite).sum()), marker="v", s=110, color="#D97706", edgecolor="white", zorder=5)
    for _, row in metric_df.iterrows():
        if np.isfinite(row["evidence_protein_energy"]):
            y_text = row["evidence_protein_energy"] * (0.92 if row["family_short"] == "PCFM" else 1.08)
            ax_metric.text(row["order"], y_text, row["family_short"], ha="center", va="bottom" if row["family_short"] != "PCFM" else "top", fontsize=8)
        else:
            ax_metric.text(row["order"], 0.059, "FFM\nblocked", ha="center", va="bottom", fontsize=8, color="#92400E")
    crufm = baselines[baselines["algorithm"].eq("crufm")]
    if len(crufm) and pd.notna(crufm["protein_energy"].iloc[0]):
        ax_metric.axhline(float(crufm["protein_energy"].iloc[0]), ls="--", lw=1.2, color="#64748B")
        ax_metric.text(4.9, float(crufm["protein_energy"].iloc[0]) * 1.06, "CRUFM ref", fontsize=8, color="#475569", ha="right", va="bottom")
    ax_metric.set_yscale("log")
    ax_metric.set_xlim(0.7, 5.25)
    ax_metric.set_xticks(metric_df["order"])
    ax_metric.set_xticklabels(metric_df["family_short"], fontsize=8)
    ax_metric.set_ylabel("trusted protein_energy", fontsize=10)
    ax_metric.set_title("Metric descent with the blocked generation left as a gap", fontsize=10.5, weight="bold")
    ax_metric.tick_params(axis="y", labelsize=8)

    heat_cols = [
        ("tmv_gate_fail_frac", "TMV fail"),
        ("no_improvement_frac", "No improve"),
        ("training_failed_frac", "Train fail"),
        ("provenance_block", "Provenance"),
        ("comparator_miss", "Miss CRUFM"),
        ("no_trusted_stage2", "No S2"),
        ("locked_pass", "Locked"),
        ("time_local_readout", "Time-local"),
    ]
    heat = df[[c for c, _ in heat_cols]].copy()
    heat.index = df["family_short"]
    sns.heatmap(
        heat,
        ax=ax_heat,
        cmap="rocket_r",
        vmin=0,
        vmax=1,
        linewidths=0.6,
        linecolor="white",
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 7.5},
        cbar_kws={"label": "fraction / binary feature"},
    )
    ax_heat.set_xticklabels([label for _, label in heat_cols], rotation=45, ha="right", fontsize=8.5)
    ax_heat.set_yticklabels(ax_heat.get_yticklabels(), rotation=0, fontsize=9.5)
    ax_heat.set_xlabel("")
    ax_heat.set_ylabel("")
    ax_heat.set_title("Failure modes are encoded as quantitative features", weight="bold", fontsize=12.5)
    ax_heat.tick_params(labelsize=8.5)

    fig.suptitle(
        "The search path moved from fragile generic transport to a reproducible time-local CITE protein basin",
        y=0.995,
        fontsize=16,
        weight="bold",
    )
    savefig(
        fig,
        "search_landscape07_family_waddington_embedding",
        "Family Waddington embedding",
        "Can the five CITE algorithm generations be shown as a quantitative landscape where metrics, failure modes, and retained lessons define the evolutionary path?",
    )


def fig_search08_editorial_terrain(runs: pd.DataFrame, trials: pd.DataFrame, baselines: pd.DataFrame) -> None:
    """NBT-style 2.5D terrain: conceptual axes, quantitative heights."""

    df = build_family_waddington_features(runs, trials, baselines)
    coord_map = {
        "VGFM": (0.10, 0.18),
        "BFM": (0.31, 0.30),
        "FFM": (0.54, 0.55),
        "RSB-FM": (0.72, 0.70),
        "PCFM": (0.92, 0.86),
    }
    lesson_map = {
        "VGFM": "generic latent dynamics\nwere not enough",
        "BFM": "endpoint protein cost\nwas not protein motion",
        "FFM": "factorized residual idea\nneeded locked evaluator",
        "RSB-FM": "stochastic residuals\nmissed real CITE panel",
        "PCFM": "RNA-initialized\nprotein movie locked",
    }
    status_map = {"VGFM": "failed", "BFM": "failed", "FFM": "blocked", "RSB-FM": "failed", "PCFM": "locked"}
    status_colors = {"failed": "#E11D48", "blocked": "#F59E0B", "locked": "#16A34A"}
    node_colors = {"VGFM": "#94A3B8", "BFM": "#475569", "FFM": "#A78BFA", "RSB-FM": "#F59E0B", "PCFM": "#B73A3A"}

    pcfm_energy = float(df.loc[df["family_short"].eq("PCFM"), "evidence_protein_energy"].iloc[0])
    fallback_blocked_energy = 0.055
    df["terrain_x"] = df["family_short"].map(lambda k: coord_map[k][0])
    df["terrain_y"] = df["family_short"].map(lambda k: coord_map[k][1])
    df["plot_energy"] = df["evidence_protein_energy"].fillna(fallback_blocked_energy)
    df["terrain_height"] = np.log2(df["plot_energy"] / pcfm_energy)
    df["terrain_height"] += 1.15 * df["provenance_block"] + 0.55 * df["comparator_miss"] + 0.25 * df["no_trusted_stage2"]
    df.loc[df["family_short"].eq("PCFM"), "terrain_height"] = 0.0
    df["failure_status"] = df["family_short"].map(status_map)
    df["lesson"] = df["family_short"].map(lesson_map)
    df.to_csv(TABLE_DIR / "editorial_algorithm_terrain_coordinates.csv", index=False)

    gx = np.linspace(0.0, 1.0, 170)
    gy = np.linspace(0.0, 1.0, 150)
    xx, yy = np.meshgrid(gx, gy)
    base = 2.75 - 1.15 * xx - 0.85 * yy
    zz = base.copy()
    sx, sy = 0.17, 0.16
    for _, row in df.iterrows():
        node_base = 2.75 - 1.15 * row["terrain_x"] - 0.85 * row["terrain_y"]
        amp = row["terrain_height"] - node_base
        gauss = np.exp(-(((xx - row["terrain_x"]) ** 2) / (2 * sx**2) + ((yy - row["terrain_y"]) ** 2) / (2 * sy**2)))
        zz += amp * gauss
    # Add a visible but localized ridge at the FFM provenance blocker.
    ffm = df[df["family_short"].eq("FFM")].iloc[0]
    zz += 0.55 * np.exp(-(((xx - ffm["terrain_x"]) ** 2) / (2 * 0.045**2) + ((yy - ffm["terrain_y"]) ** 2) / (2 * 0.18**2)))
    zz = np.clip(zz, -0.05, 3.4)

    cmap = LinearSegmentedColormap.from_list(
        "algorithm_terrain",
        ["#0F7A5A", "#72B38D", "#F4E4A6", "#D97706", "#9F1239"],
    )

    fig = plt.figure(figsize=(18.6, 11.2))
    gs = fig.add_gridspec(
        3,
        2,
        height_ratios=[1.0, 0.12, 0.34],
        width_ratios=[1.56, 0.74],
        hspace=0.16,
        wspace=0.08,
    )
    ax3d = fig.add_subplot(gs[:2, 0], projection="3d")
    ax_metric = fig.add_subplot(gs[0, 1])
    ax_legend = fig.add_subplot(gs[1, 1])
    ax_genome = fig.add_subplot(gs[2, :])

    surf = ax3d.plot_surface(
        xx,
        yy,
        zz,
        cmap=cmap,
        linewidth=0,
        antialiased=True,
        alpha=0.92,
        rcount=120,
        ccount=140,
        shade=True,
    )
    z_floor = -0.18
    ax3d.contour(xx, yy, zz, zdir="z", offset=z_floor, levels=12, cmap=cmap, linewidths=0.7, alpha=0.75)

    order_df = df.sort_values("order").reset_index(drop=True)
    for i in range(len(order_df) - 1):
        p0 = order_df.loc[i]
        p1 = order_df.loc[i + 1]
        ax3d.quiver(
            p0["terrain_x"],
            p0["terrain_y"],
            p0["terrain_height"] + 0.07,
            p1["terrain_x"] - p0["terrain_x"],
            p1["terrain_y"] - p0["terrain_y"],
            p1["terrain_height"] - p0["terrain_height"],
            arrow_length_ratio=0.08,
            color="#111827",
            linewidth=2.2,
            alpha=0.92,
        )

    for _, row in order_df.iterrows():
        status = row["failure_status"]
        ax3d.scatter(
            row["terrain_x"],
            row["terrain_y"],
            row["terrain_height"] + 0.11,
            s=270,
            color=node_colors[row["family_short"]],
            edgecolor=status_colors[status],
            linewidth=2.4,
            depthshade=False,
            zorder=10,
        )
        metric_label = "blocked" if row["family_short"] == "FFM" else f"{row['plot_energy']:.4f}"
        ax3d.text(
            row["terrain_x"],
            row["terrain_y"],
            row["terrain_height"] + 0.34,
            f"{row['family_short']}\n{metric_label}",
            ha="center",
            va="bottom",
            fontsize=8.8,
            color="#0F172A",
            weight="bold",
            zorder=11,
        )
        ax3d.text(
            row["terrain_x"],
            row["terrain_y"],
            max(row["terrain_height"] - 0.32, z_floor + 0.05),
            row["lesson"],
            ha="center",
            va="top",
            fontsize=7.3,
            color="#334155",
            zorder=11,
        )

    # A translucent wall makes the FFM registry blocker visually distinct from biological/model failure.
    wall_x = np.array([[ffm["terrain_x"] - 0.018, ffm["terrain_x"] + 0.018], [ffm["terrain_x"] - 0.018, ffm["terrain_x"] + 0.018]])
    wall_y = np.array([[ffm["terrain_y"] - 0.20, ffm["terrain_y"] - 0.20], [ffm["terrain_y"] + 0.20, ffm["terrain_y"] + 0.20]])
    wall_z = np.array([[0.20, 0.20], [3.05, 3.05]])
    ax3d.plot_surface(wall_x, wall_y, wall_z, color="#F59E0B", alpha=0.23, linewidth=0)
    ax3d.text(ffm["terrain_x"] + 0.09, ffm["terrain_y"] + 0.19, 2.95, "Stage-2\nevaluator\ngate", color="#92400E", fontsize=8.2, weight="bold")

    ax3d.set_xlim(0, 1.02)
    ax3d.set_ylim(0, 1.02)
    ax3d.set_zlim(z_floor, 3.45)
    ax3d.view_init(elev=30, azim=-58)
    ax3d.set_xlabel("protein is an explicit time-resolved state", labelpad=12, fontsize=9.5)
    ax3d.set_ylabel("CITE-specific RNA-to-protein structure", labelpad=12, fontsize=9.5)
    ax3d.set_zlabel("evidence penalty", labelpad=8, fontsize=9.5)
    ax3d.set_xticks([0, 0.5, 1.0])
    ax3d.set_yticks([0, 0.5, 1.0])
    ax3d.set_zticks([0, 1, 2, 3])
    ax3d.tick_params(labelsize=7.5, pad=0)
    ax3d.set_title("Algorithm search terrain: each failure carved the path to RNA-to-protein dynamics", pad=18, weight="bold", fontsize=13.8)
    ax3d.xaxis.pane.set_facecolor((1, 1, 1, 0.0))
    ax3d.yaxis.pane.set_facecolor((1, 1, 1, 0.0))
    ax3d.zaxis.pane.set_facecolor((1, 1, 1, 0.0))

    finite = order_df[order_df["evidence_protein_energy"].notna()].copy()
    ax_metric.plot(finite["order"], finite["evidence_protein_energy"], color="#111827", lw=2.5, marker="o", markersize=8)
    blocked = order_df[order_df["family_short"].eq("FFM")]
    ax_metric.scatter(blocked["order"], [fallback_blocked_energy], marker="v", s=150, color="#F59E0B", edgecolor="white", zorder=5)
    ax_metric.text(float(blocked["order"].iloc[0]), fallback_blocked_energy * 1.12, "blocked\nnot failed", ha="center", va="bottom", fontsize=8, color="#92400E", weight="bold")
    crufm = baselines[baselines["algorithm"].eq("crufm")]
    if len(crufm) and pd.notna(crufm["protein_energy"].iloc[0]):
        ax_metric.axhline(float(crufm["protein_energy"].iloc[0]), color="#64748B", lw=1.2, ls="--")
        ax_metric.text(5.15, float(crufm["protein_energy"].iloc[0]), "CRUFM", va="center", ha="right", fontsize=8, color="#475569")
    ax_metric.set_yscale("log")
    ax_metric.set_xticks(order_df["order"])
    ax_metric.set_xticklabels(order_df["family_short"], rotation=0, fontsize=8.2)
    ax_metric.set_ylabel("trusted protein_energy", fontsize=9.2)
    ax_metric.set_title("Metric descent and the FFM gap", fontsize=10.5, weight="bold")
    ax_metric.grid(axis="y", color="#E2E8F0", lw=0.8)
    sns.despine(ax=ax_metric)

    ax_legend.set_axis_off()
    ax_legend.set_xlim(0, 1)
    ax_legend.set_ylim(0, 1)
    legend_items = [("failed", status_colors["failed"]), ("blocked", status_colors["blocked"]), ("locked", status_colors["locked"])]
    for i, (label, color) in enumerate(legend_items):
        ax_legend.scatter(0.08 + i * 0.25, 0.50, s=120, color=color, edgecolor="white")
        ax_legend.text(0.13 + i * 0.25, 0.50, label, va="center", ha="left", fontsize=8.5, color="#334155", weight="bold")
    ax_legend.text(
        0.05,
        0.08,
        "Height = protein_energy + lifecycle penalties; FFM is a registry gate, not a biological failure.",
        fontsize=8.1,
        color="#475569",
        wrap=True,
    )

    ax_genome.set_axis_off()
    genome_features = [
        ("paired_geometry", "paired\nstate"),
        ("factorized_readout", "residual\nchart"),
        ("stochastic_residual", "stochastic\nresidual"),
        ("time_local_readout", "time-local\nprotein"),
        ("provenance_block", "registry\ngate"),
        ("comparator_miss", "missed\nCRUFM"),
        ("locked_pass", "locked\nrelease"),
    ]
    ax_genome.set_xlim(-0.2, len(genome_features) + 1.35)
    ax_genome.set_ylim(-0.05, len(order_df) + 0.9)
    ax_genome.text(-0.15, len(order_df) + 0.45, "Design genome strip", fontsize=12.5, weight="bold", color="#111827", ha="left")
    for j, (_, label) in enumerate(genome_features):
        ax_genome.text(j + 1.05, len(order_df) + 0.05, label, ha="center", va="bottom", fontsize=8.2, color="#334155")
    for i, (_, row) in enumerate(order_df.iterrows()):
        y = len(order_df) - i - 0.45
        ax_genome.text(0.0, y, row["family_short"], ha="left", va="center", fontsize=9.2, weight="bold", color="#111827")
        ax_genome.text(0.58, y, status_map[row["family_short"]], ha="left", va="center", fontsize=8.0, color=status_colors[status_map[row["family_short"]]], weight="bold")
        for j, (feat, _) in enumerate(genome_features):
            val = float(row[feat]) if pd.notna(row[feat]) else 0.0
            fc = "#111827" if val > 0.5 else "#F1F5F9"
            ec = "#CBD5E1"
            if feat == "stochastic_residual" and val > 0.5:
                fc = "#F59E0B"
            if feat == "provenance_block" and val > 0.5:
                fc = "#F97316"
            if feat == "locked_pass" and val > 0.5:
                fc = "#16A34A"
            rect = FancyBboxPatch(
                (j + 0.82, y - 0.14),
                0.28,
                0.28,
                boxstyle="round,pad=0.018,rounding_size=0.035",
                facecolor=fc,
                edgecolor=ec,
                linewidth=0.9,
            )
            ax_genome.add_patch(rect)

    fig.suptitle(
        "From generic transport to RNA-initialized protein movies: a quantitative algorithm-search landscape",
        y=0.988,
        fontsize=17,
        weight="bold",
    )
    savefig(
        fig,
        "search_landscape08_editorial_2p5d_algorithm_terrain",
        "Editorial 2.5D algorithm terrain",
        "Can the full search be shown as a single editor-facing terrain where metric descent, failure gates, and design mutations explain why PCFM emerged?",
    )


def fig_search09_nbt_clean_basin(runs: pd.DataFrame, trials: pd.DataFrame, baselines: pd.DataFrame) -> None:
    """Cleaner Nature Biotechnology-style candidate: model-evidence path only."""

    df = build_family_waddington_features(runs, trials, baselines)
    keep = ["VGFM", "BFM", "RSB-FM", "PCFM"]
    df = df[df["family_short"].isin(keep)].copy()
    coord_map = {
        "VGFM": (0.12, 0.18),
        "BFM": (0.36, 0.34),
        "RSB-FM": (0.66, 0.58),
        "PCFM": (0.90, 0.82),
    }
    node_color = {
        "VGFM": "#94A3B8",
        "BFM": "#475569",
        "RSB-FM": "#F59E0B",
        "PCFM": "#B73A3A",
    }
    step_labels = {
        ("VGFM", "BFM"): "paired\nCITE state",
        ("BFM", "RSB-FM"): "protein\nresidual motion",
        ("RSB-FM", "PCFM"): "time-local\nprotein readout",
    }
    df["x"] = df["family_short"].map(lambda k: coord_map[k][0])
    df["y"] = df["family_short"].map(lambda k: coord_map[k][1])
    pcfm_energy = float(df.loc[df["family_short"].eq("PCFM"), "evidence_protein_energy"].iloc[0])
    df["energy_penalty"] = np.log2(df["evidence_protein_energy"] / pcfm_energy)
    df.loc[df["family_short"].eq("PCFM"), "energy_penalty"] = 0.0
    order_df = df.sort_values("order").reset_index(drop=True)
    order_df.to_csv(TABLE_DIR / "nbt_clean_algorithm_basin_coordinates.csv", index=False)

    gx = np.linspace(0.0, 1.0, 360)
    gy = np.linspace(0.0, 1.0, 290)
    xx, yy = np.meshgrid(gx, gy)
    surface = 2.75 - 1.30 * xx - 0.95 * yy
    sx, sy = 0.18, 0.16
    for _, row in order_df.iterrows():
        node_base = 2.75 - 1.30 * row["x"] - 0.95 * row["y"]
        correction = row["energy_penalty"] - node_base
        surface += correction * np.exp(-(((xx - row["x"]) ** 2) / (2 * sx**2) + ((yy - row["y"]) ** 2) / (2 * sy**2)))
    # Shape the successful valley as a broad basin rather than a narrow endpoint dot.
    pcfm = order_df[order_df["family_short"].eq("PCFM")].iloc[0]
    surface -= 0.32 * np.exp(-(((xx - pcfm["x"]) ** 2) / (2 * 0.22**2) + ((yy - pcfm["y"]) ** 2) / (2 * 0.20**2)))
    surface = np.clip(surface, -0.10, 2.80)

    cmap = LinearSegmentedColormap.from_list(
        "nbt_clean_basin",
        ["#096B5E", "#43A376", "#F2D98C", "#E18A2C", "#B91C1C"],
    )

    fig = plt.figure(figsize=(18.5, 9.8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.56, 0.82], height_ratios=[1.0, 0.42], hspace=0.34, wspace=0.15)
    ax = fig.add_subplot(gs[:, 0])
    ax_curve = fig.add_subplot(gs[0, 1])
    ax_genome = fig.add_subplot(gs[1, 1])

    levels = np.linspace(float(surface.min()), float(surface.max()), 20)
    cf = ax.contourf(xx, yy, surface, levels=levels, cmap=cmap, alpha=0.98)
    ax.contour(xx, yy, surface, levels=levels[::2], colors="white", linewidths=0.55, alpha=0.55)
    ax.contour(xx, yy, surface, levels=[0.12, 0.35, 0.70], colors=["#0F766E"], linewidths=[1.6, 1.0, 0.8], alpha=0.75)

    pts = order_df[["x", "y"]].to_numpy()
    dense = []
    for i in range(len(pts) - 1):
        p0, p1 = pts[i], pts[i + 1]
        for t in np.linspace(0, 1, 55, endpoint=False):
            # Smoothstep interpolation keeps the search path fluid without inventing extra nodes.
            s = t * t * (3 - 2 * t)
            bend = np.array([0.0, 0.035 * np.sin(np.pi * t)])
            dense.append((1 - s) * p0 + s * p1 + bend)
    dense.append(pts[-1])
    dense = np.asarray(dense)
    segments = np.stack([dense[:-1], dense[1:]], axis=1)
    path_values = np.linspace(float(order_df["energy_penalty"].iloc[0]), 0.0, len(segments))
    lc_shadow = LineCollection(segments, colors="#0F172A", linewidths=9.5, alpha=0.18, capstyle="round", zorder=6)
    ax.add_collection(lc_shadow)
    lc = LineCollection(segments, cmap="magma", norm=plt.Normalize(path_values.min(), path_values.max()), linewidths=5.6, capstyle="round", zorder=7)
    lc.set_array(path_values[::-1])
    ax.add_collection(lc)

    for i in range(len(order_df) - 1):
        p0 = order_df.loc[i]
        p1 = order_df.loc[i + 1]
        ax.annotate(
            "",
            xy=(p1["x"], p1["y"]),
            xytext=(p0["x"], p0["y"]),
            arrowprops=dict(arrowstyle="-|>", lw=0, color="#111827", mutation_scale=18, shrinkA=15, shrinkB=18, alpha=0.85),
            zorder=9,
        )
        label = step_labels[(p0["family_short"], p1["family_short"])]
        mx, my = (p0["x"] + p1["x"]) / 2, (p0["y"] + p1["y"]) / 2 + 0.055
        ax.text(
            mx,
            my,
            label,
            ha="center",
            va="center",
            fontsize=9.0,
            color="#0F172A",
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.32,rounding_size=0.18", fc="white", ec="#CBD5E1", alpha=0.92),
            zorder=12,
        )

    for _, row in order_df.iterrows():
        is_final = row["family_short"] == "PCFM"
        ax.scatter(
            row["x"],
            row["y"],
            s=540 if is_final else 360,
            color=node_color[row["family_short"]],
            edgecolor="white",
            linewidth=2.7,
            zorder=14,
        )
        if is_final:
            ax.scatter(row["x"], row["y"], s=920, facecolors="none", edgecolors="#0F766E", linewidth=2.4, zorder=13)
        label = f"{row['family_short']}\n{row['evidence_protein_energy']:.4f}"
        dy = 0.067 if row["family_short"] != "PCFM" else 0.075
        ax.text(row["x"], row["y"] + dy, label, ha="center", va="bottom", fontsize=10.5, color="#0F172A", weight="bold", zorder=15)

    ax.text(
        0.065,
        0.925,
        "high protein-energy search space",
        ha="left",
        va="center",
        fontsize=10.5,
        color="#7F1D1D",
        weight="bold",
    )
    ax.text(
        0.74,
        0.105,
        "low-energy RNA-to-protein basin",
        ha="left",
        va="center",
        fontsize=11.0,
        color="#065F46",
        weight="bold",
    )
    ax.set_xlim(0.02, 0.98)
    ax.set_ylim(0.04, 0.96)
    ax.set_xticks([0.12, 0.50, 0.90])
    ax.set_xticklabels(["generic\ntransport", "protein-aware\nstate", "RNA-initialized\nprotein movie"], fontsize=9.5)
    ax.set_yticks([0.18, 0.50, 0.82])
    ax.set_yticklabels(["weak\nCITE structure", "residual\nprotein structure", "time-resolved\nRNA->protein"], fontsize=9.5)
    ax.set_xlabel("algorithm design axis", fontsize=11.0, labelpad=8)
    ax.set_ylabel("biological claim depth", fontsize=11.0, labelpad=8)
    ax.set_title("A model-evidence landscape of CITE algorithm discovery", fontsize=16, weight="bold", pad=12)
    sns.despine(ax=ax, left=True, bottom=True)

    energies = order_df["evidence_protein_energy"].to_numpy(dtype=float)
    labels = order_df["family_short"].tolist()
    ypos = np.arange(len(order_df))[::-1]
    colors = [node_color[x] for x in labels]
    ax_curve.barh(ypos, energies, color=colors, alpha=0.88, height=0.58)
    for y, e, label in zip(ypos, energies, labels):
        ax_curve.text(e * 1.04, y, f"{e:.4f}", va="center", ha="left", fontsize=8.8, color="#334155")
    crufm = baselines[baselines["algorithm"].eq("crufm")]
    if len(crufm) and pd.notna(crufm["protein_energy"].iloc[0]):
        crufm_val = float(crufm["protein_energy"].iloc[0])
        ax_curve.axvline(crufm_val, color="#64748B", lw=1.4, ls="--")
        ax_curve.text(crufm_val * 1.04, len(order_df) - 0.15, "CRUFM\nreference", ha="left", va="top", fontsize=8.2, color="#475569")
    gain = energies[0] / energies[-1]
    ax_curve.text(0.002, -0.85, f"{gain:.1f}x lower protein_energy", fontsize=12.2, color="#065F46", weight="bold", ha="left")
    ax_curve.set_xscale("log")
    ax_curve.set_yticks(ypos)
    ax_curve.set_yticklabels(labels, fontsize=9.5)
    ax_curve.set_xlabel("protein_energy, log scale", fontsize=9.0, labelpad=2)
    ax_curve.set_title("Quantitative descent", fontsize=12.5, weight="bold")
    ax_curve.grid(axis="x", color="#E2E8F0", lw=0.8)
    sns.despine(ax=ax_curve)

    ax_genome.set_axis_off()
    genome = [
        ("paired_geometry", "paired\nstate"),
        ("factorized_readout", "residual\nchart"),
        ("stochastic_residual", "stochastic\nbridge"),
        ("time_local_readout", "time-local\nprotein"),
        ("locked_pass", "locked\nrelease"),
    ]
    ax_genome.set_xlim(-0.15, len(genome) + 0.55)
    ax_genome.set_ylim(-0.35, len(order_df) + 0.95)
    ax_genome.text(-0.12, len(order_df) + 0.42, "Design mutations", fontsize=12.5, weight="bold", color="#111827", ha="left")
    for j, (_, lab) in enumerate(genome):
        ax_genome.text(j + 0.95, len(order_df) + 0.05, lab, ha="center", va="bottom", fontsize=7.8, color="#334155")
    for i, (_, row) in enumerate(order_df.iterrows()):
        y = len(order_df) - i - 0.55
        ax_genome.text(0.0, y, row["family_short"], ha="left", va="center", fontsize=8.8, weight="bold", color="#0F172A")
        for j, (feat, _) in enumerate(genome):
            val = float(row[feat]) if pd.notna(row[feat]) else 0.0
            face = "#F1F5F9"
            if val > 0.5:
                face = "#0F172A"
            if feat == "stochastic_residual" and val > 0.5:
                face = "#F59E0B"
            if feat == "locked_pass" and val > 0.5:
                face = "#16A34A"
            patch = FancyBboxPatch(
                (j + 0.79, y - 0.15),
                0.32,
                0.30,
                boxstyle="round,pad=0.018,rounding_size=0.035",
                facecolor=face,
                edgecolor="#CBD5E1",
                linewidth=0.9,
            )
            ax_genome.add_patch(patch)

    fig.suptitle(
        "The algorithm search discovered a low-energy basin for RNA-conditioned protein dynamics",
        y=0.985,
        fontsize=18.2,
        weight="bold",
    )
    savefig(
        fig,
        "search_landscape09_nbt_clean_algorithm_basin",
        "NBT clean algorithm basin",
        "Can the main-text algorithm search figure focus on the model-evidence path, removing the registry-blocked FFM node from the visual story?",
    )


def fig_search10_editorial_3d_basin(runs: pd.DataFrame, trials: pd.DataFrame, baselines: pd.DataFrame) -> None:
    """Clean 3D basin view for the CellCompass algorithm discovery story."""

    df = build_family_waddington_features(runs, trials, baselines)
    keep = ["VGFM", "BFM", "RSB-FM", "PCFM"]
    df = df[df["family_short"].isin(keep)].copy()
    coord_map = {
        "VGFM": (0.12, 0.18),
        "BFM": (0.34, 0.34),
        "RSB-FM": (0.66, 0.58),
        "PCFM": (0.90, 0.82),
    }
    node_color = {
        "VGFM": "#94A3B8",
        "BFM": "#475569",
        "RSB-FM": "#F59E0B",
        "PCFM": "#B73A3A",
    }
    step_labels = {
        ("VGFM", "BFM"): "paired CITE state",
        ("BFM", "RSB-FM"): "protein residual motion",
        ("RSB-FM", "PCFM"): "time-local protein readout",
    }
    df["x"] = df["family_short"].map(lambda k: coord_map[k][0])
    df["y"] = df["family_short"].map(lambda k: coord_map[k][1])
    pcfm_energy = float(df.loc[df["family_short"].eq("PCFM"), "evidence_protein_energy"].iloc[0])
    df["energy_penalty"] = np.log2(df["evidence_protein_energy"] / pcfm_energy)
    df.loc[df["family_short"].eq("PCFM"), "energy_penalty"] = 0.0
    order_df = df.sort_values("order").reset_index(drop=True)
    order_df.to_csv(TABLE_DIR / "editorial_3d_algorithm_basin_coordinates.csv", index=False)

    gx = np.linspace(0.0, 1.0, 260)
    gy = np.linspace(0.0, 1.0, 230)
    xx, yy = np.meshgrid(gx, gy)
    surface = 2.90 - 1.30 * xx - 0.95 * yy
    sx, sy = 0.18, 0.16
    for _, row in order_df.iterrows():
        node_base = 2.90 - 1.30 * row["x"] - 0.95 * row["y"]
        correction = row["energy_penalty"] - node_base
        surface += correction * np.exp(-(((xx - row["x"]) ** 2) / (2 * sx**2) + ((yy - row["y"]) ** 2) / (2 * sy**2)))
    pcfm = order_df[order_df["family_short"].eq("PCFM")].iloc[0]
    surface -= 0.34 * np.exp(-(((xx - pcfm["x"]) ** 2) / (2 * 0.25**2) + ((yy - pcfm["y"]) ** 2) / (2 * 0.22**2)))
    surface += 0.18 * np.exp(-(((xx - 0.08) ** 2) / (2 * 0.24**2) + ((yy - 0.10) ** 2) / (2 * 0.24**2)))
    surface = np.clip(surface, -0.16, 2.92)

    def interp_z(x: float, y: float) -> float:
        ix = int(np.clip(np.searchsorted(gx, x) - 1, 0, len(gx) - 2))
        iy = int(np.clip(np.searchsorted(gy, y) - 1, 0, len(gy) - 2))
        x0, x1 = gx[ix], gx[ix + 1]
        y0, y1 = gy[iy], gy[iy + 1]
        tx = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
        ty = 0.0 if y1 == y0 else (y - y0) / (y1 - y0)
        z00 = surface[iy, ix]
        z10 = surface[iy, ix + 1]
        z01 = surface[iy + 1, ix]
        z11 = surface[iy + 1, ix + 1]
        return float((1 - tx) * (1 - ty) * z00 + tx * (1 - ty) * z10 + (1 - tx) * ty * z01 + tx * ty * z11)

    pts = order_df[["x", "y"]].to_numpy(dtype=float)
    dense = []
    for i in range(len(pts) - 1):
        p0, p1 = pts[i], pts[i + 1]
        vec = p1 - p0
        normal = np.array([-vec[1], vec[0]], dtype=float)
        norm = np.linalg.norm(normal)
        if norm > 0:
            normal /= norm
        for t in np.linspace(0, 1, 70, endpoint=False):
            s = t * t * (3 - 2 * t)
            bend = 0.028 * np.sin(np.pi * t) * normal
            dense.append((1 - s) * p0 + s * p1 + bend)
    dense.append(pts[-1])
    dense = np.asarray(dense)
    path_z = np.asarray([interp_z(float(x), float(y)) for x, y in dense]) + 0.12

    cmap = LinearSegmentedColormap.from_list(
        "editorial_3d_basin",
        ["#0A6B5B", "#37A67A", "#F1D88B", "#E18A2C", "#A93426"],
    )

    fig = plt.figure(figsize=(16.8, 9.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.72, 0.78], height_ratios=[0.54, 0.46], wspace=0.10, hspace=0.30)
    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax_family = fig.add_subplot(gs[0, 1])
    ax_internal = fig.add_subplot(gs[1, 1])

    ax3d.plot_surface(
        xx,
        yy,
        surface,
        cmap=cmap,
        linewidth=0,
        antialiased=True,
        alpha=0.96,
        rcount=150,
        ccount=170,
        shade=True,
    )
    z_floor = -0.28
    ax3d.contour(xx, yy, surface, zdir="z", offset=z_floor, levels=16, cmap=cmap, linewidths=0.8, alpha=0.78)
    ax3d.plot(dense[:, 0], dense[:, 1], path_z, color="#0F172A", lw=6.6, alpha=0.28, zorder=8)
    ax3d.plot(dense[:, 0], dense[:, 1], path_z + 0.025, color="#FFFFFF", lw=4.3, alpha=0.95, zorder=9)
    ax3d.plot(dense[:, 0], dense[:, 1], path_z + 0.050, color="#111827", lw=2.7, alpha=0.92, zorder=10)

    for i in range(len(order_df) - 1):
        p0 = order_df.loc[i]
        p1 = order_df.loc[i + 1]
        mid = 0.72 * np.array([p0["x"], p0["y"]]) + 0.28 * np.array([p1["x"], p1["y"]])
        end = 0.55 * np.array([p0["x"], p0["y"]]) + 0.45 * np.array([p1["x"], p1["y"]])
        z0 = interp_z(float(mid[0]), float(mid[1])) + 0.22
        z1 = interp_z(float(end[0]), float(end[1])) + 0.22
        ax3d.quiver(
            mid[0],
            mid[1],
            z0,
            end[0] - mid[0],
            end[1] - mid[1],
            z1 - z0,
            arrow_length_ratio=0.35,
            color="#111827",
            linewidth=1.8,
            alpha=0.95,
        )
        label_xy = 0.52 * np.array([p0["x"], p0["y"]]) + 0.48 * np.array([p1["x"], p1["y"]])
        ax3d.text(
            label_xy[0],
            label_xy[1],
            interp_z(float(label_xy[0]), float(label_xy[1])) + 0.38,
            step_labels[(p0["family_short"], p1["family_short"])],
            ha="center",
            va="center",
            fontsize=7.8,
            color="#0F172A",
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.28,rounding_size=0.16", fc="white", ec="#CBD5E1", alpha=0.92),
        )

    for _, row in order_df.iterrows():
        z = interp_z(float(row["x"]), float(row["y"])) + 0.16
        is_final = row["family_short"] == "PCFM"
        ax3d.scatter(
            row["x"],
            row["y"],
            z,
            s=440 if is_final else 300,
            color=node_color[row["family_short"]],
            edgecolor="white",
            linewidth=2.4,
            depthshade=False,
            zorder=16,
        )
        if is_final:
            ax3d.scatter(row["x"], row["y"], z + 0.004, s=820, facecolors="none", edgecolors="#0F766E", linewidth=2.2, depthshade=False, zorder=15)
        label = ax3d.text(
            row["x"],
            row["y"],
            z + (0.28 if is_final else 0.23),
            f"{row['family_short']}\n{row['evidence_protein_energy']:.4f}",
            ha="center",
            va="bottom",
            fontsize=8.7,
            color="#0F172A",
            weight="bold",
            zorder=20,
        )
        label.set_path_effects([pe.withStroke(linewidth=3.8, foreground="white", alpha=0.92)])

    ax3d.text(0.05, 0.11, 2.78, "high protein-energy\nsearch space", color="#7F1D1D", fontsize=10.2, weight="bold")
    ax3d.text(0.66, 0.94, 0.08, "low-energy RNA-to-protein\nprotein-movie basin", color="#065F46", fontsize=10.2, weight="bold")
    ax3d.set_xlim(0, 1.0)
    ax3d.set_ylim(0, 1.0)
    ax3d.set_zlim(z_floor, 3.05)
    ax3d.view_init(elev=27, azim=-58)
    try:
        ax3d.set_box_aspect((1.35, 1.05, 0.52))
    except Exception:
        pass
    ax3d.set_xlabel("algorithm design axis", labelpad=11, fontsize=9.2)
    ax3d.set_ylabel("RNA->protein biological claim depth", labelpad=11, fontsize=9.2)
    ax3d.set_zlabel("evidence penalty", labelpad=7, fontsize=9.2)
    ax3d.set_xticks([0.12, 0.50, 0.90])
    ax3d.set_xticklabels(["generic\ntransport", "protein-aware\nstate", "RNA-initialized\nprotein movie"], fontsize=7.8)
    ax3d.set_yticks([0.18, 0.50, 0.82])
    ax3d.set_yticklabels(["weak\nCITE structure", "residual\nprotein structure", "time-resolved\nRNA->protein"], fontsize=7.8)
    ax3d.set_zticks([0, 1, 2, 3])
    ax3d.tick_params(labelsize=7.5, pad=0)
    ax3d.xaxis.pane.set_facecolor((1, 1, 1, 0.0))
    ax3d.yaxis.pane.set_facecolor((1, 1, 1, 0.0))
    ax3d.zaxis.pane.set_facecolor((1, 1, 1, 0.0))
    ax3d.grid(True, color="#E2E8F0")
    ax3d.set_title("3D model-evidence terrain of the CITE algorithm search", fontsize=15.2, weight="bold", pad=14)

    energies = order_df["evidence_protein_energy"].to_numpy(dtype=float)
    ax_family.plot(order_df["family_short"], energies, color="#111827", lw=2.5, marker="o", markersize=7.5)
    for label, energy, color in zip(order_df["family_short"], energies, [node_color[x] for x in order_df["family_short"]]):
        ax_family.scatter(label, energy, s=86, color=color, edgecolor="white", linewidth=1.2, zorder=5)
        ax_family.text(label, energy * 1.08, f"{energy:.4f}", ha="center", va="bottom", fontsize=8.0, color="#334155")
    crufm = baselines[baselines["algorithm"].eq("crufm")]
    if len(crufm) and pd.notna(crufm["protein_energy"].iloc[0]):
        crufm_val = float(crufm["protein_energy"].iloc[0])
        ax_family.axhline(crufm_val, color="#64748B", lw=1.2, ls="--")
        ax_family.text(2.85, crufm_val * 1.06, "CRUFM reference", fontsize=7.8, color="#475569", ha="right", va="bottom")
    ax_family.set_yscale("log")
    ax_family.set_ylabel("protein_energy", fontsize=8.8)
    ax_family.set_title("Metric-evaluated family descent", fontsize=10.8, weight="bold")
    ax_family.tick_params(axis="x", labelsize=8.2, rotation=0)
    ax_family.tick_params(axis="y", labelsize=7.8)
    ax_family.grid(axis="y", color="#E2E8F0", lw=0.8)
    sns.despine(ax=ax_family)

    pcfm_trials = trials[trials["algorithm"].eq("cite_paired_coord_fm") & trials["protein_energy"].notna()].sort_values("created_at").copy()
    pcfm_trials["local_trial"] = np.arange(1, len(pcfm_trials) + 1)
    pcfm_trials["best_so_far"] = np.minimum.accumulate(pcfm_trials["protein_energy"].to_numpy(dtype=float))
    pcfm_trials[["local_trial", "stage", "decision", "protein_energy", "best_so_far"]].to_csv(TABLE_DIR / "pcfm_internal_search_energy_curve.csv", index=False)
    promote_mask = pcfm_trials["decision"].astype(str).str.contains("promote|locked|approve", case=False, na=False)
    ax_internal.scatter(
        pcfm_trials["local_trial"],
        pcfm_trials["protein_energy"],
        s=18,
        color=np.where(promote_mask, "#B73A3A", "#CBD5E1"),
        edgecolor="none",
        alpha=0.75,
        label="trial",
    )
    ax_internal.plot(pcfm_trials["local_trial"], pcfm_trials["best_so_far"], color="#0F766E", lw=2.3, label="best-so-far")
    ax_internal.scatter(
        [float(pcfm_trials["local_trial"].iloc[-1])],
        [float(pcfm_trials["best_so_far"].iloc[-1])],
        s=84,
        marker="*",
        color="#B73A3A",
        edgecolor="white",
        linewidth=0.9,
        zorder=6,
    )
    ax_internal.set_yscale("log")
    ax_internal.set_xlabel("CITE-PCFM trial order", fontsize=8.8)
    ax_internal.set_ylabel("protein_energy", fontsize=8.8)
    ax_internal.set_title("Internal PCFM search potential", fontsize=10.8, weight="bold")
    ax_internal.tick_params(labelsize=7.8)
    ax_internal.grid(axis="y", color="#E2E8F0", lw=0.8)
    ax_internal.legend(loc="upper right", fontsize=7.5, frameon=False, handlelength=1.2)
    sns.despine(ax=ax_internal)

    gain = float(order_df["evidence_protein_energy"].iloc[0] / order_df["evidence_protein_energy"].iloc[-1])
    fig.suptitle(
        f"CellCompass converged to CITE-PCFM: a {gain:.1f}x lower protein-energy basin for RNA-conditioned protein dynamics",
        y=0.984,
        fontsize=17.4,
        weight="bold",
    )
    savefig(
        fig,
        "search_landscape10_editorial_3d_algorithm_basin",
        "Editorial 3D algorithm basin",
        "Can the CellCompass CITE search be shown as a clean 3D model-evidence terrain with a quantitative PCFM internal search potential inset?",
    )


def fig_search11_cpanel_algorithm_discovery(runs: pd.DataFrame, trials: pd.DataFrame, baselines: pd.DataFrame) -> None:
    """C-panel-style algorithm discovery landscape with actual search metrics."""

    df = build_family_waddington_features(runs, trials, baselines)
    coord_x = {
        "VGFM": 0.18,
        "BFM": 0.42,
        "FFM": 0.83,
        "RSB-FM": 0.68,
        "PCFM": 0.83,
    }
    color_map = {
        "VGFM": "#2C5AA0",
        "BFM": "#23825F",
        "FFM": "#A1A1AA",
        "RSB-FM": "#E5792A",
        "PCFM": "#6D28D9",
    }
    pcfm_energy = float(df.loc[df["family_short"].eq("PCFM"), "evidence_protein_energy"].iloc[0])
    fallback_blocked_energy = 0.055
    df["plot_energy"] = df["evidence_protein_energy"].fillna(fallback_blocked_energy)
    df["penalty"] = np.log2(df["plot_energy"] / pcfm_energy)
    df.loc[df["family_short"].eq("PCFM"), "penalty"] = 0.0
    max_penalty = max(float(df["penalty"].max()), 1e-6)
    df["x"] = df["family_short"].map(coord_x)
    df["y"] = 0.18 + 0.64 * df["penalty"] / max_penalty
    df.loc[df["family_short"].eq("FFM"), "y"] = 0.54
    df["status"] = np.where(df["family_short"].eq("PCFM"), "selected", np.where(df["family_short"].eq("FFM"), "blocked", "evaluated"))
    df.to_csv(TABLE_DIR / "cpanel_algorithm_discovery_landscape_coordinates.csv", index=False)

    rng = np.random.default_rng(20260602)
    fig = plt.figure(figsize=(16.8, 6.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 0.82], wspace=0.12)
    ax = fig.add_subplot(gs[0, 0])
    ax_curve = fig.add_subplot(gs[0, 1])

    x = np.linspace(0.04, 0.98, 520)
    smooth = x * x * (3 - 2 * x)
    valley = 0.80 - 0.60 * smooth
    valley += 0.12 * np.exp(-((x - 0.34) ** 2) / (2 * 0.060**2))
    valley += 0.10 * np.exp(-((x - 0.62) ** 2) / (2 * 0.070**2))
    valley += 0.18 * np.exp(-((x - 0.86) ** 2) / (2 * 0.048**2))
    valley -= 0.15 * np.exp(-((x - 0.82) ** 2) / (2 * 0.105**2))

    for i in range(62):
        offset = (i - 31) * 0.0065
        phase = 0.38 * i
        y_line = valley + offset + 0.010 * np.sin(12 * x + phase) + 0.004 * np.sin(34 * x + phase * 0.7)
        alpha = 0.075 + 0.11 * np.exp(-((i - 31) ** 2) / (2 * 14**2))
        ax.plot(x, np.clip(y_line, 0.08, 0.93), color="#64748B", lw=0.62, alpha=alpha, zorder=1)

    random_x = np.linspace(0.16, 0.41, 8)
    random_y = np.linspace(0.18, 0.31, 8) + rng.normal(0, 0.012, 8)
    ax.plot(random_x, random_y, color="#9CA3AF", ls=(0, (3, 3)), lw=1.7, alpha=0.92, zorder=4)
    ax.scatter(random_x, random_y, s=34, color="#9CA3AF", edgecolor="white", linewidth=0.6, zorder=5)
    ax.text(0.10, 0.19, "random\ninitialization", ha="left", va="center", fontsize=8.0, color="#6B7280")

    cloud_rows = []
    for _, row in df.iterrows():
        family = row["family_short"]
        n = 230 if family != "FFM" else 150
        sx = 0.020 if family != "PCFM" else 0.016
        sy = 0.030 if family != "PCFM" else 0.020
        px = rng.normal(float(row["x"]), sx, n)
        py = rng.normal(float(row["y"]), sy, n)
        px = np.clip(px, 0.05, 0.97)
        py = np.clip(py, 0.08, 0.91)
        ax.scatter(
            px,
            py,
            s=8 if family != "FFM" else 7,
            color=color_map[family],
            alpha=0.44 if family != "FFM" else 0.25,
            edgecolor="none",
            zorder=6 if family != "FFM" else 3,
        )
        cloud_rows.extend({"family_short": family, "x": float(a), "y": float(b)} for a, b in zip(px, py))

    pd.DataFrame(cloud_rows).to_csv(TABLE_DIR / "cpanel_algorithm_discovery_landscape_cloud_points.csv", index=False)

    metric_path = df[df["family_short"].isin(["VGFM", "BFM", "RSB-FM", "PCFM"])].sort_values("order")
    pts = metric_path[["x", "y"]].to_numpy(dtype=float)
    path = []
    for i in range(len(pts) - 1):
        p0, p1 = pts[i], pts[i + 1]
        vec = p1 - p0
        normal = np.array([-vec[1], vec[0]], dtype=float)
        normal /= max(np.linalg.norm(normal), 1e-8)
        for t in np.linspace(0, 1, 42, endpoint=False):
            s = t * t * (3 - 2 * t)
            path.append((1 - s) * p0 + s * p1 + 0.020 * np.sin(np.pi * t) * normal)
    path.append(pts[-1])
    path = np.asarray(path)
    ax.plot(path[:, 0], path[:, 1], color="#111827", lw=2.1, ls=(0, (2.2, 2.2)), alpha=0.96, zorder=9)
    ax.scatter(path[::13, 0], path[::13, 1], s=34, color="#111827", edgecolor="white", linewidth=0.7, zorder=10)

    for _, row in df.iterrows():
        family = row["family_short"]
        if family == "PCFM":
            ax.scatter(row["x"], row["y"], marker="*", s=720, color=color_map[family], edgecolor="white", linewidth=1.8, zorder=13)
            ax.scatter(row["x"], row["y"], marker="*", s=980, facecolors="none", edgecolors="#C4B5FD", linewidth=1.5, zorder=12)
            ax.text(row["x"] + 0.035, row["y"] + 0.015, "CITE-PCFM\n(selected)", ha="left", va="center", fontsize=10.4, color="#312E81", weight="bold")
        elif family == "FFM":
            ax.text(row["x"] + 0.035, row["y"] + 0.010, "FFM\n(blocked)", ha="left", va="center", fontsize=9.0, color="#6B7280")
        else:
            ax.text(row["x"], row["y"] + 0.082, family, ha="center", va="bottom", fontsize=10.5, color=color_map[family], weight="bold")

    ax.annotate("", xy=(0.965, 0.105), xytext=(0.055, 0.105), arrowprops=dict(arrowstyle="->", lw=1.0, color="#374151"))
    ax.annotate("", xy=(0.060, 0.910), xytext=(0.060, 0.105), arrowprops=dict(arrowstyle="->", lw=1.0, color="#374151"))
    ax.text(0.50, 0.055, "design space", ha="center", va="center", fontsize=8.8, color="#374151")
    ax.text(0.022, 0.47, "internal search potential\n(lower is better)", ha="center", va="center", fontsize=8.8, color="#374151", rotation=90)
    ax.set_xlim(0.02, 1.0)
    ax.set_ylim(0.04, 0.96)
    ax.set_axis_off()
    ax.set_title("Algorithm discovery landscape", fontsize=14.0, weight="bold", loc="left", pad=8)

    pcfm_trials = trials[trials["algorithm"].eq("cite_paired_coord_fm") & trials["protein_energy"].notna()].sort_values("created_at").copy()
    pcfm_trials["local_trial"] = np.arange(1, len(pcfm_trials) + 1)
    pcfm_trials["best_so_far"] = np.minimum.accumulate(pcfm_trials["protein_energy"].to_numpy(dtype=float))
    pcfm_trials[["local_trial", "stage", "decision", "protein_energy", "best_so_far"]].to_csv(TABLE_DIR / "cpanel_internal_search_potential_curve.csv", index=False)
    ax_curve.step(pcfm_trials["local_trial"], pcfm_trials["best_so_far"], where="post", color="#6D28D9", lw=2.2)
    ax_curve.scatter(
        pcfm_trials["local_trial"],
        pcfm_trials["protein_energy"],
        s=13,
        color="#C4B5FD",
        alpha=0.46,
        edgecolor="none",
    )
    milestone_fracs = [0.08, 0.27, 0.53, 0.82]
    milestone_labels = [
        "RNA-only rollout\ninsufficient",
        "paired protein\ngeometry",
        "endpoint readout\ncalibration",
        "CITE-PCFM\nselected",
    ]
    for idx, (frac, label) in enumerate(zip(milestone_fracs, milestone_labels), start=1):
        pos = int(np.clip(round(frac * (len(pcfm_trials) - 1)), 0, len(pcfm_trials) - 1))
        x0 = float(pcfm_trials["local_trial"].iloc[pos])
        y0 = float(pcfm_trials["best_so_far"].iloc[pos])
        ax_curve.scatter([x0], [y0], s=48, color="white", edgecolor="#6D28D9", linewidth=1.2, zorder=6)
        ax_curve.text(x0 + 2.2, y0 * (1.18 if idx < 4 else 1.28), f"{idx}\n{label}", ha="left", va="bottom", fontsize=7.5, color="#312E81")
    ax_curve.set_yscale("log")
    ax_curve.set_xlabel("search iterations", fontsize=8.4)
    ax_curve.set_ylabel("protein_energy", fontsize=8.4)
    ax_curve.set_title("internal search potential / protein_energy", fontsize=10.4, weight="bold")
    ax_curve.text(0.02, 0.96, "high", transform=ax_curve.transAxes, ha="left", va="top", fontsize=7.3, color="#374151")
    ax_curve.text(0.02, 0.04, "low", transform=ax_curve.transAxes, ha="left", va="bottom", fontsize=7.3, color="#374151")
    ax_curve.grid(axis="y", color="#E5E7EB", lw=0.75)
    ax_curve.tick_params(labelsize=7.6)
    sns.despine(ax=ax_curve)

    fig.suptitle(
        "Algorithm discovery landscape and internal search potential",
        y=0.985,
        fontsize=16.0,
        weight="bold",
    )
    savefig(
        fig,
        "search_landscape11_cpanel_algorithm_discovery",
        "C-panel algorithm discovery landscape",
        "Can the algorithm-search result be shown in the C-panel style: terrain, candidate clouds, search path, PCFM star, and internal search potential?",
    )


def fig_search12_multibasin_algorithm_landscape(runs: pd.DataFrame, trials: pd.DataFrame, baselines: pd.DataFrame) -> None:
    """Multi-basin editorial landscape: several candidate basins plus real search-potential inset."""

    df = build_family_waddington_features(runs, trials, baselines)
    coord_map = {
        "VGFM": (0.17, 0.78),
        "BFM": (0.42, 0.64),
        "RSB-FM": (0.68, 0.48),
        "FFM": (0.83, 0.56),
        "PCFM": (0.82, 0.20),
    }
    color_map = {
        "VGFM": "#2C5AA0",
        "BFM": "#23825F",
        "FFM": "#A1A1AA",
        "RSB-FM": "#E5792A",
        "PCFM": "#6D28D9",
    }
    pcfm_energy = float(df.loc[df["family_short"].eq("PCFM"), "evidence_protein_energy"].iloc[0])
    fallback_blocked_energy = 0.055
    df["plot_energy"] = df["evidence_protein_energy"].fillna(fallback_blocked_energy)
    df["penalty"] = np.log2(df["plot_energy"] / pcfm_energy)
    df.loc[df["family_short"].eq("PCFM"), "penalty"] = 0.0
    df["x"] = df["family_short"].map(lambda k: coord_map[k][0])
    df["y"] = df["family_short"].map(lambda k: coord_map[k][1])
    df["status"] = np.where(df["family_short"].eq("PCFM"), "selected", np.where(df["family_short"].eq("FFM"), "blocked", "evaluated"))
    df.to_csv(TABLE_DIR / "multibasin_algorithm_landscape_coordinates.csv", index=False)

    gx = np.linspace(0.02, 0.98, 560)
    gy = np.linspace(0.06, 0.91, 430)
    xx, yy = np.meshgrid(gx, gy)
    terrain = 1.08 - 0.34 * xx + 0.12 * yy
    basin_params = {
        "VGFM": (0.30, 0.075, 0.082),
        "BFM": (0.38, 0.082, 0.082),
        "RSB-FM": (0.43, 0.080, 0.090),
        "FFM": (0.18, 0.070, 0.080),
        "PCFM": (0.62, 0.105, 0.105),
    }
    for _, row in df.iterrows():
        depth, sx, sy = basin_params[row["family_short"]]
        if row["family_short"] != "PCFM" and np.isfinite(row["penalty"]):
            depth *= 0.90 / (1.0 + 0.12 * row["penalty"])
        gauss = np.exp(-(((xx - row["x"]) ** 2) / (2 * sx**2) + ((yy - row["y"]) ** 2) / (2 * sy**2)))
        terrain -= depth * gauss

    ridge_terms = [
        (0.32, 0.56, 0.11, 0.22, 0.13),
        (0.55, 0.56, 0.10, 0.20, 0.12),
        (0.79, 0.42, 0.11, 0.16, 0.14),
        (0.88, 0.58, 0.075, 0.18, 0.10),
    ]
    for cx, cy, sx, sy, amp in ridge_terms:
        terrain += amp * np.exp(-(((xx - cx) ** 2) / (2 * sx**2) + ((yy - cy) ** 2) / (2 * sy**2)))
    terrain += 0.018 * np.sin(15 * xx + 2.5 * yy) + 0.012 * np.sin(27 * yy)
    terrain = np.clip(terrain, 0.28, 1.30)

    rng = np.random.default_rng(20260602)
    fig = plt.figure(figsize=(16.8, 6.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.58, 0.80], wspace=0.12)
    ax = fig.add_subplot(gs[0, 0])
    ax_curve = fig.add_subplot(gs[0, 1])

    levels = np.linspace(float(terrain.min()), float(terrain.max()), 46)
    ax.contourf(xx, yy, terrain, levels=levels, cmap="Greys_r", alpha=0.13, antialiased=True)
    ax.contour(xx, yy, terrain, levels=levels, colors="#94A3B8", linewidths=0.42, alpha=0.24)
    ax.contour(xx, yy, terrain, levels=levels[::4], colors="#475569", linewidths=0.62, alpha=0.17)

    cloud_rows = []
    for _, row in df.iterrows():
        family = row["family_short"]
        n = 250 if family != "FFM" else 165
        sx = 0.028 if family != "PCFM" else 0.022
        sy = 0.032 if family != "PCFM" else 0.026
        px = rng.normal(float(row["x"]), sx, n)
        py = rng.normal(float(row["y"]), sy, n)
        px = np.clip(px, 0.04, 0.96)
        py = np.clip(py, 0.08, 0.88)
        ax.scatter(
            px,
            py,
            s=9 if family != "FFM" else 7,
            color=color_map[family],
            alpha=0.42 if family != "FFM" else 0.23,
            edgecolor="none",
            zorder=6 if family != "FFM" else 4,
        )
        cloud_rows.extend({"family_short": family, "x": float(a), "y": float(b)} for a, b in zip(px, py))
    pd.DataFrame(cloud_rows).to_csv(TABLE_DIR / "multibasin_algorithm_landscape_cloud_points.csv", index=False)

    path_order = ["VGFM", "BFM", "RSB-FM", "PCFM"]
    pts = df.set_index("family_short").loc[path_order, ["x", "y"]].to_numpy(dtype=float)
    path = []
    for i in range(len(pts) - 1):
        p0, p1 = pts[i], pts[i + 1]
        vec = p1 - p0
        normal = np.array([-vec[1], vec[0]], dtype=float)
        normal /= max(np.linalg.norm(normal), 1e-8)
        for t in np.linspace(0, 1, 58, endpoint=False):
            s = t * t * (3 - 2 * t)
            bend = 0.035 * np.sin(np.pi * t) * normal
            if i == 2:
                bend += np.array([0.020 * np.sin(np.pi * t), -0.030 * np.sin(np.pi * t)])
            path.append((1 - s) * p0 + s * p1 + bend)
    path.append(pts[-1])
    path = np.asarray(path)
    ax.plot(path[:, 0], path[:, 1], color="white", lw=5.1, alpha=0.96, zorder=8)
    ax.plot(path[:, 0], path[:, 1], color="#111827", lw=2.3, ls=(0, (2.3, 2.3)), alpha=0.96, zorder=9)
    ax.scatter(path[::14, 0], path[::14, 1], s=34, color="#111827", edgecolor="white", linewidth=0.7, zorder=10)

    transition_labels = {
        ("VGFM", "BFM"): "1",
        ("BFM", "RSB-FM"): "2",
        ("RSB-FM", "PCFM"): "3",
    }
    for i, (a, b) in enumerate(zip(path_order[:-1], path_order[1:])):
        p0 = df[df["family_short"].eq(a)].iloc[0]
        p1 = df[df["family_short"].eq(b)].iloc[0]
        mx, my = (float(p0["x"]) + float(p1["x"])) / 2, (float(p0["y"]) + float(p1["y"])) / 2
        dy = 0.048 if i < 2 else -0.020
        ax.text(
            mx,
            my + dy,
            transition_labels[(a, b)],
            ha="center",
            va="center",
            fontsize=8.0,
            color="#0F172A",
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.25,rounding_size=0.14", fc="white", ec="#CBD5E1", alpha=0.90),
            zorder=12,
        )

    for _, row in df.iterrows():
        family = row["family_short"]
        if family == "PCFM":
            ax.scatter(row["x"], row["y"], marker="*", s=760, color=color_map[family], edgecolor="white", linewidth=1.8, zorder=15)
            ax.scatter(row["x"], row["y"], marker="*", s=1040, facecolors="none", edgecolors="#C4B5FD", linewidth=1.5, zorder=14)
            ax.text(row["x"] + 0.035, row["y"] + 0.005, "CITE-PCFM\nselected", ha="left", va="center", fontsize=10.4, color="#312E81", weight="bold")
        elif family == "FFM":
            ax.text(row["x"] + 0.030, row["y"] + 0.002, "FFM\nblocked", ha="left", va="center", fontsize=8.8, color="#6B7280")
        else:
            ax.text(row["x"], row["y"] + 0.078, family, ha="center", va="bottom", fontsize=10.2, color=color_map[family], weight="bold")

    ax.annotate("", xy=(0.965, 0.085), xytext=(0.055, 0.085), arrowprops=dict(arrowstyle="->", lw=1.0, color="#374151"))
    ax.annotate("", xy=(0.060, 0.890), xytext=(0.060, 0.085), arrowprops=dict(arrowstyle="->", lw=1.0, color="#374151"))
    ax.text(0.50, 0.040, "design space", ha="center", va="center", fontsize=8.8, color="#374151")
    ax.text(0.022, 0.46, "search potential\n(lower is better)", ha="center", va="center", fontsize=8.8, color="#374151", rotation=90)
    ax.set_xlim(0.02, 1.0)
    ax.set_ylim(0.04, 0.92)
    ax.set_axis_off()
    ax.set_title("Multi-basin algorithm discovery landscape", fontsize=13.8, weight="bold", loc="left", pad=8)

    pcfm_trials = trials[trials["algorithm"].eq("cite_paired_coord_fm") & trials["protein_energy"].notna()].sort_values("created_at").copy()
    pcfm_trials["local_trial"] = np.arange(1, len(pcfm_trials) + 1)
    pcfm_trials["best_so_far"] = np.minimum.accumulate(pcfm_trials["protein_energy"].to_numpy(dtype=float))
    pcfm_trials[["local_trial", "stage", "decision", "protein_energy", "best_so_far"]].to_csv(TABLE_DIR / "multibasin_internal_search_potential_curve.csv", index=False)
    ax_curve.step(pcfm_trials["local_trial"], pcfm_trials["best_so_far"], where="post", color="#6D28D9", lw=2.2)
    ax_curve.scatter(
        pcfm_trials["local_trial"],
        pcfm_trials["protein_energy"],
        s=13,
        color="#C4B5FD",
        alpha=0.44,
        edgecolor="none",
    )
    milestone_fracs = [0.08, 0.28, 0.55, 0.84]
    milestone_labels = [
        "RNA-only rollout\ninsufficient",
        "paired protein\ngeometry",
        "endpoint readout\ncalibration",
        "CITE-PCFM\nselected",
    ]
    for idx, (frac, label) in enumerate(zip(milestone_fracs, milestone_labels), start=1):
        pos = int(np.clip(round(frac * (len(pcfm_trials) - 1)), 0, len(pcfm_trials) - 1))
        x0 = float(pcfm_trials["local_trial"].iloc[pos])
        y0 = float(pcfm_trials["best_so_far"].iloc[pos])
        ax_curve.scatter([x0], [y0], s=48, color="white", edgecolor="#6D28D9", linewidth=1.2, zorder=6)
        ax_curve.text(x0 + 2.0, y0 * (1.18 if idx < 4 else 1.28), f"{idx}\n{label}", ha="left", va="bottom", fontsize=7.5, color="#312E81")
    ax_curve.set_yscale("log")
    ax_curve.set_xlabel("search iterations", fontsize=8.4)
    ax_curve.set_ylabel("protein_energy", fontsize=8.4)
    ax_curve.set_title("internal search potential / protein_energy", fontsize=10.4, weight="bold")
    ax_curve.text(0.02, 0.96, "high", transform=ax_curve.transAxes, ha="left", va="top", fontsize=7.3, color="#374151")
    ax_curve.text(0.02, 0.04, "low", transform=ax_curve.transAxes, ha="left", va="bottom", fontsize=7.3, color="#374151")
    ax_curve.grid(axis="y", color="#E5E7EB", lw=0.75)
    ax_curve.tick_params(labelsize=7.6)
    sns.despine(ax=ax_curve)

    fig.suptitle(
        "Algorithm discovery landscape and internal search potential",
        y=0.985,
        fontsize=16.0,
        weight="bold",
    )
    savefig(
        fig,
        "search_landscape12_multibasin_algorithm_landscape",
        "Multi-basin algorithm discovery landscape",
        "Can the C panel show several candidate basins and ridge-crossing transitions rather than a one-dimensional downhill path?",
    )


def fig_search13_multibasin_algorithm_landscape_3d(runs: pd.DataFrame, trials: pd.DataFrame, baselines: pd.DataFrame) -> None:
    """3D multi-basin landscape with candidate basins and ridge-crossing search path."""

    df = build_family_waddington_features(runs, trials, baselines)
    coord_map = {
        "VGFM": (0.17, 0.68),
        "BFM": (0.42, 0.72),
        "RSB-FM": (0.68, 0.60),
        "FFM": (0.83, 0.56),
        "PCFM": (0.82, 0.20),
    }
    color_map = {
        "VGFM": "#2C5AA0",
        "BFM": "#23825F",
        "FFM": "#A1A1AA",
        "RSB-FM": "#E5792A",
        "PCFM": "#6D28D9",
    }
    pcfm_energy = float(df.loc[df["family_short"].eq("PCFM"), "evidence_protein_energy"].iloc[0])
    fallback_blocked_energy = 0.055
    df["plot_energy"] = df["evidence_protein_energy"].fillna(fallback_blocked_energy)
    df["penalty"] = np.log2(df["plot_energy"] / pcfm_energy)
    df.loc[df["family_short"].eq("PCFM"), "penalty"] = 0.0
    df["x"] = df["family_short"].map(lambda k: coord_map[k][0])
    df["y"] = df["family_short"].map(lambda k: coord_map[k][1])
    df["status"] = np.where(df["family_short"].eq("PCFM"), "selected", np.where(df["family_short"].eq("FFM"), "blocked", "evaluated"))
    df.to_csv(TABLE_DIR / "multibasin_3d_algorithm_landscape_coordinates.csv", index=False)
    # FFM is omitted from the main visual path because it was provenance/evaluator blocked,
    # not a trusted biological/modeling failure.
    visible_order = ["VGFM", "BFM", "RSB-FM", "PCFM"]
    df_visible = df[df["family_short"].isin(visible_order)].copy()

    gx = np.linspace(0.02, 0.98, 300)
    gy = np.linspace(0.06, 0.91, 250)
    xx, yy = np.meshgrid(gx, gy)
    terrain = 1.15 - 0.32 * xx + 0.10 * yy
    basin_params = {
        "VGFM": (0.35, 0.075, 0.082),
        "BFM": (0.43, 0.082, 0.082),
        "RSB-FM": (0.47, 0.080, 0.090),
        "PCFM": (0.72, 0.110, 0.110),
    }
    for _, row in df_visible.iterrows():
        depth, sx, sy = basin_params[row["family_short"]]
        if row["family_short"] != "PCFM" and np.isfinite(row["penalty"]):
            depth *= 0.92 / (1.0 + 0.12 * row["penalty"])
        terrain -= depth * np.exp(-(((xx - row["x"]) ** 2) / (2 * sx**2) + ((yy - row["y"]) ** 2) / (2 * sy**2)))

    ridge_terms = [
        (0.31, 0.56, 0.11, 0.23, 0.16),
        (0.55, 0.57, 0.11, 0.22, 0.15),
        (0.78, 0.43, 0.12, 0.16, 0.18),
        (0.89, 0.58, 0.075, 0.18, 0.12),
    ]
    for cx, cy, sx, sy, amp in ridge_terms:
        terrain += amp * np.exp(-(((xx - cx) ** 2) / (2 * sx**2) + ((yy - cy) ** 2) / (2 * sy**2)))
    terrain += 0.018 * np.sin(15 * xx + 2.5 * yy) + 0.012 * np.sin(27 * yy)
    terrain = np.clip(terrain, 0.22, 1.36)

    def interp_z(x: float, y: float) -> float:
        ix = int(np.clip(np.searchsorted(gx, x) - 1, 0, len(gx) - 2))
        iy = int(np.clip(np.searchsorted(gy, y) - 1, 0, len(gy) - 2))
        x0, x1 = gx[ix], gx[ix + 1]
        y0, y1 = gy[iy], gy[iy + 1]
        tx = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
        ty = 0.0 if y1 == y0 else (y - y0) / (y1 - y0)
        z00 = terrain[iy, ix]
        z10 = terrain[iy, ix + 1]
        z01 = terrain[iy + 1, ix]
        z11 = terrain[iy + 1, ix + 1]
        return float((1 - tx) * (1 - ty) * z00 + tx * (1 - ty) * z10 + (1 - tx) * ty * z01 + tx * ty * z11)

    path_order = ["VGFM", "BFM", "RSB-FM", "PCFM"]
    pts = df.set_index("family_short").loc[path_order, ["x", "y"]].to_numpy(dtype=float)
    path = []
    for i in range(len(pts) - 1):
        p0, p1 = pts[i], pts[i + 1]
        vec = p1 - p0
        normal = np.array([-vec[1], vec[0]], dtype=float)
        normal /= max(np.linalg.norm(normal), 1e-8)
        for t in np.linspace(0, 1, 78, endpoint=False):
            s = t * t * (3 - 2 * t)
            bend = 0.034 * np.sin(np.pi * t) * normal
            if i == 2:
                bend += np.array([0.024 * np.sin(np.pi * t), -0.034 * np.sin(np.pi * t)])
            path.append((1 - s) * p0 + s * p1 + bend)
    path.append(pts[-1])
    path = np.asarray(path)
    path_z = np.asarray([interp_z(float(x), float(y)) for x, y in path]) + 0.035

    terrain_cmap = LinearSegmentedColormap.from_list(
        "multibasin_grey_3d",
        ["#F8FAFC", "#E2E8F0", "#CBD5E1", "#94A3B8", "#475569"],
    )
    rng = np.random.default_rng(20260602)
    fig = plt.figure(figsize=(17.2, 7.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.62, 0.78], wspace=0.08)
    try:
        ax3d = fig.add_subplot(gs[0, 0], projection="3d", computed_zorder=False)
    except TypeError:
        ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    ax_curve = fig.add_subplot(gs[0, 1])

    z_floor = float(terrain.min()) - 0.18
    surf = ax3d.plot_surface(
        xx,
        yy,
        terrain,
        cmap=terrain_cmap,
        linewidth=0,
        antialiased=True,
        alpha=0.76,
        rcount=150,
        ccount=180,
        shade=True,
        zorder=0,
    )
    if hasattr(surf, "set_sort_zpos"):
        surf.set_sort_zpos(z_floor - 2.0)
    ax3d.contour(xx, yy, terrain, zdir="z", offset=z_floor, levels=18, cmap=terrain_cmap, linewidths=0.78, alpha=0.72, zorder=1)

    cloud_rows = []
    for _, row in df_visible.iterrows():
        family = row["family_short"]
        n = 170
        sx = 0.026 if family != "PCFM" else 0.020
        sy = 0.030 if family != "PCFM" else 0.024
        px = np.clip(rng.normal(float(row["x"]), sx, n), 0.04, 0.96)
        py = np.clip(rng.normal(float(row["y"]), sy, n), 0.08, 0.88)
        pz = np.asarray([interp_z(float(a), float(b)) for a, b in zip(px, py)]) + rng.normal(0.030, 0.006, n)
        sc = ax3d.scatter(
            px,
            py,
            pz,
            s=15,
            color=color_map[family],
            alpha=0.86,
            edgecolor="none",
            depthshade=False,
            zorder=30,
        )
        if hasattr(sc, "set_sort_zpos"):
            sc.set_sort_zpos(3.0)
        cloud_rows.extend({"family_short": family, "x": float(a), "y": float(b), "z": float(c)} for a, b, c in zip(px, py, pz))
    pd.DataFrame(cloud_rows).to_csv(TABLE_DIR / "multibasin_3d_algorithm_landscape_cloud_points.csv", index=False)

    path_z = path_z + 0.055
    ax3d.plot(path[:, 0], path[:, 1], path_z, color="white", lw=6.2, alpha=0.98, zorder=12)
    ax3d.plot(path[:, 0], path[:, 1], path_z + 0.010, color="#111827", lw=2.8, ls=(0, (2.2, 2.2)), alpha=0.98, zorder=13)
    ax3d.scatter(path[::17, 0], path[::17, 1], path_z[::17] + 0.018, s=42, color="#111827", edgecolor="white", linewidth=0.75, depthshade=False, zorder=14)

    transition_labels = {
        ("VGFM", "BFM"): "1",
        ("BFM", "RSB-FM"): "2",
        ("RSB-FM", "PCFM"): "3",
    }
    for i, (a, b) in enumerate(zip(path_order[:-1], path_order[1:])):
        p0 = df[df["family_short"].eq(a)].iloc[0]
        p1 = df[df["family_short"].eq(b)].iloc[0]
        mx, my = (float(p0["x"]) + float(p1["x"])) / 2, (float(p0["y"]) + float(p1["y"])) / 2
        z = interp_z(mx, my) + (0.24 if i < 2 else 0.22)
        ax3d.text(
            mx,
            my,
            z,
            transition_labels[(a, b)],
            ha="center",
            va="center",
            fontsize=9.5,
            color="#0F172A",
            weight="bold",
            bbox=dict(boxstyle="circle,pad=0.28", fc="white", ec="#CBD5E1", alpha=0.94),
            zorder=15,
        )

    node_labels = {
        "VGFM": "CITE-VGFM",
        "BFM": "CITE-BFM",
        "RSB-FM": "CITE-RSB-FM",
        "PCFM": "CITE-PCFM\nselected",
    }
    for _, row in df_visible.iterrows():
        family = row["family_short"]
        z = interp_z(float(row["x"]), float(row["y"])) + 0.060
        if family == "PCFM":
            ax3d.scatter(row["x"], row["y"], z, marker="*", s=760, color=color_map[family], edgecolor="white", linewidth=1.9, depthshade=False, zorder=45)
            ax3d.scatter(row["x"], row["y"], z + 0.004, marker="*", s=1040, facecolors="none", edgecolors="#C4B5FD", linewidth=1.5, depthshade=False, zorder=44)
            label = ax3d.text(
                row["x"] + 0.030,
                row["y"] - 0.015,
                z + 0.090,
                node_labels[family],
                ha="left",
                va="center",
                fontsize=9.0,
                color="#312E81",
                weight="bold",
                bbox=dict(boxstyle="round,pad=0.18,rounding_size=0.08", fc="white", ec="none", alpha=0.78),
            )
        else:
            label = ax3d.text(
                row["x"],
                row["y"],
                z + 0.095,
                node_labels[family],
                ha="center",
                va="bottom",
                fontsize=8.6,
                color=color_map[family],
                weight="bold",
                bbox=dict(boxstyle="round,pad=0.16,rounding_size=0.08", fc="white", ec="none", alpha=0.76),
            )
        label.set_path_effects([pe.withStroke(linewidth=3.2, foreground="white", alpha=0.92)])

    ax3d.set_xlim(0.03, 0.96)
    ax3d.set_ylim(0.08, 0.88)
    ax3d.set_zlim(z_floor, 1.62)
    ax3d.view_init(elev=38, azim=-59)
    try:
        ax3d.set_box_aspect((1.35, 0.95, 0.50))
    except Exception:
        pass
    ax3d.set_xlabel("protein handling tested", labelpad=10, fontsize=8.7)
    ax3d.set_ylabel("real-data search result", labelpad=10, fontsize=8.7)
    ax3d.set_zlabel("", labelpad=4, fontsize=8.4)
    ax3d.set_xticks([0.17, 0.42, 0.68, 0.82])
    ax3d.set_xticklabels(
        [
            "RNA+protein+\ncell number",
            "RNA+protein\nbalanced",
            "protein-\ncorrection path",
            "protein series\nfrom RNA",
        ],
        fontsize=6.7,
    )
    ax3d.set_yticks([0.20, 0.48, 0.64, 0.78])
    ax3d.set_yticklabels(
        [
            "CITE-PCFM\nselected",
            "CITE-RSB-FM\nStage 2 failed",
            "CITE-BFM\nStage 2 failed",
            "CITE-VGFM\nStage 1 failed",
        ],
        fontsize=6.7,
    )
    ax3d.set_zticks([0.32, 0.80, 1.28])
    ax3d.set_zticklabels(["low", "mid", "high"], fontsize=6.9)
    ax3d.xaxis.pane.set_facecolor((1, 1, 1, 0.0))
    ax3d.yaxis.pane.set_facecolor((1, 1, 1, 0.0))
    ax3d.zaxis.pane.set_facecolor((1, 1, 1, 0.0))
    ax3d.grid(True, color="#E5E7EB")
    ax3d.text2D(
        0.02,
        0.02,
        "Higher = generated protein profiles are worse; lower = closer to real CITE-seq protein profiles.\n"
        "Path labels: 1 stop modeling cell-number change; 2 test a protein-correction path; 3 generate protein time series from RNA.",
        transform=ax3d.transAxes,
        fontsize=7.1,
        color="#334155",
        bbox=dict(boxstyle="round,pad=0.32,rounding_size=0.12", fc="white", ec="#CBD5E1", alpha=0.90),
    )
    ax3d.text2D(
        0.02,
        0.90,
        "Routes tested by CellCompass\n"
        "CITE-VGFM: Stage 1 failed; RNA, protein and cell-number fit was unstable\n"
        "CITE-BFM: Stage 2 failed; future protein profiles were still wrong\n"
        "CITE-RSB-FM: Stage 2 failed; generated proteins moved away from real CITE profiles\n"
        "CITE-PCFM: selected; RNA generated a protein time series",
        transform=ax3d.transAxes,
        fontsize=7.0,
        color="#1F2937",
        va="top",
        bbox=dict(boxstyle="round,pad=0.38,rounding_size=0.12", fc="white", ec="#CBD5E1", alpha=0.90),
    )
    ax3d.set_title("Tested routes from CITE-seq time course to protein dynamics", fontsize=13.2, weight="bold", pad=10)

    pcfm_trials = trials[trials["algorithm"].eq("cite_paired_coord_fm") & trials["protein_energy"].notna()].sort_values("created_at").copy()
    # Use one comparable claim-stage axis. The Stage-1 feasibility point was run on a
    # debug dataset; mixing it into best-so-far makes the first Stage-2 drop unreadable.
    pcfm_trials = pcfm_trials[~pcfm_trials["stage"].eq("stage1_feasibility")].copy()
    pcfm_trials["local_trial"] = np.arange(1, len(pcfm_trials) + 1)
    pcfm_trials["best_so_far"] = np.minimum.accumulate(pcfm_trials["protein_energy"].to_numpy(dtype=float))
    pcfm_trials[["local_trial", "stage", "decision", "protein_energy", "best_so_far"]].to_csv(TABLE_DIR / "multibasin_3d_internal_search_potential_curve.csv", index=False)

    repair_windows = [
        (1.0, 2.0, 2, "#EDE9FE", "fix protein target and loss", "0.535 -> 0.287"),
        (2.0, 4.0, 4, "#F5F3FF", "larger protein-prediction network", "0.287 -> 0.071"),
        (4.0, 19.0, 19, "#EEF2FF", "match final-day protein profile", "0.071 -> 0.0247"),
        (25.0, 31.0, 31, "#F3E8FF", "fix time encoding for protein prediction", "0.0247 -> 0.0236"),
        (37.0, 38.0, 38, "#FCE7F3", "give days 3, 4 and 7 more weight", "0.0236 -> 0.0228"),
    ]
    for x0, x1, _, color, _, _ in repair_windows:
        ax_curve.axvspan(x0, x1, facecolor=color, edgecolor="none", linewidth=0, alpha=0.62, zorder=0)

    ax_curve.scatter(
        pcfm_trials["local_trial"],
        pcfm_trials["protein_energy"],
        s=13,
        color="#C4B5FD",
        alpha=0.44,
        edgecolor="none",
        label="all trials",
        zorder=2,
    )
    ax_curve.plot(
        pcfm_trials["local_trial"],
        pcfm_trials["protein_energy"],
        color="#C4B5FD",
        lw=0.9,
        alpha=0.38,
        zorder=1,
    )
    ax_curve.step(
        pcfm_trials["local_trial"],
        pcfm_trials["best_so_far"],
        where="post",
        color="#6D28D9",
        lw=2.6,
        label="best kept",
        zorder=4,
    )
    gate = 0.022906601428985596
    ax_curve.axhline(gate, color="#111827", lw=1.1, ls="--", zorder=1)
    ax_curve.text(
        69.5,
        gate * 1.10,
        "claim threshold",
        ha="right",
        va="bottom",
        fontsize=6.8,
        color="#111827",
    )

    for idx, (_, _, trial_no, _, _, _) in enumerate(repair_windows, start=1):
        pos = int(np.argmin(np.abs(pcfm_trials["local_trial"].to_numpy(dtype=float) - trial_no)))
        x0 = float(pcfm_trials["local_trial"].iloc[pos])
        y0 = float(pcfm_trials["best_so_far"].iloc[pos])
        ax_curve.scatter([x0], [y0], s=56, color="white", edgecolor="#6D28D9", linewidth=1.25, zorder=6)
        ax_curve.text(
            x0,
            y0 * 1.23,
            str(idx),
            ha="center",
            va="center",
            fontsize=7.2,
            color="#312E81",
            weight="bold",
            bbox=dict(boxstyle="circle,pad=0.18", fc="white", ec="#6D28D9", alpha=0.95),
            zorder=7,
        )
    repair_text = "\n".join(
        ["start: protein error 0.535"]
        + [f"{i}. {name}: {delta}" for i, (_, _, _, _, name, delta) in enumerate(repair_windows, start=1)]
    )
    ax_curve.text(
        0.57,
        0.96,
        "Shaded tuning periods\n" + repair_text,
        transform=ax_curve.transAxes,
        ha="left",
        va="top",
        fontsize=6.25,
        color="#312E81",
        linespacing=1.08,
        bbox=dict(boxstyle="round,pad=0.36,rounding_size=0.12", fc="white", ec="#DDD6FE", alpha=0.93),
        zorder=8,
    )
    ax_curve.set_yscale("log")
    ax_curve.set_ylim(0.018, 0.75)
    ax_curve.set_xlabel("CITE-PCFM real-data trial order", fontsize=8.4)
    ax_curve.set_ylabel("surface-protein distribution error\nlower is better", fontsize=8.4)
    ax_curve.set_title("CITE-PCFM trial record: shaded periods mark actual model changes", fontsize=9.6, weight="bold")
    ax_curve.grid(axis="y", color="#E5E7EB", lw=0.75)
    ax_curve.tick_params(labelsize=7.6)
    sns.despine(ax=ax_curve)

    fig.suptitle(
        "CellCompass search produced an RNA-to-protein CITE-seq dynamics model",
        y=0.985,
        fontsize=16.0,
        weight="bold",
    )
    savefig(
        fig,
        "search_landscape13_multibasin_algorithm_landscape_3d",
        "3D multi-basin algorithm discovery landscape",
        "A 3D multi-basin landscape rendering of the CITE algorithm search, paired with the real PCFM internal search-potential curve.",
    )


def write_notes(runs: pd.DataFrame, baselines: pd.DataFrame) -> None:
    final = runs[runs["is_final_lock"]].tail(1)
    same = runs[runs["dataset_id"].eq(FINAL_DATASET) & runs["protein_energy"].notna()].copy()
    promoted = same[same["is_promoted"]].copy()
    raw_best = same.loc[same["protein_energy"].idxmin()]
    promoted_best = promoted.loc[promoted["protein_energy"].idxmin()]
    final_metric = float(final["protein_energy"].iloc[0]) if len(final) else float("nan")

    lines = []
    lines.append("# Quantitative Algorithm Search Landscape\n\n")
    lines.append("Source: raw `~/.cellcompass` training algorithm registries, campaign trials, run metrics, and final-panel builtin baseline artifacts.\n\n")
    lines.append("## What changed\n")
    lines.append("- Added explicit five-algorithm lineage map so the evolution is visible before the reader sees embeddings or metric curves.\n")
    lines.append("- Added same-panel metric curve with promoted-best trace, raw rejected minimum, final locked run, and builtin reference levels.\n")
    lines.append("- Added CITE-PCFM design embedding from metrics plus config knobs, so the search is visible as a path in design space.\n")
    lines.append("- Added global CITE-family embedding to show where final PCFM sits relative to earlier CITE algorithm families.\n")
    lines.append("- Added a five-family Waddington-style embedding where each algorithm generation is positioned by metrics plus failure-mode features, not by prose.\n")
    lines.append("- Added an editorial 2.5D terrain with conceptual axes, quantitative height, a visible FFM registry gate, metric descent, and a design-genome strip.\n")
    lines.append("- Added a cleaner NBT-style model-evidence basin that removes the registry-blocked FFM node from the main visual story.\n\n")
    lines.append("- Added a C-panel-style discovery landscape with grey terrain, candidate algorithm clouds, a PCFM selection star, and an internal search-potential inset.\n\n")
    lines.append("- Added a multi-basin C-panel variant where candidate families occupy distinct basins separated by ridges, avoiding a one-dimensional downhill path.\n\n")
    lines.append("- Added a 3D multi-basin C-panel variant with the same candidate basins and audited internal PCFM search-potential curve.\n\n")
    lines.append("## How to read the landscape\n")
    lines.append("- `search_landscape00` is the missing lifecycle panel: VGFM, BFM/balanced, Factorized FM, Residual SB-FM, and PCFM are shown as a five-generation algorithm lineage with one failed assumption and one retained lesson per generation.\n")
    lines.append("- `search_landscape01` is the lifecycle curve: each point is a real same-panel run, the red trace is the promoted best-so-far, the star is the final locked run, and the black X is the raw rejected minimum.\n")
    lines.append("- `search_landscape03` is the PCFM design basin: trials are embedded by metrics and configuration knobs, making the search path visible in a quantitative design space.\n")
    lines.append("- `search_landscape04` is the global algorithm-family map: final PCFM should appear as a different region from VGFM/BFM/residual/factorized attempts, not as an isolated lucky point.\n")
    lines.append("- `search_landscape05` is the mechanism-level waterfall: it explains which conceptual repairs changed `-log10(protein_energy)` and which changes mostly saturated.\n")
    lines.append("- `search_landscape06` is the parameter-response check: it shows why later tuning became local rather than a new mechanistic discovery.\n\n")
    lines.append("- `search_landscape07` is the requested algorithm Waddington landscape: the five generations are embedded by protein metrics, W1/TMV evidence, promotion/rejection rates, and failure-mode vectors such as TMV failure, provenance block, CRUFM miss, and final lock.\n")
    lines.append("- `search_landscape08` is the editor-facing terrain: x/y are interpretable design axes, z is evidence penalty from protein_energy plus lifecycle penalties, FFM is shown as a registry gate rather than a biological failure, and the bottom strip shows design mutations.\n")
    lines.append("- `search_landscape09` is the cleaner main-text candidate: it shows only the model-evidence path VGFM -> BFM -> RSB-FM -> PCFM, with FFM kept out of the visual story because it was registry-blocked rather than metric-evaluated.\n\n")
    lines.append("- `search_landscape11` follows the article-layout C-panel reference: the left panel is an editorial algorithm terrain and the right panel is the actual CITE-PCFM internal protein_energy best-so-far curve.\n\n")
    lines.append("- `search_landscape12` is the preferred C-panel refinement: several candidate basins, ridge-crossing transitions, FFM as a separate blocked basin, and the same audited internal search-potential inset.\n\n")
    lines.append("- `search_landscape13` is the 3D version of the refined C-panel: the same basins are rendered as a surface with candidate clouds and a path over the terrain.\n\n")
    lines.append("## Key quantitative facts\n")
    lines.append(f"- Final locked CITE-PCFM protein_energy: `{final_metric:.6f}`.\n")
    lines.append(f"- Best promoted same-panel record: `{float(promoted_best['protein_energy']):.6f}` from `{promoted_best['run_id']}`.\n")
    lines.append(f"- Raw same-panel minimum: `{float(raw_best['protein_energy']):.6f}` from `{raw_best['run_id']}`; this is shown separately because it was not the locked algorithm.\n")
    crufm = baselines[baselines["algorithm"].eq("crufm")]
    if len(crufm) and pd.notna(crufm["protein_energy"].iloc[0]):
        lines.append(f"- CRUFM same-panel reference protein_energy: `{float(crufm['protein_energy'].iloc[0]):.6f}`.\n")
    lines.append("\n## Figure catalog\n")
    for fig in FIGURES:
        lines.append(f"- `{fig['name']}`: {fig['question']}. File: `{fig['png']}`\n")
    SEARCH_NOTE_PATH.write_text("".join(lines))

    existing = STORY_PATH.read_text() if STORY_PATH.exists() else "# CITE-PCFM Deep Story Package\n"
    marker = "\n## 3. Quantitative Algorithm Search Landscape\n"
    block = marker + "\n"
    block += "This section was added after the lifecycle critique: it uses raw `~/.cellcompass` registries to make the algorithm evolution quantitative rather than only schematic.\n\n"
    block += f"- Final locked CITE-PCFM protein_energy: `{final_metric:.6f}`.\n"
    block += f"- Best promoted same-panel record: `{float(promoted_best['protein_energy']):.6f}`.\n"
    block += f"- Raw same-panel minimum, shown separately from the locked algorithm: `{float(raw_best['protein_energy']):.6f}`.\n"
    if len(crufm) and pd.notna(crufm["protein_energy"].iloc[0]):
        block += f"- CRUFM same-panel reference protein_energy: `{float(crufm['protein_energy'].iloc[0]):.6f}`.\n"
    block += "- Five-generation lineage: `cite_vgfm -> cite_bfm/balanced -> cite_factorized_fm -> cite_residual_sb_fm -> cite_paired_coord_fm`.\n"
    block += "- Main panels to use: `search_landscape13_multibasin_algorithm_landscape_3d`, `search_landscape12_multibasin_algorithm_landscape`, `search_landscape11_cpanel_algorithm_discovery`, `search_landscape09_nbt_clean_algorithm_basin`, `search_landscape08_editorial_2p5d_algorithm_terrain`, `search_landscape01_same_panel_metric_curve`, and `search_landscape05_mechanism_gain_waterfall`.\n"
    block += "- `search_landscape07_family_waddington_embedding` is the clearest quantitative answer to the lifecycle critique: each family is a basin in a PCA landscape built from metric, W1/TMV, promotion/rejection, failure-mode, and design-repair features.\n"
    block += "- `search_landscape08_editorial_2p5d_algorithm_terrain` is the strongest NBT-style visual candidate: conceptual axes explain the design evolution, while height/metric panels keep the figure quantitative.\n"
    block += "- `search_landscape09_nbt_clean_algorithm_basin` is the cleaner main-text candidate: it removes FFM from the visual path and keeps the focus on metric-evaluated model search.\n"
    block += "- `search_landscape11_cpanel_algorithm_discovery` matches the planned A4 C-panel language: grey discovery terrain, colored candidate clouds, PCFM star, and a real internal search-potential inset.\n"
    block += "- `search_landscape12_multibasin_algorithm_landscape` is the refined C-panel candidate: it removes the confusing random-initialization cue and shows several local basins separated by ridges.\n"
    block += "- `search_landscape13_multibasin_algorithm_landscape_3d` is the 3D rendering of the refined basin view, useful if the main figure needs more visual depth.\n"
    block += "- Editorial logic: the metric rose when the search left endpoint-only repair and entered a time-local protein supervision basin; later raw minima were not automatically accepted because the lifecycle locked reproducible evidence, not the single lowest scalar.\n"
    block += "- Suggested one-glance layout: left, same-panel metric curve; center, PCFM design embedding; right, mechanism gain waterfall. Put the global algorithm embedding in supplement unless the article needs a broader CellCompass discovery panel.\n"
    block += f"- Full note: `{SEARCH_NOTE_PATH}`.\n"
    if marker in existing:
        prefix, rest = existing.split(marker, 1)
        next_section = rest.find("\n## ")
        tail = rest[next_section:] if next_section >= 0 else ""
        existing = prefix.rstrip() + "\n" + block.rstrip() + "\n\n" + tail.lstrip("\n")
    else:
        existing = existing.rstrip() + "\n" + block
    STORY_PATH.write_text(existing)


def write_manifest() -> None:
    MANIFEST_PATH.write_text(json.dumps({"figures": FIGURES, "note": str(SEARCH_NOTE_PATH), "storyboard": str(STORY_PATH)}, indent=2))


def main() -> None:
    ensure_dirs()
    runs = load_runs()
    trials = load_trials()
    baselines = load_builtin_baselines(FINAL_DATASET)
    runs.to_csv(TABLE_DIR / "cellcompass_cite_algorithm_runs.csv", index=False)
    trials.to_csv(TABLE_DIR / "cellcompass_cite_algorithm_trials.csv", index=False)
    baselines.to_csv(TABLE_DIR / "cellcompass_final_panel_builtin_baselines.csv", index=False)

    fig_search00_five_algorithm_lineage(baselines)
    fig_search01_final_panel_curve(runs, baselines)
    fig_search02_family_records(runs)
    fig_search03_pcfm_embedding(runs)
    fig_search04_global_embedding(runs)
    fig_search05_mechanism_waterfall()
    fig_search06_parameter_response(runs)
    fig_search07_family_waddington_embedding(runs, trials, baselines)
    fig_search08_editorial_terrain(runs, trials, baselines)
    fig_search09_nbt_clean_basin(runs, trials, baselines)
    fig_search10_editorial_3d_basin(runs, trials, baselines)
    fig_search11_cpanel_algorithm_discovery(runs, trials, baselines)
    fig_search12_multibasin_algorithm_landscape(runs, trials, baselines)
    fig_search13_multibasin_algorithm_landscape_3d(runs, trials, baselines)
    write_notes(runs, baselines)
    write_manifest()
    print(json.dumps({"figures": len(FIGURES), "out_root": str(OUT_ROOT), "note": str(SEARCH_NOTE_PATH)}, indent=2))


if __name__ == "__main__":
    main()
