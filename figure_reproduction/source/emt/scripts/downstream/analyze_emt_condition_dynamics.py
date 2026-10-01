#!/usr/bin/env python3
"""Downstream EMT dynamics analysis from condition-specific CytoBridge VGFM rollouts.

This script does not train a substitute dynamics model.  It consumes the locked
CytoBridge evaluation_trajectory.npz artifacts produced by the trained VGFM runs,
projects those model-native latent rollouts onto observed gene-program scores via
a post-hoc latent-to-program readout fit on observed cells, and compares generated
routes/endpoints against same-condition observed time courses.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUTPUT_DIR = Path(globals().get("output_dir", "/Users/zhenyizhang/Documents/GitHub/CytoBridge-agent-runtime-v2-20260309/emt"))
PREP_DIR = OUTPUT_DIR / "preprocessed"
COND_DIR = PREP_DIR / "conditions"
TRAIN_ROOT = OUTPUT_DIR / "training_runs"
DOWN_DIR = OUTPUT_DIR / "downstream" / "emt_condition_dynamics"
FIG_DIR = OUTPUT_DIR / "figures" / "emt_condition_dynamics"
DOWN_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

TIME_LABELS = ["0d", "8h", "1d", "3d", "7d"]
TIME_INDEX_TO_LABEL = {0: "0d", 1: "8h", 2: "1d", 3: "3d", 4: "7d"}
TIME_LABEL_TO_INDEX = {v: k for k, v in TIME_INDEX_TO_LABEL.items()}
CORE_PROGRAMS = ["epithelial_score", "hybrid_score", "mesenchymal_score"]
PROGRAMS = [
    "epithelial_score", "hybrid_score", "mesenchymal_score", "emt_core_score",
    "early_immediate_response_score", "early_inflammatory_tnf_score",
    "intermediate_partial_emt_score", "late_ecm_mesenchymal_score", "growth_cell_cycle_score",
]
PHASE_PROGRAMS = {
    "early": ["early_immediate_response_score", "early_inflammatory_tnf_score", "hybrid_score"],
    "intermediate": ["intermediate_partial_emt_score", "hybrid_score", "emt_core_score"],
    "late": ["late_ecm_mesenchymal_score", "mesenchymal_score", "emt_core_score"],
    "growth": ["growth_cell_cycle_score"],
}


def condition_from_run_dir(run_dir: Path) -> str:
    # run labels are ...__emt_A549_EGF_vgfm
    m = re.search(r"emt_([A-Za-z0-9]+_(?:EGF|TGFB1|TNF))_vgfm", run_dir.name)
    if m:
        return m.group(1)
    # fallback to artifact input path if needed
    art = run_dir / "artifacts" / "model_artifact.json"
    if art.exists():
        data = json.loads(art.read_text())
        p = Path(data.get("input_adata_path", ""))
        if p.name.endswith(".h5ad"):
            return p.stem
    raise ValueError(f"Cannot parse condition from run dir {run_dir}")


def discover_runs() -> Dict[str, Path]:
    out = {}
    for rd in sorted(TRAIN_ROOT.glob("*__final__vgfm__emt_*_vgfm")):
        if (rd / "artifacts" / "evaluation_trajectory.npz").exists() and (rd / "artifacts" / "metrics.json").exists():
            cond = condition_from_run_dir(rd)
            out[cond] = rd
    return out


def load_eval_traj(npz_path: Path):
    z = np.load(npz_path, allow_pickle=True)
    time_points = z["time_points"].astype(float)
    points = [z[f"points_{i:06d}"] for i in range(len(time_points))]
    weights = [z[f"weights_{i:06d}"] for i in range(len(time_points))]
    manifest = json.loads(z["manifest_json"].item()) if "manifest_json" in z.files else {}
    return time_points, points, weights, manifest


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=float)
    v = np.asarray(values, dtype=float)
    ok = np.isfinite(v) & np.isfinite(w) & (w >= 0)
    if ok.sum() == 0 or w[ok].sum() <= 0:
        return float(np.nanmean(v[ok])) if ok.any() else np.nan
    return float(np.average(v[ok], weights=w[ok]))


def corr_safe(x, y, kind="pearson") -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.std(x[ok]) < 1e-9 or np.std(y[ok]) < 1e-9:
        return np.nan
    if kind == "spearman":
        return float(spearmanr(x[ok], y[ok]).statistic)
    return float(pearsonr(x[ok], y[ok]).statistic)


def fit_program_readout(adata: ad.AnnData, condition: str):
    X = np.asarray(adata.obsm["X_latent"], dtype=np.float32)
    scores = adata.obs[PROGRAMS].astype(float)
    # Ridge is smooth and auditable; RF is used only for a non-linear sensitivity check.
    model = make_pipeline(StandardScaler(), Ridge(alpha=10.0, random_state=0))
    model.fit(X, scores.values)
    pred = model.predict(X)
    r2 = {p: float(r2_score(scores[p].values, pred[:, i])) for i, p in enumerate(PROGRAMS)}
    # If ridge readout is poor for a score, generated values are still reported with a warning.
    return model, r2


def latent_endpoint_nn_stats(gen_endpoint: np.ndarray, obs_endpoint: np.ndarray) -> Dict[str, float]:
    if len(gen_endpoint) == 0 or len(obs_endpoint) == 0:
        return {"generated_to_observed_nn_mean": np.nan, "observed_to_generated_nn_mean": np.nan, "centroid_distance": np.nan}
    nn_obs = NearestNeighbors(n_neighbors=1).fit(obs_endpoint)
    d_go = nn_obs.kneighbors(gen_endpoint, return_distance=True)[0][:, 0]
    nn_gen = NearestNeighbors(n_neighbors=1).fit(gen_endpoint)
    d_og = nn_gen.kneighbors(obs_endpoint, return_distance=True)[0][:, 0]
    return {
        "generated_to_observed_nn_mean": float(np.mean(d_go)),
        "generated_to_observed_nn_median": float(np.median(d_go)),
        "observed_to_generated_nn_mean": float(np.mean(d_og)),
        "observed_to_generated_nn_median": float(np.median(d_og)),
        "centroid_distance": float(np.linalg.norm(np.mean(gen_endpoint, axis=0) - np.mean(obs_endpoint, axis=0))),
    }


def classify_route(row: pd.Series) -> str:
    mes = row.get("generated_delta_mesenchymal_score", np.nan)
    epi = row.get("generated_delta_epithelial_score", np.nan)
    hyb = row.get("generated_delta_hybrid_score", np.nan)
    growth = row.get("observed_delta_growth_cell_cycle_score", np.nan)
    # z-score-like program scores: 0.15-0.25 changes are visible but conservative.
    if np.isfinite(mes) and mes > 0.25 and (not np.isfinite(epi) or epi < -0.05):
        return "mesenchymal conversion"
    if np.isfinite(hyb) and hyb > 0.15 and (not np.isfinite(mes) or mes <= 0.25):
        return "partial/hybrid EMT"
    if np.isfinite(epi) and epi >= -0.05 and (not np.isfinite(mes) or mes < 0.15):
        return "epithelial-retaining response"
    if np.isfinite(growth) and abs(growth) > 0.20 and (not np.isfinite(mes) or abs(mes) < 0.20):
        return "growth-associated response"
    return "mixed/weak EMT route"


def main() -> None:
    runs = discover_runs()
    if len(runs) != 12:
        print(f"[warn] expected 12 condition runs, found {len(runs)}: {sorted(runs)}")
    records = []
    traj_rows = []
    endpoint_rows = []
    phase_rank_rows = []
    readout_rows = []
    per_condition_json = {}

    for condition, run_dir in sorted(runs.items()):
        adata_path = COND_DIR / f"{condition}.h5ad"
        adata = ad.read_h5ad(adata_path)
        cell_line = str(adata.obs["cell_line"].iloc[0])
        stimulus = str(adata.obs["stimulus"].iloc[0])
        model, readout_r2 = fit_program_readout(adata, condition)
        for prog, r2 in readout_r2.items():
            readout_rows.append({"condition": condition, "program": prog, "latent_readout_r2": r2})

        t_grid, points, weights, traj_manifest = load_eval_traj(run_dir / "artifacts" / "evaluation_trajectory.npz")
        metrics = json.loads((run_dir / "artifacts" / "metrics.json").read_text())
        w1 = metrics.get("w1_scores", [])
        tmv = metrics.get("tmv_scores", [])

        # Observed program means at five measured snapshots.
        obs_means = {}
        obs_counts = {}
        for ti, label in TIME_INDEX_TO_LABEL.items():
            mask = adata.obs["time_point_processed"].astype(int).values == ti
            obs_counts[label] = int(mask.sum())
            obs_means[label] = {p: float(np.nanmean(adata.obs.loc[mask, p].astype(float))) if mask.sum() else np.nan for p in PROGRAMS}

        # Generated program trajectory from locked CytoBridge rollout.
        pred_scores_by_t = []
        for i, t in enumerate(t_grid):
            pred = model.predict(points[i])
            row = {"condition": condition, "cell_line": cell_line, "stimulus": stimulus, "model_time": float(t), "model_time_index": int(round(float(t))) if abs(t - round(t)) < 1e-6 else np.nan}
            row["generated_mass_sum"] = float(np.nansum(weights[i]))
            for j, p in enumerate(PROGRAMS):
                row[f"generated_{p}"] = weighted_mean(pred[:, j], weights[i])
            traj_rows.append(row)
            pred_scores_by_t.append(row)

        # Compare generated and observed at measured snapshots.
        measured_rows = []
        for ti, label in TIME_INDEX_TO_LABEL.items():
            idx = int(np.argmin(np.abs(t_grid - ti)))
            pred = model.predict(points[idx])
            mask = adata.obs["time_point_processed"].astype(int).values == ti
            obs_endpoint_latent = np.asarray(adata.obsm["X_latent"])[mask]
            latent_stats = latent_endpoint_nn_stats(points[idx], obs_endpoint_latent) if ti == 4 else {}
            comp = {
                "condition": condition,
                "cell_line": cell_line,
                "stimulus": stimulus,
                "time_index": ti,
                "time_label": label,
                "observed_n_cells": int(mask.sum()),
                "generated_n_particles": int(points[idx].shape[0]),
                "generated_mass_sum": float(np.nansum(weights[idx])),
            }
            for p in PROGRAMS:
                obs_val = obs_means[label][p]
                gen_val = weighted_mean(pred[:, PROGRAMS.index(p)], weights[idx])
                comp[f"observed_{p}"] = obs_val
                comp[f"generated_{p}"] = gen_val
                comp[f"generated_minus_observed_{p}"] = float(gen_val - obs_val) if np.isfinite(gen_val) and np.isfinite(obs_val) else np.nan
            if ti == 4:
                comp.update(latent_stats)
            endpoint_rows.append(comp)
            measured_rows.append(comp)

        # Trend agreement across measured snapshots for core programs.
        summary = {
            "condition": condition,
            "cell_line": cell_line,
            "stimulus": stimulus,
            "run_id": run_dir.name,
            "run_dir": str(run_dir),
            "evaluation_trajectory_path": str(run_dir / "artifacts" / "evaluation_trajectory.npz"),
            "metrics_path": str(run_dir / "artifacts" / "metrics.json"),
            "mean_w1": float(np.nanmean(w1)) if w1 else np.nan,
            "endpoint_w1": float(w1[-1]) if w1 else np.nan,
            "mean_tmv": float(np.nanmean(tmv)) if tmv else np.nan,
            "endpoint_tmv": float(tmv[-1]) if tmv else np.nan,
            "observed_baseline_n": obs_counts["0d"],
            "observed_endpoint_n": obs_counts["7d"],
            "observed_endpoint_baseline_count_ratio": float(obs_counts["7d"] / max(obs_counts["0d"], 1)),
        }
        for p in PROGRAMS:
            obs_vec = [obs_means[TIME_INDEX_TO_LABEL[i]][p] for i in range(5)]
            gen_vec = [measured_rows[i][f"generated_{p}"] for i in range(5)]
            summary[f"trend_pearson_{p}"] = corr_safe(obs_vec, gen_vec, "pearson")
            summary[f"trend_spearman_{p}"] = corr_safe(obs_vec, gen_vec, "spearman")
            summary[f"observed_delta_{p}"] = float(obs_vec[-1] - obs_vec[0]) if np.all(np.isfinite([obs_vec[-1], obs_vec[0]])) else np.nan
            summary[f"generated_delta_{p}"] = float(gen_vec[-1] - gen_vec[0]) if np.all(np.isfinite([gen_vec[-1], gen_vec[0]])) else np.nan
            summary[f"endpoint_error_{p}"] = float(gen_vec[-1] - obs_vec[-1]) if np.all(np.isfinite([gen_vec[-1], obs_vec[-1]])) else np.nan
        summary["route_trend_match_core_mean"] = float(np.nanmean([summary[f"trend_pearson_{p}"] for p in CORE_PROGRAMS]))
        summary["route_trend_match_core_min"] = float(np.nanmin([summary[f"trend_pearson_{p}"] for p in CORE_PROGRAMS]))
        summary["route_class"] = classify_route(pd.Series(summary))
        # endpoint latent stats attached in ti=4 row
        endpoint4 = [r for r in measured_rows if r["time_index"] == 4][0]
        for k in ["generated_to_observed_nn_mean", "observed_to_generated_nn_mean", "centroid_distance"]:
            summary[f"endpoint_latent_{k}"] = endpoint4.get(k, np.nan)
        records.append(summary)
        per_condition_json[condition] = {"summary": summary, "readout_r2": readout_r2, "observed_time_counts": obs_counts, "trajectory_manifest": traj_manifest}

        # Program ranking by observed phase deltas and generated consistency.
        windows = {"early": (0, 1), "intermediate": (2, 3), "late": (3, 4), "growth": (0, 4)}
        measured_by_t = {r["time_index"]: r for r in measured_rows}
        for phase, (a, b) in windows.items():
            for p in PHASE_PROGRAMS[phase]:
                obs_delta = measured_by_t[b][f"observed_{p}"] - measured_by_t[a][f"observed_{p}"]
                gen_delta = measured_by_t[b][f"generated_{p}"] - measured_by_t[a][f"generated_{p}"]
                phase_rank_rows.append({
                    "condition": condition,
                    "cell_line": cell_line,
                    "stimulus": stimulus,
                    "phase": phase,
                    "program": p,
                    "observed_delta": float(obs_delta),
                    "generated_delta": float(gen_delta),
                    "agreement_sign": int(np.sign(obs_delta) == np.sign(gen_delta)) if np.isfinite(obs_delta) and np.isfinite(gen_delta) else 0,
                    "rank_score": float(abs(obs_delta) * (1.0 if np.sign(obs_delta) == np.sign(gen_delta) else 0.5)),
                })

    summary_df = pd.DataFrame(records).sort_values(["cell_line", "stimulus"])
    traj_df = pd.DataFrame(traj_rows)
    endpoint_df = pd.DataFrame(endpoint_rows)
    phase_df = pd.DataFrame(phase_rank_rows)
    readout_df = pd.DataFrame(readout_rows)

    # Rank phase programs within each condition/phase.
    phase_df["rank_within_condition_phase"] = phase_df.groupby(["condition", "phase"])["rank_score"].rank(ascending=False, method="first").astype(int)
    phase_df = phase_df.sort_values(["condition", "phase", "rank_within_condition_phase"])

    # Shared-vs-context-specific summary: programs with same sign across >=75% conditions are shared.
    shared_rows = []
    for p in PROGRAMS:
        deltas = summary_df[f"observed_delta_{p}"].dropna()
        pos_frac = float((deltas > 0).mean()) if len(deltas) else np.nan
        neg_frac = float((deltas < 0).mean()) if len(deltas) else np.nan
        shared_rows.append({
            "program": p,
            "mean_observed_delta": float(deltas.mean()) if len(deltas) else np.nan,
            "positive_fraction": pos_frac,
            "negative_fraction": neg_frac,
            "shared_direction": "shared_increase" if pos_frac >= 0.75 else ("shared_decrease" if neg_frac >= 0.75 else "context_specific"),
            "strongest_increase_condition": summary_df.loc[summary_df[f"observed_delta_{p}"].idxmax(), "condition"] if len(deltas) else "",
            "strongest_decrease_condition": summary_df.loc[summary_df[f"observed_delta_{p}"].idxmin(), "condition"] if len(deltas) else "",
        })
    shared_df = pd.DataFrame(shared_rows)

    # Write tables.
    summary_path = DOWN_DIR / "condition_route_summary.csv"
    traj_path = DOWN_DIR / "generated_program_trajectories.csv"
    endpoint_path = DOWN_DIR / "endpoint_and_measured_snapshot_comparison.csv"
    phase_path = DOWN_DIR / "phase_program_rankings.csv"
    readout_path = DOWN_DIR / "latent_program_readout_quality.csv"
    shared_path = DOWN_DIR / "shared_vs_context_specific_programs.csv"
    summary_df.to_csv(summary_path, index=False)
    traj_df.to_csv(traj_path, index=False)
    endpoint_df.to_csv(endpoint_path, index=False)
    phase_df.to_csv(phase_path, index=False)
    readout_df.to_csv(readout_path, index=False)
    shared_df.to_csv(shared_path, index=False)

    # Figures.
    sns.set_context("talk")
    sns.set_style("whitegrid")

    # 1. Core route heatmap: generated endpoint-baseline deltas.
    heat = summary_df.set_index("condition")[[f"generated_delta_{p}" for p in CORE_PROGRAMS] + ["observed_endpoint_baseline_count_ratio", "route_trend_match_core_mean"]]
    heat.columns = ["gen Δ epithelial", "gen Δ hybrid", "gen Δ mesenchymal", "obs 7d/0d cells", "core trend r"]
    plt.figure(figsize=(9, 7))
    sns.heatmap(heat, cmap="vlag", center=0, annot=True, fmt=".2f")
    plt.title("Condition-specific generated EMT route summaries")
    plt.tight_layout()
    f1 = FIG_DIR / "condition_route_heatmap.png"
    plt.savefig(f1, dpi=180)
    plt.close()

    # 2. Observed vs generated mesenchymal time-course facets.
    plot_rows = []
    for _, r in endpoint_df.iterrows():
        for p in CORE_PROGRAMS:
            plot_rows.append({"condition": r["condition"], "cell_line": r["cell_line"], "stimulus": r["stimulus"], "time_label": r["time_label"], "time_index": r["time_index"], "program": p.replace("_score", ""), "source": "observed", "score": r[f"observed_{p}"]})
            plot_rows.append({"condition": r["condition"], "cell_line": r["cell_line"], "stimulus": r["stimulus"], "time_label": r["time_label"], "time_index": r["time_index"], "program": p.replace("_score", ""), "source": "generated", "score": r[f"generated_{p}"]})
    plot_df = pd.DataFrame(plot_rows)
    g = sns.relplot(data=plot_df, x="time_index", y="score", hue="source", style="source", col="condition", col_wrap=4, row=None, kind="line", markers=True, facet_kws={"sharey": False}, height=2.4, aspect=1.25)
    g.set_titles("{col_name}")
    g.set_axis_labels("time (0,8h,1d,3d,7d)", "program score")
    for ax in g.axes.flat:
        ax.set_xticks([0,1,2,3,4])
        ax.set_xticklabels(TIME_LABELS, rotation=45)
    plt.tight_layout()
    f2 = FIG_DIR / "observed_vs_generated_core_program_timecourses.png"
    plt.savefig(f2, dpi=180)
    plt.close()

    # 3. Route classes and endpoint quality.
    plt.figure(figsize=(10, 5))
    order = summary_df.sort_values("endpoint_w1")["condition"].tolist()
    sns.barplot(data=summary_df, x="condition", y="endpoint_w1", hue="route_class", order=order, dodge=False)
    plt.xticks(rotation=60, ha="right")
    plt.ylabel("Endpoint W1 in latent space")
    plt.title("Generated endpoint fit and inferred EMT route class")
    plt.tight_layout()
    f3 = FIG_DIR / "endpoint_w1_by_route_class.png"
    plt.savefig(f3, dpi=180)
    plt.close()

    # 4. Phase program rankings top program per phase/condition.
    top_phase = phase_df[phase_df["rank_within_condition_phase"] == 1].copy()
    plt.figure(figsize=(12, 5))
    sns.scatterplot(data=top_phase[top_phase["phase"].isin(["early", "intermediate", "late"])], x="condition", y="phase", size="rank_score", hue="program", sizes=(80, 350))
    plt.xticks(rotation=60, ha="right")
    plt.title("Top early/intermediate/late EMT-associated program per condition")
    plt.tight_layout()
    f4 = FIG_DIR / "top_phase_programs.png"
    plt.savefig(f4, dpi=180)
    plt.close()

    manifest = {
        "analysis": "emt_condition_specific_cytobridge_downstream",
        "model_family": "vgfm",
        "dynamics_source": "locked CytoBridge evaluation_trajectory.npz from each trained condition-specific model",
        "posthoc_readout": "Ridge latent-to-gene-program score readout fit on observed cells within each condition; used only to annotate model-native latent rollouts.",
        "time_axis": TIME_INDEX_TO_LABEL,
        "input_condition_dir": str(COND_DIR),
        "training_runs_root": str(TRAIN_ROOT),
        "output_tables": {
            "condition_route_summary": str(summary_path),
            "generated_program_trajectories": str(traj_path),
            "endpoint_and_measured_snapshot_comparison": str(endpoint_path),
            "phase_program_rankings": str(phase_path),
            "latent_program_readout_quality": str(readout_path),
            "shared_vs_context_specific_programs": str(shared_path),
        },
        "figures": [str(f1), str(f2), str(f3), str(f4)],
        "warnings": [
            "Gene-program scores on generated states are latent readouts, not direct RNA reconstruction; readout R2 is reported per condition/program.",
            "Evaluation trajectories are deterministic VGFM rollouts with sigma=0 from observed baseline particles and model-learned growth weights.",
            "Endpoint W1/TMV are latent-space metrics from the training artifact and should be interpreted relative to the shared PCA scale."
        ],
        "per_condition": per_condition_json,
    }
    manifest_path = DOWN_DIR / "downstream_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps({
        "condition_route_summary": str(summary_path),
        "figures": manifest["figures"],
        "route_classes": summary_df["route_class"].value_counts().to_dict(),
        "shared_programs": shared_df[["program", "shared_direction"]].to_dict(orient="records"),
    }, indent=2))


if __name__ == "__main__":
    main()
