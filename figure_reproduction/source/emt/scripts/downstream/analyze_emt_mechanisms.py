#!/usr/bin/env python3
"""Mechanism-focused analysis of condition-specific EMT routes.

Inputs are the existing Cook/Vanderhyden preprocessed AnnData files and the locked
CytoBridge VGFM evaluation trajectories.  The script annotates model-native
latent rollouts with pathway/EMT/growth module readouts, quantifies which
mechanistic axes explain route specificity, and writes tables/figures for a
mechanism report.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUTPUT_DIR = Path(globals().get("output_dir", "/Users/zhenyizhang/Documents/GitHub/CytoBridge-agent-runtime-v2-20260309/emt"))
PREP = OUTPUT_DIR / "preprocessed" / "emt_cook_vanderhyden_all_conditions.h5ad"
COND_DIR = OUTPUT_DIR / "preprocessed" / "conditions"
TRAIN_ROOT = OUTPUT_DIR / "training_runs"
OUT = OUTPUT_DIR / "downstream" / "emt_mechanisms"
FIG = OUTPUT_DIR / "figures" / "emt_mechanisms"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

TIME_INDEX_TO_LABEL = {0: "0d", 1: "8h", 2: "1d", 3: "3d", 4: "7d"}
MODULES: Dict[str, List[str]] = {
    # Stimulus-proximal and canonical EMT mechanisms.
    "EGFR_AP1_MAPK": ["EGFR", "ERBB2", "ERBB3", "AREG", "EREG", "HBEGF", "FOS", "FOSB", "JUN", "JUNB", "JUND", "EGR1", "EGR2", "EGR3", "DUSP1", "DUSP5", "DUSP6", "SPRY2"],
    "TNF_NFKB_inflammatory": ["NFKBIA", "NFKBIZ", "NFKB1", "NFKB2", "RELA", "RELB", "TNFAIP3", "BIRC3", "ICAM1", "IL6", "CXCL1", "CXCL2", "CXCL3", "CXCL8", "CCL2", "CCL20"],
    "TGFB_SMAD_targets": ["TGFB1", "TGFB2", "TGFBR1", "TGFBR2", "SMAD2", "SMAD3", "SMAD4", "SMAD7", "TGFBI", "SERPINE1", "CTGF", "PMEPA1", "ID1", "ID2", "JUNB"],
    "EMT_TF_axis": ["ZEB1", "ZEB2", "SNAI1", "SNAI2", "TWIST1", "TWIST2", "PRRX1", "FOXC2", "SOX4", "HMGA2"],
    "ECM_remodeling": ["COL1A1", "COL1A2", "COL3A1", "COL5A1", "COL5A2", "FN1", "SPARC", "THBS1", "VCAN", "ITGA5", "LOX", "MMP2", "MMP9", "MMP14", "TIMP1", "TAGLN", "ACTA2"],
    "epithelial_junction": ["EPCAM", "CDH1", "KRT8", "KRT18", "KRT19", "KRT7", "CLDN3", "CLDN4", "CLDN7", "OCLN", "TJP1", "DSP", "DSG2", "JUP", "ESRP1", "GRHL2", "OVOL2"],
    "partial_basal_hybrid": ["LAMC2", "LAMB3", "ITGA6", "ITGB4", "KRT14", "KRT17", "KRT5", "S100A2", "JAG1", "NOTCH1", "TGFBI", "SERPINE1", "MMP7", "PLAUR", "ANXA1"],
    "cell_cycle_growth": ["MKI67", "TOP2A", "PCNA", "MCM2", "MCM3", "MCM4", "MCM5", "MCM6", "MCM7", "CDK1", "CCNB1", "CCNB2", "CDC20", "AURKA", "AURKB", "UBE2C", "TYMS", "RRM2"],
    "interferon_JAK_STAT": ["STAT1", "STAT2", "IRF1", "IRF7", "ISG15", "IFIT1", "IFIT2", "IFIT3", "IFI6", "IFI27", "OAS1", "OAS2", "MX1", "CXCL10"],
    "stress_p53_apoptosis": ["TP53", "CDKN1A", "MDM2", "GADD45A", "DDIT3", "ATF3", "BAX", "BBC3", "PMAIP1", "BTG2", "IER3", "HMOX1"],
}
MODULE_ORDER = list(MODULES.keys())
CORE_ROUTE_MODULES = ["TGFB_SMAD_targets", "EMT_TF_axis", "ECM_remodeling", "epithelial_junction", "partial_basal_hybrid", "TNF_NFKB_inflammatory", "EGFR_AP1_MAPK", "cell_cycle_growth"]


def condition_from_run_dir(run_dir: Path) -> str:
    m = re.search(r"emt_([A-Za-z0-9]+_(?:EGF|TGFB1|TNF))_vgfm", run_dir.name)
    if m:
        return m.group(1)
    raise ValueError(f"Cannot parse condition from {run_dir.name}")


def discover_runs() -> Dict[str, Path]:
    runs = {}
    for rd in sorted(TRAIN_ROOT.glob("*__final__vgfm__emt_*_vgfm")):
        if (rd / "artifacts" / "evaluation_trajectory.npz").exists():
            runs[condition_from_run_dir(rd)] = rd
    return runs


def score_modules(adata: ad.AnnData) -> tuple[pd.DataFrame, Dict[str, List[str]]]:
    genes_upper = pd.Index([str(g).upper() for g in adata.var_names])
    gene_to_idx = {g: i for i, g in enumerate(genes_upper)}
    X = adata.X.tocsr() if sparse.issparse(adata.X) else np.asarray(adata.X)
    scores = pd.DataFrame(index=adata.obs_names)
    matched = {}
    for module, genes in MODULES.items():
        idx, mg = [], []
        for g in genes:
            gi = gene_to_idx.get(g.upper())
            if gi is not None:
                idx.append(gi); mg.append(str(adata.var_names[gi]))
        matched[module] = mg
        if not idx:
            scores[module] = 0.0
            continue
        M = X[:, idx]
        if sparse.issparse(M):
            M = M.toarray()
        M = np.asarray(M, dtype=np.float32)
        mu = np.nanmean(M, axis=0, keepdims=True)
        sd = np.nanstd(M, axis=0, keepdims=True); sd[sd < 1e-6] = 1.0
        scores[module] = ((M - mu) / sd).mean(axis=1)
    return scores, matched


def load_traj(npz_path: Path):
    z = np.load(npz_path, allow_pickle=True)
    t = z["time_points"].astype(float)
    points = [z[f"points_{i:06d}"] for i in range(len(t))]
    weights = [z[f"weights_{i:06d}"] for i in range(len(t))]
    man = json.loads(z["manifest_json"].item()) if "manifest_json" in z.files else {}
    return t, points, weights, man


def wmean(vals, weights):
    vals = np.asarray(vals, dtype=float)
    weights = np.asarray(weights, dtype=float)
    ok = np.isfinite(vals) & np.isfinite(weights) & (weights >= 0)
    if ok.sum() == 0:
        return np.nan
    sw = weights[ok].sum()
    return float(np.average(vals[ok], weights=weights[ok])) if sw > 0 else float(np.nanmean(vals[ok]))


def safe_corr(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.std(x[ok]) < 1e-8 or np.std(y[ok]) < 1e-8:
        return np.nan
    return float(pearsonr(x[ok], y[ok]).statistic)


def anova_additive(df: pd.DataFrame, ycol: str) -> Dict[str, float]:
    d = df[[ycol, "cell_line", "stimulus"]].dropna().copy()
    y = d[ycol].values.astype(float)
    if len(d) < 6 or np.var(y) < 1e-12:
        return {"cell_line_fraction": np.nan, "stimulus_fraction": np.nan, "context_residual_fraction": np.nan, "total_ss": float(np.sum((y-y.mean())**2))}
    intercept = np.ones((len(d), 1))
    X_cell = pd.get_dummies(d["cell_line"], drop_first=True).values.astype(float)
    X_stim = pd.get_dummies(d["stimulus"], drop_first=True).values.astype(float)
    def sse(X):
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        return float(np.sum(resid**2))
    X_full = np.column_stack([intercept, X_cell, X_stim])
    X_no_cell = np.column_stack([intercept, X_stim])
    X_no_stim = np.column_stack([intercept, X_cell])
    sse_full = sse(X_full); sse_no_cell = sse(X_no_cell); sse_no_stim = sse(X_no_stim)
    ss_cell = max(0.0, sse_no_cell - sse_full)
    ss_stim = max(0.0, sse_no_stim - sse_full)
    ss_context = max(0.0, sse_full)
    denom = ss_cell + ss_stim + ss_context
    return {
        "cell_line_fraction": float(ss_cell / denom) if denom else np.nan,
        "stimulus_fraction": float(ss_stim / denom) if denom else np.nan,
        "context_residual_fraction": float(ss_context / denom) if denom else np.nan,
        "total_ss": float(np.sum((y-y.mean())**2)),
    }


def infer_mechanism(row: pd.Series) -> str:
    tgfb = row.get("gen_delta_TGFB_SMAD_targets", 0)
    tf = row.get("gen_delta_EMT_TF_axis", 0)
    ecm = row.get("gen_delta_ECM_remodeling", 0)
    epi = row.get("gen_delta_epithelial_junction", 0)
    partial = row.get("gen_delta_partial_basal_hybrid", 0)
    tnf = row.get("gen_delta_TNF_NFKB_inflammatory", 0)
    egfr = row.get("gen_delta_EGFR_AP1_MAPK", 0)
    cycle = row.get("gen_delta_cell_cycle_growth", 0)
    if tgfb > 0.12 and (tf > 0.05 or ecm > 0.15) and epi < 0.05:
        return "TGFβ/EMT-TF/ECM conversion axis"
    if partial > 0.18 and epi >= -0.08:
        return "partial-basal hybrid program with epithelial retention"
    if tnf > 0.18 and tf < 0.12 and ecm < 0.15:
        return "NFκB/AP-1 inflammatory route"
    if egfr > 0.12 and partial > 0.08 and abs(tf) < 0.15:
        return "EGFR/AP-1 plasticity route"
    if abs(cycle) > 0.12 and max(abs(tf), abs(ecm), abs(partial)) < 0.18:
        return "growth/cell-cycle dominated route"
    return "mixed context-specific route"


def main():
    print("[load] combined adata", flush=True)
    combined = ad.read_h5ad(PREP)
    module_scores, matched = score_modules(combined)
    X_latent = np.asarray(combined.obsm["X_latent"], dtype=np.float32)
    Y = module_scores[MODULE_ORDER].values.astype(np.float32)
    readout = make_pipeline(StandardScaler(), Ridge(alpha=10.0, random_state=0))
    readout.fit(X_latent, Y)
    Y_hat = readout.predict(X_latent)
    readout_rows = []
    for j, module in enumerate(MODULE_ORDER):
        readout_rows.append({
            "module": module,
            "matched_gene_count": len(matched[module]),
            "matched_genes": ";".join(matched[module]),
            "global_readout_r2": float(r2_score(Y[:, j], Y_hat[:, j])),
        })

    obs = combined.obs[["condition", "cell_line", "stimulus", "time_point_processed", "time_point"]].copy()
    for module in MODULE_ORDER:
        obs[module] = module_scores[module].values
    obs_means = obs.groupby(["condition", "cell_line", "stimulus", "time_point_processed", "time_point"], observed=True)[MODULE_ORDER].mean().reset_index()

    runs = discover_runs()
    cond_rows = []
    traj_rows = []
    phase_rows = []
    driver_rows = []
    gene_pc = np.asarray(combined.varm.get("PCs", np.zeros((combined.n_vars, X_latent.shape[1]))), dtype=np.float32)
    var_names = np.asarray(combined.var_names.astype(str))
    gene_module_membership = {g: [] for g in var_names}
    upper_to_gene = {str(g).upper(): str(g) for g in var_names}
    for module, genes in matched.items():
        for g in genes:
            gene_module_membership[g].append(module)

    for condition, run_dir in sorted(runs.items()):
        cond_adata = ad.read_h5ad(COND_DIR / f"{condition}.h5ad")
        cell_line = str(cond_adata.obs["cell_line"].iloc[0]); stimulus = str(cond_adata.obs["stimulus"].iloc[0])
        t_grid, points, weights, traj_manifest = load_traj(run_dir / "artifacts" / "evaluation_trajectory.npz")
        pred_by_time = readout.predict(np.vstack(points))
        # Split predictions back by time.
        offset = 0
        gen_means = []
        for i, t in enumerate(t_grid):
            n = points[i].shape[0]
            pred = pred_by_time[offset:offset+n]; offset += n
            row = {"condition": condition, "cell_line": cell_line, "stimulus": stimulus, "model_time": float(t), "mass_sum": float(np.nansum(weights[i]))}
            for j, module in enumerate(MODULE_ORDER):
                row[f"gen_{module}"] = wmean(pred[:, j], weights[i])
            traj_rows.append(row)
            gen_means.append(row)
        gen0 = gen_means[0]; gen4 = gen_means[int(np.argmin(abs(t_grid-4.0)))]
        gen1 = gen_means[int(np.argmin(abs(t_grid-1.0)))]
        gen2 = gen_means[int(np.argmin(abs(t_grid-2.0)))]
        gen3 = gen_means[int(np.argmin(abs(t_grid-3.0)))]

        cobs = obs_means[obs_means["condition"] == condition].copy()
        by_t = {int(r.time_point_processed): r for _, r in cobs.iterrows()}
        rec = {"condition": condition, "cell_line": cell_line, "stimulus": stimulus, "run_dir": str(run_dir)}
        rec["model_log_mass_change"] = float(np.log((gen4["mass_sum"] + 1e-12) / (gen0["mass_sum"] + 1e-12)))
        for module in MODULE_ORDER:
            rec[f"obs_delta_{module}"] = float(by_t[4][module] - by_t[0][module]) if 4 in by_t and 0 in by_t else np.nan
            rec[f"gen_delta_{module}"] = float(gen4[f"gen_{module}"] - gen0[f"gen_{module}"])
            rec[f"early_obs_delta_{module}"] = float(by_t[1][module] - by_t[0][module]) if 1 in by_t and 0 in by_t else np.nan
            rec[f"early_gen_delta_{module}"] = float(gen1[f"gen_{module}"] - gen0[f"gen_{module}"])
            rec[f"intermediate_obs_delta_{module}"] = float(by_t[3][module] - by_t[2][module]) if 3 in by_t and 2 in by_t else np.nan
            rec[f"intermediate_gen_delta_{module}"] = float(gen3[f"gen_{module}"] - gen2[f"gen_{module}"])
            rec[f"late_obs_delta_{module}"] = float(by_t[4][module] - by_t[3][module]) if 4 in by_t and 3 in by_t else np.nan
            rec[f"late_gen_delta_{module}"] = float(gen4[f"gen_{module}"] - gen3[f"gen_{module}"])
            obs_vec = [by_t[k][module] if k in by_t else np.nan for k in range(5)]
            gen_vec = [gen_means[int(np.argmin(abs(t_grid-float(k))))][f"gen_{module}"] for k in range(5)]
            rec[f"trend_r_{module}"] = safe_corr(obs_vec, gen_vec)
        rec["mechanism_class"] = infer_mechanism(pd.Series(rec))
        cond_rows.append(rec)

        # Phase mechanism ranking.
        phase_defs = {
            "early_0d_to_8h": ("early_obs_delta_", "early_gen_delta_"),
            "intermediate_1d_to_3d": ("intermediate_obs_delta_", "intermediate_gen_delta_"),
            "late_3d_to_7d": ("late_obs_delta_", "late_gen_delta_"),
            "endpoint_0d_to_7d": ("obs_delta_", "gen_delta_"),
        }
        for phase, (op, gp) in phase_defs.items():
            for module in MODULE_ORDER:
                od = rec[f"{op}{module}"]; gd = rec[f"{gp}{module}"]
                phase_rows.append({"condition": condition, "cell_line": cell_line, "stimulus": stimulus, "phase": phase, "module": module, "observed_delta": od, "generated_delta": gd, "sign_match": int(np.sign(od)==np.sign(gd)) if np.isfinite(od) and np.isfinite(gd) else 0, "rank_score": float(abs(od) * (1.0 if np.sign(od)==np.sign(gd) else 0.5)) if np.isfinite(od) else np.nan})

        # PCA-loading latent displacement genes from generated endpoint route.
        init_latent = points[0]
        final_latent = points[int(np.argmin(abs(t_grid-4.0)))]
        w0 = weights[0]; wf = weights[int(np.argmin(abs(t_grid-4.0)))]
        delta_lat = np.average(final_latent, axis=0, weights=wf) - np.average(init_latent, axis=0, weights=w0)
        gene_scores = gene_pc @ delta_lat
        top_idx = np.argsort(-np.abs(gene_scores))[:50]
        for rank, gi in enumerate(top_idx, 1):
            gene = str(var_names[gi])
            driver_rows.append({"condition": condition, "cell_line": cell_line, "stimulus": stimulus, "rank": rank, "gene": gene, "latent_pca_driver_score": float(gene_scores[gi]), "abs_score": float(abs(gene_scores[gi])), "module_membership": ";".join(gene_module_membership.get(gene, []))})

    cond_df = pd.DataFrame(cond_rows).sort_values(["cell_line", "stimulus"])
    traj_df = pd.DataFrame(traj_rows)
    phase_df = pd.DataFrame(phase_rows)
    driver_df = pd.DataFrame(driver_rows)
    readout_df = pd.DataFrame(readout_rows)
    phase_df["rank_within_condition_phase"] = phase_df.groupby(["condition", "phase"])["rank_score"].rank(ascending=False, method="first")

    # Variance decomposition across module endpoint deltas.
    var_rows = []
    for source in ["obs", "gen"]:
        for module in MODULE_ORDER:
            tmp = cond_df[["condition", "cell_line", "stimulus", f"{source}_delta_{module}"]].rename(columns={f"{source}_delta_{module}": "delta"})
            res = anova_additive(tmp, "delta")
            res.update({"source": source, "module": module})
            var_rows.append(res)
    var_df = pd.DataFrame(var_rows)

    # Mechanistic contrast table between cell lines and stimuli.
    contrast_rows = []
    for source in ["obs", "gen"]:
        for module in MODULE_ORDER:
            col = f"{source}_delta_{module}"
            stim_mean = cond_df.groupby("stimulus")[col].mean().to_dict()
            cell_mean = cond_df.groupby("cell_line")[col].mean().to_dict()
            contrast_rows.append({"source": source, "module": module, **{f"stimulus_mean_{k}": v for k, v in stim_mean.items()}, **{f"cell_line_mean_{k}": v for k, v in cell_mean.items()}, "range_across_conditions": float(cond_df[col].max() - cond_df[col].min())})
    contrast_df = pd.DataFrame(contrast_rows)

    # Write tables.
    paths = {
        "mechanism_condition_summary": OUT / "mechanism_condition_summary.csv",
        "mechanism_trajectory_modules": OUT / "mechanism_trajectory_modules.csv",
        "mechanism_phase_rankings": OUT / "mechanism_phase_rankings.csv",
        "mechanism_variance_decomposition": OUT / "mechanism_variance_decomposition.csv",
        "mechanism_contrasts": OUT / "mechanism_contrasts.csv",
        "mechanism_latent_pca_gene_drivers": OUT / "mechanism_latent_pca_gene_drivers.csv",
        "mechanism_module_readout_quality": OUT / "mechanism_module_readout_quality.csv",
    }
    cond_df.to_csv(paths["mechanism_condition_summary"], index=False)
    traj_df.to_csv(paths["mechanism_trajectory_modules"], index=False)
    phase_df.sort_values(["condition", "phase", "rank_within_condition_phase"]).to_csv(paths["mechanism_phase_rankings"], index=False)
    var_df.to_csv(paths["mechanism_variance_decomposition"], index=False)
    contrast_df.to_csv(paths["mechanism_contrasts"], index=False)
    driver_df.to_csv(paths["mechanism_latent_pca_gene_drivers"], index=False)
    readout_df.to_csv(paths["mechanism_module_readout_quality"], index=False)

    sns.set_style("whitegrid"); sns.set_context("talk")
    # Heatmap generated deltas.
    heat = cond_df.set_index("condition")[[f"gen_delta_{m}" for m in CORE_ROUTE_MODULES]]
    heat.columns = [c.replace("gen_delta_", "") for c in heat.columns]
    plt.figure(figsize=(12, 7))
    sns.heatmap(heat, cmap="vlag", center=0, annot=True, fmt=".2f")
    plt.title("VGFM-generated endpoint mechanism deltas")
    plt.tight_layout(); f1 = FIG / "mechanism_generated_delta_heatmap.png"; plt.savefig(f1, dpi=180); plt.close()

    # Variance decomposition stacked bar generated.
    vgen = var_df[var_df["source"] == "gen"].set_index("module").loc[MODULE_ORDER]
    stack = vgen[["cell_line_fraction", "stimulus_fraction", "context_residual_fraction"]]
    plt.figure(figsize=(11, 5))
    bottom = np.zeros(len(stack))
    colors = ["#4C78A8", "#F58518", "#54A24B"]
    labels = ["cell-line", "stimulus", "context residual"]
    for i, col in enumerate(stack.columns):
        plt.bar(stack.index, stack[col], bottom=bottom, label=labels[i], color=colors[i])
        bottom += stack[col].fillna(0).values
    plt.xticks(rotation=60, ha="right"); plt.ylabel("fraction of endpoint-delta variation")
    plt.title("Additive variance decomposition of generated mechanism routes")
    plt.legend(); plt.tight_layout(); f2 = FIG / "mechanism_variance_decomposition.png"; plt.savefig(f2, dpi=180); plt.close()

    # Stimulus proximal vs EMT-TF/ECM state space.
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=cond_df, x="gen_delta_TNF_NFKB_inflammatory", y="gen_delta_TGFB_SMAD_targets", hue="cell_line", style="stimulus", size="gen_delta_ECM_remodeling", sizes=(80, 420))
    for _, r in cond_df.iterrows():
        plt.text(r["gen_delta_TNF_NFKB_inflammatory"]+0.005, r["gen_delta_TGFB_SMAD_targets"]+0.005, r["condition"], fontsize=8)
    plt.axhline(0, color="grey", lw=0.8); plt.axvline(0, color="grey", lw=0.8)
    plt.title("Stimulus-proximal axes separate context-specific EMT routes")
    plt.tight_layout(); f3 = FIG / "mechanism_stimulus_axis_scatter.png"; plt.savefig(f3, dpi=180); plt.close()

    # Mass/growth versus EMT mechanics.
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=cond_df, x="model_log_mass_change", y="gen_delta_cell_cycle_growth", hue="mechanism_class", style="stimulus", size="gen_delta_EMT_TF_axis", sizes=(80, 420))
    for _, r in cond_df.iterrows():
        plt.text(r["model_log_mass_change"]+0.005, r["gen_delta_cell_cycle_growth"]+0.005, r["condition"], fontsize=8)
    plt.axhline(0, color="grey", lw=0.8); plt.axvline(0, color="grey", lw=0.8)
    plt.title("VGFM mass change reveals growth-coupled versus EMT-coupled routes")
    plt.tight_layout(); f4 = FIG / "mechanism_mass_growth_scatter.png"; plt.savefig(f4, dpi=180); plt.close()

    # Top phase modules plot.
    top_phase = phase_df[phase_df["rank_within_condition_phase"] == 1].copy()
    top_phase = top_phase[top_phase["phase"].isin(["early_0d_to_8h", "intermediate_1d_to_3d", "late_3d_to_7d"])]
    plt.figure(figsize=(13, 5))
    sns.scatterplot(data=top_phase, x="condition", y="phase", hue="module", size="rank_score", sizes=(80, 400))
    plt.xticks(rotation=60, ha="right")
    plt.title("Dominant phase-specific mechanism module per condition")
    plt.tight_layout(); f5 = FIG / "mechanism_top_phase_modules.png"; plt.savefig(f5, dpi=180); plt.close()

    manifest = {
        "analysis": "condition_specific_emt_mechanisms",
        "dynamics_source": "locked CytoBridge VGFM evaluation_trajectory.npz artifacts",
        "posthoc_readout": "Global ridge readout from shared X_latent to observed gene-module scores; used to annotate generated model-native trajectories.",
        "module_gene_sets": MODULES,
        "matched_genes": matched,
        "tables": {k: str(v) for k, v in paths.items()},
        "figures": [str(f1), str(f2), str(f3), str(f4), str(f5)],
        "warnings": [
            "Module scores are gene-set summaries, not causal perturbation proof.",
            "Generated module trajectories are readouts from model-native latent rollouts; readout R2 is reported.",
            "Additive variance decomposition has one observation per cell-line/stimulus pair; residual is interpreted as context-specific interaction, not replicated experimental error.",
            "PCA-loading gene drivers are latent route contributors, not direct RNA reconstruction or causal driver validation."
        ],
    }
    (OUT / "mechanism_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"tables": {k: str(v) for k, v in paths.items()}, "figures": manifest["figures"], "mechanism_classes": cond_df["mechanism_class"].value_counts().to_dict()}, indent=2))

if __name__ == "__main__":
    main()
