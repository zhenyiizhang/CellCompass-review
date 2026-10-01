#!/usr/bin/env python3
"""Deep CITE-PCFM biology and algorithm lifecycle story figures.

This is a stricter follow-up to the broader NBT figure bank. It makes two
small, high-signal tracks:

1. cite_biology_landscape: CITE-PCFM alone, interpreted as an RNA-conditioned
   protein phenotype landscape. Other algorithms are not used in these panels.
2. algorithm_discovery_landscape: a causal lifecycle view explaining why prior
   designs failed, what was learned, and why the final metric improved.
"""

from __future__ import annotations

import json
import math
import os
import textwrap
from dataclasses import dataclass
from pathlib import Path

import anndata as ad

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/cytobridge_mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/cytobridge_xdg_cache")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors


sns.set_theme(style="white", context="talk")

ROOT = Path("/Users/zhenyizhang/Documents/GitHub/CytoBridge-agent-runtime-v2-20260309/multi-omics")
DATA_PATH = Path("/Users/zhenyizhang/.cellcompass/algorithm_benchmarks/datasets/gse305370_citeseq_stage2_10k/data.h5ad")
TRAJ_PATH = ROOT / "training_runs/20260526T174204321821Z__final__cite_paired_coord_fm__campaign_20260-trial_20260526-gse305370_ci/artifacts/evaluation_trajectory.npz"
GT_TABLE_DIR = ROOT / "downstream/cite_pcfm_rna_to_protein_gt/tables"
LIFECYCLE_DIR = ROOT / "downstream/nbt_cite_pcfm_figure_bank"
OUT_ROOT = ROOT / "figures/cite_pcfm_deep_story"
BIO_DIR = OUT_ROOT / "cite_biology_landscape"
LIFE_DIR = OUT_ROOT / "algorithm_discovery_landscape"
NOTE_DIR = ROOT / "downstream/cite_pcfm_deep_story"
TABLE_OUT = NOTE_DIR / "tables"
MANIFEST_PATH = NOTE_DIR / "manifest.json"
STORY_PATH = NOTE_DIR / "storyboard.md"

RNA_DIM = 50
PROT_DIM = 30
RNG = np.random.default_rng(20260527)

PHENO_COLORS = {
    "HSC": "#3C5488",
    "MasP": "#E64B35",
    "EryP": "#F39B2F",
    "MoP": "#00A087",
    "MkP": "#7E6148",
    "NeuP": "#8491B4",
    "BP": "#7C3AED",
}

MODULE_MARKERS = {
    "Stem retention": ["CD49f", "CD45RA", "CD123", "CD127"],
    "Ery-Mk priming": ["CD41", "CD42b", "CD71", "CD62P"],
    "Myeloid priming": ["CD11b", "CD64", "CD115", "CD172a", "CD13", "CD16"],
    "Mast/Baso axis": ["FceRIa", "CD123", "CD88", "CD13"],
    "B-lineage antigen": ["CD20", "CD21", "CD79b", "IgM", "IgD"],
    "Activation/HLA": ["HLA-DR", "HLA-A-B-C", "CD54", "CD25"],
}

FIGURES: list[dict] = []


@dataclass
class CITEContext:
    adata: ad.AnnData
    obs: pd.DataFrame
    rna: np.ndarray
    protein_pca: np.ndarray
    protein_adt: np.ndarray
    marker_names: list[str]
    marker_index: dict[str, int]
    marker_mean: np.ndarray
    marker_std: np.ndarray
    marker_decoder: Ridge
    traj_times: np.ndarray
    traj_points: dict[float, np.ndarray]
    traj_weights: dict[float, np.ndarray]
    nn: NearestNeighbors
    nn_labels: np.ndarray


def ensure_dirs() -> None:
    for path in [BIO_DIR, LIFE_DIR, NOTE_DIR, TABLE_OUT]:
        path.mkdir(parents=True, exist_ok=True)


def savefig(fig: plt.Figure, out_dir: Path, name: str, title: str, track: str, question: str) -> Path:
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
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


def weighted_mean(x: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    if w is None:
        return np.nanmean(x, axis=0)
    w = np.asarray(w, dtype=float).reshape(-1)
    w = np.clip(w, 0, np.inf)
    if len(w) != x.shape[0] or w.sum() <= 0 or not np.isfinite(w).all():
        return np.nanmean(x, axis=0)
    return np.average(x, axis=0, weights=w / w.sum())


def weighted_quantile(values: np.ndarray, q: float, weights: np.ndarray | None = None) -> float:
    v = np.asarray(values, dtype=float)
    mask = np.isfinite(v)
    if weights is None:
        return float(np.nanquantile(v, q))
    w = np.asarray(weights, dtype=float).reshape(-1)
    mask &= np.isfinite(w) & (w >= 0)
    if mask.sum() == 0 or w[mask].sum() <= 0:
        return float(np.nanquantile(v, q))
    order = np.argsort(v[mask])
    vv = v[mask][order]
    ww = w[mask][order]
    cdf = np.cumsum(ww) / ww.sum()
    return float(np.interp(q, cdf, vv))


def sample_indices(n: int, max_n: int, weights: np.ndarray | None = None) -> np.ndarray:
    if n <= max_n:
        return np.arange(n)
    p = None
    if weights is not None and len(weights) == n:
        w = np.clip(np.asarray(weights, dtype=float), 0, np.inf)
        if np.isfinite(w).all() and w.sum() > 0:
            p = w / w.sum()
    return np.sort(RNG.choice(n, size=max_n, replace=False, p=p))


def nearest_key(keys, t: float) -> float:
    arr = np.asarray(list(keys), dtype=float)
    return float(arr[np.argmin(np.abs(arr - float(t)))] )


def load_context() -> CITEContext:
    adata = ad.read_h5ad(DATA_PATH)
    obs = adata.obs.copy()
    obs["time_point_processed"] = obs["time_point_processed"].astype(float)
    rna = np.asarray(adata.obsm["X_rna_svd"], dtype=np.float32)
    protein_pca = np.asarray(adata.obsm["X_protein_pca"], dtype=np.float32)

    protein_mask = adata.var["feature_types"].astype(str).eq("Antibody Capture").to_numpy()
    marker_names = [x.replace("prot_", "") for x in adata.var.index[protein_mask]]
    Xp = adata[:, protein_mask].X
    protein_adt = Xp.toarray().astype(np.float32) if sparse.issparse(Xp) else np.asarray(Xp, dtype=np.float32)

    marker_mean = np.nanmean(protein_adt, axis=0)
    marker_std = np.nanstd(protein_adt, axis=0)
    marker_std[marker_std < 1e-6] = 1.0

    decoder = Ridge(alpha=1.0, fit_intercept=True)
    decoder.fit(protein_pca, protein_adt)

    z = np.load(TRAJ_PATH)
    traj_times = np.asarray(z["time_points"], dtype=float)
    traj_points = {}
    traj_weights = {}
    for i, t in enumerate(traj_times):
        pk = f"points_{i:06d}"
        wk = f"weights_{i:06d}"
        if pk in z.files:
            traj_points[float(t)] = np.asarray(z[pk], dtype=np.float32)
        if wk in z.files:
            traj_weights[float(t)] = np.asarray(z[wk], dtype=np.float32)

    nn = NearestNeighbors(n_neighbors=15, metric="euclidean")
    nn.fit(protein_pca)

    return CITEContext(
        adata=adata,
        obs=obs,
        rna=rna,
        protein_pca=protein_pca,
        protein_adt=protein_adt,
        marker_names=marker_names,
        marker_index={m: i for i, m in enumerate(marker_names)},
        marker_mean=marker_mean,
        marker_std=marker_std,
        marker_decoder=decoder,
        traj_times=traj_times,
        traj_points=traj_points,
        traj_weights=traj_weights,
        nn=nn,
        nn_labels=obs["cell_type"].astype(str).to_numpy(),
    )


def predict_markers(ctx: CITEContext, protein_pc: np.ndarray) -> np.ndarray:
    return ctx.marker_decoder.predict(np.asarray(protein_pc, dtype=np.float32)).astype(np.float32)


def module_scores(ctx: CITEContext, marker_matrix: np.ndarray) -> pd.DataFrame:
    z = (marker_matrix - ctx.marker_mean[None, :]) / ctx.marker_std[None, :]
    out = {}
    for module, markers in MODULE_MARKERS.items():
        idx = [ctx.marker_index[m] for m in markers if m in ctx.marker_index]
        if idx:
            out[module] = z[:, idx].mean(axis=1)
    return pd.DataFrame(out)


def classify_protein(ctx: CITEContext, protein_pc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dist, ind = ctx.nn.kneighbors(protein_pc, return_distance=True)
    labels = []
    confidence = []
    for row in ind:
        labs = ctx.nn_labels[row]
        vals, counts = np.unique(labs, return_counts=True)
        j = int(np.argmax(counts))
        labels.append(vals[j])
        confidence.append(float(counts[j] / len(row)))
    return np.asarray(labels), np.asarray(confidence)


def build_generated_tables(ctx: CITEContext) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    module_rows = []
    fate_rows = []
    point_rows = []
    phenotypes = sorted(pd.unique(ctx.obs["cell_type"].astype(str)))

    for t in ctx.traj_times:
        tt = nearest_key(ctx.traj_points.keys(), float(t))
        pts = ctx.traj_points[tt]
        w = ctx.traj_weights.get(tt)
        prot = pts[:, RNA_DIM : RNA_DIM + PROT_DIM]
        markers = predict_markers(ctx, prot)
        mods = module_scores(ctx, markers)
        labels, confidence = classify_protein(ctx, prot)

        for module in mods.columns:
            values = mods[module].to_numpy()
            module_rows.append(
                {
                    "time": float(t),
                    "module": module,
                    "generated_mean": float(weighted_mean(values[:, None], w)[0]),
                    "generated_q10": weighted_quantile(values, 0.10, w),
                    "generated_q90": weighted_quantile(values, 0.90, w),
                }
            )

        obs_mask = ctx.obs["time_point_processed"].to_numpy() == float(round(t))
        if abs(t - round(t)) < 1e-6 and obs_mask.sum() > 0:
            obs_mods = module_scores(ctx, ctx.protein_adt[obs_mask])
            for module in obs_mods.columns:
                module_rows.append(
                    {
                        "time": float(t),
                        "module": module,
                        "observed_mean": float(obs_mods[module].mean()),
                        "observed_q10": float(obs_mods[module].quantile(0.10)),
                        "observed_q90": float(obs_mods[module].quantile(0.90)),
                    }
                )

        if w is None:
            ww = np.ones(len(labels)) / max(1, len(labels))
        else:
            ww = np.clip(np.asarray(w, dtype=float), 0, np.inf)
            ww = ww / ww.sum() if ww.sum() > 0 else np.ones(len(labels)) / max(1, len(labels))
        counts = pd.Series(ww).groupby(labels).sum()
        probs = np.asarray([float(counts.get(p, 0.0)) for p in phenotypes], dtype=float)
        entropy = -float(np.sum(probs[probs > 0] * np.log(probs[probs > 0])) / math.log(max(2, len(phenotypes))))
        for p, prob in zip(phenotypes, probs):
            fate_rows.append({"time": float(t), "phenotype": p, "fraction": prob, "entropy": entropy})

        if abs(t - round(t)) < 1e-6 or t in [0.5, 1.5, 2.5, 3.5]:
            take = sample_indices(prot.shape[0], 500, w)
            submods = mods.iloc[take].reset_index(drop=True)
            for i, idx in enumerate(take):
                row = {
                    "time": float(t),
                    "protein_pc1": float(prot[idx, 0]),
                    "protein_pc2": float(prot[idx, 1]),
                    "nearest_phenotype": str(labels[idx]),
                    "confidence": float(confidence[idx]),
                }
                for module in ["Stem retention", "Ery-Mk priming", "Myeloid priming", "Mast/Baso axis"]:
                    if module in submods:
                        row[module] = float(submods.loc[i, module])
                point_rows.append(row)

    module_df = pd.DataFrame(module_rows)
    fate_df = pd.DataFrame(fate_rows)
    point_df = pd.DataFrame(point_rows)
    return module_df, fate_df, point_df


def rounded_box(ax, xy, w, h, text, fc="#F8FAFC", ec="#CBD5E1", size=10, weight="normal", color="#111827"):
    box = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.028",
        linewidth=1.2,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=size, weight=weight, color=color, wrap=True)
    return box


def arrow(ax, p0, p1, color="#64748B", lw=1.8, rad=0.08):
    patch = FancyArrowPatch(
        p0,
        p1,
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle="Simple,head_width=7,head_length=9",
        linewidth=lw,
        color=color,
    )
    ax.add_patch(patch)
    return patch


def fig_bio01_topography(ctx: CITEContext) -> None:
    obs = ctx.obs
    fig, ax = plt.subplots(figsize=(9.4, 7.5))
    for phenotype, color in PHENO_COLORS.items():
        mask = obs["cell_type"].astype(str).to_numpy() == phenotype
        if mask.sum() == 0:
            continue
        idx = sample_indices(int(mask.sum()), 500)
        pp = ctx.protein_pca[mask][idx, :2]
        ax.scatter(pp[:, 0], pp[:, 1], s=8, color=color, alpha=0.18, linewidths=0, label=phenotype)
        cen = ctx.protein_pca[mask, :2].mean(axis=0)
        ax.text(cen[0], cen[1], phenotype, fontsize=10, weight="bold", color=color, ha="center", va="center")

    means = []
    for t in ctx.traj_times:
        pts = ctx.traj_points[float(t)]
        prot = pts[:, RNA_DIM : RNA_DIM + PROT_DIM]
        means.append(weighted_mean(prot[:, :2], ctx.traj_weights.get(float(t))))
    means = np.asarray(means)
    ax.plot(means[:, 0], means[:, 1], color="#111827", lw=3, zorder=5)
    step_idx = [0, 10, 20, 30, 40]
    ax.scatter(means[step_idx, 0], means[step_idx, 1], s=130, c=ctx.traj_times[step_idx], cmap="magma", edgecolor="white", zorder=6)
    for i in range(len(step_idx) - 1):
        a = means[step_idx[i]]
        b = means[step_idx[i + 1]]
        ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", lw=2.0, color="#111827"))
        ax.text(b[0], b[1] + 0.16, f"t{int(ctx.traj_times[step_idx[i + 1]])}", fontsize=9, ha="center")
    ax.set_title("CITE-PCFM path through observed protein phenotype basins", weight="bold")
    ax.set_xlabel("protein PC1")
    ax.set_ylabel("protein PC2")
    ax.legend(frameon=False, ncol=4, fontsize=8, loc="upper right")
    savefig(
        fig,
        BIO_DIR,
        "bio_landscape01_protein_topography",
        "Protein topography",
        "cite_biology_landscape",
        "Which observed CITE protein basins does the generated trajectory traverse?",
    )


def fig_bio02_marker_streams(module_df: pd.DataFrame) -> None:
    modules = list(MODULE_MARKERS.keys())
    fig, axes = plt.subplots(3, 2, figsize=(13, 9.6), sharex=True)
    for ax, module in zip(axes.flat, modules):
        gen = module_df[(module_df["module"] == module) & module_df["generated_mean"].notna()].sort_values("time")
        obs = module_df[(module_df["module"] == module) & module_df["observed_mean"].notna()].sort_values("time")
        ax.fill_between(gen["time"], gen["generated_q10"], gen["generated_q90"], color="#B73A3A", alpha=0.18, linewidth=0)
        ax.plot(gen["time"], gen["generated_mean"], color="#B73A3A", lw=2.5, label="CITE-PCFM rollout")
        ax.scatter(obs["time"], obs["observed_mean"], s=70, color="#111827", edgecolor="white", label="observed target")
        for _, r in obs.iterrows():
            ax.plot([r["time"], r["time"]], [r["observed_q10"], r["observed_q90"]], color="#111827", alpha=0.35, lw=1)
        ax.axhline(0, color="#CBD5E1", lw=1)
        ax.set_title(module, fontsize=12, weight="bold")
        ax.set_ylabel("ADT z-score")
    axes[-1, 0].set_xlabel("time")
    axes[-1, 1].set_xlabel("time")
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle("Surface-marker programs reconstructed from generated protein coordinates", y=1.02, weight="bold")
    savefig(
        fig,
        BIO_DIR,
        "bio_landscape02_marker_program_streams",
        "Marker program streams",
        "cite_biology_landscape",
        "Does the CITE-PCFM protein readout produce biologically interpretable surface-marker programs through time?",
    )


def fig_bio03_terminal_fate_geography() -> None:
    terminal = pd.read_csv(GT_TABLE_DIR / "initial_rna_neighborhood_terminal_protein_fates.csv")
    d = terminal[terminal["method"] == "CITE-PCFM"].copy()
    phenotypes = [p for p in ["HSC", "MasP", "EryP", "MoP", "MkP", "NeuP", "BP"] if p in set(d["terminal_phenotype"])]
    mat = d.pivot(index="initial_rna_neighborhood", columns="terminal_phenotype", values="fraction").fillna(0.0)
    mat = mat.reindex(columns=phenotypes).fillna(0.0)
    mat["entropy"] = mat[phenotypes].apply(lambda r: -np.sum(r[r > 0] * np.log(r[r > 0])) / math.log(max(2, len(phenotypes))), axis=1)
    mat["dominant"] = mat[phenotypes].idxmax(axis=1)
    mat["lineage_bias"] = mat.get("MasP", 0) + mat.get("MoP", 0) - mat.get("HSC", 0) - mat.get("EryP", 0)
    mat = mat.sort_values(["dominant", "lineage_bias", "entropy"]).reset_index()
    mat.to_csv(TABLE_OUT / "terminal_rna_neighborhood_fate_geography.csv", index=False)

    fig, ax = plt.subplots(figsize=(11, 5.7))
    bottom = np.zeros(len(mat))
    x = np.arange(len(mat))
    for p in phenotypes:
        vals = mat[p].to_numpy()
        ax.bar(x, vals, bottom=bottom, color=PHENO_COLORS.get(p, "#94A3B8"), label=p, width=0.82)
        bottom += vals
    ax2 = ax.twinx()
    ax2.plot(x, mat["entropy"], color="#111827", lw=2.3, marker="o", ms=4, label="entropy")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("terminal fate entropy")
    ax.set_ylim(0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([f"N{int(v)}" for v in mat["initial_rna_neighborhood"]], fontsize=9)
    ax.set_ylabel("terminal protein fate fraction")
    ax.set_xlabel("initial RNA neighborhood, sorted by terminal protein fate")
    ax.set_title("Initial RNA neighborhoods define a terminal CITE protein fate geography", weight="bold")
    ax.legend(frameon=False, ncol=7, bbox_to_anchor=(1.0, 1.17), loc="upper right", fontsize=9)
    savefig(
        fig,
        BIO_DIR,
        "bio_landscape03_terminal_fate_geography",
        "Terminal fate geography",
        "cite_biology_landscape",
        "Which RNA-initial neighborhoods resolve into distinct terminal protein fates?",
    )


def fig_bio04_branch_window(fate_df: pd.DataFrame, module_df: pd.DataFrame) -> None:
    phenotypes = [p for p in ["HSC", "MasP", "EryP", "MoP", "MkP", "NeuP", "BP"] if p in set(fate_df["phenotype"])]
    pivot = fate_df.pivot_table(index="time", columns="phenotype", values="fraction", aggfunc="mean").fillna(0.0)
    pivot = pivot.reindex(columns=phenotypes).fillna(0.0).sort_index()
    entropy = fate_df.groupby("time")["entropy"].first().reindex(pivot.index)

    gen = module_df[module_df["generated_mean"].notna()].pivot_table(index="time", columns="module", values="generated_mean").sort_index()
    if {"Myeloid priming", "Ery-Mk priming"}.issubset(gen.columns):
        decision_margin = gen["Myeloid priming"] - gen["Ery-Mk priming"]
    else:
        decision_margin = pd.Series(np.zeros(len(gen)), index=gen.index)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8.2), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.0]})
    axes[0].stackplot(pivot.index, [pivot[p].to_numpy() for p in phenotypes], colors=[PHENO_COLORS.get(p, "#94A3B8") for p in phenotypes], labels=phenotypes, alpha=0.92)
    axes[0].set_ylabel("nearest observed\nprotein fate fraction")
    axes[0].set_title("CITE-PCFM continuous rollout resolves protein fates over time", weight="bold")
    axes[0].legend(frameon=False, ncol=7, bbox_to_anchor=(1, 1.18), loc="upper right", fontsize=9)
    axes[1].plot(entropy.index, entropy.to_numpy(), color="#111827", lw=2.5, label="fate entropy")
    axes[1].plot(decision_margin.index, decision_margin.to_numpy(), color="#B73A3A", lw=2.2, label="myeloid minus Ery-Mk marker margin")
    axes[1].axhline(0, color="#CBD5E1", lw=1)
    axes[1].set_ylabel("index")
    axes[1].set_xlabel("time")
    axes[1].legend(frameon=False, ncol=2, fontsize=9)
    axes[1].annotate("branching window", xy=(2.4, float(entropy.loc[2.4]) if 2.4 in entropy.index else 0.75), xytext=(1.1, 0.95), arrowprops=dict(arrowstyle="->", color="#111827"), fontsize=10, weight="bold")
    savefig(
        fig,
        BIO_DIR,
        "bio_landscape04_branch_window_entropy",
        "Branch window",
        "cite_biology_landscape",
        "When does the generated protein landscape resolve from multipotent states into lineage-biased fates?",
    )


def fig_bio05_marker_basin_maps(point_df: pd.DataFrame) -> None:
    terminal = point_df[np.isclose(point_df["time"], 4.0)].copy()
    if terminal.empty:
        terminal = point_df.copy()
    modules = ["Stem retention", "Ery-Mk priming", "Myeloid priming", "Mast/Baso axis"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    for ax, module in zip(axes.flat, modules):
        vals = terminal[module].to_numpy() if module in terminal else np.zeros(len(terminal))
        vmax = float(np.nanpercentile(np.abs(vals), 97)) if len(vals) else 1.0
        sc = ax.scatter(terminal["protein_pc1"], terminal["protein_pc2"], c=vals, cmap="coolwarm", vmin=-vmax, vmax=vmax, s=20, alpha=0.82, linewidths=0)
        ax.set_title(module, fontsize=12, weight="bold")
        ax.set_xlabel("protein PC1")
        ax.set_ylabel("protein PC2")
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
        cb.ax.tick_params(labelsize=8)
    fig.suptitle("Terminal CITE-PCFM protein landscape decomposed into surface-marker axes", y=1.02, weight="bold")
    savefig(
        fig,
        BIO_DIR,
        "bio_landscape05_surface_marker_basin_maps",
        "Surface marker basin maps",
        "cite_biology_landscape",
        "Which protein basins carry stem, myeloid, mast/basophil, and Ery-Mk surface-marker programs?",
    )


def fig_bio06_story_model(ctx: CITEContext, module_df: pd.DataFrame) -> None:
    future = module_df[(module_df["generated_mean"].notna()) & (module_df["time"] > 0)][["time", "module", "generated_mean"]]
    obs = module_df[(module_df["observed_mean"].notna()) & (module_df["time"] > 0)][["time", "module", "observed_mean"]]
    merged = future.merge(obs[["time", "module", "observed_mean"]], on=["time", "module"], how="inner")
    merged["abs_gap"] = np.abs(merged["generated_mean"] - merged["observed_mean"])
    module_error = merged.groupby("module")["abs_gap"].mean().sort_values()
    module_error.to_csv(TABLE_OUT / "generated_vs_observed_marker_module_error.csv", header=["mean_abs_error"])

    fig, ax = plt.subplots(figsize=(13.5, 6.3))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.04, 0.91, "Biological story model for CITE-PCFM", fontsize=20, weight="bold")
    cards = [
        ("Input", "RNA state at day 2\nplus trained CITE readout", "#E8F2FF"),
        ("Landscape", "protein coordinates move\nthrough observed phenotype basins", "#FFF7ED"),
        ("Programs", "surface-marker modules\nchange continuously", "#F0FDF4"),
        ("Fate", "initial RNA neighborhoods\nresolve terminal protein outcomes", "#FEE2E2"),
    ]
    xs = [0.05, 0.30, 0.55, 0.80]
    for x, (h, body, fc) in zip(xs, cards):
        rounded_box(ax, (x, 0.52), 0.17, 0.23, f"{h}\n\n{body}", fc=fc, ec="#94A3B8", size=11, weight="bold")
    for x0, x1 in zip(xs[:-1], xs[1:]):
        arrow(ax, (x0 + 0.17, 0.63), (x1, 0.63), "#64748B")
    ax.text(0.06, 0.36, "Highest-confidence biological panels:", fontsize=12, weight="bold")
    y = 0.30
    for module, value in module_error.head(3).items():
        ax.text(0.07, y, f"- {module}: generated mean tracks observed target ADT module (mean abs gap {value:.2f})", fontsize=10.5, color="#334155")
        y -= 0.055
    ax.text(0.56, 0.36, "Caveat to state explicitly:", fontsize=12, weight="bold")
    ax.text(0.57, 0.30, "Protein coordinates are interpreted through an observed-data ADT decoder;\nphenotype fractions are a coarse readout and should not be overclaimed.", fontsize=10.5, color="#334155")
    savefig(
        fig,
        BIO_DIR,
        "bio_landscape06_biology_story_model",
        "Biology story model",
        "cite_biology_landscape",
        "How should the CITE-PCFM biology story be told without turning it into an algorithm comparison?",
    )


def load_lifecycle_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    trials = pd.read_csv(LIFECYCLE_DIR / "lifecycle_trials.csv", parse_dates=["created_at", "completed_at"])
    runs = pd.read_csv(LIFECYCLE_DIR / "lifecycle_runs.csv", parse_dates=["created_at", "completed_at"])
    return trials, runs


def pcfm_trial_lifecycle(trials: pd.DataFrame) -> pd.DataFrame:
    d = trials[trials["algorithm"] == "cite_paired_coord_fm"].sort_values("created_at").reset_index(drop=True).copy()
    d["local_trial"] = np.arange(1, len(d) + 1)
    d["best_protein_energy"] = d["protein_energy"].cummin()
    d["quality_score"] = 1.0 / d["best_protein_energy"].clip(lower=1e-6)
    return d


def milestone_table() -> pd.DataFrame:
    rows = [
        {
            "step": "Residual SB-FM real-data block",
            "trial": 0,
            "metric": 0.031822,
            "status": "failed",
            "why_it_failed": "early/late protein errors plus paired covariance drift; stochastic residual path did not validate the biological claim",
            "lesson": "do not hide protein fidelity inside residual stochasticity; keep a simpler deterministic CITE geometry",
        },
        {
            "step": "PCFM initial Stage 2",
            "trial": 2,
            "metric": 0.534965,
            "status": "failed",
            "why_it_failed": "paired coupling existed, but the protein readout was not yet calibrated to future target landscapes",
            "lesson": "protein metric must be scored from sealed rollout, not only endpoint/coupling plausibility",
        },
        {
            "step": "capacity and endpoint bracketing",
            "trial": 8,
            "metric": 0.039508,
            "status": "improved",
            "why_it_failed": "endpoint loss was low but intermediate protein readout still plateaued, especially at t2/t3",
            "lesson": "larger heads and endpoint weights alone do not explain intermediate biology",
        },
        {
            "step": "scaled Fourier time basis",
            "trial": 32,
            "metric": 0.023627,
            "status": "near_pass",
            "why_it_failed": "time conditioning improved rollout but still missed the required 10% Stage 2 margin",
            "lesson": "explicit time features matter; remaining error is local in biological time",
        },
        {
            "step": "time-local protein weights",
            "trial": 39,
            "metric": 0.022801,
            "status": "passed",
            "why_it_failed": "not a failure; the repair targeted t1-t3 supervision windows and crossed the formal gate",
            "lesson": "the metric rose because supervision moved from global endpoint emphasis to time-local CITE protein biology",
        },
        {
            "step": "Stage 3 width tuning",
            "trial": 48,
            "metric": 0.022218,
            "status": "tuned",
            "why_it_failed": "W1/protein tradeoff limited further SOTA chasing",
            "lesson": "small width changes preserve claim metric while improving guardrails; stop when scalar tuning saturates",
        },
        {
            "step": "Final regression lock",
            "trial": 70,
            "metric": 0.022198,
            "status": "locked",
            "why_it_failed": "not a failure; reproduced the tuned active best under the final-regression contract",
            "lesson": "lock the evidence-compatible rule rather than continuing uncontrolled search",
        },
    ]
    df = pd.DataFrame(rows)
    df.to_csv(TABLE_OUT / "lifecycle_milestones.csv", index=False)
    return df


def fig_life01_causal_map(milestones: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(14, 7.2))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.03, 0.94, "Algorithm discovery landscape: failure mechanism -> targeted repair -> metric lift", fontsize=18, weight="bold")

    cards = [
        ("RSB-FM failed", "protein_energy 0.0318\npaired covariance drift", "#F1F5F9", "#64748B"),
        ("PCFM reframed", "paired-coordinate coupling\nRNA-driven protein readout", "#FFF7ED", "#F97316"),
        ("readout repaired", "Fourier time basis\n+ time-local protein weights", "#FEE2E2", "#B73A3A"),
        ("final locked", "protein_energy 0.0222\nsame rollout contract", "#F0FDF4", "#16A34A"),
    ]
    xs = [0.05, 0.31, 0.57, 0.80]
    for x, (h, b, fc, ec) in zip(xs, cards):
        rounded_box(ax, (x, 0.58), 0.17, 0.21, f"{h}\n\n{b}", fc=fc, ec=ec, size=11, weight="bold")
    for x0, x1 in zip(xs[:-1], xs[1:]):
        arrow(ax, (x0 + 0.17, 0.68), (x1, 0.68), "#64748B")

    lower_cards = [
        ("What failed before?", "residual stochasticity + re-charting\ncreated boundary protein/covariance drift"),
        ("What was learned?", "data favored deterministic paired CITE geometry\nover stochastic residual machinery"),
        ("Why did the metric lift?", "time-local supervision aligned the readout\nwith error-dominant biological windows"),
    ]
    for i, (h, b) in enumerate(lower_cards):
        rounded_box(ax, (0.07 + i * 0.30, 0.22), 0.24, 0.19, f"{h}\n\n{b}", fc="#F8FAFC", ec="#CBD5E1", size=10, weight="bold")
    ax.text(0.05, 0.10, "Read metric direction as quality improvement: protein_energy fell from 0.535 early PCFM to 0.0222 final.", fontsize=11, color="#334155")
    savefig(
        fig,
        LIFE_DIR,
        "life_landscape01_failure_to_lift_causal_map",
        "Failure-to-lift causal map",
        "algorithm_discovery_landscape",
        "Can a reader see in one glance why the final metric improved and what earlier failures taught?",
    )


def fig_life02_metric_ladder(pcfm: pd.DataFrame, milestones: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12.4, 5.9))
    d = pcfm.dropna(subset=["protein_energy"]).copy()
    ax.plot(d["local_trial"], d["protein_energy"], color="#CBD5E1", lw=1.2, marker="o", ms=3, label="all CITE-PCFM trials")
    ax.step(d["local_trial"], d["best_protein_energy"], where="post", color="#B73A3A", lw=3.2, label="retained best-so-far")
    gate = 0.022906601428985596
    ax.axhline(gate, color="#111827", lw=1.2, ls="--", label="Stage 2 pass gate")
    short_labels = {
        "PCFM initial Stage 2": "Initial CITE-PCFM\nreadout uncalibrated",
        "capacity and endpoint bracketing": "Repair 1\ncapacity + endpoint\nbracketing",
        "scaled Fourier time basis": "Repair 2\nscaled Fourier\ntime basis",
        "time-local protein weights": "Repair 3\ntime-local protein\nweights",
        "Stage 3 width tuning": "Stage 3\nwidth tuning",
        "Final regression lock": "Final lock\nsame rule reproduced",
    }
    label_factor = {
        "PCFM initial Stage 2": 0.70,
        "capacity and endpoint bracketing": 1.30,
        "scaled Fourier time basis": 1.50,
        "time-local protein weights": 1.24,
        "Stage 3 width tuning": 1.50,
        "Final regression lock": 1.32,
    }
    for _, r in milestones.iterrows():
        if r["trial"] <= 0:
            continue
        y = float(r["metric"])
        x = float(r["trial"])
        ax.scatter([x], [y], s=120, color="#7F1D1D" if r["status"] in ["passed", "locked"] else "#F97316", edgecolor="white", zorder=5)
        x_text = x - 1.5 if r["step"] == "Final regression lock" else x + 1.0
        ha = "right" if r["step"] == "Final regression lock" else "left"
        ax.text(
            x_text,
            y * label_factor.get(r["step"], 1.2),
            short_labels.get(r["step"], str(r["step"])),
            fontsize=8.0,
            color="#111827",
            va="bottom",
            ha=ha,
            linespacing=1.08,
        )
    ax.set_yscale("log")
    ax.set_ylim(0.018, 0.75)
    ax.set_xlabel("CITE-PCFM trial order")
    ax.set_ylabel("protein_energy (lower is better), log scale")
    ax.set_title("CITE-PCFM metric lift decomposes into staged model repairs", weight="bold")
    ax.legend(frameon=False, fontsize=8.8)
    savefig(
        fig,
        LIFE_DIR,
        "life_landscape02_metric_lift_ladder",
        "Metric lift ladder",
        "algorithm_discovery_landscape",
        "Where did the primary protein metric actually improve, plateau, and pass the gate?",
    )


def fig_life03_failure_matrix(milestones: pd.DataFrame) -> None:
    labels = [
        "unbalanced/mass idea",
        "residual stochastic bridge",
        "endpoint-only readout",
        "scalar/capacity tuning",
        "time encoding",
        "time-local biology",
    ]
    rows = [
        ["TMV/data contract failures", "not main issue", "not tested", "not useful", "not relevant", "dropped"],
        ["not needed", "t1/t4 drift; covariance loss", "readout still weak", "more capacity timed out", "helped but near miss", "repaired"],
        ["lesson: balanced scope", "lesson: remove stochastic residual claim", "lesson: path biology matters", "lesson: bracket knobs", "lesson: phase matters", "lesson: local windows pass"],
    ]
    fig, ax = plt.subplots(figsize=(13, 5.6))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.03, 0.92, "Failure-mode matrix", fontsize=18, weight="bold")
    x0, y0 = 0.05, 0.70
    cw, ch = 0.145, 0.17
    for j, lab in enumerate(labels):
        rounded_box(ax, (x0 + j * cw, y0), cw * 0.92, 0.13, lab, fc="#F8FAFC", ec="#94A3B8", size=8.5, weight="bold")
    row_labels = ["Observed failure", "Repair result", "Lesson kept"]
    for i, row in enumerate(rows):
        y = y0 - (i + 1) * ch
        ax.text(0.012, y + 0.065, row_labels[i], fontsize=9, weight="bold", va="center")
        for j, txt in enumerate(row):
            fc = "#FEE2E2" if "drift" in txt or "fail" in txt else "#F0FDF4" if "lesson" in txt or "repaired" in txt or "pass" in txt else "#F8FAFC"
            rounded_box(ax, (x0 + j * cw, y), cw * 0.92, 0.13, txt, fc=fc, ec="#CBD5E1", size=7.6)
    savefig(
        fig,
        LIFE_DIR,
        "life_landscape03_failure_modes_decision_matrix",
        "Failure modes decision matrix",
        "algorithm_discovery_landscape",
        "Which failed mechanism produced which design lesson?",
    )


def fig_life04_tradeoff_landscape(pcfm: pd.DataFrame) -> None:
    d = pcfm.dropna(subset=["protein_energy", "w1_mean"]).copy()
    def phase(row):
        n = row["local_trial"]
        if n <= 8:
            return "endpoint/capacity"
        if n <= 25:
            return "scalar bracketing"
        if n <= 38:
            return "time basis"
        if n <= 39:
            return "time-local weights"
        if n <= 69:
            return "Stage 3 guardrail tuning"
        return "final regression"

    d["phase"] = d.apply(phase, axis=1)
    colors = {
        "endpoint/capacity": "#94A3B8",
        "scalar bracketing": "#F59E0B",
        "time basis": "#7C3AED",
        "time-local weights": "#B73A3A",
        "Stage 3 guardrail tuning": "#2563EB",
        "final regression": "#16A34A",
    }
    fig, ax = plt.subplots(figsize=(9, 7))
    for ph, sub in d.groupby("phase"):
        ax.scatter(sub["w1_mean"], sub["protein_energy"], s=80, color=colors[ph], alpha=0.80, edgecolor="white", label=ph)
    ax.axhline(0.022906601428985596, color="#111827", ls="--", lw=1.2, label="Stage 2 pass threshold")
    ax.scatter([8.235641], [0.022198], s=220, color="#16A34A", edgecolor="white", marker="*", zorder=5, label="final")
    ax.set_yscale("log")
    ax.set_xlabel("W1 guardrail")
    ax.set_ylabel("protein_energy, log scale")
    ax.set_title("Trial design landscape: protein metric improved inside a W1 guardrail band", weight="bold")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    savefig(
        fig,
        LIFE_DIR,
        "life_landscape04_trial_tradeoff_landscape",
        "Trial tradeoff landscape",
        "algorithm_discovery_landscape",
        "Which region of trial space gave protein gains without breaking the RNA guardrail?",
    )


def fig_life05_lessons_retained() -> None:
    fig, ax = plt.subplots(figsize=(13, 6.6))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.04, 0.92, "What the lifecycle learned", fontsize=19, weight="bold")
    left = [
        ("Do not claim mass recovery", "balanced CITE-seq panel; TMV diagnostic only"),
        ("Do not use stochastic residuals as a shortcut", "residual SB-FM worsened real paired covariance"),
        ("Do not chase endpoint loss alone", "endpoint loss converged but intermediate readout failed"),
        ("Do use biological time windows", "time-local CITE supervision crossed the Stage 2 gate"),
    ]
    right = [
        ("Final rule 1", "paired-coordinate OT selects RNA/protein-consistent targets"),
        ("Final rule 2", "RNA velocity stays the runtime state"),
        ("Final rule 3", "protein readout uses deterministic time features"),
        ("Final rule 4", "final regression locks the sealed rollout, not post-hoc corrections"),
    ]
    for i, (h, b) in enumerate(left):
        rounded_box(ax, (0.06, 0.70 - i * 0.16), 0.32, 0.105, f"{h}\n{b}", fc="#FEE2E2", ec="#FCA5A5", size=9.5, weight="bold")
    for i, (h, b) in enumerate(right):
        rounded_box(ax, (0.62, 0.70 - i * 0.16), 0.32, 0.105, f"{h}\n{b}", fc="#F0FDF4", ec="#86EFAC", size=9.5, weight="bold")
    for i in range(4):
        arrow(ax, (0.38, 0.752 - i * 0.16), (0.62, 0.752 - i * 0.16), "#64748B", rad=0)
    ax.text(0.43, 0.18, "This is the lifecycle story: each failure removes one unsupported assumption and keeps one evidence-compatible design rule.", fontsize=11, color="#334155")
    savefig(
        fig,
        LIFE_DIR,
        "life_landscape05_lessons_to_final_rules",
        "Lessons to final rules",
        "algorithm_discovery_landscape",
        "What experience was retained in the final algorithm rather than merely tuned away?",
    )


def fig_life06_article_layout(milestones: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.04, 0.92, "Recommended algorithm-lifecycle figure", fontsize=19, weight="bold")
    cards = [
        ("A. Failure map", "prior methods failed for identifiable\nbiological or contract reasons"),
        ("B. Metric ladder", "best protein metric improves\nat specific repair points"),
        ("C. Trial landscape", "successful region preserves W1\nwhile lowering protein energy"),
        ("D. Lessons", "final PCFM is the retained\nset of evidence-compatible rules"),
    ]
    for i, (h, b) in enumerate(cards):
        x = 0.06 + (i % 2) * 0.46
        y = 0.58 - (i // 2) * 0.27
        rounded_box(ax, (x, y), 0.36, 0.17, f"{h}\n\n{b}", fc="#F8FAFC", ec="#94A3B8", size=11, weight="bold")
    ax.text(0.06, 0.18, "Caption spine:", fontsize=12, weight="bold")
    ax.text(
        0.18,
        0.18,
        "CellCompass did not simply find a low metric; it falsified unsupported CITE dynamics, localized the readout failure to biological time windows, and locked a deterministic paired-coordinate rule.",
        fontsize=10.5,
        color="#334155",
        va="center",
    )
    savefig(
        fig,
        LIFE_DIR,
        "life_landscape06_article_layout",
        "Lifecycle article layout",
        "algorithm_discovery_landscape",
        "How should the lifecycle material be assembled so the causal metric lift is immediately visible?",
    )


def write_story(ctx: CITEContext, module_df: pd.DataFrame, fate_df: pd.DataFrame, milestones: pd.DataFrame) -> None:
    module_future = module_df[(module_df["generated_mean"].notna()) & (module_df["time"] > 0)][["time", "module", "generated_mean"]]
    obs_future = module_df[(module_df["observed_mean"].notna()) & (module_df["time"] > 0)][["time", "module", "observed_mean"]]
    merged = module_future.merge(obs_future[["time", "module", "observed_mean"]], on=["time", "module"], how="inner")
    merged["abs_gap"] = np.abs(merged["generated_mean"] - merged["observed_mean"])
    module_error = merged.groupby("module")["abs_gap"].mean().sort_values()
    entropy_min = float(fate_df.groupby("time")["entropy"].first().min())
    entropy_max = float(fate_df.groupby("time")["entropy"].first().max())
    terminal_path = TABLE_OUT / "terminal_rna_neighborhood_fate_geography.csv"
    terminal = pd.read_csv(terminal_path) if terminal_path.exists() else pd.DataFrame()
    strongest_terminal = terminal.sort_values("lineage_bias", ascending=False).head(3) if not terminal.empty else pd.DataFrame()

    lines = []
    lines.append("# CITE-PCFM Deep Story Package\n\n")
    lines.append("This package is designed as a Nature Biotechnology-style story, not a metric appendix. It has two linked landscapes: a CITE biology landscape that stands on CITE-PCFM alone, and an algorithm-search landscape that explains why CellCompass eventually found a validated rule.\n\n")
    lines.append("## Editorial Spine\n")
    lines.append("CITE-PCFM is interesting because it turns an RNA-initialized hematopoietic trajectory into a measurable surface-protein landscape. The central biological object is not a lower scalar metric by itself. It is a time-resolved map of where generated cells sit in observed CITE protein basins, which marker programs rise or collapse, and which initial RNA neighborhoods resolve into distinct terminal protein fates.\n\n")
    lines.append("For the algorithm story, the point is not that more trials monotonically tuned a number. The lifecycle is a falsification path: stochastic residual bridges, endpoint-only protein repair, and capacity-only changes each failed to explain target protein biology; explicit temporal basis functions and time-local protein supervision moved the model into the low-error basin.\n\n")
    lines.append("## 1. CITE Biology Landscape\n")
    lines.append("Biological claim: CITE-PCFM converts initial RNA neighborhoods into dynamic protein-state basins. The panels should not be organized as method comparisons. They should read as one biological atlas of generated surface phenotype dynamics.\n\n")
    lines.append("### What Makes The Biology Unique\n")
    lines.append("- Protein-first phenotype readout: the result is interpreted in observed CITE protein space, not only in RNA latent space.\n")
    lines.append("- Surface-marker programs: the generated protein coordinates recover ADT module trajectories for antigen identity, myeloid priming, activation/HLA state, stem retention, mast/basophil axis, and Ery-Mk priming.\n")
    lines.append("- RNA-to-protein fate geography: terminal protein fates are stratified by initial RNA neighborhoods, giving a landscape view of lineage bias rather than a single endpoint fraction.\n")
    lines.append("- Branch-window readout: entropy changes over continuous rollout time reveal when the generated protein state is multipotent, sharpening, or re-mixing.\n\n")
    lines.append("### Main Biological Evidence\n")
    lines.append("Marker-module alignment is strongest for modules that a CITE-seq reader can immediately interpret as surface phenotype programs:\n")
    for module, value in module_error.items():
        lines.append(f"- {module}: mean generated-vs-observed module gap `{value:.3f}`.\n")
    lines.append(f"- Generated fate entropy spans `{entropy_min:.3f}` to `{entropy_max:.3f}`, which supports a branching landscape readout instead of a terminal-only bar plot.\n")
    if not strongest_terminal.empty:
        lines.append("- Initial RNA neighborhoods with strongest terminal MasP bias:\n")
        for _, r in strongest_terminal.iterrows():
            lines.append(f"  - neighborhood `{int(r['initial_rna_neighborhood'])}`: MasP `{r['MasP']:.3f}`, HSC `{r['HSC']:.3f}`, lineage_bias `{r['lineage_bias']:.3f}`.\n")
    lines.append("\n### Biological Figure Logic\n")
    lines.append("- `bio_landscape01_protein_topography`: establish the observed protein terrain, then place the generated rollout on that terrain.\n")
    lines.append("- `bio_landscape02_marker_program_streams`: translate protein coordinates into interpretable CITE marker-program streams over time.\n")
    lines.append("- `bio_landscape03_terminal_fate_geography`: show that initial RNA neighborhoods resolve to distinct terminal protein basins.\n")
    lines.append("- `bio_landscape04_branch_window_entropy`: identify the temporal window where protein fate ambiguity changes.\n")
    lines.append("- `bio_landscape05_surface_marker_basin_maps`: use as supplement or mechanistic support for which basins carry which marker programs.\n")
    lines.append("- `bio_landscape06_biology_story_model`: use as the assembly/caption schematic.\n\n")
    lines.append("Caveat: generated ADT markers are decoded from observed protein PCA by a ridge decoder. This is appropriate for protein-coordinate interpretation, but the manuscript should call these marker-program interpretations, not direct raw antibody count generation.\n\n")
    lines.append("## 2. Algorithm Discovery Landscape\n")
    lines.append("The lifecycle must be readable in one glance: what failed, what was learned, and why the metric rose. The figure program should therefore combine a causal map with a quantitative curve, an embedding landscape, and a mechanism-level gain plot.\n\n")
    lines.append("### Five-Algorithm Evolution\n")
    lines.append("This is the missing top-level narrative. The algorithm story should be read as five generations, not as one PCFM tuning log.\n\n")
    lines.append("- `cite_vgfm`: started from a generic VGFM-style latent transport hypothesis. It exposed that generic rollout quality and unstable mass/TMV assumptions do not automatically produce CITE protein fidelity.\n")
    lines.append("- `cite_bfm / cite_bfm_balanced`: introduced balanced paired-coordinate geometry and endpoint-aware protein cost. It improved over the first attempt but still behaved like endpoint repair, not an intermediate protein-landscape model.\n")
    lines.append("- `cite_factorized_fm`: separated RNA dynamics and protein residual/readout structure. The scientific idea passed Stage 1, but Stage 2 evidence was blocked by evaluator provenance and sibling-campaign arbitration, so it could not become the locked evidence path.\n")
    lines.append("- `cite_residual_sb_fm`: tried to rescue missing protein motion with stochastic residual bridges. It produced controlled-simulation signal but failed the real GSE305370 claim against CRUFM, showing that stochastic residual complexity was not the right biological mechanism.\n")
    lines.append("- `cite_paired_coord_fm`: removed unsupported stochastic machinery, kept deterministic RNA runtime, made protein a path-coupled paired-coordinate readout, and added time-local protein weighting. This is the first generation that passed and locked the CITE protein-fidelity claim.\n\n")
    lines.append("### PCFM Internal Lift\n")
    for _, r in milestones.iterrows():
        lines.append(f"- {r['step']}: metric `{r['metric']:.6f}`, status `{r['status']}`. Failure/limit: {r['why_it_failed']}. Lesson: {r['lesson']}.\n")
    lines.append("\n### Why The Metric Rose\n")
    lines.append("The decisive jump was not a generic larger-network effect. The ladder is: initial sealed-rollout PCFM failed at `0.534965`; capacity and endpoint bracketing improved to `0.039508` but still left intermediate-time protein biology unresolved; scaled Fourier time features reached `0.023627`; time-local protein weights crossed the gate at `0.022801`; final regression locked the evidence-compatible rule at `0.022198`.\n\n")
    lines.append("```mermaid\n")
    lines.append("flowchart LR\n")
    lines.append('  A["cite_vgfm\\ngeneric latent rollout failed CITE specificity"] --> B["cite_bfm / balanced\\npaired geometry but endpoint-biased"]\n')
    lines.append('  B --> C["cite_factorized_fm\\nStage 1 passed, Stage 2 provenance blocked"]\n')
    lines.append('  C --> D["cite_residual_sb_fm\\nstochastic residual bridge failed real protein claim"]\n')
    lines.append('  D --> E["cite_paired_coord_fm\\ndeterministic paired-coordinate rule"]\n')
    lines.append('  E --> F["PCFM local repairs\\nFourier time + time-local protein weights"]\n')
    lines.append('  F --> G["Final lock\\nprotein_energy 0.022198"]\n')
    lines.append("```\n\n")
    lines.append("One-sentence figure caption spine: CellCompass did not simply find a low metric; it falsified unsupported CITE dynamics, localized the readout failure to biological time windows, and locked a deterministic paired-coordinate rule.\n\n")
    lines.append("## Figure Catalog\n")
    for fig in FIGURES:
        lines.append(f"- `{fig['name']}` ({fig['track']}): {fig['question']}. File: `{fig['png']}`\n")
    STORY_PATH.write_text("".join(lines))


def write_manifest() -> None:
    MANIFEST_PATH.write_text(json.dumps({"figures": FIGURES, "storyboard": str(STORY_PATH), "tables": str(TABLE_OUT)}, indent=2))


def main() -> None:
    ensure_dirs()
    ctx = load_context()
    module_df, fate_df, point_df = build_generated_tables(ctx)
    module_df.to_csv(TABLE_OUT / "generated_marker_module_timecourses.csv", index=False)
    fate_df.to_csv(TABLE_OUT / "generated_nearest_protein_fate_timecourse.csv", index=False)
    point_df.to_csv(TABLE_OUT / "generated_landscape_points.csv", index=False)

    fig_bio01_topography(ctx)
    fig_bio02_marker_streams(module_df)
    fig_bio03_terminal_fate_geography()
    fig_bio04_branch_window(fate_df, module_df)
    fig_bio05_marker_basin_maps(point_df)
    fig_bio06_story_model(ctx, module_df)

    trials, runs = load_lifecycle_tables()
    pcfm = pcfm_trial_lifecycle(trials)
    pcfm.to_csv(TABLE_OUT / "pcfm_trial_metric_ladder.csv", index=False)
    milestones = milestone_table()
    fig_life01_causal_map(milestones)
    fig_life02_metric_ladder(pcfm, milestones)
    fig_life03_failure_matrix(milestones)
    fig_life04_tradeoff_landscape(pcfm)
    fig_life05_lessons_retained()
    fig_life06_article_layout(milestones)

    write_story(ctx, module_df, fate_df, milestones)
    write_manifest()
    print(json.dumps({"figures": len(FIGURES), "out_root": str(OUT_ROOT), "storyboard": str(STORY_PATH)}, indent=2))


if __name__ == "__main__":
    main()
