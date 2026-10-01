#!/usr/bin/env python3
"""State-matched fate divergence analysis for the clone/BMF case.

The figure asks a focused biological question: can cells that occupy a very
similar early RNA state still produce different barcode-derived future fates?
It uses LARRY clone fate composition as the observed future and BMF branch
probabilities as the model readout for the same day-2 source cells.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cellcompass_mpl_cache")
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import anndata as ad
import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.neighbors import NearestNeighbors


ROOT = Path(__file__).resolve().parents[2]
TABLE_DIR = ROOT / "downstream/cellcompass_clone_case"
FIG_DIR = ROOT / "figures/cellcompass_clone_case"
ADATA_PATH = Path("/Users/zhenyizhang/.cellcompass/algorithm_benchmarks/datasets/weinreb_rawrebuild_full_k10mindiff1/data.h5ad")
BMF_TABLE = ROOT / "downstream/clone_fate_spring_final/clone_cuvgf_bmf_final_day2_fate_probabilities.csv"

FATE_PALETTE = {
    "Undifferentiated": "#2E8B57",
    "Neutrophil": "#19B7C7",
    "Monocyte": "#8C564B",
    "Baso": "#7E57C2",
    "Mast": "#D95F9F",
    "Meg": "#2563EB",
    "Erythroid": "#F28E2B",
    "Eos": "#D62728",
    "Lymphoid": "#8FBCE6",
    "Ccr7_DC": "#B8B34A",
    "pDC": "#6B7280",
    "Other": "#CBD5E1",
}
MAIN_FATES = ["Neutrophil", "Monocyte", "Baso", "Undifferentiated", "Mast", "Meg", "Erythroid", "Eos"]
ENTROPY_CMAP = LinearSegmentedColormap.from_list("future_entropy", ["#E5E7EB", "#93C5FD", "#7E57C2", "#DC2626"])
RNA_PROGRAMS = {
    "Stem": ["Meis1", "Mecom", "Procr", "Mpl", "Kit"],
    "Neutrophil": ["Cebpe", "Lyz2", "S100a8", "S100a9"],
    "Monocyte": ["Irf8", "Csf1r", "Lyz2", "Ly6c2"],
    "Baso/Mast": ["Gata2", "Ms4a2", "Cpa3", "Prss34", "Mcpt8"],
    "Meg/Ery": ["Pf4", "Ppbp", "Itga2b", "Car1", "Car2"],
}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial"],
            "font.size": 8,
            "axes.linewidth": 1.0,
            "axes.edgecolor": "#111827",
            "axes.titlesize": 8.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
        }
    )


def load_obs(adata: ad.AnnData) -> pd.DataFrame:
    obs = adata.obs[["time_point_processed", "Cell type annotation", "SPRING-x", "SPRING-y", "clone"]].copy()
    obs["t"] = obs["time_point_processed"].astype(int)
    obs["cell_type"] = obs["Cell type annotation"].astype(str)
    return obs.rename(columns={"SPRING-x": "spring_x", "SPRING-y": "spring_y"})


def normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(np.asarray(x, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    s = x.sum(axis=1, keepdims=True)
    return np.divide(x, s, out=np.zeros_like(x), where=s > 0)


def entropy(prob: np.ndarray) -> np.ndarray:
    p = np.clip(prob, 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=1) / np.log(prob.shape[1])


def js_divergence(u: np.ndarray, v: np.ndarray) -> float:
    u = np.clip(u, 1e-12, 1.0)
    v = np.clip(v, 1e-12, 1.0)
    m = 0.5 * (u + v)
    return float(0.5 * np.sum(u * np.log(u / m)) + 0.5 * np.sum(v * np.log(v / m)))


def avg_pairwise_js(mat: np.ndarray) -> float:
    vals = []
    for i in range(len(mat)):
        for j in range(i + 1, len(mat)):
            vals.append(js_divergence(mat[i], mat[j]))
    return float(np.mean(vals)) if vals else 0.0


def style_spring(ax: plt.Axes, obs: pd.DataFrame) -> None:
    x = obs["spring_x"].to_numpy(float)
    y = obs["spring_y"].to_numpy(float)
    padx = 0.035 * (np.nanmax(x) - np.nanmin(x))
    pady = 0.035 * (np.nanmax(y) - np.nanmin(y))
    ax.set_xlim(float(np.nanmin(x) - padx), float(np.nanmax(x) + padx))
    ax.set_ylim(float(np.nanmin(y) - pady), float(np.nanmax(y) + pady))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_observed_background(ax: plt.Axes, obs: pd.DataFrame, alpha: float = 0.06) -> None:
    for fate, color in FATE_PALETTE.items():
        if fate == "Other":
            continue
        d = obs[obs["cell_type"].eq(fate)]
        if d.empty:
            continue
        ax.scatter(d["spring_x"], d["spring_y"], s=0.7, c=color, alpha=alpha, linewidth=0, rasterized=True, zorder=0)


def add_fate_labels(ax: plt.Axes, obs: pd.DataFrame) -> None:
    terminal = obs[obs["t"].eq(2)]
    offsets = {
        "Undifferentiated": (-20, 20),
        "Neutrophil": (0, -35),
        "Monocyte": (20, 20),
        "Baso": (-25, -15),
        "Mast": (-15, -25),
        "Meg": (-25, 20),
        "Erythroid": (20, -15),
        "Eos": (0, 10),
        "Lymphoid": (0, 10),
        "Ccr7_DC": (0, 10),
        "pDC": (0, -10),
    }
    for fate, color in FATE_PALETTE.items():
        d = terminal[terminal["cell_type"].eq(fate)]
        if len(d) < 20:
            continue
        dx, dy = offsets.get(fate, (0, 0))
        ax.text(
            float(d["spring_x"].median()) + dx,
            float(d["spring_y"].median()) + dy,
            fate.replace("Undifferentiated", "Undiff."),
            color=color,
            fontsize=7,
            weight="bold",
            ha="center",
            va="center",
            path_effects=[pe.Stroke(linewidth=2.0, foreground="white", alpha=0.95), pe.Normal()],
            zorder=80,
        )


def select_state_matched_pocket(
    valid: pd.DataFrame,
    latent: np.ndarray,
    truth: np.ndarray,
    pred: np.ndarray,
    fates: list[str],
    k: int = 18,
) -> tuple[int, np.ndarray, pd.DataFrame]:
    nn = NearestNeighbors(n_neighbors=min(k, len(valid)), n_jobs=-1).fit(latent)
    dists, neighbors = nn.kneighbors(latent)
    truth_arg = np.asarray(fates, dtype=object)[truth.argmax(axis=1)]
    pred_arg = np.asarray(fates, dtype=object)[pred.argmax(axis=1)]
    xy = valid[["spring_x", "spring_y"]].to_numpy(float)

    rows = []
    for i in range(len(valid)):
        ids = neighbors[i]
        truth_div = avg_pairwise_js(truth[ids])
        pred_div = avg_pairwise_js(pred[ids])
        n_truth = len(set(truth_arg[ids]))
        n_pred = len(set(pred_arg[ids]))
        fit = float(np.mean([1.0 - js_divergence(truth[j], pred[j]) for j in ids]))
        rel = float(valid.iloc[ids]["truth_reliability"].mean())
        med_lat = float(np.median(dists[i, 1:])) if len(ids) > 1 else 0.0
        med_spring = float(np.median(np.linalg.norm(xy[ids] - xy[i], axis=1)))
        score = truth_div * np.sqrt(max(pred_div, 1e-6)) * np.log1p(min(n_truth, n_pred)) * rel * fit
        score /= max(med_lat, 1e-6) * (1.0 + med_spring / 350.0)
        rows.append(
            {
                "valid_row": i,
                "source_index": int(valid.iloc[i]["source_index"]),
                "score": score,
                "truth_pairwise_js": truth_div,
                "pred_pairwise_js": pred_div,
                "n_truth_dominant_fates": n_truth,
                "n_pred_dominant_fates": n_pred,
                "mean_truth_pred_match": fit,
                "mean_truth_reliability": rel,
                "median_latent_distance": med_lat,
                "median_spring_distance": med_spring,
            }
        )
    score_df = pd.DataFrame(rows).sort_values("score", ascending=False)
    center = int(score_df.iloc[0]["valid_row"])
    return center, neighbors[center], score_df


def compact_prob_table(frame: pd.DataFrame, fates: list[str], prefix: str) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    for fate in MAIN_FATES:
        col = f"{prefix}_{fate}"
        if col in frame:
            out[fate] = frame[col].fillna(0.0).to_numpy(float)
        else:
            out[fate] = 0.0
    used_cols = [f"{prefix}_{f}" for f in MAIN_FATES if f"{prefix}_{f}" in frame]
    all_cols = [f"{prefix}_{f}" for f in fates]
    other_cols = [c for c in all_cols if c not in used_cols and c in frame]
    out["Other"] = frame[other_cols].fillna(0.0).sum(axis=1).to_numpy(float) if other_cols else 0.0
    arr = normalize_rows(out.to_numpy(float))
    return pd.DataFrame(arr, columns=out.columns, index=frame.index)


def expression_block(adata: ad.AnnData, obs_index: np.ndarray, gene_index: list[int]) -> np.ndarray:
    block = adata.X[np.asarray(obs_index, dtype=int), :][:, np.asarray(gene_index, dtype=int)]
    if sparse.issparse(block):
        block = block.toarray()
    return np.asarray(block, dtype=float)


def direct_rna_program_scores(
    adata: ad.AnnData,
    query_index: np.ndarray,
    background_index: np.ndarray,
    programs: dict[str, list[str]],
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    var_to_idx = {str(g): i for i, g in enumerate(adata.var_names)}
    used = {name: [g for g in genes if g in var_to_idx] for name, genes in programs.items()}
    used = {name: genes for name, genes in used.items() if genes}
    all_genes = sorted({g for genes in used.values() for g in genes})
    gene_index = [var_to_idx[g] for g in all_genes]

    bg = expression_block(adata, background_index, gene_index)
    mean = bg.mean(axis=0)
    std = bg.std(axis=0)
    std[std < 1e-6] = 1.0
    q = (expression_block(adata, query_index, gene_index) - mean[None, :]) / std[None, :]
    gene_pos = {g: i for i, g in enumerate(all_genes)}
    scores = {}
    for name, genes in used.items():
        scores[name] = q[:, [gene_pos[g] for g in genes]].mean(axis=1)
    return pd.DataFrame(scores), used


def plot_stacked_bars(ax: plt.Axes, prob: pd.DataFrame, labels: list[str], title: str) -> None:
    y = np.arange(len(prob))
    left = np.zeros(len(prob))
    for fate in prob.columns:
        vals = prob[fate].to_numpy(float)
        ax.barh(y, vals, left=left, color=FATE_PALETTE.get(fate, "#CBD5E1"), edgecolor="white", linewidth=0.45, height=0.74)
        left += vals
    ax.set_xlim(0, 1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlabel("future fate probability")
    ax.set_title(title, loc="left", pad=5)
    ax.grid(axis="x", color="#E5E7EB", lw=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_gene_program_companion(adata: ad.AnnData, obs: pd.DataFrame, valid: pd.DataFrame, pocket: pd.DataFrame) -> None:
    valid_scores, _ = direct_rna_program_scores(
        adata,
        valid["source_index"].to_numpy(int),
        valid["source_index"].to_numpy(int),
        RNA_PROGRAMS,
    )
    _, used = direct_rna_program_scores(
        adata,
        valid["source_index"].to_numpy(int)[:1],
        valid["source_index"].to_numpy(int),
        RNA_PROGRAMS,
    )
    pd.DataFrame({"program": list(used), "genes_used": [",".join(genes) for genes in used.values()]}).to_csv(
        TABLE_DIR / "clone_state_matched_fate_divergence_rna_programs_used.csv", index=False
    )

    def corr(x: np.ndarray, y: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        if int(mask.sum()) < 4:
            return float("nan")
        x = x[mask] - np.mean(x[mask])
        y = y[mask] - np.mean(y[mask])
        denom = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
        return float(np.sum(x * y) / denom) if denom > 1e-12 else float("nan")

    targets = {
        "Stem": ["Undifferentiated"],
        "Neutrophil": ["Neutrophil"],
        "Monocyte": ["Monocyte"],
        "Baso/Mast": ["Baso", "Mast"],
        "Meg/Ery": ["Meg", "Erythroid"],
    }
    program_colors = {
        "Stem": FATE_PALETTE["Undifferentiated"],
        "Neutrophil": FATE_PALETTE["Neutrophil"],
        "Monocyte": FATE_PALETTE["Monocyte"],
        "Baso/Mast": FATE_PALETTE["Baso"],
        "Meg/Ery": FATE_PALETTE["Erythroid"],
    }
    program_labels = {
        "Stem": "Stem",
        "Neutrophil": "Neutrophil",
        "Monocyte": "Monocyte",
        "Baso/Mast": "Baso/Mast",
        "Meg/Ery": "Meg/Ery",
    }

    def target_probability(frame: pd.DataFrame, prefix: str, fates_: list[str]) -> np.ndarray:
        cols = [f"{prefix}_{f}" for f in fates_ if f"{prefix}_{f}" in frame]
        if not cols:
            return np.zeros(len(frame), dtype=float)
        return frame[cols].fillna(0.0).to_numpy(float).sum(axis=1)

    curve_rows = []
    corr_rows = []
    for program, target_fates in targets.items():
        if program not in valid_scores:
            continue
        score = valid_scores[program].to_numpy(float)
        truth_target = target_probability(valid, "truth", target_fates)
        pred_target = target_probability(valid, "pred", target_fates)
        corr_rows.append(
            {
                "program": program,
                "target_fates": "+".join(target_fates),
                "barcode_future_r": corr(score, truth_target),
                "bmf_future_r": corr(score, pred_target),
            }
        )
        order = np.argsort(score, kind="mergesort")
        bins = np.array_split(order, 9)
        for b, idx in enumerate(bins):
            if len(idx) == 0:
                continue
            curve_rows.append(
                {
                    "program": program,
                    "target_fates": "+".join(target_fates),
                    "program_percentile": (b + 0.5) / len(bins),
                    "mean_rna_program_z": float(np.mean(score[idx])),
                    "barcode_future_mean": float(np.mean(truth_target[idx])),
                    "bmf_future_mean": float(np.mean(pred_target[idx])),
                    "n_cells": int(len(idx)),
                }
            )
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(TABLE_DIR / "clone_state_matched_fate_divergence_program_future_correlations.csv", index=False)
    curves = pd.DataFrame(curve_rows)
    curves.to_csv(TABLE_DIR / "clone_state_matched_fate_divergence_program_future_curves.csv", index=False)

    fig = plt.figure(figsize=(8.9, 3.65))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.0, 1.0], wspace=0.32)
    ax_map = fig.add_subplot(gs[0, 0])
    ax_truth = fig.add_subplot(gs[0, 1])
    ax_pred = fig.add_subplot(gs[0, 2])

    draw_observed_background(ax_map, obs, alpha=0.025)
    plot_fates = ["Undifferentiated", "Neutrophil", "Monocyte", "Baso", "Mast", "Meg", "Erythroid"]
    for fate in plot_fates:
        d = valid[valid["truth_dominant_fate"].eq(fate)]
        if d.empty:
            continue
        ax_map.scatter(d["spring_x"], d["spring_y"], s=6.5, c=FATE_PALETTE[fate], alpha=0.72, linewidth=0, rasterized=True, label=fate)
    style_spring(ax_map, obs)
    ax_map.set_title("A  Eligible day-2 barcode cells\ncolored by barcode future", loc="left", pad=4)
    ax_map.text(
        0.02,
        0.02,
        f"n={len(valid):,} eligible cells\nno reliability threshold",
        transform=ax_map.transAxes,
        fontsize=6.7,
        ha="left",
        va="bottom",
        color="#374151",
    )
    ax_map.legend(frameon=False, fontsize=5.8, loc="upper right", markerscale=1.3, handletextpad=0.1, borderpad=0.1)

    def plot_curve_panel(ax: plt.Axes, y_col: str, title: str, annotate_col: str) -> None:
        for program in targets:
            d = curves[curves["program"].eq(program)]
            if d.empty:
                continue
            color = program_colors[program]
            ax.plot(
                d["program_percentile"],
                d[y_col],
                color=color,
                lw=1.8,
                marker="o",
                ms=3.3,
                alpha=0.95,
            )
            last = d.iloc[-1]
            r_row = corr_df[corr_df["program"].eq(program)]
            r = float(r_row.iloc[0][annotate_col]) if not r_row.empty else float("nan")
            ax.text(
                1.005,
                float(last[y_col]),
                f"{program_labels[program]}  r={r:.2f}",
                color=color,
                fontsize=6.2,
                ha="left",
                va="center",
            )
        ax.set_xlim(0.03, 1.34)
        ax.set_ylim(-0.02, 0.72)
        ax.set_xticks([0.06, 0.50, 0.94])
        ax.set_xticklabels(["low", "mid", "high"])
        ax.set_xlabel("RNA program score rank")
        ax.set_ylabel("future fate probability")
        ax.set_title(title, loc="left", pad=4)
        ax.grid(color="#E5E7EB", lw=0.7)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plot_curve_panel(ax_truth, "barcode_future_mean", "B  Barcode-measured future", "barcode_future_r")
    plot_curve_panel(ax_pred, "bmf_future_mean", "C  BMF-predicted future", "bmf_future_r")

    fig.suptitle("Across day-2 progenitors, RNA programs shift future-fate probability", x=0.02, y=1.02, ha="left", fontsize=11.5, weight="bold")
    fig.subplots_adjust(top=0.78, bottom=0.23, left=0.055, right=0.985)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIG_DIR / f"panel_c_state_matched_fate_priming.{ext}", dpi=360 if ext == "png" else None)
    plt.close(fig)


def main() -> None:
    setup_style()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    adata = ad.read_h5ad(ADATA_PATH)
    obs = load_obs(adata)
    bmf = pd.read_csv(BMF_TABLE)

    fates = [c.removeprefix("pred_") for c in bmf.columns if c.startswith("pred_")]
    truth_cols = [f"truth_{f}" for f in fates]
    pred_cols = [f"pred_{f}" for f in fates]

    valid = bmf[bmf["eligible"].astype(bool)].copy()
    valid = valid.reset_index(drop=True)
    truth = normalize_rows(valid[truth_cols].to_numpy(float))
    pred = normalize_rows(valid[pred_cols].to_numpy(float))
    valid["future_fate_entropy"] = entropy(truth)
    valid["bmf_branch_entropy"] = entropy(pred)
    valid["truth_dominant_fate"] = np.asarray(fates, dtype=object)[truth.argmax(axis=1)]
    valid["bmf_dominant_fate"] = np.asarray(fates, dtype=object)[pred.argmax(axis=1)]

    source_idx = valid["source_index"].to_numpy(int)
    latent = np.asarray(adata.obsm["X_latent"], dtype=np.float32)[source_idx, :10]
    center, neighbor_ids, score_df = select_state_matched_pocket(valid, latent, truth, pred, fates, k=18)
    score_df.to_csv(TABLE_DIR / "clone_state_matched_fate_divergence_neighborhood_scores.csv", index=False)

    pocket = valid.iloc[neighbor_ids].copy().reset_index(drop=True)
    pocket["latent_distance_from_center"] = np.linalg.norm(latent[neighbor_ids] - latent[center], axis=1)
    # Keep the bar panels readable while preserving the major observed alternatives.
    exemplar_idx = []
    for fate in ["Neutrophil", "Baso", "Monocyte", "Undifferentiated", "Mast", "Eos"]:
        cand = pocket[pocket["truth_dominant_fate"].eq(fate)].sort_values("truth_reliability", ascending=False)
        exemplar_idx.extend(cand.head(2).index.tolist())
    if len(exemplar_idx) < 10:
        exemplar_idx.extend(pocket.sort_values("truth_reliability", ascending=False).head(10).index.tolist())
    exemplar_idx = list(dict.fromkeys(exemplar_idx))[:12]
    exemplars = pocket.loc[exemplar_idx].copy()
    exemplars = (
        exemplars.sort_values(["truth_dominant_fate", "truth_reliability", "latent_distance_from_center"], ascending=[True, False, True])
        .drop_duplicates("clone", keep="first")
        .head(10)
        .copy()
    )
    fate_order = {f: i for i, f in enumerate(["Baso", "Neutrophil", "Monocyte", "Undifferentiated", "Mast", "Meg", "Erythroid", "Eos"])}
    exemplars["_fate_order"] = exemplars["truth_dominant_fate"].map(fate_order).fillna(99)
    exemplars = exemplars.sort_values(["_fate_order", "truth_reliability"], ascending=[True, False]).drop(columns="_fate_order")
    labels = [f"clone {int(c)}" for c in exemplars["clone"].to_numpy()]

    pocket.to_csv(TABLE_DIR / "clone_state_matched_fate_divergence_selected_pocket.csv", index=False)
    exemplars.to_csv(TABLE_DIR / "clone_state_matched_fate_divergence_bar_examples.csv", index=False)

    all_corr_rows = []
    for fate in MAIN_FATES:
        if f"truth_{fate}" not in valid or f"pred_{fate}" not in valid:
            continue
        r = np.corrcoef(valid[f"truth_{fate}"].fillna(0).to_numpy(float), valid[f"pred_{fate}"].fillna(0).to_numpy(float))[0, 1]
        all_corr_rows.append({"fate": fate, "pearson_truth_vs_bmf": float(r)})
    pd.DataFrame(all_corr_rows).to_csv(TABLE_DIR / "clone_state_matched_fate_divergence_truth_bmf_correlations.csv", index=False)

    fig = plt.figure(figsize=(9.4, 6.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.08, 1.16], height_ratios=[1.04, 0.96], wspace=0.32, hspace=0.52)
    ax_entropy = fig.add_subplot(gs[0, 0])
    ax_zoom = fig.add_subplot(gs[1, 0])
    ax_truth = fig.add_subplot(gs[0, 1])
    ax_pred = fig.add_subplot(gs[1, 1])

    draw_observed_background(ax_entropy, obs, alpha=0.035)
    sc = ax_entropy.scatter(
        valid["spring_x"].to_numpy(float),
        valid["spring_y"].to_numpy(float),
        c=valid["future_fate_entropy"].to_numpy(float),
        cmap=ENTROPY_CMAP,
        vmin=0.0,
        vmax=0.78,
        s=5.5,
        alpha=0.88,
        linewidth=0,
        rasterized=True,
        zorder=10,
    )
    ax_entropy.scatter(
        pocket["spring_x"].to_numpy(float),
        pocket["spring_y"].to_numpy(float),
        s=30,
        facecolors="none",
        edgecolors="#111827",
        linewidth=0.45,
        alpha=0.7,
        zorder=24,
    )
    center_row = valid.iloc[center]
    ax_entropy.scatter([center_row["spring_x"]], [center_row["spring_y"]], s=62, marker="*", c="#111827", edgecolor="white", linewidth=0.8, zorder=30)
    add_fate_labels(ax_entropy, obs)
    style_spring(ax_entropy, obs)
    ax_entropy.set_title("A  Barcode future-fate uncertainty", loc="left", pad=7)
    cb = fig.colorbar(sc, ax=ax_entropy, orientation="horizontal", fraction=0.045, pad=0.035, aspect=28)
    cb.set_ticks([0, 0.78])
    cb.set_ticklabels(["low", "high"])
    cb.ax.tick_params(labelsize=6, length=2, pad=1)
    cb.set_label("barcode future-fate entropy", fontsize=6.5, labelpad=1)

    draw_observed_background(ax_zoom, obs[obs["t"].eq(0)], alpha=0.05)
    for fate, color in FATE_PALETTE.items():
        d = pocket[pocket["truth_dominant_fate"].eq(fate)]
        if d.empty:
            continue
        ax_zoom.scatter(
            d["spring_x"],
            d["spring_y"],
            s=42,
            c=color,
            edgecolor="white",
            linewidth=0.7,
            alpha=0.95,
            label=fate,
            zorder=20,
        )
    ax_zoom.scatter([center_row["spring_x"]], [center_row["spring_y"]], s=92, marker="*", c="#111827", edgecolor="white", linewidth=0.8, zorder=30)
    cx, cy = float(center_row["spring_x"]), float(center_row["spring_y"])
    margin = max(170.0, float(np.max(np.abs(pocket[["spring_x", "spring_y"]].to_numpy(float) - np.array([cx, cy]))) + 70.0))
    ax_zoom.set_xlim(cx - margin, cx + margin)
    ax_zoom.set_ylim(cy - margin, cy + margin)
    ax_zoom.set_aspect("equal", adjustable="box")
    ax_zoom.set_xticks([])
    ax_zoom.set_yticks([])
    for spine in ax_zoom.spines.values():
        spine.set_visible(False)
    ax_zoom.set_title("B  State-matched cells with different clonal futures", loc="left", pad=7)
    ax_zoom.legend(frameon=False, fontsize=6.3, loc="upper right", ncol=1, handletextpad=0.2, borderpad=0.1)
    ax_zoom.text(
        0.02,
        0.02,
        f"{len(pocket)} state-matched day-2 cells\nmedian latent distance = {pocket['latent_distance_from_center'].median():.2f}",
        transform=ax_zoom.transAxes,
        fontsize=7,
        ha="left",
        va="bottom",
        color="#374151",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.90, "pad": 2.0},
        zorder=50,
    )

    truth_prob = compact_prob_table(exemplars, fates, "truth")
    pred_prob = compact_prob_table(exemplars, fates, "pred")
    plot_stacked_bars(ax_truth, truth_prob, labels, "C  Barcode readout: terminal fate composition")
    plot_stacked_bars(ax_pred, pred_prob, labels, "D  BMF readout: model future-fate probabilities")

    handles = [
        mpl.patches.Patch(facecolor=FATE_PALETTE[f], edgecolor="none", label=f)
        for f in truth_prob.columns
        if truth_prob[f].sum() + pred_prob[f].sum() > 0.03
    ]
    fig.legend(
        handles=handles,
        frameon=False,
        fontsize=6.3,
        loc="lower center",
        bbox_to_anchor=(0.71, 0.018),
        ncol=min(5, len(handles)),
        handlelength=0.9,
        columnspacing=0.9,
    )

    fig.subplots_adjust(top=0.95, bottom=0.16, left=0.06, right=0.985)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIG_DIR / f"panel_c_state_matched_fate_divergence.{ext}", dpi=360 if ext == "png" else None)
    plt.close(fig)
    plot_gene_program_companion(adata, obs, valid, pocket)
    print(f"Wrote {FIG_DIR / 'panel_c_state_matched_fate_divergence.png'}")
    print(f"Wrote {FIG_DIR / 'panel_c_state_matched_fate_priming.png'}")
    print(f"Wrote {TABLE_DIR / 'clone_state_matched_fate_divergence_selected_pocket.csv'}")


if __name__ == "__main__":
    main()
