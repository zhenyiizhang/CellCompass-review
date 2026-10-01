#!/usr/bin/env python3
"""Early RNA determinants of day-2 neutrophil/monocyte fate bias.

This analysis asks why neutrophil and monocyte barcode futures are not well
explained by mature marker programs at day 2.  It scans all measured genes in
eligible day-2 barcode cells and keeps genes that agree in two independent
readouts:

1. association with LARRY barcode future-fate probability;
2. association with BMF future-fate probability for the same cells.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cellcompass_mpl_cache")
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse


ROOT = Path(__file__).resolve().parents[2]
TABLE_DIR = ROOT / "downstream/cellcompass_clone_case"
FIG_DIR = ROOT / "figures/cellcompass_clone_case"
ADATA_PATH = Path("/Users/zhenyizhang/.cellcompass/algorithm_benchmarks/datasets/weinreb_rawrebuild_full_k10mindiff1/data.h5ad")
BMF_TABLE = ROOT / "downstream/clone_fate_spring_final/clone_cuvgf_bmf_final_day2_fate_probabilities.csv"

FATE_COLOR = {
    "Neutrophil": "#19B7C7",
    "Monocyte": "#8C564B",
}
MATURE_MARKERS = {
    "Neutrophil": ["Cebpe", "Lyz2", "S100a8", "S100a9"],
    "Monocyte": ["Irf8", "Csf1r", "Lyz2", "Ly6c2"],
}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.linewidth": 1.0,
            "axes.edgecolor": "#111827",
            "axes.titlesize": 8.8,
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


def normalize_expr(adata: ad.AnnData, obs_index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = adata.X[np.asarray(obs_index, dtype=int), :]
    if sparse.issparse(x):
        x = x.toarray()
    x = np.asarray(x, dtype=np.float32)
    x = (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + 1e-6)
    return x, np.asarray(adata.var_names.astype(str))


def corr_vector(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    y = (y - y.mean()) / (y.std() + 1e-9)
    return (x.T @ y) / max(1, x.shape[0] - 1)


def residual(y: np.ndarray, cov: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    cov = np.asarray(cov, dtype=float)
    design = np.c_[np.ones(len(y)), cov]
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    return y - design @ beta


def binned_curve(score: np.ndarray, y: np.ndarray, bins: int = 9) -> pd.DataFrame:
    order = np.argsort(score, kind="mergesort")
    rows = []
    for i, idx in enumerate(np.array_split(order, bins)):
        rows.append(
            {
                "rank": (i + 0.5) / bins,
                "score_mean": float(np.mean(score[idx])),
                "future_mean": float(np.mean(y[idx])),
                "n_cells": int(len(idx)),
            }
        )
    return pd.DataFrame(rows)


def gene_module(x: np.ndarray, genes: np.ndarray, selected: list[str]) -> np.ndarray:
    idx = [int(np.where(genes == g)[0][0]) for g in selected if g in genes]
    if not idx:
        return np.zeros(x.shape[0], dtype=float)
    return x[:, idx].mean(axis=1)


def scan_fate(genes: np.ndarray, x: np.ndarray, valid: pd.DataFrame, fate: str) -> tuple[pd.DataFrame, list[str], list[str]]:
    truth = valid[f"truth_{fate}"].to_numpy(float)
    pred = valid[f"pred_{fate}"].to_numpy(float)
    mature = [g for g in MATURE_MARKERS[fate] if g in set(genes)]
    mature_score = gene_module(x, genes, mature)

    out = pd.DataFrame(
        {
            "gene": genes,
            "r_barcode": corr_vector(x, truth),
            "r_bmf": corr_vector(x, pred),
            "r_barcode_after_mature": corr_vector(x, residual(truth, mature_score)),
            "r_bmf_after_mature": corr_vector(x, residual(pred, mature_score)),
        }
    )
    out["agreement_score"] = np.minimum(out["r_barcode"], out["r_bmf"])
    out["residual_agreement_score"] = np.minimum(out["r_barcode_after_mature"], out["r_bmf_after_mature"])
    out["mature_marker"] = out["gene"].isin(mature)
    top = out[~out["mature_marker"]].sort_values("residual_agreement_score", ascending=False).head(8)["gene"].tolist()
    return out.sort_values("residual_agreement_score", ascending=False), top, mature


def plot_gene_agreement(ax: plt.Axes, table: pd.DataFrame, fate: str, top: list[str], mature: list[str]) -> None:
    color = FATE_COLOR[fate]
    ax.scatter(table["r_barcode"], table["r_bmf"], s=7, c="#CBD5E1", alpha=0.45, linewidth=0, rasterized=True)
    top_df = table[table["gene"].isin(top)]
    mat_df = table[table["gene"].isin(mature)]
    ax.scatter(top_df["r_barcode"], top_df["r_bmf"], s=22, c=color, edgecolor="white", linewidth=0.5, zorder=5)
    ax.scatter(mat_df["r_barcode"], mat_df["r_bmf"], s=28, marker="x", c="#111827", linewidth=1.0, zorder=6)
    for _, row in pd.concat([top_df.head(7), mat_df]).iterrows():
        ax.text(
            float(row["r_barcode"]) + 0.006,
            float(row["r_bmf"]) + 0.003,
            str(row["gene"]),
            fontsize=6.2,
            color=color if row["gene"] in top else "#111827",
            ha="left",
            va="center",
        )
    lim = max(0.34, float(np.nanmax(table[["r_barcode", "r_bmf"]].to_numpy())) + 0.04)
    ax.plot([-0.08, lim], [-0.08, lim], color="#E5E7EB", lw=1.0, zorder=0)
    ax.axhline(0, color="#CBD5E1", lw=0.8)
    ax.axvline(0, color="#CBD5E1", lw=0.8)
    ax.set_xlim(-0.08, lim)
    ax.set_ylim(-0.08, lim)
    ax.set_xlabel("gene vs barcode future")
    ax.set_ylabel("gene vs BMF future")
    ax.set_title(f"{fate}: early genes supported by both readouts", loc="left", pad=4)
    ax.grid(color="#EEF2F7", lw=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_module_curves(ax: plt.Axes, x: np.ndarray, genes: np.ndarray, valid: pd.DataFrame, fate: str, top: list[str], mature: list[str]) -> None:
    color = FATE_COLOR[fate]
    truth = valid[f"truth_{fate}"].to_numpy(float)
    pred = valid[f"pred_{fate}"].to_numpy(float)
    top_score = gene_module(x, genes, top)
    mature_score = gene_module(x, genes, mature)
    curve_max = 0.0
    for score, label, style, alpha in [
        (top_score, "early-gene module", "-", 1.0),
        (mature_score, "mature-marker module", "--", 0.72),
    ]:
        for y, source, shade in [(truth, "barcode", color), (pred, "BMF", "#7C3AED")]:
            c = binned_curve(score, y)
            curve_max = max(curve_max, float(c["future_mean"].max()))
            ax.plot(
                c["rank"],
                c["future_mean"],
                linestyle=style,
                color=shade,
                lw=1.8 if label.startswith("early") else 1.35,
                marker="o" if label.startswith("early") else None,
                ms=3.0,
                alpha=alpha,
                label=f"{source}: {label}",
            )
    ax.set_xlim(0.04, 0.96)
    ax.set_xticks([0.06, 0.50, 0.94])
    ax.set_xticklabels(["low", "mid", "high"])
    ax.set_ylim(-0.02, max(0.45, min(0.72, curve_max + 0.10)))
    ax.set_xlabel("cells ranked by module score")
    ax.set_ylabel(f"{fate} future probability")
    ax.set_title(f"{fate}: early genes explain more than mature markers", loc="left", pad=4)
    ax.grid(color="#E5E7EB", lw=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=5.8, loc="upper left", handlelength=1.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> None:
    setup_style()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    adata = ad.read_h5ad(ADATA_PATH)
    bmf = pd.read_csv(BMF_TABLE)
    valid = bmf[bmf["eligible"].astype(bool)].copy()
    x, genes = normalize_expr(adata, valid["source_index"].to_numpy(int))

    results: dict[str, tuple[pd.DataFrame, list[str], list[str]]] = {}
    for fate in ["Neutrophil", "Monocyte"]:
        table, top, mature = scan_fate(genes, x, valid, fate)
        table.insert(0, "fate", fate)
        table.to_csv(TABLE_DIR / f"clone_day2_{fate.lower()}_early_gene_bias_scan.csv", index=False)
        results[fate] = (table, top, mature)

    summary_rows = []
    for fate, (table, top, mature) in results.items():
        for label, geneset in [("early_candidate_module", top), ("mature_marker_module", mature)]:
            score = gene_module(x, genes, geneset)
            for source, y in [
                ("barcode", valid[f"truth_{fate}"].to_numpy(float)),
                ("bmf", valid[f"pred_{fate}"].to_numpy(float)),
            ]:
                yy = y - y.mean()
                ss = score - score.mean()
                denom = float(np.sqrt(np.sum(yy * yy) * np.sum(ss * ss)))
                r = float(np.sum(yy * ss) / denom) if denom > 1e-12 else float("nan")
                summary_rows.append({"fate": fate, "module": label, "source": source, "r": r, "genes": ",".join(geneset)})
    pd.DataFrame(summary_rows).to_csv(TABLE_DIR / "clone_day2_neutrophil_monocyte_bias_module_summary.csv", index=False)

    fig = plt.figure(figsize=(8.9, 6.0))
    gs = fig.add_gridspec(2, 2, wspace=0.28, hspace=0.36)
    ax_n_scatter = fig.add_subplot(gs[0, 0])
    ax_m_scatter = fig.add_subplot(gs[0, 1])
    ax_n_curve = fig.add_subplot(gs[1, 0])
    ax_m_curve = fig.add_subplot(gs[1, 1])

    for fate, ax_scatter, ax_curve in [
        ("Neutrophil", ax_n_scatter, ax_n_curve),
        ("Monocyte", ax_m_scatter, ax_m_curve),
    ]:
        table, top, mature = results[fate]
        plot_gene_agreement(ax_scatter, table, fate, top, mature)
        plot_module_curves(ax_curve, x, genes, valid, fate, top, mature)

    fig.suptitle("Neutrophil and monocyte futures are biased by early RNA states before mature markers", x=0.02, y=0.99, ha="left", fontsize=12, weight="bold")
    fig.subplots_adjust(top=0.90, bottom=0.10, left=0.075, right=0.985)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIG_DIR / f"panel_c_day2_neutrophil_monocyte_bias_mechanism.{ext}", dpi=360 if ext == "png" else None)
    plt.close(fig)
    print(f"Wrote {FIG_DIR / 'panel_c_day2_neutrophil_monocyte_bias_mechanism.png'}")


if __name__ == "__main__":
    main()
