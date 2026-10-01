#!/usr/bin/env python3
"""NBT-style showcase figures for the CITE-PCFM multi-omic algorithm case.

The goal is not another metric dump. These figures frame the result as an
article story: CellCompass iteratively designed a multi-omic dynamics model,
then CITE-PCFM demonstrated RNA-initialized protein-coordinate dynamics against
observed CITE-seq targets.
"""

from __future__ import annotations

import json
from pathlib import Path
from matplotlib.path import Path as MplPath
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, PathPatch, Rectangle, Wedge

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA


sns.set_theme(style="white", context="talk")

ROOT = Path("/Users/zhenyizhang/Documents/GitHub/CytoBridge-agent-runtime-v2-20260309/multi-omics")
DATA_PATH = Path("/Users/zhenyizhang/.cellcompass/algorithm_benchmarks/datasets/gse305370_citeseq_stage2_10k/data.h5ad")
GT_DIR = ROOT / "downstream/cite_pcfm_rna_to_protein_gt"
TABLE_DIR = GT_DIR / "tables"
OUT_DIR = ROOT / "figures/nbt_cite_pcfm_showcase"
NOTE_PATH = ROOT / "downstream/nbt_cite_pcfm_showcase/figure_storyboard.md"
MANIFEST_PATH = ROOT / "downstream/nbt_cite_pcfm_showcase/manifest.json"

RNA_DIM = 50
PROT_DIM = 30
RNG = np.random.default_rng(20260527)

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

TRAJECTORIES = {
    "CITE-PCFM": ROOT / "training_runs/20260526T174204321821Z__final__cite_paired_coord_fm__campaign_20260-trial_20260526-gse305370_ci/artifacts/evaluation_trajectory.npz",
    "CRUFM": ROOT / "training_runs/20260525T220649168798Z__pilot__crufm__campaign_20260-stage2_c-baseline-crufm-gse305370_ci/artifacts/evaluation_trajectory.npz",
    "balanced OT-CFM": ROOT / "training_runs/20260525T215826526439Z__pilot__balanced_ot_cfm__campaign_20260-stage2_c-baseline-balanced_ot_-gse305370_ci/artifacts/evaluation_trajectory.npz",
}


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    NOTE_PATH.parent.mkdir(parents=True, exist_ok=True)


def savefig(fig: plt.Figure, name: str) -> Path:
    path = OUT_DIR / name
    fig.savefig(path, dpi=320, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return path


def rounded_box(ax, xy, w, h, text, fc="#F8FAFC", ec="#B8C2CC", size=11, weight="normal"):
    box = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.035",
        linewidth=1.2,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=size, weight=weight, wrap=True)
    return box


def curved_arrow(ax, p0, p1, color="#64748B", lw=1.8, rad=0.2, alpha=1.0, arrow=True):
    style = "Simple,head_width=7,head_length=9" if arrow else "-"
    patch = FancyArrowPatch(
        p0,
        p1,
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle=style,
        linewidth=lw,
        color=color,
        alpha=alpha,
    )
    ax.add_patch(patch)
    return patch


def sample_rows(n: int, max_n: int) -> np.ndarray:
    if n <= max_n:
        return np.arange(n)
    return np.sort(RNG.choice(n, size=max_n, replace=False))


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


def nearest_time(keys, t: float) -> float:
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


def load_tables() -> dict:
    return {
        "phenotype": pd.read_csv(TABLE_DIR / "phenotype.csv"),
        "per_time": pd.read_csv(TABLE_DIR / "per_time.csv"),
        "score": pd.read_csv(TABLE_DIR / "summary_scorecard_raw.csv"),
        "terminal": pd.read_csv(TABLE_DIR / "initial_rna_neighborhood_terminal_protein_fates.csv"),
    }


def figure_a_evidence_to_model() -> Path:
    fig, ax = plt.subplots(figsize=(14.5, 7.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.02, 0.96, "A", fontsize=18, weight="bold")
    ax.text(0.08, 0.96, "Multi-omic evidence becomes an RNA-initialized protein dynamics model", fontsize=19, weight="bold")

    # Evidence cards
    cards = [
        ("RNA time series", "cell-state geometry\nand temporal snapshots", "#E8F2FF"),
        ("CITE protein PCs", "paired phenotype\ncoordinates", "#FFF1E6"),
        ("Observed targets", "time-matched protein\ndistribution", "#EEF9F0"),
    ]
    for i, (title, sub, fc) in enumerate(cards):
        y = 0.72 - i * 0.18
        rounded_box(ax, (0.05, y), 0.22, 0.12, f"{title}\n{sub}", fc=fc, ec="#CBD5E1", size=11, weight="bold")
        # Cell dots
        cx, cy = 0.072, y + 0.06
        for k in range(18):
            ax.scatter(cx + RNG.normal(0, 0.012), cy + RNG.normal(0, 0.025), s=12, color=list(PHENO_COLORS.values())[k % 6], alpha=0.8, zorder=3)

    rounded_box(ax, (0.36, 0.63), 0.25, 0.19, "CellCompass design step\n\nchoose assumptions allowed by evidence", fc="#F8FAFC", ec="#7C3AED", size=12, weight="bold")
    ax.text(0.485, 0.58, "rejects unstable or unsupported mechanisms\nkeeps the evidence-compatible rule", ha="center", va="top", fontsize=10, color="#475569")

    # RNA trajectory schematic
    ax.plot([0.69, 0.76, 0.83, 0.91], [0.72, 0.78, 0.70, 0.80], color="#B73A3A", lw=3.2)
    for x, y, t in zip([0.69, 0.76, 0.83, 0.91], [0.72, 0.78, 0.70, 0.80], ["t0", "t1", "t2", "t3"]):
        ax.add_patch(Circle((x, y), 0.026, fc="#FEE2E2", ec="#B73A3A", lw=1.2))
        ax.text(x, y - 0.055, t, ha="center", fontsize=9)
    ax.text(0.80, 0.87, "RNA state trajectory", ha="center", fontsize=13, weight="bold", color="#7F1D1D")

    rounded_box(ax, (0.68, 0.39), 0.25, 0.15, "protein readout\n$h_{\\eta}(R_t, t)$", fc="#FFF7ED", ec="#F97316", size=15, weight="bold")
    ax.text(0.805, 0.34, "protein-coordinate trajectory\nwithout requiring protein at inference", ha="center", va="top", fontsize=10, color="#475569")
    for x in [0.75, 0.82, 0.89]:
        curved_arrow(ax, (x, 0.67), (0.805, 0.55), color="#F97316", lw=1.4, rad=-0.2)

    for p0, p1 in [((0.27, 0.78), (0.36, 0.74)), ((0.27, 0.60), (0.36, 0.72)), ((0.27, 0.42), (0.36, 0.69)), ((0.61, 0.72), (0.68, 0.73)), ((0.61, 0.66), (0.68, 0.48))]:
        curved_arrow(ax, p0, p1, color="#64748B", lw=1.8, rad=0.04)

    rounded_box(ax, (0.36, 0.20), 0.57, 0.095, "Article message: the model is not only a better score; it is a designed biological assumption\nRNA dynamics plus a learned CITE protein phenotype readout.", fc="#F0FDF4", ec="#16A34A", size=12, weight="bold")
    return savefig(fig, "fig_nbt6A_evidence_to_model_schematic.png")


def figure_b_design_genealogy() -> Path:
    fig, ax = plt.subplots(figsize=(15, 7.2))
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.0, 0.96, "B", fontsize=18, weight="bold")
    ax.text(0.07, 0.96, "The design search converged through biological and metric-driven failure modes", fontsize=18, weight="bold")

    nodes = [
        ("CITE-VGFM", "unbalanced multi-omic idea\nfailed Stage 1 feasibility", 10, "failed", "#CBD5E1"),
        ("CITE-BFM", "balanced revision\nprotein claim still weak", 13, "failed", "#CBD5E1"),
        ("Factorized FM", "separate RNA/protein structure\nprovenance blocked", 1, "failed", "#CBD5E1"),
        ("Residual SB-FM", "stochastic bridge mechanism\nfailed against baselines", 27, "failed", "#CBD5E1"),
        ("CITE-PCFM", "paired-coordinate coupling\nRNA-driven protein readout", 70, "locked", "#FCA5A5"),
    ]
    xs = np.linspace(0.10, 0.90, len(nodes))
    ys = [0.50, 0.62, 0.47, 0.61, 0.52]
    for i in range(len(nodes) - 1):
        curved_arrow(ax, (xs[i] + 0.07, ys[i]), (xs[i + 1] - 0.07, ys[i + 1]), color="#94A3B8", lw=2.2, rad=0.12)

    for x, y, (name, desc, trials, status, fc) in zip(xs, ys, nodes):
        color = "#B73A3A" if status == "locked" else "#64748B"
        ax.add_patch(Circle((x, y), 0.066, fc=fc, ec=color, lw=2.0))
        ax.text(x, y + 0.005, name, ha="center", va="center", fontsize=11, weight="bold", color="#111827")
        ax.text(x, y - 0.12, desc, ha="center", va="top", fontsize=9.5, color="#334155")
        ax.text(x, y + 0.103, f"{trials} trials", ha="center", fontsize=10, color=color, weight="bold")
        if status == "locked":
            ax.text(x, y - 0.205, "locked final regression", ha="center", fontsize=10.5, color="#B73A3A", weight="bold")

    # Stage strip for final algorithm
    stage_x = [0.57, 0.66, 0.75, 0.84]
    labels = ["Stage 1\nfeasible", "Stage 2\nclaim passed", "Stage 3\ntuned", "Final\nlocked"]
    for x, label in zip(stage_x, labels):
        rounded_box(ax, (x, 0.17), 0.075, 0.08, label, fc="#FFF7ED", ec="#FB923C", size=8.5, weight="bold")
    for x0, x1 in zip(stage_x[:-1], stage_x[1:]):
        curved_arrow(ax, (x0 + 0.075, 0.21), (x1, 0.21), color="#FB923C", lw=1.4, rad=0)

    ax.text(0.07, 0.16, "What changed?", fontsize=13, weight="bold")
    bullets = [
        "dropped unbalanced/stochastic claims when evidence did not support them",
        "kept paired CITE protein geometry as the model-shaping signal",
        "separated RNA dynamics from protein readout at inference",
    ]
    for i, b in enumerate(bullets):
        ax.text(0.07, 0.11 - i * 0.045, f"• {b}", fontsize=10.5, color="#334155")
    return savefig(fig, "fig_nbt6B_cellcompass_design_genealogy.png")


def figure_c_fate_river(tables: dict) -> Path:
    pheno = tables["phenotype"]
    phenotypes = ["HSC", "MasP", "EryP", "MoP", "EryP"]
    phenotypes = [p for p in ["HSC", "MasP", "EryP", "MoP", "MkP", "NeuP"] if p in set(pheno["phenotype"])]
    panels = ["Observed target", "CITE-PCFM", "CRUFM", "balanced OT-CFM"]
    fig, axes = plt.subplots(4, 1, figsize=(12.5, 8.5), sharex=True)
    for ax, panel in zip(axes, panels):
        values = []
        times = sorted(pheno["time"].unique())
        for phenotype in phenotypes:
            d = pheno[pheno.phenotype == phenotype]
            if panel == "Observed target":
                s = d.groupby("time")["observed_fraction"].mean().reindex(times).fillna(0).values
            else:
                s = d[d.method == panel].set_index("time")["generated_fraction"].reindex(times).fillna(0).values
            values.append(s)
        ax.stackplot(
            times,
            values,
            colors=[PHENO_COLORS[p] for p in phenotypes],
            labels=phenotypes,
            alpha=0.92,
            baseline="sym",
            linewidth=0,
        )
        ax.axhline(0, color="white", lw=1.2, alpha=0.8)
        ax.text(-0.07, 0.5, panel, transform=ax.transAxes, ha="right", va="center", fontsize=12, weight="bold")
        ax.set_yticks([])
        ax.spines[["top", "right", "left"]].set_visible(False)
    axes[0].text(-0.07, 1.1, "C", transform=axes[0].transAxes, fontsize=18, weight="bold")
    axes[0].set_title("Protein phenotype river: observed target versus generated CITE protein dynamics", loc="left", fontsize=17, weight="bold")
    axes[-1].set_xlabel("time")
    axes[0].legend(ncol=6, bbox_to_anchor=(1.0, 1.34), loc="upper right", frameon=False, fontsize=9)
    fig.tight_layout()
    return savefig(fig, "fig_nbt6C_protein_phenotype_river.png")


def figure_d_protein_landscape() -> Path:
    adata = ad.read_h5ad(DATA_PATH)
    obs = adata.obs.copy()
    obs["time_point_processed"] = obs["time_point_processed"].astype(float)
    protein = np.asarray(adata.obsm["X_protein_pca"], dtype=np.float32)
    trajectories = {m: load_traj(p) for m, p in TRAJECTORIES.items()}
    times = [1.0, 2.0, 3.0, 4.0]
    fig, axes = plt.subplots(1, len(times), figsize=(15, 4.2), sharex=True, sharey=True)
    for ax, t in zip(axes, times):
        obs_mask = np.asarray(obs["time_point_processed"] == t)
        obs_prot = protein[obs_mask]
        oi = sample_rows(obs_prot.shape[0], 1000)
        sns.kdeplot(x=obs_prot[oi, 0], y=obs_prot[oi, 1], levels=7, color="#111827", linewidths=1.2, ax=ax, alpha=0.45)
        ax.scatter(obs_prot[oi, 0], obs_prot[oi, 1], s=5, color="#94A3B8", alpha=0.12, linewidths=0)
        for method, traj in trajectories.items():
            tt = nearest_time(traj["points"].keys(), t)
            pts = traj["points"][tt]
            w = traj["weights"].get(tt)
            idx = weighted_sample(pts.shape[0], 450, w)
            gp = pts[idx, RNA_DIM : RNA_DIM + PROT_DIM]
            ax.scatter(gp[:, 0], gp[:, 1], s=7, color=METHOD_COLORS[method], alpha=0.28, linewidths=0, label=method if t == times[0] else None)
            ax.scatter(gp[:, 0].mean(), gp[:, 1].mean(), s=65, color=METHOD_COLORS[method], edgecolor="white", linewidth=1.0, zorder=5)
        ax.set_title(f"target time {t:g}")
        ax.set_xlabel("protein PC1")
        ax.set_ylabel("protein PC2" if t == times[0] else "")
    axes[0].text(-0.16, 1.12, "D", transform=axes[0].transAxes, fontsize=18, weight="bold")
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle("Generated protein states against observed target landscapes", fontsize=17, weight="bold", y=1.06)
    fig.tight_layout()
    return savefig(fig, "fig_nbt6D_observed_target_protein_landscape.png")


def draw_chord(ax, theta0, theta1, r0, r1, color, width=1.0, alpha=0.5):
    p0 = np.array([r0 * np.cos(theta0), r0 * np.sin(theta0)])
    p1 = np.array([r1 * np.cos(theta1), r1 * np.sin(theta1)])
    c0 = p0 * 0.28
    c1 = p1 * 0.28
    path = MplPath([p0, c0, c1, p1], [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
    patch = PathPatch(path, facecolor="none", edgecolor=color, lw=width, alpha=alpha, capstyle="round")
    ax.add_patch(patch)


def figure_e_rna_neighborhood_chord(tables: dict) -> Path:
    term = tables["terminal"]
    d = term[term.method == "CITE-PCFM"].copy()
    phenotypes = ["EryP", "HSC", "MasP", "MoP"]
    neighborhoods = sorted(d["initial_rna_neighborhood"].unique())
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"aspect": "equal"})
    ax.axis("off")
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)
    ax.text(-1.15, 1.12, "E", fontsize=18, weight="bold")
    ax.text(0, 1.14, "RNA neighborhoods resolve terminal protein fates", ha="center", fontsize=16, weight="bold")

    left_angles = np.linspace(np.deg2rad(140), np.deg2rad(220), len(neighborhoods))
    right_angles = np.linspace(np.deg2rad(-45), np.deg2rad(45), len(phenotypes))
    for n, th in zip(neighborhoods, left_angles):
        ax.add_patch(Circle((np.cos(th), np.sin(th)), 0.045, fc="#E0F2FE", ec="#0284C7", lw=1.2))
        ax.text(1.09 * np.cos(th), 1.09 * np.sin(th), f"N{int(n)}", ha="center", va="center", fontsize=9)
    for p, th in zip(phenotypes, right_angles):
        color = PHENO_COLORS[p]
        ax.add_patch(Wedge((0, 0), 1.04, np.rad2deg(th) - 5, np.rad2deg(th) + 5, width=0.10, fc=color, ec="white", alpha=0.95))
        ax.text(1.17 * np.cos(th), 1.17 * np.sin(th), p, ha="center", va="center", fontsize=10, color=color, weight="bold")

    max_width = 11
    for _, row in d.iterrows():
        if row["terminal_phenotype"] not in phenotypes:
            continue
        frac = float(row["fraction"])
        if frac < 0.035:
            continue
        th0 = left_angles[neighborhoods.index(row["initial_rna_neighborhood"])]
        th1 = right_angles[phenotypes.index(row["terminal_phenotype"])]
        draw_chord(ax, th0, th1, 0.98, 0.98, PHENO_COLORS[row["terminal_phenotype"]], width=max(0.5, frac * max_width), alpha=0.45)
    ax.text(0, -1.15, "CITE-PCFM terminal protein phenotype fractions from t0 RNA neighborhoods", ha="center", fontsize=10, color="#475569")
    return savefig(fig, "fig_nbt6E_rna_neighborhood_terminal_chord.png")


def figure_f_tradeoff_radar(tables: dict) -> Path:
    score = tables["score"].copy()
    # Use future-time scorecard generated by the previous script.
    metrics = [
        ("protein_energy", "protein\nenergy"),
        ("centroid_displacement", "centroid\nalignment"),
        ("covariance_error", "RNA-protein\nstructure"),
        ("nearest_distance", "target\nproximity"),
        ("variance_error", "protein\ndiversity"),
        ("phenotype_fraction_error", "phenotype\nfractions"),
    ]
    norm = score.copy()
    for col, _ in metrics:
        lo, hi = norm[col].min(), norm[col].max()
        norm[col] = 0.0 if hi == lo else 1.0 - (norm[col] - lo) / (hi - lo)  # higher is better
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(7.6, 7.6), subplot_kw={"polar": True})
    ax.text(-0.1, 1.11, "F", transform=ax.transAxes, fontsize=18, weight="bold")
    for method, color in METHOD_COLORS.items():
        vals = norm[norm.method == method][[m[0] for m in metrics]].iloc[0].tolist()
        vals += vals[:1]
        ax.plot(angles, vals, color=color, lw=2.5, label=method)
        ax.fill(angles, vals, color=color, alpha=0.10)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m[1] for m in metrics], fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_yticklabels([])
    ax.grid(color="#CBD5E1", lw=0.9)
    ax.set_title("What the new model solves, and what biology still exposes", y=1.12, fontsize=15, weight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.13), frameon=False, fontsize=10)
    return savefig(fig, "fig_nbt6F_biological_tradeoff_radar.png")


def write_storyboard(figures: list[Path]) -> None:
    text = """# NBT-style CITE-PCFM Figure Package

These figures are intended as article-thinking drafts, not final publication art. The message is:

CellCompass used failures and formal checks to converge on CITE-PCFM, a multi-omic dynamics design where RNA state is rolled forward and protein-coordinate phenotype is read out from the generated RNA trajectory.

## Candidate main Figure 6

- **6A** introduces the biological evidence-to-model assumption: paired CITE protein coordinates shape a protein readout on RNA dynamics.
- **6C** is the most accessible biological panel: observed and generated protein phenotype rivers over time.
- **6D** shows target protein landscapes with generated states overlaid.
- **6F** is the honest scorecard: CITE-PCFM improves protein-coordinate agreement and covariance/diversity, while phenotype fractions remain a biological limitation.

## Candidate Extended Data or Box Figure

- **6B** shows the CellCompass design genealogy: multiple proposed algorithms, failures, revisions and final locked release.
- **6E** shows a less common RNA-neighborhood-to-terminal-protein chord view. It is useful if the story focuses on RNA-conditioned protein fate.

## Wording caution

Do not call CRUFM or balanced OT-CFM RNA-only baselines in this CITE benchmark unless a separate audit proves their state excludes protein coordinates. The current trajectories are joint 80-dimensional states.
"""
    text += "\n## Files\n\n"
    for p in figures:
        text += f"- `{p}`\n"
    NOTE_PATH.write_text(text, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    tables = load_tables()
    figures = [
        figure_a_evidence_to_model(),
        figure_b_design_genealogy(),
        figure_c_fate_river(tables),
        figure_d_protein_landscape(),
        figure_e_rna_neighborhood_chord(tables),
        figure_f_tradeoff_radar(tables),
    ]
    write_storyboard(figures)
    manifest = {
        "analysis": "NBT-style CITE-PCFM showcase figures",
        "figures": [str(p) for p in figures],
        "storyboard": str(NOTE_PATH),
        "source_tables": [str(p) for p in sorted(TABLE_DIR.glob("*.csv"))],
        "dataset": str(DATA_PATH),
        "note": "Figures avoid heatmap-style presentation except source score tables; outputs are intended as main-figure candidates and extended-data drafts.",
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
