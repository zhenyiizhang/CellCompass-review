#!/usr/bin/env python3
"""Evaluate BMF growth fields directly on observed cells in SPRING space.

No rollout is used here.  Each observed cell is kept at its measured SPRING
coordinate.  The final BMF branch_growth_net is evaluated at that cell's
observed latent state and observed model time.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cellcompass_mpl_cache")
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import anndata as ad
import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compose_clone_scdiffeq_style_biology as base  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
TABLE_DIR = ROOT / "downstream/cellcompass_clone_case"
FIG_DIR = ROOT / "figures/cellcompass_clone_case"

TIME_LABELS = {0: "day 2", 1: "day 4", 2: "day 6"}
SHOW_BRANCHES = ["Neutrophil", "Monocyte", "Baso", "Mast", "Meg", "Erythroid"]
MISSING_CLONE_STRINGS = {"", "nan", "none", "null", "na", "<na>"}


def setup_style() -> None:
    base.setup_style()
    mpl.rcParams.update(
        {
            "axes.titlesize": 9.0,
            "axes.titleweight": "bold",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def spring_limits(obs: pd.DataFrame) -> tuple[tuple[float, float], tuple[float, float]]:
    x = obs["spring_x"].to_numpy(float)
    y = obs["spring_y"].to_numpy(float)
    padx = 0.035 * (np.nanmax(x) - np.nanmin(x))
    pady = 0.035 * (np.nanmax(y) - np.nanmin(y))
    return (float(np.nanmin(x) - padx), float(np.nanmax(x) + padx)), (
        float(np.nanmin(y) - pady),
        float(np.nanmax(y) + pady),
    )


def style_spring(ax, obs: pd.DataFrame) -> None:
    (xmin, xmax), (ymin, ymax) = spring_limits(obs)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def evaluate_observed_growth(adata: ad.AnnData, obs: pd.DataFrame, fates: list[str]) -> pd.DataFrame:
    latent = np.asarray(adata.obsm["X_latent"], dtype=np.float32)
    model, _ = base.load_bmf_model(latent.shape[1], len(fates))
    x = torch.as_tensor(latent, dtype=torch.float32)
    t = torch.as_tensor(obs["t"].to_numpy(float)[:, None], dtype=torch.float32)
    branch_eye = torch.eye(len(fates), dtype=torch.float32)

    rows = obs[["spring_x", "spring_y", "t", "cell_type", "clone_str"]].copy()
    with torch.no_grad():
        gate = torch.softmax(model.gate_net(torch.cat([x, t], dim=1)), dim=1).cpu().numpy()
        growth_mat = np.zeros((latent.shape[0], len(fates)), dtype=np.float32)
        for j, fate in enumerate(fates):
            b = branch_eye[j].repeat(latent.shape[0], 1)
            inp = torch.cat([x, t, b], dim=1)
            growth_mat[:, j] = model.branch_growth_net(inp).cpu().numpy().reshape(-1)

    for j, fate in enumerate(fates):
        rows[f"gate_{fate}"] = gate[:, j]
        rows[f"growth_{fate}"] = growth_mat[:, j]
    rows["gate_weighted_growth"] = np.sum(gate * growth_mat, axis=1)
    rows["max_growth"] = np.max(growth_mat, axis=1)
    rows["max_growth_branch"] = np.asarray(fates, dtype=object)[np.argmax(growth_mat, axis=1)]
    rows.to_csv(TABLE_DIR / "clone_bmf_observed_cell_growth_spring.csv", index=False)
    return rows


def zscore_by_group(df: pd.DataFrame, value: str, group: str, out: str) -> pd.DataFrame:
    df = df.copy()
    df[out] = np.nan
    for _, idx in df.groupby(group).groups.items():
        vals = df.loc[idx, value].to_numpy(float)
        mu = float(np.nanmean(vals))
        sd = float(np.nanstd(vals))
        df.loc[idx, out] = (vals - mu) / max(sd, 1e-8)
    return df


def rank_score_by_group(df: pd.DataFrame, value: str, group: str, out: str) -> pd.DataFrame:
    df = df.copy()
    df[out] = np.nan
    for _, idx in df.groupby(group).groups.items():
        ranks = df.loc[idx, value].rank(method="average", pct=True).to_numpy(float)
        df.loc[idx, out] = 2.0 * (ranks - 0.5)
    return df


def compute_clone_derived_growth(obs: pd.DataFrame, growth: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare model growth with realized barcode clone expansion.

    Realized clone growth is defined for interval t -> t+1:
        log((N_clone,t+1 + 0.5) / (N_clone,t + 0.5)).
    It is assigned back to cells from the same clone at time t.
    """
    barcoded = obs[~obs["clone_str"].astype(str).str.lower().isin(MISSING_CLONE_STRINGS)].copy()
    counts = barcoded.groupby(["clone_str", "t"]).size().unstack(fill_value=0)
    rows = []
    for t in [0, 1]:
        if t not in counts.columns:
            counts[t] = 0
        if t + 1 not in counts.columns:
            counts[t + 1] = 0
        for clone, row in counts.iterrows():
            n_current = int(row[t])
            if n_current <= 0:
                continue
            n_next = int(row[t + 1])
            rows.append(
                {
                    "clone_str": str(clone),
                    "t": int(t),
                    "interval": f"{TIME_LABELS[t]}->{TIME_LABELS[t + 1]}",
                    "n_current": n_current,
                    "n_next": n_next,
                    "clone_derived_growth": float(np.log((n_next + 0.5) / (n_current + 0.5))),
                }
            )
    clone_truth = pd.DataFrame(rows)
    cell_compare = growth.merge(clone_truth, on=["clone_str", "t"], how="inner")
    cell_compare = zscore_by_group(cell_compare, "gate_weighted_growth", "t", "predicted_growth_z")
    cell_compare = zscore_by_group(cell_compare, "clone_derived_growth", "t", "clone_derived_growth_z")
    cell_compare = rank_score_by_group(cell_compare, "gate_weighted_growth", "t", "predicted_growth_rank")
    cell_compare = rank_score_by_group(cell_compare, "clone_derived_growth", "t", "clone_derived_growth_rank")

    clone_compare = (
        cell_compare.groupby(["clone_str", "t", "interval", "n_current", "n_next"], as_index=False)
        .agg(
            predicted_growth=("gate_weighted_growth", "mean"),
            predicted_growth_z=("predicted_growth_z", "mean"),
            clone_derived_growth=("clone_derived_growth", "first"),
            clone_derived_growth_z=("clone_derived_growth_z", "first"),
            n_cells_for_prediction=("source_index" if "source_index" in cell_compare.columns else "spring_x", "size"),
        )
    )
    clone_compare = rank_score_by_group(clone_compare, "predicted_growth", "interval", "predicted_growth_rank")
    clone_compare = rank_score_by_group(clone_compare, "clone_derived_growth", "interval", "clone_derived_growth_rank")

    corr_rows = []
    for interval, d0 in list(clone_compare.groupby("interval")) + [("combined", clone_compare)]:
        for min_n in [1, 2, 3, 5]:
            d = d0[d0["n_current"].ge(min_n)].copy()
            if len(d) < 4:
                corr = np.nan
                corr_z = np.nan
                corr_rank = np.nan
            else:
                corr = float(np.corrcoef(d["predicted_growth"], d["clone_derived_growth"])[0, 1])
                corr_z = float(np.corrcoef(d["predicted_growth_z"], d["clone_derived_growth_z"])[0, 1])
                corr_rank = float(np.corrcoef(d["predicted_growth_rank"], d["clone_derived_growth_rank"])[0, 1])
            corr_rows.append(
                {
                    "interval": interval,
                    "min_current_clone_cells": min_n,
                    "n_clone_time_points": int(len(d)),
                    "pearson_raw": corr,
                    "pearson_within_interval_z": corr_z,
                    "rank_correlation": corr_rank,
                }
            )
    corr = pd.DataFrame(corr_rows)
    clone_truth.to_csv(TABLE_DIR / "clone_barcode_derived_growth_by_clone_interval.csv", index=False)
    cell_compare.to_csv(TABLE_DIR / "clone_bmf_observed_vs_barcode_growth_cells.csv", index=False)
    clone_compare.to_csv(TABLE_DIR / "clone_bmf_observed_vs_barcode_growth_clone_level.csv", index=False)
    corr.to_csv(TABLE_DIR / "clone_bmf_observed_vs_barcode_growth_correlation.csv", index=False)
    return cell_compare, clone_compare, corr


def add_labels(ax, obs: pd.DataFrame, labels: list[str]) -> None:
    terminal = obs[obs["t"].eq(2)]
    offsets = {
        "Neutrophil": (16, -26),
        "Monocyte": (22, 22),
        "Baso": (-24, -16),
        "Mast": (-16, -30),
        "Meg": (-24, 18),
        "Erythroid": (20, -18),
        "Undifferentiated": (-22, 24),
    }
    for fate in labels:
        d = terminal[terminal["cell_type"].eq(fate)]
        if len(d) < 30:
            continue
        dx, dy = offsets.get(fate, (0, 0))
        ax.text(
            float(d["spring_x"].median()) + dx,
            float(d["spring_y"].median()) + dy,
            fate.replace("Undifferentiated", "Undiff."),
            fontsize=6.7,
            color=base.FATE_PALETTE.get(fate, "#111827"),
            weight="bold",
            ha="center",
            va="center",
            path_effects=[pe.Stroke(linewidth=2.1, foreground="white", alpha=0.96), pe.Normal()],
            zorder=20,
        )


def scatter_growth(ax, df: pd.DataFrame, obs: pd.DataFrame, value: str, title: str, *, vmin: float, vmax: float, cmap) -> None:
    bg = obs.sample(n=min(len(obs), 26000), random_state=20260609)
    ax.scatter(bg["spring_x"], bg["spring_y"], s=0.8, color="#94A3B8", alpha=0.08, linewidth=0, zorder=0)
    sc = ax.scatter(
        df["spring_x"],
        df["spring_y"],
        c=df[value],
        s=2.6,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        alpha=0.82,
        linewidth=0,
        rasterized=True,
        zorder=5,
    )
    style_spring(ax, obs)
    ax.set_title(title, loc="left")
    return sc


def panel_observed_growth(obs: pd.DataFrame, growth: pd.DataFrame, fates: list[str]) -> None:
    cmap = mpl.colors.LinearSegmentedColormap.from_list("growth", ["#EFF6FF", "#93C5FD", "#F59E0B", "#B91C1C"])
    all_vals = growth[[f"growth_{f}" for f in fates]].to_numpy(float).reshape(-1)
    vmin, vmax = np.nanpercentile(all_vals, [2, 98])

    fig = plt.figure(figsize=(12.4, 6.7))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1.05], height_ratios=[1, 1], wspace=0.055, hspace=0.18)

    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    for ax, t in zip(axes, [0, 1, 2]):
        d = growth[growth["t"].eq(t)]
        sc = scatter_growth(
            ax,
            d,
            obs,
            "gate_weighted_growth",
            f"observed {TIME_LABELS[t]}\nmodel-weighted growth",
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
        )
        if t == 2:
            add_labels(ax, obs, [f for f in SHOW_BRANCHES + ["Undifferentiated"] if f in fates])

    branch_axes = [fig.add_subplot(gs[1, i]) for i in range(3)]
    branch_show = [f for f in ["Neutrophil", "Meg", "Mast"] if f in fates]
    for ax, fate in zip(branch_axes, branch_show):
        d = growth[growth["t"].eq(1)]
        sc = scatter_growth(
            ax,
            d,
            obs,
            f"growth_{fate}",
            f"{fate} branch growth\non observed day-4 cells",
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
        )

    ax_line = fig.add_subplot(gs[:, 3])
    summary_rows = []
    for t, d_t in growth.groupby("t"):
        for fate in [f for f in SHOW_BRANCHES if f in fates]:
            w = d_t[f"gate_{fate}"].to_numpy(float)
            g = d_t[f"growth_{fate}"].to_numpy(float)
            summary_rows.append(
                {
                    "t": int(t),
                    "time_label": TIME_LABELS.get(int(t), str(t)),
                    "branch_fate": fate,
                    "gate_weighted_mean_growth": float(np.sum(w * g) / max(float(w.sum()), 1e-12)),
                    "unweighted_mean_growth": float(np.mean(g)),
                    "mean_gate": float(np.mean(w)),
                }
            )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(TABLE_DIR / "clone_bmf_observed_growth_summary.csv", index=False)
    for fate in [f for f in SHOW_BRANCHES if f in fates]:
        d = summary[summary["branch_fate"].eq(fate)].sort_values("t")
        ax_line.plot(
            d["t"],
            d["gate_weighted_mean_growth"],
            marker="o",
            ms=4,
            lw=2,
            color=base.FATE_PALETTE.get(fate, "#64748B"),
            label=fate,
        )
    ax_line.set_xticks([0, 1, 2])
    ax_line.set_xticklabels(["day 2", "day 4", "day 6"])
    ax_line.set_ylabel("gate-weighted mean growth rate")
    ax_line.set_title("growth field summarized on observed cells", loc="left")
    ax_line.grid(axis="y", color="#E5E7EB", lw=0.75)
    ax_line.legend(frameon=False, fontsize=6.4, ncol=1, loc="upper right")
    sns.despine(ax=ax_line)

    cax = fig.add_axes([0.072, 0.075, 0.18, 0.017])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cb.set_label("instantaneous growth rate from BMF  d log mass / dt", fontsize=6.6, labelpad=1)
    cb.ax.tick_params(labelsize=5.8, pad=1)

    fig.suptitle(
        "BMF growth field evaluated directly on observed cells",
        x=0.02,
        y=0.995,
        ha="left",
        fontsize=13,
        weight="bold",
    )
    fig.text(
        0.02,
        0.02,
        "No rollout is used. Each observed cell stays at its measured SPRING coordinate; color is branch_growth_net evaluated at the observed latent state and time.",
        fontsize=7.1,
        color="#475569",
    )
    fig.subplots_adjust(left=0.035, right=0.985, top=0.88, bottom=0.12)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(FIG_DIR / f"panel_c_growthrate01_observed_cells_spring.{ext}", dpi=360 if ext == "png" else None)
    plt.close(fig)


def panel_observed_vs_clone_growth(obs: pd.DataFrame, cell_compare: pd.DataFrame, clone_compare: pd.DataFrame, corr: pd.DataFrame) -> None:
    cmap = mpl.colors.LinearSegmentedColormap.from_list("growth_compare", ["#2166AC", "#F8FAFC", "#B2182B"])
    fig = plt.figure(figsize=(12.5, 6.25))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 0.96], wspace=0.08, hspace=0.18)
    panel_defs = [
        (0, "predicted_growth_rank", "BMF predicted growth\non day-2 barcoded cells"),
        (0, "clone_derived_growth_rank", "barcode-derived clone growth\nday 2 -> day 4"),
        (1, "predicted_growth_rank", "BMF predicted growth\non day-4 barcoded cells"),
        (1, "clone_derived_growth_rank", "barcode-derived clone growth\nday 4 -> day 6"),
    ]
    axes = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(4)]
    for ax, (t, value, title) in zip(axes, panel_defs):
        d = cell_compare[cell_compare["t"].eq(t)].copy()
        bg = obs[obs["t"].eq(t)]
        if len(bg) > 22000:
            bg = bg.sample(n=22000, random_state=20260609 + t)
        ax.scatter(bg["spring_x"], bg["spring_y"], s=0.8, color="#94A3B8", alpha=0.08, linewidth=0, zorder=0)
        sc = ax.scatter(
            d["spring_x"],
            d["spring_y"],
            c=d[value],
            s=4.0,
            cmap=cmap,
            vmin=-1.0,
            vmax=1.0,
            alpha=0.82,
            linewidth=0,
            rasterized=True,
            zorder=5,
        )
        style_spring(ax, obs)
        ax.set_title(title, loc="left")
        ax.text(
            0.02,
            0.025,
            "ranked within interval",
            transform=ax.transAxes,
            fontsize=6.2,
            color="#334155",
            bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="none", alpha=0.82),
        )

    ax_scatter = fig.add_subplot(gs[:, 2])
    plot_df = clone_compare[clone_compare["n_current"].ge(2)].copy()
    for interval, marker in [("day 2->day 4", "o"), ("day 4->day 6", "s")]:
        d = plot_df[plot_df["interval"].eq(interval)]
        ax_scatter.scatter(
            d["predicted_growth_rank"],
            d["clone_derived_growth_rank"],
            s=np.clip(8 + 8 * np.sqrt(d["n_current"].to_numpy(float)), 14, 52),
            marker=marker,
            alpha=0.55,
            edgecolor="white",
            linewidth=0.45,
            label=interval,
        )
    ax_scatter.axhline(0, color="#CBD5E1", lw=0.9)
    ax_scatter.axvline(0, color="#CBD5E1", lw=0.9)
    ax_scatter.set_xlim(-1.05, 1.05)
    ax_scatter.set_ylim(-1.05, 1.05)
    ax_scatter.set_xlabel("BMF predicted growth rank")
    ax_scatter.set_ylabel("barcode-derived growth rank")
    ax_scatter.set_title("clone-level agreement", loc="left")
    ax_scatter.grid(color="#E5E7EB", lw=0.75)
    ax_scatter.legend(frameon=False, fontsize=6.6, loc="lower right")
    c = corr[(corr["interval"].eq("combined")) & (corr["min_current_clone_cells"].eq(2))]
    if not c.empty:
        ax_scatter.text(
            0.03,
            0.97,
            f"combined rank r = {float(c['rank_correlation'].iloc[0]):.2f}\n"
            f"n = {int(c['n_clone_time_points'].iloc[0])} clone-intervals\n"
            "filter: current clone cells >= 2",
            transform=ax_scatter.transAxes,
            ha="left",
            va="top",
            fontsize=7.0,
            color="#334155",
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#CBD5E1", alpha=0.88),
        )
    sns.despine(ax=ax_scatter)

    cax = fig.add_axes([0.07, 0.075, 0.17, 0.017])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cb.set_label("within-interval rank score", fontsize=6.6, labelpad=1)
    cb.ax.tick_params(labelsize=5.8, pad=1)

    fig.suptitle(
        "BMF growth field compared with barcode-derived clone expansion",
        x=0.02,
        y=0.995,
        ha="left",
        fontsize=13,
        weight="bold",
    )
    fig.text(
        0.02,
        0.02,
        "Barcode-derived growth is clone-level log fold-change in observed cell count across adjacent time points, assigned back to barcoded cells at the earlier time point.",
        fontsize=7.1,
        color="#475569",
    )
    fig.subplots_adjust(left=0.035, right=0.985, top=0.87, bottom=0.12)
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(FIG_DIR / f"panel_c_growthrate02_observed_vs_clone_growth.{ext}", dpi=360 if ext == "png" else None)
    plt.close(fig)


def main() -> None:
    setup_style()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    adata = ad.read_h5ad(base.ADATA_PATH)
    obs = base.load_obs(adata)
    fates = base.fate_names(pd.read_csv(base.BMF_TABLE))
    growth = evaluate_observed_growth(adata, obs, fates)
    panel_observed_growth(obs, growth, fates)
    cell_compare, clone_compare, corr = compute_clone_derived_growth(obs, growth)
    panel_observed_vs_clone_growth(obs, cell_compare, clone_compare, corr)
    print(
        {
            "n_observed_cells": int(len(growth)),
            "figure": str(FIG_DIR / "panel_c_growthrate01_observed_cells_spring.png"),
            "comparison_figure": str(FIG_DIR / "panel_c_growthrate02_observed_vs_clone_growth.png"),
            "tables": [
                str(TABLE_DIR / "clone_bmf_observed_cell_growth_spring.csv"),
                str(TABLE_DIR / "clone_bmf_observed_growth_summary.csv"),
                str(TABLE_DIR / "clone_barcode_derived_growth_by_clone_interval.csv"),
                str(TABLE_DIR / "clone_bmf_observed_vs_barcode_growth_cells.csv"),
                str(TABLE_DIR / "clone_bmf_observed_vs_barcode_growth_clone_level.csv"),
                str(TABLE_DIR / "clone_bmf_observed_vs_barcode_growth_correlation.csv"),
            ],
            "gate_weighted_growth_range": [
                float(growth["gate_weighted_growth"].min()),
                float(growth["gate_weighted_growth"].max()),
            ],
            "correlation": corr.to_dict(orient="records"),
        }
    )


if __name__ == "__main__":
    main()
