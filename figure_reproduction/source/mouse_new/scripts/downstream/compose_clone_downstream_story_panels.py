#!/usr/bin/env python3
"""Clone downstream story panels for the CellCompass case study.

The figures use locked downstream tables only.  No model is retrained.
The biological question is whether the branch-mass model turns day-2
expression states into a clone-fate probability landscape rather than a
single deterministic fate label.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cellcompass_mpl_cache")
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, Normalize
import anndata as ad
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[2]
TABLE_DIR = ROOT / "downstream/cellcompass_clone_case"
FINAL_DIR = ROOT / "downstream/clone_fate_spring_final"
FIG_DIR = ROOT / "figures/cellcompass_clone_case"

RUN_TABLES = {
    "BMF": FINAL_DIR / "clone_cuvgf_bmf_final_day2_fate_probabilities.csv",
    "Dynamical OT": FINAL_DIR / "dynamical_ot_claim_sota_day2_fate_probabilities.csv",
    "VGFM": FINAL_DIR / "vgfm_no_clone_anchor_day2_fate_probabilities.csv",
}

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
}

RUN_COLORS = {
    "BMF": "#6D28D9",
    "Dynamical OT": "#94A3B8",
    "VGFM": "#2C5AA0",
    "BMF top-1 collapse": "#F59E0B",
}

ADATA_PATH = Path("/Users/zhenyizhang/.cellcompass/algorithm_benchmarks/datasets/weinreb_rawrebuild_full_k10mindiff1/data.h5ad")
ALGO_PATH = Path("/Users/zhenyizhang/.cellcompass/training_algorithms/clone_cuvgf/algorithm.py")
FINAL_RUN_DIR = ROOT / "training_runs/20260607T113528987593Z__final__clone_cuvgf__campaign_20260-trial_20260607-weinreb_rawr"
FINAL_MODEL_STATE = FINAL_RUN_DIR / "artifacts/model_state.pt"
FINAL_RESOLVED_CONFIG = FINAL_RUN_DIR / "resolved_config.yaml"
TIME_LABELS = {0: "day 2", 1: "day 4", 2: "day 6"}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial"],
            "font.size": 8,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "mathtext.sf": "Arial",
            "mathtext.default": "regular",
            "axes.linewidth": 1.1,
            "axes.edgecolor": "#111827",
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
        }
    )


def fate_names(df: pd.DataFrame) -> list[str]:
    out = []
    for col in df.columns:
        if not col.startswith("truth_"):
            continue
        fate = col.removeprefix("truth_")
        if f"pred_{fate}" in df.columns:
            out.append(fate)
    return out


def normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(np.asarray(x, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    s = x.sum(axis=1, keepdims=True)
    return np.divide(x, s, out=np.zeros_like(x), where=s > 0)


def entropy(p: np.ndarray) -> np.ndarray:
    q = np.clip(p, 1e-12, 1.0)
    return -(q * np.log(q)).sum(axis=1)


def weighted_pearson(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    if int(mask.sum()) < 3:
        return np.nan
    x = x[mask]
    y = y[mask]
    w = w[mask] / w[mask].sum()
    mx = float(np.sum(w * x))
    my = float(np.sum(w * y))
    vx = float(np.sum(w * (x - mx) ** 2))
    vy = float(np.sum(w * (y - my) ** 2))
    if vx <= 1e-12 or vy <= 1e-12:
        return np.nan
    return float(np.sum(w * (x - mx) * (y - my)) / np.sqrt(vx * vy))


def load_run(name: str) -> dict:
    df = pd.read_csv(RUN_TABLES[name])
    fates = fate_names(df)
    mask = df["eligible"].astype(bool).to_numpy()
    truth = normalize_rows(df.loc[mask, [f"truth_{f}" for f in fates]].to_numpy(float))
    pred = normalize_rows(df.loc[mask, [f"pred_{f}" for f in fates]].to_numpy(float))
    weights = df.loc[mask, "metric_weight"].to_numpy(float)
    weights = weights / max(float(weights.sum()), 1e-12)
    return {
        "name": name,
        "df": df,
        "mask": mask,
        "fates": fates,
        "truth": truth,
        "pred": pred,
        "weights": weights,
        "truth_entropy": entropy(truth),
        "pred_entropy": entropy(pred),
        "truth_eff": np.exp(entropy(truth)),
        "pred_eff": np.exp(entropy(pred)),
    }


def spring_limits(df: pd.DataFrame) -> tuple[tuple[float, float], tuple[float, float]]:
    x = df["spring_x"].to_numpy(float)
    y = df["spring_y"].to_numpy(float)
    padx = 0.035 * (np.nanmax(x) - np.nanmin(x))
    pady = 0.035 * (np.nanmax(y) - np.nanmin(y))
    return (float(np.nanmin(x) - padx), float(np.nanmax(x) + padx)), (float(np.nanmin(y) - pady), float(np.nanmax(y) + pady))


def style_embedding_axis(ax, df: pd.DataFrame) -> None:
    (xmin, xmax), (ymin, ymax) = spring_limits(df)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def add_fate_labels(ax, df: pd.DataFrame, labels: np.ndarray, mask: np.ndarray, fates: list[str], *, min_n: int = 40) -> None:
    offsets = {
        "Undifferentiated": (-18, 28),
        "Neutrophil": (18, -34),
        "Monocyte": (28, 32),
        "Baso": (-34, -18),
        "Mast": (-18, -38),
        "Meg": (-28, 24),
        "Erythroid": (25, -22),
    }
    for fate in fates:
        sel = mask & (labels == fate)
        if int(sel.sum()) < min_n:
            continue
        x = float(np.nanmedian(df.loc[sel, "spring_x"]))
        y = float(np.nanmedian(df.loc[sel, "spring_y"]))
        dx, dy = offsets.get(fate, (0, 0))
        ax.text(
            x + dx,
            y + dy,
            fate.replace("Undifferentiated", "Undiff."),
            ha="center",
            va="center",
            fontsize=7.0,
            color=FATE_PALETTE.get(fate, "#111827"),
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.16,rounding_size=0.08", fc="white", ec="none", alpha=0.84),
        )


def dominant_labels(mat: np.ndarray, fates: list[str]) -> np.ndarray:
    idx = np.argmax(mat, axis=1)
    return np.asarray([fates[i] for i in idx], dtype=object)


def draw_dominant_map(ax, run: dict, mode: str) -> None:
    df = run["df"]
    mask = run["mask"]
    mat = run["truth"] if mode == "truth" else run["pred"]
    labels = dominant_labels(mat, run["fates"])
    full_labels = np.full(len(df), "", dtype=object)
    full_labels[mask] = labels
    ax.scatter(df["spring_x"], df["spring_y"], s=2.2, color="#CBD5E1", alpha=0.22, linewidth=0)
    for fate in run["fates"]:
        sel = mask & (full_labels == fate)
        if not np.any(sel):
            continue
        ax.scatter(
            df.loc[sel, "spring_x"],
            df.loc[sel, "spring_y"],
            s=5.0,
            color=FATE_PALETTE.get(fate, "#111827"),
            alpha=0.82,
            linewidth=0,
            label=fate,
        )
    style_embedding_axis(ax, df)
    add_fate_labels(ax, df, full_labels, mask, run["fates"], min_n=45)
    title = "clone-derived future fate" if mode == "truth" else "BMF branch fate"
    ax.set_title(title, loc="left")


def draw_entropy_map(ax, run: dict, mode: str) -> None:
    df = run["df"]
    mask = run["mask"]
    values = run["truth_eff"] if mode == "truth" else run["pred_eff"]
    full = np.full(len(df), np.nan)
    full[mask] = values
    cmap = LinearSegmentedColormap.from_list("eff_fates", ["#F8FAFC", "#C4B5FD", "#7C3AED", "#2E1065"])
    ax.scatter(df["spring_x"], df["spring_y"], s=2.2, color="#CBD5E1", alpha=0.18, linewidth=0)
    sc = ax.scatter(
        df.loc[mask, "spring_x"],
        df.loc[mask, "spring_y"],
        c=full[mask],
        s=5.4,
        cmap=cmap,
        vmin=1.0,
        vmax=5.3,
        alpha=0.90,
        linewidth=0,
    )
    style_embedding_axis(ax, df)
    ax.set_title("future-fate ambiguity" if mode == "truth" else "BMF branch mixture", loc="left")
    cb = plt.colorbar(sc, ax=ax, fraction=0.044, pad=0.012)
    cb.set_label("effective fates", fontsize=7)
    cb.ax.tick_params(labelsize=6)


def panel_c1_atlas(run: dict) -> None:
    fig = plt.figure(figsize=(12.3, 3.7))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.15, 1.15, 1.02, 1.02], wspace=0.10)
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    draw_dominant_map(axes[0], run, "truth")
    draw_dominant_map(axes[1], run, "pred")
    draw_entropy_map(axes[2], run, "truth")
    draw_entropy_map(axes[3], run, "pred")
    fig.suptitle("Day-2 progenitors carry clone-resolved future fate mixtures", x=0.02, y=1.04, ha="left", fontsize=14.5, weight="bold")
    fig.subplots_adjust(left=0.025, right=0.985, top=0.86, bottom=0.07)
    save_figure(fig, "panel_c1_clone_fate_atlas")


def draw_probability_field(ax, run: dict, fate: str) -> None:
    df = run["df"]
    mask = run["mask"]
    fidx = run["fates"].index(fate)
    values = run["pred"][:, fidx]
    color = FATE_PALETTE[fate]
    cmap = LinearSegmentedColormap.from_list(f"prob_{fate}", ["#F8FAFC", "#DDE8F0", color])
    ax.scatter(df["spring_x"], df["spring_y"], s=2.0, color="#CBD5E1", alpha=0.16, linewidth=0)
    sc = ax.scatter(
        df.loc[mask, "spring_x"],
        df.loc[mask, "spring_y"],
        c=values,
        s=5.8,
        cmap=cmap,
        norm=Normalize(vmin=0, vmax=max(0.35, float(np.nanpercentile(values, 98)))),
        alpha=0.92,
        linewidth=0,
    )
    style_embedding_axis(ax, df)
    ax.set_title(fate, color=color, loc="left", fontsize=9.2)
    cb = plt.colorbar(sc, ax=ax, fraction=0.047, pad=0.010)
    cb.set_label("BMF probability", fontsize=6.4)
    cb.ax.tick_params(labelsize=5.8)


def panel_c2_probability_fields(run: dict) -> None:
    fates = ["Undifferentiated", "Neutrophil", "Monocyte", "Baso", "Mast", "Meg", "Erythroid", "Eos"]
    fig = plt.figure(figsize=(11.2, 6.0))
    gs = fig.add_gridspec(2, 4, wspace=0.08, hspace=0.22)
    axes = [fig.add_subplot(gs[i // 4, i % 4]) for i in range(len(fates))]
    for ax, fate in zip(axes, fates):
        draw_probability_field(ax, run, fate)
    fig.suptitle("Branch model maps early expression to terminal-fate probability fields", x=0.02, y=0.995, ha="left", fontsize=14.5, weight="bold")
    fig.subplots_adjust(left=0.025, right=0.985, top=0.92, bottom=0.04)
    save_figure(fig, "panel_c2_bmf_branch_probability_fields")


def binned_entropy_calibration(runs: dict[str, dict]) -> pd.DataFrame:
    base = runs["BMF"]
    q = np.quantile(base["truth_entropy"], np.linspace(0, 1, 7))
    q[0] -= 1e-8
    rows = []
    for name, run in runs.items():
        idx = np.digitize(run["truth_entropy"], q[1:-1], right=True)
        for b in sorted(np.unique(idx)):
            sel = idx == b
            w = run["weights"][sel]
            w = w / max(float(w.sum()), 1e-12)
            rows.append(
                {
                    "run": name,
                    "bin": int(b),
                    "truth_effective_fates": float(np.sum(w * run["truth_eff"][sel])),
                    "pred_effective_fates": float(np.sum(w * run["pred_eff"][sel])),
                }
            )
    return pd.DataFrame(rows)


def ambiguity_strata(runs: dict[str, dict]) -> pd.DataFrame:
    truth_entropy = runs["BMF"]["truth_entropy"]
    cuts = np.quantile(truth_entropy, [0.0, 1 / 3, 2 / 3, 1.0])
    cuts[0] -= 1e-8
    labels = np.array(["low", "mid", "high"], dtype=object)
    strata = labels[np.digitize(truth_entropy, cuts[1:-1], right=True)]
    rows = []
    for name, run in runs.items():
        truth = run["truth"]
        pred = run["pred"]
        brier = ((truth - pred) ** 2).sum(axis=1)
        cosine_denom = np.linalg.norm(truth, axis=1) * np.linalg.norm(pred, axis=1)
        cosine = np.divide((truth * pred).sum(axis=1), cosine_denom, out=np.zeros(len(truth)), where=cosine_denom > 0)
        for label in labels:
            sel = strata == label
            w = run["weights"][sel]
            w = w / max(float(w.sum()), 1e-12)
            rows.append(
                {
                    "run": name,
                    "ambiguity": label,
                    "brier": float(np.sum(w * brier[sel])),
                    "cosine": float(np.sum(w * cosine[sel])),
                    "truth_effective_fates": float(np.sum(w * run["truth_eff"][sel])),
                    "n": int(sel.sum()),
                }
            )
    return pd.DataFrame(rows)


def collapse_diagnostic(run: dict) -> pd.DataFrame:
    truth = run["truth"]
    pred = run["pred"]
    weights = run["weights"]
    top1 = np.zeros_like(pred)
    top1[np.arange(pred.shape[0]), np.argmax(pred, axis=1)] = 1.0
    rows = []
    for label, yhat in [("BMF", pred), ("BMF top-1 collapse", top1)]:
        per_fate = [weighted_pearson(truth[:, j], yhat[:, j], weights) for j in range(truth.shape[1])]
        rows.append(
            {
                "variant": label,
                "CFCorr": float(np.nanmean(per_fate)),
                "Brier": float(np.sum(weights * ((truth - yhat) ** 2).sum(axis=1))),
                "effective_fates": float(np.sum(weights * np.exp(entropy(yhat)))),
            }
        )
    return pd.DataFrame(rows)


def per_fate_summary(run: dict) -> pd.DataFrame:
    rows = []
    for j, fate in enumerate(run["fates"]):
        rows.append(
            {
                "fate": fate,
                "truth_mean": float(np.sum(run["weights"] * run["truth"][:, j])),
                "pred_mean": float(np.sum(run["weights"] * run["pred"][:, j])),
                "correlation": weighted_pearson(run["truth"][:, j], run["pred"][:, j], run["weights"]),
            }
        )
    return pd.DataFrame(rows).sort_values("correlation", ascending=False)


def draw_entropy_calibration(ax, cal: pd.DataFrame) -> None:
    for name in ["VGFM", "Dynamical OT", "BMF"]:
        d = cal[cal["run"].eq(name)]
        ax.plot(
            d["truth_effective_fates"],
            d["pred_effective_fates"],
            marker="o",
            ms=4.0,
            lw=2.2 if name == "BMF" else 1.5,
            color=RUN_COLORS[name],
            label=name,
            alpha=0.96,
        )
    ax.plot([1, 5.4], [1, 5.4], color="#CBD5E1", lw=1.0, ls="--", zorder=0)
    ax.set_xlabel("clone truth: effective future fates")
    ax.set_ylabel("model prediction: effective fates")
    ax.set_title("branch model tracks clone uncertainty", loc="left")
    ax.set_xlim(1.0, 5.35)
    ax.set_ylim(0.95, 2.35)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.grid(color="#E5E7EB", lw=0.7)
    sns.despine(ax=ax)


def draw_ambiguity(ax, strata: pd.DataFrame) -> None:
    order = ["low", "mid", "high"]
    x = np.arange(len(order))
    width = 0.23
    for i, name in enumerate(["VGFM", "Dynamical OT", "BMF"]):
        d = strata[strata["run"].eq(name)].set_index("ambiguity").loc[order]
        ax.bar(x + (i - 1) * width, d["brier"], width=width, color=RUN_COLORS[name], label=name, alpha=0.95)
    ax.set_xticks(x)
    ax.set_xticklabels(["low\nambiguity", "mid", "high"])
    ax.set_ylabel("probability error\nlower is better")
    ax.set_title("gain is largest in ambiguous cells", loc="left")
    ax.grid(axis="y", color="#E5E7EB", lw=0.7)
    sns.despine(ax=ax)


def draw_collapse(ax, diag: pd.DataFrame) -> None:
    x = np.arange(len(diag))
    colors = [RUN_COLORS[v] for v in diag["variant"]]
    ax.bar(x, diag["CFCorr"], color=colors, width=0.58, edgecolor="white", linewidth=1.0)
    for i, row in diag.iterrows():
        ax.text(i, row["CFCorr"] + 0.018, f"{row['CFCorr']:.3f}", ha="center", va="bottom", fontsize=8)
        ax.text(i, 0.015, f"effective fates\n{row['effective_fates']:.2f}", ha="center", va="bottom", fontsize=6.2, color="#475569")
    ax.set_xticks(x)
    ax.set_xticklabels(["branch\nmixture", "forced\nsingle fate"])
    ax.set_ylim(0, 0.50)
    ax.set_ylabel("CFCorr\nhigher is better")
    ax.set_title("single-fate collapse loses signal", loc="left")
    ax.grid(axis="y", color="#E5E7EB", lw=0.7)
    sns.despine(ax=ax)


def draw_per_fate(ax, summary: pd.DataFrame) -> None:
    d = summary[summary["truth_mean"].gt(0.003)].copy()
    d = d.sort_values("correlation", ascending=True)
    y = np.arange(len(d))
    colors = [FATE_PALETTE.get(f, "#64748B") for f in d["fate"]]
    ax.hlines(y, 0, d["correlation"], color=colors, lw=2.3, alpha=0.92)
    ax.scatter(d["correlation"], y, s=38, color=colors, edgecolor="white", linewidth=0.8, zorder=3)
    ax.axvline(0, color="#CBD5E1", lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(d["fate"], fontsize=7)
    ax.set_xlim(-0.08, 0.86)
    ax.set_xlabel("truth-prediction correlation")
    ax.set_title("fate-specific recovery", loc="left")
    ax.grid(axis="x", color="#E5E7EB", lw=0.7)
    sns.despine(ax=ax, left=True)


def panel_c3_branch_mechanism(runs: dict[str, dict]) -> None:
    cal = binned_entropy_calibration(runs)
    strata = ambiguity_strata(runs)
    diag = collapse_diagnostic(runs["BMF"])
    pf = per_fate_summary(runs["BMF"])
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    cal.to_csv(TABLE_DIR / "clone_downstream_entropy_calibration.csv", index=False)
    strata.to_csv(TABLE_DIR / "clone_downstream_ambiguity_strata.csv", index=False)
    diag.to_csv(TABLE_DIR / "clone_downstream_branch_collapse.csv", index=False)
    pf.to_csv(TABLE_DIR / "clone_downstream_per_fate_summary.csv", index=False)

    fig = plt.figure(figsize=(13.5, 3.65))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.12, 1.00, 0.78, 1.02], wspace=0.52)
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    draw_entropy_calibration(axes[0], cal)
    draw_ambiguity(axes[1], strata)
    draw_collapse(axes[2], diag)
    draw_per_fate(axes[3], pf)
    fig.suptitle("Branch mixtures recover probabilistic clone fate", x=0.02, y=1.03, ha="left", fontsize=14.5, weight="bold")
    fig.subplots_adjust(left=0.045, right=0.985, top=0.84, bottom=0.16)
    save_figure(fig, "panel_c3_branch_mixture_evidence")


def clone_key_series(x: pd.Series) -> pd.Series:
    return x.astype(str).str.replace(r"\.0$", "", regex=True)


def selected_clone_mixtures(obs: pd.DataFrame, run: dict, selected: list[str]) -> pd.DataFrame:
    fates = run["fates"]
    bmf = run["df"].copy()
    bmf["clone_str"] = clone_key_series(bmf["clone"])
    rows = []
    for clone in selected:
        g_obs = obs[obs["clone_str"].eq(clone)]
        terminal = g_obs[g_obs["t"].eq(2)]
        if len(terminal) == 0:
            continue
        truth_counts = terminal["cell_type"].value_counts()
        truth = np.asarray([truth_counts.get(f, 0) for f in fates], dtype=float)
        truth = truth / max(float(truth.sum()), 1.0)
        g_pred = bmf[bmf["clone_str"].eq(clone) & bmf["eligible"].astype(bool)]
        if len(g_pred) > 0:
            w = g_pred["metric_weight"].to_numpy(float)
            w = w / max(float(w.sum()), 1e-12)
            pred = np.average(g_pred[[f"pred_{f}" for f in fates]].to_numpy(float), axis=0, weights=w)
            pred = pred / max(float(pred.sum()), 1e-12)
        else:
            pred = np.zeros(len(fates), dtype=float)
        for source, vec in [("observed descendants", truth), ("BMF from day-2 cells", pred)]:
            for fate, value in zip(fates, vec):
                rows.append(
                    {
                        "clone": clone,
                        "source": source,
                        "fate": fate,
                        "fraction": float(value),
                        "n_day2": int((g_obs["t"] == 0).sum()),
                        "n_day4": int((g_obs["t"] == 1).sum()),
                        "n_day6": int((g_obs["t"] == 2).sum()),
                    }
                )
    return pd.DataFrame(rows)


def draw_clone_trajectory_map(ax, obs: pd.DataFrame, selected: list[str], clone_colors: dict[str, str]) -> None:
    rng = np.random.default_rng(20260609)
    bg_idx = rng.choice(obs.index.to_numpy(), size=min(24000, len(obs)), replace=False)
    ax.scatter(obs.loc[bg_idx, "spring_x"], obs.loc[bg_idx, "spring_y"], s=1.4, color="#CBD5E1", alpha=0.12, linewidth=0)

    # Light terminal fate cloud makes the endpoint basins readable without competing
    # with clone paths.
    term = obs[obs["t"].eq(2)]
    for fate, color in FATE_PALETTE.items():
        d = term[term["cell_type"].eq(fate)]
        if len(d) == 0:
            continue
        sample = d.sample(n=min(len(d), 1100), random_state=sum(ord(c) for c in fate))
        ax.scatter(sample["spring_x"], sample["spring_y"], s=2.0, color=color, alpha=0.08, linewidth=0)

    label_offsets = {
        "2431": (-80, 65),
        "2484": (95, 75),
        "3530": (85, -95),
        "4141": (-115, 78),
    }
    for clone in selected:
        g = obs[obs["clone_str"].eq(clone)]
        color = clone_colors[clone]
        centroids = []
        for t in [0, 1, 2]:
            d = g[g["t"].eq(t)]
            if len(d) == 0:
                continue
            centroids.append((t, float(d["spring_x"].median()), float(d["spring_y"].median())))
        if len(centroids) >= 2:
            xs = [c[1] for c in centroids]
            ys = [c[2] for c in centroids]
            line = ax.plot(xs, ys, color=color, lw=2.7, alpha=0.96, zorder=10)[0]
            line.set_path_effects([pe.Stroke(linewidth=5.2, foreground="white", alpha=0.88), pe.Normal()])
            for i in range(len(xs) - 1):
                ax.annotate(
                    "",
                    xy=(xs[i + 1], ys[i + 1]),
                    xytext=(xs[i], ys[i]),
                    arrowprops=dict(arrowstyle="-|>", lw=2.2, color=color, shrinkA=4, shrinkB=4),
                    zorder=12,
                )
        for t, marker, size in [(0, "o", 60), (1, "s", 34), (2, "D", 46)]:
            d = g[g["t"].eq(t)]
            if len(d) == 0:
                continue
            ax.scatter(
                d["spring_x"],
                d["spring_y"],
                s=size,
                marker=marker,
                facecolor=color,
                edgecolor="white",
                linewidth=1.0 if t < 2 else 0.75,
                alpha=0.94,
                zorder=15,
            )
        if centroids:
            x0, y0 = centroids[0][1], centroids[0][2]
            dx, dy = label_offsets.get(clone, (0, 70))
            ax.text(
                x0 + dx,
                y0 + dy,
                f"clone {clone}",
                color=color,
                fontsize=7.2,
                weight="bold",
                ha="center",
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.13", fc="white", ec="none", alpha=0.78),
                zorder=20,
            )

    style_embedding_axis(ax, obs.rename(columns={"spring_x": "spring_x", "spring_y": "spring_y"}))
    ax.set_title("observed barcode descendants", loc="left")
    ax.text(
        0.015,
        0.02,
        "one clone = same barcode across many cells\n"
        "solid line = median position per day\n"
        "circles: day 2  |  squares: day 4  |  diamonds: day 6\n"
        "point color = barcode clone identity, not fate",
        transform=ax.transAxes,
        fontsize=7.0,
        color="#334155",
        bbox=dict(boxstyle="round,pad=0.26", fc="white", ec="#CBD5E1", alpha=0.86),
    )


def load_bmf_model(latent_dim: int, n_fates: int):
    import torch
    import yaml

    # The training algorithm imports CytoBridge, whose plotting namespace imports
    # cellrank.  The saved BMF model class does not use plotting, so provide a
    # minimal placeholder when cellrank is not installed.
    sys.modules.setdefault("cellrank", types.ModuleType("cellrank"))
    spec = importlib.util.spec_from_file_location("clone_cuvgf_algorithm_for_story", str(ALGO_PATH))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    with FINAL_RESOLVED_CONFIG.open() as f:
        resolved = yaml.safe_load(f)
    model = mod.BranchDynamicalModel(latent_dim, resolved["model"], n_fates=n_fates).to(torch.device("cpu"))
    state_obj = torch.load(FINAL_MODEL_STATE, map_location="cpu", weights_only=False)
    state = {
        k: (torch.as_tensor(v) if isinstance(v, np.ndarray) else v)
        for k, v in state_obj["all_model"]["model_state_dict"].items()
    }
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, resolved


def project_latent_to_spring(query: np.ndarray, ref_latent: np.ndarray, ref_spring: np.ndarray, *, k: int = 25) -> np.ndarray:
    tree = cKDTree(np.asarray(ref_latent, dtype=np.float32))
    d, idx = tree.query(np.asarray(query, dtype=np.float32), k=min(k, len(ref_latent)))
    if idx.ndim == 1:
        d = d[:, None]
        idx = idx[:, None]
    w = 1.0 / (np.asarray(d, dtype=float) + 1e-4)
    w = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
    return np.einsum("ij,ijk->ik", w, ref_spring[np.asarray(idx, dtype=int)])


def project_snapshot_to_time_matched_spring(
    query: np.ndarray,
    obs: pd.DataFrame,
    latent: np.ndarray,
    model_time: float,
    *,
    k: int = 25,
) -> np.ndarray:
    """Project latent states onto the observed SPRING map using matched days.

    The BMF state is integrated in latent space.  For display, day-2-like,
    day-4-like and day-6-like states are projected with the corresponding
    observed reference cells instead of a pooled all-day reference, which makes
    terminal branch endpoints easier to read against day-6 fate basins.
    """
    ref_t = int(np.clip(np.rint(float(model_time)), 0, 2))
    ref_mask = obs["t"].eq(ref_t).to_numpy()
    if int(ref_mask.sum()) < k:
        ref_mask = np.ones(len(obs), dtype=bool)
    return project_latent_to_spring(
        np.asarray(query, dtype=np.float32),
        np.asarray(latent[ref_mask], dtype=np.float32),
        obs.loc[ref_mask, ["spring_x", "spring_y"]].to_numpy(float),
        k=k,
    )


def terminal_fate_probabilities(
    query: np.ndarray,
    obs: pd.DataFrame,
    latent: np.ndarray,
    fates: list[str],
    *,
    k: int = 50,
) -> np.ndarray:
    """KNN terminal phenotype probabilities for generated day-6 latent states."""
    ref_mask = obs["t"].eq(2).to_numpy()
    ref_latent = np.asarray(latent[ref_mask], dtype=np.float32)
    ref_labels = obs.loc[ref_mask, "cell_type"].astype(str).to_numpy()
    code = {f: i for i, f in enumerate(fates)}
    ref_codes = np.asarray([code.get(x, -1) for x in ref_labels], dtype=int)
    keep = ref_codes >= 0
    ref_latent = ref_latent[keep]
    ref_codes = ref_codes[keep]
    tree = cKDTree(ref_latent)
    d, idx = tree.query(np.asarray(query, dtype=np.float32), k=min(k, len(ref_latent)))
    if idx.ndim == 1:
        d = d[:, None]
        idx = idx[:, None]
    weights = 1.0 / (np.asarray(d, dtype=float) + 1e-4)
    weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    out = np.zeros((idx.shape[0], len(fates)), dtype=float)
    for row in range(idx.shape[0]):
        np.add.at(out[row], ref_codes[idx[row]], weights[row])
    return normalize_rows(out)


def compute_bmf_branch_paths(adata, obs: pd.DataFrame, selected: list[str], fates: list[str]) -> pd.DataFrame:
    import torch

    latent = np.asarray(adata.obsm["X_latent"], dtype=np.float32)
    ref_spring = obs[["spring_x", "spring_y"]].to_numpy(float)
    model, resolved = load_bmf_model(latent.shape[1], len(fates))
    integration_dt = float(((resolved.get("evaluation") or {}).get("trajectory_integration_dt") or 0.0125))
    rows = []
    eye = torch.eye(len(fates), dtype=torch.float32)
    with torch.no_grad():
        for clone in selected:
            source_positions = np.where(obs["clone_str"].eq(clone).to_numpy() & obs["t"].eq(0).to_numpy())[0]
            if len(source_positions) == 0:
                continue
            x0 = torch.as_tensor(latent[source_positions], dtype=torch.float32)
            t0_col = torch.zeros((x0.shape[0], 1), dtype=torch.float32)
            gate = torch.softmax(model.gate_net(torch.cat([x0, t0_col], dim=1)), dim=1)
            x = x0.repeat_interleave(len(fates), dim=0).contiguous()
            branch_onehot = eye.repeat(x0.shape[0], 1).contiguous()
            source_rep = np.repeat(source_positions, len(fates))
            branch_rep = np.tile(np.arange(len(fates)), x0.shape[0])
            logw = torch.log((gate.reshape(-1) / max(float(x0.shape[0]), 1.0)).clamp_min(1e-12)).reshape(-1, 1)

            snapshot_times = np.linspace(0.0, 2.0, 9)
            snapshots = {0.0: x.cpu().numpy().copy()}
            current_t = 0.0
            for target_t in snapshot_times[1:]:
                while current_t < target_t - 1e-9:
                    dt = min(integration_dt, target_t - current_t)
                    t_col = torch.full((x.shape[0], 1), current_t, dtype=x.dtype)
                    inp = torch.cat([x, t_col, branch_onehot], dim=1)
                    x = x + model.branch_velocity_net(inp) * dt
                    logw = logw + model.branch_growth_net(inp) * dt
                    current_t += dt
                snapshots[target_t] = x.cpu().numpy().copy()

            endpoint_probs = terminal_fate_probabilities(snapshots[2.0], obs, latent, fates, k=50)
            final_weights = torch.exp(logw).cpu().numpy().reshape(-1)
            final_weights = final_weights / max(float(final_weights.sum()), 1e-12)
            ordered_times = list(snapshot_times)
            chunks = [
                project_snapshot_to_time_matched_spring(snapshots[t], obs, latent, t, k=25)
                for t in ordered_times
            ]
            # Anchor model t0 exactly at the observed day-2 SPRING coordinates.
            chunks[0] = obs.iloc[source_rep][["spring_x", "spring_y"]].to_numpy(float)
            for particle_i, (source_pos, branch_idx, branch_w) in enumerate(zip(source_rep, branch_rep, final_weights)):
                terminal_idx = int(np.argmax(endpoint_probs[particle_i]))
                prob_payload = {f"endpoint_prob::{f}": float(endpoint_probs[particle_i, j]) for j, f in enumerate(fates)}
                for time_idx, t in enumerate(ordered_times):
                    rows.append(
                        {
                            "clone": clone,
                            "source_pos": int(source_pos),
                            "branch_fate": fates[int(branch_idx)],
                            "terminal_fate": fates[terminal_idx],
                            "terminal_fate_conf": float(endpoint_probs[particle_i, terminal_idx]),
                            "branch_weight": float(branch_w),
                            "model_time": float(t),
                            "spring_x": float(chunks[time_idx][particle_i, 0]),
                            "spring_y": float(chunks[time_idx][particle_i, 1]),
                            **prob_payload,
                        }
                    )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["rank"] = out.groupby(["clone", "source_pos"])["branch_weight"].rank(method="first", ascending=False)
    return out


def draw_bmf_rollout_map(ax, obs: pd.DataFrame, paths: pd.DataFrame, selected: list[str], clone_colors: dict[str, str]) -> None:
    rng = np.random.default_rng(20260610)
    bg_idx = rng.choice(obs.index.to_numpy(), size=min(22000, len(obs)), replace=False)
    ax.scatter(obs.loc[bg_idx, "spring_x"], obs.loc[bg_idx, "spring_y"], s=1.4, color="#CBD5E1", alpha=0.12, linewidth=0)

    term = obs[obs["t"].eq(2)]
    for fate, color in FATE_PALETTE.items():
        d = term[term["cell_type"].eq(fate)]
        if len(d) == 0:
            continue
        sample = d.sample(n=min(len(d), 900), random_state=sum(ord(c) for c in fate) + 17)
        ax.scatter(sample["spring_x"], sample["spring_y"], s=2.0, color=color, alpha=0.055, linewidth=0)

    path_weights = paths.drop_duplicates(["clone", "source_pos", "branch_fate"])[
        ["clone", "source_pos", "branch_fate", "branch_weight", "terminal_fate", "terminal_fate_conf"]
    ].copy()
    path_weights["rank_within_cell"] = path_weights.groupby(["clone", "source_pos"])["branch_weight"].rank(
        method="first", ascending=False
    )
    keep_keys = path_weights[(path_weights["rank_within_cell"].le(3)) & (path_weights["branch_weight"].ge(0.004))][
        ["clone", "source_pos", "branch_fate"]
    ]
    visible = paths.merge(keep_keys, on=["clone", "source_pos", "branch_fate"], how="inner")
    n_day2_by_clone = path_weights.groupby("clone")["source_pos"].nunique().to_dict()

    for clone in selected:
        clone_paths = visible[visible["clone"].eq(clone)].copy()
        if clone_paths.empty:
            continue
        n_day2 = max(int(n_day2_by_clone.get(clone, 1)), 1)
        for (source_pos, branch_fate), d in clone_paths.groupby(["source_pos", "branch_fate"]):
            d = d.sort_values("model_time")
            fate = str(d["terminal_fate"].iloc[0])
            color = FATE_PALETTE.get(fate, "#64748B")
            weight = float(d["branch_weight"].iloc[0])
            per_cell_prob = float(np.clip(weight * n_day2, 0.0, 1.0))
            lw = 0.35 + 2.9 * np.sqrt(per_cell_prob)
            alpha = 0.18 + 0.58 * np.sqrt(per_cell_prob)
            line = ax.plot(d["spring_x"], d["spring_y"], color=color, lw=lw, alpha=alpha, zorder=12)[0]
            line.set_path_effects([pe.Stroke(linewidth=lw + 1.2, foreground="white", alpha=0.72), pe.Normal()])
            if len(d) >= 3 and per_cell_prob >= 0.08:
                i = len(d) - 3
                ax.annotate(
                    "",
                    xy=(d.iloc[i + 1]["spring_x"], d.iloc[i + 1]["spring_y"]),
                    xytext=(d.iloc[i]["spring_x"], d.iloc[i]["spring_y"]),
                    arrowprops=dict(arrowstyle="-|>", lw=max(0.8, lw * 0.52), color=color, alpha=alpha, shrinkA=2, shrinkB=2),
                    zorder=18,
                )
            ax.scatter(
                d["spring_x"].iloc[1:-1:2],
                d["spring_y"].iloc[1:-1:2],
                s=4.5 + 16 * per_cell_prob,
                marker="o",
                facecolor=color,
                edgecolor="none",
                alpha=min(0.62, alpha),
                zorder=14,
            )
            end = d.iloc[-1]
            ax.scatter(
                end["spring_x"],
                end["spring_y"],
                s=18 + 95 * per_cell_prob,
                marker="o",
                facecolor=color,
                edgecolor="white",
                linewidth=0.55,
                alpha=0.88,
                zorder=22,
            )

    for clone in selected:
        d0 = obs[obs["clone_str"].eq(clone) & obs["t"].eq(0)]
        ax.scatter(
            d0["spring_x"],
            d0["spring_y"],
            s=72,
            marker="o",
            facecolor=clone_colors[clone],
            edgecolor="white",
            linewidth=1.2,
            alpha=0.97,
            zorder=20,
        )
    style_embedding_axis(ax, obs)
    ax.set_title("individual BMF branch paths from day-2 barcode cells", loc="left")
    ax.text(
        0.015,
        0.02,
        "same day-2 cells, model rollout only\n"
        "each line = one cell on one branch\n"
        "small dots = model states to day 6\n"
        "color = endpoint phenotype neighborhood\n"
        "line width = branch probability",
        transform=ax.transAxes,
        fontsize=7.0,
        color="#334155",
        bbox=dict(boxstyle="round,pad=0.26", fc="white", ec="#CBD5E1", alpha=0.86),
    )


def draw_combined_clone_bmf_map(
    ax,
    obs: pd.DataFrame,
    paths: pd.DataFrame,
    selected: list[str],
    clone_markers: dict[str, str],
) -> None:
    rng = np.random.default_rng(20260611)
    term = obs[obs["t"].eq(2)]
    for fate, color in FATE_PALETTE.items():
        d = term[term["cell_type"].eq(fate)]
        if len(d) == 0:
            continue
        sample = d.sample(n=min(len(d), 1400), random_state=sum(ord(c) for c in fate) + 31)
        ax.scatter(sample["spring_x"], sample["spring_y"], s=2.0, color=color, alpha=0.06, linewidth=0, zorder=1)

    path_weights = paths.drop_duplicates(["clone", "source_pos", "branch_fate"])[
        ["clone", "source_pos", "branch_fate", "branch_weight", "terminal_fate"]
    ].copy()
    path_weights["rank_within_cell"] = path_weights.groupby(["clone", "source_pos"])["branch_weight"].rank(
        method="first", ascending=False
    )
    keep_keys = path_weights[(path_weights["rank_within_cell"].le(3)) & (path_weights["branch_weight"].ge(0.004))][
        ["clone", "source_pos", "branch_fate"]
    ]
    visible = paths.merge(keep_keys, on=["clone", "source_pos", "branch_fate"], how="inner")
    n_day2_by_clone = path_weights.groupby("clone")["source_pos"].nunique().to_dict()

    for clone in selected:
        marker = clone_markers[clone]
        g = obs[obs["clone_str"].eq(clone)]
        for fate, d in g.groupby("cell_type"):
            ax.scatter(
                d["spring_x"],
                d["spring_y"],
                s=32 if len(d) > 1 else 46,
                marker=marker,
                facecolor=FATE_PALETTE.get(fate, "#64748B"),
                edgecolor="white",
                linewidth=0.65,
                alpha=0.88,
                zorder=15,
            )

        d0 = g[g["t"].eq(0)]
        if len(d0) > 0:
            x0 = float(d0["spring_x"].median())
            y0 = float(d0["spring_y"].median())
            ax.scatter(
                d0["spring_x"],
                d0["spring_y"],
                s=86,
                marker=marker,
                facecolor="white",
                edgecolor="#111827",
                linewidth=1.3,
                alpha=0.98,
                zorder=25,
            )
            ax.text(
                x0,
                y0 + 35,
                f"clone {clone}",
                ha="center",
                va="bottom",
                fontsize=7.0,
                color="#111827",
                weight="bold",
                bbox=dict(boxstyle="round,pad=0.11", fc="white", ec="none", alpha=0.75),
                zorder=28,
            )

        clone_paths = visible[visible["clone"].eq(clone)].copy()
        n_day2 = max(int(n_day2_by_clone.get(clone, 1)), 1)
        for (source_pos, branch_fate), d in clone_paths.groupby(["source_pos", "branch_fate"]):
            d = d.sort_values("model_time")
            fate = str(d["terminal_fate"].iloc[0])
            color = FATE_PALETTE.get(fate, "#64748B")
            weight = float(d["branch_weight"].iloc[0])
            per_cell_prob = float(np.clip(weight * n_day2, 0.0, 1.0))
            lw = 0.35 + 2.55 * np.sqrt(per_cell_prob)
            alpha = 0.16 + 0.50 * np.sqrt(per_cell_prob)
            model_line = ax.plot(d["spring_x"], d["spring_y"], color=color, lw=lw, alpha=alpha, zorder=11)[0]
            model_line.set_path_effects([pe.Stroke(linewidth=lw + 1.0, foreground="white", alpha=0.68), pe.Normal()])
            end = d.iloc[-1]
            ax.scatter(
                end["spring_x"],
                end["spring_y"],
                s=34 + 120 * per_cell_prob,
                marker=marker,
                facecolor="white",
                edgecolor=color,
                linewidth=1.1,
                alpha=0.96,
                zorder=23,
            )

    style_embedding_axis(ax, obs)
    ax.set_title("observed descendants and BMF branches on the same fate map", loc="left")


def draw_clone_mixture_bars(ax, mix: pd.DataFrame, selected: list[str], clone_colors: dict[str, str]) -> None:
    fates = [f for f in FATE_PALETTE if f in set(mix["fate"])]
    y_positions = {}
    labels = []
    y = 0
    for clone in selected:
        d = mix[mix["clone"].eq(clone)]
        if d.empty:
            continue
        n0 = int(d["n_day2"].iloc[0])
        n2 = int(d["n_day6"].iloc[0])
        obs_vec = d[d["source"].eq("observed descendants")]["fraction"].to_numpy(float)
        obs_vec = obs_vec[obs_vec > 0]
        eff = float(np.exp(-np.sum(obs_vec * np.log(np.clip(obs_vec, 1e-12, 1.0))))) if len(obs_vec) else 1.0
        labels.append((y + 0.42, f"clone {clone}\n{n0} day-2 | {n2} day-6\neff={eff:.1f}"))
        for source in ["observed descendants", "BMF from day-2 cells"]:
            y_positions[(clone, source)] = y
            y += 0.48
        y += 0.22

    for (clone, source), yy in y_positions.items():
        d = mix[mix["clone"].eq(clone) & mix["source"].eq(source)].set_index("fate")
        left = 0.0
        for fate in fates:
            value = float(d["fraction"].get(fate, 0.0))
            if value <= 0:
                continue
            ax.barh(yy, value, left=left, height=0.32, color=FATE_PALETTE[fate], edgecolor="white", linewidth=0.45)
            left += value
        ax.text(
            -0.018,
            yy,
            "obs" if source.startswith("observed") else "BMF",
            ha="right",
            va="center",
            fontsize=7.0,
            color="#111827",
            weight="bold" if not source.startswith("observed") else "normal",
        )
    for yy, label in labels:
        ax.text(
            -0.19,
            yy,
            label,
            ha="right",
            va="center",
            fontsize=7.1,
            color="#111827",
            weight="bold",
        )

    ax.set_xlim(-0.20, 1.0)
    ax.set_ylim(-0.35, max(y_positions.values()) + 0.42)
    ax.invert_yaxis()
    ax.set_yticks([])
    ax.set_xlabel("terminal fate fraction")
    ax.set_title("barcode truth versus BMF branch mixture", loc="left")
    ax.grid(axis="x", color="#E5E7EB", lw=0.7)
    sns.despine(ax=ax, left=True)


def panel_c4_clone_trajectories(run: dict) -> None:
    # Main-text case studies: representative well-recovered real barcodes that
    # span a near-single fate, a two-fate mixture and broader fate mixtures.
    # Systematic non-case-study performance is shown in panel C3.
    selected = ["2431", "2484", "3530", "4061"]
    clone_markers = {
        "2431": "o",
        "2484": "s",
        "3530": "D",
        "4061": "^",
    }
    adata = ad.read_h5ad(ADATA_PATH)
    obs = adata.obs[["clone", "time_point_processed", "Cell type annotation", "SPRING-x", "SPRING-y"]].copy()
    obs["clone_str"] = clone_key_series(obs["clone"])
    obs["t"] = obs["time_point_processed"].astype(int)
    obs["cell_type"] = obs["Cell type annotation"].astype(str)
    obs = obs.rename(columns={"SPRING-x": "spring_x", "SPRING-y": "spring_y"})

    mix = selected_clone_mixtures(obs, run, selected)
    paths = compute_bmf_branch_paths(adata, obs, selected, run["fates"])
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    mix.to_csv(TABLE_DIR / "clone_downstream_selected_clone_mixtures.csv", index=False)
    paths.to_csv(TABLE_DIR / "clone_downstream_selected_bmf_branch_paths.csv", index=False)

    fig = plt.figure(figsize=(13.9, 5.75))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.52, 0.82], wspace=0.14)
    ax_map = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[0, 1])
    draw_combined_clone_bmf_map(ax_map, obs, paths, selected, clone_markers)
    shape_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=clone_markers[clone],
            color="none",
            markerfacecolor="white",
            markeredgecolor="#111827",
            markeredgewidth=1.1,
            markersize=6.2,
            label=f"clone {clone}",
        )
        for clone in selected
    ]
    ax_map.legend(
        handles=shape_handles,
        title="barcode shape",
        frameon=False,
        fontsize=6.2,
        title_fontsize=6.6,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.985),
        ncol=2,
        columnspacing=0.7,
        handletextpad=0.35,
    )
    draw_clone_mixture_bars(ax_bar, mix, selected, {})
    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markeredgecolor="white", markersize=6, label=fate)
        for fate, color in FATE_PALETTE.items()
        if fate in {"Undifferentiated", "Neutrophil", "Monocyte", "Baso", "Mast", "Meg"}
    ]
    ax_bar.legend(
        handles=handles,
        title="day-6 fates",
        frameon=False,
        fontsize=6.2,
        title_fontsize=6.6,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.02),
        ncol=2,
        columnspacing=0.6,
        handletextpad=0.35,
    )
    fig.suptitle("BMF recovers clone fate breadth from early barcode states", x=0.02, y=0.985, ha="left", fontsize=14.8, weight="bold")
    fig.subplots_adjust(left=0.025, right=0.985, top=0.90, bottom=0.06)
    save_figure(fig, "panel_c4_larry_clone_trajectory_case_studies")


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        out = FIG_DIR / f"{stem}.{ext}"
        fig.savefig(out, dpi=360 if ext == "png" else None)
        print(out)
    plt.close(fig)


def main() -> None:
    setup_style()
    runs = {name: load_run(name) for name in RUN_TABLES}
    panel_c1_atlas(runs["BMF"])
    panel_c2_probability_fields(runs["BMF"])
    panel_c3_branch_mechanism(runs)
    panel_c4_clone_trajectories(runs["BMF"])


if __name__ == "__main__":
    main()
