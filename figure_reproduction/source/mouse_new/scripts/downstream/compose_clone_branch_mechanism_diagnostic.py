#!/usr/bin/env python3
"""Post-hoc diagnostics for why branch structure helps clone-fate recovery."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cellcompass_mpl_cache")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "figures/cellcompass_clone_case"
TABLE_DIR = ROOT / "downstream/cellcompass_clone_case"

RUN_TABLES = {
    "BMF": ROOT / "downstream/clone_fate_spring_final/clone_cuvgf_bmf_final_day2_fate_probabilities.csv",
    "Dynamical OT": ROOT / "downstream/clone_fate_spring_final/dynamical_ot_claim_sota_day2_fate_probabilities.csv",
    "VGFM": ROOT / "downstream/clone_fate_spring_final/vgfm_no_clone_anchor_day2_fate_probabilities.csv",
}

PALETTE = {
    "BMF": "#6D28D9",
    "Dynamical OT": "#94A3B8",
    "VGFM": "#2C5AA0",
    "BMF top-1 collapse": "#F59E0B",
}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.linewidth": 1.1,
            "axes.edgecolor": "#111827",
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def fate_names(df: pd.DataFrame) -> list[str]:
    names = []
    for col in df.columns:
        if col.startswith("truth_"):
            name = col.removeprefix("truth_")
            if f"pred_{name}" in df.columns:
                names.append(name)
    return names


def normalize_rows(x: np.ndarray) -> np.ndarray:
    s = x.sum(axis=1, keepdims=True)
    return np.divide(x, s, out=np.zeros_like(x), where=s > 0)


def entropy(p: np.ndarray) -> np.ndarray:
    q = np.clip(p, 1e-12, 1.0)
    return -(q * np.log(q)).sum(axis=1)


def weighted_pearson(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    mx = float(np.sum(w * x))
    my = float(np.sum(w * y))
    vx = float(np.sum(w * (x - mx) ** 2))
    vy = float(np.sum(w * (y - my) ** 2))
    if vx <= 1e-12 or vy <= 1e-12:
        return np.nan
    return float(np.sum(w * (x - mx) * (y - my)) / np.sqrt(vx * vy))


def macro_cfcorr(truth: np.ndarray, pred: np.ndarray, weights: np.ndarray) -> float:
    vals = [weighted_pearson(truth[:, j], pred[:, j], weights) for j in range(truth.shape[1])]
    return float(np.nanmean(vals))


def brier(truth: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return ((truth - pred) ** 2).sum(axis=1)


def cosine(truth: np.ndarray, pred: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(truth, axis=1) * np.linalg.norm(pred, axis=1)
    return np.divide((truth * pred).sum(axis=1), denom, out=np.zeros(truth.shape[0]), where=denom > 0)


def load_run(name: str) -> dict:
    df = pd.read_csv(RUN_TABLES[name])
    fates = fate_names(df)
    mask = df["eligible"].astype(bool).to_numpy()
    weights = df.loc[mask, "metric_weight"].to_numpy(float)
    weights = weights / weights.sum()
    truth = normalize_rows(np.nan_to_num(df.loc[mask, [f"truth_{f}" for f in fates]].to_numpy(float)))
    pred = normalize_rows(np.nan_to_num(df.loc[mask, [f"pred_{f}" for f in fates]].to_numpy(float)))
    out = {
        "name": name,
        "df": df,
        "eligible_mask": mask,
        "weights": weights,
        "truth": truth,
        "pred": pred,
        "truth_entropy": entropy(truth),
        "pred_entropy": entropy(pred),
        "truth_effective_fates": np.exp(entropy(truth)),
        "pred_effective_fates": np.exp(entropy(pred)),
        "brier": brier(truth, pred),
        "cosine": cosine(truth, pred),
        "fates": fates,
    }
    return out


def binned_entropy_calibration(runs: dict[str, dict]) -> pd.DataFrame:
    bmf = runs["BMF"]
    q = np.quantile(bmf["truth_entropy"], np.linspace(0, 1, 7))
    q[0] -= 1e-8
    rows = []
    for name, data in runs.items():
        idx = np.digitize(data["truth_entropy"], q[1:-1], right=True)
        for b in sorted(np.unique(idx)):
            sel = idx == b
            w = data["weights"][sel]
            w = w / w.sum()
            rows.append(
                {
                    "run": name,
                    "bin": int(b),
                    "truth_entropy": float(np.sum(w * data["truth_entropy"][sel])),
                    "pred_entropy": float(np.sum(w * data["pred_entropy"][sel])),
                    "truth_effective_fates": float(np.sum(w * data["truth_effective_fates"][sel])),
                    "pred_effective_fates": float(np.sum(w * data["pred_effective_fates"][sel])),
                }
            )
    return pd.DataFrame(rows)


def ambiguity_strata(runs: dict[str, dict]) -> pd.DataFrame:
    truth_entropy = runs["BMF"]["truth_entropy"]
    cuts = np.quantile(truth_entropy, [0, 1 / 3, 2 / 3, 1])
    cuts[0] -= 1e-8
    labels = np.array(["low", "mid", "high"])
    strata = labels[np.digitize(truth_entropy, cuts[1:-1], right=True)]
    rows = []
    for name, data in runs.items():
        for label in labels:
            sel = strata == label
            w = data["weights"][sel]
            w = w / w.sum()
            rows.append(
                {
                    "run": name,
                    "ambiguity": label,
                    "brier": float(np.sum(w * data["brier"][sel])),
                    "cosine": float(np.sum(w * data["cosine"][sel])),
                    "truth_effective_fates": float(np.sum(w * data["truth_effective_fates"][sel])),
                    "n": int(sel.sum()),
                }
            )
    return pd.DataFrame(rows)


def collapse_diagnostic(bmf: dict) -> pd.DataFrame:
    truth = bmf["truth"]
    pred = bmf["pred"]
    w = bmf["weights"]
    top1 = np.zeros_like(pred)
    top1[np.arange(pred.shape[0]), np.argmax(pred, axis=1)] = 1.0
    rows = []
    for label, yhat in [("BMF", pred), ("BMF top-1 collapse", top1)]:
        rows.append(
            {
                "variant": label,
                "CFCorr": macro_cfcorr(truth, yhat, w),
                "Brier": float(np.sum(w * brier(truth, yhat))),
                "prediction_entropy": float(np.sum(w * entropy(yhat))),
                "effective_fates": float(np.sum(w * np.exp(entropy(yhat)))),
            }
        )
    return pd.DataFrame(rows)


def draw_spring_entropy(ax, bmf: dict) -> None:
    df = bmf["df"]
    mask = bmf["eligible_mask"]
    x = df["spring_x"].to_numpy(float)
    y = df["spring_y"].to_numpy(float)
    ax.scatter(x[~mask], y[~mask], s=2.2, color="#CBD5E1", alpha=0.24, linewidth=0)
    eff = np.full(len(df), np.nan)
    eff[mask] = bmf["truth_effective_fates"]
    cmap = LinearSegmentedColormap.from_list("fate_entropy", ["#F8FAFC", "#C4B5FD", "#6D28D9", "#2E1065"])
    sc = ax.scatter(x[mask], y[mask], c=eff[mask], s=4.5, cmap=cmap, vmin=1.0, vmax=6.2, alpha=0.88, linewidth=0)
    ax.set_title("Clone truth is locally multi-fate", loc="left")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("day-2 expression neighborhood")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(False)
    cb = plt.colorbar(sc, ax=ax, fraction=0.040, pad=0.015)
    cb.set_label("effective future fates", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    ax.text(
        0.02,
        0.03,
        "color = exp(entropy) of clone-derived\nday-6 fate distribution",
        transform=ax.transAxes,
        fontsize=6.6,
        color="#334155",
        bbox=dict(boxstyle="round,pad=0.24", fc="white", ec="none", alpha=0.82),
    )


def draw_entropy_calibration(ax, cal: pd.DataFrame) -> None:
    for name in ["VGFM", "Dynamical OT", "BMF"]:
        d = cal[cal["run"].eq(name)]
        ax.plot(
            d["truth_effective_fates"],
            d["pred_effective_fates"],
            marker="o",
            lw=2.0 if name == "BMF" else 1.5,
            color=PALETTE[name],
            label=name,
            alpha=0.95,
        )
    ax.plot([1, 2.35], [1, 2.35], color="#CBD5E1", lw=1.0, ls="--", zorder=0)
    ax.set_xlim(1.0, 6.2)
    ax.set_ylim(0.95, 2.35)
    ax.set_xlabel("clone truth: effective future fates")
    ax.set_ylabel("model prediction: effective fates")
    ax.set_title("Branch model tracks fate uncertainty", loc="left")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.grid(color="#E5E7EB", lw=0.7)
    sns.despine(ax=ax)


def draw_ambiguity(ax, strata: pd.DataFrame) -> None:
    order = ["low", "mid", "high"]
    x = np.arange(len(order))
    width = 0.23
    for i, name in enumerate(["VGFM", "Dynamical OT", "BMF"]):
        d = strata[strata["run"].eq(name)].set_index("ambiguity").loc[order]
        ax.bar(x + (i - 1) * width, d["brier"], width=width, color=PALETTE[name], label=name, alpha=0.95)
    ax.set_xticks(x)
    ax.set_xticklabels(["low\nambiguity", "mid", "high"])
    ax.set_ylabel("Brier error\nlower is better")
    ax.set_title("BMF best in ambiguous progenitors", loc="left")
    ax.grid(axis="y", color="#E5E7EB", lw=0.7)
    sns.despine(ax=ax)


def draw_collapse(ax, diag: pd.DataFrame) -> None:
    x = np.arange(len(diag))
    colors = [PALETTE[v] for v in diag["variant"]]
    ax.bar(x, diag["CFCorr"], color=colors, width=0.58, edgecolor="white", linewidth=1.0)
    for i, row in diag.iterrows():
        ax.text(i, row["CFCorr"] + 0.018, f"{row['CFCorr']:.3f}", ha="center", va="bottom", fontsize=8)
        ax.text(i, 0.018, f"Brier\n{row['Brier']:.3f}", ha="center", va="bottom", fontsize=6.5, color="#475569")
    ax.set_xticks(x)
    ax.set_xticklabels(["full branch\nmixture", "top-1 fate\ncollapse"])
    ax.set_ylim(0, 0.50)
    ax.set_ylabel("CFCorr\nhigher is better")
    ax.set_title("Branch collapse loses signal", loc="left")
    ax.grid(axis="y", color="#E5E7EB", lw=0.7)
    sns.despine(ax=ax)


def main() -> None:
    setup_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    runs = {name: load_run(name) for name in RUN_TABLES}
    cal = binned_entropy_calibration(runs)
    strata = ambiguity_strata(runs)
    diag = collapse_diagnostic(runs["BMF"])
    cal.to_csv(TABLE_DIR / "clone_branch_entropy_calibration.csv", index=False)
    strata.to_csv(TABLE_DIR / "clone_branch_ambiguity_strata.csv", index=False)
    diag.to_csv(TABLE_DIR / "clone_branch_collapse_diagnostic.csv", index=False)

    fig = plt.figure(figsize=(13.6, 4.25))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.24, 1.00, 1.02, 0.82], wspace=0.43)
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    draw_spring_entropy(axes[0], runs["BMF"])
    draw_entropy_calibration(axes[1], cal)
    draw_ambiguity(axes[2], strata)
    draw_collapse(axes[3], diag)
    fig.suptitle(
        "Branch structure explains clone-fate recovery",
        x=0.02,
        y=1.02,
        ha="left",
        fontsize=15.5,
        weight="bold",
        color="#111827",
    )
    fig.text(
        0.02,
        -0.015,
        "Post-hoc diagnostic: top-1 collapse keeps BMF endpoints but removes its multi-branch fate probabilities; no model is retrained.",
        fontsize=7.2,
        color="#475569",
    )
    fig.subplots_adjust(left=0.035, right=0.99, top=0.86, bottom=0.18)
    for ext in ["png", "pdf", "svg"]:
        out = FIG_DIR / f"panel_b_branch_mechanism_diagnostic.{ext}"
        fig.savefig(out, dpi=360 if ext == "png" else None, bbox_inches="tight")
        print(out)


if __name__ == "__main__":
    main()
