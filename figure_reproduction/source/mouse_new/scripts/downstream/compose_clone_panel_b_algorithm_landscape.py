#!/usr/bin/env python3
"""Compose a CellCompass clone-case panel B.

This figure mirrors the multi-omics panel-B logic, but uses the real
Clone-CUVGF campaign records and clone-fate metrics.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cellcompass_mpl_cache")

import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN_DIR = Path.home() / ".cellcompass/algorithm_campaigns/clone_cuvgf/campaign_20260606T144856902012Z"
TRIAL_DIR = CAMPAIGN_DIR / "trials"
FINAL_METRIC_CSV = ROOT / "downstream/clone_fate_spring_final/final_metric_summary.csv"
STAGE2_METRIC_CSV = ROOT / "downstream/clone_fate_spring/stage2_metric_summary.csv"
FIG_DIR = ROOT / "figures/cellcompass_clone_case"
TABLE_DIR = ROOT / "downstream/cellcompass_clone_case"


COLORS = {
    "purple": "#6D28D9",
    "purple_dark": "#3B1A93",
    "purple_light": "#C4B5FD",
    "teal": "#0F766E",
    "green": "#2E8B57",
    "orange": "#E5792A",
    "red": "#DC2626",
    "blue": "#2C5AA0",
    "gray": "#CBD5E1",
    "gray_dark": "#475569",
    "ink": "#111827",
}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.linewidth": 1.2,
            "axes.edgecolor": COLORS["ink"],
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
        }
    )


def extract_cfcorr(metrics: dict) -> float | None:
    custom = metrics.get("custom_metrics") or {}
    if custom.get("CFCorr") is not None:
        return float(custom["CFCorr"])
    if metrics.get("primary_metric") == "CFCorr" and metrics.get("primary_value") is not None:
        return float(metrics["primary_value"])
    if metrics.get("secondary_metric") == "CFCorr" and metrics.get("secondary_value") is not None:
        return float(metrics["secondary_value"])
    return None


def extract_brier(metrics: dict) -> float | None:
    custom = metrics.get("custom_metrics") or {}
    value = custom.get("CFCorr_brier") or custom.get("Brier")
    return None if value is None else float(value)


def load_trial_curve() -> pd.DataFrame:
    rows = []
    for path in sorted(TRIAL_DIR.glob("trial_*.json")):
        data = json.loads(path.read_text())
        metrics = data.get("metrics_summary") or {}
        cfcorr = extract_cfcorr(metrics)
        if cfcorr is None:
            continue
        reasons = " | ".join(data.get("decision_reasons") or [])
        rows.append(
            {
                "trial_id": data.get("trial_id", path.stem),
                "created_at": data.get("created_at"),
                "stage": data.get("stage"),
                "decision": data.get("decision"),
                "status": data.get("status"),
                "CFCorr": cfcorr,
                "Brier": extract_brier(metrics),
                "W1": float(metrics["w1_mean"]) if metrics.get("w1_mean") is not None else np.nan,
                "TMV": float(metrics["tmv_mean"]) if metrics.get("tmv_mean") is not None else np.nan,
                "primary_metric": metrics.get("primary_metric"),
                "primary_value": metrics.get("primary_value"),
                "secondary_metric": metrics.get("secondary_metric"),
                "secondary_value": metrics.get("secondary_value"),
                "failed_mass_gate": "TMV hard gate failed" in reasons,
                "decision_reasons": reasons,
                "source_file": str(path),
            }
        )
    df = pd.DataFrame(rows).sort_values("created_at").reset_index(drop=True)
    df["trial_order"] = np.arange(1, len(df) + 1)
    accepted = df["decision"].eq("promote")
    best = []
    running = np.nan
    for _, row in df.iterrows():
        if row["decision"] == "promote":
            running = row["CFCorr"] if not np.isfinite(running) else max(running, row["CFCorr"])
        best.append(running)
    df["best_accepted_CFCorr"] = best
    return df


def load_metrics() -> pd.DataFrame:
    final = pd.read_csv(FINAL_METRIC_CSV)
    final = final[["run_key", "run_label", "CFCorr", "CFCorr_brier", "w1_mean", "tmv_mean"]].copy()
    final["source"] = str(FINAL_METRIC_CSV)
    return final


def landscape_surface():
    gx = np.linspace(0.02, 0.98, 260)
    gy = np.linspace(0.08, 0.92, 220)
    xx, yy = np.meshgrid(gx, gy)
    # "Fate mismatch" surface: lower basin means higher CFCorr.
    zz = 1.05 - 0.15 * xx - 0.10 * yy
    basins = [
        (0.16, 0.28, 0.12, 0.13, 0.12),  # VGFM
        (0.44, 0.40, 0.13, 0.14, 0.17),  # FP
        (0.82, 0.76, 0.15, 0.16, 0.52),  # BMF selected
    ]
    for cx, cy, sx, sy, depth in basins:
        zz -= depth * np.exp(-(((xx - cx) ** 2) / (2 * sx**2) + ((yy - cy) ** 2) / (2 * sy**2)))
    ridges = [
        (0.32, 0.56, 0.10, 0.20, 0.09),
        (0.58, 0.34, 0.12, 0.18, 0.08),
        (0.70, 0.62, 0.10, 0.17, 0.06),
    ]
    for cx, cy, sx, sy, amp in ridges:
        zz += amp * np.exp(-(((xx - cx) ** 2) / (2 * sx**2) + ((yy - cy) ** 2) / (2 * sy**2)))
    zz += 0.010 * np.sin(17 * xx + 2 * yy) + 0.008 * np.cos(21 * yy)
    zz = np.clip(zz, 0.34, 1.15)
    return gx, gy, xx, yy, zz


def interp_surface(gx, gy, zz, x: float, y: float) -> float:
    ix = int(np.clip(np.searchsorted(gx, x) - 1, 0, len(gx) - 2))
    iy = int(np.clip(np.searchsorted(gy, y) - 1, 0, len(gy) - 2))
    x0, x1 = gx[ix], gx[ix + 1]
    y0, y1 = gy[iy], gy[iy + 1]
    tx = 0 if x1 == x0 else (x - x0) / (x1 - x0)
    ty = 0 if y1 == y0 else (y - y0) / (y1 - y0)
    return float(
        (1 - tx) * (1 - ty) * zz[iy, ix]
        + tx * (1 - ty) * zz[iy, ix + 1]
        + (1 - tx) * ty * zz[iy + 1, ix]
        + tx * ty * zz[iy + 1, ix + 1]
    )


def draw_landscape(ax) -> None:
    gx, gy, xx, yy, zz = landscape_surface()
    cmap = LinearSegmentedColormap.from_list(
        "clone_landscape",
        ["#F8FAFC", "#E2E8F0", "#CBD5E1", "#94A3B8", "#475569"],
    )
    floor = float(zz.min()) - 0.11
    surf = ax.plot_surface(
        xx,
        yy,
        zz,
        cmap=cmap,
        linewidth=0,
        antialiased=True,
        alpha=0.72,
        rcount=130,
        ccount=160,
        shade=True,
        zorder=0,
    )
    if hasattr(surf, "set_sort_zpos"):
        surf.set_sort_zpos(floor - 2)
    ax.contour(xx, yy, zz, zdir="z", offset=floor, levels=16, cmap=cmap, linewidths=0.7, alpha=0.72)

    models = pd.DataFrame(
        [
            {
                "key": "VGFM",
                "label": "VGFM\n(no clone labels)",
                "x": 0.16,
                "y": 0.28,
                "cfcorr": 0.0355,
                "color": COLORS["blue"],
                "note": "single expression path;\nclone fate not used",
            },
            {
                "key": "Clone-CUVGF-FP",
                "label": "Clone-CUVGF-FP\n(terminal-fate penalty)",
                "x": 0.44,
                "y": 0.40,
                "cfcorr": 0.0679,
                "color": COLORS["orange"],
                "note": "clone fate used as\nan endpoint penalty",
            },
            {
                "key": "Clone-CUVGF-BMF",
                "label": "Clone-CUVGF-BMF\n(branch-mass)",
                "x": 0.84,
                "y": 0.76,
                "cfcorr": 0.4271,
                "color": COLORS["purple"],
                "note": "one starting cell can\nsplit across fate branches",
            },
        ]
    )
    rng = np.random.default_rng(20260608)
    cloud_rows = []
    for _, row in models.iterrows():
        n = 160
        sx = 0.025 if row["key"] != "Clone-CUVGF-BMF" else 0.020
        sy = 0.030 if row["key"] != "Clone-CUVGF-BMF" else 0.024
        px = np.clip(rng.normal(row["x"], sx, n), 0.05, 0.95)
        py = np.clip(rng.normal(row["y"], sy, n), 0.10, 0.90)
        pz = np.asarray([interp_surface(gx, gy, zz, a, b) for a, b in zip(px, py)]) + rng.normal(0.035, 0.006, n)
        ax.scatter(px, py, pz, s=12, color=row["color"], alpha=0.86, edgecolor="none", depthshade=False, zorder=20)
        cloud_rows.extend({"model": row["key"], "x": a, "y": b, "z": c} for a, b, c in zip(px, py, pz))

    path_keys = ["VGFM", "Clone-CUVGF-FP", "Clone-CUVGF-BMF"]
    pts = models.set_index("key").loc[path_keys, ["x", "y"]].to_numpy(float)
    path = []
    for i in range(len(pts) - 1):
        p0, p1 = pts[i], pts[i + 1]
        vec = p1 - p0
        normal = np.array([-vec[1], vec[0]], dtype=float)
        normal /= max(np.linalg.norm(normal), 1e-8)
        for t in np.linspace(0, 1, 70, endpoint=False):
            s = t * t * (3 - 2 * t)
            bend = 0.025 * np.sin(np.pi * t) * normal
            path.append((1 - s) * p0 + s * p1 + bend)
    path.append(pts[-1])
    path = np.asarray(path)
    path_z = np.asarray([interp_surface(gx, gy, zz, x, y) for x, y in path]) + 0.075
    ax.plot(path[:, 0], path[:, 1], path_z, color="white", lw=5.2, alpha=0.98, zorder=35)
    ax.plot(path[:, 0], path[:, 1], path_z + 0.006, color=COLORS["ink"], lw=2.3, ls=(0, (2.4, 2.4)), alpha=0.95, zorder=36)

    for i, (a, b) in enumerate(zip(path_keys[:-1], path_keys[1:]), start=1):
        p0 = models.loc[models["key"].eq(a)].iloc[0]
        p1 = models.loc[models["key"].eq(b)].iloc[0]
        mx, my = (p0["x"] + p1["x"]) / 2, (p0["y"] + p1["y"]) / 2
        mz = interp_surface(gx, gy, zz, mx, my) + 0.18
        ax.text(
            mx,
            my,
            mz,
            str(i),
            ha="center",
            va="center",
            fontsize=8.0,
            color=COLORS["ink"],
            weight="bold",
            bbox=dict(boxstyle="circle,pad=0.22", fc="white", ec="#CBD5E1", alpha=0.96),
            zorder=50,
        )

    for _, row in models.iterrows():
        z = interp_surface(gx, gy, zz, row["x"], row["y"]) + 0.075
        if row["key"] == "Clone-CUVGF-BMF":
            ax.scatter(
                [row["x"]],
                [row["y"]],
                [z + 0.010],
                marker="*",
                s=520,
                color=row["color"],
                edgecolor="white",
                linewidth=1.8,
                depthshade=False,
                zorder=55,
            )
        label = ax.text(
            row["x"] + (0.020 if row["key"] == "Clone-CUVGF-BMF" else 0),
            row["y"] + 0.030,
            z + 0.145,
            row["label"],
            ha="left" if row["key"] == "Clone-CUVGF-BMF" else "center",
            va="bottom",
            fontsize=7.1,
            color=row["color"],
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.16,rounding_size=0.08", fc="white", ec="none", alpha=0.82),
            zorder=60,
        )
        label.set_path_effects([pe.withStroke(linewidth=2.7, foreground="white", alpha=0.95)])

    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(0.10, 0.90)
    ax.set_zlim(floor, 1.18)
    ax.view_init(elev=36, azim=-58)
    try:
        ax.set_box_aspect((1.40, 0.95, 0.48))
    except Exception:
        pass
    ax.set_xlabel("how clone fate labels enter", labelpad=9, fontsize=7.5)
    ax.set_ylabel("how future fate is represented", labelpad=9, fontsize=7.5)
    ax.set_zlabel("clone-fate error\nlower is better", labelpad=5, fontsize=7.5)
    ax.set_xticks([0.16, 0.44, 0.84])
    ax.set_xticklabels(["not used", "endpoint\npenalty", "branch\nprobabilities"], fontsize=6.6)
    ax.set_yticks([0.28, 0.40, 0.76])
    ax.set_yticklabels(["single\npath", "penalized\nsingle path", "multiple\nfate branches"], fontsize=6.4)
    ax.set_zticks([0.45, 0.75, 1.05])
    ax.set_zticklabels(["low", "mid", "high"], fontsize=6.5)
    ax.xaxis.pane.set_facecolor((1, 1, 1, 0))
    ax.yaxis.pane.set_facecolor((1, 1, 1, 0))
    ax.zaxis.pane.set_facecolor((1, 1, 1, 0))
    ax.grid(True, color="#E5E7EB")
    ax.set_title("Model families tested by CellCompass", fontsize=10.2, weight="bold", pad=4)
    ax.text2D(
        0.03,
        0.02,
        "Path labels: 1 add clone-derived fate labels as an endpoint penalty; 2 replace one future path with fate branches.\n"
        "These are model families, not tuning snapshots. Clone labels guide training only; prediction starts from day-2 expression.",
        transform=ax.transAxes,
        fontsize=6.5,
        color="#334155",
        bbox=dict(boxstyle="round,pad=0.30,rounding_size=0.08", fc="white", ec="#CBD5E1", alpha=0.92),
    )

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    models.to_csv(TABLE_DIR / "clone_panel_b_landscape_design_points.csv", index=False)
    pd.DataFrame(cloud_rows).to_csv(TABLE_DIR / "clone_panel_b_landscape_cloud_points.csv", index=False)


def draw_optimization(ax, trials: pd.DataFrame) -> None:
    windows = [
        (1, 15.5, "#FEF3C7", "1. fate cost in transport\nweak gain; mass failures"),
        (15.5, 24.5, "#D1FAE5", "2. stable mass + branches\n0.212 -> 0.315"),
        (24.5, 53.5, "#EDE9FE", "3. tune branch dynamics\n0.369 -> 0.427"),
    ]
    for start, end, color, label in windows:
        ax.axvspan(start, end, color=color, alpha=0.42, lw=0, zorder=0)
        ax.text(
            (start + end) / 2,
            0.492,
            label.split(".", 1)[0],
            ha="center",
            va="top",
            fontsize=7.0,
            color="#334155",
            weight="bold",
            bbox=dict(boxstyle="circle,pad=0.18", fc="white", ec="none", alpha=0.82),
            zorder=6,
        )
    ax.text(
        0.035,
        0.955,
        "1 fate cost in transport: weak gain; mass failures\n"
        "2 stable mass + fate branches: 0.212 -> 0.315\n"
        "3 branch dynamics tuned: 0.369 -> 0.427",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.0,
        color="#334155",
        linespacing=1.12,
        bbox=dict(boxstyle="round,pad=0.28,rounding_size=0.06", fc="white", ec="none", alpha=0.78),
        zorder=7,
    )

    all_nonfailed = trials[~trials["failed_mass_gate"]]
    failed = trials[trials["failed_mass_gate"]]
    accepted = trials[trials["decision"].eq("promote")]
    ax.plot(all_nonfailed["trial_order"], all_nonfailed["CFCorr"], color="#CBD5E1", lw=1.0, alpha=0.70, zorder=1)
    ax.scatter(
        all_nonfailed["trial_order"],
        all_nonfailed["CFCorr"],
        s=14,
        color="#CBD5E1",
        alpha=0.70,
        edgecolor="white",
        linewidth=0.35,
        label="all tested trials",
        zorder=2,
    )
    ax.scatter(failed["trial_order"], failed["CFCorr"], s=24, marker="x", color=COLORS["red"], alpha=0.72, linewidth=1.2, label="failed mass check", zorder=3)
    ax.step(
        accepted["trial_order"],
        accepted["best_accepted_CFCorr"],
        where="post",
        lw=2.9,
        color=COLORS["purple"],
        label="best clone-fate score so far",
        zorder=4,
    )

    dyn_ot = 0.27729041744232064
    no_clone = 0.035505608150077944
    ax.axhline(dyn_ot, color="#64748B", lw=1.1, ls="--", zorder=1)
    ax.text(53.2, dyn_ot + 0.006, "dynamical OT baseline 0.277", fontsize=6.2, ha="right", va="bottom", color="#475569")
    ax.axhline(no_clone, color="#94A3B8", lw=0.9, ls=":", zorder=1)
    ax.text(53.2, no_clone + 0.006, "VGFM no-clone 0.036", fontsize=6.2, ha="right", va="bottom", color="#64748B")

    callouts = [
        (9, 0.06790463649534836, "FP active best\n0.068"),
        (17, 0.2117971318108328, "first branch-mass\ntrial 0.212"),
        (24, 0.31529552232794483, "passes strongest\nbaseline"),
        (27, 0.4242971339621398, "fate branches\nstabilize"),
        (50, 0.4271099373187193, "locked\n0.427"),
    ]
    for x, y, label in callouts:
        ax.scatter([x], [y], s=52, facecolor="white", edgecolor=COLORS["purple"], linewidth=1.4, zorder=7)
        dy = 0.055 if x < 40 else -0.060
        ax.annotate(
            label,
            xy=(x, y),
            xytext=(x + (2.0 if x < 24 else (1.5 if x < 35 else -4.5)), y + dy),
            fontsize=6.4,
            color=COLORS["purple_dark"],
            ha="left" if x < 35 else "right",
            va="center",
            arrowprops=dict(arrowstyle="-", color=COLORS["purple_light"], lw=0.8),
            zorder=8,
        )

    ax.set_xlim(0, 54)
    ax.set_ylim(0.0, 0.51)
    ax.set_xlabel("real-data trial order", fontsize=7.7)
    ax.set_ylabel("clone-fate agreement\nCFCorr, higher is better", fontsize=7.7)
    ax.set_title("Clone-CUVGF-BMF optimization history", fontsize=10.2, weight="bold", pad=4)
    ax.grid(axis="y", color="#E5E7EB", lw=0.8)
    ax.legend(loc="lower right", frameon=False, fontsize=6.4, handlelength=1.5)
    sns.despine(ax=ax)


def draw_metric_panel(ax, metrics: pd.DataFrame) -> None:
    order = [
        ("vgfm_no_clone_anchor", "VGFM\nno clone", COLORS["blue"]),
        ("dynamical_ot_claim_sota", "Dynamical\nOT", "#94A3B8"),
        ("clone_cuvgf_bmf_final", "Clone-CUVGF-\nBMF", COLORS["purple"]),
    ]
    rows = []
    for key, label, color in order:
        row = metrics.loc[metrics["run_key"].eq(key)].iloc[0].to_dict()
        row["short_label"] = label
        row["color"] = color
        rows.append(row)
    df = pd.DataFrame(rows)
    x = np.arange(len(df))
    bars = ax.bar(x, df["CFCorr"], color=df["color"], width=0.62, edgecolor="white", linewidth=1.0, zorder=3)
    for i, (_, row) in enumerate(df.iterrows()):
        ax.scatter([i], [row["CFCorr"]], s=58, color=row["color"], edgecolor="white", linewidth=1.0, zorder=4)
        ax.text(i, row["CFCorr"] + 0.020, f"{row['CFCorr']:.3f}", ha="center", va="bottom", fontsize=7.4, color="#334155")
        ax.text(i, 0.012, f"Brier\n{row['CFCorr_brier']:.3f}", ha="center", va="bottom", fontsize=5.8, color="#64748B")
    ax.axhline(df.loc[df["run_key"].eq("dynamical_ot_claim_sota"), "CFCorr"].iloc[0], color="#64748B", lw=1.0, ls="--", alpha=0.8)
    ax.text(0.05, 0.458, "+54% vs dynamical OT", fontsize=8.4, color=COLORS["purple_dark"], weight="bold")
    ax.text(0.05, 0.420, "day-2 cells scored against\nclone-derived day-6 fates", fontsize=6.4, color="#475569")
    ax.set_xticks(x)
    ax.set_xticklabels(df["short_label"], fontsize=7.0)
    ax.set_ylim(0, 0.50)
    ax.set_ylabel("CFCorr\nhigher is better", fontsize=7.7)
    ax.set_title("Final clone-fate metric", fontsize=10.2, weight="bold", pad=4)
    ax.grid(axis="y", color="#E5E7EB", lw=0.8, zorder=0)
    sns.despine(ax=ax)


def compose() -> None:
    setup_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    trials = load_trial_curve()
    metrics = load_metrics()
    trials.to_csv(TABLE_DIR / "clone_panel_b_trial_curve.csv", index=False)
    metrics.to_csv(TABLE_DIR / "clone_panel_b_final_metrics.csv", index=False)

    fig = plt.figure(figsize=(14.7, 5.05), constrained_layout=False)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.55, 1.05, 0.62], wspace=0.16)
    ax_land = fig.add_subplot(gs[0, 0], projection="3d", computed_zorder=False)
    ax_curve = fig.add_subplot(gs[0, 1])
    ax_metric = fig.add_subplot(gs[0, 2])

    draw_landscape(ax_land)
    draw_optimization(ax_curve, trials)
    draw_metric_panel(ax_metric, metrics)

    fig.suptitle(
        "CellCompass searches a clone-aware fate-dynamics model",
        x=0.02,
        y=0.995,
        ha="left",
        fontsize=17.5,
        weight="bold",
        color=COLORS["ink"],
    )
    fig.text(
        0.02,
        0.01,
        "Clone-CUVGF-BMF: Clone-Censored Unbalanced Velocity-Growth Flow Matching with Branch-Mass Fate dynamics.",
        fontsize=7.0,
        color="#475569",
    )
    fig.subplots_adjust(left=0.025, right=0.99, top=0.89, bottom=0.12)

    for ext in ("png", "pdf", "svg"):
        out = FIG_DIR / f"panel_b_clone_algorithm_landscape.{ext}"
        fig.savefig(out, dpi=360 if ext == "png" else None)
        print(out)


if __name__ == "__main__":
    compose()
