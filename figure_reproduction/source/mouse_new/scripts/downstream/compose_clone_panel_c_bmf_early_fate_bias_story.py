#!/usr/bin/env python3
"""Compact panel C for the clone/BMF biological story.

The panel is intentionally narrow in scope:
1. day-2 progenitors carry barcode-derived future-fate bias on the SPRING map;
2. BMF preserves the continuous barcode future trends for major RNA programs;
3. neutrophil/monocyte futures are better explained by early-state RNA modules
   than by mature markers in natural day-2 cells, while BMF perturbations show
   that mature programs remain effective if forcibly activated.
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
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[2]
TABLE_DIR = ROOT / "downstream/cellcompass_clone_case"
FIG_DIR = ROOT / "figures/cellcompass_clone_case"
ADATA_PATH = Path("/Users/zhenyizhang/.cellcompass/algorithm_benchmarks/datasets/weinreb_rawrebuild_full_k10mindiff1/data.h5ad")
BMF_TABLE = ROOT / "downstream/clone_fate_spring_final/clone_cuvgf_bmf_final_day2_fate_probabilities.csv"
VGFM_TABLE = ROOT / "downstream/clone_fate_spring_final/vgfm_no_clone_anchor_day2_fate_probabilities.csv"
PATH_TABLE = TABLE_DIR / "clone_downstream_selected_bmf_branch_paths.csv"
PARTICLE_TABLE = TABLE_DIR / "clone_bmf_particle_spring_growth_mass_particles.csv"
CURVE_TABLE = TABLE_DIR / "clone_state_matched_fate_divergence_program_future_curves.csv"
CORR_TABLE = TABLE_DIR / "clone_state_matched_fate_divergence_program_future_correlations.csv"
MODULE_TABLE = TABLE_DIR / "clone_day2_neutrophil_monocyte_bias_module_summary.csv"
PERTURB_TABLE = TABLE_DIR / "clone_day2_fate_bias_perturbation_target_response.csv"

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
PROGRAM_COLOR = {
    "Neutrophil": FATE_PALETTE["Neutrophil"],
    "Monocyte": FATE_PALETTE["Monocyte"],
    "Baso/Mast": FATE_PALETTE["Mast"],
    "Meg/Ery": FATE_PALETTE["Erythroid"],
}
PLOT_FATES = ["Undifferentiated", "Neutrophil", "Monocyte", "Baso", "Mast", "Meg", "Erythroid"]


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.linewidth": 1.0,
            "axes.edgecolor": "#111827",
            "axes.titlesize": 9.4,
            "axes.titleweight": "bold",
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
        }
    )


def add_bar_stripes(ax: plt.Axes, bars, colors: list[str], *, spacing: float = 0.28, lw: float = 0.8) -> None:
    """Draw explicit vector diagonal stripes inside vertical bar patches.

    Matplotlib hatch patterns can disappear or become weak after SVG/PDF import
    into Illustrator. These clipped-free line segments are ordinary vector
    paths, so they survive downstream figure editing.
    """
    for patch, color in zip(bars, colors):
        x0 = float(patch.get_x())
        x1 = x0 + float(patch.get_width())
        y0 = float(patch.get_y())
        y1 = y0 + float(patch.get_height())
        if not np.isfinite([x0, x1, y0, y1]).all() or y1 <= y0:
            continue
        for offset in np.arange(-0.85, 1.0, spacing):
            pts: list[tuple[float, float]] = []
            for u, v in ((0.0, -offset), (1.0, 1.0 - offset), (offset, 0.0), (1.0 + offset, 1.0)):
                if -1e-9 <= u <= 1.0 + 1e-9 and -1e-9 <= v <= 1.0 + 1e-9:
                    pts.append((min(max(u, 0.0), 1.0), min(max(v, 0.0), 1.0)))
            if len(pts) < 2:
                continue
            (u0, v0), (u1, v1) = pts[0], pts[1]
            ax.plot(
                [x0 + u0 * (x1 - x0), x0 + u1 * (x1 - x0)],
                [y0 + v0 * (y1 - y0), y0 + v1 * (y1 - y0)],
                color=color,
                lw=lw,
                alpha=0.95,
                solid_capstyle="butt",
                zorder=patch.get_zorder() + 0.3,
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


def draw_observed_background(ax: plt.Axes, obs: pd.DataFrame, alpha: float = 0.055) -> None:
    for fate, color in FATE_PALETTE.items():
        if fate == "Other":
            continue
        d = obs[obs["cell_type"].eq(fate)]
        if d.empty:
            continue
        ax.scatter(d["spring_x"], d["spring_y"], s=0.9, c=color, alpha=alpha, linewidth=0, rasterized=True, zorder=0)


def add_fate_labels(ax: plt.Axes, obs: pd.DataFrame) -> None:
    terminal = obs[obs["t"].eq(2)]
    offsets = {
        "Undifferentiated": (-35, 15),
        "Neutrophil": (10, -30),
        "Monocyte": (10, 35),
        "Baso": (-25, -20),
        "Mast": (-10, -28),
        "Meg": (-30, 20),
        "Erythroid": (25, -20),
    }
    for fate in PLOT_FATES:
        d = terminal[terminal["cell_type"].eq(fate)]
        if len(d) < 20:
            continue
        dx, dy = offsets.get(fate, (0, 0))
        label = fate.replace("Undifferentiated", "Undiff.")
        ax.text(
            float(d["spring_x"].median()) + dx,
            float(d["spring_y"].median()) + dy,
            label,
            color=FATE_PALETTE[fate],
            fontsize=8.2,
            weight="bold",
            ha="center",
            va="center",
            path_effects=[pe.Stroke(linewidth=2.2, foreground="white", alpha=0.95), pe.Normal()],
            zorder=80,
        )


def eligible_day2_table() -> pd.DataFrame:
    bmf = pd.read_csv(BMF_TABLE)
    fates = [c.removeprefix("pred_") for c in bmf.columns if c.startswith("pred_")]
    truth_cols = [f"truth_{f}" for f in fates]
    pred_cols = [f"pred_{f}" for f in fates]
    valid = bmf[
        bmf["eligible"].astype(bool)
    ].copy()
    truth = normalize_rows(valid[truth_cols].fillna(0).to_numpy(float))
    bmf_pred = normalize_rows(valid[pred_cols].fillna(0).to_numpy(float))
    fate_names = np.asarray(fates, dtype=object)
    valid["barcode_dominant_future"] = fate_names[truth.argmax(axis=1)]
    valid["bmf_dominant_future"] = fate_names[bmf_pred.argmax(axis=1)]

    vgfm = pd.read_csv(VGFM_TABLE)
    vgfm = vgfm[
        vgfm["eligible"].astype(bool)
    ].copy()
    vgfm_pred = normalize_rows(vgfm[pred_cols].fillna(0).to_numpy(float))
    vgfm["vgfm_dominant_future"] = fate_names[vgfm_pred.argmax(axis=1)]
    valid = valid.merge(
        vgfm[["source_index", "vgfm_dominant_future"]],
        on="source_index",
        how="left",
        validate="one_to_one",
    )
    return valid


def all_day2_bmf_table() -> pd.DataFrame:
    bmf = pd.read_csv(BMF_TABLE)
    fates = [c.removeprefix("pred_") for c in bmf.columns if c.startswith("pred_")]
    pred_cols = [f"pred_{f}" for f in fates]
    pred = normalize_rows(bmf[pred_cols].fillna(0).to_numpy(float))
    out = bmf.copy()
    out["bmf_dominant_future"] = np.asarray(fates, dtype=object)[pred.argmax(axis=1)]
    return out


def day2_zoom_bounds(valid: pd.DataFrame) -> tuple[float, float, float, float]:
    x0, x1 = valid["spring_x"].quantile([0.05, 0.95]).to_numpy(float)
    y0, y1 = valid["spring_y"].quantile([0.05, 0.95]).to_numpy(float)
    dx = 0.10 * (x1 - x0)
    dy = 0.10 * (y1 - y0)
    return x0 - dx, x1 + dx, y0 - dy, y1 + dy


def draw_future_points(ax: plt.Axes, valid: pd.DataFrame, column: str, size: float, alpha: float, zorder: int) -> None:
    for fate in PLOT_FATES:
        d = valid[valid[column].eq(fate)]
        if d.empty:
            continue
        ax.scatter(
            d["spring_x"],
            d["spring_y"],
            s=size,
            c=FATE_PALETTE[fate],
            alpha=alpha,
            linewidth=0,
            zorder=zorder,
        )


def assign_terminal_fate_from_spring(paths: pd.DataFrame, obs: pd.DataFrame, *, k: int = 25) -> pd.DataFrame:
    final_t = float(paths["model_time"].max())
    final = paths[paths["model_time"].eq(final_t)].copy()
    terminal_obs = obs[obs["t"].eq(2)].copy()
    tree = cKDTree(terminal_obs[["spring_x", "spring_y"]].to_numpy(float))
    dist, idx = tree.query(final[["spring_x", "spring_y"]].to_numpy(float), k=min(k, len(terminal_obs)))
    if idx.ndim == 1:
        dist = dist[:, None]
        idx = idx[:, None]
    neighbor_labels = terminal_obs["cell_type"].to_numpy(object)[idx]
    weights = 1.0 / (np.asarray(dist, dtype=float) + 1e-3)
    assigned = []
    for row_labels, row_weights in zip(neighbor_labels, weights):
        scores: dict[str, float] = {}
        for label, weight in zip(row_labels, row_weights):
            scores[str(label)] = scores.get(str(label), 0.0) + float(weight)
        assigned.append(max(scores.items(), key=lambda kv: kv[1])[0])
    final["terminal_fate_from_spring"] = assigned
    keys = final[["source_index", "branch_fate", "terminal_fate_from_spring"]]
    return paths.merge(keys, on=["source_index", "branch_fate"], how="left", validate="many_to_one")


def draw_bmf_terminal_average_paths(ax: plt.Axes, obs: pd.DataFrame, valid: pd.DataFrame) -> None:
    if not PARTICLE_TABLE.exists():
        return
    paths = pd.read_csv(PARTICLE_TABLE)
    valid_sources = set(valid["source_index"].astype(int).tolist())
    paths = paths[paths["source_index"].astype(int).isin(valid_sources)].copy()
    paths = paths[paths["branch_fate"].isin(PLOT_FATES)].copy()
    paths = assign_terminal_fate_from_spring(paths, obs)
    paths = paths[paths["terminal_fate_from_spring"].isin(PLOT_FATES)].copy()
    terminal_mass = (
        paths[paths["model_time"].eq(float(paths["model_time"].max()))]
        .groupby("terminal_fate_from_spring")["mass"]
        .sum()
        .sort_values(ascending=False)
    )
    max_mass = float(terminal_mass.max()) if len(terminal_mass) else 1.0
    max_day_mass = float(paths.groupby("model_time")["mass"].sum().max()) if len(paths) else 1.0
    keep = [f for f in PLOT_FATES if f in terminal_mass.index]
    for terminal_fate in keep:
        raw = paths[paths["terminal_fate_from_spring"].eq(terminal_fate)].copy()
        color = FATE_PALETTE.get(terminal_fate, "#111827")
        rows = []
        for t, g in raw.groupby("model_time", sort=True):
            w = g["mass"].clip(lower=0).to_numpy(float)
            if float(w.sum()) <= 0:
                x = float(g["spring_x"].mean())
                y = float(g["spring_y"].mean())
            else:
                x = float(np.average(g["spring_x"].to_numpy(float), weights=w))
                y = float(np.average(g["spring_y"].to_numpy(float), weights=w))
            rows.append((float(t), x, y, float(w.sum())))
        d = pd.DataFrame(rows, columns=["model_time", "spring_x", "spring_y", "mass"]).sort_values("model_time")
        if len(d) < 2:
            continue
        relative_mass = float(terminal_mass.loc[terminal_fate]) / max(max_mass, 1e-9)
        lw = 0.75 + 2.0 * (relative_mass ** 0.45)
        alpha = 0.28 + 0.62 * min(1.0, relative_mass ** 0.5)
        ax.plot(
            d["spring_x"],
            d["spring_y"],
            color=color,
            lw=lw,
            alpha=alpha,
            solid_capstyle="round",
            zorder=55,
            path_effects=[pe.Stroke(linewidth=lw + 2.4, foreground="white", alpha=0.78), pe.Normal()],
        )
        ax.scatter(
            d["spring_x"],
            d["spring_y"],
            s=18 + 42 * (d["mass"].to_numpy(float) / max(max_day_mass, 1e-9)),
            facecolor=color,
            edgecolor="white",
            linewidth=0.85,
            alpha=alpha,
            zorder=63,
        )
        ax.annotate(
            "",
            xy=(d["spring_x"].iloc[-1], d["spring_y"].iloc[-1]),
            xytext=(d["spring_x"].iloc[-2], d["spring_y"].iloc[-2]),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=0, shrinkB=0, alpha=alpha),
            zorder=64,
        )


def plot_future_zoom(ax: plt.Axes, obs: pd.DataFrame, valid: pd.DataFrame, column: str, title: str, bounds: tuple[float, float, float, float]) -> None:
    x0, x1, y0, y1 = bounds
    bg = obs[obs["t"].eq(0)]
    ax.scatter(bg["spring_x"], bg["spring_y"], s=0.8, c="#CBD5E1", alpha=0.20, linewidth=0, rasterized=True, zorder=0)
    draw_future_points(ax, valid, column=column, size=8.0, alpha=0.74, zorder=10)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#CBD5E1")
        spine.set_linewidth(0.8)
    ax.set_title(title, loc="left", pad=3, fontsize=7.8)


def plot_fate_landscape(
    ax: plt.Axes,
    ax_bmf_zoom: plt.Axes,
    ax_truth: plt.Axes,
    ax_vgfm: plt.Axes,
    obs: pd.DataFrame,
    all_day2_bmf: pd.DataFrame,
    valid: pd.DataFrame,
) -> None:
    draw_observed_background(ax, obs, alpha=0.045)
    draw_future_points(ax, all_day2_bmf, column="bmf_dominant_future", size=5.2, alpha=0.70, zorder=15)
    add_fate_labels(ax, obs)
    style_spring(ax, obs)
    bounds = day2_zoom_bounds(valid)
    x0, x1, y0, y1 = bounds
    rect = mpl.patches.Rectangle(
        (x0, y0),
        x1 - x0,
        y1 - y0,
        fill=False,
        ec="#111827",
        lw=0.7,
        ls=(0, (3, 2)),
        alpha=0.55,
        zorder=75,
    )
    ax.add_patch(rect)
    ax.set_title("C1  BMF future map", loc="left", pad=5)
    plot_future_zoom(ax_bmf_zoom, obs, valid, "bmf_dominant_future", "BMF prediction", bounds)
    plot_future_zoom(ax_truth, obs, valid, "barcode_dominant_future", "barcode-measured future", bounds)
    plot_future_zoom(ax_vgfm, obs, valid, "vgfm_dominant_future", "VGFM prediction", bounds)
    ax.text(
        0.02,
        0.02,
        f"Main: BMF-predicted future for all {len(all_day2_bmf):,} day-2 cells\nInsets: all {len(valid):,} eligible barcode cells, BMF vs barcode truth vs VGFM\nDashed box marks the zoomed barcode-cell region",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.8,
        color="#374151",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#CBD5E1", lw=0.8, alpha=0.90),
        zorder=90,
    )
    handles = [
        mpl.lines.Line2D([0], [0], marker="o", color="none", markerfacecolor=FATE_PALETTE[f], markeredgecolor="none", markersize=5, label=f)
        for f in ["Neutrophil", "Monocyte", "Baso", "Mast", "Meg", "Erythroid", "Undifferentiated"]
    ]
    ax.legend(handles=handles, frameon=False, loc="upper left", bbox_to_anchor=(0.01, 0.99), ncol=2, handletextpad=0.2, columnspacing=0.8)


def plot_program_curves(ax: plt.Axes) -> None:
    curves = pd.read_csv(CURVE_TABLE)
    corr = pd.read_csv(CORR_TABLE).set_index("program")
    programs = ["Baso/Mast", "Meg/Ery", "Neutrophil", "Monocyte"]
    for program in programs:
        d = curves[curves["program"].eq(program)].sort_values("program_percentile")
        if d.empty:
            continue
        color = PROGRAM_COLOR[program]
        ax.plot(d["program_percentile"], d["barcode_future_mean"], color=color, lw=2.0, marker="o", ms=3.2, label=f"{program} barcode")
        ax.plot(d["program_percentile"], d["bmf_future_mean"], color=color, lw=1.6, ls="--", alpha=0.72, label=f"{program} BMF")
        last = d.iloc[-1]
        rb = float(corr.loc[program, "barcode_future_r"]) if program in corr.index else float("nan")
        rm = float(corr.loc[program, "bmf_future_r"]) if program in corr.index else float("nan")
        ax.text(1.012, float(last["barcode_future_mean"]), f"{program}\nr={rb:.2f}/{rm:.2f}", color=color, fontsize=6.1, ha="left", va="center")
    ax.set_xlim(0.035, 1.26)
    ax.set_ylim(-0.02, 0.72)
    ax.set_xticks([0.06, 0.50, 0.94])
    ax.set_xticklabels(["low", "mid", "high"])
    ax.set_xlabel("RNA program score rank")
    ax.set_ylabel("future fate probability")
    ax.set_title("C2  Barcode-measured and BMF-predicted futures vary with RNA programs", loc="left", pad=5)
    ax.grid(color="#E5E7EB", lw=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    style_handles = [
        mpl.lines.Line2D([0], [0], color="#111827", lw=2.0, label="barcode-measured"),
        mpl.lines.Line2D([0], [0], color="#111827", lw=1.7, ls="--", alpha=0.65, label="BMF-predicted"),
    ]
    ax.legend(handles=style_handles, frameon=False, loc="upper left")


def plot_natural_vs_model_action(ax_left: plt.Axes, ax_right: plt.Axes) -> None:
    module = pd.read_csv(MODULE_TABLE)
    perturb = pd.read_csv(PERTURB_TABLE)
    fates = ["Neutrophil", "Monocyte"]
    modules = [("early_candidate_module", "early-state genes"), ("mature_marker_module", "mature markers")]
    natural_groups = [
        ("Neutrophil", "barcode", "Neu\nbarcode"),
        ("Neutrophil", "bmf", "Neu\nBMF"),
        ("Monocyte", "barcode", "Mono\nbarcode"),
        ("Monocyte", "bmf", "Mono\nBMF"),
    ]
    x = np.arange(len(natural_groups), dtype=float)
    width = 0.32

    def natural_value(fate: str, source: str, mod_key: str) -> float:
        d = module[module["fate"].eq(fate) & module["module"].eq(mod_key) & module["source"].eq(source)]
        return float(d["r"].iloc[0])

    early_vals = [natural_value(fate, source, "early_candidate_module") for fate, source, _ in natural_groups]
    mature_vals = [natural_value(fate, source, "mature_marker_module") for fate, source, _ in natural_groups]
    group_colors = [FATE_PALETTE[fate] for fate, _source, _label in natural_groups]
    mature_face = [mpl.colors.to_rgba(color, 0.22) for color in group_colors]

    ax_left.axhline(0, color="#111827", lw=0.8)
    bars_early = ax_left.bar(x - width / 2, early_vals, width=width, color=group_colors, edgecolor=group_colors, linewidth=0.8, label="early-state genes")
    bars_mature = ax_left.bar(x + width / 2, mature_vals, width=width, color=mature_face, edgecolor=group_colors, linewidth=1.0, label="mature markers")
    add_bar_stripes(ax_left, bars_mature, group_colors)
    for bars in (bars_early, bars_mature):
        for patch in bars:
            val = float(patch.get_height())
            xpos = float(patch.get_x() + patch.get_width() / 2)
            ax_left.text(xpos, val + (0.015 if val >= 0 else -0.025), f"{val:.2f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=6.3, color="#374151")

    perturb_groups = [("Neutrophil", "Neu"), ("Monocyte", "Mono")]
    xr = np.arange(len(perturb_groups), dtype=float)
    delta_early = []
    delta_mature = []
    perturb_colors = []
    for fate, _label in perturb_groups:
        perturb_colors.append(FATE_PALETTE[fate])
        for store, mod_label in [(delta_early, "early-state genes"), (delta_mature, "mature markers")]:
            p = perturb[
                perturb["target"].eq(fate)
                & perturb["module"].eq(mod_label)
                & perturb["strength_z"].eq(5.0)
            ]
            store.append(float(p["delta_from_baseline"].iloc[0]))

    ax_right.axhline(0, color="#111827", lw=0.8)
    mature_perturb_face = [mpl.colors.to_rgba(color, 0.22) for color in perturb_colors]
    bars_delta_early = ax_right.bar(xr - width / 2, delta_early, width=width, color=perturb_colors, edgecolor=perturb_colors, linewidth=0.8, label="early-state genes")
    bars_delta_mature = ax_right.bar(xr + width / 2, delta_mature, width=width, color=mature_perturb_face, edgecolor=perturb_colors, linewidth=1.0, label="mature markers")
    add_bar_stripes(ax_right, bars_delta_mature, perturb_colors)
    for bars in (bars_delta_early, bars_delta_mature):
        for patch in bars:
            val = float(patch.get_height())
            xpos = float(patch.get_x() + patch.get_width() / 2)
            ax_right.text(xpos, val + 0.008, f"+{val:.3f}", ha="center", va="bottom", fontsize=6.5, color="#374151")

    ax_left.set_xticks(x)
    ax_left.set_xticklabels([label for *_rest, label in natural_groups])
    ax_right.set_xticks(xr)
    ax_right.set_xticklabels([label for _fate, label in perturb_groups])

    for ax in (ax_left, ax_right):
        ax.grid(axis="y", color="#E5E7EB", lw=0.7)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax_left.set_ylim(-0.05, 0.54)
    ax_left.set_ylabel("Pearson r")
    ax_left.set_title("C3  Natural signal: early > mature", loc="left", pad=5)
    ax_left.legend(
        frameon=False,
        loc="upper left",
        ncol=2,
        handlelength=1.0,
        columnspacing=0.8,
    )
    ax_right.set_ylim(0.0, 0.205)
    ax_right.set_ylabel("change in target future probability")
    ax_right.set_title("BMF +5z response", loc="left", pad=5)


def main() -> None:
    setup_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    adata = ad.read_h5ad(ADATA_PATH)
    obs = load_obs(adata)
    valid = eligible_day2_table()
    all_day2_bmf = all_day2_bmf_table()

    fig = plt.figure(figsize=(11.6, 6.0))
    outer = fig.add_gridspec(2, 2, width_ratios=[1.18, 1.08], height_ratios=[1.0, 0.82], wspace=0.22, hspace=0.34)
    left = outer[:, 0].subgridspec(1, 2, width_ratios=[1.0, 0.38], wspace=0.055)
    ax_map = fig.add_subplot(left[0, 0])
    side = left[0, 1].subgridspec(3, 1, hspace=0.12)
    ax_bmf_zoom = fig.add_subplot(side[0, 0])
    ax_truth = fig.add_subplot(side[1, 0])
    ax_vgfm = fig.add_subplot(side[2, 0])
    ax_curve = fig.add_subplot(outer[0, 1])
    bottom = outer[1, 1].subgridspec(1, 2, wspace=0.42)
    ax_nat = fig.add_subplot(bottom[0, 0])
    ax_pert = fig.add_subplot(bottom[0, 1])

    plot_fate_landscape(ax_map, ax_bmf_zoom, ax_truth, ax_vgfm, obs, all_day2_bmf, valid)
    plot_program_curves(ax_curve)
    plot_natural_vs_model_action(ax_nat, ax_pert)

    fig.suptitle("c  Early RNA states encode clonal fate bias before mature marker acquisition", x=0.018, y=0.99, ha="left", fontsize=15, weight="bold")
    fig.subplots_adjust(top=0.90, bottom=0.105, left=0.045, right=0.985)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIG_DIR / f"panel_c_bmf_early_fate_bias_story.{ext}", dpi=420 if ext == "png" else None)
    plt.close(fig)
    print(f"Wrote {FIG_DIR / 'panel_c_bmf_early_fate_bias_story.png'}")


if __name__ == "__main__":
    main()
