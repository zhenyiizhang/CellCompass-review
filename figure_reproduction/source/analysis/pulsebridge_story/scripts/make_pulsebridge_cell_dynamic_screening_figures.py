#!/usr/bin/env python3
"""Publication-style PulseBridge HCT116 dynamic screening figures.

Outputs:
- Main figure: route, metabolic validation, cancer-drug programs, and the
  perturbable adaptive/metabolic branch.
- Supplement 1: route/validation/detail panels.
- Supplement 2: gene-program perturbation and same-norm random controls.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cellcompass_pulsebridge_cell_figures")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/cellcompass_dynamo_numba")
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrow, FancyArrowPatch


ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "outputs" / "tables"
FIG = ROOT / "outputs" / "figures"
STREAM = ROOT / "outputs" / "downstream" / "dynamo_stream"

INK = "#111827"
MUTED = "#6B7280"
GRID = "#E5E7EB"
BLUE = "#245B8F"
CYAN = "#168E91"
GOLD = "#B28A00"
PURPLE = "#7D4E9B"
RED = "#B23A48"
GREEN = "#647F2E"
GRAY = "#7A7A7A"
LIGHT_BLUE = "#E6F2F6"
LIGHT_GOLD = "#F3E8B4"

TIME_CMAP = LinearSegmentedColormap.from_list(
    "pulsebridge_mist_blue_time",
    [
        (0.00, "#D9D3E8"),
        (0.32, "#AEB9DD"),
        (0.68, "#6D9AC4"),
        (1.00, "#1F6F8F"),
    ],
)

_DYNAMO_ADATA = None

CLASS_COLORS = {
    "desired": BLUE,
    "immune": CYAN,
    "metabolic": GOLD,
    "adaptive": PURPLE,
    "remodeling": GRAY,
}

STATIONS = [
    ("tumor_suppressor_reactivation", "DKK1/GAS5\nreactivation", BLUE, 0.42),
    ("ifitm3_antiviral_membrane", "IFITM3\nantiviral", CYAN, -0.34),
    ("sat1_polyamine_catabolism", "SAT1\npolyamine", GOLD, 0.48),
    ("mitochondrial_remodeling", "MT genes\nmetabolic", GREEN, -0.42),
    ("antigen_presentation_surface", "B2M/HLA\nsurface", CYAN, 0.40),
    ("lgals1_adhesion_escape", "LGALS1\nadaptive", PURPLE, -0.38),
    ("oncogene_proliferation_repression", "EREG/PCNA\nrepression", BLUE, 0.38),
]

PROGRAM_DISPLAY = [
    ("tumor_suppressor_reactivation", "DKK1/GAS5\nreactivation", "known response", BLUE),
    ("antigen_presentation_surface", "B2M/HLA\nimmune surface", "known/related", CYAN),
    ("oncogene_proliferation_repression", "EREG/PCNA\nproliferation down", "known response", BLUE),
    ("sat1_polyamine_catabolism", "SAT1\npolyamine", "adaptive/metabolic", GOLD),
    ("mitochondrial_remodeling", "MT genes\nmetabolic", "adaptive/metabolic", GREEN),
    ("lgals1_adhesion_escape", "LGALS1\nadaptive", "adaptive branch", PURPLE),
]

PERTURB_TARGETS = [
    ("reactivation", "reactivation", 5.0, "DKK1/GAS5 +5z", BLUE),
    ("immune_surface", "immune_surface", 5.0, "B2M/HLA/IFITM3 +5z", CYAN),
    ("adaptive_metabolic", "adaptive_metabolic", 5.0, "SAT1/LGALS1/MT +5z", GOLD),
    ("adaptive_metabolic", "adaptive_metabolic", -5.0, "SAT1/LGALS1/MT -5z", GOLD),
]

ADAPTIVE_PROGRAM_ORDER = [
    "source_aligned_antitumor",
    "reactivation",
    "immune_surface",
    "proliferation_repression",
    "adaptive_metabolic",
    "therapeutic_margin",
]

PROGRAM_LABELS = {
    "source_aligned_antitumor": "anti-tumor-like",
    "reactivation": "DKK1/GAS5",
    "immune_surface": "B2M/HLA/IFITM3",
    "proliferation_repression": "EREG/PCNA down",
    "adaptive_metabolic": "SAT1/LGALS1/MT",
    "therapeutic_margin": "desired-adaptive\nbalance",
}


def load_tables() -> dict[str, pd.DataFrame]:
    names = {
        "axis": "new_endpoint_vs_dynamic_axis_summary.csv",
        "endpoint": "new_endpoint_vs_dynamic_summary.csv",
        "metrics": "existing_metric_summary_long.csv",
        "timing": "pulsebridge_rollout_biology_event_timing.csv",
        "profile": "pulsebridge_rollout_neighbor_biology_profile.csv",
        "rollout_module": "pulsebridge_perturbation_module_timecourse.csv",
        "rollout_gene": "pulsebridge_perturbation_gene_delta.csv",
        "gene_align": "final_pcie_real_vs_random_gene_alignment.csv",
        "match": "story_observed_generated_match_by_day.csv",
        "targeted_day": "targeted_drug_response_module_by_day.csv",
        "perturb_it": "pulsebridge_gene_perturbation_initial_terminal_scores.csv",
        "perturb_program": "pulsebridge_gene_perturbation_program_scores.csv",
        "random": "pulsebridge_gene_perturbation_random_control_summary.csv",
    }
    return {key: pd.read_csv(TAB / name) for key, name in names.items()}


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "Arial",
            "font.sans-serif": ["Arial"],
            "font.size": 8.2,
            "axes.labelsize": 8.0,
            "axes.titlesize": 9.3,
            "axes.titleweight": "bold",
            "axes.edgecolor": INK,
            "axes.linewidth": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 6.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def panel_label(ax: plt.Axes, letter: str) -> None:
    ax.text(-0.08, 1.08, letter, transform=ax.transAxes, fontsize=13, fontweight="bold", color=INK, va="top")


def clean_grid(ax: plt.Axes, axis: str = "y") -> None:
    ax.grid(axis=axis, color=GRID, lw=0.7)
    ax.set_axisbelow(True)


def _style_dynamo_stream_ax(ax: plt.Axes, xy: np.ndarray) -> None:
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    for text in list(ax.texts):
        if "umap" in text.get_text().lower():
            text.remove()
    for patch in list(ax.patches):
        if isinstance(patch, FancyArrow) and not isinstance(patch, FancyArrowPatch):
            patch.remove()
    for collection in ax.collections:
        if isinstance(collection, LineCollection):
            collection.set_linewidth(0.62)
            collection.set_color("#1F2937")
            collection.set_alpha(0.78)
    x0, x1 = np.quantile(xy[:, 0], [0.006, 0.994])
    y0, y1 = np.quantile(xy[:, 1], [0.006, 0.994])
    xpad = 0.045 * (x1 - x0)
    ypad = 0.045 * (y1 - y0)
    ax.set_xlim(x0 - xpad, x1 + xpad)
    ax.set_ylim(y0 - ypad, y1 + ypad)


def _remove_added_axes(fig: plt.Figure, before_axes: set[plt.Axes], keep_ax: plt.Axes) -> None:
    for extra_ax in list(fig.axes):
        if extra_ax is not keep_ax and extra_ax not in before_axes:
            extra_ax.remove()


def _load_dynamo_projected_adata():
    global _DYNAMO_ADATA
    if _DYNAMO_ADATA is not None:
        return _DYNAMO_ADATA
    input_h5ad = STREAM / "pulsebridge_full_umap_velocity_for_dynamo.h5ad"
    if not input_h5ad.exists():
        raise FileNotFoundError(f"Missing Dynamo input: {input_h5ad}")

    import anndata as ad
    import dynamo as dyn

    adata = ad.read_h5ad(input_h5ad)
    required = ["X_umap", "velocity_umap", "X_latent", "velocity_latent"]
    missing = [key for key in required if key not in adata.obsm]
    if missing:
        raise KeyError(f"Missing adata.obsm keys required by Dynamo plotting: {missing}")
    _DYNAMO_ADATA = dyn.tl.cell_velocities(
        adata,
        ekey="X",
        vkey="velocity",
        X=adata.obsm["X_latent"],
        V=adata.obsm["velocity_latent"],
        X_embedding=adata.obsm["X_umap"],
        basis="umap",
        method="pearson",
        add_transition_key="velocity_cell_pearson_transition_matrix",
        add_velocity_key="velocity_cell_umap",
        enforce=True,
    )
    if "velocity_cell_umap" not in _DYNAMO_ADATA.obsm:
        raise RuntimeError("velocity_cell_umap missing after dyn.tl.cell_velocities")
    return _DYNAMO_ADATA


def main_panel_dynamo_stream(ax: plt.Axes) -> None:
    try:
        import dynamo as dyn

        adata = _load_dynamo_projected_adata()
        xy = np.asarray(adata.obsm["X_umap"], dtype=float)
        before_axes = set(ax.figure.axes)
        dyn.pl.streamline_plot(
            adata,
            basis="umap",
            vector="velocity_cell",
            color="time_point_processed",
            cmap=TIME_CMAP,
            pointsize=0.070,
            show_legend="none",
            show_arrowed_spines=False,
            method="gaussian",
            density=0.66,
            linewidth=0.62,
            streamline_alpha=0.78,
            arrowsize=0.48,
            ax=ax,
            save_show_or_return="return",
            s_kwargs_dict={"alpha": 0.22, "rasterized": True},
        )
        _remove_added_axes(ax.figure, before_axes, ax)
        _style_dynamo_stream_ax(ax, xy)
    except Exception as exc:
        fallback = FIG / "story_dynamo_native_stream_umap_black.png"
        ax.set_axis_off()
        if fallback.exists():
            ax.imshow(plt.imread(fallback))
        ax.text(
            0.02,
            0.98,
            f"Dynamo stream fallback\n{type(exc).__name__}: {exc}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.4,
            color=MUTED,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 2.0},
        )

    ax.set_title("", loc="center")
    ax.set_title("", loc="right")
    ax.set_title("PulseBridge stream in measured cell-state space", loc="left", pad=8)
    ax.text(
        0.02,
        0.94,
        "HCT116 | Well-TEMP-seq | 5-AZA-CdR day 0-3",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.1,
        color=INK,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.76, "pad": 1.7},
    )
    ax.text(
        0.02,
        0.865,
        "dark streamlines = projected PulseBridge velocity",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.6,
        color=MUTED,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.70, "pad": 1.4},
    )
    grad_ax = ax.inset_axes([0.59, 0.055, 0.30, 0.030])
    grad_ax.imshow(np.linspace(0, 1, 256)[None, :], aspect="auto", cmap=TIME_CMAP)
    grad_ax.set_axis_off()
    ax.text(0.59, 0.095, "day 0", transform=ax.transAxes, ha="left", va="bottom", fontsize=6.2, color=MUTED, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.68, "pad": 0.5})
    ax.text(0.89, 0.095, "day 3", transform=ax.transAxes, ha="right", va="bottom", fontsize=6.2, color=MUTED, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.68, "pad": 0.5})
    panel_label(ax, "B")


def panel_route(ax: plt.Axes, data: dict[str, pd.DataFrame], letter: str = "A", title: str | None = None) -> None:
    axis = data["axis"]
    endpoint = data["endpoint"]
    timing = data["timing"].set_index("module")
    gen = axis[axis["source"].eq("generated")].sort_values("time")
    obs = axis[axis["source"].eq("observed")].sort_values("time")
    ratio = float(endpoint.loc[endpoint["quantity"].eq("mean_path_to_endpoint_ratio"), "value"].iloc[0])
    path_len = float(endpoint.loc[endpoint["quantity"].eq("mean_generated_path_length"), "value"].iloc[0])
    endpoint_len = float(endpoint.loc[endpoint["quantity"].eq("mean_generated_endpoint_displacement"), "value"].iloc[0])

    ax.plot(gen["time"], gen["mean_axis"], color=BLUE, lw=3.3, solid_capstyle="round", label="PulseBridge route")
    ax.fill_between(
        gen["time"],
        gen["mean_axis"] - 0.18 * gen["std_axis"],
        gen["mean_axis"] + 0.18 * gen["std_axis"],
        color=LIGHT_BLUE,
        lw=0,
        zorder=0,
    )
    y0 = float(obs.loc[obs["time"].eq(0), "mean_axis"].iloc[0])
    y3 = float(obs.loc[obs["time"].eq(3), "mean_axis"].iloc[0])
    ax.plot([0, 3], [y0, y3], color=RED, lw=1.5, ls="--", label="endpoint shortcut")
    ax.errorbar(
        obs["time"],
        obs["mean_axis"],
        yerr=0.13 * obs["std_axis"],
        fmt="o",
        ms=4.8,
        color=INK,
        ecolor="#BDBDBD",
        capsize=2,
        zorder=5,
        label="observed days",
    )

    for module, text, color, offset in STATIONS:
        row = timing.loc[module]
        x = float(row["peak_time"])
        y = float(np.interp(x, gen["time"], gen["mean_axis"]))
        ax.scatter(x, y, s=58, color=color, edgecolor="white", lw=1.0, zorder=6)
        yy = y + offset
        ax.plot([x, x], [y, yy - 0.04 * np.sign(offset)], color=color, lw=0.8, alpha=0.78)
        ha = "right" if x > 2.82 else "center"
        tx = min(x, 2.98)
        ax.text(
            tx,
            yy,
            text,
            color=color,
            fontsize=7.0,
            fontweight="bold",
            ha=ha,
            va="center",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.35},
        )

    ax.text(
        0.015,
        0.95,
        f"{ratio:.2f}x longer route\n{path_len:.1f} path vs {endpoint_len:.1f} shortcut",
        transform=ax.transAxes,
        color=BLUE,
        fontsize=8.9,
        fontweight="bold",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "#D9E7EF", "pad": 3.0},
    )
    ax.set_xlim(-0.05, 3.10)
    ax.set_ylim(-1.07, 1.62)
    ax.set_xlabel("5-AZA-CdR response time (days)")
    ax.set_ylabel("response axis")
    ax.set_title(title or "PulseBridge converts endpoint comparison into a staged response route", loc="left", pad=6)
    ax.legend(frameon=False, ncol=3, loc="lower right", handlelength=1.6)
    clean_grid(ax, "y")
    panel_label(ax, letter)


def panel_validation(ax: plt.Axes, data: dict[str, pd.DataFrame], letter: str = "B") -> None:
    metrics = data["metrics"]
    focus = metrics[(metrics["metric"].eq("pcie")) & (metrics["algorithm"].isin(["PulseBridge", "WFR-FM"]))].copy()
    focus["algorithm"] = pd.Categorical(focus["algorithm"], ["PulseBridge", "WFR-FM"])
    focus = focus.sort_values("algorithm")
    vals = focus["value"].to_numpy(float)
    colors = [BLUE, "#B9B9B9"]
    ax.bar([0, 1], vals, color=colors, width=0.56)
    for i, val in enumerate(vals):
        ax.text(i, val + 0.035, f"{val:.3f}", ha="center", fontsize=7.7, fontweight="bold", color=colors[i])
    ax.text(0.5, 1.08, f"{100 * (1 - vals[0] / vals[1]):.1f}% lower", ha="center", va="center", color=BLUE, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["PulseBridge", "WFR-FM"])
    ax.set_ylabel("PCIE")
    ax.set_ylim(0, 1.20)
    ax.set_title("metabolic validation", loc="left")
    clean_grid(ax, "y")
    panel_label(ax, letter)


def panel_gene_alignment(ax: plt.Axes, data: dict[str, pd.DataFrame], letter: str | None = None) -> None:
    rr = data["gene_align"].set_index("gene").reindex(["MT-RNR2", "RPS2", "MT-CO1"]).reset_index()
    y = np.arange(len(rr))
    ax.hlines(y, rr["random_same"] * 100, rr["real_same"] * 100, color="#C8CDD3", lw=1.7)
    ax.scatter(rr["random_same"] * 100, y, color="#969CA3", s=30, label="random")
    ax.scatter(rr["real_same"] * 100, y, color=BLUE, s=50, edgecolor="white", linewidth=0.7, label="PulseBridge")
    for i, row in rr.iterrows():
        ax.text(row["real_same"] * 100 + 1.2, i, f"{row['real_same'] * 100:.0f}%", va="center", fontsize=7.2)
    ax.axvline(50, color="#D7DBDF", lw=0.8, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(rr["gene"])
    ax.invert_yaxis()
    ax.set_xlim(45, 103)
    ax.set_xlabel("same sign with metabolic pulse (%)")
    ax.set_title("gene-space direction", loc="left")
    ax.legend(frameon=False, loc="lower right")
    clean_grid(ax, "x")
    if letter:
        panel_label(ax, letter)


def panel_program_timeline(ax: plt.Axes, data: dict[str, pd.DataFrame], letter: str = "C") -> None:
    timing = data["timing"].set_index("module")
    rows = []
    for module, label, group, color in PROGRAM_DISPLAY:
        row = timing.loc[module]
        rows.append(
            {
                "module": module,
                "label": label,
                "group": group,
                "color": color,
                "peak_time": float(row["peak_time"]),
                "peak": float(row["peak_signed_delta"]),
                "end": float(row["end_signed_delta"]),
                "retained": float(row["retained_fraction_at_day3"]),
            }
        )
    df = pd.DataFrame(rows)
    y = np.arange(len(df))
    ax.hlines(y, 0, df["peak_time"], color="#D7DBDF", lw=1.1, zorder=0)
    sizes = 90 + 180 * (df["peak"].abs() / max(df["peak"].abs().max(), 1e-6))
    ax.scatter(df["peak_time"], y, s=sizes, color=df["color"], edgecolor="white", linewidth=1.0, zorder=3)
    for i, row in df.iterrows():
        ax.text(row["peak_time"] + 0.06, i, f"{row['peak']:.2f}", va="center", fontsize=7.0, color=row["color"], fontweight="bold")
    ax.axvspan(0, 1.2, color="#F5F7FA", zorder=-3)
    ax.axvspan(1.2, 2.2, color="#FFFFFF", zorder=-3)
    ax.axvspan(2.2, 3.05, color="#F9F4DD", zorder=-3)
    ax.text(0.55, -0.72, "early", ha="center", color=MUTED, fontsize=7.1)
    ax.text(1.70, -0.72, "middle", ha="center", color=MUTED, fontsize=7.1)
    ax.text(2.63, -0.72, "late", ha="center", color=MUTED, fontsize=7.1)
    ax.set_yticks(y)
    ax.set_yticklabels(df["label"], fontsize=7.4)
    ax.invert_yaxis()
    ax.set_xlim(-0.02, 3.12)
    ax.set_xlabel("peak time on PulseBridge route")
    ax.set_title("programs ordered on the response route", loc="left")
    clean_grid(ax, "x")
    panel_label(ax, letter)


def panel_initial_terminal(ax: plt.Axes, data: dict[str, pd.DataFrame], letter: str = "D", targets=PERTURB_TARGETS) -> None:
    it = data["perturb_it"]
    rows = []
    for module, program, strength, label, color in targets:
        row = it[
            it["perturbation_module"].eq(module)
            & it["program"].eq(program)
            & it["strength_z"].eq(strength)
        ].iloc[0]
        rows.append(
            {
                "label": label,
                "color": color,
                "initial": float(row["initial_delta_from_baseline"]),
                "terminal": float(row["terminal_delta_from_baseline"]),
                "gain": float(row["dynamic_gain_over_initial"]),
            }
        )
    df = pd.DataFrame(rows)
    y = np.arange(len(df))
    for i, row in df.iterrows():
        ax.plot([row["initial"], row["terminal"]], [i, i], color=row["color"], lw=1.6, alpha=0.82)
        ax.scatter(row["initial"], i, s=48, facecolor="white", edgecolor=row["color"], linewidth=1.4, zorder=3)
        ax.scatter(row["terminal"], i, s=54, facecolor=row["color"], edgecolor="white", linewidth=0.8, zorder=4)
    ax.axvline(0, color=INK, lw=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(df["label"], fontsize=7.3)
    ax.invert_yaxis()
    ax.set_xlabel("target-program delta from baseline")
    ax.set_title("perturbation propagation", loc="left")
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color=INK, markerfacecolor="white", lw=0, label="initial kNN"),
            Line2D([0], [0], marker="o", color=INK, markerfacecolor=INK, lw=0, label="terminal rollout"),
        ],
        frameon=False,
        loc="lower right",
        handletextpad=0.4,
    )
    clean_grid(ax, "x")
    panel_label(ax, letter)


def panel_adaptive_balance(ax: plt.Axes, data: dict[str, pd.DataFrame], letter: str = "E") -> None:
    pr = data["perturb_program"]
    keep = pr[
        pr["perturbation_module"].eq("adaptive_metabolic")
        & pr["strength_z"].isin([-5.0, 5.0])
        & pr["program"].isin(ADAPTIVE_PROGRAM_ORDER)
    ].copy()
    y = np.arange(len(ADAPTIVE_PROGRAM_ORDER))
    width = 0.34
    for off, strength, label, color in [(-width / 2, -5.0, "-5z adaptive", "#8DB6C8"), (width / 2, 5.0, "+5z adaptive", GOLD)]:
        sub = keep[keep["strength_z"].eq(strength)].set_index("program").reindex(ADAPTIVE_PROGRAM_ORDER)
        ax.barh(y + off, sub["delta_from_baseline"], height=width, color=color, edgecolor="white", linewidth=0.5, label=label)
    ax.axvline(0, color=INK, lw=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels([PROGRAM_LABELS[p] for p in ADAPTIVE_PROGRAM_ORDER], fontsize=7.2)
    ax.invert_yaxis()
    ax.set_xlabel("terminal program delta")
    ax.set_title("adaptive perturbation shifts desired/adaptive balance", loc="left")
    ax.legend(frameon=False, loc="lower right")
    clean_grid(ax, "x")
    panel_label(ax, letter)


def panel_random_control(ax: plt.Axes, data: dict[str, pd.DataFrame], letter: str = "F", compact: bool = True) -> None:
    rand = data["random"]
    specs = [
        ("reactivation", "reactivation", 5.0, "DKK1/GAS5\n+5z", BLUE),
        ("immune_surface", "immune_surface", 5.0, "B2M/HLA\n+5z", CYAN),
        ("adaptive_metabolic", "adaptive_metabolic", 5.0, "SAT1/LGALS1/MT\n+5z", GOLD),
        ("adaptive_metabolic", "therapeutic_margin", -5.0, "adaptive -5z\nbalance", PURPLE),
    ]
    rows = []
    for module, program, strength, label, color in specs:
        row = rand[
            rand["perturbation_module"].eq(module)
            & rand["program"].eq(program)
            & rand["strength_z"].eq(strength)
        ].iloc[0]
        rows.append(
            {
                "label": label,
                "actual": float(row["actual_terminal_delta"]),
                "random": float(row["random_mean_terminal_delta"]),
                "sd": float(row["random_sd_terminal_delta"]),
                "z": float(row["signed_z_vs_random"]),
                "color": color,
            }
        )
    df = pd.DataFrame(rows)
    y = np.arange(len(df))
    ax.errorbar(df["random"], y, xerr=df["sd"], fmt="o", ms=4, color="#9AA1AA", ecolor="#C9CDD2", capsize=2, label="random mean ± sd")
    ax.scatter(df["actual"], y, s=58, color=df["color"], edgecolor="white", linewidth=0.8, zorder=4, label="gene program")
    for i, row in df.iterrows():
        ax.text(row["actual"] + (0.012 if row["actual"] >= 0 else -0.012), i, f"z={row['z']:.1f}", va="center", ha="left" if row["actual"] >= 0 else "right", fontsize=6.8, color=row["color"])
    ax.axvline(0, color=INK, lw=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(df["label"], fontsize=7.1)
    ax.invert_yaxis()
    ax.set_xlabel("terminal delta vs baseline")
    ax.set_title("stronger than random pushes", loc="left")
    ax.legend(frameon=False, loc="lower right")
    clean_grid(ax, "x")
    panel_label(ax, letter)


def main_panel_route(ax: plt.Axes, data: dict[str, pd.DataFrame]) -> None:
    axis = data["axis"]
    endpoint = data["endpoint"]
    timing = data["timing"].set_index("module")
    gen = axis[axis["source"].eq("generated")].sort_values("time")
    obs = axis[axis["source"].eq("observed")].sort_values("time")
    ratio = float(endpoint.loc[endpoint["quantity"].eq("mean_path_to_endpoint_ratio"), "value"].iloc[0])
    path_len = float(endpoint.loc[endpoint["quantity"].eq("mean_generated_path_length"), "value"].iloc[0])
    endpoint_len = float(endpoint.loc[endpoint["quantity"].eq("mean_generated_endpoint_displacement"), "value"].iloc[0])

    ax.plot(gen["time"], gen["mean_axis"], color=BLUE, lw=3.4, solid_capstyle="round")
    ax.plot(
        [0, 3],
        [
            float(obs.loc[obs["time"].eq(0), "mean_axis"].iloc[0]),
            float(obs.loc[obs["time"].eq(3), "mean_axis"].iloc[0]),
        ],
        color=RED,
        ls="--",
        lw=1.5,
    )
    ax.scatter(
        obs["time"],
        obs["mean_axis"],
        s=34,
        color=INK,
        zorder=5,
    )
    for _, row in obs.iterrows():
        ax.text(
            row["time"],
            row["mean_axis"] - 0.13,
            f"d{int(row['time'])}",
            ha="center",
            va="top",
            fontsize=6.2,
            color=INK,
            fontweight="bold",
        )

    station_specs = [
        ("tumor_suppressor_reactivation", "DKK1/GAS5\nreactivation", BLUE, 0.34),
        ("antigen_presentation_surface", "B2M/HLA\nimmune surface", CYAN, 0.32),
        ("sat1_polyamine_catabolism", "SAT1/LGALS1/MT\nadaptive-metabolic", GOLD, 0.50),
        ("oncogene_proliferation_repression", "EREG/PCNA\nrepression", BLUE, 0.20),
    ]
    for module, label_text, color, dy in station_specs:
        row = timing.loc[module]
        x = float(row["peak_time"])
        y = float(np.interp(x, gen["time"], gen["mean_axis"]))
        ax.scatter(x, y, s=64, color=color, edgecolor="white", lw=1.0, zorder=6)
        yy = y + dy
        tx = min(x, 2.98)
        ha = "right" if x > 2.80 else "center"
        ax.plot([x, x], [y, yy - 0.04], color=color, lw=0.8)
        ax.text(
            tx,
            yy,
            label_text,
            color=color,
            ha=ha,
            va="bottom",
            fontsize=7.2,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.55},
        )

    ax.text(
        0.02,
        0.94,
        f"{ratio:.2f}x longer route\n{path_len:.1f} path vs {endpoint_len:.1f} shortcut",
        transform=ax.transAxes,
        color=BLUE,
        fontsize=8.8,
        fontweight="bold",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "#D9E7EF", "pad": 3.0},
    )
    ax.text(
        1.05,
        0.20,
        "PulseBridge route",
        color=BLUE,
        fontsize=7.1,
        fontweight="bold",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.1},
    )
    shortcut_y = float(
        np.interp(
            1.42,
            [0, 3],
            [
                float(obs.loc[obs["time"].eq(0), "mean_axis"].iloc[0]),
                float(obs.loc[obs["time"].eq(3), "mean_axis"].iloc[0]),
            ],
        )
    )
    ax.text(
        1.55,
        shortcut_y - 0.36,
        "endpoint shortcut",
        color=RED,
        fontsize=7.0,
        fontweight="bold",
        ha="center",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.1},
    )
    ax.text(
        0.985,
        0.055,
        "HCT116 colorectal cancer\n5-AZA-CdR + metabolic RNA labeling",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.0,
        color=INK,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.80, "pad": 2.0},
    )
    ax.set_xlim(-0.05, 3.10)
    ax.set_ylim(-1.05, 1.62)
    ax.set_xlabel("5-AZA-CdR response time (days)")
    ax.set_ylabel("response axis")
    ax.set_title("PulseBridge learns a staged drug-response route", loc="left", pad=6)
    clean_grid(ax, "y")
    panel_label(ax, "A")


def main_panel_validation(ax: plt.Axes, data: dict[str, pd.DataFrame]) -> None:
    metrics = data["metrics"]
    focus = metrics[(metrics["metric"].eq("pcie")) & (metrics["algorithm"].isin(["PulseBridge", "WFR-FM"]))].copy()
    focus["algorithm"] = pd.Categorical(focus["algorithm"], ["PulseBridge", "WFR-FM"])
    focus = focus.sort_values("algorithm")
    vals = focus["value"].to_numpy(float)
    x = np.array([0.0, 0.82])
    ax.bar(x, vals, width=0.46, color=[BLUE, "#B9B9B9"])
    for xx, val, color in zip(x, vals, [BLUE, "#A4A4A4"]):
        ax.text(xx, val + 0.035, f"{val:.3f}", ha="center", color=color, fontsize=8.0, fontweight="bold")
    ax.text(np.mean(x), 1.10, "31.1% lower", ha="center", color=BLUE, fontweight="bold", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(["PulseBridge", "WFR-FM"])
    ax.set_ylabel("PCIE")
    ax.set_ylim(0, 1.22)
    ax.set_title("direction validated by metabolic RNA", loc="left")
    clean_grid(ax, "y")
    panel_label(ax, "B")


def main_panel_programs(ax: plt.Axes, data: dict[str, pd.DataFrame]) -> None:
    profile = data["profile"]
    specs = [
        (["tumor_suppressor_reactivation"], "DKK1/GAS5\nreactivation", BLUE, 0.035, 0.260),
        (["oncogene_proliferation_repression"], "EREG/PCNA\nrepression", RED, 0.020, 0.370),
        (["antigen_presentation_surface", "ifitm3_antiviral_membrane"], "B2M/HLA/IFITM3\nimmune", CYAN, 0.030, 0.175),
        (["sat1_polyamine_catabolism", "lgals1_adhesion_escape", "mitochondrial_remodeling"], "SAT1/LGALS1/MT\nadaptive", GOLD, 0.030, 0.630),
    ]

    def combined_curve(modules: list[str]) -> pd.DataFrame:
        sub = profile[profile["module"].isin(modules)]
        curve = (
            sub.pivot_table(index="time", values="signed_delta_from_t0", aggfunc="mean")
            .rename(columns={"signed_delta_from_t0": "value"})
            .reset_index()
            .sort_values("time")
        )
        return curve

    ax.axhline(0, color=INK, lw=0.78, zorder=1)
    for modules, label_text, color, text_dy, label_y in specs:
        curve = combined_curve(modules)
        is_red = color == RED
        ax.plot(curve["time"], curve["value"], color=color, lw=2.65 if is_red else 2.25, solid_capstyle="round", zorder=6 if is_red else 4)
        ax.fill_between(curve["time"], curve["value"], 0, color=color, alpha=0.050 if is_red else 0.055, lw=0, zorder=2)
        peak = curve.loc[curve["value"].idxmax()]
        end = curve[curve["time"].eq(3.0)].iloc[0]
        ax.scatter([peak["time"]], [peak["value"]], s=34, color=color, edgecolor="white", linewidth=0.8, zorder=7 if is_red else 5)
        ax.scatter([end["time"]], [end["value"]], s=32, marker="s", color=color, edgecolor="white", linewidth=0.8, zorder=7 if is_red else 5)
        if peak["time"] < 2.95:
            ax.text(
                peak["time"],
                peak["value"] + text_dy,
                f"peak d{peak['time']:.1f}",
                color=color,
                fontsize=6.2,
                ha="center",
                va="bottom",
                fontweight="bold",
            )
        ax.text(
            3.03,
            label_y,
            f"{label_text}  D3 {end['value']:+.2f}",
            color=color,
            fontsize=6.15,
            ha="left",
            va="center",
            fontweight="bold",
        )
    ax.text(
        0.02,
        0.95,
        "positive = expected response direction",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.8,
        color=MUTED,
    )
    ax.set_xlim(-0.05, 3.34)
    ax.set_ylim(-0.16, 0.92)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xlabel("PulseBridge route time (days)")
    ax.set_ylabel("program change from day0")
    ax.set_title("source-paper programs become timed route waves", loc="left")
    clean_grid(ax, "y")
    panel_label(ax, "C")


def main_panel_adaptive(ax: plt.Axes, data: dict[str, pd.DataFrame]) -> None:
    pr = data["perturb_program"]
    order = [
        "reactivation",
        "immune_surface",
        "proliferation_repression",
        "source_aligned_antitumor",
        "adaptive_metabolic",
        "therapeutic_margin",
    ]
    labels = [
        "DKK1/GAS5\nreactivation",
        "B2M/HLA/IFITM3\nimmune surface",
        "EREG/PCNA\nproliferation down",
        "anti-tumor-like\nresponse",
        "SAT1/LGALS1/MT\nadaptive branch",
        "desired-adaptive\nbalance",
    ]
    keep = pr[
        pr["perturbation_module"].eq("adaptive_metabolic")
        & pr["strength_z"].isin([-5.0, 5.0])
        & pr["program"].isin(order)
    ]
    y = np.arange(len(order))
    width = 0.32
    specs = [
        (-width / 2, -5.0, "lower SAT1/LGALS1/MT", "#8DB6C8"),
        (width / 2, 5.0, "higher SAT1/LGALS1/MT", GOLD),
    ]
    for off, strength, label_text, color in specs:
        sub = keep[keep["strength_z"].eq(strength)].set_index("program").reindex(order)
        ax.barh(y + off, sub["delta_from_baseline"], height=width, color=color, edgecolor="white", linewidth=0.6, label=label_text)
        for yi, val in zip(y + off, sub["delta_from_baseline"]):
            ha = "left" if val >= 0 else "right"
            dx = 0.008 if val >= 0 else -0.008
            ax.text(val + dx, yi, f"{val:+.2f}", va="center", ha=ha, fontsize=5.8, color=color, fontweight="bold")
    ax.axvline(0, color=INK, lw=0.85)
    ax.text(
        0.98,
        0.02,
        "balance = anti-tumor-like − adaptive",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.5,
        color=PURPLE,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.4},
    )
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.1)
    ax.invert_yaxis()
    ax.set_xlim(-0.27, 0.23)
    ax.set_xlabel("predicted day3 program change vs unperturbed route")
    ax.set_title("SAT1/LGALS1/MT branch controls response trade-off", loc="left")
    ax.legend(frameon=False, loc="upper right", fontsize=6.2, handlelength=1.8)
    clean_grid(ax, "x")
    panel_label(ax, "D")


def panel_observed_generated_match(ax: plt.Axes, data: dict[str, pd.DataFrame], letter: str = "A") -> None:
    df = data["match"]
    ax.plot(df["time"], df["observed_axis_mean"], color=INK, marker="o", lw=1.8, label="observed")
    ax.plot(df["time"], df["generated_axis_mean"], color=BLUE, marker="o", lw=2.0, label="generated")
    ax.fill_between(df["time"], df["observed_axis_mean"] - 0.12 * df["observed_axis_std"], df["observed_axis_mean"] + 0.12 * df["observed_axis_std"], color="#EDEFF2", lw=0)
    ax.fill_between(df["time"], df["generated_axis_mean"] - 0.12 * df["generated_axis_std"], df["generated_axis_mean"] + 0.12 * df["generated_axis_std"], color=LIGHT_BLUE, lw=0)
    ax.set_xlabel("day")
    ax.set_ylabel("response axis")
    ax.set_title("generated and observed response days", loc="left")
    ax.set_xticks([0, 1, 2, 3])
    ax.legend(frameon=False, loc="lower right")
    clean_grid(ax, "y")
    panel_label(ax, letter)


def panel_program_profiles(ax: plt.Axes, data: dict[str, pd.DataFrame], letter: str = "C") -> None:
    profile = data["profile"]
    mods = [
        ("tumor_suppressor_reactivation", "DKK1/GAS5", BLUE),
        ("antigen_presentation_surface", "B2M/HLA", CYAN),
        ("oncogene_proliferation_repression", "EREG/PCNA down", BLUE),
        ("sat1_polyamine_catabolism", "SAT1", GOLD),
        ("mitochondrial_remodeling", "MT", GREEN),
        ("lgals1_adhesion_escape", "LGALS1", PURPLE),
    ]
    for module, label, color in mods:
        sub = profile[profile["module"].eq(module)].sort_values("time")
        ax.plot(sub["time"], sub["signed_delta_from_t0"], lw=1.9, color=color, label=label)
    ax.axhline(0, color=INK, lw=0.75)
    ax.set_xlabel("route time")
    ax.set_ylabel("signed delta from day0")
    ax.set_title("program waves on the generated route", loc="left")
    ax.legend(frameon=False, ncol=2, loc="upper left", bbox_to_anchor=(0.0, -0.18))
    clean_grid(ax, "y")
    panel_label(ax, letter)


def panel_gene_shifts(ax: plt.Axes, data: dict[str, pd.DataFrame], letter: str = "D") -> None:
    genes = ["SAT1", "MT-CO1", "LGALS1", "B2M", "DKK1", "PCNA", "EREG"]
    colors = {"SAT1": GOLD, "MT-CO1": GREEN, "LGALS1": PURPLE, "B2M": CYAN, "DKK1": BLUE, "PCNA": RED, "EREG": RED}
    df = data["rollout_gene"].set_index("gene").reindex(genes).reset_index()
    y = np.arange(len(df))
    ax.barh(y, df["pulsebridge_day3_minus_day0"], color=[colors[g] for g in df["gene"]], height=0.58)
    ax.axvline(0, color=INK, lw=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(df["gene"])
    ax.invert_yaxis()
    ax.set_xlabel("rollout day3 - day0")
    ax.set_title("gene readouts behind programs", loc="left")
    clean_grid(ax, "x")
    panel_label(ax, letter)


def panel_all_initial_terminal(ax: plt.Axes, data: dict[str, pd.DataFrame], letter: str = "A") -> None:
    targets = [
        ("reactivation", "reactivation", -5.0, "DKK1/GAS5 -5z", BLUE),
        ("reactivation", "reactivation", 5.0, "DKK1/GAS5 +5z", BLUE),
        ("immune_surface", "immune_surface", -5.0, "B2M/HLA -5z", CYAN),
        ("immune_surface", "immune_surface", 5.0, "B2M/HLA +5z", CYAN),
        ("proliferation_genes", "proliferation_repression", -5.0, "EREG/PCNA -5z", RED),
        ("proliferation_genes", "proliferation_repression", 5.0, "EREG/PCNA +5z", RED),
        ("adaptive_metabolic", "adaptive_metabolic", -5.0, "SAT1/LGALS1/MT -5z", GOLD),
        ("adaptive_metabolic", "adaptive_metabolic", 5.0, "SAT1/LGALS1/MT +5z", GOLD),
    ]
    panel_initial_terminal(ax, data, letter=letter, targets=targets)
    ax.set_title("direct perturbation versus terminal rollout", loc="left")


def panel_random_full(ax: plt.Axes, data: dict[str, pd.DataFrame], letter: str = "C") -> None:
    rand = data["random"]
    specs = [
        ("reactivation", "reactivation", -5.0, "DKK1/GAS5 -5z", BLUE),
        ("reactivation", "reactivation", 5.0, "DKK1/GAS5 +5z", BLUE),
        ("immune_surface", "immune_surface", -5.0, "B2M/HLA -5z", CYAN),
        ("immune_surface", "immune_surface", 5.0, "B2M/HLA +5z", CYAN),
        ("adaptive_metabolic", "adaptive_metabolic", -5.0, "adaptive -5z", GOLD),
        ("adaptive_metabolic", "adaptive_metabolic", 5.0, "adaptive +5z", GOLD),
        ("adaptive_metabolic", "therapeutic_margin", -5.0, "balance after adaptive -5z", PURPLE),
        ("adaptive_metabolic", "therapeutic_margin", 5.0, "balance after adaptive +5z", PURPLE),
        ("proliferation_genes", "proliferation_repression", -5.0, "EREG/PCNA -5z", RED),
        ("proliferation_genes", "proliferation_repression", 5.0, "EREG/PCNA +5z", RED),
    ]
    rows = []
    for module, program, strength, label, color in specs:
        row = rand[
            rand["perturbation_module"].eq(module)
            & rand["program"].eq(program)
            & rand["strength_z"].eq(strength)
        ].iloc[0].to_dict()
        row["label"] = label
        row["color"] = color
        rows.append(row)
    show = pd.DataFrame(rows)
    y = np.arange(len(show))
    ax.errorbar(show["random_mean_terminal_delta"], y, xerr=show["random_sd_terminal_delta"], fmt="o", ms=3.4, color="#9AA1AA", ecolor="#C9CDD2", capsize=2, label="random")
    ax.scatter(show["actual_terminal_delta"], y, s=40, color=show["color"], edgecolor="white", linewidth=0.6, label="actual")
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(show["label"], fontsize=6.1)
    ax.invert_yaxis()
    ax.set_xlabel("terminal delta")
    ax.set_title("same-norm random-push controls", loc="left")
    ax.legend(frameon=False, loc="lower right")
    clean_grid(ax, "x")
    panel_label(ax, letter)


def write_panel_summary(data: dict[str, pd.DataFrame]) -> None:
    endpoint = data["endpoint"]
    metrics = data["metrics"]
    pr = data["perturb_program"]
    ratio = float(endpoint.loc[endpoint["quantity"].eq("mean_path_to_endpoint_ratio"), "value"].iloc[0])
    p = float(metrics[(metrics["metric"].eq("pcie")) & metrics["algorithm"].eq("PulseBridge")]["value"].iloc[0])
    w = float(metrics[(metrics["metric"].eq("pcie")) & metrics["algorithm"].eq("WFR-FM")]["value"].iloc[0])

    def adaptive_delta(strength: float, program: str) -> float:
        return float(
            pr[
                pr["perturbation_module"].eq("adaptive_metabolic")
                & pr["program"].eq(program)
                & pr["strength_z"].eq(strength)
            ]["delta_from_baseline"].iloc[0]
        )

    rows = [
        ("main_A", f"PulseBridge route is {ratio:.2f}x longer than endpoint shortcut.", "Dynamic route, not endpoint jump."),
        ("main_B", f"Projected PulseBridge velocity stream is shown in full-cell UMAP space; quantitative PCIE remains {p:.3f} vs WFR-FM {w:.3f}.", "PulseBridge defines a directional cancer-cell response flow in the measured state space."),
        ("main_C", "DKK1/GAS5 peaks near day1.6, immune remodeling near day2.4, SAT1/LGALS1/MT near day2.3, and EREG/PCNA repression is retained at day3.", "Source-paper programs become ordered route waves rather than endpoint-only differences."),
        (
            "main_D",
            f"Higher SAT1/LGALS1/MT changes DKK1/GAS5 {adaptive_delta(5.0, 'reactivation'):+.3f}, B2M/HLA/IFITM3 {adaptive_delta(5.0, 'immune_surface'):+.3f}, desired-adaptive balance {adaptive_delta(5.0, 'therapeutic_margin'):+.3f}; lower SAT1/LGALS1/MT changes balance {adaptive_delta(-5.0, 'therapeutic_margin'):+.3f}.",
            "SAT1/LGALS1/MT branch is a coupled adaptive-response axis: it tracks reactivation/immune remodeling but shifts the response toward adaptation.",
        ),
    ]
    pd.DataFrame(rows, columns=["panel", "evidence", "message"]).to_csv(TAB / "pulsebridge_dynamic_screening_figure_summary.csv", index=False)


def make_main(data: dict[str, pd.DataFrame]) -> None:
    fig = plt.figure(figsize=(12.6, 8.2), dpi=240)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0], height_ratios=[1.02, 1.0], wspace=0.42, hspace=0.42)
    main_panel_route(fig.add_subplot(gs[0, 0]), data)
    main_panel_dynamo_stream(fig.add_subplot(gs[0, 1]))
    main_panel_programs(fig.add_subplot(gs[1, 0]), data)
    main_panel_adaptive(fig.add_subplot(gs[1, 1]), data)
    fig.savefig(FIG / "story_pulsebridge_dynamic_screening_main.png", bbox_inches="tight")
    fig.savefig(FIG / "story_pulsebridge_dynamic_screening_main.pdf", bbox_inches="tight")
    fig.savefig(FIG / "story_pulsebridge_dynamic_screening_main_compact.png", bbox_inches="tight")
    fig.savefig(FIG / "story_pulsebridge_dynamic_screening_main_compact.pdf", bbox_inches="tight")
    plt.close(fig)


def make_supp_route(data: dict[str, pd.DataFrame]) -> None:
    fig = plt.figure(figsize=(14.6, 8.7), dpi=220)
    gs = fig.add_gridspec(2, 3, wspace=0.62, hspace=0.46)
    panel_observed_generated_match(fig.add_subplot(gs[0, 0]), data, "A")
    panel_validation(fig.add_subplot(gs[0, 1]), data, "B")
    panel_gene_alignment(fig.add_subplot(gs[0, 2]), data, "C")
    panel_program_profiles(fig.add_subplot(gs[1, 0:2]), data, "D")
    panel_gene_shifts(fig.add_subplot(gs[1, 2]), data, "E")
    fig.savefig(FIG / "story_pulsebridge_dynamic_screening_supp_route_validation.png", bbox_inches="tight")
    fig.savefig(FIG / "story_pulsebridge_dynamic_screening_supp_route_validation.pdf", bbox_inches="tight")
    plt.close(fig)


def make_supp_perturb(data: dict[str, pd.DataFrame]) -> None:
    fig = plt.figure(figsize=(15.8, 9.2), dpi=220)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.16, 1.0, 1.18], wspace=0.78, hspace=0.72)
    panel_all_initial_terminal(fig.add_subplot(gs[:, 0]), data, "A")
    panel_adaptive_balance(fig.add_subplot(gs[0, 1]), data, "B")
    panel_random_control(fig.add_subplot(gs[0, 2]), data, "C")
    panel_random_full(fig.add_subplot(gs[1, 1:3]), data, "D")
    fig.savefig(FIG / "story_pulsebridge_dynamic_screening_supp_perturbation_controls.png", bbox_inches="tight")
    fig.savefig(FIG / "story_pulsebridge_dynamic_screening_supp_perturbation_controls.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    style()
    data = load_tables()
    make_main(data)
    make_supp_route(data)
    make_supp_perturb(data)
    write_panel_summary(data)
    print(FIG / "story_pulsebridge_dynamic_screening_main.png")
    print(FIG / "story_pulsebridge_dynamic_screening_supp_route_validation.png")
    print(FIG / "story_pulsebridge_dynamic_screening_supp_perturbation_controls.png")


if __name__ == "__main__":
    main()
