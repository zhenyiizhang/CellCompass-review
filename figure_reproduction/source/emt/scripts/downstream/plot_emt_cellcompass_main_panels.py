#!/usr/bin/env python3
"""Draw main-figure EMT panels from CellCompass outputs only.

This script is intentionally a plotting layer. It reads the locked
CellCompass downstream tables and mechanism tables, and does not rerun models
or define new biological route classes.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import wrap

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[3]
CONDITION_DIR = ROOT / "emt" / "downstream" / "emt_condition_dynamics"
MECH_DIR = ROOT / "emt" / "downstream" / "emt_mechanisms"
OUT_DIR = ROOT / "emt" / "figures" / "emt_cellcompass_main_panels"

CELL_LINES = ["A549", "DU145", "MCF7", "OVCA420"]
STIMULI = ["EGF", "TGFB1", "TNF"]
TIME_LABELS = ["0d", "8h", "1d", "3d", "7d"]
CONDITION_ORDER = [f"{cell}_{stim}" for cell in CELL_LINES for stim in STIMULI]
CONDITION_IDS = {condition: i + 1 for i, condition in enumerate(CONDITION_ORDER)}

INK = "#101828"
TEXT = "#344054"
MUTED = "#667085"
GRID = "#EAECF0"
AXIS = "#475467"
OBSERVED_COLOR = "#2F80C0"
GENERATED_COLOR = "#E69500"

ROUTE_COLORS = {
    "mesenchymal conversion": "#C7363D",
    "partial/hybrid EMT": "#E5922C",
    "growth-retaining response": "#1D9A9A",
    "epithelial-retaining response": "#506B8F",
    "mixed/weak EMT route": "#7B61B9",
}

MECH_COLORS = {
    "TGFbeta/EMT-TF/ECM conversion axis": "#B8323C",
    "partial-basal hybrid program with epithelial retention": "#D9892B",
    "NFkB/AP-1 inflammatory route": "#1D9A9A",
    "mixed context-specific route": "#65758B",
}

MECH_LABELS = {
    "TGFβ/EMT-TF/ECM conversion axis": "TGF-beta / EMT-TF / ECM",
    "TGFbeta/EMT-TF/ECM conversion axis": "TGF-beta / EMT-TF / ECM",
    "partial-basal hybrid program with epithelial retention": "partial-basal hybrid",
    "NFκB/AP-1 inflammatory route": "NF-kB inflammatory",
    "NFkB/AP-1 inflammatory route": "NF-kB inflammatory",
    "mixed context-specific route": "mixed context-specific",
}

MODULE_LABELS = {
    "TGFB_SMAD_targets": "TGF-beta/SMAD",
    "EMT_TF_axis": "EMT-TF",
    "ECM_remodeling": "ECM",
    "epithelial_junction": "epithelial junction",
    "partial_basal_hybrid": "partial/basal",
    "TNF_NFKB_inflammatory": "TNF/NF-kB",
    "EGFR_AP1_MAPK": "EGFR/AP-1",
    "cell_cycle_growth": "growth",
    "interferon_JAK_STAT": "IFN/JAK-STAT",
    "stress_p53_apoptosis": "stress/p53",
}

MODULE_COLORS = {
    "TGFB_SMAD_targets": "#B8323C",
    "EMT_TF_axis": "#C9435A",
    "ECM_remodeling": "#D9892B",
    "epithelial_junction": "#506B8F",
    "partial_basal_hybrid": "#E5922C",
    "TNF_NFKB_inflammatory": "#1D9A9A",
    "EGFR_AP1_MAPK": "#4D95D8",
    "cell_cycle_growth": "#5E6C84",
    "interferon_JAK_STAT": "#7B61B9",
    "stress_p53_apoptosis": "#8E4B3A",
}

FACTOR_COLORS = {
    "cell-line identity": "#4E79A7",
    "stimulus": "#F28E2B",
    "cell-line x stimulus context": "#59A14F",
}

MECH_MODULES = [
    "TGFB_SMAD_targets",
    "EMT_TF_axis",
    "ECM_remodeling",
    "epithelial_junction",
    "partial_basal_hybrid",
    "TNF_NFKB_inflammatory",
    "EGFR_AP1_MAPK",
    "cell_cycle_growth",
]

LAYER_MODULES = [
    ("Stimulus-proximal pathway", ["TNF_NFKB_inflammatory", "TGFB_SMAD_targets", "EGFR_AP1_MAPK", "ECM_remodeling"]),
    ("Cancer-cell state gate", ["epithelial_junction", "partial_basal_hybrid", "interferon_JAK_STAT"]),
    ("Growth / stress layer", ["cell_cycle_growth", "stress_p53_apoptosis"]),
]

ROUTE_DIAGNOSTIC_MODULES = {
    "TGFβ/EMT-TF/ECM conversion axis": [
        "TGFB_SMAD_targets",
        "EMT_TF_axis",
        "ECM_remodeling",
        "epithelial_junction",
    ],
    "partial-basal hybrid program with epithelial retention": [
        "partial_basal_hybrid",
        "epithelial_junction",
        "interferon_JAK_STAT",
        "stress_p53_apoptosis",
    ],
    "NFκB/AP-1 inflammatory route": [
        "TNF_NFKB_inflammatory",
        "EMT_TF_axis",
        "ECM_remodeling",
        "partial_basal_hybrid",
    ],
}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial"],
            "font.size": 8,
            "axes.titlesize": 9.2,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "axes.edgecolor": TEXT,
            "text.color": INK,
        }
    )


def save_all(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT_DIR / f"{stem}.{ext}", bbox_inches="tight", facecolor="white")


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    route = pd.read_csv(CONDITION_DIR / "condition_route_summary.csv")
    route["display_route"] = route.apply(display_route_group, axis=1)
    snap = pd.read_csv(CONDITION_DIR / "endpoint_and_measured_snapshot_comparison.csv")
    mech = pd.read_csv(MECH_DIR / "mechanism_condition_summary.csv")
    traj = pd.read_csv(MECH_DIR / "mechanism_trajectory_modules.csv")
    var = pd.read_csv(MECH_DIR / "mechanism_variance_decomposition.csv")
    return route, snap, mech, traj, var


def display_route_group(row: pd.Series) -> str:
    """Biology-facing visualization bin from CellCompass output fields.

    This is only a display grouping for EMT-focused main figures. It uses
    generated endpoint EMT-core and hybrid deltas already reported by
    CellCompass, matching the x/y axes in the route atlas. Endpoint cell count
    is deliberately not used for the main route color because scRNA-seq cell
    recovery can mix biology and sampling. It does not replace the source
    route_class in the report.
    """

    core = row["generated_delta_emt_core_score"]
    hybrid = row["generated_delta_hybrid_score"]
    if core >= 0.30:
        return "mesenchymal conversion"
    if hybrid >= 0.18:
        return "partial/hybrid EMT"
    if abs(core) < 0.08 and hybrid < 0.12:
        return "epithelial-retaining response"
    return "mixed/weak EMT route"


def pretty_condition(condition: str) -> str:
    return condition.replace("_", "+")


def condition_key_text(columns: int = 2) -> str:
    labels = [f"{CONDITION_IDS[c]}. {pretty_condition(c)}" for c in CONDITION_ORDER]
    if columns == 1:
        return "\n".join(labels)
    half = int(np.ceil(len(labels) / columns))
    rows = []
    for i in range(half):
        left = labels[i]
        right = labels[i + half] if i + half < len(labels) else ""
        rows.append(f"{left:<17} {right}")
    return "\n".join(rows)


def wrap_text(text: str, width: int = 34) -> str:
    return "\n".join(wrap(text, width=width))


def add_card(
    ax: plt.Axes,
    xy: tuple[float, float],
    wh: tuple[float, float],
    title: str,
    body: str,
    color: str,
    *,
    title_size: float = 9.5,
    body_size: float = 7.5,
    wrap_width: int = 38,
) -> None:
    x, y = xy
    w, h = wh
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.006",
            fc="white",
            ec="#D0D5DD",
            lw=0.8,
            transform=ax.transAxes,
        )
    )
    ax.add_patch(Rectangle((x, y + h - 0.012), w, 0.012, fc=color, ec="none", transform=ax.transAxes))
    ax.text(x + 0.025, y + h - 0.06, title, transform=ax.transAxes, ha="left", va="top", fontsize=title_size, fontweight="bold", color=INK)
    ax.text(x + 0.025, y + h - 0.135, wrap_text(body, wrap_width), transform=ax.transAxes, ha="left", va="top", fontsize=body_size, color=TEXT, linespacing=1.20)


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = "#98A2B3", lw: float = 0.9) -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=11, lw=lw, color=color, transform=ax.transAxes))


def panel01_dialogue_loop() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11.2, 5.7))
    ax.axis("off")
    ax.text(0.02, 0.965, "Human-CellCompass analysis loop for cancer EMT", fontsize=13.2, fontweight="bold", color=INK, ha="left", va="top", transform=ax.transAxes)
    ax.text(0.02, 0.885, "Actual EMT run: 12 condition-specific VGFM models followed by mechanism-focused CellCompass follow-up.", fontsize=8.2, color=AXIS, ha="left", transform=ax.transAxes)

    # Dataset strip.
    add_card(
        ax,
        (0.035, 0.55),
        (0.25, 0.24),
        "Biological request",
        "Build context-aware virtual-cell dynamics for the Cook and Vanderhyden EMT time course.",
        "#135E96",
        wrap_width=31,
    )
    ax.text(0.045, 0.49, "cell lines: A549, DU145, MCF7, OVCA420", fontsize=7.6, color=TEXT, transform=ax.transAxes)
    ax.text(0.045, 0.43, "stimuli: EGF, TGFB1, TNF", fontsize=7.6, color=TEXT, transform=ax.transAxes)
    ax.text(0.045, 0.37, "time: 0d, 8h, 1d, 3d, 7d", fontsize=7.6, color=TEXT, transform=ax.transAxes)

    arrow(ax, (0.305, 0.69), (0.365, 0.69))
    ax.text(0.465, 0.805, "CellCompass", fontsize=12.2, fontweight="bold", color="#135E96", ha="center", transform=ax.transAxes)
    for theta, color in zip(np.linspace(0, 2 * np.pi, 8, endpoint=False), ["#135E96", "#1D9A9A"] * 4):
        ax.plot([0.465, 0.465 + 0.027 * np.cos(theta)], [0.74, 0.74 + 0.027 * np.sin(theta)], color=color, lw=1.3, transform=ax.transAxes)
    ax.add_patch(Circle((0.465, 0.74), 0.010, fc="#E9F5FF", ec="#135E96", lw=1.1, transform=ax.transAxes))

    add_card(
        ax,
        (0.355, 0.51),
        (0.28, 0.22),
        "Round 1 output",
        "Train one VGFM per cell line x stimulus; compare generated endpoints with observed EMT programs.",
        "#135E96",
        body_size=6.7,
        wrap_width=43,
    )
    add_card(
        ax,
        (0.355, 0.22),
        (0.28, 0.22),
        "Round 2 output",
        "Ask why routes differ; annotate rollouts with pathway, EMT, epithelial, hybrid, growth and stress modules.",
        "#135E96",
        body_size=6.7,
        wrap_width=43,
    )

    arrow(ax, (0.655, 0.69), (0.705, 0.69))
    add_card(
        ax,
        (0.72, 0.56),
        (0.25, 0.22),
        "Route-level result",
        "The same EMT stimulus produces different virtual-cell routes across cancer-cell contexts.",
        "#135E96",
        body_size=7.0,
        wrap_width=30,
    )
    add_card(
        ax,
        (0.72, 0.28),
        (0.25, 0.22),
        "Mechanism-level result",
        "Route specificity is linked to stimulus pathway, cancer-cell state gate and growth/mass coupling.",
        "#135E96",
        body_size=7.0,
        wrap_width=30,
    )

    # Small process ladder at bottom.
    xs = [0.18, 0.38, 0.58, 0.78]
    labs = ["question", "condition-specific\nmodels", "route atlas", "mechanism\nhypotheses"]
    for i, (x, lab) in enumerate(zip(xs, labs)):
        ax.add_patch(Circle((x, 0.11), 0.025, fc="#F2F4F7", ec="#98A2B3", lw=1, transform=ax.transAxes))
        ax.text(x, 0.055, lab, ha="center", va="top", fontsize=7.4, color=TEXT, transform=ax.transAxes)
        if i < len(xs) - 1:
            arrow(ax, (x + 0.032, 0.11), (xs[i + 1] - 0.032, 0.11), color="#98A2B3", lw=1.0)

    save_all(fig, "panel01_human_cellcompass_emt_loop")
    return fig


def panel02_route_cards(route: pd.DataFrame, snap: pd.DataFrame) -> plt.Figure:
    fig = plt.figure(figsize=(10.9, 7.2))
    gs = fig.add_gridspec(5, 4, height_ratios=[0.58, 1, 1, 1, 1], width_ratios=[0.34, 1, 1, 1], hspace=0.34, wspace=0.20)
    title_ax = fig.add_subplot(gs[0, :])
    title_ax.axis("off")
    title_ax.text(0.0, 0.88, "CellCompass calls condition-specific EMT routes", fontsize=13.4, fontweight="bold", color="#101828", ha="left")
    title_ax.text(0.0, 0.58, "Observed and generated EMT-core trajectories are compared within each condition.", fontsize=8.4, color="#475467", ha="left")
    title_ax.text(
        0.0,
        0.34,
        "Display route = visualization bin from CellCompass outputs: endpoint EMT-core/mesenchymal deltas + observed 7d/0d mass ratio.",
        fontsize=7.1,
        color="#667085",
        ha="left",
    )

    x = np.arange(len(TIME_LABELS))
    for i, cell in enumerate(CELL_LINES, start=1):
        la = fig.add_subplot(gs[i, 0])
        la.axis("off")
        la.text(0.98, 0.52, cell, ha="right", va="center", fontsize=12, fontweight="bold", color=INK)
        for j, stim in enumerate(STIMULI, start=1):
            cond = f"{cell}_{stim}"
            row = route.loc[route["condition"] == cond].iloc[0]
            c = ROUTE_COLORS.get(row["display_route"], "#667085")
            sub = snap.loc[snap["condition"] == cond].sort_values("time_index")
            obs = sub["observed_emt_core_score"].to_numpy()
            gen = sub["generated_emt_core_score"].to_numpy()
            ax = fig.add_subplot(gs[i, j])
            if i == 1:
                ax.set_title(stim, fontsize=10.2, fontweight="bold", color="#101828", pad=10)
            ax.axhline(0, color="#D0D5DD", lw=0.65)
            ax.plot(x, obs - obs[0], color="#111827", lw=1.35, marker="o", ms=2.8, zorder=3)
            ax.plot(x, gen - gen[0], color="#667085", lw=1.35, ls=(0, (3, 1.6)), marker="o", ms=2.8, zorder=4)
            ax.add_patch(Rectangle((0, 0.982), 1, 0.018, fc=c, ec="none", transform=ax.transAxes, clip_on=False, zorder=5))
            ax.set_xlim(-0.2, 4.2)
            ax.set_ylim(-0.18, 1.25)
            ax.set_xticks(x)
            ax.set_xticklabels(TIME_LABELS)
            ax.set_yticks([0, 0.5, 1.0])
            ax.grid(axis="y", color="#EAECF0", lw=0.55)
            if j != 1:
                ax.set_yticklabels([])
                ax.spines["left"].set_visible(False)
            else:
                ax.set_ylabel("")
            ax.tick_params(length=2.4, width=0.65, color="#475467")
            ax.spines["bottom"].set_color("#475467")
            ax.spines["left"].set_color("#475467")
            ax.spines["bottom"].set_linewidth(0.65)
            ax.spines["left"].set_linewidth(0.65)
            route_label = row["display_route"].replace(" response", "").replace(" route", "")
            ax.text(0.02, 0.89, wrap_text(route_label, 24), transform=ax.transAxes, ha="left", va="top", fontsize=6.9, fontweight="bold", color=c)
            ax.text(0.02, 0.07, f"n={int(row['observed_endpoint_n'])}; mass x{row['observed_endpoint_baseline_count_ratio']:.2f}", transform=ax.transAxes, ha="left", va="bottom", fontsize=6.4, color="#667085")

    handles = [
        Line2D([0], [0], color="#101828", lw=1.8, marker="o", ms=4, label="observed"),
        Line2D([0], [0], color="#667085", lw=1.8, ls=(0, (3, 1.7)), marker="o", ms=4, label="CellCompass generated"),
    ]
    route_handles = [Line2D([0], [0], marker="s", lw=0, ms=7, markerfacecolor=c, markeredgecolor=c, label=r) for r, c in ROUTE_COLORS.items()]
    fig.legend(handles=handles + route_handles, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.01))
    save_all(fig, "panel02_condition_route_cards")
    return fig


def panel02_condition_route_atlas(route: pd.DataFrame) -> plt.Figure:
    """Compact all-condition route atlas.

    This replaces the 12-card timecourse grid in the main figure. The grid is
    useful as a supplement, but the main text needs a first-pass map of which
    cancer contexts move through EMT and which contexts mainly change endpoint
    mass without strong EMT-core movement.
    """

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    fig.subplots_adjust(left=0.10, right=0.67, top=0.80, bottom=0.16)
    ax.set_facecolor("white")
    ax.axvline(0, color="#CBD5E1", lw=0.9, zorder=0)
    ax.axhline(1, color="#CBD5E1", lw=0.9, zorder=0)
    ax.axvline(0.30, color=mpl.colors.to_rgba("#B8323C", 0.35), lw=0.9, ls=(0, (3, 2)), zorder=0)
    ax.axhline(0.18, color=mpl.colors.to_rgba("#E5922C", 0.40), lw=0.9, ls=(0, (3, 2)), zorder=0)

    ax.text(
        0.315,
        1.33,
        "large EMT-core shift",
        ha="left",
        va="top",
        fontsize=8.2,
        fontweight="bold",
        color="#B8323C",
    )
    ax.text(
        0.62,
        0.205,
        "partial-EMT threshold",
        ha="left",
        va="bottom",
        fontsize=7.6,
        fontweight="bold",
        color="#B36B00",
    )

    for _, row in route.iterrows():
        color = ROUTE_COLORS.get(row["display_route"], "#667085")
        face = color
        edge = "white"
        ax.scatter(
            row["generated_delta_emt_core_score"],
            row["generated_delta_hybrid_score"],
            s=92,
            marker="o",
            facecolor=face,
            edgecolor=edge,
            linewidth=1.15,
            zorder=4,
        )
        ax.text(
            row["generated_delta_emt_core_score"],
            row["generated_delta_hybrid_score"],
            str(CONDITION_IDS[row["condition"]]),
            ha="center",
            va="center",
            fontsize=6.2,
            fontweight="bold",
            color="white",
            zorder=5,
        )

    ax.set_xlim(-0.16, 1.22)
    ax.set_ylim(-0.10, 1.42)
    ax.set_xlabel("generated endpoint EMT-core program shift")
    ax.set_ylabel("generated endpoint partial-EMT / hybrid shift")
    fig.suptitle("Same EMT stimuli produce distinct cancer-cell routes", x=0.105, y=0.965, ha="left", fontsize=13.2, fontweight="bold", color="#101828")
    fig.text(
        0.105,
        0.905,
        "Each numbered point is one cell-line x stimulus model. Colors summarize generated endpoint EMT program shifts.",
        ha="left",
        va="bottom",
        fontsize=7.5,
        color="#667085",
    )
    ax.grid(color="#EAECF0", lw=0.75, zorder=0)
    ax.spines["bottom"].set_color("#475467")
    ax.spines["left"].set_color("#475467")
    ax.tick_params(length=3, color="#475467")

    present_routes = [r for r in ROUTE_COLORS if r in set(route["display_route"])]
    route_handles = [
        Line2D([0], [0], marker="o", lw=0, markersize=7.5, markerfacecolor=c, markeredgecolor=c, label=r)
        for r, c in ROUTE_COLORS.items()
        if r in present_routes
    ]
    ax.legend(
        handles=route_handles,
        loc="center left",
        bbox_to_anchor=(1.03, 0.70),
        frameon=False,
        borderaxespad=0.0,
    )
    fig.text(0.765, 0.38, "Condition ID", ha="left", va="bottom", fontsize=7.4, fontweight="bold", color=INK)
    fig.text(0.765, 0.35, condition_key_text(columns=1), ha="left", va="top", fontsize=6.4, color=TEXT, family="Arial")
    save_all(fig, "panel02_condition_route_atlas")
    return fig


def panel02b_observed_generated_timecourses(route: pd.DataFrame, snap: pd.DataFrame) -> plt.Figure:
    """Clean small-multiple version of the observed/generated EMT-core plot."""

    fig, axes = plt.subplots(3, 4, figsize=(10.8, 6.1))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.84, bottom=0.16, wspace=0.34, hspace=0.50)
    fig.suptitle(
        "EMT-core dynamics by condition",
        x=0.075,
        y=0.955,
        ha="left",
        fontsize=12.4,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.075,
        0.905,
        "Observed cells and CellCompass-generated trajectories are centered to the 0d state within each cell-line x stimulus context.",
        ha="left",
        va="top",
        fontsize=7.6,
        color=MUTED,
    )
    fig.text(
        0.018,
        0.52,
        "Delta EMT-core score from 0d",
        rotation=90,
        ha="center",
        va="center",
        fontsize=8.4,
        color=TEXT,
    )
    x = np.arange(len(TIME_LABELS))
    conditions = [f"{cell}_{stim}" for cell in CELL_LINES for stim in STIMULI]
    for ax, cond in zip(axes.flat, conditions):
        sub = snap.loc[snap["condition"] == cond].sort_values("time_index")
        obs = sub["observed_emt_core_score"].to_numpy()
        gen = sub["generated_emt_core_score"].to_numpy()
        obs_delta = obs - obs[0]
        gen_delta = gen - gen[0]
        low = min(obs_delta.min(), gen_delta.min())
        high = max(obs_delta.max(), gen_delta.max())
        pad = max(0.08, 0.18 * (high - low + 1e-6))

        ax.axhline(0, color="#BFC7D5", lw=0.75, zorder=0)
        ax.fill_between(x, obs_delta, gen_delta, color="#98A2B3", alpha=0.13, linewidth=0, zorder=1)
        ax.plot(x, obs_delta, color=OBSERVED_COLOR, lw=1.65, marker="o", ms=3.4, zorder=3)
        ax.plot(
            x,
            gen_delta,
            color=GENERATED_COLOR,
            lw=1.65,
            ls=(0, (3.0, 1.7)),
            marker="X",
            ms=3.9,
            markeredgewidth=0.35,
            zorder=4,
        )
        ax.set_title(pretty_condition(cond), fontsize=9.4, color=INK, pad=5.5)
        ax.set_xlim(-0.25, 4.25)
        ax.set_ylim(low - pad, high + pad)
        ax.set_xticks(x)
        ax.set_xticklabels(TIME_LABELS, rotation=35, ha="right")
        ax.grid(axis="y", color="#E6EAF0", lw=0.65)
        ax.spines["bottom"].set_color("#98A2B3")
        ax.spines["left"].set_color("#98A2B3")
        ax.spines["bottom"].set_linewidth(0.7)
        ax.spines["left"].set_linewidth(0.7)
        ax.tick_params(length=2.2, width=0.7, color="#98A2B3", labelcolor=TEXT)
    line_handles = [
        Line2D([0], [0], color=OBSERVED_COLOR, lw=1.8, marker="o", ms=4, label="observed cells"),
        Line2D([0], [0], color=GENERATED_COLOR, lw=1.8, ls=(0, (3.2, 1.8)), marker="X", ms=4, label="CellCompass generated"),
        Rectangle((0, 0), 1, 1, facecolor="#98A2B3", alpha=0.15, edgecolor="none", label="model-data gap"),
    ]
    fig.legend(
        handles=line_handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.52, -0.006),
    )
    save_all(fig, "panel02b_observed_generated_emt_core_timecourses")
    return fig


def panel03_endpoint_validation(route: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.35))
    fig.subplots_adjust(left=0.08, right=0.985, top=0.74, bottom=0.26, wspace=0.24)
    pairs = [
        ("observed_delta_emt_core_score", "generated_delta_emt_core_score", "EMT-core endpoint shift"),
        ("observed_delta_mesenchymal_score", "generated_delta_mesenchymal_score", "Mesenchymal endpoint shift"),
    ]
    for ax, (obs_col, gen_col, title) in zip(axes, pairs):
        xs = route[obs_col].to_numpy()
        ys = route[gen_col].to_numpy()
        corr = float(np.corrcoef(xs, ys)[0, 1])
        mn = min(xs.min(), ys.min()) - 0.08
        mx = max(xs.max(), ys.max()) + 0.08
        ax.plot([mn, mx], [mn, mx], color="#98A2B3", lw=1.15, ls=(0, (3, 2)), zorder=0)
        for _, row in route.iterrows():
            c = ROUTE_COLORS.get(row["display_route"], "#667085")
            ax.scatter(
                row[obs_col],
                row[gen_col],
                s=78,
                marker="o",
                facecolor=c,
                edgecolor="white",
                linewidth=1.10,
                alpha=0.95,
                zorder=3,
            )
            ax.text(
                row[obs_col],
                row[gen_col],
                str(CONDITION_IDS[row["condition"]]),
                ha="center",
                va="center",
                fontsize=5.8,
                fontweight="bold",
                color="white",
                zorder=4,
            )
        ax.axvline(0, color="#D0D5DD", lw=0.85)
        ax.axhline(0, color="#D0D5DD", lw=0.85)
        ax.set_xlim(mn, mx)
        ax.set_ylim(mn, mx)
        ax.set_title(f"{title}\nr = {corr:.2f}", fontsize=11, fontweight="bold", color="#101828")
        ax.set_xlabel("observed endpoint delta")
        ax.set_ylabel("generated endpoint delta")
        ax.grid(color="#EAECF0", lw=0.75)
        ax.spines["bottom"].set_color("#475467")
        ax.spines["left"].set_color("#475467")
        ax.tick_params(axis="both", length=3, color="#475467")
    fig.suptitle("Generated endpoints recover observed EMT program shifts", x=0.08, y=0.965, ha="left", fontsize=13.0, fontweight="bold", color="#101828")
    fig.text(
        0.08,
        0.90,
        "Each numbered point is one condition-specific CellCompass model; IDs match the route atlas.",
        ha="left",
        va="top",
        fontsize=7.5,
        color="#667085",
    )
    present_routes = [r for r in ROUTE_COLORS if r in set(route["display_route"])]
    route_handles = [
        Line2D([0], [0], marker="o", lw=0, markersize=7, markerfacecolor=c, markeredgecolor=c, label=r)
        for r, c in ROUTE_COLORS.items()
        if r in present_routes
    ]
    fig.legend(route_handles, [h.get_label() for h in route_handles], loc="lower center", bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False)
    save_all(fig, "panel03_endpoint_validation_scatter")
    return fig


def panel04_mechanism_fingerprint(mech: pd.DataFrame) -> plt.Figure:
    module_index = {module: i for i, module in enumerate(MECH_MODULES)}
    angles = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, len(MECH_MODULES), endpoint=False)
    value_max = max(1.25, float(mech[[f"gen_delta_{m}" for m in MECH_MODULES]].max().max()))
    neg_max = max(0.40, abs(float(mech[[f"gen_delta_{m}" for m in MECH_MODULES]].min().min())))

    fig = plt.figure(figsize=(10.8, 5.9))
    gs = fig.add_gridspec(
        5,
        4,
        height_ratios=[0.54, 1, 1, 1, 1],
        width_ratios=[0.34, 1, 1, 1],
        hspace=0.16,
        wspace=0.25,
    )
    title_ax = fig.add_subplot(gs[0, :])
    title_ax.axis("off")
    title_ax.text(
        0,
        0.88,
        "Mechanism fingerprints across EMT contexts",
        ha="left",
        va="top",
        fontsize=12.6,
        fontweight="bold",
        color=INK,
    )
    title_ax.text(
        0,
        0.50,
        "Each glyph is one CellCompass-generated endpoint. Fixed spokes are decoded gene programs; spoke length is the program increase from baseline.",
        ha="left",
        va="top",
        fontsize=7.5,
        color=MUTED,
    )

    for j, stim in enumerate(STIMULI, start=1):
        ax = fig.add_subplot(gs[0, j])
        ax.axis("off")
        ax.text(0.50, 0.12, stim, ha="center", va="center", fontsize=10.4, fontweight="bold", color=INK, transform=ax.transAxes)

    def draw_glyph(ax: plt.Axes, vals: dict[str, float], mechanism: str) -> None:
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_xlim(-1.10, 1.10)
        ax.set_ylim(-1.10, 1.10)
        ax.add_patch(Circle((0, 0), 0.16, fc="white", ec="#D0D5DD", lw=0.85, zorder=2))
        ax.add_patch(Circle((0, 0), 0.88, fc="none", ec="#EEF2F6", lw=0.85, zorder=0))
        for module in MECH_MODULES:
            val = vals[module]
            angle = angles[module_index[module]]
            ux, uy = np.cos(angle), np.sin(angle)
            ax.plot([0.20 * ux, 0.88 * ux], [0.20 * uy, 0.88 * uy], color="#EEF2F6", lw=0.9, zorder=0)
            if val >= 0:
                r = 0.22 + 0.82 * min(val / value_max, 1.0)
                ax.plot([0.22 * ux, r * ux], [0.22 * uy, r * uy], color=MODULE_COLORS[module], lw=2.7, solid_capstyle="round", alpha=0.95, zorder=3)
                ax.scatter(r * ux, r * uy, s=14 + 22 * min(val / value_max, 1.0), color=MODULE_COLORS[module], edgecolor="white", lw=0.45, zorder=4)
            elif val <= -0.08:
                r = 0.22 + 0.48 * min(abs(val) / neg_max, 1.0)
                ax.plot([-0.22 * ux, -r * ux], [-0.22 * uy, -r * uy], color="#4E6A83", lw=2.0, solid_capstyle="round", alpha=0.78, zorder=3)

    mech_indexed = mech.set_index("condition")
    for i, cell in enumerate(CELL_LINES, start=1):
        la = fig.add_subplot(gs[i, 0])
        la.axis("off")
        la.text(0.98, 0.50, cell, ha="right", va="center", fontsize=11.4, fontweight="bold", color=INK)
        for j, stim in enumerate(STIMULI, start=1):
            cond = f"{cell}_{stim}"
            row = mech_indexed.loc[cond]
            vals = {module: float(row[f"gen_delta_{module}"]) for module in MECH_MODULES}
            ax = fig.add_subplot(gs[i, j])
            draw_glyph(ax, vals, str(row["mechanism_class"]))

    legend_modules = [
        "TGFB_SMAD_targets",
        "EMT_TF_axis",
        "ECM_remodeling",
        "partial_basal_hybrid",
        "TNF_NFKB_inflammatory",
        "EGFR_AP1_MAPK",
        "epithelial_junction",
        "cell_cycle_growth",
    ]
    handles = [
        Line2D([0], [0], color=MODULE_COLORS[m], lw=2.4, marker="o", ms=4.2, label=MODULE_LABELS[m])
        for m in legend_modules
    ]
    handles.append(Line2D([0], [0], color="#4E6A83", lw=2.0, label="decreased module"))
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.55, 0.00), columnspacing=1.0, handlelength=1.6)
    save_all(fig, "panel04_mechanism_fingerprint_cards")
    return fig


def panel05_three_layer_mechanism(var: pd.DataFrame) -> plt.Figure:
    gen = var.loc[var["source"] == "gen"].set_index("module")
    groups = [
        ("Stimulus response", ["TNF_NFKB_inflammatory", "TGFB_SMAD_targets", "EGFR_AP1_MAPK", "ECM_remodeling"]),
        ("Cancer-cell state", ["epithelial_junction", "partial_basal_hybrid", "interferon_JAK_STAT"]),
        ("Growth / stress", ["cell_cycle_growth", "stress_p53_apoptosis"]),
    ]
    factors = [
        ("cell-line identity", "cell_line_fraction", FACTOR_COLORS["cell-line identity"]),
        ("stimulus", "stimulus_fraction", FACTOR_COLORS["stimulus"]),
        ("cell-line x stimulus context", "context_residual_fraction", FACTOR_COLORS["cell-line x stimulus context"]),
    ]

    rows: list[tuple[str, str]] = []
    for group_name, modules in groups:
        for module in modules:
            if module in gen.index:
                rows.append((group_name, module))

    fig, ax = plt.subplots(figsize=(9.6, 5.5))
    fig.subplots_adjust(left=0.34, right=0.98, top=0.78, bottom=0.20)
    y_positions = np.arange(len(rows))[::-1]
    bar_height = 0.66

    for idx, (group_name, module) in enumerate(rows):
        y = y_positions[idx]
        row = gen.loc[module]
        left = 0.0
        values = [float(row[col]) for _, col, _ in factors]
        for label, col, color in factors:
            frac = float(row[col])
            ax.barh(y, frac, left=left, height=bar_height, color=color, edgecolor="white", linewidth=0.8)
            if frac >= 0.36:
                ax.text(left + frac / 2, y, f"{frac:.2f}", ha="center", va="center", fontsize=6.8, fontweight="bold", color="white")
            left += frac

    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.8, len(rows) - 0.2)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([MODULE_LABELS[module] for _, module in rows], fontsize=8.2)
    ax.set_xlabel("fraction of generated endpoint module variance")
    ax.set_xticks([0, 0.25, 0.50, 0.75, 1.0])
    ax.set_xticklabels(["0", "0.25", "0.50", "0.75", "1.0"])
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=3, color=AXIS)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)

    # Group labels and separators.
    start = 0
    for group_name, modules in groups:
        present = [m for m in modules if m in gen.index]
        if not present:
            continue
        end = start + len(present) - 1
        y_top = y_positions[start] + bar_height / 2 + 0.12
        y_bot = y_positions[end] - bar_height / 2 - 0.12
        ax.plot([-0.055, -0.055], [y_bot, y_top], color="#D0D5DD", lw=1.3, clip_on=False)
        ax.text(-0.145, (y_top + y_bot) / 2, group_name, ha="right", va="center", fontsize=7.8, fontweight="bold", color=TEXT, transform=ax.get_yaxis_transform())
        if end < len(rows) - 1:
            ax.axhline(y_positions[end] - 0.55, color="#EAECF0", lw=1.0)
        start = end + 1

    fig.suptitle("EMT programs are shaped by stimulus and cancer-cell context", x=0.34, y=0.96, ha="left", fontsize=12.6, fontweight="bold", color=INK)
    fig.text(
        0.34,
        0.89,
        "Generated endpoint variation is decomposed into cell-line, stimulus and cell-line x stimulus components.",
        ha="left",
        va="top",
        fontsize=7.8,
        color=MUTED,
    )

    factor_handles = [Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="none", label=label) for label, _, color in factors]
    ax.legend(handles=factor_handles, loc="lower center", bbox_to_anchor=(0.52, -0.27), ncol=3, frameon=False)

    save_all(fig, "panel05_three_layer_mechanism")
    return fig


def panel06_representative_mechanism_timing(traj: pd.DataFrame, mech: pd.DataFrame) -> plt.Figure:
    reps = ["A549_TGFB1", "DU145_TNF", "OVCA420_TNF", "MCF7_TGFB1"]
    mech_indexed = mech.set_index("condition")
    candidate_modules = [
        module
        for module in MODULE_LABELS
        if f"gen_delta_{module}" in mech.columns and f"gen_{module}" in traj.columns
    ]

    def top_endpoint_modules(condition: str, n: int = 4) -> list[str]:
        row = mech_indexed.loc[condition]
        ranked = sorted(
            candidate_modules,
            key=lambda module: float(row[f"gen_delta_{module}"]),
            reverse=True,
        )
        return ranked[:n]

    def diagnostic_modules(condition: str) -> list[str]:
        """Show the evidence used by the mechanism label, not only top hits."""
        mech_class = str(mech_indexed.loc[condition, "mechanism_class"])
        modules = ROUTE_DIAGNOSTIC_MODULES.get(mech_class)
        if modules is None:
            # Mixed routes have no single defining rule; show their largest endpoint shifts.
            return top_endpoint_modules(condition, n=4)
        return [module for module in modules if module in candidate_modules]

    line_colors = {f"gen_{module}": MODULE_COLORS.get(module, MUTED) for module in candidate_modules}
    line_colors["mass_sum"] = "#98A2B3"
    fig, axes = plt.subplots(1, 4, figsize=(12.1, 4.5), sharex=True)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.70, bottom=0.17, wspace=0.30)
    for ax, cond in zip(axes, reps):
        modules = diagnostic_modules(cond)
        cols = [f"gen_{module}" for module in modules] + ["mass_sum"]
        sub = traj.loc[traj["condition"] == cond].sort_values("model_time")
        x = sub["model_time"].to_numpy()
        for col in cols:
            y = sub[col].to_numpy()
            if col != "mass_sum":
                y = y - y[0]
            label = "effective mass" if col == "mass_sum" else MODULE_LABELS[col.replace("gen_", "")]
            linestyle = (0, (3, 1.8)) if col == "mass_sum" else "-"
            ax.plot(x, y, color=line_colors[col], lw=1.8, ls=linestyle, label=label)
        ax.axhline(0, color="#D0D5DD", lw=0.8)
        mech_label = MECH_LABELS.get(str(mech_indexed.loc[cond, "mechanism_class"]), str(mech_indexed.loc[cond, "mechanism_class"]))
        ax.set_title(f"{pretty_condition(cond)}\n{mech_label}", fontsize=8.8, fontweight="bold", color=INK)
        ax.set_xticks([0, 1, 2, 3, 4])
        ax.set_xticklabels(TIME_LABELS)
        ax.grid(color=GRID, lw=0.7)
        ax.spines["bottom"].set_color(AXIS)
        ax.spines["left"].set_color(AXIS)
        ax.tick_params(length=3, color=AXIS)
        ax.legend(loc="upper left", frameon=False, fontsize=6.5)
    axes[0].set_ylabel("module delta from t0\nor effective mass")
    fig.suptitle("Representative routes reveal different program timing", x=0.02, y=0.985, ha="left", fontsize=13.0, fontweight="bold", color=INK)
    fig.text(
        0.02,
        0.905,
        "For clean route classes, plotted gene programs are the rule-defining modules; mixed routes show the largest endpoint increases. Gray dashed line shows effective mass.",
        ha="left",
        va="top",
        fontsize=7.5,
        color=MUTED,
    )
    save_all(fig, "panel06_representative_mechanism_timing")
    return fig


def main() -> None:
    setup_style()
    route, snap, mech, traj, var = load_tables()
    panel01_dialogue_loop()
    panel02_condition_route_atlas(route)
    panel02b_observed_generated_timecourses(route, snap)
    panel03_endpoint_validation(route)
    panel04_mechanism_fingerprint(mech)
    panel05_three_layer_mechanism(var)
    panel06_representative_mechanism_timing(traj, mech)
    plt.close("all")
    print(f"Wrote EMT CellCompass main panels to {OUT_DIR}")


if __name__ == "__main__":
    main()
