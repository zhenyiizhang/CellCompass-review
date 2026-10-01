#!/usr/bin/env python3
"""Redraw condition-specific EMT CellCompass results for a Cell/NBT-style story.

The script intentionally uses only the latest CellCompass downstream tables. It
does not modify the original report figures.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[3]
TABLE_DIR = ROOT / "emt" / "downstream" / "emt_condition_dynamics"
OUT_DIR = ROOT / "emt" / "figures" / "emt_condition_dynamics_redrawn"

CELL_LINES = ["A549", "DU145", "MCF7", "OVCA420"]
STIMULI = ["EGF", "TGFB1", "TNF"]
TIME_LABELS = ["0d", "8h", "1d", "3d", "7d"]


ROUTE_COLORS = {
    "EMT conversion": "#C43C39",
    "partial/hybrid EMT": "#E9962E",
    "growth-retaining": "#557DAF",
    "epithelial-retaining": "#6F7B8C",
    "mixed/weak EMT": "#8C6BB1",
}

STIMULUS_MARKERS = {
    "EGF": "o",
    "TGFB1": "^",
    "TNF": "s",
}

PROGRAM_LABELS = {
    "epithelial_score": "Epithelial",
    "hybrid_score": "Hybrid",
    "mesenchymal_score": "Mesenchymal",
    "emt_core_score": "EMT core",
    "early_immediate_response_score": "Immediate response",
    "early_inflammatory_tnf_score": "Inflammatory/TNF",
    "intermediate_partial_emt_score": "Intermediate partial EMT",
    "late_ecm_mesenchymal_score": "Late ECM/mesenchymal",
    "growth_cell_cycle_score": "Growth/cell cycle",
}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial"],
            "font.size": 8,
            "axes.titlesize": 10,
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
            "axes.linewidth": 0.8,
        }
    )


def display_route(row: pd.Series) -> str:
    """Biology-facing route label derived from generated endpoint changes.

    The original CellCompass route_class is preserved in the source table. For
    figure readability, this label is computed from the generated EMT-core and
    hybrid shifts shown on the route plots. This avoids hidden criteria that
    make threshold labels disagree with point colors.
    """

    core = row["generated_delta_emt_core_score"]
    hybrid = row["generated_delta_hybrid_score"]

    if core >= 0.30:
        return "EMT conversion"
    if hybrid >= 0.18:
        return "partial/hybrid EMT"
    if abs(core) < 0.08 and hybrid < 0.12:
        return "epithelial-retaining"
    return "mixed/weak EMT"


def add_route_labels(summary: pd.DataFrame) -> pd.DataFrame:
    summary = summary.copy()
    summary["display_route"] = summary.apply(display_route, axis=1)
    summary["low_endpoint_support"] = summary["observed_endpoint_n"] < 150
    return summary


def save_all(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT_DIR / f"{stem}.{ext}", bbox_inches="tight", facecolor="white")


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = add_route_labels(pd.read_csv(TABLE_DIR / "condition_route_summary.csv"))
    snap = pd.read_csv(TABLE_DIR / "endpoint_and_measured_snapshot_comparison.csv")
    traj = pd.read_csv(TABLE_DIR / "generated_program_trajectories.csv")
    shared = pd.read_csv(TABLE_DIR / "shared_vs_context_specific_programs.csv")
    return summary, snap, traj, shared


def clean_axis(ax: plt.Axes) -> None:
    ax.spines["left"].set_color("#2E3440")
    ax.spines["bottom"].set_color("#2E3440")
    ax.tick_params(colors="#2E3440", length=3, width=0.8)


def route_card_grid(summary: pd.DataFrame, snap: pd.DataFrame) -> plt.Figure:
    fig = plt.figure(figsize=(12.2, 8.1))
    gs = GridSpec(
        5,
        4,
        figure=fig,
        height_ratios=[0.42, 1, 1, 1, 1],
        width_ratios=[0.32, 1, 1, 1],
        hspace=0.38,
        wspace=0.18,
    )
    title_ax = fig.add_subplot(gs[0, :])
    title_ax.axis("off")
    title_ax.text(
        0,
        0.72,
        "Condition-specific virtual EMT routes",
        fontsize=18,
        fontweight="bold",
        color="#101828",
        ha="left",
    )
    title_ax.text(
        0,
        0.13,
        "Separate VGFM dynamics were trained for each cancer cell line x stimulus; cards show observed and generated EMT-core movement from baseline.",
        fontsize=9,
        color="#475467",
        ha="left",
    )

    for j, stim in enumerate(STIMULI, start=1):
        ax = fig.add_subplot(gs[0, j])
        ax.axis("off")
        ax.text(0.5, -0.12, stim, ha="center", va="center", fontsize=12, fontweight="bold", color="#1F2937")

    y_lim = (-0.16, 1.18)
    x = np.arange(len(TIME_LABELS))

    for i, cell in enumerate(CELL_LINES, start=1):
        label_ax = fig.add_subplot(gs[i, 0])
        label_ax.axis("off")
        label_ax.text(0.98, 0.52, cell, ha="right", va="center", fontsize=12, fontweight="bold", color="#1F2937")

        for j, stim in enumerate(STIMULI, start=1):
            cond = f"{cell}_{stim}"
            row = summary.loc[summary["condition"] == cond].iloc[0]
            color = ROUTE_COLORS[row["display_route"]]
            ax = fig.add_subplot(gs[i, j])
            sub = snap.loc[snap["condition"] == cond].sort_values("time_index")

            obs = sub["observed_emt_core_score"].to_numpy()
            gen = sub["generated_emt_core_score"].to_numpy()
            obs_delta = obs - obs[0]
            gen_delta = gen - gen[0]

            # Soft route-colored card background.
            ax.add_patch(
                FancyBboxPatch(
                    (0, 0),
                    1,
                    1,
                    boxstyle="round,pad=0.018,rounding_size=0.018",
                    linewidth=0.8,
                    edgecolor=color,
                    facecolor=mpl.colors.to_rgba(color, 0.075),
                    transform=ax.transAxes,
                    clip_on=False,
                    zorder=-10,
                )
            )
            ax.axhline(0, color="#CBD5E1", lw=0.8, zorder=0)
            ax.plot(x, obs_delta, color="#111827", lw=1.6, marker="o", ms=3.3, label="observed")
            ax.plot(x, gen_delta, color=color, lw=1.8, ls=(0, (3, 1.8)), marker="o", ms=3.2, label="generated")
            ax.set_ylim(*y_lim)
            ax.set_xticks(x)
            ax.set_xticklabels(TIME_LABELS, rotation=0)
            ax.set_yticks([0, 0.5, 1.0])
            ax.grid(axis="y", color="#E5E7EB", lw=0.7)
            clean_axis(ax)
            if j != 1:
                ax.set_yticklabels([])
                ax.spines["left"].set_visible(False)
            else:
                ax.set_ylabel("delta EMT core", color="#344054")

            route_text = row["display_route"].replace("epithelial-", "epi-\n")
            ax.text(0.04, 0.92, route_text, transform=ax.transAxes, ha="left", va="top", fontsize=7.4, color=color, fontweight="bold")
            mass_text = f"mass x{row['observed_endpoint_baseline_count_ratio']:.2f}"
            ax.text(0.04, 0.08, mass_text, transform=ax.transAxes, ha="left", va="bottom", fontsize=7, color="#475467")
            if row["low_endpoint_support"]:
                ax.text(
                    0.97,
                    0.08,
                    f"low n={int(row['observed_endpoint_n'])}",
                    transform=ax.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=7,
                    color="#B42318",
                )

    handles = [
        Line2D([0], [0], color="#111827", lw=1.8, marker="o", ms=4, label="observed cells"),
        Line2D([0], [0], color="#475467", lw=1.8, ls=(0, (3, 1.8)), marker="o", ms=4, label="CellCompass rollout, dashed"),
    ]
    route_handles = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=v, markeredgecolor=v, ms=8, label=k)
        for k, v in ROUTE_COLORS.items()
    ]
    fig.legend(handles + route_handles, [h.get_label() for h in handles + route_handles], loc="lower center", ncol=7, frameon=False, bbox_to_anchor=(0.5, -0.02))
    save_all(fig, "panel01_condition_specific_emt_route_cards")
    return fig


def endpoint_validation(summary: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.25), constrained_layout=True)
    pairs = [
        ("observed_delta_emt_core_score", "generated_delta_emt_core_score", "EMT-core endpoint shift"),
        ("observed_delta_mesenchymal_score", "generated_delta_mesenchymal_score", "Mesenchymal endpoint shift"),
    ]
    for ax, (xcol, ycol, title) in zip(axes, pairs):
        xs = summary[xcol].to_numpy()
        ys = summary[ycol].to_numpy()
        corr = float(np.corrcoef(xs, ys)[0, 1])
        mn = min(xs.min(), ys.min()) - 0.08
        mx = max(xs.max(), ys.max()) + 0.08
        ax.plot([mn, mx], [mn, mx], color="#94A3B8", lw=1.2, ls=(0, (3, 2)), zorder=0)
        for _, row in summary.iterrows():
            c = ROUTE_COLORS[row["display_route"]]
            marker = STIMULUS_MARKERS[row["stimulus"]]
            size = 42 + 0.055 * min(row["observed_endpoint_n"], 1800)
            face = c if not row["low_endpoint_support"] else "white"
            ax.scatter(
                row[xcol],
                row[ycol],
                s=size,
                marker=marker,
                facecolor=face,
                edgecolor=c,
                linewidth=1.6,
                alpha=0.94,
                zorder=3,
            )
        for cond in ["DU145_TGFB1", "A549_TGFB1", "A549_TNF", "DU145_TNF", "OVCA420_TNF"]:
            row = summary.loc[summary["condition"] == cond].iloc[0]
            ax.text(row[xcol] + 0.018, row[ycol] + 0.018, cond.replace("_", "+"), fontsize=6.8, color="#334155")
        ax.set_title(f"{title}\nr = {corr:.2f}", fontweight="bold", color="#101828")
        ax.set_xlabel("observed endpoint delta")
        ax.set_ylabel("generated endpoint delta")
        ax.set_xlim(mn, mx)
        ax.set_ylim(mn, mx)
        ax.grid(color="#E5E7EB", lw=0.75)
        clean_axis(ax)

    route_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=v, markeredgecolor=v, ms=7, label=k)
        for k, v in ROUTE_COLORS.items()
    ]
    marker_handles = [
        Line2D([0], [0], marker=m, color="#475467", markerfacecolor="white", markeredgewidth=1.4, lw=0, label=s)
        for s, m in STIMULUS_MARKERS.items()
    ]
    fig.legend(route_handles + marker_handles, [h.get_label() for h in route_handles + marker_handles], loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=4, frameon=False)
    fig.suptitle("Generated endpoint programs are compared within each condition", x=0.02, y=1.04, ha="left", fontsize=14, fontweight="bold", color="#101828")
    save_all(fig, "panel02_generated_vs_observed_endpoint_validation")
    return fig


def representative_dynamics(summary: pd.DataFrame, snap: pd.DataFrame) -> plt.Figure:
    reps = ["DU145_TGFB1", "DU145_TNF", "A549_TGFB1", "OVCA420_TNF"]
    labels = {
        "DU145_TGFB1": "TGFB1 route\nDU145 + TGFB1",
        "DU145_TNF": "retaining route\nDU145 + TNF",
        "A549_TGFB1": "strong EMT trend\nA549 + TGFB1",
        "OVCA420_TNF": "growth-biased route\nOVCA420 + TNF",
    }
    fig = plt.figure(figsize=(11.2, 5.8))
    outer = GridSpec(2, 4, figure=fig, height_ratios=[1.18, 0.88], hspace=0.42, wspace=0.35)
    x = np.arange(len(TIME_LABELS))
    for idx, cond in enumerate(reps):
        row = summary.loc[summary["condition"] == cond].iloc[0]
        color = ROUTE_COLORS[row["display_route"]]
        sub = snap.loc[snap["condition"] == cond].sort_values("time_index")

        ax = fig.add_subplot(outer[0, idx])
        obs = sub["observed_emt_core_score"].to_numpy()
        gen = sub["generated_emt_core_score"].to_numpy()
        obs_d = obs - obs[0]
        gen_d = gen - gen[0]
        ax.axhline(0, color="#CBD5E1", lw=0.8)
        ax.plot(x, obs_d, color="#111827", lw=2.0, marker="o", ms=4, label="observed")
        ax.plot(x, gen_d, color=color, lw=2.0, ls=(0, (3, 1.8)), marker="o", ms=4, label="generated")
        ax.set_title(labels[cond], color=color, fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(TIME_LABELS)
        ax.set_ylim(-0.15, 1.20)
        ax.set_yticks([0, 0.5, 1.0])
        ax.grid(axis="y", color="#E5E7EB", lw=0.75)
        if idx == 0:
            ax.set_ylabel("delta EMT core")
        clean_axis(ax)

        ax2 = fig.add_subplot(outer[1, idx])
        obs_mass = sub["observed_n_cells"].to_numpy() / sub["observed_n_cells"].iloc[0]
        gen_mass = sub["generated_mass_sum"].to_numpy()
        ax2.plot(x, obs_mass, color="#111827", lw=1.8, marker="o", ms=3.8)
        ax2.plot(x, gen_mass, color=color, lw=1.8, ls=(0, (3, 1.8)), marker="o", ms=3.8)
        ax2.axhline(1, color="#CBD5E1", lw=0.8)
        ax2.set_xticks(x)
        ax2.set_xticklabels(TIME_LABELS)
        ax2.set_ylim(0, max(3.0, obs_mass.max() * 1.1, gen_mass.max() * 1.1))
        ax2.grid(axis="y", color="#E5E7EB", lw=0.75)
        if idx == 0:
            ax2.set_ylabel("relative cell mass")
        clean_axis(ax2)

        if row["low_endpoint_support"]:
            ax.text(0.98, 0.06, f"endpoint n={int(row['observed_endpoint_n'])}", transform=ax.transAxes, ha="right", va="bottom", color="#B42318", fontsize=7)

    handles = [
        Line2D([0], [0], color="#111827", lw=2, marker="o", ms=4, label="observed cells"),
        Line2D([0], [0], color="#475467", lw=2, ls=(0, (3, 1.8)), marker="o", ms=4, label="CellCompass rollout, dashed"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Representative condition-specific routes separate EMT movement from mass expansion", x=0.02, y=1.02, ha="left", fontsize=14, fontweight="bold", color="#101828")
    save_all(fig, "panel03_representative_condition_dynamics")
    return fig


def route_phase_diagram(summary: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    ax.axvline(0, color="#CBD5E1", lw=0.9)
    ax.axhline(1, color="#CBD5E1", lw=0.9)
    ax.axvspan(0.30, 1.25, color="#FEE2E2", alpha=0.28, zorder=-3)
    ax.axhspan(1.5, 2.8, color="#DBEAFE", alpha=0.25, zorder=-4)
    ax.text(0.34, 2.68, "EMT conversion\nzone", color="#991B1B", fontsize=8, fontweight="bold")
    ax.text(-0.13, 2.68, "growth-biased\nzone", color="#1D4ED8", fontsize=8, fontweight="bold")
    for _, row in summary.iterrows():
        c = ROUTE_COLORS[row["display_route"]]
        marker = STIMULUS_MARKERS[row["stimulus"]]
        face = c if not row["low_endpoint_support"] else "white"
        ax.scatter(
            row["generated_delta_emt_core_score"],
            row["observed_endpoint_baseline_count_ratio"],
            s=95,
            marker=marker,
            facecolor=face,
            edgecolor=c,
            linewidth=1.6,
            zorder=4,
        )
        label = row["condition"].replace("_", "+")
        if row["condition"] in ["DU145_TGFB1", "DU145_TNF", "A549_TGFB1", "A549_TNF", "OVCA420_TNF", "MCF7_TGFB1"]:
            ax.text(row["generated_delta_emt_core_score"] + 0.02, row["observed_endpoint_baseline_count_ratio"] + 0.03, label, fontsize=7, color="#334155")
    ax.set_xlabel("generated endpoint delta EMT core")
    ax.set_ylabel("observed 7d / 0d cell count ratio")
    ax.set_xlim(-0.14, 1.18)
    ax.set_ylim(0, 2.85)
    ax.set_title("Cancer-cell context gates EMT movement and population expansion", fontsize=13, fontweight="bold", color="#101828", loc="left")
    ax.grid(color="#E5E7EB", lw=0.75)
    clean_axis(ax)

    route_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=v, markeredgecolor=v, ms=7, label=k)
        for k, v in ROUTE_COLORS.items()
    ]
    stim_handles = [
        Line2D([0], [0], marker=m, color="#475467", markerfacecolor="white", lw=0, markeredgewidth=1.4, ms=7, label=s)
        for s, m in STIMULUS_MARKERS.items()
    ]
    low_handle = Line2D([0], [0], marker="o", color="#B42318", markerfacecolor="white", lw=0, markeredgewidth=1.4, ms=7, label="endpoint n < 150")
    ax.legend(handles=route_handles + stim_handles + [low_handle], loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    save_all(fig, "panel04_route_phase_diagram")
    return fig


def program_layer_summary(shared: pd.DataFrame) -> plt.Figure:
    plot_order = [
        "intermediate_partial_emt_score",
        "hybrid_score",
        "early_inflammatory_tnf_score",
        "late_ecm_mesenchymal_score",
        "emt_core_score",
        "mesenchymal_score",
        "growth_cell_cycle_score",
        "epithelial_score",
    ]
    df = shared.set_index("program").loc[plot_order].reset_index()
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    y = np.arange(len(df))[::-1]
    colors = ["#0E9384" if d == "shared_increase" else "#64748B" for d in df["shared_direction"]]
    ax.barh(y, df["positive_fraction"] * 100, color=colors, height=0.56, alpha=0.92)
    for yi, (_, row) in zip(y, df.iterrows()):
        ax.text(row["positive_fraction"] * 100 + 1.5, yi, f"{row['positive_fraction']*100:.0f}% contexts", va="center", fontsize=7.5, color="#334155")
        ax.text(3, yi, f"strongest: {row['strongest_increase_condition'].replace('_','+')}", va="center", fontsize=7.1, color="white" if row["positive_fraction"] > 0.55 else "#334155")
    ax.set_yticks(y)
    ax.set_yticklabels([PROGRAM_LABELS[p] for p in df["program"]])
    ax.set_xlim(0, 105)
    ax.set_xlabel("conditions with increased observed program score at endpoint (%)")
    ax.set_title("Shared partial-EMT programs coexist with context-specific mesenchymal conversion", fontsize=12.5, fontweight="bold", loc="left", color="#101828")
    ax.grid(axis="x", color="#E5E7EB", lw=0.75)
    clean_axis(ax)
    handles = [
        Rectangle((0, 0), 1, 1, facecolor="#0E9384", label="shared increase"),
        Rectangle((0, 0), 1, 1, facecolor="#64748B", label="context-specific"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False)
    save_all(fig, "panel05_program_layer_summary")
    return fig


def overview(summary: pd.DataFrame, snap: pd.DataFrame, shared: pd.DataFrame) -> plt.Figure:
    fig = plt.figure(figsize=(13.2, 10.2))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.15, 1.0], width_ratios=[1.0, 1.05], hspace=0.36, wspace=0.28)

    # A: compact route phase diagram
    ax = fig.add_subplot(gs[0, 0])
    ax.axvline(0, color="#CBD5E1", lw=0.9)
    ax.axhline(1, color="#CBD5E1", lw=0.9)
    ax.axvspan(0.30, 1.25, color="#FEE2E2", alpha=0.28, zorder=-3)
    ax.axhspan(1.5, 2.8, color="#DBEAFE", alpha=0.25, zorder=-4)
    for _, row in summary.iterrows():
        c = ROUTE_COLORS[row["display_route"]]
        marker = STIMULUS_MARKERS[row["stimulus"]]
        face = c if not row["low_endpoint_support"] else "white"
        ax.scatter(row["generated_delta_emt_core_score"], row["observed_endpoint_baseline_count_ratio"], s=90, marker=marker, facecolor=face, edgecolor=c, linewidth=1.6, zorder=4)
    for cond in ["DU145_TGFB1", "A549_TGFB1", "OVCA420_TNF", "DU145_TNF", "A549_TNF"]:
        row = summary.loc[summary["condition"] == cond].iloc[0]
        ax.text(row["generated_delta_emt_core_score"] + 0.02, row["observed_endpoint_baseline_count_ratio"] + 0.035, cond.replace("_", "+"), fontsize=7, color="#334155")
    ax.set_xlim(-0.14, 1.18)
    ax.set_ylim(0, 2.85)
    ax.set_xlabel("generated endpoint delta EMT core")
    ax.set_ylabel("observed 7d / 0d cell count ratio")
    ax.set_title("a  Context gates EMT movement and mass expansion", loc="left", fontsize=12.5, fontweight="bold")
    ax.grid(color="#E5E7EB", lw=0.75)
    clean_axis(ax)

    # B: endpoint validation
    ax = fig.add_subplot(gs[0, 1])
    xs = summary["observed_delta_emt_core_score"].to_numpy()
    ys = summary["generated_delta_emt_core_score"].to_numpy()
    mn = min(xs.min(), ys.min()) - 0.08
    mx = max(xs.max(), ys.max()) + 0.08
    corr = float(np.corrcoef(xs, ys)[0, 1])
    ax.plot([mn, mx], [mn, mx], color="#94A3B8", lw=1.2, ls=(0, (3, 2)))
    for _, row in summary.iterrows():
        c = ROUTE_COLORS[row["display_route"]]
        face = c if not row["low_endpoint_support"] else "white"
        ax.scatter(row["observed_delta_emt_core_score"], row["generated_delta_emt_core_score"], s=80, marker=STIMULUS_MARKERS[row["stimulus"]], facecolor=face, edgecolor=c, linewidth=1.5)
    ax.text(0.04, 0.92, f"trend agreement r={corr:.2f}", transform=ax.transAxes, fontsize=9, fontweight="bold", color="#334155")
    ax.set_xlim(mn, mx)
    ax.set_ylim(mn, mx)
    ax.set_xlabel("observed endpoint delta EMT core")
    ax.set_ylabel("generated endpoint delta EMT core")
    ax.set_title("b  Generated endpoints reproduce condition trends", loc="left", fontsize=12.5, fontweight="bold")
    ax.grid(color="#E5E7EB", lw=0.75)
    clean_axis(ax)

    # C: representative dynamics
    subgs = GridSpecFromSubplotSpec(2, 3, subplot_spec=gs[1, 0], hspace=0.28, wspace=0.25)
    reps = ["DU145_TGFB1", "DU145_TNF", "OVCA420_TNF"]
    x = np.arange(len(TIME_LABELS))
    for col, cond in enumerate(reps):
        row = summary.loc[summary["condition"] == cond].iloc[0]
        color = ROUTE_COLORS[row["display_route"]]
        sub = snap.loc[snap["condition"] == cond].sort_values("time_index")
        obs = sub["observed_emt_core_score"].to_numpy()
        gen = sub["generated_emt_core_score"].to_numpy()
        ax1 = fig.add_subplot(subgs[0, col])
        ax1.axhline(0, color="#CBD5E1", lw=0.8)
        ax1.plot(x, obs - obs[0], color="#111827", marker="o", ms=3.3, lw=1.6)
        ax1.plot(x, gen - gen[0], color=color, marker="o", ms=3.3, lw=1.6, ls=(0, (3, 1.8)))
        ax1.set_title(cond.replace("_", "+"), fontsize=9, color=color, fontweight="bold")
        ax1.set_ylim(-0.1, 0.75)
        ax1.set_xticks([])
        if col == 0:
            ax1.set_ylabel("delta EMT core")
        else:
            ax1.set_yticklabels([])
            ax1.spines["left"].set_visible(False)
        ax1.grid(axis="y", color="#E5E7EB", lw=0.7)
        clean_axis(ax1)

        ax2 = fig.add_subplot(subgs[1, col])
        obs_mass = sub["observed_n_cells"].to_numpy() / sub["observed_n_cells"].iloc[0]
        gen_mass = sub["generated_mass_sum"].to_numpy()
        ax2.axhline(1, color="#CBD5E1", lw=0.8)
        ax2.plot(x, obs_mass, color="#111827", marker="o", ms=3.3, lw=1.6)
        ax2.plot(x, gen_mass, color=color, marker="o", ms=3.3, lw=1.6, ls=(0, (3, 1.8)))
        ax2.set_xticks(x)
        ax2.set_xticklabels(TIME_LABELS, fontsize=6)
        ax2.set_ylim(0, 2.85)
        if col == 0:
            ax2.set_ylabel("relative mass")
        else:
            ax2.set_yticklabels([])
            ax2.spines["left"].set_visible(False)
        ax2.grid(axis="y", color="#E5E7EB", lw=0.7)
        clean_axis(ax2)
    fig.text(0.062, 0.45, "c  Representative virtual-cell routes", fontsize=12.5, fontweight="bold", color="#101828")

    # D: program layer summary
    ax = fig.add_subplot(gs[1, 1])
    plot_order = [
        "intermediate_partial_emt_score",
        "hybrid_score",
        "early_inflammatory_tnf_score",
        "late_ecm_mesenchymal_score",
        "emt_core_score",
        "mesenchymal_score",
        "growth_cell_cycle_score",
        "epithelial_score",
    ]
    df = shared.set_index("program").loc[plot_order].reset_index()
    y = np.arange(len(df))[::-1]
    colors = ["#0E9384" if d == "shared_increase" else "#64748B" for d in df["shared_direction"]]
    ax.barh(y, df["positive_fraction"] * 100, color=colors, height=0.58)
    for yi, (_, r) in zip(y, df.iterrows()):
        ax.text(r["positive_fraction"] * 100 + 1.4, yi, f"{r['positive_fraction']*100:.0f}%", va="center", fontsize=7, color="#334155")
    ax.set_yticks(y)
    ax.set_yticklabels([PROGRAM_LABELS[p] for p in df["program"]], fontsize=7)
    ax.set_xlim(0, 105)
    ax.set_xlabel("conditions increased at endpoint (%)")
    ax.set_title("d  Shared partial-EMT programs, context-specific conversion", loc="left", fontsize=12.5, fontweight="bold")
    ax.grid(axis="x", color="#E5E7EB", lw=0.75)
    clean_axis(ax)
    ax.legend(
        handles=[
            Rectangle((0, 0), 1, 1, facecolor="#0E9384", label="shared increase"),
            Rectangle((0, 0), 1, 1, facecolor="#64748B", label="context-specific"),
        ],
        loc="lower right",
        frameon=False,
    )

    fig.suptitle("CellCompass resolves context-specific EMT routes from condition-specific virtual-cell dynamics", x=0.02, y=0.99, ha="left", fontsize=16.5, fontweight="bold", color="#101828")
    fig.text(
        0.02,
        0.956,
        "Observed cells and CellCompass rollouts are compared within each cell line x stimulus condition; open markers indicate endpoint n < 150.",
        fontsize=8.5,
        color="#475467",
        ha="left",
    )

    route_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=v, markeredgecolor=v, ms=7, label=k)
        for k, v in ROUTE_COLORS.items()
    ]
    stim_handles = [
        Line2D([0], [0], marker=m, color="#475467", markerfacecolor="white", lw=0, markeredgewidth=1.4, ms=7, label=s)
        for s, m in STIMULUS_MARKERS.items()
    ]
    obs_gen_handles = [
        Line2D([0], [0], color="#111827", lw=1.8, marker="o", ms=4, label="observed"),
        Line2D([0], [0], color="#475467", lw=1.8, ls=(0, (3, 1.8)), marker="o", ms=4, label="generated, dashed"),
    ]
    fig.legend(handles=route_handles + stim_handles + obs_gen_handles, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.005))
    save_all(fig, "emt_cell_nbt_story_redraw_overview")
    return fig


def main() -> None:
    setup_style()
    summary, snap, traj, shared = load_tables()
    route_card_grid(summary, snap)
    endpoint_validation(summary)
    representative_dynamics(summary, snap)
    route_phase_diagram(summary)
    program_layer_summary(shared)
    overview(summary, snap, shared)
    plt.close("all")
    print(f"Wrote figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
