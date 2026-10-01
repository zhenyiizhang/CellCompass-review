#!/usr/bin/env python3
"""Nature-style main figure for the PulseBridge HCT116 case study."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cellcompass_nature_main")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/cellcompass_nature_main_numba")
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm


REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "analysis" / "pulsebridge_story" / "outputs"
FIG = OUT / "figures"
TAB = OUT / "tables"
UMAP_H5AD = OUT / "downstream" / "dynamo_stream" / "pulsebridge_full_umap_velocity_for_dynamo.h5ad"

TEAL = "#166C8A"
GREEN = "#31572C"
BLUE = "#4F86C6"
ORANGE = "#F28E2B"
RED = "#B23A48"
PURPLE = "#7A4EAB"
BROWN = "#756A60"
GRAY = "#7D8790"
DARK = "#222222"

DAY_COLORS = {0.0: GREEN, 1.0: BLUE, 2.0: ORANGE, 3.0: RED}


def read_table(name: str) -> pd.DataFrame:
    return pd.read_csv(TAB / name)


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial"],
            "font.size": 8.5,
            "axes.titlesize": 9.2,
            "axes.labelsize": 8.4,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "legend.fontsize": 7.4,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#333333",
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "savefig.dpi": 320,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def clean_axis(ax, grid: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(True, color="#E5E5E5", linewidth=0.7, zorder=0)
    else:
        ax.grid(False)


def panel_label(ax, label: str, x: float = -0.12, y: float = 1.10) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        color="#111111",
    )


def plot_route(ax) -> None:
    axis_df = read_table("new_endpoint_vs_dynamic_axis_summary.csv")
    summary = read_table("new_endpoint_vs_dynamic_summary.csv").set_index("quantity")["value"]
    ratio = float(summary["mean_path_to_endpoint_ratio"])
    gen = axis_df[axis_df["source"] == "generated"].sort_values("time")
    obs = axis_df[axis_df["source"] == "observed"].sort_values("time")
    ax.plot(gen["time"], gen["mean_axis"], color=TEAL, lw=2.4, label="CellCompass route")
    endpoint = np.interp(gen["time"], [0, 3], [obs.iloc[0]["mean_axis"], obs.iloc[-1]["mean_axis"]])
    ax.plot(gen["time"], endpoint, color=GRAY, lw=1.35, ls=(0, (4, 2)), label="endpoint shortcut")
    ax.scatter(
        obs["time"],
        obs["mean_axis"],
        s=48,
        color=[DAY_COLORS[float(t)] for t in obs["time"]],
        edgecolor="white",
        linewidth=0.8,
        zorder=4,
    )
    ax.text(
        0.04,
        0.93,
        f"route / endpoint = {ratio:.2f}x",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color=DARK,
        bbox={"boxstyle": "round,pad=0.24", "fc": "white", "ec": "#D7D7D7", "alpha": 0.96},
    )
    ax.set_title("Continuous route, not just endpoint shift")
    ax.set_xlabel("5-AZA-CdR treatment day")
    ax.set_ylabel("response-axis coordinate")
    ax.set_xticks([0, 1, 2, 3])
    clean_axis(ax, grid=False)
    ax.legend(frameon=False, loc="lower right", handlelength=2.3)
    panel_label(ax, "a")


def plot_match(ax) -> None:
    df = read_table("story_observed_generated_match_by_day.csv")
    lim_min = min(df["observed_axis_mean"].min(), df["generated_axis_mean"].min()) - 0.16
    lim_max = max(df["observed_axis_mean"].max(), df["generated_axis_mean"].max()) + 0.16
    ax.plot([lim_min, lim_max], [lim_min, lim_max], color="#909090", lw=1.0, ls=(0, (4, 2)), zorder=1)
    for _, row in df.iterrows():
        t = float(row["time"])
        ax.scatter(
            row["observed_axis_mean"],
            row["generated_axis_mean"],
            s=52,
            color=DAY_COLORS[t],
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        ax.text(row["observed_axis_mean"] + 0.025, row["generated_axis_mean"] + 0.018, f"d{int(t)}", color=DAY_COLORS[t], fontsize=7.6)
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Generated days match observed states")
    ax.set_xlabel("observed response mean")
    ax.set_ylabel("generated response mean")
    ax.text(
        0.05,
        0.95,
        f"max centroid error/radius {df['centroid_distance_over_radius'].max():.2f}\n"
        f"max state distance {df['state_fraction_tvd'].max():.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.6,
        color=DARK,
        bbox={"boxstyle": "round,pad=0.22", "fc": "white", "ec": "#D7D7D7", "alpha": 0.94},
    )
    clean_axis(ax, grid=False)
    panel_label(ax, "b", x=-0.24, y=1.14)


def plot_pcie(ax) -> None:
    metrics = read_table("existing_metric_summary_long.csv")
    pcie = metrics[metrics["metric"] == "pcie"].set_index("algorithm")["value"]
    vals = pd.Series({"PulseBridge": float(pcie["PulseBridge"]), "WFR-FM": float(pcie["WFR-FM"])})
    colors = [TEAL, "#A7AFB7"]
    ax.bar(vals.index, vals.values, color=colors, width=0.55, zorder=2)
    for i, v in enumerate(vals.values):
        ax.text(i, v + 0.025, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    improvement = 100.0 * (1.0 - vals["PulseBridge"] / vals["WFR-FM"])
    pairs = read_table("pcie_gene_space_anchor_pairs.csv")
    leak = float(np.average(pairs["leakage_over_budget"], weights=pairs["confidence"]))
    ax.text(
        0.50,
        0.78,
        f"{improvement:.1f}% lower\nPCIE",
        ha="center",
        va="center",
        color=TEAL,
        fontsize=8.4,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.24", "fc": "white", "ec": "#D9E8EF", "alpha": 0.96},
    )
    ax.text(
        0.04,
        0.96,
        f"heldout anchors n={len(pairs):,}\nleakage over budget {leak:.4f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.4,
        color=DARK,
        bbox={"boxstyle": "round,pad=0.22", "fc": "white", "ec": "#DDDDDD", "alpha": 0.94},
    )
    ax.set_ylim(0, 1.13)
    ax.set_ylabel("PCIE (lower is better)")
    ax.set_title("Heldout old/new RNA validates direction")
    clean_axis(ax, grid=False)
    panel_label(ax, "c")


def plot_gene_decomposition(ax) -> None:
    genes = read_table("pcie_gene_space_top_genes.csv").head(10).iloc[::-1].copy()
    colors = [TEAL if v >= 0 else RED for v in genes["mean_unit_contribution"]]
    ax.barh(genes["gene"], genes["mean_unit_contribution"], color=colors, height=0.72)
    ax.axvline(0, color="#333333", lw=0.8)
    ax.set_xlabel("mean contribution to PCIE cosine")
    ax.set_title("Gene axes supporting pulse alignment")
    ax.tick_params(axis="y", length=0)
    clean_axis(ax, grid=False)
    panel_label(ax, "d")


def plot_real_vs_random(ax) -> None:
    real = read_table("pcie_gene_velocity_umap_values.csv")
    real_summary = real.groupby("gene").agg(real_same=("same_sign", "mean")).reset_index()
    rand = read_table("pcie_random_velocity_umap_summary.csv").rename(columns={"same_sign_fraction": "random_same"})
    selected = ["MT-RNR2", "RPS2", "MT-CO1", "MALAT1", "SAT1", "LGALS1"]
    comp = real_summary.merge(rand[["gene", "random_same"]], on="gene")
    comp = comp[comp["gene"].isin(selected)].set_index("gene").loc[selected].reset_index()
    y = np.arange(len(comp))[::-1]
    for yi, (_, row) in zip(y, comp.iterrows()):
        ax.plot([row["random_same"], row["real_same"]], [yi, yi], color="#C8CDD2", lw=1.2, zorder=1)
        ax.scatter(row["random_same"], yi, s=34, color="#FFFFFF", edgecolor=GRAY, linewidth=1.1, zorder=3, label="random velocity" if yi == y[0] else None)
        ax.scatter(row["real_same"], yi, s=40, color=TEAL, edgecolor="white", linewidth=0.7, zorder=4, label="PulseBridge" if yi == y[0] else None)
        ax.text(row["real_same"] + 0.025, yi, f"{row['real_same']:.0%}", ha="left", va="center", fontsize=7.2, color=TEAL)
    ax.axvline(0.5, color="#888888", lw=0.9, ls=(0, (3, 2)))
    ax.set_yticks(y)
    ax.set_yticklabels(comp["gene"])
    ax.set_xlim(0.34, 1.05)
    ax.set_xlabel("same-sign fraction")
    ax.set_title("Alignment exceeds random velocity")
    ax.legend(frameon=False, loc="lower right", handletextpad=0.4)
    clean_axis(ax, grid=False)
    panel_label(ax, "e")


def draw_umap_one(axs) -> None:
    df = read_table("pcie_gene_velocity_umap_values.csv")
    sub = df[df["gene"] == "MT-RNR2"].copy()
    umap = ad.read_h5ad(UMAP_H5AD).obsm["X_umap"]
    bg_idx = np.arange(umap.shape[0])
    rng = np.random.default_rng(109)
    if len(bg_idx) > 7000:
        bg_idx = rng.choice(bg_idx, size=7000, replace=False)
    cols = [
        ("pulsebridge_gene_velocity_scaled", "PulseBridge", "RdBu_r", TwoSlopeNorm(vcenter=0, vmin=-1, vmax=1)),
        ("metabolic_gene_velocity_scaled", "metabolic pulse", "RdBu_r", TwoSlopeNorm(vcenter=0, vmin=-1, vmax=1)),
        ("signed_agreement_scaled", "agreement", "viridis", TwoSlopeNorm(vcenter=0, vmin=-1, vmax=1)),
    ]
    for i, (col, title, cmap, norm) in enumerate(cols):
        ax = axs[i]
        ax.scatter(umap[bg_idx, 0], umap[bg_idx, 1], c="#D7DADF", s=2, alpha=0.18, linewidths=0, rasterized=True)
        ax.scatter(
            sub["umap_1"],
            sub["umap_2"],
            c=sub[col],
            cmap=cmap,
            norm=norm,
            s=7.8,
            alpha=0.88,
            linewidths=0,
            rasterized=True,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=7.9)
        for spine in ax.spines.values():
            spine.set_linewidth(0.7)
            spine.set_color("#333333")
        ax.set_aspect("equal", adjustable="box")
    axs[0].text(-0.18, 1.20, "f", transform=axs[0].transAxes, ha="left", va="top", fontsize=12, fontweight="bold")
    axs[1].text(0.50, 1.27, "MT-RNR2 gene velocity on PCIE anchors", transform=axs[1].transAxes, ha="center", va="bottom", fontsize=9.2)
    axs[2].text(0.98, -0.09, "98% same sign", transform=axs[2].transAxes, ha="right", va="top", fontsize=7.5, color=TEAL)


def plot_state_shift(ax) -> None:
    state = read_table("story_early_late_state_summary.csv")
    style = {"observed": "-", "generated": (0, (4, 2))}
    marker = {"observed": "o", "generated": "s"}
    for source in ["observed", "generated"]:
        sub = state[state["source"] == source]
        ax.plot(sub["time"], sub["early"], color=GREEN, ls=style[source], marker=marker[source], lw=1.8, ms=4, label=f"early {source}")
        ax.plot(sub["time"], sub["late"], color=RED, ls=style[source], marker=marker[source], lw=1.8, ms=4, label=f"late {source}")
    ax.set_title("Population shifts into late response states")
    ax.set_xlabel("treatment day")
    ax.set_ylabel("cell fraction")
    ax.set_xticks([0, 1, 2, 3])
    ax.set_ylim(0, 0.82)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.55, 1.02), handlelength=2)
    clean_axis(ax, grid=False)
    panel_label(ax, "g")


def plot_biology_timing(ax) -> None:
    timing = read_table("pulsebridge_rollout_biology_event_timing.csv")
    order = [
        "tumor_suppressor_reactivation",
        "ifitm3_antiviral_membrane",
        "antigen_presentation_surface",
        "sat1_polyamine_catabolism",
        "mitochondrial_remodeling",
        "lgals1_adhesion_escape",
        "oncogene_proliferation_repression",
    ]
    labels = {
        "tumor_suppressor_reactivation": "DKK1/GAS5 reactivation",
        "ifitm3_antiviral_membrane": "IFITM3 antiviral",
        "antigen_presentation_surface": "B2M/HLA surface",
        "sat1_polyamine_catabolism": "SAT1 polyamine",
        "mitochondrial_remodeling": "mitochondrial remodeling",
        "lgals1_adhesion_escape": "LGALS1 adaptive",
        "oncogene_proliferation_repression": "EREG/PCNA repression",
    }
    colors = {
        "desired": GREEN,
        "immune": TEAL,
        "metabolic": PURPLE,
        "adaptive": ORANGE,
        "remodeling": BROWN,
    }
    timing = timing.set_index("module").loc[order].reset_index()
    y = np.arange(len(timing))[::-1]
    for yi, (_, row) in zip(y, timing.iterrows()):
        color = colors.get(row["class"], GRAY)
        ax.plot([row["first_half_peak_time"], row["peak_time"]], [yi, yi], color=color, lw=2.0)
        ax.scatter(row["first_half_peak_time"], yi, s=36, facecolor="white", edgecolor=color, linewidth=1.4, zorder=3)
        ax.scatter(row["peak_time"], yi, s=46, marker="D", color=color, edgecolor="white", linewidth=0.7, zorder=4)
        ax.text(row["peak_time"] + 0.08, yi, f"{row['peak_time']:.1f} d", va="center", ha="left", fontsize=7.2, color=DARK)
    ax.set_yticks(y)
    ax.set_yticklabels([labels[m] for m in timing["module"]])
    ax.set_xlim(0, 3.25)
    ax.set_xlabel("generated trajectory time")
    ax.set_title("Drug-response programs are temporally ordered")
    clean_axis(ax, grid=False)
    panel_label(ax, "h")


def save_individual_panels() -> None:
    panel_dir = FIG / "nature_main_panels"
    panel_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        ("a_dynamic_route", (4.3, 3.1), lambda fig: plot_route(fig.add_subplot(111))),
        ("b_generated_observed_match", (3.6, 3.2), lambda fig: plot_match(fig.add_subplot(111))),
        ("c_pcie_scorecard", (4.1, 3.1), lambda fig: plot_pcie(fig.add_subplot(111))),
        ("d_gene_space_decomposition", (4.5, 3.2), lambda fig: plot_gene_decomposition(fig.add_subplot(111))),
        ("e_real_vs_random_velocity", (4.4, 3.2), lambda fig: plot_real_vs_random(fig.add_subplot(111))),
        ("g_state_shift", (4.2, 3.2), lambda fig: plot_state_shift(fig.add_subplot(111))),
        ("h_program_timing", (7.2, 3.1), lambda fig: plot_biology_timing(fig.add_subplot(111))),
    ]
    for name, size, draw in specs:
        fig = plt.figure(figsize=size)
        draw(fig)
        fig.savefig(panel_dir / f"panel_{name}.png", dpi=320, bbox_inches="tight")
        fig.savefig(panel_dir / f"panel_{name}.pdf", bbox_inches="tight")
        plt.close(fig)

    fig = plt.figure(figsize=(5.4, 2.4))
    grid = fig.add_gridspec(1, 3, left=0.04, right=0.98, top=0.78, bottom=0.10, wspace=0.07)
    draw_umap_one([fig.add_subplot(grid[0, i]) for i in range(3)])
    fig.savefig(panel_dir / "panel_f_mtrnr2_gene_velocity_umap.png", dpi=320, bbox_inches="tight")
    fig.savefig(panel_dir / "panel_f_mtrnr2_gene_velocity_umap.pdf", bbox_inches="tight")
    plt.close(fig)


def save_publication_composite() -> None:
    """Save a two-column-width composite suitable for manuscript assembly."""
    fig = plt.figure(figsize=(7.25, 10.15))
    outer = fig.add_gridspec(
        5,
        6,
        height_ratios=[1.02, 1.05, 0.92, 1.10, 1.08],
        width_ratios=[1, 1, 1, 1, 1, 1],
        left=0.070,
        right=0.985,
        top=0.985,
        bottom=0.065,
        hspace=0.74,
        wspace=0.82,
    )

    ax_a = fig.add_subplot(outer[0, 0:3])
    ax_b = fig.add_subplot(outer[0, 3:6])
    ax_c = fig.add_subplot(outer[1, 0:3])
    ax_e = fig.add_subplot(outer[1, 3:6])
    ax_d = fig.add_subplot(outer[2, 0:6])
    umap_grid = outer[3, 0:6].subgridspec(1, 3, wspace=0.08)
    ax_f = [fig.add_subplot(umap_grid[0, i]) for i in range(3)]
    ax_g = fig.add_subplot(outer[4, 0:3])
    ax_h = fig.add_subplot(outer[4, 3:6])

    plot_route(ax_a)
    plot_match(ax_b)
    plot_pcie(ax_c)
    plot_gene_decomposition(ax_d)
    plot_real_vs_random(ax_e)
    draw_umap_one(ax_f)
    plot_state_shift(ax_g)
    plot_biology_timing(ax_h)

    publication_titles = {
        ax_a: "Dynamic route exceeds endpoint",
        ax_b: "Generated states match observations",
        ax_c: "Metabolic RNA validates direction",
        ax_d: "Gene axes of pulse alignment",
        ax_e: "PulseBridge exceeds random velocity",
        ax_g: "Late-response state shift",
        ax_h: "Ordered drug-response programs",
    }
    for ax, title in publication_titles.items():
        ax.set_title(title)

    ax_h.set_yticklabels(["DKK1/GAS5", "IFITM3", "B2M/HLA", "SAT1", "mitochondrial", "LGALS1", "EREG/PCNA"])
    handles, _ = ax_g.get_legend_handles_labels()
    ax_g.legend(
        handles,
        ["early obs.", "late obs.", "early gen.", "late gen."],
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.56, 1.03),
        columnspacing=0.9,
        handlelength=1.6,
    )

    all_axes = [ax_a, ax_b, ax_c, ax_d, ax_e, ax_g, ax_h] + ax_f
    label_positions = {
        ax_a: (-0.12, 1.08),
        ax_b: (-0.12, 1.08),
        ax_c: (-0.12, 1.08),
        ax_d: (-0.12, 1.08),
        ax_e: (-0.12, 1.08),
        ax_g: (-0.12, 1.08),
        ax_h: (-0.12, 1.08),
        ax_f[0]: (-0.08, 1.12),
    }
    for ax in [ax_a, ax_b, ax_c, ax_d, ax_e, ax_g, ax_h]:
        ax.title.set_fontsize(7.4)
        ax.xaxis.label.set_fontsize(6.8)
        ax.yaxis.label.set_fontsize(6.8)
        ax.tick_params(labelsize=6.4)
        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontsize(5.9)
            for handle in legend.legend_handles:
                try:
                    handle.set_markersize(4.0)
                except AttributeError:
                    pass
    for ax in ax_f:
        ax.title.set_fontsize(7.0)
    for ax in all_axes:
        for text in ax.texts:
            if text.get_text() in set("abcdefgh"):
                text.set_fontsize(9.6)
                if ax in label_positions:
                    text.set_position(label_positions[ax])
            else:
                text.set_fontsize(min(text.get_fontsize(), 6.6))

    png = FIG / "story_nature_main_figure_v3_publication.png"
    pdf = FIG / "story_nature_main_figure_v3_publication.pdf"
    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    set_style()
    FIG.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(13.9, 9.7))
    outer = fig.add_gridspec(
        3,
        3,
        height_ratios=[1.0, 1.06, 1.0],
        width_ratios=[1.08, 1.0, 1.15],
        left=0.055,
        right=0.985,
        top=0.915,
        bottom=0.075,
        hspace=0.48,
        wspace=0.44,
    )

    ax_a = fig.add_subplot(outer[0, 0])
    ax_b = fig.add_subplot(outer[0, 1])
    ax_c = fig.add_subplot(outer[0, 2])
    ax_d = fig.add_subplot(outer[1, 0])
    ax_e = fig.add_subplot(outer[1, 1])
    umap_grid = outer[1, 2].subgridspec(1, 3, wspace=0.05)
    ax_f = [fig.add_subplot(umap_grid[0, i]) for i in range(3)]
    ax_g = fig.add_subplot(outer[2, 0])
    ax_h = fig.add_subplot(outer[2, 1:])

    plot_route(ax_a)
    plot_match(ax_b)
    plot_pcie(ax_c)
    plot_gene_decomposition(ax_d)
    plot_real_vs_random(ax_e)
    draw_umap_one(ax_f)
    plot_state_shift(ax_g)
    plot_biology_timing(ax_h)

    fig.suptitle(
        "CellCompass resolves HCT116 5-AZA-CdR response as a pulse-validated trajectory",
        fontsize=13.6,
        fontweight="bold",
        y=0.975,
    )
    png = FIG / "story_nature_main_figure_v2.png"
    pdf = FIG / "story_nature_main_figure_v2.pdf"
    fig.savefig(png, dpi=320, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    save_individual_panels()
    save_publication_composite()
    print(png)
    print(pdf)
    print(FIG / "nature_main_panels")
    print(FIG / "story_nature_main_figure_v3_publication.png")


if __name__ == "__main__":
    main()
