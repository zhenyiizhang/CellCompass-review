#!/usr/bin/env python3
"""Final evidence panel for source consistency and PulseBridge-specific insights."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pulsebridge_final_result_mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "analysis" / "pulsebridge_story" / "outputs"
TAB = OUT / "tables"
FIG = OUT / "figures"

TEAL = "#19778F"
GREEN = "#2D6A3F"
RED = "#B43B4A"
ORANGE = "#D97732"
PURPLE = "#7353A4"
BROWN = "#7A6B5A"
GRAY = "#7D8790"
DARK = "#222222"


def table(name: str) -> pd.DataFrame:
    return pd.read_csv(TAB / name)


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.4,
            "axes.titlesize": 9.1,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def clean(ax, grid: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(True, axis="y", color="#E6E6E6", lw=0.7, zorder=0)
    else:
        ax.grid(False)


def label(ax, s: str, x: float = -0.12, y: float = 1.08) -> None:
    ax.text(x, y, s, transform=ax.transAxes, ha="left", va="top", fontsize=11, fontweight="bold")


def plot_source_marker_check(ax) -> pd.DataFrame:
    df = table("new_original_paper_marker_readout_by_layer.csv")
    genes = ["DKK1", "GAS5", "EREG", "MALAT1", "STAT1"]
    layers = ["log1p_norm", "new", "old"]
    role_expect = {
        "DKK1": "up",
        "GAS5": "up",
        "EREG": "down",
        "MALAT1": "down",
        "STAT1": "up",
    }
    layer_label = {"log1p_norm": "total", "new": "new RNA", "old": "old RNA"}
    marker = {"log1p_norm": "s", "new": "o", "old": "^"}
    color = {"log1p_norm": TEAL, "new": GREEN, "old": GRAY}
    rows = []
    xs = np.arange(len(genes))
    offsets = {"log1p_norm": -0.18, "new": 0.0, "old": 0.18}
    for layer in layers:
        vals = []
        for gene in genes:
            sub = df[(df["gene"] == gene) & (df["layer"] == layer) & (df["time"] == 3.0)]
            delta = float(sub["delta_vs_day0"].iloc[0])
            expected = role_expect[gene]
            agrees = (delta > 0 and expected == "up") or (delta < 0 and expected == "down")
            vals.append(delta)
            rows.append(
                {
                    "gene": gene,
                    "layer": layer_label[layer],
                    "day3_delta_vs_day0": delta,
                    "source_expected": expected,
                    "direction_agrees": agrees,
                }
            )
        ax.scatter(xs + offsets[layer], vals, s=38, marker=marker[layer], color=color[layer], edgecolor="white", linewidth=0.6, label=layer_label[layer], zorder=3)
    ax.axhline(0, color="#444444", lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(genes)
    ax.set_ylabel("day3 - day0 expression")
    ax.set_title("Source-paper marker check")
    ax.set_ylim(-0.72, 3.75)
    ax.legend(frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0.00, 1.00), handletextpad=0.3, columnspacing=0.8)
    ax.text(3.72, 3.15, "MALAT1 is not\nreproduced here", ha="center", va="center", fontsize=6.8, color=RED)
    clean(ax, grid=True)
    label(ax, "a")
    return pd.DataFrame(rows)


def plot_program_class_dynamics(ax) -> pd.DataFrame:
    prof = table("pulsebridge_rollout_neighbor_biology_profile.csv")
    keep = ["desired", "immune", "metabolic", "adaptive", "remodeling"]
    colors = {"desired": GREEN, "immune": TEAL, "metabolic": PURPLE, "adaptive": ORANGE, "remodeling": BROWN}
    summary = prof[prof["class"].isin(keep)].groupby(["class", "time"], as_index=False).agg(
        mean_signed_delta=("signed_delta_from_t0", "mean"),
        max_signed_delta=("signed_delta_from_t0", "max"),
    )
    for cls in keep:
        sub = summary[summary["class"] == cls].sort_values("time")
        ax.plot(sub["time"], sub["mean_signed_delta"], lw=1.9, color=colors[cls], label=cls)
    ax.axhline(0, color="#555555", lw=0.8)
    ax.set_xlabel("generated trajectory time")
    ax.set_ylabel("mean signed program change")
    ax.set_title("PulseBridge orders response programs")
    ax.set_xlim(0, 3.0)
    ax.legend(frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(-0.02, 1.02), columnspacing=0.8, handlelength=1.4)
    clean(ax, grid=True)
    label(ax, "b")
    return summary


def plot_state_shift(ax) -> None:
    state = table("story_early_late_state_summary.csv")
    spec = {
        ("observed", "early"): (GREEN, "-", "o", "early obs."),
        ("generated", "early"): (GREEN, (0, (4, 2)), "s", "early gen."),
        ("observed", "late"): (RED, "-", "o", "late obs."),
        ("generated", "late"): (RED, (0, (4, 2)), "s", "late gen."),
    }
    for (source, col), (c, ls, mk, lab) in spec.items():
        sub = state[state["source"] == source].sort_values("time")
        ax.plot(sub["time"], sub[col], color=c, ls=ls, marker=mk, ms=4, lw=1.8, label=lab)
    ax.set_xlabel("treatment day")
    ax.set_ylabel("cell fraction")
    ax.set_title("Response is carried by late states")
    ax.set_xticks([0, 1, 2, 3])
    ax.set_ylim(0, 0.82)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.57, 1.03), handlelength=1.5, columnspacing=0.8)
    clean(ax, grid=True)
    label(ax, "c")


def plot_event_timing(ax) -> None:
    timing = table("pulsebridge_rollout_biology_event_timing.csv")
    order = [
        "tumor_suppressor_reactivation",
        "ifitm3_antiviral_membrane",
        "sat1_polyamine_catabolism",
        "mitochondrial_remodeling",
        "antigen_presentation_surface",
        "lgals1_adhesion_escape",
        "oncogene_proliferation_repression",
    ]
    label_map = {
        "tumor_suppressor_reactivation": "DKK1/GAS5",
        "ifitm3_antiviral_membrane": "IFITM3",
        "sat1_polyamine_catabolism": "SAT1",
        "mitochondrial_remodeling": "mitochondrial",
        "antigen_presentation_surface": "B2M/HLA",
        "lgals1_adhesion_escape": "LGALS1",
        "oncogene_proliferation_repression": "EREG/PCNA",
    }
    color_map = {"desired": GREEN, "immune": TEAL, "metabolic": PURPLE, "adaptive": ORANGE, "remodeling": BROWN}
    timing = timing.set_index("module").loc[order].reset_index()
    y = np.arange(len(timing))[::-1]
    for yi, (_, row) in zip(y, timing.iterrows()):
        c = color_map.get(row["class"], GRAY)
        ax.plot([row["first_half_peak_time"], row["peak_time"]], [yi, yi], lw=2, color=c)
        ax.scatter(row["first_half_peak_time"], yi, s=32, facecolor="white", edgecolor=c, lw=1.3, zorder=3)
        ax.scatter(row["peak_time"], yi, s=42, marker="D", color=c, edgecolor="white", lw=0.6, zorder=4)
        ax.text(row["peak_time"] + 0.07, yi, f"{row['peak_time']:.1f}d", va="center", ha="left", fontsize=6.8)
    ax.set_yticks(y)
    ax.set_yticklabels([label_map[m] for m in timing["module"]])
    ax.set_xlim(0, 3.24)
    ax.set_xlabel("generated trajectory time")
    ax.set_title("New insight: timing, not just markers")
    clean(ax)
    label(ax, "d")


def plot_alignment_control(ax) -> pd.DataFrame:
    real = table("pcie_gene_velocity_umap_values.csv").groupby("gene", as_index=False).agg(real_same=("same_sign", "mean"))
    rand = table("pcie_random_velocity_umap_summary.csv").rename(columns={"same_sign_fraction": "random_same"})
    genes = ["MT-RNR2", "RPS2", "MT-CO1", "MALAT1", "SAT1", "LGALS1"]
    comp = real.merge(rand[["gene", "random_same"]], on="gene")
    comp = comp[comp["gene"].isin(genes)].set_index("gene").loc[genes].reset_index()
    y = np.arange(len(comp))[::-1]
    for yi, (_, row) in zip(y, comp.iterrows()):
        ax.plot([row["random_same"], row["real_same"]], [yi, yi], color="#C7CCD1", lw=1.2)
        ax.scatter(row["random_same"], yi, s=34, facecolor="white", edgecolor=GRAY, lw=1.1, label="random velocity" if yi == y[0] else None, zorder=3)
        ax.scatter(row["real_same"], yi, s=39, color=TEAL, edgecolor="white", lw=0.6, label="PulseBridge" if yi == y[0] else None, zorder=4)
        ax.text(row["real_same"] + 0.022, yi, f"{row['real_same']:.0%}", va="center", ha="left", fontsize=6.8, color=TEAL)
    ax.axvline(0.5, color="#888888", lw=0.8, ls=(0, (3, 2)))
    ax.set_yticks(y)
    ax.set_yticklabels(comp["gene"])
    ax.set_xlim(0.34, 1.04)
    ax.set_xlabel("same-sign fraction")
    ax.set_title("Direction is not random drift")
    ax.legend(frameon=False, loc="lower right", handletextpad=0.4)
    clean(ax)
    label(ax, "e")
    return comp


def write_claim_summary(marker_check: pd.DataFrame, program_summary: pd.DataFrame, align: pd.DataFrame) -> None:
    marker_check.to_csv(TAB / "final_source_marker_direction_check.csv", index=False)
    program_summary.to_csv(TAB / "final_pulsebridge_program_class_dynamics.csv", index=False)
    align.to_csv(TAB / "final_pcie_real_vs_random_gene_alignment.csv", index=False)

    claims = [
        {
            "claim": "Well-TEMP-seq source-paper consistency",
            "evidence": "DKK1/GAS5 new RNA and total RNA rise after 5-AZA-CdR; EREG total/old RNA decreases.",
            "interpretation": "Consistent with demethylation/reactivation and partial oncogene-repression axes.",
            "status": "supported",
        },
        {
            "claim": "Boundary case",
            "evidence": "MALAT1 increases in the local prepared PulseBridge table instead of decreasing.",
            "interpretation": "Do not use MALAT1 as positive source-paper reproduction evidence.",
            "status": "caution",
        },
        {
            "claim": "PulseBridge dynamic insight",
            "evidence": "Generated route is 3.46x longer than endpoint shortcut and matches observed day states.",
            "interpretation": "Drug response is a staged route, not a static endpoint contrast.",
            "status": "new dynamic analysis",
        },
        {
            "claim": "State timing",
            "evidence": "Early states fall from 0.650 to 0.048 observed; late states rise from 0.069 to 0.600 observed.",
            "interpretation": "The response is carried by a late state redistribution that the generated trajectory reproduces.",
            "status": "new dynamic analysis",
        },
        {
            "claim": "Gene-space directional validation",
            "evidence": "MT-RNR2/RPS2/MT-CO1 real same-sign fractions are 98%/92%/79% versus random near 50%.",
            "interpretation": "Metabolic alignment is visible in gene space and not explained by random velocity.",
            "status": "validated model-specific evidence",
        },
        {
            "claim": "New hypothesis candidates",
            "evidence": "SAT1/polyamine, LGALS1/adaptive adhesion, mitochondrial, and proteostasis/iron programs are time-ordered along rollout.",
            "interpretation": "These are PulseBridge-nominated follow-up hypotheses, not causal mechanisms.",
            "status": "hypothesis-generating",
        },
        {
            "claim": "Counterfactual perturbation readiness",
            "evidence": "PulseBridge learns a continuous vector field that can be queried from chosen initial states and intervention/context labels when such labels are available.",
            "interpretation": "The HCT116 case validates the route and direction needed for counterfactual use; multi-drug or dose counterfactual claims require additional perturbation arms.",
            "status": "model capability, not fully validated in this single-drug case",
        },
    ]
    pd.DataFrame(claims).to_csv(TAB / "final_pulsebridge_claim_summary.csv", index=False)


def main() -> None:
    style()
    FIG.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(10.8, 7.3))
    gs = fig.add_gridspec(
        3,
        2,
        height_ratios=[1.0, 1.05, 1.0],
        left=0.075,
        right=0.985,
        top=0.93,
        bottom=0.08,
        hspace=0.62,
        wspace=0.35,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[2, :])

    marker_check = plot_source_marker_check(ax_a)
    program_summary = plot_program_class_dynamics(ax_b)
    plot_state_shift(ax_c)
    plot_event_timing(ax_d)
    align = plot_alignment_control(ax_e)
    fig.suptitle("PulseBridge final result: source consistency plus dynamic drug-response insights", fontsize=11.8, fontweight="bold", y=0.982)
    png = FIG / "story_final_result_evidence_panel.png"
    pdf = FIG / "story_final_result_evidence_panel.pdf"
    fig.savefig(png, dpi=360, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    write_claim_summary(marker_check, program_summary, align)
    print(png)
    print(pdf)
    print(TAB / "final_pulsebridge_claim_summary.csv")


if __name__ == "__main__":
    main()
