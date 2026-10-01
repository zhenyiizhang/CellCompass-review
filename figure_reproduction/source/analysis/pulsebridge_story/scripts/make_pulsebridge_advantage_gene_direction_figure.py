from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cellcompass_pulsebridge_fig")

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "outputs" / "tables"
FIG = ROOT / "outputs" / "figures"

BLUE = "#0E6F8E"
RED = "#B23A48"
GREY = "#4F4F4F"
LIGHT_BLUE = "#E6F2F6"
COLORS = {
    "desired": "#2F5D8C",
    "immune": "#1B8A8F",
    "metabolic": "#A17C00",
    "adaptive": "#8A4F9E",
    "remodeling": "#6B7F2A",
    "translation": "#6D5D9E",
    "stress": "#757575",
}


PROGRAMS = [
    ("tumor_suppressor_reactivation", "DKK1/GAS5\nreactivation", "desired", 0.40),
    ("ifitm3_antiviral_membrane", "IFITM3\nantiviral", "immune", -0.36),
    ("sat1_polyamine_catabolism", "SAT1\npolyamine", "metabolic", 0.46),
    ("mitochondrial_remodeling", "MT genes\nmetabolic", "remodeling", -0.42),
    ("antigen_presentation_surface", "B2M/HLA\nsurface", "immune", 0.40),
    ("lgals1_adhesion_escape", "LGALS1\nadaptive", "adaptive", -0.39),
    ("oncogene_proliferation_repression", "EREG/PCNA\nrepression", "desired", 0.40),
]

MODULE_ORDER = [
    "reactivation",
    "immune_surface",
    "proliferation_repression",
    "adaptive_metabolic",
    "proteostasis_iron_reduction",
]
MODULE_LABELS = {
    "reactivation": "DKK1/GAS5\nreactivation",
    "immune_surface": "B2M/HLA/IFITM3\nimmune surface",
    "proliferation_repression": "EREG/PCNA\nproliferation down",
    "adaptive_metabolic": "SAT1/LGALS1/MT\nadaptive-metabolic",
    "proteostasis_iron_reduction": "HSP/FTL\nproteostasis/iron down",
}
MODULE_COLORS = {
    "reactivation": COLORS["desired"],
    "immune_surface": COLORS["immune"],
    "proliferation_repression": COLORS["desired"],
    "adaptive_metabolic": COLORS["metabolic"],
    "proteostasis_iron_reduction": COLORS["stress"],
}
MODULE_SIDE = {
    "reactivation": "anti-tumor",
    "immune_surface": "immune surface",
    "proliferation_repression": "growth down",
    "adaptive_metabolic": "adaptive branch",
    "proteostasis_iron_reduction": "stress remodeled",
}

GENE_ORDER = ["SAT1", "MT-CO1", "LGALS1", "B2M", "DKK1", "PCNA", "HSP90AB1", "FTL", "EREG"]
GENE_COLORS = {
    "SAT1": COLORS["metabolic"],
    "MT-CO1": COLORS["remodeling"],
    "LGALS1": COLORS["adaptive"],
    "B2M": COLORS["immune"],
    "DKK1": COLORS["desired"],
    "PCNA": COLORS["desired"],
    "HSP90AB1": COLORS["stress"],
    "FTL": COLORS["stress"],
    "EREG": COLORS["desired"],
}


def load() -> dict[str, pd.DataFrame]:
    return {
        "axis": pd.read_csv(TAB / "new_endpoint_vs_dynamic_axis_summary.csv"),
        "endpoint": pd.read_csv(TAB / "new_endpoint_vs_dynamic_summary.csv"),
        "metrics": pd.read_csv(TAB / "existing_metric_summary_long.csv"),
        "timing": pd.read_csv(TAB / "pulsebridge_rollout_biology_event_timing.csv"),
        "real_random": pd.read_csv(TAB / "final_pcie_real_vs_random_gene_alignment.csv"),
        "perturb_module": pd.read_csv(TAB / "pulsebridge_perturbation_module_timecourse.csv"),
        "perturb_gene": pd.read_csv(TAB / "pulsebridge_perturbation_gene_delta.csv"),
    }


def label(ax, s: str) -> None:
    ax.text(-0.07, 1.07, s, transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")


def panel_route(ax, data: dict[str, pd.DataFrame]) -> None:
    axis = data["axis"]
    endpoint = data["endpoint"]
    timing = data["timing"].set_index("module")
    gen = axis[axis["source"] == "generated"].sort_values("time")
    obs = axis[axis["source"] == "observed"].sort_values("time")
    ratio = float(endpoint.loc[endpoint["quantity"] == "mean_path_to_endpoint_ratio", "value"].iloc[0])
    path_len = float(endpoint.loc[endpoint["quantity"] == "mean_generated_path_length", "value"].iloc[0])
    endpoint_len = float(endpoint.loc[endpoint["quantity"] == "mean_generated_endpoint_displacement", "value"].iloc[0])

    ax.plot(gen["time"], gen["mean_axis"], color=BLUE, lw=3.7, label="PulseBridge route")
    ax.fill_between(
        gen["time"],
        gen["mean_axis"] - 0.22 * gen["std_axis"],
        gen["mean_axis"] + 0.22 * gen["std_axis"],
        color=LIGHT_BLUE,
        lw=0,
        zorder=0,
    )
    ax.plot(
        [0, 3],
        [float(obs.loc[obs["time"] == 0, "mean_axis"].iloc[0]), float(obs.loc[obs["time"] == 3, "mean_axis"].iloc[0])],
        color=RED,
        ls="--",
        lw=1.6,
        label="endpoint",
    )
    ax.errorbar(
        obs["time"],
        obs["mean_axis"],
        yerr=0.16 * obs["std_axis"],
        fmt="o",
        color=GREY,
        ecolor="#BDBDBD",
        capsize=2,
        ms=5,
        label="observed days",
        zorder=5,
    )

    for module, text, cls, offset in PROGRAMS:
        row = timing.loc[module]
        x = float(row["peak_time"])
        y = float(np.interp(x, gen["time"], gen["mean_axis"]))
        c = COLORS[cls]
        ax.scatter(x, y, s=72, color=c, edgecolor="white", lw=1.0, zorder=7)
        yy = y + offset
        ax.plot([x, x], [y, yy - 0.05 * np.sign(offset)], color=c, lw=0.85, alpha=0.72)
        text_x = 2.92 if x > 2.85 else x
        ha = "right" if x > 2.85 else "center"
        ax.text(
            text_x,
            yy,
            text,
            ha=ha,
            va="center",
            fontsize=7.6,
            color=c,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.55},
        )

    ax.text(
        0.02,
        0.94,
        f"{ratio:.2f}x route\nlength {path_len:.1f} vs endpoint {endpoint_len:.1f}",
        transform=ax.transAxes,
        fontsize=9.5,
        fontweight="bold",
        color=BLUE,
        va="top",
        bbox={"facecolor": "white", "edgecolor": "#D6D6D6", "pad": 3.2},
    )
    ax.text(
        0.98,
        0.08,
        "route stations order cancer-drug response programs",
        transform=ax.transAxes,
        fontsize=8.6,
        color="#222222",
        ha="right",
        va="bottom",
    )
    ax.set_xlim(-0.05, 3.12)
    ax.set_ylim(-1.05, 1.58)
    ax.set_xlabel("time along 5-AZA-CdR response route (days)")
    ax.set_ylabel("response axis")
    ax.legend(frameon=False, ncol=3, loc="lower right", fontsize=8)
    ax.grid(axis="y", color="#E8E8E8", lw=0.7)
    label(ax, "A")


def panel_validation(spec, fig, data: dict[str, pd.DataFrame]) -> None:
    sub = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=spec, height_ratios=[1.0, 0.82], hspace=0.58)
    ax_pcie = fig.add_subplot(sub[0])
    ax_gene = fig.add_subplot(sub[1])

    metrics = data["metrics"]
    focus = metrics[(metrics["metric"] == "pcie") & metrics["algorithm"].isin(["PulseBridge", "WFR-FM"])].copy()
    focus["algorithm"] = pd.Categorical(focus["algorithm"], ["PulseBridge", "WFR-FM"])
    focus = focus.sort_values("algorithm")
    ax_pcie.bar(focus["algorithm"], focus["value"], color=[BLUE, "#B8B8B8"], width=0.55)
    for i, row in enumerate(focus.itertuples()):
        ax_pcie.text(i, row.value + 0.035, f"{row.value:.3f}", ha="center", fontsize=8)
    p = float(focus.loc[focus["algorithm"] == "PulseBridge", "value"].iloc[0])
    w = float(focus.loc[focus["algorithm"] == "WFR-FM", "value"].iloc[0])
    ax_pcie.text(
        0.5,
        0.91,
        f"{100 * (1 - p / w):.1f}% lower",
        transform=ax_pcie.transAxes,
        ha="center",
        color=BLUE,
        fontweight="bold",
    )
    ax_pcie.set_ylabel("PCIE")
    ax_pcie.set_ylim(0, 1.18)
    ax_pcie.set_title("direction validated by heldout metabolic RNA", fontsize=9)
    ax_pcie.grid(axis="y", color="#ECECEC", lw=0.6)
    label(ax_pcie, "B")

    rr = data["real_random"].set_index("gene").reindex(["MT-RNR2", "RPS2", "MT-CO1"]).dropna().reset_index()
    y = np.arange(len(rr))
    ax_gene.hlines(y, rr["random_same"] * 100, rr["real_same"] * 100, color="#B9B9B9", lw=1.4, zorder=1)
    ax_gene.scatter(rr["random_same"] * 100, y, color="#8A8A8A", s=26, label="random", zorder=2)
    ax_gene.scatter(rr["real_same"] * 100, y, color=BLUE, s=42, label="PulseBridge", zorder=3, edgecolor="white", lw=0.7)
    for i, row in rr.iterrows():
        ax_gene.text(row["real_same"] * 100 + 1.2, i, f"{row['real_same'] * 100:.0f}%", va="center", fontsize=7.4)
    ax_gene.axvline(50, color="#D4D4D4", lw=0.8, ls="--")
    ax_gene.set_yticks(y)
    ax_gene.set_yticklabels(["MT-RNR2", "RPS2", "MT-CO1"], fontsize=7.5)
    ax_gene.invert_yaxis()
    ax_gene.set_xlim(45, 103)
    ax_gene.set_xlabel("same sign with pulse (%)", fontsize=7.8)
    ax_gene.set_title("gene-space check", fontsize=8.3)
    ax_gene.legend(frameon=False, fontsize=7, loc="lower right")
    ax_gene.grid(axis="x", color="#ECECEC", lw=0.6)


def panel_cancer_gene_evidence(spec, fig, data: dict[str, pd.DataFrame]) -> None:
    sub = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=spec, width_ratios=[1.18, 0.92], wspace=0.42)
    ax_mod = fig.add_subplot(sub[0])
    ax_gene = fig.add_subplot(sub[1])

    support = (
        data["perturb_module"][data["perturb_module"]["time"] == 3.0]
        .set_index("module")
        .reindex(MODULE_ORDER)
        .reset_index()
    )
    y = np.arange(len(support))
    vals = support["delta_from_day0"].astype(float).to_numpy()
    colors = [MODULE_COLORS[m] for m in support["module"]]
    ax_mod.barh(y, vals, color=colors, height=0.58)
    ax_mod.axvline(0, color="#2A2A2A", lw=0.85)
    for i, row in support.iterrows():
        val = float(row["delta_from_day0"])
        x = val + (0.035 if val >= 0 else -0.035)
        ha = "left" if val >= 0 else "right"
        ax_mod.text(x, i, f"{val:+.2f}", va="center", ha=ha, fontsize=7.5)
        ax_mod.text(1.55, i, MODULE_SIDE[row["module"]], va="center", ha="left", fontsize=7.2, color="#333333")
    ax_mod.set_yticks(y)
    ax_mod.set_yticklabels([MODULE_LABELS[m] for m in support["module"]], fontsize=7.6)
    ax_mod.invert_yaxis()
    ax_mod.set_xlim(-0.95, 2.18)
    ax_mod.set_xlabel("directional score change\n(PulseBridge rollout day3 - day0)")
    ax_mod.set_title("directional rollout programs", fontsize=9)
    ax_mod.grid(axis="x", color="#ECECEC", lw=0.6)
    label(ax_mod, "C")

    genes = data["perturb_gene"].set_index("gene").reindex(GENE_ORDER).reset_index()
    y2 = np.arange(len(genes))
    vals2 = genes["pulsebridge_day3_minus_day0"].astype(float).to_numpy()
    colors2 = [GENE_COLORS.get(g, "#777777") for g in genes["gene"]]
    ax_gene.barh(y2, vals2, color=colors2, height=0.58)
    ax_gene.axvline(0, color="#2A2A2A", lw=0.85)
    for i, row in genes.iterrows():
        val = float(row["pulsebridge_day3_minus_day0"])
        if val >= 0:
            ax_gene.text(val + 0.045, i, f"{val:+.2f}", va="center", ha="left", fontsize=7.2)
        else:
            ax_gene.text(val / 2, i, f"{val:+.2f}", va="center", ha="center", fontsize=7.2, color="white")
    ax_gene.set_yticks(y2)
    ax_gene.set_yticklabels(genes["gene"], fontsize=7.6)
    ax_gene.invert_yaxis()
    ax_gene.set_xlim(-0.86, 1.08)
    ax_gene.set_xlabel("rollout gene day3 - day0")
    ax_gene.set_title("PulseBridge rollout gene shifts", fontsize=9)
    ax_gene.grid(axis="x", color="#ECECEC", lw=0.6)


def write_summary(data: dict[str, pd.DataFrame]) -> None:
    endpoint = data["endpoint"]
    metrics = data["metrics"]
    rr = data["real_random"].set_index("gene")
    support = data["perturb_module"][data["perturb_module"]["time"] == 3.0].set_index("module")
    genes = data["perturb_gene"].set_index("gene")
    ratio = float(endpoint.loc[endpoint["quantity"] == "mean_path_to_endpoint_ratio", "value"].iloc[0])
    p = float(metrics[(metrics["metric"] == "pcie") & (metrics["algorithm"] == "PulseBridge")]["value"].iloc[0])
    w = float(metrics[(metrics["metric"] == "pcie") & (metrics["algorithm"] == "WFR-FM")]["value"].iloc[0])
    rows = [
        ["A", f"Route is {ratio:.2f}x longer than endpoint shortcut.", "PulseBridge orders source-aligned and adaptive/metabolic cancer-drug response programs on one response route."],
        [
            "B",
            f"PCIE {p:.3f} vs {w:.3f}; MT-RNR2 {rr.loc['MT-RNR2','real_same']*100:.1f}% vs random {rr.loc['MT-RNR2','random_same']*100:.1f}%.",
            "Heldout metabolic RNA validates the learned response direction, with gene-space support on MT/RPS axes.",
        ],
        [
            "C",
            f"PulseBridge rollout modules: DKK1/GAS5 {support.loc['reactivation','delta_from_day0']:+.2f}, immune surface {support.loc['immune_surface','delta_from_day0']:+.2f}, EREG/PCNA {support.loc['proliferation_repression','delta_from_day0']:+.2f}, adaptive-metabolic {support.loc['adaptive_metabolic','delta_from_day0']:+.2f}; rollout genes include SAT1 {genes.loc['SAT1','pulsebridge_day3_minus_day0']:+.2f}, MT-CO1 {genes.loc['MT-CO1','pulsebridge_day3_minus_day0']:+.2f}, LGALS1 {genes.loc['LGALS1','pulsebridge_day3_minus_day0']:+.2f}, PCNA {genes.loc['PCNA','pulsebridge_day3_minus_day0']:+.2f}.",
            "The PulseBridge rollout is tied to cancer-drug biology: demethylation/reactivation, immune-surface remodeling, proliferation repression, and adaptive metabolic stress.",
        ],
    ]
    out = pd.DataFrame(rows, columns=["panel", "evidence", "interpretation"])
    out.to_csv(TAB / "pulsebridge_advantage_cancer_gene_panel_summary.csv", index=False)
    out.to_csv(TAB / "pulsebridge_advantage_gene_direction_panel_summary.csv", index=False)


def main() -> None:
    data = load()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(13.6, 8.25), dpi=190)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1.05], width_ratios=[0.78, 1.42], hspace=0.53, wspace=0.32)
    ax_a = fig.add_subplot(gs[0, :])
    panel_route(ax_a, data)
    panel_validation(gs[1, 0], fig, data)
    panel_cancer_gene_evidence(gs[1, 1], fig, data)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "story_pulsebridge_advantage_cancer_gene_main.png", bbox_inches="tight")
    fig.savefig(FIG / "story_pulsebridge_advantage_cancer_gene_main.pdf", bbox_inches="tight")
    fig.savefig(FIG / "story_pulsebridge_advantage_gene_direction_main.png", bbox_inches="tight")
    fig.savefig(FIG / "story_pulsebridge_advantage_gene_direction_main.pdf", bbox_inches="tight")
    fig.savefig(FIG / "story_pulsebridge_advantage_main.png", bbox_inches="tight")
    fig.savefig(FIG / "story_pulsebridge_advantage_main.pdf", bbox_inches="tight")
    plt.close(fig)
    write_summary(data)
    print(FIG / "story_pulsebridge_advantage_cancer_gene_main.png")
    print(FIG / "story_pulsebridge_advantage_cancer_gene_main.pdf")


if __name__ == "__main__":
    main()
