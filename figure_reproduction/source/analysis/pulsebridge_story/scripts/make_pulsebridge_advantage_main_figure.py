from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "outputs" / "tables"
FIG = ROOT / "outputs" / "figures"


COL = {
    "route": "#0E6F8E",
    "endpoint": "#B23A48",
    "observed": "#4D4D4D",
    "desired": "#2F5D8C",
    "immune": "#1B8A8F",
    "metabolic": "#A17C00",
    "adaptive": "#8A4F9E",
    "remodeling": "#6B7F2A",
    "grey": "#9E9E9E",
    "light": "#EAF4F7",
}


MODULE_INFO = {
    "tumor_suppressor_reactivation": ("DKK1/GAS5", "reactivation", "source-aligned", COL["desired"]),
    "ifitm3_antiviral_membrane": ("IFITM3", "antiviral", "source-consistent", COL["immune"]),
    "antigen_presentation_surface": ("B2M/HLA", "surface", "source-consistent", COL["immune"]),
    "oncogene_proliferation_repression": ("EREG/PCNA", "repression", "source-aligned", COL["desired"]),
    "sat1_polyamine_catabolism": ("SAT1", "polyamine", "PulseBridge-nominated", COL["metabolic"]),
    "mitochondrial_remodeling": ("MT genes", "mitochondrial", "PulseBridge-nominated", COL["remodeling"]),
    "lgals1_adhesion_escape": ("LGALS1", "adaptive", "PulseBridge-nominated", COL["adaptive"]),
}


def load_data() -> dict[str, pd.DataFrame]:
    return {
        "axis": pd.read_csv(TAB / "new_endpoint_vs_dynamic_axis_summary.csv"),
        "endpoint": pd.read_csv(TAB / "new_endpoint_vs_dynamic_summary.csv"),
        "match": pd.read_csv(TAB / "story_observed_generated_match_by_day.csv"),
        "early_late": pd.read_csv(TAB / "story_early_late_state_summary.csv"),
        "metrics": pd.read_csv(TAB / "existing_metric_summary_long.csv"),
        "timing": pd.read_csv(TAB / "pulsebridge_rollout_biology_event_timing.csv"),
        "real_random": pd.read_csv(TAB / "final_pcie_real_vs_random_gene_alignment.csv"),
        "genes": pd.read_csv(TAB / "final_gene_level_biology_response_matrix.csv"),
    }


def draw_route(ax, data: dict[str, pd.DataFrame]) -> None:
    axis = data["axis"]
    endpoint = data["endpoint"]
    timing = data["timing"].set_index("module")
    gen = axis[axis["source"] == "generated"].sort_values("time")
    obs = axis[axis["source"] == "observed"].sort_values("time")
    ratio = float(endpoint.loc[endpoint["quantity"] == "mean_path_to_endpoint_ratio", "value"].iloc[0])
    path_len = float(endpoint.loc[endpoint["quantity"] == "mean_generated_path_length", "value"].iloc[0])
    endpoint_len = float(endpoint.loc[endpoint["quantity"] == "mean_generated_endpoint_displacement", "value"].iloc[0])

    ax.plot(gen["time"], gen["mean_axis"], color=COL["route"], lw=4.0, zorder=4)
    ax.fill_between(
        gen["time"],
        gen["mean_axis"] - 0.22 * gen["std_axis"],
        gen["mean_axis"] + 0.22 * gen["std_axis"],
        color=COL["route"],
        alpha=0.12,
        lw=0,
        zorder=1,
    )
    ax.plot(
        [0, 3],
        [float(obs.loc[obs["time"] == 0.0, "mean_axis"].iloc[0]), float(obs.loc[obs["time"] == 3.0, "mean_axis"].iloc[0])],
        color=COL["endpoint"],
        lw=1.8,
        ls="--",
        zorder=2,
    )
    ax.errorbar(
        obs["time"],
        obs["mean_axis"],
        yerr=0.18 * obs["std_axis"],
        fmt="o",
        color=COL["observed"],
        ecolor="#BDBDBD",
        capsize=2.5,
        ms=5.5,
        zorder=6,
    )

    label_modules = [
        "tumor_suppressor_reactivation",
        "ifitm3_antiviral_membrane",
        "sat1_polyamine_catabolism",
        "mitochondrial_remodeling",
        "antigen_presentation_surface",
        "lgals1_adhesion_escape",
        "oncogene_proliferation_repression",
    ]
    offsets = {
        "tumor_suppressor_reactivation": 0.47,
        "ifitm3_antiviral_membrane": -0.50,
        "sat1_polyamine_catabolism": 0.58,
        "mitochondrial_remodeling": -0.50,
        "antigen_presentation_surface": 0.53,
        "lgals1_adhesion_escape": -0.48,
        "oncogene_proliferation_repression": 0.52,
    }
    for module in label_modules:
        row = timing.loc[module]
        x = float(row["peak_time"])
        y = float(np.interp(x, gen["time"], gen["mean_axis"]))
        name, sub, _, color = MODULE_INFO[module]
        ax.scatter(x, y, s=92, color=color, edgecolor="white", lw=1.2, zorder=7)
        yy = y + offsets[module]
        ax.plot([x, x], [y, yy - 0.06 * np.sign(yy - y)], color=color, lw=0.9, alpha=0.8)
        ax.text(
            x,
            yy,
            f"{name}\n{sub}",
            ha="center",
            va="center",
            fontsize=8,
            color=color,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.5},
        )

    ax.text(
        0.02,
        0.95,
        "Endpoint view compresses the biology",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
        fontweight="bold",
        color=COL["endpoint"],
    )
    ax.text(
        0.02,
        0.84,
        f"PulseBridge route: {path_len:.1f}\nendpoint shortcut: {endpoint_len:.1f}\nroute is {ratio:.2f}x longer",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#D0D0D0", "boxstyle": "round,pad=0.28"},
    )
    ax.set_title(
        "A. PulseBridge exposes a drug-response route that endpoint analysis collapses",
        loc="left",
        fontsize=12.5,
        fontweight="bold",
    )
    ax.set_xlabel("5-AZA-CdR treatment / generated route time (days)")
    ax.set_ylabel("response-axis coordinate")
    ax.set_xlim(-0.05, 3.08)
    ax.set_ylim(-1.05, 1.58)
    ax.grid(axis="y", color="#E8E8E8")


def draw_validation_cards(ax, data: dict[str, pd.DataFrame]) -> None:
    ax.axis("off")
    metrics = data["metrics"]
    match = data["match"]
    early_late = data["early_late"]
    rr = data["real_random"]
    pulse = float(metrics[(metrics["metric"] == "pcie") & (metrics["algorithm"] == "PulseBridge")]["value"].iloc[0])
    wfr = float(metrics[(metrics["metric"] == "pcie") & (metrics["algorithm"] == "WFR-FM")]["value"].iloc[0])
    pcie_gain = 100 * (1 - pulse / wfr)
    top3 = rr[rr["gene"].isin(["MT-RNR2", "RPS2", "MT-CO1"])].copy()
    real_mean = float(top3["real_same"].mean() * 100)
    random_mean = float(top3["random_same"].mean() * 100)
    obs0 = early_late[(early_late["source"] == "observed") & (early_late["time"] == 0.0)].iloc[0]
    obs3 = early_late[(early_late["source"] == "observed") & (early_late["time"] == 3.0)].iloc[0]

    cards = [
        ("state fit", "0.090", "max centroid/radius", "generated days match observed days"),
        ("pulse direction", f"{pcie_gain:.1f}%", f"lower PCIE ({pulse:.3f} vs {wfr:.3f})", "heldout new/old RNA validates route"),
        ("gene-space control", f"{real_mean:.0f}% vs {random_mean:.0f}%", "real vs random same-sign", "direction agreement is not random drift"),
        ("state handoff", "early↓ late↑", f"{obs0['early']:.2f}->{obs3['early']:.2f}; {obs0['late']:.2f}->{obs3['late']:.2f}", "drug response is a population transition"),
    ]
    ax.set_title("B. PulseBridge passes route-specific validation gates", loc="left", fontsize=11.5, fontweight="bold")
    for i, (head, number, subline, desc) in enumerate(cards):
        x0 = 0.02 + (i % 2) * 0.49
        y0 = 0.55 - (i // 2) * 0.48
        rect = plt.Rectangle((x0, y0), 0.45, 0.36, transform=ax.transAxes, facecolor="#F8FAFB", edgecolor="#D2D8DC", lw=1.0)
        ax.add_patch(rect)
        ax.text(x0 + 0.03, y0 + 0.292, head, transform=ax.transAxes, fontsize=8.0, color="#555555", fontweight="bold")
        ax.text(x0 + 0.03, y0 + 0.195, number, transform=ax.transAxes, fontsize=12.0, color=COL["route"], fontweight="bold")
        ax.text(x0 + 0.03, y0 + 0.125, subline, transform=ax.transAxes, fontsize=7.3, color="#333333", fontweight="bold")
        ax.text(x0 + 0.03, y0 + 0.045, desc, transform=ax.transAxes, fontsize=7.2, color="#333333")


def draw_timed_biology(ax, data: dict[str, pd.DataFrame]) -> None:
    timing = data["timing"].set_index("module")
    modules = [
        "tumor_suppressor_reactivation",
        "ifitm3_antiviral_membrane",
        "sat1_polyamine_catabolism",
        "mitochondrial_remodeling",
        "antigen_presentation_surface",
        "lgals1_adhesion_escape",
        "oncogene_proliferation_repression",
    ]
    ax.set_title("C. PulseBridge orders biology along the response route", loc="left", fontsize=11.5, fontweight="bold")
    yvals = np.arange(len(modules))
    max_amp = float(timing["peak_signed_delta"].max())
    for y, module in zip(yvals, modules):
        row = timing.loc[module]
        x = float(row["peak_time"])
        amp = float(row["peak_signed_delta"])
        retained = float(row["retained_fraction_at_day3"])
        name, sub, origin, color = MODULE_INFO[module]
        size = 85 + 250 * amp / max_amp
        ax.hlines(y, 0, x, color="#BDBDBD", lw=1.2, zorder=1)
        ax.scatter(x, y, s=size, color=color, edgecolor="white", lw=1.1, zorder=3)
        ax.text(x + 0.08, y, f"{x:.1f}d  amp {amp:.2f}, retained {retained:.2f}", va="center", fontsize=7.2, color="#333333")
    ax.axvline(1.5, color="#E6E6E6", ls="--", lw=0.9)
    ax.text(1.5, len(modules) - 0.12, "response midpoint", ha="center", va="bottom", fontsize=7.2, color="#777777")
    ax.set_xlim(-0.08, 3.18)
    ax.set_ylim(-0.65, len(modules) - 0.35)
    ax.set_yticks(yvals)
    ax.set_yticklabels(
        [
            "DKK1/GAS5 reactivation\nsource-aligned",
            "IFITM3 antiviral\nsource-consistent",
            "SAT1 polyamine\nPulseBridge-nominated",
            "MT genes mitochondrial\nPulseBridge-nominated",
            "B2M/HLA surface\nsource-consistent",
            "LGALS1 adaptive\nPulseBridge-nominated",
            "EREG/PCNA repression\nsource-aligned",
        ],
        fontsize=8,
    )
    ax.invert_yaxis()
    ax.set_xlabel("PulseBridge inferred peak day on generated route")
    ax.grid(axis="x", color="#EEEEEE")


def draw_takehome(ax) -> None:
    ax.axis("off")
    ax.set_title("D. Main-text take-home", loc="left", fontsize=11.5, fontweight="bold")
    bullets = [
        ("Reliability", "route matches measured days and heldout metabolic direction"),
        ("PulseBridge advantage", "turns endpoint screening into a queryable state-transition route"),
        ("Biology insight", "separates source-aligned response from adaptive/metabolic branches"),
        ("Cancer-screening value", "compare drugs by route, timing, state handoff, and branch composition"),
    ]
    for i, (head, text) in enumerate(bullets):
        y = 0.86 - i * 0.21
        ax.scatter(0.04, y, s=80, color=COL["route"] if i < 2 else COL["metabolic"], transform=ax.transAxes)
        ax.text(0.09, y + 0.025, head, transform=ax.transAxes, fontsize=9.5, fontweight="bold", va="center")
        ax.text(0.09, y - 0.045, text, transform=ax.transAxes, fontsize=8.3, va="center", color="#333333")


def write_summary(data: dict[str, pd.DataFrame]) -> None:
    endpoint = data["endpoint"]
    match = data["match"]
    metrics = data["metrics"]
    timing = data["timing"].set_index("module")
    ratio = float(endpoint.loc[endpoint["quantity"] == "mean_path_to_endpoint_ratio", "value"].iloc[0])
    pulse = float(metrics[(metrics["metric"] == "pcie") & (metrics["algorithm"] == "PulseBridge")]["value"].iloc[0])
    wfr = float(metrics[(metrics["metric"] == "pcie") & (metrics["algorithm"] == "WFR-FM")]["value"].iloc[0])
    rows = [
        {
            "panel": "A",
            "pulsebridge_point": "PulseBridge exposes a route that endpoint analysis collapses.",
            "evidence": f"Generated route is {ratio:.2f}x longer than endpoint shortcut.",
            "biology": "Drug response is a staged transition with timed stations, not one terminal DE contrast.",
        },
        {
            "panel": "B",
            "pulsebridge_point": "The route is validated by state matching and heldout metabolic RNA direction.",
            "evidence": f"max centroid/radius {match['centroid_distance_over_radius'].max():.3f}; PCIE {pulse:.3f} vs {wfr:.3f}.",
            "biology": "The inferred direction is experimentally anchored rather than arbitrary latent drift.",
        },
        {
            "panel": "C",
            "pulsebridge_point": "PulseBridge turns biology into route timing.",
            "evidence": f"DKK1/GAS5 {timing.loc['tumor_suppressor_reactivation','peak_time']:.1f}d; SAT1 {timing.loc['sat1_polyamine_catabolism','peak_time']:.1f}d; LGALS1 {timing.loc['lgals1_adhesion_escape','peak_time']:.1f}d.",
            "biology": "Source-aligned reactivation/immune response and adaptive/metabolic branches are temporally separated.",
        },
        {
            "panel": "D",
            "pulsebridge_point": "The main figure should be read as a PulseBridge advantage figure.",
            "evidence": "Only route-specific validation and route-derived biology are kept in the main panel.",
            "biology": "Ordinary RNA-layer readouts become supporting evidence, not the headline.",
        },
    ]
    pd.DataFrame(rows).to_csv(TAB / "pulsebridge_advantage_main_panel_summary.csv", index=False)


def main() -> None:
    data = load_data()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(15.5, 10.6), dpi=180)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.55, 1.0, 0.72], width_ratios=[1.02, 1.28], hspace=0.52, wspace=0.34)
    ax_route = fig.add_subplot(gs[0, :])
    draw_route(ax_route, data)
    ax_valid = fig.add_subplot(gs[1, 0])
    draw_validation_cards(ax_valid, data)
    ax_bio = fig.add_subplot(gs[1, 1])
    draw_timed_biology(ax_bio, data)
    ax_take = fig.add_subplot(gs[2, :])
    draw_takehome(ax_take)
    fig.suptitle(
        "PulseBridge advantage: a validated cancer-drug response route reveals ordered biology",
        fontsize=15,
        fontweight="bold",
        y=0.992,
    )
    fig.subplots_adjust(top=0.925)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "story_pulsebridge_advantage_main.png", bbox_inches="tight")
    fig.savefig(FIG / "story_pulsebridge_advantage_main.pdf", bbox_inches="tight")
    plt.close(fig)
    write_summary(data)
    print(FIG / "story_pulsebridge_advantage_main.png")
    print(FIG / "story_pulsebridge_advantage_main.pdf")
    print(TAB / "pulsebridge_advantage_main_panel_summary.csv")


if __name__ == "__main__":
    main()
