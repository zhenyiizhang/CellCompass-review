from __future__ import annotations

from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "outputs" / "tables"
FIG = ROOT / "outputs" / "figures"


COLORS = {
    "pulse": "#0E6F8E",
    "observed": "#4A4A4A",
    "endpoint": "#B23A48",
    "desired": "#2F5D8C",
    "immune": "#1B8A8F",
    "metabolic": "#9C7A00",
    "adaptive": "#8A4F9E",
    "remodeling": "#6A7F39",
    "caution": "#C43B3B",
    "old": "#B56576",
    "new": "#2A9D8F",
    "total": "#365C8D",
    "grey": "#9E9E9E",
}


MODULE_LABELS = {
    "tumor_suppressor_reactivation": "DKK1/GAS5\nreactivation",
    "ifitm3_antiviral_membrane": "IFITM3\nantiviral",
    "sat1_polyamine_catabolism": "SAT1\npolyamine",
    "mitochondrial_remodeling": "MT genes\nmitochondrial",
    "antigen_presentation_surface": "B2M/HLA\nsurface",
    "lgals1_adhesion_escape": "LGALS1\nadaptive",
    "oncogene_proliferation_repression": "EREG/PCNA\nrepression",
}


CLASS_COLORS = {
    "desired": COLORS["desired"],
    "immune": COLORS["immune"],
    "metabolic": COLORS["metabolic"],
    "adaptive": COLORS["adaptive"],
    "remodeling": COLORS["remodeling"],
}


def panel_label(ax, label: str) -> None:
    ax.text(
        -0.08,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
        ha="left",
    )


def load_tables() -> dict[str, pd.DataFrame]:
    return {
        "axis": pd.read_csv(TAB / "new_endpoint_vs_dynamic_axis_summary.csv"),
        "endpoint": pd.read_csv(TAB / "new_endpoint_vs_dynamic_summary.csv"),
        "match": pd.read_csv(TAB / "story_observed_generated_match_by_day.csv"),
        "early_late": pd.read_csv(TAB / "story_early_late_state_summary.csv"),
        "metrics": pd.read_csv(TAB / "existing_metric_summary_long.csv"),
        "timing": pd.read_csv(TAB / "pulsebridge_rollout_biology_event_timing.csv"),
        "genes": pd.read_csv(TAB / "final_gene_level_biology_response_matrix.csv"),
        "axis_biology": pd.read_csv(TAB / "final_biology_axis_response_matrix.csv"),
        "real_random": pd.read_csv(TAB / "final_pcie_real_vs_random_gene_alignment.csv"),
    }


def route_panel(ax, data: dict[str, pd.DataFrame]) -> None:
    axis = data["axis"]
    endpoint = data["endpoint"]
    timing = data["timing"].copy()
    gen = axis[axis["source"] == "generated"].sort_values("time")
    obs = axis[axis["source"] == "observed"].sort_values("time")

    path_ratio = float(endpoint.loc[endpoint["quantity"] == "mean_path_to_endpoint_ratio", "value"].iloc[0])
    path_length = float(endpoint.loc[endpoint["quantity"] == "mean_generated_path_length", "value"].iloc[0])
    endpoint_len = float(endpoint.loc[endpoint["quantity"] == "mean_generated_endpoint_displacement", "value"].iloc[0])

    ax.plot(gen["time"], gen["mean_axis"], color=COLORS["pulse"], lw=3.2, label="PulseBridge generated route")
    ax.fill_between(
        gen["time"],
        gen["mean_axis"] - 0.22 * gen["std_axis"],
        gen["mean_axis"] + 0.22 * gen["std_axis"],
        color=COLORS["pulse"],
        alpha=0.11,
        lw=0,
    )
    ax.errorbar(
        obs["time"],
        obs["mean_axis"],
        yerr=0.18 * obs["std_axis"],
        fmt="o",
        color=COLORS["observed"],
        ecolor="#BDBDBD",
        capsize=2.2,
        ms=5.5,
        label="observed days",
        zorder=5,
    )
    ax.plot(
        [0, 3],
        [float(obs.loc[obs["time"] == 0.0, "mean_axis"].iloc[0]), float(obs.loc[obs["time"] == 3.0, "mean_axis"].iloc[0])],
        color=COLORS["endpoint"],
        lw=1.6,
        ls="--",
        label="endpoint shortcut",
        zorder=1,
    )

    y_interp = np.interp(timing["peak_time"], gen["time"], gen["mean_axis"])
    timing["route_y"] = y_interp
    timing = timing[timing["module"].isin(MODULE_LABELS)].sort_values("peak_time")
    offsets = {
        "tumor_suppressor_reactivation": 0.52,
        "ifitm3_antiviral_membrane": -0.42,
        "sat1_polyamine_catabolism": 0.62,
        "mitochondrial_remodeling": -0.50,
        "antigen_presentation_surface": 0.48,
        "lgals1_adhesion_escape": -0.47,
        "oncogene_proliferation_repression": 0.52,
    }
    for _, row in timing.iterrows():
        color = CLASS_COLORS.get(row["class"], "#555555")
        x = float(row["peak_time"])
        y = float(row["route_y"])
        ax.scatter(x, y, s=82, color=color, edgecolor="white", linewidth=1.2, zorder=6)
        yy = y + offsets.get(row["module"], 0.45)
        ax.plot([x, x], [y, yy - 0.08 * np.sign(yy - y)], color=color, lw=0.9, alpha=0.75)
        ax.text(
            x,
            yy,
            MODULE_LABELS[row["module"]],
            ha="center",
            va="center",
            fontsize=7.4,
            color=color,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.6},
        )

    ax.text(
        0.02,
        0.96,
        f"route length {path_length:.1f} vs endpoint {endpoint_len:.1f}\n{path_ratio:.2f}x longer than endpoint shortcut",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="#222222",
        bbox={"facecolor": "white", "edgecolor": "#D0D0D0", "boxstyle": "round,pad=0.28"},
    )
    ax.set_title("A. A pulse-validated drug-response route, not an endpoint jump", loc="left", fontsize=12, fontweight="bold")
    ax.set_xlabel("5-AZA-CdR treatment time / generated route time (days)")
    ax.set_ylabel("response-axis coordinate")
    ax.set_xlim(-0.06, 3.08)
    ax.grid(axis="y", color="#E6E6E6", lw=0.7)
    ax.legend(frameon=False, loc="lower right", ncol=3, fontsize=8)


def match_panel(ax, data: dict[str, pd.DataFrame]) -> None:
    match = data["match"]
    ax.scatter(match["observed_axis_mean"], match["generated_axis_mean"], s=58, color=COLORS["pulse"], edgecolor="white", lw=0.8)
    lim_min = min(match["observed_axis_mean"].min(), match["generated_axis_mean"].min()) - 0.08
    lim_max = max(match["observed_axis_mean"].max(), match["generated_axis_mean"].max()) + 0.08
    ax.plot([lim_min, lim_max], [lim_min, lim_max], color="#999999", ls="--", lw=1.0)
    for _, row in match.iterrows():
        ax.text(row["observed_axis_mean"] + 0.025, row["generated_axis_mean"] + 0.025, f"day{int(row['time'])}", fontsize=7.5)
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_xlabel("observed response axis")
    ax.set_ylabel("generated response axis")
    ax.set_title("B. Generated days match observations", loc="left", fontsize=10.5, fontweight="bold")
    ax.text(
        0.03,
        0.96,
        f"max centroid/radius = {match['centroid_distance_over_radius'].max():.3f}\nmax state TVD = {match['state_fraction_tvd'].max():.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "#D0D0D0", "boxstyle": "round,pad=0.25"},
    )
    ax.grid(color="#ECECEC", lw=0.6)


def state_handoff_panel(ax, data: dict[str, pd.DataFrame]) -> None:
    df = data["early_late"]
    for source, style, alpha in [("observed", "-", 1.0), ("generated", "--", 0.9)]:
        sub = df[df["source"] == source].sort_values("time")
        ax.plot(sub["time"], sub["early"], ls=style, color=COLORS["endpoint"], lw=2.0, alpha=alpha, label=f"{source} early")
        ax.plot(sub["time"], sub["late"], ls=style, color=COLORS["pulse"], lw=2.0, alpha=alpha, label=f"{source} late")
    ax.set_ylim(-0.03, 0.82)
    ax.set_xlim(-0.05, 3.05)
    ax.set_xlabel("treatment day")
    ax.set_ylabel("state fraction")
    ax.set_title("C. Early-to-late state handoff", loc="left", fontsize=10.5, fontweight="bold")
    ax.text(
        0.05,
        0.94,
        "early states collapse\nlate states expand",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "#D0D0D0", "boxstyle": "round,pad=0.25"},
    )
    ax.legend(frameon=False, fontsize=7, loc="center right")
    ax.grid(axis="y", color="#ECECEC", lw=0.6)


def reliability_panel(spec, fig, data: dict[str, pd.DataFrame]) -> None:
    subgs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=spec, hspace=0.42, height_ratios=[1.0, 1.05])
    ax_top = fig.add_subplot(subgs[0])
    metrics = data["metrics"]
    focus = metrics[(metrics["metric"] == "pcie") & (metrics["algorithm"].isin(["PulseBridge", "WFR-FM"]))].copy()
    focus["algorithm"] = pd.Categorical(focus["algorithm"], ["PulseBridge", "WFR-FM"])
    focus = focus.sort_values("algorithm")
    ax_top.bar(focus["algorithm"], focus["value"], color=[COLORS["pulse"], "#B8B8B8"], width=0.55)
    for i, row in enumerate(focus.itertuples()):
        ax_top.text(i, row.value + 0.025, f"{row.value:.3f}", ha="center", va="bottom", fontsize=8)
    pulse = float(focus.loc[focus["algorithm"] == "PulseBridge", "value"].iloc[0])
    wfr = float(focus.loc[focus["algorithm"] == "WFR-FM", "value"].iloc[0])
    ax_top.text(
        0.5,
        0.94,
        f"{100 * (1 - pulse / wfr):.1f}% lower\nheldout pulse error",
        transform=ax_top.transAxes,
        ha="center",
        va="top",
        fontsize=8,
        fontweight="bold",
        color=COLORS["pulse"],
    )
    ax_top.set_ylabel("PCIE")
    ax_top.set_title("D. Heldout metabolic direction validates route", loc="left", fontsize=10.5, fontweight="bold")
    ax_top.set_ylim(0, 1.15)
    ax_top.grid(axis="y", color="#ECECEC", lw=0.6)

    ax_bot = fig.add_subplot(subgs[1])
    rr = data["real_random"].set_index("gene").reindex(["MT-RNR2", "RPS2", "MT-CO1"]).dropna().reset_index()
    y = np.arange(len(rr))
    ax_bot.hlines(y, rr["random_same"] * 100, rr["real_same"] * 100, color="#BDBDBD", lw=1.4)
    ax_bot.scatter(rr["random_same"] * 100, y, color="#777777", s=32, label="random")
    ax_bot.scatter(rr["real_same"] * 100, y, color=COLORS["metabolic"], s=42, label="PulseBridge")
    ax_bot.set_yticks(y)
    ax_bot.set_yticklabels(rr["gene"])
    ax_bot.invert_yaxis()
    ax_bot.set_xlim(45, 103)
    ax_bot.set_xlabel("same sign with pulse (%)")
    ax_bot.legend(frameon=False, fontsize=7, loc="lower right")
    ax_bot.grid(axis="x", color="#ECECEC", lw=0.6)


def marker_layer_panel(ax, data: dict[str, pd.DataFrame]) -> None:
    genes = data["genes"].set_index("gene")
    order = ["DKK1", "GAS5", "EREG", "PCNA", "B2M", "IFITM3"]
    plot_df = genes.reindex(order).reset_index()
    y = np.arange(len(plot_df))
    layers = [
        ("total_day3_minus_day0", -0.22, COLORS["total"], "total"),
        ("new_day3_minus_day0", 0.0, COLORS["new"], "new RNA"),
        ("old_day3_minus_day0", 0.22, COLORS["old"], "old RNA"),
    ]
    for col, offset, color, label in layers:
        ax.barh(y + offset, plot_df[col], height=0.18, color=color, label=label)
    ax.axvline(0, color="#222222", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.invert_yaxis()
    ax.set_xlabel("measured day3 - day0")
    ax.set_title("E. RNA-layer evidence for response genes", loc="left", fontsize=10.5, fontweight="bold")
    ax.text(
        0.04,
        0.05,
        "DKK1/GAS5: nascent RNA reactivation\nEREG/PCNA: total/old repression",
        transform=ax.transAxes,
        fontsize=8,
    )
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    ax.grid(axis="x", color="#ECECEC", lw=0.6)


def biology_axis_panel(ax, data: dict[str, pd.DataFrame]) -> None:
    timing = data["timing"].copy()
    order = [
        "tumor_suppressor_reactivation",
        "ifitm3_antiviral_membrane",
        "sat1_polyamine_catabolism",
        "mitochondrial_remodeling",
        "antigen_presentation_surface",
        "lgals1_adhesion_escape",
        "oncogene_proliferation_repression",
    ]
    labels = {
        "tumor_suppressor_reactivation": "DKK1/GAS5 reactivation\nsource-aligned",
        "ifitm3_antiviral_membrane": "IFITM3 antiviral\nsource-consistent",
        "sat1_polyamine_catabolism": "SAT1 polyamine\nnew hypothesis",
        "mitochondrial_remodeling": "MT remodeling\nnew hypothesis",
        "antigen_presentation_surface": "B2M/HLA surface\nsource-consistent",
        "lgals1_adhesion_escape": "LGALS1 adaptive\nnew caution",
        "oncogene_proliferation_repression": "EREG/PCNA repression\nsource-aligned",
    }
    colors = {
        "tumor_suppressor_reactivation": COLORS["desired"],
        "ifitm3_antiviral_membrane": COLORS["immune"],
        "sat1_polyamine_catabolism": COLORS["metabolic"],
        "mitochondrial_remodeling": "#7C8E1F",
        "antigen_presentation_surface": COLORS["immune"],
        "lgals1_adhesion_escape": COLORS["adaptive"],
        "oncogene_proliferation_repression": COLORS["desired"],
    }
    timing = timing.set_index("module").reindex(order).reset_index()
    y = np.arange(len(timing))
    x = timing["peak_time"].astype(float).to_numpy()
    amp = timing["peak_signed_delta"].astype(float).to_numpy()
    retained = timing["retained_fraction_at_day3"].astype(float).to_numpy()
    point_sizes = 90 + 250 * (amp / np.nanmax(amp))
    for i, row in timing.iterrows():
        module = row["module"]
        color = colors[module]
        ax.hlines(i, 0, float(row["peak_time"]), color="#BDBDBD", lw=1.3, zorder=1)
        ax.scatter(
            float(row["peak_time"]),
            i,
            s=float(point_sizes[i]),
            color=color,
            edgecolor="white",
            linewidth=1.0,
            zorder=3,
        )
        ax.text(
            float(row["peak_time"]) + 0.08,
            i,
            f"amp {float(row['peak_signed_delta']):.2f}; retained {float(row['retained_fraction_at_day3']):.2f}",
            va="center",
            fontsize=7.2,
            color="#333333",
        )
    ax.axvline(1.5, color="#E0E0E0", lw=0.8, ls="--")
    ax.text(1.5, len(timing) - 0.25, "midpoint", ha="center", va="bottom", fontsize=7, color="#777777")
    ax.set_yticks(y)
    ax.set_yticklabels([labels[a] for a in timing["module"]])
    ax.invert_yaxis()
    ax.set_xlim(0, 3.55)
    ax.set_xlabel("PulseBridge inferred peak day on generated route")
    ax.set_title("F. Route-timed biology programs", loc="left", fontsize=10.5, fontweight="bold")
    ax.text(
        0.98,
        0.04,
        "dot size = route amplitude\ntext = retained fraction at day3",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.5,
        bbox={"facecolor": "white", "edgecolor": "#D0D0D0", "boxstyle": "round,pad=0.22"},
    )
    ax.grid(axis="x", color="#ECECEC", lw=0.6)


def write_panel_summary(data: dict[str, pd.DataFrame]) -> None:
    endpoint = data["endpoint"]
    match = data["match"]
    metrics = data["metrics"]
    genes = data["genes"].set_index("gene")
    timing = data["timing"].set_index("module")
    pulse = float(metrics[(metrics["metric"] == "pcie") & (metrics["algorithm"] == "PulseBridge")]["value"].iloc[0])
    wfr = float(metrics[(metrics["metric"] == "pcie") & (metrics["algorithm"] == "WFR-FM")]["value"].iloc[0])
    path_ratio = float(endpoint.loc[endpoint["quantity"] == "mean_path_to_endpoint_ratio", "value"].iloc[0])
    rows = [
        {
            "panel": "A",
            "result": f"PulseBridge route is {path_ratio:.2f}x longer than endpoint shortcut and carries timed gene-program stations.",
            "biology_conclusion": "5-AZA-CdR response is a staged cell-state transition rather than a static endpoint difference.",
            "insight_type": "reliability plus dynamic biology",
        },
        {
            "panel": "B",
            "result": f"Generated day slices match observed days; max centroid/radius {match['centroid_distance_over_radius'].max():.3f}, max state TVD {match['state_fraction_tvd'].max():.3f}.",
            "biology_conclusion": "The generated trajectory is constrained by measured treatment-time states.",
            "insight_type": "model reliability",
        },
        {
            "panel": "C",
            "result": "Observed and generated early states collapse while late states expand across days.",
            "biology_conclusion": "The drug response is carried by a population state handoff, not uniform gene changes in all cells.",
            "insight_type": "state-level biology",
        },
        {
            "panel": "D",
            "result": f"PulseBridge PCIE {pulse:.3f} versus WFR-FM {wfr:.3f}.",
            "biology_conclusion": "The learned direction is validated against heldout metabolic RNA pulse direction.",
            "insight_type": "directional reliability",
        },
        {
            "panel": "E",
            "result": f"DKK1 new RNA {genes.loc['DKK1', 'new_day3_minus_day0']:+.3f}; GAS5 new RNA {genes.loc['GAS5', 'new_day3_minus_day0']:+.3f}; EREG old RNA {genes.loc['EREG', 'old_day3_minus_day0']:+.3f}; PCNA old RNA {genes.loc['PCNA', 'old_day3_minus_day0']:+.3f}.",
            "biology_conclusion": "Measured RNA layers support reactivation, immune-like remodeling, and total/old-layer repression readouts.",
            "insight_type": "source-paper-aligned measured RNA evidence",
        },
        {
            "panel": "F",
            "result": f"DKK1/GAS5 peak {timing.loc['tumor_suppressor_reactivation', 'peak_time']:.1f}d; SAT1 peak {timing.loc['sat1_polyamine_catabolism', 'peak_time']:.1f}d; MT peak {timing.loc['mitochondrial_remodeling', 'peak_time']:.1f}d; LGALS1 peak {timing.loc['lgals1_adhesion_escape', 'peak_time']:.1f}d.",
            "biology_conclusion": "PulseBridge places source-aligned, metabolic, mitochondrial, and adaptive programs at distinct positions on the generated route.",
            "insight_type": "PulseBridge route timing and new biological insight",
        },
    ]
    pd.DataFrame(rows).to_csv(TAB / "cell_nbt_biology_main_figure_panel_summary.csv", index=False)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    data = load_tables()
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

    fig = plt.figure(figsize=(15.8, 12.2), dpi=180)
    gs = fig.add_gridspec(3, 3, height_ratios=[1.5, 1.0, 1.22], hspace=0.58, wspace=0.62)

    ax_route = fig.add_subplot(gs[0, :])
    route_panel(ax_route, data)

    ax_match = fig.add_subplot(gs[1, 0])
    match_panel(ax_match, data)

    ax_state = fig.add_subplot(gs[1, 1])
    state_handoff_panel(ax_state, data)

    reliability_panel(gs[1, 2], fig, data)

    ax_marker = fig.add_subplot(gs[2, 0])
    marker_layer_panel(ax_marker, data)

    ax_bio = fig.add_subplot(gs[2, 1:])
    biology_axis_panel(ax_bio, data)

    fig.suptitle(
        "PulseBridge resolves HCT116 5-AZA-CdR response into a validated route and new drug-biology axes",
        fontsize=14,
        fontweight="bold",
        y=0.992,
    )
    fig.subplots_adjust(top=0.935)
    fig.savefig(FIG / "story_cell_nbt_biology_results_main.png", bbox_inches="tight")
    fig.savefig(FIG / "story_cell_nbt_biology_results_main.pdf", bbox_inches="tight")
    plt.close(fig)
    write_panel_summary(data)
    print(FIG / "story_cell_nbt_biology_results_main.png")
    print(FIG / "story_cell_nbt_biology_results_main.pdf")
    print(TAB / "cell_nbt_biology_main_figure_panel_summary.csv")


if __name__ == "__main__":
    main()
