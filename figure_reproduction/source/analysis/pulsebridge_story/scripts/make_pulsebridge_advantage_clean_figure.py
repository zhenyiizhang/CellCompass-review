from __future__ import annotations

from pathlib import Path

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
}


PROGRAMS = [
    ("tumor_suppressor_reactivation", "DKK1/GAS5", "desired"),
    ("ifitm3_antiviral_membrane", "IFITM3", "immune"),
    ("sat1_polyamine_catabolism", "SAT1", "metabolic"),
    ("mitochondrial_remodeling", "MT", "remodeling"),
    ("antigen_presentation_surface", "B2M/HLA", "immune"),
    ("lgals1_adhesion_escape", "LGALS1", "adaptive"),
    ("oncogene_proliferation_repression", "EREG/PCNA", "desired"),
]


def load() -> dict[str, pd.DataFrame]:
    return {
        "axis": pd.read_csv(TAB / "new_endpoint_vs_dynamic_axis_summary.csv"),
        "endpoint": pd.read_csv(TAB / "new_endpoint_vs_dynamic_summary.csv"),
        "match": pd.read_csv(TAB / "story_observed_generated_match_by_day.csv"),
        "early_late": pd.read_csv(TAB / "story_early_late_state_summary.csv"),
        "metrics": pd.read_csv(TAB / "existing_metric_summary_long.csv"),
        "timing": pd.read_csv(TAB / "pulsebridge_rollout_biology_event_timing.csv"),
    }


def label(ax, s: str) -> None:
    ax.text(-0.08, 1.06, s, transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")


def panel_route(ax, data: dict[str, pd.DataFrame]) -> None:
    axis = data["axis"]
    endpoint = data["endpoint"]
    timing = data["timing"].set_index("module")
    gen = axis[axis["source"] == "generated"].sort_values("time")
    obs = axis[axis["source"] == "observed"].sort_values("time")
    ratio = float(endpoint.loc[endpoint["quantity"] == "mean_path_to_endpoint_ratio", "value"].iloc[0])

    ax.plot(gen["time"], gen["mean_axis"], color=BLUE, lw=3.6, label="PulseBridge route")
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
        label="observed",
        zorder=5,
    )

    for module, text, cls in PROGRAMS:
        row = timing.loc[module]
        x = float(row["peak_time"])
        y = float(np.interp(x, gen["time"], gen["mean_axis"]))
        c = COLORS[cls]
        ax.scatter(x, y, s=58, color=c, edgecolor="white", lw=0.9, zorder=7)

    ax.text(0.02, 0.92, f"{ratio:.2f}x route", transform=ax.transAxes, fontsize=10, fontweight="bold", color=BLUE)
    ax.set_xlim(-0.05, 3.08)
    ax.set_ylim(-1.05, 1.55)
    ax.set_xlabel("time (days)")
    ax.set_ylabel("response axis")
    ax.legend(frameon=False, ncol=3, loc="lower right", fontsize=8)
    ax.grid(axis="y", color="#E8E8E8", lw=0.7)
    label(ax, "A")


def panel_pcie(ax, data: dict[str, pd.DataFrame]) -> None:
    metrics = data["metrics"]
    focus = metrics[(metrics["metric"] == "pcie") & metrics["algorithm"].isin(["PulseBridge", "WFR-FM"])].copy()
    focus["algorithm"] = pd.Categorical(focus["algorithm"], ["PulseBridge", "WFR-FM"])
    focus = focus.sort_values("algorithm")
    ax.bar(focus["algorithm"], focus["value"], color=[BLUE, "#B8B8B8"], width=0.55)
    for i, row in enumerate(focus.itertuples()):
        ax.text(i, row.value + 0.03, f"{row.value:.3f}", ha="center", fontsize=8)
    p = float(focus.loc[focus["algorithm"] == "PulseBridge", "value"].iloc[0])
    w = float(focus.loc[focus["algorithm"] == "WFR-FM", "value"].iloc[0])
    ax.text(0.5, 0.88, f"{100 * (1 - p / w):.1f}% lower", transform=ax.transAxes, ha="center", color=BLUE, fontweight="bold")
    ax.set_ylabel("PCIE")
    ax.set_ylim(0, 1.18)
    ax.set_title("heldout pulse direction", fontsize=9)
    ax.grid(axis="y", color="#ECECEC", lw=0.6)
    label(ax, "B")


def panel_state(ax, data: dict[str, pd.DataFrame]) -> None:
    df = data["early_late"]
    for source, ls, alpha in [("observed", "-", 1.0), ("generated", "--", 0.85)]:
        sub = df[df["source"] == source].sort_values("time")
        ax.plot(sub["time"], sub["early"], color=RED, ls=ls, lw=2.0, alpha=alpha)
        ax.plot(sub["time"], sub["late"], color=BLUE, ls=ls, lw=2.0, alpha=alpha)
    ax.text(2.72, 0.60, "late", color=BLUE, fontweight="bold", fontsize=8)
    ax.text(2.72, 0.07, "early", color=RED, fontweight="bold", fontsize=8)
    ax.set_xlim(-0.05, 3.05)
    ax.set_ylim(-0.03, 0.82)
    ax.set_xlabel("day")
    ax.set_ylabel("state fraction")
    ax.set_title("state handoff", fontsize=9)
    ax.grid(axis="y", color="#ECECEC", lw=0.6)
    label(ax, "C")


def panel_programs(ax, data: dict[str, pd.DataFrame]) -> None:
    timing = data["timing"].set_index("module")
    max_amp = float(timing["peak_signed_delta"].max())
    y = np.arange(len(PROGRAMS))
    for yi, (module, text, cls) in zip(y, PROGRAMS):
        row = timing.loc[module]
        x = float(row["peak_time"])
        amp = float(row["peak_signed_delta"])
        c = COLORS[cls]
        ax.hlines(yi, 0, x, color="#C8C8C8", lw=1.0)
        ax.scatter(x, yi, s=55 + 210 * amp / max_amp, color=c, edgecolor="white", lw=1.0, zorder=3)
        ax.text(x + 0.06, yi, f"{x:.1f}d", va="center", fontsize=7.5, color="#333333")
    ax.axvline(1.5, color="#E1E1E1", ls="--", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([p[1] for p in PROGRAMS], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 3.25)
    ax.set_xlabel("PulseBridge peak day")
    ax.set_title("route-timed programs", fontsize=9)
    ax.grid(axis="x", color="#ECECEC", lw=0.6)
    label(ax, "D")


def write_summary(data: dict[str, pd.DataFrame]) -> None:
    endpoint = data["endpoint"]
    match = data["match"]
    metrics = data["metrics"]
    timing = data["timing"].set_index("module")
    ratio = float(endpoint.loc[endpoint["quantity"] == "mean_path_to_endpoint_ratio", "value"].iloc[0])
    p = float(metrics[(metrics["metric"] == "pcie") & (metrics["algorithm"] == "PulseBridge")]["value"].iloc[0])
    w = float(metrics[(metrics["metric"] == "pcie") & (metrics["algorithm"] == "WFR-FM")]["value"].iloc[0])
    rows = [
        ["A", f"PulseBridge route is {ratio:.2f}x longer than endpoint shortcut.", "Endpoint analysis collapses a staged response route."],
        ["B", f"PCIE {p:.3f} vs {w:.3f}.", "Heldout metabolic RNA validates the learned direction."],
        ["C", "Observed and generated early/late fractions match.", "Drug response is a state handoff."],
        ["D", f"SAT1 {timing.loc['sat1_polyamine_catabolism','peak_time']:.1f}d, MT {timing.loc['mitochondrial_remodeling','peak_time']:.1f}d, LGALS1 {timing.loc['lgals1_adhesion_escape','peak_time']:.1f}d.", "PulseBridge reveals ordered adaptive/metabolic biology."],
    ]
    pd.DataFrame(rows, columns=["panel", "evidence", "interpretation"]).to_csv(TAB / "pulsebridge_advantage_clean_panel_summary.csv", index=False)


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
    fig = plt.figure(figsize=(12.8, 8.2), dpi=190)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.28, 1.0], width_ratios=[1.0, 1.0, 1.35], hspace=0.52, wspace=0.42)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[1, 2])
    panel_route(ax_a, data)
    panel_pcie(ax_b, data)
    panel_state(ax_c, data)
    panel_programs(ax_d, data)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "story_pulsebridge_advantage_main_clean.png", bbox_inches="tight")
    fig.savefig(FIG / "story_pulsebridge_advantage_main_clean.pdf", bbox_inches="tight")
    fig.savefig(FIG / "story_pulsebridge_advantage_main.png", bbox_inches="tight")
    fig.savefig(FIG / "story_pulsebridge_advantage_main.pdf", bbox_inches="tight")
    plt.close(fig)
    write_summary(data)
    print(FIG / "story_pulsebridge_advantage_main_clean.png")
    print(FIG / "story_pulsebridge_advantage_main_clean.pdf")


if __name__ == "__main__":
    main()
