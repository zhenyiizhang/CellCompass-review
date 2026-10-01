from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parent
OUT_PDF = ROOT / "figures" / "cellcompass_model_design_lifecycle.pdf"
OUT_PNG = ROOT / "build" / "cellcompass_model_design_lifecycle_preview.png"


INK = "#263238"
MUTED = "#5f6b70"
TEAL = "#117a8b"
TEAL_LIGHT = "#eaf5f6"
BLUE = "#3d6fb4"
BLUE_LIGHT = "#edf3fb"
ORANGE = "#c8732a"
ORANGE_LIGHT = "#fff3e8"
GREEN = "#3b7d5a"
GREEN_LIGHT = "#edf7f1"
GREY_LIGHT = "#f5f6f6"
LINE = "#a8b1b4"


def box(ax, x, y, w, h, *, face, edge, radius=0.018, lw=1.15, z=2):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        linewidth=lw,
        edgecolor=edge,
        facecolor=face,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, x0, y0, x1, y1, *, color=TEAL, lw=1.55, style="-|>", z=3, rad=0):
    patch = FancyArrowPatch(
        (x0, y0),
        (x1, y1),
        arrowstyle=style,
        mutation_scale=13,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def text(ax, x, y, value, *, size=9, weight="normal", color=INK, ha="left", va="top", style="normal"):
    ax.text(
        x,
        y,
        value,
        fontsize=size,
        fontweight=weight,
        color=color,
        ha=ha,
        va=va,
        fontstyle=style,
        family="DejaVu Sans",
        linespacing=1.28,
        zorder=4,
    )


def stage(ax, x, number, title, lines, color, face):
    y, w, h = 0.52, 0.145, 0.235
    box(ax, x, y, w, h, face=face, edge=color)
    text(ax, x + 0.014, y + h - 0.021, str(number), size=10.5, weight="bold", color=color)
    text(ax, x + 0.039, y + h - 0.021, title, size=10, weight="bold", color=INK)
    text(ax, x + 0.014, y + h - 0.076, "\n".join(lines), size=8.25, color=MUTED)
    return x, y, w, h


def main():
    fig, ax = plt.subplots(figsize=(18.2, 9.0))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    text(ax, 0.035, 0.962, "Detailed design of a CellCompass study", size=18, weight="bold")
    text(
        ax,
        0.035,
        0.922,
        "Measurements determine the dynamical quantity, mathematical model and evidence used for evaluation.",
        size=10.5,
        color=MUTED,
    )

    # Scientific reviews above the main sequence
    review_specs = [
        (0.205, "Measurement review", "Does the state and mass representation\nmatch what the experiment measures?"),
        (0.405, "Mathematical review", "Do the coupling, path and learned fields\nrecover the intended population quantity?"),
        (0.605, "Computational review", "Do the fitted losses, parameters and\nsimulation reproduce the equations?"),
        (0.805, "Evaluation review", "Does the metric use the correct data and\nmeasure the stated biological quantity?"),
    ]
    for x, title, body in review_specs:
        box(ax, x, 0.807, 0.16, 0.085, face=GREY_LIGHT, edge=LINE, radius=0.012, lw=0.9)
        text(ax, x + 0.08, 0.873, title, size=8.5, weight="bold", ha="center")
        text(ax, x + 0.08, 0.847, body, size=7.2, color=MUTED, ha="center")
        arrow(ax, x + 0.08, 0.807, x + 0.08, 0.765, color=LINE, lw=0.9, style="-|>")

    stages = [
        (0.025, 1, "Biological question\nand measurements",
         ["population and condition", "observation times", "assay-specific data", "prediction goal"], TEAL, TEAL_LIGHT),
        (0.188, 2, "Dynamical\nquantity",
         ["cell-state representation", "probability or positive measure", "deterministic or stochastic path", "roles of each measurement"], BLUE, BLUE_LIGHT),
        (0.351, 3, "Mathematical\nmodel",
         ["endpoint coupling", "conditional path", "velocity, source or score", "objective and model time"], BLUE, BLUE_LIGHT),
        (0.514, 4, "Fitting and\nsimulation",
         ["fit candidate CytoBridge models", "use fixed reference methods", "simulate the same initial cells", "generate states and readouts"], ORANGE, ORANGE_LIGHT),
        (0.677, 5, "Scientific\nevaluation",
         ["numerical validity", "future population agreement", "measurement-specific metric", "component and held-out analyses"], ORANGE, ORANGE_LIGHT),
        (0.840, 6, "Biological\ninterpretation",
         ["gene and pathway dynamics", "branch allocation", "cross-modal prediction", "perturbable effects and scope"], GREEN, GREEN_LIGHT),
    ]
    coords = [stage(ax, *spec) for spec in stages]
    for left, right in zip(coords[:-1], coords[1:]):
        x, y, w, h = left
        x2, _, _, _ = right
        arrow(ax, x + w + 0.004, y + h / 2, x2 - 0.004, y + h / 2)

    # Adaptive revision paths
    text(ax, 0.035, 0.465, "Adaptive revision", size=9.2, weight="bold", color=ORANGE)
    text(
        ax,
        0.035,
        0.435,
        "Data inspection or scientific review can return the study to the stage that needs revision.",
        size=8.2,
        color=MUTED,
    )
    arrow(ax, 0.741, 0.515, 0.271, 0.475, color=ORANGE, lw=1.1, style="-|>", rad=-0.12)
    text(ax, 0.505, 0.418, "revise measurement role or state representation", size=7.4, color=ORANGE, ha="center")
    arrow(ax, 0.741, 0.515, 0.432, 0.487, color=ORANGE, lw=1.1, style="-|>", rad=-0.08)
    text(ax, 0.585, 0.455, "revise model formulation", size=7.4, color=ORANGE, ha="center")
    arrow(ax, 0.741, 0.515, 0.596, 0.497, color=ORANGE, lw=1.1, style="-|>", rad=-0.04)
    text(ax, 0.663, 0.487, "refit or check numerics", size=7.4, color=ORANGE, ha="center")

    # Measurement-to-model examples
    text(ax, 0.035, 0.335, "Examples of measurement-to-model translation", size=11.5, weight="bold")
    examples = [
        (0.035, "DynBench", "Expression snapshots and abundance\n→ velocity, source and regulatory fields\n→ distribution, growth and fate metrics\n→ perturbable effects", TEAL_LIGHT, TEAL),
        (0.278, "PulseBridge", "Metabolic-pulse and control directions\n→ endpoint-preserving conditional path\n→ held-out pulse-direction agreement", BLUE_LIGHT, BLUE),
        (0.521, "LineageBridge", "Clone composition across observations\n→ branch-specific positive measures\n→ state, mass and fate-probability trajectories", ORANGE_LIGHT, ORANGE),
        (0.764, "ProteinBridge", "Paired RNA and protein during fitting\n→ protein-informed coupling and readout\n→ RNA-only simulation with protein prediction", GREEN_LIGHT, GREEN),
    ]
    for x, title, body, face, edge in examples:
        box(ax, x, 0.135, 0.205, 0.155, face=face, edge=edge, radius=0.014, lw=1.0)
        text(ax, x + 0.014, 0.267, title, size=9.5, weight="bold", color=edge)
        text(ax, x + 0.014, 0.231, body, size=7.2, color=INK)

    text(
        ax,
        0.5,
        0.064,
        "The fitted trajectory is the common source for distributional evaluation and downstream biological analysis.",
        size=8.8,
        color=MUTED,
        ha="center",
        va="center",
        style="italic",
    )

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(OUT_PNG, dpi=180, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


if __name__ == "__main__":
    main()
