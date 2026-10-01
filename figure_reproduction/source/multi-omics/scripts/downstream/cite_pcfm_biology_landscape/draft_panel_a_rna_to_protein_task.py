#!/usr/bin/env python3
"""Draft panel A with right side as a computational CITE-seq task.

This is a visual mockup, not a data-analysis panel.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle, Circle, Polygon
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "figures" / "cite_pcfm_deep_story" / "biology_landscape_discovery"

NAVY = "#172B63"
TEAL = "#08716F"
INK = "#121722"
GRAY = "#5F6670"
LIGHT = "#F7F9FB"
EDGE = "#172033"
RNA = "#7EA6D8"
RNA_DARK = "#2F6FAF"
ADT = "#A965C8"
ADT_DARK = "#7C3DA4"
PURPLE = "#6F32D6"


def add_box(ax, x, y, w, h, ec=EDGE, fc="white", lw=2.0, r=0.035, z=1):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={r}",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        zorder=z,
    )
    ax.add_patch(box)
    return box


def add_arrow(ax, x0, y0, x1, y1, color="#777", lw=2.5, ms=22, z=5):
    arr = FancyArrowPatch(
        (x0, y0),
        (x1, y1),
        arrowstyle="-|>",
        mutation_scale=ms,
        linewidth=lw,
        color=color,
        shrinkA=0,
        shrinkB=0,
        zorder=z,
    )
    ax.add_patch(arr)
    return arr


def add_text(ax, x, y, s, size=12, weight="normal", color=INK, ha="left", va="center", **kwargs):
    return ax.text(x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha, va=va, **kwargs)


def add_matrix(ax, x, y, w, h, color, rng_seed=0, nrow=4, ncol=7, alpha=1.0):
    rng = np.random.default_rng(rng_seed)
    vals = rng.uniform(0.25, 1.0, size=(nrow, ncol))
    cell_w = w / ncol
    cell_h = h / nrow
    for i in range(nrow):
        for j in range(ncol):
            c = plt.matplotlib.colors.to_rgba(color, alpha=alpha * vals[i, j])
            ax.add_patch(
                Rectangle(
                    (x + j * cell_w, y + (nrow - 1 - i) * cell_h),
                    cell_w * 0.96,
                    cell_h * 0.9,
                    facecolor=c,
                    edgecolor="white",
                    linewidth=0.5,
                    zorder=3,
                )
            )


def add_star(ax, cx, cy, r_outer=0.035, r_inner=0.012, color="#2365B5"):
    pts = []
    for k in range(16):
        ang = np.pi / 2 + k * np.pi / 8
        r = r_outer if k % 2 == 0 else r_inner
        pts.append((cx + r * np.cos(ang), cy + r * np.sin(ang)))
    ax.add_patch(Polygon(pts, closed=True, facecolor="white", edgecolor=color, linewidth=2.0, zorder=6))
    ax.add_patch(Circle((cx, cy), r_inner * 0.55, facecolor=color, edgecolor="none", zorder=7))


def draw_person_and_paper(ax):
    ax.add_patch(Circle((0.063, 0.61), 0.027, facecolor="#303640", edgecolor="none"))
    ax.add_patch(Polygon([[0.025, 0.49], [0.101, 0.49], [0.089, 0.545], [0.063, 0.565], [0.037, 0.545]],
                         closed=True, facecolor="#303640", edgecolor="none"))
    # paper screen
    ax.add_patch(Rectangle((0.13, 0.50), 0.16, 0.15, facecolor="white", edgecolor="#444", linewidth=3))
    ax.add_patch(Rectangle((0.135, 0.475), 0.15, 0.022, facecolor="#C8C8C8", edgecolor="#C8C8C8"))
    ax.add_patch(Polygon([[0.185, 0.475], [0.215, 0.475], [0.226, 0.43], [0.174, 0.43]],
                         closed=True, facecolor="#B5B5B5", edgecolor="none"))
    ax.add_patch(Rectangle((0.162, 0.592), 0.085, 0.038, facecolor="#D3D3D3", edgecolor="none"))
    add_text(ax, 0.205, 0.611, "Paper", size=12, ha="center")
    ax.add_patch(Rectangle((0.162, 0.545), 0.038, 0.028, facecolor="#D8D8D8", edgecolor="none"))
    ax.plot([0.214, 0.245], [0.565, 0.565], color="#C9C9C9", lw=2)
    ax.plot([0.214, 0.245], [0.548, 0.548], color="#C9C9C9", lw=2)


def draw_day_strip(ax, x0, y0, labels, colors, radius=0.016, gap=0.035, fs=10):
    for i, lab in enumerate(labels):
        x = x0 + i * gap
        ax.add_patch(Circle((x, y0), radius, facecolor=colors[i], edgecolor=EDGE, linewidth=1.2, zorder=3))
        add_text(ax, x, y0, lab, size=fs, color="white", ha="center", va="center", zorder=4)
        if i < len(labels) - 1:
            ax.plot([x + radius, x0 + (i + 1) * gap - radius], [y0, y0], color="#6D7480", lw=1.6, zorder=2)


def draw_right_task(ax):
    add_text(
        ax,
        0.820,
        0.79,
        "RNA-initialized surface-protein dynamics",
        size=21,
        weight="bold",
        color=NAVY,
        ha="center",
    )

    add_box(ax, 0.668, 0.585, 0.305, 0.13, ec=NAVY, fc="white", lw=2.0, r=0.018)
    add_text(ax, 0.820, 0.695, "Reference: public paired CITE-seq states", size=12.0, weight="bold", ha="center", color=INK)
    days = ["d2", "d3", "d4", "d7", "d10"]
    for i, day in enumerate(days):
        x = 0.703 + i * 0.052
        add_text(ax, x, 0.657, day, size=8.5, color=GRAY, ha="center")
        add_matrix(ax, x - 0.022, 0.620, 0.044, 0.027, RNA, rng_seed=10 + i, nrow=3, ncol=5)
        add_matrix(ax, x - 0.022, 0.591, 0.044, 0.027, ADT, rng_seed=20 + i, nrow=3, ncol=5)
    add_text(ax, 0.930, 0.633, "RNA", size=9.5, color=RNA_DARK, ha="left")
    add_text(ax, 0.930, 0.604, "ADT", size=9.5, color=ADT_DARK, ha="left")

    add_arrow(ax, 0.820, 0.575, 0.820, 0.525, color="#808080", lw=2.0, ms=18)
    add_text(ax, 0.838, 0.548, "define RNA-only prediction task", size=10.0, color=GRAY, ha="left")

    # task row
    add_box(ax, 0.642, 0.388, 0.112, 0.095, ec="#AFC4DC", fc="#F4F8FC", lw=1.6, r=0.012)
    add_text(ax, 0.698, 0.470, "Initial RNA", size=10.5, weight="bold", ha="center")
    add_text(ax, 0.698, 0.448, "day 2 only", size=9.5, color=GRAY, ha="center")
    for rr in range(3):
        for cc in range(4):
            ax.add_patch(Circle((0.669 + cc * 0.018, 0.411 + rr * 0.019), 0.0063,
                                facecolor=RNA_DARK if (rr + cc) % 2 else "#A7C5E6",
                                edgecolor="none", alpha=0.95))

    add_arrow(ax, 0.754, 0.435, 0.782, 0.435, color="#777", lw=2.3, ms=18)
    add_box(ax, 0.782, 0.385, 0.112, 0.10, ec=NAVY, fc="white", lw=2.5, r=0.018)
    add_text(ax, 0.838, 0.453, "CITE-PCFM", size=12.3, weight="bold", color=NAVY, ha="center")
    # trajectory icon inside model
    xs = np.linspace(0.800, 0.875, 5)
    ys = np.array([0.415, 0.405, 0.397, 0.428, 0.438])
    ax.plot(xs, ys, color=NAVY, lw=2.5, zorder=3)
    ax.scatter(xs, ys, s=35, color="#8BADE2", edgecolor=NAVY, linewidth=1.5, zorder=4)

    add_arrow(ax, 0.894, 0.435, 0.920, 0.435, color="#777", lw=2.3, ms=18)
    add_box(ax, 0.920, 0.388, 0.075, 0.095, ec="#D6B8EA", fc="#FBF7FD", lw=1.6, r=0.012)
    add_text(ax, 0.958, 0.470, "ADT states", size=9.7, weight="bold", ha="center")
    add_text(ax, 0.958, 0.448, "days 2-10", size=9.2, color=GRAY, ha="center")
    for rr in range(3):
        for cc in range(4):
            ax.add_patch(Circle((0.940 + cc * 0.0155, 0.411 + rr * 0.019), 0.0058,
                                facecolor=ADT_DARK if (rr + cc) % 2 else "#C9B0E4",
                                edgecolor="none", alpha=0.95))

    add_box(ax, 0.678, 0.245, 0.307, 0.07, ec=NAVY, fc="#FFFFFF", lw=1.8, r=0.015)
    add_text(ax, 0.832, 0.286, "Generated ADT states are compared with", size=11.2, ha="center")
    add_text(ax, 0.832, 0.262, "observed protein landscapes", size=11.2, weight="bold", color=NAVY, ha="center")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(14.6, 6.2), dpi=220)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(Rectangle((0.005, 0.01), 0.99, 0.975, facecolor="white", edgecolor="black", linewidth=1.2))

    add_text(
        ax,
        0.02,
        0.94,
        "a From a CITE-seq time course to an RNA-to-protein dynamics model",
        size=24,
        weight="bold",
        color="#111111",
    )

    # column titles
    add_text(ax, 0.16, 0.82, "Scientific request", size=20, weight="bold", color=NAVY, ha="center")
    add_text(ax, 0.49, 0.82, "Problem formulation", size=20, weight="bold", color=INK, ha="center")

    # left column
    draw_person_and_paper(ax)
    add_box(ax, 0.035, 0.235, 0.225, 0.155, ec=NAVY, fc="white", lw=2.2, r=0.02)
    # small speech tail
    ax.add_patch(Polygon([[0.07, 0.39], [0.082, 0.445], [0.11, 0.39]], closed=True, facecolor="white", edgecolor=NAVY, linewidth=2.2))
    add_text(
        ax,
        0.047,
        0.315,
        "I hope to download this\n"
        "dataset and use its CITE-seq\n"
        "multi-omics data to design\n"
        "a new algorithm.",
        size=12.2,
        va="center",
    )
    add_text(ax, 0.15, 0.155, "A broad biological\nmodeling request", size=17, ha="center", va="center")

    add_arrow(ax, 0.285, 0.53, 0.325, 0.53, color="#777", lw=2.2, ms=28)

    # middle column card
    card_w, card_h = 0.155, 0.145
    add_box(ax, 0.34, 0.53, card_w, card_h, ec=TEAL, fc="white", lw=2.0, r=0.015)
    add_text(ax, 0.35, 0.655, "1. Dataset preparation", size=11.8, color=TEAL)
    ax.add_patch(Rectangle((0.375, 0.57), 0.036, 0.053, facecolor="#F4F4F4", edgecolor="#999", linewidth=1.5))
    ax.plot([0.395, 0.411], [0.623, 0.607], color="#999", lw=1.4)
    add_text(ax, 0.455, 0.605, "GEO", size=15, ha="center")
    add_text(ax, 0.417, 0.555, "Science 2024;\nGSE305370", size=11.5, ha="center", va="top")

    add_box(ax, 0.515, 0.53, card_w, card_h, ec=TEAL, fc="white", lw=2.0, r=0.015)
    add_text(ax, 0.527, 0.655, "2. Time-course\n   structure", size=11.8, color=TEAL, va="center")
    draw_day_strip(ax, 0.535, 0.598, ["2", "3", "4", "7", "10"], ["#7AA6D9", "#58B7C4", "#94BF5B", "#F1A22F", "#D36A69"], fs=8.6)
    add_text(ax, 0.592, 0.552, "days 2, 3, 4, 7, 10", size=12.5, ha="center")

    add_box(ax, 0.34, 0.215, card_w, card_h, ec=TEAL, fc="white", lw=2.0, r=0.015)
    add_text(ax, 0.35, 0.34, "3. Paired measurements", size=11.8, color=TEAL)
    add_text(ax, 0.405, 0.283, "RNA", size=13, color=RNA_DARK, ha="center", weight="bold")
    add_text(ax, 0.458, 0.283, "ADT", size=13, color=ADT_DARK, ha="center", weight="bold")
    add_text(ax, 0.431, 0.283, "+", size=18, ha="center", weight="bold")
    add_text(ax, 0.417, 0.235, "RNA + 103 ADTs", size=12.8, ha="center")

    add_box(ax, 0.515, 0.215, card_w, card_h, ec=TEAL, fc="white", lw=2.0, r=0.015)
    add_text(ax, 0.527, 0.34, "4. Model formulation", size=11.8, color=TEAL)
    add_text(ax, 0.532, 0.297, r"$dX_{\mathrm{RNA}}/dt=f(X_{\mathrm{RNA}})$", size=12.2)
    add_text(ax, 0.532, 0.265, r"$X_{\mathrm{RNA}}\rightarrow X_{\mathrm{ADT}}$", size=12.2)
    add_text(ax, 0.592, 0.225, "Paired-Coordinate\nFlow Matching", size=11.2, ha="center")

    # CellCompass hub
    add_star(ax, 0.465, 0.45, r_outer=0.035, r_inner=0.012, color="#2365B5")
    add_text(ax, 0.495, 0.45, "CellCompass", size=21, weight="bold", color="#2365B5")
    for x, y in [(0.418, 0.53), (0.592, 0.53), (0.418, 0.36), (0.592, 0.36)]:
        ax.plot([0.465, x], [0.45, y], color=TEAL, lw=1.4, ls=(0, (2, 4)))

    add_arrow(ax, 0.68, 0.53, 0.715, 0.53, color="#777", lw=2.2, ms=28)

    draw_right_task(ax)

    for ext in ("png", "pdf", "svg"):
        out = OUT_DIR / f"panel_a_rna_to_protein_task_public_cite_redesign.{ext}"
        fig.savefig(out, bbox_inches="tight", pad_inches=0.03, dpi=300)
    plt.close(fig)
    print(OUT_DIR / "panel_a_rna_to_protein_task_public_cite_redesign.png")


if __name__ == "__main__":
    main()
