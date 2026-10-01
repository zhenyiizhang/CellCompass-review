"""Shared MultistageOT-inspired plotting style for CITE-PCFM biology figures."""

from __future__ import annotations

from collections.abc import Iterable

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np


BACKGROUND = "#E5E5E5"
BACKGROUND_EDGE = "#111111"
INK = "#111827"
MUTED = "#64748B"
GRID = "#E5E7EB"

PHENO_ORDER = ["HSC", "MasP", "NeuP", "EryP", "MkP", "MoP", "BP"]
PHENO_COLORS = {
    "HSC": "#2F855A",
    "MasP": "#D62728",
    "NeuP": "#E377C2",
    "EryP": "#F28E2B",
    "MkP": "#7B61B3",
    "MoP": "#8C564B",
    "BP": "#1F77B4",
}
TIME_COLORS = {
    2: "#006FFC",
    3: "#1EBFC1",
    4: "#91FE46",
    7: "#FEA840",
    10: "#D00426",
}


def setup() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "axes.edgecolor": "#111827",
            "axes.labelcolor": "#111827",
            "xtick.color": "#111827",
            "ytick.color": "#111827",
            "axes.titleweight": "bold",
            "axes.titlesize": 11.5,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.0,
            "savefig.facecolor": "white",
        }
    )


def clean_embedding_axis(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_aspect("equal", adjustable="box")


def clean_metric_axis(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, lw=0.7, alpha=0.8)
    ax.set_axisbelow(True)


def scatter_background(
    ax: plt.Axes,
    xy: np.ndarray,
    *,
    s: float = 10.0,
    alpha: float = 0.88,
    outline: bool = True,
    outline_scale: float = 1.75,
    outline_linewidth: float = 1.0,
    outline_alpha: float = 1.0,
    zorder: int = 0,
) -> None:
    if outline:
        ax.scatter(
            xy[:, 0],
            xy[:, 1],
            s=s * outline_scale,
            facecolor="none",
            edgecolor=BACKGROUND_EDGE,
            linewidth=outline_linewidth,
            alpha=outline_alpha,
            rasterized=True,
            zorder=zorder,
        )
    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        s=s,
        facecolor=BACKGROUND,
        edgecolor="none",
        alpha=alpha,
        rasterized=True,
        zorder=zorder + 1,
    )


def scatter_groups(
    ax: plt.Axes,
    xy: np.ndarray,
    labels: np.ndarray,
    palette: dict[str, str],
    *,
    order: Iterable[str] | None = None,
    s: float = 13.0,
    alpha: float = 0.84,
    zorder: int = 3,
) -> None:
    label_order = list(order) if order is not None else sorted(np.unique(labels.astype(str)))
    for label in label_order:
        mask = labels.astype(str) == str(label)
        if not np.any(mask):
            continue
        ax.scatter(
            xy[mask, 0],
            xy[mask, 1],
            s=s,
            color=palette.get(str(label), "#64748B"),
            alpha=alpha,
            edgecolor="none",
            rasterized=True,
            zorder=zorder,
        )


def add_median_labels(
    ax: plt.Axes,
    xy: np.ndarray,
    labels: np.ndarray,
    *,
    order: Iterable[str] | None = None,
    min_cells: int = 25,
    color_map: dict[str, str] | None = None,
    fontsize: float = 8.5,
) -> None:
    label_order = list(order) if order is not None else sorted(np.unique(labels.astype(str)))
    for label in label_order:
        mask = labels.astype(str) == str(label)
        if int(mask.sum()) < min_cells:
            continue
        center = np.nanmedian(xy[mask], axis=0)
        color = INK if color_map is None else color_map.get(str(label), INK)
        text = ax.text(
            center[0],
            center[1],
            str(label),
            ha="center",
            va="center",
            fontsize=fontsize,
            color=color,
            weight="bold",
            zorder=20,
        )
        text.set_path_effects([pe.withStroke(linewidth=3.0, foreground="white", alpha=0.92)])


def title_left(ax: plt.Axes, title: str, *, fontsize: float = 11.5) -> None:
    ax.set_title(title, loc="left", fontsize=fontsize, weight="bold", color=INK, pad=6)


def note_box(ax: plt.Axes, text: str, *, x: float = 0.012, y: float = 0.985) -> None:
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        fontsize=8.3,
        color="#334155",
        va="top",
        zorder=30,
    )
