#!/usr/bin/env python3
"""Fig06Q: ridgeline summary of RNA-to-protein relay programs.

This plot is a visual summary of Fig06N. It uses the natural HSC-start
CITE-PCFM rollout time-course table and collapses marker-level heatmaps into
branch-specific RNA and ADT program waves along rollout pseudotime.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/cytobridge_mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/cytobridge_xdg_cache")
os.environ.setdefault("NUMBA_CACHE_DIR", "/private/tmp/numba_cache")
os.environ.setdefault("NUMBA_DISABLE_CACHE", "1")
for key in ["MPLCONFIGDIR", "XDG_CACHE_HOME", "NUMBA_CACHE_DIR"]:
    Path(os.environ[key]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from fig06_rna_protein_regulatory_map import OUT_DIR
from fig06n_rna_protein_relay_heatmap import BRANCH_ORDER, BRANCH_SPECS, FIG_DIR
from multistageot_style import setup


TIMECOURSE_PATH = OUT_DIR / "fig06n_hsc_seed_natural_rna_protein_relay_timecourse.csv"
BRANCH_COUNTS_PATH = OUT_DIR / "fig06n_hsc_seed_terminal_branch_counts.csv"


def minmax(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    lo = float(np.nanmin(v))
    hi = float(np.nanmax(v))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-8:
        return np.full(v.shape, 0.5, dtype=float)
    return (v - lo) / (hi - lo)


def build_program_waves(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for branch in BRANCH_ORDER:
        for feature_type in ["RNA", "ADT"]:
            sub = df[(df["terminal_branch"] == branch) & (df["feature_type"] == feature_type)].copy()
            if sub.empty:
                continue
            pivot = sub.pivot_table(index="feature", columns="time_index", values="row_minmax_scaled").sort_index()
            wave = pivot.to_numpy(dtype=float).mean(axis=0)
            times = (
                sub[["time_index", "rollout_time"]]
                .drop_duplicates()
                .sort_values("time_index")["rollout_time"]
                .to_numpy(dtype=float)
            )
            wave = minmax(gaussian_filter1d(wave, sigma=1.15, mode="nearest"))
            peak_t = float(times[int(np.nanargmax(wave))])
            for t, y in zip(times, wave):
                rows.append(
                    {
                        "terminal_branch": branch,
                        "feature_type": feature_type,
                        "rollout_time": float(t),
                        "program_wave": float(y),
                        "n_features": int(pivot.shape[0]),
                        "peak_time": peak_t,
                    }
                )
    return pd.DataFrame(rows)


def plot_ridgeline(waves: pd.DataFrame, branch_counts: pd.DataFrame, out_png: Path) -> None:
    setup()
    row_specs = []
    for branch in BRANCH_ORDER:
        if branch not in set(waves["terminal_branch"]):
            continue
        for feature_type in ["RNA", "ADT"]:
            if ((waves["terminal_branch"] == branch) & (waves["feature_type"] == feature_type)).any():
                row_specs.append((branch, feature_type))

    fig, ax = plt.subplots(figsize=(8.6, 7.1))
    fig.subplots_adjust(left=0.26, right=0.92, top=0.82, bottom=0.13)
    vertical_gap = 0.72
    amplitude = 0.58
    y_positions = np.arange(len(row_specs))[::-1] * vertical_gap

    for y0, (branch, feature_type) in zip(y_positions, row_specs):
        sub = waves[(waves["terminal_branch"] == branch) & (waves["feature_type"] == feature_type)].sort_values("rollout_time")
        x = sub["rollout_time"].to_numpy(dtype=float)
        y = sub["program_wave"].to_numpy(dtype=float)
        color = BRANCH_SPECS[branch]["color"]
        alpha = 0.74 if feature_type == "RNA" else 0.38
        edge_alpha = 0.95 if feature_type == "RNA" else 0.72
        ax.hlines(y0, x.min(), x.max(), color="#111827", lw=0.55, alpha=0.8, zorder=1)
        ax.fill_between(x, y0, y0 + amplitude * y, color=color, alpha=alpha, lw=0, zorder=3)
        ax.plot(x, y0 + amplitude * y, color=color, lw=1.25, alpha=edge_alpha, zorder=4)
        peak_t = float(sub["peak_time"].iloc[0])
        ax.plot(
            [peak_t, peak_t],
            [y0, y0 + amplitude * float(np.nanmax(y))],
            color=color,
            lw=0.85,
            alpha=0.60,
            zorder=5,
        )

        if feature_type == "RNA":
            n = int(branch_counts.set_index("terminal_branch").loc[branch, "n_hsc_seed_particles"])
            label = f"{BRANCH_SPECS[branch]['title']}\n{branch}, n={n}"
            ax.text(
                -0.12,
                y0 + 0.19,
                label,
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="center",
                color=color,
                fontsize=8.6,
                weight="bold",
                linespacing=1.05,
            )
        ax.text(
            -0.02,
            y0 + 0.18,
            feature_type,
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            color="#2563EB" if feature_type == "RNA" else "#111827",
            fontsize=7.7,
            weight="bold",
        )

    ax.set_xlim(0, 4)
    ax.set_ylim(-0.28, y_positions.max() + 0.85)
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_xticklabels(["t0", "t1", "t2", "t3", "t4"], fontsize=9)
    ax.set_yticks([])
    ax.set_xlabel("CITE-PCFM rollout pseudotime", fontsize=9.5)
    ax.annotate(
        "",
        xy=(4.0, -0.19),
        xytext=(0.0, -0.19),
        arrowprops={"arrowstyle": "->", "lw": 1.3, "color": "#111827"},
        annotation_clip=False,
    )
    ax.text(2.0, -0.46, "RNA-initialized protein movie", ha="center", va="top", fontsize=8.4, color="#334155")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)

    fig.suptitle(
        "Fig06Q. Branch-specific RNA and ADT programs form relay waves along CITE-PCFM pseudotime",
        x=0.01,
        y=0.965,
        ha="left",
        fontsize=12.8,
        weight="bold",
    )
    fig.text(
        0.01,
        0.895,
        (
            "Ridgelines summarize Fig06N: each row is the mean row-scaled activity of selected RNA genes or ADT markers "
            "within HSC-start trajectories grouped by generated terminal protein branch."
        ),
        ha="left",
        va="top",
        fontsize=8.6,
        color="#334155",
    )
    fig.savefig(out_png, dpi=360, bbox_inches="tight", facecolor="white")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> dict[str, object]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(TIMECOURSE_PATH)
    branch_counts = pd.read_csv(BRANCH_COUNTS_PATH)
    waves = build_program_waves(df)
    wave_path = OUT_DIR / "fig06q_relay_program_ridgeline_waves.csv"
    waves.to_csv(wave_path, index=False)
    fig_path = FIG_DIR / "fig06q_relay_program_ridgeline.png"
    plot_ridgeline(waves, branch_counts, fig_path)
    manifest = {
        "analysis": "fig06q_relay_program_ridgeline",
        "figure": str(fig_path),
        "figure_pdf": str(fig_path.with_suffix(".pdf")),
        "tables": {"program_waves": str(wave_path)},
        "source": str(TIMECOURSE_PATH),
        "interpretation": "Visual summary of Fig06N natural HSC-start RNA/ADT relay dynamics; not a perturbation analysis.",
    }
    manifest_path = OUT_DIR / "fig06q_relay_program_ridgeline_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    main()
