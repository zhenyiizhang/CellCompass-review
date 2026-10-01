#!/usr/bin/env python3
"""Particle-style BMF growth/mass visualization on SPRING.

This keeps the visual style close to the clearer particle-rollout panel:
observed SPRING cells as a light background, generated BMF particles as points.

Top row:
    branch-colored particles; point size/opacity = current mass.

Bottom row:
    generated particles colored by current growth rate d log(mass)/dt.
    Point size is kept small and stable so growth is not confused with mass.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cellcompass_mpl_cache")
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import anndata as ad
import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compose_clone_scdiffeq_style_biology as base  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
TABLE_DIR = ROOT / "downstream/cellcompass_clone_case"
FIG_DIR = ROOT / "figures/cellcompass_clone_case"

SNAPSHOT_TIMES = np.asarray([0.0, 1.0, 2.0], dtype=float)
TIME_LABELS = {0.0: "day 2", 1.0: "day 4", 2.0: "day 6"}
OBS_T_BY_MODEL_TIME = {0.0: 0, 1.0: 1, 2.0: 2}
SHOW_FATES = ["Undifferentiated", "Neutrophil", "Monocyte", "Baso", "Mast", "Meg", "Erythroid"]


def setup_style() -> None:
    base.setup_style()
    mpl.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9.0,
            "axes.titleweight": "bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def spring_extent(obs: pd.DataFrame) -> tuple[float, float, float, float]:
    x = obs["spring_x"].to_numpy(float)
    y = obs["spring_y"].to_numpy(float)
    padx = 0.035 * (np.nanmax(x) - np.nanmin(x))
    pady = 0.035 * (np.nanmax(y) - np.nanmin(y))
    return (
        float(np.nanmin(x) - padx),
        float(np.nanmax(x) + padx),
        float(np.nanmin(y) - pady),
        float(np.nanmax(y) + pady),
    )


def style_spring_axis(ax, obs: pd.DataFrame) -> None:
    xmin, xmax, ymin, ymax = spring_extent(obs)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def project_latent_to_spring(query: np.ndarray, ref_latent: np.ndarray, ref_spring: np.ndarray, *, k: int = 25) -> np.ndarray:
    tree = cKDTree(np.asarray(ref_latent, dtype=np.float32))
    d, idx = tree.query(np.asarray(query, dtype=np.float32), k=min(k, len(ref_latent)))
    if idx.ndim == 1:
        d = d[:, None]
        idx = idx[:, None]
    w = 1.0 / (np.asarray(d, dtype=float) + 1e-4)
    w = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
    return np.einsum("ij,ijk->ik", w, ref_spring[np.asarray(idx, dtype=int)])


def project_snapshot(
    query: np.ndarray,
    *,
    model_time: float,
    obs: pd.DataFrame,
    latent: np.ndarray,
    source_idx: np.ndarray,
    parent: np.ndarray,
) -> np.ndarray:
    obs_t = OBS_T_BY_MODEL_TIME[float(model_time)]
    if obs_t == 0:
        return obs.iloc[source_idx[parent]][["spring_x", "spring_y"]].to_numpy(float)
    ref_mask = obs["t"].eq(obs_t).to_numpy()
    return project_latent_to_spring(
        np.asarray(query, dtype=np.float32),
        np.asarray(latent[ref_mask], dtype=np.float32),
        obs.loc[ref_mask, ["spring_x", "spring_y"]].to_numpy(float),
        k=25,
    )


def rollout_bmf(model, resolved: dict, x0_np: np.ndarray, fates: list[str]) -> dict:
    integration_dt = float(((resolved.get("evaluation") or {}).get("trajectory_integration_dt") or 0.0125))
    integration_dt = max(integration_dt, 1e-4)
    x0 = torch.as_tensor(np.asarray(x0_np, dtype=np.float32), dtype=torch.float32)
    n0 = int(x0.shape[0])
    n_fates = len(fates)
    branch_eye = torch.eye(n_fates, dtype=x0.dtype)

    with torch.no_grad():
        t0_col = torch.zeros((n0, 1), dtype=x0.dtype)
        q = torch.softmax(model.gate_net(torch.cat([x0, t0_col], dim=1)), dim=1)
        x = x0.repeat_interleave(n_fates, dim=0).contiguous()
        branch_onehot = branch_eye.repeat(n0, 1).contiguous()
        parent = np.repeat(np.arange(n0, dtype=np.int64), n_fates)
        branch_idx = np.tile(np.arange(n_fates, dtype=np.int64), n0)
        # Official BMF simulation convention: generated total mass starts at 1.
        logw = torch.log((q.reshape(-1) / max(float(n0), 1.0)).clamp_min(1e-12)).reshape(-1, 1)

        snapshots: dict[float, np.ndarray] = {}
        weights: dict[float, np.ndarray] = {}
        growth: dict[float, np.ndarray] = {}

        def record(t_val: float) -> None:
            t_col = torch.full((x.shape[0], 1), float(t_val), dtype=x.dtype)
            inp = torch.cat([x, t_col, branch_onehot], dim=1)
            snapshots[float(t_val)] = x.cpu().numpy().copy()
            weights[float(t_val)] = torch.exp(logw).cpu().numpy().reshape(-1).copy()
            growth[float(t_val)] = model.branch_growth_net(inp).cpu().numpy().reshape(-1).copy()

        current_t = float(SNAPSHOT_TIMES[0])
        record(current_t)
        for target_t in SNAPSHOT_TIMES[1:]:
            target_t = float(target_t)
            while current_t < target_t - 1e-9:
                dt = min(integration_dt, target_t - current_t)
                t_col = torch.full((x.shape[0], 1), current_t, dtype=x.dtype)
                inp = torch.cat([x, t_col, branch_onehot], dim=1)
                x = x + model.branch_velocity_net(inp) * dt
                logw = logw + model.branch_growth_net(inp) * dt
                current_t += dt
            record(target_t)
    return {"snapshots": snapshots, "weights": weights, "growth": growth, "parent": parent, "branch_idx": branch_idx}


def make_particle_table(obs: pd.DataFrame, latent: np.ndarray, sim: dict, source_idx: np.ndarray, fates: list[str]) -> pd.DataFrame:
    rows = []
    fate_names = np.asarray(fates, dtype=object)
    for t in SNAPSHOT_TIMES:
        xy = project_snapshot(
            sim["snapshots"][float(t)],
            model_time=float(t),
            obs=obs,
            latent=latent,
            source_idx=source_idx,
            parent=sim["parent"],
        )
        w = np.asarray(sim["weights"][float(t)], dtype=float)
        g = np.asarray(sim["growth"][float(t)], dtype=float)
        ranks = pd.Series(w).rank(method="average", pct=True).to_numpy(float)
        rows.append(
            pd.DataFrame(
                {
                    "model_time": float(t),
                    "biological_day": TIME_LABELS[float(t)],
                    "source_index": source_idx[sim["parent"]],
                    "branch_fate": fate_names[sim["branch_idx"]],
                    "spring_x": xy[:, 0],
                    "spring_y": xy[:, 1],
                    "mass": w,
                    "mass_rank_within_day": ranks,
                    "growth_rate": g,
                    "mass_times_growth": w * g,
                }
            )
        )
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(TABLE_DIR / "clone_bmf_particle_spring_growth_mass_particles.csv", index=False)
    summary = (
        out.groupby(["model_time", "biological_day"], as_index=False)
        .agg(total_mass=("mass", "sum"), mass_weighted_growth=("mass_times_growth", "sum"))
    )
    summary["weighted_mean_growth"] = summary["mass_weighted_growth"] / np.maximum(summary["total_mass"], 1e-12)
    summary.to_csv(TABLE_DIR / "clone_bmf_particle_spring_growth_mass_summary.csv", index=False)
    return out


def add_fate_labels(ax, obs: pd.DataFrame, fates: list[str]) -> None:
    terminal = obs[obs["t"].eq(2)]
    offsets = {
        "Neutrophil": (16, -26),
        "Monocyte": (23, 22),
        "Baso": (-24, -14),
        "Mast": (-18, -28),
        "Meg": (-24, 18),
        "Erythroid": (18, -18),
        "Undifferentiated": (-20, 23),
    }
    for fate in fates:
        d = terminal[terminal["cell_type"].eq(fate)]
        if len(d) < 30:
            continue
        dx, dy = offsets.get(fate, (0, 0))
        ax.text(
            float(d["spring_x"].median()) + dx,
            float(d["spring_y"].median()) + dy,
            fate.replace("Undifferentiated", "Undiff."),
            fontsize=6.5,
            color=base.FATE_PALETTE.get(fate, "#111827"),
            weight="bold",
            ha="center",
            va="center",
            path_effects=[pe.Stroke(linewidth=2.1, foreground="white", alpha=0.95), pe.Normal()],
            zorder=30,
        )


def draw_background(ax, obs: pd.DataFrame) -> None:
    bg = obs
    if len(bg) > 36000:
        bg = bg.sample(n=36000, random_state=20260610)
    ax.scatter(bg["spring_x"], bg["spring_y"], s=0.55, color="#CBD5E1", alpha=0.12, linewidth=0, zorder=0)


def sample_for_plot(d: pd.DataFrame, *, n: int, seed: int) -> pd.DataFrame:
    if len(d) <= n:
        return d.copy()
    rng = np.random.default_rng(seed)
    prob = d["mass"].to_numpy(float)
    prob = prob / max(float(prob.sum()), 1e-12)
    return d.iloc[rng.choice(len(d), size=n, replace=False, p=prob)].copy()


def plot_particle_spring(obs: pd.DataFrame, particles: pd.DataFrame, fates: list[str]) -> None:
    shown_fates = [f for f in SHOW_FATES if f in fates]
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 7.05), sharex=True, sharey=True)
    growth_vals = particles["growth_rate"].to_numpy(float)
    gvmin = float(np.nanpercentile(growth_vals, 2))
    gvmax = float(np.nanpercentile(growth_vals, 98))
    growth_norm = mpl.colors.Normalize(vmin=gvmin, vmax=gvmax)
    growth_cmap = mpl.cm.Oranges
    growth_mappable = mpl.cm.ScalarMappable(norm=growth_norm, cmap=growth_cmap)

    for col, t in enumerate(SNAPSHOT_TIMES):
        label = TIME_LABELS[float(t)]
        d = particles[particles["model_time"].eq(float(t))].copy()
        d = sample_for_plot(d.sort_values("mass", ascending=True), n=24000, seed=20260610 + col)
        d = d.sort_values("mass", ascending=True)

        for row in range(2):
            ax = axes[row, col]
            draw_background(ax, obs)
            style_spring_axis(ax, obs)
            add_fate_labels(ax, obs, shown_fates)

        ax_mass = axes[0, col]
        for fate in shown_fates:
            sub = d[d["branch_fate"].eq(fate)]
            if sub.empty:
                continue
            rank = sub["mass_rank_within_day"].to_numpy(float)
            ax_mass.scatter(
                sub["spring_x"],
                sub["spring_y"],
                s=0.8 + 8.0 * rank,
                color=base.FATE_PALETTE.get(fate, "#64748B"),
                alpha=0.10 + 0.58 * rank,
                linewidth=0,
                zorder=3,
            )
        total_mass = float(particles.loc[particles["model_time"].eq(float(t)), "mass"].sum())
        ax_mass.set_title(f"{label}: branch particles, sized by mass", loc="left")
        ax_mass.text(
            0.02,
            0.03,
            f"sum mass = {total_mass:.3g}",
            transform=ax_mass.transAxes,
            fontsize=6.8,
            color="#334155",
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#CBD5E1", alpha=0.92),
            zorder=40,
        )

        ax_growth = axes[1, col]
        # For growth, use color only for growth; point size is deliberately stable.
        ax_growth.scatter(
            d["spring_x"],
            d["spring_y"],
            c=d["growth_rate"],
            s=3.2,
            cmap=growth_cmap,
            norm=growth_norm,
            alpha=0.58,
            linewidth=0,
            zorder=3,
        )
        weighted_g = float(
            particles.loc[particles["model_time"].eq(float(t)), "mass_times_growth"].sum()
            / max(total_mass, 1e-12)
        )
        ax_growth.set_title(f"{label}: growth rate of particles", loc="left")
        ax_growth.text(
            0.02,
            0.03,
            f"weighted mean growth = {weighted_g:.3g}",
            transform=ax_growth.transAxes,
            fontsize=6.8,
            color="#334155",
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#CBD5E1", alpha=0.92),
            zorder=40,
        )

    axes[0, 0].text(
        -0.08,
        0.5,
        "Mass\nparticle weight",
        transform=axes[0, 0].transAxes,
        rotation=90,
        ha="center",
        va="center",
        fontsize=9,
        weight="bold",
        color="#111827",
    )
    axes[1, 0].text(
        -0.08,
        0.5,
        "Growth\n d log(mass)/dt",
        transform=axes[1, 0].transAxes,
        rotation=90,
        ha="center",
        va="center",
        fontsize=9,
        weight="bold",
        color="#111827",
    )

    handles = [
        mpl.lines.Line2D([0], [0], marker="o", color="none", markerfacecolor=base.FATE_PALETTE[f], markersize=5.2, label=f.replace("Undifferentiated", "Undiff."))
        for f in shown_fates
    ]
    axes[0, 2].legend(handles=handles, frameon=False, fontsize=6.4, loc="upper right", bbox_to_anchor=(1.02, 1.02), ncol=1)
    cax = fig.add_axes([0.925, 0.20, 0.014, 0.28])
    cb = fig.colorbar(growth_mappable, cax=cax)
    cb.set_label("particle growth rate", fontsize=7.2)
    cb.ax.tick_params(labelsize=6.4)

    fig.suptitle("BMF rollout on SPRING: mass and growth shown as particles", x=0.03, y=0.985, ha="left", fontsize=13.2, weight="bold")
    fig.text(
        0.03,
        0.025,
        "Top row: branch-colored particles; size/opacity show current mass. Bottom row: same generated particles colored by current growth rate. No density smoothing is used.",
        fontsize=7.2,
        color="#475569",
    )
    fig.subplots_adjust(left=0.065, right=0.905, top=0.91, bottom=0.08, wspace=0.03, hspace=0.16)
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(FIG_DIR / f"panel_c_growth_mass01_bmf_particle_spring.{ext}", dpi=420 if ext == "png" else None)
    plt.close(fig)


def main() -> None:
    setup_style()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    adata = ad.read_h5ad(base.ADATA_PATH)
    obs = base.load_obs(adata)
    bmf = pd.read_csv(base.BMF_TABLE)
    fates = base.fate_names(bmf)
    latent = np.asarray(adata.obsm["X_latent"], dtype=np.float32)
    source_idx = np.flatnonzero(obs["t"].eq(0).to_numpy())
    model, resolved = base.load_bmf_model(latent.shape[1], len(fates))
    sim = rollout_bmf(model, resolved, latent[source_idx], fates)
    particles = make_particle_table(obs, latent, sim, source_idx, fates)
    plot_particle_spring(obs, particles, fates)
    summary = pd.read_csv(TABLE_DIR / "clone_bmf_particle_spring_growth_mass_summary.csv")
    print(
        {
            "n_day2_source_cells": int(len(source_idx)),
            "n_branch_particles_per_time": int(len(source_idx) * len(fates)),
            "figure": str(FIG_DIR / "panel_c_growth_mass01_bmf_particle_spring.png"),
            "tables": [
                str(TABLE_DIR / "clone_bmf_particle_spring_growth_mass_particles.csv"),
                str(TABLE_DIR / "clone_bmf_particle_spring_growth_mass_summary.csv"),
            ],
            "summary": summary.to_dict(orient="records"),
        }
    )


if __name__ == "__main__":
    main()
