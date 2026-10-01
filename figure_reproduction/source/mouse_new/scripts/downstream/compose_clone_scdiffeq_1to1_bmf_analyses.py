#!/usr/bin/env python3
"""One-to-one scDiffEq analysis map using the locked clone BMF model.

This script does not reuse the rejected gene-influence panels.  It builds a
separate scDiffEq-style analysis suite and records exactly how each manuscript
analysis is translated to the CellCompass/BMF setting.

Original scDiffEq analysis repo used for the map:
    https://github.com/scDiffEq/scdiffeq-analyses
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
import seaborn as sns
import torch
import scipy.sparse as sp
import scipy.stats
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compose_clone_scdiffeq_style_biology as base  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "downstream/cellcompass_clone_case/scdiffeq_1to1_bmf"
FIG_DIR = ROOT / "figures/cellcompass_clone_case/scdiffeq_1to1_bmf"
ADATA_PATH = Path("/Users/zhenyizhang/.cellcompass/algorithm_benchmarks/datasets/weinreb_rawrebuild_full_k10mindiff1/data.h5ad")
BMF_TABLE = ROOT / "downstream/clone_fate_spring_final/clone_cuvgf_bmf_final_day2_fate_probabilities.csv"
METRIC_SUMMARY = ROOT / "downstream/clone_fate_spring_final/final_metric_summary.csv"
BENCHMARK_SUMMARY = ROOT / "downstream/weinreb_supp_W1_metric_trial/unique_test_all_fates_summary.csv"

PRIMARY_FATES = ["Undifferentiated", "Neutrophil", "Monocyte", "Baso", "Mast", "Meg", "Erythroid"]
FATE_GROUPS = {
    "Neutrophil": ["Neutrophil"],
    "Monocyte": ["Monocyte"],
    "Baso/Mast": ["Baso", "Mast"],
    "Meg/Ery": ["Meg", "Erythroid"],
    "Undifferentiated": ["Undifferentiated"],
}
SCDEQ_PERTURB_GENES = ["Cebpe", "Dach1", "Lmo4", "Mxd1"]
LINEAGE_GENES = [
    "Spi1",
    "Gfi1",
    "Irf8",
    "Klf4",
    "Cebpe",
    "Gata1",
    "Klf1",
    "Pf4",
    "Itga2b",
    "Cpa3",
    "Meis1",
    "Mecom",
]


def setup_style() -> None:
    base.setup_style()
    mpl.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.titleweight": "bold",
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
        }
    )


def savefig(fig: plt.Figure, stem: str) -> str:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIG_DIR / f"{stem}.{ext}", dpi=360 if ext == "png" else None)
    plt.close(fig)
    return str(FIG_DIR / f"{stem}.png")


def normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(np.asarray(x, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    s = x.sum(axis=1, keepdims=True)
    return np.divide(x, s, out=np.zeros_like(x), where=s > 0)


def pearson_small_n(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r for three-time-point observed trend checks."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return float("nan")
    x = x[mask] - np.nanmean(x[mask])
    y = y[mask] - np.nanmean(y[mask])
    denom = float(np.sqrt(np.sum(x**2) * np.sum(y**2)))
    if denom <= 1e-12:
        return float("nan")
    return float(np.sum(x * y) / denom)


def fate_names(bmf: pd.DataFrame) -> list[str]:
    return [c.removeprefix("pred_") for c in bmf.columns if c.startswith("pred_")]


def subset_sources(bmf: pd.DataFrame, max_cells: int = 360) -> np.ndarray:
    cand = bmf[bmf["eligible"].astype(bool) & bmf["source_label"].eq("Undifferentiated")].copy()
    truth_cols = [c for c in cand.columns if c.startswith("truth_")]
    pred_cols = [c for c in cand.columns if c.startswith("pred_")]
    truth = normalize_rows(cand[truth_cols].to_numpy(float))
    pred = normalize_rows(cand[pred_cols].to_numpy(float))
    truth_ent = -(np.clip(truth, 1e-12, 1) * np.log(np.clip(truth, 1e-12, 1))).sum(axis=1)
    pred_ent = -(np.clip(pred, 1e-12, 1) * np.log(np.clip(pred, 1e-12, 1))).sum(axis=1)
    cand["selection_score"] = cand["truth_reliability"].fillna(0) * (truth_ent + 0.5 * pred_ent)
    return cand.sort_values("selection_score", ascending=False)["source_index"].head(max_cells).to_numpy(int)


def project_latent_to_spring(query: np.ndarray, ref_latent: np.ndarray, ref_spring: np.ndarray, k: int = 25) -> np.ndarray:
    tree = cKDTree(np.asarray(ref_latent, dtype=np.float32))
    d, idx = tree.query(np.asarray(query, dtype=np.float32), k=min(k, len(ref_latent)))
    if idx.ndim == 1:
        d = d[:, None]
        idx = idx[:, None]
    w = 1.0 / (np.asarray(d, dtype=float) + 1e-4)
    w = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
    return np.einsum("ij,ijk->ik", w, ref_spring[np.asarray(idx, dtype=int)])


def compute_velocity_graph_like_scdiffeq(
    latent: np.ndarray,
    velocity: np.ndarray,
    *,
    n_pcs: int = 10,
    n_neighbors: int = 30,
) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    x = np.asarray(latent[:, :n_pcs], dtype=np.float32)
    v = np.asarray(velocity[:, :n_pcs], dtype=np.float32)
    n_neighbors = int(max(5, min(n_neighbors, len(x) - 1)))
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, n_jobs=-1)
    nn.fit(x)
    _, nbr = nn.kneighbors(x)
    nbr = nbr[:, 1:]
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for i in range(len(x)):
        # scDiffEq's ComputeCosines uses direct+second-order neighbors and
        # mean-centers local displacements before computing cosine alignment.
        iter_idx = np.unique(np.append(nbr[i], nbr[nbr[i]].reshape(-1))).astype(int)
        if len(iter_idx) == 0 or np.all(v[i] == 0):
            continue
        dx = x[iter_idx] - x[i]
        dx = dx - dx.mean(axis=1, keepdims=True)
        vnorm = np.linalg.norm(v[i])
        denom = np.linalg.norm(dx, axis=1) * vnorm
        cos = np.divide(dx @ v[i], denom, out=np.zeros_like(denom), where=denom > 1e-12)
        rows.extend([i] * len(iter_idx))
        cols.extend(iter_idx.tolist())
        vals.extend(np.nan_to_num(cos, nan=0.0).tolist())
    shape = (len(x), len(x))
    vals = np.asarray(vals, dtype=float)
    return (
        sp.csr_matrix((np.clip(vals, 0, 1), (rows, cols)), shape=shape),
        sp.csr_matrix((np.clip(vals, -1, 0), (rows, cols)), shape=shape),
    )


def transition_matrix_like_scdiffeq(
    graph: sp.csr_matrix,
    graph_neg: sp.csr_matrix,
    *,
    scale: float = 10.0,
) -> sp.csr_matrix:
    graph = graph.tolil(copy=True)
    confidence = graph.tocsr().max(axis=1).toarray().ravel()
    graph.setdiag(np.clip(np.percentile(confidence, 98) - confidence, 0, 1))
    pos = graph.tocsr()
    pos.data = np.expm1(pos.data * scale)
    neg = graph_neg.copy()
    neg.data = np.expm1(-neg.data * scale)
    trans = (pos - neg).tocsr()
    trans.setdiag(0)
    trans.eliminate_zeros()
    row_abs = np.asarray(np.abs(trans).sum(axis=1)).ravel()
    inv = np.divide(1.0, row_abs, out=np.zeros_like(row_abs), where=row_abs > 1e-12)
    return trans.multiply(inv[:, None]).tocsr()


def velocity_embedding_like_scdiffeq(x_emb: np.ndarray, trans: sp.csr_matrix) -> np.ndarray:
    x_emb = np.asarray(x_emb, dtype=float)
    v_emb = np.zeros_like(x_emb, dtype=float)
    for i in range(trans.shape[0]):
        start, end = trans.indptr[i], trans.indptr[i + 1]
        idx = trans.indices[start:end]
        prob = trans.data[start:end]
        if len(idx) == 0:
            continue
        dx = x_emb[idx] - x_emb[i]
        norm = np.linalg.norm(dx, axis=1)
        dx = np.divide(dx, norm[:, None], out=np.zeros_like(dx), where=norm[:, None] > 1e-12)
        v_emb[i] = prob.dot(dx) - prob.mean() * dx.sum(axis=0)
    return v_emb


def grid_velocity_like_scdiffeq(
    x_emb: np.ndarray,
    v_emb: np.ndarray,
    *,
    density: float = 1.0,
    smooth: float = 0.5,
    n_neighbors: int | None = None,
    min_mass: float = 1.0,
    cutoff_percentile: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(x_emb).all(axis=1) & np.isfinite(v_emb).all(axis=1)
    x_emb = np.asarray(x_emb[finite], dtype=float)
    v_emb = np.asarray(v_emb[finite], dtype=float)
    grids = []
    for dim in range(2):
        lo = float(np.min(x_emb[:, dim]))
        hi = float(np.max(x_emb[:, dim]))
        pad = 0.01 * abs(hi - lo)
        grids.append(np.linspace(lo - pad, hi + pad, int(50 * density)))
    x_grid = np.vstack([g.ravel() for g in np.meshgrid(*grids)]).T
    if n_neighbors is None:
        n_neighbors = int(x_emb.shape[0] / 50)
    n_neighbors = int(max(5, min(n_neighbors, x_emb.shape[0])))
    nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
    nn.fit(x_emb)
    dists, nbr = nn.kneighbors(x_grid)
    step = float(np.mean([g[1] - g[0] for g in grids]))
    weight = scipy.stats.norm.pdf(dists, scale=max(step * smooth, 1e-6))
    p_mass = weight.sum(axis=1)
    v_grid = (v_emb[nbr] * weight[:, :, None]).sum(axis=1)
    v_grid = np.divide(v_grid, np.maximum(1, p_mass)[:, None])
    ns = int(np.sqrt(len(v_grid)))
    stream = v_grid.T.reshape(2, ns, ns)
    mass = np.sqrt((stream**2).sum(axis=0))
    min_mass_value = np.clip(10 ** (min_mass - 6), None, float(np.nanmax(mass)) * 0.9)
    cutoff = mass < min_mass_value
    length = np.sum(np.mean(np.abs(v_emb[nbr]), axis=1), axis=1).T.reshape(ns, ns)
    cutoff |= length < np.percentile(length, cutoff_percentile)
    stream[0][cutoff] = np.nan
    stream[1][cutoff] = np.nan
    return np.stack([np.unique(x_grid[:, 0]), np.unique(x_grid[:, 1])]), stream


def project_snapshot(
    query: np.ndarray,
    model_time: float,
    obs: pd.DataFrame,
    latent: np.ndarray,
    source_idx: np.ndarray,
    parent: np.ndarray,
) -> np.ndarray:
    obs_t = int(round(float(model_time)))
    obs_t = max(0, min(2, obs_t))
    if obs_t == 0:
        return obs.iloc[source_idx[parent]][["spring_x", "spring_y"]].to_numpy(float)
    mask = obs["t"].eq(obs_t).to_numpy()
    return project_latent_to_spring(
        query,
        latent[mask],
        obs.loc[mask, ["spring_x", "spring_y"]].to_numpy(float),
        k=25,
    )


def style_spring(ax: plt.Axes, obs: pd.DataFrame) -> None:
    base.style_embedding_axis(ax, obs)


def draw_observed_background(ax: plt.Axes, obs: pd.DataFrame, *, by_fate: bool = True, alpha: float = 0.18) -> None:
    bg = obs if len(obs) <= 38000 else obs.sample(n=38000, random_state=11)
    if by_fate:
        for fate, color in base.FATE_PALETTE.items():
            d = bg[bg["cell_type"].eq(fate)]
            if d.empty:
                continue
            ax.scatter(d["spring_x"], d["spring_y"], s=1.0, c=color, alpha=alpha, linewidth=0, zorder=0)
    else:
        ax.scatter(bg["spring_x"], bg["spring_y"], s=0.9, c="#CBD5E1", alpha=0.20, linewidth=0, zorder=0)
    style_spring(ax, obs)


def add_fate_labels(ax: plt.Axes, obs: pd.DataFrame, fates: list[str] = PRIMARY_FATES) -> None:
    terminal = obs[obs["t"].eq(2)]
    offsets = {
        "Undifferentiated": (-20, 20),
        "Neutrophil": (0, -35),
        "Monocyte": (20, 20),
        "Baso": (-25, -15),
        "Mast": (-15, -25),
        "Meg": (-25, 20),
        "Erythroid": (20, -15),
    }
    for fate in fates:
        d = terminal[terminal["cell_type"].eq(fate)]
        if len(d) < 20:
            continue
        dx, dy = offsets.get(fate, (0, 0))
        ax.text(
            float(d["spring_x"].median()) + dx,
            float(d["spring_y"].median()) + dy,
            fate.replace("Undifferentiated", "Undiff."),
            color=base.FATE_PALETTE.get(fate, "#111827"),
            fontsize=7,
            weight="bold",
            ha="center",
            va="center",
            path_effects=[pe.Stroke(linewidth=2.0, foreground="white", alpha=0.95), pe.Normal()],
            zorder=50,
        )


def energy_distance(x: np.ndarray, y: np.ndarray, max_n: int = 1200, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    if len(x) > max_n:
        x = x[rng.choice(len(x), max_n, replace=False)]
    if len(y) > max_n:
        y = y[rng.choice(len(y), max_n, replace=False)]
    from scipy.spatial.distance import cdist

    xy = cdist(x, y).mean()
    xx = cdist(x, x).mean()
    yy = cdist(y, y).mean()
    return float(2 * xy - xx - yy)


def decoded_gene_score(adata: ad.AnnData, latent_block: np.ndarray, gene: str) -> np.ndarray | None:
    if gene not in adata.var_names:
        return None
    idx = int(np.where(np.asarray(adata.var_names) == gene)[0][0])
    pcs = np.asarray(adata.varm["PCs"], dtype=np.float32)
    return np.asarray(latent_block, dtype=np.float32) @ pcs[idx]


def branch_velocity_and_growth(model, latent_block: np.ndarray, fates: list[str], t_value: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = torch.as_tensor(latent_block.astype(np.float32), dtype=torch.float32)
    n = x.shape[0]
    n_fates = len(fates)
    with torch.no_grad():
        t = torch.full((n, 1), float(t_value), dtype=x.dtype)
        gate = torch.softmax(model.gate_net(torch.cat([x, t], dim=1)), dim=1)
        eye = torch.eye(n_fates, dtype=x.dtype)
        v_all = []
        g_all = []
        for j in range(n_fates):
            b = eye[j].repeat(n, 1)
            inp = torch.cat([x, t, b], dim=1)
            v_all.append(model.branch_velocity_net(inp).cpu().numpy())
            g_all.append(model.branch_growth_net(inp).cpu().numpy().reshape(-1))
    v = np.stack(v_all, axis=1)
    g = np.stack(g_all, axis=1)
    return gate.cpu().numpy(), v, g


def weighted_velocity_growth(model, latent_block: np.ndarray, fates: list[str], t_value: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gate, v, g = branch_velocity_and_growth(model, latent_block, fates, t_value=t_value)
    vw = np.einsum("nf,nfd->nd", gate, v)
    gw = np.einsum("nf,nf->n", gate, g)
    return gate, vw, gw


def fate_fraction_from_prob(prob: np.ndarray, fates: list[str], fate_group: list[str]) -> np.ndarray:
    idx = [fates.index(f) for f in fate_group if f in fates]
    if not idx:
        return np.zeros(prob.shape[0], dtype=float)
    return prob[:, idx].sum(axis=1)


def run_baseline_sim(adata: ad.AnnData, obs: pd.DataFrame, bmf: pd.DataFrame, fates: list[str], max_cells: int = 360):
    source_idx = subset_sources(bmf, max_cells=max_cells)
    latent = np.asarray(adata.obsm["X_latent"], dtype=np.float32)
    model, resolved = base.load_bmf_model(latent.shape[1], len(fates))
    sim = base.simulate_bmf(model, resolved, latent[source_idx], fates, snapshot_times=np.linspace(0.0, 2.0, 9))
    return model, resolved, source_idx, sim


def plot_fig2a_observed_atlas(obs: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.7))
    draw_observed_background(axes[0], obs, by_fate=True, alpha=0.32)
    add_fate_labels(axes[0], obs)
    axes[0].set_title("observed LARRY cell states", loc="left")
    cmap = {0: "#7EA6D9", 1: "#69C6CE", 2: "#F2A65A"}
    labels = {0: "day 2", 1: "day 4", 2: "day 6"}
    for t, color in cmap.items():
        d = obs[obs["t"].eq(t)]
        d = d.sample(n=min(5500, len(d)), random_state=20 + t)
        axes[1].scatter(d["spring_x"], d["spring_y"], s=1.2, c=color, alpha=0.32, linewidth=0, label=labels[t])
    style_spring(axes[1], obs)
    axes[1].legend(frameon=False, fontsize=7, loc="lower left")
    axes[1].set_title("observed sampling times", loc="left")
    fig.suptitle("Figure 2A analogue: observed clone time-course manifold", x=0.02, y=0.99, ha="left", weight="bold")
    return savefig(fig, "fig2a_observed_larry_spring_atlas")


def plot_fig2b_velocity_stream(adata: ad.AnnData, obs: pd.DataFrame, model, fates: list[str]) -> str:
    latent = np.asarray(adata.obsm["X_latent"], dtype=np.float32)
    mask = obs["t"].eq(0).to_numpy()
    idx = np.where(mask)[0]
    gate, v, growth = weighted_velocity_growth(model, latent[idx], fates, t_value=0.0)
    xy0 = obs.iloc[idx][["spring_x", "spring_y"]].to_numpy(float)

    graph, graph_neg = compute_velocity_graph_like_scdiffeq(latent[idx], v, n_pcs=10, n_neighbors=30)
    trans = transition_matrix_like_scdiffeq(graph, graph_neg, scale=10.0)
    v_emb = velocity_embedding_like_scdiffeq(xy0, trans)
    x_grid, v_grid = grid_velocity_like_scdiffeq(xy0, v_emb, density=1.0, smooth=0.5, cutoff_percentile=0.1)

    growth_rows = []
    growth_by_t = {}
    for obs_t, day in [(0, 2), (1, 4), (2, 6)]:
        tidx = np.where(obs["t"].eq(obs_t).to_numpy())[0]
        _, _, gt = weighted_velocity_growth(model, latent[tidx], fates, t_value=float(obs_t))
        growth_by_t[obs_t] = (tidx, gt)
        for cell_idx, value in zip(tidx, gt):
            growth_rows.append(
                {
                    "cell_index": int(cell_idx),
                    "time_point_processed": int(obs_t),
                    "day": int(day),
                    "cell_type": str(obs.iloc[cell_idx]["cell_type"]),
                    "spring_x": float(obs.iloc[cell_idx]["spring_x"]),
                    "spring_y": float(obs.iloc[cell_idx]["spring_y"]),
                    "bmf_growth_dlogmass_dt": float(value),
                }
            )
    pd.DataFrame(growth_rows).to_csv(OUT_DIR / "fig2b_fig4ef_bmf_growth_on_observed_timepoints.csv", index=False)
    all_growth = np.asarray([r["bmf_growth_dlogmass_dt"] for r in growth_rows], dtype=float)
    vmin, vmax = np.nanpercentile(all_growth, [2, 98])

    fig = plt.figure(figsize=(8.8, 4.1))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.05], wspace=0.10)
    ax_stream = fig.add_subplot(gs[0, 0])
    ax_growth = fig.add_subplot(gs[0, 1])

    draw_observed_background(ax_stream, obs, by_fate=True, alpha=0.12)
    add_fate_labels(ax_stream, obs)
    ax_stream.streamplot(
        x_grid[0],
        x_grid[1],
        v_grid[0],
        v_grid[1],
        color="#111827",
        linewidth=0.55,
        density=2.5,
        arrowsize=0.8,
        maxlength=4.0,
        integration_direction="both",
        zorder=20,
    )
    ax_stream.set_title("BMF velocity stream", loc="left")

    draw_observed_background(ax_growth, obs, by_fate=True, alpha=0.055)
    add_fate_labels(ax_growth, obs)
    markers = {0: "o", 1: "^", 2: "s"}
    sizes = {0: 3.2, 1: 3.4, 2: 3.6}
    handles = []
    last_sc = None
    for obs_t, day in [(0, 2), (1, 4), (2, 6)]:
        tidx, gt = growth_by_t[obs_t]
        d = obs.iloc[tidx]
        order = np.argsort(gt)
        last_sc = ax_growth.scatter(
            d["spring_x"].to_numpy(float)[order],
            d["spring_y"].to_numpy(float)[order],
            c=gt[order],
            s=sizes[obs_t],
            marker=markers[obs_t],
            cmap="magma",
            vmin=vmin,
            vmax=vmax,
            alpha=0.62 if obs_t < 2 else 0.80,
            linewidth=0,
            rasterized=True,
            zorder=12,
        )
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker=markers[obs_t],
                color="none",
                markerfacecolor="#111827",
                markeredgecolor="none",
                markersize=5,
                label=f"day {day}",
            )
        )
    ax_growth.set_title("BMF growth on observed cells across time", loc="left")
    ax_growth.legend(handles=handles, frameon=False, loc="lower left", fontsize=7)
    cb = plt.colorbar(last_sc, ax=ax_growth, fraction=0.046, pad=0.02)
    cb.set_label("BMF d log mass / dt", fontsize=7)
    fig.suptitle("Figure 2B / 4E-F analogue: BMF velocity and growth on SPRING", x=0.02, y=0.99, ha="left", weight="bold")
    return savefig(fig, "fig2b_fig4ef_bmf_velocity_growth_spring")


def plot_fig2def_time_recovery(adata: ad.AnnData, obs: pd.DataFrame, source_idx: np.ndarray, sim: dict) -> tuple[str, str]:
    latent = np.asarray(adata.obsm["X_latent"], dtype=np.float32)
    rows = []
    for mt, obs_t in [(0.0, 0), (1.0, 1), (2.0, 2)]:
        gen = sim["snapshots"][mt]
        w = sim["weights"][mt]
        if len(gen) > 4000:
            rng = np.random.default_rng(int(100 + obs_t))
            p = w / np.maximum(w.sum(), 1e-12)
            take = rng.choice(len(gen), size=4000, replace=False, p=p)
            gen = gen[take]
        target = latent[obs["t"].eq(obs_t).to_numpy()]
        rows.append(
            {
                "model_time": mt,
                "observed_time_point_processed": obs_t,
                "biological_day": {0: 2, 1: 4, 2: 6}[obs_t],
                "energy_distance_latent": energy_distance(gen, target, max_n=1200, seed=obs_t),
                "n_generated_particles": int(len(sim["snapshots"][mt])),
                "n_observed_cells": int((obs["t"].eq(obs_t)).sum()),
            }
        )
    out = pd.DataFrame(rows)
    csv_path = OUT_DIR / "fig2def_bmf_time_recovery_distribution_distance.csv"
    out.to_csv(csv_path, index=False)
    fig, ax = plt.subplots(figsize=(3.9, 2.75))
    ax.plot(out["biological_day"], out["energy_distance_latent"], color="#6D28D9", lw=2.5, marker="o", ms=5)
    ax.set_xlabel("observed day")
    ax.set_ylabel("latent distribution error")
    ax.set_title("time-point recovery", loc="left")
    ax.grid(axis="y", color="#E5E7EB")
    sns.despine(ax=ax)
    fig.suptitle("Figure 2E-F / S2 analogue: generated particles match observed time points", x=0.02, y=0.99, ha="left", weight="bold")
    return savefig(fig, "fig2def_bmf_time_recovery_distribution_distance"), str(csv_path)


def run_rollout_with_delta(model, resolved, latent0: np.ndarray, delta: np.ndarray, fates: list[str], obs: pd.DataFrame, latent: np.ndarray) -> np.ndarray:
    sim = base.simulate_bmf(model, resolved, latent0 + delta[None, :], fates, snapshot_times=np.asarray([0.0, 1.0, 2.0]))
    endpoint = base.terminal_fate_probabilities(sim["final_latent"], obs, latent, fates, k=30)
    return base.aggregate_terminal_fates(sim, endpoint, latent0.shape[0], len(fates))


def plot_fig3ab_cd_perturb_scan(adata: ad.AnnData, obs: pd.DataFrame, model, resolved, source_idx: np.ndarray, fates: list[str]) -> tuple[str, str]:
    latent = np.asarray(adata.obsm["X_latent"], dtype=np.float32)
    latent0 = latent[source_idx]
    found = [g for g in SCDEQ_PERTURB_GENES if g in adata.var_names]
    delta_unit, used = base.program_delta(adata, SCDEQ_PERTURB_GENES, 1.0)
    z_values = np.asarray([-10, -7.5, -5, -2.5, 0, 2.5, 5, 7.5, 10], dtype=float)
    rows = []
    for z in z_values:
        pred = run_rollout_with_delta(model, resolved, latent0, z * delta_unit, fates, obs, latent)
        for fate in fates:
            rows.append({"z": float(z), "fate": fate, "fraction": float(pred[:, fates.index(fate)].mean()), "genes_used": ",".join(used)})
    out = pd.DataFrame(rows)
    csv_path = OUT_DIR / "fig3ab_cd_bmf_multigene_perturb_zscan.csv"
    out.to_csv(csv_path, index=False)
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2), gridspec_kw={"width_ratios": [1.05, 1.0]})
    for fate in ["Neutrophil", "Monocyte", "Baso", "Mast", "Erythroid", "Meg"]:
        d = out[out["fate"].eq(fate)]
        if d.empty:
            continue
        axes[0].plot(d["z"], d["fraction"], lw=2.0, marker="o", ms=3.4, color=base.FATE_PALETTE.get(fate, "#64748B"), label=fate)
    axes[0].axvline(0, color="#111827", lw=1, alpha=0.4)
    axes[0].set_xlabel("RNA perturbation strength (z)")
    axes[0].set_ylabel("terminal fate fraction")
    axes[0].set_title("multi-gene z scan", loc="left")
    axes[0].legend(frameon=False, fontsize=6.5, ncol=2)
    axes[0].grid(axis="y", color="#E5E7EB")
    display_fates = [f for f in PRIMARY_FATES if f in fates]
    other_fates = [f for f in fates if f not in display_fates]
    for z, label, x in [(0.0, "baseline", 0), (10.0, "+10z", 1)]:
        d = out[out["z"].eq(z)].set_index("fate")["fraction"]
        bottom = 0
        for fate in display_fates:
            val = float(d.get(fate, 0.0))
            if val <= 0:
                continue
            axes[1].bar(x, val, bottom=bottom, color=base.FATE_PALETTE.get(fate, "#94A3B8"), edgecolor="white", linewidth=0.45)
            bottom += val
        other = float(d.reindex(other_fates).fillna(0).sum()) if other_fates else 0.0
        if other > 0:
            axes[1].bar(x, other, bottom=bottom, color="#CBD5E1", edgecolor="white", linewidth=0.45)
            bottom += other
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["baseline", "+10z"])
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("terminal fate fraction")
    axes[1].set_title("Figure 3A-B endpoint comparison", loc="left")
    axes[1].text(0.02, -0.22, "genes: " + ", ".join(used), transform=axes[1].transAxes, fontsize=7, color="#334155")
    sns.despine(fig=fig)
    fig.suptitle("Figure 3A-D analogue: BMF in silico perturbation of the scDiffEq gene set", x=0.02, y=0.99, ha="left", weight="bold")
    return savefig(fig, "fig3abcd_bmf_multigene_perturb_zscan"), str(csv_path)


def branch_gate_screen(adata: ad.AnnData, model, latent0: np.ndarray, fates: list[str], z: float = 5.0) -> pd.DataFrame:
    pcs = np.asarray(adata.varm["PCs"], dtype=np.float32)
    genes = np.asarray(adata.var_names, dtype=str)
    with torch.no_grad():
        x = torch.as_tensor(latent0.astype(np.float32), dtype=torch.float32)
        t0 = torch.zeros((x.shape[0], 1), dtype=x.dtype)
        base_gate = torch.softmax(model.gate_net(torch.cat([x, t0], dim=1)), dim=1).cpu().numpy()
    rows = []
    batch = 128
    for start in range(0, len(genes), batch):
        stop = min(start + batch, len(genes))
        deltas = z * pcs[start:stop]
        x_batch = latent0[:, None, :] + deltas[None, :, :]
        flat = torch.as_tensor(x_batch.reshape(-1, latent0.shape[1]).astype(np.float32), dtype=torch.float32)
        with torch.no_grad():
            gate = torch.softmax(model.gate_net(torch.cat([flat, torch.zeros((flat.shape[0], 1), dtype=flat.dtype)], dim=1)), dim=1).cpu().numpy()
        gate = gate.reshape(latent0.shape[0], stop - start, len(fates)).mean(axis=0)
        delta = gate - base_gate.mean(axis=0, keepdims=True)
        for local, gene in enumerate(genes[start:stop]):
            row = {"gene": gene, "z": float(z)}
            for j, fate in enumerate(fates):
                row[f"delta_gate_{fate}"] = float(delta[local, j])
                row[f"gate_{fate}"] = float(gate[local, j])
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "fig3efg_kl_bmf_all_gene_branch_gate_screen.csv", index=False)
    return out


def plot_fig3efg_kl_gene_screen(screen: pd.DataFrame, fates: list[str]) -> str:
    groups = {
        "GMP-like": ["Neutrophil", "Monocyte"],
        "MEP-like": ["Meg", "Erythroid"],
        "Baso/Mast": ["Baso", "Mast"],
    }
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.6), sharey=True)
    for ax, (name, group) in zip(axes, groups.items()):
        cols = [f"delta_gate_{f}" for f in group if f"delta_gate_{f}" in screen.columns]
        d = screen[["gene"] + cols].copy()
        d["score"] = d[cols].sum(axis=1)
        d = d.reindex(d["score"].abs().sort_values(ascending=False).head(16).index).sort_values("score")
        colors = ["#0F766E" if v > 0 else "#E76F51" for v in d["score"]]
        ax.barh(d["gene"], d["score"], color=colors, alpha=0.88)
        ax.axvline(0, color="#111827", lw=0.8)
        ax.set_title(name, loc="left")
        ax.set_xlabel("+5z change in branch probability")
        ax.grid(axis="x", color="#E5E7EB")
    sns.despine(fig=fig)
    fig.suptitle("Figure 3E-G/K-L analogue: all-gene BMF branch-gate screen", x=0.02, y=0.99, ha="left", weight="bold")
    return savefig(fig, "fig3efg_kl_bmf_all_gene_branch_gate_screen")


def plot_fig3j_spi1_scan(adata: ad.AnnData, obs: pd.DataFrame, model, resolved, source_idx: np.ndarray, fates: list[str]) -> tuple[str, str]:
    latent = np.asarray(adata.obsm["X_latent"], dtype=np.float32)
    latent0 = latent[source_idx]
    gene = "Spi1" if "Spi1" in adata.var_names else str(adata.var_names[0])
    delta_unit, used = base.program_delta(adata, [gene], 1.0)
    z_values = np.asarray([-10, -5, -2.5, 0, 2.5, 5, 10], dtype=float)
    rows = []
    for z in z_values:
        pred = run_rollout_with_delta(model, resolved, latent0, z * delta_unit, fates, obs, latent)
        for fate in ["Neutrophil", "Monocyte", "Baso"]:
            if fate in fates:
                rows.append({"gene": gene, "z": float(z), "fate": fate, "fraction": float(pred[:, fates.index(fate)].mean())})
    out = pd.DataFrame(rows)
    csv_path = OUT_DIR / "fig3j_bmf_spi1_zscan.csv"
    out.to_csv(csv_path, index=False)
    fig, ax = plt.subplots(figsize=(4.4, 3.1))
    for fate in ["Neutrophil", "Monocyte", "Baso"]:
        d = out[out["fate"].eq(fate)]
        if d.empty:
            continue
        ax.plot(d["z"], d["fraction"], marker="o", ms=3.6, lw=2.2, color=base.FATE_PALETTE.get(fate, "#64748B"), label=fate)
    ax.axvline(0, color="#111827", lw=1, alpha=0.35)
    ax.set_xlabel(f"{gene} perturbation strength (z)")
    ax.set_ylabel("terminal fate fraction")
    ax.set_title("single-gene z scan", loc="left")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(axis="y", color="#E5E7EB")
    sns.despine(ax=ax)
    fig.suptitle("Figure 3J analogue: BMF single-gene perturbation scan", x=0.02, y=0.99, ha="left", weight="bold")
    return savefig(fig, "fig3j_bmf_spi1_zscan"), str(csv_path)


def compute_fig4d_g_stats(adata: ad.AnnData, obs: pd.DataFrame, model, fates: list[str]) -> pd.DataFrame:
    latent = np.asarray(adata.obsm["X_latent"], dtype=np.float32)
    idx = np.where(obs["t"].eq(0).to_numpy())[0]
    if len(idx) > 1800:
        idx = np.random.default_rng(41).choice(idx, size=1800, replace=False)
    gate, v, g = branch_velocity_and_growth(model, latent[idx], fates, t_value=0.0)
    rows = []
    for j, fate in enumerate(fates):
        rows.append(
            pd.DataFrame(
                {
                    "fate": fate,
                    "branch_probability": gate[:, j],
                    "velocity_norm": np.linalg.norm(v[:, j, :], axis=1),
                    "growth_rate": g[:, j],
                }
            )
        )
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(OUT_DIR / "fig4d_bmf_branch_velocity_growth_statistics.csv", index=False)
    return out


def plot_fig4d_stats(stats: pd.DataFrame) -> str:
    show = stats[stats["fate"].isin(PRIMARY_FATES)]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4), sharex=True)
    sns.boxplot(data=show, x="fate", y="velocity_norm", ax=axes[0], color="#93C5FD", fliersize=0.4, linewidth=0.7)
    sns.boxplot(data=show, x="fate", y="growth_rate", ax=axes[1], color="#FDBA74", fliersize=0.4, linewidth=0.7)
    for ax in axes:
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", color="#E5E7EB")
    axes[0].set_title("branch velocity magnitude", loc="left")
    axes[1].set_title("branch growth rate", loc="left")
    sns.despine(fig=fig)
    fig.suptitle("Figure 4D analogue: BMF branch movement and branch growth", x=0.02, y=0.99, ha="left", weight="bold")
    return savefig(fig, "fig4d_bmf_branch_velocity_growth_statistics")


def compute_fig4g_gene_velocity_corr(adata: ad.AnnData, obs: pd.DataFrame, model, fates: list[str]) -> pd.DataFrame:
    latent = np.asarray(adata.obsm["X_latent"], dtype=np.float32)
    pcs = np.asarray(adata.varm["PCs"], dtype=np.float32)
    idx = np.where(obs["t"].eq(0).to_numpy())[0]
    if len(idx) > 2200:
        idx = np.random.default_rng(42).choice(idx, size=2200, replace=False)
    x0 = latent[idx]
    gate, v, g = branch_velocity_and_growth(model, x0, fates, t_value=0.0)
    expr = x0 @ pcs.T
    rows = []
    for j, fate in enumerate(fates):
        dzdt = v[:, j, :] @ pcs.T
        for gene_i, gene in enumerate(np.asarray(adata.var_names, dtype=str)):
            x = expr[:, gene_i]
            vx = dzdt[:, gene_i]
            rows.append(
                {
                    "gene": gene,
                    "fate": fate,
                    "expression_velocity_corr": base.pearson(x, vx),
                    "expression_growth_corr": base.pearson(x, g[:, j]),
                    "mean_branch_probability": float(gate[:, j].mean()),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "fig4g_bmf_gene_expression_velocity_growth_correlations.csv", index=False)
    return out


def plot_fig4g_gene_corr(corr: pd.DataFrame) -> str:
    focus = corr[corr["fate"].isin(["Neutrophil", "Monocyte", "Meg", "Erythroid", "Baso", "Mast"])].copy()
    readable = []
    for gene in LINEAGE_GENES + sum(base.PROGRAMS.values(), []):
        if gene in set(focus["gene"]):
            readable.append(gene)
    # Keep the full all-gene CSV, but the figure uses interpretable hematopoietic
    # genes so this panel is not dominated by unannotated Riken/Olfr genes.
    top_genes = list(dict.fromkeys(readable))[:24]
    if len(top_genes) < 10:
        focus["score"] = focus["expression_velocity_corr"].abs() + focus["expression_growth_corr"].abs()
        top_genes += [g for g in focus.groupby("gene")["score"].max().sort_values(ascending=False).index.tolist() if g not in top_genes][: 16 - len(top_genes)]
    d = focus[focus["gene"].isin(top_genes)]
    mat_v = d.pivot_table(index="gene", columns="fate", values="expression_velocity_corr", aggfunc="mean").loc[top_genes]
    mat_g = d.pivot_table(index="gene", columns="fate", values="expression_growth_corr", aggfunc="mean").loc[top_genes]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 5.7), sharey=True)
    vals = np.concatenate([mat_v.to_numpy(float).ravel(), mat_g.to_numpy(float).ravel()])
    lim = max(0.2, float(np.nanpercentile(np.abs(vals), 97)))
    sns.heatmap(mat_v, ax=axes[0], cmap="vlag", center=0, vmin=-lim, vmax=lim, cbar_kws={"label": "corr"})
    sns.heatmap(mat_g, ax=axes[1], cmap="vlag", center=0, vmin=-lim, vmax=lim, cbar_kws={"label": "corr"})
    axes[0].set_title("expression vs RNA velocity")
    axes[1].set_title("expression vs branch growth")
    for ax in axes:
        ax.set_xlabel("")
        ax.set_ylabel("")
    fig.suptitle("Figure 4G analogue: genes coupled to BMF velocity or growth", x=0.02, y=0.99, ha="left", weight="bold")
    return savefig(fig, "fig4g_bmf_gene_velocity_growth_correlations")


def weighted_mean_std(values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if int(mask.sum()) == 0:
        return float("nan"), float("nan")
    values = values[mask]
    weights = weights[mask]
    weights = weights / max(float(weights.sum()), 1e-12)
    mean = float(np.sum(weights * values))
    var = float(np.sum(weights * (values - mean) ** 2))
    return mean, float(np.sqrt(max(var, 0.0)))


def observed_clone_terminal_fates(
    obs: pd.DataFrame,
    fates: list[str],
    *,
    min_terminal_cells: int = 2,
    min_purity: float = 0.55,
) -> pd.DataFrame:
    terminal = obs[(obs["t"].eq(2)) & (obs["clone_str"].ne("")) & (obs["cell_type"].isin(fates))]
    rows = []
    for clone, d in terminal.groupby("clone_str", observed=False):
        counts = d["cell_type"].value_counts()
        total = int(counts.sum())
        if total < min_terminal_cells:
            continue
        fate = str(counts.index[0])
        purity = float(counts.iloc[0] / total)
        if purity < min_purity:
            continue
        rows.append(
            {
                "clone_str": clone,
                "observed_terminal_fate": fate,
                "terminal_cells": total,
                "terminal_purity": purity,
            }
        )
    return pd.DataFrame(rows)


def observed_gene_trends_by_clone_terminal_fate(
    adata: ad.AnnData,
    obs: pd.DataFrame,
    genes: list[str],
    fates: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    clone_fates = observed_clone_terminal_fates(obs, fates)
    if clone_fates.empty:
        return pd.DataFrame(), clone_fates
    fate_by_clone = dict(zip(clone_fates["clone_str"], clone_fates["observed_terminal_fate"]))
    use = obs["clone_str"].isin(fate_by_clone).to_numpy()
    obs_latent = np.asarray(adata.obsm["X_latent"], dtype=np.float32)
    rows = []
    for gene in genes:
        score = decoded_gene_score(adata, obs_latent, gene)
        if score is None:
            continue
        tmp = obs.loc[use, ["t", "clone_str"]].copy()
        tmp["terminal_fate"] = tmp["clone_str"].map(fate_by_clone)
        tmp["score"] = score[use]
        for (t, fate), d in tmp.groupby(["t", "terminal_fate"], observed=False):
            if fate not in fates or len(d) < 8:
                continue
            rows.append(
                {
                    "model_time": float(t),
                    "gene": gene,
                    "terminal_fate": str(fate),
                    "observed_expression_mean": float(d["score"].mean()),
                    "observed_expression_std": float(d["score"].std(ddof=0)),
                    "n_observed_cells": int(len(d)),
                    "n_clones": int(d["clone_str"].nunique()),
                }
            )
    return pd.DataFrame(rows), clone_fates


def observed_all_cell_gene_trends(adata: ad.AnnData, obs: pd.DataFrame, genes: list[str]) -> pd.DataFrame:
    obs_latent = np.asarray(adata.obsm["X_latent"], dtype=np.float32)
    rows = []
    for gene in genes:
        score = decoded_gene_score(adata, obs_latent, gene)
        if score is None:
            continue
        tmp = obs[["t"]].copy()
        tmp["score"] = score
        for t, d in tmp.groupby("t", observed=False):
            rows.append(
                {
                    "model_time": float(t),
                    "gene": gene,
                    "observed_expression_mean": float(d["score"].mean()),
                    "observed_expression_std": float(d["score"].std(ddof=0)),
                    "n_observed_cells": int(len(d)),
                }
            )
    return pd.DataFrame(rows)


def bmf_all_particle_gene_trends(adata: ad.AnnData, sim: dict, genes: list[str]) -> pd.DataFrame:
    rows = []
    for t, block in sim["snapshots"].items():
        w = np.asarray(sim["weights"][t], dtype=float)
        for gene in genes:
            score = decoded_gene_score(adata, block, gene)
            if score is None:
                continue
            mean, std = weighted_mean_std(score, w)
            rows.append(
                {
                    "model_time": float(t),
                    "gene": gene,
                    "decoded_expression_mean": mean,
                    "decoded_expression_std": std,
                    "n_generated_particles": int(len(score)),
                    "generated_mass": float(np.nansum(w)),
                }
            )
    return pd.DataFrame(rows)


def plot_fig4hi_jklm_gene_expression(
    adata: ad.AnnData,
    obs: pd.DataFrame,
    source_idx: np.ndarray,
    model,
    resolved: dict,
    fates: list[str],
) -> tuple[str, str, str, str, str, str, str]:
    latent = np.asarray(adata.obsm["X_latent"], dtype=np.float32)
    genes = [g for g in LINEAGE_GENES if g in adata.var_names][:8]
    dense_times = np.linspace(0.0, 2.0, 41)
    sim = base.simulate_bmf(model, resolved, latent[source_idx], fates, snapshot_times=dense_times)
    terminal_probs = base.terminal_fate_probabilities(sim["final_latent"], obs, latent, fates, k=30)
    realized_idx = terminal_probs.argmax(axis=1)
    realized_fate = np.asarray([fates[i] for i in realized_idx], dtype=object)
    rows = []
    for t, block in sim["snapshots"].items():
        w = sim["weights"][t]
        for gene in genes:
            score = decoded_gene_score(adata, block, gene)
            if score is None:
                continue
            for j, fate in enumerate(fates):
                sel = realized_idx == j
                mean, std = weighted_mean_std(score[sel], w[sel])
                rows.append(
                    {
                        "model_time": float(t),
                        "gene": gene,
                        "terminal_fate": fate,
                        "decoded_expression_mean": mean,
                        "decoded_expression_std": std,
                        "n_generated_particles": int(sel.sum()),
                        "generated_mass": float(np.nansum(w[sel])),
                    }
                )
    out = pd.DataFrame(rows)
    csv_path = OUT_DIR / "fig4hi_jklm_bmf_decoded_gene_expression_trajectories.csv"
    out.to_csv(csv_path, index=False)
    all_source_idx = np.where(obs["t"].eq(0).to_numpy())[0]
    sim_all = base.simulate_bmf(model, resolved, latent[all_source_idx], fates, snapshot_times=dense_times)
    all_bmf = bmf_all_particle_gene_trends(adata, sim_all, genes)
    all_bmf_path = OUT_DIR / "fig4hi_jklm_bmf_all_particle_gene_trends.csv"
    all_bmf.to_csv(all_bmf_path, index=False)
    all_obs = observed_all_cell_gene_trends(adata, obs, genes)
    all_obs_path = OUT_DIR / "fig4hi_jklm_observed_all_cell_gene_trends.csv"
    all_obs.to_csv(all_obs_path, index=False)
    all_corr_rows = []
    for gene in genes:
        m = all_bmf[all_bmf["gene"].eq(gene)].sort_values("model_time")
        o = all_obs[all_obs["gene"].eq(gene)].sort_values("model_time")
        if m.empty or o.empty:
            continue
        ox = o["model_time"].to_numpy(float)
        model_at_obs = np.interp(ox, m["model_time"].to_numpy(float), m["decoded_expression_mean"].to_numpy(float))
        obs_y = o["observed_expression_mean"].to_numpy(float)
        all_corr_rows.append(
            {
                "gene": gene,
                "n_observed_timepoints": int(len(ox)),
                "n_observed_cells_min": int(o["n_observed_cells"].min()),
                "bmf_vs_observed_temporal_corr": pearson_small_n(model_at_obs, obs_y),
                "bmf_day6_minus_day2": float(model_at_obs[-1] - model_at_obs[0]) if len(model_at_obs) >= 2 else float("nan"),
                "observed_day6_minus_day2": float(obs_y[-1] - obs_y[0]) if len(obs_y) >= 2 else float("nan"),
            }
        )
    all_corr = pd.DataFrame(all_corr_rows)
    all_corr_path = OUT_DIR / "fig4hi_jklm_bmf_vs_observed_all_cell_gene_trend_correlations.csv"
    all_corr.to_csv(all_corr_path, index=False)
    particle_path = OUT_DIR / "fig4hi_jklm_bmf_final_particle_terminal_fate_assignment.csv"
    pd.DataFrame(
        {
            "parent": sim["parent"],
            "branch_idx": sim["branch_idx"],
            "realized_terminal_fate": realized_fate,
            "final_weight": sim["final_weights"],
            **{f"prob_{fate}": terminal_probs[:, j] for j, fate in enumerate(fates)},
        }
    ).to_csv(particle_path, index=False)

    obs_trend, clone_fates = observed_gene_trends_by_clone_terminal_fate(adata, obs, genes, fates)
    obs_path = OUT_DIR / "fig4hi_jklm_observed_clone_terminal_gene_expression_trends.csv"
    obs_trend.to_csv(obs_path, index=False)
    clone_fate_path = OUT_DIR / "fig4hi_jklm_observed_clone_terminal_fate_labels.csv"
    clone_fates.to_csv(clone_fate_path, index=False)

    corr_rows = []
    for gene in genes:
        for fate in ["Neutrophil", "Monocyte", "Meg", "Erythroid"]:
            m = out[out["gene"].eq(gene) & out["terminal_fate"].eq(fate)].sort_values("model_time")
            o = obs_trend[obs_trend["gene"].eq(gene) & obs_trend["terminal_fate"].eq(fate)].sort_values("model_time")
            if m.empty or o.empty:
                continue
            ox = o["model_time"].to_numpy(float)
            model_at_obs = np.interp(ox, m["model_time"].to_numpy(float), m["decoded_expression_mean"].to_numpy(float))
            obs_y = o["observed_expression_mean"].to_numpy(float)
            corr_rows.append(
                {
                    "gene": gene,
                    "terminal_fate": fate,
                    "n_observed_timepoints": int(len(ox)),
                    "n_observed_cells_min": int(o["n_observed_cells"].min()),
                    "n_observed_clones_min": int(o["n_clones"].min()),
                    "bmf_vs_observed_temporal_corr": pearson_small_n(model_at_obs, obs_y),
                    "bmf_day6_minus_day2": float(model_at_obs[-1] - model_at_obs[0]) if len(model_at_obs) >= 2 else float("nan"),
                    "observed_day6_minus_day2": float(obs_y[-1] - obs_y[0]) if len(obs_y) >= 2 else float("nan"),
                }
            )
    corr = pd.DataFrame(corr_rows)
    corr_path = OUT_DIR / "fig4hi_jklm_bmf_vs_observed_clone_terminal_gene_trend_correlations.csv"
    corr.to_csv(corr_path, index=False)

    fig, axes = plt.subplots(2, 4, figsize=(10.8, 4.9), sharex=True)
    axes = axes.ravel()
    for ax, gene in zip(axes, genes):
        pool = out[out["gene"].eq(gene)]["decoded_expression_mean"].to_numpy(float)
        obs_pool = obs_trend[obs_trend["gene"].eq(gene)]["observed_expression_mean"].to_numpy(float)
        pool = np.concatenate([pool[np.isfinite(pool)], obs_pool[np.isfinite(obs_pool)]])
        lo = float(np.nanpercentile(pool, 2)) if len(pool) else 0.0
        hi = float(np.nanpercentile(pool, 98)) if len(pool) else 1.0
        span = max(hi - lo, 1e-8)
        for fate in ["Neutrophil", "Monocyte", "Meg", "Erythroid"]:
            d = out[out["gene"].eq(gene) & out["terminal_fate"].eq(fate)].sort_values("model_time")
            if d.empty:
                continue
            y = np.clip((d["decoded_expression_mean"].to_numpy(float) - lo) / span, -0.05, 1.05)
            s = d["decoded_expression_std"].to_numpy(float) / span
            color = base.FATE_PALETTE.get(fate, "#64748B")
            x = d["model_time"].to_numpy(float)
            ax.plot(x, y, lw=1.8, color=color, label=fate)
            ax.fill_between(x, y - s, y + s, color=color, alpha=0.13, linewidth=0)
        ax.set_title(gene, loc="left")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(axis="y", color="#E5E7EB")
    axes[0].legend(frameon=False, fontsize=6.3)
    for ax in axes[-4:]:
        ax.set_xlabel("time after day 2")
    axes[0].set_ylabel("relative decoded expression")
    axes[4].set_ylabel("relative decoded expression")
    sns.despine(fig=fig)
    fig.suptitle("Figure 4H-M analogue: BMF gene trends by realized terminal fate", x=0.02, y=0.99, ha="left", weight="bold")
    main_fig = savefig(fig, "fig4hi_jklm_bmf_decoded_gene_expression_trajectories")

    if not corr.empty:
        fig, ax = plt.subplots(figsize=(4.6, 3.9))
        show = corr[corr["terminal_fate"].isin(["Neutrophil", "Monocyte", "Meg", "Erythroid"])].copy()
        for fate, d in show.groupby("terminal_fate", observed=False):
            ax.scatter(
                d["observed_day6_minus_day2"],
                d["bmf_day6_minus_day2"],
                s=42,
                color=base.FATE_PALETTE.get(fate, "#64748B"),
                edgecolor="white",
                linewidth=0.55,
                alpha=0.9,
                label=fate,
            )
            for _, r in d.iterrows():
                if abs(float(r["observed_day6_minus_day2"])) + abs(float(r["bmf_day6_minus_day2"])) > show[["observed_day6_minus_day2", "bmf_day6_minus_day2"]].abs().to_numpy().max() * 0.35:
                    ax.text(
                        float(r["observed_day6_minus_day2"]),
                        float(r["bmf_day6_minus_day2"]),
                        str(r["gene"]),
                        fontsize=6.0,
                        color="#111827",
                    )
        lim = float(np.nanpercentile(np.abs(show[["observed_day6_minus_day2", "bmf_day6_minus_day2"]].to_numpy(float)), 98))
        lim = max(lim, 1e-6)
        ax.axhline(0, color="#94A3B8", lw=0.8)
        ax.axvline(0, color="#94A3B8", lw=0.8)
        ax.plot([-lim, lim], [-lim, lim], color="#64748B", lw=0.8, ls=":")
        ax.set_xlim(-lim * 1.15, lim * 1.15)
        ax.set_ylim(-lim * 1.15, lim * 1.15)
        overall_r = pearson_small_n(show["observed_day6_minus_day2"], show["bmf_day6_minus_day2"])
        ax.set_title(f"BMF vs observed clone-terminal trend (r={overall_r:.2f})", loc="left")
        ax.set_xlabel("observed day6 - day2 decoded expression")
        ax.set_ylabel("BMF day6 - day2 decoded expression")
        ax.legend(frameon=False, fontsize=6.5, loc="best")
        ax.grid(color="#E5E7EB", lw=0.6)
        sns.despine(ax=ax)
        sanity_fig = savefig(fig, "fig4hi_jklm_bmf_vs_observed_gene_trend_sanity")
    else:
        sanity_fig = ""
    if not all_corr.empty:
        fig, axes = plt.subplots(2, 4, figsize=(10.8, 4.9), sharex=True)
        axes = axes.ravel()
        for ax, gene in zip(axes, genes):
            m = all_bmf[all_bmf["gene"].eq(gene)].sort_values("model_time")
            o = all_obs[all_obs["gene"].eq(gene)].sort_values("model_time")
            pool = np.concatenate(
                [
                    m["decoded_expression_mean"].to_numpy(float),
                    o["observed_expression_mean"].to_numpy(float),
                ]
            )
            lo = float(np.nanpercentile(pool[np.isfinite(pool)], 2))
            hi = float(np.nanpercentile(pool[np.isfinite(pool)], 98))
            span = max(hi - lo, 1e-8)
            x = m["model_time"].to_numpy(float)
            y = np.clip((m["decoded_expression_mean"].to_numpy(float) - lo) / span, -0.05, 1.05)
            s = m["decoded_expression_std"].to_numpy(float) / span
            ax.plot(x, y, lw=2.0, color="#6D28D9", label="BMF")
            ax.fill_between(x, y - s, y + s, color="#6D28D9", alpha=0.12, linewidth=0)
            oy = np.clip((o["observed_expression_mean"].to_numpy(float) - lo) / span, -0.05, 1.05)
            ax.plot(o["model_time"], oy, lw=1.1, color="#111827", ls="--", label="observed")
            ax.scatter(
                o["model_time"],
                oy,
                s=np.clip(o["n_observed_cells"].to_numpy(float), 500, 20000) / 170,
                color="#111827",
                edgecolor="white",
                linewidth=0.35,
                zorder=30,
            )
            r = all_corr.loc[all_corr["gene"].eq(gene), "bmf_vs_observed_temporal_corr"]
            title = f"{gene}  r={float(r.iloc[0]):.2f}" if len(r) and np.isfinite(r.iloc[0]) else gene
            ax.set_title(title, loc="left")
            ax.set_ylim(-0.05, 1.05)
            ax.grid(axis="y", color="#E5E7EB")
        axes[0].legend(frameon=False, fontsize=6.3)
        for ax in axes[-4:]:
            ax.set_xlabel("time after day 2")
        axes[0].set_ylabel("relative decoded expression")
        axes[4].set_ylabel("relative decoded expression")
        sns.despine(fig=fig)
        fig.suptitle("All-cell gene trend sanity check: BMF generated vs observed", x=0.02, y=0.99, ha="left", weight="bold")
        all_sanity_fig = savefig(fig, "fig4hi_jklm_bmf_vs_observed_all_cell_gene_trends")
    else:
        all_sanity_fig = ""
    return main_fig, str(csv_path), sanity_fig, str(corr_path), str(obs_path), all_sanity_fig, str(all_corr_path)


def plot_figs1_manifold_recovery(adata: ad.AnnData, obs: pd.DataFrame, model, resolved, fates: list[str]) -> tuple[str, str]:
    latent = np.asarray(adata.obsm["X_latent"], dtype=np.float32)
    source_all = np.where(obs["t"].eq(0).to_numpy())[0]
    terminal = latent[obs["t"].eq(2).to_numpy()]
    tree = cKDTree(terminal)
    rows = []
    for n in [25, 50, 100, 200, 400, 800]:
        if n > len(source_all):
            continue
        idx = np.random.default_rng(120 + n).choice(source_all, size=n, replace=False)
        sim = base.simulate_bmf(model, resolved, latent[idx], fates, snapshot_times=np.asarray([0.0, 2.0]))
        d, nn = tree.query(sim["final_latent"], k=1)
        covered = len(np.unique(nn[d <= np.nanpercentile(d, 80)]))
        rows.append({"n_initial_cells": n, "terminal_manifold_coverage": covered / len(terminal), "median_nn_distance": float(np.median(d))})
    out = pd.DataFrame(rows)
    csv_path = OUT_DIR / "figs1_bmf_manifold_recovery.csv"
    out.to_csv(csv_path, index=False)
    fig, ax = plt.subplots(figsize=(4.2, 3.1))
    ax.plot(out["n_initial_cells"], out["terminal_manifold_coverage"], marker="o", color="#0F766E", lw=2.2)
    ax.set_xscale("log")
    ax.set_xlabel("day-2 cells sampled")
    ax.set_ylabel("terminal manifold coverage")
    ax.set_title("manifold recovery", loc="left")
    ax.grid(axis="y", color="#E5E7EB")
    sns.despine(ax=ax)
    fig.suptitle("Figure S1 analogue: BMF coverage from sampled initial cells", x=0.02, y=0.99, ha="left", weight="bold")
    return savefig(fig, "figs1_bmf_manifold_recovery"), str(csv_path)


def plot_figs2a_well_time_corr(adata: ad.AnnData) -> tuple[str, str]:
    obs = adata.obs.copy()
    rows = []
    X = np.asarray(adata.obsm["X_latent"], dtype=np.float32)
    groups = obs.groupby(["Well", "time_point_processed"], observed=False)
    means = {}
    for key, idx in groups.indices.items():
        if len(idx) >= 10:
            means[key] = X[np.asarray(idx)].mean(axis=0)
    keys = list(means)
    mat = np.eye(len(keys), dtype=float)
    for i, ki in enumerate(keys):
        for j, kj in enumerate(keys):
            if i < j:
                r = base.pearson(means[ki], means[kj])
                mat[i, j] = mat[j, i] = r
    df = pd.DataFrame(mat, index=[f"{k[0]}:t{k[1]}" for k in keys], columns=[f"{k[0]}:t{k[1]}" for k in keys])
    csv_path = OUT_DIR / "figs2a_well_time_latent_mean_correlations.csv"
    df.to_csv(csv_path)
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    sns.heatmap(df, cmap="Blues", vmin=0.7, vmax=1.0, ax=ax, cbar_kws={"label": "Pearson r"})
    ax.set_title("well/time mean-state correlation", loc="left")
    fig.suptitle("Figure S2A analogue: reproducibility of replicate time-course states", x=0.02, y=0.99, ha="left", weight="bold")
    return savefig(fig, "figs2a_well_time_latent_mean_correlations"), str(csv_path)


def plot_fate_benchmark() -> tuple[str, str]:
    rows = []
    if BENCHMARK_SUMMARY.exists():
        b = pd.read_csv(BENCHMARK_SUMMARY)
        for _, r in b.iterrows():
            rows.append({"method": r["method"], "accuracy": r["accuracy"], "negative_cross_entropy": r["negative_cross_entropy"]})
    elif METRIC_SUMMARY.exists():
        b = pd.read_csv(METRIC_SUMMARY)
        for _, r in b.iterrows():
            rows.append({"method": r["run_key"], "accuracy": r.get("CFCorr", np.nan), "negative_cross_entropy": np.nan})
    out = pd.DataFrame(rows)
    csv_path = OUT_DIR / "fig2d_s7_bmf_fate_prediction_benchmark.csv"
    out.to_csv(csv_path, index=False)
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9))
    d = out.sort_values("accuracy", ascending=False)
    axes[0].bar(d["method"], d["accuracy"], color=["#6D28D9" if "Clone" in m else "#CBD5E1" for m in d["method"]])
    axes[0].set_ylabel("fate prediction accuracy")
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].set_title("accuracy", loc="left")
    if out["negative_cross_entropy"].notna().any():
        d = out.sort_values("negative_cross_entropy", ascending=False)
        axes[1].bar(d["method"], d["negative_cross_entropy"], color=["#6D28D9" if "Clone" in m else "#CBD5E1" for m in d["method"]])
        axes[1].set_ylabel("negative cross entropy")
        axes[1].tick_params(axis="x", rotation=35)
        axes[1].set_title("probability calibration", loc="left")
    sns.despine(fig=fig)
    fig.suptitle("Figure 2D / S7 analogue: LARRY fate-prediction benchmark", x=0.02, y=0.99, ha="left", weight="bold")
    return savefig(fig, "fig2d_s7_bmf_fate_prediction_benchmark"), str(csv_path)


def write_analysis_map(paths: dict[str, str]) -> str:
    rows = [
        ("Figure2A", "Observed LARRY manifold and cell states", "Observed SPRING manifold by fate and time", "direct data analogue", paths.get("fig2a"), ""),
        ("Figure2B", "scDiffEq velocity stream plot", "BMF branch-weighted velocity plus growth field on SPRING", "model analogue", paths.get("fig2b"), ""),
        ("Figure2D", "Fate-prediction benchmark", "Clone-CUVGF/BMF vs baselines on LARRY fate probabilities", "metric analogue", paths.get("fig2d"), paths.get("fig2d_csv")),
        ("Figure2E/F", "Learned time concordance / interpolation", "BMF rollout distribution distance to observed day 2/4/6 states", "model analogue", paths.get("fig2def"), paths.get("fig2def_csv")),
        ("Figure3A/B", "Perturbed vs unperturbed terminal states", "BMF +10z RNA perturbation endpoint fate composition", "model analogue", paths.get("fig3abcd"), paths.get("fig3abcd_csv")),
        ("Figure3C/D", "Perturbation z-score scan", "BMF multi-gene z scan across terminal fate fractions", "model analogue", paths.get("fig3abcd"), paths.get("fig3abcd_csv")),
        ("Figure3E/F/G", "Genome-scale fate perturbation screens for Neutrophil/Monocyte/Baso", "All-gene BMF branch-gate screen for GMP-like and Baso/Mast branches", "fast model-native screen", paths.get("fig3efg_kl"), str(OUT_DIR / "fig3efg_kl_bmf_all_gene_branch_gate_screen.csv")),
        ("Figure3H/I", "Human hematopoiesis velocity stream examples", "Not a separate dataset here; mapped to BMF SPRING velocity/growth field", "same as Figure2B analogue", paths.get("fig2b"), ""),
        ("Figure3J", "SPI1 perturbation z scan", "BMF Spi1 z scan over terminal Neutrophil/Monocyte/Baso output", "model analogue", paths.get("fig3j"), paths.get("fig3j_csv")),
        ("Figure3K/L", "GMP and MEP perturbation volcanoes", "All-gene BMF gate screen summarized by GMP-like and MEP-like branch groups", "fast model-native screen", paths.get("fig3efg_kl"), str(OUT_DIR / "fig3efg_kl_bmf_all_gene_branch_gate_screen.csv")),
        ("Figure3M", "Pancreas scDiffEq feature panel", "No pancreas analogue in this clone case; represented by BMF feature panels in Figure2B/4H-M", "not same biological dataset", paths.get("fig4hijklm"), paths.get("fig4hijklm_csv")),
        ("Figure4A/B/C", "Best/worst fate prediction examples and F-hat comparison", "Benchmark and BMF per-cell fate probability table", "metric analogue", paths.get("fig2d"), str(BMF_TABLE)),
        ("Figure4D", "Drift and diffusion statistics", "BMF branch velocity magnitude and branch growth statistics", "model analogue", paths.get("fig4d"), str(OUT_DIR / "fig4d_bmf_branch_velocity_growth_statistics.csv")),
        ("Figure4E/F", "Drift and diffusion UMAPs", "BMF velocity and growth fields on SPRING", "model analogue", paths.get("fig2b"), ""),
        ("Figure4G", "Gene expression correlations with drift/diffusion", "Gene expression correlations with decoded velocity and branch growth", "model analogue", paths.get("fig4g"), str(OUT_DIR / "fig4g_bmf_gene_expression_velocity_growth_correlations.csv")),
        ("Figure4H/I", "Spatial expression maps for selected genes", "Decoded gene expression along BMF particles assigned to realized terminal fate", "model analogue", paths.get("fig4hijklm"), paths.get("fig4hijklm_csv")),
        ("Figure4J/K/L/M", "Smoothed gene expression trajectories by fate", "Dense BMF gene trajectories grouped by final observed terminal-fate basin; all-cell observed trend check added", "model + observed sanity check", paths.get("fig4hijklm"), paths.get("fig4hijklm_all_cell_sanity_csv")),
        ("FigureS1", "Manifold recovery with increasing initial cells", "Terminal manifold coverage from sampled BMF day-2 cells", "model analogue", paths.get("figs1"), paths.get("figs1_csv")),
        ("FigureS2A", "Replicate/time mean expression correlations", "Well/time latent-state correlations", "data analogue", paths.get("figs2a"), paths.get("figs2a_csv")),
        ("FigureS2B/C", "Train/test time-point density plots", "Covered by observed atlas plus BMF time-point recovery", "data/model analogue", paths.get("fig2a"), paths.get("fig2def_csv")),
        ("FigureS3/S4/S5", "Velocity-ratio hyperparameter and entropy analyses", "No direct ratio hyperparameter in locked BMF; closest outputs are algorithm-search and growth/mass diagnostics", "not one-to-one model parameter", str(ROOT / "figures/cellcompass_clone_case/panel_b_clone_algorithm_landscape.png"), str(ROOT / "downstream/cellcompass_clone_case/clone_bmf_particle_spring_growth_mass_summary.csv")),
        ("FigureS6/S7", "Fate prediction confusion / negative cross entropy", "Clone-CUVGF/BMF benchmark table", "metric analogue", paths.get("fig2d"), paths.get("fig2d_csv")),
        ("FigureS9", "Decomposed drift/diffusion velocity", "BMF branch-weighted velocity and branch growth decomposition", "model analogue", paths.get("fig2b"), str(OUT_DIR / "fig4d_bmf_branch_velocity_growth_statistics.csv")),
        ("FigureS10/S11", "Cross-seed perturbation reproducibility", "Single locked CellCompass model; output map records one-run BMF screen, not cross-seed reproducibility", "not available without retraining seeds", paths.get("fig3efg_kl"), str(OUT_DIR / "fig3efg_kl_bmf_all_gene_branch_gate_screen.csv")),
        ("FigureS12", "Pancreas secondary dataset", "No pancreas clone case analogue; map only", "not same biological dataset", "", ""),
    ]
    out = pd.DataFrame(rows, columns=["scdiffeq_panel", "scdiffeq_analysis", "bmf_analysis", "mapping_status", "figure", "table"])
    path = OUT_DIR / "scdiffeq_to_bmf_1to1_analysis_map.csv"
    out.to_csv(path, index=False)
    md = OUT_DIR / "scdiffeq_to_bmf_1to1_analysis_map.md"
    with md.open("w") as f:
        f.write("# scDiffEq-to-BMF one-to-one analysis map\n\n")
        f.write("Original source: https://github.com/scDiffEq/scdiffeq-analyses and https://www.scdiffeq.com/analyses.html\n\n")
        for _, row in out.iterrows():
            f.write(f"## {row['scdiffeq_panel']}\n")
            f.write(f"- scDiffEq: {row['scdiffeq_analysis']}\n")
            f.write(f"- BMF: {row['bmf_analysis']}\n")
            f.write(f"- status: {row['mapping_status']}\n")
            if str(row["figure"]):
                f.write(f"- figure: {row['figure']}\n")
            if str(row["table"]):
                f.write(f"- table: {row['table']}\n")
            f.write("\n")
    return str(path)


def main() -> None:
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    adata = ad.read_h5ad(ADATA_PATH)
    obs = base.load_obs(adata)
    bmf = pd.read_csv(BMF_TABLE)
    fates = fate_names(bmf)
    model, resolved, source_idx, sim = run_baseline_sim(adata, obs, bmf, fates, max_cells=340)

    paths: dict[str, str] = {}
    paths["fig2a"] = plot_fig2a_observed_atlas(obs)
    paths["fig2b"] = plot_fig2b_velocity_stream(adata, obs, model, fates)
    paths["fig2d"], paths["fig2d_csv"] = plot_fate_benchmark()
    paths["fig2def"], paths["fig2def_csv"] = plot_fig2def_time_recovery(adata, obs, source_idx, sim)
    paths["fig3abcd"], paths["fig3abcd_csv"] = plot_fig3ab_cd_perturb_scan(adata, obs, model, resolved, source_idx, fates)
    screen = branch_gate_screen(adata, model, np.asarray(adata.obsm["X_latent"], dtype=np.float32)[source_idx], fates, z=5.0)
    paths["fig3efg_kl"] = plot_fig3efg_kl_gene_screen(screen, fates)
    paths["fig3j"], paths["fig3j_csv"] = plot_fig3j_spi1_scan(adata, obs, model, resolved, source_idx, fates)
    stats = compute_fig4d_g_stats(adata, obs, model, fates)
    paths["fig4d"] = plot_fig4d_stats(stats)
    corr = compute_fig4g_gene_velocity_corr(adata, obs, model, fates)
    paths["fig4g"] = plot_fig4g_gene_corr(corr)
    (
        paths["fig4hijklm"],
        paths["fig4hijklm_csv"],
        paths["fig4hijklm_terminal_sanity"],
        paths["fig4hijklm_terminal_sanity_csv"],
        paths["fig4hijklm_observed_terminal_csv"],
        paths["fig4hijklm_all_cell_sanity"],
        paths["fig4hijklm_all_cell_sanity_csv"],
    ) = plot_fig4hi_jklm_gene_expression(adata, obs, source_idx, model, resolved, fates)
    paths["figs1"], paths["figs1_csv"] = plot_figs1_manifold_recovery(adata, obs, model, resolved, fates)
    paths["figs2a"], paths["figs2a_csv"] = plot_figs2a_well_time_corr(adata)
    map_path = write_analysis_map(paths)
    print(f"Wrote map: {map_path}")
    print(f"Wrote figures: {FIG_DIR}")
    print(f"Wrote tables: {OUT_DIR}")


if __name__ == "__main__":
    main()
