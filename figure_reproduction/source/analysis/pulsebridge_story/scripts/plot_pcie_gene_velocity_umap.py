#!/usr/bin/env python3
"""UMAP visualization of PCIE-matched gene velocities.

For selected genes, plot the two quantities that are compared by the PCIE
evaluation after nearest-particle matching:

- PulseBridge: generated trajectory finite-difference tangent projected to
  the gene through PCA loadings.
- Metabolic pulse: heldout RNA-labeling velocity projected to the same gene.

The plotted cells are heldout PCIE anchors at days 0, 1, and 2. Day 3 is not
included because PCIE needs a forward finite-difference tangent.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cellcompass_pcie_gene_umap")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/cellcompass_pcie_gene_umap_numba")
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from sklearn.neighbors import NearestNeighbors


REPO = Path(__file__).resolve().parents[3]
BUNDLE = REPO / "downloads" / "pulsebridge_welltemp_hct116_20260627"
OUT = REPO / "analysis" / "pulsebridge_story" / "outputs"
FIG = OUT / "figures"
TAB = OUT / "tables"
DOWNSTREAM = OUT / "downstream" / "pcie_gene_space"
ADATA_PATH = (
    BUNDLE
    / "home"
    / "ubuntu"
    / ".cellcompass"
    / "algorithm_benchmarks"
    / "datasets"
    / "pcib_welltemp_hct116_raw_pool_20k"
    / "data.h5ad"
)
TRAJ_PATH = (
    BUNDLE
    / "data"
    / "cytobridge"
    / "runs"
    / "night_wave_20260621_0140"
    / "existing_deep_raw_a"
    / "training_runs"
    / "20260620T234044691964Z__pilot__pulse_control_identifiability_bridge__campaign_20260-trial_20260620-pcib_welltem"
    / "artifacts"
    / "evaluation_trajectory.npz"
)
METRICS_PATH = TRAJ_PATH.parent / "metrics.json"
UMAP_H5AD = OUT / "downstream" / "dynamo_stream" / "pulsebridge_full_umap_velocity_for_dynamo.h5ad"
GENE_TABLE = TAB / "pcie_gene_space_top_genes.csv"


def build_nuisance_basis(latent: np.ndarray, obs: pd.DataFrame, rank: int, rng: np.random.Generator) -> np.ndarray:
    times = obs["time_point_processed"].astype(float).to_numpy()
    sample = obs["sample_name"].astype(str).to_numpy()
    control = (times == 0.0) & (sample == "HCT116_control")
    baseline = (times == 0.0) & (sample == "HCT116_0d")
    x_control = latent[control]
    x_baseline = latent[baseline]
    n = min(4000, len(x_control), len(x_baseline))
    if n < 10:
        return np.zeros((latent.shape[1], 0), dtype=np.float32)
    control_idx = rng.choice(len(x_control), n, replace=False)
    baseline_idx = rng.choice(len(x_baseline), n, replace=False)
    delta = x_baseline[baseline_idx] - x_control[control_idx]
    delta = delta - delta.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(delta, full_matrices=False)
    q, _ = np.linalg.qr(vh[:rank].T.astype(np.float32))
    return q[:, :rank].astype(np.float32)


def project_perp(x: np.ndarray, basis: np.ndarray) -> np.ndarray:
    if basis.size == 0:
        return x.astype(np.float32, copy=False)
    return (x - (x @ basis) @ basis.T).astype(np.float32, copy=False)


def selected_genes() -> list[str]:
    if GENE_TABLE.exists():
        top = pd.read_csv(GENE_TABLE).head(8)["gene"].astype(str).tolist()
    else:
        top = ["MT-RNR2", "MALAT1", "RPL13", "RPS2", "MT-CO2", "MT-CO1"]
    priority = ["MT-RNR2", "MALAT1", "RPS2", "MT-CO1", "SAT1", "LGALS1"]
    out = []
    for gene in priority + top:
        if gene not in out:
            out.append(gene)
    return out[:6]


def compute_selected_gene_values(genes: list[str]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1729)
    adata = ad.read_h5ad(ADATA_PATH)
    umap_adata = ad.read_h5ad(UMAP_H5AD)
    if list(umap_adata.obs_names) != list(adata.obs_names):
        raise RuntimeError("UMAP h5ad obs_names do not match data h5ad obs_names")
    umap = np.asarray(umap_adata.obsm["X_umap"], dtype=np.float32)
    times = adata.obs["time_point_processed"].astype(float).to_numpy()

    with METRICS_PATH.open() as f:
        metrics = json.load(f)
    custom = metrics.get("custom_metrics", {})
    nuisance_rank = int(custom.get("pcib_nuisance_rank", 3))

    latent = np.asarray(adata.obsm["X_latent"], dtype=np.float32)
    metabolic = np.asarray(adata.obsm["X_metabolic_velocity"], dtype=np.float32)
    obs = adata.obs.copy()
    split = obs["mvab_anchor_split"].astype(str).to_numpy()
    confidence = obs["mvab_anchor_confidence"].astype(float).to_numpy().astype(np.float32)
    pcs = np.asarray(adata.varm["PCs"], dtype=np.float32)[:, : latent.shape[1]]
    gene_to_idx = {str(g): int(adata.var_names.get_loc(g)) for g in genes if g in set(adata.var_names)}
    genes = [g for g in genes if g in gene_to_idx]
    gene_loadings = np.stack([pcs[gene_to_idx[g]] for g in genes], axis=0).astype(np.float32)

    basis = build_nuisance_basis(latent, obs, rank=nuisance_rank, rng=rng)
    z = np.load(TRAJ_PATH)
    time_points = np.asarray(z["time_points"], dtype=float)
    observed_indices = np.asarray(z["observed_time_indices"], dtype=int)
    observed_times = np.asarray(z["observed_time_points"], dtype=float)

    rows = []
    for time_i, traj_idx in enumerate(observed_indices):
        traj_idx = int(traj_idx)
        if traj_idx + 1 >= len(time_points):
            continue
        anchor_time = float(observed_times[time_i])
        anchor_rows = np.where((times == anchor_time) & (split == "heldout") & (confidence > 0))[0]
        if anchor_rows.size == 0:
            continue
        anchor_x = latent[anchor_rows]
        pulse_perp = project_perp(metabolic[anchor_rows], basis)
        anchor_conf = confidence[anchor_rows]
        valid = (np.linalg.norm(pulse_perp, axis=1) > 1e-8) & (anchor_conf > 0)
        anchor_rows = anchor_rows[valid]
        anchor_x = anchor_x[valid]
        pulse_perp = pulse_perp[valid]
        anchor_conf = anchor_conf[valid]
        if anchor_rows.size == 0:
            continue

        cur = np.asarray(z[f"points_{traj_idx:06d}"], dtype=np.float32)
        nxt = np.asarray(z[f"points_{traj_idx + 1:06d}"], dtype=np.float32)
        n = min(len(cur), len(nxt))
        cur, nxt = cur[:n], nxt[:n]
        if n > 4096:
            particle_idx = rng.choice(np.arange(n), size=4096, replace=False)
            cur_ref = cur[particle_idx]
            nxt_ref = nxt[particle_idx]
        else:
            cur_ref, nxt_ref = cur, nxt
        dt = max(abs(float(time_points[traj_idx + 1] - time_points[traj_idx])), 1e-8)
        gen_velocity = (nxt_ref - cur_ref) / dt
        nn = NearestNeighbors(n_neighbors=1).fit(cur_ref)
        dist, ind = nn.kneighbors(anchor_x)
        gen_perp = project_perp(gen_velocity[ind[:, 0]], basis)

        gen_gene = gen_perp @ gene_loadings.T
        pulse_gene = pulse_perp @ gene_loadings.T
        denom = np.maximum(np.abs(gen_gene) + np.abs(pulse_gene), 1e-8)
        signed_agreement = (gen_gene * pulse_gene) / denom

        for local_j, gene in enumerate(genes):
            for i, cell_idx in enumerate(anchor_rows):
                rows.append(
                    {
                        "gene": gene,
                        "anchor_time": anchor_time,
                        "cell_index": int(cell_idx),
                        "cell": str(adata.obs_names[cell_idx]),
                        "umap_1": float(umap[cell_idx, 0]),
                        "umap_2": float(umap[cell_idx, 1]),
                        "confidence": float(anchor_conf[i]),
                        "anchor_distance": float(dist[i, 0]),
                        "pulsebridge_gene_velocity": float(gen_gene[i, local_j]),
                        "metabolic_gene_velocity": float(pulse_gene[i, local_j]),
                        "signed_agreement": float(signed_agreement[i, local_j]),
                        "same_sign": bool(np.sign(gen_gene[i, local_j]) == np.sign(pulse_gene[i, local_j])),
                    }
                )
        print(f"[umap] anchor_time={anchor_time:g} pairs={len(anchor_rows)}", flush=True)
    return pd.DataFrame(rows), umap, times


def symmetric_norm(values: np.ndarray) -> TwoSlopeNorm:
    vmax = float(np.nanpercentile(np.abs(values), 98))
    vmax = max(vmax, 1e-8)
    return TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)


def add_scaled_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pulsebridge_gene_velocity_scaled"] = np.nan
    df["metabolic_gene_velocity_scaled"] = np.nan
    df["signed_agreement_scaled"] = np.nan
    for gene, idx in df.groupby("gene").groups.items():
        sub = df.loc[idx]
        pb = sub["pulsebridge_gene_velocity"].to_numpy(dtype=float)
        met = sub["metabolic_gene_velocity"].to_numpy(dtype=float)
        pb_scale = max(float(np.nanpercentile(np.abs(pb), 98)), 1e-8)
        met_scale = max(float(np.nanpercentile(np.abs(met), 98)), 1e-8)
        pb_scaled = np.clip(pb / pb_scale, -1, 1)
        met_scaled = np.clip(met / met_scale, -1, 1)
        agree = np.sign(pb_scaled) * np.sign(met_scaled) * np.minimum(np.abs(pb_scaled), np.abs(met_scaled))
        df.loc[idx, "pulsebridge_gene_velocity_scaled"] = pb_scaled
        df.loc[idx, "metabolic_gene_velocity_scaled"] = met_scaled
        df.loc[idx, "signed_agreement_scaled"] = agree
    return df


def draw_umap_panels(
    df: pd.DataFrame,
    umap: np.ndarray,
    times: np.ndarray,
    genes: list[str],
    output_path: Path | None = None,
    title: str = "PCIE-matched gene velocity on observed UMAP anchors (source-scaled)",
    first_col_title: str = "PulseBridge generated tangent",
) -> None:
    fig, axes = plt.subplots(len(genes), 3, figsize=(12.8, 3.05 * len(genes)), sharex=True, sharey=True)
    if len(genes) == 1:
        axes = axes.reshape(1, -1)
    bg_idx = np.arange(umap.shape[0])
    rng = np.random.default_rng(2718)
    if len(bg_idx) > 9000:
        bg_idx = rng.choice(bg_idx, size=9000, replace=False)

    for row_i, gene in enumerate(genes):
        sub = df[df["gene"] == gene].copy()
        vel_norm = TwoSlopeNorm(vcenter=0.0, vmin=-1.0, vmax=1.0)
        agree_norm = TwoSlopeNorm(vcenter=0.0, vmin=-1.0, vmax=1.0)
        for col_i, (col, title, cmap, norm) in enumerate(
            [
                ("pulsebridge_gene_velocity_scaled", first_col_title, "RdBu_r", vel_norm),
                ("metabolic_gene_velocity_scaled", "metabolic pulse", "RdBu_r", vel_norm),
                ("signed_agreement_scaled", "signed agreement", "viridis", agree_norm),
            ]
        ):
            ax = axes[row_i, col_i]
            ax.scatter(
                umap[bg_idx, 0],
                umap[bg_idx, 1],
                c="#D3D6DA",
                s=3,
                alpha=0.22,
                linewidths=0,
                rasterized=True,
            )
            sc = ax.scatter(
                sub["umap_1"],
                sub["umap_2"],
                c=sub[col],
                cmap=cmap,
                norm=norm,
                s=11 + 20 * np.clip(sub["confidence"].to_numpy(), 0, 1),
                alpha=0.86,
                linewidths=0.12,
                edgecolors="white",
                rasterized=True,
            )
            if row_i == 0:
                ax.set_title(title, fontsize=10.5)
            if col_i == 0:
                same = float(sub["same_sign"].mean())
                ax.set_ylabel(f"{gene}\n{same:.0%} same sign", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal", adjustable="box")
            cbar = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.01)
            cbar.ax.tick_params(labelsize=7)
    fig.suptitle(
        title,
        fontsize=14.5,
        fontweight="bold",
        y=0.995,
    )
    path = output_path or (FIG / "story_pcie_gene_velocity_umap.png")
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    DOWNSTREAM.mkdir(parents=True, exist_ok=True)
    genes = selected_genes()
    df, umap, times = compute_selected_gene_values(genes)
    df = add_scaled_columns(df)
    genes = [g for g in genes if g in set(df["gene"])]
    table_path = TAB / "pcie_gene_velocity_umap_values.csv"
    df.to_csv(table_path, index=False)
    draw_umap_panels(df, umap, times, genes)
    manifest = {
        "analysis": "PCIE-matched selected gene velocities on observed UMAP anchors",
        "selected_genes": genes,
        "row_count": int(len(df)),
        "anchor_cell_count_per_gene": int(len(df) / max(len(genes), 1)),
        "outputs": {
            "figure": str(FIG / "story_pcie_gene_velocity_umap.png"),
            "table": str(table_path),
        },
        "method": {
            "pulsebridge_gene_velocity": "nearest generated particle finite-difference tangent, nuisance-projected, then dotted with the gene PCA loading",
            "metabolic_gene_velocity": "heldout metabolic pulse vector, nuisance-projected, then dotted with the same gene PCA loading",
            "scaled_columns": "for plotting, each gene/source is divided by its own 98th percentile absolute velocity; raw velocities remain in the table",
            "visual_cells": "heldout PCIE anchor cells at days 0, 1, and 2 shown on observed UMAP coordinates",
        },
    }
    (OUT / "pcie_gene_velocity_umap_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
