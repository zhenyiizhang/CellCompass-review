#!/usr/bin/env python3
"""Prepare BMF velocity/growth inputs for plotting with scDiffEq itself."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cellcompass_mpl_cache")
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import anndata as ad
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compose_clone_scdiffeq_1to1_bmf_analyses as ana  # noqa: E402
import compose_clone_scdiffeq_style_biology as base  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "downstream/cellcompass_clone_case/scdiffeq_1to1_bmf"
VELOCITY_INPUT = OUT_DIR / "fig2b_scdiffeq_velocity_input.h5ad"
GROWTH_TABLE = OUT_DIR / "fig2b_fig4ef_bmf_growth_on_observed_timepoints.csv"


def chunked_weighted_velocity_growth(
    model,
    latent: np.ndarray,
    fates: list[str],
    t_value: float,
    chunk_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    velocity_chunks = []
    growth_chunks = []
    for start in range(0, latent.shape[0], chunk_size):
        end = min(start + chunk_size, latent.shape[0])
        _, velocity, growth = ana.weighted_velocity_growth(model, latent[start:end], fates, t_value=t_value)
        velocity_chunks.append(velocity.astype(np.float32))
        growth_chunks.append(growth.astype(float))
    return np.concatenate(velocity_chunks, axis=0), np.concatenate(growth_chunks, axis=0)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    adata = ad.read_h5ad(ana.ADATA_PATH)
    obs = base.load_obs(adata)
    latent = np.asarray(adata.obsm["X_latent"], dtype=np.float32)

    bmf = pd.read_csv(ana.BMF_TABLE)
    fates = ana.fate_names(bmf)
    model, _ = base.load_bmf_model(latent.shape[1], len(fates))

    all_idx = np.arange(obs.shape[0], dtype=int)
    velocity = np.zeros_like(latent, dtype=np.float32)
    growth_all = np.zeros(obs.shape[0], dtype=float)
    for obs_t in sorted(obs["t"].unique()):
        tidx = np.where(obs["t"].eq(obs_t).to_numpy())[0]
        vt, gt = chunked_weighted_velocity_growth(model, latent[tidx], fates, t_value=float(obs_t))
        velocity[tidx] = vt
        growth_all[tidx] = gt

    plot_obs = obs.copy()
    categories = [f for f in base.FATE_PALETTE if f in set(plot_obs["cell_type"])]
    plot_obs["cell_type"] = pd.Categorical(plot_obs["cell_type"].astype(str), categories=categories)
    plot_obs["bmf_growth_dlogmass_dt"] = growth_all.astype(float)
    plot_obs.index = [str(i) for i in all_idx]

    velocity_adata = ad.AnnData(X=latent.astype(np.float32), obs=plot_obs)
    velocity_adata.obsm["X_pca"] = latent.astype(np.float32)
    velocity_adata.obsm["X_drift"] = velocity.astype(np.float32)
    # scDiffEq velocity_stream reads X_umap internally; here it is the published SPRING map.
    velocity_adata.obsm["X_umap"] = plot_obs[["spring_x", "spring_y"]].to_numpy(np.float32)
    velocity_adata.write_h5ad(VELOCITY_INPUT)

    growth_rows = []
    day_lookup = {0: 2, 1: 4, 2: 6}
    for cell_idx, row, value in zip(all_idx, obs.itertuples(index=False), growth_all):
        obs_t = int(row.t)
        growth_rows.append(
            {
                "cell_index": int(cell_idx),
                "time_point_processed": obs_t,
                "day": int(day_lookup.get(obs_t, obs_t)),
                "cell_type": str(row.cell_type),
                "spring_x": float(row.spring_x),
                "spring_y": float(row.spring_y),
                "bmf_growth_dlogmass_dt": float(value),
            }
            )
    pd.DataFrame(growth_rows).to_csv(GROWTH_TABLE, index=False)
    print(f"Wrote {VELOCITY_INPUT}")
    print(f"Wrote {GROWTH_TABLE}")


if __name__ == "__main__":
    main()
