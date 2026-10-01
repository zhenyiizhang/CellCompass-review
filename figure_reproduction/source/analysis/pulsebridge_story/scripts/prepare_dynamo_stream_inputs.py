#!/usr/bin/env python3
"""Prepare full-cell PulseBridge velocity and UMAP inputs for dynamo stream plotting."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cellcompass_dynamo_matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/cellcompass_dynamo_numba")
os.environ.setdefault("NUMBA_DISABLE_JIT", "0")

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc


def load_story_module():
    script = Path(__file__).resolve().parent / "pulsebridge_story_analysis.py"
    spec = importlib.util.spec_from_file_location("pulsebridge_story_analysis", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {script}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    mod = load_story_module()
    mod.ensure_inputs()
    out = mod.OUT / "downstream" / "dynamo_stream"
    out.mkdir(parents=True, exist_ok=True)

    cyto_path = str(mod.REPO / "CytoBridge-main")
    import sys

    if cyto_path not in sys.path:
        sys.path.insert(0, cyto_path)
    from CytoBridge.tl.downstream import compute_velocity_bundle, build_velocity_graph_bundle

    source = ad.read_h5ad(mod.ADATA_PATH if mod.ADATA_PATH.exists() else mod.PREPARED_H5AD_PATH, backed="r")
    x_latent = np.asarray(source.obsm["X_latent"], dtype=np.float32)
    times = source.obs["time_point_processed"].astype(float).to_numpy()
    obs = source.obs.copy()
    obs["time_point_processed"] = pd.to_numeric(obs["time_point_processed"], errors="coerce").astype(float)
    obs["treatment_day"] = pd.Categorical(
        [f"day {int(round(float(x)))}" for x in obs["time_point_processed"]],
        categories=["day 0", "day 1", "day 2", "day 3"],
        ordered=True,
    )

    # Lightweight AnnData: dynamo/scVelo need coordinates and velocity, not the full expression matrix.
    adata = ad.AnnData(X=np.zeros((x_latent.shape[0], 1), dtype=np.float32), obs=obs)
    adata.var_names = ["placeholder"]
    adata.obsm["X_latent"] = x_latent

    print(f"[dynamo-prep] computing full UMAP for {adata.n_obs} cells", flush=True)
    sc.pp.neighbors(adata, n_neighbors=35, use_rep="X_latent", random_state=31)
    sc.tl.umap(adata, random_state=31, min_dist=0.35)

    print("[dynamo-prep] computing PulseBridge latent velocity", flush=True)
    model, record = mod.load_pulsebridge_model_for_downstream(latent_dim=x_latent.shape[1])
    vel = compute_velocity_bundle(adata=adata, model=model, device="cpu", output_dir=str(out))

    print("[dynamo-prep] projecting velocity to UMAP basis", flush=True)
    graph = build_velocity_graph_bundle(
        adata=adata,
        output_dir=str(out),
        n_neighbors=35,
        n_jobs=4,
        reuse_neighbors=True,
        preferred_basis="umap",
        strict_preferred=True,
    )

    if "velocity_umap" not in adata.obsm:
        raise RuntimeError("velocity_umap missing after build_velocity_graph_bundle(preferred_basis='umap')")

    h5ad_path = out / "pulsebridge_full_umap_velocity_for_dynamo.h5ad"
    adata.write_h5ad(h5ad_path)
    coord_table = out / "pulsebridge_full_umap_velocity_cells.csv"
    pd.DataFrame(
        {
            "cell": adata.obs_names.to_numpy(),
            "umap_1": adata.obsm["X_umap"][:, 0],
            "umap_2": adata.obsm["X_umap"][:, 1],
            "velocity_umap_1": adata.obsm["velocity_umap"][:, 0],
            "velocity_umap_2": adata.obsm["velocity_umap"][:, 1],
            "time_point_processed": adata.obs["time_point_processed"].to_numpy(),
            "treatment_day": adata.obs["treatment_day"].astype(str).to_numpy(),
        }
    ).to_csv(coord_table, index=False)

    manifest = {
        "analysis": "prepare-dynamo-stream-inputs",
        "n_cells": int(adata.n_obs),
        "downsampled": False,
        "basis": "umap_from_X_latent_full_cells",
        "velocity_embedding": "velocity_umap from build_velocity_graph_bundle(preferred_basis='umap')",
        "input_h5ad_path": str((mod.ADATA_PATH if mod.ADATA_PATH.exists() else mod.PREPARED_H5AD_PATH).relative_to(mod.REPO)),
        "model_state_path": str(mod.MODEL_STATE_PATH.relative_to(mod.REPO)),
        "model_artifact_path": str(mod.MODEL_ARTIFACT_PATH.relative_to(mod.REPO)),
        "algorithm_id": str(record.get("algorithm_id", "")),
        "stage": str(record.get("stage", "")),
        "artifacts": {
            "dynamo_h5ad": str(h5ad_path),
            "cell_table": str(coord_table),
            "velocity": vel.get("artifacts", {}),
            "velocity_graph": graph.get("artifacts", {}),
        },
        "warnings": vel.get("warnings", []) + graph.get("warnings", []),
    }
    manifest_path = out / "prepare_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
