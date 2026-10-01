#!/usr/bin/env python3
"""Draw full-cell scVelo stream plots with continuous treatment-time coloring."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cellcompass_scvelo_matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/cellcompass_scvelo_numba")
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scvelo as scv
from sklearn.decomposition import PCA


REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "analysis" / "pulsebridge_story" / "outputs"
STREAM = OUT / "downstream" / "velocity_stream"
FIG = OUT / "figures"
DYNAMO_STREAM = OUT / "downstream" / "dynamo_stream"


def load_story_module():
    script = Path(__file__).resolve().parent / "pulsebridge_story_analysis.py"
    spec = importlib.util.spec_from_file_location("pulsebridge_story_analysis", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {script}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def style_stream_axes(fig: plt.Figure, xy: np.ndarray) -> None:
    if not fig.axes:
        return
    ax = max(fig.axes, key=lambda item: item.get_position().width * item.get_position().height)
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    x0, x1 = np.quantile(xy[:, 0], [0.005, 0.995])
    y0, y1 = np.quantile(xy[:, 1], [0.005, 0.995])
    xpad = 0.04 * (x1 - x0)
    ypad = 0.04 * (y1 - y0)
    ax.set_xlim(x0 - xpad, x1 + xpad)
    ax.set_ylim(y0 - ypad, y1 + ypad)


def draw_scvelo_stream(adata: ad.AnnData, basis: str, out_png: Path) -> dict:
    xy = np.asarray(adata.obsm[f"X_{basis}"], dtype=float)[:, :2]
    fig, ax = plt.subplots(figsize=(7, 5.8))
    scv.pl.velocity_embedding_stream(
        adata,
        basis=basis,
        vkey="velocity",
        color="time_point_processed",
        color_map="viridis",
        colorbar=True,
        legend_loc="right margin",
        density=0.78,
        linewidth=0.62,
        arrow_color="black",
        arrow_size=0.5,
        alpha=0.28,
        size=18,
        title="",
        frameon=False,
        show=False,
        ax=ax,
    )
    style_stream_axes(fig, xy)
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return {
        "renderer": "scvelo.pl.velocity_embedding_stream",
        "basis": basis,
        "vkey": "velocity",
        "color_key": "time_point_processed",
        "cell_colormap": "viridis",
        "streamline_color": "black",
        "density": 0.78,
        "linewidth": 0.62,
        "arrow_size": 0.5,
        "alpha": 0.28,
        "downsampled": False,
        "n_cells": int(adata.n_obs),
    }


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-pca",
        action="store_true",
        help="Also build the full-cell PCA velocity graph and draw a PCA stream plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mod = load_story_module()
    mod.ensure_inputs()
    cyto_path = str(REPO / "CytoBridge-main")
    if cyto_path not in sys.path:
        sys.path.insert(0, cyto_path)
    if args.include_pca:
        from CytoBridge.tl.downstream import build_velocity_graph_bundle

    input_h5ad = DYNAMO_STREAM / "pulsebridge_full_umap_velocity_for_dynamo.h5ad"
    if not input_h5ad.exists():
        raise FileNotFoundError(
            f"Missing {input_h5ad}. Run prepare_dynamo_stream_inputs.py first."
        )

    STREAM.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(input_h5ad)
    adata.obs["time_point_processed"] = pd.to_numeric(
        adata.obs["time_point_processed"], errors="coerce"
    ).astype(float)
    if "treatment_day" not in adata.obs:
        adata.obs["treatment_day"] = pd.Categorical(
            [f"day {int(round(float(x)))}" for x in adata.obs["time_point_processed"]],
            categories=["day 0", "day 1", "day 2", "day 3"],
            ordered=True,
        )

    required = ["X_umap", "velocity_umap", "X_latent", "velocity_latent"]
    missing = [key for key in required if key not in adata.obsm]
    if missing:
        raise KeyError(f"Missing required adata.obsm keys: {missing}")
    if "velocity_graph" not in adata.uns:
        raise KeyError("Missing adata.uns['velocity_graph'] for scVelo UMAP stream")

    umap_png = FIG / "story_scvelo_native_stream_umap_full_time.png"
    umap_stats = draw_scvelo_stream(adata, "umap", umap_png)

    pca_png = FIG / "story_scvelo_native_stream_pca_full_time.png"
    pca_stats = None
    graph_pca = None
    if args.include_pca:
        pca_adata = adata.copy()
        pca_adata.obsm["X_pca"] = PCA(n_components=2, random_state=31).fit_transform(
            np.asarray(pca_adata.obsm["X_latent"], dtype=np.float32)
        ).astype(np.float32)
        graph_pca = build_velocity_graph_bundle(
            adata=pca_adata,
            output_dir=str(STREAM),
            n_jobs=4,
            reuse_neighbors=False,
            preferred_basis="pca",
            strict_preferred=True,
        )
        pca_stats = draw_scvelo_stream(pca_adata, "pca", pca_png)

    manifest = {
        "analysis": "scvelo-full-time-stream",
        "scvelo_version": getattr(scv, "__version__", "unknown"),
        "input_h5ad": str(input_h5ad),
        "n_cells": int(adata.n_obs),
        "downsampled": False,
        "cell_color": {
            "obs_key": "time_point_processed",
            "type": "continuous_numeric",
            "range": [
                float(np.nanmin(adata.obs["time_point_processed"].to_numpy(dtype=float))),
                float(np.nanmax(adata.obs["time_point_processed"].to_numpy(dtype=float))),
            ],
            "colormap": "viridis",
        },
        "figures": {
            "umap": str(umap_png),
            "pca": str(pca_png) if pca_stats is not None else None,
        },
        "umap_stream": umap_stats,
        "pca_stream": pca_stats,
        "pca_graph": graph_pca,
        "method_notes": [
            "UMAP stream reuses full-cell X_umap, velocity_umap, and velocity_graph from prepare_dynamo_stream_inputs.py.",
            "PCA stream is optional via --include-pca; when enabled it uses full-cell PCA(X_latent) and build_velocity_graph_bundle(preferred_basis='pca') before scVelo plotting.",
            "No cell downsampling is applied.",
            "Plots use scvelo.pl.velocity_embedding_stream directly; only visual parameters are adjusted.",
        ],
        "warnings": (graph_pca or {}).get("warnings", []),
    }
    manifest_path = STREAM / "scvelo_full_time_stream_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
