#!/usr/bin/env python3
"""CITE-PCFM induced protein velocity stream on the observed protein UMAP.

This script is intentionally separate from the distribution-validation figure.
It computes a model-induced protein readout velocity by pushing the trusted
50D RNA ODE velocity through the protein readout h_eta(r,t):

    q(t) = h_eta(r(t), t)
    dq/dt = d h_eta / d r * dr/dt + d h_eta / d t

The stream plot is a local flow diagnostic on the observed protein manifold,
not measured protein velocity and not continuous trajectory evidence by itself.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Callable

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/cytobridge_mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/cytobridge_xdg_cache")
os.environ.setdefault("NUMBA_CACHE_DIR", "/private/tmp/numba_cache")
os.environ.setdefault("NUMBA_DISABLE_CACHE", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")
for key in ["MPLCONFIGDIR", "XDG_CACHE_HOME", "NUMBA_CACHE_DIR"]:
    Path(os.environ[key]).mkdir(parents=True, exist_ok=True)

import anndata as ad
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from multistageot_style import PHENO_COLORS, PHENO_ORDER

try:
    import umap  # type: ignore

    HAVE_UMAP = True
except Exception:
    HAVE_UMAP = False


ROOT = Path("/Users/zhenyizhang/Documents/GitHub/CytoBridge-agent-runtime-v2-20260309/multi-omics")
REPO_ROOT = ROOT.parent
CYTOBRIDGE_SRC = REPO_ROOT / "CytoBridge-main"
ALGORITHM_PATH = Path("/Users/zhenyizhang/.cellcompass/training_algorithms/cite_paired_coord_fm/algorithm.py")
TRAINED_MODEL_H5AD = ROOT / "training_runs/20260526T174204321821Z__final__cite_paired_coord_fm__campaign_20260-trial_20260526-gse305370_ci/artifacts/trained_model.h5ad"

FIG_DIR = ROOT / "figures/cite_pcfm_deep_story/biology_landscape_discovery"
OUT_DIR = ROOT / "downstream/cite_pcfm_deep_story/biology_landscape_discovery_protein_pushforward_velocity"

RNA_DIM = 50
PROTEIN_DIM = 30
RANDOM_STATE = 20260528


def ensure_import_paths() -> None:
    if CYTOBRIDGE_SRC.exists() and str(CYTOBRIDGE_SRC) not in sys.path:
        sys.path.insert(0, str(CYTOBRIDGE_SRC))


def load_readout_input() -> Callable[[torch.Tensor, torch.Tensor, int, float], torch.Tensor]:
    """Load the exact CITE-PCFM readout input helper from the algorithm file."""
    spec = importlib.util.spec_from_file_location("cite_pcfm_algorithm_runtime", ALGORITHM_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load algorithm module from {ALGORITHM_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, "_readout_input")


def fit_observed_protein_umap(protein: np.ndarray) -> tuple[str, StandardScaler, object, np.ndarray]:
    scaler = StandardScaler().fit(protein)
    x_scaled = scaler.transform(protein)
    if HAVE_UMAP:
        reducer = umap.UMAP(
            n_neighbors=35,
            min_dist=0.22,
            metric="euclidean",
            random_state=RANDOM_STATE,
            low_memory=True,
        )
        emb = reducer.fit_transform(x_scaled)
        return "UMAP", scaler, reducer, emb.astype(np.float32)
    reducer = PCA(n_components=2, random_state=RANDOM_STATE)
    emb = reducer.fit_transform(x_scaled)
    return "PCA fallback", scaler, reducer, emb.astype(np.float32)


def compute_pushforward_protein_velocity(
    model,
    readout_input: Callable[[torch.Tensor, torch.Tensor, int, float], torch.Tensor],
    rna: np.ndarray,
    time_col: np.ndarray,
    *,
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return q=h_eta(r,t), dq/dt, and trusted RNA velocity dr/dt."""
    model = model.cpu().eval()
    for param in model.parameters():
        param.requires_grad_(False)

    n_fourier = int(getattr(model, "cite_pcfm_time_fourier_features", 0))
    time_fourier_scale = float(getattr(model, "cite_pcfm_time_fourier_scale", 1.0))

    q_parts: list[np.ndarray] = []
    vq_parts: list[np.ndarray] = []
    vr_parts: list[np.ndarray] = []

    for start in range(0, rna.shape[0], batch_size):
        end = min(start + batch_size, rna.shape[0])
        r = torch.as_tensor(rna[start:end], dtype=torch.float32)
        t = torch.as_tensor(time_col[start:end], dtype=torch.float32)
        zero_q = torch.zeros((end - start, PROTEIN_DIM), dtype=torch.float32)

        with torch.no_grad():
            full_velocity = model.velocity_net(torch.cat([r, zero_q, t], dim=1))
            v_r = full_velocity[:, :RNA_DIM].detach()

        def readout_func(r_arg: torch.Tensor, t_arg: torch.Tensor) -> torch.Tensor:
            return model.protein_readout(readout_input(r_arg, t_arg, n_fourier, time_fourier_scale))

        q, v_q = torch.autograd.functional.jvp(
            readout_func,
            (r, t),
            (v_r, torch.ones_like(t)),
            create_graph=False,
            strict=False,
        )
        q_parts.append(q.detach().cpu().numpy().astype(np.float32))
        vq_parts.append(v_q.detach().cpu().numpy().astype(np.float32))
        vr_parts.append(v_r.detach().cpu().numpy().astype(np.float32))

    return np.vstack(q_parts), np.vstack(vq_parts), np.vstack(vr_parts)


def main() -> dict:
    ensure_import_paths()
    from CytoBridge.pl.downstream import plot_velocity_stream_bundle
    from CytoBridge.tl.downstream import build_velocity_graph_bundle
    from CytoBridge.utils import load_model_from_adata

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model_adata = ad.read_h5ad(TRAINED_MODEL_H5AD)
    if "X_rna_svd" not in model_adata.obsm or "X_protein_pca" not in model_adata.obsm:
        raise KeyError("trained_model.h5ad must contain obsm['X_rna_svd'] and obsm['X_protein_pca']")
    if "time_point_processed" not in model_adata.obs:
        raise KeyError("trained_model.h5ad must contain obs['time_point_processed']")

    rna = np.asarray(model_adata.obsm["X_rna_svd"], dtype=np.float32)[:, :RNA_DIM]
    protein = np.asarray(model_adata.obsm["X_protein_pca"], dtype=np.float32)[:, :PROTEIN_DIM]
    time_col = np.asarray(model_adata.obs["time_point_processed"], dtype=np.float32).reshape(-1, 1)

    emb_name, _, _, protein_umap = fit_observed_protein_umap(protein)

    model = load_model_from_adata(model_adata)
    readout_input = load_readout_input()
    q_pred, protein_velocity, rna_velocity = compute_pushforward_protein_velocity(
        model,
        readout_input,
        rna,
        time_col,
        batch_size=512,
    )

    stream_adata = ad.AnnData(X=protein, obs=model_adata.obs.copy())
    stream_adata.obs_names = model_adata.obs_names.copy()
    stream_adata.obsm["X_latent"] = protein
    stream_adata.obsm["velocity_latent"] = protein_velocity
    stream_adata.obsm["X_umap"] = protein_umap
    stream_adata.obs["time_label"] = [
        f"t{int(float(t))}" if float(t).is_integer() else f"t{float(t):g}"
        for t in stream_adata.obs["time_point_processed"]
    ]
    if "cell_type" in stream_adata.obs:
        present_cell_types = [label for label in PHENO_ORDER if label in set(stream_adata.obs["cell_type"].astype(str))]
        stream_adata.obs["cell_type"] = pd.Categorical(
            stream_adata.obs["cell_type"].astype(str),
            categories=present_cell_types,
            ordered=True,
        )
        stream_adata.uns["cell_type_colors"] = [PHENO_COLORS[label] for label in present_cell_types]

    graph_dir = OUT_DIR / "protein_umap_graph"
    graph_dir.mkdir(parents=True, exist_ok=True)
    graph = build_velocity_graph_bundle(
        stream_adata,
        output_dir=str(graph_dir),
        n_neighbors=30,
        n_jobs=1,
        reuse_neighbors=False,
        vkey="velocity",
        preferred_basis="umap",
        strict_preferred=True,
    )

    plot_outputs = {}
    warnings = list(graph.get("warnings", []))
    for color_key, suffix in [("cell_type", "celltype"), ("time_label", "time")]:
        out_dir = OUT_DIR / f"protein_umap_stream_{suffix}"
        out_dir.mkdir(parents=True, exist_ok=True)
        plot = plot_velocity_stream_bundle(
            stream_adata,
            output_dir=str(out_dir),
            dim_reduction="umap",
            color_key=color_key,
            vkey="velocity",
        )
        warnings.extend(plot.get("warnings", []))
        src = Path(plot.get("artifacts", {}).get("velocity_stream", out_dir / "velocity_stream_umap.png"))
        dst_png = FIG_DIR / f"fig01f_protein_umap_pushforward_velocity_by_{suffix}.png"
        if src.exists():
            shutil.copyfile(src, dst_png)
        plot_outputs[suffix] = {
            "color_key": color_key,
            "source_png": str(src),
            "png": str(dst_png),
            "plot_payload": plot,
        }

    diagnostics = {
        "readout_vs_observed_protein_rmse": float(np.sqrt(np.mean((q_pred - protein) ** 2))),
        "readout_vs_observed_protein_mae": float(np.mean(np.abs(q_pred - protein))),
        "mean_rna_velocity_norm": float(np.linalg.norm(rna_velocity, axis=1).mean()),
        "mean_pushforward_protein_velocity_norm": float(np.linalg.norm(protein_velocity, axis=1).mean()),
        "median_pushforward_protein_velocity_norm": float(np.median(np.linalg.norm(protein_velocity, axis=1))),
        "max_pushforward_protein_velocity_norm": float(np.linalg.norm(protein_velocity, axis=1).max()),
    }

    coord_table = OUT_DIR / "protein_umap_pushforward_velocity_coordinates.csv"
    pd.DataFrame(
        {
            "cell_id": model_adata.obs_names.astype(str),
            "protein_umap_1": protein_umap[:, 0],
            "protein_umap_2": protein_umap[:, 1],
            "cell_type": stream_adata.obs["cell_type"].astype(str).to_numpy() if "cell_type" in stream_adata.obs else "",
            "time_point_processed": time_col[:, 0],
            "pushforward_protein_velocity_norm": np.linalg.norm(protein_velocity, axis=1),
            "rna_velocity_norm": np.linalg.norm(rna_velocity, axis=1),
            "readout_observed_protein_error": np.linalg.norm(q_pred - protein, axis=1),
        }
    ).to_csv(coord_table, index=False)

    summary = {
        "analysis": "cite_pcfm_protein_umap_pushforward_velocity_stream",
        "skill": "downstream-scvelo-stream",
        "evidence_type": "local model-induced protein readout flow diagnostic",
        "trained_model_h5ad": str(TRAINED_MODEL_H5AD),
        "algorithm_path": str(ALGORITHM_PATH),
        "basis": f"observed protein {emb_name} fitted from X_protein_pca",
        "state_space": "observed X_protein_pca[:, :30]",
        "velocity_space": "dq/dt = J_r h_eta(r,t) v_theta([r, zero_protein, t])[:, :50] + partial_t h_eta(r,t)",
        "anchor_note": "Velocity vectors are assigned to observed paired cells on the observed protein PCA/UMAP manifold; q=h_eta(r,t) is used only to derive the local readout velocity.",
        "not_measured_velocity_note": "CITE-seq has no measured splicing-style protein velocity here; this is a CITE-PCFM-induced protein readout velocity.",
        "figures": {
            "celltype": plot_outputs["celltype"]["png"],
            "time": plot_outputs["time"]["png"],
        },
        "tables": {
            "coordinates": str(coord_table),
        },
        "graph_payload": graph,
        "plots": plot_outputs,
        "diagnostics": diagnostics,
        "warnings": warnings
        + [
            "This stream plot is a local flow diagnostic, not continuous trajectory evidence.",
            "Protein velocity is push-forward readout velocity from the RNA ODE, not an independently observed protein velocity.",
        ],
    }
    manifest_path = OUT_DIR / "protein_umap_pushforward_velocity_manifest.json"
    manifest_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print(json.dumps(
        {
            "celltype_png": plot_outputs["celltype"]["png"],
            "time_png": plot_outputs["time"]["png"],
            "manifest": str(manifest_path),
            "coordinate_table": str(coord_table),
            "diagnostics": diagnostics,
        },
        indent=2,
    ))
    return summary


if __name__ == "__main__":
    main()
