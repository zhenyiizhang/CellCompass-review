#!/usr/bin/env python3
"""RNA-protein regulatory map for CITE-PCFM.

This script interrogates the learned differentiable relationship

    dr/dt = f_theta(r, t)
    q = h_eta(r, t)
    dq/dt = d h_eta / d r * dr/dt + d h_eta / d t

where r is the 50D RNA latent state and q is the 30D protein latent state.
Gene/protein-space interpretation is made through documented linear
projections:

    r ~= scale_R(X_RNA @ C_R)
    ADT ~= decoder_P(q)

The analysis computes RNA-program directional derivatives instead of a giant
gene-by-protein Jacobian. This keeps the biological question explicit:
which RNA programs change the learned protein readout or protein velocity?
"""

from __future__ import annotations

import importlib.util
import json
import os
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
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from multistageot_style import PHENO_COLORS, PHENO_ORDER
from protein_phenotype_readout import ProteinPhenotypeReadout, train_mlp_protein_phenotype_readout


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "mathtext.sf": "Arial",
        "mathtext.cal": "Arial",
        "mathtext.tt": "Arial",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


ROOT = Path("/Users/zhenyizhang/Documents/GitHub/CytoBridge-agent-runtime-v2-20260309/multi-omics")
REPO_ROOT = ROOT.parent
CYTOBRIDGE_SRC = REPO_ROOT / "CytoBridge-main"
ALGORITHM_PATH = Path("/Users/zhenyizhang/.cellcompass/training_algorithms/cite_paired_coord_fm/algorithm.py")
DATA_PATH = Path("/Users/zhenyizhang/.cellcompass/algorithm_benchmarks/datasets/gse305370_citeseq_stage2_10k/data.h5ad")
TRAINED_MODEL_H5AD = ROOT / "training_runs/20260526T174204321821Z__final__cite_paired_coord_fm__campaign_20260-trial_20260526-gse305370_ci/artifacts/trained_model.h5ad"

FIG_DIR = ROOT / "figures/cite_pcfm_deep_story/biology_landscape_discovery"
OUT_DIR = ROOT / "downstream/cite_pcfm_deep_story/biology_landscape_discovery_rna_protein_regulatory_map"

RNA_DIM = 50
PROTEIN_DIM = 30
MODEL_SAMPLE_N = 1400
FINITE_DIFF_EPS = 0.5
RANDOM_STATE = 20260528

RNA_PROGRAMS = {
    "Myeloid": {
        "genes": ["MPO", "CTSG", "AZU1", "ELANE", "PRTN3", "CEBPA", "SPI1", "LYZ", "LST1", "S100A8", "S100A9"],
        "color": "#D62728",
    },
    "Erythroid": {
        "genes": ["GATA1", "KLF1", "HBA1", "HBA2", "HBB", "ALAS2", "GYPB"],
        "color": "#F28E2B",
    },
    "Mk": {
        "genes": ["ITGA2B", "PF4", "PPBP", "GP9", "GP1BA", "VWF", "NFE2", "FLI1", "TUBB1", "GFI1B", "RUNX1"],
        "color": "#7B61B3",
    },
    "HSC": {
        "genes": ["CD34", "HLF", "MEIS1", "MECOM", "HOXA9", "PROM1", "SPINK2", "AVP", "MLLT3", "PROCR"],
        "color": "#2F855A",
    },
}

MARKER_GROUPS = {
    "myeloid_adt": ["CD11b", "CD11c", "CD13", "CD16", "CD64", "CD115", "CD172a", "CD88", "LOX-1", "HLA-DR"],
    "ery_mk_adt": ["CD71", "CD41", "CD42b", "CD62P", "CD105", "CD49b"],
    "stem_progenitor_adt": ["CD49f", "CD123", "CD45RA", "CD105", "CD45"],
    "hla_activation_adt": ["HLA-DR", "HLA-A-B-C", "CD54", "CD62L", "CD45"],
}

OBSERVED_TREND_MARKERS = {
    "HSC": ["HLA-DR", "CD62L", "CD31", "CD123", "CD32", "CD41", "CD71"],
    "Erythroid": ["CD41", "CD32", "CD71", "CD62L", "HLA-DR", "CD13"],
    "Mk": ["CD41", "CD62P", "CD32", "CD49f", "HLA-DR", "CD45RA"],
    "Myeloid": ["CD11c", "CD64", "CD45RA", "CD71", "CD41", "CD88"],
}


def ensure_import_paths() -> None:
    if CYTOBRIDGE_SRC.exists() and str(CYTOBRIDGE_SRC) not in sys.path:
        sys.path.insert(0, str(CYTOBRIDGE_SRC))


def load_readout_input() -> Callable[[torch.Tensor, torch.Tensor, int, float], torch.Tensor]:
    spec = importlib.util.spec_from_file_location("cite_pcfm_algorithm_runtime", ALGORITHM_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load algorithm module from {ALGORITHM_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, "_readout_input")


def clean_marker(name: str) -> str:
    return str(name).replace("prot_", "").replace("rna_", "")


def feature_masks(adata: ad.AnnData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    var_names = np.asarray([str(v) for v in adata.var_names])
    rna_mask = np.asarray([v.startswith("rna_") for v in var_names])
    protein_mask = np.asarray([v.startswith("prot_") for v in var_names])
    return var_names, rna_mask, protein_mask


def dense(x) -> np.ndarray:
    return x.toarray() if sparse.issparse(x) else np.asarray(x)


def usable_protein_marker(marker: str) -> bool:
    bad = ["Mouse-Ig", "Rat-Ig", "IgG"]
    return not any(token in marker for token in bad)


def weighted_average(x: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    if weights is None:
        return np.nanmean(x, axis=0)
    w = np.asarray(weights, dtype=float).reshape(-1)
    w = np.clip(w, 0, np.inf)
    if w.sum() <= 0:
        return np.nanmean(x, axis=0)
    return np.average(x, axis=0, weights=w / w.sum())


def projection_audit_and_scales(adata: ad.AnnData) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    var_names, rna_mask, protein_mask = feature_masks(adata)
    pcs = np.asarray(adata.varm["PCs"], dtype=np.float64)

    x_rna = adata.X[:, rna_mask]
    rna_unscaled = np.asarray(x_rna @ pcs[rna_mask, :RNA_DIM], dtype=np.float64)
    rna_latent = np.asarray(adata.obsm["X_rna_svd"], dtype=np.float64)[:, :RNA_DIM]
    rna_corr = [np.corrcoef(rna_unscaled[:, j], rna_latent[:, j])[0, 1] for j in range(RNA_DIM)]
    rna_std = np.std(rna_unscaled, axis=0)
    rna_std = np.where(rna_std > 1e-8, rna_std, 1.0)

    x_protein = dense(adata.X[:, protein_mask]).astype(np.float64)
    protein_unscaled = np.asarray(x_protein @ pcs[protein_mask, RNA_DIM : RNA_DIM + PROTEIN_DIM], dtype=np.float64)
    protein_latent = np.asarray(adata.obsm["X_protein_pca"], dtype=np.float64)[:, :PROTEIN_DIM]
    protein_corr = [np.corrcoef(protein_unscaled[:, j], protein_latent[:, j])[0, 1] for j in range(PROTEIN_DIM)]

    prep = dict(adata.uns.get("cytobridge_preprocessing", {}))
    audit = {
        "rna_latent_relation": "X_rna_svd is standardized TruncatedSVD scores; unscaled scores are X_RNA @ varm['PCs'][:, :50]",
        "protein_latent_relation": "X_protein_pca is standardized PCA scores; unscaled scores are X_ADT @ varm['PCs'][:, 50:80]",
        "rna_projection_mean_abs_corr": float(np.mean(np.abs(rna_corr))),
        "rna_projection_median_abs_corr": float(np.median(np.abs(rna_corr))),
        "protein_projection_mean_abs_corr": float(np.mean(np.abs(protein_corr))),
        "protein_projection_median_abs_corr": float(np.median(np.abs(protein_corr))),
        "rna_svd_explained_variance_ratio_sum": float(prep.get("rna_svd_explained_variance_ratio_sum", np.nan)),
        "protein_pca_explained_variance_ratio_sum": float(prep.get("protein_pca_explained_variance_ratio_sum", np.nan)),
    }
    return audit, rna_std, x_protein


def fit_protein_decoder(adata: ad.AnnData, x_protein: np.ndarray, marker_names: list[str]) -> tuple[Ridge, pd.DataFrame, dict[str, float]]:
    q = np.asarray(adata.obsm["X_protein_pca"], dtype=np.float64)[:, :PROTEIN_DIM]
    decoder = Ridge(alpha=1e-6)
    decoder.fit(q, x_protein)
    pred = decoder.predict(q)
    rows = []
    for j, marker in enumerate(marker_names):
        rows.append({"protein_marker": marker, "r2": float(r2_score(x_protein[:, j], pred[:, j]))})
    r2_df = pd.DataFrame(rows)
    summary = {
        "decoder": "Ridge(alpha=1e-6): ADT ~= intercept + X_protein_pca @ coef.T",
        "mean_marker_r2": float(r2_df["r2"].mean()),
        "median_marker_r2": float(r2_df["r2"].median()),
        "min_marker_r2": float(r2_df["r2"].min()),
    }
    return decoder, r2_df, summary


def build_program_directions(
    adata: ad.AnnData,
    rna_component_std: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray], pd.DataFrame]:
    var_names, rna_mask, _ = feature_masks(adata)
    rna_var_names = var_names[rna_mask]
    rna_lookup = {clean_marker(name): i for i, name in enumerate(rna_var_names)}
    pcs_rna = np.asarray(adata.varm["PCs"], dtype=np.float64)[rna_mask, :RNA_DIM]

    all_genes = []
    gene_rows = []
    for program, spec in RNA_PROGRAMS.items():
        for gene in spec["genes"]:
            available = gene in rna_lookup
            gene_rows.append({"program": program, "gene": gene, "available": bool(available)})
            if available:
                all_genes.append(gene)
    unique_genes = sorted(set(all_genes))
    selected_cols = np.asarray([rna_lookup[g] for g in unique_genes], dtype=int)
    x_sel = dense(adata.X[:, rna_mask][:, selected_cols]).astype(np.float64)
    gene_mean = x_sel.mean(axis=0)
    gene_std = x_sel.std(axis=0)
    gene_std = np.where(gene_std > 1e-8, gene_std, 1.0)
    x_z = (x_sel - gene_mean[None, :]) / gene_std[None, :]
    selected_lookup = {gene: i for i, gene in enumerate(unique_genes)}

    score_df = pd.DataFrame({"obs_name": np.asarray(adata.obs_names.astype(str))})
    directions: dict[str, np.ndarray] = {}
    direction_rows = []
    for program, spec in RNA_PROGRAMS.items():
        genes = [g for g in spec["genes"] if g in selected_lookup]
        if not genes:
            raise ValueError(f"No available genes for program {program}")
        idx = np.asarray([selected_lookup[g] for g in genes], dtype=int)
        score_df[program] = x_z[:, idx].mean(axis=1)
        rna_cols = np.asarray([rna_lookup[g] for g in genes], dtype=int)
        # Direction for +1 z-score in each program gene, averaged across genes:
        # dr_scaled/dz_g = std_gene * C_g / std_component.
        dr_dz = (gene_std[idx, None] * pcs_rna[rna_cols, :]) / rna_component_std[None, :]
        direction = dr_dz.mean(axis=0)
        directions[program] = direction.astype(np.float32)
        direction_rows.append(
            {
                "program": program,
                "available_genes": ",".join(genes),
                "n_available_genes": len(genes),
                "missing_genes": ",".join([g for g in spec["genes"] if g not in selected_lookup]),
                "latent_direction_norm": float(np.linalg.norm(direction)),
            }
        )
    return score_df, pd.DataFrame(gene_rows), directions, pd.DataFrame(direction_rows)


def class_probability_matrix(readout: ProteinPhenotypeReadout, x: np.ndarray) -> np.ndarray:
    return readout.predict_proba_ordered(x)


def select_model_sample(adata: ad.AnnData, n: int) -> np.ndarray:
    rng = np.random.default_rng(RANDOM_STATE)
    if adata.n_obs <= n:
        return np.arange(adata.n_obs)
    groups = adata.obs[["time_point_processed", "cell_type"]].astype(str).agg("|".join, axis=1).to_numpy()
    selected = []
    per_group = max(8, n // max(1, len(np.unique(groups))))
    for group in np.unique(groups):
        idx = np.where(groups == group)[0]
        take = min(idx.shape[0], per_group)
        selected.extend(rng.choice(idx, size=take, replace=False).tolist())
    selected = np.asarray(selected, dtype=int)
    if selected.shape[0] < n:
        rest = np.setdiff1d(np.arange(adata.n_obs), selected, assume_unique=False)
        extra = rng.choice(rest, size=n - selected.shape[0], replace=False)
        selected = np.concatenate([selected, extra])
    elif selected.shape[0] > n:
        selected = rng.choice(selected, size=n, replace=False)
    return np.sort(selected)


def model_readout_q(
    model,
    readout_input: Callable[[torch.Tensor, torch.Tensor, int, float], torch.Tensor],
    rna: np.ndarray,
    time_col: np.ndarray,
    batch_size: int = 384,
) -> np.ndarray:
    model = model.cpu().eval()
    n_fourier = int(getattr(model, "cite_pcfm_time_fourier_features", 0))
    time_fourier_scale = float(getattr(model, "cite_pcfm_time_fourier_scale", 1.0))
    out = []
    with torch.no_grad():
        for start in range(0, rna.shape[0], batch_size):
            end = min(start + batch_size, rna.shape[0])
            r = torch.as_tensor(rna[start:end], dtype=torch.float32)
            t = torch.as_tensor(time_col[start:end], dtype=torch.float32)
            q = model.protein_readout(readout_input(r, t, n_fourier, time_fourier_scale))
            out.append(q.detach().cpu().numpy().astype(np.float32))
    return np.vstack(out)


def model_pushforward_vq(
    model,
    readout_input: Callable[[torch.Tensor, torch.Tensor, int, float], torch.Tensor],
    rna: np.ndarray,
    time_col: np.ndarray,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    model = model.cpu().eval()
    for param in model.parameters():
        param.requires_grad_(False)
    n_fourier = int(getattr(model, "cite_pcfm_time_fourier_features", 0))
    time_fourier_scale = float(getattr(model, "cite_pcfm_time_fourier_scale", 1.0))
    q_parts = []
    vq_parts = []
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
    return np.vstack(q_parts), np.vstack(vq_parts)


def decoder_delta(decoder: Ridge, delta_q: np.ndarray) -> np.ndarray:
    return np.asarray(delta_q, dtype=np.float64) @ np.asarray(decoder.coef_, dtype=np.float64).T


def finite_difference_model_effects(
    model,
    readout_input,
    rna: np.ndarray,
    time_col: np.ndarray,
    directions: dict[str, np.ndarray],
    decoder: Ridge,
    marker_names: list[str],
    phenotype_clf: ProteinPhenotypeReadout,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    static_rows = []
    velocity_rows = []
    basin_rows = []
    for program, direction in directions.items():
        u = direction.reshape(1, -1).astype(np.float32)
        r_plus = rna + FINITE_DIFF_EPS * u
        r_minus = rna - FINITE_DIFF_EPS * u

        q_plus = model_readout_q(model, readout_input, r_plus, time_col)
        q_minus = model_readout_q(model, readout_input, r_minus, time_col)
        dq = (q_plus - q_minus) / (2.0 * FINITE_DIFF_EPS)
        dadt = decoder_delta(decoder, dq)

        _, vq_plus = model_pushforward_vq(model, readout_input, r_plus, time_col)
        _, vq_minus = model_pushforward_vq(model, readout_input, r_minus, time_col)
        dvq = (vq_plus - vq_minus) / (2.0 * FINITE_DIFF_EPS)
        dvadt = decoder_delta(decoder, dvq)

        p_plus = class_probability_matrix(phenotype_clf, q_plus)
        p_minus = class_probability_matrix(phenotype_clf, q_minus)
        dp = (p_plus - p_minus) / (2.0 * FINITE_DIFF_EPS)

        for j, marker in enumerate(marker_names):
            static_rows.append(
                {
                    "program": program,
                    "protein_marker": marker,
                    "mean_dADT_dprogram": float(np.mean(dadt[:, j])),
                    "median_dADT_dprogram": float(np.median(dadt[:, j])),
                    "mean_abs_dADT_dprogram": float(np.mean(np.abs(dadt[:, j]))),
                }
            )
            velocity_rows.append(
                {
                    "program": program,
                    "protein_marker": marker,
                    "mean_dADT_velocity_dprogram": float(np.mean(dvadt[:, j])),
                    "median_dADT_velocity_dprogram": float(np.median(dvadt[:, j])),
                    "mean_abs_dADT_velocity_dprogram": float(np.mean(np.abs(dvadt[:, j]))),
                }
            )
        for k, phenotype in enumerate(PHENO_ORDER):
            basin_rows.append(
                {
                    "program": program,
                    "phenotype": phenotype,
                    "mean_dprotein_basin_probability_dprogram": float(np.mean(dp[:, k])),
                    "mean_abs_dprotein_basin_probability_dprogram": float(np.mean(np.abs(dp[:, k]))),
                }
            )
    return pd.DataFrame(static_rows), pd.DataFrame(velocity_rows), pd.DataFrame(basin_rows)


def observed_correlations(
    score_df: pd.DataFrame,
    x_protein: np.ndarray,
    marker_names: list[str],
    obs: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for program in RNA_PROGRAMS:
        x = score_df[program].to_numpy(dtype=float)
        for j, marker in enumerate(marker_names):
            if np.nanstd(x) < 1e-8 or np.nanstd(x_protein[:, j]) < 1e-8:
                rho, pval, pear, pear_p = np.nan, np.nan, np.nan, np.nan
            else:
                rho, pval = spearmanr(x, x_protein[:, j])
                pear, pear_p = pearsonr(x, x_protein[:, j])
            rows.append(
                {
                    "program": program,
                    "protein_marker": marker,
                    "scope": "all_cells",
                    "spearman_rho": float(rho),
                    "spearman_p": float(pval),
                    "pearson_r": float(pear),
                    "pearson_p": float(pear_p),
                    "n_cells": int(x.shape[0]),
                }
            )
        for time in sorted(obs["time_point_processed"].astype(float).unique()):
            mask = np.isclose(obs["time_point_processed"].astype(float).to_numpy(), time)
            xt = x[mask]
            for j, marker in enumerate(marker_names):
                yt = x_protein[mask, j]
                if xt.shape[0] < 10 or np.nanstd(xt) < 1e-8 or np.nanstd(yt) < 1e-8:
                    continue
                rho, pval = spearmanr(xt, yt)
                rows.append(
                    {
                        "program": program,
                        "protein_marker": marker,
                        "scope": f"time_{time:g}",
                        "spearman_rho": float(rho),
                        "spearman_p": float(pval),
                        "pearson_r": np.nan,
                        "pearson_p": np.nan,
                        "n_cells": int(xt.shape[0]),
                    }
                )
    return pd.DataFrame(rows)


def observed_program_marker_trends(
    score_df: pd.DataFrame,
    x_protein: np.ndarray,
    marker_names: list[str],
    n_bins: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    marker_to_idx = {m: i for i, m in enumerate(marker_names)}
    z_protein = StandardScaler().fit_transform(x_protein)
    trend_rows = []
    shift_rows = []
    for program, wanted_markers in OBSERVED_TREND_MARKERS.items():
        scores = score_df[program].to_numpy(dtype=float)
        ranked_bins = pd.qcut(pd.Series(scores).rank(method="first"), q=n_bins, labels=False).to_numpy()
        q25, q75 = np.quantile(scores, [0.25, 0.75])
        low_mask = scores <= q25
        high_mask = scores >= q75
        for marker in wanted_markers:
            if marker not in marker_to_idx:
                continue
            vals = z_protein[:, marker_to_idx[marker]]
            rho, pval = spearmanr(scores, vals)
            high = vals[high_mask]
            low = vals[low_mask]
            effect = float(np.mean(high) - np.mean(low))
            se = float(np.sqrt(np.var(high, ddof=1) / high.shape[0] + np.var(low, ddof=1) / low.shape[0]))
            shift_rows.append(
                {
                    "program": program,
                    "protein_marker": marker,
                    "spearman_rho": float(rho),
                    "spearman_p": float(pval),
                    "high_quartile_mean_zADT": float(np.mean(high)),
                    "low_quartile_mean_zADT": float(np.mean(low)),
                    "high_minus_low_zADT": effect,
                    "high_minus_low_se": se,
                    "high_minus_low_ci95_low": effect - 1.96 * se,
                    "high_minus_low_ci95_high": effect + 1.96 * se,
                    "n_high": int(high.shape[0]),
                    "n_low": int(low.shape[0]),
                }
            )
            for b in range(n_bins):
                mask = ranked_bins == b
                bin_vals = vals[mask]
                trend_rows.append(
                    {
                        "program": program,
                        "protein_marker": marker,
                        "score_bin": int(b + 1),
                        "score_bin_percentile_mid": float((b + 0.5) * 100.0 / n_bins),
                        "mean_rna_program_score": float(np.mean(scores[mask])),
                        "mean_zADT": float(np.mean(bin_vals)),
                        "sem_zADT": float(np.std(bin_vals, ddof=1) / np.sqrt(bin_vals.shape[0])),
                        "n_cells": int(mask.sum()),
                        "spearman_rho_all_cells": float(rho),
                    }
                )
    return pd.DataFrame(trend_rows), pd.DataFrame(shift_rows)


def marker_panel(marker_names: list[str]) -> list[str]:
    keep = []
    seen = set()
    for markers in MARKER_GROUPS.values():
        for marker in markers:
            if marker in marker_names and marker not in seen:
                keep.append(marker)
                seen.add(marker)
    return keep


def top_rows_per_program(df: pd.DataFrame, value_col: str, n: int = 7) -> pd.DataFrame:
    parts = []
    for program in RNA_PROGRAMS:
        sub = df[df["program"] == program].copy()
        sub = sub[sub["protein_marker"].map(usable_protein_marker)]
        sub["abs_value"] = sub[value_col].abs()
        parts.append(sub.sort_values("abs_value", ascending=False).head(n))
    return pd.concat(parts, ignore_index=True)


def style_axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#CBD5E1")
    ax.tick_params(labelsize=8)
    ax.grid(axis="x", color="#E2E8F0", lw=0.65, alpha=0.8)


def plot_projection_audit(audit: dict[str, object], decoder_summary: dict[str, float], r2_df: pd.DataFrame, out_png: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.2), constrained_layout=True)
    ax = axes[0]
    ax.axis("off")
    equation = (
        "Learned CITE-PCFM relation\n\n"
        "dr/dt = f_theta(r,t)\n"
        "q = h_eta(r,t)\n"
        "dq/dt = J_h(r,t) f_theta(r,t) + partial_t h_eta(r,t)\n\n"
        "Gene/program -> ADT derivative\n"
        "dADT/du = D_P J_h E_R u"
    )
    ax.text(0.02, 0.96, equation, va="top", fontsize=10.4, color="#0F172A", linespacing=1.35)

    ax = axes[1]
    labels = ["RNA SVD\nprojection", "protein PCA\nprojection"]
    vals = [audit["rna_projection_mean_abs_corr"], audit["protein_projection_mean_abs_corr"]]
    ax.bar(labels, vals, color=["#2563EB", "#7B61B3"], alpha=0.88)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("mean absolute component correlation", fontsize=9)
    ax.set_title("latent projection audit", loc="left", fontsize=12, weight="bold")
    ax.text(
        0.05,
        0.12,
        f"RNA EVR {audit['rna_svd_explained_variance_ratio_sum']:.3f}\nprotein EVR {audit['protein_pca_explained_variance_ratio_sum']:.3f}",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#334155",
        bbox={"boxstyle": "round,pad=0.14", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
    )
    style_axis(ax)

    ax = axes[2]
    ax.hist(r2_df["r2"], bins=np.linspace(0, 1, 24), color="#0F766E", alpha=0.82)
    ax.axvline(decoder_summary["median_marker_r2"], color="#0F172A", lw=1.4)
    ax.set_xlim(0, 1)
    ax.set_xlabel("per-marker decoder R2", fontsize=9)
    ax.set_ylabel("ADT markers", fontsize=9)
    ax.set_title("protein latent -> ADT decoder", loc="left", fontsize=12, weight="bold")
    ax.text(
        0.05,
        0.92,
        f"median R2 {decoder_summary['median_marker_r2']:.3f}\nmean R2 {decoder_summary['mean_marker_r2']:.3f}",
        transform=ax.transAxes,
        va="top",
        fontsize=8.5,
        color="#334155",
        bbox={"boxstyle": "round,pad=0.14", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
    )
    style_axis(ax)
    fig.suptitle("Projection audit for RNA-protein regulatory derivatives", x=0.01, ha="left", fontsize=15.5, weight="bold")
    fig.savefig(out_png, dpi=330, bbox_inches="tight", facecolor="white")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_lollipop(df: pd.DataFrame, value_col: str, title: str, xlabel: str, out_png: Path) -> None:
    plot_df = top_rows_per_program(df, value_col, n=7)
    plot_df["label"] = plot_df["program"] + " -> " + plot_df["protein_marker"]
    plot_df = plot_df.iloc[::-1].reset_index(drop=True)
    y = np.arange(plot_df.shape[0])
    fig, ax = plt.subplots(figsize=(10.8, 8.4), constrained_layout=True)
    ax.axvline(0, color="#CBD5E1", lw=1.0)
    for i, row in plot_df.iterrows():
        color = RNA_PROGRAMS[row["program"]]["color"]
        ax.hlines(y[i], 0, row[value_col], color=color, lw=2.8, alpha=0.42)
        ax.scatter(row[value_col], y[i], s=52, color=color, edgecolor="white", linewidth=0.8, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["label"], fontsize=8.2)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_title(title, loc="left", fontsize=13.2, weight="bold")
    lim = max(0.02, float(np.nanmax(np.abs(plot_df[value_col]))) * 1.25)
    ax.set_xlim(-lim, lim)
    style_axis(ax)
    fig.savefig(out_png, dpi=330, bbox_inches="tight", facecolor="white")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_basin_dotplot(basin_df: pd.DataFrame, out_png: Path) -> None:
    programs = ["HSC", "Erythroid", "Mk", "Myeloid"]
    plot_order = PHENO_ORDER[::-1]
    signed_vals = basin_df["mean_dprotein_basin_probability_dprogram"].to_numpy(dtype=float) * 1e3
    vmax = max(0.10, float(np.nanmax(np.abs(signed_vals))) * 1.24)
    fig, axes = plt.subplots(1, 4, figsize=(15.2, 5.3), sharex=True, sharey=True, constrained_layout=True)
    for ax, program in zip(axes, programs):
        sub = basin_df[basin_df["program"] == program].set_index("phenotype").reindex(plot_order).reset_index()
        vals = sub["mean_dprotein_basin_probability_dprogram"].to_numpy(dtype=float) * 1e3
        y = np.arange(len(plot_order))
        colors = ["#DC2626" if v >= 0 else "#2563EB" for v in vals]
        ax.axvspan(0, vmax, color="#FEE2E2", alpha=0.20, zorder=0)
        ax.axvspan(-vmax, 0, color="#DBEAFE", alpha=0.20, zorder=0)
        ax.axvline(0, color="#64748B", lw=1.0)
        ax.barh(y, vals, color=colors, alpha=0.78, height=0.58, edgecolor="white", linewidth=0.6)
        for yi, val in zip(y, vals):
            if abs(val) < 0.015:
                continue
            ax.text(
                val + (0.018 if val >= 0 else -0.018),
                yi,
                f"{val:+.2f}",
                ha="left" if val >= 0 else "right",
                va="center",
                fontsize=7.0,
                color="#0F172A",
            )
        ax.set_title(f"+ {program} RNA direction", fontsize=10.8, weight="bold", color=RNA_PROGRAMS[program]["color"])
        ax.set_yticks(y)
        ax.set_yticklabels(plot_order, fontsize=8.0)
        ax.set_xlim(-vmax, vmax)
        ax.set_xlabel("dP(basin)/dprogram\nx10^-3", fontsize=7.9)
        ax.grid(axis="x", color="#E2E8F0", lw=0.65, alpha=0.85)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#CBD5E1")
        ax.tick_params(labelsize=7.6)
        ax.text(
            0.02,
            0.02,
            "left: less likely\nright: more likely",
            transform=ax.transAxes,
            fontsize=6.8,
            color="#475569",
            va="bottom",
        )
    axes[0].set_ylabel("protein phenotype basin", fontsize=8.6)
    fig.suptitle(
        "Counterfactual RNA-program push on protein phenotype basins\n"
        "For each observed RNA state r: compare h_eta(r + eps*u, t) versus h_eta(r - eps*u, t); positive bars mean the basin becomes more likely along +RNA-program direction.",
        x=0.01,
        ha="left",
        fontsize=13.2,
        weight="bold",
    )
    fig.savefig(out_png, dpi=330, bbox_inches="tight", facecolor="white")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_observed_vs_model(corr_df: pd.DataFrame, static_df: pd.DataFrame, marker_names: list[str], out_png: Path) -> None:
    corr_all = corr_df[corr_df["scope"] == "all_cells"][["program", "protein_marker", "spearman_rho"]]
    merged = corr_all.merge(static_df[["program", "protein_marker", "mean_dADT_dprogram"]], on=["program", "protein_marker"], how="inner")
    merged = merged[merged["protein_marker"].map(usable_protein_marker)].copy()
    # Keep interpretable marker panel and high observed/model signal markers.
    panel = set(marker_panel(marker_names))
    merged["score"] = merged["spearman_rho"].abs() + merged["mean_dADT_dprogram"].abs() / (merged["mean_dADT_dprogram"].abs().max() + 1e-8)
    selected = pd.concat(
        [
            merged[merged["protein_marker"].isin(panel)],
            merged.sort_values("score", ascending=False).head(50),
        ],
        ignore_index=True,
    ).drop_duplicates(["program", "protein_marker"])

    fig, ax = plt.subplots(figsize=(7.2, 6.0), constrained_layout=True)
    for program in RNA_PROGRAMS:
        sub = selected[selected["program"] == program]
        ax.scatter(
            sub["spearman_rho"],
            sub["mean_dADT_dprogram"],
            s=48,
            color=RNA_PROGRAMS[program]["color"],
            alpha=0.76,
            edgecolor="white",
            linewidth=0.6,
            label=program,
        )
    if selected.shape[0] > 3:
        rho, pval = spearmanr(selected["spearman_rho"], selected["mean_dADT_dprogram"])
    else:
        rho, pval = np.nan, np.nan
    ax.axhline(0, color="#CBD5E1", lw=0.9)
    ax.axvline(0, color="#CBD5E1", lw=0.9)
    ax.set_xlabel("observed paired RNA-program vs ADT Spearman rho", fontsize=9)
    ax.set_ylabel("learned readout derivative dADT/dprogram", fontsize=9)
    ax.set_title("Observed correlation versus learned readout derivative", loc="left", fontsize=13.0, weight="bold")
    ax.text(
        0.04,
        0.96,
        f"selected marker pairs\nSpearman={rho:+.2f}, p={pval:.1e}",
        transform=ax.transAxes,
        va="top",
        fontsize=8.3,
        color="#334155",
        bbox={"boxstyle": "round,pad=0.14", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
    )
    ax.legend(frameon=False, fontsize=8.0, loc="lower right")
    style_axis(ax)
    fig.savefig(out_png, dpi=330, bbox_inches="tight", facecolor="white")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_observed_binned_trends(trend_df: pd.DataFrame, out_png: Path) -> None:
    panel_order = ["HSC", "Erythroid", "Mk", "Myeloid"]
    marker_palette = [
        "#0F766E",
        "#2563EB",
        "#9333EA",
        "#EA580C",
        "#DC2626",
        "#64748B",
        "#111827",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.6), sharex=True, sharey=False, constrained_layout=True)
    axes = axes.ravel()
    for ax, program in zip(axes, panel_order):
        sub = trend_df[trend_df["program"] == program].copy()
        markers = [m for m in OBSERVED_TREND_MARKERS[program] if m in set(sub["protein_marker"])]
        for i, marker in enumerate(markers):
            d = sub[sub["protein_marker"] == marker].sort_values("score_bin")
            if d.empty:
                continue
            rho = float(d["spearman_rho_all_cells"].iloc[0])
            color = marker_palette[i % len(marker_palette)]
            ax.plot(
                d["score_bin_percentile_mid"],
                d["mean_zADT"],
                marker="o",
                ms=3.8,
                lw=1.8,
                color=color,
                label=f"{marker} ({rho:+.2f})",
            )
            ax.fill_between(
                d["score_bin_percentile_mid"].to_numpy(dtype=float),
                (d["mean_zADT"] - d["sem_zADT"]).to_numpy(dtype=float),
                (d["mean_zADT"] + d["sem_zADT"]).to_numpy(dtype=float),
                color=color,
                alpha=0.11,
                linewidth=0,
            )
        ax.axhline(0, color="#CBD5E1", lw=0.9)
        ax.set_title(f"{program} RNA program bins", loc="left", fontsize=11.2, weight="bold", color=RNA_PROGRAMS[program]["color"])
        ax.set_xlabel("RNA program score percentile bin", fontsize=8.6)
        ax.set_ylabel("mean observed ADT z-score", fontsize=8.6)
        ax.legend(frameon=False, fontsize=7.0, ncol=1, loc="best", handlelength=1.4)
        style_axis(ax)
    fig.suptitle(
        "Observed paired CITE sanity check: RNA program score versus ADT abundance\n"
        "Real cells only; each line is an observed protein marker binned by RNA program score",
        x=0.01,
        ha="left",
        fontsize=14.2,
        weight="bold",
    )
    fig.savefig(out_png, dpi=330, bbox_inches="tight", facecolor="white")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_observed_high_low_shifts(shift_df: pd.DataFrame, out_png: Path) -> None:
    panel_order = ["HSC", "Erythroid", "Mk", "Myeloid"]
    fig, axes = plt.subplots(1, 4, figsize=(15.4, 4.8), sharex=True, constrained_layout=True)
    for ax, program in zip(axes, panel_order):
        sub = shift_df[shift_df["program"] == program].copy()
        marker_order = [m for m in OBSERVED_TREND_MARKERS[program] if m in set(sub["protein_marker"])]
        sub["marker_order"] = sub["protein_marker"].map({m: i for i, m in enumerate(marker_order)})
        sub = sub.sort_values("marker_order").reset_index(drop=True)
        y = np.arange(sub.shape[0])[::-1]
        color = RNA_PROGRAMS[program]["color"]
        ax.axvline(0, color="#CBD5E1", lw=1.0)
        for row_i, row in sub.iterrows():
            yi = y[row_i]
            x = float(row["high_minus_low_zADT"])
            ci_low = float(row["high_minus_low_ci95_low"])
            ci_high = float(row["high_minus_low_ci95_high"])
            ax.hlines(yi, 0, x, color=color, alpha=0.34, lw=3.2)
            ax.hlines(yi, ci_low, ci_high, color="#334155", alpha=0.36, lw=1.0)
            ax.scatter(x, yi, s=42, color=color, edgecolor="white", linewidth=0.8, zorder=3)
            ax.text(
                x + (0.06 if x >= 0 else -0.06),
                yi,
                f"{x:+.2f}",
                ha="left" if x >= 0 else "right",
                va="center",
                fontsize=6.9,
                color="#334155",
            )
        labels = [f"{row.protein_marker}\nrho {row.spearman_rho:+.2f}" for row in sub.itertuples()]
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7.1)
        ax.set_title(program, fontsize=11.2, weight="bold", color=color)
        ax.set_xlabel("high RNA-program Q4 - low Q1\nobserved ADT z-score", fontsize=7.8)
        style_axis(ax)
    max_abs = max(0.8, float(np.nanmax(np.abs(shift_df["high_minus_low_zADT"]))) * 1.18)
    for ax in axes:
        ax.set_xlim(-max_abs, max_abs)
    fig.suptitle(
        "Observed high-versus-low RNA program marker shifts\n"
        "Real paired CITE cells only; error bars are approximate 95% CI for Q4-Q1 difference",
        x=0.01,
        ha="left",
        fontsize=14.2,
        weight="bold",
    )
    fig.savefig(out_png, dpi=330, bbox_inches="tight", facecolor="white")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> dict[str, object]:
    ensure_import_paths()
    from CytoBridge.utils import load_model_from_adata

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(DATA_PATH)
    model_adata = ad.read_h5ad(TRAINED_MODEL_H5AD)
    if adata.n_obs != model_adata.n_obs:
        raise ValueError(f"data/model n_obs mismatch: {adata.n_obs} vs {model_adata.n_obs}")

    var_names, _, protein_mask = feature_masks(adata)
    marker_names = [clean_marker(v) for v in var_names[protein_mask]]

    audit, rna_component_std, x_protein = projection_audit_and_scales(adata)
    decoder, decoder_r2_df, decoder_summary = fit_protein_decoder(adata, x_protein, marker_names)
    score_df, gene_availability_df, directions, direction_df = build_program_directions(adata, rna_component_std)

    labels = adata.obs["cell_type"].astype(str).to_numpy()
    protein_latent = np.asarray(adata.obsm["X_protein_pca"], dtype=np.float32)[:, :PROTEIN_DIM]
    phenotype_clf, classifier_metrics, classifier_confusion = train_mlp_protein_phenotype_readout(
        protein_latent, labels, PHENO_ORDER, RANDOM_STATE
    )

    sample_idx = select_model_sample(model_adata, MODEL_SAMPLE_N)
    rna_sample = np.asarray(model_adata.obsm["X_rna_svd"], dtype=np.float32)[sample_idx, :RNA_DIM]
    time_sample = np.asarray(model_adata.obs["time_point_processed"], dtype=np.float32)[sample_idx].reshape(-1, 1)

    model = load_model_from_adata(model_adata)
    readout_input = load_readout_input()
    static_df, velocity_df, basin_df = finite_difference_model_effects(
        model,
        readout_input,
        rna_sample,
        time_sample,
        directions,
        decoder,
        marker_names,
        phenotype_clf,
    )
    corr_df = observed_correlations(score_df, x_protein, marker_names, adata.obs)
    observed_trend_df, observed_shift_df = observed_program_marker_trends(score_df, x_protein, marker_names)

    projection_path = OUT_DIR / "projection_audit.json"
    decoder_r2_path = OUT_DIR / "protein_decoder_marker_r2.csv"
    gene_path = OUT_DIR / "rna_program_gene_availability.csv"
    direction_path = OUT_DIR / "rna_program_latent_directions.csv"
    corr_path = OUT_DIR / "observed_rna_program_adt_correlations.csv"
    observed_trend_path = OUT_DIR / "observed_program_adt_binned_trends.csv"
    observed_shift_path = OUT_DIR / "observed_program_adt_high_low_shifts.csv"
    static_path = OUT_DIR / "learned_readout_adt_sensitivity.csv"
    velocity_path = OUT_DIR / "learned_velocity_adt_sensitivity.csv"
    basin_path = OUT_DIR / "learned_program_basin_probability_sensitivity.csv"
    confusion_path = OUT_DIR / "protein_phenotype_readout_confusion.csv"
    projection_path.write_text(json.dumps({"projection": audit, "protein_decoder": decoder_summary}, indent=2), encoding="utf-8")
    decoder_r2_df.to_csv(decoder_r2_path, index=False)
    gene_availability_df.to_csv(gene_path, index=False)
    direction_df.to_csv(direction_path, index=False)
    corr_df.to_csv(corr_path, index=False)
    observed_trend_df.to_csv(observed_trend_path, index=False)
    observed_shift_df.to_csv(observed_shift_path, index=False)
    static_df.to_csv(static_path, index=False)
    velocity_df.to_csv(velocity_path, index=False)
    basin_df.to_csv(basin_path, index=False)
    classifier_confusion.to_csv(confusion_path)

    fig_a = FIG_DIR / "fig06a_rna_protein_projection_audit.png"
    fig_b = FIG_DIR / "fig06b_observed_rna_adt_correlation.png"
    fig_c = FIG_DIR / "fig06c_learned_readout_jacobian_adt_sensitivity.png"
    fig_d = FIG_DIR / "fig06d_learned_protein_velocity_sensitivity.png"
    fig_e = FIG_DIR / "fig06e_program_effect_on_protein_basins.png"
    fig_f = FIG_DIR / "fig06f_observed_vs_learned_rna_protein_coupling.png"
    fig_g = FIG_DIR / "fig06g_observed_rna_program_adt_binned_trends.png"
    fig_h = FIG_DIR / "fig06h_observed_rna_program_high_low_marker_shifts.png"

    plot_projection_audit(audit, decoder_summary, decoder_r2_df, fig_a)
    plot_lollipop(
        corr_df[corr_df["scope"] == "all_cells"],
        "spearman_rho",
        "Observed paired RNA-program to ADT correlations",
        "Spearman rho across observed paired CITE cells",
        fig_b,
    )
    plot_lollipop(
        static_df,
        "mean_dADT_dprogram",
        "Learned protein readout derivative",
        "mean dADT/dRNA-program direction",
        fig_c,
    )
    plot_lollipop(
        velocity_df,
        "mean_dADT_velocity_dprogram",
        "Learned protein velocity derivative",
        "mean d(ADT velocity)/dRNA-program direction",
        fig_d,
    )
    plot_basin_dotplot(basin_df, fig_e)
    plot_observed_vs_model(corr_df, static_df, marker_names, fig_f)
    plot_observed_binned_trends(observed_trend_df, fig_g)
    plot_observed_high_low_shifts(observed_shift_df, fig_h)

    manifest = {
        "analysis": "rna_protein_regulatory_map",
        "data_h5ad": str(DATA_PATH),
        "trained_model_h5ad": str(TRAINED_MODEL_H5AD),
        "algorithm_path": str(ALGORITHM_PATH),
        "figures": {
            "projection_audit": str(fig_a),
            "observed_correlation": str(fig_b),
            "readout_jacobian": str(fig_c),
            "velocity_sensitivity": str(fig_d),
            "basin_sensitivity": str(fig_e),
            "observed_vs_learned": str(fig_f),
            "observed_binned_trends": str(fig_g),
            "observed_high_low_shifts": str(fig_h),
        },
        "tables": {
            "projection_audit": str(projection_path),
            "protein_decoder_marker_r2": str(decoder_r2_path),
            "rna_program_gene_availability": str(gene_path),
            "rna_program_latent_directions": str(direction_path),
            "observed_rna_program_adt_correlations": str(corr_path),
            "observed_program_adt_binned_trends": str(observed_trend_path),
            "observed_program_adt_high_low_shifts": str(observed_shift_path),
            "learned_readout_adt_sensitivity": str(static_path),
            "learned_velocity_adt_sensitivity": str(velocity_path),
            "learned_program_basin_probability_sensitivity": str(basin_path),
            "protein_phenotype_readout_confusion": str(confusion_path),
        },
        "semantic_contract": {
            "model_equation": "dr/dt=f_theta(r,t), q=h_eta(r,t), dq/dt=J_h f_theta + partial_t h_eta",
            "rna_program_direction": "mean latent response to +1 z-score in available program genes, using TruncatedSVD loadings and recovered component scales",
            "adt_decoder": "Ridge decoder from observed X_protein_pca to original ADT values",
            "readout_derivative": "finite difference d h_eta(r,t) / d RNA-program direction",
            "velocity_derivative": "finite difference d pushforward_protein_velocity(r,t) / d RNA-program direction",
            "not_claimed": "not causal perturbation and not observed protein velocity; this is a model-derived local regulatory map",
        },
        "projection_audit": audit,
        "protein_decoder_summary": decoder_summary,
        "protein_phenotype_classifier": classifier_metrics,
        "finite_difference_eps": FINITE_DIFF_EPS,
        "model_sample_n": int(sample_idx.shape[0]),
    }
    manifest_path = OUT_DIR / "rna_protein_regulatory_map_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    main()
