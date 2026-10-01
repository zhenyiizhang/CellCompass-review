#!/usr/bin/env python3
"""Prepare Cook/Vanderhyden GSE147405 EMT time-course scRNA-seq for CytoBridge.

Outputs:
- preprocessed/emt_cook_vanderhyden_all_conditions.h5ad
- preprocessed/conditions/<cell_line>_<stimulus>.h5ad for 12 induction time courses
- preprocessed/preprocessing_manifest.json

The script downloads the 12 GEO processed UMI matrices and metadata sidecars when absent,
filters to induction labels 0d, 8h, 1d, 3d, 7d, computes a single normalization/HVG/PCA
framework across all conditions, adds canonical time_point_processed and X_latent, and adds
consistent EMT/growth gene-program scores.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from urllib.request import urlopen, Request

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from sklearn.preprocessing import StandardScaler

OUTPUT_DIR = Path(globals().get("output_dir", "/Users/zhenyizhang/Documents/GitHub/CytoBridge-agent-runtime-v2-20260309/emt"))
PAPER_DIR = OUTPUT_DIR / "paper_intake"
DOWNLOAD_DIR = PAPER_DIR / "downloads"
PREP_DIR = OUTPUT_DIR / "preprocessed"
COND_DIR = PREP_DIR / "conditions"
PREP_DIR.mkdir(parents=True, exist_ok=True)
COND_DIR.mkdir(parents=True, exist_ok=True)
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

GEO_BASE = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE147nnn/GSE147405/suppl"
CELL_LINES = ["A549", "DU145", "MCF7", "OVCA420"]
STIMULI = ["EGF", "TGFB1", "TNF"]
ALLOWED_TIMES = ["0d", "8h", "1d", "3d", "7d"]
TIME_TO_INDEX = {"0d": 0, "8h": 1, "1d": 2, "3d": 3, "7d": 4}
TIME_TO_DAYS = {"0d": 0.0, "8h": 8.0 / 24.0, "1d": 1.0, "3d": 3.0, "7d": 7.0}

# Compact, transparent gene-program framework.  Scores are computed from the same global
# log-normalized matrix for all conditions as mean z-scored expression over available genes.
GENE_PROGRAMS: Dict[str, List[str]] = {
    "epithelial_score": [
        "EPCAM", "KRT8", "KRT18", "KRT19", "KRT7", "CDH1", "CLDN3", "CLDN4",
        "CLDN7", "OCLN", "TJP1", "MUC1", "DSP", "DSG2", "JUP", "ESRP1", "GRHL2", "OVOL2",
    ],
    "hybrid_score": [
        "ITGA6", "ITGB4", "LAMC2", "LAMB3", "KRT14", "KRT17", "KRT5", "VIM",
        "CDH1", "EPCAM", "JAG1", "NOTCH1", "AREG", "TGFBI", "SERPINE1", "MMP7", "S100A2",
    ],
    "mesenchymal_score": [
        "VIM", "CDH2", "FN1", "ZEB1", "ZEB2", "SNAI1", "SNAI2", "TWIST1",
        "TWIST2", "TAGLN", "ACTA2", "COL1A1", "COL1A2", "COL3A1", "ITGA5", "MMP2", "MMP9", "SPARC",
    ],
    "emt_core_score": [
        "VIM", "FN1", "CDH2", "ZEB1", "ZEB2", "SNAI1", "SNAI2", "TWIST1", "TGFBI", "SERPINE1",
        "MMP2", "MMP9", "ITGA5", "SPARC", "COL1A1", "COL1A2",
    ],
    "early_immediate_response_score": [
        "FOS", "JUN", "JUNB", "JUND", "FOSB", "EGR1", "EGR2", "EGR3", "ATF3", "IER2",
        "DUSP1", "DUSP5", "DUSP6", "NR4A1", "NR4A2", "KLF4", "KLF6", "BTG2",
    ],
    "early_inflammatory_tnf_score": [
        "NFKBIA", "NFKBIZ", "TNFAIP3", "CXCL8", "CXCL1", "CXCL2", "CXCL3", "IL6",
        "ICAM1", "BIRC3", "RELA", "JUNB", "CCL2", "CCL20",
    ],
    "intermediate_partial_emt_score": [
        "TGFBI", "SERPINE1", "LAMC2", "LAMB3", "ITGA6", "ITGB4", "JAG1", "NOTCH1",
        "MMP7", "S100A2", "KRT14", "KRT17", "AREG", "EREG", "PLAUR", "ANXA1",
    ],
    "late_ecm_mesenchymal_score": [
        "COL1A1", "COL1A2", "COL3A1", "COL5A1", "COL5A2", "FN1", "SPARC", "THBS1",
        "ITGA5", "TAGLN", "ACTA2", "MMP2", "MMP9", "LOX", "VCAN", "TIMP1",
    ],
    "growth_cell_cycle_score": [
        "MKI67", "TOP2A", "PCNA", "MCM2", "MCM3", "MCM4", "MCM5", "MCM6", "MCM7",
        "CDK1", "CCNB1", "CCNB2", "CDC20", "AURKA", "AURKB", "UBE2C", "TYMS", "RRM2",
    ],
}
PHASE_PROGRAMS = {
    "early": ["early_immediate_response_score", "early_inflammatory_tnf_score"],
    "intermediate": ["intermediate_partial_emt_score", "hybrid_score"],
    "late": ["late_ecm_mesenchymal_score", "mesenchymal_score", "emt_core_score"],
    "growth": ["growth_cell_cycle_score"],
}


def sha256_file(path: Path, max_mb: int | None = None) -> str:
    h = hashlib.sha256()
    limit = None if max_mb is None else max_mb * 1024 * 1024
    read = 0
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            if limit is not None and read + len(chunk) > limit:
                chunk = chunk[: max(0, limit - read)]
            h.update(chunk)
            read += len(chunk)
            if limit is not None and read >= limit:
                break
    return h.hexdigest()


def download_if_missing(url: str, dest: Path, retries: int = 4) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            req = Request(url, headers={"User-Agent": "CytoBridge-paper-intake/1.0"})
            with urlopen(req, timeout=60) as r, tmp.open("wb") as f:
                while True:
                    block = r.read(1024 * 1024)
                    if not block:
                        break
                    f.write(block)
            tmp.replace(dest)
            return
        except Exception as exc:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            if attempt == retries:
                raise RuntimeError(f"Failed to download {url}: {exc}") from exc
            time.sleep(2 * attempt)


def expected_files() -> List[Tuple[str, str, Path, str]]:
    out = []
    for cl in CELL_LINES:
        for stim in STIMULI:
            for kind, suffix in [("matrix", "UMI_matrix"), ("metadata", "metadata")]:
                name = f"GSE147405_{cl}_{stim}_TimeCourse_{suffix}.csv.gz"
                out.append((cl, stim, DOWNLOAD_DIR / name, f"{GEO_BASE}/{name}"))
    return out


def ensure_downloads() -> None:
    for cl, stim, path, url in expected_files():
        print(f"[download-check] {path.name}", flush=True)
        download_if_missing(url, path)


def read_condition(cl: str, stim: str) -> ad.AnnData:
    matrix_path = DOWNLOAD_DIR / f"GSE147405_{cl}_{stim}_TimeCourse_UMI_matrix.csv.gz"
    meta_path = DOWNLOAD_DIR / f"GSE147405_{cl}_{stim}_TimeCourse_metadata.csv.gz"
    if not matrix_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Missing matrix or metadata for {cl}_{stim}")

    meta = pd.read_csv(meta_path, low_memory=False)
    if "Unnamed: 0" in meta.columns:
        meta = meta.rename(columns={"Unnamed: 0": "barcode"})
    elif meta.columns[0] not in {"barcode", "Barcode"}:
        meta = meta.rename(columns={meta.columns[0]: "barcode"})
    if "Barcode" in meta.columns and "barcode" not in meta.columns:
        meta = meta.rename(columns={"Barcode": "barcode"})
    meta["barcode"] = meta["barcode"].astype(str)
    meta = meta.set_index("barcode", drop=False)
    meta.index = meta.index.astype(str)

    # Induction-only scope requested by user: remove *_rm/withdrawal/reversal labels.
    meta["Time"] = meta["Time"].astype(str)
    keep = meta["Time"].isin(ALLOWED_TIMES)
    if "Doublet" in meta.columns:
        keep &= meta["Doublet"].astype(str).str.lower().eq("singlet")
    meta = meta.loc[keep].copy()
    if meta.empty:
        raise ValueError(f"No induction cells remain for {cl}_{stim}")

    print(f"[read] {cl}_{stim}: loading dense GEO CSV {matrix_path.name}", flush=True)
    df = pd.read_csv(matrix_path, index_col=0, low_memory=False)
    # Columns are cells/barcodes, rows are genes. Subset while still dense, then convert to sparse.
    df.columns = df.columns.astype(str)
    common_cells = [bc for bc in meta.index if bc in df.columns]
    missing = len(meta.index) - len(common_cells)
    if len(common_cells) == 0:
        raise ValueError(f"No overlap between metadata and count columns for {cl}_{stim}")
    if missing:
        print(f"[warn] {cl}_{stim}: {missing} filtered metadata cells absent from matrix", flush=True)
    meta = meta.loc[common_cells].copy()
    sub = df.loc[:, common_cells]
    # Preserve integer counts in a sparse matrix; AnnData orientation is cells x genes.
    X = sparse.csr_matrix(sub.T.astype(np.float32).values)
    var = pd.DataFrame(index=pd.Index(sub.index.astype(str), name="gene"))
    obs = meta.copy()
    obs.index.name = None
    obs["cell_line"] = cl
    obs["stimulus"] = stim
    obs["condition"] = f"{cl}_{stim}"
    obs["source_file"] = matrix_path.name
    obs["dataset"] = "GSE147405_Cook_Vanderhyden_TimeCourse"
    obs["paper_subset"] = "in_vitro_EMT_induction_timecourse"
    obs["time_point_raw"] = obs["Time"].astype(str)
    obs["time_point"] = obs["Time"].astype(str)
    obs["original_time_label"] = obs["Time"].astype(str)
    obs["numeric_time_days"] = obs["time_point"].map(TIME_TO_DAYS).astype(float)
    obs["time_point_processed"] = obs["time_point"].map(TIME_TO_INDEX).astype(int)
    a = ad.AnnData(X=X, obs=obs, var=var)
    a.var_names_make_unique()
    a.obs_names_make_unique()
    return a


def score_programs(adata: ad.AnnData) -> Dict[str, List[str]]:
    # Use z-scored global log-normalized expression over all retained cells. This makes scores
    # directly comparable across conditions while keeping gene identities auditable.
    genes_upper = pd.Index([str(g).upper() for g in adata.var_names])
    gene_to_idx = {g: i for i, g in enumerate(genes_upper)}
    X = adata.X
    if sparse.issparse(X):
        X = X.tocsr()
    used: Dict[str, List[str]] = {}
    for score_name, gene_list in GENE_PROGRAMS.items():
        idx = []
        matched = []
        for g in gene_list:
            gi = gene_to_idx.get(g.upper())
            if gi is not None:
                idx.append(gi)
                matched.append(adata.var_names[gi])
        used[score_name] = matched
        if len(idx) == 0:
            adata.obs[score_name] = 0.0
            continue
        M = X[:, idx]
        if sparse.issparse(M):
            M = M.toarray()
        M = np.asarray(M, dtype=np.float32)
        mu = M.mean(axis=0, keepdims=True)
        sd = M.std(axis=0, keepdims=True)
        sd[sd < 1e-6] = 1.0
        adata.obs[score_name] = ((M - mu) / sd).mean(axis=1).astype(np.float32)
    adata.uns["emt_gene_programs"] = {k: [str(x) for x in v] for k, v in GENE_PROGRAMS.items()}
    adata.uns["emt_gene_programs_matched"] = {k: [str(x) for x in v] for k, v in used.items()}
    adata.uns["emt_phase_programs"] = PHASE_PROGRAMS
    return used


def main() -> None:
    ensure_downloads()
    parts = []
    for cl in CELL_LINES:
        for stim in STIMULI:
            parts.append(read_condition(cl, stim))
    print("[concat] combining conditions", flush=True)
    combined = ad.concat(parts, join="inner", merge="same", label=None, index_unique=None)
    combined.obs_names_make_unique()
    combined.var_names_make_unique()
    combined.layers["counts"] = combined.X.copy()

    # Basic QC fields and filtering. The metadata is already singlets; keep all non-empty cells/genes.
    sc.pp.calculate_qc_metrics(combined, inplace=True, percent_top=None, log1p=False)
    sc.pp.filter_cells(combined, min_counts=1)
    sc.pp.filter_genes(combined, min_cells=3)

    # Global normalized/log expression matrix for gene-program scores and a shared PCA latent basis.
    sc.pp.normalize_total(combined, target_sum=1e4)
    sc.pp.log1p(combined)
    combined.uns["X_semantics"] = "X is log1p(CP10k-normalized UMI counts); raw integer UMI counts are in layer 'counts'."

    print("[score] EMT/growth gene programs", flush=True)
    matched = score_programs(combined)

    print("[hvg/pca] shared HVG/PCA framework", flush=True)
    sc.pp.highly_variable_genes(combined, n_top_genes=2000, flavor="seurat", subset=False)
    hvg_mask = combined.var["highly_variable"].values.astype(bool)
    if int(hvg_mask.sum()) < 50:
        raise RuntimeError("Too few HVGs selected for PCA")
    X_hvg = combined[:, hvg_mask].X
    if sparse.issparse(X_hvg):
        X_hvg_dense = X_hvg.toarray().astype(np.float32)
    else:
        X_hvg_dense = np.asarray(X_hvg, dtype=np.float32)
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_scaled = scaler.fit_transform(X_hvg_dense)
    from sklearn.decomposition import PCA
    n_comps = min(50, X_scaled.shape[0] - 1, X_scaled.shape[1])
    pca = PCA(n_components=n_comps, svd_solver="randomized", random_state=0)
    X_pca = pca.fit_transform(X_scaled).astype(np.float32)
    combined.obsm["X_pca"] = X_pca
    combined.obsm["X_latent"] = X_pca
    combined.uns["pca"] = {
        "variance_ratio": pca.explained_variance_ratio_.astype(float).tolist(),
        "variance": pca.explained_variance_.astype(float).tolist(),
        "params": {"zero_center": True, "use_highly_variable": True, "n_comps": int(n_comps)},
    }
    combined.varm["PCs"] = np.zeros((combined.n_vars, n_comps), dtype=np.float32)
    combined.varm["PCs"][hvg_mask, :] = pca.components_.T.astype(np.float32)
    combined.uns["X_latent_semantics"] = (
        "Shared 50-dimensional PCA on 2000 globally selected HVGs from log1p(CP10k) expression; "
        "same basis is reused for every condition-specific model. Gene-space interpretation uses full log expression X and counts layer."
    )
    combined.uns["pca_reconstruction_proxy"] = {
        "cumulative_explained_variance_ratio": float(np.sum(pca.explained_variance_ratio_)),
        "n_hvgs": int(hvg_mask.sum()),
        "note": "Explained variance of shared PCA on HVG-scaled log expression; full gene-level scores are not reconstructed from PCA."
    }

    # Make categorical metadata save-safe.
    for col in ["cell_line", "stimulus", "condition", "source_file", "dataset", "paper_subset", "time_point", "original_time_label", "Phase", "Cluster", "Treatment", "CellLine", "Time"]:
        if col in combined.obs.columns:
            combined.obs[col] = combined.obs[col].astype("category")
    combined.obs["time_point_processed"] = combined.obs["time_point_processed"].astype(int)

    all_path = PREP_DIR / "emt_cook_vanderhyden_all_conditions.h5ad"
    print(f"[write] {all_path}", flush=True)
    combined.write_h5ad(all_path, compression="gzip")

    condition_paths = {}
    condition_summary = []
    for condition in sorted(combined.obs["condition"].astype(str).unique()):
        sub = combined[combined.obs["condition"].astype(str).values == condition].copy()
        # Keep per-condition time indices as the same canonical 0..4 mapping.
        cpath = COND_DIR / f"{condition}.h5ad"
        sub.write_h5ad(cpath, compression="gzip")
        condition_paths[condition] = str(cpath)
        counts = sub.obs["time_point"].astype(str).value_counts().reindex(ALLOWED_TIMES).fillna(0).astype(int).to_dict()
        condition_summary.append({
            "condition": condition,
            "cell_line": str(sub.obs["cell_line"].iloc[0]),
            "stimulus": str(sub.obs["stimulus"].iloc[0]),
            "n_cells": int(sub.n_obs),
            "n_genes": int(sub.n_vars),
            "time_counts": counts,
            "path": str(cpath),
        })

    manifest = {
        "paper_source": str(OUTPUT_DIR / "emt.pdf"),
        "geo_accession": "GSE147405",
        "doi": "10.1038/s41467-020-16066-2",
        "input_download_dir": str(DOWNLOAD_DIR),
        "combined_h5ad": str(all_path),
        "condition_h5ads": condition_paths,
        "conditions": condition_summary,
        "time_mapping": TIME_TO_INDEX,
        "numeric_time_days": TIME_TO_DAYS,
        "allowed_time_labels": ALLOWED_TIMES,
        "excluded_time_scope": "Removal/withdrawal/reversal labels such as *_rm were excluded by filtering Time to 0d,8h,1d,3d,7d.",
        "x_semantics": combined.uns["X_semantics"],
        "x_latent_semantics": combined.uns["X_latent_semantics"],
        "pca_reconstruction_proxy": combined.uns["pca_reconstruction_proxy"],
        "gene_programs": GENE_PROGRAMS,
        "gene_programs_matched": {k: [str(x) for x in v] for k, v in matched.items()},
        "phase_programs": PHASE_PROGRAMS,
        "files_sha256_first_16mb": {p.name: sha256_file(p, max_mb=16) for _, _, p, _ in expected_files() if p.exists()},
    }
    mpath = PREP_DIR / "preprocessing_manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"combined_h5ad": str(all_path), "manifest": str(mpath), "n_cells": int(combined.n_obs), "n_genes": int(combined.n_vars), "conditions": len(condition_summary)}, indent=2))


if __name__ == "__main__":
    main()
