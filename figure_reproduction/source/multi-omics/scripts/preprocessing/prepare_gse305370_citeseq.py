#!/usr/bin/env python3
"""Prepare GSE305370 CITE-seq multi-omics AnnData for CytoBridge.

Input: paper-intake downloaded all-donor/all-day h5ad from GEO.
Output: paper_input.h5ad and gse305370_citeseq_preprocessed.h5ad with:
  - obs['time_point_processed'] integer day order 0..4 for day 2/3/4/7/10
  - obsm['X_latent'] concatenated standardized RNA TruncatedSVD and ADT PCA features
  - modality-specific latent embeddings in obsm
  - provenance in uns

The source matrix is already transformed (not raw counts): RNA entries are sparse
positive normalized/log-like values and ADT entries are dense centered values.
Therefore this script does not normalize/log1p again; it derives a compact latent
representation while preserving the original transformed feature matrix.
"""
from __future__ import annotations

import json
import shutil
import hashlib
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import StandardScaler

OUTPUT_DIR = Path('/Users/zhenyizhang/Documents/GitHub/CytoBridge-agent-runtime-v2-20260309/multi-omics')
DOWNLOAD_DIR = OUTPUT_DIR / 'paper_intake' / 'downloads'
SOURCE_H5AD = DOWNLOAD_DIR / 'GSE305370_citeseq_alldonors_alldays.h5ad'
PAPER_INPUT_H5AD = OUTPUT_DIR / 'paper_intake' / 'paper_input.h5ad'
PREPROCESSED_H5AD = OUTPUT_DIR / 'paper_intake' / 'gse305370_citeseq_preprocessed.h5ad'
PROVENANCE_JSON = OUTPUT_DIR / 'paper_intake' / 'gse305370_citeseq_preprocessing_provenance.json'

TIME_ORDER = ['day 2', 'day 3', 'day 4', 'day 7', 'day 10']
TIME_TO_INT = {name: i for i, name in enumerate(TIME_ORDER)}
RANDOM_STATE = 0
RNA_COMPONENTS = 50
PROTEIN_COMPONENTS = 30


def sha256_file(path: Path, block: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(block), b''):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    if not SOURCE_H5AD.exists():
        raise FileNotFoundError(f'Missing downloaded source h5ad: {SOURCE_H5AD}')
    print(f'Reading {SOURCE_H5AD}')
    adata = ad.read_h5ad(SOURCE_H5AD)
    print(f'Loaded shape={adata.shape}')

    required_obs = {'time', 'donor', 'cell_type'}
    missing = sorted(required_obs - set(adata.obs.columns))
    if missing:
        raise ValueError(f'Missing required obs columns: {missing}')
    if 'feature_types' not in adata.var.columns:
        raise ValueError("Missing var['feature_types']; cannot split RNA/ADT modalities")

    # Clean indexes for safe h5ad writing.
    adata.obs_names_make_unique()
    adata.var_names_make_unique()
    adata.obs.index.name = None
    adata.var.index.name = None

    # Establish biologically meaningful temporal order from experimental days.
    time_str = adata.obs['time'].astype(str)
    unknown = sorted(set(time_str) - set(TIME_TO_INT))
    if unknown:
        raise ValueError(f'Unexpected time labels: {unknown}')
    adata.obs['time_point_processed'] = pd.Categorical(
        time_str.map(TIME_TO_INT).astype(int), categories=list(range(len(TIME_ORDER))), ordered=True
    )
    adata.obs['time_day_numeric'] = time_str.str.extract(r'(\d+)')[0].astype(int).values
    adata.obs['label'] = adata.obs['cell_type'].astype(str).values
    adata.obs['donor'] = adata.obs['donor'].astype(str).values

    feature_types = adata.var['feature_types'].astype(str)
    rna_mask = np.asarray(feature_types == 'Gene Expression')
    protein_mask = np.asarray(feature_types == 'Antibody Capture')
    if rna_mask.sum() == 0 or protein_mask.sum() == 0:
        raise ValueError(f'Expected both RNA and antibody features, got RNA={rna_mask.sum()}, protein={protein_mask.sum()}')

    X = adata.X.tocsr() if sparse.issparse(adata.X) else sparse.csr_matrix(adata.X)
    X_rna = X[:, rna_mask]
    X_protein = X[:, protein_mask]

    # RNA is high-dimensional and sparse. TruncatedSVD preserves sparse input and gives
    # an explicit linear reconstruction basis stored in varm['PCs'].
    n_rna = min(RNA_COMPONENTS, X_rna.shape[1] - 1, X_rna.shape[0] - 1)
    print(f'Fitting RNA TruncatedSVD n_components={n_rna}')
    rna_svd = TruncatedSVD(n_components=n_rna, random_state=RANDOM_STATE)
    X_rna_svd = rna_svd.fit_transform(X_rna).astype(np.float32)
    rna_scaler = StandardScaler(with_mean=True, with_std=True)
    X_rna_latent = rna_scaler.fit_transform(X_rna_svd).astype(np.float32)

    # ADT panel is small (103 features) and dense/centered; PCA is appropriate.
    n_prot = min(PROTEIN_COMPONENTS, X_protein.shape[1], X_protein.shape[0] - 1)
    print(f'Fitting protein PCA n_components={n_prot}')
    X_protein_dense = X_protein.toarray().astype(np.float32)
    protein_pca = PCA(n_components=n_prot, random_state=RANDOM_STATE)
    X_protein_pca = protein_pca.fit_transform(X_protein_dense).astype(np.float32)
    prot_scaler = StandardScaler(with_mean=True, with_std=True)
    X_protein_latent = prot_scaler.fit_transform(X_protein_pca).astype(np.float32)

    X_latent = np.concatenate([X_rna_latent, X_protein_latent], axis=1).astype(np.float32)
    if not np.isfinite(X_latent).all():
        raise ValueError('X_latent contains NaN or Inf')

    adata.obsm['X_rna_svd'] = X_rna_latent
    adata.obsm['X_protein_pca'] = X_protein_latent
    adata.obsm['X_latent'] = X_latent

    # Store a combined linear loading matrix where RNA components occupy RNA rows and
    # protein components occupy ADT rows. This gives an audit trail for gene/protein links.
    pcs = np.zeros((adata.n_vars, n_rna + n_prot), dtype=np.float32)
    pcs[np.where(rna_mask)[0], :n_rna] = rna_svd.components_.T.astype(np.float32)
    pcs[np.where(protein_mask)[0], n_rna:] = protein_pca.components_.T.astype(np.float32)
    adata.varm['PCs'] = pcs

    adata.uns['cytobridge_preprocessing'] = {
        'source_h5ad': str(SOURCE_H5AD),
        'source_sha256': sha256_file(SOURCE_H5AD),
        'selected_asset_url': 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE305nnn/GSE305370/suppl/GSE305370_citeseq_alldonors_alldays.h5ad',
        'paper_source': '/Users/zhenyizhang/Documents/GitHub/CytoBridge-agent-runtime-v2-20260309/multi-omics_paper/science.adi8577-3.pdf',
        'geo_accession': 'GSE305370',
        'time_key': 'time',
        'time_order': TIME_ORDER,
        'label_key': 'cell_type',
        'donor_key': 'donor',
        'rna_features': int(rna_mask.sum()),
        'protein_features': int(protein_mask.sum()),
        'X_before_preprocessing': 'Downloaded h5ad contains already-transformed CITE-seq matrix: sparse normalized/log-like RNA plus dense centered ADT/protein values; no normalize_total/log1p was applied here.',
        'X_latent': 'Concatenated standardized 50D RNA TruncatedSVD and 30D ADT PCA features.',
        'rna_svd_explained_variance_ratio_sum': float(np.sum(rna_svd.explained_variance_ratio_)),
        'protein_pca_explained_variance_ratio_sum': float(np.sum(protein_pca.explained_variance_ratio_)),
        'random_state': RANDOM_STATE,
    }

    counts = adata.obs.groupby(['time', 'donor', 'cell_type'], observed=True).size().reset_index(name='n_cells')
    provenance = dict(adata.uns['cytobridge_preprocessing'])
    provenance['shape'] = list(adata.shape)
    provenance['time_counts'] = adata.obs['time'].value_counts().to_dict()
    provenance['donor_counts'] = adata.obs['donor'].value_counts().to_dict()
    provenance['cell_type_counts'] = adata.obs['cell_type'].value_counts().to_dict()
    provenance['rna_svd_explained_variance_ratio_first10'] = rna_svd.explained_variance_ratio_[:10].astype(float).tolist()
    provenance['protein_pca_explained_variance_ratio_first10'] = protein_pca.explained_variance_ratio_[:10].astype(float).tolist()

    PREPROCESSED_H5AD.parent.mkdir(parents=True, exist_ok=True)
    print(f'Writing {PREPROCESSED_H5AD}')
    adata.write_h5ad(PREPROCESSED_H5AD, compression='gzip')
    # Paper-intake convention expected by workflow tools.
    print(f'Copying to {PAPER_INPUT_H5AD}')
    shutil.copy2(PREPROCESSED_H5AD, PAPER_INPUT_H5AD)
    provenance['preprocessed_h5ad'] = str(PREPROCESSED_H5AD)
    provenance['paper_input_h5ad'] = str(PAPER_INPUT_H5AD)
    provenance['output_sha256'] = sha256_file(PREPROCESSED_H5AD)
    provenance['paper_input_sha256'] = sha256_file(PAPER_INPUT_H5AD)
    PROVENANCE_JSON.write_text(json.dumps(provenance, indent=2, sort_keys=True), encoding='utf-8')
    print(json.dumps({
        'preprocessed_h5ad': str(PREPROCESSED_H5AD),
        'paper_input_h5ad': str(PAPER_INPUT_H5AD),
        'shape': list(adata.shape),
        'latent_shape': list(adata.obsm['X_latent'].shape),
        'time_counts': provenance['time_counts'],
        'provenance_json': str(PROVENANCE_JSON),
    }, indent=2))


if __name__ == '__main__':
    main()
