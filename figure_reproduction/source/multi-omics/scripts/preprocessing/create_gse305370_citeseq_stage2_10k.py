#!/usr/bin/env python
"""Create a 10k stratified GSE305370 CITE-seq benchmark subset for Stage 2 claim validation.

The source paper_input.h5ad is already script-backed and contains biologically
ordered time_point_processed plus paired X_latent = [X_rna_svd, X_protein_pca].
This script only creates a deterministic stratified subset that preserves time,
cell_type, donor, and paired RNA/protein latent matrices for campaign-scale claim
validation.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import scanpy as sc

SOURCE = Path('/Users/zhenyizhang/Documents/GitHub/CytoBridge-agent-runtime-v2-20260309/multi-omics/paper_intake/paper_input.h5ad')
OUT = Path('/Users/zhenyizhang/Documents/GitHub/CytoBridge-agent-runtime-v2-20260309/multi-omics/paper/debug_subsets/GSE305370_citeseq_stage2_10000.h5ad')
N_PER_TIME = 2000
SEED = 20260525


def main() -> None:
    adata = sc.read_h5ad(SOURCE)
    required_obs = ['time_point_processed', 'cell_type', 'donor', 'label']
    for key in required_obs:
        if key not in adata.obs:
            raise KeyError(f'missing obs[{key!r}]')
    for key in ['X_latent', 'X_rna_svd', 'X_protein_pca']:
        if key not in adata.obsm:
            raise KeyError(f'missing obsm[{key!r}]')
    times = sorted(adata.obs['time_point_processed'].astype(float).unique())
    rng = np.random.default_rng(SEED)
    selected: list[np.ndarray] = []
    # Stratify within each time by cell_type x donor when possible; allocate by
    # observed proportions with at least one cell for non-empty strata.
    for t in times:
        mask_t = adata.obs['time_point_processed'].astype(float).to_numpy() == float(t)
        idx_t = np.where(mask_t)[0]
        obs_t = adata.obs.iloc[idx_t]
        strata = obs_t[['cell_type', 'donor']].astype(str).agg('||'.join, axis=1).to_numpy()
        unique, counts = np.unique(strata, return_counts=True)
        target_total = min(N_PER_TIME, len(idx_t))
        raw = counts / counts.sum() * target_total
        alloc = np.floor(raw).astype(int)
        # Preserve rare non-empty strata whenever budget allows.
        if target_total >= len(unique):
            alloc = np.maximum(alloc, 1)
        # Adjust to exact target_total.
        while alloc.sum() > target_total:
            j = int(np.argmax(alloc - 1))
            if alloc[j] > 1:
                alloc[j] -= 1
            else:
                break
        remainder = target_total - int(alloc.sum())
        if remainder > 0:
            frac_order = np.argsort(-(raw - np.floor(raw)))
            for j in frac_order[:remainder]:
                alloc[j] += 1
        for u, n in zip(unique, alloc):
            loc = idx_t[strata == u]
            n = int(min(n, len(loc)))
            if n > 0:
                selected.append(rng.choice(loc, size=n, replace=False))
    sel = np.sort(np.concatenate(selected))
    sub = adata[sel].copy()
    # Ensure canonical ordering/category compactness and retain paired latent semantics.
    sub.obs['time_point_processed'] = sub.obs['time_point_processed'].astype(float).astype(int).astype('category')
    sub.obs['label'] = sub.obs['label'].astype('category')
    sub.uns['stage2_subset'] = {
        'source_path': str(SOURCE),
        'seed': int(SEED),
        'n_per_time_target': int(N_PER_TIME),
        'selection': 'deterministic stratified by time_point_processed x cell_type x donor',
        'paired_latent_semantics': 'X_latent concatenates X_rna_svd first 50 dimensions and X_protein_pca next 30 dimensions',
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    sub.write_h5ad(OUT)
    print(f'wrote {OUT} shape={sub.shape}')
    print(sub.obs['time_point_processed'].value_counts().sort_index().to_dict())


if __name__ == '__main__':
    main()
