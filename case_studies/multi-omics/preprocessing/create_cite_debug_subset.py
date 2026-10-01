
from __future__ import annotations
import pathlib
import numpy as np
import pandas as pd
import anndata as ad

SOURCE = pathlib.Path('/Users/zhenyizhang/Documents/GitHub/CytoBridge-agent-runtime-v2-20260309/multi-omics/paper_intake/paper_input.h5ad')
OUT = pathlib.Path('/Users/zhenyizhang/Documents/GitHub/CytoBridge-agent-runtime-v2-20260309/multi-omics/paper/debug_subsets/GSE305370_citeseq_debug_3000.h5ad')
OUT.parent.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 17
TARGET_PER_TIME = 600
TIME_KEY = 'time_point_processed'
STRATA = ['cell_type', 'donor']

rng = np.random.default_rng(RANDOM_STATE)
a = ad.read_h5ad(SOURCE, backed='r')
obs = a.obs[[TIME_KEY] + STRATA].copy()
obs['_row'] = np.arange(a.n_obs)
selected = []
for t, df_t in obs.groupby(TIME_KEY, observed=True):
    n_target = min(TARGET_PER_TIME, len(df_t))
    # Allocate approximately proportional samples per cell_type/donor stratum with at least one
    # when possible, then trim/fill to the exact per-time target.
    parts = []
    grouped = list(df_t.groupby(STRATA, observed=True))
    for _, g in grouped:
        quota = max(1, int(round(n_target * len(g) / len(df_t))))
        take = min(quota, len(g))
        parts.append(rng.choice(g['_row'].to_numpy(), size=take, replace=False))
    rows = np.concatenate(parts) if parts else np.array([], dtype=int)
    rows = np.unique(rows)
    if rows.size > n_target:
        rows = rng.choice(rows, size=n_target, replace=False)
    elif rows.size < n_target:
        remaining = np.setdiff1d(df_t['_row'].to_numpy(), rows, assume_unique=False)
        if remaining.size:
            fill = rng.choice(remaining, size=min(n_target - rows.size, remaining.size), replace=False)
            rows = np.concatenate([rows, fill])
    selected.append(rows.astype(int))
idx = np.sort(np.concatenate(selected).astype(int))
subset = a[idx].to_memory()
subset.uns['debug_subset'] = {
    'source_path': str(SOURCE),
    'random_state': RANDOM_STATE,
    'target_per_time': TARGET_PER_TIME,
    'selection': 'time-stratified with proportional cell_type/donor coverage',
    'purpose': 'CITE-VGFM implementation/runtime preview only; not final biological evidence',
}
subset.write_h5ad(OUT)
print(f'wrote {OUT} shape={subset.shape}')
print(subset.obs[TIME_KEY].value_counts().sort_index().to_dict())
