import os
import anndata as ad
import numpy as np

src = '/Users/zhenyizhang/Documents/GitHub/CytoBridge-agent-runtime-v2-20260309/multi-omics/paper_intake/paper_input.h5ad'
out = '/Users/zhenyizhang/Documents/GitHub/CytoBridge-agent-runtime-v2-20260309/multi-omics/paper/debug_subsets/GSE305370_citeseq_stage1_5000_proportional.h5ad'
os.makedirs(os.path.dirname(out), exist_ok=True)
adata = ad.read_h5ad(src)
rng = np.random.default_rng(7)
counts = adata.obs['time_point_processed'].value_counts().sort_index()
props = counts / counts.sum()
target = (props * 5000).round().astype(int)
while int(target.sum()) != 5000:
    diff = 5000 - int(target.sum())
    frac = props * 5000 - (props * 5000).round()
    if diff > 0:
        k = frac.idxmax(); target.loc[k] += 1
    else:
        k = frac.idxmin(); target.loc[k] -= 1
idxs = []
for t, n in target.items():
    pool = np.where(adata.obs['time_point_processed'].astype(str).values == str(t))[0]
    idxs.append(rng.choice(pool, size=int(n), replace=False))
idx = np.concatenate(idxs)
rng.shuffle(idx)
subset = adata[idx].copy()
subset.uns.setdefault('benchmark_preparation', {})
subset.uns['benchmark_preparation']['source_path'] = src
subset.uns['benchmark_preparation']['sampling'] = 'proportional_by_time_point_processed_total_5000_seed_7'
subset.write_h5ad(out)
print(out)
print(subset.shape)
print(subset.obs['time_point_processed'].value_counts().sort_index())
