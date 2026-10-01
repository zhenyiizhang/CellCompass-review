"""Create a small synthetic time course for trying the CellCompass interface."""
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def main():
    output = Path(__file__).resolve().parents[1] / 'output/demo/toy_time_course.h5ad'
    if output.exists():
        raise SystemExit(f'Example already exists: {output}')
    rng = np.random.default_rng(42)
    times = np.repeat([0.0, 1.0, 2.0], 200)
    # Independent samples at each time, not paired cell trajectories.
    branch = rng.choice([-1.0, 1.0], size=len(times))
    position = np.column_stack([
        times + rng.normal(0, 0.15, len(times)),
        branch * times * 0.8 + rng.normal(0, 0.2, len(times)),
    ])
    loadings = rng.normal(0, 0.25, (2, 32))
    rates = np.exp(1.5 + position @ loadings)
    counts = rng.poisson(rates).astype(np.float32)
    log_expression = np.log1p(counts / counts.sum(axis=1, keepdims=True) * 1e4)
    data = ad.AnnData(
        X=log_expression.astype(np.float32),
        obs=pd.DataFrame({'time_point_processed': times},
                         index=[f'cell_{i:04d}' for i in range(len(times))]),
        var=pd.DataFrame(index=[f'synthetic_gene_{i:02d}' for i in range(32)]),
    )
    data.layers['counts'] = counts
    data.obsm['X_latent'] = PCA(n_components=8, random_state=42).fit_transform(log_expression).astype(np.float32)
    data.uns['description'] = 'Synthetic branching time course for software demonstration. No biological gene identities.'
    data.uns['seed'] = 42
    output.parent.mkdir(parents=True, exist_ok=True)
    data.write_h5ad(output, compression='gzip')
    print(f'Created {data.n_obs} cells, {data.n_vars} artificial genes, 3 time points.')
    print(f'Use this path in CellCompass: {output}')


if __name__ == '__main__':
    main()
