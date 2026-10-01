#!/usr/bin/env python3
"""UMAP/protein-signal interpretability figures for CITE-PCFM.

This script visualizes what lower rollout protein_energy means biologically:
RNA-state rollout positions are embedded with observed RNA states, while generated
protein-coordinate signals are overlaid on the same RNA UMAP. It compares the
locked CITE-PCFM trajectory against baseline trajectories available from the
benchmark source runs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import StandardScaler

try:
    import umap  # type: ignore
    HAVE_UMAP = True
except Exception:
    from sklearn.decomposition import PCA
    HAVE_UMAP = False

sns.set_theme(style="whitegrid", context="paper")

OUTPUT_DIR = Path('/Users/zhenyizhang/Documents/GitHub/CytoBridge-agent-runtime-v2-20260309/multi-omics')
DATA_PATH = Path('/Users/zhenyizhang/.cellcompass/algorithm_benchmarks/datasets/gse305370_citeseq_stage2_10k/data.h5ad')
FINAL_RUN = OUTPUT_DIR / 'training_runs/20260526T174204321821Z__final__cite_paired_coord_fm__campaign_20260-trial_20260526-gse305370_ci'
FIG_DIR = OUTPUT_DIR / 'figures/downstream_cite_pcfm_umap_protein'
TAB_DIR = OUTPUT_DIR / 'downstream/cite_pcfm_umap_protein/tables'
MANIFEST = OUTPUT_DIR / 'downstream/cite_pcfm_umap_protein/umap_protein_manifest.json'
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST.parent.mkdir(parents=True, exist_ok=True)

METHODS = {
    'CITE-PCFM': {
        'trajectory': FINAL_RUN / 'artifacts/evaluation_trajectory.npz',
        'metrics': FINAL_RUN / 'artifacts/metrics.json',
        'kind': 'custom_final',
    },
    'CRUFM': {
        'trajectory': OUTPUT_DIR / 'training_runs/20260525T220649168798Z__pilot__crufm__campaign_20260-stage2_c-baseline-crufm-gse305370_ci/artifacts/evaluation_trajectory.npz',
        'metrics': Path('/Users/zhenyizhang/.cellcompass/algorithm_benchmarks/datasets/gse305370_citeseq_stage2_10k/baseline_artifacts/builtin_crufm/metrics.json'),
        'kind': 'best_protein_baseline',
    },
    'balanced OT-CFM': {
        'trajectory': OUTPUT_DIR / 'training_runs/20260525T215826526439Z__pilot__balanced_ot_cfm__campaign_20260-stage2_c-baseline-balanced_ot_-gse305370_ci/artifacts/evaluation_trajectory.npz',
        'metrics': Path('/Users/zhenyizhang/.cellcompass/algorithm_benchmarks/datasets/gse305370_citeseq_stage2_10k/baseline_artifacts/builtin_balanced_ot_cfm/metrics.json'),
        'kind': 'rna_only_balanced_baseline',
    },
}

RNA_DIM = 50
PROT_DIM = 30
RNG = np.random.default_rng(20260527)


def extract_metric(metrics_path: Path, key: str):
    if not metrics_path.exists():
        return np.nan
    with open(metrics_path) as f:
        m = json.load(f)
    if key in m:
        return m[key]
    if isinstance(m.get('custom_metrics'), dict) and key in m['custom_metrics']:
        return m['custom_metrics'][key]
    if key == 'w1_mean' and isinstance(m.get('w1_scores'), dict):
        vals = list(m['w1_scores'].values())
        return float(np.mean(vals)) if vals else np.nan
    return np.nan


def load_traj(path: Path):
    z = np.load(path)
    time_points = z['time_points']
    observed_time_points = z.get('observed_time_points', np.array([0, 1, 2, 3, 4], dtype=float))
    observed_time_indices = z.get('observed_time_indices', np.array([0, 10, 20, 30, 40], dtype=int))
    points = {float(time_points[i]): z[f'points_{i:06d}'] for i in range(len(time_points)) if f'points_{i:06d}' in z}
    return time_points, observed_time_points, observed_time_indices, points


def sample_rows(n, max_n):
    if n <= max_n:
        return np.arange(n)
    return np.sort(RNG.choice(n, max_n, replace=False))


def masp_affinity(prot, centroid, scale):
    d = np.linalg.norm((prot - centroid) / scale, axis=1)
    # Higher is more MasP-like; transform distance to a visually stable 0-1 affinity.
    return np.exp(-0.5 * d)


def safe_color_limits(vals_by_method):
    allv = np.concatenate([np.asarray(v) for v in vals_by_method if len(v)])
    return float(np.nanpercentile(allv, 2)), float(np.nanpercentile(allv, 98))


def main():
    adata = ad.read_h5ad(DATA_PATH)
    rna = np.asarray(adata.obsm['X_rna_svd'], dtype=np.float32)
    prot = np.asarray(adata.obsm['X_protein_pca'], dtype=np.float32)
    latent = np.asarray(adata.obsm['X_latent'], dtype=np.float32)
    obs = adata.obs.copy()
    obs['time_point_processed'] = obs['time_point_processed'].astype(float)

    # Observed protein phenotype centroids. These are latent CITE protein coordinates,
    # not named antibody intensities, because marker names are not available in this
    # benchmark contract.
    classes = list(pd.Categorical(obs['cell_type']).categories)
    prot_centroids = {c: prot[np.asarray(obs['cell_type'] == c)].mean(axis=0) for c in classes}
    prot_scale = prot.std(axis=0) + 1e-6
    masp_centroid = prot_centroids['MasP']
    mop_centroid = prot_centroids['MoP']
    eryp_centroid = prot_centroids['EryP']
    obs_masp_affinity = masp_affinity(prot, masp_centroid, prot_scale)

    # Protein-space cell-type readout used only for interpretation.
    knn = KNeighborsClassifier(n_neighbors=25, weights='distance')
    knn.fit(prot, obs['cell_type'])

    # Load method trajectories.
    method_frames = []
    full_method_arrays = {}
    for method, spec in METHODS.items():
        if not Path(spec['trajectory']).exists():
            continue
        time_points, obs_times, obs_idx, points_by_time = load_traj(Path(spec['trajectory']))
        full_method_arrays[method] = {'points_by_time': points_by_time, 'obs_times': obs_times}
        for t in obs_times:
            pts = points_by_time[float(t)]
            idx = sample_rows(pts.shape[0], 700)
            rr = pts[idx, :RNA_DIM]
            pp = pts[idx, RNA_DIM:RNA_DIM+PROT_DIM]
            aff = masp_affinity(pp, masp_centroid, prot_scale)
            cell_pred = knn.predict(pp)
            method_frames.append(pd.DataFrame({
                'method': method,
                'time': float(t),
                'row': idx,
                'masp_affinity': aff,
                'predicted_cell_type': cell_pred,
                'protein_distance_to_MasP': np.linalg.norm((pp-masp_centroid)/prot_scale, axis=1),
                'protein_distance_to_MoP': np.linalg.norm((pp-mop_centroid)/prot_scale, axis=1),
                'protein_distance_to_EryP': np.linalg.norm((pp-eryp_centroid)/prot_scale, axis=1),
            }))
    method_meta = pd.concat(method_frames, ignore_index=True)

    # Fit observed RNA UMAP plus sampled generated RNA states, so generated rollout
    # positions and observed background share the same RNA-state coordinate system.
    obs_idx = sample_rows(rna.shape[0], 5000)
    fit_blocks = [rna[obs_idx]]
    transform_blocks = [('Observed', None, rna[obs_idx])]
    row_cursor = len(obs_idx)
    sampled_keys = []
    for method, arrs in full_method_arrays.items():
        for t in [0.0, 1.0, 2.0, 3.0, 4.0]:
            pts = arrs['points_by_time'][float(t)]
            idx = sample_rows(pts.shape[0], 450)
            block = pts[idx, :RNA_DIM]
            fit_blocks.append(block)
            sampled_keys.append((method, t, idx, row_cursor, row_cursor + len(idx)))
            row_cursor += len(idx)
    fit_mat = np.vstack(fit_blocks)
    scaler = StandardScaler().fit(fit_mat)
    fit_scaled = scaler.transform(fit_mat)
    if HAVE_UMAP:
        reducer = umap.UMAP(n_neighbors=40, min_dist=0.25, metric='euclidean', random_state=20260527)
        emb = reducer.fit_transform(fit_scaled)
        embedding_name = 'UMAP'
    else:
        reducer = PCA(n_components=2, random_state=20260527)
        emb = reducer.fit_transform(fit_scaled)
        embedding_name = 'PCA fallback (UMAP unavailable)'

    observed_emb = emb[:len(obs_idx)]
    observed_df = pd.DataFrame({
        'umap1': observed_emb[:, 0], 'umap2': observed_emb[:, 1],
        'cell_type': np.asarray(obs['cell_type'])[obs_idx],
        'time': np.asarray(obs['time_point_processed'])[obs_idx],
        'masp_affinity': obs_masp_affinity[obs_idx],
        'kind': 'observed'
    })
    gen_rows = []
    for method, t, idx, start, end in sampled_keys:
        pts = full_method_arrays[method]['points_by_time'][float(t)][idx]
        pp = pts[:, RNA_DIM:RNA_DIM+PROT_DIM]
        cell_pred = knn.predict(pp)
        e = emb[start:end]
        gen_rows.append(pd.DataFrame({
            'method': method, 'time': t, 'row': idx,
            'umap1': e[:, 0], 'umap2': e[:, 1],
            'masp_affinity': masp_affinity(pp, masp_centroid, prot_scale),
            'predicted_cell_type': cell_pred,
        }))
    gen_df = pd.concat(gen_rows, ignore_index=True)

    observed_df.to_csv(TAB_DIR / 'observed_umap_background.csv', index=False)
    gen_df.to_csv(TAB_DIR / 'generated_umap_points_with_masp_affinity.csv', index=False)

    # Summary of protein-signal interpretability by method/time.
    summary = gen_df.groupby(['method', 'time']).agg(
        mean_masp_affinity=('masp_affinity', 'mean'),
        median_masp_affinity=('masp_affinity', 'median'),
        masp_fraction=('predicted_cell_type', lambda x: np.mean(np.asarray(x) == 'MasP')),
        mop_fraction=('predicted_cell_type', lambda x: np.mean(np.asarray(x) == 'MoP')),
        hsc_fraction=('predicted_cell_type', lambda x: np.mean(np.asarray(x) == 'HSC')),
    ).reset_index()
    summary.to_csv(TAB_DIR / 'generated_protein_signal_summary_by_method_time.csv', index=False)

    metric_rows = []
    for method, spec in METHODS.items():
        metric_rows.append({
            'method': method,
            'method_type': spec['kind'],
            'protein_energy': extract_metric(Path(spec['metrics']), 'protein_energy'),
            'w1_mean': extract_metric(Path(spec['metrics']), 'w1_mean'),
            'paired_cov_error': extract_metric(Path(spec['metrics']), 'paired_cov_error'),
        })
    metric_df = pd.DataFrame(metric_rows)
    metric_df.to_csv(TAB_DIR / 'umap_methods_metric_context.csv', index=False)

    # Figure 15: observed RNA UMAP with cell types and observed MasP protein affinity.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    palette = dict(zip(classes, sns.color_palette('tab10', n_colors=len(classes))))
    for c in classes:
        d = observed_df[observed_df.cell_type == c]
        axes[0].scatter(d.umap1, d.umap2, s=5, alpha=0.45, color=palette[c], label=c, linewidths=0)
    axes[0].set_title(f'Observed RNA-state {embedding_name}: cell types')
    axes[0].set_xlabel('RNA UMAP 1'); axes[0].set_ylabel('RNA UMAP 2')
    axes[0].legend(markerscale=2.5, fontsize=7, ncol=2, frameon=True)
    sc = axes[1].scatter(observed_df.umap1, observed_df.umap2, c=observed_df.masp_affinity, cmap='magma', s=5, alpha=0.6, linewidths=0)
    axes[1].set_title('Observed protein signal on RNA-state UMAP\n(MasP-like protein-centroid affinity)')
    axes[1].set_xlabel('RNA UMAP 1'); axes[1].set_ylabel('RNA UMAP 2')
    cb = fig.colorbar(sc, ax=axes[1], fraction=0.046); cb.set_label('MasP protein affinity')
    fig.savefig(FIG_DIR / 'fig15_observed_rna_umap_protein_signal.png', dpi=300)
    fig.savefig(FIG_DIR / 'fig15_observed_rna_umap_protein_signal.pdf')
    plt.close(fig)

    # Figure 16: method-by-time RNA UMAP rollout, colored by generated MasP protein affinity.
    methods = [m for m in METHODS if m in set(gen_df.method)]
    times = [0.0, 1.0, 2.0, 3.0, 4.0]
    vmin, vmax = safe_color_limits([gen_df.masp_affinity.values, observed_df.masp_affinity.values])
    fig, axes = plt.subplots(len(methods), len(times), figsize=(3.0*len(times), 3.0*len(methods)), sharex=True, sharey=True, constrained_layout=True)
    if len(methods) == 1:
        axes = np.asarray([axes])
    for i, method in enumerate(methods):
        for j, t in enumerate(times):
            ax = axes[i, j]
            bg = observed_df[observed_df.time == t]
            ax.scatter(bg.umap1, bg.umap2, s=3, color='lightgray', alpha=0.22, linewidths=0)
            d = gen_df[(gen_df.method == method) & (gen_df.time == t)]
            sc = ax.scatter(d.umap1, d.umap2, c=d.masp_affinity, cmap='magma', vmin=vmin, vmax=vmax, s=8, alpha=0.85, linewidths=0)
            if i == 0: ax.set_title(f't={t:g}')
            if j == 0: ax.set_ylabel(method)
            ax.set_xticks([]); ax.set_yticks([])
    cb = fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.65, pad=0.01)
    cb.set_label('Generated MasP-like protein affinity')
    fig.suptitle('RNA-state rollout on shared UMAP, overlaid with generated protein signal', y=1.02)
    fig.savefig(FIG_DIR / 'fig16_method_time_rna_umap_generated_protein_overlay.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIG_DIR / 'fig16_method_time_rna_umap_generated_protein_overlay.pdf', bbox_inches='tight')
    plt.close(fig)

    # Figure 17: trajectories of RNA UMAP centroids with protein affinity encoded.
    cent = gen_df.groupby(['method', 'time']).agg(
        umap1=('umap1', 'mean'), umap2=('umap2', 'mean'), masp_affinity=('masp_affinity', 'mean')
    ).reset_index()
    fig, ax = plt.subplots(figsize=(7, 5.8), constrained_layout=True)
    ax.scatter(observed_df.umap1, observed_df.umap2, s=3, color='lightgray', alpha=0.18, linewidths=0, label='observed cells')
    markers = {'CITE-PCFM': 'o', 'CRUFM': 's', 'balanced OT-CFM': '^'}
    colors = {'CITE-PCFM': '#d62728', 'CRUFM': '#1f77b4', 'balanced OT-CFM': '#2ca02c'}
    for method in methods:
        d = cent[cent.method == method].sort_values('time')
        ax.plot(d.umap1, d.umap2, color=colors.get(method), lw=2.2, label=method)
        pts = ax.scatter(d.umap1, d.umap2, c=d.masp_affinity, cmap='magma', vmin=vmin, vmax=vmax, s=90, marker=markers.get(method, 'o'), edgecolor='black', linewidth=0.6)
        for _, r in d.iterrows():
            ax.text(r.umap1, r.umap2, f"{r.time:g}", fontsize=7, ha='center', va='center', color='white')
    ax.set_title('Centroid trajectory in RNA UMAP with generated protein affinity')
    ax.set_xlabel('RNA UMAP 1'); ax.set_ylabel('RNA UMAP 2')
    ax.legend(frameon=True, fontsize=8, loc='best')
    cb = fig.colorbar(pts, ax=ax); cb.set_label('Mean MasP protein affinity')
    fig.savefig(FIG_DIR / 'fig17_rna_umap_centroid_paths_with_protein_affinity.png', dpi=300)
    fig.savefig(FIG_DIR / 'fig17_rna_umap_centroid_paths_with_protein_affinity.pdf')
    plt.close(fig)

    # Figure 18: put the abstract metric next to the visual protein readout.
    plot_df = summary.merge(metric_df[['method', 'protein_energy']], on='method', how='left')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    sns.barplot(data=metric_df, x='method', y='protein_energy', hue='method_type', dodge=False, ax=axes[0])
    axes[0].set_title('Abstract metric: lower protein_energy')
    axes[0].set_xlabel(''); axes[0].tick_params(axis='x', rotation=25)
    axes[0].legend([], [], frameon=False)
    sns.lineplot(data=plot_df, x='time', y='mean_masp_affinity', hue='method', marker='o', ax=axes[1])
    axes[1].set_title('Visual readout: generated MasP-like protein signal over time')
    axes[1].set_ylabel('Mean generated MasP protein affinity')
    axes[1].set_xlabel('time')
    fig.savefig(FIG_DIR / 'fig18_metric_to_visual_protein_signal_bridge.png', dpi=300)
    fig.savefig(FIG_DIR / 'fig18_metric_to_visual_protein_signal_bridge.pdf')
    plt.close(fig)

    # Figure 19: generated phenotype fractions on RNA trajectory, emphasizing biological plausibility.
    frac_long = summary.melt(id_vars=['method', 'time'], value_vars=['masp_fraction', 'mop_fraction', 'hsc_fraction'], var_name='phenotype_fraction', value_name='fraction')
    frac_long['phenotype'] = frac_long['phenotype_fraction'].str.replace('_fraction', '', regex=False).str.replace('masp','MasP').str.replace('mop','MoP').str.replace('hsc','HSC')
    fig, axes = plt.subplots(1, len(methods), figsize=(4.2*len(methods), 3.8), sharey=True, constrained_layout=True)
    if len(methods) == 1: axes = [axes]
    for ax, method in zip(axes, methods):
        d = frac_long[frac_long.method == method]
        sns.lineplot(data=d, x='time', y='fraction', hue='phenotype', marker='o', ax=ax)
        ax.set_title(method)
        ax.set_ylim(0, 1)
        ax.set_ylabel('generated phenotype fraction')
        ax.legend(fontsize=7)
    fig.suptitle('Protein-space phenotype readouts from generated trajectories', y=1.03)
    fig.savefig(FIG_DIR / 'fig19_generated_protein_phenotype_fractions.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIG_DIR / 'fig19_generated_protein_phenotype_fractions.pdf', bbox_inches='tight')
    plt.close(fig)

    # Prepare manifest.
    figures = [str(p) for p in sorted(FIG_DIR.glob('fig*.png'))]
    tables = [str(p) for p in sorted(TAB_DIR.glob('*.csv'))]
    cite_metric = metric_df[metric_df.method == 'CITE-PCFM'].iloc[0].to_dict()
    crufm_metric = metric_df[metric_df.method == 'CRUFM'].iloc[0].to_dict()
    manifest = {
        'analysis': 'CITE-PCFM UMAP/protein-dynamics interpretability',
        'dataset': str(DATA_PATH),
        'embedding': embedding_name,
        'trajectory_contract': 'RNA UMAP coordinates are generated rollout RNA dimensions 0:50 embedded with observed X_rna_svd; protein overlays use generated protein dimensions 50:80 / X_protein_pca centroids.',
        'protein_signal_contract': 'No antibody names were available in this benchmark; visual protein signals are latent CITE protein-PC phenotype affinities, especially MasP protein-centroid affinity.',
        'methods': {m: {k: str(v) for k, v in spec.items()} for m, spec in METHODS.items()},
        'key_metric_bridge': {
            'cite_pcfm_protein_energy': float(cite_metric['protein_energy']),
            'crufm_protein_energy': float(crufm_metric['protein_energy']),
            'relative_reduction_vs_crufm_percent': float((crufm_metric['protein_energy'] - cite_metric['protein_energy']) / crufm_metric['protein_energy'] * 100),
            'visual_interpretation': 'Lower protein_energy appears as generated RNA-state trajectories whose protein-coordinate overlays better preserve phenotype-consistent protein affinity patterns on the RNA UMAP.'
        },
        'figures': figures,
        'tables': tables,
        'warnings': [
            'Protein signals are latent protein-PC phenotype affinities, not named antibody intensities, because direct marker names are absent from the prepared h5ad.',
            'Baseline trajectories are reused from saved evaluation_trajectory.npz source runs; no target endpoint repair is performed.',
            'Cell-type/protein phenotype labels are readouts from observed protein-space classifiers/centroids, not lineage-tracing truth.'
        ]
    }
    with open(MANIFEST, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2)[:5000])


if __name__ == '__main__':
    main()
