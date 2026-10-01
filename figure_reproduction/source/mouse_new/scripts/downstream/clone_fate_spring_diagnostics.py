#!/usr/bin/env python3
"""Generate clone-fate SPRING diagnostics from locked CytoBridge evaluation trajectories.

This script does not train or integrate a new model. It consumes saved
model-native evaluation_trajectory.npz artifacts and applies the exact CFCorr
truth/readout logic used by clone_cuvgf's campaign evaluator.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

ADATA_PATH = Path('/Users/zhenyizhang/.cellcompass/algorithm_benchmarks/datasets/weinreb_rawrebuild_full_k10mindiff1/data.h5ad')
OUT_DIR = Path('/Users/zhenyizhang/Documents/GitHub/CytoBridge-agent-runtime-v2-20260309/mouse_new')
FIG_DIR = OUT_DIR / 'figures' / 'clone_fate_spring'
TABLE_DIR = OUT_DIR / 'downstream' / 'clone_fate_spring'
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

RUNS = {
    'clone_cuvgf_active_best': {
        'label': 'Clone-CUVGF-FP Stage 2 active best',
        'run_id': '20260606T183113438084Z__pilot__clone_cuvgf__campaign_20260-trial_20260606-weinreb_rawr',
        'trajectory': OUT_DIR / 'training_runs/20260606T183113438084Z__pilot__clone_cuvgf__campaign_20260-trial_20260606-weinreb_rawr/artifacts/evaluation_trajectory.npz',
        'metrics': OUT_DIR / 'training_runs/20260606T183113438084Z__pilot__clone_cuvgf__campaign_20260-trial_20260606-weinreb_rawr/artifacts/metrics.json',
        'role': 'failed_candidate_active_best',
    },
    'vgfm_no_clone_anchor': {
        'label': 'VGFM no-clone anchor',
        'run_id': '20260606T160523640377Z__pilot__vgfm__campaign_20260-stage1_f-baseline-vgfm-weinreb_rawr',
        'trajectory': OUT_DIR / 'training_runs/20260606T160523640377Z__pilot__vgfm__campaign_20260-stage1_f-baseline-vgfm-weinreb_rawr/artifacts/evaluation_trajectory.npz',
        'metrics': OUT_DIR / 'training_runs/20260606T160523640377Z__pilot__vgfm__campaign_20260-stage1_f-baseline-vgfm-weinreb_rawr/artifacts/metrics.json',
        'role': 'no_clone_anchor',
    },
    'dynamical_ot_claim_sota': {
        'label': 'Dynamical OT CFCorr baseline',
        'run_id': '20260606T160327488320Z__pilot__dynamical_ot__campaign_20260-stage1_f-baseline-dynamical_ot-weinreb_rawr',
        'trajectory': OUT_DIR / 'training_runs/20260606T160327488320Z__pilot__dynamical_ot__campaign_20260-stage1_f-baseline-dynamical_ot-weinreb_rawr/artifacts/evaluation_trajectory.npz',
        'metrics': OUT_DIR / 'training_runs/20260606T160327488320Z__pilot__dynamical_ot__campaign_20260-stage1_f-baseline-dynamical_ot-weinreb_rawr/artifacts/metrics.json',
        'role': 'stage2_claim_baseline',
    },
}

CLONE_KEY = 'clone'
LABEL_KEY = 'Cell type annotation'
TIME_KEY = 'time_point_processed'
LATENT_KEY = 'X_latent'
ALPHA = 5.0
TERMINAL_K = 30
KNN_CHUNK_SIZE = 512

MISSING_CLONE_STRINGS = {'', 'nan', 'none', 'null', 'na', '<na>'}


def clone_key(value: Any) -> str:
    text = str(value).strip()
    if text.lower() in MISSING_CLONE_STRINGS:
        return ''
    try:
        f = float(text)
        if np.isfinite(f) and abs(f - int(f)) < 1e-9:
            return str(int(f))
    except Exception:
        pass
    return text


def weighted_pearson(x, y, w, eps=1e-12):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    if int(mask.sum()) < 3:
        return np.nan
    x = x[mask]; y = y[mask]; w = w[mask]
    sw = float(w.sum())
    if sw <= eps:
        return np.nan
    w = w / sw
    mx = float(np.sum(w * x)); my = float(np.sum(w * y))
    vx = float(np.sum(w * (x - mx) ** 2)); vy = float(np.sum(w * (y - my) ** 2))
    if vx <= eps or vy <= eps:
        return np.nan
    return float(np.sum(w * (x - mx) * (y - my)) / np.sqrt(vx * vy))


def knn_fate_probabilities(query, reference, ref_codes, n_fates, k=30, chunk_size=512, eps=1e-6):
    query = np.asarray(query, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    labels = np.asarray(ref_codes, dtype=np.int64)
    k = max(1, min(int(k), reference.shape[0]))
    out = np.zeros((query.shape[0], n_fates), dtype=np.float32)
    ref_norm = np.sum(reference * reference, axis=1)[None, :]
    for start in range(0, query.shape[0], int(chunk_size)):
        stop = min(start + int(chunk_size), query.shape[0])
        q = query[start:stop]
        q_norm = np.sum(q * q, axis=1)[:, None]
        dist = np.maximum(q_norm + ref_norm - 2.0 * q @ reference.T, 0.0)
        idx = np.argpartition(dist, kth=k - 1, axis=1)[:, :k]
        dsel = np.take_along_axis(dist, idx, axis=1)
        weights = 1.0 / (np.sqrt(dsel) + eps)
        for row in range(idx.shape[0]):
            codes = labels[idx[row]]
            binc = np.bincount(codes, weights=weights[row], minlength=n_fates).astype(np.float32)
            total = float(binc.sum())
            if total > 0:
                out[start + row] = binc / total
    return out


def load_metrics(path: Path) -> Dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def build_truth(adata):
    obs = adata.obs
    t0_mask = np.asarray(obs[TIME_KEY].astype(float) == 0.0)
    final_mask = np.asarray(obs[TIME_KEY].astype(float) == 2.0)
    t0_indices = np.where(t0_mask)[0]
    final_indices = np.where(final_mask)[0]
    final_labels_raw = np.asarray(obs.iloc[final_indices][LABEL_KEY].astype(str))
    fate_labels = np.asarray(sorted(np.unique(final_labels_raw).tolist()), dtype=object)
    fate_to_code = {str(f): i for i, f in enumerate(fate_labels)}
    final_codes = np.asarray([fate_to_code[str(f)] for f in final_labels_raw], dtype=np.int64)
    n_fates = int(len(fate_labels))
    global_counts = np.bincount(final_codes, minlength=n_fates).astype(np.float64)
    global_prior = global_counts / max(float(global_counts.sum()), 1.0)
    final_clones = np.asarray([clone_key(v) for v in obs.iloc[final_indices][CLONE_KEY].to_numpy()], dtype=object)
    clone_truth: Dict[str, np.ndarray] = {}
    clone_terminal_n: Dict[str, int] = {}
    for c in sorted({str(x) for x in final_clones if x}):
        mask = final_clones == c
        n_c = int(mask.sum())
        counts = np.bincount(final_codes[mask], minlength=n_fates).astype(np.float64)
        truth = (counts + ALPHA * global_prior) / (float(n_c) + ALPHA)
        clone_truth[c] = truth
        clone_terminal_n[c] = n_c
    t0_clones_all = np.asarray([clone_key(v) for v in obs.iloc[t0_indices][CLONE_KEY].to_numpy()], dtype=object)
    return {
        't0_indices': t0_indices,
        'final_indices': final_indices,
        'fate_labels': fate_labels,
        'fate_to_code': fate_to_code,
        'final_codes': final_codes,
        'final_latent': np.asarray(adata.obsm[LATENT_KEY][final_indices], dtype=np.float32),
        'clone_truth': clone_truth,
        'clone_terminal_n': clone_terminal_n,
        't0_clones_all': t0_clones_all,
    }


def align_trajectory_to_t0(adata, t0_indices, points0):
    t0_latent = np.asarray(adata.obsm[LATENT_KEY][t0_indices], dtype=np.float32)
    tree = cKDTree(t0_latent)
    dist, pos = tree.query(np.asarray(points0, dtype=np.float32), k=1)
    if float(np.max(dist)) > 1e-4:
        raise RuntimeError(f'Cannot align trajectory t0 points exactly; max nearest distance={float(np.max(dist))}')
    if len(np.unique(pos)) != len(pos):
        raise RuntimeError('Trajectory t0 alignment produced duplicate source rows.')
    return t0_indices[pos], dist


def compute_run_table(adata, truth, run_key, spec):
    traj = np.load(spec['trajectory'], allow_pickle=True)
    points0 = np.asarray(traj['points_000000'], dtype=np.float32)
    terminal_pred = np.asarray(traj[f'points_{len(traj["time_points"])-1:06d}'], dtype=np.float32)
    source_indices, align_dist = align_trajectory_to_t0(adata, truth['t0_indices'], points0)
    pred = knn_fate_probabilities(
        terminal_pred, truth['final_latent'], truth['final_codes'], len(truth['fate_labels']),
        k=TERMINAL_K, chunk_size=KNN_CHUNK_SIZE,
    )
    source_obs = adata.obs.iloc[source_indices]
    source_clones = np.asarray([clone_key(v) for v in source_obs[CLONE_KEY].to_numpy()], dtype=object)
    eligible = np.asarray([c in truth['clone_truth'] for c in source_clones], dtype=bool)
    clone_t0_counts: Dict[str, int] = {}
    for c in source_clones[eligible]:
        clone_t0_counts[str(c)] = clone_t0_counts.get(str(c), 0) + 1
    rows = []
    weights = []
    for row_i, obs_i in enumerate(source_indices):
        c = str(source_clones[row_i])
        base = {
            'run_key': run_key,
            'run_label': spec['label'],
            'run_id': spec['run_id'],
            'source_obs_name': str(adata.obs_names[obs_i]),
            'source_index': int(obs_i),
            'clone': c,
            'eligible': bool(c in truth['clone_truth']),
            'spring_x': float(adata.obs.iloc[obs_i]['SPRING-x']),
            'spring_y': float(adata.obs.iloc[obs_i]['SPRING-y']),
            'source_label': str(adata.obs.iloc[obs_i][LABEL_KEY]),
        }
        if c in truth['clone_truth']:
            n_term = float(truth['clone_terminal_n'][c])
            reliability = n_term / (n_term + ALPHA)
            weight = reliability / max(float(clone_t0_counts.get(c, 1)), 1.0)
        else:
            reliability = 0.0
            weight = 0.0
        base['truth_reliability'] = reliability
        base['metric_weight'] = weight
        weights.append(weight)
        for f_idx, fate in enumerate(truth['fate_labels']):
            base[f'truth_{fate}'] = float(truth['clone_truth'][c][f_idx]) if c in truth['clone_truth'] else np.nan
            base[f'pred_{fate}'] = float(pred[row_i, f_idx])
        rows.append(base)
    df = pd.DataFrame(rows)
    # summary metrics recomputed from table for verification
    per_fate_corr = {}
    w = df['metric_weight'].to_numpy(float)
    for fate in truth['fate_labels']:
        per_fate_corr[str(fate)] = weighted_pearson(df[f'pred_{fate}'].to_numpy(float), df[f'truth_{fate}'].to_numpy(float), w)
    valid = [v for v in per_fate_corr.values() if np.isfinite(v)]
    cfcorr = float(np.mean(valid)) if valid else np.nan
    return df, {'run_key': run_key, 'run_label': spec['label'], 'CFCorr_recomputed': cfcorr, 'per_fate_corr': per_fate_corr, 'max_t0_alignment_distance': float(np.max(align_dist))}


def scatter(ax, df, col, title, cmap='viridis', vmin=None, vmax=None):
    vals = df[col].to_numpy(float)
    sca = ax.scatter(df['spring_x'], df['spring_y'], c=vals, s=5, cmap=cmap, vmin=vmin, vmax=vmax, linewidths=0, alpha=0.8)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect('equal', adjustable='box')
    return sca


def make_figures(all_tables: Dict[str, pd.DataFrame], summaries: Dict[str, Any], fate_labels):
    # Choose biologically frequent / informative fates for compact panels.
    selected_fates = ['Baso', 'Mast', 'Meg', 'Monocyte', 'Neutrophil', 'Undifferentiated']
    selected_fates = [f for f in selected_fates if f in set(map(str, fate_labels))]
    cand = all_tables['clone_cuvgf_active_best']
    # Truth maps for candidate/source cells (same source set as all runs).
    n = len(selected_fates)
    fig, axes = plt.subplots(2, n, figsize=(3.1*n, 6.2), constrained_layout=True)
    for j, fate in enumerate(selected_fates):
        im0 = scatter(axes[0, j], cand, f'truth_{fate}', f'Clone truth: {fate}', vmin=0, vmax=max(0.5, float(np.nanmax(cand[f"truth_{fate}"]))))
        im1 = scatter(axes[1, j], cand, f'pred_{fate}', f'Candidate pred: {fate}', vmin=0, vmax=max(0.5, float(np.nanmax(cand[f"truth_{fate}"]))))
        fig.colorbar(im0, ax=axes[0, j], fraction=0.046, pad=0.02)
        fig.colorbar(im1, ax=axes[1, j], fraction=0.046, pad=0.02)
    fig.suptitle('Clone-CUVGF active-best fate probabilities on day-2 SPRING cells', fontsize=13)
    p = FIG_DIR / 'candidate_truth_vs_pred_selected_fates.png'
    fig.savefig(p, dpi=180)
    plt.close(fig)

    # Compare candidate and baselines for three major fates.
    compare_fates = [f for f in ['Mast', 'Baso', 'Neutrophil'] if f in set(map(str, fate_labels))]
    for fate in compare_fates:
        fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
        vmax = max(0.5, float(np.nanmax(cand[f'truth_{fate}'])))
        im = scatter(axes[0], cand, f'truth_{fate}', f'Clone truth {fate}', vmin=0, vmax=vmax)
        fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.02)
        for ax, key in zip(axes[1:], ['clone_cuvgf_active_best', 'vgfm_no_clone_anchor', 'dynamical_ot_claim_sota']):
            im = scatter(ax, all_tables[key], f'pred_{fate}', f'{RUNS[key]["label"]}\npred {fate}', vmin=0, vmax=vmax)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        fig.suptitle(f'SPRING fate-probability comparison: {fate}', fontsize=13)
        p = FIG_DIR / f'spring_comparison_{fate}.png'
        fig.savefig(p, dpi=180)
        plt.close(fig)

    # Bar summary of CFCorr and Brier from metrics files.
    rows=[]
    for key, spec in RUNS.items():
        m = load_metrics(spec['metrics'])
        cm = m.get('custom_metrics', {})
        rows.append({'method': spec['label'], 'CFCorr': cm.get('CFCorr', np.nan), 'Brier': cm.get('CFCorr_brier', np.nan), 'W1': np.mean(m.get('w1_scores', [np.nan])), 'TMV': np.mean(m.get('tmv_scores', [np.nan]))})
    metric_df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for ax, metric, ylabel in zip(axes, ['CFCorr', 'W1', 'TMV'], ['CFCorr (higher better)', 'W1 (lower better)', 'TMV (lower better)']):
        ax.bar(metric_df['method'], metric_df[metric])
        ax.set_ylabel(ylabel)
        ax.tick_params(axis='x', labelrotation=35, labelsize=8)
        ax.set_title(metric)
    fig.suptitle('Stage 2 diagnostic metric comparison')
    p = FIG_DIR / 'stage2_metric_comparison.png'
    fig.savefig(p, dpi=180)
    plt.close(fig)


def main():
    adata = sc.read_h5ad(ADATA_PATH)
    truth = build_truth(adata)
    all_tables = {}
    summaries = {}
    for key, spec in RUNS.items():
        df, summary = compute_run_table(adata, truth, key, spec)
        df.to_csv(TABLE_DIR / f'{key}_day2_fate_probabilities.csv', index=False)
        all_tables[key] = df
        summaries[key] = summary
    make_figures(all_tables, summaries, truth['fate_labels'])
    # compact summary table from campaign metrics
    metric_rows=[]
    for key, spec in RUNS.items():
        m = load_metrics(spec['metrics'])
        cm = m.get('custom_metrics', {})
        metric_rows.append({
            'run_key': key,
            'run_label': spec['label'],
            'run_id': spec['run_id'],
            'role': spec['role'],
            'CFCorr': cm.get('CFCorr'),
            'CFCorr_recomputed_from_table': summaries[key]['CFCorr_recomputed'],
            'CFCorr_brier': cm.get('CFCorr_brier'),
            'CFCorr_prediction_entropy': cm.get('CFCorr_prediction_entropy'),
            'CFCorr_truth_entropy': cm.get('CFCorr_truth_entropy'),
            'w1_mean': float(np.mean(m.get('w1_scores', [np.nan]))),
            'tmv_mean': float(np.mean(m.get('tmv_scores', [np.nan]))),
            'trajectory_path': str(spec['trajectory']),
            'metrics_path': str(spec['metrics']),
        })
    metric_df = pd.DataFrame(metric_rows)
    metric_df.to_csv(TABLE_DIR / 'stage2_metric_summary.csv', index=False)
    runs_json = {
        key: {k: (str(v) if isinstance(v, Path) else v) for k, v in spec.items()}
        for key, spec in RUNS.items()
    }
    manifest = {
        'adata_path': str(ADATA_PATH),
        'output_table_dir': str(TABLE_DIR),
        'output_figure_dir': str(FIG_DIR),
        'runs': runs_json,
        'fate_labels': [str(x) for x in truth['fate_labels'].tolist()],
        'truth_contract': {
            'clone_key': CLONE_KEY,
            'label_key': LABEL_KEY,
            'time_key': TIME_KEY,
            'latent_key': LATENT_KEY,
            'source_processed_time': 0.0,
            'terminal_processed_time': 2.0,
            'dirichlet_alpha': ALPHA,
            'terminal_knn': TERMINAL_K,
            'knn_chunk_size': KNN_CHUNK_SIZE,
            'prediction_source': 'locked model-native evaluation_trajectory.npz terminal endpoints',
        },
        'summaries': summaries,
        'figures': [str(p) for p in sorted(FIG_DIR.glob('*.png'))],
    }
    with (TABLE_DIR / 'manifest.json').open('w') as f:
        json.dump(manifest, f, indent=2, allow_nan=True)
    print(json.dumps({'metric_summary': metric_rows, 'manifest': str(TABLE_DIR / 'manifest.json'), 'figures': manifest['figures']}, indent=2, allow_nan=True))

if __name__ == '__main__':
    main()
