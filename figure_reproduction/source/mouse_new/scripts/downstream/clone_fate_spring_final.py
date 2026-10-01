#!/usr/bin/env python3
"""Final-regression clone-fate SPRING diagnostics for Clone-CUVGF-BMF.

This script consumes the locked final CytoBridge custom-model artifact and
same-panel no-clone/builtin baseline evaluation trajectories.  It does not train
a new model.  For the clone-aware final candidate, it reconstructs the approved
BranchDynamicalModel from the saved final model_state.pt and performs the same
sealed t0 branch-weighted rollout used by the custom simulation_hook, preserving
per-day-2-cell source provenance.  Clone-derived day-6 fate probabilities are
used only by the fixed evaluator/readout and figure layers.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import yaml
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

ADATA_PATH = Path('/Users/zhenyizhang/.cellcompass/algorithm_benchmarks/datasets/weinreb_rawrebuild_full_k10mindiff1/data.h5ad')
OUT_DIR = Path('/Users/zhenyizhang/Documents/GitHub/CytoBridge-agent-runtime-v2-20260309/mouse_new')
TABLE_DIR = OUT_DIR / 'downstream' / 'clone_fate_spring_final'
FIG_DIR = OUT_DIR / 'figures' / 'clone_fate_spring_final'
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

ALGO_PATH = Path('/Users/zhenyizhang/.cellcompass/training_algorithms/clone_cuvgf/algorithm.py')
FINAL_RUN_DIR = OUT_DIR / 'training_runs/20260607T113528987593Z__final__clone_cuvgf__campaign_20260-trial_20260607-weinreb_rawr'
FINAL_MODEL_STATE = FINAL_RUN_DIR / 'artifacts/model_state.pt'
FINAL_METRICS = FINAL_RUN_DIR / 'artifacts/metrics.json'
FINAL_RESOLVED_CONFIG = FINAL_RUN_DIR / 'resolved_config.yaml'
FINAL_TRAJECTORY = FINAL_RUN_DIR / 'artifacts/evaluation_trajectory.npz'

RUNS = {
    'clone_cuvgf_bmf_final': {
        'label': 'Clone-CUVGF-BMF final locked release',
        'run_id': '20260607T113528987593Z__final__clone_cuvgf__campaign_20260-trial_20260607-weinreb_rawr',
        'role': 'final_clone_aware_candidate',
        'metrics': FINAL_METRICS,
        'trajectory': FINAL_TRAJECTORY,
        'model_state': FINAL_MODEL_STATE,
        'resolved_config': FINAL_RESOLVED_CONFIG,
    },
    'vgfm_no_clone_anchor': {
        'label': 'VGFM no-clone anchor',
        'run_id': '20260606T160523640377Z__pilot__vgfm__campaign_20260-stage1_f-baseline-vgfm-weinreb_rawr',
        'role': 'nearest_no_clone_anchor',
        'trajectory': OUT_DIR / 'training_runs/20260606T160523640377Z__pilot__vgfm__campaign_20260-stage1_f-baseline-vgfm-weinreb_rawr/artifacts/evaluation_trajectory.npz',
        'metrics': OUT_DIR / 'training_runs/20260606T160523640377Z__pilot__vgfm__campaign_20260-stage1_f-baseline-vgfm-weinreb_rawr/artifacts/metrics.json',
    },
    'dynamical_ot_claim_sota': {
        'label': 'Dynamical OT strongest CFCorr baseline',
        'run_id': '20260606T160327488320Z__pilot__dynamical_ot__campaign_20260-stage1_f-baseline-dynamical_ot-weinreb_rawr',
        'role': 'strongest_same_panel_claim_baseline',
        'trajectory': OUT_DIR / 'training_runs/20260606T160327488320Z__pilot__dynamical_ot__campaign_20260-stage1_f-baseline-dynamical_ot-weinreb_rawr/artifacts/evaluation_trajectory.npz',
        'metrics': OUT_DIR / 'training_runs/20260606T160327488320Z__pilot__dynamical_ot__campaign_20260-stage1_f-baseline-dynamical_ot-weinreb_rawr/artifacts/metrics.json',
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


def knn_fate_probabilities(query, reference, ref_codes, n_fates, k=30, eps=1e-6):
    query = np.asarray(query, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    labels = np.asarray(ref_codes, dtype=np.int64)
    k = max(1, min(int(k), reference.shape[0]))
    out = np.zeros((query.shape[0], n_fates), dtype=np.float32)
    tree = cKDTree(reference)
    dsel, idx = tree.query(query, k=k)
    if k == 1:
        dsel = dsel[:, None]
        idx = idx[:, None]
    ww = 1.0 / (np.asarray(dsel, dtype=np.float32) + eps)
    for row in range(idx.shape[0]):
        codes = labels[np.asarray(idx[row], dtype=np.int64)]
        binc = np.bincount(codes, weights=ww[row], minlength=n_fates).astype(np.float32)
        total = float(binc.sum())
        if total > 0:
            out[row] = binc / total
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
    return {
        't0_indices': t0_indices,
        'final_indices': final_indices,
        'fate_labels': fate_labels,
        'fate_to_code': fate_to_code,
        'final_codes': final_codes,
        'final_latent': np.asarray(adata.obsm[LATENT_KEY][final_indices], dtype=np.float32),
        'clone_truth': clone_truth,
        'clone_terminal_n': clone_terminal_n,
    }


def add_truth_and_rows(adata, truth, source_indices, pred, run_key, spec):
    source_obs = adata.obs.iloc[source_indices]
    source_clones = np.asarray([clone_key(v) for v in source_obs[CLONE_KEY].to_numpy()], dtype=object)
    eligible = np.asarray([c in truth['clone_truth'] for c in source_clones], dtype=bool)
    clone_t0_counts: Dict[str, int] = {}
    for c in source_clones[eligible]:
        clone_t0_counts[str(c)] = clone_t0_counts.get(str(c), 0) + 1
    rows = []
    for row_i, obs_i in enumerate(source_indices):
        c = str(source_clones[row_i])
        if c in truth['clone_truth']:
            n_term = float(truth['clone_terminal_n'][c])
            reliability = n_term / (n_term + ALPHA)
            weight = reliability / max(float(clone_t0_counts.get(c, 1)), 1.0)
        else:
            reliability = 0.0
            weight = 0.0
        base = {
            'run_key': run_key,
            'run_label': spec['label'],
            'run_id': spec['run_id'],
            'source_obs_name': str(adata.obs_names[obs_i]),
            'source_index': int(obs_i),
            'clone': c,
            'eligible': bool(c in truth['clone_truth']),
            'truth_reliability': float(reliability),
            'metric_weight': float(weight),
            'spring_x': float(adata.obs.iloc[obs_i]['SPRING-x']),
            'spring_y': float(adata.obs.iloc[obs_i]['SPRING-y']),
            'source_label': str(adata.obs.iloc[obs_i][LABEL_KEY]),
        }
        for f_idx, fate in enumerate(truth['fate_labels']):
            base[f'truth_{fate}'] = float(truth['clone_truth'][c][f_idx]) if c in truth['clone_truth'] else np.nan
            base[f'pred_{fate}'] = float(pred[row_i, f_idx])
        rows.append(base)
    df = pd.DataFrame(rows)
    per_fate_corr = {}
    w = df['metric_weight'].to_numpy(float)
    for fate in truth['fate_labels']:
        per_fate_corr[str(fate)] = weighted_pearson(df[f'pred_{fate}'].to_numpy(float), df[f'truth_{fate}'].to_numpy(float), w)
    valid = [v for v in per_fate_corr.values() if np.isfinite(v)]
    cfcorr = float(np.mean(valid)) if valid else np.nan
    pred_mat = df[[f'pred_{f}' for f in truth['fate_labels']]].to_numpy(float)
    truth_mat = df[[f'truth_{f}' for f in truth['fate_labels']]].to_numpy(float)
    finite_rows = np.isfinite(pred_mat).all(axis=1) & np.isfinite(truth_mat).all(axis=1) & np.isfinite(w) & (w > 0)
    if int(finite_rows.sum()) > 0:
        brier = float(np.average(np.sum((pred_mat[finite_rows] - truth_mat[finite_rows]) ** 2, axis=1), weights=w[finite_rows]))
    else:
        brier = float('nan')
    return df, {'run_key': run_key, 'run_label': spec['label'], 'CFCorr_recomputed': cfcorr, 'Brier_recomputed': brier, 'per_fate_corr': per_fate_corr}


def compute_baseline_table(adata, truth, run_key, spec):
    traj = np.load(spec['trajectory'], allow_pickle=True)
    points0 = np.asarray(traj['points_000000'], dtype=np.float32)
    terminal_pred = np.asarray(traj[f'points_{len(traj["time_points"])-1:06d}'], dtype=np.float32)
    t0_latent = np.asarray(adata.obsm[LATENT_KEY][truth['t0_indices']], dtype=np.float32)
    tree = cKDTree(t0_latent)
    dist, pos = tree.query(points0, k=1)
    if float(np.max(dist)) > 1e-4 or len(np.unique(pos)) != len(pos):
        raise RuntimeError(f'Cannot align baseline {run_key} trajectory t0 points to unique source rows; max_dist={float(np.max(dist))}')
    source_indices = truth['t0_indices'][pos]
    pred = knn_fate_probabilities(terminal_pred, truth['final_latent'], truth['final_codes'], len(truth['fate_labels']), k=TERMINAL_K)
    df, summary = add_truth_and_rows(adata, truth, source_indices, pred, run_key, spec)
    summary['max_t0_alignment_distance'] = float(np.max(dist))
    summary['prediction_source'] = 'locked baseline evaluation_trajectory terminal endpoints'
    return df, summary


def load_algorithm_module():
    spec = importlib.util.spec_from_file_location('clone_cuvgf_algorithm_final_downstream', str(ALGO_PATH))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def compute_final_branch_table(adata, truth, run_key, spec):
    alg = load_algorithm_module()
    with spec['resolved_config'].open() as f:
        resolved = yaml.safe_load(f)
    n_fates = len(truth['fate_labels'])
    latent_dim = int(adata.obsm[LATENT_KEY].shape[1])
    model = alg.BranchDynamicalModel(latent_dim, resolved['model'], n_fates=n_fates).to(torch.device('cpu'))
    state_obj = torch.load(spec['model_state'], map_location='cpu', weights_only=False)
    state = {
        k: (torch.as_tensor(v) if isinstance(v, np.ndarray) else v)
        for k, v in state_obj['all_model']['model_state_dict'].items()
    }
    model.load_state_dict(state, strict=True)
    model.eval()
    t0_indices = truth['t0_indices']
    x0 = torch.as_tensor(np.asarray(adata.obsm[LATENT_KEY][t0_indices], dtype=np.float32), dtype=torch.float32)
    times = [0.0, 1.0, 2.0]
    integration_dt = float(((resolved.get('evaluation') or {}).get('trajectory_integration_dt') or 0.0125))
    with torch.no_grad():
        n0 = int(x0.shape[0])
        t0_col = torch.full((n0, 1), times[0], dtype=x0.dtype)
        q = torch.softmax(model.gate_net(torch.cat([x0, t0_col], dim=1)), dim=1)
        branch_eye = torch.eye(n_fates, dtype=x0.dtype)
        x = x0.repeat_interleave(n_fates, dim=0).contiguous()
        branch_onehot = branch_eye.repeat(n0, 1).contiguous()
        parent = np.repeat(np.arange(n0, dtype=np.int64), n_fates)
        weights0 = (q.reshape(-1) / max(float(n0), 1.0)).clamp_min(1e-12)
        logw = torch.log(weights0).reshape(-1, 1)
        current_t = times[0]
        for target_t in times[1:]:
            while current_t < target_t - 1e-9:
                dt = min(integration_dt, target_t - current_t)
                t_col = torch.full((x.shape[0], 1), current_t, dtype=x.dtype)
                inp = torch.cat([x, t_col, branch_onehot], dim=1)
                x = x + model.branch_velocity_net(inp) * dt
                logw = logw + model.branch_growth_net(inp) * dt
                current_t += dt
        branch_final_points = x.cpu().numpy().astype(np.float32)
        branch_final_weights = torch.exp(logw).cpu().numpy().reshape(-1).astype(np.float64)
    endpoint_fate = knn_fate_probabilities(branch_final_points, truth['final_latent'], truth['final_codes'], n_fates, k=TERMINAL_K)
    pred = np.zeros((len(t0_indices), n_fates), dtype=np.float64)
    denom = np.zeros(len(t0_indices), dtype=np.float64)
    np.add.at(pred, parent, endpoint_fate * branch_final_weights[:, None])
    np.add.at(denom, parent, branch_final_weights)
    pred = pred / np.maximum(denom[:, None], 1e-12)
    df, summary = add_truth_and_rows(adata, truth, t0_indices, pred, run_key, spec)
    summary.update({
        'prediction_source': 'reconstructed final BranchDynamicalModel sealed branch-weighted rollout from saved model_state.pt',
        'n_source_cells': int(len(t0_indices)),
        'n_branch_particles': int(branch_final_points.shape[0]),
        'integration_dt': integration_dt,
        'gate_entropy_mean': float(np.mean(-np.sum(q.cpu().numpy() * np.log(np.maximum(q.cpu().numpy(), 1e-12)), axis=1))),
    })
    return df, summary


def scatter(ax, df, col, title, cmap='viridis', vmin=None, vmax=None):
    vals = df[col].to_numpy(float)
    sca = ax.scatter(df['spring_x'], df['spring_y'], c=vals, s=5, cmap=cmap, vmin=vmin, vmax=vmax, linewidths=0, alpha=0.82)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect('equal', adjustable='box')
    return sca


def make_figures(all_tables: Dict[str, pd.DataFrame], fate_labels):
    final = all_tables['clone_cuvgf_bmf_final']
    selected_fates = [f for f in ['Baso', 'Mast', 'Meg', 'Monocyte', 'Neutrophil', 'Undifferentiated'] if f in set(map(str, fate_labels))]
    n = len(selected_fates)
    fig, axes = plt.subplots(2, n, figsize=(3.1*n, 6.2), constrained_layout=True)
    for j, fate in enumerate(selected_fates):
        vmax = max(0.5, float(np.nanmax(final[f'truth_{fate}'])))
        im0 = scatter(axes[0, j], final, f'truth_{fate}', f'Clone truth: {fate}', vmin=0, vmax=vmax)
        im1 = scatter(axes[1, j], final, f'pred_{fate}', f'Final BMF pred: {fate}', vmin=0, vmax=vmax)
        fig.colorbar(im0, ax=axes[0, j], fraction=0.046, pad=0.02)
        fig.colorbar(im1, ax=axes[1, j], fraction=0.046, pad=0.02)
    fig.suptitle('Final Clone-CUVGF-BMF fate probabilities on day-2 SPRING cells', fontsize=13)
    fig.savefig(FIG_DIR / 'final_truth_vs_pred_selected_fates.png', dpi=180)
    plt.close(fig)

    for fate in [f for f in ['Baso', 'Mast', 'Neutrophil'] if f in set(map(str, fate_labels))]:
        fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
        vmax = max(0.5, float(np.nanmax(final[f'truth_{fate}'])))
        im = scatter(axes[0], final, f'truth_{fate}', f'Clone truth {fate}', vmin=0, vmax=vmax)
        fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.02)
        for ax, key in zip(axes[1:], ['clone_cuvgf_bmf_final', 'vgfm_no_clone_anchor', 'dynamical_ot_claim_sota']):
            im = scatter(ax, all_tables[key], f'pred_{fate}', f'{RUNS[key]["label"]}\npred {fate}', vmin=0, vmax=vmax)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        fig.suptitle(f'Final SPRING fate-probability comparison: {fate}', fontsize=13)
        fig.savefig(FIG_DIR / f'final_spring_comparison_{fate}.png', dpi=180)
        plt.close(fig)


def main():
    adata = sc.read_h5ad(ADATA_PATH)
    truth = build_truth(adata)
    all_tables = {}
    summaries = {}
    for key, spec in RUNS.items():
        if key == 'clone_cuvgf_bmf_final':
            df, summary = compute_final_branch_table(adata, truth, key, spec)
        else:
            df, summary = compute_baseline_table(adata, truth, key, spec)
        df.to_csv(TABLE_DIR / f'{key}_day2_fate_probabilities.csv', index=False)
        all_tables[key] = df
        summaries[key] = summary
    make_figures(all_tables, truth['fate_labels'])

    metric_rows = []
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
            'Brier_recomputed_from_table': summaries[key]['Brier_recomputed'],
            'CFCorr_prediction_entropy': cm.get('CFCorr_prediction_entropy'),
            'CFCorr_truth_entropy': cm.get('CFCorr_truth_entropy'),
            'w1_mean': float(np.mean(m.get('w1_scores', [np.nan]))),
            'tmv_mean': float(np.mean(m.get('tmv_scores', [np.nan]))),
            'trajectory_path': str(spec.get('trajectory', '')),
            'metrics_path': str(spec['metrics']),
        })
    metric_df = pd.DataFrame(metric_rows)
    metric_df.to_csv(TABLE_DIR / 'final_metric_summary.csv', index=False)

    # Summary bar figure after metrics are available.
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for ax, metric, ylabel in zip(axes, ['CFCorr', 'w1_mean', 'tmv_mean'], ['CFCorr (higher better)', 'W1 mean (lower better)', 'TMV mean (lower better)']):
        ax.bar(metric_df['run_label'], metric_df[metric])
        ax.set_ylabel(ylabel)
        ax.tick_params(axis='x', labelrotation=35, labelsize=8)
        ax.set_title(metric)
    fig.suptitle('Final locked candidate versus same-panel no-clone/builtin baselines')
    fig.savefig(FIG_DIR / 'final_metric_comparison.png', dpi=180)
    plt.close(fig)

    manifest = {
        'adata_path': str(ADATA_PATH),
        'output_table_dir': str(TABLE_DIR),
        'output_figure_dir': str(FIG_DIR),
        'algorithm_id': 'clone_cuvgf',
        'proposal_id': 'proposal_20260607T050854815037Z',
        'campaign_id': 'campaign_20260606T144856902012Z',
        'release_status': 'final_regression_locked_complete',
        'runs': {k: {kk: (str(vv) if isinstance(vv, Path) else vv) for kk, vv in spec.items()} for k, spec in RUNS.items()},
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
            'prediction_source_candidate': 'saved final model_state.pt reconstructed through approved model-native branch simulation hook semantics',
            'prediction_source_baselines': 'locked model-native evaluation_trajectory.npz terminal endpoints',
        },
        'summaries': summaries,
        'figures': [str(p) for p in sorted(FIG_DIR.glob('*.png'))],
        'tables': [str(p) for p in sorted(TABLE_DIR.glob('*.csv'))],
    }
    with (TABLE_DIR / 'manifest.json').open('w') as f:
        json.dump(manifest, f, indent=2, allow_nan=True)
    print(json.dumps({'metric_summary': metric_rows, 'manifest': str(TABLE_DIR / 'manifest.json'), 'figures': manifest['figures']}, indent=2, allow_nan=True))


if __name__ == '__main__':
    main()
