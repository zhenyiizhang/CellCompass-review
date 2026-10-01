#!/usr/bin/env python3
"""Biologically oriented downstream analyses for CITE-PCFM.

Uses the locked final-regression evaluation trajectory and observed CITE-seq
cell-type labels as a readout contract. Cell-type/fate results are therefore
model-predicted phenotype readouts from rollout states, not lineage-tracing truth.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

OUTPUT_DIR = Path('/Users/zhenyizhang/Documents/GitHub/CytoBridge-agent-runtime-v2-20260309/multi-omics')
FIG_DIR = OUTPUT_DIR / 'figures' / 'downstream_cite_pcfm_biology'
TABLE_DIR = OUTPUT_DIR / 'downstream' / 'cite_pcfm_biology' / 'tables'
MANIFEST_DIR = OUTPUT_DIR / 'downstream' / 'cite_pcfm_biology'
for p in (FIG_DIR, TABLE_DIR, MANIFEST_DIR):
    p.mkdir(parents=True, exist_ok=True)

DATA_PATH = Path('/Users/zhenyizhang/.cellcompass/algorithm_benchmarks/datasets/gse305370_citeseq_stage2_10k/data.h5ad')
FINAL_RUN_DIR = Path('/Users/zhenyizhang/Documents/GitHub/CytoBridge-agent-runtime-v2-20260309/multi-omics/training_runs/20260526T174204321821Z__final__cite_paired_coord_fm__campaign_20260-trial_20260526-gse305370_ci')
FINAL_TRAJ = FINAL_RUN_DIR / 'artifacts' / 'evaluation_trajectory.npz'
FINAL_METRICS = FINAL_RUN_DIR / 'artifacts' / 'metrics.json'

sns.set_theme(style='whitegrid', context='paper', font_scale=1.1)
CT_ORDER = ['HSC', 'EryP', 'MkP', 'NeuP', 'MasP', 'MoP', 'BP']
CT_COLORS = {
    'HSC': '#4C72B0',
    'EryP': '#C44E52',
    'MkP': '#8172B3',
    'NeuP': '#55A868',
    'MasP': '#CCB974',
    'MoP': '#64B5CD',
    'BP': '#8C8C8C',
}


def savefig(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()


def load_data():
    adata = sc.read_h5ad(DATA_PATH)
    X = np.asarray(adata.obsm['X_latent'], dtype=np.float32)
    obs = adata.obs.copy()
    obs['time_point_processed'] = obs['time_point_processed'].astype(float)
    obs['cell_type'] = obs['cell_type'].astype(str)
    z = np.load(FINAL_TRAJ, allow_pickle=True)
    return adata, X, obs, z


def train_cell_type_readout(X, obs):
    # Multinomial classifier gives probability readout; KNN sanity check confirms separability.
    y = obs['cell_type'].values
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight='balanced', multi_class='auto', C=1.0, random_state=7),
    )
    clf.fit(X, y)
    pred = clf.predict(X)
    bal_acc = float(balanced_accuracy_score(y, pred))
    cm = pd.DataFrame(confusion_matrix(y, pred, labels=clf.classes_), index=clf.classes_, columns=clf.classes_)
    cm.to_csv(TABLE_DIR / 'cell_type_readout_confusion_matrix.csv')
    # KNN only as a diagnostic, not used for fate probabilities.
    knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15, weights='distance'))
    knn.fit(X, y)
    knn_acc = float(balanced_accuracy_score(y, knn.predict(X)))
    return clf, {'balanced_accuracy_logistic': bal_acc, 'balanced_accuracy_knn_training_diagnostic': knn_acc, 'classes': list(clf.classes_)}


def trajectory_cell_type_probabilities(clf, z):
    times = z['time_points']
    obs_times = z['observed_time_points']
    obs_indices = set(z['observed_time_indices'].tolist())
    classes = list(clf.classes_)
    rows = []
    entropy_rows = []
    for j, t in enumerate(times):
        pts = z[f'points_{j:06d}']
        probs = clf.predict_proba(pts)
        mean_prob = probs.mean(axis=0)
        pred_labels = np.asarray(classes)[np.argmax(probs, axis=1)]
        for cls, val in zip(classes, mean_prob):
            rows.append({'time': float(t), 'cell_type': cls, 'probability': float(val), 'is_observed_time': j in obs_indices})
        # per-particle entropy tracks branch ambiguity over continuous rollout
        ent = -(probs * np.log(np.clip(probs, 1e-9, 1.0))).sum(axis=1)
        entropy_rows.append({'time': float(t), 'mean_entropy': float(ent.mean()), 'dominant_type': pd.Series(pred_labels).mode().iloc[0]})
    prob_df = pd.DataFrame(rows)
    entropy_df = pd.DataFrame(entropy_rows)
    prob_df.to_csv(TABLE_DIR / 'rollout_cell_type_probabilities.csv', index=False)
    entropy_df.to_csv(TABLE_DIR / 'rollout_cell_type_entropy.csv', index=False)
    return prob_df, entropy_df


def observed_cell_type_fractions(obs):
    ct = pd.crosstab(obs['time_point_processed'], obs['cell_type'], normalize='index')
    ct = ct.reindex(columns=[c for c in CT_ORDER if c in ct.columns] + [c for c in ct.columns if c not in CT_ORDER])
    out = ct.reset_index().melt(id_vars='time_point_processed', var_name='cell_type', value_name='observed_fraction')
    out.to_csv(TABLE_DIR / 'observed_cell_type_fractions.csv', index=False)
    return out


def plot_cell_type_dynamics(prob_df, obs_frac):
    keep = [c for c in CT_ORDER if c in prob_df.cell_type.unique()]
    df = prob_df[prob_df.cell_type.isin(keep)].copy()
    plt.figure(figsize=(9.0, 5.4))
    ax = sns.lineplot(data=df, x='time', y='probability', hue='cell_type', linewidth=2.2,
                      palette={c: CT_COLORS.get(c, '#777777') for c in keep})
    # observed fractions at integer times, shown as points
    obs2 = obs_frac[obs_frac.cell_type.isin(keep)].rename(columns={'time_point_processed': 'time', 'observed_fraction': 'probability'})
    sns.scatterplot(data=obs2, x='time', y='probability', hue='cell_type', palette={c: CT_COLORS.get(c, '#777777') for c in keep},
                    marker='X', s=55, ax=ax, legend=False, edgecolor='black', linewidth=0.4)
    ax.set_xlabel('Continuous rollout time')
    ax.set_ylabel('Cell-type probability / observed fraction')
    ax.set_title('Rollout-derived phenotype composition tracks hematopoietic shifts')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
    savefig(FIG_DIR / 'fig9_rollout_cell_type_composition.png')


def transition_matrix_from_rollout(clf, z, obs):
    # Source labels are real t0 labels for the 2000 source particles in evaluation trajectory.
    source_obs = obs[obs['time_point_processed'] == 0].copy()
    pts_final = z[f'points_{len(z["time_points"])-1:06d}']
    probs_final = clf.predict_proba(pts_final)
    classes = list(clf.classes_)
    n = min(len(source_obs), probs_final.shape[0])
    source_labels = source_obs['cell_type'].values[:n]
    probs_final = probs_final[:n]
    rows = []
    for src in sorted(pd.unique(source_labels)):
        mask = source_labels == src
        vals = probs_final[mask].mean(axis=0)
        for cls, val in zip(classes, vals):
            rows.append({'source_cell_type_t0': src, 'predicted_target_cell_type_t4': cls, 'probability': float(val), 'n_source': int(mask.sum())})
    trans = pd.DataFrame(rows)
    pivot = trans.pivot(index='source_cell_type_t0', columns='predicted_target_cell_type_t4', values='probability').fillna(0)
    pivot = pivot.reindex(index=[c for c in CT_ORDER if c in pivot.index], columns=[c for c in CT_ORDER if c in pivot.columns])
    pivot.to_csv(TABLE_DIR / 'rollout_t0_to_t4_transition_matrix.csv')

    plt.figure(figsize=(7.2, 5.8))
    ax = sns.heatmap(pivot, cmap='mako', annot=True, fmt='.2f', cbar_kws={'label': 'Mean predicted target probability'})
    ax.set_xlabel('Predicted t=4 phenotype')
    ax.set_ylabel('Observed source t=0 phenotype')
    ax.set_title('Rollout-predicted t0→t4 phenotype transition structure')
    savefig(FIG_DIR / 'fig10_t0_t4_transition_heatmap.png')
    return trans, pivot


def protein_alignment_by_cell_type(z, obs):
    # Distinctive method readout: protein subspace alignment by phenotype/time.
    rows = []
    obs_X = None
    # Use observed protein coordinates from X_latent split. Evaluation points are paired RNA(0:50)+protein(50:80).
    # For each observed time and cell type, compare predicted protein centroid to observed protein centroid.
    for j, t in zip(z['observed_time_indices'], z['observed_time_points']):
        pts = z[f'points_{int(j):06d}']
        pred_p = pts[:, 50:80]
        pred_r = pts[:, :50]
        mask_time = obs['time_point_processed'].values == float(t)
        for ct in sorted(obs.loc[mask_time, 'cell_type'].unique()):
            mask = mask_time & (obs['cell_type'].values == ct)
            # Use global predicted cell-type-free centroid as conservative comparison to observed phenotype centroid;
            # phenotype-specific predicted centroids require classifier assignment and are reported separately elsewhere.
            obs_p = np.asarray(adata_global.obsm['X_latent'])[mask, 50:80]
            obs_r = np.asarray(adata_global.obsm['X_latent'])[mask, :50]
            if obs_p.shape[0] < 5:
                continue
            rows.append({
                'time': float(t),
                'cell_type': ct,
                'n_observed': int(obs_p.shape[0]),
                'observed_protein_centroid_norm': float(np.linalg.norm(obs_p.mean(axis=0))),
                'observed_rna_centroid_norm': float(np.linalg.norm(obs_r.mean(axis=0))),
                'rollout_global_protein_centroid_distance': float(np.linalg.norm(pred_p.mean(axis=0) - obs_p.mean(axis=0))),
                'rollout_global_rna_centroid_distance': float(np.linalg.norm(pred_r.mean(axis=0) - obs_r.mean(axis=0))),
            })
    df = pd.DataFrame(rows)
    df.to_csv(TABLE_DIR / 'phenotype_centroid_alignment.csv', index=False)
    return df


def classifier_assigned_protein_centroids(clf, z, obs, adata):
    classes = list(clf.classes_)
    rows = []
    X = np.asarray(adata.obsm['X_latent'])
    for j, t in zip(z['observed_time_indices'], z['observed_time_points']):
        pts = z[f'points_{int(j):06d}']
        probs = clf.predict_proba(pts)
        pred_idx = np.argmax(probs, axis=1)
        mask_time = obs['time_point_processed'].values == float(t)
        for ci, ct in enumerate(classes):
            pred_mask = pred_idx == ci
            obs_mask = mask_time & (obs['cell_type'].values == ct)
            if pred_mask.sum() < 5 or obs_mask.sum() < 5:
                continue
            pred_p = pts[pred_mask, 50:80]
            pred_r = pts[pred_mask, :50]
            obs_p = X[obs_mask, 50:80]
            obs_r = X[obs_mask, :50]
            rows.append({
                'time': float(t),
                'cell_type': ct,
                'n_predicted': int(pred_mask.sum()),
                'n_observed': int(obs_mask.sum()),
                'protein_centroid_distance': float(np.linalg.norm(pred_p.mean(axis=0) - obs_p.mean(axis=0))),
                'rna_centroid_distance': float(np.linalg.norm(pred_r.mean(axis=0) - obs_r.mean(axis=0))),
                'protein_dispersion_pred': float(np.mean(np.linalg.norm(pred_p - pred_p.mean(axis=0), axis=1))),
                'protein_dispersion_obs': float(np.mean(np.linalg.norm(obs_p - obs_p.mean(axis=0), axis=1))),
            })
    df = pd.DataFrame(rows)
    df.to_csv(TABLE_DIR / 'classifier_assigned_modality_centroid_alignment.csv', index=False)
    return df


def plot_modality_alignment(df):
    if df.empty:
        return
    keep = [c for c in CT_ORDER if c in df.cell_type.unique()]
    plot_df = df[df.cell_type.isin(keep)].copy()
    plt.figure(figsize=(9.0, 5.2))
    ax = sns.lineplot(data=plot_df, x='time', y='protein_centroid_distance', hue='cell_type', marker='o', linewidth=2,
                      palette={c: CT_COLORS.get(c, '#777777') for c in keep})
    ax.set_xlabel('Observed target time')
    ax.set_ylabel('Protein centroid distance in CITE-PCFM readout space')
    ax.set_title('Phenotype-specific protein alignment from rollout-assigned cells')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
    savefig(FIG_DIR / 'fig11_phenotype_protein_alignment.png')

    tmp = plot_df[['time', 'cell_type', 'protein_centroid_distance', 'rna_centroid_distance']].melt(
        id_vars=['time', 'cell_type'], var_name='modality', value_name='centroid_distance')
    tmp['modality'] = tmp['modality'].map({'protein_centroid_distance': 'Protein', 'rna_centroid_distance': 'RNA'})
    plt.figure(figsize=(7.4, 5.0))
    ax = sns.boxplot(data=tmp, x='modality', y='centroid_distance', hue='modality', palette={'RNA': '#4C72B0', 'Protein': '#C44E52'}, legend=False)
    sns.stripplot(data=tmp, x='modality', y='centroid_distance', color='black', alpha=0.35, size=3, ax=ax)
    ax.set_xlabel('Modality subspace')
    ax.set_ylabel('Phenotype/time centroid distance')
    ax.set_title('Paired-coordinate readout: RNA and protein alignment diagnostics')
    savefig(FIG_DIR / 'fig12_rna_vs_protein_alignment_box.png')


def plot_entropy(entropy_df):
    plt.figure(figsize=(7.2, 4.6))
    ax = sns.lineplot(data=entropy_df, x='time', y='mean_entropy', linewidth=2.4, color='#C44E52')
    ax.set_xlabel('Continuous rollout time')
    ax.set_ylabel('Mean phenotype probability entropy')
    ax.set_title('Continuous branching ambiguity along the generated trajectory')
    savefig(FIG_DIR / 'fig13_rollout_branch_entropy.png')


def protein_axis_loadings(adata, obs):
    # Identify protein latent axes associated with biological phenotypes; not gene/protein marker names, but CITE protein PCs.
    Xp = np.asarray(adata.obsm['X_latent'])[:, 50:80]
    rows = []
    for k in range(Xp.shape[1]):
        vals = Xp[:, k]
        group_means = obs.assign(val=vals).groupby('cell_type')['val'].mean()
        top = group_means.sort_values(ascending=False)
        rows.append({
            'protein_pc': k + 1,
            'range_across_cell_types': float(group_means.max() - group_means.min()),
            'top_cell_type': str(top.index[0]),
            'bottom_cell_type': str(top.index[-1]),
            'top_mean': float(top.iloc[0]),
            'bottom_mean': float(top.iloc[-1]),
        })
    df = pd.DataFrame(rows).sort_values('range_across_cell_types', ascending=False)
    df.to_csv(TABLE_DIR / 'protein_pc_cell_type_association.csv', index=False)
    top_axes = df.head(6)['protein_pc'].astype(int).tolist()
    mean_mat = []
    for ct in [c for c in CT_ORDER if c in obs.cell_type.unique()]:
        mask = obs.cell_type.values == ct
        mean_mat.append(pd.Series(Xp[mask][:, [a-1 for a in top_axes]].mean(axis=0), index=[f'Protein PC{a}' for a in top_axes], name=ct))
    mat = pd.DataFrame(mean_mat)
    plt.figure(figsize=(7.2, 4.8))
    ax = sns.heatmap(mat, cmap='vlag', center=0, annot=True, fmt='.2f', cbar_kws={'label': 'Observed mean coordinate'})
    ax.set_xlabel('Protein latent axis')
    ax.set_ylabel('Cell type')
    ax.set_title('Biological phenotype structure in CITE protein latent coordinates')
    savefig(FIG_DIR / 'fig14_protein_pc_cell_type_structure.png')
    return df


def main():
    global adata_global
    adata, X, obs, z = load_data()
    adata_global = adata
    clf, readout_metrics = train_cell_type_readout(X, obs)
    prob_df, entropy_df = trajectory_cell_type_probabilities(clf, z)
    obs_frac = observed_cell_type_fractions(obs)
    plot_cell_type_dynamics(prob_df, obs_frac)
    trans, pivot = transition_matrix_from_rollout(clf, z, obs)
    assigned = classifier_assigned_protein_centroids(clf, z, obs, adata)
    plot_modality_alignment(assigned)
    plot_entropy(entropy_df)
    protein_pc = protein_axis_loadings(adata, obs)

    with open(FINAL_METRICS) as f:
        metrics = json.load(f)
    manifest = {
        'analysis': 'CITE-PCFM biologically oriented downstream analyses',
        'dataset': str(DATA_PATH),
        'final_run_dir': str(FINAL_RUN_DIR),
        'trajectory_artifact': str(FINAL_TRAJ),
        'cell_type_readout_contract': 'Observed cell_type labels train a multinomial classifier on paired X_latent. Rollout states are annotated by classifier probabilities; these are phenotype readouts, not lineage-tracing truth.',
        'readout_metrics': readout_metrics,
        'distinctive_method_characteristics': [
            'paired RNA/protein latent state with RNA dimensions 0:50 and protein dimensions 50:80',
            'model-generated continuous rollout from t=0, not static endpoint matching',
            'protein readout fidelity evaluated from sealed EvaluationTrajectory without endpoint repair',
            'balanced-only mass; TMV is diagnostic, not a biological growth claim'
        ],
        'key_final_metrics': {
            'w1_mean': float(np.mean(metrics.get('w1_scores', []))),
            'protein_energy': float(metrics.get('custom_metrics', {}).get('protein_energy')),
            'paired_cov_error': float(metrics.get('custom_metrics', {}).get('paired_cov_error')),
        },
        'figures': sorted(str(p) for p in FIG_DIR.glob('*.png')),
        'tables': sorted(str(p) for p in TABLE_DIR.glob('*.csv')),
        'warnings': [
            'Cell-type/fate probabilities are classifier readouts from generated states, not lineage-tracing ground truth.',
            'Protein PC axes are latent CITE protein coordinates; no direct antibody-marker names were available in this prepared benchmark contract.',
            'Balanced-only model: do not interpret TMV or cell-count shifts as mechanistic growth.'
        ]
    }
    (MANIFEST_DIR / 'biology_downstream_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))

if __name__ == '__main__':
    main()
