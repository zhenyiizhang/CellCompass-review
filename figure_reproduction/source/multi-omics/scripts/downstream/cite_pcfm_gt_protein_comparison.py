#!/usr/bin/env python3
"""Ground-truth-first protein dynamics comparison for CITE-PCFM.

Every main figure compares generated CITE protein outputs to observed target cells
at matched time points. Methods: CITE-PCFM, CRUFM, balanced OT-CFM.
"""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import StandardScaler

try:
    import umap  # type: ignore
    HAVE_UMAP = True
except Exception:
    HAVE_UMAP = False

sns.set_theme(style="whitegrid", context="paper")

OUTPUT_DIR = Path('/Users/zhenyizhang/Documents/GitHub/CytoBridge-agent-runtime-v2-20260309/multi-omics')
DATA_PATH = Path('/Users/zhenyizhang/.cellcompass/algorithm_benchmarks/datasets/gse305370_citeseq_stage2_10k/data.h5ad')
FIG_DIR = OUTPUT_DIR / 'figures/downstream_cite_pcfm_gt_protein_comparison'
SCRIPT_PATH = OUTPUT_DIR / 'scripts/downstream/cite_pcfm_gt_protein_comparison.py'
OUT_DIR = OUTPUT_DIR / 'downstream/cite_pcfm_gt_protein_comparison'
TAB_DIR = OUT_DIR / 'tables'
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

METHODS = {
    'CITE-PCFM': {
        'trajectory': OUTPUT_DIR / 'training_runs/20260526T174204321821Z__final__cite_paired_coord_fm__campaign_20260-trial_20260526-gse305370_ci/artifacts/evaluation_trajectory.npz',
        'metrics': OUTPUT_DIR / 'training_runs/20260526T174204321821Z__final__cite_paired_coord_fm__campaign_20260-trial_20260526-gse305370_ci/artifacts/metrics.json',
    },
    'CRUFM': {
        'trajectory': OUTPUT_DIR / 'training_runs/20260525T220649168798Z__pilot__crufm__campaign_20260-stage2_c-baseline-crufm-gse305370_ci/artifacts/evaluation_trajectory.npz',
        'metrics': OUTPUT_DIR / 'training_runs/20260525T220649168798Z__pilot__crufm__campaign_20260-stage2_c-baseline-crufm-gse305370_ci/artifacts/metrics.json',
    },
    'balanced OT-CFM': {
        'trajectory': OUTPUT_DIR / 'training_runs/20260525T215826526439Z__pilot__balanced_ot_cfm__campaign_20260-stage2_c-baseline-balanced_ot_-gse305370_ci/artifacts/evaluation_trajectory.npz',
        'metrics': OUTPUT_DIR / 'training_runs/20260525T215826526439Z__pilot__balanced_ot_cfm__campaign_20260-stage2_c-baseline-balanced_ot_-gse305370_ci/artifacts/metrics.json',
    },
}
RNA_DIM = 50
PROT_DIM = 30
RNG = np.random.default_rng(20260527)
TIMEPOINTS = [1.0, 2.0, 3.0, 4.0]
ALL_TIMES = [0.0, 1.0, 2.0, 3.0, 4.0]


def sample_idx(n, max_n):
    if n <= max_n:
        return np.arange(n)
    return np.sort(RNG.choice(n, size=max_n, replace=False))


def load_traj(path: Path):
    z = np.load(path)
    time_points = z['time_points']
    points = {}
    for i, t in enumerate(time_points):
        k = f'points_{i:06d}'
        if k in z:
            points[float(t)] = z[k]
    return points


def energy_distance(X, Y, max_n=800):
    ix = sample_idx(X.shape[0], max_n)
    iy = sample_idx(Y.shape[0], max_n)
    Xs, Ys = X[ix], Y[iy]
    dxy = pairwise_distances(Xs, Ys).mean()
    dxx = pairwise_distances(Xs, Xs).mean()
    dyy = pairwise_distances(Ys, Ys).mean()
    return float(2*dxy - dxx - dyy)


def mmd_rbf(X, Y, gamma=None, max_n=700):
    ix = sample_idx(X.shape[0], max_n)
    iy = sample_idx(Y.shape[0], max_n)
    Xs, Ys = X[ix], Y[iy]
    if gamma is None:
        Z = np.vstack([Xs, Ys])
        # robust median heuristic on a subsample
        iz = sample_idx(Z.shape[0], min(500, Z.shape[0]))
        D = pairwise_distances(Z[iz], Z[iz])
        med = np.median(D[D > 0]) if np.any(D > 0) else 1.0
        gamma = 1.0 / (2 * med * med + 1e-9)
    Kxx = np.exp(-gamma * pairwise_distances(Xs, Xs, squared=True)).mean()
    Kyy = np.exp(-gamma * pairwise_distances(Ys, Ys, squared=True)).mean()
    Kxy = np.exp(-gamma * pairwise_distances(Xs, Ys, squared=True)).mean()
    return float(Kxx + Kyy - 2*Kxy)


def nearest_centroid_summary(P, centroids, labels):
    C = np.vstack([centroids[c] for c in labels])
    D = pairwise_distances(P, C)
    idx = D.argmin(axis=1)
    return np.asarray(labels, dtype=object)[idx], D[np.arange(P.shape[0]), idx]


def main():
    adata = ad.read_h5ad(DATA_PATH)
    obs = adata.obs.copy()
    obs_time = obs['time_point_processed'].astype(float).to_numpy()
    cell_type = obs['cell_type'].astype(str).to_numpy()
    rna_obs = np.asarray(adata.obsm['X_rna_svd'], dtype=np.float32)
    prot_obs = np.asarray(adata.obsm['X_protein_pca'], dtype=np.float32)
    labels = sorted(pd.unique(cell_type))

    # Observed protein phenotype centroids and classifiers.
    centroids = {c: prot_obs[cell_type == c].mean(axis=0) for c in labels}
    prot_scale = prot_obs.std(axis=0) + 1e-6
    masp_centroid = centroids['MasP']
    def masp_aff(P):
        d = np.linalg.norm((P - masp_centroid) / prot_scale, axis=1)
        return np.exp(-0.5 * d)

    knn_prot = KNeighborsClassifier(n_neighbors=25, weights='distance')
    knn_prot.fit(prot_obs, cell_type)

    # Protein UMAP/PCA embedding fit jointly on observed protein and sampled generated protein.
    trajs = {m: load_traj(spec['trajectory']) for m, spec in METHODS.items()}
    fit_blocks = [prot_obs[sample_idx(prot_obs.shape[0], 5000)]]
    block_refs = [('Observed_all', None, 0, len(fit_blocks[0]))]
    cursor = len(fit_blocks[0])
    gen_sample_refs = []
    for method, pts_by_t in trajs.items():
        for t in TIMEPOINTS:
            P = pts_by_t[t][:, RNA_DIM:RNA_DIM+PROT_DIM]
            idx = sample_idx(P.shape[0], 500)
            fit_blocks.append(P[idx])
            gen_sample_refs.append((method, t, idx, cursor, cursor+len(idx)))
            cursor += len(idx)
    fit = np.vstack(fit_blocks)
    scaler_p = StandardScaler().fit(fit)
    fit_s = scaler_p.transform(fit)
    if HAVE_UMAP:
        reducer_p = umap.UMAP(n_neighbors=35, min_dist=0.25, random_state=20260527, metric='euclidean')
        emb = reducer_p.fit_transform(fit_s)
        emb_name = 'protein UMAP'
    else:
        reducer_p = PCA(n_components=2, random_state=20260527)
        emb = reducer_p.fit_transform(fit_s)
        emb_name = 'protein PCA projection'

    # Transform all observed per-time by fitting reducer jointly only for plotting? For UMAP transform is available.
    # Use fit embedding for sampled observed background and generated samples.
    obs_bg_n = block_refs[0][3]
    obs_bg_idx = sample_idx(prot_obs.shape[0], 5000)
    # block fit used a fresh sample_idx call in fit_blocks[0]; reproduce not possible, so make observed plotting transform directly.
    if HAVE_UMAP:
        obs_emb_all = reducer_p.transform(scaler_p.transform(prot_obs))
    else:
        obs_emb_all = reducer_p.transform(scaler_p.transform(prot_obs))

    gen_emb_rows = []
    for method, t, idx, start, end in gen_sample_refs:
        pts = trajs[method][t][idx]
        P = pts[:, RNA_DIM:RNA_DIM+PROT_DIM]
        pred = knn_prot.predict(P)
        e = emb[start:end]
        gen_emb_rows.append(pd.DataFrame({
            'method': method, 'time': t, 'umap1': e[:,0], 'umap2': e[:,1],
            'masp_affinity': masp_aff(P), 'predicted_phenotype': pred
        }))
    gen_emb_df = pd.concat(gen_emb_rows, ignore_index=True)
    obs_emb_df = pd.DataFrame({
        'time': obs_time, 'cell_type': cell_type,
        'umap1': obs_emb_all[:,0], 'umap2': obs_emb_all[:,1],
        'masp_affinity': masp_aff(prot_obs)
    })

    # Shared RNA UMAP for residual maps.
    rna_fit_blocks = [rna_obs[sample_idx(rna_obs.shape[0], 5000)]]
    rna_refs = []
    cursor = rna_fit_blocks[0].shape[0]
    for method, pts_by_t in trajs.items():
        for t in TIMEPOINTS:
            R = pts_by_t[t][:, :RNA_DIM]
            idx = sample_idx(R.shape[0], 500)
            rna_fit_blocks.append(R[idx])
            rna_refs.append((method, t, idx, cursor, cursor+len(idx)))
            cursor += len(idx)
    rna_fit = np.vstack(rna_fit_blocks)
    scaler_r = StandardScaler().fit(rna_fit)
    if HAVE_UMAP:
        reducer_r = umap.UMAP(n_neighbors=40, min_dist=0.25, random_state=20260528, metric='euclidean')
        rna_emb = reducer_r.fit_transform(scaler_r.transform(rna_fit))
        rna_emb_name = 'RNA UMAP'
    else:
        reducer_r = PCA(n_components=2, random_state=20260528)
        rna_emb = reducer_r.fit_transform(scaler_r.transform(rna_fit))
        rna_emb_name = 'RNA PCA projection'
    obs_rna_emb = reducer_r.transform(scaler_r.transform(rna_obs)) if HAVE_UMAP else reducer_r.transform(scaler_r.transform(rna_obs))

    # For expected observed protein signal on RNA UMAP, use observed nearest RNA neighbors at matched time.
    residual_rows = []
    for method, t, idx, start, end in rna_refs:
        pts = trajs[method][t][idx]
        R = pts[:, :RNA_DIM]
        P = pts[:, RNA_DIM:RNA_DIM+PROT_DIM]
        mask = obs_time == t
        nn = NearestNeighbors(n_neighbors=35).fit(rna_obs[mask])
        neigh = nn.kneighbors(R, return_distance=False)
        obs_aff_local = masp_aff(prot_obs[mask])
        expected = obs_aff_local[neigh].mean(axis=1)
        generated = masp_aff(P)
        e = rna_emb[start:end]
        residual_rows.append(pd.DataFrame({
            'method': method, 'time': t, 'umap1': e[:,0], 'umap2': e[:,1],
            'generated_masp_affinity': generated,
            'expected_observed_masp_affinity': expected,
            'residual_generated_minus_observed': generated - expected,
            'abs_residual': np.abs(generated - expected),
        }))
    residual_df = pd.concat(residual_rows, ignore_index=True)

    # Metrics against observed target protein distributions and centroid/phenotype summaries.
    metric_rows = []
    frac_rows = []
    centroid_rows = []
    # observed fractions per time
    for t in ALL_TIMES:
        mask = obs_time == t
        counts = pd.Series(cell_type[mask]).value_counts(normalize=True).reindex(labels, fill_value=0)
        for lab, val in counts.items():
            frac_rows.append({'source': 'Observed', 'method': 'Observed', 'time': t, 'phenotype': lab, 'fraction': float(val)})
    for method, pts_by_t in trajs.items():
        for t in TIMEPOINTS:
            Pgen = pts_by_t[t][:, RNA_DIM:RNA_DIM+PROT_DIM]
            Pobs = prot_obs[obs_time == t]
            metric_rows.append({
                'method': method, 'time': t,
                'protein_energy_distance': energy_distance(Pgen, Pobs),
                'protein_mmd_rbf': mmd_rbf(Pgen, Pobs),
                'centroid_l2': float(np.linalg.norm(Pgen.mean(axis=0) - Pobs.mean(axis=0))),
                'mean_abs_masp_affinity_residual_on_rna_umap': float(residual_df[(residual_df.method==method)&(residual_df.time==t)].abs_residual.mean()),
            })
            pred = knn_prot.predict(Pgen)
            counts = pd.Series(pred).value_counts(normalize=True).reindex(labels, fill_value=0)
            for lab, val in counts.items():
                frac_rows.append({'source': 'Generated', 'method': method, 'time': t, 'phenotype': lab, 'fraction': float(val)})
            # nearest observed phenotype centroid displacement, generated vs observed target cells
            gen_lab, gen_dist = nearest_centroid_summary(Pgen, centroids, labels)
            obs_lab, obs_dist = nearest_centroid_summary(Pobs, centroids, labels)
            centroid_rows.append({
                'method': method, 'time': t,
                'generated_mean_nearest_centroid_distance': float(gen_dist.mean()),
                'observed_mean_nearest_centroid_distance': float(obs_dist.mean()),
                'excess_nearest_centroid_distance': float(gen_dist.mean() - obs_dist.mean()),
                'generated_masp_centroid_distance': float(np.linalg.norm(Pgen.mean(axis=0) - masp_centroid)),
                'observed_masp_centroid_distance': float(np.linalg.norm(Pobs.mean(axis=0) - masp_centroid)),
            })

    metric_df = pd.DataFrame(metric_rows)
    frac_df = pd.DataFrame(frac_rows)
    centroid_df = pd.DataFrame(centroid_rows)
    obs_emb_df.to_csv(TAB_DIR/'observed_protein_embedding_points.csv', index=False)
    gen_emb_df.to_csv(TAB_DIR/'generated_protein_embedding_points.csv', index=False)
    residual_df.to_csv(TAB_DIR/'rna_umap_masp_affinity_residuals.csv', index=False)
    metric_df.to_csv(TAB_DIR/'per_time_protein_discrepancy_against_observed.csv', index=False)
    frac_df.to_csv(TAB_DIR/'observed_vs_generated_protein_phenotype_fractions.csv', index=False)
    centroid_df.to_csv(TAB_DIR/'protein_centroid_displacement_summary.csv', index=False)

    # Figure 1: observed vs generated in protein space, side-by-side for each method/time.
    methods = list(METHODS.keys())
    times = TIMEPOINTS
    fig, axes = plt.subplots(len(times), len(methods)+1, figsize=(3.0*(len(methods)+1), 2.7*len(times)), sharex=True, sharey=True, constrained_layout=True)
    palette = dict(zip(labels, sns.color_palette('tab10', n_colors=len(labels))))
    for i, t in enumerate(times):
        ax = axes[i,0]
        d = obs_emb_df[obs_emb_df.time==t]
        idx = sample_idx(len(d), 800)
        dd = d.iloc[idx]
        for lab in labels:
            sub=dd[dd.cell_type==lab]
            ax.scatter(sub.umap1, sub.umap2, s=5, alpha=0.55, color=palette[lab], linewidths=0)
        if i==0: ax.set_title('Observed target')
        ax.set_ylabel(f't={t:g}')
        for j, method in enumerate(methods, start=1):
            ax=axes[i,j]
            gd=gen_emb_df[(gen_emb_df.method==method)&(gen_emb_df.time==t)]
            for lab in labels:
                sub=gd[gd.predicted_phenotype==lab]
                ax.scatter(sub.umap1, sub.umap2, s=7, alpha=0.65, color=palette[lab], linewidths=0)
            if i==0: ax.set_title(method)
            ax.set_xticks([]); ax.set_yticks([])
    handles=[plt.Line2D([0],[0],marker='o',color='w',label=lab,markerfacecolor=palette[lab],markersize=5) for lab in labels]
    fig.legend(handles=handles, loc='lower center', ncol=len(labels), bbox_to_anchor=(0.5,-0.01), fontsize=7)
    fig.suptitle(f'Observed vs generated target protein distributions in shared {emb_name}', y=1.01)
    fig.savefig(FIG_DIR/'fig20_observed_vs_generated_protein_space_by_time.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIG_DIR/'fig20_observed_vs_generated_protein_space_by_time.pdf', bbox_inches='tight')
    plt.close(fig)

    # Figure 2: residual map on RNA UMAP, generated minus expected observed protein signal.
    vmax = np.nanpercentile(np.abs(residual_df.residual_generated_minus_observed), 98)
    fig, axes = plt.subplots(len(times), len(methods), figsize=(3.2*len(methods), 2.8*len(times)), sharex=True, sharey=True, constrained_layout=True)
    for i,t in enumerate(times):
        mask=obs_time==t
        bg_idx=sample_idx(mask.sum(), 700)
        bg=obs_rna_emb[mask][bg_idx]
        for j,method in enumerate(methods):
            ax=axes[i,j]
            ax.scatter(bg[:,0], bg[:,1], s=4, color='lightgray', alpha=0.18, linewidths=0)
            d=residual_df[(residual_df.method==method)&(residual_df.time==t)]
            sc=ax.scatter(d.umap1,d.umap2,c=d.residual_generated_minus_observed,cmap='coolwarm',vmin=-vmax,vmax=vmax,s=9,alpha=0.9,linewidths=0)
            if i==0: ax.set_title(method)
            if j==0: ax.set_ylabel(f't={t:g}')
            ax.set_xticks([]); ax.set_yticks([])
    cb=fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.65, pad=0.01)
    cb.set_label('Generated - observed-expected MasP protein affinity')
    fig.suptitle(f'Where methods make protein-signal errors on shared {rna_emb_name}', y=1.01)
    fig.savefig(FIG_DIR/'fig21_rna_umap_generated_minus_observed_protein_residuals.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIG_DIR/'fig21_rna_umap_generated_minus_observed_protein_residuals.pdf', bbox_inches='tight')
    plt.close(fig)

    # Figure 3: time-by-method heatmap of protein discrepancy against observed target cells.
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.5), constrained_layout=True)
    for ax, metric, title in zip(axes, ['protein_energy_distance','protein_mmd_rbf','mean_abs_masp_affinity_residual_on_rna_umap'], ['Energy distance in protein PCA','RBF MMD in protein PCA','Mean |protein-signal residual| on RNA UMAP']):
        mat=metric_df.pivot(index='method', columns='time', values=metric).loc[methods]
        sns.heatmap(mat, annot=True, fmt='.3f', cmap='viridis_r', ax=ax, cbar=True)
        ax.set_title(title)
        ax.set_xlabel('target time'); ax.set_ylabel('')
    fig.suptitle('Protein discrepancy against observed target cells; lower is better', y=1.08)
    fig.savefig(FIG_DIR/'fig22_time_by_method_observed_protein_discrepancy_heatmap.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIG_DIR/'fig22_time_by_method_observed_protein_discrepancy_heatmap.pdf', bbox_inches='tight')
    plt.close(fig)

    # Figure 4: observed vs generated phenotype fractions over time for key phenotypes.
    key_pheno=[x for x in ['HSC','MasP','MoP','EryP','MkP','NeuP','BP'] if x in labels]
    fig, axes = plt.subplots(1, len(methods), figsize=(4.4*len(methods), 4.0), sharey=True, constrained_layout=True)
    for ax, method in zip(axes, methods):
        d_obs=frac_df[(frac_df.source=='Observed') & (frac_df.phenotype.isin(key_pheno))]
        d_gen=frac_df[(frac_df.method==method) & (frac_df.source=='Generated') & (frac_df.phenotype.isin(key_pheno))]
        for ph in key_pheno:
            o=d_obs[d_obs.phenotype==ph].sort_values('time')
            g=d_gen[d_gen.phenotype==ph].sort_values('time')
            ax.plot(o.time,o.fraction,color=palette[ph],lw=2.0,alpha=0.9,label=f'Observed {ph}' if method==methods[0] else None)
            ax.plot(g.time,g.fraction,color=palette[ph],lw=1.5,alpha=0.9,ls='--',label=f'Generated {ph}' if method==methods[0] else None)
        ax.set_title(method)
        ax.set_xlabel('time')
        ax.set_ylim(0, 0.75)
    axes[0].set_ylabel('phenotype fraction')
    fig.legend(loc='lower center', ncol=4, fontsize=7, bbox_to_anchor=(0.5,-0.05))
    fig.suptitle('Do generated protein phenotypes follow observed target fractions? solid=observed, dashed=generated', y=1.03)
    fig.savefig(FIG_DIR/'fig23_observed_vs_generated_protein_phenotype_fractions.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIG_DIR/'fig23_observed_vs_generated_protein_phenotype_fractions.pdf', bbox_inches='tight')
    plt.close(fig)

    # Figure 5: centroid displacement / nearest-centroid excess.
    cdf=centroid_df.merge(metric_df[['method','time','centroid_l2']],on=['method','time'])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
    sns.lineplot(data=cdf, x='time', y='centroid_l2', hue='method', marker='o', ax=axes[0])
    axes[0].set_title('Generated vs observed protein centroid displacement')
    axes[0].set_ylabel('L2 distance between protein centroids')
    sns.lineplot(data=cdf, x='time', y='excess_nearest_centroid_distance', hue='method', marker='o', ax=axes[1])
    axes[1].axhline(0, color='black', lw=1, ls=':')
    axes[1].set_title('Excess nearest phenotype-centroid error')
    axes[1].set_ylabel('generated mean NN-centroid distance - observed')
    fig.suptitle('Visual mismatch summary in observed protein-phenotype space', y=1.04)
    fig.savefig(FIG_DIR/'fig24_protein_centroid_displacement_summary.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIG_DIR/'fig24_protein_centroid_displacement_summary.pdf', bbox_inches='tight')
    plt.close(fig)

    # Summaries: winners per metric/time and aggregate.
    winners=[]
    for metric in ['protein_energy_distance','protein_mmd_rbf','centroid_l2','mean_abs_masp_affinity_residual_on_rna_umap']:
        for t in TIMEPOINTS:
            d=metric_df[metric_df.time==t].sort_values(metric)
            winners.append({'metric':metric,'time':t,'best_method':d.iloc[0].method,'best_value':float(d.iloc[0][metric])})
    winner_df=pd.DataFrame(winners)
    winner_df.to_csv(TAB_DIR/'per_time_metric_winners.csv', index=False)
    agg=metric_df.groupby('method').mean(numeric_only=True).reset_index().sort_values('protein_energy_distance')
    agg.to_csv(TAB_DIR/'aggregate_metric_summary.csv', index=False)

    manifest={
        'analysis':'Ground-truth-first observed-vs-generated CITE protein dynamics comparison',
        'dataset':str(DATA_PATH),
        'script':str(SCRIPT_PATH),
        'methods':{m:{k:str(v) for k,v in spec.items()} for m,spec in METHODS.items()},
        'embedding_contract':{
            'observed_rna':'obsm["X_rna_svd"]',
            'observed_protein':'obsm["X_protein_pca"]',
            'time':'obs["time_point_processed"]',
            'labels':'obs["cell_type"]',
            'generated':'evaluation_trajectory.npz points split into RNA dims 0:50 and protein dims 50:80'
        },
        'figures':[str(p) for p in sorted(FIG_DIR.glob('fig*.png'))],
        'tables':[str(p) for p in sorted(TAB_DIR.glob('*.csv'))],
        'aggregate_metric_summary':agg.to_dict(orient='records'),
        'winner_counts':winner_df.groupby(['metric','best_method']).size().reset_index(name='n_timepoints').to_dict(orient='records'),
        'interpretation_boundary':'All main figures compare generated outputs to observed target cells at matched time points. Protein signals are latent X_protein_pca phenotype signals, not named antibody intensities.'
    }
    with open(OUT_DIR/'gt_protein_comparison_manifest.json','w') as f:
        json.dump(manifest,f,indent=2)
    print(json.dumps(manifest, indent=2)[:6000])

if __name__ == '__main__':
    main()
