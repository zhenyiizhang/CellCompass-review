#!/usr/bin/env python3
"""Extensive downstream analysis for CITE-PCFM.

Generates a rich set of observed-vs-generated and method-comparison figures
across protein distribution fidelity, RNA/protein embeddings, phenotype readouts,
centroid/variance/covariance structure, local residuals, and temporal dynamics.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil

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
RNG = np.random.default_rng(20260527)
OUTPUT_DIR = Path('/Users/zhenyizhang/Documents/GitHub/CytoBridge-agent-runtime-v2-20260309/multi-omics')
DATA_PATH = Path('/Users/zhenyizhang/.cellcompass/algorithm_benchmarks/datasets/gse305370_citeseq_stage2_10k/data.h5ad')
FIG_DIR = OUTPUT_DIR / 'figures/downstream_cite_pcfm_extensive'
OUT_DIR = OUTPUT_DIR / 'downstream/cite_pcfm_extensive'
TAB_DIR = OUT_DIR / 'tables'
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

METHODS = {
    'CITE-PCFM': OUTPUT_DIR / 'training_runs/20260526T174204321821Z__final__cite_paired_coord_fm__campaign_20260-trial_20260526-gse305370_ci/artifacts/evaluation_trajectory.npz',
    'CRUFM': OUTPUT_DIR / 'training_runs/20260525T220649168798Z__pilot__crufm__campaign_20260-stage2_c-baseline-crufm-gse305370_ci/artifacts/evaluation_trajectory.npz',
    'balanced OT-CFM': OUTPUT_DIR / 'training_runs/20260525T215826526439Z__pilot__balanced_ot_cfm__campaign_20260-stage2_c-baseline-balanced_ot_-gse305370_ci/artifacts/evaluation_trajectory.npz',
}
METRICS = {
    'CITE-PCFM': OUTPUT_DIR / 'training_runs/20260526T174204321821Z__final__cite_paired_coord_fm__campaign_20260-trial_20260526-gse305370_ci/artifacts/metrics.json',
    'CRUFM': OUTPUT_DIR / 'training_runs/20260525T220649168798Z__pilot__crufm__campaign_20260-stage2_c-baseline-crufm-gse305370_ci/artifacts/metrics.json',
    'balanced OT-CFM': OUTPUT_DIR / 'training_runs/20260525T215826526439Z__pilot__balanced_ot_cfm__campaign_20260-stage2_c-baseline-balanced_ot_-gse305370_ci/artifacts/metrics.json',
}
RNA_DIM, PROT_DIM = 50, 30
TIMES = [0.0,1.0,2.0,3.0,4.0]
TARGET_TIMES = [1.0,2.0,3.0,4.0]
METHOD_COLORS={'CITE-PCFM':'#d62728','CRUFM':'#1f77b4','balanced OT-CFM':'#2ca02c'}

def sample_idx(n, max_n):
    if n <= max_n: return np.arange(n)
    return np.sort(RNG.choice(n, max_n, replace=False))

def load_traj(path):
    z=np.load(path)
    t=z['time_points']
    pts={float(tt): z[f'points_{i:06d}'] for i,tt in enumerate(t) if f'points_{i:06d}' in z}
    return pts

def energy_distance(X,Y,max_n=700):
    X=X[sample_idx(X.shape[0],max_n)]; Y=Y[sample_idx(Y.shape[0],max_n)]
    return float(2*pairwise_distances(X,Y).mean()-pairwise_distances(X,X).mean()-pairwise_distances(Y,Y).mean())

def mmd(X,Y,max_n=600):
    X=X[sample_idx(X.shape[0],max_n)]; Y=Y[sample_idx(Y.shape[0],max_n)]
    Z=np.vstack([X,Y]); iz=sample_idx(Z.shape[0], min(500,Z.shape[0])); D=pairwise_distances(Z[iz],Z[iz]); med=np.median(D[D>0]) if np.any(D>0) else 1.0
    g=1/(2*med*med+1e-9)
    return float(np.exp(-g*pairwise_distances(X,X,squared=True)).mean()+np.exp(-g*pairwise_distances(Y,Y,squared=True)).mean()-2*np.exp(-g*pairwise_distances(X,Y,squared=True)).mean())

def metric_json(path):
    if not path.exists(): return {}
    with open(path) as f: m=json.load(f)
    cm=m.get('custom_metrics',{}) if isinstance(m.get('custom_metrics'),dict) else {}
    out={}
    for k in ['protein_energy','rna_energy_guardrail','paired_cov_error']:
        out[k]=m.get(k,cm.get(k,np.nan))
    if 'w1_scores' in m and isinstance(m['w1_scores'],dict): out['w1_mean']=float(np.mean(list(m['w1_scores'].values())))
    else: out['w1_mean']=m.get('w1_mean',np.nan)
    out['tmv_mean']=m.get('tmv_mean', np.nan)
    return out

def fit_embedding(blocks, kind='umap', seed=0):
    X=np.vstack(blocks); scaler=StandardScaler().fit(X); Xs=scaler.transform(X)
    if HAVE_UMAP and kind=='umap':
        red=umap.UMAP(n_neighbors=35,min_dist=0.25,random_state=seed,metric='euclidean')
    else:
        red=PCA(n_components=2,random_state=seed)
    emb=red.fit_transform(Xs)
    return scaler, red, emb

def transform(red, scaler, X):
    return red.transform(scaler.transform(X))

def savefig(fig, name, caption, inventory):
    png=FIG_DIR/f'{name}.png'; pdf=FIG_DIR/f'{name}.pdf'
    fig.savefig(png,dpi=250,bbox_inches='tight'); fig.savefig(pdf,bbox_inches='tight'); plt.close(fig)
    inventory.append({'path':str(png),'caption':caption})

def main():
    figs=[]
    adata=ad.read_h5ad(DATA_PATH)
    obs=adata.obs.copy(); obs_time=obs['time_point_processed'].astype(float).to_numpy(); cell_type=obs['cell_type'].astype(str).to_numpy()
    rna=np.asarray(adata.obsm['X_rna_svd'],dtype=np.float32)[:,:RNA_DIM]
    prot=np.asarray(adata.obsm['X_protein_pca'],dtype=np.float32)[:,:PROT_DIM]
    labels=sorted(pd.unique(cell_type)); palette=dict(zip(labels,sns.color_palette('tab10',len(labels))))
    traj={m:load_traj(p) for m,p in METHODS.items()}
    centroids={c:prot[cell_type==c].mean(0) for c in labels}
    prot_scale=prot.std(0)+1e-6
    def affinity(P, lab):
        d=np.linalg.norm((P-centroids[lab])/prot_scale, axis=1); return np.exp(-0.5*d)
    knn=KNeighborsClassifier(n_neighbors=25,weights='distance').fit(prot,cell_type)

    # Embeddings shared across observed and generated samples.
    prot_blocks=[prot[sample_idx(prot.shape[0],5000)]]; prot_refs=[]; cur=len(prot_blocks[0])
    rna_blocks=[rna[sample_idx(rna.shape[0],5000)]]; rna_refs=[]; rcur=len(rna_blocks[0])
    for m, pts in traj.items():
        for t in TARGET_TIMES:
            P=pts[t][:,RNA_DIM:RNA_DIM+PROT_DIM]; idx=sample_idx(P.shape[0],450); prot_blocks.append(P[idx]); prot_refs.append((m,t,idx,cur,cur+len(idx))); cur+=len(idx)
            R=pts[t][:,:RNA_DIM]; ridx=sample_idx(R.shape[0],450); rna_blocks.append(R[ridx]); rna_refs.append((m,t,ridx,rcur,rcur+len(ridx))); rcur+=len(ridx)
    ps, pred, pemb=fit_embedding(prot_blocks,'umap',11); rs, rred, remb=fit_embedding(rna_blocks,'umap',12)
    prot_emb_obs=transform(pred,ps,prot); rna_emb_obs=transform(rred,rs,rna)

    # Tables
    rows=[]; frac_rows=[]; cent_rows=[]; resid_rows=[]; cov_rows=[]; var_rows=[]
    for t in TIMES:
        mask=obs_time==t
        counts=pd.Series(cell_type[mask]).value_counts(normalize=True).reindex(labels,fill_value=0)
        for lab,val in counts.items(): frac_rows.append({'method':'Observed','source':'Observed','time':t,'phenotype':lab,'fraction':float(val)})
    for m,pts in traj.items():
        for t in TARGET_TIMES:
            Z=pts[t]; R=Z[:,:RNA_DIM]; P=Z[:,RNA_DIM:RNA_DIM+PROT_DIM]; Po=prot[obs_time==t]; Ro=rna[obs_time==t]
            # RNA-neighborhood residuals for HSC/MasP/MoP/EryP affinities
            nn=NearestNeighbors(n_neighbors=35).fit(Ro); neigh=nn.kneighbors(R,return_distance=False)
            res_abs=[]
            for lab in [x for x in ['HSC','MasP','MoP','EryP','MkP','NeuP'] if x in labels]:
                exp=affinity(Po,lab)[neigh].mean(1); gen=affinity(P,lab); ar=np.abs(gen-exp).mean(); res_abs.append(ar)
                resid_rows.append({'method':m,'time':t,'phenotype':lab,'mean_abs_residual':float(ar),'mean_signed_residual':float((gen-exp).mean())})
            rows.append({'method':m,'time':t,'protein_energy_distance':energy_distance(P,Po),'protein_mmd':mmd(P,Po),'rna_energy_distance':energy_distance(R,Ro),'centroid_l2_protein':float(np.linalg.norm(P.mean(0)-Po.mean(0))),'centroid_l2_rna':float(np.linalg.norm(R.mean(0)-Ro.mean(0))),'mean_abs_affinity_residual':float(np.mean(res_abs))})
            predlab=knn.predict(P); counts=pd.Series(predlab).value_counts(normalize=True).reindex(labels,fill_value=0)
            for lab,val in counts.items(): frac_rows.append({'method':m,'source':'Generated','time':t,'phenotype':lab,'fraction':float(val)})
            C=np.vstack([centroids[c] for c in labels]); D=pairwise_distances(P,C); Do=pairwise_distances(Po,C)
            cent_rows.append({'method':m,'time':t,'mean_nearest_centroid_gen':float(D.min(1).mean()),'mean_nearest_centroid_obs':float(Do.min(1).mean()),'excess_nearest_centroid':float(D.min(1).mean()-Do.min(1).mean())})
            cov_rows.append({'method':m,'time':t,'protein_cov_frobenius':float(np.linalg.norm(np.cov(P,rowvar=False)-np.cov(Po,rowvar=False),'fro')),'rna_cov_frobenius':float(np.linalg.norm(np.cov(R,rowvar=False)-np.cov(Ro,rowvar=False),'fro'))})
            for pc in range(min(10,PROT_DIM)):
                var_rows.append({'method':m,'time':t,'protein_pc':pc+1,'generated_mean':float(P[:,pc].mean()),'observed_mean':float(Po[:,pc].mean()),'mean_error':float(P[:,pc].mean()-Po[:,pc].mean()),'generated_var':float(P[:,pc].var()),'observed_var':float(Po[:,pc].var()),'var_ratio':float((P[:,pc].var()+1e-9)/(Po[:,pc].var()+1e-9))})
    metric_df=pd.DataFrame(rows); frac_df=pd.DataFrame(frac_rows); cent_df=pd.DataFrame(cent_rows); resid_df=pd.DataFrame(resid_rows); cov_df=pd.DataFrame(cov_rows); var_df=pd.DataFrame(var_rows)
    for name,df in [('matched_time_metrics',metric_df),('phenotype_fractions',frac_df),('centroid_errors',cent_df),('affinity_residuals',resid_df),('covariance_errors',cov_df),('protein_pc_mean_variance_errors',var_df)]: df.to_csv(TAB_DIR/f'{name}.csv',index=False)

    # 1 metrics overview, 2-5 per metric heatmaps
    mj=[]
    for m in METHODS: d=metric_json(METRICS[m]); d['method']=m; mj.append(d)
    mj=pd.DataFrame(mj); mj.to_csv(TAB_DIR/'campaign_metric_context.csv',index=False)
    fig,ax=plt.subplots(figsize=(7,4)); sns.barplot(data=mj.melt(id_vars='method',value_vars=['protein_energy','paired_cov_error','rna_energy_guardrail']),x='variable',y='value',hue='method',ax=ax); ax.set_title('Campaign metric context (lower is better)'); savefig(fig,'fig01_campaign_metric_context','Campaign-level metric context.',figs)
    for k,name in enumerate(['protein_energy_distance','protein_mmd','rna_energy_distance','mean_abs_affinity_residual'],start=2):
        fig,ax=plt.subplots(figsize=(5,3.2)); mat=metric_df.pivot(index='method',columns='time',values=name).loc[list(METHODS)] ; sns.heatmap(mat,annot=True,fmt='.3f',cmap='viridis_r',ax=ax); ax.set_title(name); savefig(fig,f'fig{k:02d}_{name}_heatmap',f'Time-by-method heatmap for {name}.',figs)
    # 6 centroid/cov summaries
    for k,(df,val,title) in enumerate([(metric_df,'centroid_l2_protein','Protein centroid L2'),(cent_df,'excess_nearest_centroid','Excess nearest phenotype centroid'),(cov_df,'protein_cov_frobenius','Protein covariance error'),(cov_df,'rna_cov_frobenius','RNA covariance error')],start=6):
        fig,ax=plt.subplots(figsize=(6,3.5)); sns.lineplot(data=df,x='time',y=val,hue='method',marker='o',ax=ax); ax.set_title(title); savefig(fig,f'fig{k:02d}_{val}',title,figs)
    # 10-13 observed distributions by time in RNA/protein embeddings
    for t in TARGET_TIMES:
        for emb,arr,name,baseid in [('protein',prot_emb_obs,'Observed target protein UMAP',10),('rna',rna_emb_obs,'Observed target RNA UMAP',14)]:
            fig,ax=plt.subplots(figsize=(4.2,3.8)); mask=obs_time==t; idx=sample_idx(mask.sum(),1000); sub=np.where(mask)[0][idx]
            for lab in labels:
                ii=sub[cell_type[sub]==lab]; ax.scatter(arr[ii,0],arr[ii,1],s=6,alpha=.55,color=palette[lab],label=lab if t==1 and emb=='protein' else None,linewidths=0)
            ax.set_title(f'{name}, t={t:g}'); ax.set_xticks([]); ax.set_yticks([]); savefig(fig,f'fig{baseid+int(t)-1:02d}_{emb}_observed_t{int(t)}',f'Observed {emb} embedding at t={t:g}.',figs)
    # 18-29 generated vs observed protein embedding per method/time (12 figures)
    c=18
    for m,pts in traj.items():
        for t in TARGET_TIMES:
            fig,ax=plt.subplots(figsize=(4.3,3.8)); mask=obs_time==t; oi=np.where(mask)[0][sample_idx(mask.sum(),800)]; ax.scatter(prot_emb_obs[oi,0],prot_emb_obs[oi,1],s=5,c='lightgray',alpha=.35,label='Observed target',linewidths=0)
            P=pts[t][:,RNA_DIM:RNA_DIM+PROT_DIM]; idx=sample_idx(P.shape[0],600); E=transform(pred,ps,P[idx]); labs=knn.predict(P[idx])
            for lab in labels:
                sub=labs==lab; ax.scatter(E[sub,0],E[sub,1],s=8,alpha=.75,color=palette[lab],linewidths=0)
            ax.set_title(f'{m}: generated vs observed protein, t={t:g}'); ax.set_xticks([]); ax.set_yticks([]); savefig(fig,f'fig{c:02d}_protein_gt_{m.replace(" ","_").replace("-","")}_t{int(t)}',f'{m} generated protein vs observed target t={t:g}.',figs); c+=1
    # 30-41 RNA residual maps per method/time for MasP affinity
    vmax=None
    # precompute residual plot rows with coords for MasP
    rr=[]
    for m,pts in traj.items():
        for t in TARGET_TIMES:
            Z=pts[t]; R=Z[:,:RNA_DIM]; P=Z[:,RNA_DIM:RNA_DIM+PROT_DIM]; idx=sample_idx(R.shape[0],600); Ri=R[idx]; Pi=P[idx]
            mask=obs_time==t; nn=NearestNeighbors(n_neighbors=35).fit(rna[mask]); neigh=nn.kneighbors(Ri,return_distance=False)
            exp=affinity(prot[mask],'MasP')[neigh].mean(1); gen=affinity(Pi,'MasP'); E=transform(rred,rs,Ri)
            for a,b,g,e in zip(E[:,0],E[:,1],gen,exp): rr.append({'method':m,'time':t,'x':a,'y':b,'resid':g-e})
    rr=pd.DataFrame(rr); vmax=np.percentile(np.abs(rr.resid),98)
    c=30
    for m in METHODS:
        for t in TARGET_TIMES:
            fig,ax=plt.subplots(figsize=(4.2,3.8)); mask=obs_time==t; oi=np.where(mask)[0][sample_idx(mask.sum(),800)]; ax.scatter(rna_emb_obs[oi,0],rna_emb_obs[oi,1],s=4,c='lightgray',alpha=.22,linewidths=0)
            d=rr[(rr.method==m)&(rr.time==t)]; sc=ax.scatter(d.x,d.y,c=d.resid,cmap='coolwarm',vmin=-vmax,vmax=vmax,s=10,alpha=.9,linewidths=0); ax.set_title(f'{m}: MasP residual, t={t:g}'); ax.set_xticks([]); ax.set_yticks([]); fig.colorbar(sc,ax=ax,label='gen - obs expected'); savefig(fig,f'fig{c:02d}_rna_residual_{m.replace(" ","_").replace("-","")}_t{int(t)}',f'{m} RNA UMAP MasP protein residual t={t:g}.',figs); c+=1
    rr.to_csv(TAB_DIR/'masp_residual_plot_points.csv',index=False)
    # 42-48 phenotype fractions per phenotype
    c=42
    for lab in [x for x in ['HSC','MasP','MoP','EryP','MkP','NeuP','BP'] if x in labels]:
        fig,ax=plt.subplots(figsize=(5.2,3.6)); od=frac_df[(frac_df.method=='Observed')&(frac_df.phenotype==lab)].sort_values('time'); ax.plot(od.time,od.fraction,color='black',lw=2.5,label='Observed')
        for m in METHODS:
            gd=frac_df[(frac_df.method==m)&(frac_df.phenotype==lab)].sort_values('time'); ax.plot(gd.time,gd.fraction,marker='o',color=METHOD_COLORS[m],label=m)
        ax.set_title(f'Observed vs generated {lab} protein phenotype fraction'); ax.set_ylim(0, max(.1, ax.get_ylim()[1])); ax.legend(fontsize=7); savefig(fig,f'fig{c:02d}_phenotype_fraction_{lab}',f'Observed versus generated {lab} protein phenotype fractions.',figs); c+=1
    # 49-58 protein PC mean errors and variances
    for pc in range(1,6):
        fig,axes=plt.subplots(1,2,figsize=(8.5,3.3),constrained_layout=True); d=var_df[var_df.protein_pc==pc]
        sns.lineplot(data=d,x='time',y='mean_error',hue='method',marker='o',ax=axes[0]); axes[0].axhline(0,c='k',ls=':'); axes[0].set_title(f'Protein PC{pc} mean error')
        sns.lineplot(data=d,x='time',y='var_ratio',hue='method',marker='o',ax=axes[1]); axes[1].axhline(1,c='k',ls=':'); axes[1].set_title(f'Protein PC{pc} variance ratio')
        savefig(fig,f'fig{48+pc:02d}_protein_pc{pc}_mean_variance_error',f'Protein PC{pc} mean and variance errors.',figs)
    # 54-56 aggregate rank bars
    rank_rows=[]
    for metric in ['protein_energy_distance','protein_mmd','rna_energy_distance','centroid_l2_protein','mean_abs_affinity_residual']:
        for t in TARGET_TIMES:
            d=metric_df[metric_df.time==t].sort_values(metric)
            for rank,(_,row) in enumerate(d.iterrows(),1): rank_rows.append({'metric':metric,'time':t,'method':row.method,'rank':rank})
    rank_df=pd.DataFrame(rank_rows); rank_df.to_csv(TAB_DIR/'per_metric_ranks.csv',index=False)
    fig,ax=plt.subplots(figsize=(7,3.8)); sns.barplot(data=rank_df.groupby('method')['rank'].mean().reset_index(),x='method',y='rank',palette=METHOD_COLORS,ax=ax); ax.set_title('Average rank across matched-time diagnostic metrics'); ax.set_ylabel('mean rank (lower better)'); savefig(fig,'fig54_average_metric_rank','Average method rank across diagnostic metrics.',figs)
    # final combined summary scatter figs 55-56
    agg=metric_df.groupby('method').mean(numeric_only=True).reset_index(); agg.to_csv(TAB_DIR/'aggregate_summary.csv',index=False)
    fig,ax=plt.subplots(figsize=(5,4)); sns.scatterplot(data=agg,x='rna_energy_distance',y='protein_energy_distance',hue='method',s=120,palette=METHOD_COLORS,ax=ax); ax.set_title('Aggregate RNA vs protein distribution error'); savefig(fig,'fig55_aggregate_rna_vs_protein_error','Aggregate RNA/protein error tradeoff.',figs)
    fig,ax=plt.subplots(figsize=(5,4)); sns.scatterplot(data=agg,x='centroid_l2_protein',y='mean_abs_affinity_residual',hue='method',s=120,palette=METHOD_COLORS,ax=ax); ax.set_title('Centroid vs local protein residual tradeoff'); savefig(fig,'fig56_centroid_vs_local_residual','Protein centroid and local residual tradeoff.',figs)

    manifest={'analysis':'Extensive CITE-PCFM downstream analysis','n_figures_png':len(figs),'figures':figs,'tables':[str(p) for p in sorted(TAB_DIR.glob('*.csv'))],'dataset':str(DATA_PATH),'methods':{m:str(p) for m,p in METHODS.items()},'key_aggregate_summary':agg.to_dict(orient='records'),'notes':['All generated trajectories are compared to observed target cells at matched time points where applicable.','Protein analyses use latent X_protein_pca coordinates, not named antibody intensities.','Phenotype readouts use observed cell_type labels/classifier in protein space; they are not lineage-tracing truth.']}
    with open(OUT_DIR/'extensive_downstream_manifest.json','w') as f: json.dump(manifest,f,indent=2)
    print(json.dumps({'n_figures_png':len(figs),'manifest':str(OUT_DIR/'extensive_downstream_manifest.json'),'aggregate':manifest['key_aggregate_summary']}, indent=2))

if __name__=='__main__': main()
