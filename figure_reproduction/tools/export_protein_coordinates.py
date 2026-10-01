"""Freeze display coordinates from final stored model trajectories (no training).

This maintainer-only step computes the original seeded UMAP display transform.
The public rebuild uses these saved coordinates and does not fit UMAP or a model.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
import anndata as ad
from sklearn.preprocessing import StandardScaler
import umap

def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    args=p.parse_args()
    root=Path(__file__).resolve().parents[1];out=root/'plot_data/protein';out.mkdir(parents=True,exist_ok=True)
    a=ad.read_h5ad(args.data)
    protein=np.asarray(a.obsm['X_protein_pca'],dtype=np.float32)
    rna=np.asarray(a.obsm['X_rna_svd'],dtype=np.float32)
    def embed(x):
        scaler=StandardScaler().fit(x)
        reducer=umap.UMAP(n_neighbors=35,min_dist=.22,metric='euclidean',random_state=20260528,low_memory=True)
        xy=reducer.fit_transform(scaler.transform(x))
        return scaler,reducer,xy
    print('Projecting saved observed protein coordinates',flush=True)
    scale,reducer,observed=embed(protein)
    print('Projecting saved observed RNA coordinates',flush=True)
    _,_,rna_xy=embed(rna)
    obs=a.obs[['cell_type','time_point_processed']].copy()
    obs[['protein_x','protein_y']]=observed
    obs[['rna_x','rna_y']]=rna_xy
    obs.to_csv(out/'observed_coordinates.csv',index=False)
    target_source=root/'source/multi-omics/downstream/cite_pcfm_deep_story/biology_landscape_discovery_tables/fig01_generated_protein_umap_target_time.csv'
    target=pd.read_csv(target_source)
    run_names={
        'ProteinBridge':'20260526T174204321821Z__final__cite_paired_coord_fm__campaign_20260-trial_20260526-gse305370_ci',
        'CITE-PCFM':'20260526T174204321821Z__final__cite_paired_coord_fm__campaign_20260-trial_20260526-gse305370_ci',
        'CRUFM':'20260525T220649168798Z__pilot__crufm__campaign_20260-stage2_c-baseline-crufm-gse305370_ci',
        'balanced OT-CFM':'20260525T215826526439Z__pilot__balanced_ot_cfm__campaign_20260-stage2_c-baseline-balanced_ot_-gse305370_ci'}
    for method,group in target.groupby('method'):
        with np.load(args.workspace/'multi-omics/training_runs'/run_names[method]/'artifacts/evaluation_trajectory.npz',allow_pickle=False) as z:
            ix=int(np.argmin(np.abs(z['time_points']-4)))
            state=z[f'points_{ix:06d}'][group.particle.to_numpy(int),50:80]
        target.loc[group.index,['umap1','umap2']]=reducer.transform(scale.transform(state))
    target.to_csv(out/'generated_target_coordinates.csv',index=False)
    coord=root/'source/multi-omics/downstream/cite_pcfm_deep_story/biology_landscape_discovery_protein_rollout_trajectory/cite_pcfm_rollout_trajectory_protein_umap_coordinates.csv'
    df=pd.read_csv(coord)
    ids=np.sort(df.particle_id.unique()).astype(int)
    first=df.sort_values('rollout_time').drop_duplicates('particle_id').set_index('particle_id')
    basins=first.loc[ids,'terminal_basin'].astype(str).to_numpy(dtype='U')
    run=args.workspace/'multi-omics/training_runs/20260526T174204321821Z__final__cite_paired_coord_fm__campaign_20260-trial_20260526-gse305370_ci/artifacts/evaluation_trajectory.npz'
    with np.load(run,allow_pickle=False) as z:
        times=z['time_points']
        pts=np.concatenate([z[f'points_{i:06d}'][ids,50:80] for i in range(len(times))])
    print('Projecting saved selected protein trajectories',flush=True)
    xy=reducer.transform(scale.transform(pts)).reshape(len(times),len(ids),2)
    np.savez_compressed(out/'selected_paths.npz',xy=xy,times=times,particle_ids=ids,terminal_basin=basins)
    (out/'PROVENANCE.json').write_text(json.dumps({'source_data':str(args.data),'source_data_sha256':hashlib.sha256(args.data.read_bytes()).hexdigest(),'trajectory':str(run),'trajectory_sha256':hashlib.sha256(run.read_bytes()).hexdigest(),'seed':20260528,'umap_version':umap.__version__,'note':"Display coordinates recalculated once with the original UMAP settings. Observed cells, all three methods' target-time states and selected trajectories use the same fitted protein projection. The resulting coordinates are bundled for deterministic plotting. Dynamical models were not refitted. Historical coordinates remain in source/ for comparison."},indent=2)+'\n')
    print(out,flush=True)

if __name__=='__main__':main()
