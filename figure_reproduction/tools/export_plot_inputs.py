"""Maintainer-only: freeze small numeric inputs from the original study workspace.

End users never run this script. No fitting, inference or selection is performed.
Only saved coordinates, annotations and evaluation trajectory arrays are copied.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import numpy as np
import anndata as ad

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--workspace',type=Path,required=True)
    parser.add_argument('--cellcompass-home',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    out=root/'plot_data';out.mkdir(exist_ok=True)
    provenance=[]
    def compact(name,path,keys):
        data=ad.read_h5ad(path)
        dest=out/name;dest.mkdir(exist_ok=True)
        data.obs.to_csv(dest/'obs.csv')
        if name=='worm':
            data.var.to_csv(dest/'var.csv')
            np.savez_compressed(dest/'varm.npz',PCs=np.asarray(data.varm['PCs']))
        arrays={k:np.asarray(data.obsm[k]) for k in keys}
        np.savez_compressed(dest/'arrays.npz',**arrays)
        provenance.append({'dataset':name,'source':str(path),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'obsm':keys,'cells':data.n_obs})
    compact('lineage',args.cellcompass_home/'algorithm_benchmarks/datasets/weinreb_rawrebuild_full_k10mindiff1/data.h5ad',['X_latent'])
    compact('pulse',args.workspace/'analysis/pulsebridge_story/outputs/downstream/dynamo_stream/pulsebridge_full_umap_velocity_for_dynamo.h5ad',['X_umap','velocity_umap','X_latent','velocity_latent'])
    worm=args.workspace/'lcfbm_packer_ea6cebc5_data_model_bundle'
    compact('worm',worm/'data/converted_input.h5ad',['X_latent','X_pca'])
    particle=args.workspace/'mouse_new/downstream/cellcompass_clone_case/clone_bmf_particle_spring_growth_mass_particles.csv'
    pd=None
    import pandas as pd
    pd.read_csv(particle).to_csv(out/'lineage/particles.csv.gz',index=False,compression='gzip')
    for p in (worm/'figures/initial_visualization').glob('*.csv'):
        shutil.copy2(p,out/'worm'/p.name)
    traj=worm/'trained_model/final_run/artifacts/evaluation_trajectory.npz'
    with np.load(traj,allow_pickle=False) as z:
        values={k:z[k] for k in z.files if k.startswith(('points_','time_','observed_time'))}
        np.savez_compressed(out/'worm/trajectory.npz',**values)
    shutil.copy2(worm/'trained_model/final_run/artifacts/metrics.json',out/'worm/metrics.json')
    provenance.append({'dataset':'worm_trajectory','source':str(traj),'source_sha256':hashlib.sha256(traj.read_bytes()).hexdigest()})
    (out/'PROVENANCE.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(provenance,indent=2))

if __name__=='__main__':main()
