"""Export the numerical grid passed by Dynamo 1.5.3 to Matplotlib streamplot.

No rendered image is sampled. The clone-only rebuild consumes these original
plotting arrays with Matplotlib, without installing Dynamo or rerunning projection.
"""
from pathlib import Path
import argparse
import json
import os
os.environ.setdefault('NUMBA_DISABLE_JIT','1')
os.environ.setdefault('MPLBACKEND','Agg')
import numpy as np
import anndata as ad
import dynamo as dyn
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True)
    args=parser.parse_args()
    out=Path(__file__).resolve().parents[1]/'plot_data/pulse'
    a=ad.read_h5ad(args.input)
    print('Projecting final latent velocities with Dynamo 1.5.3',flush=True)
    a=dyn.tl.cell_velocities(a,ekey='X',vkey='velocity',X=a.obsm['X_latent'],V=a.obsm['velocity_latent'],X_embedding=a.obsm['X_umap'],basis='umap',method='pearson',add_transition_key='velocity_cell_pearson_transition_matrix',add_velocity_key='velocity_cell_umap',enforce=True)
    original=Axes.streamplot
    def capture(self,x,y,u,v,*args,**kwargs):
        assert not args
        arrays={k:np.asarray(np.ma.filled(vv,np.nan)) for k,vv in [('x',x),('y',y),('u',u),('v',v)]}
        options={'array_options':[]}
        for k,val in kwargs.items():
            if isinstance(val,np.ndarray):
                arrays[k]=np.ma.filled(val,np.nan);options['array_options'].append(k)
            elif hasattr(val,'name'):options[k]=val.name
            elif val is None or isinstance(val,(str,int,float,bool,list,tuple)):options[k]=val
            else:raise TypeError(f'Unserializable stream option {k}: {type(val)}')
        options.update(color='#1F2937',linewidth=.62,density=.66,arrowsize=.48)
        np.savez_compressed(out/'stream_grid.npz',**arrays)
        (out/'stream_options.json').write_text(json.dumps(options,indent=2)+'\n')
        return original(self,x,y,u,v,**kwargs)
    Axes.streamplot=capture
    dyn.pl.streamline_plot(a,basis='umap',vector='velocity_cell',color='time_point_processed',pointsize=.070,show_legend='none',show_arrowed_spines=False,method='gaussian',density=.66,linewidth=.62,streamline_alpha=.78,arrowsize=.48,save_show_or_return='return',s_kwargs_dict={'alpha':.22,'rasterized':True})
    assert (out/'stream_grid.npz').exists()
    print(out/'stream_grid.npz',flush=True)

if __name__=='__main__':main()
