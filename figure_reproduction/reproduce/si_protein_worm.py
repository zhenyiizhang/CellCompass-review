"""Plain-style redraws of the ProteinBridge and LCFBM supplementary panels
(SI 13 summary/continued pages, SI 16, SI 17)."""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .style import apply_style,clean,panel_label,legend,emit,BLACK,BLUE,RED,GRAY,LIGHT

METHODS=['ProteinBridge','CRUFM','balanced OT-CFM']
METHOD_COLORS={'ProteinBridge':BLUE,'CRUFM':RED,'balanced OT-CFM':GRAY}
SOURCE_NAME={'ProteinBridge':'CITE-PCFM','CRUFM':'CRUFM','balanced OT-CFM':'balanced OT-CFM'}
METHOD_LABEL={'ProteinBridge':'ProteinBridge','CRUFM':'CRUFM','balanced OT-CFM':'Balanced OT-CFM'}

def method_handles():
    return [Line2D([],[],color=METHOD_COLORS[m],marker='o',ms=3.2,lw=1.1,label=METHOD_LABEL[m]) for m in METHODS]

# ------------------------------------------------------------------- SI 13

def protein_quantitative(means,per,tables,stem):
    """One figure: mean and per-time protein energy distance plus covariance,
    nearest-target, phenotype-composition and variance diagnostics."""
    apply_style()
    per=pd.DataFrame(per)
    pt=tables['per_time'];pheno=tables['phenotype'];nearest=tables['nearest'];variance=tables['variance']
    fig,axes=plt.subplots(2,3,figsize=(10.2,6.0),gridspec_kw={'wspace':.45,'hspace':.5})
    ax=axes[0,0]
    y=np.arange(len(METHODS))
    ax.barh(y,[means[m] for m in METHODS],height=.55,color=[METHOD_COLORS[m] for m in METHODS])
    ax.set_yticks(y);ax.set_yticklabels([METHOD_LABEL[m] for m in METHODS]);ax.invert_yaxis()
    ax.set_xlabel('Mean protein energy distance, t1–t4');ax.set_xlim(0,max(means.values())*1.12)
    ax.set_title('Mean distribution error',loc='left');clean(ax);panel_label(ax,'a',x=-.62)
    ax=axes[0,1]
    for m in METHODS:
        d=per[per.method.eq(m)].sort_values('target_time')
        ax.plot(d.target_time,d.protein_energy,marker='o',ms=3.2,lw=1.1,color=METHOD_COLORS[m])
    ax.set_yscale('log');ax.set_xticks([1,2,3,4]);ax.set_xlabel('Target time');ax.set_ylabel('Protein energy distance')
    ax.set_title('Distribution error by time',loc='left');clean(ax);panel_label(ax,'b',x=-.28)
    ax=axes[0,2]
    for m in METHODS:
        d=pt[pt.method.eq(SOURCE_NAME[m])].sort_values('time')
        ax.plot(d.time,d.rna_protein_cov_error,marker='o',ms=3.2,lw=1.1,color=METHOD_COLORS[m])
    ax.set_xlabel('Target time');ax.set_ylabel('Relative covariance error');ax.set_xticks([0,1,2,3,4])
    ax.set_title('RNA–protein covariance',loc='left');clean(ax);panel_label(ax,'c',x=-.28)
    ax=axes[1,0]
    for m in METHODS:
        v=np.sort(nearest[nearest.method.eq(SOURCE_NAME[m])&nearest.time.gt(0)].nearest_distance.dropna().to_numpy())
        ax.plot(v,np.arange(1,len(v)+1)/max(1,len(v)),lw=1.1,color=METHOD_COLORS[m])
    ax.set_xlabel('Distance to nearest observed protein state');ax.set_ylabel('Cumulative fraction');ax.set_ylim(0,1.02)
    ax.set_title('Proximity to observed states',loc='left');clean(ax);panel_label(ax,'d',x=-.28)
    ax=axes[1,1]
    phenotypes=[p for p in ['HSC','MasP','EryP','MoP','MkP','NeuP'] if p in set(pheno.phenotype)]
    d=pheno[pheno.time.gt(0)].groupby(['method','phenotype'],as_index=False).fraction_abs_error.mean()
    y=np.arange(len(phenotypes));h=.26
    for i,m in enumerate(METHODS):
        sub=d[d.method.eq(SOURCE_NAME[m])].set_index('phenotype').reindex(phenotypes)
        ax.barh(y+(i-1)*h,sub.fraction_abs_error,height=h,color=METHOD_COLORS[m])
    ax.set_yticks(y);ax.set_yticklabels(phenotypes);ax.invert_yaxis()
    ax.set_xlabel('Mean absolute phenotype-fraction error')
    ax.set_title('Phenotype composition error',loc='left');clean(ax);panel_label(ax,'e',x=-.28)
    ax=axes[1,2]
    for m in METHODS:
        d=variance[variance.method.eq(SOURCE_NAME[m])].sort_values('time')
        ax.plot(d.time,d.protein_variance_relative_l1,marker='o',ms=3.2,lw=1.1,color=METHOD_COLORS[m])
    ax.set_xlabel('Target time');ax.set_ylabel('Relative variance error');ax.set_xticks([0,1,2,3,4])
    ax.set_title('Protein-state variance',loc='left');clean(ax);panel_label(ax,'f',x=-.28)
    fig.legend(handles=method_handles(),loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,-.005),columnspacing=2.0)
    fig.subplots_adjust(left=.12,right=.985,top=.93,bottom=.12)
    emit(fig,stem)

# ------------------------------------------------------------------- SI 16/17

def density_contour(ax,pts,color,xlim,ylim,*,lw=.9,alpha=.9,fill=False):
    from scipy.stats import gaussian_kde
    pts=np.asarray(pts,dtype=float);pts=pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0]<8:return
    kde=gaussian_kde(pts.T)
    xx,yy=np.meshgrid(np.linspace(*xlim,160),np.linspace(*ylim,160))
    zz=kde(np.vstack([xx.ravel(),yy.ravel()])).reshape(xx.shape)
    vals=zz[np.isfinite(zz)];vals=vals[vals>0]
    if vals.size<10:return
    levels=np.unique(np.quantile(vals,[.72,.92]))
    if levels.size<2:return
    if fill:ax.contourf(xx,yy,zz,levels=np.r_[levels,vals.max()],colors=[color],alpha=.10,zorder=1)
    ax.contour(xx,yy,zz,levels=levels,colors=[color],linewidths=lw,alpha=alpha,zorder=5)

def worm_windows(adata,trajectory):
    z,keys=trajectory
    xy=np.asarray(adata.obsm['X_pca'],dtype=float)[:,:2]
    obs_time=adata.obs.time_point_processed.to_numpy();labels=adata.obs.time_bin_label.astype(str).to_numpy()
    idx=np.asarray(z['observed_time_indices'],dtype=int);tp=np.asarray(z['observed_time_points'],dtype=float)
    valid=[(i,t) for i,t in zip(idx,tp) if 0<=i<len(keys)]
    windows=[]
    for i,t in valid:
        mask=obs_time==t
        gen=np.asarray(z[keys[i]],dtype=float)[:,:2]
        windows.append({'label':str(labels[mask][0]),'observed':xy[mask],'generated':gen,'time':t})
    return xy,windows

def lcfbm_distribution_match(adata,trajectory,stem):
    """Observed and generated populations in the first two PCs for six windows."""
    apply_style()
    xy,windows=worm_windows(adata,trajectory)
    gen_all=np.vstack([w['generated'] for w in windows]);allxy=np.vstack([xy,gen_all])
    pad=.07*(allxy.max(axis=0)-allxy.min(axis=0)+1e-6)
    xlim=(allxy[:,0].min()-pad[0],allxy[:,0].max()+pad[0]);ylim=(allxy[:,1].min()-pad[1],allxy[:,1].max()+pad[1])
    fig,axes=plt.subplots(2,3,figsize=(9.6,6.0),sharex=True,sharey=True)
    for k,(ax,w) in enumerate(zip(axes.flat,windows)):
        ax.scatter(xy[:,0],xy[:,1],s=2,c='#E6E6E6',lw=0,alpha=.6,rasterized=True,zorder=0)
        density_contour(ax,w['observed'],BLACK,xlim,ylim,lw=.8,alpha=.7,fill=True)
        density_contour(ax,w['generated'],BLUE,xlim,ylim,lw=.9)
        ax.scatter(w['observed'][:,0],w['observed'][:,1],s=5,c=BLACK,lw=0,alpha=.3,rasterized=True,zorder=2)
        ax.scatter(w['generated'][:,0],w['generated'][:,1],s=8,facecolors='none',edgecolors=BLUE,linewidths=.45,alpha=.65,rasterized=True,zorder=6)
        oc=np.median(w['observed'],axis=0);gc=np.median(w['generated'],axis=0)
        ax.scatter(*oc,s=36,c=BLACK,edgecolor='white',linewidth=.7,zorder=8)
        ax.scatter(*gc,s=36,c=BLUE,edgecolor='white',linewidth=.7,zorder=8)
        ax.set_title(f'{w["label"]} min',loc='left')
        ax.set_xlim(*xlim);ax.set_ylim(*ylim);clean(ax)
        panel_label(ax,chr(ord('a')+k),x=-.14)
    for ax in axes[-1]:ax.set_xlabel('PC1')
    for ax in axes[:,0]:ax.set_ylabel('PC2')
    handles=[Line2D([],[],marker='o',ls='',color=BLACK,ms=4,label='Observed cells'),
             Line2D([],[],marker='o',ls='',markerfacecolor='white',markeredgecolor=BLUE,ms=4,label='LCFBM generated cells'),
             Line2D([],[],color=BLACK,lw=.8,label='Observed density'),Line2D([],[],color=BLUE,lw=.9,label='Generated density'),
             Line2D([],[],marker='o',ls='',color=BLACK,markeredgecolor='white',ms=6,label='Observed median'),
             Line2D([],[],marker='o',ls='',color=BLUE,markeredgecolor='white',ms=6,label='Generated median')]
    fig.legend(handles=handles,loc='lower center',ncol=6,frameon=False,bbox_to_anchor=(.5,-.005),columnspacing=1.4)
    fig.subplots_adjust(left=.07,right=.99,top=.93,bottom=.13,wspace=.15,hspace=.35)
    emit(fig,stem)

def lcfbm_w1_by_window(adata,trajectory,w1_scores,stem):
    """Wasserstein-1 distance between generated and observed populations per prediction window."""
    apply_style()
    _,windows=worm_windows(adata,trajectory)
    labels=[w['label'] for w in windows][1:1+len(w1_scores)]
    fig,ax=plt.subplots(figsize=(5.6,3.6))
    x=np.arange(len(w1_scores))
    ax.bar(x,w1_scores,width=.55,color=BLUE)
    ax.set_xticks(x);ax.set_xticklabels([f'{l} min' for l in labels],rotation=30,ha='right')
    ax.set_ylabel('Wasserstein-1 distance');ax.set_xlabel('Prediction window')
    ax.set_ylim(0,max(w1_scores)*1.1);clean(ax)
    fig.subplots_adjust(left=.15,right=.98,top=.96,bottom=.3)
    emit(fig,stem)

# ------------------------------------------------------------------- SI 12

PHENO_ORDER=['HSC','MasP','NeuP','EryP','MkP','MoP','BP']
# Muted palette; hue assignment follows main Fig. 6 (HSC green, NeuP teal, MasP red, EryP orange, MkP purple, MoP brown).
PHENO_COLORS={'HSC':'#5E8F6E','MasP':'#B85C5C','NeuP':'#5F9EA8','EryP':'#D19A5B','MkP':'#8A7BAF','MoP':'#9C7B5E','BP':'#6B7FA3'}

def protein_method_comparison(obs,gen,stem):
    """Observed RNA/protein maps and terminal protein states generated by each method."""
    apply_style()
    target=obs[np.isclose(obs.time_point_processed,4)]
    fig,axes=plt.subplots(2,3,figsize=(9.6,6.2),gridspec_kw={'wspace':.12,'hspace':.3})
    axes=axes.ravel()
    xy=np.vstack([obs[['protein_x','protein_y']].to_numpy(),gen[['umap1','umap2']].to_numpy()])
    lim=np.quantile(xy,[.004,.996],axis=0);pad=.05*(lim[1]-lim[0]);lim+=np.array([-pad,pad])
    def cell_types(ax,frame,cols):
        pts=frame[cols].to_numpy()
        ax.scatter(pts[:,0],pts[:,1],s=3,c='#E6E6E6',lw=0,rasterized=True,zorder=0)
        for ct in PHENO_ORDER:
            d=frame[frame.cell_type.eq(ct)]
            if len(d):ax.scatter(d[cols[0]],d[cols[1]],s=2.5,c=PHENO_COLORS[ct],lw=0,alpha=.7,rasterized=True,zorder=2)
    cell_types(axes[0],obs,['rna_x','rna_y']);axes[0].set_title('Observed RNA states',loc='left')
    cell_types(axes[1],obs,['protein_x','protein_y']);axes[1].set_title('Observed surface-protein states',loc='left')
    cell_types(axes[2],target,['protein_x','protein_y']);axes[2].set_title('Observed terminal states, t4',loc='left')
    for ax,m in zip(axes[3:],METHODS):
        ax.scatter(target.protein_x,target.protein_y,s=3.5,c='#BDBDBD',lw=0,alpha=.7,rasterized=True,zorder=0)
        sub=gen[gen.method.replace({'CITE-PCFM':'ProteinBridge'}).eq(m)];assert len(sub)>0,m
        ax.scatter(sub.umap1,sub.umap2,s=3.5,c=BLACK,lw=0,alpha=.65,rasterized=True,zorder=2)
        ax.set_title(f'{METHOD_LABEL[m]} generated, t4',loc='left')
    for i,ax in enumerate(axes):
        ax.set_xticks([]);ax.set_yticks([])
        for sp in ax.spines.values():sp.set_visible(False)
        if i:ax.set_xlim(*lim[:,0]);ax.set_ylim(*lim[:,1])
        ax.set_xlabel('RNA UMAP 1' if i==0 else 'Protein UMAP 1');ax.set_ylabel('RNA UMAP 2' if i==0 else 'Protein UMAP 2')
        panel_label(ax,chr(ord('a')+i),x=-.06,y=1.0)
    present=[c for c in PHENO_ORDER if c in set(obs.cell_type)]
    handles=[Line2D([],[],marker='o',ls='',color=PHENO_COLORS[c],ms=5,label=c) for c in present]
    handles+=[Line2D([],[],marker='o',ls='',color='#BDBDBD',ms=5,label='Observed terminal cells (d–f)'),Line2D([],[],marker='o',ls='',color=BLACK,ms=5,label='Generated cells (d–f)')]
    fig.legend(handles=handles,loc='lower center',ncol=len(handles),frameon=False,bbox_to_anchor=(.5,-.005),columnspacing=1.3)
    fig.subplots_adjust(left=.05,right=.99,top=.93,bottom=.1)
    emit(fig,stem)

# ------------------------------------------------------------------- SI 14

PROGRAM_COLORS={'Myeloid':'#D62728','Erythroid':'#F28E2B','Mk':'#7B61B3','HSC':'#2F855A'}
PROGRAM_LABEL={'Myeloid':'Myeloid','Erythroid':'Erythroid','Mk':'Megakaryocyte','HSC':'HSC'}

def protein_readout_validation(r,corr,static,trend,stem):
    """Observed program–ADT correlations versus learned readout derivatives, plus binned trends."""
    apply_style()
    markers=sorted(corr.protein_marker.unique())
    all_cells=corr[corr.scope.eq('all_cells')][['program','protein_marker','spearman_rho']]
    merged=all_cells.merge(static[['program','protein_marker','mean_dADT_dprogram']],on=['program','protein_marker'],how='inner')
    merged=merged[merged.protein_marker.map(r.usable_protein_marker)].copy()
    panel=set(r.marker_panel(markers))
    merged['score']=merged.spearman_rho.abs()+merged.mean_dADT_dprogram.abs()/(merged.mean_dADT_dprogram.abs().max()+1e-8)
    selected=pd.concat([merged[merged.protein_marker.isin(panel)],merged.sort_values('score',ascending=False).head(50)],ignore_index=True).drop_duplicates(['program','protein_marker'])
    fig=plt.figure(figsize=(12.4,6.2))
    gs=fig.add_gridspec(2,5,width_ratios=[1.35,1,.42,1,.42],wspace=.35,hspace=.5)
    ax=fig.add_subplot(gs[:,0])
    ax.axhline(0,color=LIGHT,lw=.7,zorder=0);ax.axvline(0,color=LIGHT,lw=.7,zorder=0)
    for program in r.RNA_PROGRAMS:
        sub=selected[selected.program.eq(program)]
        ax.scatter(sub.spearman_rho,sub.mean_dADT_dprogram,s=18,color=PROGRAM_COLORS[program],alpha=.85,lw=0,label=PROGRAM_LABEL[program])
    ax.set_xlabel('Observed RNA-program–ADT Spearman ρ');ax.set_ylabel('Learned readout derivative, dADT/dprogram')
    ax.set_title('Observed association versus learned readout',loc='left')
    clean(ax);legend(ax,loc='lower right',title=None);panel_label(ax,'a',x=-.2)
    order=['HSC','Erythroid','Mk','Myeloid'];slots=[gs[0,1],gs[0,3],gs[1,1],gs[1,3]]
    marker_palette=['#1F77B4','#D62728','#2CA02C','#9467BD','#FF7F0E','#8C564B','#000000']
    for k,(program,slot) in enumerate(zip(order,slots)):
        ax=fig.add_subplot(slot)
        sub=trend[trend.program.eq(program)]
        ms=[m for m in r.OBSERVED_TREND_MARKERS[program] if m in set(sub.protein_marker)]
        for i,marker in enumerate(ms):
            d=sub[sub.protein_marker.eq(marker)].sort_values('score_bin')
            c=marker_palette[i%len(marker_palette)]
            ax.errorbar(d.score_bin_percentile_mid,d.mean_zADT,yerr=d.sem_zADT,marker='o',ms=2.6,lw=1.0,elinewidth=.6,capsize=1.5,color=c,label=marker)
        ax.axhline(0,color=LIGHT,lw=.7,zorder=0)
        ax.set_title(f'{PROGRAM_LABEL[program]} program',loc='left')
        ax.set_xlabel('RNA program score percentile');ax.set_ylabel('Observed ADT z-score')
        ax.set_xticks([0,25,50,75,100]);clean(ax)
        legend(ax,loc='upper left',bbox_to_anchor=(1.02,1.02),labelspacing=.25,handlelength=1.2)
        panel_label(ax,chr(ord('b')+k),x=-.3)
    fig.subplots_adjust(left=.06,right=.99,top=.93,bottom=.1)
    emit(fig,stem)
