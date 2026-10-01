"""Plain-style redraws of the supplementary panels (EMT and PulseBridge).

Every function consumes the same frozen tables as the original scripts and
changes presentation only: Arial, black text, small stable palette, no
in-figure subtitles or performance statements, frameless legends.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .style import apply_style,clean,panel_label,legend,embedding_axis,emit,BLACK,BLUE,RED,GRAY,LIGHT,TAB

# ---------------------------------------------------------------- EMT (SI 2-4)

ROUTE_ORDER=['mesenchymal conversion','partial/hybrid EMT','epithelial-retaining response','mixed/weak EMT route']
ROUTE_COLORS={'mesenchymal conversion':TAB['red'],'partial/hybrid EMT':TAB['orange'],
              'epithelial-retaining response':TAB['gray'],'mixed/weak EMT route':TAB['purple']}
ROUTE_SHORT={'mesenchymal conversion':'Mesenchymal conversion','partial/hybrid EMT':'Partial/hybrid EMT',
             'epithelial-retaining response':'Epithelial-retaining','mixed/weak EMT route':'Mixed/weak EMT'}
STIMULUS_MARKERS={'EGF':'o','TGFB1':'^','TNF':'s'}
STIMULUS_LABEL={'EGF':'EGF','TGFB1':'TGF-β1','TNF':'TNF'}

def pretty(condition):
    return condition.replace('_TGFB1','+TGF-β1').replace('_','+')

def emt_timecourses(m,snap,stem):
    """SI 2: observed versus generated EMT-core change from 0d, 12 conditions."""
    apply_style()
    fig,axes=plt.subplots(3,4,figsize=(9.6,6.0),sharex=True)
    x=np.arange(len(m.TIME_LABELS))
    conditions=[f'{c}_{s}' for c in m.CELL_LINES for s in m.STIMULI]
    for ax,cond in zip(axes.flat,conditions):
        sub=snap.loc[snap.condition.eq(cond)].sort_values('time_index')
        obs=sub.observed_emt_core_score.to_numpy();gen=sub.generated_emt_core_score.to_numpy()
        obs=obs-obs[0];gen=gen-gen[0]
        ax.plot(x,obs,color=BLACK,marker='o',ms=3.2,lw=1.1,label='Observed cells')
        ax.plot(x,gen,color=BLUE,marker='o',ms=3.2,lw=1.1,ls=(0,(4,2)),label='CellCompass generated')
        low=min(obs.min(),gen.min());high=max(obs.max(),gen.max());pad=max(.08,.15*(high-low))
        ax.set_ylim(low-pad,high+pad);ax.set_xlim(-.3,4.3)
        ax.set_title(pretty(cond),pad=3)
        ax.set_xticks(x);ax.set_xticklabels(m.TIME_LABELS)
        clean(ax)
    for ax in axes[:,0]:ax.set_ylabel('Δ EMT-core score from 0d')
    for ax in axes[-1]:ax.set_xlabel('Time')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,-.01),columnspacing=2.0)
    fig.subplots_adjust(left=.075,right=.99,top=.95,bottom=.13,wspace=.36,hspace=.5)
    emit(fig,stem)

def emt_endpoint_scatter(m,route,stem):
    """SI 3: generated versus observed endpoint program shifts."""
    apply_style()
    fig,axes=plt.subplots(1,2,figsize=(8.0,4.6))
    pairs=[('observed_delta_emt_core_score','generated_delta_emt_core_score','EMT-core endpoint shift'),
           ('observed_delta_mesenchymal_score','generated_delta_mesenchymal_score','Mesenchymal endpoint shift')]
    for ax,(oc,gc,title) in zip(axes,pairs):
        xs=route[oc].to_numpy();ys=route[gc].to_numpy()
        lo=min(xs.min(),ys.min())-.1;hi=max(xs.max(),ys.max())+.1
        ax.plot([lo,hi],[lo,hi],color=LIGHT,lw=.8,ls=(0,(4,2)),zorder=0)
        for _,row in route.iterrows():
            c=ROUTE_COLORS[row.display_route]
            ax.scatter(row[oc],row[gc],s=34,facecolor=c,edgecolor=c,linewidth=.5,zorder=3)
            ax.annotate(str(m.CONDITION_IDS[row.condition]),(row[oc],row[gc]),xytext=(4,3),textcoords='offset points',fontsize=8,color=BLACK)
        ax.set_xlim(lo,hi);ax.set_ylim(lo,hi);ax.set_aspect('equal',adjustable='box')
        ax.set_title(title);ax.set_xlabel('Observed endpoint change');ax.set_ylabel('Generated endpoint change')
        clean(ax)
    present=[r for r in ROUTE_ORDER if r in set(route.display_route)]
    handles=[Line2D([],[],marker='o',ls='',color=ROUTE_COLORS[r],ms=5,label=ROUTE_SHORT[r]) for r in present]
    fig.legend(handles=handles,loc='lower center',ncol=len(handles),frameon=False,bbox_to_anchor=(.5,.13),columnspacing=1.6)
    items=[f'{m.CONDITION_IDS[c]} {pretty(c)}' for c in m.CONDITION_ORDER]
    key='\n'.join('     '.join(items[i:i+6]) for i in range(0,len(items),6))
    fig.text(.5,.02,key,ha='center',va='bottom',fontsize=8,color=BLACK,linespacing=1.5)
    fig.subplots_adjust(left=.08,right=.985,top=.93,bottom=.32,wspace=.3)
    emit(fig,stem)

def emt_route_phase(summary,stem):
    """SI 4 left: generated EMT-core shift versus observed cell-count ratio."""
    apply_style()
    fig,ax=plt.subplots(figsize=(4.6,4.0))
    for _,row in summary.iterrows():
        c=ROUTE_COLORS[row.display_route];marker=STIMULUS_MARKERS[row.stimulus]
        low=bool(row.observed_endpoint_n<150)
        ax.scatter(row.generated_delta_emt_core_score,row.observed_endpoint_baseline_count_ratio,s=40,marker=marker,
                   facecolor='white' if low else c,edgecolor=c,linewidth=1.0,zorder=4)
    ax.set_xlabel('Generated endpoint Δ EMT-core score')
    ax.set_ylabel('Observed 7d / 0d cell-count ratio')
    ax.set_xlim(-.15,1.2);ax.set_ylim(0,2.9)
    clean(ax)
    route_handles=[Line2D([],[],marker='o',ls='',color=ROUTE_COLORS[r],ms=5,label=ROUTE_SHORT[r]) for r in ROUTE_ORDER if r in set(summary.display_route)]
    stim_handles=[Line2D([],[],marker=mk,ls='',markerfacecolor='white',markeredgecolor=BLACK,markeredgewidth=.9,ms=5,label=STIMULUS_LABEL[s]) for s,mk in STIMULUS_MARKERS.items()]
    open_handle=Line2D([],[],marker='o',ls='',markerfacecolor='white',markeredgecolor=GRAY,markeredgewidth=.9,ms=5,label='Endpoint n < 150')
    legend(ax,handles=route_handles+stim_handles+[open_handle],loc='upper center',bbox_to_anchor=(.5,-.16),ncol=2,labelspacing=.35,handletextpad=.4,columnspacing=1.6)
    fig.subplots_adjust(left=.14,right=.98,top=.97,bottom=.13)
    emit(fig,stem)

def emt_program_prevalence(shared,labels,stem):
    """SI 4 right: share of conditions with an observed endpoint increase per program."""
    apply_style()
    order=['intermediate_partial_emt_score','hybrid_score','early_inflammatory_tnf_score','late_ecm_mesenchymal_score',
           'emt_core_score','mesenchymal_score','growth_cell_cycle_score','epithelial_score']
    df=shared.set_index('program').loc[order].reset_index()
    fig,ax=plt.subplots(figsize=(4.8,4.0))
    y=np.arange(len(df))[::-1]
    colors=[BLUE if d=='shared_increase' else GRAY for d in df.shared_direction]
    ax.barh(y,df.positive_fraction*100,color=colors,height=.55)
    for yi,(_,row) in zip(y,df.iterrows()):
        ax.text(row.positive_fraction*100+1.5,yi,pretty(row.strongest_increase_condition),va='center',ha='left',fontsize=8,color=BLACK)
    ax.set_yticks(y);ax.set_yticklabels([labels[p] for p in df.program])
    ax.set_xlim(0,128);ax.set_xticks([0,25,50,75,100])
    ax.set_xlabel('Conditions with increased endpoint score (%)')
    clean(ax)
    handles=[Patch(facecolor=BLUE,label='Shared increase'),Patch(facecolor=GRAY,label='Context-specific')]
    legend(ax,handles=handles,loc='upper center',bbox_to_anchor=(.5,-.16),ncol=2,columnspacing=1.6)
    fig.subplots_adjust(left=.36,right=.98,top=.97,bottom=.2)
    emit(fig,stem)

# --------------------------------------------------------- PulseBridge (SI 5-6)

def pulse_overview(n,stem):
    """SI 5 as one figure: match, same-sign fractions, gene contributions and MT-RNR2 maps."""
    apply_style()
    fig=plt.figure(figsize=(10.4,6.4))
    gs=fig.add_gridspec(2,2,width_ratios=[1,1.35],height_ratios=[1,1],wspace=.35,hspace=.55)
    # a
    ax=fig.add_subplot(gs[0,0])
    df=n.read_table('story_observed_generated_match_by_day.csv')
    lo=min(df.observed_axis_mean.min(),df.generated_axis_mean.min())-.16
    hi=max(df.observed_axis_mean.max(),df.generated_axis_mean.max())+.16
    ax.plot([lo,hi],[lo,hi],color=LIGHT,lw=.8,ls=(0,(4,2)),zorder=0)
    ax.scatter(df.observed_axis_mean,df.generated_axis_mean,s=30,color=BLUE,zorder=3)
    for _,row in df.iterrows():
        ax.annotate(f'day {int(row.time)}',(row.observed_axis_mean,row.generated_axis_mean),xytext=(5,-3),textcoords='offset points',fontsize=8)
    ax.set_xlim(lo,hi);ax.set_ylim(lo,hi);ax.set_aspect('equal',adjustable='box')
    ax.set_xlabel('Observed response mean');ax.set_ylabel('Generated response mean')
    clean(ax);panel_label(ax,'a',x=-.28)
    # b
    ax=fig.add_subplot(gs[0,1])
    real=n.read_table('pcie_gene_velocity_umap_values.csv').groupby('gene').agg(real=('same_sign','mean')).reset_index()
    rand=n.read_table('pcie_random_velocity_umap_summary.csv').rename(columns={'same_sign_fraction':'random'})
    genes=['MT-RNR2','RPS2','MT-CO1','MALAT1','SAT1','LGALS1']
    comp=real.merge(rand[['gene','random']],on='gene').set_index('gene').loc[genes].reset_index()
    y=np.arange(len(comp));h=.36
    ax.barh(y-h/2,comp.real,height=h,color=BLUE,label='PulseBridge velocity')
    ax.barh(y+h/2,comp.random,height=h,color=GRAY,label='Random velocity')
    ax.axvline(.5,color=LIGHT,lw=.8,ls=(0,(4,2)),zorder=0)
    ax.set_yticks(y);ax.set_yticklabels(comp.gene);ax.invert_yaxis()
    ax.set_xlim(0,1.0);ax.set_xlabel('Same-sign fraction')
    clean(ax);legend(ax,loc='upper center',bbox_to_anchor=(.5,-.22),ncol=2,columnspacing=2.0);panel_label(ax,'b',x=-.2)
    # c
    ax=fig.add_subplot(gs[1,0])
    genes=n.read_table('pcie_gene_space_top_genes.csv').head(10).iloc[::-1]
    colors=[BLUE if v>=0 else RED for v in genes.mean_unit_contribution]
    ax.barh(genes.gene,genes.mean_unit_contribution,color=colors,height=.6)
    ax.axvline(0,color=BLACK,lw=.6)
    ax.set_xlabel('Mean contribution to alignment cosine');ax.tick_params(axis='y',length=0)
    clean(ax);panel_label(ax,'c',x=-.28)
    # d
    sub_gs=gs[1,1].subgridspec(1,4,wspace=.12,width_ratios=[1,1,1,.08])
    df=n.read_table('pcie_gene_velocity_umap_values.csv');sub=df[df.gene.eq('MT-RNR2')]
    umap=n.ad.read_h5ad(n.UMAP_H5AD).obsm['X_umap']
    rng=np.random.default_rng(109);idx=np.arange(umap.shape[0])
    if len(idx)>7000:idx=rng.choice(idx,size=7000,replace=False)
    specs=[('pulsebridge_gene_velocity_scaled','PulseBridge velocity'),('metabolic_gene_velocity_scaled','Metabolic pulse'),('signed_agreement_scaled','Local agreement')]
    norm=TwoSlopeNorm(vcenter=0,vmin=-1,vmax=1);axes=[]
    for k,(col,title) in enumerate(specs):
        ax=fig.add_subplot(sub_gs[0,k]);axes.append(ax)
        ax.scatter(umap[idx,0],umap[idx,1],c='#E0E0E0',s=2,alpha=.5,linewidths=0,rasterized=True)
        sc=ax.scatter(sub.umap_1,sub.umap_2,c=sub[col],cmap='RdBu_r',norm=norm,s=6,alpha=.9,linewidths=0,rasterized=True)
        ax.set_title(title,loc='left',pad=4);embedding_axis(ax)
        ax.set_xlabel('UMAP 1')
        if k==0:ax.set_ylabel('UMAP 2')
    cax=fig.add_subplot(sub_gs[0,3]);pos=cax.get_position()
    cax.set_position([pos.x0,pos.y0+pos.height*.2,pos.width,pos.height*.6])
    cb=fig.colorbar(sc,cax=cax)
    cb.set_ticks([-1,0,1]);cb.set_label('Scaled value')
    cb.outline.set_linewidth(.5);cb.ax.tick_params(width=.5,length=2)
    panel_label(axes[0],'d',x=-.2)
    fig.subplots_adjust(left=.1,right=.97,top=.95,bottom=.09)
    pos=cax.get_position();cax.set_position([pos.x0,pos.y0+pos.height*.2,pos.width,pos.height*.6])
    emit(fig,stem)

def pulse_perturbation_controls(s,data,stem):
    """SI 6: initial versus terminal program shifts and random-shift controls."""
    apply_style()
    it=data['perturb_it'];pr=data['perturb_program'];rand=data['random']
    fig,axes=plt.subplots(2,2,figsize=(10.0,8.6),gridspec_kw={'width_ratios':[1,1],'height_ratios':[1,1.45],'hspace':.6,'wspace':.75})
    below=dict(loc='upper center',bbox_to_anchor=(.5,-.2),ncol=2,columnspacing=1.8)
    # a: direct perturbation versus terminal rollout (paired bars).
    ax=axes[0,0]
    targets=[('reactivation','reactivation',-5.,'DKK1/GAS5 −5z'),('reactivation','reactivation',5.,'DKK1/GAS5 +5z'),
             ('immune_surface','immune_surface',-5.,'B2M/HLA −5z'),('immune_surface','immune_surface',5.,'B2M/HLA +5z'),
             ('proliferation_genes','proliferation_repression',-5.,'EREG/PCNA −5z'),('proliferation_genes','proliferation_repression',5.,'EREG/PCNA +5z'),
             ('adaptive_metabolic','adaptive_metabolic',-5.,'SAT1/LGALS1/MT −5z'),('adaptive_metabolic','adaptive_metabolic',5.,'SAT1/LGALS1/MT +5z')]
    rows=[]
    for module,program,strength,label in targets:
        r=it[it.perturbation_module.eq(module)&it.program.eq(program)&it.strength_z.eq(strength)].iloc[0]
        rows.append((label,float(r.initial_delta_from_baseline),float(r.terminal_delta_from_baseline)))
    y=np.arange(len(rows));h=.36
    ax.barh(y-h/2,[r[1] for r in rows],height=h,color=GRAY,label='Initial (perturbed starting cells)')
    ax.barh(y+h/2,[r[2] for r in rows],height=h,color=BLUE,label='Terminal rollout')
    ax.axvline(0,color=BLACK,lw=.6)
    ax.set_yticks(y);ax.set_yticklabels([r[0] for r in rows]);ax.invert_yaxis()
    ax.set_xlabel('Target-program Δ from baseline')
    clean(ax);legend(ax,**below);panel_label(ax,'a',x=-.5)
    # b: adaptive perturbation shifts the adaptive/anti-tumour balance.
    ax=axes[0,1]
    keep=pr[pr.perturbation_module.eq('adaptive_metabolic')&pr.strength_z.isin([-5.,5.])&pr.program.isin(s.ADAPTIVE_PROGRAM_ORDER)]
    y=np.arange(len(s.ADAPTIVE_PROGRAM_ORDER));h=.36
    for off,strength,label,color in [(-h/2,-5.,'Adaptive −5z',GRAY),(h/2,5.,'Adaptive +5z',BLUE)]:
        sub=keep[keep.strength_z.eq(strength)].set_index('program').reindex(s.ADAPTIVE_PROGRAM_ORDER)
        ax.barh(y+off,sub.delta_from_baseline,height=h,color=color,label=label)
    ax.axvline(0,color=BLACK,lw=.6)
    ax.set_yticks(y);ax.set_yticklabels([s.PROGRAM_LABELS[p].replace('\n',' ') for p in s.ADAPTIVE_PROGRAM_ORDER]);ax.invert_yaxis()
    ax.set_xlabel('Terminal program Δ from baseline')
    clean(ax);legend(ax,**below);panel_label(ax,'b',x=-.6)
    # c: shifts versus same-norm random controls, raw control points behind bars.
    specs=[('reactivation','reactivation',-5.,'DKK1/GAS5 −5z'),('reactivation','reactivation',5.,'DKK1/GAS5 +5z'),
           ('immune_surface','immune_surface',-5.,'B2M/HLA −5z'),('immune_surface','immune_surface',5.,'B2M/HLA +5z'),
           ('adaptive_metabolic','adaptive_metabolic',-5.,'SAT1/LGALS1/MT −5z'),('adaptive_metabolic','adaptive_metabolic',5.,'SAT1/LGALS1/MT +5z'),
           ('adaptive_metabolic','therapeutic_margin',-5.,'Balance after adaptive −5z'),('adaptive_metabolic','therapeutic_margin',5.,'Balance after adaptive +5z'),
           ('proliferation_genes','proliferation_repression',-5.,'EREG/PCNA −5z'),('proliferation_genes','proliferation_repression',5.,'EREG/PCNA +5z')]
    long=pd.read_csv(s.TAB/'pulsebridge_gene_perturbation_random_control_long.csv')
    gs=axes[1,0].get_gridspec();axes[1,0].remove();axes[1,1].remove()
    ax=fig.add_subplot(gs[1,:])
    rng=np.random.default_rng(0);y=np.arange(len(specs))
    for yi,(module,program,strength,label) in zip(y,specs):
        sub=long[long.perturbation_module.eq(module)&long.program.eq(program)&long.strength_z.eq(strength)]
        act=float(sub[sub.control_type.eq('actual')].terminal_delta_from_baseline.iloc[0])
        rnd=sub[sub.control_type.eq('random')].terminal_delta_from_baseline.to_numpy()
        ax.barh(yi-.2,act,height=.32,color=BLUE,zorder=2)
        ax.scatter(rnd,yi+.22+rng.uniform(-.09,.09,len(rnd)),s=8,color=GRAY,alpha=.8,lw=0,zorder=3)
    ax.axvline(0,color=BLACK,lw=.6)
    ax.set_yticks(y);ax.set_yticklabels([t[3] for t in specs]);ax.set_ylim(len(specs)-.5,-.6)
    ax.set_xlabel('Terminal Δ from baseline');clean(ax)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend(ax,handles=[Patch(color=BLUE,label='Gene-program shift'),Line2D([],[],marker='o',ls='',color=GRAY,ms=3.5,label='Same-norm random shift (24 draws)')],
           loc='upper center',bbox_to_anchor=(.5,-.14),ncol=2,columnspacing=1.8)
    panel_label(ax,'c',x=-.22)
    fig.subplots_adjust(left=.2,right=.985,top=.95,bottom=.1)
    emit(fig,stem)
