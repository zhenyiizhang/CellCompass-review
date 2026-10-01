"""Plain-style redraws of the LineageBridge supplementary panels (SI 7-10)."""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .style import apply_style,clean,panel_label,legend,embedding_axis,emit,BLACK,BLUE,RED,GRAY,LIGHT

MODEL_COLORS={'BMF':BLUE,'VGFM':GRAY,'Dynamical OT':RED}
MODEL_LABELS={'BMF':'LineageBridge','VGFM':'VGFM','Dynamical OT':'Dynamical OT'}
MAIN_FATES=['Undifferentiated','Neutrophil','Monocyte','Baso','Mast','Meg','Erythroid','Eos']

def spring_frame(adata):
    obs=adata.obs[['time_point_processed','Cell type annotation','SPRING-x','SPRING-y','clone']].copy()
    obs['t']=obs.time_point_processed.astype(int)
    obs['cell_type']=obs['Cell type annotation'].astype(str)
    obs['clone_str']=obs.clone.astype(str).str.replace(r'\.0$','',regex=True)
    return obs.rename(columns={'SPRING-x':'spring_x','SPRING-y':'spring_y'})

def spring_limits(ax,obs):
    x=obs.spring_x.to_numpy(float);y=obs.spring_y.to_numpy(float)
    px=.035*(np.nanmax(x)-np.nanmin(x));py=.035*(np.nanmax(y)-np.nanmin(y))
    ax.set_xlim(np.nanmin(x)-px,np.nanmax(x)+px);ax.set_ylim(np.nanmin(y)-py,np.nanmax(y)+py)
    embedding_axis(ax)

def fate_background(ax,obs,palette,*,s=.7,alpha=.08,terminal_only=False):
    frame=obs[obs.t.eq(2)] if terminal_only else obs
    for fate,color in palette.items():
        d=frame[frame.cell_type.eq(fate)]
        if len(d):ax.scatter(d.spring_x,d.spring_y,s=s,c=color,alpha=alpha,linewidth=0,rasterized=True,zorder=0)

def fate_labels(ax,obs,palette):
    terminal=obs[obs.t.eq(2)]
    for fate,color in palette.items():
        d=terminal[terminal.cell_type.eq(fate)]
        if len(d)<20:continue
        ax.text(float(d.spring_x.median()),float(d.spring_y.median()),fate.replace('Undifferentiated','Undiff.'),
                color=BLACK,fontsize=8,ha='center',va='center',zorder=80)

def fate_legend(fates,palette):
    return [Line2D([],[],marker='o',ls='',color=palette[f],ms=5,label=f) for f in fates]

# ------------------------------------------------------------------ SI 7

def state_matched_divergence(s,adata,bmf,stem):
    """State-matched day-2 progenitors with different clonal futures."""
    apply_style()
    obs=spring_frame(adata);palette=s.FATE_PALETTE
    fates=[c.removeprefix('pred_') for c in bmf.columns if c.startswith('pred_')]
    valid=bmf[bmf.eligible.astype(bool)].reset_index(drop=True).copy()
    truth=s.normalize_rows(valid[[f'truth_{f}' for f in fates]].to_numpy(float))
    pred=s.normalize_rows(valid[[f'pred_{f}' for f in fates]].to_numpy(float))
    valid['future_fate_entropy']=s.entropy(truth)
    valid['truth_dominant_fate']=np.asarray(fates,dtype=object)[truth.argmax(axis=1)]
    latent=np.asarray(adata.obsm['X_latent'],dtype=np.float32)[valid.source_index.to_numpy(int),:10]
    center,neighbor_ids,_=s.select_state_matched_pocket(valid,latent,truth,pred,fates,k=18)
    pocket=valid.iloc[neighbor_ids].copy().reset_index(drop=True)
    pocket['latent_distance_from_center']=np.linalg.norm(latent[neighbor_ids]-latent[center],axis=1)
    idx=[]
    for fate in ['Neutrophil','Baso','Monocyte','Undifferentiated','Mast','Eos']:
        idx.extend(pocket[pocket.truth_dominant_fate.eq(fate)].sort_values('truth_reliability',ascending=False).head(2).index.tolist())
    if len(idx)<10:idx.extend(pocket.sort_values('truth_reliability',ascending=False).head(10).index.tolist())
    ex=pocket.loc[list(dict.fromkeys(idx))[:12]].copy()
    ex=ex.sort_values(['truth_dominant_fate','truth_reliability','latent_distance_from_center'],ascending=[True,False,True]).drop_duplicates('clone').head(10)
    order={f:i for i,f in enumerate(['Baso','Neutrophil','Monocyte','Undifferentiated','Mast','Meg','Erythroid','Eos'])}
    ex['_o']=ex.truth_dominant_fate.map(order).fillna(99)
    ex=ex.sort_values(['_o','truth_reliability'],ascending=[True,False])
    labels=[f'Clone {int(c)}' for c in ex.clone]
    truth_prob=s.compact_prob_table(ex,fates,'truth');pred_prob=s.compact_prob_table(ex,fates,'pred')

    fig=plt.figure(figsize=(9.0,6.0))
    gs=fig.add_gridspec(2,2,width_ratios=[1.0,1.15],height_ratios=[1,1],wspace=.4,hspace=.42)
    ax_a=fig.add_subplot(gs[0,0]);ax_b=fig.add_subplot(gs[1,0]);ax_c=fig.add_subplot(gs[0,1]);ax_d=fig.add_subplot(gs[1,1])
    # a: barcode-derived future-fate entropy on day-2 states.
    fate_background(ax_a,obs,palette,alpha=.03)
    sc=ax_a.scatter(valid.spring_x,valid.spring_y,c=valid.future_fate_entropy,cmap='Blues',vmin=0,vmax=.78,s=4,alpha=.9,linewidth=0,rasterized=True,zorder=10)
    ax_a.scatter(pocket.spring_x,pocket.spring_y,s=22,facecolors='none',edgecolors=BLACK,linewidth=.5,zorder=24)
    c_row=valid.iloc[center]
    fate_labels(ax_a,obs,palette);spring_limits(ax_a,obs)
    ax_a.set_title('Barcode future-fate uncertainty, day-2 cells',loc='left',pad=8)
    cb=fig.colorbar(sc,ax=ax_a,orientation='horizontal',fraction=.04,pad=.03,aspect=30)
    cb.set_ticks([0,.78]);cb.set_ticklabels(['Low','High']);cb.set_label('Future-fate entropy',labelpad=1)
    cb.outline.set_linewidth(.5);cb.ax.tick_params(length=2,width=.5)
    panel_label(ax_a,'a',x=-.14,y=1.0)
    # b: the state-matched pocket, colored by dominant observed future fate.
    fate_background(ax_b,obs[obs.t.eq(0)],palette,s=3.0,alpha=.3)
    present=[]
    for fate,color in palette.items():
        d=pocket[pocket.truth_dominant_fate.eq(fate)]
        if d.empty:continue
        present.append(fate)
        ax_b.scatter(d.spring_x,d.spring_y,s=30,c=color,edgecolor='white',linewidth=.5,zorder=20)
    cx,cy=float(c_row.spring_x),float(c_row.spring_y)
    margin=max(170.,float(np.max(np.abs(pocket[['spring_x','spring_y']].to_numpy(float)-np.array([cx,cy]))))+70.)
    ax_b.set_xlim(cx-margin,cx+margin);ax_b.set_ylim(cy-margin,cy+margin);embedding_axis(ax_b)
    ax_b.set_title('State-matched cells, dominant observed fate',loc='left',pad=8)
    legend(ax_b,handles=fate_legend(present,palette),loc='upper left',bbox_to_anchor=(1.0,1.0),labelspacing=.3)
    panel_label(ax_b,'b',x=-.14,y=1.0)
    # c/d: stacked probability bars.
    def stacked(ax,prob,title):
        y=np.arange(len(prob));left=np.zeros(len(prob))
        for fate in prob.columns:
            v=prob[fate].to_numpy(float)
            ax.barh(y,v,left=left,color=palette.get(fate,LIGHT),edgecolor='white',linewidth=.4,height=.7)
            left+=v
        ax.set_xlim(0,1);ax.set_yticks(y);ax.set_yticklabels(labels);ax.invert_yaxis()
        ax.set_xlabel('Future-fate probability');ax.set_title(title,loc='left');clean(ax)
    stacked(ax_c,truth_prob,'Barcode readout: terminal fate composition')
    stacked(ax_d,pred_prob,'LineageBridge: predicted future-fate probabilities')
    panel_label(ax_c,'c',x=-.28);panel_label(ax_d,'d',x=-.28)
    used=[f for f in truth_prob.columns if truth_prob[f].sum()+pred_prob[f].sum()>.03]
    fig.legend(handles=[Patch(facecolor=palette.get(f,LIGHT),label=f) for f in used],loc='lower center',ncol=len(used),frameon=False,bbox_to_anchor=(.5,-.005),columnspacing=1.2,handlelength=1.0)
    fig.subplots_adjust(left=.06,right=.985,top=.94,bottom=.12)
    emit(fig,stem)

# ------------------------------------------------------------------ SI 8

def branch_mixture(m,runs,stem):
    """Branch-mixture evidence: calibration, ambiguity strata, collapse, per-fate recovery."""
    apply_style()
    cal=m.binned_entropy_calibration(runs);strata=m.ambiguity_strata(runs)
    diag=m.collapse_diagnostic(runs['BMF']);pf=m.per_fate_summary(runs['BMF'])
    fig,axes=plt.subplots(2,2,figsize=(7.6,6.2),gridspec_kw={'width_ratios':[1.15,1.0],'wspace':.45,'hspace':.55})
    ax=axes[0,0]
    for name in ['VGFM','Dynamical OT','BMF']:
        d=cal[cal.run.eq(name)]
        ax.plot(d.truth_effective_fates,d.pred_effective_fates,marker='o',ms=3.2,lw=1.1,color=MODEL_COLORS[name],label=MODEL_LABELS[name])
    ax.plot([1,5.4],[1,5.4],color=LIGHT,lw=.8,ls=(0,(4,2)),zorder=0)
    ax.set_xlabel('Observed effective future fates');ax.set_ylabel('Predicted effective fates')
    ax.set_xlim(1,5.35);ax.set_ylim(.95,2.35);clean(ax)
    ax.set_title('Uncertainty calibration',loc='left');panel_label(ax,'a',x=-.2)
    ax=axes[0,1]
    order=['low','mid','high'];x=np.arange(3);w=.24
    for i,name in enumerate(['VGFM','Dynamical OT','BMF']):
        d=strata[strata.run.eq(name)].set_index('ambiguity').loc[order]
        ax.bar(x+(i-1)*w,d.brier,width=w,color=MODEL_COLORS[name],label=MODEL_LABELS[name])
    ax.set_xticks(x);ax.set_xticklabels(['Low','Mid','High']);ax.set_xlabel('Observed fate ambiguity')
    ax.set_ylabel('Probability error (Brier)');clean(ax)
    ax.set_title('Error by ambiguity',loc='left');panel_label(ax,'b',x=-.22)
    ax=axes[1,0]
    x=np.arange(len(diag))
    ax.bar(x,diag.CFCorr,width=.5,color=[BLUE,GRAY])
    ax.set_xticks(x);ax.set_xticklabels(['Branch mixture','Forced single fate']);ax.set_ylim(0,.5);ax.set_xlim(-.6,1.6)
    ax.set_ylabel('Cross-fate correlation (CFCorr)');clean(ax)
    ax.set_title('Single-fate collapse',loc='left');panel_label(ax,'c',x=-.2)
    ax=axes[1,1]
    d=pf[pf.truth_mean.gt(.003)].sort_values('correlation',ascending=True)
    y=np.arange(len(d))
    ax.barh(y,d.correlation,height=.6,color=BLUE)
    ax.set_yticks(y);ax.set_yticklabels(d.fate);ax.set_xlim(0,.86)
    ax.set_xlabel('Observed–predicted correlation');clean(ax)
    ax.set_title('Per-fate recovery, LineageBridge',loc='left');panel_label(ax,'d',x=-.42)
    handles=[Line2D([],[],color=MODEL_COLORS[n],marker='o',ms=3.2,lw=1.1,label=MODEL_LABELS[n]) for n in ['BMF','VGFM','Dynamical OT']]
    fig.legend(handles=handles,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,-.005),columnspacing=2.0)
    fig.subplots_adjust(left=.1,right=.98,top=.93,bottom=.12)
    emit(fig,stem)

# ------------------------------------------------------------------ SI 9

def clone_case_studies(m,run,adata,paths,stem):
    """Representative LARRY clones: observed descendants and model branches on the fate map."""
    apply_style()
    obs=spring_frame(adata);palette=m.FATE_PALETTE
    selected=['2431','2484','3530','4061'];markers={'2431':'o','2484':'s','3530':'D','4061':'^'}
    mix=m.selected_clone_mixtures(obs,run,selected)
    fig=plt.figure(figsize=(11.0,4.9))
    gs=fig.add_gridspec(1,2,width_ratios=[1.45,.9],wspace=.08)
    ax=fig.add_subplot(gs[0,0]);bx=fig.add_subplot(gs[0,1])
    fate_background(ax,obs,palette,s=2.0,alpha=.06,terminal_only=True)
    pw=paths.drop_duplicates(['clone','source_pos','branch_fate'])[['clone','source_pos','branch_fate','branch_weight','terminal_fate']].copy()
    pw['rank']=pw.groupby(['clone','source_pos']).branch_weight.rank(method='first',ascending=False)
    keep=pw[pw['rank'].le(3)&pw.branch_weight.ge(.004)][['clone','source_pos','branch_fate']]
    visible=paths.merge(keep,on=['clone','source_pos','branch_fate'],how='inner')
    n_day2=pw.groupby('clone').source_pos.nunique().to_dict()
    for clone in selected:
        mk=markers[clone];g=obs[obs.clone_str.eq(clone)]
        for fate,d in g.groupby('cell_type'):
            ax.scatter(d.spring_x,d.spring_y,s=26,marker=mk,facecolor=palette.get(fate,GRAY),edgecolor='white',linewidth=.5,alpha=.9,zorder=15)
        d0=g[g.t.eq(0)]
        if len(d0):
            ax.scatter(d0.spring_x,d0.spring_y,s=60,marker=mk,facecolor='white',edgecolor=BLACK,linewidth=1.0,zorder=25)
        cp=visible[visible.clone.astype(str).eq(clone)];n=max(int(n_day2.get(int(clone),1)),1)
        for (_,_),d in cp.groupby(['source_pos','branch_fate']):
            d=d.sort_values('model_time');fate=str(d.terminal_fate.iloc[0]);color=palette.get(fate,GRAY)
            p=float(np.clip(float(d.branch_weight.iloc[0])*n,0,1))
            ax.plot(d.spring_x,d.spring_y,color=color,lw=.4+1.8*np.sqrt(p),alpha=.25+.55*np.sqrt(p),zorder=11)
            end=d.iloc[-1]
            ax.scatter(end.spring_x,end.spring_y,s=24+80*p,marker=mk,facecolor='white',edgecolor=color,linewidth=.9,zorder=23)
    spring_limits(ax,obs)
    ax.set_title('Observed descendants and LineageBridge branches, SPRING map',loc='left')
    shape=[Line2D([],[],marker=markers[c],ls='',markerfacecolor='white',markeredgecolor=BLACK,markeredgewidth=.9,ms=5.5,label=f'Clone {c}') for c in selected]
    legend(ax,handles=shape,loc='upper left',ncol=1,labelspacing=.3)
    panel_label(ax,'a',x=-.02)
    # b: observed versus predicted terminal composition per clone.
    fates=[f for f in palette if f in set(mix.fate)]
    ypos={};ticks=[];y=0
    for clone in selected:
        d=mix[mix.clone.eq(clone)]
        if d.empty:continue
        n0=int(d.n_day2.iloc[0]);n2=int(d.n_day6.iloc[0])
        for source,label in [('observed descendants','Observed'),('BMF from day-2 cells','LineageBridge')]:
            ypos[(clone,source)]=y;ticks.append((y,f'Clone {clone}, {label}' if False else label));y+=.5
        bx.text(-.02,(ypos[(clone,'observed descendants')]+ypos[(clone,'BMF from day-2 cells')])/2,f'Clone {clone}\n{n0} day-2, {n2} day-6 cells',
                ha='right',va='center',fontsize=8,color=BLACK,transform=bx.get_yaxis_transform())
        y+=.35
    for (clone,source),yy in ypos.items():
        d=mix[mix.clone.eq(clone)&mix.source.eq(source)].set_index('fate');left=0.
        for fate in fates:
            v=float(d.fraction.get(fate,0.))
            if v<=0:continue
            bx.barh(yy,v,left=left,height=.36,color=palette[fate],edgecolor='white',linewidth=.4)
            left+=v
        bx.text(1.02,yy,'Observed' if source.startswith('observed') else 'LineageBridge',ha='left',va='center',fontsize=8,transform=bx.get_yaxis_transform())
    bx.set_xlim(0,1);bx.set_ylim(-.3,max(ypos.values())+.35);bx.invert_yaxis()
    bx.set_yticks([]);bx.spines['left'].set_visible(False)
    bx.set_xlabel('Terminal fate fraction');clean(bx)
    bx.set_title('Terminal composition',loc='left')
    show=[f for f in ['Undifferentiated','Neutrophil','Monocyte','Baso','Mast','Meg'] if f in fates]
    fig.legend(handles=fate_legend(show,palette),loc='lower center',ncol=6,frameon=False,bbox_to_anchor=(.5,-.01),columnspacing=1.4)
    panel_label(bx,'b',x=-.3)
    fig.subplots_adjust(left=.02,right=.9,top=.93,bottom=.13)
    emit(fig,stem)

# ------------------------------------------------------------------ SI 10

def program_push(data,palette,programs,stem):
    """Program perturbations redirect predicted terminal fates."""
    apply_style()
    fates=[f for f in palette if f in set(data.fate)]
    programs=[p for p in programs if p in set(data.program)]
    scenarios=['baseline']+[f'{p} +5z' for p in programs]+[f'{p} -5z' for p in programs]
    scenarios=[s for s in scenarios if s in set(data.scenario)]
    fig,(ax,bx)=plt.subplots(1,2,figsize=(10.0,3.9),gridspec_kw={'width_ratios':[2.4,1.3],'wspace':.45})
    for i,scenario in enumerate(scenarios):
        row=data[data.scenario.eq(scenario)].set_index('fate').fraction;bottom=0
        for fate in fates:
            v=float(row.get(fate,0));ax.bar(i,v,bottom=bottom,width=.62,color=palette[fate],edgecolor='white',linewidth=.4);bottom+=v
    ax.set_xticks(range(len(scenarios)));ax.set_xticklabels([s.replace('baseline','Baseline').replace('-5z','−5z') for s in scenarios],rotation=40,ha='right')
    ax.set_ylabel('Terminal fate fraction');ax.set_ylim(0,1);ax.set_xlim(-.6,len(scenarios)-.4);clean(ax)
    ax.set_title('Predicted terminal fate composition',loc='left');panel_label(ax,'a',x=-.1)
    shown=[f for f in fates if data[data.fate.eq(f)].fraction.max()>.015]
    legend(ax,handles=fate_legend(shown,palette),loc='upper center',bbox_to_anchor=(.5,-.36),ncol=len(shown),columnspacing=1.2)
    base=data[data.scenario.eq('baseline')].set_index('fate').fraction
    pert=scenarios[1:]
    mat=np.array([[float(data[data.scenario.eq(sc)].set_index('fate').fraction.get(f,0)-base.get(f,0)) for f in shown] for sc in pert])
    lim=float(np.abs(mat).max())
    im=bx.imshow(mat,cmap='RdBu_r',vmin=-lim,vmax=lim,aspect='auto')
    bx.set_xticks(range(len(shown)));bx.set_xticklabels(shown,rotation=40,ha='right')
    bx.set_yticks(range(len(pert)));bx.set_yticklabels([sc.replace('-5z','−5z') for sc in pert])
    for sp in bx.spines.values():sp.set_visible(False)
    bx.tick_params(length=0)
    cb=fig.colorbar(im,ax=bx,fraction=.05,pad=.04);cb.set_label('Change in terminal fraction')
    cb.outline.set_linewidth(.5);cb.ax.tick_params(width=.5,length=2)
    bx.set_title('Change from baseline',loc='left');panel_label(bx,'b',x=-.55)
    fig.subplots_adjust(left=.07,right=.98,top=.9,bottom=.3)
    emit(fig,stem)

# ------------------------------------------------------------------ SI 11

def fate_probability_maps(m,runs,stem):
    """Barcode-derived versus predicted future-fate probabilities on the SPRING map, eligible cells only."""
    apply_style()
    fates=['Baso','Mast','Meg','Monocyte','Neutrophil','Undifferentiated']
    rows=[('truth','Barcode-derived'),('BMF','LineageBridge'),('VGFM','VGFM'),('Dynamical OT','Dynamical OT')]
    base=runs['BMF']['df'];base=base.loc[base.eligible.astype(bool)]
    assert len(base)==1728,'Unexpected final eligible cell set'
    fig,axes=plt.subplots(4,6,figsize=(11.4,8.0),gridspec_kw={'wspace':.08,'hspace':.25})
    for row,(name,label) in enumerate(rows):
        df=base if name=='truth' else runs[name]['df'].loc[runs[name]['df'].eligible.astype(bool)]
        assert np.array_equal(df.source_index,base.source_index)
        for col,fate in enumerate(fates):
            ax=axes[row,col];column=f'truth_{fate}' if name=='truth' else f'pred_{fate}'
            vmax=max(.5,float(base[f'truth_{fate}'].max()))
            sc=ax.scatter(df.spring_x,df.spring_y,c=df[column],s=3,cmap='viridis',vmin=0,vmax=vmax,lw=0,alpha=.9,rasterized=True)
            embedding_axis(ax)
            if row==0:ax.set_title(fate,pad=4)
            if col==0:ax.set_ylabel(label,rotation=90,labelpad=6)
            if row==len(rows)-1:
                cax=ax.inset_axes([.15,-.14,.7,.05])
                cb=fig.colorbar(sc,cax=cax,orientation='horizontal');cb.set_ticks([0,round(vmax,2)])
                cb.outline.set_linewidth(.5);cb.ax.tick_params(width=.5,length=2,labelsize=8)
    fig.text(.5,.035,'Future-fate probability (colour scale set per fate)',ha='center',va='top',fontsize=9)
    fig.subplots_adjust(left=.04,right=.99,top=.95,bottom=.09)
    emit(fig,stem)
