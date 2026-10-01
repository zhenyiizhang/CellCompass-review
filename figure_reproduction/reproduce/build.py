"""Data-only figure reconstruction. Never trains or imports the agent runtime."""
from __future__ import annotations
import argparse
import contextlib
import json
from pathlib import Path
import sys
import types
import traceback

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymupdf as pdf

from .legacy import ROOT,load
from . import si_schematic,si_plain,si_lineage,si_protein_worm

SOURCE=ROOT/'source'
DATA=ROOT/'plot_data'
E='emt/scripts/downstream/'
P='analysis/pulsebridge_story/scripts/'
L='mouse_new/scripts/downstream/'
B='multi-omics/scripts/downstream/'
C=B+'cite_pcfm_biology_landscape/'
W='lcfbm_packer_ea6cebc5_data_model_bundle/scripts/'
D='dynbench_results_share_20260618_170313/figure_drafts/'
OUT=ROOT/'output/reproduced'

def cells(name):
    obs=pd.read_csv(DATA/name/'obs.csv',index_col=0,low_memory=False)
    with np.load(DATA/name/'arrays.npz',allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
    result=types.SimpleNamespace(obs=obs,obsm=arrays,n_obs=len(obs),shape=(len(obs),0))
    if (DATA/name/'var.csv').exists():
        result.var=pd.read_csv(DATA/name/'var.csv',index_col=0)
        result.var_names=result.var.index
        with np.load(DATA/name/'varm.npz',allow_pickle=False) as z:result.varm={k:z[k] for k in z.files}
    return result

def reader(name):
    data=cells(name)
    return types.SimpleNamespace(read_h5ad=lambda _path:data)

def panel_dir(group):
    out=OUT/'panels'/group;out.mkdir(parents=True,exist_ok=True);return out

@contextlib.contextmanager
def capture(group):
    """Redirect only plot exports, never scientific input reads."""
    target=panel_dir(group)
    original=matplotlib.figure.Figure.savefig
    def save(fig,path,*args,**kwargs):
        # Keep historical identifiers in the immutable source, but use paper names.
        import re
        def paper_name(label):
            label=re.sub(r'Clone-CU?VGF-\s*BMF','LineageBridge',str(label))
            label=label.replace('CITE-PCFM','ProteinBridge')
            return re.sub(r'\bBMF\b','LineageBridge',label)
        def tick_name(label):
            updated=paper_name(label)
            if '\n' in str(label):updated=updated.replace('LineageBridge','Lineage\nBridge')
            return updated
        if not getattr(fig,'_paper_names_applied',False):
            from matplotlib.ticker import FixedFormatter,FuncFormatter
            from matplotlib.category import StrCategoryFormatter
            for ax in fig.axes:
                for axis in (ax.xaxis,ax.yaxis):
                    old=axis.get_major_formatter()
                    if isinstance(old,(FixedFormatter,FuncFormatter,StrCategoryFormatter)):
                        axis.set_major_formatter(FuncFormatter(lambda value,pos,formatter=old:tick_name(formatter(value,pos))))
            fig._paper_names_applied=True
        for item in fig.findobj(match=matplotlib.text.Text):
            item.set_text(paper_name(item.get_text()))
        name=Path(path).name
        # Always produce vector PDF plus a compact preview. Legacy SVG exports
        # are redundant for the assembled figures and can be very large.
        suffix=Path(name).suffix.lower()
        if suffix=='.svg':return
        if suffix=='.png':kwargs['dpi']=160
        return original(fig,target/name,*args,**kwargs)
    matplotlib.figure.Figure.savefig=save
    try:yield target
    finally:
        matplotlib.figure.Figure.savefig=original
        plt.close('all')

def emit(fig,stem):
    fig.savefig(stem+'.pdf',bbox_inches='tight')
    fig.savefig(stem+'.png',dpi=160,bbox_inches='tight')
    plt.close(fig)

def compose(name,panels,columns=1,width=842,height=595,natural=False):
    """Assemble regenerated vector panels. No reference scientific plots are read.

    With natural=True a single panel is centred at its native size (capped to
    the page) instead of being stretched to fill the page."""
    dest=OUT/(name+'.pdf');dest.parent.mkdir(parents=True,exist_ok=True)
    doc=pdf.open();page=doc.new_page(width=width,height=height)
    rows=(len(panels)+columns-1)//columns
    margin,gap=12,10
    cw=(width-2*margin-(columns-1)*gap)/columns
    ch=(height-2*margin-(rows-1)*gap)/rows
    for i,p in enumerate(panels):
        rect=pdf.Rect(margin+(i%columns)*(cw+gap),margin+(i//columns)*(ch+gap),margin+(i%columns)*(cw+gap)+cw,margin+(i//columns)*(ch+gap)+ch)
        src=pdf.open(p)
        if natural:
            pw,ph=src[0].rect.width,src[0].rect.height
            scale=min(1.0,rect.width/pw,rect.height/ph)
            w,h=pw*scale,ph*scale
            cx,cy=(rect.x0+rect.x1)/2,(rect.y0+rect.y1)/2
            rect=pdf.Rect(cx-w/2,cy-h/2,cx+w/2,cy+h/2)
        page.show_pdf_page(rect,src,0)
    doc.save(dest,garbage=4,deflate=True)
    page.get_pixmap(matrix=pdf.Matrix(1.3,1.3),alpha=False).save(dest.with_suffix('.png'))
    doc.close()

def artwork_clip(label,box):
    """Use only the explicitly specified conceptual/narrative artwork region."""
    source=next((ROOT/'artwork'/label).glob('*.ai'))
    doc=pdf.open(source);rect=doc[0].rect
    clip=pdf.Rect(box[0]*rect.width,box[1]*rect.height,box[2]*rect.width,box[3]*rect.height)
    out=panel_dir('schematics')/(label+'_'+('_'.join(map(str,box)))+'.pdf')
    crop=pdf.open();page=crop.new_page(width=clip.width,height=clip.height)
    page.show_pdf_page(page.rect,doc,0,clip=clip)
    crop.save(out,garbage=4,deflate=True);crop.close();doc.close()
    return out

def assemble(name,placements,width=595,height=842):
    """Place vector panels at explicit normalized page coordinates."""
    out=OUT/(name+'.pdf');doc=pdf.open();page=doc.new_page(width=width,height=height)
    for path,box in placements:
        src=pdf.open(path)
        page.show_pdf_page(pdf.Rect(box[0]*width,box[1]*height,box[2]*width,box[3]*height),src,0)
    doc.save(out,garbage=4,deflate=True)
    page.get_pixmap(matrix=pdf.Matrix(1.5,1.5),alpha=False).save(out.with_suffix('.png'))
    doc.close()

def si_1():
    # A conceptual diagram has no numerical input; it is drawn from the caption text.
    with capture('schematics') as p:si_schematic.model_lifecycle(str(p/'model_lifecycle'))
    compose('Supplementary_Figure_01',[p/'model_lifecycle.pdf'])

def emt():
    m=load(E+'plot_emt_cellcompass_main_panels.py');m.setup_style()
    from .emt_fingerprint import plot_fingerprint
    m.panel04_mechanism_fingerprint=lambda mech: plot_fingerprint(
        mech,modules=m.MECH_MODULES,colors=m.MODULE_COLORS,labels=m.MODULE_LABELS,
        cell_lines=m.CELL_LINES,stimuli=m.STIMULI,save_all=m.save_all)
    with capture('emt') as p:m.main()
    r=load(E+'redraw_emt_cell_nbt_story.py')
    with capture('emt'):r.main()
    route,snap,_,_,_=m.load_tables()
    summary,_,_,shared=r.load_tables()
    summary['display_route']=summary.apply(m.display_route_group,axis=1)
    with capture('emt'):
        si_plain.emt_timecourses(m,snap,str(p/'si02_emt_timecourses'))
        si_plain.emt_endpoint_scatter(m,route,str(p/'si03_endpoint_scatter'))
        si_plain.emt_route_phase(summary,str(p/'si04a_route_phase'))
        si_plain.emt_program_prevalence(shared,r.PROGRAM_LABELS,str(p/'si04b_program_prevalence'))
    compose('Supplementary_Figure_02',[p/'si02_emt_timecourses.pdf'])
    compose('Supplementary_Figure_03',[p/'si03_endpoint_scatter.pdf'])
    compose('Supplementary_Figure_04',[p/'si04a_route_phase.pdf',p/'si04b_program_prevalence.pdf'],columns=2,natural=True)

def pulse():
    n=load(P+'make_nature_main_figure_v2.py',ad=reader('pulse'))
    s=load(P+'make_pulsebridge_cell_dynamic_screening_figures.py')
    with capture('pulse') as p:
        si_plain.pulse_overview(n,str(p/'si05_pulse_overview'))
        si_plain.pulse_perturbation_controls(s,s.load_tables(),str(p/'si06_perturbation_controls'))
    compose('Supplementary_Figure_05',[p/'si05_pulse_overview.pdf'])
    compose('Supplementary_Figure_06',[p/'si06_perturbation_controls.pdf'])

def lineage():
    table=SOURCE/'mouse_new/downstream/cellcompass_clone_case'
    m=load(L+'compose_clone_downstream_story_panels.py',ad=reader('lineage'))
    runs={name:m.load_run(name) for name in m.RUN_TABLES}
    checks=OUT/'checks';checks.mkdir(parents=True,exist_ok=True)
    m.TABLE_DIR=checks
    data=cells('lineage')
    paths=pd.read_csv(table/'clone_downstream_selected_bmf_branch_paths.csv')
    s=load(L+'compose_clone_state_matched_fate_divergence.py',ad=reader('lineage'),TABLE_DIR=checks)
    push=load(L+'compose_clone_scdiffeq_style_biology.py')
    with capture('lineage') as p:
        si_lineage.state_matched_divergence(s,data,pd.read_csv(s.BMF_TABLE),str(p/'si07_state_matched_fate_divergence'))
        si_lineage.branch_mixture(m,runs,str(p/'si08_branch_mixture_evidence'))
        si_lineage.clone_case_studies(m,runs['BMF'],data,paths,str(p/'si09_clone_case_studies'))
        si_lineage.program_push(pd.read_csv(table/'clone_scdiffeq_program_push_fate_output.csv'),push.FATE_PALETTE,list(push.PROGRAMS),str(p/'si10_program_push'))
    for number,stem in [(7,'si07_state_matched_fate_divergence'),(8,'si08_branch_mixture_evidence'),(9,'si09_clone_case_studies'),(10,'si10_program_push')]:
        compose(f'Supplementary_Figure_{number:02}',[p/(stem+'.pdf')])
    # Eligible-only variant (1728 cells) drawn in the shared plain style.
    with capture('lineage'):
        si_lineage.fate_probability_maps(m,runs,str(p/'si11_fate_probability_maps'))
    compose('Supplementary_Figure_11',[p/'si11_fate_probability_maps.pdf'])

def protein_tables():
    m=load(B+'cite_pcfm_nbt_figure_bank.py')
    tables=m.load_tables()
    means,per=protein_energy()
    with capture('protein') as p:
        si_protein_worm.protein_quantitative(means,per,tables,str(p/'si13_protein_quantitative'))
    compose('Supplementary_Figure_13',[p/'si13_protein_quantitative.pdf'])
    r=load(C+'fig06_rna_protein_regulatory_map.py')
    table=r.OUT_DIR
    corr=pd.read_csv(table/'observed_rna_program_adt_correlations.csv')
    static=pd.read_csv(table/'learned_readout_adt_sensitivity.csv')
    trend=pd.read_csv(table/'observed_program_adt_binned_trends.csv')
    with capture('protein'):
        si_protein_worm.protein_readout_validation(r,corr,static,trend,str(p/'si14_readout_validation'))
    compose('Supplementary_Figure_14',[p/'si14_readout_validation.pdf'])

def worm():
    m=load(W+'visualize_worm_bundle.py');m.setup_style()
    def trajectory(_):
        z=np.load(DATA/'worm/trajectory.npz',allow_pickle=False)
        return z,sorted(k for k in z.files if k.startswith('points_'))
    m.load_trajectory=trajectory
    data=cells('worm')
    w1=json.loads((DATA/'worm/metrics.json').read_text())['w1_scores']
    with capture('worm') as p:
        si_protein_worm.lcfbm_distribution_match(data,trajectory(None),str(p/'si16_lcfbm_distribution_match'))
        si_protein_worm.lcfbm_w1_by_window(data,trajectory(None),w1,str(p/'si17_lcfbm_w1_by_window'))
    compose('Supplementary_Figure_16',[p/'si16_lcfbm_distribution_match.pdf'])
    compose('Supplementary_Figure_17',[p/'si17_lcfbm_w1_by_window.pdf'],natural=True)
    tree=load(W+'make_worm_panel_c_lineage_future.py',vb=m)
    flow=pd.read_csv(DATA/'worm/panel_c_lcfbm_endpoint_nearest_lineage_flow.csv')
    valid=flow[(flow.source_root!='missing')&(flow.nearest_final_root!='missing')]
    counts=pd.crosstab(valid.source_root,valid.nearest_final_root)
    concordant=int((valid.source_root==valid.nearest_final_root).sum())
    search=load(W+'make_worm_algorithm_search_panel.py')
    with capture('worm'):
        fig,axs=plt.subplots(1,2,figsize=(11,4),gridspec_kw={'width_ratios':[1.5,1]})
        search.draw_history(axs[0],pd.read_csv(DATA/'worm/panel_b_lcfbm_algorithm_trial_history.csv'))
        search.draw_final_metric(axs[1],pd.read_csv(DATA/'worm/panel_b_lcfbm_baseline_metrics.csv'))
        fig.tight_layout();emit(fig,'worm_search')
        fig=plt.figure(figsize=(12,6))
        ax=fig.add_axes([.01,.04,.53,.9]);tree.draw_lineage_atlas(ax,data,counts,concordant,len(valid))
        tree.draw_fixed_gene_dynamics_across_lineages(fig,(.59,.10,.39,.78),data,DATA/'worm')
        emit(fig,'worm_lineage_genes')
    assemble('Supplementary_Figure_15',[
        # Clip below the '(LCFBM)' caption line, above the panel b heading.
        (artwork_clip('Supplementary_15',(0,0,1,.3115)),(.015,.01,.985,.31)),
        (p/'worm_search.pdf',(.025,.32,.975,.61)),
        (p/'worm_lineage_genes.pdf',(.015,.62,.985,.985))])

def protein_energy():
    """Locked protein energy distances (mean and per target time) for the three methods."""
    m=load(C+'fig01_distribution_and_landscape.py')
    means={};per=[]
    for method,spec in m.METHODS.items():
        rows,mean=m.load_locked_protein_energy(method,spec['metrics'],{'points':{t:None for t in [1.,2.,3.,4.]}})
        means[method]=mean;per.extend(rows)
    assert round(means['ProteinBridge'],5)==.02220
    assert round(means['CRUFM'],5)==.02558
    pd.DataFrame(per).to_csv(OUT/'checks/protein_energy_recomputed.csv',index=False)
    return means,per

def protein_distribution():
    style=load(C+'multistageot_style.py')
    m=load(C+'fig01_distribution_and_landscape.py',**{k:getattr(style,k) for k in ['PHENO_COLORS','PHENO_ORDER','scatter_background','scatter_groups','add_median_labels','clean_embedding_axis']})
    obs=pd.read_csv(DATA/'protein/observed_coordinates.csv')
    gen=pd.read_csv(DATA/'protein/generated_target_coordinates.csv')
    target=obs[np.isclose(obs.time_point_processed,4)]
    means,per=protein_energy()
    with capture('protein') as p:
        si_protein_worm.protein_method_comparison(obs,gen,str(p/'si12_protein_method_comparison'))
    compose('Supplementary_Figure_12',[p/'si12_protein_method_comparison.pdf'])

def main_figures():
    import importlib
    # Figure 1 is a manually authored conceptual diagram, not a result plot.
    assemble('Main_Figure_01',[(artwork_clip('Figure_1',(0,0,1,1)),(0,0,1,1))])
    sys.path.insert(0,str(SOURCE/D))
    benchmark=importlib.import_module('plot_dynbench_results_5panel_final')
    design=importlib.import_module('plot_dynbench_main_figure')
    baseline,diff,_alg_diff,algorithms=benchmark.load_data()
    with capture('benchmark') as p:
        specs=[('b',lambda ax:design.draw_panel_b(ax)),
               ('c',lambda ax:benchmark.draw_panel_c(ax,baseline)),
               ('d',lambda ax:benchmark.draw_panel_d_clean(ax,baseline)),
               ('e',lambda ax:benchmark.draw_panel_e_baselines(ax,diff)),
               ('f',lambda ax:benchmark.draw_panel_g_selected20(ax)),
               ('g',lambda ax:benchmark.draw_panel_f_brgu_metrics(ax,baseline,algorithms))]
        for name,fn in specs:
            fig,ax=plt.subplots(figsize=(5.6,3.7));fn(ax);emit(fig,'benchmark_'+name)
    assemble('Main_Figure_02',[
        (artwork_clip('Figure_2',(0,0,1,.24)),(.015,.01,.985,.24)),
        (p/'benchmark_b.pdf',(.015,.245,.48,.48)),(p/'benchmark_c.pdf',(.49,.245,.985,.48)),
        (p/'benchmark_d.pdf',(.015,.49,.48,.735)),(p/'benchmark_e.pdf',(.49,.49,.985,.735)),
        (p/'benchmark_g.pdf',(.015,.745,.48,.985)),(p/'benchmark_f.pdf',(.49,.745,.985,.985))])
    emt_main_figure()
    pulse_main()
    lineage_main()
    protein_main()

def emt_main_figure():
    e=panel_dir('emt')
    emt_main_timing()
    assemble('Main_Figure_03',[
        (artwork_clip('Figure_3',(0,0,1,.263)),(.01,.005,.99,.263)),
        (artwork_clip('Figure_3',(0,.27,.505,.497)),(.01,.275,.505,.497)),
        (artwork_clip('Figure_3',(0,.50,.35,.535)),(.01,.50,.353,.535)),
        (artwork_clip('Figure_3',(0,.54,.508,.805)),(.01,.54,.508,.805)),
        (e/'panel02_condition_route_atlas.pdf',(.52,.275,.99,.52)),
        (e/'panel04_mechanism_fingerprint_cards.pdf',(.52,.535,.99,.80)),
        (e/'main_timing.pdf',(.01,.815,.51,.99)),
        (e/'panel05_three_layer_mechanism.pdf',(.52,.815,.99,.99))])
def emt_main_timing():
    m=load(E+'plot_emt_cellcompass_main_panels.py')
    _,_,mech,traj,_=m.load_tables()
    indexed=mech.set_index('condition')
    with capture('emt'):
        fig,axes=plt.subplots(1,2,figsize=(7,3.5))
        for ax,condition in zip(axes,['A549_TGFB1','MCF7_TGFB1']):
            kind=str(indexed.loc[condition,'mechanism_class'])
            programs=m.ROUTE_DIAGNOSTIC_MODULES[kind]
            sub=traj[traj.condition.eq(condition)].sort_values('model_time')
            for program in programs:
                y=sub['gen_'+program].to_numpy()
                ax.plot(sub.model_time,y-y[0],color=m.MODULE_COLORS[program],label=m.MODULE_LABELS[program],lw=1.4)
            ax.plot(sub.model_time,sub.mass_sum,color='#98A2B3',ls='--',label='effective mass',lw=1.4)
            ax.set_xticks(range(5),m.TIME_LABELS,fontsize=8)
            ax.set_title(m.pretty_condition(condition),fontsize=10)
            ax.legend(fontsize=6,frameon=False,loc='upper left')
            ax.spines[['top','right']].set_visible(False)
        axes[0].set_ylabel('module change from t0 / effective mass',fontsize=8)
        fig.suptitle('Representative CellCompass-generated route dynamics',fontsize=11)
        fig.tight_layout();emit(fig,'main_timing')

def pulse_main():
    m=load(P+'make_pulsebridge_algorithm_search_figure.py');m.style()
    tables=SOURCE/'analysis/pulsebridge_story/outputs/tables'
    with capture('pulse') as p:
        fig,axs=plt.subplots(1,2,figsize=(10,4),gridspec_kw={'width_ratios':[1.8,1]})
        m.plot_panel_b(axs[0],pd.read_csv(tables/'pulsebridge_algorithm_full_lifecycle_events.csv'),pd.read_csv(tables/'pulsebridge_algorithm_stage_pass_counts.csv'))
        m.plot_panel_c(axs[1],pd.read_csv(tables/'pulsebridge_algorithm_pcie_baselines.csv'))
        fig.tight_layout();emit(fig,'pulse_search')
        s=load(P+'make_pulsebridge_cell_dynamic_screening_figures.py');s.style();data=s.load_tables()
        fig=plt.figure(figsize=(11,6));gs=fig.add_gridspec(2,2,width_ratios=[1.1,1],hspace=.5,wspace=.3)
        ax=fig.add_subplot(gs[:,0]);draw_frozen_pulse_stream(ax,s)
        programs=fig.add_subplot(gs[0,1]);s.main_panel_programs(programs,data)
        for text in programs.texts:
            for prefix,y in [('DKK1/GAS5',.22),('EREG/PCNA',.42),('B2M/HLA/IFITM3',.02),('SAT1/LGALS1/MT',.69)]:
                if text.get_text().startswith(prefix):text.set_y(y)
        adaptive=fig.add_subplot(gs[1,1]);s.main_panel_adaptive(adaptive,data)
        bottom,_=adaptive.get_ylim();adaptive.set_ylim(bottom,-1.5)
        emit(fig,'pulse_biology')
    assemble('Main_Figure_04',[
        (artwork_clip('Figure_4',(0,0,1,.32)),(.01,.005,.99,.315)),
        (p/'pulse_search.pdf',(.015,.33,.985,.59)),
        (p/'pulse_biology.pdf',(.01,.61,.99,.99))])

def draw_frozen_pulse_stream(ax,style):
    with np.load(DATA/'pulse/stream_grid.npz',allow_pickle=False) as z:
        arrays={k:z[k] for k in z.files}
    options=json.loads((DATA/'pulse/stream_options.json').read_text())
    for key in options.pop('array_options'):options[key]=arrays[key]
    data=cells('pulse');xy=data.obsm['X_umap']
    ax.scatter(xy[:,0],xy[:,1],c=data.obs.time_point_processed,cmap=style.TIME_CMAP,s=3,alpha=.22,lw=0,rasterized=True)
    ax.streamplot(arrays['x'],arrays['y'],arrays['u'],arrays['v'],**options)
    style._style_dynamo_stream_ax(ax,xy)
    ax.set_title('Projected drug-response velocity field',loc='left',fontsize=10)

def lineage_main():
    table=SOURCE/'mouse_new/downstream/cellcompass_clone_case'
    m=load(L+'compose_clone_panel_b_algorithm_landscape.py',TABLE_DIR=OUT/'checks')
    m.load_trial_curve=lambda:pd.read_csv(table/'clone_panel_b_trial_curve.csv')
    m.load_metrics=lambda:pd.read_csv(table/'clone_panel_b_final_metrics.csv')
    with capture('lineage') as p:
        m.compose()
        b=load(L+'compose_clone_panel_c_bmf_early_fate_bias_story.py',ad=reader('lineage'),PARTICLE_TABLE=DATA/'lineage/particles.csv.gz')
        b.main()
    assemble('Main_Figure_05',[
        (artwork_clip('Figure_5',(0,0,1,.36)),(.01,.005,.99,.355)),
        (p/'panel_b_clone_algorithm_landscape.pdf',(.01,.37,.99,.655)),
        (p/'panel_c_bmf_early_fate_bias_story.pdf',(.01,.67,.99,.99))])

def protein_main():
    table=SOURCE/'multi-omics/downstream/cite_pcfm_deep_story/tables'
    m=load(B+'cite_pcfm_algorithm_search_landscape/make_algorithm_search_landscape.py',TABLE_DIR=OUT/'checks')
    runs=pd.read_csv(table/'cellcompass_cite_algorithm_runs.csv')
    trials=pd.read_csv(table/'cellcompass_cite_algorithm_trials.csv')
    baseline=pd.read_csv(table/'cellcompass_final_panel_builtin_baselines.csv')
    with capture('protein') as p:
        m.fig_search13_multibasin_algorithm_landscape_3d(runs,trials,baseline)
        protein_metric_panel(baseline)
        style=load(C+'multistageot_style.py')
        h=load(C+'fig01h_protein_rollout_on_fig01a_style.py',**{k:getattr(style,k) for k in ['PHENO_COLORS','PHENO_ORDER','add_median_labels','clean_embedding_axis','setup']})
        body=load(C+'compose_panel_c_rna_protein_relay.py')
        body.draw_rollout_paths=h.draw_rollout_paths
        def draw(ax,show_title=True,compact=False):
            obs=pd.read_csv(DATA/'protein/observed_coordinates.csv')
            xy=obs[['protein_x','protein_y']].to_numpy();labels=obs.cell_type.to_numpy()
            with np.load(DATA/'protein/selected_paths.npz',allow_pickle=False) as z:
                paths=z['xy'];times=z['times'];basins=z['terminal_basin']
            h.draw_observed_fig01a_style(ax,xy,labels,label_order=body.OBSERVED_LABELS_FOR_C1)
            h.draw_rollout_paths(ax,paths,times,basins,include_labels=body.REPRESENTATIVE_PATHS,skip_labels=(),show_arrows=False,max_faint_per_basin=3,marker_size=23)
            body.add_start_zoom_inset(ax,obs_emb=xy,labels=labels,path_xy=paths,time_points=times,terminal_basin=basins)
            ax.set_xticks([]);ax.set_yticks([]);ax.set_aspect('equal')
            for spine in ax.spines.values():spine.set_visible(False)
        body.draw_trajectory=draw
        body.main()
    assemble('Main_Figure_06',[
        (artwork_clip('Figure_6',(0,0,1,.335)),(.01,.005,.99,.33)),
        (p/'search_landscape13_multibasin_algorithm_landscape_3d.pdf',(.01,.34,.75,.625)),
        (p/'protein_metric_comparison.pdf',(.76,.34,.99,.625)),
        (p/'panel_c_rna_to_surface_protein_relay_compact_a4.pdf',(.01,.64,.99,.99))])

def protein_metric_panel(baseline):
    metrics=load(C+'fig01_distribution_and_landscape.py')
    spec=metrics.METHODS['ProteinBridge']
    _,value=metrics.load_locked_protein_energy('ProteinBridge',spec['metrics'],{'points':{t:None for t in [1.,2.,3.,4.]}})
    reference=float(baseline.loc[baseline.algorithm.eq('crufm'),'protein_energy'].iloc[0])
    other=baseline.loc[~baseline.algorithm.eq('crufm'),'protein_energy'].dropna()
    fig,ax=plt.subplots(figsize=(3,4.2))
    ax.bar([0,1],np.array([value,reference])-.018,bottom=.018,width=.45,color=['#7664AB','#63A78B'])
    ax.plot([2,2],[other.min(),other.max()],color='#8C8C8C',lw=3)
    ax.scatter([2,2],[other.min(),other.max()],color='#8C8C8C',s=18)
    for i,v in enumerate([value,reference]):ax.text(i,v*1.08,f'{v:.4f}',ha='center',fontsize=8)
    ax.set_yscale('log');ax.set_ylim(.018,1.05);ax.set_xlim(-.6,2.6)
    ticks=[.02,.03,.1,.3,1];ax.set_yticks(ticks,[str(t) for t in ticks]);ax.minorticks_off()
    ax.set_xticks([0,1,2],['ProteinBridge','CRUFM','Other baselines'],rotation=45,ha='right',fontsize=8)
    ax.set_ylabel('mean protein energy distance')
    ax.set_title(f'{100*(1-value/reference):.1f}% lower than CRUFM',fontsize=9)
    ax.spines[['top','right']].set_visible(False)
    fig.tight_layout();emit(fig,'protein_metric_comparison')

JOBS={'si1':si_1,'emt':emt,'pulse':pulse,'lineage':lineage,'protein':protein_tables,'protein-distribution':protein_distribution,'main':main_figures}

def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only',nargs='+',choices=list(JOBS),default=list(JOBS))
    parser.add_argument('--output',type=Path,default=OUT)
    args=parser.parse_args()
    OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'checks').mkdir(exist_ok=True)
    results=[]
    jobs=list(args.only)
    if 'main' in jobs and 'emt' not in jobs:jobs.insert(jobs.index('main'),'emt')
    for name in jobs:
        print('REBUILD',name,flush=True)
        try:
            JOBS[name]();results.append({'job':name,'status':'passed'})
        except Exception:
            error=traceback.format_exc();print(error,file=sys.stderr)
            results.append({'job':name,'status':'failed','error':error})
    (OUT/'build_report.json').write_text(json.dumps(results,indent=2)+'\n')
    if any(r['status']=='failed' for r in results):raise SystemExit(1)
    if set(jobs)==set(JOBS):
        expected=[f'Main_Figure_{i:02}' for i in range(1,7)]+[f'Supplementary_Figure_{i:02}' for i in range(1,15)]
        for stem in expected:
            for suffix in ['.pdf','.png']:
                assert (OUT/(stem+suffix)).stat().st_size>1000, f'Missing figure: {stem+suffix}'
        print('VERIFIED: 6 main figures and 14 supplementary figures',flush=True)

if __name__=='__main__':main()
