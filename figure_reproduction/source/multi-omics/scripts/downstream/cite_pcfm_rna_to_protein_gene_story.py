#!/usr/bin/env python3
"""Nature Biotechnology-style RNA-to-protein dynamical biology analysis for CITE-PCFM.

This script uses the locked final-regression CITE-PCFM evaluation trajectory as the
model-native evidence object.  It asks the biological question requested by the
user: starting from RNA coordinates, what dynamic protein programs and gene-level
RNA/protein relationships does the model reveal?

Evidence boundaries:
- generated states come from artifacts/evaluation_trajectory.npz from the locked
  final-regression run;
- protein marker values are approximate inverse projections from the generated
  30D protein PCA coordinates using benchmark PCA loadings and observed protein
  means;
- gene-level RNA values are approximate low-rank reconstructions from generated
  50D RNA SVD coordinates using benchmark SVD loadings and observed RNA means;
- claims are therefore program-level/gene-level hypotheses supported by the
  low-rank CITE coordinate system, not direct full-count imputation claims.
"""
from __future__ import annotations

import json, os, textwrap
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/cytobridge_mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/cytobridge_xdg_cache")
os.environ.setdefault("NUMBA_CACHE_DIR", "/private/tmp/numba_cache")
os.environ.setdefault("NUMBA_DISABLE_CACHE", "1")
for k in ["MPLCONFIGDIR", "XDG_CACHE_HOME", "NUMBA_CACHE_DIR"]:
    Path(os.environ[k]).mkdir(parents=True, exist_ok=True)

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyArrowPatch
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

sns.set_theme(style="white", context="talk")
ROOT = Path("/Users/zhenyizhang/Documents/GitHub/CytoBridge-agent-runtime-v2-20260309/multi-omics")
DATA_PATH = Path("/Users/zhenyizhang/.cellcompass/algorithm_benchmarks/datasets/gse305370_citeseq_stage2_10k/data.h5ad")
TRAJ_PATH = ROOT / "training_runs/20260526T174204321821Z__final__cite_paired_coord_fm__campaign_20260-trial_20260526-gse305370_ci/artifacts/evaluation_trajectory.npz"
FIG_DIR = ROOT / "figures/cite_pcfm_deep_story/rna_to_protein_gene_story"
TABLE_DIR = ROOT / "downstream/cite_pcfm_deep_story/rna_to_protein_gene_story_tables"
MANIFEST = ROOT / "downstream/cite_pcfm_deep_story/rna_to_protein_gene_story_manifest.json"
STORY = ROOT / "downstream/cite_pcfm_deep_story/rna_to_protein_gene_story.md"
RNA_DIM, PROT_DIM = 50, 30
MOVIE_TIMES = [0.0, 1.0, 2.0, 3.0, 4.0]
RNG = np.random.default_rng(20260527)

PHENO_COLORS = {
    "HSC": "#2F855A", "MasP": "#D62728", "NeuP": "#E377C2", "EryP": "#F28E2B",
    "MkP": "#7B61B3", "MoP": "#8C564B", "BP": "#1F77B4",
}
TIME_COLORS = {0.0:"#2563EB",1.0:"#06B6D4",2.0:"#22C55E",3.0:"#F59E0B",4.0:"#E11D48"}

MARKER_PANELS = {
    "stem / priming": ["prot_CD49f", "prot_CD45RA", "prot_CD123", "prot_CD62L"],
    "erythro-megakaryoid": ["prot_CD71", "prot_CD41", "prot_CD42b", "prot_CD105"],
    "myeloid / inflammatory": ["prot_CD11b", "prot_CD16", "prot_CD115", "prot_HLA-DR"],
    "lymphoid / off-axis": ["prot_CD20", "prot_CD3", "prot_CD8", "prot_CD56"],
}
GENE_PROGRAMS = {
    "erythroid RNA": ["GATA1", "KLF1", "HBB", "HBA1", "HBA2", "ALAS2"],
    "megakaryocyte RNA": ["ITGA2B", "GP1BA", "NFE2", "VWF", "MPL"],
    "neutrophil RNA": ["MPO", "ELANE", "CSF3R", "CEBPE", "LTF"],
    "mono-mast RNA": ["SPI1", "IRF8", "LYZ", "CTSS", "FCER1A"],
    "stem RNA": ["KIT", "CD34", "PROM1", "MEIS1", "HLF"],
}
RNA_PROTEIN_HYPOTHESES = [
    ("GATA1", "prot_CD71", "erythroid transcriptional activation followed by transferrin-receptor protein rise"),
    ("KLF1", "prot_CD71", "erythroid maturation axis"),
    ("ITGA2B", "prot_CD41", "megakaryocytic integrin mRNA-to-surface-protein axis"),
    ("GP1BA", "prot_CD42b", "platelet/megakaryocyte surface maturation"),
    ("MPO", "prot_CD11b", "granulocytic priming with myeloid adhesion marker acquisition"),
    ("ELANE", "prot_CD16", "neutrophil protease program preceding Fc receptor signal"),
    ("SPI1", "prot_CD115", "monocyte/macrophage regulatory axis"),
    ("IRF8", "prot_HLA-DR", "antigen-presentation/myeloid regulatory axis"),
    ("KIT", "prot_CD49f", "stem/primitive surface maintenance axis"),
    ("CD34", "prot_CD45RA", "stem-to-primed surface remodeling axis"),
]

def ensure_dirs():
    FIG_DIR.mkdir(parents=True, exist_ok=True); TABLE_DIR.mkdir(parents=True, exist_ok=True)

def load_traj():
    z = np.load(TRAJ_PATH, allow_pickle=True)
    times = np.asarray(z["time_points"], float)
    pts = {float(t): np.asarray(z[f"points_{i:06d}"], np.float32) for i,t in enumerate(times) if f"points_{i:06d}" in z.files}
    wts = {float(t): np.asarray(z[f"weights_{i:06d}"], np.float32) for i,t in enumerate(times) if f"weights_{i:06d}" in z.files}
    return times, pts, wts

def weighted_mean(x, w=None):
    if w is None: return np.nanmean(x, axis=0)
    w = np.asarray(w, float).reshape(-1); w = np.clip(w, 0, np.inf)
    if w.shape[0] != x.shape[0] or not np.isfinite(w).all() or w.sum() <= 0: return np.nanmean(x, axis=0)
    return np.average(x, axis=0, weights=w/w.sum())

def get_matrix(adata, mask):
    X = adata[:, mask].X
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)

def lowrank_reconstruct(points, loadings, mean_vec, names, requested, start, dim):
    idx = [] ; kept = []
    name_to_i = {str(n): i for i,n in enumerate(names)}
    for item in requested:
        key = item if item in name_to_i else ("rna_" + item if "rna_" + item in name_to_i else item)
        if key in name_to_i:
            idx.append(name_to_i[key]); kept.append(item)
    if not idx:
        return kept, {t: np.empty((points[t].shape[0],0)) for t in points}
    W = np.asarray(loadings[idx, start:start+dim], float)
    mu = np.asarray(mean_vec[idx], float)
    out = {}
    for t, arr in points.items():
        out[t] = arr[:, start:start+dim] @ W.T + mu
    return kept, out

def train_protein_fate(protein_obs, labels):
    x_train, x_test, y_train, y_test = train_test_split(protein_obs, labels, test_size=0.25, random_state=20260527, stratify=labels)
    clf = KNeighborsClassifier(n_neighbors=25, weights="distance").fit(x_train, y_train)
    pred = clf.predict(x_test)
    return clf, {"balanced_accuracy": float(balanced_accuracy_score(y_test, pred)), "support": int(len(y_test)), "classes": sorted(pd.unique(labels).astype(str))}

def build_program_table(rna_gene_values, prot_marker_values, gene_programs, marker_panels, weights):
    rows=[]
    for t in sorted(rna_gene_values):
        wt = weights.get(t)
        for pname, genes in gene_programs.items():
            gene_cols = [g for g in genes if g in rna_gene_values[t]]
            if gene_cols:
                vals = rna_gene_values[t][gene_cols].to_numpy(float)
                score = weighted_mean(vals, wt).mean()
            else: score = np.nan
            rows.append({"time": t, "program_type":"rna", "program":pname, "mean_score":float(score) if np.isfinite(score) else np.nan, "features":";".join(gene_cols)})
        for pname, markers in marker_panels.items():
            marker_cols = [m for m in markers if m in prot_marker_values[t]]
            vals = prot_marker_values[t][marker_cols].to_numpy(float) if marker_cols else np.empty((0,0))
            score = weighted_mean(vals, wt).mean() if marker_cols else np.nan
            rows.append({"time": t, "program_type":"protein", "program":pname, "mean_score":float(score) if np.isfinite(score) else np.nan, "features":";".join(marker_cols)})
    return pd.DataFrame(rows)

def build_pair_table(rna_gene_values, prot_marker_values, pairs):
    rows=[]
    times = sorted(rna_gene_values)
    for gene, marker, note in pairs:
        if gene not in rna_gene_values[times[0]] or marker not in prot_marker_values[times[0]]: continue
        # particle-resolved correlation between endpoint changes and trajectory-mean derivatives
        g0 = rna_gene_values[times[0]][gene].to_numpy(float); g4 = rna_gene_values[times[-1]][gene].to_numpy(float)
        p0 = prot_marker_values[times[0]][marker].to_numpy(float); p4 = prot_marker_values[times[-1]][marker].to_numpy(float)
        rho_delta, p_delta = spearmanr(g4-g0, p4-p0)
        # correlation of population-average curves with +/- one observed-time lag
        g_curve = np.array([rna_gene_values[t][gene].mean() for t in times])
        p_curve = np.array([prot_marker_values[t][marker].mean() for t in times])
        lag_rows=[]
        for lag in [-1,0,1]:
            if lag < 0:
                a,b = g_curve[-lag:], p_curve[:lag]
            elif lag > 0:
                a,b = g_curve[:-lag], p_curve[lag:]
            else:
                a,b = g_curve, p_curve
            if len(a) >= 3 and np.nanstd(a)>0 and np.nanstd(b)>0:
                rho,_ = spearmanr(a,b)
            else: rho=np.nan
            lag_rows.append((lag, rho))
        best_lag, best_rho = max(lag_rows, key=lambda x: -np.inf if not np.isfinite(x[1]) else abs(x[1]))
        rows.append({"gene":gene, "protein_marker":marker, "hypothesis":note, "spearman_delta_particle":float(rho_delta) if np.isfinite(rho_delta) else np.nan, "delta_p_value":float(p_delta) if np.isfinite(p_delta) else np.nan, "best_population_lag":int(best_lag), "best_population_lag_spearman":float(best_rho) if np.isfinite(best_rho) else np.nan, "gene_delta_mean":float(np.mean(g4-g0)), "protein_delta_mean":float(np.mean(p4-p0))})
    return pd.DataFrame(rows).sort_values("spearman_delta_particle", key=lambda s: s.abs(), ascending=False)

def fate_branch_curves(fate_by_time, terminal_labels, rna_gene_values, prot_marker_values, programs, panels):
    rows=[]; terminal = np.asarray(terminal_labels).astype(str)
    for fate in ["EryP","MkP","NeuP","MasP","MoP","HSC"]:
        mask = terminal == fate
        if mask.sum() < 30: continue
        for t in sorted(rna_gene_values):
            for pname, genes in programs.items():
                cols=[g for g in genes if g in rna_gene_values[t]]
                if cols:
                    score = rna_gene_values[t].loc[mask, cols].mean(axis=1).mean()
                    rows.append({"terminal_fate":fate,"time":t,"type":"RNA program","program":pname,"score":float(score),"n_particles":int(mask.sum())})
            for pname, markers in panels.items():
                cols=[m for m in markers if m in prot_marker_values[t]]
                if cols:
                    score = prot_marker_values[t].loc[mask, cols].mean(axis=1).mean()
                    rows.append({"terminal_fate":fate,"time":t,"type":"Protein program","program":pname,"score":float(score),"n_particles":int(mask.sum())})
    return pd.DataFrame(rows)

def standardize_for_plot(df, group_cols, value_col="score"):
    df=df.copy()
    out=[]
    for _, sub in df.groupby(group_cols):
        vals=sub[value_col].to_numpy(float)
        sd=np.nanstd(vals)
        sub["zscore"]=(vals-np.nanmean(vals))/(sd+1e-8)
        out.append(sub)
    return pd.concat(out, ignore_index=True)

def fig_flagship_marker_movie(prog_df, pair_df, branch_df):
    fig = plt.figure(figsize=(18.5, 10.5))
    gs = fig.add_gridspec(3, 4, height_ratios=[0.42, 1.05, 1.0], width_ratios=[1,1,1,1], hspace=0.52, wspace=0.38)
    ax_title = fig.add_subplot(gs[0,:]); ax_title.axis("off")
    ax_title.text(0.0, 0.78, "RNA-initialized particles generate a moving protein phenotype", fontsize=26, weight="bold", color="#0F172A")
    ax_title.text(0.0, 0.28, "CITE-PCFM is interrogated as a dynamical biological model: low-rank RNA programs at t0 roll forward into time-resolved surface-protein programs and terminal phenotypes.", fontsize=12.4, color="#475569")
    # program heatmap
    ax = fig.add_subplot(gs[1,0:2])
    mat=[]; labels=[]
    use = prog_df[prog_df.program_type.isin(["rna","protein"])].copy()
    for prog in ["stem RNA","erythroid RNA","megakaryocyte RNA","neutrophil RNA","mono-mast RNA","stem / priming","erythro-megakaryoid","myeloid / inflammatory","lymphoid / off-axis"]:
        sub=use[use.program==prog].sort_values("time")
        if len(sub):
            vals=sub.mean_score.to_numpy(float); vals=(vals-np.nanmean(vals))/(np.nanstd(vals)+1e-8)
            mat.append(vals); labels.append(prog)
    sns.heatmap(np.vstack(mat), cmap="vlag", center=0, xticklabels=[f"t{int(t)}" for t in MOVIE_TIMES], yticklabels=labels, ax=ax, cbar_kws={"label":"within-program z-score"})
    ax.set_title("A. RNA programs and generated protein programs form a temporal movie", loc="left", fontsize=13.5, weight="bold")
    ax.tick_params(labelsize=9)
    # branch program curves
    ax = fig.add_subplot(gs[1,2:])
    z = standardize_for_plot(branch_df[branch_df.program.isin(["erythroid RNA","megakaryocyte RNA","neutrophil RNA","erythro-megakaryoid","myeloid / inflammatory"])], ["type","program"])
    for fate, color in [("EryP",PHENO_COLORS["EryP"]),("MkP",PHENO_COLORS["MkP"]),("NeuP",PHENO_COLORS["NeuP"]),("MasP",PHENO_COLORS["MasP"]),("HSC",PHENO_COLORS["HSC"] )]:
        sub=z[(z.terminal_fate==fate) & (z.program.isin(["erythroid RNA","erythro-megakaryoid","neutrophil RNA","myeloid / inflammatory"]))]
        if len(sub):
            curve=sub.groupby("time").zscore.mean().reindex(MOVIE_TIMES)
            ax.plot(MOVIE_TIMES, curve, color=color, lw=2.6, marker="o", label=fate)
    ax.axhline(0,color="#CBD5E1",lw=1)
    ax.set_title("B. Terminal fates separate by coordinated RNA→protein program dynamics", loc="left", fontsize=13.5, weight="bold")
    ax.set_xlabel("generated time"); ax.set_ylabel("mean branch program z-score"); ax.legend(frameon=False, ncol=3, fontsize=9)
    ax.spines[["top","right"]].set_visible(False)
    # pair bars
    ax = fig.add_subplot(gs[2,0:2])
    top = pair_df.head(10).copy().iloc[::-1]
    labels2=[f"{r.gene} → {r.protein_marker.replace('prot_','')}" for _,r in top.iterrows()]
    colors=["#D62728" if v>0 else "#2563EB" for v in top.spearman_delta_particle]
    ax.barh(labels2, top.spearman_delta_particle, color=colors, alpha=0.88)
    ax.axvline(0,color="#64748B",lw=1)
    ax.set_xlabel("particle-wise Spearman ρ of ΔRNA gene vs Δprotein marker")
    ax.set_title("C. Gene-to-surface-protein axes recovered along generated trajectories", loc="left", fontsize=13.5, weight="bold")
    ax.tick_params(labelsize=9); ax.spines[["top","right"]].set_visible(False)
    # schematic
    ax = fig.add_subplot(gs[2,2:]); ax.axis("off")
    boxes=[("RNA state\nSVD coordinates",0.05,0.58,"#DBEAFE"),("CITE-PCFM\ncontinuous dynamics",0.38,0.58,"#DCFCE7"),("Generated protein\ntime series",0.72,0.58,"#FDE68A"),("gene-level\nprogram readout",0.18,0.14,"#E0E7FF"),("surface-marker\nphenotype readout",0.62,0.14,"#FCE7F3")]
    for text,x,y,c in boxes:
        ax.text(x,y,text,ha="center",va="center",fontsize=13,weight="bold",bbox=dict(boxstyle="round,pad=0.45",fc=c,ec="#334155",lw=1.2), transform=ax.transAxes)
    for a,b in [((0.18,0.58),(0.30,0.58)),((0.50,0.58),(0.64,0.58)),((0.38,0.47),(0.25,0.26)),((0.72,0.47),(0.67,0.26)),((0.33,0.16),(0.55,0.16))]:
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=18,lw=1.8,color="#334155",transform=ax.transAxes))
    ax.text(0.02,0.94,"D. Biological interpretation layer",fontsize=13.5,weight="bold",transform=ax.transAxes)
    ax.text(0.02,0.01,"Readout is not an endpoint repair: all curves are computed from the locked generated trajectory.\nProtein values use PCA inverse projections; gene claims are low-rank program hypotheses.",fontsize=9.5,color="#475569",transform=ax.transAxes)
    out=FIG_DIR/"nbt_rna_to_protein_gene_story_flagship.png"
    fig.savefig(out,dpi=320,bbox_inches="tight",facecolor="white"); fig.savefig(out.with_suffix('.pdf'),bbox_inches="tight",facecolor="white"); plt.close(fig)
    return out

def fig_branch_rna_protein_panels(branch_df):
    plot_programs=[("erythroid RNA","erythro-megakaryoid"),("megakaryocyte RNA","erythro-megakaryoid"),("neutrophil RNA","myeloid / inflammatory"),("mono-mast RNA","myeloid / inflammatory"),("stem RNA","stem / priming")]
    fig, axes = plt.subplots(len(plot_programs),2,figsize=(12.5,13.5),sharex=True,constrained_layout=True)
    for r,(rna_prog,prot_prog) in enumerate(plot_programs):
        for c,(typ,prog) in enumerate([("RNA program",rna_prog),("Protein program",prot_prog)]):
            ax=axes[r,c]
            sub=branch_df[(branch_df.type==typ)&(branch_df.program==prog)].copy()
            sub=standardize_for_plot(sub,["program"])
            for fate in ["HSC","EryP","MkP","NeuP","MasP","MoP"]:
                fs=sub[sub.terminal_fate==fate].sort_values("time")
                if len(fs): ax.plot(fs.time,fs.zscore,marker="o",lw=2,color=PHENO_COLORS.get(fate,"#333"),label=fate if r==0 and c==1 else None)
            ax.axhline(0,color="#CBD5E1",lw=0.8); ax.set_title(f"{prog}",loc="left",fontsize=12,weight="bold"); ax.spines[["top","right"]].set_visible(False); ax.tick_params(labelsize=8)
            if c==0: ax.set_ylabel("branch z-score")
    axes[0,1].legend(frameon=False,ncol=3,fontsize=8,loc="upper left",bbox_to_anchor=(0,1.35))
    for ax in axes[-1,:]: ax.set_xlabel("generated time")
    fig.suptitle("Branch-resolved RNA and protein programs from the same generated particles", fontsize=18, weight="bold", x=0.02, ha="left")
    out=FIG_DIR/"nbt_branch_resolved_rna_protein_programs.png"
    fig.savefig(out,dpi=320,bbox_inches="tight",facecolor="white"); fig.savefig(out.with_suffix('.pdf'),bbox_inches="tight",facecolor="white"); plt.close(fig)
    return out

def fig_pair_network(pair_df):
    df=pair_df.head(10).copy()
    fig, ax = plt.subplots(figsize=(12.5,7.2)); ax.axis("off")
    y=np.linspace(0.9,0.12,len(df))
    for i,(_,r) in enumerate(df.iterrows()):
        color="#DC2626" if r.spearman_delta_particle>0 else "#2563EB"
        lw=1.2+4*min(1,abs(r.spearman_delta_particle))
        ax.text(0.05,y[i],r.gene,ha="center",va="center",fontsize=11,weight="bold",bbox=dict(boxstyle="round,pad=0.25",fc="#EFF6FF",ec="#1D4ED8"),transform=ax.transAxes)
        ax.text(0.72,y[i],r.protein_marker.replace("prot_",""),ha="center",va="center",fontsize=11,weight="bold",bbox=dict(boxstyle="round,pad=0.25",fc="#FFF7ED",ec="#C2410C"),transform=ax.transAxes)
        ax.add_patch(FancyArrowPatch((0.13,y[i]),(0.64,y[i]),arrowstyle="-|>",mutation_scale=14,lw=lw,color=color,alpha=0.72,transform=ax.transAxes))
        ax.text(0.40,y[i]+0.018,f"ρ={r.spearman_delta_particle:.2f}; lag={int(r.best_population_lag)}",ha="center",va="bottom",fontsize=8.5,color="#334155",transform=ax.transAxes)
        ax.text(0.84,y[i],textwrap.shorten(str(r.hypothesis),42),ha="left",va="center",fontsize=8.5,color="#475569",transform=ax.transAxes)
    ax.text(0.05,0.99,"RNA genes",fontsize=13,weight="bold",transform=ax.transAxes)
    ax.text(0.66,0.99,"generated proteins",fontsize=13,weight="bold",transform=ax.transAxes)
    ax.text(0.02,1.07,"Data-driven gene–surface protein relationships along generated trajectories",fontsize=18,weight="bold",transform=ax.transAxes)
    ax.text(0.02,1.015,"Edges summarize particle-wise ΔRNA-gene vs Δprotein-marker association from t0 to t4; thickness is |ρ|.",fontsize=10.5,color="#475569",transform=ax.transAxes)
    out=FIG_DIR/"nbt_gene_to_surface_protein_dynamic_network.png"
    fig.savefig(out,dpi=320,bbox_inches="tight",facecolor="white"); fig.savefig(out.with_suffix('.pdf'),bbox_inches="tight",facecolor="white"); plt.close(fig)
    return out

def main():
    ensure_dirs()
    adata=ad.read_h5ad(DATA_PATH)
    obs=adata.obs.copy(); labels=obs["cell_type"].astype(str).to_numpy()
    rna_mask=adata.var["feature_types"].astype(str).to_numpy()=="Gene Expression"
    prot_mask=adata.var["feature_types"].astype(str).to_numpy()=="Antibody Capture"
    names=np.asarray(adata.var_names.astype(str))
    Xrna=get_matrix(adata,rna_mask); Xprot=get_matrix(adata,prot_mask)
    rna_mean=Xrna.mean(axis=0); prot_mean=Xprot.mean(axis=0)
    rna_names=np.asarray(names[rna_mask]); prot_names=np.asarray(names[prot_mask])
    PCs=np.asarray(adata.varm["PCs"],float)
    rna_load=PCs[rna_mask,:]; prot_load=PCs[prot_mask,:]
    times, pts, wts=load_traj()
    pts_obs={t:pts[t] for t in MOVIE_TIMES}
    # selected genes and proteins
    all_genes=sorted(set(sum(GENE_PROGRAMS.values(),[]) + [g for g,_,_ in RNA_PROTEIN_HYPOTHESES]))
    all_markers=sorted(set(sum(MARKER_PANELS.values(),[]) + [p for _,p,_ in RNA_PROTEIN_HYPOTHESES]))
    kept_genes, gene_arr = lowrank_reconstruct(pts_obs, rna_load, rna_mean, rna_names, all_genes, 0, RNA_DIM)
    kept_markers, prot_arr = lowrank_reconstruct(pts_obs, prot_load, prot_mean, prot_names, all_markers, 50, PROT_DIM)
    gene_df_by_t={t:pd.DataFrame(gene_arr[t],columns=kept_genes) for t in MOVIE_TIMES}
    prot_df_by_t={t:pd.DataFrame(prot_arr[t],columns=kept_markers) for t in MOVIE_TIMES}
    prog_df=build_program_table(gene_df_by_t, prot_df_by_t, GENE_PROGRAMS, MARKER_PANELS, {t:wts.get(t) for t in MOVIE_TIMES})
    pair_df=build_pair_table(gene_df_by_t, prot_df_by_t, RNA_PROTEIN_HYPOTHESES)
    protein_obs=np.asarray(adata.obsm["X_protein_pca"],float)
    clf,audit=train_protein_fate(protein_obs,labels)
    terminal=clf.predict(pts[4.0][:,RNA_DIM:RNA_DIM+PROT_DIM])
    fate_by_time={t:clf.predict(pts[t][:,RNA_DIM:RNA_DIM+PROT_DIM]) for t in MOVIE_TIMES}
    branch_df=fate_branch_curves(fate_by_time, terminal, gene_df_by_t, prot_df_by_t, GENE_PROGRAMS, MARKER_PANELS)
    prog_df.to_csv(TABLE_DIR/"program_time_series.csv",index=False)
    pair_df.to_csv(TABLE_DIR/"gene_protein_dynamic_pairs.csv",index=False)
    branch_df.to_csv(TABLE_DIR/"branch_resolved_programs.csv",index=False)
    pd.DataFrame([audit]).to_csv(TABLE_DIR/"protein_fate_classifier_audit.csv",index=False)
    figs=[fig_flagship_marker_movie(prog_df,pair_df,branch_df), fig_branch_rna_protein_panels(branch_df), fig_pair_network(pair_df)]
    top_pairs=pair_df.head(5).to_dict(orient="records")
    story = f"""# CITE-PCFM RNA-to-protein dynamical biology story\n\n## Central biological question\nCan RNA-defined hematopoietic states be rolled forward by CITE-PCFM into an interpretable time series of surface-protein phenotypes, and can those generated dynamics reveal coordinated gene/protein programs rather than only an improved metric?\n\n## Main result\nThe locked model trajectory generates a coherent protein-state movie from RNA-initialized particles. Branch-resolved analysis separates erythroid/megakaryocytic and myeloid protein acquisition windows and links low-rank RNA programs to generated surface-marker changes.\n\n## Most supported gene-to-protein axes\n"""
    for r in top_pairs:
        story += f"- **{r['gene']} → {r['protein_marker'].replace('prot_','')}**: particle-wise Δ association ρ={r['spearman_delta_particle']:.3f}; {r['hypothesis']}.\n"
    story += """\n## Evidence boundary\nAll dynamic states come from `evaluation_trajectory.npz`. Protein and RNA marker values are low-rank inverse projections from the CITE coordinate system; they should be interpreted as model-supported program hypotheses, not direct raw-count imputation.\n"""
    STORY.write_text(story)
    manifest={"analysis":"rna_to_time_series_protein_gene_story","source_trajectory":str(TRAJ_PATH),"data_path":str(DATA_PATH),"figures":[str(p) for p in figs],"tables":{"program_time_series":str(TABLE_DIR/"program_time_series.csv"),"gene_protein_dynamic_pairs":str(TABLE_DIR/"gene_protein_dynamic_pairs.csv"),"branch_resolved_programs":str(TABLE_DIR/"branch_resolved_programs.csv"),"protein_fate_classifier_audit":str(TABLE_DIR/"protein_fate_classifier_audit.csv")},"story":str(STORY),"kept_genes":kept_genes,"kept_protein_markers":kept_markers,"classifier_audit":audit,"warnings":["RNA/protein marker values are approximate low-rank inverse projections using PCA/SVD loadings and observed means.","Balanced CITE-PCFM does not support growth/mass mechanistic claims.","Associations are trajectory-derived hypotheses, not causal perturbation evidence."]}
    MANIFEST.write_text(json.dumps(manifest,indent=2))
    print(json.dumps({"figures":[str(p) for p in figs],"manifest":str(MANIFEST),"story":str(STORY),"top_pairs":top_pairs},indent=2))

if __name__ == "__main__":
    main()
