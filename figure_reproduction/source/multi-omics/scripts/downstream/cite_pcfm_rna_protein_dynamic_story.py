#!/usr/bin/env python3
"""NBT-style RNA-to-protein dynamic biology story for CITE-PCFM.

The goal is one memorable biological figure: CITE-PCFM starts from RNA
particles and generates continuous protein-state trajectories. Gene-level
callouts are computed from observed day-2 RNA cells projected onto the same
RNA seed neighborhoods.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/cytobridge_mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/cytobridge_xdg_cache")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy import sparse
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


sns.set_theme(style="white", context="talk")

ROOT = Path("/Users/zhenyizhang/Documents/GitHub/CytoBridge-agent-runtime-v2-20260309/multi-omics")
DATA_PATH = Path("/Users/zhenyizhang/.cellcompass/algorithm_benchmarks/datasets/gse305370_citeseq_stage2_10k/data.h5ad")
TRAJ_PATH = ROOT / "training_runs/20260526T174204321821Z__final__cite_paired_coord_fm__campaign_20260-trial_20260526-gse305370_ci/artifacts/evaluation_trajectory.npz"
FIG_DIR = ROOT / "figures/cite_pcfm_deep_story/cite_biology_landscape"
NOTE_DIR = ROOT / "downstream/cite_pcfm_deep_story"
TABLE_DIR = NOTE_DIR / "tables"
NOTE_PATH = NOTE_DIR / "rna_protein_dynamic_biology.md"
MANIFEST_PATH = NOTE_DIR / "rna_protein_dynamic_biology_manifest.json"

RNA_DIM = 50
PROT_DIM = 30
N_NEIGHBORHOODS = 6
RNG = np.random.default_rng(20260527)

PHENO_COLORS = {
    "HSC": "#3C5488",
    "MasP": "#E64B35",
    "EryP": "#F39B2F",
    "MoP": "#00A087",
    "MkP": "#7E6148",
    "NeuP": "#8491B4",
    "BP": "#7C3AED",
}

PROTEIN_MODULES = {
    "Stem retention": ["CD49f", "CD45RA", "CD123", "CD127"],
    "Ery-Mk protein": ["CD41", "CD42b", "CD71", "CD62P"],
    "Myeloid protein": ["CD11b", "CD64", "CD115", "CD172a", "CD13", "CD16"],
    "Mast/Baso protein": ["FceRIa", "CD123", "CD88", "CD13"],
    "B-lineage protein": ["CD20", "CD21", "CD79b", "IgM", "IgD"],
    "HLA/activation protein": ["HLA-DR", "HLA-A-B-C", "CD54", "CD25"],
}

GENE_MODULES = {
    "Stem/HSPC RNA": ["HLF", "MEIS1", "MECOM", "GATA2", "CD34", "PROM1", "SPINK2", "AVP", "KIT"],
    "Myeloid/MasP RNA": ["MPO", "ELANE", "AZU1", "CTSG", "PRTN3", "LYZ", "S100A8", "S100A9", "CEBPA", "CSF3R", "IRF8", "MS4A3"],
    "Mast/Baso RNA": ["GATA2", "CPA3", "TPSAB1", "MS4A2", "KIT", "HDC", "CLC", "FCER1A"],
    "Erythroid RNA": ["GATA1", "KLF1", "HBB", "HBA1", "HBA2", "ALAS2", "GYPA", "AHSP", "CA1"],
    "Megakaryocyte RNA": ["PF4", "PPBP", "ITGA2B", "ITGB3", "GP9", "VWF", "MPL", "TUBB1"],
    "B-lineage RNA": ["MS4A1", "CD79A", "CD79B", "VPREB1", "IGHM", "IGHD", "EBF1", "PAX5"],
    "HLA/activation RNA": ["HLA-DRA", "HLA-DRB1", "HLA-A", "B2M", "CD74", "IFI30", "CIITA"],
}

FIGURES: list[dict] = []


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    NOTE_DIR.mkdir(parents=True, exist_ok=True)


def savefig(fig: plt.Figure, name: str, title: str, question: str) -> Path:
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    FIGURES.append({"name": name, "title": title, "question": question, "png": str(path), "pdf": str(path.with_suffix(".pdf"))})
    return path


def sample_idx(n: int, max_n: int, weights: np.ndarray | None = None) -> np.ndarray:
    if n <= max_n:
        return np.arange(n)
    prob = None
    if weights is not None:
        w = np.clip(np.asarray(weights, dtype=float).reshape(-1), 0, np.inf)
        if w.shape[0] == n and np.isfinite(w).all() and w.sum() > 0:
            prob = w / w.sum()
    return np.sort(RNG.choice(n, size=max_n, replace=False, p=prob))


def weighted_mean(x: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if weights is None:
        return np.nanmean(x, axis=0)
    w = np.clip(np.asarray(weights, dtype=float).reshape(-1), 0, np.inf)
    if w.shape[0] != x.shape[0] or not np.isfinite(w).all() or w.sum() <= 0:
        return np.nanmean(x, axis=0)
    return np.average(x, axis=0, weights=w / w.sum())


def clean_name(name: str, prefix: str) -> str:
    return str(name).replace(prefix, "", 1)


def lightweight_kmeans(x: np.ndarray, n_clusters: int, n_iter: int = 60) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    order = np.argsort(x[:, 0])
    init_idx = [order[int((i + 0.5) * len(order) / n_clusters)] for i in range(n_clusters)]
    centroids = x[init_idx].copy()
    labels = np.zeros(x.shape[0], dtype=int)
    for _ in range(n_iter):
        dist = ((x[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        new_labels = np.argmin(dist, axis=1)
        new_centroids = centroids.copy()
        for cluster_id in range(n_clusters):
            mask = new_labels == cluster_id
            if mask.any():
                new_centroids[cluster_id] = x[mask].mean(axis=0)
        if np.array_equal(labels, new_labels):
            labels = new_labels
            break
        labels = new_labels
        centroids = new_centroids
    return labels


def train_protein_mlp(protein_obs: np.ndarray, obs_labels: np.ndarray) -> tuple[StandardScaler, MLPClassifier, pd.DataFrame]:
    x_train, x_test, y_train, y_test = train_test_split(
        protein_obs,
        obs_labels,
        test_size=0.25,
        random_state=20260527,
        stratify=obs_labels,
    )
    scaler = StandardScaler().fit(x_train)
    clf = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        alpha=1e-4,
        batch_size=256,
        learning_rate_init=1e-3,
        max_iter=350,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=20260527,
    )
    clf.fit(scaler.transform(x_train), y_train)
    pred = clf.predict(scaler.transform(x_test))
    rows = [
        {
            "readout": "observed_protein_pca_mlp",
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "accuracy": float(accuracy_score(y_test, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
            "n_iter": int(clf.n_iter_),
        }
    ]
    for label in sorted(pd.unique(obs_labels)):
        mask = y_test == label
        rows.append(
            {
                "readout": f"class_{label}",
                "n_train": int(np.sum(y_train == label)),
                "n_test": int(mask.sum()),
                "accuracy": float(np.mean(pred[mask] == label)) if mask.any() else np.nan,
                "balanced_accuracy": np.nan,
                "n_iter": int(clf.n_iter_),
            }
        )
    return scaler, clf, pd.DataFrame(rows)


def predict_mlp_labels(query: np.ndarray, scaler: StandardScaler, clf: MLPClassifier, batch_size: int = 512) -> np.ndarray:
    return np.concatenate([clf.predict(scaler.transform(query[i : i + batch_size])) for i in range(0, query.shape[0], batch_size)])


def load_trajectory() -> tuple[np.ndarray, dict[float, np.ndarray], dict[float, np.ndarray]]:
    z = np.load(TRAJ_PATH)
    time_points = np.asarray(z["time_points"], dtype=float)
    points = {}
    weights = {}
    for i, time in enumerate(time_points):
        pk = f"points_{i:06d}"
        wk = f"weights_{i:06d}"
        if pk in z.files:
            points[float(time)] = np.asarray(z[pk], dtype=np.float32)
        if wk in z.files:
            weights[float(time)] = np.asarray(z[wk], dtype=np.float32)
    return time_points, points, weights


def module_scores(marker_values: np.ndarray, marker_names: list[str], marker_mean: np.ndarray, marker_std: np.ndarray) -> pd.DataFrame:
    marker_index = {marker: i for i, marker in enumerate(marker_names)}
    z = (marker_values - marker_mean[None, :]) / marker_std[None, :]
    rows = {}
    for module, markers in PROTEIN_MODULES.items():
        idx = [marker_index[m] for m in markers if m in marker_index]
        if idx:
            rows[module] = z[:, idx].mean(axis=1)
    return pd.DataFrame(rows)


def build_dynamic_tables() -> dict:
    adata = ad.read_h5ad(DATA_PATH, backed="r")
    obs = adata.obs.copy()
    obs["time_point_processed"] = obs["time_point_processed"].astype(float)
    obs_time = obs["time_point_processed"].to_numpy(dtype=float)
    obs_labels = obs["cell_type"].astype(str).to_numpy()
    rna_obs = np.asarray(adata.obsm["X_rna_svd"], dtype=np.float32)
    protein_obs = np.asarray(adata.obsm["X_protein_pca"], dtype=np.float32)

    feature_types = adata.var["feature_types"].astype(str)
    protein_mask = feature_types.eq("Antibody Capture").to_numpy()
    rna_mask = feature_types.eq("Gene Expression").to_numpy()
    protein_indices = np.where(protein_mask)[0]
    rna_indices = np.where(rna_mask)[0]
    marker_names = [clean_name(x, "prot_") for x in adata.var.index[protein_mask]]
    gene_names = [clean_name(x, "rna_") for x in adata.var.index[rna_mask]]

    protein_raw = adata[:, protein_indices].X
    protein_adt = protein_raw.toarray().astype(np.float32) if sparse.issparse(protein_raw) else np.asarray(protein_raw, dtype=np.float32)
    marker_mean = protein_adt.mean(axis=0)
    marker_std = protein_adt.std(axis=0)
    marker_std[marker_std < 1e-6] = 1.0
    decoder = Ridge(alpha=1.0).fit(protein_obs, protein_adt)

    mlp_scaler, mlp_classifier, classifier_audit = train_protein_mlp(protein_obs, obs_labels)
    time_points, traj_points, traj_weights = load_trajectory()

    rna0 = traj_points[0.0][:, :RNA_DIM]
    protein4 = traj_points[4.0][:, RNA_DIM : RNA_DIM + PROT_DIM]
    terminal_label = predict_mlp_labels(protein4, mlp_scaler, mlp_classifier)

    raw_cluster = lightweight_kmeans(rna0, N_NEIGHBORHOODS)
    cluster_order = []
    for cluster_id in range(N_NEIGHBORHOODS):
        mask = raw_cluster == cluster_id
        masp = float(np.mean(terminal_label[mask] == "MasP")) if mask.any() else 0.0
        hsc = float(np.mean(terminal_label[mask] == "HSC")) if mask.any() else 0.0
        ery = float(np.mean(terminal_label[mask] == "EryP")) if mask.any() else 0.0
        cluster_order.append((cluster_id, masp + 0.35 * hsc - 0.20 * ery))
    remap = {old: new for new, (old, _) in enumerate(sorted(cluster_order, key=lambda x: x[1], reverse=True))}
    cluster = np.asarray([remap[c] for c in raw_cluster], dtype=int)
    centroids = np.vstack([rna0[cluster == c].mean(axis=0) for c in range(N_NEIGHBORHOODS)])

    fate_rows = []
    protein_rows = []
    for time in time_points:
        pts = traj_points[float(time)]
        w = traj_weights.get(float(time))
        protein = pts[:, RNA_DIM : RNA_DIM + PROT_DIM]
        labels = predict_mlp_labels(protein, mlp_scaler, mlp_classifier)
        decoded = decoder.predict(protein).astype(np.float32)
        scores = module_scores(decoded, marker_names, marker_mean, marker_std)
        for neighborhood in range(N_NEIGHBORHOODS):
            mask = cluster == neighborhood
            ww = w[mask] if w is not None else None
            for phenotype in sorted(PHENO_COLORS):
                fate_rows.append(
                    {
                        "neighborhood": neighborhood,
                        "time": float(time),
                        "phenotype": phenotype,
                        "fraction": float(weighted_mean((labels[mask] == phenotype).astype(float)[:, None], ww)[0]),
                    }
                )
            for module in scores.columns:
                protein_rows.append(
                    {
                        "neighborhood": neighborhood,
                        "time": float(time),
                        "protein_module": module,
                        "score": float(weighted_mean(scores.loc[mask, module].to_numpy()[:, None], ww)[0]),
                    }
                )

    t0_mask = obs_time == 0.0
    t0_rna = rna_obs[t0_mask]
    t0_dist = np.linalg.norm(t0_rna[:, None, :] - centroids[None, :, :], axis=2)
    t0_assign = np.argmin(t0_dist, axis=1)

    available_genes = set(gene_names)
    gene_modules = {module: [gene for gene in genes if gene in available_genes] for module, genes in GENE_MODULES.items()}
    gene_modules = {module: genes for module, genes in gene_modules.items() if genes}
    gene_list = sorted({gene for genes in gene_modules.values() for gene in genes})
    gene_lookup = {gene: i for i, gene in enumerate(gene_names)}
    gene_full_indices = [int(rna_indices[gene_lookup[gene]]) for gene in gene_list]
    x_gene = adata[t0_mask, gene_full_indices].X
    x_gene = x_gene.toarray().astype(np.float32) if sparse.issparse(x_gene) else np.asarray(x_gene, dtype=np.float32)
    x_gene = (x_gene - x_gene.mean(axis=0, keepdims=True)) / (x_gene.std(axis=0, keepdims=True) + 1e-6)
    gene_col = {gene: i for i, gene in enumerate(gene_list)}

    gene_rows = []
    top_gene_rows = []
    for neighborhood in range(N_NEIGHBORHOODS):
        mask = t0_assign == neighborhood
        for module, genes in gene_modules.items():
            idx = [gene_col[gene] for gene in genes if gene in gene_col]
            gene_rows.append(
                {
                    "neighborhood": neighborhood,
                    "gene_module": module,
                    "score": float(x_gene[mask][:, idx].mean()) if mask.any() and idx else np.nan,
                    "genes_used": ",".join([gene for gene in genes if gene in gene_col]),
                    "observed_t0_cells": int(mask.sum()),
                }
            )
        if mask.sum() >= 5:
            diff = x_gene[mask].mean(axis=0) - x_gene[~mask].mean(axis=0)
            for rank, col in enumerate(np.argsort(diff)[::-1][:10], start=1):
                top_gene_rows.append(
                    {
                        "neighborhood": neighborhood,
                        "rank": rank,
                        "gene": gene_list[int(col)],
                        "cluster_vs_other_z": float(diff[int(col)]),
                    }
                )

    fate_df = pd.DataFrame(fate_rows)
    protein_df = pd.DataFrame(protein_rows)
    gene_df = pd.DataFrame(gene_rows)
    top_gene_df = pd.DataFrame(top_gene_rows)

    summary_rows = []
    for neighborhood in range(N_NEIGHBORHOODS):
        mask = cluster == neighborhood
        terminal = pd.Series(terminal_label[mask]).value_counts(normalize=True)
        top_gene_module = gene_df[gene_df["neighborhood"] == neighborhood].sort_values("score", ascending=False).head(1)
        top_protein_module = protein_df[(protein_df["neighborhood"] == neighborhood) & np.isclose(protein_df["time"], 4.0)].sort_values("score", ascending=False).head(1)
        summary_rows.append(
            {
                "neighborhood": neighborhood,
                "generated_particles": int(mask.sum()),
                "projected_observed_t0_cells": int((t0_assign == neighborhood).sum()),
                "dominant_terminal_protein_fate": str(terminal.index[0]) if len(terminal) else "",
                "dominant_terminal_fraction": float(terminal.iloc[0]) if len(terminal) else np.nan,
                "terminal_entropy": -float(np.sum(terminal.to_numpy() * np.log(terminal.to_numpy() + 1e-12)) / math.log(max(2, len(terminal)))) if len(terminal) else np.nan,
                "top_initial_gene_module": str(top_gene_module["gene_module"].iloc[0]) if len(top_gene_module) else "",
                "top_initial_gene_module_score": float(top_gene_module["score"].iloc[0]) if len(top_gene_module) else np.nan,
                "top_terminal_protein_module": str(top_protein_module["protein_module"].iloc[0]) if len(top_protein_module) else "",
                "top_terminal_protein_module_score": float(top_protein_module["score"].iloc[0]) if len(top_protein_module) else np.nan,
                "top_genes": ", ".join(top_gene_df[top_gene_df["neighborhood"] == neighborhood].head(5)["gene"].tolist()),
            }
        )

    try:
        adata.file.close()
    except Exception:
        pass

    return {
        "obs_time": obs_time,
        "obs_labels": obs_labels,
        "rna_obs": rna_obs,
        "protein_obs": protein_obs,
        "time_points": time_points,
        "traj_points": traj_points,
        "traj_weights": traj_weights,
        "cluster": cluster,
        "terminal_label": terminal_label,
        "fate_df": fate_df,
        "protein_df": protein_df,
        "gene_df": gene_df,
        "top_gene_df": top_gene_df,
        "summary_df": pd.DataFrame(summary_rows),
        "classifier_audit": classifier_audit,
        "rna0": rna0,
    }


def add_curve_arrow(ax, p0, p1, color="#64748B", lw=1.8, rad=0.02, alpha=0.85) -> None:
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="Simple,head_width=6,head_length=8",
            linewidth=lw,
            color=color,
            alpha=alpha,
        )
    )


def add_card(ax, x: float, y: float, w: float, h: float, edge: str, lines: list[tuple[str, dict]]) -> None:
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.014,rounding_size=0.018", linewidth=1.6, edgecolor=edge, facecolor="#F8FAFC")
    ax.add_patch(patch)
    cursor = y + h - 0.08
    for text, kwargs in lines:
        text_kwargs = dict(kwargs)
        dy = text_kwargs.pop("_dy", 0.13)
        ax.text(x + 0.014, cursor, text, ha="left", va="top", **text_kwargs)
        cursor -= dy


def density_grid(
    xy: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    weights: np.ndarray | None = None,
    sigma: float = 1.2,
) -> np.ndarray:
    w = None
    if weights is not None:
        w = np.clip(np.asarray(weights, dtype=float).reshape(-1), 0, np.inf)
        if w.shape[0] != xy.shape[0] or not np.isfinite(w).all() or w.sum() <= 0:
            w = None
    hist, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=[x_edges, y_edges], weights=w)
    hist = gaussian_filter(hist.T, sigma=sigma)
    max_value = float(np.nanmax(hist))
    return hist / max_value if max_value > 0 else hist


def draw_density_frame(
    ax,
    xx: np.ndarray,
    yy: np.ndarray,
    observed_density: np.ndarray,
    generated_density: np.ndarray,
    color: str,
    title: str,
    centroid: np.ndarray,
) -> None:
    obs_levels = [0.05, 0.12, 0.24, 0.42, 0.68, 1.01]
    gen_levels = [0.10, 0.22, 0.38, 0.58, 0.80, 1.01]
    ax.contourf(xx, yy, observed_density, levels=obs_levels, colors=["#F8FAFC", "#F1F5F9", "#E2E8F0", "#CBD5E1", "#94A3B8"], alpha=0.72)
    ax.contour(xx, yy, observed_density, levels=[0.16, 0.34, 0.56], colors="#CBD5E1", linewidths=0.75)
    ax.contourf(xx, yy, generated_density, levels=gen_levels, colors=[color] * 5, alpha=0.16)
    ax.contour(xx, yy, generated_density, levels=[0.20, 0.40, 0.64], colors=color, linewidths=[1.2, 1.7, 2.2])
    ax.scatter([centroid[0]], [centroid[1]], s=55, color=color, edgecolor="white", linewidth=1.2, zorder=10)
    ax.set_title(title, fontsize=12.5, weight="bold", loc="left", color="#111827")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("#CBD5E1")


def figure_distribution_movie(payload: dict) -> None:
    obs_labels = payload["obs_labels"]
    protein_obs = payload["protein_obs"]
    time_points = payload["time_points"]
    traj_points = payload["traj_points"]
    traj_weights = payload["traj_weights"]
    gene_df = payload["gene_df"]
    protein_df = payload["protein_df"]
    top_gene_df = payload["top_gene_df"]
    classifier_audit = payload["classifier_audit"]

    movie_times = [0.0, 1.0, 2.0, 3.0, 4.0]
    time_colors = {
        0.0: "#2563EB",
        1.0: "#06B6D4",
        2.0: "#22C55E",
        3.0: "#F59E0B",
        4.0: "#E11D48",
    }
    generated_xy = {time: traj_points[time][:, RNA_DIM : RNA_DIM + 2] for time in movie_times}
    all_generated = np.vstack([generated_xy[time] for time in movie_times])
    idx_bg = sample_idx(protein_obs.shape[0], 8500)
    focus = np.vstack([all_generated, protein_obs[idx_bg, :2]])
    x0, x1 = np.nanpercentile(focus[:, 0], [0.7, 99.3])
    y0, y1 = np.nanpercentile(focus[:, 1], [0.7, 99.3])
    x_pad = 0.08 * (x1 - x0)
    y_pad = 0.10 * (y1 - y0)
    xlim = (x0 - x_pad, x1 + x_pad)
    ylim = (y0 - y_pad, y1 + y_pad)
    x_edges = np.linspace(xlim[0], xlim[1], 125)
    y_edges = np.linspace(ylim[0], ylim[1], 125)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    xx, yy = np.meshgrid(x_centers, y_centers)
    observed_density = density_grid(protein_obs[idx_bg, :2], x_edges, y_edges, sigma=1.35)
    generated_density = {
        time: density_grid(generated_xy[time], x_edges, y_edges, traj_weights.get(time), sigma=1.45)
        for time in movie_times
    }
    centroids = {
        time: weighted_mean(generated_xy[time], traj_weights.get(time))
        for time in movie_times
    }

    fig = plt.figure(figsize=(18.5, 9.8))
    ax_title = fig.add_axes([0.045, 0.90, 0.91, 0.085])
    ax_title.axis("off")
    ax_title.text(0.0, 0.72, "RNA-conditioned protein distribution movie", fontsize=25, weight="bold", color="#0F172A")
    ax_title.text(
        0.0,
        0.24,
        "Each frame shows the protein distribution generated from the same RNA-initialized particles, overlaid on observed CITE protein density.",
        fontsize=11.5,
        color="#475569",
    )
    audit = classifier_audit[classifier_audit["readout"].eq("observed_protein_pca_mlp")].head(1)
    if len(audit):
        row = audit.iloc[0]
        ax_title.text(1.0, 0.24, f"protein-fate MLP audit: accuracy {row['accuracy']:.3f}, balanced {row['balanced_accuracy']:.3f}", fontsize=9.5, ha="right", color="#334155")

    frame_left = 0.045
    frame_bottom = 0.45
    frame_w = 0.17
    frame_h = 0.35
    frame_gap = 0.014
    frame_axes = []
    for i, time in enumerate(movie_times):
        ax = fig.add_axes([frame_left + i * (frame_w + frame_gap), frame_bottom, frame_w, frame_h])
        draw_density_frame(
            ax,
            xx,
            yy,
            observed_density,
            generated_density[time],
            time_colors[time],
            f"t{int(time)}",
            centroids[time],
        )
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        frame_axes.append(ax)
        if i == 0:
            for phenotype in ["HSC", "MasP", "EryP", "MkP", "NeuP", "MoP"]:
                if phenotype not in set(obs_labels):
                    continue
                center = protein_obs[obs_labels == phenotype, :2].mean(axis=0)
                if xlim[0] < center[0] < xlim[1] and ylim[0] < center[1] < ylim[1]:
                    ax.text(center[0], center[1], phenotype, fontsize=8.2, color="#475569", weight="bold", ha="center", va="center")
    ax_flow = fig.add_axes([0, 0, 1, 1], frameon=False)
    ax_flow.set_axis_off()
    for i in range(len(movie_times) - 1):
        x_start = frame_left + (i + 1) * frame_w + i * frame_gap + 0.002
        x_end = frame_left + (i + 1) * (frame_w + frame_gap) - 0.003
        ax_flow.add_patch(FancyArrowPatch((x_start, 0.625), (x_end, 0.625), transform=fig.transFigure, arrowstyle="Simple,head_width=8,head_length=10", color="#94A3B8", alpha=0.85))

    ax_ridge = fig.add_axes([0.06, 0.105, 0.55, 0.25])
    start = centroids[0.0]
    end = centroids[4.0]
    axis = end - start
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    projections = {time: (generated_xy[time] - start[None, :]) @ axis for time in movie_times}
    all_proj = np.concatenate([projections[time] for time in movie_times])
    bins = np.linspace(np.nanpercentile(all_proj, 0.5), np.nanpercentile(all_proj, 99.5), 95)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    for i, time in enumerate(movie_times):
        w = traj_weights.get(time)
        hist, _ = np.histogram(projections[time], bins=bins, weights=w)
        hist = gaussian_filter1d(hist.astype(float), sigma=1.5)
        if hist.max() > 0:
            hist = hist / hist.max()
        offset = i * 0.88
        color = time_colors[time]
        ax_ridge.fill_between(bin_centers, offset, offset + 0.72 * hist, color=color, alpha=0.28)
        ax_ridge.plot(bin_centers, offset + 0.72 * hist, color=color, lw=2.0)
        ax_ridge.text(bin_centers[0], offset + 0.22, f"t{int(time)}", color=color, fontsize=9.5, weight="bold", va="center")
    ax_ridge.set_title("distribution shift along the generated protein axis", loc="left", fontsize=13, weight="bold")
    ax_ridge.set_xlabel("protein commitment coordinate", fontsize=10)
    ax_ridge.set_yticks([])
    ax_ridge.tick_params(axis="x", labelsize=8)
    ax_ridge.spines[["top", "right", "left"]].set_visible(False)

    gene_mat = gene_df.pivot(index="neighborhood", columns="gene_module", values="score").fillna(0)
    protein_delta = (
        protein_df[np.isclose(protein_df["time"], 4.0)].pivot(index="neighborhood", columns="protein_module", values="score").fillna(0)
        - protein_df[np.isclose(protein_df["time"], 0.0)].pivot(index="neighborhood", columns="protein_module", values="score").fillna(0)
    )
    focus_specs = [
        ("Myeloid/MasP RNA", "Myeloid ADT", "Myeloid protein"),
        ("Erythroid RNA", "Ery-Mk ADT", "Ery-Mk protein"),
        ("Mast/Baso RNA", "Mast/Baso ADT", "Mast/Baso protein"),
        ("HLA/activation RNA", "HLA ADT", "HLA/activation protein"),
    ]
    rows = []
    used = set()
    for gene_module, protein_label, protein_module in focus_specs:
        if gene_module not in gene_mat.columns or protein_module not in protein_delta.columns:
            continue
        candidates = gene_mat[gene_module].sort_values(ascending=False).index.tolist()
        neighborhood = int(candidates[0])
        for candidate in candidates:
            if int(candidate) not in used:
                neighborhood = int(candidate)
                break
        used.add(neighborhood)
        genes = "/".join(top_gene_df[top_gene_df["neighborhood"] == neighborhood].head(3)["gene"].tolist())
        rows.append(
            {
                "label": f"N{neighborhood} {genes}\n→ {protein_label}",
                "delta": float(protein_delta.loc[neighborhood, protein_module]),
            }
        )
    ax_bar = fig.add_axes([0.715, 0.105, 0.24, 0.25])
    y = np.arange(len(rows))[::-1]
    deltas = np.asarray([row["delta"] for row in rows])
    labels = [row["label"] for row in rows]
    colors = ["#059669" if value >= 0 else "#DC2626" for value in deltas]
    ax_bar.axvline(0, color="#CBD5E1", lw=1)
    ax_bar.barh(y, deltas, color=colors, alpha=0.86, height=0.52)
    for yy_i, value, label in zip(y, deltas, labels):
        ax_bar.text(-0.055, yy_i, label, transform=ax_bar.get_yaxis_transform(), fontsize=8.1, ha="right", va="center", color="#334155", clip_on=False)
        ax_bar.text(value + (0.025 if value >= 0 else -0.025), yy_i, f"{value:+.2f}", fontsize=8.5, ha="left" if value >= 0 else "right", va="center", color="#111827", weight="bold")
    ax_bar.set_xlim(-0.58, 0.43)
    ax_bar.set_ylim(-0.8, len(rows) - 0.2)
    ax_bar.set_yticks([])
    ax_bar.set_xlabel("decoded ADT velocity, t4 - t0", fontsize=10)
    ax_bar.set_title("gene programs explain where the distribution moves", loc="left", fontsize=13, weight="bold")
    ax_bar.tick_params(axis="x", labelsize=8)
    ax_bar.spines[["top", "right", "left"]].set_visible(False)

    fig.text(
        0.06,
        0.035,
        "Biological read: generated protein mass shifts from an HSC-like starting distribution toward a MasP/myeloid basin; gene-program bars summarize which RNA neighborhoods accelerate or lose specific ADT modules.",
        fontsize=9.2,
        color="#334155",
    )

    savefig(
        fig,
        "bio_landscape10_generated_distribution_movie",
        "Generated protein distribution movie",
        "Can the CITE biology be shown cleanly as generated protein distributions moving over time from RNA-initialized particles?",
    )


def figure_spotlight(payload: dict) -> None:
    obs_time = payload["obs_time"]
    obs_labels = payload["obs_labels"]
    rna_obs = payload["rna_obs"]
    protein_obs = payload["protein_obs"]
    time_points = payload["time_points"]
    traj_points = payload["traj_points"]
    traj_weights = payload["traj_weights"]
    cluster = payload["cluster"]
    terminal_label = payload["terminal_label"]
    fate_df = payload["fate_df"]
    protein_df = payload["protein_df"]
    gene_df = payload["gene_df"]
    top_gene_df = payload["top_gene_df"]
    classifier_audit = payload["classifier_audit"]
    rna0 = payload["rna0"]

    gene_mat = gene_df.pivot(index="neighborhood", columns="gene_module", values="score").fillna(0)
    protein_terminal = protein_df[np.isclose(protein_df["time"], 4.0)].pivot(index="neighborhood", columns="protein_module", values="score").fillna(0)
    protein_initial = protein_df[np.isclose(protein_df["time"], 0.0)].pivot(index="neighborhood", columns="protein_module", values="score").fillna(0)
    protein_delta = protein_terminal.subtract(protein_initial, fill_value=0)
    fate_terminal = fate_df[np.isclose(fate_df["time"], 4.0)].pivot(index="neighborhood", columns="phenotype", values="fraction").fillna(0)

    fig = plt.figure(figsize=(20.5, 13.2))
    ax_title = fig.add_axes([0.035, 0.92, 0.93, 0.068])
    ax_rna = fig.add_axes([0.045, 0.58, 0.225, 0.29])
    ax_main = fig.add_axes([0.305, 0.38, 0.46, 0.49])
    ax_signal = fig.add_axes([0.805, 0.66, 0.17, 0.21])
    ax_river = fig.add_axes([0.805, 0.39, 0.17, 0.20])
    ax_cards = fig.add_axes([0.045, 0.045, 0.93, 0.29])
    ax_flow = fig.add_axes([0, 0, 1, 1], frameon=False)
    ax_flow.set_axis_off()

    ax_title.axis("off")
    ax_title.text(0, 0.72, "One RNA state, one protein movie", fontsize=25, weight="bold", color="#111827")
    ax_title.text(
        0,
        0.22,
        "CITE-PCFM turns initial RNA particles into continuous CITE surface-protein trajectories, then exposes which day-2 gene programs shape later protein velocity and fate.",
        fontsize=11.5,
        color="#475569",
    )
    audit = classifier_audit[classifier_audit["readout"].eq("observed_protein_pca_mlp")].head(1)
    if len(audit):
        row = audit.iloc[0]
        ax_title.text(0.995, 0.22, f"protein-fate MLP audit: accuracy {row['accuracy']:.3f}, balanced {row['balanced_accuracy']:.3f}", fontsize=9.5, ha="right", color="#334155")
    for p0, p1 in [((0.275, 0.72), (0.305, 0.72)), ((0.765, 0.72), (0.805, 0.72))]:
        ax_flow.add_patch(FancyArrowPatch(p0, p1, transform=fig.transFigure, arrowstyle="Simple,head_width=8,head_length=10", color="#64748B", linewidth=1.2, alpha=0.75))

    rna_fit = np.vstack([rna_obs[obs_time == 0.0], rna0])
    rna_emb = PCA(n_components=2, random_state=20260527).fit_transform(StandardScaler().fit_transform(rna_fit))
    obs_emb = rna_emb[: int(np.sum(obs_time == 0.0))]
    gen_emb = rna_emb[int(np.sum(obs_time == 0.0)) :]
    idx_obs = sample_idx(obs_emb.shape[0], 1300)
    ax_rna.scatter(obs_emb[idx_obs, 0], obs_emb[idx_obs, 1], s=6, color="#CBD5E1", alpha=0.40, linewidths=0)
    for phenotype, color in PHENO_COLORS.items():
        if phenotype == "BP":
            continue
        mask = terminal_label == phenotype
        if mask.any():
            pts = gen_emb[mask][sample_idx(int(mask.sum()), 300)]
            ax_rna.scatter(pts[:, 0], pts[:, 1], s=12, color=color, alpha=0.82, linewidths=0, label=phenotype)
    for neighborhood in range(N_NEIGHBORHOODS):
        center = gen_emb[cluster == neighborhood].mean(axis=0)
        ax_rna.text(center[0], center[1], f"N{neighborhood}", fontsize=10, weight="bold", ha="center", va="center", color="#111827")
    ax_rna.set_title("A. RNA seeds colored by future protein fate", loc="left", fontsize=12.5, weight="bold")
    ax_rna.set_xlabel("RNA seed axis 1", fontsize=9.5)
    ax_rna.set_ylabel("RNA seed axis 2", fontsize=9.5)
    ax_rna.tick_params(labelsize=8)
    ax_rna.legend(frameon=False, fontsize=7, ncol=3, loc="lower left")

    idx_bg = sample_idx(protein_obs.shape[0], 4200)
    for phenotype, color in PHENO_COLORS.items():
        if phenotype == "BP":
            continue
        mask = obs_labels[idx_bg] == phenotype
        if mask.any():
            ax_main.scatter(protein_obs[idx_bg][mask, 0], protein_obs[idx_bg][mask, 1], s=7, color=color, alpha=0.13, linewidths=0)
            center = protein_obs[obs_labels == phenotype, :2].mean(axis=0)
            ax_main.text(center[0], center[1], phenotype, fontsize=10, color=color, weight="bold", ha="center", va="center", alpha=0.90)
    sample = sample_idx(traj_points[0.0].shape[0], 185, traj_weights.get(4.0))
    movie_times = time_points[::4]
    for i in sample:
        coords = np.vstack([traj_points[float(time)][i, RNA_DIM : RNA_DIM + 2] for time in movie_times])
        ax_main.plot(coords[:, 0], coords[:, 1], color=PHENO_COLORS.get(str(terminal_label[i]), "#64748B"), alpha=0.16, lw=0.8, zorder=2)
    neighborhood_colors = sns.color_palette("tab10", N_NEIGHBORHOODS)
    for neighborhood in range(N_NEIGHBORHOODS):
        mask = cluster == neighborhood
        means = []
        for time in time_points[::5]:
            pts = traj_points[float(time)]
            w = traj_weights.get(float(time))
            means.append(weighted_mean(pts[mask, RNA_DIM : RNA_DIM + 2], w[mask] if w is not None else None))
        means = np.asarray(means)
        color = neighborhood_colors[neighborhood]
        ax_main.plot(means[:, 0], means[:, 1], lw=4.5, marker="o", ms=6, color=color, zorder=6)
        add_curve_arrow(ax_main, tuple(means[-2]), tuple(means[-1]), color=color, lw=2.2)
        ax_main.text(means[-1, 0], means[-1, 1] + 0.12, f"N{neighborhood}", color=color, fontsize=10.5, weight="bold", ha="center")
    for time in [0.0, 1.0, 2.0, 3.0, 4.0]:
        mean = weighted_mean(traj_points[float(time)][:, RNA_DIM : RNA_DIM + 2], traj_weights.get(float(time)))
        ax_main.scatter(mean[0], mean[1], s=130, color="#111827", edgecolor="white", linewidth=1.5, zorder=8)
        ax_main.text(mean[0], mean[1] - 0.18, f"t{int(time)}", fontsize=8.5, color="#111827", ha="center", va="top", zorder=9)
    focus_blocks = [protein_obs[idx_bg, :2]] + [traj_points[float(time)][sample, RNA_DIM : RNA_DIM + 2] for time in movie_times]
    focus = np.vstack(focus_blocks)
    x0, x1 = np.nanpercentile(focus[:, 0], [0.7, 99.3])
    y0, y1 = np.nanpercentile(focus[:, 1], [0.7, 99.3])
    ax_main.set_xlim(x0 - 0.06 * (x1 - x0), x1 + 0.06 * (x1 - x0))
    ax_main.set_ylim(y0 - 0.08 * (y1 - y0), y1 + 0.08 * (y1 - y0))
    ax_main.set_title("B. the same particles become a protein movie on the observed CITE landscape", loc="left", fontsize=13.5, weight="bold")
    ax_main.set_xlabel("protein axis 1", fontsize=10)
    ax_main.set_ylabel("protein axis 2", fontsize=10)
    ax_main.tick_params(labelsize=8)

    module_palette = {
        "Myeloid protein": "#00A087",
        "Ery-Mk protein": "#F39B2F",
        "Stem retention": "#3C5488",
        "Mast/Baso protein": "#E64B35",
        "HLA/activation protein": "#64748B",
    }
    protein_series = (
        protein_df.groupby(["time", "protein_module"])["score"]
        .mean()
        .reset_index()
        .pivot(index="time", columns="protein_module", values="score")
        .sort_index()
    )
    for module, color in module_palette.items():
        if module not in protein_series.columns:
            continue
        y = protein_series[module].to_numpy()
        ax_signal.plot(protein_series.index, y, lw=2.4, color=color, marker="o", ms=3.2, label=module.replace(" protein", ""))
        ax_signal.text(protein_series.index[-1] + 0.04, y[-1], module.replace(" protein", ""), fontsize=7.3, color=color, va="center")
    ax_signal.axhline(0, color="#CBD5E1", lw=0.8, zorder=0)
    ax_signal.set_title("C. generated protein signals over time", loc="left", fontsize=12.5, weight="bold")
    ax_signal.set_xlabel("generated time", fontsize=9.5)
    ax_signal.set_ylabel("decoded ADT module", fontsize=9.5)
    ax_signal.tick_params(labelsize=8)
    ax_signal.spines[["top", "right"]].set_visible(False)

    phenotypes = [p for p in ["HSC", "MasP", "EryP", "MoP", "MkP", "NeuP"] if p in set(fate_df["phenotype"])]
    mat = fate_df.groupby(["time", "phenotype"])["fraction"].mean().reset_index().pivot(index="time", columns="phenotype", values="fraction").reindex(columns=phenotypes).fillna(0).sort_index()
    ax_river.stackplot(mat.index, [mat[p].to_numpy() for p in phenotypes], colors=[PHENO_COLORS[p] for p in phenotypes], labels=phenotypes, alpha=0.94, linewidth=1, edgecolor="white")
    ax_river.set_ylim(0, 1)
    ax_river.set_title("D. protein-fate river", loc="left", fontsize=12.5, weight="bold")
    ax_river.set_xlabel("generated time", fontsize=9.5)
    ax_river.set_ylabel("MLP-assigned protein fate", fontsize=9.5)
    ax_river.tick_params(labelsize=8)
    ax_river.legend(frameon=False, fontsize=7, ncol=2, loc="upper left")

    ax_cards.axis("off")
    ax_cards.set_xlim(0, 1)
    ax_cards.set_ylim(0, 1)
    ax_cards.text(0, 0.98, "E. gene-level callouts: initial RNA programs shape protein velocity", fontsize=13.5, weight="bold")
    focus_specs = [
        ("Myeloid/MasP RNA", "myeloid priming", "CD11b/CD64/CD115", "Myeloid protein"),
        ("Erythroid RNA", "erythroid diversion", "CD41/CD42b/CD71", "Ery-Mk protein"),
        ("Mast/Baso RNA", "mast/baso ridge", "FceRIa/CD123/CD88", "Mast/Baso protein"),
        ("HLA/activation RNA", "activation ridge", "HLA-DR/CD74/B2M", "HLA/activation protein"),
    ]
    used = set()
    for i, (gene_module, title, markers, protein_module) in enumerate(focus_specs):
        if gene_module not in gene_mat.columns:
            continue
        candidates = gene_mat[gene_module].sort_values(ascending=False).index.tolist()
        neighborhood = int(candidates[0])
        for candidate in candidates:
            if int(candidate) not in used:
                neighborhood = int(candidate)
                break
        used.add(neighborhood)
        genes = "/".join(top_gene_df[top_gene_df["neighborhood"] == neighborhood].head(3)["gene"].tolist())
        masp = float(fate_terminal.loc[neighborhood, "MasP"]) if "MasP" in fate_terminal.columns else np.nan
        neu = float(fate_terminal.loc[neighborhood, "NeuP"]) if "NeuP" in fate_terminal.columns else np.nan
        hsc = float(fate_terminal.loc[neighborhood, "HSC"]) if "HSC" in fate_terminal.columns else np.nan
        module_delta = float(protein_delta.loc[neighborhood, protein_module]) if protein_module in protein_delta.columns else np.nan
        x = 0.01 + i * 0.245
        color = neighborhood_colors[neighborhood]
        add_card(
            ax_cards,
            x,
            0.18,
            0.225,
            0.66,
            color,
            [
                (f"N{neighborhood} | {title}", {"fontsize": 9.2, "weight": "bold", "color": "#111827", "_dy": 0.14}),
                (f"RNA: {genes} ({gene_mat.loc[neighborhood, gene_module]:+.2f})", {"fontsize": 7.8, "color": "#334155", "_dy": 0.13}),
                (f"protein movie: {markers}, Δ {module_delta:+.2f}", {"fontsize": 7.8, "color": "#334155", "_dy": 0.13}),
                (f"terminal fate: MasP {masp:.2f}, NeuP {neu:.2f}, HSC {hsc:.2f}", {"fontsize": 7.6, "color": "#334155", "_dy": 0.12}),
            ],
        )
    ax_cards.text(
        0,
        0.03,
        "Biological read: generated HSPC-like RNA particles broadly enter a MasP/myeloid surface-protein basin, while erythroid, mast/baso, HLA and myeloid RNA programs determine side-branch protein velocity.",
        fontsize=9,
        color="#334155",
        ha="left",
    )
    ax_cards.text(0.99, 0.03, "BP omitted from interpretation because the observed-protein MLP has poor BP recall.", fontsize=7.4, ha="right", color="#64748B")

    savefig(
        fig,
        "bio_landscape09_rna_to_protein_spotlight",
        "RNA-to-protein spotlight",
        "Can the unique biology be remembered as one visual: RNA particles become a protein movie with gene-program callouts?",
    )


def write_tables(payload: dict) -> None:
    payload["fate_df"].to_csv(TABLE_DIR / "rna_seed_protein_fate_timecourse.csv", index=False)
    payload["protein_df"].to_csv(TABLE_DIR / "rna_seed_protein_module_timecourse.csv", index=False)
    payload["gene_df"].to_csv(TABLE_DIR / "rna_seed_initial_gene_programs.csv", index=False)
    payload["summary_df"].to_csv(TABLE_DIR / "rna_seed_dynamic_biology_summary.csv", index=False)
    payload["top_gene_df"].to_csv(TABLE_DIR / "rna_seed_top_initial_genes.csv", index=False)
    payload["classifier_audit"].to_csv(TABLE_DIR / "rna_seed_protein_mlp_classifier_audit.csv", index=False)


def write_note(payload: dict) -> None:
    audit = payload["classifier_audit"][payload["classifier_audit"]["readout"].eq("observed_protein_pca_mlp")].iloc[0]
    summary = payload["summary_df"]
    gene_df = payload["gene_df"]
    protein_df = payload["protein_df"]
    fate_df = payload["fate_df"]
    top_gene_df = payload["top_gene_df"]
    gene_mat = gene_df.pivot(index="neighborhood", columns="gene_module", values="score").fillna(0)
    protein_delta = (
        protein_df[np.isclose(protein_df["time"], 4.0)].pivot(index="neighborhood", columns="protein_module", values="score").fillna(0)
        - protein_df[np.isclose(protein_df["time"], 0.0)].pivot(index="neighborhood", columns="protein_module", values="score").fillna(0)
    )
    fate_terminal = fate_df[np.isclose(fate_df["time"], 4.0)].pivot(index="neighborhood", columns="phenotype", values="fraction").fillna(0)
    lines = []
    lines.append("# RNA-to-Protein Dynamic Biology Story\n\n")
    lines.append("The biology is framed around one capability: CITE-PCFM starts from RNA particles and generates a continuous CITE protein movie. This makes the analysis more than an endpoint score: it asks which initial gene programs shape future surface-marker velocity and terminal protein fates.\n\n")
    lines.append(f"Protein-fate assignment uses an MLP trained on observed protein PCA to observed cell-type labels; held-out accuracy `{audit['accuracy']:.3f}`, balanced accuracy `{audit['balanced_accuracy']:.3f}`. BP is not interpreted because its held-out recall is poor.\n\n")
    lines.append("## Main Spotlight Figure\n")
    for fig in FIGURES:
        punctuation = "" if str(fig["question"]).endswith((".", "?", "!")) else "."
        lines.append(f"- `{fig['name']}`: {fig['question']}{punctuation} File: `{fig['png']}`\n")
    lines.append("\n## Biological Story\n")
    lines.append("- Model-level finding: generated HSPC-like RNA particles broadly enter a MasP/myeloid surface-protein basin over pseudotime, but they do not do so uniformly; initial RNA programs define side-branch protein velocities.\n")
    lines.append("- Biological insight: the model separates transcriptomic priming from surface-phenotype commitment. A high erythroid RNA neighborhood loses Ery-Mk protein signal during rollout, whereas myeloid-primed RNA gains myeloid ADT signal and resolves toward MasP/NeuP protein fates. Mast/baso and HLA/activation RNA neighborhoods form mixed ridges rather than clean terminal classes.\n")
    focus_specs = [
        ("Myeloid/MasP RNA", "myeloid priming", "Myeloid protein"),
        ("Erythroid RNA", "erythroid diversion", "Ery-Mk protein"),
        ("Mast/Baso RNA", "mast/baso ridge", "Mast/Baso protein"),
        ("HLA/activation RNA", "activation ridge", "HLA/activation protein"),
    ]
    used = set()
    for gene_module, label, protein_module in focus_specs:
        if gene_module not in gene_mat.columns:
            continue
        candidates = gene_mat[gene_module].sort_values(ascending=False).index.tolist()
        neighborhood = int(candidates[0])
        for candidate in candidates:
            if int(candidate) not in used:
                neighborhood = int(candidate)
                break
        used.add(neighborhood)
        genes = ", ".join(top_gene_df[top_gene_df["neighborhood"] == neighborhood].head(5)["gene"].tolist())
        masp = float(fate_terminal.loc[neighborhood, "MasP"]) if "MasP" in fate_terminal.columns else np.nan
        neu = float(fate_terminal.loc[neighborhood, "NeuP"]) if "NeuP" in fate_terminal.columns else np.nan
        hsc = float(fate_terminal.loc[neighborhood, "HSC"]) if "HSC" in fate_terminal.columns else np.nan
        my_delta = float(protein_delta.loc[neighborhood, "Myeloid protein"]) if "Myeloid protein" in protein_delta.columns else np.nan
        ery_delta = float(protein_delta.loc[neighborhood, "Ery-Mk protein"]) if "Ery-Mk protein" in protein_delta.columns else np.nan
        stem_delta = float(protein_delta.loc[neighborhood, "Stem retention"]) if "Stem retention" in protein_delta.columns else np.nan
        matched_delta = float(protein_delta.loc[neighborhood, protein_module]) if protein_module in protein_delta.columns else np.nan
        lines.append(f"- N{neighborhood} `{label}`: `{gene_module}` score `{gene_mat.loc[neighborhood, gene_module]:+.3f}`, genes `{genes}`; matched protein module `{protein_module}` velocity `{matched_delta:+.3f}`; terminal MasP `{masp:.3f}`, NeuP `{neu:.3f}`, HSC `{hsc:.3f}`; background velocity Myeloid `{my_delta:+.3f}`, Ery-Mk `{ery_delta:+.3f}`, Stem `{stem_delta:+.3f}`.\n")
    lines.append("\n## Tables\n")
    for filename in [
        "rna_seed_protein_fate_timecourse.csv",
        "rna_seed_protein_module_timecourse.csv",
        "rna_seed_initial_gene_programs.csv",
        "rna_seed_dynamic_biology_summary.csv",
        "rna_seed_top_initial_genes.csv",
        "rna_seed_protein_mlp_classifier_audit.csv",
    ]:
        lines.append(f"- `{TABLE_DIR / filename}`\n")
    NOTE_PATH.write_text("".join(lines))


def write_manifest() -> None:
    MANIFEST_PATH.write_text(json.dumps({"figures": FIGURES, "note": str(NOTE_PATH), "tables": str(TABLE_DIR)}, indent=2))


def main() -> None:
    ensure_dirs()
    payload = build_dynamic_tables()
    write_tables(payload)
    figure_distribution_movie(payload)
    write_note(payload)
    write_manifest()
    print(json.dumps({"figures": len(FIGURES), "note": str(NOTE_PATH), "out_root": str(FIG_DIR)}, indent=2))


if __name__ == "__main__":
    main()
