#!/usr/bin/env python3
"""CITE-PCFM NBT-style RNA-to-protein biology downstream v2.

This script is intentionally fresh: it builds the biology figures directly
from the benchmark AnnData and locked trajectory artifacts, rather than
reusing prior multi-omics downstream scripts.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/cytobridge_mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/cytobridge_xdg_cache")
os.environ.setdefault("NUMBA_CACHE_DIR", "/private/tmp/numba_cache")
os.environ.setdefault("NUMBA_DISABLE_CACHE", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")
for key in ["MPLCONFIGDIR", "XDG_CACHE_HOME", "NUMBA_CACHE_DIR"]:
    os.makedirs(os.environ[key], exist_ok=True)

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy import sparse
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

try:
    import umap  # type: ignore

    HAVE_UMAP = True
except Exception:
    HAVE_UMAP = False

try:
    import scvelo as scv  # type: ignore

    HAVE_SCVELO = True
except Exception:
    scv = None
    HAVE_SCVELO = False


sns.set_theme(style="white", context="talk")

ROOT = Path("/Users/zhenyizhang/Documents/GitHub/CytoBridge-agent-runtime-v2-20260309/multi-omics")
DATA_PATH = Path("/Users/zhenyizhang/.cellcompass/algorithm_benchmarks/datasets/gse305370_citeseq_stage2_10k/data.h5ad")
OUT_ROOT = ROOT / "downstream/cite_pcfm_deep_story/biology_v2"
FIG_DIR = ROOT / "figures/cite_pcfm_deep_story/cite_biology_landscape_v2"
TABLE_DIR = OUT_ROOT / "tables"
MANIFEST_PATH = OUT_ROOT / "manifest.json"
NOTE_PATH = OUT_ROOT / "biology_story.md"

RNA_DIM = 50
PROTEIN_DIM = 30
RNG = np.random.default_rng(20260527)

METHODS = {
    "CITE-PCFM": {
        "trajectory": ROOT / "training_runs/20260526T174204321821Z__final__cite_paired_coord_fm__campaign_20260-trial_20260526-gse305370_ci/artifacts/evaluation_trajectory.npz",
        "color": "#E11D48",
        "kind": "rna_driven_protein_readout",
    },
    "CRUFM": {
        "trajectory": ROOT / "training_runs/20260525T220649168798Z__pilot__crufm__campaign_20260-stage2_c-baseline-crufm-gse305370_ci/artifacts/evaluation_trajectory.npz",
        "color": "#2563EB",
        "kind": "joint_latent_baseline",
    },
    "balanced OT-CFM": {
        "trajectory": ROOT / "training_runs/20260525T215826526439Z__pilot__balanced_ot_cfm__campaign_20260-stage2_c-baseline-balanced_ot_-gse305370_ci/artifacts/evaluation_trajectory.npz",
        "color": "#F59E0B",
        "kind": "joint_latent_baseline",
    },
}

TIME_GRID = [0.0, 1.0, 2.0, 3.0, 4.0]
VALIDATION_TIMES = [1.0, 2.0, 3.0, 4.0]
PHENO_ORDER = ["HSC", "MasP", "EryP", "MoP", "MkP", "NeuP"]
PHENO_COLORS = {
    "HSC": "#2F855A",
    "MasP": "#D62728",
    "EryP": "#F28E2B",
    "MoP": "#8C564B",
    "MkP": "#7B61B3",
    "NeuP": "#E377C2",
    "BP": "#1F77B4",
}

PROTEIN_MODULES = {
    "Stem retention": ["CD49f", "CD45RA", "CD123", "CD127"],
    "Ery-Mk protein": ["CD41", "CD42b", "CD71", "CD62P"],
    "Myeloid protein": ["CD11b", "CD64", "CD115", "CD172a", "CD13", "CD16"],
    "Mast/Baso protein": ["FceRIa", "CD123", "CD88", "CD13"],
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

MODULE_PAIRS = [
    ("Myeloid/MasP RNA", "Myeloid protein", "myeloid"),
    ("Erythroid RNA", "Ery-Mk protein", "erythroid"),
    ("Mast/Baso RNA", "Mast/Baso protein", "mast/baso"),
    ("HLA/activation RNA", "HLA/activation protein", "HLA"),
    ("Stem/HSPC RNA", "Stem retention", "stem"),
]

FIGURES: list[dict] = []


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)


def savefig(fig: plt.Figure, name: str, title: str, question: str) -> Path:
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    FIGURES.append({"name": name, "title": title, "question": question, "png": str(path), "pdf": str(path.with_suffix(".pdf"))})
    return path


def clean_name(name: str, prefix: str) -> str:
    return str(name).replace(prefix, "", 1)


def as_dense(x) -> np.ndarray:
    return x.toarray().astype(np.float32) if sparse.issparse(x) else np.asarray(x, dtype=np.float32)


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


def entropy_from_prob(prob: np.ndarray) -> float:
    p = np.clip(np.asarray(prob, dtype=float), 1e-12, 1.0)
    p = p / p.sum()
    return float(-np.sum(p * np.log(p)) / math.log(len(p)))


def load_traj(path: Path) -> dict:
    z = np.load(path)
    time_points = np.asarray(z["time_points"], dtype=float)
    points = {}
    weights = {}
    for i, time in enumerate(time_points):
        point_key = f"points_{i:06d}"
        weight_key = f"weights_{i:06d}"
        if point_key in z.files:
            points[float(time)] = np.asarray(z[point_key], dtype=np.float32)
        if weight_key in z.files:
            weights[float(time)] = np.asarray(z[weight_key], dtype=np.float32)
    return {"time_points": time_points, "points": points, "weights": weights}


def nearest_time(points: dict[float, np.ndarray], target: float) -> float:
    arr = np.asarray(list(points.keys()), dtype=float)
    return float(arr[np.argmin(np.abs(arr - float(target)))])


def density_grid(xy: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray, sigma: float = 1.1) -> np.ndarray:
    hist, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=[x_edges, y_edges])
    hist = gaussian_filter(hist.T.astype(float), sigma=sigma)
    max_value = float(np.nanmax(hist))
    return hist / max_value if max_value > 0 else hist


def fit_embedding(x: np.ndarray, n_neighbors: int, min_dist: float, seed: int) -> tuple[str, StandardScaler, object, np.ndarray]:
    scaler = StandardScaler().fit(x)
    xs = scaler.transform(x)
    if HAVE_UMAP:
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric="euclidean", random_state=seed, low_memory=True)
        emb = reducer.fit_transform(xs)
        return "UMAP", scaler, reducer, emb
    reducer = PCA(n_components=2, random_state=seed)
    emb = reducer.fit_transform(xs)
    return "PCA fallback", scaler, reducer, emb


def transform_embedding(scaler: StandardScaler, reducer: object, x: np.ndarray) -> np.ndarray:
    return reducer.transform(scaler.transform(x))


def train_fate_classifier(protein: np.ndarray, labels: np.ndarray) -> tuple[StandardScaler, MLPClassifier, pd.DataFrame, pd.DataFrame]:
    x_train, x_test, y_train, y_test = train_test_split(protein, labels, test_size=0.25, random_state=20260527, stratify=labels)
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
    summary = pd.DataFrame(
        [
            {
                "readout": "observed_protein_pca_mlp",
                "n_train": int(len(y_train)),
                "n_test": int(len(y_test)),
                "accuracy": float(accuracy_score(y_test, pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
                "n_iter": int(clf.n_iter_),
            }
        ]
    )
    report = classification_report(y_test, pred, output_dict=True, zero_division=0)
    rows = []
    for label in sorted(set(labels)):
        vals = report.get(label, {})
        rows.append(
            {
                "class": label,
                "precision": float(vals.get("precision", np.nan)),
                "recall": float(vals.get("recall", np.nan)),
                "f1": float(vals.get("f1-score", np.nan)),
                "support": int(vals.get("support", 0)),
            }
        )
    return scaler, clf, summary, pd.DataFrame(rows)


def fate_prob(protein: np.ndarray, scaler: StandardScaler, clf: MLPClassifier, batch_size: int = 512) -> pd.DataFrame:
    probs = []
    for start in range(0, protein.shape[0], batch_size):
        probs.append(clf.predict_proba(scaler.transform(protein[start : start + batch_size])))
    out = np.vstack(probs)
    return pd.DataFrame(out, columns=list(clf.classes_))


def module_scores(values: np.ndarray, marker_names: list[str], marker_mean: np.ndarray, marker_std: np.ndarray) -> pd.DataFrame:
    marker_index = {marker: i for i, marker in enumerate(marker_names)}
    z = (values - marker_mean[None, :]) / marker_std[None, :]
    rows = {}
    for module, markers in PROTEIN_MODULES.items():
        idx = [marker_index[m] for m in markers if m in marker_index]
        if idx:
            rows[module] = z[:, idx].mean(axis=1)
    return pd.DataFrame(rows)


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


def load_data_and_readouts() -> dict:
    adata = ad.read_h5ad(DATA_PATH, backed="r")
    obs = adata.obs.copy()
    obs["time_point_processed"] = obs["time_point_processed"].astype(float)
    labels = obs["cell_type"].astype(str).to_numpy()
    obs_time = obs["time_point_processed"].to_numpy(dtype=float)
    rna = np.asarray(adata.obsm["X_rna_svd"], dtype=np.float32)
    protein = np.asarray(adata.obsm["X_protein_pca"], dtype=np.float32)

    feature_types = adata.var["feature_types"].astype(str)
    protein_mask = feature_types.eq("Antibody Capture").to_numpy()
    rna_mask = feature_types.eq("Gene Expression").to_numpy()
    protein_indices = np.where(protein_mask)[0]
    rna_indices = np.where(rna_mask)[0]
    marker_names = [clean_name(x, "prot_") for x in adata.var.index[protein_mask]]
    gene_names = [clean_name(x, "rna_") for x in adata.var.index[rna_mask]]

    protein_adt = as_dense(adata[:, protein_indices].X)
    marker_mean = protein_adt.mean(axis=0)
    marker_std = protein_adt.std(axis=0)
    marker_std[marker_std < 1e-6] = 1.0
    decoder = Ridge(alpha=1.0).fit(protein, protein_adt)

    available_genes = set(gene_names)
    gene_modules = {
        module: [gene for gene in genes if gene in available_genes]
        for module, genes in GENE_MODULES.items()
    }
    gene_modules = {module: genes for module, genes in gene_modules.items() if genes}
    gene_list = sorted({gene for genes in gene_modules.values() for gene in genes})
    gene_lookup = {gene: i for i, gene in enumerate(gene_names)}
    gene_full_indices = [int(rna_indices[gene_lookup[gene]]) for gene in gene_list]
    gene_values_t0 = as_dense(adata[obs_time == 0.0, gene_full_indices].X)
    gene_values_t0 = (gene_values_t0 - gene_values_t0.mean(axis=0, keepdims=True)) / (gene_values_t0.std(axis=0, keepdims=True) + 1e-6)

    try:
        adata.file.close()
    except Exception:
        pass

    return {
        "obs": obs,
        "labels": labels,
        "obs_time": obs_time,
        "rna": rna,
        "protein": protein,
        "marker_names": marker_names,
        "marker_mean": marker_mean,
        "marker_std": marker_std,
        "decoder": decoder,
        "gene_modules": gene_modules,
        "gene_list": gene_list,
        "gene_values_t0": gene_values_t0,
        "t0_rna": rna[obs_time == 0.0],
        "t0_labels": labels[obs_time == 0.0],
    }


def build_payload() -> dict:
    data = load_data_and_readouts()
    trajectories = {method: load_traj(spec["trajectory"]) for method, spec in METHODS.items()}
    scaler, clf, fate_audit, fate_class_report = train_fate_classifier(data["protein"], data["labels"])
    pcfm = trajectories["CITE-PCFM"]
    rna0 = pcfm["points"][0.0][:, :RNA_DIM]
    protein4 = pcfm["points"][4.0][:, RNA_DIM : RNA_DIM + PROTEIN_DIM]
    terminal_prob = fate_prob(protein4, scaler, clf)

    raw_cluster = lightweight_kmeans(rna0, n_clusters=6)
    terminal_label = terminal_prob.idxmax(axis=1).to_numpy()
    cluster_order = []
    for cluster_id in range(6):
        mask = raw_cluster == cluster_id
        masp = float(np.mean(terminal_label[mask] == "MasP")) if mask.any() else 0.0
        hsc = float(np.mean(terminal_label[mask] == "HSC")) if mask.any() else 0.0
        ery = float(np.mean(terminal_label[mask] == "EryP")) if mask.any() else 0.0
        cluster_order.append((cluster_id, masp + 0.35 * hsc - 0.20 * ery))
    remap = {old: new for new, (old, _) in enumerate(sorted(cluster_order, key=lambda x: x[1], reverse=True))}
    neighborhoods = np.asarray([remap[c] for c in raw_cluster], dtype=int)
    centroids = np.vstack([rna0[neighborhoods == n].mean(axis=0) for n in range(6)])

    t0_dist = np.linalg.norm(data["t0_rna"][:, None, :] - centroids[None, :, :], axis=2)
    t0_assign = np.argmin(t0_dist, axis=1)
    gene_col = {gene: i for i, gene in enumerate(data["gene_list"])}
    gene_rows = []
    top_gene_rows = []
    for neighborhood in range(6):
        mask = t0_assign == neighborhood
        for module, genes in data["gene_modules"].items():
            idx = [gene_col[gene] for gene in genes if gene in gene_col]
            gene_rows.append(
                {
                    "neighborhood": neighborhood,
                    "gene_module": module,
                    "score": float(data["gene_values_t0"][mask][:, idx].mean()) if mask.any() and idx else np.nan,
                    "genes_used": ",".join([gene for gene in genes if gene in gene_col]),
                    "observed_t0_cells": int(mask.sum()),
                }
            )
        if mask.sum() >= 5:
            diff = data["gene_values_t0"][mask].mean(axis=0) - data["gene_values_t0"][~mask].mean(axis=0)
            for rank, col in enumerate(np.argsort(diff)[::-1][:10], start=1):
                top_gene_rows.append(
                    {
                        "neighborhood": neighborhood,
                        "rank": rank,
                        "gene": data["gene_list"][int(col)],
                        "cluster_vs_other_z": float(diff[int(col)]),
                    }
                )

    fate_rows = []
    protein_rows = []
    particle_rows = []
    for time in pcfm["time_points"]:
        time = float(time)
        pts = pcfm["points"][time]
        weights = pcfm["weights"].get(time)
        protein = pts[:, RNA_DIM : RNA_DIM + PROTEIN_DIM]
        probs = fate_prob(protein, scaler, clf)
        decoded = data["decoder"].predict(protein).astype(np.float32)
        module_df = module_scores(decoded, data["marker_names"], data["marker_mean"], data["marker_std"])
        for neighborhood in range(6):
            mask = neighborhoods == neighborhood
            local_weights = weights[mask] if weights is not None else None
            mean_probs = {col: float(weighted_mean(probs.loc[mask, col].to_numpy()[:, None], local_weights)[0]) for col in probs.columns}
            for phenotype, value in mean_probs.items():
                fate_rows.append({"neighborhood": neighborhood, "time": time, "phenotype": phenotype, "probability": value})
            fate_rows.append({"neighborhood": neighborhood, "time": time, "phenotype": "_entropy", "probability": entropy_from_prob(np.asarray(list(mean_probs.values())))})
            for module in module_df.columns:
                protein_rows.append(
                    {
                        "neighborhood": neighborhood,
                        "time": time,
                        "protein_module": module,
                        "score": float(weighted_mean(module_df.loc[mask, module].to_numpy()[:, None], local_weights)[0]),
                    }
                )
        if any(np.isclose(time, np.asarray(TIME_GRID))):
            for i in sample_idx(pts.shape[0], 1200, weights):
                particle_rows.append(
                    {
                        "particle": int(i),
                        "time": time,
                        "neighborhood": int(neighborhoods[i]),
                        "terminal_fate": str(terminal_label[i]),
                        "terminal_masp_probability": float(terminal_prob.loc[i, "MasP"]) if "MasP" in terminal_prob.columns else np.nan,
                    }
                )

    gene_df = pd.DataFrame(gene_rows)
    protein_df = pd.DataFrame(protein_rows)
    fate_df = pd.DataFrame(fate_rows)
    top_gene_df = pd.DataFrame(top_gene_rows)

    protein_initial = protein_df[np.isclose(protein_df["time"], 0.0)].pivot(index="neighborhood", columns="protein_module", values="score").fillna(0)
    protein_terminal = protein_df[np.isclose(protein_df["time"], 4.0)].pivot(index="neighborhood", columns="protein_module", values="score").fillna(0)
    protein_delta = protein_terminal.subtract(protein_initial, fill_value=0)
    gene_mat = gene_df.pivot(index="neighborhood", columns="gene_module", values="score").fillna(0)
    focus_neighborhoods = {
        "myeloid": int(gene_mat["Myeloid/MasP RNA"].idxmax()),
        "erythroid": int(gene_mat["Erythroid RNA"].idxmax()),
        "mast_baso": int(gene_mat["Mast/Baso RNA"].idxmax()),
        "hla": int(gene_mat["HLA/activation RNA"].idxmax()),
    }
    state_rows = []
    for neighborhood in range(6):
        terminal = fate_df[(fate_df["neighborhood"] == neighborhood) & np.isclose(fate_df["time"], 4.0)]
        terminal = terminal[~terminal["phenotype"].eq("_entropy")]
        top_module = gene_mat.loc[neighborhood].sort_values(ascending=False).index[0]
        state_rows.append(
            {
                "neighborhood": neighborhood,
                "top_initial_gene_module": top_module,
                "top_initial_gene_module_score": float(gene_mat.loc[neighborhood, top_module]),
                "dominant_terminal_fate": str(terminal.sort_values("probability", ascending=False).iloc[0]["phenotype"]),
                "dominant_terminal_probability": float(terminal["probability"].max()),
                "top_genes": ", ".join(top_gene_df[top_gene_df["neighborhood"] == neighborhood].head(5)["gene"].tolist()),
            }
        )

    return {
        "data": data,
        "trajectories": trajectories,
        "fate_scaler": scaler,
        "fate_classifier": clf,
        "fate_audit": fate_audit,
        "fate_class_report": fate_class_report,
        "terminal_prob": terminal_prob,
        "terminal_label": terminal_label,
        "neighborhoods": neighborhoods,
        "focus_neighborhoods": focus_neighborhoods,
        "gene_df": gene_df,
        "protein_df": protein_df,
        "fate_df": fate_df,
        "top_gene_df": top_gene_df,
        "protein_delta": protein_delta,
        "gene_mat": gene_mat,
        "state_df": pd.DataFrame(state_rows),
        "particle_df": pd.DataFrame(particle_rows),
    }


def build_protein_umap(payload: dict) -> None:
    data = payload["data"]
    trajectories = payload["trajectories"]
    blocks = [data["protein"]]
    refs = [("observed", "", np.nan, None, 0, data["protein"].shape[0])]
    cursor = data["protein"].shape[0]
    for method, traj in trajectories.items():
        for time in VALIDATION_TIMES:
            tt = nearest_time(traj["points"], time)
            pts = traj["points"][tt]
            idx = sample_idx(pts.shape[0], 700, traj["weights"].get(tt))
            blocks.append(pts[idx, RNA_DIM : RNA_DIM + PROTEIN_DIM])
            refs.append(("generated", method, time, idx, cursor, cursor + len(idx)))
            cursor += len(idx)
    pcfm = trajectories["CITE-PCFM"]
    movie_idx = sample_idx(pcfm["points"][0.0].shape[0], 1100, pcfm["weights"].get(0.0))
    for time in TIME_GRID:
        pts = pcfm["points"][time]
        idx = movie_idx
        blocks.append(pts[idx, RNA_DIM : RNA_DIM + PROTEIN_DIM])
        refs.append(("pcfm_movie", "CITE-PCFM", time, idx, cursor, cursor + len(idx)))
        cursor += len(idx)
    fit = np.vstack(blocks)
    emb_name, scaler, reducer, emb = fit_embedding(fit, n_neighbors=35, min_dist=0.25, seed=20260527)
    obs = payload["data"]["obs"].copy()
    obs_emb = emb[: data["protein"].shape[0]]
    observed_umap = pd.DataFrame(
        {
            "umap1": obs_emb[:, 0],
            "umap2": obs_emb[:, 1],
            "time": data["obs_time"],
            "cell_type": data["labels"],
        }
    )
    rows = []
    for kind, method, time, idx, start, end in refs[1:]:
        e = emb[start:end]
        for row_pos, particle_id in enumerate(idx):
            rows.append(
                {
                    "kind": kind,
                    "method": method,
                    "time": float(time),
                    "particle": int(particle_id),
                    "umap1": float(e[row_pos, 0]),
                    "umap2": float(e[row_pos, 1]),
                    "neighborhood": int(payload["neighborhoods"][particle_id]) if method == "CITE-PCFM" else -1,
                    "terminal_fate": str(payload["terminal_label"][particle_id]) if method == "CITE-PCFM" else "",
                }
            )
    payload["protein_umap_name"] = emb_name
    payload["protein_umap_observed"] = observed_umap
    payload["protein_umap_generated"] = pd.DataFrame(rows)
    payload["protein_umap_xlim"] = tuple(np.nanpercentile(emb[:, 0], [0.5, 99.5]))
    payload["protein_umap_ylim"] = tuple(np.nanpercentile(emb[:, 1], [0.5, 99.5]))


def build_rna_umap(payload: dict) -> None:
    data = payload["data"]
    pcfm = payload["trajectories"]["CITE-PCFM"]
    rna0 = pcfm["points"][0.0][:, :RNA_DIM]
    idx = sample_idx(rna0.shape[0], 1600)
    fit = np.vstack([data["t0_rna"], rna0[idx]])
    emb_name, scaler, reducer, emb = fit_embedding(fit, n_neighbors=35, min_dist=0.28, seed=20260528)
    observed = pd.DataFrame(
        {
            "umap1": emb[: data["t0_rna"].shape[0], 0],
            "umap2": emb[: data["t0_rna"].shape[0], 1],
            "cell_type": data["t0_labels"],
        }
    )
    generated = pd.DataFrame(
        {
            "umap1": emb[data["t0_rna"].shape[0] :, 0],
            "umap2": emb[data["t0_rna"].shape[0] :, 1],
            "particle": idx,
            "neighborhood": payload["neighborhoods"][idx],
            "terminal_fate": payload["terminal_label"][idx],
            "terminal_masp_probability": payload["terminal_prob"].loc[idx, "MasP"].to_numpy() if "MasP" in payload["terminal_prob"].columns else np.nan,
        }
    )
    payload["rna_umap_name"] = emb_name
    payload["rna_umap_observed"] = observed
    payload["rna_umap_generated"] = generated


def metric_summary(payload: dict) -> pd.DataFrame:
    rows = []
    obs = payload["data"]["protein"]
    obs_time = payload["data"]["obs_time"]
    for method, traj in payload["trajectories"].items():
        for time in VALIDATION_TIMES:
            tt = nearest_time(traj["points"], time)
            pred = traj["points"][tt][:, RNA_DIM : RNA_DIM + PROTEIN_DIM]
            target = obs[obs_time == time]
            pred_idx = sample_idx(pred.shape[0], 900, traj["weights"].get(tt))
            obs_idx = sample_idx(target.shape[0], 900)
            pred_s = pred[pred_idx]
            obs_s = target[obs_idx]
            dxy = np.linalg.norm(obs_s[:, None, :] - pred_s[None, :, :], axis=2).mean()
            dxx = np.linalg.norm(obs_s[:, None, :] - obs_s[None, :, :], axis=2).mean()
            dyy = np.linalg.norm(pred_s[:, None, :] - pred_s[None, :, :], axis=2).mean()
            pe = float(max(0.0, 2 * dxy - dxx - dyy))
            rows.append({"method": method, "time": time, "protein_energy": pe})
    return pd.DataFrame(rows)


def add_density_panel(ax, obs_df: pd.DataFrame, gen_df: pd.DataFrame, color: str, pe: float, xlim: tuple[float, float], ylim: tuple[float, float], show_labels: bool = False) -> None:
    all_x = np.linspace(xlim[0], xlim[1], 110)
    all_y = np.linspace(ylim[0], ylim[1], 110)
    xx, yy = np.meshgrid(0.5 * (all_x[:-1] + all_x[1:]), 0.5 * (all_y[:-1] + all_y[1:]))
    obs_den = density_grid(obs_df[["umap1", "umap2"]].to_numpy(), all_x, all_y, sigma=1.05)
    gen_den = density_grid(gen_df[["umap1", "umap2"]].to_numpy(), all_x, all_y, sigma=1.20)
    ax.contourf(xx, yy, obs_den, levels=[0.06, 0.14, 0.28, 0.48, 0.72, 1.01], colors=["#F8FAFC", "#F1F5F9", "#E2E8F0", "#CBD5E1", "#94A3B8"], alpha=0.82)
    ax.contour(xx, yy, obs_den, levels=[0.22, 0.46, 0.70], colors="#64748B", linewidths=[0.55, 0.75, 0.95], alpha=0.55)
    ax.contourf(xx, yy, gen_den, levels=[0.12, 0.26, 0.44, 0.64, 0.84, 1.01], colors=[color] * 5, alpha=0.10)
    ax.contour(xx, yy, gen_den, levels=[0.22, 0.44, 0.66], colors=color, linewidths=[1.15, 1.8, 2.5])
    if show_labels:
        for phenotype, pheno_color in PHENO_COLORS.items():
            sub = obs_df[obs_df["cell_type"].eq(phenotype)]
            if len(sub):
                center = sub[["umap1", "umap2"]].mean().to_numpy()
                ax.text(center[0], center[1], phenotype, fontsize=7.5, color=pheno_color, weight="bold", ha="center", va="center")
    ax.text(0.98, 0.94, f"PE {pe:.3f}", transform=ax.transAxes, ha="right", va="top", fontsize=7.8, color="#0F172A", bbox={"boxstyle": "round,pad=0.20", "facecolor": "white", "edgecolor": "#CBD5E1"})
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#CBD5E1")
        spine.set_linewidth(0.9)


def figure_01_gt_validation(payload: dict) -> None:
    metric_df = payload["metric_df"]
    obs = payload["protein_umap_observed"]
    gen = payload["protein_umap_generated"]
    xlim = payload["protein_umap_xlim"]
    ylim = payload["protein_umap_ylim"]
    methods = list(METHODS.keys())
    fig = plt.figure(figsize=(18.8, 9.4))
    title_ax = fig.add_axes([0.045, 0.925, 0.91, 0.06])
    title_ax.axis("off")
    title_ax.text(0, 0.72, "Generated protein states lie on the real CITE landscape", fontsize=23.5, weight="bold", color="#0F172A")
    title_ax.text(0, 0.18, "Grey terrain is matched observed protein UMAP density; colored contours are algorithm-generated protein distributions.", fontsize=10.8, color="#475569")

    grid_left, grid_bottom = 0.075, 0.285
    panel_w, panel_h = 0.150, 0.170
    x_gap, y_gap = 0.018, 0.040
    for col, time in enumerate(VALIDATION_TIMES):
        fig.text(grid_left + col * (panel_w + x_gap), grid_bottom + len(methods) * (panel_h + y_gap) - 0.010, f"t{int(time)}", fontsize=12, weight="bold")
    for row, method in enumerate(methods):
        y = grid_bottom + (len(methods) - 1 - row) * (panel_h + y_gap)
        color = METHODS[method]["color"]
        future = metric_df[(metric_df["method"] == method) & (metric_df["time"] > 0)]["protein_energy"].mean()
        fig.text(grid_left - 0.018, y + panel_h / 2, f"{method}\nfuture PE {future:.3f}", fontsize=9.5, weight="bold", color=color, ha="right", va="center")
        for col, time in enumerate(VALIDATION_TIMES):
            x = grid_left + col * (panel_w + x_gap)
            ax = fig.add_axes([x, y, panel_w, panel_h])
            obs_t = obs[np.isclose(obs["time"], time)]
            gen_t = gen[(gen["kind"] == "generated") & (gen["method"] == method) & np.isclose(gen["time"], time)]
            pe = float(metric_df[(metric_df["method"] == method) & np.isclose(metric_df["time"], time)]["protein_energy"].iloc[0])
            add_density_panel(ax, obs_t, gen_t, color, pe, xlim, ylim, show_labels=(row == 0 and col == 0))

    ax_line = fig.add_axes([0.805, 0.580, 0.165, 0.260])
    for method, spec in METHODS.items():
        d = metric_df[metric_df["method"] == method]
        ax_line.plot(d["time"], d["protein_energy"], marker="o", lw=2.2, color=spec["color"], label=method)
    ax_line.set_yscale("log")
    ax_line.set_title("protein-energy over time", loc="left", fontsize=12, weight="bold")
    ax_line.set_xlabel("time", fontsize=9)
    ax_line.set_ylabel("PE, log", fontsize=9)
    ax_line.legend(frameon=False, fontsize=7)
    ax_line.spines[["top", "right"]].set_visible(False)

    ax_bar = fig.add_axes([0.805, 0.315, 0.165, 0.180])
    aggregate = metric_df.groupby("method")["protein_energy"].mean().reindex(methods)
    values = (aggregate - aggregate.min()) / (aggregate.max() - aggregate.min() + 1e-9)
    y_pos = np.arange(len(methods))
    ax_bar.barh(y_pos, values, color=[METHODS[m]["color"] for m in methods], alpha=0.86, height=0.55)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(methods, fontsize=7.8)
    ax_bar.invert_yaxis()
    ax_bar.set_xlim(0, 1.0)
    ax_bar.set_title("future mismatch", loc="left", fontsize=12, weight="bold")
    ax_bar.set_xlabel("normalized PE", fontsize=9)
    ax_bar.spines[["top", "right", "left"]].set_visible(False)
    for y, method in zip(y_pos, methods):
        ax_bar.text(values.loc[method] + 0.03, y, f"{aggregate.loc[method]:.3f}", va="center", fontsize=8.3, weight="bold")

    fig.text(0.075, 0.225, "This panel establishes model credibility only: realistic protein support, not yet the biological discovery.", fontsize=9.2, color="#334155")
    savefig(fig, "bio_v2_01_gt_protein_umap_validation", "GT protein UMAP validation", "Do generated protein states lie on the observed CITE protein landscape?")


def figure_02_rna_to_protein_movie(payload: dict) -> None:
    obs = payload["protein_umap_observed"]
    gen = payload["protein_umap_generated"]
    xlim = payload["protein_umap_xlim"]
    ylim = payload["protein_umap_ylim"]
    fig = plt.figure(figsize=(18.5, 5.8))
    title_ax = fig.add_axes([0.045, 0.860, 0.91, 0.105])
    title_ax.axis("off")
    title_ax.text(0, 0.75, "One RNA initialization becomes a protein movie", fontsize=23.5, weight="bold", color="#0F172A")
    title_ax.text(0, 0.22, "CITE-PCFM rolls the same RNA-initialized particles forward and generates time-resolved protein distributions.", fontsize=10.8, color="#475569")
    time_colors = {0.0: "#2563EB", 1.0: "#06B6D4", 2.0: "#22C55E", 3.0: "#F59E0B", 4.0: "#E11D48"}
    left, bottom, width, height, gap = 0.055, 0.265, 0.165, 0.480, 0.018
    centroids = []
    for col, time in enumerate(TIME_GRID):
        ax = fig.add_axes([left + col * (width + gap), bottom, width, height])
        obs_t = obs[np.isclose(obs["time"], time)]
        if obs_t.empty:
            obs_t = obs[np.isclose(obs["time"], min(max(time, 1.0), 4.0))]
        gen_t = gen[(gen["kind"] == "pcfm_movie") & np.isclose(gen["time"], time)]
        color = time_colors[time]
        all_x = np.linspace(xlim[0], xlim[1], 110)
        all_y = np.linspace(ylim[0], ylim[1], 110)
        xx, yy = np.meshgrid(0.5 * (all_x[:-1] + all_x[1:]), 0.5 * (all_y[:-1] + all_y[1:]))
        obs_den = density_grid(obs_t[["umap1", "umap2"]].to_numpy(), all_x, all_y)
        gen_den = density_grid(gen_t[["umap1", "umap2"]].to_numpy(), all_x, all_y)
        ax.contourf(xx, yy, obs_den, levels=[0.06, 0.16, 0.32, 0.56, 0.82, 1.01], colors=["#F8FAFC", "#F1F5F9", "#E2E8F0", "#CBD5E1", "#94A3B8"], alpha=0.74)
        ax.contour(xx, yy, gen_den, levels=[0.20, 0.42, 0.66], colors=color, linewidths=[1.2, 1.8, 2.5])
        ax.scatter(gen_t["umap1"], gen_t["umap2"], s=4, color=color, alpha=0.10, linewidths=0)
        c = gen_t[["umap1", "umap2"]].mean().to_numpy()
        centroids.append(c)
        ax.scatter([c[0]], [c[1]], s=55, color=color, edgecolor="white", linewidth=1.1, zorder=5)
        ax.set_title(f"t{int(time)}", loc="left", fontsize=12, weight="bold")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#CBD5E1")
    ax_path = fig.add_axes([0.135, 0.060, 0.68, 0.130])
    ax_path.axis("off")
    ax_path.set_xlim(0, 1)
    ax_path.set_ylim(0, 1)
    for i, time in enumerate(TIME_GRID):
        x = 0.08 + i * 0.20
        ax_path.scatter([x], [0.45], s=110, color=time_colors[time], edgecolor="white", linewidth=1.2, zorder=3)
        ax_path.text(x, 0.12, f"t{int(time)}", ha="center", fontsize=10, color="#334155")
        if i < len(TIME_GRID) - 1:
            ax_path.add_patch(FancyArrowPatch((x + 0.025, 0.45), (x + 0.175, 0.45), arrowstyle="Simple,head_width=6,head_length=8", color="#94A3B8", alpha=0.75))
    ax_path.text(0.5, 0.88, "same initial RNA particles, protein distribution changes over generated time", ha="center", fontsize=10, color="#334155")
    savefig(fig, "bio_v2_02_rna_to_protein_movie", "RNA-to-protein movie", "How do the same RNA-initialized particles evolve into protein distributions over time?")


def figure_03_initial_rna_fate_map(payload: dict) -> None:
    observed = payload["rna_umap_observed"]
    generated = payload["rna_umap_generated"]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), constrained_layout=True)
    axes[0].scatter(observed["umap1"], observed["umap2"], s=7, color="#CBD5E1", alpha=0.50, linewidths=0)
    sc = axes[0].scatter(generated["umap1"], generated["umap2"], c=generated["terminal_masp_probability"], cmap="magma", s=12, alpha=0.88, linewidths=0)
    axes[0].set_title("t0 RNA states colored by future MasP protein probability", loc="left", fontsize=12, weight="bold")
    cb = fig.colorbar(sc, ax=axes[0], fraction=0.040, pad=0.01)
    cb.set_label("future MasP protein probability")
    for phenotype, color in PHENO_COLORS.items():
        sub = generated[generated["terminal_fate"].eq(phenotype)]
        if len(sub) and phenotype != "BP":
            axes[1].scatter(sub["umap1"], sub["umap2"], s=10, color=color, alpha=0.75, linewidths=0, label=phenotype)
    axes[1].set_title("same t0 RNA map colored by terminal protein fate", loc="left", fontsize=12, weight="bold")
    axes[1].legend(frameon=False, fontsize=8, ncol=3, loc="lower left")
    for ax in axes:
        ax.set_xlabel(f"RNA {payload['rna_umap_name']} 1")
        ax.set_ylabel(f"RNA {payload['rna_umap_name']} 2")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#CBD5E1")
    fig.suptitle("Early RNA encodes future protein fate", fontsize=18, weight="bold")
    savefig(fig, "bio_v2_03_initial_rna_future_protein_fate", "Initial RNA future fate map", "Does the early RNA state already encode future protein fate?")


def figure_04_commitment_time(payload: dict) -> None:
    fate = payload["fate_df"]
    focus = payload["focus_neighborhoods"]
    selected = [focus["myeloid"], focus["erythroid"], focus["mast_baso"], focus["hla"]]
    labels = {focus["myeloid"]: "myeloid N", focus["erythroid"]: "erythroid N", focus["mast_baso"]: "mast/baso N", focus["hla"]: "HLA N"}
    colors = sns.color_palette("tab10", 6)
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4), constrained_layout=True)
    metrics = [("_entropy", "protein-fate entropy", "branch ambiguity"), ("MasP", "MasP probability", "myeloid commitment"), ("HSC", "HSC-retention probability", "stem retention")]
    for ax, (phenotype, title, ylabel) in zip(axes, metrics):
        for neighborhood in selected:
            d = fate[(fate["neighborhood"] == neighborhood) & (fate["phenotype"] == phenotype)].sort_values("time")
            ax.plot(d["time"], d["probability"], lw=2.1, color=colors[neighborhood], label=f"{labels[neighborhood]}{neighborhood}")
        ax.axvspan(1.0, 3.0, color="#FDE68A", alpha=0.22, zorder=0)
        ax.set_title(title, loc="left", fontsize=12, weight="bold")
        ax.set_xlabel("generated time")
        ax.set_ylabel(ylabel)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=7.8)
    fig.suptitle("Protein commitment opens during the middle rollout window", fontsize=18, weight="bold")
    savefig(fig, "bio_v2_04_commitment_time_landscape", "Commitment-time landscape", "When does protein fate start to sharpen?")


def kernel_velocity_grid(xy: np.ndarray, vel: np.ndarray, value: np.ndarray, xlim: tuple[float, float], ylim: tuple[float, float], grid_n: int = 32) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gx = np.linspace(xlim[0], xlim[1], grid_n)
    gy = np.linspace(ylim[0], ylim[1], grid_n)
    xx, yy = np.meshgrid(gx, gy)
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1)
    scale = 0.12 * max(xlim[1] - xlim[0], ylim[1] - ylim[0])
    diff = grid[:, None, :] - xy[None, :, :]
    weights = np.exp(-0.5 * (diff[:, :, 0] ** 2 + diff[:, :, 1] ** 2) / (scale**2))
    weight_sum = weights.sum(axis=1) + 1e-9
    u = (weights @ vel[:, 0]) / weight_sum
    v = (weights @ vel[:, 1]) / weight_sum
    val = (weights @ value) / weight_sum
    density = weight_sum / weight_sum.max()
    u[density < 0.08] = np.nan
    v[density < 0.08] = np.nan
    val[density < 0.08] = np.nan
    return xx, yy, u.reshape(xx.shape), v.reshape(xx.shape), val.reshape(xx.shape)


def figure_05_velocity_field(payload: dict) -> None:
    gen = payload["protein_umap_generated"]
    pcfm = payload["trajectories"]["CITE-PCFM"]
    xlim = payload["protein_umap_xlim"]
    ylim = payload["protein_umap_ylim"]
    module_values = {}
    for time in TIME_GRID:
        pts = pcfm["points"][time]
        decoded = payload["data"]["decoder"].predict(pts[:, RNA_DIM : RNA_DIM + PROTEIN_DIM]).astype(np.float32)
        module_values[time] = module_scores(decoded, payload["data"]["marker_names"], payload["data"]["marker_mean"], payload["data"]["marker_std"])
    pairs = []
    for t0, t1 in zip(TIME_GRID[:-1], TIME_GRID[1:]):
        a = gen[(gen["kind"] == "pcfm_movie") & np.isclose(gen["time"], t0)].set_index("particle")
        b = gen[(gen["kind"] == "pcfm_movie") & np.isclose(gen["time"], t1)].set_index("particle")
        common = a.index.intersection(b.index)
        if len(common) == 0:
            continue
        xy = 0.5 * (a.loc[common, ["umap1", "umap2"]].to_numpy() + b.loc[common, ["umap1", "umap2"]].to_numpy())
        vel = b.loc[common, ["umap1", "umap2"]].to_numpy() - a.loc[common, ["umap1", "umap2"]].to_numpy()
        for module in ["Myeloid protein", "Ery-Mk protein", "Stem retention"]:
            delta = module_values[t1].loc[common, module].to_numpy() - module_values[t0].loc[common, module].to_numpy()
            pairs.append(pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "u": vel[:, 0], "v": vel[:, 1], "delta": delta, "module": module}))
    vf = pd.concat(pairs, ignore_index=True)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), constrained_layout=True)
    cmap = {"Myeloid protein": "Greens", "Ery-Mk protein": "Oranges", "Stem retention": "Blues"}
    for ax, module in zip(axes, ["Myeloid protein", "Ery-Mk protein", "Stem retention"]):
        d = vf[vf["module"] == module]
        xx, yy, u, v, val = kernel_velocity_grid(d[["x", "y"]].to_numpy(), d[["u", "v"]].to_numpy(), d["delta"].to_numpy(), xlim, ylim)
        vmax = np.nanpercentile(np.abs(val), 95)
        im = ax.contourf(xx, yy, val, levels=np.linspace(-vmax, vmax, 13), cmap="coolwarm", alpha=0.60, extend="both")
        ax.streamplot(xx, yy, np.nan_to_num(u), np.nan_to_num(v), color="#111827", density=1.1, linewidth=0.8, arrowsize=0.8)
        ax.set_title(module, loc="left", fontsize=12, weight="bold")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#CBD5E1")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.70, pad=0.012, label="local module velocity")
    fig.suptitle("Protein dynamics as a local velocity field", fontsize=18, weight="bold")
    savefig(fig, "bio_v2_05_protein_velocity_field", "Protein velocity field", "How do surface-marker programs dynamically increase or decrease?")


def figure_06_transfer_function(payload: dict) -> None:
    gene_mat = payload["gene_mat"]
    protein_delta = payload["protein_delta"]
    rows = []
    for gene_module, protein_module, label in MODULE_PAIRS:
        if gene_module not in gene_mat.columns or protein_module not in protein_delta.columns:
            continue
        x = gene_mat[gene_module].to_numpy()
        y = protein_delta[protein_module].to_numpy()
        corr = float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 else np.nan
        best_neighborhood = int(gene_mat[gene_module].idxmax())
        rows.append({"gene_module": gene_module, "protein_module": protein_module, "label": label, "corr": corr, "top_neighborhood": best_neighborhood, "top_gene_score": float(gene_mat.loc[best_neighborhood, gene_module]), "top_protein_velocity": float(protein_delta.loc[best_neighborhood, protein_module])})
    df = pd.DataFrame(rows).sort_values("top_protein_velocity", ascending=True)
    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    y = np.arange(len(df))
    colors = ["#DC2626" if v < 0 else "#059669" for v in df["top_protein_velocity"]]
    ax.axvline(0, color="#CBD5E1", lw=1)
    ax.hlines(y, 0, df["top_protein_velocity"], color=colors, lw=2)
    ax.scatter(df["top_protein_velocity"], y, s=85, color=colors, edgecolor="white", linewidth=1.0, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.label}: N{r.top_neighborhood}" for r in df.itertuples()], fontsize=9)
    for yy, row in zip(y, df.itertuples()):
        ax.text(row.top_protein_velocity + (0.035 if row.top_protein_velocity >= 0 else -0.035), yy, f"{row.top_protein_velocity:+.2f}", va="center", ha="left" if row.top_protein_velocity >= 0 else "right", fontsize=9, weight="bold")
    ax.set_xlabel("protein-module velocity, t4 - t0")
    ax.tick_params(axis="x", labelsize=9)
    ax.set_title("RNA program -> protein velocity transfer function", loc="left", fontsize=14, weight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    df.to_csv(TABLE_DIR / "rna_program_to_protein_velocity_transfer.csv", index=False)
    savefig(fig, "bio_v2_06_rna_program_to_protein_velocity", "RNA program to protein velocity", "Which RNA gene programs control protein velocity?")


def case_study_figure(payload: dict, case_key: str, name: str, title: str, gene_module: str, protein_module: str, purpose: str) -> None:
    neighborhood = payload["focus_neighborhoods"][case_key]
    top_genes = payload["top_gene_df"][payload["top_gene_df"]["neighborhood"] == neighborhood].head(5)
    gene_mat = payload["gene_mat"]
    protein_delta = payload["protein_delta"]
    fate = payload["fate_df"]
    gen = payload["protein_umap_generated"]
    xlim = payload["protein_umap_xlim"]
    ylim = payload["protein_umap_ylim"]
    colors = sns.color_palette("tab10", 6)
    fig = plt.figure(figsize=(14.5, 5.2))
    ax_gene = fig.add_axes([0.055, 0.235, 0.23, 0.565])
    ax_umap = fig.add_axes([0.335, 0.205, 0.33, 0.640])
    ax_fate = fig.add_axes([0.725, 0.235, 0.22, 0.535])
    ax_gene.barh(np.arange(len(top_genes)), top_genes["cluster_vs_other_z"], color=colors[neighborhood], alpha=0.85)
    ax_gene.set_yticks(np.arange(len(top_genes)))
    ax_gene.set_yticklabels(top_genes["gene"], fontsize=9)
    ax_gene.invert_yaxis()
    ax_gene.set_xlabel("RNA z-score vs other neighborhoods", fontsize=10, labelpad=3)
    ax_gene.set_title(f"N{neighborhood} initial RNA program\n{gene_module}: {gene_mat.loc[neighborhood, gene_module]:+.2f}", loc="left", fontsize=12, weight="bold")
    ax_gene.tick_params(axis="x", labelsize=8)
    ax_gene.spines[["top", "right"]].set_visible(False)

    centers = []
    for time in TIME_GRID:
        d = gen[(gen["kind"] == "pcfm_movie") & np.isclose(gen["time"], time) & (gen["neighborhood"] == neighborhood)]
        ax_umap.scatter(d["umap1"], d["umap2"], s=9, color=colors[neighborhood], alpha=0.14 + 0.14 * time / 4.0, linewidths=0)
        if len(d):
            center = d[["umap1", "umap2"]].mean().to_numpy()
            centers.append((time, center))
            ax_umap.scatter([center[0]], [center[1]], s=60, color=colors[neighborhood], edgecolor="white", linewidth=1.1, zorder=3)
            ax_umap.text(center[0], center[1], f"t{int(time)}", fontsize=8, color="#111827", ha="center", va="center")
    for (_, start), (_, end) in zip(centers[:-1], centers[1:]):
        ax_umap.add_patch(FancyArrowPatch(tuple(start), tuple(end), arrowstyle="Simple,head_width=5,head_length=7", color="#111827", alpha=0.42, linewidth=0.7))
    ax_umap.set_xlim(*xlim)
    ax_umap.set_ylim(*ylim)
    ax_umap.set_xticks([])
    ax_umap.set_yticks([])
    ax_umap.set_title(f"protein UMAP trajectory\n{protein_module} velocity {protein_delta.loc[neighborhood, protein_module]:+.3f}", loc="left", fontsize=12, weight="bold")
    for spine in ax_umap.spines.values():
        spine.set_color("#CBD5E1")

    ph = [p for p in ["HSC", "MasP", "NeuP", "EryP", "MkP"] if p in set(fate["phenotype"])]
    for phenotype in ph:
        d = fate[(fate["neighborhood"] == neighborhood) & (fate["phenotype"] == phenotype)].sort_values("time")
        ax_fate.plot(d["time"], d["probability"], lw=2.0, marker="o", ms=3.5, color=PHENO_COLORS.get(phenotype, "#64748B"), label=phenotype)
    ax_fate.set_title("protein fate probabilities", loc="left", fontsize=12, weight="bold")
    ax_fate.set_xlabel("generated time", fontsize=10, labelpad=3)
    ax_fate.set_ylabel("probability", fontsize=10)
    ax_fate.tick_params(axis="both", labelsize=8)
    ax_fate.legend(frameon=False, fontsize=8)
    ax_fate.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title, fontsize=18, weight="bold", x=0.055, ha="left")
    fig.text(0.055, 0.045, purpose, fontsize=9.2, color="#334155")
    savefig(fig, name, title, purpose)


def figure_09_mixed_ridge(payload: dict) -> None:
    focus = payload["focus_neighborhoods"]
    neighborhoods = [focus["mast_baso"], focus["hla"]]
    names = ["mast/baso RNA ridge", "HLA/activation RNA ridge"]
    gen = payload["protein_umap_generated"]
    fate = payload["fate_df"]
    xlim = payload["protein_umap_xlim"]
    ylim = payload["protein_umap_ylim"]
    colors = sns.color_palette("tab10", 6)
    fig = plt.figure(figsize=(14.0, 5.6))
    for i, (neighborhood, label) in enumerate(zip(neighborhoods, names)):
        ax = fig.add_axes([0.055 + i * 0.48, 0.18, 0.26, 0.62])
        for time in TIME_GRID:
            d = gen[(gen["kind"] == "pcfm_movie") & np.isclose(gen["time"], time) & (gen["neighborhood"] == neighborhood)]
            ax.scatter(d["umap1"], d["umap2"], s=8, color=colors[neighborhood], alpha=0.12 + 0.16 * time / 4.0, linewidths=0)
        ax.set_title(f"N{neighborhood} {label}", loc="left", fontsize=12, weight="bold")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#CBD5E1")
        ax2 = fig.add_axes([0.325 + i * 0.48, 0.18, 0.18, 0.62])
        ph = ["HSC", "MasP", "NeuP", "EryP", "MkP"]
        mat = fate[(fate["neighborhood"] == neighborhood) & (fate["phenotype"].isin(ph))].pivot(index="time", columns="phenotype", values="probability").fillna(0).sort_index()
        ax2.stackplot(mat.index, [mat[p] for p in mat.columns], colors=[PHENO_COLORS[p] for p in mat.columns], labels=mat.columns, alpha=0.90, edgecolor="white", linewidth=0.7)
        ax2.set_ylim(0, 1)
        ax2.set_xlabel("time")
        ax2.set_ylabel("fate probability")
        ax2.set_title("mixed fate ridge", loc="left", fontsize=12, weight="bold")
        ax2.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Mast/baso and HLA programs form mixed protein ridges", fontsize=18, weight="bold", x=0.055, ha="left")
    savefig(fig, "bio_v2_09_mixed_ridge_biology", "Mixed-ridge biology", "Are mast/baso and HLA/activation RNA programs terminal fates or transitional ridges?")


def figure_10_counterfactual(payload: dict) -> None:
    gene_mat = payload["gene_mat"]
    gen = payload["protein_umap_generated"]
    xlim = payload["protein_umap_xlim"]
    ylim = payload["protein_umap_ylim"]
    centers = []
    for neighborhood in range(6):
        t0 = gen[(gen["kind"] == "pcfm_movie") & np.isclose(gen["time"], 0.0) & (gen["neighborhood"] == neighborhood)][["umap1", "umap2"]].mean().to_numpy()
        t4 = gen[(gen["kind"] == "pcfm_movie") & np.isclose(gen["time"], 4.0) & (gen["neighborhood"] == neighborhood)][["umap1", "umap2"]].mean().to_numpy()
        centers.append({"neighborhood": neighborhood, "x0": t0[0], "y0": t0[1], "dx": t4[0] - t0[0], "dy": t4[1] - t0[1]})
    center_df = pd.DataFrame(centers).set_index("neighborhood")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), constrained_layout=True)
    cases = [("Myeloid/MasP RNA", "exploratory myeloid +1 RNA-program shift", "#059669"), ("Erythroid RNA", "exploratory erythroid +1 RNA-program shift", "#DC2626")]
    for ax, (module, title, color) in zip(axes, cases):
        x = gene_mat[module].to_numpy().reshape(-1, 1)
        dx_model = LinearRegression().fit(x, center_df["dx"].to_numpy())
        dy_model = LinearRegression().fit(x, center_df["dy"].to_numpy())
        ax.scatter(center_df["x0"], center_df["y0"], s=70, color="#CBD5E1", edgecolor="white", linewidth=1.0, label="control t0")
        for neighborhood in center_df.index:
            start = center_df.loc[neighborhood, ["x0", "y0"]].to_numpy()
            observed_end = start + center_df.loc[neighborhood, ["dx", "dy"]].to_numpy()
            shifted_score = gene_mat.loc[neighborhood, module] + 1.0
            pred_shift = np.array([dx_model.predict([[shifted_score]])[0], dy_model.predict([[shifted_score]])[0]])
            pred_end = start + pred_shift
            ax.plot([start[0], observed_end[0]], [start[1], observed_end[1]], color="#94A3B8", lw=1.0, alpha=0.70)
            ax.add_patch(FancyArrowPatch(tuple(start), tuple(pred_end), arrowstyle="Simple,head_width=5,head_length=7", color=color, alpha=0.72, linewidth=1.0))
            ax.text(start[0], start[1], f"N{neighborhood}", fontsize=8, ha="center", va="center", color="#111827")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, loc="left", fontsize=12, weight="bold")
        for spine in ax.spines.values():
            spine.set_color("#CBD5E1")
    fig.suptitle("In silico RNA-program perturbation as a testable hypothesis generator", fontsize=17, weight="bold")
    fig.text(0.02, 0.02, "Exploratory: derived from locked rollout neighborhood transfer functions because direct gene-space rerollout is not available in the evaluation artifact.", fontsize=8.3, color="#64748B")
    savefig(fig, "bio_v2_10_in_silico_rna_program_perturbation", "In silico RNA program perturbation", "If RNA programs are strengthened or weakened, how does the predicted protein movie change?")


def write_tables(payload: dict) -> None:
    payload["fate_audit"].to_csv(TABLE_DIR / "fate_classifier_audit_summary.csv", index=False)
    payload["fate_class_report"].to_csv(TABLE_DIR / "fate_classifier_class_report.csv", index=False)
    payload["metric_df"].to_csv(TABLE_DIR / "gt_validation_protein_energy_by_method_time.csv", index=False)
    payload["fate_df"].to_csv(TABLE_DIR / "neighborhood_fate_probability_timecourse.csv", index=False)
    payload["protein_df"].to_csv(TABLE_DIR / "neighborhood_protein_module_timecourse.csv", index=False)
    payload["gene_df"].to_csv(TABLE_DIR / "neighborhood_initial_gene_modules.csv", index=False)
    payload["top_gene_df"].to_csv(TABLE_DIR / "neighborhood_top_initial_genes.csv", index=False)
    payload["state_df"].to_csv(TABLE_DIR / "state_structure_summary.csv", index=False)
    payload["protein_umap_observed"].to_csv(TABLE_DIR / "protein_umap_observed.csv", index=False)
    payload["protein_umap_generated"].to_csv(TABLE_DIR / "protein_umap_generated.csv", index=False)
    payload["rna_umap_observed"].to_csv(TABLE_DIR / "rna_umap_observed_t0.csv", index=False)
    payload["rna_umap_generated"].to_csv(TABLE_DIR / "rna_umap_generated_t0_future_fate.csv", index=False)


def write_note(payload: dict) -> None:
    audit = payload["fate_audit"].iloc[0]
    focus = payload["focus_neighborhoods"]
    protein_delta = payload["protein_delta"]
    gene_mat = payload["gene_mat"]
    fate = payload["fate_df"]
    lines = [
        "# CITE-PCFM RNA-to-Protein Biology v2\n\n",
        "This analysis is organized as a biological landscape story enabled by RNA-to-protein dynamics, not as a benchmark appendix.\n\n",
        f"Fate readout: observed protein-PCA MLP classifier, accuracy `{audit['accuracy']:.3f}`, balanced accuracy `{audit['balanced_accuracy']:.3f}`. BP is retained in audit tables but not interpreted as a major fate because its support is small.\n\n",
        "## Biological Spine\n\n",
        "CITE-PCFM converts RNA initialization into a protein-state movie. The strongest biological claim is that RNA lineage priming and surface-protein commitment are related but separable.\n\n",
    ]
    myeloid = focus["myeloid"]
    erythroid = focus["erythroid"]
    terminal = fate[np.isclose(fate["time"], 4.0)]
    masp_ery = terminal[(terminal["neighborhood"] == erythroid) & (terminal["phenotype"] == "MasP")]["probability"].iloc[0]
    lines.extend(
        [
            f"- Myeloid priming N{myeloid}: `Myeloid/MasP RNA` score `{gene_mat.loc[myeloid, 'Myeloid/MasP RNA']:+.3f}` and myeloid protein velocity `{protein_delta.loc[myeloid, 'Myeloid protein']:+.3f}`.\n",
            f"- Erythroid diversion N{erythroid}: `Erythroid RNA` score `{gene_mat.loc[erythroid, 'Erythroid RNA']:+.3f}`, Ery-Mk protein velocity `{protein_delta.loc[erythroid, 'Ery-Mk protein']:+.3f}`, terminal MasP probability `{masp_ery:.3f}`.\n",
            "- Mast/baso and HLA/activation programs behave as mixed ridges rather than clean terminal fates.\n\n",
            "## Figure Program\n\n",
        ]
    )
    for fig in FIGURES:
        lines.append(f"- `{fig['name']}`: {fig['question']} File: `{fig['png']}`\n")
    NOTE_PATH.write_text("".join(lines), encoding="utf-8")


def write_manifest(payload: dict) -> None:
    manifest = {
        "analysis": "CITE-PCFM NBT biology v2",
        "dataset": str(DATA_PATH),
        "trajectory_source": "locked evaluation trajectory artifacts",
        "protein_embedding": payload.get("protein_umap_name"),
        "rna_embedding": payload.get("rna_umap_name"),
        "scvelo_import_available": HAVE_SCVELO,
        "velocity_field_evidence_type": "local flow diagnostic from rollout-derived protein UMAP displacement",
        "perturbation_evidence_type": "exploratory transfer-function counterfactual, not direct model rerollout",
        "figures": FIGURES,
        "tables": [str(p) for p in sorted(TABLE_DIR.glob("*.csv"))],
        "note": str(NOTE_PATH),
        "warnings": [
            "Protein fate probabilities use an observed protein-PCA MLP classifier; classifier audit is included.",
            "Generated ADT marker modules are decoded from observed protein PCA with a ridge decoder.",
            "The perturbation panel is exploratory because the locked evaluation artifact does not expose direct gene-space rerollout.",
            "The protein velocity field is a scVelo-style local diagnostic from model rollout displacements, not independent RNA velocity evidence.",
        ],
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    payload = build_payload()
    build_protein_umap(payload)
    build_rna_umap(payload)
    payload["metric_df"] = metric_summary(payload)
    write_tables(payload)

    figure_01_gt_validation(payload)
    figure_02_rna_to_protein_movie(payload)
    figure_03_initial_rna_fate_map(payload)
    figure_04_commitment_time(payload)
    figure_05_velocity_field(payload)
    figure_06_transfer_function(payload)
    case_study_figure(
        payload,
        "myeloid",
        "bio_v2_07_myeloid_priming_case_study",
        "Myeloid RNA priming becomes myeloid surface-protein gain",
        "Myeloid/MasP RNA",
        "Myeloid protein",
        "Positive biological anchor: MPO/CTSG/AZU1/PRTN3/ELANE-high RNA neighborhoods gain myeloid ADT signal and move toward MasP/NeuP protein states.",
    )
    case_study_figure(
        payload,
        "erythroid",
        "bio_v2_08_erythroid_diversion_case_study",
        "Erythroid RNA priming is diverted at the protein level",
        "Erythroid RNA",
        "Ery-Mk protein",
        "Biological hook: KLF1/GATA1/HBA/HBB-high RNA neighborhoods lose Ery-Mk protein velocity and retain substantial MasP protein probability.",
    )
    figure_09_mixed_ridge(payload)
    figure_10_counterfactual(payload)

    write_note(payload)
    write_manifest(payload)
    print(json.dumps({"figures": len(FIGURES), "figure_dir": str(FIG_DIR), "manifest": str(MANIFEST_PATH)}, indent=2))


if __name__ == "__main__":
    main()
