#!/usr/bin/env python3
"""Ground-truth-first RNA-to-protein downstream analyses for CITE-PCFM.

The analysis focuses on the feature that distinguishes CITE-PCFM from joint
latent baselines: CITE-PCFM can start from RNA coordinates and produce a
time-conditioned protein readout. Every main figure compares generated protein
coordinates against observed target cells at the matched time point.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import StandardScaler

try:
    import umap  # type: ignore

    HAVE_UMAP = True
except Exception:
    HAVE_UMAP = False


sns.set_theme(style="whitegrid", context="paper")

OUTPUT_DIR = Path("/Users/zhenyizhang/Documents/GitHub/CytoBridge-agent-runtime-v2-20260309/multi-omics")
DATA_PATH = Path("/Users/zhenyizhang/.cellcompass/algorithm_benchmarks/datasets/gse305370_citeseq_stage2_10k/data.h5ad")

FIG_DIR = OUTPUT_DIR / "figures/downstream_cite_pcfm_rna_to_protein_gt"
TABLE_DIR = OUTPUT_DIR / "downstream/cite_pcfm_rna_to_protein_gt/tables"
MANIFEST_PATH = OUTPUT_DIR / "downstream/cite_pcfm_rna_to_protein_gt/rna_to_protein_gt_manifest.json"
INTERPRETATION_PATH = OUTPUT_DIR / "downstream/cite_pcfm_rna_to_protein_gt/interpretation.md"

FINAL_RUN = OUTPUT_DIR / "training_runs/20260526T174204321821Z__final__cite_paired_coord_fm__campaign_20260-trial_20260526-gse305370_ci"
METHODS = {
    "CITE-PCFM": {
        "trajectory": FINAL_RUN / "artifacts/evaluation_trajectory.npz",
        "metrics": FINAL_RUN / "artifacts/metrics.json",
        "type": "rna_driven_protein_readout",
        "color": "#d62728",
    },
    "CRUFM": {
        "trajectory": OUTPUT_DIR / "training_runs/20260525T220649168798Z__pilot__crufm__campaign_20260-stage2_c-baseline-crufm-gse305370_ci/artifacts/evaluation_trajectory.npz",
        "metrics": Path("/Users/zhenyizhang/.cellcompass/algorithm_benchmarks/datasets/gse305370_citeseq_stage2_10k/baseline_artifacts/builtin_crufm/metrics.json"),
        "type": "joint_latent_rollout_baseline",
        "color": "#1f77b4",
    },
    "balanced OT-CFM": {
        "trajectory": OUTPUT_DIR / "training_runs/20260525T215826526439Z__pilot__balanced_ot_cfm__campaign_20260-stage2_c-baseline-balanced_ot_-gse305370_ci/artifacts/evaluation_trajectory.npz",
        "metrics": Path("/Users/zhenyizhang/.cellcompass/algorithm_benchmarks/datasets/gse305370_citeseq_stage2_10k/baseline_artifacts/builtin_balanced_ot_cfm/metrics.json"),
        "type": "joint_latent_rollout_baseline",
        "color": "#2ca02c",
    },
}

RNA_DIM = 50
PROTEIN_DIM = 30
RNG = np.random.default_rng(20260527)
PHENOTYPES = ["HSC", "MasP", "EryP", "MoP", "MkP", "NeuP"]


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def metric_value(metrics: dict, name: str) -> float:
    if name in metrics:
        try:
            return float(metrics[name])
        except Exception:
            return np.nan
    custom = metrics.get("custom_metrics")
    if isinstance(custom, dict) and name in custom:
        try:
            return float(custom[name])
        except Exception:
            return np.nan
    return np.nan


def sample_indices(n: int, max_n: int) -> np.ndarray:
    if n <= max_n:
        return np.arange(n)
    return np.sort(RNG.choice(n, size=max_n, replace=False))


def weighted_choice(n: int, max_n: int, weights: np.ndarray | None = None) -> np.ndarray:
    if n <= max_n:
        return np.arange(n)
    prob = None
    if weights is not None:
        w = np.asarray(weights, dtype=float).reshape(-1)
        w = np.clip(w, 0.0, np.inf)
        if w.shape[0] == n and np.isfinite(w).all() and w.sum() > 0:
            prob = w / w.sum()
    return np.sort(RNG.choice(n, size=max_n, replace=False, p=prob))


def energy_distance(observed: np.ndarray, predicted: np.ndarray, weights: np.ndarray | None = None, max_points: int = 1200, seed: int = 0) -> float:
    observed = np.asarray(observed, dtype=np.float32)
    predicted = np.asarray(predicted, dtype=np.float32)
    if observed.ndim != 2 or predicted.ndim != 2 or observed.shape[0] == 0 or predicted.shape[0] == 0:
        return np.nan
    rng = np.random.default_rng(seed)
    if observed.shape[0] > max_points:
        observed = observed[rng.choice(observed.shape[0], size=max_points, replace=False)]
    if predicted.shape[0] > max_points:
        prob = None
        if weights is not None:
            w = np.asarray(weights, dtype=float).reshape(-1)
            w = np.clip(w, 0.0, np.inf)
            if w.shape[0] == predicted.shape[0] and np.isfinite(w).all() and w.sum() > 0:
                prob = w / w.sum()
        predicted = predicted[rng.choice(predicted.shape[0], size=max_points, replace=False, p=prob)]
    dxy = np.linalg.norm(observed[:, None, :] - predicted[None, :, :], axis=2).mean()
    dxx = np.linalg.norm(observed[:, None, :] - observed[None, :, :], axis=2).mean()
    dyy = np.linalg.norm(predicted[:, None, :] - predicted[None, :, :], axis=2).mean()
    return float(max(0.0, 2.0 * dxy - dxx - dyy))


def load_trajectory(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    time_points = np.asarray(z["time_points"], dtype=float)
    observed_times = np.asarray(z.get("observed_time_points", np.array([0, 1, 2, 3, 4], dtype=float)), dtype=float)
    observed_indices = np.asarray(z.get("observed_time_indices", np.array([0, 10, 20, 30, 40], dtype=int)), dtype=int)
    points = {}
    weights = {}
    for i, t in enumerate(time_points):
        pk = f"points_{i:06d}"
        wk = f"weights_{i:06d}"
        if pk in z.files:
            points[float(t)] = np.asarray(z[pk], dtype=np.float32)
        if wk in z.files:
            weights[float(t)] = np.asarray(z[wk], dtype=np.float32)
    return {
        "time_points": time_points,
        "observed_times": observed_times,
        "observed_indices": observed_indices,
        "points": points,
        "weights": weights,
    }


def nearest_time(times: Iterable[float], t: float) -> float:
    arr = np.asarray(list(times), dtype=float)
    return float(arr[np.argmin(np.abs(arr - float(t)))])


def weighted_mean(x: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if weights is None:
        return x.mean(axis=0)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if w.shape[0] != x.shape[0] or not np.isfinite(w).all() or w.sum() <= 0:
        return x.mean(axis=0)
    return np.average(x, axis=0, weights=w)


def cov_block_error(obs_rna: np.ndarray, obs_prot: np.ndarray, pred_rna: np.ndarray, pred_prot: np.ndarray, weights: np.ndarray | None = None) -> float:
    max_n = 1200
    oi = sample_indices(obs_rna.shape[0], max_n)
    pi = weighted_choice(pred_rna.shape[0], max_n, weights)
    obs_joint = np.concatenate([obs_rna[oi], obs_prot[oi]], axis=1)
    pred_joint = np.concatenate([pred_rna[pi], pred_prot[pi]], axis=1)
    obs_cov = np.cov(obs_joint, rowvar=False)
    pred_cov = np.cov(pred_joint, rowvar=False)
    rd = obs_rna.shape[1]
    obs_block = obs_cov[:rd, rd:]
    pred_block = pred_cov[:rd, rd:]
    return float(np.linalg.norm(obs_block - pred_block) / (np.linalg.norm(obs_block) + 1e-8))


def phenotype_affinity(protein: np.ndarray, centroid: np.ndarray, scale: np.ndarray) -> np.ndarray:
    z = (protein - centroid) / scale
    dist = np.linalg.norm(z, axis=1)
    return np.exp(-0.5 * dist)


def add_panel_label(ax, label: str) -> None:
    ax.text(
        -0.08,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="bottom",
        ha="right",
    )


def build_core_tables(adata: ad.AnnData, trajectories: Dict[str, dict]) -> dict:
    obs = adata.obs.copy()
    obs["time_point_processed"] = obs["time_point_processed"].astype(float)
    rna = np.asarray(adata.obsm["X_rna_svd"], dtype=np.float32)
    protein = np.asarray(adata.obsm["X_protein_pca"], dtype=np.float32)

    classes = list(pd.Categorical(obs["cell_type"]).categories)
    knn = KNeighborsClassifier(n_neighbors=25, weights="distance")
    knn.fit(protein, obs["cell_type"])
    centroids = {c: protein[np.asarray(obs["cell_type"] == c)].mean(axis=0) for c in classes}
    scale = protein.std(axis=0) + 1e-6

    per_time_rows = []
    phenotype_rows = []
    centroid_rows = []
    nearest_rows = []
    cov_rows = []
    variance_rows = []

    for method, traj in trajectories.items():
        for t in sorted(traj["observed_times"]):
            tt = nearest_time(traj["points"].keys(), float(t))
            pts = traj["points"][tt]
            w = traj["weights"].get(tt)
            pred_rna = pts[:, :RNA_DIM]
            pred_prot = pts[:, RNA_DIM : RNA_DIM + PROTEIN_DIM]
            obs_mask = np.asarray(obs["time_point_processed"] == float(t))
            obs_rna = rna[obs_mask]
            obs_prot = protein[obs_mask]
            obs_labels = np.asarray(obs["cell_type"])[obs_mask]
            pred_labels = knn.predict(pred_prot)

            pe = energy_distance(obs_prot, pred_prot, weights=w, seed=1000 + int(float(t) * 31))
            re = energy_distance(obs_rna, pred_rna, weights=w, seed=2000 + int(float(t) * 31))
            cov_err = cov_block_error(obs_rna, obs_prot, pred_rna, pred_prot, weights=w)
            nn = NearestNeighbors(n_neighbors=1).fit(obs_prot)
            pi = weighted_choice(pred_prot.shape[0], 1000, w)
            nearest_dist = nn.kneighbors(pred_prot[pi], return_distance=True)[0].reshape(-1)
            obs_var = obs_prot.var(axis=0)
            pred_var = pred_prot[weighted_choice(pred_prot.shape[0], 1200, w)].var(axis=0)
            var_l1 = float(np.mean(np.abs(pred_var - obs_var) / (obs_var + 1e-6)))
            pred_mean = weighted_mean(pred_prot, w)
            obs_mean = obs_prot.mean(axis=0)
            mean_dist = float(np.linalg.norm(pred_mean - obs_mean))

            per_time_rows.append(
                {
                    "method": method,
                    "time": float(t),
                    "protein_energy": pe,
                    "rna_energy": re,
                    "rna_protein_cov_error": cov_err,
                    "nearest_observed_protein_distance_mean": float(nearest_dist.mean()),
                    "nearest_observed_protein_distance_median": float(np.median(nearest_dist)),
                    "protein_variance_relative_l1": var_l1,
                    "protein_mean_displacement": mean_dist,
                    "method_type": METHODS[method]["type"],
                }
            )

            for c in PHENOTYPES:
                if c not in classes:
                    continue
                obs_frac = float(np.mean(obs_labels == c))
                pred_frac = float(np.mean(pred_labels == c))
                phenotype_rows.append(
                    {
                        "method": method,
                        "time": float(t),
                        "phenotype": c,
                        "observed_fraction": obs_frac,
                        "generated_fraction": pred_frac,
                        "fraction_abs_error": abs(pred_frac - obs_frac),
                    }
                )
                obs_c = obs_prot[obs_labels == c]
                pred_c = pred_prot[pred_labels == c]
                if obs_c.shape[0] >= 3 and pred_c.shape[0] >= 3:
                    centroid_rows.append(
                        {
                            "method": method,
                            "time": float(t),
                            "phenotype": c,
                            "observed_count": int(obs_c.shape[0]),
                            "generated_count": int(pred_c.shape[0]),
                            "centroid_displacement": float(np.linalg.norm(pred_c.mean(axis=0) - obs_c.mean(axis=0))),
                        }
                    )
                else:
                    centroid_rows.append(
                        {
                            "method": method,
                            "time": float(t),
                            "phenotype": c,
                            "observed_count": int(obs_c.shape[0]),
                            "generated_count": int(pred_c.shape[0]),
                            "centroid_displacement": np.nan,
                        }
                    )

            sample_dist = nearest_dist[: min(400, nearest_dist.shape[0])]
            for value in sample_dist:
                nearest_rows.append({"method": method, "time": float(t), "nearest_distance": float(value)})
            cov_rows.append({"method": method, "time": float(t), "rna_protein_cov_error": cov_err})
            variance_rows.append({"method": method, "time": float(t), "protein_variance_relative_l1": var_l1})

    tables = {
        "per_time": pd.DataFrame(per_time_rows),
        "phenotype": pd.DataFrame(phenotype_rows),
        "centroid": pd.DataFrame(centroid_rows),
        "nearest": pd.DataFrame(nearest_rows),
        "covariance": pd.DataFrame(cov_rows),
        "variance": pd.DataFrame(variance_rows),
        "classes": classes,
        "knn": knn,
        "centroids": centroids,
        "protein_scale": scale,
        "obs": obs,
        "rna": rna,
        "protein": protein,
    }
    return tables


def figure_01_protein_distribution(tables: dict, trajectories: Dict[str, dict]) -> Path:
    obs = tables["obs"]
    protein = tables["protein"]
    knn = tables["knn"]
    times = [0.0, 1.0, 2.0, 3.0, 4.0]
    columns = ["Observed"] + list(METHODS.keys())
    palette = dict(zip(tables["classes"], sns.color_palette("tab10", len(tables["classes"]))))
    fig, axes = plt.subplots(len(times), len(columns), figsize=(14, 14), sharex=True, sharey=True)
    per_time = tables["per_time"]
    for i, t in enumerate(times):
        obs_mask = np.asarray(obs["time_point_processed"] == t)
        oi = sample_indices(int(obs_mask.sum()), 650)
        obs_prot = protein[obs_mask][oi]
        obs_lab = np.asarray(obs["cell_type"])[obs_mask][oi]
        ax = axes[i, 0]
        for c in tables["classes"]:
            mm = obs_lab == c
            if mm.any():
                ax.scatter(obs_prot[mm, 0], obs_prot[mm, 1], s=7, alpha=0.65, color=palette[c], linewidths=0, label=c if i == 0 else None)
        ax.set_title(f"Observed target, t={t:g}")
        ax.set_ylabel("protein PC2")
        if i == 0:
            add_panel_label(ax, "A")
        for j, method in enumerate(METHODS.keys(), start=1):
            traj = trajectories[method]
            tt = nearest_time(traj["points"].keys(), t)
            pts = traj["points"][tt]
            w = traj["weights"].get(tt)
            idx = weighted_choice(pts.shape[0], 650, w)
            pred_prot = pts[idx, RNA_DIM : RNA_DIM + PROTEIN_DIM]
            pred_lab = knn.predict(pred_prot)
            ax = axes[i, j]
            for c in tables["classes"]:
                mm = pred_lab == c
                if mm.any():
                    ax.scatter(pred_prot[mm, 0], pred_prot[mm, 1], s=7, alpha=0.65, color=palette[c], linewidths=0)
            metric = per_time[(per_time.method == method) & (per_time.time == t)]["protein_energy"]
            metric_text = f"PE={float(metric.iloc[0]):.3f}" if len(metric) else ""
            ax.set_title(f"{method}, t={t:g}\n{metric_text}")
        for ax in axes[i, :]:
            ax.set_xlabel("protein PC1")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.suptitle("Observed target versus generated protein-coordinate distributions", y=0.995)
    fig.tight_layout(rect=[0, 0, 0.96, 0.98])
    out = FIG_DIR / "fig01_observed_vs_generated_protein_space.png"
    fig.savefig(out, dpi=250)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out


def figure_02_error_heatmap(tables: dict) -> Path:
    per_time = tables["per_time"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    metrics = [
        ("protein_energy", "Protein energy\n(lower is better)"),
        ("rna_protein_cov_error", "RNA-protein covariance error"),
        ("nearest_observed_protein_distance_mean", "Nearest observed protein distance"),
    ]
    for ax, (metric, title) in zip(axes, metrics):
        mat = per_time.pivot(index="method", columns="time", values=metric).loc[list(METHODS.keys())]
        sns.heatmap(mat, annot=True, fmt=".3f", cmap="viridis_r", ax=ax, cbar=True)
        ax.set_title(title)
        ax.set_xlabel("time")
        ax.set_ylabel("")
    add_panel_label(axes[0], "B")
    fig.tight_layout()
    out = FIG_DIR / "fig02_per_time_error_heatmaps.png"
    fig.savefig(out, dpi=300)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out


def figure_03_phenotype_fractions(tables: dict) -> Path:
    pheno = tables["phenotype"]
    phenotypes = [p for p in ["HSC", "MasP", "EryP", "MoP"] if p in set(pheno["phenotype"])]
    fig, axes = plt.subplots(1, len(phenotypes), figsize=(4.1 * len(phenotypes), 3.8), sharey=True)
    if len(phenotypes) == 1:
        axes = [axes]
    for ax, phenotype in zip(axes, phenotypes):
        d = pheno[pheno.phenotype == phenotype]
        obs_line = d.groupby("time")["observed_fraction"].mean().reset_index()
        ax.plot(obs_line.time, obs_line.observed_fraction, color="black", linewidth=2.4, marker="o", label="observed target")
        for method, spec in METHODS.items():
            md = d[d.method == method]
            ax.plot(md.time, md.generated_fraction, color=spec["color"], linewidth=1.6, marker="o", label=method)
        ax.set_title(phenotype)
        ax.set_xlabel("time")
        ax.set_ylabel("phenotype fraction")
        ax.set_ylim(0, max(0.75, float(d[["observed_fraction", "generated_fraction"]].max().max()) * 1.1))
    axes[-1].legend(frameon=False, fontsize=8, loc="upper right")
    add_panel_label(axes[0], "C")
    fig.suptitle("Observed and generated protein-phenotype fractions over time", y=1.03)
    fig.tight_layout()
    out = FIG_DIR / "fig03_observed_vs_generated_phenotype_fractions.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out


def figure_04_centroid_displacement(tables: dict) -> Path:
    centroid = tables["centroid"]
    d = centroid[centroid.phenotype.isin(["HSC", "MasP", "EryP", "MoP"])].copy()
    d["time_phenotype"] = d["time"].astype(int).astype(str) + ":" + d["phenotype"]
    mat = d.pivot_table(index="method", columns="time_phenotype", values="centroid_displacement", aggfunc="mean").loc[list(METHODS.keys())]
    fig, ax = plt.subplots(figsize=(13, 3.5))
    sns.heatmap(mat, annot=True, fmt=".2f", cmap="mako_r", ax=ax)
    ax.set_title("Protein phenotype centroid displacement to observed target")
    ax.set_xlabel("time:phenotype")
    ax.set_ylabel("")
    add_panel_label(ax, "D")
    fig.tight_layout()
    out = FIG_DIR / "fig04_centroid_displacement_heatmap.png"
    fig.savefig(out, dpi=300)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out


def build_rna_residual_table(tables: dict, trajectories: Dict[str, dict]) -> pd.DataFrame:
    obs = tables["obs"]
    rna = tables["rna"]
    protein = tables["protein"]
    centroids = tables["centroids"]
    scale = tables["protein_scale"]
    masp_centroid = centroids["MasP"]
    obs_masp = phenotype_affinity(protein, masp_centroid, scale)

    fit_blocks = []
    labels = []
    obs_idx = sample_indices(rna.shape[0], 3500)
    fit_blocks.append(rna[obs_idx])
    labels.extend([("observed", "", float(obs["time_point_processed"].iloc[i]), int(i)) for i in obs_idx])
    for method, traj in trajectories.items():
        for t in [1.0, 2.0, 3.0, 4.0]:
            tt = nearest_time(traj["points"].keys(), t)
            pts = traj["points"][tt]
            w = traj["weights"].get(tt)
            idx = weighted_choice(pts.shape[0], 350, w)
            fit_blocks.append(pts[idx, :RNA_DIM])
            labels.extend([(method, "generated", t, int(i)) for i in idx])
    fit_mat = np.vstack(fit_blocks)
    scaled = StandardScaler().fit_transform(fit_mat)
    if HAVE_UMAP:
        reducer = umap.UMAP(n_neighbors=35, min_dist=0.25, random_state=20260527)
        emb = reducer.fit_transform(scaled)
        map_name = "UMAP"
    else:
        reducer = PCA(n_components=2, random_state=20260527)
        emb = reducer.fit_transform(scaled)
        map_name = "PCA"

    rows = []
    offset = 0
    for block, meta in zip(fit_blocks, [labels[: len(obs_idx)]]):
        del block, meta
    for pos, (kind_a, kind_b, t, idx) in enumerate(labels):
        rows.append({"map1": emb[pos, 0], "map2": emb[pos, 1], "method": kind_a, "kind": kind_b, "time": t, "row": idx, "map": map_name})
    emb_df = pd.DataFrame(rows)

    residual_rows = []
    for method, traj in trajectories.items():
        for t in [1.0, 2.0, 3.0, 4.0]:
            obs_mask = np.asarray(obs["time_point_processed"] == t)
            nn = NearestNeighbors(n_neighbors=40).fit(rna[obs_mask])
            obs_aff = obs_masp[obs_mask]
            tt = nearest_time(traj["points"].keys(), t)
            pts = traj["points"][tt]
            w = traj["weights"].get(tt)
            idx = weighted_choice(pts.shape[0], 450, w)
            pred_rna = pts[idx, :RNA_DIM]
            pred_prot = pts[idx, RNA_DIM : RNA_DIM + PROTEIN_DIM]
            neigh = nn.kneighbors(pred_rna, return_distance=False)
            expected = obs_aff[neigh].mean(axis=1)
            generated = phenotype_affinity(pred_prot, masp_centroid, scale)
            map_part = emb_df[(emb_df.method == method) & (emb_df.kind == "generated") & (emb_df.time == t)]
            # Use nearest available map rows for the plotted subset; preserve residual values.
            for k in range(min(len(idx), len(map_part))):
                residual_rows.append(
                    {
                        "method": method,
                        "time": t,
                        "map1": float(map_part.iloc[k]["map1"]),
                        "map2": float(map_part.iloc[k]["map2"]),
                        "generated_masp_affinity": float(generated[k]),
                        "expected_observed_masp_affinity": float(expected[k]),
                        "residual": float(generated[k] - expected[k]),
                        "map": map_name,
                    }
                )
    residual_df = pd.DataFrame(residual_rows)
    residual_df.to_csv(TABLE_DIR / "rna_local_masp_residuals.csv", index=False)
    return residual_df


def figure_05_rna_residual_map(residual_df: pd.DataFrame) -> Path:
    times = [1.0, 2.0, 3.0, 4.0]
    methods = list(METHODS.keys())
    vmax = float(np.nanpercentile(np.abs(residual_df["residual"]), 98))
    fig, axes = plt.subplots(len(methods), len(times), figsize=(13, 8.5), sharex=True, sharey=True)
    for i, method in enumerate(methods):
        for j, t in enumerate(times):
            ax = axes[i, j]
            d = residual_df[(residual_df.method == method) & (residual_df.time == t)]
            sc = ax.scatter(d.map1, d.map2, c=d.residual, cmap="coolwarm", vmin=-vmax, vmax=vmax, s=9, linewidths=0, alpha=0.85)
            if i == 0:
                ax.set_title(f"t={t:g}")
            if j == 0:
                ax.set_ylabel(method)
            ax.set_xticks([])
            ax.set_yticks([])
    cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), fraction=0.025, pad=0.01)
    cbar.set_label("Generated MasP affinity - local observed expectation")
    add_panel_label(axes[0, 0], "E")
    fig.suptitle("RNA-local protein residuals: error map rather than raw generated signal", y=0.995)
    out = FIG_DIR / "fig05_rna_local_protein_residual_map.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out


def figure_06_input_neighborhood_terminal_fates(tables: dict, trajectories: Dict[str, dict]) -> Path:
    obs = tables["obs"]
    rna = tables["rna"]
    knn = tables["knn"]
    t0_mask = np.asarray(obs["time_point_processed"] == 0.0)
    kmeans = KMeans(n_clusters=8, random_state=20260527, n_init=20).fit(rna[t0_mask])
    rows = []
    for method, traj in trajectories.items():
        pts0 = traj["points"][nearest_time(traj["points"].keys(), 0.0)]
        pts4 = traj["points"][nearest_time(traj["points"].keys(), 4.0)]
        clusters = kmeans.predict(pts0[:, :RNA_DIM])
        pred_terminal = knn.predict(pts4[:, RNA_DIM : RNA_DIM + PROTEIN_DIM])
        for cluster_id in range(8):
            mask = clusters == cluster_id
            if mask.sum() == 0:
                continue
            for phenotype in ["HSC", "MasP", "EryP", "MoP"]:
                rows.append(
                    {
                        "method": method,
                        "initial_rna_neighborhood": cluster_id,
                        "terminal_phenotype": phenotype,
                        "fraction": float(np.mean(pred_terminal[mask] == phenotype)),
                    }
                )
    df = pd.DataFrame(rows)
    df.to_csv(TABLE_DIR / "initial_rna_neighborhood_terminal_protein_fates.csv", index=False)
    fig, axes = plt.subplots(1, len(METHODS), figsize=(13, 3.8), sharey=True)
    for ax, method in zip(axes, METHODS.keys()):
        d = df[df.method == method]
        mat = d.pivot(index="initial_rna_neighborhood", columns="terminal_phenotype", values="fraction").fillna(0)
        sns.heatmap(mat, vmin=0, vmax=1, cmap="rocket_r", ax=ax, cbar=ax is axes[-1], annot=True, fmt=".2f")
        ax.set_title(method)
        ax.set_xlabel("terminal protein phenotype")
        ax.set_ylabel("t0 RNA neighborhood")
    add_panel_label(axes[0], "F")
    fig.suptitle("Input-conditioned terminal protein phenotypes from RNA neighborhoods", y=1.03)
    fig.tight_layout()
    out = FIG_DIR / "fig06_rna_neighborhood_terminal_protein_fates.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out


def figure_07_covariance_error(tables: dict) -> Path:
    cov = tables["covariance"].pivot(index="method", columns="time", values="rna_protein_cov_error").loc[list(METHODS.keys())]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    sns.heatmap(cov, annot=True, fmt=".3f", cmap="flare_r", ax=ax)
    ax.set_title("RNA-protein covariance block mismatch")
    ax.set_xlabel("time")
    ax.set_ylabel("")
    add_panel_label(ax, "G")
    fig.tight_layout()
    out = FIG_DIR / "fig07_rna_protein_covariance_error.png"
    fig.savefig(out, dpi=300)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out


def figure_08_variance_matching(tables: dict) -> Path:
    var_df = tables["variance"]
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    sns.lineplot(data=var_df, x="time", y="protein_variance_relative_l1", hue="method", marker="o", ax=ax, palette={m: METHODS[m]["color"] for m in METHODS})
    ax.set_title("Protein-coordinate diversity mismatch")
    ax.set_ylabel("relative L1 variance error")
    ax.set_xlabel("time")
    add_panel_label(ax, "H")
    fig.tight_layout()
    out = FIG_DIR / "fig08_protein_diversity_variance_error.png"
    fig.savefig(out, dpi=300)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out


def figure_09_nearest_distance(tables: dict) -> Path:
    nearest = tables["nearest"]
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.boxplot(data=nearest, x="time", y="nearest_distance", hue="method", ax=ax, palette={m: METHODS[m]["color"] for m in METHODS}, showfliers=False)
    ax.set_title("Distance from generated protein states to nearest observed target protein state")
    ax.set_ylabel("nearest observed protein distance")
    ax.set_xlabel("time")
    add_panel_label(ax, "I")
    fig.tight_layout()
    out = FIG_DIR / "fig09_nearest_observed_protein_distance.png"
    fig.savefig(out, dpi=300)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out


def figure_10_summary_scorecard(tables: dict) -> Path:
    # Score the future prediction task only. t=0 is an initialization sanity
    # check and should not dominate the downstream interpretation.
    per_time = tables["per_time"]
    per_time = per_time[per_time["time"] > 0].copy()
    phenotype = tables["phenotype"]
    phenotype = phenotype[phenotype["time"] > 0].copy()
    centroid = tables["centroid"]
    centroid = centroid[centroid["time"] > 0].copy()
    summary_rows = []
    for method in METHODS:
        d = per_time[per_time.method == method]
        ph = phenotype[phenotype.method == method]
        ce = centroid[centroid.method == method]
        summary_rows.append(
            {
                "method": method,
                "protein_energy": d["protein_energy"].mean(),
                "phenotype_fraction_error": ph["fraction_abs_error"].mean(),
                "centroid_displacement": ce["centroid_displacement"].mean(),
                "covariance_error": d["rna_protein_cov_error"].mean(),
                "nearest_distance": d["nearest_observed_protein_distance_mean"].mean(),
                "variance_error": d["protein_variance_relative_l1"].mean(),
            }
        )
    score = pd.DataFrame(summary_rows)
    score.to_csv(TABLE_DIR / "summary_scorecard_raw.csv", index=False)
    metric_cols = [c for c in score.columns if c != "method"]
    norm = score.copy()
    for col in metric_cols:
        lo, hi = score[col].min(), score[col].max()
        norm[col] = 0.0 if hi == lo else (score[col] - lo) / (hi - lo)
    norm_long = norm.melt(id_vars="method", var_name="metric", value_name="normalized_error")
    norm.to_csv(TABLE_DIR / "summary_scorecard_normalized.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    sns.barplot(data=norm_long, x="metric", y="normalized_error", hue="method", ax=ax, palette={m: METHODS[m]["color"] for m in METHODS})
    ax.set_title("Normalized ground-truth mismatch scorecard")
    ax.set_ylabel("normalized error, lower is better")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=25)
    add_panel_label(ax, "J")
    fig.tight_layout()
    out = FIG_DIR / "fig10_normalized_gt_mismatch_scorecard.png"
    fig.savefig(out, dpi=300)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out


def write_outputs(tables: dict, residual_df: pd.DataFrame, figures: List[Path], trajectories: Dict[str, dict]) -> None:
    for name in ["per_time", "phenotype", "centroid", "nearest", "covariance", "variance"]:
        tables[name].to_csv(TABLE_DIR / f"{name}.csv", index=False)

    per_time = tables["per_time"]
    future_per_time = per_time[per_time["time"] > 0].copy()
    phenotype = tables["phenotype"]
    future_phenotype = phenotype[phenotype["time"] > 0].copy()
    best_protein = future_per_time.groupby("method")["protein_energy"].mean().sort_values()
    best_fraction = future_phenotype.groupby("method")["fraction_abs_error"].mean().sort_values()

    metrics = {}
    for method, spec in METHODS.items():
        raw = read_json(spec["metrics"])
        metrics[method] = {
            "metrics_path": str(spec["metrics"]),
            "protein_energy_from_metrics_json": metric_value(raw, "protein_energy"),
            "paired_cov_error_from_metrics_json": metric_value(raw, "paired_cov_error"),
            "method_type": spec["type"],
        }

    manifest = {
        "analysis": "CITE-PCFM RNA-to-protein ground-truth comparison",
        "dataset": str(DATA_PATH),
        "observed_features": {
            "rna": "obsm['X_rna_svd']",
            "protein": "obsm['X_protein_pca']",
            "time": "obs['time_point_processed']",
            "phenotype": "obs['cell_type']",
        },
        "state_contract": "Generated trajectory points are 80-dimensional [RNA(50), protein(30)]. CITE-PCFM protein coordinates are generated by the model protein readout; baselines are joint-latent rollout baselines unless separately audited.",
        "methods": {m: {"trajectory": str(spec["trajectory"]), "metrics": str(spec["metrics"]), "type": spec["type"]} for m, spec in METHODS.items()},
        "metric_context": metrics,
        "figures": [str(p) for p in figures],
        "tables": [str(p) for p in sorted(TABLE_DIR.glob("*.csv"))],
        "warnings": [
            "These are distributional comparisons against observed snapshots, not lineage-tracing ground truth.",
            "Named antibody columns are not available in the prepared benchmark; protein analyses use latent CITE protein PCA coordinates.",
            "Joint-latent baselines should not be described as RNA-only baselines unless a separate audit proves they did not use protein coordinates as state.",
        ],
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    text = [
        "# CITE-PCFM RNA-to-protein ground-truth comparison",
        "",
        "This analysis reframes the downstream evidence around observed target cells. The key question is whether generated protein-coordinate dynamics match the observed protein-coordinate distribution at the same time point.",
        "",
        "## Highest-level numeric readout",
        "",
        "Mean future-time protein_energy (t>0), lower is better:",
    ]
    for method, value in best_protein.items():
        text.append(f"- {method}: {value:.4f}")
    text.extend(["", "Mean future-time phenotype-fraction absolute error (t>0), lower is better:"])
    for method, value in best_fraction.items():
        text.append(f"- {method}: {value:.4f}")
    text.extend(
        [
            "",
            "## Figure guide",
            "",
            "- Fig. 01 shows observed target protein distributions side by side with generated protein distributions.",
            "- Fig. 02 summarizes per-time quantitative discrepancies.",
            "- Fig. 03 compares observed and generated protein-phenotype fractions over time.",
            "- Fig. 04 measures phenotype centroid displacement in protein space.",
            "- Fig. 05 maps local RNA-state protein residuals rather than raw generated signal.",
            "- Fig. 06 asks whether initial RNA neighborhoods produce different terminal protein phenotypes.",
            "- Fig. 07 checks RNA-protein covariance structure.",
            "- Fig. 08 checks protein-coordinate diversity.",
            "- Fig. 09 checks how close generated protein states are to observed target protein states.",
            "- Fig. 10 gives a normalized mismatch scorecard across metrics.",
            "",
            "## Interpretation caution",
            "",
            "The prepared benchmark stores joint X_latent=[RNA, protein]. CITE-PCFM is the only compared method here explicitly framed as RNA-driven protein readout, while the listed baselines are treated as joint-latent rollout baselines. Claims should be written accordingly.",
        ]
    )
    INTERPRETATION_PATH.write_text("\n".join(text) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    adata = ad.read_h5ad(DATA_PATH)
    trajectories = {method: load_trajectory(Path(spec["trajectory"])) for method, spec in METHODS.items()}
    tables = build_core_tables(adata, trajectories)
    figures = [
        figure_01_protein_distribution(tables, trajectories),
        figure_02_error_heatmap(tables),
        figure_03_phenotype_fractions(tables),
        figure_04_centroid_displacement(tables),
    ]
    residual_df = build_rna_residual_table(tables, trajectories)
    figures.extend(
        [
            figure_05_rna_residual_map(residual_df),
            figure_06_input_neighborhood_terminal_fates(tables, trajectories),
            figure_07_covariance_error(tables),
            figure_08_variance_matching(tables),
            figure_09_nearest_distance(tables),
            figure_10_summary_scorecard(tables),
        ]
    )
    write_outputs(tables, residual_df, figures, trajectories)
    print(json.dumps({"figures": [str(p) for p in figures], "manifest": str(MANIFEST_PATH), "interpretation": str(INTERPRETATION_PATH)}, indent=2))


if __name__ == "__main__":
    main()
