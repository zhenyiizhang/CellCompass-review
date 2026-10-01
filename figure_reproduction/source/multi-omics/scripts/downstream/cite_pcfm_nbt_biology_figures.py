#!/usr/bin/env python3
"""NBT-style CITE-PCFM biology figures focused on RNA-to-protein dynamics."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/cytobridge_mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/cytobridge_xdg_cache")
os.environ.setdefault("NUMBA_CACHE_DIR", "/private/tmp/numba_cache")
os.environ.setdefault("NUMBA_DISABLE_CACHE", "1")
for key in ["MPLCONFIGDIR", "XDG_CACHE_HOME", "NUMBA_CACHE_DIR"]:
    os.makedirs(os.environ[key], exist_ok=True)

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import StandardScaler


sns.set_theme(style="white", context="talk")

ROOT = Path("/Users/zhenyizhang/Documents/GitHub/CytoBridge-agent-runtime-v2-20260309/multi-omics")
DATA_PATH = Path("/Users/zhenyizhang/.cellcompass/algorithm_benchmarks/datasets/gse305370_citeseq_stage2_10k/data.h5ad")
TRAJ_PATH = ROOT / "training_runs/20260526T174204321821Z__final__cite_paired_coord_fm__campaign_20260-trial_20260526-gse305370_ci/artifacts/evaluation_trajectory.npz"
FIG_DIR = ROOT / "figures/cite_pcfm_deep_story/cite_biology_landscape"
TABLE_DIR = ROOT / "downstream/cite_pcfm_deep_story/tables"
MANIFEST_PATH = ROOT / "downstream/cite_pcfm_deep_story/rna_protein_nbt_biology_manifest.json"
STORYBOARD_PATH = ROOT / "downstream/cite_pcfm_deep_story/storyboard.md"

RNA_DIM = 50
PROT_DIM = 30
RNG = np.random.default_rng(20260527)
MOVIE_TIMES = [0.0, 1.0, 2.0, 3.0, 4.0]
PHENO_ORDER = ["HSC", "MasP", "NeuP", "EryP", "MkP", "MoP"]
PHENO_COLORS = {
    "HSC": "#2F855A",
    "MasP": "#D62728",
    "NeuP": "#E377C2",
    "EryP": "#F28E2B",
    "MkP": "#7B61B3",
    "MoP": "#8C564B",
    "BP": "#1F77B4",
}
TIME_COLORS = {
    0.0: "#2563EB",
    1.0: "#06B6D4",
    2.0: "#22C55E",
    3.0: "#F59E0B",
    4.0: "#E11D48",
}


def sample_idx(n: int, max_n: int, weights: np.ndarray | None = None) -> np.ndarray:
    if n <= max_n:
        return np.arange(n)
    prob = None
    if weights is not None:
        w = np.clip(np.asarray(weights, dtype=float).reshape(-1), 0, np.inf)
        if w.shape[0] == n and np.isfinite(w).all() and w.sum() > 0:
            prob = w / w.sum()
    return np.sort(RNG.choice(n, size=max_n, replace=False, p=prob))


def density_grid(xy: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray, sigma: float = 1.15) -> np.ndarray:
    hist, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=[x_edges, y_edges])
    hist = gaussian_filter(hist.T.astype(float), sigma=sigma)
    max_value = float(np.nanmax(hist))
    return hist / max_value if max_value > 0 else hist


def weighted_mean(x: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    if weights is None:
        return np.nanmean(x, axis=0)
    w = np.clip(np.asarray(weights, dtype=float).reshape(-1), 0, np.inf)
    if w.shape[0] != x.shape[0] or not np.isfinite(w).all() or w.sum() <= 0:
        return np.nanmean(x, axis=0)
    return np.average(x, axis=0, weights=w / w.sum())


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


def train_fate_readout(protein: np.ndarray, labels: np.ndarray) -> tuple[KNeighborsClassifier, pd.DataFrame]:
    x_train, x_test, y_train, y_test = train_test_split(protein, labels, test_size=0.25, random_state=20260527, stratify=labels)
    knn = KNeighborsClassifier(n_neighbors=25, weights="distance")
    knn.fit(x_train, y_train)
    pred = knn.predict(x_test)
    rows = [
        {
            "classifier": "protein_pca_knn25_distance",
            "class": "overall",
            "support": int(len(y_test)),
            "accuracy": float(accuracy_score(y_test, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
            "recall": np.nan,
            "used_for_main_fate": True,
            "note": "Nearest-observed protein phenotype readout used for generated fate assignment.",
        }
    ]
    for label in sorted(pd.unique(labels)):
        mask = y_test == label
        rows.append(
            {
                "classifier": "protein_pca_knn25_distance",
                "class": str(label),
                "support": int(mask.sum()),
                "accuracy": np.nan,
                "balanced_accuracy": np.nan,
                "recall": float(recall_score(y_test, pred, labels=[label], average="macro", zero_division=0)),
                "used_for_main_fate": str(label) != "BP",
                "note": "BP is retained in the audit table but excluded from biological interpretation because of low support/recall.",
            }
        )
    return knn, pd.DataFrame(rows)


def fit_embeddings(rna_obs, protein_obs, obs_time, time_points, traj_points, traj_weights):
    protein_fit_blocks = [protein_obs[sample_idx(protein_obs.shape[0], 6500)]]
    for time in MOVIE_TIMES:
        protein_fit_blocks.append(traj_points[time][:, RNA_DIM : RNA_DIM + PROT_DIM])
    protein_fit = np.vstack(protein_fit_blocks)
    protein_scaler = StandardScaler().fit(protein_fit)
    protein_umap = umap.UMAP(n_neighbors=35, min_dist=0.23, metric="euclidean", random_state=20260527)
    protein_umap.fit(protein_scaler.transform(protein_fit))
    obs_protein_umap = protein_umap.transform(protein_scaler.transform(protein_obs))
    generated_protein_umap = {}
    for time in time_points[::2]:
        generated_protein_umap[float(time)] = protein_umap.transform(protein_scaler.transform(traj_points[float(time)][:, RNA_DIM : RNA_DIM + PROT_DIM]))

    t0_mask = obs_time == 0.0
    rna_fit = np.vstack([rna_obs[t0_mask], traj_points[0.0][:, :RNA_DIM]])
    rna_scaler = StandardScaler().fit(rna_fit)
    rna_umap = umap.UMAP(n_neighbors=35, min_dist=0.25, metric="euclidean", random_state=20260528)
    rna_emb = rna_umap.fit_transform(rna_scaler.transform(rna_fit))
    obs_rna_t0_umap = rna_emb[: int(t0_mask.sum())]
    gen_rna_t0_umap = rna_emb[int(t0_mask.sum()) :]
    return {
        "protein_scaler": protein_scaler,
        "protein_umap": protein_umap,
        "obs_protein_umap": obs_protein_umap,
        "generated_protein_umap": generated_protein_umap,
        "obs_rna_t0_umap": obs_rna_t0_umap,
        "gen_rna_t0_umap": gen_rna_t0_umap,
    }


def figure_11_movie(payload: dict) -> Path:
    obs_emb = payload["obs_protein_umap"]
    obs_time = payload["obs_time"]
    gen_emb = payload["generated_protein_umap"]
    weights = payload["traj_weights"]
    all_xy = np.vstack([obs_emb] + [gen_emb[t] for t in MOVIE_TIMES])
    x0, x1 = np.nanpercentile(all_xy[:, 0], [0.5, 99.5])
    y0, y1 = np.nanpercentile(all_xy[:, 1], [0.5, 99.5])
    xlim = (x0 - 0.06 * (x1 - x0), x1 + 0.06 * (x1 - x0))
    ylim = (y0 - 0.08 * (y1 - y0), y1 + 0.08 * (y1 - y0))
    x_edges = np.linspace(xlim[0], xlim[1], 115)
    y_edges = np.linspace(ylim[0], ylim[1], 115)
    xx, yy = np.meshgrid(0.5 * (x_edges[:-1] + x_edges[1:]), 0.5 * (y_edges[:-1] + y_edges[1:]))
    obs_density = density_grid(obs_emb, x_edges, y_edges, sigma=1.15)

    fig = plt.figure(figsize=(18.8, 8.0))
    title_ax = fig.add_axes([0.045, 0.89, 0.91, 0.09])
    title_ax.axis("off")
    title_ax.text(0, 0.72, "RNA seeds generate a protein-state movie", fontsize=24, weight="bold", color="#0F172A")
    title_ax.text(0, 0.24, "CITE-PCFM starts from the same RNA-initialized particles and rolls them forward into a moving distribution on the observed CITE protein UMAP.", fontsize=11, color="#475569")

    left, bottom, w, h, gap = 0.045, 0.33, 0.168, 0.43, 0.018
    centroids = []
    for i, time in enumerate(MOVIE_TIMES):
        ax = fig.add_axes([left + i * (w + gap), bottom, w, h])
        color = TIME_COLORS[time]
        gd = density_grid(gen_emb[time], x_edges, y_edges, sigma=1.28)
        ax.contourf(xx, yy, obs_density, levels=[0.06, 0.14, 0.25, 0.42, 0.65, 1.01], colors=["#FFFFFF", "#F8FAFC", "#EEF2F7", "#DCE4EE", "#B7C4D5"], alpha=0.90)
        ax.contour(xx, yy, obs_density, levels=[0.18, 0.36, 0.58], colors="#8794A8", linewidths=0.7, alpha=0.55)
        ax.contourf(xx, yy, gd, levels=[0.12, 0.24, 0.42, 0.64, 0.84, 1.01], colors=[color] * 5, alpha=0.12)
        ax.contour(xx, yy, gd, levels=[0.20, 0.42, 0.66], colors=color, linewidths=[1.2, 1.8, 2.4])
        center = weighted_mean(gen_emb[time], weights.get(time))
        centroids.append(center)
        ax.scatter([center[0]], [center[1]], s=58, color=color, edgecolor="white", linewidth=1.2, zorder=10)
        ax.set_title(f"t{int(time)}", loc="left", fontsize=13, weight="bold", color="#111827")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#CBD5E1")
            spine.set_linewidth(0.9)
    ridge_ax = fig.add_axes([0.08, 0.09, 0.70, 0.16])
    centroids = np.vstack(centroids)
    axis = centroids[-1] - centroids[0]
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    projections = {time: (gen_emb[time] - centroids[0]) @ axis for time in MOVIE_TIMES}
    all_proj = np.concatenate([projections[t] for t in MOVIE_TIMES])
    bins = np.linspace(np.nanpercentile(all_proj, 0.5), np.nanpercentile(all_proj, 99.5), 100)
    centers = 0.5 * (bins[:-1] + bins[1:])
    for i, time in enumerate(MOVIE_TIMES):
        hist, _ = np.histogram(projections[time], bins=bins)
        hist = gaussian_filter1d(hist.astype(float), sigma=1.4)
        hist = hist / hist.max() if hist.max() > 0 else hist
        offset = i * 0.78
        color = TIME_COLORS[time]
        ridge_ax.fill_between(centers, offset, offset + hist * 0.55, color=color, alpha=0.25)
        ridge_ax.plot(centers, offset + hist * 0.55, color=color, lw=2.0)
        ridge_ax.text(centers[0], offset + 0.19, f"t{int(time)}", fontsize=9, color=color, weight="bold", va="center")
    ridge_ax.set_title("generated protein mass shifts along the movie coordinate", loc="left", fontsize=12.2, weight="bold")
    ridge_ax.set_xlabel("protein UMAP movie coordinate", fontsize=9.5)
    ridge_ax.set_yticks([])
    ridge_ax.tick_params(axis="x", labelsize=8)
    ridge_ax.spines[["top", "right", "left"]].set_visible(False)
    fig.text(0.81, 0.18, "Unique readout:\nRNA initialization is turned into\ncontinuous protein-state dynamics,\nnot only an endpoint label.", fontsize=13, color="#0F172A", weight="bold", ha="left", va="center")
    out = FIG_DIR / "bio_landscape11_rna_to_protein_umap_movie.png"
    fig.savefig(out, dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def figure_12_future_fate_map(payload: dict) -> Path:
    obs_rna = payload["obs_rna_t0_umap"]
    gen_rna = payload["gen_rna_t0_umap"]
    terminal_labels = payload["terminal_labels"]
    cluster = payload["cluster"]
    obs_labels_t0 = payload["obs_labels"][payload["obs_time"] == 0.0]
    neighborhood_colors = sns.color_palette("tab10", int(np.max(cluster)) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14.6, 5.4), gridspec_kw={"width_ratios": [1.08, 1.0]}, constrained_layout=True)
    axes[0].scatter(obs_rna[:, 0], obs_rna[:, 1], s=5, color="#CBD5E1", alpha=0.35, linewidths=0)
    for nb in sorted(np.unique(cluster)):
        mask = cluster == nb
        axes[0].scatter(gen_rna[mask, 0], gen_rna[mask, 1], s=10, color=neighborhood_colors[int(nb)], alpha=0.66, linewidths=0, label=f"N{nb}")
        center = gen_rna[mask].mean(axis=0)
        axes[0].text(center[0], center[1], f"N{nb}", fontsize=10, weight="bold", color="#111827", ha="center", va="center")
    axes[0].set_title("Initial RNA seed neighborhoods", loc="left", fontsize=13, weight="bold")
    axes[0].set_xlabel("RNA UMAP 1")
    axes[0].set_ylabel("RNA UMAP 2")
    axes[0].tick_params(labelsize=8)
    axes[0].spines[["top", "right"]].set_visible(False)

    frac_rows = []
    for nb in sorted(np.unique(cluster)):
        seed_mask = cluster == nb
        for pheno in PHENO_ORDER:
            frac_rows.append({"neighborhood": int(nb), "phenotype": pheno, "fraction": float(np.mean(terminal_labels[seed_mask] == pheno))})
    frac = pd.DataFrame(frac_rows)
    bottom = np.zeros(len(sorted(np.unique(cluster))))
    x = np.arange(len(sorted(np.unique(cluster))))
    for pheno in PHENO_ORDER:
        vals = frac[frac["phenotype"].eq(pheno)].sort_values("neighborhood")["fraction"].to_numpy()
        axes[1].bar(x, vals, bottom=bottom, color=PHENO_COLORS[pheno], edgecolor="white", linewidth=0.8, label=pheno)
        bottom += vals
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"N{nb}" for nb in sorted(np.unique(cluster))])
    axes[1].set_ylim(0, 1.0)
    axes[1].set_title("Future protein fate composition", loc="left", fontsize=13, weight="bold")
    axes[1].set_xlabel("initial RNA neighborhood")
    axes[1].set_ylabel("terminal protein-fate fraction")
    axes[1].tick_params(labelsize=8)
    axes[1].spines[["top", "right"]].set_visible(False)
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=PHENO_COLORS[p], markeredgecolor="none", markersize=6, label=p) for p in PHENO_ORDER]
    axes[1].legend(handles=handles, frameon=False, fontsize=8, ncol=3, loc="lower left")
    fig.suptitle("Initial RNA neighborhoods encode different future protein-fate mixtures", fontsize=18, weight="bold")
    frac.to_csv(TABLE_DIR / "nbt_rna_neighborhood_terminal_fate_composition.csv", index=False)
    out = FIG_DIR / "bio_landscape12_initial_rna_future_protein_fate_map.png"
    fig.savefig(out, dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def figure_13_commitment(payload: dict) -> Path:
    fate_df = payload["fate_df"]
    entropy_df = payload["entropy_df"]
    phenos = [p for p in PHENO_ORDER if p in set(fate_df["phenotype"])]
    mat = fate_df.groupby(["time", "phenotype"])["fraction"].mean().reset_index().pivot(index="time", columns="phenotype", values="fraction").reindex(columns=phenos).fillna(0).sort_index()
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 4.8), gridspec_kw={"width_ratios": [1.35, 1.0]}, constrained_layout=True)
    axes[0].stackplot(mat.index, [mat[p].to_numpy() for p in phenos], colors=[PHENO_COLORS[p] for p in phenos], labels=phenos, alpha=0.92, linewidth=0.9, edgecolor="white")
    axes[0].set_title("Protein fate composition over generated time", loc="left", fontsize=13, weight="bold")
    axes[0].set_xlabel("generated time")
    axes[0].set_ylabel("KNN-assigned protein fate")
    axes[0].legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
    global_entropy = entropy_df.groupby("time")["entropy"].mean().reset_index()
    axes[1].plot(global_entropy["time"], global_entropy["entropy"], color="#111827", lw=3.0, marker="o", label="mean neighborhood entropy")
    for nb in sorted(entropy_df["neighborhood"].unique()):
        d = entropy_df[entropy_df["neighborhood"].eq(nb)]
        axes[1].plot(d["time"], d["entropy"], color="#94A3B8", lw=1.1, alpha=0.50)
    axes[1].set_title("Commitment sharpens as entropy changes", loc="left", fontsize=13, weight="bold")
    axes[1].set_xlabel("generated time")
    axes[1].set_ylabel("protein-fate entropy")
    axes[1].spines[["top", "right"]].set_visible(False)
    fig.suptitle("The protein movie reveals a commitment window rather than a single endpoint", fontsize=18, weight="bold")
    out = FIG_DIR / "bio_landscape13_commitment_time_landscape.png"
    fig.savefig(out, dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def figure_14_velocity(payload: dict) -> Path:
    obs_emb = payload["obs_protein_umap"]
    gen_emb = payload["generated_protein_umap"]
    times = sorted(gen_emb)
    pair_times = [(times[i], times[i + 1]) for i in range(len(times) - 1)]
    starts = []
    vecs = []
    for t0, t1 in pair_times:
        p0 = gen_emb[t0]
        p1 = gen_emb[t1]
        idx = sample_idx(p0.shape[0], 350)
        starts.append(p0[idx])
        vecs.append(p1[idx] - p0[idx])
    starts = np.vstack(starts)
    vecs = np.vstack(vecs)
    all_xy = np.vstack([obs_emb, starts])
    x0, x1 = np.nanpercentile(all_xy[:, 0], [0.6, 99.4])
    y0, y1 = np.nanpercentile(all_xy[:, 1], [0.6, 99.4])
    xlim = (x0 - 0.06 * (x1 - x0), x1 + 0.06 * (x1 - x0))
    ylim = (y0 - 0.08 * (y1 - y0), y1 + 0.08 * (y1 - y0))
    x_edges = np.linspace(xlim[0], xlim[1], 100)
    y_edges = np.linspace(ylim[0], ylim[1], 100)
    xx_den, yy_den = np.meshgrid(0.5 * (x_edges[:-1] + x_edges[1:]), 0.5 * (y_edges[:-1] + y_edges[1:]))
    obs_den = density_grid(obs_emb, x_edges, y_edges, sigma=1.15)
    gx = np.linspace(xlim[0], xlim[1], 42)
    gy = np.linspace(ylim[0], ylim[1], 42)
    grid = np.array([(x, y) for y in gy for x in gx])
    nn = NearestNeighbors(n_neighbors=90).fit(starts)
    dist, ind = nn.kneighbors(grid)
    bw = np.nanmedian(dist[:, -1]) + 1e-6
    weights = np.exp(-0.5 * (dist / bw) ** 2)
    weights = weights / (weights.sum(axis=1, keepdims=True) + 1e-12)
    smooth_vec = np.einsum("ij,ijk->ik", weights, vecs[ind])
    speed = np.linalg.norm(smooth_vec, axis=1)
    u = smooth_vec[:, 0].reshape(len(gy), len(gx))
    v = smooth_vec[:, 1].reshape(len(gy), len(gx))
    speed_grid = speed.reshape(len(gy), len(gx))
    fig, ax = plt.subplots(figsize=(8.2, 6.6))
    ax.contourf(xx_den, yy_den, obs_den, levels=[0.05, 0.12, 0.22, 0.36, 0.55, 1.01], colors=["#FFFFFF", "#F8FAFC", "#EEF2F7", "#DCE4EE", "#B7C4D5"], alpha=0.95)
    ax.contour(xx_den, yy_den, obs_den, levels=[0.18, 0.36, 0.58], colors="#8794A8", linewidths=0.8, alpha=0.6)
    strm = ax.streamplot(gx, gy, u, v, color=speed_grid, cmap="magma", density=1.35, linewidth=1.3, arrowsize=1.1)
    cb = fig.colorbar(strm.lines, ax=ax, fraction=0.044, pad=0.02)
    cb.set_label("generated protein velocity speed")
    ax.set_title("Protein velocity field inferred from the generated movie", loc="left", fontsize=14, weight="bold")
    ax.set_xlabel("protein UMAP 1")
    ax.set_ylabel("protein UMAP 2")
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    out = FIG_DIR / "bio_landscape14_protein_velocity_field.png"
    fig.savefig(out, dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    grid_df = pd.DataFrame({"umap1": grid[:, 0], "umap2": grid[:, 1], "velocity_umap1": smooth_vec[:, 0], "velocity_umap2": smooth_vec[:, 1], "speed": speed})
    grid_df.to_csv(TABLE_DIR / "protein_velocity_field_grid.csv", index=False)
    return out


def build_fate_tables(time_points, traj_points, traj_weights, cluster, fate_labels_by_time):
    rows = []
    entropy_rows = []
    for time in time_points:
        labels = fate_labels_by_time[float(time)]
        for neighborhood in sorted(np.unique(cluster)):
            mask = cluster == neighborhood
            vals = pd.Series(labels[mask]).value_counts(normalize=True)
            entropy = -float(np.sum(vals.to_numpy() * np.log(vals.to_numpy() + 1e-12)) / np.log(max(2, len(vals)))) if len(vals) else np.nan
            entropy_rows.append({"time": float(time), "neighborhood": int(neighborhood), "entropy": entropy})
            for phenotype in sorted(set(PHENO_ORDER + ["BP"])):
                rows.append({"time": float(time), "neighborhood": int(neighborhood), "phenotype": phenotype, "fraction": float(np.mean(labels[mask] == phenotype))})
    return pd.DataFrame(rows), pd.DataFrame(entropy_rows)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    adata = ad.read_h5ad(DATA_PATH)
    obs = adata.obs.copy()
    obs_time = obs["time_point_processed"].astype(float).to_numpy()
    obs_labels = obs["cell_type"].astype(str).to_numpy()
    rna_obs = np.asarray(adata.obsm["X_rna_svd"], dtype=np.float32)
    protein_obs = np.asarray(adata.obsm["X_protein_pca"], dtype=np.float32)
    time_points, traj_points, traj_weights = load_trajectory()

    fate_knn, audit_df = train_fate_readout(protein_obs, obs_labels)
    audit_df.to_csv(TABLE_DIR / "fate_classifier_audit.csv", index=False)
    embeddings = fit_embeddings(rna_obs, protein_obs, obs_time, time_points, traj_points, traj_weights)
    fate_labels_by_time = {}
    for time in time_points:
        fate_labels_by_time[float(time)] = fate_knn.predict(traj_points[float(time)][:, RNA_DIM : RNA_DIM + PROT_DIM])
    terminal_labels = fate_labels_by_time[4.0]
    rna0 = traj_points[0.0][:, :RNA_DIM]
    order = np.argsort(rna0[:, 0])
    bins = np.array_split(order, 6)
    cluster = np.zeros(rna0.shape[0], dtype=int)
    for i, idx in enumerate(bins):
        cluster[idx] = i
    fate_df, entropy_df = build_fate_tables(time_points, traj_points, traj_weights, cluster, fate_labels_by_time)
    fate_df.to_csv(TABLE_DIR / "nbt_commitment_fate_fractions.csv", index=False)
    entropy_df.to_csv(TABLE_DIR / "nbt_commitment_entropy.csv", index=False)

    payload = {
        **embeddings,
        "obs_time": obs_time,
        "obs_labels": obs_labels,
        "traj_points": traj_points,
        "traj_weights": traj_weights,
        "terminal_labels": terminal_labels,
        "fate_df": fate_df,
        "entropy_df": entropy_df,
        "time_points": time_points,
        "cluster": cluster,
    }
    movie_rows = []
    for time in MOVIE_TIMES:
        emb = embeddings["generated_protein_umap"][time]
        labels = fate_labels_by_time[time]
        for i in range(emb.shape[0]):
            movie_rows.append({"time": time, "particle": i, "protein_umap1": emb[i, 0], "protein_umap2": emb[i, 1], "fate": labels[i]})
    pd.DataFrame(movie_rows).to_csv(TABLE_DIR / "nbt_protein_umap_movie_points.csv", index=False)
    seed_df = pd.DataFrame({
        "rna_umap1": embeddings["gen_rna_t0_umap"][:, 0],
        "rna_umap2": embeddings["gen_rna_t0_umap"][:, 1],
        "terminal_protein_fate": terminal_labels,
        "neighborhood": cluster,
    })
    seed_df.to_csv(TABLE_DIR / "nbt_rna_seed_future_protein_fate_points.csv", index=False)

    figures = [
        figure_11_movie(payload),
        figure_12_future_fate_map(payload),
        figure_13_commitment(payload),
        figure_14_velocity(payload),
    ]
    MANIFEST_PATH.write_text(
        json.dumps(
            {
                "figures": [str(p) for p in figures],
                "tables": {
                    "fate_classifier_audit": str(TABLE_DIR / "fate_classifier_audit.csv"),
                    "movie_points": str(TABLE_DIR / "nbt_protein_umap_movie_points.csv"),
                    "rna_seed_future_fate": str(TABLE_DIR / "nbt_rna_seed_future_protein_fate_points.csv"),
                    "commitment_fractions": str(TABLE_DIR / "nbt_commitment_fate_fractions.csv"),
                    "commitment_entropy": str(TABLE_DIR / "nbt_commitment_entropy.csv"),
                    "velocity_field": str(TABLE_DIR / "protein_velocity_field_grid.csv"),
                },
                "fate_standard": "KNN-25 distance-weighted classifier trained on held-out audited observed protein PCA; BP retained in audit but excluded from main biological interpretation.",
            },
            indent=2,
        )
    )
    if STORYBOARD_PATH.exists():
        text = STORYBOARD_PATH.read_text()
        marker = "### Flagship Figure\n"
        insert = (
            "### Planned NBT Biology Figure Set\n\n"
            "- `bio_landscape11_rna_to_protein_umap_movie`: CITE-PCFM-only RNA-to-protein movie on observed protein UMAP.\n"
            "- `bio_landscape12_initial_rna_future_protein_fate_map`: initial RNA seeds colored by future protein fate.\n"
            "- `bio_landscape13_commitment_time_landscape`: generated protein-fate composition and entropy over time.\n"
            "- `bio_landscape14_protein_velocity_field`: kernel-smoothed velocity field from generated protein UMAP trajectories.\n\n"
        )
        if "bio_landscape11_rna_to_protein_umap_movie" not in text:
            text = text.replace(marker, insert + marker)
            STORYBOARD_PATH.write_text(text)
    print(json.dumps({"figures": [str(p) for p in figures], "manifest": str(MANIFEST_PATH)}, indent=2))


if __name__ == "__main__":
    main()
