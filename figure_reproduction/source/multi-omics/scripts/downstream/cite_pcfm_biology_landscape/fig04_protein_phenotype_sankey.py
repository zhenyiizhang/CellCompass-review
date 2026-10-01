#!/usr/bin/env python3
"""Protein phenotype transition Sankey for CITE-PCFM rollout particles.

The plot tracks the same RNA-initialized CITE-PCFM particles across the five
biological time anchors. At each time, the generated protein block is assigned
to an observed protein-PCA phenotype basin by a fixed MLP probability readout. The ribbons
show weighted hard-MAP phenotype transitions between adjacent time anchors.

This is not an observed lineage-tracing result and not a multimodal cell-type
transition. It is a model-derived protein phenotype transition landscape.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/cytobridge_mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/cytobridge_xdg_cache")
os.environ.setdefault("NUMBA_CACHE_DIR", "/private/tmp/numba_cache")
os.environ.setdefault("NUMBA_DISABLE_CACHE", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")
for key in ["MPLCONFIGDIR", "XDG_CACHE_HOME", "NUMBA_CACHE_DIR"]:
    Path(os.environ[key]).mkdir(parents=True, exist_ok=True)

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Rectangle
from multistageot_style import PHENO_COLORS, PHENO_ORDER
from protein_phenotype_readout import ProteinPhenotypeReadout, train_mlp_protein_phenotype_readout


ROOT = Path("/Users/zhenyizhang/Documents/GitHub/CytoBridge-agent-runtime-v2-20260309/multi-omics")
DATA_PATH = Path("/Users/zhenyizhang/.cellcompass/algorithm_benchmarks/datasets/gse305370_citeseq_stage2_10k/data.h5ad")
TRAJECTORY_PATH = ROOT / "training_runs/20260526T174204321821Z__final__cite_paired_coord_fm__campaign_20260-trial_20260526-gse305370_ci/artifacts/evaluation_trajectory.npz"

FIG_DIR = ROOT / "figures/cite_pcfm_deep_story/biology_landscape_discovery"
OUT_DIR = ROOT / "downstream/cite_pcfm_deep_story/biology_landscape_discovery_protein_phenotype_sankey"

RNA_DIM = 50
PROTEIN_DIM = 30
RANDOM_STATE = 20260528
BIOLOGICAL_TIMES = [0.0, 1.0, 2.0, 3.0, 4.0]
MIN_RIBBON_FRACTION = 0.0015


def load_trajectory(path: Path) -> dict[str, object]:
    z = np.load(path)
    times = np.asarray(z["time_points"], dtype=float)
    points: dict[float, np.ndarray] = {}
    weights: dict[float, np.ndarray] = {}
    for i, time in enumerate(times):
        pkey = f"points_{i:06d}"
        wkey = f"weights_{i:06d}"
        if pkey in z.files:
            points[float(time)] = np.asarray(z[pkey], dtype=np.float32)
        if wkey in z.files:
            weights[float(time)] = np.asarray(z[wkey], dtype=np.float64)
    return {"time_points": times, "points": points, "weights": weights}


def nearest_time(available: np.ndarray, target: float) -> float:
    available = np.asarray(available, dtype=float)
    return float(available[np.argmin(np.abs(available - target))])


def normalized_weights(weights: np.ndarray | None, n: int) -> np.ndarray:
    if weights is None:
        return np.full(n, 1.0 / n, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if w.shape[0] != n or not np.isfinite(w).all():
        return np.full(n, 1.0 / n, dtype=np.float64)
    w = np.clip(w, 0.0, np.inf)
    total = float(w.sum())
    return w / total if total > 0 else np.full(n, 1.0 / n, dtype=np.float64)


def class_probability_matrix(readout: ProteinPhenotypeReadout, protein: np.ndarray) -> np.ndarray:
    return readout.predict_proba_ordered(protein)


def time_labels(adata: ad.AnnData, times: np.ndarray) -> list[str]:
    labels = []
    obs_time = adata.obs["time"].astype(str).to_numpy() if "time" in adata.obs else None
    for time in BIOLOGICAL_TIMES:
        label = f"t{int(time)}"
        if obs_time is not None:
            vals = pd.Series(obs_time[np.isclose(times, time)]).dropna()
            if not vals.empty:
                label = str(vals.mode().iloc[0])
        labels.append(label)
    return labels


def assign_particle_states(
    trajectory: dict[str, object],
    readout: ProteinPhenotypeReadout,
) -> tuple[pd.DataFrame, dict[float, np.ndarray], np.ndarray, dict[float, np.ndarray]]:
    time_points = np.asarray(trajectory["time_points"], dtype=float)
    points_by_time: dict[float, np.ndarray] = trajectory["points"]  # type: ignore[assignment]
    weights_by_time: dict[float, np.ndarray] = trajectory["weights"]  # type: ignore[assignment]
    first_time = nearest_time(time_points, BIOLOGICAL_TIMES[0])
    n_particles = points_by_time[first_time].shape[0]
    base_weights = normalized_weights(weights_by_time.get(first_time), n_particles)

    hard_by_time: dict[float, np.ndarray] = {}
    prob_by_time: dict[float, np.ndarray] = {}
    table = pd.DataFrame({"particle_id": np.arange(n_particles, dtype=int), "base_weight": base_weights})

    for display_time in BIOLOGICAL_TIMES:
        t = nearest_time(time_points, display_time)
        state = points_by_time[t]
        if state.shape[0] != n_particles:
            raise ValueError(f"particle count changed at time {t}: {state.shape[0]} != {n_particles}")
        protein = np.asarray(state[:, RNA_DIM : RNA_DIM + PROTEIN_DIM], dtype=np.float32)
        probs = class_probability_matrix(readout, protein)
        hard = np.asarray(PHENO_ORDER, dtype=object)[np.argmax(probs, axis=1)]
        hard_by_time[display_time] = hard
        prob_by_time[display_time] = probs
        table[f"t{int(display_time)}_trajectory_time"] = t
        table[f"t{int(display_time)}_protein_phenotype"] = hard
        table[f"t{int(display_time)}_phenotype_probability"] = probs.max(axis=1)
    return table, hard_by_time, base_weights, prob_by_time


def transition_table(hard_by_time: dict[float, np.ndarray], weights: np.ndarray) -> pd.DataFrame:
    rows = []
    for i in range(len(BIOLOGICAL_TIMES) - 1):
        t0 = BIOLOGICAL_TIMES[i]
        t1 = BIOLOGICAL_TIMES[i + 1]
        source = hard_by_time[t0]
        target = hard_by_time[t1]
        for src in PHENO_ORDER:
            src_mask = source == src
            for dst in PHENO_ORDER:
                mask = src_mask & (target == dst)
                mass = float(weights[mask].sum())
                rows.append(
                    {
                        "time_from": t0,
                        "time_to": t1,
                        "source_phenotype": src,
                        "target_phenotype": dst,
                        "weight_fraction": mass,
                        "n_particles": int(mask.sum()),
                    }
                )
    return pd.DataFrame(rows)


def node_mass_table(hard_by_time: dict[float, np.ndarray], weights: np.ndarray) -> pd.DataFrame:
    rows = []
    for time in BIOLOGICAL_TIMES:
        labels = hard_by_time[time]
        for pheno in PHENO_ORDER:
            mask = labels == pheno
            rows.append(
                {
                    "time": time,
                    "phenotype": pheno,
                    "weight_fraction": float(weights[mask].sum()),
                    "n_particles": int(mask.sum()),
                }
            )
    return pd.DataFrame(rows)


def compute_node_positions(node_df: pd.DataFrame) -> dict[float, dict[str, tuple[float, float]]]:
    positions: dict[float, dict[str, tuple[float, float]]] = {}
    for time in BIOLOGICAL_TIMES:
        cursor = 0.0
        positions[time] = {}
        sub = node_df[np.isclose(node_df["time"].astype(float), time)].set_index("phenotype")
        for pheno in PHENO_ORDER:
            mass = float(sub.loc[pheno, "weight_fraction"]) if pheno in sub.index else 0.0
            positions[time][pheno] = (cursor, cursor + mass)
            cursor += mass
    return positions


def compute_flow_segments(
    flow_df: pd.DataFrame,
    node_pos: dict[float, dict[str, tuple[float, float]]],
) -> list[dict[str, object]]:
    segments: list[dict[str, object]] = []
    for i in range(len(BIOLOGICAL_TIMES) - 1):
        t0 = BIOLOGICAL_TIMES[i]
        t1 = BIOLOGICAL_TIMES[i + 1]
        sub = flow_df[np.isclose(flow_df["time_from"].astype(float), t0)]

        src_cursor = {pheno: node_pos[t0][pheno][0] for pheno in PHENO_ORDER}
        dst_cursor = {pheno: node_pos[t1][pheno][0] for pheno in PHENO_ORDER}

        for src in PHENO_ORDER:
            for dst in PHENO_ORDER:
                row = sub[(sub["source_phenotype"] == src) & (sub["target_phenotype"] == dst)]
                if row.empty:
                    continue
                mass = float(row["weight_fraction"].iloc[0])
                if mass <= 0:
                    continue
                y0a = src_cursor[src]
                y0b = y0a + mass
                src_cursor[src] = y0b
                y1a = dst_cursor[dst]
                y1b = y1a + mass
                dst_cursor[dst] = y1b
                segments.append(
                    {
                        "time_from": t0,
                        "time_to": t1,
                        "source": src,
                        "target": dst,
                        "mass": mass,
                        "y0a": y0a,
                        "y0b": y0b,
                        "y1a": y1a,
                        "y1b": y1b,
                    }
                )
    return segments


def draw_ribbon(ax: plt.Axes, x0: float, x1: float, y0a: float, y0b: float, y1a: float, y1b: float, color: str, alpha: float) -> None:
    curvature = 0.45 * (x1 - x0)
    verts = [
        (x0, y0a),
        (x0 + curvature, y0a),
        (x1 - curvature, y1a),
        (x1, y1a),
        (x1, y1b),
        (x1 - curvature, y1b),
        (x0 + curvature, y0b),
        (x0, y0b),
        (x0, y0a),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    patch = PathPatch(MplPath(verts, codes), facecolor=color, edgecolor="none", alpha=alpha, zorder=1)
    ax.add_patch(patch)


def plot_sankey(
    node_df: pd.DataFrame,
    flow_df: pd.DataFrame,
    time_names: list[str],
    classifier_metrics: dict[str, object],
    out_png: Path,
) -> None:
    node_pos = compute_node_positions(node_df)
    segments = compute_flow_segments(flow_df, node_pos)
    x_by_time = {time: float(i) for i, time in enumerate(BIOLOGICAL_TIMES)}
    node_width = 0.10

    fig, ax = plt.subplots(figsize=(13.8, 6.2), constrained_layout=True)

    for seg in sorted(segments, key=lambda x: float(x["mass"])):
        mass = float(seg["mass"])
        if mass < MIN_RIBBON_FRACTION:
            continue
        source = str(seg["source"])
        alpha = 0.18 + 0.42 * min(1.0, mass / 0.12)
        draw_ribbon(
            ax,
            x_by_time[float(seg["time_from"])] + node_width / 2,
            x_by_time[float(seg["time_to"])] - node_width / 2,
            float(seg["y0a"]),
            float(seg["y0b"]),
            float(seg["y1a"]),
            float(seg["y1b"]),
            PHENO_COLORS[source],
            alpha,
        )

    for time_i, time in enumerate(BIOLOGICAL_TIMES):
        x = x_by_time[time]
        for pheno in PHENO_ORDER:
            y0, y1 = node_pos[time][pheno]
            h = y1 - y0
            if h <= 0:
                continue
            rect = Rectangle(
                (x - node_width / 2, y0),
                node_width,
                h,
                facecolor=PHENO_COLORS[pheno],
                edgecolor="white",
                linewidth=0.8,
                zorder=4,
            )
            ax.add_patch(rect)
            if h >= 0.035:
                ax.text(
                    x,
                    y0 + h / 2,
                    f"{pheno}\n{h * 100:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=7.2,
                    color="white" if pheno not in {"EryP", "NeuP"} else "#0F172A",
                    weight="bold",
                    zorder=5,
                )
        ax.text(x, 1.035, time_names[time_i], ha="center", va="bottom", fontsize=10.5, weight="bold", color="#0F172A")
        ax.axvline(x, color="#CBD5E1", lw=0.7, alpha=0.5, zorder=0)

    ax.set_xlim(-0.35, len(BIOLOGICAL_TIMES) - 0.65)
    ax.set_ylim(0.0, 1.075)
    ax.set_xticks([])
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_yticklabels([f"{int(v * 100)}%" for v in np.linspace(0, 1, 6)], fontsize=8)
    ax.set_ylabel("weighted particle fraction", fontsize=9)
    ax.grid(axis="y", color="#E2E8F0", lw=0.7, alpha=0.8, zorder=0)
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")

    handles = [
        Line2D([0], [0], marker="s", linestyle="None", markerfacecolor=PHENO_COLORS[p], markeredgecolor="none", markersize=9, label=p)
        for p in PHENO_ORDER
    ]
    ax.legend(handles=handles, title="Protein phenotype", frameon=False, fontsize=8.3, title_fontsize=8.8, loc="center left", bbox_to_anchor=(1.01, 0.5))

    acc = classifier_metrics["accuracy"]
    bal_acc = classifier_metrics["balanced_accuracy"]
    ax.set_title("CITE-PCFM generated protein phenotype transitions", loc="left", fontsize=15.0, weight="bold")
    ax.text(
        0.012,
        0.945,
        "same RNA-initialized particles tracked across generated protein states; labels are protein-PCA MLP hard-MAP basins",
        transform=ax.transAxes,
        va="top",
        fontsize=8.8,
        color="#334155",
        bbox={"boxstyle": "round,pad=0.16", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
    )
    fig.text(
        0.01,
        -0.012,
        f"Ribbons below {MIN_RIBBON_FRACTION * 100:.2f}% are omitted from the visual but retained in tables. Protein readout holdout accuracy={acc:.3f}, balanced accuracy={bal_acc:.3f}.",
        fontsize=8.7,
        color="#475569",
    )
    fig.savefig(out_png, dpi=330, bbox_inches="tight", facecolor="white")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> dict[str, object]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(DATA_PATH)
    protein = np.asarray(adata.obsm["X_protein_pca"], dtype=np.float32)[:, :PROTEIN_DIM]
    labels = adata.obs["cell_type"].astype(str).to_numpy()
    times = np.asarray(adata.obs["time_point_processed"], dtype=float)

    readout, classifier_metrics, confusion_df = train_mlp_protein_phenotype_readout(
        protein, labels, PHENO_ORDER, RANDOM_STATE
    )
    trajectory = load_trajectory(TRAJECTORY_PATH)
    particle_df, hard_by_time, base_weights, prob_by_time = assign_particle_states(trajectory, readout)
    node_df = node_mass_table(hard_by_time, base_weights)
    flow_df = transition_table(hard_by_time, base_weights)

    particle_path = OUT_DIR / "protein_phenotype_sankey_particle_states.csv"
    node_path = OUT_DIR / "protein_phenotype_sankey_node_masses.csv"
    flow_path = OUT_DIR / "protein_phenotype_sankey_flows.csv"
    confusion_path = OUT_DIR / "protein_phenotype_sankey_readout_holdout_confusion.csv"
    particle_df.to_csv(particle_path, index=False)
    node_df.to_csv(node_path, index=False)
    flow_df.to_csv(flow_path, index=False)
    confusion_df.to_csv(confusion_path)

    out_png = FIG_DIR / "fig04_protein_phenotype_sankey.png"
    plot_sankey(node_df, flow_df, time_labels(adata, times), classifier_metrics, out_png)

    terminal_counts = (
        node_df[np.isclose(node_df["time"].astype(float), BIOLOGICAL_TIMES[-1])]
        .set_index("phenotype")["weight_fraction"]
        .reindex(PHENO_ORDER)
        .fillna(0.0)
        .to_dict()
    )
    manifest = {
        "analysis": "protein_phenotype_transition_sankey",
        "data_h5ad": str(DATA_PATH),
        "trajectory_path": str(TRAJECTORY_PATH),
        "figure": str(out_png),
        "tables": {
            "particle_states": str(particle_path),
            "node_masses": str(node_path),
            "flows": str(flow_path),
            "holdout_confusion": str(confusion_path),
        },
        "semantic_contract": {
            "transition_units": "same CITE-PCFM rollout particles tracked across generated protein states",
            "phenotype_assignment": "hard MAP from observed-protein-PCA MLP class probabilities",
            "weighting": "t0 trajectory particle weights reused across all time anchors to conserve particle mass",
            "not_claimed": "not observed lineage tracing and not multimodal cell-type transition",
        },
        "readout": {
            "feature_space": "trajectory state[:, 50:80] classified against observed adata.obsm['X_protein_pca']",
            "classifier": "StandardScaler + MLPClassifier",
            "hidden_layer_sizes": [64, 32],
            "timepoints": BIOLOGICAL_TIMES,
            "classes": PHENO_ORDER,
            "min_ribbon_fraction_visual_only": MIN_RIBBON_FRACTION,
        },
        "classifier_holdout_metrics": classifier_metrics,
        "terminal_generated_hard_fraction": terminal_counts,
    }
    manifest_path = OUT_DIR / "protein_phenotype_sankey_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    main()
