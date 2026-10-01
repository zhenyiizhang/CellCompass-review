#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cellcompass_pulsebridge_perturbation")
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/cellcompass_pulsebridge_perturbation_numba")

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


REPO = Path(__file__).resolve().parents[3]
BUNDLE = REPO / "downloads" / "pulsebridge_welltemp_hct116_20260627"
OUT = REPO / "analysis" / "pulsebridge_story" / "outputs"
FIG = OUT / "figures"
TAB = OUT / "tables"
ADATA_PATH = (
    BUNDLE
    / "home"
    / "ubuntu"
    / ".cellcompass"
    / "algorithm_benchmarks"
    / "datasets"
    / "pcib_welltemp_hct116_raw_pool_20k"
    / "data.h5ad"
)
TRAJ_PATH = (
    BUNDLE
    / "data"
    / "cytobridge"
    / "runs"
    / "night_wave_20260621_0140"
    / "existing_deep_raw_a"
    / "training_runs"
    / "20260620T234044691964Z__pilot__pulse_control_identifiability_bridge__campaign_20260-trial_20260620-pcib_welltem"
    / "artifacts"
    / "evaluation_trajectory.npz"
)

BLUE = "#0E6F8E"
RED = "#B23A48"
GREY = "#777777"
STATE_COLORS = ["#2F5D8C", "#5F8FB8", "#4BA3A1", "#A6A053", "#A17C00", "#8A4F9E"]

GENES = [
    "DKK1",
    "GAS5",
    "B2M",
    "HLA-A",
    "HLA-B",
    "HLA-C",
    "IFITM3",
    "EREG",
    "PCNA",
    "SAT1",
    "LGALS1",
    "MT-CO1",
    "MT-CO2",
    "MT-CO3",
    "MT-ND4",
    "MT-CYB",
    "HSP90AB1",
    "HSP90AA1",
    "HSPD1",
    "FTL",
]

MODULES = {
    "source_aligned_antitumor": {
        "label": "source-aligned\nanti-tumor",
        "up": ["DKK1", "GAS5", "B2M", "HLA-A", "HLA-B", "HLA-C", "IFITM3"],
        "down": ["EREG", "PCNA"],
        "color": "#2F5D8C",
    },
    "reactivation": {
        "label": "DKK1/GAS5\nreactivation",
        "up": ["DKK1", "GAS5"],
        "down": [],
        "color": "#2F5D8C",
    },
    "immune_surface": {
        "label": "B2M/HLA/IFITM3\nimmune surface",
        "up": ["B2M", "HLA-A", "HLA-B", "HLA-C", "IFITM3"],
        "down": [],
        "color": "#1B8A8F",
    },
    "proliferation_repression": {
        "label": "EREG/PCNA\nproliferation down",
        "up": [],
        "down": ["EREG", "PCNA"],
        "color": "#2F5D8C",
    },
    "adaptive_metabolic": {
        "label": "SAT1/LGALS1/MT\nadaptive-metabolic",
        "up": ["SAT1", "LGALS1", "MT-CO1", "MT-CO2", "MT-CO3", "MT-ND4", "MT-CYB"],
        "down": [],
        "color": "#A17C00",
    },
    "proteostasis_iron_reduction": {
        "label": "HSP/FTL\nproteostasis/iron down",
        "up": [],
        "down": ["HSP90AB1", "HSP90AA1", "HSPD1", "FTL"],
        "color": "#777777",
    },
}

GENE_PLOT_ORDER = ["SAT1", "MT-CO1", "LGALS1", "B2M", "DKK1", "PCNA", "HSP90AB1", "FTL", "EREG"]
GENE_COLORS = {
    "SAT1": "#A17C00",
    "MT-CO1": "#6B7F2A",
    "LGALS1": "#8A4F9E",
    "B2M": "#1B8A8F",
    "DKK1": "#2F5D8C",
    "PCNA": "#2F5D8C",
    "HSP90AB1": "#777777",
    "FTL": "#777777",
    "EREG": "#2F5D8C",
}


def dense(x):
    return x.toarray() if sparse.issparse(x) else np.asarray(x)


def load_all_trajectory() -> tuple[np.ndarray, list[np.ndarray]]:
    z = np.load(TRAJ_PATH)
    time_points = np.asarray(z["time_points"], dtype=float)
    points = [np.asarray(z[f"points_{i:06d}"], dtype=np.float32) for i in range(len(time_points))]
    return time_points, points


def build_response_axis(latent: np.ndarray, times: np.ndarray) -> dict:
    pca = PCA(n_components=5, random_state=7)
    scores = pca.fit_transform(latent)
    cors = []
    for j in range(scores.shape[1]):
        c = np.corrcoef(scores[:, j], times)[0, 1]
        cors.append(0.0 if np.isnan(c) else c)
    axis_idx = int(np.argmax(np.abs(cors)))
    sign = -1.0 if cors[axis_idx] < 0 else 1.0
    axis_values = sign * scores[:, axis_idx]
    edges = np.quantile(axis_values, np.linspace(0, 1, 7))
    edges[0] -= 1e-6
    edges[-1] += 1e-6
    return {"pca": pca, "axis_idx": axis_idx, "sign": sign, "edges": edges}


def assign_states(points: np.ndarray, axis: dict) -> np.ndarray:
    vals = axis["sign"] * axis["pca"].transform(points)[:, axis["axis_idx"]]
    return np.clip(np.digitize(vals, axis["edges"][1:-1], right=False), 0, 5)


def selected_expression(adata: ad.AnnData) -> tuple[list[str], np.ndarray, np.ndarray]:
    present = [g for g in GENES if g in set(adata.var_names)]
    idx = [int(adata.var_names.get_loc(g)) for g in present]
    layer = adata.layers["log1p_norm"] if "log1p_norm" in adata.layers else adata.X
    expr = dense(layer[:, idx]).astype(np.float32)
    mu = expr.mean(axis=0, keepdims=True)
    sigma = expr.std(axis=0, keepdims=True)
    zexpr = (expr - mu) / np.maximum(sigma, 1e-6)
    return present, expr, zexpr


def neighbor_readout(
    nn: NearestNeighbors,
    points: np.ndarray,
    expr: np.ndarray,
    zexpr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    _dist, ind = nn.kneighbors(points)
    return expr[ind].mean(axis=1), zexpr[ind].mean(axis=1)


def module_scores(zreadout: np.ndarray, gene_names: list[str]) -> dict[str, np.ndarray]:
    gene_to_col = {g: i for i, g in enumerate(gene_names)}
    scores = {}
    for module, meta in MODULES.items():
        up_cols = [gene_to_col[g] for g in meta["up"] if g in gene_to_col]
        down_cols = [gene_to_col[g] for g in meta["down"] if g in gene_to_col]
        vals = np.zeros(zreadout.shape[0], dtype=np.float32)
        pieces = 0
        if up_cols:
            vals += zreadout[:, up_cols].mean(axis=1)
            pieces += 1
        if down_cols:
            vals -= zreadout[:, down_cols].mean(axis=1)
            pieces += 1
        if pieces > 1:
            vals /= float(pieces)
        scores[module] = vals
    return scores


def compute() -> dict[str, pd.DataFrame]:
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    adata = ad.read_h5ad(ADATA_PATH)
    latent = np.asarray(adata.obsm["X_latent"], dtype=np.float32)
    times = adata.obs["time_point_processed"].astype(float).to_numpy()
    axis = build_response_axis(latent, times)
    time_points, traj = load_all_trajectory()
    t0_idx = int(np.argmin(np.abs(time_points - 0.0)))
    t1_idx = int(np.argmin(np.abs(time_points - 1.0)))
    t2_idx = int(np.argmin(np.abs(time_points - 2.0)))
    t3_idx = int(np.argmin(np.abs(time_points - 3.0)))

    states0 = assign_states(traj[t0_idx], axis)
    states3 = assign_states(traj[t3_idx], axis)
    state_names = [f"state_{i+1}" for i in range(6)]
    transition_rows = []
    for i, origin in enumerate(state_names):
        mask = states0 == i
        denom = int(mask.sum())
        for j, final in enumerate(state_names):
            count = int(((states3 == j) & mask).sum())
            transition_rows.append(
                {
                    "origin_state": origin,
                    "final_state": final,
                    "count": count,
                    "fraction_of_origin": float(count / max(denom, 1)),
                }
            )
    transition = pd.DataFrame(transition_rows)
    transition.to_csv(TAB / "pulsebridge_perturbation_state_transition.csv", index=False)

    gene_names, expr, zexpr = selected_expression(adata)
    nn = NearestNeighbors(n_neighbors=25).fit(latent)
    expr_by_time: dict[float, np.ndarray] = {}
    z_by_time: dict[float, np.ndarray] = {}
    module_by_time: dict[float, dict[str, np.ndarray]] = {}
    for idx in [t0_idx, t1_idx, t2_idx, t3_idx]:
        t = float(time_points[idx])
        expr_read, z_read = neighbor_readout(nn, traj[idx], expr, zexpr)
        expr_by_time[t] = expr_read
        z_by_time[t] = z_read
        module_by_time[t] = module_scores(z_read, gene_names)

    module_rows = []
    for module, meta in MODULES.items():
        start = module_by_time[0.0][module]
        for t in [0.0, 1.0, 2.0, 3.0]:
            vals = module_by_time[t][module]
            module_rows.append(
                {
                    "module": module,
                    "label": meta["label"].replace("\n", " "),
                    "time": t,
                    "mean_score": float(vals.mean()),
                    "delta_from_day0": float(vals.mean() - start.mean()),
                    "sem": float(vals.std(ddof=1) / np.sqrt(len(vals))),
                }
            )
    module_time = pd.DataFrame(module_rows)
    module_time.to_csv(TAB / "pulsebridge_perturbation_module_timecourse.csv", index=False)

    origin_rows = []
    for i, origin in enumerate(state_names):
        mask = states0 == i
        if not mask.any():
            continue
        row = {
            "origin_state": origin,
            "particle_count": int(mask.sum()),
            "final_late_state_fraction": float(np.isin(states3[mask], [4, 5]).mean()),
            "final_state_mode": state_names[int(np.bincount(states3[mask], minlength=6).argmax())],
        }
        for module in MODULES:
            start = module_by_time[0.0][module][mask].mean()
            end = module_by_time[3.0][module][mask].mean()
            row[f"{module}_delta"] = float(end - start)
            row[f"{module}_day3"] = float(end)
        row["therapeutic_minus_adaptive_delta"] = float(
            row["source_aligned_antitumor_delta"] - row["adaptive_metabolic_delta"]
        )
        origin_rows.append(row)
    origin_summary = pd.DataFrame(origin_rows)
    origin_summary.to_csv(TAB / "pulsebridge_perturbation_origin_response_summary.csv", index=False)

    gene_to_col = {g: i for i, g in enumerate(gene_names)}
    gene_rows = []
    for gene in GENE_PLOT_ORDER:
        if gene not in gene_to_col:
            continue
        col = gene_to_col[gene]
        gene_rows.append(
            {
                "gene": gene,
                "pulsebridge_day0_neighbor_mean": float(expr_by_time[0.0][:, col].mean()),
                "pulsebridge_day3_neighbor_mean": float(expr_by_time[3.0][:, col].mean()),
                "pulsebridge_day3_minus_day0": float(expr_by_time[3.0][:, col].mean() - expr_by_time[0.0][:, col].mean()),
                "pulsebridge_z_day3_minus_day0": float(z_by_time[3.0][:, col].mean() - z_by_time[0.0][:, col].mean()),
            }
        )
    gene_delta = pd.DataFrame(gene_rows)
    gene_delta.to_csv(TAB / "pulsebridge_perturbation_gene_delta.csv", index=False)
    return {
        "transition": transition,
        "origin": origin_summary,
        "module_time": module_time,
        "gene_delta": gene_delta,
    }


def draw_alluvial(ax, transition: pd.DataFrame) -> None:
    state_names = [f"state_{i+1}" for i in range(6)]
    ypos = {s: 5 - i for i, s in enumerate(state_names)}
    ax.scatter(np.zeros(6), [ypos[s] for s in state_names], s=90, color=STATE_COLORS, edgecolor="white", zorder=3)
    ax.scatter(np.ones(6), [ypos[s] for s in state_names], s=90, color=STATE_COLORS, edgecolor="white", zorder=3)
    for s in state_names:
        ax.text(-0.04, ypos[s], s, ha="right", va="center", fontsize=8)
        ax.text(1.04, ypos[s], s, ha="left", va="center", fontsize=8)
    for row in transition.itertuples():
        frac = float(row.fraction_of_origin)
        if frac < 0.04:
            continue
        y0 = ypos[row.origin_state]
        y1 = ypos[row.final_state]
        final_idx = int(row.final_state.split("_")[1]) - 1
        path = MplPath(
            [(0.0, y0), (0.32, y0), (0.68, y1), (1.0, y1)],
            [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4],
        )
        ax.add_patch(
            PathPatch(
                path,
                facecolor="none",
                edgecolor=STATE_COLORS[final_idx],
                lw=0.5 + 16 * frac,
                alpha=0.30 + 0.35 * min(frac, 0.8),
                capstyle="round",
            )
        )
    ax.text(0, 5.62, "day0 origin", ha="center", fontsize=9, fontweight="bold")
    ax.text(1, 5.62, "PulseBridge day3 fate", ha="center", fontsize=9, fontweight="bold")
    ax.set_xlim(-0.14, 1.14)
    ax.set_ylim(-0.55, 5.8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("PulseBridge 5-AZA-CdR rollout: state fates", fontsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_origin_scatter(ax, origin: pd.DataFrame) -> None:
    origin = origin.copy()
    origin["state_index"] = origin["origin_state"].str.split("_").str[1].astype(int)
    origin = origin.sort_values("state_index")
    x = np.arange(len(origin))
    ax.plot(
        x,
        origin["source_aligned_antitumor_delta"],
        color=BLUE,
        marker="o",
        lw=1.9,
        label="anti-tumor-like",
    )
    ax.plot(
        x,
        origin["adaptive_metabolic_delta"],
        color="#A17C00",
        marker="o",
        lw=1.9,
        label="adaptive/metabolic",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(origin["origin_state"], rotation=25, ha="right")
    ax.set_ylabel("program induction\n(PulseBridge day3 - day0)")
    ax.set_title("State-specific rollout biology")
    ax.set_ylim(0.18, 0.50)
    ax.grid(axis="y", color="#ECECEC", lw=0.6)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")

    ax2 = ax.twinx()
    ax2.plot(
        x,
        origin["final_late_state_fraction"],
        color="#444444",
        marker="s",
        lw=1.2,
        ls="--",
        label="late fate",
    )
    ax2.set_ylim(0, 1.08)
    ax2.set_ylabel("final late-state fraction", color="#444444")
    ax2.tick_params(axis="y", colors="#444444")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(True)


def draw_gene_delta(ax, gene_delta: pd.DataFrame) -> None:
    df = gene_delta.set_index("gene").reindex(GENE_PLOT_ORDER).dropna().reset_index()
    y = np.arange(len(df))
    vals = df["pulsebridge_day3_minus_day0"].to_numpy()
    colors = [GENE_COLORS.get(g, "#777777") for g in df["gene"]]
    ax.barh(y, vals, color=colors, height=0.62)
    ax.axvline(0, color="#222222", lw=0.85)
    for i, row in df.iterrows():
        val = float(row["pulsebridge_day3_minus_day0"])
        if val >= 0:
            ax.text(val + 0.04, i, f"{val:+.2f}", va="center", ha="left", fontsize=7.5, color="#222222")
        else:
            ax.text(val / 2, i, f"{val:+.2f}", va="center", ha="center", fontsize=7.5, color="white")
    ax.set_yticks(y)
    ax.set_yticklabels(df["gene"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("PulseBridge rollout day3 - day0")
    ax.set_title("Generated rollout gene readout")
    ax.set_xlim(-0.86, 1.05)
    ax.grid(axis="x", color="#ECECEC", lw=0.6)


def plot(results: dict[str, pd.DataFrame]) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(13.0, 4.8), dpi=200)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.05, 1.05], wspace=0.46)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    draw_alluvial(ax_a, results["transition"])
    draw_origin_scatter(ax_b, results["origin"])
    draw_gene_delta(ax_c, results["gene_delta"])
    for ax, letter in [(ax_a, "A"), (ax_b, "B"), (ax_c, "C")]:
        ax.text(-0.12, 1.08, letter, transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")
    fig.savefig(FIG / "story_pulsebridge_rollout_biology.png", bbox_inches="tight")
    fig.savefig(FIG / "story_pulsebridge_rollout_biology.pdf", bbox_inches="tight")
    fig.savefig(FIG / "story_pulsebridge_perturbation_biology.png", bbox_inches="tight")
    fig.savefig(FIG / "story_pulsebridge_perturbation_biology.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    results = compute()
    plot(results)
    print(FIG / "story_pulsebridge_rollout_biology.png")
    print(FIG / "story_pulsebridge_rollout_biology.pdf")
    print(TAB / "pulsebridge_perturbation_origin_response_summary.csv")
    print(TAB / "pulsebridge_perturbation_gene_delta.csv")


if __name__ == "__main__":
    main()
