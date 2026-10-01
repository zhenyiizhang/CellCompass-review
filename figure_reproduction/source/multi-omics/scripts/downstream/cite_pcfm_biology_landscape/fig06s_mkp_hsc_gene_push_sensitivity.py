#!/usr/bin/env python3
"""Fig06S: MkP target gene sensitivity from HSC-start CITE-PCFM rollouts.

This panel asks a target-guidance question rather than a generic benchmark
question: among megakaryocytic RNA genes/program components, which +5z RNA
directions make an HSC-start CITE-PCFM rollout terminate with a stronger MkP
protein phenotype?

Contract:
    HSC t0 RNA seed r0
    -> r0 + 5 * single-gene latent direction
    -> RNA ODE rollout
    -> terminal protein q=h_eta(r_T,T)
    -> audited protein-phenotype MLP and decoded Mk ADT module

The result is an in-silico model query, not a causal perturbation experiment.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/cytobridge_mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/cytobridge_xdg_cache")
os.environ.setdefault("NUMBA_CACHE_DIR", "/private/tmp/numba_cache")
os.environ.setdefault("NUMBA_DISABLE_CACHE", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")
for key in ["MPLCONFIGDIR", "XDG_CACHE_HOME", "NUMBA_CACHE_DIR"]:
    Path(os.environ[key]).mkdir(parents=True, exist_ok=True)

import anndata as ad
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch, PathPatch, Rectangle
from matplotlib.path import Path as MplPath
from scipy import sparse

from fig06_rna_protein_regulatory_map import (
    DATA_PATH,
    FIG_DIR,
    OUT_DIR,
    PHENO_ORDER,
    PROTEIN_DIM,
    RANDOM_STATE,
    RNA_DIM,
    TRAINED_MODEL_H5AD,
    clean_marker,
    ensure_import_paths,
    feature_masks,
    projection_audit_and_scales,
    load_readout_input,
)
from fig06j_program_push_terminal_composition import (
    BATCH_SIZE,
    PUSH_EPS,
    TRAJECTORY_PATH,
    condition_rna,
    load_trajectory_seed,
    rollout_terminal_q,
)
from fig06n_rna_protein_relay_heatmap import fit_feature_decoder
from multistageot_style import PHENO_COLORS, clean_metric_axis, setup
from protein_phenotype_readout import train_mlp_protein_phenotype_readout


ROOT = Path("/Users/zhenyizhang/Documents/GitHub/CytoBridge-agent-runtime-v2-20260309/multi-omics")
CYTOBRIDGE_SRC = ROOT.parent / "CytoBridge-main"

TARGET_PHENOTYPE = "MkP"
MK_ADT_MARKERS = ["CD41", "CD42b", "CD62P", "CD49b"]

GENE_SPECS = [
    {"gene": "NFE2", "class": "Mk regulator", "role": "megakaryocyte maturation TF"},
    {"gene": "FLI1", "class": "Mk regulator", "role": "ETS megakaryocyte regulator"},
    {"gene": "GFI1B", "class": "Mk regulator", "role": "erythro-megakaryocyte regulator"},
    {"gene": "RUNX1", "class": "Mk regulator", "role": "hematopoietic/Mk regulator"},
    {"gene": "TAL1", "class": "Ery-Mk regulator", "role": "erythroid-megakaryocyte TF"},
    {"gene": "GATA1", "class": "Ery-Mk regulator", "role": "erythroid-megakaryocyte TF"},
    {"gene": "KLF1", "class": "Ery-Mk regulator", "role": "erythroid-biased TF"},
    {"gene": "ITGA2B", "class": "Mk marker", "role": "CD41 alpha-IIb transcript"},
    {"gene": "GP9", "class": "Mk marker", "role": "GPIb complex transcript"},
    {"gene": "GP1BA", "class": "Mk marker", "role": "GPIb alpha transcript"},
    {"gene": "GP6", "class": "Mk marker", "role": "platelet collagen receptor"},
    {"gene": "VWF", "class": "Mk effector", "role": "platelet/endothelial effector"},
    {"gene": "TUBB1", "class": "Mk effector", "role": "platelet beta-tubulin"},
    {"gene": "THBS1", "class": "Mk effector", "role": "platelet alpha-granule matrix"},
    {"gene": "FERMT3", "class": "Mk effector", "role": "integrin activation effector"},
    {"gene": "MYL9", "class": "Mk effector", "role": "contractile platelet program"},
    {"gene": "PF4", "class": "Mk effector", "role": "platelet factor 4, absent if not in panel"},
    {"gene": "PPBP", "class": "Mk effector", "role": "platelet basic protein, absent if not in panel"},
]

GENE_CLASS_COLORS = {
    "Mk regulator": "#7C3AED",
    "Ery-Mk regulator": "#EA580C",
    "Mk marker": "#DB2777",
    "Mk effector": "#0F766E",
}


def dense(x) -> np.ndarray:
    return x.toarray() if sparse.issparse(x) else np.asarray(x)


def hsc_t0_particle_ids(adata: ad.AnnData, r0: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    times = np.asarray(adata.obs["time_point_processed"], dtype=float)
    labels = adata.obs["cell_type"].astype(str).to_numpy()
    t0_idx = np.where(np.isclose(times, 0.0))[0]
    rna = np.asarray(adata.obsm["X_rna_svd"], dtype=np.float32)[:, :RNA_DIM]
    if t0_idx.shape[0] != r0.shape[0]:
        raise ValueError(f"t0 observed count {t0_idx.shape[0]} != trajectory particles {r0.shape[0]}")
    hsc_ids = np.where(labels[t0_idx] == "HSC")[0].astype(int)
    audit = {
        "t0_observed_count": int(t0_idx.shape[0]),
        "trajectory_particle_count": int(r0.shape[0]),
        "hsc_t0_particle_count": int(hsc_ids.shape[0]),
        "hsc_t0_fraction": float(hsc_ids.shape[0] / max(1, t0_idx.shape[0])),
        "rna_mean_abs_diff_points0_vs_observed_t0": float(np.mean(np.abs(r0 - rna[t0_idx]))),
        "rna_max_abs_diff_points0_vs_observed_t0": float(np.max(np.abs(r0 - rna[t0_idx]))),
    }
    return hsc_ids, audit


def build_single_gene_directions(
    adata: ad.AnnData,
    rna_component_std: np.ndarray,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    var_names, rna_mask, _ = feature_masks(adata)
    rna_var_names = var_names[rna_mask]
    rna_lookup = {clean_marker(name): i for i, name in enumerate(rna_var_names)}
    pcs_rna = np.asarray(adata.varm["PCs"], dtype=np.float64)[rna_mask, :RNA_DIM]

    requested = [spec["gene"] for spec in GENE_SPECS]
    available = [gene for gene in requested if gene in rna_lookup]
    selected_cols = np.asarray([rna_lookup[gene] for gene in available], dtype=int)
    x_sel = dense(adata.X[:, rna_mask][:, selected_cols]).astype(np.float64)
    gene_std = x_sel.std(axis=0)
    gene_std = np.where(gene_std > 1e-8, gene_std, 1.0)
    std_lookup = {gene: float(gene_std[i]) for i, gene in enumerate(available)}

    directions: dict[str, np.ndarray] = {}
    rows = []
    spec_lookup = {spec["gene"]: spec for spec in GENE_SPECS}
    for gene in requested:
        spec = spec_lookup[gene]
        is_available = gene in rna_lookup
        direction = None
        direction_norm = np.nan
        gene_feature_std = np.nan
        if is_available:
            rna_col = rna_lookup[gene]
            gene_feature_std = std_lookup[gene]
            # +1 gene z-score in original RNA feature space changes the
            # standardized RNA SVD coordinate by std_gene * PC / std_component.
            direction = (gene_feature_std * pcs_rna[rna_col, :]) / rna_component_std
            directions[gene] = direction.astype(np.float32)
            direction_norm = float(np.linalg.norm(direction))
        rows.append(
            {
                "gene": gene,
                "gene_class": spec["class"],
                "role": spec["role"],
                "available": bool(is_available),
                "gene_feature_std": float(gene_feature_std) if np.isfinite(gene_feature_std) else np.nan,
                "latent_direction_norm": direction_norm,
            }
        )
    return directions, pd.DataFrame(rows)


def fit_mk_adt_decoder(adata: ad.AnnData) -> tuple[object, list[str], pd.DataFrame]:
    var_names, _, protein_mask = feature_masks(adata)
    adt_names = [clean_marker(v) for v in var_names[protein_mask]]
    adt_lookup = {name: i for i, name in enumerate(adt_names)}
    markers = [marker for marker in MK_ADT_MARKERS if marker in adt_lookup]
    if not markers:
        raise ValueError("No Mk ADT markers available for decoder.")
    idx = np.asarray([adt_lookup[marker] for marker in markers], dtype=int)
    x_protein = dense(adata.X[:, protein_mask][:, idx]).astype(np.float64)
    q = np.asarray(adata.obsm["X_protein_pca"], dtype=np.float32)[:, :PROTEIN_DIM]
    decoder, audit_df = fit_feature_decoder(q, x_protein, markers, feature_type="ADT")
    return decoder, markers, audit_df


def summarize_q(
    readout,
    adt_decoder,
    adt_markers: list[str],
    q: np.ndarray,
) -> tuple[dict[str, float], pd.DataFrame]:
    proba = readout.predict_proba_ordered(q)
    pred = readout.predict(q).astype(str)
    decoded_adt = np.asarray(adt_decoder.predict(q), dtype=np.float64)
    module = decoded_adt.mean(axis=1)
    summary: dict[str, float] = {
        "n_particles": int(q.shape[0]),
        "hard_fraction_MkP": float(np.mean(pred == TARGET_PHENOTYPE)),
        "mk_adt_module_z": float(np.mean(module)),
    }
    for j, phenotype in enumerate(PHENO_ORDER):
        summary[f"p_{phenotype}"] = float(np.mean(proba[:, j]))
        summary[f"hard_fraction_{phenotype}"] = float(np.mean(pred == phenotype))
    for j, marker in enumerate(adt_markers):
        summary[f"adt_{marker}_z"] = float(np.mean(decoded_adt[:, j]))
    particle_df = pd.DataFrame(
        {
            "particle_id": np.arange(q.shape[0], dtype=int),
            "predicted_phenotype": pred,
            "p_MkP": proba[:, PHENO_ORDER.index(TARGET_PHENOTYPE)],
            "mk_adt_module_z": module,
        }
    )
    for j, marker in enumerate(adt_markers):
        particle_df[f"adt_{marker}_z"] = decoded_adt[:, j]
    return summary, particle_df


def run_gene_pushes(
    model,
    readout_input,
    phenotype_readout,
    adt_decoder,
    adt_markers: list[str],
    r0_hsc: np.ndarray,
    time_points: np.ndarray,
    directions: dict[str, np.ndarray],
    direction_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    baseline_q = rollout_terminal_q(model, readout_input, r0_hsc, time_points, batch_size=BATCH_SIZE)
    baseline_summary, baseline_particles = summarize_q(phenotype_readout, adt_decoder, adt_markers, baseline_q)

    summary_rows = []
    particle_frames = []
    available_genes = direction_df[direction_df["available"]]["gene"].tolist()
    for gene_i, gene in enumerate(available_genes, start=1):
        print(f"[fig06s] rolling {gene_i}/{len(available_genes)}: {gene}", flush=True)
        for condition, direction, scale in [
            ("-5z", directions[gene], -PUSH_EPS),
            ("0", None, 0.0),
            ("+5z", directions[gene], PUSH_EPS),
        ]:
            if condition == "0":
                summary = baseline_summary.copy()
                particles = baseline_particles.copy()
            else:
                q = rollout_terminal_q(
                    model,
                    readout_input,
                    condition_rna(r0_hsc, direction, scale),
                    time_points,
                    batch_size=BATCH_SIZE,
                )
                summary, particles = summarize_q(phenotype_readout, adt_decoder, adt_markers, q)
            spec_row = direction_df[direction_df["gene"] == gene].iloc[0].to_dict()
            summary_rows.append({"gene": gene, "condition": condition, **spec_row, **summary})
            particles.insert(0, "condition", condition)
            particles.insert(0, "gene", gene)
            particle_frames.append(particles)

    summary_df = pd.DataFrame(summary_rows)
    particle_df = pd.concat(particle_frames, ignore_index=True)

    delta_rows = []
    for gene, sub in summary_df.groupby("gene", sort=False):
        base = sub[sub["condition"] == "0"].iloc[0]
        plus = sub[sub["condition"] == "+5z"].iloc[0]
        minus = sub[sub["condition"] == "-5z"].iloc[0]
        row = {
            "gene": gene,
            "gene_class": str(base["gene_class"]),
            "role": str(base["role"]),
            "latent_direction_norm": float(base["latent_direction_norm"]),
            "n_particles": int(base["n_particles"]),
            "baseline_p_MkP": float(base["p_MkP"]),
            "plus_p_MkP": float(plus["p_MkP"]),
            "minus_p_MkP": float(minus["p_MkP"]),
            "plus_minus_baseline_p_MkP_pp": 100.0 * (float(plus["p_MkP"]) - float(base["p_MkP"])),
            "plus_minus_minus_p_MkP_pp": 100.0 * (float(plus["p_MkP"]) - float(minus["p_MkP"])),
            "baseline_hard_fraction_MkP": float(base["hard_fraction_MkP"]),
            "plus_hard_fraction_MkP": float(plus["hard_fraction_MkP"]),
            "minus_hard_fraction_MkP": float(minus["hard_fraction_MkP"]),
            "plus_minus_baseline_hard_MkP_pp": 100.0
            * (float(plus["hard_fraction_MkP"]) - float(base["hard_fraction_MkP"])),
            "baseline_mk_adt_module_z": float(base["mk_adt_module_z"]),
            "plus_mk_adt_module_z": float(plus["mk_adt_module_z"]),
            "minus_mk_adt_module_z": float(minus["mk_adt_module_z"]),
            "plus_minus_baseline_mk_adt_module_z": float(plus["mk_adt_module_z"]) - float(base["mk_adt_module_z"]),
            "plus_minus_minus_mk_adt_module_z": float(plus["mk_adt_module_z"]) - float(minus["mk_adt_module_z"]),
        }
        for phenotype in PHENO_ORDER:
            row[f"delta_p_{phenotype}_pp"] = 100.0 * (float(plus[f"p_{phenotype}"]) - float(base[f"p_{phenotype}"]))
        for marker in adt_markers:
            row[f"delta_adt_{marker}_z"] = float(plus[f"adt_{marker}_z"]) - float(base[f"adt_{marker}_z"])
        delta_rows.append(row)
    delta_df = pd.DataFrame(delta_rows).sort_values("plus_minus_baseline_p_MkP_pp", ascending=False)
    return summary_df, delta_df, particle_df


def plot_mkp_gene_sensitivity(delta_df: pd.DataFrame, summary_df: pd.DataFrame, out_png: Path) -> None:
    setup()
    plot_df = delta_df.copy().sort_values("plus_minus_baseline_p_MkP_pp", ascending=True)
    heat_df = delta_df.sort_values("plus_minus_baseline_p_MkP_pp", ascending=False).head(12).copy()
    stack_genes = heat_df["gene"].head(4).tolist()

    fig = plt.figure(figsize=(15.5, 8.8))
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.15, 1.22, 0.95],
        height_ratios=[1.05, 1.0],
        left=0.065,
        right=0.985,
        top=0.835,
        bottom=0.185,
        wspace=0.36,
        hspace=0.42,
    )
    ax_rank = fig.add_subplot(gs[:, 0])
    ax_heat = fig.add_subplot(gs[0, 1:])
    ax_scatter = fig.add_subplot(gs[1, 1])
    ax_stack = fig.add_subplot(gs[1, 2])

    # A. Gene ranking.
    y = np.arange(plot_df.shape[0])
    vals = plot_df["plus_minus_baseline_p_MkP_pp"].to_numpy(dtype=float)
    ax_rank.axvline(0, color="#94A3B8", lw=1.0)
    for yi, (_, row) in zip(y, plot_df.iterrows()):
        color = GENE_CLASS_COLORS.get(row["gene_class"], "#64748B")
        x = float(row["plus_minus_baseline_p_MkP_pp"])
        ax_rank.hlines(yi, 0, x, color=color, lw=3.0, alpha=0.35)
        ax_rank.scatter(x, yi, s=58, color=color, edgecolor="white", linewidth=0.8, zorder=3)
        ax_rank.text(
            x + (0.09 if x >= 0 else -0.09),
            yi,
            f"{x:+.1f}",
            ha="left" if x >= 0 else "right",
            va="center",
            fontsize=7.2,
            color="#334155",
        )
    ax_rank.set_yticks(y)
    ax_rank.set_yticklabels(plot_df["gene"], fontsize=8.6)
    ax_rank.set_xlabel("Delta terminal P(MkP), +5z vs 0\npercentage points", fontsize=8.8)
    ax_rank.set_title("A  MkP target-gene sensitivity", loc="left", fontsize=12.2, weight="bold")
    clean_metric_axis(ax_rank, grid_axis="x")
    lim = max(1.0, float(np.nanmax(np.abs(vals))) * 1.22)
    ax_rank.set_xlim(-lim, lim)

    # B. Phenotype redistribution for the strongest genes.
    matrix = heat_df[[f"delta_p_{p}_pp" for p in PHENO_ORDER]].to_numpy(dtype=float)
    vmax = max(0.6, float(np.nanmax(np.abs(matrix))) * 1.05)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax_heat.imshow(matrix, aspect="auto", cmap="RdBu_r", norm=norm)
    ax_heat.set_xticks(np.arange(len(PHENO_ORDER)))
    ax_heat.set_xticklabels(PHENO_ORDER, fontsize=8.2)
    ax_heat.set_yticks(np.arange(heat_df.shape[0]))
    ax_heat.set_yticklabels(heat_df["gene"], fontsize=8.0)
    ax_heat.set_title("B  Where does the terminal protein distribution move?", loc="left", fontsize=12.2, weight="bold")
    ax_heat.set_xlabel("terminal protein phenotype basin", fontsize=8.6)
    ax_heat.tick_params(length=0)
    for spine in ax_heat.spines.values():
        spine.set_visible(False)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if abs(v) >= max(0.55, vmax * 0.28):
                ax_heat.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=6.8, color="#111827")
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.024, pad=0.012)
    cbar.set_label("Delta probability, pp", fontsize=8.0)
    cbar.ax.tick_params(labelsize=7.0)

    # C. Concordance between MkP probability and decoded Mk ADT markers.
    for gene_class, class_df in delta_df.groupby("gene_class", sort=False):
        ax_scatter.scatter(
            class_df["plus_minus_baseline_p_MkP_pp"],
            class_df["plus_minus_baseline_mk_adt_module_z"],
            s=72,
            color=GENE_CLASS_COLORS.get(gene_class, "#64748B"),
            edgecolor="white",
            linewidth=0.8,
            alpha=0.92,
            label=gene_class,
        )
    top_labels = set(heat_df["gene"].head(6).tolist()) | set(delta_df.sort_values("plus_minus_baseline_p_MkP_pp").head(2)["gene"].tolist())
    for _, row in delta_df[delta_df["gene"].isin(top_labels)].iterrows():
        ax_scatter.text(
            float(row["plus_minus_baseline_p_MkP_pp"]) + 0.06,
            float(row["plus_minus_baseline_mk_adt_module_z"]) + 0.003,
            str(row["gene"]),
            fontsize=7.2,
            color="#334155",
        )
    ax_scatter.axhline(0, color="#CBD5E1", lw=0.9)
    ax_scatter.axvline(0, color="#CBD5E1", lw=0.9)
    ax_scatter.set_xlabel("Delta terminal P(MkP), pp", fontsize=8.6)
    ax_scatter.set_ylabel("Delta decoded Mk ADT module z", fontsize=8.6)
    ax_scatter.set_title("C  Phenotype shift matches Mk surface markers", loc="left", fontsize=12.2, weight="bold")
    clean_metric_axis(ax_scatter, grid_axis="both")

    # D. Stacked phenotype probabilities for baseline and top +5z genes.
    stack_order = ["0"] + [f"{gene}+5z" for gene in stack_genes]
    stack_values = []
    baseline = summary_df[(summary_df["gene"] == stack_genes[0]) & (summary_df["condition"] == "0")].iloc[0]
    stack_values.append([float(baseline[f"p_{p}"]) for p in PHENO_ORDER])
    for gene in stack_genes:
        row = summary_df[(summary_df["gene"] == gene) & (summary_df["condition"] == "+5z")].iloc[0]
        stack_values.append([float(row[f"p_{p}"]) for p in PHENO_ORDER])
    stack_arr = np.asarray(stack_values, dtype=float)
    x = np.arange(stack_arr.shape[0])
    bottom = np.zeros(stack_arr.shape[0], dtype=float)
    for j, phenotype in enumerate(PHENO_ORDER):
        ax_stack.bar(
            x,
            stack_arr[:, j],
            bottom=bottom,
            width=0.72,
            color=PHENO_COLORS.get(phenotype, "#94A3B8"),
            edgecolor="white",
            linewidth=0.55,
            label=phenotype,
        )
        bottom += stack_arr[:, j]
    ax_stack.set_xticks(x)
    ax_stack.set_xticklabels(stack_order, rotation=38, ha="right", fontsize=7.5)
    ax_stack.set_ylim(0, 1)
    ax_stack.set_ylabel("terminal phenotype probability", fontsize=8.3)
    ax_stack.set_title("D  Top-gene terminal composition", loc="left", fontsize=12.2, weight="bold")
    clean_metric_axis(ax_stack, grid_axis="y")

    handles = [Patch(facecolor=color, edgecolor="white", label=label) for label, color in GENE_CLASS_COLORS.items()]
    fig.legend(handles=handles, frameon=False, ncol=4, loc="lower left", bbox_to_anchor=(0.06, 0.065), fontsize=8.2)
    pheno_handles = [Patch(facecolor=PHENO_COLORS[p], edgecolor="white", label=p) for p in PHENO_ORDER]
    ax_stack.legend(handles=pheno_handles, frameon=False, fontsize=6.8, ncol=2, loc="upper left", bbox_to_anchor=(1.02, 1.03))

    fig.suptitle(
        "Fig06S. HSC-start in-silico MkP target guidance by single-gene RNA directions",
        x=0.01,
        y=0.965,
        ha="left",
        fontsize=14.0,
        weight="bold",
    )
    fig.text(
        0.01,
        0.914,
        "Each candidate gene is increased by +5 z-score in RNA feature space, mapped through RNA SVD, rolled by CITE-PCFM, and read as terminal protein phenotype plus decoded Mk ADT markers.",
        ha="left",
        fontsize=9.0,
        color="#334155",
    )
    fig.savefig(out_png, dpi=360, bbox_inches="tight", facecolor="white")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _label_with_outline(ax: plt.Axes, x: float, y: float, label: str, *, dx: float = 0.04, dy: float = 0.0) -> None:
    text = ax.text(
        x + dx,
        y + dy,
        label,
        fontsize=7.7,
        color="#1E293B",
        ha="left" if dx >= 0 else "right",
        va="center",
        weight="bold" if label in {"THBS1", "TUBB1", "GP9", "NFE2", "KLF1"} else "normal",
        zorder=20,
    )
    text.set_path_effects([pe.withStroke(linewidth=2.8, foreground="white", alpha=0.95)])


def _draw_surface_marker_glyph(
    ax: plt.Axes,
    cx: float,
    cy: float,
    values: np.ndarray,
    marker_labels: list[str],
    *,
    vmax: float,
) -> None:
    angles = np.deg2rad([96, 24, -48, -120])
    marker_colors = {
        "CD41": "#E11D48",
        "CD42b": "#F97316",
        "CD62P": "#7C3AED",
        "CD49b": "#0F766E",
    }
    ax.scatter([cx], [cy], s=18, color="#CBD5E1", edgecolor="white", linewidth=0.55, zorder=4)
    for val, marker, theta in zip(values, marker_labels, angles):
        signed = float(val)
        mag = min(1.0, abs(signed) / max(vmax, 1e-8))
        length = 0.13 + 0.34 * mag
        direction = 1.0 if signed >= 0 else -1.0
        x2 = cx + direction * length * np.cos(theta)
        y2 = cy + direction * length * np.sin(theta)
        color = marker_colors.get(marker, "#64748B") if signed >= 0 else "#2563EB"
        ax.plot([cx, x2], [cy, y2], color=color, lw=1.8 + 2.0 * mag, alpha=0.82, solid_capstyle="round", zorder=3)
        ax.scatter([x2], [y2], s=22 + 70 * mag, color=color, edgecolor="white", linewidth=0.55, alpha=0.95, zorder=5)


def plot_mkp_signature_map(delta_df: pd.DataFrame, summary_df: pd.DataFrame, out_png: Path) -> None:
    """Cleaner main-text style MkP target-guidance visualization."""

    setup()
    df = delta_df.copy()
    df["mk_selectivity_pp"] = df["delta_p_MkP_pp"] - df["delta_p_EryP_pp"]
    df["abs_mk_selectivity_pp"] = df["mk_selectivity_pp"].abs()
    top_gene_order = df.sort_values("plus_minus_baseline_p_MkP_pp", ascending=False)["gene"].head(7).tolist()
    marker_rows = top_gene_order + ["KLF1"]

    fig = plt.figure(figsize=(15.8, 8.4))
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.55, 1.08, 0.92],
        height_ratios=[1.02, 0.82],
        left=0.06,
        right=0.985,
        top=0.825,
        bottom=0.135,
        wspace=0.36,
        hspace=0.42,
    )
    ax_map = fig.add_subplot(gs[:, 0])
    ax_bias = fig.add_subplot(gs[0, 1])
    ax_marker = fig.add_subplot(gs[1, 1])
    ax_prob = fig.add_subplot(gs[:, 2])

    # A. Main signature-response map.
    x = df["plus_minus_baseline_p_MkP_pp"].to_numpy(dtype=float)
    y = df["plus_minus_baseline_mk_adt_module_z"].to_numpy(dtype=float)
    x_min, x_max = min(-2.2, float(x.min()) - 0.7), max(6.8, float(x.max()) + 0.8)
    y_min, y_max = min(-0.12, float(y.min()) - 0.035), max(0.23, float(y.max()) + 0.035)
    ax_map.add_patch(
        Rectangle((0, 0), x_max, y_max, facecolor="#F3E8FF", edgecolor="none", alpha=0.43, zorder=0)
    )
    ax_map.axvline(0, color="#CBD5E1", lw=1.0, zorder=1)
    ax_map.axhline(0, color="#CBD5E1", lw=1.0, zorder=1)
    for gene_class, sub in df.groupby("gene_class", sort=False):
        ax_map.scatter(
            sub["plus_minus_baseline_p_MkP_pp"],
            sub["plus_minus_baseline_mk_adt_module_z"],
            s=92,
            color=GENE_CLASS_COLORS.get(gene_class, "#64748B"),
            edgecolor="white",
            linewidth=1.1,
            alpha=0.96,
            label=gene_class,
            zorder=5,
        )
    label_offsets = {
        "THBS1": (0.08, 0.010),
        "TUBB1": (0.10, -0.010),
        "GP9": (0.08, 0.004),
        "GP1BA": (0.08, 0.008),
        "VWF": (0.08, -0.006),
        "ITGA2B": (0.08, -0.010),
        "NFE2": (0.08, 0.006),
        "FLI1": (0.08, 0.008),
        "KLF1": (-0.10, -0.002),
        "GATA1": (0.08, 0.006),
    }
    for _, row in df.iterrows():
        gene = str(row["gene"])
        if gene in label_offsets or float(row["plus_minus_baseline_p_MkP_pp"]) >= 2.5:
            dx, dy = label_offsets.get(gene, (0.08, 0.0))
            _label_with_outline(
                ax_map,
                float(row["plus_minus_baseline_p_MkP_pp"]),
                float(row["plus_minus_baseline_mk_adt_module_z"]),
                gene,
                dx=dx,
                dy=dy,
            )
    ax_map.text(
        0.035,
        0.965,
        "Mk protein-acquisition quadrant\nhigher P(MkP) and higher Mk ADT module",
        transform=ax_map.transAxes,
        va="top",
        fontsize=8.3,
        color="#5B21B6",
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "#E9D5FF", "alpha": 0.92},
    )
    ax_map.set_xlim(x_min, x_max)
    ax_map.set_ylim(y_min, y_max)
    ax_map.set_xlabel("Delta terminal P(MkP), +5z RNA direction vs baseline (percentage points)", fontsize=9.1)
    ax_map.set_ylabel("Delta decoded Mk ADT module z\nCD41/CD42b/CD62P/CD49b", fontsize=9.1)
    ax_map.set_title("A  RNA directions that acquire Mk protein phenotype", loc="left", fontsize=12.4, weight="bold")
    ax_map.legend(frameon=False, fontsize=8.1, loc="lower right", handletextpad=0.4)
    clean_metric_axis(ax_map, grid_axis="both")

    # B. Mk versus Ery bifurcation fork.
    ax_bias.set_title("B  Mk-vs-Ery bifurcation fork", loc="left", fontsize=12.0, weight="bold")
    ax_bias.set_xlim(0, 1)
    ax_bias.set_ylim(-1.03, 1.03)
    for verts, color in [
        ([(0.08, 0.0), (0.36, 0.04), (0.66, 0.54), (0.96, 0.78)], "#7B61B3"),
        ([(0.08, 0.0), (0.36, -0.04), (0.66, -0.54), (0.96, -0.78)], "#EA580C"),
    ]:
        path = MplPath(verts, [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
        ax_bias.add_patch(PathPatch(path, facecolor="none", edgecolor=color, lw=5.2, alpha=0.13, capstyle="round"))
        ax_bias.add_patch(PathPatch(path, facecolor="none", edgecolor=color, lw=1.4, alpha=0.58, capstyle="round"))
    ax_bias.scatter([0.08], [0.0], s=118, color="#CBD5E1", edgecolor="white", linewidth=1.2, zorder=5)
    ax_bias.text(0.08, -0.15, "HSC\nstart", ha="center", va="top", fontsize=7.4, color="#475569")
    ax_bias.text(0.98, 0.81, "MkP", ha="right", va="bottom", fontsize=9.1, weight="bold", color="#6D28D9")
    ax_bias.text(0.98, -0.82, "EryP", ha="right", va="top", fontsize=9.1, weight="bold", color="#EA580C")

    fork_genes = ["THBS1", "TUBB1", "GP9", "VWF", "GP1BA", "ITGA2B", "NFE2", "FLI1", "GATA1", "KLF1"]
    fork_df = df.set_index("gene").reindex(fork_genes).dropna(subset=["mk_selectivity_pp"]).reset_index()
    max_bias = max(5.0, float(np.nanmax(np.abs(fork_df["mk_selectivity_pp"]))) * 1.08)
    occupied: dict[int, int] = {}
    for _, row in fork_df.iterrows():
        gene = str(row["gene"])
        val = float(row["mk_selectivity_pp"])
        mag = min(1.0, abs(val) / max_bias)
        sign = 1.0 if val >= 0 else -1.0
        x_branch = 0.27 + 0.58 * mag
        y_branch = sign * (0.10 + 0.63 * mag)
        bin_key = int(round(10 * x_branch))
        n_seen = occupied.get(bin_key, 0)
        occupied[bin_key] = n_seen + 1
        y_branch += sign * 0.035 * (n_seen % 3)
        color = "#7B61B3" if val >= 0 else "#EA580C"
        gene_color = GENE_CLASS_COLORS.get(str(row["gene_class"]), color)
        ax_bias.annotate(
            "",
            xy=(x_branch, y_branch),
            xytext=(0.19 + 0.42 * mag, 0.0),
            arrowprops={"arrowstyle": "->", "lw": 1.2 + 2.2 * mag, "color": color, "alpha": 0.70},
            zorder=4,
        )
        ax_bias.scatter([x_branch], [y_branch], s=48 + 72 * mag, color=gene_color, edgecolor="white", linewidth=0.9, zorder=6)
        dx = 0.025 if val >= 0 else -0.025
        if gene in {"THBS1", "GP9", "NFE2", "KLF1", "GATA1"}:
            text = ax_bias.text(
                x_branch + dx,
                y_branch,
                gene,
                ha="left" if val >= 0 else "right",
                va="center",
                fontsize=7.2,
                weight="bold" if gene in {"THBS1", "GP9", "KLF1", "NFE2"} else "normal",
                color="#1E293B" if val >= 0 else "#EA580C",
                zorder=8,
            )
            text.set_path_effects([pe.withStroke(linewidth=2.5, foreground="white", alpha=0.95)])
    ax_bias.text(
        0.50,
        -0.96,
        "distance from HSC = |DeltaP(MkP)-DeltaP(EryP)|",
        ha="center",
        va="bottom",
        fontsize=6.9,
        color="#64748B",
    )
    ax_bias.set_xticks([])
    ax_bias.set_yticks([])
    for spine in ax_bias.spines.values():
        spine.set_visible(False)

    # C. Marker-response glyphs: small supporting panel, not the main result.
    marker_cols = ["delta_adt_CD41_z", "delta_adt_CD42b_z", "delta_adt_CD62P_z", "delta_adt_CD49b_z"]
    marker_labels = ["CD41", "CD42b", "CD62P", "CD49b"]
    marker_df = df.set_index("gene").reindex(marker_rows)
    marker_matrix = marker_df[marker_cols].to_numpy(dtype=float)
    vmax = max(0.12, float(np.nanmax(np.abs(marker_matrix))) * 1.04)
    ax_marker.set_title("C  Surface-marker acquisition glyphs", loc="left", fontsize=12.0, weight="bold")
    ax_marker.set_xlim(-0.58, 3.72)
    ax_marker.set_ylim(-0.76, 1.66)
    glyph_positions = [(i % 4, 1 - i // 4) for i in range(len(marker_rows))]
    for gene, vals, (cx, cy) in zip(marker_rows, marker_matrix, glyph_positions):
        _draw_surface_marker_glyph(ax_marker, float(cx), float(cy), vals, marker_labels, vmax=vmax)
        label_color = "#EA580C" if gene == "KLF1" else "#1E293B"
        text = ax_marker.text(cx, cy - 0.42, gene, ha="center", va="top", fontsize=7.4, color=label_color, weight="bold")
        text.set_path_effects([pe.withStroke(linewidth=2.3, foreground="white", alpha=0.95)])
    ax_marker.text(
        0.01,
        0.04,
        "spoke length = delta ADT z; blue spokes decrease",
        transform=ax_marker.transAxes,
        fontsize=7.2,
        color="#475569",
        va="bottom",
    )
    legend_x = 0.35
    legend_y = 1.57
    marker_colors = ["#E11D48", "#F97316", "#7C3AED", "#0F766E"]
    for k, (marker, color) in enumerate(zip(marker_labels, marker_colors)):
        ax_marker.scatter([legend_x + 0.34 * k], [legend_y], s=30, color=color, edgecolor="white", linewidth=0.5, clip_on=False)
        ax_marker.text(legend_x + 0.34 * k, legend_y - 0.10, marker, ha="center", va="top", fontsize=6.4, color="#475569", clip_on=False)
    ax_marker.set_xticks([])
    ax_marker.set_yticks([])
    for spine in ax_marker.spines.values():
        spine.set_visible(False)

    # D. Actual terminal probability change for top candidates.
    prob_rows = top_gene_order + ["NFE2", "KLF1"]
    prob_df = df.set_index("gene").reindex(list(dict.fromkeys(prob_rows))).reset_index()
    y_prob = np.arange(prob_df.shape[0])[::-1]
    base = prob_df["baseline_p_MkP"].to_numpy(dtype=float)
    plus = prob_df["plus_p_MkP"].to_numpy(dtype=float)
    ax_prob.axvline(float(np.nanmedian(base)), color="#CBD5E1", lw=1.0, zorder=0)
    for yi, (_, row), b, p in zip(y_prob, prob_df.iterrows(), base, plus):
        color = GENE_CLASS_COLORS.get(str(row["gene_class"]), "#64748B")
        xs = np.linspace(float(b), float(p), 18)
        if p >= b:
            sizes = np.linspace(12, 88, xs.shape[0])
            alphas = np.linspace(0.10, 0.88, xs.shape[0])
        else:
            sizes = np.linspace(88, 12, xs.shape[0])
            alphas = np.linspace(0.88, 0.10, xs.shape[0])
        for x_i, size_i, alpha_i in zip(xs, sizes, alphas):
            ax_prob.scatter(
                [x_i],
                [yi],
                s=float(size_i),
                color=color,
                alpha=float(alpha_i),
                edgecolor="none",
                zorder=3,
            )
        ax_prob.scatter(b, yi, s=34, color="#CBD5E1", edgecolor="white", linewidth=0.8, zorder=5)
        ax_prob.scatter(p, yi, s=82, color=color, edgecolor="white", linewidth=1.0, zorder=6)
        ax_prob.text(
            p + (0.006 if p >= b else -0.006),
            yi,
            f"{100 * (p - b):+.1f} pp",
            ha="left" if p >= b else "right",
            va="center",
            fontsize=7.1,
            color="#334155",
        )
    ax_prob.set_yticks(y_prob)
    ax_prob.set_yticklabels(prob_df["gene"], fontsize=8.2)
    ax_prob.set_xlim(0.12, 0.235)
    ax_prob.set_xlabel("mean terminal P(MkP)", fontsize=8.6)
    ax_prob.set_title("D  Terminal MkP probability comets", loc="left", fontsize=12.0, weight="bold")
    ax_prob.text(
        0.03,
        0.965,
        "gray bead: baseline\ncolored comet head: +5z",
        transform=ax_prob.transAxes,
        fontsize=7.2,
        color="#475569",
        va="top",
        bbox={"boxstyle": "round,pad=0.16", "facecolor": "white", "edgecolor": "none", "alpha": 0.86},
    )
    clean_metric_axis(ax_prob, grid_axis="x")

    fig.suptitle(
        "Fig06T. CITE-PCFM prioritizes Mk protein-transfer RNA directions from HSC starts",
        x=0.01,
        y=0.965,
        ha="left",
        fontsize=14.2,
        weight="bold",
    )
    fig.text(
        0.01,
        0.913,
        "Same 972 HSC-labeled t0 RNA seeds; each candidate gene is shifted by +5 z-score, rolled through CITE-PCFM, and read as terminal protein phenotype plus Mk surface-marker module.",
        ha="left",
        fontsize=9.0,
        color="#334155",
    )
    fig.savefig(out_png, dpi=380, bbox_inches="tight", facecolor="white")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> dict[str, object]:
    ensure_import_paths()
    if CYTOBRIDGE_SRC.exists() and str(CYTOBRIDGE_SRC) not in sys.path:
        sys.path.insert(0, str(CYTOBRIDGE_SRC))
    from CytoBridge.utils import load_model_from_adata

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(DATA_PATH)
    model_adata = ad.read_h5ad(TRAINED_MODEL_H5AD)
    _, rna_component_std, _ = projection_audit_and_scales(adata)
    directions, direction_df = build_single_gene_directions(adata, rna_component_std)

    labels = adata.obs["cell_type"].astype(str).to_numpy()
    protein_latent = np.asarray(adata.obsm["X_protein_pca"], dtype=np.float32)[:, :PROTEIN_DIM]
    phenotype_readout, classifier_metrics, confusion_df = train_mlp_protein_phenotype_readout(
        protein_latent,
        labels,
        PHENO_ORDER,
        RANDOM_STATE,
    )
    adt_decoder, adt_markers, adt_decoder_audit = fit_mk_adt_decoder(adata)

    model = load_model_from_adata(model_adata)
    readout_input = load_readout_input()
    time_points, r0 = load_trajectory_seed()
    hsc_ids, seed_audit = hsc_t0_particle_ids(adata, r0)
    r0_hsc = r0[hsc_ids]

    summary_df, delta_df, particle_df = run_gene_pushes(
        model,
        readout_input,
        phenotype_readout,
        adt_decoder,
        adt_markers,
        r0_hsc,
        time_points,
        directions,
        direction_df,
    )

    summary_path = OUT_DIR / "fig06s_mkp_hsc_gene_push_summary.csv"
    delta_path = OUT_DIR / "fig06s_mkp_hsc_gene_push_deltas.csv"
    particle_path = OUT_DIR / "fig06s_mkp_hsc_gene_push_particle_readouts.csv"
    direction_path = OUT_DIR / "fig06s_mkp_gene_latent_directions.csv"
    confusion_path = OUT_DIR / "fig06s_mkp_gene_push_readout_confusion.csv"
    adt_decoder_path = OUT_DIR / "fig06s_mkp_adt_decoder_audit.csv"
    summary_df.to_csv(summary_path, index=False)
    delta_df.to_csv(delta_path, index=False)
    particle_df.to_csv(particle_path, index=False)
    direction_df.to_csv(direction_path, index=False)
    confusion_df.to_csv(confusion_path)
    adt_decoder_audit.to_csv(adt_decoder_path, index=False)

    fig_path = FIG_DIR / "fig06s_mkp_hsc_gene_push_sensitivity.png"
    plot_mkp_gene_sensitivity(delta_df, summary_df, fig_path)
    signature_fig_path = FIG_DIR / "fig06t_mkp_hsc_gene_push_signature_map.png"
    plot_mkp_signature_map(delta_df, summary_df, signature_fig_path)

    top = delta_df.head(8)[
        [
            "gene",
            "gene_class",
            "plus_minus_baseline_p_MkP_pp",
            "plus_minus_baseline_mk_adt_module_z",
            "baseline_p_MkP",
            "plus_p_MkP",
        ]
    ].to_dict(orient="records")
    manifest = {
        "analysis": "fig06s_mkp_hsc_gene_push_sensitivity",
        "evidence_type": "model-derived HSC-start single-gene RNA-direction counterfactual",
        "figure": str(fig_path),
        "figure_pdf": str(fig_path.with_suffix(".pdf")),
        "signature_map_figure": str(signature_fig_path),
        "signature_map_figure_pdf": str(signature_fig_path.with_suffix(".pdf")),
        "tables": {
            "summary": str(summary_path),
            "deltas": str(delta_path),
            "particle_readouts": str(particle_path),
            "gene_latent_directions": str(direction_path),
            "phenotype_readout_confusion": str(confusion_path),
            "mk_adt_decoder_audit": str(adt_decoder_path),
        },
        "data_h5ad": str(DATA_PATH),
        "trained_model_h5ad": str(TRAINED_MODEL_H5AD),
        "trajectory_seed_source": str(TRAJECTORY_PATH),
        "seed_contract": seed_audit,
        "model_contract": "r0_HSC + 5z gene direction -> RNA ODE rollout -> terminal q=h_eta(r_T,T)",
        "gene_direction_contract": "single-gene +1 z-score mapped to standardized RNA SVD as std_gene * PC_gene / std_component",
        "target_phenotype": TARGET_PHENOTYPE,
        "mk_adt_markers": adt_markers,
        "push_eps_z": float(PUSH_EPS),
        "available_gene_count": int(direction_df["available"].sum()),
        "missing_genes": direction_df.loc[~direction_df["available"], "gene"].tolist(),
        "classifier_metrics": classifier_metrics,
        "top_gene_effects": top,
        "semantic_warning": "This ranking proposes model-sensitive RNA directions for MkP protein phenotype induction; it is not a validated causal target list.",
    }
    manifest_path = OUT_DIR / "fig06s_mkp_hsc_gene_push_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    main()
