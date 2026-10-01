#!/usr/bin/env python3
"""Random-velocity negative control for PCIE-matched gene velocity UMAP."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


REPO = Path(__file__).resolve().parents[3]
SCRIPT = Path(__file__).resolve().parent / "plot_pcie_gene_velocity_umap.py"
OUT = REPO / "analysis" / "pulsebridge_story" / "outputs"
FIG = OUT / "figures"
TAB = OUT / "tables"
BUNDLE = REPO / "downloads" / "pulsebridge_welltemp_hct116_20260627"
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
METRICS_PATH = TRAJ_PATH.parent / "metrics.json"
UMAP_H5AD = OUT / "downstream" / "dynamo_stream" / "pulsebridge_full_umap_velocity_for_dynamo.h5ad"


def load_base_module():
    spec = importlib.util.spec_from_file_location("plot_pcie_gene_velocity_umap", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def compute_random_velocity_values(genes: list[str], seed: int = 20260628):
    base = load_base_module()
    rng = np.random.default_rng(seed)
    adata = ad.read_h5ad(ADATA_PATH)
    umap_adata = ad.read_h5ad(UMAP_H5AD)
    if list(umap_adata.obs_names) != list(adata.obs_names):
        raise RuntimeError("UMAP h5ad obs_names do not match data h5ad obs_names")
    umap = np.asarray(umap_adata.obsm["X_umap"], dtype=np.float32)
    times = adata.obs["time_point_processed"].astype(float).to_numpy()

    with METRICS_PATH.open() as f:
        metrics = json.load(f)
    custom = metrics.get("custom_metrics", {})
    nuisance_rank = int(custom.get("pcib_nuisance_rank", 3))

    latent = np.asarray(adata.obsm["X_latent"], dtype=np.float32)
    metabolic = np.asarray(adata.obsm["X_metabolic_velocity"], dtype=np.float32)
    obs = adata.obs.copy()
    split = obs["mvab_anchor_split"].astype(str).to_numpy()
    confidence = obs["mvab_anchor_confidence"].astype(float).to_numpy().astype(np.float32)
    pcs = np.asarray(adata.varm["PCs"], dtype=np.float32)[:, : latent.shape[1]]
    gene_to_idx = {str(g): int(adata.var_names.get_loc(g)) for g in genes if g in set(adata.var_names)}
    genes = [g for g in genes if g in gene_to_idx]
    gene_loadings = np.stack([pcs[gene_to_idx[g]] for g in genes], axis=0).astype(np.float32)

    basis = base.build_nuisance_basis(latent, obs, rank=nuisance_rank, rng=rng)
    z = np.load(TRAJ_PATH)
    time_points = np.asarray(z["time_points"], dtype=float)
    observed_indices = np.asarray(z["observed_time_indices"], dtype=int)
    observed_times = np.asarray(z["observed_time_points"], dtype=float)

    rows = []
    norm_reference = []
    for time_i, traj_idx in enumerate(observed_indices):
        traj_idx = int(traj_idx)
        if traj_idx + 1 >= len(time_points):
            continue
        anchor_time = float(observed_times[time_i])
        anchor_rows = np.where((times == anchor_time) & (split == "heldout") & (confidence > 0))[0]
        if anchor_rows.size == 0:
            continue
        anchor_x = latent[anchor_rows]
        pulse_perp = base.project_perp(metabolic[anchor_rows], basis)
        anchor_conf = confidence[anchor_rows]
        valid = (np.linalg.norm(pulse_perp, axis=1) > 1e-8) & (anchor_conf > 0)
        anchor_rows = anchor_rows[valid]
        anchor_x = anchor_x[valid]
        pulse_perp = pulse_perp[valid]
        anchor_conf = anchor_conf[valid]
        if anchor_rows.size == 0:
            continue

        cur = np.asarray(z[f"points_{traj_idx:06d}"], dtype=np.float32)
        nxt = np.asarray(z[f"points_{traj_idx + 1:06d}"], dtype=np.float32)
        n = min(len(cur), len(nxt))
        cur, nxt = cur[:n], nxt[:n]
        if n > 4096:
            particle_idx = rng.choice(np.arange(n), size=4096, replace=False)
            cur_ref = cur[particle_idx]
            nxt_ref = nxt[particle_idx]
        else:
            cur_ref, nxt_ref = cur, nxt
        dt = max(abs(float(time_points[traj_idx + 1] - time_points[traj_idx])), 1e-8)
        generated_velocity = (nxt_ref - cur_ref) / dt

        nn = NearestNeighbors(n_neighbors=1).fit(cur_ref)
        dist, ind = nn.kneighbors(anchor_x)
        matched_generated = generated_velocity[ind[:, 0]]
        matched_generated_perp = base.project_perp(matched_generated, basis)
        generated_norm = np.linalg.norm(matched_generated_perp, axis=1)
        norm_reference.extend(generated_norm.tolist())

        random_velocity = rng.normal(size=matched_generated_perp.shape).astype(np.float32)
        random_velocity = base.project_perp(random_velocity, basis)
        random_norm = np.linalg.norm(random_velocity, axis=1, keepdims=True)
        # Same speed scale as the matched generated tangent, but random direction.
        random_velocity = random_velocity / np.maximum(random_norm, 1e-8) * generated_norm.reshape(-1, 1)

        random_gene = random_velocity @ gene_loadings.T
        pulse_gene = pulse_perp @ gene_loadings.T
        denom = np.maximum(np.abs(random_gene) + np.abs(pulse_gene), 1e-8)
        signed_agreement = (random_gene * pulse_gene) / denom

        for local_j, gene in enumerate(genes):
            for i, cell_idx in enumerate(anchor_rows):
                rows.append(
                    {
                        "gene": gene,
                        "anchor_time": anchor_time,
                        "cell_index": int(cell_idx),
                        "cell": str(adata.obs_names[cell_idx]),
                        "umap_1": float(umap[cell_idx, 0]),
                        "umap_2": float(umap[cell_idx, 1]),
                        "confidence": float(anchor_conf[i]),
                        "anchor_distance": float(dist[i, 0]),
                        "random_gene_velocity": float(random_gene[i, local_j]),
                        # Keep these column names so the shared plotting helper can be reused.
                        "pulsebridge_gene_velocity": float(random_gene[i, local_j]),
                        "metabolic_gene_velocity": float(pulse_gene[i, local_j]),
                        "signed_agreement": float(signed_agreement[i, local_j]),
                        "same_sign": bool(np.sign(random_gene[i, local_j]) == np.sign(pulse_gene[i, local_j])),
                    }
                )
        print(f"[random-velocity] anchor_time={anchor_time:g} pairs={len(anchor_rows)}", flush=True)
    return pd.DataFrame(rows), umap, times, genes, float(np.median(norm_reference))


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    base = load_base_module()
    genes = base.selected_genes()
    df, umap, times, genes, median_generated_norm = compute_random_velocity_values(genes)
    df = base.add_scaled_columns(df)
    table_path = TAB / "pcie_random_velocity_umap_values.csv"
    figure_path = FIG / "story_pcie_random_velocity_umap.png"
    df.to_csv(table_path, index=False)
    base.draw_umap_panels(
        df,
        umap,
        times,
        genes,
        output_path=figure_path,
        title="Random latent velocity versus metabolic pulse on PCIE anchors",
        first_col_title="random latent velocity",
    )
    summary = (
        df.groupby("gene")
        .agg(
            n=("same_sign", "size"),
            same_sign_fraction=("same_sign", "mean"),
            mean_signed_agreement=("signed_agreement", "mean"),
            mean_scaled_signed_agreement=("signed_agreement_scaled", "mean"),
        )
        .reset_index()
        .sort_values("same_sign_fraction", ascending=False)
    )
    summary_path = TAB / "pcie_random_velocity_umap_summary.csv"
    summary.to_csv(summary_path, index=False)
    manifest = {
        "analysis": "random latent velocity negative control for selected PCIE genes",
        "seed": 20260628,
        "selected_genes": genes,
        "row_count": int(len(df)),
        "anchor_cell_count_per_gene": int(len(df) / max(len(genes), 1)),
        "random_velocity": "Gaussian random latent directions, nuisance-projected, then scaled to each matched generated tangent norm; directions are independent of PulseBridge and metabolic pulse.",
        "median_matched_generated_tangent_norm": median_generated_norm,
        "outputs": {
            "figure": str(figure_path),
            "values_table": str(table_path),
            "summary_table": str(summary_path),
        },
    }
    (OUT / "pcie_random_velocity_umap_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
