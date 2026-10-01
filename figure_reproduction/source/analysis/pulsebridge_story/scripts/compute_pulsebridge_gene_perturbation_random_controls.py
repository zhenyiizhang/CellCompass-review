#!/usr/bin/env python3
"""Random-push controls for PulseBridge gene perturbation rollout.

The main perturbation figure asks whether biologically chosen gene-program
directions propagate through the trained dynamics.  This control asks whether a
same-norm random latent push would produce comparable terminal program changes.
It is intentionally kept as a table-level check rather than another busy panel.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


REPO = Path(__file__).resolve().parents[3]
BASE_SCRIPT = REPO / "analysis" / "pulsebridge_story" / "scripts" / "make_pulsebridge_gene_perturbation_rollout.py"
TAB = REPO / "analysis" / "pulsebridge_story" / "outputs" / "tables"
N_RANDOM = 24
N_CELLS = 1200
SEED = 17


def load_base_module():
    spec = importlib.util.spec_from_file_location("pulsebridge_gene_perturb_base", str(BASE_SCRIPT))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def score_points(mod, points, nn, expr_ref, zexpr_ref, gene_names):
    _expr, zexpr = mod.neighbor_gene_readout(nn, points, expr_ref, zexpr_ref)
    return {k: float(v.mean()) for k, v in mod.program_scores(zexpr, gene_names).items()}


def summarize(long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    actual = long[long["control_type"].eq("actual")].copy()
    random = long[long["control_type"].eq("random")].copy()
    for key, a in actual.groupby(["perturbation_module", "strength_z", "program"]):
        r = random[
            random["perturbation_module"].eq(key[0])
            & random["strength_z"].eq(key[1])
            & random["program"].eq(key[2])
        ]
        if r.empty:
            continue
        aval = float(a["terminal_delta_from_baseline"].iloc[0])
        rvals = r["terminal_delta_from_baseline"].to_numpy(dtype=float)
        sd = float(np.std(rvals, ddof=1)) if len(rvals) > 1 else float("nan")
        rows.append(
            {
                "perturbation_module": key[0],
                "strength_z": float(key[1]),
                "program": key[2],
                "actual_terminal_delta": aval,
                "random_mean_terminal_delta": float(np.mean(rvals)),
                "random_sd_terminal_delta": sd,
                "actual_minus_random_mean": float(aval - np.mean(rvals)),
                "signed_z_vs_random": float((aval - np.mean(rvals)) / sd) if sd > 1e-12 else float("nan"),
                "random_abs_ge_actual_fraction": float(np.mean(np.abs(rvals) >= abs(aval))),
                "n_random": int(len(rvals)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    mod = load_base_module()
    rng = np.random.default_rng(SEED)
    adata = ad.read_h5ad(mod.ADATA_PATH)
    latent = np.asarray(adata.obsm["X_latent"], dtype=np.float32)
    times = adata.obs["time_point_processed"].astype(float).to_numpy()
    day0_idx = np.where(times == float(np.min(times)))[0]
    day3_idx = np.where(times == float(np.max(times)))[0]
    use = rng.choice(day0_idx, size=min(N_CELLS, len(day0_idx)), replace=False)
    x0 = latent[use]

    model, _record = mod.load_model(latent.shape[1])
    gene_names, expr, zexpr = mod.selected_expression(adata)
    nn_day0 = NearestNeighbors(n_neighbors=30).fit(latent[day0_idx])
    nn_day3 = NearestNeighbors(n_neighbors=30).fit(latent[day3_idx])
    base_init = score_points(mod, x0, nn_day0, expr[day0_idx], zexpr[day0_idx], gene_names)
    base_terminal_points = mod.euler_rollout(model, x0)
    base_terminal = score_points(mod, base_terminal_points, nn_day3, expr[day3_idx], zexpr[day3_idx], gene_names)

    rows = []
    for module, meta in mod.PERTURBATION_MODULES.items():
        for strength in (-5.0, 5.0):
            actual_delta, used = mod.program_delta(adata, meta["genes"], strength)
            directions = [("actual", 0, actual_delta)]
            target_norm = float(np.linalg.norm(actual_delta))
            for i in range(N_RANDOM):
                d = rng.normal(size=actual_delta.shape).astype(np.float32)
                d *= target_norm / max(float(np.linalg.norm(d)), 1e-8)
                directions.append(("random", i + 1, d))
            for control_type, control_id, delta in directions:
                init_scores = score_points(mod, x0 + delta[None, :], nn_day0, expr[day0_idx], zexpr[day0_idx], gene_names)
                terminal_points = mod.euler_rollout(model, x0 + delta[None, :])
                terminal_scores = score_points(mod, terminal_points, nn_day3, expr[day3_idx], zexpr[day3_idx], gene_names)
                for program in [
                    "source_aligned_antitumor",
                    "reactivation",
                    "immune_surface",
                    "proliferation_repression",
                    "adaptive_metabolic",
                    "therapeutic_margin",
                ]:
                    rows.append(
                        {
                            "perturbation_module": module,
                            "strength_z": float(strength),
                            "program": program,
                            "control_type": control_type,
                            "control_id": int(control_id),
                            "delta_norm": target_norm,
                            "initial_delta_from_baseline": float(init_scores[program] - base_init[program]),
                            "terminal_delta_from_baseline": float(terminal_scores[program] - base_terminal[program]),
                            "dynamic_gain_over_initial": float(
                                (terminal_scores[program] - base_terminal[program])
                                - (init_scores[program] - base_init[program])
                            ),
                            "genes_used": ",".join(used),
                            "n_cells": int(x0.shape[0]),
                        }
                    )

    long = pd.DataFrame(rows)
    summary = summarize(long)
    TAB.mkdir(parents=True, exist_ok=True)
    long.to_csv(TAB / "pulsebridge_gene_perturbation_random_control_long.csv", index=False)
    summary.to_csv(TAB / "pulsebridge_gene_perturbation_random_control_summary.csv", index=False)
    print(TAB / "pulsebridge_gene_perturbation_random_control_long.csv")
    print(TAB / "pulsebridge_gene_perturbation_random_control_summary.csv")


if __name__ == "__main__":
    main()
