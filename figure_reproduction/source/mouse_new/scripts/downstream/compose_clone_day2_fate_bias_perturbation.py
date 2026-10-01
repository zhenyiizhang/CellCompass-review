#!/usr/bin/env python3
"""In-model perturbation of day-2 fate-bias RNA modules.

The analysis uses the locked BMF model as a hypothesis generator.  It asks how
Neutrophil/Monocyte future probabilities change when early candidate genes or
mature marker genes are moved up/down in PCA space before BMF rollout.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cellcompass_mpl_cache")
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TABLE_DIR = ROOT / "downstream/cellcompass_clone_case"
FIG_DIR = ROOT / "figures/cellcompass_clone_case"
ADATA_PATH = Path("/Users/zhenyizhang/.cellcompass/algorithm_benchmarks/datasets/weinreb_rawrebuild_full_k10mindiff1/data.h5ad")
BMF_TABLE = ROOT / "downstream/clone_fate_spring_final/clone_cuvgf_bmf_final_day2_fate_probabilities.csv"
BASE_SCRIPT = ROOT / "scripts/downstream/compose_clone_scdiffeq_style_biology.py"

FATE_PALETTE = {
    "Undifferentiated": "#2E8B57",
    "Neutrophil": "#19B7C7",
    "Monocyte": "#8C564B",
    "Baso": "#7E57C2",
    "Mast": "#D95F9F",
    "Meg": "#2563EB",
    "Erythroid": "#F28E2B",
    "Other": "#CBD5E1",
}

PERTURBATION_MODULES = {
    ("Neutrophil", "early-state genes"): ["Calr", "F630028O10Rik", "Gsr", "Sell", "Dstn", "Sgms2", "Emb", "Golga3"],
    ("Neutrophil", "mature markers"): ["Cebpe", "Lyz2", "S100a8", "S100a9"],
    ("Monocyte", "early-state genes"): ["Casp6", "Sirpa", "Mef2c", "Emb", "Il6st", "Pou2f2", "Fth1", "Cd44"],
    ("Monocyte", "mature markers"): ["Irf8", "Csf1r", "Lyz2", "Ly6c2"],
}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.linewidth": 1.0,
            "axes.edgecolor": "#111827",
            "axes.titlesize": 9,
            "axes.titleweight": "bold",
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
        }
    )


def load_base():
    spec = importlib.util.spec_from_file_location("clone_bmf_base_for_perturbation", str(BASE_SCRIPT))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def run_rollout(base, model, resolved, x0: np.ndarray, delta: np.ndarray, obs: pd.DataFrame, latent: np.ndarray, fates: list[str]) -> np.ndarray:
    sim = base.simulate_bmf(model, resolved, x0 + delta[None, :], fates, snapshot_times=np.asarray([0.0, 1.0, 2.0]))
    endpoint = base.terminal_fate_probabilities(sim["final_latent"], obs, latent, fates, k=30)
    return base.aggregate_terminal_fates(sim, endpoint, x0.shape[0], len(fates))


def compute_perturbations() -> tuple[pd.DataFrame, list[str]]:
    base = load_base()
    adata = ad.read_h5ad(ADATA_PATH)
    obs = base.load_obs(adata)
    bmf = pd.read_csv(BMF_TABLE)
    fates = base.fate_names(bmf)
    valid = bmf[bmf["eligible"].astype(bool)].copy()
    latent = np.asarray(adata.obsm["X_latent"], dtype=np.float32)
    x0 = latent[valid["source_index"].to_numpy(int)].copy()
    model, resolved = base.load_bmf_model(latent.shape[1], len(fates))

    scenarios = [("baseline", "baseline", "baseline", 0.0, np.zeros(latent.shape[1], dtype=np.float32), "")]
    for (target, module), genes in PERTURBATION_MODULES.items():
        for strength in [-5.0, 5.0]:
            delta, used = base.program_delta(adata, genes, strength)
            scenarios.append((f"{target} {module} {strength:+.0f}z", target, module, strength, delta.astype(np.float32), ",".join(used)))

    rows = []
    for scenario, target, module, strength, delta, genes_used in scenarios:
        pred = run_rollout(base, model, resolved, x0, delta, obs, latent, fates)
        avg = pred.mean(axis=0)
        for fate, value in zip(fates, avg):
            rows.append(
                {
                    "scenario": scenario,
                    "target": target,
                    "module": module,
                    "strength_z": float(strength),
                    "fate": fate,
                    "future_probability": float(value),
                    "genes_used": genes_used,
                    "n_source_cells": int(x0.shape[0]),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "clone_day2_fate_bias_perturbation_rollout.csv", index=False)
    return out, fates


def target_response_table(out: pd.DataFrame) -> pd.DataFrame:
    baseline = out[out["scenario"].eq("baseline")].set_index("fate")["future_probability"].to_dict()
    rows = []
    for target in ["Neutrophil", "Monocyte"]:
        for module in ["early-state genes", "mature markers"]:
            for strength in [-5.0, 0.0, 5.0]:
                if strength == 0.0:
                    value = baseline[target]
                    genes = ""
                else:
                    d = out[out["target"].eq(target) & out["module"].eq(module) & out["strength_z"].eq(strength) & out["fate"].eq(target)]
                    value = float(d["future_probability"].iloc[0])
                    genes = str(d["genes_used"].iloc[0])
                rows.append(
                    {
                        "target": target,
                        "module": module,
                        "strength_z": strength,
                        "target_future_probability": value,
                        "delta_from_baseline": value - baseline[target],
                        "genes_used": genes,
                    }
                )
    resp = pd.DataFrame(rows)
    resp.to_csv(TABLE_DIR / "clone_day2_fate_bias_perturbation_target_response.csv", index=False)
    return resp


def effect_table(out: pd.DataFrame) -> pd.DataFrame:
    base = out[out["scenario"].eq("baseline")].set_index("fate")["future_probability"]
    rows = []
    for scenario, d in out[~out["scenario"].eq("baseline")].groupby("scenario"):
        meta = d.iloc[0]
        for _, row in d.iterrows():
            fate = str(row["fate"])
            rows.append(
                {
                    "scenario": scenario,
                    "target": meta["target"],
                    "module": meta["module"],
                    "strength_z": float(meta["strength_z"]),
                    "fate": fate,
                    "delta_future_probability": float(row["future_probability"] - base.get(fate, 0.0)),
                }
            )
    eff = pd.DataFrame(rows)
    eff.to_csv(TABLE_DIR / "clone_day2_fate_bias_perturbation_delta_by_fate.csv", index=False)
    return eff


def plot_response(ax: plt.Axes, resp: pd.DataFrame, target: str) -> None:
    color = FATE_PALETTE[target]
    styles = {
        "early-state genes": dict(color=color, lw=2.2, marker="o", ms=5, linestyle="-"),
        "mature markers": dict(color="#475569", lw=1.7, marker="o", ms=4, linestyle="--"),
    }
    for module, kwargs in styles.items():
        d = resp[resp["target"].eq(target) & resp["module"].eq(module)].sort_values("strength_z")
        ax.plot(d["strength_z"], d["target_future_probability"], label=module, **kwargs)
    ax.axvline(0, color="#CBD5E1", lw=0.9)
    ax.set_xticks([-5, 0, 5])
    ax.set_xticklabels(["-5z", "baseline", "+5z"])
    ax.set_ylabel(f"{target} future probability")
    ax.set_title(f"{target}: perturbation response", loc="left", pad=4)
    ax.grid(color="#E5E7EB", lw=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=6.7, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_delta_bar(ax: plt.Axes, eff: pd.DataFrame, target: str) -> None:
    d = eff[eff["target"].eq(target) & eff["module"].eq("early-state genes") & eff["strength_z"].eq(5.0)].copy()
    keep = ["Undifferentiated", "Neutrophil", "Monocyte", "Baso", "Mast", "Meg", "Erythroid"]
    d = d[d["fate"].isin(keep)].copy()
    d["order"] = d["fate"].map({f: i for i, f in enumerate(keep)})
    d = d.sort_values("order")
    y = np.arange(len(d))
    colors = [FATE_PALETTE.get(f, "#CBD5E1") for f in d["fate"]]
    ax.axvline(0, color="#111827", lw=0.8)
    ax.barh(y, d["delta_future_probability"], color=colors, edgecolor="white", linewidth=0.5, height=0.72)
    ax.set_yticks(y)
    ax.set_yticklabels(d["fate"])
    ax.invert_yaxis()
    ax.set_xlabel("change from baseline")
    ax.set_title(f"{target}: +5z early-gene side effects", loc="left", pad=4)
    ax.grid(axis="x", color="#E5E7EB", lw=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def make_figure(out: pd.DataFrame) -> None:
    resp = target_response_table(out)
    eff = effect_table(out)
    fig = plt.figure(figsize=(8.9, 5.4))
    gs = fig.add_gridspec(2, 2, wspace=0.28, hspace=0.42)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    plot_response(axes[0], resp, "Neutrophil")
    plot_response(axes[1], resp, "Monocyte")
    plot_delta_bar(axes[2], eff, "Neutrophil")
    plot_delta_bar(axes[3], eff, "Monocyte")
    fig.suptitle("BMF perturbation tests early RNA-state genes versus mature markers", x=0.02, y=0.99, ha="left", fontsize=12, weight="bold")
    fig.subplots_adjust(top=0.90, bottom=0.10, left=0.09, right=0.985)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIG_DIR / f"panel_c_day2_fate_bias_perturbation.{ext}", dpi=360 if ext == "png" else None)
    plt.close(fig)


def main() -> None:
    setup_style()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out, _ = compute_perturbations()
    make_figure(out)
    print(f"Wrote {FIG_DIR / 'panel_c_day2_fate_bias_perturbation.png'}")


if __name__ == "__main__":
    main()
