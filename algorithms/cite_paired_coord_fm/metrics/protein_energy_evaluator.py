from __future__ import annotations

from typing import Any
import json
import numpy as np


def _energy_distance(observed: np.ndarray, predicted: np.ndarray, *, weights_predicted: np.ndarray | None = None, max_points: int = 1500, seed: int = 0) -> float:
    observed = np.asarray(observed, dtype=np.float32)
    predicted = np.asarray(predicted, dtype=np.float32)
    if observed.ndim != 2 or predicted.ndim != 2 or observed.shape[0] == 0 or predicted.shape[0] == 0:
        return float("nan")
    rng = np.random.default_rng(int(seed))
    if observed.shape[0] > max_points:
        observed = observed[rng.choice(observed.shape[0], size=int(max_points), replace=False)]
    if predicted.shape[0] > max_points:
        if weights_predicted is not None:
            w = np.asarray(weights_predicted, dtype=np.float64).reshape(-1)
            w = np.clip(w, 0.0, np.inf)
            p = w / w.sum() if np.isfinite(w).all() and w.sum() > 0 else None
            idx = rng.choice(predicted.shape[0], size=int(max_points), replace=False, p=p)
        else:
            idx = rng.choice(predicted.shape[0], size=int(max_points), replace=False)
        predicted = predicted[idx]
    dxy = np.linalg.norm(observed[:, None, :] - predicted[None, :, :], axis=2).mean()
    dxx = np.linalg.norm(observed[:, None, :] - observed[None, :, :], axis=2).mean()
    dyy = np.linalg.norm(predicted[:, None, :] - predicted[None, :, :], axis=2).mean()
    return float(max(0.0, 2.0 * dxy - dxx - dyy))


def _cov_error(obs_r: np.ndarray, obs_p: np.ndarray, pred_r: np.ndarray, pred_p: np.ndarray, *, max_points: int, seed: int) -> float:
    if obs_r.shape[0] < 3 or pred_r.shape[0] < 3:
        return float("nan")
    rng = np.random.default_rng(int(seed))
    if obs_r.shape[0] > max_points:
        idx = rng.choice(obs_r.shape[0], size=int(max_points), replace=False)
        obs_r, obs_p = obs_r[idx], obs_p[idx]
    if pred_r.shape[0] > max_points:
        idx = rng.choice(pred_r.shape[0], size=int(max_points), replace=False)
        pred_r, pred_p = pred_r[idx], pred_p[idx]
    obs_joint = np.concatenate([obs_r, obs_p], axis=1)
    pred_joint = np.concatenate([pred_r, pred_p], axis=1)
    co = np.cov(obs_joint, rowvar=False)
    cp = np.cov(pred_joint, rowvar=False)
    rd = obs_r.shape[1]
    od = co[:rd, rd:]
    pd = cp[:rd, rd:]
    denom = np.linalg.norm(od) + 1e-8
    return float(np.linalg.norm(od - pd) / denom)


def _cite_cfg(config: dict[str, Any]) -> dict[str, Any]:
    plan = (config.get("training") or {}).get("plan") or []
    if plan and isinstance(plan[0], dict):
        if "cite_pcfm" in plan[0]:
            return plan[0].get("cite_pcfm") or {}
        fm = plan[0].get("flow_matching") or {}
        if "cite_pcfm" in fm:
            return fm.get("cite_pcfm") or {}
    return config.get("cite_pcfm", {}) if isinstance(config, dict) else {}


def evaluate_claim_metric(eval_context: Any) -> dict[str, Any]:
    """Rollout-derived CITE-PCFM protein metric bundle.

    The prediction source is the sealed EvaluationTrajectory/timepoint results.
    This evaluator never reruns inference, changes positions, rescales weights,
    or computes proteins from observed future RNA. Observed future proteins are
    scoring truth only.
    """
    params = getattr(eval_context, "metric_params", None) or {}
    config = getattr(eval_context, "config", None) or {}
    cite_cfg = _cite_cfg(config) if isinstance(config, dict) else {}
    rna_dim = int(params.get("rna_dim", cite_cfg.get("rna_dim", 50)))
    protein_dim = int(params.get("protein_dim", cite_cfg.get("protein_dim", 30)))
    max_points = int(params.get("energy_max_points", params.get("max_points", 1500)))
    seed = int(params.get("seed", 0))

    protein_vals: list[float] = []
    rna_vals: list[float] = []
    cov_vals: list[float] = []
    per_time: list[dict[str, float]] = []

    for pos, result in enumerate(getattr(eval_context, "timepoint_results", []) or []):
        obs = np.asarray(result.observed_points, dtype=np.float32)
        pred = np.asarray(result.predicted_points, dtype=np.float32)
        weights = np.asarray(result.predicted_weights, dtype=np.float32).reshape(-1) if getattr(result, "predicted_weights", None) is not None else None
        if obs.ndim != 2 or pred.ndim != 2 or obs.shape[1] < rna_dim + protein_dim or pred.shape[1] < rna_dim + protein_dim:
            continue
        obs_r, obs_p = obs[:, :rna_dim], obs[:, rna_dim : rna_dim + protein_dim]
        pred_r, pred_p = pred[:, :rna_dim], pred[:, rna_dim : rna_dim + protein_dim]
        p_energy = _energy_distance(obs_p, pred_p, weights_predicted=weights, max_points=max_points, seed=seed + pos)
        r_energy = _energy_distance(obs_r, pred_r, weights_predicted=weights, max_points=max_points, seed=seed + 101 + pos)
        protein_vals.append(p_energy)
        rna_vals.append(r_energy)
        entry = {
            "time_index": float(getattr(result, "time_index", pos + 1)),
            "time_point": float(getattr(result, "time_point", pos + 1.0)),
            "protein_energy": float(p_energy),
            "rna_energy_guardrail": float(r_energy),
        }
        cov = _cov_error(obs_r, obs_p, pred_r, pred_p, max_points=max_points, seed=seed + 503 + pos)
        if np.isfinite(cov):
            cov_vals.append(cov)
            entry["paired_cov_error"] = float(cov)
        per_time.append(entry)

    return {
        "protein_energy": float(np.nanmean(protein_vals)) if protein_vals else float("nan"),
        "rna_energy_guardrail": float(np.nanmean(rna_vals)) if rna_vals else float("nan"),
        "paired_cov_error": float(np.nanmean(cov_vals)) if cov_vals else float("nan"),
        "protein_energy_per_time_json": json.dumps(per_time, ensure_ascii=False, sort_keys=True),
        "protein_energy_per_time_count": int(len(per_time)),
        "claim_metric_semantics": "sealed EvaluationTrajectory z=(RNA, protein_readout chart) protein energy; observed proteins scoring truth only; no prediction repair",
    }
