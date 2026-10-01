from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch

from CytoBridge.tl.fit import build_default_training_data, validate_training_data_bundle
from CytoBridge.tl.flow_matching_backends import (
    ChunkedTransportCouplingStrategy,
    FlowMatchingBackend,
    FlowMatchingBatch,
    LinearDeterministicConditionalPath,
    NullMassStrategy,
    PairwiseCost,
)
from CytoBridge.tl.models import DynamicalModel
from CytoBridge.tl.training_algorithm import (
    EvaluationMetricsContext,
    FlowMatchingBuildContext,
    FlowMatchingLossContext,
    FlowMatchingLossResult,
    ModelBuildContext,
    TrainingAlgorithmSpec,
    TrainingDataBuilderContext,
    TrainingDataBundle,
)

_EPS = 1.0e-8
EVALUATOR_ID = "pulse_control_identifiability_bridge.claim_metric:evaluate_claim_metric:v1"


@dataclass
class PCIBPayload:
    """AnnData-row-aligned pulse/control objects used by PCIB."""

    pulse: np.ndarray
    pulse_perp: np.ndarray
    confidence: np.ndarray
    effective_confidence: np.ndarray
    split_code: np.ndarray  # 1=train/path-visible, 2=heldout/metric-visible, 0=unusable
    nuisance_basis: np.ndarray
    singular_values: np.ndarray
    nuisance_budget: float
    support_diagnostics: dict[str, Any]
    train_count: int
    heldout_count: int
    control_count: int
    labeled0_count: int
    source: str
    notes: str


# -----------------------------------------------------------------------------
# Small numerical utilities
# -----------------------------------------------------------------------------


def _as_numpy(value: Any, *, dtype=np.float32) -> np.ndarray:
    if value is None:
        raise ValueError("cannot convert None to numpy")
    if hasattr(value, "toarray"):
        value = value.toarray()
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _safe_norm(x: np.ndarray, axis: int = -1, keepdims: bool = False) -> np.ndarray:
    return np.linalg.norm(x, axis=axis, keepdims=keepdims)


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    n = _safe_norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, _EPS)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(finite):
        finite_values = values[np.isfinite(values)]
        return float(np.quantile(finite_values, q)) if finite_values.size else 0.0
    values = values[finite]
    weights = weights[finite]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cdf = np.cumsum(weights) / max(float(np.sum(weights)), _EPS)
    return float(values[min(int(np.searchsorted(cdf, float(q), side="left")), values.size - 1)])


def _orthonormalize_columns(mat: np.ndarray, latent_dim: int) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)
    if mat.ndim != 2 or mat.size == 0:
        return np.zeros((latent_dim, 0), dtype=np.float32)
    q, _ = np.linalg.qr(mat.astype(np.float64, copy=False))
    q = q[:, : min(q.shape[1], mat.shape[1])]
    return q.astype(np.float32, copy=False)


def _project_perp(x: np.ndarray, basis: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    basis = np.asarray(basis, dtype=np.float32)
    if basis.ndim != 2 or basis.shape[1] == 0:
        return x.astype(np.float32, copy=False)
    return (x - (x @ basis) @ basis.T).astype(np.float32, copy=False)


def _nearest_indices(query: np.ndarray, reference: np.ndarray, *, chunk: int = 512, device: Optional[torch.device] = None) -> np.ndarray:
    query = np.asarray(query, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    if query.shape[0] == 0 or reference.shape[0] == 0:
        return np.zeros(query.shape[0], dtype=np.int64)
    dev = device if device is not None and torch.device(device).type == "cuda" else torch.device("cpu")
    ref = torch.as_tensor(reference, dtype=torch.float32, device=dev)
    out: list[np.ndarray] = []
    block = max(int(chunk), 1)
    for start in range(0, query.shape[0], block):
        q = torch.as_tensor(query[start : start + block], dtype=torch.float32, device=dev)
        dist = torch.cdist(q, ref).pow(2)
        out.append(torch.argmin(dist, dim=1).detach().cpu().numpy().astype(np.int64))
    return np.concatenate(out, axis=0) if out else np.zeros(0, dtype=np.int64)


def _pcib_config(config: dict[str, Any] | None) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "nuisance_rank": 3,
        "nuisance_rank_cap": 5,
        "baseline_pair_max": 4096,
        "nuisance_energy_threshold": 0.90,
        "min_anchor_confidence": 0.05,
        "max_train_anchors_per_time": 4096,
        "max_metric_anchors_per_time": 2048,
        "max_metric_particles_per_time": 4096,
        "nearest_chunk": 512,
        "endpoint_cost_weight": 1.0,
        "pulse_cost_weight": 0.35,
        "nuisance_cost_weight": 0.15,
        "residual_cost_weight": 0.02,
        "selector_tau": 0.25,
        "residual_scale": 0.20,
        "residual_max_norm": 5.0,
        "velocity_loss_weight": 0.02,
        "velocity_leakage_weight": 0.01,
        "metric_leakage_weight": 0.25,
        "nuisance_budget_quantile": 0.90,
        "nuisance_budget_floor": 0.02,
        "nuisance_budget_ceiling": 0.50,
        "min_train_anchor_support_per_time": 100,
        "min_projected_pulse_fraction": 0.10,
        "endpoint_min_pulse_cosine": -0.25,
        "residual_endpoint_fraction": 0.50,
        "metric_subsample_seed": 1729,
        "claim_metric_name": "pcib_pulse_control_identifiability_error",
        "unsupported_metric_value": 1.0e6,
        "min_baseline_control_count": 10,
        "min_baseline_labeled0_train_count": 10,
        "require_nonempty_nuisance_basis": True,
    }
    if isinstance(config, dict):
        merged = dict(defaults)
        if isinstance(config.get("pcib"), dict):
            merged.update(config.get("pcib") or {})
        # Stage-local pcib values may be passed as the config itself.
        for key in defaults:
            if key in config:
                merged[key] = config[key]
        return merged
    return defaults


# -----------------------------------------------------------------------------
# Train-only nuisance/pulse construction
# -----------------------------------------------------------------------------


def _split_code_from_obs(adata) -> np.ndarray:
    if "mvab_anchor_split" not in adata.obs:
        raise ValueError("PCIB requires obs['mvab_anchor_split'] with train/heldout/unusable anchors.")
    split = np.asarray(adata.obs["mvab_anchor_split"].astype(str).values)
    allowed = {"train", "heldout", "unusable"}
    observed = set(str(x) for x in np.unique(split))
    unknown = sorted(observed.difference(allowed))
    if unknown:
        raise ValueError(
            "PCIB requires obs['mvab_anchor_split'] values in {'train','heldout','unusable'}; "
            f"found unsupported values {unknown}."
        )
    code = np.zeros(split.shape[0], dtype=np.int8)
    code[split == "train"] = 1
    code[split == "heldout"] = 2
    if int(np.sum(code == 1)) == 0 or int(np.sum(code == 2)) == 0:
        raise ValueError(
            "PCIB requires non-empty train and heldout mvab_anchor_split anchors; "
            f"train={int(np.sum(code == 1))}, heldout={int(np.sum(code == 2))}."
        )
    return code


def _confidence_from_obs(adata, cfg: dict[str, Any]) -> np.ndarray:
    if "mvab_anchor_confidence" in adata.obs:
        conf = np.asarray(adata.obs["mvab_anchor_confidence"].values, dtype=np.float32)
    else:
        raise ValueError("PCIB requires obs['mvab_anchor_confidence']; missing confidence cannot be silently imputed.")
    conf = np.nan_to_num(conf, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    conf[conf < float(cfg.get("min_anchor_confidence", 0.05))] = 0.0
    return conf


def _metabolic_pulse_from_adata(adata) -> np.ndarray:
    if "X_metabolic_velocity" in adata.obsm:
        pulse = _as_numpy(adata.obsm["X_metabolic_velocity"], dtype=np.float32)
        source = "obsm['X_metabolic_velocity']"
    elif "X_metabolic_synthesis" in adata.obsm and "X_metabolic_decay" in adata.obsm:
        pulse = _as_numpy(adata.obsm["X_metabolic_synthesis"], dtype=np.float32) - _as_numpy(
            adata.obsm["X_metabolic_decay"], dtype=np.float32
        )
        source = "obsm['X_metabolic_synthesis'] - obsm['X_metabolic_decay']"
    else:
        raise ValueError(
            "PCIB requires obsm['X_metabolic_velocity'] or both synthesis/decay metabolic sidecars."
        )
    if pulse.ndim != 2:
        raise ValueError(f"PCIB metabolic pulse must be 2D, got shape={pulse.shape} from {source}.")
    return np.nan_to_num(pulse, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def _validate_pcib_data_contract(adata) -> None:
    """Fail fast on proposal-required PCIB fields before fitting side information."""

    required_obs = [
        "time_point_processed",
        "sample_name",
        "treatment",
        "metabolic_label_4sU",
        "treated_5AZA_CdR",
        "labeling_duration",
        "decay_rate",
        "mvab_anchor_split",
        "mvab_anchor_confidence",
    ]
    missing_obs = [key for key in required_obs if key not in adata.obs]
    missing_layers = [key for key in ("new", "old") if key not in adata.layers]
    missing_obsm = []
    if "X_latent" not in adata.obsm:
        missing_obsm.append("X_latent")
    has_velocity = "X_metabolic_velocity" in adata.obsm
    has_components = "X_metabolic_synthesis" in adata.obsm and "X_metabolic_decay" in adata.obsm
    if not (has_velocity or has_components):
        missing_obsm.append("X_metabolic_velocity or both X_metabolic_synthesis/X_metabolic_decay")
    if missing_obs or missing_layers or missing_obsm:
        raise ValueError(
            "PCIB proposal data contract is incomplete: "
            f"missing_obs={missing_obs}, missing_layers={missing_layers}, missing_obsm={missing_obsm}."
        )
    x_latent = _as_numpy(adata.obsm["X_latent"], dtype=np.float32)
    if x_latent.ndim != 2 or x_latent.shape[0] != adata.n_obs or x_latent.shape[1] == 0:
        raise ValueError(f"PCIB requires row-aligned 2D obsm['X_latent']; got shape={x_latent.shape}.")
    if not np.all(np.isfinite(x_latent)):
        raise ValueError("PCIB requires finite obsm['X_latent'] values.")
    time = np.asarray(adata.obs["time_point_processed"].values, dtype=float)
    finite_time = time[np.isfinite(time)]
    if finite_time.size != adata.n_obs or np.unique(finite_time).size < 2:
        raise ValueError("PCIB requires finite obs['time_point_processed'] with at least two observed times.")


def _baseline_masks(adata, split_code: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if "sample_name" not in adata.obs or "treatment" not in adata.obs:
        raise ValueError("PCIB requires obs['sample_name'] and obs['treatment'] to identify baseline arms.")
    sample = np.asarray(adata.obs["sample_name"].astype(str).values)
    treatment = np.asarray(adata.obs["treatment"].astype(str).values)
    time = np.asarray(adata.obs["time_point_processed"].values, dtype=float)
    control = (time == float(np.min(time))) & ((sample == "HCT116_control") | (treatment == "control"))
    labeled0 = (time == float(np.min(time))) & (
        (sample == "HCT116_0d") | (treatment == "5-AZA-CdR_time0") | ((sample != "HCT116_control") & (treatment != "control"))
    )
    # Labeled baseline uses train anchors only to avoid heldout leakage; control has
    # no metabolic anchors in this prepared panel, so all control baseline cells are
    # allowed as the negative-control side of the train-only instrument.
    labeled0 = labeled0 & (split_code == 1)
    return control, labeled0


def _fit_nuisance_basis(adata, cfg: dict[str, Any], split_code: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    x = _as_numpy(adata.obsm["X_latent"], dtype=np.float32)
    latent_dim = int(x.shape[1])
    confidence = _confidence_from_obs(adata, cfg)
    control_mask, labeled0_mask = _baseline_masks(adata, split_code)
    control_idx = np.where(control_mask)[0]
    labeled_idx = np.where(labeled0_mask)[0]
    if control_idx.size == 0 or labeled_idx.size == 0:
        raise ValueError(
            "PCIB requires both HCT116_control baseline cells and HCT116_0d train baseline anchors; "
            f"found control_count={int(control_idx.size)}, labeled0_train_count={int(labeled_idx.size)}. "
            "A missing baseline arm would silently collapse the approved nuisance-control mechanism."
        )
    min_control = int(cfg.get("min_baseline_control_count", 10))
    min_labeled = int(cfg.get("min_baseline_labeled0_train_count", 10))
    if control_idx.size < min_control or labeled_idx.size < min_labeled:
        raise ValueError(
            "PCIB baseline instrument support is below the required proposal threshold: "
            f"control_count={int(control_idx.size)} (min {min_control}), "
            f"labeled0_train_count={int(labeled_idx.size)} (min {min_labeled})."
        )

    max_pairs = min(int(cfg.get("baseline_pair_max", 4096)), int(control_idx.size), int(labeled_idx.size))
    rng = np.random.default_rng(1729)
    if labeled_idx.size > max_pairs:
        labeled_idx = rng.choice(labeled_idx, size=max_pairs, replace=False)
    # Pair each labeled/IAA baseline train anchor with its nearest unlabeled control
    # cell. This is a train-only negative-control displacement instrument.
    control_x = x[control_idx]
    labeled_x = x[labeled_idx]
    nn = _nearest_indices(labeled_x, control_x, chunk=int(cfg.get("nearest_chunk", 512)), device=None)
    paired_control_x = control_x[nn]
    displacements = labeled_x - paired_control_x
    displacements = np.nan_to_num(displacements, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    baseline_weights = np.asarray(confidence[labeled_idx], dtype=np.float32)
    baseline_weights = np.nan_to_num(baseline_weights, nan=0.0, posinf=0.0, neginf=0.0)
    baseline_weights = np.maximum(baseline_weights, float(cfg.get("min_anchor_confidence", 0.05)))

    if displacements.shape[0] == 0 or float(np.linalg.norm(displacements)) <= _EPS:
        basis = np.zeros((latent_dim, 0), dtype=np.float32)
        singular = np.zeros(0, dtype=np.float32)
        nuisance_budget = 0.0
    else:
        # Uncentered confidence-weighted SVD preserves the systematic mean
        # control->labeled displacement as a possible nuisance axis while honoring
        # proposal-specified train-only baseline confidence weights.
        sqrt_w = np.sqrt(np.maximum(baseline_weights, _EPS)).reshape(-1, 1)
        weighted_displacements = displacements.astype(np.float64, copy=False) * sqrt_w.astype(np.float64, copy=False)
        u, s, vt = np.linalg.svd(weighted_displacements, full_matrices=False)
        del u
        rank_cap = max(0, min(int(cfg.get("nuisance_rank_cap", 5)), vt.shape[0], latent_dim))
        requested = max(0, min(int(cfg.get("nuisance_rank", 3)), rank_cap))
        if requested == 0 and rank_cap > 0:
            energy = np.cumsum(s * s) / max(float(np.sum(s * s)), _EPS)
            requested = int(np.searchsorted(energy, float(cfg.get("nuisance_energy_threshold", 0.90))) + 1)
            requested = max(1, min(requested, rank_cap))
        basis = _orthonormalize_columns(vt[:requested].T, latent_dim)
        singular = s[:rank_cap].astype(np.float32, copy=False)
        if basis.shape[1] > 0:
            coeff = displacements @ basis
            nuisance_fraction = np.sum(coeff * coeff, axis=1) / np.maximum(
                np.sum(displacements * displacements, axis=1), _EPS
            )
            finite_budget = np.isfinite(nuisance_fraction) & np.isfinite(baseline_weights) & (baseline_weights > 0)
            if np.any(finite_budget):
                nuisance_budget = float(
                    _weighted_quantile(
                        nuisance_fraction[finite_budget],
                        baseline_weights[finite_budget],
                        float(cfg.get("nuisance_budget_quantile", 0.90)),
                    )
                )
                nuisance_budget = max(float(cfg.get("nuisance_budget_floor", 0.02)), nuisance_budget)
                nuisance_budget = min(float(cfg.get("nuisance_budget_ceiling", 0.50)), nuisance_budget)
            else:
                nuisance_budget = float(cfg.get("nuisance_budget_floor", 0.02))
        else:
            nuisance_budget = 0.0

    return basis, singular, {
        "control_count": int(control_idx.size),
        "labeled0_train_count": int(labeled_idx.size),
        "paired_count": int(displacements.shape[0]),
        "confidence_weighted": True,
        "baseline_weight_mean": float(np.mean(baseline_weights)) if baseline_weights.size else 0.0,
        "nuisance_budget": float(nuisance_budget),
        "notes": "nearest-neighbor HCT116_control -> HCT116_0d train-only baseline displacement confidence-weighted SVD",
    }


def _support_gated_confidence(
    adata,
    cfg: dict[str, Any],
    *,
    split_code: np.ndarray,
    confidence: np.ndarray,
    pulse: np.ndarray,
    pulse_perp: np.ndarray,
    nuisance_basis: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Deterministically suppress pulse terms when support/collinearity is weak.

    The approved PCIB selector is endpoint-first: side-information terms may
    choose among endpoint-valid paths only when train-visible pulse support is
    sufficiently strong and not almost entirely explained by the baseline
    nuisance subspace.  This function implements that feasibility gate once in
    the training-data builder, before coupling/path/loss construction.  Heldout
    confidence is intentionally left unchanged for evaluation metrics.
    """

    time = np.asarray(adata.obs["time_point_processed"].values, dtype=float)
    pulse_norm = _safe_norm(pulse, axis=1)
    perp_norm = _safe_norm(pulse_perp, axis=1)
    perp_fraction = perp_norm / np.maximum(pulse_norm, _EPS)
    if nuisance_basis.ndim == 2 and nuisance_basis.shape[1] > 0:
        coeff = pulse @ nuisance_basis
        nuisance_fraction = np.sum(coeff * coeff, axis=1) / np.maximum(np.sum(pulse * pulse, axis=1), _EPS)
    else:
        nuisance_fraction = np.zeros(pulse.shape[0], dtype=np.float32)

    effective = np.asarray(confidence, dtype=np.float32).copy()
    min_support = int(cfg.get("min_train_anchor_support_per_time", 100))
    min_perp = float(cfg.get("min_projected_pulse_fraction", 0.10))
    diagnostics: dict[str, Any] = {
        "min_train_anchor_support_per_time": min_support,
        "min_projected_pulse_fraction": min_perp,
        "by_time": {},
    }

    for t in sorted(np.unique(time)):
        train_mask = (time == float(t)) & (split_code == 1) & (confidence > 0) & np.isfinite(perp_fraction)
        valid = train_mask & (pulse_norm > _EPS)
        n_valid = int(np.sum(valid))
        median_perp = float(np.median(perp_fraction[valid])) if n_valid else 0.0
        mean_nuisance = float(np.mean(nuisance_fraction[valid])) if n_valid else 0.0
        if n_valid < min_support:
            gate = 0.0
            reason = "low_train_anchor_support"
        elif median_perp < min_perp:
            gate = 0.0
            reason = "pulse_collinear_with_nuisance"
        else:
            # Smoothly downweight near the declared minimum rather than abruptly
            # changing the method for borderline panels.  On the current HCT116
            # panel this gate is near one; it mainly protects future panels and
            # negative controls.
            gate = float(np.clip((median_perp - min_perp) / max(1.0 - min_perp, _EPS), 0.0, 1.0))
            reason = "active" if gate > 0 else "suppressed"
        effective[train_mask] = effective[train_mask] * gate
        diagnostics["by_time"][str(float(t))] = {
            "train_anchor_support": n_valid,
            "median_projected_pulse_fraction": median_perp,
            "mean_nuisance_energy_fraction": mean_nuisance,
            "side_information_gate": gate,
            "decision": reason,
        }
    # Heldout anchors are metric-only; do not modify their metric confidence.
    effective[split_code != 1] = confidence[split_code != 1]
    return effective.astype(np.float32, copy=False), diagnostics


def _pcib_residual_np(
    disp: np.ndarray,
    pulse: np.ndarray,
    confidence: np.ndarray,
    *,
    residual_scale: float,
    residual_endpoint_fraction: float,
    residual_max_norm: float,
    endpoint_min_pulse_cosine: float,
) -> np.ndarray:
    """Closed-form bounded residual used consistently in selector and sampler."""

    disp = np.asarray(disp, dtype=np.float32)
    pulse = np.asarray(pulse, dtype=np.float32)
    conf = np.asarray(confidence, dtype=np.float32).reshape(-1, 1)
    if disp.shape[0] == 0:
        return np.zeros_like(disp, dtype=np.float32)
    cosine = np.sum(disp * pulse, axis=1, keepdims=True) / np.maximum(
        _safe_norm(disp, axis=1, keepdims=True) * _safe_norm(pulse, axis=1, keepdims=True), _EPS
    )
    conf = conf * (cosine >= float(endpoint_min_pulse_cosine)).astype(np.float32)
    raw = float(residual_scale) * conf * pulse
    endpoint_norm = _safe_norm(disp, axis=1, keepdims=True)
    max_norm = np.minimum(float(residual_endpoint_fraction) * endpoint_norm, float(residual_max_norm))
    raw_norm = _safe_norm(raw, axis=1, keepdims=True)
    scale = np.minimum(max_norm / np.maximum(raw_norm, _EPS), 1.0)
    return np.nan_to_num(raw * scale, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def _pcib_residual_torch(
    disp: torch.Tensor,
    pulse: torch.Tensor,
    conf: torch.Tensor,
    *,
    residual_scale: float,
    residual_endpoint_fraction: float,
    residual_max_norm: float,
    endpoint_min_pulse_cosine: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Torch counterpart returning the bounded residual and gated confidence."""

    cosine = torch.sum(disp * pulse, dim=-1, keepdim=True) / torch.clamp(
        torch.linalg.norm(disp, dim=-1, keepdim=True) * torch.linalg.norm(pulse, dim=-1, keepdim=True), min=_EPS
    )
    gated_conf = conf * (cosine >= float(endpoint_min_pulse_cosine)).to(dtype=conf.dtype)
    raw = float(residual_scale) * gated_conf * pulse
    endpoint_norm = torch.linalg.norm(disp, dim=-1, keepdim=True)
    max_norm = torch.clamp(float(residual_endpoint_fraction) * endpoint_norm, max=float(residual_max_norm))
    raw_norm = torch.linalg.norm(raw, dim=-1, keepdim=True)
    scale = torch.clamp(max_norm / torch.clamp(raw_norm, min=_EPS), max=1.0)
    return raw * scale, gated_conf


def _project_perp_torch(x: torch.Tensor, basis: torch.Tensor | None) -> torch.Tensor:
    if basis is None or basis.numel() == 0:
        return x
    return x - torch.matmul(torch.matmul(x, basis), basis.T)


def _pcib_select_residual_torch(
    disp: torch.Tensor,
    pulse: torch.Tensor,
    conf: torch.Tensor,
    *,
    nuisance_basis: torch.Tensor | None,
    nuisance_budget: float,
    residual_scale: float,
    residual_endpoint_fraction: float,
    residual_max_norm: float,
    endpoint_min_pulse_cosine: float,
    selector_tau: float,
    pulse_cost_weight: float,
    nuisance_cost_weight: float,
    residual_cost_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select the endpoint-vanishing residual by minimizing the PCIB selector.

    This is a bounded-memory residual-path selector: for each candidate endpoint
    pair it minimizes the proposal terms over a deterministic finite constrained
    residual family `{0, 0.5 r_anchor, r_anchor}`.  The same selector is called
    during coupling-cost construction and after pair sampling, so the law over
    `(x,y,r)` is selected before CFM targets are generated rather than repaired
    after evaluation.
    """

    anchor_r, gated_conf = _pcib_residual_torch(
        disp,
        pulse,
        conf,
        residual_scale=residual_scale,
        residual_endpoint_fraction=residual_endpoint_fraction,
        residual_max_norm=residual_max_norm,
        endpoint_min_pulse_cosine=endpoint_min_pulse_cosine,
    )
    basis = nuisance_basis if nuisance_basis is not None and nuisance_basis.numel() > 0 else None
    best_cost = torch.full(disp.shape[:-1], float("inf"), dtype=disp.dtype, device=disp.device)
    best_residual = torch.zeros_like(disp)
    best_pulse_cost = torch.zeros_like(best_cost)
    best_nuisance_cost = torch.zeros_like(best_cost)
    best_residual_cost = torch.zeros_like(best_cost)
    side_weight = gated_conf.reshape(disp.shape[:-1])
    pulse_norm = torch.linalg.norm(pulse, dim=-1)
    tau = float(selector_tau)

    for scale in (0.0, 0.5, 1.0):
        r = anchor_r * float(scale)
        tangent = disp + (1.0 - 2.0 * tau) * r
        tangent_perp = _project_perp_torch(tangent, basis)
        tangent_perp_norm = torch.linalg.norm(tangent_perp, dim=-1)
        cosine = torch.sum(tangent_perp * pulse, dim=-1) / torch.clamp(tangent_perp_norm * pulse_norm, min=_EPS)
        pulse_cost = side_weight * (1.0 - cosine).clamp(min=0.0) * float(pulse_cost_weight)
        residual_cost = side_weight * torch.sum(r * r, dim=-1) * float(residual_cost_weight)
        if basis is not None:
            coeff = torch.matmul(tangent, basis)
            nuisance = torch.sum(coeff * coeff, dim=-1)
            denom = torch.clamp(torch.sum(tangent * tangent, dim=-1), min=_EPS)
            leakage_over_budget = torch.relu((nuisance / denom) - float(nuisance_budget))
            nuisance_cost = side_weight * leakage_over_budget.pow(2) * float(nuisance_cost_weight)
        else:
            nuisance_cost = torch.zeros_like(best_cost)
        selector_cost = residual_cost + pulse_cost + nuisance_cost
        take = selector_cost < best_cost
        best_cost = torch.where(take, selector_cost, best_cost)
        best_residual = torch.where(take.unsqueeze(-1), r, best_residual)
        best_pulse_cost = torch.where(take, pulse_cost, best_pulse_cost)
        best_nuisance_cost = torch.where(take, nuisance_cost, best_nuisance_cost)
        best_residual_cost = torch.where(take, residual_cost, best_residual_cost)

    return best_residual, gated_conf, best_pulse_cost, best_nuisance_cost, best_residual_cost


def build_pcib_payload(adata, config: dict[str, Any] | None = None) -> PCIBPayload:
    cfg = _pcib_config(config)
    _validate_pcib_data_contract(adata)
    if "time_point_processed" not in adata.obs or "X_latent" not in adata.obsm:
        raise ValueError("PCIB requires obs['time_point_processed'] and obsm['X_latent'].")
    split_code = _split_code_from_obs(adata)
    confidence = _confidence_from_obs(adata, cfg)
    pulse = _metabolic_pulse_from_adata(adata)
    latent_dim = int(np.asarray(adata.obsm["X_latent"]).shape[1])
    if pulse.shape[1] != latent_dim:
        raise ValueError(f"PCIB pulse dimension {pulse.shape[1]} does not match X_latent dimension {latent_dim}.")
    basis, singular, meta = _fit_nuisance_basis(adata, cfg, split_code)
    if bool(cfg.get("require_nonempty_nuisance_basis", True)) and basis.shape[1] == 0:
        raise ValueError(
            "PCIB requires a non-empty train-derived nuisance basis N. Empty N is only valid for explicit "
            "no-instrument controls, not for promotable PCIB campaign evidence."
        )
    pulse_perp = _project_perp(pulse, basis)
    effective_confidence, support_diagnostics = _support_gated_confidence(
        adata,
        cfg,
        split_code=split_code,
        confidence=confidence,
        pulse=pulse,
        pulse_perp=pulse_perp,
        nuisance_basis=basis,
    )
    train_count = int(np.sum((split_code == 1) & (confidence > 0)))
    heldout_count = int(np.sum((split_code == 2) & (confidence > 0)))
    return PCIBPayload(
        pulse=pulse,
        pulse_perp=pulse_perp.astype(np.float32, copy=False),
        confidence=confidence.astype(np.float32, copy=False),
        effective_confidence=effective_confidence.astype(np.float32, copy=False),
        split_code=split_code,
        nuisance_basis=basis.astype(np.float32, copy=False),
        singular_values=singular.astype(np.float32, copy=False),
        nuisance_budget=float(meta.get("nuisance_budget", 0.0)),
        support_diagnostics=support_diagnostics,
        train_count=train_count,
        heldout_count=heldout_count,
        control_count=int(meta.get("control_count", 0)),
        labeled0_count=int(meta.get("labeled0_train_count", 0)),
        source="raw-pool-prepared Well-TEMP HCT116 PCIB training_data_builder",
        notes=str(meta.get("notes", "")),
    )


def pcib_training_data_builder(context: TrainingDataBuilderContext) -> TrainingDataBundle:
    """Augment canonical CytoBridge training data with train-only PCIB sidecars."""

    base = build_default_training_data(
        context.adata,
        context.resolved_config,
        stage=context.stage,
        device=context.device,
        output_dir=context.output_dir,
        metadata=context.metadata,
    )
    cfg = _pcib_config(context.resolved_config)
    payload = build_pcib_payload(base.adata, cfg)

    pulse_perp_by_time: list[np.ndarray] = []
    pulse_by_time: list[np.ndarray] = []
    confidence_by_time: list[np.ndarray] = []
    split_by_time: list[np.ndarray] = []
    train_pulse_perp_by_time: list[np.ndarray] = []
    train_confidence_by_time: list[np.ndarray] = []
    for rows in base.obs_indices_by_time:
        idx = np.asarray(rows, dtype=np.int64)
        pulse_by_time.append(payload.pulse[idx].astype(np.float32, copy=False))
        pulse_perp_by_time.append(payload.pulse_perp[idx].astype(np.float32, copy=False))
        confidence_by_time.append(payload.confidence[idx].astype(np.float32, copy=False))
        split_by_time.append(payload.split_code[idx].astype(np.int8, copy=False))
        train_mask = (payload.split_code[idx] == 1) & (payload.effective_confidence[idx] > 0)
        train_pulse = np.zeros_like(payload.pulse_perp[idx], dtype=np.float32)
        train_conf = np.zeros(idx.shape[0], dtype=np.float32)
        train_pulse[train_mask] = payload.pulse_perp[idx][train_mask]
        train_conf[train_mask] = payload.effective_confidence[idx][train_mask]
        train_pulse_perp_by_time.append(train_pulse)
        train_confidence_by_time.append(train_conf)

    extra = dict(base.extra_modalities_by_time or {})
    extra.update(
        {
            "pcib_pulse": pulse_by_time,
            "pcib_pulse_perp": pulse_perp_by_time,
            "pcib_confidence": confidence_by_time,
            "pcib_split_code": split_by_time,
            "pcib_train_pulse_perp": train_pulse_perp_by_time,
            "pcib_train_confidence": train_confidence_by_time,
        }
    )
    meta = dict(base.metadata or {})
    meta["pcib"] = {
        "nuisance_basis": payload.nuisance_basis.astype(np.float32, copy=False),
        "nuisance_rank": int(payload.nuisance_basis.shape[1]),
        "nuisance_budget": float(payload.nuisance_budget),
        "singular_values": payload.singular_values.astype(np.float32, copy=False),
        "support_diagnostics": payload.support_diagnostics,
        "train_anchor_count": int(payload.train_count),
        "heldout_anchor_count": int(payload.heldout_count),
        "control_count": int(payload.control_count),
        "labeled0_train_count": int(payload.labeled0_count),
        "source": payload.source,
        "notes": payload.notes,
        "leakage_control": "N/coupling/path/loss use train anchors only; PCIE uses split_code==2 heldout anchors from sealed rollout only",
        "claim_metric_evaluator": EVALUATOR_ID,
        "config": dict(cfg),
    }
    return validate_training_data_bundle(
        TrainingDataBundle(
            adata=base.adata,
            time_points=base.time_points,
            latent_by_time=base.latent_by_time,
            obs_indices_by_time=base.obs_indices_by_time,
            extra_modalities_by_time=extra,
            metadata=meta,
        )
    )


def _state_from_training_data(training_data: TrainingDataBundle, config: dict[str, Any] | None = None) -> PCIBPayload:
    meta = dict((training_data.metadata or {}).get("pcib") or {})
    if "nuisance_basis" in meta:
        basis = np.asarray(meta.get("nuisance_basis"), dtype=np.float32)
        payload = build_pcib_payload(training_data.adata, _pcib_config(meta.get("config") or config))
        # Preserve the frozen training basis exactly; recompute row-aligned pulse
        # projections through that basis for candidate or baseline evaluators.
        payload.nuisance_basis = basis
        payload.pulse_perp = _project_perp(payload.pulse, basis)
        payload.nuisance_budget = float(meta.get("nuisance_budget", payload.nuisance_budget))
        return payload
    return build_pcib_payload(training_data.adata, config)


# -----------------------------------------------------------------------------
# Coupling and endpoint-valid residual path
# -----------------------------------------------------------------------------


class PCIBBalancedPulseCoupling(ChunkedTransportCouplingStrategy):
    """Balanced OT block coupling with PCIB pulse/control cost terms."""

    def __init__(
        self,
        *,
        train_pulse_perp_by_time: list[np.ndarray],
        train_confidence_by_time: list[np.ndarray],
        nuisance_basis: np.ndarray,
        nuisance_budget: float,
        endpoint_cost_weight: float = 1.0,
        pulse_cost_weight: float = 0.35,
        nuisance_cost_weight: float = 0.15,
        residual_cost_weight: float = 0.02,
        residual_scale: float = 0.20,
        residual_max_norm: float = 5.0,
        residual_endpoint_fraction: float = 0.50,
        selector_tau: float = 0.25,
        endpoint_min_pulse_cosine: float = -0.25,
        method: str = "sinkhorn",
        reg: float = 0.05,
        use_mini_batch_balanced: bool = True,
        chunk_size: int = 1000,
        normalize_cost: bool = False,
        auto_normalize_large_cost: bool = True,
    ) -> None:
        super().__init__(
            solver_mode="balanced",
            chunk_size=int(chunk_size),
            use_mini_batch=bool(use_mini_batch_balanced),
            balanced_method=str(method),
            balanced_reg=float(reg),
            normalize_cost=bool(normalize_cost),
            auto_normalize_large_cost=bool(auto_normalize_large_cost),
        )
        self.train_pulse_perp_by_time = [np.asarray(x, dtype=np.float32) for x in train_pulse_perp_by_time]
        self.train_confidence_by_time = [np.asarray(x, dtype=np.float32).reshape(-1) for x in train_confidence_by_time]
        self.nuisance_basis = np.asarray(nuisance_basis, dtype=np.float32)
        self.nuisance_budget = float(nuisance_budget)
        self.endpoint_cost_weight = float(endpoint_cost_weight)
        self.pulse_cost_weight = float(pulse_cost_weight)
        self.nuisance_cost_weight = float(nuisance_cost_weight)
        self.residual_cost_weight = float(residual_cost_weight)
        self.residual_scale = float(residual_scale)
        self.residual_max_norm = float(residual_max_norm)
        self.residual_endpoint_fraction = float(residual_endpoint_fraction)
        self.selector_tau = float(np.clip(selector_tau, 0.0, 1.0))
        self.endpoint_min_pulse_cosine = float(endpoint_min_pulse_cosine)
        self.method = str(method)
        self.reg = float(reg)

    def describe_metadata(self) -> dict[str, Any]:
        meta = super().describe_metadata()
        meta.update(
            {
                "kind": "pcib_balanced_pulse_control_coupling",
                "method": self.method,
                "reg": self.reg,
                "endpoint_cost_weight": self.endpoint_cost_weight,
                "pulse_cost_weight": self.pulse_cost_weight,
                "nuisance_cost_weight": self.nuisance_cost_weight,
                "residual_cost_weight": self.residual_cost_weight,
                "residual_scale": self.residual_scale,
                "residual_endpoint_fraction": self.residual_endpoint_fraction,
                "residual_max_norm": self.residual_max_norm,
                "selector_tau": self.selector_tau,
                "nuisance_budget": self.nuisance_budget,
                "endpoint_min_pulse_cosine": self.endpoint_min_pulse_cosine,
                "nuisance_rank": int(self.nuisance_basis.shape[1]) if self.nuisance_basis.ndim == 2 else 0,
                "balanced_only": True,
            }
        )
        return meta

    def build_pairwise_cost_block(
        self,
        x0_block: np.ndarray,
        x1_block: np.ndarray,
        *,
        source_indices: np.ndarray,
        target_indices: np.ndarray,
        time_idx: int,
        t0: float,
        t1: float,
        device: torch.device,
    ) -> PairwiseCost:
        del target_indices
        x0_t = torch.as_tensor(x0_block, dtype=torch.float32, device=device)
        x1_t = torch.as_tensor(x1_block, dtype=torch.float32, device=device)
        disp = x1_t.unsqueeze(0) - x0_t.unsqueeze(1)
        endpoint = torch.sum(disp * disp, dim=-1) * self.endpoint_cost_weight

        if time_idx < len(self.train_pulse_perp_by_time):
            pulse_np = self.train_pulse_perp_by_time[time_idx][np.asarray(source_indices, dtype=np.int64)]
            conf_np = self.train_confidence_by_time[time_idx][np.asarray(source_indices, dtype=np.int64)]
        else:
            pulse_np = np.zeros((x0_block.shape[0], x0_block.shape[1]), dtype=np.float32)
            conf_np = np.zeros(x0_block.shape[0], dtype=np.float32)
        pulse = torch.as_tensor(pulse_np, dtype=torch.float32, device=device)
        conf = torch.as_tensor(conf_np, dtype=torch.float32, device=device).reshape(-1, 1)
        disp_flat = disp.reshape(-1, disp.shape[-1])
        pulse_flat = pulse.unsqueeze(1).expand(-1, x1_block.shape[0], -1).reshape(-1, pulse.shape[-1])
        conf_flat = conf.unsqueeze(1).expand(-1, x1_block.shape[0], -1).reshape(-1, 1)
        n_t = torch.as_tensor(self.nuisance_basis, dtype=torch.float32, device=device) if (
            self.nuisance_basis.ndim == 2 and self.nuisance_basis.shape[1] > 0
        ) else None
        residual_flat, gated_conf_flat, pulse_cost_flat, nuisance_cost_flat, residual_cost_flat = _pcib_select_residual_torch(
            disp_flat,
            pulse_flat,
            conf_flat,
            nuisance_basis=n_t,
            nuisance_budget=self.nuisance_budget,
            residual_scale=self.residual_scale,
            residual_endpoint_fraction=self.residual_endpoint_fraction,
            residual_max_norm=self.residual_max_norm,
            endpoint_min_pulse_cosine=self.endpoint_min_pulse_cosine,
            selector_tau=self.selector_tau,
            pulse_cost_weight=self.pulse_cost_weight,
            nuisance_cost_weight=self.nuisance_cost_weight,
            residual_cost_weight=self.residual_cost_weight,
        )
        residual = residual_flat.reshape_as(disp)
        side_weight = gated_conf_flat.reshape(x0_block.shape[0], x1_block.shape[0])
        residual_cost = residual_cost_flat.reshape(x0_block.shape[0], x1_block.shape[0])
        pulse_cost = pulse_cost_flat.reshape(x0_block.shape[0], x1_block.shape[0])
        nuisance_cost = nuisance_cost_flat.reshape(x0_block.shape[0], x1_block.shape[0])

        cost = endpoint + residual_cost + pulse_cost + nuisance_cost
        cost = torch.nan_to_num(cost, nan=0.0, posinf=1.0e6, neginf=0.0).to(dtype=torch.float32)
        return PairwiseCost(
            cost_matrix=cost,
            source_mass=np.ones(x0_block.shape[0], dtype=np.float32),
            target_mass=np.ones(x1_block.shape[0], dtype=np.float32),
            metadata={
                "time_idx": int(time_idx),
                "t0": float(t0),
                "t1": float(t1),
                "pcib_cost": True,
                "train_anchor_sources_in_block": int(np.sum(conf_np > 0)),
                "nuisance_budget": float(self.nuisance_budget),
                "endpoint_min_pulse_cosine": float(self.endpoint_min_pulse_cosine),
                "selector_tau": float(self.selector_tau),
                "residual_cost_weight": float(self.residual_cost_weight),
            },
        )


class PCIBEndpointResidualPath(LinearDeterministicConditionalPath):
    """Endpoint-valid path psi_s=(1-s)x0+s x1+s(1-s)r with analytic derivative."""

    def __init__(self, sigma: float = 0.0) -> None:
        if abs(float(sigma)) > 1e-12:
            raise ValueError("PCIB is deterministic; sigma must be 0.0.")
        super().__init__(sigma=0.0)
        self._residual: torch.Tensor | None = None

    def set_pair_residual(self, residual: torch.Tensor | None) -> None:
        self._residual = residual

    def _current_residual(self, x0: torch.Tensor) -> torch.Tensor:
        if self._residual is None:
            return torch.zeros_like(x0)
        return self._residual.to(device=x0.device, dtype=x0.dtype)

    def compute_mu_t(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        tt = torch.as_tensor(t, device=x0.device, dtype=x0.dtype).reshape(-1, *([1] * (x0.dim() - 1)))
        r = self._current_residual(x0)
        return (1.0 - tt) * x0 + tt * x1 + tt * (1.0 - tt) * r

    def compute_conditional_flow(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor, xt: torch.Tensor) -> torch.Tensor:
        del xt
        tt = torch.as_tensor(t, device=x0.device, dtype=x0.dtype).reshape(-1, *([1] * (x0.dim() - 1)))
        r = self._current_residual(x0)
        return (x1 - x0) + (1.0 - 2.0 * tt) * r


class PCIBFlowMatchingBackend(FlowMatchingBackend):
    """FlowMatchingBackend variant that injects endpoint-vanishing residuals."""

    path: PCIBEndpointResidualPath

    def __init__(
        self,
        *,
        path: PCIBEndpointResidualPath,
        coupling: PCIBBalancedPulseCoupling,
        train_pulse_perp_by_time: list[np.ndarray],
        train_confidence_by_time: list[np.ndarray],
        nuisance_basis: np.ndarray,
        nuisance_budget: float,
        residual_scale: float,
        residual_max_norm: float,
        residual_endpoint_fraction: float,
        selector_tau: float,
        pulse_cost_weight: float,
        nuisance_cost_weight: float,
        residual_cost_weight: float,
        endpoint_min_pulse_cosine: float,
    ) -> None:
        super().__init__(path=path, coupling=coupling, mass=NullMassStrategy())
        self.train_pulse_perp_by_time = [np.asarray(x, dtype=np.float32) for x in train_pulse_perp_by_time]
        self.train_confidence_by_time = [np.asarray(x, dtype=np.float32).reshape(-1) for x in train_confidence_by_time]
        self.nuisance_basis = np.asarray(nuisance_basis, dtype=np.float32)
        self.nuisance_budget = float(nuisance_budget)
        self.residual_scale = float(residual_scale)
        self.residual_max_norm = float(residual_max_norm)
        self.residual_endpoint_fraction = float(residual_endpoint_fraction)
        self.selector_tau = float(np.clip(selector_tau, 0.0, 1.0))
        self.pulse_cost_weight = float(pulse_cost_weight)
        self.nuisance_cost_weight = float(nuisance_cost_weight)
        self.residual_cost_weight = float(residual_cost_weight)
        self.endpoint_min_pulse_cosine = float(endpoint_min_pulse_cosine)

    def _residual_for_pair_batch(self, pair_batch, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n = pair_batch.x0.shape[0]
        latent_dim = pair_batch.x0.shape[1]
        if pair_batch.time_idx >= len(self.train_pulse_perp_by_time):
            zeros = torch.zeros(n, latent_dim, dtype=pair_batch.x0.dtype, device=device)
            conf = torch.zeros(n, 1, dtype=pair_batch.x0.dtype, device=device)
            return zeros, zeros, conf
        idx = np.asarray(pair_batch.idx0, dtype=np.int64)
        pulse_np = self.train_pulse_perp_by_time[pair_batch.time_idx][idx]
        conf_np = self.train_confidence_by_time[pair_batch.time_idx][idx]
        pulse = torch.as_tensor(pulse_np, dtype=pair_batch.x0.dtype, device=device)
        conf = torch.as_tensor(conf_np, dtype=pair_batch.x0.dtype, device=device).reshape(-1, 1)
        disp = pair_batch.x1 - pair_batch.x0
        basis = torch.as_tensor(self.nuisance_basis, dtype=pair_batch.x0.dtype, device=device) if (
            self.nuisance_basis.ndim == 2 and self.nuisance_basis.shape[1] > 0
        ) else None
        residual, conf, _, _, _ = _pcib_select_residual_torch(
            disp,
            pulse,
            conf,
            nuisance_basis=basis,
            nuisance_budget=self.nuisance_budget,
            residual_scale=self.residual_scale,
            residual_endpoint_fraction=self.residual_endpoint_fraction,
            residual_max_norm=self.residual_max_norm,
            endpoint_min_pulse_cosine=self.endpoint_min_pulse_cosine,
            selector_tau=self.selector_tau,
            pulse_cost_weight=self.pulse_cost_weight,
            nuisance_cost_weight=self.nuisance_cost_weight,
            residual_cost_weight=self.residual_cost_weight,
        )
        return residual, pulse, conf

    def sample_batch(self, X: list[np.ndarray], t_train: torch.Tensor, batch_size: int, device: torch.device) -> FlowMatchingBatch:
        if self.state is None:
            raise RuntimeError("PCIBFlowMatchingBackend.prepare(...) must be called before sample_batch(...).")
        local_times: list[torch.Tensor] = []
        global_times: list[torch.Tensor] = []
        xts: list[torch.Tensor] = []
        uts: list[torch.Tensor] = []
        gts: list[torch.Tensor] = []
        loss_weights_list: list[torch.Tensor] = []
        noises: list[torch.Tensor] = []
        residuals: list[torch.Tensor] = []
        pulses: list[torch.Tensor] = []
        confidences: list[torch.Tensor] = []

        for time_idx in range(len(t_train) - 1):
            pair_batch = self.coupling.sample_pairs(self.state, X, time_idx, batch_size, device)
            if pair_batch is None:
                continue
            residual, pulse, conf = self._residual_for_pair_batch(pair_batch, device=device)
            self.path.set_pair_residual(residual)
            t_local = self.path.sample_time(pair_batch.x0, pair_batch.x1).to(device=device, dtype=pair_batch.x0.dtype)
            eps = self.path.sample_noise_like(pair_batch.x0)
            xt = self.path.sample_xt(pair_batch.x0, pair_batch.x1, t_local, eps)
            ut = self.path.compute_conditional_flow(pair_batch.x0, pair_batch.x1, t_local, xt)
            mass_result = self.mass.compute(self.state, pair_batch, t_local, device)
            gt, loss_weights = self.path.compute_conditional_mass(mass_result.terminal_mass, t_local)
            time_interval = t_train[time_idx + 1] - t_train[time_idx]
            local_times.append(t_local)
            global_times.append(t_local * time_interval + t_train[time_idx])
            xts.append(xt)
            uts.append(ut / time_interval)
            gts.append(gt / time_interval)
            loss_weights_list.append(loss_weights)
            noises.append(eps)
            residuals.append(residual / time_interval)
            pulses.append(pulse)
            confidences.append(conf)

        if not global_times:
            empty = torch.empty(0, device=device)
            empty_latent = torch.empty(0, *X[0].shape[1:], device=device)
            empty_col = torch.empty(0, 1, device=device)
            return FlowMatchingBatch(
                t_local=empty,
                t_global=empty,
                xt=empty_latent,
                ut=empty_latent,
                gt=empty_col,
                loss_weights=empty_col,
                eps=empty_latent,
                extras={},
            )

        return FlowMatchingBatch(
            t_local=torch.cat(local_times),
            t_global=torch.cat(global_times),
            xt=torch.cat(xts),
            ut=torch.cat(uts),
            gt=torch.cat(gts),
            loss_weights=torch.cat(loss_weights_list),
            eps=torch.cat(noises),
            extras={
                "pcib_residual": torch.cat(residuals),
                "pcib_pulse_perp": torch.cat(pulses),
                "pcib_confidence": torch.cat(confidences),
                "pcib_nuisance_basis": torch.as_tensor(self.nuisance_basis, dtype=torch.float32, device=device),
                "pcib_nuisance_budget": torch.as_tensor(float(self.nuisance_budget), dtype=torch.float32, device=device),
            },
        )


# -----------------------------------------------------------------------------
# Loss hook and rollout-derived claim metric
# -----------------------------------------------------------------------------


def pcib_flow_matching_loss_hook(context: FlowMatchingLossContext) -> FlowMatchingLossResult | None:
    cfg = _pcib_config(context.stage_params)
    extras = getattr(context.batch, "extras", {}) or {}
    pulse = extras.get("pcib_pulse_perp")
    conf = extras.get("pcib_confidence")
    if pulse is None or conf is None or not context.regress_v:
        return None
    pulse = pulse.to(device=context.net_input.device, dtype=context.net_input.dtype)
    conf = conf.to(device=context.net_input.device, dtype=context.net_input.dtype)
    active = (conf.reshape(-1) > 0) & (torch.linalg.norm(pulse, dim=1) > _EPS)
    if not torch.any(active):
        return None
    v_pred = context.model.velocity_net(context.net_input)
    n_basis = extras.get("pcib_nuisance_basis")
    nuisance_budget = extras.get("pcib_nuisance_budget")
    budget = 0.0 if nuisance_budget is None else float(torch.as_tensor(nuisance_budget).detach().cpu().item())
    if n_basis is not None and n_basis.numel() > 0:
        n_basis = n_basis.to(device=v_pred.device, dtype=v_pred.dtype)
        v_perp = v_pred - (v_pred @ n_basis) @ n_basis.T
        nuisance_energy = torch.sum((v_pred @ n_basis) ** 2, dim=1, keepdim=True)
    else:
        v_perp = v_pred
        nuisance_energy = torch.zeros(v_pred.shape[0], 1, dtype=v_pred.dtype, device=v_pred.device)
    v_norm = torch.linalg.norm(v_perp, dim=1)
    p_norm = torch.linalg.norm(pulse, dim=1)
    cos = torch.sum(v_perp * pulse, dim=1) / torch.clamp(v_norm * p_norm, min=_EPS)
    align_loss = ((1.0 - cos[active]).clamp(min=0.0) * conf.reshape(-1)[active]).sum() / torch.clamp(
        conf.reshape(-1)[active].sum(), min=_EPS
    )
    leakage = nuisance_energy / torch.clamp(torch.sum(v_pred * v_pred, dim=1, keepdim=True), min=_EPS)
    leakage_over_budget = torch.relu(leakage - budget)
    leak_loss = (leakage_over_budget.reshape(-1)[active] * conf.reshape(-1)[active]).sum() / torch.clamp(
        conf.reshape(-1)[active].sum(), min=_EPS
    )
    extra = float(cfg.get("velocity_loss_weight", 0.02)) * align_loss + float(
        cfg.get("velocity_leakage_weight", 0.01)
    ) * leak_loss
    return FlowMatchingLossResult(
        extra_loss=extra,
        logs={
            "pcib_velocity_align_loss": align_loss.detach(),
            "pcib_velocity_leak_loss": leak_loss.detach(),
            "pcib_active_anchor_batch": float(active.detach().sum().cpu().item()),
        },
    )


def evaluate_claim_metric(context: EvaluationMetricsContext) -> dict[str, Any]:
    """Rollout-derived heldout pulse/control identifiability error (PCIE)."""

    cfg = _pcib_config(context.config)
    metric_name = str(cfg.get("claim_metric_name") or "pcib_pulse_control_identifiability_error")
    unsupported = float(cfg.get("unsupported_metric_value", 1.0e6))
    if context.evaluation_trajectory is None or not context.trajectory_points_by_time:
        return {
            metric_name: unsupported,
            "PCIE": unsupported,
            "pcib_stage2_metric_supported": 0.0,
            "pcib_rollout_derived_metric": 0.0,
            "pcib_heldout_anchor_count": 0.0,
            "pcib_evaluator_version": 1.0,
        }

    payload = _state_from_training_data(context.training_data, cfg)
    basis = np.asarray(payload.nuisance_basis, dtype=np.float32)
    nuisance_budget = float(payload.nuisance_budget)
    max_anchors = int(cfg.get("max_metric_anchors_per_time", 2048))
    max_particles = int(cfg.get("max_metric_particles_per_time", 4096))
    nearest_chunk = int(cfg.get("nearest_chunk", 512))
    rng = np.random.default_rng(int(cfg.get("metric_subsample_seed", 1729)))
    observed_indices = list(context.observed_time_indices or [])
    cos_pairs: list[np.ndarray] = []
    leak_pairs: list[np.ndarray] = []
    support_by_time: dict[str, int] = {}

    for time_i, obs_traj_idx in enumerate(observed_indices):
        if time_i >= len(context.training_data.obs_indices_by_time):
            continue
        if int(obs_traj_idx) + 1 >= len(context.trajectory_points_by_time):
            continue
        rows = np.asarray(context.training_data.obs_indices_by_time[time_i], dtype=np.int64)
        heldout = rows[(payload.split_code[rows] == 2) & (payload.confidence[rows] > 0)]
        if heldout.size == 0:
            continue
        if heldout.size > max_anchors:
            heldout = rng.choice(heldout, size=max_anchors, replace=False)
        anchor_x = _as_numpy(context.adata.obsm["X_latent"][heldout], dtype=np.float32)
        anchor_pulse = payload.pulse_perp[heldout].astype(np.float32, copy=False)
        anchor_conf = payload.confidence[heldout].astype(np.float32, copy=False)
        valid_anchor = (_safe_norm(anchor_pulse, axis=1) > _EPS) & (anchor_conf > 0)
        if not np.any(valid_anchor):
            continue
        anchor_x = anchor_x[valid_anchor]
        anchor_pulse = anchor_pulse[valid_anchor]
        anchor_conf = anchor_conf[valid_anchor]

        cur = _as_numpy(context.trajectory_points_by_time[int(obs_traj_idx)], dtype=np.float32)
        nxt = _as_numpy(context.trajectory_points_by_time[int(obs_traj_idx) + 1], dtype=np.float32)
        if cur.ndim != 2 or nxt.ndim != 2 or cur.shape[0] == 0 or nxt.shape[0] == 0:
            continue
        n = min(cur.shape[0], nxt.shape[0])
        cur = cur[:n]
        nxt = nxt[:n]
        if n > max_particles:
            particle_idx = rng.choice(np.arange(n), size=max_particles, replace=False)
            cur_ref = cur[particle_idx]
            nxt_ref = nxt[particle_idx]
        else:
            cur_ref = cur
            nxt_ref = nxt
        t_cur = float(context.trajectory_time_points[int(obs_traj_idx)])
        t_next = float(context.trajectory_time_points[int(obs_traj_idx) + 1])
        dt = max(abs(t_next - t_cur), _EPS)
        pred_v_ref = (nxt_ref - cur_ref) / dt
        nn = _nearest_indices(anchor_x, cur_ref, chunk=nearest_chunk, device=context.device)
        pred_v = pred_v_ref[nn]
        pred_perp = _project_perp(pred_v, basis)
        pred_n_norm_sq = np.sum((pred_v @ basis) ** 2, axis=1) if basis.shape[1] > 0 else np.zeros(pred_v.shape[0])
        pred_norm_sq = np.maximum(np.sum(pred_v * pred_v, axis=1), _EPS)
        denom = np.maximum(_safe_norm(pred_perp, axis=1) * _safe_norm(anchor_pulse, axis=1), _EPS)
        cosine = np.sum(pred_perp * anchor_pulse, axis=1) / denom
        cos_error = np.clip(1.0 - cosine, 0.0, 2.0)
        leakage = pred_n_norm_sq / pred_norm_sq
        leakage_over_budget = np.maximum(leakage - nuisance_budget, 0.0)
        finite = np.isfinite(cos_error) & np.isfinite(leakage) & np.isfinite(anchor_conf) & (anchor_conf > 0)
        if not np.any(finite):
            continue
        cos_pairs.append(np.stack([cos_error[finite], anchor_conf[finite]], axis=1))
        leak_pairs.append(np.stack([leakage_over_budget[finite], anchor_conf[finite]], axis=1))
        support_by_time[str(float(context.time_points[time_i]))] = int(np.sum(finite))

    if not cos_pairs:
        return {
            metric_name: unsupported,
            "PCIE": unsupported,
            "pcib_stage2_metric_supported": 0.0,
            "pcib_rollout_derived_metric": 1.0,
            "pcib_heldout_anchor_count": 0.0,
            "pcib_train_anchor_count": float(payload.train_count),
            "pcib_nuisance_rank": float(basis.shape[1]),
            "pcib_evaluator_version": 1.0,
        }

    cos_all = np.concatenate(cos_pairs, axis=0)
    leak_all = np.concatenate(leak_pairs, axis=0)

    def _weighted_mean(pairs: np.ndarray) -> float:
        vals = pairs[:, 0]
        weights = np.maximum(pairs[:, 1], 0.0)
        if float(np.sum(weights)) <= 0:
            return float(np.mean(vals))
        return float(np.sum(vals * weights) / np.sum(weights))

    cos_error = _weighted_mean(cos_all)
    leakage = _weighted_mean(leak_all)
    pcie = float(cos_error + float(cfg.get("metric_leakage_weight", 0.25)) * leakage)
    return {
        metric_name: pcie,
        "PCIE": pcie,
        "pcib_pcie_cosine_error": float(cos_error),
        "pcib_pcie_nuisance_leakage_above_budget": float(leakage),
        "pcib_nuisance_budget": float(nuisance_budget),
        "pcib_stage2_metric_supported": 1.0,
        "pcib_rollout_derived_metric": 1.0,
        "pcib_no_target_repair_metric": 1.0,
        "pcib_heldout_anchor_count": float(cos_all.shape[0]),
        "pcib_train_anchor_count": float(payload.train_count),
        "pcib_control_count": float(payload.control_count),
        "pcib_labeled0_train_count": float(payload.labeled0_count),
        "pcib_nuisance_rank": float(basis.shape[1]),
        "pcib_time_count_total": float(sum(support_by_time.values())),
        "pcib_evaluator_version": 1.0,
    }


# -----------------------------------------------------------------------------
# TrainingAlgorithmSpec entrypoint
# -----------------------------------------------------------------------------


def build_training_algorithm(context):
    """Return the approved PCIB custom TrainingAlgorithmSpec."""

    def model_builder(model_context: ModelBuildContext) -> torch.nn.Module:
        model = DynamicalModel(model_context.latent_dim, model_context.resolved_config["model"])
        return model

    def build_flow_matching_backend(build_context: FlowMatchingBuildContext) -> FlowMatchingBackend:
        stage_params = build_context.stage_params
        fm = (stage_params.get("flow_matching") or {}) if isinstance(stage_params, dict) else {}
        coupling_config = (fm.get("coupling") or {}) if isinstance(fm, dict) else {}
        path_config = (fm.get("path") or {}) if isinstance(fm, dict) else {}
        meta = dict((build_context.training_data.metadata or {}).get("pcib") or {})
        cfg = _pcib_config({**dict(meta.get("config") or {}), **dict(stage_params.get("pcib") or {})})
        extra = build_context.training_data.extra_modalities_by_time
        basis = np.asarray(meta.get("nuisance_basis"), dtype=np.float32)
        latent_dim = int(build_context.training_data.latent_by_time[0].shape[1])
        if basis.ndim != 2 or basis.shape[0] != latent_dim:
            raise ValueError("PCIB training_data_builder must provide metadata['pcib']['nuisance_basis'] with latent_dim rows.")
        sigma = float(path_config.get("sigma", stage_params.get("sigma", 0.0)))
        if abs(sigma) > 1e-12:
            raise ValueError("PCIB approved core is deterministic; flow_matching.path.sigma must be 0.0.")
        coupling = PCIBBalancedPulseCoupling(
            train_pulse_perp_by_time=[np.asarray(x, dtype=np.float32) for x in extra.get("pcib_train_pulse_perp", [])],
            train_confidence_by_time=[np.asarray(x, dtype=np.float32) for x in extra.get("pcib_train_confidence", [])],
            nuisance_basis=basis,
            nuisance_budget=float(meta.get("nuisance_budget", 0.0)),
            endpoint_cost_weight=float(cfg.get("endpoint_cost_weight", 1.0)),
            pulse_cost_weight=float(cfg.get("pulse_cost_weight", 0.35)),
            nuisance_cost_weight=float(cfg.get("nuisance_cost_weight", 0.15)),
            residual_cost_weight=float(cfg.get("residual_cost_weight", 0.02)),
            residual_scale=float(cfg.get("residual_scale", 0.20)),
            residual_max_norm=float(cfg.get("residual_max_norm", 5.0)),
            residual_endpoint_fraction=float(cfg.get("residual_endpoint_fraction", 0.50)),
            selector_tau=float(cfg.get("selector_tau", 0.25)),
            endpoint_min_pulse_cosine=float(cfg.get("endpoint_min_pulse_cosine", -0.25)),
            method=str(coupling_config.get("method", "sinkhorn")),
            reg=float(coupling_config.get("reg", 0.05)),
            use_mini_batch_balanced=bool(coupling_config.get("use_mini_batch_balanced", True)),
            chunk_size=int(coupling_config.get("chunk_size", 1000)),
            normalize_cost=bool(coupling_config.get("normalize_cost", False)),
            auto_normalize_large_cost=bool(coupling_config.get("auto_normalize_large_cost", True)),
        )
        path = PCIBEndpointResidualPath(sigma=0.0)
        return PCIBFlowMatchingBackend(
            path=path,
            coupling=coupling,
            train_pulse_perp_by_time=[np.asarray(x, dtype=np.float32) for x in extra.get("pcib_train_pulse_perp", [])],
            train_confidence_by_time=[np.asarray(x, dtype=np.float32) for x in extra.get("pcib_train_confidence", [])],
            nuisance_basis=basis,
            nuisance_budget=float(meta.get("nuisance_budget", 0.0)),
            residual_scale=float(cfg.get("residual_scale", 0.20)),
            residual_max_norm=float(cfg.get("residual_max_norm", 5.0)),
            residual_endpoint_fraction=float(cfg.get("residual_endpoint_fraction", 0.50)),
            selector_tau=float(cfg.get("selector_tau", 0.25)),
            pulse_cost_weight=float(cfg.get("pulse_cost_weight", 0.35)),
            nuisance_cost_weight=float(cfg.get("nuisance_cost_weight", 0.15)),
            residual_cost_weight=float(cfg.get("residual_cost_weight", 0.02)),
            endpoint_min_pulse_cosine=float(cfg.get("endpoint_min_pulse_cosine", -0.25)),
        )

    def evaluation_metrics_hook(eval_context: EvaluationMetricsContext) -> dict[str, Any]:
        return evaluate_claim_metric(eval_context)

    return TrainingAlgorithmSpec(
        algorithm_id=context.algorithm_id,
        base_config="./config.yaml",
        config_overrides={},
        training_data_builder=pcib_training_data_builder,
        flow_matching_backend_builder=build_flow_matching_backend,
        flow_matching_loss_hook=pcib_flow_matching_loss_hook,
        model_builder=model_builder,
        serialize_model=None,
        deserialize_model=None,
        stage_runner=None,
        inference_context_builder=None,
        simulation_hook=None,
        evaluation_metrics_hook=evaluation_metrics_hook,
        evaluation_metrics_params={
            "claim_metric_name": "pcib_pulse_control_identifiability_error",
            "claim_metric_direction": "lower",
            "claim_metric_evaluator": EVALUATOR_ID,
            "expected_ranges": {
                "pcib_pulse_control_identifiability_error": {"min": 0.0, "max": 2.5},
                "PCIE": {"min": 0.0, "max": 2.5},
                "pcib_pcie_cosine_error": {"min": 0.0, "max": 2.0},
                "pcib_pcie_nuisance_leakage_above_budget": {"min": 0.0, "max": 1.0},
            },
        },
        notes=(
            "PCIB balanced-only endpoint-valid flow matching: train-only HCT116_control vs HCT116_0d "
            "nuisance basis, deconfounded metabolic pulse coupling/path residuals, NullMassStrategy, "
            "and rollout-derived heldout PCIE metric."
        ),
    )
