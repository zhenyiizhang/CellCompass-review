from __future__ import annotations

from typing import Any

import numpy as np
import torch

from CytoBridge.tl.flow_matching_backends import (
    ChunkedTransportCouplingStrategy,
    CouplingState,
    FlowMatchingBackend,
    FlowMatchingBatch,
    LinearDeterministicConditionalPath,
    PairBatch,
    PairwiseCost,
    RegularizedUnbalancedConditionalPath,
    UOTMassStrategy,
)
from CytoBridge.tl.models import DynamicalModel, HyperNetwork
from CytoBridge.tl.training_algorithm import (
    EvaluationMetricsContext,
    FlowMatchingBuildContext,
    FlowMatchingLossContext,
    FlowMatchingLossResult,
    InferenceContext,
    InferenceContextBuilderContext,
    ModelBuildContext,
    SimulationContext,
    SimulationResult,
    TrainingAlgorithmSpec,
    TrainingDataBuilderContext,
    TrainingDataBundle,
)


# ---------------------------------------------------------------------------
# Shared clone / fate utilities
# ---------------------------------------------------------------------------

_MISSING_CLONE_STRINGS = {"", "nan", "none", "null", "na", "<na>"}


def _clone_key(value: Any) -> str:
    text = str(value).strip()
    if text.lower() in _MISSING_CLONE_STRINGS:
        return ""
    try:
        f = float(text)
        if np.isfinite(f) and abs(f - int(f)) < 1e-9:
            return str(int(f))
    except Exception:
        pass
    return text


def _extract_clone_arrays(adata: Any, obs_indices_by_time: list[Any]) -> list[np.ndarray]:
    if "clone" not in adata.obs:
        raise ValueError("Clone-CUVGF-BMF requires obs['clone'] for clone-aware training metadata.")
    arrays: list[np.ndarray] = []
    for indices in obs_indices_by_time:
        vals = adata.obs["clone"].iloc[np.asarray(indices, dtype=int)].to_numpy()
        arrays.append(np.asarray([_clone_key(v) for v in vals], dtype=object))
    return arrays


def _counts_by_clone(clones: np.ndarray) -> dict[str, int]:
    out: dict[str, int] = {}
    for c in clones:
        if c:
            out[str(c)] = out.get(str(c), 0) + 1
    return out


def _safe_normalize_rows(mat: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    arr = np.asarray(mat, dtype=np.float64)
    denom = arr.sum(axis=1, keepdims=True)
    out = np.zeros_like(arr, dtype=np.float64)
    ok = denom[:, 0] > 0
    out[ok] = arr[ok] / denom[ok]
    if np.any(~ok):
        if fallback is None:
            out[~ok] = 1.0 / max(arr.shape[1], 1)
        else:
            out[~ok] = np.asarray(fallback, dtype=np.float64).reshape(1, -1)
    return out.astype(np.float32)


def _build_interval_reliability(
    clone_by_time: list[np.ndarray],
    *,
    pseudocount: float,
) -> list[dict[str, float]]:
    reliabilities: list[dict[str, float]] = []
    pseudo = max(float(pseudocount), 1e-6)
    for k in range(len(clone_by_time) - 1):
        left = _counts_by_clone(clone_by_time[k])
        right = _counts_by_clone(clone_by_time[k + 1])
        keys = set(left) | set(right)
        rel: dict[str, float] = {}
        for c in keys:
            n0 = float(left.get(c, 0))
            n1 = float(right.get(c, 0))
            if n0 <= 0.0 or n1 <= 0.0:
                rel[c] = 0.0
            else:
                support = float(np.sqrt(n0 * n1))
                rel[c] = support / (support + pseudo)
        reliabilities.append(rel)
    return reliabilities


def _build_fate_annotations(
    adata: Any,
    obs_indices_by_time: list[np.ndarray],
    clone_by_time: list[np.ndarray],
    *,
    label_key: str,
    alpha: float,
    eta: float,
) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, Any]]:
    """Training-only clone-fate priors and terminal branch responsibilities.

    Nonterminal cells inherit Dirichlet-smoothed terminal clone fate posteriors
    when their clone has terminal support; unsupported clones receive the global
    terminal prior with reliability zero.  Terminal cells receive a smoothed
    one-hot branch responsibility.  These arrays are never inference-time inputs.
    """
    if label_key not in adata.obs:
        raise ValueError(f"Clone-CUVGF-BMF requires terminal obs[{label_key!r}] for fate branches.")
    alpha = max(float(alpha), 1e-6)
    eta = float(np.clip(float(eta), 0.0, 1.0))

    final_indices = np.asarray(obs_indices_by_time[-1], dtype=int)
    final_labels_raw = np.asarray(adata.obs.iloc[final_indices][label_key].astype(str))
    fate_labels = np.asarray(sorted(np.unique(final_labels_raw).tolist()), dtype=object)
    if fate_labels.shape[0] == 0:
        raise ValueError("Clone-CUVGF-BMF found no terminal fate labels.")
    fate_to_code = {str(f): i for i, f in enumerate(fate_labels)}
    final_codes = np.asarray([fate_to_code[str(f)] for f in final_labels_raw], dtype=np.int64)
    n_fates = int(fate_labels.shape[0])
    global_counts = np.bincount(final_codes, minlength=n_fates).astype(np.float64)
    global_prior = global_counts / max(float(global_counts.sum()), 1.0)

    final_clones = clone_by_time[-1]
    clone_posterior: dict[str, np.ndarray] = {}
    clone_reliability: dict[str, float] = {}
    for c in sorted({str(x) for x in final_clones if x}):
        mask = final_clones == c
        n_c = int(mask.sum())
        if n_c <= 0:
            continue
        counts = np.bincount(final_codes[mask], minlength=n_fates).astype(np.float64)
        posterior = (counts + alpha * global_prior) / (float(n_c) + alpha)
        clone_posterior[c] = posterior.astype(np.float32)
        clone_reliability[c] = float(n_c) / (float(n_c) + alpha)

    fate_by_time: list[np.ndarray] = []
    omega_by_time: list[np.ndarray] = []
    for k, indices in enumerate(obs_indices_by_time):
        n = int(len(indices))
        H = np.repeat(global_prior.astype(np.float32)[None, :], n, axis=0)
        omega = np.zeros(n, dtype=np.float32)
        if k == len(obs_indices_by_time) - 1:
            labels = np.asarray(adata.obs.iloc[np.asarray(indices, dtype=int)][label_key].astype(str))
            for row, lab in enumerate(labels):
                code = fate_to_code.get(str(lab))
                if code is None:
                    continue
                H[row] = (eta * global_prior).astype(np.float32)
                H[row, code] += float(1.0 - eta)
                omega[row] = 1.0
        else:
            clones = clone_by_time[k]
            for row, c in enumerate(clones):
                key = str(c)
                if key in clone_posterior:
                    H[row] = clone_posterior[key]
                    omega[row] = np.float32(clone_reliability[key])
        fate_by_time.append(_safe_normalize_rows(H, global_prior))
        omega_by_time.append(omega.astype(np.float32, copy=False))

    metadata = {
        "fate_labels": [str(x) for x in fate_labels.tolist()],
        "n_fates": n_fates,
        "terminal_fate_counts": {str(fate_labels[i]): int(global_counts[i]) for i in range(n_fates)},
        "global_prior": global_prior.astype(float).tolist(),
        "n_terminal_supported_clones": int(len(clone_posterior)),
        "dirichlet_alpha": float(alpha),
        "terminal_onehot_eta": float(eta),
    }
    return fate_by_time, omega_by_time, metadata


# ---------------------------------------------------------------------------
# Stable clone-censored UOT backbone (no terminal-fate dominance)
# ---------------------------------------------------------------------------


class CloneCostStats:
    def __init__(
        self,
        *,
        interval: int,
        source_cells: int,
        target_cells: int,
        informative_source_clones: int,
        informative_target_clones: int,
        shared_reliable_clones: int,
        mean_reliability: float,
    ) -> None:
        self.interval = int(interval)
        self.source_cells = int(source_cells)
        self.target_cells = int(target_cells)
        self.informative_source_clones = int(informative_source_clones)
        self.informative_target_clones = int(informative_target_clones)
        self.shared_reliable_clones = int(shared_reliable_clones)
        self.mean_reliability = float(mean_reliability)


class CloneCensoredUOTCoupling(ChunkedTransportCouplingStrategy):
    """Chunked stable UOT backbone with conservative censored clone geometry.

    BMF uses this only to fix total row masses and target-support geometry before
    branch splitting.  Terminal fate posteriors are deliberately excluded from
    this stable backbone so fate evidence cannot directly change total mass.
    """

    def __init__(
        self,
        *,
        clone_by_time: list[np.ndarray],
        reliability_by_interval: list[dict[str, float]],
        chunk_size: int = 1000,
        alpha_regm: float = 1.0,
        reg_strategy: str = "max_over_time",
        reg: float | None = None,
        reg_m: float | None = None,
        auto_reg_device: str = "cpu",
        clone_lambda: float = 0.15,
        reliability_min: float = 0.10,
        same_clone_discount: float = 0.75,
        mismatch_penalty: float = 0.25,
        cost_eps: float = 1e-6,
        regularization_block_policy: str = "first_block_per_time",
    ):
        super().__init__(
            solver_mode="uot",
            chunk_size=int(chunk_size),
            use_mini_batch=True,
            alpha_regm=float(alpha_regm),
            reg_strategy=str(reg_strategy),
            reg=reg,
            reg_m=reg_m,
            auto_reg_device=auto_reg_device,
            normalize_cost=False,
            auto_normalize_large_cost=True,
            regularization_block_policy=str(regularization_block_policy),
        )
        self.clone_by_time = clone_by_time
        self.reliability_by_interval = reliability_by_interval
        self.clone_lambda = float(clone_lambda)
        self.reliability_min = float(reliability_min)
        self.same_clone_discount = float(same_clone_discount)
        self.mismatch_penalty = float(mismatch_penalty)
        self.cost_eps = float(cost_eps)
        self._stats = self._summarize_clone_stats()

    def _summarize_clone_stats(self) -> list[CloneCostStats]:
        stats: list[CloneCostStats] = []
        for k, rel in enumerate(self.reliability_by_interval):
            src_keys = {c for c in self.clone_by_time[k] if c}
            tgt_keys = {c for c in self.clone_by_time[k + 1] if c}
            shared = [c for c in (src_keys & tgt_keys) if rel.get(c, 0.0) >= self.reliability_min]
            vals = [rel.get(c, 0.0) for c in shared]
            stats.append(
                CloneCostStats(
                    interval=k,
                    source_cells=int(len(self.clone_by_time[k])),
                    target_cells=int(len(self.clone_by_time[k + 1])),
                    informative_source_clones=int(len(src_keys)),
                    informative_target_clones=int(len(tgt_keys)),
                    shared_reliable_clones=int(len(shared)),
                    mean_reliability=float(np.mean(vals)) if vals else 0.0,
                )
            )
        return stats

    def describe_metadata(self) -> dict[str, Any]:
        return {
            "kind": "bmf_stable_clone_censored_uot",
            "solver_mode": "uot",
            "chunk_size": self.chunk_size,
            "alpha_regm": self.alpha_regm,
            "reg_strategy": self.reg_strategy,
            "fixed_reg": self.fixed_reg,
            "fixed_reg_m": self.fixed_reg_m,
            "clone_lambda": self.clone_lambda,
            "reliability_min": self.reliability_min,
            "same_clone_discount": self.same_clone_discount,
            "mismatch_penalty": self.mismatch_penalty,
            "regularization_block_policy": self.regularization_block_policy,
            "clone_stats": [s.__dict__ for s in self._stats],
        }

    def _split_blocks(self, n_source: int, n_target: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
        if not self.use_mini_batch:
            return [np.arange(n_source)], [np.arange(n_target)]
        n_chunks = max(1, n_source // self.chunk_size + 1)
        return list(np.array_split(np.arange(n_source), n_chunks)), list(np.array_split(np.arange(n_target), n_chunks))

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
        del t0, t1
        x0_t = torch.as_tensor(x0_block, dtype=torch.float32, device=device)
        x1_t = torch.as_tensor(x1_block, dtype=torch.float32, device=device)
        base = torch.cdist(x0_t, x1_t).pow(2)
        scale = torch.clamp(torch.quantile(base.detach().reshape(-1), 0.90), min=self.cost_eps)
        cost = base / scale

        same_hits = 0
        mismatch_hits = 0
        clone_lambda = float(self.clone_lambda)
        if clone_lambda > 0.0:
            src_clones = self.clone_by_time[time_idx][np.asarray(source_indices, dtype=int)]
            tgt_clones = self.clone_by_time[time_idx + 1][np.asarray(target_indices, dtype=int)]
            rel = self.reliability_by_interval[time_idx]
            src_rel = np.asarray([rel.get(str(c), 0.0) if c else 0.0 for c in src_clones], dtype=np.float32)
            tgt_rel = np.asarray([rel.get(str(c), 0.0) if c else 0.0 for c in tgt_clones], dtype=np.float32)
            reliable_pair = (src_rel[:, None] >= self.reliability_min) & (tgt_rel[None, :] >= self.reliability_min)
            same = np.equal(src_clones[:, None], tgt_clones[None, :])
            same &= (src_clones[:, None] != "") & (tgt_clones[None, :] != "")
            reliable_same = reliable_pair & same
            mismatch = reliable_pair & (~same)
            same_strength = np.minimum(src_rel[:, None], tgt_rel[None, :]) * reliable_same.astype(np.float32)
            mismatch_strength = np.sqrt(np.maximum(src_rel[:, None] * tgt_rel[None, :], 0.0)) * mismatch.astype(np.float32)
            same_t = torch.as_tensor(same_strength, dtype=torch.float32, device=device)
            mismatch_t = torch.as_tensor(mismatch_strength, dtype=torch.float32, device=device)
            cost = cost + clone_lambda * (self.mismatch_penalty * mismatch_t - self.same_clone_discount * same_t)
            same_hits = int(np.count_nonzero(same_strength > 0.0))
            mismatch_hits = int(np.count_nonzero(mismatch_strength > 0.0))

        return PairwiseCost(
            cost_matrix=cost,
            normalize_cost=False,
            metadata={
                "time_idx": int(time_idx),
                "stable_cost_scale_q90": float(scale.detach().cpu().item()),
                "stable_same_clone_pairs": same_hits,
                "stable_reliable_mismatch_pairs": mismatch_hits,
                "stable_cost_objective": "latent_geometry_plus_conservative_censored_clone_only_no_terminal_fate",
            },
        )


# ---------------------------------------------------------------------------
# BMF branch projection coupling and backend
# ---------------------------------------------------------------------------


def _store_to_top_edges(store: Any, *, max_edges_per_row: int, min_mass: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    src_parts: list[np.ndarray] = []
    tgt_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []
    source_n = int(getattr(store, "source_n"))
    target_n = int(getattr(store, "target_n"))
    original_row_mass = np.zeros(source_n, dtype=np.float64)
    original_col_mass = np.zeros(target_n, dtype=np.float64)

    def add_plan(plan: np.ndarray, src_group: np.ndarray, tgt_group: np.ndarray) -> None:
        plan_arr = np.asarray(plan, dtype=np.float64)
        src_group = np.asarray(src_group, dtype=np.int64)
        tgt_group = np.asarray(tgt_group, dtype=np.int64)
        if plan_arr.size == 0:
            return
        k = max(1, int(max_edges_per_row))
        selected_by_row: dict[int, set[int]] = {}
        for local_i in range(plan_arr.shape[0]):
            row = plan_arr[local_i]
            positive = np.flatnonzero(row > float(min_mass))
            if positive.size == 0:
                continue
            full_row_sum = float(row[positive].sum())
            original_row_mass[int(src_group[local_i])] += full_row_sum
            np.add.at(original_col_mass, tgt_group[positive].astype(np.int64), row[positive].astype(np.float64))
            if positive.size > k:
                choose = positive[np.argpartition(row[positive], -k)[-k:]]
            else:
                choose = positive
            selected_by_row[local_i] = set(int(x) for x in choose.tolist())

        # Bidirectional sparse support repair: top-k per row alone can silently
        # drop target columns, invalidating the proposal's target-column
        # constraints.  Add at least one high-mass source edge per target within
        # each stable UOT block before row rescaling.  This preserves O(n*k)
        # behavior at Weinreb scale while making true target residuals observable
        # and usually much smaller.
        if plan_arr.shape[0] > 0 and plan_arr.shape[1] > 0:
            for local_j in range(plan_arr.shape[1]):
                col = plan_arr[:, local_j]
                positive_rows = np.flatnonzero(col > float(min_mass))
                if positive_rows.size == 0:
                    continue
                best_i = int(positive_rows[np.argmax(col[positive_rows])])
                selected_by_row.setdefault(best_i, set()).add(int(local_j))

        for local_i, choose_set in selected_by_row.items():
            if not choose_set:
                continue
            row = plan_arr[local_i]
            positive = np.flatnonzero(row > float(min_mass))
            full_row_sum = float(row[positive].sum())
            choose = np.asarray(sorted(choose_set), dtype=np.int64)
            weights = row[choose]
            keep = weights > 0
            if not np.any(keep):
                continue
            choose = choose[keep]
            weights = weights[keep]
            retained_sum = float(weights.sum())
            if retained_sum > 0.0 and full_row_sum > 0.0:
                # Top-k support pruning is a scalability approximation only; it
                # must not change the stable UOT row-mass target.  Rescale each
                # retained row block to the original row mass represented by the
                # block and record omitted support as a diagnostic.
                weights = weights * (full_row_sum / retained_sum)
            src_parts.append(np.repeat(src_group[local_i], choose.size).astype(np.int64))
            tgt_parts.append(tgt_group[choose].astype(np.int64))
            weight_parts.append(weights.astype(np.float32))

    if getattr(store, "dense_plan", None) is not None:
        dense = np.asarray(store.dense_plan, dtype=np.float64)
        add_plan(dense, np.arange(dense.shape[0], dtype=np.int64), np.arange(dense.shape[1], dtype=np.int64))
    for sub, sg, tg in zip(getattr(store, "sub_plans", []) or [], getattr(store, "source_groups", []) or [], getattr(store, "target_groups", []) or []):
        add_plan(np.asarray(sub), np.asarray(sg), np.asarray(tg))

    if not src_parts:
        raise RuntimeError("BMF stable UOT store produced no positive support edges.")
    src = np.concatenate(src_parts).astype(np.int64, copy=False)
    tgt = np.concatenate(tgt_parts).astype(np.int64, copy=False)
    w = np.concatenate(weight_parts).astype(np.float32, copy=False)
    retained_row_mass = np.bincount(src, weights=w.astype(np.float64), minlength=source_n).astype(np.float64)
    if original_row_mass.sum() <= 0.0:
        original_row_mass = retained_row_mass.copy()
    row_mass_l1 = float(np.abs(retained_row_mass - original_row_mass).sum() / max(float(np.abs(original_row_mass).sum()), 1e-12))
    diag = {
        "stable_row_mass_l1_after_support_pruning": row_mass_l1,
        "stable_row_mass_sum_original": float(original_row_mass.sum()),
        "stable_row_mass_sum_retained": float(retained_row_mass.sum()),
        "stable_target_col_mass_sum_original": float(original_col_mass.sum()),
        "stable_support_zero_target_columns": float(np.count_nonzero(original_col_mass <= 0.0)),
    }
    return src, tgt, w, original_row_mass.astype(np.float32), original_col_mass.astype(np.float32), diag


def _ipf_branch_edges(
    edge_src: np.ndarray,
    edge_tgt: np.ndarray,
    edge_weight: np.ndarray,
    *,
    source_n: int,
    target_n: int,
    q_source: np.ndarray,
    q_target: np.ndarray,
    max_iter: int,
    eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Sparse branch projection over a fixed stable support.

    For each branch f, scale edge weights on the stable support to match
    source-row branch masses exactly up to IPF tolerance and target-branch masses
    as closely as support permits. Target branch totals are globally rescaled to
    the source branch totals so the branch problem is feasible; residuals are
    recorded and treated as diagnostics.
    """
    edge_src = np.asarray(edge_src, dtype=np.int64)
    edge_tgt = np.asarray(edge_tgt, dtype=np.int64)
    base_w = np.asarray(edge_weight, dtype=np.float64)
    q_source = _safe_normalize_rows(q_source)
    q_target = _safe_normalize_rows(q_target)
    n_fates = int(q_source.shape[1])
    row_total = np.bincount(edge_src, weights=base_w, minlength=source_n).astype(np.float64)
    col_total = np.bincount(edge_tgt, weights=base_w, minlength=target_n).astype(np.float64)

    src_out: list[np.ndarray] = []
    tgt_out: list[np.ndarray] = []
    br_out: list[np.ndarray] = []
    wt_out: list[np.ndarray] = []
    row_resids = []
    col_resids = []

    for f in range(n_fates):
        row_target = row_total * q_source[:, f]
        col_target = col_total * q_target[:, f]
        row_sum = float(row_target.sum())
        col_sum = float(col_target.sum())
        if row_sum <= eps:
            continue
        if col_sum > eps:
            col_target = col_target * (row_sum / col_sum)
        else:
            col_target = col_total * (row_sum / max(float(col_total.sum()), eps))
        z = base_w * np.sqrt(np.maximum(q_source[edge_src, f], eps) * np.maximum(q_target[edge_tgt, f], eps))
        z = np.maximum(z, eps * base_w.max(initial=1.0))
        for _ in range(max(1, int(max_iter))):
            rs = np.bincount(edge_src, weights=z, minlength=source_n).astype(np.float64)
            z *= (row_target[edge_src] / np.maximum(rs[edge_src], eps))
            cs = np.bincount(edge_tgt, weights=z, minlength=target_n).astype(np.float64)
            z *= (col_target[edge_tgt] / np.maximum(cs[edge_tgt], eps))
        rs = np.bincount(edge_src, weights=z, minlength=source_n).astype(np.float64)
        cs = np.bincount(edge_tgt, weights=z, minlength=target_n).astype(np.float64)
        denom_r = max(float(np.abs(row_target).sum()), eps)
        denom_c = max(float(np.abs(col_target).sum()), eps)
        row_resids.append(float(np.abs(rs - row_target).sum() / denom_r))
        col_resids.append(float(np.abs(cs - col_target).sum() / denom_c))
        keep = z > eps
        if np.any(keep):
            src_out.append(edge_src[keep])
            tgt_out.append(edge_tgt[keep])
            br_out.append(np.full(int(np.count_nonzero(keep)), f, dtype=np.int64))
            wt_out.append(z[keep].astype(np.float32))

    if not wt_out:
        raise RuntimeError("BMF branch projection produced no positive branch edges.")
    src_b = np.concatenate(src_out).astype(np.int64, copy=False)
    tgt_b = np.concatenate(tgt_out).astype(np.int64, copy=False)
    branch_b = np.concatenate(br_out).astype(np.int64, copy=False)
    weight_b = np.concatenate(wt_out).astype(np.float32, copy=False)
    cdf = np.cumsum(weight_b.astype(np.float64))
    cdf /= float(cdf[-1])
    cdf[-1] = 1.0
    diag = {
        "branch_ipf_mean_row_l1_residual": float(np.mean(row_resids)) if row_resids else float("nan"),
        "branch_ipf_max_row_l1_residual": float(np.max(row_resids)) if row_resids else float("nan"),
        "branch_ipf_mean_col_l1_residual": float(np.mean(col_resids)) if col_resids else float("nan"),
        "branch_ipf_max_col_l1_residual": float(np.max(col_resids)) if col_resids else float("nan"),
    }
    return src_b, tgt_b, branch_b, cdf.astype(np.float64), {**diag, "branch_edge_weight_sum": float(weight_b.sum()), "n_branch_edges": int(weight_b.shape[0])}, weight_b


def _joint_branch_project_three_times(
    src0: np.ndarray,
    tgt0: np.ndarray,
    w0: np.ndarray,
    src1: np.ndarray,
    tgt1: np.ndarray,
    w1: np.ndarray,
    *,
    n0: int,
    n1: int,
    n2: int,
    row_mass0: np.ndarray,
    row_mass1: np.ndarray,
    target_col0: np.ndarray,
    target_col1: np.ndarray,
    h0: np.ndarray,
    h1: np.ndarray,
    h2: np.ndarray,
    omega0: np.ndarray,
    omega1: np.ndarray,
    omega2: np.ndarray,
    source_prior_power: float,
    mid_prior_power: float,
    terminal_prior_power: float,
    max_iter: int,
    eps: float,
) -> dict[str, Any]:
    """Sparse joint BMF projection for the day2/day4/day6 Weinreb panel.

    The variables are branch-resolved sparse edge weights Gamma0(i,f,j) and
    Gamma1(j,f,l) on the stable UOT supports plus a shared day-4 branch boundary
    s_j(f).  Alternating multiplicative projections enforce the proposal's hard
    constraints at the represented support scale: day2 source branch row totals,
    day4 target totals, shared day4 branch boundary, day4 source row totals, and
    day6 target totals.  Clone/fate posteriors enter only as bounded positive
    initialization/prior factors; the returned gate targets are feasible projected
    marginals, not raw clone priors.
    """
    src0 = np.asarray(src0, dtype=np.int64)
    tgt0 = np.asarray(tgt0, dtype=np.int64)
    w0 = np.asarray(w0, dtype=np.float64)
    src1 = np.asarray(src1, dtype=np.int64)
    tgt1 = np.asarray(tgt1, dtype=np.int64)
    w1 = np.asarray(w1, dtype=np.float64)
    row_mass0 = np.asarray(row_mass0, dtype=np.float64).reshape(-1)
    row_mass1 = np.asarray(row_mass1, dtype=np.float64).reshape(-1)
    target_col0 = np.asarray(target_col0, dtype=np.float64).reshape(-1)
    target_col1 = np.asarray(target_col1, dtype=np.float64).reshape(-1)
    h0 = _safe_normalize_rows(h0).astype(np.float64)
    h1 = _safe_normalize_rows(h1).astype(np.float64)
    h2 = _safe_normalize_rows(h2).astype(np.float64)
    omega0 = np.clip(np.asarray(omega0, dtype=np.float64).reshape(-1), 0.0, 1.0)
    omega1 = np.clip(np.asarray(omega1, dtype=np.float64).reshape(-1), 0.0, 1.0)
    omega2 = np.clip(np.asarray(omega2, dtype=np.float64).reshape(-1), 0.0, 1.0)
    source_prior_power = float(max(source_prior_power, 0.0))
    mid_prior_power = float(max(mid_prior_power, 0.0))
    terminal_prior_power = float(max(terminal_prior_power, 0.0))
    n_fates = int(h0.shape[1])
    eps = max(float(eps), 1e-12)

    global_prior = _safe_normalize_rows(h2.mean(axis=0, keepdims=True))[0].astype(np.float64)

    def reliability_weighted_prior(h: np.ndarray, omega: np.ndarray) -> np.ndarray:
        # Clone/fate posteriors are bounded priors.  Unsupported or unreliable
        # rows revert to the neutral terminal prior and exert no branch-specific
        # force; high-reliability rows retain their clone/fate posterior.  These
        # priors initialize and bias the sparse projection but are never hard row
        # constraints.
        om = np.clip(omega.reshape(-1, 1), 0.0, 1.0)
        prior = (1.0 - om) * global_prior.reshape(1, -1) + om * h
        return _safe_normalize_rows(prior, global_prior).astype(np.float64)

    p0 = reliability_weighted_prior(h0, omega0)
    p1 = reliability_weighted_prior(h1, omega1)
    p2 = reliability_weighted_prior(h2, omega2)

    # Proposal target-column masses come from the unpruned stable UOT stores (or
    # observed target columns under target-constrained UOT), not from the pruned
    # support-induced columns.  The sparse projection scales to these true target
    # columns; residuals therefore diagnose real support infeasibility.
    b1_col = target_col0.astype(np.float64, copy=True)
    b2_col = target_col1.astype(np.float64, copy=True)
    if b1_col.sum() > eps:
        b1_col *= float(row_mass0.sum()) / max(float(b1_col.sum()), eps)
    if b2_col.sum() > eps:
        b2_col *= float(row_mass1.sum()) / max(float(b2_col.sum()), eps)
    support_col0 = np.bincount(tgt0, weights=w0, minlength=n1).astype(np.float64)
    support_col1 = np.bincount(tgt1, weights=w1, minlength=n2).astype(np.float64)

    # Initial branch factors are reliability-weighted priors, not hard targets.
    # p1 anchors the shared
    # boundary softly through initialization; alternating projections may move it.
    G0 = np.maximum(w0[:, None], eps) * (
        np.maximum(p0[src0], eps) ** source_prior_power
    ) * (
        np.maximum(p1[tgt0], eps) ** mid_prior_power
    )
    G1 = np.maximum(w1[:, None], eps) * (
        np.maximum(p1[src1], eps) ** mid_prior_power
    ) * (
        np.maximum(p2[tgt1], eps) ** terminal_prior_power
    )
    S = np.maximum(p1.astype(np.float64), eps)
    S = S / np.maximum(S.sum(axis=1, keepdims=True), eps)

    for _ in range(max(2, int(max_iter))):
        # Day2 source row totals are hard constraints, but the branch split q_i^0
        # is a feasible projection variable.  Do not scale to row_mass*h0 here;
        # clone priors affect G0 only through reliability-weighted costs/factors.
        rs0_total = np.zeros(n0, dtype=np.float64)
        np.add.at(rs0_total, src0, G0.sum(axis=1))
        G0 *= (row_mass0[src0] / np.maximum(rs0_total[src0], eps))[:, None]

        # Day4 total target columns for interval 0.
        col0 = np.zeros(n1, dtype=np.float64)
        np.add.at(col0, tgt0, G0.sum(axis=1))
        G0 *= (b1_col[tgt0] / np.maximum(col0[tgt0], eps))[:, None]

        # Day6 target columns for interval 1.
        col1 = np.zeros(n2, dtype=np.float64)
        np.add.at(col1, tgt1, G1.sum(axis=1))
        G1 *= (b2_col[tgt1] / np.maximum(col1[tgt1], eps))[:, None]

        # Shared day4 branch boundary: Gamma0 columns and Gamma1 source rows use
        # the same s_j(f), with different total scalings b1_j and r_j^{0,1}.
        c0f = np.zeros((n1, n_fates), dtype=np.float64)
        r1f = np.zeros((n1, n_fates), dtype=np.float64)
        np.add.at(c0f, tgt0, G0)
        np.add.at(r1f, src1, G1)
        s_from_0 = c0f / np.maximum(b1_col[:, None], eps)
        s_from_1 = r1f / np.maximum(row_mass1[:, None], eps)
        supported = ((b1_col[:, None] > eps).astype(np.float64) + (row_mass1[:, None] > eps).astype(np.float64))
        S = (s_from_0 * (b1_col[:, None] > eps) + s_from_1 * (row_mass1[:, None] > eps) + eps * p1) / np.maximum(supported + eps, eps)
        S = np.maximum(S, eps)
        S = S / np.maximum(S.sum(axis=1, keepdims=True), eps)
        target_c0f = b1_col[:, None] * S
        target_r1f = row_mass1[:, None] * S
        c0f = np.zeros((n1, n_fates), dtype=np.float64)
        r1f = np.zeros((n1, n_fates), dtype=np.float64)
        np.add.at(c0f, tgt0, G0)
        np.add.at(r1f, src1, G1)
        G0 *= target_c0f[tgt0] / np.maximum(c0f[tgt0], eps)
        G1 *= target_r1f[src1] / np.maximum(r1f[src1], eps)

    # Final feasible gate targets from projected branch masses.
    row0f = np.zeros((n0, n_fates), dtype=np.float64)
    c0f = np.zeros((n1, n_fates), dtype=np.float64)
    r1f = np.zeros((n1, n_fates), dtype=np.float64)
    col1f = np.zeros((n2, n_fates), dtype=np.float64)
    np.add.at(row0f, src0, G0)
    np.add.at(c0f, tgt0, G0)
    np.add.at(r1f, src1, G1)
    np.add.at(col1f, tgt1, G1)
    q0 = _safe_normalize_rows(row0f, global_prior)
    s = _safe_normalize_rows(c0f / np.maximum(b1_col[:, None], eps), global_prior)

    def rel_l1(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.abs(np.asarray(a) - np.asarray(b)).sum() / max(float(np.abs(b).sum()), eps))

    diag = {
        "joint_branch_projection": "alternating_sparse_projection_shared_day4_boundary",
        "joint_row0_total_l1_residual": rel_l1(row0f.sum(axis=1), row_mass0),
        "joint_day4_total_col_l1_residual": rel_l1(c0f.sum(axis=1), b1_col),
        "joint_day4_boundary_l1_residual": rel_l1(c0f, b1_col[:, None] * s),
        "joint_day4_source_boundary_l1_residual": rel_l1(r1f, row_mass1[:, None] * s),
        "joint_day6_total_col_l1_residual": rel_l1(col1f.sum(axis=1), b2_col),
        "true_target_col0_support_l1_before_projection": rel_l1(support_col0, b1_col),
        "true_target_col1_support_l1_before_projection": rel_l1(support_col1, b2_col),
        "true_target_col0_zero_supported_mass_fraction": float(b1_col[support_col0 <= eps].sum() / max(float(b1_col.sum()), eps)),
        "true_target_col1_zero_supported_mass_fraction": float(b2_col[support_col1 <= eps].sum() / max(float(b2_col.sum()), eps)),
        "projection_clone_prior_mean_l1": float(np.mean((omega0.reshape(-1, 1) * np.abs(q0 - h0)).sum(axis=1))),
        "projection_mean_reliability_t0": float(np.mean(omega0)) if omega0.size else 0.0,
        "projection_mean_reliability_t1": float(np.mean(omega1)) if omega1.size else 0.0,
        "projection_mean_reliability_t2": float(np.mean(omega2)) if omega2.size else 0.0,
        "branch_source_prior_power": float(source_prior_power),
        "branch_mid_prior_power": float(mid_prior_power),
        "branch_terminal_prior_power": float(terminal_prior_power),
    }

    out: dict[str, Any] = {"q0": q0.astype(np.float32), "s": s.astype(np.float32), "diagnostics": diag}
    for name, src, tgt, G in [("0", src0, tgt0, G0), ("1", src1, tgt1, G1)]:
        edge_mass = G.sum(axis=1)
        keep_edge = edge_mass > eps
        kept_src = src[keep_edge]
        kept_tgt = tgt[keep_edge]
        kept_G = G[keep_edge]
        rows, branches = np.nonzero(kept_G > eps)
        weights = kept_G[rows, branches].astype(np.float32)
        cdf = np.cumsum(weights.astype(np.float64))
        cdf /= float(cdf[-1])
        cdf[-1] = 1.0
        out[f"src{name}"] = kept_src[rows].astype(np.int64, copy=False)
        out[f"tgt{name}"] = kept_tgt[rows].astype(np.int64, copy=False)
        out[f"branch{name}"] = branches.astype(np.int64, copy=False)
        out[f"weight{name}"] = weights
        out[f"cdf{name}"] = cdf.astype(np.float64, copy=False)
    return out


class CloneBMFBranchCoupling:
    """Persistent-branch coupling built from stable UOT plus joint BMF lift."""

    supports_minibatch = True

    def __init__(
        self,
        *,
        stable_coupling: CloneCensoredUOTCoupling,
        fate_by_time: list[np.ndarray],
        fate_reliability_by_time: list[np.ndarray],
        fate_metadata: dict[str, Any],
        max_edges_per_row: int = 32,
        branch_ipf_iter: int = 12,
        branch_eps: float = 1e-9,
        source_prior_power: float = 0.5,
        mid_prior_power: float = 0.5,
        terminal_prior_power: float = 0.5,
    ) -> None:
        self.stable_coupling = stable_coupling
        self.fate_by_time = fate_by_time
        self.fate_reliability_by_time = fate_reliability_by_time
        self.fate_metadata = dict(fate_metadata or {})
        self.max_edges_per_row = int(max_edges_per_row)
        self.branch_ipf_iter = int(branch_ipf_iter)
        self.branch_eps = float(branch_eps)
        self.source_prior_power = float(source_prior_power)
        self.mid_prior_power = float(mid_prior_power)
        self.terminal_prior_power = float(terminal_prior_power)

    def describe_metadata(self) -> dict[str, Any]:
        return {
            "kind": "clone_cuvgf_bmf_branch_projection",
            "n_fates": int(self.fate_metadata.get("n_fates", 0)),
            "fate_labels": list(self.fate_metadata.get("fate_labels", [])),
            "max_edges_per_row": self.max_edges_per_row,
            "branch_ipf_iter": self.branch_ipf_iter,
            "source_prior_power": self.source_prior_power,
            "mid_prior_power": self.mid_prior_power,
            "terminal_prior_power": self.terminal_prior_power,
            "stable_coupling": self.stable_coupling.describe_metadata(),
        }

    def build_state(self, X: list[np.ndarray], t_train: torch.Tensor, device: torch.device) -> CouplingState:
        stable_state = self.stable_coupling.build_state(X, t_train, device)
        top_edges: list[dict[str, Any]] = []
        for time_idx in range(len(X) - 1):
            stable_store = stable_state.plan_store(time_idx)
            src, tgt, w, row_mass, target_col_mass, support_diag = _store_to_top_edges(
                stable_store,
                max_edges_per_row=self.max_edges_per_row,
                min_mass=self.branch_eps,
            )
            top_edges.append({"src": src, "tgt": tgt, "w": w, "row_mass": row_mass, "target_col_mass": target_col_mass, "support_diag": support_diag})

        if len(top_edges) == 2:
            joint = _joint_branch_project_three_times(
                top_edges[0]["src"],
                top_edges[0]["tgt"],
                top_edges[0]["w"],
                top_edges[1]["src"],
                top_edges[1]["tgt"],
                top_edges[1]["w"],
                n0=X[0].shape[0],
                n1=X[1].shape[0],
                n2=X[2].shape[0],
                row_mass0=top_edges[0]["row_mass"],
                row_mass1=top_edges[1]["row_mass"],
                target_col0=top_edges[0]["target_col_mass"],
                target_col1=top_edges[1]["target_col_mass"],
                h0=self.fate_by_time[0],
                h1=self.fate_by_time[1],
                h2=self.fate_by_time[2],
                omega0=self.fate_reliability_by_time[0] if len(self.fate_reliability_by_time) > 0 else np.ones(X[0].shape[0], dtype=np.float32),
                omega1=self.fate_reliability_by_time[1] if len(self.fate_reliability_by_time) > 1 else np.ones(X[1].shape[0], dtype=np.float32),
                omega2=self.fate_reliability_by_time[2] if len(self.fate_reliability_by_time) > 2 else np.ones(X[2].shape[0], dtype=np.float32),
                source_prior_power=self.source_prior_power,
                mid_prior_power=self.mid_prior_power,
                terminal_prior_power=self.terminal_prior_power,
                max_iter=self.branch_ipf_iter,
                eps=self.branch_eps,
            )
        else:
            joint = {}

        plan_stores = []
        pair_metadata = []
        for time_idx in range(len(X) - 1):
            row_mass = top_edges[time_idx]["row_mass"]
            support_diag = top_edges[time_idx]["support_diag"]
            if len(top_edges) == 2:
                suffix = str(time_idx)
                src_b = np.asarray(joint[f"src{suffix}"], dtype=np.int64)
                tgt_b = np.asarray(joint[f"tgt{suffix}"], dtype=np.int64)
                branch_b = np.asarray(joint[f"branch{suffix}"], dtype=np.int64)
                cdf_b = np.asarray(joint[f"cdf{suffix}"], dtype=np.float64)
                weight_b = np.asarray(joint[f"weight{suffix}"], dtype=np.float32)
                gate_targets = np.asarray(joint["q0" if time_idx == 0 else "s"], dtype=np.float32)
                diag = dict(joint.get("diagnostics", {}))
            else:
                # Fallback for non-Weinreb panels with a different number of
                # time points.  The active approved benchmark uses three times;
                # this path is marked as an approximation in metadata.
                q_src = self.fate_by_time[time_idx]
                q_tgt = self.fate_by_time[time_idx + 1]
                src_b, tgt_b, branch_b, cdf_b, diag, weight_b = _ipf_branch_edges(
                    top_edges[time_idx]["src"],
                    top_edges[time_idx]["tgt"],
                    top_edges[time_idx]["w"],
                    source_n=X[time_idx].shape[0],
                    target_n=X[time_idx + 1].shape[0],
                    q_source=q_src,
                    q_target=q_tgt,
                    max_iter=self.branch_ipf_iter,
                    eps=self.branch_eps,
                )
                gate_targets = q_src.astype(np.float32, copy=False)
            metadata_i = {
                **diag,
                **support_diag,
                "time_idx": int(time_idx),
                "source_n": int(X[time_idx].shape[0]),
                "target_n": int(X[time_idx + 1].shape[0]),
                "stable_support_edges": int(top_edges[time_idx]["src"].shape[0]),
                "stable_row_mass_sum_preserved": float(row_mass.sum()),
                "branch_projection": "joint_sparse_bmf_shared_day4_boundary" if len(top_edges) == 2 else "fallback_interval_sparse_ipf",
                "branch_edge_src": src_b,
                "branch_edge_tgt": tgt_b,
                "branch_edge_branch": branch_b,
                "branch_edge_cdf": cdf_b,
                "branch_edge_weight": weight_b,
                "stable_row_mass": row_mass,
                "gate_targets_source": gate_targets.astype(np.float32, copy=False),
            }
            pair_metadata.append({k: v for k, v in metadata_i.items() if not isinstance(v, np.ndarray)})
            # The dense values are not used for sampling; arrays live in metadata_i.
            from CytoBridge.tl.flow_matching_backends import CouplingPlanStore

            plan_stores.append(
                CouplingPlanStore.from_edges(
                    source_n=X[time_idx].shape[0],
                    target_n=X[time_idx + 1].shape[0],
                    edge_src=src_b,
                    edge_tgt=tgt_b,
                    edge_weight=weight_b,
                    terminal_mass_rows=row_mass,
                    metadata=metadata_i,
                )
            )
        return CouplingState(plan_stores=plan_stores, metadata={**self.describe_metadata(), "pair_metadata": pair_metadata})

    def sample_pairs(self, state: CouplingState, X: list[np.ndarray], time_idx: int, batch_size: int, device: torch.device) -> PairBatch | None:
        store = state.plan_store(time_idx)
        md = store.metadata
        cdf = np.asarray(md["branch_edge_cdf"], dtype=np.float64)
        edge_idx = np.searchsorted(cdf, np.random.random(size=int(batch_size)), side="left")
        edge_idx = np.clip(edge_idx, 0, cdf.shape[0] - 1)
        src = np.asarray(md["branch_edge_src"], dtype=np.int64)[edge_idx]
        tgt = np.asarray(md["branch_edge_tgt"], dtype=np.int64)[edge_idx]
        branch = np.asarray(md["branch_edge_branch"], dtype=np.int64)[edge_idx]
        row_mass = np.asarray(md["stable_row_mass"], dtype=np.float32)
        gate_targets = np.asarray(md["gate_targets_source"], dtype=np.float32)[src]
        batch = PairBatch(
            x0=torch.from_numpy(np.asarray(X[time_idx][src], dtype=np.float32)).to(device),
            x1=torch.from_numpy(np.asarray(X[time_idx + 1][tgt], dtype=np.float32)).to(device),
            idx0=src,
            idx1=tgt,
            time_idx=int(time_idx),
            sampled_terminal_mass=row_mass[src].reshape(-1, 1).astype(np.float32),
        )
        # dataclasses are not slotted; attach branch/gate data for the BMF backend.
        batch.branch_codes = branch
        batch.gate_targets = gate_targets
        return batch


class BranchFlowMatchingBackend(FlowMatchingBackend):
    """Flow-matching backend that carries branch codes into the loss hook."""

    def sample_batch(self, X: list[np.ndarray], t_train: torch.Tensor, batch_size: int, device: torch.device) -> FlowMatchingBatch:
        if self.state is None:
            raise RuntimeError("BranchFlowMatchingBackend.prepare(...) must be called before sample_batch(...).")
        local_times = []
        global_times = []
        xts = []
        uts = []
        gts = []
        loss_weights_list = []
        noises = []
        branches = []
        gate_targets = []
        gate_x = []
        gate_t = []
        n_fates = int(getattr(self.coupling, "fate_metadata", {}).get("n_fates", 0))
        for time_idx in range(len(t_train) - 1):
            pair_batch = self.coupling.sample_pairs(self.state, X, time_idx, batch_size, device)
            if pair_batch is None:
                continue
            t_local = self.path.sample_time(pair_batch.x0, pair_batch.x1).to(device=device, dtype=pair_batch.x0.dtype)
            eps = self.path.sample_noise_like(pair_batch.x0)
            xt = self.path.sample_xt(pair_batch.x0, pair_batch.x1, t_local, eps)
            ut = self.path.compute_conditional_flow(pair_batch.x0, pair_batch.x1, t_local, xt)
            mass_result = self.mass.compute(self.state, pair_batch, t_local, device)
            gt, loss_weights = self.path.compute_conditional_mass(mass_result.terminal_mass, t_local)
            time_interval = t_train[time_idx + 1] - t_train[time_idx]
            branch_codes = torch.as_tensor(getattr(pair_batch, "branch_codes"), dtype=torch.long, device=device)
            branch_onehot = torch.nn.functional.one_hot(branch_codes, num_classes=n_fates).to(dtype=pair_batch.x0.dtype)
            local_times.append(t_local)
            global_times.append(t_local * time_interval + t_train[time_idx])
            xts.append(xt)
            uts.append(ut / time_interval)
            gts.append(gt / time_interval)
            loss_weights_list.append(loss_weights)
            noises.append(eps)
            branches.append(branch_onehot)
            gate_targets.append(torch.as_tensor(getattr(pair_batch, "gate_targets"), dtype=pair_batch.x0.dtype, device=device))
            gate_x.append(pair_batch.x0)
            gate_t.append(torch.full((pair_batch.x0.shape[0], 1), float(t_train[time_idx].detach().cpu().item()), dtype=pair_batch.x0.dtype, device=device))
        if not global_times:
            empty = torch.empty(0, device=device)
            empty_latent = torch.empty(0, *X[0].shape[1:], device=device)
            empty_col = torch.empty(0, 1, device=device)
            return FlowMatchingBatch(t_local=empty, t_global=empty, xt=empty_latent, ut=empty_latent, gt=empty_col, loss_weights=empty_col, eps=empty_latent)
        return FlowMatchingBatch(
            t_local=torch.cat(local_times),
            t_global=torch.cat(global_times),
            xt=torch.cat(xts),
            ut=torch.cat(uts),
            gt=torch.cat(gts),
            loss_weights=torch.cat(loss_weights_list),
            eps=torch.cat(noises),
            extras={
                "branch_onehot": torch.cat(branches),
                "gate_targets": torch.cat(gate_targets),
                "gate_x": torch.cat(gate_x),
                "gate_t": torch.cat(gate_t),
            },
        )


# ---------------------------------------------------------------------------
# Branch model, branch loss, and sealed branch rollout
# ---------------------------------------------------------------------------


class BranchDynamicalModel(DynamicalModel):
    def __init__(self, latent_dim: int, config: dict[str, Any], *, n_fates: int):
        # Build only the minimal inherited module set and then expose BMF branch
        # heads as the operational velocity/growth modules.  This keeps package
        # optimizer/runtime conventions satisfied without leaving an untrained
        # standard VGFM head that could accidentally drive a non-branch rollout.
        config_for_base = dict(config)
        config_for_base["components"] = []
        super().__init__(latent_dim, config_for_base)
        self.n_fates = int(n_fates)
        vel_cfg = dict(config.get("branch_velocity_net") or config.get("velocity_net") or {})
        grow_cfg = dict(config.get("branch_growth_net") or config.get("growth_net") or {})
        gate_cfg = dict(config.get("gate_net") or {"hidden_dim": 256, "n_layers": 3, "residual": False})
        self.branch_velocity_net = HyperNetwork(input_dim=latent_dim + 1 + self.n_fates, output_dim=latent_dim, **vel_cfg)
        self.branch_growth_net = HyperNetwork(input_dim=latent_dim + 1 + self.n_fates, output_dim=1, **grow_cfg)
        self.gate_net = HyperNetwork(input_dim=latent_dim + 1, output_dim=self.n_fates, **gate_cfg)
        self.components = ["velocity", "growth"]
        # The package trainer computes the standard component losses before the
        # hook explicitly replaces them.  These compatible zero heads satisfy the
        # standard call signature [x,t] -> velocity/growth without becoming the
        # scientific dynamics; `_branch_loss_hook` replaces their losses and the
        # sealed simulation hook uses the branch heads below.
        self.velocity_net = _ZeroCompatibleHead(output_dim=latent_dim)
        self.growth_net = _ZeroCompatibleHead(output_dim=1)
        self.cytobridge_component_modules = {
            "velocity": ["branch_velocity_net", "gate_net"],
            "growth": ["branch_growth_net"],
        }


class _ZeroCompatibleHead(torch.nn.Module):
    def __init__(self, *, output_dim: int) -> None:
        super().__init__()
        self.output_dim = int(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.new_zeros((x.shape[0], self.output_dim))


def _build_branch_model(ctx: ModelBuildContext) -> torch.nn.Module:
    metadata = dict(getattr(ctx.training_data, "metadata", {}) or {})
    n_fates = int(metadata.get("bmf_n_fates") or ((ctx.resolved_config.get("model") or {}).get("n_fates") or 1))
    return BranchDynamicalModel(ctx.latent_dim, ctx.resolved_config["model"], n_fates=n_fates).to(ctx.device)


def _branch_loss_hook(context: FlowMatchingLossContext) -> FlowMatchingLossResult | None:
    extras = dict(getattr(context.batch, "extras", {}) or {})
    branch_onehot = extras.get("branch_onehot")
    if branch_onehot is None:
        return None
    model = context.model
    if not hasattr(model, "branch_velocity_net") or not hasattr(model, "branch_growth_net"):
        raise RuntimeError("Clone-CUVGF-BMF branch loss requires BranchDynamicalModel branch nets.")
    branch_onehot = branch_onehot.to(device=context.net_input.device, dtype=context.net_input.dtype)
    branch_input = torch.cat([context.x_input, context.t_input, branch_onehot], dim=1)
    weights = torch.as_tensor(context.batch.loss_weights, dtype=context.net_input.dtype, device=context.net_input.device)
    v_pred = model.branch_velocity_net(branch_input)
    g_pred = model.branch_growth_net(branch_input)
    v_loss = torch.mean(weights * (v_pred - context.batch.ut) ** 2)
    g_loss = torch.mean(weights * (g_pred - context.batch.gt) ** 2)

    gate_loss = torch.as_tensor(0.0, dtype=context.net_input.dtype, device=context.net_input.device)
    gate_weight = float(((context.stage_params.get("flow_matching") or {}).get("branch") or {}).get("lambda_gate", context.stage_params.get("lambda_gate", 1.0)))
    gate_targets = extras.get("gate_targets")
    if gate_targets is not None and gate_weight > 0.0:
        gate_x = extras.get("gate_x")
        gate_t = extras.get("gate_t")
        gate_targets = torch.as_tensor(gate_targets, dtype=context.net_input.dtype, device=context.net_input.device)
        gate_x = torch.as_tensor(gate_x, dtype=context.net_input.dtype, device=context.net_input.device)
        gate_t = torch.as_tensor(gate_t, dtype=context.net_input.dtype, device=context.net_input.device)
        logits = model.gate_net(torch.cat([gate_x, gate_t], dim=1))
        probs = torch.softmax(logits, dim=1)
        tgt = gate_targets / torch.clamp(gate_targets.sum(dim=1, keepdim=True), min=1e-8)
        # squared Hellinger gate supervision to feasible projection marginals
        gate_loss = torch.mean(torch.clamp(1.0 - torch.sum(torch.sqrt(torch.clamp(probs, min=1e-8)) * torch.sqrt(torch.clamp(tgt, min=1e-8)), dim=1), min=0.0, max=1.0))
    return FlowMatchingLossResult(
        replace_velocity_loss=v_loss,
        replace_growth_loss=g_loss,
        extra_loss=gate_weight * gate_loss,
        logs={"bmf_velocity_loss": float(v_loss.detach().cpu()), "bmf_growth_loss": float(g_loss.detach().cpu()), "bmf_gate_loss": float(gate_loss.detach().cpu())},
    )


def _build_inference_context(ctx: InferenceContextBuilderContext) -> InferenceContext:
    return InferenceContext(
        payload={"branch_rollout": True},
        visibility={"branch_rollout": "constant_prior"},
        provenance={
            "source": "Clone-CUVGF-BMF inference_context_builder",
            "future_rows_used": False,
            "allowed_inputs": ["t0 X_latent", "trained branch nets", "fate branch constants", "time grid"],
        },
        notes="BMF rollout expands each t0 cell into fixed fate branches with q_eta(x,t0) weights; clone labels and terminal labels are not inference inputs.",
    )


def _branch_simulation_hook(context: SimulationContext) -> SimulationResult:
    model = context.model
    if not hasattr(model, "branch_velocity_net") or not hasattr(model, "gate_net"):
        raise RuntimeError("Clone-CUVGF-BMF simulation requires branch model nets.")
    device = context.device or torch.device("cpu")
    x0 = context.x0_override
    if torch.is_tensor(x0):
        x0_t = x0.detach().to(device=device, dtype=torch.float32)
    else:
        x0_t = torch.as_tensor(x0, dtype=torch.float32, device=device)
    n0 = int(x0_t.shape[0])
    n_fates = int(getattr(model, "n_fates", 1))
    times = [float(t) for t in (context.trajectory_time_points or context.time_points)]
    if len(times) < 2:
        times = [float(context.time_points[0]), float(context.time_points[-1])]
    integration_dt = float(((context.config.get("evaluation") or {}).get("trajectory_integration_dt") or 0.01))
    integration_dt = max(integration_dt, 1e-4)
    eval_cfg = context.config.get("evaluation") or {}
    coreset_budget = int(eval_cfg.get("branch_metric_coreset_particles", 12000))
    coreset_diag: dict[str, float] = {}

    with torch.no_grad():
        t0_val = torch.full((n0, 1), times[0], dtype=x0_t.dtype, device=device)
        q = torch.softmax(model.gate_net(torch.cat([x0_t, t0_val], dim=1)), dim=1)
        branch_eye = torch.eye(n_fates, dtype=x0_t.dtype, device=device)
        x = x0_t.repeat_interleave(n_fates, dim=0).contiguous()
        branch_onehot = branch_eye.repeat(n0, 1).contiguous()
        parent = np.repeat(np.arange(n0, dtype=np.int64), n_fates)
        branch_codes = np.tile(np.arange(n_fates, dtype=np.int64), n0)
        weights0 = (q.reshape(-1) / max(float(n0), 1.0)).clamp_min(1e-12)
        logw = torch.log(weights0).reshape(-1, 1)

        out_points: list[np.ndarray] = []
        out_weights: list[np.ndarray] = []
        branch_final_points = None
        branch_final_weights = None
        current_t = float(times[0])

        def record() -> None:
            # Package W1/TMV and CFCorr must see the same sealed generated
            # measure: the weighted fate-branch particle cloud.  To keep the
            # package dense W1 matrix bounded, metrics consume a deterministic
            # weighted coreset of that full branch measure, never a parent
            # barycenter collapse or target-informed repair.
            pts = x.detach().cpu().numpy().astype(np.float32, copy=True)
            ww = torch.exp(logw).detach().cpu().numpy().reshape(-1).astype(np.float32, copy=True)
            pts, ww, diag = _weighted_systematic_coreset(pts, ww, budget=coreset_budget)
            coreset_diag.update(diag)
            out_points.append(pts)
            out_weights.append(ww)

        record()
        record_index = 0
        while record_index < len(times) - 1:
            target_t = float(times[record_index + 1])
            while current_t < target_t - 1e-9:
                dt = min(integration_dt, target_t - current_t)
                t_col = torch.full((x.shape[0], 1), current_t, dtype=x.dtype, device=device)
                inp = torch.cat([x, t_col, branch_onehot], dim=1)
                dx = model.branch_velocity_net(inp)
                dg = model.branch_growth_net(inp)
                x = x + dx * dt
                logw = logw + dg * dt
                current_t += dt
            record()
            record_index += 1
        branch_final_points = x.detach().cpu().numpy().astype(np.float32, copy=True)
        branch_final_weights = torch.exp(logw).detach().cpu().numpy().reshape(-1).astype(np.float32, copy=True)

    return SimulationResult(
        predicted_points_by_time=out_points,
        predicted_weights_by_time=out_weights,
        trajectory_time_points=times,
        artifacts={
            "branch_parent_indices": parent,
            "branch_codes": branch_codes,
            "branch_final_points": branch_final_points,
            "branch_final_weights": branch_final_weights,
            "n_source_cells": int(n0),
            "n_fates": int(n_fates),
            "simulation_contract": "sealed_t0_branch_weighted_rollout_no_future_rows",
            **coreset_diag,
        },
    )


# ---------------------------------------------------------------------------
# Rollout-derived Clone Fate Correlation metric
# ---------------------------------------------------------------------------


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _weighted_pearson(x: np.ndarray, y: np.ndarray, w: np.ndarray, eps: float = 1e-12) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    if int(mask.sum()) < 3:
        return float("nan")
    x = x[mask]
    y = y[mask]
    w = w[mask]
    sw = float(w.sum())
    if sw <= eps:
        return float("nan")
    w = w / sw
    mx = float(np.sum(w * x))
    my = float(np.sum(w * y))
    vx = float(np.sum(w * (x - mx) ** 2))
    vy = float(np.sum(w * (y - my) ** 2))
    if vx <= eps or vy <= eps:
        return float("nan")
    cov = float(np.sum(w * (x - mx) * (y - my)))
    return cov / float(np.sqrt(vx * vy))


def _weighted_systematic_coreset(points: np.ndarray, weights: np.ndarray, *, budget: int) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Deterministic weighted-particle coreset of a sealed generated measure.

    Builtin W1 computes a dense observed-by-predicted cost matrix, so emitting all
    n0*|F| branch particles on Weinreb can make the standard metric path exceed
    the preview/campaign inference budget.  This routine approximates the same
    full branch measure by systematic quantiles of its normalized cumulative mass
    and preserves total mass exactly through equal coreset weights.  It is not a
    parent barycenter collapse, does not use target cells/counts, and is used only
    to keep package metrics on the sealed branch rollout computationally bounded.
    """
    pts = np.asarray(points, dtype=np.float32)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    n = int(w.shape[0])
    budget = int(max(1, budget))
    if n <= budget:
        return pts, w.astype(np.float32), {"metric_coreset_used": 0.0, "metric_coreset_particles": float(n), "metric_coreset_source_particles": float(n)}
    total = float(w.sum())
    if not np.isfinite(total) or total <= 0.0:
        return pts[:budget], np.ones(budget, dtype=np.float32) / float(budget), {"metric_coreset_used": 1.0, "metric_coreset_particles": float(budget), "metric_coreset_source_particles": float(n)}
    order = np.argsort(np.arange(n), kind="stable")
    cdf = np.cumsum(w[order]) / total
    # Midpoint systematic quantiles avoid random state and are reproducible.
    qs = (np.arange(budget, dtype=np.float64) + 0.5) / float(budget)
    chosen = order[np.searchsorted(cdf, qs, side="left").clip(0, n - 1)]
    out_w = np.full(budget, total / float(budget), dtype=np.float32)
    return pts[chosen].astype(np.float32, copy=False), out_w, {"metric_coreset_used": 1.0, "metric_coreset_particles": float(budget), "metric_coreset_source_particles": float(n)}


def _knn_fate_probabilities(query: np.ndarray, reference: np.ndarray, ref_label_codes: np.ndarray, n_fates: int, *, k: int, chunk_size: int, eps: float = 1e-6) -> np.ndarray:
    query = np.asarray(query, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    labels = np.asarray(ref_label_codes, dtype=np.int64).reshape(-1)
    k = max(1, min(int(k), reference.shape[0]))
    out = np.zeros((query.shape[0], n_fates), dtype=np.float32)
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(reference)
        dsel, idx = tree.query(query, k=k)
        if k == 1:
            dsel = dsel[:, None]
            idx = idx[:, None]
        ww = 1.0 / (np.asarray(dsel, dtype=np.float32) + eps)
        for row in range(idx.shape[0]):
            codes = labels[np.asarray(idx[row], dtype=np.int64)]
            binc = np.bincount(codes, weights=ww[row], minlength=n_fates).astype(np.float32)
            total = float(binc.sum())
            if total > 0:
                out[row] = binc / total
        return out
    except Exception:
        pass
    ref_norm = np.sum(reference * reference, axis=1)[None, :]
    for start in range(0, query.shape[0], int(chunk_size)):
        stop = min(start + int(chunk_size), query.shape[0])
        q = query[start:stop]
        q_norm = np.sum(q * q, axis=1)[:, None]
        dist = np.maximum(q_norm + ref_norm - 2.0 * q @ reference.T, 0.0)
        idx = np.argpartition(dist, kth=k - 1, axis=1)[:, :k]
        dsel = np.take_along_axis(dist, idx, axis=1)
        ww = 1.0 / (np.sqrt(dsel) + eps)
        for row in range(idx.shape[0]):
            codes = labels[idx[row]]
            binc = np.bincount(codes, weights=ww[row], minlength=n_fates).astype(np.float32)
            total = float(binc.sum())
            if total > 0:
                out[start + row] = binc / total
    return out


def _compute_clone_fate_metric_from_context(context: EvaluationMetricsContext) -> dict[str, Any]:
    adata = context.adata
    params = dict(getattr(context, "metric_params", {}) or {})
    clone_key = str(params.get("clone_key", "clone"))
    label_key = str(params.get("label_key", "Cell type annotation"))
    time_key = str(params.get("time_key", "time_point_processed"))
    latent_key = str(params.get("latent_key", "X_latent"))
    alpha = float(params.get("dirichlet_alpha", 5.0))
    k_neighbors = int(params.get("terminal_knn", 30))
    chunk_size = int(params.get("knn_chunk_size", 512))
    min_terminal_clone_cells = int(params.get("min_terminal_clone_cells", 1))
    exclude_labels = {str(x) for x in params.get("exclude_labels", [])}

    missing_obs = [k for k in {clone_key, label_key, time_key} if k not in adata.obs]
    if missing_obs or latent_key not in adata.obsm:
        return {"CFCorr": float("nan"), "CFCorr_warning": f"missing fields obs={missing_obs}, latent={latent_key not in adata.obsm}"}
    trajectory_points = list(getattr(context, "trajectory_points_by_time", []) or [])
    if len(trajectory_points) < 2:
        return {"CFCorr": float("nan"), "CFCorr_warning": "missing evaluation trajectory"}
    terminal_pred = _as_numpy(trajectory_points[-1]).astype(np.float32)
    terminal_weights = _as_numpy((getattr(context, "trajectory_weights_by_time", []) or [None])[-1]).reshape(-1).astype(np.float64)
    artifacts = dict(getattr(getattr(context, "evaluation_trajectory", None), "artifacts", {}) or {})
    if artifacts.get("branch_final_points") is not None and artifacts.get("branch_final_weights") is not None:
        terminal_pred = np.asarray(artifacts.get("branch_final_points"), dtype=np.float32)
        terminal_weights = np.asarray(artifacts.get("branch_final_weights"), dtype=np.float64).reshape(-1)

    t0 = float(context.training_data.time_points[0])
    t_final = float(context.training_data.time_points[-1])
    t0_indices = np.asarray(context.training_data.obs_indices_by_time[0], dtype=int)
    final_indices = np.asarray(context.training_data.obs_indices_by_time[-1], dtype=int)
    n0 = int(t0_indices.shape[0])

    final_labels_raw = np.asarray(adata.obs.iloc[final_indices][label_key].astype(str))
    final_keep = np.asarray([lab not in exclude_labels for lab in final_labels_raw], dtype=bool)
    final_indices = final_indices[final_keep]
    final_labels_raw = final_labels_raw[final_keep]
    if final_indices.shape[0] == 0:
        return {"CFCorr": float("nan"), "CFCorr_warning": "no terminal cells after label filtering"}
    fate_labels = np.asarray(sorted(np.unique(final_labels_raw).tolist()), dtype=object)
    fate_to_code = {str(f): i for i, f in enumerate(fate_labels)}
    final_codes = np.asarray([fate_to_code[str(f)] for f in final_labels_raw], dtype=np.int64)
    n_fates = int(len(fate_labels))
    global_counts = np.bincount(final_codes, minlength=n_fates).astype(np.float64)
    global_prior = global_counts / max(float(global_counts.sum()), 1.0)
    final_latent = np.asarray(adata.obsm[latent_key][final_indices], dtype=np.float32)

    endpoint_fate = _knn_fate_probabilities(terminal_pred, final_latent, final_codes, n_fates, k=k_neighbors, chunk_size=chunk_size)
    parent = artifacts.get("branch_parent_indices")
    if parent is not None and terminal_pred.shape[0] != n0:
        parent = np.asarray(parent, dtype=np.int64).reshape(-1)
        if parent.shape[0] == terminal_pred.shape[0] and terminal_weights.shape[0] == terminal_pred.shape[0]:
            pred_fate = np.zeros((n0, n_fates), dtype=np.float64)
            denom = np.zeros(n0, dtype=np.float64)
            for start in range(0, terminal_pred.shape[0], max(1024, chunk_size)):
                stop = min(start + max(1024, chunk_size), terminal_pred.shape[0])
                p = parent[start:stop]
                ww = terminal_weights[start:stop]
                np.add.at(pred_fate, p, endpoint_fate[start:stop] * ww[:, None])
                np.add.at(denom, p, ww)
            pred_fate = pred_fate / np.maximum(denom[:, None], 1e-12)
        else:
            return {"CFCorr": float("nan"), "CFCorr_warning": "branch artifact shape mismatch"}
    else:
        if terminal_pred.shape[0] < n0:
            n0 = int(terminal_pred.shape[0])
            t0_indices = t0_indices[:n0]
        elif terminal_pred.shape[0] > n0:
            endpoint_fate = endpoint_fate[:n0]
        pred_fate = endpoint_fate.astype(np.float64)

    final_clones = np.asarray([_clone_key(v) for v in adata.obs.iloc[final_indices][clone_key].to_numpy()], dtype=object)
    t0_clones = np.asarray([_clone_key(v) for v in adata.obs.iloc[t0_indices][clone_key].to_numpy()], dtype=object)
    clone_truth: dict[str, np.ndarray] = {}
    clone_terminal_n: dict[str, int] = {}
    for c in sorted({str(x) for x in final_clones if x}):
        mask = final_clones == c
        n_c = int(mask.sum())
        if n_c < min_terminal_clone_cells:
            continue
        counts = np.bincount(final_codes[mask], minlength=n_fates).astype(np.float64)
        truth = (counts + alpha * global_prior) / (float(n_c) + alpha)
        clone_truth[c] = truth.astype(np.float64)
        clone_terminal_n[c] = n_c

    eligible = np.asarray([c in clone_truth for c in t0_clones], dtype=bool)
    if int(eligible.sum()) < 3:
        return {"CFCorr": float("nan"), "CFCorr_warning": "fewer than three eligible t0 cells", "CFCorr_n_eligible_cells": int(eligible.sum())}
    truth_mat = np.zeros((int(eligible.sum()), n_fates), dtype=np.float64)
    pred_mat = pred_fate[eligible].astype(np.float64)
    eligible_clones = t0_clones[eligible]
    clone_t0_counts = _counts_by_clone(eligible_clones)
    weights = np.zeros(int(eligible.sum()), dtype=np.float64)
    for row, c in enumerate(eligible_clones):
        truth_mat[row] = clone_truth[str(c)]
        n_term = float(clone_terminal_n[str(c)])
        reliability = n_term / (n_term + alpha)
        weights[row] = reliability / max(float(clone_t0_counts.get(str(c), 1)), 1.0)

    per_fate_corr: dict[str, float] = {}
    valid_corrs: list[float] = []
    for f_idx, fate in enumerate(fate_labels):
        corr = _weighted_pearson(pred_mat[:, f_idx], truth_mat[:, f_idx], weights)
        per_fate_corr[str(fate)] = float(corr) if np.isfinite(corr) else float("nan")
        if np.isfinite(corr):
            valid_corrs.append(float(corr))
    cfcorr = float(np.mean(valid_corrs)) if valid_corrs else float("nan")
    brier = float(np.average(np.sum((pred_mat - truth_mat) ** 2, axis=1), weights=weights))
    pred_entropy = float(np.average(-np.sum(pred_mat * np.log(np.maximum(pred_mat, 1e-12)), axis=1), weights=weights))
    truth_entropy = float(np.average(-np.sum(truth_mat * np.log(np.maximum(truth_mat, 1e-12)), axis=1), weights=weights))
    return {
        "CFCorr": cfcorr,
        "CFCorr_macro_pearson": cfcorr,
        "CFCorr_brier": brier,
        "CFCorr_prediction_entropy": pred_entropy,
        "CFCorr_truth_entropy": truth_entropy,
        "CFCorr_n_valid_fates": int(len(valid_corrs)),
        "CFCorr_n_fates": n_fates,
        "CFCorr_per_fate": per_fate_corr,
        "CFCorr_n_eligible_cells": int(eligible.sum()),
        "CFCorr_n_eligible_clones": int(len(set(map(str, eligible_clones)))),
        "CFCorr_n_terminal_cells": int(final_indices.shape[0]),
        "CFCorr_time0": t0,
        "CFCorr_time_final": t_final,
        "CFCorr_prediction_source_code": 2.0 if parent is not None else 1.0,
        "CFCorr_metric_contract_code": 2.0,
    }


def evaluate_claim_metric(context: EvaluationMetricsContext) -> dict[str, Any]:
    return _compute_clone_fate_metric_from_context(context)


# ---------------------------------------------------------------------------
# Training-data builder
# ---------------------------------------------------------------------------


def _build_clone_sorted_training_data(ctx: TrainingDataBuilderContext) -> TrainingDataBundle:
    adata = ctx.adata
    if "time_point_processed" not in adata.obs:
        raise ValueError("adata.obs['time_point_processed'] is required.")
    if "X_latent" not in adata.obsm:
        raise ValueError("adata.obsm['X_latent'] is required.")
    if "clone" not in adata.obs:
        raise ValueError("Clone-CUVGF-BMF requires obs['clone'].")
    device = torch.device(ctx.device or "cpu")
    time_values = np.asarray(adata.obs["time_point_processed"].astype(float))
    time_points = sorted(float(t) for t in np.unique(time_values))
    latent_by_time: list[torch.Tensor] = []
    obs_indices_by_time: list[np.ndarray] = []
    for t in time_points:
        idx = np.where(time_values == float(t))[0]
        clones = np.asarray([_clone_key(v) for v in adata.obs.iloc[idx]["clone"].to_numpy()], dtype=object)
        sort_key = np.asarray([c if c else "~~~~~~~~" for c in clones], dtype=object)
        order = np.lexsort((idx, sort_key))
        idx = idx[order].astype(np.int64, copy=False)
        latent = np.asarray(adata.obsm["X_latent"][idx], dtype=np.float32)
        latent_by_time.append(torch.tensor(latent, dtype=torch.float32, device=device))
        obs_indices_by_time.append(idx)

    clone_by_time = _extract_clone_arrays(adata, obs_indices_by_time)
    training_cfg = (ctx.resolved_config.get("training") or {}) if isinstance(ctx.resolved_config, dict) else {}
    plan0 = (training_cfg.get("plan") or [{}])[0] if isinstance(training_cfg.get("plan"), list) and training_cfg.get("plan") else {}
    coupling_cfg = (((plan0.get("flow_matching") or {}).get("coupling")) or {}) if isinstance(plan0, dict) else {}
    fate_by_time, omega_by_time, fate_metadata = _build_fate_annotations(
        adata,
        obs_indices_by_time,
        clone_by_time,
        label_key=str(coupling_cfg.get("label_key", "Cell type annotation")),
        alpha=float(coupling_cfg.get("dirichlet_alpha", 5.0)),
        eta=float(coupling_cfg.get("terminal_onehot_eta", 0.05)),
    )
    return TrainingDataBundle(
        adata=adata,
        time_points=time_points,
        latent_by_time=latent_by_time,
        obs_indices_by_time=obs_indices_by_time,
        extra_modalities_by_time={
            "bmf_fate_by_time": fate_by_time,
            "bmf_fate_reliability_by_time": omega_by_time,
        },
        metadata={
            "time_key": "time_point_processed",
            "latent_key": "X_latent",
            "row_order": "clone_sorted_with_missing_last",
            "bmf_n_fates": int(fate_metadata.get("n_fates", 0)),
            "bmf_fate_metadata": fate_metadata,
            **dict(ctx.metadata or {}),
        },
    )


# ---------------------------------------------------------------------------
# Training algorithm entry point
# ---------------------------------------------------------------------------


def build_training_algorithm(context):
    def build_flow_matching_backend(build_context: FlowMatchingBuildContext) -> FlowMatchingBackend:
        stage_params = build_context.stage_params
        device = build_context.device
        training_data = build_context.training_data
        coupling_config = (stage_params.get("flow_matching") or {}).get("coupling") or {}
        branch_config = (stage_params.get("flow_matching") or {}).get("branch") or {}
        clone_by_time = _extract_clone_arrays(training_data.adata, training_data.obs_indices_by_time)
        reliability_by_interval = _build_interval_reliability(
            clone_by_time,
            pseudocount=float(coupling_config.get("reliability_pseudocount", stage_params.get("reliability_pseudocount", 2.0))),
        )
        fate_by_time = list((training_data.extra_modalities_by_time or {}).get("bmf_fate_by_time") or [])
        fate_reliability_by_time = list((training_data.extra_modalities_by_time or {}).get("bmf_fate_reliability_by_time") or [])
        fate_metadata = dict((training_data.metadata or {}).get("bmf_fate_metadata") or {})
        if not fate_by_time:
            fate_by_time, fate_reliability_by_time, fate_metadata = _build_fate_annotations(
                training_data.adata,
                training_data.obs_indices_by_time,
                clone_by_time,
                label_key=str(coupling_config.get("label_key", stage_params.get("label_key", "Cell type annotation"))),
                alpha=float(coupling_config.get("dirichlet_alpha", stage_params.get("dirichlet_alpha", 5.0))),
                eta=float(coupling_config.get("terminal_onehot_eta", stage_params.get("terminal_onehot_eta", 0.05))),
            )

        stable = CloneCensoredUOTCoupling(
            clone_by_time=clone_by_time,
            reliability_by_interval=reliability_by_interval,
            chunk_size=int(coupling_config.get("chunk_size", stage_params.get("chunk_size", 1000))),
            alpha_regm=float(coupling_config.get("alpha_regm", stage_params.get("alpha_regm", 1.0))),
            reg_strategy=str(coupling_config.get("reg_strategy", stage_params.get("reg_strategy", "max_over_time"))),
            reg=coupling_config.get("reg", stage_params.get("reg")),
            reg_m=coupling_config.get("reg_m", stage_params.get("reg_m")),
            auto_reg_device=str(coupling_config.get("auto_reg_device", stage_params.get("auto_reg_device", "cpu"))),
            clone_lambda=float(coupling_config.get("clone_lambda", stage_params.get("clone_lambda", 0.15))),
            reliability_min=float(coupling_config.get("reliability_min", stage_params.get("reliability_min", 0.10))),
            same_clone_discount=float(coupling_config.get("same_clone_discount", stage_params.get("same_clone_discount", 0.75))),
            mismatch_penalty=float(coupling_config.get("mismatch_penalty", stage_params.get("mismatch_penalty", 0.25))),
            regularization_block_policy=str(coupling_config.get("regularization_block_policy", "first_block_per_time")),
        )
        coupling = CloneBMFBranchCoupling(
            stable_coupling=stable,
            fate_by_time=fate_by_time,
            fate_reliability_by_time=fate_reliability_by_time,
            fate_metadata=fate_metadata,
            max_edges_per_row=int(branch_config.get("max_edges_per_row", coupling_config.get("max_edges_per_row", 32))),
            branch_ipf_iter=int(branch_config.get("branch_ipf_iter", coupling_config.get("branch_ipf_iter", 12))),
            branch_eps=float(branch_config.get("branch_eps", coupling_config.get("branch_eps", 1e-9))),
            source_prior_power=float(branch_config.get("source_prior_power", coupling_config.get("source_prior_power", 0.5))),
            mid_prior_power=float(branch_config.get("mid_prior_power", coupling_config.get("mid_prior_power", 0.5))),
            terminal_prior_power=float(branch_config.get("terminal_prior_power", coupling_config.get("terminal_prior_power", 0.5))),
        )
        print("Clone-CUVGF-BMF stable clone reliability summary:")
        for stat in stable._stats:
            print(f"  interval={stat.interval} cells={stat.source_cells}->{stat.target_cells} shared_reliable_clones={stat.shared_reliable_clones} mean_reliability={stat.mean_reliability:.3f}")
        print(
            "Clone-CUVGF-BMF fate branch summary: "
            f"n_fates={fate_metadata.get('n_fates')} terminal_supported_clones={fate_metadata.get('n_terminal_supported_clones')} "
            f"lambda_gate={branch_config.get('lambda_gate', stage_params.get('lambda_gate', 1.0))}"
        )
        path_config = (stage_params.get("flow_matching") or {}).get("path") or {}
        sigma = float(path_config.get("sigma", stage_params.get("sigma", 0.0)))
        path = LinearDeterministicConditionalPath(sigma=0.0) if sigma <= 0.0 else RegularizedUnbalancedConditionalPath(sigma=sigma)
        return BranchFlowMatchingBackend(path=path, coupling=coupling, mass=UOTMassStrategy())

    return TrainingAlgorithmSpec(
        algorithm_id=context.algorithm_id,
        base_config="./config.yaml",
        config_overrides={},
        training_data_builder=_build_clone_sorted_training_data,
        flow_matching_backend_builder=build_flow_matching_backend,
        flow_matching_loss_hook=_branch_loss_hook,
        model_builder=_build_branch_model,
        serialize_model=None,
        deserialize_model=None,
        stage_runner=None,
        inference_context_builder=_build_inference_context,
        simulation_hook=_branch_simulation_hook,
        evaluation_metrics_hook=evaluate_claim_metric,
        evaluation_metrics_params={
            "metric_name": "CFCorr",
            "clone_key": "clone",
            "label_key": "Cell type annotation",
            "time_key": "time_point_processed",
            "latent_key": "X_latent",
            "dirichlet_alpha": 5.0,
            "terminal_knn": 30,
            "knn_chunk_size": 512,
            "min_terminal_clone_cells": 1,
            "exclude_labels": [],
            "expected_ranges": {"CFCorr": [-1.0, 1.0], "CFCorr_brier": [0.0, 2.0]},
        },
        notes=(
            "Clone-CUVGF-BMF implements an approved branch-mass fate lift: stable clone-censored UOT fixes total row masses; "
            "a sparse branch projection trains branch-conditioned velocity/growth and a feasible gate; sealed branch-weighted rollout supplies W1/TMV/CFCorr."
        ),
    )
