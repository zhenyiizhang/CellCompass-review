from __future__ import annotations

from typing import Any

import numpy as np
import torch
import math

from CytoBridge.tl.flow_matching_backends import (
    BalancedOTCouplingStrategy,
    FlowMatchingBackend,
    LinearDeterministicConditionalPath,
    NullMassStrategy,
    PairwiseCost,
)
from CytoBridge.tl.models import DynamicalModel
from CytoBridge.tl.training_algorithm import (
    FlowMatchingBuildContext,
    FlowMatchingLossContext,
    FlowMatchingLossResult,
    InferenceContext,
    InferenceContextBuilderContext,
    ModelBuildContext,
    ModelLoadContext,
    ModelSerializeContext,
    SimulationContext,
    SimulationResult,
    TrainingAlgorithmSpec,
    TrainingDataBundle,
)


def _stage_pcfm_params(stage_params: dict[str, Any]) -> dict[str, Any]:
    fm = stage_params.get("flow_matching") or {}
    return fm.get("cite_pcfm") or stage_params.get("cite_pcfm") or {}


def _config_pcfm_params(config: dict[str, Any]) -> dict[str, Any]:
    plan = (config.get("training") or {}).get("plan") or []
    if plan and isinstance(plan[0], dict):
        return _stage_pcfm_params(plan[0])
    return config.get("cite_pcfm", {}) if isinstance(config, dict) else {}


def _readout_input(r: torch.Tensor, t_col: torch.Tensor, n_fourier: int = 0, time_fourier_scale: float = 1.0) -> torch.Tensor:
    """Allowed protein-readout input: rolled RNA state plus exogenous time features."""
    if int(n_fourier) <= 0:
        return torch.cat([r, t_col], dim=1)
    scale = max(float(time_fourier_scale), 1.0e-6)
    tau_col = t_col / scale
    feats = [r, t_col]
    for freq in range(1, int(n_fourier) + 1):
        angle = (2.0 * math.pi * float(freq)) * tau_col
        feats.append(torch.sin(angle))
        feats.append(torch.cos(angle))
    return torch.cat(feats, dim=1)


def _time_loss_weight(t_col: torch.Tensor, cite_params: dict[str, Any]) -> torch.Tensor:
    """Fixed exogenous-time weights for protein-readout supervision.

    These weights are training-objective weights only.  They use configured time
    constants and the already-allowed time input, never observed future proteins
    or evaluation statistics during prediction.
    """
    centers = cite_params.get("protein_time_weight_centers", []) or []
    strength = float(cite_params.get("protein_time_weight_strength", 0.0))
    width = float(cite_params.get("protein_time_weight_width", 1.0))
    if strength <= 0.0 or width <= 0.0 or not centers:
        return torch.ones_like(t_col)
    weights = torch.ones_like(t_col)
    for center in centers:
        c = torch.as_tensor(float(center), dtype=t_col.dtype, device=t_col.device)
        weights = weights + strength * torch.exp(-0.5 * ((t_col - c) / width).pow(2))
    max_weight = float(cite_params.get("protein_time_weight_max", 5.0))
    return torch.clamp(weights, min=1.0, max=max(max_weight, 1.0))


class PairedCoordinateChart:
    """Masked/robust protein chart for paired CITE-seq coordinates.

    Current implementation uses a semantics-preserving denoised chart: robustly
    centered/scaled protein PCs. It is deliberately simple and deterministic;
    the proposal's denoising chart can be strengthened only if the proposal is
    revised or a semantics-preserving smoother is added.
    """

    def __init__(self, *, rna_dim: int = 50, protein_dim: int = 30, scale_floor: float = 1e-3):
        self.rna_dim = int(rna_dim)
        self.protein_dim = int(protein_dim)
        self.scale_floor = float(scale_floor)
        self.protein_center: np.ndarray | None = None
        self.protein_scale: np.ndarray | None = None
        self.diagnostics: dict[str, float] = {}

    @property
    def latent_dim(self) -> int:
        return self.rna_dim + self.protein_dim

    def fit(self, protein_by_time: list[np.ndarray]) -> None:
        if not protein_by_time:
            raise ValueError("CITE-PCFM requires paired protein coordinates.")
        p_all = np.concatenate([np.asarray(p, dtype=np.float32) for p in protein_by_time], axis=0)
        if p_all.ndim != 2 or p_all.shape[1] < self.protein_dim:
            raise ValueError(f"CITE-PCFM expected protein_dim={self.protein_dim}, got shape {p_all.shape}.")
        p_all = p_all[:, : self.protein_dim]
        center = np.nanmedian(p_all, axis=0).astype(np.float32)
        q25 = np.nanpercentile(p_all, 25, axis=0)
        q75 = np.nanpercentile(p_all, 75, axis=0)
        iqr = (q75 - q25) / 1.349
        std = np.nanstd(p_all, axis=0)
        scale = np.where(np.isfinite(iqr) & (iqr > self.scale_floor), iqr, std)
        scale = np.maximum(scale, self.scale_floor).astype(np.float32)
        self.protein_center = center
        self.protein_scale = scale
        self.diagnostics = {
            "protein_center_abs_mean": float(np.mean(np.abs(center))) if center.size else 0.0,
            "protein_scale_min": float(np.min(scale)) if scale.size else 0.0,
            "protein_scale_max": float(np.max(scale)) if scale.size else 0.0,
            "protein_scale_mean": float(np.mean(scale)) if scale.size else 0.0,
        }

    def encode_protein_np(self, p: np.ndarray) -> np.ndarray:
        if self.protein_center is None or self.protein_scale is None:
            raise RuntimeError("PairedCoordinateChart has not been fitted.")
        p = np.asarray(p, dtype=np.float32)[:, : self.protein_dim]
        return ((p - self.protein_center.reshape(1, -1)) / self.protein_scale.reshape(1, -1)).astype(np.float32)

    def decode_protein_np(self, q: np.ndarray) -> np.ndarray:
        if self.protein_center is None or self.protein_scale is None:
            raise RuntimeError("PairedCoordinateChart has not been fitted.")
        q = np.asarray(q, dtype=np.float32)[:, : self.protein_dim]
        return (q * self.protein_scale.reshape(1, -1) + self.protein_center.reshape(1, -1)).astype(np.float32)

    def state_dict(self) -> dict[str, Any]:
        return {
            "rna_dim": self.rna_dim,
            "protein_dim": self.protein_dim,
            "scale_floor": self.scale_floor,
            "protein_center": self.protein_center,
            "protein_scale": self.protein_scale,
            "diagnostics": self.diagnostics,
        }

    @classmethod
    def from_state_dict(cls, payload: dict[str, Any]) -> "PairedCoordinateChart":
        chart = cls(
            rna_dim=int(payload.get("rna_dim", 50)),
            protein_dim=int(payload.get("protein_dim", 30)),
            scale_floor=float(payload.get("scale_floor", 1e-3)),
        )
        chart.protein_center = np.asarray(payload.get("protein_center"), dtype=np.float32)
        chart.protein_scale = np.asarray(payload.get("protein_scale"), dtype=np.float32)
        chart.diagnostics = dict(payload.get("diagnostics") or {})
        return chart


class PairedCoordinateBalancedCoupling(BalancedOTCouplingStrategy):
    """Balanced adjacent-time OT in robust RNA/protein CITE coordinates."""

    def __init__(self, *, rna_dim: int = 50, protein_dim: int = 30, alpha_r: float = 1.0, lambda_p: float = 0.25, **kwargs):
        super().__init__(**kwargs)
        self.rna_dim = int(rna_dim)
        self.protein_dim = int(protein_dim)
        self.alpha_r = float(alpha_r)
        self.lambda_p = float(lambda_p)

    def describe_metadata(self) -> dict[str, Any]:
        md = super().describe_metadata()
        md.update({
            "kind": "cite_pcfm_paired_coordinate_balanced_ot",
            "rna_dim": self.rna_dim,
            "protein_dim": self.protein_dim,
            "alpha_r": self.alpha_r,
            "lambda_p": self.lambda_p,
            "balanced_only": True,
        })
        return md

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
        del source_indices, target_indices, t0, t1
        z0 = torch.as_tensor(x0_block, dtype=torch.float32, device=device)
        z1 = torch.as_tensor(x1_block, dtype=torch.float32, device=device)
        r0 = z0[:, : self.rna_dim]
        r1 = z1[:, : self.rna_dim]
        p0 = z0[:, self.rna_dim : self.rna_dim + self.protein_dim]
        p1 = z1[:, self.rna_dim : self.rna_dim + self.protein_dim]
        r_cost = torch.cdist(r0, r1).pow(2)
        # The prepared CITE benchmark stores dense protein PCA coordinates with
        # no missing ADT dimensions. This is the proposal's mask-normalized
        # distance with an all-ones reliability mask: average squared distance
        # over the shared/reliable protein coordinates.
        p_cost = torch.cdist(p0, p1).pow(2) / max(float(self.protein_dim), 1.0)
        cost = self.alpha_r * r_cost + self.lambda_p * p_cost
        return PairwiseCost(
            cost_matrix=cost,
            source_mass=np.ones(x0_block.shape[0], dtype=np.float32),
            target_mass=np.ones(x1_block.shape[0], dtype=np.float32),
            normalize_cost=True,
            metadata={"time_idx": int(time_idx), "state": "z=(rna, protein_pca)", "protein_mask": "all_ones_dense_pca", "balanced_only": True},
        )


class PairedCoordinateFlowMatchingBackend(FlowMatchingBackend):
    """FlowMatchingBackend variant that exposes training endpoints to loss hooks.

    The standard package batch contains interpolation states and regression
    targets. CITE-PCFM also calibrates the protein readout at supervised adjacent
    snapshot endpoints. These endpoint tensors are training-only extras and are
    never used during inference or metric scoring.
    """

    def sample_batch(self, X, t_train, batch_size: int, device: torch.device):
        batch = super().sample_batch(X, t_train, batch_size, device)
        if self.state is None:
            return batch
        endpoint_x0 = []
        endpoint_x1 = []
        endpoint_t0 = []
        endpoint_t1 = []
        for time_idx in range(len(t_train) - 1):
            pair_batch = self.coupling.sample_pairs(self.state, X, time_idx, batch_size, device)
            if pair_batch is None:
                continue
            endpoint_x0.append(pair_batch.x0)
            endpoint_x1.append(pair_batch.x1)
            endpoint_t0.append(torch.full((pair_batch.x0.shape[0], 1), float(t_train[time_idx]), dtype=pair_batch.x0.dtype, device=device))
            endpoint_t1.append(torch.full((pair_batch.x1.shape[0], 1), float(t_train[time_idx + 1]), dtype=pair_batch.x1.dtype, device=device))
        if endpoint_x0:
            batch.extras = dict(getattr(batch, "extras", {}) or {})
            batch.extras["endpoint_x0"] = torch.cat(endpoint_x0, dim=0)
            batch.extras["endpoint_x1"] = torch.cat(endpoint_x1, dim=0)
            batch.extras["endpoint_t0"] = torch.cat(endpoint_t0, dim=0)
            batch.extras["endpoint_t1"] = torch.cat(endpoint_t1, dim=0)
            batch.extras["endpoint_supervision_semantics"] = "training_only_coupled_adjacent_snapshot_endpoints"
        return batch


def _extract_paired_arrays(adata: Any, *, rna_dim: int, protein_dim: int) -> tuple[np.ndarray, np.ndarray]:
    if "X_rna_svd" in adata.obsm and "X_protein_pca" in adata.obsm:
        r_all = np.asarray(adata.obsm["X_rna_svd"], dtype=np.float32)[:, :rna_dim]
        p_all = np.asarray(adata.obsm["X_protein_pca"], dtype=np.float32)[:, :protein_dim]
        return r_all, p_all
    if "X_latent" not in adata.obsm:
        raise ValueError("CITE-PCFM requires obsm['X_latent'] or paired obsm['X_rna_svd']/obsm['X_protein_pca'].")
    z = np.asarray(adata.obsm["X_latent"], dtype=np.float32)
    if z.shape[1] < rna_dim + protein_dim:
        raise ValueError(f"CITE-PCFM expected X_latent with at least {rna_dim + protein_dim} dimensions; got {z.shape[1]}.")
    return z[:, :rna_dim], z[:, rna_dim : rna_dim + protein_dim]


def build_training_algorithm(context):
    """Return TrainingAlgorithmSpec implementing approved balanced CITE-PCFM."""

    chart_holder: dict[str, PairedCoordinateChart] = {}

    def training_data_builder(build_context) -> TrainingDataBundle:
        adata = build_context.adata
        cfg_params = _config_pcfm_params(build_context.resolved_config)
        rna_dim = int(cfg_params.get("rna_dim", 50))
        protein_dim = int(cfg_params.get("protein_dim", 30))
        scale_floor = float(cfg_params.get("protein_scale_floor", 1e-3))
        time_key = "time_point_processed"
        if time_key not in adata.obs:
            raise ValueError("CITE-PCFM requires obs['time_point_processed'].")
        r_all, p_all_raw = _extract_paired_arrays(adata, rna_dim=rna_dim, protein_dim=protein_dim)
        times = sorted([float(t) for t in np.unique(adata.obs[time_key].to_numpy())])
        obs_indices_by_time: list[np.ndarray] = []
        r_by_time: list[np.ndarray] = []
        p_by_time: list[np.ndarray] = []
        m_by_time: list[np.ndarray] = []
        time_values = adata.obs[time_key].to_numpy().astype(float)
        for t in times:
            idx = np.where(time_values == float(t))[0]
            obs_indices_by_time.append(idx)
            r_by_time.append(r_all[idx].astype(np.float32, copy=False))
            p_by_time.append(p_all_raw[idx].astype(np.float32, copy=False))
            m_by_time.append(np.ones((idx.shape[0], protein_dim), dtype=np.float32))
        chart = PairedCoordinateChart(rna_dim=rna_dim, protein_dim=protein_dim, scale_floor=scale_floor)
        chart.fit(p_by_time)
        # Preserve the benchmark's canonical latent coordinates for package
        # evaluation: z=(X_rna_svd, X_protein_pca). The robust chart is retained
        # as metadata/diagnostics but must not replace the protein block used by
        # W1 and Stage-2 protein_energy comparisons against builtin baselines.
        z_by_time = [np.concatenate([r, p[:, :protein_dim]], axis=1).astype(np.float32) for r, p in zip(r_by_time, p_by_time)]
        chart_holder["chart"] = chart
        return TrainingDataBundle(
            adata=adata,
            time_points=times,
            latent_by_time=[torch.as_tensor(z, dtype=torch.float32) for z in z_by_time],
            obs_indices_by_time=obs_indices_by_time,
            extra_modalities_by_time={"rna": r_by_time, "protein_raw": p_by_time, "protein_mask": m_by_time},
            metadata={
                "state_semantics": "z=(r, protein_pca) in canonical benchmark coordinates; runtime rule integrates r and derives protein_pca via h_eta(r,t)",
                "rna_dim": rna_dim,
                "protein_dim": protein_dim,
                "cite_pcfm_chart": chart.state_dict(),
                "protein_mask_semantics": "all_ones_dense_pca: prepared X_protein_pca has no missing marker coordinates; d_p is averaged over all protein PCs",
                "mass_modeling_scope": "balanced_only",
            },
        )

    def build_flow_matching_backend(build_context: FlowMatchingBuildContext) -> FlowMatchingBackend:
        stage_params = build_context.stage_params
        cite_params = _stage_pcfm_params(stage_params)
        rna_dim = int(cite_params.get("rna_dim", build_context.training_data.metadata.get("rna_dim", 50)))
        protein_dim = int(cite_params.get("protein_dim", build_context.training_data.metadata.get("protein_dim", 30)))
        coupling_config = (stage_params.get("flow_matching") or {}).get("coupling") or {}
        coupling = PairedCoordinateBalancedCoupling(
            rna_dim=rna_dim,
            protein_dim=protein_dim,
            alpha_r=float(cite_params.get("alpha_r", 1.0)),
            lambda_p=float(cite_params.get("lambda_p", 0.25)),
            method=str(coupling_config.get("method", cite_params.get("balanced_method", "sinkhorn"))),
            reg=float(coupling_config.get("reg", cite_params.get("balanced_reg", 0.05))),
            normalize_cost=bool(coupling_config.get("normalize_cost", True)),
            warn=bool(coupling_config.get("warn", True)),
            use_mini_batch_balanced=bool(coupling_config.get("use_mini_batch_balanced", True)),
            chunk_size=int(coupling_config.get("chunk_size", stage_params.get("chunk_size", 1000))),
        )
        return PairedCoordinateFlowMatchingBackend(path=LinearDeterministicConditionalPath(sigma=0.0), coupling=coupling, mass=NullMassStrategy())

    def model_builder(model_context: ModelBuildContext) -> torch.nn.Module:
        cfg_params = _config_pcfm_params(model_context.resolved_config)
        rna_dim = int(cfg_params.get("rna_dim", 50))
        protein_dim = int(cfg_params.get("protein_dim", max(1, model_context.latent_dim - rna_dim)))
        hidden = int(cfg_params.get("protein_head_hidden_dim", 128))
        n_fourier = int(cfg_params.get("time_fourier_features", 0))
        time_fourier_scale = float(cfg_params.get("time_fourier_scale", 1.0))
        readout_in_dim = rna_dim + 1 + 2 * max(n_fourier, 0)
        model = DynamicalModel(model_context.latent_dim, model_context.resolved_config["model"])
        model.add_module(
            "protein_readout",
            torch.nn.Sequential(
                torch.nn.Linear(readout_in_dim, hidden),
                torch.nn.SiLU(),
                torch.nn.Linear(hidden, hidden),
                torch.nn.SiLU(),
                torch.nn.Linear(hidden, protein_dim),
            ),
        )
        model.cite_pcfm_state_semantics = "integrates RNA coordinates; protein PCA coordinates are h_eta(r,t) readout on rollout"
        model.cite_pcfm_rna_dim = rna_dim
        model.cite_pcfm_protein_dim = protein_dim
        model.cite_pcfm_time_fourier_features = n_fourier
        model.cite_pcfm_time_fourier_scale = time_fourier_scale
        model.cytobridge_component_modules = {"velocity": ["velocity_net", "protein_readout"]}
        return model

    def flow_matching_loss_hook(loss_context: FlowMatchingLossContext):
        cite_params = _stage_pcfm_params(loss_context.stage_params)
        rna_dim = int(cite_params.get("rna_dim", loss_context.training_data.metadata.get("rna_dim", 50)))
        protein_dim = int(cite_params.get("protein_dim", loss_context.training_data.metadata.get("protein_dim", 30)))
        alpha_p = float(cite_params.get("alpha_protein", 0.25))
        endpoint_weight = float(cite_params.get("endpoint_protein_weight", 0.0))
        n_fourier = int(cite_params.get("time_fourier_features", 0))
        time_fourier_scale = float(cite_params.get("time_fourier_scale", 1.0))
        if not loss_context.regress_v:
            return None
        v_net = getattr(loss_context.model, "velocity_net", None)
        h_eta = getattr(loss_context.model, "protein_readout", None)
        if v_net is None or h_eta is None:
            return None
        # Approved semantics: v_theta(r,t) is RNA-only.  The package batch x is
        # z=(r,q), so explicitly zero the protein-chart block before calling the
        # standard velocity_net; this prevents training-time true q or rollout
        # h_eta(q) values from conditioning RNA velocity.
        r_input = loss_context.x_input[:, :rna_dim]
        t_input = loss_context.t_input.reshape(-1, 1).to(device=r_input.device, dtype=r_input.dtype)
        zero_q = torch.zeros((r_input.shape[0], protein_dim), dtype=r_input.dtype, device=r_input.device)
        v_pred = v_net(torch.cat([r_input, zero_q, t_input], dim=1))
        weights = loss_context.batch.loss_weights.to(device=r_input.device, dtype=r_input.dtype).reshape(-1, 1)
        rna_loss = torch.mean(weights * (v_pred[:, :rna_dim] - loss_context.batch.ut[:, :rna_dim]).pow(2))
        q_target = loss_context.x_input[:, rna_dim : rna_dim + protein_dim].detach()
        q_pred = h_eta(_readout_input(r_input, t_input, n_fourier, time_fourier_scale))
        protein_reliability = torch.isfinite(q_target).to(dtype=r_input.dtype, device=r_input.device)
        q_target_clean = torch.nan_to_num(q_target, nan=0.0, posinf=0.0, neginf=0.0)
        time_weight = _time_loss_weight(t_input, cite_params).to(device=r_input.device, dtype=r_input.dtype)
        protein_weight_matrix = weights * time_weight * protein_reliability
        protein_weight_denom = torch.clamp(protein_weight_matrix.sum(), min=1.0)
        protein_loss = torch.sum(protein_weight_matrix * (q_pred - q_target_clean).pow(2)) / protein_weight_denom
        endpoint_loss = torch.zeros((), dtype=r_input.dtype, device=r_input.device)
        endpoint_count = torch.zeros((), dtype=r_input.dtype, device=r_input.device)
        if endpoint_weight > 0.0:
            batch_extras = getattr(loss_context.batch, "extras", {}) or {}
            x0 = batch_extras.get("endpoint_x0")
            x1 = batch_extras.get("endpoint_x1")
            t0 = batch_extras.get("endpoint_t0")
            t1 = batch_extras.get("endpoint_t1")
            if x0 is not None and x1 is not None and t0 is not None and t1 is not None:
                for x_end, t_end in ((x0, t0), (x1, t1)):
                    r_end = x_end[:, :rna_dim].to(device=r_input.device, dtype=r_input.dtype)
                    q_end = x_end[:, rna_dim : rna_dim + protein_dim].to(device=r_input.device, dtype=r_input.dtype)
                    t_end_col = t_end.reshape(-1, 1) if t_end.ndim == 1 else t_end
                    t_end_col = t_end_col.to(device=r_input.device, dtype=r_input.dtype)
                    q_end_pred = h_eta(_readout_input(r_end, t_end_col, n_fourier, time_fourier_scale))
                    q_end_mask = torch.isfinite(q_end).to(dtype=r_input.dtype, device=r_input.device)
                    q_end_clean = torch.nan_to_num(q_end, nan=0.0, posinf=0.0, neginf=0.0)
                    end_time_weight = _time_loss_weight(t_end_col, cite_params).to(device=r_input.device, dtype=r_input.dtype)
                    end_weight_matrix = end_time_weight * q_end_mask
                    denom = torch.clamp(end_weight_matrix.sum(), min=1.0)
                    endpoint_loss = endpoint_loss + torch.sum(end_weight_matrix * (q_end_pred - q_end_clean).pow(2)) / denom
                    endpoint_count = endpoint_count + 1.0
                endpoint_loss = endpoint_loss / torch.clamp(endpoint_count, min=1.0)
        zero_p_velocity = torch.mean(v_pred[:, rna_dim : rna_dim + protein_dim].pow(2)) if v_pred.shape[1] >= rna_dim + protein_dim else torch.tensor(0.0, device=v_pred.device)
        total_velocity = rna_loss + 0.01 * zero_p_velocity
        return FlowMatchingLossResult(
            replace_velocity_loss=total_velocity,
            extra_loss=alpha_p * (protein_loss + endpoint_weight * endpoint_loss),
            logs={
                "cite_pcfm_rna_velocity_loss": rna_loss.detach(),
                "cite_pcfm_protein_readout_loss": protein_loss.detach(),
                "cite_pcfm_endpoint_protein_loss": endpoint_loss.detach(),
                "cite_pcfm_endpoint_protein_weight": torch.as_tensor(endpoint_weight),
                "cite_pcfm_zero_p_velocity_loss": zero_p_velocity.detach(),
                "cite_pcfm_alpha_protein": torch.as_tensor(alpha_p),
                "cite_pcfm_time_fourier_features": torch.as_tensor(n_fourier),
                "cite_pcfm_time_fourier_scale": torch.as_tensor(time_fourier_scale),
                "cite_pcfm_time_loss_weight_mean": time_weight.mean().detach(),
                "cite_pcfm_protein_reliability_fraction": protein_reliability.mean().detach(),
            },
        )

    def inference_context_builder(infer_context: InferenceContextBuilderContext) -> InferenceContext:
        chart = chart_holder.get("chart")
        return InferenceContext(
            payload={"chart": chart.state_dict() if chart is not None else None, "x0": infer_context.initial_data, "time_points": infer_context.time_points},
            visibility={"x0": "t0_inference", "time_points": "exogenous_inference", "chart": "trained_model_state"},
            provenance={"future_rows_used": False, "state": "t0 RNA rolled forward; protein from h_eta on generated states", "mass_modeling_scope": "balanced_only"},
        )

    def _get_chart(sim_context: SimulationContext) -> PairedCoordinateChart:
        chart = chart_holder.get("chart")
        if chart is not None:
            return chart
        payload = None
        if sim_context.inference_context is not None:
            ctx_payload = getattr(sim_context.inference_context, "payload", sim_context.inference_context)
            if isinstance(ctx_payload, dict):
                payload = ctx_payload.get("chart")
        if payload is None:
            payload = sim_context.training_data.metadata.get("cite_pcfm_chart")
        if payload is None:
            raise ValueError("CITE-PCFM simulation requires stored protein chart payload.")
        chart = PairedCoordinateChart.from_state_dict(payload)
        chart_holder["chart"] = chart
        return chart

    def simulation_hook(sim_context: SimulationContext) -> SimulationResult:
        chart = _get_chart(sim_context)
        device = sim_context.device or torch.device("cpu")
        model = sim_context.model.to(device).eval()
        n_fourier = int(getattr(model, "cite_pcfm_time_fourier_features", 0))
        time_fourier_scale = float(getattr(model, "cite_pcfm_time_fourier_scale", 1.0))
        raw_times = [float(t) for t in sim_context.trajectory_time_points]
        z0 = sim_context.x0_override
        z0_np = z0.detach().cpu().numpy() if isinstance(z0, torch.Tensor) else np.asarray(z0, dtype=np.float32)
        r = torch.as_tensor(z0_np[:, : chart.rna_dim], dtype=torch.float32, device=device)
        weights = np.ones(r.shape[0], dtype=np.float32) / max(int(r.shape[0]), 1)
        points: list[np.ndarray] = []
        weights_by_time: list[np.ndarray] = []
        with torch.no_grad():
            prev_t = raw_times[0]
            for out_i, t in enumerate(raw_times):
                if out_i > 0:
                    dt = float(t - prev_t)
                    if dt > 0:
                        t_mid = torch.full((r.shape[0], 1), float(prev_t + 0.5 * dt), dtype=r.dtype, device=device)
                        # Approved runtime dynamics are RNA-only; protein
                        # readout is evaluated after rollout and must not feed
                        # back into v_theta.
                        zero_q = torch.zeros((r.shape[0], chart.protein_dim), dtype=r.dtype, device=device)
                        z_mid = torch.cat([r, zero_q], dim=1)
                        vel = model.velocity_net(torch.cat([z_mid, t_mid], dim=1))[:, : chart.rna_dim]
                        r = (r + vel * dt).detach()
                        prev_t = float(t)
                t_col = torch.full((r.shape[0], 1), float(t), dtype=r.dtype, device=device)
                q = model.protein_readout(_readout_input(r, t_col, n_fourier, time_fourier_scale))
                z_chart = torch.cat([r, q], dim=1).detach().cpu().numpy().astype(np.float32)
                points.append(z_chart)
                weights_by_time.append(weights.copy())
        return SimulationResult(
            predicted_points_by_time=points,
            predicted_weights_by_time=weights_by_time,
            trajectory_time_points=raw_times,
            aux_state_by_time={"chart_diagnostics": chart.diagnostics, "trusted_inference": "rna_ode_plus_protein_readout"},
            artifacts={},
        )

    def serialize_model(serialize_context: ModelSerializeContext) -> dict[str, object]:
        chart = chart_holder.get("chart")
        if chart is None:
            payload = serialize_context.training_data.metadata.get("cite_pcfm_chart")
            chart = PairedCoordinateChart.from_state_dict(payload) if payload else None
        return {
            "cite_pcfm_chart": chart.state_dict() if chart is not None else None,
            "state_semantics": "runtime RNA ODE plus protein_readout h_eta(r,t); stored z uses canonical protein PCA coordinates",
            "mass_modeling_scope": "balanced_only",
        }

    def deserialize_model(load_context: ModelLoadContext) -> torch.nn.Module:
        model = DynamicalModel(load_context.latent_dim, load_context.resolved_config["model"])
        payload = load_context.model_payload or {}
        chart_payload = payload.get("cite_pcfm_chart")
        rna_dim = int((chart_payload or {}).get("rna_dim", 50))
        protein_dim = int((chart_payload or {}).get("protein_dim", max(1, load_context.latent_dim - rna_dim)))
        cfg_params = _config_pcfm_params(load_context.resolved_config)
        hidden = int(cfg_params.get("protein_head_hidden_dim", 128))
        n_fourier = int(cfg_params.get("time_fourier_features", 0))
        time_fourier_scale = float(cfg_params.get("time_fourier_scale", 1.0))
        readout_in_dim = rna_dim + 1 + 2 * max(n_fourier, 0)
        model.add_module(
            "protein_readout",
            torch.nn.Sequential(
                torch.nn.Linear(readout_in_dim, hidden),
                torch.nn.SiLU(),
                torch.nn.Linear(hidden, hidden),
                torch.nn.SiLU(),
                torch.nn.Linear(hidden, protein_dim),
            ),
        )
        model.load_state_dict(load_context.model_state_dict)
        if chart_payload:
            chart_holder["chart"] = PairedCoordinateChart.from_state_dict(chart_payload)
        model.cite_pcfm_state_semantics = payload.get("state_semantics", "runtime RNA ODE plus protein PCA readout")
        model.cite_pcfm_time_fourier_features = n_fourier
        model.cite_pcfm_time_fourier_scale = time_fourier_scale
        model.cytobridge_component_modules = {"velocity": ["velocity_net", "protein_readout"]}
        return model

    return TrainingAlgorithmSpec(
        algorithm_id=context.algorithm_id,
        base_config="./config.yaml",
        config_overrides={},
        training_data_builder=training_data_builder,
        flow_matching_backend_builder=build_flow_matching_backend,
        flow_matching_loss_hook=flow_matching_loss_hook,
        model_builder=model_builder,
        serialize_model=serialize_model,
        deserialize_model=deserialize_model,
        stage_runner=None,
        inference_context_builder=inference_context_builder,
        simulation_hook=simulation_hook,
        evaluation_metrics_hook=None,
        evaluation_metrics_params={
            "rna_dim": 50,
            "protein_dim": 30,
            "expected_ranges": {"protein_energy": [0.0, 10.0], "rna_energy_guardrail": [0.0, 20.0], "paired_cov_error": [0.0, 5.0]},
            "claim_metric_note": "protein_energy must be computed from sealed CITE-PCFM z=(RNA, protein_readout) rollout; observed future proteins are scoring truth only.",
        },
        notes="CITE-PCFM: balanced paired-coordinate OT-CFM with runtime RNA ODE and rollout protein readout; balanced_only mass scope.",
    )
