# cite_paired_coord_fm

CITE Paired-Coordinate Flow Matching: balanced temporal CITE-seq dynamics using paired RNA/protein coordinate coupling and rollout protein readout.

> Runtime catalog fields are read from `manifest.yaml: description` and `manifest.yaml: requirements`.

## Requirements
Requires temporal paired CITE-seq AnnData with obs['time_point_processed'], obsm['X_latent'], and paired protein coordinates/masks in obsm/layers/uns. Balanced-only; TMV diagnostic; designed for rollout-derived protein fidelity with anti-shortcut controls.

## Config Baseline
- Seed source: `vgfm` (fixed by workspace initializer)
- Workspace config: `config.yaml`
- Manifest base_config: `./config.yaml`

## Training Plan Mode
- Builtin template is seeded for `flow_matching`, but custom algorithms may also use `neural_ode` stages.
- If you introduce nonstandard per-stage optimization semantics, implement `stage_runner(...)` instead of forcing the logic into additive hooks.

## Generic Runtime Hooks
- `model_builder(...)` lets the algorithm replace the builtin `DynamicalModel`.
- Proposal-required trainable modules must be model-owned or explicitly selected into the optimizer; backend/path/coupling-local `nn.Module` objects are not trained by default.
- `serialize_model(...)` / `deserialize_model(...)` are only needed when reload requires extra payload beyond config + state_dict.
- `stage_runner(...)` is the full training-stage override for stages you own; return `None` for ordinary `flow_matching`/`neural_ode` stages that should continue through the package default runner.
- `simulation_hook(...)` is the evaluation-time prediction override; builtin `W1/TMV` definitions stay fixed.

## Default Coupling API
- The generated `CustomCoupling` already uses `ChunkedTransportCouplingStrategy`.
- Customize `CustomCoupling.build_pairwise_cost_block(...)` for one bounded source/target block when only pairwise geometry changes.
- With `chunk_size: 1000`, an adjacent gap with about 1000 source cells and 1000 target cells gives one `1000 x 1000` cost/coupling matrix; do not add unnecessary mini-batch complexity for tiny/small simulations.
- Use `CostBasedPairwiseOTCouplingStrategy.build_pairwise_cost(...)` only when a full adjacent-pair cost matrix is acceptable.
- Drop down to a custom `CouplingStrategy.build_state(...)` / `sample_pairs(...)` only when coupling needs global cross-time coordination or streaming/sparse state not covered by the chunked-cost API.
- `build_flow_matching_backend(build_context)` receives `build_context.training_data`.
- Add `training_data_builder` only when you need extra aligned modalities.
- Add `flow_matching_loss_hook` when you need additive differentiable loss or an explicit replacement for one builtin component loss (`replace_velocity_loss`, `replace_growth_loss`, `replace_score_loss`). Builtin model heads take `loss_context.net_input` in `[x, t]` order; prefer `loss_context.x_input` and `loss_context.t_input` instead of manually slicing when possible.

- Add `evaluation_metrics_hook` only when you need additive custom metrics.
- Put custom metric hyperparameters or external evaluation knobs in `evaluation_metrics_params`.
- Do not modify builtin `W1` or `TMV`; custom metrics must be appended under `custom_metrics`.

## Scalability Requirement
- Scalability is scale-dependent, not a checkbox. Full cost is acceptable for small adjacent gaps whose cost/coupling matrix is about `1000 x 1000`; large real-data gaps such as Weinreb need bounded-memory chunking/sparse/streaming behavior.
- `use_mini_batch=True` or a `chunk_size` value does not prove large-data scalability unless full cost/mask/kernel/plan state is avoided before chunking.
- If a real benchmark has large adjacent time pairs and the method only changes pairwise geometry, use `ChunkedTransportCouplingStrategy` before writing custom sampler code.
- If the chunked-cost API is insufficient, avoid full dense cost/mask/plan state by implementing a streaming, sparse, landmark, or coreset coupling path.
- Document the actual memory path in `IMPLEMENTATION_MAP.md`; do not rely on a mini-batch flag as the evidence.

## Files
- manifest.yaml
- algorithm.py
- config.yaml
- IMPLEMENTATION_MAP.md

## Entry point
- build_training_algorithm(context) -> TrainingAlgorithmSpec
