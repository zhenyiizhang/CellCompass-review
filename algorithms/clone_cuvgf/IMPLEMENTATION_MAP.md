# Implementation Map: clone_cuvgf

## Binding

- Algorithm: `clone_cuvgf` / Clone-CUVGF-BMF.
- Active proposal: `proposal_20260607T050854815037Z`.
- Required anchor baseline: `vgfm`.
- Nearest builtin rationale: BMF keeps VGFM's positive-measure velocity-growth flow-matching backbone, UOT row-mass growth semantics, package W1/TMV definitions, and 3000-epoch flow-matching training discipline, but lifts the endpoint training problem into persistent terminal-fate branches.
- Proposal-specific BMF changes:
  1. Build terminal clone fate posteriors and reliability weights.
  2. Solve a stable clone-censored UOT backbone that excludes terminal-fate posterior dominance and fixes total row masses.
  3. Build a sparse joint branch projection over stable UOT support, using alternating scaling to decompose fixed row masses into persistent terminal-fate branches with a shared day-4 branch boundary.
  4. Replace standard velocity/growth loss with branch-conditioned velocity/growth loss and add a gate loss to feasible branch marginals.
  5. Use a sealed branch-weighted model-native rollout from t0 for W1/TMV/CFCorr; clone labels and terminal labels are not inference-time inputs.
- Reused components: `ChunkedTransportCouplingStrategy`, `CouplingPlanStore`, `FlowMatchingBackend`, `LinearDeterministicConditionalPath`, `UOTMassStrategy`, package trainer optimizer/checkpoint loop, builtin W1, builtin TMV, and additive custom-metric hook.
- Replaced/custom components: `model_builder` creates `BranchDynamicalModel`; `flow_matching_loss_hook` replaces velocity/growth losses with BMF branch losses; `simulation_hook` performs sealed branch-weighted rollout; `inference_context_builder` documents allowed inference inputs. Builtin W1/TMV are not replaced.

## Builtin Source Pass

- Closest builtin anchor: `vgfm`.
- Source/docs read: `CytoBridge-main/docs/runtime/custom-algorithms/README.md`, `CytoBridge-main/docs/runtime/flow-matching/extension-points.md`, `training_algorithm.py`, `flow_matching_backends.py`, `trainer.py`, `custom_stage_loop.py`, `models.py`, and `analysis.py`.
- Package contracts used:
  - `TrainingDataBundle.latent_by_time` and `obs_indices_by_time` are canonical.
  - `FlowMatchingBackend` owns coupling/path/mass but not model architecture.
  - `flow_matching_loss_hook` may explicitly replace builtin component losses.
  - `evaluation_metrics_hook` is additive and may not overwrite W1/TMV.
  - `simulation_hook` must return a full t0-to-final trajectory with weights.
  - `InferenceContext` provenance declares allowed t0/model/constant inputs.
- Main symbols implemented in `algorithm.py`: `_build_fate_annotations`, `CloneCensoredUOTCoupling`, `CloneBMFBranchCoupling`, `BranchFlowMatchingBackend`, `BranchDynamicalModel`, `_branch_loss_hook`, `_build_inference_context`, `_branch_simulation_hook`, `_compute_clone_fate_metric_from_context`, and `build_training_algorithm`.

## Claim Metric Contract: CFCorr

### Metric identity
- Metric names: `CFCorr` and `CFCorr_macro_pearson`.
- Direction: greater is better.
- Claim object: eligible day-2 cells' sealed rollout-derived terminal fate probability vectors should agree with clone-derived day-6 terminal fate distributions.
- Stage 2 gate: compare against the strongest same-panel CFCorr baseline with W1 guardrail and TMV hard gate. No-clone VGFM is the user-facing anchor; `dynamical_ot` is the current strongest comparator record.

### Prediction source
- Prediction source is the sealed CytoBridge evaluation trajectory returned by `_branch_simulation_hook`.
- Runtime expands each day-2 cell into one weighted particle per terminal-fate branch with weights `q_eta(f|x_i,t0)/n0`.
- Branch particles are integrated with learned `branch_velocity_net(t,x,e_f)` and `branch_growth_net(t,x,e_f)` on the requested dense trajectory grid.
- Builtin W1/TMV and CFCorr receive the same sealed full branch-weighted trajectory semantics; no parent-barycenter collapse is used for trusted metrics. To keep the package dense W1 matrix bounded, the simulation hook may emit a deterministic weighted coreset of the full branch particle measure (`evaluation.branch_metric_coreset_particles`), while full terminal branch endpoints/weights remain in trajectory artifacts for CFCorr. At final time, generated branch endpoints are mapped to terminal fate probabilities with the fixed distance-weighted KNN readout against observed day-6 `X_latent` and day-6 `Cell type annotation` labels.
- The evaluator aggregates branch endpoint fate probabilities back to the original day-2 parent cell by terminal branch weights, then computes CFCorr.
- Fixed readout parameters are in `evaluation_metrics_params`: `terminal_knn=30`, `knn_chunk_size=512`, and `latent_key='X_latent'`.

### Truth source and semantics
- Truth source is day-6 terminal clone composition.
- For each terminal-supported clone, the evaluator forms `G_c=(n_{c,f}+alpha*pi_f)/(n_{c,K}+alpha)` using `dirichlet_alpha=5.0` and the global terminal prior.
- A day-2 cell is eligible only when its clone has at least one terminal day-6 clone-labeled cell (`min_terminal_clone_cells=1`).
- Grouping key: `obs['clone']`, normalized through `_clone_key`; each eligible day-2 source cell maps to its terminal-supported clone truth vector.
- Timepoints: prediction starts from processed time `0` and is evaluated at processed time `2`; processed time `1` remains part of W1/TMV trajectory evaluation and branch-boundary risk diagnostics.
- Label source: final-time `obs['Cell type annotation']` provides terminal fate labels for both KNN readout and terminal clone-fate truth counts.
- The benchmark card documents clone barcodes, processed times 0/1/2, cell type annotations, and SPRING coordinates; side-information audit artifacts remain report evidence.

### Leakage controls
- Clone IDs, terminal labels, clone posteriors `H_c`, UOT row IDs, branch plans, target cells, and terminal snapshots are not inference-time model inputs.
- Training-only clone/fate information is used to form branch projection targets and coupling samples before training.
- Inference-time payload declares only branch rollout constants and trained model state; `_branch_simulation_hook` receives a t0-restricted inference context when the builder is active.
- The metric does not move particles, choose target cells for particles, repair endpoints, or rescale weights. KNN readout/truth construction is evaluation-only and shared for candidate and baselines.
- Builtin W1/TMV are computed from the same evaluation trajectory before the additive CFCorr hook runs.

### Proposal mapping
- Proposal `H_c`, `h_i^k`, and `omega_i^k` map to `_build_fate_annotations`.
- Stable UOT plans `pi^{0,k}` and row masses `r_i^{0,k}` map to `CloneCensoredUOTCoupling` and `UOTMassStrategy`; terminal fate posteriors are excluded from the stable UOT cost.
- Branch projection variables `Gamma^0`, `Gamma^1`, shared `s_j(f)`, and gate targets are represented by sparse support branch edges in `CloneBMFBranchCoupling`, produced by `_store_to_top_edges` plus `_joint_branch_project_three_times` on the three-time Weinreb panel. Clone posteriors enter this projection as reliability-weighted bounded priors using `omega_i^k`; source row totals are hard only after summing across branches.
- Feasible branch gate targets `q_i^k` map to `gate_targets_source` in coupling metadata and are supervised by `_branch_loss_hook` through squared-Hellinger gate loss.
- Branch-conditioned `v_theta(t,x,e_f)` and `g_phi(t,x,e_f)` map to `BranchDynamicalModel.branch_velocity_net` and `branch_growth_net`.
- Runtime branch gate `q_eta(f|x,t0)` maps to `BranchDynamicalModel.gate_net` and is used by `_branch_simulation_hook` to initialize branch particle weights.
- Generated measure `hat_mu_t` maps to the full weighted branch trajectory returned by `SimulationResult`.
- CFCorr maps to `_compute_clone_fate_metric_from_context` and `evaluate_claim_metric`.

### Gap mapping
- Original failure mode: no-clone temporal dynamics can match day-4/day-6 marginals and mass while missing clone-conditioned day-2 fate bias; previous Clone-CUVGF-FP improved CFCorr only weakly and was far below the strongest same-panel baseline.
- BMF addresses the FP failure by separating total row-mass construction from fate supervision, so clone/fate evidence changes branch decomposition rather than total row mass.
- Passing CFCorr together with W1/TMV guardrails supports the original biological claim only if diagnostics are consistent: eligible counts, per-fate correlations, Brier/calibration, prediction entropy, projection residuals, branch residuals, and SPRING maps must not contradict the scalar result.

## Runtime Hot Path / Scalability Contract

- Stable UOT precomputation before epoch loop: yes. `CloneCensoredUOTCoupling` subclasses `ChunkedTransportCouplingStrategy`.
- Branch projection precomputation before epoch loop: yes. `CloneBMFBranchCoupling.build_state` constructs sparse branch edge metadata once.
- OT/UOT solved inside epoch loop: no. Epoch training samples from cached stable/branch plan stores.
- Full adjacent dense state: no. Stable plans are solved in clone-sorted chunks; BMF keeps at most `max_edges_per_row` stable support edges per source row before branch expansion.
- Branch support expansion: edge count is approximately `n_source * max_edges_per_row * n_fates` after IPF pruning, not `n_source*n_target*n_fates`.
- Known approximation: the implemented joint projection uses bidirectionally repaired sparse stable support and alternating multiplicative projections rather than a dense all-pairs global optimizer. It explicitly owns both intervals, reliability-weighted bounded clone/fate priors, feasible projected gate targets, original stable target-column masses, and shared day-4 branch fractions; total-row, true target-column, support-coverage, and day-4 boundary residuals are recorded in coupling metadata.
- Natural no-clone comparator: builtin `vgfm`; strongest current same-panel CFCorr comparator is `dynamical_ot`.

## Mapping Table

| Step ID | Pseudocode Step | Implementation File | Main Symbols | Status | Semantic Check | Deviation Status | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P1 | Build clone-fate branch metadata. | `algorithm.py` | `_build_fate_annotations`, `_extract_clone_arrays` | completed | Computes terminal fate prior, clone posteriors, reliabilities, terminal smoothed one-hot responsibilities; unsupported evidence is neutral with zero reliability. | exact | Training/evaluation metadata only; not concatenated into inference state. |
| P2 | Solve stable UOT backbones. | `algorithm.py` + package | `CloneCensoredUOTCoupling`, `build_pairwise_cost_block`, package UOT solver | completed | Stable cost uses transcriptomic geometry plus conservative censored clone term; terminal fate posteriors are not included; row masses come from UOT plan stores. | exact to stable-mass part | `lambda_fate` is disabled in config to avoid obsolete FP row-mass control. |
| P3 | Solve joint multi-interval branch projection. | `algorithm.py` | `CloneBMFBranchCoupling`, `_store_to_top_edges`, `_joint_branch_project_three_times` | completed | Decomposes stable row masses over terminal-fate branches on sparse stable support, uses `omega`-weighted bounded clone/fate priors, supervises feasible projected gate marginals rather than raw posteriors, and enforces/reports the shared day-4 branch boundary and original stable day-4/day-6 target columns across both intervals. | bidirectional sparse-support approximation | Stable row masses and original stable target-column masses are carried through support pruning; residuals are recorded as `joint_*_residual`, `true_target_col*_support_l1_before_projection`, and zero-supported target mass metadata. |
| P4 | Train branch-conditioned velocity/growth/gate model. | `algorithm.py`, `config.yaml`, package trainer | `BranchDynamicalModel`, `BranchFlowMatchingBackend`, `_branch_loss_hook` | completed | Replaces standard v/g losses with branch-conditioned v/g losses and Hellinger gate loss to feasible branch targets. | exact to branch-loss semantics | Standard package optimizer/checkpoint loop is reused; branch nets are model-owned. Compatible zero `velocity_net`/`growth_net` stubs satisfy the trainer's pre-hook standard-loss call, but their losses are replaced and sealed rollout uses branch nets. |
| P5 | Roll out persistent branch-weighted dynamics. | `algorithm.py` + package evaluator | `_build_inference_context`, `_branch_simulation_hook`, `_weighted_systematic_coreset`, `SimulationResult` | completed | Starts only from day-2 `X_latent`; initializes branch weights with learned `q_eta`; integrates branch v/g nets; returns a deterministic weighted coreset of the full branch measure for W1/TMV and full endpoint artifacts for CFCorr. | coreset approximation to full branch measure | No clone labels, terminal labels, target counts, UOT rows, or target cells are used during inference; no parent barycenter collapse. |rows, or target cells are used to construct predictions. |
| P6 | Evaluate and visualize. | `algorithm.py`; downstream pending | `_compute_clone_fate_metric_from_context`, `evaluate_claim_metric`; SPRING scripts pending | metric completed; visualization pending | CFCorr uses branch-weighted terminal generated endpoints, fixed terminal KNN readout, clone-derived truth, and clone-balanced/reliability weights. | metric exact | SPRING visualizations should be generated only after validated/locked rollout evidence. |
| S1 | Support and projection diagnostics. | `algorithm.py` metadata; campaign logs | `pair_metadata`, `branch_ipf_*_residual`, edge counts | partially completed | Coupling state records stable support size, branch edge count, row/column residuals, and row-mass totals. | needs campaign evidence | Add per-trial diagnosis artifacts after preview/campaign if metrics are surprising. |
| T1 | Trainable module ownership. | `algorithm.py`, `config.yaml` | `BranchDynamicalModel`, `cytobridge_component_modules`, branch config stanzas | completed | Branch velocity/growth/gate modules are model-owned and selected by package training conventions. | exact | Syntax check passed after authoring. |

## Current Required Diagnostics Before Reporting

- Run preview to verify full-data setup, branch edge construction, branch loss hook, sealed simulation hook, W1/TMV, and CFCorr evaluator.
- Inspect implementation-review/inference-review feedback before campaign evidence if runtime hooks trigger reviewers.
- Compare candidate Stage 1/2 metrics against no-clone `vgfm` and strongest same-panel CFCorr baseline (`dynamical_ot` in current records).
- For CFCorr reporting include eligible cell/clone counts, terminal fate counts, Brier score, prediction entropy, truth entropy, per-fate correlations, evaluator fingerprint, and side-information audit path.
- For branch-projection reporting include stable support edge counts, branch edge counts, preserved row-mass summaries, true target-column support residuals, joint row/column/day4-boundary residuals, fate labels, and chunk-size/support sensitivity where feasible.
- For mass/growth claims report W1 and TMV from package rollout only.
- Do not post-hoc rescale weights, move particles, or use target counts during rollout.
- Generate SPRING visualizations only from trained model-native branch rollouts and the fixed CFCorr readout/truth definitions.
