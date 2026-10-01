# Implementation Map: cite_paired_coord_fm

## Baseline Anchor

- Required anchor baseline: `balanced_ot_cfm`.
- Why this is the nearest builtin: CITE-PCFM is a balanced deterministic OT-CFM method with `NullMassStrategy`, a linear deterministic conditional path, and adjacent-time balanced OT coupling. It differs from `balanced_ot_cfm` by adding paired CITE protein coordinates to the coupling cost and by training a rollout protein readout.
- Components reused or intentionally changed: reuses `BalancedOTCouplingStrategy`, `LinearDeterministicConditionalPath`, `FlowMatchingBackend`, `NullMassStrategy`, `DynamicalModel`, and package trainer/evaluation. It changes cost geometry (`PairedCoordinateBalancedCoupling`), data packing/chart (`training_data_builder`), velocity/readout loss (`flow_matching_loss_hook`), and simulation to integrate RNA with `h_eta` protein readout.

## Claim Metric Contract

### Primary metric: `protein_energy`

- **Proposal claim object:** rollout-derived protein distribution fidelity for paired CITE-seq proteins predictable from RNA/time.
- **Prediction source:** sealed `SimulationResult.predicted_points_by_time` / package `EvaluationTrajectory` produced by `simulation_hook`; each point is `z=(generated RNA, protein_readout(generated RNA,t))`.
- **Truth / observed source:** same-time `EvaluationTimepointResult.observed_points` protein block from the prepared paired CITE AnnData / benchmark timepoint results.
- **Truth semantics evidence:** approved proposal defines CITE-seq as same-cell RNA/protein observations; prepared data exposes `obsm['X_rna_svd']`, `obsm['X_protein_pca']`, and paired `obsm['X_latent']=(RNA,protein)` with `obs['time_point_processed']` as the biological time axis. The metric is distributional CITE protein fidelity, not fate/lineage truth.
- **Grouping key:** none for the primary gate; it is a global same-time distributional metric unless later stratified diagnostics are explicitly added.
- **Timepoints:** all evaluated observed future timepoints supplied in `EvaluationTimepointResult` / `timepoint_results`, excluding t0 when the package evaluation treats t0 as initial condition.
- **Label source:** none; labels/cell types/lineages are not used to define or score this metric.
- **Leakage controls:** prediction is produced before metric scoring from t0 RNA, exogenous time grid, trained model state, and fixed chart constants only. The metric must not rerun inference, move particles, repair proteins, read observed future RNA/protein during prediction, rescale weights to target counts, or use endpoint matching separate from the sealed trajectory.
- **Proposal mapping:** PROPOSAL.md P6 defines inference as `dR_t/dt=v_theta(R_t,t)` followed by `P_hat_t=h_eta(R_t,t)`; P7 defines protein evaluation from the sealed generated trajectory; Claimed Capability requires lower rollout `protein_energy` from `h_eta`, not endpoint repair.
- **Gap mapping / why it cannot pass if the gap is unsolved:** the motivating gap is whether paired RNA/protein CITE measurements can improve rollout-derived protein fidelity beyond RNA-only balanced dynamics after the prior stochastic residual CITE direction failed. A valid Stage 2 claim requires this metric plus anti-shortcut controls: `lambda_p=0` RNA-only coupling with identical readout, shuffled protein chart/pairs, time-only/collapsed protein heads, and compatible builtin/RNA-only rollout controls. If protein fidelity is only endpoint repair, a time-only/collapsed readout, or generic RNA-only imputation, those controls should match or erase the candidate advantage, so the primary metric alone is not sufficient scientific proof without those controls.
- **Implementation / wiring:** scoring code is `metrics/protein_energy_evaluator.py:evaluate_claim_metric`. The campaign claim metric spec references this external evaluator path/function; `TrainingAlgorithmSpec.evaluation_metrics_hook=None` is intentional because the Stage 2 campaign evaluator computes the metric from saved candidate/baseline `EvaluationTrajectory` objects with matching provenance rather than from an algorithm-private in-training hook.
- **Status:** implemented for current campaign use.

### Diagnostic metric: `rna_energy_guardrail`

- **Proposal claim object:** RNA distribution guardrail paired with the protein metric; builtin W1 remains the canonical RNA metric.
- **Prediction source:** sealed generated RNA block from the same `SimulationResult` / `EvaluationTrajectory` used for `protein_energy`.
- **Truth / observed source:** same-time observed RNA block from `EvaluationTimepointResult.observed_points`.
- **Grouping key / label source:** none.
- **Timepoints:** same evaluated observed future timepoints as `protein_energy`.
- **Leakage controls:** read-only additive diagnostic; no feedback into simulation and no correction of predictions.
- **Proposal mapping:** PROPOSAL.md P7 preserves builtin W1/TMV and adds protein metric diagnostics; this energy diagnostic helps detect whether protein improvement comes with RNA distribution collapse.
- **Gap mapping:** if `protein_energy` improves while RNA distribution quality collapses, the paired CITE dynamics claim is not solved; this diagnostic supports the W1 guardrail but is not a replacement for W1.
- **Implementation / wiring:** `metrics/protein_energy_evaluator.py:evaluate_claim_metric` via the same external campaign evaluator.
- **Status:** implemented diagnostic.

### Diagnostic metric: `paired_cov_error`

- **Proposal claim object:** preservation of same-cell RNA/protein covariance structure in the generated paired chart.
- **Prediction source:** sealed generated RNA/protein blocks from the same rollout.
- **Truth / observed source:** same-time observed RNA/protein blocks.
- **Grouping key / label source:** none.
- **Timepoints:** same evaluated observed future timepoints as `protein_energy`.
- **Leakage controls:** read-only diagnostic; it does not rerun inference or alter candidate predictions.
- **Proposal mapping:** PROPOSAL.md novelty/expected outcome require protein-fidelity improvements to be marker/time-localized and paired-geometry dependent.
- **Gap mapping:** this diagnostic helps distinguish paired RNA/protein structure from independent marginal protein matching, but it is not the primary gate alone.
- **Implementation / wiring:** `metrics/protein_energy_evaluator.py:evaluate_claim_metric` via the same external campaign evaluator.
- **Status:** implemented diagnostic.

## Mapping Table

| Step ID | Pseudocode Step | Implementation File | Line Number(s) | Status | Semantic Check | Deviation Status | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P1 | Validate and pack paired data | `algorithm.py` | 180-238 | implemented | Validates `time_point_processed`; uses `X_rna_svd` + `X_protein_pca` when present, otherwise paired `X_latent`; preserves RNA as canonical runtime block. | exact | Prepared paper AnnData has `X_rna_svd` (50), `X_protein_pca` (30), and `X_latent` (80). |
| P2 | Build denoised protein chart / protein-coordinate metadata | `algorithm.py` | 44-120, 220-245 | implemented | Chart is fit only from observed training protein coordinates and stored as preprocessing/model metadata; package-facing latent state remains canonical `z=(X_rna_svd, X_protein_pca)` so candidate/baseline W1 and protein_energy use the same coordinate system. | acceptable approximation | Uses robust median/IQR chart as metadata/diagnostic normalization support rather than replacing canonical evaluation coordinates; all-ones dense-PCA mask is explicit in metadata. |
| P3 | Solve paired-coordinate balanced couplings | `algorithm.py`; `config.yaml` | 123-178, 250-270; config coupling section | implemented | Balanced plan uses RNA cost plus `lambda_p` mask-normalized protein PCA cost; in prepared dense-PCA data the mask is all ones and protein term is averaged over protein PCs. | exact for current dense CITE benchmark | Uses package chunked balanced OT/Sinkhorn plan-store path. |
| P4 | Sample conditional RNA/protein path supervision | Package `FlowMatchingBackend` with custom data/coupling plus `PairedCoordinateFlowMatchingBackend` endpoint extras | `algorithm.py` 180-270 and package sampler | implemented | Samples paired endpoints from `pi_k`; path is deterministic linear in z=(RNA, protein PCA); protein target is the same path point protein block; endpoint extras expose training-only adjacent snapshot endpoints for readout calibration. | exact | Uses package standard flow-matching batch sampler and a thin custom wrapper that stores endpoint tensors in `batch.extras`; endpoint extras are not used during inference or metric scoring. |
| P5 | Optimize flow and protein-readout losses | `algorithm.py` | loss hook | implemented | Replaces velocity loss with RNA velocity loss from `v_theta(r,t)` by zeroing protein block inputs before velocity_net; trains `h_eta(r,t)` on path-sampled protein PCA states with package `loss_weights` and, when `endpoint_protein_weight>0`, on training-only coupled endpoint protein PCA targets. | exact for balanced dense CITE benchmark | Endpoint calibration was added after Stage-2 diagnosis because protein_energy scores endpoint rollout protein distributions; this uses supervised training snapshots/couplings only and does not repair evaluation predictions. |ighting; dense prepared protein PCA has all coordinates reliable, while non-finite coordinates are safely masked. |
| P6 | Inference rollout for new valid t0 cells | `algorithm.py` | simulation hook | implemented | Inference uses only t0 RNA, time, and trained model constants; integrates RNA with protein block inputs zeroed and evaluates `protein_readout` on generated RNA/time in canonical protein PCA coordinates. | exact | No OT rows, future snapshots, target statistics, or correction are used; `h_eta` output does not feed back into velocity. |
| P7 | Evaluate and gate | `metrics/protein_energy_evaluator.py`; campaign claim metric spec | full metric file | implemented | Additive metrics read sealed package `timepoint_results`; builtin W1/TMV remain untouched; TMV is diagnostic only. | exact | Stage 2 campaign claim metric spec must pass evaluator path `metrics/protein_energy_evaluator.py` and function `evaluate_claim_metric`. |
| S1 | Scalability / memory path | `algorithm.py`; package chunked balanced OT | 123-178, 240-258; config chunk_size=1000 | implemented with bounded benchmark scope | Custom code builds bounded cost blocks through `build_pairwise_cost_block`; this is acceptable for current benchmark panels but not claimed as a universal atlas-scale streaming solution. | acceptable approximation | Before atlas-scale or missing-protein panels, inspect plan memory/chunk quality or implement native sparse/streaming `CouplingPlanStore.from_edges`. |
| T1 | Trainable module ownership | `algorithm.py` | model builder / deserialize sections | implemented | `protein_readout` is attached as a model submodule and registered in `model.cytobridge_component_modules={"velocity": ["velocity_net", "protein_readout"]}`, so v-stage optimizer selects it; loss hook adds differentiable readout and endpoint calibration losses. | exact | Syntax/import and one-epoch preview passed after endpoint-calibration/canonical-coordinate edits. |

## Implementation Diagnosis

- Required anchor baseline: `balanced_ot_cfm`.
- Package classes/functions reused: `BalancedOTCouplingStrategy`, `LinearDeterministicConditionalPath`, `NullMassStrategy`, `FlowMatchingBackend`, `DynamicalModel`, `TrainingDataBundle`, `SimulationResult`.
- Custom classes/functions: `PairedCoordinateChart`, `PairedCoordinateBalancedCoupling`, `training_data_builder`, `flow_matching_loss_hook`, `simulation_hook`, `metrics/protein_energy_evaluator.py:evaluate_claim_metric`.
- Natural degeneration/ablation: set `cite_pcfm.lambda_p=0` with the same readout; set `alpha_protein=0` to remove protein readout training; shuffled protein/chart controls require separate data perturbation/config hooks before Stage 2 claim evidence; time-only/collapsed heads are controls, not trusted candidate semantics.
- OT hot path: coupling cost is built in bounded blocks via package `BalancedOTCouplingStrategy`/chunked transport path and custom code does not solve OT in the epoch hot path. This is adequate for the current Stage 1/2 CITE panels, but it is not a blanket atlas-scale guarantee; atlas-scale use requires a follow-up sparse/streaming plan-store check.
- Component opening order for diagnosis: data/chart -> paired cost/coupling -> path/loss -> simulation/inference -> claim metric -> baseline provenance.
