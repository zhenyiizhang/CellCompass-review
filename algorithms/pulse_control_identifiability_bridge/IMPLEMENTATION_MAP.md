# Implementation Map: pulse_control_identifiability_bridge

## Metadata
- Algorithm ID: `pulse_control_identifiability_bridge`
- Approved proposal: `proposal_20260620T213736784915Z`
- Required anchor baseline: `balanced_ot_cfm`
- Nearest engineering anchor: balanced deterministic OT-CFM with custom pulse/control side-information.
- Current revision: repaired after implementation-review revise verdicts, active proposal revision `proposal_20260620T213736784915Z`, and inference-review Claim Metric Contract audit. Implements confidence-weighted train-only nuisance estimation, train-derived nuisance budget, hard fail-fast proposal data-contract validation, support/collinearity gating, endpoint-compatible side-information suppression, the approved finite bounded residual-family selector over endpoint-valid paths, projected-pulse cost, and the reviewed evaluator-provenance rule requiring PCIE to bind through the active workspace `evaluation_metrics_hook` / evaluator identity `pulse_control_identifiability_bridge.claim_metric:evaluate_claim_metric:v1` rather than stale campaign evaluator URIs.s on `P_perp dot{psi}_tau`, residual regularization, and an explicitly audited rollout-derived PCIE metric contract.

## Mapping Table

| Step ID | Pseudocode Step | Implementation File | Status | Semantic Check | Deviation Status | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| P1 | Validate data and split contract: require canonical time/latent fields, train/heldout/unusable anchor split, baseline arms, new/old layers, and metabolic pulse sidecars; heldout anchors are metric-only. | `algorithm.py:_split_code_from_obs`, `_validate_pcib_data_contract`, `build_pcib_payload`; prepared H5AD contract | completed | Missing proposal-required obs/layers/obm fields raise failures; train and heldout anchors are both required; `mvab_anchor_confidence` is required rather than inferred. | exact | Heldout anchors are excluded from train-side path/coupling/loss construction and used only by PCIE evaluation. |
| P2 | Estimate baseline nuisance instrument `N` and train-derived allowed nuisance budget from HCT116_control versus HCT116_0d baseline contrast with train-only baseline confidence weights. | `algorithm.py:_baseline_masks`, `_fit_nuisance_basis`, `build_pcib_payload` | completed | t=0 control/labeled masks are deterministic; labeled baseline uses train anchors; missing or below-threshold baseline arms raise `ValueError`; baseline confidence weights enter weighted displacement SVD and weighted nuisance-budget quantile; rank is capped. | exact | No fallback no-instrument path is used for promoted PCIB evidence. |
| P3 | Build deconfounded pulse anchors using `P_perp = I - NN^T`; compute confidence, support, and collinearity diagnostics; downweight weak/collinear train side information. | `algorithm.py:_project_perp`, `_support_gated_confidence`, `build_pcib_payload` | completed | `pulse_perp` is projected through frozen `N`; `effective_confidence` gates only train anchors by per-time support and projected-pulse fraction; heldout confidence is preserved for metrics. | exact | Side-information gates are deterministic train/evaluation metadata, not learned shortcuts. |
| P4 | Select balanced endpoint-valid adjacent path laws over endpoint pairs and the proposal-approved finite endpoint-vanishing residual family; evaluate residual regularization and pulse/leakage terms on the actual path tangent `dot psi_tau`; suppress endpoint-incompatible side terms; avoid unbalanced repair and dense global plans. | `algorithm.py:_pcib_select_residual_torch`, `PCIBBalancedPulseCoupling.build_pairwise_cost_block`, `PCIBEndpointResidualPath`; `config.yaml:pcib` | completed | `_pcib_select_residual_torch` minimizes the selector over the finite family `{0, 0.5 r_anchor, r_anchor}` and uses `P_perp dot psi_tau` plus nuisance-over-budget and residual penalties; chunked balanced coupling avoids a full dense global plan. | exact proposal revision | Chunked blocks are a scalable approximation and must be monitored for variance. |
| P5 | Train conditional flow matching on selected endpoint-valid path samples with one model-native velocity field; no W1/TMV/PCIE head or static predictor. | `algorithm.py:PCIBFlowMatchingBackend.sample_batch`, `pcib_flow_matching_loss_hook`, `build_training_algorithm` | completed | `sample_batch` returns standard CFM fields from the endpoint-residual path; the loss hook applies alignment/leakage penalties to `model.velocity_net(context.net_input)` only; the velocity net is the only trainable module. | exact | Extra losses are differentiable training regularizers, not separate predictors. |
| P6 | Roll out deterministically from t0 `X_latent` and trained model using package-native evaluation; no future snapshots, heldout anchors, target stats, coupling rows, mass scaling, or endpoint repair. | standard CytoBridge evaluation trajectory; `algorithm.py:build_training_algorithm`, deterministic `PCIBEndpointResidualPath`; `config.yaml` mass/noise settings | completed | PCIB does not override package rollout (`inference_context_builder=None`, `simulation_hook=None`); `sigma=0.0`; `NullMassStrategy`; metrics consume `EvaluationMetricsContext.evaluation_trajectory`. | exact | Balanced-only deterministic contract is preserved. |
| P7 | Evaluate rollout-derived heldout PCIE, deconfounded cosine, nuisance leakage, support, evaluator identity, and require same evaluator/W1 backend for baselines. | `algorithm.py:evaluate_claim_metric`, `CLAIM_METRIC_EVALUATOR_ID`; campaign baseline/control tools | completed | PCIE uses frozen training state, heldout anchors only, generated trajectory finite differences, nearest generated particle matching, `P_perp` projection, leakage above budget, and component/support returns. | exact for evaluator; external comparator execution pending | Candidate, builtin/reference, locked baselines, and controls must use the same PCIE evaluator parameters and W1 backend provenance. |
| P8 | Downstream after locked release uses locked trajectory and PCIE diagnostics for biological figures; gene claims require projection support. | downstream workflow after final regression | completed | Training code makes no downstream biological claims before final-regression lock; downstream consumes locked model/evaluation artifacts. | exact lifecycle separation | Not a pre-training blocker. |
| S1 | Scalability / memory path for real HCT116 panel. | `algorithm.py:_nearest_indices`, `_fit_nuisance_basis`, `PCIBBalancedPulseCoupling`, `evaluate_claim_metric`; `config.yaml:pcib` caps | completed | Baseline matching is capped/chunked; coupling uses bounded chunked balanced blocks; metric anchors/particles and nearest matching are capped/chunked. | accepted scalable approximation | Preview should be rerun after revisions; monitor chunk counts and memory. |
| T1 | Trainable module ownership and mass/stochasticity constraints. | `algorithm.py:build_training_algorithm`, path/backend/loss hooks; `config.yaml` | completed | The optimizer owns standard velocity-net parameters; PCIB hook adds differentiable extra loss; no growth head, stochastic head, metric head, or decoder is added. | exact | Balanced-only deterministic contract preserved. |

## Claim Metric Contract

### Metric identity and direction
- Metric name: `pcib_pulse_control_identifiability_error` (`PCIE`), lower is better.
- Component diagnostics: heldout deconfounded cosine, nuisance leakage, leakage-above-budget, support counts, valid matched anchor counts, generated velocity norm, projected pulse norm, and evaluator identity.
- Evaluator identity: `pulse_control_identifiability_bridge.claim_metric:evaluate_claim_metric:v1` declared in `algorithm.py` and surfaced through evaluation parameters/provenance.

### Prediction source
- Source object: sealed CytoBridge `EvaluationTrajectory` generated by package-native rollout from the observed t0 latent population using the trained PCIB velocity model.
- Concrete arrays: `context.evaluation_trajectory.trajectory_points_by_time`, indexed by generated trajectory time values.
- Predicted local tangent: for each scored anchor time `t_i`, compute generated finite-difference velocities from `trajectory_points_by_time[obs_traj_idx] -> trajectory_points_by_time[obs_traj_idx+1]`, i.e. `(x_gen(t_next) - x_gen(t_i)) / (t_next - t_i)`, where `t_next` is the next generated trajectory slice, not a future observed-cell lookup. The trajectory-step/dt provenance must be recorded for paper claims.
- Anchor-to-prediction matching: for every eligible heldout anchor row at time `t_i`, nearest generated current particle in the generated `t_i` slice is selected; the corresponding generated finite-difference velocity is the prediction used for that heldout anchor neighborhood.
- Forbidden prediction sources: future observed target cells, target distribution statistics, target counts, cell barcodes, stored coupling rows/plans, heldout metabolic anchors during rollout, post-hoc endpoint repair, mass rescaling, and any static PCIE/W1/TMV head.

### Truth / observed source
- Source fields: heldout metabolic pulse anchors from `obsm['X_metabolic_velocity']` and confidence from `obs['mvab_anchor_confidence']` in the prepared raw-pool-derived Well-TEMP HCT116 panel.
- Heldout definition: rows with `obs['mvab_anchor_split'] == 'heldout'` or equivalently split code `2` inside the PCIB payload.
- Projection: observed pulse anchors are projected by the train-derived nuisance subspace `N` via `P_perp = I - N N^T`; `N` and the nuisance budget are fitted before evaluation from train-side HCT116_control versus HCT116_0d baseline contrast and are not refit on heldout anchors.
- Confidence rule: anchors with nonpositive or missing confidence are excluded; remaining anchor contributions are confidence-weighted.
- Biological-truth boundary: these fields are metabolic-pulse treatment-response tangent anchors under the preprocessing/proposal contract, not terminal fate, lineage, or descendant labels.

### Grouping key and aggregation
- Primary grouping key: per heldout anchor row / local anchor neighborhood at its observed time point.
- Matching/grouping rule: each heldout anchor row is matched to its nearest generated current particle at the same generated/observed anchor time; the matched generated particle defines the local neighborhood whose generated finite-difference velocity is scored.
- Aggregation: confidence-weighted average across eligible heldout anchors after nearest generated-particle matching; global PCIE is the confidence-weighted aggregate over scored anchor rows and generated finite-difference intervals.
- No categorical lineage/fate grouping is used. `cell_type`, barcode, terminal fate, or descendant labels are not used as metric truth or grouping semantics.
- Diagnostic groupings reported/inspectable: time interval, heldout support count, valid matched anchor count, and nuisance leakage components.

### Timepoints and intervals
- Time key: canonical `obs['time_point_processed']` / generated trajectory time values.
- Dataset timepoints for the registered benchmark: `0.0`, `1.0`, `2.0`, `3.0` days.
- Scored anchor times: heldout anchors at an observed time `t_i` are scored only when the generated trajectory contains a same-time slice and a following generated trajectory slice needed for local finite differences.
- Current dataset interpretation: observed anchor times at `0.0`, `1.0`, and `2.0` can contribute when heldout anchors and generated slices exist; final-time anchors at `3.0` are skipped if no next generated slice exists.
- Generated-step rule: the code scores the local generated finite-difference step `obs_traj_idx -> obs_traj_idx+1` in the generated trajectory grid, which may be the package dense step (for example 0.1) rather than the next observed-day interval. This is intentional because metabolic pulse anchors are local tangents; reports must record the actual trajectory-step provenance.
- Candidate and all baselines/controls must use the same trajectory time grid/resolution policy and the same finite-difference interval rule.

### Label/source semantics
- Label source for metric eligibility: `obs['mvab_anchor_split']` and `obs['mvab_anchor_confidence']` only.
- Side-information source for nuisance definition: `obs['sample_name']`, `obs['treated_5AZA_CdR']`, `obs['IAA_treatment']`, `obs['metabolic_label_4sU']`, and baseline-arm membership as defined by the preprocessing contract/proposal.
- No terminal fate, lineage, cell-type, barcode, perturbation-response class, or future sample label is interpreted as PCIE truth.
- Missing or low-confidence anchors are treated as unknown/unusable for PCIE, not as negative examples.

### Leakage controls
- Train-only nuisance control: `N` and the nuisance budget are derived from train-side baseline/control information before evaluation; heldout pulse anchors do not fit `N`, coupling, residual selector costs, path samples, or the velocity model.
- Rollout boundary: evaluation consumes the sealed package-generated trajectory and does not mutate generated positions or weights.
- Metric-only heldout use: heldout metabolic pulse anchors are read only inside `evaluate_claim_metric` after rollout has been generated.
- Comparator consistency: candidate, builtin/reference, locked prior-algorithm baselines, shuffled-pulse controls, and no-instrument controls must be scored with the identical evaluator code/config, capped anchor/particle settings, trajectory time-grid policy, and matching W1 backend provenance.
- Negative controls required for biological interpretation: shuffling heldout pulse anchors should reduce/degrade deconfounded alignment; removing or shuffling the nuisance instrument should increase leakage or remove the PCIE advantage. These controls test that PCIE is not an incidental geometry score.

### Proposal mapping
- Approved proposal: `proposal_20260620T213736784915Z`.
- P6 mapping: the proposal requires package-native deterministic rollout from t0 with no heldout/future correction; implemented by leaving `inference_context_builder=None` and `simulation_hook=None`, using deterministic path/noise settings, and consuming `EvaluationTrajectory` only after rollout.
- P7 mapping: the proposal defines PCIE as a heldout, rollout-derived, nuisance-projected pulse/control identifiability error using frozen `N`, deconfounded pulse anchors, finite-difference generated tangents, and nuisance leakage above a train-derived budget.
- Distribution-recovery mapping: PCIB remains a balanced endpoint-valid flow-matching model; PCIE is an additive Stage-2 claim metric and never replaces W1/TMV or the standard rollout trajectory.

### Gap mapping / why this validates the motivating gap
- Original motivating gap: prior balanced temporal dynamics on the HCT116 5-AZA-CdR Well-TEMP setting can fit expression movement while leaving the generated treatment trajectory explainable by baseline labeling/IAA/control nuisance rather than identifiable deconfounded treatment-response pulse structure.
- What PCIE tests: PCIE rewards generated local tangents that align with heldout metabolic pulse anchors after projecting away the train-estimated baseline nuisance subspace, while penalizing velocity energy leaking into that nuisance subspace above the train-derived budget.
- Why it is not an easier proxy: a generic smoother or endpoint-matching model can obtain good W1 but should not simultaneously achieve high heldout `P_perp` pulse alignment and low nuisance leakage if it has not learned the deconfounded treatment-response mechanism. Conversely, high PCIE with high leakage, velocity-norm inflation, insufficient heldout support, shuffled-pulse success, or no-instrument success invalidates the claim.
- Required interpretation: Stage-2 success requires PCIE at least 10% lower than the strongest comparable strict builtin/reference baseline under identical evaluator/W1 provenance, W1 within balanced guardrails, and negative-control degradation. If W1 improves but PCIE/controls do not, the identifiability claim is falsified.

## Reviewer Blocker Remediation Checklist
- Missing/confidence-unweighted nuisance estimator: addressed via confidence-weighted baseline displacement SVD and weighted nuisance-budget quantile.
- P4 residual-family proposal drift: addressed by reviewed proposal revisions ending at `proposal_20260620T213736784915Z`, which explicitly define the finite bounded residual-family estimator implemented by `_pcib_select_residual_torch`.
- Evaluator-provenance campaign drift: addressed by active proposal revision `proposal_20260620T213736784915Z`, which requires trusted campaign evidence to bind PCIE through the reviewed active-workspace `evaluation_metrics_hook` / evaluator identity `pulse_control_identifiability_bridge.claim_metric:evaluate_claim_metric:v1`; stale campaign evidence with mismatched evaluator provenance is obsolete.
- P4 selector missing residual/path terms and projected-pulse denominator: addressed by `_pcib_select_residual_torch`, which chooses residuals over the approved finite bounded family, adds residual `||r||^2` cost, evaluates the pulse term on `P_perp dot psi_tau`, and evaluates leakage above the train-derived nuisance budget on the N-axis tangent.
- Missing support/collinearity/endpoint gating: addressed via `_support_gated_confidence` and endpoint cosine gates in coupling/residual construction.
- Missing baseline arms silently collapsing PCIB: addressed by hard fail-fast validation in `_fit_nuisance_basis` plus `_validate_pcib_data_contract`; no-instrument behavior is reserved for explicit controls and cannot be promoted as PCIB evidence.
- Prior malformed map: addressed by this exact-column parseable table.
- Inference-review Claim Metric Contract blocker: addressed by the explicit PCIE prediction source, truth source, grouping key, timepoints/intervals, label source, leakage controls, proposal mapping, and motivating-gap mapping above.

## Self-Check
- Proposal semantics preserved: balanced-only, deterministic rollout, endpoint-valid paths, train-only nuisance basis and budget, heldout-only PCIE, and the revised proposal-approved finite residual-family endpoint-valid path-law selector with projected deconfounded pulse term.
- No target-count lookup, mass repair, endpoint repair, metric head, direct PCIE/W1/TMV predictor, or barcode/coupling-row inference path.
- Documentation now satisfies the strict Claim Metric Contract audit requested by the inference evaluator; rerun preview/review before trusted campaign evidence.
- Remaining lifecycle evidence: fresh preview, implementation/inference reviewer approval if triggered, campaign trials, strict baselines/controls with matching provenance, stage gates, final regression, downstream biology, and paper review.
