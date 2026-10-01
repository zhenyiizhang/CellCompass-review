# Implementation Map: brgusfm_rh5_0613

Proposal ID: `proposal_20260613T235954411760Z`
Workspace: `/lustre/home/2501111653/.cellcompass/training_algorithms/brgusfm_rh5_0613`
Required anchor baseline: `vgfm`
Status: completed current-source implementation map for trusted review/training under the active approved proposal.

Clean-run scope: use only the current output directory, the five allowed task packages/preprocessed inputs/visible ground truth for diagnosis/scoring, canonical/local visible scorer/verifier code under the current output directory, and normal CytoBridge package docs/code. Do not use benchmark registries, prior result roots, hidden cases, simulator/generator source, old archived algorithms, or case-specific hard-coded outputs.

## Required proposal-to-implementation mapping table

| Step ID | Pseudocode Step | Implementation File | Status | Semantic Check | Deviation Status | Notes |
|---|---|---|---|---|---|---|
| P1 | Build the data contract from AnnData/task metadata; preserve measured gene-space state, times, gene names, labels, target times, and clean output roots. | `algorithm.py:54-137`; `algorithm.py:1272-1291`; `config.yaml:84-87` | completed | `_adata_matrix` reads the AnnData matrix directly; `_obs_time_array` reads `time_point_processed`/`time`; task/output roots are resolved only from config/current run; inference context exposes only `x0`, exogenous time grid, and t0 row alignment. | no_deviation | Allowed task sidecars are limited to `prediction_targets.json` and `fate_classifier.pkl`; ground-truth files are not inference inputs. |
| P2 | Solve adjacent source-relaxed, target-constrained UOT endpoint plans and expose row sums/conditional pair probabilities for training supervision. | `algorithm.py:288-344`; `algorithm.py:347-360`; `algorithm.py:384-424`; `algorithm.py:431-488`; `algorithm.py:1246-1270`; `config.yaml:61-75` | completed | `CustomCoupling` subclasses the package chunked UOT coupling, builds bounded pairwise gene-space cost blocks with unit source/target masses, and `build_flow_matching_backend` uses `RegularizedUnbalancedConditionalPath` plus `UOTMassStrategy`. Pair ids are kept only as training supervision extras. | no_deviation | Locked Remaining-Hard5 intervals were audited as below `chunk_size=20000`, so each locked-panel interval is single-block and implements the intended adjacent endpoint plan; larger future panels remain a semantics/scalability risk. |
| P3 | Construct stochastic bridge samples with flow, score/noise, and UOT row-sum growth targets scaled by true interval length. | `algorithm.py:431-488`; `algorithm.py:1260-1263`; `config.yaml:1-17`; `config.yaml:35-44`; `config.yaml:72-73` | completed | `sample_batch` samples local times/noise, calls the package conditional path for `xt` and `compute_conditional_flow`, calls mass/path conditional-mass targets, and divides velocity/growth targets by `time_interval`. | no_deviation | Stochastic bridge training is through the flow-matching backend, not export-time randomization. |
| P4 | Build future-label/branch targets by transport dynamic programming and train path/node branch CE. | `algorithm.py:362-424`; `algorithm.py:507-567`; `config.yaml:45-60` | completed | Final labels/readouts become one-hot targets; earlier targets are propagated backward through the coupling store exactly as transport-DP future-label targets; branch path CE and bounded observed-node CE train `q_eta`. | no_deviation | The prior non-proposal observed-label anchor was removed; node CE now supervises only the DP targets and does not replace them with current-label targets. |
| P5 | Train BRGU fields and regulatory losses: sparse signed operator inside drift, residual drift, branch-conditioned growth, score component, branch/regulatory penalties. | `algorithm.py:140-276`; `algorithm.py:491-610`; `config.yaml:1-60` | completed | `BranchRegulatoryVelocityNet.forward` returns `A_psi x + b + r`; `BranchConditionedGrowthNet` shares the branch network; `BRGUSFMModel` registers velocity/growth/score modules; the loss hook adds branch CE, A sparsity, residual-bypass, Hutchinson/Jacobian, and smooth-A losses. | no_deviation | No direct W1/TMV/total-score/component metric predictor or static sklearn/Ridge/RandomForest model is trained. |
| P6 | Roll out from t0 with trained fields to produce a weighted stochastic positive-measure trajectory. | `algorithm.py:624-729`; `algorithm.py:1272-1354`; `config.yaml:76-83` | completed | `simulation_hook` starts from t0 particles with unit source weights, integrates score-corrected velocity plus fixed-seed stochastic noise, and updates log-weights by learned `growth_net`; `inference_context_builder` records no future observed rows/statistics or ground-truth inference use. | no_deviation | No target-count lookup, endpoint movement correction, or post-hoc weight rescaling is used for W1/TMV/trajectory evidence. |
| P7 | Export six DynBench files from model-native trajectory/fields/readouts, including the approved M1/M2 derivative-query exception. | `algorithm.py:758-841`; `algorithm.py:844-869`; `algorithm.py:978-1214`; `config.yaml:84-90` | completed | Export requires a trusted `EvaluationTrajectory`; holdout predictions use trajectory slices/weights; M1/M2 query trained fields at task-defined `train.h5ad` rows only for local derivative readouts; M4 uses t0-rooted final trajectory plus classifier/branch posterior; M5 uses matched control/knockdown root rollouts; M6 uses generated trajectory states for regulatory/growth-driver queries. | approved_exception | The M1/M2 observed-row derivative query exception is explicitly approved in `proposal_20260613T220752006803Z`; it does not construct rollout positions, weights, mass, fate, perturbation, held-out distributions, or GRN queries. |
| P8 | Evaluate generated outputs with allowed visible scorer/verifier and record numeric component diagnostics. | `algorithm.py:872-933`; `algorithm.py:1198-1214`; `algorithm.py:1356-1358`; `config.yaml:84-87`; `scripts/evaluation/score_dynbench_visible.py` | completed | `_run_public_dynbench_tools` searches only explicit/current-output/task roots, runs the scorer after files are written, parses numeric `total_score` and M1-M6 values, and writes nonnumeric provenance to `dynbench_diagnostics.json`; `evaluation_metrics_hook` returns these diagnostics. | no_deviation | Scorer output is evidence only; scorer code is not edited and generated files are not repaired after scoring. |

## Detailed current-source self-check

### P1 data contract
- Matrix/time extraction: `algorithm.py:54-66`.
- Case/task/output path resolution: `algorithm.py:69-127`.
- Gene-name inference: `algorithm.py:130-137`.
- Inference provenance: `algorithm.py:1272-1291`.
- Configured clean output/task/scorer roots: `config.yaml:84-87`.
- Semantic check: measured gene-space is used directly; no PCA/latent replacement or hidden case source is introduced.

### P2 adjacent endpoint plans
- UOT coupling class and bounded cost block: `algorithm.py:288-344`.
- Backend preparation hook: `algorithm.py:347-360`.
- Coupling-store branch target propagation: `algorithm.py:384-424`.
- Pair sampling with source/target ids: `algorithm.py:431-488`.
- Backend construction with `RegularizedUnbalancedConditionalPath` and `UOTMassStrategy`: `algorithm.py:1246-1270`.
- Coupling config: `config.yaml:61-75`.
- Semantic check: source is relaxed through UOT row masses while target mass is constrained through the package UOT path; pair ids are training-only and not inference inputs.

### P3 stochastic bridge samples
- Bridge sampling, conditional flow, mass target, and interval scaling: `algorithm.py:431-488`.
- Conditional path creation: `algorithm.py:1260-1263`.
- Score-enabled components/training strategy: `config.yaml:1-17`, `config.yaml:35-44`.
- Path sigma: `config.yaml:72-73`.
- Semantic check: stochasticity is learned through bridge/score targets and evaluated by the model-native rollout knobs; current evaluation config keeps both score correction and fixed-seed noise enabled (`score_weight=0.1`, `sigma=0.005`).

### P4 branch targets
- Label arrays and one-hot label handling: `algorithm.py:362-382`.
- Transport-DP target propagation without current-label anchoring: `algorithm.py:384-424`.
- Path/interpolation branch CE: `algorithm.py:507-534`.
- Node-level branch CE: `algorithm.py:536-567`.
- Branch loss config: `config.yaml:45-60`.
- Semantic check: branch targets are future-label distributions over non-root classes, not cell IDs or memorized root fate files.

### P5 trained fields and regulatory losses
- Regulatory drift decomposition `A x + b + residual`: `algorithm.py:140-201`.
- Branch-conditioned growth: `algorithm.py:204-236`.
- Model assembly/module ownership: `algorithm.py:239-276`.
- Branch/regulatory loss hook: `algorithm.py:491-610`.
- Configured loss weights and bounded samples: `config.yaml:45-60`.
- Semantic check: `A_psi` participates in the actual drift and is tied to sparsity/Jacobian/residual-bypass penalties rather than being export-only.

### P6 t0-rooted positive-measure rollout
- Score-corrected drift helper: `algorithm.py:624-633`.
- Deterministic/counterfactual helper with evaluation knobs: `algorithm.py:636-729`.
- Inference context: `algorithm.py:1272-1291`.
- Trusted stochastic simulation hook: `algorithm.py:1293-1354`.
- Evaluation rollout config: `config.yaml:76-83`.
- Semantic check: simulation starts from t0 particles and exogenous trajectory times; learned growth integrates log-weights from unit source masses; current evaluation config uses nonzero fixed-seed stochastic noise and score correction rather than the earlier deterministic drift-only diagnostic setting.

### P7 six-file DynBench export
- Trusted-trajectory guard: `algorithm.py:978-985`.
- Holdout prediction from `EvaluationTrajectory`: `algorithm.py:1007-1020`.
- M1/M2 task-row derivative query helper and field export: `algorithm.py:797-808`, `algorithm.py:1022-1033`.
- Generated GRN query helper and GRN/growth-driver export: `algorithm.py:811-841`, `algorithm.py:1125-1152`.
- Root fate from t0/root rollout slice plus classifier/branch posterior: `algorithm.py:1035-1068`.
- Perturbation from matched control/intervened root rollouts with direct measured-coordinate knockdown and terminal fate readout: `algorithm.py:1070-1123`.
- Provenance and numeric diagnostics: `algorithm.py:1162-1214`.
- Semantic check: all six files are generated from trained model artifacts, allowed task sidecars, and the approved M1/M2 field-query exception only.

### P8 evaluation and diagnostics
- Visible scorer/verifier runner: `algorithm.py:872-933`.
- Scorer invocation and parsed M1-M6 metrics: `algorithm.py:1198-1207`.
- Diagnostics sidecar write: `algorithm.py:1208-1214`.
- Metrics hook registration: `algorithm.py:1356-1358`.
- Local visible scorer path: `config.yaml:84-87`.
- Semantic check: `score_dynbench_visible.py` is an allowed current-output evaluator; it is invoked only after outputs are written and does not feed back into output construction.

## Claim metric/export contract

- Claim object: visible Remaining-Hard5 DynBench `total_score` with component floors over exactly the five locked native panel datasets listed in `supervisor/stage2_panel_datasets.json`.
- Gate authority: the framework-level locked Stage 2 gate remains unmodified. The local executable scorer path configured at `config.yaml:84-87` is used for run-local visible diagnostics and campaign custom metrics.
- Scorer timing: the scorer is invoked only after `holdout_prediction.csv`, `velocity_field.csv`, `growth_rates.csv`, `per_cell_fate.json`, `perturbation_results.json`, and `driver_genes.json` are written.

| Component | Prediction source | Truth / observed source used only by scorer | Grouping / timepoints | Proposal object / gap mapping | Leakage controls |
|---|---|---|---|---|---|
| `M1_velocity` | `velocity_field.csv` from `model.velocity_net` queried at train-row coordinates (`algorithm.py:1022-1033`). | Visible `ground_truth/<case>/velocity_ground_truth.csv` used only by scorer. | One row per training cell in task row order. | Tests learned local drift including sparse regulatory component. | Approved derivative-query exception only; rows do not construct rollout/mass/fate/perturbation/GRN. |
| `M2_growth` | `growth_rates.csv` from `model.growth_net` at train-row coordinates (`algorithm.py:1022-1033`). | Visible `ground_truth/<case>/full_metadata.csv:growth_rate` used only by scorer. | One row per training cell in task row order. | Tests learned state-dependent growth/mass field. | Same M1/M2 derivative-query boundary. |
| `M3_distribution` | `holdout_prediction.csv` from `EvaluationTrajectory` slices and weights (`algorithm.py:1007-1020`). | Visible held-out counts/metadata used only by scorer. | Target times from `prediction_targets.json` or default task times. | Tests model-native distribution rollout. | Requires trusted trajectory; no fallback scoring after timeout/missing trajectory. |
| `M4_fate` | `per_cell_fate.json` from t0/root rollout final slice plus classifier/branch posterior (`algorithm.py:1035-1068`). | Visible per-cell fate truth used only by scorer. | Root-cell IDs from t0 AnnData rows. | Tests branch/fate-basin posterior carried by rollout. | No root fate answer file is read for inference/export. |
| `M5_perturbation` | `perturbation_results.json` from matched control and direct z-score -5 knockdown root rollouts to terminal fate (`algorithm.py:1070-1147`). | Visible perturbation truth used only by scorer. | Per gene and fate class. | Tests structural response of learned drift/growth/fate readout. | Raw perturbed-minus-control deltas averaged over a small fixed set of stochastic replicas; no response damping or truth calibration. |
| `M6_grn` | `driver_genes.json` from mean `A_psi` and growth gradients on generated trajectory query states (`algorithm.py:811-841`, `algorithm.py:1125-1152`). | Visible GRN/growth-driver truth used only by scorer. | Directed gene pairs and gene-level growth drivers. | Tests sparse regulatory operator and growth-driver derivatives. | Query states are generated from `EvaluationTrajectory`, not future observed rows/counts. |
| `total_score` | Arithmetic aggregate of generated component files as parsed by the scorer (`algorithm.py:1198-1207`). | Six visible component truth sources used only by scorer. | Per case, then campaign mean over five native datasets. | Validates integrated six-output claim only when component floors pass. | Metric outputs never repair generated files or train direct score heads. |

## Risk watchlist

1. UOT/growth instability: monitor row masses, predicted total mass, W1/TMV, and particle-weight effective sample size.
2. Branch calibration/collapse: monitor branch entropy, calibration, fate entropy, and shuffled-label or anchor-ablation controls if available; M5 now uses `dynbench.perturbation_replicates` matched stochastic replicas to control finite-rollout variance without changing the perturbation operator.
3. Regulatory bypass: monitor residual ratio, A/Jacobian loss, smooth-A loss, A-disabled controls, and GRN seed stability.
4. Stochastic variance: compare stochastic evaluation trajectory and matched perturbation rollouts for seed sensitivity; record evaluation `sigma=0.005`, `score_weight=0.1`, and seed usage in trial diagnostics.
5. Multi-objective conflict: tune sparse config leaves only and record component tradeoffs.
6. Clean-run overfitting: continue static checks for prior artifacts, case-specific constants, and answer-file reads outside scorer-only paths.
7. Future scalability: block/revise if future panels require multi-block UOT without a semantics-preserving sparse/full-plan estimator.

## Files changed

- `/lustre/home/2501111653/.cellcompass/training_algorithms/brgusfm_rh5_0613/algorithm.py`
- `/lustre/home/2501111653/.cellcompass/training_algorithms/brgusfm_rh5_0613/config.yaml`
- `/lustre/home/2501111653/.cellcompass/training_algorithms/brgusfm_rh5_0613/manifest.yaml`
- `/lustre/home/2501111653/.cellcompass/training_algorithms/brgusfm_rh5_0613/IMPLEMENTATION_MAP.md`
- `/lustre/home/2501111653/CytoBridge-agent/cytobridge_output/dynbench_total_algorithm_remaininghard5_codex55_clean1/scripts/evaluation/score_dynbench_visible.py`
