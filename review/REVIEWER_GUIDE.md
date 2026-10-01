# Guide to the code accompanying CellCompass

## Inspect and redraw the reported results

Run `python review/check_archive.py` to verify the packaged files. The manifest
covers source code and numerical plotting inputs. Run the figure command in the
root README to regenerate six main figures and fourteen supplementary figures.
The command rejects reads from the original workstation's study directories.
It verifies that all 20 PDF/PNG pairs were generated and records elapsed time.

`figure_reproduction/REPRODUCTION.md` records the plotting choices. In particular,
protein error is read from the recorded high-dimensional evaluation, not computed
from a two-dimensional UMAP. Frozen UMAP and streamline coordinates avoid rerunning
visualization fits. Figure 3e uses the seven-program September revision.
The accepted SI panels can be compared to the current Overleaf PDF; current source
versions are recorded in `metadata/source_versions.json`.

## Inspect the agent and model design

Start at `cytobridge_agent/runtime_v2/agent.py` and `tool_registry.py`.
The stage procedures live in `cytobridge_agent/skills/`, including
`algorithm/algorithm-orchestrator/`, `algorithm/authoring/` and
`algorithm/campaign-tuning/`. Independent proposal and implementation reviews,
staged comparisons and the final confirmation are implemented by the runtime.
The low-level names in the source are retained for compatibility.

The numerical extension interface is
`CytoBridge-main/CytoBridge/tl/training_algorithm.py`. Endpoint coupling,
conditional paths and mass rules are implemented in `flow_matching_backends.py`.
Custom implementations expose `build_training_algorithm(context)` and provide
their own data, model, loss, simulation and evaluation hooks as needed.

The archived custom models are:

| Paper name | Runtime identifier | Source selected |
|---|---|---|
| PulseBridge | `pulse_control_identifiability_bridge` | Archived workspace of locked campaign `campaign_20260620T224339184404Z` |
| LineageBridge | `clone_cuvgf` | Algorithm snapshot attached to final run `20260607T113528987593Z` |
| ProteinBridge | `cite_paired_coord_fm` | Algorithm snapshot attached to final run `20260526T174204321821Z` |
| BRGU-SFM | `brgusfm_rh5_0613` | Final evaluation snapshot `snapshot_20260616T090901103022Z` |

Each directory includes the implementation, configuration and manifest.
`review/smoke_runtime.py` loads these using the supplied registry. Historical
absolute paths in archived configuration files describe the original run. Set
`ckpt_dir` to a writable local output directory before fitting. Raw archived
scripts and configurations are not all portable command-line entry points.

## Data and a full refit

| Study | Public data | Prepared inputs needed by the model |
|---|---|---|
| EMT | [GSE147405](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE147405) | Separate induction conditions, `obs['time_point_processed']`, 50-dimensional `obsm['X_latent']` |
| PulseBridge | [GSE194357](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE194357) | Total/new/old RNA, control and labelled day-0 states, pulse directions and confidence, 30-dimensional state |
| LineageBridge | [GSE140802](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE140802) | Day-2/4/6 states, clone labels, terminal cell-type annotations and processed time |
| ProteinBridge | [GSE305370](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE305370) | Paired RNA and ADT, processed time, 50 RNA and 30 protein coordinates with modality metadata |

The original preprocessing scripts available locally are in `case_studies/`.
The model's data-builder function and manifest specify the exact fields it reads.
The accession links identify source data; they do not substitute for the authors'
prepared matrices, train/evaluation splits or trained checkpoints. Those larger
assets are not in this source ZIP. A complete retraining reproduction needs them
or their reconstruction with the stated preprocessing and splits.

Fitting fixed algorithms requires the modeling environment but no agent API.
Repeating autonomous model development additionally requires a configured LLM
provider. A new design run may select a different candidate. It is not a replay of
the saved design history. Optional literature retrieval requires a separately
prepared local corpus; literature PDFs, full-text indexes and pretrained embedding
caches have not been redistributed in this archive.

## DynBench

`benchmark/dynbench/simulators/` supplies synthetic data generation and
`benchmark/dynbench/eval/` implements the six scores. The original developer README
also describes older pilot scenarios. The paper's final BRGU-SFM 100-case list and
scores are in `case_studies/dynbench/final_evaluation/`; these identify the manuscript
evaluation rather than the pilot results in the developer README.

The full 100 prepared task packages and their hidden ground-truth arrays are not
bundled. The generator can create fresh scenarios. A fresh synthetic scenario is a
software demonstration and is not an exact reproduction of the paper's 100 cases.
Evaluation truth must remain outside any agent-accessible task directory when
running a comparison. The six scores are velocity, growth drivers, held-out
distribution, fate, perturbation response and regulatory recovery.

## Version and access

This snapshot is prepared for confidential review. The ZIP and Git tag identify
the review version. No credentials or personal conversation histories are included.
The journal can distribute the ZIP through its confidential submission system.
A private GitHub URL alone does not grant access. Named invitations can be managed
separately by the authors if the editor requests them.
