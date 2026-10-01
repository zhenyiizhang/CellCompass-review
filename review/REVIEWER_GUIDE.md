# Models and data

For installation and a first analysis, see [Getting started](../docs/getting-started.md).

## CellCompass

The agent is implemented in `cytobridge_agent/`. Start with
`runtime_v2/agent.py` and `runtime_v2/tool_registry.py`. The scientific procedures
for proposing, reviewing and testing models are in `cytobridge_agent/skills/`.

The dynamical engine is in `CytoBridge-main/CytoBridge/`. Its extension interface,
`tl/training_algorithm.py`, lets a new method define its data preparation,
model, training objective, simulation and evaluation. Transport couplings and
flow-matching paths are implemented in `tl/flow_matching_backends.py`.
The source retains the `CytoBridge` package name for compatibility.

## Study implementations

| Method or study | Implementation | Experimental data |
|---|---|---|
| EMT | `CytoBridge-main/CytoBridge/configs/vgfm.yaml`, `case_studies/emt/` | [GSE147405](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE147405), time-course scRNA-seq |
| PulseBridge | `algorithms/pulse_control_identifiability_bridge/` | [GSE194357](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE194357), total, newly synthesized and pre-existing RNA |
| LineageBridge | `algorithms/clone_cuvgf/` | [GSE140802](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE140802), RNA and clone barcodes |
| ProteinBridge | `algorithms/cite_paired_coord_fm/` | [GSE305370](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE305370), paired RNA and surface proteins |
| BRGU-SFM | `algorithms/brgusfm_rh5_0613/` | DynBench100 |

Each custom method includes `algorithm.py`, `config.yaml` and `manifest.yaml`.
The data-loading code specifies the additional AnnData fields used by that
method. Available preprocessing scripts are in `case_studies/`.
Configurations from the original fits retain their original paths. Update data
and output paths before rerunning them.

EMT uses separate induction conditions and 50-dimensional RNA coordinates.
PulseBridge uses a 30-dimensional state with RNA-labeling directions and their
confidence. LineageBridge uses day-2, day-4 and day-6 samples with clone labels
and terminal annotations. ProteinBridge uses 50 RNA and 30 protein coordinates
with paired-modality metadata.

The repository includes source code and compact data for figure reproduction.
The full prepared biological matrices and trained checkpoints are not included.
Refitting the study models therefore requires preparing the public data with
the preprocessing and evaluation splits described in the paper and SI.

## DynBench100

`benchmark/dynbench/simulators/` contains scenario generators and
`benchmark/dynbench/eval/` contains the six evaluation measures: velocity,
growth drivers, held-out distribution, fate, perturbation response and
regulatory recovery. The final BRGU-SFM 100-case evaluation list and scores are
in `case_studies/dynbench/final_evaluation/`.

The complete 100 prepared task packages and hidden ground-truth arrays are not
included. The generators can create new scenarios for testing. When evaluating
an agent, keep ground truth outside its accessible task directory.

## Figures and software checks

See [figure reproduction](../figure_reproduction/README.md) for the six main
figures and fourteen supplementary figures. This redraws saved analysis results
without refitting models or calling a language model.

[Software checks](VALIDATION.md) record what has been tested.
`metadata/source_versions.json` and `metadata/source_files.json` record the
origins of the supplied implementations.
