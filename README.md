# CellCompass — code for peer review

Code accompanying **Building virtual cell dynamics that evolve with biological measurements**.
This snapshot accompanies `CellCompass_0808_new_pz_revised_v6.docx` and SI revision
`c378bdb`. The private repository is https://github.com/zhenyiizhang/CellCompass-review.

## Start here

The fastest review path redraws the six main figures and Supplementary Figures
1–14 from bundled analysis data. It needs a CPU, no API key and no model training.
Use a separate environment for this path because its plotting dependencies differ
from those of the original modeling runtime.

```bash
git clone https://github.com/zhenyiizhang/CellCompass-review.git
cd CellCompass-review
python3.14 -m venv .venv-figures
source .venv-figures/bin/activate
python -m pip install -r figure_reproduction/requirements-reproduce.txt
python review/check_archive.py
python review/reproduce_figures.py
```

The same commands work after extracting the review ZIP, starting at `cd`.
Windows activation is `.venv-figures\Scripts\activate`.
The output is `output/review_figures/`, with PDFs, PNG previews and a check report.
Numerical panels are redrawn from tables, coordinates and saved trajectories.
Conceptual diagrams and dialogue artwork use the archived vector masters.
These exports reproduce the analysis, not a pixel-identical copy of the edited
Word figures. The three C. elegans figures removed from the current SI are excluded.

## What is included

| Manuscript component | Source |
|---|---|
| Scientific agent, independent review and staged model development | `cytobridge_agent/` |
| Transport, bridge and flow-matching engine | `CytoBridge-main/CytoBridge/` |
| DynBench simulation, agent runners and six evaluation measures | `benchmark/` |
| BRGU-SFM | `algorithms/brgusfm_rh5_0613/` |
| EMT using the built-in velocity–growth model | `CytoBridge-main/CytoBridge/configs/vgfm.yaml`, `case_studies/emt/` |
| PulseBridge | `algorithms/pulse_control_identifiability_bridge/` |
| LineageBridge | `algorithms/clone_cuvgf/` |
| ProteinBridge | `algorithms/cite_paired_coord_fm/` |
| Analysis scripts, compact figure data and plotting | `figure_reproduction/` |
| Source revisions and file hashes | `metadata/` |

The Python package and some algorithm identifiers retain their development names.
`CytoBridge` in imports denotes the manuscript's extensible dynamical engine.
Renaming these identifiers would break the archived implementations.

## Modeling runtime

Python 3.10 is the reference runtime used for the local smoke checks. Install it
in a second environment. A CUDA-capable GPU is recommended for full model fitting.

```bash
conda create -n cellcompass-review python=3.10 -y
conda activate cellcompass-review
python -m pip install -c requirements-runtime-constraints.txt -r requirements-agent.txt
python -m pip install -c requirements-runtime-constraints.txt -e ./CytoBridge-main
python -m pip install -c requirements-runtime-constraints.txt -e .
python review/smoke_runtime.py
cellcompass --help
```

The runtime smoke check exercises transport and flow matching on synthetic data
and loads the four archived custom model implementations. It makes no LLM call.
For an interactive agent study, configure your own provider and model using the
CLI options in [the runtime guide](docs/runtime-upstream.md). An LLM-backed run
uses the reviewer's provider account and incurs that provider's normal charges.
The web interface is optional (`cd web && npm ci && npm run build`).

See [review instructions](review/REVIEWER_GUIDE.md) for data access, model sources,
validation scope and the distinction between a saved-result redraw, a fixed-model
refit and a new agent-driven design run. Full biological training matrices,
checkpoints, paid API credentials and literature full-text caches are not bundled.
Public dataset accession numbers and the required prepared-data fields are listed
in that guide. Full raw-data-to-result reproduction is not claimed by the quick check.

## Access during peer review

This repository is private. Its GitHub URL works only for authorized accounts.
The versioned ZIP is suitable for the journal's confidential supplementary-code
upload and does not require reviewers to disclose their GitHub identities.
See [submission wording](review/CODE_AVAILABILITY.txt).

The source revision records distinguish the current development-runtime snapshot
from the final case-specific algorithm snapshots. Local modifications present at
packaging are recorded by file hash. See `review/VALIDATION.md` for checks actually run.
Existing third-party notices are retained. No additional public license is granted
by preparing this private review archive.
