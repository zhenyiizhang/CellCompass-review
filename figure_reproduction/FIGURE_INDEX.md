# Figure index

The numbering follows the six-main-figure manuscript and the current 14-figure SI.

**All entries below are covered by `python -m reproduce.build`.** The notes
describe the archived source history. The complete data-only rebuild map and
the reconstructed variants are documented in [REPRODUCTION.md](REPRODUCTION.md).

| Figure | Contents | Notes |
|---|---|---|
| Main 1 | CellCompass overview | Manual Illustrator artwork. No complete plotting script identified. |
| Main 2 | DynBench100 | Result-panel generators and checked alternatives. Final panel arrangement is in Illustrator. |
| Main 3 | EMT | Program trajectories, observed/generated comparisons and assembly. |
| Main 4 | PulseBridge | Panel generators plus Illustrator assembly. Current code may post-date the manuscript export. |
| Main 5 | LineageBridge | Panel generators plus Illustrator assembly. |
| Main 6 | ProteinBridge | Panel generators plus Illustrator assembly. |
| SI 1 | Model-design lifecycle | Generate the lifecycle diagram. The accepted PDF is in the SI repository. |
| SI 2 | EMT time courses | Observed versus generated EMT-core change from 0d, redrawn in `reproduce/si_plain.py` from the snapshot table. |
| SI 3 | EMT endpoint validation | Generated versus observed endpoint shifts, redrawn in `reproduce/si_plain.py` from the route summary. |
| SI 4 | EMT routes and program layers | Route phase diagram and program prevalence, redrawn in `reproduce/si_plain.py`. |
| SI 5 | PulseBridge direction and observed-state comparisons | Multiple component plots assembled in the SI. Check original output stems against accepted PDF. |
| SI 6 | PulseBridge perturbation controls | Random-control analysis and program perturbation plots. |
| SI 7 | State-matched fate divergence | Pocket selection reused from the archived script; drawn in `reproduce/si_lineage.py`. |
| SI 8 | Branch-mixture evidence | Calibration, ambiguity strata, collapse and per-fate recovery; drawn in `reproduce/si_lineage.py`. |
| SI 9 | Clone case studies | Selected clones with stored branch paths; drawn in `reproduce/si_lineage.py`. |
| SI 10 | Lineage program perturbations | Stacked terminal fractions and largest change per perturbation; drawn in `reproduce/si_lineage.py`. |
| SI 11 | Clone-supported fate predictions | Barcode-derived versus predicted fate probabilities for 1,728 eligible day-2 cells; drawn in `reproduce/si_lineage.py`. |
| SI 12 | Protein method comparison | Observed RNA/protein maps and generated terminal protein states per method; drawn in `reproduce/si_protein_worm.py`. |
| SI 13 | Protein quantitative checks | Mean/per-time protein energy distance, covariance, nearest-target ECDF, phenotype error and variance in one figure (`reproduce/si_protein_worm.py`). |
| SI 14 | Protein readout validation | Observed program–ADT correlations versus learned readout derivatives and binned trends; drawn in `reproduce/si_protein_worm.py`. |
