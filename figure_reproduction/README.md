# Figure reproduction for the CellCompass review snapshot

Use the [root README](../README.md) and run `python review/reproduce_figures.py`
from the repository root. This edition contains the six main figures and
Supplementary Figures 1–14. It omits the retired C. elegans study.

The code and plotting inputs were exported from the private
`zhenyiizhang/CellCompass-figure-code` repository at commit `1acc110`.
Numerical panels are redrawn from bundled final analysis data. Conceptual and
dialogue panels use the retained Illustrator artwork. Model training and the
agent's design runs are not repeated by the plotting command.

`reproduce/` contains the rendering entry points, `plot_data/` contains compact
numerical inputs, and `source/` preserves original analysis and plotting scripts.
`metadata/` records provenance and hashes. Original analysis scripts may require
large external inputs; the documented redraw entry point does not.

See [reproduction choices](REPRODUCTION.md) for frozen display coordinates,
fitted-model outputs and conceptual artwork. Historical metadata and source
comments may refer to earlier figure sets. The root reviewer guide and the
generated list of 20 figures define the current review scope.
