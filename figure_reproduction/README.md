# Reproduce the figures

These scripts redraw six main figures and fourteen supplementary figures from
the included analysis data. They do not require an API key or model training.

Use a separate plotting environment because these dependencies differ from
those used by the agent. From the repository root:

```bash
python3.14 -m venv .venv-figures
source .venv-figures/bin/activate
python -m pip install -r figure_reproduction/requirements-reproduce.txt
python review/reproduce_figures.py
```

PDFs and PNG previews are saved to `output/review_figures/`. The script checks
that all 20 figures were generated. It refuses to overwrite an earlier run,
so move that output directory before running it again.

Numerical panels are plotted from saved tables, coordinates and trajectories.
Conceptual diagrams use the original Illustrator artwork. Some layouts differ
from the manually assembled manuscript figures.

`reproduce/` contains the plotting entry points, `plot_data/` contains compact
inputs, and `source/` contains the original analysis scripts. See
[the figure index](FIGURE_INDEX.md) and
[reproduction details](REPRODUCTION.md) for individual panels.
