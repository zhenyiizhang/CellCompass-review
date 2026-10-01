# Figure sources and reproduction choices

The final analysis is held fixed. The new entry points load drawing functions
from the original scripts, substitute repository-relative inputs, and suppress
imports and entry points used only for model fitting or analysis. This is not a
general-purpose execution sandbox. Missing inputs cause a build failure.

## Figure-to-input map

| Figures | Bundled input | Rebuild function in `reproduce/build.py` |
|---|---|---|
| Main 1, SI 1 | Conceptual artwork (Main 1 stored vector; SI 1 drawn from the caption in `reproduce/si_schematic.py`) | `main_figures`, `si_1` |
| Main 2 | `source/dynbench_results_share_20260618_170313/` result tables | `main_figures` |
| Main 3, SI 2–4 | `source/emt/downstream/` program, endpoint and route tables | `emt`, `main_figures` |
| Main 4, SI 5–6 | `source/analysis/pulsebridge_story/outputs/tables/` and `plot_data/pulse/` | `pulse`, `pulse_main` |
| Main 5, SI 7–11 | `source/mouse_new/downstream/cellcompass_clone_case/` and `plot_data/lineage/` | `lineage`, `lineage_main` |
| Main 6, SI 12–14 | `source/multi-omics/downstream/`, selected final evaluation JSON files and `plot_data/protein/` | `protein_main`, `protein_tables`, `protein_distribution` |

Original source paths and file hashes are recorded in
`metadata/file_manifest.csv`. Exported compact inputs have provenance notes in
`plot_data/`. These source paths document origin; the rendering command does
not open them. `metadata/plot_data_manifest.json` records the bundled input
hashes. No raw RNA matrix or model checkpoint is needed for the redraw.

## Explicit reconstruction choices

- Figure 3e, corrected 19 September 2026: seven pathway/state programs are shown
  at the author's request. The cell-cycle RNA score is omitted from this panel
  only, without changing the source data or other analyses. Increase/decrease
  symbols are explained in a compact note. Negative values stay on their
  assigned program directions, using dashed lines and open circles. Both signs
  use one common absolute-score scale, with no negative-value display cutoff.
  The 84 displayed scores are unchanged. `tools/check_emt_fingerprint.py` checks
  category counts, score preservation, directions, lengths and sign encoding.
  The original source snapshot is retained unchanged; the corrected renderer is
  `reproduce/emt_fingerprint.py`.

- SI 11: the eligible-only variant is reconstructed from the saved per-cell
  truth and predicted fate probabilities. It uses the same 1,728 eligible early
  cells, four rows and six selected fates. Cell ordering is checked, and each
  fate retains its own original probability scale. This replaces reliance on
  an untraced export script, not on missing predictions.
- SI 10: the manuscript's stacked bars and text summary are reconstructed from
  the saved perturbation fractions. The newer source script's additional
  lollipop comparison is not added to the manuscript panel set.
- Protein panels: final locked per-time protein energy distances are used for
  both the comparison and the reported mean. No distance is recomputed from
  two-dimensional display coordinates. The mean is approximately 0.02220 for
  ProteinBridge and 0.02558 for CRUFM.
- Protein display coordinates were regenerated once using the original UMAP
  settings (seed 20260528, 35 neighbours, minimum distance 0.22, Euclidean
  metric, UMAP 0.5.12). Observed cells, target-time generated states and selected
  trajectories use the same fitted projection. These coordinates are bundled,
  so UMAP is not rerun on each clone. This is a visualization-only projection,
  not a dynamical-model refit, and is not claimed to be bitwise identical to
  the historical embedding.
- PulseBridge streams use the bundled numerical grid produced from the saved
  latent velocities by Dynamo 1.5.3 with the original Pearson projection and
  Gaussian streamline settings. Matplotlib redraws the streams from that grid.
  No stream pixels are extracted from an existing image.
- Conceptual and dialogue artwork is cropped only from designated portions of
  the editable vector masters. Numerical result panels are regenerated. SI 1
  and Main 1 are entirely conceptual and retain their vector originals.
- Dense background scatter layers retain rasterization where the original
  plotting functions used it; axes, curves and text remain vector elements.

The Illustrator masters contain historical names and author edits. The renderer
does not rewrite those diagrams or alter manuscript text. The original files
remain separate from generated figures, so later artwork corrections can be
made without changing the numerical evidence.

## Environment and verification

Use `requirements-reproduce.txt` for the supported redraw environment. The
other requirements lists belong to the wider source archive and are not needed
for this command. The rebuild does not need PyTorch, AnnData, UMAP or Dynamo.
The maintainer export scripts do need study-specific packages and original
data, but their outputs are already included.

The automated check covers all 20 exports and rejects original-workstation
study reads. It is not a new scientific validation of the fitted models and
does not imply identical PDF hashes across operating systems or font versions.

## Supplementary panel styling

Supplementary Figures 1-14 are drawn by the plain-style modules
`reproduce/si_schematic.py`, `reproduce/si_plain.py`, `reproduce/si_lineage.py`
and `reproduce/si_protein_worm.py`. They consume the same frozen tables and
arrays as the archived scripts and change presentation only (Arial, black
text, a small fixed palette, frameless legends, no in-figure subtitles).
Supplementary Figure 13 combines the distribution-error summary and the
biological diagnostics on one page. The retired C. elegans panels are excluded
from this review snapshot.
