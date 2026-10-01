# Validation of the review archive

Checked on 1 October 2026, on macOS arm64.

## Figure reproduction

A newly created Python 3.14 environment installed only
`figure_reproduction/requirements-reproduce.txt`. Running
`python review/reproduce_figures.py` generated all six main figures and fourteen
supplementary figures as PDF/PNG pairs in 293.56 seconds. Each PDF was checked for
a single page containing graphical content. Reads from the original development
repository, the separate figure repository and personal study directories were
blocked during this run. No LLM provider or model checkpoint was used.

This verifies redrawing from bundled final analysis inputs. It does not rerun
model fitting, establish the correctness of the biological conclusions, or test
pixel identity with manually assembled manuscript artwork. The current SI
excludes the three C. elegans figures present in the earlier plotting archive.

## Runtime smoke check

The packaged source was imported using the existing Python 3.10 scientific
environment, with the reference versions listed in
`requirements-runtime-constraints.txt`. This was not a fresh runtime installation.
The CLI help command succeeded. `python review/smoke_runtime.py` completed in
10.48 seconds on CPU. It checked endpoint coupling and flow matching for 32
synthetic cells and fitted a constant velocity with maximum endpoint error
1.67e-6. All four archived custom algorithms loaded and constructed their training
specifications through the supplied registry.

These checks do not exercise an LLM-driven design campaign or full biological
training. The prepared biological matrices, checkpoints and all 100 benchmark
ground-truth packages are not distributed in this archive. See the reviewer guide
for data sources and the inputs required for a full refit.

## File integrity and provenance

`metadata/review_manifest.json` identifies the supplied files by SHA-256.
`python review/check_archive.py` checks those hashes, parses the Python sources
and looks for selected token/private-key patterns. This is a limited credential
scan, not a security guarantee. Source origins and original hashes are retained
separately in `metadata/source_files.json`.

Packaging changes are confined to review documentation, verification scripts,
the active figure selection, and archive indexes. The four model implementations
are copied without modification. Internal package and algorithm identifiers are
preserved. The development repository and manuscript were not changed.

The GitHub Actions workflow is available for a manually requested Linux figure
check. It has not been run as part of this local validation.
