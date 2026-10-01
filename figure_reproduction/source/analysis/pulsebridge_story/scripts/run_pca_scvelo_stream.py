#!/usr/bin/env python3
"""Run the PulseBridge PCA-basis scVelo stream plot only."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import anndata as ad


def main() -> None:
    script = Path(__file__).resolve().parent / "pulsebridge_story_analysis.py"
    spec = importlib.util.spec_from_file_location("pulsebridge_story_analysis", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {script}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.ensure_inputs()
    adata = ad.read_h5ad(mod.ADATA_PATH if mod.ADATA_PATH.exists() else mod.PREPARED_H5AD_PATH, backed="r")
    payload = mod.plot_story_scvelo_velocity_stream_pca_only(adata)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
