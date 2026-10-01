"""Trajectory plotting bundle wrappers."""
from __future__ import annotations

from pathlib import Path
from typing import Dict


def plot_sde_trajectories_bundle(adata, output_dir: str, color_key: str = None) -> Dict[str, object]:
    from CytoBridge.pl.plot import plot_sde_trajectories

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    plot_sde_trajectories(
        adata=adata,
        output_path=str(outdir),
        color_key=color_key,
    )
    artifacts = {}
    for file_name in ["SDE_Trajectories_Plot.png"]:
        p = outdir / file_name
        if p.exists():
            artifacts[file_name] = str(p)
    return {
        "space_used": "latent",
        "projection_backend": "none",
        "artifacts": artifacts,
        "warnings": [],
    }
