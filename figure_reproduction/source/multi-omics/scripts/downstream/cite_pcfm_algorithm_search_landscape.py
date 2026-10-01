#!/usr/bin/env python3
"""Backward-compatible entry point for the CITE-PCFM algorithm landscape figures."""

from __future__ import annotations

import runpy
from pathlib import Path


SCRIPT = Path(__file__).with_name("cite_pcfm_algorithm_search_landscape") / "make_algorithm_search_landscape.py"


if __name__ == "__main__":
    runpy.run_path(str(SCRIPT), run_name="__main__")
