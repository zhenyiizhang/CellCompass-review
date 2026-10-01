#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path


SOURCE_SCRIPT = Path(
    "/lustre/home/2501111653/CytoBridge-agent/benchmark/results/"
    "total_score_algorithm_100case_component_eval/"
    "run_external_eval_100case_components.py"
)

source = SOURCE_SCRIPT.read_text(encoding="utf-8")
source = source.replace(
    "total_score_algorithm_100case_component_eval",
    "total_score_algorithm_100case_component_eval_final",
)
source = source.replace(
    "snapshot_20260615T015348659275Z",
    "snapshot_20260616T090901103022Z",
)
source = source.replace(
    "trial_20260615T015348563705Z",
    "trial_20260616T090900937187Z",
)
source = source.replace(
    "supervisor external 100-case eval copy of active best",
    "supervisor external 100-case eval copy of final locked release",
)
source = source.replace(
    "uses active-best snapshot code/config",
    "uses final locked-release snapshot code/config",
)

code = compile(source, str(SOURCE_SCRIPT), "exec")
globals_dict = {"__name__": "__main__", "__file__": str(SOURCE_SCRIPT)}
exec(code, globals_dict)
