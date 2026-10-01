[PLANNER_VIZ_BRIEF_AUTO]
Visualization intent is auto-inferred for this downstream turn. Treat this block as a rendering contract.

User goal:
{user_goal}

User instruction:
{user_instruction}

Structured Viz Brief (JSON):
```json
{viz_brief_json}
```

Execution contract:
1. Rerun the relevant analysis chain before plotting; do not only reuse old figures.
2. Visualization priority:
   - P1: rerun relevant CytoBridge analysis chain to refresh evidence
   - P2: custom redraw via `execute_python` + `save_pubfig`
   - P3: add extra CytoBridge APIs in `execute_python` when key evidence is still missing
3. Save figures with descriptive names and publication-quality exports.
4. Report policy default: `main_only` (core figure coverage, no appendix auto-generation).
5. If `basis_preference` is present (and especially when `strict_basis=true`), do not switch to another embedding basis silently.
[/PLANNER_VIZ_BRIEF_AUTO]
