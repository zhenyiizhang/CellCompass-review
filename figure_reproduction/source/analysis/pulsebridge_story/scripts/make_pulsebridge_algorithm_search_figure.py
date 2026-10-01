#!/usr/bin/env python3
"""PulseBridge proposal and tuning history figure.

This figure is intentionally separate from the biology figures.  It summarizes
the CellCompass search process: proposal revisions, PulseBridge tuning trials,
and the final PCIE comparison.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pulsebridge_algorithm_search_mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

try:
    import yaml
except Exception:  # pragma: no cover - only used as a robust local fallback.
    yaml = None


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
TAB = ROOT / "outputs" / "tables"
FIG = ROOT / "outputs" / "figures"
REMOTE = REPO / "downloads" / "pulsebridge_remote_history_20260629" / "pulsebridge_history_export_20260629"

INK = "#111827"
MUTED = "#5B6673"
GRID = "#E5E7EB"
LIGHT_GRID = "#F3F4F6"
BLUE = "#245B8F"
TEAL = "#168E91"
PURPLE = "#6D28D9"
LAVENDER = "#F3F0FF"
GREEN = "#2F7D4E"
GOLD = "#C58A16"
RED = "#B23A48"
GRAY = "#94A3B8"
PALE_YELLOW = "#FFF8E6"
PALE_GREEN = "#EAF7F0"
PALE_PURPLE = "#F5F0FF"


FALLBACK_BASELINES = [
    {"algorithm": "PulseBridge", "pcie": 0.680449, "kind": "candidate"},
    {"algorithm": "WFR-FM", "pcie": 0.988217, "kind": "strict builtin"},
    {"algorithm": "SF2M", "pcie": 0.997507, "kind": "builtin"},
    {"algorithm": "Unbalanced OT", "pcie": 1.006741, "kind": "builtin"},
    {"algorithm": "VGFM", "pcie": 1.008919, "kind": "builtin"},
    {"algorithm": "CRUFM", "pcie": 1.010840, "kind": "builtin"},
    {"algorithm": "Cyto simulation", "pcie": 1.014287, "kind": "builtin"},
    {"algorithm": "Dynamical OT", "pcie": 1.015928, "kind": "builtin"},
    {"algorithm": "RUOT", "pcie": 1.021014, "kind": "builtin"},
]


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 8.2,
            "axes.labelsize": 8.0,
            "axes.titlesize": 10.0,
            "axes.titleweight": "bold",
            "axes.edgecolor": INK,
            "axes.linewidth": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 7.3,
            "ytick.labelsize": 7.3,
            "legend.fontsize": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel_label(ax: plt.Axes, letter: str, x: float = -0.08, y: float = 1.08) -> None:
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=14, fontweight="bold", color=INK, va="top")


def clean_grid(ax: plt.Axes, axis: str = "y") -> None:
    ax.grid(axis=axis, color=GRID, lw=0.75)
    ax.set_axisbelow(True)


def add_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = PURPLE, lw: float = 2.2) -> None:
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        lw=lw,
        color=color,
        shrinkA=7,
        shrinkB=9,
        zorder=2,
    )
    ax.add_patch(arr)


def read_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def load_remote_proposals() -> pd.DataFrame:
    attempts = read_jsonl(REMOTE / "research_idea" / "registry" / "attempts.jsonl")
    proposal_ids: list[str] = []
    for row in attempts:
        proposal_id = row.get("proposal_id")
        text = str(row.get("feedback") or row.get("reason") or row.get("summary") or "")
        if proposal_id and "Created algorithm proposal" in text and proposal_id not in proposal_ids:
            proposal_ids.append(proposal_id)
    if not proposal_ids:
        proposal_ids = [
            "proposal_20260620T183011352200Z",
            "proposal_20260620T201531810408Z",
            "proposal_20260620T213736784915Z",
        ]

    issue_by_order = [
        "campaign trials produced no usable metrics",
        "absolute evaluator debug; stale campaign evidence",
        "reviewed PCIE evaluator and endpoint-valid residual selector",
    ]
    xs = [1.0, 3.7, 6.5]
    ys = [0.34, 0.47, 0.78]
    rows = []
    for i, proposal_id in enumerate(proposal_ids, start=1):
        rows.append(
            {
                "proposal_order": i,
                "proposal_id": proposal_id,
                "short_label": f"proposal v{i}",
                "decision": "approve",
                "main_issue": issue_by_order[min(i - 1, len(issue_by_order) - 1)],
                "restart_or_repair": "locked path" if i == len(proposal_ids) else "restart",
                "x": xs[min(i - 1, len(xs) - 1)],
                "y": ys[min(i - 1, len(ys) - 1)],
                "source": str(REMOTE / "research_idea" / "registry" / "attempts.jsonl"),
            }
        )
    return pd.DataFrame(rows)


def load_remote_previews() -> pd.DataFrame:
    files = sorted((REMOTE / "runtime_previews").glob("pulse_control_identifiability_bridge_pilot_*.preview.json"))
    rows = []
    for i, path in enumerate(files, start=1):
        data = read_json(path)
        rows.append(
            {
                "preview_order": i,
                "preview_id": data.get("preview_id", path.stem),
                "ok": bool(data.get("ok")),
                "stage": data.get("stage"),
                "algorithm_id": data.get("algorithm_id") or data.get("training_algorithm_id"),
                "source": str(path),
            }
        )
    return pd.DataFrame(rows)


def load_remote_campaign_summary() -> pd.DataFrame:
    rows = []
    for path in sorted((REMOTE / "algorithm_campaigns").glob("campaign_*/campaign.json")):
        data = read_json(path)
        rows.append(
            {
                "campaign_id": data.get("campaign_id"),
                "status": data.get("status"),
                "algorithm_validated": data.get("algorithm_validated"),
                "proposal_id": data.get("proposal_id"),
                "created_at": data.get("created_at"),
                "updated_at": data.get("updated_at"),
                "archived_reason": data.get("archived_reason", ""),
                "source": str(path),
            }
        )
    return pd.DataFrame(rows)


def locked_campaign_dir() -> Path:
    candidates = []
    for path in sorted((REMOTE / "algorithm_campaigns").glob("campaign_*/campaign.json")):
        data = read_json(path)
        if data.get("status") == "locked" and data.get("algorithm_validated") is True:
            candidates.append(path.parent)
    if not candidates:
        candidates = sorted((REMOTE / "algorithm_campaigns").glob("campaign_*"))
    if not candidates:
        raise FileNotFoundError(f"No campaign records found under {REMOTE / 'algorithm_campaigns'}")
    return candidates[-1]


def find_run_dir(run_id: str) -> Path | None:
    if not run_id:
        return None
    matches = sorted((REMOTE / "training_runs").glob(f"{run_id}*"))
    return matches[0] if matches else None


def load_pcib_config(run_id: str) -> dict:
    run_dir = find_run_dir(run_id)
    if run_dir is None:
        return {}
    cfg = run_dir / "algorithm_snapshot" / "config.yaml"
    if not cfg.exists():
        return {}
    text = cfg.read_text()
    if yaml is not None:
        try:
            return (yaml.safe_load(text) or {}).get("pcib", {}) or {}
        except Exception:
            pass
    parsed = {}
    for key in ["pulse_cost_weight", "residual_scale", "velocity_loss_weight", "nuisance_cost_weight", "residual_cost_weight"]:
        m = re.search(rf"^\s*{re.escape(key)}:\s*([-+0-9.eE]+)\s*$", text, re.MULTILINE)
        if m:
            parsed[key] = float(m.group(1))
    return parsed


def load_remote_trials() -> pd.DataFrame:
    campaign_dir = locked_campaign_dir()
    rows = []
    for order, path in enumerate(sorted((campaign_dir / "trials").glob("trial_*.json")), start=1):
        data = read_json(path)
        metrics = data.get("metrics_summary") or {}
        custom = metrics.get("custom_metrics") or {}
        run_ids = data.get("run_ids") or []
        run_id = run_ids[0] if run_ids else ""
        pcib = load_pcib_config(run_id)
        pcie = custom.get("PCIE", custom.get("pcib_pulse_control_identifiability_error", np.nan))
        rows.append(
            {
                "trial_order": order,
                "trial_id": data.get("trial_id"),
                "campaign_id": data.get("campaign_id"),
                "stage": data.get("stage", "").replace("_", " "),
                "decision": data.get("decision"),
                "status": data.get("status"),
                "run_id": run_id,
                "w1_mean": metrics.get("w1_mean", np.nan),
                "pcie": pcie,
                "pulse_cost_weight": pcib.get("pulse_cost_weight", np.nan),
                "residual_scale": pcib.get("residual_scale", np.nan),
                "velocity_loss_weight": pcib.get("velocity_loss_weight", np.nan),
                "source": str(path),
            }
        )
    return pd.DataFrame(rows)


def timestamp_from_id(text: str) -> str:
    match = re.search(r"(20\d{6}T\d{12,})", text or "")
    return match.group(1) if match else ""


def run_metrics(run_dir: Path) -> tuple[float, float]:
    metrics_path = run_dir / "artifacts" / "metrics.json"
    if not metrics_path.exists():
        return np.nan, np.nan
    data = read_json(metrics_path)
    custom = data.get("custom_metrics") or {}
    w1 = data.get("w1_scores") or []
    w1_mean = float(np.mean(w1)) if w1 else np.nan
    pcie = custom.get("PCIE", custom.get("pcib_pulse_control_identifiability_error", np.nan))
    return w1_mean, float(pcie) if pcie is not None and np.isfinite(pcie) else np.nan


def load_full_lifecycle_events(trials: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
    """Build a single ordered history from remote campaign trials and run metrics."""
    wfrfm_pcie = float(baselines.loc[baselines["algorithm"].eq("WFR-FM"), "pcie"].iloc[0])
    events: list[dict] = []
    tied_run_ids = set(str(r) for r in trials["run_id"].dropna() if str(r))

    # Archived campaign stage-1 rejects are part of the real search history even
    # though they produced no usable metrics.
    for path in sorted((REMOTE / "algorithm_campaigns").glob("campaign_*/trials/trial_*.json")):
        data = read_json(path)
        if data.get("campaign_id") == locked_campaign_dir().name:
            continue
        metrics = data.get("metrics_summary") or {}
        custom = metrics.get("custom_metrics") or {}
        pcie = custom.get("PCIE", custom.get("pcib_pulse_control_identifiability_error", np.nan))
        events.append(
            {
                "time_key": data.get("created_at") or timestamp_from_id(data.get("trial_id", "")),
                "event_type": "archived stage1 reject",
                "stage_group": "stage1 setup",
                "stage": str(data.get("stage", "")).replace("_", " "),
                "decision": data.get("decision"),
                "status": data.get("status"),
                "run_id": "",
                "trial_id": data.get("trial_id"),
                "w1_mean": metrics.get("w1_mean", np.nan),
                "pcie": pcie,
                "pulse_cost_weight": np.nan,
                "residual_scale": np.nan,
                "velocity_loss_weight": np.nan,
                "source": str(path),
            }
        )

    # Every remote PulseBridge training metric is plotted. Runs tied to a formal
    # locked-campaign trial inherit the trial decision; untied runs remain visible
    # as diagnostic or extra training attempts.
    trial_by_run = {str(row["run_id"]): row for _, row in trials.iterrows() if str(row.get("run_id", ""))}
    for run_dir in sorted((REMOTE / "training_runs").glob("*pulse_control_identifiability_bridge*")):
        run_id = run_dir.name
        w1_mean, pcie = run_metrics(run_dir)
        pcib = load_pcib_config(run_id)
        trial_row = trial_by_run.get(run_id)
        if trial_row is not None:
            stage = str(trial_row["stage"])
            decision = str(trial_row["decision"])
            status = str(trial_row["status"])
            trial_id = str(trial_row["trial_id"])
            if stage.startswith("stage1") or stage.startswith("stage2"):
                stage_group = "stage1-2 claim"
            else:
                stage_group = "stage3 tuning"
            event_type = f"formal {decision}"
        else:
            stage = "manual diagnostic" if "__final__" in run_id else "extra training run"
            decision = "diagnostic"
            status = "completed"
            trial_id = ""
            if "__final__" in run_id:
                stage_group = "debug repair"
            elif timestamp_from_id(run_id) < "20260620T232500":
                stage_group = "stage1-2 claim"
            else:
                stage_group = "stage3 tuning"
            event_type = "manual diagnostic" if "__final__" in run_id else "extra metric run"
        events.append(
            {
                "time_key": timestamp_from_id(run_id),
                "event_type": event_type,
                "stage_group": stage_group,
                "stage": stage,
                "decision": decision,
                "status": status,
                "run_id": run_id,
                "trial_id": trial_id,
                "w1_mean": w1_mean,
                "pcie": pcie,
                "pulse_cost_weight": pcib.get("pulse_cost_weight", np.nan),
                "residual_scale": pcib.get("residual_scale", np.nan),
                "velocity_loss_weight": pcib.get("velocity_loss_weight", np.nan),
                "source": str(run_dir),
                "formal_trial": run_id in tied_run_ids,
            }
        )

    # Aborted locked-campaign trials have no run metrics but should still appear.
    for _, row in trials[trials["decision"].eq("aborted")].iterrows():
        events.append(
            {
                "time_key": timestamp_from_id(str(row["trial_id"])),
                "event_type": "aborted formal trial",
                "stage_group": "stage3 tuning",
                "stage": row["stage"],
                "decision": row["decision"],
                "status": row["status"],
                "run_id": "",
                "trial_id": row["trial_id"],
                "w1_mean": np.nan,
                "pcie": np.nan,
                "pulse_cost_weight": np.nan,
                "residual_scale": np.nan,
                "velocity_loss_weight": np.nan,
                "source": row["source"],
                "formal_trial": True,
            }
        )

    df = pd.DataFrame(events).sort_values("time_key").reset_index(drop=True)
    df["attempt_order"] = np.arange(1, len(df) + 1)
    df["pcie_reduction_vs_wfrfm_pct"] = (wfrfm_pcie - df["pcie"]) / wfrfm_pcie * 100.0
    best = []
    current = -np.inf
    for value in df["pcie_reduction_vs_wfrfm_pct"]:
        if np.isfinite(value):
            current = max(current, float(value))
        best.append(current if np.isfinite(current) else np.nan)
    df["best_so_far"] = best
    return df


def load_baselines() -> pd.DataFrame:
    metrics_path = TAB / "existing_metric_summary_long.csv"
    if metrics_path.exists():
        df = pd.read_csv(metrics_path)
        pcie = df[df["metric"].eq("pcie")].copy()
        if not pcie.empty:
            pcie = pcie.rename(columns={"value": "pcie"})
            pcie["kind"] = np.where(pcie["algorithm"].eq("PulseBridge"), "candidate", "builtin/reference")
            pcie["source"] = str(metrics_path)
            return pcie[["algorithm", "pcie", "kind", "source"]]
    fallback = pd.DataFrame(FALLBACK_BASELINES)
    fallback["source"] = "fallback_from_remote_evidence_ledger_values"
    return fallback


def stage_label(stage: str, stage_group: str = "", event_type: str = "") -> str:
    stage = str(stage)
    stage_group = str(stage_group)
    event_type = str(event_type)
    if event_type == "manual diagnostic":
        return "debug"
    if stage.startswith("stage1"):
        return "stage1"
    if stage.startswith("stage2"):
        return "stage2"
    if stage.startswith("stage3"):
        return "stage3"
    if stage_group == "stage1-2 claim":
        return "stage2"
    if stage_group == "stage3 tuning":
        return "stage3"
    return "debug"


def stage_pass_counts(events: pd.DataFrame) -> pd.DataFrame:
    formal = events[events["formal_trial"].eq(True)].copy()
    formal["stage_label"] = [stage_label(s, g, e) for s, g, e in zip(formal["stage"], formal["stage_group"], formal["event_type"])]
    rows = []
    for stage in ["stage1", "stage2", "stage3"]:
        sub = formal[formal["stage_label"].eq(stage)]
        numeric = sub.dropna(subset=["pcie"])
        rows.append(
            {
                "stage": stage,
                "formal_trials": len(sub),
                "promote_pass": int(sub["decision"].eq("promote").sum()),
                "reject": int(sub["decision"].eq("reject").sum()),
                "aborted": int(sub["decision"].eq("aborted").sum()),
                "numeric_trials": len(numeric),
                "pcie_claim_success_ge_10pct": int(numeric["pcie_reduction_vs_wfrfm_pct"].ge(10).sum()),
            }
        )
    return pd.DataFrame(rows)


def write_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    TAB.mkdir(parents=True, exist_ok=True)
    proposals = load_remote_proposals()
    previews = load_remote_previews()
    campaigns = load_remote_campaign_summary()
    trials = load_remote_trials()
    baselines = load_baselines()
    wfrfm_pcie = float(baselines.loc[baselines["algorithm"].eq("WFR-FM"), "pcie"].iloc[0])
    trials["pcie_reduction_vs_wfrfm_pct"] = (wfrfm_pcie - trials["pcie"]) / wfrfm_pcie * 100.0
    full_events = load_full_lifecycle_events(trials, baselines)
    counts = stage_pass_counts(full_events)
    proposals.to_csv(TAB / "pulsebridge_algorithm_proposal_revisions.csv", index=False)
    previews.to_csv(TAB / "pulsebridge_algorithm_preview_history.csv", index=False)
    campaigns.to_csv(TAB / "pulsebridge_algorithm_campaign_summary.csv", index=False)
    trials.to_csv(TAB / "pulsebridge_algorithm_tuning_trials.csv", index=False)
    full_events.to_csv(TAB / "pulsebridge_algorithm_full_lifecycle_events.csv", index=False)
    counts.to_csv(TAB / "pulsebridge_algorithm_stage_pass_counts.csv", index=False)
    baselines.to_csv(TAB / "pulsebridge_algorithm_pcie_baselines.csv", index=False)
    return proposals, previews, campaigns, trials, full_events, counts, baselines


def plot_panel_a(ax: plt.Axes, proposals: pd.DataFrame, previews: pd.DataFrame, campaigns: pd.DataFrame) -> None:
    ax.set_title("Proposal revisions and restarts")
    ax.set_xlim(0, 8.2)
    ax.set_ylim(0, 1.04)

    bands = [
        (0.0, 2.5, PALE_YELLOW, "proposal v1"),
        (2.5, 5.25, PALE_GREEN, "proposal v2"),
        (5.25, 8.2, PALE_PURPLE, "proposal v3"),
    ]
    for x0, x1, color, label in bands:
        ax.axvspan(x0, x1, color=color, zorder=0)
        ax.text((x0 + x1) / 2, 0.98, label, ha="center", va="top", fontsize=8.0, color=MUTED, fontweight="bold")

    xs = proposals["x"].to_numpy()
    ys = proposals["y"].to_numpy()
    for start, end in zip(zip(xs[:-1], ys[:-1]), zip(xs[1:], ys[1:])):
        add_arrow(ax, start, end, color=PURPLE, lw=2.3)

    for _, row in proposals.iterrows():
        color = GREEN if row["restart_or_repair"] == "locked path" else GOLD
        ax.scatter(row["x"], row["y"], s=185, color=color, edgecolor="white", lw=1.8, zorder=3)
        ax.text(row["x"], row["y"], str(int(row["proposal_order"])), ha="center", va="center", color="white", fontsize=9.2, fontweight="bold", zorder=4)

    failed_x = [0.65, 1.35, 2.05, 3.08, 4.35]
    failed_y = [0.17, 0.23, 0.18, 0.31, 0.28]
    ax.scatter(failed_x, failed_y, marker="x", s=46, color=RED, lw=1.6, zorder=3, label="failed / stale path")
    for x, y in zip(failed_x[:3], failed_y[:3]):
        ax.plot([x, 0.98], [y, 0.31], color="#D6B66A", lw=0.8, alpha=0.8, zorder=1)
    for x, y in zip(failed_x[3:], failed_y[3:]):
        ax.plot([x, 3.7], [y, 0.45], color="#B5D0C0", lw=0.8, alpha=0.8, zorder=1)

    preview_y = 0.06
    preview_x = np.linspace(0.55, 7.65, max(len(previews), 1))
    for x, (_, item) in zip(preview_x, previews.iterrows()):
        if bool(item["ok"]):
            ax.scatter(x, preview_y, s=20, color="#CBD5E1", edgecolor="white", lw=0.4, zorder=3)
        else:
            ax.scatter(x, preview_y, s=32, marker="x", color=RED, lw=1.3, zorder=4)
    preview_pass = int(previews["ok"].sum()) if "ok" in previews else 0
    ax.text(7.95, preview_y, f"{len(previews)} previews\n{preview_pass} pass", ha="right", va="center", fontsize=7.0, color=MUTED)

    archived_count = int(campaigns["status"].eq("archived").sum()) if "status" in campaigns else 0
    locked_count = int(campaigns["status"].eq("locked").sum()) if "status" in campaigns else 0

    ax.text(
        0.18,
        0.88,
        f"{len(proposals)} proposal versions\n= initial + {max(len(proposals) - 1, 0)} revisions",
        ha="left",
        va="top",
        fontsize=8.0,
        color=INK,
        fontweight="bold",
    )
    ax.text(0.40, 0.52, "no usable\nmetrics", ha="left", va="center", fontsize=7.1, color=RED)
    ax.text(2.95, 0.63, "evaluator URI /\nstale evidence", ha="left", va="center", fontsize=7.1, color=GOLD)
    ax.text(5.85, 0.50, "reviewed PCIE evaluator\nendpoint-valid residuals", ha="left", va="center", fontsize=7.1, color=PURPLE)
    ax.text(6.92, 0.84, f"{locked_count} locked\ncampaign", ha="left", va="center", fontsize=7.1, color=GREEN, fontweight="bold")
    ax.text(3.45, 0.16, f"{archived_count} stale campaigns\narchived", ha="center", va="center", fontsize=7.0, color=MUTED)

    ax.set_xlabel("CellCompass proposal lifecycle")
    ax.set_ylabel("evidence readiness")
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_yticklabels(["preview", "repair", "locked"])
    ax.set_xticks([])
    panel_label(ax, "A")
    clean_grid(ax)


def plot_panel_b(ax: plt.Axes, events: pd.DataFrame, counts: pd.DataFrame) -> None:
    ax.set_title("PulseBridge PCIE across 7 formal campaign trials")
    formal = events[events["formal_trial"].eq(True)].copy()
    formal = formal.sort_values("attempt_order").reset_index(drop=True)
    formal["formal_order"] = np.arange(1, len(formal) + 1)
    metric = formal.dropna(subset=["pcie_reduction_vs_wfrfm_pct"]).copy()
    metric["metric_order"] = metric["formal_order"]
    metric["metric_best_so_far"] = metric["pcie"].cummin()
    metric["display_stage"] = [stage_label(s, g, e) for s, g, e in zip(metric["stage"], metric["stage_group"], metric["event_type"])]
    wfrfm_pcie = float(load_baselines().loc[lambda d: d["algorithm"].eq("WFR-FM"), "pcie"].iloc[0])
    success_cutoff = 0.90 * wfrfm_pcie

    stage_specs = [
        ("stage1", PALE_YELLOW, "stage1"),
        ("stage2", PALE_GREEN, "stage2"),
        ("stage3", PALE_PURPLE, "stage3"),
    ]
    for stage, color, label in stage_specs:
        sub = metric[metric["display_stage"].eq(stage)]
        if sub.empty:
            continue
        x0 = float(sub["metric_order"].min()) - 0.5
        x1 = float(sub["metric_order"].max()) + 0.5
        ax.axvspan(x0, x1, color=color, zorder=0)
        ax.text((x0 + x1) / 2, 1.030, label, ha="center", va="top", fontsize=7.0, color=MUTED, fontweight="bold")

    ax.scatter(metric["metric_order"], metric["pcie"], s=30, color="#C7D0DC", edgecolor="white", lw=0.5, zorder=2)

    promote = metric[metric["decision"].eq("promote")]
    if not promote.empty:
        ax.scatter(promote["metric_order"], promote["pcie"], s=74, color=PURPLE, edgecolor="white", lw=1.2, zorder=5)
    rejects = metric[metric["decision"].eq("reject")]
    if not rejects.empty:
        ax.scatter(rejects["metric_order"], rejects["pcie"], s=58, marker="x", color=RED, lw=1.5, zorder=5)

    ax.step(metric["metric_order"], metric["metric_best_so_far"], where="post", color=PURPLE, lw=2.6, label="best so far")
    ax.axhline(wfrfm_pcie, color="#7C8A9A", lw=1.0, ls=(0, (4, 2)))
    ax.axhline(success_cutoff, color=GREEN, lw=1.0, ls=(0, (2, 2)))
    ax.text(float(metric["metric_order"].max()) + 0.25, wfrfm_pcie + 0.006, f"WFR-FM {wfrfm_pcie:.3f}", ha="right", va="bottom", fontsize=7.0, color=MUTED)
    ax.text(float(metric["metric_order"].max()) + 0.25, success_cutoff - 0.006, "10% target", ha="right", va="top", fontsize=7.0, color=GREEN)

    for _, row in promote.iterrows():
        if row["pcie"] <= success_cutoff or row["metric_order"] <= 5:
            ax.text(row["metric_order"], row["pcie"] - 0.014, f"{row['pcie']:.3f}", ha="center", va="top", fontsize=6.9, color=PURPLE, fontweight="bold")

    first_success = metric[metric["pcie"].le(success_cutoff)].head(1)
    if not first_success.empty:
        row = first_success.iloc[0]
        ax.annotate(
            "first PCIE\nsuccess",
            xy=(row["metric_order"], row["pcie"]),
            xytext=(row["metric_order"] - 1.05, 0.755),
            arrowprops=dict(arrowstyle="-", color=PURPLE, lw=1.0),
            fontsize=7.0,
            color=PURPLE,
            ha="center",
            fontweight="bold",
        )

    tune = metric[metric["pulse_cost_weight"].eq(0.65)] if "pulse_cost_weight" in metric else pd.DataFrame()
    tune_first = tune.dropna(subset=["pcie_reduction_vs_wfrfm_pct"]).head(1)
    if not tune_first.empty:
        row = tune_first.iloc[0]
        tune_x = float(row["metric_order"])
        tune_y = float(row["pcie"])
    else:
        tune_x = float(first_success["metric_order"].iloc[0]) if not first_success.empty else 3.0
        tune_y = float(first_success["pcie"].iloc[0]) if not first_success.empty else 0.682
    ax.annotate(
        "tune pulse/residual/alignment\n0.901 -> 0.682",
        xy=(tune_x, tune_y),
        xytext=(max(tune_x - 1.9, 1.0), 0.790),
        arrowprops=dict(arrowstyle="-", color=PURPLE, lw=1.0),
        fontsize=7.1,
        color=PURPLE,
        ha="center",
    )
    final = metric.loc[metric["pcie"].idxmin()] if not metric.empty else None
    final_x = float(final["metric_order"]) if final is not None else float(metric["metric_order"].max())
    final_y = float(final["pcie"]) if final is not None else 0.680
    ax.annotate(
        "final: PCIE 0.680",
        xy=(final_x, final_y),
        xytext=(final_x - 1.10, 0.720),
        arrowprops=dict(arrowstyle="-", color=PURPLE, lw=1.0),
        fontsize=7.1,
        color=PURPLE,
        ha="center",
        fontweight="bold",
    )

    summary_lines = []
    for _, row in counts.iterrows():
        stage = str(row["stage"]).replace("stage", "S")
        if row["stage"] == "stage2":
            suffix = f"; {int(row['pcie_claim_success_ge_10pct'])} claim"
        else:
            suffix = ""
        summary_lines.append(f"{stage}: {int(row['promote_pass'])}/{int(row['formal_trials'])} pass{suffix}")
    ax.text(
        0.02,
        0.94,
        "Formal campaign pass counts\n" + "\n".join(summary_lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.0,
        color=INK,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=GRID, alpha=0.92),
        zorder=8,
    )

    op_text = (
        "Key operations\n"
        "3 pulse cost 0.35->0.65\n"
        "  residual scale 0.20->0.30\n"
        "  velocity loss 0.02->0.06\n"
        "4 weaker tune: 0.725, reject\n"
        "6 old pulse setting: 0.901, reject\n"
        "7 tuned rerun locks best"
    )
    ax.text(
        0.55,
        0.94,
        op_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.8,
        color=MUTED,
        zorder=8,
    )

    aborted = formal[formal["decision"].eq("aborted")]
    if not aborted.empty:
        ax.scatter(aborted["formal_order"], [0.665] * len(aborted), s=44, marker="s", color=GRAY, edgecolor="white", lw=0.8, zorder=5)
        ax.text(float(aborted["formal_order"].iloc[0]), 0.682, "aborted", ha="center", va="bottom", fontsize=6.8, color=MUTED)

    ax.set_xlim(0.5, float(formal["formal_order"].max()) + 0.5)
    ax.set_ylim(0.64, 1.035)
    ax.set_xlabel("formal campaign trial order")
    ax.set_ylabel("PCIE (lower is better)")
    ax.set_xticks(range(1, int(formal["formal_order"].max()) + 1))
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PURPLE, markeredgecolor="white", markersize=7, label="promote"),
        Line2D([0], [0], marker="x", color=RED, lw=0, markersize=7, label="reject"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=GRAY, markeredgecolor="white", markersize=6, label="aborted"),
        Line2D([0], [0], color=PURPLE, lw=2.6, label="best so far"),
    ]
    ax.legend(handles=handles, frameon=False, loc="center right", bbox_to_anchor=(0.995, 0.46), ncol=1, handletextpad=0.45)
    panel_label(ax, "B")
    clean_grid(ax)


def plot_panel_c(ax: plt.Axes, baselines: pd.DataFrame) -> None:
    ax.set_title("Final PCIE metric")
    ordered = baselines.sort_values("pcie", ascending=True).reset_index(drop=True)
    colors = [PURPLE if a == "PulseBridge" else "#9AA8BA" for a in ordered["algorithm"]]
    bars = ax.bar(np.arange(len(ordered)), ordered["pcie"], color=colors, width=0.72, edgecolor="white", lw=0.7, zorder=3)
    wfrfm_pcie = float(ordered.loc[ordered["algorithm"].eq("WFR-FM"), "pcie"].iloc[0])
    ax.axhline(wfrfm_pcie, color="#7C8A9A", lw=1.0, ls=(0, (4, 2)))
    ax.axhline(wfrfm_pcie * 0.90, color=GREEN, lw=1.0, ls=(0, (2, 2)))
    ax.text(len(ordered) - 0.2, wfrfm_pcie + 0.01, f"WFR-FM {wfrfm_pcie:.3f}", ha="right", va="bottom", fontsize=7.0, color=MUTED)
    ax.text(len(ordered) - 0.2, wfrfm_pcie * 0.90 - 0.012, "10% target", ha="right", va="top", fontsize=7.0, color=GREEN)

    pulse = float(ordered.loc[ordered["algorithm"].eq("PulseBridge"), "pcie"].iloc[0])
    improvement = (wfrfm_pcie - pulse) / wfrfm_pcie * 100.0
    ax.text(
        -0.18,
        0.825,
        f"{improvement:.1f}% lower\nthan WFR-FM",
        ha="left",
        va="top",
        fontsize=9.3,
        color=PURPLE,
        fontweight="bold",
    )
    ax.text(0, pulse + 0.028, f"{pulse:.3f}", ha="center", va="bottom", fontsize=7.4, color=PURPLE, fontweight="bold")
    for i, (bar, value) in enumerate(zip(bars, ordered["pcie"])):
        if i != 0:
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.3f}", ha="center", va="bottom", fontsize=6.6, color=MUTED, rotation=90)

    ax.set_xticks(np.arange(len(ordered)))
    ax.set_xticklabels(ordered["algorithm"], rotation=32, ha="right")
    ax.set_ylabel("PCIE (lower is better)")
    ax.set_ylim(0.60, 1.08)
    panel_label(ax, "C")
    clean_grid(ax)


def draw_header_rule(fig: plt.Figure) -> None:
    # A light rule keeps the title visually close to the reference landscape panel.
    ax = fig.add_axes([0.018, 0.902, 0.965, 0.002])
    ax.axis("off")
    box = FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.0,rounding_size=0.0", transform=ax.transAxes, facecolor=LIGHT_GRID, edgecolor="none")
    ax.add_patch(box)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    style()
    proposals, previews, campaigns, trials, full_events, counts, baselines = write_tables()

    fig = plt.figure(figsize=(11.8, 5.0), constrained_layout=False)
    gs = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.55, 0.82],
        left=0.065,
        right=0.975,
        bottom=0.14,
        top=0.82,
        wspace=0.22,
    )
    ax_b = fig.add_subplot(gs[0, 0])
    ax_c = fig.add_subplot(gs[0, 1])

    fig.text(
        0.02,
        0.958,
        "PulseBridge optimization and final pulse-identifiability metric",
        ha="left",
        va="top",
        fontsize=18,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.02,
        0.045,
        "Remote provenance: B shows only the seven formal trials in the locked campaign; "
        "proposal/setup/debug history is omitted from the plotted curve.",
        ha="left",
        va="bottom",
        fontsize=7.6,
        color=MUTED,
    )
    draw_header_rule(fig)

    plot_panel_b(ax_b, full_events, counts)
    plot_panel_c(ax_c, baselines)

    png = FIG / "story_pulsebridge_algorithm_search_summary.png"
    pdf = FIG / "story_pulsebridge_algorithm_search_summary.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")
    print(f"Wrote {TAB / 'pulsebridge_algorithm_proposal_revisions.csv'}")
    print(f"Wrote {TAB / 'pulsebridge_algorithm_preview_history.csv'}")
    print(f"Wrote {TAB / 'pulsebridge_algorithm_campaign_summary.csv'}")
    print(f"Wrote {TAB / 'pulsebridge_algorithm_tuning_trials.csv'}")
    print(f"Wrote {TAB / 'pulsebridge_algorithm_full_lifecycle_events.csv'}")
    print(f"Wrote {TAB / 'pulsebridge_algorithm_stage_pass_counts.csv'}")
    print(f"Wrote {TAB / 'pulsebridge_algorithm_pcie_baselines.csv'}")


if __name__ == "__main__":
    main()
