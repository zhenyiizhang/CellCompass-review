#!/usr/bin/env python3
"""Generate downstream comparison figures for locked CITE-PCFM release.

This script uses final-regression metrics, benchmark baseline metrics, and the
locked model-generated evaluation trajectory. It does not rerun training or
construct predictions from future observations.
"""
from __future__ import annotations

import json
import math
import html
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.decomposition import PCA

OUTPUT_DIR = Path('/Users/zhenyizhang/Documents/GitHub/CytoBridge-agent-runtime-v2-20260309/multi-omics')
FIG_DIR = OUTPUT_DIR / 'figures' / 'downstream_cite_pcfm'
TABLE_DIR = OUTPUT_DIR / 'downstream' / 'cite_pcfm' / 'tables'
MANIFEST_DIR = OUTPUT_DIR / 'downstream' / 'cite_pcfm'
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

DATASET_ID = 'gse305370_citeseq_stage2_10k'
DATA_PATH = Path('/Users/zhenyizhang/.cellcompass/algorithm_benchmarks/datasets/gse305370_citeseq_stage2_10k/data.h5ad')
FINAL_RUN_DIR = Path('/Users/zhenyizhang/Documents/GitHub/CytoBridge-agent-runtime-v2-20260309/multi-omics/training_runs/20260526T174204321821Z__final__cite_paired_coord_fm__campaign_20260-trial_20260526-gse305370_ci')
FINAL_METRICS = FINAL_RUN_DIR / 'artifacts' / 'metrics.json'
FINAL_TRAJ = FINAL_RUN_DIR / 'artifacts' / 'evaluation_trajectory.npz'
BASELINE_DIR = Path('/Users/zhenyizhang/.cellcompass/algorithm_benchmarks/datasets/gse305370_citeseq_stage2_10k/baseline_artifacts')

sns.set_theme(style='whitegrid', context='paper', font_scale=1.15)
PALETTE = {
    'CITE-PCFM': '#C44E52',
    'dynamical_ot': '#4C72B0',
    'crufm': '#55A868',
    'balanced_ot_cfm': '#8172B3',
    'vgfm': '#CCB974',
    'sf2m': '#64B5CD',
    'unbalanced_ot': '#8C8C8C',
    'ruot': '#937860',
    'cyto_simulation': '#DA8BC3',
    'wfrfm': '#BBBBBB',
}

METHOD_LABELS = {
    'CITE-PCFM': 'CITE-PCFM',
    'crufm': 'CRUFM',
    'vgfm': 'VGFM',
    'sf2m': 'SF2M',
    'balanced_ot_cfm': 'Balanced OT-CFM',
    'unbalanced_ot': 'Unbalanced OT',
    'ruot': 'RUOT',
    'dynamical_ot': 'Dynamical OT',
    'cyto_simulation': 'CytoSimulation',
    'wfrfm': 'WFR-FM',
}

SURFACE_PROTEIN_ERROR_LABEL = 'surface-protein distribution error'


def safe_mean(xs):
    if xs is None:
        return np.nan
    arr = np.asarray(xs, dtype=float)
    if arr.size == 0:
        return np.nan
    return float(np.nanmean(arr))


def load_metrics(path: Path) -> dict:
    return json.loads(path.read_text())


def parse_per_time(custom: dict) -> list[dict]:
    raw = custom.get('protein_energy_per_time_json')
    if not raw:
        return []
    try:
        return json.loads(raw)
    except Exception:
        return []


def collect_metric_tables():
    rows = []
    per_time_rows = []
    final = load_metrics(FINAL_METRICS)
    final_custom = final.get('custom_metrics', {}) or {}
    rows.append({
        'method': 'CITE-PCFM',
        'method_type': 'custom_final',
        'w1_mean': safe_mean(final.get('w1_scores')),
        'tmv_mean': safe_mean(final.get('tmv_scores')),
        'protein_energy': final_custom.get('protein_energy', np.nan),
        'rna_energy_guardrail': final_custom.get('rna_energy_guardrail', np.nan),
        'paired_cov_error': final_custom.get('paired_cov_error', np.nan),
        'runtime_sec': final.get('runtime_sec', np.nan),
        'metrics_path': str(FINAL_METRICS),
    })
    for i, w1 in enumerate(final.get('w1_scores', [])):
        pass
    for rec in parse_per_time(final_custom):
        rec = dict(rec)
        rec['method'] = 'CITE-PCFM'
        rec['method_type'] = 'custom_final'
        per_time_rows.append(rec)

    for mpath in sorted(BASELINE_DIR.glob('builtin_*/metrics.json')):
        method = mpath.parent.name.replace('builtin_', '')
        m = load_metrics(mpath)
        custom = m.get('custom_metrics', {}) or {}
        rows.append({
            'method': method,
            'method_type': 'builtin_baseline',
            'w1_mean': m.get('w1_mean', safe_mean(m.get('w1_scores'))),
            'tmv_mean': m.get('tmv_mean', safe_mean(m.get('tmv_scores'))),
            'protein_energy': custom.get('protein_energy', np.nan),
            'rna_energy_guardrail': custom.get('rna_energy_guardrail', np.nan),
            'residual_energy': custom.get('residual_energy', np.nan),
            'paired_cov_error': custom.get('paired_cov_error', np.nan),
            'runtime_sec': m.get('runtime_sec', np.nan),
            'metrics_path': str(mpath),
        })
        for rec in parse_per_time(custom):
            rec = dict(rec)
            rec['method'] = method
            rec['method_type'] = 'builtin_baseline'
            per_time_rows.append(rec)

    summary = pd.DataFrame(rows)
    per_time = pd.DataFrame(per_time_rows)
    summary.to_csv(TABLE_DIR / 'method_metric_summary.csv', index=False)
    per_time.to_csv(TABLE_DIR / 'protein_energy_per_time.csv', index=False)
    return summary, per_time


def savefig(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
    plt.savefig(path.with_suffix('.svg'), bbox_inches='tight')
    plt.close()


def plot_protein_energy_bar(
    summary: pd.DataFrame,
    output_name: str = 'fig1_protein_energy_bar',
    figsize: tuple[float, float] = (7.8, 4.9),
    compact: bool = False,
):
    df = summary.dropna(subset=['protein_energy']).copy()
    df = df.sort_values('protein_energy')
    df['method_label'] = df['method'].map(METHOD_LABELS).fillna(df['method'])
    best_baseline_row = df[df.method != 'CITE-PCFM'].sort_values('protein_energy').iloc[0]
    best_baseline = float(best_baseline_row['protein_energy'])
    best_baseline_method = str(best_baseline_row['method'])
    best_baseline_label = METHOD_LABELS.get(best_baseline_method, best_baseline_method)
    ours = float(df[df.method == 'CITE-PCFM']['protein_energy'].iloc[0])
    improvement = (best_baseline - ours) / best_baseline * 100 if best_baseline > 0 else np.nan
    df = df.reset_index(drop=True)

    fig, ax = plt.subplots(figsize=figsize)
    y = np.arange(len(df))
    x_min = 0.018
    colors = []
    for method in df['method']:
        if method == 'CITE-PCFM':
            colors.append('#6D28D9')
        elif method == best_baseline_method:
            colors.append('#23825F')
        else:
            colors.append('#CBD5E1')

    ax.barh(
        y,
        df['protein_energy'].to_numpy(dtype=float) - x_min,
        left=x_min,
        height=0.58 if compact else 0.70,
        color=colors,
        edgecolor='white',
        linewidth=1.1,
        zorder=3,
    )
    ax.scatter(
        df['protein_energy'],
        y,
        s=np.where(df['method'].eq('CITE-PCFM'), 52 if compact else 66, 27 if compact else 34),
        color=colors,
        edgecolor='white',
        linewidth=0.8,
        zorder=4,
    )
    ax.axvline(best_baseline, color='#64748B', linestyle=(0, (4, 3)), linewidth=1.15, alpha=0.9, zorder=2)

    ax.set_xscale('log')
    ax.set_xlim(x_min, 1.05)
    ax.set_yticks(y)
    ax.set_yticklabels(df['method_label'], fontsize=8.4 if compact else 9.5)
    ax.invert_yaxis()
    ax.set_xlabel(f'{SURFACE_PROTEIN_ERROR_LABEL} (log scale; lower is better)', fontsize=9.3 if compact else 10.5)
    ax.set_ylabel('')
    ax.set_title(
        'CITE-PCFM best matches real CITE surface-protein distributions',
        loc='left',
        fontsize=12.2 if compact else 14.2,
        weight='bold',
        color='#111827',
        pad=18 if compact else 22,
    )
    subtitle = (
        f'{improvement:.1f}% lower error than {best_baseline_label}; RNA-only input for CITE-PCFM versus RNA+observed ADT input for baselines.'
        if compact
        else f'{improvement:.1f}% lower error than {best_baseline_label}; CITE-PCFM starts from RNA only, whereas baselines start from paired RNA+observed ADT states.'
    )
    ax.text(
        0.0,
        1.015,
        subtitle,
        transform=ax.transAxes,
        fontsize=7.9 if compact else 9.1,
        color='#475569',
        va='bottom',
        ha='left',
    )

    input_box = 'Starting-state input\nCITE-PCFM: RNA only\nAll baselines: RNA + observed ADT'
    ax.text(
        0.64 if compact else 0.52,
        0.87 if compact else 0.88,
        input_box,
        transform=ax.transAxes,
        fontsize=7.4 if compact else 8.4,
        color='#334155',
        va='top',
        ha='left',
        linespacing=1.45,
        bbox=dict(boxstyle='round,pad=0.34,rounding_size=0.08', facecolor='white', edgecolor='#CBD5E1', linewidth=0.9),
        zorder=6,
    )

    for yy, (_, row) in enumerate(df.iterrows()):
        val = float(row['protein_energy'])
        label = f'{val:.4f}' if val < 0.1 else f'{val:.3f}'
        color = '#4C1D95' if row['method'] == 'CITE-PCFM' else '#334155'
        ax.text(val * 1.05, yy, label, va='center', ha='left', fontsize=7.7 if compact else 8.8, color=color)

    ax.xaxis.set_major_locator(mticker.FixedLocator([0.02, 0.03, 0.1, 0.3, 1.0]))
    ax.xaxis.set_major_formatter(mticker.FixedFormatter(['0.02', '0.03', '0.10', '0.30', '1.0']))
    ax.xaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.grid(axis='x', which='major', color='#E2E8F0', linewidth=0.9)
    ax.grid(axis='x', which='minor', color='#F1F5F9', linewidth=0.45)
    ax.grid(axis='y', visible=False)
    ax.tick_params(axis='y', length=0, pad=6)
    ax.tick_params(axis='x', colors='#334155', labelsize=8.2 if compact else 9.0)
    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_color('#CBD5E1')
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    savefig(FIG_DIR / f'{output_name}.png')


def plot_protein_energy_bar_landscape(summary: pd.DataFrame):
    df = summary.dropna(subset=['protein_energy']).copy()
    df = df.sort_values('protein_energy').reset_index(drop=True)
    best_baseline_row = df[df.method != 'CITE-PCFM'].sort_values('protein_energy').iloc[0]
    best_baseline = float(best_baseline_row['protein_energy'])
    best_baseline_method = str(best_baseline_row['method'])
    best_baseline_label = METHOD_LABELS.get(best_baseline_method, best_baseline_method)
    ours = float(df[df.method == 'CITE-PCFM']['protein_energy'].iloc[0])
    improvement = (best_baseline - ours) / best_baseline * 100 if best_baseline > 0 else np.nan
    xlabels = {
        'CITE-PCFM': 'CITE-\nPCFM',
        'crufm': 'CRUFM',
        'vgfm': 'VGFM',
        'sf2m': 'SF2M',
        'balanced_ot_cfm': 'Bal.\nOT-CFM',
        'unbalanced_ot': 'Unbal.\nOT',
        'ruot': 'RUOT',
        'dynamical_ot': 'Dyn.\nOT',
        'cyto_simulation': 'CytoSim',
    }
    df['method_label'] = df['method'].map(xlabels).fillna(df['method'])
    colors = [
        '#6D28D9' if method == 'CITE-PCFM' else '#23825F' if method == best_baseline_method else '#CBD5E1'
        for method in df['method']
    ]

    fig, ax = plt.subplots(figsize=(5.45, 3.8))
    x = np.arange(len(df))
    vals = df['protein_energy'].to_numpy(dtype=float)
    ax.bar(x, vals, width=0.62, color=colors, edgecolor='white', linewidth=1.1, zorder=3)
    ax.scatter(x, vals, s=np.where(df['method'].eq('CITE-PCFM'), 54, 30), color=colors, edgecolor='white', linewidth=0.8, zorder=4)
    ax.axhline(best_baseline, color='#64748B', linestyle=(0, (4, 3)), linewidth=1.1, alpha=0.9, zorder=2)

    ax.set_yscale('log')
    ax.set_ylim(0.018, 1.05)
    ax.set_xlim(-0.65, len(df) - 0.35)
    ax.set_xticks(x)
    ax.set_xticklabels(df['method_label'], fontsize=8.2)
    ax.set_ylabel(f'{SURFACE_PROTEIN_ERROR_LABEL}\n(log scale; lower is better)', fontsize=9.0)
    ax.set_xlabel('')
    ax.set_title(
        'CITE-PCFM best matches real CITE surface-protein distributions',
        loc='left',
        fontsize=9.6,
        weight='bold',
        color='#111827',
        pad=16,
    )
    ax.text(
        0.0,
        1.015,
        f'{improvement:.1f}% lower error than {best_baseline_label}; CITE-PCFM uses RNA-only input, while all baselines use RNA+observed ADT input.',
        transform=ax.transAxes,
        fontsize=6.5,
        color='#475569',
        va='bottom',
        ha='left',
    )
    ax.text(
        0.03,
        0.91,
        'Starting-state input\nCITE-PCFM: RNA only\nAll baselines: RNA + observed ADT',
        transform=ax.transAxes,
        fontsize=6.2,
        color='#334155',
        va='top',
        ha='left',
        linespacing=1.35,
        bbox=dict(boxstyle='round,pad=0.32,rounding_size=0.08', facecolor='white', edgecolor='#CBD5E1', linewidth=0.9),
        zorder=6,
    )
    for xx, row in df.iterrows():
        val = float(row['protein_energy'])
        label = f'{val:.4f}' if val < 0.1 else f'{val:.3f}'
        color = '#4C1D95' if row['method'] == 'CITE-PCFM' else '#334155'
        ax.text(xx, val * 1.10, label, va='bottom', ha='center', fontsize=6.9, color=color)

    ax.yaxis.set_major_locator(mticker.FixedLocator([0.02, 0.03, 0.1, 0.3, 1.0]))
    ax.yaxis.set_major_formatter(mticker.FixedFormatter(['0.02', '0.03', '0.10', '0.30', '1.0']))
    ax.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.grid(axis='y', which='major', color='#E2E8F0', linewidth=0.9)
    ax.grid(axis='y', which='minor', color='#F1F5F9', linewidth=0.45)
    ax.grid(axis='x', visible=False)
    ax.tick_params(axis='x', length=0, pad=5, labelsize=7.4)
    ax.tick_params(axis='y', colors='#334155', labelsize=7.4)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_color('#CBD5E1')
    ax.spines['bottom'].set_color('#CBD5E1')
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    savefig(FIG_DIR / 'fig1_protein_energy_bar_landscape.png')


def write_protein_energy_bar_landscape_ai_svg(summary: pd.DataFrame):
    df = summary.dropna(subset=['protein_energy']).copy()
    df = df.sort_values('protein_energy').reset_index(drop=True)
    best_baseline_row = df[df.method != 'CITE-PCFM'].sort_values('protein_energy').iloc[0]
    best_baseline = float(best_baseline_row['protein_energy'])
    best_baseline_method = str(best_baseline_row['method'])
    best_baseline_label = METHOD_LABELS.get(best_baseline_method, best_baseline_method)
    ours = float(df[df.method == 'CITE-PCFM']['protein_energy'].iloc[0])
    improvement = (best_baseline - ours) / best_baseline * 100 if best_baseline > 0 else np.nan
    xlabels = {
        'CITE-PCFM': ['CITE-', 'PCFM'],
        'crufm': ['CRUFM'],
        'vgfm': ['VGFM'],
        'sf2m': ['SF2M'],
        'balanced_ot_cfm': ['Bal.', 'OT-CFM'],
        'unbalanced_ot': ['Unbal.', 'OT'],
        'ruot': ['RUOT'],
        'dynamical_ot': ['Dyn.', 'OT'],
        'cyto_simulation': ['CytoSim'],
    }

    width, height = 420.0, 280.0
    left, right, top, bottom = 55.0, 18.0, 62.0, 42.0
    plot_w = width - left - right
    plot_h = height - top - bottom
    y_min, y_max = 0.018, 1.05
    log_min, log_max = math.log10(y_min), math.log10(y_max)

    def sx(i: int) -> float:
        return left + (i + 0.5) * plot_w / len(df)

    def sy(v: float) -> float:
        frac = (math.log10(v) - log_min) / (log_max - log_min)
        return top + plot_h * (1.0 - frac)

    def esc(text: str) -> str:
        return html.escape(str(text), quote=True)

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}pt" height="{height}pt" viewBox="0 0 {width} {height}">',
        '<rect x="0" y="0" width="420" height="280" fill="#ffffff"/>',
        '<text x="55" y="24" font-family="Arial, Helvetica, sans-serif" font-size="13.2" font-weight="700" fill="#111827">CITE-PCFM best matches real CITE surface-protein distributions</text>',
        f'<text x="55" y="42" font-family="Arial, Helvetica, sans-serif" font-size="7.4" fill="#475569">{improvement:.1f}% lower error than {esc(best_baseline_label)}; CITE-PCFM uses RNA-only input, while all baselines use RNA+observed ADT input.</text>',
        '<text x="55" y="55" font-family="Arial, Helvetica, sans-serif" font-size="7.4" fill="#334155">surface-protein distribution error (log scale; lower is better)</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#ffffff"/>',
    ]

    ticks = [0.02, 0.03, 0.1, 0.3, 1.0]
    for tick in ticks:
        y = sy(tick)
        lines.append(f'<line x1="{left}" y1="{y:.3f}" x2="{left + plot_w}" y2="{y:.3f}" stroke="#E2E8F0" stroke-width="1.0"/>')
        label = f'{tick:.2f}' if tick < 0.1 else f'{tick:.2f}' if tick < 1 else '1.0'
        lines.append(f'<text x="{left - 10}" y="{y + 2.5:.3f}" font-family="Arial, Helvetica, sans-serif" font-size="8.3" text-anchor="end" fill="#334155">{label}</text>')

    minor_ticks = [0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for tick in minor_ticks:
        y = sy(tick)
        lines.append(f'<line x1="{left}" y1="{y:.3f}" x2="{left + plot_w}" y2="{y:.3f}" stroke="#F1F5F9" stroke-width="0.55"/>')

    baseline_y = sy(best_baseline)
    lines.append(f'<line x1="{left}" y1="{baseline_y:.3f}" x2="{left + plot_w}" y2="{baseline_y:.3f}" stroke="#64748B" stroke-width="1.05" stroke-dasharray="4 3"/>')
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#CBD5E1" stroke-width="1.1"/>')
    lines.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#CBD5E1" stroke-width="1.1"/>')

    bar_w = min(24.0, plot_w / len(df) * 0.58)
    for i, row in df.iterrows():
        method = str(row['method'])
        val = float(row['protein_energy'])
        color = '#6D28D9' if method == 'CITE-PCFM' else '#23825F' if method == best_baseline_method else '#CBD5E1'
        x = sx(i)
        y = sy(val)
        h = top + plot_h - y
        lines.append(f'<rect x="{x - bar_w / 2:.3f}" y="{y:.3f}" width="{bar_w:.3f}" height="{h:.3f}" fill="{color}" stroke="#ffffff" stroke-width="1.0"/>')
        lines.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="{4.0 if method == "CITE-PCFM" else 3.3}" fill="{color}" stroke="#ffffff" stroke-width="0.9"/>')
        label = f'{val:.4f}' if val < 0.1 else f'{val:.3f}'
        text_color = '#4C1D95' if method == 'CITE-PCFM' else '#334155'
        lines.append(f'<text x="{x:.3f}" y="{y - 7.0:.3f}" font-family="Arial, Helvetica, sans-serif" font-size="7.0" text-anchor="middle" fill="{text_color}">{label}</text>')
        for j, text in enumerate(xlabels.get(method, [method])):
            lines.append(f'<text x="{x:.3f}" y="{top + plot_h + 15 + j * 9:.3f}" font-family="Arial, Helvetica, sans-serif" font-size="8.0" text-anchor="middle" fill="#262626">{esc(text)}</text>')

    lines.extend([
        '<rect x="63" y="76" width="122" height="39" rx="1.5" fill="#ffffff" stroke="#CBD5E1" stroke-width="1.0"/>',
        '<text x="69" y="89" font-family="Arial, Helvetica, sans-serif" font-size="7.0" fill="#334155">Starting-state input</text>',
        '<text x="69" y="101" font-family="Arial, Helvetica, sans-serif" font-size="7.0" fill="#334155">CITE-PCFM: RNA only</text>',
        '<text x="69" y="113" font-family="Arial, Helvetica, sans-serif" font-size="7.0" fill="#334155">All baselines: RNA + observed ADT</text>',
        '</svg>',
    ])

    out = FIG_DIR / 'fig1_protein_energy_bar_landscape_ai_safe.svg'
    out.write_text('\n'.join(lines), encoding='utf-8')
    return out


def write_protein_energy_bar_panel_b_inset_ai_svg(summary: pd.DataFrame):
    df = summary.dropna(subset=['protein_energy']).copy()
    df = df.sort_values('protein_energy').reset_index(drop=True)
    best_baseline_row = df[df.method != 'CITE-PCFM'].sort_values('protein_energy').iloc[0]
    best_baseline = float(best_baseline_row['protein_energy'])
    best_baseline_method = str(best_baseline_row['method'])
    ours = float(df[df.method == 'CITE-PCFM']['protein_energy'].iloc[0])
    improvement = (best_baseline - ours) / best_baseline * 100 if best_baseline > 0 else np.nan
    labels = {
        'CITE-PCFM': 'PCFM',
        'crufm': 'CRUFM',
        'vgfm': 'VGFM',
        'sf2m': 'SF2M',
        'balanced_ot_cfm': 'Bal',
        'unbalanced_ot': 'UOT',
        'ruot': 'RUOT',
        'dynamical_ot': 'DOT',
        'cyto_simulation': 'Cyto',
    }

    width, height = 250.0, 190.0
    left, right, top, bottom = 34.0, 8.0, 42.0, 30.0
    plot_w = width - left - right
    plot_h = height - top - bottom
    y_min, y_max = 0.018, 1.05
    log_min, log_max = math.log10(y_min), math.log10(y_max)

    def sx(i: int) -> float:
        return left + (i + 0.5) * plot_w / len(df)

    def sy(v: float) -> float:
        frac = (math.log10(v) - log_min) / (log_max - log_min)
        return top + plot_h * (1.0 - frac)

    def esc(text: str) -> str:
        return html.escape(str(text), quote=True)

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}pt" height="{height}pt" viewBox="0 0 {width} {height}">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        '<text x="34" y="16" font-family="Arial, Helvetica, sans-serif" font-size="8.8" font-weight="700" fill="#111827">Surface-protein distribution error</text>',
        f'<text x="34" y="28" font-family="Arial, Helvetica, sans-serif" font-size="5.8" fill="#475569">{improvement:.1f}% lower than CRUFM; PCFM uses RNA only</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#ffffff"/>',
    ]

    ticks = [0.02, 0.03, 0.1, 0.3, 1.0]
    for tick in ticks:
        y = sy(tick)
        lines.append(f'<line x1="{left}" y1="{y:.3f}" x2="{left + plot_w}" y2="{y:.3f}" stroke="#E2E8F0" stroke-width="0.8"/>')
        label = f'{tick:.2f}' if tick < 1 else '1.0'
        lines.append(f'<text x="{left - 5}" y="{y + 2.0:.3f}" font-family="Arial, Helvetica, sans-serif" font-size="5.6" text-anchor="end" fill="#334155">{label}</text>')

    for tick in [0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        y = sy(tick)
        lines.append(f'<line x1="{left}" y1="{y:.3f}" x2="{left + plot_w}" y2="{y:.3f}" stroke="#F1F5F9" stroke-width="0.4"/>')

    baseline_y = sy(best_baseline)
    lines.append(f'<line x1="{left}" y1="{baseline_y:.3f}" x2="{left + plot_w}" y2="{baseline_y:.3f}" stroke="#64748B" stroke-width="0.9" stroke-dasharray="3.5 2.8"/>')
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#CBD5E1" stroke-width="0.9"/>')
    lines.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#CBD5E1" stroke-width="0.9"/>')

    bar_w = min(14.0, plot_w / len(df) * 0.52)
    for i, row in df.iterrows():
        method = str(row['method'])
        val = float(row['protein_energy'])
        color = '#6D28D9' if method == 'CITE-PCFM' else '#23825F' if method == best_baseline_method else '#CBD5E1'
        x = sx(i)
        y = sy(val)
        h = top + plot_h - y
        lines.append(f'<rect x="{x - bar_w / 2:.3f}" y="{y:.3f}" width="{bar_w:.3f}" height="{h:.3f}" fill="{color}" stroke="#ffffff" stroke-width="0.8"/>')
        lines.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="{2.8 if method == "CITE-PCFM" else 2.35}" fill="{color}" stroke="#ffffff" stroke-width="0.75"/>')
        if method in {'CITE-PCFM', best_baseline_method, 'vgfm'} or val >= 0.15:
            label = f'{val:.4f}' if val < 0.1 else f'{val:.2f}'
            text_color = '#4C1D95' if method == 'CITE-PCFM' else '#334155'
            lines.append(f'<text x="{x:.3f}" y="{y - 4.8:.3f}" font-family="Arial, Helvetica, sans-serif" font-size="5.5" text-anchor="middle" fill="{text_color}">{label}</text>')
        lines.append(f'<text x="{x:.3f}" y="{top + plot_h + 11:.3f}" font-family="Arial, Helvetica, sans-serif" font-size="5.6" text-anchor="middle" fill="#262626">{esc(labels.get(method, method))}</text>')

    lines.extend([
        '<rect x="41" y="49" width="78" height="26" rx="1.2" fill="#ffffff" stroke="#CBD5E1" stroke-width="0.8"/>',
        '<text x="45" y="58" font-family="Arial, Helvetica, sans-serif" font-size="5.4" fill="#334155">Input</text>',
        '<text x="45" y="66" font-family="Arial, Helvetica, sans-serif" font-size="5.4" fill="#334155">PCFM: RNA only</text>',
        '<text x="45" y="73" font-family="Arial, Helvetica, sans-serif" font-size="5.4" fill="#334155">Baselines: RNA + ADT</text>',
        '</svg>',
    ])

    out = FIG_DIR / 'fig1_protein_energy_bar_panel_b_inset_ai_safe.svg'
    out.write_text('\n'.join(lines), encoding='utf-8')
    return out


def write_protein_energy_micro_inset_ai_svg(summary: pd.DataFrame):
    df = summary.dropna(subset=['protein_energy']).copy()
    df = df.sort_values('protein_energy').reset_index(drop=True)
    pcfm = float(df.loc[df.method.eq('CITE-PCFM'), 'protein_energy'].iloc[0])
    baselines = df[~df.method.eq('CITE-PCFM')].copy()
    crufm = float(baselines.loc[baselines.method.eq('crufm'), 'protein_energy'].iloc[0])
    other = baselines[~baselines.method.eq('crufm')]['protein_energy'].to_numpy(dtype=float)
    other_min = float(np.nanmin(other))
    other_max = float(np.nanmax(other))
    improvement = (crufm - pcfm) / crufm * 100 if crufm > 0 else np.nan

    width, height = 155.0, 112.0
    left, right, top, bottom = 42.0, 12.0, 30.0, 23.0
    plot_w = width - left - right
    y_min, y_max = 0.018, 1.05
    log_min, log_max = math.log10(y_min), math.log10(y_max)

    def sx(v: float) -> float:
        frac = (math.log10(v) - log_min) / (log_max - log_min)
        return left + plot_w * frac

    rows = [
        ('PCFM', 'RNA only', pcfm, '#6D28D9'),
        ('CRUFM', 'RNA+ADT', crufm, '#23825F'),
        ('others', 'RNA+ADT', None, '#CBD5E1'),
    ]
    ys = [50.0, 70.0, 90.0]

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}pt" height="{height}pt" viewBox="0 0 {width} {height}">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        '<text x="6" y="13" font-family="Arial, Helvetica, sans-serif" font-size="7.8" font-weight="700" fill="#111827">Protein distribution error</text>',
        f'<text x="6" y="24" font-family="Arial, Helvetica, sans-serif" font-size="5.8" fill="#475569">PCFM is {improvement:.1f}% lower than best paired baseline</text>',
    ]

    for tick, label in [(0.02, '0.02'), (0.03, '0.03'), (0.1, '0.1'), (1.0, '1.0')]:
        x = sx(tick)
        lines.append(f'<line x1="{x:.3f}" y1="{top}" x2="{x:.3f}" y2="{height - bottom}" stroke="#E2E8F0" stroke-width="0.6"/>')
        lines.append(f'<text x="{x:.3f}" y="{height - 7}" font-family="Arial, Helvetica, sans-serif" font-size="4.9" text-anchor="middle" fill="#64748B">{label}</text>')

    for (name, input_label, value, color), y in zip(rows, ys):
        lines.append(f'<text x="6" y="{y + 2.0:.3f}" font-family="Arial, Helvetica, sans-serif" font-size="5.8" fill="#334155">{name}</text>')
        lines.append(f'<text x="6" y="{y + 9.0:.3f}" font-family="Arial, Helvetica, sans-serif" font-size="4.8" fill="#64748B">{input_label}</text>')
        if value is None:
            x0, x1 = sx(other_min), sx(other_max)
            lines.append(f'<line x1="{x0:.3f}" y1="{y:.3f}" x2="{x1:.3f}" y2="{y:.3f}" stroke="{color}" stroke-width="4.8" stroke-linecap="round"/>')
            lines.append(f'<circle cx="{x0:.3f}" cy="{y:.3f}" r="2.4" fill="#CBD5E1" stroke="#ffffff" stroke-width="0.7"/>')
            lines.append(f'<circle cx="{x1:.3f}" cy="{y:.3f}" r="2.4" fill="#CBD5E1" stroke="#ffffff" stroke-width="0.7"/>')
            lines.append(f'<text x="{x0:.3f}" y="{y - 5.5:.3f}" font-family="Arial, Helvetica, sans-serif" font-size="4.8" text-anchor="middle" fill="#64748B">{other_min:.3f}</text>')
            lines.append(f'<text x="{x1:.3f}" y="{y - 5.5:.3f}" font-family="Arial, Helvetica, sans-serif" font-size="4.8" text-anchor="middle" fill="#64748B">{other_max:.2f}</text>')
        else:
            x = sx(value)
            lines.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="3.5" fill="{color}" stroke="#ffffff" stroke-width="0.9"/>')
            label = f'{value:.4f}'
            lines.append(f'<text x="{x + 5.0:.3f}" y="{y + 1.8:.3f}" font-family="Arial, Helvetica, sans-serif" font-size="5.5" fill="{color}">{label}</text>')

    lines.append('</svg>')
    out = FIG_DIR / 'fig1_protein_energy_micro_inset_ai_safe.svg'
    out.write_text('\n'.join(lines), encoding='utf-8')
    return out


def write_protein_energy_micro_inset_vertical_ai_svg(summary: pd.DataFrame):
    df = summary.dropna(subset=['protein_energy']).copy()
    df = df.sort_values('protein_energy').reset_index(drop=True)
    pcfm = float(df.loc[df.method.eq('CITE-PCFM'), 'protein_energy'].iloc[0])
    baselines = df[~df.method.eq('CITE-PCFM')].copy()
    crufm = float(baselines.loc[baselines.method.eq('crufm'), 'protein_energy'].iloc[0])
    other = baselines[~baselines.method.eq('crufm')]['protein_energy'].to_numpy(dtype=float)
    other_min = float(np.nanmin(other))
    other_max = float(np.nanmax(other))
    improvement = (crufm - pcfm) / crufm * 100 if crufm > 0 else np.nan

    width, height = 120.0, 135.0
    left, right, top, bottom = 25.0, 8.0, 34.0, 23.0
    plot_w = width - left - right
    plot_h = height - top - bottom
    y_min, y_max = 0.018, 1.05
    log_min, log_max = math.log10(y_min), math.log10(y_max)

    def sy(v: float) -> float:
        frac = (math.log10(v) - log_min) / (log_max - log_min)
        return top + plot_h * (1.0 - frac)

    def esc(text: str) -> str:
        return html.escape(str(text), quote=True)

    x_pcfm = left + plot_w * 0.20
    x_crufm = left + plot_w * 0.50
    x_other = left + plot_w * 0.80
    base_y = top + plot_h
    bar_w = 12.0

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}pt" height="{height}pt" viewBox="0 0 {width} {height}">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        '<text x="6" y="13" font-family="Arial, Helvetica, sans-serif" font-size="7.5" font-weight="700" fill="#111827">Protein error</text>',
        f'<text x="6" y="24" font-family="Arial, Helvetica, sans-serif" font-size="5.3" fill="#475569">RNA-only PCFM, {improvement:.1f}% lower</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#ffffff"/>',
    ]

    for tick, label in [(0.02, '0.02'), (0.03, '0.03'), (0.1, '0.1'), (0.3, '0.3'), (1.0, '1')]:
        y = sy(tick)
        lines.append(f'<line x1="{left}" y1="{y:.3f}" x2="{left + plot_w}" y2="{y:.3f}" stroke="#E2E8F0" stroke-width="0.7"/>')
        lines.append(f'<text x="{left - 4}" y="{y + 1.8:.3f}" font-family="Arial, Helvetica, sans-serif" font-size="4.9" text-anchor="end" fill="#64748B">{label}</text>')

    cr_y = sy(crufm)
    lines.append(f'<line x1="{left}" y1="{cr_y:.3f}" x2="{left + plot_w}" y2="{cr_y:.3f}" stroke="#64748B" stroke-width="0.85" stroke-dasharray="3 2.5"/>')
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{base_y}" stroke="#CBD5E1" stroke-width="0.85"/>')
    lines.append(f'<line x1="{left}" y1="{base_y}" x2="{left + plot_w}" y2="{base_y}" stroke="#CBD5E1" stroke-width="0.85"/>')

    for x, val, color, name, value_label in [
        (x_pcfm, pcfm, '#6D28D9', 'PCFM', f'{pcfm:.4f}'),
        (x_crufm, crufm, '#23825F', 'CRUFM', f'{crufm:.4f}'),
    ]:
        y = sy(val)
        lines.append(f'<rect x="{x - bar_w / 2:.3f}" y="{y:.3f}" width="{bar_w:.3f}" height="{base_y - y:.3f}" fill="{color}" stroke="#ffffff" stroke-width="0.8"/>')
        lines.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="2.7" fill="{color}" stroke="#ffffff" stroke-width="0.7"/>')
        lines.append(f'<text x="{x:.3f}" y="{y - 5.0:.3f}" font-family="Arial, Helvetica, sans-serif" font-size="5.0" text-anchor="middle" fill="{color}">{esc(value_label)}</text>')
        lines.append(f'<text x="{x:.3f}" y="{base_y + 10.0:.3f}" font-family="Arial, Helvetica, sans-serif" font-size="5.5" text-anchor="middle" fill="#262626">{esc(name)}</text>')

    other_low_y = sy(other_min)
    other_high_y = sy(other_max)
    lines.append(f'<line x1="{x_other:.3f}" y1="{other_high_y:.3f}" x2="{x_other:.3f}" y2="{other_low_y:.3f}" stroke="#CBD5E1" stroke-width="6.0" stroke-linecap="round"/>')
    lines.append(f'<circle cx="{x_other:.3f}" cy="{other_low_y:.3f}" r="2.5" fill="#CBD5E1" stroke="#ffffff" stroke-width="0.7"/>')
    lines.append(f'<circle cx="{x_other:.3f}" cy="{other_high_y:.3f}" r="2.5" fill="#CBD5E1" stroke="#ffffff" stroke-width="0.7"/>')
    lines.append(f'<text x="{x_other:.3f}" y="{other_high_y - 5.0:.3f}" font-family="Arial, Helvetica, sans-serif" font-size="5.0" text-anchor="middle" fill="#64748B">{other_max:.2f}</text>')
    lines.append(f'<text x="{x_other:.3f}" y="{base_y + 10.0:.3f}" font-family="Arial, Helvetica, sans-serif" font-size="5.5" text-anchor="middle" fill="#262626">others</text>')
    lines.append('</svg>')

    out = FIG_DIR / 'fig1_protein_energy_micro_inset_vertical_ai_safe.svg'
    out.write_text('\n'.join(lines), encoding='utf-8')
    return out


def write_protein_energy_horizontal_strip_ai_svg(summary: pd.DataFrame):
    df = summary.dropna(subset=['protein_energy']).copy()
    df = df.sort_values('protein_energy').reset_index(drop=True)
    pcfm = float(df.loc[df.method.eq('CITE-PCFM'), 'protein_energy'].iloc[0])
    baselines = df[~df.method.eq('CITE-PCFM')].copy()
    crufm = float(baselines.loc[baselines.method.eq('crufm'), 'protein_energy'].iloc[0])
    other = baselines[~baselines.method.eq('crufm')]['protein_energy'].to_numpy(dtype=float)
    other_min = float(np.nanmin(other))
    other_max = float(np.nanmax(other))
    improvement = (crufm - pcfm) / crufm * 100 if crufm > 0 else np.nan

    width, height = 250.0, 58.0
    left, right = 18.0, 10.0
    axis_y = 32.0
    axis_w = width - left - right
    x_min, x_max = 0.018, 1.05
    log_min, log_max = math.log10(x_min), math.log10(x_max)

    def sx(v: float) -> float:
        frac = (math.log10(v) - log_min) / (log_max - log_min)
        return left + axis_w * frac

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}pt" height="{height}pt" viewBox="0 0 {width} {height}">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        '<text x="6" y="12" font-family="Arial, Helvetica, sans-serif" font-size="7.8" font-weight="700" fill="#111827">Surface-protein distribution error</text>',
        f'<text x="6" y="23" font-family="Arial, Helvetica, sans-serif" font-size="5.8" fill="#475569">RNA-only PCFM is {improvement:.1f}% lower than best paired-input baseline</text>',
        f'<line x1="{left}" y1="{axis_y}" x2="{left + axis_w}" y2="{axis_y}" stroke="#CBD5E1" stroke-width="1.0"/>',
    ]

    for tick, label in [(0.02, '0.02'), (0.03, '0.03'), (0.1, '0.1'), (0.3, '0.3'), (1.0, '1.0')]:
        x = sx(tick)
        lines.append(f'<line x1="{x:.3f}" y1="{axis_y - 3}" x2="{x:.3f}" y2="{axis_y + 3}" stroke="#CBD5E1" stroke-width="0.7"/>')
        lines.append(f'<text x="{x:.3f}" y="{axis_y + 14}" font-family="Arial, Helvetica, sans-serif" font-size="4.8" text-anchor="middle" fill="#64748B">{label}</text>')

    x_other0, x_other1 = sx(other_min), sx(other_max)
    lines.append(f'<line x1="{x_other0:.3f}" y1="{axis_y}" x2="{x_other1:.3f}" y2="{axis_y}" stroke="#CBD5E1" stroke-width="5.0" stroke-linecap="round"/>')
    lines.append(f'<circle cx="{x_other0:.3f}" cy="{axis_y}" r="2.3" fill="#CBD5E1" stroke="#ffffff" stroke-width="0.7"/>')
    lines.append(f'<circle cx="{x_other1:.3f}" cy="{axis_y}" r="2.3" fill="#CBD5E1" stroke="#ffffff" stroke-width="0.7"/>')
    lines.append(f'<text x="{(x_other0 + x_other1) / 2:.3f}" y="{axis_y - 7.2}" font-family="Arial, Helvetica, sans-serif" font-size="5.2" text-anchor="middle" fill="#64748B">other baselines</text>')

    x_pcfm, x_crufm = sx(pcfm), sx(crufm)
    lines.append(f'<circle cx="{x_pcfm:.3f}" cy="{axis_y}" r="4.1" fill="#6D28D9" stroke="#ffffff" stroke-width="0.9"/>')
    lines.append(f'<circle cx="{x_crufm:.3f}" cy="{axis_y}" r="3.6" fill="#23825F" stroke="#ffffff" stroke-width="0.9"/>')
    lines.append(f'<text x="{x_pcfm:.3f}" y="{axis_y - 8.4}" font-family="Arial, Helvetica, sans-serif" font-size="5.5" text-anchor="middle" fill="#4C1D95">PCFM</text>')
    lines.append(f'<text x="{x_crufm + 13.0:.3f}" y="{axis_y - 8.4}" font-family="Arial, Helvetica, sans-serif" font-size="5.5" text-anchor="middle" fill="#23825F">CRUFM</text>')
    lines.append(f'<text x="{x_pcfm:.3f}" y="{axis_y + 23.5}" font-family="Arial, Helvetica, sans-serif" font-size="4.8" text-anchor="middle" fill="#4C1D95">{pcfm:.4f}</text>')
    lines.append(f'<text x="{x_crufm + 13.0:.3f}" y="{axis_y + 23.5}" font-family="Arial, Helvetica, sans-serif" font-size="4.8" text-anchor="middle" fill="#23825F">{crufm:.4f}</text>')
    lines.append('</svg>')

    out = FIG_DIR / 'fig1_protein_energy_horizontal_strip_ai_safe.svg'
    out.write_text('\n'.join(lines), encoding='utf-8')
    return out


def write_protein_energy_vertical_compact_ai_svg(summary: pd.DataFrame):
    df = summary.dropna(subset=['protein_energy']).copy()
    df = df.sort_values('protein_energy').reset_index(drop=True)
    pcfm = float(df.loc[df.method.eq('CITE-PCFM'), 'protein_energy'].iloc[0])
    baselines = df[~df.method.eq('CITE-PCFM')].copy()
    crufm = float(baselines.loc[baselines.method.eq('crufm'), 'protein_energy'].iloc[0])
    best_baseline_method = 'crufm'
    improvement = (crufm - pcfm) / crufm * 100 if crufm > 0 else np.nan
    labels = {
        'CITE-PCFM': 'PCFM',
        'crufm': 'CR',
        'vgfm': 'VG',
        'sf2m': 'SF',
        'balanced_ot_cfm': 'Bal',
        'unbalanced_ot': 'UOT',
        'ruot': 'RU',
        'dynamical_ot': 'DOT',
        'cyto_simulation': 'Cy',
    }

    width, height = 178.0, 142.0
    left, right, top, bottom = 24.0, 6.0, 31.0, 22.0
    plot_w = width - left - right
    plot_h = height - top - bottom
    y_min, y_max = 0.018, 1.05
    log_min, log_max = math.log10(y_min), math.log10(y_max)

    def sx(i: int) -> float:
        return left + (i + 0.5) * plot_w / len(df)

    def sy(v: float) -> float:
        frac = (math.log10(v) - log_min) / (log_max - log_min)
        return top + plot_h * (1.0 - frac)

    def esc(text: str) -> str:
        return html.escape(str(text), quote=True)

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}pt" height="{height}pt" viewBox="0 0 {width} {height}">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        '<text x="24" y="12" font-family="Arial, Helvetica, sans-serif" font-size="7.6" font-weight="700" fill="#111827">Surface-protein error</text>',
        f'<text x="24" y="23" font-family="Arial, Helvetica, sans-serif" font-size="5.2" fill="#475569">PCFM RNA-only; baselines RNA+ADT; {improvement:.1f}% lower than CR</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#ffffff"/>',
    ]

    for tick, label in [(0.02, '0.02'), (0.03, '0.03'), (0.1, '0.1'), (0.3, '0.3'), (1.0, '1')]:
        y = sy(tick)
        lines.append(f'<line x1="{left}" y1="{y:.3f}" x2="{left + plot_w}" y2="{y:.3f}" stroke="#E2E8F0" stroke-width="0.72"/>')
        lines.append(f'<text x="{left - 4}" y="{y + 1.8:.3f}" font-family="Arial, Helvetica, sans-serif" font-size="5.0" text-anchor="end" fill="#64748B">{label}</text>')
    for tick in [0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        y = sy(tick)
        lines.append(f'<line x1="{left}" y1="{y:.3f}" x2="{left + plot_w}" y2="{y:.3f}" stroke="#F1F5F9" stroke-width="0.38"/>')

    cr_y = sy(crufm)
    lines.append(f'<line x1="{left}" y1="{cr_y:.3f}" x2="{left + plot_w}" y2="{cr_y:.3f}" stroke="#64748B" stroke-width="0.9" stroke-dasharray="3.4 2.6"/>')
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#CBD5E1" stroke-width="0.9"/>')
    lines.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#CBD5E1" stroke-width="0.9"/>')

    bar_w = min(11.0, plot_w / len(df) * 0.58)
    for i, row in df.iterrows():
        method = str(row['method'])
        val = float(row['protein_energy'])
        color = '#6D28D9' if method == 'CITE-PCFM' else '#23825F' if method == best_baseline_method else '#CBD5E1'
        x = sx(i)
        y = sy(val)
        h = top + plot_h - y
        lines.append(f'<rect x="{x - bar_w / 2:.3f}" y="{y:.3f}" width="{bar_w:.3f}" height="{h:.3f}" fill="{color}" stroke="#ffffff" stroke-width="0.75"/>')
        lines.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="{2.7 if method == "CITE-PCFM" else 2.2}" fill="{color}" stroke="#ffffff" stroke-width="0.65"/>')
        if method in {'CITE-PCFM', 'crufm', 'vgfm'}:
            label = f'{val:.4f}' if val < 0.1 else f'{val:.2f}'
            text_color = '#4C1D95' if method == 'CITE-PCFM' else '#334155'
            lines.append(f'<text x="{x:.3f}" y="{y - 4.8:.3f}" font-family="Arial, Helvetica, sans-serif" font-size="4.9" text-anchor="middle" fill="{text_color}">{label}</text>')
        lines.append(f'<text x="{x:.3f}" y="{top + plot_h + 11.0:.3f}" font-family="Arial, Helvetica, sans-serif" font-size="5.1" text-anchor="middle" fill="#262626">{esc(labels.get(method, method))}</text>')

    lines.append('</svg>')
    out = FIG_DIR / 'fig1_protein_energy_vertical_compact_ai_safe.svg'
    out.write_text('\n'.join(lines), encoding='utf-8')
    return out


def plot_pareto(summary: pd.DataFrame):
    df = summary.dropna(subset=['w1_mean', 'protein_energy']).copy()
    plt.figure(figsize=(7.2, 5.4))
    ax = sns.scatterplot(data=df, x='w1_mean', y='protein_energy', hue='method', style='method_type', s=120,
                         palette={m: PALETTE.get(m, '#999999') for m in df['method'].unique()})
    ax.set_yscale('log')
    ax.set_xlabel('Wasserstein-1 mean (lower is better)')
    ax.set_ylabel('Protein energy (lower is better; log scale)')
    ax.set_title('Distribution fit vs CITE protein fidelity trade-off')
    for _, r in df.iterrows():
        dx = -0.02 if r['method'] == 'CITE-PCFM' else 0.02
        ax.annotate(r['method'], (r['w1_mean'], r['protein_energy']), xytext=(5, 3), textcoords='offset points', fontsize=8)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
    savefig(FIG_DIR / 'fig2_w1_protein_pareto.png')


def plot_per_time(per_time: pd.DataFrame, summary: pd.DataFrame):
    if per_time.empty:
        return
    keep = ['CITE-PCFM', 'crufm', 'balanced_ot_cfm', 'dynamical_ot', 'vgfm']
    df = per_time[per_time['method'].isin(keep)].copy()
    df['time_point'] = pd.to_numeric(df['time_point'], errors='coerce')
    df = df.dropna(subset=['time_point', 'protein_energy'])
    plt.figure(figsize=(8.2, 5.0))
    ax = sns.lineplot(data=df, x='time_point', y='protein_energy', hue='method', marker='o', linewidth=2.2,
                      palette={m: PALETTE.get(m, '#999999') for m in df['method'].unique()})
    ax.set_yscale('log')
    ax.set_xlabel('Observed target time')
    ax.set_ylabel('Protein energy by target time (log; lower is better)')
    ax.set_title('Time-local CITE protein prediction error from rollout')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
    savefig(FIG_DIR / 'fig3_time_local_protein_energy.png')

    if 'paired_cov_error' in df.columns:
        plt.figure(figsize=(8.2, 5.0))
        ax = sns.lineplot(data=df, x='time_point', y='paired_cov_error', hue='method', marker='o', linewidth=2.2,
                          palette={m: PALETTE.get(m, '#999999') for m in df['method'].unique()})
        ax.set_xlabel('Observed target time')
        ax.set_ylabel('Paired covariance error (lower is better)')
        ax.set_title('Cross-modality covariance preservation over rollout time')
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
        savefig(FIG_DIR / 'fig4_paired_covariance_error_time.png')


def plot_w1_per_gap(summary: pd.DataFrame):
    rows = []
    final = load_metrics(FINAL_METRICS)
    for i, val in enumerate(final.get('w1_scores', []), start=1):
        rows.append({'method': 'CITE-PCFM', 'gap': f'{i-1}→{i}', 'w1': val, 'method_type': 'custom_final'})
    for mpath in sorted(BASELINE_DIR.glob('builtin_*/metrics.json')):
        method = mpath.parent.name.replace('builtin_', '')
        m = load_metrics(mpath)
        for i, val in enumerate(m.get('w1_scores', []) or [], start=1):
            rows.append({'method': method, 'gap': f'{i-1}→{i}', 'w1': val, 'method_type': 'builtin_baseline'})
    df = pd.DataFrame(rows)
    df.to_csv(TABLE_DIR / 'w1_per_gap.csv', index=False)
    keep = ['CITE-PCFM', 'dynamical_ot', 'crufm', 'vgfm', 'balanced_ot_cfm']
    df = df[df.method.isin(keep)]
    plt.figure(figsize=(8.2, 5.0))
    ax = sns.lineplot(data=df, x='gap', y='w1', hue='method', marker='o', linewidth=2.2,
                      palette={m: PALETTE.get(m, '#999999') for m in df['method'].unique()})
    ax.set_xlabel('Adjacent temporal gap')
    ax.set_ylabel('W1 by temporal gap (lower is better)')
    ax.set_title('Distribution-fit guardrail across time gaps')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
    savefig(FIG_DIR / 'fig5_w1_per_gap.png')


def plot_training_losses():
    m = load_metrics(FINAL_METRICS)
    logs = []
    for stage in m.get('training_summary', {}).get('stages', []):
        logs.extend(stage.get('loss_hook_logs', []) or [])
    if not logs:
        return
    df = pd.DataFrame(logs)
    cols = ['cite_pcfm_rna_velocity_loss', 'cite_pcfm_protein_readout_loss', 'cite_pcfm_endpoint_protein_loss']
    keep = ['epoch'] + [c for c in cols if c in df.columns]
    df = df[keep].copy()
    # downsample for readability, but retain endpoints
    if len(df) > 500:
        idx = np.unique(np.r_[np.linspace(0, len(df)-1, 500, dtype=int), [0, len(df)-1]])
        dfp = df.iloc[idx]
    else:
        dfp = df
    long = dfp.melt(id_vars='epoch', var_name='loss_component', value_name='loss')
    label_map = {
        'cite_pcfm_rna_velocity_loss': 'RNA velocity supervision',
        'cite_pcfm_protein_readout_loss': 'Protein readout',
        'cite_pcfm_endpoint_protein_loss': 'Endpoint protein readout',
    }
    long['loss_component'] = long['loss_component'].map(label_map).fillna(long['loss_component'])
    plt.figure(figsize=(8.2, 5.0))
    ax = sns.lineplot(data=long, x='epoch', y='loss', hue='loss_component', linewidth=1.8)
    ax.set_yscale('log')
    ax.set_xlabel('Training epoch')
    ax.set_ylabel('Loss component (log scale)')
    ax.set_title('CITE-PCFM learns protein readout while preserving RNA dynamics')
    ax.legend(frameon=False)
    savefig(FIG_DIR / 'fig6_training_loss_components.png')


def plot_model_native_trajectory():
    if not FINAL_TRAJ.exists() or not DATA_PATH.exists():
        return
    import scanpy as sc

    adata = sc.read_h5ad(DATA_PATH)
    obs_lat = np.asarray(adata.obsm['X_latent'])
    time_obs = np.asarray(adata.obs['time_point_processed'], dtype=float)
    z = np.load(FINAL_TRAJ, allow_pickle=True)
    times = z['time_points']
    obs_times = z['observed_time_points']
    obs_indices = z['observed_time_indices']
    # use trajectory means plus a deterministic sample for PCA fit
    rng = np.random.default_rng(7)
    sample_obs_idx = rng.choice(obs_lat.shape[0], size=min(2500, obs_lat.shape[0]), replace=False)
    traj_sample = []
    traj_time = []
    center_rows = []
    for j, t in enumerate(times):
        pts = z[f'points_{j:06d}']
        center_rows.append({'time': float(t), **{f'd{i}': v for i, v in enumerate(pts.mean(axis=0))}})
        if j % 5 == 0 or j in set(obs_indices.tolist()):
            idx = rng.choice(pts.shape[0], size=min(150, pts.shape[0]), replace=False)
            traj_sample.append(pts[idx])
            traj_time.extend([float(t)] * len(idx))
    Xfit = np.vstack([obs_lat[sample_obs_idx], *traj_sample])
    pca = PCA(n_components=2, random_state=7).fit(Xfit)
    obs2 = pca.transform(obs_lat[sample_obs_idx])
    traj2 = pca.transform(np.vstack(traj_sample))
    centers = []
    obs_centers = []
    for j, t in enumerate(times):
        pts = z[f'points_{j:06d}']
        centers.append(pca.transform(pts.mean(axis=0, keepdims=True))[0])
    for t in obs_times:
        pts = obs_lat[time_obs == t]
        obs_centers.append(pca.transform(pts.mean(axis=0, keepdims=True))[0])
    centers = np.asarray(centers)
    obs_centers = np.asarray(obs_centers)
    plt.figure(figsize=(7.0, 5.8))
    ax = plt.gca()
    sca = ax.scatter(obs2[:, 0], obs2[:, 1], c=time_obs[sample_obs_idx], s=4, alpha=0.18, cmap='viridis', label='Observed cells')
    ax.plot(centers[:, 0], centers[:, 1], color=PALETTE['CITE-PCFM'], linewidth=2.8, label='CITE-PCFM rollout center')
    ax.scatter(centers[obs_indices, 0], centers[obs_indices, 1], color=PALETTE['CITE-PCFM'], s=55, edgecolor='black', zorder=4, label='Rollout at observed times')
    ax.scatter(obs_centers[:, 0], obs_centers[:, 1], marker='X', color='black', s=75, zorder=5, label='Observed time centroids')
    for k, t in enumerate(obs_times):
        ax.annotate(f't={t:g}', obs_centers[k], xytext=(4, 4), textcoords='offset points', fontsize=8)
    cb = plt.colorbar(sca, ax=ax, pad=0.01)
    cb.set_label('Observed time')
    ax.set_xlabel('Latent PC1')
    ax.set_ylabel('Latent PC2')
    ax.set_title('Model-native continuous rollout in paired RNA/protein latent space')
    ax.legend(frameon=False, loc='best', fontsize=8)
    savefig(FIG_DIR / 'fig7_model_native_rollout_pca.png')

    # RNA/protein subspace drift magnitude along rollout
    rna_cent, prot_cent = [], []
    for j in range(len(times)):
        pts = z[f'points_{j:06d}']
        rna_cent.append(np.linalg.norm(pts[:, :50].mean(axis=0) - z['points_000000'][:, :50].mean(axis=0)))
        prot_cent.append(np.linalg.norm(pts[:, 50:80].mean(axis=0) - z['points_000000'][:, 50:80].mean(axis=0)))
    drift = pd.DataFrame({'time': times, 'RNA subspace centroid drift': rna_cent, 'Protein subspace centroid drift': prot_cent})
    drift.to_csv(TABLE_DIR / 'rollout_subspace_drift.csv', index=False)
    long = drift.melt(id_vars='time', var_name='subspace', value_name='centroid_drift')
    plt.figure(figsize=(7.4, 4.8))
    ax = sns.lineplot(data=long, x='time', y='centroid_drift', hue='subspace', linewidth=2.4)
    for t in obs_times:
        ax.axvline(t, color='gray', alpha=0.18, linewidth=0.8)
    ax.set_xlabel('Continuous rollout time')
    ax.set_ylabel('Centroid displacement from t=0')
    ax.set_title('Continuous RNA/protein latent evolution generated by CITE-PCFM')
    ax.legend(frameon=False)
    savefig(FIG_DIR / 'fig8_rollout_subspace_drift.png')


def main():
    summary, per_time = collect_metric_tables()
    plot_protein_energy_bar(summary)
    plot_pareto(summary)
    plot_per_time(per_time, summary)
    plot_w1_per_gap(summary)
    plot_training_losses()
    plot_model_native_trajectory()

    # concise improvement summary
    ours = summary.loc[summary.method == 'CITE-PCFM'].iloc[0]
    baselines = summary[summary.method != 'CITE-PCFM'].dropna(subset=['protein_energy'])
    best_baseline = baselines.sort_values('protein_energy').iloc[0]
    sota_w1 = summary[summary.method != 'CITE-PCFM'].dropna(subset=['w1_mean']).sort_values('w1_mean').iloc[0]
    manifest = {
        'analysis': 'CITE-PCFM downstream comparison figures',
        'dataset_id': DATASET_ID,
        'final_run_dir': str(FINAL_RUN_DIR),
        'final_metrics_path': str(FINAL_METRICS),
        'final_evaluation_trajectory': str(FINAL_TRAJ),
        'baseline_artifacts_dir': str(BASELINE_DIR),
        'figures_dir': str(FIG_DIR),
        'tables_dir': str(TABLE_DIR),
        'figures': sorted(str(p) for p in FIG_DIR.glob('*.png')),
        'key_results': {
            'cite_pcfm_w1_mean': float(ours['w1_mean']),
            'cite_pcfm_protein_energy': float(ours['protein_energy']),
            'best_baseline_for_protein_energy': best_baseline['method'],
            'best_baseline_protein_energy': float(best_baseline['protein_energy']),
            'protein_energy_relative_reduction_vs_best_baseline_percent': float((best_baseline['protein_energy'] - ours['protein_energy']) / best_baseline['protein_energy'] * 100.0),
            'sota_w1_baseline': sota_w1['method'],
            'sota_w1_mean': float(sota_w1['w1_mean']),
            'w1_note': 'CITE-PCFM is not the W1 SOTA; the validated advantage is rollout-derived CITE protein fidelity with W1 retained as a guardrail.'
        },
        'evidence_boundary': 'Figures use locked final-regression metrics and the saved model-generated evaluation trajectory. Observed target data are used only for scoring/visualization, not to construct predictions.',
    }
    (MANIFEST_DIR / 'downstream_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))

if __name__ == '__main__':
    main()
