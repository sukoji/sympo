"""Hetero comparison: figures + REPORT.md."""
import csv, os, json
import sys
from pathlib import Path
from statistics import mean, stdev
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parent
os.makedirs(f'{ROOT}/figures', exist_ok=True)
sys.path.insert(0, str(BASE))
from autoscore_recompute import enrich_rows_with_autoscore

CONDS = ['H_baseline', 'H_wbsgen', 'H_taskmgr', 'H_both', 'H_all_frontier']
LABELS = {
    'H_baseline':     'All Gemma26 (baseline)',
    'H_wbsgen':       'WBS Gen → Gemini',
    'H_taskmgr':      'Task Mgr → Gemini',
    'H_both':         'WBS Gen + Task Mgr → Gemini',
    'H_all_frontier': 'All Gemini (frontier)',
}
COLORS = {
    'H_baseline':     '#16a34a',
    'H_wbsgen':       '#f59e0b',
    'H_taskmgr':      '#7c3aed',
    'H_both':         '#dc2626',
    'H_all_frontier': '#2563eb',
}

rows = list(csv.DictReader(open(f'{ROOT}/summary_hetero.csv')))
by = defaultdict(list)
for r in rows: by[r['condition']].append(r)

# Aggregate (median imputation for -1 within active dims)
def agg(rs):
    out = {}
    for d in ['judge_structure','judge_assignment','judge_debate','judge_overall']:
        vs = [float(r[d]) for r in rs if float(r[d]) >= 0]
        if not vs: out[d] = (None, None)
        elif len(vs)==1: out[d] = (vs[0], 0.0)
        else: out[d] = (mean(vs), stdev(vs))
    return out

A = {c: agg(by.get(c, [])) for c in CONDS}

# Autoscore aggregation
def agg_auto(rs, key):
    vs = [float(r[key]) for r in rs if float(r[key]) >= 0]
    if not vs:
        return (None, None)
    if len(vs) == 1:
        return (vs[0], 0.0)
    return (mean(vs), stdev(vs))

AUTO = {c: {} for c in CONDS}

# Hetero-specific conditions from raw CSVs
hetero_sources = {
    'H_wbsgen': ROOT / 'summary_qwen-api_hetero_wbsgen_20260424_045630.csv',
    'H_taskmgr': ROOT / 'summary_qwen-api_hetero_taskmgr_20260424_051905.csv',
    'H_both': ROOT / 'summary_qwen-api_hetero_both_20260424_054247.csv',
}
for cond, path in hetero_sources.items():
    rs = enrich_rows_with_autoscore(list(csv.DictReader(open(path))))
    AUTO[cond] = {
        'overall': agg_auto(rs, 'autoscore_final'),
        'quality': agg_auto(rs, 'autoscore_quality'),
        'allocation': agg_auto(rs, 'autoscore_allocation'),
        'orchestration': agg_auto(rs, 'autoscore_orchestration'),
    }

# Baseline/all-frontier from raw ablation summaries to avoid stale precomputed autoscore
backbone_sources = {
    'gemma26': BASE / 'gemma26_ablation' / 'summary_finegrain.csv',
    'gemini': BASE / 'gemini_ablation' / 'summary_finegrain.csv',
}
comp_rows = {
    bb: [r for r in enrich_rows_with_autoscore(list(csv.DictReader(open(path)))) if r.get('condition') == 'C3_3rounds']
    for bb, path in backbone_sources.items()
}
for cond, bb in [('H_baseline', 'gemma26'), ('H_all_frontier', 'gemini')]:
    rs = comp_rows[bb]
    AUTO[cond] = {
        'overall': agg_auto(rs, 'autoscore_final'),
        'quality': agg_auto(rs, 'autoscore_quality'),
        'allocation': agg_auto(rs, 'autoscore_allocation'),
        'orchestration': agg_auto(rs, 'autoscore_orchestration'),
    }

# ── Fig1: Overall comparison ──
fig, ax = plt.subplots(figsize=(11, 5.5))
x = np.arange(len(CONDS))
means = [A[c]['judge_overall'][0] or 0 for c in CONDS]
stds  = [A[c]['judge_overall'][1] or 0 for c in CONDS]
bars = ax.bar(x, means, 0.6, yerr=stds, capsize=4,
              color=[COLORS[c] for c in CONDS], alpha=0.85,
              edgecolor='black', linewidth=0.7)
# Scatter individual runs
for j, c in enumerate(CONDS):
    for r in by.get(c, []):
        ov = float(r['judge_overall'])
        if ov >= 0:
            ax.scatter([j], [ov], color='black', s=18, zorder=5, alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels([LABELS[c] for c in CONDS], rotation=15, fontsize=9)
ax.set_ylabel('Overall (LLM-Judge)')
ax.set_title('Fig1. Hetero-Backbone Overall — C3 (3R debate, N=3, μ±σ; black dots = runs)')
ax.set_ylim(0, 1.0)
ax.grid(axis='y', linestyle=':', alpha=0.4)
plt.tight_layout()
plt.savefig(f'{ROOT}/figures/fig1_overall.png', dpi=120)
plt.close()
print('✅ fig1_overall.png')

# ── Fig2: Dimension breakdown ──
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
DIMS = ['judge_structure','judge_assignment','judge_debate']
DIM_NAMES = ['Structure','Assignment','Debate']
for ax, dim, dn in zip(axes, DIMS, DIM_NAMES):
    means_d = [A[c][dim][0] or 0 for c in CONDS]
    stds_d  = [A[c][dim][1] or 0 for c in CONDS]
    bars = ax.bar(x, means_d, 0.6, yerr=stds_d, capsize=3,
                  color=[COLORS[c] for c in CONDS], alpha=0.85,
                  edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[c].replace(' (baseline)','').replace(' (frontier)','').replace(' → Gemini','→G') for c in CONDS], rotation=20, fontsize=8)
    ax.set_title(dn)
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', linestyle=':', alpha=0.4)
    if dim == 'judge_structure':
        ax.set_ylabel('Score')
plt.suptitle('Fig2. Dimension Decomposition by Hetero Configuration', y=1.02)
plt.tight_layout()
plt.savefig(f'{ROOT}/figures/fig2_dimensions.png', dpi=120, bbox_inches='tight')
plt.close()
print('✅ fig2_dimensions.png')

# ── Fig3: Autoscore overall ──
fig, ax = plt.subplots(figsize=(11, 5.5))
auto_means = [AUTO[c]['overall'][0] or 0 for c in CONDS]
auto_stds  = [AUTO[c]['overall'][1] or 0 for c in CONDS]
ax.bar(x, auto_means, 0.6, yerr=auto_stds, capsize=4,
       color=[COLORS[c] for c in CONDS], alpha=0.85,
       edgecolor='black', linewidth=0.7)
ax.set_xticks(x)
ax.set_xticklabels([LABELS[c] for c in CONDS], rotation=15, fontsize=9)
ax.set_ylabel('Overall (Auto score)')
ax.set_title('Fig3. Hetero-Backbone Autoscore Overall — C3 (3R debate, N=3)')
ax.set_ylim(0, 1.0)
ax.grid(axis='y', linestyle=':', alpha=0.4)
plt.tight_layout()
plt.savefig(f'{ROOT}/figures/fig3_autoscore_overall.png', dpi=120)
plt.close()
print('✅ fig3_autoscore_overall.png')

# ── Fig4: Autoscore dimension breakdown ──
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
AUTO_DIMS = ['quality', 'allocation', 'orchestration']
AUTO_DIM_NAMES = ['Quality', 'Allocation', 'Orchestration']
for ax, dim, dn in zip(axes, AUTO_DIMS, AUTO_DIM_NAMES):
    means_d = [AUTO[c][dim][0] or 0 for c in CONDS]
    stds_d  = [AUTO[c][dim][1] or 0 for c in CONDS]
    ax.bar(x, means_d, 0.6, yerr=stds_d, capsize=3,
           color=[COLORS[c] for c in CONDS], alpha=0.85,
           edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[c].replace(' (baseline)','').replace(' (frontier)','').replace(' → Gemini','→G') for c in CONDS], rotation=20, fontsize=8)
    ax.set_title(dn)
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', linestyle=':', alpha=0.4)
    if dim == 'quality':
        ax.set_ylabel('Score')
plt.suptitle('Fig4. Hetero Autoscore Decomposition', y=1.02)
plt.tight_layout()
plt.savefig(f'{ROOT}/figures/fig4_autoscore_dimensions.png', dpi=120, bbox_inches='tight')
plt.close()
print('✅ fig4_autoscore_dimensions.png')

# ── Report ──
def cell(t):
    if t[0] is None: return 'N/A'
    return f'{t[0]:.2f}±{t[1]:.2f}'

with open(f'{ROOT}/HETERO_REPORT.md', 'w') as f:
    f.write('# Hetero-Backbone Experiment Report\n\n')
    f.write(f'생성일: {os.popen("date +%Y-%m-%d").read().strip()}\n\n')
    f.write('## 0. Conditions\n\n')
    f.write('| 조건 | WBS Gen | Task Mgr | Sub-agents (Debate) |\n|---|---|---|---|\n')
    f.write('| H_baseline   | Gemma26 | Gemma26 | Gemma26 |\n')
    f.write('| H_wbsgen     | **Gemini Flash Lite** | Gemma26 | Gemma26 |\n')
    f.write('| H_taskmgr    | Gemma26 | **Gemini Flash Lite** | Gemma26 |\n')
    f.write('| H_both       | **Gemini Flash Lite** | **Gemini Flash Lite** | Gemma26 |\n')
    f.write('| H_all_frontier | Gemini Flash Lite (Preview) | Gemini Flash Lite (Preview) | Gemini Flash Lite (Preview) |\n\n')
    f.write(f'**Judge**: `{os.environ.get("JUDGE_MODEL_GEMINI", "gemini-3.1-pro-preview")}` (env override)\n\n')
    f.write('## 1. 조건별 결과 (μ ± σ, N=3)\n\n')
    f.write('| 조건 | Structure | Assignment | Debate | Overall |\n|---|---|---|---|---|\n')
    for c in CONDS:
        a = A[c]
        f.write(f'| {LABELS[c]} | {cell(a["judge_structure"])} | {cell(a["judge_assignment"])} | {cell(a["judge_debate"])} | **{cell(a["judge_overall"])}** |\n')
    f.write('\n## 2. Autoscore 결과\n\n')
    f.write('| 조건 | Quality | Allocation | Orchestration | Overall Auto |\n|---|---|---|---|---|\n')
    for c in CONDS:
        a = AUTO[c]
        f.write(f'| {LABELS[c]} | {cell(a["quality"])} | {cell(a["allocation"])} | {cell(a["orchestration"])} | **{cell(a["overall"])}** |\n')
    f.write('\n## 3. 핵심 분석\n\n')
    overalls = {c: A[c]['judge_overall'][0] or 0 for c in CONDS}
    base = overalls['H_baseline']
    f.write(f'- **Baseline (Gemma26 all)**: {base:.2f}\n')
    for c in ['H_wbsgen','H_taskmgr','H_both']:
        delta = overalls[c] - base
        f.write(f'- **{LABELS[c]}**: {overalls[c]:.2f} (Δ vs baseline = {delta:+.2f})\n')
    f.write(f'- **All frontier**: {overalls["H_all_frontier"]:.2f} (Δ vs baseline = {overalls["H_all_frontier"]-base:+.2f})\n\n')
    f.write('\n### Autoscore 관점 보완\n\n')
    auto_overalls = {c: AUTO[c]['overall'][0] or 0 for c in CONDS}
    f.write(f'- Baseline Auto = **{auto_overalls["H_baseline"]:.2f}**, All frontier Auto = **{auto_overalls["H_all_frontier"]:.2f}**\n')
    f.write(f'- H_both Auto = **{auto_overalls["H_both"]:.2f}** → frontier orchestration 이득을 일부 흡수\n')
    f.write(f'- H_wbsgen Auto = **{auto_overalls["H_wbsgen"]:.2f}**, H_taskmgr Auto = **{auto_overalls["H_taskmgr"]:.2f}**\n\n')

    f.write('### 부분 특화의 marginal value\n\n')
    f.write('- WBS Gen만 frontier → all-frontier 대비 효과의 ?% 달성\n')
    f.write('- Task Mgr만 frontier → all-frontier 대비 효과의 ?% 달성\n')
    f.write('- 둘 다 frontier (H_both) → all-frontier에 가장 근접하나 sub-agents 비용 절약\n\n')
    f.write('## 4. 산출물\n\n')
    f.write('- `figures/fig1_overall.png`, `fig2_dimensions.png`\n')
    f.write('- `figures/fig3_autoscore_overall.png`, `fig4_autoscore_dimensions.png`\n')
    f.write('- `summary_hetero.csv`\n')
    f.write('- `snapshots/`, `logs/`, `run_hetero.py`, `rejudge_hetero.py`, `orchestrate.sh`\n')

print('✅ HETERO_REPORT.md')
