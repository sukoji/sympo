"""Compaction v3 RERUN report — 32K context server."""
import csv, os
from statistics import mean, stdev
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

EXP = '/home/piai/ai_course/agent_test/eval_results/compaction_v3_experiment'
os.makedirs(f'{EXP}/figures', exist_ok=True)

CONDS = ['C_minimal', 'C_filter', 'C_claude']
LABELS = {
    'C_minimal': 'C_minimal (last 30 + PM 10)',
    'C_filter':  'C_filter (last 8 + PM 4) [default]',
    'C_claude':  'C_claude (4K threshold + bounded REPLACE summary)',
}
COLORS = {'C_minimal': '#dc2626', 'C_filter': '#16a34a', 'C_claude': '#2563eb'}

rows = list(csv.DictReader(open(f'{EXP}/summary_judged.csv')))
by = defaultdict(list)
for r in rows: by[r['mode']].append(r)

def agg(rs, key):
    vs = [float(r[key]) for r in rs if float(r[key]) >= 0]
    if not vs: return (None, None)
    if len(vs)==1: return (vs[0], 0.0)
    return (mean(vs), stdev(vs))

A = {c: {
    'S_med':   agg(by.get(c, []), 'S_med'),
    'A_med':   agg(by.get(c, []), 'A_med'),
    'D_med':   agg(by.get(c, []), 'D_med'),
    'overall': agg(by.get(c, []), 'overall_median'),
    'failures': agg(by.get(c, []), 'sub_agent_failures'),
} for c in CONDS}

# Fig1: Overall
fig, ax = plt.subplots(figsize=(10, 5.5))
x = np.arange(len(CONDS))
means = [A[c]['overall'][0] or 0 for c in CONDS]
stds = [A[c]['overall'][1] or 0 for c in CONDS]
ax.bar(x, means, 0.55, yerr=stds, capsize=4, color=[COLORS[c] for c in CONDS],
       alpha=0.85, edgecolor='black', linewidth=0.7)
for j, c in enumerate(CONDS):
    for r in by.get(c, []):
        ov = float(r['overall_median'])
        if ov >= 0: ax.scatter([j], [ov], color='black', s=22, zorder=5, alpha=0.7)
ax.set_xticks(x); ax.set_xticklabels([LABELS[c] for c in CONDS], rotation=10, fontsize=9)
ax.set_ylabel('Overall (LLM-Judge median)')
ax.set_title('Compaction v3 RERUN — Overall (C3 3R, Gemma-4-26B 32K ctx, N=3)')
ax.set_ylim(0, 1.0); ax.grid(axis='y', linestyle=':', alpha=0.4)
plt.tight_layout()
plt.savefig(f'{EXP}/figures/fig1_overall.png', dpi=120); plt.close()
print('✅ fig1_overall.png')

# Fig2: Failures
fig, ax = plt.subplots(figsize=(9, 5))
fmeans = [A[c]['failures'][0] or 0 for c in CONDS]
fstds = [A[c]['failures'][1] or 0 for c in CONDS]
ax.bar(x, fmeans, 0.55, yerr=fstds, capsize=4, color=[COLORS[c] for c in CONDS],
       alpha=0.85, edgecolor='black', linewidth=0.7)
ax.set_xticks(x); ax.set_xticklabels([c.replace('C_','') for c in CONDS])
ax.set_ylabel('Sub-agent failures per run')
ax.set_title('Sub-agent Failures (32K context — should be near 0 if fix worked)')
ax.grid(axis='y', linestyle=':', alpha=0.4)
plt.tight_layout()
plt.savefig(f'{EXP}/figures/fig2_failures.png', dpi=120); plt.close()
print('✅ fig2_failures.png')

# Report
def cell(t):
    if t[0] is None: return 'N/A'
    return f'{t[0]:.2f}±{t[1]:.2f}'

with open(f'{EXP}/COMPACTION_V3_RERUN_REPORT.md','w') as f:
    f.write('# Compaction v3 RERUN — 32K Context\n\n')
    f.write(f'생성일: {os.popen("date +%Y-%m-%d").read().strip()} | Gemma-4-26B Q4_K_M (32K ctx, 3×V100), N=3, Flash Lite Judge ×3 median\n\n')
    f.write('## 변경점 vs 이전\n\n')
    f.write('- 서버 ctx-size: 16K → **32K** (`--ctx-size 32768`)\n')
    f.write('- 가설: 16K 한계 초과로 인한 sub-agent 실패가 사라져야 함\n\n')
    f.write('## 결과\n\n')
    f.write('| Mode | Structure | Assignment | Debate | **Overall** | Failures |\n|---|---|---|---|---|---|\n')
    for c in CONDS:
        a = A[c]
        f.write(f'| {LABELS[c]} | {cell(a["S_med"])} | {cell(a["A_med"])} | {cell(a["D_med"])} | **{cell(a["overall"])}** | {cell(a["failures"])} |\n')
    f.write('\n## 핵심 비교 (vs 16K 결과)\n\n')
    om = A['C_minimal']['overall'][0] or 0
    of = A['C_filter']['overall'][0] or 0
    oc = A['C_claude']['overall'][0] or 0
    fc = A['C_claude']['failures'][0] or 0
    fm = A['C_minimal']['failures'][0] or 0
    ff = A['C_filter']['failures'][0] or 0
    f.write(f'- C_minimal: {om:.3f} (failures: {fm:.1f}/run)\n')
    f.write(f'- C_filter: {of:.3f} (failures: {ff:.1f}/run)\n')
    f.write(f'- C_claude: {oc:.3f} (failures: {fc:.1f}/run)\n\n')
    f.write('이전 16K에서 C_claude failures = 6.33/run → 32K로 변화 정도 보고.\n')
print('✅ COMPACTION_V3_RERUN_REPORT.md')
