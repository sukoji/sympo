"""v3 report: 4 conditions (minimal, filter, claude_v2_buggy, claude_v3_fixed)."""
import csv, os
from statistics import mean, stdev
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

EXP = '/home/piai/ai_course/agent_test/eval_results/compaction_v2_experiment'
os.makedirs(f'{EXP}/figures_v3', exist_ok=True)

CONDS = ['C_minimal', 'C_filter', 'C_claude', 'C_claude_v3']
LABELS = {
    'C_minimal':    'C_minimal (last 30 + PM 10)',
    'C_filter':     'C_filter (last 8 + PM 4) [default]',
    'C_claude':     'C_claude_v2 BUGGY (cumulative summary)',
    'C_claude_v3':  'C_claude_v3 FIXED (bounded REPLACE summary)',
}
COLORS = {'C_minimal': '#dc2626', 'C_filter': '#16a34a', 'C_claude': '#94a3b8', 'C_claude_v3': '#2563eb'}

rows = list(csv.DictReader(open(f'{EXP}/summary_judged_v3.csv')))
by = defaultdict(list)
for r in rows: by[r['mode']].append(r)

def agg(rs, key):
    vs = [float(r[key]) for r in rs if float(r[key]) >= 0]
    if not vs: return (None, None)
    if len(vs)==1: return (vs[0], 0.0)
    return (mean(vs), stdev(vs))

A = {}
for c in CONDS:
    A[c] = {
        'S_med':   agg(by.get(c, []), 'S_med'),
        'A_med':   agg(by.get(c, []), 'A_med'),
        'D_med':   agg(by.get(c, []), 'D_med'),
        'overall': agg(by.get(c, []), 'overall_median'),
        'failures': agg(by.get(c, []), 'sub_agent_failures'),
    }

fig, ax = plt.subplots(figsize=(11, 5.5))
x = np.arange(len(CONDS))
means = [A[c]['overall'][0] or 0 for c in CONDS]
stds = [A[c]['overall'][1] or 0 for c in CONDS]
ax.bar(x, means, 0.55, yerr=stds, capsize=4, color=[COLORS[c] for c in CONDS],
       alpha=0.85, edgecolor='black', linewidth=0.7)
for j, c in enumerate(CONDS):
    for r in by.get(c, []):
        ov = float(r['overall_median'])
        if ov >= 0:
            ax.scatter([j], [ov], color='black', s=22, zorder=5, alpha=0.7)
ax.set_xticks(x); ax.set_xticklabels([LABELS[c] for c in CONDS], rotation=10, fontsize=8)
ax.set_ylabel('Overall (LLM-Judge median)')
ax.set_title('Compaction v3 — Overall (C3 3R, Gemma-4-26B, Pro Preview judge×3)')
ax.set_ylim(0, 1.0); ax.grid(axis='y', linestyle=':', alpha=0.4)
plt.tight_layout()
plt.savefig(f'{EXP}/figures_v3/fig1_overall.png', dpi=120); plt.close()
print('✅ fig1_overall.png')

# Failure bar
fig, ax = plt.subplots(figsize=(10, 5))
fmeans = [A[c]['failures'][0] or 0 for c in CONDS]
fstds = [A[c]['failures'][1] or 0 for c in CONDS]
ax.bar(x, fmeans, 0.55, yerr=fstds, capsize=4, color=[COLORS[c] for c in CONDS],
       alpha=0.85, edgecolor='black', linewidth=0.7)
ax.set_xticks(x); ax.set_xticklabels([c.replace('C_','') for c in CONDS])
ax.set_ylabel('Sub-agent API failures per run')
ax.set_title('Sub-agent Failures by Compaction Mode (lower = better)')
ax.grid(axis='y', linestyle=':', alpha=0.4)
plt.tight_layout()
plt.savefig(f'{EXP}/figures_v3/fig2_failures.png', dpi=120); plt.close()
print('✅ fig2_failures.png')

# Report
def cell(t):
    if t[0] is None: return 'N/A'
    return f'{t[0]:.2f}±{t[1]:.2f}'

with open(f'{EXP}/COMPACTION_V3_REPORT.md', 'w') as f:
    f.write('# Compaction v3 — Bug Fix Report\n\n')
    f.write('이전 v2의 C_claude는 cumulative summary append 버그로 unbounded growth 발생.\n')
    f.write('v3에서 REPLACE 방식으로 수정 (≤2000자 cap, PM 4개로 축소).\n\n')
    f.write('## 결과 (median of 3 judge trials, N=3)\n\n')
    f.write('| Mode | Structure | Assignment | Debate | **Overall** | Sub-agent failures |\n|---|---|---|---|---|---|\n')
    for c in CONDS:
        a = A[c]
        f.write(f'| {LABELS[c]} | {cell(a["S_med"])} | {cell(a["A_med"])} | {cell(a["D_med"])} | **{cell(a["overall"])}** | {cell(a["failures"])} |\n')
    f.write('\n## 핵심 발견\n\n')
    om = A['C_minimal']['overall'][0] or 0
    of = A['C_filter']['overall'][0] or 0
    oc2 = A['C_claude']['overall'][0] or 0
    oc3 = A['C_claude_v3']['overall'][0] or 0
    f.write(f'- C_minimal: {om:.3f}\n- C_filter: {of:.3f}\n- C_claude_v2 (BUGGY): {oc2:.3f}\n- C_claude_v3 (FIXED): {oc3:.3f}\n\n')
    f.write(f'- Δ (v3 vs v2): {oc3-oc2:+.3f}\n')
    f.write(f'- Δ (v3 vs filter): {oc3-of:+.3f}\n\n')
    fc2 = A['C_claude']['failures'][0] or 0
    fc3 = A['C_claude_v3']['failures'][0] or 0
    f.write(f'- v2 sub-agent failures: {fc2:.1f}/run, v3 fixed: {fc3:.1f}/run (Δ={fc3-fc2:+.1f})\n')
print('✅ COMPACTION_V3_REPORT.md')
