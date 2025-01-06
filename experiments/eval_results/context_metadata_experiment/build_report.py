"""Context Metadata Ablation report + figures.

The preferred source is `summary_judged.csv` from LLM-as-Judge median runs.
If any judge trial/dimension is invalid (-1), fall back to local autoscore
summaries so failed judge calls are never plotted as real zero scores.
"""
import csv, os, json
from statistics import mean, stdev
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

EXP = '/home/piai/ai_course/agent_test/eval_results/context_metadata_experiment'
os.makedirs(f'{EXP}/figures', exist_ok=True)

CONDS = ['M_resume', 'M_disc', 'M_both']
LABELS = {
    'M_resume': 'M_resume (resume only)',
    'M_disc':   'M_disc (eDISC only)',
    'M_both':   'M_both (resume + eDISC)',
}
COLORS = {'M_resume': '#f59e0b', 'M_disc': '#7c3aed', 'M_both': '#16a34a'}

def agg(rs, key):
    vs = [float(r[key]) for r in rs if float(r[key]) >= 0]
    if not vs: return (None, None)
    if len(vs)==1: return (vs[0], 0.0)
    return (mean(vs), stdev(vs))

def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def judge_complete(rows, by, summary):
    if not rows:
        return False
    for c in CONDS:
        if len(by.get(c, [])) != 3:
            return False
        for key in ('S_med', 'A_med', 'D_med', 'overall'):
            if summary[c][key][0] is None:
                return False
        for r in by.get(c, []):
            for key in (
                'S_t1', 'S_t2', 'S_t3', 'S_med',
                'A_t1', 'A_t2', 'A_t3', 'A_med',
                'D_t1', 'D_t2', 'D_t3', 'D_med',
                'overall_median',
            ):
                try:
                    if float(r[key]) < 0:
                        return False
                except (KeyError, TypeError, ValueError):
                    return False
    return True

def load_judge_summary():
    rows = read_csv(f'{EXP}/summary_judged.csv')
    by = defaultdict(list)
    for r in rows:
        by[r['mode']].append(r)
    summary = {c: {
        'S_med':   agg(by.get(c, []), 'S_med'),
        'A_med':   agg(by.get(c, []), 'A_med'),
        'D_med':   agg(by.get(c, []), 'D_med'),
        'overall': agg(by.get(c, []), 'overall_median'),
    } for c in CONDS}
    return rows, by, summary

def load_autoscore_summary():
    gemma = read_csv('/home/piai/ai_course/agent_test/eval_results/gemma26_ablation/summary_qwen-api_gemma26_ablation_20260423_171707.csv')
    disc = read_csv(f'{EXP}/summary_qwen-api_context_disc_only_20260425_021742.csv')
    by = {
        'M_resume': [r for r in gemma if r.get('condition') == 'C3_3rounds'],
        'M_both': [r for r in gemma if r.get('condition') == 'C4_with_disc'],
        'M_disc': disc,
    }
    summary = {c: {
        'quality':       agg(by.get(c, []), 'autoscore_quality'),
        'allocation':    agg(by.get(c, []), 'autoscore_allocation'),
        'orchestration': agg(by.get(c, []), 'autoscore_orchestration'),
        'overall':       agg(by.get(c, []), 'autoscore_final'),
    } for c in CONDS}
    return by, summary

judge_rows, judge_by, judge_summary = load_judge_summary()
use_judge = judge_complete(judge_rows, judge_by, judge_summary)
auto_by, auto_summary = load_autoscore_summary()

if use_judge:
    source_label = 'LLM-Judge median'
    A = judge_summary
    by = judge_by
    overall_key = 'overall'
    dims = [('S_med', 'Structure'), ('A_med', 'Assignment'), ('D_med', 'Debate')]
    y_label = 'Overall (LLM-Judge median)'
    fig_title = 'Fig1. Context Metadata — Overall (LLM-Judge median, N=3)'
else:
    source_label = 'Autoscore fallback (Judge invalid)'
    A = auto_summary
    by = auto_by
    overall_key = 'overall'
    dims = [('quality', 'Quality'), ('allocation', 'Allocation'), ('orchestration', 'Orchestration')]
    y_label = 'Overall (autoscore_final)'
    fig_title = 'Fig1. Context Metadata — Overall (autoscore fallback, N=3)'

# Fig1: Overall
fig, ax = plt.subplots(figsize=(10, 5.5))
x = np.arange(len(CONDS))
means = [A[c][overall_key][0] or 0 for c in CONDS]
stds = [A[c][overall_key][1] or 0 for c in CONDS]
ax.bar(x, means, 0.55, yerr=stds, capsize=4, color=[COLORS[c] for c in CONDS],
       alpha=0.85, edgecolor='black', linewidth=0.7)
if use_judge:
    for j, c in enumerate(CONDS):
        for r in by.get(c, []):
            ov = float(r['overall_median'])
            if ov >= 0: ax.scatter([j], [ov], color='black', s=22, zorder=5, alpha=0.7)
else:
    for j, c in enumerate(CONDS):
        for r in by.get(c, []):
            ov = float(r['autoscore_final'])
            if ov >= 0: ax.scatter([j], [ov], color='black', s=22, zorder=5, alpha=0.7)
ax.set_xticks(x); ax.set_xticklabels([LABELS[c] for c in CONDS], rotation=10, fontsize=9)
ax.set_ylabel(y_label)
ax.set_title(fig_title)
ax.set_ylim(0, 1.0); ax.grid(axis='y', linestyle=':', alpha=0.4)
plt.tight_layout()
plt.savefig(f'{EXP}/figures/fig1_overall.png', dpi=120); plt.close()
print('✅ fig1_overall.png')

# Fig2: 차원 분해 (Assignment 핵심)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
for ax, (dim, dn) in zip(axes, dims):
    means_d = [A[c][dim][0] or 0 for c in CONDS]
    stds_d = [A[c][dim][1] or 0 for c in CONDS]
    ax.bar(x, means_d, 0.55, yerr=stds_d, capsize=3,
           color=[COLORS[c] for c in CONDS], alpha=0.85, edgecolor='black', linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels([c.replace('M_','') for c in CONDS], rotation=15, fontsize=10)
    ax.set_title(dn); ax.set_ylim(0, 1.05); ax.grid(axis='y', linestyle=':', alpha=0.4)
    if dim == 'S_med': ax.set_ylabel('Score')
axes[0].set_ylabel('Score')
plt.suptitle(f'Fig2. Dimension Decomposition ({source_label})', y=1.02)
plt.tight_layout()
plt.savefig(f'{EXP}/figures/fig2_dimensions.png', dpi=120, bbox_inches='tight'); plt.close()
print('✅ fig2_dimensions.png')

# Report
def cell(t):
    if t[0] is None: return 'N/A'
    return f'{t[0]:.2f}±{t[1]:.2f}'

with open(f'{EXP}/CONTEXT_METADATA_REPORT.md', 'w') as f:
    f.write('# Context Metadata Ablation — Report\n\n')
    f.write(f'생성일: {os.popen("date +%Y-%m-%d").read().strip()}  |  Qwen API, N=3  |  Figure source: {source_label}\n\n')
    f.write('## 0. 가설\n\n')
    f.write('- **H1**: M_both > M_resume > M_disc (skill match 정보 + behavior 모두 best)\n')
    f.write('- **H2**: M_disc < M_resume (skill 정보 없으면 assignment 약화)\n\n')
    f.write('## 1. 조건\n\n')
    f.write('| Mode | 이력서 (tech_stack, strengths, yoe) | eDISC (behavior type) | 베이스 |\n|---|---|---|---|\n')
    f.write('| **M_resume** | ✅ 사용 | ❌ | gemma26 C3_3rounds (use_disc=False) 재활용 |\n')
    f.write('| **M_disc**   | ❌ stripped (monkey-patch) | ✅ | NEW: C4 + monkey-patch로 이력서 비움 |\n')
    f.write('| **M_both**   | ✅ 사용 | ✅ | gemma26 C4_with_disc (use_disc=True) 재활용 |\n\n')
    if use_judge:
        f.write('## 2. 결과 (Judge median ×3, N=3)\n\n')
        f.write('| Mode | Structure | Assignment | Debate | **Overall** |\n|---|---|---|---|---|\n')
        for c in CONDS:
            a = A[c]
            f.write(f'| {LABELS[c]} | {cell(a["S_med"])} | {cell(a["A_med"])} | {cell(a["D_med"])} | **{cell(a["overall"])}** |\n')
    else:
        f.write('## 2. 결과 (Autoscore fallback, N=3)\n\n')
        f.write('외부 Judge 결과가 없거나 일부 trial/dimension에 `-1` sentinel이 포함되어 완전한 judged summary로 인정하지 않았습니다. 아래 표와 figures는 로컬 저장 summary의 autoscore 기반입니다.\n\n')
        f.write('| Mode | Quality | Allocation | Orchestration | **Overall** |\n|---|---|---|---|---|\n')
        for c in CONDS:
            a = A[c]
            f.write(f'| {LABELS[c]} | {cell(a["quality"])} | {cell(a["allocation"])} | {cell(a["orchestration"])} | **{cell(a["overall"])}** |\n')
    f.write('\n## 3. 핵심 분석\n\n')
    om = A['M_resume'][overall_key][0] or 0
    od = A['M_disc'][overall_key][0] or 0
    ob = A['M_both'][overall_key][0] or 0
    f.write(f'- **M_resume**: {om:.3f}\n- **M_disc**: {od:.3f}\n- **M_both**: {ob:.3f}\n\n')
    f.write(f'- Δ (both vs resume): {ob-om:+.3f}\n')
    f.write(f'- Δ (resume vs disc): {om-od:+.3f}\n\n')
    f.write('## 4. 한계\n\n')
    f.write('- Gemma-4-26B의 phantom member ID 배정 문제로 Assignment 차원이 모든 mode에서 0 가까움 가능\n')
    f.write('- 시간차 confound: M_resume·M_both는 며칠 전 데이터 재활용\n')
    f.write('- N=3 pilot, 단일 PRD\n')
    f.write('- M_disc는 monkey-patch로 tech_stack 비움 — 시스템 fallback 동작에 영향\n')
    if not use_judge:
        f.write('- Judge 결과의 `-1`은 품질 점수가 아니라 평가 실패 sentinel 값입니다.\n')
print('✅ CONTEXT_METADATA_REPORT.md')
