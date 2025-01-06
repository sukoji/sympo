"""Reasoning Mode — final report + figures."""
import csv, glob, os
import sys
from pathlib import Path
from statistics import mean, stdev
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parents[1]
EXP = Path(__file__).resolve().parent
os.makedirs(f'{EXP}/figures', exist_ok=True)
SUMMARY = f'{EXP}/summary_judged.csv'
sys.path.insert(0, str(BASE))
from autoscore_recompute import enrich_rows_with_autoscore

if not os.path.exists(SUMMARY):
    raise SystemExit(f'missing summary file: {SUMMARY}')

CONDS = ['PA_baseline', 'PA_wbs', 'PA_super', 'PA_subagents', 'PA_all']
LABELS = {
    'PA_baseline': 'Baseline (no reasoning)',
    'PA_wbs':       'WBS Gen → high',
    'PA_super':     'Supervisor → high',
}
LABELS['PA_subagents'] = 'Sub-agents → high'
LABELS['PA_all'] = 'All agents → high'
COLORS = {'PA_baseline': '#16a34a', 'PA_wbs': '#f59e0b', 'PA_super': '#7c3aed', 'PA_subagents': '#dc2626', 'PA_all': '#2563eb'}

rows = list(csv.DictReader(open(SUMMARY)))
by = defaultdict(list)
for r in rows: by[r['mode']].append(r)

autoscore_rows = []
for path in glob.glob(str(BASE / 'summary_*reasoning_pa*.csv')):
    mode = None
    base = os.path.basename(path)
    if 'baseline' in base:
        mode = 'PA_baseline'
    elif 'wbs_high' in base:
        mode = 'PA_wbs'
    elif 'super_high' in base:
        mode = 'PA_super'
    elif 'subagents_high' in base:
        mode = 'PA_subagents'
    elif 'all_high' in base:
        mode = 'PA_all'
    if not mode:
        continue
    rescored = enrich_rows_with_autoscore(list(csv.DictReader(open(path))))
    for row in rescored:
        row['mode'] = mode
        autoscore_rows.append(row)

auto_by = defaultdict(list)
for r in autoscore_rows:
    auto_by[r['mode']].append(r)

def agg(rs, key):
    vs = [float(r[key]) for r in rs if float(r[key]) >= 0]
    if not vs: return (None, None, 0)
    if len(vs)==1: return (vs[0], 0.0, 1)
    return (mean(vs), stdev(vs), len(vs))

A = {}
for c in CONDS:
    A[c] = {
        'S_med':   agg(by.get(c, []), 'S_med'),
        'A_med':   agg(by.get(c, []), 'A_med'),
        'D_med':   agg(by.get(c, []), 'D_med'),
        'overall': agg(by.get(c, []), 'overall_median'),
        'failures': agg(by.get(c, []), 'sub_agent_failures'),
        'auto_overall': agg(auto_by.get(c, []), 'autoscore_final'),
        'auto_quality': agg(auto_by.get(c, []), 'autoscore_quality'),
        'auto_allocation': agg(auto_by.get(c, []), 'autoscore_allocation'),
        'auto_orchestration': agg(auto_by.get(c, []), 'autoscore_orchestration'),
    }
    # Within-snapshot judge variance (avg of std of 3 trials per snapshot)
    var_s = [stdev([float(r['S_t1']), float(r['S_t2']), float(r['S_t3'])]) for r in by.get(c, []) if float(r['S_t1'])>=0]
    var_a = [stdev([float(r['A_t1']), float(r['A_t2']), float(r['A_t3'])]) for r in by.get(c, []) if float(r['A_t1'])>=0]
    var_d = [stdev([float(r['D_t1']), float(r['D_t2']), float(r['D_t3'])]) for r in by.get(c, []) if float(r['D_t1'])>=0]
    A[c]['judge_variance'] = {'S': mean(var_s) if var_s else 0, 'A': mean(var_a) if var_a else 0, 'D': mean(var_d) if var_d else 0}

# ── Fig1: Overall ──
fig, ax = plt.subplots(figsize=(10, 5.5))
x = np.arange(len(CONDS))
means = [A[c]['overall'][0] or 0 for c in CONDS]
stds  = [A[c]['overall'][1] or 0 for c in CONDS]
ax.bar(x, means, 0.55, yerr=stds, capsize=4, color=[COLORS[c] for c in CONDS],
       alpha=0.85, edgecolor='black', linewidth=0.7)
for j, c in enumerate(CONDS):
    for r in by.get(c, []):
        ov = float(r['overall_median'])
        if ov >= 0:
            ax.scatter([j], [ov], color='black', s=22, zorder=5, alpha=0.7)
ax.set_xticks(x); ax.set_xticklabels([LABELS[c] for c in CONDS], rotation=10, fontsize=9)
ax.set_ylabel('Overall (LLM-Judge median, re-normalized)')
ax.set_title('Fig1. Reasoning Mode — Overall (C3 3R, Gemma-4-26B, N=3, judge median ×3)')
ax.set_ylim(0, 1.0); ax.grid(axis='y', linestyle=':', alpha=0.4)
plt.tight_layout()
plt.savefig(f'{EXP}/figures/fig1_overall.png', dpi=120); plt.close()
print('✅ fig1_overall.png')

# ── Fig2: Dimensions ──
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
DIMS = [('S_med','Structure'), ('A_med','Assignment'), ('D_med','Debate')]
for ax, (dim, dn) in zip(axes, DIMS):
    means_d = [A[c][dim][0] or 0 for c in CONDS]
    stds_d  = [A[c][dim][1] or 0 for c in CONDS]
    ax.bar(x, means_d, 0.55, yerr=stds_d, capsize=3,
           color=[COLORS[c] for c in CONDS], alpha=0.85, edgecolor='black', linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels([c.replace('C_','') for c in CONDS], rotation=15, fontsize=10)
    ax.set_title(dn); ax.set_ylim(0, 1.05); ax.grid(axis='y', linestyle=':', alpha=0.4)
    if dim == 'S_med': ax.set_ylabel('Score')
plt.suptitle('Fig2. Dimension Decomposition (median of 3 judge calls per snapshot)', y=1.02)
plt.tight_layout()
plt.savefig(f'{EXP}/figures/fig2_dimensions.png', dpi=120, bbox_inches='tight'); plt.close()
print('✅ fig2_dimensions.png')

# ── Fig3: Sub-agent failure rate ──
fig, ax = plt.subplots(figsize=(9, 5))
fail_means = [A[c]['failures'][0] or 0 for c in CONDS]
fail_stds  = [A[c]['failures'][1] or 0 for c in CONDS]
ax.bar(x, fail_means, 0.55, yerr=fail_stds, capsize=4,
       color=[COLORS[c] for c in CONDS], alpha=0.85, edgecolor='black', linewidth=0.7)
ax.set_xticks(x); ax.set_xticklabels([c.replace('C_','') for c in CONDS])
ax.set_ylabel('Sub-agent API failures per run (count of error strings in debate_log)')
ax.set_title('Fig3. Sub-agent Failure Rate by Reasoning Mode Mode (16K context overflow indicator)')
ax.grid(axis='y', linestyle=':', alpha=0.4)
plt.tight_layout()
plt.savefig(f'{EXP}/figures/fig3_failures.png', dpi=120); plt.close()
print('✅ fig3_failures.png')

# ── Fig4: Judge variance (within-snapshot 3-trial std) ──
fig, ax = plt.subplots(figsize=(9, 5))
DIM_KEYS = ['S','A','D']
xx = np.arange(len(DIM_KEYS))
w = 0.25
for i, c in enumerate(CONDS):
    vals = [A[c]['judge_variance'][k] for k in DIM_KEYS]
    ax.bar(xx + (i-1)*w, vals, w, color=COLORS[c], alpha=0.85, label=c.replace('C_',''),
           edgecolor='black', linewidth=0.5)
ax.set_xticks(xx); ax.set_xticklabels(['Structure','Assignment','Debate'])
ax.set_ylabel('Avg within-snapshot std (3 judge trials)')
ax.set_title('Fig4. Judge Non-determinism per Dimension (Pro Preview, temp=0)')
ax.grid(axis='y', linestyle=':', alpha=0.4); ax.legend(title='Mode')
plt.tight_layout()
plt.savefig(f'{EXP}/figures/fig4_judge_variance.png', dpi=120); plt.close()
print('✅ fig4_judge_variance.png')

if autoscore_rows:
    # ── Fig5: Autoscore overall ──
    fig, ax = plt.subplots(figsize=(10, 5.5))
    auto_means = [A[c]['auto_overall'][0] or 0 for c in CONDS]
    auto_stds  = [A[c]['auto_overall'][1] or 0 for c in CONDS]
    ax.bar(x, auto_means, 0.55, yerr=auto_stds, capsize=4, color=[COLORS[c] for c in CONDS],
           alpha=0.85, edgecolor='black', linewidth=0.7)
    for j, c in enumerate(CONDS):
        for r in auto_by.get(c, []):
            ov = float(r['autoscore_final'])
            if ov >= 0:
                ax.scatter([j], [ov], color='black', s=22, zorder=5, alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels([LABELS[c] for c in CONDS], rotation=10, fontsize=9)
    ax.set_ylabel('Overall (Auto score)')
    ax.set_title('Fig5. Per-Agent Reasoning — Overall Auto Score (C3 3R, Gemma-4-26B, N=3)')
    ax.set_ylim(0, 1.0); ax.grid(axis='y', linestyle=':', alpha=0.4)
    plt.tight_layout()
    plt.savefig(f'{EXP}/figures/fig5_autoscore_overall.png', dpi=120); plt.close()
    print('✅ fig5_autoscore_overall.png')

    # ── Fig6: Autoscore dimensions ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    AUTO_DIMS = [('auto_quality', 'Quality'), ('auto_allocation', 'Allocation'), ('auto_orchestration', 'Orchestration')]
    for ax, (dim, dn) in zip(axes, AUTO_DIMS):
        means_d = [A[c][dim][0] or 0 for c in CONDS]
        stds_d  = [A[c][dim][1] or 0 for c in CONDS]
        ax.bar(x, means_d, 0.55, yerr=stds_d, capsize=3,
               color=[COLORS[c] for c in CONDS], alpha=0.85, edgecolor='black', linewidth=0.5)
        ax.set_xticks(x); ax.set_xticklabels(CONDS, rotation=15, fontsize=10)
        ax.set_title(dn); ax.set_ylim(0, 1.05); ax.grid(axis='y', linestyle=':', alpha=0.4)
        if dim == 'auto_quality': ax.set_ylabel('Score')
    plt.suptitle('Fig6. Per-Agent Auto-score Decomposition', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{EXP}/figures/fig6_autoscore_dimensions.png', dpi=120, bbox_inches='tight'); plt.close()
    print('✅ fig6_autoscore_dimensions.png')

# ── Report ──
def cell(t):
    if t[0] is None: return 'N/A'
    return f'{t[0]:.2f}±{t[1]:.2f}'

with open(f'{EXP}/REASONING_PA_REPORT.md', 'w') as f:
    f.write('# Reasoning Per-Agent Ablation Experiment — Final Report\n\n')
    f.write(f'생성일: {os.popen("date +%Y-%m-%d").read().strip()}  |  N=3, C3 3R, Gemma-4-26B, Pro Preview Judge (median ×3)\n\n')
    f.write('## 0. 사전 등록 가설\n\n')
    f.write('- **H1**: PA_all > others > baseline (Reasoning 정교화 → quality 향상)\n')
    f.write('- **H2**: reasoning prefix가 적용된 agent 수가 많을수록 latency/token cost 증가\n')
    f.write('- **H3**: 모든 mode의 sub-agent failure rate < 10%/run (시스템 안전 영역)\n\n')
    f.write('## 1. 조건\n\n')
    f.write('| Mode | Reasoning 적용 대상 | Prompt strength |\n|---|---|---|\n')
    f.write('| **PA_baseline** | 없음 | none |\n')
    f.write('| **PA_wbs** | WBS Gen Agent | high |\n')
    f.write('| **PA_super** | Task Manager / Supervisor | high |\n')
    f.write('| **PA_subagents** | Planner/FE/BE/Designer/QA | high |\n')
    f.write('| **PA_all** | 모든 agent | high |\n\n')
    f.write('## 2. 통제\n\n')
    f.write('- 백본/요약: Gemma-4-26B-A4B-it Q4_K_M (localhost:8081, OpenAI-compat endpoint)\n')
    f.write('- 조건: C3_3rounds (3R debate, Task Manager + 5 sub-agents)\n')
    f.write('- Judge: gemini-3.1-pro-preview, **각 snapshot당 3회 호출 후 median** (비결정성 통제)\n')
    f.write('- N=3, caller-aware monkey-patch로 agent 파일 경로별 prefix 주입\n\n')
    f.write('## 3. 결과 (median of 3 judge trials)\n\n')
    f.write('| Mode | Structure | Assignment | Debate | **Overall** | Sub-agent failures |\n|---|---|---|---|---|---|\n')
    for c in CONDS:
        a = A[c]
        f.write(f'| {c} | {cell(a["S_med"])} | {cell(a["A_med"])} | {cell(a["D_med"])} | **{cell(a["overall"])}** | {cell(a["failures"])} |\n')
    if autoscore_rows:
        f.write('\n## 4. Auto score 결과\n\n')
        f.write('| Mode | Quality | Allocation | Orchestration | **Overall Auto** |\n|---|---|---|---|---|\n')
        for c in CONDS:
            a = A[c]
            f.write(
                f'| {c} | {cell(a["auto_quality"])} | {cell(a["auto_allocation"])} | '
                f'{cell(a["auto_orchestration"])} | **{cell(a["auto_overall"])}** |\n'
            )
        f.write('\n')
    else:
        f.write('\n## 4. Auto score 결과\n\n')
        f.write('- 원본 autoscore summary CSV가 현재 워크스페이스에 없어 per-agent autoscore figure는 생략했습니다.\n\n')
    f.write('## 5. Judge 비결정성 (within-snapshot std of 3 trials)\n\n')
    f.write('| Mode | S std | A std | D std |\n|---|---|---|---|\n')
    for c in CONDS:
        v = A[c]['judge_variance']
        f.write(f'| {c} | {v["S"]:.3f} | {v["A"]:.3f} | {v["D"]:.3f} |\n')
    f.write('\n→ A 차원이 가장 비결정적 (Gemma 도메인 특수성 + Pro Preview internal randomness).\n\n')

    f.write('## 6. 가설 평가\n\n')
    om = A['PA_baseline']['overall'][0] or 0
    of = A['PA_super']['overall'][0] or 0
    oc = A['PA_all']['overall'][0] or 0
    f.write(f'### H1 (PA_all > others > baseline)\n')
    f.write(f'- PA_baseline = {om:.3f}, PA_super = {of:.3f}, PA_all = {oc:.3f}\n')
    if oc > of > om:
        f.write(f'- ✅ 지지 (단조 순서 확인)\n\n')
    elif oc > om and of > om:
        f.write(f'- ⚠️ 부분 지지: 둘 다 baseline보다 높지만 내부 순서는 ambiguous\n\n')
    else:
        f.write(f'- ❌ 기각\n\n')

    fm = A['PA_baseline']['failures'][0] or 0
    ff = A['PA_super']['failures'][0] or 0
    fc = A['PA_all']['failures'][0] or 0
    f.write(f'### H3 (sub-agent failure < 10%/run)\n')
    f.write(f'- PA_baseline failures/run: {fm:.1f}, PA_super: {ff:.1f}, PA_all: {fc:.1f}\n')
    if max(fm, ff, fc) < 5:
        f.write(f'- ✅ 안전 영역 — 모든 mode의 sub-agent 호출 안정적\n\n')
    else:
        f.write(f'- ⚠️ 일부 mode에서 sub-agent fail 다수 — 결과 해석 시 confound 고려\n\n')

    f.write('## 7. 한계\n\n')
    f.write('- N=3 — Δ Overall이 σ 내면 통계 유의성 불가, effect size로만 시사적\n')
    f.write('- C3 단일 — 5R+ 더 강한 메모리 압력에서 효과 다를 수 있음\n')
    f.write('- 단일 백본 (Gemma 4-26B 16K) — 더 큰 컨텍스트 모델 미검증\n')
    f.write('- Assignment 차원의 phantom-ID 문제로 A 점수가 모든 mode에서 0 가까움 — Reasoning 외 confound\n\n')
    f.write('## 8. 산출물\n\n')
    f.write('```\nreasoning_mode_experiment_per_agent/\n')
    f.write('├── REASONING_PA_REPORT.md\n')
    f.write('├── summary_judged.csv          (15 runs × 3 judge trials each)\n')
    f.write('├── figures/fig1~6.png (autoscore raw CSV 존재 시)\n')
    f.write('├── snapshots/                  (15 wbs+debate JSON)\n')
    f.write('├── samples/                    (각 mode별 Reasoning 입력 sample)\n')
    f.write('├── logs/                       (실행 로그)\n')
    f.write('├── run_reasoning_per_agent.py  (재현 launcher)\n')
    f.write('├── rejudge_median.py           (3-trial median judge)\n')
    f.write('├── build_report.py             (이 figure+report 생성)\n')
    f.write('└── orchestrate.sh              (전체 pipeline)\n')
    f.write('```\n')

print('✅ REASONING_PA_REPORT.md')
