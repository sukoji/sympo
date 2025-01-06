"""Reasoning Mode — final report + figures."""
import csv, glob, os
import sys
from pathlib import Path
from statistics import mean, stdev
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parents[1]
EXP = Path(__file__).resolve().parent
os.makedirs(f'{EXP}/figures', exist_ok=True)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BASE))
from autoscore_recompute import enrich_rows_with_autoscore
from eval.llm_judge import JUDGE_MODEL_GEMINI

mpl.rcParams["font.family"] = ["NanumGothic", "DejaVu Sans"]
mpl.rcParams["axes.unicode_minus"] = False

CONDS = ['R_none', 'R_high', 'R_max']
LABELS = {
    'R_none': 'R_none (no reasoning instruction)',
    'R_high':  'R_high (단계적 추론 지시)',
    'R_max':   'R_max (CoT 4-step 강제)',
}
COLORS = {'R_none': '#16a34a', 'R_high': '#f59e0b', 'R_max': '#dc2626'}

rows = list(csv.DictReader(open(f'{EXP}/summary_judged.csv')))
by = defaultdict(list)
for r in rows: by[r['mode']].append(r)

autoscore_rows = []
for path in glob.glob(f'{EXP}/summary_qwen-api_reasoning_*.csv'):
    mode = None
    base = os.path.basename(path)
    if 'reasoning_none' in base:
        mode = 'R_none'
    elif 'reasoning_high' in base:
        mode = 'R_high'
    elif 'reasoning_max' in base:
        mode = 'R_max'
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


def _zoom_limits(means, stds, pad=0.015, min_span=0.06):
    lows = [m - s for m, s in zip(means, stds)]
    highs = [m + s for m, s in zip(means, stds)]
    lo = min(lows) - pad
    hi = max(highs) + pad
    if hi - lo < min_span:
        mid = (hi + lo) / 2
        lo = mid - min_span / 2
        hi = mid + min_span / 2
    return max(0.0, lo), min(1.0, hi)


def _plot_dual_overall(outfile, means, stds, points_by_mode, ylabel, title):
    fig, (ax_full, ax_zoom) = plt.subplots(
        1, 2, figsize=(12.8, 5.4), gridspec_kw={'width_ratios': [1.0, 1.15]}
    )
    x = np.arange(len(CONDS))
    colors = [COLORS[c] for c in CONDS]

    for ax in (ax_full, ax_zoom):
        ax.bar(x, means, 0.55, yerr=stds, capsize=4, color=colors,
               alpha=0.85, edgecolor='black', linewidth=0.7)
        for j, c in enumerate(CONDS):
            for val in points_by_mode.get(c, []):
                ax.scatter([j], [val], color='black', s=22, zorder=5, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[c] for c in CONDS], rotation=10, fontsize=9)
        ax.grid(axis='y', linestyle=':', alpha=0.4)

    ax_full.set_ylabel(ylabel)
    ax_full.set_ylim(0, 1.0)
    ax_full.set_title(f'{title}\nFull Scale')

    lo, hi = _zoom_limits(means, stds)
    ax_zoom.set_ylim(lo, hi)
    ax_zoom.set_title(f'{title}\nZoomed Comparison')
    baseline = means[0]
    ax_zoom.axhline(baseline, color='gray', linestyle='--', linewidth=1, alpha=0.6)
    for i, m in enumerate(means):
        delta = m - baseline
        delta_txt = 'baseline' if i == 0 else f'{delta:+.03f}'
        ax_zoom.text(i, m + stds[i] + 0.004, f'{m:.3f}\n{delta_txt}',
                     ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(outfile, dpi=120)
    plt.close()

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

# ── Fig1: Overall ──
x = np.arange(len(CONDS))
means = [A[c]['overall'][0] or 0 for c in CONDS]
stds  = [A[c]['overall'][1] or 0 for c in CONDS]
judge_points = {
    c: [float(r['overall_median']) for r in by.get(c, []) if float(r['overall_median']) >= 0]
    for c in CONDS
}
_plot_dual_overall(
    f'{EXP}/figures/fig1_overall.png',
    means,
    stds,
    judge_points,
    'Overall (LLM-Judge, archived run summaries)',
    'Fig1. Reasoning Mode — Overall (C3 3R, Gemma-4-26B)',
)
print('✅ fig1_overall.png')

# ── Fig2: Dimensions ──
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)
DIMS = [('S_med','Structure'), ('A_med','Assignment'), ('D_med','Debate')]
for ax, (dim, dn) in zip(axes, DIMS):
    means_d = [A[c][dim][0] or 0 for c in CONDS]
    stds_d  = [A[c][dim][1] or 0 for c in CONDS]
    ax.bar(x, means_d, 0.55, yerr=stds_d, capsize=3,
           color=[COLORS[c] for c in CONDS], alpha=0.85, edgecolor='black', linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels([c.replace('C_','') for c in CONDS], rotation=15, fontsize=10)
    ax.set_title(dn)
    lo, hi = _zoom_limits(means_d, stds_d, pad=0.02, min_span=0.12)
    ax.set_ylim(lo, hi)
    ax.grid(axis='y', linestyle=':', alpha=0.4)
    ax.axhline(means_d[0], color='gray', linestyle='--', linewidth=1, alpha=0.5)
    if dim == 'S_med':
        ax.set_ylabel('Score')
plt.suptitle('Fig2. Dimension Decomposition (archived judge outputs)', y=1.02)
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

# ── Fig5: Autoscore overall ──
auto_means = [A[c]['auto_overall'][0] or 0 for c in CONDS]
auto_stds  = [A[c]['auto_overall'][1] or 0 for c in CONDS]
auto_points = {
    c: [float(r['autoscore_final']) for r in auto_by.get(c, []) if float(r['autoscore_final']) >= 0]
    for c in CONDS
}
_plot_dual_overall(
    f'{EXP}/figures/fig5_autoscore_overall.png',
    auto_means,
    auto_stds,
    auto_points,
    'Overall (Auto score)',
    'Fig5. Reasoning Mode — Overall Auto Score (C3 3R, Gemma-4-26B)',
)
print('✅ fig5_autoscore_overall.png')

# ── Fig6: Autoscore dimensions ──
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)
AUTO_DIMS = [
    ('auto_quality', 'Quality'),
    ('auto_allocation', 'Allocation'),
    ('auto_orchestration', 'Orchestration'),
]
for ax, (dim, dn) in zip(axes, AUTO_DIMS):
    means_d = [A[c][dim][0] or 0 for c in CONDS]
    stds_d  = [A[c][dim][1] or 0 for c in CONDS]
    ax.bar(x, means_d, 0.55, yerr=stds_d, capsize=3,
           color=[COLORS[c] for c in CONDS], alpha=0.85, edgecolor='black', linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels([c.replace('R_','') for c in CONDS], rotation=15, fontsize=10)
    is_saturated = max(means_d) - min(means_d) < 0.005 and min(means_d) > 0.98
    ax.set_title(f'{dn} (guardrail)' if is_saturated else dn)
    lo, hi = ((0.97, 1.005) if is_saturated else _zoom_limits(means_d, stds_d, pad=0.02, min_span=0.10))
    ax.set_ylim(lo, hi)
    ax.grid(axis='y', linestyle=':', alpha=0.4)
    ax.axhline(means_d[0], color='gray', linestyle='--', linewidth=1, alpha=0.5)
    if is_saturated:
        ax.text(1, 0.975, 'Ceiling\nNot discriminative', ha='center', va='bottom',
                fontsize=8, color='gray')
    if dim == 'auto_quality':
        ax.set_ylabel('Score')
plt.suptitle('Fig6. Auto-score Dimension Decomposition', y=1.02)
plt.tight_layout()
plt.savefig(f'{EXP}/figures/fig6_autoscore_dimensions.png', dpi=120, bbox_inches='tight'); plt.close()
print('✅ fig6_autoscore_dimensions.png')

# ── Fig7: Judge overall boxplot ──
fig, ax = plt.subplots(figsize=(9.5, 5.4))
judge_series = [
    [float(r['overall_median']) for r in by.get(c, []) if float(r['overall_median']) >= 0]
    for c in CONDS
]
bp = ax.boxplot(
    judge_series,
    patch_artist=True,
    widths=0.5,
    tick_labels=[c.replace('R_', '') for c in CONDS],
    medianprops=dict(color='black', linewidth=1.4),
    boxprops=dict(linewidth=1.0),
    whiskerprops=dict(linewidth=1.0),
    capprops=dict(linewidth=1.0),
)
for patch, c in zip(bp['boxes'], CONDS):
    patch.set_facecolor(COLORS[c])
    patch.set_alpha(0.45)
ax.set_ylabel('Overall (LLM-Judge)')
ax.set_title('Fig7. Judge Overall Distribution by Reasoning Mode')
ax.set_ylim(*_zoom_limits(
    [mean(v) for v in judge_series],
    [stdev(v) if len(v) > 1 else 0.0 for v in judge_series],
    pad=0.02,
    min_span=0.12,
))
ax.grid(axis='y', linestyle=':', alpha=0.4)
plt.tight_layout()
plt.savefig(f'{EXP}/figures/fig7_judge_boxplot.png', dpi=120)
plt.close()
print('✅ fig7_judge_boxplot.png')

# ── Fig9: Combined Judge × Autoscore overall (paired bars per condition) ──
fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(13.4, 5.4),
                                        gridspec_kw={'width_ratios': [1.0, 1.15]})
xx = np.arange(len(CONDS))
w = 0.36
judge_means = [A[c]['overall'][0] or 0 for c in CONDS]
judge_stds  = [A[c]['overall'][1] or 0 for c in CONDS]
auto_means  = [A[c]['auto_overall'][0] or 0 for c in CONDS]
auto_stds   = [A[c]['auto_overall'][1] or 0 for c in CONDS]
judge_pts = {c: [float(r['overall_median']) for r in by.get(c, []) if float(r['overall_median']) >= 0] for c in CONDS}
auto_pts  = {c: [float(r['autoscore_final']) for r in auto_by.get(c, []) if float(r['autoscore_final']) >= 0] for c in CONDS}

for ax in (ax_full, ax_zoom):
    bj = ax.bar(xx - w/2, judge_means, w, yerr=judge_stds, capsize=4,
                color='#2563eb', alpha=0.85, edgecolor='black', linewidth=0.7,
                label='LLM-Judge (Gemini Pro Preview)')
    ba = ax.bar(xx + w/2, auto_means, w, yerr=auto_stds, capsize=4,
                color='#f59e0b', alpha=0.85, edgecolor='black', linewidth=0.7,
                label='AutoScore (deterministic)')
    for j, c in enumerate(CONDS):
        for v in judge_pts.get(c, []):
            ax.scatter([j - w/2], [v], color='black', s=18, zorder=5, alpha=0.7)
        for v in auto_pts.get(c, []):
            ax.scatter([j + w/2], [v], color='black', s=18, zorder=5, alpha=0.7)
    ax.set_xticks(xx)
    ax.set_xticklabels([LABELS[c] for c in CONDS], rotation=10, fontsize=9)
    ax.grid(axis='y', linestyle=':', alpha=0.4)
    ax.legend(loc='lower right', fontsize=9)

ax_full.set_ylabel('Overall score')
ax_full.set_ylim(0, 1.05)
ax_full.set_title('Fig9. Judge × AutoScore — Overall Comparison\nFull Scale')

lo, hi = _zoom_limits(judge_means + auto_means, judge_stds + auto_stds, pad=0.02, min_span=0.10)
ax_zoom.set_ylim(lo, hi)
ax_zoom.set_title('Fig9. Judge × AutoScore — Overall Comparison\nZoomed')
for j, c in enumerate(CONDS):
    ax_zoom.text(j - w/2, judge_means[j] + judge_stds[j] + 0.004,
                 f'{judge_means[j]:.3f}', ha='center', va='bottom', fontsize=8, color='#1e3a8a')
    ax_zoom.text(j + w/2, auto_means[j] + auto_stds[j] + 0.004,
                 f'{auto_means[j]:.3f}', ha='center', va='bottom', fontsize=8, color='#92400e')

plt.tight_layout()
plt.savefig(f'{EXP}/figures/fig9_judge_vs_autoscore_overall.png', dpi=120)
plt.close()
print('✅ fig9_judge_vs_autoscore_overall.png')

# ── Fig10: Combined Judge × Autoscore dimensions (side-by-side panels) ──
# Judge dims (S/A/D) vs Autoscore dims (Quality/Allocation/Orchestration) — different axes but unified figure
fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), sharey=False)
JUDGE_DIMS = [('S_med','Structure'), ('A_med','Assignment'), ('D_med','Debate')]
AUTO_DIMS_ROW = [('auto_quality','Quality'), ('auto_allocation','Allocation'), ('auto_orchestration','Orchestration')]

for ax, (dim, dn) in zip(axes[0], JUDGE_DIMS):
    means_d = [A[c][dim][0] or 0 for c in CONDS]
    stds_d  = [A[c][dim][1] or 0 for c in CONDS]
    ax.bar(xx, means_d, 0.55, yerr=stds_d, capsize=3,
           color=[COLORS[c] for c in CONDS], alpha=0.85, edgecolor='black', linewidth=0.5)
    ax.set_xticks(xx); ax.set_xticklabels([c.replace('R_','') for c in CONDS], fontsize=10)
    ax.set_title(f'Judge — {dn}')
    lo, hi = _zoom_limits(means_d, stds_d, pad=0.02, min_span=0.12)
    ax.set_ylim(lo, hi)
    ax.axhline(means_d[0], color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.grid(axis='y', linestyle=':', alpha=0.4)
    if dim == 'S_med': ax.set_ylabel('LLM-Judge score')
    for i, m in enumerate(means_d):
        ax.text(i, m + stds_d[i] + 0.004, f'{m:.2f}', ha='center', va='bottom', fontsize=8)

for ax, (dim, dn) in zip(axes[1], AUTO_DIMS_ROW):
    means_d = [A[c][dim][0] or 0 for c in CONDS]
    stds_d  = [A[c][dim][1] or 0 for c in CONDS]
    ax.bar(xx, means_d, 0.55, yerr=stds_d, capsize=3,
           color=[COLORS[c] for c in CONDS], alpha=0.85, edgecolor='black', linewidth=0.5,
           hatch='//')
    ax.set_xticks(xx); ax.set_xticklabels([c.replace('R_','') for c in CONDS], fontsize=10)
    is_saturated = max(means_d) - min(means_d) < 0.005 and min(means_d) > 0.98
    ax.set_title(f'AutoScore — {dn}{" (ceiling)" if is_saturated else ""}')
    lo, hi = ((0.97, 1.005) if is_saturated else _zoom_limits(means_d, stds_d, pad=0.02, min_span=0.10))
    ax.set_ylim(lo, hi)
    ax.axhline(means_d[0], color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.grid(axis='y', linestyle=':', alpha=0.4)
    if dim == 'auto_quality': ax.set_ylabel('AutoScore')
    for i, m in enumerate(means_d):
        ax.text(i, m + stds_d[i] + 0.002, f'{m:.3f}', ha='center', va='bottom', fontsize=8)

plt.suptitle('Fig10. Judge × AutoScore — Dimension Decomposition (top: LLM-Judge / bottom: AutoScore)', y=1.00)
plt.tight_layout()
plt.savefig(f'{EXP}/figures/fig10_judge_vs_autoscore_dimensions.png', dpi=120, bbox_inches='tight')
plt.close()
print('✅ fig10_judge_vs_autoscore_dimensions.png')

# ── Fig8: Autoscore overall boxplot ──
fig, ax = plt.subplots(figsize=(9.5, 5.4))
auto_series = [
    [float(r['autoscore_final']) for r in auto_by.get(c, []) if float(r['autoscore_final']) >= 0]
    for c in CONDS
]
bp = ax.boxplot(
    auto_series,
    patch_artist=True,
    widths=0.5,
    tick_labels=[c.replace('R_', '') for c in CONDS],
    medianprops=dict(color='black', linewidth=1.4),
    boxprops=dict(linewidth=1.0),
    whiskerprops=dict(linewidth=1.0),
    capprops=dict(linewidth=1.0),
)
for patch, c in zip(bp['boxes'], CONDS):
    patch.set_facecolor(COLORS[c])
    patch.set_alpha(0.45)
for i, c in enumerate(CONDS, start=1):
    vals = auto_series[i-1]
    offsets = np.linspace(-0.08, 0.08, len(vals)) if vals else []
    for off, v in zip(offsets, vals):
        ax.scatter(i + off, v, color=COLORS[c], edgecolor='black', s=45, zorder=4, alpha=0.9)
        ax.text(i + off, v + 0.004, f'{v:.3f}', ha='center', va='bottom', fontsize=7)
ax.set_ylabel('Overall (Auto score)')
ax.set_title('Fig8. Autoscore Overall Distribution by Reasoning Mode')
ax.set_ylim(*_zoom_limits(
    [mean(v) for v in auto_series],
    [stdev(v) if len(v) > 1 else 0.0 for v in auto_series],
    pad=0.02,
    min_span=0.12,
))
ax.grid(axis='y', linestyle=':', alpha=0.4)
plt.tight_layout()
plt.savefig(f'{EXP}/figures/fig8_autoscore_boxplot.png', dpi=120)
plt.close()
print('✅ fig8_autoscore_boxplot.png')

# ── Report ──
def cell(t):
    if t[0] is None: return 'N/A'
    return f'{t[0]:.2f}±{t[1]:.2f}'

with open(f'{EXP}/REASONING_REPORT.md', 'w') as f:
    f.write('# Reasoning Mode Ablation Experiment — Final Report\n\n')
    f.write(f'생성일: {os.popen("date +%Y-%m-%d").read().strip()}  |  N=3, C3 3R, Gemma-4-26B, {JUDGE_MODEL_GEMINI} judge\n\n')
    f.write('## 0. 사전 등록 가설\n\n')
    f.write('- **H1**: R_max > R_high > R_none (Reasoning 정교화 → quality 향상)\n')
    f.write('- **H2**: latency R_max > R_high > R_none (요약 LLM 추가)\n')
    f.write('- **H3**: 모든 mode의 sub-agent failure rate < 10%/run (시스템 안전 영역)\n\n')
    f.write('## 1. 조건 (Reasoning instruction strength, prompt-engineered)\n\n')
    f.write('| Mode | 적용 범위 | Prompt prefix |\n|---|---|---|\n')
    f.write('| **R_none** | 모든 agent | (없음 — default) |\n')
    f.write('| **R_high** | 모든 agent | "단계적으로 추론. 옵션 비교 + trade-off 명시." |\n')
    f.write('| **R_max**  | 모든 agent | CoT 4-step (제약·대안·trade-off·결정) 강제 |\n\n')
    f.write('## 2. 통제\n\n')
    f.write('- 백본/요약: Gemma-4-26B-A4B-it Q4_K_M (localhost:8081, OpenAI-compat endpoint)\n')
    f.write('- 조건: C3_3rounds (3R debate, Task Manager + 5 sub-agents)\n')
    f.write(f'- Judge: {JUDGE_MODEL_GEMINI}, reasoning run 당시 저장된 archived judge output 사용\n')
    f.write('- N=3, 코드 무수정 (monkey-patch)\n\n')
    f.write('## 3. 결과 (archived judge outputs)\n\n')
    f.write('| Mode | Structure | Assignment | Debate | **Overall** | Sub-agent failures |\n|---|---|---|---|---|---|\n')
    for c in CONDS:
        a = A[c]
        f.write(f'| {c} | {cell(a["S_med"])} | {cell(a["A_med"])} | {cell(a["D_med"])} | **{cell(a["overall"])}** | {cell(a["failures"])} |\n')
    f.write('\n## 4. Auto score 결과\n\n')
    f.write('| Mode | Quality | Allocation | Orchestration | **Overall Auto** |\n|---|---|---|---|---|\n')
    for c in CONDS:
        a = A[c]
        f.write(
            f'| {c} | {cell(a["auto_quality"])} | {cell(a["auto_allocation"])} | '
            f'{cell(a["auto_orchestration"])} | **{cell(a["auto_overall"])}** |\n'
        )
    f.write('\n→ `Quality`는 세 mode 모두 **1.00**으로 포화되어, 비교 지표라기보다 **guardrail**로 해석하는 편이 맞습니다. ')
    f.write('즉 reasoning mode 차이는 주로 `Allocation`과 `Orchestration`에서 읽어야 합니다.\n')
    f.write('\n## 5. 가설 평가\n\n')
    om = A['R_none']['overall'][0] or 0
    of = A['R_high']['overall'][0] or 0
    oc = A['R_max']['overall'][0] or 0
    f.write(f'### H1 (R_max > R_high > R_none)\n')
    f.write(f'- R_none = {om:.3f}, R_high = {of:.3f}, R_max = {oc:.3f}\n')
    if oc > of > om:
        f.write(f'- ✅ 지지 (단조 순서 확인)\n\n')
    elif oc > om and of > om:
        f.write(f'- ⚠️ 부분 지지: 둘 다 baseline보다 높지만 high vs max 순서 ambiguous\n\n')
    else:
        f.write(f'- ❌ 기각\n\n')

    fm = A['R_none']['failures'][0] or 0
    ff = A['R_high']['failures'][0] or 0
    fc = A['R_max']['failures'][0] or 0
    f.write(f'### H3 (sub-agent failure < 10%/run)\n')
    f.write(f'- R_none failures/run: {fm:.1f}, R_high: {ff:.1f}, R_max: {fc:.1f}\n')
    if max(fm, ff, fc) < 5:
        f.write(f'- ✅ 안전 영역 — 모든 mode의 sub-agent 호출 안정적\n\n')
    else:
        f.write(f'- ⚠️ 일부 mode에서 sub-agent fail 다수 — 결과 해석 시 confound 고려\n\n')

    f.write('## 6. 한계\n\n')
    f.write('- N=3 — Δ Overall이 σ 내면 통계 유의성 불가, effect size로만 시사적\n')
    f.write('- C3 단일 — 5R+ 더 강한 메모리 압력에서 효과 다를 수 있음\n')
    f.write('- 단일 백본 (Gemma 4-26B 16K) — 더 큰 컨텍스트 모델 미검증\n')
    f.write('- Assignment 차이는 특히 보수적으로 해석해야 함. `R_max` assignment는 0.73/0.60/0.67로 분산이 상대적으로 크고, ')
    f.write('N=3 기준 95% CI가 넓어 (`약 0.51~0.83`) mode 간 우열을 강하게 주장하기 어렵습니다.\n')
    f.write('- 따라서 reasoning mode의 안정적 신호는 Assignment보다 **Debate/Overall 및 autoscore overall**에서 읽는 편이 타당합니다.\n\n')
    f.write('## 7. 산출물\n\n')
    f.write('```\nreasoning_mode_experiment/\n')
    f.write('├── REASONING_REPORT.md\n')
    f.write('├── summary_judged.csv          (archived judge output canonicalized)\n')
    f.write('├── figures/fig1~8.png\n')
    f.write('├── snapshots/                  (9 wbs+debate JSON)\n')
    f.write('├── samples/                    (각 mode별 Reasoning 입력 sample)\n')
    f.write('├── logs/                       (실행 로그)\n')
    f.write('├── run_reasoning.py            (재현 launcher)\n')
    f.write('├── rejudge_median.py           (3-trial median judge)\n')
    f.write('├── build_report.py             (이 figure+report 생성)\n')
    f.write('└── orchestrate.sh              (전체 pipeline)\n')
    f.write('```\n')

print('✅ REASONING_REPORT.md')
