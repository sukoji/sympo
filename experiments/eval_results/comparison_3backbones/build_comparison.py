"""3-백본 비교 figure + summary 생성.
입력:
  eval_results/gemma_ablation/summary_finegrain.csv
  eval_results/gemini_ablation/summary_finegrain.csv
  eval_results/qwen_ablation/summary_finegrain.csv
출력:
  figures/fig1_overall.png        — 조건별 Overall 막대(3 backbone × 6 cond) + scatter
  figures/fig2_dimensions.png     — S/A/D 차원 분해 (3행 × 3열)
  figures/fig3_trajectory.png     — 조건 진행에 따른 Overall trajectory(3 backbone)
  figures/fig4_radar.png          — C3 기준 radar (S, A, D)
  COMPARISON_REPORT.md            — 마크다운 비교 리포트
"""
import csv, os
import sys
from pathlib import Path
from statistics import mean, stdev
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from autoscore_recompute import enrich_rows_with_autoscore, recompute_autoscore

ROOT = BASE
OUT = ROOT / 'comparison_4backbones'
os.makedirs(f'{OUT}/figures', exist_ok=True)

CONDS = ['C0_llm_only','C1_with_assign','C2_1round','C3_3rounds','C4_with_disc','C5_5rounds']
COND_LABELS = ['C0 LLM only','C1 +assign','C2 +1R','C3 +3R','C4 +eDISC','C5 +5R']
BACKBONES = ['gemma','qwen','gemma26','gemini']
COLORS = {'gemma':'#d97706','qwen':'#7c3aed','gemma26':'#16a34a','gemini':'#2563eb'}
REMOTE_API = {'gemma': False, 'qwen': False, 'gemma26': False, 'gemini': True}
# 모델 정식명·버전·파라미터 수 (figure legend·report용)
MODEL_INFO = {
    'gemma':   'Gemma-4-E4B-it (~4B)',
    'qwen':    'Qwen3-14B (14B)',
    'gemma26': 'Gemma-4-26B-A4B-it Q4_K_M (26B / 4B active)',
    'gemini':  'Gemini 3.1 Flash Lite Preview',
}

# eval2 §4.4 weights with N/A re-normalization
W = {'structure':0.40, 'assignment':0.35, 'debate':0.25}
# Active dims by condition
ACTIVE = {
    'C0_llm_only':     ['structure'],
    'C1_with_assign':  ['structure','assignment'],
    'C2_1round':       ['structure','assignment','debate'],
    'C3_3rounds':      ['structure','assignment','debate'],
    'C4_with_disc':    ['structure','assignment','debate'],
    'C5_5rounds':      ['structure','assignment','debate'],
}

def read_csv(path, key_map):
    """key_map: {'structure': csv_col, 'assignment': csv_col, 'debate': csv_col}"""
    rows = []
    for r in csv.DictReader(open(path)):
        cond = r.get('condition','')
        if cond not in CONDS: continue
        out = {'condition': cond}
        for k, col in key_map.items():
            v = r.get(col, '')
            try:
                fv = float(v)
            except:
                fv = -1.0
            out[k] = fv
        rows.append(out)
    return rows

def read_autoscore_csv(path):
    base_rows = []
    for r in csv.DictReader(open(path)):
        cond = r.get('condition', '')
        if cond not in CONDS:
            continue
        base_rows.append(r)
    rows = []
    for r in enrich_rows_with_autoscore(base_rows):
        rows.append({
            'condition': r['condition'],
            'overall': float(r['autoscore_final']),
            'quality': float(r['autoscore_quality']),
            'allocation': float(r['autoscore_allocation']),
            'orchestration': float(r['autoscore_orchestration']),
        })
    return rows

# Gemma: prefer patched rejudge_v3 (echo-failure aware) if available, else original
gemma_csv = f'{ROOT}/gemma_ablation/summary_rejudge_v3.csv'
import os as _os
if not _os.path.exists(gemma_csv):
    gemma_csv = f'{ROOT}/gemma_ablation/summary_finegrain.csv'
gemma_all = read_csv(gemma_csv,
                     {'structure':'structure','assignment':'assignment','debate':'debate'})
# Filter to official 17 snapshots (= those listed in pre-rejudge summary_finegrain.csv)
import csv as _csv
_offset = open(f'{ROOT}/gemma_ablation/summary_finegrain.csv')
_official = {row['snapshot'] for row in _csv.DictReader(_offset)}
_offset.close()
# Re-read the rejudge CSV with snapshot names to filter
def _read_with_snap(path, mp):
    out = []
    for r in _csv.DictReader(open(path)):
        cond = r.get('condition','')
        if cond not in CONDS: continue
        if r.get('snapshot','') not in _official: continue
        d = {'condition': cond, 'snapshot': r['snapshot']}
        for k, col in mp.items():
            try: d[k] = float(r.get(col,''))
            except: d[k] = -1.0
        out.append(d)
    return out
gemma = _read_with_snap(gemma_csv,
                        {'structure':'structure','assignment':'assignment','debate':'debate'})

# Manual denylist — parser fallback artifacts in rejudge_v3.
# Judge response was verbose mid-thinking text without proper JSON; parser grabbed first digit
# and clamped to 1.0. Reason field inspection confirms these are NOT real evaluations.
# Format: {(condition, run_id, dim): True}
_GEMMA_DENYLIST = {
    ('C2_1round',    1, 'structure'): True,   # reason: "...A=0.2. Let's double check B..."
    ('C3_3rounds',   1, 'debate'):    True,   # reason: "Let's use 0.8" — judge wanted 0.8 not 1.0
    ('C5_5rounds',   1, 'assignment'): True,  # reason: "Team members: 5. Assigned: 6 (none)..."
    ('C5_5rounds',   1, 'structure'): True,   # reason: "Score for B: ...→ 0.4..."
}
# Note: keyed by (cond, run_id_in_filename, dim). r3 in filename is run 3 of original; we look up by
# CSV row index since rejudge has duplicates; using snapshot-name is safer:
_GEMMA_DENYLIST_BY_SNAP = {
    ('wbs_snapshot_C2_1round_r3_gemma4-api_piai_20260421_011707.json', 'structure'): True,
    ('wbs_snapshot_C3_3rounds_r3_gemma4-api_piai_20260421_022430.json', 'debate'):    True,
    ('wbs_snapshot_C5_5rounds_r2_gemma4-api_piai_20260421_045220.json', 'assignment'):True,
    ('wbs_snapshot_C5_5rounds_r3_gemma4-api_piai_20260421_050058.json', 'structure'): True,
}
# Apply denylist
for r in gemma:
    snap = r.get('snapshot','')
    for d in ['structure','assignment','debate']:
        if (snap, d) in _GEMMA_DENYLIST_BY_SNAP:
            r[d] = -1.0
# Assignment는 결정론적 rule-based (autoscore_allocation, eval2 §5 공식) 사용 — LLM Judge A의 sparse/noise 문제 회피
qwen = read_csv(f'{ROOT}/qwen_ablation/summary_finegrain.csv',
                {'structure':'judge_structure','assignment':'autoscore_allocation','debate':'judge_debate'})
gemini = read_csv(f'{ROOT}/gemini_ablation/summary_finegrain.csv',
                  {'structure':'judge_structure','assignment':'autoscore_allocation','debate':'judge_debate'})
gemma26 = read_csv(f'{ROOT}/gemma26_ablation/summary_finegrain.csv',
                   {'structure':'judge_structure','assignment':'autoscore_allocation','debate':'judge_debate'})

# Gemma는 metadata.json에서 autoscore_allocation 읽어와 별도 매핑 (rejudge_v3에는 rule-based 없음)
import json as _jsn
_gm_md = _jsn.load(open(f'{ROOT}/gemma_ablation/experiment_metadata.json'))
_order_g = sum([[c]*3 for c in CONDS], [])
# (cond, run_pos_in_cond) → autoscore.allocation
_gm_alloc = {}
_per_cond_idx = {c: 0 for c in CONDS}
for c, entry in zip(_order_g, _gm_md):
    _per_cond_idx[c] += 1
    rid = _per_cond_idx[c]
    alloc = recompute_autoscore(dict(entry, condition=c)).get('allocation', None)
    _gm_alloc[(c, rid)] = float(alloc) if alloc is not None else -1.0
# Need to map snapshot filename → (cond, rid). Use existing gemma list with snapshot names + filename pattern
import re as _re
for r in gemma:
    snap = r.get('snapshot','')
    m = _re.match(r'wbs_snapshot_([CR]\d_\w+)_r(\d+)_', snap)
    if m:
        cond = m.group(1)
        rid = int(m.group(2))
        # Map condition name from snapshot to actual key
        cond_map = {'C0_llm': 'C0_llm_only', 'C1_with': 'C1_with_assign',
                    'C2_1round': 'C2_1round', 'C3_3rounds': 'C3_3rounds',
                    'C4_with': 'C4_with_disc', 'C5_5rounds': 'C5_5rounds'}
        # Actually snapshot has full name like C0_llm_only — re-extract
        # 명시적 조건 리스트 alternation — `\w+_(?:...)` 패턴은 greedy 백트래킹 한계로 일부(C2_1round 등) 매칭 실패
        m2 = _re.match(r'wbs_snapshot_(' + '|'.join(CONDS) + r')_r(\d+)_', snap)
        if m2:
            full_cond = m2.group(1)
            rid2 = int(m2.group(2))
            r['assignment'] = _gm_alloc.get((full_cond, rid2), -1.0)

# Autoscore rows for dedicated model-wise autoscore figures
qwen_auto = read_autoscore_csv(f'{ROOT}/qwen_ablation/summary_finegrain.csv')
gemini_auto = read_autoscore_csv(f'{ROOT}/gemini_ablation/summary_finegrain.csv')
gemma26_auto = read_autoscore_csv(f'{ROOT}/gemma26_ablation/summary_finegrain.csv')

gemma_auto = []
for cond, entry in zip(_order_g, _gm_md):
    auto = recompute_autoscore(dict(entry, condition=cond))
    gemma_auto.append({
        'condition': cond,
        'overall': float(auto.get('autoscore', -1.0)),
        'quality': float(auto.get('quality', -1.0)),
        'allocation': float(auto.get('allocation', -1.0)),
        'orchestration': float(auto.get('orchestration', -1.0)),
    })

def overall_renorm(r):
    """Active-dim re-normalized overall. -1 means N/A → exclude + renormalize."""
    cond = r['condition']
    active = ACTIVE[cond]
    valid = [(d, r[d]) for d in active if r[d] >= 0]
    if not valid: return None
    wsum = sum(W[d] for d, _ in valid)
    return sum(W[d] * v for d, v in valid) / wsum

def aggregate(rows):
    """Returns {cond: {dim: (mean, std), 'overall': (mean, std), 'n': N}}.
    ACTIVE 제약: 조건상 평가하지 않는 dim은 N/A로 강제 (예: C0의 Assignment, Debate)."""
    by = defaultdict(list)
    for r in rows: by[r['condition']].append(r)
    agg = {}
    for cond in CONDS:
        rs = by.get(cond, [])
        out = {'n': len(rs)}
        for d in ['structure','assignment','debate']:
            if d not in ACTIVE[cond]:
                out[d] = (None, None)
                continue
            vs = [r[d] for r in rs if r[d] >= 0]
            if not vs:
                out[d] = (None, None)
            elif len(vs) == 1:
                out[d] = (vs[0], 0.0)
            else:
                out[d] = (mean(vs), stdev(vs))
        os_ = [overall_renorm(r) for r in rs]
        os_ = [x for x in os_ if x is not None]
        if not os_:
            out['overall'] = (None, None)
        elif len(os_) == 1:
            out['overall'] = (os_[0], 0.0)
        else:
            out['overall'] = (mean(os_), stdev(os_))
        out['_overalls'] = os_
        agg[cond] = out
    return agg

# ──────────────────────────────────────────────────
# Median imputation for Judge-failed -1 values (noise reduction)
# Only impute -1 within ACTIVE dims (Judge failure ≠ stage absence).
# Use per-(backbone, dim) median of valid scores as fill value.
# ──────────────────────────────────────────────────
from statistics import median
def impute_minus_ones(rows, label=''):
    # Compute backbone-level median per dim from valid scores within active dims
    medians = {}
    for d in ['structure','assignment','debate']:
        vals = [r[d] for r in rows if d in ACTIVE[r['condition']] and r[d] >= 0]
        medians[d] = median(vals) if vals else None
    # Apply imputation
    n_imputed = 0
    for r in rows:
        for d in ACTIVE[r['condition']]:
            if r[d] < 0 and medians[d] is not None:
                r[d] = medians[d]
                n_imputed += 1
    if n_imputed:
        print(f'  [imputation] {label}: filled {n_imputed} -1 values with per-dim medians {medians}')
    return rows

gemma  = impute_minus_ones(gemma,  'gemma')
qwen   = impute_minus_ones(qwen,   'qwen')
gemma26 = impute_minus_ones(gemma26, 'gemma26')
gemini = impute_minus_ones(gemini, 'gemini')

A = {bb: aggregate(d) for bb, d in [('gemma',gemma),('qwen',qwen),('gemma26',gemma26),('gemini',gemini)]}

def aggregate_autoscore(rows):
    by = defaultdict(list)
    for r in rows:
        by[r['condition']].append(r)
    agg = {}
    for cond in CONDS:
        rs = by.get(cond, [])
        out = {'n': len(rs)}
        for d in ['quality', 'allocation', 'orchestration', 'overall']:
            vs = [r[d] for r in rs if r[d] >= 0]
            if not vs:
                out[d] = (None, None)
            elif len(vs) == 1:
                out[d] = (vs[0], 0.0)
            else:
                out[d] = (mean(vs), stdev(vs))
            out[f'_{d}s'] = vs
        agg[cond] = out
    return agg

AUTO = {
    bb: aggregate_autoscore(rows)
    for bb, rows in [
        ('gemma', gemma_auto),
        ('qwen', qwen_auto),
        ('gemma26', gemma26_auto),
        ('gemini', gemini_auto),
    ]
}

# ──────────────────────────────────────────────────
# Timing 데이터 수집 (per backbone × condition)
# ──────────────────────────────────────────────────
import json as _json
TIMING = {bb: {c: [] for c in CONDS} for bb in BACKBONES}

# gemma: metadata.json (run 순서: C0r1..C0r3, C1r1..C1r3, ..., C5r3)
_gm = _json.load(open(f'{ROOT}/gemma_ablation/experiment_metadata.json'))
_order = sum([[c]*3 for c in CONDS], [])
for c, entry in zip(_order, _gm):
    if entry.get('elapsed_sec'):
        TIMING['gemma'][c].append(float(entry['elapsed_sec']))

# qwen, gemma26, gemini: CSV elapsed_sec 컬럼
for bb, path in [('qwen', f'{ROOT}/qwen_ablation/summary_finegrain.csv'),
                 ('gemma26', f'{ROOT}/gemma26_ablation/summary_finegrain.csv'),
                 ('gemini', f'{ROOT}/gemini_ablation/summary_finegrain.csv')]:
    for r in csv.DictReader(open(path)):
        c = r.get('condition','')
        if c in CONDS:
            try: TIMING[bb][c].append(float(r['elapsed_sec']))
            except: pass

# GPU 개수 보정 — 동일 V100 GPU 개수로 정규화한 GPU-time(=compute resource consumed)으로 비교
GPU_COUNT = {'gemma': 2, 'qwen': 3, 'gemma26': 3, 'gemini': 1}  # Gemini는 API라 1로 표기 (참고용)
GPU_NOTE = {'gemma': '2×V100', 'qwen': '3×V100', 'gemma26': '3×V100', 'gemini': 'API'}

T_AGG = {bb: {} for bb in BACKBONES}
for bb in BACKBONES:
    g = GPU_COUNT[bb]
    for c in CONDS:
        vs = TIMING[bb][c]
        if not vs: T_AGG[bb][c] = (None, None, 0)
        elif len(vs) == 1: T_AGG[bb][c] = (vs[0]*g, 0.0, 1)
        else: T_AGG[bb][c] = (mean(vs)*g, stdev(vs)*g, len(vs))

# ──────────────────────────────────────────────────
# API-centric metrics: externalized tokens / external API spend
# Only remote backbone(gemini) counts as external; self-hosted backbones are 0 in API view.
# ──────────────────────────────────────────────────
API_RAW = {
    bb: {c: {'tokens': [], 'cost': []} for c in CONDS}
    for bb in BACKBONES
}

for c, entry in zip(_order_g, _gm_md):
    tok = float(entry.get('token_cost', {}).get('est_total_tokens', 0) or 0)
    cost = float(entry.get('token_cost', {}).get('est_cost_usd', 0) or 0)
    API_RAW['gemma'][c]['tokens'].append(tok)
    API_RAW['gemma'][c]['cost'].append(cost)

for bb, path in [('qwen', f'{ROOT}/qwen_ablation/summary_finegrain.csv'),
                 ('gemma26', f'{ROOT}/gemma26_ablation/summary_finegrain.csv'),
                 ('gemini', f'{ROOT}/gemini_ablation/summary_finegrain.csv')]:
    for r in csv.DictReader(open(path)):
        c = r.get('condition', '')
        if c not in CONDS:
            continue
        try:
            API_RAW[bb][c]['tokens'].append(float(r.get('est_tokens', 0) or 0))
        except:
            pass
        try:
            API_RAW[bb][c]['cost'].append(float(r.get('est_cost_usd', 0) or 0))
        except:
            pass

API_AGG = {bb: {} for bb in BACKBONES}
for bb in BACKBONES:
    for c in CONDS:
        tk = API_RAW[bb][c]['tokens']
        cs = API_RAW[bb][c]['cost']
        API_AGG[bb][c] = {
            'tokens': (mean(tk), stdev(tk) if len(tk) > 1 else 0.0, len(tk)) if tk else (None, None, 0),
            'cost': (mean(cs), stdev(cs) if len(cs) > 1 else 0.0, len(cs)) if cs else (None, None, 0),
        }

# ──────────────────────────────────────────────────
# Fig1: Overall grouped bar + per-run scatter
# ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5.5))
x = np.arange(len(CONDS))
w = 0.21  # 4 bars per group
for i, bb in enumerate(BACKBONES):
    means = [A[bb][c]['overall'][0] or 0 for c in CONDS]
    stds  = [A[bb][c]['overall'][1] or 0 for c in CONDS]
    bars = ax.bar(x + (i-1.5)*w, means, w, yerr=stds, capsize=3,
                  color=COLORS[bb], alpha=0.85, label=MODEL_INFO[bb], edgecolor='black', linewidth=0.5)
    # scatter individual runs
    for j, c in enumerate(CONDS):
        for ov in A[bb][c]['_overalls']:
            ax.scatter([j + (i-1.5)*w], [ov], color='black', s=14, zorder=5, alpha=0.6)
ax.set_xticks(x)
ax.set_xticklabels(COND_LABELS, rotation=15)
ax.set_ylabel('Overall (LLM-Judge, re-normalized)')
ax.set_title('Fig1. Overall Score by Backbone × Condition (μ±σ; black dots = individual runs)')
ax.set_ylim(0, 1.0)
ax.grid(axis='y', linestyle=':', alpha=0.4)
ax.legend(loc='upper left', frameon=True, fontsize=9, title='Backbone (model · params)')
plt.tight_layout()
plt.savefig(f'{OUT}/figures/fig1_overall.png', dpi=120)
plt.close()
print('✅ fig1_overall.png')

# ──────────────────────────────────────────────────
# Fig2: 3차원 분해 (3 subplots: S, A, D)
# ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
DIM_NAMES = ['Structure','Assignment','Debate']
DIMS = ['structure','assignment','debate']
for ax, dim, dn in zip(axes, DIMS, DIM_NAMES):
    for i, bb in enumerate(BACKBONES):
        means = []
        stds = []
        for c in CONDS:
            m, s = A[bb][c][dim]
            means.append(m if m is not None else 0)
            stds.append(s if s is not None else 0)
        bars = ax.bar(x + (i-1.5)*w, means, w, yerr=stds, capsize=2,
                     color=COLORS[bb], alpha=0.85, label=MODEL_INFO[bb], edgecolor='black', linewidth=0.4)
        # Mark N/A with hatch
        for j, c in enumerate(CONDS):
            if A[bb][c][dim][0] is None:
                ax.bar(x[j] + (i-1.5)*w, 1.0, w, color='lightgray', alpha=0.5, hatch='///', edgecolor='gray', linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(COND_LABELS, rotation=20, fontsize=8)
    ax.set_title(dn)
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', linestyle=':', alpha=0.4)
    if dim == 'structure':
        ax.set_ylabel('Score')
        ax.legend(loc='upper right', fontsize=7, title='Backbone')
plt.suptitle('Fig2. Dimension Decomposition (gray hatch = N/A, stage absent)', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUT}/figures/fig2_dimensions.png', dpi=120, bbox_inches='tight')
plt.close()
print('✅ fig2_dimensions.png')

# ──────────────────────────────────────────────────
# Fig3: Overall trajectory (line) — 백본별 ablation 진행 패턴
# ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
for bb in BACKBONES:
    means = [A[bb][c]['overall'][0] or 0 for c in CONDS]
    stds  = [A[bb][c]['overall'][1] or 0 for c in CONDS]
    ax.errorbar(range(len(CONDS)), means, yerr=stds,
                marker='o', markersize=8, linewidth=2.2, capsize=4,
                color=COLORS[bb], label=MODEL_INFO[bb], alpha=0.9)
ax.set_xticks(range(len(CONDS)))
ax.set_xticklabels(COND_LABELS, rotation=15)
ax.set_ylabel('Overall (re-normalized)')
ax.set_title('Fig3. Ablation Trajectory: Component Contribution by Backbone')
# 라운드 효과를 강조하기 위해 데이터 실제 범위에 맞춰 zoom in
ax.set_ylim(0.40, 0.80)
ax.grid(axis='both', linestyle=':', alpha=0.4)
ax.legend(loc='lower right', fontsize=9, title='Backbone (model · params)')
plt.tight_layout()
plt.savefig(f'{OUT}/figures/fig3_trajectory.png', dpi=120)
plt.close()
print('✅ fig3_trajectory.png')

# ──────────────────────────────────────────────────
# Fig4: Radar (C3 = full system) — 3 dim
# ──────────────────────────────────────────────────
from math import pi
fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
dims = ['Structure','Assignment','Debate']
N = len(dims)
angles = [n / N * 2 * pi for n in range(N)] + [0]
for bb in BACKBONES:
    vals = [A[bb]['C3_3rounds'][d][0] or 0 for d in DIMS]
    vals += [vals[0]]
    ax.plot(angles, vals, color=COLORS[bb], linewidth=2.5, label=MODEL_INFO[bb], alpha=0.85)
    ax.fill(angles, vals, color=COLORS[bb], alpha=0.12)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(dims, fontsize=11)
ax.set_ylim(0, 1.0)
ax.set_yticks([0.2, 0.4, 0.6, 0.8])
ax.set_title('Fig4. C3 (3R Debate, full system) — Dimension Radar', y=1.08, fontsize=12)
ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.12), fontsize=9, title='Backbone (model · params)')
plt.tight_layout()
plt.savefig(f'{OUT}/figures/fig4_radar.png', dpi=120, bbox_inches='tight')
plt.close()
print('✅ fig4_radar.png')

# ──────────────────────────────────────────────────
# Fig5: Latency comparison (per condition × backbone, log scale)
# ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5.5))
for i, bb in enumerate(BACKBONES):
    means = [T_AGG[bb][c][0] / 60 if T_AGG[bb][c][0] else 0 for c in CONDS]  # convert to minutes
    stds  = [T_AGG[bb][c][1] / 60 if T_AGG[bb][c][1] else 0 for c in CONDS]
    bars = ax.bar(x + (i-1.5)*w, means, w, yerr=stds, capsize=3,
                  color=COLORS[bb], alpha=0.85, label=MODEL_INFO[bb], edgecolor='black', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(COND_LABELS, rotation=15)
ax.set_ylabel('GPU-minutes per run (wall-clock × GPU count)')
ax.set_title('Fig5. Inference Cost by Backbone × Condition (GPU-normalized; Gemini=API as 1 unit)')
ax.set_yscale('log')
ax.grid(axis='y', linestyle=':', alpha=0.4, which='both')
ax.legend(loc='upper left', fontsize=9, title='Backbone (model · params)')
plt.tight_layout()
plt.savefig(f'{OUT}/figures/fig5_latency.png', dpi=120)
plt.close()
print('✅ fig5_latency.png')

# ──────────────────────────────────────────────────
# Fig6: Pareto front (latency vs Overall) — 4 백본 × 6 조건
# ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6.5))
COND_MARKERS = {'C0_llm_only':'o', 'C1_with_assign':'s', 'C2_1round':'^',
                'C3_3rounds':'D', 'C4_with_disc':'v', 'C5_5rounds':'P'}
all_pts = []  # (latency_min, overall, bb, cond) — for Pareto front
for bb in BACKBONES:
    xs, ys = [], []
    for c in CONDS:
        lat = T_AGG[bb][c][0]
        ov  = A[bb][c]['overall'][0]
        if lat is None or ov is None: continue
        x_min = lat / 60
        ax.scatter([x_min], [ov], color=COLORS[bb], marker=COND_MARKERS[c],
                   s=130, alpha=0.85, edgecolor='black', linewidth=0.8,
                   zorder=5)
        # Annotate condition
        ax.annotate(c.split('_')[0], (x_min, ov), xytext=(5, 5), textcoords='offset points',
                    fontsize=7, alpha=0.7, color='dimgray')
        xs.append(x_min); ys.append(ov)
        all_pts.append((x_min, ov, bb, c))
    # Trajectory line within each backbone (sorted by latency)
    if len(xs) > 1:
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        ax.plot([xs[i] for i in order], [ys[i] for i in order],
                color=COLORS[bb], alpha=0.35, linewidth=1.2, linestyle=':')

# Pareto front (lower-right is dominated; we want upper-left high overall + low latency)
# Point (x, y) is on front if no other point has (x' <= x AND y' >= y AND (x' < x OR y' > y))
front = []
for p in all_pts:
    dominated = False
    for q in all_pts:
        if q is p: continue
        if q[0] <= p[0] and q[1] >= p[1] and (q[0] < p[0] or q[1] > p[1]):
            dominated = True; break
    if not dominated: front.append(p)
front.sort()
if front:
    ax.plot([p[0] for p in front], [p[1] for p in front],
            color='red', linewidth=2.5, alpha=0.7, label='Pareto front', zorder=3)

ax.set_xscale('log')
ax.set_xlabel('GPU-minutes per run (wall-clock × GPU count, log scale)')
ax.set_ylabel('Overall score (re-normalized)')
ax.set_title('Fig6. Performance vs GPU-cost — Pareto frontier (GPU-normalized)')
ax.grid(axis='both', linestyle=':', alpha=0.4, which='both')

# Two legends: backbones (color) + conditions (marker shape)
from matplotlib.lines import Line2D
bb_handles = [Line2D([0],[0], marker='s', color='w', markerfacecolor=COLORS[bb],
                     markersize=10, label=MODEL_INFO[bb], markeredgecolor='black')
              for bb in BACKBONES]
cond_handles = [Line2D([0],[0], marker=COND_MARKERS[c], color='gray', markersize=9,
                       label=COND_LABELS[CONDS.index(c)], linestyle='None')
                for c in CONDS]
leg1 = ax.legend(handles=bb_handles, loc='lower right', fontsize=8, title='Backbone', framealpha=0.9)
ax.add_artist(leg1)
leg2 = ax.legend(handles=cond_handles + [Line2D([0],[0], color='red', linewidth=2, label='Pareto front')],
                 loc='lower left', fontsize=8, title='Condition', framealpha=0.9)
plt.tight_layout()
plt.savefig(f'{OUT}/figures/fig6_pareto.png', dpi=120)
plt.close()
print('✅ fig6_pareto.png')

# ──────────────────────────────────────────────────
# Fig7: Performance + Efficiency in 2-row aligned bars (단순 가시성)
# ──────────────────────────────────────────────────
fig, (axU, axL) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                gridspec_kw={'height_ratios':[1, 1]})
# Top: Overall
for i, bb in enumerate(BACKBONES):
    means = [A[bb][c]['overall'][0] or 0 for c in CONDS]
    stds  = [A[bb][c]['overall'][1] or 0 for c in CONDS]
    axU.bar(x + (i-1.5)*w, means, w, yerr=stds, capsize=3,
            color=COLORS[bb], alpha=0.85, label=MODEL_INFO[bb], edgecolor='black', linewidth=0.5)
axU.set_ylabel('Overall (re-normalized)', fontsize=11)
axU.set_ylim(0, 1.0)
axU.set_title('Performance: LLM-Judge / Rule-based composite (higher = better)', fontsize=10)
axU.grid(axis='y', linestyle=':', alpha=0.4)
axU.legend(loc='upper left', fontsize=9, title='Backbone (model · params)', framealpha=0.95)

# Bottom: Latency (log scale, inverted concept by setting "lower=better" in title)
for i, bb in enumerate(BACKBONES):
    means = [T_AGG[bb][c][0] / 60 if T_AGG[bb][c][0] else 0 for c in CONDS]
    stds  = [T_AGG[bb][c][1] / 60 if T_AGG[bb][c][1] else 0 for c in CONDS]
    axL.bar(x + (i-1.5)*w, means, w, yerr=stds, capsize=3,
            color=COLORS[bb], alpha=0.85, edgecolor='black', linewidth=0.5)
axL.set_ylabel('GPU-minutes per run (log scale)', fontsize=11)
axL.set_yscale('log')
axL.set_title('Efficiency: GPU-cost (wall-clock × GPU count; lower = better)', fontsize=10)
axL.grid(axis='y', linestyle=':', alpha=0.4, which='both')
axL.set_xticks(x); axL.set_xticklabels(COND_LABELS, rotation=15, fontsize=10)
axL.invert_yaxis()  # 아래로 갈수록 느림 → 위쪽이 빠름 (좋음)
# 위쪽이 좋음/위쪽이 좋음 매칭

plt.suptitle('Fig7. Performance vs Efficiency — 4 Backbones × 6 Conditions', fontsize=13, y=0.995)
plt.tight_layout()
plt.savefig(f'{OUT}/figures/fig7_perf_vs_eff.png', dpi=120, bbox_inches='tight')
plt.close()
print('✅ fig7_perf_vs_eff.png')

# ──────────────────────────────────────────────────
# Fig8: Small multiples — 백본별 latency vs Overall trajectory
# ──────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
COND_MARKERS_S = {'C0_llm_only':'o', 'C1_with_assign':'s', 'C2_1round':'^',
                  'C3_3rounds':'D', 'C4_with_disc':'v', 'C5_5rounds':'P'}
for ax, bb in zip(axes.flatten(), BACKBONES):
    xs, ys, lbls = [], [], []
    for c in CONDS:
        lat = T_AGG[bb][c][0]
        ov  = A[bb][c]['overall'][0]
        if lat is None or ov is None: continue
        x_min = lat / 60
        ax.scatter([x_min], [ov], color=COLORS[bb], marker=COND_MARKERS_S[c],
                   s=200, alpha=0.85, edgecolor='black', linewidth=1, zorder=5)
        ax.annotate(c.split('_')[0], (x_min, ov), xytext=(7, 7), textcoords='offset points',
                    fontsize=10, fontweight='bold', color='black')
        xs.append(x_min); ys.append(ov); lbls.append(c)
    # Trajectory line in condition order (C0→C1→C2→C3→C4→C5)
    if len(xs) > 1:
        ax.plot(xs, ys, color=COLORS[bb], alpha=0.4, linewidth=1.5)
    ax.set_xscale('log')
    ax.set_title(MODEL_INFO[bb], fontsize=11, fontweight='bold', color=COLORS[bb])
    ax.grid(axis='both', linestyle=':', alpha=0.4, which='both')
    ax.set_ylim(0.35, 0.85)
# Axis labels only on outer
for ax in axes[1,:]: ax.set_xlabel('Latency per run (minutes, log scale)', fontsize=10)
for ax in axes[:,0]: ax.set_ylabel('Overall (re-normalized)', fontsize=10)
plt.suptitle('Fig8. Performance vs Efficiency — per-backbone trajectories (C0→C5)', fontsize=13, y=0.995)
plt.tight_layout()
plt.savefig(f'{OUT}/figures/fig8_small_multiples.png', dpi=120, bbox_inches='tight')
plt.close()
print('✅ fig8_small_multiples.png')

# ──────────────────────────────────────────────────
# Fig9: Autoscore overall by backbone × condition
# ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5.5))
for i, bb in enumerate(BACKBONES):
    means = [AUTO[bb][c]['overall'][0] or 0 for c in CONDS]
    stds  = [AUTO[bb][c]['overall'][1] or 0 for c in CONDS]
    ax.bar(x + (i-1.5)*w, means, w, yerr=stds, capsize=3,
           color=COLORS[bb], alpha=0.85, label=MODEL_INFO[bb], edgecolor='black', linewidth=0.5)
    for j, c in enumerate(CONDS):
        for ov in AUTO[bb][c]['_overalls']:
            ax.scatter([j + (i-1.5)*w], [ov], color='black', s=14, zorder=5, alpha=0.6)
ax.set_xticks(x)
ax.set_xticklabels(COND_LABELS, rotation=15)
ax.set_ylabel('Overall (autoscore)')
ax.set_title('Fig9. Autoscore Overall by Backbone × Condition (μ±σ; black dots = individual runs)')
ax.set_ylim(0, 1.0)
ax.grid(axis='y', linestyle=':', alpha=0.4)
ax.legend(loc='upper left', frameon=True, fontsize=9, title='Backbone (model · params)')
plt.tight_layout()
plt.savefig(f'{OUT}/figures/fig9_autoscore_overall.png', dpi=120)
plt.close()
print('✅ fig9_autoscore_overall.png')

# ──────────────────────────────────────────────────
# Fig10: Autoscore dimension decomposition
# ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
AUTO_DIMS = ['quality', 'allocation', 'orchestration']
AUTO_DIM_NAMES = ['Quality', 'Allocation', 'Orchestration']
for ax, dim, dn in zip(axes, AUTO_DIMS, AUTO_DIM_NAMES):
    for i, bb in enumerate(BACKBONES):
        means = []
        stds = []
        for c in CONDS:
            m, s = AUTO[bb][c][dim]
            means.append(m if m is not None else 0)
            stds.append(s if s is not None else 0)
        ax.bar(x + (i-1.5)*w, means, w, yerr=stds, capsize=2,
               color=COLORS[bb], alpha=0.85, label=MODEL_INFO[bb], edgecolor='black', linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(COND_LABELS, rotation=20, fontsize=8)
    ax.set_title(dn)
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', linestyle=':', alpha=0.4)
    if dim == 'quality':
        ax.set_ylabel('Score')
        ax.legend(loc='upper right', fontsize=7, title='Backbone')
plt.suptitle('Fig10. Autoscore Dimension Decomposition', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUT}/figures/fig10_autoscore_dimensions.png', dpi=120, bbox_inches='tight')
plt.close()
print('✅ fig10_autoscore_dimensions.png')

# ──────────────────────────────────────────────────
# Fig11: LLM-AutoScore alignment view
# ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5))
ax, ax2 = axes
COND_MARKERS_J = {
    'C0_llm_only': 'o',
    'C1_with_assign': 's',
    'C2_1round': '^',
    'C3_3rounds': 'D',
    'C4_with_disc': 'v',
    'C5_5rounds': 'P',
}

# Left: condition-wise alignment scatter
for bb in BACKBONES:
    for c in CONDS:
        x0 = A[bb][c]['overall'][0]
        y0 = AUTO[bb][c]['overall'][0]
        if x0 is None or y0 is None:
            continue
        ax.scatter(
            [x0], [y0],
            color=COLORS[bb],
            marker=COND_MARKERS_J[c],
            s=150 if bb == 'gemma26' else 110,
            alpha=0.9,
            edgecolor='black',
            linewidth=1.3 if bb == 'gemma26' else 0.8,
            zorder=5,
        )
        ax.annotate(
            c.split('_')[0],
            (x0, y0),
            xytext=(6, 4),
            textcoords='offset points',
            fontsize=8,
            color='dimgray',
        )

ax.set_xlim(0.40, 0.82)
ax.set_ylim(0.72, 0.93)
ax.set_xlabel('LLM Overall')
ax.set_ylabel('Autoscore Overall')
ax.set_title('Condition-wise LLM-AutoScore alignment')
ax.grid(alpha=0.3, linestyle=':')

# Right: backbone centroid over operative conditions
OPERATIVE_CONDS = ['C2_1round', 'C3_3rounds', 'C4_with_disc', 'C5_5rounds']
for bb in BACKBONES:
    xs = [A[bb][c]['overall'][0] for c in OPERATIVE_CONDS if A[bb][c]['overall'][0] is not None]
    ys = [AUTO[bb][c]['overall'][0] for c in OPERATIVE_CONDS if AUTO[bb][c]['overall'][0] is not None]
    if not xs or not ys:
        continue
    mx = mean(xs)
    my = mean(ys)
    sx = stdev(xs) if len(xs) > 1 else 0.0
    sy = stdev(ys) if len(ys) > 1 else 0.0
    ax2.errorbar(
        [mx], [my],
        xerr=[sx], yerr=[sy],
        fmt='o',
        ms=13 if bb == 'gemma26' else 10,
        color=COLORS[bb],
        ecolor=COLORS[bb],
        elinewidth=1.4,
        capsize=4,
        markeredgecolor='black',
        markeredgewidth=1.5 if bb == 'gemma26' else 0.9,
        zorder=6,
    )
    centroid_offsets = {
        'gemma': (8, -26),
        'qwen': (8, 8),
        'gemma26': (12, -14),
        'gemini': (8, 14),
    }
    ax2.annotate(
        f'{bb}\nLLM={mx:.3f}\nAuto={my:.3f}',
        (mx, my),
        xytext=centroid_offsets.get(bb, (8, 6)),
        textcoords='offset points',
        fontsize=8.5,
        fontweight='bold' if bb == 'gemma26' else 'normal',
        color=COLORS[bb],
    )

ax2.set_xlim(0.40, 0.84)
ax2.set_ylim(0.72, 0.93)
ax2.set_xlabel('LLM Overall')
ax2.set_ylabel('Autoscore Overall')
ax2.set_title('Backbone centroid over C2-C5')
ax2.grid(alpha=0.3, linestyle=':')

from matplotlib.lines import Line2D
bb_handles = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS[bb], markersize=10,
           markeredgecolor='black', label=MODEL_INFO[bb])
    for bb in BACKBONES
]
cond_handles = [
    Line2D([0], [0], marker=COND_MARKERS_J[c], color='gray', linestyle='None', markersize=8,
           label=COND_LABELS[CONDS.index(c)])
    for c in CONDS
]
ax.legend(handles=bb_handles, loc='lower right', fontsize=8, title='Backbone', framealpha=0.95)
ax2.legend(handles=cond_handles, loc='lower right', fontsize=8, title='Condition', framealpha=0.95)
plt.suptitle('Fig11. LLM-AutoScore Alignment View', y=0.98, fontsize=13)
plt.tight_layout()
plt.savefig(f'{OUT}/figures/fig11_llm_auto_alignment.png', dpi=120, bbox_inches='tight')
plt.close()
print('✅ fig11_llm_auto_alignment.png')

# ──────────────────────────────────────────────────
# Fig12: API-centric deployment view
# ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5))
ax, ax2 = axes
OPERATIVE_CONDS = ['C2_1round', 'C3_3rounds', 'C4_with_disc', 'C5_5rounds']

for bb in BACKBONES:
    llm_vals = []
    lat_vals = []
    api_tokens = []
    api_costs = []
    for c in OPERATIVE_CONDS:
        llm = A[bb][c]['overall'][0]
        lat = T_AGG[bb][c][0]
        tk = API_AGG[bb][c]['tokens'][0]
        cs = API_AGG[bb][c]['cost'][0]
        if llm is not None:
            llm_vals.append(llm)
        if lat is not None:
            lat_vals.append(lat / 60.0)
        if REMOTE_API[bb] and tk is not None:
            api_tokens.append(tk)
        if REMOTE_API[bb] and cs is not None:
            api_costs.append(cs)
    if not llm_vals:
        continue
    llm_mean = mean(llm_vals)
    lat_mean = mean(lat_vals) if lat_vals else 1.0
    token_mean = mean(api_tokens) if api_tokens else 0.0
    api_cost_mean = mean(api_costs) if api_costs else 0.0
    ax.scatter(
        [token_mean / 1000.0], [llm_mean],
        s=max(130, lat_mean * 1.2),
        color=COLORS[bb],
        alpha=0.88,
        edgecolor='black',
        linewidth=1.4 if bb == 'gemma26' else 0.8,
        zorder=5,
    )
    ax.annotate(
        f'{bb}\nLLM={llm_mean:.3f}',
        (token_mean / 1000.0, llm_mean),
        xytext=(8, 6),
        textcoords='offset points',
        fontsize=9,
        fontweight='bold' if bb == 'gemma26' else 'normal',
        color=COLORS[bb],
    )
    xs = np.array([0, 1_000, 5_000, 10_000, 50_000, 100_000])
    ys = xs * api_cost_mean
    ax2.plot(xs, ys, color=COLORS[bb], linewidth=2.2, alpha=0.9, label=MODEL_INFO[bb])

ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('Externalized tokens per run (thousands)')
ax.set_ylabel('LLM Judge overall')
ax.set_title('LLM quality vs API dependency')
ax.set_xlim(-1, 24)
ax.set_ylim(0.60, 0.82)
ax.grid(alpha=0.3, linestyle=':')

ax2.set_xlabel('Monthly runs')
ax2.set_ylabel('External API spend / month (USD)')
ax2.set_title('Monthly external spend sensitivity')
ax2.grid(alpha=0.3, linestyle=':')
ax2.legend(loc='upper left', fontsize=8, title='Backbone', framealpha=0.95)

plt.suptitle('Fig12. API-Centric Deployment View (external spend only; local infra excluded)', y=0.98, fontsize=13)
plt.tight_layout()
plt.savefig(f'{OUT}/figures/fig12_api_deployment_view.png', dpi=120, bbox_inches='tight')
plt.close()
print('✅ fig12_api_deployment_view.png')

# ──────────────────────────────────────────────────
# Fig13: Gemma-4-26B only — debate-stage AutoScore vs LLM Judge
# ──────────────────────────────────────────────────
ROUND_CONDS = ['C0_llm_only', 'C1_with_assign', 'C2_1round', 'C3_3rounds', 'C5_5rounds']
ROUND_LABELS = ['C0\nLLM only', 'C1\n+assign', 'C2\n+1R', 'C3\n+3R', 'C5\n+5R']

def _parse_score(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f >= 0 else None

gemma26_round_rows = list(csv.DictReader(open(f'{ROOT}/gemma26_ablation/summary_finegrain.csv')))
round_auto = {c: [] for c in ROUND_CONDS}
round_judge = {c: [] for c in ROUND_CONDS}
round_elapsed = {c: [] for c in ROUND_CONDS}
for r in gemma26_round_rows:
    c = r.get('condition', '')
    if c not in ROUND_CONDS:
        continue
    av = _parse_score(r.get('autoscore_final'))
    jv = _parse_score(r.get('judge_overall'))
    tv = _parse_score(r.get('elapsed_sec'))
    if av is not None:
        round_auto[c].append(av)
    if jv is not None:
        round_judge[c].append(jv)
    if tv is not None:
        round_elapsed[c].append(tv)

def _mean_std(vals):
    if not vals:
        return None, None
    if len(vals) == 1:
        return mean(vals), 0.0
    return mean(vals), stdev(vals)

xr = np.arange(len(ROUND_CONDS))
auto_means, auto_stds, judge_means, judge_stds = [], [], [], []
elapsed_means_min, elapsed_stds_min = [], []
for c in ROUND_CONDS:
    m, s = _mean_std(round_auto[c])
    auto_means.append(m if m is not None else np.nan)
    auto_stds.append(s if s is not None else 0.0)
    m, s = _mean_std(round_judge[c])
    judge_means.append(m if m is not None else np.nan)
    judge_stds.append(s if s is not None else 0.0)
    m, s = _mean_std([v / 60.0 for v in round_elapsed[c]])
    elapsed_means_min.append(m if m is not None else np.nan)
    elapsed_stds_min.append(s if s is not None else 0.0)

round_labels = ROUND_LABELS

fig, (ax, ax_eff) = plt.subplots(1, 2, figsize=(15, 6.4), gridspec_kw={'width_ratios': [1.25, 1.0]})
ax.errorbar(
    xr, auto_means, yerr=auto_stds,
    marker='o', markersize=8, linewidth=2.4, capsize=4,
    color=COLORS['gemma26'], label='AutoScore v2 (full-scope)',
)
ax.errorbar(
    xr, judge_means, yerr=judge_stds,
    marker='s', markersize=7, linewidth=2.2, capsize=4,
    color='#111827', label='LLM Judge overall',
)
ax.axvspan(2.72, 3.28, color='#fef3c7', alpha=0.45, zorder=0)
ax.text(3, 0.925, 'SELECT\nC3', ha='center', va='top', fontsize=9, color='#92400e', fontweight='bold')

for i, c in enumerate(ROUND_CONDS):
    for v in round_auto[c]:
        ax.scatter(i - 0.055, v, s=26, color=COLORS['gemma26'], edgecolor='white', linewidth=0.5, alpha=0.85, zorder=5)
    for v in round_judge[c]:
        ax.scatter(i + 0.055, v, s=26, color='#111827', edgecolor='white', linewidth=0.5, alpha=0.85, zorder=5)

ax.set_xticks(xr)
ax.set_xticklabels(round_labels)
ax.set_ylabel('Score')
ax.set_ylim(0.38, 0.94)
ax.set_title('Fig13. Gemma-4-26B Stage Trajectory — AutoScore v2 and LLM Judge')
ax.grid(axis='y', linestyle=':', alpha=0.4)
ax.legend(loc='lower right', framealpha=0.95, fontsize=9)

ax_eff.errorbar(
    elapsed_means_min, judge_means,
    xerr=elapsed_stds_min, yerr=judge_stds,
    color='#2563eb', marker='o', markersize=7, linewidth=1.8, capsize=4,
    label='LLM Judge trajectory',
)
for i, (c, t, j) in enumerate(zip(ROUND_CONDS, elapsed_means_min, judge_means)):
    if np.isnan(t) or np.isnan(j):
        continue
    is_c3 = c == 'C3_3rounds'
    ax_eff.scatter(
        [t], [j],
        s=180 if is_c3 else 90,
        marker='*' if is_c3 else 'o',
        color='#f59e0b' if is_c3 else '#2563eb',
        edgecolor='#111827',
        linewidth=1.1,
        zorder=5,
        label='Selected C3 elbow' if is_c3 else None,
    )
    ax_eff.annotate(
        c.split('_')[0],
        (t, j),
        xytext=(8, 8 if not is_c3 else 12),
        textcoords='offset points',
        fontsize=9,
        fontweight='bold' if is_c3 else 'normal',
        color='#92400e' if is_c3 else '#1d4ed8',
    )

c3_i = ROUND_CONDS.index('C3_3rounds')
c5_i = ROUND_CONDS.index('C5_5rounds')
c2_i = ROUND_CONDS.index('C2_1round')
delta_c35_judge = judge_means[c5_i] - judge_means[c3_i]
delta_c35_auto = auto_means[c5_i] - auto_means[c3_i]
delta_c35_t = elapsed_means_min[c5_i] - elapsed_means_min[c3_i]
delta_c23_judge = judge_means[c3_i] - judge_means[c2_i]
delta_c23_t = elapsed_means_min[c3_i] - elapsed_means_min[c2_i]
ax_eff.annotate(
    f'C5: AutoScore +{delta_c35_auto:.3f},\nJudge {delta_c35_judge:.3f}, +{delta_c35_t:.1f} min vs C3',
    xy=(elapsed_means_min[c5_i], judge_means[c5_i]),
    xytext=(elapsed_means_min[c3_i] + 0.5, judge_means[c3_i] - 0.08),
    arrowprops=dict(arrowstyle='->', color='#64748b', lw=1.0),
    fontsize=8.5,
    color='#475569',
    bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#cbd5e1', alpha=0.92),
)
ax_eff.annotate(
    f'C3: peak LLM Judge\n(+{delta_c23_judge:.3f} vs C2)',
    xy=(elapsed_means_min[c3_i], judge_means[c3_i]),
    xytext=(elapsed_means_min[c2_i] + 0.4, judge_means[c2_i] + 0.035),
    arrowprops=dict(arrowstyle='->', color='#92400e', lw=1.1),
    fontsize=8.5,
    color='#92400e',
    bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#fde68a', alpha=0.92),
)
ax_eff.set_xlabel('Elapsed time per run (min)')
ax_eff.set_ylabel('LLM Judge overall')
ax_eff.set_title('Runtime-aware elbow selection\nC3 balances quality and time')
ax_eff.set_xlim(0, max(elapsed_means_min) + 2.0)
ax_eff.set_ylim(0.60, 0.83)
ax_eff.grid(axis='both', linestyle=':', alpha=0.35)
handles, labels = ax_eff.get_legend_handles_labels()
dedup = dict(zip(labels, handles))
ax_eff.legend(dedup.values(), dedup.keys(), loc='lower right', framealpha=0.95, fontsize=8)

fig.text(
    0.5, 0.02,
    'C4(eDISC) excluded to isolate discussion-stage changes. C3 is selected as the runtime-aware elbow: peak LLM Judge, high AutoScore, less cost than C5.',
    ha='center', va='bottom', fontsize=9, color='dimgray',
)
plt.tight_layout(rect=(0, 0.06, 1, 1))
plt.savefig(f'{OUT}/figures/fig13_gemma26_round_trajectory.png', dpi=120, bbox_inches='tight')
plt.close()
print('✅ fig13_gemma26_round_trajectory.png')

# ──────────────────────────────────────────────────
# Fig14: Slide-ready model selection view — why Gemma-4-26B
# ──────────────────────────────────────────────────
SELECT_CONDS = ['C2_1round', 'C3_3rounds', 'C5_5rounds']
SHORT_MODEL = {
    'gemma': 'Gemma-4B',
    'qwen': 'Qwen3-14B',
    'gemma26': 'Gemma-4-26B',
    'gemini': 'Gemini API',
}

selection = {}
for bb in BACKBONES:
    llm_vals, auto_vals, gpu_min_vals, external_tokens = [], [], [], []
    for c in SELECT_CONDS:
        llm = A[bb][c]['overall'][0]
        auto = AUTO[bb][c]['overall'][0]
        gpu_min = T_AGG[bb][c][0] / 60.0 if T_AGG[bb][c][0] is not None else None
        api_tok = API_AGG[bb][c]['tokens'][0]
        if llm is not None:
            llm_vals.append(llm)
        if auto is not None:
            auto_vals.append(auto)
        if gpu_min is not None:
            gpu_min_vals.append(gpu_min)
        external_tokens.append(api_tok if REMOTE_API[bb] and api_tok is not None else 0.0)
    llm_mean = mean(llm_vals) if llm_vals else 0.0
    auto_mean = mean(auto_vals) if auto_vals else 0.0
    gpu_mean = mean(gpu_min_vals) if gpu_min_vals else 0.0
    selection[bb] = {
        'llm': llm_mean,
        'auto': auto_mean,
        'gpu_min': gpu_mean,
        'llm_eff': llm_mean / gpu_mean if gpu_mean else 0.0,
        'auto_eff': auto_mean / gpu_mean if gpu_mean else 0.0,
        'external_tokens_k': mean(external_tokens) / 1000.0 if external_tokens else 0.0,
    }

fig, axes = plt.subplots(2, 2, figsize=(14, 8.2))
ax_q, ax_t, ax_e, ax_api = axes.flatten()
bb_x = np.arange(len(BACKBONES))
bb_labels = [SHORT_MODEL[bb] for bb in BACKBONES]
bar_colors = [COLORS[bb] for bb in BACKBONES]
bar_edges = ['#111827' if bb == 'gemma26' else '#334155' for bb in BACKBONES]
bar_widths = [2.5 if bb == 'gemma26' else 0.8 for bb in BACKBONES]

# A. Quality signals
metric_w = 0.34
llm_scores = [selection[bb]['llm'] for bb in BACKBONES]
auto_scores = [selection[bb]['auto'] for bb in BACKBONES]
ax_q.axvspan(1.55, 2.45, color='#dcfce7', alpha=0.42, zorder=0)
ax_q.bar(
    bb_x - metric_w / 2, llm_scores, metric_w,
    color='#111827', alpha=0.86,
    edgecolor=bar_edges, linewidth=bar_widths, label='LLM Judge',
)
ax_q.bar(
    bb_x + metric_w / 2, auto_scores, metric_w,
    color=bar_colors, alpha=0.86,
    edgecolor=bar_edges, linewidth=bar_widths, label='AutoScore v2',
)
for i, (llm_v, auto_v) in enumerate(zip(llm_scores, auto_scores)):
    fw = 'bold' if BACKBONES[i] == 'gemma26' else 'normal'
    ax_q.text(i - metric_w / 2, llm_v + 0.008, f'{llm_v:.3f}', ha='center', fontsize=8.5, fontweight=fw)
    ax_q.text(i + metric_w / 2, auto_v + 0.008, f'{auto_v:.3f}', ha='center', fontsize=8.5, fontweight=fw)
ax_q.set_xticks(bb_x)
ax_q.set_xticklabels(bb_labels, rotation=10)
ax_q.set_ylim(0.50, 0.88)
ax_q.set_ylabel('Mean score')
ax_q.set_title('A. Quality signals over C2/C3/C5\nLLM Judge and AutoScore v2 shown separately')
ax_q.grid(axis='y', linestyle=':', alpha=0.35)
ax_q.legend(loc='lower right', fontsize=8, framealpha=0.95)

# B. Runtime cost
gpu_mins = [selection[bb]['gpu_min'] for bb in BACKBONES]
ax_t.bar(bb_x, gpu_mins, color=bar_colors, alpha=0.82, edgecolor=bar_edges, linewidth=bar_widths)
for i, v in enumerate(gpu_mins):
    ax_t.text(i, v * 1.08, f'{v:.1f}', ha='center', fontsize=9)
ax_t.set_xticks(bb_x)
ax_t.set_xticklabels(bb_labels, rotation=10)
ax_t.set_yscale('log')
ax_t.set_ylim(min(gpu_mins) * 0.65, max(gpu_mins) * 1.6)
ax_t.set_ylabel('GPU-minutes / run (log)')
ax_t.set_title('B. Runtime cost\nlower is better')
ax_t.grid(axis='y', linestyle=':', alpha=0.35, which='both')

# C. Local efficiency: exclude API-only Gemini from the local candidate decision.
LOCAL_BBS = ['gemma', 'qwen', 'gemma26']
local_x = np.arange(len(LOCAL_BBS))
local_auto_eff = [selection[bb]['auto_eff'] for bb in LOCAL_BBS]
local_llm_eff = [selection[bb]['llm_eff'] for bb in LOCAL_BBS]
eff_w = 0.34
ax_e.axvspan(1.55, 2.45, color='#dcfce7', alpha=0.42, zorder=0)
ax_e.bar(
    local_x - eff_w / 2, local_llm_eff, eff_w,
    color='#111827',
    alpha=0.86,
    edgecolor=['#111827' if bb == 'gemma26' else '#334155' for bb in LOCAL_BBS],
    linewidth=[2.5 if bb == 'gemma26' else 0.8 for bb in LOCAL_BBS],
    label='LLM Judge / GPU-min',
)
ax_e.bar(
    local_x + eff_w / 2, local_auto_eff, eff_w,
    color=[COLORS[bb] for bb in LOCAL_BBS],
    alpha=0.86,
    edgecolor=['#111827' if bb == 'gemma26' else '#334155' for bb in LOCAL_BBS],
    linewidth=[2.5 if bb == 'gemma26' else 0.8 for bb in LOCAL_BBS],
    label='AutoScore v2 / GPU-min',
)
eff_top = max(local_auto_eff + local_llm_eff)
for i, (llm_v, auto_v) in enumerate(zip(local_llm_eff, local_auto_eff)):
    fw = 'bold' if LOCAL_BBS[i] == 'gemma26' else 'normal'
    ax_e.text(i - eff_w / 2, llm_v + eff_top * 0.035, f'{llm_v:.3f}', ha='center', fontsize=8.5, fontweight=fw)
    ax_e.text(i + eff_w / 2, auto_v + eff_top * 0.035, f'{auto_v:.3f}', ha='center', fontsize=8.5, fontweight=fw)
ax_e.set_xticks(local_x)
ax_e.set_xticklabels([SHORT_MODEL[bb] for bb in LOCAL_BBS], rotation=10)
ax_e.set_ylim(0, eff_top * 1.22)
ax_e.set_ylabel('Score / GPU-minute')
ax_e.set_title('C. Local deployment efficiency\nhigher is better; API baseline excluded')
ax_e.grid(axis='y', linestyle=':', alpha=0.35)
ax_e.legend(loc='upper left', fontsize=8, framealpha=0.95)

# D. External dependency
ext_tokens = [selection[bb]['external_tokens_k'] for bb in BACKBONES]
ax_api.bar(bb_x, ext_tokens, color=bar_colors, alpha=0.82, edgecolor=bar_edges, linewidth=bar_widths)
for i, v in enumerate(ext_tokens):
    label = 'local' if v == 0 else f'{v:.1f}k'
    ax_api.text(i, max(v, 0.4), label, ha='center', fontsize=9,
                fontweight='bold' if BACKBONES[i] == 'gemma26' else 'normal')
ax_api.set_xticks(bb_x)
ax_api.set_xticklabels(bb_labels, rotation=10)
ax_api.set_ylabel('External API tokens / run (k)')
ax_api.set_title('D. External API dependency\nlower is better for offline/privacy constraints')
ax_api.set_ylim(0, max(ext_tokens + [1.0]) * 1.25)
ax_api.grid(axis='y', linestyle=':', alpha=0.35)

fig.suptitle('Fig14. Model Selection View — Gemma-4-26B as the Best Local Backbone', fontsize=14, y=0.99)
fig.text(
    0.5, 0.015,
    'Decision conditions: C2/C3/C5 only. C4(eDISC) is excluded to avoid mixing context-ablation effects into backbone selection.',
    ha='center', fontsize=9, color='dimgray',
)
plt.tight_layout(rect=(0, 0.04, 1, 0.96))
plt.savefig(f'{OUT}/figures/fig14_gemma26_selection_view.png', dpi=120, bbox_inches='tight')
plt.close()
print('✅ fig14_gemma26_selection_view.png')

# ──────────────────────────────────────────────────
# COMPARISON_REPORT.md
# ──────────────────────────────────────────────────
def cell(t):
    if t[0] is None: return 'N/A'
    return f'{t[0]:.2f}±{t[1]:.2f}'

with open(f'{OUT}/COMPARISON_REPORT.md', 'w') as f:
    f.write('# 4-Backbone Ablation Comparison Report\n\n')
    f.write('생성일: 2026-04-23\n\n')
    f.write('## 0. 비교 대상 모델\n\n')
    f.write('| 백본 라벨 | 정식 모델명 | 파라미터 |\n|---|---|---|\n')
    f.write('| **gemma**   | google/gemma-4-E4B-it                            | ~4B (effective)             |\n')
    f.write('| **qwen**    | Qwen/Qwen3-14B                                    | 14B                         |\n')
    f.write('| **gemma26** | gemma-4-26B-A4B-it Q4_K_M (GGUF)                 | 26B total / 4B active (MoE) |\n')
    f.write('| **gemini**  | gemini-3.1-flash-lite-preview                     | (비공개)                     |\n\n')
    f.write('## 1. 실험 통제 (모든 백본 동일)\n\n')
    f.write('| 항목 | 값 |\n|---|---|\n')
    f.write('| 조건 | C0~C5 6 조건 |\n')
    f.write('| 반복 | N=3 |\n')
    f.write('| PRD | sample_data/sample_prd.txt (AI 고객서비스 플랫폼) |\n')
    f.write('| 팀원 | sample_data/sample_members/ 6명 |\n')
    f.write('| Structure / Debate | LLM-Judge (Gemini 3.1 Pro Preview) |\n')
    f.write('| **Assignment** | **Rule-based (`autoscore_allocation`, eval2 §5 deterministic 공식)** |\n')
    f.write('| Overall | eval2 §4.4 active-dim re-normalized (S=0.40, A=0.35, D=0.25) |\n\n')
    f.write('> **Assignment 차원 변경 사유**: Gemma 4B의 긴 WBS에서 LLM Judge가 echo·truncation으로 빈번히 실패(N=2/17) → 평균이 무의미해짐. 이를 회피하기 위해 결정론적 규칙(`0.30×PlanningScore + 0.30×(1-Gini) + 0.20×Feasibility + 0.20×BufferAdequacy`)으로 전환. 모든 4 백본에 동일 적용해 일관성 보장.\n\n')

    f.write('## 2. 조건별 Overall 비교\n\n')
    f.write('| 조건 | Gemma (4B) | Qwen (14B) | Gemma26 (4B/MoE) | Gemini |\n|---|---|---|---|---|\n')
    for c, cl in zip(CONDS, COND_LABELS):
        f.write(f'| {cl} | {cell(A["gemma"][c]["overall"])} | {cell(A["qwen"][c]["overall"])} | {cell(A["gemma26"][c]["overall"])} | {cell(A["gemini"][c]["overall"])} |\n')

    f.write('\n## 3. 차원별 비교\n\n')
    for dim, dn in zip(DIMS, DIM_NAMES):
        f.write(f'### 3.{DIMS.index(dim)+1} {dn}\n\n')
        f.write('| 조건 | Gemma | Qwen | Gemma26 | Gemini |\n|---|---|---|---|---|\n')
        for c, cl in zip(CONDS, COND_LABELS):
            f.write(f'| {cl} | {cell(A["gemma"][c][dim])} | {cell(A["qwen"][c][dim])} | {cell(A["gemma26"][c][dim])} | {cell(A["gemini"][c][dim])} |\n')
        f.write('\n')

    f.write('## 4. 핵심 발견\n\n')
    g_o  = [A['gemma'][c]['overall'][0] or 0 for c in CONDS]
    q_o  = [A['qwen'][c]['overall'][0] or 0 for c in CONDS]
    g26_o = [A['gemma26'][c]['overall'][0] or 0 for c in CONDS]
    n_o  = [A['gemini'][c]['overall'][0] or 0 for c in CONDS]
    f.write('**4.1 백본 순위 — C3 (full system) 기준**\n\n')
    rank = sorted([('Gemini', n_o[3]), ('Gemma26', g26_o[3]), ('Qwen', q_o[3]), ('Gemma', g_o[3])], key=lambda x: -x[1])
    for i, (bb, sc) in enumerate(rank, 1):
        f.write(f'  {i}. {bb}: {sc:.2f}\n')
    f.write('\n')
    gap_g26_n = mean([n_o[i] - g26_o[i] for i in range(6)])
    gap_g26_q = mean([g26_o[i] - q_o[i] for i in range(6)])
    f.write(f'- 평균 격차: Gemini − Gemma26 ≈ {gap_g26_n:+.2f}, Gemma26 − Qwen ≈ {gap_g26_q:+.2f}\n')
    f.write('- **Gemma26(4B active MoE)이 Qwen(14B dense)보다 일관 높음** — 동급 활성 파라미터에서 MoE 구조 우위 시사\n')
    f.write(f'- Gemini와 Gemma26 격차는 평균 {gap_g26_n:+.2f}로 좁음 (특히 C3 동률 근처)\n\n')

    f.write('**4.2 토론 라운드 효과 (C1 → C2 변화)**\n\n')
    f.write(f'- Gemma:   {g_o[2]-g_o[1]:+.2f} (큰 단조 상승)\n')
    f.write(f'- Qwen:    {q_o[2]-q_o[1]:+.2f} (하락 — 토론 도입 시 분산↑)\n')
    f.write(f'- Gemma26: {g26_o[2]-g26_o[1]:+.2f}\n')
    f.write(f'- Gemini:  {n_o[2]-n_o[1]:+.2f} (천장 근처)\n\n')
    f.write('**4.3 eDISC 효과 (C3 → C4)**\n\n')
    f.write(f'- Gemma:   {g_o[4]-g_o[3]:+.2f}\n')
    f.write(f'- Qwen:    {q_o[4]-q_o[3]:+.2f}\n')
    f.write(f'- Gemma26: {g26_o[4]-g26_o[3]:+.2f}\n')
    f.write(f'- Gemini:  {n_o[4]-n_o[3]:+.2f}\n\n')
    f.write('**4.4 백본별 최적 조건 (Overall μ 최대)**\n\n')
    for bb in BACKBONES:
        os_ = [(c, A[bb][c]['overall'][0] or 0) for c in CONDS]
        best = max(os_, key=lambda x: x[1])
        f.write(f'- {bb}: {best[0]} = {best[1]:.2f}\n')

    f.write('\n## 5. Autoscore 비교\n\n')
    f.write('| 조건 | Gemma (4B) | Qwen (14B) | Gemma26 (4B/MoE) | Gemini |\n|---|---|---|---|---|\n')
    for c, cl in zip(CONDS, COND_LABELS):
        f.write(
            f'| {cl} | {cell(AUTO["gemma"][c]["overall"])} | {cell(AUTO["qwen"][c]["overall"])} | '
            f'{cell(AUTO["gemma26"][c]["overall"])} | {cell(AUTO["gemini"][c]["overall"])} |\n'
        )
    f.write('\n### 5.1 Autoscore 차원별 비교\n\n')
    for dim, dn in zip(AUTO_DIMS, AUTO_DIM_NAMES):
        f.write(f'#### {dn}\n\n')
        f.write('| 조건 | Gemma | Qwen | Gemma26 | Gemini |\n|---|---|---|---|---|\n')
        for c, cl in zip(CONDS, COND_LABELS):
            f.write(
                f'| {cl} | {cell(AUTO["gemma"][c][dim])} | {cell(AUTO["qwen"][c][dim])} | '
                f'{cell(AUTO["gemma26"][c][dim])} | {cell(AUTO["gemini"][c][dim])} |\n'
            )
        f.write('\n')

    f.write('\n## 6. Judge Reliability — 백본별 N/A 비율 (rev.3 패치 결과)\n\n')
    f.write('패치된 Judge(rev.3, max_tokens=1500 + echo 감지)로 Gemma snapshot 17개 재심사 결과:\n\n')
    f.write('| 백본 | Source CSV | S valid | A valid | D valid |\n|---|---|---|---|---|\n')
    src_map = {'gemma': gemma, 'qwen': qwen, 'gemma26': gemma26, 'gemini': gemini}
    src_label = {
        'gemma':   'rejudge_v3 (patched)',
        'qwen':    'finegrain (original)',
        'gemma26': 'finegrain (original; C5 r2/r3 re-judged with gemini-3.1-pro)',
        'gemini':  'finegrain (original)',
    }
    for bb in BACKBONES:
        rs = src_map[bb]
        n = len(rs)
        sv = sum(1 for r in rs if r['structure']>=0)
        av = sum(1 for r in rs if r['assignment']>=0)
        dv = sum(1 for r in rs if r['debate']>=0)
        f.write(f'| {bb} | {src_label[bb]} | {sv}/{n} | {av}/{n} | {dv}/{n} |\n')
    f.write('\n')
    f.write('**핵심 발견**: Gemma의 원본 Assignment "0.0" 점수 다수는 **실제 Judge 평가가 아니라 파서 fallback 아티팩트**였음. 패치된 Judge가 정직하게 "Judge JSON 형성 실패 → N/A"로 분류.\n')
    f.write('17개 Gemma 스냅샷 중 **A 차원이 실제로 평가된 것은 3건뿐** (C3 r1=0.00, C4 r1=0.00, C5 r2=1.00).\n')
    f.write('Gemma A 평균은 N=1~2 통계로만 산출되며, 본 비교의 Gemma A 컬럼은 신뢰도 낮음 — 절대값보다는 "Gemma WBS는 Judge가 안정적으로 채점하기 어렵다" 자체를 백본 품질의 신호로 해석할 것.\n\n')
    f.write('Qwen·Gemini는 사용자 요청에 따라 rejudge하지 않고 원본 사용 → 같은 echo 감지 패치가 적용되지 않았으므로 절대값 비교 시 주의 (특히 Qwen A=1.0의 일부도 같은 fallback 아티팩트일 가능성).\n\n')
    f.write('## 7. Special Note — Gemma 4B vs Gemma26 Assignment 격차\n\n')
    f.write('Gemma C1 Overall = **0.21**, Gemma26 C1 ≈ **{:.2f}**. 같은 Gemma 패밀리·동일 PRD에서 큰 격차.\n\n'.format(A["gemma26"]["C1_with_assign"]["overall"][0] or 0))
    f.write('**원인**: Gemma 4B는 직군 라벨("Data Engineer", "Backend Developer")을 생성하나 실제 팀원 ID(MBR-XXXX)와 매핑하지 못함 → Judge Skill Fit 0점. Gemma26은 같은 PRD에서 멤버 ID까지 정확히 매핑.\n')
    f.write('→ MoE 활성 파라미터는 같은 4B지만, **26B total knowledge로 도메인 매칭 능력이 질적으로 다름** 시사.\n\n')
    # Timing comparison
    f.write('## 8. 소요 시간 비교 (per run, μ ± σ in seconds)\n\n')
    f.write('| 조건 | Gemma (4B) | Qwen (14B) | Gemma26 (4B/MoE) | Gemini |\n|---|---|---|---|---|\n')
    for c, cl in zip(CONDS, COND_LABELS):
        cells = []
        for bb in BACKBONES:
            m, s, n = T_AGG[bb][c]
            if m is None: cells.append('N/A')
            else: cells.append(f'{m:.0f}±{s:.0f}s')
        f.write(f'| {cl} | ' + ' | '.join(cells) + ' |\n')
    f.write('\n')
    # Per-backbone total
    f.write('**조건당 평균 (전체 18 runs 가중)**:\n\n')
    for bb in BACKBONES:
        total = sum((T_AGG[bb][c][0] or 0) * T_AGG[bb][c][2] for c in CONDS)
        n_total = sum(T_AGG[bb][c][2] for c in CONDS)
        avg = total / n_total if n_total else 0
        wall = total / 60  # wall-clock minutes for entire 18 runs
        f.write(f'- **{bb}**: avg {avg:.0f}s/run, total wall-clock ≈ {wall:.0f}분 ({wall/60:.1f}h)\n')
    f.write('\n**해석**:\n')
    f.write('- 조건이 복잡할수록(C5 5R 토론) 모든 백본에서 시간 증가\n')
    f.write('- API 백본(Gemini)이 로컬 백본보다 일관되게 빠름\n')
    f.write('- Gemma26 GGUF Q4 양자화는 Qwen3-14B FP16보다 빠름 (모델 크기·양자화 영향)\n')
    f.write('- Gemma 4B는 4B 모델치곤 느린데, 응답 토큰 수가 많아서(WBS 78~88 task) 시간 비례\n\n')

    f.write('## 9. 한계 (Validity Caveats)\n\n')
    f.write('- **Self-preference bias (Gemini 만 해당)**: 생성·판정 모두 Gemini → Panickssery 2024 효과로 점수 일부 부풀림. Gemma·Qwen은 cross-vendor라 무관.\n')
    f.write('- **Judge max_output_tokens=500 cap**: reason 필드 truncation 다수, 일부 Assignment 1.0이 Judge echo 후 default cap 의심. 단, 동일 조건에서 3 백본 모두 평가했으므로 **상대 비교는 일관**.\n')
    f.write('- **N=3**: pilot 수준. 조건 간 차이가 σ 내인 경우 통계적 확정 불가.\n')
    f.write('- **단일 PRD/팀**: 다른 도메인 일반화 미검증.\n\n')
    f.write('## 10. 산출물\n\n')
    f.write('- `figures/fig1_overall.png` — 조건별 Overall 막대 (4 백본 × 6 조건 + scatter)\n')
    f.write('- `figures/fig2_dimensions.png` — Structure/Assignment/Debate 차원 분해\n')
    f.write('- `figures/fig3_trajectory.png` — 조건 진행에 따른 Overall trajectory\n')
    f.write('- `figures/fig4_radar.png` — C3 (full system) 차원 radar\n')
    f.write('- `figures/fig5_latency.png` — 조건별 소요 시간 (log scale)\n')
    f.write('- `figures/fig9_autoscore_overall.png` — 조건별 autoscore overall 막대\n')
    f.write('- `figures/fig10_autoscore_dimensions.png` — autoscore quality/allocation/orchestration 분해\n')
    f.write('- `figures/fig11_llm_auto_alignment.png` — LLM overall과 AutoScore v2 정렬성 확인 figure\n')
    f.write('- `figures/fig12_api_deployment_view.png` — API 의존성(외부 토큰/월간 외부 지출)까지 포함한 배포 관점 figure\n')
    f.write('- `figures/fig13_gemma26_round_trajectory.png` — Gemma-4-26B 단독 C0/C1/C2/C3/C5 AutoScore v2와 LLM Judge 궤적\n')
    f.write('- `figures/fig14_gemma26_selection_view.png` — Gemma-4-26B 선택 근거: AutoScore/LLM Judge, GPU-min, local efficiency, external API dependency\n')
    f.write('- `summary_4backbones.csv` — 정리된 비교용 테이블\n')

# CSV summary
with open(f'{OUT}/summary_4backbones.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow([
        'condition','backbone','N',
        'S_mean','S_std','A_mean','A_std','D_mean','D_std','Overall_mean','Overall_std',
        'AutoQuality_mean','AutoQuality_std','AutoAllocation_mean','AutoAllocation_std',
        'AutoOrchestration_mean','AutoOrchestration_std','AutoOverall_mean','AutoOverall_std'
    ])
    for c in CONDS:
        for bb in BACKBONES:
            a = A[bb][c]
            row = [c, bb, a['n']]
            for d in DIMS:
                m, s = a[d]
                row += [f'{m:.4f}' if m is not None else 'NA', f'{s:.4f}' if s is not None else 'NA']
            m, s = a['overall']
            row += [f'{m:.4f}' if m is not None else 'NA', f'{s:.4f}' if s is not None else 'NA']
            for d in AUTO_DIMS + ['overall']:
                m, s = AUTO[bb][c][d]
                row += [f'{m:.4f}' if m is not None else 'NA', f'{s:.4f}' if s is not None else 'NA']
            w.writerow(row)

print(f'✅ COMPARISON_REPORT.md')
print(f'✅ summary_3backbones.csv')
print(f'\nOutput: {OUT}/')
