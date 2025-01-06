"""C4(eDISC) 제외한 figure를 figures2/에 별도 생성.
build_comparison.py의 데이터 로딩·집계 로직을 그대로 import해서 CONDS만 필터링."""
import sys, os
from pathlib import Path
BASE = Path(__file__).resolve().parents[1]
COMP_DIR = BASE / "comparison_3backbones"
sys.path.insert(0, str(COMP_DIR))

# Import the master build module (executes data loading)
import importlib.util
spec = importlib.util.spec_from_file_location("build_comparison",
    COMP_DIR / "build_comparison.py")
bc = importlib.util.module_from_spec(spec)
# Override OUT path before exec
import builtins
_orig_open = builtins.open
# Just run it then re-render figures with filtered CONDS
spec.loader.exec_module(bc)

import matplotlib.pyplot as plt
import numpy as np
from math import pi

# Filter conditions: drop C4
CONDS_F = [c for c in bc.CONDS if c != 'C4_with_disc']
COND_LABELS_F = [bc.COND_LABELS[i] for i, c in enumerate(bc.CONDS) if c != 'C4_with_disc']
BACKBONES = bc.BACKBONES
COLORS = bc.COLORS
MODEL_INFO = bc.MODEL_INFO
A = bc.A
AUTO = bc.AUTO
T_AGG = bc.T_AGG
DIMS = ['structure','assignment','debate']
DIM_NAMES = ['Structure','Assignment','Debate']
AUTO_DIMS = ['quality', 'allocation', 'orchestration']
AUTO_DIM_NAMES = ['Quality', 'Allocation', 'Orchestration']

OUT = BASE / 'comparison_4backbones'
os.makedirs(f'{OUT}/figures2', exist_ok=True)

x = np.arange(len(CONDS_F))
w = 0.21

# Fig1: Overall bar (no C4)
fig, ax = plt.subplots(figsize=(10, 5.5))
for i, bb in enumerate(BACKBONES):
    means = [A[bb][c]['overall'][0] or 0 for c in CONDS_F]
    stds  = [A[bb][c]['overall'][1] or 0 for c in CONDS_F]
    ax.bar(x + (i-1.5)*w, means, w, yerr=stds, capsize=3,
           color=COLORS[bb], alpha=0.85, label=MODEL_INFO[bb], edgecolor='black', linewidth=0.5)
    for j, c in enumerate(CONDS_F):
        for ov in A[bb][c]['_overalls']:
            ax.scatter([j + (i-1.5)*w], [ov], color='black', s=14, zorder=5, alpha=0.6)
ax.set_xticks(x); ax.set_xticklabels(COND_LABELS_F, rotation=15)
ax.set_ylabel('Overall (re-normalized)')
ax.set_title('Fig1 (no C4). Overall Score by Backbone × Condition (μ±σ)')
ax.set_ylim(0, 1.0); ax.grid(axis='y', linestyle=':', alpha=0.4)
ax.legend(loc='upper left', frameon=True, fontsize=9, title='Backbone (model · params)')
plt.tight_layout()
plt.savefig(f'{OUT}/figures2/fig1_overall.png', dpi=120); plt.close()
print('✅ fig1_overall.png')

# Fig2: 3 dim decomposition (no C4)
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
for ax, dim, dn in zip(axes, DIMS, DIM_NAMES):
    for i, bb in enumerate(BACKBONES):
        means = []; stds = []
        for c in CONDS_F:
            m, s = A[bb][c][dim]
            means.append(m if m is not None else 0)
            stds.append(s if s is not None else 0)
        ax.bar(x + (i-1.5)*w, means, w, yerr=stds, capsize=2,
               color=COLORS[bb], alpha=0.85, label=MODEL_INFO[bb], edgecolor='black', linewidth=0.4)
        for j, c in enumerate(CONDS_F):
            if A[bb][c][dim][0] is None:
                ax.bar(x[j] + (i-1.5)*w, 1.0, w, color='lightgray', alpha=0.5, hatch='///', edgecolor='gray', linewidth=0.3)
    ax.set_xticks(x); ax.set_xticklabels(COND_LABELS_F, rotation=20, fontsize=8)
    ax.set_title(dn); ax.set_ylim(0, 1.05); ax.grid(axis='y', linestyle=':', alpha=0.4)
    if dim == 'structure':
        ax.set_ylabel('Score'); ax.legend(loc='upper right', fontsize=7, title='Backbone')
plt.suptitle('Fig2 (no C4). Dimension Decomposition (gray hatch = N/A, stage absent)', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUT}/figures2/fig2_dimensions.png', dpi=120, bbox_inches='tight'); plt.close()
print('✅ fig2_dimensions.png')

# Fig3: trajectory (no C4)
fig, ax = plt.subplots(figsize=(10, 5))
for bb in BACKBONES:
    means = [A[bb][c]['overall'][0] or 0 for c in CONDS_F]
    stds  = [A[bb][c]['overall'][1] or 0 for c in CONDS_F]
    ax.errorbar(range(len(CONDS_F)), means, yerr=stds,
                marker='o', markersize=8, linewidth=2.2, capsize=4,
                color=COLORS[bb], label=MODEL_INFO[bb], alpha=0.9)
ax.set_xticks(range(len(CONDS_F))); ax.set_xticklabels(COND_LABELS_F, rotation=15)
ax.set_ylabel('Overall (re-normalized)')
ax.set_title('Fig3 (no C4). Ablation Trajectory: Component Contribution by Backbone')
ax.set_ylim(0.40, 0.80); ax.grid(axis='both', linestyle=':', alpha=0.4)
ax.legend(loc='lower right', fontsize=9, title='Backbone (model · params)')
plt.tight_layout()
plt.savefig(f'{OUT}/figures2/fig3_trajectory.png', dpi=120); plt.close()
print('✅ fig3_trajectory.png')

# Fig4: radar — C3 already (no change), but rename for consistency
fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
N = len(DIMS)
angles = [n / N * 2 * pi for n in range(N)] + [0]
for bb in BACKBONES:
    vals = [A[bb]['C3_3rounds'][d][0] or 0 for d in DIMS]
    vals += [vals[0]]
    ax.plot(angles, vals, color=COLORS[bb], linewidth=2.5, label=MODEL_INFO[bb], alpha=0.85)
    ax.fill(angles, vals, color=COLORS[bb], alpha=0.12)
ax.set_xticks(angles[:-1]); ax.set_xticklabels(DIM_NAMES, fontsize=11)
ax.set_ylim(0, 1.0); ax.set_yticks([0.2, 0.4, 0.6, 0.8])
ax.set_title('Fig4 (no C4). C3 (3R Debate, full system) — Dimension Radar', y=1.08, fontsize=12)
ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.12), fontsize=9, title='Backbone (model · params)')
plt.tight_layout()
plt.savefig(f'{OUT}/figures2/fig4_radar.png', dpi=120, bbox_inches='tight'); plt.close()
print('✅ fig4_radar.png')

# Fig5: latency (no C4)
fig, ax = plt.subplots(figsize=(10, 5.5))
for i, bb in enumerate(BACKBONES):
    means = [T_AGG[bb][c][0] / 60 if T_AGG[bb][c][0] else 0 for c in CONDS_F]
    stds  = [T_AGG[bb][c][1] / 60 if T_AGG[bb][c][1] else 0 for c in CONDS_F]
    ax.bar(x + (i-1.5)*w, means, w, yerr=stds, capsize=3,
           color=COLORS[bb], alpha=0.85, label=MODEL_INFO[bb], edgecolor='black', linewidth=0.5)
ax.set_xticks(x); ax.set_xticklabels(COND_LABELS_F, rotation=15)
ax.set_ylabel('Elapsed time per run (minutes)')
ax.set_title('Fig5 (no C4). Inference Latency by Backbone × Condition (μ±σ)')
ax.set_yscale('log')
ax.grid(axis='y', linestyle=':', alpha=0.4, which='both')
ax.legend(loc='upper left', fontsize=9, title='Backbone (model · params)')
plt.tight_layout()
plt.savefig(f'{OUT}/figures2/fig5_latency.png', dpi=120); plt.close()
print('✅ fig5_latency.png')

# Fig6: Pareto (no C4)
COND_MARKERS = {'C0_llm_only':'o', 'C1_with_assign':'s', 'C2_1round':'^',
                'C3_3rounds':'D', 'C5_5rounds':'P'}
fig, ax = plt.subplots(figsize=(11, 6.5))
all_pts = []
for bb in BACKBONES:
    xs, ys = [], []
    for c in CONDS_F:
        lat = T_AGG[bb][c][0]; ov = A[bb][c]['overall'][0]
        if lat is None or ov is None: continue
        x_min = lat / 60
        ax.scatter([x_min], [ov], color=COLORS[bb], marker=COND_MARKERS[c],
                   s=130, alpha=0.85, edgecolor='black', linewidth=0.8, zorder=5)
        ax.annotate(c.split('_')[0], (x_min, ov), xytext=(5, 5), textcoords='offset points',
                    fontsize=7, alpha=0.7, color='dimgray')
        xs.append(x_min); ys.append(ov)
        all_pts.append((x_min, ov, bb, c))
    if len(xs) > 1:
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        ax.plot([xs[i] for i in order], [ys[i] for i in order],
                color=COLORS[bb], alpha=0.35, linewidth=1.2, linestyle=':')
front = []
for p in all_pts:
    dom = False
    for q in all_pts:
        if q is p: continue
        if q[0] <= p[0] and q[1] >= p[1] and (q[0] < p[0] or q[1] > p[1]):
            dom = True; break
    if not dom: front.append(p)
front.sort()
if front:
    ax.plot([p[0] for p in front], [p[1] for p in front],
            color='red', linewidth=2.5, alpha=0.7, zorder=3)
ax.set_xscale('log'); ax.set_xlabel('Latency per run (minutes, log scale)')
ax.set_ylabel('Overall (re-normalized)')
ax.set_title('Fig6 (no C4). Performance vs Efficiency — Pareto frontier')
ax.grid(axis='both', linestyle=':', alpha=0.4, which='both')
from matplotlib.lines import Line2D
bb_h = [Line2D([0],[0], marker='s', color='w', markerfacecolor=COLORS[bb], markersize=10,
               label=MODEL_INFO[bb], markeredgecolor='black') for bb in BACKBONES]
cd_h = [Line2D([0],[0], marker=COND_MARKERS[c], color='gray', markersize=9,
               label=COND_LABELS_F[CONDS_F.index(c)], linestyle='None') for c in CONDS_F]
leg1 = ax.legend(handles=bb_h, loc='lower right', fontsize=8, title='Backbone')
ax.add_artist(leg1)
ax.legend(handles=cd_h + [Line2D([0],[0], color='red', linewidth=2, label='Pareto front')],
          loc='lower left', fontsize=8, title='Condition')
plt.tight_layout()
plt.savefig(f'{OUT}/figures2/fig6_pareto.png', dpi=120); plt.close()
print('✅ fig6_pareto.png')

# Fig7: 2-row aligned bars (no C4)
fig, (axU, axL) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
for i, bb in enumerate(BACKBONES):
    means = [A[bb][c]['overall'][0] or 0 for c in CONDS_F]
    stds  = [A[bb][c]['overall'][1] or 0 for c in CONDS_F]
    axU.bar(x + (i-1.5)*w, means, w, yerr=stds, capsize=3,
            color=COLORS[bb], alpha=0.85, label=MODEL_INFO[bb], edgecolor='black', linewidth=0.5)
axU.set_ylabel('Overall (re-normalized)', fontsize=11)
axU.set_ylim(0, 1.0); axU.grid(axis='y', linestyle=':', alpha=0.4)
axU.set_title('Performance: composite score (higher = better)', fontsize=10)
axU.legend(loc='upper left', fontsize=9, title='Backbone (model · params)', framealpha=0.95)
for i, bb in enumerate(BACKBONES):
    means = [T_AGG[bb][c][0] / 60 if T_AGG[bb][c][0] else 0 for c in CONDS_F]
    stds  = [T_AGG[bb][c][1] / 60 if T_AGG[bb][c][1] else 0 for c in CONDS_F]
    axL.bar(x + (i-1.5)*w, means, w, yerr=stds, capsize=3,
            color=COLORS[bb], alpha=0.85, edgecolor='black', linewidth=0.5)
axL.set_ylabel('Latency per run (min, log)', fontsize=11)
axL.set_yscale('log'); axL.invert_yaxis()
axL.grid(axis='y', linestyle=':', alpha=0.4, which='both')
axL.set_title('Efficiency: latency (lower = better)', fontsize=10)
axL.set_xticks(x); axL.set_xticklabels(COND_LABELS_F, rotation=15, fontsize=10)
plt.suptitle('Fig7 (no C4). Performance vs Efficiency — 4 Backbones', fontsize=13, y=0.995)
plt.tight_layout()
plt.savefig(f'{OUT}/figures2/fig7_perf_vs_eff.png', dpi=120, bbox_inches='tight'); plt.close()
print('✅ fig7_perf_vs_eff.png')

# Fig8: small multiples (no C4)
fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
for ax, bb in zip(axes.flatten(), BACKBONES):
    xs, ys = [], []
    for c in CONDS_F:
        lat = T_AGG[bb][c][0]; ov = A[bb][c]['overall'][0]
        if lat is None or ov is None: continue
        x_min = lat / 60
        ax.scatter([x_min], [ov], color=COLORS[bb], marker=COND_MARKERS[c],
                   s=200, alpha=0.85, edgecolor='black', linewidth=1, zorder=5)
        ax.annotate(c.split('_')[0], (x_min, ov), xytext=(7, 7), textcoords='offset points',
                    fontsize=10, fontweight='bold', color='black')
        xs.append(x_min); ys.append(ov)
    if len(xs) > 1:
        ax.plot(xs, ys, color=COLORS[bb], alpha=0.4, linewidth=1.5)
    ax.set_xscale('log')
    ax.set_title(MODEL_INFO[bb], fontsize=11, fontweight='bold', color=COLORS[bb])
    ax.grid(axis='both', linestyle=':', alpha=0.4, which='both')
    ax.set_ylim(0.35, 0.85)
for ax in axes[1,:]: ax.set_xlabel('Latency per run (min, log)', fontsize=10)
for ax in axes[:,0]: ax.set_ylabel('Overall (re-normalized)', fontsize=10)
plt.suptitle('Fig8 (no C4). Performance vs Efficiency — per-backbone trajectories', fontsize=13, y=0.995)
plt.tight_layout()
plt.savefig(f'{OUT}/figures2/fig8_small_multiples.png', dpi=120, bbox_inches='tight'); plt.close()
print('✅ fig8_small_multiples.png')

# Fig9: Autoscore overall (no C4)
fig, ax = plt.subplots(figsize=(10, 5.5))
for i, bb in enumerate(BACKBONES):
    means = [AUTO[bb][c]['overall'][0] or 0 for c in CONDS_F]
    stds  = [AUTO[bb][c]['overall'][1] or 0 for c in CONDS_F]
    ax.bar(x + (i-1.5)*w, means, w, yerr=stds, capsize=3,
           color=COLORS[bb], alpha=0.85, label=MODEL_INFO[bb], edgecolor='black', linewidth=0.5)
    for j, c in enumerate(CONDS_F):
        for ov in AUTO[bb][c]['_overalls']:
            ax.scatter([j + (i-1.5)*w], [ov], color='black', s=14, zorder=5, alpha=0.6)
ax.set_xticks(x); ax.set_xticklabels(COND_LABELS_F, rotation=15)
ax.set_ylabel('Overall (autoscore)')
ax.set_title('Fig9 (no C4). Autoscore Overall by Backbone × Condition (μ±σ)')
ax.set_ylim(0, 1.0); ax.grid(axis='y', linestyle=':', alpha=0.4)
ax.legend(loc='upper left', frameon=True, fontsize=9, title='Backbone (model · params)')
plt.tight_layout()
plt.savefig(f'{OUT}/figures2/fig9_autoscore_overall.png', dpi=120); plt.close()
print('✅ fig9_autoscore_overall.png')

# Fig10: Autoscore dimensions (no C4)
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
for ax, dim, dn in zip(axes, AUTO_DIMS, AUTO_DIM_NAMES):
    for i, bb in enumerate(BACKBONES):
        means = []
        stds = []
        for c in CONDS_F:
            m, s = AUTO[bb][c][dim]
            means.append(m if m is not None else 0)
            stds.append(s if s is not None else 0)
        ax.bar(x + (i-1.5)*w, means, w, yerr=stds, capsize=2,
               color=COLORS[bb], alpha=0.85, label=MODEL_INFO[bb], edgecolor='black', linewidth=0.4)
    ax.set_xticks(x); ax.set_xticklabels(COND_LABELS_F, rotation=20, fontsize=8)
    ax.set_title(dn); ax.set_ylim(0, 1.05); ax.grid(axis='y', linestyle=':', alpha=0.4)
    if dim == 'quality':
        ax.set_ylabel('Score'); ax.legend(loc='upper right', fontsize=7, title='Backbone')
plt.suptitle('Fig10 (no C4). Autoscore Dimension Decomposition', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUT}/figures2/fig10_autoscore_dimensions.png', dpi=120, bbox_inches='tight'); plt.close()
print('✅ fig10_autoscore_dimensions.png')

# Fig11: LLM-AutoScore alignment view (no C4)
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
ax, ax2 = axes
COND_MARKERS_J = {
    'C0_llm_only': 'o',
    'C1_with_assign': 's',
    'C2_1round': '^',
    'C3_3rounds': 'D',
    'C5_5rounds': 'P',
}

for bb in BACKBONES:
    for c in CONDS_F:
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
        ax.annotate(c.split('_')[0], (x0, y0), xytext=(6, 4), textcoords='offset points',
                    fontsize=8, color='dimgray')

ax.set_xlim(0.40, 0.82); ax.set_ylim(0.72, 0.93)
ax.set_xlabel('LLM Overall'); ax.set_ylabel('Autoscore Overall')
ax.set_title('Condition-wise LLM-AutoScore alignment (no C4)')
ax.grid(alpha=0.3, linestyle=':')

OPERATIVE_CONDS_F = ['C2_1round', 'C3_3rounds', 'C5_5rounds']
for bb in BACKBONES:
    xs = [A[bb][c]['overall'][0] for c in OPERATIVE_CONDS_F if A[bb][c]['overall'][0] is not None]
    ys = [AUTO[bb][c]['overall'][0] for c in OPERATIVE_CONDS_F if AUTO[bb][c]['overall'][0] is not None]
    if not xs or not ys:
        continue
    mx = np.mean(xs); my = np.mean(ys)
    sx = np.std(xs, ddof=1) if len(xs) > 1 else 0.0
    sy = np.std(ys, ddof=1) if len(ys) > 1 else 0.0
    ax2.errorbar(
        [mx], [my], xerr=[sx], yerr=[sy],
        fmt='o', ms=13 if bb == 'gemma26' else 10,
        color=COLORS[bb], ecolor=COLORS[bb], elinewidth=1.4, capsize=4,
        markeredgecolor='black', markeredgewidth=1.5 if bb == 'gemma26' else 0.9, zorder=6
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

ax2.set_xlim(0.40, 0.84); ax2.set_ylim(0.72, 0.93)
ax2.set_xlabel('LLM Overall'); ax2.set_ylabel('Autoscore Overall')
ax2.set_title('Backbone centroid over C2-C3-C5')
ax2.grid(alpha=0.3, linestyle=':')

from matplotlib.lines import Line2D
bb_h = [Line2D([0],[0], marker='o', color='w', markerfacecolor=COLORS[bb], markersize=10,
               markeredgecolor='black', label=MODEL_INFO[bb]) for bb in BACKBONES]
cd_h = [Line2D([0],[0], marker=COND_MARKERS_J[c], color='gray', markersize=8,
               label=COND_LABELS_F[CONDS_F.index(c)], linestyle='None') for c in CONDS_F]
ax.legend(handles=bb_h, loc='lower right', fontsize=8, title='Backbone', framealpha=0.95)
ax2.legend(handles=cd_h, loc='lower right', fontsize=8, title='Condition', framealpha=0.95)
plt.suptitle('Fig11 (no C4). LLM-AutoScore Alignment View', y=0.98, fontsize=13)
plt.tight_layout()
plt.savefig(f'{OUT}/figures2/fig11_llm_auto_alignment.png', dpi=120, bbox_inches='tight'); plt.close()
print('✅ fig11_llm_auto_alignment.png')

print(f'\nOutput: {OUT}/figures2/')
