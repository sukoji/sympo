"""
논문용 Figure 생성 스크립트
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
실험 결과 CSV를 읽어 학술 논문 수준의 차트를 생성합니다.

사용법:
  python eval/generate_figures.py eval_results/summary_gemini_combined.csv
"""
import csv
import os
import sys
from collections import defaultdict
from typing import Dict, List

import matplotlib
matplotlib.use('Agg')  # headless
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# ── 한글 폰트 설정 ──
def _setup_korean_font():
    candidates = [
        '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
        '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    ]
    for path in candidates:
        if os.path.exists(path):
            fm.fontManager.addfont(path)
            prop = fm.FontProperties(fname=path)
            plt.rcParams['font.family'] = prop.get_name()
            plt.rcParams['axes.unicode_minus'] = False
            return
    # fallback
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False

_setup_korean_font()

# ── 스타일 설정 ──
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'figure.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

COLORS = {
    'C0_llm_only': '#94a3b8',
    'C1_with_assign': '#60a5fa',
    'C2_1round': '#34d399',
    'C3_3rounds': '#f97316',
    'C4_with_disc': '#a78bfa',
    'R0_no_rag': '#94a3b8',
    'R1_vanilla': '#60a5fa',
    'R2_hybrid': '#f97316',
    'R3_graph': '#a78bfa',
    'R4_agentic': '#f43f5e',
}

SHORT_LABELS = {
    'C0_llm_only': 'C0',
    'C1_with_assign': 'C1',
    'C2_1round': 'C2',
    'C3_3rounds': 'C3',
    'C4_with_disc': 'C4',
    'R0_no_rag': 'No RAG',
    'R1_vanilla': 'Vanilla',
    'R2_hybrid': 'Hybrid',
    'R3_graph': 'Graph',
    'R4_agentic': 'Agentic',
}


def load_csv(path: str) -> List[dict]:
    rows = []
    with open(path, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            for k, v in row.items():
                try:
                    if '.' in str(v): row[k] = float(v)
                    elif str(v).isdigit(): row[k] = int(v)
                    elif v in ('True', 'False'): row[k] = v == 'True'
                except: pass
            rows.append(row)
    return rows


def group_stats(rows, cond_filter=None):
    """조건별 평균/표준편차 계산"""
    groups = defaultdict(list)
    for r in rows:
        c = r.get('condition', '')
        if cond_filter and c not in cond_filter:
            continue
        groups[c].append(r)

    stats = {}
    for c, group in groups.items():
        s = {}
        for key in group[0].keys():
            vals = [r[key] for r in group if isinstance(r.get(key), (int, float))]
            if vals:
                s[key] = {'mean': np.mean(vals), 'std': np.std(vals)}
        stats[c] = s
    return stats


def fig1_ablation_bar(stats, out_dir):
    """Figure 1: Ablation Study — 핵심 지표 비교 (Grouped Bar)"""
    conds = [c for c in ['C0_llm_only', 'C1_with_assign', 'C2_1round', 'C3_3rounds', 'C4_with_disc'] if c in stats]
    if len(conds) < 2:
        print("  [Skip] fig1: 조건 부족")
        return

    metrics = [
        ('success_rate', 'Success Rate', 1.0),
        ('mece_score', 'MECE Score', 1.0),
        ('buffer_ratio_pct', 'Buffer Ratio (%)', 30),
        ('schedule_feasibility', 'Schedule Feasibility', 1.0),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(3.5 * len(metrics), 4))
    if len(metrics) == 1: axes = [axes]

    for ax, (key, label, scale) in zip(axes, metrics):
        means = [stats[c].get(key, {}).get('mean', 0) for c in conds]
        stds = [stats[c].get(key, {}).get('std', 0) for c in conds]
        colors = [COLORS.get(c, '#999') for c in conds]
        labels = [SHORT_LABELS.get(c, c) for c in conds]

        bars = ax.bar(labels, means, yerr=stds, color=colors, capsize=4, edgecolor='white', linewidth=0.5)
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.set_ylim(0, scale * 1.15 if max(means) > 0 else 1)

        for bar, m in zip(bars, means):
            if m > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02 * scale,
                        f'{m:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    fig.suptitle('Figure 1: Ablation Study — Generation Quality Metrics', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(out_dir, 'fig1_ablation_quality.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✅ {path}")


def fig2_ablation_assignment(stats, out_dir):
    """Figure 2: Assignment Quality Metrics"""
    conds = [c for c in ['C0_llm_only', 'C1_with_assign', 'C2_1round', 'C3_3rounds', 'C4_with_disc'] if c in stats]
    if len(conds) < 2:
        print("  [Skip] fig2: 조건 부족")
        return

    metrics = [
        ('planning_score', 'Planning Score'),
        ('workload_gini', 'Workload Gini (lower=better)'),
        ('comm_efficiency', 'Comm. Efficiency'),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(3.5 * len(metrics), 4))
    if len(metrics) == 1: axes = [axes]

    for ax, (key, label) in zip(axes, metrics):
        means = [stats[c].get(key, {}).get('mean', 0) for c in conds]
        stds = [stats[c].get(key, {}).get('std', 0) for c in conds]
        colors = [COLORS.get(c, '#999') for c in conds]
        labels_x = [SHORT_LABELS.get(c, c) for c in conds]

        bars = ax.bar(labels_x, means, yerr=stds, color=colors, capsize=4, edgecolor='white', linewidth=0.5)
        ax.set_title(label, fontsize=11, fontweight='bold')
        for bar, m in zip(bars, means):
            if m > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f'{m:.3f}', ha='center', va='bottom', fontsize=8)

    fig.suptitle('Figure 2: Ablation Study — Assignment Quality', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(out_dir, 'fig2_ablation_assignment.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✅ {path}")


def fig3_cost_efficiency(stats, out_dir):
    """Figure 3: Cost-Quality Trade-off (Scatter)"""
    conds = [c for c in ['C0_llm_only', 'C1_with_assign', 'C2_1round', 'C3_3rounds', 'C4_with_disc'] if c in stats]
    if len(conds) < 2:
        print("  [Skip] fig3: 조건 부족")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    for c in conds:
        tokens = stats[c].get('est_tokens', {}).get('mean', 0)
        feas = stats[c].get('schedule_feasibility', {}).get('mean', 0)
        buf = stats[c].get('buffer_ratio_pct', {}).get('mean', 0)
        label = SHORT_LABELS.get(c, c)
        color = COLORS.get(c, '#999')
        size = max(50, buf * 15)  # 버퍼 비율로 크기 조절
        ax.scatter(tokens, feas, s=size, c=color, label=f'{label} (buf={buf:.0f}%)',
                   edgecolors='white', linewidth=1.5, zorder=5)
        ax.annotate(label, (tokens, feas), textcoords="offset points", xytext=(8, 8),
                    fontsize=10, fontweight='bold', color=color)

    ax.set_xlabel('Estimated Tokens', fontsize=11)
    ax.set_ylabel('Schedule Feasibility', fontsize=11)
    ax.set_title('Figure 3: Cost-Quality Trade-off\n(bubble size = buffer ratio)', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    path = os.path.join(out_dir, 'fig3_cost_quality.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✅ {path}")


def fig4_rag_comparison(stats, out_dir):
    """Figure 4: RAG Strategy Comparison"""
    conds = [c for c in ['R0_no_rag', 'R1_vanilla', 'R2_hybrid', 'R3_graph', 'R4_agentic'] if c in stats]
    if len(conds) < 2:
        print("  [Skip] fig4: RAG 조건 부족")
        return

    metrics = [
        ('faithfulness', 'Faithfulness'),
        ('success_rate', 'Success Rate'),
        ('buffer_ratio_pct', 'Buffer Ratio (%)'),
        ('mece_score', 'MECE Score'),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(3.5 * len(metrics), 4))
    if len(metrics) == 1: axes = [axes]

    for ax, (key, label) in zip(axes, metrics):
        means = [stats[c].get(key, {}).get('mean', 0) for c in conds]
        stds = [stats[c].get(key, {}).get('std', 0) for c in conds]
        colors = [COLORS.get(c, '#999') for c in conds]
        labels_x = [SHORT_LABELS.get(c, c) for c in conds]

        bars = ax.bar(labels_x, means, yerr=stds, color=colors, capsize=4, edgecolor='white', linewidth=0.5)
        ax.set_title(label, fontsize=11, fontweight='bold')
        for bar, m in zip(bars, means):
            if m > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f'{m:.2f}', ha='center', va='bottom', fontsize=8)

    fig.suptitle('Figure 4: RAG Strategy Comparison', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(out_dir, 'fig4_rag_comparison.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✅ {path}")


def fig5_radar(stats, out_dir):
    """Figure 5: Radar Chart — 조건별 종합 비교"""
    conds = [c for c in ['C0_llm_only', 'C1_with_assign', 'C3_3rounds', 'C4_with_disc'] if c in stats]
    if len(conds) < 2:
        print("  [Skip] fig5: 조건 부족")
        return

    axes_def = [
        ('success_rate', 'SR'),
        ('mece_score', 'MECE'),
        ('granularity_fitness', 'Granularity'),
        ('schedule_feasibility', 'Feasibility'),
        ('comm_efficiency', 'Efficiency'),
    ]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2 * np.pi, len(axes_def), endpoint=False).tolist()
    angles += angles[:1]

    for c in conds:
        values = [stats[c].get(key, {}).get('mean', 0) for key, _ in axes_def]
        values += values[:1]
        color = COLORS.get(c, '#999')
        label = SHORT_LABELS.get(c, c)
        ax.plot(angles, values, 'o-', color=color, label=label, linewidth=2, markersize=5)
        ax.fill(angles, values, color=color, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([label for _, label in axes_def], fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_title('Figure 5: Multi-dimensional Quality Radar', fontsize=13, fontweight='bold', y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    plt.tight_layout()
    path = os.path.join(out_dir, 'fig5_radar.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✅ {path}")


def main(csv_path: str):
    rows = load_csv(csv_path)
    out_dir = os.path.join(os.path.dirname(csv_path), 'figures')
    os.makedirs(out_dir, exist_ok=True)

    stats = group_stats(rows)
    print(f"\n📊 Figure 생성 ({len(rows)}행, {len(stats)}개 조건)")
    print(f"   조건: {list(stats.keys())}")

    fig1_ablation_bar(stats, out_dir)
    fig2_ablation_assignment(stats, out_dir)
    fig3_cost_efficiency(stats, out_dir)
    fig4_rag_comparison(stats, out_dir)
    fig5_radar(stats, out_dir)

    print(f"\n✅ 모든 Figure 저장: {out_dir}/")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        result_dir = os.path.join(os.path.dirname(__file__), '..', 'eval_results')
        csvs = sorted([f for f in os.listdir(result_dir) if f == 'summary_gemini_combined.csv'])
        if csvs:
            main(os.path.join(result_dir, csvs[0]))
        else:
            print("사용법: python eval/generate_figures.py <csv_path>")
    else:
        main(sys.argv[1])
