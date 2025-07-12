"""Build a professional Gantt chart PNG for the README Outputs slide.

Data is taken from the WBS task cards on the same slide (6 members, 22 tasks).
The old embedded chart was a single diagonal staircase (no parallelism); this
renders a real phase-grouped plan: owners work in parallel, dependencies flow
finish-to-start, weeks on the axis.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets" / "figures" / "gantt_ko.png"

mpl.rcParams.update({
    "font.family": "Malgun Gothic",
    "axes.unicode_minus": False,
})

# member -> color (matches the WBS task-card accent colors)
OWNER = {
    "이동헌": "#4f46e5",
    "장선열": "#0d9488",
    "이주성": "#f97316",
    "박선민": "#16a34a",
    "전석호": "#f59e0b",
    "윤수빈": "#db2777",
}

# (phase title, [ (owner, task, start_day, duration_days), ... ])
PHASES = [
    ("P1 · 데이터 기반", [
        ("이동헌", "POS/재고 연동 API", 0, 5),
        ("장선열", "개인정보 비식별화", 0, 6),
        ("이주성", "소매 트렌드 분석", 0, 5),
        ("윤수빈", "소비자 수요 예측", 0, 6),
    ]),
    ("P2 · 고객·판촉", [
        ("박선민", "고객 데이터 클러스터링", 6, 7),
        ("박선민", "파일럿 매장·KPI 설정", 13, 7),
        ("이주성", "세그먼트 프로모션 설계", 6, 6),
        ("윤수빈", "프로모션 콘텐츠 제작", 6, 8),
    ]),
    ("P3 · 개발·QA", [
        ("전석호", "맞춤형 알림 서비스", 13, 8),
        ("전석호", "커뮤니티·배송 추적", 21, 7),
        ("이동헌", "통합 기능 테스트(QA)", 13, 7),
        ("이동헌", "앱 성능 최적화", 20, 8),
    ]),
    ("P4 · 현장 개선", [
        ("박선민", "고객 동선 분석·도면", 21, 7),
        ("이주성", "진열 개선 가이드라인", 21, 7),
        ("이주성", "매장 진열 리디자인 실행", 28, 8),
        ("장선열", "현장 피드백 수집·분석", 21, 7),
        ("장선열", "리디자인 최적화 분석", 28, 2),
        ("장선열", "디자인-데이터 파이프라인", 30, 2),
    ]),
    ("P5 · 성과 분석", [
        ("윤수빈", "매출 증대 효과 분석", 31, 6),
        ("윤수빈", "고객 행동 변화 분석", 37, 6),
        ("전석호", "전점 확대 전략 수립", 31, 5),
        ("박선민", "최종 성과 보고서 작성", 31, 6),
    ]),
]

C_TEXT = "#1e293b"
C_MUTED = "#64748b"
C_GRID = "#e2e8f0"
PHASE_BANDS = ["#f8fafc", "#eef2ff", "#f0fdfa", "#fff7ed", "#fefce8"]

# flatten into rows (top -> bottom)
rows = []          # (owner, task, start, dur)
phase_spans = []   # (title, y_start, y_end, band_color)
y = 0
for pi, (title, tasks) in enumerate(PHASES):
    y_start = y
    for owner, task, start, dur in tasks:
        rows.append((owner, task, start, dur))
        y += 1
    phase_spans.append((title, y_start, y, PHASE_BANDS[pi % len(PHASE_BANDS)]))

n = len(rows)
total_days = max(s + d for _, _, s, d in rows)  # 43
week = 5  # business-week gridline spacing

fig, ax = plt.subplots(figsize=(12.93, 6.4))
fig.subplots_adjust(left=0.30, right=0.995, top=0.91, bottom=0.06)

ax.set_xlim(0, total_days + 1.5)
ax.set_ylim(n - 0.5, -1.4)  # inverted, extra headroom on top for week labels

# phase background bands (full width) + phase labels in a far-left vertical strip
for title, ys, ye, color in phase_spans:
    ax.axhspan(ys - 0.5, ye - 0.5, color=color, zorder=0)
    ax.annotate(
        title, xy=(0, (ys + ye) / 2 - 0.5),
        xytext=(-0.285, (ys + ye) / 2 - 0.5), textcoords=("axes fraction", "data"),
        va="center", ha="center", rotation=90, fontsize=7.6,
        fontweight="bold", color=C_TEXT,
    )
# thin divider between the phase strip and the task-label column
ax.add_line(plt.Line2D([-0.255, -0.255], [0, 1], transform=ax.transAxes,
                       clip_on=False, color=C_GRID, lw=1.0))

# week gridlines + labels along the top
for wx in range(0, total_days + 1, week):
    ax.axvline(wx, color=C_GRID, lw=1.0, zorder=1)
    ax.text(wx, -1.05, f"{wx//week + 1}주", ha="center", va="center",
            fontsize=8, color=C_MUTED)
ax.axvline(0, color="#cbd5e1", lw=1.4, zorder=1)

BAR_H = 0.56
tick_labels = []
bar_geo = []  # (y, start, end) for arrows
for i, (owner, task, start, dur) in enumerate(rows):
    color = OWNER[owner]
    box = FancyBboxPatch(
        (start, i - BAR_H / 2), dur, BAR_H,
        boxstyle="round,pad=0,rounding_size=0.28",
        linewidth=0, facecolor=color, alpha=0.92, zorder=3,
    )
    ax.add_patch(box)
    ax.text(start + dur + 0.5, i, f"{dur}d", va="center", ha="left",
            fontsize=7.2, color=C_MUTED, zorder=4)
    tick_labels.append(task)
    bar_geo.append((i, start, start + dur))

# finish-to-start dependency arrows for consecutive same-owner tasks
for idx in range(len(rows) - 1):
    o1, _, s1, d1 = rows[idx]
    o2, _, s2, d2 = rows[idx + 1]
    if o1 == o2 and s2 >= s1 + d1 - 0.01:
        y1, e1 = bar_geo[idx][0], bar_geo[idx][2]
        y2, st2 = bar_geo[idx + 1][0], bar_geo[idx + 1][1]
        arr = FancyArrowPatch(
            (e1, y1), (st2, y2),
            connectionstyle="arc3,rad=0.0",
            arrowstyle="-|>", mutation_scale=8,
            lw=1.0, color="#94a3b8", alpha=0.65, zorder=2,
        )
        ax.add_patch(arr)

ax.set_yticks(range(n))
ax.set_yticklabels(tick_labels, fontsize=8.2, color=C_TEXT)
ax.tick_params(axis="y", length=0, pad=6)
ax.set_xticks([])
for spine in ax.spines.values():
    spine.set_visible(False)

# owner legend along the top-right
handles = [plt.Line2D([0], [0], marker="s", linestyle="", markersize=8,
                      markerfacecolor=c, markeredgecolor="none", label=o)
           for o, c in OWNER.items()]
ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 1.005),
          ncol=6, frameon=False, fontsize=8, handletextpad=0.3,
          columnspacing=1.1)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=200, facecolor="white")
plt.close(fig)
print("wrote", OUT.relative_to(ROOT), f"({total_days}d, {n} tasks)")
