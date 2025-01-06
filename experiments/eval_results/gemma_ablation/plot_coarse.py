"""Gemma Ablation 결과 시각화 — cleaned CSV 기반."""
import csv
import sys
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

mpl.rcParams["font.family"] = ["NanumGothic", "DejaVu Sans"]
mpl.rcParams["axes.unicode_minus"] = False
EXP = Path(__file__).resolve().parent
BASE = EXP.parents[0]
sys.path.insert(0, str(BASE))
from autoscore_recompute import enrich_rows_with_autoscore

CSV = EXP / "summary_coarse_cleaned.csv"
OUT = EXP / "figures_coarse"
import os; os.makedirs(OUT, exist_ok=True)

rows = enrich_rows_with_autoscore(list(csv.DictReader(open(CSV))))
COND_ORDER = ["C0_llm_only", "C1_with_assign", "C2_1round", "C3_3rounds", "C4_with_disc", "C5_5rounds"]
COND_LABEL = {
    "C0_llm_only": "C0\nLLM only",
    "C1_with_assign": "C1\n+assign",
    "C2_1round": "C2\n+1R debate",
    "C3_3rounds": "C3\n+3R debate",
    "C4_with_disc": "C4\n+eDISC",
    "C5_5rounds": "C5\n+5R debate",
}

def parse(v):
    if v in ("", None): return None
    try:
        f = float(v)
        return f if f >= 0 else None
    except: return None

by_cond = defaultdict(list)
for r in rows:
    if int(r.get("total_tasks", 0) or 0) == 0:  # 실패 제외
        continue
    by_cond[r["condition"]].append(r)


# ─── Fig 1: Overall Judge Score by Condition (bar + run dots) ───
fig, ax = plt.subplots(figsize=(9, 5.2))
xs, means, stds = [], [], []
scatter_x, scatter_y = [], []
for i, cond in enumerate(COND_ORDER):
    vals = [parse(r["judge_overall"]) for r in by_cond[cond]]
    vals = [v for v in vals if v is not None]
    if not vals: continue
    xs.append(i)
    means.append(np.mean(vals))
    stds.append(np.std(vals, ddof=0))
    scatter_x += [i]*len(vals); scatter_y += vals

bars = ax.bar(xs, means, yerr=stds, capsize=6, color="#6366F1", alpha=0.75,
              edgecolor="#3730A3", linewidth=1.5, error_kw=dict(ecolor="#1E1B4B", lw=1.4))
ax.scatter(scatter_x, scatter_y, c="#F59E0B", s=40, zorder=3, edgecolor="white", lw=1)
for b, m in zip(bars, means):
    ax.text(b.get_x()+b.get_width()/2, m+0.015, f"{m:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_xticks(xs)
ax.set_xticklabels([COND_LABEL[COND_ORDER[i]] for i in xs])
ax.set_ylabel("LLM Judge Overall score (0-1)", fontsize=11)
ax.set_title("Gemma-4-E4B — 조건별 Overall Judge 점수 (N=3, 오렌지 점=개별 run)", fontsize=12, pad=12)
ax.set_ylim(0, 1)
ax.grid(axis="y", alpha=0.3, ls="--")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUT}/fig1_overall_by_condition.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ {OUT}/fig1_overall_by_condition.png")


# ─── Fig 2: 차원별 점수 (grouped bar, N/A는 hatch 패턴으로 명시) ───
fig, ax = plt.subplots(figsize=(10, 5.5))
DIMS = [("judge_structure", "Structure", "#EF4444"),
        ("judge_assignment", "Assignment", "#10B981"),
        ("judge_debate", "Debate", "#3B82F6")]
n_cond = len(COND_ORDER)
width = 0.25
x = np.arange(n_cond)

label_added = {lbl: False for _, lbl, _ in DIMS}

for i, (key, label, color) in enumerate(DIMS):
    for j, cond in enumerate(COND_ORDER):
        vals = [parse(r[key]) for r in by_cond[cond]]
        vals = [v for v in vals if v is not None]
        bx = x[j] + (i-1)*width
        if not vals: continue
        m = np.mean(vals); s = np.std(vals)
        legend_label = label if not label_added[label] else None
        # 0.0인 경우 (파이프라인 단계 부재로 논리적으로 0) — hatch 패턴 + 연한 색으로 실제 측정과 구분
        if m < 0.01:
            ax.bar(bx, 0.03, width, color=color, alpha=0.25,
                   edgecolor=color, hatch="////", linewidth=1, label=legend_label)
            ax.text(bx, 0.05, "0\n(단계 부재)", ha="center", va="bottom",
                    fontsize=7, color=color, style="italic")
        else:
            ax.bar(bx, m, width, yerr=s, capsize=3, color=color, alpha=0.85,
                   edgecolor="white", label=legend_label,
                   error_kw=dict(ecolor="#333", lw=1))
            ax.text(bx, m+0.025, f"{m:.2f}", ha="center", fontsize=8)
        if legend_label: label_added[label] = True

ax.set_xticks(x)
ax.set_xticklabels([COND_LABEL[c] for c in COND_ORDER])
ax.set_ylabel("Judge dimension score (0-1)", fontsize=11)
ax.set_title("Gemma-4-E4B — 조건별 Judge 차원 점수 (사선 패턴=해당 단계 부재 → 0)",
             fontsize=12, pad=12)
ax.set_ylim(0, 1.1)

# 중복 라벨 제거 + 범례 정리
handles, labels = ax.get_legend_handles_labels()
seen = {}
for h, l in zip(handles, labels):
    if l not in seen: seen[l] = h
ax.legend(seen.values(), seen.keys(), loc="upper right", frameon=False, fontsize=9)

ax.grid(axis="y", alpha=0.3, ls="--")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUT}/fig2_dimension_breakdown.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ {OUT}/fig2_dimension_breakdown.png")


# ─── Fig 3: Judge Overall vs Rule-based Autoscore ───
fig, ax = plt.subplots(figsize=(9, 5.2))
colors_cond = {"C0_llm_only":"#9CA3AF", "C1_with_assign":"#F59E0B", "C2_1round":"#10B981",
               "C3_3rounds":"#3B82F6", "C4_with_disc":"#8B5CF6", "C5_5rounds":"#EF4444"}
for cond in COND_ORDER:
    js = [parse(r["judge_overall"]) for r in by_cond[cond]]
    aus = [parse(r["autoscore_final"]) for r in by_cond[cond]]
    pairs = [(j, a) for j, a in zip(js, aus) if j is not None and a is not None]
    if pairs:
        jx, ay = zip(*pairs)
        ax.scatter(jx, ay, c=colors_cond[cond], s=90, label=COND_LABEL[cond].replace("\n"," "),
                  edgecolor="white", lw=1.2, alpha=0.9)

ax.plot([0,1],[0,1], ls="--", color="gray", alpha=0.5, label="y=x")
ax.set_xlim(0, 1); ax.set_ylim(0.75, 0.95)
ax.set_xlabel("LLM Judge Overall (Gemini 3.1 Pro)", fontsize=11)
ax.set_ylabel("Rule-based Autoscore", fontsize=11)
ax.set_title("Gemma — Judge vs 규칙 기반 점수 비교 (둘은 다른 축을 본다)", fontsize=12, pad=12)
ax.legend(loc="lower right", fontsize=9, ncol=2, frameon=False)
ax.grid(alpha=0.3, ls="--")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUT}/fig3_judge_vs_autoscore.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ {OUT}/fig3_judge_vs_autoscore.png")


# ─── Fig 4: Runtime variability ───
fig, ax = plt.subplots(figsize=(9, 5.2))
xs, elapseds_all = [], []
for i, cond in enumerate(COND_ORDER):
    times = [float(r["elapsed_sec"]) for r in by_cond[cond]]
    for t in times:
        xs.append(i); elapseds_all.append(t/60)  # 분 단위

ax.scatter(xs, elapseds_all, s=90, c="#8B5CF6", edgecolor="white", lw=1.4, alpha=0.85, zorder=3)
for i, cond in enumerate(COND_ORDER):
    times = [float(r["elapsed_sec"])/60 for r in by_cond[cond]]
    if times:
        ax.plot([i-0.25, i+0.25], [np.mean(times)]*2, lw=3, c="#1E1B4B", alpha=0.8)

ax.set_xticks(range(len(COND_ORDER)))
ax.set_xticklabels([COND_LABEL[c] for c in COND_ORDER])
ax.set_ylabel("Run elapsed time (min)", fontsize=11)
ax.set_title("Gemma — 조건별 실행 시간 (점=개별 run, 막대=평균)", fontsize=12, pad=12)
ax.grid(axis="y", alpha=0.3, ls="--")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUT}/fig4_runtime.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ {OUT}/fig4_runtime.png")

print("\n모든 figure 저장 완료.")
