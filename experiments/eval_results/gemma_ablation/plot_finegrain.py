"""Fine-grained 재심사 결과 시각화."""
import csv, os
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

mpl.rcParams["font.family"] = ["NanumGothic", "DejaVu Sans"]
mpl.rcParams["axes.unicode_minus"] = False

CSV = "eval_results/gemma_ablation/summary_finegrain.csv"
OUT = "eval_results/gemma_ablation/figures_finegrain"
os.makedirs(OUT, exist_ok=True)

rows = list(csv.DictReader(open(CSV)))
COND = ["C0_llm_only","C1_with_assign","C2_1round","C3_3rounds","C4_with_disc","C5_5rounds"]
LABEL = {"C0_llm_only":"C0\nLLM only","C1_with_assign":"C1\n+assign",
         "C2_1round":"C2\n+1R debate","C3_3rounds":"C3\n+3R debate",
         "C4_with_disc":"C4\n+eDISC","C5_5rounds":"C5\n+5R debate"}

by_cond = defaultdict(list)
for r in rows:
    by_cond[r['condition']].append(r)


# ─── Fig 1: Overall(full) ───
fig, ax = plt.subplots(figsize=(9.5, 5.2))
xs, means, stds, scatter_x, scatter_y = [], [], [], [], []
for i, c in enumerate(COND):
    vals = [float(r['overall_full']) for r in by_cond[c]]
    if not vals: continue
    xs.append(i); means.append(np.mean(vals)); stds.append(np.std(vals, ddof=0))
    scatter_x += [i]*len(vals); scatter_y += vals

bars = ax.bar(xs, means, yerr=stds, capsize=6, color="#6366F1", alpha=0.75,
              edgecolor="#3730A3", linewidth=1.5, error_kw=dict(ecolor="#1E1B4B", lw=1.4))
ax.scatter(scatter_x, scatter_y, c="#F59E0B", s=50, zorder=3, edgecolor="white", lw=1.2)
for b, m in zip(bars, means):
    ax.text(b.get_x()+b.get_width()/2, m+0.02, f"{m:.3f}", ha="center", va="bottom",
            fontsize=10, fontweight="bold")
ax.set_xticks(xs); ax.set_xticklabels([LABEL[COND[i]] for i in xs])
ax.set_ylabel("LLM Judge Overall score (0-1)", fontsize=11)
ax.set_title("Gemma-4-E4B — 조건별 Overall (fine-grained rubric, 단계 부재=0 포함)",
             fontsize=12, pad=12)
ax.set_ylim(0, 1); ax.grid(axis="y", alpha=0.3, ls="--")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout(); plt.savefig(f"{OUT}/fig1_overall.png", dpi=150, bbox_inches="tight"); plt.close()


# ─── Fig 2: 차원별 ───
fig, ax = plt.subplots(figsize=(11, 5.5))
DIMS = [("structure","Structure","#EF4444"),
        ("assignment","Assignment","#10B981"),
        ("debate","Debate","#3B82F6")]
width = 0.25; x = np.arange(len(COND))
label_added = {d[1]: False for d in DIMS}

na_label_added = False
for i, (key, label, color) in enumerate(DIMS):
    for j, c in enumerate(COND):
        raw_vals = [float(r[key]) for r in by_cond[c]]
        valid = [v for v in raw_vals if v >= 0]
        bx = x[j] + (i-1)*width
        leg = label if not label_added[label] else None

        # 모든 run이 N/A(-1) = 파이프라인 단계 부재
        if not valid:
            na_leg = "N/A (단계 부재)" if not na_label_added else None
            ax.bar(bx, 1.0, width, color="#E5E7EB", alpha=0.3,
                   edgecolor="#9CA3AF", hatch="////", linewidth=0.8, label=na_leg)
            ax.text(bx, 0.5, "N/A\n(단계 부재)", ha="center", va="center",
                    fontsize=8, color="#6B7280", fontweight="bold")
            if na_leg: na_label_added = True
        else:
            # 실제 Judge 측정값 (0.00도 정당한 측정 결과로 표기)
            m = np.mean(valid); s = np.std(valid) if len(valid) > 1 else 0
            bar_h = max(m, 0.018)  # 0일 때도 얇게 보이도록 최소 높이
            ax.bar(bx, bar_h, width, yerr=s if m > 0 or s > 0 else None, capsize=3,
                   color=color, alpha=0.85 if m > 0 else 0.5,
                   edgecolor="white", label=leg, error_kw=dict(ecolor="#333", lw=1))
            ax.text(bx, bar_h + 0.025, f"{m:.2f}",
                    ha="center", fontsize=8, fontweight="bold" if m == 0 else "normal",
                    color=color if m == 0 else "black")
            if leg: label_added[label] = True

ax.set_xticks(x); ax.set_xticklabels([LABEL[c] for c in COND])
ax.set_ylabel("Judge dimension score (0-1)"); ax.set_ylim(0, 1.15)
ax.set_title("Gemma-4-E4B — fine-grained 루브릭 차원별 점수 (오차막대=σ)", pad=12)
ax.legend(loc="upper left", fontsize=9, frameon=False)
ax.grid(axis="y", alpha=0.3, ls="--")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout(); plt.savefig(f"{OUT}/fig2_dimensions.png", dpi=150, bbox_inches="tight"); plt.close()


# ─── Fig 3: 이전 rubric vs 새 rubric Overall 비교 ───
old_overall = {"C0_llm_only":(0.12, 0.00),"C1_with_assign":(0.365, 0.049),
               "C2_1round":(0.58, 0.059),"C3_3rounds":(0.579, 0.126),
               "C4_with_disc":(0.499, 0.173),"C5_5rounds":(0.495, 0.074)}
new_overall = {}
for c in COND:
    vs = [float(r['overall_full']) for r in by_cond[c]]
    new_overall[c] = (np.mean(vs), np.std(vs))

fig, ax = plt.subplots(figsize=(10, 5.2))
x = np.arange(len(COND)); w = 0.38
old_m = [old_overall[c][0] for c in COND]; old_s = [old_overall[c][1] for c in COND]
new_m = [new_overall[c][0] for c in COND]; new_s = [new_overall[c][1] for c in COND]
ax.bar(x - w/2, old_m, w, yerr=old_s, capsize=4, color="#9CA3AF", alpha=0.8,
       label="Old rubric (coarse)", error_kw=dict(ecolor="#4B5563"))
ax.bar(x + w/2, new_m, w, yerr=new_s, capsize=4, color="#6366F1", alpha=0.85,
       label="New rubric (fine-grained)", error_kw=dict(ecolor="#312E81"))
for i, (om, nm) in enumerate(zip(old_m, new_m)):
    ax.text(i - w/2, om + 0.02, f"{om:.2f}", ha="center", fontsize=8)
    ax.text(i + w/2, nm + 0.02, f"{nm:.2f}", ha="center", fontsize=8, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels([LABEL[c] for c in COND])
ax.set_ylabel("Judge Overall score"); ax.set_ylim(0, 0.85)
ax.set_title("루브릭 재설계 전/후 Overall 비교", pad=12)
ax.legend(loc="upper right", frameon=False)
ax.grid(axis="y", alpha=0.3, ls="--")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout(); plt.savefig(f"{OUT}/fig3_rubric_compare.png", dpi=150, bbox_inches="tight"); plt.close()

print(f"✅ figures 저장: {OUT}/")
for f in sorted(os.listdir(OUT)):
    print(f"  - {f}")
