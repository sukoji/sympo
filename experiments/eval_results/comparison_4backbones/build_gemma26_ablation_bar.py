"""Build a bar figure for Gemma-4-26B ablation stages only.

Output:
  eval_results/comparison_4backbones/figures_interpretation/figH_gemma26_ablation_bars.png
"""
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1] / "comparison_4backbones"
SUMMARY = ROOT / "summary_4backbones.csv"
OUT = ROOT / "figures_interpretation"
OUT.mkdir(parents=True, exist_ok=True)

CONDS = ["C1_with_assign", "C2_1round", "C3_3rounds", "C5_5rounds"]
LABELS = ["C1\n+assign", "C2\n+1R", "C3\n+3R", "C5\n+5R"]
FOCUS = "gemma26"


def parse_float(value):
    if value in (None, "", "NA"):
        return None
    return float(value)


rows = {}
with open(SUMMARY, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row["backbone"] == FOCUS and row["condition"] in CONDS:
            rows[row["condition"]] = row

missing = [c for c in CONDS if c not in rows]
if missing:
    raise SystemExit(f"Missing Gemma-4-26B rows: {missing}")

llm = np.array([parse_float(rows[c]["Overall_mean"]) for c in CONDS], dtype=float)
llm_std = np.array([parse_float(rows[c]["Overall_std"]) or 0.0 for c in CONDS], dtype=float)
auto = np.array([parse_float(rows[c]["AutoOverall_mean"]) for c in CONDS], dtype=float)
auto_std = np.array([parse_float(rows[c]["AutoOverall_std"]) or 0.0 for c in CONDS], dtype=float)

plt.rcParams.update({
    "font.family": "NanumGothic",
    "axes.unicode_minus": False,
    "font.size": 17,
    "axes.titlesize": 22,
    "axes.labelsize": 19,
    "xtick.labelsize": 17,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "figure.dpi": 140,
})

fig, ax = plt.subplots(figsize=(14.5, 7.2))
x = np.arange(len(CONDS))
width = 0.34

ax.axvspan(1.58, 2.42, color="#fef3c7", alpha=0.78, zorder=0)
ax.axhline(llm[2], color="#111827", linestyle=":", linewidth=1.4, alpha=0.42)

llm_colors = ["#94a3b8", "#4b5563", "#111827", "#6b7280"]
auto_colors = ["#86efac", "#22c55e", "#16a34a", "#65a30d"]

bars_llm = ax.bar(
    x - width / 2,
    llm,
    width,
    yerr=llm_std,
    capsize=5,
    color=llm_colors,
    edgecolor="#111827",
    linewidth=[0.8, 0.9, 2.1, 0.9],
    label="LLM Judge",
    zorder=3,
)
bars_auto = ax.bar(
    x + width / 2,
    auto,
    width,
    yerr=auto_std,
    capsize=5,
    color=auto_colors,
    edgecolor="#14532d",
    linewidth=[0.8, 0.9, 2.1, 0.9],
    label="AutoScore",
    zorder=3,
)

bars_llm[3].set_alpha(0.78)
bars_auto[3].set_alpha(0.78)

for bars, vals in [(bars_llm, llm), (bars_auto, auto)]:
    for idx, (bar, val) in enumerate(zip(bars, vals)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.018,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold" if idx == 2 else "normal",
            color="#111827",
        )

ax.annotate(
    f"C5: Auto +{auto[3] - auto[2]:.3f}, LLM {llm[3] - llm[2]:+.3f}",
    xy=(3, max(llm[3], auto[3])),
    xytext=(2.64, 0.55),
    arrowprops=dict(arrowstyle="->", color="#64748b", lw=1.2),
    fontsize=15,
    color="#475569",
    bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#cbd5e1", alpha=0.96),
)

ax.set_title("Gemma-4-26B 단계별 Ablation 결과")
ax.set_ylabel("Mean score")
ax.set_xticks(x)
ax.set_xticklabels(LABELS)
ax.set_ylim(0.36, 0.96)
ax.grid(axis="y", linestyle=":", alpha=0.36, zorder=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="upper left", framealpha=0.96, ncols=2)

out = OUT / "figH_gemma26_ablation_bars.png"
plt.savefig(out, dpi=170, bbox_inches="tight")
plt.close(fig)
print(f"saved {out}")
