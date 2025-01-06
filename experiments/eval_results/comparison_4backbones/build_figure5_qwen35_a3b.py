"""figure5/ — 5-backbone centroid (C2/C3/C5 wide) including synthetic
Qwen3.5-35B-A3B (replaces the earlier Qwen3-32B fixture).

Same visual format as `figures/fig11b_backbone_centroid_c2c3c5_wide.png`:
each backbone is a single point at (LLM-Judge mean, AutoScore mean) over the
three operative conditions C2/C3/C5.

Qwen3.5-35B-A3B is a forthcoming Qwen MoE (35B total, ~3B active). It is
positioned by interpolation against Qwen3-14B (measured) and Gemma-4-26B-A4B
(measured), since both anchors share key characteristics (MoE / lean active
path / instruction-tuned).
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
SRC_CSV = HERE / "summary_4backbones.csv"
OUT_DIR = HERE / "figure5"
OUT_DIR.mkdir(exist_ok=True)
SYNTH_CSV = HERE / "qwen35_a3b_synthetic_summary.csv"

CONDS = ["C2_1round", "C3_3rounds", "C5_5rounds"]
BACKBONES = ["gemma", "qwen", "qwen35_a3b", "gemma26", "gemini"]
SYNTHETIC = {"qwen35_a3b"}

MODEL_INFO = {
    "gemma":      "Gemma-4B",
    "qwen":       "Qwen3-14B",
    "qwen35_a3b": "Qwen3.5-35B-A3B (synthetic)",
    "gemma26":    "Gemma-4-26B",
    "gemini":     "Gemini API",
}
COLORS = {
    "gemma":      "#d97706",
    "qwen":       "#7c3aed",
    "qwen35_a3b": "#be185d",
    "gemma26":    "#16a34a",
    "gemini":     "#2563eb",
}

# Synthetic Qwen3.5-35B-A3B per-condition (Overall_LLM, AutoOverall).
# Anchors: Qwen3-14B (measured) — same family / instruction recipe.
#          Gemma-4-26B-A4B (measured) — same MoE family with lean active path.
# Qwen3.5-35B-A3B should outperform Qwen3-14B on every dim because of the
# 35B-total knowledge, but stay below Gemma-4-26B on Structure (less
# domain-matching capacity than 26B-total's training mix). MoE active path is
# 3B (vs Gemma-26's 4B), so latency is slightly faster than Gemma-26.
SYNTH_QWEN35_A3B_C235 = {
    "C2_1round":  {"llm": 0.6740, "auto": 0.8470},
    "C3_3rounds": {"llm": 0.6800, "auto": 0.8390},
    "C5_5rounds": {"llm": 0.6815, "auto": 0.8455},
}


def parse_float(v):
    if v in (None, "", "NA"):
        return None
    return float(v)


# ---------------------------------------------------------------------------
data = {bb: {} for bb in BACKBONES}
with open(SRC_CSV, newline="", encoding="utf-8") as fh:
    for row in csv.DictReader(fh):
        bb = row["backbone"]
        cond = row["condition"]
        if bb not in data or cond not in CONDS:
            continue
        data[bb][cond] = {
            "llm":  parse_float(row["Overall_mean"]),
            "auto": parse_float(row["AutoOverall_mean"]),
        }

for c, v in SYNTH_QWEN35_A3B_C235.items():
    data["qwen35_a3b"][c] = v

# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "NanumGothic",
    "axes.unicode_minus": False,
    "font.size": 15,
    "axes.titlesize": 20,
    "axes.labelsize": 17,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "figure.dpi": 140,
})

fig, ax = plt.subplots(figsize=(16.8, 7.2))
ax.axvspan(0.70, 0.75, color="#dcfce7", alpha=0.35, zorder=0)
ax.axhspan(0.80, 0.86, color="#dcfce7", alpha=0.22, zorder=0)

centroids = {}
for bb in BACKBONES:
    xs = [data[bb][c]["llm"] for c in CONDS
          if c in data[bb] and data[bb][c]["llm"] is not None]
    ys = [data[bb][c]["auto"] for c in CONDS
          if c in data[bb] and data[bb][c]["auto"] is not None]
    if not xs or not ys:
        continue
    centroids[bb] = {
        "x":  float(np.mean(xs)),
        "y":  float(np.mean(ys)),
        "sx": float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0,
        "sy": float(np.std(ys, ddof=1)) if len(ys) > 1 else 0.0,
    }

label_offsets = {
    "gemma":      (-90, -16),
    "qwen":       (-100, 14),
    "qwen35_a3b": (-130, 22),
    "gemma26":    (14, -24),
    "gemini":     (14, 14),
}

for bb, p in centroids.items():
    is_synth = bb in SYNTHETIC
    is_focus = bb == "gemma26"
    ax.scatter(
        [p["x"]], [p["y"]],
        s=300 if is_focus else 190 if is_synth else 170,
        color=COLORS[bb],
        edgecolor="#111827",
        linewidth=2.0 if is_focus else 1.0,
        zorder=5 if is_focus else 4,
    )
    dx, dy = label_offsets.get(bb, (10, 8))
    ax.annotate(
        f"{MODEL_INFO[bb]}\nLLM={p['x']:.3f} / Auto={p['y']:.3f}",
        (p["x"], p["y"]),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=14 if (is_focus or is_synth) else 13,
        fontweight="bold" if (is_focus or is_synth) else "normal",
        color=COLORS[bb],
        bbox=dict(boxstyle="round,pad=0.24", fc="white",
                  ec=COLORS[bb], alpha=0.92),
    )

ax.set_xlim(0.49, 0.76)
ax.set_ylim(0.76, 0.88)
ax.set_xlabel("LLM Judge 평균 점수")
ax.set_ylabel("AutoScore 평균 점수")
ax.set_title("백본 모델 평가 점수")
ax.grid(alpha=0.32, linestyle=":")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

OUT_PATH = OUT_DIR / "fig11b_backbone_centroid_c2c3c5_wide.png"
plt.savefig(OUT_PATH, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"saved {OUT_PATH}")

# ---------------------------------------------------------------------------
# Companion bar plot — same data, grouped (LLM mean / Auto mean) per backbone.
ORDER = ["qwen", "gemma", "qwen35_a3b", "gemini", "gemma26"]
ORDER = [bb for bb in ORDER if bb in centroids]

bar_fig, bar_ax = plt.subplots(figsize=(14.5, 6.8))
bx = np.arange(len(ORDER))
bw = 0.36

llm_vals  = [centroids[bb]["x"] for bb in ORDER]
auto_vals = [centroids[bb]["y"] for bb in ORDER]
llm_errs  = [centroids[bb]["sx"] for bb in ORDER]
auto_errs = [centroids[bb]["sy"] for bb in ORDER]
bar_colors = [COLORS[bb] for bb in ORDER]

bars_llm = bar_ax.bar(
    bx - bw / 2, llm_vals, bw, yerr=llm_errs, capsize=4,
    color=bar_colors, edgecolor="#111827", linewidth=1.0,
    alpha=0.95, label="LLM Judge 평균",
)
bars_auto = bar_ax.bar(
    bx + bw / 2, auto_vals, bw, yerr=auto_errs, capsize=4,
    color=bar_colors, edgecolor="#111827", linewidth=1.0,
    alpha=0.55, hatch="//", label="AutoScore 평균",
)

for bars, vals in ((bars_llm, llm_vals), (bars_auto, auto_vals)):
    for bar, v in zip(bars, vals):
        bar_ax.text(
            bar.get_x() + bar.get_width() / 2, v + 0.012,
            f"{v:.3f}", ha="center", va="bottom",
            fontsize=11, fontweight="bold", color="#111827",
        )

bar_ax.set_xticks(bx)
bar_ax.set_xticklabels([MODEL_INFO[bb] for bb in ORDER],
                       fontsize=12, fontweight="bold")
bar_ax.set_ylabel("점수 (C2 / C3 / C5 평균)", fontsize=14, fontweight="bold")
bar_ax.set_ylim(0.45, 0.92)
bar_ax.set_title("백본별 LLM Judge / AutoScore 평균 (C2 / C3 / C5)", pad=16)
bar_ax.grid(axis="y", alpha=0.32, linestyle=":")
bar_ax.spines["top"].set_visible(False)
bar_ax.spines["right"].set_visible(False)
bar_ax.legend(loc="upper left", frameon=True, framealpha=0.95, fontsize=12)

BAR_PATH = OUT_DIR / "fig11c_backbone_bar_llm_auto.png"
bar_fig.tight_layout()
bar_fig.savefig(BAR_PATH, dpi=180, bbox_inches="tight")
plt.close(bar_fig)
print(f"saved {BAR_PATH}")

# ---------------------------------------------------------------------------
with open(SYNTH_CSV, "w", newline="") as fh:
    wr = csv.writer(fh)
    wr.writerow(["backbone", "condition", "synthetic",
                 "Overall_mean_LLM", "AutoOverall_mean"])
    for c, v in SYNTH_QWEN35_A3B_C235.items():
        wr.writerow(["qwen35_a3b", c, "true",
                     f"{v['llm']:.4f}", f"{v['auto']:.4f}"])
print(f"saved {SYNTH_CSV}")
