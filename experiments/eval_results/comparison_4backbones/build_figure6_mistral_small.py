"""figure6/ — 6-backbone centroid (C2/C3/C5 wide) including synthetic
Qwen3.5-35B-A3B and Mistral Small 3.2 24B Instruct.

Same visual format as `figures/fig11b_backbone_centroid_c2c3c5_wide.png`.
The Qwen synthetic entry was previously Qwen3-32B; replaced with
Qwen3.5-35B-A3B (35B total / ~3B active MoE) for a fairer same-class
comparison against Gemma-4-26B-A4B (26B / 4B active MoE).
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
OUT_DIR = HERE / "figure6"
OUT_DIR.mkdir(exist_ok=True)
MISTRAL_CSV = HERE / "mistral_small_synthetic_summary.csv"

CONDS = ["C2_1round", "C3_3rounds", "C5_5rounds"]
BACKBONES = ["gemma", "qwen", "qwen35_a3b", "mistral_small", "gemma26", "gemini"]
SYNTHETIC = {"qwen35_a3b", "mistral_small"}

MODEL_INFO = {
    "gemma":         "Gemma-4B",
    "qwen":          "Qwen3-14B",
    "qwen35_a3b":    "Qwen3.5-35B-A3B (synthetic)",
    "mistral_small": "Mistral-Small-3.2-24B (synthetic)",
    "gemma26":       "Gemma-4-26B",
    "gemini":        "Gemini API",
}
COLORS = {
    "gemma":         "#d97706",
    "qwen":          "#7c3aed",
    "qwen35_a3b":    "#be185d",
    "mistral_small": "#0891b2",
    "gemma26":       "#16a34a",
    "gemini":        "#2563eb",
}

# Qwen3.5-35B-A3B: identical to figure5 fixture.
SYNTH_QWEN35_A3B_C235 = {
    "C2_1round":  {"llm": 0.6740, "auto": 0.8470},
    "C3_3rounds": {"llm": 0.6800, "auto": 0.8390},
    "C5_5rounds": {"llm": 0.6815, "auto": 0.8455},
}
# Mistral Small 3.2 24B Instruct: dense 24B, 2025-06.
SYNTH_MISTRAL_SMALL_C235 = {
    "C2_1round":  {"llm": 0.6535, "auto": 0.8285},
    "C3_3rounds": {"llm": 0.6720, "auto": 0.8240},
    "C5_5rounds": {"llm": 0.6735, "auto": 0.8330},
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
for c, v in SYNTH_MISTRAL_SMALL_C235.items():
    data["mistral_small"][c] = v

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
    "gemma":         (-90, -16),
    "qwen":          (-100, 14),
    "qwen35_a3b":    (-150, 14),
    "mistral_small": (14, -28),
    "gemma26":       (14, -24),
    "gemini":        (14, 14),
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
    if is_focus:
        ax.scatter([p["x"]], [p["y"]], s=520, facecolors="none",
                   edgecolors="#f59e0b", linewidths=2.4, zorder=6)
    if is_synth:
        ax.scatter([p["x"]], [p["y"]], s=560, facecolors="none",
                   edgecolors="#f59e0b", linewidths=2.0, linestyle="--", zorder=6)

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
ax.set_title("토의 라운드별 백본 성능 중심점 (6 backbones, +Qwen3.5-35B-A3B, +Mistral-Small-3.2-24B)")
ax.grid(alpha=0.32, linestyle=":")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.text(
    0.5, -0.035,
    "각 점은 1회·3회·5회 토의 조건의 평균입니다. 금색 테두리는 Gemma-4-26B, "
    "금색 점선 테두리는 합성(미측정) 백본(Qwen3.5-35B-A3B, Mistral-Small-3.2-24B)입니다.",
    ha="center", fontsize=12.5, color="#475569",
)

OUT_PATH = OUT_DIR / "fig11b_backbone_centroid_c2c3c5_wide.png"
plt.savefig(OUT_PATH, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"saved {OUT_PATH}")

# ---------------------------------------------------------------------------
with open(MISTRAL_CSV, "w", newline="") as fh:
    wr = csv.writer(fh)
    wr.writerow(["backbone", "condition", "synthetic",
                 "Overall_mean_LLM", "AutoOverall_mean"])
    for c, v in SYNTH_MISTRAL_SMALL_C235.items():
        wr.writerow(["mistral_small", c, "true",
                     f"{v['llm']:.4f}", f"{v['auto']:.4f}"])
print(f"saved {MISTRAL_CSV}")
