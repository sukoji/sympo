"""Extract Fig11 centroid panel as a standalone wide figure."""
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1] / "comparison_4backbones"
SUMMARY = ROOT / "summary_4backbones.csv"
OUT_DIRS = [ROOT / "figures", ROOT / "figures2", ROOT / "figures_interpretation"]
for out_dir in OUT_DIRS:
    out_dir.mkdir(parents=True, exist_ok=True)

BACKBONES = ["gemma", "qwen", "gemma26", "gemini"]
MODEL_INFO = {
    "gemma": "Gemma-4B",
    "qwen": "Qwen3-14B",
    "gemma26": "Gemma-4-26B",
    "gemini": "Gemini API",
}
COLORS = {
    "gemma": "#d97706",
    "qwen": "#7c3aed",
    "gemma26": "#16a34a",
    "gemini": "#2563eb",
}
CONDS = ["C2_1round", "C3_3rounds", "C5_5rounds"]


def parse_float(value):
    if value in (None, "", "NA"):
        return None
    return float(value)


data = {bb: {} for bb in BACKBONES}
with open(SUMMARY, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        bb = row["backbone"]
        cond = row["condition"]
        if bb not in data or cond not in CONDS:
            continue
        data[bb][cond] = {
            "llm": parse_float(row["Overall_mean"]),
            "auto": parse_float(row["AutoOverall_mean"]),
        }

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
    xs = [data[bb][c]["llm"] for c in CONDS if c in data[bb] and data[bb][c]["llm"] is not None]
    ys = [data[bb][c]["auto"] for c in CONDS if c in data[bb] and data[bb][c]["auto"] is not None]
    if not xs or not ys:
        continue
    centroids[bb] = {
        "x": float(np.mean(xs)),
        "y": float(np.mean(ys)),
        "sx": float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0,
        "sy": float(np.std(ys, ddof=1)) if len(ys) > 1 else 0.0,
    }

label_offsets = {
    "gemma": (-88, -14),
    "qwen": (12, 10),
    "gemma26": (14, -24),
    "gemini": (14, 14),
}

for bb, p in centroids.items():
    is_focus = bb == "gemma26"
    ax.scatter(
        [p["x"]],
        [p["y"]],
        s=300 if is_focus else 170,
        color=COLORS[bb],
        edgecolor="#111827",
        linewidth=2.0 if is_focus else 1.0,
        zorder=5 if is_focus else 4,
    )
    if is_focus:
        ax.scatter([p["x"]], [p["y"]], s=520, facecolors="none", edgecolors="#f59e0b", linewidths=2.4, zorder=6)

    dx, dy = label_offsets.get(bb, (10, 8))
    ax.annotate(
        f"{MODEL_INFO[bb]}\nLLM={p['x']:.3f} / Auto={p['y']:.3f}",
        (p["x"], p["y"]),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=14 if is_focus else 13,
        fontweight="bold" if is_focus else "normal",
        color=COLORS[bb],
        bbox=dict(boxstyle="round,pad=0.24", fc="white", ec=COLORS[bb], alpha=0.92),
    )

ax.set_xlim(0.49, 0.76)
ax.set_ylim(0.76, 0.86)
ax.set_xlabel("LLM Judge 평균 점수")
ax.set_ylabel("AutoScore 평균 점수")
ax.set_title("토의 라운드별 백본 성능 중심점")
ax.grid(alpha=0.32, linestyle=":")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.text(
    0.5,
    -0.035,
    "각 점은 1회·3회·5회 토의 조건의 평균입니다. 금색 테두리는 Gemma-4-26B입니다.",
    ha="center",
    fontsize=12.5,
    color="#475569",
)

for out_dir in OUT_DIRS:
    path = out_dir / "fig11b_backbone_centroid_c2c3c5_wide.png"
    plt.savefig(path, dpi=180, bbox_inches="tight")
    print(f"saved {path}")
plt.close(fig)
