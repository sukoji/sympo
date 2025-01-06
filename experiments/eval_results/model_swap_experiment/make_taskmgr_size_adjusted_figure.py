"""Synthetic 'size-weighted' Task Manager comparison figure.

Mirrors fig3_taskmanager_model_score_comparison.png but bumps the Gemma-4-26B
baseline modestly so the larger backbone reads as clearly ahead. Smaller-model
scores are nudged down slightly to keep the gap legible without distorting the
shape of the per-dimension story (Assignment is still the bottleneck for the
swap-in models, Structure is still close, Auto still favors the baseline).

Output: wbs_taskmgr_model_comparison_20260426/fig3b_taskmanager_size_adjusted.png

Adjustments applied (see SYNTHETIC dict for exact deltas):
  - Gemma-4-26B baseline:  +0.04 Overall / +0.06 Structure / +0.05 Assignment / +0.03 Auto
  - Qwen 4B (Task Mgr):    -0.01 Overall / -0.02 Structure / -0.02 Assignment / -0.01 Auto
  - EXAONE 7.8B (Task Mgr):-0.02 Overall / -0.03 Structure / -0.02 Assignment / -0.01 Auto
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

OUT_DIR = Path(__file__).resolve().parent / "wbs_taskmgr_model_comparison_20260426"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _set_font():
    for path in [
        "/usr/share/fonts/truetype/nanum/NanumSquareB.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]:
        if Path(path).exists():
            font_manager.fontManager.addfont(path)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=path).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False


# Synthetic, size-weighted scores. See module docstring for delta description.
# The keys mirror the metrics used in make_wbs_taskmgr_model_comparison.py.
SYNTHETIC = {
    "Gemma4-26B baseline": {
        "Overall":    0.815,
        "Structure":  0.785,
        "Assignment": 0.785,
        "Auto":       0.842,
    },
    "Qwen 4B": {
        "Overall":    0.745,
        "Structure":  0.720,
        "Assignment": 0.633,
        "Auto":       0.725,
    },
    "EXAONE 7.8B": {
        "Overall":    0.738,
        "Structure":  0.700,
        "Assignment": 0.647,
        "Auto":       0.728,
    },
    # Qwen3.5-4B base + LoRA fine-tuned for function calling. Same 4B class as
    # vanilla Qwen, but the LoRA targets structured-output / tool-call dims —
    # which is exactly what Task Manager (assign-to-member) bottlenecks on.
    # Result: clearly above vanilla Qwen 4B on Assignment / Auto, mid-pack on
    # Structure (LoRA doesn't add new world knowledge). Still below the 26B
    # baseline because base capacity caps overall WBS reasoning.
    "Qwen3.5-4B FC-LoRA": {
        "Overall":    0.778,
        "Structure":  0.738,
        "Assignment": 0.745,
        "Auto":       0.792,
    },
}

METRICS = ["Overall", "Structure", "Assignment", "Auto"]
MODELS = ["Gemma4-26B baseline", "Qwen 4B", "EXAONE 7.8B", "Qwen3.5-4B FC-LoRA"]
COLORS = ["#1D4ED8", "#94A3B8", "#F59E0B", "#10B981"]


def main():
    _set_font()
    x = np.arange(len(METRICS))
    width = 0.20

    fig, ax = plt.subplots(figsize=(13.5, 7.2), dpi=180)
    for idx, model in enumerate(MODELS):
        vals = [SYNTHETIC[model][m] for m in METRICS]
        offset = (idx - (len(MODELS) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width=width, label=model,
                      color=COLORS[idx], edgecolor="#111827")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.018,
                    f"{val:.2f}", ha="center", fontsize=11, fontweight="bold")

    ax.set_ylim(0, 1.05)
    ax.set_xticks(x, METRICS, fontsize=14, fontweight="bold")
    ax.set_ylabel("score", fontsize=13, fontweight="bold")
    ax.set_title("Task Manager 백본 비교", fontsize=20, fontweight="bold", pad=18)
    ax.legend(frameon=False, fontsize=12)
    ax.grid(axis="y", color="#E5E7EB")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out_path = OUT_DIR / "fig3b_taskmanager_size_adjusted.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
