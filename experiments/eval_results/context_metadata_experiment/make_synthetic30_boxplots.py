"""Boxplot figures from the 30-run synthetic fixture for the context metadata ablation."""
from __future__ import annotations

import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

EXP = "/home/piai/ai_course/agent_test/eval_results/context_metadata_experiment"
FIX = f"{EXP}/synthetic_30run_fixture"
OUT = f"{FIX}/figures"
os.makedirs(OUT, exist_ok=True)

CSV_PATH = f"{FIX}/summary_judged_synthetic_30.csv"
CONDS = ["M_resume", "M_disc", "M_both"]
LABELS = {
    "M_resume": "Resume only",
    "M_disc": "eDISC only",
    "M_both": "Resume + eDISC",
}
COLORS = {
    "M_resume": "#9ca3af",
    "M_disc": "#93c5fd",
    "M_both": "#2563eb",
}


def read_rows():
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if float(r["overall_median"]) >= 0]


def style_box(bp, conds):
    for patch, c in zip(bp["boxes"], conds):
        patch.set_facecolor(COLORS[c])
        patch.set_alpha(0.85)
        patch.set_edgecolor("#111827")
    for med in bp["medians"]:
        med.set_color("#111827")
        med.set_linewidth(2)
    for whisk in bp["whiskers"] + bp["caps"]:
        whisk.set_color("#111827")
    for fl in bp["fliers"]:
        fl.set(marker="o", markersize=4, alpha=0.5)


def boxplot_panel(ax, by, key, title, ylim, rng):
    x = np.arange(len(CONDS))
    data = [[float(r[key]) for r in by[c]] for c in CONDS]
    bp = ax.boxplot(data, positions=x, widths=0.55, patch_artist=True)
    style_box(bp, CONDS)
    means = [np.mean(v) for v in data]
    for i, m in enumerate(means):
        ax.text(i, ylim[1] - (ylim[1] - ylim[0]) * 0.05,
                f"μ={m:.2f}", ha="center", fontsize=11, fontweight="bold",
                color="#111827")
    ax.set_title(title, fontsize=15, fontweight="bold", pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[c] for c in CONDS], fontsize=12)
    ax.tick_params(axis="y", labelsize=11)
    ax.set_ylim(*ylim)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    rows = read_rows()
    by = defaultdict(list)
    for r in rows:
        by[r["mode"]].append(r)
    rng = np.random.default_rng(2604)

    # Fig1: Overall boxplot
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    boxplot_panel(ax, by, "overall_median",
                  "LLM Judge Overall by Metadata Condition",
                  (0.50, 0.85), rng)
    ax.set_ylabel("LLM Judge Overall", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig1_overall_box_n30.png", dpi=180)
    plt.close(fig)

    # Fig2: 3-panel dimension boxplots
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6))
    boxplot_panel(axes[0], by, "S_med", "Structure", (0.65, 0.88), rng)
    boxplot_panel(axes[1], by, "A_med", "Assignment", (0.28, 0.58), rng)
    boxplot_panel(axes[2], by, "D_med", "Debate", (0.65, 0.98), rng)
    axes[0].set_ylabel("Score", fontsize=13)
    fig.suptitle("LLM Judge Dimension Scores", fontsize=17, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig2_dimensions_box_n30.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Fig3: Assignment-focus boxplot
    fig, ax = plt.subplots(figsize=(10.0, 6.0))
    boxplot_panel(ax, by, "A_med",
                  "R&R Matching Quality: Assignment Score",
                  (0.25, 0.60), rng)
    ax.set_ylabel("Assignment Score", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig3_assignment_box_n30.png", dpi=180)
    plt.close(fig)

    # Fig4: 4-in-1 grid (overall + 3 dims) for compact reporting
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.5))
    boxplot_panel(axes[0, 0], by, "overall_median", "Overall (weighted)", (0.55, 0.80), rng)
    boxplot_panel(axes[0, 1], by, "S_med", "Structure", (0.65, 0.88), rng)
    boxplot_panel(axes[1, 0], by, "A_med", "Assignment", (0.28, 0.58), rng)
    boxplot_panel(axes[1, 1], by, "D_med", "Debate", (0.65, 0.98), rng)
    for ax in axes[:, 0]:
        ax.set_ylabel("Score", fontsize=13)
    fig.suptitle("Context Metadata Ablation — Boxplots",
                 fontsize=18, fontweight="bold", y=1.00)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_grid_box_n30.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved boxplots to {OUT}")


if __name__ == "__main__":
    main()
