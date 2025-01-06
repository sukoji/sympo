"""Generate figures from the 30-run synthetic fixture for the context metadata ablation."""
from __future__ import annotations

import csv
import os
from collections import defaultdict
from statistics import mean, stdev

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
DIM_COLORS = {
    "Structure": "#94a3b8",
    "Assignment": "#2563eb",
    "Debate": "#38bdf8",
}


def read_rows():
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if float(r["overall_median"]) >= 0]


def agg(rows, key):
    values = [float(r[key]) for r in rows if float(r[key]) >= 0]
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    return mean(values), stdev(values)


def annotate_bars(ax, bars, dy=0.018, fmt="{:.2f}"):
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + dy,
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )


def main():
    rows = read_rows()
    by = defaultdict(list)
    for r in rows:
        by[r["mode"]].append(r)

    x = np.arange(len(CONDS))

    # Fig1: Overall judge with N=30 jittered points
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    means = [agg(by[c], "overall_median")[0] for c in CONDS]
    stds = [agg(by[c], "overall_median")[1] for c in CONDS]
    bars = ax.bar(
        x,
        means,
        yerr=stds,
        capsize=5,
        color=[COLORS[c] for c in CONDS],
        edgecolor="#111827",
        linewidth=0.8,
        width=0.58,
        alpha=0.92,
    )
    rng = np.random.default_rng(2604)
    for i, c in enumerate(CONDS):
        vals = [float(r["overall_median"]) for r in by[c]]
        jitter = rng.uniform(-0.14, 0.14, size=len(vals))
        ax.scatter(
            np.full(len(vals), i) + jitter,
            vals,
            color="#111827",
            s=22,
            alpha=0.55,
            zorder=4,
        )
    annotate_bars(ax, bars)
    ax.set_title("LLM Judge Overall by Metadata Condition (N=30)", fontsize=18, fontweight="bold", pad=14)
    ax.set_ylabel("LLM Judge Overall", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[c] for c in CONDS], fontsize=13)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig1_overall_n30.png", dpi=180)
    plt.close(fig)

    # Fig2: Dimension decomposition
    dims = [("S_med", "Structure"), ("A_med", "Assignment"), ("D_med", "Debate")]
    fig, ax = plt.subplots(figsize=(12.5, 6.0))
    width = 0.23
    for j, (key, name) in enumerate(dims):
        vals = [agg(by[c], key)[0] for c in CONDS]
        errs = [agg(by[c], key)[1] for c in CONDS]
        offset = (j - 1) * width
        bars = ax.bar(
            x + offset,
            vals,
            width,
            yerr=errs,
            capsize=4,
            color=DIM_COLORS[name],
            edgecolor="#111827",
            linewidth=0.6,
            alpha=0.95 if name == "Assignment" else 0.62,
            label=name,
        )
        if name == "Assignment":
            annotate_bars(ax, bars)
    ax.set_title("LLM Judge Dimension Scores (N=30)", fontsize=18, fontweight="bold", pad=14)
    ax.set_ylabel("Score", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[c] for c in CONDS], fontsize=13)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.legend(frameon=False, fontsize=12, ncol=3, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig2_dimensions_n30.png", dpi=180)
    plt.close(fig)

    # Fig3: Assignment focus (R&R matching)
    fig, ax = plt.subplots(figsize=(9.8, 5.8))
    vals = [agg(by[c], "A_med")[0] for c in CONDS]
    errs = [agg(by[c], "A_med")[1] for c in CONDS]
    bars = ax.bar(
        x,
        vals,
        yerr=errs,
        capsize=5,
        color=[COLORS[c] for c in CONDS],
        edgecolor="#111827",
        linewidth=0.8,
        width=0.56,
    )
    for i, c in enumerate(CONDS):
        run_vals = [float(r["A_med"]) for r in by[c] if float(r["A_med"]) >= 0]
        jitter = rng.uniform(-0.13, 0.13, size=len(run_vals))
        ax.scatter(
            np.full(len(run_vals), i) + jitter,
            run_vals,
            color="#0f172a",
            s=24,
            alpha=0.6,
            zorder=4,
        )
    annotate_bars(ax, bars)
    ax.set_title("R&R Matching Quality: Assignment Score (N=30)", fontsize=17, fontweight="bold", pad=14)
    ax.set_ylabel("Assignment Score", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[c] for c in CONDS], fontsize=13)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig3_assignment_focus_n30.png", dpi=180)
    plt.close(fig)

    # Fig4: Box plot of overall by condition (shows distribution shape, N=30)
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    data = [[float(r["overall_median"]) for r in by[c]] for c in CONDS]
    bp = ax.boxplot(
        data,
        positions=x,
        widths=0.55,
        patch_artist=True,
        medianprops=dict(color="#111827", linewidth=2),
        flierprops=dict(marker="o", markersize=4, alpha=0.5),
    )
    for patch, c in zip(bp["boxes"], CONDS):
        patch.set_facecolor(COLORS[c])
        patch.set_alpha(0.85)
        patch.set_edgecolor("#111827")
    for i, c in enumerate(CONDS):
        vals = data[i]
        jitter = rng.uniform(-0.13, 0.13, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, color="#111827", s=18, alpha=0.45, zorder=4)
    ax.set_title("Distribution of LLM Judge Overall (N=30 per condition)", fontsize=17, fontweight="bold", pad=14)
    ax.set_ylabel("LLM Judge Overall", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[c] for c in CONDS], fontsize=13)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylim(0.5, 0.85)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig4_distribution_n30.png", dpi=180)
    plt.close(fig)

    print(f"Saved figures to {OUT}")


if __name__ == "__main__":
    main()
