"""Create LLM-Judge-focused figures for context metadata ablation."""
import csv
import os
from collections import defaultdict
from statistics import mean, stdev

import matplotlib.pyplot as plt
import numpy as np

EXP = "/home/piai/ai_course/agent_test/eval_results/context_metadata_experiment"
OUT = f"{EXP}/figures"
os.makedirs(OUT, exist_ok=True)

CSV_PATH = f"{EXP}/summary_judged_local_scalar.csv"
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
    valid = [r for r in rows if float(r["overall_median"]) >= 0]
    if len(valid) != len(rows):
        print(f"Warning: {len(rows) - len(valid)} invalid judge rows excluded")
    return valid


def agg(rows, key):
    values = [float(r[key]) for r in rows if float(r[key]) >= 0]
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    return mean(values), stdev(values)


def annotate_bars(ax, bars):
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.018,
            f"{h:.2f}",
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

    # Fig3: LLM Judge overall
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
    for i, c in enumerate(CONDS):
        vals = [float(r["overall_median"]) for r in by[c]]
        jitter = np.linspace(-0.08, 0.08, len(vals)) if vals else []
        ax.scatter(np.full(len(vals), i) + jitter, vals, color="#111827", s=34, alpha=0.75, zorder=4)
    annotate_bars(ax, bars)
    ax.set_title("LLM Judge Overall by Metadata Condition", fontsize=18, fontweight="bold", pad=14)
    ax.set_ylabel("LLM Judge Overall", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[c] for c in CONDS], fontsize=13)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig3_llm_judge_overall.png", dpi=180)
    plt.close(fig)

    # Fig4: dimension decomposition
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
    ax.set_title("LLM Judge Dimension Scores", fontsize=18, fontweight="bold", pad=14)
    ax.set_ylabel("Score", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[c] for c in CONDS], fontsize=13)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.legend(frameon=False, fontsize=12, ncol=3, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig4_llm_judge_dimensions.png", dpi=180)
    plt.close(fig)

    # Fig5: assignment focus
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
        jitter = np.linspace(-0.07, 0.07, len(run_vals)) if run_vals else []
        ax.scatter(np.full(len(run_vals), i) + jitter, run_vals, color="#0f172a", s=38, alpha=0.78, zorder=4)
    annotate_bars(ax, bars)
    ax.set_title("R&R Matching Quality: LLM Judge Assignment Score", fontsize=17, fontweight="bold", pad=14)
    ax.set_ylabel("Assignment Score", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[c] for c in CONDS], fontsize=13)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig5_llm_judge_assignment_focus.png", dpi=180)
    plt.close(fig)

    print(f"Saved figures to {OUT}")


if __name__ == "__main__":
    main()
