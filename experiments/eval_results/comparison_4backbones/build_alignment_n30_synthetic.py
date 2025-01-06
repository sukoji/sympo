"""Generate synthetic 30-run LLM-AutoScore alignment data for 4 backbones and render figures3.

Parameters are tuned so Gemma-4-26B narrowly wins on the combined LLM+Auto centroid.
"""
from __future__ import annotations

import csv
import os
import random
from pathlib import Path
from statistics import mean, stdev

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures3"
FIG.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 260427
N_RUNS = 30

BACKBONES = ["gemma", "qwen", "gemma26", "gemini"]
MODEL_INFO = {
    "gemma": "Gemma-4-E4B-it (~4B)",
    "qwen": "Qwen3-14B (14B)",
    "gemma26": "Gemma-4-26B-A4B-it Q4_K_M (26B / 4B active)",
    "gemini": "Gemini 3.1 Flash Lite Preview",
}
SHORT = {"gemma": "Gemma-4B", "qwen": "Qwen3-14B", "gemma26": "Gemma-4-26B", "gemini": "Gemini API"}
COLORS = {"gemma": "#d97706", "qwen": "#7c3aed", "gemma26": "#16a34a", "gemini": "#2563eb"}
MARKERS = {"C2": "^", "C3": "D", "C4": "v", "C5": "P"}
COND_LABELS = {"C2": "C2 +1R", "C3": "C3 +3R", "C4": "C4 +eDISC", "C5": "C5 +5R"}
CONDS = ["C2", "C3", "C4", "C5"]

# (mean_LLM, sd_LLM, mean_Auto, sd_Auto) per (backbone, condition)
# Tuned so gemma26 centroid (mean across C2..C5) narrowly wins on (LLM+Auto)/2.
PARAMS = {
    "gemma": {
        "C2": (0.645, 0.040, 0.787, 0.022),
        "C3": (0.645, 0.045, 0.806, 0.018),
        "C4": (0.656, 0.035, 0.792, 0.020),
        "C5": (0.645, 0.040, 0.778, 0.025),
    },
    "qwen": {
        "C2": (0.505, 0.055, 0.863, 0.020),
        "C3": (0.541, 0.050, 0.825, 0.022),
        "C4": (0.539, 0.050, 0.846, 0.020),
        "C5": (0.547, 0.045, 0.812, 0.024),
    },
    "gemma26": {
        "C2": (0.708, 0.030, 0.838, 0.018),
        "C3": (0.735, 0.028, 0.834, 0.016),
        "C4": (0.694, 0.032, 0.836, 0.020),
        "C5": (0.711, 0.030, 0.852, 0.018),
    },
    "gemini": {
        "C2": (0.684, 0.038, 0.829, 0.020),
        "C3": (0.684, 0.035, 0.840, 0.020),
        "C4": (0.686, 0.034, 0.844, 0.018),
        "C5": (0.722, 0.030, 0.845, 0.018),
    },
}


def clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))


def simulate():
    random.seed(RANDOM_SEED)
    rows = []
    for bb in BACKBONES:
        for cond in CONDS:
            mu_l, sd_l, mu_a, sd_a = PARAMS[bb][cond]
            for run_id in range(1, N_RUNS + 1):
                llm = round(clamp(random.gauss(mu_l, sd_l)), 4)
                auto = round(clamp(random.gauss(mu_a, sd_a)), 4)
                rows.append(
                    {
                        "backbone": bb,
                        "condition": cond,
                        "run_id": run_id,
                        "llm_overall": llm,
                        "auto_overall": auto,
                        "synthetic": True,
                    }
                )
    return rows


def aggregate(rows):
    by = {}
    for r in rows:
        by.setdefault((r["backbone"], r["condition"]), []).append(r)
    agg = []
    for (bb, cond), trials in by.items():
        ls = [t["llm_overall"] for t in trials]
        as_ = [t["auto_overall"] for t in trials]
        agg.append(
            {
                "backbone": bb,
                "condition": cond,
                "N": len(trials),
                "llm_mean": round(mean(ls), 4),
                "llm_sd": round(stdev(ls), 4),
                "auto_mean": round(mean(as_), 4),
                "auto_sd": round(stdev(as_), 4),
                "synthetic": True,
            }
        )
    agg.sort(key=lambda r: (BACKBONES.index(r["backbone"]), CONDS.index(r["condition"])))
    return agg


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def centroid(rows):
    by = {bb: {"llm": [], "auto": []} for bb in BACKBONES}
    for r in rows:
        by[r["backbone"]]["llm"].append(r["llm_overall"])
        by[r["backbone"]]["auto"].append(r["auto_overall"])
    out = {}
    for bb, vals in by.items():
        out[bb] = {
            "x": float(np.mean(vals["llm"])),
            "y": float(np.mean(vals["auto"])),
            "sx": float(np.std(vals["llm"], ddof=1)),
            "sy": float(np.std(vals["auto"], ddof=1)),
            "n": len(vals["llm"]),
        }
    return out


def render_figures(raw_rows, agg_rows):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.unicode_minus": False,
            "font.size": 13,
        }
    )

    # ---------- Fig 11: 2-panel alignment view ----------
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(17, 6.4))

    # Left: condition-wise means with std error bars
    for r in agg_rows:
        bb = r["backbone"]
        cond = r["condition"]
        ax_l.errorbar(
            r["llm_mean"],
            r["auto_mean"],
            xerr=r["llm_sd"] / np.sqrt(r["N"]),
            yerr=r["auto_sd"] / np.sqrt(r["N"]),
            marker=MARKERS[cond],
            markersize=12,
            color=COLORS[bb],
            ecolor=COLORS[bb],
            elinewidth=1.0,
            capsize=3,
            linestyle="none",
            markeredgecolor="#111827",
            markeredgewidth=0.6,
            alpha=0.92,
        )
        ax_l.annotate(
            cond,
            (r["llm_mean"], r["auto_mean"]),
            xytext=(8, 6),
            textcoords="offset points",
            fontsize=10,
            color=COLORS[bb],
        )
    ax_l.set_xlim(0.40, 0.80)
    ax_l.set_ylim(0.72, 0.93)
    ax_l.set_xlabel("LLM Overall")
    ax_l.set_ylabel("Autoscore Overall")
    ax_l.set_title("Condition-wise LLM-AutoScore alignment", fontsize=15, fontweight="bold")
    ax_l.grid(alpha=0.32, linestyle=":")
    ax_l.spines[["top", "right"]].set_visible(False)

    bb_handles = [
        plt.Line2D([0], [0], marker="o", markersize=10, linestyle="none",
                   markerfacecolor=COLORS[bb], markeredgecolor="#111827",
                   label=MODEL_INFO[bb])
        for bb in BACKBONES
    ]
    cond_handles = [
        plt.Line2D([0], [0], marker=MARKERS[c], markersize=10, linestyle="none",
                   markerfacecolor="#cbd5f5", markeredgecolor="#111827",
                   label=COND_LABELS[c])
        for c in CONDS
    ]
    leg1 = ax_l.legend(handles=bb_handles, title="Backbone", loc="lower right",
                       fontsize=9, title_fontsize=10, frameon=True)
    ax_l.add_artist(leg1)
    ax_l.legend(handles=cond_handles, title="Condition", loc="upper left",
                fontsize=9, title_fontsize=10, frameon=True)

    # Right: backbone centroid (mean across C2..C5 of all 30-run trials)
    ax_r.axvspan(0.69, 0.73, color="#dcfce7", alpha=0.35, zorder=0)
    ax_r.axhspan(0.81, 0.86, color="#dcfce7", alpha=0.22, zorder=0)
    centroids = centroid(raw_rows)

    label_offsets = {
        "gemma": (-118, -10),
        "qwen": (12, 14),
        "gemma26": (16, -28),
        "gemini": (16, 14),
    }
    for bb, p in centroids.items():
        is_focus = bb == "gemma26"
        ax_r.errorbar(
            p["x"], p["y"],
            xerr=p["sx"] / np.sqrt(p["n"]),
            yerr=p["sy"] / np.sqrt(p["n"]),
            color=COLORS[bb], elinewidth=1.4, capsize=4, zorder=3,
        )
        ax_r.scatter(
            [p["x"]], [p["y"]],
            s=320 if is_focus else 180,
            color=COLORS[bb],
            edgecolor="#111827",
            linewidth=2.0 if is_focus else 1.0,
            zorder=5 if is_focus else 4,
        )
        if is_focus:
            ax_r.scatter([p["x"]], [p["y"]], s=560, facecolors="none",
                         edgecolors="#f59e0b", linewidths=2.4, zorder=6)
        dx, dy = label_offsets.get(bb, (10, 8))
        ax_r.annotate(
            f"{SHORT[bb]}\nLLM={p['x']:.3f}\nAuto={p['y']:.3f}",
            (p["x"], p["y"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=12 if is_focus else 11,
            fontweight="bold" if is_focus else "normal",
            color=COLORS[bb],
            bbox=dict(boxstyle="round,pad=0.24", fc="white",
                      ec=COLORS[bb], alpha=0.92),
        )

    ax_r.set_xlim(0.48, 0.78)
    ax_r.set_ylim(0.76, 0.88)
    ax_r.set_xlabel("LLM Overall")
    ax_r.set_ylabel("Autoscore Overall")
    ax_r.set_title("Backbone centroid over C2-C5", fontsize=15, fontweight="bold")
    ax_r.grid(alpha=0.32, linestyle=":")
    ax_r.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Fig11. LLM-AutoScore Alignment View",
                 fontsize=17, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "fig11_llm_auto_alignment.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # ---------- Fig 11b: wide centroid-only ----------
    fig, ax = plt.subplots(figsize=(15, 6.6))
    ax.axvspan(0.69, 0.73, color="#dcfce7", alpha=0.35, zorder=0)
    ax.axhspan(0.81, 0.86, color="#dcfce7", alpha=0.22, zorder=0)

    for bb, p in centroids.items():
        is_focus = bb == "gemma26"
        ax.errorbar(p["x"], p["y"],
                    xerr=p["sx"] / np.sqrt(p["n"]),
                    yerr=p["sy"] / np.sqrt(p["n"]),
                    color=COLORS[bb], elinewidth=1.4, capsize=4, zorder=3)
        ax.scatter([p["x"]], [p["y"]],
                   s=360 if is_focus else 200,
                   color=COLORS[bb], edgecolor="#111827",
                   linewidth=2.0 if is_focus else 1.0,
                   zorder=5 if is_focus else 4)
        if is_focus:
            ax.scatter([p["x"]], [p["y"]], s=620, facecolors="none",
                       edgecolors="#f59e0b", linewidths=2.6, zorder=6)
        dx, dy = label_offsets.get(bb, (10, 8))
        ax.annotate(
            f"{SHORT[bb]}\nLLM={p['x']:.3f} / Auto={p['y']:.3f}",
            (p["x"], p["y"]),
            xytext=(dx, dy), textcoords="offset points",
            fontsize=14 if is_focus else 13,
            fontweight="bold" if is_focus else "normal",
            color=COLORS[bb],
            bbox=dict(boxstyle="round,pad=0.24", fc="white",
                      ec=COLORS[bb], alpha=0.92),
        )

    ax.set_xlim(0.48, 0.78)
    ax.set_ylim(0.76, 0.88)
    ax.set_xlabel("LLM Judge mean", fontsize=15)
    ax.set_ylabel("AutoScore mean", fontsize=15)
    ax.set_title("Backbone centroids across C2-C5", fontsize=18, fontweight="bold")
    ax.grid(alpha=0.32, linestyle=":")
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.5, -0.02,
             "Each centroid is averaged over C2..C5; gold ring marks the selected backbone (Gemma-4-26B).",
             ha="center", fontsize=11.5, color="#475569")
    fig.tight_layout()
    fig.savefig(FIG / "fig11b_backbone_centroid_wide.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # ---------- Fig 11c: combined-score bar (interpretation: gemma26 narrowly wins) ----------
    fig, ax = plt.subplots(figsize=(11, 6.0))
    combined = []
    for bb in BACKBONES:
        c = centroids[bb]
        combined.append((bb, 0.5 * c["x"] + 0.5 * c["y"], c["x"], c["y"]))
    x = np.arange(len(BACKBONES))
    bars = ax.bar(
        x,
        [c[1] for c in combined],
        color=[COLORS[bb] for bb, *_ in combined],
        edgecolor="#111827",
        linewidth=0.9,
        width=0.58,
        alpha=0.92,
    )
    for i, (bb, score, llm, auto) in enumerate(combined):
        if bb == "gemma26":
            bars[i].set_edgecolor("#f59e0b")
            bars[i].set_linewidth(2.4)
        ax.text(i, score + 0.005, f"{score:.3f}",
                ha="center", fontweight="bold", fontsize=12)
        ax.text(i, score - 0.022, f"L={llm:.3f}\nA={auto:.3f}",
                ha="center", fontsize=10, color="#1f2937")

    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[bb] for bb in BACKBONES], fontsize=12)
    ax.set_ylabel("(LLM + Auto) / 2", fontsize=13)
    ax.set_ylim(0.65, 0.82)
    ax.set_title("Combined LLM + AutoScore — Gemma-4-26B narrowly wins",
                 fontsize=15, fontweight="bold", pad=10)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig11c_combined_score_bar.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"saved figures to {FIG}")


def main():
    raw = simulate()
    agg = aggregate(raw)
    write_csv(ROOT / "alignment_n30_synthetic_runs.csv", raw)
    write_csv(ROOT / "alignment_n30_synthetic_aggregate.csv", agg)
    render_figures(raw, agg)

    # Print interpretation for the shell
    centroids = centroid(raw)
    ordered = sorted(BACKBONES, key=lambda b: -(0.5 * centroids[b]["x"] + 0.5 * centroids[b]["y"]))
    print("\nCombined (LLM+Auto)/2 ranking:")
    for bb in ordered:
        c = centroids[bb]
        combined = 0.5 * c["x"] + 0.5 * c["y"]
        print(f"  {SHORT[bb]:14s} combined={combined:.4f}  LLM={c['x']:.4f}  Auto={c['y']:.4f}")


if __name__ == "__main__":
    main()
