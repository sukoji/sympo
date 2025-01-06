"""Build slide-oriented decomposed figures for result interpretation.

This script intentionally separates interpretation questions that were mixed
in the all-in-one comparison figures:
  1. What changes from C0 LLM-only to the selected system stage?
  2. How does one backbone behave as discussion rounds increase?
  3. Which model should be selected at C3?
  4. What is the quality/cost tradeoff?
  5. Where are the round/model patterns visible at a glance?

Output:
  eval_results/comparison_4backbones/figures_interpretation/
"""
import csv
import json
import os
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


BASE = Path(__file__).resolve().parents[1]
ROOT = BASE
OUT = ROOT / "comparison_4backbones" / "figures_interpretation"
OUT.mkdir(parents=True, exist_ok=True)

SUMMARY = ROOT / "comparison_4backbones" / "summary_4backbones.csv"

CONDS = ["C0_llm_only", "C1_with_assign", "C2_1round", "C3_3rounds", "C4_with_disc", "C5_5rounds"]
ROUND_CONDS = ["C0_llm_only", "C1_with_assign", "C2_1round", "C3_3rounds", "C5_5rounds"]
ROUND_LABELS = ["C0\nLLM-only", "C1\n+assign", "C2\n+1R", "C3\n+3R", "C5\n+5R"]
BACKBONES = ["gemma", "qwen", "gemma26", "gemini"]
SHORT = {
    "gemma": "Gemma-4B",
    "qwen": "Qwen3-14B",
    "gemma26": "Gemma-4-26B",
    "gemini": "Gemini API",
}
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
DIM_COLORS = {
    "llm": "#111827",
    "auto": "#16a34a",
    "time": "#64748b",
    "selected": "#f59e0b",
    "muted": "#cbd5e1",
}
REMOTE_API = {"gemma": False, "qwen": False, "gemma26": False, "gemini": True}
GPU_COUNT = {"gemma": 2, "qwen": 3, "gemma26": 3, "gemini": 1}
FOCUS_BB = "gemma26"

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.dpi": 120,
})

FS_NOTE = 12
FS_ANNOT = 12
FS_VALUE = 12
FS_SUPTITLE = 18


def parse_float(v):
    if v in (None, "", "NA"):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def load_summary():
    data = {bb: {} for bb in BACKBONES}
    with open(SUMMARY, newline="") as f:
        for r in csv.DictReader(f):
            bb = r["backbone"]
            cond = r["condition"]
            if bb not in data:
                continue
            data[bb][cond] = {
                "llm": parse_float(r.get("Overall_mean")),
                "llm_std": parse_float(r.get("Overall_std")) or 0.0,
                "auto": parse_float(r.get("AutoOverall_mean")),
                "auto_std": parse_float(r.get("AutoOverall_std")) or 0.0,
                "structure": parse_float(r.get("S_mean")),
                "assignment": parse_float(r.get("A_mean")),
                "debate": parse_float(r.get("D_mean")),
                "auto_quality": parse_float(r.get("AutoQuality_mean")),
                "auto_allocation": parse_float(r.get("AutoAllocation_mean")),
                "auto_orchestration": parse_float(r.get("AutoOrchestration_mean")),
            }
    return data


def load_timing():
    timing = {bb: {c: [] for c in CONDS} for bb in BACKBONES}

    gm_path = ROOT / "gemma_ablation" / "experiment_metadata.json"
    if gm_path.exists():
        rows = json.load(open(gm_path))
        order = sum([[c] * 3 for c in CONDS], [])
        for c, entry in zip(order, rows):
            if entry.get("elapsed_sec"):
                timing["gemma"][c].append(float(entry["elapsed_sec"]))

    for bb, rel in [
        ("qwen", "qwen_ablation/summary_finegrain.csv"),
        ("gemma26", "gemma26_ablation/summary_finegrain.csv"),
        ("gemini", "gemini_ablation/summary_finegrain.csv"),
    ]:
        path = ROOT / rel
        if not path.exists():
            continue
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                c = r.get("condition", "")
                if c not in CONDS:
                    continue
                v = parse_float(r.get("elapsed_sec"))
                if v is not None:
                    timing[bb][c].append(v)

    out = {bb: {} for bb in BACKBONES}
    for bb in BACKBONES:
        for c in CONDS:
            vals = timing[bb][c]
            if not vals:
                out[bb][c] = {
                    "wall_min": None,
                    "wall_std": 0.0,
                    "gpu_min": None,
                    "gpu_std": 0.0,
                }
                continue
            wall = [v / 60.0 for v in vals]
            gpu = [v * GPU_COUNT[bb] / 60.0 for v in vals]
            out[bb][c] = {
                "wall_min": mean(wall),
                "wall_std": stdev(wall) if len(wall) > 1 else 0.0,
                "gpu_min": mean(gpu),
                "gpu_std": stdev(gpu) if len(gpu) > 1 else 0.0,
            }
    return out


DATA = load_summary()
TIMING = load_timing()


def savefig(name):
    path = OUT / name
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"saved {path}")


def style_axis(ax):
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def annotate_delta(ax, x0, x1, y0, y1, color, label, bold=False, offset=(8, 0)):
    ax.annotate(
        label,
        xy=(x1, y1),
        xytext=offset,
        textcoords="offset points",
        va="center",
        fontsize=FS_ANNOT,
        fontweight="bold" if bold else "normal",
        color=color,
    )


# Fig A: LLM-only vs selected system stage
fig, axes = plt.subplots(1, 2, figsize=(16.2, 7.0), sharey=False)
for ax, metric, title, ylim in [
    (axes[0], "llm", "A. LLM Judge: C0 baseline to C3 system", (0.38, 0.82)),
    (axes[1], "auto", "B. AutoScore v2: missing-stage penalty made explicit", (0.38, 0.90)),
]:
    label_offsets = {
        "llm": {
            "gemma": (8, -2),
            "qwen": (8, -2),
            "gemma26": (8, 0),
            "gemini": (8, 0),
        },
        "auto": {
            "gemini": (8, 16),
            "qwen": (8, 5),
            "gemma26": (8, -6),
            "gemma": (8, -17),
        },
    }[metric]
    for bb in BACKBONES:
        y0 = DATA[bb]["C0_llm_only"][metric]
        y1 = DATA[bb]["C3_3rounds"][metric]
        if y0 is None or y1 is None:
            continue
        focus = bb == FOCUS_BB
        color = COLORS[bb]
        alpha = 1.0 if focus else 0.45
        lw = 3.0 if focus else 1.7
        ax.plot([0, 1], [y0, y1], color=color, alpha=alpha, linewidth=lw, marker="o", markersize=8)
        ax.scatter([1], [y1], s=170 if focus else 80, color=color, edgecolor="#111827", linewidth=1.4 if focus else 0.6, zorder=5)
        delta = y1 - y0
        annotate_delta(ax, 0, 1, y0, y1, color, f"{SHORT[bb]}  {delta:+.3f}", bold=focus, offset=label_offsets[bb])
    ax.axvspan(-0.18, 0.18, color="#e5e7eb", alpha=0.8, zorder=0)
    ax.axvspan(0.82, 1.18, color="#dcfce7", alpha=0.55, zorder=0)
    ax.text(0, ylim[1] - 0.02, "LLM-only\nbaseline", ha="center", va="top", fontsize=FS_NOTE, color="#475569")
    ax.text(1, ylim[1] - 0.02, "Selected\nC3", ha="center", va="top", fontsize=FS_NOTE, color="#166534", fontweight="bold")
    ax.set_xlim(-0.28, 1.70 if metric == "auto" else 1.55)
    ax.set_ylim(*ylim)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["C0", "C3"])
    ax.set_ylabel("Mean score")
    ax.set_title(title)
    style_axis(ax)
fig.suptitle("Fig A. Decompose System Effect: LLM-only vs Selected C3", y=1.02, fontsize=FS_SUPTITLE)
fig.text(
    0.5, -0.02,
    "C0 has no assignment/debate stages. AutoScore is full-scope, so missing stages are intentionally penalized.",
    ha="center", fontsize=FS_NOTE, color="#64748b",
)
savefig("figA_llm_only_vs_c3_system_effect.png")


# Fig B: Gemma-4-26B round trajectory and runtime
fig, (ax_s, ax_t) = plt.subplots(2, 1, figsize=(14.5, 9.6), sharex=True, gridspec_kw={"height_ratios": [1.2, 0.8]})
x = np.arange(len(ROUND_CONDS))
auto = [DATA[FOCUS_BB][c]["auto"] for c in ROUND_CONDS]
auto_std = [DATA[FOCUS_BB][c]["auto_std"] for c in ROUND_CONDS]
llm = [DATA[FOCUS_BB][c]["llm"] for c in ROUND_CONDS]
llm_std = [DATA[FOCUS_BB][c]["llm_std"] for c in ROUND_CONDS]
wall = [TIMING[FOCUS_BB][c]["wall_min"] for c in ROUND_CONDS]
wall_std = [TIMING[FOCUS_BB][c]["wall_std"] for c in ROUND_CONDS]

ax_s.axvspan(2.72, 3.28, color="#fef3c7", alpha=0.65, zorder=0)
ax_s.errorbar(x, auto, yerr=auto_std, color=COLORS[FOCUS_BB], linewidth=2.7, marker="o", markersize=8, capsize=4, label="AutoScore v2")
ax_s.errorbar(x, llm, yerr=llm_std, color="#111827", linewidth=2.4, marker="s", markersize=7, capsize=4, label="LLM Judge")
ax_s.scatter([3], [llm[3]], s=210, marker="*", color=DIM_COLORS["selected"], edgecolor="#111827", linewidth=1.0, zorder=6)
ax_s.annotate(
    "C3 selected:\npeak LLM Judge",
    xy=(3, llm[3]),
    xytext=(2.15, llm[3] + 0.07),
    arrowprops=dict(arrowstyle="->", color="#92400e", lw=1.0),
    fontsize=FS_ANNOT,
    color="#92400e",
    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#fde68a", alpha=0.94),
)
ax_s.annotate(
    f"C5 costs +{wall[4]-wall[3]:.1f} min\nLLM Judge {llm[4]-llm[3]:+.3f}",
    xy=(4, llm[4]),
    xytext=(3.55, llm[4] - 0.10),
    arrowprops=dict(arrowstyle="->", color="#64748b", lw=1.0),
    fontsize=FS_ANNOT,
    color="#475569",
    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cbd5e1", alpha=0.94),
)
ax_s.set_ylabel("Mean score")
ax_s.set_ylim(0.38, 0.94)
ax_s.set_title("Fig B. Gemma-4-26B Round Trajectory: score improves until C3, C5 is not worth the extra time")
ax_s.legend(loc="lower right", framealpha=0.95)
style_axis(ax_s)

bar_colors = ["#d1d5db", "#94a3b8", "#60a5fa", DIM_COLORS["selected"], "#94a3b8"]
hatches = ["//", "", "", "", "\\\\"]
bars = ax_t.bar(x, wall, yerr=wall_std, capsize=3, color=bar_colors, edgecolor="#334155", linewidth=0.8)
for b, h in zip(bars, hatches):
    b.set_hatch(h)
for i, v in enumerate(wall):
    ax_t.text(i, v + max(wall) * 0.035, f"{v:.1f}m", ha="center", fontsize=FS_VALUE, fontweight="bold" if i == 3 else "normal")
ax_t.set_ylabel("Wall-clock min/run")
ax_t.set_xticks(x)
ax_t.set_xticklabels(ROUND_LABELS)
ax_t.set_ylim(0, max(wall) * 1.23)
ax_t.set_title("Runtime grows with discussion rounds; C3 is the elbow")
style_axis(ax_t)
savefig("figB_gemma26_round_score_runtime.png")


# Fig C: C3 model selection, quality signals separated
fig, ax = plt.subplots(figsize=(14.2, 6.8))
x = np.arange(len(BACKBONES))
w = 0.34
llm_c3 = [DATA[bb]["C3_3rounds"]["llm"] for bb in BACKBONES]
auto_c3 = [DATA[bb]["C3_3rounds"]["auto"] for bb in BACKBONES]
ax.axvspan(1.55, 2.45, color="#dcfce7", alpha=0.55, zorder=0)
bars1 = ax.bar(x - w / 2, llm_c3, w, color="#111827", alpha=0.88, label="LLM Judge", edgecolor="#111827")
bars2 = ax.bar(x + w / 2, auto_c3, w, color=[COLORS[bb] for bb in BACKBONES], alpha=0.88, label="AutoScore v2", edgecolor="#111827")
for i, bb in enumerate(BACKBONES):
    if bb == "gemini":
        bars1[i].set_hatch("//")
        bars2[i].set_hatch("//")
    if bb == FOCUS_BB:
        bars1[i].set_linewidth(2.4)
        bars2[i].set_linewidth(2.4)
for i, (a, b) in enumerate(zip(llm_c3, auto_c3)):
    fw = "bold" if BACKBONES[i] == FOCUS_BB else "normal"
    ax.text(i - w / 2, a + 0.012, f"{a:.3f}", ha="center", fontsize=FS_VALUE, fontweight=fw)
    ax.text(i + w / 2, b + 0.012, f"{b:.3f}", ha="center", fontsize=FS_VALUE, fontweight=fw)
ax.set_xticks(x)
ax.set_xticklabels([SHORT[bb] for bb in BACKBONES])
ax.set_ylim(0.50, 0.90)
ax.set_ylabel("C3 mean score")
ax.set_title("Fig C. Model Selection at C3: keep LLM Judge and AutoScore separate")
style_axis(ax)
legend_extra = [Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="#111827", hatch="//", label="External API")]
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles + legend_extra, labels + ["External API"], loc="lower right", framealpha=0.95)
savefig("figC_c3_model_selection_quality_signals.png")


# Fig D: C3 quality/cost tradeoff
fig, ax = plt.subplots(figsize=(13.8, 7.6))
points = []
for bb in BACKBONES:
    gpu = TIMING[bb]["C3_3rounds"]["gpu_min"]
    llm_score = DATA[bb]["C3_3rounds"]["llm"]
    auto_score = DATA[bb]["C3_3rounds"]["auto"]
    points.append((gpu, llm_score, auto_score, bb))

for gpu, llm_score, auto_score, bb in points:
    if gpu is None or llm_score is None:
        continue
    focus = bb == FOCUS_BB
    size = 220 + 650 * max(auto_score or 0, 0)
    marker = "s" if REMOTE_API[bb] else "o"
    face = "white" if REMOTE_API[bb] else COLORS[bb]
    ax.scatter(
        [gpu], [llm_score],
        s=size,
        marker=marker,
        facecolor=face,
        edgecolor=COLORS[bb],
        linewidth=2.6 if focus else 1.6,
        alpha=0.95 if focus else 0.75,
        zorder=5,
    )
    offset = {
        "gemma": (10, -16),
        "qwen": (-8, 18),
        "gemma26": (12, -26),
        "gemini": (10, 8),
    }[bb]
    ha = "right" if bb == "qwen" else "left"
    ax.annotate(
        f"{SHORT[bb]}\nLLM={llm_score:.3f}\nAuto={auto_score:.3f}",
        (gpu, llm_score),
        xytext=offset,
        textcoords="offset points",
        fontsize=FS_ANNOT,
        fontweight="bold" if focus else "normal",
        color=COLORS[bb],
        ha=ha,
    )
ax.axhline(DATA[FOCUS_BB]["C3_3rounds"]["llm"], color=COLORS[FOCUS_BB], linestyle=":", alpha=0.45)
ax.axvline(TIMING[FOCUS_BB]["C3_3rounds"]["gpu_min"], color=COLORS[FOCUS_BB], linestyle=":", alpha=0.45)
ax.set_xscale("log")
ax.set_xlim(1.1, max(p[0] for p in points if p[0] is not None) * 1.45)
ax.set_ylim(0.525, 0.76)
ax.set_xlabel("GPU-minutes per run at C3 (log; Gemini shown as API unit)")
ax.set_ylabel("LLM Judge overall at C3")
ax.set_title("Fig D. C3 Quality/Cost Tradeoff: Gemma-4-26B is the best local operating point")
ax.grid(axis="both", linestyle=":", alpha=0.35, which="both")
handles = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS[FOCUS_BB], markeredgecolor=COLORS[FOCUS_BB], markersize=10, label="Local model"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor="white", markeredgecolor=COLORS["gemini"], markersize=10, label="External API"),
]
ax.legend(handles=handles, loc="lower left", framealpha=0.95)
savefig("figD_c3_quality_cost_tradeoff.png")


# Fig E: Model × round heatmap for quick pattern reading
fig, axes = plt.subplots(1, 2, figsize=(16.0, 7.0), sharey=True)
for ax, metric, title, cmap, vmin, vmax in [
    (axes[0], "llm", "A. LLM Judge overall", "Greys", 0.42, 0.80),
    (axes[1], "auto", "B. AutoScore v2", "YlGn", 0.40, 0.88),
]:
    matrix = np.array([[DATA[bb][c][metric] for c in ROUND_CONDS] for bb in BACKBONES], dtype=float)
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    for r, bb in enumerate(BACKBONES):
        for col, c in enumerate(ROUND_CONDS):
            val = matrix[r, col]
            color = "white" if val > (vmin + vmax) / 2 and metric == "llm" else "#111827"
            ax.text(col, r, f"{val:.3f}", ha="center", va="center", fontsize=FS_VALUE, color=color, fontweight="bold" if (bb == FOCUS_BB and c == "C3_3rounds") else "normal")
    # C0 is stage-limited; hatch the entire column.
    ax.add_patch(Rectangle((-0.5, -0.5), 1.0, len(BACKBONES), fill=False, hatch="///", edgecolor="#94a3b8", linewidth=0.0))
    # Selected cell.
    ax.add_patch(Rectangle((2.5, 1.5), 1.0, 1.0, fill=False, edgecolor=DIM_COLORS["selected"], linewidth=3.0))
    # API row.
    ax.add_patch(Rectangle((-0.5, 2.5), len(ROUND_CONDS), 1.0, fill=False, hatch="\\\\", edgecolor="#2563eb", linewidth=0.0))
    ax.set_xticks(np.arange(len(ROUND_CONDS)))
    ax.set_xticklabels(ROUND_LABELS)
    ax.set_yticks(np.arange(len(BACKBONES)))
    ax.set_yticklabels([SHORT[bb] for bb in BACKBONES])
    ax.set_title(title)
    ax.tick_params(length=0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.ax.tick_params(labelsize=8)
fig.suptitle("Fig E. Model × Round Decomposition: read patterns without mixing axes", y=1.02, fontsize=FS_SUPTITLE)
fig.text(0.5, -0.02, "Gold border = selected Gemma-4-26B C3. Hatched C0 = stage-limited baseline. Hatched Gemini row = external API.", ha="center", fontsize=FS_NOTE, color="#64748b")
savefig("figE_model_round_heatmap.png")


# Fig F: Stepwise gains for Gemma-4-26B
fig, ax = plt.subplots(figsize=(14.2, 6.8))
steps = ["C0->C1", "C1->C2", "C2->C3", "C3->C5"]
llm_delta = [llm[i + 1] - llm[i] for i in range(len(llm) - 1)]
auto_delta = [auto[i + 1] - auto[i] for i in range(len(auto) - 1)]
time_delta = [wall[i + 1] - wall[i] for i in range(len(wall) - 1)]
x = np.arange(len(steps))
w = 0.34
colors_llm = [DIM_COLORS["llm"] if v >= 0 else "#94a3b8" for v in llm_delta]
colors_auto = [COLORS[FOCUS_BB] if v >= 0 else "#86efac" for v in auto_delta]
bars_l = ax.bar(x - w / 2, llm_delta, w, color=colors_llm, edgecolor="#111827", label="LLM Judge delta")
bars_a = ax.bar(x + w / 2, auto_delta, w, color=colors_auto, edgecolor="#166534", label="AutoScore delta")
for i, b in enumerate(bars_l):
    if llm_delta[i] < 0:
        b.set_hatch("//")
for i, b in enumerate(bars_a):
    if auto_delta[i] < 0:
        b.set_hatch("//")
for i, (dl, da, dt) in enumerate(zip(llm_delta, auto_delta, time_delta)):
    ax.text(i - w / 2, dl + (0.006 if dl >= 0 else -0.014), f"{dl:+.3f}", ha="center", va="bottom" if dl >= 0 else "top", fontsize=FS_VALUE)
    ax.text(i + w / 2, da + (0.006 if da >= 0 else -0.014), f"{da:+.3f}", ha="center", va="bottom" if da >= 0 else "top", fontsize=FS_VALUE)
    ax.text(i, -0.085, f"time {dt:+.1f}m", ha="center", fontsize=FS_NOTE, color="#475569")
ax.axhline(0, color="#111827", linewidth=0.8)
ax.axvspan(1.55, 2.45, color="#fef3c7", alpha=0.55, zorder=0)
ax.set_xticks(x)
ax.set_xticklabels(steps)
ax.set_ylim(-0.10, 0.22)
ax.set_ylabel("Score delta")
ax.set_title("Fig F. Gemma-4-26B Stepwise Gains: C2->C3 improves Judge; C3->C5 costs time and hurts Judge")
ax.legend(loc="upper right", framealpha=0.95)
style_axis(ax)
savefig("figF_gemma26_stepwise_gain_tradeoff.png")


with open(OUT / "README.md", "w") as f:
    f.write("# Interpretation Figure Set\n\n")
    f.write("이 폴더는 기존 comparison figure를 발표 해석 단위로 다시 분해한 산출물입니다.\n\n")
    f.write("## 추천 사용 순서\n\n")
    f.write("1. `figA_llm_only_vs_c3_system_effect.png` — LLM-only 대비 시스템 구성의 효과를 먼저 보여줍니다.\n")
    f.write("2. `figB_gemma26_round_score_runtime.png` — Gemma-4-26B 내부에서 C3가 왜 선택되는지 설명합니다.\n")
    f.write("3. `figC_c3_model_selection_quality_signals.png` — C3 조건에서 모델별 품질 신호를 분리해 비교합니다.\n")
    f.write("4. `figD_c3_quality_cost_tradeoff.png` — C3에서 Gemma-4-26B가 local 운영점으로 타당함을 보여줍니다.\n")
    f.write("5. `figE_model_round_heatmap.png` 또는 `figF_gemma26_stepwise_gain_tradeoff.png` — 보조 근거로 사용합니다.\n\n")
    f.write("## 산출물\n\n")
    f.write("- `figA_llm_only_vs_c3_system_effect.png`: LLM-only(C0)와 선택 시스템(C3) 비교\n")
    f.write("- `figB_gemma26_round_score_runtime.png`: Gemma-4-26B 단일 모델 라운드별 점수/시간\n")
    f.write("- `figC_c3_model_selection_quality_signals.png`: C3에서 모델별 AutoScore와 LLM Judge 분리 비교\n")
    f.write("- `figD_c3_quality_cost_tradeoff.png`: C3 모델별 품질-비용 tradeoff\n")
    f.write("- `figE_model_round_heatmap.png`: 모델 x 라운드 패턴 heatmap\n")
    f.write("- `figF_gemma26_stepwise_gain_tradeoff.png`: Gemma-4-26B 단계별 증가분과 시간 비용\n\n")
    f.write("시각 규칙: Gemma-4-26B는 초록/굵은 테두리, 선택 C3는 금색, C0 baseline은 빗금, Gemini API는 빗금 또는 hollow marker로 표시했습니다.\n")

print(f"\nOutput: {OUT}")
