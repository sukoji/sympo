"""
RAG 전략 비교 실험 분석 및 시각화 (R0~R4 × 2 백엔드)
"""
from __future__ import annotations

import csv
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

_KOREAN_FONT_PATHS = [
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]
for _fp in _KOREAN_FONT_PATHS:
    try:
        if os.path.exists(_fp):
            fm.fontManager.addfont(_fp)
            fam = fm.FontProperties(fname=_fp).get_name()
            plt.rcParams["font.family"] = fam
            break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent
BACKENDS = [("gemini", "backend_gemini"), ("gemma4-api", "backend_gemma4_api")]
CONDITIONS = ["R0", "R1", "R2", "R3", "R4"]
COND_LABEL = {
    "R0": "R0 (none)",
    "R1": "R1 (Vanilla)",
    "R2": "R2 (Hybrid)",
    "R3": "R3 (Graph)",
    "R4": "R4 (Agentic)",
}

KEY_METRICS = [
    ("planning_score",       "planning_score",       "Planning Score (↑)",       "higher"),
    ("workload_gini",        "gini",                 "Workload Gini (↓)",        "lower"),
    ("schedule_feasibility", "feasibility",          "Schedule Feasibility (↑)", "higher"),
    ("success_rate",         "success_rate",         "Success Rate (↑)",         "higher"),
    ("mece_score",           "mece_score",           "MECE Score (↑)",           "higher"),
    ("granularity_fitness",  "granularity_fitness",  "Granularity Fitness (↑)",  "higher"),
    ("ragas_faithfulness",   "faithfulness",         "RAGAS Faithfulness (↑)",   "higher"),
    ("buffer_ratio",         "buffer_ratio_pct",     "Buffer Ratio (↑, 0.15~0.30 권장)", "range"),
    ("communication_efficiency", "efficiency",       "Comm Efficiency (↑)",       "higher"),
    ("supervisor_intervention", "intervention_ratio", "Supervisor 개입율 (↓)",    "lower"),
    ("autoscore",            "autoscore",            "AutoScore (↑)",            "higher"),
]
JUDGE_METRICS = [
    ("structure",  "Judge-Structure (↑)",  "higher"),
    ("assignment", "Judge-Assignment (↑)", "higher"),
    ("debate",     "Judge-Debate (↑)",     "higher"),
    ("overall",    "Judge-Overall (↑)",    "higher"),
]
_BUFFER_LABEL_ORIG = "Buffer Ratio (↑, 0.15~0.30 권장)"


def _scale_value(label, v):
    if label == _BUFFER_LABEL_ORIG and isinstance(v, (int, float)) and not math.isnan(v):
        return v / 100.0
    return v


def all_metric_labels():
    labels = [(lbl, d) for (_, _, lbl, d) in KEY_METRICS]
    labels += [(lbl, d) for (_, lbl, d) in JUDGE_METRICS]
    return labels


def load_runs(backend_dir):
    runs = []
    rd = backend_dir / "runs"
    if not rd.exists():
        return runs
    for p in sorted(rd.glob("*.json")):
        try:
            runs.append(json.load(open(p, encoding="utf-8")))
        except Exception as e:
            print(f"[WARN] {p}: {e}")
    return runs


def aggregate(runs):
    agg = {c: {} for c in CONDITIONS}
    for run in runs:
        cond = run["condition"]
        m = run.get("metrics", {})
        for block, field, label, direction in KEY_METRICS:
            sub = m.get(block, {}) if isinstance(m.get(block), dict) else {}
            v = sub.get(field)
            if v is None: continue
            try: v = float(v)
            except: continue
            if v == -1: continue  # N/A
            agg[cond].setdefault(label, {"values": [], "direction": direction})
            agg[cond][label]["values"].append(v)
        j = run.get("judge") or {}
        if j:
            for dim, label, direction in JUDGE_METRICS:
                if dim == "overall":
                    v = j.get("overall")
                else:
                    v = (j.get(dim) or {}).get("score") if isinstance(j.get(dim), dict) else None
                if v is None: continue
                try: v = float(v)
                except: continue
                if v < 0: continue
                agg[cond].setdefault(label, {"values": [], "direction": direction})
                agg[cond][label]["values"].append(v)
    for cond in CONDITIONS:
        for label, rec in agg[cond].items():
            vs = rec["values"]
            rec["mean"] = statistics.mean(vs) if vs else float("nan")
            rec["std"] = statistics.stdev(vs) if len(vs) >= 2 else 0.0
    return agg


# ── 플롯 ──
def plot_conditions(agg, out_path, title, n_runs):
    """조건별(R0~R4) × 지표 비교 플롯"""
    labels = [lbl for (lbl, _) in all_metric_labels()]
    colors = ["#888888", "#4e79a7", "#f28e2c", "#59a14f", "#e15759"]
    fig, ax = plt.subplots(figsize=(18, 6))
    w = 0.15
    x = list(range(len(labels)))
    for i, cond in enumerate(CONDITIONS):
        means = [_scale_value(l, agg[cond].get(l, {}).get("mean", float("nan"))) for l in labels]
        stds = [_scale_value(l, agg[cond].get(l, {}).get("std", 0.0)) for l in labels]
        offset = (i - (len(CONDITIONS)-1)/2) * w
        ax.bar([v+offset for v in x], means, w, yerr=stds, capsize=2,
               label=COND_LABEL[cond], color=colors[i])
    # Judge 경계선
    n_obj = len(KEY_METRICS)
    ax.axvline(n_obj - 0.5, color="gray", linestyle="--", alpha=0.5)
    ax.text(n_obj - 0.4, 1.07, "Judge →", fontsize=9, color="gray")
    # Buffer 권장구간
    if _BUFFER_LABEL_ORIG in labels:
        bi = labels.index(_BUFFER_LABEL_ORIG)
        ax.axvspan(bi - 0.5, bi + 0.5, ymin=0.15/1.15, ymax=0.30/1.15,
                   color="#2ca02c", alpha=0.15, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Score / Fraction (0~1)")
    ax.set_ylim(0, 1.15)
    ax.set_title(f"{title} (N={n_runs}/조건)")
    ax.legend(fontsize=8, ncol=5)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_cross_backend(all_agg, out_path):
    """백엔드별 R0~R4 비교 (10 bars/지표)"""
    labels = [lbl for (lbl, _) in all_metric_labels()]
    fig, ax = plt.subplots(figsize=(20, 6))
    x = list(range(len(labels)))
    backends = list(all_agg.keys())
    n_group = len(backends) * len(CONDITIONS)
    w = 0.85 / n_group
    shades = {
        "gemini": ["#1f3b6a", "#3b5998", "#6689c0", "#9bb4dd", "#c5d5ef"],
        "gemma4-api": ["#8b2d1d", "#c0362a", "#e15759", "#f28e85", "#fbc4be"],
    }
    for bi, backend in enumerate(backends):
        for ci, cond in enumerate(CONDITIONS):
            means = [_scale_value(l, all_agg[backend][cond].get(l, {}).get("mean", float("nan"))) for l in labels]
            stds = [_scale_value(l, all_agg[backend][cond].get(l, {}).get("std", 0.0)) for l in labels]
            offset = (bi * len(CONDITIONS) + ci - (n_group - 1) / 2) * w
            color = shades.get(backend, ["#777"]*5)[ci]
            ax.bar([v+offset for v in x], means, w, yerr=stds, capsize=2,
                   label=f"{backend}/{cond}", color=color)
    n_obj = len(KEY_METRICS)
    ax.axvline(n_obj - 0.5, color="gray", linestyle="--", alpha=0.5)
    ax.text(n_obj - 0.4, 1.07, "Judge →", fontsize=9, color="gray")
    if _BUFFER_LABEL_ORIG in labels:
        bi = labels.index(_BUFFER_LABEL_ORIG)
        ax.axvspan(bi - 0.5, bi + 0.5, ymin=0.15/1.15, ymax=0.30/1.15,
                   color="#2ca02c", alpha=0.15, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Score / Fraction (0~1)")
    ax.set_ylim(0, 1.15)
    ax.set_title(f"RAG 전략 × LLM 백엔드 교차 비교 (N=5/조건)")
    ax.legend(fontsize=7, ncol=5)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def write_summary_table(agg, out_path, backend):
    labels = [lbl for (lbl, _) in all_metric_labels()]
    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"# RAG 전략 비교 — {backend} (N=5/조건)\n\n")
        f.write("| 지표 | " + " | ".join(COND_LABEL[c] for c in CONDITIONS) + " |\n")
        f.write("|---" * (len(CONDITIONS) + 1) + "|\n")
        for lbl in labels:
            row = [lbl]
            for c in CONDITIONS:
                r = agg[c].get(lbl, {})
                m, s = r.get("mean"), r.get("std", 0)
                row.append(f"{m:.4f} ± {s:.4f}" if m is not None and not math.isnan(m) else "—")
            f.write("| " + " | ".join(row) + " |\n")


def write_summary_csv(agg, out_path, backend):
    labels = [lbl for (lbl, _) in all_metric_labels()]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["metric"] + [f"{c}_mean" for c in CONDITIONS] + [f"{c}_std" for c in CONDITIONS]
        w.writerow(header)
        for lbl in labels:
            row = [lbl]
            for c in CONDITIONS:
                m = agg[c].get(lbl, {}).get("mean")
                row.append(f"{m:.4f}" if m is not None and not math.isnan(m) else "")
            for c in CONDITIONS:
                s = agg[c].get(lbl, {}).get("std", 0)
                row.append(f"{s:.4f}")
            w.writerow(row)


def write_raw_csv(runs, out_path):
    metric_cols = [(b, f, l) for (b, f, l, _) in KEY_METRICS]
    judge_cols = [(d, l) for (d, l, _) in JUDGE_METRICS]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["backend", "condition", "iter", "elapsed_sec"]
                   + [l for _,_,l in metric_cols] + [l for _,l in judge_cols])
        for r in runs:
            m = r.get("metrics", {})
            j = r.get("judge") or {}
            row = [r.get("backend"), r.get("condition"), r.get("iter"), r.get("elapsed_sec")]
            for b, fld, _ in metric_cols:
                v = m.get(b, {}).get(fld) if isinstance(m.get(b), dict) else None
                row.append(f"{v:.4f}" if isinstance(v, (int, float)) and v != -1 else "")
            for dim, _ in judge_cols:
                v = j.get("overall") if dim == "overall" else (j.get(dim) or {}).get("score")
                row.append(f"{v:.4f}" if isinstance(v, (int, float)) and v >= 0 else "")
            w.writerow(row)


# ── 통계 검정 (Mann-Whitney U + Cliff's δ) ──
def _mann_whitney_u(a, b):
    """양측 U 검정, scipy 없으면 fallback. 작은 표본용."""
    try:
        from scipy.stats import mannwhitneyu
        if len(a) < 2 or len(b) < 2: return None, None
        res = mannwhitneyu(a, b, alternative="two-sided")
        return float(res.statistic), float(res.pvalue)
    except Exception:
        return None, None


def _cliffs_delta(a, b):
    """Cliff's δ = (#(a>b) − #(a<b)) / (n_a × n_b)"""
    if not a or not b: return float("nan")
    gt = sum(1 for x in a for y in b if x > y)
    lt = sum(1 for x in a for y in b if x < y)
    return (gt - lt) / (len(a) * len(b))


def statistical_tests(agg_runs_by_backend):
    """각 백엔드 내 R0 vs R1~R4 Mann-Whitney + Cliff's δ + Holm-Bonferroni"""
    out = {}
    for backend, runs in agg_runs_by_backend.items():
        # condition별 값 수집 (Judge-Overall 기준)
        by_cond = {c: [] for c in CONDITIONS}
        for r in runs:
            j = r.get("judge") or {}
            if j.get("overall") is not None and j["overall"] >= 0:
                by_cond[r["condition"]].append(j["overall"])
        tests = []
        for c in ["R1", "R2", "R3", "R4"]:
            u, p = _mann_whitney_u(by_cond["R0"], by_cond[c])
            delta = _cliffs_delta(by_cond[c], by_cond["R0"])
            tests.append({"vs": f"R0 vs {c}", "U": u, "p": p, "cliffs_delta": delta,
                          "n_R0": len(by_cond["R0"]), "n_other": len(by_cond[c])})
        # Holm-Bonferroni
        pvals = [t["p"] for t in tests if t["p"] is not None]
        if pvals:
            sorted_idx = sorted(range(len(tests)), key=lambda i: (tests[i]["p"] if tests[i]["p"] is not None else 1.0))
            m = len(pvals)
            for rank, i in enumerate(sorted_idx):
                if tests[i]["p"] is not None:
                    tests[i]["p_holm"] = min(tests[i]["p"] * (m - rank), 1.0)
        out[backend] = tests
    return out


def main():
    all_agg = {}
    all_runs_by_backend = {}
    all_runs = []
    for backend, sub in BACKENDS:
        bd = ROOT / sub
        if not bd.exists():
            print(f"[SKIP] {bd}")
            continue
        runs = load_runs(bd)
        if not runs:
            print(f"[SKIP] {bd}/runs 비어있음")
            continue
        agg = aggregate(runs)
        all_agg[backend] = agg
        all_runs_by_backend[backend] = runs
        all_runs.extend(runs)

        plot_conditions(agg, bd / "comparison_plot.png", f"RAG 전략 비교 — {backend}", 5)
        write_summary_table(agg, bd / "summary_table.md", backend)
        write_summary_csv(agg, bd / "summary.csv", backend)
        write_raw_csv(runs, bd / "raw_iterations.csv")
        print(f"[OK] {backend}: {len(runs)} runs → {bd}")

    if not all_agg:
        print("[ERROR] 집계할 결과 없음")
        sys.exit(1)

    plot_cross_backend(all_agg, ROOT / "cross_backend_plot.png")
    write_raw_csv(all_runs, ROOT / "all_iterations.csv")

    # 통계 검정
    tests = statistical_tests(all_runs_by_backend)
    stats_path = ROOT / "statistical_tests.json"
    stats_path.write_text(json.dumps(tests, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] 통계검정 → {stats_path}")


if __name__ == "__main__":
    main()
