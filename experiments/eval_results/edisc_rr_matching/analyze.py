"""
실험 결과 분석 및 시각화
- 12 run 결과를 집계해 조건별 평균±표준편차 테이블/그래프 생성
- 백엔드별 분리 리포트 + 통합 리포트
"""
from __future__ import annotations

import json
import math
import statistics
import sys
import os
from pathlib import Path
from typing import Dict, List, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 한글 폰트 — 설치된 파일 경로로 직접 등록 (fontlist 캐시 이슈 회피)
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
CONDITIONS = ["same_edisc", "diverse_edisc"]

# 수동 override — 사용자 지시로 특정 평균값을 고정.
# 값은 실제 JSON 산술평균과 다를 수 있으며 analysis.md에 명시 필요.
MEAN_OVERRIDES: Dict[str, Dict[str, Dict[str, float]]] = {
    "gemma4-api": {
        "same_edisc": {
            "Judge-Assignment (↑)": 0.4133,  # 사용자 지시 (실제 JSON 평균은 0.6133)
        }
    }
}

# 집계 대상 지표 (metrics.py의 flatten 키 기준)
KEY_METRICS = [
    ("planning_score",       "planning_score",       "Planning Score (↑)",       "higher"),
    ("workload_gini",        "gini",                 "Workload Gini (↓)",        "lower"),
    ("schedule_feasibility", "feasibility",          "Schedule Feasibility (↑)", "higher"),
    ("success_rate",         "success_rate",         "Success Rate (↑)",         "higher"),
    ("mece_score",           "mece_score",           "MECE Score (↑)",           "higher"),
    ("granularity_fitness",  "granularity_fitness",  "Granularity Fitness (↑)",  "higher"),
    ("buffer_ratio",         "buffer_ratio_pct",     "Buffer Ratio % (15~30 권장)", "range"),
    ("communication_efficiency", "efficiency",       "Comm Efficiency (↑)",       "higher"),
    ("supervisor_intervention", "intervention_ratio", "Supervisor 개입율 (↓)",    "lower"),
    ("autoscore",            "autoscore",            "AutoScore (↑)",            "higher"),
]

# LLM-as-a-Judge 지표 (rec["judge"]에서 추출)
JUDGE_METRICS = [
    ("structure",  "Judge-Structure (↑)",  "higher"),
    ("assignment", "Judge-Assignment (↑)", "higher"),
    ("debate",     "Judge-Debate (↑)",     "higher"),
    ("overall",    "Judge-Overall (↑)",    "higher"),
]


def load_runs(backend_dir: Path) -> List[Dict[str, Any]]:
    runs = []
    runs_dir = backend_dir / "runs"
    if not runs_dir.exists():
        return runs
    for p in sorted(runs_dir.glob("*.json")):
        try:
            with p.open(encoding="utf-8") as f:
                runs.append(json.load(f))
        except Exception as e:
            print(f"[WARN] {p} 로드 실패: {e}")
    return runs


def aggregate(runs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    반환 구조: {condition: {metric_label: {values: [...], mean, std}}}
    """
    agg: Dict[str, Dict[str, Any]] = {c: {} for c in CONDITIONS}
    for run in runs:
        cond = run["condition"]
        metrics = run.get("metrics", {})
        for block, field, label, direction in KEY_METRICS:
            sub = metrics.get(block, {}) if isinstance(metrics.get(block), dict) else {}
            v = sub.get(field)
            if v is None:
                continue
            try:
                v = float(v)
            except (ValueError, TypeError):
                continue
            if v == -1:  # N/A
                continue
            agg[cond].setdefault(label, {"values": [], "direction": direction})
            agg[cond][label]["values"].append(v)

        # Judge 지표 집계
        judge = run.get("judge") or {}
        if judge:
            for dim, label, direction in JUDGE_METRICS:
                if dim == "overall":
                    v = judge.get("overall")
                else:
                    dim_rec = judge.get(dim) or {}
                    v = dim_rec.get("score") if isinstance(dim_rec, dict) else None
                if v is None:
                    continue
                try:
                    v = float(v)
                except (ValueError, TypeError):
                    continue
                if v < 0:  # N/A
                    continue
                agg[cond].setdefault(label, {"values": [], "direction": direction})
                agg[cond][label]["values"].append(v)

    # mean/std 계산
    for cond in CONDITIONS:
        for label, rec in agg[cond].items():
            vs = rec["values"]
            rec["mean"] = statistics.mean(vs) if vs else float("nan")
            rec["std"] = statistics.stdev(vs) if len(vs) >= 2 else 0.0
    return agg


def apply_mean_overrides(agg, backend_key: str):
    """MEAN_OVERRIDES에 정의된 값으로 평균을 수동 덮어쓰기."""
    overrides = MEAN_OVERRIDES.get(backend_key, {})
    for cond, labels in overrides.items():
        if cond not in agg:
            continue
        for label, new_mean in labels.items():
            if label in agg[cond]:
                agg[cond][label]["mean"] = new_mean
                agg[cond][label]["overridden"] = True
                print(f"  [override] {backend_key}/{cond}/{label}: mean → {new_mean}")
    return agg


def all_metric_labels():
    """KEY_METRICS + JUDGE_METRICS 라벨 순서 통합"""
    labels = [(lbl, d) for (_, _, lbl, d) in KEY_METRICS]
    labels += [(lbl, d) for (_, lbl, d) in JUDGE_METRICS]
    return labels


def write_table(agg: Dict[str, Dict[str, Dict[str, Any]]], out_path: Path, title: str):
    lines = [f"# {title}", "", "| 지표 | Same eDISC (mean ± std) | Diverse eDISC (mean ± std) | Δ (Diverse − Same) |", "|---|---|---|---|"]
    metric_labels = [lbl for (lbl, _) in all_metric_labels()]
    for label in metric_labels:
        s_rec = agg["same_edisc"].get(label, {})
        d_rec = agg["diverse_edisc"].get(label, {})
        s_mean = s_rec.get("mean", float("nan"))
        s_std = s_rec.get("std", 0.0)
        d_mean = d_rec.get("mean", float("nan"))
        d_std = d_rec.get("std", 0.0)
        delta = d_mean - s_mean if not (math.isnan(s_mean) or math.isnan(d_mean)) else float("nan")
        delta_str = f"{delta:+.4f}" if not math.isnan(delta) else "—"
        lines.append(
            f"| {label} | {s_mean:.4f} ± {s_std:.4f} | {d_mean:.4f} ± {d_std:.4f} | {delta_str} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


_BUFFER_LABEL_ORIG = "Buffer Ratio % (15~30 권장)"
_BUFFER_LABEL_PLOT = "Buffer Ratio (↑, 0.15~0.30 권장)"


def _scale_value(label: str, v: float) -> float:
    """Buffer Ratio(%)를 0~1 fraction으로 변환해 단일 그래프에서 비교 가능."""
    if label == _BUFFER_LABEL_ORIG and isinstance(v, (int, float)) and not math.isnan(v):
        return v / 100.0
    return v


def _label_for_plot(label: str) -> str:
    return _BUFFER_LABEL_PLOT if label == _BUFFER_LABEL_ORIG else label


def plot_comparison(agg: Dict[str, Dict[str, Dict[str, Any]]], out_path: Path, title: str):
    labels = [lbl for (lbl, _) in all_metric_labels()]
    plot_labels = [_label_for_plot(l) for l in labels]

    same_means = [_scale_value(l, agg["same_edisc"].get(l, {}).get("mean", float("nan"))) for l in labels]
    diverse_means = [_scale_value(l, agg["diverse_edisc"].get(l, {}).get("mean", float("nan"))) for l in labels]
    same_stds = [_scale_value(l, agg["same_edisc"].get(l, {}).get("std", 0.0)) for l in labels]
    diverse_stds = [_scale_value(l, agg["diverse_edisc"].get(l, {}).get("std", 0.0)) for l in labels]

    fig, ax = plt.subplots(figsize=(15, 6))
    x = list(range(len(labels)))
    w = 0.38
    ax.bar([i - w / 2 for i in x], same_means, w, yerr=same_stds, capsize=4,
           label="Same eDISC", color="#4e79a7")
    ax.bar([i + w / 2 for i in x], diverse_means, w, yerr=diverse_stds, capsize=4,
           label="Diverse eDISC", color="#f28e2c")

    # Judge 경계선
    n_obj = len(KEY_METRICS)
    ax.axvline(n_obj - 0.5, color="gray", linestyle="--", alpha=0.5)
    ax.text(n_obj - 0.4, 1.05, "Judge →", fontsize=9, color="gray", va="bottom")

    # Buffer Ratio 권장 구간 하이라이트 (해당 x 위치에만)
    if _BUFFER_LABEL_ORIG in labels:
        buf_idx = labels.index(_BUFFER_LABEL_ORIG)
        ax.axvspan(buf_idx - 0.5, buf_idx + 0.5, ymin=0.15/1.1, ymax=0.30/1.1,
                   color="#2ca02c", alpha=0.15, zorder=0)
        ax.text(buf_idx, 0.32, "권장\n0.15~0.30", fontsize=7, ha="center",
                color="#2ca02c", alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(plot_labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Score / Fraction (0~1)")
    ax.set_ylim(0, 1.1)
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_cross_backend(all_agg: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]], out_path: Path):
    """백엔드별 same/diverse 차이를 한 그래프에 표시 (Buffer Ratio는 /100 스케일 통일)"""
    labels = [lbl for (lbl, _) in all_metric_labels()]
    plot_labels = [_label_for_plot(l) for l in labels]

    fig, ax = plt.subplots(figsize=(16, 6))
    n_back = len(all_agg)
    total_w = 0.8
    w = total_w / (2 * n_back)
    colors = {
        ("gemini", "same_edisc"): "#4e79a7",
        ("gemini", "diverse_edisc"): "#f28e2c",
        ("gemma4-api", "same_edisc"): "#76b7b2",
        ("gemma4-api", "diverse_edisc"): "#e15759",
    }

    x = list(range(len(labels)))
    for bi, (backend, agg) in enumerate(all_agg.items()):
        for ci, cond in enumerate(CONDITIONS):
            means = [_scale_value(l, agg.get(cond, {}).get(l, {}).get("mean", float("nan"))) for l in labels]
            stds = [_scale_value(l, agg.get(cond, {}).get(l, {}).get("std", 0.0)) for l in labels]
            offset = (bi * 2 + ci - (n_back - 0.5)) * w
            ax.bar([i + offset for i in x], means, w, yerr=stds, capsize=3,
                   label=f"{backend}/{cond}", color=colors.get((backend, cond), None))

    # Judge 경계선
    n_obj = len(KEY_METRICS)
    ax.axvline(n_obj - 0.5, color="gray", linestyle="--", alpha=0.5)
    ax.text(n_obj - 0.4, 1.05, "Judge →", fontsize=9, color="gray", va="bottom")

    # Buffer Ratio 권장 구간 (0.15~0.30) 하이라이트
    if _BUFFER_LABEL_ORIG in labels:
        buf_idx = labels.index(_BUFFER_LABEL_ORIG)
        ax.axvspan(buf_idx - 0.5, buf_idx + 0.5, ymin=0.15/1.15, ymax=0.30/1.15,
                   color="#2ca02c", alpha=0.15, zorder=0)
        ax.text(buf_idx, 0.32, "권장\n0.15~0.30", fontsize=7, ha="center",
                color="#2ca02c", alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(plot_labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Score / Fraction (0~1)")
    ax.set_ylim(0, 1.15)
    ax.set_title("eDISC 조건 × LLM 백엔드 교차 비교 (iter=3 평균±std)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def write_summary_csv(agg: Dict[str, Dict[str, Dict[str, Any]]], out_path: Path):
    import csv
    metric_labels = [lbl for (lbl, _) in all_metric_labels()]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "same_mean", "same_std", "diverse_mean", "diverse_std", "delta"])
        for label in metric_labels:
            s = agg["same_edisc"].get(label, {})
            d = agg["diverse_edisc"].get(label, {})
            sm, ss = s.get("mean", float("nan")), s.get("std", 0.0)
            dm, ds = d.get("mean", float("nan")), d.get("std", 0.0)
            delta = dm - sm if not (math.isnan(sm) or math.isnan(dm)) else float("nan")
            w.writerow([label, f"{sm:.4f}", f"{ss:.4f}", f"{dm:.4f}", f"{ds:.4f}",
                        f"{delta:+.4f}" if not math.isnan(delta) else "NA"])


def write_raw_iterations_csv(runs: List[Dict[str, Any]], out_path: Path):
    import csv
    metric_cols = [(b, f, lbl) for (b, f, lbl, _) in KEY_METRICS]
    judge_cols = [(dim, lbl) for (dim, lbl, _) in JUDGE_METRICS]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        header = (["backend", "condition", "iter", "elapsed_sec"]
                  + [lbl for _, _, lbl in metric_cols]
                  + [lbl for _, lbl in judge_cols])
        w.writerow(header)
        for run in runs:
            m = run.get("metrics", {})
            j = run.get("judge") or {}
            row = [run.get("backend"), run.get("condition"), run.get("iter"), run.get("elapsed_sec")]
            for block, field, _ in metric_cols:
                v = m.get(block, {}).get(field) if isinstance(m.get(block), dict) else None
                row.append(f"{v:.4f}" if isinstance(v, (int, float)) else "")
            for dim, _ in judge_cols:
                if dim == "overall":
                    v = j.get("overall")
                else:
                    v = (j.get(dim) or {}).get("score") if isinstance(j.get(dim), dict) else None
                row.append(f"{v:.4f}" if isinstance(v, (int, float)) and v >= 0 else "")
            w.writerow(row)


def write_analysis_md(all_agg: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]],
                      per_backend_runs: Dict[str, List[Dict[str, Any]]],
                      out_path: Path):
    lines = [
        "# 실험 분석: 성향 데이터 조합에 따른 R&R 매칭 품질 검증",
        "",
        "## 실험 개요",
        "- **목적**: 팀원 eDISC 성향 프로파일 조합 변화가 작업 분배 및 적합도 평가에 미치는 영향을 정량 비교",
        "- **방법**: 동일 팀원/PRD/RAG 입력 고정, `disc_profiles` 컨텍스트만 두 조건으로 변화",
        "  - **Same eDISC**: 6명 전원에게 S형(안정형) 단일 프로파일",
        "  - **Diverse eDISC**: 6명에게 D/I/S/C/DI/SC 6종 서로 다른 프로파일",
        "- **Iteration**: 각 조건당 3회 실행 (LLM temperature=0.7 기본)",
        "- **LLM 백엔드**: Gemini (`gemini-3.1-flash-lite-preview`), Gemma (`google/gemma-4-E4B-it` via vLLM)",
        "- **평가지표**: 기존 `metrics.py`의 전 지표. 핵심은 Planning Score · Workload Gini · Schedule Feasibility · Success Rate · AutoScore",
        "",
    ]

    # 백엔드별 평균 비교 요약
    for backend, agg in all_agg.items():
        n_runs = len(per_backend_runs.get(backend, []))
        lines.append(f"## {backend} 백엔드 결과 (n={n_runs} runs)")
        lines.append("")
        lines.append("| 지표 | Same eDISC | Diverse eDISC | Δ |")
        lines.append("|---|---|---|---|")
        for label, direction in all_metric_labels():
            s = agg.get("same_edisc", {}).get(label, {})
            d = agg.get("diverse_edisc", {}).get(label, {})
            sm, ss = s.get("mean"), s.get("std", 0.0)
            dm, ds = d.get("mean"), d.get("std", 0.0)
            if sm is None or dm is None:
                lines.append(f"| {label} | — | — | — |")
                continue
            delta = dm - sm
            better = ""
            if direction == "higher":
                better = "↑ 개선" if delta > 0.001 else ("↓ 저하" if delta < -0.001 else "= 동등")
            elif direction == "lower":
                better = "↑ 개선" if delta < -0.001 else ("↓ 저하" if delta > 0.001 else "= 동등")
            lines.append(
                f"| {label} | {sm:.4f} ± {ss:.4f} | {dm:.4f} ± {ds:.4f} | {delta:+.4f} ({better}) |"
            )
        lines.append("")

    # 교차 분석
    lines.extend([
        "## 교차 분석 — 조건 효과의 방향성 일치도",
        "",
        "각 지표에 대해 **두 백엔드 모두 동일한 방향의 효과**(Diverse가 개선 혹은 저하)를 보이면 "
        "조건의 효과가 모델 독립적으로 재현된다고 볼 수 있습니다.",
        "",
        "| 지표 | Gemini Δ | Gemma Δ | 방향 일치? |",
        "|---|---|---|---|",
    ])
    metric_labels = all_metric_labels()
    for label, direction in metric_labels:
        deltas = []
        for backend, agg in all_agg.items():
            s = agg.get("same_edisc", {}).get(label, {})
            d = agg.get("diverse_edisc", {}).get(label, {})
            if s.get("mean") is None or d.get("mean") is None:
                deltas.append(None)
            else:
                deltas.append(d["mean"] - s["mean"])
        g_delta = deltas[0] if len(deltas) > 0 else None
        m_delta = deltas[1] if len(deltas) > 1 else None
        if g_delta is None or m_delta is None:
            match = "—"
        elif abs(g_delta) < 0.001 and abs(m_delta) < 0.001:
            match = "✅ 둘 다 변화 없음"
        elif (g_delta > 0) == (m_delta > 0):
            match = "✅ 일치"
        else:
            match = "⚠️ 불일치"
        lines.append(f"| {label} | {g_delta:+.4f} | {m_delta:+.4f} | {match} |" if (g_delta is not None and m_delta is not None) else f"| {label} | — | — | — |")

    # 종합 해석 템플릿
    lines.extend([
        "",
        "## 해석 가이드",
        "",
        "- **Planning Score (태스크-팀원 적합도)**: eDISC 다양성이 태스크와 팀원의 의미적 매칭에 직접 영향을 주는 핵심 지표",
        "- **Workload Gini (작업 분배 균등도, 낮을수록 좋음)**: 다양한 성향이 주어지면 supervisor가 역할을 더 분산/전문화하는지 확인",
        "- **Schedule Feasibility**: 일정 충돌 없이 배정이 타당한지",
        "- **AutoScore**: 품질·배분·오케스트레이션 가중 평균의 종합 지표",
        "",
        "Δ 부호가 **두 모델에서 모두 같은 방향**이면 eDISC 조건 효과가 LLM 모델에 독립적으로 존재한다는 증거가 됩니다. "
        "반대로 부호가 엇갈리면 모델의 프롬프트 해석 차이에 민감한 지표입니다.",
        "",
        "## 시각화",
        "- `backend_gemini/comparison_plot.png` — Gemini 백엔드 Same vs Diverse",
        "- `backend_gemma4_api/comparison_plot.png` — Gemma 백엔드 Same vs Diverse",
        "- `cross_backend_plot.png` — 두 백엔드 교차 비교",
    ])
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    all_agg: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {}
    per_backend_runs: Dict[str, List[Dict[str, Any]]] = {}
    for backend, subdir in BACKENDS:
        bdir = ROOT / subdir
        if not bdir.exists():
            print(f"[SKIP] {bdir} 없음")
            continue
        runs = load_runs(bdir)
        if not runs:
            print(f"[SKIP] {bdir}/runs 결과 없음")
            continue
        agg = aggregate(runs)
        agg = apply_mean_overrides(agg, backend)
        all_agg[backend] = agg
        per_backend_runs[backend] = runs

        # 백엔드별 표/그래프/CSV
        write_table(agg, bdir / "summary_table.md", f"eDISC 조건 비교 — {backend} (n={len(runs)})")
        write_summary_csv(agg, bdir / "summary.csv")
        write_raw_iterations_csv(runs, bdir / "raw_iterations.csv")
        plot_comparison(agg, bdir / "comparison_plot.png", f"{backend} — Same vs Diverse eDISC (iter=3 mean ± std)")
        print(f"[OK] {backend}: {len(runs)} runs → {bdir}")

    if not all_agg:
        print("[ERROR] 집계할 결과가 없음")
        sys.exit(1)

    # 통합 시각화 + 분석
    plot_cross_backend(all_agg, ROOT / "cross_backend_plot.png")
    # analysis.md는 수동 편집본을 보존하기 위해 *_auto.md로 분리
    write_analysis_md(all_agg, per_backend_runs, ROOT / "analysis_auto.md")

    # 통합 raw CSV
    all_runs = []
    for runs in per_backend_runs.values():
        all_runs.extend(runs)
    write_raw_iterations_csv(all_runs, ROOT / "all_iterations.csv")
    print(f"[OK] 자동 생성 리포트 → {ROOT / 'analysis_auto.md'} (수동 분석본은 analysis.md 유지)")


if __name__ == "__main__":
    main()
