"""
실험 결과 분석 + 마크다운 리포트 생성
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
사용법:
  # 단일 CSV
  python eval/analyze_results.py eval_results/summary_mock_anon_20260101_120000.csv

  # 여러 팀원 CSV 합쳐서 분석 (glob 패턴 지원)
  python eval/analyze_results.py "eval_results/summary_gemini_*.csv"
  python eval/analyze_results.py eval_results/summary_a.csv eval_results/summary_b.csv
"""
import csv
import glob
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Any

# 분석 대상 지표 정의
METRICS = [
    ("faithfulness",        "Faithfulness (RAG 충실도, NLI)", "higher", "생성 품질"),
    ("success_rate",        "Success Rate (PRD 커버리지)",   "higher", "생성 품질"),
    ("mece_score",          "MECE Score (Minto 1987)",       "higher", "생성 품질"),
    ("granularity_fitness", "Granularity (PMBOK 7판)",       "higher", "생성 품질"),
    ("planning_score",      "Planning Score (배분 적합도)",  "higher", "배분 품질"),
    ("workload_gini",       "Workload Gini (업무 균등성)",   "lower",  "배분 품질"),
    ("schedule_feasibility","Schedule Feasibility (일정 현실성)", "higher", "배분 품질"),
    ("buffer_ratio_pct",    "Buffer Ratio (%)",              "range",  "배분 품질"),
    ("interaction_turns",   "Interaction Turns",             "info",   "오케스트레이션"),
    ("supervisor_ratio",    "Supervisor 개입율",              "lower",  "오케스트레이션"),
    ("comm_efficiency",     "Communication Efficiency",      "higher", "오케스트레이션"),
    ("est_tokens",          "Est. Token Cost",               "lower",  "비용"),
    ("est_cost_usd",        "Est. Cost (USD)",               "lower",  "비용"),
]

# N/A 센티넬값 — metrics.py에서 RAG 없음 조건 등에 -1 할당. 통계에서 제외.
NA_SENTINEL = -1


def load_csv(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 숫자 변환
            for k, v in row.items():
                try:
                    if "." in str(v):
                        row[k] = float(v)
                    elif v.isdigit():
                        row[k] = int(v)
                    elif v in ("True", "False"):
                        row[k] = v == "True"
                except (ValueError, AttributeError):
                    pass
            rows.append(row)
    return rows


def group_by_condition(rows: List[dict]) -> Dict[str, List[dict]]:
    groups = defaultdict(list)
    for r in rows:
        groups[r.get("condition", "unknown")].append(r)
    return dict(groups)


def _to_float_or_none(v):
    """값을 float로 변환. 변환 불가 시 None."""
    if v is None:
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def calc_stats(values: list, drop_na: bool = True) -> dict:
    """평균/표준편차/최대/최소 + N/A(-1) 제외.

    drop_na: True면 NA_SENTINEL(-1) 값을 통계에서 제외 (기본).
             faithfulness 같이 조건별 N/A 섞인 필드의 평균 왜곡 방지.
    """
    cleaned = []
    n_na = 0
    for v in values:
        f = _to_float_or_none(v)
        if f is None:
            continue
        if drop_na and f == NA_SENTINEL:
            n_na += 1
            continue
        cleaned.append(f)
    if not cleaned:
        return {"mean": 0, "std": 0, "min": 0, "max": 0, "n": 0, "n_na": n_na}
    n = len(cleaned)
    mean = sum(cleaned) / n
    std = (sum((v - mean) ** 2 for v in cleaned) / n) ** 0.5
    return {
        "mean": round(mean, 4),
        "std": round(std, 4),
        "min": round(min(cleaned), 4),
        "max": round(max(cleaned), 4),
        "n": n,
        "n_na": n_na,
    }


# ── rev.2: 통계 검정 (Mann-Whitney U + Cliff's δ + Holm-Bonferroni) ──

def mann_whitney_u(a: list, b: list) -> Dict[str, float]:
    """양측 Mann-Whitney U test. scipy 없으면 수동 계산 (근사 정규)."""
    a = [float(x) for x in a if _to_float_or_none(x) not in (None, float(NA_SENTINEL))]
    b = [float(x) for x in b if _to_float_or_none(x) not in (None, float(NA_SENTINEL))]
    if len(a) < 2 or len(b) < 2:
        return {"U": 0.0, "p": 1.0, "n_a": len(a), "n_b": len(b)}
    try:
        from scipy.stats import mannwhitneyu
        res = mannwhitneyu(a, b, alternative="two-sided")
        return {"U": float(res.statistic), "p": float(res.pvalue),
                "n_a": len(a), "n_b": len(b)}
    except ImportError:
        # Fallback: normal approximation
        import math
        combined = [(v, 0) for v in a] + [(v, 1) for v in b]
        combined.sort(key=lambda x: x[0])
        ranks = {}
        for i, (v, _) in enumerate(combined):
            ranks.setdefault(v, []).append(i + 1)
        rank_a = sum(sum(ranks[v]) / len(ranks[v]) for v in a)
        n1, n2 = len(a), len(b)
        u1 = rank_a - n1 * (n1 + 1) / 2
        u = min(u1, n1 * n2 - u1)
        mu = n1 * n2 / 2
        sigma = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
        z = (u - mu) / sigma if sigma > 0 else 0
        p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
        return {"U": float(u), "p": float(p), "n_a": n1, "n_b": n2}


def cliffs_delta(a: list, b: list) -> Dict[str, float]:
    """Cliff's δ 효과 크기. |δ|<0.147 negligible, <0.33 small, <0.474 medium, else large."""
    a = [float(x) for x in a if _to_float_or_none(x) not in (None, float(NA_SENTINEL))]
    b = [float(x) for x in b if _to_float_or_none(x) not in (None, float(NA_SENTINEL))]
    if not a or not b:
        return {"delta": 0.0, "magnitude": "n/a"}
    gt = sum(1 for x in a for y in b if x > y)
    lt = sum(1 for x in a for y in b if x < y)
    delta = (gt - lt) / (len(a) * len(b))
    ad = abs(delta)
    mag = ("negligible" if ad < 0.147
           else "small" if ad < 0.33
           else "medium" if ad < 0.474
           else "large")
    return {"delta": round(delta, 4), "magnitude": mag}


def holm_bonferroni(pvalues: List[float], alpha: float = 0.05) -> List[bool]:
    """Holm-Bonferroni step-down. 반환: 각 p가 유의(True/False) 리스트."""
    n = len(pvalues)
    indexed = sorted(enumerate(pvalues), key=lambda x: x[1])
    reject = [False] * n
    for rank, (orig_idx, p) in enumerate(indexed):
        threshold = alpha / (n - rank)
        if p <= threshold:
            reject[orig_idx] = True
        else:
            break
    return reject


def load_csvs(paths: List[str]) -> List[Dict[str, Any]]:
    """여러 CSV 파일을 합쳐 한번에 로드 (팀 실험 결과 merge용)."""
    all_rows = []
    for p in paths:
        if not os.path.isfile(p):
            print(f"⚠️  파일 없음: {p}")
            continue
        rows = load_csv(p)
        # runner_id 컬럼 없으면 파일명에서 추정
        basename = os.path.basename(p)
        for r in rows:
            if not r.get("runner_id"):
                parts = basename.replace("summary_", "").replace(".csv", "").split("_")
                r["runner_id"] = parts[1] if len(parts) >= 2 else "unknown"
        all_rows.extend(rows)
    return all_rows


def generate_report(csv_path, rows: List[dict] = None) -> str:
    """마크다운 분석 리포트 생성.

    csv_path: str(단일) | List[str](멀티)
    rows: 외부에서 미리 로드한 rows를 재사용할 때 전달
    """
    if rows is None:
        if isinstance(csv_path, list):
            rows = load_csvs(csv_path)
        else:
            rows = load_csv(csv_path)
    groups = group_by_condition(rows)

    cond_order = ["C0_llm_only", "C1_with_assign", "C2_1round", "C3_3rounds", "C4_with_disc"]
    cond_labels = {
        "C0_llm_only": "C0: LLM 단독",
        "C1_with_assign": "C1: +배정",
        "C2_1round": "C2: +1R 토론",
        "C3_3rounds": "C3: +3R 토론",
        "C4_with_disc": "C4: +eDISC",
    }

    lines = []
    lines.append("# symPO Ablation Study 분석 리포트")
    lines.append(f"\n생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if isinstance(csv_path, list):
        lines.append(f"\n데이터: {len(csv_path)}개 CSV 병합")
        for p in csv_path:
            lines.append(f"  - `{os.path.basename(p)}`")
    else:
        lines.append(f"\n데이터: `{os.path.basename(csv_path)}`")
    lines.append(f"\n총 실행 횟수: {len(rows)}회 ({len(groups)}개 조건)")
    # 팀원 runner_id 분포 (멀티 소스일 때만)
    runners = sorted({str(r.get("runner_id", "")) for r in rows if r.get("runner_id")})
    if len(runners) > 1:
        lines.append(f"\n참여 팀원: {', '.join(runners)} ({len(runners)}명)")

    # ── 실험 설정 요약 ──
    backend = rows[0].get("backend", "unknown") if rows else "unknown"
    lines.append(f"\nLLM 백엔드: **{backend}**\n")

    lines.append("## 1. 조건별 요약 (Mean ± Std)")
    lines.append("")

    # 카테고리별 테이블
    categories = ["생성 품질", "배분 품질", "오케스트레이션", "비용"]
    for cat in categories:
        cat_metrics = [(key, label, direction, c) for key, label, direction, c in METRICS if c == cat]
        if not cat_metrics:
            continue

        lines.append(f"### {cat}")
        # 테이블 헤더
        header = "| 지표 |"
        sep = "|------|"
        for cond in cond_order:
            if cond in groups:
                label = cond_labels.get(cond, cond)
                header += f" {label} |"
                sep += "------|"
        lines.append(header)
        lines.append(sep)

        for key, label, direction, _ in cat_metrics:
            row_str = f"| {label} |"
            best_val = None
            best_cond = None

            # 조건별 통계 계산
            cond_stats = {}
            for cond in cond_order:
                if cond not in groups:
                    continue
                vals = [r.get(key, 0) for r in groups[cond] if r.get(key) is not None]
                vals = [v for v in vals if isinstance(v, (int, float))]
                stats = calc_stats(vals)
                cond_stats[cond] = stats

                # 최적 조건 찾기
                if direction == "higher" and (best_val is None or stats["mean"] > best_val):
                    best_val = stats["mean"]
                    best_cond = cond
                elif direction == "lower" and (best_val is None or stats["mean"] < best_val):
                    best_val = stats["mean"]
                    best_cond = cond

            for cond in cond_order:
                if cond not in groups:
                    continue
                stats = cond_stats.get(cond, {"mean": 0, "std": 0})
                val_str = f"{stats['mean']:.3f}±{stats['std']:.3f}"
                if cond == best_cond and direction in ("higher", "lower"):
                    val_str = f"**{val_str}**"
                row_str += f" {val_str} |"
            lines.append(row_str)
        lines.append("")

    # ── 핵심 발견사항 ──
    lines.append("## 2. 핵심 발견사항 (Key Findings)")
    lines.append("")

    # C0 vs C3 비교 (토론 효과)
    if "C0_llm_only" in groups and "C3_3rounds" in groups:
        c0_sr = calc_stats([r.get("success_rate", 0) for r in groups["C0_llm_only"]])["mean"]
        c3_sr = calc_stats([r.get("success_rate", 0) for r in groups["C3_3rounds"]])["mean"]
        delta_sr = c3_sr - c0_sr

        c0_gini = calc_stats([r.get("workload_gini", 0) for r in groups["C0_llm_only"]])["mean"]
        c3_gini = calc_stats([r.get("workload_gini", 0) for r in groups["C3_3rounds"]])["mean"]

        c0_buf = calc_stats([r.get("buffer_ratio_pct", 0) for r in groups["C0_llm_only"]])["mean"]
        c3_buf = calc_stats([r.get("buffer_ratio_pct", 0) for r in groups["C3_3rounds"]])["mean"]

        lines.append("### RQ1: 멀티에이전트 토론이 WBS 품질을 향상시키는가?")
        lines.append(f"- Success Rate: C0({c0_sr:.2%}) → C3({c3_sr:.2%}), 변화량 {delta_sr:+.2%}")
        lines.append(f"- Workload Gini: C0({c0_gini:.3f}) → C3({c3_gini:.3f}) {'(개선)' if c3_gini < c0_gini else '(악화)'}")
        lines.append(f"- Buffer Ratio: C0({c0_buf:.1f}%) → C3({c3_buf:.1f}%)")
        lines.append("")

    # C3 vs C4 비교 (eDISC 효과)
    if "C3_3rounds" in groups and "C4_with_disc" in groups:
        c3_ps = calc_stats([r.get("planning_score", 0) for r in groups["C3_3rounds"]])["mean"]
        c4_ps = calc_stats([r.get("planning_score", 0) for r in groups["C4_with_disc"]])["mean"]
        c3_eff = calc_stats([r.get("comm_efficiency", 0) for r in groups["C3_3rounds"]])["mean"]
        c4_eff = calc_stats([r.get("comm_efficiency", 0) for r in groups["C4_with_disc"]])["mean"]

        lines.append("### RQ3: eDISC 성향이 역할 배분에 기여하는가?")
        lines.append(f"- Planning Score: C3({c3_ps:.4f}) → C4({c4_ps:.4f}), 변화량 {c4_ps - c3_ps:+.4f}")
        lines.append(f"- Comm Efficiency: C3({c3_eff:.2%}) → C4({c4_eff:.2%})")
        lines.append("")

    # ── 비용 효율성 ──
    lines.append("### 비용 효율성")
    for cond in cond_order:
        if cond not in groups:
            continue
        tokens = calc_stats([r.get("est_tokens", 0) for r in groups[cond]])
        cost = calc_stats([r.get("est_cost_usd", 0) for r in groups[cond]])
        elapsed = calc_stats([r.get("elapsed_sec", 0) for r in groups[cond]])
        label = cond_labels.get(cond, cond)
        lines.append(f"- {label}: {tokens['mean']:.0f} tokens, ${cost['mean']:.4f}, {elapsed['mean']:.1f}s")
    lines.append("")

    # ── 통계 검정 (Mann-Whitney U + Cliff's δ + Holm-Bonferroni) ──
    lines.append("## 3. 통계 검정 (rev.2: Mann-Whitney U + Cliff's δ + Holm-Bonferroni)")
    lines.append("")
    lines.append("독립 표본 비교 (비모수, n<30 적합). Cliff's δ로 효과 크기 보완.")
    lines.append("")

    # 주요 비교 쌍 정의: (label, cond_a, cond_b)
    comparison_pairs = [
        ("C0 vs C3 (토론 효과)",        "C0_llm_only",  "C3_3rounds"),
        ("C1 vs C3 (+토론 기여)",       "C1_with_assign", "C3_3rounds"),
        ("C3 vs C4 (eDISC 효과)",       "C3_3rounds",    "C4_with_disc"),
        ("R0 vs R1 (RAG 효과)",         "R0_no_rag",     "R1_vanilla"),
    ]
    # 주요 지표
    focus_metrics = [
        ("success_rate", "Success Rate"),
        ("mece_score", "MECE"),
        ("buffer_ratio_pct", "Buffer %"),
        ("workload_gini", "Gini"),
        ("schedule_feasibility", "Feasibility"),
        ("faithfulness", "Faithfulness"),
    ]

    stat_rows = []
    raw_pvalues = []
    for pair_label, ca, cb in comparison_pairs:
        if ca not in groups or cb not in groups:
            continue
        for mkey, mlabel in focus_metrics:
            vals_a = [r.get(mkey, 0) for r in groups[ca]]
            vals_b = [r.get(mkey, 0) for r in groups[cb]]
            u = mann_whitney_u(vals_a, vals_b)
            d = cliffs_delta(vals_a, vals_b)
            if u["n_a"] < 2 or u["n_b"] < 2:
                continue
            stat_rows.append((pair_label, mlabel, u, d))
            raw_pvalues.append(u["p"])

    # Holm-Bonferroni 보정
    rejects = holm_bonferroni(raw_pvalues, alpha=0.05) if raw_pvalues else []

    if stat_rows:
        lines.append("| 비교 | 지표 | n_a | n_b | U | p (raw) | Holm 유의 | Cliff's δ | 크기 |")
        lines.append("|------|------|----:|----:|---:|--------:|:---------:|----------:|------|")
        for (pair_label, mlabel, u, d), sig in zip(stat_rows, rejects):
            mark = "**✅**" if sig else "—"
            lines.append(
                f"| {pair_label} | {mlabel} | {u['n_a']} | {u['n_b']} "
                f"| {u['U']:.1f} | {u['p']:.4f} | {mark} "
                f"| {d['delta']:.3f} | {d['magnitude']} |"
            )
    else:
        lines.append("_(비교 가능한 조건 쌍이 없습니다.)_")
    lines.append("")
    lines.append("**주의**: n<10에서는 통계적 유의성보다 **효과 크기(Cliff's δ)**가 신뢰 지표.")
    lines.append("Holm-Bonferroni는 family-wise error 제어 (α=0.05, step-down).")
    lines.append("")

    # ── 원시 데이터 ──
    lines.append("## 4. 원시 데이터")
    lines.append("")
    lines.append("| 조건 | Run | Tasks | SR | MECE | Gini | Feas. | Eff. | Tokens | Time |")
    lines.append("|------|-----|-------|----|------|------|-------|------|--------|------|")
    for r in rows:
        cond = r.get("condition", "")
        label = cond_labels.get(cond, cond)[:8]
        lines.append(
            f"| {label} | {r.get('run_id', '')} "
            f"| {r.get('total_tasks', 0)} "
            f"| {r.get('success_rate', 0):.2f} "
            f"| {r.get('mece_score', 0):.2f} "
            f"| {r.get('workload_gini', 0):.3f} "
            f"| {r.get('schedule_feasibility', 0):.2f} "
            f"| {r.get('comm_efficiency', 0):.2f} "
            f"| {r.get('est_tokens', 0):.0f} "
            f"| {r.get('elapsed_sec', 0):.1f}s |"
        )

    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # 가장 최근 summary CSV 자동 찾기
        result_dir = os.path.join(os.path.dirname(__file__), "..", "eval_results")
        csvs = sorted([f for f in os.listdir(result_dir) if f.startswith("summary_") and f.endswith(".csv")]) if os.path.isdir(result_dir) else []
        if not csvs:
            print("사용법: python eval/analyze_results.py <summary_csv_path_or_glob> [추가 CSV...]")
            sys.exit(1)
        csv_path = os.path.join(result_dir, csvs[-1])
        print(f"최근 파일 사용: {csv_path}")
    else:
        # 여러 인자 지원 + glob 확장
        raw_args = sys.argv[1:]
        expanded = []
        for a in raw_args:
            matches = sorted(glob.glob(a))
            if matches:
                expanded.extend(matches)
            else:
                expanded.append(a)
        csv_path = expanded if len(expanded) > 1 else expanded[0]
        if isinstance(csv_path, list):
            print(f"병합 분석: {len(csv_path)}개 CSV")

    report = generate_report(csv_path)

    # 리포트 저장
    report_dir = os.path.join(os.path.dirname(__file__), "..", "eval_results")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n📝 리포트 저장: {report_path}")
    print("\n" + report)
