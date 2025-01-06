"""
신뢰성 분석 (rev.2, eval2 §9 스펙)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
반복 실행 결과의 신뢰성을 정량화합니다:
  - ICC(2,k): 조건 내 반복 측정의 일치도 (LLM 비결정성 분리용)
  - Cohen's κ: 1차·2차 judge 범주 일치도
  - Spearman ρ: 1차·2차 judge 순위 일치도
  - Bootstrap 95% CI: 평균의 신뢰구간

사용법:
  python eval/reliability.py eval_results/summary_gemini_*.csv
"""
import argparse
import csv
import math
import os
import random
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ──────────────────────────────────────────────────────────
# 데이터 로드
# ──────────────────────────────────────────────────────────
def load_csv(path: str) -> List[Dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in list(row.items()):
                try:
                    if "." in str(v) and v not in ("", "True", "False"):
                        row[k] = float(v)
                    elif str(v).lstrip("-").isdigit():
                        row[k] = int(v)
                except (ValueError, AttributeError):
                    pass
            rows.append(row)
    return rows


def _numeric(v, na_sentinel=-1):
    try:
        f = float(v)
        return None if f == na_sentinel else f
    except (ValueError, TypeError):
        return None


# ──────────────────────────────────────────────────────────
# ICC(2,k) — Two-way random, average measures
# ──────────────────────────────────────────────────────────
def icc_2k(matrix: List[List[float]]) -> Dict[str, float]:
    """
    matrix: rows=targets(조건/런), cols=raters(반복 또는 judge)
    ICC(2,k) — two-way random, absolute agreement, average measures.
    반환: {icc, ci_low, ci_high, n, k}
    """
    n = len(matrix)
    if n < 2:
        return {"icc": 0.0, "n": n, "k": 0, "note": "insufficient rows"}
    k = len(matrix[0])
    if k < 2 or any(len(r) != k for r in matrix):
        return {"icc": 0.0, "n": n, "k": k, "note": "inconsistent columns"}

    grand = sum(sum(r) for r in matrix) / (n * k)
    row_means = [sum(r) / k for r in matrix]
    col_means = [sum(matrix[i][j] for i in range(n)) / n for j in range(k)]

    SST = sum((matrix[i][j] - grand) ** 2 for i in range(n) for j in range(k))
    SSR = k * sum((rm - grand) ** 2 for rm in row_means)
    SSC = n * sum((cm - grand) ** 2 for cm in col_means)
    SSE = SST - SSR - SSC

    MSR = SSR / (n - 1)
    MSE = SSE / ((n - 1) * (k - 1)) if (n - 1) * (k - 1) > 0 else 0
    MSC = SSC / (k - 1) if k > 1 else 0

    denom = MSR + (MSC - MSE) / n
    icc = (MSR - MSE) / denom if denom > 0 else 0.0
    return {"icc": round(icc, 4), "n": n, "k": k, "MSR": MSR, "MSE": MSE}


# ──────────────────────────────────────────────────────────
# Cohen's κ (범주형)
# ──────────────────────────────────────────────────────────
def cohens_kappa(pairs: List[Tuple[str, str]]) -> Dict[str, float]:
    """pairs: [(rater_a_label, rater_b_label), ...]"""
    if not pairs:
        return {"kappa": 0.0, "n": 0}
    n = len(pairs)
    labels = sorted({lbl for pair in pairs for lbl in pair})
    obs = sum(1 for a, b in pairs if a == b) / n
    pa = {lbl: sum(1 for a, _ in pairs if a == lbl) / n for lbl in labels}
    pb = {lbl: sum(1 for _, b in pairs if b == lbl) / n for lbl in labels}
    exp = sum(pa[l] * pb[l] for l in labels)
    kappa = (obs - exp) / (1 - exp) if exp < 1 else 0.0
    return {"kappa": round(kappa, 4), "observed_agreement": round(obs, 4), "n": n}


# ──────────────────────────────────────────────────────────
# Spearman ρ (순위 상관)
# ──────────────────────────────────────────────────────────
def spearman_rho(a: List[float], b: List[float]) -> Dict[str, float]:
    if len(a) != len(b) or len(a) < 3:
        return {"rho": 0.0, "n": len(a)}
    try:
        from scipy.stats import spearmanr
        rho, p = spearmanr(a, b)
        return {"rho": round(float(rho), 4), "p": round(float(p), 4), "n": len(a)}
    except ImportError:
        pass
    # Fallback: rank 변환 후 Pearson
    def rank(xs):
        idx = sorted(range(len(xs)), key=lambda i: xs[i])
        r = [0] * len(xs)
        for pos, i in enumerate(idx):
            r[i] = pos + 1
        return r
    ra, rb = rank(a), rank(b)
    ma = sum(ra) / len(ra)
    mb = sum(rb) / len(rb)
    cov = sum((ra[i] - ma) * (rb[i] - mb) for i in range(len(ra)))
    va = math.sqrt(sum((x - ma) ** 2 for x in ra))
    vb = math.sqrt(sum((x - mb) ** 2 for x in rb))
    rho = cov / (va * vb) if va * vb > 0 else 0
    return {"rho": round(rho, 4), "n": len(a)}


# ──────────────────────────────────────────────────────────
# Bootstrap CI
# ──────────────────────────────────────────────────────────
def bootstrap_ci(values: List[float], n_boot: int = 1000, ci: float = 0.95,
                 seed: int = 42) -> Dict[str, float]:
    values = [v for v in values if v is not None]
    if len(values) < 2:
        return {"mean": values[0] if values else 0, "ci_low": 0, "ci_high": 0, "n": len(values)}
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        sample = [rng.choice(values) for _ in range(len(values))]
        means.append(sum(sample) / len(sample))
    means.sort()
    lo_idx = int(n_boot * (1 - ci) / 2)
    hi_idx = int(n_boot * (1 + ci) / 2) - 1
    return {
        "mean": round(sum(values) / len(values), 4),
        "ci_low": round(means[lo_idx], 4),
        "ci_high": round(means[hi_idx], 4),
        "n": len(values),
    }


# ──────────────────────────────────────────────────────────
# CLI: 전체 신뢰성 리포트
# ──────────────────────────────────────────────────────────
def analyze_reliability(csv_path: str) -> Dict:
    rows = load_csv(csv_path)
    groups = defaultdict(list)
    for r in rows:
        groups[r.get("condition", "?")].append(r)

    report = {"source": os.path.basename(csv_path), "conditions": {}}

    focus = ["success_rate", "mece_score", "buffer_ratio_pct",
             "workload_gini", "schedule_feasibility"]

    for cond, runs in groups.items():
        cond_block = {"n_runs": len(runs), "metrics": {}}
        for mkey in focus:
            vals = [_numeric(r.get(mkey)) for r in runs]
            vals = [v for v in vals if v is not None]
            if len(vals) >= 2:
                cond_block["metrics"][mkey] = bootstrap_ci(vals)
        report["conditions"][cond] = cond_block

    # Cross-judge 일치도 (judge_overall vs judge2_overall)
    pairs_primary = []
    pairs_cross = []
    for r in rows:
        p1 = _numeric(r.get("judge_overall"))
        p2 = _numeric(r.get("judge2_overall"))
        if p1 is not None and p2 is not None:
            pairs_primary.append(p1)
            pairs_cross.append(p2)
    if len(pairs_primary) >= 3:
        report["cross_judge"] = {
            "spearman": spearman_rho(pairs_primary, pairs_cross),
            "mean_abs_diff": round(
                sum(abs(a - b) for a, b in zip(pairs_primary, pairs_cross)) / len(pairs_primary), 4
            ),
            "n": len(pairs_primary),
        }
    else:
        report["cross_judge"] = {"note": "교차심사 데이터 부족 (judge_overall / judge2_overall 양쪽 필요)"}

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="symPO 신뢰성 분석")
    parser.add_argument("csv", nargs="?", default=None, help="summary CSV 경로")
    args = parser.parse_args()

    if args.csv is None:
        d = os.path.join(os.path.dirname(__file__), "..", "eval_results")
        files = sorted([f for f in os.listdir(d) if f.startswith("summary_") and f.endswith(".csv")])
        if not files:
            print("사용법: python eval/reliability.py <summary_csv_path>")
            sys.exit(1)
        args.csv = os.path.join(d, files[-1])
        print(f"최근 파일 사용: {args.csv}")

    report = analyze_reliability(args.csv)
    import json
    print(json.dumps(report, ensure_ascii=False, indent=2))

    out = os.path.join(os.path.dirname(args.csv), "reliability_report.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n📊 저장: {out}")
