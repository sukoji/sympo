"""
감도 분석 (rev.2, eval2 §4 스펙)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
측정 하이퍼파라미터에 대한 결과의 민감도를 검증:
  - τ (Success Rate 유사도 임계): 0.40, 0.45, 0.50
  - θ (Planning Score 유사도 임계): 0.50, 0.60
  - AutoScore 가중치 ±0.10 grid search

목적: 특정 임계/가중치 선택에 따라 결론이 바뀌는지 검증.
      rev.2 metrics.py에 저장된 WBS 스냅샷 기반 사후 재계산.

사용법:
  python eval/sensitivity.py eval_results/wbs_snapshot_*.json
  python eval/sensitivity.py --csv eval_results/summary_gemini_*.csv
"""
import argparse
import csv
import itertools
import json
import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ──────────────────────────────────────────────────────────
# 1. Success Rate τ 감도 (단일 WBS 스냅샷 기반)
# ──────────────────────────────────────────────────────────
def sr_tau_sensitivity(snapshot_path: str, taus: List[float] = None) -> Dict:
    """
    저장된 WBS 스냅샷에서 PRD 기능 커버리지 τ를 다르게 적용했을 때
    Success Rate가 어떻게 변하는지 측정.
    """
    from metrics import calc_success_rate
    from schemas.prd_schema import PRDInput
    from schemas.wbs_schema import WBSTask, TaskLevel

    if taus is None:
        taus = [0.40, 0.45, 0.50]

    with open(snapshot_path, encoding="utf-8") as f:
        snap = json.load(f)

    # 스냅샷의 WBS를 WBSTask로 복원 (간이: dict 그대로 전달)
    tasks = snap.get("wbs_tasks", [])

    # PRD features는 스냅샷에 없을 수 있음 → sample_data에서 로드 시도
    try:
        from data_pipeline.prd_parser import PRDParser
        prd_path = os.path.join(os.path.dirname(__file__), "..", "sample_data", "sample_prd.txt")
        if os.path.exists(prd_path):
            text = open(prd_path, encoding="utf-8").read()
            prd = PRDParser.from_text(text, "Sample")
            features = prd.key_features
        else:
            features = []
    except Exception:
        features = []

    results = {}
    for tau in taus:
        # calc_success_rate의 임계가 하드코딩된 경우 monkey patch 필요
        # 여기서는 임계 파라미터 지원을 가정 (없으면 단순 표시)
        try:
            sr = calc_success_rate(features, tasks, threshold=tau)
            results[f"tau={tau}"] = sr
        except TypeError:
            # 하위 호환: threshold 파라미터 미지원 시
            sr = calc_success_rate(features, tasks)
            results[f"tau={tau}_default"] = sr
            break

    return {"snapshot": os.path.basename(snapshot_path),
            "taus_evaluated": taus, "results": results}


# ──────────────────────────────────────────────────────────
# 2. AutoScore 가중치 grid search (CSV 기반)
# ──────────────────────────────────────────────────────────
# eval2 §4: 생성 품질/배분/오케스트레이션 가중 평균 (합=1.0)
# AutoScore 로직은 eval_results.autoscore_recompute.recompute_autoscore()가
# canonical이다. 런타임과 과거 CSV 재산출이 같은 함수를 보도록 유지한다.
DEFAULT_WEIGHTS = {"quality": 0.45, "allocation": 0.35, "orchestration": 0.20}

def autoscore(row: Dict, weights: Dict[str, float]) -> float:
    from eval_results.autoscore_recompute import recompute_autoscore

    return recompute_autoscore(row, top_level_weights=weights).get("autoscore", 0.0)


def weight_grid_search(csv_path: str, step: float = 0.10) -> Dict:
    """
    AutoScore 가중치 ±step grid에서 조건별 랭킹이 얼마나 흔들리는지 측정.
    """
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)

    if not rows:
        return {"error": "empty CSV"}

    # 조건별 그룹
    groups = {}
    for r in rows:
        groups.setdefault(r.get("condition", "?"), []).append(r)

    # 가중치 grid: quality ∈ {0.35, 0.45, 0.55}, allocation ∈ {0.25, 0.35, 0.45}
    grid = []
    for q in [0.35, 0.45, 0.55]:
        for a in [0.25, 0.35, 0.45]:
            o = 1.0 - q - a
            if 0.0 <= o <= 1.0:
                grid.append({"quality": q, "allocation": a, "orchestration": round(o, 2)})

    # 각 grid 포인트에서 조건별 평균 AutoScore → 랭킹
    rankings = {}
    for w in grid:
        scores = {}
        for cond, grp in groups.items():
            vals = [autoscore(r, w) for r in grp]
            scores[cond] = sum(vals) / len(vals) if vals else 0.0
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        key = f"q{w['quality']:.2f}_a{w['allocation']:.2f}_o{w['orchestration']:.2f}"
        rankings[key] = [
            {"rank": i + 1, "condition": c, "score": round(s, 4)}
            for i, (c, s) in enumerate(ranked)
        ]

    # Top-1 안정성
    top1_counts = {}
    for r in rankings.values():
        if r:
            top1_counts[r[0]["condition"]] = top1_counts.get(r[0]["condition"], 0) + 1

    return {
        "source": os.path.basename(csv_path),
        "grid_points": len(grid),
        "top1_stability": top1_counts,
        "rankings_by_weight": rankings,
    }


# ──────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="symPO 감도 분석")
    parser.add_argument("target", nargs="?", default=None,
                        help="WBS snapshot JSON (τ 감도) 또는 summary CSV (가중치 grid)")
    parser.add_argument("--csv", default=None, help="summary CSV 명시 지정")
    args = parser.parse_args()

    target = args.csv or args.target
    if target is None:
        d = os.path.join(os.path.dirname(__file__), "..", "eval_results")
        files = sorted([f for f in os.listdir(d) if f.startswith("summary_") and f.endswith(".csv")])
        if not files:
            print("사용법: python eval/sensitivity.py <csv_or_snapshot>")
            sys.exit(1)
        target = os.path.join(d, files[-1])
        print(f"최근 파일 사용: {target}")

    if target.endswith(".csv"):
        report = weight_grid_search(target)
        label = "weight_grid"
    elif target.endswith(".json"):
        report = sr_tau_sensitivity(target)
        label = "tau_sensitivity"
    else:
        print(f"지원 형식: .csv (가중치 grid) 또는 .json (τ 감도). 받은 값: {target}")
        sys.exit(2)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    out = os.path.join(os.path.dirname(target), f"sensitivity_{label}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n📊 저장: {out}")
