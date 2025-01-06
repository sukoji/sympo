"""
팀원 실험 결과 CSV 병합 도구
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
여러 팀원이 각자 실행한 `summary_{backend}_{runner}_{timestamp}.csv` 파일을
하나로 합치고, runner_id/condition 단위 요약을 출력한다.

사용법:
  # 디렉터리 전체 자동 병합
  python eval/merge_results.py

  # glob 패턴 직접 지정
  python eval/merge_results.py "eval_results/summary_gemini_*.csv"

  # 출력 파일명 지정
  python eval/merge_results.py -o team_merged.csv eval_results/summary_*.csv
"""
import argparse
import csv
import glob
import os
import sys
from collections import defaultdict
from datetime import datetime


DEFAULT_RESULT_DIR = os.path.join(os.path.dirname(__file__), "..", "eval_results")


def _load_rows(path: str):
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    basename = os.path.basename(path)
    for r in rows:
        if not r.get("runner_id"):
            # summary_{backend}_{runner_id}_{timestamp}.csv 파싱
            stem = basename.replace("summary_", "").replace(".csv", "")
            parts = stem.split("_")
            r["runner_id"] = parts[1] if len(parts) >= 2 else "unknown"
        r["_source_file"] = basename
    return rows


def merge_csvs(paths, output_path: str) -> dict:
    all_rows = []
    for p in paths:
        if not os.path.isfile(p):
            print(f"⚠️  skip (없음): {p}")
            continue
        rows = _load_rows(p)
        all_rows.extend(rows)
        print(f"  + {os.path.basename(p)}: {len(rows)}행")

    if not all_rows:
        print("❌ 병합할 행이 없습니다.")
        return {}

    # 모든 컬럼 union
    all_keys = set()
    for r in all_rows:
        all_keys.update(r.keys())
    fieldnames = sorted(all_keys)
    # runner_id, condition, run_id 앞에 고정
    lead = ["runner_id", "condition", "label", "run_id", "backend", "_source_file"]
    fieldnames = [k for k in lead if k in all_keys] + [
        k for k in fieldnames if k not in lead
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    # 요약 통계
    per_runner = defaultdict(int)
    per_cond = defaultdict(int)
    per_runner_cond = defaultdict(int)
    for r in all_rows:
        runner = r.get("runner_id", "unknown")
        cond = r.get("condition", "unknown")
        per_runner[runner] += 1
        per_cond[cond] += 1
        per_runner_cond[(runner, cond)] += 1

    return {
        "total_rows": len(all_rows),
        "runners": dict(per_runner),
        "conditions": dict(per_cond),
        "per_runner_cond": dict(per_runner_cond),
        "output": output_path,
    }


def _print_summary(summary: dict):
    print(f"\n📊 병합 결과")
    print(f"  총 실행: {summary['total_rows']}회")
    print(f"  팀원 수: {len(summary['runners'])}")
    print(f"\n  팀원별 실행 수:")
    for runner, n in sorted(summary["runners"].items()):
        print(f"    {runner:<15} {n}회")
    print(f"\n  조건별 실행 수:")
    for cond, n in sorted(summary["conditions"].items()):
        print(f"    {cond:<20} {n}회")
    print(f"\n  조건 × 팀원 매트릭스:")
    runners = sorted(summary["runners"].keys())
    conditions = sorted(summary["conditions"].keys())
    header = "    " + "condition".ljust(22) + "".join(r.ljust(12) for r in runners)
    print(header)
    for cond in conditions:
        line = "    " + cond.ljust(22)
        for runner in runners:
            line += str(summary["per_runner_cond"].get((runner, cond), 0)).ljust(12)
        print(line)

    print(f"\n💾 저장: {summary['output']}")
    print(f"\n다음 단계:")
    print(f"  python eval/analyze_results.py {summary['output']}")


def main():
    parser = argparse.ArgumentParser(description="팀 실험 CSV 병합기")
    parser.add_argument(
        "paths",
        nargs="*",
        help="병합할 CSV 경로 또는 glob 패턴. 비워두면 eval_results/summary_*.csv 전체.",
    )
    parser.add_argument(
        "-o", "--output", default=None, help="출력 CSV 경로 (기본: eval_results/merged_<timestamp>.csv)"
    )
    args = parser.parse_args()

    if not args.paths:
        pattern = os.path.join(DEFAULT_RESULT_DIR, "summary_*.csv")
        paths = sorted(glob.glob(pattern))
        if not paths:
            print(f"❌ {pattern} 에 해당하는 파일이 없습니다.")
            sys.exit(1)
    else:
        paths = []
        for a in args.paths:
            matches = sorted(glob.glob(a))
            paths.extend(matches if matches else [a])

    if args.output:
        output_path = args.output
    else:
        os.makedirs(DEFAULT_RESULT_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(DEFAULT_RESULT_DIR, f"merged_{ts}.csv")

    print(f"🔀 병합 대상: {len(paths)}개 파일")
    summary = merge_csvs(paths, output_path)
    if summary:
        _print_summary(summary)


if __name__ == "__main__":
    main()
