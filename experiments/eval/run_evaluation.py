"""
Run Evaluation — GT-Free 평가 파이프라인 메인 실행기
──────────────────────────────────────────────────────────────────────────
실행:
  python eval/run_evaluation.py [--domains ecommerce fintech] [--output eval_results]

평가 흐름:
  1. Benchmark 케이스 생성 (Planted Constraints)
  2. 각 케이스에 대해:
     a. Baseline WBS 생성 (단일 LLM Zero-Shot)
     b. symPO WBS 생성 (다중 에이전트 토론)
  3. 각 WBS에 대해:
     a. Structural Integrity 검사
     b. Constraint Satisfaction Rate 측정
     c. LLM-as-Judge SxS 비교
     d. Red Team 결함 분석
  4. 종합 리포트 생성 → eval_results/eval_report.json + eval_results/eval_summary.md

출력 지표:
  - CSR (Constraint Satisfaction Rate): symPO vs Baseline
  - SI Score (Structural Integrity): symPO vs Baseline
  - SxS Win Rate: symPO 승률
  - Red Team Delta: Baseline 결함수 - symPO 결함수 (양수이면 symPO 우위)
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import List, Dict, Any

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from eval.benchmark_generator import generate_benchmark_cases, save_benchmark_cases, BenchmarkCase
from eval.structural_checker import check_structural_integrity, format_report
from eval.constraint_checker import check_constraints
from eval.baseline_runner import generate_baseline_wbs
from eval.llm_judge import run_sxs_evaluation
from eval.red_team import run_red_team


def run_sympo_wbs(case: BenchmarkCase) -> list:
    """symPO 파이프라인으로 WBS 생성 (토론 포함)"""
    try:
        from agents.state import WBSState, create_initial_state
        from orchestration.debate_loop import execute_sympo_flow
        from persona_engine.persona_builder import PersonaBuilder

        team_summary = PersonaBuilder.generate_team_summary(case.team)
        supervisor_persona = PersonaBuilder.build_supervisor_persona(
            supervisor_name="PM 에이전트", team_summary=team_summary
        )
        member_personas = PersonaBuilder.build_all_personas(case.team)
        all_personas = {"supervisor": supervisor_persona, **member_personas}

        state: WBSState = create_initial_state(
            prd=case.prd,
            team_members=case.team,
            agent_personas=all_personas,
        )
        state["max_rounds"] = 2  # 평가 목적으로 라운드 수 제한

        final_state = None
        for state in execute_sympo_flow(state, max_rounds=2):
            final_state = state

        if final_state is None:
            return []

        return (
            final_state.get("final_wbs")
            or final_state.get("current_wbs_draft")
            or []
        )
    except Exception as e:
        print(f"[symPO] WBS 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        return []


def evaluate_case(case: BenchmarkCase, output_dir: str) -> Dict[str, Any]:
    """단일 BenchmarkCase에 대해 전체 평가 수행"""
    print(f"\n{'='*60}")
    print(f"케이스: {case.case_id} | 도메인: {case.domain} | 난이도: {case.difficulty}")
    print(f"{'='*60}")

    result: Dict[str, Any] = {
        "case_id": case.case_id,
        "domain": case.domain,
        "difficulty": case.difficulty,
        "project_name": getattr(case.prd, "project_name", ""),
        "timestamp": datetime.now().isoformat(),
    }

    # ── Step 1: Baseline WBS 생성 ──────────────────────────────────────
    print("\n[1/4] Baseline WBS 생성 중...")
    t0 = time.time()
    baseline_tasks = generate_baseline_wbs(case.prd, case.team)
    result["baseline_task_count"] = len(baseline_tasks)
    result["baseline_gen_time"] = round(time.time() - t0, 1)
    print(f"  → {len(baseline_tasks)}개 태스크 ({result['baseline_gen_time']}s)")

    # ── Step 2: symPO WBS 생성 ──────────────────────────────────────
    print("\n[2/4] symPO WBS 생성 중...")
    t0 = time.time()
    sympo_tasks = run_sympo_wbs(case)
    result["sympo_task_count"] = len(sympo_tasks)
    result["sympo_gen_time"] = round(time.time() - t0, 1)
    print(f"  → {len(sympo_tasks)}개 태스크 ({result['sympo_gen_time']}s)")

    if not baseline_tasks and not sympo_tasks:
        print("  ⚠️ 두 시스템 모두 WBS 생성 실패. 케이스 스킵.")
        result["skipped"] = True
        return result

    # ── Step 3: Structural Integrity ──────────────────────────────────
    print("\n[3/4] 구조적 무결성 검사...")

    si_baseline = check_structural_integrity(baseline_tasks)
    si_sympo = check_structural_integrity(sympo_tasks)
    print(f"  Baseline SI: {si_baseline.overall_score:.1f} ({si_baseline.grade})")
    print(f"  symPO  SI: {si_sympo.overall_score:.1f} ({si_sympo.grade})")

    result["structural_integrity"] = {
        "baseline": si_baseline.to_dict(),
        "sympo": si_sympo.to_dict(),
        "si_delta": round(si_sympo.overall_score - si_baseline.overall_score, 1),
    }

    # ── Step 4a: Constraint Satisfaction Rate ──────────────────────────
    print("\n[4a/4] Constraint Satisfaction Rate...")

    csr_baseline = check_constraints(baseline_tasks, case.constraints, use_llm_fallback=True)
    csr_sympo = check_constraints(sympo_tasks, case.constraints, use_llm_fallback=True)
    print(f"  Baseline CSR: {csr_baseline.rate:.1%} (High: {csr_baseline.high_severity_rate:.1%})")
    print(f"  symPO  CSR: {csr_sympo.rate:.1%} (High: {csr_sympo.high_severity_rate:.1%})")

    result["constraint_satisfaction"] = {
        "baseline": csr_baseline.to_dict(),
        "sympo": csr_sympo.to_dict(),
        "csr_delta": round(csr_sympo.rate - csr_baseline.rate, 4),
        "high_csr_delta": round(csr_sympo.high_severity_rate - csr_baseline.high_severity_rate, 4),
    }

    # ── Step 4b: LLM-as-Judge SxS ─────────────────────────────────────
    print("\n[4b/4] LLM-as-Judge SxS 평가...")
    constraints_text = "\n".join(c.description for c in case.constraints)
    sxs = run_sxs_evaluation(
        case.prd,
        baseline_tasks,
        sympo_tasks,
        constraints_text=constraints_text,
    )
    print(f"  승자: {sxs.winner} | Baseline: {sxs.score_a:.1f} vs symPO: {sxs.score_b:.1f}")

    result["sxs_evaluation"] = sxs.to_dict()

    # ── Step 4c: Red Team ──────────────────────────────────────────────
    print("\n[4c/4] Red Team 평가...")
    rt_baseline = run_red_team(case.prd, baseline_tasks, label="Baseline")
    rt_sympo = run_red_team(case.prd, sympo_tasks, label="symPO")
    print(f"  Baseline 결함: {rt_baseline.flaw_count}건 (High: {rt_baseline.high_severity_count})")
    print(f"  symPO  결함: {rt_sympo.flaw_count}건 (High: {rt_sympo.high_severity_count})")

    result["red_team"] = {
        "baseline": rt_baseline.to_dict(),
        "sympo": rt_sympo.to_dict(),
        "flaw_delta": rt_baseline.flaw_count - rt_sympo.flaw_count,  # 양수 = symPO 우위
        "high_flaw_delta": rt_baseline.high_severity_count - rt_sympo.high_severity_count,
    }

    return result


def aggregate_results(case_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """케이스별 결과를 종합하여 요약 통계 생성"""
    valid = [r for r in case_results if not r.get("skipped")]
    n = max(len(valid), 1)

    def avg(key_path: str) -> float:
        vals = []
        for r in valid:
            obj = r
            for k in key_path.split("."):
                obj = obj.get(k, {}) if isinstance(obj, dict) else {}
            if isinstance(obj, (int, float)):
                vals.append(obj)
        return round(sum(vals) / max(len(vals), 1), 4)

    # SxS 승률
    winners = [r.get("sxs_evaluation", {}).get("winner", "TIE") for r in valid]
    sympo_wins = sum(1 for w in winners if w == "sympo")
    baseline_wins = sum(1 for w in winners if w == "baseline")
    ties = sum(1 for w in winners if w == "TIE")

    return {
        "total_cases": len(case_results),
        "valid_cases": n,
        "summary": {
            # Structural Integrity
            "si_baseline_avg": avg("structural_integrity.baseline.overall_score"),
            "si_sympo_avg": avg("structural_integrity.sympo.overall_score"),
            "si_delta_avg": avg("structural_integrity.si_delta"),
            # Constraint Satisfaction
            "csr_baseline_avg": avg("constraint_satisfaction.baseline.constraint_satisfaction_rate"),
            "csr_sympo_avg": avg("constraint_satisfaction.sympo.constraint_satisfaction_rate"),
            "csr_delta_avg": avg("constraint_satisfaction.csr_delta"),
            "high_csr_baseline_avg": avg("constraint_satisfaction.baseline.high_severity_rate"),
            "high_csr_sympo_avg": avg("constraint_satisfaction.sympo.high_severity_rate"),
            # SxS
            "sxs_sympo_win_rate": round(sympo_wins / n, 4),
            "sxs_baseline_win_rate": round(baseline_wins / n, 4),
            "sxs_tie_rate": round(ties / n, 4),
            "sxs_score_baseline_avg": avg("sxs_evaluation.score_baseline"),
            "sxs_score_sympo_avg": avg("sxs_evaluation.score_sympo"),
            "sxs_score_delta_avg": avg("sxs_evaluation.score_delta"),
            # Red Team
            "rt_baseline_flaw_avg": avg("red_team.baseline.flaw_count"),
            "rt_sympo_flaw_avg": avg("red_team.sympo.flaw_count"),
            "rt_flaw_delta_avg": avg("red_team.flaw_delta"),
            "rt_high_flaw_delta_avg": avg("red_team.high_flaw_delta"),
        },
    }


def generate_markdown_summary(aggregate: Dict[str, Any], case_results: List[Dict[str, Any]]) -> str:
    """Markdown 형식의 요약 리포트 생성"""
    s = aggregate["summary"]
    lines = [
        "# symPO GT-Free 평가 보고서",
        f"생성 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"평가 케이스: {aggregate['valid_cases']}개",
        "",
        "## 종합 결과 요약",
        "",
        "| 지표 | Baseline | symPO | Delta (↑ symPO 우위) |",
        "|------|---------|---------|----------------------|",
        f"| 구조적 무결성 점수 | {s['si_baseline_avg']:.1f} | {s['si_sympo_avg']:.1f} | {s['si_delta_avg']:+.1f} |",
        f"| 제약 충족률 (CSR) | {s['csr_baseline_avg']:.1%} | {s['csr_sympo_avg']:.1%} | {s['csr_delta_avg']:+.1%} |",
        f"| CSR (High 중요도) | {s['high_csr_baseline_avg']:.1%} | {s['high_csr_sympo_avg']:.1%} | {s['high_csr_sympo_avg']-s['high_csr_baseline_avg']:+.1%} |",
        f"| SxS Judge 점수 | {s['sxs_score_baseline_avg']:.1f} | {s['sxs_score_sympo_avg']:.1f} | {s['sxs_score_delta_avg']:+.1f} |",
        f"| Red Team 결함 수 | {s['rt_baseline_flaw_avg']:.1f} | {s['rt_sympo_flaw_avg']:.1f} | {s['rt_flaw_delta_avg']:+.1f} ↓ |",
        "",
        "## SxS 판정 결과",
        f"- symPO 승: **{s['sxs_sympo_win_rate']:.0%}**",
        f"- Baseline 승: {s['sxs_baseline_win_rate']:.0%}",
        f"- 무승부: {s['sxs_tie_rate']:.0%}",
        "",
        "## 케이스별 상세",
        "",
    ]

    for r in case_results:
        if r.get("skipped"):
            lines.append(f"### {r['case_id']} — ⚠️ SKIP")
            continue

        sxs = r.get("sxs_evaluation", {})
        csr = r.get("constraint_satisfaction", {})
        si = r.get("structural_integrity", {})
        rt = r.get("red_team", {})

        lines += [
            f"### {r['case_id']} ({r['domain']})",
            f"- WBS 태스크 수: Baseline={r.get('baseline_task_count', 0)}, symPO={r.get('sympo_task_count', 0)}",
            f"- SI 점수: Baseline={si.get('baseline', {}).get('overall_score', 0):.1f}, "
            f"symPO={si.get('sympo', {}).get('overall_score', 0):.1f}",
            f"- CSR: Baseline={csr.get('baseline', {}).get('constraint_satisfaction_rate', 0):.1%}, "
            f"symPO={csr.get('sympo', {}).get('constraint_satisfaction_rate', 0):.1%}",
            f"- SxS 승자: **{sxs.get('winner', 'N/A')}** "
            f"(Baseline {sxs.get('score_baseline', 0):.1f} vs symPO {sxs.get('score_sympo', 0):.1f})",
            f"- Red Team 결함: Baseline={rt.get('baseline', {}).get('flaw_count', 0)}, "
            f"symPO={rt.get('sympo', {}).get('flaw_count', 0)}",
            "",
        ]

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="symPO GT-Free 평가 파이프라인")
    parser.add_argument("--domains", nargs="+",
                        default=["ecommerce", "fintech", "bigdata"],
                        choices=["ecommerce", "fintech", "erp", "bigdata", "saas"],
                        help="평가할 도메인 목록")
    parser.add_argument("--output", default="eval_results",
                        help="결과 저장 디렉토리")
    parser.add_argument("--team-size", type=int, default=5)
    parser.add_argument("--budget-weeks", type=int, default=12)
    parser.add_argument("--skip-sympo", action="store_true",
                        help="symPO 실행 스킵 (구조 검증 전용)")
    args = parser.parse_args()

    output_dir = os.path.join(_ROOT, args.output)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  symPO Evaluation Pipeline")
    print(f"  도메인: {args.domains}")
    print(f"  출력: {output_dir}")
    print(f"{'='*60}\n")

    # 벤치마크 케이스 생성
    cases = generate_benchmark_cases(
        domains=args.domains,
        team_size=args.team_size,
        budget_weeks=args.budget_weeks,
    )
    bench_path = save_benchmark_cases(cases, output_dir)
    print(f"벤치마크 케이스 저장: {bench_path}")

    # 케이스별 평가
    case_results = []
    for case in cases:
        if args.skip_sympo:
            # Baseline만 생성 후 구조 검사
            baseline_tasks = generate_baseline_wbs(case.prd, case.team)
            si = check_structural_integrity(baseline_tasks)
            print(format_report(si))
            continue
        result = evaluate_case(case, output_dir)
        case_results.append(result)

    if not case_results:
        print("\n평가 케이스 없음 (--skip-sympo 모드였나요?)")
        return

    # 종합 집계
    aggregate = aggregate_results(case_results)

    # JSON 저장
    full_report = {
        "aggregate": aggregate,
        "cases": case_results,
    }
    json_path = os.path.join(output_dir, "eval_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, ensure_ascii=False, indent=2, default=str)

    # Markdown 요약 저장
    md_path = os.path.join(output_dir, "eval_summary.md")
    md_content = generate_markdown_summary(aggregate, case_results)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    # 콘솔 출력
    print("\n" + "="*60)
    print("  최종 집계 결과")
    print("="*60)
    s = aggregate["summary"]
    print(f"  SI Delta:     {s['si_delta_avg']:+.1f}  (symPO {'우위' if s['si_delta_avg'] > 0 else '열위'})")
    print(f"  CSR Delta:    {s['csr_delta_avg']:+.1%}  (symPO {'우위' if s['csr_delta_avg'] > 0 else '열위'})")
    print(f"  SxS Win Rate: {s['sxs_sympo_win_rate']:.0%}")
    print(f"  RT Flaw Delta:{s['rt_flaw_delta_avg']:+.1f}  (양수=symPO 결함 적음)")
    print(f"\n  리포트: {json_path}")
    print(f"  요약:   {md_path}")


if __name__ == "__main__":
    main()
