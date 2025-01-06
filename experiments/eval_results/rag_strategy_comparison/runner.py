"""
실험: RAG 전략 변화에 따른 WBS 생성 품질 검증 (eval2.txt §6 RQ2)

- 조건 (RAG 전략): R0 없음 / R1 Vanilla / R2 Hybrid / R3 Graph / R4 Agentic
- eDISC: Diverse 고정 (6명 D·I·S·C·DI·SC)
- max_rounds = 2 (RQ2 baseline 고정)
- iteration = 5 (N≥5 프로토콜 준수)
- 백엔드: gemini, gemma4-api
- Judge (Structure/Assignment/Debate/Overall) + snapshot 저장

주의: 코드 수정 없이 외부 러너만. 기존 eDISC runner 구조 재활용.
"""
from __future__ import annotations

import os
import sys
import json
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple

PROJ_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ_ROOT))

from data_pipeline.prd_parser import PRDParser
from data_pipeline.member_parser import MemberParser
from data_pipeline.vector_store import WBSVectorStore
from data_pipeline.rag_strategies import get_strategy
from data_pipeline.disc_parser import DiscProfile
from persona_engine.persona_builder import PersonaBuilder
from agents.state import create_initial_state
from orchestration.debate_loop import execute_sympo_flow
from metrics import compute_all_metrics
from eval.llm_judge import evaluate_wbs, JUDGE_MODEL_GEMINI

# Diverse eDISC 합성 프로파일(기존 eDISC 실험과 동일 로직 재사용)
EDISC_RUNNER = PROJ_ROOT / "eval_results" / "edisc_rr_matching" / "runner.py"
_edisc_mod = None
if EDISC_RUNNER.exists():
    import importlib.util
    spec = importlib.util.spec_from_file_location("edisc_runner", EDISC_RUNNER)
    _edisc_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_edisc_mod)

SAMPLE_DIR = PROJ_ROOT / "sample_data"
MEMBERS_DIR = SAMPLE_DIR / "sample_members"

# RAG 조건 정의 (eval2.txt §6 RQ2)
RAG_CONDITIONS = {
    "R0": None,           # RAG 없음 — 빈 rag_reference_wbs / rag_meeting_logs
    "R1": "vanilla",
    "R2": "hybrid",
    "R3": "graph",
    "R4": "agentic",
}


def _load_team():
    team = []
    for p in sorted(MEMBERS_DIR.glob("member_*.txt")):
        text = p.read_text(encoding="utf-8")
        name = p.stem.replace("member_", "")
        team.append(MemberParser.from_resume_text(text, name=name))
    return team


def _load_prd_and_refs():
    prd_text = (SAMPLE_DIR / "sample_prd.txt").read_text(encoding="utf-8")
    prd = PRDParser.from_text(prd_text, "AI 기반 고객 서비스 플랫폼")
    ref_wbs_text = (SAMPLE_DIR / "sample_reference_wbs.txt").read_text(encoding="utf-8")
    meeting_text = (SAMPLE_DIR / "sample_meeting_transcript.txt").read_text(encoding="utf-8")
    return prd, ref_wbs_text, meeting_text


def build_diverse_disc(team_members) -> Dict[str, str]:
    """이전 eDISC 실험의 Diverse 프로파일 생성 함수 재사용"""
    if _edisc_mod is not None and hasattr(_edisc_mod, "build_disc_contexts"):
        return _edisc_mod.build_disc_contexts("diverse_edisc", [m.name for m in team_members])
    # fallback: 빈 DISC
    return {m.name: "" for m in team_members}


def build_rag_contexts(strategy_key, prd, team_members, ref_wbs_text, meeting_text):
    """
    RAG 전략에 따라 context를 준비.
    R0 (strategy_key=None)이면 빈 리스트 반환.
    """
    if strategy_key is None:
        return [], []

    vs = WBSVectorStore()
    vs.add_prd(prd)
    for m in team_members:
        vs.add_member(m)
    vs.add_reference_wbs(ref_wbs_text, prd.project_name)
    vs.add_meeting_log(meeting_text)

    strategy = get_strategy(strategy_key)
    kw = dict(documents=vs._documents, vectorstore=vs._vectorstore, embeddings=vs._embeddings)
    rag_wbs = strategy.retrieve(f"{prd.project_name} WBS 일정", "reference_wbs", k=3, **kw)
    rag_meet = strategy.retrieve("일정 버퍼 교훈", "meeting_log", k=3, **kw)
    return [d["content"] for d in rag_wbs], [d["content"] for d in rag_meet]


def run_one(backend, condition_key, iter_num,
            prd, team_members, ref_wbs_text, meeting_text,
            max_rounds=2, progress_writer=None):
    os.environ["LLM_BACKEND"] = backend

    strategy_key = RAG_CONDITIONS[condition_key]
    rag_wbs, rag_meet = build_rag_contexts(strategy_key, prd, team_members, ref_wbs_text, meeting_text)
    disc_contexts = build_diverse_disc(team_members)

    team_summary = PersonaBuilder.generate_team_summary(team_members)
    sup_persona = PersonaBuilder.build_supervisor_persona("PM 에이전트", team_summary)
    member_personas = PersonaBuilder.build_all_personas(team_members)
    all_personas = {"supervisor": sup_persona, **member_personas}

    initial_state = create_initial_state(
        prd=prd,
        team_members=team_members,
        agent_personas=all_personas,
        min_rounds=1,
        max_rounds=max_rounds,
    )
    initial_state["rag_reference_wbs"] = rag_wbs
    initial_state["rag_meeting_logs"] = rag_meet
    initial_state["disc_profiles"] = disc_contexts

    final_state = initial_state
    step_count = 0
    last_agent = None
    for state in execute_sympo_flow(initial_state, max_rounds):
        final_state = state
        step_count += 1
        agent = state.get("_current_agent_acting")
        if agent and agent != last_agent:
            last_agent = agent
            line = (f"    · step={step_count} r={state.get('current_round', 0)} "
                    f"agent={agent} msgs={len(state.get('debate_log', []))}")
            print(line, flush=True)
            if progress_writer:
                progress_writer(line)

    # 메트릭 산출
    metrics = compute_all_metrics(
        final_state=final_state,
        prd=prd,
        team_members=team_members,
        output_dir=str(PROJ_ROOT / "eval_results" / "rag_strategy_comparison" / "_tmp_metrics"),
        experiment_config={
            "rag_strategy": strategy_key or "none",
            "condition": condition_key,
            "min_rounds": 1,
            "max_rounds": max_rounds,
            "team_size": len(team_members),
            "note": f"rag={condition_key};iter={iter_num};backend={backend};edisc=diverse",
        },
    )

    # Judge 3차원 평가
    judge_result = None
    judge_err = None
    try:
        tasks_for_judge = final_state.get("final_wbs") or final_state.get("current_wbs_draft") or []
        debate_for_judge = final_state.get("debate_log", [])
        cross_judge = bool(os.environ.get("ANTHROPIC_API_KEY"))
        if progress_writer:
            progress_writer(f"    · judge 시작 (cross={cross_judge}) tasks={len(tasks_for_judge)} debate={len(debate_for_judge)}")
        judge_result = evaluate_wbs(
            wbs_tasks=tasks_for_judge,
            team_members=team_members,
            debate_log=debate_for_judge,
            eval_dims=["structure", "assignment", "debate"],
            judge_model=JUDGE_MODEL_GEMINI,
            cross_judge=cross_judge,
        )
    except Exception as je:
        judge_err = f"{type(je).__name__}: {je}"
        print(f"    [judge 실패] {judge_err}", flush=True)

    # 스냅샷 저장용 직렬화
    def _task_to_dict(t):
        try:
            return t.model_dump(mode="python")
        except Exception:
            try:
                return t.dict()
            except Exception:
                return {"repr": repr(t)[:200]}

    def _msg_to_dict(m):
        if isinstance(m, dict):
            return m
        try:
            return m.model_dump(mode="python")
        except Exception:
            out = {}
            for k in ["timestamp", "agent_role", "agent_name", "message",
                     "message_type", "related_task_id", "buffer_days_proposed"]:
                v = getattr(m, k, None)
                if hasattr(v, "value"):
                    v = v.value
                out[k] = v
            return out

    snapshot = {
        "backend": backend,
        "condition": condition_key,
        "rag_strategy": strategy_key,
        "iter": iter_num,
        "final_wbs": [_task_to_dict(t) for t in (final_state.get("final_wbs") or [])],
        "current_wbs_draft": [_task_to_dict(t) for t in (final_state.get("current_wbs_draft") or [])],
        "debate_log": [_msg_to_dict(m) for m in final_state.get("debate_log", [])],
    }

    return {
        "backend": backend,
        "condition": condition_key,
        "rag_strategy": strategy_key,
        "iter": iter_num,
        "metrics": metrics,
        "judge": judge_result,
        "judge_error": judge_err,
        "debate_log_len": len(final_state.get("debate_log", [])),
        "final_wbs_task_count": len(final_state.get("final_wbs") or final_state.get("current_wbs_draft") or []),
        "_snapshot": snapshot,
    }


def main():
    backends = [b.strip() for b in os.environ.get("RAG_BACKENDS", "gemini,gemma4-api").split(",") if b.strip()]
    iterations = int(os.environ.get("RAG_ITER", "5"))
    max_rounds = int(os.environ.get("RAG_MAX_ROUNDS", "2"))
    conditions = list(RAG_CONDITIONS.keys())

    out_root = PROJ_ROOT / "eval_results" / "rag_strategy_comparison"
    out_root.mkdir(parents=True, exist_ok=True)
    progress_log = out_root / "progress.log"

    def write_progress(line):
        with progress_log.open("a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {line}\n")

    print("=" * 70)
    print("실험: RAG 전략 변화에 따른 WBS 생성 품질 검증 (eval2 §6 RQ2)")
    print(f"  backends={backends} conditions={conditions} iter={iterations} max_rounds={max_rounds}")
    print("  eDISC=diverse (고정)")
    print("=" * 70, flush=True)
    write_progress(f"start backends={backends} conditions={conditions} iter={iterations} max_rounds={max_rounds}")

    prd, ref_wbs_text, meeting_text = _load_prd_and_refs()
    team_members = _load_team()
    print(f"[setup] 팀원 {len(team_members)}명: {[m.name for m in team_members]}", flush=True)
    write_progress(f"team={[m.name for m in team_members]}")

    total_runs = len(backends) * len(conditions) * iterations
    run_idx = 0
    all_results = []

    for backend in backends:
        backend_dir = out_root / f"backend_{backend.replace('-', '_')}"
        runs_dir = backend_dir / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        (runs_dir / "snapshots").mkdir(exist_ok=True)

        for condition in conditions:
            for it in range(1, iterations + 1):
                run_idx += 1
                tag = f"{backend}|{condition}|iter{it}"
                header = f"[{run_idx}/{total_runs}] >>> {tag} 시작 @ {datetime.now().strftime('%H:%M:%S')}"
                print(header, flush=True)
                write_progress(header)

                t0 = time.time()
                try:
                    rec = run_one(backend, condition, it, prd, team_members,
                                  ref_wbs_text, meeting_text,
                                  max_rounds=max_rounds, progress_writer=write_progress)
                    elapsed = time.time() - t0
                    rec["elapsed_sec"] = round(elapsed, 2)

                    # 스냅샷 분리 저장
                    snap = rec.pop("_snapshot", None)
                    if snap is not None:
                        snap_path = runs_dir / "snapshots" / f"{condition}_iter{it}.json"
                        snap_path.write_text(
                            json.dumps(snap, ensure_ascii=False, indent=2, default=str),
                            encoding="utf-8",
                        )

                    out_path = runs_dir / f"{condition}_iter{it}.json"
                    out_path.write_text(
                        json.dumps(rec, ensure_ascii=False, indent=2, default=str),
                        encoding="utf-8",
                    )

                    m = rec["metrics"]
                    j = rec.get("judge") or {}
                    judge_str = (
                        f"judge[S={j.get('structure',{}).get('score','NA')} "
                        f"A={j.get('assignment',{}).get('score','NA')} "
                        f"D={j.get('debate',{}).get('score','NA')} "
                        f"O={j.get('overall','NA')}]"
                        if rec.get("judge") else "judge=NA"
                    )
                    fai = m.get("ragas_faithfulness", {})
                    fai_str = f"faith={fai.get('faithfulness','NA')}"
                    summary = (
                        f"[{run_idx}/{total_runs}] <<< {tag} 완료 ({elapsed:.1f}s) "
                        f"planning={m.get('planning_score', {}).get('planning_score', 0):.4f} "
                        f"gini={m.get('workload_gini', {}).get('gini', 0):.4f} "
                        f"SR={m.get('success_rate', {}).get('success_rate', 0):.4f} "
                        f"autoscore={m.get('autoscore', {}).get('autoscore', 0):.4f} "
                        f"{fai_str} {judge_str}"
                    )
                    print(summary, flush=True)
                    write_progress(summary)
                    all_results.append(rec)
                except Exception as e:
                    err = f"[{run_idx}/{total_runs}] !!! {tag} 실패: {type(e).__name__}: {e}"
                    print(err, flush=True)
                    print(traceback.format_exc(), flush=True)
                    write_progress(err)
                    (runs_dir / f"{condition}_iter{it}.error.txt").write_text(
                        f"{err}\n\n{traceback.format_exc()}", encoding="utf-8"
                    )

    all_path = out_root / "all_runs.json"
    all_path.write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\n전체 결과 저장: {all_path}", flush=True)
    write_progress(f"DONE total={len(all_results)} saved={all_path}")


if __name__ == "__main__":
    main()
