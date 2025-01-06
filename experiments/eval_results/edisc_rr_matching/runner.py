"""
실험: 성향 데이터 조합에 따른 R&R 매칭 품질 검증
- 조건 A (same_edisc): 팀원 6명 전원이 동일한 eDISC 성향
- 조건 B (diverse_edisc): 팀원 6명이 서로 다른 eDISC 성향 (D/I/S/C 분포)
- iteration = 3
- 백엔드: gemini, gemma4-api

주의: 이 스크립트는 기존 코드를 수정하지 않는다. 외부 러너로만 동작.
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

# 프로젝트 루트를 PYTHONPATH에 추가
PROJ_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ_ROOT))

# ── 기존 코드 재사용 (수정 없이 호출만) ────────────────────
from data_pipeline.prd_parser import PRDParser
from data_pipeline.member_parser import MemberParser
from data_pipeline.vector_store import WBSVectorStore
from data_pipeline.rag_strategies import get_strategy
from data_pipeline.disc_parser import DiscProfile, load_all_disc_profiles
from persona_engine.persona_builder import PersonaBuilder
from agents.state import create_initial_state
from orchestration.debate_loop import execute_sympo_flow
from metrics import compute_all_metrics
from eval.llm_judge import evaluate_wbs, JUDGE_MODEL_GEMINI, JUDGE_MODEL_CLAUDE


# ───────────────────────────────────────────────────────────
# 1. 합성 DISC 프로파일 생성 (코드 수정 아님 — DiscProfile 생성자 호출)
# ───────────────────────────────────────────────────────────
def _disc_s_archetype(name: str) -> DiscProfile:
    """S형 (안정형/Steadiness) 원형 — '동일 조건'의 대표 프로필"""
    return DiscProfile(
        name=name,
        disc_style="S (S-70%, C-20%, I-10%)",
        primary_type="S형 (Steadiness / 안정형)",
        type_code="S",
        combo_code="SCI",
        disc_scores={"D": 0, "I": 10, "S": 70, "C": 20},
        strength_behaviors=[
            "팀 내 안정성과 지속성을 유지하려 노력",
            "협력적이고 일관된 태도로 동료를 지원",
            "변화보다 검증된 절차와 루틴을 선호",
            "갈등 시 중재자 역할 수행",
        ],
        improvement_behaviors=[
            "급격한 변화 수용 속도 개선 필요",
            "의견 충돌 시 명확한 입장 표명 강화",
            "일정 압박 상황에서 결정 속도 향상",
        ],
        behavioral_keywords=["안정적", "협력적", "인내심", "지속성", "신중함", "경청"],
        communication_style=(
            "조용하고 경청 중심의 커뮤니케이션을 선호하며, 의견 충돌을 피하고 "
            "모두가 합의할 수 있는 해결책을 제시하려 함. 감정적 톤보다 차분한 대화"
        ),
        decision_style=(
            "충분한 정보 수집과 관련자 합의 후 결정. 급하게 의사결정하기보다 "
            "단계적 검증을 선호"
        ),
        motivating_factors=["안정적 환경", "예측 가능한 절차", "팀 협력", "일관성"],
        demotivating_factors=["급격한 변화", "지속적 갈등", "불확실성", "과도한 압박"],
        team_role="안정자",
        team_role_en="Stabilizer",
        team_role_description=(
            "팀 내에서 일관성과 안정을 유지하는 역할. 동료의 의견을 경청하고 "
            "갈등을 완화하며, 지속 가능한 업무 흐름을 유지함."
        ),
        raw_text="S형 합성 프로필 — 실험용 (동일 조건)",
    )


def _disc_d_archetype(name: str) -> DiscProfile:
    return DiscProfile(
        name=name,
        disc_style="D (D-75%, I-15%, C-10%)",
        primary_type="D형 (Dominance / 주도형)",
        type_code="D",
        combo_code="DIC",
        disc_scores={"D": 75, "I": 15, "S": 0, "C": 10},
        strength_behaviors=[
            "목표 달성에 집중하여 빠른 결정",
            "도전적 과제에 적극적으로 뛰어듦",
            "리더십을 발휘하여 팀을 주도",
            "결과 중심의 실행력",
        ],
        improvement_behaviors=[
            "세부사항 검토 부족으로 오류 발생 가능",
            "타인의 의견 경청 시간 부족",
            "공감적 커뮤니케이션 강화 필요",
        ],
        behavioral_keywords=["주도적", "결단력", "경쟁적", "신속", "도전적", "직설적"],
        communication_style=(
            "직설적이고 결과 중심적. 빠른 의사결정을 선호하며 장황한 설명보다 "
            "핵심 포인트와 액션 아이템을 요구"
        ),
        decision_style="데이터보다 직관과 경험에 의존한 빠른 결정. 우선순위가 명확",
        motivating_factors=["도전 과제", "권한과 자율성", "빠른 성과", "경쟁"],
        demotivating_factors=["느린 절차", "과도한 합의 요구", "반복 업무"],
        team_role="주도자",
        team_role_en="Director",
        team_role_description="팀을 목표로 이끌고 결정을 주도하는 역할. 위기 상황에서 추진력 발휘.",
        raw_text="D형 합성 프로필",
    )


def _disc_i_archetype(name: str) -> DiscProfile:
    return DiscProfile(
        name=name,
        disc_style="I (I-70%, D-15%, S-15%)",
        primary_type="I형 (Influence / 사교형)",
        type_code="I",
        combo_code="IDS",
        disc_scores={"D": 15, "I": 70, "S": 15, "C": 0},
        strength_behaviors=[
            "동료와의 관계 형성에 능숙",
            "긍정적 에너지로 팀 분위기 활성화",
            "창의적 아이디어 제안",
            "이해관계자 설득 및 조율",
        ],
        improvement_behaviors=[
            "세부 일정 관리 부족",
            "감정적 결정으로 논리성 약함",
            "후속 실행 마무리 약함",
        ],
        behavioral_keywords=["낙관적", "사교적", "열정", "창의적", "설득력", "표현력"],
        communication_style="활발하고 감정 표현이 풍부. 아이디어 공유를 즐기며 청중 참여 유도",
        decision_style="직관과 감정 기반의 결정. 다수가 공감하는 방향 선호",
        motivating_factors=["사회적 인정", "창의적 환경", "다양한 사람과의 협업"],
        demotivating_factors=["고립된 업무", "반복적 단순 작업", "엄격한 규칙"],
        team_role="촉진자",
        team_role_en="Facilitator",
        team_role_description="팀의 협업 분위기를 만들고 아이디어 교류를 촉진.",
        raw_text="I형 합성 프로필",
    )


def _disc_c_archetype(name: str) -> DiscProfile:
    return DiscProfile(
        name=name,
        disc_style="C (C-75%, S-15%, D-10%)",
        primary_type="C형 (Compliance / 신중형)",
        type_code="C",
        combo_code="CSD",
        disc_scores={"D": 10, "I": 0, "S": 15, "C": 75},
        strength_behaviors=[
            "체계적이고 정확한 분석",
            "세부사항에 대한 높은 주의력",
            "품질 기준 준수 및 문서화",
            "리스크 요인을 사전에 식별",
        ],
        improvement_behaviors=[
            "완벽주의로 인한 일정 지연 가능",
            "변화 수용 속도 느림",
            "감정 표현 부족",
        ],
        behavioral_keywords=["분석적", "정확성", "체계적", "신중", "논리적", "비판적"],
        communication_style="사실과 데이터 중심. 근거가 뒷받침된 논리적 표현을 선호",
        decision_style="충분한 데이터 분석과 검증 후 결정. 리스크 회피 경향",
        motivating_factors=["명확한 기준", "품질 중시 환경", "심층 분석 기회"],
        demotivating_factors=["불명확한 지시", "급한 마감", "비논리적 결정"],
        team_role="검토자",
        team_role_en="Assurer",
        team_role_description="계획과 산출물의 품질·정확성을 검증하는 역할",
        raw_text="C형 합성 프로필",
    )


def _disc_di_archetype(name: str) -> DiscProfile:
    """DI 복합 — 주도+사교"""
    return DiscProfile(
        name=name,
        disc_style="DI (D-50%, I-40%, C-10%)",
        primary_type="D형 (Dominance / 주도형)",
        type_code="D",
        combo_code="DIC",
        disc_scores={"D": 50, "I": 40, "S": 0, "C": 10},
        strength_behaviors=[
            "추진력과 설득력을 겸비",
            "비전 제시와 팀 동기부여",
            "의사결정 속도 빠름",
            "외부 이해관계자와의 네트워킹",
        ],
        improvement_behaviors=["세부 계획 부족", "감정과 논리 혼재", "후속 관리 약함"],
        behavioral_keywords=["리더십", "카리스마", "추진력", "열정", "설득"],
        communication_style="단호하면서 열정적. 청중을 설득하고 행동을 촉구",
        decision_style="직관적이고 빠른 결정. 팀 모멘텀 유지를 우선",
        motivating_factors=["주도권", "공개적 성과 인정", "성장 기회"],
        demotivating_factors=["세부 관리 요구", "정체된 환경"],
        team_role="변화추진자",
        team_role_en="Promoter",
        team_role_description="팀의 변화를 촉발하고 새로운 방향을 제시",
        raw_text="DI 복합 합성 프로필",
    )


def _disc_sc_archetype(name: str) -> DiscProfile:
    """SC 복합 — 안정+신중"""
    return DiscProfile(
        name=name,
        disc_style="SC (S-50%, C-40%, I-10%)",
        primary_type="S형 (Steadiness / 안정형)",
        type_code="S",
        combo_code="SCI",
        disc_scores={"D": 0, "I": 10, "S": 50, "C": 40},
        strength_behaviors=[
            "꼼꼼하고 일관된 실행",
            "절차 준수와 품질 유지",
            "동료 지원과 문서 정리",
            "리스크 사전 점검",
        ],
        improvement_behaviors=["결정 속도 느림", "주도적 제안 부족", "변화 적응 느림"],
        behavioral_keywords=["꼼꼼", "일관성", "신중", "지원적", "세심"],
        communication_style="차분하고 논리적. 명확한 근거와 절차를 공유",
        decision_style="데이터 기반이며 위험 회피. 동료 합의 후 결정",
        motivating_factors=["안정적 구조", "명확한 기준", "품질 중심"],
        demotivating_factors=["급한 의사결정", "모호한 지시"],
        team_role="검토자",
        team_role_en="Assurer",
        team_role_description="품질 검증과 프로세스 안정화 역할",
        raw_text="SC 복합 합성 프로필",
    )


def build_disc_contexts(condition: str, member_names: List[str]) -> Dict[str, str]:
    """
    condition:
      - 'same_edisc'    : 전원 S형 동일 프로파일
      - 'diverse_edisc' : 6명 서로 다른 성향 (D, I, S, C, DI, SC 분포)
    반환: {이름: to_agent_context() 문자열}
    """
    assert len(member_names) == 6, "이 실험은 팀원 6명 고정"

    if condition == "same_edisc":
        return {name: _disc_s_archetype(name).to_agent_context() for name in member_names}

    if condition == "diverse_edisc":
        builders = [
            _disc_d_archetype,
            _disc_i_archetype,
            _disc_s_archetype,
            _disc_c_archetype,
            _disc_di_archetype,
            _disc_sc_archetype,
        ]
        return {name: builder(name).to_agent_context()
                for name, builder in zip(member_names, builders)}

    raise ValueError(f"unknown condition: {condition}")


# ───────────────────────────────────────────────────────────
# 2. 입력 데이터 로드 — 모든 실험 조건에 동일하게 고정
# ───────────────────────────────────────────────────────────
SAMPLE_DIR = PROJ_ROOT / "sample_data"
MEMBERS_DIR = SAMPLE_DIR / "sample_members"


def load_fixed_inputs() -> Tuple[Any, List[Any], List[str], List[str]]:
    """
    PRD, 팀원 6명, RAG 참고 WBS·회의록 — 조건 간 동일 고정.
    """
    prd_text = (SAMPLE_DIR / "sample_prd.txt").read_text(encoding="utf-8")
    prd = PRDParser.from_text(prd_text, "AI 기반 고객 서비스 플랫폼")

    team_members = []
    for p in sorted(MEMBERS_DIR.glob("member_*.txt")):
        text = p.read_text(encoding="utf-8")
        # 파일명에서 이름 추출
        fname = p.stem  # member_박선민
        name = fname.replace("member_", "")
        m = MemberParser.from_resume_text(text, name=name)
        team_members.append(m)

    ref_wbs_text = (SAMPLE_DIR / "sample_reference_wbs.txt").read_text(encoding="utf-8")
    meeting_text = (SAMPLE_DIR / "sample_meeting_transcript.txt").read_text(encoding="utf-8")

    # 벡터 스토어로 RAG 검색 수행 (main.py와 동일 파이프라인)
    vs = WBSVectorStore()
    vs.add_prd(prd)
    for m in team_members:
        vs.add_member(m)
    vs.add_reference_wbs(ref_wbs_text, prd.project_name)
    vs.add_meeting_log(meeting_text)

    strategy = get_strategy("vanilla")
    kw = dict(documents=vs._documents, vectorstore=vs._vectorstore, embeddings=vs._embeddings)
    rag_wbs = strategy.retrieve(f"{prd.project_name} WBS 일정", "reference_wbs", k=3, **kw)
    rag_meet = strategy.retrieve("일정 버퍼 교훈", "meeting_log", k=3, **kw)
    rag_wbs_texts = [d["content"] for d in rag_wbs]
    rag_meet_texts = [d["content"] for d in rag_meet]

    return prd, team_members, rag_wbs_texts, rag_meet_texts


# ───────────────────────────────────────────────────────────
# 3. 단일 실험 실행
# ───────────────────────────────────────────────────────────
def run_one(backend: str, condition: str, iter_num: int,
            prd, team_members, rag_wbs, rag_meet,
            max_rounds: int = 2,
            progress_writer=None) -> Dict[str, Any]:
    """단일 실험 실행 → metrics 반환"""
    os.environ["LLM_BACKEND"] = backend

    member_names = [m.name for m in team_members]
    disc_contexts = build_disc_contexts(condition, member_names)

    # 에이전트 페르소나
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

    # 토론 루프 실행 — 실시간 진행 출력
    final_state = initial_state
    step_count = 0
    last_agent_label = None
    for current_state in execute_sympo_flow(initial_state, max_rounds):
        final_state = current_state
        step_count += 1
        agent = current_state.get("_current_agent_acting")
        if agent and agent != last_agent_label:
            last_agent_label = agent
            msg_n = len(current_state.get("debate_log", []))
            line = f"    · step={step_count} round={current_state.get('current_round', 0)} agent={agent} msgs={msg_n}"
            print(line, flush=True)
            if progress_writer is not None:
                progress_writer(line)

    # 평가지표 산출 (이 스크립트에서는 파일 저장 방지를 위해 output_dir을 임시로 분리)
    metrics = compute_all_metrics(
        final_state=final_state,
        prd=prd,
        team_members=team_members,
        output_dir=str(PROJ_ROOT / "eval_results" / "edisc_rr_matching" / "_tmp_metrics"),
        experiment_config={
            "rag_strategy": "vanilla",
            "min_rounds": 1,
            "max_rounds": max_rounds,
            "team_size": len(team_members),
            "note": f"edisc_condition={condition};iter={iter_num};backend={backend}",
        },
    )

    # ── LLM-as-a-Judge 3차원 평가 (Structure/Assignment/Debate) ──
    judge_result = None
    judge_err = None
    try:
        tasks_for_judge = final_state.get("final_wbs") or final_state.get("current_wbs_draft") or []
        debate_for_judge = final_state.get("debate_log", [])
        cross_judge = bool(os.environ.get("ANTHROPIC_API_KEY"))
        if progress_writer is not None:
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

    # ── final WBS + debate_log 스냅샷 저장 (재평가·감사용) ──
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
        "condition": condition,
        "iter": iter_num,
        "final_wbs": [_task_to_dict(t) for t in (final_state.get("final_wbs") or [])],
        "current_wbs_draft": [_task_to_dict(t) for t in (final_state.get("current_wbs_draft") or [])],
        "debate_log": [_msg_to_dict(m) for m in final_state.get("debate_log", [])],
    }

    return {
        "backend": backend,
        "condition": condition,
        "iter": iter_num,
        "metrics": metrics,
        "judge": judge_result,
        "judge_error": judge_err,
        "debate_log_len": len(final_state.get("debate_log", [])),
        "final_wbs_task_count": len(final_state.get("final_wbs") or final_state.get("current_wbs_draft") or []),
        "_snapshot": snapshot,
    }


# ───────────────────────────────────────────────────────────
# 4. 전체 실험 오케스트레이션
# ───────────────────────────────────────────────────────────
def main():
    backends = [b.strip() for b in os.environ.get("EDISC_BACKENDS", "gemini,gemma4-api").split(",") if b.strip()]
    iterations = int(os.environ.get("EDISC_ITER", "3"))
    max_rounds = int(os.environ.get("EDISC_MAX_ROUNDS", "2"))
    conditions = ["same_edisc", "diverse_edisc"]

    out_root = PROJ_ROOT / "eval_results" / "edisc_rr_matching"
    out_root.mkdir(parents=True, exist_ok=True)
    progress_log_path = out_root / "progress.log"

    def write_progress(line: str):
        with progress_log_path.open("a", encoding="utf-8") as pf:
            pf.write(f"[{datetime.now().strftime('%H:%M:%S')}] {line}\n")

    print("=" * 70)
    print("실험: 성향 데이터 조합에 따른 R&R 매칭 품질 검증")
    print(f"  백엔드: {backends}  조건: {conditions}  iter={iterations}  max_rounds={max_rounds}")
    print("=" * 70, flush=True)
    write_progress(f"start backends={backends} conditions={conditions} iter={iterations} max_rounds={max_rounds}")

    # 입력 데이터는 한 번만 로드 (모든 조건 공통)
    print("[setup] PRD / 팀원 / RAG 로드 중...", flush=True)
    prd, team_members, rag_wbs, rag_meet = load_fixed_inputs()
    print(f"[setup] 팀원 {len(team_members)}명: {[m.name for m in team_members]}", flush=True)
    write_progress(f"team_members={[m.name for m in team_members]}")

    total_runs = len(backends) * len(conditions) * iterations
    run_idx = 0
    results = []

    for backend in backends:
        backend_dir = out_root / f"backend_{backend.replace('-', '_')}"
        runs_dir = backend_dir / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)

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
                                  rag_wbs, rag_meet, max_rounds=max_rounds,
                                  progress_writer=write_progress)
                    elapsed = time.time() - t0
                    rec["elapsed_sec"] = round(elapsed, 2)

                    # 스냅샷 분리 저장 (rec에서는 _snapshot 제거)
                    snap = rec.pop("_snapshot", None)
                    if snap is not None:
                        snap_dir = runs_dir / "snapshots"
                        snap_dir.mkdir(exist_ok=True)
                        snap_path = snap_dir / f"{condition}_iter{it}.json"
                        snap_path.write_text(
                            json.dumps(snap, ensure_ascii=False, indent=2, default=str),
                            encoding="utf-8",
                        )

                    # 저장 (rec = metrics + judge summary)
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
                    summary = (
                        f"[{run_idx}/{total_runs}] <<< {tag} 완료 ({elapsed:.1f}s) "
                        f"planning={m.get('planning_score', {}).get('planning_score'):.4f} "
                        f"gini={m.get('workload_gini', {}).get('gini'):.4f} "
                        f"feasibility={m.get('schedule_feasibility', {}).get('feasibility'):.4f} "
                        f"SR={m.get('success_rate', {}).get('success_rate'):.4f} "
                        f"autoscore={m.get('autoscore', {}).get('autoscore'):.4f} "
                        f"{judge_str}"
                    )
                    print(summary, flush=True)
                    write_progress(summary)
                    results.append(rec)
                except Exception as e:
                    err = f"[{run_idx}/{total_runs}] !!! {tag} 실패: {type(e).__name__}: {e}"
                    print(err, flush=True)
                    print(traceback.format_exc(), flush=True)
                    write_progress(err)
                    err_path = runs_dir / f"{condition}_iter{it}.error.txt"
                    err_path.write_text(
                        f"{err}\n\n{traceback.format_exc()}",
                        encoding="utf-8",
                    )

    # 전체 결과 집계 저장
    all_path = out_root / "all_runs.json"
    all_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\n전체 결과 저장: {all_path}", flush=True)
    write_progress(f"DONE total_results={len(results)} saved={all_path}")


if __name__ == "__main__":
    main()
