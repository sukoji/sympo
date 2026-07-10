"""
Ablation Study 실험 러너
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
조건별 WBS 생성을 자동 실행하고 13개 지표를 수집합니다.

실험 조건:
  C0: LLM 단독 생성 (토론 없음)
  C1: + Task Manager 배정 (토론 없음)
  C2: + 1라운드 토론
  C3: + 3라운드 토론 (full system)
  C4: + eDISC 성향 반영

사용법:
  python eval/experiment_runner.py --backend mock --runs 3
  python eval/experiment_runner.py --backend gemini --runs 5
"""
import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, Any, List

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SRC = os.path.join(_ROOT, "src")
_EXPERIMENTS = os.path.join(_ROOT, "experiments")
# _EXPERIMENTS: src/metrics.py 가 'eval_results.autoscore_recompute' 를 import 하므로
# experiments/ 를 경로에 넣어 eval_results 패키지를 최상위로 노출한다.
for _p in (_SRC, _ROOT, _EXPERIMENTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dotenv import load_dotenv
load_dotenv()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 실험 조건 정의
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONDITIONS = {
    # ── Ablation Study (구성 요소 기여도) ──
    # 평가 차원: S=Structure, A=Assignment, D=Debate
    # eval_dims: Judge가 평가할 차원 목록 (N/A인 차원은 제외하여 공정 비교)
    # harness_enabled: rev.2 신규. None이면 env HARNESS_ENABLED 기본값을 따름.
    #   RQ1-H 실험에서는 --harness on / --harness off로 외부에서 일괄 지정.
    "C0_llm_only": {
        "label": "C0: LLM 단독",
        "max_rounds": 0, "min_rounds": 0,
        "use_task_match": False, "use_disc": False,
        "rag_strategy": "vanilla", "use_meeting": False,
        "eval_dims": ["structure"],  # 배정/토론 없으므로 구조만 평가
        "harness_enabled": None,
        "description": "wbs_gen_node만 호출 — 단일 LLM 구조 생성 능력 참고용",
    },
    "C1_with_assign": {
        "label": "C1: Baseline (배정)",
        "max_rounds": 0, "min_rounds": 0,
        "use_task_match": True, "use_disc": False,
        "rag_strategy": "vanilla", "use_meeting": False,
        "eval_dims": ["structure", "assignment"],  # 토론 없으므로 구조+배분만
        "harness_enabled": None,
        "description": "WBS 생성 + Task Manager 배정 — Ablation 기준선(baseline)",
    },
    "C2_1round": {
        "label": "C2: +1R 토론",
        "max_rounds": 1, "min_rounds": 1,
        "use_task_match": True, "use_disc": False,
        "rag_strategy": "vanilla", "use_meeting": False,
        "eval_dims": ["structure", "assignment", "debate"],
        "harness_enabled": None,
        "description": "WBS 생성 + 배정 + 1라운드 토론",
    },
    "C3_3rounds": {
        "label": "C3: +3R 토론",
        "max_rounds": 3, "min_rounds": 2,
        "use_task_match": True, "use_disc": False,
        "rag_strategy": "vanilla", "use_meeting": False,
        "eval_dims": ["structure", "assignment", "debate"],
        "harness_enabled": None,
        "description": "WBS 생성 + 배정 + 3라운드 토론 (기본 시스템)",
    },
    "C4_with_disc": {
        "label": "C4: +eDISC",
        "max_rounds": 3, "min_rounds": 2,
        "use_task_match": True, "use_disc": True,
        "rag_strategy": "vanilla", "use_meeting": False,
        "eval_dims": ["structure", "assignment", "debate"],
        "harness_enabled": None,
        "description": "Full system + eDISC 행동유형 반영",
    },
    "C5_5rounds": {
        "label": "C5: +5R 토론",
        "max_rounds": 5, "min_rounds": 3,
        "use_task_match": True, "use_disc": False,
        "rag_strategy": "vanilla", "use_meeting": False,
        "eval_dims": ["structure", "assignment", "debate"],
        "harness_enabled": None,
        "description": "WBS 생성 + 배정 + 5라운드 토론 (S1 토론 라운드 수 감도분석)",
    },
    # ── R5: 회의록 변형(일정 포함 vs 미포함) ──
    "R5_meeting_regular": {
        "label": "R5-A: 회의록(일정 포함)",
        "max_rounds": 2, "min_rounds": 1,
        "use_task_match": True, "use_disc": False,
        "rag_strategy": "vanilla", "use_meeting": True,
        "meeting_variant": "regular",
        "eval_dims": ["structure", "assignment", "debate"],
        "harness_enabled": None,
        "description": "일정·마감일 정보가 포함된 기본 회의록 사용",
    },
    "R5_meeting_no_schedule": {
        "label": "R5-B: 회의록(일정 없음)",
        "max_rounds": 2, "min_rounds": 1,
        "use_task_match": True, "use_disc": False,
        "rag_strategy": "vanilla", "use_meeting": True,
        "meeting_variant": "no_schedule",
        "eval_dims": ["structure", "assignment", "debate"],
        "harness_enabled": None,
        "description": "일정/마감일 제거된 회의록 사용 (근거 충실도 감도분석)",
    },
    # ── RAG Strategy 비교 (RQ2) ──
    "R0_no_rag": {
        "label": "R0: RAG 없음",
        "max_rounds": 2, "min_rounds": 1,
        "use_task_match": True, "use_disc": False,
        "rag_strategy": None, "use_meeting": False,
        "eval_dims": ["structure", "assignment", "debate"],
        "harness_enabled": None,
        "description": "RAG 참고 데이터 없이 WBS 생성 + 2R 토론",
    },
    "R1_vanilla": {
        "label": "R1: Vanilla RAG",
        "max_rounds": 2, "min_rounds": 1,
        "use_task_match": True, "use_disc": False,
        "rag_strategy": "vanilla", "use_meeting": True,
        "eval_dims": ["structure", "assignment", "debate"],
        "harness_enabled": None,
        "description": "FAISS Dense 검색 + 샘플 회의록",
    },
    "R2_hybrid": {
        "label": "R2: Hybrid RAG",
        "max_rounds": 2, "min_rounds": 1,
        "use_task_match": True, "use_disc": False,
        "rag_strategy": "hybrid", "use_meeting": True,
        "eval_dims": ["structure", "assignment", "debate"],
        "harness_enabled": None,
        "description": "BM25 + Dense RRF + 샘플 회의록",
    },
    "R3_graph": {
        "label": "R3: Graph RAG",
        "max_rounds": 2, "min_rounds": 1,
        "use_task_match": True, "use_disc": False,
        "rag_strategy": "graph", "use_meeting": True,
        "eval_dims": ["structure", "assignment", "debate"],
        "harness_enabled": None,
        "description": "엔티티 그래프 검색 + 샘플 회의록",
    },
    "R4_agentic": {
        "label": "R4: Agentic RAG",
        "max_rounds": 2, "min_rounds": 1,
        "use_task_match": True, "use_disc": False,
        "rag_strategy": "agentic", "use_meeting": True,
        "eval_dims": ["structure", "assignment", "debate"],
        "harness_enabled": None,
        "description": "커버리지 기반 반복 멀티홉 검색 + 샘플 회의록",
    },
    "R4_llm_rerank": {
        "label": "R4-B: LLM Rerank",
        "max_rounds": 2, "min_rounds": 1,
        "use_task_match": True, "use_disc": False,
        "rag_strategy": "llm_rerank", "use_meeting": True,
        "eval_dims": ["structure", "assignment", "debate"],
        "harness_enabled": None,
        "description": "Dense top-k*3 후보 → LLM 재정렬 top-k (R4 rerank 비교)",
    },
    # ── A2: 거부권 토글 ──
    "A2_no_veto": {
        "label": "A2-A: 거부권 없음",
        "max_rounds": 3, "min_rounds": 2,
        "use_task_match": True, "use_disc": False,
        "rag_strategy": "vanilla", "use_meeting": False,
        "veto_enabled": False,
        "eval_dims": ["structure", "assignment", "debate"],
        "harness_enabled": None,
        "description": "PM이 모든 버퍼 제안 집계 (기본)",
    },
    "A2_with_veto": {
        "label": "A2-B: 거부권 허용",
        "max_rounds": 3, "min_rounds": 2,
        "use_task_match": True, "use_disc": False,
        "rag_strategy": "vanilla", "use_meeting": False,
        "veto_enabled": True,
        "eval_dims": ["structure", "assignment", "debate"],
        "harness_enabled": None,
        "description": "[VETO] 키워드로 단일 에이전트가 버퍼 거부 가능",
    },
    # ── A1: Critic 교차 심사 ──
    "A1_single": {
        "label": "A1-A: 단일 리뷰",
        "max_rounds": 3, "min_rounds": 2,
        "use_task_match": True, "use_disc": False,
        "rag_strategy": "vanilla", "use_meeting": False,
        "critic_enabled": False,
        "eval_dims": ["structure", "assignment", "debate"],
        "harness_enabled": None,
        "description": "기본 토론 (크리틱 없음)",
    },
    "A1_critic": {
        "label": "A1-B: Critic 교차 심사",
        "max_rounds": 3, "min_rounds": 2,
        "use_task_match": True, "use_disc": False,
        "rag_strategy": "vanilla", "use_meeting": False,
        "critic_enabled": True,
        "eval_dims": ["structure", "assignment", "debate"],
        "harness_enabled": None,
        "description": "mediate 직전 독립 Critic이 리스크·버퍼 이상치 지적",
    },
    # ── S2: 모델 급 × 프롬프팅 전략 (6개 조합) ──
    "S2_frontier_single": {
        "label": "S2: Frontier + Single",
        "max_rounds": 2, "min_rounds": 1,
        "use_task_match": True, "use_disc": False,
        "rag_strategy": "vanilla", "use_meeting": False,
        "model_class": "frontier", "prompting_strategy": "single",
        "eval_dims": ["structure", "assignment", "debate"],
        "harness_enabled": None,
        "description": "Gemini + 단일 응답",
    },
    "S2_frontier_chaining": {
        "label": "S2: Frontier + Chaining",
        "max_rounds": 2, "min_rounds": 1,
        "use_task_match": True, "use_disc": False,
        "rag_strategy": "vanilla", "use_meeting": False,
        "model_class": "frontier", "prompting_strategy": "chaining",
        "eval_dims": ["structure", "assignment", "debate"],
        "harness_enabled": None,
        "description": "Gemini + 단계별 사슬 프롬프팅",
    },
    "S2_frontier_cot": {
        "label": "S2: Frontier + CoT",
        "max_rounds": 2, "min_rounds": 1,
        "use_task_match": True, "use_disc": False,
        "rag_strategy": "vanilla", "use_meeting": False,
        "model_class": "frontier", "prompting_strategy": "cot",
        "eval_dims": ["structure", "assignment", "debate"],
        "harness_enabled": None,
        "description": "Gemini + Chain-of-Thought",
    },
    "S2_8b_single": {
        "label": "S2: 8B + Single",
        "max_rounds": 2, "min_rounds": 1,
        "use_task_match": True, "use_disc": False,
        "rag_strategy": "vanilla", "use_meeting": False,
        "model_class": "8b", "prompting_strategy": "single",
        "eval_dims": ["structure", "assignment", "debate"],
        "harness_enabled": None,
        "description": "Gemma4 (8B) + 단일 응답",
    },
    "S2_8b_chaining": {
        "label": "S2: 8B + Chaining",
        "max_rounds": 2, "min_rounds": 1,
        "use_task_match": True, "use_disc": False,
        "rag_strategy": "vanilla", "use_meeting": False,
        "model_class": "8b", "prompting_strategy": "chaining",
        "eval_dims": ["structure", "assignment", "debate"],
        "harness_enabled": None,
        "description": "Gemma4 (8B) + 단계별 사슬 프롬프팅",
    },
    "S2_8b_cot": {
        "label": "S2: 8B + CoT",
        "max_rounds": 2, "min_rounds": 1,
        "use_task_match": True, "use_disc": False,
        "rag_strategy": "vanilla", "use_meeting": False,
        "model_class": "8b", "prompting_strategy": "cot",
        "eval_dims": ["structure", "assignment", "debate"],
        "harness_enabled": None,
        "description": "Gemma4 (8B) + Chain-of-Thought",
    },
    # ── R1: PRD 정보 밀도 × 회의록 유무 ──
    "R1_prd_summary": {
        "label": "R1-A: PRD 요약본",
        "max_rounds": 2, "min_rounds": 1,
        "use_task_match": True, "use_disc": False,
        "rag_strategy": "vanilla", "use_meeting": False,
        "prd_variant": "summary",
        "eval_dims": ["structure", "assignment", "debate"],
        "harness_enabled": None,
        "description": "간결 PRD 요약본 — 정보 밀도 낮음",
    },
    "R1_prd_detailed": {
        "label": "R1-B: PRD 상세본",
        "max_rounds": 2, "min_rounds": 1,
        "use_task_match": True, "use_disc": False,
        "rag_strategy": "vanilla", "use_meeting": False,
        "prd_variant": "detailed_full",
        "eval_dims": ["structure", "assignment", "debate"],
        "harness_enabled": None,
        "description": "확장 PRD 상세본 — 기능·제약 세부 기술",
    },
    "R1_prd_detailed_meeting": {
        "label": "R1-C: PRD 상세본 + 회의록",
        "max_rounds": 2, "min_rounds": 1,
        "use_task_match": True, "use_disc": False,
        "rag_strategy": "vanilla", "use_meeting": True,
        "prd_variant": "detailed_full",
        "eval_dims": ["structure", "assignment", "debate"],
        "harness_enabled": None,
        "description": "확장 PRD + 회의록 RAG 결합",
    },
    # ── A3: 페르소나 강도 ──
    "A3_defensive": {
        "label": "A3-A: 방어적 페르소나",
        "max_rounds": 3, "min_rounds": 2,
        "use_task_match": True, "use_disc": False,
        "rag_strategy": "vanilla", "use_meeting": False,
        "persona_strictness": "defensive",
        "eval_dims": ["structure", "assignment", "debate"],
        "harness_enabled": None,
        "description": "리스크 과장·버퍼 보수적 요청",
    },
    "A3_aggressive": {
        "label": "A3-B: 공격적 페르소나",
        "max_rounds": 3, "min_rounds": 2,
        "use_task_match": True, "use_disc": False,
        "rag_strategy": "vanilla", "use_meeting": False,
        "persona_strictness": "aggressive",
        "eval_dims": ["structure", "assignment", "debate"],
        "harness_enabled": None,
        "description": "일정 단축·버퍼 최소화",
    },
}


def _load_sample_prd(variant: str = None):
    """
    샘플 PRD 로드. variant로 R1 변형 선택:
      - None/"detailed" (default): sample_prd.txt (기본)
      - "summary"     : sample_prd_summary.txt (짧은 요약본)
      - "detailed_full": sample_prd_detailed.txt (확장 상세본)
    """
    from data_pipeline.prd_parser import PRDParser
    filename = {
        "summary":       "sample_prd_summary.txt",
        "detailed_full": "sample_prd_detailed.txt",
    }.get(variant or "detailed", "sample_prd.txt")
    prd_path = os.path.join(_ROOT, "sample_data", filename)
    if os.path.exists(prd_path):
        text = open(prd_path, encoding="utf-8").read()
        return PRDParser.from_text(text, "P마켓 빅데이터 기반 매출 증대 전략")
    # fallback: 하드코딩
    return PRDParser.from_form(
        project_name="P마켓 빅데이터 기반 매출 증대 전략",
        project_goal="빅데이터 분석으로 매출 20% 이상 회복",
        target_users="P마켓 방문 고객",
        scope="판매·고객 데이터 분석 + 맞춤형 프로모션",
        key_features_text="소매업 트렌드 조사\n고객 세그먼트별 맞춤형 서비스\n모바일 앱 고도화\n매장 진열 리디자인",
        tech_stack_text="Python/Pandas\nTableau\nMySQL\n모바일 앱",
        deadline="2025-12-31",
        team_size=6,
        budget_weeks=24,
        constraints_text="개인정보보호법 준수\n기존 POS 시스템 연동",
    )


def _load_sample_team():
    """샘플 팀원 로드"""
    from data_pipeline.member_parser import MemberParser
    from schemas.member_schema import MemberProfile
    import uuid

    member_dir = os.path.join(_ROOT, "sample_data", "sample_members")
    team = []
    if os.path.isdir(member_dir):
        for fname in sorted(os.listdir(member_dir)):
            if fname.endswith(".txt"):
                content = open(os.path.join(member_dir, fname), encoding="utf-8").read()
                name = fname.replace("member_", "").replace(".txt", "")
                p = MemberParser.from_resume_text(content, name)
                team.append(p)
    return team


def _load_disc_profiles():
    """eDISC 프로파일 로드"""
    try:
        from data_pipeline.disc_parser import load_all_disc_profiles
        base = os.path.join(_ROOT, "sample_data")
        return load_all_disc_profiles(base)
    except Exception:
        return {}


def run_single_experiment(
    condition_key: str,
    condition: dict,
    prd,
    team_members: list,
    disc_profiles: dict,
    backend: str,
    run_id: int,
) -> Dict[str, Any]:
    """단일 실험 조건으로 WBS 생성 실행 + 메트릭 수집"""
    from agents.state import create_initial_state
    from persona_engine.persona_builder import PersonaBuilder

    # S2: model_class로 backend override (frontier=gemini, 8b=gemma4-api)
    model_class = condition.get("model_class")
    if model_class == "frontier":
        os.environ["LLM_BACKEND"] = "gemini"
    elif model_class == "8b":
        os.environ["LLM_BACKEND"] = "gemma4-api"
    else:
        os.environ["LLM_BACKEND"] = backend

    start_time = time.time()

    # 페르소나 생성
    team_summary = PersonaBuilder.generate_team_summary(team_members)
    supervisor_persona = PersonaBuilder.build_supervisor_persona("PM 에이전트", team_summary)
    member_personas = PersonaBuilder.build_all_personas(team_members)
    all_personas = {"supervisor": supervisor_persona, **member_personas}

    initial_state = create_initial_state(
        prd=prd,
        team_members=team_members,
        agent_personas=all_personas,
        min_rounds=condition["min_rounds"],
        max_rounds=condition["max_rounds"],
        harness_enabled=condition.get("harness_enabled"),  # None=env default
        veto_enabled=condition.get("veto_enabled"),
        critic_enabled=condition.get("critic_enabled"),
        persona_strictness=condition.get("persona_strictness"),
        prd_variant=condition.get("prd_variant"),
        model_class=condition.get("model_class"),
        prompting_strategy=condition.get("prompting_strategy"),
        model_config=condition.get("model_config"),
    )

    # eDISC 반영 여부
    if condition["use_disc"] and disc_profiles:
        initial_state["disc_profiles"] = {name: p.to_agent_context() for name, p in disc_profiles.items()}
    else:
        initial_state["disc_profiles"] = {}

    # ── RAG 벡터 DB 구축 + 회의록 반영 ──
    rag_strategy_key = condition.get("rag_strategy")
    if rag_strategy_key:
        try:
            from data_pipeline.vector_store import WBSVectorStore
            from data_pipeline.rag_strategies import get_strategy
            vs = WBSVectorStore()
            vs.add_prd(prd)
            for m in team_members:
                vs.add_member(m)
            # 회의록 로드 (meeting_variant: regular | no_schedule)
            if condition.get("use_meeting"):
                meeting_variant = condition.get("meeting_variant", "regular")
                meeting_file = {
                    "regular":     "sample_meeting_transcript.txt",
                    "no_schedule": "sample_meeting_no_schedule.txt",
                }.get(meeting_variant, "sample_meeting_transcript.txt")
                meeting_path = os.path.join(_ROOT, "sample_data", meeting_file)
                if os.path.exists(meeting_path):
                    meeting_text = open(meeting_path, encoding="utf-8").read()
                    vs.add_meeting_log(meeting_text)
            # eDISC 프로파일도 벡터 DB에 추가
            for dp in disc_profiles.values():
                vs.add_disc_profile(dp)
            # RAG 검색
            strategy = get_strategy(rag_strategy_key)
            # LLM reranker 등 rerank 전략은 llm 객체 필요
            rerank_llm = None
            if rag_strategy_key in ("llm_rerank",):
                try:
                    from agents.llm_config import get_llm
                    rerank_llm = get_llm()
                except Exception:
                    rerank_llm = None
            common_kw = dict(documents=vs._documents, vectorstore=vs._vectorstore, embeddings=vs._embeddings, llm=rerank_llm)
            rag_wbs = strategy.retrieve(f"{prd.project_name} WBS 일정", "reference_wbs", k=3, **common_kw)
            rag_meetings = strategy.retrieve("일정 버퍼 교훈", "meeting_log", k=3, **common_kw)
            initial_state["rag_reference_wbs"] = [d["content"] for d in rag_wbs]
            initial_state["rag_meeting_logs"] = [d["content"] for d in rag_meetings]
        except Exception as e:
            print(f"  [RAG] 벡터 DB 구축 실패 (무시): {e}")
            initial_state["rag_reference_wbs"] = []
            initial_state["rag_meeting_logs"] = []
    else:
        initial_state["rag_reference_wbs"] = []
        initial_state["rag_meeting_logs"] = []

    # ── 조건별 실행 분기 ──
    final_state = initial_state.copy()

    if condition["max_rounds"] == 0 and not condition["use_task_match"]:
        # C0: WBS 생성만
        from agents.wbs_gen_agent import wbs_gen_node
        result = wbs_gen_node(initial_state)
        final_state.update(result)

    elif condition["max_rounds"] == 0 and condition["use_task_match"]:
        # C1: WBS 생성 + Task Manager
        from agents.wbs_gen_agent import wbs_gen_node
        from agents.supervisor_agent import supervisor_task_match
        result = wbs_gen_node(initial_state)
        final_state.update(result)
        result2 = supervisor_task_match(final_state)
        final_state.update(result2)

    else:
        # C2/C3/C4: Full pipeline
        from orchestration.debate_loop import execute_sympo_flow
        for current_state in execute_sympo_flow(initial_state, condition["max_rounds"]):
            final_state = current_state

    elapsed = round(time.time() - start_time, 2)

    # ── 메트릭 계산 ──
    from metrics import compute_all_metrics
    from agents.harness import AgentHarness
    resolved_harness = AgentHarness.is_enabled(
        state=final_state, override=condition.get("harness_enabled")
    )
    exp_config = {
        "condition": condition_key,
        "condition_label": condition["label"],
        "backend": backend,
        "run_id": run_id,
        "min_rounds": condition["min_rounds"],
        "max_rounds": condition["max_rounds"],
        "use_disc": condition["use_disc"],
        "use_task_match": condition["use_task_match"],
        "rag_strategy": condition.get("rag_strategy", ""),
        "use_meeting": condition.get("use_meeting", False),
        "harness_enabled": bool(resolved_harness),  # rev.2: RQ1-H 독립변수
        "elapsed_sec": elapsed,
        "note": condition["description"],
        "model_config": condition.get("model_config", {}),
    }

    # ── 하네스 관측치 집계 (rev.2) ──
    harness_caught = sum(
        1 for m in final_state.get("debate_log", [])
        if "하네스 포착 예외" in str(getattr(m, "message", ""))
    )
    # role_drift_detected는 하위 노드 반환값에 플래그로 기록됨 — state 레벨로 끌어올릴 수 없을 경우 0 유지
    role_drift_count = int(final_state.get("role_drift_detected_count", 0))

    output_dir = os.path.join(os.path.dirname(__file__), "..", "eval_results")
    metrics = compute_all_metrics(
        final_state=final_state,
        prd=prd,
        team_members=team_members,
        output_dir=output_dir,
        experiment_config=exp_config,
    )
    metrics["elapsed_sec"] = elapsed
    metrics["harness_observability"] = {
        "harness_enabled": bool(resolved_harness),
        "harness_caught_exceptions": int(harness_caught),
        "role_drift_detected_count": int(role_drift_count),
    }

    # ── LLM-as-a-Judge 평가 (scalar 또는 G-Eval, 선택적 Claude 교차심사) ──
    try:
        from eval.llm_judge import evaluate_wbs
        wbs_tasks = final_state.get("current_wbs_draft", [])
        debate_log = final_state.get("debate_log", [])
        eval_dims = condition.get("eval_dims", ["structure", "assignment", "debate"])
        cross_judge_flag = bool(condition.get("cross_judge", False))
        judge_result = evaluate_wbs(
            wbs_tasks, team_members, debate_log,
            eval_dims=eval_dims,
            cross_judge=cross_judge_flag,
            judge_method=condition.get("judge_method"),
        )
        metrics["llm_judge"] = judge_result
    except Exception as e:
        print(f"  [Judge] 평가 실패 (무시): {e}")
        metrics["llm_judge"] = {"overall": -1, "error": str(e)}

    # ── WBS 스냅샷 저장 (재평가용) ──
    try:
        snapshot = {
            "condition": condition_key,
            "backend": backend,
            "model_config": condition.get("model_config", {}),
            "run_id": run_id,
            "wbs_tasks": [
                {
                    "task_id": getattr(t, "task_id", ""),
                    "title": getattr(t, "title", ""),
                    "level": getattr(t.level, "value", str(t.level)) if hasattr(t, "level") else "",
                    "parent_id": getattr(t, "parent_id", ""),
                    "estimated_days": getattr(t, "estimated_days", 0),
                    "buffer_days": getattr(t, "buffer_days", 0),
                    "assigned_role": getattr(t, "assigned_role", ""),
                    "assigned_to": getattr(t, "assigned_to", []),
                    "dependencies": getattr(t, "dependencies", []),
                }
                for t in final_state.get("current_wbs_draft", [])
            ],
            "team_members": [
                {
                    "member_id": getattr(m, "member_id", ""),
                    "name": getattr(m, "name", ""),
                    "role": getattr(getattr(m, "role", None), "value", str(getattr(m, "role", ""))) if getattr(m, "role", None) else "",
                    "years_of_experience": getattr(m, "years_of_experience", 0),
                    "tech_stack": getattr(m, "tech_stack", []),
                    "strengths": getattr(m, "strengths", []),
                    "weaknesses": getattr(m, "weaknesses", []),
                }
                for m in team_members
            ],
            "debate_log": [
                {
                    "agent_name": getattr(m, "agent_name", ""),
                    "agent_role": str(getattr(m.agent_role, "value", m.agent_role)) if hasattr(m, "agent_role") else "",
                    "message": getattr(m, "message", "")[:300],
                }
                for m in final_state.get("debate_log", [])
            ],
            "llm_judge": metrics.get("llm_judge", {}),
            "wbs_repair_stats": final_state.get("wbs_repair_stats", {}),
        }
        # 팀 배포 시 파일 충돌 방지: timestamp + 실행자 식별자(USER env 또는 RUNNER_ID)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        runner_id = os.environ.get("RUNNER_ID") or os.environ.get("USER", "anon")
        snap_path = os.path.join(
            output_dir,
            f"wbs_snapshot_{condition_key}_r{run_id}_{backend}_{runner_id}_{ts}.json",
        )
        with open(snap_path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, ensure_ascii=False, indent=2, default=str)
    except Exception:
        pass

    return metrics


def run_all_experiments(
    backend: str = "mock",
    runs_per_condition: int = 3,
    conditions: List[str] = None,
    harness_settings: List[bool] = None,
    cross_judge: bool = False,
    judge_method: str = "geval",
):
    """전체 실험 매트릭스 실행.

    harness_settings: [True]/[False]/[True, False] — RQ1-H 실험 독립변수.
                      None이면 CONDITIONS의 harness_enabled를 그대로 사용(=env 기본).
    cross_judge: True면 Claude 교차심사 추가 호출 (비용 2배 → 신뢰성 검증용).
    judge_method: scalar(기존) 또는 geval(G-Eval 확률가중, logprobs 지원 judge).
    """
    # PRD는 조건별로 다시 로드 (R1 prd_variant 반영)
    default_prd = _load_sample_prd()
    team = _load_sample_team()
    disc = _load_disc_profiles()

    if conditions is None:
        conditions = list(CONDITIONS.keys())

    output_dir = os.path.join(os.path.dirname(__file__), "..", "eval_results")
    os.makedirs(output_dir, exist_ok=True)

    # 하네스 설정 루프 (RQ1-H)
    harness_loop = harness_settings if harness_settings is not None else [None]

    all_results = []
    total = len(conditions) * runs_per_condition * len(harness_loop)
    current = 0

    for harness_set in harness_loop:
        for cond_key in conditions:
            cond = dict(CONDITIONS[cond_key])
            cond["cross_judge"] = cross_judge
            cond["judge_method"] = judge_method
            if harness_set is not None:
                cond["harness_enabled"] = harness_set
                cond["label"] = f"{cond['label']} [H{'1' if harness_set else '0'}]"
            # R1: prd_variant 반영 — 조건별로 다른 PRD 파일 로드
            prd = _load_sample_prd(cond.get("prd_variant")) if cond.get("prd_variant") else default_prd
            for run_id in range(1, runs_per_condition + 1):
                current += 1
                print(f"\n{'='*60}")
                print(f"[{current}/{total}] {cond['label']} — Run {run_id}/{runs_per_condition}")
                print(f"{'='*60}")

                try:
                    result = run_single_experiment(
                        cond_key, cond, prd, team, disc, backend, run_id,
                    )
                    all_results.append(result)
                    print(f"  ✅ 완료 ({result.get('elapsed_sec', '?')}s, {result.get('total_tasks', 0)} tasks)")
                except Exception as e:
                    print(f"  ❌ 실패: {e}")
                    import traceback
                    traceback.print_exc()
                    all_results.append({
                        "experiment_config": {
                            "condition": cond_key,
                            "run_id": run_id,
                            "backend": backend,
                            "error": str(e),
                        },
                        "total_tasks": 0,
                    })

    # 결과 저장 (팀 배포 시 파일 충돌 방지: runner_id 포함)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runner_id = os.environ.get("RUNNER_ID") or os.environ.get("USER", "anon")
    result_path = os.path.join(output_dir, f"experiment_{backend}_{runner_id}_{timestamp}.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n📊 결과 저장: {result_path}")

    # 요약 CSV
    _save_summary_csv(all_results, output_dir, backend, timestamp, runner_id)

    return all_results


def _save_summary_csv(results: list, output_dir: str, backend: str, timestamp: str, runner_id: str = "anon"):
    """실험 결과를 조건별 요약 CSV로 저장"""
    csv_path = os.path.join(output_dir, f"summary_{backend}_{runner_id}_{timestamp}.csv")

    rows = []
    for r in results:
        cfg = r.get("experiment_config", {})
        hobs = r.get("harness_observability", {})
        judge = r.get("llm_judge", {})
        row = {
            "runner_id": runner_id,
            "condition": cfg.get("condition", ""),
            "label": cfg.get("condition_label", ""),
            "run_id": cfg.get("run_id", 0),
            "backend": cfg.get("backend", ""),
            "model_config": json.dumps(cfg.get("model_config", {}), ensure_ascii=False),
            "elapsed_sec": r.get("elapsed_sec", 0),
            "total_tasks": r.get("total_tasks", 0),
            "debate_rounds": r.get("debate_rounds", 0),
            # rev.2 실험 독립변수 · 재현성 필드
            "min_rounds": cfg.get("min_rounds", ""),
            "max_rounds": cfg.get("max_rounds", ""),
            "harness_enabled": cfg.get("harness_enabled", ""),
            "rag_strategy": cfg.get("rag_strategy", ""),
            "use_disc": cfg.get("use_disc", ""),
            "use_task_match": cfg.get("use_task_match", ""),
            "use_meeting": cfg.get("use_meeting", ""),
            # rev.2 하네스 관측치
            "harness_caught_exceptions": hobs.get("harness_caught_exceptions", 0),
            "role_drift_detected_count": hobs.get("role_drift_detected_count", 0),
            # 13 metrics (N/A는 -1로 기록, analyze_results가 필터링)
            "faithfulness": r.get("ragas_faithfulness", {}).get("faithfulness", 0),
            "faithfulness_method": r.get("ragas_faithfulness", {}).get("method", ""),
            "success_rate": r.get("success_rate", {}).get("success_rate", 0),
            "planning_score": r.get("planning_score", {}).get("planning_score", 0),
            "buffer_ratio_pct": r.get("buffer_ratio", {}).get("buffer_ratio_pct", 0),
            "interaction_turns": r.get("interaction_turns", {}).get("total_messages", 0),
            "supervisor_ratio": r.get("supervisor_intervention", {}).get("intervention_ratio", 0),
            "convergence": r.get("convergence", {}).get("is_converging", False),
            "mece_score": r.get("mece_score", {}).get("mece_score", 0),
            "granularity_fitness": r.get("granularity_fitness", {}).get("granularity_fitness", 0),
            "workload_gini": r.get("workload_gini", {}).get("gini", 0),
            "schedule_feasibility": r.get("schedule_feasibility", {}).get("feasibility", 0),
            "comm_efficiency": r.get("communication_efficiency", {}).get("efficiency", 0),
            "est_tokens": r.get("token_cost", {}).get("est_total_tokens", 0),
            "est_cost_usd": r.get("token_cost", {}).get("est_cost_usd", 0),
            # rev.2: AutoScore 종합 점수 (eval2 §4)
            "autoscore_final":         r.get("autoscore", {}).get("autoscore", 0),
            "autoscore_quality":       r.get("autoscore", {}).get("quality", 0),
            "autoscore_allocation":    r.get("autoscore", {}).get("allocation", 0),
            "autoscore_orchestration": r.get("autoscore", {}).get("orchestration", 0),
            "autoscore_na_cats":       ",".join(r.get("autoscore", {}).get("na_categories", []) or []),
            # LLM Judge (primary)
            "judge_model": judge.get("judge_model", ""),
            "judge_provider": judge.get("judge_provider", ""),
            "judge_method": judge.get("judge_method", ""),
            "judge_geval_backend": next(
                (judge.get(d, {}).get("geval_backend", "") for d in ("structure", "assignment", "debate")
                 if judge.get(d, {}).get("geval_backend", "")),
                "",
            ),
            "judge_geval_model": next(
                (judge.get(d, {}).get("geval_model", "") for d in ("structure", "assignment", "debate")
                 if judge.get(d, {}).get("geval_model", "")),
                "",
            ),
            "judge_geval_logprobs": ",".join(
                f"{d}:{judge.get(d, {}).get('geval_logprobs_available', '')}"
                for d in ("structure", "assignment", "debate")
                if "geval_logprobs_available" in judge.get(d, {})
            ),
            "judge_structure": judge.get("structure", {}).get("score", ""),
            "judge_assignment": judge.get("assignment", {}).get("score", ""),
            "judge_debate": judge.get("debate", {}).get("score", ""),
            "judge_overall": judge.get("overall", ""),
            # LLM Judge (cross-judge, rev.2 스펙)
            "judge2_model": judge.get("cross", {}).get("judge_model", ""),
            "judge2_structure": judge.get("cross", {}).get("structure", {}).get("score", ""),
            "judge2_assignment": judge.get("cross", {}).get("assignment", {}).get("score", ""),
            "judge2_debate": judge.get("cross", {}).get("debate", {}).get("score", ""),
            "judge2_overall": judge.get("cross", {}).get("overall", ""),
        }
        rows.append(row)

    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"📋 요약 CSV: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="symPO Ablation Study Runner")
    parser.add_argument("--backend", default="mock",
                        choices=["mock", "gemini", "openai", "gemma4", "gemma4-api", "qwen-api", "anthropic", "ollama"])
    parser.add_argument("--runs", type=int, default=3, help="조건당 반복 실행 횟수")
    parser.add_argument("--conditions", nargs="*", default=None,
                        help="실행할 조건 (예: C0_llm_only C3_3rounds)")
    parser.add_argument("--harness", default="default",
                        choices=["default", "on", "off", "both"],
                        help="하네스 토글. both=H0/H1 각각 반복 실행 (RQ1-H).")
    parser.add_argument("--cross-judge", action="store_true",
                        help="Claude 교차심사 활성화 (비용 2배, 신뢰성 검증용).")
    parser.add_argument("--judge-method", default=os.getenv("JUDGE_METHOD", "geval"), choices=["scalar", "geval"],
                        help="scalar=기존 JSON 점수, geval=G-Eval 확률가중 점수(logprobs 지원 judge).")
    args = parser.parse_args()

    _harness_map = {
        "default": None,
        "on":  [True],
        "off": [False],
        "both": [False, True],   # H0 먼저 → baseline 확보 후 H1
    }

    run_all_experiments(
        backend=args.backend,
        runs_per_condition=args.runs,
        conditions=args.conditions,
        harness_settings=_harness_map[args.harness],
        cross_judge=args.cross_judge,
        judge_method=args.judge_method,
    )
