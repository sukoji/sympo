"""
LangGraph 공유 상태(State) 정의
모든 에이전트가 읽고 쓰는 중앙 상태 그래프 객체입니다.
"""
from typing import List, Optional, Dict, Any, Annotated
from typing_extensions import TypedDict
import operator

from schemas.prd_schema import PRDInput
from schemas.member_schema import MemberProfile
from schemas.wbs_schema import WBSTask, DebateMessage


class WBSState(TypedDict):
    """
    symPO 공유 상태 그래프
    LangGraph StateGraph의 중앙 상태 객체
    """
    # ─── 입력 데이터 ───────────────────────────────────
    prd: PRDInput                              # 프로젝트 요구사항
    team_members: List[MemberProfile]          # 팀원 프로필 목록
    agent_personas: Dict[str, str]             # {member_id: persona_prompt}

    # ─── 현재 WBS 초안 ────────────────────────────────
    current_wbs_draft: List[WBSTask]           # 현재 WBS 초안
    proposed_tasks: List[Dict[str, Any]]       # 에이전트 제안 태스크 목록

    # ─── 토론 로그 (누적 추가) ─────────────────────────
    debate_log: Annotated[List[DebateMessage], operator.add]
    mcp_tool_trace: Annotated[List[Dict[str, Any]], operator.add]  # MCP-compatible tool call trace

    # ─── RAG 컨텍스트 ─────────────────────────────────
    rag_reference_wbs: List[str]               # 참고 WBS 검색 결과
    rag_meeting_logs: List[str]                # 참고 회의록 검색 결과
    disc_profiles: Dict[str, str]              # {이름: to_agent_context() 문자열} — eDISC 행동유형 컨텍스트

    # ─── 오케스트레이션 제어 ──────────────────────────
    current_round: int                         # 현재 토론 라운드 (0부터 시작)
    min_rounds: int                            # 최소 보장 토론 라운드 수
    max_rounds: int                            # 최대 토론 라운드 수
    consensus_reached: bool                    # 합의 도달 여부
    pending_conflicts: List[str]               # 미해결 갈등 목록

    # ─── 태스크 할당 및 Context (VFS 역할) ───────────────
    assigned_tasks: Dict[str, str]             # Task ID -> Agent 역할
    called_agents: List[str]                   # 현재 호출된 에이전트들의 역할
    calling_context: Dict[str, str]            # {agent_role_node: member_id} (동적 매핑용)
    l2_candidate_pools: Dict[str, List[str]]   # {l2_task_id: [member_id]} — L2별 후보 풀
    task_candidate_pools: Dict[str, List[str]] # {task_id: [member_id]} — L3별 후보 풀
    l2_review_lenses: Dict[str, List[str]]     # {l2_task_id: [capability_label]} — L2별 검토 관점(자유 라벨)
    l2_calling_context: Dict[str, Dict[str, str]] # {l2_task_id: {agent_role_node: member_id}} — L2별 동적 발화자
    assignment_evidence: Dict[str, str]        # {task_id: assignment rationale} — 후보/배정 근거
    supervisor_rationale: Optional[str]        # 에이전트 호출 사유
    member_role_map: Dict[str, str]            # {member_id: role_value} — LLM 비노출 내부 라우팅 전용

    # ─── 에이전트 응답 (임시) ─────────────────────────
    supervisor_proposal: Optional[str]         # 슈퍼바이저 제안
    planner_response: Optional[str]            # 플래너 응답
    frontend_response: Optional[str]           # FE 개발자 응답
    backend_response: Optional[str]            # BE 개발자 응답
    designer_response: Optional[str]           # 디자이너 응답
    qa_response: Optional[str]                 # QA 응답

    # ─── WBS 구조 재설계 제어 ─────────────────────────
    wbs_revision_needed: bool                  # 구조적 WBS 재설계 필요 여부 (버퍼 조정과 구분)
    wbs_revision_hints: List[str]              # 재설계 시 반영할 구체적 지침
    current_wbs_revision: int                  # 현재까지 WBS 재생성 횟수
    # ─── 수렴 모니터링 ──────────────────────────
    total_days_history: List[float]             # 각 라운드별 전체 일정 합계 (수렴 체크용)

    # ─── 누적 중재 카운트 (finalize 요약용) ────────
    # 매 mediate 라운드마다 가산되어, finalize에서 "총 N건 재배정/버퍼 반영" 표시 시 사용.
    # 단일 라운드 count만 표시하면 마지막 라운드가 비어 있을 때 0/0으로 오해하게 됨.
    cumulative_reassignments: int               # 누적 재배정 건수
    cumulative_buffers_applied: int             # 누적 버퍼 조정 태스크 수
    cumulative_new_tasks: int                   # 누적 추가 태스크 수

    # ─── 최종 출력 ────────────────────────────────────
    final_wbs: Optional[List[WBSTask]]         # 최종 확정 WBS
    generation_summary: Optional[str]          # 생성 요약문
    member_project_roles: List[Dict[str, Any]] # 최종 배정 태스크 기반 팀원별 프로젝트 역할
    # ─── PM 결정 잠금 (롤백 방지) ────────────────
    locked_assignments: Dict[str, str]          # {task_id: member_id} — PM이 확정한 배정은 재배정 불가

    # ─── L2 태스크별 에이전트 매핑 ──────────────────
    l2_agent_mapping: Dict[str, List[str]]     # {l2_task_id: [agent_roles]} — L2별 호출 에이전트

    # ─── 모델 설정 ──────────────────────────────────
    model_config: Optional[Dict[str, str]]     # {node_name: model_id/backend}


    # ─── WBS 재설계 제어 (create_initial_state에서 설정) ──
    max_wbs_revisions: int                     # WBS 구조 재생성 최대 횟수
    error_message: Optional[str]               # 실행 중 발생한 오류 메시지

    # ─── 실험 독립변수 ─────────────────────────────
    harness_enabled: Optional[bool]            # 하네스 적용 여부 (RQ1-H 변수). None/미지정 시 env HARNESS_ENABLED 기본값
    max_free_turns: Optional[int]              # 자유 토론 최대 턴 (외부화, 기본 3)
    veto_enabled: Optional[bool]               # A2: 에이전트 버퍼 제안 거부권 허용 여부 (기본 False — PM 집계, True — 단일 에이전트 거부권)
    critic_enabled: Optional[bool]             # A1: 크리틱 교차 심사 에이전트 활성화 여부
    persona_strictness: Optional[str]          # A3: 페르소나 엄격도 ("defensive" | "aggressive" | "neutral")
    prd_variant: Optional[str]                 # R1: PRD 변형 ("summary" | "detailed" | "detailed_meeting")
    model_class: Optional[str]                 # S2: 모델 급 ("frontier" | "8b")
    prompting_strategy: Optional[str]          # S2: 프롬프팅 ("single" | "chaining" | "cot")

    # ─── 내부 제어 (LangGraph 전용) ───────────────
    _free_turn_count: Optional[int]            # 자유 토론 현재 턴
    _anyone_spoke_in_free: Optional[bool]      # 자유 토론 중 실질 발언 발생 여부
    _current_l2_task_id: Optional[str]         # 현재 토론 중인 L2 태스크 ID
    _current_agent_acting: Optional[str]       # 현재 활성 에이전트 레이블 (UI 표시용)
    _l2_debate_cutoff: Optional[bool]          # PM 중간 개입으로 L2 토론 조기 종료 플래그


def create_initial_state(
    prd: PRDInput,
    team_members: List[MemberProfile],
    agent_personas: Dict[str, str],
    min_rounds: int = 2,
    max_rounds: int = 3,
    max_wbs_revisions: int = 1,
    model_config: Dict[str, str] = None,
    harness_enabled: Optional[bool] = None,
    max_free_turns: int = 3,
    veto_enabled: Optional[bool] = None,
    critic_enabled: Optional[bool] = None,
    persona_strictness: Optional[str] = None,
    prd_variant: Optional[str] = None,
    model_class: Optional[str] = None,
    prompting_strategy: Optional[str] = None,
) -> WBSState:
    """
    초기 상태 생성 헬퍼.
    team_members의 role 정보를 member_role_map으로 분리 저장하고,
    state에 저장되는 team_members에서는 role을 None으로 제거합니다.
    LangSmith 등 모니터링 툴에서 WBS Gen 노드 입력에 role이 노출되지 않습니다.
    """
    # role 추출 → 내부 라우팅 맵
    member_role_map = {
        m.member_id: m.role.value
        for m in team_members
        if m.role is not None
    }
    # team_members에서 role 제거 (LLM 비노출)
    stripped_members = [m.model_copy(update={"role": None}) for m in team_members]

    return WBSState(
        prd=prd,
        team_members=stripped_members,
        agent_personas=agent_personas,
        member_role_map=member_role_map,
        current_wbs_draft=[],
        proposed_tasks=[],
        debate_log=[],
        mcp_tool_trace=[],
        rag_reference_wbs=[],
        rag_meeting_logs=[],
        disc_profiles={},
        current_round=0,
        min_rounds=min_rounds,
        max_rounds=max_rounds,
        consensus_reached=False,
        pending_conflicts=[],
        assigned_tasks={},
        called_agents=[],
        calling_context={},
        l2_candidate_pools={},
        task_candidate_pools={},
        l2_review_lenses={},
        l2_calling_context={},
        assignment_evidence={},
        locked_assignments={},
        l2_agent_mapping={},
        model_config=model_config or {},
        supervisor_rationale=None,
        supervisor_proposal=None,
        planner_response=None,
        frontend_response=None,
        backend_response=None,
        designer_response=None,
        qa_response=None,
        final_wbs=None,
        generation_summary=None,
        member_project_roles=[],
        error_message=None,
        wbs_revision_needed=False,
        wbs_revision_hints=[],
        current_wbs_revision=0,
        max_wbs_revisions=max_wbs_revisions,
        _free_turn_count=0,
        _anyone_spoke_in_free=False,
        _current_l2_task_id=None,
        total_days_history=[],
        cumulative_reassignments=0,
        cumulative_buffers_applied=0,
        cumulative_new_tasks=0,
        harness_enabled=harness_enabled,
        max_free_turns=max_free_turns,
        veto_enabled=veto_enabled,
        critic_enabled=critic_enabled,
        persona_strictness=persona_strictness,
        prd_variant=prd_variant,
        model_class=model_class,
        prompting_strategy=prompting_strategy,
    )
