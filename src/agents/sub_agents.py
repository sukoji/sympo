"""
서브 에이전트 모듈 (플래너, FE, BE, 디자이너, QA)
각 에이전트는 주입된 페르소나 기반으로 WBS 초안에 검토 의견을 제시합니다.
"""
from datetime import datetime
from typing import List, Optional, Any
from langsmith import traceable

from agents.state import WBSState
from agents.llm_config import get_llm, normalize_content
from schemas.wbs_schema import DebateMessage, AgentRole, WBSTask


def _profile_text(member, role_value: str = "") -> str:
    if not member:
        return ""
    parts = [
        role_value,
        " ".join(getattr(member, "tech_stack", []) or []),
        " ".join(getattr(member, "primary_skills", []) or []),
        " ".join(getattr(member, "strengths", []) or []),
        getattr(member, "raw_resume_text", "") or "",
    ]
    return " ".join(str(p) for p in parts if p).lower()


def _lens_member_score(member, lens: str, role_value: str = "") -> float:
    keywords = {
        "planner": ["plan", "pm", "기획", "전략", "요구사항", "분석", "business", "strategy"],
        "frontend": ["front", "react", "vue", "ui", "ux", "mobile", "앱", "모바일", "고객"],
        "backend": ["back", "api", "server", "db", "data", "데이터", "pipeline", "infra", "시스템"],
        "designer": ["design", "figma", "ux", "brand", "marketing", "디자인", "브랜드", "마케팅"],
        "qa": ["qa", "test", "검증", "테스트", "quality", "보안", "성과", "측정"],
    }.get((lens or "").lower(), [str(lens or "").lower()])
    text = _profile_text(member, role_value)
    return sum(1.0 for kw in keywords if kw and kw.lower() in text) + min(float(getattr(member, "years_of_experience", 0) or 0), 10.0) * 0.05


def _current_l2_context(state: WBSState) -> dict:
    current_l2 = state.get("_current_l2_task_id")
    if not current_l2:
        return {}
    return (state.get("l2_calling_context") or {}).get(current_l2, {}) or {}


def _get_member_by_role(state: WBSState, role_keyword: str, node_name: str = None):
    """
    역할 키워드 또는 할당된 컨텍스트로 팀원 찾기.
    비즈니스 팀원이 포함된 경우에도 항상 유효한 멤버를 반환합니다.
    """
    team = state.get("team_members", [])
    ctx = state.get("calling_context") or {}
    l2_ctx = _current_l2_context(state)

    # 1. 현재 L2에 대해 supervisor가 동적으로 지정한 후보 발화자를 최우선 사용
    if node_name:
        assigned_mid = l2_ctx.get(node_name)
        if assigned_mid:
            for m in team:
                if m.member_id == assigned_mid:
                    return m

    # 2. 현재 L2 후보 풀 안에서 이 검토 렌즈에 가장 가까운 멤버 선택
    current_l2 = state.get("_current_l2_task_id")
    if current_l2:
        role_map = state.get("member_role_map") or {}
        pool_ids = (state.get("l2_candidate_pools") or {}).get(current_l2, [])
        pool_members = [m for m in team if m.member_id in set(pool_ids)]
        if pool_members:
            lens = node_name or role_keyword
            return max(
                pool_members,
                key=lambda m: _lens_member_score(m, lens, role_map.get(m.member_id, "") or (m.role.value if m.role else "")),
            )

    # 3. calling_context에 명시된 member_id fallback
    if node_name:
        assigned_mid = ctx.get(node_name)
        if assigned_mid:
            for m in team:
                if m.member_id == assigned_mid:
                    return m

    # 4. member_role_map 기반 역할 키워드 매칭 (state에서 role이 strip된 경우 대비)
    role_map = state.get("member_role_map") or {}
    for m in team:
        role_value = role_map.get(m.member_id, "") or (m.role.value if m.role else "")
        if role_keyword.lower() in role_value.lower():
            return m

    # 5. calling_context에 있는 다른 멤버들 중에서 중복되지 않게 선택 (Monopoly 방지)
    used_mids = set(ctx.values())
    for m in team:
        if m.member_id not in used_mids:
            return m

    # 6. 최후 fallback: 팀 내 멤버 중 골고루 선택 (해시 기반 분산)
    if team:
        idx = abs(hash(role_keyword)) % len(team)
        return team[idx]
    return None


def _infer_project_type(state: WBSState) -> str:
    """state의 PRD로부터 프로젝트 유형 추론 ('dev' | 'business')"""
    prd = state.get("prd")
    if not prd:
        return "dev"
    combined = " ".join([
        prd.project_goal or "",
        " ".join(prd.key_features or []),
        " ".join(prd.tech_stack_requirements or []),
    ]).lower()
    biz_keywords = [
        "마케팅", "프로모션", "매출", "매장", "빅데이터 분석", "고객 분석",
        "브랜딩", "소매", "유기농", "마트", "판매전략", "경영지원",
        "crm", "캠페인", "시장조사", "물류", "재고",
    ]
    dev_keywords = [
        "api", "서버", "배포", "docker", "kubernetes", "fastapi",
        "react", "vue", "django", "spring", "ci/cd",
    ]
    biz_score = sum(1 for kw in biz_keywords if kw in combined)
    dev_score = sum(1 for kw in dev_keywords if kw in combined)
    return "business" if biz_score > dev_score else "dev"


def _get_persona(state: WBSState, member_id: str) -> str:
    """
    멤버 ID로 페르소나 프롬프트 조회.
    state["persona_strictness"]에 따라 행동 지시를 추가:
      - "defensive"  : 방어적 — 리스크를 과장하고 버퍼를 보수적으로 제안
      - "aggressive" : 공격적 — 일정을 단축하고 새로운 도전 과제를 적극 발굴
      - 그 외 (neutral/None): 기본 페르소나 그대로
    """
    base = (state.get("agent_personas") or {}).get(member_id, "")
    strictness = (state.get("persona_strictness") or "neutral").lower()
    if strictness == "defensive":
        return base + (
            "\n\n[행동 지침 — 방어적] 리스크를 과장하여 보고하라. "
            "일정이 빠듯해 보이면 반드시 추가 버퍼를 요청하고, 의존성·장애·보안 이슈를 적극적으로 제기하라. "
            "'안전'과 '확실성'을 '속도'보다 우선한다."
        )
    if strictness == "aggressive":
        return base + (
            "\n\n[행동 지침 — 공격적] 일정을 단축할 여지를 찾아 제안하라. "
            "버퍼 추가 요청을 최소화하고, 병렬 처리·재사용·자동화로 기존 일정 내 소화할 방법을 우선 검토하라. "
            "'속도'와 '성과'를 '안전'보다 우선한다."
        )
    return base


def _get_member_tasks(state: WBSState, member_id: str) -> List[WBSTask]:
    """해당 멤버에게 배정된 태스크 목록 반환.
    _current_l2_task_id가 설정되어 있으면 해당 L2의 자식 L3만 반환."""
    if not member_id:
        return []
    all_tasks = state.get("current_wbs_draft") or []
    current_l2 = state.get("_current_l2_task_id")
    if current_l2:
        # 현재 토론 중인 L2의 직접 자식 L3 태스크만
        tasks_in_l2 = [
            t for t in all_tasks
            if (t.parent_id == current_l2 or t.task_id.startswith(current_l2.replace("L2", "L3")))
            and member_id in (t.assigned_to or [])
        ]
        # 해당 L2 태스크 아래 내 배정이 없으면 해당 L2 전체를 fallback으로
        if not tasks_in_l2:
            tasks_in_l2 = [
                t for t in all_tasks
                if t.parent_id == current_l2
            ]
        return tasks_in_l2
    return [
        t for t in all_tasks
        if member_id in (t.assigned_to or [])
    ]


def _format_member_tasks(tasks: List[WBSTask]) -> str:
    """멤버 배정 태스크를 프롬프트용 텍스트로 변환"""
    if not tasks:
        return "(배정된 태스크 없음)"
    lines = []
    for t in tasks:
        lines.append(
            f"- [{t.task_id}] {t.title} "
            f"({t.estimated_days}일, {t.assigned_role}, Level: {t.level.value})"
        )
    return "\n".join(lines)


def _format_member_profile(member) -> str:
    """멤버 프로필을 프롬프트용 텍스트로 변환"""
    if not member:
        return "프로필 없음"
    return (
        f"경력 {member.years_of_experience}년, 기술: {', '.join(member.tech_stack[:4])}, "
        f"강점: {', '.join(member.strengths) if isinstance(member.strengths, list) else member.strengths}, "
        f"약점: {', '.join(member.weaknesses) if isinstance(member.weaknesses, list) else member.weaknesses}"
    )


def _format_team_summary(state: WBSState, exclude_member_id: str = None) -> str:
    """전체 팀원 요약 (재배정 건의용)"""
    team = state.get("team_members", [])
    if not team:
        return "(팀 정보 없음)"
    lines = []
    for m in team:
        marker = " ← 나" if m.member_id == exclude_member_id else ""
        lines.append(
            f"- {m.name} ({m.member_id}): "
            f"경력 {m.years_of_experience}년, "
            f"기술: {', '.join(m.tech_stack[:3])}, "
            f"강점: {', '.join(m.strengths[:2]) if isinstance(m.strengths, list) else str(m.strengths)[:30]}"
            f"{marker}"
        )
    return "\n".join(lines)


def _get_active_roster(state: WBSState, exclude_member_id: str = None) -> tuple:
    """현재 세션 참여 중인 에이전트 명단 반환.
    Returns: (roster_block_str, active_names_set)
    현재 L2의 동적 후보 발화자, 없으면 calling_context에 있는 member_id만 '참여자'로 간주.
    """
    team = state.get("team_members", [])
    calling_context = _current_l2_context(state) or (state.get("calling_context") or {})
    active_mids = set(calling_context.values())
    if not active_mids:
        return "(참여자 명단 없음 — 전체 팀원이 참여 중)", {m.name for m in team}

    id_to_member = {m.member_id: m for m in team}
    # role_key (planner/frontend/...) → member 매핑 역순 조회
    role_labels_ko = {
        "planner": "플래너",
        "frontend": "FE/고객 접점",
        "backend": "BE/데이터 분석",
        "designer": "디자이너/브랜딩",
        "qa": "QA/성과 검증",
    }
    current_l2 = state.get("_current_l2_task_id")
    header = f"[이번 L2({current_l2}) 검토 후보 명단 — 이 사람들에게만 업무 협업을 제안 가능]" if current_l2 else "[이번 세션 참여자 명단 — 이 사람들에게만 업무 협업을 제안 가능]"
    lines = [header]
    active_names = set()
    for role_key, mid in calling_context.items():
        m = id_to_member.get(mid)
        if not m:
            continue
        active_names.add(m.name)
        marker = " ← 나" if m.member_id == exclude_member_id else ""
        role_label = role_labels_ko.get(role_key, role_key)
        lines.append(
            f"- {m.name} ({m.member_id}) / {role_label} — 경력 {m.years_of_experience}년, "
            f"기술: {', '.join(m.tech_stack[:3])}{marker}"
        )
    lines.append(
        "⛔ 위 명단에 없는 팀원의 이름을 절대 언급하지 마세요. "
        "세션에 없는 인원에게 업무를 넘기려고 하면 PM이 차단합니다."
    )
    return "\n".join(lines), active_names


def _build_pm_directive(state: WBSState) -> str:
    """최근 PM 중재/결정 메시지에서 하드 제약을 추출해 System Directive 블록 생성.
    LLM이 PM 경고를 무시하고 동일 행동을 반복하지 못하도록 강제 주입."""
    logs = state.get("debate_log", []) or []
    recent_pm = [
        m for m in logs[-8:]
        if m.message_type in ("mediation", "decision")
        and ("SUPERVISOR" in str(m.agent_role) or "PM" in str(m.agent_name) or "슈퍼바이저" in str(m.agent_role))
    ]
    if not recent_pm:
        return ""

    directives = []
    last_msgs = [m.message for m in recent_pm[-3:]]
    combined = " ".join(last_msgs)

    # 패턴별 하드 제약 매핑
    if "범위 과잉" in combined or "마이크로 태스크" in combined or "0.5일" in combined:
        directives.append(
            "일정 연장/버퍼 추가 제안은 절대 금지. 0.5일 단위 태스크 분할도 금지. "
            "대신 기존 일정 내에서 병렬 처리·재사용·자동화로 소화할 기술적 대안만 제시."
        )
    if "반복적으로 차감" in combined or "일정 보호" in combined:
        directives.append(
            "특정 태스크에서 일정을 빼서 다른 태스크에 돌려쓰는 제안 금지. "
            "전체 예산 조율은 PM이 단일 진입점에서만 수행."
        )
    if "발언자" in combined and "넘깁니다" in combined:
        directives.append(
            "직전 2턴 이상 동일 주장을 반복한 내가 맞다면 이번에는 반드시 새로운 관점 또는 [PASS] 선택."
        )
    if "토론 리다이렉트" in combined or "구현 세부사항" in combined:
        directives.append(
            "코드 컨벤션/네이밍/포맷 등 구현 세부사항 토론 금지. "
            "오직 버퍼 산정·R&R 적합성·일정 리스크에만 집중."
        )
    if "역할" in combined and ("사칭" in combined or "맞는 관점" in combined):
        directives.append(
            "이번 턴에 지정된 검토 관점과 본인 프로필 근거 안에서만 발언. 근거 없이 다른 사람의 업무를 대변 금지."
        )
    if "충분한 합의" in combined or "조기 수렴" in combined:
        directives.append(
            "PM이 합의 종료를 선언했음. 이번 턴은 반드시 [PASS] 또는 '특별 이의 없음'으로 마무리."
        )

    if not directives:
        # 일반 폴백: 최근 PM 발언 요지를 그대로 주입
        directives.append(
            f"직전 PM 결정: {recent_pm[-1].message[:200]} — 해당 결정과 충돌하는 제안 금지."
        )

    block = "\n".join(f"  • {d}" for d in directives)
    return (
        "\n═══════════ [System Directive — PM 하드 제약 (위반 시 응답 폐기)] ═══════════\n"
        f"{block}\n"
        "위 지시를 위반하는 응답은 오케스트레이터가 묵살합니다. 이번 턴은 이 제약 안에서만 발언하세요.\n"
        "═════════════════════════════════════════════════════════════════════════════\n"
    )


# PASS 허용 지시어 — 중복 발언 방지용, 모든 서브에이전트 프롬프트 최상단에 부착
_PASS_DIRECTIVE = """
═══ [System Directive — 중복 발언 방지] ═══
현재 논의 중인 L2 태스크가 **당신에게 지정된 검토 관점과 명확히 무관**하거나,
직전 발언에 추가할 근거 있는 검토 의견이 전혀 없다면 아래 한 줄로 턴을 넘기세요.

[PASS] <이름/검토관점>: 본 태스크는 내 검토 관점 밖이거나 추가 이의 없음.

PASS 조건:
- L2 태스크가 내 검토 관점(프론트엔드/백엔드/디자인/QA/플래닝 중 하나)과 **전혀 무관**
- 앞선 2명의 발언자가 이미 내가 제기하려던 요점을 **동일하게** 언급함
- 내가 제안할 기술적 대안이 **일반론**(예: "테스트 잘 하자")뿐이고 구체적 내용 없음
PASS하지 않을 경우에는 아래 발화 구조를 참고하되, 자연스러운 검토 의견으로 작성하세요.
══════════════════════════════════════════════════════════
"""


# 발화 구조 가이드 — 모든 서브에이전트 프롬프트 말미에 부착
_STRUCTURED_FORMAT_INSTRUCTION = """
═══ 발화 구조 가이드 (구조화된 상호 검토) ═══
아래 항목 중 실제로 필요한 내용만 골라 3~5문장으로 작성하세요.
⚠️ 위 [PASS] 조건에 해당하면 의견을 만들지 말고 [PASS] 한 줄만 출력하세요.

- 직전 발언에 대한 짧은 평가: 강점, 누락, 또는 반박 근거 중 하나
- 내 전문 분야에서 보이는 구체적 리스크 또는 기술적 대안
- 필요한 경우에만 일정/버퍼/R&R 조정 요청

섹션 제목을 반드시 붙일 필요는 없습니다. 단순 동의만 반복하지 말고 task ID나 담당 영역을 근거로 말하세요.
"""


def _get_prompting_prefix(state: WBSState) -> str:
    """
    S2 prompting_strategy 플래그에 따라 프롬프트 앞에 붙일 지시문 반환.
    - single    : 빈 문자열 (기본 — 단일 응답)
    - chaining  : 단계별 중간 결론을 먼저 정리
    - cot       : Chain-of-Thought (추론 과정 명시)
    """
    strat = (state.get("prompting_strategy") or "single").lower()
    if strat == "cot":
        return (
            "[추론 지시] 답을 내기 전에 '생각 과정:'으로 시작하여 "
            "2~3개의 핵심 추론 스텝을 번호로 나열한 뒤, '결론:'으로 최종 답을 정리하세요.\n\n"
        )
    if strat == "chaining":
        return (
            "[사슬 지시] 먼저 (A) 핵심 리스크 1개, (B) 이 리스크의 영향 범위, "
            "(C) 완화 방안 순으로 단계별로 답하세요. 각 단계를 `A)` `B)` `C)`로 표기.\n\n"
        )
    return ""


def _format_debate_history(state: WBSState, max_messages: int = 8) -> str:
    """최근 토론 내역 + PM 결정사항 요약 (Bug 2: 메모리 단절 방지)"""
    logs = state.get("debate_log", [])
    if not logs:
        return "아직 토론이 시작되지 않았습니다. 내가 첫 번째 발언자입니다."

    # PM(슈퍼바이저) 결정사항만 별도 추출 (가장 중요 — 재반복 방지)
    pm_decisions = [
        msg for msg in logs
        if msg.message_type in ("mediation", "decision")
        and ("SUPERVISOR" in str(msg.agent_role) or "슈퍼바이저" in str(msg.agent_role) or "PM" in str(msg.agent_name))
    ]
    lines = []
    if pm_decisions:
        lines.append("=== PM 확정 결정사항 (이미 반영됨 — 재요구 불필요) ===")
        for msg in pm_decisions[-4:]:
            lines.append(f"[{msg.agent_name}]: {msg.message}")
        lines.append("=" * 50)

    # 최근 N개 일반 발언
    recent = logs[-max_messages:]
    lines.append("--- 최근 토론 내역 ---")
    for msg in recent:
        lines.append(f"[{msg.agent_name} / {msg.agent_role.value if hasattr(msg.agent_role, 'value') else msg.agent_role}]: {msg.message}")
    lines.append("--------------------")
    return "\n".join(lines)


def _get_l2_context_block(state: WBSState) -> str:
    """현재 토론 중인 L2 태스크가 있으면 집중 범위를 명시하는 컨텍스트 블록 반환."""
    current_l2_id = state.get("_current_l2_task_id")
    if not current_l2_id:
        return ""
    wbs_draft = state.get("current_wbs_draft") or []
    l2_task = next((t for t in wbs_draft if t.task_id == current_l2_id), None)
    l2_title = l2_task.title if l2_task else "알 수 없음"
    importance = (l2_task.importance if l2_task else "Medium")
    return (
        f"\n⚠️ [현재 검토 대상 L2 기능그룹: {current_l2_id} — {l2_title} (중요도: {importance})]\n"
        "이 L2 태스크와 그 하위 L3 태스크들에만 집중하여 의견을 제시하십시오.\n"
        "다른 L2 그룹의 태스크는 이번 턴에서 논의하지 않습니다.\n"
    )


def _get_disc_context(state: WBSState, member_name: str) -> str:
    """팀원 이름으로 eDISC 행동유형 컨텍스트 반환 (없으면 빈 문자열)"""
    disc_profiles = state.get("disc_profiles") or {}
    if not disc_profiles:
        return ""
    # 정확 일치 우선, 그 다음 부분 일치
    import re as _re
    clean_name = _re.sub(r"\s+", "", member_name)
    for name, ctx in disc_profiles.items():
        if _re.sub(r"\s+", "", name) == clean_name:
            return f"\n[eDISC 행동유형 프로파일]\n{ctx}\n"
    for name, ctx in disc_profiles.items():
        cn = _re.sub(r"\s+", "", name)
        if clean_name in cn or cn in clean_name:
            return f"\n[eDISC 행동유형 프로파일]\n{ctx}\n"
    return ""


def _check_pure_agreement(state: WBSState, last_n: int = 3) -> bool:
    """최근 last_n개 메시지가 모두 '동의' 패턴이고 새로운 버퍼/리스크 제안이 없으면 True.
    Supervisor가 조기 수렴 판단에 사용합니다."""
    logs = [
        m for m in state.get("debate_log", [])
        if m.message_type not in ("mediation", "decision", "pass")
    ]
    if len(logs) < last_n:
        return False
    recent = logs[-last_n:]
    agreement_kw = ["동의", "맞습니다", "적절합니다", "타당합니다", "공감", "충분히 반영", "잘 반영", "동의하며"]
    new_risk_kw  = ["위험", "리스크", "지연", "버퍼", "buffer", "반대", "이의", "추가 필요", "누락", "부족"]
    pure_agree_count = sum(
        1 for m in recent
        if any(k in m.message for k in agreement_kw)
        and not any(k in m.message for k in new_risk_kw)
        and not m.buffer_days_proposed
    )
    return pure_agree_count >= last_n - 1  # ≥2/3 이 순수 동의


def _scope_creep_guard(state: WBSState) -> str:
    """현재 L3 태스크 수 기반 신규 태스크 제안 제약 문자열 반환.
    8-80 Rule: 작업 패키지 하나의 작업량은 8시간~80시간(1일~2주) 사이.
    L3 태스크 수가 기준을 넘으면 NEW_TASK 금지 지침 반환."""
    draft = state.get("current_wbs_draft") or []
    l3_count = len([t for t in draft if (t.level.value if hasattr(t.level, "value") else str(t.level)) == "L3"])
    if l3_count >= 20:
        return (
            f"\n🚫 [범위 제한] 현재 L3 태스크: {l3_count}개. 태스크 추가는 PM만 가능합니다.\n"
            "기존 태스크의 일정/버퍼/배분에만 집중하세요. 0.5일 미만 마이크로 태스크 분할은 금지입니다.\n"
        )
    if l3_count >= 15:
        return (
            f"\n⚠️ [범위 주의] 현재 L3 태스크: {l3_count}개. 추가 태스크 없이 기존 항목 검토에 집중하세요.\n"
        )
    return ""


def _extract_buffer_days(text: str) -> Optional[int]:
    """텍스트에서 'N일' 또는 'N day' 패턴을 찾아 숫자로 변환 (수렴도 계산용)"""
    import re
    # '버퍼 N일', 'N일 버퍼', 'N day buffer', 'N-day', 'N일' 등 다양한 패턴 대응
    patterns = [
        r'버퍼\s*(\d+)\s*일',
        r'(\d+)\s*일\s*버퍼',
        r'(\d+)\s*(일|day|days)',
    ]
    for p in patterns:
        match = re.search(p, text, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def _create_message(
    agent_role: AgentRole,
    agent_name: str,
    content: str,
    msg_type: str = "comment",
    task_id: str = None,
    buffer_days: int = None,
) -> DebateMessage:
    return DebateMessage(
        timestamp=datetime.now().strftime("%H:%M:%S"),
        agent_role=agent_role,
        agent_name=agent_name,
        message=content,
        message_type=msg_type,
        related_task_id=task_id,
        buffer_days_proposed=buffer_days,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  플래너 에이전트
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@traceable(name="Agent_Planner")
def planner_agent(state: WBSState) -> WBSState:
    """플래너: 전체 일정 구조, 병목 리스크, 버퍼 + R&R 적합성 종합 검토"""
    model_id = state.get("model_config", {}).get("planner")
    llm = get_llm(temperature=0.5, max_tokens=768, model_id=model_id)
    member = _get_member_by_role(state, "planner", "planner") or _get_member_by_role(state, "PM", "planner")
    if not member:
        return {}
    
    member_name = member.name
    member_id = member.member_id
    persona = _get_persona(state, member_id) if member_id else ""

    wbs_draft = state.get("current_wbs_draft") or []
    _cur_l2 = state.get("_current_l2_task_id")
    wbs_tree = _summarize_wbs_for_l2(wbs_draft, _cur_l2) if _cur_l2 else _summarize_wbs(wbs_draft)
    rag_info = "\n".join(state.get("rag_reference_wbs", [])[:2])
    my_tasks = _get_member_tasks(state, member_id)
    my_tasks_summary = _format_member_tasks(my_tasks)
    member_profile = _format_member_profile(member)
    team_summary = _format_team_summary(state, exclude_member_id=member_id)

    project_type = _infer_project_type(state)
    l2_context_block = _get_l2_context_block(state)
    scope_guard = _scope_creep_guard(state)
    disc_context = _get_disc_context(state, member_name)
    rag_context = state.get("_candidate_rag_context") or ""
    roster_block, _active_names = _get_active_roster(state, exclude_member_id=member_id)
    pm_directive = _build_pm_directive(state)
    review_instructions = """[지침]
1. **전수 검토**: 내게 배정된 모든 태스크 ID를 언급하며 각각의 적합성을 평가하세요.
2. **일정 현실성**: 과소/과대 산정된 태스크를 지적하세요. 조정은 0.5일 또는 1.0일 단위로만 제안.
3. **버퍼 요청**: 리스크가 있는 건에 buffer_days(숫자)를 제안하세요. 다른 태스크에서 차감할 필요 없습니다 — PM이 전체 예산을 조율합니다.
4. **상호작용**: 다른 에이전트의 의견에 대해 동의만 하지 말고, 새로운 관점의 문제를 지적하세요.
⛔ 신규 태스크 제안 금지 — 기존 태스크의 description에 병합하거나 PM에게 요청하세요.
⛔ 이전 턴에서 이미 논의된 내용을 반복하지 마세요."""

    prompt = f"""═══ 정체성 (절대 변경 금지) ═══
나는 [{member_name}]이며, 이번 검토 관점은 [플래너]입니다. member_id: {member_id}
이 정체성을 토론 전체에서 유지하세요. 내 프로필 근거 없이 다른 사람의 의견을 대변하지 마세요.
{pm_directive}
{disc_context}
{persona}
{l2_context_block}{scope_guard}
나의 프로필: {member_profile}

[내게 배정된 태스크 ({len(my_tasks)}개)]
{my_tasks_summary}

[전체 WBS 초안]
{wbs_tree}

{roster_block}

[참고 데이터]
{rag_info}

[현재까지의 토론 내역]
{_format_debate_history(state)}

{review_instructions}
{_STRUCTURED_FORMAT_INSTRUCTION}
반드시 [{member_name} / 플래너] 로 시작하세요."""

    response = llm.invoke([{"role": "user", "content": _PASS_DIRECTIVE + _get_prompting_prefix(state) + prompt}])
    response_text = normalize_content(response.content) if hasattr(response, "content") else str(response)

    # 수렴도 계산을 위한 버퍼 숫자 추출
    buffer_days = _extract_buffer_days(response_text)
    msg_type = "comment"
    if buffer_days is not None:
        msg_type = "buffer_request"
    if "반대" in response_text or "이의" in response_text:
        msg_type = "objection"

    # member.name을 명시적으로 사용 (사용자 요청)
    msg = _create_message(
        AgentRole.PLANNER, member.name, response_text, msg_type,
        task_id=state.get("_current_l2_task_id"), buffer_days=buffer_days,
    )
    return {"planner_response": response_text, "debate_log": [msg]}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  프론트엔드 개발자 에이전트
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@traceable(name="Agent_Frontend")
def frontend_agent(state: WBSState) -> WBSState:
    """FE 개발자: 프론트엔드 태스크 복잡도, 버퍼, 일정·구조 종합 검토"""
    model_id = state.get("model_config", {}).get("frontend")
    llm = get_llm(temperature=0.6, max_tokens=768, model_id=model_id)
    member = _get_member_by_role(state, "frontend", "frontend") or _get_member_by_role(state, "fullstack", "frontend")
    if not member:
        return {}
 
    member_name = member.name
    member_id = member.member_id
    persona = _get_persona(state, member_id) if member_id else ""

    wbs_draft = state.get("current_wbs_draft") or []
    _cur_l2 = state.get("_current_l2_task_id")
    wbs_tree = _summarize_wbs_for_l2(wbs_draft, _cur_l2) if _cur_l2 else _summarize_wbs(wbs_draft)
    prd = state.get("prd")
    my_tasks = _get_member_tasks(state, member_id)
    my_tasks_summary = _format_member_tasks(my_tasks)
    member_profile = _format_member_profile(member)
    team_summary = _format_team_summary(state, exclude_member_id=member_id)

    project_type = _infer_project_type(state)
    l2_context_block = _get_l2_context_block(state)
    scope_guard = _scope_creep_guard(state)
    disc_context = _get_disc_context(state, member_name)
    rag_context = state.get("_candidate_rag_context") or ""
    roster_block, _active_names = _get_active_roster(state, exclude_member_id=member_id)
    pm_directive = _build_pm_directive(state)
    tech_hint = f"기술스택 요구사항: {', '.join(prd.tech_stack_requirements if prd else [])}" if prd and prd.tech_stack_requirements else ""

    agent_label = "고객 접점/앱 담당자" if project_type == "business" else "FE 개발자"

    review_instructions = f"""[지침]
1. **R&R 적합성**: 내 배정 태스크 중 내 역량과 불일치하는 것을 지적하고, 더 적합한 팀원을 제안하세요.
2. **일정 현실성**: 과소/과대 산정된 태스크를 지적하세요. 조정은 0.5일 또는 1.0일 단위로만 제안.
3. **버퍼 요청**: 불확실성이 큰 태스크에 buffer_days(숫자)를 제안하세요. 다른 태스크에서 차감할 필요 없습니다 — PM이 전체 예산을 조율합니다.
4. **배분 이의**: {agent_label} 태스크가 다른 직군에 잘못 배정된 경우 지적하세요.
⛔ 신규 태스크 제안 금지 — 누락된 항목은 기존 태스크 description에 병합하거나 PM에게 요청하세요.
⛔ 이전 턴에서 이미 논의된 내용을 반복하지 마세요."""

    prompt = f"""═══ 정체성 (절대 변경 금지) ═══
나는 [{member_name}]이며, 이번 검토 관점은 [{agent_label}]입니다. member_id: {member_id}
이 정체성을 토론 전체에서 유지하세요. 내 프로필 근거 없이 다른 사람의 의견을 대변하지 마세요.
{pm_directive}
{disc_context}
{persona}
{l2_context_block}{scope_guard}
나의 프로필: {member_profile}

[내게 배정된 태스크 ({len(my_tasks)}개)]
{my_tasks_summary}

[전체 WBS 초안]
{wbs_tree}

{roster_block}

{tech_hint}

[현재까지의 토론 내역]
{_format_debate_history(state)}

{review_instructions}
{_STRUCTURED_FORMAT_INSTRUCTION}
반드시 [{member_name} / {agent_label}] 로 시작하세요."""

    response = llm.invoke([{"role": "user", "content": _PASS_DIRECTIVE + _get_prompting_prefix(state) + prompt}])
    response_text = normalize_content(response.content) if hasattr(response, "content") else str(response)

    # 수렴도 계산을 위한 버퍼 숫자 추출
    buffer_days = _extract_buffer_days(response_text)
    msg_type = "buffer_request" if buffer_days is not None else "comment"
    if "반대" in response_text or "이의" in response_text:
        msg_type = "objection"

    msg = _create_message(
        AgentRole.FRONTEND, member_name, response_text, msg_type,
        task_id=state.get("_current_l2_task_id"), buffer_days=buffer_days,
    )
    return {"frontend_response": response_text, "debate_log": [msg]}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  백엔드 개발자 에이전트
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@traceable(name="Agent_Backend")
def backend_agent(state: WBSState) -> WBSState:
    """BE 개발자: 백엔드 태스크 리스크, 일정·구조·배분 종합 검토"""
    model_id = state.get("model_config", {}).get("backend")
    llm = get_llm(temperature=0.6, max_tokens=768, model_id=model_id)
    member = _get_member_by_role(state, "backend", "backend") or _get_member_by_role(state, "fullstack", "backend")
    if not member:
        return {}
 
    member_name = member.name
    member_id = member.member_id
    persona = _get_persona(state, member_id) if member_id else ""

    wbs_draft = state.get("current_wbs_draft") or []
    _cur_l2 = state.get("_current_l2_task_id")
    wbs_tree = _summarize_wbs_for_l2(wbs_draft, _cur_l2) if _cur_l2 else _summarize_wbs(wbs_draft)
    prd = state.get("prd")
    my_tasks = _get_member_tasks(state, member_id)
    my_tasks_summary = _format_member_tasks(my_tasks)
    constraints = "\n".join((prd.special_constraints if prd else None) or ["없음"])
    member_profile = _format_member_profile(member)
    team_summary = _format_team_summary(state, exclude_member_id=member_id)

    project_type = _infer_project_type(state)
    l2_context_block = _get_l2_context_block(state)
    scope_guard = _scope_creep_guard(state)
    disc_context = _get_disc_context(state, member_name)
    roster_block, _active_names = _get_active_roster(state, exclude_member_id=member_id)
    pm_directive = _build_pm_directive(state)
    tech_hint = f"기술스택 요구사항: {', '.join(prd.tech_stack_requirements if prd else [])}" if prd and prd.tech_stack_requirements else ""

    agent_label = "데이터/시스템 분석 담당자" if project_type == "business" else "BE 개발자"

    review_instructions = f"""[지침]
1. **R&R 적합성**: 내 배정 태스크 중 내 역량과 불일치하는 것을 명시하고, 대안 팀원을 제안하세요.
2. **일정 현실성**: 과소 산정된 태스크를 지적하세요. 조정은 0.5일 또는 1.0일 단위로만 제안.
3. **버퍼 요청**: 외부 연동, 데이터 품질 이슈 등에 buffer_days(숫자)를 제안하세요. 다른 태스크에서 차감할 필요 없습니다 — PM이 전체 예산을 조율합니다.
4. **배분 이의**: {agent_label} 태스크가 다른 직군에 잘못 배정된 경우 지적하세요.
⛔ 신규 태스크 제안 금지 — 누락된 항목은 기존 태스크 description에 병합하거나 PM에게 요청하세요.
⛔ 이전 턴에서 이미 논의된 내용을 반복하지 마세요."""

    prompt = f"""═══ 정체성 (절대 변경 금지) ═══
나는 [{member_name}]이며, 이번 검토 관점은 [{agent_label}]입니다. member_id: {member_id}
이 정체성을 토론 전체에서 유지하세요. 내 프로필 근거 없이 다른 사람의 의견을 대변하지 마세요.
{pm_directive}
{disc_context}
{persona}
{l2_context_block}{scope_guard}
나의 프로필: {member_profile}

[내게 배정된 태스크 ({len(my_tasks)}개)]
{my_tasks_summary}

[전체 WBS 초안]
{wbs_tree}

{roster_block}

{tech_hint}
특수 제약: {constraints}

[현재까지의 토론 내역]
{_format_debate_history(state)}

{review_instructions}
{_STRUCTURED_FORMAT_INSTRUCTION}
반드시 [{member_name} / {agent_label}] 로 시작하세요."""

    response = llm.invoke([{"role": "user", "content": _PASS_DIRECTIVE + _get_prompting_prefix(state) + prompt}])
    response_text = normalize_content(response.content) if hasattr(response, "content") else str(response)

    # 수렴도 계산을 위한 버퍼 숫자 추출
    buffer_days = _extract_buffer_days(response_text)
    msg_type = "buffer_request" if buffer_days is not None else "comment"
    if "반대" in response_text or "이의" in response_text:
        msg_type = "objection"

    # member.name을 명시적으로 사용 (사용자 요청)
    msg = _create_message(
        AgentRole.BACKEND, member.name, response_text, msg_type,
        task_id=state.get("_current_l2_task_id"), buffer_days=buffer_days,
    )
    return {"backend_response": response_text, "debate_log": [msg]}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  디자이너 에이전트
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@traceable(name="Agent_Designer")
def designer_agent(state: WBSState) -> WBSState:
    """디자이너: 디자인 프로세스 일정, 협업 시간, 구조·배분 종합 검토"""
    model_id = state.get("model_config", {}).get("designer")
    llm = get_llm(temperature=0.7, max_tokens=768, model_id=model_id)
    member = _get_member_by_role(state, "designer", "designer")
    if not member:
        return {}
 
    member_name = member.name
    member_id = member.member_id
    persona = _get_persona(state, member_id) if member_id else ""

    wbs_draft = state.get("current_wbs_draft") or []
    _cur_l2 = state.get("_current_l2_task_id")
    wbs_tree = _summarize_wbs_for_l2(wbs_draft, _cur_l2) if _cur_l2 else _summarize_wbs(wbs_draft)
    my_tasks = _get_member_tasks(state, member_id)
    my_tasks_summary = _format_member_tasks(my_tasks)
    member_profile = _format_member_profile(member)
    team_summary = _format_team_summary(state, exclude_member_id=member_id)

    project_type = _infer_project_type(state)
    l2_context_block = _get_l2_context_block(state)
    scope_guard = _scope_creep_guard(state)
    disc_context = _get_disc_context(state, member_name)
    roster_block, _active_names = _get_active_roster(state, exclude_member_id=member_id)
    pm_directive = _build_pm_directive(state)

    agent_label = "브랜딩/디자인 담당자" if project_type == "business" else "디자이너"

    review_instructions = f"""[지침]
1. **R&R 적합성**: 내 배정 태스크 중 내 디자인 역량과 맞지 않는 것을 명시하고, 대안을 제안하세요.
2. **일정 현실성**: 디자인 리뷰/피드백 사이클이 반영 안 된 태스크를 지적하세요. 조정은 0.5일 또는 1.0일 단위로만 제안.
3. **버퍼 요청**: 디자인 시안 검토, 핸드오프 리스크 등에 buffer_days(숫자)를 제안하세요. 다른 태스크에서 차감할 필요 없습니다 — PM이 전체 예산을 조율합니다.
4. **배분 이의**: {agent_label} 태스크가 다른 직군에 잘못 배정된 경우 건의하세요.
⛔ 신규 태스크 제안 금지 — 누락된 항목은 기존 태스크 description에 병합하거나 PM에게 요청하세요.
⛔ 이전 턴에서 이미 논의된 내용을 반복하지 마세요."""

    prompt = f"""═══ 정체성 (절대 변경 금지) ═══
나는 [{member_name}]이며, 이번 검토 관점은 [{agent_label}]입니다. member_id: {member_id}
이 정체성을 토론 전체에서 유지하세요. 내 프로필 근거 없이 다른 사람의 의견을 대변하지 마세요.
{pm_directive}
{disc_context}
{persona}
{l2_context_block}{scope_guard}
나의 프로필: {member_profile}

[내게 배정된 태스크 ({len(my_tasks)}개)]
{my_tasks_summary}

[전체 WBS 초안]
{wbs_tree}

{roster_block}

[현재까지의 토론 내역]
{_format_debate_history(state)}

{review_instructions}
{_STRUCTURED_FORMAT_INSTRUCTION}
반드시 [{member_name} / {agent_label}] 로 시작하세요."""

    response = llm.invoke([{"role": "user", "content": _PASS_DIRECTIVE + _get_prompting_prefix(state) + prompt}])
    response_text = normalize_content(response.content) if hasattr(response, "content") else str(response)

    # 수렴도 계산을 위한 버퍼 숫자 추출
    buffer_days = _extract_buffer_days(response_text)
    msg_type = "buffer_request" if buffer_days is not None else "comment"
    if "반대" in response_text or "이의" in response_text:
        msg_type = "objection"

    # member.name을 명시적으로 사용 (사용자 요청)
    msg = _create_message(
        AgentRole.DESIGNER, member.name, response_text, msg_type,
        task_id=state.get("_current_l2_task_id"), buffer_days=buffer_days,
    )
    return {"designer_response": response_text, "debate_log": [msg]}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  QA 에이전트
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@traceable(name="Agent_QA")
def qa_agent(state: WBSState) -> WBSState:
    """QA 엔지니어: 테스트 일정, 품질 보증, 구조·배분 종합 검토"""
    model_id = state.get("model_config", {}).get("qa")
    llm = get_llm(temperature=0.4, max_tokens=768, model_id=model_id)
    member = _get_member_by_role(state, "qa", "qa")
    if not member:
        return {}
 
    member_name = member.name
    member_id = member.member_id
    persona = _get_persona(state, member_id) if member_id else ""

    wbs_draft = state.get("current_wbs_draft") or []
    _cur_l2 = state.get("_current_l2_task_id")
    wbs_tree = _summarize_wbs_for_l2(wbs_draft, _cur_l2) if _cur_l2 else _summarize_wbs(wbs_draft)
    rag_info = "\n".join(state.get("rag_meeting_logs", [])[:2])
    my_tasks = _get_member_tasks(state, member_id)
    my_tasks_summary = _format_member_tasks(my_tasks)
    member_profile = _format_member_profile(member)
    team_summary = _format_team_summary(state, exclude_member_id=member_id)

    # 전체 QA / 개발 일수 비율 계산
    qa_tasks_all = [t for t in wbs_draft if "qa" in t.assigned_role.lower() or "테스트" in t.title.lower()]
    total_qa_days = sum(t.estimated_days for t in qa_tasks_all)
    total_dev_days = sum(t.estimated_days for t in wbs_draft if t.level.value == "L1" and "개발" in t.title)

    project_type = _infer_project_type(state)
    l2_context_block = _get_l2_context_block(state)
    scope_guard = _scope_creep_guard(state)
    disc_context = _get_disc_context(state, member_name)
    roster_block, _active_names = _get_active_roster(state, exclude_member_id=member_id)
    pm_directive = _build_pm_directive(state)
    ratio_info = f"QA/검증 관련 {total_qa_days}일 / 주요 실행 {total_dev_days}일 (비율: {round(total_qa_days/total_dev_days*100, 1) if total_dev_days else 0}%)"

    agent_label = "성과 검증 담당자" if project_type == "business" else "QA 엔지니어"

    review_instructions = f"""[일정 데이터]
{ratio_info}

[지침]
1. **R&R 적합성**: 내 배정 태스크 중 내 QA 역량과 맞지 않는 항목을 지적하고, 대안 팀원을 제안하세요.
2. **일정 현실성**: QA/검증 일정이 부족한 태스크를 지적하세요. 조정은 0.5일 또는 1.0일 단위로만 제안.
3. **버퍼 요청**: 버그 수정 사이클, 파일럿 테스트 등에 buffer_days(숫자)를 제안하세요. 다른 태스크에서 차감할 필요 없습니다 — PM이 전체 예산을 조율합니다.
4. **배분 이의**: QA 없이 릴리즈되는 Phase가 있다면 강력히 이의를 제기하세요.
⛔ 신규 태스크 제안 금지 — 누락된 항목은 기존 태스크 description에 병합하거나 PM에게 요청하세요.
⛔ 이전 턴에서 이미 논의된 내용을 반복하지 마세요."""

    prompt = f"""═══ 정체성 (절대 변경 금지) ═══
나는 [{member_name}]이며, 이번 검토 관점은 [{agent_label}]입니다. member_id: {member_id}
이 정체성을 토론 전체에서 유지하세요. 내 프로필 근거 없이 다른 사람의 의견을 대변하지 마세요.
{pm_directive}
{disc_context}
{persona}
{l2_context_block}{scope_guard}
나의 프로필: {member_profile}

[내게 배정된 태스크 ({len(my_tasks)}개)]
{my_tasks_summary}

[전체 WBS 초안]
{wbs_tree}

{roster_block}

[현재까지의 토론 내역]
{_format_debate_history(state)}

{review_instructions}
{_STRUCTURED_FORMAT_INSTRUCTION}
반드시 [{member_name} / {agent_label}] 로 시작하세요."""

    response = llm.invoke([{"role": "user", "content": _PASS_DIRECTIVE + _get_prompting_prefix(state) + prompt}])
    response_text = normalize_content(response.content) if hasattr(response, "content") else str(response)

    # 수렴도 계산을 위한 버퍼 숫자 추출
    buffer_days = _extract_buffer_days(response_text)
    # QA는 기본적으로 objection 성향이 강함
    msg_type = "objection" if buffer_days is None else "buffer_request"

    # member.name을 명시적으로 사용 (사용자 요청)
    msg = _create_message(
        AgentRole.QA, member.name, response_text, msg_type,
        task_id=state.get("_current_l2_task_id"), buffer_days=buffer_days,
    )
    return {"qa_response": response_text, "debate_log": [msg]}


@traceable(name="Agent_CandidateReviewer")
def candidate_review_agent(state: WBSState, member) -> WBSState:
    """후보 멤버 직접 검토: 사전 직무 노드 없이 현재 L2 후보자가 자기 프로필 근거로 의견을 낸다."""
    if not member:
        return {}
    model_id = state.get("model_config", {}).get("candidate")
    llm = get_llm(temperature=0.55, max_tokens=768, model_id=model_id)

    member_name = member.name
    member_id = member.member_id
    persona = _get_persona(state, member_id)
    wbs_draft = state.get("current_wbs_draft") or []
    current_l2 = state.get("_current_l2_task_id")
    wbs_tree = _summarize_wbs_for_l2(wbs_draft, current_l2) if current_l2 else _summarize_wbs(wbs_draft)
    my_tasks = _get_member_tasks(state, member_id)
    my_tasks_summary = _format_member_tasks(my_tasks)
    member_profile = _format_member_profile(member)
    l2_context_block = _get_l2_context_block(state)
    scope_guard = _scope_creep_guard(state)
    disc_context = _get_disc_context(state, member_name)
    roster_block, _active_names = _get_active_roster(state, exclude_member_id=member_id)
    pm_directive = _build_pm_directive(state)
    rag_context = state.get("_candidate_rag_context") or ""
    review_lenses = (state.get("l2_review_lenses") or {}).get(current_l2, []) if current_l2 else []
    lens_text = ", ".join(review_lenses) if review_lenses else "작업 적합성, 일정 리스크, 품질 리스크"

    prompt = f"""═══ 정체성 (절대 변경 금지) ═══
나는 [{member_name}]입니다. member_id: {member_id}
이번 프로젝트에서 내 직무는 아직 확정되지 않았습니다. 아래 L2 후보 풀에 포함된 후보자로서, 내 경험/기술/성향 근거로만 검토하세요.
{pm_directive}
{disc_context}
{persona}
{l2_context_block}{scope_guard}
나의 프로필: {member_profile}

[이번 L2의 주요 검토 관점]
{lens_text}

[내게 이미 배정된 태스크 ({len(my_tasks)}개)]
{my_tasks_summary}

[검토 대상 WBS]
{wbs_tree}

{rag_context}

{roster_block}

[현재까지의 토론 내역]
{_format_debate_history(state)}

[지침]
1. 내가 이 L2/L3의 적합한 담당자인지, 아니면 다른 후보가 더 적합한지 근거를 들어 말하세요.
2. 일정/버퍼/R&R 리스크가 있으면 task ID와 함께 구체적으로 지적하세요.
3. 내 프로필과 무관한 영역은 억지로 대변하지 말고 [PASS] 하세요.
4. 신규 태스크 제안은 금지하고, 누락은 기존 태스크 description 보강 또는 PM 검토 요청으로 표현하세요.

{_STRUCTURED_FORMAT_INSTRUCTION}
반드시 [{member_name} / 후보 검토자] 로 시작하세요."""

    response = llm.invoke([{"role": "user", "content": _PASS_DIRECTIVE + _get_prompting_prefix(state) + prompt}])
    response_text = normalize_content(response.content) if hasattr(response, "content") else str(response)
    buffer_days = _extract_buffer_days(response_text)
    msg_type = "buffer_request" if buffer_days is not None else "comment"
    if "반대" in response_text or "이의" in response_text or "부적합" in response_text:
        msg_type = "objection"
    if PASS_TOKEN in response_text.upper():
        msg_type = "pass"

    msg = _create_message(
        AgentRole.CANDIDATE, member_name, response_text, msg_type,
        task_id=current_l2, buffer_days=buffer_days,
    )
    return {"debate_log": [msg]}


def _summarize_wbs(tasks: List[WBSTask]) -> str:
    """WBS 태스크 목록을 L1→L2→L3 계층 트리로 변환.
    L1/L2는 담당자 없이 역할 카테고리만 표시, L3에만 실제 담당자 표시.
    """
    if not tasks:
        return "(WBS 없음)"

    lines = []
    l1_tasks = [t for t in tasks if t.level.value == "L1"]

    for l1 in l1_tasks:
        lines.append(f"📁 [{l1.task_id}] {l1.title} — {l1.estimated_days}일 (직군: {l1.assigned_role})")

        l2_tasks = [t for t in tasks if t.level.value == "L2" and t.parent_id == l1.task_id]
        for l2 in l2_tasks:
            lines.append(f"  ├─ [{l2.task_id}] {l2.title} — {l2.estimated_days}일 (직군: {l2.assigned_role})")

            l3_tasks = [t for t in tasks if t.level.value == "L3" and t.parent_id == l2.task_id]
            for l3 in l3_tasks:
                # L3에만 실제 담당자 표시
                assignees = ", ".join(l3.assigned_to) if l3.assigned_to else f"[미배정/{l3.assigned_role}]"
                lines.append(f"  │  └─ [{l3.task_id}] {l3.title} — {l3.estimated_days}일 (담당: {assignees})")

    total_l1_days = sum(t.estimated_days for t in l1_tasks)
    lines.append(f"\n총 L1 Phase 합계: {total_l1_days}일 ({round(total_l1_days/5, 1)}주)")

    return "\n".join(lines)


def _summarize_wbs_for_l2(tasks: List[WBSTask], l2_id: str) -> str:
    """현재 토론 중인 L2 하나와 그 하위 L3만 표시하는 축약 트리.
    에이전트가 다른 L2 태스크를 참조하지 않도록 범위를 제한합니다."""
    l2_task = next((t for t in tasks if t.task_id == l2_id), None)
    if not l2_task:
        return "(해당 L2 태스크 없음)"

    l3_tasks = [t for t in tasks if t.level.value == "L3" and t.parent_id == l2_id]
    lines = [
        f"📂 [{l2_task.task_id}] {l2_task.title} — {l2_task.estimated_days}일"
        f" | 중요도: {getattr(l2_task, 'importance', 'Medium')} | 직군: {l2_task.assigned_role}"
    ]
    for l3 in l3_tasks:
        assignees = ", ".join(l3.assigned_to) if l3.assigned_to else f"[미배정/{l3.assigned_role}]"
        lines.append(
            f"  └─ [{l3.task_id}] {l3.title}"
            f" — {l3.estimated_days}일 (담당: {assignees}, 버퍼: {l3.buffer_days}일)"
        )
    total_l3 = sum(t.estimated_days for t in l3_tasks)
    lines.append(f"\n이 L2 소계: {len(l3_tasks)}개 L3 태스크, {total_l3}일")
    return "\n".join(lines)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Critic Agent (A1: cross-check, 독립 리뷰어)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@traceable(name="Agent_Critic")
def critic_agent(state: WBSState) -> dict:
    """
    A1 크리틱 에이전트 — 어느 직군에도 속하지 않는 독립 리뷰어.
    역할: (1) 토론 내 과도한 낙관/누락 리스크 지적, (2) 버퍼 불균형 cross-check,
         (3) 팩트 모순 포착. 실명 없음·발언 1회만.
    state["veto_enabled"]=True일 때 버퍼 이상치에 [VETO] 태그 부여.
    """
    model_id = state.get("model_config", {}).get("critic")
    llm = get_llm(temperature=0.3, max_tokens=512, model_id=model_id)

    wbs_draft = state.get("current_wbs_draft") or []
    wbs_tree = _summarize_wbs(wbs_draft)
    recent_log = "\n".join(
        f"[{m.agent_name}]: {m.message[:180]}"
        for m in state.get("debate_log", [])[-8:]
        if getattr(m, "message_type", "") not in ("mediation", "decision")
    )
    veto_on = bool(state.get("veto_enabled"))
    veto_hint = (
        "\n※ 명백히 과도하거나 근거 빈약한 버퍼는 메시지 첫 줄에 `[VETO] <task_id>` 표기 허용."
        if veto_on else ""
    )

    prompt = f"""[크리틱 / 독립 리뷰어]
당신은 어느 직군에도 속하지 않는 독립 리뷰어입니다. 아래를 교차 검증하고 2~4줄 이내로 핵심 지적만 하세요.

검증 포인트:
1. 토론에서 누락된 리스크(보안/법규/데이터/운영)
2. 버퍼 분배의 편향(특정 직군 과보호 또는 방치)
3. 팩트 모순(서로 다른 에이전트의 상충 주장 포착){veto_hint}

[WBS 요약]
{wbs_tree}

[최근 토론 발언]
{recent_log}

출력: "[크리틱] ..." 로 시작하는 한국어 2~4줄. 동의만 하지 말고 반드시 한 가지 이상 약점을 지적하세요.
"""
    try:
        response = llm.invoke([{"role": "user", "content": prompt}])
        text = normalize_content(response.content) if hasattr(response, "content") else str(response)
    except Exception as e:
        text = f"[크리틱] (리뷰 생략 — {str(e)[:80]})"

    msg = DebateMessage(
        timestamp=datetime.now().strftime("%H:%M:%S"),
        agent_role=AgentRole.CRITIC,
        agent_name="크리틱",
        message=text,
        message_type="objection",
    )
    return {"debate_log": [msg]}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  상호 검토 에이전트 (Structured Deliberation)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PASS_TOKEN = "[PASS]"


@traceable(name="Agent_FreeDiscussion")
def free_discussion_agent(
    state: WBSState,
    agent_role: AgentRole,
    agent_name: str,
    member,
) -> dict:
    """
    상호 검토 턴: 에이전트가 최근 대화를 보고 추가 발언 여부를 판단.
    - 할 말이 있으면 → 3문장 이내 응답
    - 할 말이 없으면 → "[PASS]" 반환
    Returns: {"spoke": bool, "debate_log": [...]} 또는 {"spoke": False}
    """
    model_id = state.get("model_config", {}).get(str(agent_role.value if hasattr(agent_role, 'value') else agent_role))
    llm = get_llm(temperature=0.5, max_tokens=400, model_id=model_id)
    persona = _get_persona(state, member.member_id) if member else ""
    member_profile = _format_member_profile(member)

    # 프로젝트 유형에 따른 역할 라벨 결정 (다른 에이전트들과 일관성 유지)
    project_type = _infer_project_type(state)
    role_labels = {
        AgentRole.PLANNER:  "플래너",
        AgentRole.FRONTEND: ("FE 개발자" if project_type == "dev" else "고객 접점/앱 담당자"),
        AgentRole.BACKEND:  ("BE 개발자" if project_type == "dev" else "데이터/시스템 분석 담당자"),
        AgentRole.DESIGNER: ("디자이너" if project_type == "dev" else "브랜딩/디자인 담당자"),
        AgentRole.QA:       ("QA 엔지니어" if project_type == "dev" else "성과 검증 담당자"),
        AgentRole.CANDIDATE: "후보 검토자",
    }
    role_label = role_labels.get(agent_role, str(agent_role))

    # 최근 토론 내역을 넉넉히 (상호 검토용)
    debate_history = _format_debate_history(state, max_messages=12)
    roster_block, _active_names = _get_active_roster(state, exclude_member_id=member.member_id if member else None)
    pm_directive = _build_pm_directive(state)

    prompt = f"""═══ 정체성 (절대 변경 금지) ═══
나는 [{member.name if member else '팀원'}]이며, 이번 검토 관점은 [{role_label}]입니다.
프로필: {member_profile}
{pm_directive}

{roster_block}

{debate_history}

[상호 검토 지시]
위 토론 내역을 읽고, 내 전문 분야({role_label})에서 의미 있는 보완이 있으면 발언하세요.
다음 중 하나 이상이면 발언할 수 있습니다:
  - 누락된 리스크 또는 의존성이 보임
  - 기존 발언의 일정/버퍼/R&R 판단에 보완 근거가 있음
  - PM 결정 전에 확인해야 할 질문이나 trade-off가 있음

다만, 내가 이전에 한 말과 거의 같거나 PM이 이미 반영 완료한 사항을 반복하는 경우에는 {PASS_TOKEN}을 출력하세요.

발언 시: [{member.name if member else '팀원'} / {role_label}] 로 시작, 3문장 이내.
PASS 시: 정확히 아래 한 줄만 출력:
{PASS_TOKEN}"""

    response = llm.invoke([{"role": "user", "content": prompt}])
    response_text = normalize_content(response.content) if hasattr(response, "content") else str(response)

    # [PASS] 감지 - 더 유연한 대소문자 및 괄호 여부 체크
    clean_resp = response_text.strip().upper()
    is_pass = (
        PASS_TOKEN in clean_resp or 
        "PASS" in clean_resp or 
        "의견이 없습니다" in clean_resp or 
        len(clean_resp) < 8
    )

    if is_pass:
        pass_msg = _create_message(
            agent_role, 
            member.name if member else agent_name, 
            "추가 의견이 없습니다. [PASS]", 
            "pass",
            task_id=state.get("_current_l2_task_id"),
        )
        return {"spoke": False, "debate_log": [pass_msg]}

    # 실질 응답 → debate_log에 추가
    buffer_days = _extract_buffer_days(response_text)
    msg_type = "comment"
    if buffer_days is not None:
        msg_type = "buffer_request"
    if "반대" in response_text or "이의" in response_text:
        msg_type = "objection"
    if "동의" in response_text or "찬성" in response_text:
        msg_type = "agreement"

    msg = _create_message(
        agent_role, member.name if member else agent_name, response_text, msg_type,
        task_id=state.get("_current_l2_task_id"), buffer_days=buffer_days,
    )
    return {"spoke": True, "debate_log": [msg]}
