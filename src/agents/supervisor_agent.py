"""
글로벌 슈퍼바이저(PM) 에이전트
전체 WBS 생성 흐름을 통제하고 에이전트 간 합의를 도출합니다.
"""
import json
import os
import re
import math
from datetime import datetime
from typing import Any, List, Dict, Optional
from langsmith import traceable

from agents.state import WBSState
from agents.llm_config import get_llm, normalize_content
from schemas.wbs_schema import DebateMessage, AgentRole, WBSTask, WBSLevel

def _format_rationale(rationale) -> str:
    """JSON 형태의 사유를 사람이 읽기 좋은 일반 텍스트로 변환 (AI 말투 억제)"""
    if not rationale:
        return "특별한 사유 없음"
    if isinstance(rationale, dict):
        lines = []
        if "workload_summary" in rationale:
            ws = rationale["workload_summary"]
            if isinstance(ws, dict):
                ws_str = ", ".join([f"{k}({v}건)" for k, v in ws.items()])
                lines.append(f"업무 부하 현황: {ws_str}")
            else:
                lines.append(f"업무 부하 현황: {ws}")
        
        if "load_balancing" in rationale or "load_balancing_strategy" in rationale:
            lb = rationale.get("load_balancing") or rationale.get("load_balancing_strategy")
            lines.append(f"부하 균형 조정: {lb}")
        
        if "strategic_reasoning" in rationale or "strategic_call" in rationale or "strategic_calls" in rationale:
            sr = rationale.get("strategic_reasoning") or rationale.get("strategic_call") or rationale.get("strategic_calls")
            lines.append(f"전략적 판단 근거: {sr}")
            
        # 나머지 키들도 볼드 없이 처리
        for k, v in rationale.items():
            if k not in ["workload_summary", "load_balancing", "load_balancing_strategy", "strategic_reasoning", "strategic_call", "strategic_calls"]:
                key_name = k.replace("_", " ").title()
                lines.append(f"{key_name}: {v}")
        
        return "\n".join(lines).replace("**", "") # 모든 볼드 제거
    return str(rationale).replace("**", "")

def _repair_truncated_json_object(text: str) -> str:
    """
    Best-effort repair for model outputs truncated near the end of a JSON object.
    Keeps already-complete leading fields such as allocations/calling_context usable.
    """
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE).strip()
        s = re.sub(r"\s*```$", "", s).strip()

    start = s.find("{")
    if start > 0:
        s = s[start:]

    stack = []
    in_string = False
    escaped = False
    last_top_level_end = -1

    for i, ch in enumerate(s):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                opener = stack[-1]
                if (opener == "{" and ch == "}") or (opener == "[" and ch == "]"):
                    stack.pop()
                    if not stack:
                        last_top_level_end = i + 1
                else:
                    break

    if last_top_level_end > 0:
        return s[:last_top_level_end]

    repaired = s.rstrip()
    if in_string:
        repaired += '"'

    repaired = re.sub(r",\s*$", "", repaired)
    for opener in reversed(stack):
        repaired += "}" if opener == "{" else "]"
    return repaired

def _log(state: WBSState, agent_role: AgentRole, agent_name: str, message: str,
         msg_type: str = "comment", task_id: str = None, buffer_days: int = None) -> DebateMessage:
    return DebateMessage(
        timestamp=datetime.now().strftime("%H:%M:%S"),
        agent_role=agent_role,
        agent_name=agent_name,
        message=message,
        message_type=msg_type,
        related_task_id=task_id,
        buffer_days_proposed=buffer_days,
    )

@traceable(name="Agent_Task_Manager")
def supervisor_task_match(state: WBSState) -> WBSState:
    """
    Phase 2: Task 매칭 Manager (Supervisor)
    Phase 1의 WBS 초안을 보고 어떤 에이전트들을 호출할지 결정하며, 팀원의 역량 프로필을 기반으로 R&R을 매핑합니다.
    """
    model_config = state.get("model_config", {}) or {}
    model_id = model_config.get("task_manager") or model_config.get("supervisor")
    llm = get_llm(temperature=0.3, max_tokens=4096, model_id=model_id)
    wbs = state.get("current_wbs_draft", [])
    team = state.get("team_members", [])
    state_role_map = state.get("member_role_map") or {}

    # 만약 초안이 비어있으면 대체 로직 방어
    wbs_summary = "\n".join([f"- {t.task_id}: {t.title} (작업 성격: {t.assigned_role})" for t in wbs]) if wbs else "WBS 내역 없음"
    # 역할(role)은 의도적으로 제외 — supervisor가 기술·강점만 보고 R&R을 독립적으로 결정하도록
    team_summary = "\n".join([
        f"- {m.member_id} ({m.name}): "
        f"경력 {m.years_of_experience}년, "
        f"기술스택: {', '.join(m.tech_stack)}, "
        f"강점: {', '.join(m.strengths) if isinstance(m.strengths, list) else m.strengths}, "
        f"약점: {', '.join(m.weaknesses) if isinstance(m.weaknesses, list) else m.weaknesses}"
        for m in team
    ])
    allowed_member_ids = [m.member_id for m in team]
    allowed_member_ids_text = ", ".join(allowed_member_ids) if allowed_member_ids else "없음"

    # 회의록 컨텍스트 — 팀원의 발언에서 역량·성향·리스크 파악용
    meeting_logs = state.get("rag_meeting_logs", [])
    meeting_context = "\n".join(meeting_logs[:3]) if meeting_logs else ""
    meeting_section = ""
    if meeting_context:
        meeting_section = f"""
[참고 회의록 — 팀원 발언에서 역량·성향·리스크를 파악하여 배정에 반영하세요]
{meeting_context}
"""

    # eDISC 행동유형 컨텍스트
    disc_profiles = state.get("disc_profiles") or {}
    disc_section = ""
    if disc_profiles:
        disc_lines = []
        for name, ctx in disc_profiles.items():
            disc_lines.append(f"── {name} ──\n{ctx}")
        disc_section = "\n[eDISC 행동유형 프로파일 — 의사소통 스타일·동기·팀 역할을 배정에 반영하세요]\n" + "\n".join(disc_lines) + "\n"

    # L2 태스크 추출 (에이전트 매핑 대상)
    l2_tasks = [t for t in wbs if (t.level.value if hasattr(t.level, "value") else str(t.level)) == "L2"]
    l2_summary = "\n".join([
        f"- {t.task_id}: {t.title} (중요도: {getattr(t, 'importance', 'Medium')}, 작업 성격: {t.assigned_role})"
        for t in l2_tasks
    ]) if l2_tasks else "L2 태스크 없음"

    # L3 태스크만 추출 (배정 대상)
    l3_tasks = [t for t in wbs if (t.level.value if hasattr(t.level, "value") else str(t.level)) == "L3"]
    l3_summary = "\n".join([f"- {t.task_id}: {t.title} (작업 성격: {t.assigned_role}, 상위: {t.parent_id})" for t in l3_tasks]) if l3_tasks else "L3 태스크 없음"

    # L2 ID 예시 (첫 3개)
    l2_ids_example = [t.task_id for t in l2_tasks[:3]]
    l2_id_example1 = l2_ids_example[0] if l2_ids_example else "L2-01-01"
    l2_id_example2 = l2_ids_example[1] if len(l2_ids_example) > 1 else "L2-01-02"

    prompt = f"""[Task 매칭 Manager]
방금 전 WBS Gen Agent가 빈 R&R 상태의 3단계 WBS 초안을 생성했습니다.

[WBS 배정 원칙 — 반드시 준수]
- L1(Phase), L2(기능그룹)은 요약 항목입니다. 담당자를 배정하지 마세요.
- 실제 담당자는 L3(세부 태스크)에만 배정합니다. 아래 L3 목록에만 allocations을 작성하세요.
- 직무명으로 사람을 먼저 고정하지 마세요. 각 L2/L3 작업의 요구 역량, 경험, 성향, 현재 부하를 기준으로 후보 풀을 먼저 구성한 뒤 가장 적합한 담당자를 배정하세요.

팀원 역량 프로필:
{team_summary}

[허용 팀원 ID 목록 — 반드시 그대로 복사]
{allowed_member_ids_text}

[ID 및 출력 형식 하드 제약]
- allocations와 l2_candidate_pools에는 위 허용 팀원 ID만 사용하세요.
- 새 MBR ID를 만들거나, 이름/직무명/agent 이름을 assignee로 쓰지 마세요.
- allocations의 key는 아래 L3 세부 태스크 ID만 사용하세요.
- allocations의 value는 항상 팀원 ID 문자열 배열이어야 합니다. 예: ["{team[0].member_id if team else 'MBR-XXXX'}"]

{meeting_section}{disc_section}
[L2 기능그룹 목록 — 에이전트 매핑 대상 ({len(l2_tasks)}개)]
{l2_summary}

[배정 대상 — L3 세부 태스크 ({len(l3_tasks)}개)]
{l3_summary}

[전체 WBS 맥락 (참고용)]
{wbs_summary}

[리소스 관리 및 상호작용 지침]
1. 후보 우선: 각 L2마다 2~4명의 후보 풀을 만들고, 해당 후보 중에서 L3 담당자를 배정하세요.
2. 부하 균형: 특정 팀원에게 태스크가 몰리면 두 번째로 적합한 후보에게 분산하세요.
3. 리스크 방어: 각 L2에는 가능하면 수행 후보 + 검증/품질 후보가 함께 들어가도록 후보 풀을 구성하세요.
4. 역할 사후 도출: assigned_role은 사전 직무 고정이 아니라 작업 성격/배정 결과를 설명하는 라벨로 취급하세요.
5. 판단 근거 기술:
   - rationale: 후보 풀 선정, 부하 균형, 리스크 커버리지 근거를 설명하세요.
   - assignment_evidence: 각 L3 task_id별 배정 근거를 한 문장으로 작성하세요.

```json
{{
  "called_agents": ["candidate"],
  "rationale": "L2별 후보 풀을 먼저 구성하고, 수행 관점과 검증 관점을 함께 호출했습니다.",
  "l2_candidate_pools": {{
    "{l2_id_example1}": ["{team[0].member_id if team else 'MBR-XXXX'}", "{team[1].member_id if len(team) > 1 else 'MBR-YYYY'}"]
  }},
  "allocations": {{
    "L3-01-01-01": ["{team[0].member_id if team else 'MBR-XXXX'}"],
    "L3-01-01-02": ["{team[1].member_id if len(team) > 1 else 'MBR-XXXX'}"]
  }},
  "assignment_evidence": {{
    "L3-01-01-01": "요구 역량과 기술 스택이 가장 직접적으로 일치하고 현재 부하가 낮음"
  }}
}}
```"""

    response = normalize_content(llm.invoke([{"role": "user", "content": prompt}]).content)

    # 후보 멤버 직접 호출 방식. called_agents는 기존 UI/평가 호환용 표식만 유지한다.
    called_agents = ["candidate"]
    rationale = "초기 리스크 점검 및 토의 진행"
    allocations = {}
    calling_context = {}
    l2_agent_mapping = {}
    l2_candidate_pools = {}
    task_candidate_pools = {}
    assignment_evidence = {}
    role_assignments_rationale = {}
    # DEBUG: Keep transient LLM responses out of the repository root.
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    debug_dir = os.path.join(project_root, "generated", "debug")
    os.makedirs(debug_dir, exist_ok=True)
    debug_path = os.path.join(debug_dir, "llm_debug_task_match.txt")
    with open(debug_path, "w", encoding="utf-8") as f:
        f.write(f"--- LLM RESPONSE ---\n{response}\n--- END RESPONSE ---")

    # 1차: ```json ... ``` 코드 블록 추출
    json_str = None
    code_block_match = re.search(r"```json\s*(.*?)```", response, re.DOTALL)
    if code_block_match:
        json_str = code_block_match.group(1).strip()
    else:
        # 2차: 순수 { ... } 블록 추출
        brace_match = re.search(r"\{.*\}", response, re.DOTALL)
        if brace_match:
            json_str = brace_match.group()

    if json_str:
        try:
            data = json.loads(json_str)
        except Exception as first_error:
            try:
                repaired_json_str = _repair_truncated_json_object(json_str)
                data = json.loads(repaired_json_str)
                with open(debug_path, "a", encoding="utf-8") as f:
                    f.write(f"\n\n--- Parse Repaired after error: {first_error} ---")
            except Exception as e:
                data = None
                calling_context = {}
                with open(debug_path, "a", encoding="utf-8") as f:
                    f.write(f"\n\n--- Parse FAILED: {e} ---\nOriginal error: {first_error}\nJSON string was: {json_str[:500]}")

        if data is not None:
            # 노드 키 정규화 (LLM이 임의의 명칭을 사용했을 경우 대비)
            def _norm_k(k: str) -> str:
                k = k.lower()
                if any(x in k for x in ["plan", "pm", "strategy", "biz", "기획", "전략"]): return "planner"
                if any(x in k for x in ["front", "mobile", "app", "ui", "ux", "고객", "모바일"]): return "frontend"
                if any(x in k for x in ["back", "data", "system", "infra", "server", "db", "데이터", "시스템"]): return "backend"
                if any(x in k for x in ["design", "brand", "market", "디자인", "브랜드"]): return "designer"
                if any(x in k for x in ["qa", "test", "val", "검증", "테스트"]): return "qa"
                return k

            rationale = data.get("rationale", rationale)
            role_assignments_rationale = data.get("role_assignments_rationale", {})
            assignment_evidence = data.get("assignment_evidence", {}) if isinstance(data.get("assignment_evidence", {}), dict) else {}
            allocations = data.get("allocations", {})

            raw_l2_pools = data.get("l2_candidate_pools", {})
            if isinstance(raw_l2_pools, dict):
                for l2_id, mids in raw_l2_pools.items():
                    if isinstance(mids, str):
                        mids = [mids]
                    if isinstance(mids, list):
                        l2_candidate_pools[str(l2_id)] = [mid for mid in mids if mid in {m.member_id for m in team}]

            with open(debug_path, "a", encoding="utf-8") as f:
                f.write(f"\n\n--- Parse Success ---\nNormalized Called: {called_agents}\nAllocations: {len(allocations)} tasks\nL2 Mapping: {l2_agent_mapping}\nNormalized Context Keys: {list(calling_context.keys())}")
    else:
        calling_context = {}
        with open(debug_path, "a", encoding="utf-8") as f:
            f.write(f"\n\n--- No JSON found in response! ---")

    # 후보 풀은 LLM 제안이 있더라도 deterministic guard로 보강한다.
    auto_l2_pools, task_candidate_pools, auto_evidence = _build_candidate_pools(wbs, team, state_role_map)
    for l2_id, mids in auto_l2_pools.items():
        merged = list(dict.fromkeys((l2_candidate_pools.get(l2_id) or []) + mids))
        l2_candidate_pools[l2_id] = merged[:4]
    assignment_evidence = {**auto_evidence, **assignment_evidence}

    # 고정 직무 노드 호출은 사용하지 않는다. L2별 후보 멤버가 직접 발화하며,
    # l2_agent_mapping/called_agents는 기존 UI/평가 코드 호환을 위한 "candidate" 표식만 유지한다.
    l2_review_lenses = _derive_l2_review_lenses(wbs, l2_candidate_pools, team, state_role_map)
    l2_agent_mapping = {t.task_id: ["candidate"] for t in l2_tasks}
    called_agents = ["candidate"] if l2_tasks else []
    l2_calling_context = {}
    calling_context = {}

    # WBS 초안에 담당자 업데이트 — L3에만 적용 (L1/L2는 빈 배정 유지)
    valid_member_ids = {m.member_id for m in team}
    # state의 member_role_map 사용 (team_members에서 role이 strip되므로 필수)
    updated_wbs = []
    for t in wbs:
        level_val = t.level.value if hasattr(t.level, "value") else str(t.level)
        if level_val in ("L1", "L2"):
            # 요약 노드: 담당자 없이 역할 카테고리만 유지
            updated_wbs.append(t.model_copy(update={"assigned_to": [], "required_role": t.assigned_role}))
            continue
        # L3만 LLM 배정 반영
        assigned = allocations.get(str(t.task_id), [])
        if isinstance(assigned, str):
            assigned = [assigned]
        assigned = [mid for mid in assigned if mid in valid_member_ids]
        update_fields = {"assigned_to": assigned, "required_role": t.assigned_role}
        if assigned:
            update_fields["assigned_role"] = _derive_task_capability_label(t)
        updated_wbs.append(t.model_copy(update=update_fields))

    # 스마트 배정 후처리 (L3 후보 풀·역량·부하 균형; L1/L2는 건너뜀)
    updated_wbs = _smart_assign(
        updated_wbs, team,
        member_role_map=state_role_map,
        locked_assignments=state.get("locked_assignments", {}),
        l2_candidate_pools=l2_candidate_pools,
        task_candidate_pools=task_candidate_pools,
    )

    # 배정 결과 요약 텍스트 생성 — _smart_assign 결과(updated_wbs) 기준으로 실제 L3 배정 반영
    member_map = {m.member_id: m.name for m in team}
    allocation_lines = []
    for t in updated_wbs:
        level_val = t.level.value if hasattr(t.level, "value") else str(t.level)
        if level_val == "L3" and t.assigned_to:
            names = ", ".join(member_map.get(mid, mid) for mid in t.assigned_to)
            allocation_lines.append(f"  [{t.task_id}] {t.title[:20]} → {names}")
        if len(allocation_lines) >= 8:
            break
    allocation_summary = "\n".join(allocation_lines) if allocation_lines else "  (L3 배정 없음)"
    # calling_context 값은 member_id → 이름으로 변환
    called_names = ", ".join(member_map.get(mid, mid) for mid in calling_context.values()) if calling_context else "없음"
    pool_lines = []
    for l2 in l2_tasks[:5]:
        mids = l2_candidate_pools.get(l2.task_id, [])
        names = ", ".join(member_map.get(mid, mid) for mid in mids)
        if names:
            pool_lines.append(f"  [{l2.task_id}] {names}")
    pool_summary = "\n".join(pool_lines) if pool_lines else "  (후보 풀 없음)"

    # 슈라이저(PM) - 역할 명칭 사용
    pm_name = "Task Manager"
    formatted_rationale = _format_rationale(rationale)
    
    # 팀원별 배정 사유 정리
    rr_rationale_lines = []
    if role_assignments_rationale:
        rr_rationale_lines.append("\n[멤버별 R&R 배정 근거]")
        for mid, reason in role_assignments_rationale.items():
            name = member_map.get(mid, mid)
            rr_rationale_lines.append(f"• {name}: {reason}")
    rr_rationale_text = "\n".join(rr_rationale_lines)

    msg = _log(
        state, AgentRole.SUPERVISOR, pm_name,
        f"L2 후보 풀 기반으로 R&R 배분 완료. 호출 후보: {called_names}\nL2 후보 풀 (일부):\n{pool_summary}\nL3 배정 현황 (일부):\n{allocation_summary}\n{rr_rationale_text}\n\n{formatted_rationale}",
        "proposal"
    )

    return {
        "current_wbs_draft": updated_wbs,
        "called_agents": called_agents,
        "calling_context": calling_context,
        "l2_candidate_pools": l2_candidate_pools,
        "task_candidate_pools": task_candidate_pools,
        "l2_review_lenses": l2_review_lenses,
        "l2_calling_context": l2_calling_context,
        "assignment_evidence": assignment_evidence,
        "l2_agent_mapping": l2_agent_mapping,
        "locked_assignments": state.get("locked_assignments", {}),
        "supervisor_rationale": formatted_rationale,
        "debate_log": [msg],
    }

def supervisor_check_and_intervene(state: WBSState) -> Optional[dict]:
    """
    토론 중간 PM 개입 검사 (경량 — LLM 미사용).
    아래 패턴 감지 시 즉각 중재 메시지를 반환합니다:
    1. 연속 순수 동의 ≥ 3턴  → 조기 수렴 선언
    2. 지엽적 구현 세부사항 키워드 감지 → WBS 산정 범위로 리다이렉트
    3. 동일 에이전트가 연속 2+ 발언  → 발언 독점 차단
    반환값: {"debate_log": [DebateMessage], "consensus_reached": bool} 또는 None
    """
    logs = state.get("debate_log", [])
    if len(logs) < 3:
        return None

    recent = [m for m in logs if m.message_type not in ("mediation", "decision", "pass")][-4:]
    if not recent:
        return None

    # ─── 패턴 1: 연속 순수 동의 ───────────────────────
    agree_kw = ["동의", "맞습니다", "적절합니다", "타당합니다", "공감", "충분히 반영", "잘 반영"]
    new_risk_kw = ["위험", "리스크", "지연", "버퍼", "반대", "이의", "추가 필요", "누락"]
    pure_agree = sum(
        1 for m in recent
        if any(k in m.message for k in agree_kw)
        and not any(k in m.message for k in new_risk_kw)
        and not m.buffer_days_proposed
    )
    if pure_agree >= 3:
        msg = _log(
            state, AgentRole.SUPERVISOR, "Task Manager (PM)",
            "✅ 주요 리스크와 버퍼에 대해 충분한 합의가 이루어졌습니다. "
            "추가적인 동의 발언은 불필요합니다. 이 L2 그룹 토론을 마무리합니다.",
            "decision",
        )
        return {"debate_log": [msg], "_l2_debate_cutoff": True}

    # ─── 패턴 2: 지엽적 구현 세부사항 탈선 ───────────────
    offtopic_kw = [
        "iso 8601", "utc", "timestamp format", "타임스탬프 형식", "인코딩", "네이밍 컨벤션",
        "주석 스타일", "코드 컨벤션", "변수명", "함수명", "주석",
    ]
    offtopic_hits = [
        m for m in recent
        if any(k in m.message.lower() for k in offtopic_kw)
    ]
    if offtopic_hits:
        topic = next(
            k for m in offtopic_hits for k in offtopic_kw if k in m.message.lower()
        )
        msg = _log(
            state, AgentRole.SUPERVISOR, "Task Manager (PM)",
            f"⛔ [토론 리다이렉트] '{topic}' 관련 구현 세부사항은 L3 WBS 일정 산정 범위를 벗어납니다. "
            "해당 논의는 구현 단계로 이관하고, 현재는 **버퍼 산정과 R&R 적합성**에 집중해 주세요.",
            "mediation",
        )
        return {"debate_log": [msg]}

    # ─── 패턴 3: 발언 독점 (동일 에이전트 연속 2+) ─────────
    if len(recent) >= 2:
        last_two_names = [m.agent_name for m in recent[-2:]]
        if len(set(last_two_names)) == 1 and last_two_names[0] not in ("Task Manager", "Task Manager (PM)", "시스템"):
            monopolist = last_two_names[0]
            msg = _log(
                state, AgentRole.SUPERVISOR, "Task Manager (PM)",
                f"🔁 [{monopolist}] 님, 핵심 주장은 충분히 전달되었습니다. "
                "다른 전문가들의 관점도 들어보겠습니다. 다음 발언자에게 넘깁니다.",
                "mediation",
            )
            return {"debate_log": [msg]}

    # ─── 패턴 4: 마이크로 태스크 증식 감지 ───────────────
    micro_task_kw = ["0.5일", "반나절", "0.5d", "검증 스크립트", "가독성 검증", "정합성 검증"]
    micro_count = sum(
        1 for m in recent
        if any(k in m.message for k in micro_task_kw)
    )
    if micro_count >= 2:
        msg = _log(
            state, AgentRole.SUPERVISOR, "Task Manager (PM)",
            "⛔ [범위 과잉] 0.5일 단위의 마이크로 태스크 제안이 반복되고 있습니다. "
            "WBS는 최소 1일 단위의 의미 있는 작업 패키지로 구성해야 합니다. "
            "세부 검증 항목은 기존 태스크의 description에 병합하세요.",
            "mediation",
        )
        return {"debate_log": [msg]}

    # ─── 패턴 5: 후보 기반 라우팅에서는 원직무와 발화 렌즈가 다를 수 있다.
    # 과거의 "원직무 기반 역할 탈선" 검사는 후보 우선 구조와 충돌하므로 제거한다.
    # 정체성 위반은 AgentHarness와 프롬프트의 "본인 프로필 근거" 지시로 방어한다.

    # ─── 패턴 6: 동일 태스크 반복 차감 감지 ───────────────
    # 최근 4개 발언에서 동일 태스크 ID가 3회 이상 언급 + "차감/단축/줄여" 패턴
    import re as _re
    task_mentions = {}
    subtract_kw = ["차감", "단축", "줄여", "빼서", "삭감", "축소"]
    for m in recent:
        ids_found = _re.findall(r'L[123]-\d{2}(?:-\d{2}){0,2}', m.message)
        has_subtract = any(k in m.message for k in subtract_kw)
        if has_subtract:
            for tid in ids_found:
                task_mentions[tid] = task_mentions.get(tid, 0) + 1
    over_targeted = [tid for tid, cnt in task_mentions.items() if cnt >= 3]
    if over_targeted:
        msg = _log(
            state, AgentRole.SUPERVISOR, "Task Manager (PM)",
            f"⛔ [일정 보호] {', '.join(over_targeted)} 태스크에서 일정이 반복적으로 차감되고 있습니다. "
            "한 태스크에서 무한정 일정을 빼는 것은 금지합니다. 전체 예산 조율은 PM이 중재 단계에서 처리합니다.",
            "mediation",
        )
        return {"debate_log": [msg]}

    return None


@traceable(name="Agent_Supervisor_Mediate")
def supervisor_mediate(state: WBSState) -> WBSState:
    """
    슈퍼바이저 중재: 토론 내용을 분석하여 실제 버퍼 및 리스크 요인을 추출 및 반영
    """
    model_id = state.get("model_config", {}).get("supervisor")
    llm = get_llm(temperature=0.2, max_tokens=768, model_id=model_id)

    debate_history = "\n".join([
        f"[{m.agent_name}]: {m.message}" for m in state.get("debate_log", [])[-10:]
    ])

    # 재배정 판단을 위해 팀원 목록도 제공
    team = state.get("team_members", [])
    team_summary = "\n".join([
        f"- {m.member_id} ({m.name}): {', '.join(m.tech_stack[:3])}, 강점: {', '.join(m.strengths) if isinstance(m.strengths, list) else m.strengths}"
        for m in team
    ])

    # 현재 WBS L3 태스크 요약 (PM 중재 컨텍스트용)
    wbs_draft = state.get("current_wbs_draft", [])
    l3_wbs_summary = "\n".join([
        f"- {t.task_id}: {t.title} (담당: {', '.join(t.assigned_to) or '미배정'}, {t.estimated_days}일)"
        for t in wbs_draft if (t.level.value if hasattr(t.level, "value") else str(t.level)) == "L3"
    ]) or "L3 태스크 없음"

    prompt = f"""[슈퍼바이저(PM)] 토론 중재 — 버퍼 반영 + R&R 재배정 + 신규 태스크 추가

팀원 목록 (재배정 시 아래 member_id만 사용):
{team_summary}

현재 L3 태스크 배정 현황:
{l3_wbs_summary}

지침:
1. 에이전트들의 의견을 종합하여 각 L3 태스크별 버퍼 일수와 리스크 사유를 추출하세요.
2. **상세 R&R 중재 및 상태 업데이트 (State Mutation)**: 에이전트들이 "내 직무가 아니다", "누구와 R&R을 바꿔달라"고 요청한 경우 이를 **반드시** 읽고 타당하다면 `reassignments`에 기록하여 즉시 반영하십시오. 
3. **Zero-Sum Buffer 검증**: 에이전트가 버퍼 증액을 요구할 때, 다른 태스크의 일정을 단축하여 전체 기간을 상쇄하는 '대안'을 제시했는지 확인하고 이를 조율하세요.
4. **신규 태스크 추가 및 오버엔지니어링 차단**: 타당한 기술적 필수 태스크는 추가하되, 범위를 벗어난 과도하게 복잡한 시스템(예: TCAD, 몬테카를로 등) 도입 제안은 거절하거나 일반적인 구현 태스크로 순화하십시오.
5. 모든 주요 리스크가 반영되었고 배분·일정이 합리적이라면 "consensus_reached": true를 포함하세요.
6. 구조적 문제(L1/L2 계층 오류 등)로 인해 WBS 자체를 갈아엎어야 한다면 "wbs_revision_needed": true를 설정하세요.
7. 출력 형식 (JSON 코드 블록으로만 응답):
```json
{{
  "consensus_reached": bool,
  "wbs_revision_needed": bool,
  "revision_hints": ["힌트1"],
  "tasks": {{"task_id": {{"buffer_days": float, "risk": "핵심 사유 한 문장"}}}},
  "reassignments": {{"L3_task_id": "새_member_id"}},
  "reassignment_rationale": {{"L3_task_id": "재배정 사유 (eDISC 및 역량 관점)"}},
  "new_tasks": [
    {{"id": "L3-02-03-04", "title": "보안 설계", "level": "L3", "parent_id": "L2-02-03", "assigned_role": "Backend Developer", "estimated_days": 3, "dependencies": ["L3-02-03-01"]}}
  ]
}}
```

토론 내역:
{debate_history}"""

    response_raw = normalize_content(llm.invoke([{"role": "user", "content": prompt}]).content)

    # JSON 파싱 (코드 블록 우선)
    extracted_data = {}
    code_block = re.search(r"```json\s*(.*?)```", response_raw, re.DOTALL)
    json_src = code_block.group(1) if code_block else response_raw
    brace_match = re.search(r"\{.*\}", json_src, re.DOTALL)
    if brace_match:
        try: extracted_data = json.loads(brace_match.group())
        except: pass

    extracted_buffers = extracted_data.get("tasks", {})
    llm_consensus = extracted_data.get("consensus_reached", False)
    llm_revision_needed = extracted_data.get("wbs_revision_needed", False)
    revision_hints = extracted_data.get("revision_hints", [])
    reassignments = extracted_data.get("reassignments", {})
    new_task_proposals = extracted_data.get("new_tasks", [])

    # 에이전트 발언에서 NEW_TASK: {...} 패턴 추가 파싱
    for msg in state.get("debate_log", [])[-10:]:
        msg_text = getattr(msg, "message", "")
        for match in re.finditer(r"NEW_TASK:\s*(\{.*?\})", msg_text, re.DOTALL):
            try:
                proposal = json.loads(match.group(1))
                new_task_proposals.append(proposal)
            except Exception:
                pass

    # 하드 리밋 체크 (라운드 제한)
    current_round = state.get("current_round", 0)
    max_rounds = state.get("max_rounds", 3)
    completed_rounds = current_round + 1
    is_limit_reached = completed_rounds >= max_rounds

    # 슈라이저(PM) - 역할 명칭 사용
    pm_name = "PM 에이전트"

    # 버퍼 및 재배정 로직 먼저 수행 (updated_tasks 확보)
    team = state.get("team_members", [])
    member_map = {m.member_id: m.name for m in team}
    valid_member_ids = {m.member_id for m in team}
    state_role_map = state.get("member_role_map") or {}

    # ── A2 Veto Guard: veto_enabled 시 에이전트가 명시적으로 거부한 task_id 버퍼는 제거 ──
    vetoed_task_ids: set = set()
    if state.get("veto_enabled"):
        veto_kw = ["[VETO]", "거부권", "veto:", "강력 반대", "절대 반대"]
        for msg in state.get("debate_log", [])[-20:]:
            text = getattr(msg, "message", "") or ""
            if not any(kw.lower() in text.lower() for kw in veto_kw):
                continue
            related = getattr(msg, "related_task_id", None)
            if related:
                vetoed_task_ids.add(related)
            # 메시지 내 L3-xx-xx 패턴도 수집
            for m_id in re.findall(r"L3-\d{2}-\d{2}-\d{2}", text):
                vetoed_task_ids.add(m_id)
        if vetoed_task_ids:
            extracted_buffers = {tid: v for tid, v in extracted_buffers.items() if tid not in vetoed_task_ids}

    updated_tasks = _apply_dynamic_buffers(state.get("current_wbs_draft", []), extracted_buffers)
    updated_tasks, reassignment_log = _apply_reassignments(
        updated_tasks, reassignments, valid_member_ids,
        team=team, member_role_map=state_role_map,
    )

    # 조기 수렴(Early Convergence) 체크: 최근 3회 일정 변동폭이 0.5일 이내인지 확인
    total_days = sum(t.total_days for t in updated_tasks if (t.level.value if hasattr(t.level, "value") else str(t.level)) == "L3")
    days_history = state.get("total_days_history", []) + [total_days]
    
    is_stable = False
    if len(days_history) >= 3:
        last_3 = days_history[-3:]
        delta = max(last_3) - min(last_3)
        if delta <= 0.5:
            is_stable = True

    is_min_rounds_met = completed_rounds >= state.get("min_rounds", 2)
    # 최소 라운드를 채우지 못했으면 무조건 consensus_reached는 False
    is_consensus = (llm_consensus and is_min_rounds_met) or is_limit_reached or (is_stable and is_min_rounds_met)

    # WBS 구조 재설계 여부 판단 (한도 내에서만 허용, 합의 달성 시에는 재설계 불필요)
    current_revision = state.get("current_wbs_revision", 0)
    max_revisions = state.get("max_wbs_revisions", 1)
    should_revise = llm_revision_needed and not is_consensus and current_revision < max_revisions

    # ── 에이전트 제안 신규 태스크 추가 (Bug 3 fix) ──────────────────────────
    added_task_log = []
    if new_task_proposals:
        existing_ids = {t.task_id for t in updated_tasks}
        for proposal in new_task_proposals:
            try:
                new_id = str(proposal.get("id") or proposal.get("task_id", ""))
                if not new_id or new_id in existing_ids:
                    continue
                parent_id = proposal.get("parent_id")
                level_str = proposal.get("level", "L3")
                if not level_str.startswith("L"):
                    level_str = "L3"
                if level_str == "L3" and (not parent_id or str(parent_id) not in existing_ids):
                    continue
                est_days = float(proposal.get("estimated_days", 2.0))
                raw_imp = str(proposal.get("importance", "Medium")).strip().capitalize()
                if raw_imp not in ("High", "Medium", "Low"):
                    raw_imp = "Medium"
                new_task = WBSTask(
                    task_id=new_id,
                    level=WBSLevel(level_str),
                    parent_id=str(parent_id) if parent_id else None,
                    title=str(proposal.get("title", "추가 태스크")),
                    description=str(proposal.get("description", "")),
                    assigned_to=[],
                    required_role=str(proposal.get("assigned_role", "")),
                    assigned_role=str(proposal.get("assigned_role", "Backend Developer")),
                    estimated_days=est_days,
                    buffer_days=0.0,
                    total_days=est_days,
                    dependencies=proposal.get("dependencies", []),
                    deliverables=[f"{proposal.get('title', '추가 태스크')} 완료"],
                    importance=raw_imp,
                )
                updated_tasks.append(new_task)
                existing_ids.add(new_id)
                added_task_log.append(new_id)
            except Exception:
                continue

        # 신규 태스크에 역할 기반 담당자 배정 (_smart_assign으로 처리)
        if added_task_log:
            updated_tasks = _smart_assign(
                updated_tasks, team,
                member_role_map=state_role_map,
                locked_assignments=state.get("locked_assignments", {}),
                l2_candidate_pools=state.get("l2_candidate_pools", {}),
                task_candidate_pools=state.get("task_candidate_pools", {}),
            )

    # ── locked_assignments 업데이트: PM 재배정 결정을 잠금 (Bug 1 fix) ──────
    prev_locked = dict(state.get("locked_assignments") or {})
    for task_id, new_mid in reassignment_log:
        prev_locked[task_id] = new_mid
    new_locked = prev_locked

    # 자동 스케줄링 재계산 (버퍼 반영 결과 반영)
    updated_tasks = _calculate_automatic_schedule(updated_tasks)

    # 상태 메시지 구성
    reassignment_note = f" (R&R 재배정 {len(reassignment_log)}건 적용)" if reassignment_log else ""
    added_note = f" (신규 태스크 {len(added_task_log)}건 추가: {', '.join(added_task_log)})" if added_task_log else ""

    # 누적 카운트 업데이트 (finalize가 전체 라운드 합계를 표시하도록)
    cum_reassign   = state.get("cumulative_reassignments", 0)   + len(reassignment_log)
    cum_buffers    = state.get("cumulative_buffers_applied", 0) + len(extracted_buffers)
    cum_new_tasks  = state.get("cumulative_new_tasks", 0)       + len(added_task_log)

    # 이번 라운드 + 누적 모두 표시 (사용자가 "왜 0건이지?"로 오해하지 않도록)
    this_round_txt = f"이번 라운드 {len(reassignments)}건 재배정, {len(extracted_buffers)}건 버퍼 조정"
    cumul_txt = f"누적 {cum_reassign}건 재배정, {cum_buffers}건 버퍼 조정"
    formatted_rationale = f"토론 내용을 분석하여 일정 및 R&R을 조정했습니다. ({this_round_txt} / {cumul_txt}{added_note})"

    if should_revise:
        hints_str = " / ".join(revision_hints) if revision_hints else "전반적인 구조 재검토"
        status_msg = f"⚠️ 에이전트 검토 결과 WBS 구조 재설계가 필요합니다. WBS Gen Agent를 재호출합니다. 재설계 방향: {hints_str}"
    elif is_consensus:
        if is_stable and not (llm_consensus or is_limit_reached):
            status_msg = f"최근 3회 토론 결과 전체 일정 변동폭이 0.5일 이내로 안정화되어, 의견이 충분히 수렴된 것으로 간주하고 토론을 조기 종료합니다. ({total_days}일 확정)"
        else:
            status_msg = f"토론된 리스크들을 검토하여 일정 버퍼에 최종 반영했습니다{reassignment_note}{added_note}. 이대로 진행하겠습니다."
    else:
        status_msg = f"제시된 리스크들을 WBS에 중간 반영했습니다{reassignment_note}{added_note}. 추가적인 의견이 있다면 말씀해 주세요."

    # 모든 메시지에 PM 역할 명칭 적용
    msgs = [_log(state, AgentRole.SUPERVISOR, pm_name, status_msg, "mediation")]
    # 재배정 건별 토론 로그 추가
    # 재배정 사유 로깅 (eDISC 및 역량 반영)
    reassignment_rationale = extracted_data.get("reassignment_rationale", {})
    for task_id, new_mid in reassignment_log:
        reason = reassignment_rationale.get(task_id, "역량 및 성향 적합성 판단에 따른 배정")
        name = member_map.get(new_mid, new_mid)
        msgs.append(_log(state, AgentRole.SUPERVISOR, pm_name,
                         f"🔄 [{task_id}] R&R 재배정(잠금): {name} - {reason}", "decision", task_id=task_id))
    # 신규 태스크 추가 로그
    for new_tid in added_task_log:
        msgs.append(_log(state, AgentRole.SUPERVISOR, pm_name,
                         f"➕ [{new_tid}] 에이전트 제안 신규 태스크 추가됨", "decision", task_id=new_tid))

    return {
        "current_wbs_draft": updated_tasks,
        "consensus_reached": is_consensus,
        "wbs_revision_needed": should_revise,
        "wbs_revision_hints": revision_hints if should_revise else [],
        "locked_assignments": new_locked,
        "supervisor_rationale": formatted_rationale,
        "debate_log": msgs,
        "current_round": state.get("current_round", 0) + 1,
        "total_days_history": days_history,
        "cumulative_reassignments": cum_reassign,
        "cumulative_buffers_applied": cum_buffers,
        "cumulative_new_tasks": cum_new_tasks,
    }

@traceable(name="Agent_Supervisor_Finalize")
def supervisor_finalize(state: WBSState) -> WBSState:
    model_id = state.get("model_config", {}).get("supervisor")
    # finalize는 가벼운 요약이므로 temperature를 낮춤
    llm = get_llm(temperature=0.1, max_tokens=512, model_id=model_id)
    final_tasks = state.get("current_wbs_draft", [])
    # 슈퍼바이저(PM) 역할 명칭 사용
    pm_name = "PM 에이전트"
    
    # 총 기간 계산 (L1 기준)
    l1_tasks = [t for t in final_tasks if t.level == WBSLevel.L1]
    total_days = sum(t.total_days for t in l1_tasks) if l1_tasks else 0
    total_msgs = len(state.get("debate_log", [])) + 1
    
    # 누적 중재 카운트 (전체 라운드 합계)
    cum_reassign  = state.get("cumulative_reassignments", 0)
    cum_buffers   = state.get("cumulative_buffers_applied", 0)
    cum_new_tasks = state.get("cumulative_new_tasks", 0)
    rounds_run    = state.get("current_round", 0)
    member_project_roles = _derive_member_project_roles(
        final_tasks,
        state.get("team_members", []),
        state.get("member_role_map", {}) or {},
        state.get("prd"),
    )

    summary = (
        f"## WBS 확정 보고서\n"
        f"- 실무급 3단계 계층 구조 설계\n"
        f"- 총 프로젝트 기간: {round(total_days/5, 1)}주 (총 {total_days}일)\n"
        f"- 토론 라운드: {rounds_run}회 / 전체 메시지: {total_msgs}건\n"
        f"- 누적 중재 반영: 재배정 {cum_reassign}건, 버퍼 조정 {cum_buffers}건, 신규 태스크 {cum_new_tasks}건"
    )

    # Finalize 메시지는 누적 합계 기반으로 재구성 (단일 라운드 rationale이 0이어도 오해 없게)
    if cum_reassign == 0 and cum_buffers == 0 and cum_new_tasks == 0:
        final_rationale = (
            "토론을 종합한 결과, 초안 WBS가 이미 적정 상태로 판단되어 추가 조정은 없었습니다. "
            "이대로 확정합니다."
        )
    else:
        final_rationale = (
            f"토론 내용을 분석하여 일정 및 R&R을 조정했습니다. "
            f"(총 {rounds_run}라운드 누적: {cum_reassign}건 재배정, {cum_buffers}건 버퍼 조정, {cum_new_tasks}건 신규 태스크 추가)"
        )
    formatted_rationale = _format_rationale(final_rationale)
    msg = _log(state, AgentRole.SUPERVISOR, "Task Manager", formatted_rationale, "proposal")
    
    return {
        "final_wbs": final_tasks,
        "generation_summary": summary,
        "member_project_roles": member_project_roles,
        "supervisor_rationale": formatted_rationale,
        "debate_log": [msg],
    }

# ─── 역할 명칭 및 호환 맵 ──────────────────────────────────────────────────
_ROLE_LABELS: dict = {
    "PM": "PM / 총괄",
    "Planner": "플래너",
    "Backend Developer": "BE 개발자",
    "Frontend Developer": "FE 개발자",
    "Fullstack Developer": "풀스택 개발자",
    "Data Engineer": "데이터 엔지니어",
    "DevOps": "데브옵스",
    "Designer": "디자이너",
    "QA Engineer": "QA 엔지니어",
    "Data Analyst": "데이터 분석가",
    "Marketing Planner": "마케팅 플래너",
    "Business Analyst": "비즈니스 분석가",
    "Mobile Developer": "모바일 개발자",
    "Operations Manager": "운영 매니저",
}


def _prd_role_context(prd) -> str:
    if not prd:
        return ""
    parts = [
        getattr(prd, "project_name", ""),
        getattr(prd, "project_goal", ""),
        getattr(prd, "target_users", ""),
        getattr(prd, "scope", ""),
        " ".join(getattr(prd, "key_features", []) or []),
        " ".join(getattr(prd, "tech_stack_requirements", []) or []),
        " ".join(getattr(prd, "special_constraints", []) or []),
        getattr(prd, "raw_text", "") or "",
    ]
    return " ".join(str(p) for p in parts if p)


def _task_role_context(task) -> str:
    parts = [
        getattr(task, "task_id", ""),
        getattr(task, "title", ""),
        getattr(task, "description", ""),
        getattr(task, "assigned_role", ""),
        getattr(task, "required_role", ""),
        " ".join(getattr(task, "deliverables", []) or []),
        " ".join(getattr(task, "risk_factors", []) or []),
    ]
    return " ".join(str(p) for p in parts if p)


def _has_any(text: str, *keywords: str) -> bool:
    lowered = (text or "").lower()
    return any(k.lower() in lowered for k in keywords)


def _domain_prefix(text: str) -> str:
    if _has_any(text, "crm", "고객 관계", "리텐션", "세그먼트", "segmentation", "campaign", "캠페인"):
        if _has_any(text, "세그먼트", "segmentation"):
            return "고객 세그먼트"
        return "CRM"
    if _has_any(text, "추천", "recommend", "ranking", "랭킹", "개인화", "personalization"):
        return "추천/개인화"
    if _has_any(text, "dashboard", "대시보드", "report", "리포트", "bi", "kpi", "지표"):
        return "분석 대시보드"
    if _has_any(text, "정산", "결제", "payment", "billing", "invoice", "매출"):
        return "정산/매출"
    if _has_any(text, "물류", "재고", "wms", "erp", "pos", "운영", "공급망"):
        return "운영/물류"
    if _has_any(text, "mobile", "모바일", "앱", "ios", "android", "react native"):
        return "모바일 앱"
    if _has_any(text, "security", "보안", "인증", "권한", "auth"):
        return "보안/인증"
    if _has_any(text, "data warehouse", "데이터 웨어하우스", "etl", "pipeline", "파이프라인", "lakehouse"):
        return "데이터 플랫폼"
    return "프로젝트"


def _derive_project_role_name(stat: dict, mixes: list, prd_text: str) -> str:
    """고정 직군 대신 PRD와 실제 배정 태스크 문맥으로 프로젝트 역할명을 만든다."""
    if not mixes:
        return _ROLE_LABELS.get(stat.get("original_role"), stat.get("original_role") or "미배정")

    task_text = " ".join(stat.get("task_texts") or [])
    combined = f"{task_text} {prd_text}"
    domain = _domain_prefix(combined)

    if _has_any(task_text, "모델", "model", "ml", "머신러닝", "예측", "prediction", "추천", "recommend", "llm", "ai"):
        suffix = "데이터 사이언티스트"
    elif _has_any(task_text, "분석", "analytics", "dashboard", "대시보드", "리포트", "report", "kpi", "지표", "sql", "세그먼트"):
        suffix = "데이터 애널리스트"
    elif _has_any(task_text, "crm", "캠페인", "campaign", "리텐션", "고객 여정", "퍼널", "마케팅"):
        suffix = "CRM/고객전략 담당"
    elif _has_any(task_text, "etl", "파이프라인", "pipeline", "airflow", "kafka", "spark", "warehouse", "웨어하우스", "db", "데이터베이스"):
        suffix = "데이터 파이프라인 엔지니어"
    elif _has_any(task_text, "api", "server", "서버", "백엔드", "backend", "인증", "권한", "연동", "integration"):
        suffix = "서비스 백엔드 엔지니어"
    elif _has_any(task_text, "ui", "ux", "화면", "frontend", "프론트", "react", "next.js", "component", "컴포넌트"):
        suffix = "프론트엔드/화면 구현 담당"
    elif _has_any(task_text, "디자인", "figma", "프로토타입", "와이어프레임", "사용자 경험", "핸드오프"):
        suffix = "UX/UI 설계자"
    elif _has_any(task_text, "qa", "test", "테스트", "검증", "품질", "회귀", "자동화"):
        suffix = "품질 검증 엔지니어"
    elif _has_any(task_text, "요구사항", "기획", "정책", "전략", "로드맵", "보고", "최종 보고"):
        suffix = "기획/요구사항 매니저"
    elif _has_any(task_text, "운영", "재고", "물류", "pos", "wms", "erp", "공급망"):
        suffix = "운영 프로세스 담당"
    else:
        dominant = mixes[0].get("role", "")
        suffix = _ROLE_LABELS.get(dominant, dominant or "프로젝트 담당")

    if domain == "프로젝트":
        return suffix
    if suffix.startswith(domain):
        return suffix
    return f"{domain} {suffix}"


def _derive_member_project_roles(tasks: list, team: list, member_role_map: dict = None, prd=None) -> list[dict]:
    """최종 L3 배정 태스크 묶음으로 팀원별 프로젝트 역할을 사후 산출한다."""
    role_map = member_role_map or {}
    team_by_id = {m.member_id: m for m in team}
    prd_text = _prd_role_context(prd)
    member_stats: dict = {
        m.member_id: {
            "member_id": m.member_id,
            "member_name": m.name,
            "original_role": role_map.get(m.member_id) or _resolve_member_role(m, role_map) or "",
            "task_count": 0,
            "total_days": 0.0,
            "role_mix": {},
            "top_tasks": [],
            "task_texts": [],
        }
        for m in team
    }

    for task in tasks or []:
        level_val = task.level.value if hasattr(task.level, "value") else str(task.level)
        if level_val != "L3":
            continue
        assigned = getattr(task, "assigned_to", []) or []
        if isinstance(assigned, str):
            assigned = [assigned]
        task_role = getattr(task, "assigned_role", None) or _derive_task_capability_label(task)
        days = float(getattr(task, "total_days", None) or getattr(task, "estimated_days", 0) or 0)
        for mid in assigned:
            if mid not in member_stats:
                member = team_by_id.get(mid)
                member_stats[mid] = {
                    "member_id": mid,
                    "member_name": getattr(member, "name", mid),
                    "original_role": role_map.get(mid) or "",
                    "task_count": 0,
                    "total_days": 0.0,
                    "role_mix": {},
                    "top_tasks": [],
                    "task_texts": [],
                }
            stat = member_stats[mid]
            stat["task_count"] += 1
            stat["total_days"] += days
            mix = stat["role_mix"].setdefault(task_role, {"role": task_role, "task_count": 0, "days": 0.0})
            mix["task_count"] += 1
            mix["days"] += days
            if len(stat["top_tasks"]) < 5:
                stat["top_tasks"].append({
                    "task_id": getattr(task, "task_id", ""),
                    "title": getattr(task, "title", ""),
                    "days": days,
                    "role": task_role,
                })
            stat["task_texts"].append(_task_role_context(task))

    results = []
    for mid, stat in member_stats.items():
        mixes = sorted(
            stat["role_mix"].values(),
            key=lambda x: (float(x.get("days", 0)), int(x.get("task_count", 0))),
            reverse=True,
        )
        project_role = _derive_project_role_name(stat, mixes, prd_text)
        role_label = project_role
        if mixes:
            rationale = (
                f"최종 배정 {stat['task_count']}개/{round(stat['total_days'], 1)}일의 제목·설명·산출물과 PRD 주제를 기준으로 "
                f"{role_label} 역할로 산출"
            )
        else:
            rationale = "최종 L3 배정 태스크가 없어 원 입력 역할을 유지"
        results.append({
            "member_id": stat["member_id"],
            "member_name": stat["member_name"],
            "original_role": stat["original_role"],
            "project_role": project_role,
            "project_role_label": role_label,
            "task_count": stat["task_count"],
            "total_days": round(stat["total_days"], 2),
            "role_mix": [
                {
                    "role": item["role"],
                    "role_label": _ROLE_LABELS.get(item["role"], item["role"]),
                    "task_count": item["task_count"],
                    "days": round(item["days"], 2),
                }
                for item in mixes
            ],
            "top_tasks": stat["top_tasks"],
            "rationale": rationale,
        })

    return sorted(results, key=lambda x: (-float(x["total_days"]), x["member_name"]))

# task의 assigned_role 문자열 → 배정 가능한 MemberRole.value 목록
_ROLE_COMPAT: dict = {
    # 소프트웨어 개발 직군
    "PM":                  ["PM", "Planner", "Business Analyst", "Operations Manager"],
    "Planner":             ["Planner", "PM", "Business Analyst", "Marketing Planner"],
    "Backend Developer":   ["Backend Developer", "Fullstack Developer", "Data Engineer"],
    "Frontend Developer":  ["Frontend Developer", "Fullstack Developer", "Mobile Developer"],
    "Fullstack Developer": ["Fullstack Developer", "Backend Developer", "Frontend Developer"],
    "Data Engineer":       ["Data Engineer", "Backend Developer", "Data Analyst"],
    "DevOps":              ["DevOps", "Backend Developer"],
    "Designer":            ["Designer", "Marketing Planner"],
    "QA Engineer":         ["QA Engineer", "Business Analyst"],
    # 비즈니스 / 마케팅 직군 — Planner도 기획성 업무에 매칭되도록 포함
    "Data Analyst":        ["Data Analyst", "Data Engineer", "Business Analyst", "Planner"],
    "Marketing Planner":   ["Marketing Planner", "Planner", "Designer", "Business Analyst", "PM"],
    "Business Analyst":    ["Business Analyst", "Planner", "PM", "Marketing Planner"],
    "Mobile Developer":    ["Mobile Developer", "Frontend Developer", "Fullstack Developer"],
    "Operations Manager":  ["Operations Manager", "Business Analyst", "PM", "Planner"],
}


def _find_role_candidates(task_role: str, team: list, member_role_map: dict = None) -> list:
    """assigned_role 호환 멤버 목록 반환 (경력 내림차순).
    member_role_map: {member_id: role_value} — team_members에서 role이 strip된 경우 복원용
    """
    compatible = _ROLE_COMPAT.get(task_role, [task_role])
    candidates = []
    for m in team:
        # role 해상: state_role_map → m.role → tech 추론
        role_value = _resolve_member_role(m, member_role_map)

        # 1. 직군이 직접 일치하거나 호환 리스트에 있는 경우
        if role_value and role_value in compatible:
            candidates.append(m)
            continue
        # 2. 직군은 다르지만 기술스택이나 강점에 핵심 키워드가 포함된 경우 (fallback)
        keywords = [c.split()[0].lower() for c in compatible if ' ' in c] + [c.lower() for c in compatible]
        m_text = " ".join(m.tech_stack + (m.strengths if isinstance(m.strengths, list) else [m.strengths])).lower()
        if any(kw in m_text for kw in keywords if len(kw) > 3):
            candidates.append(m)

    # 그래도 없으면 원본 후보군(전체 멤버) 반환 - 단, 경력순
    candidates.sort(key=lambda m: m.years_of_experience, reverse=True)
    return candidates or sorted(team, key=lambda m: m.years_of_experience, reverse=True)


_LENS_KEYWORDS: dict = {
    "planner": [
        "요구사항", "기획", "전략", "정책", "범위", "일정", "로드맵", "분석", "시장",
        "planning", "requirement", "strategy", "policy", "scope", "roadmap",
    ],
    "frontend": [
        "ui", "ux", "화면", "프론트", "모바일", "앱", "react", "vue", "next", "android",
        "ios", "고객", "접점", "컴포넌트", "frontend", "mobile",
    ],
    "backend": [
        "api", "서버", "백엔드", "db", "database", "데이터베이스", "인증", "권한", "연동",
        "시스템", "fastapi", "django", "spring", "node", "pipeline", "데이터", "etl",
    ],
    "designer": [
        "디자인", "브랜드", "figma", "시안", "프로토타입", "동선", "진열", "ux 리서치",
        "마케팅", "캠페인", "design", "brand", "prototype",
    ],
    "qa": [
        "qa", "테스트", "검증", "품질", "성능", "보안", "리스크", "파일럿", "측정",
        "test", "validation", "quality", "security", "monitoring",
    ],
}

_LENS_ROLE_PRIORS: dict = {
    "planner": {"PM", "Planner", "Business Analyst", "Marketing Planner", "Operations Manager"},
    "frontend": {"Frontend Developer", "Fullstack Developer", "Mobile Developer"},
    "backend": {"Backend Developer", "Fullstack Developer", "Data Engineer", "DevOps", "Data Analyst"},
    "designer": {"Designer", "Marketing Planner", "Business Analyst"},
    "qa": {"QA Engineer", "Business Analyst", "Operations Manager", "DevOps"},
}

_LENS_LABELS: dict = {
    "planner": "Planner",
    "frontend": "Frontend Developer",
    "backend": "Backend Developer",
    "designer": "Designer",
    "qa": "QA Engineer",
}

_LENS_LABELS_KO: dict = {
    "planner": "기획/요구사항",
    "frontend": "사용자 접점/UI",
    "backend": "시스템/API/데이터",
    "designer": "디자인/브랜딩",
    "qa": "품질/검증/리스크",
}


def _task_text(task) -> str:
    parts = [
        getattr(task, "title", ""),
        getattr(task, "description", ""),
        getattr(task, "assigned_role", ""),
        getattr(task, "required_role", ""),
        " ".join(getattr(task, "deliverables", []) or []),
        " ".join(getattr(task, "risk_factors", []) or []),
    ]
    return " ".join(str(p) for p in parts if p).lower()


def _member_text(member, member_role_map: dict = None) -> str:
    role = _resolve_member_role(member, member_role_map or {}) or ""
    parts = [
        role,
        " ".join(getattr(member, "tech_stack", []) or []),
        " ".join(getattr(member, "primary_skills", []) or []),
        " ".join(getattr(member, "strengths", []) or []),
        " ".join(getattr(member, "weaknesses", []) or []),
        getattr(member, "raw_resume_text", "") or "",
    ]
    return " ".join(str(p) for p in parts if p).lower()


def _lens_scores_for_text(text: str) -> Dict[str, float]:
    scores = {}
    for lens, keywords in _LENS_KEYWORDS.items():
        scores[lens] = sum(1.0 for kw in keywords if kw.lower() in text)
    return scores


def _dominant_lenses_for_task(task, max_lenses: int = 3) -> List[str]:
    scores = _lens_scores_for_text(_task_text(task))
    ranked = [lens for lens, score in sorted(scores.items(), key=lambda kv: kv[1], reverse=True) if score > 0]
    if not ranked:
        fallback_role = str(getattr(task, "assigned_role", "") or getattr(task, "required_role", "")).lower()
        if any(k in fallback_role for k in ("front", "mobile")):
            ranked = ["frontend"]
        elif any(k in fallback_role for k in ("back", "data", "devops")):
            ranked = ["backend"]
        elif any(k in fallback_role for k in ("design", "marketing")):
            ranked = ["designer"]
        elif any(k in fallback_role for k in ("qa", "test")):
            ranked = ["qa"]
        else:
            ranked = ["planner"]
    return ranked[:max_lenses]


def _derive_task_capability_label(task) -> str:
    """태스크 내용에서 필요한 검토/수행 역량 라벨을 사후 도출한다."""
    lens = _dominant_lenses_for_task(task, 1)[0]
    return _LENS_LABELS.get(lens, getattr(task, "required_role", "") or getattr(task, "assigned_role", "Planner"))


def _member_lens_score(member, lens: str, member_role_map: dict = None) -> float:
    text = _member_text(member, member_role_map)
    role = _resolve_member_role(member, member_role_map or {}) or ""
    score = sum(1.0 for kw in _LENS_KEYWORDS.get(lens, []) if kw.lower() in text)
    if role in _LENS_ROLE_PRIORS.get(lens, set()):
        score += 2.5
    score += min(float(getattr(member, "years_of_experience", 0) or 0), 10.0) * 0.08
    return score


def _capability_score_for_task(task, member, member_role_map: dict = None) -> float:
    task_lenses = _dominant_lenses_for_task(task, 3)
    task_blob = _task_text(task)
    member_blob = _member_text(member, member_role_map)
    score = 0.0
    for lens in task_lenses:
        lens_weight = 1.0 + min(_lens_scores_for_text(task_blob).get(lens, 0), 4) * 0.15
        score += _member_lens_score(member, lens, member_role_map) * lens_weight
    # 직접 기술/도메인 단어가 겹치면 보너스. 짧은 일반 단어는 제외한다.
    task_terms = {tok for tok in re.split(r"[^0-9a-zA-Z가-힣+#.]+", task_blob) if len(tok) >= 3}
    member_terms = {tok for tok in re.split(r"[^0-9a-zA-Z가-힣+#.]+", member_blob) if len(tok) >= 3}
    score += min(len(task_terms & member_terms), 8) * 0.35
    return score


def _child_l3_tasks(tasks: list, l2_id: str) -> list:
    return [
        t for t in tasks
        if (t.level.value if hasattr(t.level, "value") else str(t.level)) == "L3"
        and getattr(t, "parent_id", None) == l2_id
    ]


def _build_candidate_pools(
    tasks: list,
    team: list,
    member_role_map: dict = None,
    max_pool_size: int = 4,
) -> tuple:
    """역할명 고정 대신 L2/L3 작업 내용과 멤버 역량으로 후보 풀을 구성한다."""
    if not team:
        return {}, {}, {}

    l2_pools: Dict[str, List[str]] = {}
    task_pools: Dict[str, List[str]] = {}
    evidence: Dict[str, str] = {}
    l2_tasks = [
        t for t in tasks
        if (t.level.value if hasattr(t.level, "value") else str(t.level)) == "L2"
    ]
    l3_tasks = [
        t for t in tasks
        if (t.level.value if hasattr(t.level, "value") else str(t.level)) == "L3"
    ]

    for task in l3_tasks:
        scored = sorted(
            ((m, _capability_score_for_task(task, m, member_role_map)) for m in team),
            key=lambda item: item[1],
            reverse=True,
        )
        chosen = [m.member_id for m, _ in scored[:max(2, min(max_pool_size, len(team)))]]
        task_pools[task.task_id] = chosen
        if scored:
            best = scored[0][0]
            lenses = ", ".join(_dominant_lenses_for_task(task, 3))
            evidence[task.task_id] = f"작업 요구 역량({lenses})과 {best.name}의 기술/강점 유사도가 가장 높음"

    for l2 in l2_tasks:
        children = _child_l3_tasks(tasks, l2.task_id)
        bundle = children or [l2]
        aggregate: Dict[str, float] = {m.member_id: 0.0 for m in team}
        for child in bundle:
            for m in team:
                aggregate[m.member_id] += _capability_score_for_task(child, m, member_role_map)
        ranked = sorted(team, key=lambda m: (aggregate.get(m.member_id, 0.0), getattr(m, "years_of_experience", 0)), reverse=True)
        pool_size = max(2, min(max_pool_size, len(team)))
        pool = [m.member_id for m in ranked[:pool_size]]

        # 리스크 방어: 상위 점수가 한 명에게 몰려도 최소 2명 이상, 가능하면 다른 관점 1명 포함
        pool_roles = {_resolve_member_role(m, member_role_map or {}) for m in ranked if m.member_id in pool}
        if len(pool) < len(team) and len(pool_roles) <= 1:
            for m in ranked[pool_size:]:
                role = _resolve_member_role(m, member_role_map or {})
                if role not in pool_roles:
                    pool[-1] = m.member_id
                    break
        l2_pools[l2.task_id] = pool

    return l2_pools, task_pools, evidence


def _derive_l2_agent_mapping(tasks: list, l2_pools: dict, team: list, member_role_map: dict = None) -> dict:
    """L2마다 필요한 검토 렌즈를 선택한다. 렌즈는 직무 배정이 아니라 토론 관점이다."""
    mapping: Dict[str, List[str]] = {}
    team_by_id = {m.member_id: m for m in team}
    for l2_id, pool in l2_pools.items():
        l2 = next((t for t in tasks if t.task_id == l2_id), None)
        bundle = ([l2] if l2 else []) + _child_l3_tasks(tasks, l2_id)
        lens_scores: Dict[str, float] = {k: 0.0 for k in _LENS_KEYWORDS}
        for task in bundle:
            for lens, score in _lens_scores_for_text(_task_text(task)).items():
                lens_scores[lens] += score
        # 후보 중 실제 발화 가능한 역량이 있는 렌즈를 보강
        for mid in pool:
            member = team_by_id.get(mid)
            if not member:
                continue
            for lens in _LENS_KEYWORDS:
                member_score = _member_lens_score(member, lens, member_role_map)
                # 경력 기본점만으로 무관한 렌즈가 선택되는 것을 막는다.
                if member_score >= 0.8:
                    lens_scores[lens] += min(member_score, 3.0) * 0.2
        selected = [lens for lens, score in sorted(lens_scores.items(), key=lambda kv: kv[1], reverse=True) if score > 0][:3]
        if "qa" not in selected:
            text = " ".join(_task_text(t) for t in bundle)
            if any(k in text for k in ("테스트", "검증", "보안", "리스크", "파일럿", "품질")):
                selected = (selected + ["qa"])[:3]
        mapping[l2_id] = selected or ["planner", "qa"]
    return mapping


def _derive_l2_review_lenses(tasks: list, l2_pools: dict, team: list, member_role_map: dict = None) -> dict:
    """고정 직무명이 아닌 자유 역량 라벨로 L2 검토 관점을 도출한다."""
    raw = _derive_l2_agent_mapping(tasks, l2_pools, team, member_role_map)
    return {
        l2_id: [_LENS_LABELS_KO.get(lens, lens) for lens in lenses]
        for l2_id, lenses in raw.items()
    }


def _build_l2_calling_context(
    l2_agent_mapping: dict,
    l2_candidate_pools: dict,
    team: list,
    member_role_map: dict = None,
) -> dict:
    """L2별 검토 렌즈를 후보 풀 안의 실제 멤버에게 동적으로 연결한다."""
    team_by_id = {m.member_id: m for m in team}
    result: Dict[str, Dict[str, str]] = {}
    for l2_id, lenses in l2_agent_mapping.items():
        pool_ids = l2_candidate_pools.get(l2_id) or [m.member_id for m in team]
        pool_members = [team_by_id[mid] for mid in pool_ids if mid in team_by_id] or team
        used: set = set()
        context: Dict[str, str] = {}
        for lens in lenses:
            ranked = sorted(
                pool_members,
                key=lambda m: (
                    m.member_id in used,
                    -_member_lens_score(m, lens, member_role_map),
                    -float(getattr(m, "years_of_experience", 0) or 0),
                ),
            )
            if ranked:
                chosen = ranked[0]
                context[lens] = chosen.member_id
                used.add(chosen.member_id)
        result[l2_id] = context
    return result


def _get_member_role_str(member) -> Optional[str]:
    """MemberProfile 객체에서 role 문자열을 안전하게 추출하여 일관된 명칭으로 반환"""
    if not member or not getattr(member, 'role', None):
        return None
    # Enum인 경우 .value, 문자열인 경우 그대로 반환
    role_raw = getattr(member.role, 'value', str(member.role))
    # _ROLE_LABELS에 정의된 사람이면 매핑된 명칭 사용, 아니면 원본 사용
    return _ROLE_LABELS.get(role_raw, role_raw)


def _infer_role_from_tech(member) -> Optional[str]:
    """member.role이 없을 때 tech_stack/strengths로 직군 추론 (기본값 오염 방지용 최종 폴백)."""
    if not member:
        return None
    tech = " ".join(getattr(member, 'tech_stack', []) or []).lower()
    strengths = getattr(member, 'strengths', []) or []
    if isinstance(strengths, list):
        strengths_str = " ".join(strengths).lower()
    else:
        strengths_str = str(strengths).lower()
    text = tech + " " + strengths_str

    checks = [
        (["figma", "sketch", "photoshop", "adobe xd", "디자인"], "Designer"),
        (["react native", "android", "ios", "firebase", "모바일"], "Mobile Developer"),
        (["selenium", "cypress", "pytest", "postman", "k6", "테스트 자동화"], "QA Engineer"),
        (["tableau", "power bi", "pandas", "빅데이터 분석", "세그먼테이션"], "Data Analyst"),
        (["kafka", "spark", "airflow", "데이터 파이프라인", "etl"], "Data Engineer"),
        (["terraform", "k8s", "kubernetes", "aws", "devops", "ci/cd"], "DevOps"),
        (["react", "vue", "angular", "next.js", "nextjs", "tailwind", "typescript"], "Frontend Developer"),
        (["django", "fastapi", "spring", "node.js", "백엔드", "api 설계"], "Backend Developer"),
        (["crm", "google analytics", "sns 마케팅", "프로모션 기획"], "Marketing Planner"),
        (["erp", "wms", "재고관리", "pos", "매장 운영", "공급망"], "Operations Manager"),
        (["excel", "powerpoint", "리서치", "사업 전략"], "Business Analyst"),
    ]
    for keywords, role in checks:
        if any(k in text for k in keywords):
            return role
    return None


def _resolve_member_role(member, role_map: dict = None) -> Optional[str]:
    """여러 소스에서 멤버의 직군을 해상. 순서:
    1) state_role_map (원본 role 보존)
    2) member.role (stripped되지 않은 경우)
    3) tech_stack/strengths 기반 추론
    None 반환 시 호출측이 task_role로 폴백."""
    if member is None:
        return None
    if role_map:
        mapped = role_map.get(getattr(member, 'member_id', None))
        if mapped:
            return mapped
    raw = getattr(member, 'role', None)
    if raw:
        return getattr(raw, 'value', str(raw))
    return _infer_role_from_tech(member)


def _smart_assign(
    tasks: list,
    team: list,
    member_role_map: dict = None,
    locked_assignments: dict = None,
    l2_candidate_pools: dict = None,
    task_candidate_pools: dict = None,
) -> list:
    """
    실무 상식 기반 태스크 배정 후처리.

    핵심 원칙:
    - L1/L2는 요약(summary) 노드이므로 assigned_to=[] (담당자 없음)
    - L3(최하단 작업 패키지)에만 실제 담당자를 배정
    - locked_assignments에 있는 태스크는 PM 결정으로 잠금 — 재배정 불가

    적용 규칙 (L3 전용):
    1. 잠금 우선   — locked_assignments에 있으면 무조건 해당 멤버로 고정
    2. 후보 풀 우선 — L2/L3 후보 풀 안에서 역량 점수와 부하를 같이 평가
    3. LLM 존중   — LLM 배정은 후보/역량 점수가 크게 밀리지 않을 때 보너스로 반영
    4. 연속성 원칙  — 동일 부모(parent_id) 하위 태스크는 같은 담당자 유지
    5. 부하 균형    — 특정 멤버 집중을 점수 패널티로 완화
    """
    _locked = locked_assignments or {}
    _role_map = member_role_map or {}
    _l2_pools = l2_candidate_pools or {}
    _task_pools = task_candidate_pools or {}
    member_workload: dict = {m.member_id: 0.0 for m in team}
    parent_assignee: dict = {}   # parent_task_id → member_id (L3 연속성용)
    team_by_id = {m.member_id: m for m in team}

    def _level_order(t):
        v = t.level.value if hasattr(t.level, "value") else str(t.level)
        return {"L1": 0, "L2": 1, "L3": 2}.get(v, 1)

    sorted_tasks = sorted(tasks, key=_level_order)
    updated_map: dict = {}

    for t in sorted_tasks:
        task_role = t.assigned_role
        level_val = t.level.value if hasattr(t.level, "value") else str(t.level)

        # ── L1/L2: 요약 노드 — 담당자 없이 역할 카테고리만 유지 ──────────
        if level_val in ("L1", "L2"):
            updated_map[t.task_id] = t.model_copy(update={
                "assigned_to": [],
                "required_role": t.required_role or task_role,
            })
            continue

        # ── L3 이하: 실제 담당자 배정 ────────────────────────────────────
        pool_ids = list(dict.fromkeys(
            (_task_pools.get(t.task_id) or []) +
            (_l2_pools.get(getattr(t, "parent_id", None)) or [])
        ))
        candidates = [team_by_id[mid] for mid in pool_ids if mid in team_by_id]
        if not candidates:
            ranked_all = sorted(team, key=lambda m: _capability_score_for_task(t, m, _role_map), reverse=True)
            candidates = ranked_all[:max(2, min(4, len(ranked_all)))] or team

        # 규칙 1 — PM 잠금 배정 절대 우선
        if t.task_id in _locked:
            chosen = _locked[t.task_id]
            # 잠금된 멤버가 팀에 없으면 일반 로직으로 폴백
            if not any(m.member_id == chosen for m in team):
                chosen = None
        else:
            chosen = None

        if not chosen:
            current_assignee = t.assigned_to[0] if t.assigned_to else None
            if current_assignee and current_assignee in team_by_id and all(c.member_id != current_assignee for c in candidates):
                candidates.append(team_by_id[current_assignee])

        if not chosen:
            # 규칙 3 — 연속성: 부모와 같은 담당자가 후보이면 상속 보너스
            if t.parent_id and t.parent_id in parent_assignee:
                parent_mid = parent_assignee[t.parent_id]
                current_max_workload = max(list(member_workload.values()) or [0.0])
                if any(c.member_id == parent_mid for c in candidates) and member_workload.get(parent_mid, 0) <= current_max_workload * 1.15:
                    chosen = parent_mid

        if not chosen:
            current_assignee = t.assigned_to[0] if t.assigned_to else None
            avg_workload = sum(member_workload.values()) / len(member_workload) if member_workload else 0.0

            def _assignment_score(member) -> float:
                base = _capability_score_for_task(t, member, _role_map)
                workload = member_workload.get(member.member_id, 0.0)
                workload_penalty = 0.0 if avg_workload <= 0 else workload / max(avg_workload, 1.0)
                continuity_bonus = 1.2 if t.parent_id and parent_assignee.get(t.parent_id) == member.member_id else 0.0
                llm_bonus = 0.8 if current_assignee == member.member_id else 0.0
                return base + continuity_bonus + llm_bonus - workload_penalty

            best = max(candidates, key=_assignment_score)
            chosen = best.member_id

        member_workload[chosen] = member_workload.get(chosen, 0) + t.total_days
        if t.parent_id:
            parent_assignee[t.parent_id] = chosen

        update_fields: dict = {"assigned_to": [chosen], "required_role": t.required_role or task_role}
        update_fields["assigned_role"] = _derive_task_capability_label(t)
        updated_map[t.task_id] = t.model_copy(update=update_fields)

    # ── 0-task 팀원 구제: 놀고 있는 멤버가 있으면 부하 최고 멤버에서 이전 ──
    # (_ROLE_COMPAT 매핑 공백으로 특정 멤버가 아예 제외되는 상황 방지)
    idle_members = [m for m in team if member_workload.get(m.member_id, 0) == 0]
    if idle_members and team:
        for idle in idle_members:
            # 부하 최고 멤버의 L3 태스크 중 하나를 이전 (잠금된 태스크 제외)
            busiest = max(team, key=lambda m: member_workload.get(m.member_id, 0))
            if busiest.member_id == idle.member_id or member_workload.get(busiest.member_id, 0) <= 0:
                continue
            # busiest가 담당하는 L3 태스크 중 비잠금·낮은 중요도 우선 탐색
            donor_candidates = [
                t for t in updated_map.values()
                if (t.level.value if hasattr(t.level, "value") else str(t.level)) == "L3"
                and idle.member_id not in t.assigned_to
                and busiest.member_id in t.assigned_to
                and t.task_id not in _locked
            ]
            if not donor_candidates:
                continue
            # 중요도 Low > Medium > High 순 + 짧은 것 우선 (부하 최소 영향)
            def _prio(t):
                imp = {"low": 0, "medium": 1, "high": 2}.get(str(getattr(t, "importance", "medium")).lower(), 1)
                return (imp, float(getattr(t, "total_days", getattr(t, "estimated_days", 1)) or 1))
            donor = min(donor_candidates, key=_prio)
            moved_days = float(donor.total_days or donor.estimated_days or 1)
            donor_score = _capability_score_for_task(donor, team_by_id.get(busiest.member_id, busiest), _role_map)
            idle_score = _capability_score_for_task(donor, idle, _role_map)
            if donor_score > 0 and idle_score < donor_score * 0.65:
                continue
            # 태스크의 담당자 업데이트. 역할 라벨은 사람의 원직무가 아니라 태스크 요구역량으로 유지한다.
            updated_map[donor.task_id] = donor.model_copy(update={
                "assigned_to": [idle.member_id],
                "assigned_role": _derive_task_capability_label(donor),
            })
            member_workload[idle.member_id] = member_workload.get(idle.member_id, 0) + moved_days
            member_workload[busiest.member_id] = member_workload.get(busiest.member_id, 0) - moved_days

    # 원래 순서 유지
    return [updated_map.get(t.task_id, t) for t in tasks]


def _convert_json_to_wbs_tasks(data: list, team: list, role_map: dict = None) -> list:
    tasks = []
    # role_map: {member_id: role_value} — member.role이 None인 경우 대비
    effective_role_map = role_map or {}
    member_map = {}
    # 팀에서 가장 흔한 역할을 fallback 기본값으로 사용 (Backend 편향 방지)
    _team_roles = [
        (m.role.value if m.role else None) or effective_role_map.get(m.member_id)
        for m in team
    ]
    _team_roles = [r for r in _team_roles if r]
    fallback_role = max(set(_team_roles), key=_team_roles.count) if _team_roles else "Backend Developer"
    for m in team:
        r = (m.role.value if m.role else None) or effective_role_map.get(m.member_id, fallback_role)
        member_map.setdefault(r, []).append(m.member_id)

    all_ids = [m.member_id for m in team]

    for item in data:
        role = item.get("assigned_role") or fallback_role
        assignees = member_map.get(role, all_ids[:2])
        
        raw_imp = str(item.get("importance", "Medium")).strip().capitalize()
        if raw_imp not in ("High", "Medium", "Low"):
            raw_imp = "Medium"
        tasks.append(WBSTask(
            task_id=str(item.get("id")),
            level=WBSLevel(item.get("level", "L1")),
            parent_id=str(item.get("parent_id")) if item.get("parent_id") else None,
            title=item.get("title", "Untitled"),
            description=item.get("description", ""),
            assigned_to=assignees,
            assigned_role=role,
            estimated_days=float(item.get("estimated_days", 1.0)),
            buffer_days=0.0,
            total_days=float(item.get("estimated_days", 1.0)),
            dependencies=item.get("dependencies", []),
            start_week=item.get("start_week"),
            end_week=item.get("end_week"),
            deliverables=[f"{item.get('title')} 완료"],
            importance=raw_imp,
        ))
    
    # 자동 스케줄링 적용
    tasks = _calculate_automatic_schedule(tasks)
    return tasks

def _rollup_parent_dates(tasks: List[WBSTask]) -> List[WBSTask]:
    """
    Bottom-up roll-up: 자식 태스크의 실제 값으로 부모(L1/L2) 값을 재계산.
    - start_week / end_week: min/max
    - buffer_days, estimated_days, total_days: 자식 합산 (메트릭 일관성)
    자식이 없거나 값 미설정인 경우 기존 값 유지.
    """
    def children_of(parent_id: str) -> List[WBSTask]:
        return [t for t in tasks if t.parent_id == parent_id]

    # L2 먼저(L3 자식), 이후 L1(L2 자식) 순으로 bottom-up 처리
    for level in (WBSLevel.L2, WBSLevel.L1):
        for t in [x for x in tasks if x.level == level]:
            kids = children_of(t.task_id)
            if not kids:
                continue
            # 날짜 roll-up
            starts = [c.start_week for c in kids if c.start_week is not None]
            ends   = [c.end_week   for c in kids if c.end_week   is not None]
            if starts and ends:
                t.start_week = min(starts)
                t.end_week   = max(ends)
            # 일수 roll-up: L3 실제 값 집계
            t.estimated_days = round(sum(c.estimated_days for c in kids), 1)
            t.buffer_days    = round(sum(c.buffer_days    for c in kids), 1)
            t.total_days     = round(sum(c.total_days     for c in kids), 1)

    return tasks


def _calculate_automatic_schedule(tasks: List[WBSTask]) -> List[WBSTask]:
    """
    의존성(dependencies)과 계층 구조를 기반으로 시작/종료 주차를 자동 계산합니다.
    Pass 1: L3 → L2 → L1 순으로 의존성·순차성 기반 일정 결정 (top-down constraint)
    Pass 2: _rollup_parent_dates 로 L1/L2 기간을 자식 실제 범위로 재보정 (bottom-up roll-up)
    """
    if not tasks:
        return tasks

    task_map = {t.task_id: t for t in tasks}
    processed_ids = set()

    l1_tasks = sorted([t for t in tasks if t.level == WBSLevel.L1], key=lambda x: x.task_id)

    def resolve_schedule(tid, visited=None):
        if visited is None: visited = set()
        if tid in visited or tid in processed_ids: return
        visited.add(tid)

        t = task_map.get(tid)
        if not t: return

        # 1. 시작 주차 결정 (최소 1)
        start_w = t.start_week or 1

        # 1-1. 명시적 의존성 FS (선행 종료 + 1주 버퍼)
        if t.dependencies:
            for dep_id in t.dependencies:
                resolve_schedule(dep_id, visited)
                dep_task = task_map.get(dep_id)
                if dep_task and dep_task.end_week:
                    start_w = max(start_w, dep_task.end_week + 1)

        # 1-2. L1 Phase 간 순차성 (L1-01 → L1-02 ...)
        if t.level == WBSLevel.L1:
            idx = l1_tasks.index(t)
            if idx > 0:
                prev_l1 = l1_tasks[idx - 1]
                resolve_schedule(prev_l1.task_id, visited)
                if prev_l1.end_week:
                    start_w = max(start_w, prev_l1.end_week + 1)

        # 1-3. 부모 제약 (자식은 부모 시작일 이후)
        if t.parent_id:
            resolve_schedule(t.parent_id, visited)
            parent_task = task_map.get(t.parent_id)
            if parent_task and parent_task.start_week:
                start_w = max(start_w, parent_task.start_week)

        # 2. 종료 주차 (total_days = estimated + buffer)
        duration_weeks = math.ceil(t.total_days / 5)
        end_w = start_w + max(0, duration_weeks - 1)

        t.start_week = start_w
        t.end_week   = end_w
        processed_ids.add(tid)

    for t in tasks:
        resolve_schedule(t.task_id)

    # Pass 2: 인원별 일정 압축 (Person-Level Compaction)
    # 같은 사람에게 배정된 L3 태스크를 시간순 정렬 후,
    # 이전 태스크 종료 직후 다음 태스크가 시작되도록 조정.
    # 단, 태스크 의존성 제약(선행 태스크 종료 후 시작)은 위반하지 않음.
    tasks = _compact_person_schedule(tasks, task_map)

    # Pass 3: 부모 일정을 자식 실제 범위로 roll-up
    tasks = _rollup_parent_dates(tasks)

    return tasks


def _compact_person_schedule(tasks: List[WBSTask], task_map: dict) -> List[WBSTask]:
    """
    인원별 일정 압축: 한 사람의 태스크 사이에 유휴 기간이 없도록 재배치.
    - L3 태스크만 대상 (L1/L2는 roll-up으로 자동 조정)
    - 의존성 제약: 선행 태스크 end_week 이전으로는 당길 수 없음
    - 부모 제약: 부모 start_week 이전으로는 당길 수 없음
    """
    # 인원별 L3 태스크 그룹핑
    person_tasks: dict = {}  # member_id → [task, ...]
    for t in tasks:
        level_val = t.level.value if hasattr(t.level, 'value') else str(t.level)
        if level_val != "L3":
            continue
        for mid in (t.assigned_to or []):
            person_tasks.setdefault(mid, []).append(t)

    for mid, ptasks in person_tasks.items():
        if len(ptasks) < 2:
            continue

        # 현재 start_week 기준 정렬
        ptasks.sort(key=lambda t: (t.start_week or 1, t.task_id))

        # 이 사람의 가용 시작 주차 (첫 태스크의 시작부터)
        person_next_available = ptasks[0].start_week or 1

        for t in ptasks:
            # 의존성으로 인한 최소 시작 주차
            dep_min = 1
            for dep_id in (t.dependencies or []):
                dep_task = task_map.get(dep_id)
                if dep_task and dep_task.end_week:
                    dep_min = max(dep_min, dep_task.end_week + 1)

            # 부모 제약
            parent_min = 1
            if t.parent_id:
                parent_task = task_map.get(t.parent_id)
                if parent_task and parent_task.start_week:
                    parent_min = parent_task.start_week

            # 실제 시작 = max(인원 가용, 의존성 최소, 부모 최소)
            actual_start = max(person_next_available, dep_min, parent_min)

            duration_weeks = max(1, math.ceil(t.total_days / 5))
            t.start_week = actual_start
            t.end_week = actual_start + duration_weeks - 1

            # 이 사람의 다음 가용 주차 갱신
            person_next_available = t.end_week + 1

    return tasks

def _apply_reassignments(
    tasks: list,
    reassignments: dict,
    valid_member_ids: set,
    team: list = None,
    member_role_map: dict = None,
):
    """
    재배정 요청을 WBS 태스크에 적용합니다.
    reassignments: {task_id: new_member_id}
    반환: (updated_tasks, [(task_id, new_member_id), ...])
    """
    if not reassignments:
        return tasks, []
    updated = []
    log = []
    for t in tasks:
        new_mid = reassignments.get(t.task_id)
        if new_mid and new_mid in valid_member_ids:
            update_fields = {
                "assigned_to": [new_mid],
                "assigned_role": _derive_task_capability_label(t),
            }
            updated.append(t.model_copy(update=update_fields))
            log.append((t.task_id, new_mid))
        else:
            updated.append(t)
    return updated, log


def _apply_dynamic_buffers(tasks: list, buffers: dict) -> list:
    updated = []
    for t in tasks:
        if t.task_id not in buffers:
            updated.append(t)
            continue
        b_info = buffers.get(t.task_id, {})
        new_buffer = float(b_info.get("buffer_days", 0.0))
        risk = b_info.get("risk", "")
        
        if new_buffer == 0:
            if "Backend" in t.assigned_role: new_buffer = 1.0
            if "Frontend" in t.assigned_role: new_buffer = 1.0
        
        updated.append(t.model_copy(update={
            "buffer_days": new_buffer,
            "total_days": t.estimated_days + new_buffer,
            "buffer_rationale": risk or "기본 버퍼 반영",
            "risk_factors": [risk] if risk else []
        }))
    return updated

def _generate_initial_wbs_fallback(prd, team) -> list:
    # 팩백용 기본 2단계 구조
    return [] 
