"""
Phase 1: WBS Generation Agent
PRD와 회의록 등을 분석하여 프로젝트의 순수 WBS 구조(JSON)를 도출합니다.
"""
import json
import os
import re
from typing import Any, List
from datetime import datetime

def repair_truncated_json(text: str) -> str:
    """반만 온 JSON(리스트 등)을 강제로 닫고, 요소 간 누락된 콤마 등을 보정합니다."""
    text = text.strip()
    if not text: return "[]"
    
    # 1. 태스크 직후 닫는 중괄호가 연달아 나오는 패턴 보정 (}, { -> } , {)
    # LLM이 가끔 콤마를 빼고 다음 객체를 시작함
    text = re.sub(r'\}\s*\{', '}, {', text)
    
    # 2. 마지막으로 완전히 닫힌 JSON 객체(})까지만 자르기
    #    키-값 쌍 중간에서 잘린 경우 (예: "title": "...) 를 안전하게 처리
    last_complete_obj = text.rfind('}')
    if last_complete_obj != -1:
        # } 이후 불완전한 텍스트 제거
        candidate = text[:last_complete_obj + 1]
        # 대괄호로 감싸져 있는지 확인
        first_bracket = candidate.find('[')
        if first_bracket != -1:
            candidate = candidate[first_bracket:]
            # trailing comma 정리 후 대괄호 닫기
            candidate = candidate.rstrip().rstrip(',')
            if not candidate.endswith(']'):
                candidate += ']'
            # 빠른 검증: 파싱 가능한지 확인
            try:
                json.loads(candidate)
                return candidate
            except Exception:
                pass  # 아래 일반 로직으로 폴백
    
    braces = 0
    brackets = 0
    in_string = False
    escape = False
    repaired_idx = len(text)

    for i, char in enumerate(text):
        if char == '"' and not escape:
            in_string = not in_string
        elif char == '\\' and not escape:
            escape = True
            continue
        else:
            escape = False

        if not in_string:
            if char == '{': braces += 1
            elif char == '}': braces -= 1
            elif char == '[': brackets += 1
            elif char == ']': brackets -= 1
            
            # 파싱 불가능한 지점이나 너무 많은 닫는 괄호 방어
            if braces < 0 or brackets < 0:
                repaired_idx = i
                break
    
    # 문제 발생 지점 이전까지만 슬라이싱
    repaired_text = text[:repaired_idx]
    
    # 문자열 닫기
    if in_string: repaired_text += '"'
    
    # 후처리 (trailing comma 등)
    repaired_text = repaired_text.rstrip().rstrip(',')
    
    # 괄호 밸런스 맞추기
    # 우선 열려있는 중괄호 닫기
    current_braces = repaired_text.count('{') - repaired_text.count('}')
    while current_braces > 0:
        repaired_text += '}'
        current_braces -= 1
        
    # 대괄호 닫기
    current_brackets = repaired_text.count('[') - repaired_text.count(']')
    while current_brackets > 0:
        repaired_text += ']'
        current_brackets -= 1
        
    return repaired_text

from langsmith import traceable

from agents.state import WBSState
from agents.llm_config import get_llm, normalize_content
from schemas.wbs_schema import DebateMessage, AgentRole, WBSTask, WBSLevel
from agents.supervisor_agent import _calculate_automatic_schedule


def _detect_project_type(prd) -> str:
    """
    PRD 컨텍스트에서 프로젝트 유형 추론.
    반환값: "dev" (소프트웨어 개발) | "business" (비즈니스/마케팅)
    """
    if not prd:
        return "dev"
    combined = " ".join([
        prd.project_goal or "",
        " ".join(prd.key_features or []),
        " ".join(prd.tech_stack_requirements or []),
        prd.raw_text or "",
    ]).lower()

    dev_keywords = [
        "api", "서버", "개발", "프론트엔드", "백엔드", "배포", "database",
        "github", "docker", "kubernetes", "fastapi", "react", "vue", "django",
        "스프링", "spring", "마이크로서비스", "ci/cd",
    ]
    biz_keywords = [
        "마케팅", "프로모션", "매출", "매장", "빅데이터 분석", "고객 분석",
        "브랜딩", "소매", "유기농", "마트", "판매전략", "경영지원",
        "crm", "캠페인", "시장조사", "물류", "재고", "운영",
    ]
    dev_score = sum(1 for kw in dev_keywords if kw in combined)
    biz_score = sum(1 for kw in biz_keywords if kw in combined)
    return "business" if biz_score > dev_score else "dev"


_ROLE_LIST_DEV = (
    "Backend Developer, Frontend Developer, Data Engineer, DevOps, "
    "Designer, QA Engineer, PM, Planner, Fullstack Developer"
)
_ROLE_LIST_BIZ = (
    "Marketing Planner, Data Analyst, Business Analyst, Designer, "
    "Operations Manager, Mobile Developer, PM, QA Engineer, Data Engineer"
)

_PHASE_GUIDE_DEV = """1. L1 Phase는 전체 프로젝트를 5~7개의 핵심 단계로 구성하십시오. (예: 요구분석/환경설정, 인프라/DB 구축, 핵심 비즈니스 로직 개발, 프론트엔드 구현, QA/테스트, 배포/운영)
2. 각 L1 Phase 하위에 3~4개의 L2 기능 그룹을 배치하십시오. ⛔ L1 하나당 L2가 2개 미만인 것은 금지됩니다.
3. 각 L2 기능 그룹 하위에 3~4개의 L3 세부 태스크를 배치하십시오. ⛔ L2 하나에 L3가 2개 미만인 것은 금지됩니다."""

_PHASE_GUIDE_BIZ = """1. L1 Phase는 전체 프로젝트를 5~7개의 핵심 단계로 구성하십시오. (예: 시장조사/데이터 수집, 고객 분석/세분화, 전략 기획, 브랜딩/콘텐츠 제작, 채널별 마케팅 실행, 성과 측정/분석, 최종 보고)
2. 각 L1 Phase 하위에 3~4개의 L2 기능 그룹을 배치하십시오. ⛔ L1 하나당 L2가 2개 미만인 것은 금지됩니다.
3. 각 L2 기능 그룹 하위에 3~4개의 L3 세부 태스크를 배치하십시오. ⛔ L2 하나에 L3가 2개 미만인 것은 금지됩니다."""

_L3_STUB_BY_ROLE = {
    "Backend Developer":   ("핵심 로직 구현 및 단위 테스트", "백엔드 기능 구현 및 기본 검증"),
    "Frontend Developer":  ("화면 구현 및 상태 연동", "프론트엔드 UI 구현 및 API 연동"),
    "Fullstack Developer": ("클라이언트·서버 통합 구현", "엔드 투 엔드 기능 구현"),
    "Mobile Developer":    ("모바일 화면 구현 및 통합", "모바일 앱 기능 구현 및 디바이스 검증"),
    "Data Engineer":       ("데이터 파이프라인 구성 및 검증", "ETL·연동 구성 및 품질 검증"),
    "Data Analyst":        ("지표 정의 및 탐색적 분석", "데이터 분석 및 인사이트 도출"),
    "DevOps":              ("인프라 프로비저닝 및 배포", "환경 구성 및 배포 파이프라인 구축"),
    "Designer":            ("시안 제작 및 핸드오프", "디자인 가이드 작성 및 개발팀 핸드오프"),
    "QA Engineer":         ("테스트 케이스 작성 및 실행", "기능·통합 테스트 수행 및 결함 관리"),
    "PM":                  ("진행 관리 및 이해관계자 리뷰", "작업 조율·리뷰 및 리스크 관리"),
    "Planner":             ("상세 요구사항 정리 및 문서화", "요구사항 구체화 및 기준 문서 작성"),
    "Marketing Planner":   ("캠페인 기획 및 실행 계획 수립", "프로모션 기획 상세화 및 실행 준비"),
    "Business Analyst":    ("요구사항 분석 및 정책 문서화", "업무 흐름 분석 및 정책·기준 정리"),
    "Operations Manager":  ("운영 프로세스 설계 및 적용", "운영 가이드 수립 및 현장 적용"),
}


def _retry_missing_l3(tasks: list, llm, team: list) -> tuple:
    """L2에 자식 L3가 누락된 경우, 해당 L2들만 모아 초점화된 재프롬프트로 L3 생성 요청.

    같은 백본(현재 LLM)으로 한 번 더 호출한다. 기존 L1 부족 재시도(`wbs_gen_node`)와
    동일 패턴이며, Gemini/Gemma 어느 쪽이든 같은 조건에서 같은 방식으로 발동한다.
    강한 백본은 첫 호출에서 L3를 잘 만들어 발동 안 함, 약한 백본에서 주로 발동.

    Returns:
        (병합된 tasks, 보강된 L3 개수)
    """
    def _lvl(t):
        return t.level.value if hasattr(t.level, "value") else str(t.level)

    l3_parents: set = {t.parent_id for t in tasks if _lvl(t) == "L3" and t.parent_id}
    missing_l2 = [t for t in tasks if _lvl(t) == "L2" and t.task_id not in l3_parents]
    if not missing_l2:
        return tasks, 0

    l2_block = "\n".join(
        f"- {t.task_id} | title: {t.title} | role: {getattr(t, 'assigned_role', '') or getattr(t, 'required_role', '')} | est: {getattr(t, 'estimated_days', 0)}일 | desc: {(getattr(t, 'description', '') or '')[:80]}"
        for t in missing_l2
    )
    retry_prompt = f"""이전 응답에서 다음 L2 기능그룹들에 **L3 세부 태스크가 누락**되었습니다.
각 L2마다 **2~3개의 구체적인 L3 세부 태스크**를 생성하십시오.

[L3 누락 L2 목록 ({len(missing_l2)}개)]
{l2_block}

[출력 규칙]
- 각 L3의 "parent_id"는 위 L2 ID와 정확히 일치 (L2-XX-YY → 자식은 parent_id=L2-XX-YY)
- "id"는 "L3-XX-YY-01", "L3-XX-YY-02" 형식 (L2-XX-YY의 자식)
- "level": "L3"
- "estimated_days": 3~7일 범위 정수
- "assigned_role": 상위 L2의 role과 동일 또는 호환 직군
- "description": 태스크 목적/활동/산출물 1문장
- "importance": "High" | "Medium" | "Low"
- "dependencies": 선행 L3 id 리스트 (없으면 [])

응답은 반드시 아래 JSON 리스트 형식만, L3 항목만 반환하십시오:
```json
[
  {{"id": "L3-01-01-01", "parent_id": "L2-01-01", "level": "L3", "title": "...", "description": "...", "estimated_days": 5, "assigned_role": "...", "importance": "Medium", "dependencies": []}}
]
```"""
    try:
        resp = normalize_content(llm.invoke([{"role": "user", "content": retry_prompt}]).content)
    except Exception as e:
        print(f"  [L3 재시도] LLM 호출 실패: {e}")
        return tasks, 0

    # JSON 추출: 코드 펜스 → [ ... ] 블록 → repair
    json_str = None
    m = re.search(r"```json\s*(.*?)```", resp, re.DOTALL)
    if m:
        json_str = m.group(1).strip()
    else:
        s, e = resp.find('['), resp.rfind(']')
        if s != -1 and e > s:
            json_str = resp[s:e + 1]
        elif s != -1:
            json_str = repair_truncated_json(resp[s:])
    if not json_str:
        return tasks, 0

    try:
        data = json.loads(json_str)
    except Exception:
        try:
            data = json.loads(repair_truncated_json(json_str))
        except Exception as e:
            print(f"  [L3 재시도] JSON 파싱 실패: {e}")
            return tasks, 0

    valid_l2_ids = {t.task_id for t in missing_l2}
    try:
        new_tasks = _convert_json_to_wbs_tasks(data, team)
    except Exception as e:
        print(f"  [L3 재시도] task 변환 실패: {e}")
        return tasks, 0

    # 실제 L3이고 누락 L2의 자식인 것만 채택 (중복 id 방지)
    existing_ids = {t.task_id for t in tasks}
    filtered: list = []
    for t in new_tasks:
        if _lvl(t) != "L3":
            continue
        if t.parent_id not in valid_l2_ids:
            continue
        if t.task_id in existing_ids:
            continue
        filtered.append(t)
        existing_ids.add(t.task_id)

    return tasks + filtered, len(filtered)


def _ensure_l3_coverage(tasks: list) -> tuple:
    """모든 L2에 자식 L3가 0개이면 기본 L3 stub 1개를 자동 합성한다.

    목적: 백본 LLM이 L3 분해를 불완전하게 수행해도 `supervisor_task_match`와
    `_smart_assign`의 L3 기반 배정 로직이 정상 동작하도록 보장하는 공통 안전망.
    프롬프트와 하이퍼파라미터는 손대지 않으므로 백본 간 비교 실험의 공정성
    (독립변수 = 백본 모델)이 유지된다. 강한 백본은 거의 발동하지 않고 약한
    백본에서 자주 발동하므로, 발동 횟수 자체가 "L3 분해 능력" 지표가 된다.

    산출 포맷: 실제 LLM이 생성한 L3와 외형상 구분되지 않도록 ID·제목·설명을
    구성한다 (고정 접미사·라벨 없음, 역할별 짧은 실행 문구 사용).

    Returns:
        (보강된 tasks, 합성된 L3 개수)
    """
    def _lvl(t):
        return t.level.value if hasattr(t.level, "value") else str(t.level)

    # 실제 L3 자식이 있는 부모 집합 — 기존 L3가 하나라도 있으면 합성 skip
    l3_children_by_parent: dict = {}
    for t in tasks:
        if _lvl(t) == "L3" and t.parent_id:
            l3_children_by_parent.setdefault(t.parent_id, []).append(t)

    default_stub = ("세부 실행 작업 수행", "상위 기능 그룹의 세부 실행 작업을 수행합니다.")

    synth: list = []
    for t in tasks:
        if _lvl(t) != "L2" or t.task_id in l3_children_by_parent:
            continue
        l2_suffix = t.task_id.replace("L2-", "", 1)
        stub_id = f"L3-{l2_suffix}-01"
        est = max(1.0, float(getattr(t, "estimated_days", 1) or 1) * 0.8)
        role = (getattr(t, "assigned_role", "") or getattr(t, "required_role", "") or "PM")
        title, desc = _L3_STUB_BY_ROLE.get(role, default_stub)
        synth.append(WBSTask(
            task_id=stub_id,
            level=WBSLevel.L3,
            parent_id=t.task_id,
            title=title,
            description=desc,
            assigned_to=[],
            required_role=role,
            assigned_role=role,
            estimated_days=est,
            total_days=est,
            importance=getattr(t, "importance", "Medium"),
        ))

    return tasks + synth, len(synth)


@traceable(name="Phase1_WBS_Gen")
def wbs_gen_node(state: WBSState) -> WBSState:
    """
    Phase 1: WBS 초안 생성기 (초기 생성 및 에이전트 피드백 기반 재설계 모두 담당)
    """
    model_id = state.get("model_config", {}).get("wbs_gen")
    llm = get_llm(temperature=0.35, max_tokens=65000, model_id=model_id)
    prd = state.get("prd") or state["prd"]  # 필수값: 없으면 KeyError가 올바른 동작
    team = state.get("team_members") or []
    rag_context = "\n".join(state.get("rag_reference_wbs", []))

    # 재설계 여부 확인
    is_revision = state.get("wbs_revision_needed", False)
    revision_hints = state.get("wbs_revision_hints", [])
    current_revision = state.get("current_wbs_revision", 0)
    if is_revision:
        current_revision += 1

    # 역할(role)은 의도적으로 제외 — WBS Gen Agent도 기술·강점만 보고 assigned_role을 독립적으로 결정하도록
    team_summary = "\n".join([
        f"- {m.name} ({m.years_of_experience}년): {', '.join(m.tech_stack[:3])}"
        for m in team
    ])

    budget_weeks = getattr(prd, 'budget_weeks', None) or 12
    budget_days_total = budget_weeks * 5
    deadline_hint = f"마감일: {prd.deadline} / " if getattr(prd, 'deadline', None) else ""
    constraints_hint = "\n".join(getattr(prd, 'special_constraints', []) or []) or "없음"

    revision_context = ""
    if is_revision and revision_hints:
        revision_context = "\n[⚠️ 이전 에이전트 검토 결과 — 반드시 아래 사항을 반영하여 재설계하세요]\n" + "\n".join(f"- {h}" for h in revision_hints) + "\n"

    # 프로젝트 유형 감지 → Phase 지침 및 역할 목록 동적 결정
    project_type = _detect_project_type(prd)
    phase_guide = _PHASE_GUIDE_BIZ if project_type == "business" else _PHASE_GUIDE_DEV
    role_list = _ROLE_LIST_BIZ if project_type == "business" else _ROLE_LIST_DEV

    prompt = f"""[WBS Gen Agent] 실무급 3단계 WBS {"재설계" if is_revision else "설계"} ({"수정 " + str(current_revision) + "차" if is_revision else "최초 생성"})
프로젝트: {prd.project_name}
목표: {prd.project_goal}
핵심 기능/과제: {", ".join(prd.key_features)}
{deadline_hint}예산 기준 총 기간: {budget_weeks}주 ({budget_days_total}일)
팀 구성 ({len(team)}명): {team_summary}
특수 제약사항: {constraints_hint}
참고 WBS 데이터: {rag_context or '없음'}{revision_context}

[지침]
{phase_guide}
1. **WBS 깊이와 상세도 (Depth & Breadth) - 중간 규모**:
   - **태스크 밀도(Density) 규칙**: 전체 태스크 수를 **50~80개 범위**로 유지하십시오. 너무 많거나 너무 적으면 실패입니다.
   - 프로젝트 구성을 **5~7개의 L1 Phase**로, 각 L1 하위에 **3~4개의 L2 기능그룹**을, 각 L2 하위에 **3~4개의 구체적인 L3 세부 태스크**를 배치하십시오.
   - ⛔ **금지 규칙**: L1 하나당 L2가 2개 미만이거나, L2 하나당 L3가 2개 미만인 경우.
   - ⛔ 같은 유형의 태스크를 단순 반복하지 마십시오. 각 L3 태스크는 고유한 산출물과 검증 방법을 가져야 합니다.
   - 유사한 세부 작업은 하나의 L3로 묶어 지나친 분할을 피하십시오.
   - L3는 실제 담당자가 **3~7일** 내에 완료할 수 있는 구체적인 작업 단위여야 합니다.
3. **프로젝트 순차성(Sequencing) 강화**: 모든 업무가 프로젝트 초반에 집중되지 않도록, Phase 간의 논리적 선후 관계를 고려하여 일정을 설계하십시오. 특히 L1 Phase들이 합리적인 순서로 배치되도록 하십시오.
4. **현실적인 일정 산정 (보수적 접근)**:
   - 단순 구현 시간만 계산하지 말고, **[분석 - 설계 - 구현 - 코드리뷰/QA - 버그 수정]**의 전체 주기를 고려하여 보수적으로 산정하십시오.
   - L1 Phase별 estimated_days: 최소 {max(15, budget_days_total // 5)}일 ~ 최대 {min(80, budget_days_total // 2)}일
   - L2 기능그룹 estimated_days: 최소 5일 ~ 최대 25일
   - L3 세부 태스크 estimated_days: 최소 3일 ~ 최대 12일
   - L1들의 estimated_days 합계가 {budget_days_total}일({budget_weeks}주)에 근접하거나 약간 상회(버퍼 고려)하도록 배분하십시오.
5. **의존성(Dependencies)**: 반드시 `dependencies` 필드를 활용하여 선행되어야 하는 태스크 ID 목록을 명시하십시오. (예: L3-01-01-02가 완료되어야 L3-01-01-03 가능)
6. assigned_role은 각 태스크 성격에 맞게 구체적으로 지정 ({role_list})
   - L1(Phase), L2(기능그룹)의 assigned_role은 해당 영역의 주요 직군 카테고리입니다 (실제 담당자 배정은 L3에서만 이루어짐)
   - L3 세부 태스크에만 실제 담당자가 배정되므로, L3의 assigned_role을 가장 구체적으로 지정하십시오.
7. **description의 전문성**: "무엇을(목적), 어떻게(활동), 무엇이 나오는지(산출물), 어떻게 확인하는지(검증)"를 포함하여 상세히 기술하십시오. 
   - 예: "고객 인증 모듈 개발: OAuth2 기반 소셜 로그인을 구현하며, 완료 시 JWT 토큰 발급 API와 Postman 테스트 결과서가 산출됨."
8. 반드시 아래 JSON 리스트 형식으로만 답변하십시오.

[응답 JSON 필드 가이드]
- "id": "L1-01", "L2-01-01", "L3-01-01-01" 형식
- "title": 구체적인 태스크명
- "description": 해당 태스크의 목적·범위·산출물을 1~2문장으로
- "level": "L1", "L2", "L3"
- "parent_id": L1은 null, L2/L3은 상위 ID 문자열
- "estimated_days": 위 일정 기준에 따른 현실적인 작업일 (정수)
- "assigned_role": 담당 직군 (구체적으로)
- "dependencies": [선행_태스크_ID_목록] (없으면 빈 리스트 [])
- "importance": "High" | "Medium" | "Low" (L1/L2/L3 모두 필수 기입)
  * High  (전체의 약 25~35%): **크리티컬 패스(Critical Path)** 상의 태스크, 기술적 블로커, 외부 의존성이 높은 핵심 아키텍처 (예: DB 스키마 확정, 핵심 비즈니스 로직, 인프라 배포 자동화)
  * Medium(전체의 약 45~55%): 일반 기능 개발, UI 컴포넌트 구현, 단위 테스트 케이스, 상세 문서화
  * Low   (전체의 약 15~25%): 부가 기능, 성능 최적화(Non-critical), 보조 스크립트, 관리 도구
  * ⚠️ 전문적인 WBS일수록 High가 프로젝트 초반 핵심 기반 시설에 집중됩니다.
- "start_week": 시작 주차 (1부터 시작하는 정수)
- "end_week": 종료 주차 (start_week + (estimated_days/5) 올림 형태)
- "deliverables": [해당 태스크 완료 시 생성되는 구체적 산출물 명칭 리스트 (최소 1-2개)]
- "risk_factors": [해당 태스크 진행 시 발생 가능한 주요 리스크 요인 리스트 (L3 필수)]

[순차성 및 스케줄링 핵심 규칙]
- 모든 L1 Phase가 주차 1에 몰리지 않게 하십시오. (Phase 1: 1~2주, Phase 2: 3~4주... 등 순차 배분)
- 하위 태스크(L2, L3)는 반드시 상위 Phase의 주차 범위 내에 있어야 하며, 의존 관계를 엄격히 따져 시작일을 뒤로 밀어주십시오.

출력은 반드시 마크다운 코드 블록(```json ... ```) 안에 담긴 JSON 리스트여야 합니다."""

    response = normalize_content(llm.invoke([{"role": "user", "content": prompt}]).content)
    
    # JSON 추출 보강
    json_str = response.strip()
    if "```json" in json_str:
        json_str = json_str.split("```json")[1].split("```")[0].strip()
    elif "```" in json_str:
        json_str = json_str.split("```")[1].split("```")[0].strip()
        
    initial_tasks = []
    
    # DEBUG: Keep transient LLM responses out of the repository root.
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    debug_dir = os.path.join(project_root, "generated", "debug")
    os.makedirs(debug_dir, exist_ok=True)
    debug_path = os.path.join(debug_dir, "llm_debug_wbs_gen.txt")
    with open(debug_path, "w", encoding="utf-8") as f:
        f.write(f"--- LLM RESPONSE ---\n{response}\n--- END RESPONSE ---")
        
    # 1차 시도: Markdown 추출
    json_str_match = re.search(r"```json\s*(.*?)```", response, re.DOTALL)
    json_str = json_str_match.group(1) if json_str_match else response
    
    # 2차 시도 (추출 실패 시 rfind 활용)
    if not json_str_match:
        start_idx = response.find('[')
        end_idx = response.rfind(']')
        if start_idx != -1:
            if end_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx+1]
            else:
                # 짤린 경우 -> 복구 시도
                json_str = repair_truncated_json(response[start_idx:])

    # 3차 시도: 짤린 JSON 강제 복구 (이미 json_str이 정해졌더라도 재시도)
    try:
        tasks_data = json.loads(json_str)
        initial_tasks = _convert_json_to_wbs_tasks(tasks_data, team)
    except Exception as e:
        print(f"Standard JSON Parsing Error: {e}")
        try:
            repaired_json = repair_truncated_json(json_str)
            tasks_data = json.loads(repaired_json)
            initial_tasks = _convert_json_to_wbs_tasks(tasks_data, team)
        except Exception:
            # 4차 시도: 정규표현식을 이용한 필드 추출 (최후의 수단 - 따옴표 누락 등 대응)
            print("Trying Regex-based task extraction as last resort...")
            initial_tasks = _extract_tasks_via_regex(response, team)
        
    # ── L1 부족 시 1회 재시도 (Flash Lite 토큰 한계 대응) ──
    l1_count = len([t for t in initial_tasks if (t.level.value if hasattr(t.level, "value") else str(t.level)) == "L1"])
    if initial_tasks and l1_count < 3 and not is_revision:
        print(f"⚠️ L1 Phase가 {l1_count}개뿐 (최소 3개 필요). 재시도합니다...")
        retry_prompt = f"""이전 응답에서 L1 Phase가 {l1_count}개뿐이었습니다. 반드시 7~9개의 L1 Phase를 포함하는 완전한 WBS를 생성하세요.
L2/L3 세부사항은 간략하게 하되, L1 Phase는 반드시 7개 이상 포함하세요.

{prompt}"""
        try:
            retry_response = normalize_content(llm.invoke([{"role": "user", "content": retry_prompt}]).content)
            retry_match = re.search(r"```json\s*(.*?)```", retry_response, re.DOTALL)
            retry_json = retry_match.group(1) if retry_match else retry_response
            if not retry_match:
                s = retry_response.find('[')
                e = retry_response.rfind(']')
                if s != -1 and e > s: retry_json = retry_response[s:e+1]
                elif s != -1: retry_json = repair_truncated_json(retry_response[s:])
            try:
                retry_data = json.loads(retry_json)
            except:
                retry_data = json.loads(repair_truncated_json(retry_json))
            retry_tasks = _convert_json_to_wbs_tasks(retry_data, team)
            retry_l1 = len([t for t in retry_tasks if (t.level.value if hasattr(t.level, "value") else str(t.level)) == "L1"])
            if retry_l1 > l1_count:
                print(f"  ✅ 재시도 성공: L1 {l1_count}개 → {retry_l1}개, 총 {len(retry_tasks)}개 태스크")
                initial_tasks = retry_tasks
                l1_count = retry_l1
            else:
                print(f"  ⚠️ 재시도에서도 L1 {retry_l1}개. 원본 유지.")
        except Exception as e:
            print(f"  ❌ 재시도 실패: {e}")

    retry_l3_count = 0
    if not initial_tasks:
        print(f"Failed to parse ANY tasks from response. Check {debug_path}")
        log_msg = "⚠️ **[Phase 1] WBS 초안 생성 실패**\nLLM 응답에서 유효한 데이터를 추출하지 못했습니다. 다시 시도해 주세요."
        msg_type = "comment"
        synth_l3_count = 0
    else:
        # 1차 보강: 같은 백본에 초점화 재프롬프트로 L3 생성 요청 (자연어 L3 확보)
        initial_tasks, retry_l3_count = _retry_missing_l3(initial_tasks, llm, team)
        if retry_l3_count:
            print(f"🔁 [L3 Retry] 누락 L2에 대해 재프롬프트로 L3 {retry_l3_count}개 확보 (백본: {os.getenv('LLM_BACKEND', 'unknown')})")
        # 2차 보강 (최후 수단): 재시도로도 채워지지 않은 L2에 템플릿 스텁 삽입
        initial_tasks, synth_l3_count = _ensure_l3_coverage(initial_tasks)
        if synth_l3_count:
            print(f"🛡️ [Safety Net] 재시도 후에도 L3 부재 L2 {synth_l3_count}개 감지 — 템플릿 스텁 자동 합성")
        # L2 count validation
        l1_ids = [t.task_id for t in initial_tasks if (t.level.value if hasattr(t.level, "value") else str(t.level)) == "L1"]
        l2_per_l1 = {}
        for t in initial_tasks:
            if (t.level.value if hasattr(t.level, "value") else str(t.level)) == "L2" and t.parent_id:
                l2_per_l1[t.parent_id] = l2_per_l1.get(t.parent_id, 0) + 1
        single_l2_l1s = [lid for lid in l1_ids if l2_per_l1.get(lid, 0) < 2]
        print(f"   L2 per L1 distribution: {l2_per_l1}")
        if single_l2_l1s:
            print(f"⚠️ L2 count warning: L1s with only 1 L2 child: {single_l2_l1s}")
        # 안전망 발동 카운트는 서버 로그에만 기록 (UI/debate_log에는 노출 X — 백본 간 결과물 형식 통일)
        if is_revision:
            log_msg = f"🔄 **[WBS 재설계 {current_revision}차] 에이전트 피드백 반영 완료**\n총 {len(initial_tasks)}개의 태스크로 재구성했습니다."
        else:
            log_msg = f"✅ **[Phase 1] 초기 3단계 WBS 초안 도출 완료**\n총 {len(initial_tasks)}개의 태스크가 식별되었습니다."
        msg_type = "proposal"

    msg = DebateMessage(
        timestamp=datetime.now().strftime("%H:%M:%S"),
        agent_role=AgentRole.SUPERVISOR,
        agent_name="WBS Gen Agent",
        message=log_msg,
        message_type=msg_type
    )
    return {
        "current_wbs_draft": initial_tasks,
        "debate_log": [msg],
        "generation_summary": f"WBS {'재설계 ' + str(current_revision) + '차' if is_revision else '초안'} 생성 완료 ({len(initial_tasks)}개 태스크)",
        # 재설계 완료 후 상태 초기화 (다음 토론 라운드를 위해 리셋)
        "wbs_revision_needed": False,
        "wbs_revision_hints": [],
        "current_wbs_revision": current_revision,
        "current_round": 0,
        "consensus_reached": False,
        "wbs_repair_stats": {
            "retry_l3_count": retry_l3_count,
            "synthetic_l3_count": synth_l3_count,
        },
    }

def _convert_json_to_wbs_tasks(data: list, team: list, role_map: dict = None) -> list:
    tasks = []
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
        # 이 단계에서는 아직 구체적인 배정이 아닌 직군(role)만 명시됨
        role = item.get("assigned_role") or fallback_role
        
        # 실제 멤버 지정은 완전히 비워두며, Phase 2에서 팀원 프로필을 바탕으로 Task Manager가 최적 배분
        assignees = []
        
        raw_importance = str(item.get("importance", "Medium")).strip().capitalize()
        if raw_importance not in ("High", "Medium", "Low"):
            raw_importance = "Medium"

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
            deliverables=item.get("deliverables", [f"{item.get('title')} 완료"]),
            risk_factors=item.get("risk_factors", []),
            importance=raw_importance,
        ))
    
    # 자동 스케줄링 적용
    tasks = _calculate_automatic_schedule(tasks)
    return tasks


def _extract_tasks_via_regex(text: str, team: list) -> list:
    """
    JSON 파싱 실패 시 정규표현식으로 { "id": "...", "title": "..." } 패턴을 강제 추출.
    따옴표 누락이나 사소한 문법 오류에 매우 강인함.
    """
    import re
    tasks = []
    # 개별 객체 { ... } 추출 시도
    # 단순하게 "id", "title" 등이 포함된 블록을 찾음
    pattern = r"\{[^{}]*?\"id\"[^{}]*?\}"
    matches = re.findall(pattern, text, re.DOTALL)
    
    for m in matches:
        try:
            # 개별 필드 추출
            tid = re.search(r"\"id\"\s*:\s*\"(.*?)\"", m)
            title = re.search(r"\"title\"\s*:\s*\"(.*?)\"", m)
            level = re.search(r"\"level\"\s*:\s*\"(.*?)\"", m)
            parent = re.search(r"\"parent_id\"\s*:\s*\"(.*?)\"", m)
            est = re.search(r"\"estimated_days\"\s*:\s*(\d+)", m)
            role = re.search(r"\"assigned_role\"\s*:\s*\"(.*?)\"", m)
            desc = re.search(r"\"description\"\s*:\s*\"(.*?)\"", m)

            if tid and title:
                # dependencies 추출 (문자열 리스트 형태)
                deps_match = re.search(r"\"dependencies\"\s*:\s*\[(.*?)\]", m)
                deps = [d.strip().strip('"') for d in deps_match.group(1).split(',')] if deps_match and deps_match.group(1).strip() else []

                # start_week, end_week 추출
                sw = re.search(r"\"start_week\"\s*:\s*(\d+)", m)
                ew = re.search(r"\"end_week\"\s*:\s*(\d+)", m)

                # importance 추출
                imp_match = re.search(r"\"importance\"\s*:\s*\"(High|Medium|Low)\"", m, re.IGNORECASE)
                raw_imp = imp_match.group(1).capitalize() if imp_match else "Medium"

                # deliverables / risk_factors 추출 시도
                deliv_match = re.search(r"\"deliverables\"\s*:\s*\[(.*?)\]", m)
                deliv = [d.strip().strip('"') for d in deliv_match.group(1).split(',')] if deliv_match and deliv_match.group(1).strip() else [f"{title.group(1)} 완료"]
                
                risk_match = re.search(r"\"risk_factors\"\s*:\s*\[(.*?)\]", m)
                risks = [r.strip().strip('"') for r in risk_match.group(1).split(',')] if risk_match and risk_match.group(1).strip() else []

                tasks.append(WBSTask(
                    task_id=tid.group(1),
                    level=WBSLevel(level.group(1) if level else "L1"),
                    parent_id=parent.group(1) if (parent and parent.group(1) != "null") else None,
                    title=title.group(1),
                    description=desc.group(1) if desc else "",
                    assigned_to=[],
                    assigned_role=role.group(1) if role else "Backend Developer",
                    estimated_days=float(est.group(1)) if est else 1.0,
                    buffer_days=0.0,
                    total_days=float(est.group(1)) if est else 1.0,
                    dependencies=deps,
                    start_week=int(sw.group(1)) if sw else None,
                    end_week=int(ew.group(1)) if ew else None,
                    deliverables=deliv,
                    risk_factors=risks,
                    importance=raw_imp,
                ))
        except Exception:
            continue

    # 자동 스케줄링 적용
    tasks = _calculate_automatic_schedule(tasks)
    return tasks
