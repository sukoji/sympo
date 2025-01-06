"""
Baseline Runner — 단일 LLM Zero-Shot WBS 생성 (비교 대조군)
──────────────────────────────────────────────────────────
동일한 PRD와 팀 정보를 주고 LLM에게 한 번에 WBS를 작성하도록 요청.
symPO 다중 에이전트 결과와 비교하기 위한 Baseline.
"""

from __future__ import annotations
import json
import os
import re
import sys
from typing import List

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from agents.llm_config import get_llm, normalize_content
from agents.wbs_gen_agent import repair_truncated_json, _convert_json_to_wbs_tasks
from schemas.wbs_schema import WBSTask


def generate_baseline_wbs(prd, team: list) -> List[WBSTask]:
    """
    단일 LLM에게 Zero-Shot으로 WBS 생성을 요청.
    symPO의 다중 에이전트 토론 없이, 하나의 LLM 호출로 완성된 WBS를 얻는다.
    """
    llm = get_llm(temperature=0.3, max_tokens=6000)

    budget_weeks = getattr(prd, 'budget_weeks', None) or 12
    budget_days = budget_weeks * 5
    team_summary = "\n".join([
        f"- {m.name} ({m.years_of_experience}년): {', '.join(m.tech_stack[:3])}"
        for m in team
    ])
    key_features = ", ".join(getattr(prd, "key_features", []))
    constraints = "\n".join(getattr(prd, "special_constraints", []) or []) or "없음"

    prompt = f"""당신은 시니어 PM입니다. 아래 프로젝트 정보를 바탕으로 실무 수준의 3단계 WBS를 작성하십시오.

프로젝트명: {getattr(prd, 'project_name', '')}
목표: {getattr(prd, 'project_goal', '')}
핵심 기능: {key_features}
총 기간: {budget_weeks}주 ({budget_days}일)
팀 구성:
{team_summary}
특수 제약사항: {constraints}

[요구사항]
1. L1(Phase, 4~5개), L2(기능그룹, 각 L1당 2~4개), L3(세부 태스크, 각 L2당 2~4개)로 구성
2. 각 L1당 L2는 반드시 2개 이상
3. 의존성(dependencies), 중요도(importance: High/Medium/Low), start_week, end_week 모두 포함
4. 현실적인 일정 산정 (L3: 2~10일, L2: 4~20일)
5. 특수 제약사항을 반드시 태스크에 반영

JSON 형식으로만 답하십시오:
```json
[
  {{"id": "L1-01", "title": "...", "level": "L1", "parent_id": null,
    "estimated_days": 15, "assigned_role": "...", "dependencies": [],
    "importance": "High", "start_week": 1, "end_week": 3, "description": "..."}}
]
```"""

    try:
        response = normalize_content(llm.invoke([{"role": "user", "content": prompt}]).content)

        # JSON 추출
        json_match = re.search(r"```json\s*(.*?)```", response, re.DOTALL)
        json_str = json_match.group(1) if json_match else response

        if not json_match:
            start = response.find('[')
            end = response.rfind(']')
            if start != -1:
                json_str = response[start:end+1] if end > start else repair_truncated_json(response[start:])

        try:
            data = json.loads(json_str)
        except Exception:
            data = json.loads(repair_truncated_json(json_str))

        tasks = _convert_json_to_wbs_tasks(data, team)
        print(f"[Baseline] WBS 생성 완료: {len(tasks)}개 태스크")
        return tasks

    except Exception as e:
        print(f"[Baseline] 생성 실패: {e}")
        return []
