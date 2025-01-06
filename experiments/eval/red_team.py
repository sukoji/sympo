"""
Red Team Evaluator — 가상 스트레스 테스트
──────────────────────────────────────────────────────────────────────────
까다로운 시니어 개발자 페르소나 LLM이 WBS의 논리적 허점과 누락 작업을 찾아냄.
Baseline과 symPO WBS 각각에 대해 Critical Flaw 수를 비교.

Critical Flaw 유형:
  - MISSING_TASK     : 명백히 필요한 태스크가 누락됨
  - WRONG_DEPENDENCY : 의존성 순서 오류 (논리적)
  - UNREALISTIC_TIME : 비현실적인 일정 (너무 짧거나 너무 긺)
  - RISK_NOT_COVERED : PRD에서 언급된 리스크가 반영 안 됨
  - ROLE_MISMATCH    : 업무 성격과 맞지 않는 직군 배정

RedTeamResult:
  critical_flaws     — 발견된 결함 목록
  flaw_count         — 결함 수 (적을수록 좋음)
  severity_breakdown — 유형별 집계
"""

from __future__ import annotations
import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import List, Dict

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


@dataclass
class Flaw:
    flaw_id: str
    flaw_type: str          # MISSING_TASK | WRONG_DEPENDENCY | UNREALISTIC_TIME | RISK_NOT_COVERED | ROLE_MISMATCH
    related_task_id: str    # 관련 태스크 ID (없으면 "N/A")
    description: str        # 구체적 결함 설명
    severity: str = "High"  # "High" | "Medium"


@dataclass
class RedTeamResult:
    critical_flaws: List[Flaw] = field(default_factory=list)
    flaw_count: int = 0
    high_severity_count: int = 0
    severity_breakdown: Dict[str, int] = field(default_factory=dict)
    raw_response: str = ""

    def to_dict(self) -> dict:
        return {
            "flaw_count": self.flaw_count,
            "high_severity_count": self.high_severity_count,
            "severity_breakdown": self.severity_breakdown,
            "flaws": [
                {
                    "flaw_id": f.flaw_id,
                    "type": f.flaw_type,
                    "task": f.related_task_id,
                    "severity": f.severity,
                    "description": f.description,
                }
                for f in self.critical_flaws
            ],
        }


def run_red_team(prd, tasks: list, label: str = "") -> RedTeamResult:
    """
    까다로운 시니어 개발자 페르소나로 WBS의 결함을 찾아내게 한다.

    Args:
        prd: PRDInput
        tasks: WBSTask 목록
        label: 로그용 레이블 ("Baseline" | "symPO")
    """
    try:
        from agents.llm_config import get_llm, normalize_content
    except ImportError:
        return RedTeamResult(raw_response="LLM 임포트 실패")

    project_name = getattr(prd, "project_name", "")
    project_goal = getattr(prd, "project_goal", "")
    key_features = "\n".join(f"  - {f}" for f in getattr(prd, "key_features", []))
    constraints = "\n".join(f"  - {c}" for c in (getattr(prd, "special_constraints", []) or []))

    # WBS 요약 (계층 구조 유지)
    wbs_lines = []
    for t in tasks:
        level = getattr(t, "level", "")
        level_str = level.value if hasattr(level, "value") else str(level)
        indent = {"L1": "", "L2": "  ", "L3": "    "}.get(level_str, "")
        days = getattr(t, "estimated_days", 0)
        buf = getattr(t, "buffer_days", 0)
        role = getattr(t, "assigned_role", "")
        deps = getattr(t, "dependencies", [])
        wbs_lines.append(
            f"{indent}[{getattr(t, 'task_id', '')}] {getattr(t, 'title', '')} "
            f"({days}d+{buf}d, {role})"
            + (f" → deps:{deps}" if deps else "")
        )

    wbs_text = "\n".join(wbs_lines)

    prompt = f"""당신은 15년 경력의 까다로운 시니어 아키텍트/PM으로, WBS 검토에서 어떤 허점도 그냥 넘기지 않습니다.
아래 프로젝트 정보와 WBS를 보고, "이 일정표대로 프로젝트를 진행했을 때 발생할 수 있는
치명적인 논리 결함, 누락된 작업, 비현실적 일정을 모두 찾아내십시오."

[프로젝트 정보]
- 프로젝트명: {project_name}
- 목표: {project_goal}
- 핵심 기능:
{key_features}
- 특수 제약사항:
{constraints if constraints else "  없음"}

[WBS]
{wbs_text}

다음 유형의 결함만 보고하십시오 (사소한 스타일 이슈 제외):
- MISSING_TASK: 명백히 필요한 태스크가 WBS에 없음
- WRONG_DEPENDENCY: 의존성 순서가 논리적으로 잘못됨
- UNREALISTIC_TIME: 태스크 기간이 비현실적 (너무 짧거나 너무 긺)
- RISK_NOT_COVERED: PRD/제약사항에서 언급된 리스크가 반영 안 됨
- ROLE_MISMATCH: 업무 성격과 맞지 않는 직군 배정

JSON 형식으로만 답하십시오:
{{
  "flaws": [
    {{
      "flaw_id": "F001",
      "flaw_type": "MISSING_TASK",
      "related_task_id": "없으면 N/A",
      "severity": "High 또는 Medium",
      "description": "구체적인 결함 설명 (한국어, 1~2문장)"
    }}
  ],
  "summary": "전반적인 WBS 품질 총평 (2~3문장)"
}}"""

    try:
        llm = get_llm(temperature=0.2, max_tokens=2048)
        response = normalize_content(llm.invoke([{"role": "user", "content": prompt}]).content)

        # JSON 파싱
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        data = json.loads(json_match.group(0)) if json_match else {}

        flaws_raw = data.get("flaws", [])
        flaws = []
        for f in flaws_raw:
            try:
                flaws.append(Flaw(
                    flaw_id=f.get("flaw_id", "F???"),
                    flaw_type=f.get("flaw_type", "MISSING_TASK"),
                    related_task_id=f.get("related_task_id", "N/A"),
                    description=f.get("description", ""),
                    severity=f.get("severity", "High"),
                ))
            except Exception:
                continue

        # 집계
        breakdown: Dict[str, int] = {}
        for flaw in flaws:
            breakdown[flaw.flaw_type] = breakdown.get(flaw.flaw_type, 0) + 1

        high_count = sum(1 for f in flaws if f.severity == "High")

        print(f"[RedTeam:{label}] 결함 {len(flaws)}건 발견 (High: {high_count}건)")

        return RedTeamResult(
            critical_flaws=flaws,
            flaw_count=len(flaws),
            high_severity_count=high_count,
            severity_breakdown=breakdown,
            raw_response=response[:300],
        )

    except Exception as e:
        return RedTeamResult(raw_response=f"평가 실패: {e}")
