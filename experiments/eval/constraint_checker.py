"""
Constraint Satisfaction Checker — Planted Constraint 충족률 측정
──────────────────────────────────────────────────────────────────
1단계: 키워드 매칭 (rule-based)
2단계: 애매한 케이스는 LLM-as-Judge로 재판정

ConstraintResult:
  satisfied     — 충족된 제약 목록
  violated      — 미충족된 제약 목록
  rate          — 충족률 (0.0 ~ 1.0)
  high_severity_rate — High 중요도 제약만의 충족률
"""

from __future__ import annotations
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from eval.benchmark_generator import PlantedConstraint


@dataclass
class ConstraintResult:
    satisfied: List[PlantedConstraint] = field(default_factory=list)
    violated: List[PlantedConstraint] = field(default_factory=list)
    rate: float = 0.0
    high_severity_rate: float = 0.0
    detail: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "constraint_satisfaction_rate": self.rate,
            "high_severity_rate": self.high_severity_rate,
            "satisfied_count": len(self.satisfied),
            "violated_count": len(self.violated),
            "total_count": len(self.satisfied) + len(self.violated),
            "violated_ids": [c.constraint_id for c in self.violated],
            "detail": self.detail,
        }


def check_constraints(
    tasks: list,
    constraints: List[PlantedConstraint],
    use_llm_fallback: bool = True,
) -> ConstraintResult:
    """
    WBS 태스크 목록과 Planted Constraint 목록을 받아 충족 여부 검사.

    Args:
        tasks: WBSTask 목록
        constraints: PlantedConstraint 목록
        use_llm_fallback: 키워드 미충족 시 LLM 재판정 여부
    """
    result = ConstraintResult()
    if not tasks or not constraints:
        return result

    # WBS 전체 텍스트 블롭 생성 (title + description)
    task_blob = " ".join(
        f"{getattr(t, 'title', '')} {getattr(t, 'description', '')} {getattr(t, 'buffer_rationale', '')}"
        for t in tasks
    ).lower()

    for con in constraints:
        matched = _keyword_check(task_blob, con.check_keywords)

        # 수치 조건이 있는 경우 구조 체크
        if matched and con.expected_value is not None:
            matched = _structural_check(tasks, con)

        detail_entry = {
            "constraint_id": con.constraint_id,
            "type": con.constraint_type,
            "severity": con.severity,
            "satisfied": matched,
            "method": "keyword",
            "description": con.description[:80],
        }

        # LLM 폴백: 키워드 미충족 & High severity
        if not matched and use_llm_fallback and con.severity == "High":
            llm_result = _llm_judge_constraint(tasks, con)
            if llm_result is not None:
                matched = llm_result
                detail_entry["method"] = "llm_fallback"
                detail_entry["satisfied"] = matched

        if matched:
            result.satisfied.append(con)
        else:
            result.violated.append(con)

        result.detail.append(detail_entry)

    total = len(constraints)
    result.rate = round(len(result.satisfied) / max(total, 1), 4)

    high_cons = [c for c in constraints if c.severity == "High"]
    high_sat = sum(1 for c in result.satisfied if c.severity == "High")
    result.high_severity_rate = round(high_sat / max(len(high_cons), 1), 4)

    return result


def _keyword_check(task_blob: str, keywords: List[str]) -> bool:
    """키워드 중 하나라도 태스크 블롭에 존재하면 True"""
    return any(kw.lower() in task_blob for kw in keywords)


def _structural_check(tasks: list, con: PlantedConstraint) -> bool:
    """
    수치 조건이 있는 제약에 대해 구조적 검증.
    check_fn_name에 따라 다른 로직 적용.
    """
    fn = con.check_fn_name
    val = con.expected_value

    if fn == "schedule_ratio_check":
        # QA 기간 비율 ≥ expected_value
        qa_tasks = [t for t in tasks if any(
            kw.lower() in (getattr(t, "title", "") + getattr(t, "description", "")).lower()
            for kw in ["qa", "테스트", "검증", "품질"]
        )]
        total_days = sum(getattr(t, "estimated_days", 0) or 0 for t in tasks if _level_val(t) == "L3")
        qa_days = sum(getattr(t, "estimated_days", 0) or 0 for t in qa_tasks if _level_val(t) == "L3")
        return (qa_days / max(total_days, 1)) >= (val or 0.20)

    elif fn == "deadline_check":
        # 특정 키워드 태스크의 end_week ≤ expected_value
        matched_tasks = [t for t in tasks if any(
            kw.lower() in (getattr(t, "title", "") + getattr(t, "description", "")).lower()
            for kw in con.check_keywords
        )]
        if not matched_tasks:
            return False
        return all((getattr(t, "end_week", 99) or 99) <= val for t in matched_tasks)

    elif fn == "duration_check":
        # 특정 키워드 태스크의 estimated_days 합계 ≥ expected_value
        matched_tasks = [t for t in tasks if any(
            kw.lower() in (getattr(t, "title", "") + getattr(t, "description", "")).lower()
            for kw in con.check_keywords
        )]
        if not matched_tasks:
            return False
        total = sum(getattr(t, "estimated_days", 0) or 0 for t in matched_tasks)
        return total >= val

    return True  # keyword_match는 이미 처리됨


def _llm_judge_constraint(tasks: list, con: PlantedConstraint) -> Optional[bool]:
    """
    LLM-as-Judge: 제약이 WBS에 반영되었는지 LLM이 판정.
    실패 시 None 반환 (폴백 없이 keyword 결과 유지).
    """
    try:
        from agents.llm_config import get_llm, normalize_content

        # WBS 요약본 생성 (L3 태스크 제목 + 설명)
        wbs_summary = "\n".join(
            f"- [{getattr(t, 'task_id', '')}] {getattr(t, 'title', '')} : {getattr(t, 'description', '')[:80]}"
            for t in tasks
            if _level_val(t) == "L3"
        )[:3000]  # 토큰 절약

        prompt = f"""다음 WBS 태스크 목록을 검토하고, 아래 제약 조건이 충족되었는지 판단하십시오.

[제약 조건]
{con.description}

[WBS 태스크 요약 (L3)]
{wbs_summary}

위 제약 조건이 WBS에 명확하게 반영되어 있으면 "YES", 그렇지 않으면 "NO"만 답하십시오."""

        llm = get_llm(temperature=0.0, max_tokens=16)
        response = normalize_content(llm.invoke([{"role": "user", "content": prompt}]).content)
        return "YES" in response.upper()

    except Exception:
        return None


def _level_val(task) -> str:
    lv = getattr(task, "level", None)
    if lv is None:
        return ""
    return lv.value if hasattr(lv, "value") else str(lv)
