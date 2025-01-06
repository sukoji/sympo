"""
Structural Integrity Checker — GT 없이 100% 자동 검증 가능한 구조적 무결성 지표
──────────────────────────────────────────────────────────────────────────────
지표:
  1. Dependency Error Rate   — 후행 태스크가 선행 태스크보다 일찍 시작하는 비율
  2. 8-80 Rule Violation     — L3 태스크 estimated_days가 1일 미만 또는 10일 초과 비율
  3. MECE Rollup Error       — L3 합계 vs L2 estimated_days 오차율 ≥ 50% 비율
  4. L2 Coverage             — L1당 L2 개수 < 2 인 L1 비율
  5. Orphan Task Rate        — parent_id가 있지만 부모 태스크가 존재하지 않는 비율
  6. Empty Assignment Rate   — assigned_to가 비어있는 L3 태스크 비율
  7. Importance Distribution — High/Medium/Low 비율 (이상적: 25-35 / 45-55 / 15-25)

반환: StructuralReport (dict 포함)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class StructuralReport:
    """구조적 무결성 검증 결과"""
    # ── 1. Dependency
    dependency_error_count: int = 0
    dependency_error_rate: float = 0.0
    dependency_errors: List[dict] = field(default_factory=list)

    # ── 2. 8-80 Rule
    rule_8_80_violations: int = 0
    rule_8_80_rate: float = 0.0
    rule_8_80_detail: List[dict] = field(default_factory=list)

    # ── 3. MECE Rollup
    mece_error_count: int = 0
    mece_error_rate: float = 0.0
    mece_errors: List[dict] = field(default_factory=list)

    # ── 4. L2 Coverage
    l1_count: int = 0
    l1_single_l2_count: int = 0
    l2_coverage_error_rate: float = 0.0
    l2_distribution: Dict[str, int] = field(default_factory=dict)

    # ── 5. Orphan Tasks
    orphan_count: int = 0
    orphan_rate: float = 0.0

    # ── 6. Empty Assignment
    empty_assign_count: int = 0
    empty_assign_rate: float = 0.0

    # ── 7. Importance Distribution
    importance_dist: Dict[str, float] = field(default_factory=dict)
    importance_balanced: bool = False

    # ── 종합 점수
    overall_score: float = 0.0     # 0 ~ 100 (높을수록 좋음)
    grade: str = "F"               # A/B/C/D/F

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dependency_error_rate": self.dependency_error_rate,
            "dependency_errors": self.dependency_errors[:5],
            "rule_8_80_rate": self.rule_8_80_rate,
            "rule_8_80_violations": self.rule_8_80_violations,
            "mece_error_rate": self.mece_error_rate,
            "mece_errors": self.mece_errors[:5],
            "l2_coverage_error_rate": self.l2_coverage_error_rate,
            "l2_distribution": self.l2_distribution,
            "orphan_rate": self.orphan_rate,
            "empty_assign_rate": self.empty_assign_rate,
            "importance_dist": self.importance_dist,
            "importance_balanced": self.importance_balanced,
            "overall_score": self.overall_score,
            "grade": self.grade,
        }


def check_structural_integrity(tasks: list) -> StructuralReport:
    """
    WBS 태스크 목록을 받아 구조적 무결성 검사를 수행하고 StructuralReport 반환.
    """
    report = StructuralReport()
    if not tasks:
        return report

    id_map: Dict[str, Any] = {t.task_id: t for t in tasks}
    l3_tasks = [t for t in tasks if _level(t) == "L3"]
    l2_tasks = [t for t in tasks if _level(t) == "L2"]
    l1_tasks = [t for t in tasks if _level(t) == "L1"]

    # ── 1. Dependency Error Rate ──────────────────────────────────────────
    dep_errors = []
    for t in tasks:
        t_start = getattr(t, "start_week", None) or 0
        for dep_id in (t.dependencies or []):
            dep = id_map.get(dep_id)
            if dep is None:
                continue
            dep_end = getattr(dep, "end_week", None) or 0
            if t_start > 0 and dep_end > 0 and t_start < dep_end:
                dep_errors.append({
                    "task_id": t.task_id,
                    "depends_on": dep_id,
                    "task_start_week": t_start,
                    "dep_end_week": dep_end,
                })
    report.dependency_error_count = len(dep_errors)
    report.dependency_errors = dep_errors
    report.dependency_error_rate = round(len(dep_errors) / max(len(tasks), 1), 4)

    # ── 2. 8-80 Rule (1일 미만 또는 10일 초과 = 80시간 초과) ──────────────
    violations = []
    for t in l3_tasks:
        days = getattr(t, "estimated_days", 0) or 0
        if days < 1.0 or days > 10.0:
            violations.append({
                "task_id": t.task_id,
                "estimated_days": days,
                "violation": "too_short" if days < 1.0 else "too_long",
            })
    report.rule_8_80_violations = len(violations)
    report.rule_8_80_detail = violations
    report.rule_8_80_rate = round(len(violations) / max(len(l3_tasks), 1), 4)

    # ── 3. MECE Rollup Error ──────────────────────────────────────────────
    mece_errors = []
    for l2 in l2_tasks:
        children = [t for t in l3_tasks if t.parent_id == l2.task_id]
        if not children:
            continue
        l3_sum = sum(getattr(t, "estimated_days", 0) or 0 for t in children)
        l2_days = getattr(l2, "estimated_days", 0) or 0
        if l2_days > 0:
            ratio = abs(l3_sum - l2_days) / l2_days
            if ratio >= 0.5:  # 50% 이상 오차
                mece_errors.append({
                    "l2_id": l2.task_id,
                    "l2_days": l2_days,
                    "l3_sum": round(l3_sum, 1),
                    "error_ratio": round(ratio, 2),
                })
    report.mece_error_count = len(mece_errors)
    report.mece_errors = mece_errors
    report.mece_error_rate = round(len(mece_errors) / max(len(l2_tasks), 1), 4)

    # ── 4. L2 Coverage ────────────────────────────────────────────────────
    l2_per_l1: Dict[str, int] = {}
    for l2 in l2_tasks:
        if l2.parent_id:
            l2_per_l1[l2.parent_id] = l2_per_l1.get(l2.parent_id, 0) + 1
    single_l2 = [lid for lid in [t.task_id for t in l1_tasks] if l2_per_l1.get(lid, 0) < 2]
    report.l1_count = len(l1_tasks)
    report.l1_single_l2_count = len(single_l2)
    report.l2_distribution = {lid: l2_per_l1.get(lid, 0) for lid in [t.task_id for t in l1_tasks]}
    report.l2_coverage_error_rate = round(len(single_l2) / max(len(l1_tasks), 1), 4)

    # ── 5. Orphan Task Rate ───────────────────────────────────────────────
    orphans = [
        t for t in tasks
        if t.parent_id and t.parent_id not in id_map
    ]
    report.orphan_count = len(orphans)
    report.orphan_rate = round(len(orphans) / max(len(tasks), 1), 4)

    # ── 6. Empty Assignment Rate (L3 기준) ────────────────────────────────
    empty_l3 = [
        t for t in l3_tasks
        if not getattr(t, "assigned_to", None)
    ]
    report.empty_assign_count = len(empty_l3)
    report.empty_assign_rate = round(len(empty_l3) / max(len(l3_tasks), 1), 4)

    # ── 7. Importance Distribution ────────────────────────────────────────
    imp_counts: Dict[str, int] = {"High": 0, "Medium": 0, "Low": 0}
    for t in tasks:
        imp = str(getattr(t, "importance", "Medium") or "Medium").capitalize()
        imp = imp if imp in imp_counts else "Medium"
        imp_counts[imp] += 1
    total = max(len(tasks), 1)
    report.importance_dist = {k: round(v / total, 3) for k, v in imp_counts.items()}
    high_pct = imp_counts["High"] / total
    low_pct = imp_counts["Low"] / total
    report.importance_balanced = (0.15 <= high_pct <= 0.45) and (low_pct >= 0.05)

    # ── 종합 점수 ─────────────────────────────────────────────────────────
    # 각 항목 0~100점, 가중 평균
    scores = {
        "dependency":  max(0.0, 1.0 - report.dependency_error_rate * 5) * 100,
        "rule_8_80":   max(0.0, 1.0 - report.rule_8_80_rate) * 100,
        "mece":        max(0.0, 1.0 - report.mece_error_rate) * 100,
        "l2_coverage": max(0.0, 1.0 - report.l2_coverage_error_rate) * 100,
        "orphan":      max(0.0, 1.0 - report.orphan_rate * 10) * 100,
        "empty_assign": max(0.0, 1.0 - report.empty_assign_rate) * 100,
        "importance":  100.0 if report.importance_balanced else 50.0,
    }
    weights = {
        "dependency": 0.25,
        "rule_8_80": 0.15,
        "mece": 0.15,
        "l2_coverage": 0.20,
        "orphan": 0.10,
        "empty_assign": 0.10,
        "importance": 0.05,
    }
    report.overall_score = round(sum(scores[k] * weights[k] for k in scores), 1)
    if report.overall_score >= 90:
        report.grade = "A"
    elif report.overall_score >= 75:
        report.grade = "B"
    elif report.overall_score >= 60:
        report.grade = "C"
    elif report.overall_score >= 45:
        report.grade = "D"
    else:
        report.grade = "F"

    return report


def _level(task) -> str:
    """WBSLevel enum 또는 문자열에서 "L1"/"L2"/"L3" 반환"""
    lv = getattr(task, "level", None)
    if lv is None:
        return ""
    return lv.value if hasattr(lv, "value") else str(lv)


def format_report(report: StructuralReport) -> str:
    """사람이 읽기 쉬운 형태로 보고서 출력"""
    lines = [
        "=" * 60,
        "  WBS 구조적 무결성 검사 보고서",
        "=" * 60,
        f"  종합 점수: {report.overall_score:.1f} / 100  (등급: {report.grade})",
        "-" * 60,
        f"  [1] 의존성 오류율:        {report.dependency_error_rate:.1%}  "
        f"({report.dependency_error_count}건)",
        f"  [2] 8-80 Rule 위반율:    {report.rule_8_80_rate:.1%}  "
        f"({report.rule_8_80_violations}건)",
        f"  [3] MECE 롤업 오류율:    {report.mece_error_rate:.1%}  "
        f"({report.mece_error_count}건)",
        f"  [4] L2 커버리지 부족 L1: {report.l2_coverage_error_rate:.1%}  "
        f"({report.l1_single_l2_count}/{report.l1_count}개)",
        f"  [5] 고아 태스크 비율:    {report.orphan_rate:.1%}  "
        f"({report.orphan_count}건)",
        f"  [6] 배정 누락율 (L3):   {report.empty_assign_rate:.1%}  "
        f"({report.empty_assign_count}건)",
        f"  [7] 중요도 분포:         "
        f"High={report.importance_dist.get('High', 0):.1%}  "
        f"Med={report.importance_dist.get('Medium', 0):.1%}  "
        f"Low={report.importance_dist.get('Low', 0):.1%}  "
        f"({'균형' if report.importance_balanced else '불균형'})",
    ]

    if report.dependency_errors:
        lines += ["", "  ⚠️  의존성 오류 상위 3건:"]
        for e in report.dependency_errors[:3]:
            lines.append(f"    - {e['task_id']} 시작(W{e['task_start_week']}) < "
                         f"{e['depends_on']} 종료(W{e['dep_end_week']})")

    if report.mece_errors:
        lines += ["", "  ⚠️  MECE 오류 상위 3건:"]
        for e in report.mece_errors[:3]:
            lines.append(f"    - {e['l2_id']}: L2={e['l2_days']}d, L3합계={e['l3_sum']}d "
                         f"(오차 {e['error_ratio']:.0%})")

    lines.append("=" * 60)
    return "\n".join(lines)
