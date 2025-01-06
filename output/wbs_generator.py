"""
계층적 WBS 생성기
에이전트 합의 상태에서 L1→L2→L3 계층 구조의 WBS를 추출합니다.
"""
import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from schemas.wbs_schema import WBSTask, WBSLevel, WBSOutput, DebateMessage
from schemas.prd_schema import PRDInput
from schemas.member_schema import MemberProfile


class WBSGenerator:
    """
    최종 WBS 생성 및 구조화
    - 계층적 WBS (L1/L2/L3) 구축
    - 일정 계산 (버퍼 포함)
    - 담당 R&R 명확화
    """

    def __init__(self, output_dir: str = "./generated"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate(
        self,
        tasks: List[WBSTask],
        prd: PRDInput,
        team: List[MemberProfile],
        debate_log: List[DebateMessage],
        summary: str = "",
    ) -> WBSOutput:
        """최종 WBS 출력 패키지 생성"""
        # 일정 계산 (주차 배정)
        tasks_with_schedule = self._assign_schedule(tasks)

        # 총 기간 계산 (가장 늦게 끝나는 엔드 위크 기준)
        if tasks_with_schedule:
            max_end_week = max(t.end_week for t in tasks_with_schedule if t.level == WBSLevel.L1)
            total_weeks = float(max_end_week)
        else:
            total_weeks = 0.0

        if total_weeks < 1.0:
             # 엔드 위크가 설정되지 않았거나 너무 짧은 경우 재계산
             l1_tasks = [t for t in tasks_with_schedule if t.level == WBSLevel.L1]
             total_days = sum(t.total_days for t in l1_tasks)
             total_weeks = round(total_days / 5, 1)

        # 사용된 페르소나 목록 (role이 None인 경우 대비)
        personas_used = list(set([
            m.role.value for m in team if m.role is not None
        ]))

        # Pydantic v2 double-import 방지: 인스턴스를 dict로 변환 후 재검증
        return WBSOutput(
            project_name=prd.project_name,
            total_weeks=total_weeks,
            tasks=[t.model_dump() for t in tasks_with_schedule],
            debate_log=[m.model_dump() for m in debate_log],
            summary=summary or f"총 {total_weeks}주 일정으로 {len(tasks)}개 태스크 생성",
            agent_personas_used=personas_used,
        )

    def _assign_schedule(self, tasks: List[WBSTask]) -> List[WBSTask]:
        """의존성 기반 주차 배정"""
        # 의존성 그래프 기반 위상 정렬
        scheduled = {}
        updated_tasks = []

        l1_tasks = [t for t in tasks if t.level == WBSLevel.L1]
        l2_tasks = [t for t in tasks if t.level == WBSLevel.L2]

        # L1 순차 배정
        current_week = 1
        for task in l1_tasks:
            duration_weeks = round(task.total_days / 5, 1)
            task_updated = task.model_copy(update={
                "start_week": int(current_week),
                "end_week": int(current_week + duration_weeks),
            })
            scheduled[task.task_id] = int(current_week + duration_weeks)
            updated_tasks.append(task_updated)
            current_week += duration_weeks

        # L2 배정 (부모 기준 내 순차 배정)
        l2_offsets = {}
        for task in l2_tasks:
            pid = task.parent_id
            if pid not in l2_offsets:
                parent_task = next((t for t in l1_tasks if t.task_id == pid), None)
                l2_offsets[pid] = parent_task.start_week if parent_task and parent_task.start_week else 1
            
            p_start = l2_offsets[pid]
            duration_weeks = round(task.total_days / 5, 1)
            task_updated = task.model_copy(update={
                "start_week": int(p_start),
                "end_week": int(p_start + duration_weeks),
            })
            updated_tasks.append(task_updated)
            # 다음 L2 태스크는 현재 작업이 끝난 후 시작하도록 오프셋 누적
            l2_offsets[pid] = p_start + duration_weeks
            
        # L3 배정 (부모 L2 기준 내 순차 배정)
        l3_tasks = [t for t in tasks if t.level == WBSLevel.L3]
        l3_offsets = {}
        for task in l3_tasks:
            pid = task.parent_id
            if pid not in l3_offsets:
                parent_task = next((t for t in updated_tasks if t.task_id == pid), None)
                l3_offsets[pid] = parent_task.start_week if parent_task and parent_task.start_week else 1
            
            p_start = l3_offsets[pid]
            duration_weeks = round(task.total_days / 5, 1)
            task_updated = task.model_copy(update={
                "start_week": int(p_start),
                "end_week": int(p_start + duration_weeks),
            })
            updated_tasks.append(task_updated)
            # 다음 L3 태스크는 현재 작업이 끝난 후 시작하도록 오프셋 누적
            l3_offsets[pid] = p_start + duration_weeks

        return updated_tasks

    def get_hierarchical_dict(self, tasks: List[WBSTask]) -> Dict:
        """계층적 딕셔너리 구조로 변환"""
        l1 = [t for t in tasks if t.level == WBSLevel.L1]
        l2 = [t for t in tasks if t.level == WBSLevel.L2]
        l3 = [t for t in tasks if t.level == WBSLevel.L3]

        hierarchy = []
        for l1_task in l1:
            children_l2 = [t for t in l2 if t.parent_id == l1_task.task_id]
            l2_items = []
            for l2_task in children_l2:
                children_l3 = [t for t in l3 if t.parent_id == l2_task.task_id]
                l2_items.append({
                    **l2_task.model_dump(),
                    "children": [t.model_dump() for t in children_l3],
                })
            hierarchy.append({
                **l1_task.model_dump(),
                "children": l2_items,
            })
        return {"wbs": hierarchy}

    def get_member_name_map(self, team: List[MemberProfile]) -> Dict[str, str]:
        """member_id → 이름 매핑"""
        return {m.member_id: m.name for m in team}

    def compute_critical_path(self, tasks: List[WBSTask]) -> List[str]:
        """크리티컬 패스 계산 (가장 긴 경로)"""
        l1 = sorted(
            [t for t in tasks if t.level == WBSLevel.L1],
            key=lambda x: x.total_days,
            reverse=True,
        )
        return [t.task_id for t in l1]
