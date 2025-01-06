"""
PRD (Product Requirement Document) 스키마 정의
"""
from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import date


class PRDInput(BaseModel):
    """프로젝트 요구사항 문서 스키마"""

    project_name: str = Field(..., description="프로젝트명")
    project_goal: str = Field(..., description="프로젝트 최종 목표")
    target_users: str = Field(..., description="타깃 사용자 및 고객")
    scope: str = Field(..., description="프로젝트 범위 (포함/제외 기능)")
    key_features: List[str] = Field(
        ..., description="핵심 기능 목록 (예: ['로그인', '대시보드', ...])"
    )
    tech_stack_requirements: List[str] = Field(
        default_factory=list,
        description="기술 스택 요구사항 (예: ['React', 'FastAPI', 'PostgreSQL'])",
    )
    deadline: Optional[str] = Field(
        None, description="최종 마감일 (예: 2025-06-30)"
    )
    team_size: int = Field(default=5, description="팀원 수")
    budget_weeks: Optional[int] = Field(
        None, description="예산 기준 총 개발 주 수"
    )
    special_constraints: List[str] = Field(
        default_factory=list,
        description="특수 제약사항 (예: ['레거시 시스템 연동', '보안 인증 필요'])",
    )
    raw_text: Optional[str] = Field(
        None, description="원본 PRD 텍스트 (파싱 전)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "project_name": "AI 기반 고객 서비스 플랫폼",
                "project_goal": "AI 챗봇과 실시간 분석 대시보드를 통해 고객 만족도 30% 향상",
                "target_users": "B2B 중소기업 고객사 상담팀",
                "scope": "챗봇 UI, 관리자 대시보드 포함 / 모바일 앱 제외",
                "key_features": ["AI 챗봇 인터페이스", "실시간 분석 대시보드", "CRM 연동", "리포트 자동화"],
                "tech_stack_requirements": ["React", "Python FastAPI", "PostgreSQL", "OpenAI API"],
                "deadline": "2025-09-30",
                "team_size": 5,
                "budget_weeks": 16,
                "special_constraints": ["기존 CRM 시스템과 REST API 연동 필수"],
            }
        }
