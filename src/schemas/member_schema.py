"""
팀원 메타데이터 스키마 정의
"""
from enum import Enum
from typing import List, Optional, Dict
from pydantic import BaseModel, Field


class MemberRole(str, Enum):
    """팀원 직군 분류"""
    # ── 소프트웨어 개발 직군 ──────────────────────────
    PM = "PM"
    PLANNER = "Planner"
    FRONTEND = "Frontend Developer"
    BACKEND = "Backend Developer"
    FULLSTACK = "Fullstack Developer"
    DESIGNER = "Designer"
    QA = "QA Engineer"
    DATA = "Data Engineer"
    DEVOPS = "DevOps"
    # ── 비즈니스 / 마케팅 직군 ────────────────────────
    DATA_ANALYST = "Data Analyst"          # 데이터 분석가
    MARKETING = "Marketing Planner"        # 마케팅 기획자
    BUSINESS = "Business Analyst"          # 경영/사업 분석가
    MOBILE = "Mobile Developer"            # 모바일 앱 개발자
    OPERATIONS = "Operations Manager"      # 운영 매니저 (매장/물류 등)


class PersonalityTrait(str, Enum):
    """성향 특성"""
    DETAIL_ORIENTED = "꼼꼼함"
    BIG_PICTURE = "큰그림형"
    COLLABORATIVE = "협력적"
    INDEPENDENT = "독립적"
    RISK_AVERSE = "리스크 회피형"
    RISK_TAKER = "도전적"
    FAST_EXECUTOR = "빠른 실행"
    THOROUGH_PLANNER = "철저한 계획"


class PastProject(BaseModel):
    """과거 프로젝트 경험"""
    name: str
    role: str
    duration_weeks: int
    technologies: List[str]
    notable_outcomes: str
    challenges: Optional[str] = None


class MemberProfile(BaseModel):
    """팀원 메타데이터 스키마"""
    member_id: str = Field(..., description="팀원 고유 ID (예: MBR-001)")
    name: str
    role: Optional[MemberRole] = Field(None, description="주 직군 (WBS 생성 시 LLM에 노출 안 함 — member_role_map으로 분리)")
    sub_roles: List[str] = Field(
        default_factory=list, description="보조 가능 역할"
    )
    years_of_experience: float = Field(..., description="총 경력 연수")
    tech_stack: List[str] = Field(..., description="보유 기술 스택")
    primary_skills: List[str] = Field(
        ..., description="핵심 강점 기술 (3~5개)"
    )
    strengths: List[str] = Field(..., description="업무 강점")
    weaknesses: List[str] = Field(..., description="업무 약점 / 성장 포인트")
    personality_traits: List[PersonalityTrait] = Field(
        default_factory=list, description="성향 특성"
    )
    past_projects: List[PastProject] = Field(
        default_factory=list, description="주요 과거 프로젝트"
    )
    preferred_task_types: List[str] = Field(
        default_factory=list,
        description="선호 업무 유형 (예: ['설계', '구현', '리뷰'])",
    )
    known_bottlenecks: List[str] = Field(
        default_factory=list,
        description="알려진 병목/리스크 요인 (예: ['새 기술 러닝 커브'])",
    )
    availability_percent: int = Field(
        default=100, description="프로젝트 투입 가능 비율 (%)"
    )
    raw_resume_text: Optional[str] = Field(
        None, description="원본 이력서/CV 텍스트"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "member_id": "MBR-001",
                "name": "김철수",
                "role": "Backend Developer",
                "years_of_experience": 4.5,
                "tech_stack": ["Python", "FastAPI", "PostgreSQL", "Redis", "Docker"],
                "primary_skills": ["API 설계", "DB 최적화", "Docker 컨테이너화"],
                "strengths": ["체계적인 문서화", "빠른 API 개발", "문제 분석력"],
                "weaknesses": ["새 프레임워크 러닝 커브", "프론트엔드 협업 소통"],
                "personality_traits": ["꼼꼼함", "독립적"],
            }
        }
