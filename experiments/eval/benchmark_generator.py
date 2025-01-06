"""
Benchmark Generator — "Planted Constraints" 방식의 합성 평가 데이터셋 생성
────────────────────────────────────────────────────────────────────────
GT WBS 없이 평가하기 위해 PRD + 팀원 프로필에 검증 가능한 제약(Constraint)을
심어둔다. 각 제약은 최종 WBS에서 자동으로 검증 가능한 형태로 정의된다.

BenchmarkCase:
  prd           — PRDInput (synthetic)
  team          — List[MemberProfile] (synthetic)
  constraints   — List[PlantedConstraint]  ← 이것이 GT 역할
  meta          — 도메인·난이도 등 메타 정보
"""

from __future__ import annotations
import json
import os
import random
import sys
from dataclasses import dataclass, field
from typing import List, Optional

# ── 프로젝트 루트를 sys.path에 추가 ──────────────────────────────────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from schemas.wbs_schema import WBSTask


# ──────────────────────────────────────────────────────────────────────────
# 데이터 클래스
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class PlantedConstraint:
    """평가용으로 PRD에 심어둔 단일 제약 조건"""
    constraint_id: str
    constraint_type: str           # "tech" | "schedule" | "conflict" | "risk"
    description: str               # 자연어 설명 (PRD에 명시됨)
    check_keywords: List[str]      # 이 키워드들이 WBS task 설명에 존재하면 충족
    check_fn_name: str             # structural_checker 의 검증 함수 이름 (옵션)
    expected_value: Optional[float] = None   # 수치 조건 (예: QA 비율 ≥ 0.20)
    severity: str = "High"         # "High" | "Medium"


@dataclass
class BenchmarkCase:
    """단일 평가 케이스"""
    case_id: str
    domain: str                    # "ecommerce" | "fintech" | "erp" | "saas" | "bigdata"
    difficulty: str                # "easy" | "medium" | "hard"
    prd: object                    # PRDInput
    team: list                     # List[MemberProfile]
    constraints: List[PlantedConstraint]
    description: str = ""
    meta: dict = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────
# 합성 데이터 템플릿
# ──────────────────────────────────────────────────────────────────────────

_DOMAIN_TEMPLATES = {
    "ecommerce": {
        "project_name": "스마트 이커머스 플랫폼 구축",
        "project_goal": "중소 셀러를 위한 AI 기반 상품 추천 및 결제 통합 이커머스 플랫폼",
        "target_users": "자사몰 구축이 필요한 중소 이커머스 사업자",
        "scope": "상품 관리, 결제, 추천 엔진 포함 / 외부 물류 연동 제외",
        "key_features": [
            "회원가입/로그인 (OAuth 2.0)",
            "상품 등록 및 검색",
            "AI 추천 엔진 (협업 필터링)",
            "장바구니 및 주문 관리",
            "결제 연동 (아임포트 PG)",
            "셀러 대시보드",
            "배송 추적 API 연동",
        ],
        "tech_stack": ["React", "FastAPI", "PostgreSQL", "Redis", "Docker"],
        "planted_constraints": [
            PlantedConstraint(
                constraint_id="EC-01",
                constraint_type="tech",
                description="결제 모듈은 반드시 외부 PG사인 '아임포트(iamport)'를 연동해야 하며, "
                            "연동 테스트 및 웹훅 처리 태스크가 WBS에 명시되어야 함",
                check_keywords=["아임포트", "iamport", "pg", "결제 연동", "웹훅", "webhook"],
                check_fn_name="keyword_match",
                severity="High",
            ),
            PlantedConstraint(
                constraint_id="EC-02",
                constraint_type="schedule",
                description="QA 및 테스트 기간이 전체 프로젝트 일정의 20% 이상 확보되어야 함",
                check_keywords=["qa", "테스트", "검증", "품질"],
                check_fn_name="schedule_ratio_check",
                expected_value=0.20,
                severity="High",
            ),
            PlantedConstraint(
                constraint_id="EC-03",
                constraint_type="risk",
                description="AI 추천 엔진의 콜드 스타트 문제(신규 사용자 데이터 없음)에 대한 "
                            "대응 방안이 WBS에 태스크로 포함되어야 함",
                check_keywords=["콜드 스타트", "cold start", "신규 사용자", "초기 데이터", "폴백"],
                check_fn_name="keyword_match",
                severity="Medium",
            ),
            PlantedConstraint(
                constraint_id="EC-04",
                constraint_type="conflict",
                description="디자이너가 3주차에 연차 예정이므로, 와이어프레임은 2주차 내 완료되어야 함",
                check_keywords=["와이어프레임", "wireframe", "ui 설계", "프로토타입"],
                check_fn_name="deadline_check",
                expected_value=2.0,  # end_week ≤ 2
                severity="High",
            ),
        ],
    },

    "fintech": {
        "project_name": "디지털 자산 관리 플랫폼 (로보어드바이저)",
        "project_goal": "개인 투자자를 위한 AI 기반 포트폴리오 자동 리밸런싱 및 세금 최적화 서비스",
        "target_users": "자산 관리가 필요한 개인 투자자 및 고액 자산가",
        "scope": "KYC, 포트폴리오 관리, 자동 리포트 포함 / 해외 거래소 직접 연동 제외",
        "key_features": [
            "본인인증 (KYC) 모듈",
            "포트폴리오 생성 및 관리",
            "로보어드바이저 엔진 (MPT 기반)",
            "실시간 시세 데이터 연동",
            "자동 리밸런싱 스케줄러",
            "세금 보고서 생성",
            "관리자 대시보드 및 모니터링",
        ],
        "tech_stack": ["React", "Spring Boot", "PostgreSQL", "Kafka", "AWS"],
        "planted_constraints": [
            PlantedConstraint(
                constraint_id="FT-01",
                constraint_type="tech",
                description="KYC 모듈은 금융감독원 가이드라인을 준수해야 하며, "
                            "외부 본인인증 업체(NICE평가정보 또는 KCB) API 연동 태스크가 필요함",
                check_keywords=["kyc", "본인인증", "nice", "kcb", "금융감독원", "실명확인"],
                check_fn_name="keyword_match",
                severity="High",
            ),
            PlantedConstraint(
                constraint_id="FT-02",
                constraint_type="risk",
                description="실시간 시세 데이터 외부 API(예: 한국투자증권 Open API)의 "
                            "장애 시 폴백(fallback) 처리 태스크가 WBS에 포함되어야 함",
                check_keywords=["폴백", "fallback", "장애", "api 장애", "캐시", "mock 데이터"],
                check_fn_name="keyword_match",
                severity="High",
            ),
            PlantedConstraint(
                constraint_id="FT-03",
                constraint_type="schedule",
                description="보안 감사(Security Audit) 및 침투 테스트가 출시 전 최소 2주 수행되어야 함",
                check_keywords=["보안 감사", "security audit", "침투 테스트", "penetration", "취약점"],
                check_fn_name="duration_check",
                expected_value=10.0,  # estimated_days ≥ 10
                severity="High",
            ),
            PlantedConstraint(
                constraint_id="FT-04",
                constraint_type="conflict",
                description="백엔드 개발자 2명 중 1명이 4주차에 다른 프로젝트로 이동 예정이므로, "
                            "핵심 API는 3주차까지 완료되어야 함",
                check_keywords=["핵심 api", "core api", "포트폴리오 api", "리밸런싱 api"],
                check_fn_name="deadline_check",
                expected_value=3.0,
                severity="High",
            ),
        ],
    },

    "erp": {
        "project_name": "중소기업 통합 ERP 시스템 구축",
        "project_goal": "제조업 중소기업을 위한 생산 계획, 재고 관리, 회계 통합 ERP 플랫폼",
        "target_users": "디지털 전환이 필요한 중소 제조 기업",
        "scope": "재고, 회계, 생산 관리 포함 / 외부 인사 연동 제외",
        "key_features": [
            "생산 계획 및 MRP 엔진",
            "실시간 재고 관리",
            "구매 발주 및 공급망 관리",
            "회계/원가 관리 모듈",
            "인사/급여 관리",
            "레거시 시스템 데이터 마이그레이션",
            "보고서 및 대시보드",
        ],
        "tech_stack": ["Vue.js", "Spring Boot", "Oracle DB", "Docker", "Jenkins"],
        "planted_constraints": [
            PlantedConstraint(
                constraint_id="ERP-01",
                constraint_type="tech",
                description="기존 레거시 시스템(AS/400 기반)의 데이터를 Oracle DB로 마이그레이션하는 "
                            "ETL 파이프라인 구축 태스크가 반드시 포함되어야 함",
                check_keywords=["마이그레이션", "migration", "etl", "as/400", "레거시", "데이터 이관"],
                check_fn_name="keyword_match",
                severity="High",
            ),
            PlantedConstraint(
                constraint_id="ERP-02",
                constraint_type="schedule",
                description="사용자 교육(UAT) 기간을 최소 2주 확보해야 하며, "
                            "현장 직원 대상 교육 자료 제작 태스크가 포함되어야 함",
                check_keywords=["uat", "사용자 교육", "교육 자료", "사용자 인수 테스트", "현장 교육"],
                check_fn_name="duration_check",
                expected_value=10.0,
                severity="High",
            ),
            PlantedConstraint(
                constraint_id="ERP-03",
                constraint_type="risk",
                description="회계 연도 마감 시즌(12월)에 시스템 오픈이 불가하므로, "
                            "일정이 11월 또는 1월 이후로 설계되어야 함",
                check_keywords=["오픈", "go-live", "출시", "배포", "런칭"],
                check_fn_name="timing_constraint",
                severity="High",
            ),
            PlantedConstraint(
                constraint_id="ERP-04",
                constraint_type="conflict",
                description="MRP 엔진과 회계 모듈은 데이터 정합성을 위해 동시에 개발되어야 하며, "
                            "통합 테스트가 별도로 필요함",
                check_keywords=["mrp 통합", "회계 연동", "데이터 정합성", "통합 테스트"],
                check_fn_name="keyword_match",
                severity="Medium",
            ),
        ],
    },

    "bigdata": {
        "project_name": "실시간 빅데이터 분석 플랫폼",
        "project_goal": "10TB+ 로그 데이터를 실시간 처리하여 이상 탐지 및 비즈니스 인사이트를 제공하는 데이터 플랫폼",
        "target_users": "대용량 데이터를 분석하는 데이터 사이언티스트 및 전략 팀",
        "scope": "로그 수집, 실시간 분석, 대시보드 포함 / 데이터 소스 직접 정제 모듈 제외",
        "key_features": [
            "실시간 로그 수집 파이프라인 (Kafka)",
            "스트림 처리 엔진 (Spark Streaming)",
            "이상 탐지 ML 모델",
            "데이터 레이크 구축 (S3 + Delta Lake)",
            "실시간 대시보드 (Grafana)",
            "데이터 거버넌스 및 카탈로그",
            "CI/CD 파이프라인 (MLOps)",
        ],
        "tech_stack": ["Kafka", "Spark", "Airflow", "Python", "AWS S3", "PostgreSQL"],
        "planted_constraints": [
            PlantedConstraint(
                constraint_id="BD-01",
                constraint_type="tech",
                description="GDPR 준수를 위해 개인식별정보(PII) 마스킹 처리 태스크가 "
                            "데이터 수집 단계에 반드시 포함되어야 함",
                check_keywords=["gdpr", "pii", "개인정보", "마스킹", "익명화", "암호화"],
                check_fn_name="keyword_match",
                severity="High",
            ),
            PlantedConstraint(
                constraint_id="BD-02",
                constraint_type="schedule",
                description="ML 모델 학습에 필요한 히스토리 데이터 수집 기간(최소 4주)이 "
                            "WBS 일정에 반영되어야 함",
                check_keywords=["데이터 수집", "히스토리 데이터", "학습 데이터", "data collection"],
                check_fn_name="duration_check",
                expected_value=20.0,  # ≥ 20 days (4 weeks)
                severity="High",
            ),
            PlantedConstraint(
                constraint_id="BD-03",
                constraint_type="risk",
                description="Kafka 클러스터 장애 시 데이터 유실 방지를 위한 "
                            "재처리(Replay) 메커니즘 설계 태스크가 필요함",
                check_keywords=["재처리", "replay", "데이터 유실", "fault tolerance", "장애 복구"],
                check_fn_name="keyword_match",
                severity="High",
            ),
            PlantedConstraint(
                constraint_id="BD-04",
                constraint_type="conflict",
                description="데이터 엔지니어가 MLOps 파이프라인과 데이터 레이크를 병렬 담당하기 "
                            "어려우므로, 두 작업 사이에 최소 1주 여유 기간이 필요함",
                check_keywords=["mlops", "데이터 레이크", "data lake", "파이프라인"],
                check_fn_name="keyword_match",
                severity="Medium",
            ),
        ],
    },

    "saas": {
        "project_name": "B2B SaaS 프로젝트 관리 툴",
        "project_goal": "중소 기업팀을 위한 AI 기반 프로젝트 관리 및 협업 플랫폼 (Notion + Jira 경쟁 제품)",
        "target_users": "비대면 협업이 잦은 IT 기업 및 스타트업 팀",
        "scope": "협업 에디터, 태스크 관리, 과금 연동 포함 / 온프레미스 구축 버전 제외",
        "key_features": [
            "멀티 테넌트 아키텍처",
            "실시간 협업 에디터 (OT/CRDT)",
            "AI 태스크 자동 생성",
            "Slack/JIRA/GitHub 연동",
            "사용량 기반 과금 (Stripe)",
            "SSO (SAML 2.0)",
            "감사 로그 (Audit Log)",
        ],
        "tech_stack": ["React", "Node.js", "PostgreSQL", "Redis", "WebSocket"],
        "planted_constraints": [
            PlantedConstraint(
                constraint_id="SaaS-01",
                constraint_type="tech",
                description="멀티 테넌트 데이터 격리를 위해 Row-Level Security (RLS) 또는 "
                            "스키마 분리 방식이 DB 설계 태스크에 명시되어야 함",
                check_keywords=["멀티 테넌트", "multi-tenant", "rls", "row level security", "데이터 격리"],
                check_fn_name="keyword_match",
                severity="High",
            ),
            PlantedConstraint(
                constraint_id="SaaS-02",
                constraint_type="risk",
                description="실시간 협업 에디터의 충돌 해결(Conflict Resolution) 알고리즘 검증을 "
                            "위한 부하 테스트 태스크가 포함되어야 함",
                check_keywords=["충돌 해결", "conflict", "crdt", "ot", "operational transform", "부하 테스트"],
                check_fn_name="keyword_match",
                severity="High",
            ),
            PlantedConstraint(
                constraint_id="SaaS-03",
                constraint_type="schedule",
                description="Stripe 결제 연동은 베타 출시 2주 전까지 완료되어야 하며, "
                            "Stripe 웹훅 테스트 및 sandbox 검증 태스크가 필요함",
                check_keywords=["stripe", "결제", "웹훅", "webhook", "sandbox"],
                check_fn_name="keyword_match",
                severity="High",
            ),
            PlantedConstraint(
                constraint_id="SaaS-04",
                constraint_type="conflict",
                description="SOC 2 Type II 인증을 위한 감사 준비 기간(6주)이 일정에 반영되어야 하며, "
                            "보안 팀 리뷰 시간이 확보되어야 함",
                check_keywords=["soc 2", "soc2", "감사", "audit", "보안 인증", "컴플라이언스"],
                check_fn_name="duration_check",
                expected_value=30.0,
                severity="High",
            ),
        ],
    },
}

_TEAM_TEMPLATES = [
    {"name": "김준혁", "years": 6, "stack": ["React", "TypeScript", "Next.js"], "role": "Frontend Developer"},
    {"name": "이서연", "years": 8, "stack": ["FastAPI", "Python", "PostgreSQL"], "role": "Backend Developer"},
    {"name": "박민준", "years": 4, "stack": ["Kafka", "Spark", "Airflow"], "role": "Data Engineer"},
    {"name": "최유진", "years": 5, "stack": ["Figma", "UI/UX", "Prototyping"], "role": "Designer"},
    {"name": "정태양", "years": 7, "stack": ["Docker", "Kubernetes", "AWS"], "role": "DevOps"},
    {"name": "한소희", "years": 3, "stack": ["Pytest", "Selenium", "JMeter"], "role": "QA Engineer"},
    {"name": "오동현", "years": 10, "stack": ["Java", "Spring Boot", "Oracle"], "role": "Backend Developer"},
    {"name": "윤채원", "years": 5, "stack": ["Vue.js", "GraphQL", "REST API"], "role": "Frontend Developer"},
]


def _make_prd(domain: str, budget_weeks: int = 12):
    """PRDInput 합성 생성"""
    try:
        from schemas.prd_schema import PRDInput
    except ImportError:
        # PRDInput이 없으면 SimpleNamespace로 대체
        from types import SimpleNamespace
        tpl = _DOMAIN_TEMPLATES[domain]
        return SimpleNamespace(
            project_name=tpl["project_name"],
            project_goal=tpl["project_goal"],
            key_features=tpl["key_features"],
            tech_stack_requirements=tpl["tech_stack"],
            budget_weeks=budget_weeks,
            deadline=None,
            special_constraints=[c.description for c in tpl["planted_constraints"]],
            raw_text=" ".join(tpl["key_features"]),
        )

    tpl = _DOMAIN_TEMPLATES[domain]
    return PRDInput(
        project_name=tpl["project_name"],
        project_goal=tpl["project_goal"],
        target_users=tpl.get("target_users", "일반 사용자"),
        scope=tpl.get("scope", "전체 범위"),
        key_features=tpl["key_features"],
        tech_stack_requirements=tpl["tech_stack"],
        budget_weeks=budget_weeks,
        special_constraints=[c.description for c in tpl["planted_constraints"]],
    )


def _make_team(size: int = 5):
    """MemberProfile 합성 생성"""
    try:
        from schemas.member_schema import MemberProfile
    except ImportError:
        from types import SimpleNamespace
        selected = random.sample(_TEAM_TEMPLATES, min(size, len(_TEAM_TEMPLATES)))
        return [
            SimpleNamespace(
                member_id=f"M{i+1:02d}",
                name=t["name"],
                years_of_experience=t["years"],
                tech_stack=t["stack"],
                role=None,
                strengths=[],
                primary_skills=t["stack"],
            )
            for i, t in enumerate(selected)
        ]

    selected = random.sample(_TEAM_TEMPLATES, min(size, len(_TEAM_TEMPLATES)))
    members = []
    for i, t in enumerate(selected):
        members.append(MemberProfile(
            member_id=f"M{i+1:02d}",
            name=t["name"],
            years_of_experience=t.get("years", 3.0),
            tech_stack=t.get("stack", ["Python"]),
            primary_skills=t.get("stack", ["Python"])[:3],
            strengths=["성실함", "기술적 이해도"],
            weaknesses=["러닝커브", "도메인 지식"],
        ))
    return members


def generate_benchmark_cases(
    domains: Optional[List[str]] = None,
    team_size: int = 5,
    budget_weeks: int = 12,
) -> List[BenchmarkCase]:
    """
    지정한 도메인들에 대해 BenchmarkCase 목록을 생성.
    domains=None 이면 전체 5개 도메인 모두 생성.
    """
    if domains is None:
        domains = list(_DOMAIN_TEMPLATES.keys())

    cases = []
    difficulty_map = {
        "ecommerce": "medium",
        "fintech": "hard",
        "erp": "hard",
        "bigdata": "medium",
        "saas": "medium",
    }

    for domain in domains:
        if domain not in _DOMAIN_TEMPLATES:
            continue
        tpl = _DOMAIN_TEMPLATES[domain]
        prd = _make_prd(domain, budget_weeks=budget_weeks)
        team = _make_team(team_size)
        constraints = tpl["planted_constraints"]

        cases.append(BenchmarkCase(
            case_id=f"BC-{domain.upper()[:3]}-001",
            domain=domain,
            difficulty=difficulty_map.get(domain, "medium"),
            prd=prd,
            team=team,
            constraints=constraints,
            description=tpl["project_goal"],
        ))

    return cases


def save_benchmark_cases(cases: List[BenchmarkCase], output_dir: str) -> str:
    """BenchmarkCase 목록을 JSON으로 저장 (검증용)"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "benchmark_cases.json")

    serializable = []
    for c in cases:
        serializable.append({
            "case_id": c.case_id,
            "domain": c.domain,
            "difficulty": c.difficulty,
            "description": c.description,
            "project_name": getattr(c.prd, "project_name", ""),
            "constraints": [
                {
                    "constraint_id": con.constraint_id,
                    "type": con.constraint_type,
                    "description": con.description,
                    "check_keywords": con.check_keywords,
                    "severity": con.severity,
                }
                for con in c.constraints
            ],
        })

    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)

    return path
