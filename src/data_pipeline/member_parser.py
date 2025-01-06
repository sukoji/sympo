"""
팀원 메타데이터 파서
이력서/CV 텍스트 또는 폼 입력에서 MemberProfile 구조를 생성합니다.
API 없이도 규칙 기반으로 동작하며, LLM 연동 시 정밀도가 향상됩니다.
"""
import re
import uuid
from typing import List, Optional

from schemas.member_schema import MemberProfile, MemberRole, PastProject, PersonalityTrait
import json
from agents.llm_config import get_llm, normalize_content
from langchain_core.messages import HumanMessage, SystemMessage


# 기술/역량 키워드 → 직군 분류 규칙 (중복 시 가중치)
ROLE_KEYWORDS: dict = {
    # ── 소프트웨어 개발 ──────────────────────────────────────────────────────
    MemberRole.FRONTEND: [
        "react", "vue", "angular", "next.js", "typescript", "frontend", "framer motion",
        "프론트", "vanilla js", "tailwind", "styled-components", "web", "웹",
    ],
    MemberRole.BACKEND: [
        "python", "fastapi", "django", "flask", "java", "spring", "node",
        "api", "backend", "백엔드", "redis", "celery", "mysql", "mongodb",
    ],
    MemberRole.DESIGNER: [
        "figma", "sketch", "photoshop", "illustrator", "design", "디자인",
        "ui/ux", "prototyping", "프로토타이핑", "zeplin", "ux", "ui",
    ],
    MemberRole.QA: [
        "qa", "test", "테스트", "selenium", "cypress", "pytest", "jest",
        "automation", "자동화", "검증", "품질", "quality",
    ],
    MemberRole.DATA: [
        "kafka", "spark", "hadoop", "airflow", "big data", "빅데이터",
        "data engineering", "pipeline", "etl", "postgresql",
    ],
    MemberRole.DEVOPS: [
        "docker", "kubernetes", "aws", "gcp", "azure", "ci/cd",
        "terraform", "ansible", "infra", "인프라", "devops",
    ],
    # ── 비즈니스 / 마케팅 ────────────────────────────────────────────────────
    MemberRole.DATA_ANALYST: [
        "데이터 분석", "data analysis", "sql", "tableau", "power bi", "분석가",
        "통계", "r언어", "pandas", "시각화", "분석 리포트", "분석보고서",
    ],
    MemberRole.MARKETING: [
        "마케팅", "marketing", "프로모션", "promotion", "sns", "광고", "캠페인",
        "crm", "고객 관리", "브랜드", "branding", "콘텐츠", "content",
    ],
    MemberRole.BUSINESS: [
        "경영", "business", "전략", "strategy", "기획", "사업계획", "컨설팅",
        "시장조사", "market research", "bm", "비즈니스 모델", "경영지원",
    ],
    MemberRole.MOBILE: [
        "ios", "android", "flutter", "react native", "swift", "kotlin",
        "mobile", "모바일", "앱 개발", "app", "앱",
    ],
    MemberRole.OPERATIONS: [
        "운영", "operation", "물류", "logistics", "매장", "store", "md",
        "재고", "inventory", "공급망", "scm", "현장", "서비스 운영",
    ],
}

STRENGTH_KEYWORDS = [
    "꼼꼼", "빠른", "협력", "소통", "분석", "창의", "리더십",
    "문서화", "체계", "책임감", "주도", "경험",
]

WEAKNESS_KEYWORDS = [
    "러닝커브", "learning curve", "부족", "미흡", "어려움",
    "소통 어려움", "시간 관리", "완벽주의",
]


class MemberParser:
    """
    팀원 메타데이터 파서
    - 폼 직접 입력 → MemberProfile
    - 이력서 텍스트 → MemberProfile (규칙 기반 NLP)
    """

    @staticmethod
    def from_form(
        name: str,
        role: str,
        years_of_experience: float,
        tech_stack_text: str,
        strengths_text: str,
        weaknesses_text: str,
        personality_text: str,
        past_projects_text: str,
        member_id: Optional[str] = None,
        availability: int = 100,
    ) -> MemberProfile:
        """Streamlit 폼에서 MemberProfile 생성"""
        mid = member_id or f"MBR-{uuid.uuid4().hex[:4].upper()}"
        tech = [t.strip() for t in tech_stack_text.strip().splitlines() if t.strip()]
        strengths = [s.strip() for s in strengths_text.strip().splitlines() if s.strip()]
        weaknesses = [w.strip() for w in weaknesses_text.strip().splitlines() if w.strip()]

        # 성향 파싱
        traits = MemberParser._parse_traits(personality_text)

        # 과거 프로젝트 간단 파싱
        past = MemberParser._parse_past_projects_text(past_projects_text)

        # 역할 매핑
        role_enum = MemberParser._map_role(role)

        return MemberProfile(
            member_id=mid,
            name=name,
            role=role_enum,
            years_of_experience=years_of_experience,
            tech_stack=tech if tech else ["미정"],
            primary_skills=tech[:3] if tech else ["미정"],
            strengths=strengths if strengths else ["성실함"],
            weaknesses=weaknesses if weaknesses else ["미정"],
            personality_traits=traits,
            past_projects=past,
            availability_percent=availability,
        )

    @staticmethod
    def from_resume_text(resume_text: str, name: str = "미정") -> MemberProfile:
        """이력서 텍스트에서 MemberProfile 자동 추출 (규칙 기반)"""
        mid = f"MBR-{uuid.uuid4().hex[:4].upper()}"
        lower_text = resume_text.lower()

        # 텍스트에서 이름 추출 시도 (규칙 기반)
        extracted_name = name
        import re
        name_match = re.search(r'(?:이름|성명|Name|이 름)\s*[:\s]\s*([가-힣a-zA-Z ]{2,10})', resume_text)
        if name_match:
            extracted_name = name_match.group(1).strip()
        elif name == "미정" or name.endswith((".pdf", ".txt", ".md")):
            # 앞의 3줄 정도에서 이름처럼 보이는 것 찾기 (공백으로 구분된 2~4자 한글)
            lines = [l.strip() for l in resume_text.splitlines() if l.strip()]
            for line in lines[:3]:
                if re.match(r'^[가-힣]{2,4}$', line):
                    extracted_name = line
                    break
        
        name = extracted_name

        # 기술 스택 추출 (알려진 키워드 매칭)
        known_techs = [
            "Python", "Java", "React", "Vue", "Angular", "Node.js", "FastAPI",
            "Django", "Flask", "Spring", "Docker", "Kubernetes", "AWS", "GCP",
            "PostgreSQL", "MySQL", "MongoDB", "Redis", "TypeScript", "JavaScript",
            "Figma", "Selenium", "Pytest", "Jest", "Cypress", "Git", "Jira",
            "Kafka", "Spark", "Airflow", "Terraform", "Next.js", "Sketch",
            "Go", "gRPC", "Hadoop", "k6", "GitHub Actions", "Adobe XD",
        ]
        tech_found = [t for t in known_techs if t.lower() in lower_text]

        # 기술 스택 섹션에서 직접 파싱 (위 known_techs에 없는 항목 포함)
        tech_section = MemberParser._extract_section(resume_text, ["기술 스택", "기술스택", "tech stack"])
        if tech_section:
            extra = [t.strip() for t in re.split(r"[,\n]", tech_section) if t.strip() and len(t.strip()) > 1]
            for e in extra:
                if e not in tech_found:
                    tech_found.append(e)

        # 역할 추론 (tech_found 기반)
        role = MemberParser._infer_role_from_text(lower_text, tech_found)

        # 경력 연수 추출
        years = MemberParser._extract_years(resume_text)

        # 강점: 섹션 텍스트 우선 추출, 없으면 키워드 매칭 fallback
        strength_section = MemberParser._extract_section(resume_text, ["주요 강점", "강점", "strengths"])
        if strength_section:
            strengths = [s.strip() for s in strength_section.splitlines() if s.strip()]
        else:
            matched = MemberParser._extract_keywords(resume_text, STRENGTH_KEYWORDS)
            strengths = matched if matched else ["성실함"]

        # 약점: 섹션 텍스트 우선 추출
        weakness_section = MemberParser._extract_section(resume_text, ["약점/성장 포인트", "약점", "weaknesses", "성장 포인트"])
        if weakness_section:
            weaknesses = [w.strip() for w in weakness_section.splitlines() if w.strip()]
        else:
            matched = MemberParser._extract_keywords(resume_text, WEAKNESS_KEYWORDS)
            weaknesses = matched if matched else ["성장 중"]

        return MemberProfile(
            member_id=mid,
            name=name,
            role=role,
            years_of_experience=years,
            tech_stack=tech_found if tech_found else ["미정"],
            primary_skills=tech_found[:3] if tech_found else ["미정"],
            strengths=strengths,
            weaknesses=weaknesses,
            raw_resume_text=resume_text,
        )

    @staticmethod
    def _extract_json_from_llm_response(raw: str) -> dict:
        """
        LLM 응답 문자열에서 JSON 객체를 안전하게 추출합니다.
        단계: markdown fence 제거 → JSON 경계 탐색 → 제어문자 정리 → repair → 파싱
        """
        text = raw.strip()

        # 1단계: markdown code fence 제거
        text = re.sub(r'```(?:json)?\s*', '', text).strip()

        # 2단계: JSON 객체 경계 탐색 ({ ... })
        first_brace = text.find('{')
        last_brace = text.rfind('}')
        if first_brace != -1 and last_brace > first_brace:
            text = text[first_brace:last_brace + 1]

        # 3단계: 제어문자 제거 (탭/줄바꿈은 유지)
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)

        # 4단계: 빠른 파싱 시도
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 5단계: 일반적 LLM 오류 복구
        repaired = text
        # trailing comma 제거: ,} → }  ,] → ]
        repaired = re.sub(r',\s*([}\]])', r'\1', repaired)
        # 연속 객체 사이 누락 콤마: }{ → },{
        repaired = re.sub(r'\}\s*\{', '},{', repaired)
        # 작은따옴표 → 큰따옴표 (JSON 표준)
        repaired = repaired.replace("'", '"')

        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass

        # 6단계: 괄호 밸런스 강제 맞추기
        open_braces = repaired.count('{') - repaired.count('}')
        open_brackets = repaired.count('[') - repaired.count(']')
        repaired = repaired.rstrip().rstrip(',')
        repaired += '}' * max(0, open_braces)
        repaired += ']' * max(0, open_brackets)

        try:
            return json.loads(repaired)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON 복구 실패: {e}\n원문 앞부분: {raw[:300]}")

    @staticmethod
    def _safe_parse_trait(t: str) -> Optional[PersonalityTrait]:
        """PersonalityTrait 문자열을 Enum key 또는 value로 매칭"""
        # Enum key로 직접 매칭 (예: "DETAIL_ORIENTED")
        try:
            return PersonalityTrait[t]
        except KeyError:
            pass
        # Enum value(한국어)로 매칭 (예: "꼼꼼함")
        for member in PersonalityTrait:
            if member.value == t:
                return member
        # 부분 매칭 (예: "꼼꼼" → "꼼꼼함")
        t_lower = t.lower().strip()
        for member in PersonalityTrait:
            if t_lower in member.value or t_lower in member.name.lower():
                return member
        return None

    @staticmethod
    def _safe_parse_project(p: dict) -> Optional[PastProject]:
        """PastProject dict를 유연하게 파싱 (필드명 변형 허용)"""
        try:
            # duration 필드 이름 변형 허용
            duration = p.get("duration_weeks")
            if duration is None:
                # 흔한 변형: duration_month(s), duration, weeks 등
                for alt_key in ["duration_months", "duration_month", "duration", "weeks"]:
                    if alt_key in p:
                        val = p[alt_key]
                        if "month" in alt_key:
                            duration = int(float(val) * 4)  # 월 → 주 변환
                        else:
                            duration = int(float(val))
                        break
            if duration is None:
                duration = 4  # 기본값

            return PastProject(
                name=p.get("name", "미정"),
                role=p.get("role", "팀원"),
                duration_weeks=int(duration),
                technologies=p.get("technologies", []),
                notable_outcomes=p.get("notable_outcomes", p.get("outcomes", "미정")),
            )
        except Exception as e:
            print(f"[WARN] PastProject 파싱 실패: {e} | 데이터: {p}")
            return None

    @staticmethod
    def from_resume_text_llm(resume_text: str, name: str = "미정") -> MemberProfile:
        """LLM을 사용하여 이력서에서 정밀한 MemberProfile 추출"""
        try:
            llm = get_llm(temperature=0.0, max_tokens=2000)
            # MockLLM인 경우 실제 LLM의 상세 분석 결과를 시뮬레이션
            from agents.llm_config import MockLLM
            if isinstance(llm, MockLLM):
                base = MemberParser.from_resume_text(resume_text, name)

                # 상세 분석 시뮬레이션 (구체적인 문장 생성)
                if base.name != "미정":
                    name_ref = base.name
                else:
                    name_ref = "이 팀원"

                detailed_strengths = []
                for s in base.strengths[:3]:
                    detailed_strengths.append(f"{name_ref}님은 {s} 분야에서 탁월한 전문성을 보유하고 있으며, 프로젝트의 품질을 높이는 데 기여할 수 있는 역량을 갖추고 있습니다.")
                if not detailed_strengths:
                    detailed_strengths = [f"{name_ref}님은 체계적인 업무 처리와 성실한 태도를 바탕으로 안정적인 협업이 가능한 강점을 지니고 있습니다."]

                detailed_weaknesses = []
                for w in base.weaknesses[:2]:
                    detailed_weaknesses.append(f"{w} 측면에서는 향후 기술적 심화 학습이나 실무 경험을 통해 더욱 완성도 높은 전문가로 성장할 가능성이 큽니다.")
                if not detailed_weaknesses:
                    detailed_weaknesses = [f"{name_ref}님은 새로운 프레임워크나 최신 트렌드 습득에 있어 다소 시간이 필요할 수 있으나, 꾸준한 노력을 통해 극복 중입니다."]

                base.strengths = detailed_strengths
                base.weaknesses = detailed_weaknesses
                return base

            system_prompt = """당신은 HR 전문 데이터 추출 전문가입니다.
제공된 이력서 텍스트에서 다음 JSON 스펙에 맞는 데이터를 추출하여 반환하세요.
모든 필드는 한국어로 작성하고, 리스트 필드는 최소 3개 이상의 항목을 포함하도록 노력하세요.

반드시 아래 형식의 JSON만 출력하세요. 설명이나 주석은 절대 포함하지 마세요.

{
  "name": "이름 (이력서 본문에서 실명을 찾아 기재. 찾을 수 없으면 '미정')",
  "role": "PM | PLANNER | FRONTEND | BACKEND | FULLSTACK | DESIGNER | QA | DATA | DEVOPS | DATA_ANALYST | MARKETING | BUSINESS | MOBILE | OPERATIONS 중 하나",
  "years_of_experience": 숫자,
  "tech_stack": ["기술1", "기술2"],
  "primary_skills": ["핵심역량1", "핵심역량2", "핵심역량3"],
  "strengths": ["업무 강점을 구체적 문장(10자 이상)으로 3개"],
  "weaknesses": ["성장 포인트를 구체적 문장(10자 이상)으로 3개"],
  "personality_traits": ["DETAIL_ORIENTED", "BIG_PICTURE", "COLLABORATIVE", "INDEPENDENT", "RISK_AVERSE", "RISK_TAKER", "FAST_EXECUTOR", "THOROUGH_PLANNER" 중 해당하는 것만],
  "past_projects": [{"name": "프로젝트명", "role": "역할", "duration_weeks": 숫자, "technologies": ["기술"], "notable_outcomes": "성과"}]
}"""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"다음 이력서에서 데이터를 추출하세요:\n\n{resume_text}")
            ]

            response = llm.invoke(messages)
            raw_content = normalize_content(response.content)
            print(f"[LLM 이력서 파싱] 응답 길이: {len(raw_content)}, 앞부분: {raw_content[:200]}")

            # 안전한 JSON 추출 (다단계 복구 포함)
            data = MemberParser._extract_json_from_llm_response(raw_content)

            # Enum 및 ID 처리
            mid = f"MBR-{uuid.uuid4().hex[:4].upper()}"
            role = MemberParser._map_role(data.get("role", ""))

            # PersonalityTrait: key와 value 모두 허용
            traits = []
            for t in data.get("personality_traits", []):
                parsed = MemberParser._safe_parse_trait(t)
                if parsed:
                    traits.append(parsed)

            # PastProject: 필드명 변형 허용
            projects = []
            for p in data.get("past_projects", []):
                parsed = MemberParser._safe_parse_project(p)
                if parsed:
                    projects.append(parsed)

            # years_of_experience 안전 변환
            try:
                yoe = float(data.get("years_of_experience", 2.0))
            except (ValueError, TypeError):
                yoe = 2.0

            return MemberProfile(
                member_id=mid,
                name=data.get("name", name),
                role=role,
                years_of_experience=yoe,
                tech_stack=data.get("tech_stack", ["미정"]),
                primary_skills=data.get("primary_skills", data.get("tech_stack", ["미정"])[:3]),
                strengths=data.get("strengths", ["성실함"]),
                weaknesses=data.get("weaknesses", ["미정"]),
                personality_traits=traits,
                past_projects=projects,
                raw_resume_text=resume_text
            )
        except Exception as e:
            print(f"[WARN] LLM 파싱 실패 (규칙 기반 폴백): {e}")
            return MemberParser.from_resume_text(resume_text, name)

    @staticmethod
    def _extract_section(text: str, headers: List[str]) -> str:
        """지정된 헤더 이후 다음 섹션 직전까지의 텍스트를 추출합니다."""
        lines = text.splitlines()
        start_idx = None
        inline_content = ""
        for i, line in enumerate(lines):
            stripped = line.strip()
            for header in headers:
                if stripped.lower().startswith(header.lower()):
                    rest = stripped[len(header):].strip()
                    if rest.startswith(":"):
                        rest = rest[1:].strip()
                    inline_content = rest
                    start_idx = i + 1
                    break
            if start_idx is not None:
                break

        if start_idx is None:
            return ""

        content_lines = [inline_content] if inline_content else []
        for line in lines[start_idx:]:
            # 빈 줄 2개 이상 or 새 섹션 헤더(짧은 줄로만 구성된 제목줄) 만나면 중단
            stripped = line.strip()
            if not stripped:
                if content_lines:
                    # 이미 내용이 있고 빈 줄이면 섹션 종료로 간주
                    break
                continue
            # 새 섹션 헤더 패턴: 짧고 콜론으로 끝나거나 문장 부호 없는 짧은 줄
            looks_like_header = (
                len(stripped) < 30
                and not any(c in stripped for c in ['(', ',', '.'])
                and (
                    stripped.endswith(":")
                    or stripped.endswith("：")
                    or stripped.lower().rstrip(":：") in {
                        "기술 스택", "기술스택", "tech stack", "주요 강점", "강점",
                        "strengths", "약점", "weaknesses", "성장 포인트",
                    }
                )
            )
            if looks_like_header:
                # 기존 내용이 있으면 섹션 헤더로 간주해 종료
                if content_lines:
                    break
            content_lines.append(stripped)

        return "\n".join(content_lines).strip()

    @staticmethod
    def _map_role(role_str: str) -> MemberRole:
        """역할 문자열 → MemberRole Enum 변환 (소프트웨어·비즈니스 직군 모두 지원)"""
        mapping = {
            # 소프트웨어 개발 직군
            "pm": MemberRole.PM,
            "프로젝트 매니저": MemberRole.PM,
            "플래너": MemberRole.PLANNER,
            "planner": MemberRole.PLANNER,
            "프론트엔드": MemberRole.FRONTEND,
            "frontend": MemberRole.FRONTEND,
            "백엔드": MemberRole.BACKEND,
            "backend": MemberRole.BACKEND,
            "풀스택": MemberRole.FULLSTACK,
            "fullstack": MemberRole.FULLSTACK,
            "디자이너": MemberRole.DESIGNER,
            "designer": MemberRole.DESIGNER,
            "qa": MemberRole.QA,
            "테스터": MemberRole.QA,
            "품질": MemberRole.QA,
            "data engineer": MemberRole.DATA,
            "데이터 엔지니어": MemberRole.DATA,
            "devops": MemberRole.DEVOPS,
            "인프라": MemberRole.DEVOPS,
            # 비즈니스 / 마케팅 직군
            "데이터 분석": MemberRole.DATA_ANALYST,
            "data analyst": MemberRole.DATA_ANALYST,
            "분석가": MemberRole.DATA_ANALYST,
            "마케팅": MemberRole.MARKETING,
            "marketing": MemberRole.MARKETING,
            "마케터": MemberRole.MARKETING,
            "경영": MemberRole.BUSINESS,
            "business": MemberRole.BUSINESS,
            "기획자": MemberRole.BUSINESS,
            "경영지원": MemberRole.BUSINESS,
            "모바일": MemberRole.MOBILE,
            "mobile": MemberRole.MOBILE,
            "앱 개발": MemberRole.MOBILE,
            "운영": MemberRole.OPERATIONS,
            "operation": MemberRole.OPERATIONS,
            "매장": MemberRole.OPERATIONS,
            "물류": MemberRole.OPERATIONS,
        }
        lower = role_str.lower().strip()
        for k, v in mapping.items():
            if k in lower:
                return v
        # 마지막 fallback: ROLE_KEYWORDS 기반 추론
        return MemberParser._infer_role_from_text(lower, [])

    @staticmethod
    def _infer_role_from_text(text: str, techs: List[str]) -> MemberRole:
        """텍스트 + 기술스택으로 역할 추론"""
        scores: dict = {role: 0 for role in MemberRole}
        for role, keywords in ROLE_KEYWORDS.items():
            for kw in keywords:
                if kw in text:
                    scores[role] = scores.get(role, 0) + 1
        
        # 가중치 조정 (강력한 직군 지표)
        if "figma" in text and "sketch" in text:
            scores[MemberRole.DESIGNER] += 3
        if "qa" in text or "pytest" in text or "selenium" in text:
            scores[MemberRole.QA] += 3
        if "kafka" in text or "spark" in text or "airflow" in text:
            scores[MemberRole.DATA] += 3
        if "react" in text or "next.js" in text:
            scores[MemberRole.FRONTEND] += 2
        if ("aws" in text or "docker" in text) and "terraform" in text:
            scores[MemberRole.DEVOPS] += 2
            
        # 신호가 전혀 없으면 BACKEND 기본값(편향 원인)을 피하기 위해
        # 가장 첫 번째 후보군으로 순환 배치 - 하지만 결정적 해시로 배분하여 한 명에만 쏠리지 않도록 함.
        # 호출자가 이름을 붙여주므로 text 해시로 분산.
        if max(scores.values()) > 0:
            return max(scores, key=scores.get)
        # 결정적 fallback: text 해시 기반 분산 (한 쪽으로 몰리는 현상 방지)
        _fallback_pool = [
            MemberRole.BACKEND, MemberRole.FRONTEND, MemberRole.FULLSTACK,
            MemberRole.DESIGNER, MemberRole.QA, MemberRole.DATA,
        ]
        return _fallback_pool[abs(hash(text)) % len(_fallback_pool)]

    @staticmethod
    def _extract_years(text: str) -> float:
        """텍스트에서 경력 연수 추출"""
        patterns = [
            r"(\d+(?:\.\d+)?)\s*년\s*경력",
            r"(?:경력|실무)\s*[:：]?\s*(\d+(?:\.\d+)?)\s*년",
            r"(\d+(?:\.\d+)?)\s*years?\s*of\s*experience",
            r"experience[:\s]+(\d+(?:\.\d+)?)\s*years?",
        ]
        for p in patterns:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                return float(m.group(1))
        return 2.0  # 기본값

    @staticmethod
    def _extract_keywords(text: str, keywords: List[str]) -> List[str]:
        """텍스트에서 키워드 목록 추출"""
        found = []
        for kw in keywords:
            if kw in text:
                found.append(kw)
        return found[:3]  # 최대 3개

    @staticmethod
    def _parse_traits(text: str) -> List[PersonalityTrait]:
        """성향 텍스트를 PersonalityTrait 목록으로 변환"""
        trait_map = {
            "꼼꼼": PersonalityTrait.DETAIL_ORIENTED,
            "큰그림": PersonalityTrait.BIG_PICTURE,
            "협력": PersonalityTrait.COLLABORATIVE,
            "독립": PersonalityTrait.INDEPENDENT,
            "리스크 회피": PersonalityTrait.RISK_AVERSE,
            "도전": PersonalityTrait.RISK_TAKER,
            "빠른 실행": PersonalityTrait.FAST_EXECUTOR,
            "철저한 계획": PersonalityTrait.THOROUGH_PLANNER,
        }
        traits = []
        lower = text.lower()
        for key, trait in trait_map.items():
            if key in lower:
                traits.append(trait)
        return traits

    @staticmethod
    def _parse_past_projects_text(text: str) -> List[PastProject]:
        """간단한 텍스트 형식 과거 프로젝트 파싱"""
        projects = []
        for block in text.strip().split("\n\n"):
            if block.strip():
                lines = block.strip().splitlines()
                projects.append(
                    PastProject(
                        name=lines[0].strip() if lines else "미정",
                        role="팀원",
                        duration_weeks=4,
                        technologies=[],
                        notable_outcomes=lines[1].strip() if len(lines) > 1 else "미정",
                    )
                )
        return projects[:3]  # 최대 3개
