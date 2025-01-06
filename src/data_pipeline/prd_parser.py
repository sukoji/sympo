"""
PRD 텍스트 파싱 모듈
텍스트 기반 PRD에서 구조화된 PRDInput 객체를 생성합니다.
"""
import re
from typing import List, Optional
from schemas.prd_schema import PRDInput


class PRDParser:
    """
    프로젝트 요구사항 문서 파서
    - 직접 입력 폼 → PRDInput 변환
    - 텍스트 파일 → PRDInput 변환 (키워드 기반 파싱)
    """

    # 자동 감지 키워드 맵
    FEATURE_KEYWORDS = [
        "기능", "feature", "요구사항", "requirement", "구현", "개발"
    ]
    GOAL_KEYWORDS = ["목표", "goal", "objective", "aim", "목적"]
    SCOPE_KEYWORDS = ["범위", "scope", "포함", "제외", "include", "exclude"]

    @staticmethod
    def from_form(
        project_name: str,
        project_goal: str,
        target_users: str,
        scope: str,
        key_features_text: str,
        tech_stack_text: str,
        deadline: Optional[str],
        team_size: int,
        budget_weeks: Optional[int],
        constraints_text: str,
    ) -> PRDInput:
        """
        Streamlit 입력 폼에서 PRDInput 생성
        key_features_text, tech_stack_text, constraints_text는 줄바꿈 구분
        """
        key_features = [
            f.strip() for f in key_features_text.strip().splitlines() if f.strip()
        ]
        tech_stack = [
            t.strip() for t in tech_stack_text.strip().splitlines() if t.strip()
        ]
        constraints = [
            c.strip() for c in constraints_text.strip().splitlines() if c.strip()
        ]

        return PRDInput(
            project_name=project_name,
            project_goal=project_goal,
            target_users=target_users,
            scope=scope,
            key_features=key_features if key_features else ["미정"],
            tech_stack_requirements=tech_stack,
            deadline=deadline if deadline else None,
            team_size=team_size,
            budget_weeks=budget_weeks,
            special_constraints=constraints,
        )

    @staticmethod
    def from_text(raw_text: str, project_name: str = "미정") -> PRDInput:
        """
        원시 텍스트 PRD를 파싱하여 PRDInput 생성.
        소프트웨어 PRD 및 비즈니스/마케팅 PRD 모두 지원합니다.
        """
        lines = raw_text.strip().splitlines()

        # ── 프로젝트명 추출 ────────────────────────────────────────────────────
        extracted_name = PRDParser._extract_project_name(raw_text, project_name)

        # ── 목표 추출 ──────────────────────────────────────────────────────────
        goal = PRDParser._extract_section(lines, ["목표", "goal", "objective", "목적"])
        if not goal:
            # 비즈니스 PRD: 첫 단락 전체를 목표로 사용
            paragraphs = [p.strip() for p in raw_text.split("\n\n") if p.strip()]
            goal = paragraphs[0][:300] if paragraphs else raw_text[:300]
            # 마크다운 볼드/특수문자 제거
            goal = re.sub(r"\*\*(.+?)\*\*", r"\1", goal)

        # ── 범위 추출 ──────────────────────────────────────────────────────────
        scope = PRDParser._extract_section(lines, ["범위", "scope"])
        if not scope:
            scope = "전체 업무 범위 포함"

        # ── 핵심 기능/과제 추출 (한국어 번호 목록 우선) ────────────────────────
        features_raw = PRDParser._extract_korean_numbered_features(raw_text)
        if not features_raw:
            features_raw = PRDParser._extract_list_section(
                lines, ["기능", "feature", "요구사항", "과제", "추진"]
            )

        # ── 기술 스택 추출 ─────────────────────────────────────────────────────
        tech_raw = PRDParser._extract_list_section(lines, ["기술", "tech", "stack"])

        # ── 제약사항 추출 ──────────────────────────────────────────────────────
        constraints_raw = PRDParser._extract_list_section(
            lines, ["제약", "constraint", "조건", "주의", "고려"]
        )

        # ── 타깃 사용자 추출 ───────────────────────────────────────────────────
        target_users = PRDParser._extract_target_users(raw_text)

        return PRDInput(
            project_name=extracted_name,
            project_goal=goal,
            target_users=target_users,
            scope=scope,
            key_features=features_raw if features_raw else ["기능 미정"],
            tech_stack_requirements=tech_raw,
            special_constraints=constraints_raw,
            raw_text=raw_text,
        )

    @staticmethod
    def _extract_project_name(raw_text: str, default: str = "미정") -> str:
        """
        프로젝트명 추출 우선순위:
        1. 볼드+따옴표 브랜드명 (**'P마켓'**, **"P마켓"**)
        2. 짧은 볼드 브랜드명 (**P마켓**) — 숫자·단위 제외
        3. 첫 줄 헤더에서 콜론 뒤 값 ("비즈니스 상황: 대형 마켓 매출 증대")
        4. 첫 줄 전체
        """
        # 1. 볼드+따옴표 조합 (**'P마켓'**, **"P마켓"**)
        bold_quote = re.search(r"\*\*['\"\u2018\u2019\u201c\u201d](.+?)['\"\u2018\u2019\u201c\u201d]\*\*", raw_text)
        if bold_quote:
            candidate = bold_quote.group(1).strip()
            if 1 <= len(candidate) <= 30 and not re.search(r"\d", candidate):
                return candidate

        # 2. 볼드 텍스트 (**....**) — 숫자, 단위, 메타 단어 제외
        meta_words = {"비즈니스", "상황", "참고", "주의", "목표", "범위", "기능", "요구사항", "추가"}
        for match in re.finditer(r"\*\*(.+?)\*\*", raw_text):
            candidate = match.group(1).strip().strip("'\"")
            # 숫자 포함(연도·금액 등) 또는 메타 단어 포함이면 제외
            if (2 <= len(candidate) <= 20
                    and not re.search(r"\d", candidate)
                    and not any(w in candidate for w in meta_words)):
                return candidate

        # 3. 첫 줄 헤더 "비즈니스 상황: XXX" 형식 → 콜론 뒤 값 사용
        for line in raw_text.strip().splitlines()[:3]:
            cleaned = re.sub(r"\*\*|##|#", "", line).strip()
            if ":" in cleaned:
                after_colon = cleaned.split(":", 1)[1].strip()
                if after_colon and 2 <= len(after_colon) <= 50:
                    return after_colon

        # 4. 첫 줄 전체
        for line in raw_text.strip().splitlines()[:5]:
            cleaned = re.sub(r"\*\*|##|#", "", line).strip()
            if cleaned and len(cleaned) > 2:
                return cleaned[:50]

        return default

    @staticmethod
    def _extract_korean_numbered_features(raw_text: str) -> list:
        """
        한국어 번호 목록 추출:
        - "첫째, ~" / "둘째, ~" / "셋째, ~" / "넷째, ~" / "다섯째, ~"
        - "1. ~" / "2. ~" 형식
        """
        features = []

        # 한국어 서수 패턴
        ordinals = ["첫째", "둘째", "셋째", "넷째", "다섯째", "여섯째", "일곱째"]
        for ordinal in ordinals:
            pattern = rf"{ordinal}[,\s、]+(.+?)(?=\n|$)"
            match = re.search(pattern, raw_text)
            if match:
                feat = match.group(1).strip().rstrip(".")
                if feat:
                    features.append(feat)

        if features:
            return features

        # 숫자 목록 패턴 (1. 2. 3.)
        for match in re.finditer(r"^\s*\d+[.）\)]\s*(.+)$", raw_text, re.MULTILINE):
            feat = match.group(1).strip()
            if feat and len(feat) > 3:
                features.append(feat)

        return features

    @staticmethod
    def _extract_target_users(raw_text: str) -> str:
        """타깃 사용자/고객 정보 추출"""
        # 1. 명시적 '타깃 사용자', '대상 고객' 섹션
        section_match = re.search(
            r"(?:타깃\s*사용자|대상\s*고객|target\s*user)[^\n]*\n(.+?)(?:\n\n|\Z)",
            raw_text, re.IGNORECASE | re.DOTALL
        )
        if section_match:
            return section_match.group(1).strip()[:100]

        # 2. 고객 유형 키워드 빈도 기반 추출 (중복 제거, 등장 순서 유지)
        user_keywords = re.findall(
            r"(학부모|학생\s*자녀|노인|실버|유아|어린이|주부|직장인|회원|소비자|고객)",
            raw_text
        )
        if user_keywords:
            # 중복 제거 (순서 유지)
            seen = set()
            unique = []
            for kw in user_keywords:
                kw_clean = kw.strip()
                if kw_clean not in seen:
                    seen.add(kw_clean)
                    unique.append(kw_clean)
            return ", ".join(unique[:5]) + " 등"

        return "일반 고객"

    @staticmethod
    def _extract_section(lines: List[str], keywords: List[str]) -> str:
        """섹션 헤더 이후 텍스트 추출"""
        capture = False
        result_lines = []
        for line in lines:
            low = line.lower()
            if any(kw in low for kw in keywords):
                capture = True
                continue
            if capture:
                if line.startswith("#") or line.startswith("=="):
                    break
                if line.strip():
                    result_lines.append(line.strip())
                    if len(result_lines) >= 3:
                        break
        return " ".join(result_lines)

    @staticmethod
    def _extract_list_section(lines: List[str], keywords: List[str]) -> List[str]:
        """섹션 내 목록 항목 추출"""
        capture = False
        items = []
        for line in lines:
            low = line.lower()
            if any(kw in low for kw in keywords):
                capture = True
                continue
            if capture:
                if line.startswith("#") or line.startswith("=="):
                    break
                stripped = line.strip().lstrip("-•*").strip()
                if stripped:
                    items.append(stripped)
                elif items:
                    break
        return items
