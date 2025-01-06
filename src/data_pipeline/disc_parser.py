"""
eDISC PDF 파서
sample_data/eDISC_*.pdf 파일을 읽어 구조화된 DiscProfile로 변환합니다.

추출 항목:
  - 이름, DISC 스타일 (C/D/I/S 비율)
  - 주 유형 (C형/D형/I형/S형)
  - 강점 행동 목록
  - 보완 행동 목록
  - 의사소통 스타일
  - 의사결정 스타일
  - 동기부여 환경 (강화/감소)
  - 팀 역할 (검토자/변화추진자 등)
  - 행동 서술 키워드
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class DiscProfile:
    name: str                              # 이름
    disc_style: str                        # "CSD (C-50%, S-45%, D-5%)"
    primary_type: str                      # "C형 (Compliance / 신중형)"
    type_code: str                         # "C", "D", "I", "S", "CD", "CS" 등
    combo_code: str                        # "CSI", "SCI", "CDS" 등 복합 코드
    disc_scores: Dict[str, int]            # {"C": 50, "S": 45, "D": 5, "I": 0}
    strength_behaviors: List[str]          # 감점 행동 목록
    improvement_behaviors: List[str]       # 보완 행동 목록
    behavioral_keywords: List[str]         # 행동 서술 단어 목록
    communication_style: str              # 의사소통 스타일 요약
    decision_style: str                    # 의사결정 스타일 요약
    motivating_factors: List[str]          # 동기부여 환경
    demotivating_factors: List[str]        # 동기 감소 환경
    team_role: str                         # 세부 유형: 검토자/개발자/전문가/적극참여자 등
    team_role_en: str = ""                 # 영문: Assurer/Developer/Specialist/Participator 등
    team_role_description: str = ""        # 역할 설명 (PDF에서 추출한 1~2문장)
    raw_text: str = ""                     # 전체 추출 텍스트 (RAG용)

    def to_rag_text(self) -> str:
        """벡터 DB 인덱싱용 구조화 텍스트"""
        strength_str  = "\n".join(f"  - {b}" for b in self.strength_behaviors[:8])
        improve_str   = "\n".join(f"  - {b}" for b in self.improvement_behaviors[:6])
        motivate_str  = "\n".join(f"  - {f}" for f in self.motivating_factors[:6])
        demotivate_str= "\n".join(f"  - {f}" for f in self.demotivating_factors[:4])
        kw_str        = ", ".join(self.behavioral_keywords[:10])

        return f"""[eDISC 행동유형 프로파일] {self.name}
DISC 스타일: {self.disc_style}
복합 코드: {self.combo_code}
주 유형: {self.primary_type}
세부 팀 역할: {self.team_role} ({self.team_role_en})
역할 설명: {self.team_role_description[:200]}
행동 키워드: {kw_str}

강점 행동:
{strength_str}

보완 행동:
{improve_str}

의사소통 스타일:
{self.communication_style[:300]}

의사결정 스타일:
{self.decision_style[:300]}

동기부여 환경:
{motivate_str}

동기 감소 환경:
{demotivate_str}
"""

    def to_agent_context(self) -> str:
        """에이전트 프롬프트 삽입용 간결 컨텍스트"""
        kws = ", ".join(self.behavioral_keywords[:6])
        strengths = " / ".join(self.strength_behaviors[:4])
        improvements = " / ".join(self.improvement_behaviors[:3])
        return (
            f"DISC({self.disc_style}) | 복합코드:{self.combo_code} | 유형:{self.primary_type}\n"
            f"  세부역할: {self.team_role} ({self.team_role_en})\n"
            f"  역할특성: {self.team_role_description[:120]}\n"
            f"  행동키워드: {kws}\n"
            f"  강점행동: {strengths}\n"
            f"  보완포인트: {improvements}\n"
            f"  의사소통: {self.communication_style[:150]}\n"
            f"  동기부여: {', '.join(self.motivating_factors[:3])}\n"
            f"  스트레스요인: {', '.join(self.demotivating_factors[:2])}"
        )


# ────────────────────────────────────────────────────────────
# 내부 파싱 유틸
# ────────────────────────────────────────────────────────────

def _extract_text_pypdf(pdf_path: str) -> str:
    """pypdf로 전체 텍스트 추출"""
    try:
        import pypdf
        reader = pypdf.PdfReader(pdf_path)
        pages = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:
                pages.append("")
        return "\n".join(pages)
    except Exception as e:
        print(f"[DiscParser] PDF 읽기 실패 ({pdf_path}): {e}")
        return ""


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _extract_bullet_section(text: str, header_pattern: str, stop_patterns: List[str]) -> List[str]:
    """헤더 패턴 이후 불릿 항목을 추출"""
    m = re.search(header_pattern, text)
    if not m:
        return []
    start = m.end()
    # 다음 섹션 헤더까지만 잘라냄
    end = len(text)
    for sp in stop_patterns:
        sm = re.search(sp, text[start:])
        if sm:
            end = min(end, start + sm.start())
    section = text[start:end]
    # • 또는 ⚫ 또는 줄 단위 불릿 항목 추출
    bullets = re.findall(r"[•·✦▪▸◆\-]\s*(.+?)(?=\n|$)", section)
    if not bullets:
        # 개행으로 분리된 일반 텍스트도 fallback으로 처리
        bullets = [l.strip() for l in section.split("\n") if len(l.strip()) > 5]
    return [_clean(b) for b in bullets if len(_clean(b)) > 3]


def _parse_disc_scores(text: str) -> Dict[str, int]:
    """DISC 비율 파싱: "CSD (C - 50%, S - 45%, D - 5%)" 등"""
    scores = {"D": 0, "I": 0, "S": 0, "C": 0}
    m = re.search(r"(?:DISC|스타일)[^\n]*?([DISC]+)\s*\(([^)]+)\)", text)
    if m:
        detail = m.group(2)
        for part in re.findall(r"([DISC])\s*[-:]\s*(\d+)", detail):
            scores[part[0]] = int(part[1])
    # 두 번째 패턴: "D : 5, I : 0, S : 100, C : 0"
    if all(v == 0 for v in scores.values()):
        for part in re.findall(r"([DISC])\s*[:\-]\s*(\d+)", text):
            scores[part[0]] = int(part[1])
    return scores


def _parse_primary_type(text: str) -> tuple[str, str]:
    """
    주 유형 파싱.
    반환: (type_code, full_label) e.g. ("C", "C형 (Compliance / 신중형)")
    """
    type_map = {
        "Dominance": ("D", "D형 (Dominance / 주도형)"),
        "Influence":  ("I", "I형 (Influence / 사교형)"),
        "Steadiness": ("S", "S형 (Steadiness / 안정형)"),
        "Compliance": ("C", "C형 (Compliance / 신중형)"),
        "주도형": ("D", "D형 (주도형)"),
        "사교형": ("I", "I형 (사교형)"),
        "안정형": ("S", "S형 (안정형)"),
        "신중형": ("C", "C형 (신중형)"),
    }
    m = re.search(r"주된 행동스타일은\s+(.+?)[이이고,]", text)
    if m:
        raw = m.group(1).strip()
        for k, v in type_map.items():
            if k in raw:
                return v
        return ("?", raw)

    # 폴백: 스타일 코드에서 첫 글자
    m2 = re.search(r"스타일은\s*:\s*([DISC]+)", text)
    if m2:
        code = m2.group(1)[0]
        return type_map.get(code, (code, code + "형"))
    return ("?", "알 수 없음")


def _parse_name(text: str, filename: str) -> str:
    """이름 파싱 — 파일명에서 우선 추출"""
    bn = os.path.basename(filename)
    m = re.search(r"eDISC[_\-](.+?)\.pdf", bn, re.IGNORECASE)
    if m:
        return m.group(1)
    # 텍스트 내 "이 동 헌" 등 spaced name
    m2 = re.search(r"이\s+름\s*[:\n]\s*(.+?)[\n\r]", text)
    if m2:
        return _clean(m2.group(1))
    return "알 수 없음"


def _parse_team_role(text: str) -> tuple:
    """
    팀 역할 파싱: PDF의 'Team Roles' 섹션에서 세부 유형 추출.
    반환: (한글역할, 영문역할, 역할설명)
    예: ("개발자", "Developer", "개발자는 문제해결을 즐기며...")
    """
    # Extended DISC 표준 16가지 팀 역할
    role_map = {
        "검토자":     "Assurer",
        "개발자":     "Developer",
        "전문가":     "Specialist",
        "적극참여자": "Participator",
        "변화추진자": "Promoter",
        "조정자":     "Coordinator",
        "주도자":     "Director",
        "안정자":     "Stabilizer",
        "촉진자":     "Facilitator",
        "분위기조성자": "Harmonizer",
        "전진자":     "Advancer",
        "실행자":     "Implementor",
        "관찰자":     "Observer",
        "조력자":     "Supporter",
        "완벽주의자": "Perfectionist",
        "설계자":     "Architect",
    }
    en_to_kr = {v: k for k, v in role_map.items()}

    # 패턴 1: "한글역할 (EnglishRole)" — 직접 매칭 (가장 정확)
    for kr_name, en_name in role_map.items():
        # "적극참여자 (Participator)" 또는 "검토자\n (Assurer)" 패턴
        pattern = rf'{re.escape(kr_name)}\s*\(\s*{en_name}\s*\)'
        m = re.search(pattern, text)
        if m:
            # 역할명 이후 ~500자를 가져와 _clean으로 공백 정리
            desc_raw = text[m.end():m.end() + 600]
            desc = _clean(desc_raw)[:300]
            return kr_name, en_name, desc

    # 패턴 2: 영문명으로 역매칭 (한글이 깨진 경우)
    for en_name, kr_name in en_to_kr.items():
        m2 = re.search(rf'\(\s*{en_name}\s*\)', text[5000:])
        if m2:
            desc_raw = text[5000 + m2.end():5000 + m2.end() + 600]
            desc = _clean(desc_raw)[:300]
            return kr_name, en_name, desc

    # 패턴 3: "Team Roles" 섹션 내에서 역할 키워드 찾기
    team_section_start = text.find("Team Roles")
    if team_section_start > 0:
        section = text[team_section_start:team_section_start + 2000]
        for kr_name, en_name in role_map.items():
            if kr_name in section:
                idx = section.find(kr_name)
                desc_raw = section[idx + len(kr_name):]
                desc = _clean(desc_raw)[:300]
                return kr_name, en_name, desc

    return "미분류", "", ""


def _parse_section_paragraph(text: str, header_pattern: str, stop_patterns: List[str]) -> str:
    """헤더 이후 첫 단락 파싱 (의사소통 스타일 등)"""
    m = re.search(header_pattern, text)
    if not m:
        return ""
    start = m.end()
    end = len(text)
    for sp in stop_patterns:
        sm = re.search(sp, text[start:])
        if sm:
            end = min(end, start + sm.start())
    para = _clean(text[start:end])
    return para[:400]


# ────────────────────────────────────────────────────────────
# 메인 파서
# ────────────────────────────────────────────────────────────

def parse_disc_pdf(pdf_path: str) -> Optional[DiscProfile]:
    """단일 eDISC PDF를 파싱해 DiscProfile 반환"""
    raw = _extract_text_pypdf(pdf_path)
    if not raw:
        return None

    name = _parse_name(raw, pdf_path)
    disc_scores = _parse_disc_scores(raw)
    type_code, primary_type = _parse_primary_type(raw)

    # DISC 스타일 표기
    score_parts = " ".join(f"{k}-{v}%" for k, v in disc_scores.items() if v > 0)
    style_codes = "".join(k for k, v in sorted(disc_scores.items(), key=lambda x: -x[1]) if v > 0)
    disc_style = f"{style_codes} ({score_parts})" if score_parts else primary_type

    # ── 강점 행동 ─────────────────────────────────────────
    strength_behaviors = _extract_bullet_section(
        raw,
        r"(?:감점|감점 행동|강점 행동)[은의\s]*[?]?",
        [r"보완해야", r"보완 행동", r"의사소통", r"동기부여", r"팀 활동"]
    )
    if not strength_behaviors:
        strength_behaviors = _extract_bullet_section(
            raw,
            rf"{re.escape(name)}님의 감점 행동",
            [r"보완해야", r"의사소통"]
        )

    # ── 보완 행동 ─────────────────────────────────────────
    improvement_behaviors = _extract_bullet_section(
        raw,
        r"보완해야 할 행동",
        [r"의사소통", r"동기부여", r"팀 활동", r"Extended DISC"]
    )

    # ── 행동 키워드 ───────────────────────────────────────
    kw_text = _parse_section_paragraph(
        raw,
        r"행동스타일을 설명하는 단어",
        [r"의사소통", r"동기", r"학습"]
    )
    behavioral_keywords = [_clean(k) for k in re.split(r"[,、，\n]", kw_text) if len(_clean(k)) > 1]

    # ── 의사소통 스타일 ───────────────────────────────────
    communication_style = _parse_section_paragraph(
        raw,
        r"의사소통 스타일",
        [r"의사결정", r"동기부여", r"회피하는", r"팀 활동"]
    )

    # ── 의사결정 스타일 ───────────────────────────────────
    decision_style = _parse_section_paragraph(
        raw,
        r"의사결정 스타일",
        [r"동기부여", r"회피하는", r"팀 활동", r"학습 스타일"]
    )

    # ── 동기부여 환경 ─────────────────────────────────────
    motivating_factors = _extract_bullet_section(
        raw,
        r"동기부여[를을]? 강화",
        [r"동기부여[를을]? 감소", r"회피하는", r"팀 활동"]
    )
    if not motivating_factors:
        motivating_factors = _extract_bullet_section(
            raw,
            r"동기부여 되는 환경",
            [r"동기부여.*감소", r"회피"]
        )

    # ── 동기 감소 환경 ────────────────────────────────────
    demotivating_factors = _extract_bullet_section(
        raw,
        r"동기부여[를을]? 감소",
        [r"팀 활동", r"회피하는", r"학습", r"Part III", r"직군"]
    )
    if not demotivating_factors:
        demotivating_factors = _extract_bullet_section(
            raw,
            r"회피하는 환경",
            [r"팀 활동", r"Part III"]
        )

    # ── 팀 역할 (세부 유형) ─────────────────────────────────
    team_role_kr, team_role_en, team_role_desc = _parse_team_role(raw)

    # ── 복합 코드 (CSI, SCI 등) ──────────────────────────
    combo_code = style_codes  # 이미 점수 내림차순으로 정렬된 코드

    return DiscProfile(
        name=name,
        disc_style=disc_style,
        primary_type=primary_type,
        type_code=type_code,
        combo_code=combo_code,
        disc_scores=disc_scores,
        strength_behaviors=strength_behaviors,
        improvement_behaviors=improvement_behaviors,
        behavioral_keywords=behavioral_keywords,
        communication_style=communication_style,
        decision_style=decision_style,
        motivating_factors=motivating_factors,
        demotivating_factors=demotivating_factors,
        team_role=team_role_kr,
        team_role_en=team_role_en,
        team_role_description=team_role_desc,
        raw_text=raw[:3000],
    )


def load_all_disc_profiles(disc_dir: str) -> Dict[str, DiscProfile]:
    """
    디렉토리에서 eDISC_*.pdf 를 모두 파싱.
    반환: {이름: DiscProfile}
    """
    profiles: Dict[str, DiscProfile] = {}
    if not os.path.isdir(disc_dir):
        return profiles

    for fname in sorted(os.listdir(disc_dir)):
        if not (fname.lower().startswith("edisc_") and fname.lower().endswith(".pdf")):
            continue
        path = os.path.join(disc_dir, fname)
        try:
            profile = parse_disc_pdf(path)
            if profile:
                profiles[profile.name] = profile
                print(f"[DiscParser] 로드됨: {profile.name} | {profile.disc_style} | {profile.primary_type}")
        except Exception as e:
            print(f"[DiscParser] {fname} 파싱 오류: {e}")

    return profiles


def match_disc_to_member(
    disc_profiles: Dict[str, DiscProfile],
    member_name: str,
) -> Optional[DiscProfile]:
    """팀원 이름으로 DISC 프로파일 매칭 (공백 무시, 부분 일치 지원)"""
    clean_name = re.sub(r"\s+", "", member_name)
    for disc_name, profile in disc_profiles.items():
        clean_disc = re.sub(r"\s+", "", disc_name)
        if clean_name == clean_disc or clean_name in clean_disc or clean_disc in clean_name:
            return profile
    return None
