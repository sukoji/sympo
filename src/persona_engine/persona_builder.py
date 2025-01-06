"""팀원 프로필 기반 LLM 페르소나 생성."""
from typing import Dict, List

from schemas.member_schema import MemberProfile, MemberRole
from persona_engine.persona_templates import get_persona_template


_ROLE_TO_TEMPLATE = {
    MemberRole.PLANNER: "planner",
    MemberRole.FRONTEND: "frontend",
    MemberRole.BACKEND: "backend",
    MemberRole.DESIGNER: "designer",
    MemberRole.QA: "qa",
    MemberRole.PM: "supervisor",
    MemberRole.FULLSTACK: "backend",
    MemberRole.DATA: "backend",
    MemberRole.DEVOPS: "backend",
    MemberRole.DATA_ANALYST: "planner",
    MemberRole.MARKETING: "designer",
    MemberRole.BUSINESS: "planner",
    MemberRole.MOBILE: "frontend",
    MemberRole.OPERATIONS: "planner",
}


class PersonaBuilder:
    @staticmethod
    def generate_team_summary(team_members: List[MemberProfile]) -> str:
        lines = []
        for m in team_members:
            role = m.role.value if m.role else "미지정"
            skills = ", ".join((m.primary_skills or m.tech_stack)[:4])
            lines.append(f"- {m.name} ({role}, {m.years_of_experience}년): {skills}")
        return "\n".join(lines)

    @staticmethod
    def build_supervisor_persona(name: str, team_summary: str) -> str:
        template = get_persona_template("supervisor")
        return template.format(name=name) + f"\n\n[팀 구성 요약]\n{team_summary}"

    @staticmethod
    def _infer_template(member: MemberProfile) -> str:
        if member.role and member.role in _ROLE_TO_TEMPLATE:
            return _ROLE_TO_TEMPLATE[member.role]
        tech = " ".join(member.tech_stack).lower()
        if any(k in tech for k in ("react", "vue", "frontend", "ui")):
            return "frontend"
        if any(k in tech for k in ("figma", "design", "ux")):
            return "designer"
        if any(k in tech for k in ("test", "qa", "selenium")):
            return "qa"
        if any(k in tech for k in ("plan", "pm", "기획")):
            return "planner"
        return "backend"

    @staticmethod
    def build_persona(member: MemberProfile) -> str:
        template_key = PersonaBuilder._infer_template(member)
        template = get_persona_template(template_key)
        past = "; ".join(p.name for p in member.past_projects[:3]) or "정보 없음"
        return template.format(
            name=member.name,
            years=member.years_of_experience,
            skills=", ".join(member.primary_skills or member.tech_stack[:5]),
            strengths="; ".join(member.strengths[:3]),
            weaknesses="; ".join(member.weaknesses[:2]),
            past_projects=past,
            weaknesses_note="",
            strengths_note="",
        )

    @staticmethod
    def build_all_personas(team_members: List[MemberProfile]) -> Dict[str, str]:
        return {m.member_id: PersonaBuilder.build_persona(m) for m in team_members}
