from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse
import asyncio
import json
import os
import shutil
import tempfile

from pydantic import BaseModel
from typing import List, Optional

# symPO Imports
from data_pipeline.prd_parser import PRDParser
from data_pipeline.member_parser import MemberParser
from persona_engine.persona_builder import PersonaBuilder
from agents.state import create_initial_state
from orchestration.debate_loop import execute_sympo_flow
import os
from dotenv import load_dotenv

load_dotenv()

# ── eDISC 프로파일 사전 로드 ───────────────────────────────────
_disc_profiles: dict = {}  # {이름: DiscProfile}
_disc_agent_contexts: dict = {}  # {이름: to_agent_context() 문자열}
USER_DISC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_data", "user_edisc")
USER_DISC_MODE_FILE = os.path.join(USER_DISC_DIR, ".mode")

def _load_disc_profiles():
    global _disc_profiles, _disc_agent_contexts
    try:
        from data_pipeline.disc_parser import load_all_disc_profiles
        base = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_data")
        mode = "add"
        try:
            if os.path.exists(USER_DISC_MODE_FILE):
                mode = open(USER_DISC_MODE_FILE, encoding="utf-8").read().strip() or "add"
        except Exception:
            mode = "add"
        profiles = {} if mode == "replace" else load_all_disc_profiles(base)
        user_profiles = load_all_disc_profiles(USER_DISC_DIR)
        profiles.update(user_profiles)
        _disc_profiles = profiles
        _disc_agent_contexts = {name: p.to_agent_context() for name, p in profiles.items()}
        if profiles:
            print(f"[eDISC] {len(profiles)}명 프로파일 로드 완료: {list(profiles.keys())}")
    except Exception as e:
        print(f"[eDISC] 프로파일 로드 실패 (무시): {e}")

_load_disc_profiles()

def _disc_status_payload():
    profiles_info = []
    for name, p in _disc_profiles.items():
        profiles_info.append({
            "name": name,
            "disc_style": p.disc_style,
            "primary_type": p.primary_type,
            "type_code": p.type_code,
            "combo_code": getattr(p, "combo_code", ""),
            "team_role": p.team_role,
            "team_role_en": getattr(p, "team_role_en", ""),
        })
    return {
        "loaded": len(_disc_profiles),
        "profiles": profiles_info,
        "vector_doc_type": "disc_profile",
    }

app = FastAPI(title="symPO API", description="AI Agent WBS Generation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CLAUDE_DESIGN_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "experiments", "claude_design")
if os.path.isdir(CLAUDE_DESIGN_DIR):
    app.mount("/claude-design-static", StaticFiles(directory=CLAUDE_DESIGN_DIR), name="claude-design-static")

class MemberInput(BaseModel):
    name: str
    role: str
    resume: Optional[str] = None
    tech: Optional[str] = None
    strengths: Optional[str] = None
    weaknesses: Optional[str] = None
    # 추가 필드 (LLM 파싱 데이터 보존용)
    years_of_experience: Optional[float] = 2.0
    tech_stack: Optional[List[str]] = []
    primary_skills: Optional[List[str]] = []
    personality_traits: Optional[List[str]] = []
    past_projects: Optional[List[dict]] = []
    raw_resume_text: Optional[str] = None

class GenerationRequest(BaseModel):
    project_name: str
    project_goal: str
    target_users: str
    scope: str
    key_features: str
    tech_stack: str
    deadline: str
    budget_weeks: int
    team_size: int
    constraints: str
    members_data: List[MemberInput]
    ref_wbs_text: str = ""
    meeting_text: str = ""
    min_rounds: int = 2
    max_rounds: int = 5
    prd_raw_text: Optional[str] = None
    llm_backend: Optional[str] = None  # web UI에서 LLM 백엔드 선택값
    rag_strategy: str = "vanilla"       # RAG 검색 전략: vanilla|hybrid|graph|agentic

@app.get("/")
async def serve_frontend():
    """React-like 단일 페이지 웹 UI 서빙"""
    frontend_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "index.html")
    return FileResponse(frontend_path)

@app.get("/claude-design")
async def serve_claude_design():
    """Claude design shell with the same live FastAPI functionality."""
    return FileResponse(os.path.join(CLAUDE_DESIGN_DIR, "SYMPO System Demo.html"))

@app.get("/api/disc-status")
async def get_disc_status():
    """로드된 eDISC 프로파일 목록 반환"""
    return JSONResponse(_disc_status_payload())

@app.post("/api/disc-upload")
async def upload_disc_profiles(
    files: List[UploadFile] = File(...),
    mode: str = Form("add"),
):
    """eDISC PDF 업로드. mode=add는 기존 프로필에 병합, mode=replace는 업로드 프로필로 교체."""
    global _disc_profiles, _disc_agent_contexts

    normalized_mode = (mode or "add").strip().lower()
    if normalized_mode not in {"add", "replace"}:
        return JSONResponse({"error": "mode must be add or replace"}, status_code=400)

    os.makedirs(USER_DISC_DIR, exist_ok=True)

    from data_pipeline.disc_parser import parse_disc_pdf

    parsed = {}
    target_names = set()
    failed = []
    for upload in files:
        original_name = upload.filename or "uploaded_edisc.pdf"
        if not original_name.lower().endswith(".pdf"):
            failed.append({"file": original_name, "error": "PDF 파일만 지원"})
            continue

        safe_name = os.path.basename(original_name).replace(" ", "_")
        if not safe_name.lower().startswith("edisc_"):
            safe_name = "eDISC_" + safe_name

        tmp_dir = None
        tmp_path = None
        try:
            tmp_dir = tempfile.mkdtemp(prefix="sympo_edisc_")
            tmp_path = os.path.join(tmp_dir, safe_name)
            with open(tmp_path, "wb") as tmp:
                shutil.copyfileobj(upload.file, tmp)
            profile = parse_disc_pdf(tmp_path)
            if not profile:
                failed.append({"file": original_name, "error": "eDISC 프로파일 파싱 실패"})
                continue

            target_name = f"eDISC_{profile.name}.pdf"
            target_path = os.path.join(USER_DISC_DIR, target_name)
            shutil.copyfile(tmp_path, target_path)
            target_names.add(target_name)
            parsed[profile.name] = profile
        except Exception as e:
            failed.append({"file": original_name, "error": str(e)})
        finally:
            try:
                if tmp_dir and os.path.isdir(tmp_dir):
                    shutil.rmtree(tmp_dir)
            except OSError:
                pass
            try:
                await upload.close()
            except Exception:
                pass

    if not parsed:
        return JSONResponse({
            "error": "업로드된 eDISC PDF를 파싱하지 못했습니다.",
            "failed": failed,
            **_disc_status_payload(),
        }, status_code=400)

    if normalized_mode == "replace":
        for fname in os.listdir(USER_DISC_DIR):
            if fname.lower().endswith(".pdf") and fname not in target_names:
                try:
                    os.remove(os.path.join(USER_DISC_DIR, fname))
                except OSError:
                    pass
        _disc_profiles = parsed
    else:
        _disc_profiles.update(parsed)
    _disc_agent_contexts = {name: p.to_agent_context() for name, p in _disc_profiles.items()}
    try:
        os.makedirs(USER_DISC_DIR, exist_ok=True)
        with open(USER_DISC_MODE_FILE, "w", encoding="utf-8") as f:
            f.write(normalized_mode)
    except Exception:
        pass

    return JSONResponse({
        **_disc_status_payload(),
        "mode": normalized_mode,
        "added": len(parsed),
        "failed": failed,
        "message": f"eDISC {len(parsed)}개를 {'교체' if normalized_mode == 'replace' else '추가'}했습니다. 다음 WBS 생성부터 vector store에 반영됩니다.",
    })

@app.get("/api/sample")
async def get_sample_data():
    """샘플 PRD 및 팀원 데이터 반환"""
    base = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_data")
    members = []
    member_dir = os.path.join(base, "sample_members")
    if os.path.isdir(member_dir):
        from data_pipeline.member_parser import MemberParser
        for fname in sorted(os.listdir(member_dir)):
            if fname.endswith(".txt"):
                try:
                    content = open(os.path.join(member_dir, fname), encoding="utf-8").read()
                    name = fname.replace("member_", "").replace(".txt", "")
                    p = MemberParser.from_resume_text(content, name)
                    members.append({
                        "name": p.name,
                        # role은 의도적으로 제외 — supervisor가 기술·강점만 보고 R&R을 독립 결정하도록
                        "role": "",
                        "tech": ", ".join(p.tech_stack[:5]),
                        "strengths": " / ".join(p.strengths[:2]) if isinstance(p.strengths, list) else str(p.strengths),
                        "weaknesses": " / ".join(p.weaknesses[:2]) if isinstance(p.weaknesses, list) else str(p.weaknesses),
                    })
                except Exception:
                    pass

    prd_path = os.path.join(base, "sample_prd.txt")
    prd_text = open(prd_path, encoding="utf-8").read() if os.path.exists(prd_path) else ""

    from data_pipeline.prd_parser import PRDParser
    try:
        prd = PRDParser.from_text(prd_text, "샘플 프로젝트")
        return JSONResponse({
            "project_name": prd.project_name,
            "project_goal": prd.project_goal,
            "key_features": prd.key_features[:6],
            "tech_stack": prd.tech_stack_requirements[:6],
            "deadline": str(prd.deadline) if prd.deadline else "",
            "budget_weeks": prd.budget_weeks or 12,
            "constraints": prd.special_constraints or [],
            "members": members,
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/api/sample/pmarket")
async def get_pmarket_sample():
    """P마켓 샘플 PRD 데이터 반환"""
    return JSONResponse({
        "project_name": "P마켓 빅데이터 기반 매출 증대 전략",
        "project_goal": "2025년 판매·고객 데이터 빅데이터 분석으로 고객 세그먼트별 맞춤 프로모션을 실행하고, 모바일 앱·매장 진열 개선을 통해 전년 대비 매출 20% 이상 회복",
        "target_users": "수지지역 P마켓 방문 고객 (학부모, 노인·실버, 직장인, 학생)",
        "scope": "포함: 판매·고객 데이터 분석, 맞춤형 프로모션 기획·실행, 모바일 앱 기능 개선, 매장 진열 리디자인 / 제외: 신규 매장 개점, 공급업체 교체",
        "key_features": ["소매업 트렌드·소비자 수요 조사 및 데이터 분석", "고객 특성(세그먼트)별 맞춤형 서비스·프로모션 설계", "모바일 알림·배송·커뮤니티 서비스 고도화", "매장 실내 진열 및 고객 동선 리디자인"],
        "tech_stack": ["Python/Pandas (데이터 분석)", "Tableau / Power BI (시각화)", "MySQL (판매 데이터)", "모바일 앱 (Android/iOS)", "CRM 플랫폼"],
        "deadline": "2025-12-31",
        "budget_weeks": 24,
        "constraints": ["개인정보보호법 준수 (고객 데이터 익명화 필수)", "기존 POS·재고 시스템과 데이터 연동", "예산 범위 내 외부 광고비 최소화", "파일럿 프로모션은 1개 매장 우선 적용 후 전점 확대"],
        "members": [
            {"name": "김데이터", "role": "Data Analyst", "tech": "Python, Pandas, Tableau, SQL", "strengths": "빅데이터 분석, 고객 세그먼테이션", "weaknesses": "모바일 개발 경험 부족"},
            {"name": "이마케터", "role": "Marketing Planner", "tech": "CRM, Google Analytics, SNS 마케팅", "strengths": "프로모션 기획, 고객 커뮤니케이션", "weaknesses": "데이터 분석 심화 부족"},
            {"name": "박기획", "role": "Business Analyst", "tech": "Excel, PowerPoint, Figma, 리서치", "strengths": "사업 전략, 고객 리서치, 프레젠테이션", "weaknesses": "기술 구현 이해 부족"},
            {"name": "최모바일", "role": "Mobile Developer", "tech": "React Native, Android, iOS, Firebase", "strengths": "모바일 앱 개발, 푸시 알림 시스템", "weaknesses": "백엔드 설계 경험 부족"},
            {"name": "정운영", "role": "Operations Manager", "tech": "ERP, WMS, 재고관리, POS", "strengths": "매장 운영, 공급망 관리, 직원 교육", "weaknesses": "디지털 기술 적응 느림"},
        ],
    })


@app.post("/api/wbs/generate")
async def generate_wbs(req: GenerationRequest):
    # 이 엔드포인트는 상태 모델을 세팅하고 SSE에서 스트리밍할 수 있도록 준비하는 역할을 합니다.
    # 단순화를 위해 스트리밍 자체를 SSE 엔드포인트에 통합합니다.
    return {"status": "ready", "message": "Use /api/wbs/stream to start generation"}

async def yield_state_updates(req: GenerationRequest):
    # 웹 UI에서 선택한 LLM 백엔드 적용
    if req.llm_backend:
        os.environ["LLM_BACKEND"] = req.llm_backend

    # Parse PRD
    if req.prd_raw_text:
        prd = PRDParser.from_text(req.prd_raw_text, req.project_name)
    else:
        prd = PRDParser.from_form(
            project_name=req.project_name or "미정",
            project_goal=req.project_goal or "미정",
            target_users=req.target_users or "미정",
            scope=req.scope or "전체",
            key_features_text=req.key_features or "기능 미정",
            tech_stack_text=req.tech_stack or "",
            deadline=req.deadline or None,
            team_size=req.team_size,
            budget_weeks=int(req.budget_weeks),
            constraints_text=req.constraints or "",
        )
    
    # Parse Team
    team_members = []
    from schemas.member_schema import MemberProfile
    import uuid
    for m in req.members_data:
        role_enum = MemberParser._map_role(m.role)
        # 콤마 구분자 처리 (사용자가 UI에서 수정한 경우 대비)
        tech_list = [t.strip() for t in (m.tech or "").split(",") if t.strip()]
        if not tech_list and m.tech_stack: tech_list = m.tech_stack # pydantic은 리스트 직접 매칭 안 될 수 있으므로 주의

        str_list = [s.strip() for s in (m.strengths or "").split(" / ") if s.strip()]
        if not str_list: str_list = [s.strip() for s in (m.strengths or "").split(",") if s.strip()]
        
        weak_list = [w.strip() for w in (m.weaknesses or "").split(" / ") if w.strip()]
        if not weak_list: weak_list = [w.strip() for w in (m.weaknesses or "").split(",") if w.strip()]

        member = MemberProfile(
            member_id=f"MBR-{uuid.uuid4().hex[:4].upper()}",
            name=m.name or "팀원",
            role=role_enum,
            years_of_experience=m.years_of_experience if m.years_of_experience is not None else 2.0,
            tech_stack=tech_list or ["미정"],
            primary_skills=m.primary_skills if m.primary_skills else tech_list[:3] or ["미정"],
            strengths=str_list or ["성실함"],
            weaknesses=weak_list or ["미정"],
            personality_traits=m.personality_traits if m.personality_traits else [],
            past_projects=m.past_projects if m.past_projects else [],
            raw_resume_text=m.raw_resume_text or m.resume or ""
        )
        team_members.append(member)
            
    # Set default member if empty
    from schemas.member_schema import MemberRole
    if not team_members:
        team_members = [
            MemberProfile(
                member_id="MBR-001", name="BE 개발자", role=MemberRole.BACKEND, years_of_experience=4.0,
                tech_stack=["Python", "FastAPI"], primary_skills=["API 개발"], strengths=["체계적"],
                weaknesses=["프론트엔드 취약"], availability_percent=100
            )
        ]

    team_summary = PersonaBuilder.generate_team_summary(team_members)
    supervisor_persona = PersonaBuilder.build_supervisor_persona("PM 에이전트", team_summary)
    member_personas = PersonaBuilder.build_all_personas(team_members)
    all_personas = {"supervisor": supervisor_persona, **member_personas}

    # ── RAG 벡터 DB 구축 및 전략 기반 검색 ──────────────────────────
    from data_pipeline.vector_store import WBSVectorStore
    from data_pipeline.rag_strategies import get_strategy, STRATEGY_INFO
    _rag_strategy_key = req.rag_strategy or "vanilla"
    _rag_strategy = get_strategy(_rag_strategy_key)

    _vs = WBSVectorStore()
    _vs.add_prd(prd)
    for _m in team_members:
        _vs.add_member(_m)
    if req.ref_wbs_text:
        _vs.add_reference_wbs(req.ref_wbs_text, prd.project_name)
    if req.meeting_text:
        _vs.add_meeting_log(req.meeting_text)
    for _dp in _disc_profiles.values():
        _vs.add_disc_profile(_dp)

    _common_kw = dict(documents=_vs._documents, vectorstore=_vs._vectorstore, embeddings=_vs._embeddings)
    _rag_wbs      = _rag_strategy.retrieve(f"{prd.project_name} WBS 일정", "reference_wbs", k=3, **_common_kw)
    _rag_meetings = _rag_strategy.retrieve("일정 버퍼 교훈", "meeting_log", k=3, **_common_kw)
    rag_wbs_texts      = [d["content"] for d in _rag_wbs]
    rag_meeting_texts  = [d["content"] for d in _rag_meetings]

    # RAG 메타데이터 (SSE 스트림으로 전송)
    def _safe(obj):
        """JSON 직렬화 불가 값 제거"""
        if isinstance(obj, dict):
            return {k: _safe(v) for k, v in obj.items() if not k.startswith("_doc_idx")}
        if isinstance(obj, list):
            return [_safe(i) for i in obj]
        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        return str(obj)

    _rag_metadata = {
        "strategy_key":  _rag_strategy_key,
        "strategy_name": STRATEGY_INFO.get(_rag_strategy_key, {}).get("name", _rag_strategy_key),
        "strategy_icon": STRATEGY_INFO.get(_rag_strategy_key, {}).get("icon", "🔍"),
        "strategy_detail": STRATEGY_INFO.get(_rag_strategy_key, {}).get("detail", ""),
        "wbs_results":     [_safe(d) for d in _rag_wbs],
        "meeting_results": [_safe(d) for d in _rag_meetings],
        "doc_stats":       _vs.get_stats(),
    }
    # ──────────────────────────────────────────────────────────────────

    initial_state = create_initial_state(
        prd=prd,
        team_members=team_members,
        agent_personas=all_personas,
        min_rounds=req.min_rounds,
        max_rounds=req.max_rounds,
    )
    initial_state["rag_reference_wbs"] = rag_wbs_texts
    initial_state["rag_meeting_logs"]  = rag_meeting_texts
    initial_state["disc_profiles"] = _disc_agent_contexts
    
    final_state = None
    try:
        for current_state in execute_sympo_flow(initial_state, req.max_rounds):
            final_state = current_state

            # WBS 직렬화 (ID 대신 이름 매핑 포함)
            member_name_map = {m.member_id: m.name for m in team_members}
            wbs_draft = []
            for t in current_state.get("current_wbs_draft", []):
                # ID 목록을 이름 목록으로 변환
                assignee_ids = getattr(t, 'assigned_to', [])
                if isinstance(assignee_ids, str): assignee_ids = [assignee_ids]
                assignee_names = [member_name_map.get(mid, mid) for mid in assignee_ids if mid]

                wbs_draft.append({
                    "task_id": t.task_id,
                    "title": t.title,
                    "level": t.level.value if hasattr(t.level, 'value') else t.level,
                    "assigned_role": getattr(t, 'assigned_role', ''),
                    "required_role": getattr(t, 'required_role', ''),
                    "assigned_to": assignee_names,  # 이름 목록으로 전송
                    "parent_id": getattr(t, 'parent_id', None),
                    "estimated_days": getattr(t, 'estimated_days', 0),
                    "buffer_days": getattr(t, 'buffer_days', 0),
                    "total_days": getattr(t, 'total_days', 0),
                    "start_week": getattr(t, 'start_week', None),
                    "end_week": getattr(t, 'end_week', None),
                    "dependencies": getattr(t, 'dependencies', []),
                    "importance": getattr(t, 'importance', 'Medium'),
                    "risk_factors": getattr(t, 'risk_factors', []),
                    "deliverables": getattr(t, 'deliverables', []),
                })

            debate_log = []
            for m in current_state.get("debate_log", []):
                debate_log.append({
                    "timestamp": m.timestamp,
                    "agent_name": m.agent_name,
                    "agent_role": str(getattr(m.agent_role, 'value', m.agent_role)),
                    "message": m.message,
                })

            def _map_candidate_pool_names(pool):
                mapped = {}
                for task_id, member_ids in (pool or {}).items():
                    if isinstance(member_ids, str):
                        member_ids = [member_ids]
                    mapped[task_id] = [member_name_map.get(mid, mid) for mid in (member_ids or []) if mid]
                return mapped

            payload = {
                "acting_agent": current_state.get("_current_agent_acting"),
                "current_round": current_state.get("current_round", 1),
                "wbs_draft": wbs_draft,
                "debate_log": debate_log,
                "mcp_tool_trace": current_state.get("mcp_tool_trace", []),
                "consensus_reached": current_state.get("consensus_reached", False),
                "called_agents": current_state.get("called_agents", []),
                "l2_candidate_pools": current_state.get("l2_candidate_pools", {}),
                "task_candidate_pools": current_state.get("task_candidate_pools", {}),
                "l2_candidate_pool_names": _map_candidate_pool_names(current_state.get("l2_candidate_pools", {})),
                "task_candidate_pool_names": _map_candidate_pool_names(current_state.get("task_candidate_pools", {})),
                "l2_review_lenses": current_state.get("l2_review_lenses", {}),
                "l2_calling_context": current_state.get("l2_calling_context", {}),
                "assignment_evidence": current_state.get("assignment_evidence", {}),
                "member_project_roles": current_state.get("member_project_roles", []),
                "wbs_revision_needed": current_state.get("wbs_revision_needed", False),
                "current_wbs_revision": current_state.get("current_wbs_revision", 0),
                "wbs_mediate_branch_round": current_state.get("_wbs_mediate_branch_round"),
            }

            yield json.dumps(payload, ensure_ascii=False)
            await asyncio.sleep(0.01)

        # ── 최종 평가 지표 계산 + 파일 저장 + SSE 전송 ──
        if final_state:
            try:
                from metrics import compute_all_metrics

                experiment_config = {
                    "rag_strategy": req.rag_strategy,
                    "min_rounds": req.min_rounds,
                    "max_rounds": req.max_rounds,
                    "team_size": req.team_size,
                    "budget_weeks": req.budget_weeks,
                    "llm_backend": req.llm_backend or os.environ.get("LLM_BACKEND", "unknown"),
                    "note": "",  # 프론트엔드에서 전달 가능
                }

                full_metrics = compute_all_metrics(
                    final_state=final_state,
                    prd=prd,
                    team_members=team_members,
                    experiment_config=experiment_config,
                )

                # SSE로 전송할 때는 프론트엔드에서 쓰는 형태로 변환
                metrics_for_sse = {
                    "ragas_faithfulness":      full_metrics.get("ragas_faithfulness"),
                    "interaction_turns":       full_metrics.get("interaction_turns"),
                    "supervisor_intervention": full_metrics.get("supervisor_intervention"),
                    "success_rate":            full_metrics.get("success_rate"),
                    "planning_score":          full_metrics.get("planning_score"),
                    "buffer_ratio":            full_metrics.get("buffer_ratio"),
                    "convergence":             full_metrics.get("convergence"),
                    "mece_score":              full_metrics.get("mece_score"),
                    "granularity_fitness":     full_metrics.get("granularity_fitness"),
                    "workload_gini":           full_metrics.get("workload_gini"),
                    "schedule_feasibility":    full_metrics.get("schedule_feasibility"),
                    "communication_efficiency":full_metrics.get("communication_efficiency"),
                    "token_cost":              full_metrics.get("token_cost"),
                    "autoscore":               full_metrics.get("autoscore"),
                }
                yield json.dumps({"metrics": metrics_for_sse}, ensure_ascii=False)

                # ── LLM-as-Judge 평가 (상위 LLM이 구조/배분/토론 3차원 채점) ──
                if os.getenv("ENABLE_LLM_JUDGE", "true").lower() in ("true", "1", "yes", "on"):
                    yield json.dumps({"llm_judge_status": "pending"}, ensure_ascii=False)
                    await asyncio.sleep(0.01)
                    try:
                        from eval.llm_judge import evaluate_wbs
                        judge_result = evaluate_wbs(
                            wbs_tasks=final_state.get("final_wbs") or final_state.get("current_wbs_draft") or [],
                            team_members=team_members,
                            debate_log=final_state.get("debate_log") or [],
                            eval_dims=["structure", "assignment", "debate"],
                            cross_judge=False,
                        )
                        yield json.dumps({"llm_judge": judge_result}, ensure_ascii=False)
                    except Exception as je:
                        print(f"[WARN] LLM Judge 평가 실패: {je}")
                        yield json.dumps({"llm_judge": {"error": str(je), "overall": -1}}, ensure_ascii=False)
            except Exception as e:
                print(f"[WARN] 평가 지표 계산/저장 실패: {e}")

            # ── RAG 메타데이터 스트리밍 ──
            try:
                yield json.dumps({"rag_metadata": _rag_metadata}, ensure_ascii=False)
            except Exception:
                pass

    except Exception as e:
        yield json.dumps({"error": str(e)})

@app.post("/api/wbs/stream")
async def wbs_stream(req: GenerationRequest, request: Request):
    return EventSourceResponse(yield_state_updates(req))


# ── 이력서 파싱 엔드포인트 ─────────────────────────────────────
class ResumeParseRequest(BaseModel):
    filename: str
    content: str        # 텍스트(.txt/.md) 또는 PDF base64 인코딩 문자열
    is_pdf: bool = False
    llm_backend: Optional[str] = None

def _extract_pdf_text(b64_data: str) -> str:
    """PDF 데이터에서 텍스트 추출 및 정제 (중복 라인 제거 및 파편화된 텍스트 결합)"""
    import base64, io
    from pypdf import PdfReader
    try:
        raw = base64.b64decode(b64_data)
        reader = PdfReader(io.BytesIO(raw))
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                # 1. 라인별로 나누고 공백 제거
                lines = [line.strip() for line in extracted.splitlines()]
                # 2. 중복 라인 제거 (eDISC PDF 등에서 발생하는 중복 텍스트 처리)
                seen = []
                for line in lines:
                    if line and (not seen or line != seen[-1]):
                        seen.append(line)
                # 3. 아주 짧은 라인이 연속되면 합쳐서 단어 복원 시도
                cleaned_lines = []
                buffer = ""
                for line in seen:
                    if len(line) <= 1: # 한 글자 단위 분절 처리
                        buffer += line
                    else:
                        if buffer:
                            cleaned_lines.append(buffer)
                            buffer = ""
                        cleaned_lines.append(line)
                if buffer: cleaned_lines.append(buffer)
                text += "\n".join(cleaned_lines) + "\n"
        return text
    except Exception as e:
        print(f"[ERROR] PDF 텍스트 추출 실패: {e}")
        return ""

def _clean_member_name(filename: str) -> str:
    """파일명에서 이름만 추출 (개선된 폴백 처리)"""
    import re
    name = os.path.splitext(filename)[0]
    name = re.sub(r'^(이력서|resume|cv|portfolio|포트폴리오|eDISC)[_ \-]*', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s*[\(\[].*?[\)\]]', '', name)
    name = name.replace('_', ' ').replace('-', ' ').strip()
    
    if not name:
        # "이력서.pdf" 같이 이름이 다 날아간 경우 파일명 원본에서 확장자만 뗀 것 사용
        orig_name = os.path.splitext(filename)[0]
        return orig_name if orig_name else "알 수 없는 팀원"
    return name

@app.post("/api/parse-resume")
async def parse_resume(req: ResumeParseRequest):
    """이력서(텍스트 또는 PDF) 파싱 — main.py 이력서 업로드 로직과 동일"""
    _prev_backend = os.environ.get("LLM_BACKEND")
    try:
        import re

        # 웹 UI에서 선택한 LLM 백엔드 적용 (호출 범위 내에서만 유효)
        if req.llm_backend:
            os.environ["LLM_BACKEND"] = req.llm_backend

        # ① 파일명에서 초기 이름 추출 (Streamlit의 _clean_member_name)
        cleaned_name = _clean_member_name(req.filename)

        # ② PDF면 텍스트 추출, 아니면 그대로 사용
        if req.is_pdf:
            content = _extract_pdf_text(req.content)
        else:
            content = req.content

        if not content.strip():
            return JSONResponse({"error": "텍스트를 추출할 수 없습니다."}, status_code=422)

        # ③ LLM 기반 파싱 (Streamlit의 MemberParser.from_resume_text_llm)
        profile = MemberParser.from_resume_text_llm(content, name=cleaned_name)

        # ④ 콘텐츠에서 이름 재추출 시도 (Streamlit과 동일한 패턴)
        name_match = re.search(
            r'(?:이름|성명|Name|이 름)\s*[:\s]\s*([가-힣a-zA-Z ]{2,10})', content
        )
        if name_match:
            profile.name = name_match.group(1).strip()
        else:
            lines = content.splitlines()
            name_line = next((l for l in lines[:10] if "이름:" in l), None)
            if name_line:
                profile.name = name_line.replace("이름:", "").strip()
            else:
                # LLM이 '미정'을 반환했거나 추출 실패 시 파일명에서 가져온 이름 사용
                if profile.name == "미정":
                    profile.name = cleaned_name

        return JSONResponse(profile.dict())
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        # 백엔드 env 원복 (다른 요청에 영향 주지 않도록)
        if req.llm_backend:
            if _prev_backend is None:
                os.environ.pop("LLM_BACKEND", None)
            else:
                os.environ["LLM_BACKEND"] = _prev_backend


# ── 샘플 회의록 엔드포인트 ────────────────────────────────────
@app.get("/api/sample/meeting")
async def get_sample_meeting():
    """샘플 회의록 텍스트 반환"""
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_data", "sample_meeting_transcript.txt")
    if not os.path.exists(path):
        return JSONResponse({"error": "샘플 회의록 파일을 찾을 수 없습니다."}, status_code=404)
    text = open(path, encoding="utf-8").read()
    return JSONResponse({"text": text})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
