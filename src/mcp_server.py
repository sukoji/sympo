"""
symPO MCP Server
━━━━━━━━━━━━━━━━━━
벡터 DB 검색, RAG 전략 실행, WBS 품질 평가를 외부 LLM 에이전트에 tool로 제공합니다.

실행:
  python mcp_server.py                          # stdio 모드 (Claude Code 연동)
  python mcp_server.py --transport sse --port 8001  # SSE 모드 (웹 클라이언트)

Claude Code 연동 (.claude/settings.local.json):
  {
    "mcpServers": {
      "sympo": {
        "command": "python",
        "args": ["/home/piai/ai_course/agent_test/mcp_server.py"]
      }
    }
  }
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("LLM_BACKEND", "mock")

from dotenv import load_dotenv
load_dotenv()

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "symPO WBS Agent",
    instructions="symPO 프로젝트의 벡터 DB, RAG 검색, WBS 품질 평가 도구를 제공합니다.",
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 벡터 DB 관련 tools
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_vs = None

def _get_vs():
    global _vs
    if _vs is None:
        from data_pipeline.vector_store import WBSVectorStore
        from data_pipeline.prd_parser import PRDParser
        from data_pipeline.member_parser import MemberParser
        from data_pipeline.disc_parser import load_all_disc_profiles

        _vs = WBSVectorStore()

        # 샘플 데이터 자동 인덱싱
        base = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_data")

        # PRD
        prd_path = os.path.join(base, "sample_prd.txt")
        if os.path.exists(prd_path):
            prd = PRDParser.from_text(open(prd_path, encoding="utf-8").read(), "P마켓")
            _vs.add_prd(prd)

        # 팀원
        member_dir = os.path.join(base, "sample_members")
        if os.path.isdir(member_dir):
            for f in sorted(os.listdir(member_dir)):
                if f.endswith(".txt"):
                    content = open(os.path.join(member_dir, f), encoding="utf-8").read()
                    name = f.replace("member_", "").replace(".txt", "")
                    m = MemberParser.from_resume_text(content, name)
                    _vs.add_member(m)

        # 회의록
        meeting_path = os.path.join(base, "sample_meeting_transcript.txt")
        if os.path.exists(meeting_path):
            _vs.add_meeting_log(open(meeting_path, encoding="utf-8").read())

        # eDISC
        disc = load_all_disc_profiles(base)
        for p in disc.values():
            _vs.add_disc_profile(p)

    return _vs


@mcp.tool()
def vector_search(query: str, doc_type: str = "", top_k: int = 5) -> str:
    """
    벡터 DB에서 의미 기반 검색을 수행합니다.

    Args:
        query: 검색 쿼리 (자연어)
        doc_type: 문서 유형 필터 (prd, member, reference_wbs, meeting_log, disc_profile). 비워두면 전체 검색.
        top_k: 반환할 결과 수

    Returns:
        검색 결과 JSON (content, metadata, score)
    """
    vs = _get_vs()
    if doc_type:
        results = vs.retrieve_by_type(query, doc_type, k=top_k)
    else:
        results = vs.retrieve_all_context(query, k=top_k)

    output = []
    for doc, score in results:
        output.append({
            "content": doc.page_content[:300],
            "metadata": doc.metadata,
            "score": round(float(score), 4),
        })
    return json.dumps(output, ensure_ascii=False, indent=2)


@mcp.tool()
def vector_stats() -> str:
    """벡터 DB의 현재 문서 수와 유형별 통계를 반환합니다."""
    vs = _get_vs()
    stats = vs.get_stats()
    return json.dumps(stats, ensure_ascii=False, indent=2)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RAG 전략 tools
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@mcp.tool()
def rag_retrieve(query: str, strategy: str = "vanilla", doc_type: str = "meeting_log", top_k: int = 3) -> str:
    """
    지정된 RAG 전략으로 문서를 검색합니다.

    Args:
        query: 검색 쿼리
        strategy: RAG 전략 (vanilla, hybrid, graph, agentic)
        doc_type: 문서 유형 (prd, member, reference_wbs, meeting_log, disc_profile)
        top_k: 반환할 결과 수

    Returns:
        검색 결과 JSON (content, metadata, strategy_info)
    """
    from data_pipeline.rag_strategies import get_strategy, STRATEGY_INFO

    vs = _get_vs()
    strat = get_strategy(strategy)
    common = dict(documents=vs._documents, vectorstore=vs._vectorstore, embeddings=vs._embeddings)
    results = strat.retrieve(query, doc_type, k=top_k, **common)

    return json.dumps({
        "strategy": strategy,
        "strategy_name": STRATEGY_INFO.get(strategy, {}).get("name", strategy),
        "results": [{"content": r["content"][:300], "metadata": {k: v for k, v in r.get("metadata", {}).items() if k != "_doc_idx"}} for r in results],
    }, ensure_ascii=False, indent=2)


@mcp.tool()
def rag_compare(query: str, doc_type: str = "meeting_log", top_k: int = 3) -> str:
    """
    4가지 RAG 전략(vanilla, hybrid, graph, agentic)으로 동일 쿼리를 검색하여 결과를 비교합니다.

    Args:
        query: 검색 쿼리
        doc_type: 문서 유형
        top_k: 전략당 반환 수
    """
    from data_pipeline.rag_strategies import get_strategy, STRATEGY_INFO

    vs = _get_vs()
    common = dict(documents=vs._documents, vectorstore=vs._vectorstore, embeddings=vs._embeddings)
    comparison = {}

    for key in ["vanilla", "hybrid", "graph", "agentic"]:
        strat = get_strategy(key)
        results = strat.retrieve(query, doc_type, k=top_k, **common)
        comparison[key] = {
            "name": STRATEGY_INFO.get(key, {}).get("name", key),
            "result_count": len(results),
            "top_result": results[0]["content"][:200] if results else "(없음)",
        }

    return json.dumps(comparison, ensure_ascii=False, indent=2)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 팀원/eDISC 프로필 tools
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@mcp.tool()
def list_disc_profiles() -> str:
    """로드된 eDISC 행동유형 프로필 전체 목록을 반환합니다."""
    from data_pipeline.disc_parser import load_all_disc_profiles
    base = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_data")
    profiles = load_all_disc_profiles(base)

    result = []
    for name, p in profiles.items():
        result.append({
            "name": name,
            "combo_code": p.combo_code,
            "primary_type": p.primary_type,
            "team_role": p.team_role,
            "team_role_en": p.team_role_en,
            "description": p.team_role_description[:100],
        })
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
def get_member_context(member_name: str) -> str:
    """
    특정 팀원의 벡터 DB 컨텍스트(이력서 + eDISC 프로필)를 통합 반환합니다.

    Args:
        member_name: 팀원 이름 (예: 진석호, 박선민)
    """
    vs = _get_vs()
    results = vs.retrieve_member_context(member_name, k=5)
    docs = []
    for doc, score in results:
        docs.append({
            "content": doc.page_content[:400],
            "type": doc.metadata.get("doc_type", ""),
            "score": round(float(score), 4),
        })
    return json.dumps(docs, ensure_ascii=False, indent=2)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# WBS 품질 평가 tools
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@mcp.tool()
def evaluate_wbs_snapshot(snapshot_path: str) -> str:
    """
    저장된 WBS 스냅샷 파일을 로드하여 13개 자동 지표를 계산합니다.

    Args:
        snapshot_path: WBS 스냅샷 JSON 파일 경로 (예: eval_results/wbs_snapshot_C3_3rounds_r1_gemini.json)
    """
    if not os.path.exists(snapshot_path):
        return json.dumps({"error": f"파일 없음: {snapshot_path}"})

    data = json.load(open(snapshot_path, encoding="utf-8"))
    tasks = data.get("wbs_tasks", [])
    debate = data.get("debate_log", [])

    # 간이 메트릭 (full compute_all_metrics는 WBSTask 객체 필요)
    l1 = [t for t in tasks if t.get("level") == "L1"]
    l2 = [t for t in tasks if t.get("level") == "L2"]
    l3 = [t for t in tasks if t.get("level") == "L3"]

    total_est = sum(t.get("estimated_days", 0) for t in l1)
    total_buf = sum(t.get("buffer_days", 0) for t in l1)
    buf_ratio = round(total_buf / max(total_est, 0.001) * 100, 2)

    in_range = sum(1 for t in l3 if 1 <= t.get("estimated_days", 0) <= 10)
    granularity = round(in_range / max(len(l3), 1), 4)

    return json.dumps({
        "total_tasks": len(tasks),
        "l1_count": len(l1),
        "l2_count": len(l2),
        "l3_count": len(l3),
        "buffer_ratio_pct": buf_ratio,
        "granularity_fitness": granularity,
        "debate_messages": len(debate),
        "llm_judge": data.get("llm_judge", {}),
    }, ensure_ascii=False, indent=2)


@mcp.tool()
def list_experiments() -> str:
    """eval_results 디렉토리의 실험 결과 파일 목록을 반환합니다."""
    result_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "experiments", "eval_results")
    if not os.path.isdir(result_dir):
        return json.dumps({"error": "eval_results 디렉토리 없음"})

    files = {}
    for f in sorted(os.listdir(result_dir)):
        if f.startswith("summary_") and f.endswith(".csv"):
            files.setdefault("summaries", []).append(f)
        elif f.startswith("wbs_snapshot_") and f.endswith(".json"):
            files.setdefault("snapshots", []).append(f)
        elif f.startswith("experiment_") and f.endswith(".json"):
            files.setdefault("experiments", []).append(f)
        elif f.endswith(".md"):
            files.setdefault("reports", []).append(f)
        elif f.endswith(".png"):
            files.setdefault("figures", []).append(f)

    return json.dumps(files, ensure_ascii=False, indent=2)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 오케스트레이션 / MCP tool-call 구조 tools
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@mcp.tool()
def orchestration_tool_catalog() -> str:
    """
    LangGraph phase가 사용하는 MCP-compatible tool catalog를 반환합니다.

    각 tool은 `server.tool` 형태의 stable name을 가지며, 현재 런타임은 같은
    이름으로 tool trace를 남깁니다. 외부 MCP client는 이 catalog를 기준으로
    어떤 기능을 별도 MCP server로 분리할지 판단할 수 있습니다.
    """
    from orchestration.mcp_tool_layer import tool_catalog_dicts

    return json.dumps(tool_catalog_dicts(), ensure_ascii=False, indent=2)


@mcp.tool()
def orchestration_phase_plan() -> str:
    """
    현재 WBS 생성 파이프라인의 LangGraph phase와 MCP tool 대응 관계를 반환합니다.
    """
    phases = [
        {
            "phase": "wbs_generation",
            "tools": ["wbs-server.generate_draft"],
            "state_updates": ["current_wbs_draft", "debate_log", "wbs_revision_needed"],
            "description": "PRD와 팀 프로필에서 WBS 초안을 만들거나 구조 재설계를 수행",
        },
        {
            "phase": "task_match",
            "tools": ["assignment-server.match_tasks"],
            "state_updates": [
                "assigned_tasks",
                "called_agents",
                "calling_context",
                "l2_candidate_pools",
                "l2_agent_mapping",
                "assignment_evidence",
            ],
            "description": "Task Manager가 L3 배정과 L2별 후보 풀을 결정",
        },
        {
            "phase": "l2_debate",
            "tools": ["debate-server.candidate_review", "debate-server.role_review"],
            "state_updates": ["debate_log"],
            "description": "L2 태스크별 후보/전문 에이전트 검토 발언을 누적",
        },
        {
            "phase": "free_discussion",
            "tools": ["debate-server.free_discussion"],
            "state_updates": ["debate_log"],
            "description": "후보 간 상호 검토와 PASS 기반 조기 수렴",
        },
        {
            "phase": "supervisor_mediate",
            "tools": ["supervisor-server.mediate"],
            "state_updates": [
                "current_wbs_draft",
                "current_round",
                "consensus_reached",
                "wbs_revision_needed",
                "wbs_revision_hints",
                "debate_log",
            ],
            "description": "PM이 갈등을 중재하고 다음 phase를 결정",
        },
        {
            "phase": "finalize",
            "tools": ["supervisor-server.finalize"],
            "state_updates": ["final_wbs", "generation_summary", "debate_log"],
            "description": "최종 WBS와 생성 요약을 확정",
        },
    ]
    return json.dumps(phases, ensure_ascii=False, indent=2)


@mcp.tool()
def inspect_wbs_snapshot_state(snapshot_path: str) -> str:
    """
    저장된 snapshot을 MCP/LangGraph state 관점에서 요약합니다.

    Args:
        snapshot_path: WBS snapshot JSON 파일 경로
    """
    if not os.path.exists(snapshot_path):
        return json.dumps({"error": f"파일 없음: {snapshot_path}"}, ensure_ascii=False)

    data = json.load(open(snapshot_path, encoding="utf-8"))
    tasks = data.get("wbs_tasks") or data.get("final_wbs") or []
    debate = data.get("debate_log", [])
    tool_trace = data.get("mcp_tool_trace", [])
    levels = {"L1": 0, "L2": 0, "L3": 0}
    for task in tasks:
        level = task.get("level")
        if isinstance(level, dict):
            level = level.get("value")
        if level in levels:
            levels[level] += 1
    assigned = sum(1 for task in tasks if task.get("assigned_to") or task.get("assignee"))
    return json.dumps(
        {
            "snapshot_path": snapshot_path,
            "task_count": len(tasks),
            "level_counts": levels,
            "assigned_task_count": assigned,
            "debate_message_count": len(debate),
            "mcp_tool_call_count": len(tool_trace),
            "last_tool_calls": tool_trace[-5:],
            "has_llm_judge": bool(data.get("llm_judge")),
        },
        ensure_ascii=False,
        indent=2,
    )


@mcp.tool()
def compact_snapshot_debate_history(snapshot_path: str, max_messages: int = 8, pm_decisions: int = 4) -> str:
    """
    저장된 snapshot의 debate_log를 현재 runtime compaction 정책과 같은 방식으로 축약합니다.

    Args:
        snapshot_path: WBS snapshot JSON 파일 경로
        max_messages: 최근 일반 발화 유지 개수
        pm_decisions: PM/Task Manager decision 유지 개수
    """
    if not os.path.exists(snapshot_path):
        return json.dumps({"error": f"파일 없음: {snapshot_path}"}, ensure_ascii=False)

    data = json.load(open(snapshot_path, encoding="utf-8"))
    logs = data.get("debate_log", [])

    def _is_pm_decision(item):
        role = str(item.get("agent_role", "")).lower()
        name = str(item.get("agent_name", "")).lower()
        msg_type = str(item.get("message_type", "")).lower()
        return (
            msg_type == "decision"
            or "supervisor" in role
            or "supervisor" in name
            or "task manager" in name
            or "pm" in name
            or "슈퍼" in name
        )

    pm_logs = [item for item in logs if _is_pm_decision(item)][-max(pm_decisions, 0):]
    recent_logs = logs[-max(max_messages, 0):]
    selected = []
    seen = set()
    for item in pm_logs + recent_logs:
        key = (
            item.get("timestamp"),
            item.get("agent_name"),
            item.get("message"),
            item.get("related_task_id"),
        )
        if key in seen:
            continue
        seen.add(key)
        selected.append(item)

    return json.dumps(
        {
            "snapshot_path": snapshot_path,
            "source_messages": len(logs),
            "selected_messages": len(selected),
            "policy": {
                "recent_messages": max_messages,
                "pm_decisions": pm_decisions,
            },
            "messages": selected,
        },
        ensure_ascii=False,
        indent=2,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Resources (정적 데이터 노출)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@mcp.resource("sympo://eval-framework")
def get_eval_framework() -> str:
    """평가 프레임워크 전체 문서를 반환합니다."""
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "experiments", "eval_results", "EVALUATION_FRAMEWORK.md")
    if os.path.exists(path):
        return open(path, encoding="utf-8").read()
    return "(EVALUATION_FRAMEWORK.md 없음)"


@mcp.resource("sympo://sample-prd")
def get_sample_prd() -> str:
    """샘플 PRD 텍스트를 반환합니다."""
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_data", "sample_prd.txt")
    if os.path.exists(path):
        return open(path, encoding="utf-8").read()
    return "(sample_prd.txt 없음)"


@mcp.resource("sympo://meeting-transcript")
def get_meeting_transcript() -> str:
    """샘플 회의록 텍스트를 반환합니다."""
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_data", "sample_meeting_transcript.txt")
    if os.path.exists(path):
        return open(path, encoding="utf-8").read()
    return "(meeting transcript 없음)"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", default="stdio", choices=["stdio", "sse"])
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    if args.transport == "sse":
        mcp.run(transport="sse", port=args.port)
    else:
        mcp.run(transport="stdio")
