"""
symPO LangGraph DAG 빌더.

실행 로직은 `orchestration.debate_loop`의 phase 함수들을 공유한다.
LangGraph 경로와 sequential fallback 경로가 같은 phase 구현을 쓰도록 유지해
기존 실험/평가 동작과 UI 스트리밍 단위를 보존한다.
"""
try:
    from langgraph.graph import StateGraph, END, START
    from langgraph.config import get_stream_writer
    HAS_LANGGRAPH = True
except ImportError:
    StateGraph = object
    END = "__end__"
    START = "__start__"
    HAS_LANGGRAPH = False
    def get_stream_writer():
        return lambda _: None

from agents.state import WBSState
from orchestration.mcp_tool_layer import STATE_TRACE_KEY
from orchestration.debate_loop import (
    ensure_runtime_state,
    phase_critic_review,
    phase_finalize,
    phase_free_discussion,
    phase_l2_debate,
    phase_supervisor_mediate,
    phase_task_match,
    phase_wbs_generation,
    route_after_mediate,
    route_after_wbs_generation,
)


def _run_phase_node(state: WBSState, phase_fn) -> WBSState:
    """Phase generator의 중간 state를 LangGraph custom stream으로 포워딩한다."""
    try:
        writer = get_stream_writer()
    except Exception:
        writer = lambda _: None

    final_state = ensure_runtime_state(state)
    initial_debate_len = len(final_state.get("debate_log", []) or [])
    initial_tool_len = len(final_state.get(STATE_TRACE_KEY, []) or [])
    phase_iter = phase_fn(final_state)

    try:
        while True:
            intermediate = next(phase_iter)
            try:
                writer(intermediate)
            except Exception:
                pass
            final_state = intermediate
    except StopIteration as done:
        if done.value is not None:
            final_state = done.value

    # LangGraph의 WBSState는 debate_log에 operator.add reducer가 걸려 있다.
    # 노드는 "전체 로그"가 아니라 이번 phase에서 새로 생긴 로그만 반환해야
    # 다음 노드에서 중복 누적되지 않는다. custom stream에는 위에서 전체 state를
    # 그대로 내보내 UI/SSE 호환성을 유지한다.
    patch = dict(final_state)
    final_log = list(final_state.get("debate_log", []) or [])
    if len(final_log) >= initial_debate_len:
        patch["debate_log"] = final_log[initial_debate_len:]
    final_tool_trace = list(final_state.get(STATE_TRACE_KEY, []) or [])
    if len(final_tool_trace) >= initial_tool_len:
        patch[STATE_TRACE_KEY] = final_tool_trace[initial_tool_len:]
    return patch


def wbs_generation_node(state: WBSState) -> WBSState:
    return _run_phase_node(state, phase_wbs_generation)


def task_match_node(state: WBSState) -> WBSState:
    return _run_phase_node(state, phase_task_match)


def l2_debate_node(state: WBSState) -> WBSState:
    return _run_phase_node(state, phase_l2_debate)


def free_discussion_phase_node(state: WBSState) -> WBSState:
    return _run_phase_node(state, phase_free_discussion)


def critic_review_node(state: WBSState) -> WBSState:
    return _run_phase_node(state, phase_critic_review)


def supervisor_mediate_node(state: WBSState) -> WBSState:
    return _run_phase_node(state, phase_supervisor_mediate)


def finalize_node(state: WBSState) -> WBSState:
    return _run_phase_node(state, phase_finalize)


def build_wbs_graph() -> StateGraph:
    """Phase 단위 LangGraph DAG."""
    graph = StateGraph(WBSState)
    graph.add_node("wbs_generation", wbs_generation_node)
    graph.add_node("task_match", task_match_node)
    graph.add_node("l2_debate", l2_debate_node)
    graph.add_node("free_discussion", free_discussion_phase_node)
    graph.add_node("critic_review", critic_review_node)
    graph.add_node("supervisor_mediate", supervisor_mediate_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "wbs_generation")
    graph.add_conditional_edges(
        "wbs_generation",
        route_after_wbs_generation,
        {
            "task_match": "task_match",
            "finalize": "finalize",
        },
    )
    graph.add_edge("task_match", "l2_debate")
    graph.add_edge("l2_debate", "free_discussion")
    graph.add_edge("free_discussion", "critic_review")
    graph.add_edge("critic_review", "supervisor_mediate")
    graph.add_conditional_edges(
        "supervisor_mediate",
        route_after_mediate,
        {
            "wbs_generation": "wbs_generation",
            "task_match": "task_match",
            "finalize": "finalize",
        },
    )
    graph.add_edge("finalize", END)
    return graph


def compile_graph():
    graph = build_wbs_graph()
    return graph.compile()


_compiled_graph = None


def get_compiled_graph():
    """LangGraph 미설치 시 None 반환."""
    if not HAS_LANGGRAPH:
        return None
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = compile_graph()
    return _compiled_graph
