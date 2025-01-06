"""
symPO debate orchestration — phase generators shared by LangGraph and sequential paths.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, Generator, Iterable, List, Optional

from agents.harness import AgentHarness
from agents.state import WBSState
from agents.sub_agents import PASS_TOKEN, candidate_review_agent, critic_agent, free_discussion_agent
from agents.supervisor_agent import (
    supervisor_check_and_intervene,
    supervisor_finalize,
    supervisor_mediate,
    supervisor_task_match,
)
from agents.wbs_gen_agent import wbs_gen_node
from orchestration.graph_builder import get_compiled_graph
from orchestration.mcp_tool_layer import STATE_TRACE_KEY, call_state_tool
from schemas.wbs_schema import AgentRole, WBSLevel


def ensure_runtime_state(state: WBSState) -> WBSState:
    merged = dict(state)
    merged.setdefault("debate_log", [])
    merged.setdefault(STATE_TRACE_KEY, [])
    merged.setdefault("_free_turn_count", 0)
    merged.setdefault("_anyone_spoke_in_free", False)
    merged.setdefault("_current_l2_task_id", None)
    merged.setdefault("_current_agent_acting", None)
    merged.setdefault("_l2_debate_cutoff", False)
    return merged  # type: ignore[return-value]


def _merge_state(state: WBSState, patch: Dict[str, Any]) -> WBSState:
    merged = ensure_runtime_state(state)
    for key, value in (patch or {}).items():
        if key == "debate_log" and value:
            merged["debate_log"] = list(merged.get("debate_log", [])) + list(value)
        elif key == STATE_TRACE_KEY and value:
            merged[STATE_TRACE_KEY] = list(merged.get(STATE_TRACE_KEY, [])) + list(value)
        else:
            merged[key] = value
    return merged  # type: ignore[return-value]


def _level_value(task) -> str:
    lvl = getattr(task, "level", None)
    return lvl.value if hasattr(lvl, "value") else str(lvl or "")


def _sorted_l2_tasks(state: WBSState) -> List:
    order = {"High": 0, "Medium": 1, "Low": 2}
    tasks = state.get("current_wbs_draft") or []
    l2 = [t for t in tasks if _level_value(t) == "L2"]
    return sorted(l2, key=lambda t: order.get(getattr(t, "importance", "Medium"), 1))


def _member_by_id(state: WBSState, member_id: str):
    for m in state.get("team_members", []):
        if m.member_id == member_id:
            return m
    return None


def _run_phase(phase_fn, state: WBSState) -> Generator[WBSState, None, WBSState]:
    current = ensure_runtime_state(state)
    gen = phase_fn(current)
    try:
        while True:
            step = next(gen)
            current = step
            yield current
    except StopIteration as done:
        if done.value is not None:
            current = done.value
    return current


def phase_wbs_generation(state: WBSState) -> Generator[WBSState, None, WBSState]:
    current = ensure_runtime_state(state)
    current["_current_agent_acting"] = "WBS Gen Agent"

    def _run():
        return wbs_gen_node(current)

    patch = call_state_tool(current, "wbs-server.generate_draft", _run, label="WBS generation")
    current = _merge_state(current, patch)
    yield current
    return current


def phase_task_match(state: WBSState) -> Generator[WBSState, None, WBSState]:
    current = ensure_runtime_state(state)
    current["_current_agent_acting"] = "Task Manager"

    def _run():
        return supervisor_task_match(current)

    patch = call_state_tool(current, "assignment-server.match_tasks", _run, label="Task match")
    current = _merge_state(current, patch)
    yield current
    return current


def phase_l2_debate(state: WBSState) -> Generator[WBSState, None, WBSState]:
    current = ensure_runtime_state(state)
    pools = current.get("l2_candidate_pools") or {}
    consecutive_pass: Dict[str, int] = {}

    for l2_task in _sorted_l2_tasks(current):
        l2_id = l2_task.task_id
        current["_current_l2_task_id"] = l2_id
        current["_l2_debate_cutoff"] = False
        consecutive_pass[l2_id] = 0
        member_ids = pools.get(l2_id) or []
        for mid in member_ids[:4]:
            member = _member_by_id(current, mid)
            if not member:
                continue
            current["_current_agent_acting"] = member.name

            def _run(member=member):
                wrapped = AgentHarness.wrap_sub_agent(
                    lambda s, m=member: candidate_review_agent(s, m),
                    role_name="Candidate Reviewer",
                )
                return wrapped(current)

            patch = call_state_tool(
                current,
                "debate-server.candidate_review",
                _run,
                label=f"L2 candidate review ({l2_id})",
            )
            current = _merge_state(current, patch)
            last_msgs = current.get("debate_log", [])[-1:]
            if last_msgs and PASS_TOKEN in getattr(last_msgs[0], "message", "").upper():
                consecutive_pass[l2_id] += 1
            else:
                consecutive_pass[l2_id] = 0
            intervention = supervisor_check_and_intervene(current)
            if intervention:
                current = _merge_state(current, intervention)
                if intervention.get("_l2_debate_cutoff"):
                    break
            if consecutive_pass[l2_id] >= 2:
                break
            yield current
        yield current

    current["_current_l2_task_id"] = None
    return current


def _free_discussion_roles(state: WBSState) -> Iterable[tuple]:
    mapping = {
        "planner_node": (AgentRole.PLANNER, "플래너"),
        "frontend_node": (AgentRole.FRONTEND, "FE"),
        "backend_node": (AgentRole.BACKEND, "BE"),
        "designer_node": (AgentRole.DESIGNER, "디자이너"),
        "qa_node": (AgentRole.QA, "QA"),
    }
    ctx = state.get("calling_context") or {}
    for node_name, (role, label) in mapping.items():
        mid = ctx.get(node_name)
        member = _member_by_id(state, mid) if mid else None
        if member:
            yield role, label, member


def phase_free_discussion(state: WBSState) -> Generator[WBSState, None, WBSState]:
    current = ensure_runtime_state(state)
    max_turns = int(current.get("max_free_turns") or 3)
    current["_free_turn_count"] = 0
    current["_anyone_spoke_in_free"] = False

    while current["_free_turn_count"] < max_turns:
        spoke_this_round = False
        for role, label, member in _free_discussion_roles(current):
            current["_current_agent_acting"] = member.name

            def _run(role=role, label=label, member=member):
                wrapped = AgentHarness.wrap_sub_agent(
                    lambda s, r=role, n=label, m=member: free_discussion_agent(s, r, n, m),
                    role_name=str(role),
                )
                return wrapped(current)

            patch = call_state_tool(
                current,
                "debate-server.free_discussion",
                _run,
                label=f"Free discussion ({member.name})",
            )
            current = _merge_state(current, patch)
            if patch.get("_anyone_spoke_in_free") or (
                patch.get("debate_log")
                and not PASS_TOKEN in getattr(patch["debate_log"][-1], "message", "").upper()
            ):
                spoke_this_round = True
                current["_anyone_spoke_in_free"] = True
            intervention = supervisor_check_and_intervene(current)
            if intervention:
                current = _merge_state(current, intervention)
            yield current

        current["_free_turn_count"] = int(current.get("_free_turn_count") or 0) + 1
        if not spoke_this_round:
            break
        if (
            current.get("current_round", 0) >= current.get("min_rounds", 0)
            and current.get("_anyone_spoke_in_free")
            and _recent_pass_streak(current)
        ):
            break

    return current


def _recent_pass_streak(state: WBSState, n: int = 3) -> bool:
    logs = [m for m in state.get("debate_log", []) if getattr(m, "message_type", "") not in ("mediation", "decision")]
    if len(logs) < n:
        return False
    return all(PASS_TOKEN in getattr(m, "message", "").upper() for m in logs[-n:])


def phase_critic_review(state: WBSState) -> Generator[WBSState, None, WBSState]:
    current = ensure_runtime_state(state)
    if not current.get("critic_enabled"):
        yield current
        return current
    current["_current_agent_acting"] = "Critic"

    def _run():
        return critic_agent(current)

    patch = call_state_tool(current, "debate-server.role_review", _run, label="Critic review")
    current = _merge_state(current, patch)
    yield current
    return current


def phase_supervisor_mediate(state: WBSState) -> Generator[WBSState, None, WBSState]:
    current = ensure_runtime_state(state)
    current["_current_agent_acting"] = "PM 에이전트"

    def _run():
        return supervisor_mediate(current)

    patch = call_state_tool(current, "supervisor-server.mediate", _run, label="Supervisor mediate")
    current = _merge_state(current, patch)
    current["current_round"] = int(current.get("current_round", 0)) + 1
    yield current
    return current


def phase_finalize(state: WBSState) -> Generator[WBSState, None, WBSState]:
    current = ensure_runtime_state(state)
    current["_current_agent_acting"] = "PM Finalize"

    def _run():
        return supervisor_finalize(current)

    patch = call_state_tool(current, "supervisor-server.finalize", _run, label="Finalize")
    current = _merge_state(current, patch)
    yield current
    return current


def route_after_wbs_generation(state: WBSState) -> str:
    if int(state.get("max_rounds", 0) or 0) <= 0:
        return "finalize"
    return "task_match"


def route_after_mediate(state: WBSState) -> str:
    if state.get("wbs_revision_needed"):
        return "wbs_generation"
    if state.get("consensus_reached"):
        return "finalize"
    if int(state.get("current_round", 0) or 0) >= int(state.get("max_rounds", 3) or 3):
        return "finalize"
    return "task_match"


def run_sequential_debate(state: WBSState, max_rounds: int) -> Generator[WBSState, None, None]:
    current = ensure_runtime_state(state)
    current["max_rounds"] = max_rounds

    while True:
        for step in _run_phase(phase_wbs_generation, current):
            current = step
            yield current
        if route_after_wbs_generation(current) == "finalize":
            for step in _run_phase(phase_finalize, current):
                current = step
                yield current
            return

        for step in _run_phase(phase_task_match, current):
            current = step
            yield current
        for step in _run_phase(phase_l2_debate, current):
            current = step
            yield current
        for step in _run_phase(phase_free_discussion, current):
            current = step
            yield current
        for step in _run_phase(phase_critic_review, current):
            current = step
            yield current
        for step in _run_phase(phase_supervisor_mediate, current):
            current = step
            yield current

        nxt = route_after_mediate(current)
        if nxt == "finalize":
            for step in _run_phase(phase_finalize, current):
                current = step
                yield current
            return
        if nxt == "wbs_generation":
            continue
        # task_match — next debate round


def execute_sympo_flow(state: WBSState, max_rounds: int) -> Generator[WBSState, None, None]:
    graph = get_compiled_graph()
    base = ensure_runtime_state(state)
    base["max_rounds"] = max_rounds

    if graph is not None:
        try:
            for chunk in graph.stream(base, stream_mode="custom"):
                if isinstance(chunk, dict):
                    yield _merge_state(base, chunk)
            return
        except Exception as exc:
            print(f"[WARN] LangGraph 실행 실패, sequential fallback: {exc}")

    yield from run_sequential_debate(base, max_rounds)
