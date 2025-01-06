"""Internal MCP-compatible tool-call layer.

This module intentionally does not start an external MCP server.  It defines
the same boundary the project would expose through MCP later:

- tool names are stable `server.tool` identifiers
- each invocation records structured metadata in state
- handlers keep the existing Python call path, so current experiments and UI
  behavior stay unchanged

The layer is a compatibility shim first, an extraction point second.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from time import perf_counter
from typing import Any, Callable, Dict, Iterable, Optional


STATE_TRACE_KEY = "mcp_tool_trace"


@dataclass(frozen=True)
class MCPToolSpec:
    """Minimal description of an internal MCP-style tool."""

    server: str
    name: str
    description: str
    input_keys: tuple[str, ...] = ()
    output_keys: tuple[str, ...] = ()

    @property
    def qualified_name(self) -> str:
        return f"{self.server}.{self.name}"


TOOL_SPECS: tuple[MCPToolSpec, ...] = (
    MCPToolSpec(
        server="wbs-server",
        name="generate_draft",
        description="Generate or revise the initial WBS draft from PRD and team context.",
        input_keys=("prd", "team_members", "wbs_revision_hints", "model_config"),
        output_keys=("current_wbs_draft", "debate_log", "wbs_revision_needed"),
    ),
    MCPToolSpec(
        server="assignment-server",
        name="match_tasks",
        description="Assign L3 tasks, build L2 candidate pools, and compute assignment evidence.",
        input_keys=("current_wbs_draft", "team_members", "member_role_map", "model_config"),
        output_keys=(
            "assigned_tasks",
            "called_agents",
            "calling_context",
            "l2_candidate_pools",
            "task_candidate_pools",
            "l2_agent_mapping",
            "assignment_evidence",
            "debate_log",
        ),
    ),
    MCPToolSpec(
        server="debate-server",
        name="candidate_review",
        description="Run one selected member review turn for the current L2 task.",
        input_keys=("current_wbs_draft", "debate_log", "_current_l2_task_id", "l2_candidate_pools", "_candidate_rag_context"),
        output_keys=("debate_log",),
    ),
    MCPToolSpec(
        server="rag-server",
        name="retrieve_candidate_context",
        description="Select retrieved WBS and meeting evidence for a candidate review turn.",
        input_keys=("rag_reference_wbs", "rag_meeting_logs", "_current_l2_task_id"),
        output_keys=("_candidate_rag_context",),
    ),
    MCPToolSpec(
        server="debate-server",
        name="role_review",
        description="Run one role-based specialist review turn.",
        input_keys=("current_wbs_draft", "debate_log", "calling_context", "_current_l2_task_id"),
        output_keys=("debate_log",),
    ),
    MCPToolSpec(
        server="debate-server",
        name="free_discussion",
        description="Run free discussion or critic review turns.",
        input_keys=("current_wbs_draft", "debate_log", "l2_candidate_pools"),
        output_keys=("debate_log",),
    ),
    MCPToolSpec(
        server="supervisor-server",
        name="mediate",
        description="Analyze debate, apply PM decisions, and choose next orchestration action.",
        input_keys=("current_wbs_draft", "assigned_tasks", "debate_log", "current_round"),
        output_keys=(
            "current_wbs_draft",
            "current_round",
            "consensus_reached",
            "wbs_revision_needed",
            "wbs_revision_hints",
            "debate_log",
        ),
    ),
    MCPToolSpec(
        server="supervisor-server",
        name="finalize",
        description="Finalize WBS and produce generation summary.",
        input_keys=("current_wbs_draft", "assigned_tasks", "debate_log"),
        output_keys=("final_wbs", "generation_summary", "member_project_roles", "debate_log"),
    ),
)

_SPEC_BY_NAME = {spec.qualified_name: spec for spec in TOOL_SPECS}


def list_tool_specs() -> tuple[MCPToolSpec, ...]:
    """Return the MCP-compatible tool catalog."""

    return TOOL_SPECS


def tool_catalog_dicts() -> list[dict[str, Any]]:
    """Return JSON-serializable tool catalog metadata."""

    return [
        {
            "server": spec.server,
            "name": spec.name,
            "qualified_name": spec.qualified_name,
            "description": spec.description,
            "input_keys": list(spec.input_keys),
            "output_keys": list(spec.output_keys),
        }
        for spec in TOOL_SPECS
    ]


def ensure_tool_trace(state: Dict[str, Any]) -> None:
    """Ensure state has a mutable tool trace list."""

    state.setdefault(STATE_TRACE_KEY, [])


def _summarize_keys(value: Any) -> list[str]:
    if isinstance(value, dict):
        return sorted(str(k) for k in value.keys() if not str(k).startswith("_"))
    return []


def _append_trace(state: Dict[str, Any], trace: Dict[str, Any]) -> None:
    ensure_tool_trace(state)
    state[STATE_TRACE_KEY].append(trace)


def call_state_tool(
    state: Dict[str, Any],
    qualified_name: str,
    handler: Callable[[], Any],
    *,
    label: Optional[str] = None,
    input_keys: Optional[Iterable[str]] = None,
) -> Any:
    """Run a Python handler as an MCP-style tool call and trace the result.

    The handler is executed directly to preserve existing behavior.  Exceptions
    are re-raised after tracing so callers keep their current error handling.
    """

    spec = _SPEC_BY_NAME.get(qualified_name)
    server, _, name = qualified_name.partition(".")
    started = perf_counter()
    selected_input_keys = tuple(input_keys or (spec.input_keys if spec else ()))

    trace = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "server": spec.server if spec else server,
        "tool": spec.name if spec else name,
        "qualified_name": qualified_name,
        "label": label or qualified_name,
        "input_keys": [key for key in selected_input_keys if key in state],
        "ok": False,
    }
    try:
        result = handler()
    except Exception as exc:
        trace["elapsed_ms"] = round((perf_counter() - started) * 1000, 2)
        trace["error"] = f"{type(exc).__name__}: {exc}"
        _append_trace(state, trace)
        raise

    trace["elapsed_ms"] = round((perf_counter() - started) * 1000, 2)
    trace["ok"] = True
    trace["output_keys"] = _summarize_keys(result)
    _append_trace(state, trace)
    return result
