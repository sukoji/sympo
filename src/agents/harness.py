"""Sub-agent harness for stability experiments (RQ1-H)."""
import os
from typing import Any, Callable, Dict, Optional

from agents.state import WBSState
from schemas.wbs_schema import DebateMessage


class AgentHarness:
  @staticmethod
  def is_enabled(state: Optional[WBSState] = None, override: Optional[bool] = None) -> bool:
    if override is not None:
      return bool(override)
    if state is not None and state.get("harness_enabled") is not None:
      return bool(state.get("harness_enabled"))
    return os.environ.get("HARNESS_ENABLED", "true").lower() in ("1", "true", "yes")

  @staticmethod
  def wrap_sub_agent(agent_func: Callable[[WBSState], Any], role_name: str = ""):
    def wrapper(state: WBSState):
      if not AgentHarness.is_enabled(state):
        return agent_func(state)
      state = dict(state)
      state["_harness_role_anchor"] = role_name
      drift_count = int(state.get("_role_drift_count", 0) or 0)
      caught = int(state.get("_harness_caught_exceptions", 0) or 0)
      try:
        result = agent_func(state)
      except Exception as exc:
        caught += 1
        msg = DebateMessage(
          timestamp="",
          agent_role=role_name or "Agent",
          agent_name="Harness",
          message=f"[Harness] {type(exc).__name__}: {exc}",
          message_type="mediation",
        )
        return {"debate_log": [msg], "_harness_caught_exceptions": caught}
      if isinstance(result, dict):
        for item in result.get("debate_log", []) or []:
          anchor = state.get("_harness_role_anchor")
          if anchor and getattr(item, "agent_role", None) and str(item.agent_role) != str(anchor):
            drift_count += 1
        result = dict(result)
        result["_role_drift_count"] = drift_count
        result["_harness_caught_exceptions"] = caught
      return result

    return wrapper
