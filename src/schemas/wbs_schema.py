"""WBS 및 토론 로그 스키마."""
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class WBSLevel(str, Enum):
    L1 = "L1"
    L2 = "L2"
    L3 = "L3"


class AgentRole(str, Enum):
    SUPERVISOR = "Supervisor"
    PLANNER = "Planner"
    FRONTEND = "Frontend Developer"
    BACKEND = "Backend Developer"
    DESIGNER = "Designer"
    QA = "QA Engineer"
    CRITIC = "Critic"
    CANDIDATE = "Candidate Reviewer"


class WBSTask(BaseModel):
    task_id: str
    title: str
    level: WBSLevel
    parent_id: Optional[str] = None
    description: str = ""
    assigned_to: List[str] = Field(default_factory=list)
    assigned_role: str = ""
    required_role: str = ""
    estimated_days: float = 1.0
    buffer_days: float = 0.0
    total_days: float = 1.0
    dependencies: List[str] = Field(default_factory=list)
    start_week: Optional[int] = None
    end_week: Optional[int] = None
    deliverables: List[str] = Field(default_factory=list)
    risk_factors: List[str] = Field(default_factory=list)
    importance: str = "Medium"


class DebateMessage(BaseModel):
    timestamp: str
    agent_role: AgentRole
    agent_name: str
    message: str
    message_type: str = "comment"
    task_id: Optional[str] = None
    buffer_days_proposed: Optional[int] = None


class WBSOutput(BaseModel):
    project_name: str
    total_weeks: float
    tasks: List[Dict[str, Any]]
    debate_log: List[Dict[str, Any]]
    summary: str = ""
    agent_personas_used: List[str] = Field(default_factory=list)
