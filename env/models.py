"""
OpenEnv typed models for EmailTriageEnv.
Observation, Action, Reward, and State schemas.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Email representation
# ---------------------------------------------------------------------------

class Email(BaseModel):
    id: str
    from_addr: str = Field(alias="from")
    to_addr: str = Field(alias="to")
    subject: str
    body: str
    timestamp: str
    has_attachment: bool
    is_read: bool = False
    label: Optional[str] = None
    priority: Optional[str] = None
    routed_to: Optional[str] = None
    reply_drafted: Optional[str] = None
    archived: bool = False
    escalated: bool = False
    escalation_reason: Optional[str] = None

    model_config = {"populate_by_name": True}


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class EmailSummary(BaseModel):
    """Lightweight email summary returned in observations (no full body by default)."""
    id: str
    from_addr: str
    subject: str
    timestamp: str
    has_attachment: bool
    is_read: bool
    label: Optional[str] = None
    priority: Optional[str] = None
    routed_to: Optional[str] = None
    archived: bool
    escalated: bool
    has_draft: bool


class Observation(BaseModel):
    """What the agent sees at each step."""
    emails: List[EmailSummary] = Field(description="All emails in the inbox")
    current_email: Optional[Dict[str, Any]] = Field(
        default=None, description="Full details of the currently focused email, if any"
    )
    task_id: str = Field(description="Current active task identifier")
    task_description: str = Field(description="Natural-language description of what the agent must accomplish")
    step_count: int = Field(description="Number of steps taken in current episode")
    max_steps: int = Field(description="Maximum steps allowed")
    available_actions: List[str] = Field(description="List of valid action types at this step")
    info: Dict[str, Any] = Field(default_factory=dict, description="Additional task-specific context")


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class Action(BaseModel):
    """
    Agent action. One of:
      - classify:      email_id + category
      - prioritize:    email_id + priority_level
      - route:         email_id + department
      - draft_reply:   email_id + reply_body
      - archive:       email_id
      - escalate:      email_id + reason
      - read:          email_id  (focus and read full email)
      - done:          signal episode completion
    """
    action_type: str = Field(description="One of: classify, prioritize, route, draft_reply, archive, escalate, read, done")
    email_id: Optional[str] = Field(default=None, description="Target email identifier")
    category: Optional[str] = Field(default=None, description="Category for classify action")
    priority_level: Optional[str] = Field(default=None, description="Priority for prioritize action")
    department: Optional[str] = Field(default=None, description="Department for route action")
    reply_body: Optional[str] = Field(default=None, description="Reply text for draft_reply action")
    reason: Optional[str] = Field(default=None, description="Reason text for escalate action")


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

class Reward(BaseModel):
    """Step-level reward signal."""
    value: float = Field(ge=0.0, le=1.0, description="Reward value for this step [0.0, 1.0]")
    components: Dict[str, float] = Field(
        default_factory=dict,
        description="Breakdown of reward by component (correctness, tone, completeness, efficiency)"
    )
    message: str = Field(default="", description="Human-readable explanation of the reward")
    cumulative: float = Field(ge=0.0, le=1.0, description="Cumulative episode reward so far")


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class EnvState(BaseModel):
    """Full internal environment state (for state() endpoint)."""
    task_id: str
    emails: List[Dict[str, Any]]
    step_count: int
    done: bool
    total_reward: float
    action_history: List[Dict[str, Any]]
