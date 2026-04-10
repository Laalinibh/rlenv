from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field


class CRMActionType(str, Enum):
    ANALYZE = "analyze"
    RESPOND = "respond"
    WORKFLOW = "workflow"
    FINALIZE = "finalize"
    HANDOFF = "handoff"


class CRMWorkflowType(str, Enum):
    VERIFY_ID = "verify_identity"
    FREEZE_CARD = "freeze_card"
    REISSUE_CARD = "reissue_card"
    REFUND_DISPUTE = "refund_dispute"
    OFFER_SAVINGS = "offer_savings"
    WAIVE_FEE = "waive_fee"
    NONE = "none"


class CriticalEventType(str, Enum):
    CLARIFYING_Q = "clarifying_q"
    SUGGESTION = "suggestion"
    CONFIRMATION = "confirmation"
    HANDOFF = "handoff"
    ESCALATION_TRIGGER = "escalation_trigger"
    REPAIR_MOMENT = "repair_moment"
    TOOL_FAILURE = "tool_failure"
    AUTH_CHECKPOINT = "auth_checkpoint"
    COMPLIANCE_DISCLOSURE = "compliance_disclosure"


class CRMAction(Action):
    action_type: CRMActionType = Field(..., description="Primary agent action")
    rationale: str = Field(default="", description="Reasoning for auditability")
    response_text: str = Field(default="", description="Customer-facing response")
    intent: str = Field(default="", description="Detected customer intent")
    workflow: CRMWorkflowType = Field(default=CRMWorkflowType.NONE)
    extracted_slots: Dict[str, str] = Field(default_factory=dict)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    event_type: Optional[CriticalEventType] = Field(
        default=None,
        description="Critical interaction point for event impact tracking",
    )
    tool_status: str = Field(default="ok", description="ok|failed|timeout")


class TaskDefinition(BaseModel):
    task_id: str
    difficulty: str
    title: str
    objective: str
    max_steps: int


class TurnUsefulness(BaseModel):
    correctness: float = Field(default=0.0, ge=0.0, le=2.0)
    completeness: float = Field(default=0.0, ge=0.0, le=2.0)
    clarity: float = Field(default=0.0, ge=0.0, le=2.0)
    actionability: float = Field(default=0.0, ge=0.0, le=2.0)
    safety: float = Field(default=0.0, ge=0.0, le=2.0)
    normalized_usefulness: float = Field(default=0.0, ge=0.0, le=100.0)


class AccountContext(BaseModel):
    """Customer account data visible to the agent — mirrors a real CRM record."""
    customer_name: str = ""
    tier: str = ""
    tenure_months: int = 0
    monthly_revenue: float = 0.0
    lifetime_value: float = 0.0
    open_tickets: int = 0
    recent_nps: int = 0
    risk_flags: List[str] = Field(default_factory=list)


class PolicyInfo(BaseModel):
    """Compliance policy the agent must follow."""
    policy_id: str = ""
    title: str = ""
    requirement: str = ""


class InteractionRecord(BaseModel):
    """Prior support interaction for context."""
    date: str = ""
    channel: str = ""
    summary: str = ""
    resolution: str = ""
    satisfaction: str = ""


class KnowledgeArticle(BaseModel):
    """Knowledge base article the agent can reference."""
    topic: str = ""
    content: str = ""


class CRMObservation(Observation):
    task: TaskDefinition
    customer_message: str = ""
    conversation_history: List[Dict[str, str]] = Field(default_factory=list)
    required_slots: List[str] = Field(default_factory=list)
    collected_slots: Dict[str, str] = Field(default_factory=dict)
    outstanding_risks: List[str] = Field(default_factory=list)
    workflow_actions_taken: List[str] = Field(default_factory=list)
    nps_proxy: float = Field(default=0.0, ge=-100.0, le=100.0)
    progress_score: float = Field(default=0.0, ge=0.0, le=1.0)
    turn_usefulness: TurnUsefulness = Field(default_factory=TurnUsefulness)
    session_satisfaction_hat: float = Field(default=0.0, ge=0.0, le=1.0)
    critical_event_impacts: Dict[str, float] = Field(default_factory=dict)
    guidance: str = ""
    done_reason: Optional[str] = None
    # Rich CRM context fields
    account_context: AccountContext = Field(default_factory=AccountContext)
    compliance_policies: List[PolicyInfo] = Field(default_factory=list)
    prior_interactions: List[InteractionRecord] = Field(default_factory=list)
    knowledge_base: List[KnowledgeArticle] = Field(default_factory=list)
