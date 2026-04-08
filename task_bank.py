from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set


@dataclass(frozen=True)
class CRMTaskSpec:
    task_id: str
    difficulty: str
    title: str
    objective: str
    customer_opening: str
    expected_intent: str
    required_slots: List[str]
    required_workflows: Set[str]
    optional_workflows: Set[str] = field(default_factory=set)
    max_steps: int = 14
    high_risk: bool = False


TASKS: Dict[str, CRMTaskSpec] = {
    "easy_card_freeze": CRMTaskSpec(
        task_id="easy_card_freeze",
        difficulty="easy",
        title="Freeze compromised debit card",
        objective="Verify customer identity and freeze card after suspicious transactions.",
        customer_opening=(
            "I see two card transactions I didn't make. Please stop my debit card right now."
        ),
        expected_intent="card_fraud",
        required_slots=["customer_id", "last4_card"],
        required_workflows={"verify_identity", "freeze_card"},
        optional_workflows={"reissue_card"},
        max_steps=10,
        high_risk=True,
    ),
    "medium_dispute_retention": CRMTaskSpec(
        task_id="medium_dispute_retention",
        difficulty="medium",
        title="Dispute fee and save at-risk customer",
        objective=(
            "Resolve refund dispute, de-escalate frustration, and retain customer with a"
            " compliant offer."
        ),
        customer_opening=(
            "You charged me two monthly fees and support ignored me. I'm closing all my accounts."
        ),
        expected_intent="fee_dispute",
        required_slots=["customer_id", "disputed_amount", "month"],
        required_workflows={"verify_identity", "refund_dispute", "waive_fee"},
        optional_workflows={"offer_savings"},
        max_steps=12,
        high_risk=False,
    ),
    "hard_business_churn_prevention": CRMTaskSpec(
        task_id="hard_business_churn_prevention",
        difficulty="hard",
        title="Prevent SMB churn after payment failures",
        objective=(
            "Fix repeated payment failures, recover trust, and propose a value-preserving"
            " retention plan that improves NPS."
        ),
        customer_opening=(
            "Three payroll payments failed this week. If this isn't solved today, we're moving banks."
        ),
        expected_intent="payment_failure",
        required_slots=["customer_id", "company_name", "incident_count", "payroll_date"],
        required_workflows={"verify_identity", "refund_dispute", "offer_savings"},
        optional_workflows={"waive_fee"},
        max_steps=14,
        high_risk=True,
    ),
}
