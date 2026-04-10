from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Set


@dataclass(frozen=True)
class CustomerProfile:
    """Simulated CRM account record the agent can reference."""
    customer_id: str
    name: str
    tier: str  # bronze / silver / gold / platinum
    tenure_months: int
    monthly_revenue: float
    lifetime_value: float
    open_tickets: int
    recent_nps: int  # -100 to 100
    risk_flags: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class CompliancePolicy:
    """Regulatory / internal policy the agent must respect."""
    policy_id: str
    title: str
    requirement: str
    applies_to: List[str]  # list of workflow IDs this applies to


@dataclass(frozen=True)
class PriorInteraction:
    """Past support interaction for context."""
    date: str
    channel: str
    summary: str
    resolution: str
    satisfaction: str  # positive / neutral / negative


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
    customer_profile: CustomerProfile = field(default_factory=lambda: CustomerProfile(
        customer_id="C-00000", name="Unknown", tier="bronze",
        tenure_months=0, monthly_revenue=0.0, lifetime_value=0.0,
        open_tickets=0, recent_nps=0,
    ))
    compliance_policies: List[CompliancePolicy] = field(default_factory=list)
    prior_interactions: List[PriorInteraction] = field(default_factory=list)
    knowledge_base: List[Dict[str, str]] = field(default_factory=list)


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
        customer_profile=CustomerProfile(
            customer_id="C-98210",
            name="Maria Gonzalez",
            tier="silver",
            tenure_months=26,
            monthly_revenue=89.50,
            lifetime_value=2327.0,
            open_tickets=0,
            recent_nps=42,
            risk_flags=["fraud_alert_active"],
        ),
        compliance_policies=[
            CompliancePolicy(
                policy_id="POL-FRAUD-01",
                title="Fraud Response SLA",
                requirement="Card must be frozen within 5 agent turns of a confirmed fraud report. "
                            "Agent must verify identity before taking any account action.",
                applies_to=["verify_identity", "freeze_card"],
            ),
            CompliancePolicy(
                policy_id="POL-AUTH-01",
                title="Identity Verification Standard",
                requirement="Customer must provide customer_id and last4_card before any "
                            "account-modifying workflow. Do not disclose account details before verification.",
                applies_to=["verify_identity"],
            ),
        ],
        prior_interactions=[
            PriorInteraction(
                date="2024-01-15",
                channel="chat",
                summary="Asked about international transaction fees",
                resolution="Fees explained; no action needed",
                satisfaction="positive",
            ),
        ],
        knowledge_base=[
            {"topic": "Card Freeze Procedure", "content": "Frozen cards block all new authorizations within 30 seconds. Pending transactions may still post. A replacement card is auto-mailed within 3-5 business days."},
            {"topic": "Fraud Liability", "content": "Under Regulation E, customer liability is $0 if reported within 2 business days. After 2 days, liability caps at $50. After 60 days, full liability."},
        ],
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
        customer_profile=CustomerProfile(
            customer_id="C-44571",
            name="James Park",
            tier="gold",
            tenure_months=48,
            monthly_revenue=215.00,
            lifetime_value=10320.0,
            open_tickets=2,
            recent_nps=-35,
            risk_flags=["churn_risk_high", "escalation_history"],
        ),
        compliance_policies=[
            CompliancePolicy(
                policy_id="POL-REFUND-01",
                title="Fee Dispute Resolution",
                requirement="Refunds over $25 require customer_id and disputed_amount confirmation. "
                            "Agent must not promise refund before verification. Maximum refund window is 90 days.",
                applies_to=["refund_dispute", "waive_fee"],
            ),
            CompliancePolicy(
                policy_id="POL-RETAIN-01",
                title="Retention Offer Guardrails",
                requirement="Retention offers (fee waiver, savings bonus) may only be extended to customers "
                            "with tenure >= 12 months. Must disclose that offer is a one-time courtesy. "
                            "Never disparage previous agents or admit systemic failure.",
                applies_to=["waive_fee", "offer_savings"],
            ),
            CompliancePolicy(
                policy_id="POL-ESCA-01",
                title="Escalation Protocol",
                requirement="If customer explicitly demands a supervisor 3+ times, agent must offer handoff. "
                            "De-escalation should be attempted first with empathy and concrete resolution.",
                applies_to=["verify_identity", "refund_dispute"],
            ),
        ],
        prior_interactions=[
            PriorInteraction(
                date="2024-02-10",
                channel="phone",
                summary="Called about double charge on Feb statement ($47.99 x2)",
                resolution="Ticket opened, promised callback — no callback made",
                satisfaction="negative",
            ),
            PriorInteraction(
                date="2024-01-05",
                channel="chat",
                summary="Asked about upgrading to premium checking",
                resolution="Provided comparison; customer said would think about it",
                satisfaction="neutral",
            ),
        ],
        knowledge_base=[
            {"topic": "Fee Waiver Policy", "content": "Monthly maintenance fee ($12.99) may be waived once per 12 months for gold/platinum customers. Silver customers eligible for 50% waiver. Bronze ineligible."},
            {"topic": "Retention Playbook", "content": "For high-LTV churn-risk customers: 1) Acknowledge frustration, 2) Resolve the immediate issue, 3) Offer tangible goodwill (fee waiver, rate match), 4) Confirm satisfaction before closing."},
            {"topic": "Dispute Timeline", "content": "Reg E disputes must be acknowledged within 10 business days. Provisional credit issued within 10 days for amounts over $50. Investigation concludes within 45 days."},
        ],
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
        customer_profile=CustomerProfile(
            customer_id="C-77832",
            name="David Chen",
            tier="platinum",
            tenure_months=72,
            monthly_revenue=4850.00,
            lifetime_value=349200.0,
            open_tickets=3,
            recent_nps=-62,
            risk_flags=["churn_risk_critical", "smb_payroll_dependency", "regulatory_exposure"],
        ),
        compliance_policies=[
            CompliancePolicy(
                policy_id="POL-SMB-01",
                title="Business Account Payment SLA",
                requirement="Payroll payment failures must be investigated within 1 business day. "
                            "Agent must collect company_name, incident_count, and payroll_date. "
                            "Failed payroll may trigger NACHA regulatory reporting if not resolved in 48h.",
                applies_to=["verify_identity", "refund_dispute"],
            ),
            CompliancePolicy(
                policy_id="POL-SMB-02",
                title="Business Retention Authority",
                requirement="Retention offers for platinum SMB accounts may include: fee waiver (up to 6 months), "
                            "dedicated account manager assignment, and priority processing. Total goodwill "
                            "value must not exceed 2% of annual revenue. Must get verbal confirmation of acceptance.",
                applies_to=["offer_savings", "waive_fee"],
            ),
            CompliancePolicy(
                policy_id="POL-NACHA-01",
                title="NACHA ACH Return Handling",
                requirement="ACH returns coded R01 (insufficient funds) or R09 (uncollected funds) require "
                            "bank-side investigation. Agent must NOT promise immediate reprocessing. Disclose "
                            "that reprocessing takes 1-2 business days after root cause is confirmed.",
                applies_to=["refund_dispute"],
            ),
            CompliancePolicy(
                policy_id="POL-ESCA-02",
                title="Critical Business Escalation",
                requirement="If the customer's business operations are actively impaired (e.g., employees not "
                            "paid), agent should offer expedited resolution path and provide a case reference "
                            "number. Do not minimize the impact.",
                applies_to=["verify_identity", "refund_dispute", "offer_savings"],
            ),
        ],
        prior_interactions=[
            PriorInteraction(
                date="2024-03-11",
                channel="phone",
                summary="Reported first payroll failure (Mon). Told it was a 'system glitch'",
                resolution="Agent resubmitted payment; said it would clear by Tue",
                satisfaction="negative",
            ),
            PriorInteraction(
                date="2024-03-12",
                channel="phone",
                summary="Second failure reported (Tue). Customer angry, asked for supervisor",
                resolution="Supervisor unavailable; promised callback within 2 hours — not made",
                satisfaction="negative",
            ),
            PriorInteraction(
                date="2024-03-13",
                channel="chat",
                summary="Third failure (Wed). Customer threatening to leave, employees unpaid",
                resolution="Chat agent opened urgent ticket #INC-88291, no resolution yet",
                satisfaction="negative",
            ),
        ],
        knowledge_base=[
            {"topic": "ACH Payment Processing", "content": "ACH payroll batches process at 6AM and 2PM ET. Returns are received by 4PM same day. Common return codes: R01 (NSF), R02 (account closed), R09 (uncollected funds), R10 (unauthorized). R01/R09 for business accounts usually indicate a hold or processing delay, not actual insufficient funds."},
            {"topic": "SMB Retention Playbook", "content": "For platinum SMB churn prevention: 1) Acknowledge severity immediately — employees not being paid is a crisis, 2) Provide concrete timeline for resolution, 3) Offer fee waiver (up to 6 months) and dedicated account manager, 4) Follow up within 24h to confirm resolution, 5) Document everything for regulatory file."},
            {"topic": "Payroll Failure Remediation", "content": "Steps: verify account status and holds → check ACH batch logs → identify return code → clear holds if bank-side error → resubmit batch with priority processing flag → confirm with customer. Average resolution time: 4-8 hours for priority cases."},
            {"topic": "NACHA Compliance", "content": "Under NACHA rules, originators (our bank) must investigate ACH returns within 2 business days. Repeated failures (3+ in 60 days) trigger automatic review. Customer must be notified of investigation and expected timeline. False promises about reprocessing timelines violate NACHA Operating Rules Section 2.5."},
        ],
    ),
}
