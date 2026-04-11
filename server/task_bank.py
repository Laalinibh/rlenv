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
    # ── EASY TASKS ──────────────────────────────────────────────────
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
    "easy_password_reset": CRMTaskSpec(
        task_id="easy_password_reset",
        difficulty="easy",
        title="Reset locked online-banking password",
        objective="Verify customer identity and reset their online banking password after multiple failed login attempts.",
        customer_opening="I've been locked out of my online banking after too many wrong passwords. I need access right now to pay a bill.",
        expected_intent="account_access",
        required_slots=["customer_id", "last4_card"],
        required_workflows={"verify_identity"},
        optional_workflows={"reissue_card"},
        max_steps=8,
        high_risk=True,
        customer_profile=CustomerProfile(
            customer_id="C-55190",
            name="Sarah Mitchell",
            tier="bronze",
            tenure_months=8,
            monthly_revenue=45.00,
            lifetime_value=360.0,
            open_tickets=0,
            recent_nps=15,
            risk_flags=["multiple_failed_logins"],
        ),
        compliance_policies=[
            CompliancePolicy(
                policy_id="POL-AUTH-02",
                title="Account Lockout Verification",
                requirement="After 5+ failed login attempts, identity must be reverified via customer_id and last4_card before unlocking. Agent must NOT reset password without verification.",
                applies_to=["verify_identity"],
            ),
            CompliancePolicy(
                policy_id="POL-SEC-01",
                title="Password Security Standards",
                requirement="Temporary passwords must be sent only to the registered email or phone on file. Never read a temporary password aloud or type it in chat.",
                applies_to=["verify_identity"],
            ),
        ],
        prior_interactions=[
            PriorInteraction(date="2024-03-01", channel="chat", summary="Set up online banking", resolution="Account created successfully", satisfaction="positive"),
        ],
        knowledge_base=[
            {"topic": "Password Reset Process", "content": "After identity verification, a temporary password link is sent to the registered email. Link expires in 15 minutes. Customer must create a new password meeting complexity requirements (8+ chars, uppercase, number, symbol)."},
            {"topic": "Account Lockout Policy", "content": "Accounts lock after 5 consecutive failed attempts. Lockout duration: 30 minutes for auto-unlock, or immediate via agent-assisted reset after verification."},
        ],
    ),
    "easy_direct_deposit_setup": CRMTaskSpec(
        task_id="easy_direct_deposit_setup",
        difficulty="easy",
        title="Set up direct deposit for new account",
        objective="Help customer configure direct deposit by providing routing and account numbers, and verify their employer details.",
        customer_opening="I just opened my account last week and my employer needs my direct deposit info. Can you help me set that up?",
        expected_intent="account_setup",
        required_slots=["customer_id"],
        required_workflows={"verify_identity"},
        optional_workflows=set(),
        max_steps=8,
        high_risk=False,
        customer_profile=CustomerProfile(
            customer_id="C-61024",
            name="Alex Rivera",
            tier="bronze",
            tenure_months=1,
            monthly_revenue=0.0,
            lifetime_value=0.0,
            open_tickets=0,
            recent_nps=50,
        ),
        compliance_policies=[
            CompliancePolicy(
                policy_id="POL-ACCT-01",
                title="Account Information Disclosure",
                requirement="Routing and account numbers may only be provided after identity verification. Never share full account number in chat—mask all but last 4 digits.",
                applies_to=["verify_identity"],
            ),
        ],
        prior_interactions=[],
        knowledge_base=[
            {"topic": "Direct Deposit Setup", "content": "Provide the bank routing number (021000021) and the customer's account number. Deposits typically begin 1-2 pay cycles after employer processes the form. Voided check image available in online banking."},
            {"topic": "New Account Checklist", "content": "Within first 30 days: set up direct deposit, enroll in online banking, configure alerts, review fee schedule. Early direct deposit setup qualifies for monthly fee waiver."},
        ],
    ),
    # ── MEDIUM TASKS ────────────────────────────────────────────────
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
    "medium_wire_recall": CRMTaskSpec(
        task_id="medium_wire_recall",
        difficulty="medium",
        title="Recall an outgoing domestic wire transfer",
        objective="Help customer request a recall of a domestic wire sent to the wrong recipient, following compliance procedures.",
        customer_opening="I just sent a $2,500 wire to the wrong account number 20 minutes ago. Can you get my money back?",
        expected_intent="wire_recall",
        required_slots=["customer_id", "disputed_amount", "month"],
        required_workflows={"verify_identity", "refund_dispute"},
        optional_workflows={"offer_savings"},
        max_steps=12,
        high_risk=True,
        customer_profile=CustomerProfile(
            customer_id="C-38290",
            name="Patricia Hoffman",
            tier="gold",
            tenure_months=36,
            monthly_revenue=175.00,
            lifetime_value=6300.0,
            open_tickets=0,
            recent_nps=28,
            risk_flags=["recent_large_wire"],
        ),
        compliance_policies=[
            CompliancePolicy(
                policy_id="POL-WIRE-01",
                title="Wire Transfer Recall Protocol",
                requirement="Wire recall requests must be submitted within 30 minutes for best chance of recovery. "
                            "Agent must collect customer_id, amount, and date. Cannot guarantee recovery — receiving "
                            "bank may decline. Must set accurate expectations with customer.",
                applies_to=["verify_identity", "refund_dispute"],
            ),
            CompliancePolicy(
                policy_id="POL-WIRE-02",
                title="Wire Fraud Screening",
                requirement="If wire was sent under duress or due to a scam, agent must escalate to fraud team "
                            "immediately and file a SAR (Suspicious Activity Report) within 24 hours.",
                applies_to=["refund_dispute"],
            ),
            CompliancePolicy(
                policy_id="POL-AUTH-01",
                title="Identity Verification Standard",
                requirement="Customer must provide customer_id before any account-modifying workflow.",
                applies_to=["verify_identity"],
            ),
        ],
        prior_interactions=[
            PriorInteraction(date="2024-02-20", channel="phone", summary="Set up wire transfer capability on account", resolution="Wire service enabled", satisfaction="positive"),
        ],
        knowledge_base=[
            {"topic": "Wire Recall Process", "content": "Domestic wire recalls are processed via SWIFT gpi recall (MT 199). Success rate is ~60% within first hour, dropping to ~15% after 24 hours. The receiving bank has no obligation to return funds if already credited to beneficiary."},
            {"topic": "Wire Transfer Fees", "content": "Domestic wire: $25 outgoing, $15 incoming. Wire recall attempt fee: $30 (waived if bank error). International wire: $45 outgoing."},
            {"topic": "Customer Communication", "content": "Always set accurate expectations: recall is a REQUEST, not a guarantee. Provide case reference number. Follow up within 48 hours with status update."},
        ],
    ),
    "medium_credit_limit_increase": CRMTaskSpec(
        task_id="medium_credit_limit_increase",
        difficulty="medium",
        title="Process credit limit increase request",
        objective="Evaluate and process a credit limit increase request, checking eligibility criteria and explaining the soft pull process.",
        customer_opening="I've had my card for over a year and my income has gone up. I'd like to increase my credit limit from $5,000 to $10,000.",
        expected_intent="credit_limit",
        required_slots=["customer_id", "disputed_amount"],
        required_workflows={"verify_identity", "refund_dispute"},
        optional_workflows={"offer_savings"},
        max_steps=12,
        high_risk=False,
        customer_profile=CustomerProfile(
            customer_id="C-72841",
            name="Kenji Yamamoto",
            tier="silver",
            tenure_months=15,
            monthly_revenue=132.00,
            lifetime_value=1980.0,
            open_tickets=0,
            recent_nps=38,
        ),
        compliance_policies=[
            CompliancePolicy(
                policy_id="POL-CREDIT-01",
                title="Credit Limit Increase Policy",
                requirement="Increases >$3,000 require income verification. Soft pull is performed first; "
                            "hard pull only if customer consents. Must disclose that a hard inquiry may "
                            "impact credit score. Minimum 6 months account history required.",
                applies_to=["verify_identity", "refund_dispute"],
            ),
            CompliancePolicy(
                policy_id="POL-FCRA-01",
                title="Fair Credit Reporting Act Compliance",
                requirement="Must provide adverse action notice if request is denied. Customer has right "
                            "to request the credit report used in the decision within 60 days.",
                applies_to=["refund_dispute"],
            ),
        ],
        prior_interactions=[
            PriorInteraction(date="2023-12-15", channel="app", summary="Activated new credit card", resolution="Card activated", satisfaction="positive"),
            PriorInteraction(date="2024-02-01", channel="chat", summary="Asked about rewards program", resolution="Explained cashback tiers", satisfaction="positive"),
        ],
        knowledge_base=[
            {"topic": "Credit Limit Eligibility", "content": "Criteria: 6+ months account history, no late payments in last 12 months, utilization <80%. Soft pull checks: payment history, current utilization, income-to-debt ratio."},
            {"topic": "Credit Inquiry Types", "content": "Soft pull: no credit score impact, used for pre-qualification. Hard pull: may reduce score 5-10 points, stays on report 2 years. Always obtain explicit consent before hard pull."},
        ],
    ),
    # ── HARD TASKS ──────────────────────────────────────────────────
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
    "hard_elder_fraud_recovery": CRMTaskSpec(
        task_id="hard_elder_fraud_recovery",
        difficulty="hard",
        title="Recover elderly customer's funds after romance scam",
        objective=(
            "Help an elderly customer who has been sending wire transfers to a romance scammer. "
            "Freeze outgoing wires, initiate fraud recovery, and connect to elder abuse resources "
            "while maintaining dignity and empathy."
        ),
        customer_opening=(
            "My daughter made me call. She says the person I've been sending money to online "
            "isn't real. I've sent about $15,000 over three months. I don't know what to believe."
        ),
        expected_intent="elder_fraud",
        required_slots=["customer_id", "company_name", "incident_count", "payroll_date"],
        required_workflows={"verify_identity", "refund_dispute", "offer_savings"},
        optional_workflows={"waive_fee"},
        max_steps=14,
        high_risk=True,
        customer_profile=CustomerProfile(
            customer_id="C-12477",
            name="Margaret O'Brien",
            tier="gold",
            tenure_months=216,
            monthly_revenue=95.00,
            lifetime_value=20520.0,
            open_tickets=0,
            recent_nps=65,
            risk_flags=["elder_customer", "repeated_wire_pattern", "fraud_vulnerability"],
        ),
        compliance_policies=[
            CompliancePolicy(
                policy_id="POL-ELDER-01",
                title="Elder Financial Exploitation Protocol",
                requirement="If agent suspects elder financial exploitation, must: 1) file SAR within 24h, "
                            "2) freeze outgoing wire capability, 3) offer to connect with Adult Protective "
                            "Services. Must treat customer with dignity — never blame the victim.",
                applies_to=["verify_identity", "refund_dispute"],
            ),
            CompliancePolicy(
                policy_id="POL-FRAUD-02",
                title="Wire Fraud Recovery",
                requirement="For suspected fraud wires, initiate recall on all transfers within last 72 hours. "
                            "File IC3 complaint reference for FBI tracking. Provide customer with FTC identity "
                            "theft report number. Recovery rate declines rapidly after 48 hours.",
                applies_to=["refund_dispute", "offer_savings"],
            ),
            CompliancePolicy(
                policy_id="POL-ESCA-03",
                title="Vulnerable Customer Handling",
                requirement="Vulnerable customers (elderly, distressed) require extended interaction time. "
                            "Do not rush. Offer to have a trusted family member join the call. Document "
                            "customer's mental state observations in case notes for regulatory file.",
                applies_to=["verify_identity", "refund_dispute", "offer_savings"],
            ),
            CompliancePolicy(
                policy_id="POL-AML-01",
                title="Anti-Money Laundering Reporting",
                requirement="Repeated wire transfers to unverified individuals totaling >$10,000 require CTR "
                            "filing. Agent must NOT disclose to customer that a SAR is being filed.",
                applies_to=["refund_dispute"],
            ),
        ],
        prior_interactions=[
            PriorInteraction(date="2024-01-10", channel="branch", summary="Sent $3,000 wire to overseas account", resolution="Wire processed as requested", satisfaction="positive"),
            PriorInteraction(date="2024-02-14", channel="phone", summary="Sent $5,000 wire, asked about fee waiver", resolution="Wire sent, fee not waived (standard)", satisfaction="neutral"),
            PriorInteraction(date="2024-03-05", channel="phone", summary="Sent $7,000 wire, teller noted concern", resolution="Wire processed, internal flag added", satisfaction="neutral"),
        ],
        knowledge_base=[
            {"topic": "Romance Scam Indicators", "content": "Red flags: online-only relationship, escalating financial requests, urgency, secrecy, wire transfers to unfamiliar recipients. FBI IC3 reports $1.3B in romance scam losses annually. Elderly victims average $9,000 in losses."},
            {"topic": "Wire Recall for Fraud", "content": "For fraud-related wires: recall all transfers within 72h window. Contact receiving banks directly. File SAR (not disclosed to customer). Coordinate with fraud investigation team. Recovery rate: ~30% for wires <48h old, <5% after 72h."},
            {"topic": "Elder Abuse Resources", "content": "Adult Protective Services: 1-800-677-1116. FTC Identity Theft: IdentityTheft.gov. FBI IC3: ic3.gov. AARP Fraud Helpline: 1-877-908-3360. Local law enforcement for immediate danger."},
            {"topic": "Victim Communication Guidelines", "content": "Never blame the victim. Use phrases: 'This is not your fault', 'These scammers are very sophisticated', 'We're here to help protect your accounts'. Avoid: 'You should have known', 'Why did you send money'."},
        ],
    ),
    "hard_mortgage_hardship": CRMTaskSpec(
        task_id="hard_mortgage_hardship",
        difficulty="hard",
        title="Process mortgage hardship forbearance request",
        objective=(
            "Help a customer who has lost their job apply for mortgage forbearance, "
            "explaining options, collecting documentation requirements, and ensuring "
            "CFPB compliance for loss mitigation procedures."
        ),
        customer_opening=(
            "I was laid off two weeks ago and I can't make next month's mortgage payment. "
            "I heard there are options to pause payments but I don't want to lose my house. "
            "What can I do?"
        ),
        expected_intent="mortgage_hardship",
        required_slots=["customer_id", "company_name", "incident_count", "payroll_date"],
        required_workflows={"verify_identity", "refund_dispute", "offer_savings"},
        optional_workflows={"waive_fee"},
        max_steps=14,
        high_risk=True,
        customer_profile=CustomerProfile(
            customer_id="C-88156",
            name="Robert & Linda Thompson",
            tier="silver",
            tenure_months=84,
            monthly_revenue=1850.00,
            lifetime_value=155400.0,
            open_tickets=0,
            recent_nps=40,
            risk_flags=["income_loss_reported", "mortgage_delinquency_risk"],
        ),
        compliance_policies=[
            CompliancePolicy(
                policy_id="POL-CFPB-01",
                title="CFPB Loss Mitigation Requirements",
                requirement="Under CFPB Regulation X (12 CFR 1024.41), servicer must acknowledge loss mitigation "
                            "application within 5 business days. Must not initiate foreclosure while complete "
                            "application is pending. Must evaluate all available options.",
                applies_to=["verify_identity", "refund_dispute"],
            ),
            CompliancePolicy(
                policy_id="POL-FORB-01",
                title="Forbearance Program Terms",
                requirement="Standard forbearance: 3-6 months, extendable to 12. Payments deferred to end of "
                            "loan term. Must disclose: forbearance does NOT forgive payments, interest continues "
                            "to accrue, credit reporting will show 'in forbearance'. Must provide written agreement.",
                applies_to=["refund_dispute", "offer_savings"],
            ),
            CompliancePolicy(
                policy_id="POL-SCRA-01",
                title="Servicemembers Civil Relief Act Check",
                requirement="Agent must check if borrower is active military (SCRA protected). SCRA caps interest "
                            "at 6% and prohibits foreclosure during active duty + 1 year. Must ask about military "
                            "status before processing any hardship application.",
                applies_to=["verify_identity"],
            ),
            CompliancePolicy(
                policy_id="POL-FAIR-01",
                title="Fair Lending & Non-Discrimination",
                requirement="All forbearance decisions must be based solely on financial criteria. Agent must not "
                            "consider race, religion, national origin, sex, or familial status. Must offer same "
                            "options to all similarly situated borrowers.",
                applies_to=["refund_dispute", "offer_savings"],
            ),
        ],
        prior_interactions=[
            PriorInteraction(date="2024-01-15", channel="app", summary="Made regular mortgage payment", resolution="Payment processed on time", satisfaction="positive"),
            PriorInteraction(date="2024-02-15", channel="app", summary="Made regular mortgage payment", resolution="Payment processed on time", satisfaction="positive"),
            PriorInteraction(date="2024-03-10", channel="phone", summary="Called about auto-pay — wants to pause", resolution="Auto-pay paused per request", satisfaction="neutral"),
        ],
        knowledge_base=[
            {"topic": "Forbearance Options", "content": "1) Standard forbearance: 3-6 months, deferred payments. 2) Repayment plan: reduced payments over 6-12 months. 3) Loan modification: permanent change to rate/term. 4) Partial claim (FHA only): junior lien for missed payments. Each option has different credit reporting implications."},
            {"topic": "Required Documentation", "content": "Hardship application requires: hardship letter explaining circumstances, last 2 pay stubs (or termination letter), last 2 months bank statements, most recent tax return, monthly expense budget. Incomplete applications cannot be evaluated per CFPB rules."},
            {"topic": "Forbearance vs Modification", "content": "Forbearance: temporary pause, payments due later. Modification: permanent change to loan terms (rate, duration, principal). Forbearance is faster (approved in days) while modification requires full underwriting (30-60 days). Forbearance is the right first step for temporary hardship."},
            {"topic": "Credit Impact Disclosure", "content": "During forbearance: account reported as 'in forbearance' (not delinquent) if entered before missing a payment. If already delinquent: reported as delinquent AND in forbearance. After forbearance: resuming payments rebuilds history. Modification may require trial period of 3 on-time payments."},
        ],
    ),
}
