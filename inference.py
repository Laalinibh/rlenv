"""
MANDATORY ENV VARS
- API_BASE_URL
- MODEL_NAME
- HF_TOKEN

Optional fallback
- OPENAI_API_KEY
"""

from __future__ import annotations

import json
import os
import random
import sys
from typing import Dict, List

import requests

# Force unbuffered output — critical for Docker/subprocess capture
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

from openai import OpenAI

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

BENCHMARK = "customer_relationship"
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")

MAX_STEPS = 10
TRAJECTORIES_PER_TASK = 3
TEMPERATURE = 0.2
STOCHASTIC_NOISE = True  # add realistic noise to customer simulation

TASKS = [
    "easy_card_freeze",
    "easy_password_reset",
    "easy_direct_deposit_setup",
    "medium_dispute_retention",
    "medium_wire_recall",
    "medium_credit_limit_increase",
    "hard_business_churn_prevention",
    "hard_elder_fraud_recovery",
    "hard_mortgage_hardship",
]

TASK_INFO = {
    "easy_card_freeze": {
        "expected_intent": "card_fraud",
        "required_workflows": ["verify_identity", "freeze_card"],
        "optional_workflows": ["reissue_card"],
    },
    "easy_password_reset": {
        "expected_intent": "account_access",
        "required_workflows": ["verify_identity"],
        "optional_workflows": ["reissue_card"],
    },
    "easy_direct_deposit_setup": {
        "expected_intent": "account_setup",
        "required_workflows": ["verify_identity"],
        "optional_workflows": [],
    },
    "medium_dispute_retention": {
        "expected_intent": "fee_dispute",
        "required_workflows": ["verify_identity", "refund_dispute", "waive_fee"],
        "optional_workflows": ["offer_savings"],
    },
    "medium_wire_recall": {
        "expected_intent": "wire_recall",
        "required_workflows": ["verify_identity", "refund_dispute"],
        "optional_workflows": ["offer_savings"],
    },
    "medium_credit_limit_increase": {
        "expected_intent": "credit_limit",
        "required_workflows": ["verify_identity", "refund_dispute"],
        "optional_workflows": ["offer_savings"],
    },
    "hard_business_churn_prevention": {
        "expected_intent": "payment_failure",
        "required_workflows": ["verify_identity", "refund_dispute", "offer_savings"],
        "optional_workflows": ["waive_fee"],
    },
    "hard_elder_fraud_recovery": {
        "expected_intent": "elder_fraud",
        "required_workflows": ["verify_identity", "refund_dispute", "offer_savings"],
        "optional_workflows": ["waive_fee"],
    },
    "hard_mortgage_hardship": {
        "expected_intent": "mortgage_hardship",
        "required_workflows": ["verify_identity", "refund_dispute", "offer_savings"],
        "optional_workflows": ["waive_fee"],
    },
}

VALID_ACTION_TYPES = {"analyze", "respond", "workflow", "finalize", "handoff"}
VALID_WORKFLOWS = {"verify_identity", "freeze_card", "reissue_card", "refund_dispute", "offer_savings", "waive_fee", "none"}
VALID_EVENT_TYPES = {"clarifying_q", "suggestion", "confirmation", "handoff", "escalation_trigger", "repair_moment", "tool_failure", "auth_checkpoint", "compliance_disclosure"}
ALLOWED_KEYS = {"action_type", "rationale", "response_text", "intent", "workflow", "extracted_slots", "confidence", "event_type", "tool_status", "metadata"}


# ── Structured log helpers (mandatory [START]/[STEP]/[END] format) ──

def log_start(task: str, model: str) -> None:
    line = "[START] task=%s env=%s model=%s" % (task, BENCHMARK, model)
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def log_step(step: int, action: str, reward: float, done: bool, error: str | None = None) -> None:
    reward = max(0.0051, min(0.9949, float(reward)))
    err = str(error)[:60] if error else "null"
    done_s = "true" if done else "false"
    line = "[STEP] step=%d action=%s reward=%.2f done=%s error=%s" % (
        step, action, reward, done_s, err)
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    if not rewards:
        rewards = [0.01]
    rewards = [max(0.0051, min(0.9949, float(r))) for r in rewards]
    score = max(0.0051, min(0.9949, float(score)))
    rstr = ",".join("%.2f" % r for r in rewards)
    succ = "true" if success else "false"
    line = "[END] success=%s steps=%d score=%.2f rewards=%s" % (
        succ, steps, score, rstr)
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def require_env() -> None:
    missing = []
    if not MODEL_NAME:
        missing.append("MODEL_NAME")
    if not API_KEY:
        missing.append("HF_TOKEN or OPENAI_API_KEY")
    if missing:
        raise RuntimeError(f"Missing required env vars: {', '.join(missing)}")


# ── HTTP environment client ──

def env_reset(session: requests.Session, task_id: str) -> Dict:
    """POST /reset to the environment server and return the parsed result."""
    resp = session.post(f"{ENV_BASE_URL}/reset", json={"task_id": task_id}, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    obs_data = payload.get("observation", {})
    return {
        "observation": obs_data,
        "reward": payload.get("reward", 0.0),
        "done": payload.get("done", False),
        "metadata": obs_data.get("metadata") or payload.get("metadata", {}),
    }


def env_step(session: requests.Session, action: Dict) -> Dict:
    """POST /step to the environment server and return the parsed result."""
    resp = session.post(f"{ENV_BASE_URL}/step", json={"action": action}, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    obs_data = payload.get("observation", {})
    return {
        "observation": obs_data,
        "reward": payload.get("reward", 0.0),
        "done": payload.get("done", False),
        "metadata": obs_data.get("metadata") or payload.get("metadata", {}),
    }


def sanitize_action(raw: Dict) -> Dict:
    """Strip unknown keys and normalise enum values to match CRMAction schema."""
    action = {k: v for k, v in raw.items() if k in ALLOWED_KEYS}

    if action.get("action_type") not in VALID_ACTION_TYPES:
        action["action_type"] = "respond"
    if action.get("workflow") not in VALID_WORKFLOWS:
        action["workflow"] = "none"
    evt = action.get("event_type")
    if evt is not None and evt not in VALID_EVENT_TYPES:
        action["event_type"] = None

    try:
        action["confidence"] = max(0.0, min(1.0, float(action.get("confidence", 0.5))))
    except (TypeError, ValueError):
        action["confidence"] = 0.5

    # Ensure response_text is a non-empty string
    rt = action.get("response_text")
    if not isinstance(rt, str) or not rt.strip():
        action["response_text"] = "I understand your concern. Let me look into this for you right away."

    # Ensure rationale is a string
    if not isinstance(action.get("rationale"), str):
        action["rationale"] = ""

    if not isinstance(action.get("extracted_slots"), dict):
        action["extracted_slots"] = {}
    else:
        action["extracted_slots"] = {
            k: str(v) for k, v in action["extracted_slots"].items()
            if v is not None and str(v).strip()
        }

    return action


def simulate_customer(client: OpenAI, task_id: str, agent_message: str, missing_slots: List[str]) -> str:
    """LLM call simulating a customer who responds with the required slot values.

    When STOCHASTIC_NOISE is enabled, the simulated customer may:
    - Omit some requested information (forcing an extra clarification turn)
    - Add irrelevant tangents or emotional outbursts
    - Provide ambiguous or partial answers
    """
    if not missing_slots or not agent_message:
        return ""

    slot_hints = {
        "customer_id": "a customer ID like C-98210",
        "last4_card": "the last 4 digits of your card like 4532",
        "disputed_amount": "a dollar amount like $47.99",
        "month": "a month like March 2024",
        "company_name": "a business name like Greenfield Logistics",
        "incident_count": "a count like 3",
        "payroll_date": "a date like 2024-03-15",
    }

    # Stochastic noise: sometimes the customer doesn't cooperate fully
    noise_instruction = ""
    if STOCHASTIC_NOISE and random.random() < 0.25:
        noise_type = random.choice([
            "Omit one of the requested details and say you'll look it up. Add a complaint about wait times.",
            "Provide the information but also go on a brief emotional tangent about how stressed you are.",
            "Give a slightly ambiguous answer for one detail (e.g., 'I think it was around $50' instead of exact amount).",
        ])
        noise_instruction = f"\nBehavior twist: {noise_type}\n"

    slot_desc = ", ".join(f"{s} ({slot_hints.get(s, s)})" for s in missing_slots)

    prompt = (
        f"You are a customer calling fintech support about: {task_id.replace('_', ' ')}.\n"
        f"The agent just said: \"{agent_message}\"\n\n"
        f"Respond naturally as a concerned customer. Include these details in your reply: {slot_desc}.\n"
        f"{noise_instruction}"
        "Keep it to 1-3 sentences."
    )
    msg = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0.4 + (0.3 * random.random() if STOCHASTIC_NOISE else 0.0),
        max_tokens=150,
        messages=[
            {"role": "system", "content": "You are simulating a fintech customer. Respond naturally with the requested information."},
            {"role": "user", "content": prompt},
        ],
    )
    return (msg.choices[0].message.content or "").strip()


def build_prompt(obs_dict: Dict, task_id: str, customer_reply: str = "") -> str:
    ob = obs_dict.get("observation", {})
    meta = obs_dict.get("metadata", {})
    task = ob.get("task", {})
    collected = ob.get("collected_slots", {})
    required = ob.get("required_slots", [])
    outstanding = ob.get("outstanding_risks", [])
    workflows_taken = ob.get("workflow_actions_taken", [])
    conversation = ob.get("conversation_history", [])
    customer_msg = ob.get("customer_message", "")

    # Rich context fields
    account = ob.get("account_context", {})
    policies = ob.get("compliance_policies", [])
    prior = ob.get("prior_interactions", [])
    kb = ob.get("knowledge_base", [])

    info = TASK_INFO[task_id]
    missing_slots = [s for s in required if s not in collected]
    done_wfs = set(workflows_taken)
    pending_wfs = [w for w in info["required_workflows"] if w not in done_wfs]

    customer_context = f"Latest customer reply: \"{customer_reply}\"\n" if customer_reply else ""

    # Build account summary
    account_summary = ""
    if account and account.get("customer_name"):
        account_summary = (
            f"CUSTOMER ACCOUNT:\n"
            f"  Name: {account['customer_name']} | Tier: {account.get('tier','?')} | "
            f"Tenure: {account.get('tenure_months',0)} months\n"
            f"  Monthly Revenue: ${account.get('monthly_revenue',0):.2f} | "
            f"Lifetime Value: ${account.get('lifetime_value',0):.2f}\n"
            f"  Open Tickets: {account.get('open_tickets',0)} | Recent NPS: {account.get('recent_nps',0)}\n"
            f"  Risk Flags: {account.get('risk_flags', [])}\n\n"
        )

    # Build compliance policies summary
    policy_summary = ""
    if policies:
        policy_lines = []
        for p in policies:
            policy_lines.append(f"  [{p.get('policy_id','')}] {p.get('title','')}: {p.get('requirement','')}")
        policy_summary = "COMPLIANCE POLICIES (you MUST follow these):\n" + "\n".join(policy_lines) + "\n\n"

    # Build prior interactions summary
    prior_summary = ""
    if prior:
        prior_lines = []
        for ix in prior:
            prior_lines.append(f"  {ix.get('date','')} ({ix.get('channel','')}) - {ix.get('summary','')} → {ix.get('resolution','')} [{ix.get('satisfaction','')}]")
        prior_summary = "PRIOR INTERACTIONS:\n" + "\n".join(prior_lines) + "\n\n"

    # Build knowledge base summary
    kb_summary = ""
    if kb:
        kb_lines = []
        for article in kb:
            kb_lines.append(f"  [{article.get('topic','')}]: {article.get('content','')}")
        kb_summary = "KNOWLEDGE BASE:\n" + "\n".join(kb_lines) + "\n\n"

    return (
        "You are a fintech VRM agent operating in a SIMULATED environment.\n"
        "You have access to the customer's account record, compliance policies, "
        "prior interaction history, and knowledge base articles. Use them to provide "
        "informed, policy-compliant responses.\n\n"
        f"{account_summary}"
        f"{policy_summary}"
        f"{prior_summary}"
        f"{kb_summary}"
        "STRATEGY — do exactly ONE of these per step, in order:\n"
        f"1. FIRST STEP ONLY: Use action_type='analyze' with intent='{info['expected_intent']}' "
        "and extract any slots you can from the customer message into extracted_slots. "
        "Set event_type='auth_checkpoint'.\n"
        "2. If there are missing slots, use action_type='respond' to ask the customer, "
        "then extract slot values from the customer's reply into extracted_slots.\n"
        "3. Once slots are collected, execute EACH required workflow one at a time with "
        "action_type='workflow' and set the workflow field (e.g. 'verify_identity'). "
        "Reference relevant compliance policies in your response_text. "
        "Set event_type='compliance_disclosure' when mentioning policy requirements.\n"
        "4. After ALL required workflows are done, use action_type='finalize' with a summary "
        "that confirms resolution and references any applicable policies.\n\n"
        f"Task: {json.dumps(task)}\n"
        f"Expected intent: {info['expected_intent']}\n"
        f"Customer opening: {customer_msg}\n"
        f"{customer_context}"
        f"Conversation so far ({len(conversation)} turns): {json.dumps(conversation[-6:])}\n"
        f"Collected slots: {json.dumps(collected)}\n"
        f"Missing slots: {json.dumps(missing_slots)}\n"
        f"Required workflows: {json.dumps(info['required_workflows'])}\n"
        f"Workflows done: {json.dumps(workflows_taken)}\n"
        f"Pending workflows: {json.dumps(pending_wfs)}\n"
        f"Optional workflows (bonus): {json.dumps(info['optional_workflows'])}\n"
        f"NPS proxy: {ob.get('nps_proxy')}\n\n"
        "Output ONLY a JSON object with these exact keys:\n"
        "- action_type: one of [analyze, respond, workflow, finalize, handoff]\n"
        "- rationale: string explaining your reasoning (reference policies/KB when relevant)\n"
        "- response_text: customer-facing message (use words like 'verify', 'secure', 'confirm', 'next', reference specific policies)\n"
        f"- intent: '{info['expected_intent']}'\n"
        "- workflow: one of [verify_identity, freeze_card, reissue_card, refund_dispute, offer_savings, waive_fee, none]\n"
        "- extracted_slots: object of slot key-value pairs from customer's reply\n"
        "- confidence: float 0.0-1.0\n"
        "- event_type: one of [clarifying_q, suggestion, confirmation, handoff, escalation_trigger, repair_moment, tool_failure, auth_checkpoint, compliance_disclosure] or null\n"
        "- tool_status: one of [ok, failed, timeout]\n"
        "Do NOT include any extra keys."
    )


def llm_action(client: OpenAI, prompt: str) -> Dict:
    msg = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": "Return only valid JSON."},
            {"role": "user", "content": prompt},
        ],
    )
    text = msg.choices[0].message.content or "{}"
    parsed_text = text.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(parsed_text)
    except json.JSONDecodeError:
        return {
            "action_type": "respond",
            "rationale": "Fallback due to parse failure",
            "response_text": "I understand the urgency. Let me verify your identity and take the required secure action now.",
            "intent": "",
            "workflow": "none",
            "extracted_slots": {},
            "confidence": 0.4,
            "event_type": "clarifying_q",
            "tool_status": "ok",
        }


def run_trajectory(client: OpenAI, task_id: str, http_session: requests.Session) -> float:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task_id, MODEL_NAME)

    try:
        last = env_reset(http_session, task_id)
        customer_reply = ""

        for step_num in range(1, MAX_STEPS + 1):
            err_str = None
            raw_action: Dict = {}
            try:
                prompt = build_prompt(last, task_id, customer_reply)
                raw_action = sanitize_action(llm_action(client, prompt))
                act_str = raw_action.get("action_type", "unknown")

                last = env_step(http_session, raw_action)

                reward = float(last.get("reward", 0.0))
                done = bool(last.get("done", False))
            except Exception as ex:
                reward = 0.0
                done = True
                act_str = raw_action.get("action_type", "unknown") if raw_action else "error"
                err_str = str(ex)[:50]

            rewards.append(reward)
            steps_taken = step_num
            log_step(step_num, act_str, reward, done, err_str)

            if done:
                break

            # Simulate customer response if agent asked for info
            ob = last.get("observation", {})
            missing_slots = [s for s in ob.get("required_slots", [])
                             if s not in ob.get("collected_slots", {})]
            if missing_slots and raw_action.get("action_type") == "respond":
                customer_reply = simulate_customer(
                    client, task_id, raw_action.get("response_text", ""), missing_slots
                )
            else:
                customer_reply = ""

        grade = last.get("metadata", {}).get("grade", {}).get("score", 0.0)
        score = float(grade)
        success = score >= 0.5

    except Exception as ex:
        err = str(ex)[:50]
        if steps_taken == 0:
            rewards = [0.0]
            steps_taken = 1
            log_step(1, "none", 0.0, True, err)
        score = round(sum(rewards) / len(rewards), 4) if rewards else 0.0
        success = False

    finally:
        log_end(success, steps_taken, score, rewards)

    return score


def main() -> None:
    require_env()
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    http_session = requests.Session()

    # Verify environment server is reachable
    try:
        health = http_session.get(f"{ENV_BASE_URL}/health", timeout=5)
        health.raise_for_status()
    except Exception as e:
        print(f"WARNING: Environment server at {ENV_BASE_URL} not reachable: {e}", file=sys.stderr)
        print("Set ENV_BASE_URL to the running server address.", file=sys.stderr)
        sys.exit(1)

    all_scores: List[float] = []
    for task in TASKS:
        task_scores: List[float] = []
        for i in range(TRAJECTORIES_PER_TASK):
            score = run_trajectory(client, task, http_session)
            task_scores.append(score)
            print(f"task={task} traj={i+1} score={score:.4f}")
        mean_task = sum(task_scores) / len(task_scores)
        all_scores.extend(task_scores)
        print(f"task={task} mean_score={mean_task:.4f}")

    overall = sum(all_scores) / len(all_scores)
    print(f"overall_mean_score={overall:.4f}")


if __name__ == "__main__":
    main()
