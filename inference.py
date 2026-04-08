"""
MANDATORY ENV VARS
- API_BASE_URL
- MODEL_NAME
- HF_TOKEN

Optional fallback
- OPENAI_API_KEY
- ENV_BASE_URL (default: http://localhost:8000)
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

import requests
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:8000")

MAX_STEPS = 10
TRAJECTORIES_PER_TASK = 3
TEMPERATURE = 0.2

TASKS = [
    "easy_card_freeze",
    "medium_dispute_retention",
    "hard_business_churn_prevention",
]


def require_env() -> None:
    missing = []
    if not MODEL_NAME:
        missing.append("MODEL_NAME")
    if not API_KEY:
        missing.append("HF_TOKEN or OPENAI_API_KEY")
    if missing:
        raise RuntimeError(f"Missing required env vars: {', '.join(missing)}")


def call_reset(task_id: str) -> Dict:
    r = requests.post(f"{ENV_BASE_URL}/reset", json={"task_id": task_id}, timeout=30)
    r.raise_for_status()
    return r.json()


def call_step(action: Dict) -> Dict:
    r = requests.post(f"{ENV_BASE_URL}/step", json=action, timeout=30)
    r.raise_for_status()
    return r.json()


def build_prompt(obs: Dict) -> str:
    ob = obs.get("observation", {})
    task = ob.get("task", {})
    return (
        "You are a fintech VRM agent. Produce one JSON action. "
        "Priorities: gather missing slots, execute required workflows, then finalize.\n"
        f"Task: {json.dumps(task)}\n"
        f"Collected slots: {json.dumps(ob.get('collected_slots', {}))}\n"
        f"Required slots: {json.dumps(ob.get('required_slots', []))}\n"
        f"Workflows taken: {json.dumps(ob.get('workflow_actions_taken', []))}\n"
        f"Outstanding risks: {json.dumps(ob.get('outstanding_risks', []))}\n"
        f"NPS proxy: {ob.get('nps_proxy')}\n"
        "Output JSON with keys: action_type, rationale, response_text, intent, workflow, extracted_slots, confidence, event_type, tool_status"
    )


def llm_action(client: OpenAI, prompt: str) -> Dict:
    msg = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        max_tokens=280,
        messages=[
            {"role": "system", "content": "Return only valid JSON."},
            {"role": "user", "content": prompt},
        ],
    )
    text = msg.choices[0].message.content or "{}"
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {
            "action_type": "respond",
            "rationale": "Fallback due to parse failure",
            "response_text": "I understand the urgency. I will verify your identity and take the required secure action now.",
            "intent": "",
            "workflow": "none",
            "extracted_slots": {},
            "confidence": 0.4,
            "event_type": "clarifying_q",
            "tool_status": "ok",
        }


def run_trajectory(client: OpenAI, task_id: str) -> float:
    obs = call_reset(task_id)
    last = obs
    for _ in range(MAX_STEPS):
        prompt = build_prompt(last)
        action = llm_action(client, prompt)
        last = call_step(action)
        if last.get("done", False):
            break

    grade = (
        last.get("observation", {})
        .get("metadata", {})
        .get("grade", {})
        .get("score", 0.0)
    )
    return float(grade)


def main() -> None:
    require_env()
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    all_scores: List[float] = []
    for task in TASKS:
        task_scores: List[float] = []
        for i in range(TRAJECTORIES_PER_TASK):
            score = run_trajectory(client, task)
            task_scores.append(score)
            print(f"task={task} traj={i+1} score={score:.4f}")
        mean_task = sum(task_scores) / len(task_scores)
        all_scores.extend(task_scores)
        print(f"task={task} mean_score={mean_task:.4f}")

    overall = sum(all_scores) / len(all_scores)
    print(f"overall_mean_score={overall:.4f}")


if __name__ == "__main__":
    main()
