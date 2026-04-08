from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import (
        CRMAction,
        CRMActionType,
        CRMObservation,
        CRMWorkflowType,
        CriticalEventType,
        TaskDefinition,
        TurnUsefulness,
    )
    from .graders import CRMTaskGrader
    from .task_bank import TASKS, CRMTaskSpec
except ImportError:
    from models import (
        CRMAction,
        CRMActionType,
        CRMObservation,
        CRMWorkflowType,
        CriticalEventType,
        TaskDefinition,
        TurnUsefulness,
    )
    from server.graders import CRMTaskGrader
    from server.task_bank import TASKS, CRMTaskSpec


class CustomerRelationshipEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = False

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task: Optional[CRMTaskSpec] = None
        self._history: List[Dict[str, str]] = []
        self._collected_slots: Dict[str, str] = {}
        self._workflows: List[str] = []
        self._used_handoff = False
        self._done = False
        self._nps_proxy = -15.0

        self._turn_usefulness_100: float = 0.0
        self._sum_usefulness_100: float = 0.0
        self._turn_count: int = 0
        self._tool_failures: int = 0
        self._repair_loops: int = 0

        self._event_deltas: Dict[str, List[float]] = defaultdict(list)
        self._last_turn_usefulness = TurnUsefulness()

    def reset(self, seed=None, episode_id=None, **kwargs) -> CRMObservation:
        task_id = kwargs.get("task_id") or "easy_card_freeze"
        if task_id not in TASKS:
            task_id = "easy_card_freeze"
        self._task = TASKS[task_id]

        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
        self._history = [{"role": "customer", "text": self._task.customer_opening}]
        self._collected_slots = {}
        self._workflows = []
        self._used_handoff = False
        self._done = False
        self._nps_proxy = -20.0 if self._task.high_risk else -10.0

        self._turn_usefulness_100 = 0.0
        self._sum_usefulness_100 = 0.0
        self._turn_count = 0
        self._tool_failures = 0
        self._repair_loops = 0
        self._event_deltas = defaultdict(list)
        self._last_turn_usefulness = TurnUsefulness()

        return self._obs(reward=0.0, done=False, done_reason=None)

    def step(self, action: CRMAction, timeout_s=None, **kwargs) -> CRMObservation:  # type: ignore[override]
        if self._task is None:
            return self.reset(**kwargs)

        if self._done:
            return self._obs(reward=0.0, done=True, done_reason="episode_already_done")

        self._state.step_count += 1
        reward = -0.01
        done_reason = None

        # action log
        self._history.append(
            {
                "role": "agent",
                "text": action.response_text or action.rationale or action.action_type.value,
            }
        )

        # Intent analysis
        if action.action_type == CRMActionType.ANALYZE:
            intent_match = action.intent == self._task.expected_intent
            reward += 0.10 if intent_match else -0.02
            self._workflows.append(action.intent if action.intent else "unknown_intent")

        # Slot extraction
        for slot, value in action.extracted_slots.items():
            if slot in self._task.required_slots and value and slot not in self._collected_slots:
                self._collected_slots[slot] = value
                reward += 0.07

        # Workflow operations
        if action.action_type == CRMActionType.WORKFLOW:
            wf = action.workflow.value
            if wf != CRMWorkflowType.NONE.value:
                self._workflows.append(wf)
                if wf in self._task.required_workflows:
                    reward += 0.10
                    self._nps_proxy += 6.0
                elif wf in self._task.optional_workflows:
                    reward += 0.04
                    self._nps_proxy += 3.0
                else:
                    reward -= 0.03

        # Handoff and finalize
        if action.action_type == CRMActionType.HANDOFF:
            self._used_handoff = True
            reward -= 0.15
            self._nps_proxy -= 8.0

        if action.action_type == CRMActionType.FINALIZE:
            self._done = True
            done_reason = "agent_finalized"
            reward += 0.05

        # Tool failure and repair tracking
        if action.tool_status in {"failed", "timeout"}:
            self._tool_failures += 1
            reward -= 0.06
        if "sorry" in action.response_text.lower() or "correct" in action.response_text.lower():
            self._repair_loops += 1

        # Turn-level usefulness formula from requirements
        turn = self._compute_turn_usefulness(action)
        self._last_turn_usefulness = turn
        self._turn_usefulness_100 = turn.normalized_usefulness
        self._sum_usefulness_100 += self._turn_usefulness_100
        self._turn_count += 1

        reward += (self._turn_usefulness_100 / 100.0 - 0.5) * 0.12

        # Critical-point event impact: delta_e = E[sat_post - sat_pre | event=e]
        sat_pre = self._session_sat_hat()
        self._nps_proxy += (self._turn_usefulness_100 / 100.0 - 0.5) * 10.0
        sat_post = self._session_sat_hat()
        if action.event_type is not None:
            self._event_deltas[action.event_type.value].append(round(sat_post - sat_pre, 4))

        # Loop/latency penalty
        if self._state.step_count >= self._task.max_steps:
            self._done = True
            done_reason = done_reason or "max_steps"
            reward -= 0.08

        # High-risk verification penalty
        if self._done and self._task.high_risk and "verify_identity" not in self._workflows:
            reward -= 0.20
            self._nps_proxy -= 15.0

        if self._done:
            grade = self._grade()
            reward += (grade["score"] - 0.5) * 0.2

        reward = max(-1.0, min(1.0, round(reward, 4)))
        return self._obs(reward=reward, done=self._done, done_reason=done_reason)

    def _avg_usefulness_100(self) -> float:
        return self._sum_usefulness_100 / max(1, self._turn_count)

    def _session_sat_hat(self) -> float:
        assert self._task is not None
        slot_hits = sum(1 for s in self._task.required_slots if self._collected_slots.get(s))
        slot_score = slot_hits / max(1, len(self._task.required_slots))
        wf_hits = len(set(self._workflows).intersection(self._task.required_workflows))
        workflow_score = wf_hits / max(1, len(self._task.required_workflows))
        task_success = 0.40 * slot_score + 0.45 * workflow_score + 0.15 * (1.0 if self._done else 0.0)
        return CRMTaskGrader.session_satisfaction_hat(
            task_success=task_success,
            avg_usefulness_01=self._avg_usefulness_100() / 100.0,
            cost=min(1.0, self._state.step_count / max(1, self._task.max_steps)),
            tool_failures=min(1.0, self._tool_failures / 3.0),
            repair_loops=min(1.0, self._repair_loops / 3.0),
            handoff_penalty=1.0 if self._used_handoff else 0.0,
        )

    def _grade(self) -> Dict[str, float]:
        assert self._task is not None
        result = CRMTaskGrader.grade(
            expected_intent=self._task.expected_intent,
            required_slots=self._task.required_slots,
            collected_slots=self._collected_slots,
            required_workflows=self._task.required_workflows,
            optional_workflows=self._task.optional_workflows,
            workflows_taken=self._workflows,
            finalized=self._done,
            used_handoff=self._used_handoff,
            step_count=self._state.step_count,
            max_steps=self._task.max_steps,
            usefulness_100=self._avg_usefulness_100(),
            tool_failures=self._tool_failures,
            repair_loops=self._repair_loops,
        )
        data = {"score": result.score}
        data.update(result.breakdown)
        return data

    def _compute_turn_usefulness(self, action: CRMAction) -> TurnUsefulness:
        text = action.response_text.lower()
        correctness = 1.8 if action.intent == (self._task.expected_intent if self._task else "") else 1.0
        completeness = min(2.0, 0.5 + 0.5 * len(action.extracted_slots))
        clarity = 1.6 if len(action.response_text.split()) > 8 else 1.1
        actionability = 1.7 if any(k in text for k in ["next", "please", "confirm", "verify"]) else 1.0
        safety = 1.8 if any(k in text for k in ["secure", "identity", "policy", "consent"]) else 1.0

        usefulness = CRMTaskGrader.normalized_usefulness(
            correctness=correctness,
            completeness=completeness,
            clarity=clarity,
            actionability=actionability,
            safety=safety,
        )
        return TurnUsefulness(
            correctness=round(correctness, 4),
            completeness=round(completeness, 4),
            clarity=round(clarity, 4),
            actionability=round(actionability, 4),
            safety=round(safety, 4),
            normalized_usefulness=round(usefulness, 4),
        )

    def _critical_event_impacts(self) -> Dict[str, float]:
        impacts: Dict[str, float] = {}
        for event, deltas in self._event_deltas.items():
            impacts[event] = round(sum(deltas) / len(deltas), 4) if deltas else 0.0
        return impacts

    def _obs(self, reward: float, done: bool, done_reason: Optional[str]) -> CRMObservation:
        assert self._task is not None
        grade = self._grade()
        outstanding = [s for s in self._task.required_slots if s not in self._collected_slots]
        guidance = (
            "Use structured critical events (clarify/suggest/confirm/handoff/escalation/repair/tool_failure/auth/compliance), "
            "improve turn-usefulness dimensions, and maximize session satisfaction probability."
        )
        metadata = {
            "grade": grade,
            "task_id": self._task.task_id,
            "required_workflows": sorted(self._task.required_workflows),
            "optional_workflows": sorted(self._task.optional_workflows),
            "evaluation_notes": {
                "offline_split": ["time_based", "customer_disjoint"],
                "target_metrics": ["spearman", "mae", "quadratic_weighted_kappa", "roc_auc", "pr_auc", "calibration_error"],
            },
        }
        return CRMObservation(
            task=TaskDefinition(
                task_id=self._task.task_id,
                difficulty=self._task.difficulty,
                title=self._task.title,
                objective=self._task.objective,
                max_steps=self._task.max_steps,
            ),
            customer_message=self._task.customer_opening,
            conversation_history=self._history,
            required_slots=self._task.required_slots,
            collected_slots=self._collected_slots,
            outstanding_risks=outstanding,
            workflow_actions_taken=self._workflows,
            nps_proxy=max(-100.0, min(100.0, round(self._nps_proxy, 2))),
            progress_score=grade["score"],
            turn_usefulness=self._last_turn_usefulness,
            session_satisfaction_hat=round(self._session_sat_hat(), 4),
            critical_event_impacts=self._critical_event_impacts(),
            guidance=guidance,
            done_reason=done_reason,
            done=done,
            reward=reward,
            metadata=metadata,
        )

    @property
    def state(self) -> State:
        return self._state
