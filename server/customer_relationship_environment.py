from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Set
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import (
        AccountContext,
        CRMAction,
        CRMActionType,
        CRMObservation,
        CRMWorkflowType,
        CriticalEventType,
        InteractionRecord,
        KnowledgeArticle,
        PolicyInfo,
        TaskDefinition,
        TurnUsefulness,
    )
    from .graders import CRMTaskGrader
    from .task_bank import TASKS, CRMTaskSpec
except ImportError:
    from models import (
        AccountContext,
        CRMAction,
        CRMActionType,
        CRMObservation,
        CRMWorkflowType,
        CriticalEventType,
        InteractionRecord,
        KnowledgeArticle,
        PolicyInfo,
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
        self._compliance_disclosures: int = 0
        self._auth_checkpoints: int = 0
        self._mentioned_policies: Set[str] = set()

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
        self._compliance_disclosures = 0
        self._auth_checkpoints = 0
        self._mentioned_policies = set()

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

        # Repair loop detection: structural signals, not keyword matching.
        # A repair loop is when the agent repeats the same action type as the
        # previous turn without advancing the conversation (no new slots, no
        # new workflow, same action type as last turn).
        if len(self._history) >= 3:
            prev_turns = [h for h in self._history if h["role"] == "agent"]
            if len(prev_turns) >= 2:
                # Same text repeated nearly verbatim indicates a loop
                current_text = action.response_text.strip().lower()
                prev_text = prev_turns[-2]["text"].strip().lower()
                if current_text and prev_text and (
                    current_text == prev_text
                    or (len(current_text) > 20 and current_text[:20] == prev_text[:20])
                ):
                    self._repair_loops += 1

        # Compliance-awareness tracking
        if action.event_type == CriticalEventType.COMPLIANCE_DISCLOSURE:
            self._compliance_disclosures += 1
            reward += 0.04
        if action.event_type == CriticalEventType.AUTH_CHECKPOINT:
            self._auth_checkpoints += 1
            reward += 0.03

        # Policy-aware response bonus: check if agent references policy concepts
        text_lower = action.response_text.lower()
        for policy in (self._task.compliance_policies if self._task else []):
            pid = policy.policy_id
            if pid not in self._mentioned_policies:
                # Check if agent's response shows awareness of the policy
                policy_keywords = set(policy.requirement.lower().split()) - {
                    "the", "a", "an", "is", "to", "of", "and", "or", "in", "for",
                    "must", "may", "not", "be", "that", "with", "this", "from",
                }
                # Agent gets credit for mentioning 3+ relevant keywords from a policy
                matches = sum(1 for kw in policy_keywords if len(kw) > 4 and kw in text_lower)
                if matches >= 3:
                    self._mentioned_policies.add(pid)
                    reward += 0.05

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
            compliance_disclosures=self._compliance_disclosures,
            auth_checkpoints=self._auth_checkpoints,
            policies_referenced=len(self._mentioned_policies),
            total_policies=len(self._task.compliance_policies),
            high_risk=self._task.high_risk,
        )
        data = {"score": result.score}
        data.update(result.breakdown)
        return data

    def _compute_turn_usefulness(self, action: CRMAction) -> TurnUsefulness:
        text = action.response_text.lower()
        words = action.response_text.split()
        word_count = len(words)

        # === Correctness (0-2): intent match + slot relevance + workflow appropriateness ===
        intent_correct = action.intent == (self._task.expected_intent if self._task else "")
        slots_provided = len(action.extracted_slots)
        slots_relevant = sum(
            1 for s in action.extracted_slots
            if self._task and s in self._task.required_slots
        )
        correctness = 0.5
        if intent_correct:
            correctness += 0.8
        if slots_relevant > 0:
            correctness += min(0.5, slots_relevant * 0.25)
        if action.action_type == CRMActionType.WORKFLOW and action.workflow.value != "none":
            wf = action.workflow.value
            if self._task and wf in self._task.required_workflows:
                correctness += 0.2
        correctness = min(2.0, correctness)

        # === Completeness (0-2): slots extracted + conversation progression ===
        completeness = 0.3
        completeness += min(0.7, slots_provided * 0.35)
        if action.rationale and len(action.rationale) > 10:
            completeness += 0.3
        if action.response_text and word_count > 5:
            completeness += 0.3
        # Bonus for addressing outstanding risks
        if self._task:
            outstanding = [s for s in self._task.required_slots if s not in self._collected_slots]
            if action.action_type == CRMActionType.RESPOND and outstanding:
                completeness += 0.3  # asking for missing info
        completeness = min(2.0, completeness)

        # === Clarity (0-2): response length, structure, specificity ===
        clarity = 0.4
        if 8 < word_count < 80:
            clarity += 0.6  # right length — not too terse, not rambling
        elif word_count >= 5:
            clarity += 0.3
        # Contains specific numbers, dates, or IDs (shows specificity)
        if any(c.isdigit() for c in action.response_text):
            clarity += 0.3
        # Structural signals: sentences > 1 indicate organized response
        sentence_count = max(1, action.response_text.count('.') + action.response_text.count('?'))
        if 2 <= sentence_count <= 5:
            clarity += 0.3
        clarity = min(2.0, clarity)

        # === Actionability (0-2): does the response move the episode forward? ===
        actionability = 0.4
        if action.action_type in (CRMActionType.WORKFLOW, CRMActionType.FINALIZE):
            actionability += 0.8  # concrete action
        # Asking questions or providing information moves things forward
        if action.response_text.count('?') >= 1 and action.action_type == CRMActionType.RESPOND:
            actionability += 0.4  # soliciting information
        elif action.action_type == CRMActionType.ANALYZE:
            actionability += 0.4  # diagnostic action
        if action.confidence >= 0.7:
            actionability += 0.2
        # Penalize vague/empty responses
        if word_count < 3 and action.action_type == CRMActionType.RESPOND:
            actionability = max(0.0, actionability - 0.5)
        actionability = min(2.0, actionability)

        # === Safety/Compliance (0-2): policy adherence signals ===
        safety = 0.4
        # Identity verification awareness
        if any(k in text for k in ["identity", "verify", "authenticate", "security"]):
            safety += 0.4
        # Policy/compliance references
        if any(k in text for k in ["policy", "regulation", "compliance", "disclosure", "consent"]):
            safety += 0.4
        # Proper event type usage
        if action.event_type in (CriticalEventType.AUTH_CHECKPOINT, CriticalEventType.COMPLIANCE_DISCLOSURE):
            safety += 0.4
        # Acknowledging high-risk context
        if self._task and self._task.high_risk:
            if any(k in text for k in ["urgent", "priority", "immediately", "right away", "protect"]):
                safety += 0.2
        # Penalize dangerous omissions: workflow without prior auth
        if (action.action_type == CRMActionType.WORKFLOW
                and "verify_identity" not in self._workflows
                and self._task and self._task.high_risk):
            safety = max(0.0, safety - 0.6)
        safety = min(2.0, safety)

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
            "improve turn-usefulness dimensions, and maximize session satisfaction probability. "
            "Consult the compliance_policies and knowledge_base in the observation to inform your responses."
        )
        metadata = {
            "grade": grade,
            "task_id": self._task.task_id,
            "required_workflows": sorted(self._task.required_workflows),
            "optional_workflows": sorted(self._task.optional_workflows),
            "compliance_score": {
                "disclosures": self._compliance_disclosures,
                "auth_checkpoints": self._auth_checkpoints,
                "policies_referenced": len(self._mentioned_policies),
                "total_policies": len(self._task.compliance_policies),
            },
            "evaluation_notes": {
                "offline_split": ["time_based", "customer_disjoint"],
                "target_metrics": ["spearman", "mae", "quadratic_weighted_kappa", "roc_auc", "pr_auc", "calibration_error"],
            },
        }

        # Build rich context from task spec
        profile = self._task.customer_profile
        account_ctx = AccountContext(
            customer_name=profile.name,
            tier=profile.tier,
            tenure_months=profile.tenure_months,
            monthly_revenue=profile.monthly_revenue,
            lifetime_value=profile.lifetime_value,
            open_tickets=profile.open_tickets,
            recent_nps=profile.recent_nps,
            risk_flags=list(profile.risk_flags),
        )
        policies = [
            PolicyInfo(policy_id=p.policy_id, title=p.title, requirement=p.requirement)
            for p in self._task.compliance_policies
        ]
        interactions = [
            InteractionRecord(
                date=i.date, channel=i.channel, summary=i.summary,
                resolution=i.resolution, satisfaction=i.satisfaction,
            )
            for i in self._task.prior_interactions
        ]
        kb = [
            KnowledgeArticle(topic=a["topic"], content=a["content"])
            for a in self._task.knowledge_base
        ]

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
            account_context=account_ctx,
            compliance_policies=policies,
            prior_interactions=interactions,
            knowledge_base=kb,
        )

    @property
    def state(self) -> State:
        return self._state
