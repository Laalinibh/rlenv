from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Dict, List, Set


@dataclass
class GradeResult:
    score: float
    breakdown: Dict[str, float]


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + exp(-x))


class CRMTaskGrader:
    """Deterministic grader implementing usefulness + satisfaction framework."""

    DEFAULT_WEIGHTS = {
        "w_c": 2.0,
        "w_cp": 1.0,
        "w_cl": 1.0,
        "w_a": 1.0,
        "w_s": 2.0,
    }

    @staticmethod
    def normalized_usefulness(
        correctness: float,
        completeness: float,
        clarity: float,
        actionability: float,
        safety: float,
        weights: Dict[str, float] | None = None,
    ) -> float:
        # U_t = 100 * (w_c*C + w_cp*P + w_cl*L + w_a*A + w_s*S) / (2*(sum weights))
        w = weights or CRMTaskGrader.DEFAULT_WEIGHTS
        numerator = (
            w["w_c"] * correctness
            + w["w_cp"] * completeness
            + w["w_cl"] * clarity
            + w["w_a"] * actionability
            + w["w_s"] * safety
        )
        denominator = 2.0 * (w["w_c"] + w["w_cp"] + w["w_cl"] + w["w_a"] + w["w_s"])
        if denominator <= 0:
            return 0.0
        return max(0.0, min(100.0, 100.0 * numerator / denominator))

    @staticmethod
    def session_satisfaction_hat(
        task_success: float,
        avg_usefulness_01: float,
        cost: float,
        tool_failures: float,
        repair_loops: float,
        handoff_penalty: float,
    ) -> float:
        # p_hat_sat = sigma(beta0 + beta1*TaskSuccess + beta2*avg(U) - beta3*Cost - beta4*ToolFailures
        #                    - beta5*RepairLoops - beta6*HandoffPenalty)
        beta0, beta1, beta2, beta3, beta4, beta5, beta6 = (-1.0, 2.2, 2.0, 1.1, 1.0, 0.7, 1.3)
        x = (
            beta0
            + beta1 * task_success
            + beta2 * avg_usefulness_01
            - beta3 * cost
            - beta4 * tool_failures
            - beta5 * repair_loops
            - beta6 * handoff_penalty
        )
        return max(0.0, min(1.0, sigmoid(x)))

    @staticmethod
    def grade(
        *,
        expected_intent: str,
        required_slots: List[str],
        collected_slots: Dict[str, str],
        required_workflows: Set[str],
        optional_workflows: Set[str],
        workflows_taken: List[str],
        finalized: bool,
        used_handoff: bool,
        step_count: int,
        max_steps: int,
        usefulness_100: float,
        tool_failures: int,
        repair_loops: int,
    ) -> GradeResult:
        intent_score = 1.0 if expected_intent and expected_intent in workflows_taken[0:1] else 0.0

        slot_hits = sum(1 for s in required_slots if collected_slots.get(s))
        slot_score = slot_hits / max(1, len(required_slots))

        wf_set = set(workflows_taken)
        req_hits = len(required_workflows.intersection(wf_set))
        workflow_score = req_hits / max(1, len(required_workflows))

        optional_bonus = 0.0
        if optional_workflows:
            optional_bonus = min(0.08, len(optional_workflows.intersection(wf_set)) * 0.04)

        task_success = 0.40 * slot_score + 0.45 * workflow_score + 0.15 * (1.0 if finalized else 0.0)
        efficiency = max(0.0, 1.0 - (step_count / max(1, max_steps)) * 0.6)
        handoff_penalty = 1.0 if used_handoff else 0.0

        sat_hat = CRMTaskGrader.session_satisfaction_hat(
            task_success=task_success,
            avg_usefulness_01=usefulness_100 / 100.0,
            cost=min(1.0, step_count / max(1, max_steps)),
            tool_failures=min(1.0, tool_failures / 3.0),
            repair_loops=min(1.0, repair_loops / 3.0),
            handoff_penalty=handoff_penalty,
        )

        score = (
            0.18 * intent_score
            + 0.24 * slot_score
            + 0.23 * workflow_score
            + 0.17 * sat_hat
            + 0.10 * (usefulness_100 / 100.0)
            + 0.08 * efficiency
            + optional_bonus
            - 0.06 * handoff_penalty
        )
        score = max(0.0, min(1.0, round(score, 4)))

        return GradeResult(
            score=score,
            breakdown={
                "intent": round(intent_score, 4),
                "slots": round(slot_score, 4),
                "workflow": round(workflow_score, 4),
                "task_success": round(task_success, 4),
                "usefulness_100": round(usefulness_100, 4),
                "sat_hat": round(sat_hat, 4),
                "efficiency": round(efficiency, 4),
                "tool_failures": float(tool_failures),
                "repair_loops": float(repair_loops),
                "handoff_penalty": float(handoff_penalty),
                "optional_bonus": round(optional_bonus, 4),
            },
        )
