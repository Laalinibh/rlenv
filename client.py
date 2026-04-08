from __future__ import annotations

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import CRMAction, CRMObservation


class CustomerRelationshipEnv(EnvClient[CRMAction, CRMObservation, State]):
    def _step_payload(self, action: CRMAction) -> Dict:
        return action.model_dump()

    def _parse_result(self, payload: Dict) -> StepResult[CRMObservation]:
        obs_data = payload.get("observation", {})
        obs = CRMObservation.model_validate(
            {
                **obs_data,
                "done": payload.get("done", False),
                "reward": payload.get("reward", 0.0),
            }
        )
        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
