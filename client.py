# app/client.py
from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from app.models import StudentAction, StudentObservation, StudentState


class StudentEnvClient(EnvClient[StudentAction, StudentObservation, StudentState]):
    def _step_payload(self, action: StudentAction) -> dict:
        """Convert typed action to JSON payload."""
        return {
            "action_type": action.action_type,
            "task_name": action.task_name,
            "hours": action.hours,
            "amount": action.amount,
        }

    def _parse_result(self, payload: dict) -> StepResult:
        """Parse JSON response into StepResult."""
        obs_data = payload.get("observation", {})
        return StepResult(
            observation=StudentObservation(
                done=payload.get("done", False),
                reward=payload.get("reward"),
                day=obs_data.get("day", 1),
                energy=obs_data.get("energy", 100),
                stress=obs_data.get("stress", 10),
                money=obs_data.get("money", 2000),
                tasks=obs_data.get("tasks", []),
                message=obs_data.get("message", ""),
            ),
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> StudentState:
        """Parse JSON into State object."""
        return StudentState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            current_day=payload.get("current_day", 1),
            total_reward=payload.get("total_reward", 0.0),
        )