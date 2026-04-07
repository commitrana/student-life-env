# app/env.py - Updated with 0-1 Normalized Rewards
import random
import uuid
from openenv.core.env_server import Environment
from app.models import StudentObservation, StudentAction, StudentState, Task
from app.reward import compute_reward  # Will also update this


class StudentEnv(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        self._state = StudentState()
        self._tasks = []
        self._energy = 100
        self._stress = 10
        self._money = 2000
        self._day = 1

    def reset(self, seed=None, episode_id=None, **kwargs) -> StudentObservation:
        """Start a new episode."""
        self._state = StudentState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            current_day=1,
            total_reward=0.0
        )
        self._day = 1
        self._energy = 100
        self._stress = 10
        self._money = 2000
        self._tasks = [
            Task(name="Assignment", deadline=3, progress=0.0, completed=False),
            Task(name="Hackathon", deadline=5, progress=0.0, completed=False),
            Task(name="Exam Prep", deadline=7, progress=0.0, completed=False),
            Task(name="Project", deadline=10, progress=0.0, completed=False),
            Task(name="Research Paper", deadline=8, progress=0.0, completed=False),
        ]

        return StudentObservation(
            done=False,
            reward=None,
            day=self._day,
            energy=self._energy,
            stress=self._stress,
            money=self._money,
            tasks=self._tasks,
            message="New semester started. Plan your tasks wisely!"
        )

    def step(self, action: StudentAction, timeout_s=None, **kwargs) -> StudentObservation:
        """Process an action and return next observation."""
        self._state.step_count += 1

        # Apply action effects
        reward = self._apply_action(action)
        self._state.total_reward += reward

        # Advance day
        self._day += 1

        # Apply deadline penalties (normalized to 0-1)
        for task in self._tasks:
            if self._day > task.deadline and not task.completed:
                reward = max(0.0, reward - 0.02)  # Was -2, now -0.02

        # Check if episode is done
        all_completed = all(t.completed for t in self._tasks)
        max_days_reached = self._day > 12
        done = all_completed or max_days_reached

        # Final reward if all tasks completed (normalized to 0-1)
        if all_completed and not max_days_reached:
            reward += 0.5  # Was 50, now 0.5
            message = f"Congratulations! You completed all tasks by day {self._day}!"
        elif max_days_reached:
            message = "Semester ended. Some tasks incomplete."
        else:
            message = self._get_status_message()

        return StudentObservation(
            done=done,
            reward=reward,
            day=self._day,
            energy=self._energy,
            stress=self._stress,
            money=self._money,
            tasks=self._tasks,
            message=message
        )

    @property
    def state(self) -> StudentState:
        return self._state

    def _apply_action(self, action: StudentAction) -> float:
        reward = 0.0

        if action.action_type == "work":
            for task in self._tasks:
                if task.name == action.task_name and not task.completed:
                    progress_gain = 0.5
                    task.progress += progress_gain
                    self._energy -= 5
                    self._stress += 2

                    print(f"   📈 {task.name} progress: {task.progress:.0%}")

                    if task.progress >= 1.0:
                        task.completed = True
                        print(f"   ✅ {task.name} COMPLETED!")
                        return 0.8  # Task completion reward (normalized)

                    reward = compute_reward(
                        progress_gain=progress_gain,
                        completed=task.completed,
                        stress=self._stress,
                        energy=self._energy
                    )
                    break

        elif action.action_type == "rest":
            self._energy = min(100, self._energy + 40)
            self._stress = max(0, self._stress - 20)
            reward = 0.1  # Was 10, now 0.1
            print(f"   😴 Rested (+0.1 reward)")

        elif action.action_type == "spend":
            self._money -= action.amount or 0
            reward = 0.0 if (action.amount or 0) > 500 else 0.01  # Normalized
            print(f"   💰 Spent ${action.amount} (reward: {reward})")

        # Clamp values
        self._energy = max(0, min(100, self._energy))
        self._stress = max(0, min(100, self._stress))
        self._money = max(0, self._money)

        return reward

    def _get_status_message(self) -> str:
        completed = sum(1 for t in self._tasks if t.completed)
        total = len(self._tasks)
        return f"Day {self._day}: {completed}/{total} tasks done. Energy: {self._energy}, Stress: {self._stress}"