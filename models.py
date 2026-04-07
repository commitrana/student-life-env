# app/models.py
from pydantic import BaseModel
from typing import List, Optional
from openenv.core.env_server import Action, Observation, State  # ← ADD THIS

class Task(BaseModel):
    name: str
    deadline: int
    progress: float
    completed: bool = False

class StudentObservation(Observation):  # ← CHANGE: was Observation, now StudentObservation
    day: int
    energy: int
    stress: int
    money: int
    tasks: List[Task]
    message: Optional[str] = None

class StudentAction(Action):  # ← CHANGE: was Action, now StudentAction
    action_type: str
    task_name: Optional[str] = None
    hours: Optional[int] = None
    amount: Optional[int] = None

class StudentState(State):  # ← NEW: add this entire class
    episode_id: Optional[str] = None
    step_count: int = 0
    current_day: int = 1
    total_reward: float = 0.0