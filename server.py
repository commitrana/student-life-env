# server.py - Simplified HTTP only (no WebSocket)
from fastapi import FastAPI
from app.env import StudentEnv
from app.models import StudentAction, StudentObservation

app = FastAPI()
env = StudentEnv()

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/reset")
async def reset():
    obs = env.reset()
    return {
        "observation": obs.model_dump(),
        "done": obs.done,
        "reward": obs.reward
    }

@app.post("/step")
async def step(action: StudentAction):
    obs = env.step(action)
    return {
        "observation": obs.model_dump(),
        "done": obs.done,
        "reward": obs.reward
    }

@app.get("/state")
async def state():
    return env.state.model_dump()

@app.get("/")
def root():
    return {"status": "healthy", "message": "Server is running"}