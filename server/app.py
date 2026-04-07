# server/app.py
from fastapi import FastAPI
import uvicorn
import os
from app.env import StudentEnv
from app.models import StudentAction, StudentObservation

app = FastAPI()
env = StudentEnv()

@app.get("/")
async def root():
    return {"status": "healthy", "message": "Server is running"}

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

# ✅ ADD THIS - OpenEnv expects this
def main():
    """Main entry point for OpenEnv server"""
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)

# ✅ ADD THIS - For direct execution
if __name__ == "__main__":
    main()