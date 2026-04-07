import requests

BASE_URL = "http://localhost:8000"

print("=" * 50)
print("DEBUGGING TASK PROGRESS")
print("=" * 50)

# Reset
print("\n1. Resetting environment...")
r = requests.post(f"{BASE_URL}/reset")
data = r.json()
obs = data.get("observation", {})
tasks = obs.get("tasks", [])
print(f"   Initial tasks: {[(t.get('name'), t.get('progress')) for t in tasks]}")

# Work on Assignment
print("\n2. Working on Assignment...")
r = requests.post(f"{BASE_URL}/step", json={"action_type": "work", "task_name": "Assignment"})
data = r.json()
obs = data.get("observation", {})
tasks = obs.get("tasks", [])
print(f"   After step 1: {[(t.get('name'), t.get('progress')) for t in tasks]}")
print(f"   Reward: {data.get('reward')}")

# Work on Assignment again
print("\n3. Working on Assignment again...")
r = requests.post(f"{BASE_URL}/step", json={"action_type": "work", "task_name": "Assignment"})
data = r.json()
obs = data.get("observation", {})
tasks = obs.get("tasks", [])
print(f"   After step 2: {[(t.get('name'), t.get('progress')) for t in tasks]}")
print(f"   Reward: {data.get('reward')}")

# Check final state
print("\n4. Final state:")
r = requests.get(f"{BASE_URL}/state")
print(f"   {r.json()}")