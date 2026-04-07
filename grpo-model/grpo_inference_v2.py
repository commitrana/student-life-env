# grpo_inference_v2.py - GRPO Model with Original Pipeline
import torch
import requests
import json
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
# Add randomness to decisions
import random

def get_action_from_grpo(observation):
    # 20% chance of random action (exploration)
    if random.random() < 0.2:
        actions = ["work|Assignment", "work|Hackathon", "rest", "spend|200"]
        chosen = random.choice(actions)
        print(f"   🎲 Exploring: {chosen}")
        # ... parse and return
# Model path
MODEL_PATH = "student-grpo-model"
BASE_URL = "http://localhost:8000"

print("=" * 60)
print("🎓 GRPO TRAINED STUDENT AGENT")
print("=" * 60)

# Load GRPO Model
print("Loading GRPO model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"📍 Using device: {device}")

base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(base_model, MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
print("✅ GRPO Model loaded!")


def get_action_from_grpo(observation):
    """GRPO model se action lena - clean output ke liye"""

    # Create clean prompt
    prompt = f"""Current status:
Day {observation.get('day', 1)} | Energy: {observation.get('energy', 100)} | Stress: {observation.get('stress', 10)}
Tasks:"""

    for task in observation.get('tasks', []):
        if not task.get('completed', False):
            prompt += f"\n- {task.get('name')}: {task.get('progress', 0) * 100:.0f}% done, Deadline: Day {task.get('deadline')}"

    prompt += """

Choose action. Reply EXACTLY: work|Assignment or work|Hackathon or rest or spend|200
Action:"""

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            temperature=1.2,
            do_sample=True,
            top_p=0.9,
            top_k=50,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"   🤖 Raw: {response[:80]}")  # Debug

    # BETTER PARSING - Force correct task names
    response_lower = response.lower()

    # Check for work action
    if "work" in response_lower:
        # Always use correct task names
        # Pehle Assignment complete karo, phir Hackathon
        for task in observation.get('tasks', []):
            if task.get('name') == "Assignment" and not task.get('completed', False):
                return {"action_type": "work", "task_name": "Assignment", "amount": None}
            elif task.get('name') == "Hackathon" and not task.get('completed', False):
                return {"action_type": "work", "task_name": "Hackathon", "amount": None}
            elif task.get('name') == "Exam Prep" and not task.get('completed', False):
                return {"action_type": "work", "task_name": "Exam Prep", "amount": None}
            elif task.get('name') == "Research Paper" and not task.get('completed', False):
                return {"action_type": "work", "task_name": "Research Paper", "amount": None}
            elif task.get('name') == "Project" and not task.get('completed', False):
                return {"action_type": "work", "task_name": "Project", "amount": None}

        # Fallback
        return {"action_type": "work", "task_name": "Assignment", "amount": None}

    # Check for rest
    elif "rest" in response_lower:
        return {"action_type": "rest", "task_name": None, "amount": None}

    # Check for spend
    elif "spend" in response_lower:
        return {"action_type": "spend", "task_name": None, "amount": 200}

    # Default - work on incomplete task
    for task in observation.get('tasks', []):
        if not task.get('completed', False):
            return {"action_type": "work", "task_name": task.get('name'), "amount": None}

    return {"action_type": "work", "task_name": "Assignment", "amount": None}


# Main loop - Exactly like inference.py
print("\n🔌 Connecting to environment server...")
print("=" * 60)

try:
    # Reset environment
    response = requests.post(f"{BASE_URL}/reset")
    data = response.json()
    obs = data.get("observation", {})

    print(f"📚 {obs.get('message', 'Starting new semester!')}")
    print(
        f"   Day {obs.get('day', 1)} | Energy: {obs.get('energy', 100)} | Stress: {obs.get('stress', 10)} | Money: ${obs.get('money', 2000)}")
    print()

    total_reward = 0
    step = 0
    done = False

    while not done and step < 12:
        # Get action from GRPO model
        action = get_action_from_grpo(obs)

        # Display action
        action_desc = f"{action['action_type']}"
        if action.get('task_name'):
            action_desc += f" on {action['task_name']}"
        if action.get('amount'):
            action_desc += f" (${action['amount']})"

        print(f"\n📍 Step {step + 1}: {action_desc}")

        # Send to environment
        step_response = requests.post(f"{BASE_URL}/step", json=action)
        step_data = step_response.json()

        reward = step_data.get("reward", 0)
        done = step_data.get("done", False)
        obs = step_data.get("observation", {})

        total_reward += reward if reward else 0

        print(f"   ➕ Reward: {reward}")
        print(f"   💬 {obs.get('message', '')}")
        print(f"   📊 Day {obs.get('day', 1)} | Energy: {obs.get('energy', 100)} | Stress: {obs.get('stress', 10)}")

        step += 1

    print("\n" + "=" * 60)
    print(f"🎯 FINAL TOTAL REWARD: {total_reward}")
    print(f"✅ Episode completed in {step} steps")
    print("=" * 60)

    # No memory file created! This is pure GRPO
    print("\n🧠 Note: This uses GRPO trained model, NOT memory-based learning!")
    print("   No learning_history.json was created or used.")

except requests.exceptions.ConnectionError:
    print("❌ Server not running! Start with: uvicorn server:app --reload --port 8000")
except Exception as e:
    print(f"❌ Error: {e}")