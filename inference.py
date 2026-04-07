# inference.py - WITH MEMORY & LEARNING
import os
import requests
import json

from datetime import datetime

API_KEY = os.getenv("HF_TOKEN")
BASE_URL = "http://localhost:8000"

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
# ============================================
# LEARNING MEMORY SYSTEM
# ============================================
class LearningMemory:
    def __init__(self):
        self.previous_action = None
        self.previous_reward = 0
        self.episode_history = []
        self.best_strategy = {}
        self.load_history()

    def load_history(self):
        """Load past learning from file"""
        if os.path.exists("learning_history.json"):
            try:
                with open("learning_history.json", "r") as f:
                    data = json.load(f)
                    self.episode_history = data.get('episodes', [])
                    self.best_strategy = data.get('best_strategy', {})
                    print(f"📚 Loaded {len(self.episode_history)} past episodes of learning!")
                    if self.episode_history:
                        best_score = max([e.get('score', 0) for e in self.episode_history])
                        print(f"🏆 Best score ever: {best_score}")
            except:
                print("📚 No past learning found. Starting fresh!")

    def save_history(self):
        """Save learning to file"""
        data = {
            'episodes': self.episode_history[-20:],  # Keep last 20 episodes
            'best_strategy': self.best_strategy
        }
        with open("learning_history.json", "w") as f:
            json.dump(data, f, indent=2)

    def get_feedback(self, current_reward):
        """Generate feedback based on previous action's reward"""
        if self.previous_action is None:
            return "🎯 First decision of this episode. Make it count!"

        # Special case for task completion (reward > 80)
        if current_reward > 80:
            return f"🎉 AMAZING! You completed a task! +{current_reward} points! Great strategy!"

        diff = current_reward - self.previous_reward

        if diff > 50:
            return f"🎉 EXCELLENT! '{self.previous_action}' gave +{diff} points!"
        elif diff > 20:
            return f"👍 GOOD! '{self.previous_action}' gave +{diff} points"
        elif diff > 0:
            return f"✅ OK. '{self.previous_action}' gave +{diff} points"
        elif diff < -20:
            return f"⚠️ BAD! '{self.previous_action}' lost {abs(diff)} points. Try something different!"
        elif diff < 0:
            return f"⚠️ Not great. '{self.previous_action}' lost {abs(diff)} points"
        else:
            return f"🤔 '{self.previous_action}' gave {self.previous_reward} points"

    def update(self, action, reward):
        """Store last action for feedback"""
        action_str = f"{action['action_type']}"
        if action.get('task_name'):
            action_str += f" on {action['task_name']}"
        if action.get('amount'):
            action_str += f" (${action['amount']})"

        self.previous_action = action_str
        self.previous_reward = reward

    def learn_from_step(self, step_num, action, reward, observation):
        """Learn what works best for each situation"""
        day = observation.get('day', 1)
        action_key = f"day_{day}_{action['action_type']}"
        if action.get('task_name'):
            action_key += f"_{action['task_name']}"

        if action_key not in self.best_strategy:
            self.best_strategy[action_key] = {'total': 0, 'count': 0, 'best': -999}

        self.best_strategy[action_key]['total'] += reward
        self.best_strategy[action_key]['count'] += 1
        if reward > self.best_strategy[action_key]['best']:
            self.best_strategy[action_key]['best'] = reward

    def end_episode(self, steps_data, total_reward):
        """Store episode results and learn"""
        episode = {
            'timestamp': str(datetime.now()),
            'score': total_reward,
            'steps': len(steps_data),
            'actions': steps_data
        }
        self.episode_history.append(episode)

        # Calculate improvement
        best_score = max([e.get('score', 0) for e in self.episode_history])
        improvement = total_reward - best_score if total_reward != best_score else 0

        self.save_history()

        print("\n" + "=" * 50)
        print("📚 EPISODE LEARNING SUMMARY")
        print("=" * 50)
        print(f"   This episode score: {total_reward}")
        print(f"   Best score ever: {best_score}")
        if improvement > 0:
            print(f"   🎉 IMPROVEMENT: +{improvement} points!")
        elif improvement < 0:
            print(f"   📉 This episode was {abs(improvement)} points below best.")
        print(f"   Total episodes learned: {len(self.episode_history)}")

        # Show best strategy learned so far
        if self.best_strategy:
            print("\n   🧠 BEST STRATEGIES LEARNED:")
            sorted_strategies = sorted(self.best_strategy.items(),
                                       key=lambda x: x[1]['best'], reverse=True)[:3]
            for strategy, stats in sorted_strategies:
                if stats['count'] > 0:
                    avg = stats['total'] / stats['count']
                    print(f"      - {strategy}: avg {avg:.1f} | best {stats['best']}")

        print("=" * 50)

        return improvement > 0


# ============================================
# LLM SETUP
# ============================================
USE_LLM = API_KEY is not None and API_KEY.startswith("hf_")

if USE_LLM:
    from openai import OpenAI

    MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=API_KEY
    )
    print(f"✅ LLM ACTIVE: Using {MODEL_NAME}")


    def get_action_from_llm(observation, feedback, best_hint):
        """Use LLM to decide action with feedback and learning"""

        # Format tasks with urgency
        tasks_text = ""
        current_day = observation.get('day', 1)

        for task in observation.get('tasks', []):
            status = "✅ DONE" if task.get('completed') else f"Progress: {task.get('progress', 0) * 100:.0f}%"
            deadline = task.get('deadline', '?')
            days_left = deadline - current_day

            if task.get('completed'):
                urgency = "✅ COMPLETED"
            elif days_left <= 2:
                urgency = "🔴🔴 URGENT - MUST DO NOW!"
            elif days_left <= 4:
                urgency = "🟡 Priority - Do soon"
            else:
                urgency = "🟢 Can wait"

            tasks_text += f"  - {task.get('name')}: {status}, Deadline: Day {deadline} ({urgency})\n"

        prompt = f"""You are a student maximizing your final score. LEARN from past feedback!

FEEDBACK FROM LAST ACTION:
{feedback}

{best_hint}

CURRENT STATUS:
📅 DAY: {observation.get('day', 1)}/12
⚡ ENERGY: {observation.get('energy', 100)}/100
😰 STRESS: {observation.get('stress', 10)}/100
💰 MONEY: ${observation.get('money', 2000)}

📋 TASKS:
{tasks_text}

⚡ ACTION EFFECTS:
- work [task] → Progress +50%, Energy -5, Stress +2, Reward ~0.15 then 0.8 on completion
- rest → Energy +40, Stress -20, Reward +0.1
- spend [amount] → Reward -0.01 if >$500, +0.01 if ≤$500

🎯 RULES:
- Completing a task gives +0.8 bonus!
- Deadline penalty: -0.02 per overdue task per day
- Complete ALL tasks before Day 12 for +0.5 bonus!


🎯 SMART STRATEGY (Learn from past):
1. 🔴 URGENT tasks FIRST (deadline in 2 days or less)
2. If energy < 40 → REST (can't work when tired)
3. If stress > 70 → REST (need break)
4. Tasks at 50% progress need ONE more work to complete (+105 reward!)
5. Complete tasks in order of deadline: Assignment(Day3) → Hackathon(Day5) → Research(Day8) → Exam(Day7) → Project(Day10)

Choose BEST action based on the feedback above.
Respond EXACTLY in this format:
work|TaskName
rest
spend|amount

Your action:"""

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=30,
                temperature=0.8  # Slight randomness for exploration
            )

            result = response.choices[0].message.content.strip()
            print(f"   🤖 LLM Said: {result}")

            if "|" in result:
                parts = result.split("|")
                action_type = parts[0].lower().strip()
                if action_type == "work":
                    task_name = parts[1].strip()
                    return {"action_type": "work", "task_name": task_name, "amount": None}
                elif action_type == "spend":
                    amount = int(parts[1].strip())
                    return {"action_type": "spend", "task_name": None, "amount": amount}
            elif "rest" in result.lower():
                return {"action_type": "rest", "task_name": None, "amount": None}
            elif "spend" in result.lower():
                return {"action_type": "spend", "task_name": None, "amount": 200}

            return {"action_type": "work", "task_name": "Assignment", "amount": None}

        except Exception as e:
            print(f"   ⚠️ LLM Error: {e}, using fallback")
            return get_fallback_action(observation)

else:
    print("⚠️ No valid HF_TOKEN found. Using heuristic actions.")


    def get_action_from_llm(observation, feedback, best_hint):
        return get_fallback_action(observation)


def get_fallback_action(observation):
    """Fallback heuristic"""
    energy = observation.get('energy', 100)
    stress = observation.get('stress', 10)
    tasks = observation.get('tasks', [])
    day = observation.get('day', 1)

    if energy < 40 or stress > 70:
        return {"action_type": "rest", "task_name": None, "amount": None}

    incomplete = [t for t in tasks if not t.get('completed', False)]
    incomplete.sort(key=lambda x: x.get('deadline', 99))

    # Prioritize tasks at 50% progress
    almost_done = [t for t in incomplete if t.get('progress', 0) >= 0.45]
    if almost_done:
        return {"action_type": "work", "task_name": almost_done[0].get('name'), "amount": None}

    if incomplete:
        return {"action_type": "work", "task_name": incomplete[0].get('name'), "amount": None}

    return {"action_type": "rest", "task_name": None, "amount": None}


def get_best_action_hint(learning_memory, day):
    """Generate hint from best strategies learned"""
    best_hint = ""
    best_actions = []

    for strategy, stats in learning_memory.best_strategy.items():
        if f"day_{day}" in strategy and stats['count'] > 0:
            avg = stats['total'] / stats['count']
            if avg > 50:  # Only show good strategies
                best_actions.append((strategy, avg))

    if best_actions:
        best_actions.sort(key=lambda x: x[1], reverse=True)
        best_hint = f"\n💡 LEARNED INSIGHT: On Day {day}, the best action was '{best_actions[0][0]}' (avg {best_actions[0][1]:.0f} points). Consider similar strategy!"

    return best_hint


# ============================================
# MAIN FUNCTION
# ============================================
def main():
    print(f"[START] task=student env=openenv model={MODEL_NAME}")
    print("🔌 Connecting to environment server...")
    print("=" * 50)

    # Initialize learning memory
    learner = LearningMemory()

    try:
        response = requests.post(f"{BASE_URL}/reset")
        response.raise_for_status()
        data = response.json()
        obs = data.get("observation", {})
        print(f"📚 {obs.get('message', 'Starting new semester!')}")
        print(
            f"   Day {obs.get('day', 1)} | Energy: {obs.get('energy', 100)} | Stress: {obs.get('stress', 10)} | Money: ${obs.get('money', 2000)}")
        print()

        total_reward = 0
        step = 0
        done = False
        steps_data = []

        while not done and step < 12:
            # Get feedback from previous action
            feedback = learner.get_feedback(step) if step > 0 else "This is your first action."

            # Get hint from best strategies
            current_day = obs.get('day', 1)
            best_hint = get_best_action_hint(learner, current_day)

            # Get action with learning feedback
            action = get_action_from_llm(obs, feedback, best_hint)

            action_desc = f"{action['action_type']}"
            if action.get('task_name'):
                action_desc += f" on {action['task_name']}"
            if action.get('amount'):
                action_desc += f" (${action['amount']})"

            print(f"\n📍 Step {step + 1}: {action_desc}")

            step_response = requests.post(f"{BASE_URL}/step", json=action)
            step_response.raise_for_status()
            step_data = step_response.json()

            reward = step_data.get("reward", 0)
            done = step_data.get("done", False)
            obs = step_data.get("observation", {})

            total_reward += reward if reward else 0

            # Store step data for learning
            steps_data.append({
                'step': step + 1,
                'action': action,
                'reward': reward,
                'day': obs.get('day', 1)
            })
            print(
                f"[STEP] step={step + 1} action={action_desc} reward={reward:.2f} done={str(done).lower()} error=null")

            # Learn from this step
            learner.learn_from_step(step + 1, action, reward, obs)

            # Update learner with this action for next feedback
            learner.update(action, reward)

            print(f"   ➕ Reward: {reward}")
            print(f"   💡 {feedback}")
            print(
                f"   📊 Day {obs.get('day', 1)} | Energy: {obs.get('energy', 100)} | Stress: {obs.get('stress', 10)} | Money: ${obs.get('money', 2000)}")

            step += 1

        success = total_reward > 0
        rewards_str = ",".join([f"{r:.2f}" for r in [step_data.get('reward', 0) for step_data in steps_data]])
        print(f"[END] success={str(success).lower()} steps={step} rewards={rewards_str}")

        print("\n" + "=" * 50)
        print(f"🎯 FINAL TOTAL REWARD: {total_reward}")
        print(f"✅ Episode completed in {step} steps")

        # Learn from complete episode
        improved = learner.end_episode(steps_data, total_reward)

        if improved:
            print("\n🎉 The agent LEARNED and IMPROVED from this episode!")
        else:
            print("\n🤔 Keep running more episodes - the agent will learn better strategies!")

        print("=" * 50)

    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server!")
        print("   Make sure the server is running: uvicorn server:app --reload --port 8000")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()