# app/reward.py - Normalized to 0-1
def compute_reward(progress_gain, completed, stress, energy):
    reward = 0.0

    # Higher reward for progress (max 0.3)
    reward += (progress_gain * 30) / 100  # 0.15 for 0.5 progress

    # Big bonus for completing a task (max 0.8)
    if completed:
        reward += 0.8

    # Softer penalties
    if stress > 80:
        reward -= 0.05
    elif stress > 60:
        reward -= 0.02

    if energy < 20:
        reward -= 0.05
    elif energy < 30:
        reward -= 0.02

    # Bonus for good state
    if energy > 60 and stress < 40:
        reward += 0.1

    # Clamp between 0 and 1
    return max(0.0, min(1.0, reward))