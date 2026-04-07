def grade_easy(state):
    task = state["tasks"][0]
    return min(task.progress, 1.0)


def grade_medium(state):
    completed = sum(1 for t in state["tasks"] if t.completed)
    stress_penalty = state["stress"] / 100
    return max(0, (completed / 2) - stress_penalty)


def grade_hard(state):
    completed = sum(1 for t in state["tasks"] if t.completed)
    money_factor = state["money"] / 2000
    stress_factor = 1 - (state["stress"] / 100)

    score = (completed / 2) * 0.5 + money_factor * 0.25 + stress_factor * 0.25
    return max(0, min(score, 1))