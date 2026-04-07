---
title: Student Life Environment
emoji: 🎓
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# 🎓 Student Life Environment

An AI-powered student decision-making environment built with **OpenEnv** framework. An LLM agent makes daily decisions (work, rest, spend) to maximize academic performance and well-being over a semester.

## 📋 Overview

This environment simulates a student's life with:
- **5 Tasks** with deadlines (Assignment, Hackathon, Exam Prep, Research Paper, Project)
- **Dynamic metrics**: Energy, Stress, Money
- **Reward system**: Progress-based rewards with completion bonuses
- **LLM Agent**: Powered by Qwen 2.5 (via Hugging Face Inference API)

## 🚀 Live Demo

The Space is running at: **https://hrana20-student-life-env.hf.space**

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/reset` | Start a new semester/episode |
| `POST` | `/step` | Take an action (work/rest/spend) |
| `GET` | `/state` | Get current episode state |
| `GET` | `/health` | Health check |

## 🎮 Actions

| Action | Format | Effect |
|--------|--------|--------|
| Work | `{"action_type": "work", "task_name": "Assignment"}` | Progress +50%, Energy -5, Stress +2 |
| Rest | `{"action_type": "rest"}` | Energy +40, Stress -20 |
| Spend | `{"action_type": "spend", "amount": 200}` | Money -amount |

## 📊 Tasks & Deadlines

| Task | Deadline (Day) |
|------|----------------|
| Assignment | Day 3 |
| Hackathon | Day 5 |
| Exam Prep | Day 7 |
| Research Paper | Day 8 |
| Project | Day 10 |

## 🏆 Scoring

| Action | Reward |
|--------|--------|
| Work progress | ~0.15 |
| Task completion | 0.8 |
| Rest | 0.1 |
| All tasks completed | +0.5 bonus |

## 🛠️ Local Development

```bash
# Clone the repository
git clone https://huggingface.co/spaces/Hrana20/student-life-env

# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn server:app --reload --port 7860

# Run the agent
python inference.py