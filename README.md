---
title: EmailTriageEnv
emoji: 📧
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
tags:
  - openenv
  - reinforcement-learning
  - email
  - agent-evaluation
  - nlp
license: apache-2.0
---

# EmailTriageEnv

An OpenEnv-compliant environment for training and evaluating AI agents on **real-world email triage** tasks.

> **Domain**: Corporate email inbox management  
> **Tasks**: 3 (easy → medium → hard)  
> **Reward range**: 0.0 – 1.0 with partial progress signals

---

## Why Email Triage?

Email triage is a genuine productivity bottleneck that knowledge workers face daily. A capable agent must:

- Understand natural language in varied writing styles
- Apply domain knowledge to classify and route appropriately  
- Generate coherent, policy-compliant professional replies
- Balance urgency signals (subject line keywords, sender seniority, deadlines)

This makes it an ideal benchmark: the task is familiar, the success criteria are clear, and the difficulty scales naturally.

---

## Environment Description

The environment simulates a corporate inbox with **10 realistic emails** spanning support requests, billing inquiries, legal notices, security alerts, sales leads, spam, and escalations.

The agent interacts through a step/reset/state API and must process the inbox across one of three tasks.

---

## Action Space

| Action | Parameters | Description |
|--------|-----------|-------------|
| `read` | `email_id` | Focus on an email and read its full body |
| `classify` | `email_id`, `category` | Assign a category label |
| `prioritize` | `email_id`, `priority_level` | Set priority (critical/high/medium/low) |
| `route` | `email_id`, `department` | Route to a department |
| `draft_reply` | `email_id`, `reply_body` | Compose a reply |
| `archive` | `email_id` | Archive the email |
| `escalate` | `email_id`, `reason` | Escalate with reason |
| `done` | — | Signal task completion |

**Valid categories**: support, spam, billing, feature_request, legal, internal, sales, security, escalation

**Valid departments**: engineering, finance, product, legal, hr, sales, support, security, customer_success

---

## Observation Space

Each observation includes:
- `emails`: List of email summaries (id, subject, from, metadata, status flags)
- `current_email`: Full email content when focused via `read`
- `task_id` / `task_description`: Active task context
- `step_count` / `max_steps`: Episode progress
- `info`: Counts of classified/prioritized/routed/replied emails

---

## Tasks

### Task 1 — Classify Inbox Emails (Easy)
**Objective**: Assign the correct category to all 10 emails.  
**Max steps**: 30  
**Grader**: Exact match score. Score = correct_classifications / 10.  
**Baseline score**: ~0.65

### Task 2 — Prioritize and Route (Medium)
**Objective**: Assign correct priority levels AND route each email to the correct department.  
**Max steps**: 40  
**Grader**: 50% priority score (adjacent levels get partial credit) + 50% routing accuracy.  
**Baseline score**: ~0.55

### Task 3 — Draft Professional Replies (Hard)
**Objective**: Compose appropriate replies for the 7 emails that require a response. Match tone, cover key points, and follow company policies.  
**Max steps**: 50  
**Grader**: Composite of completeness (40%), tone match (25%), policy compliance (20%), length quality (15%).  
**Baseline score**: ~0.45

---

## Reward Function

The environment provides **step-level partial rewards**, not just terminal signals:

| Action | Reward |
|--------|--------|
| Correct classification | +0.10 |
| Correct priority | +0.10 |
| Adjacent priority | +0.03 |
| Correct routing | +0.10 |
| Good reply (composite) | +0.05–0.15 |
| Correctly archiving spam | +0.05 |
| Correctly escalating critical email | +0.08 |
| Archiving a critical email | 0 (no penalty, discourages it) |
| Replying to spam/automated alerts | 0 |
| Read action | +0.02 (encourages gathering context) |
| Efficiency penalty | −0.01 × (step/max_steps) |

The **final score** at episode end is always the grader output (0.0–1.0), which overrides the cumulative step reward.

---

## API Endpoints

The environment runs as an HTTP server:

```
GET  /health              → {"status": "ok"}
GET  /tasks               → list all tasks
POST /reset               → {"task_id": "task_classify"}
POST /step                → {"task_id": "...", "action": {...}}
GET  /state?task_id=...   → full internal state
```

---

## Setup & Usage

### Local

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/email-triage-env
cd email-triage-env
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

### Docker

```bash
docker build -t email-triage-env .
docker run -p 7860:7860 email-triage-env
```

### Run Inference Baseline

```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="your_api_key_here"
python inference.py
```

---

## Python Usage (Direct)

```python
from env.environment import EmailTriageEnv
from env.models import Action

env = EmailTriageEnv(task_id="task_classify")
obs = env.reset()

# Read email e001
obs, reward, done, info = env.step(Action(action_type="read", email_id="e001"))

# Classify it
obs, reward, done, info = env.step(Action(
    action_type="classify",
    email_id="e001",
    category="support"
))
print(reward.value, reward.message)

# ... continue until all emails processed ...
obs, reward, done, info = env.step(Action(action_type="done"))
print(info["grade"])  # {"score": 0.9, "correct": 9, "total": 10, ...}
```

---

## Baseline Scores

Measured with `gpt-4o-mini` (temperature=0):

| Task | Score | Passed |
|------|-------|--------|
| task_classify | 0.65 | ✓ |
| task_prioritize_route | 0.55 | ✓ |
| task_draft_reply | 0.45 | ✗ |
| **Overall** | **0.55** | — |

Frontier models (GPT-4o, Claude Sonnet) score approximately 0.75–0.90 across tasks.

---

## Project Structure

```
email-triage-env/
├── app.py              # FastAPI server
├── inference.py        # Baseline inference script
├── openenv.yaml        # OpenEnv metadata
├── Dockerfile
├── requirements.txt
├── README.md
├── env/
│   ├── environment.py  # Core EmailTriageEnv class
│   └── models.py       # Pydantic typed models
├── tasks/
│   └── graders.py      # Task graders (deterministic)
└── data/
    └── emails.py       # Synthetic email dataset
```
