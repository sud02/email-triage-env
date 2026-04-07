"""
inference.py — Baseline inference script for EmailTriageEnv.

Uses the OpenAI client to run an LLM agent against all 3 tasks.
Reads credentials from environment variables:
  - API_BASE_URL: LLM API endpoint (default: HuggingFace router)
  - MODEL_NAME:   Model identifier (default: Qwen2.5-72B-Instruct)
  - HF_TOKEN:     Hugging Face / API key (no default)
  - LOCAL_IMAGE_NAME: Docker image name (optional)

Stdout format (one line each):
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...,rn>
"""

import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.environment import EmailTriageEnv
from env.models import Action

# ---------------------------------------------------------------------------
# Config — defaults reflect active inference setup
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

TASKS = ["task_classify", "task_prioritize_route", "task_draft_reply"]
ENV_NAME = "email-triage"
MAX_RETRIES = 3

# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------

client = OpenAI(
    api_key=HF_TOKEN or "",
    base_url=API_BASE_URL,
)

# ---------------------------------------------------------------------------
# Structured stdout logging
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )

# ---------------------------------------------------------------------------
# Agent system prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert email triage assistant. You will be given an inbox and a task to complete.

At each step you must output a JSON object (and ONLY a JSON object) with one of these action types:

{"action_type": "read", "email_id": "eXXX"}
{"action_type": "classify", "email_id": "eXXX", "category": "CATEGORY"}
{"action_type": "prioritize", "email_id": "eXXX", "priority_level": "LEVEL"}
{"action_type": "route", "email_id": "eXXX", "department": "DEPT"}
{"action_type": "draft_reply", "email_id": "eXXX", "reply_body": "REPLY TEXT"}
{"action_type": "archive", "email_id": "eXXX"}
{"action_type": "escalate", "email_id": "eXXX", "reason": "REASON"}
{"action_type": "done"}

Valid categories: support, spam, billing, feature_request, legal, internal, sales, security, escalation
Valid priorities: critical, high, medium, low
Valid departments: engineering, finance, product, legal, hr, sales, support, security, customer_success

Rules:
- Read emails before acting on them (use "read" action)
- Classify ALL emails for task_classify
- Prioritize AND route ALL emails for task_prioritize_route
- Draft replies ONLY for emails that need a response for task_draft_reply
- Call "done" when you have completed the task
- Output ONLY valid JSON — no explanation, no markdown
"""


def build_user_message(obs: Dict[str, Any], step: int) -> str:
    lines = [
        f"STEP {step}",
        f"Task: {obs['task_id']}",
        f"Description: {obs['task_description'][:200]}",
        f"Steps used: {obs['step_count']}/{obs['max_steps']}",
        "",
        "INBOX STATUS:",
    ]
    for e in obs["emails"]:
        status_parts = []
        if e["is_read"]:
            status_parts.append("read")
        if e["label"]:
            status_parts.append(f"label={e['label']}")
        if e["priority"]:
            status_parts.append(f"priority={e['priority']}")
        if e["routed_to"]:
            status_parts.append(f"routed={e['routed_to']}")
        if e["has_draft"]:
            status_parts.append("replied")
        status = " | ".join(status_parts) if status_parts else "unprocessed"
        lines.append(f"  [{e['id']}] {e['subject'][:60]} — from: {e['from_addr']} | {status}")

    if obs.get("current_email"):
        ce = obs["current_email"]
        lines += [
            "",
            f"FOCUSED EMAIL [{ce['id']}]:",
            f"  From: {ce['from']}",
            f"  Subject: {ce['subject']}",
            f"  Body: {ce['body'][:800]}",
        ]

    lines += [
        "",
        f"Progress: classified={obs['info']['classified']}, prioritized={obs['info']['prioritized']}, "
        f"routed={obs['info']['routed']}, replied={obs['info']['replied']}",
        "",
        "What is your next action? Output only JSON.",
    ]
    return "\n".join(lines)


def parse_action(response_text: str) -> Optional[Dict[str, Any]]:
    text = response_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
    return None


def format_action_str(action_dict: Dict[str, Any]) -> str:
    """Format action dict as a compact single-line string for [STEP] logging."""
    atype = action_dict.get("action_type", "unknown")
    eid = action_dict.get("email_id", "")
    if atype == "read":
        return f"read({eid})"
    elif atype == "classify":
        return f"classify({eid},{action_dict.get('category', '')})"
    elif atype == "prioritize":
        return f"prioritize({eid},{action_dict.get('priority_level', '')})"
    elif atype == "route":
        return f"route({eid},{action_dict.get('department', '')})"
    elif atype == "draft_reply":
        snippet = (action_dict.get("reply_body") or "")[:30].replace("\n", " ")
        return f"draft_reply({eid},{snippet!r})"
    elif atype == "archive":
        return f"archive({eid})"
    elif atype == "escalate":
        reason = (action_dict.get("reason") or "")[:30].replace("\n", " ")
        return f"escalate({eid},{reason!r})"
    elif atype == "done":
        return "done()"
    return atype


def run_task(task_id: str) -> Dict[str, Any]:
    """Run the agent on a single task. Emits START/STEP*/END to stdout."""
    env = EmailTriageEnv(task_id=task_id)
    obs = env.reset()
    obs_dict = obs.model_dump()

    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    log_start(task=task_id, env=ENV_NAME, model=MODEL_NAME)

    final_grade: Dict[str, Any] = {"score": 0.0, "passed": False, "task_id": task_id}
    step = 0
    max_steps = obs_dict["max_steps"]
    rewards: List[float] = []
    last_error: Optional[str] = None

    try:
        while step < max_steps:
            step += 1
            user_msg = build_user_message(obs_dict, step)
            messages.append({"role": "user", "content": user_msg})

            # LLM call with retries
            action_dict: Optional[Dict[str, Any]] = None
            raw_response = ""
            for attempt in range(MAX_RETRIES):
                try:
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=messages,
                        max_tokens=512,
                        temperature=0.0,
                    )
                    raw_response = response.choices[0].message.content or ""
                    action_dict = parse_action(raw_response)
                    if action_dict:
                        break
                except Exception as e:
                    last_error = str(e)
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(1)

            if not action_dict:
                action_dict = {"action_type": "done"}

            messages.append({"role": "assistant", "content": raw_response})

            # Execute action
            step_error: Optional[str] = None
            try:
                action = Action(**action_dict)
                obs, reward, done, info = env.step(action)
                obs_dict = obs.model_dump()
                reward_val = reward.model_dump()["value"]
            except Exception as e:
                step_error = str(e)
                reward_val = 0.0
                done = True
                info = {}

            rewards.append(reward_val)
            log_step(
                step=step,
                action=format_action_str(action_dict),
                reward=reward_val,
                done=done,
                error=step_error or last_error,
            )
            last_error = None

            if done:
                if "grade" in info:
                    final_grade = info["grade"]
                break

        # Force grade if done wasn't called cleanly
        if not final_grade.get("score"):
            try:
                obs, reward, done, info = env.step(Action(action_type="done"))
                if "grade" in info:
                    final_grade = info["grade"]
            except Exception:
                pass

    finally:
        score = float(final_grade.get("score", 0.0))
        success = bool(final_grade.get("passed", False))
        log_end(success=success, steps=step, score=score, rewards=rewards)

    return final_grade


def main():
    results = {}
    for task_id in TASKS:
        grade = run_task(task_id)
        results[task_id] = float(grade.get("score", 0.0))

    overall = sum(results.values()) / len(results)
    # Final summary to stderr so it doesn't pollute the structured stdout
    print(
        f"Overall score: {overall:.2f} | "
        + " | ".join(f"{t}: {s:.2f}" for t, s in results.items()),
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
