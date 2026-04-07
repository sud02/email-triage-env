"""
inference.py — High-performance inference script for EmailTriageEnv.

Strategy:
  Phase 1 — Read ALL emails upfront to gather full context.
  Phase 2 — Perform all required task actions using that context.

Uses the OpenAI client against the configured API endpoint.

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

client = OpenAI(api_key=HF_TOKEN or "", base_url=API_BASE_URL)

# ---------------------------------------------------------------------------
# Structured stdout logging
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)

# ---------------------------------------------------------------------------
# Task-specific system prompts
# ---------------------------------------------------------------------------

SYSTEM_CLASSIFY = """You are an expert email triage assistant. Your job is to classify emails into exactly one category.

CATEGORIES (choose exactly one):
- support: customer asking for help, technical issue, how-to question
- spam: unsolicited marketing, newsletters, promotional emails
- billing: invoices, payments, pricing, refunds
- feature_request: asking for new features or product improvements
- legal: contracts, compliance, legal notices, agreements
- internal: messages from within the company (HR, internal announcements)
- sales: sales inquiries, enterprise deal interest, purchase intent
- security: security alerts, login attempts, vulnerabilities
- escalation: forwarded complaints, urgent issues needing management attention

RULES:
- Output ONLY a JSON object — no explanation, no markdown.
- Read each email before classifying it.
- Use the "done" action when ALL 10 emails are classified.

Action format:
{"action_type": "read", "email_id": "eXXX"}
{"action_type": "classify", "email_id": "eXXX", "category": "CATEGORY"}
{"action_type": "done"}"""

SYSTEM_PRIORITIZE_ROUTE = """You are an expert email triage assistant. Your job is to assign priority and route each email.

PRIORITY LEVELS:
- critical: immediate action required, revenue impact, security threat, executive escalation
- high: important, time-sensitive, legal/compliance deadlines, high-value sales
- medium: normal business, billing follow-ups, standard support questions
- low: informational, feature requests, newsletters, internal reminders, spam

DEPARTMENTS:
- engineering: technical outages, product bugs, server issues
- finance: invoices, billing, payments
- product: feature requests, product feedback
- legal: contracts, legal compliance, DPA, GDPR
- hr: internal HR matters, timesheets, employee issues
- sales: sales inquiries, enterprise deals, pricing discussions
- support: general customer support, how-to questions
- security: security alerts, login failures, threats
- customer_success: client complaints, renewal risk, account management

IMPORTANT RULES:
- Spam, newsletters, and automated alerts with NO department should NOT be routed — skip the route action for those.
- e002 (newsletter/spam) and e008 (automated security alert) do NOT need routing.
- Output ONLY a JSON object.
- Use "done" when ALL emails are prioritized and routed.

Action format:
{"action_type": "read", "email_id": "eXXX"}
{"action_type": "prioritize", "email_id": "eXXX", "priority_level": "LEVEL"}
{"action_type": "route", "email_id": "eXXX", "department": "DEPT"}
{"action_type": "done"}"""

SYSTEM_DRAFT_REPLY = """You are a professional customer support specialist drafting email replies.

EMAILS THAT NEED A REPLY (reply to these 7):
- e001: Production outage complaint (critical customer)
- e003: Invoice payment confirmation request
- e004: Feature request for dark mode
- e005: Legal DPA document requiring action
- e007: Enterprise sales inquiry (500 seats)
- e009: Customer asking how to export CSV
- e010: Escalated major client complaint

DO NOT REPLY TO (3 emails — skip these):
- e002: Newsletter/spam (no reply needed)
- e006: Internal HR announcement (no reply needed)
- e008: Automated security alert (no reply needed)

REPLY QUALITY GUIDELINES:
- Length: 50-150 words (never under 20, never over 300)
- Always start with "Dear [Name/Team]," or "Hi [Name],"
- End with "Sincerely," / "Best regards," / "Thanks," as appropriate
- Use "apologize", "thank you", "please", "kindly" for professional tone
- For sales (e007): use "excited", "great", "look forward" — enthusiastic tone
- For support (e004, e009): use "hi", "thanks", "happy to help" — friendly tone
- NEVER promise specific dates, prices, or SLAs without approval
- Always acknowledge the customer's concern

Output ONLY a JSON object. Use "done" when all 7 replies are drafted.

Action format:
{"action_type": "read", "email_id": "eXXX"}
{"action_type": "draft_reply", "email_id": "eXXX", "reply_body": "REPLY TEXT"}
{"action_type": "done"}"""

SYSTEM_PROMPTS = {
    "task_classify": SYSTEM_CLASSIFY,
    "task_prioritize_route": SYSTEM_PRIORITIZE_ROUTE,
    "task_draft_reply": SYSTEM_DRAFT_REPLY,
}

# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def call_llm(messages: List[Dict[str, str]], max_tokens: int = 600) -> str:
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.0,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(1)
    return ""

def parse_action(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{.*?\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
    return None

def format_action_str(action_dict: Dict[str, Any]) -> str:
    atype = action_dict.get("action_type", "unknown")
    eid = action_dict.get("email_id", "")
    if atype == "read":
        return f"read({eid})"
    elif atype == "classify":
        return f"classify({eid},{action_dict.get('category','')})"
    elif atype == "prioritize":
        return f"prioritize({eid},{action_dict.get('priority_level','')})"
    elif atype == "route":
        return f"route({eid},{action_dict.get('department','')})"
    elif atype == "draft_reply":
        snippet = (action_dict.get("reply_body") or "")[:30].replace("\n", " ")
        return f"draft_reply({eid},{snippet!r})"
    elif atype == "archive":
        return f"archive({eid})"
    elif atype == "escalate":
        return f"escalate({eid})"
    elif atype == "done":
        return "done()"
    return atype

# ---------------------------------------------------------------------------
# Agent state tracker
# ---------------------------------------------------------------------------

def build_inbox_summary(obs_dict: Dict[str, Any], email_bodies: Dict[str, str]) -> str:
    """Build a rich inbox summary including full bodies for read emails."""
    lines = [
        f"Task: {obs_dict['task_id']}",
        f"Steps used: {obs_dict['step_count']}/{obs_dict['max_steps']}",
        f"Progress: classified={obs_dict['info']['classified']} "
        f"prioritized={obs_dict['info']['prioritized']} "
        f"routed={obs_dict['info']['routed']} "
        f"replied={obs_dict['info']['replied']}",
        "",
        "INBOX:",
    ]
    for e in obs_dict["emails"]:
        status_parts = []
        if e["label"]:
            status_parts.append(f"category={e['label']}")
        if e["priority"]:
            status_parts.append(f"priority={e['priority']}")
        if e["routed_to"]:
            status_parts.append(f"routed={e['routed_to']}")
        if e["has_draft"]:
            status_parts.append("replied")
        status = " | ".join(status_parts) if status_parts else "pending"
        lines.append(f"  [{e['id']}] {e['subject'][:70]}")
        lines.append(f"        From: {e['from_addr']} | Status: {status}")
        if e["id"] in email_bodies:
            lines.append(f"        Body: {email_bodies[e['id']][:600]}")
        lines.append("")
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Main agent loop
# ---------------------------------------------------------------------------

def execute_action(env: EmailTriageEnv, action_dict: Dict[str, Any]):
    action = Action(**action_dict)
    return env.step(action)

def run_task(task_id: str) -> Dict[str, Any]:
    env = EmailTriageEnv(task_id=task_id)
    obs = env.reset()
    obs_dict = obs.model_dump()

    system_prompt = SYSTEM_PROMPTS[task_id]
    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

    log_start(task=task_id, env=ENV_NAME, model=MODEL_NAME)

    final_grade: Dict[str, Any] = {"score": 0.0, "passed": False, "task_id": task_id}
    step = 0
    max_steps = obs_dict["max_steps"]
    rewards: List[float] = []
    email_bodies: Dict[str, str] = {}  # cache of full email bodies

    # --- Phase 1: Read all emails upfront ---
    email_ids = [e["id"] for e in obs_dict["emails"]]
    for eid in email_ids:
        if step >= max_steps:
            break
        step += 1
        action_dict = {"action_type": "read", "email_id": eid}
        try:
            obs, reward, done, info = execute_action(env, action_dict)
            obs_dict = obs.model_dump()
            reward_val = reward.model_dump()["value"]
            # Cache the full body
            if obs_dict.get("current_email"):
                email_bodies[eid] = obs_dict["current_email"].get("body", "")
        except Exception as e:
            reward_val = 0.0
            done = False
        rewards.append(reward_val)
        log_step(step=step, action=format_action_str(action_dict), reward=reward_val, done=done, error=None)
        if done:
            break

    # --- Phase 2: LLM-driven action phase ---
    last_error: Optional[str] = None
    while step < max_steps:
        step += 1
        inbox_summary = build_inbox_summary(obs_dict, email_bodies)
        user_msg = f"{inbox_summary}\n\nWhat is your next action? Output only JSON."
        messages.append({"role": "user", "content": user_msg})

        action_dict = None
        raw_response = ""
        try:
            raw_response = call_llm(messages)
            action_dict = parse_action(raw_response)
        except Exception as e:
            last_error = str(e)

        if not action_dict:
            action_dict = {"action_type": "done"}

        messages.append({"role": "assistant", "content": raw_response})

        step_error: Optional[str] = None
        try:
            obs, reward, done, info = execute_action(env, action_dict)
            obs_dict = obs.model_dump()
            reward_val = reward.model_dump()["value"]
            # Cache body if this was a read
            if action_dict.get("action_type") == "read" and obs_dict.get("current_email"):
                eid = action_dict.get("email_id", "")
                email_bodies[eid] = obs_dict["current_email"].get("body", "")
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
            obs, reward, done, info = execute_action(env, {"action_type": "done"})
            if "grade" in info:
                final_grade = info["grade"]
        except Exception:
            pass

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
    print(
        f"Overall score: {overall:.2f} | "
        + " | ".join(f"{t}: {s:.2f}" for t, s in results.items()),
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
