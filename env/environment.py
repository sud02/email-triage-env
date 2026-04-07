"""
EmailTriageEnv — Main environment class.
Implements the full OpenEnv interface: step(), reset(), state()
"""

import copy
import sys
import os
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.models import (
    Action, Email, EmailSummary, EnvState, Observation, Reward
)
from data.emails import EMAILS, CATEGORIES, PRIORITIES, DEPARTMENTS


TASK_CONFIGS = {
    "task_classify": {
        "description": (
            "You have an inbox of 10 emails. Classify each email into one of the following categories: "
            f"{CATEGORIES}. Use the 'classify' action for each email. "
            "You can use 'read' to see the full email body before classifying. "
            "Call 'done' when all emails have been classified."
        ),
        "max_steps": 30,
    },
    "task_prioritize_route": {
        "description": (
            "You have an inbox of 10 emails. Assign a priority level (critical/high/medium/low) "
            "and route each email to the correct department. "
            f"Departments: {[d for d in DEPARTMENTS if d]}. "
            "Use 'prioritize' and 'route' actions. Use 'read' to see full emails. "
            "Call 'done' when all emails are prioritized and routed."
        ),
        "max_steps": 40,
    },
    "task_draft_reply": {
        "description": (
            "You have an inbox of 10 emails. Some require a reply; some do not. "
            "Read each email carefully. For emails requiring a response, use 'draft_reply' "
            "with a professional, complete reply. Match the appropriate tone and follow company policy. "
            "Do not reply to spam, internal announcements, or automated alerts. "
            "Call 'done' when you have drafted all necessary replies."
        ),
        "max_steps": 50,
    },
}


class EmailTriageEnv:
    """
    A realistic email triage environment for training and evaluating AI agents.

    Agents must classify, prioritize, route, and draft replies for a synthetic
    corporate inbox across three tasks of increasing difficulty.
    """

    AVAILABLE_ACTIONS = ["read", "classify", "prioritize", "route", "draft_reply", "archive", "escalate", "done"]
    VALID_TASKS = list(TASK_CONFIGS.keys())

    def __init__(self, task_id: str = "task_classify"):
        if task_id not in self.VALID_TASKS:
            raise ValueError(f"Unknown task: {task_id}. Choose from {self.VALID_TASKS}")
        self.task_id = task_id
        self._emails: List[Dict[str, Any]] = []
        self._step_count = 0
        self._done = False
        self._total_reward = 0.0
        self._action_history: List[Dict[str, Any]] = []
        self._focused_email_id: Optional[str] = None
        self._initialized = False

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self) -> Observation:
        """Reset environment to initial state and return first observation."""
        self._emails = [self._init_email(e) for e in EMAILS]
        self._step_count = 0
        self._done = False
        self._total_reward = 0.0
        self._action_history = []
        self._focused_email_id = None
        self._initialized = True
        return self._make_observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Execute one action and return (observation, reward, done, info).

        Reward is shaped at each step for partial progress signals.
        """
        if not self._initialized:
            raise RuntimeError("Call reset() before step()")
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        self._step_count += 1
        max_steps = TASK_CONFIGS[self.task_id]["max_steps"]

        reward_value, reward_components, reward_msg = self._execute_action(action)

        # Step penalty to discourage wasted moves
        step_fraction = self._step_count / max_steps
        efficiency_penalty = 0.01 * step_fraction
        reward_value = max(0.0, reward_value - efficiency_penalty)

        # Episode termination conditions
        episode_done = False
        if action.action_type == "done":
            episode_done = True
        elif self._step_count >= max_steps:
            episode_done = True
            reward_value = max(0.0, reward_value - 0.05)  # small penalty for running out of steps
            reward_msg += " [Episode ended: step limit reached]"

        self._done = episode_done
        self._total_reward = min(1.0, self._total_reward + reward_value * 0.1)
        self._action_history.append({
            "step": self._step_count,
            "action": action.model_dump(),
            "reward": reward_value,
        })

        reward = Reward(
            value=round(reward_value, 4),
            components=reward_components,
            message=reward_msg,
            cumulative=round(self._total_reward, 4),
        )

        # Final graded reward on done
        info: Dict[str, Any] = {}
        if episode_done:
            grade = self._run_grader()
            self._total_reward = grade["score"]
            reward = Reward(
                value=round(grade["score"], 4),
                components={"final_grade": grade["score"]},
                message=f"Episode complete. Final graded score: {grade['score']:.4f}",
                cumulative=round(grade["score"], 4),
            )
            info["grade"] = grade

        obs = self._make_observation()
        return obs, reward, episode_done, info

    def state(self) -> EnvState:
        """Return full internal environment state."""
        if not self._initialized:
            raise RuntimeError("Call reset() before state()")
        return EnvState(
            task_id=self.task_id,
            emails=copy.deepcopy(self._emails),
            step_count=self._step_count,
            done=self._done,
            total_reward=round(self._total_reward, 4),
            action_history=copy.deepcopy(self._action_history),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_email(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Create mutable email state from raw data."""
        return {
            "id": raw["id"],
            "from": raw["from"],
            "to": raw["to"],
            "subject": raw["subject"],
            "body": raw["body"],
            "timestamp": raw["timestamp"],
            "has_attachment": raw["has_attachment"],
            "is_read": False,
            "label": None,
            "priority": None,
            "routed_to": None,
            "reply_drafted": None,
            "archived": False,
            "escalated": False,
            "escalation_reason": None,
        }

    def _get_email(self, email_id: str) -> Optional[Dict[str, Any]]:
        for e in self._emails:
            if e["id"] == email_id:
                return e
        return None

    def _execute_action(self, action: Action) -> Tuple[float, Dict[str, float], str]:
        """
        Execute action and return (reward_value, components, message).
        Rewards are shaped per-step for partial progress.
        """
        atype = action.action_type

        if atype == "done":
            return 0.0, {}, "Agent signaled done."

        if atype == "read":
            email = self._get_email(action.email_id) if action.email_id else None
            if not email:
                return 0.0, {"invalid": -0.0}, "Invalid email_id for read."
            email["is_read"] = True
            self._focused_email_id = action.email_id
            return 0.02, {"read": 0.02}, f"Read email {action.email_id}."

        if atype == "classify":
            return self._action_classify(action)

        if atype == "prioritize":
            return self._action_prioritize(action)

        if atype == "route":
            return self._action_route(action)

        if atype == "draft_reply":
            return self._action_draft_reply(action)

        if atype == "archive":
            email = self._get_email(action.email_id) if action.email_id else None
            if not email:
                return 0.0, {}, "Invalid email_id for archive."
            from data.emails import EMAILS as RAW
            gt = next((e for e in RAW if e["id"] == action.email_id), {})
            email["archived"] = True
            # Archiving spam/low-priority is good; archiving critical is bad
            if gt.get("correct_priority") == "critical":
                return 0.0, {"archive_critical_penalty": -0.05}, "Archived critical email — penalty."
            if gt.get("correct_category") in ("spam", "internal"):
                return 0.05, {"archive_correct": 0.05}, "Correctly archived low-value email."
            return 0.01, {"archive": 0.01}, "Archived email."

        if atype == "escalate":
            email = self._get_email(action.email_id) if action.email_id else None
            if not email:
                return 0.0, {}, "Invalid email_id for escalate."
            from data.emails import EMAILS as RAW
            gt = next((e for e in RAW if e["id"] == action.email_id), {})
            email["escalated"] = True
            email["escalation_reason"] = action.reason
            if gt.get("correct_priority") in ("critical", "high"):
                return 0.08, {"escalate_correct": 0.08}, "Correctly escalated high-priority email."
            return 0.0, {"escalate_unnecessary": 0.0}, "Escalated a non-critical email."

        return 0.0, {}, f"Unknown action type: {atype}"

    def _action_classify(self, action: Action) -> Tuple[float, Dict[str, float], str]:
        email = self._get_email(action.email_id) if action.email_id else None
        if not email:
            return 0.0, {}, "Invalid email_id."
        if action.category not in CATEGORIES:
            return 0.0, {"invalid_category": 0.0}, f"Invalid category '{action.category}'."

        from data.emails import EMAILS as RAW
        gt = next((e for e in RAW if e["id"] == action.email_id), {})
        correct_cat = gt.get("correct_category")

        email["label"] = action.category
        if action.category == correct_cat:
            return 0.1, {"classify_correct": 0.1}, f"Correct classification: {action.category}."
        else:
            return 0.0, {"classify_wrong": 0.0}, f"Wrong classification (got {action.category}, expected {correct_cat})."

    def _action_prioritize(self, action: Action) -> Tuple[float, Dict[str, float], str]:
        email = self._get_email(action.email_id) if action.email_id else None
        if not email:
            return 0.0, {}, "Invalid email_id."
        if action.priority_level not in PRIORITIES:
            return 0.0, {}, f"Invalid priority '{action.priority_level}'."

        from data.emails import EMAILS as RAW
        priority_order = ["critical", "high", "medium", "low"]
        gt = next((e for e in RAW if e["id"] == action.email_id), {})
        correct = gt.get("correct_priority")

        email["priority"] = action.priority_level
        if action.priority_level == correct:
            return 0.1, {"priority_correct": 0.1}, f"Correct priority: {action.priority_level}."
        else:
            try:
                diff = abs(priority_order.index(action.priority_level) - priority_order.index(correct))
                partial = max(0.0, 0.05 - diff * 0.02)
            except ValueError:
                partial = 0.0
            return partial, {"priority_partial": partial}, f"Priority mismatch (got {action.priority_level}, expected {correct})."

    def _action_route(self, action: Action) -> Tuple[float, Dict[str, float], str]:
        email = self._get_email(action.email_id) if action.email_id else None
        if not email:
            return 0.0, {}, "Invalid email_id."
        if action.department not in [d for d in DEPARTMENTS if d]:
            return 0.0, {}, f"Invalid department '{action.department}'."

        from data.emails import EMAILS as RAW
        gt = next((e for e in RAW if e["id"] == action.email_id), {})
        correct_dept = gt.get("correct_department")

        email["routed_to"] = action.department
        if action.department == correct_dept:
            return 0.1, {"route_correct": 0.1}, f"Correctly routed to {action.department}."
        return 0.0, {"route_wrong": 0.0}, f"Wrong dept (got {action.department}, expected {correct_dept})."

    def _action_draft_reply(self, action: Action) -> Tuple[float, Dict[str, float], str]:
        from tasks.graders import (
            _reply_completeness, _reply_tone_score, _policy_compliance
        )
        from data.emails import EMAILS as RAW

        email = self._get_email(action.email_id) if action.email_id else None
        if not email:
            return 0.0, {}, "Invalid email_id."
        if not action.reply_body or len(action.reply_body.strip()) < 10:
            return 0.0, {}, "Reply body too short."

        gt = next((e for e in RAW if e["id"] == action.email_id), {})

        # Penalize replying to emails that don't need a reply
        if not gt.get("requires_reply"):
            email["reply_drafted"] = action.reply_body
            return 0.0, {"unnecessary_reply": 0.0}, "Replied to email that doesn't require a reply."

        completeness = _reply_completeness(action.reply_body, gt.get("reply_key_points", []))
        tone = _reply_tone_score(action.reply_body, gt.get("reply_tone"))
        policy = _policy_compliance(action.reply_body, gt.get("reply_policy", []))

        email["reply_drafted"] = action.reply_body
        step_score = 0.40 * completeness + 0.35 * tone + 0.25 * policy
        step_reward = step_score * 0.15  # scaled for step reward

        return step_reward, {
            "completeness": completeness,
            "tone": tone,
            "policy": policy,
        }, f"Reply drafted (completeness={completeness:.2f}, tone={tone:.2f}, policy={policy:.2f})."

    def _run_grader(self) -> Dict[str, Any]:
        """Run the appropriate grader for the current task."""
        from tasks.graders import grade_classify, grade_prioritize_route, grade_draft_reply
        if self.task_id == "task_classify":
            return grade_classify(self._emails)
        elif self.task_id == "task_prioritize_route":
            return grade_prioritize_route(self._emails)
        elif self.task_id == "task_draft_reply":
            return grade_draft_reply(self._emails)
        return {"score": 0.0}

    def _make_observation(self) -> Observation:
        """Build current observation."""
        config = TASK_CONFIGS[self.task_id]
        summaries = [self._email_summary(e) for e in self._emails]
        focused = None
        if self._focused_email_id:
            focused = self._get_email(self._focused_email_id)

        return Observation(
            emails=summaries,
            current_email=focused,
            task_id=self.task_id,
            task_description=config["description"],
            step_count=self._step_count,
            max_steps=config["max_steps"],
            available_actions=self.AVAILABLE_ACTIONS,
            info={
                "total_emails": len(self._emails),
                "unread": sum(1 for e in self._emails if not e["is_read"]),
                "classified": sum(1 for e in self._emails if e["label"]),
                "prioritized": sum(1 for e in self._emails if e["priority"]),
                "routed": sum(1 for e in self._emails if e["routed_to"]),
                "replied": sum(1 for e in self._emails if e["reply_drafted"]),
            },
        )

    def _email_summary(self, e: Dict[str, Any]) -> EmailSummary:
        return EmailSummary(
            id=e["id"],
            from_addr=e["from"],
            subject=e["subject"],
            timestamp=e["timestamp"],
            has_attachment=e["has_attachment"],
            is_read=e["is_read"],
            label=e["label"],
            priority=e["priority"],
            routed_to=e["routed_to"],
            archived=e["archived"],
            escalated=e["escalated"],
            has_draft=bool(e["reply_drafted"]),
        )
