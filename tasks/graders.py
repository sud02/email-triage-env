"""
Task graders for EmailTriageEnv.
Each grader scores agent performance on a task from 0.0 to 1.0.
All graders are deterministic and reproducible.
"""

from typing import Any, Dict, List, Optional
import re


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _text_contains(text: str, keywords: List[str]) -> float:
    """Return fraction of keywords found in text (case-insensitive)."""
    if not keywords:
        return 1.0
    text_lower = text.lower()
    found = sum(1 for kw in keywords if kw.lower() in text_lower)
    return found / len(keywords)


def _reply_completeness(reply: str, key_points: List[str]) -> float:
    """Score reply on how many key points it addresses (keyword heuristic)."""
    if not key_points:
        return 1.0
    total = 0.0
    for point in key_points:
        # Split point into words and check presence
        words = [w for w in re.split(r'\W+', point.lower()) if len(w) > 3]
        if not words:
            continue
        found = sum(1 for w in words if w in reply.lower())
        total += found / len(words)
    return min(total / len(key_points), 1.0)


def _reply_tone_score(reply: str, expected_tone: Optional[str]) -> float:
    """Heuristic tone scoring based on presence/absence of tone markers."""
    if not expected_tone or not reply:
        return 0.5

    reply_lower = reply.lower()
    tone_markers = {
        "professional": {
            "positive": ["dear", "sincerely", "regards", "thank you", "apologi", "please", "kindly"],
            "negative": ["hey", "lol", "btw", "gonna", "wanna", "!!!", "omg"],
        },
        "friendly": {
            "positive": ["hi", "thanks", "happy", "great", "love", "awesome", "sure", "absolutely"],
            "negative": ["hereby", "pursuant", "aforementioned"],
        },
        "enthusiastic": {
            "positive": ["excited", "great", "wonderful", "fantastic", "thrilled", "look forward", "excellent"],
            "negative": ["unfortunately", "regret", "unable", "cannot"],
        },
    }
    markers = tone_markers.get(expected_tone, {"positive": [], "negative": []})
    pos = sum(1 for m in markers["positive"] if m in reply_lower)
    neg = sum(1 for m in markers["negative"] if m in reply_lower)
    pos_score = min(pos / max(len(markers["positive"]), 1), 1.0)
    neg_penalty = min(neg * 0.15, 0.45)
    return max(0.0, min(1.0, pos_score - neg_penalty))


def _policy_compliance(reply: str, policies: List[str]) -> float:
    """
    Check reply doesn't violate key policies.
    Heuristic: detect forbidden commitment language near policy keywords.
    """
    if not policies or not reply:
        return 1.0
    reply_lower = reply.lower()
    violations = 0

    # Commitment markers that signal a policy violation
    commitment_phrases = [
        "we promise", "i promise", "we guarantee", "i guarantee",
        "we will definitely", "we can confirm", "you will receive by",
        "payment will be made", "payment will arrive", "we commit",
        "we will pay", "we will deliver by", "we will fix by",
    ]

    for policy in policies:
        policy_lower = policy.lower()
        if "do not promise" in policy_lower or "do not commit" in policy_lower:
            # Extract what we're not supposed to promise
            forbidden_subject = re.sub(r"do not (promise|commit to|quote|make)\s+", "", policy_lower)
            subject_words = [w for w in re.split(r'\W+', forbidden_subject) if len(w) > 5]
            # Violation = commitment phrase + subject keyword both present
            has_commitment = any(cp in reply_lower for cp in commitment_phrases)
            has_subject = any(w in reply_lower for w in subject_words)
            if has_commitment and has_subject:
                violations += 1
        elif "route to" in policy_lower or "route immediately" in policy_lower:
            # No penalty for routing policies — these are instructions not prohibitions
            pass
        elif "do not make" in policy_lower:
            if any(cp in reply_lower for cp in commitment_phrases):
                violations += 1

    return max(0.0, 1.0 - violations * 0.35)


# ---------------------------------------------------------------------------
# Task 1: Classify inbox emails  (easy)
# ---------------------------------------------------------------------------

def grade_classify(emails: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Score classification actions.
    Returns per-email scores and overall 0.0–1.0.
    """
    email_map = {e["id"]: e for e in emails}
    from data.emails import EMAILS
    ground_truth = {e["id"]: e["correct_category"] for e in EMAILS}

    correct = 0
    total = len(ground_truth)
    details = {}

    for email_id, correct_cat in ground_truth.items():
        email = email_map.get(email_id, {})
        assigned = email.get("label")
        is_correct = (assigned == correct_cat)
        details[email_id] = {
            "assigned": assigned,
            "correct": correct_cat,
            "score": 1.0 if is_correct else 0.0,
        }
        if is_correct:
            correct += 1

    # Partial credit: unclassified = 0, wrong = 0, correct = 1
    score = correct / total if total > 0 else 0.0

    return {
        "task_id": "task_classify",
        "score": round(score, 4),
        "correct": correct,
        "total": total,
        "details": details,
        "passed": score >= 0.7,
    }


# ---------------------------------------------------------------------------
# Task 2: Prioritize and route  (medium)
# ---------------------------------------------------------------------------

def grade_prioritize_route(emails: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Score priority assignment + department routing.
    Priority score: exact match = 1.0, adjacent level = 0.5, else 0.0
    Route score: exact match = 1.0, else 0.0 (None dept emails excluded)
    """
    from data.emails import EMAILS
    priority_order = ["critical", "high", "medium", "low"]
    email_map = {e["id"]: e for e in emails}
    gt_map = {e["id"]: e for e in EMAILS}

    priority_scores = []
    route_scores = []
    details = {}

    for email_id, gt in gt_map.items():
        email = email_map.get(email_id, {})
        assigned_priority = email.get("priority")
        assigned_dept = email.get("routed_to")

        # Priority scoring
        correct_priority = gt["correct_priority"]
        if assigned_priority == correct_priority:
            p_score = 1.0
        elif assigned_priority and correct_priority:
            try:
                diff = abs(priority_order.index(assigned_priority) - priority_order.index(correct_priority))
                p_score = max(0.0, 1.0 - diff * 0.4)
            except ValueError:
                p_score = 0.0
        else:
            p_score = 0.0
        priority_scores.append(p_score)

        # Route scoring (only for emails with a correct dept)
        correct_dept = gt["correct_department"]
        if correct_dept is not None:
            r_score = 1.0 if assigned_dept == correct_dept else 0.0
            route_scores.append(r_score)
        else:
            # Should NOT be routed
            r_score = 0.0 if assigned_dept else 1.0
            route_scores.append(r_score)

        details[email_id] = {
            "priority": {"assigned": assigned_priority, "correct": correct_priority, "score": p_score},
            "route": {"assigned": assigned_dept, "correct": correct_dept, "score": r_score},
        }

    avg_priority = sum(priority_scores) / len(priority_scores) if priority_scores else 0.0
    avg_route = sum(route_scores) / len(route_scores) if route_scores else 0.0
    score = 0.5 * avg_priority + 0.5 * avg_route

    return {
        "task_id": "task_prioritize_route",
        "score": round(score, 4),
        "priority_score": round(avg_priority, 4),
        "route_score": round(avg_route, 4),
        "details": details,
        "passed": score >= 0.65,
    }


# ---------------------------------------------------------------------------
# Task 3: Draft professional replies  (hard)
# ---------------------------------------------------------------------------

def grade_draft_reply(emails: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Score reply quality across 4 dimensions:
      - completeness: covers required key points
      - tone: matches expected tone
      - policy: doesn't violate stated policies
      - length: not too short (<20 words) or too long (>300 words)
    Only emails with requires_reply=True are graded.
    """
    from data.emails import EMAILS
    email_map = {e["id"]: e for e in emails}
    gt_map = {e["id"]: e for e in EMAILS}

    reply_emails = [e for e in EMAILS if e["requires_reply"]]
    if not reply_emails:
        return {"task_id": "task_draft_reply", "score": 0.0, "passed": False, "details": {}}

    scores = []
    details = {}

    for gt in reply_emails:
        email_id = gt["id"]
        email = email_map.get(email_id, {})
        reply = email.get("reply_drafted") or ""

        if not reply.strip():
            scores.append(0.0)
            details[email_id] = {"score": 0.0, "reason": "no reply drafted"}
            continue

        word_count = len(reply.split())
        length_score = 1.0
        if word_count < 20:
            length_score = word_count / 20
        elif word_count > 300:
            length_score = max(0.5, 300 / word_count)

        completeness = _reply_completeness(reply, gt["reply_key_points"])
        tone = _reply_tone_score(reply, gt.get("reply_tone"))
        policy = _policy_compliance(reply, gt.get("reply_policy", []))

        # Weighted composite
        email_score = (
            0.40 * completeness +
            0.25 * tone +
            0.20 * policy +
            0.15 * length_score
        )
        scores.append(email_score)
        details[email_id] = {
            "score": round(email_score, 4),
            "completeness": round(completeness, 4),
            "tone": round(tone, 4),
            "policy": round(policy, 4),
            "length_score": round(length_score, 4),
            "word_count": word_count,
        }

    score = sum(scores) / len(scores) if scores else 0.0

    return {
        "task_id": "task_draft_reply",
        "score": round(score, 4),
        "emails_graded": len(scores),
        "details": details,
        "passed": score >= 0.55,
    }
