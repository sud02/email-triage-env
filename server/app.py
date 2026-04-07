"""
server/app.py — OpenEnv-core compatible server for EmailTriageEnv.

Uses HTTPEnvServer from openenv-core to expose the environment in
multi-mode deployment (Docker + pip-installable package).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI
from openenv.core.env_server.http_server import HTTPEnvServer
from openenv.core.env_server.interfaces import (
    Action as BaseAction,
    Environment as BaseEnvironment,
    Observation as BaseObservation,
)
from pydantic import ConfigDict, Field

from env.environment import EmailTriageEnv
from env.models import Action as _EnvAction


# ---------------------------------------------------------------------------
# Action / Observation models (openenv-core compatible)
# ---------------------------------------------------------------------------

class EmailAction(BaseAction):
    """Action schema for EmailTriageEnv."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    action_type: str = Field(description="One of: read, classify, prioritize, route, draft_reply, archive, escalate, done")
    email_id: Optional[str] = Field(default=None)
    category: Optional[str] = Field(default=None)
    priority_level: Optional[str] = Field(default=None)
    department: Optional[str] = Field(default=None)
    reply_body: Optional[str] = Field(default=None)
    reason: Optional[str] = Field(default=None)


class EmailObservation(BaseObservation):
    """Observation schema for EmailTriageEnv."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    emails: List[Dict[str, Any]] = Field(default_factory=list)
    current_email: Optional[Dict[str, Any]] = None
    task_id: str = "task_classify"
    task_description: str = ""
    step_count: int = 0
    max_steps: int = 30
    available_actions: List[str] = Field(default_factory=list)
    info: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Environment adapter
# ---------------------------------------------------------------------------

class EmailEnvAdapter(BaseEnvironment):
    """Wraps EmailTriageEnv to satisfy the openenv-core Environment interface."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        super().__init__()
        self._env: Optional[EmailTriageEnv] = None

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> EmailObservation:
        task_id = episode_id or kwargs.get("task_id", "task_classify")
        self._env = EmailTriageEnv(task_id=task_id)
        obs = self._env.reset()
        return self._wrap_obs(obs, done=False, reward=0.0)

    def step(
        self,
        action: EmailAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> EmailObservation:
        if self._env is None:
            raise RuntimeError("Call reset() before step()")
        env_action = _EnvAction(
            action_type=action.action_type,
            email_id=action.email_id,
            category=action.category,
            priority_level=action.priority_level,
            department=action.department,
            reply_body=action.reply_body,
            reason=action.reason,
        )
        obs, reward, done, info = self._env.step(env_action)
        result = self._wrap_obs(obs, done=done, reward=reward.value)
        if info.get("grade"):
            result.info["grade"] = info["grade"]
        return result

    @property
    def state(self) -> Dict[str, Any]:
        if self._env is None:
            return {}
        return self._env.state().model_dump()

    # ------------------------------------------------------------------

    def _wrap_obs(self, obs: Any, done: bool, reward: float) -> EmailObservation:
        return EmailObservation(
            done=done,
            reward=reward,
            emails=[e.model_dump() for e in obs.emails],
            current_email=obs.current_email,
            task_id=obs.task_id,
            task_description=obs.task_description,
            step_count=obs.step_count,
            max_steps=obs.max_steps,
            available_actions=obs.available_actions,
            info=obs.info,
        )


# ---------------------------------------------------------------------------
# FastAPI app via HTTPEnvServer
# ---------------------------------------------------------------------------

app = FastAPI(
    title="EmailTriageEnv",
    description="OpenEnv-compliant email triage environment.",
    version="1.0.0",
)

_server = HTTPEnvServer(
    env=EmailEnvAdapter,
    action_cls=EmailAction,
    observation_cls=EmailObservation,
    max_concurrent_envs=10,
)
_server.register_routes(app)


def main() -> None:
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
