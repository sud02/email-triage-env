"""
FastAPI server exposing the OpenEnv HTTP interface for EmailTriageEnv.
Endpoints: POST /reset, POST /step, GET /state, GET /tasks, GET /health
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import Any, Dict, Optional
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env.environment import EmailTriageEnv
from env.models import Action


app = FastAPI(
    title="EmailTriageEnv",
    description="OpenEnv-compliant email triage environment for AI agent training and evaluation.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Per-task environment instances (simple session management)
_envs: Dict[str, EmailTriageEnv] = {}


def _get_env(task_id: str) -> EmailTriageEnv:
    if task_id not in _envs:
        _envs[task_id] = EmailTriageEnv(task_id=task_id)
    return _envs[task_id]


# ---------------------------------------------------------------------------
# Request/response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str = "task_classify"


class StepRequest(BaseModel):
    task_id: str = "task_classify"
    action: Dict[str, Any]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "env": "EmailTriageEnv", "version": "1.0.0"}


@app.get("/tasks")
def list_tasks():
    """List all available tasks."""
    from env.environment import TASK_CONFIGS
    return {
        "tasks": [
            {
                "id": task_id,
                "description": cfg["description"][:120] + "...",
                "max_steps": cfg["max_steps"],
            }
            for task_id, cfg in TASK_CONFIGS.items()
        ]
    }


@app.post("/reset")
def reset(req: Optional[ResetRequest] = Body(None)):
    """Reset environment and return initial observation."""
    try:
        task_id = req.task_id if req else "task_classify"
        env = _get_env(task_id)
        obs = env.reset()
        return obs.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step")
def step(req: StepRequest):
    """Execute one step in the environment."""
    try:
        env = _get_env(req.task_id)
        action = Action(**req.action)
        obs, reward, done, info = env.step(action)
        return {
            "observation": obs.model_dump(),
            "reward": reward.model_dump(),
            "done": done,
            "info": info,
        }
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state")
def state(task_id: str = Query(default="task_classify")):
    """Return full internal state."""
    try:
        env = _get_env(task_id)
        s = env.state()
        return s.model_dump()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/")
def root():
    return {
        "name": "EmailTriageEnv",
        "description": "OpenEnv email triage environment",
        "endpoints": ["/health", "/tasks", "/reset", "/step", "/state"],
        "tasks": list(EmailTriageEnv.VALID_TASKS),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
