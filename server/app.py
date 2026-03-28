"""
FastAPI application for the Autonomous Software Engineer environment.

Exposes the AutoSWEEnvironment over HTTP endpoints compatible
with the OpenEnv specification:

    POST /reset   — start a new episode
    POST /step    — execute an action
    GET  /state   — query episode metadata
    GET  /health  — liveness probe

Usage:
    # Development
    uvicorn server.app:app --reload --host 0.0.0.0 --port 7860

    # Production
    uvicorn server.app:app --host 0.0.0.0 --port 7860
"""

from __future__ import annotations

import json
import logging
import traceback
from typing import Any, Dict, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Import our environment & models
try:
    from server.auto_swe_environment import AutoSWEEnvironment, MAX_STEPS
    from models import AutoSWEAction, AutoSWEObservation
except ImportError:
    from auto_swe_env.server.auto_swe_environment import AutoSWEEnvironment, MAX_STEPS  # type: ignore
    from auto_swe_env.models import AutoSWEAction, AutoSWEObservation  # type: ignore

logger = logging.getLogger(__name__)

# ── FastAPI app ───────────────────────────────────────────────────────────
app = FastAPI(
    title="Auto-SWE OpenEnv",
    description=(
        "Autonomous Software Engineer — an OpenEnv environment where "
        "AI agents find and fix bugs in a simulated Python codebase."
    ),
    version="0.1.0",
)

# ── Per-session environment storage ───────────────────────────────────────
# For simplicity in the hackathon context we keep a dict of active envs
# keyed by session_id.  The HF Spaces auto-validator and inference.py both
# interact through HTTP, so we manage sessions server-side.
_envs: Dict[str, AutoSWEEnvironment] = {}


def _get_or_create_env(session_id: Optional[str] = None) -> tuple[str, AutoSWEEnvironment]:
    """Return an existing env for *session_id*, or create a new one."""
    if session_id and session_id in _envs:
        return session_id, _envs[session_id]
    sid = session_id or str(uuid4())
    env = AutoSWEEnvironment()
    _envs[sid] = env
    return sid, env


# ── Request / Response schemas ────────────────────────────────────────────
class ResetRequest(BaseModel):
    session_id: Optional[str] = None
    task_id: Optional[str] = None
    seed: Optional[int] = None
    episode_id: Optional[str] = None

    model_config = {"extra": "allow"}


class StepRequest(BaseModel):
    session_id: str
    action: Dict[str, Any]

    model_config = {"extra": "allow"}


class StateResponse(BaseModel):
    session_id: str
    episode_id: str
    step_count: int


class HealthResponse(BaseModel):
    status: str = "healthy"
    environment: str = "auto_swe_env"
    max_steps: int = MAX_STEPS


class ResetResponse(BaseModel):
    session_id: str
    observation: AutoSWEObservation


class StepResponse(BaseModel):
    session_id: str
    observation: AutoSWEObservation


# ── Endpoints ─────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse()


@app.post("/reset", response_model=ResetResponse)
async def reset(req: Optional[ResetRequest] = None) -> ResetResponse:
    """Start a new episode.  Optionally specify a task_id."""
    req = req or ResetRequest()
    sid, env = _get_or_create_env(req.session_id)
    try:
        obs = env.reset(
            seed=req.seed,
            episode_id=req.episode_id,
            task_id=req.task_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return ResetResponse(session_id=sid, observation=obs)


@app.post("/step", response_model=StepResponse)
async def step(req: StepRequest) -> StepResponse:
    """Execute an action in the environment."""
    if req.session_id not in _envs:
        raise HTTPException(status_code=404, detail=f"Unknown session_id: {req.session_id}")
    env = _envs[req.session_id]

    try:
        action = AutoSWEAction(**req.action)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid action: {exc}")

    obs = env.step(action)

    # If episode ended, clean up
    if obs.done:
        env.close()
        del _envs[req.session_id]

    return StepResponse(session_id=req.session_id, observation=obs)


@app.get("/state")
async def get_state(session_id: str) -> StateResponse:
    """Query the current episode state."""
    if session_id not in _envs:
        raise HTTPException(status_code=404, detail=f"Unknown session_id: {session_id}")
    env = _envs[session_id]
    st = env.state
    return StateResponse(
        session_id=session_id,
        episode_id=st.episode_id,
        step_count=st.step_count,
    )


@app.get("/schema")
async def schema() -> Dict[str, Any]:
    """Return the JSON schemas for Action and Observation."""
    return {
        "action_schema": AutoSWEAction.model_json_schema(),
        "observation_schema": AutoSWEObservation.model_json_schema(),
    }


# ── Entry-point ───────────────────────────────────────────────────────────
def main() -> None:
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
