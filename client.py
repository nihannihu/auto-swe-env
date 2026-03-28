"""
Lightweight HTTP client for the Auto-SWE environment.

This client talks directly to the FastAPI server over HTTP and provides
a Pythonic interface for ``reset()``, ``step()``, and ``state()``.

It does **not** depend on ``openenv-core`` so it can be used standalone:

    >>> from client import AutoSWEClient
    >>> c = AutoSWEClient("http://localhost:7860")
    >>> obs = c.reset(task_id="syntax_fix")
    >>> obs = c.step({"command": "read_file", "path": "math_utils.py"})
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import requests

try:
    from models import AutoSWEAction, AutoSWEObservation
except ImportError:
    from auto_swe_env.models import AutoSWEAction, AutoSWEObservation  # type: ignore


class AutoSWEClient:
    """
    Synchronous HTTP client for the Auto-SWE OpenEnv server.
    """

    def __init__(self, base_url: str = "http://localhost:7860") -> None:
        self.base_url = base_url.rstrip("/")
        self.session_id: Optional[str] = None

    # ── public API ────────────────────────────────────────────────────────

    def reset(
        self,
        task_id: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> AutoSWEObservation:
        """Reset the environment and start a new episode."""
        payload: Dict[str, Any] = {}
        if self.session_id:
            payload["session_id"] = self.session_id
        if task_id:
            payload["task_id"] = task_id
        if seed is not None:
            payload["seed"] = seed

        resp = requests.post(f"{self.base_url}/reset", json=payload)
        resp.raise_for_status()
        data = resp.json()
        self.session_id = data["session_id"]
        return AutoSWEObservation(**data["observation"])

    def step(self, action: Dict[str, Any] | AutoSWEAction) -> AutoSWEObservation:
        """Execute an action and return the observation."""
        if self.session_id is None:
            raise RuntimeError("Call reset() before step().")
        if isinstance(action, AutoSWEAction):
            action = action.model_dump()
        payload = {"session_id": self.session_id, "action": action}
        resp = requests.post(f"{self.base_url}/step", json=payload)
        resp.raise_for_status()
        data = resp.json()
        obs = AutoSWEObservation(**data["observation"])
        if obs.done:
            self.session_id = None  # session cleaned up server-side
        return obs

    def state(self) -> Dict[str, Any]:
        """Query the current episode state."""
        if self.session_id is None:
            raise RuntimeError("No active session.")
        resp = requests.get(
            f"{self.base_url}/state", params={"session_id": self.session_id}
        )
        resp.raise_for_status()
        return resp.json()

    def health(self) -> Dict[str, Any]:
        """Check server health."""
        resp = requests.get(f"{self.base_url}/health")
        resp.raise_for_status()
        return resp.json()
