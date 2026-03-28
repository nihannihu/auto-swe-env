"""
Autonomous Software Engineer — OpenEnv Environment.

An OpenEnv-compatible environment where AI agents must find and fix
bugs in a simulated Python codebase.

Quick start:
    >>> from auto_swe_env import AutoSWEClient
    >>> c = AutoSWEClient("http://localhost:7860")
    >>> obs = c.reset(task_id="syntax_fix")
    >>> obs = c.step({"command": "read_file", "path": "math_utils.py"})
"""

from .models import AutoSWEAction, AutoSWEObservation
from .client import AutoSWEClient

__all__ = [
    "AutoSWEAction",
    "AutoSWEObservation",
    "AutoSWEClient",
]
