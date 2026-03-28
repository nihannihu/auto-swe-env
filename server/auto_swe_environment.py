"""
Core environment implementation for the Autonomous Software Engineer.

This module contains the ``AutoSWEEnvironment`` class — a sandboxed,
episode-isolated environment where an AI agent must find and fix bugs
in a mock Python codebase.

The environment follows the OpenEnv ``Environment`` interface:
    * ``reset()``  — creates a fresh temp workspace with a buggy project.
    * ``step()``   — executes an ``AutoSWEAction`` and returns an ``AutoSWEObservation``.
    * ``state``    — returns the current ``State`` (episode_id, step_count).

Episode lifecycle
-----------------
1.  ``reset(task_id=…)`` sets up the workspace and returns the initial observation.
2.  The agent issues actions (read_file, write_file, run_tests, search_code, list_files).
3.  The episode ends when:
    a. The agent calls **submit_task** (triggers the deterministic grader).
    b. ``step_count >= MAX_STEPS`` (forced termination — grader is still run).
    c. *(never)* by calling ``run_tests`` alone — this just returns test output.
"""

from __future__ import annotations

import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, List, Optional
from uuid import uuid4

try:
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    # Fallback for standalone / lightweight testing without openenv-core
    from pydantic import BaseModel as _BM  # type: ignore[assignment]

    class Action(_BM):  # type: ignore[no-redef]
        pass

    class Observation(_BM):  # type: ignore[no-redef]
        done: bool = False
        reward: float = 0.0
        metadata: dict = {}

    class State(_BM):  # type: ignore[no-redef]
        episode_id: str = ""
        step_count: int = 0

# Import our domain models & tasks
# Handle both in-repo and installed package imports
try:
    from models import AutoSWEAction, AutoSWEObservation
    from tasks import TASK_REGISTRY, ALL_TASK_IDS, Task
except ImportError:
    from auto_swe_env.models import AutoSWEAction, AutoSWEObservation  # type: ignore
    from auto_swe_env.tasks import TASK_REGISTRY, ALL_TASK_IDS, Task  # type: ignore


MAX_STEPS: int = 30


class AutoSWEEnvironment:
    """
    Sandboxed environment for autonomous code-fixing tasks.

    Each episode creates a fresh temporary directory, populates it with
    a buggy project according to the selected task, and tears everything
    down on ``close()`` / ``reset()``.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True  # allow multiple WS sessions

    def __init__(self) -> None:
        self._workspace: Optional[Path] = None
        self._task: Optional[Task] = None
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._cumulative_reward: float = 0.0
        self._done: bool = False
        self._grade: Optional[float] = None

    # ── helpers ───────────────────────────────────────────────────────────

    def _list_files(self) -> List[str]:
        """Return a sorted list of relative paths in the workspace."""
        if self._workspace is None or not self._workspace.exists():
            return []
        files: List[str] = []
        for p in sorted(self._workspace.rglob("*")):
            if p.is_file() and "__pycache__" not in str(p) and ".pytest_cache" not in str(p):
                files.append(str(p.relative_to(self._workspace)))
        return files

    def _make_observation(
        self,
        *,
        reward: float = 0.0,
        file_content: Optional[str] = None,
        command_output: Optional[str] = None,
        search_results: Optional[List[str]] = None,
        error: Optional[str] = None,
    ) -> AutoSWEObservation:
        self._cumulative_reward += reward
        return AutoSWEObservation(
            task_description=self._task.description if self._task else "",
            current_directory=self._list_files(),
            file_content=file_content,
            command_output=command_output,
            search_results=search_results,
            reward=reward,
            done=self._done,
            step_count=self._state.step_count,
            max_steps=MAX_STEPS,
            error=error,
            grade=self._grade,
        )

    def _cleanup_workspace(self) -> None:
        if self._workspace and self._workspace.exists():
            shutil.rmtree(self._workspace, ignore_errors=True)
        self._workspace = None

    # ── OpenEnv interface ─────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        **kwargs: Any,
    ) -> AutoSWEObservation:
        """
        Start a new episode.

        Parameters
        ----------
        seed : int, optional
            Random seed for task selection when *task_id* is ``None``.
        episode_id : str, optional
            Custom episode identifier.
        task_id : str, optional
            One of ``"syntax_fix"``, ``"logic_fix"``, ``"refactor"``.
            If omitted a task is chosen at random.
        """
        # Tear down previous episode
        self._cleanup_workspace()
        self._done = False
        self._grade = None
        self._cumulative_reward = 0.0

        # Select task
        if seed is not None:
            random.seed(seed)

        if task_id is not None:
            if task_id not in TASK_REGISTRY:
                raise ValueError(
                    f"Unknown task_id={task_id!r}. Must be one of {ALL_TASK_IDS}"
                )
            self._task = TASK_REGISTRY[task_id]
        else:
            self._task = TASK_REGISTRY[random.choice(ALL_TASK_IDS)]

        # Fresh workspace
        self._workspace = Path(tempfile.mkdtemp(prefix="autoswe_"))
        self._task.setup(self._workspace)

        # State
        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )

        return self._make_observation()

    def step(
        self,
        action: Any,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> AutoSWEObservation:
        """Execute *action* and return the resulting observation."""
        # Deserialise if we received a raw dict (HTTP/WS layer may pass dicts)
        if isinstance(action, dict):
            action = AutoSWEAction(**action)

        if not isinstance(action, AutoSWEAction):
            self._state.step_count += 1
            return self._make_observation(
                reward=-0.05,
                error=f"Invalid action type: {type(action).__name__}. Expected AutoSWEAction.",
            )

        if self._done:
            return self._make_observation(error="Episode already ended.")

        if self._workspace is None or self._task is None:
            return self._make_observation(
                error="Environment not initialised. Call reset() first.",
            )

        self._state.step_count += 1
        cmd = action.command

        # ── dispatch ──────────────────────────────────────────────────────
        if cmd == "read_file":
            return self._handle_read_file(action)
        elif cmd == "write_file":
            return self._handle_write_file(action)
        elif cmd == "run_tests":
            return self._handle_run_tests()
        elif cmd == "search_code":
            return self._handle_search_code(action)
        elif cmd == "list_files":
            return self._handle_list_files()
        elif cmd == "submit_task":
            return self._handle_submit_task()
        else:
            return self._make_observation(
                reward=-0.05,
                error=f"Unknown command: {cmd}",
            )

    # ── action handlers ───────────────────────────────────────────────────

    def _handle_read_file(self, action: AutoSWEAction) -> AutoSWEObservation:
        if not action.path:
            return self._make_observation(reward=-0.05, error="read_file requires 'path'.")
        target = (self._workspace / action.path).resolve()  # type: ignore[operator]
        # Sandbox check
        if not str(target).startswith(str(self._workspace)):
            return self._make_observation(reward=-0.1, error="Path escapes workspace sandbox.")
        if not target.is_file():
            return self._make_observation(reward=-0.05, error=f"File not found: {action.path}")

        content = target.read_text(encoding="utf-8", errors="replace")

        # Reward shaping: bonus for reading a buggy file
        reward = 0.05
        if self._task and action.path in self._task.buggy_files:
            reward = 0.1

        return self._make_observation(reward=reward, file_content=content)

    def _handle_write_file(self, action: AutoSWEAction) -> AutoSWEObservation:
        if not action.path:
            return self._make_observation(reward=-0.05, error="write_file requires 'path'.")
        if action.content is None:
            return self._make_observation(reward=-0.05, error="write_file requires 'content'.")

        target = (self._workspace / action.path).resolve()  # type: ignore[operator]
        if not str(target).startswith(str(self._workspace)):
            return self._make_observation(reward=-0.1, error="Path escapes workspace sandbox.")

        # If writing a .py file, validate syntax first for reward shaping
        reward = 0.05
        if action.path.endswith(".py"):
            try:
                compile(action.content, action.path, "exec")
            except SyntaxError as e:
                reward = -0.1  # penalise writing broken code

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(action.content, encoding="utf-8")
        return self._make_observation(reward=reward)

    def _handle_run_tests(self) -> AutoSWEObservation:
        """Run pytest — does **NOT** end the episode."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "-v", "--tb=short"],
                cwd=str(self._workspace),
                capture_output=True,
                text=True,
                timeout=15,
            )
            output = result.stdout + "\n" + result.stderr
        except subprocess.TimeoutExpired:
            output = "ERROR: pytest timed out after 15 seconds."
        except Exception as exc:
            output = f"ERROR: Failed to run tests — {exc}"

        return self._make_observation(reward=0.0, command_output=output.strip())

    def _handle_search_code(self, action: AutoSWEAction) -> AutoSWEObservation:
        if not action.query:
            return self._make_observation(reward=-0.05, error="search_code requires 'query'.")

        matches: List[str] = []
        for fp in sorted(self._workspace.rglob("*")):  # type: ignore[union-attr]
            if (
                fp.is_file()
                and fp.suffix in (".py", ".txt", ".md", ".cfg", ".toml", ".yaml", ".yml")
                and "__pycache__" not in str(fp)
            ):
                try:
                    for i, line in enumerate(
                        fp.read_text(encoding="utf-8", errors="replace").splitlines(), 1
                    ):
                        if action.query in line:
                            rel = str(fp.relative_to(self._workspace))
                            matches.append(f"{rel}:{i}: {line.strip()}")
                except Exception:
                    continue

        return self._make_observation(reward=0.05, search_results=matches)

    def _handle_list_files(self) -> AutoSWEObservation:
        return self._make_observation(reward=0.0)

    def _handle_submit_task(self) -> AutoSWEObservation:
        """
        Terminal action — runs the deterministic grader and ends the episode.
        """
        assert self._task is not None and self._workspace is not None
        self._grade = self._task.grade(self._workspace)
        self._done = True
        return self._make_observation(reward=self._grade)

    # ── state / lifecycle ─────────────────────────────────────────────────

    @property
    def state(self) -> State:
        return self._state

    def close(self) -> None:
        self._cleanup_workspace()
