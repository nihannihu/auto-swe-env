"""
Task definitions and deterministic graders for the Auto-SWE environment.

Each task provides:
    - ``description``   : Natural-language prompt shown to the agent.
    - ``setup(path)``   : Writes the buggy mock project into *path*.
    - ``grade(path)``   : Returns a ``float`` in [0.0, 1.0] — **pure Python, no LLM**.
    - ``buggy_files``   : List of files the agent *should* touch (for reward shaping).

Grading is 100 % deterministic:
    * Easy   — ``compile()`` check (syntax only).
    * Medium — ``pytest`` pass ratio.
    * Hard   — file-content checks + ``pytest``.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


# ── base class ────────────────────────────────────────────────────────────
class Task(ABC):
    """Abstract base for all graded tasks."""

    @property
    @abstractmethod
    def task_id(self) -> str: ...

    @property
    @abstractmethod
    def difficulty(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @property
    @abstractmethod
    def buggy_files(self) -> List[str]:
        """Relative paths that contain bugs the agent should fix."""
        ...

    @abstractmethod
    def setup(self, workspace: Path) -> None:
        """Write the buggy project into *workspace* (must already exist)."""
        ...

    @abstractmethod
    def grade(self, workspace: Path) -> float:
        """Return a score in [0.0, 1.0].  Must be **deterministic**."""
        ...


# ── helpers ───────────────────────────────────────────────────────────────
def _write(workspace: Path, rel: str, content: str) -> None:
    fp = workspace / rel
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(textwrap.dedent(content).lstrip("\n"), encoding="utf-8")


def _run_pytest(workspace: Path) -> tuple[int, int]:
    """Run pytest in *workspace* and return (passed, total).

    Falls back to (0, 1) on any error so grading never crashes.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-v", "--tb=short", "-q"],
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=15,
        )
        # Parse pytest -q output: "3 passed" or "1 failed, 2 passed"
        summary = result.stdout.strip().splitlines()
        if not summary:
            return (0, 1)

        last_line = summary[-1]
        passed = 0
        failed = 0
        for token in last_line.replace(",", " ").split():
            if token == "passed":
                idx = last_line.replace(",", " ").split().index(token)
                if idx > 0:
                    try:
                        passed = int(last_line.replace(",", " ").split()[idx - 1])
                    except ValueError:
                        pass
            elif token == "failed":
                idx = last_line.replace(",", " ").split().index(token)
                if idx > 0:
                    try:
                        failed = int(last_line.replace(",", " ").split()[idx - 1])
                    except ValueError:
                        pass

        total = passed + failed
        if total == 0:
            return (0, 1)
        return (passed, total)
    except Exception:
        return (0, 1)


# ══════════════════════════════════════════════════════════════════════════
# EASY — Syntax Error (missing colon)
# ══════════════════════════════════════════════════════════════════════════
class SyntaxFixTask(Task):

    @property
    def task_id(self) -> str:
        return "syntax_fix"

    @property
    def difficulty(self) -> str:
        return "easy"

    @property
    def description(self) -> str:
        return (
            "You are an autonomous SWE. You must use read_file to understand the code, "
            "run_tests to check your work, and submit_task when you are finished.\n"
            "The file `math_utils.py` has a SyntaxError that prevents it from "
            "being imported.  Find and fix the syntax error so the file compiles "
            "successfully.\n\n"
            "HINT: The error is on the function definition line.\n\n"
            "When you have verified the bug is fixed and all tests pass, you MUST output "
            "the exact action {\"command\": \"submit_task\"} to terminate the environment "
            "and receive your final grade."
        )

    @property
    def buggy_files(self) -> List[str]:
        return ["math_utils.py"]

    # -- setup --------------------------------------------------------------
    def setup(self, workspace: Path) -> None:
        # Buggy file — missing colon after function def
        _write(
            workspace,
            "math_utils.py",
            """\
            def add(a, b):
                return a + b

            def multiply(a, b)
                return a * b

            def subtract(a, b):
                return a - b
            """,
        )

        # Tests
        _write(
            workspace,
            "test_math.py",
            """\
            from math_utils import add, multiply, subtract

            def test_add():
                assert add(2, 3) == 5

            def test_multiply():
                assert multiply(3, 4) == 12

            def test_subtract():
                assert subtract(10, 4) == 6
            """,
        )

    # -- grade --------------------------------------------------------------
    def grade(self, workspace: Path) -> float:
        source = (workspace / "math_utils.py").read_text(encoding="utf-8")
        try:
            compile(source, "math_utils.py", "exec")
            return 1.0
        except SyntaxError:
            return 0.0


# ══════════════════════════════════════════════════════════════════════════
# MEDIUM — Logic Bug (addition instead of multiplication)
# ══════════════════════════════════════════════════════════════════════════
class LogicFixTask(Task):

    @property
    def task_id(self) -> str:
        return "logic_fix"

    @property
    def difficulty(self) -> str:
        return "medium"

    @property
    def description(self) -> str:
        return (
            "You are an autonomous SWE. You must use read_file to understand the code, "
            "run_tests to check your work, and submit_task when you are finished.\n"
            "The file `math_utils.py` compiles fine, but there is a **logic bug** "
            "in one of the functions — it returns the wrong result.  Run the tests "
            "to find which function is broken, read the code, and fix the bug.\n\n"
            "HINT: One arithmetic operator is wrong.\n\n"
            "When you have verified the bug is fixed and all tests pass, you MUST output "
            "the exact action {\"command\": \"submit_task\"} to terminate the environment "
            "and receive your final grade."
        )

    @property
    def buggy_files(self) -> List[str]:
        return ["math_utils.py"]

    # -- setup --------------------------------------------------------------
    def setup(self, workspace: Path) -> None:
        # Buggy file — multiply uses + instead of *
        _write(
            workspace,
            "math_utils.py",
            """\
            def add(a, b):
                return a + b

            def multiply(a, b):
                return a + b

            def subtract(a, b):
                return a - b
            """,
        )

        _write(
            workspace,
            "test_math.py",
            """\
            from math_utils import add, multiply, subtract

            def test_add():
                assert add(2, 3) == 5

            def test_multiply():
                assert multiply(3, 4) == 12

            def test_multiply_zero():
                assert multiply(0, 100) == 0

            def test_subtract():
                assert subtract(10, 4) == 6
            """,
        )

    # -- grade --------------------------------------------------------------
    def grade(self, workspace: Path) -> float:
        _write(
            workspace,
            "test_math.py",
            """\
            from math_utils import add, multiply, subtract

            def test_add():
                assert add(2, 3) == 5

            def test_multiply():
                assert multiply(3, 4) == 12

            def test_multiply_zero():
                assert multiply(0, 100) == 0

            def test_subtract():
                assert subtract(10, 4) == 6
            """,
        )
        passed, total = _run_pytest(workspace)
        if total < 4:
            return 0.0
        return round(passed / total, 4)


# ══════════════════════════════════════════════════════════════════════════
# HARD — Multi-file Refactor (rename deprecated function)
# ══════════════════════════════════════════════════════════════════════════
class RefactorTask(Task):

    @property
    def task_id(self) -> str:
        return "refactor"

    @property
    def difficulty(self) -> str:
        return "hard"

    @property
    def description(self) -> str:
        return (
            "You are an autonomous SWE. You must use read_file to understand the code, "
            "run_tests to check your work, and submit_task when you are finished.\n"
            "The codebase contains a deprecated function name `calculate` that "
            "must be renamed to `compute` in **all** files.  The function exists "
            "in `math_utils.py` (definition) and is imported/called in "
            "`pipeline.py`.  Update **both** files so that:\n"
            "  1. The function is defined as `compute` in `math_utils.py`.\n"
            "  2. `pipeline.py` imports and calls `compute` (not `calculate`).\n"
            "  3. All tests pass.\n\n"
            "When you have verified the bug is fixed and all tests pass, you MUST output "
            "the exact action {\"command\": \"submit_task\"} to terminate the environment "
            "and receive your final grade."
        )

    @property
    def buggy_files(self) -> List[str]:
        return ["math_utils.py", "pipeline.py"]

    # -- setup --------------------------------------------------------------
    def setup(self, workspace: Path) -> None:
        # math_utils.py — uses deprecated name "calculate"
        _write(
            workspace,
            "math_utils.py",
            """\
            def add(a, b):
                return a + b

            def multiply(a, b):
                return a * b

            def subtract(a, b):
                return a - b

            def calculate(a, b, op="add"):
                \"\"\"DEPRECATED: rename to 'compute'.\"\"\"
                ops = {"add": add, "multiply": multiply, "subtract": subtract}
                return ops[op](a, b)
            """,
        )

        # pipeline.py — imports the deprecated name
        _write(
            workspace,
            "pipeline.py",
            """\
            from math_utils import calculate

            def run_pipeline(data):
                results = []
                for a, b, op in data:
                    results.append(calculate(a, b, op))
                return results
            """,
        )

        # Tests expect the new name "compute"
        _write(
            workspace,
            "test_math.py",
            """\
            from math_utils import add, multiply, subtract, compute

            def test_add():
                assert add(2, 3) == 5

            def test_multiply():
                assert multiply(3, 4) == 12

            def test_subtract():
                assert subtract(10, 4) == 6

            def test_compute_add():
                assert compute(2, 3, "add") == 5

            def test_compute_multiply():
                assert compute(3, 4, "multiply") == 12
            """,
        )

        _write(
            workspace,
            "test_pipeline.py",
            """\
            from pipeline import run_pipeline

            def test_pipeline():
                data = [(2, 3, "add"), (3, 4, "multiply")]
                assert run_pipeline(data) == [5, 12]
            """,
        )

    # -- grade --------------------------------------------------------------
    def grade(self, workspace: Path) -> float:
        math_src = (workspace / "math_utils.py").read_text(encoding="utf-8")
        pipe_src = (workspace / "pipeline.py").read_text(encoding="utf-8")

        math_updated = "def compute(" in math_src and "def calculate(" not in math_src
        pipe_updated = "compute" in pipe_src and "calculate" not in pipe_src

        if not math_updated and not pipe_updated:
            return 0.0

        # Partial credit — one file updated
        if not (math_updated and pipe_updated):
            return 0.5

        # Both files claim to be updated — restore original tests and run for final confirmation
        _write(
            workspace, "test_math.py",
            """\
            from math_utils import add, multiply, subtract, compute

            def test_add():
                assert add(2, 3) == 5

            def test_multiply():
                assert multiply(3, 4) == 12

            def test_subtract():
                assert subtract(10, 4) == 6

            def test_compute_add():
                assert compute(2, 3, "add") == 5

            def test_compute_multiply():
                assert compute(3, 4, "multiply") == 12
            """
        )
        _write(
            workspace, "test_pipeline.py",
            """\
            from pipeline import run_pipeline

            def test_pipeline():
                data = [(2, 3, "add"), (3, 4, "multiply")]
                assert run_pipeline(data) == [5, 12]
            """
        )
        
        passed, total = _run_pytest(workspace)
        if total == 6 and passed == total:
            return 1.0
        # Both renamed but tests still fail
        return 0.5


# ── registry ──────────────────────────────────────────────────────────────
TASK_REGISTRY: Dict[str, Task] = {
    "syntax_fix": SyntaxFixTask(),
    "logic_fix": LogicFixTask(),
    "refactor": RefactorTask(),
}

ALL_TASK_IDS: List[str] = list(TASK_REGISTRY.keys())
