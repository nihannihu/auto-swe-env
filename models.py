"""
Pydantic models for the Autonomous Software Engineer OpenEnv Environment.

Defines the strictly-typed Action, Observation, and State models used
throughout the environment. All data flowing through step()/reset()
is validated by these schemas.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------
class AutoSWEAction(BaseModel):
    """
    An action the agent can take inside the simulated codebase.

    Supported commands
    ------------------
    - **read_file**   : Read a file from the workspace.  Requires ``path``.
    - **write_file**  : Write / overwrite a file.        Requires ``path`` and ``content``.
    - **run_tests**   : Execute ``pytest`` on the workspace.  No extra args.
    - **search_code** : Grep for a pattern across all files.  Requires ``query``.
    - **list_files**  : List every file in the workspace.     No extra args.
    - **submit_task** : Signal that the agent is done and trigger the final grader.
    """

    command: Literal[
        "read_file",
        "write_file",
        "run_tests",
        "search_code",
        "list_files",
        "submit_task",
    ] = Field(..., description="The command the agent wants to execute. WARNING: When you have verified the bug is fixed and all tests pass, you MUST output the exact action {'command': 'submit_task'} to terminate the environment.")

    path: Optional[str] = Field(
        default=None,
        description="Relative file path inside the workspace (for read_file / write_file).",
    )
    content: Optional[str] = Field(
        default=None,
        description="File content to write (for write_file).",
    )
    query: Optional[str] = Field(
        default=None,
        description="Search pattern (for search_code).",
    )


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------
class AutoSWEObservation(BaseModel):
    """
    The observation returned to the agent after every ``step()`` or ``reset()`` call.
    """

    task_description: str = Field(
        ..., description="Natural-language description of the current task."
    )
    current_directory: List[str] = Field(
        default_factory=list,
        description="List of relative file paths currently in the workspace.",
    )
    file_content: Optional[str] = Field(
        default=None,
        description="Content of the file that was just read (read_file).",
    )
    command_output: Optional[str] = Field(
        default=None,
        description="Stdout / stderr from run_tests or other shell commands.",
    )
    search_results: Optional[List[str]] = Field(
        default=None,
        description="Lines matching the search query (search_code).",
    )
    reward: float = Field(
        default=0.0,
        description="Immediate reward for the last action (continuous, can be negative).",
    )
    done: bool = Field(
        default=False,
        description="Whether the episode has ended.",
    )
    step_count: int = Field(
        default=0,
        description="Number of steps taken so far in this episode.",
    )
    max_steps: int = Field(
        default=30,
        description="Maximum steps allowed before forced termination.",
    )
    error: Optional[str] = Field(
        default=None,
        description="Human-readable error message if the action was invalid.",
    )
    grade: Optional[float] = Field(
        default=None,
        description="Final deterministic grade (0.0–1.0) — only set when done=True.",
    )
