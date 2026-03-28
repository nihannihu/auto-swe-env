#!/usr/bin/env python3
"""
Baseline inference script for the Autonomous Software Engineer environment.

Uses the **openai** Python client to interact with any OpenAI-compatible
API endpoint (Hugging Face TGI, vLLM, etc.).

Required environment variables
-------------------------------
    API_BASE_URL   — e.g. ``https://api-inference.huggingface.co/models/…/v1``
    MODEL_NAME     — e.g. ``meta-llama/Llama-3.3-70B-Instruct``
    HF_TOKEN       — Hugging Face API token

Usage
-----
    # Start the environment server first:
    #   uvicorn server.app:app --host 0.0.0.0 --port 7860

    export API_BASE_URL="https://api-inference.huggingface.co/models/meta-llama/Llama-3.3-70B-Instruct/v1"
    export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
    export HF_TOKEN="hf_..."
    python inference.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import textwrap
from typing import Any, Dict, Optional

import requests

# ── configuration ─────────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "")
MODEL_NAME = os.environ.get("MODEL_NAME", "")
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY", "")
ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")

TASKS_TO_EVALUATE = ["syntax_fix", "logic_fix", "refactor"]


# ── LLM helper ────────────────────────────────────────────────────────────

def _call_llm(messages: list[dict[str, str]]) -> str:
    """
    Call the LLM via the openai-compatible chat/completions endpoint.

    Uses the ``openai`` Python package.
    """
    from openai import OpenAI

    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN,
    )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,  # type: ignore[arg-type]
        temperature=0.0,
        max_tokens=1024,
    )
    return response.choices[0].message.content or ""


def _parse_action(llm_output: str) -> Dict[str, Any]:
    """
    Extract a JSON action dict from the LLM's response using aggressive RegEx.
    """
    match = re.search(r'\{.*\}', llm_output, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in output.")
    
    raw = match.group(0)
    return json.loads(raw)


# ── environment helpers ───────────────────────────────────────────────────

def env_reset(task_id: str, session_id: Optional[str] = None) -> tuple[str, dict]:
    """POST /reset and return (session_id, observation_dict)."""
    payload: dict[str, Any] = {"task_id": task_id}
    if session_id:
        payload["session_id"] = session_id
    resp = requests.post(f"{ENV_URL}/reset", json=payload)
    resp.raise_for_status()
    data = resp.json()
    return data["session_id"], data["observation"]


def env_step(session_id: str, action: dict) -> dict:
    """POST /step and return the observation dict."""
    resp = requests.post(
        f"{ENV_URL}/step",
        json={"session_id": session_id, "action": action},
    )
    resp.raise_for_status()
    return resp.json()["observation"]


# ── system prompt ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""\
You are an autonomous software engineer.  You are given a task to fix bugs
in a Python codebase.  You interact with the codebase using JSON actions.

Available actions (return EXACTLY one JSON object per turn):

1. Read a file:
   {"command": "read_file", "path": "<relative_path>"}

2. Write / overwrite a file:
   {"command": "write_file", "path": "<relative_path>", "content": "<full_file_content>"}

3. Run pytest:
   {"command": "run_tests"}

4. Search for a pattern across files:
   {"command": "search_code", "query": "<search_string>"}

5. List all files:
   {"command": "list_files"}

6. Submit your solution (ends the episode, triggers grading):
   {"command": "submit_task"}

Strategy:
- First, list and read the relevant files to understand the bug.
- Then fix the code by writing the corrected file(s).
- Run tests to verify.  If they fail, read the output, fix, and retry.
- When all tests pass, call submit_task.

IMPORTANT: Reply with ONLY a JSON object, no explanation.
""")

SCHEMA_ERROR_TEMPLATE = (
    "Invalid action format. Must be valid JSON matching the schema.\n"
    "Expected format: {{\"command\": \"...\", \"path\": \"...\", \"content\": \"...\", \"query\": \"...\"}}\n"
    "Error details: {exc}"
)


# ── main loop ─────────────────────────────────────────────────────────────

def run_task(task_id: str, mock_responses: list[str] = None) -> dict:
    """Run a single task end-to-end. Returns a dict of metrics."""
    print(f"\n{'='*60}")
    print(f"  TASK: {task_id}")
    print(f"{'='*60}")

    session_id, obs = env_reset(task_id)
    print(f"  Session: {session_id}")
    print(f"  Files  : {obs['current_directory']}")
    print(f"  Task   : {obs['task_description'][:80]}…")

    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"TASK: {obs['task_description']}\n\n"
                f"FILES IN WORKSPACE: {json.dumps(obs['current_directory'])}\n\n"
                "What is your first action?  Reply with a JSON object."
            ),
        },
    ]

    max_turns = obs.get("max_steps", 30)
    syntax_errors = 0
    status = "max_steps_reached"

    for turn in range(1, max_turns + 1):
        # Call LLM
        try:
            if mock_responses:
                llm_response = mock_responses.pop(0) if mock_responses else '{"command": "submit_task"}'
            else:
                llm_response = _call_llm(messages)
        except Exception as exc:
            print(f"  [Turn {turn}] LLM error: {exc}")
            status = "crashed"
            break

        # Parse action
        try:
            action = _parse_action(llm_response)
        except Exception as exc:
            print(f"  [Turn {turn}] Parse error: {exc}")
            # Feed error back to LLM to self-correct
            err_msg = SCHEMA_ERROR_TEMPLATE.format(exc=str(exc))
            feedback = f"ERROR: {err_msg}\n\nStep {turn}/{max_turns}\n\nNext action?"
            messages.append({"role": "assistant", "content": llm_response})
            messages.append({"role": "user", "content": feedback})
            continue

        cmd = action.get("command", "?")
        print(f"  [Turn {turn}] Action: {cmd}", end="")
        if cmd in ("read_file", "write_file"):
            print(f"  path={action.get('path', '?')}", end="")
        print()

        # Execute step
        obs = env_step(session_id, action)

        if obs.get("reward", 0.0) == -0.1 and cmd == "write_file":
            syntax_errors += 1

        # Build user message for LLM from observation
        parts: list[str] = []
        if obs.get("error"):
            parts.append(f"ERROR: {obs['error']}")
        if obs.get("file_content") is not None:
            parts.append(f"FILE CONTENT:\n```\n{obs['file_content']}\n```")
        if obs.get("command_output") is not None:
            parts.append(f"TEST OUTPUT:\n```\n{obs['command_output']}\n```")
        if obs.get("search_results") is not None:
            parts.append(f"SEARCH RESULTS:\n" + "\n".join(obs["search_results"]))
        if obs.get("current_directory"):
            parts.append(f"FILES: {json.dumps(obs['current_directory'])}")
        if obs.get("grade") is not None:
            parts.append(f"FINAL GRADE: {obs['grade']}")

        feedback = "\n\n".join(parts) if parts else "Action executed."
        feedback += f"\n\nStep {obs['step_count']}/{obs['max_steps']}"

        messages.append({"role": "assistant", "content": llm_response})
        messages.append({"role": "user", "content": feedback + "\n\nNext action?"})

        if obs.get("done"):
            grade = obs.get("grade", 0.0)
            status = "submitted"
            print(f"  ✅ Episode ended — grade: {grade}")
            return {
                "score": grade,
                "steps": obs.get("step_count", turn),
                "syntax_errors": syntax_errors,
                "status": status
            }

    # If we ran out of turns without submitting
    print("  ⚠️  Max turns reached without submit_task")
    return {
        "score": 0.0,
        "steps": max_turns,
        "syntax_errors": syntax_errors,
        "status": status
    }


def main() -> None:
    # Validate env vars
    missing = []
    if not API_BASE_URL:
        missing.append("API_BASE_URL")
    if not MODEL_NAME:
        missing.append("MODEL_NAME")
    if not HF_TOKEN:
        missing.append("OPENAI_API_KEY (or HF_TOKEN)")

    if missing:
        print(f"ERROR: Missing required environment variables: {', '.join(missing)}")
        print("Set them before running inference.py")
        sys.exit(1)

    print("╔════════════════════════════════════════════════════════════╗")
    print("║  Auto-SWE Baseline Inference                             ║")
    print(f"║  Model : {MODEL_NAME[:50]:<50} ║")
    print(f"║  Env   : {ENV_URL:<50} ║")
    print("╚════════════════════════════════════════════════════════════╝")

    results: dict[str, dict] = {}
    for tid in TASKS_TO_EVALUATE:
        try:
            metrics = run_task(tid)
        except Exception as exc:
            print(f"  ❌ Task {tid} crashed: {exc}")
            metrics = {"score": 0.0, "steps": 0, "syntax_errors": 0, "status": "crashed"}
        results[tid] = metrics

    # Summary
    print(f"\n{'='*60}")
    print("  RESULTS SUMMARY")
    print(f"{'='*60}")
    for tid, m in results.items():
        grade = m["score"]
        bar = "█" * int(grade * 20) + "░" * (20 - int(grade * 20))
        print(f"  {tid:<14} {bar}  {grade:.2f} ({m['steps']} steps, {m['syntax_errors']} syntax errs)")
    avg = sum(m["score"] for m in results.values()) / len(results) if results else 0.0
    print(f"\n  Aggregate Score: {avg:.4f}")


if __name__ == "__main__":
    main()
