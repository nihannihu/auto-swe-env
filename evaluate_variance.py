#!/usr/bin/env python3
"""
Phase 2 Evaluation Harness

Programmatically runs the Auto-SWE environment across all tasks and tracks:
1. Final Score
2. Total Steps Taken
3. Syntax Error Count
4. Task Completion Status
5. Execution Time

Usage:
  python evaluate_variance.py
"""

import time
import json
import os
from inference import run_task, TASKS_TO_EVALUATE

# Inject some mock sequences for our offline test
MOCK_SCENARIOS = {
    "syntax_fix": [
        '{"command": "read_file", "path": "math_utils.py"}',
        # Syntax error injected to trigger regex recovery and tracking
        '{"command": "write_file", "path": "math_utils.py", "content": "def add(a, b):\\n    return a + b\\n\\ndef multiply(a, b):\\n    return a * b\\n\\ndef subtract(a, b):\\n    return a - b\\n"}',
        '{"command": "run_tests"}',
        '{"command": "submit_task"}'
    ],
    "logic_fix": [
        '{"command": "read_file", "path": "math_utils.py"}',
        # Valid code
        '{"command": "write_file", "path": "math_utils.py", "content": "def add(a, b):\\n    return a + b\\n\\ndef multiply(a, b):\\n    return a * b\\n\\ndef subtract(a, b):\\n    return a - b\\n"}',
        '{"command": "run_tests"}',
        '{"command": "submit_task"}'
    ],
    "refactor": [
        '{"command": "read_file", "path": "math_utils.py"}',
        '```markdown\\nHey, I am not formatting my JSON properly.\\n```', # Regex extraction / fallback test
        '{"command": "submit_task"}'
    ]
}

def main() -> None:
    print("🚀 Starting Evaluation Harness...")
    results = []
    total_eval_time = 0.0

    for task_id in TASKS_TO_EVALUATE:
        print(f"\nEvaluating: {task_id}")
        start_time = time.time()
        
        # Pull our mock LLM responses for this specific task
        mock_resp = list(MOCK_SCENARIOS.get(task_id, ['{"command": "submit_task"}']))
        
        metrics = run_task(task_id, mock_responses=mock_resp)
        elapsed = time.time() - start_time
        total_eval_time += elapsed

        metrics["task_id"] = task_id
        metrics["time_seconds"] = elapsed
        results.append(metrics)
        time.sleep(0.5)

    # Compile the Markdown Report
    report_lines = [
        "# Auto-SWE Evaluation Report",
        "",
        "## Overall Metrics",
        f"- **Total Execution Time**: {total_eval_time:.2f} seconds",
    ]

    # Verify the 20 minute limit (1200 seconds)
    if total_eval_time <= 1200.0:
        report_lines.append("- **Evaluation Status**: ✅ Passed (Under 20 mins)")
    else:
        report_lines.append("- **Evaluation Status**: ❌ FAILED OVERTIME LIMIT")
        
    report_lines.extend([
        "",
        "## Task Breakdown",
        "| Task ID | Status | Final Score | Steps | Syntax Errors | Time Taken (s) |",
        "|---|---|---|---|---|---|"
    ])

    for r in results:
        report_lines.append(f"| {r['task_id']} | {r['status']} | **{r['score']:.4f}** | {r['steps']} | {r['syntax_errors']} | {r['time_seconds']:.2f} |")

    report_path = os.path.join(os.getcwd(), "evaluation_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
        
    print(f"\n📊 Evaluation complete! Report generated at: {report_path}")

if __name__ == "__main__":
    main()
