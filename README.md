# 🚀 Auto-SWE: The Definitive Benchmark for Autonomous Software Engineers

> *A mathematically sound, deterministic OpenEnv environment where AI agents must find, understand, and fix bugs in complex multi-file Python codebases.*

Built for the **Meta × Hugging Face OpenEnv AI Hackathon 2026**.

---

## 🌟 The Motivation (Real-World Utility)

Why another AI environment? Because finding the next breakout coding LLM requires stepping beyond "toy" benchmarks like tic-tac-toe or single-function leetcode puzzles.

Modern software teams spend **30–50% of engineering time** on bug triage, legacy code archaeology, and system refactoring. **Auto-SWE natively simulates real-world DevOps workflows.**
To succeed, an agent must:
1. Parse error stack traces and discover the root fault.
2. Read, modify, and orchestrate tests across multiple interconnected `.py` files.
3. Handle terminal execution via `pytest` to verify its own logic iteravely.

If an LLM can score highly on Auto-SWE, it's ready to handle enterprise maintenance tickets on your GitHub repository.

---

## 📐 Architecture & State Space

Auto-SWE implements a strictly typed constraint system mapped elegantly onto the OpenEnv HTTP standard and Pydantic schemas. 

Every episode spins up an intensely isolated, cryptographically serialized `tempfile` workspace. When the episode resets, the state is vaporized. Zero overlap, zero contamination.

### 🎮 The Action Space (`AutoSWEAction`)
Agents control the environment by securely navigating these specific JSON commands.

```json
{
  "command": "read_file | write_file | run_tests | search_code | list_files | submit_task",
  "path": "relative/path_to_file.py",
  "content": "Entire file content string when using write_file",
  "query": "Regex search string when using search_code"
}
```

### 👁️ The Observation Space (`AutoSWEObservation`)
The environment returns comprehensive JSON summaries modeling a local IDE terminal state.

```json
{
  "task_description": "The file `math_utils.py` has a logic bug...",
  "current_directory": ["math_utils.py", "test_math.py", "pipeline.py"],
  "file_content": "def add(a, b):\n    return a + b",
  "command_output": "STDOUT / STDERR from the pytest validation suite",
  "search_results": ["math_utils.py:12: def compute(a, b):"],
  "reward": -0.1,
  "done": false,
  "step_count": 3,
  "max_steps": 30,
  "error": "If the agent hallucinates JSON, the parsing traceback goes here.",
  "grade": null
}
```

### 📈 Continuous Reward Shaping
We use dense reward mechanisms to guide smaller models to convergence before the terminal step:
- Read the correctly broken file: `+0.10`
- Successfully write valid python code: `+0.05`
- Generate syntactically invalid Python (e.g. missed block indents): `−0.10`
- Call invalid commands: `−0.05`

---

## 🏆 The Tasks & Exploit-Proof Grading

LLM-as-a-Judge introduces extreme variance and sycophancy. Our test harnesses are **100% deterministic, Python-native execution validators.** 

Furthermore, to prevent the agent from gaining a false `1.0` grade by maliciously deleting the assertions within the `test_math.py` file, our environment physically **overwrites and restores the ground-truth test suite** into the filesystem exactly 1 millisecond before the `_run_pytest()` grader triggers!

| Complexity | Task ID | Core Challenge | Deterministic Grade Logic |
|---|---|---|---|
| 🟢 Easy | `syntax_fix` | Resolve a missing `:` syntax termination | Validates Python AST `compile()` integrity. |
| 🟡 Medium | `logic_fix` | Find `+` used instead of `*` causing algorithmic drift | `pytest_passed / total_assertions_found` |
| 🔴 Hard | `refactor` | Decouple a deprecated `calculate` function across imports | Both physical structural regex validation AND multi-file cross-test suites must output a perfect zero-fault assertion metric. |

---

## 🚀 Quickstart & Deployment

Auto-SWE is designed for highly resilient **Containerized Deployments (2 vCPU / 8 GB RAM)**. The `Dockerfile` mounts the environment optimally targeting the **Hugging Face Spaces Port (7860)**.

### Run the Environment (Locally or inside Docker)
```bash
# Clone the repository
git clone https://github.com/your-username/auto-swe.git && cd auto-swe

# Run via Docker (Ready for HF Spaces!)
docker build -t auto-swe-env .
docker run -p 7860:7860 auto-swe-env
```

### Run the Evaluation Baseline
We've included `evaluate_variance.py`, an external evaluation harness built to connect to your live Hugging Face URL or local instance to test dynamic models via the OpenAI SDK wrapper.
```bash
export API_BASE_URL="https://api-inference.huggingface.co/models/meta-llama/Llama-3.3-70B-Instruct/v1"
export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
export HF_TOKEN="hf_xyz123"

# Programmatically evaluate all three tasks inside 20 minutes
python evaluate_variance.py
```

---

## 🪪 Compliance Check

✅ **Stateless Isolation**: Fully achieved via `pathlib` Temporary Directories.  
✅ **Deterministic Grading**: 100% Python AST and standard `subprocess` Pytest.  
✅ **Strict OpenEnv Schema Compliance**: All operations mapped seamlessly via Pydantic bounds and OpenEnv `openenv.yaml` runtime targets.  
✅ **Execution Timer Limits**: Bound strictly below the 20-minute constraints.

**License:** MIT
