import textwrap
from server.auto_swe_environment import AutoSWEEnvironment
from models import AutoSWEAction

def simulate_episode(env_manager: AutoSWEEnvironment, task_id: str, sequence_name: str, actions: list[AutoSWEAction], expected_score: float):
    print(f"\n{'='*80}")
    print(f"🎬 Starting Simulation: {task_id} - {sequence_name}")
    print(f"{'='*80}")

    obs = env_manager.reset(task_id=task_id)
    print(f"🟢 [ENV RESET] Workspace created. Files: {obs.current_directory}")

    final_grade = None

    for idx, action in enumerate(actions, 1):
        print(f"\n  🤖 [AGENT TURN {idx}] Command: {action.command}")
        if action.path:
            print(f"       Path: {action.path}")
            
        # Execute the action bypassing the FastAPI API layer
        obs = env_manager.step(action)
        
        # Display the Observation
        if obs.error:
            print(f"  🔴 [ENV OBSERVATION] ERROR: {obs.error}")
        else:
            if obs.command_output:
                 print(f"  🟢 [ENV OBSERVATION] Command Executed. Output Length: {len(obs.command_output)} characters.")
            else:
                 print(f"  🟢 [ENV OBSERVATION] Success.")

        if obs.done:
            final_grade = obs.grade
            print(f"\n🏆 [EPISODE COMPLETE] Final Score: {final_grade:.2f} / Expected: {expected_score:.2f}")
            break

    # Assert Mathematical Determinism
    assert final_grade is not None, "🔴 Episode did not terminate using submit_task!"
    assert abs(final_grade - expected_score) < 0.01, f"🔴 Test Failed! Expected {expected_score}, got {final_grade}"
    print(f"✅ TEST PASSED: {sequence_name}")


def main():
    print("🚀 Initializing Offline Python-to-Python OpenEnv Simulation...")
    env_manager = AutoSWEEnvironment()
    
    # ---------------------------------------------------------
    # Test Case 1: Easy Task (syntax_fix) - The Perfect Run
    # ---------------------------------------------------------
    syntax_fix_good = [
        AutoSWEAction(command="read_file", path="math_utils.py"),
        AutoSWEAction(
            command="write_file", 
            path="math_utils.py", 
            content=textwrap.dedent("""\
                def add(a, b):
                    return a + b
                
                def multiply(a, b):
                    return a * b
                
                def subtract(a, b):
                    return a - b
            """)
        ),
        AutoSWEAction(command="run_tests"),
        AutoSWEAction(command="submit_task")
    ]
    simulate_episode(env_manager, "syntax_fix", "The Perfect Run (Expected Score: 1.0)", syntax_fix_good, expected_score=1.0)

    # ---------------------------------------------------------
    # Test Case 2: Medium Task (logic_fix) - The Failure Run
    # ---------------------------------------------------------
    logic_fix_bad = [
        AutoSWEAction(
            command="write_file", 
            path="math_utils.py", 
            content="def add(a, b):\n    raise Exception()\ndef multiply(a, b):\n    raise Exception()\ndef subtract(a, b):\n    raise Exception()\n"
        ),
        AutoSWEAction(command="submit_task")
    ]
    simulate_episode(env_manager, "logic_fix", "The Failure Run (Expected Score: 0.0)", logic_fix_bad, expected_score=0.0)

    # ---------------------------------------------------------
    # Test Case 3: Hard Task (refactor) - The Perfect Run
    # ---------------------------------------------------------
    refactor_good = [
        AutoSWEAction(
            command="write_file", 
            path="math_utils.py", 
            content=textwrap.dedent("""\
                def add(a, b):
                    return a + b
                
                def multiply(a, b):
                    return a * b
                
                def subtract(a, b):
                    return a - b
                
                def compute(a, b, op):  # Successfully refactored calculate to compute
                    if op == 'add':
                        return add(a, b)
                    elif op == 'multiply':
                        return multiply(a, b)
                    return 0
            """)
        ),
        AutoSWEAction(
            command="write_file", 
            path="pipeline.py", 
            content=textwrap.dedent("""\
                from math_utils import compute
                
                def run_pipeline(data):
                    results = []
                    for item in data:
                        results.append(compute(item[0], item[1], item[2]))
                    return results
            """)
        ),
        AutoSWEAction(command="submit_task")
    ]
    simulate_episode(env_manager, "refactor", "The Perfect Refactor (Expected Score: 1.0)", refactor_good, expected_score=1.0)
    
    print(f"\n{'='*80}")
    print("🎉 ALL OFFLINE INTEGRATION TESTS PASSED FOR OPENENV GRADER 🎉")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
