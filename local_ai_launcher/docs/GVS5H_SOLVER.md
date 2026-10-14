You can call gvs5h tool (from https://github.com/slee-persis/GVS5H ) 
in **three ways**, depending on how you are working:

---

### Method 1: In Pi or Hermes (Natural Language)

Because the skill is registered in `.pi/skills/deep-solve` and `.hermes/skills/deep-solve`, you can invoke it conversationally while working in Pi or Hermes.

Just mention **`deep-solve`** or describe a hard algorithmic task:

> **Prompt to Pi or Hermes:**  
> *"Use deep-solve to implement an efficient A\* pathfinding algorithm in `src/pathfinder.py` and test it with `pytest tests/test_pathfinder.py`."*

#### What happens behind the scenes:
1. Pi or Hermes recognizes the `deep-solve` skill from its registry.
2. It pauses its conversational generation and runs the solver tool in a child process.
3. Your terminal displays the live GVS5H logs:
   ```text
   [primary] plan created with 4 initial tasks
   [worker:ideate] brainstormed 3 candidate approaches (prose only)
   [primary-manage] status: continue, next: Implement priority queue state
   [worker] task 1 -> continue, wrote_code=True
   [verifier] [TESTS: PASSED all 6 checks]
   [primary-manage] status: done
   [GVS5H] Created backup of target file: pathfinder.py.bak
   [GVS5H] Successfully delivered verified code to src/pathfinder.py
   ```
4. Pi / Hermes unblocks, sees that the tests passed, and reports back to you.

---

### Method 2: Directly from PowerShell / Command Prompt

You can run it directly from your terminal in `C:\aiWorkspace` without even opening Pi or Hermes:

```powershell
.venv\Scripts\python.exe -m local_ai_launcher.bridges.gvs5h_solver `
  --task "Implement an LRU Cache class with O(1) get and put operations" `
  --target "src/cache.py" `
  --test-cmd "pytest tests/test_cache.py"
```

#### Common Arguments:
| Argument | Description | Default |
| :--- | :--- | :--- |
| **`--task`** | Description of the function, algorithm, or bug to fix. | *(Required)* |
| **`--target`** | Path to the output file to create or update. | *(Required)* |
| **`--test-cmd`** | Verification command (`pytest`, `python -m unittest`, inline assertions). If omitted, validates syntax. | `None` |
| **`--max-iters`** | Maximum manager-worker loop cycles. | `6` |
| **`--max-seconds`**| Watchdog timeout (finishes cleanly before this limit). | `180` |
| **`--model`** | Target model on LocalAI. | `qwen-38_27b-blackwell-256k-think` |
| **`--endpoint`** | LocalAI endpoint. | `http://localhost:8080/v1/chat/completions` |

#### Quick Inline Test Example:
```powershell
.venv\Scripts\python.exe -m local_ai_launcher.bridges.gvs5h_solver `
  --task "Create a function is_palindrome(s: str) -> bool that ignores punctuation" `
  --target "scratch/palindrome.py" `
  --test-cmd "python -c 'import palindrome; assert palindrome.is_palindrome(\"A man, a plan, a canal: Panama\")'"
```

---

### Method 3: Programmatically from Python

If you want to invoke it inside an automated script or Python pipeline:

```python
from local_ai_launcher.bridges.gvs5h_solver import run_gvs5h_solve

result = run_gvs5h_solve(
    task="Implement a thread-safe rate limiter with token bucket algorithm",
    target_file="src/rate_limiter.py",
    test_cmd="pytest tests/test_rate_limiter.py",
    max_iters=5,
    max_seconds=150
)

print("Status:", result["status"])         # "success" or "failed"
print("Delivered:", result["delivered"])   # True
print("Tests:", result["tests_passed"])    # "8/8"
print("Audit Log:", result["logs_dir"])    # "C:\aiWorkspace\logs\gvs5h\..."
```
