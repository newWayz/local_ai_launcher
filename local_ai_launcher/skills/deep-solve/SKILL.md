---
name: deep-solve
description: Zero-shot ledger-based self-orchestration (GVS5H) for solving complex algorithms, mathematically dense modules, hard debugging, or logic synthesis using local Qwen 3.8 27B without context bloat or runaway reasoning.
---

# Deep Solve (GVS5H Self-Orchestrator)

The `deep-solve` skill invokes the **GVS5H** (*Gorilla vs. 5 Humans*) ledger-based self-orchestration engine. It executes a multi-turn, 5-stage sequential optimization loop (Planning $\to$ Ideation $\to$ Curation $\to$ Execution $\to$ Test Verification) using the locally served `qwen-38_27b-blackwell-256k-think` model via LocalAI.

Use this skill when:
* Implementing complex algorithms (dynamic programming, graphs, automata, parsers, state machines).
* A single-turn generation repeatedly fails subtle edge cases or gets stuck in repetitive reasoning.
* Refactoring or optimizing an isolated module with verifiable assertions or unit tests.

## Why use `deep-solve` instead of standard single-turn coding?
* **Zero Context Bloat**: Every worker invocation runs in a fresh, bounded context ($\le 8\text{k}$ tokens), operating at the highest precision band of the local Q4 KV cache.
* **Separation of Ideation & Execution**: The ideator explores algorithmic approaches and bounds in pure prose before any code is generated, preventing runaway edge-case interrogations.
* **Subprocess Test Gate**: Code is compiled and tested against a real test suite in an isolated sandbox. The manager is programmatically forbidden from signing off on a failing test suite.
* **Regression Rollback**: Tracks the best-performing solution across rounds, automatically rolling back if later iterations introduce bugs.

## How to Invoke

From your shell or tool execution:
```bash
python -m local_ai_launcher.bridges.gvs5h_solver \
  --task "<Clear, detailed task description and algorithmic requirements>" \
  --target "<Path to output file, e.g. src/my_app/solver.py>" \
  --test-cmd "<Command to verify code, e.g. pytest tests/test_solver.py>"
```

### Optional Arguments:
* `--max-iters <N>`: Maximum manager-worker iteration cycles (default: 6).
* `--max-seconds <S>`: Wall-clock watchdog limit before triggering clean finalization (default: 180s).
* `--model <model_id>`: Target model (default: `qwen-38_27b-blackwell-256k-think`).
* `--endpoint <url>`: API endpoint (default: `http://localhost:8080/v1/chat/completions`).

## Output
The command returns a JSON object summarizing the run:
```json
{
  "status": "success",
  "delivered": true,
  "target_file": "src/my_app/solver.py",
  "iterations": 3,
  "duration_seconds": 78.4,
  "tests_passed": "5/5",
  "logs_dir": "C:\\aiWorkspace\\logs\\gvs5h\\..."
}
```
Existing target files are backed up to `<target_file>.bak` before modification. Full conversation transcripts are archived to `logs/gvs5h/`.
