"""
Tests for GVS5H Zero-Shot Self-Orchestration Bridge (gvs5h_solver).
Verifies parsing helpers, sandbox lifecycle, dynamic sampling parameters,
test subprocess execution, best-so-far regression rollback, and atomic delivery.
"""

import os
import sys
import json
import time
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from local_ai_launcher.bridges.gvs5h_solver import (
    strip_think,
    strip_code,
    parse_sections,
    parse_bullets,
    extract_python_code,
    parse_tasks,
    cleanup_stale_sandboxes,
    prune_old_gvs5h_logs,
    DEFAULT_MAX_LOG_RUNS,
    GVS5HSolver,
    run_gvs5h_solve,
    ensure_deep_solve_skills,
    main as gvs5h_main,
    WS_ROOT,
    LOGS_ROOT
)


def test_strip_think():
    text = "<think>Let me reason about this problem...\nFirst consider edge cases.</think>\nFinal Answer."
    assert strip_think(text) == "Final Answer."
    assert strip_think("No think block here") == "No think block here"
    assert strip_think("") == ""


def test_strip_code():
    text = "Here is an idea:\n```python\ndef solve(): pass\n```\nAnd more text."
    stripped = strip_code(text)
    assert "def solve" not in stripped
    assert "[code omitted -- algorithmic approach only]" in stripped
    assert "Here is an idea:" in stripped


def test_parse_sections():
    raw = """
<think>discard me</think>
### PLAN
We will implement an efficient Trie.
Line two of plan.

### TASKS
- Create Node class
- Implement insert
- Implement search

### NOTES
Trie lookup is O(L).
"""
    sections = parse_sections(raw)
    assert "PLAN" in sections
    assert "We will implement an efficient Trie." in sections["PLAN"]
    assert "TASKS" in sections
    assert "- Create Node class" in sections["TASKS"]
    assert "NOTES" in sections
    assert "Trie lookup is O(L)." in sections["NOTES"]


def test_parse_bullets():
    text = "- First item\n* Second item\n1. Third item\n+ Fourth item"
    bullets = parse_bullets(text)
    assert len(bullets) == 4
    assert bullets[0] == "First item"
    assert bullets[1] == "Second item"
    assert bullets[2] == "Third item"
    assert bullets[3] == "Fourth item"


def test_extract_python_code():
    markdown = """
Here is the solution:
```python
def add(a: int, b: int) -> int:
    return a + b
```
Enjoy!
"""
    code = extract_python_code(markdown)
    assert "def add(a: int, b: int) -> int:" in code
    assert "return a + b" in code

    # Fallback to unlabeled code block
    markdown_raw = "```\nx = 42\n```"
    assert extract_python_code(markdown_raw) == "x = 42"


def test_parse_tasks():
    text = """
- [done] Initial plan written
- [todo] Implement DFS algorithm
- [wip] Handle boundary cases
- Simple task without status
"""
    tasks = parse_tasks(text)
    assert len(tasks) == 4
    assert tasks[0]["status"] == "done"
    assert tasks[0]["desc"] == "Initial plan written"
    assert tasks[1]["status"] == "pending"
    assert tasks[1]["desc"] == "Implement DFS algorithm"
    assert tasks[2]["status"] == "in_progress"
    assert tasks[3]["status"] == "pending"


def test_run_tests_subprocess(tmp_path):
    target_file = tmp_path / "calc.py"
    target_file.write_text("def multiply(a, b): return a * b\n", encoding="utf-8")

    solver = GVS5HSolver(
        task="Test multiplication",
        target_file=str(target_file),
        test_cmd=f'{sys.executable} -c "import calc; assert calc.multiply(3, 4) == 12"',
        verbose=False
    )
    solver.ws.mkdir(parents=True, exist_ok=True)
    try:
        # Write valid code into sandbox
        solver._write("solution.py", "def multiply(a, b): return a * b\n")
        res = solver._run_tests()
        assert res["ran"] is True
        assert res["success"] is True
        assert res["passed"] >= 1

        # Write failing code into sandbox
        solver._write("solution.py", "def multiply(a, b): return a + b\n")
        res_fail = solver._run_tests()
        assert res_fail["ran"] is True
        assert res_fail["success"] is False
    finally:
        shutil.rmtree(solver.ws, ignore_errors=True)


def test_best_so_far_regression_rollback(tmp_path):
    target_file = tmp_path / "algo.py"
    target_file.write_text("# Initial target\n", encoding="utf-8")
    solver = GVS5HSolver(
        task="Test rollback",
        target_file=str(target_file),
        max_iters=2,
        verbose=False
    )
    good_code = "def solve(): return 100\n"
    broken_code = "def solve(): return 0\n"

    # Mock planning and management to run 2 iterations
    solver._primary_plan = MagicMock(return_value=[{"id": 1, "desc": "Step 1", "status": "pending"}])
    solver._ideation_worker = MagicMock(return_value="Ideation")
    solver._primary_manage = MagicMock(side_effect=[
        ("continue", "Round 1 task", []),
        ("continue", "Round 2 task", []),
        ("done", "", [])
    ])

    # Round 1 produces good code, Round 2 produces broken code
    def mock_worker(task_desc, task_id, clean_slate=False):
        if task_id == 1:
            solver._write("solution.py", good_code)
        else:
            solver._write("solution.py", broken_code)
        return "done", [], "Worker done", True

    solver._worker = MagicMock(side_effect=mock_worker)

    # Test runner returns success on Round 1, failure on Round 2, and success when rolled back
    def mock_test_runner():
        cur = solver._read("solution.py")
        if cur == good_code:
            return {"ran": True, "success": True, "passed": 5, "total": 5, "feedback": "All passed"}
        else:
            return {"ran": True, "success": False, "passed": 1, "total": 5, "feedback": "Failed"}

    solver._run_tests = MagicMock(side_effect=mock_test_runner)

    # Execute the actual solve() pipeline
    result = solver.solve()
    assert result["status"] == "success"
    assert result["delivered"] is True
    # Verify rollback restored the good code
    assert target_file.read_text(encoding="utf-8") == good_code


def test_atomic_delivery_and_backup(tmp_path):
    target_file = tmp_path / "module.py"
    target_file.write_text("# Original version\nx = 1\n", encoding="utf-8")

    solver = GVS5HSolver(
        task="Update module",
        target_file=str(target_file),
        max_iters=1,
        verbose=False
    )
    new_code = "# Updated version\nx = 2\n"

    solver._primary_plan = MagicMock(return_value=[{"id": 1, "desc": "Update", "status": "pending"}])
    solver._ideation_worker = MagicMock(return_value="Ideation")
    solver._primary_manage = MagicMock(side_effect=[
        ("continue", "Write update", []),
        ("done", "", [])
    ])

    def mock_worker(task_desc, task_id, clean_slate=False):
        solver._write("solution.py", new_code)
        return "done", [], "Done", True

    solver._worker = MagicMock(side_effect=mock_worker)
    solver._run_tests = MagicMock(return_value={"ran": True, "success": True, "passed": 1, "total": 1, "feedback": "OK"})

    result = solver.solve()
    assert result["delivered"] is True
    assert target_file.read_text(encoding="utf-8").strip() == new_code.strip()

    bak_path = target_file.with_suffix(".py.bak")
    assert bak_path.exists()
    assert bak_path.read_text(encoding="utf-8") == "# Original version\nx = 1\n"


def test_full_solve_mocked_end_to_end(tmp_path):
    target_file = tmp_path / "mock_solver.py"
    recorded_temps = []

    def mock_dispatch(messages, temperature, model, endpoint, meta=None, **kwargs):
        recorded_temps.append(temperature)
        role_marker = messages[0]["content"]

        if "PRIMARY orchestrator" in role_marker and "plan" in role_marker.lower():
            return "### PLAN\nShort strategy.\n### TASKS\n- Task 1\n- Task 2"
        elif "FIRST WORKER" in role_marker:
            return "### NOTES\nBrainstormed algorithms.\n### NEXT\n- Approach A"
        elif "PRIMARY orchestrator" in role_marker and "manage" in role_marker.lower():
            return "### STATUS\ndone\n### NEXT\nnone\n### TASKS\n- [done] Task 1"
        else: # Worker
            return "### CODE\n```python\ndef solution(): return 42\n```\n### NOTES\nAll good.\n### NEXT\nnone\n### STATUS\nsolved"

    with patch("local_ai_launcher.bridges.gvs5h_solver.dispatch_chat_completion", side_effect=mock_dispatch):
        res = run_gvs5h_solve(
            task="Write solution function",
            target_file=str(target_file),
            max_iters=2,
            verbose=False
        )

        assert res["status"] == "success"
        assert res["delivered"] is True
        assert target_file.exists()
        assert "def solution(): return 42" in target_file.read_text(encoding="utf-8")

        # Verify dynamic temperature schedule (Stage 1: 0.3, Stage 2: 0.4, Stage 3: 0.2)
        assert 0.3 in recorded_temps
        assert 0.4 in recorded_temps
        assert 0.2 in recorded_temps


def test_ensure_deep_solve_skills_provisioning(tmp_path):
    pi_dir = tmp_path / ".pi" / "skills" / "deep-solve"
    hermes_dir = tmp_path / ".hermes" / "skills" / "deep-solve"
    agents_dir = tmp_path / ".agents" / "skills" / "deep-solve"

    # 1. Initial provisioning creates directories and files
    targets = [pi_dir, hermes_dir, agents_dir]
    paths = ensure_deep_solve_skills(targets)
    assert len(paths) == 3

    pi_skill = pi_dir / "SKILL.md"
    hermes_skill = hermes_dir / "SKILL.md"
    agents_skill = agents_dir / "SKILL.md"

    assert pi_skill.exists()
    assert hermes_skill.exists()
    assert agents_skill.exists()

    pi_text = pi_skill.read_text(encoding="utf-8")
    hermes_text = hermes_skill.read_text(encoding="utf-8")
    agents_text = agents_skill.read_text(encoding="utf-8")

    assert "From Pi shell / tool execution:" in pi_text
    assert "From Hermes shell / tool execution:" in hermes_text
    assert "From your shell or tool execution:" in agents_text

    # 2. Idempotent check (doesn't fail or corrupt)
    paths_second = ensure_deep_solve_skills(targets)
    assert len(paths_second) == 3
    assert pi_skill.read_text(encoding="utf-8") == pi_text

    # 3. Reinstall / wipe recovery: wipe pi directory completely
    shutil.rmtree(tmp_path / ".pi")
    assert not pi_skill.exists()

    # Provision again - should cleanly restore .pi/skills/deep-solve/SKILL.md
    paths_recovered = ensure_deep_solve_skills(targets)
    assert len(paths_recovered) == 3
    assert pi_skill.exists()
    assert "From Pi shell / tool execution:" in pi_skill.read_text(encoding="utf-8")


def test_ensure_deep_solve_skills_cli(monkeypatch, capsys, tmp_path):
    fake_skills = [tmp_path / "skill1", tmp_path / "skill2", tmp_path / "skill3"]
    monkeypatch.setattr("local_ai_launcher.bridges.gvs5h_solver.ensure_deep_solve_skills", lambda: fake_skills)
    monkeypatch.setattr(sys, "argv", ["gvs5h_solver.py", "--ensure-skills"])
    with pytest.raises(SystemExit) as exc_info:
        gvs5h_main()
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data.get("status") == "provisioned"
    assert len(data.get("skills", [])) >= 3


def test_gvs5h_solver_bounds_hardening(tmp_path):
    """Verify GVS5HSolver and helpers sanitize negative parameters and bounds."""
    target_file = tmp_path / "solution.py"
    target_file.write_text("def solve(): return 42\n", encoding="utf-8")

    # Verify negative bounds clamped to positive thresholds
    solver = GVS5HSolver(
        task="Hardening validation test",
        target_file=str(target_file),
        max_iters=-10,
        max_seconds=-500,
        test_timeout=-2.5
    )
    assert solver.max_iters >= 1
    assert solver.max_seconds >= 1
    assert solver.test_timeout >= 0.1

    # Verify stale sandbox cleanup accepts negative max_age_hours safely
    cleanup_stale_sandboxes(max_age_hours=-1.0)

    # Verify max_log_runs hardening
    solver_log = GVS5HSolver(
        task="Log bounds validation",
        target_file=str(target_file),
        max_log_runs=-20
    )
    assert solver_log.max_log_runs >= 1


def test_prune_old_gvs5h_logs(tmp_path):
    """Verify log folder rotation retains only the latest N directories."""
    logs_dir = tmp_path / "logs" / "gvs5h"
    logs_dir.mkdir(parents=True)

    # Create 15 simulated run directories with chronological timestamp prefixes
    created_dirs = []
    for i in range(15):
        ts = 1700000000 + (i * 100)
        d = logs_dir / f"{ts}_{i:02d}_slug"
        d.mkdir()
        (d / "transcript.jsonl").write_text(f"Run {i}\n", encoding="utf-8")
        created_dirs.append(d)

    assert len(list(logs_dir.iterdir())) == 15

    # Retain only latest 10 runs
    deleted = prune_old_gvs5h_logs(max_runs=10, logs_root=logs_dir)
    assert len(deleted) == 5

    # Ensure the 5 oldest directories (indices 0..4) were deleted
    for old_dir in created_dirs[:5]:
        assert not old_dir.exists()

    # Ensure the 10 newest directories (indices 5..14) are retained
    for kept_dir in created_dirs[5:]:
        assert kept_dir.exists()

    remaining = [p.name for p in logs_dir.iterdir() if p.is_dir()]
    assert len(remaining) == 10

    # Pruning again should do nothing
    deleted_second = prune_old_gvs5h_logs(max_runs=10, logs_root=logs_dir)
    assert len(deleted_second) == 0


def test_prune_old_gvs5h_logs_edge_cases(tmp_path):
    """Verify edge cases: non-existent folder, non-directory files, invalid inputs."""
    # 1. Non-existent folder returns []
    assert prune_old_gvs5h_logs(max_runs=10, logs_root=tmp_path / "nonexistent") == []

    # 2. Empty directory returns []
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    assert prune_old_gvs5h_logs(max_runs=10, logs_root=empty_dir) == []

    # 3. Non-directory files should not be touched
    file_path = empty_dir / "notes.txt"
    file_path.write_text("keep me", encoding="utf-8")
    assert prune_old_gvs5h_logs(max_runs=5, logs_root=empty_dir) == []
    assert file_path.exists()

    # 4. Negative max_runs clamps safely
    dummy = empty_dir / "1700000001_run"
    dummy.mkdir()
    prune_old_gvs5h_logs(max_runs=-5, logs_root=empty_dir)


def test_prune_logs_cli(monkeypatch, capsys, tmp_path):
    """Verify CLI --prune-logs flag executes rotation and outputs JSON."""
    logs_dir = tmp_path / "logs_cli"
    logs_dir.mkdir(parents=True)
    for i in range(5):
        d = logs_dir / f"170000000{i}_run"
        d.mkdir()

    with patch("local_ai_launcher.bridges.gvs5h_solver.LOGS_ROOT", logs_dir):
        monkeypatch.setattr(sys, "argv", ["gvs5h_solver.py", "--prune-logs", "--max-log-runs", "3"])
        with pytest.raises(SystemExit) as exc_info:
            gvs5h_main()
        assert exc_info.value.code == 0

    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data.get("status") == "pruned"
    assert data.get("pruned_count") == 2
    assert len(list(logs_dir.iterdir())) == 3


