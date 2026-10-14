"""
Unit tests for universal Windows Job Object safe runner (local_ai_launcher.bridges.safe_run).
"""

import os
import sys
import time
import subprocess
from pathlib import Path
import pytest
from local_ai_launcher.bridges.safe_run import run_contained, _setup_job_object


def test_job_object_setup():
    """Verify Job Object creates properly on Windows."""
    if sys.platform != "win32":
        pytest.skip("Windows-only Job Object test")

    h_job = _setup_job_object()
    assert h_job is not None
    import ctypes
    kernel32 = ctypes.windll.kernel32
    kernel32.CloseHandle(h_job)


def test_safe_run_success():
    """Verify normal command execution and return code pass-through."""
    ret = run_contained([sys.executable, "-c", "import sys; sys.exit(42)"])
    assert ret == 42


def test_safe_run_timeout():
    """Verify timeout terminates execution with exit code 124 within deadline."""
    t0 = time.time()
    ret = run_contained(
        [sys.executable, "-c", "import time; time.sleep(10)"],
        timeout_s=1.0,
    )
    elapsed = time.time() - t0
    assert ret == 124
    assert elapsed < 3.0, f"Timeout took too long to trigger: {elapsed}s"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows Job Object test")
def test_safe_run_reaps_detached_grandchild(tmp_path):
    """Verify detached grandchild processes are terminated upon timeout."""
    test_script = tmp_path / "detached_sub.py"
    test_script.write_text(
        "import subprocess, time, sys\n"
        "proc = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], creationflags=subprocess.DETACHED_PROCESS)\n"
        "print('PID:' + str(proc.pid), flush=True)\n"
        "time.sleep(60)\n"
    )

    workspace_root = str(Path(__file__).resolve().parent.parent.parent)
    env = os.environ.copy()
    env["PYTHONPATH"] = workspace_root + os.pathsep + env.get("PYTHONPATH", "")

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "local_ai_launcher.bridges.safe_run",
            "--timeout",
            "2.0",
            sys.executable,
            str(test_script),
        ],
        cwd=workspace_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    stdout, stderr = proc.communicate()
    assert proc.returncode == 124, f"safe_run returned code {proc.returncode} instead of 124.\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"

    pid = None
    for line in stdout.splitlines():
        if line.startswith("PID:"):
            pid = line.split(":")[1].strip()
            break

    assert pid is not None, "Failed to capture detached grandchild PID"

    # Verify detached grandchild PID is dead in Windows tasklist
    check = subprocess.run(
        ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
        capture_output=True,
        text=True,
    )
    assert pid not in check.stdout, f"Detached PID {pid} was not reaped by Job Object!"
