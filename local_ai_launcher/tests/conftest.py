"""
Global Pytest Conftest - Headless & Window Isolation Guard for local_ai_launcher tests.

Guarantees that during test runs:
1. All Windows process creation calls (subprocess.Popen, subprocess.run, etc.) are strictly headless.
2. CREATE_NEW_CONSOLE flags are completely suppressed to eliminate terminal popping and focus stealing.
3. STARTF_USESHOWWINDOW / SW_HIDE and CREATE_NO_WINDOW are enforced across all subprocess executions.
4. HEADLESS, NONINTERACTIVE, and CI environment flags are asserted throughout the test session.
"""

import os
import sys
import subprocess
import pytest

# Ensure headless environment flags are established before any launcher modules evaluate imports
os.environ["HEADLESS"] = "1"
os.environ["NONINTERACTIVE"] = "1"
os.environ["CI"] = "1"

_orig_popen = subprocess.Popen
_orig_run = subprocess.run


def _sanitize_subprocess_kwargs(kwargs):
    """Sanitizes process creation flags to enforce headless execution on Windows."""
    if sys.platform != "win32":
        return kwargs

    cf = kwargs.get("creationflags", 0)
    has_detached = bool(cf & getattr(subprocess, "DETACHED_PROCESS", 0x08))
    new_console_flag = getattr(subprocess, "CREATE_NEW_CONSOLE", 0x10)
    no_window_flag = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)

    # Completely strip CREATE_NEW_CONSOLE
    cf &= ~new_console_flag

    # Add CREATE_NO_WINDOW unless DETACHED_PROCESS is explicitly set
    if not has_detached:
        cf |= no_window_flag

    kwargs["creationflags"] = cf

    startupinfo = kwargs.get("startupinfo")
    if startupinfo is None and not has_detached:
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = 0  # SW_HIDE
        kwargs["startupinfo"] = startupinfo
    elif startupinfo is not None:
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = 0  # SW_HIDE

    return kwargs


class HeadlessPopen(_orig_popen):
    def __init__(self, *args, **kwargs):
        kwargs = _sanitize_subprocess_kwargs(kwargs)
        super().__init__(*args, **kwargs)


def headless_run(*args, **kwargs):
    kwargs = _sanitize_subprocess_kwargs(kwargs)
    if "stdin" not in kwargs:
        kwargs["stdin"] = subprocess.DEVNULL
    if "timeout" not in kwargs:
        kwargs["timeout"] = 25
    return _orig_run(*args, **kwargs)


@pytest.fixture(autouse=True, scope="session")
def enforce_headless_test_environment(tmp_path_factory):
    """Session-level fixture enforcing silent headless execution across all tests and state isolation."""
    test_state_file = tmp_path_factory.mktemp("state") / "update_state.json"
    os.environ["LOCAL_AI_LAUNCHER_UPDATE_STATE_FILE"] = str(test_state_file)
    test_dsh_home = tmp_path_factory.mktemp("dsh_home")
    os.environ["DSH_HOME"] = str(test_dsh_home)

    subprocess.Popen = HeadlessPopen
    subprocess.run = headless_run
    try:
        yield
    finally:
        subprocess.Popen = _orig_popen
        subprocess.run = _orig_run
        os.environ.pop("LOCAL_AI_LAUNCHER_UPDATE_STATE_FILE", None)
        os.environ.pop("DSH_HOME", None)
