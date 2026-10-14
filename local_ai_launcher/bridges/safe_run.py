"""
safe_run.py: Universal Windows Job Object Process Containment Runner.

Wraps arbitrary command execution inside a dedicated Windows NT Job Object configured
with JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE (0x2000). Guarantees that all child processes,
detached processes, and grandchild browser daemons (e.g. Playwright, Chromium, node.exe)
are deterministically reaped upon process exit, timeout, or cancellation signal.

Zero external dependencies (uses standard library ctypes and subprocess).
Works across all agent harnesses: Pi, Hermes, and standalone CLI scripts.
"""

import os
import sys
import time
import signal
import argparse
import subprocess
from typing import List, Optional

# Windows Win32 constants
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
JobObjectExtendedLimitInformation = 9


def _setup_job_object() -> Optional[int]:
    """Creates and configures a Windows Job Object with KILL_ON_JOB_CLOSE."""
    if sys.platform != "win32":
        return None

    import ctypes
    from ctypes import wintypes

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_uint64),
            ("WriteOperationCount", ctypes.c_uint64),
            ("OtherOperationCount", ctypes.c_uint64),
            ("ReadTransferCount", ctypes.c_uint64),
            ("WriteTransferCount", ctypes.c_uint64),
            ("OtherTransferCount", ctypes.c_uint64),
        ]

    class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER),
            ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryLimit", ctypes.c_size_t),
            ("PeakJobMemoryLimit", ctypes.c_size_t),
        ]

    kernel32 = ctypes.windll.kernel32
    h_job = kernel32.CreateJobObjectW(None, None)
    if not h_job:
        return None

    info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE

    success = kernel32.SetInformationJobObject(
        h_job,
        JobObjectExtendedLimitInformation,
        ctypes.byref(info),
        ctypes.sizeof(info),
    )
    if not success:
        kernel32.CloseHandle(h_job)
        return None

    return h_job


def run_contained(
    cmd: List[str],
    timeout_s: Optional[float] = None,
    cwd: Optional[str] = None,
    env: Optional[dict] = None,
) -> int:
    """Executes a command inside a Windows Job Object with optional timeout."""
    import ctypes
    h_job = _setup_job_object()
    kernel32 = ctypes.windll.kernel32 if sys.platform == "win32" else None

    # Spawn child process
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdin=None,
        stdout=None,
        stderr=None,
    )

    if h_job and kernel32:
        kernel32.AssignProcessToJobObject(h_job, int(proc._handle))

    # Signal handler for clean termination
    def _sig_handler(signum, frame):
        if h_job and kernel32:
            kernel32.TerminateJobObject(h_job, 130)
            kernel32.CloseHandle(h_job)
        try:
            proc.kill()
        except Exception:
            pass
        sys.exit(130)

    orig_sigint = signal.getsignal(signal.SIGINT)
    orig_sigbreak = signal.getsignal(signal.SIGBREAK) if hasattr(signal, "SIGBREAK") else None
    orig_sigterm = signal.getsignal(signal.SIGTERM)

    signal.signal(signal.SIGINT, _sig_handler)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    start_time = time.time()
    try:
        while True:
            ret = proc.poll()
            if ret is not None:
                return ret

            if timeout_s is not None and (time.time() - start_time) > timeout_s:
                sys.stderr.write(
                    f"\n[safe_run] Command timed out after {timeout_s}s. Reaping process tree...\n"
                )
                sys.stderr.flush()
                if h_job and kernel32:
                    kernel32.TerminateJobObject(h_job, 124)  # 124 standard timeout exit code
                try:
                    proc.kill()
                except Exception:
                    pass
                return 124

            time.sleep(0.05)
    finally:
        try:
            signal.signal(signal.SIGINT, orig_sigint)
            if hasattr(signal, "SIGBREAK") and orig_sigbreak is not None:
                signal.signal(signal.SIGBREAK, orig_sigbreak)
            signal.signal(signal.SIGTERM, orig_sigterm)
        except Exception:
            pass
        if h_job and kernel32:
            kernel32.CloseHandle(h_job)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run command inside a Windows Job Object with automatic process tree cleanup."
    )
    parser.add_argument(
        "--timeout",
        "-t",
        type=float,
        default=None,
        help="Optional timeout in seconds before terminating the entire job object.",
    )
    parser.add_argument(
        "--cwd",
        type=str,
        default=None,
        help="Working directory for the target command.",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="The command and arguments to execute.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cmd = args.command
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]

    if not cmd:
        sys.stderr.write("Usage: python safe_run.py [--timeout SECONDS] [--cwd DIR] <command...>\n")
        sys.exit(1)

    exit_code = run_contained(cmd, timeout_s=args.timeout, cwd=args.cwd)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
