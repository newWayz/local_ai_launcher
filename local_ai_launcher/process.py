import subprocess
import time
import sys
import os
import shutil
from pathlib import Path
import zipfile
import io
import re
import json
import threading
import urllib.request
import urllib.error
import webbrowser
import logging
import sqlite3
import atexit
import signal
try:
    import psutil
except ImportError:
    psutil = None
import configparser
import socket
import argparse
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Dict
from abc import ABC, abstractmethod

from local_ai_launcher.config import *

def build_ps_env(env_vars: dict, workdir: Optional[Path] = None, harness_name: Optional[str] = None) -> str:
    """Build a PowerShell environment variable setup string with initial working directory."""
    target_dir = workdir if workdir is not None else config.harness_start_dir
    parts = [f"Set-Location '{target_dir}';"]
    
    # Contain AI CLI tools to the workspace directory via HOME (does not override Windows USERPROFILE)
    if 'HOME' not in env_vars:
        env_vars['HOME'] = str(AI_WORKSPACE_DIR)

    # Determine if this harness uses DonSeTch instead of SearXNG
    norm_h = str(harness_name).strip().lower() if harness_name else ""
    is_donsetch_harness = (
        (norm_h == "pi" and getattr(config, "pi_donsetch_enabled", True))
        or (norm_h in ("deepseek", "dsh") and getattr(config, "deepseek_donsetch_enabled", True))
        or bool(env_vars.get("DONSETCH_BINARY") or env_vars.get("DONSETCH_BIN") or env_vars.get("DONSETCH_DSH_PREFIX"))
    )

    # Inject SearXNG metasearch endpoint for harnesses using SearXNG
    if not is_donsetch_harness and getattr(config, 'searxng_enabled', True) and str(getattr(config, 'searxng_mode', 'wsl_docker')).lower() not in ('disabled', 'off', 'none'):
        searxng_url = getattr(config, 'searxng_effective_url', 'http://127.0.0.1:8088')
        if 'SEARXNG_URL' not in env_vars:
            env_vars['SEARXNG_URL'] = searxng_url
        if 'SEARXNG_BASE_URL' not in env_vars:
            env_vars['SEARXNG_BASE_URL'] = searxng_url
        
    for k, v in env_vars.items():
        escaped_v = str(v).replace("'", "''")
        parts.append(f"$env:{k}='{escaped_v}';")
    return " ".join(parts) + " "


def run_silent_subprocess(cmd: List[str] | str, **kwargs) -> subprocess.CompletedProcess:
    """Run a subprocess silently on Windows without allocating or flashing console windows."""
    if sys.platform == 'win32':
        kwargs.setdefault('creationflags', subprocess.CREATE_NO_WINDOW)
        startupinfo = kwargs.get('startupinfo')
        if startupinfo is None:
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = 0  # SW_HIDE
            kwargs['startupinfo'] = startupinfo
    if 'stdout' not in kwargs and 'stderr' not in kwargs and not kwargs.get('capture_output'):
        kwargs.setdefault('capture_output', True)
        kwargs.setdefault('text', True)
    return subprocess.run(cmd, **kwargs)


_run_silent = run_silent_subprocess



def force_kill_port(port, allowed_names=None):
    """Forcefully kill any process listening on the specified port (Cross-platform, safe process name filter)."""
    if allowed_names is None:
        allowed_names = {
            "llama-server", "llama", "omniroute", "node", "python", "uvicorn",
            "whisper-npu", "signal-cli", "ft", "tabbyapi", "strata", "strata-vision", "powershell", "cmd", "pwsh", "uv",
            "t3", "opencode", "litellm"
        }
    try:
        if psutil is not None:
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    for conn in proc.net_connections(kind='inet'):
                        if conn.laddr.port == port:
                            if proc.info.get('pid') == os.getpid():
                                continue
                            proc_name = proc.info.get('name', '').lower()
                            if allowed_names and not any(target in proc_name for target in allowed_names):
                                logger.warning(f"Skipping kill on port {port}: Process '{proc.info.get('name')}' (PID: {proc.info['pid']}) is not a target service binary.")
                                continue
                            logger.info(f"Killing process {proc.info['name']} (PID: {proc.info['pid']}) holding port {port}")
                            kill_process_tree(proc.info['pid'])
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
    except Exception as e:
        logger.debug(f"Notice: psutil scan on port {port}: {e}")

    # On Windows, always verify via netstat to catch any zombie or hidden port bindings
    if sys.platform == 'win32':
        try:
            port_num = int(port)
            cmd = f"netstat -ano | findstr :{port_num}"
            out = subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW).decode('utf-8', errors='ignore')
            for line in out.splitlines():
                line = line.strip()
                if "LISTENING" in line:
                    parts = line.split()
                    if parts and len(parts) >= 2:
                        local_addr = parts[1]
                        if not local_addr.endswith(f":{port_num}"):
                            continue
                        p_pid = parts[-1]
                        if p_pid.isdigit() and int(p_pid) > 0:
                            target_pid = int(p_pid)
                            if target_pid == os.getpid():
                                continue
                            if psutil is not None:
                                try:
                                    p_obj = psutil.Process(target_pid)
                                    p_name = p_obj.name().lower()
                                    if allowed_names and not any(target in p_name for target in allowed_names):
                                        logger.warning(f"Skipping netstat kill on port {port_num}: Process '{p_name}' (PID: {target_pid}) is not an allowed service binary.")
                                        continue
                                except (psutil.NoSuchProcess, psutil.AccessDenied):
                                    pass
                            kill_process_tree(target_pid)
        except Exception:
            pass


def is_port_in_use(port, host="127.0.0.1"):
    """Check if a TCP port is currently open/bound on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def wait_for_port(port: int, host: str = "127.0.0.1", timeout: float = 6.0) -> bool:
    """Wait until a local port opens, up to the given timeout in seconds."""
    start = time.time()
    while time.time() - start < timeout:
        if is_port_in_use(port, host=host):
            return True
        time.sleep(0.3)
    return False


def wait_for_http_service(url, name, timeout=30):
    """Poll an HTTP endpoint until it responds with success or times out."""
    logger.info(f"Waiting for {name} to respond at {url} (up to {timeout}s)...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'HealthCheck'})
            with urllib.request.urlopen(req, timeout=2) as response:
                if response.status in (200, 204, 301, 302, 404):  # 404 = server up, endpoint not registered yet
                    logger.info(f"  -> {name} is ready! ({time.time() - start_time:.1f}s)")
                    return True
        except urllib.error.HTTPError as e:
            if e.code in (200, 204, 301, 302, 404):
                logger.info(f"  -> {name} is ready! ({time.time() - start_time:.1f}s)")
                return True
        except Exception:
            pass
        time.sleep(1)
    logger.warning(f"  -> Warning: {name} did not respond within {timeout} seconds. Proceeding anyway...")
    return False


def check_llama_server_async(url, timeout=120):
    """Check llama-server HTTP endpoint in a background thread and print signal when found without blocking script startup."""
    def _check():
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'HealthCheck'})
                with urllib.request.urlopen(req, timeout=2) as response:
                    if response.status in (200, 204, 301, 302):
                        logger.info(f"\n  -> LLM server http signal found! ({time.time() - start_time:.1f}s)")
                        return
            except Exception:
                pass
            time.sleep(1)
    t = threading.Thread(target=_check, daemon=True)
    t.start()
    return t


def preload_default_model_async(port=LLM_BACKEND_PORT, model_name=config.default_model, timeout=180, harness_name=None):
    """
    Check llama-server health and immediately preload the default model into GPU/memory
    with the target harness context in the background so it is ready before the user inputs their first prompt.
    """

    def _preload():
        health_urls = [
            f"http://127.0.0.1:{port}/health",
            f"http://127.0.0.1:{port}/healthz"
        ]
        chat_url = f"http://127.0.0.1:{port}/v1/chat/completions"
        start_time = time.time()
        server_ready = False
        
        # Step 1: Wait for LLM server HTTP health endpoint to be responsive
        while time.time() - start_time < 30:
            for h_url in health_urls:
                try:
                    req = urllib.request.Request(h_url, headers={'User-Agent': 'HealthCheck'})
                    with urllib.request.urlopen(req, timeout=2) as response:
                        if response.status in (200, 204, 301, 302):
                            server_ready = True
                            break
                except Exception:
                    pass
            if server_ready:
                break
            time.sleep(0.5)

        if not server_ready:
            logger.warning("  -> [Preload] LLM server did not respond to health check in time; skipping model preloading.")
            return

        logger.info(f"\n  -> LLM server is online ({time.time() - start_time:.1f}s). Preloading starting model '{model_name}'...")
        
        # Resolve target harness for preloading canonical agent context
        target_h = harness_name
        if not target_h:
            chosen = getattr(config, "choose_harness", None)
            try:
                from local_ai_launcher.harness_model import resolve_server_preloaded_harness
                target_h = resolve_server_preloaded_harness(chosen, model_name=model_name, cfg=config)
            except Exception as e:
                logger.debug(f"[Preload] Notice resolving server preloaded harness: {e}")
                if isinstance(chosen, (list, tuple)) and chosen:
                    valid = [h for h in chosen if h and str(h).lower() not in ("none", "off", "disabled", "false", "no")]
                    target_h = valid[0] if valid else "pi"
                elif isinstance(chosen, str) and chosen.lower() not in ("none", "off", "disabled", "false", "no"):
                    target_h = chosen
                else:
                    target_h = "pi"

        if getattr(config, "enable_context_preloading", True):
            try:
                from local_ai_launcher.context_preloader import build_harness_prefix
                prefix = build_harness_prefix(target_h, model_name)
                messages = list(prefix.get("messages", []))
                messages.append({"role": "user", "content": "ping"})
                payload_dict = {
                    "model": model_name,
                    "messages": messages,
                    "tools": prefix.get("tools", []),
                    "max_tokens": 1,
                    "temperature": 0.0
                }
            except Exception as e:
                logger.debug(f"[Preload] Prefix build notice: {e}")
                payload_dict = {
                    "model": model_name,
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 1,
                    "temperature": 0.0
                }
        else:
            payload_dict = {
                "model": model_name,
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 1,
                "temperature": 0.0
            }

        load_start = time.time()
        payload = json.dumps(payload_dict).encode("utf-8")

        while time.time() - load_start < timeout:
            try:
                req = urllib.request.Request(
                    chat_url,
                    data=payload,
                    headers={"Content-Type": "application/json", "User-Agent": f"HarnessContextPreloader/{target_h}", "Authorization": "Bearer dummy"},
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=timeout) as response:
                    if response.status == 200:
                        load_duration = time.time() - load_start
                        try:
                            from local_ai_launcher.context_preloader import mark_context_preloaded
                            mark_context_preloaded(f"http://127.0.0.1:{port}/v1", model_name, target_h)
                        except Exception:
                            pass
                        logger.info(f"  -> Model '{model_name}' successfully preloaded with '{target_h}' harness context into KV cache! ({load_duration:.1f}s)")
                        return
                    else:
                        logger.warning(f"  -> [Preload] Unexpected status code {response.status} when preloading '{model_name}'")
                        return
            except urllib.error.HTTPError as e:
                if e.code == 400 and payload_dict.get("tools"):
                    logger.debug(f"[Preload] Backend returned 400 with tools for '{model_name}', falling back to messages-only...")
                    payload_dict.pop("tools", None)
                    payload = json.dumps(payload_dict).encode("utf-8")
                    continue
                if e.code == 503:
                    # Model is still loading (e.g., FreeToken returns this), wait and retry
                    time.sleep(2)
                    continue
                try:
                    err_body = e.read().decode('utf-8', errors='replace')
                except Exception:
                    err_body = str(e)
                logger.warning(f"  -> [Preload] HTTP error {e.code} while preloading '{model_name}': {err_body}")
                return
            except Exception as e:
                logger.warning(f"  -> [Preload] Failed to preload model '{model_name}': {e}")
                return
                
        logger.warning(f"  -> [Preload] Timed out waiting for model '{model_name}' to load after {timeout}s.")

    def _safe_preload():
        try:
            _preload()
        except Exception as e:
            logger.warning(f"  -> [Preload] Daemon thread interrupted unexpectedly: {e}")

    t = threading.Thread(target=_safe_preload, daemon=True)
    t.start()
    return t


_open_log_handles = []


def _close_log_handles():
    for fh in _open_log_handles:
        try:
            fh.close()
        except Exception:
            pass

atexit.register(_close_log_handles)


_global_job_object = None
_tracked_spawned_processes = []


def get_or_create_job_object():
    """Creates or returns a Win32 Job Object configured with JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE."""
    global _global_job_object
    if sys.platform != 'win32':
        return None
    if _global_job_object is not None:
        return _global_job_object
    try:
        import ctypes
        from ctypes import wintypes
        kernel32 = ctypes.windll.kernel32
        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            return None

        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ('PerProcessUserTimeLimit', wintypes.LARGE_INTEGER),
                ('PerJobUserTimeLimit', wintypes.LARGE_INTEGER),
                ('LimitFlags', wintypes.DWORD),
                ('MinimumWorkingSetSize', ctypes.c_size_t),
                ('MaximumWorkingSetSize', ctypes.c_size_t),
                ('ActiveProcessLimit', wintypes.DWORD),
                ('Affinity', ctypes.c_size_t),
                ('PriorityClass', wintypes.DWORD),
                ('SchedulingClass', wintypes.DWORD),
            ]

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [
                ('ReadOperationCount', ctypes.c_uint64),
                ('WriteOperationCount', ctypes.c_uint64),
                ('OtherOperationCount', ctypes.c_uint64),
                ('ReadTransferCount', ctypes.c_uint64),
                ('WriteTransferCount', ctypes.c_uint64),
                ('OtherTransferCount', ctypes.c_uint64),
            ]

        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ('BasicLimitInformation', JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ('IoCounters', IO_COUNTERS),
                ('ProcessMemoryLimit', ctypes.c_size_t),
                ('JobMemoryLimit', ctypes.c_size_t),
                ('PeakProcessMemoryLimit', ctypes.c_size_t),
                ('PeakJobMemoryLimit', ctypes.c_size_t),
            ]

        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE

        JobObjectExtendedLimitInformation = 9
        res = kernel32.SetInformationJobObject(
            job,
            JobObjectExtendedLimitInformation,
            ctypes.byref(info),
            ctypes.sizeof(info)
        )
        if res:
            _global_job_object = job
    except Exception as e:
        logger.debug(f"Notice: Job Object setup: {e}")
    return _global_job_object


def assign_process_to_job(proc):
    """Assigns a subprocess to the Win32 Job Object and registers it in the tracked process list."""
    if not proc:
        return
    _tracked_spawned_processes.append(proc)
    if sys.platform != 'win32':
        return
    job = get_or_create_job_object()
    if not job:
        return
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        pid = getattr(proc, 'pid', proc) if not isinstance(proc, int) else proc
        if not isinstance(pid, int) or pid <= 0:
            return
        handle = kernel32.OpenProcess(0x1F0FFF, False, pid)
        if handle:
            kernel32.AssignProcessToJobObject(job, handle)
            kernel32.CloseHandle(handle)
    except Exception:
        pass


def kill_process_tree(proc):
    """Recursively and forcefully kills a process and all of its descendant child processes."""
    if not proc:
        return
    pid = getattr(proc, 'pid', proc) if not isinstance(proc, int) else proc
    try:
        if psutil is not None:
            try:
                parent = psutil.Process(pid)
                for child in parent.children(recursive=True):
                    try:
                        child.kill()
                    except Exception:
                        pass
                parent.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        if sys.platform == 'win32':
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW,
                timeout=5
            )
        else:
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL if hasattr(signal, 'SIGKILL') else signal.SIGTERM)
            except Exception:
                pass
    except Exception:
        pass


def purge_stale_orca_locks(data_dir: Optional[Path] = None) -> List[Path]:
    """Purge stale PID lockfiles and token artifacts left by ungracefully terminated Orca daemons."""
    target_data_dir = data_dir or getattr(config, 'orca_data_dir', ORCA_DATA_DIR)
    if not target_data_dir or not target_data_dir.exists():
        return []

    purged = []

    def _is_pid_alive(pid: int) -> bool:
        if not pid or pid <= 0:
            return False
        if psutil is not None:
            try:
                return psutil.pid_exists(pid)
            except Exception:
                pass
        if sys.platform == 'win32':
            try:
                res = subprocess.run(
                    ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                    capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW
                )
                return str(pid) in res.stdout
            except Exception:
                pass
        return False

    # 1. Inspect daemon-*.pid files in data/daemon/
    daemon_dir = target_data_dir / "daemon"
    if daemon_dir.exists():
        for pid_file in daemon_dir.glob("daemon-*.pid"):
            try:
                content = json.loads(pid_file.read_text(encoding="utf-8"))
                rec_pid = content.get("pid")
                if isinstance(rec_pid, int) and not _is_pid_alive(rec_pid):
                    token_file = daemon_dir / pid_file.name.replace(".pid", ".token")
                    try:
                        pid_file.unlink(missing_ok=True)
                        purged.append(pid_file)
                    except Exception:
                        pass
                    if token_file.exists():
                        try:
                            token_file.unlink(missing_ok=True)
                            purged.append(token_file)
                        except Exception:
                            pass
            except Exception:
                pass

    # 2. Inspect orca-runtime.json
    runtime_file = target_data_dir / "orca-runtime.json"
    if runtime_file.exists():
        try:
            content = json.loads(runtime_file.read_text(encoding="utf-8"))
            rec_pid = content.get("pid")
            if isinstance(rec_pid, int) and not _is_pid_alive(rec_pid):
                runtime_file.unlink(missing_ok=True)
                purged.append(runtime_file)
        except Exception:
            pass

    if purged:
        logger.info(f"Purged {len(purged)} stale Orca daemon lockfile(s) from {target_data_dir}")
    return purged


def kill_workspace_orca_processes():
    """Terminate lingering workspace-local Orca and daemon-host processes without affecting external editors."""
    orca_dir_str = str((AI_WORKSPACE_DIR / "orca").resolve()).lower()
    daemon_host_str = "daemon-host"

    if psutil is not None:
        for p in psutil.process_iter(['pid', 'name', 'exe', 'cmdline']):
            try:
                name = (p.info.get('name') or '').lower()
                if name != 'orca.exe':
                    continue
                exe = (p.info.get('exe') or '').lower()
                cmdline = " ".join(p.info.get('cmdline') or []).lower()
                if orca_dir_str in exe or orca_dir_str in cmdline or daemon_host_str in exe or daemon_host_str in cmdline:
                    kill_process_tree(p)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    elif sys.platform == 'win32':
        try:
            cmd = "Get-CimInstance Win32_Process -Filter \"Name = 'Orca.exe'\" | Select-Object ProcessId, ExecutablePath, CommandLine | ConvertTo-Json -Compress"
            res = subprocess.run(["powershell.exe", "-NoProfile", "-Command", cmd], capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
            if res.returncode == 0 and res.stdout.strip():
                data = json.loads(res.stdout)
                items = data if isinstance(data, list) else [data]
                for item in items:
                    pid = item.get("ProcessId")
                    exe = (item.get("ExecutablePath") or "").lower()
                    cmdline = (item.get("CommandLine") or "").lower()
                    if pid and (orca_dir_str in exe or orca_dir_str in cmdline or daemon_host_str in exe or daemon_host_str in cmdline):
                        kill_process_tree(pid)
        except Exception:
            pass


def kill_all_launcher_processes():
    """
    Tears down all processes, child processes, ports, and residual binaries started by the launcher.
    Ensures zero residual processes remain when the main console is closed.
    """
    # 1. Kill tracked processes
    for proc in list(_tracked_spawned_processes):
        try:
            kill_process_tree(proc)
        except Exception:
            pass
    _tracked_spawned_processes.clear()

    # 2. Force kill all launcher & LLM engine ports
    ports = [
        config.localai_port,         # 8080 (LocalAI Unified)
        9931,                        # llama-server.exe direct port
        config.blackwell_shim_port,  # 9940 (Blackwell shim)
        config.tabbyapi_shim_port,   # 9941 (TabbyAPI shim)
        config.freetoken_shim_port,  # 9942 (FreeToken shim)
        9932,                        # MCP server port
        OMNIROUTE_PORT,              # 20128 (OmniRoute compression proxy)
        8000,                        # TabbyAPI direct port
        9943,                        # FreeToken direct port
        getattr(config, 'cpu_llama_server_port', 8081),# 8081 (CPU llama-server / Media Director)
        getattr(config, 'cpu_reranker_port', 8082),    # 8082 (CPU Reranker microservice)
        getattr(config, 'cpu_onnx_gater_port', 8083),   # 8083 (CPU ONNX Decision Gater)
        getattr(config, 'signal_daemon_port', 8085),   # 8085 (Hermes Signal CLI Daemon)
        getattr(config, 't3_port', 3773),              # 3773 (T3 Code harness server)
        getattr(config, 'litellm_port', 4000),         # 4000 (LiteLLM proxy)
        getattr(config, 'searxng_port', 8088),         # 8088 (SearXNG Metasearch service)
    ]
    for p in ports:
        try:
            force_kill_port(p)
        except Exception:
            pass

    # 3. Explicit fallback on Windows: kill any orphan engine processes by image name
    if sys.platform == 'win32':
        for exe in (
            "llama-server.exe", "whisper-npu.exe", "omniroute.cmd", "omniroute.exe",
            "t3-resource-monitor.exe", "T3 Code (Alpha).exe", "T3 Code.exe"
        ):
            try:
                subprocess.run(
                    ["taskkill", "/F", "/T", "/IM", exe],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                    timeout=5
                )
            except Exception:
                pass
        kill_workspace_orca_processes()
        purge_stale_orca_locks()

    _close_log_handles()


def _launcher_atexit():
    if os.environ.get("AI_LAUNCHER_MAIN") == "1":
        kill_all_launcher_processes()
    else:
        for proc in list(_tracked_spawned_processes):
            try:
                kill_process_tree(proc)
            except Exception:
                pass
        _tracked_spawned_processes.clear()
        _close_log_handles()


atexit.register(_launcher_atexit)


def start_process(name, command, cwd=None, show_window=False, use_shell=True, env=None):
    """
    Start a background process and redirect its output to a log file.
    Returns the Popen object.
    """
    logger.info(f"Starting {name}...")
    try:
        # Create a log file for this process
        log_dir = AI_WORKSPACE_DIR / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        safe_name = "".join([c if c.isalnum() else "_" for c in name])
        log_file_path = log_dir / f"{safe_name}.log"
        log_file = open(log_file_path, "a")
        _open_log_handles.append(log_file)

        # Note: shell=True is required here for Windows .bat/.cmd execution and complex command strings.
        # All paths are from controlled workspace locations, mitigating command injection risk.
        if sys.platform == 'win32':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            
            # Only show window if explicitly requested (e.g., for llama-server or main script)
            # Note: CREATE_NO_WINDOW and DETACHED_PROCESS are mutually incompatible flags
            is_headless = any(os.environ.get(v) for v in ("CI", "HEADLESS", "NONINTERACTIVE", "PYTEST_CURRENT_TEST"))
            if show_window and not is_headless and is_interactive_desktop():
                startupinfo.wShowWindow = 1  # SW_SHOW = 1
                creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                startupinfo.wShowWindow = 0  # SW_HIDE = 0
                creation_flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
            
            # Use PowerShell instead of cmd.exe because the root DENY rule breaks cmd.exe completely
            if use_shell and sys.platform == 'win32':
                import base64
                # Convert the command string to UTF-16LE base64 for PowerShell
                encoded_cmd = base64.b64encode(command.encode('utf-16le')).decode('utf-8')
                command = f'powershell.exe -NoProfile -NonInteractive -EncodedCommand {encoded_cmd}'
                use_shell = False
                
            # Redirect to log file
            process = subprocess.Popen(
                command,
                cwd=cwd,
                creationflags=creation_flags,
                startupinfo=startupinfo,
                shell=use_shell,
                stdout=log_file,
                stderr=log_file,
                env=env
            )
        else:
            # For non-Windows platforms
            process = subprocess.Popen(
                command,
                cwd=cwd,
                stdout=log_file,
                stderr=log_file,
                start_new_session=True,
                env=env
            )
        
        assign_process_to_job(process)
        logger.info(f"  -> {name} started successfully (PID: {process.pid}). Logs at {log_file_path}")
        return process
    except Exception as e:
        logger.error(f"  -> Failed to start {name}: {e}")
        return None

import uuid
import base64

def is_interactive_desktop():
    """Checks whether the current process is running in an interactive Windows desktop session on the Default desktop."""
    if sys.platform != 'win32':
        return False
    if any(os.environ.get(v) for v in ("CI", "NONINTERACTIVE", "PYTEST_CURRENT_TEST")):
        return False
    try:
        import ctypes
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        hwinsta = user32.GetProcessWindowStation()
        if not hwinsta:
            return False
        # Verify that we are on the interactive 'Default' desktop, not an isolated sandbox or headless service desktop
        h_desk = user32.GetThreadDesktop(kernel32.GetCurrentThreadId())
        if not h_desk:
            return False
        buff = ctypes.create_unicode_buffer(256)
        user32.GetUserObjectInformationW(h_desk, 2, buff, 256, None)  # UOI_NAME = 2
        desk_name = buff.value.lower()
        if desk_name != "default":
            return False
    except Exception:
        pass
    return True

class WTProcessProxy:
    def __init__(self, pid):
        self.pid = pid
        try:
            self._proc = psutil.Process(pid) if psutil else None
        except Exception:
            self._proc = None

    def poll(self):
        if self._proc is None:
            # PID untracked; do not falsely report process as terminated
            return None
        try:
            if not self._proc.is_running():
                return 1
            return None
        except Exception:
            return 1

    def wait(self, timeout=None):
        if not self._proc:
            return 0
        try:
            return self._proc.wait(timeout=timeout)
        except Exception:
            return 0

    def terminate(self):
        self._shutdown_tree(force=False)

    def kill(self):
        self._shutdown_tree(force=True)

    def _shutdown_tree(self, force=False):
        """Recursively shuts down the process tree with a graceful period to protect databases and state."""
        if not self.pid:
            return
        if self._proc:
            try:
                children = self._proc.children(recursive=True)
                # 1. Send graceful terminate signal to children and parent
                for child in children:
                    try:
                        child.terminate()
                    except Exception:
                        pass
                try:
                    self._proc.terminate()
                except Exception:
                    pass

                if force:
                    for child in children:
                        try:
                            child.kill()
                        except Exception:
                            pass
                    try:
                        self._proc.kill()
                    except Exception:
                        pass
                else:
                    # Give up to 1.5s for clean shutdown (e.g. SQLite database flush)
                    gone, alive = psutil.wait_procs(children + [self._proc], timeout=1.5)
                    for p in alive:
                        try:
                            p.kill()
                        except Exception:
                            pass
                return
            except Exception:
                pass

        # Fallback to taskkill on Windows
        if sys.platform == 'win32':
            cmd = ["taskkill", "/PID", str(self.pid), "/T"]
            if force:
                cmd.append("/F")
            try:
                subprocess.run(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                    timeout=5
                )
            except Exception:
                pass


_wt_window_initialized = False

def start_in_wt_tab(command_str, cwd=None, title=None):
    global _wt_window_initialized
    """
    Launch a command as a tab in a Windows Terminal window tied to this workspace.
    Tracks the process via a temporary PID file so that local_ai_launcher can still monitor its lifetime.
    """
    if not is_interactive_desktop():
        cmd_lower = str(command_str).lower()
        if "powershell" in cmd_lower and "-noexit" in cmd_lower:
            logger.info(f"Non-interactive desktop; skipping interactive shell session for '{title or command_str[:30]}'.")
            return None
        return subprocess.Popen(
            command_str,
            cwd=cwd,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0,
        )

    if not shutil.which("wt.exe"):
        return subprocess.Popen(
            command_str,
            cwd=cwd,
            creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == 'win32' else 0,
        )

    # Auto-infer title if none was explicitly provided
    if not title:
        cmd_lower = str(command_str).lower()
        if "deepseek" in cmd_lower:
            title = "DeepSeek"
        elif "hermes" in cmd_lower:
            title = "Hermes"
        elif "antigravity" in cmd_lower or "agy" in cmd_lower:
            title = "Antigravity"
        elif "opencode" in cmd_lower:
            title = "OpenCode"
        elif "pi " in cmd_lower or "pi_cmd" in cmd_lower:
            title = "Pi"
        elif "pie" in cmd_lower:
            title = "Pie"
        elif "t3" in cmd_lower:
            title = "T3"
        elif "orca" in cmd_lower:
            title = "Orca"
        elif "lumara" in cmd_lower:
            title = "OpenLumara"
        elif "claude" in cmd_lower:
            if "local offline" in cmd_lower or "offline" in cmd_lower:
                title = "Claude Code (LOCAL OFFLINE)"
            elif "official" in cmd_lower:
                title = "Claude Code (OFFICIAL)"
            else:
                title = "Claude Code"

    # Extract command content safely
    if isinstance(command_str, (list, tuple)):
        inner_cmd = subprocess.list2cmdline([str(c) for c in command_str])
    else:
        cmd_raw = str(command_str).strip()
        if " -command " in cmd_raw.lower():
            parts = cmd_raw.split(' -Command "', 1) if ' -Command "' in cmd_raw else cmd_raw.split(' -command "', 1)
            if len(parts) == 2:
                inner_cmd = parts[1]
                if inner_cmd.endswith('"'):
                    inner_cmd = inner_cmd[:-1]
            else:
                inner_cmd = cmd_raw
        else:
            inner_cmd = cmd_raw

    # Resolve target window name
    if "AI_LAUNCHER_WINDOW_NAME" in os.environ:
        wt_window_name = os.environ["AI_LAUNCHER_WINDOW_NAME"]
    elif "WT_SESSION" in os.environ:
        wt_window_name = "0"
    else:
        import hashlib
        workspace_id = hashlib.md5(str(AI_WORKSPACE_DIR).encode('utf-8')).hexdigest()[:8]
        wt_window_name = f"AiLauncher_{workspace_id}"

    pid_file = os.path.join(tempfile.gettempdir(), f"local_ai_launcher_pid_{uuid.uuid4().hex}.txt")

    # Use Base64 UTF-16LE -EncodedCommand to eliminate all quote escaping and semicolon issues
    ps_script = f"$PID | Out-File -FilePath '{pid_file}' -Encoding ascii\n{inner_cmd}\n"
    b64_cmd = base64.b64encode(ps_script.encode('utf-16le')).decode('ascii')
    new_full_cmd = f'powershell.exe -NoExit -EncodedCommand {b64_cmd}'

    target_cwd = str(cwd) if cwd else str(config.harness_start_dir or AI_WORKSPACE_DIR)
    title_opt = f'--title "{title}" ' if title else ''
    wt_cmd = f'wt.exe -w "{wt_window_name}" nt -d "{target_cwd}" {title_opt}{new_full_cmd}'

    try:
        wt_proc = subprocess.Popen(wt_cmd, cwd=target_cwd)
        wt_proc.wait(timeout=5)
        if wt_proc.returncode != 0:
            logger.warning(f"wt.exe returned {wt_proc.returncode}, falling back to new console window.")
            fallback_flags = subprocess.CREATE_NEW_CONSOLE if (sys.platform == 'win32' and is_interactive_desktop()) else (subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0)
            return subprocess.Popen(command_str, cwd=cwd, creationflags=fallback_flags)

        # Smart registration: if already inside WT or unified, no wait needed. Otherwise 2.0s pause on cold start.
        if wt_window_name != "0" and not _wt_window_initialized:
            if "WT_SESSION" in os.environ or os.environ.get("AI_LAUNCHER_UNIFIED") == "1":
                _wt_window_initialized = True
            else:
                time.sleep(2.0)
                _wt_window_initialized = True
        else:
            time.sleep(0.3)

    except Exception as e:
        logger.warning(f"Failed to launch wt.exe ({e}), falling back to new console window.")
        fallback_flags = subprocess.CREATE_NEW_CONSOLE if (sys.platform == 'win32' and is_interactive_desktop()) else (subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0)
        return subprocess.Popen(command_str, cwd=cwd, creationflags=fallback_flags)

    # Fast polling for PID file (up to 8 seconds, 50ms intervals -> returns instantly upon launch)
    for _ in range(160):
        if os.path.exists(pid_file):
            try:
                with open(pid_file, "r") as f:
                    pid_str = f.read().strip()
                if pid_str.isdigit():
                    pid = int(pid_str)
                    try:
                        os.remove(pid_file)
                    except Exception:
                        pass
                    return WTProcessProxy(pid)
            except Exception:
                pass
        time.sleep(0.05)

    logger.warning("Failed to track WT tab PID (timeout). Auto-shutdown for this tab may not work.")
    return WTProcessProxy(None)



