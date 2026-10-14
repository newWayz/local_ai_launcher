import os
import sys
import time
import logging
import psutil
from typing import Optional, Dict, Any, List, Set, Union

from local_ai_launcher.config import config

logger = logging.getLogger("memory_clawback")

# Process names of non-essential, RAM-heavy user applications targeted during brutal clawback
BRUTAL_TARGET_PROCESS_NAMES = {
    # Web Browsers
    "chrome.exe", "msedge.exe", "firefox.exe", "brave.exe", "opera.exe", "vivaldi.exe",
    # Communication, Gaming & Streaming
    "discord.exe", "slack.exe", "teams.exe", "steam.exe", "steamwebhelper.exe",
    "epicgameslauncher.exe", "spotify.exe", "vlc.exe", "battlenet.exe", "origin.exe",
}

# Critical Windows system processes and essentials that must NEVER be terminated
PROTECTED_SYSTEM_PROCESS_NAMES = {
    "system", "system idle process", "registry", "smss.exe", "csrss.exe", "wininit.exe",
    "services.exe", "lsass.exe", "svchost.exe", "fontdrvhost.exe", "dwm.exe",
    "memory compression", "spoolsv.exe", "explorer.exe", "sihost.exe", "taskhostw.exe",
    "winlogon.exe", "runtimebroker.exe", "searchhost.exe", "shellexperiencehost.exe",
    "startmenuexperiencehost.exe", "textinputhost.exe", "ctfmon.exe", "conhost.exe",
    "windowsterminal.exe", "cmd.exe", "powershell.exe", "pwsh.exe"
}


def get_current_tree_pids() -> Set[int]:
    """Returns the set of PIDs comprising the current process, its parents, ancestors, and children."""
    pids = {os.getpid()}
    try:
        curr = psutil.Process(os.getpid())
        for parent in curr.parents():
            pids.add(parent.pid)
        for child in curr.children(recursive=True):
            pids.add(child.pid)
    except Exception:
        pass
    return pids


def evict_auxiliary_services() -> Dict[str, bool]:
    """
    Shuts down and terminates all CPU auxiliary microservices (BGE Reranker on 8082,
    ONNX Gater on 8083, Decider-4B on 8084, CPU Director on 8081).
    Reclaims ~4.0 - 5.0 GB of system RAM.
    """
    results = {}
    logger.info("[Memory Clawback] Evicting CPU auxiliary microservices to maximize available system RAM...")

    # 1. Stop CPU Reranker (Port 8082)
    try:
        from local_ai_launcher.backends.cpu_reranker import stop_cpu_reranker_server
        stop_cpu_reranker_server()
        results["cpu_reranker"] = True
    except Exception as e:
        logger.debug(f"[Memory Clawback] Notice stopping CPU Reranker: {e}")
        results["cpu_reranker"] = False

    # 2. Stop CPU ONNX Decision Gater / von (Port 8083)
    try:
        from local_ai_launcher.backends.cpu_onnx_gater import stop_cpu_onnx_gater_server
        stop_cpu_onnx_gater_server()
        results["cpu_onnx_gater"] = True
    except Exception as e:
        logger.debug(f"[Memory Clawback] Notice stopping CPU ONNX Gater: {e}")
        results["cpu_onnx_gater"] = False

    # 3. Stop CPU Decider-4b (Port 8084)
    try:
        from local_ai_launcher.backends.cpu_decider import stop_cpu_decider_server
        stop_cpu_decider_server()
        results["cpu_decider"] = True
    except Exception as e:
        logger.debug(f"[Memory Clawback] Notice stopping CPU Decider: {e}")
        results["cpu_decider"] = False

    # 4. Force-kill ports associated with auxiliary services
    aux_ports = list(dict.fromkeys([
        getattr(config, "cpu_reranker_port", 8082),
        getattr(config, "von_port", getattr(config, "cpu_onnx_gater_port", 8083)),
        getattr(config, "decider_4b_port", 8084),
        getattr(config, "decider_4b_worker_port", 8081),
        getattr(config, "cpu_llama_server_port", 8081),
    ]))
    # Note: Parakeet NPU (8000) and SearXNG (8088) are essential harness daemons and are preserved.
    for p in aux_ports:
        try:
            from local_ai_launcher.process import force_kill_port
            force_kill_port(p)
        except Exception:
            pass

    return results


def trim_process_working_sets() -> int:
    """
    Calls Windows K32EmptyWorkingSet across all accessible processes.
    Forces the Windows kernel to strip unreferenced private pages and shrink working sets.
    """
    if sys.platform != "win32":
        return 0

    import ctypes
    k32 = ctypes.windll.kernel32
    empty_ws = getattr(k32, "K32EmptyWorkingSet", None)
    if not empty_ws:
        try:
            empty_ws = ctypes.windll.psapi.EmptyWorkingSet
        except Exception:
            return 0

    trimmed = 0
    # PROCESS_QUERY_INFORMATION (0x0400) | PROCESS_SET_QUOTA (0x0100) = 0x0500
    access = 0x0500
    for p in psutil.process_iter(['pid', 'name']):
        pid = p.info['pid']
        if pid in (0, 4):
            continue
        try:
            h = k32.OpenProcess(access, False, pid)
            if h:
                if empty_ws(h):
                    trimmed += 1
                k32.CloseHandle(h)
        except Exception:
            pass

    logger.info(f"[Memory Clawback] Trimmed working sets of {trimmed} processes via K32EmptyWorkingSet.")
    return trimmed


def terminate_non_essential_applications(protected_pids: Optional[Set[int]] = None) -> List[str]:
    """
    Brutally terminates heavy, non-essential user programs (browsers, Discord, Steam, Spotify, etc.)
    Strictly safeguards Windows Explorer, system critical binaries, and active launcher/agent processes.
    """
    if sys.platform != "win32":
        return []

    safe_pids = get_current_tree_pids()
    if protected_pids:
        safe_pids.update(protected_pids)

    killed = []
    for proc in psutil.process_iter(['pid', 'name', 'exe']):
        try:
            pid = proc.info['pid']
            name = (proc.info['name'] or "").lower().strip()
            if pid in safe_pids:
                continue
            if name in PROTECTED_SYSTEM_PROCESS_NAMES or "explorer" in name:
                continue

            if name in BRUTAL_TARGET_PROCESS_NAMES:
                logger.warning(f"[BRUTAL CLAWBACK] Terminating non-essential program: {name} (PID: {pid})")
                proc.kill()
                killed.append(f"{name} (PID: {pid})")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        except Exception as e:
            logger.debug(f"[BRUTAL CLAWBACK] Notice terminating process: {e}")

    return killed


def clawback_system_memory(
    mode: Optional[Union[str, bool]] = None,
    brutal: Optional[Union[str, bool]] = None,
    protected_pids: Optional[Set[int]] = None
) -> Dict[str, Any]:
    """
    Executes comprehensive System RAM and VRAM reclamation across three tiers:
    - 'off'  : Skips RAM reclamation entirely.
    - 'light': Evicts aux microservices and trims process working sets (K32EmptyWorkingSet).
               Does not terminate user applications.
    - 'max'  : Loudly warns, evicts aux microservices, force-terminates non-essential
               heavy user applications (browsers, Discord, Steam, etc.), and trims working sets.
    """
    t0 = time.time()
    ram_before = psutil.virtual_memory().available / (1024 ** 3)

    target_mode = mode if mode is not None else brutal
    if target_mode is None:
        target_mode = getattr(config, "strata_brutal_ram_clawback", "light")

    if isinstance(target_mode, bool):
        resolved_mode = "max" if target_mode else "light"
    else:
        s = str(target_mode).strip().lower()
        if s in ("max", "brutal", "full", "heavy", "true", "1"):
            resolved_mode = "max"
        elif s in ("off", "none", "disabled", "disable", "false", "0"):
            resolved_mode = "off"
        else:
            resolved_mode = "light"

    if resolved_mode == "off":
        logger.info("[Memory Clawback] Mode is 'off'; skipping RAM reclamation.")
        return {
            "ram_before_gb": ram_before,
            "ram_after_gb": ram_before,
            "reclaimed_gb": 0.0,
            "trimmed_count": 0,
            "terminated_apps": [],
            "aux_eviction": {},
            "mode": "off"
        }

    if resolved_mode == "max":
        logger.warning("=" * 78)
        logger.warning(" [BRUTAL MEMORY CLAWBACK: MAX] Aggressive RAM reclamation active!")
        logger.warning(" Terminating non-essential user applications (Browsers, Discord, Steam, etc.)...")
        logger.warning("=" * 78)
    else:
        logger.info("[Memory Clawback: LIGHT] Evicting auxiliary microservices and trimming working sets...")

    # 1. Evict auxiliary microservices (BGE Reranker, ONNX Gater, Decider-4B, Director; Parakeet NPU on 8000 is preserved)
    evict_results = evict_auxiliary_services()

    # 2. In max mode: close non-essential non-explorer applications
    killed_apps = []
    if resolved_mode == "max":
        killed_apps = terminate_non_essential_applications(protected_pids)

    # 3. Trim working sets across all remaining accessible processes
    trimmed_count = trim_process_working_sets()

    time.sleep(0.5)
    ram_after = psutil.virtual_memory().available / (1024 ** 3)
    reclaimed_gb = max(0.0, ram_after - ram_before)

    logger.info(
        f"[Memory Clawback ({resolved_mode.upper()}) Completed in {time.time() - t0:.2f}s] "
        f"Available RAM: {ram_before:.2f} GB -> {ram_after:.2f} GB (+{reclaimed_gb:.2f} GB reclaimed). "
        f"Trimmed: {trimmed_count} processes. Terminated: {len(killed_apps)} applications."
    )

    return {
        "ram_before_gb": ram_before,
        "ram_after_gb": ram_after,
        "reclaimed_gb": reclaimed_gb,
        "trimmed_count": trimmed_count,
        "terminated_apps": killed_apps,
        "aux_eviction": evict_results,
        "mode": resolved_mode
    }


def is_strata_active() -> bool:
    """
    Checks whether Strata engine is currently active or running.
    When Strata is active, all CPU auxiliary models (Reranker, von Gater, Decider-4B)
    are strictly bypassed and kept offline to preserve host system RAM.
    """
    if not getattr(config, 'strata_bypass_aux_models', True):
        return False
    # 1. Check primary configured engine
    if getattr(config, 'engine', '') == 'strata' or getattr(config, 'blackwell_engine', '') == 'strata':
        return True
    # 2. Check active supervisor backend
    try:
        from local_ai_launcher.localai.native_server import supervisor
        if getattr(supervisor, 'active_backend_name', None) == 'strata':
            return True
    except Exception:
        pass
    # 3. Fast probe for open Strata ports (shim port 9943 or engine port 9934)
    try:
        import socket
        for p in (getattr(config, "strata_shim_port", 9943), getattr(config, "strata_engine_port", 9934)):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.05)
                if s.connect_ex(('127.0.0.1', p)) == 0:
                    return True
    except Exception:
        pass
    # 4. Check OS process table for Strata binary
    try:
        for p in psutil.process_iter(['name']):
            p_name = (p.info.get('name') or '').lower()
            if 'strata' in p_name:
                return True
    except Exception:
        pass
    return False
