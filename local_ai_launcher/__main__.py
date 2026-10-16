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
import concurrent.futures
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Dict

from local_ai_launcher.config import *
from local_ai_launcher.environment import *
from local_ai_launcher.models import *
from local_ai_launcher.process import *

from local_ai_launcher.harnesses import HARNESS_REGISTRY, ONLINE_ONLY_HARNESSES
from local_ai_launcher.mnemosyne import run_mnemosyne_verifier
from local_ai_launcher.backends.omniroute import check_omniroute
from local_ai_launcher.localai import LocalAIManager, generate_all_model_configs


def parse_cli_args():
    parser = argparse.ArgumentParser(description="LocalAI Unified Service Launcher")
    parser.add_argument("--model", type=str, help="Starting model name or alias")
    parser.add_argument("--harness", nargs="+", help="Selected harnesses")
    parser.add_argument("--start-dir", type=str, help="Initial working directory for harnesses")
    parser.add_argument("--port", type=int, help="LocalAI API port (default: 8080)")
    parser.add_argument("--no-omniroute", action="store_true", help="Disable OmniRoute proxy")
    parser.add_argument("--no-verify", action="store_true", help="Disable Mnemosyne verification")
    parser.add_argument("--build-ui", action="store_true", help="Build Hermes Web UI")
    parser.add_argument("--cpu-only", action="store_true", help="Disable CUDA")
    parser.add_argument("--no-preload", action="store_true", help="Disable preloading the default model at startup")
    parser.add_argument("--pi-search-workflow", type=str, choices=["auto-summary", "none", "summary-review"], help="Pi web search workflow mode")
    parser.add_argument("--pi-search-browser", action="store_true", help="Enable browser popup for Pi web search curation")
    parser.add_argument("--download-model", action="store_true", help="Always download missing model files automatically")
    parser.add_argument("--no-download-model", action="store_true", help="Never download missing model files")
    parser.add_argument("--headless", dest="headless", action="store_true", default=None, help="Run LocalAI in headless mode (do not open WebUI in browser)")
    parser.add_argument("--gui", "--no-headless", dest="gui", action="store_true", default=None, help="Run LocalAI with WebUI (open in browser)")
    parser.add_argument("--no-gui", dest="no_gui", action="store_true", help="Alias for --headless")
    parser.add_argument("--install-localai", action="store_true", help="Always clone/install LocalAI git repository")
    parser.add_argument("--no-install-localai", action="store_true", help="Never install LocalAI git repository")
    parser.add_argument("--update-localai", action="store_true", help="Always update LocalAI git repository from upstream")
    parser.add_argument("--no-update-localai", action="store_true", help="Never update LocalAI git repository")
    parser.add_argument("--sync-skills", action="store_true", help="Synchronize workspace skills and gateway junctions, then exit")
    parser.add_argument("--update-skills", action="store_true", help="Pull latest updates for all vendor skill suites, then exit")
    parser.add_argument("--force-update", action="store_true", help="Force updates on all active components regardless of frequency interval")
    parser.add_argument("--update-frequency-hrs", dest="update_frequency_hrs", type=str, help="Global update frequency in hours (e.g. 0 = always, 1 = 1 hour, 24 = 24 hours, 168 = weekly, or 'always', 'daily', 'weekly', 'never')")
    parser.add_argument("--update-git", action="store_true", help="Always check and update Git")
    parser.add_argument("--no-update-git", action="store_true", help="Never update Git")
    parser.add_argument("--claude-mode", choices=["local", "official", "local_claude_native"], help="Claude Code CLI routing mode: 'local' (offline Port 8080) or 'official' (Anthropic cloud)")
    parser.add_argument("--claude-desktop-mode", choices=["mcp", "gateway", "disabled", "off"], help="Claude Desktop mode: 'mcp' (official GUI + local MCP bridge) or 'gateway' (developer mode direct local)")
    parser.add_argument("--local-claude-native-engine", type=str, help="Specific engine to target for local Claude native routing (e.g. 'strata')")
    parser.add_argument("--local-claude-native-engines", nargs="+", help="Engines supporting local Claude native Anthropic messages API")
    parser.add_argument("--aside-browser", dest="aside_browser", action="store_true", default=None, help="Enable Aside browser MCP and CLI integration")
    parser.add_argument("--no-aside-browser", dest="no_aside_browser", action="store_true", help="Disable Aside browser MCP and CLI integration")
    
    args, _ = parser.parse_known_args()
    
    if args.model is not None:
        config.default_model = args.model
    if args.harness is not None:
        config.choose_harness = args.harness
    if getattr(args, "claude_mode", None) is not None:
        config.claude_mode = args.claude_mode
        config.claude_mode_explicit = True
    if getattr(args, "claude_desktop_mode", None) is not None:
        config.claude_desktop_mode = args.claude_desktop_mode
        config.claude_desktop_mode_explicit = True
    if getattr(args, "local_claude_native_engine", None) is not None:
        config.local_claude_native_engine = args.local_claude_native_engine
    if getattr(args, "local_claude_native_engines", None) is not None:
        config.local_claude_native_engines = args.local_claude_native_engines
    if args.port is not None:
        config.localai_port = args.port
    if args.start_dir is not None:
        config.harness_start_dir_rel = args.start_dir
    p = Path(config.harness_start_dir_rel)
    config.harness_start_dir = p.resolve() if p.is_absolute() else (AI_WORKSPACE_DIR / p).resolve()
    config.harness_start_dir.mkdir(parents=True, exist_ok=True)
    if args.no_omniroute:
        config.use_omniroute_proxy = False
    if args.no_verify:
        config.run_mnemosyne_verification = False
    if args.build_ui:
        config.skip_hermes_web_ui_build = False
    if args.cpu_only:
        config.use_cuda = False
    if args.no_preload:
        config.preload_default_model = False
    if args.pi_search_workflow is not None:
        config.pi_web_search_workflow = args.pi_search_workflow
    if args.pi_search_browser:
        config.pi_web_search_auto_open_browser = True
    if args.download_model:
        config.models_install_behavior = 'always_install'
    elif args.no_download_model:
        config.models_install_behavior = 'never_install'
    if args.install_localai:
        config.localai_install_behavior = 'always_install'
    elif args.no_install_localai:
        config.localai_install_behavior = 'never_install'
    if args.update_localai:
        config.localai_update_behavior = 'always_update'
    elif args.no_update_localai:
        config.localai_update_behavior = 'never_update'
    if args.headless is True or args.no_gui:
        config.localai_headless = True
    elif args.gui is True:
        config.localai_headless = False
    if getattr(args, 'force_update', False):
        config.force_update = True
    if getattr(args, 'update_frequency_hrs', None) is not None:
        config.update_frequency_hrs = args.update_frequency_hrs
    if getattr(args, 'update_git', False):
        config.git_update_behavior = 'always_update'
        config.git_auto_update = True
    elif getattr(args, 'no_update_git', False):
        config.git_update_behavior = 'never_update'
        config.git_auto_update = False
    if getattr(args, "aside_browser", None) is True:
        config.aside_browser_enabled = True
    elif getattr(args, "no_aside_browser", False):
        config.aside_browser_enabled = False

    if getattr(args, 'sync_skills', False) or getattr(args, 'update_skills', False):
        from local_ai_launcher.skills_manager import SkillManager
        suites = getattr(config, 'skill_suites', None) or None
        mgr = SkillManager(suites=suites)
        res = mgr.sync(update=getattr(args, 'update_skills', False))
        print("Skill synchronization complete:")
        print(f"  - Builtin skills: {len(res.get('builtins', []))}")
        print(f"  - Vendor suites active: {len(res.get('suites', []))}")
        print(f"  - Gateway router junctions: {len(res.get('routers', []))}")
        print(f"  - Pruned clutter folders: {len(res.get('pruned', []))}")
        print(f"  - Updater script: {res.get('update_script')}")
        sys.exit(0)


def ensure_windows_terminal():
    """If running in an interactive Windows console outside Windows Terminal, relaunch in named WT window."""
    if os.environ.get("PYTEST_CURRENT_TEST") or any(os.environ.get(v) for v in ("CI", "HEADLESS", "NONINTERACTIVE")):
        return
    if "--unified" in sys.argv or os.environ.get("AI_LAUNCHER_UNIFIED") or os.environ.get("WT_SESSION"):
        return
    if sys.platform == 'win32' and is_interactive_desktop() and shutil.which("wt.exe") and sys.stdin and sys.stdin.isatty():
        import hashlib
        workspace_id = hashlib.md5(str(AI_WORKSPACE_DIR).encode('utf-8')).hexdigest()[:8]
        win_name = f"AiLauncher_{workspace_id}"

        os.environ["AI_LAUNCHER_UNIFIED"] = "1"
        os.environ["AI_LAUNCHER_WINDOW_NAME"] = win_name
        args = [a for a in sys.argv[1:] if a != "--unified"]
        escaped_args = subprocess.list2cmdline(args)
        py_exe = sys.executable
        wt_cmd = f'wt.exe -w "{win_name}" nt -d "{AI_WORKSPACE_DIR}" --title "Local AI Cockpit" "{py_exe}" -m local_ai_launcher --unified {escaped_args}'
        try:
            subprocess.Popen(wt_cmd, cwd=str(AI_WORKSPACE_DIR))
            sys.exit(0)
        except Exception:
            pass


def signal_handler(sig, frame):
    logger.info("Interrupt received, shutting down gracefully...")
    raise KeyboardInterrupt


def warmup_voice_to_text():
    import threading, time, urllib.request, struct
    def worker():
        try:
            num_samples = int(0.5 * 16000)
            data_size = num_samples * 2
            file_size = 36 + data_size
            wav_bytes = struct.pack('<4sI4s4sIHHIIHH4sI', 
                b'RIFF', file_size, b'WAVE', 
                b'fmt ', 16, 1, 1, 16000, 32000, 2, 16, 
                b'data', data_size) + (b'\x00' * data_size)
                
            boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
            body = (
                f'--{boundary}\r\n'
                f'Content-Disposition: form-data; name="file"; filename="silence.wav"\r\n'
                f'Content-Type: audio/wav\r\n\r\n'
            ).encode('utf-8') + wav_bytes + (
                f'\r\n--{boundary}\r\n'
                f'Content-Disposition: form-data; name="model"\r\n\r\n'
                f'whisper-1\r\n'
                f'--{boundary}--\r\n'
            ).encode('utf-8')
            
            req = urllib.request.Request(
                'http://127.0.0.1:8000/v1/audio/transcriptions', 
                data=body,
                headers={'Content-Type': f'multipart/form-data; boundary={boundary}'}
            )
            for _ in range(45):
                try:
                    urllib.request.urlopen(req, timeout=10)
                    logger.info("  -> NPU Voice Transcription successfully pre-warmed in background!")
                    break
                except Exception:
                    time.sleep(1)
        except Exception:
            pass
    threading.Thread(target=worker, daemon=True).start()


def start_parakeet(processes):
    if sys.platform == 'win32':
        try:
            subprocess.run(["taskkill", "/F", "/IM", "ovms.exe"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(["taskkill", "/F", "/IM", "whisper-npu.exe"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

    python_exe = AI_WORKSPACE_DIR / ".venv" / "Scripts" / "python.exe"
    if not python_exe.exists():
        python_exe = Path(sys.executable)

    try:
        from local_ai_launcher.backends.parakeet_npu import ensure_parakeet_models, DEFAULT_MODEL_DIR
        ensure_parakeet_models(DEFAULT_MODEL_DIR)
    except Exception as e:
        logger.warning(f"Warning: Parakeet model check encountered error: {e}")

    target_dev = getattr(config, 'parakeet_device', 'NPU')
    hotkey = getattr(config, 'voice_dictation_hotkey', 'right_alt')
    cmd = [
        str(python_exe),
        "-m", "local_ai_launcher.backends.parakeet_npu",
        "--port", "8000",
        "--device", str(target_dev),
        "--hotkey", str(hotkey)
    ]
    parakeet_process = start_process(
        "Parakeet TDT V3 NPU Voice Transcription",
        cmd,
        str(AI_WORKSPACE_DIR),
        show_window=False,
        use_shell=False
    )
    if parakeet_process:
        processes.append(('Parakeet TDT V3/NPU', parakeet_process))
        logger.info(f"Parakeet TDT V3 active: REST API at http://127.0.0.1:8000/v1/audio/transcriptions | Hotkey Dictation: [{hotkey.upper()}] toggle")
        warmup_voice_to_text()


def start_voice_to_text(processes):
    mode = str(getattr(config, 'voice_to_text', 'parakeet_tdt_v3')).lower().strip()
    if mode in ('none', 'false', 'off', 'disabled'):
        logger.info("Notice: Voice-to-text is disabled. Skipping STT setup.")
        return

    if not is_genuine_intel_processor():
        logger.info("Notice: Genuine Intel processor not detected. Skipping Intel NPU Voice-to-Text setup.")
        return

    start_parakeet(processes)


def start_cpu_auxiliary_engines(processes):
    """Initializes and ensures CPU auxiliary models (Reranker & Decision engine) are ready concurrently."""
    def _run_reranker():
        if getattr(config, 'use_cpu_reranker', True):
            try:
                from local_ai_launcher.backends.cpu_reranker import ensure_reranker_model, start_cpu_reranker_server
                model_name = getattr(config, 'cpu_reranker_model', 'BAAI/bge-reranker-v2-m3')
                logger.info(f"Checking CPU Reranker model ({model_name})...")
                ensure_reranker_model(model_name)
                start_cpu_reranker_server(processes)
            except Exception as e:
                logger.debug(f"Notice: CPU Reranker startup: {e}")

    def _run_onnx_gater():
        if getattr(config, 'use_von_gater', True) or getattr(config, 'use_cpu_onnx_gater', True) or getattr(config, 'use_cpu_decision', True):
            try:
                from local_ai_launcher.backends.cpu_onnx_gater import ensure_onnx_gater_model, start_cpu_onnx_gater_server
                model_name = getattr(config, 'von_model', getattr(config, 'cpu_onnx_gater_model', 'wfzyx/von'))
                ensure_onnx_gater_model(model_name)
                start_cpu_onnx_gater_server(processes)
            except Exception as e:
                logger.warning(f"Notice: CPU von Gater server startup: {e}")

    def _run_decider():
        if getattr(config, 'use_decider_4b', True):
            try:
                from local_ai_launcher.backends.cpu_decider import ensure_decider_model, start_cpu_decider_server
                model_name = getattr(config, 'decider_4b_model', 'Mapika/decider-4b-v2.1-GGUF')
                ensure_decider_model(model_name)
                start_cpu_decider_server(processes)
            except Exception as e:
                logger.warning(f"Notice: CPU Decider-4B server startup: {e}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=3, thread_name_prefix="CpuAuxiliary") as executor:
        futures = [
            executor.submit(_run_reranker),
            executor.submit(_run_onnx_gater),
            executor.submit(_run_decider)
        ]
        concurrent.futures.wait(futures)


def start_searxng_service(processes, selected_harnesses=None):
    """Initializes SearXNG metasearch service for coding harnesses."""
    from local_ai_launcher.backends.searxng import should_start_searxng, start_searxng_server
    if not should_start_searxng(selected_harnesses):
        return
    try:
        start_searxng_server(processes)
    except Exception as e:
        logger.debug(f"Notice: SearXNG metasearch service startup: {e}")


def start_localai_engine(processes):
    """Initializes and runs LocalAI as the unified LLM API endpoint."""
    manager = LocalAIManager(port=config.localai_port, host=config.localai_host)
    api_url, proc = manager.start(processes)
    
    preload_thread = None
    if config.preload_default_model:
        preload_thread = preload_default_model_async(
            port=config.localai_port,
            model_name=config.default_model
        )

    # Automatically open LocalAI Web GUI in browser unless running in headless mode
    if not config.localai_headless:
        gui_url = f"http://localhost:{config.localai_port}"
        logger.info(f"LocalAI Web GUI active: awaiting readiness before opening {gui_url}")
        def _open_gui():
            if wait_for_port(config.localai_port, timeout=10.0):
                webbrowser.open(gui_url)
            else:
                logger.warning(f"LocalAI port {config.localai_port} did not open within 10s; attempting browser open anyway.")
                webbrowser.open(gui_url)
        threading.Thread(target=_open_gui, daemon=True).start()
    else:
        logger.info(f"LocalAI running in headless mode (Web GUI available at http://localhost:{config.localai_port})")

    return api_url, preload_thread, manager


def start_proxy(processes, target_llm_url):
    proxy_url = None
    if config.use_omniroute_proxy:
        if is_port_in_use(OMNIROUTE_PORT):
            logger.info(f"  -> OmniRoute proxy is already running on port {OMNIROUTE_PORT}.")
            proxy_url = f"http://localhost:{OMNIROUTE_PORT}/v1"
            webbrowser.open(f"http://localhost:{OMNIROUTE_PORT}")
        else:
            omniroute_installed = check_omniroute()
            if omniroute_installed:
                local_bin = AI_WORKSPACE_DIR / "node_modules" / ".bin"
                omniroute_exe = str(local_bin / "omniroute.cmd") if (local_bin / "omniroute.cmd").exists() else "omniroute"
                omniroute_process = start_process(
                    "OmniRoute (compression proxy)",
                    f'"{omniroute_exe}" --no-open',
                    str(AI_WORKSPACE_DIR),
                    show_window=False
                )
                if omniroute_process:
                    processes.append(('OmniRoute', omniroute_process))
                    if wait_for_port(OMNIROUTE_PORT, timeout=5.0):
                        proxy_url = f"http://localhost:{OMNIROUTE_PORT}/v1"
                        logger.info(f"  -> OmniRoute proxy active. Harness will use: {proxy_url}")
                        webbrowser.open(f"http://localhost:{OMNIROUTE_PORT}")
                    else:
                        logger.warning(f"  -> OmniRoute process started but port {OMNIROUTE_PORT} did not respond within 5s. Harness will connect directly to LocalAI.")
                        proxy_url = None
                else:
                    logger.warning("Warning: OmniRoute failed to start. Harness will connect directly to LocalAI.")
            else:
                logger.warning("Warning: OmniRoute not available. Harness will connect directly to LocalAI.")
    return proxy_url


def prepare_harness_targets(selected_harnesses):
    """Expands selected harnesses with required auxiliary harness dependencies (e.g. for Orca ADE)."""
    orca_selected = "orca" in selected_harnesses
    harnesses_to_run = list(selected_harnesses)
    if orca_selected:
        for aux in ("pi", "hermes", "openlumara"):
            if aux in HARNESS_REGISTRY and aux not in harnesses_to_run:
                orca_idx = harnesses_to_run.index("orca")
                harnesses_to_run.insert(orca_idx, aux)
    return harnesses_to_run


def _check_and_update_harness(name, adapter):
    logger.info(f"[{name.upper()}] Starting pre-flight verification/update...")
    try:
        is_ready = adapter.check()
        logger.info(f"[{name.upper()}] Pre-flight complete (ready={is_ready}).")
        return name, is_ready
    except Exception as e:
        logger.warning(f"[{name.upper()}] Pre-flight verification failed: {e}")
        return name, False


class HarnessPreflightSession:
    """Manages asynchronous harness pre-flight checks and updates across startup tracks."""
    def __init__(self, selected_harnesses, processes):
        self.selected_harnesses = list(selected_harnesses)
        self.harnesses_to_run = prepare_harness_targets(selected_harnesses)
        self.adapters = {}
        for h_name in self.harnesses_to_run:
            adapter_cls = HARNESS_REGISTRY.get(h_name)
            if not adapter_cls:
                logger.warning(f"Warning: Harness '{h_name}' not found in HARNESS_REGISTRY. Skipping...")
                continue
            self.adapters[h_name] = adapter_cls(str(root_dir), processes)

        max_workers = min(len(self.adapters), 4) if self.adapters else 1
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="HarnessPreFlight"
        )
        logger.info("\n--- Phase 1 & 2: Pre-Flight Harness Updates & Verification (Concurrent Track) ---")
        self.futures = {
            self.executor.submit(_check_and_update_harness, name, adapter): name
            for name, adapter in self.adapters.items()
        }
        self.ready_map = {}
        self.lock = threading.Lock()
        self.is_done = False

    def wait_and_collect(self) -> Dict[str, bool]:
        with self.lock:
            if self.is_done:
                return dict(self.ready_map)
            for future in concurrent.futures.as_completed(self.futures):
                try:
                    name, is_ready = future.result()
                    self.ready_map[name] = is_ready
                except Exception as e:
                    name = self.futures.get(future, "unknown")
                    logger.warning(f"[{name.upper()}] Pre-flight verification failed: {e}")
                    self.ready_map[name] = False
            try:
                self.executor.shutdown(wait=False)
            except Exception:
                pass
            self.is_done = True
            logger.info("--- Pre-Flight Harness Updates & Verification Completed ---\n")
            return dict(self.ready_map)

    def cancel(self):
        try:
            self.executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            try:
                self.executor.shutdown(wait=False)
            except Exception:
                pass


_active_harness_preflight: Optional[HarnessPreflightSession] = None


def start_harness_preflight_async(selected_harnesses, processes) -> Optional[HarnessPreflightSession]:
    """Initiates harness pre-flight checks concurrently in the background."""
    global _active_harness_preflight
    if not selected_harnesses:
        return None
    try:
        _active_harness_preflight = HarnessPreflightSession(selected_harnesses, processes)
        return _active_harness_preflight
    except Exception as e:
        logger.warning(f"Notice: Failed to initialize concurrent harness pre-flight: {e}")
        return None


def run_harnesses(processes, selected_harnesses, active_model_url, llm_thread):
    global _active_harness_preflight
    orca_selected = "orca" in selected_harnesses

    # Check if a matching concurrent pre-flight session was already initiated
    session = None
    if _active_harness_preflight is not None:
        if _active_harness_preflight.selected_harnesses == list(selected_harnesses):
            session = _active_harness_preflight
        _active_harness_preflight = None

    if session is not None:
        harnesses_to_run = session.harnesses_to_run
        adapters = session.adapters
        harness_ready_map = session.wait_and_collect()
    else:
        # Fallback / Direct invocation path (e.g. unit tests or isolated runs)
        harnesses_to_run = prepare_harness_targets(selected_harnesses)
        adapters = {}
        for h_name in harnesses_to_run:
            adapter_cls = HARNESS_REGISTRY.get(h_name)
            if not adapter_cls:
                logger.warning(f"Warning: Harness '{h_name}' not found in HARNESS_REGISTRY. Skipping...")
                continue
            adapters[h_name] = adapter_cls(str(root_dir), processes)

        logger.info("\n--- Phase 1 & 2: Pre-Flight Harness Updates & Verification (Simultaneous) ---")
        harness_ready_map = {}
        max_workers = min(len(adapters), 4) if adapters else 1
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="HarnessPreFlight") as executor:
            future_to_name = {
                executor.submit(_check_and_update_harness, name, adapter): name
                for name, adapter in adapters.items()
            }
            for future in concurrent.futures.as_completed(future_to_name):
                name, is_ready = future.result()
                harness_ready_map[name] = is_ready
        logger.info("--- Pre-Flight Harness Updates & Verification Completed ---\n")

    # -------------------------------------------------------------
    # Phase 3 & 4: Staged Configuration, Bridging & Launch
    # -------------------------------------------------------------
    if any(h_name not in ONLINE_ONLY_HARNESSES and harness_ready_map.get(h_name) for h_name in harnesses_to_run):
        if llm_thread is not None and llm_thread.is_alive():
            logger.info("Waiting for model to load before launching harnesses...")
            llm_thread.join(timeout=180)

    for h_name in harnesses_to_run:
        if not harness_ready_map.get(h_name, False):
            logger.warning(f"Warning: {h_name} harness installation was skipped or failed. Continuing without it...")
            continue

        adapter = adapters[h_name]
        is_aux_for_orca = orca_selected and (h_name not in selected_harnesses or h_name in ("openlumara", "lumara"))
        if is_aux_for_orca:
            if h_name == "pi":
                adapter.start(active_model_url, config.default_model, launch_tui=False)
            elif h_name == "hermes":
                adapter.start(active_model_url, config.default_model, launch_tui=False, open_browser=False, auxiliary_only=True)
            elif h_name in ("openlumara", "lumara"):
                adapter.start(active_model_url, config.default_model, launch_tui=False)
            else:
                adapter.start(active_model_url, config.default_model)
        else:
            adapter.start(active_model_url, config.default_model)


def main():
    os.environ["AI_LAUNCHER_MAIN"] = "1"
    ensure_running_in_venv()
    ensure_windows_terminal()
    signal.signal(signal.SIGINT, signal_handler)

    if sys.platform == 'win32':
        setup_windows_environment_variables()
        setup_windows_user_account()
    
    # Ensure workspace skill gateway, builtin skills, and vendor suites are synchronized
    if getattr(config, 'enable_skill_gateway', True):
        try:
            from local_ai_launcher.skills_manager import SkillManager
            auto_update = getattr(config, 'auto_update_skills_on_launch', False)
            suites = getattr(config, 'skill_suites', None) or None
            mgr = SkillManager(suites=suites)
            mgr.sync(update=auto_update)
        except Exception as e:
            logger.warning(f"Skill gateway synchronization notice: {e}")

    try:
        from local_ai_launcher.bridges.gvs5h_solver import prune_old_gvs5h_logs
        prune_old_gvs5h_logs(max_runs=getattr(config, "gvs5h_max_log_runs", 100))
    except Exception as e:
        logger.debug(f"Log pruning notice: {e}")

    processes = []
    cpu_aux_executor = None

    def _cleanup_processes_on_exit():
        global _active_harness_preflight
        if _active_harness_preflight is not None:
            try:
                _active_harness_preflight.cancel()
            except Exception:
                pass
            _active_harness_preflight = None

        if cpu_aux_executor is not None:
            try:
                cpu_aux_executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass

        for item in reversed(list(processes)):
            proc = item[1] if isinstance(item, (tuple, list)) and len(item) > 1 else item
            if proc and hasattr(proc, "poll") and proc.poll() is None:
                try:
                    if hasattr(proc, "stop"):
                        proc.stop()
                    elif hasattr(proc, "terminate"):
                        proc.terminate()
                    else:
                        kill_process_tree(proc)
                except Exception:
                    pass
        
        # Call kill_all_launcher_processes to purge all ports (8080, 9931, 9940, 9941, 9942, 9932, 20128, etc.) and binaries
        try:
            from local_ai_launcher.process import kill_all_launcher_processes
            kill_all_launcher_processes()
        except Exception:
            pass

        try:
            from local_ai_launcher.backends.cpu_decision import stop_cpu_decision_server
            stop_cpu_decision_server()
        except Exception:
            pass

        try:
            from local_ai_launcher.backends.cpu_reranker import stop_cpu_reranker_server
            stop_cpu_reranker_server()
        except Exception:
            pass

        try:
            from local_ai_launcher.backends.cpu_onnx_gater import stop_cpu_onnx_gater_server
            stop_cpu_onnx_gater_server()
        except Exception:
            pass

        try:
            from local_ai_launcher.backends.litellm import stop_litellm_proxy
            stop_litellm_proxy()
        except Exception:
            pass

    atexit.register(_cleanup_processes_on_exit)

    if sys.platform == 'win32':
        try:
            import ctypes
            from ctypes import wintypes
            def console_ctrl_handler(ctrl_type):
                if ctrl_type in (0, 1, 2, 5, 6):
                    logging.raiseExceptions = False
                    logger.handlers = []
                    _cleanup_processes_on_exit()
                    return True
                return False
                
            HandlerRoutine = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)
            _cleanup_processes_on_exit._ctrl_handler = HandlerRoutine(console_ctrl_handler)
            ctypes.windll.kernel32.SetConsoleCtrlHandler(_cleanup_processes_on_exit._ctrl_handler, True)
        except Exception as e:
            logger.info(f"Notice: Could not set console control handler: {e}")
        os.chdir(config.harness_start_dir)

    logger.info("=" * 60)
    logger.info("       LOCAL AI LAUNCHER (Unified LocalAI Engine)")
    logger.info(f"  Harness:        {config.choose_harness}")
    logger.info(f"  Start Folder:   {config.harness_start_dir}")
    logger.info(f"  LocalAI Port:   {config.localai_port}")
    logger.info(f"  OmniRoute:      {'ON (compression proxy)' if config.use_omniroute_proxy else 'OFF (direct to LocalAI)'}")
    logger.info(f"  Starting model: {config.default_model}")
    logger.info(f"  Web GUI / Mode: {'Headless (no browser popup)' if config.localai_headless else f'Active (http://localhost:{config.localai_port})'}")
    logger.info(f"  Mutual Exclusion: SINGLE_ACTIVE_BACKEND=true (Blackwell, TabbyAPI, FreeToken)")
    logger.info("=" * 60)

    if config.run_mnemosyne_verification:
        logger.info("\nVerifying Mnemosyne memory & project configurations...")
        try:
            run_mnemosyne_verifier()
        except Exception as e:
            logger.warning(f"Warning: Mnemosyne verifier encountered an issue: {e}")
            
    logger.info("\nStarting services...")

    # 0. Determine selected harnesses early to launch concurrent pre-flight update track
    if isinstance(config.choose_harness, str):
        raw_harnesses = [config.choose_harness]
    elif isinstance(config.choose_harness, (list, tuple)):
        raw_harnesses = list(config.choose_harness)
    else:
        raw_harnesses = []

    selected_harnesses = [
        h.strip() for h in raw_harnesses
        if h and str(h).strip().lower() not in ("none", "off", "disabled", "no", "false")
    ]

    # Concurrent Track 1: Initiate harness pre-flight verification/updates in background immediately
    if selected_harnesses:
        start_harness_preflight_async(selected_harnesses, processes)
    
    # 1. Close stale OmniRoute instances if port occupied
    if is_port_in_use(OMNIROUTE_PORT):
        force_kill_port(OMNIROUTE_PORT)
        time.sleep(0.5)

    # 2. Check Git and GitHub CLI
    check_git()
    check_github()

    # 3. Start Intel NPU Voice-to-Text service
    start_voice_to_text(processes)

    # 4. Start SearXNG Metasearch service (if enabled and required by active harnesses)
    start_searxng_service(processes, selected_harnesses=selected_harnesses)

    # Concurrent Track 2: Start CPU Auxiliary Decision & Reranking engines in parallel with LLM engine
    default_m = str(getattr(config, "default_model", "")).lower()
    is_strata_default = "strata" in default_m or getattr(config, "engine", "") == "strata" or getattr(config, "blackwell_engine", "") == "strata"
    
    cpu_aux_future = None
    if not is_strata_default:
        cpu_aux_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="CpuAuxTrack")
        cpu_aux_future = cpu_aux_executor.submit(start_cpu_auxiliary_engines, processes)
    else:
        logger.info("[Strata Memory Policy] Strata is primary engine/model; skipping CPU auxiliary engines at startup to preserve RAM.")

    # Check if only ONLINE_ONLY harnesses or no harnesses are launched
    skip_offline_llm = not selected_harnesses or all(h in ONLINE_ONLY_HARNESSES for h in selected_harnesses)

    if skip_offline_llm:
        if not selected_harnesses:
            logger.info("\nNo agent harness selected; running in Voice-to-Text only mode (skipping offline LLM engine).")
        else:
            logger.info("\nOnline-only harness(es) selected; skipping offline LLM engine startup.")
        llm_base_url = None
        llm_thread = None
        active_model_url = None
    else:
        # Hybrid mode GPU pre-warm & wakeup probe (wakes dGPU from D3cold and stabilizes PCIe link)
        try:
            from local_ai_launcher.environment import warmup_gpu_for_hybrid_mode
            warmup_gpu_for_hybrid_mode()
        except Exception as e:
            logger.debug(f"Hybrid GPU warmup notice: {e}")

        # Pre-flight check / auto-installation / pruning of selected model
        try:
            from local_ai_launcher.backends.model_installer import ensure_single_model_installed
            ensure_single_model_installed(config.default_model)
        except Exception as e:
            logger.warning(f"Model pre-flight installation notice: {e}")

        # Ensure official llama.cpp binary is installed (acts as baseline runner and universal fallback)
        try:
            from local_ai_launcher.backends.llamacpp import check_llamacpp
            check_llamacpp()
        except Exception as e:
            logger.warning(f"llama.cpp pre-flight verification notice: {e}")

        # Pre-flight check / auto-installation of Blackwell / Huang engine binary if target engine is active
        try:
            from local_ai_launcher.backends.blackwell_builder import check_blackwell_llamacpp
            check_blackwell_llamacpp()
        except Exception as e:
            logger.warning(f"Engine pre-flight verification notice: {e}")

        # 5. Start LocalAI Unified Engine (handles Blackwell, TabbyAPI, and FreeToken via shims)
        llm_base_url, llm_thread, localai_mgr = start_localai_engine(processes)
        if not llm_base_url:
            logger.error("Failed to initialize LocalAI engine.")
            return

        # 6. Start OmniRoute compression proxy (if enabled)
        proxy_url = start_proxy(processes, llm_base_url)
        active_model_url = proxy_url if proxy_url else llm_base_url

    # Ensure concurrent CPU auxiliary startup track is complete before launching harnesses
    if cpu_aux_future is not None:
        try:
            cpu_aux_future.result(timeout=60)
        except Exception as e:
            logger.warning(f"Notice: CPU auxiliary engines startup: {e}")
        finally:
            if cpu_aux_executor:
                cpu_aux_executor.shutdown(wait=False)
                cpu_aux_executor = None

    # 7. Check and start selected harnesses
    if selected_harnesses:
        run_harnesses(processes, selected_harnesses, active_model_url, llm_thread)
    
    logger.info("\nAll services started successfully!")
    if llm_base_url:
        logger.info(f"LocalAI Endpoint: {llm_base_url}")
        logger.info(f"Active Harness Endpoint: {active_model_url}")
    elif not selected_harnesses:
        logger.info("Local Offline LLM Engine: Not started (Voice-to-Text only mode)")
        logger.info("Intel NPU Voice-to-Text: Active")
    else:
        logger.info("Local Offline LLM Engine: Not started (online-only mode)")
        logger.info("Intel NPU Voice-to-Text: Active")
    logger.info("Press Ctrl+C to exit launcher...")

    try:
        while True:
            agents = [item[1] for item in processes if len(item) >= 3 and item[2] is True]
            if agents and all(p.poll() is not None for p in agents):
                logger.info("All tracked coding agent console windows have been closed.")
                break
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("\nMain console interrupted by user (Ctrl+C).")
    finally:
        logger.info("Main console exiting: terminating all spawned services and LLM engine...")
        _cleanup_processes_on_exit()


if __name__ == "__main__":
    parse_cli_args()
    main()
