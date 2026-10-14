import sys
import os
import time
import shutil
import logging
import subprocess
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional, Tuple, List

from local_ai_launcher.config import (
    AI_WORKSPACE_DIR,
    LOCALAI_MODELS_DIR,
    config,
    logger
)
from local_ai_launcher.localai.model_configs import generate_all_model_configs
from local_ai_launcher.backends.localai import check_localai
from local_ai_launcher.process import start_process, is_port_in_use, force_kill_port


class LocalAIManager:
    """
    Manages the lifecycle of LocalAI server (Unified Local LLM Engine).
    Provides automatic detection and fallback between Docker, WSL2, and Native execution modes.
    """
    def __init__(self, port: int = 8080, host: str = "127.0.0.1"):
        try:
            p = int(port)
            self.port = 8080 if p < 1 or p > 65535 else p
        except (ValueError, TypeError):
            self.port = 8080
        self.host = host
        self.base_url = f"http://{host}:{self.port}"
        self.api_url = f"http://{host}:{self.port}/v1"
        self.process = None

    def is_running(self) -> bool:
        """Check if LocalAI is responsive to HTTP health checks."""
        for path in ("/healthz", "/readyz", "/v1/models", "/"):
            url = f"{self.base_url}{path}"
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "LocalAIHealthCheck"})
                with urllib.request.urlopen(req, timeout=1.0) as resp:
                    if resp.status in (200, 204, 302, 404):
                        return True
            except urllib.error.HTTPError as e:
                if e.code in (200, 204, 302, 404):
                    return True
            except Exception:
                pass
        return False

    def wait_for_ready(self, timeout: int = 30) -> bool:
        """Poll until LocalAI is responsive."""
        try:
            timeout = max(1, int(timeout))
        except (ValueError, TypeError):
            timeout = 30
        logger.info(f"Waiting for LocalAI engine at {self.base_url} (up to {timeout}s)...")
        start = time.time()
        while time.time() - start < timeout:
            if self.is_running():
                logger.info(f"  -> LocalAI engine is ready ({time.time() - start:.1f}s)!")
                return True
            time.sleep(0.5)
        logger.warning(f"  -> Timed out waiting for LocalAI after {timeout}s.")
        return False

    def start(self, processes: Optional[List] = None) -> Tuple[str, Optional[subprocess.Popen]]:
        """
        Starts LocalAI engine:
        1. Generates model YAML configs for all models (Blackwell, TabbyAPI, FreeToken, llama-cpp).
        2. If already running on port, attaches to existing instance.
        3. Otherwise starts via native server / container based on environment.
        """
        # 1. Check LocalAI repository / asset installation and update status
        check_localai()

        # 2. Generate model configuration files in localai/models/
        logger.info("Generating LocalAI model configurations...")
        generate_all_model_configs()

        # 3. Check if already running
        if self.is_running():
            logger.info(f"  -> LocalAI engine is already running on port {self.port}.")
            return self.api_url, None

        if is_port_in_use(self.port):
            logger.info(f"Port {self.port} is in use but unresponsive. Freeing port...")
            force_kill_port(self.port)
            time.sleep(1.0)

        # 3. Start Native LocalAI Server
        logger.info(f"Starting LocalAI Unified Engine on port {self.port}...")
        python_exe = sys.executable
        native_script = AI_WORKSPACE_DIR / "local_ai_launcher" / "localai" / "native_server.py"

        cmd = [
            str(python_exe),
            "-m", "local_ai_launcher.localai.native_server",
            "--host", str(self.host),
            "--port", str(self.port)
        ]

        self.process = start_process(
            f"LocalAI Engine (port {self.port})",
            cmd,
            cwd=str(AI_WORKSPACE_DIR),
            show_window=config.show_llama_server_console,
            use_shell=False
        )

        if self.process and processes is not None:
            processes.append(('LocalAI Engine', self.process))

        self.wait_for_ready(timeout=25)
        return self.api_url, self.process

    def stop(self):
        """Stops the LocalAI engine process if launched by this manager."""
        if self.process and self.process.poll() is None:
            logger.info("Stopping LocalAI engine...")
            try:
                self.process.terminate()
                self.process.wait(timeout=3)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
            self.process = None
