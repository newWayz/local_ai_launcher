"""
Aside AI Browser Adapter for local_ai_launcher.

Manages Aside AI browser installation verification, local engine configuration,
and interactive session launching.
"""

import os
import sys
import shutil
import logging
from pathlib import Path
from typing import Optional

from local_ai_launcher.config import config
from local_ai_launcher.paths import AI_WORKSPACE_DIR, get_aside_binary_path
from local_ai_launcher.process import build_ps_env, start_in_wt_tab, start_process
from local_ai_launcher.harness_base import HarnessAdapter
from local_ai_launcher.backends.aside import (
    is_aside_installed,
    setup_aside_local_engine,
    get_aside_mcp_server_config,
)

logger = logging.getLogger("service_launcher")


class AsideAdapter(HarnessAdapter):
    """Adapter for the Aside AI browser agent and CLI integration."""

    TRACK_CONSOLE_CLOSE = True

    def __init__(self, root_dir: Path, processes: list):
        super().__init__("aside", root_dir, processes)
        self.aside_bin = getattr(config, "aside_browser_bin", None) or get_aside_binary_path()

    def check(self) -> bool:
        logger.info("\nChecking for Aside browser CLI...")
        installed = is_aside_installed()
        if installed:
            bin_path = getattr(config, "aside_browser_bin", None) or get_aside_binary_path()
            logger.info(f"  -> Aside CLI is installed at: {bin_path}")
            return True
        else:
            logger.warning(
                "  -> Aside CLI is not detected. Please install Aside from https://aside.com "
                "or ensure 'aside.exe' is in %LOCALAPPDATA%\\Aside\\CLI\\current or system PATH."
            )
            return False

    def start(self, llm_base_url: str, active_model: Optional[str] = None, launch_tui: bool = True):
        model_name = active_model or config.default_model
        logger.info(f"Configuring Aside browser local engine endpoint at {llm_base_url} (model: {model_name})...")
        try:
            setup_aside_local_engine(llm_base_url, active_model=model_name)
            logger.info("  -> Aside browser configured to use local AI engine.")
        except Exception as e:
            logger.warning(f"  -> Warning: Failed to configure Aside local engine: {e}")

        if not launch_tui:
            logger.info("  -> Aside browser configured in headless mode (no TUI spawned).")
            return

        bin_path = getattr(config, "aside_browser_bin", None) or get_aside_binary_path()
        aside_cmd_str = f"& '{bin_path}'" if bin_path and Path(bin_path).is_file() else "aside"

        env_dict = {
            "OPENAI_BASE_URL": llm_base_url,
            "OPENAI_API_BASE": llm_base_url,
            "OPENAI_API_KEY": "dummy",
            "OPENAI_MODEL": model_name,
        }
        env_setup = build_ps_env(env_dict, harness_name="aside")
        logger.info("Starting Aside browser interactive CLI session...")
        try:
            cmd = f'powershell.exe -NoExit -Command "chcp 65001 | Out-Null; [console]::InputEncoding=[console]::OutputEncoding=[System.Text.Encoding]::UTF8; {env_setup}{aside_cmd_str}"'
            aside_proc = start_in_wt_tab(cmd, cwd=str(config.harness_start_dir))
            self.processes.append(("Aside browser CLI", aside_proc, self.TRACK_CONSOLE_CLOSE))
        except Exception as e:
            logger.error(f"Error starting Aside browser CLI: {e}")
