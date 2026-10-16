from local_ai_launcher.process import build_ps_env
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
from local_ai_launcher.environment import *
from local_ai_launcher.models import *
from local_ai_launcher.process import *

from local_ai_launcher.harness_base import HarnessAdapter, send_background_warmup

def setup_antigravity_mcp_config(gemini_dir: Optional[Path] = None) -> Path:
    """Ensure Antigravity mcp_config.json contains configured and enabled MCP tools (local-llm, rea)."""
    base_dir = Path(gemini_dir) if gemini_dir else (AI_WORKSPACE_DIR / ".gemini")
    config_dir = base_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    mcp_config_path = config_dir / "mcp_config.json"

    mcp_config = {}
    if mcp_config_path.exists():
        try:
            with open(mcp_config_path, "r", encoding="utf-8") as f:
                mcp_config = json.load(f)
        except Exception:
            mcp_config = {}
    if not isinstance(mcp_config, dict):
        mcp_config = {}

    mcp_servers = mcp_config.setdefault("mcpServers", {})
    if not isinstance(mcp_servers, dict):
        mcp_servers = {}
        mcp_config["mcpServers"] = mcp_servers

    mcp_port = getattr(config, 'mcp_server_port', 9932)
    try:
        mcp_port = int(mcp_port)
        if not (1 <= mcp_port <= 65535):
            mcp_port = 9932
    except (ValueError, TypeError):
        mcp_port = 9932

    mcp_servers["local-llm"] = {
        "serverUrl": f"http://127.0.0.1:{mcp_port}/mcp"
    }

    if getattr(config, "rea_enabled", True):
        rea_cmd = "cmd.exe" if sys.platform == "win32" else "npx"
        rea_args = ["/d", "/c", "npx", "-y", "rea-agents@latest", "mcp"] if sys.platform == "win32" else ["-y", "rea-agents@latest", "mcp"]
        mcp_servers["rea"] = {
            "command": rea_cmd,
            "args": rea_args,
            "env": {
                "REA_MCP_INPUT_SCHEMA_PROFILE": getattr(config, "rea_schema_profile", "compact"),
            }
        }
    else:
        mcp_servers.pop("rea", None)

    if getattr(config, "aside_browser_enabled", True):
        try:
            from local_ai_launcher.backends.aside import get_aside_mcp_server_config
            aside_cfg = get_aside_mcp_server_config()
            if aside_cfg:
                mcp_servers["aside"] = aside_cfg
            else:
                mcp_servers.pop("aside", None)
        except Exception as e:
            logger.warning(f"Notice: Failed to configure Aside MCP for Antigravity: {e}")
    else:
        mcp_servers.pop("aside", None)

    try:
        from local_ai_launcher.paths import atomic_write_json
        atomic_write_json(mcp_config_path, mcp_config)
        logger.info(f"  -> Configured Antigravity MCP config at {mcp_config_path}")
    except Exception as e:
        logger.warning(f"  -> Failed to write Antigravity MCP config: {e}")

    return mcp_config_path


class AntigravityAdapter(HarnessAdapter):
    """Adapter for the Google Antigravity CLI agent."""
    
    TRACK_CONSOLE_CLOSE = False
    
    def __init__(self, root_dir, processes):
        super().__init__("agy", root_dir, processes)
        self.local_bin = AI_WORKSPACE_DIR / "node_modules" / ".bin"
        self.gemini_dir = AI_WORKSPACE_DIR / ".gemini"
        
        self.agy_exe = self.local_bin / "agy.exe"

    def _configure_settings_and_rules(self):
        """Configure settings.json and create no-feedback rule inside .gemini directory."""
        user_home = Path(os.path.expanduser("~"))
        if user_home.exists():
            ensure_directory_junction(user_home / ".gemini" / "antigravity-cli", self.gemini_dir / "antigravity-cli")
            ensure_directory_junction(user_home / ".gemini" / "config", self.gemini_dir / "config")
        setup_antigravity_mcp_config(self.gemini_dir)

        cli_dir = self.gemini_dir / "antigravity-cli"
        cli_dir.mkdir(parents=True, exist_ok=True)
        settings_path = cli_dir / "settings.json"
        
        settings_data = {}
        if settings_path.exists():
            try:
                with open(settings_path, "r", encoding="utf-8") as f:
                    settings_data = json.load(f)
            except Exception:
                pass
        
        if not isinstance(settings_data, dict):
            settings_data = {}
        
        settings_data["enableTelemetry"] = False
        try:
            with open(settings_path, "w", encoding="utf-8") as f:
                json.dump(settings_data, f, indent=2)
            logger.info(f"  -> Configured Antigravity settings: enableTelemetry set to False in {settings_path}")
        except Exception as e:
            logger.warning(f"  -> Warning: Could not update Antigravity settings.json: {e}")

        # Enforce global / workspace no feedback rule
        rules_dir = self.gemini_dir / "config" / "rules"
        rules_dir.mkdir(parents=True, exist_ok=True)
        rule_file = rules_dir / "no_feedback_prompts.md"
        rule_content = (
            "# Behavioral Instruction: No Feedback Prompts\n\n"
            "- NEVER ask the user how I am doing, how your performance was, or request feedback on past responses.\n"
            "- NEVER append closing check-in questions regarding my service quality (e.g., 'How did I do?', 'Was this helpful?').\n"
            "- Always focus strictly on answering the user's explicit task concisely and directly without conversational meta-feedback prompts.\n"
        )
        try:
            rule_file.write_text(rule_content, encoding="utf-8")
            logger.info(f"  -> Enforced Antigravity rule: feedback prompts disabled via {rule_file}")
        except Exception as e:
            logger.warning(f"  -> Warning: Could not write no_feedback_prompts rule: {e}")
    
    def _download_agy(self):
        logger.info("  -> Fetching latest Antigravity CLI URL from winget...")
        try:
            out = subprocess.check_output(
                ["winget", "show", "--id", "Google.AntigravityCLI", "--exact"], 
                text=True, encoding='utf-8', errors='ignore'
            )
            m = re.search(r'https://[^\s]+cli_windows_x64\.exe', out)
            if not m:
                logger.error("  -> Could not parse installer URL from winget output.")
                return False
            url = m.group(0)
            logger.info(f"  -> Downloading from {url} ... this may take a minute.")
            
            # Download to a temporary file first
            self.agy_exe.parent.mkdir(parents=True, exist_ok=True)
            temp_file = self.agy_exe.with_suffix('.tmp')
            req = urllib.request.Request(url, headers={'User-Agent': 'AgyDownloader'})
            with urllib.request.urlopen(req, timeout=60) as resp, open(temp_file, 'wb') as out_f:
                shutil.copyfileobj(resp, out_f)
            
            # Atomic rename/replace temp to final
            os.replace(temp_file, self.agy_exe)
            
            logger.info(f"  -> Antigravity CLI downloaded successfully to {self.agy_exe}.")
            return True
        except Exception as e:
            logger.error(f"  -> Failed to download Antigravity CLI: {e}")
            return False

    def check(self):
        logger.info(f"\nChecking for Antigravity CLI (agy)...")
        os.environ['GEMINI_DIR'] = str(self.gemini_dir)
        self._configure_settings_and_rules()
        
        is_installed = self.agy_exe.exists()
        
        if is_installed:
            logger.info(f"  -> Local Antigravity CLI is ready at {self.agy_exe}.")
            if self._should_update():
                logger.info("Updating Antigravity CLI...")
                if self._download_agy():
                    self.record_update()
            return True
        else:
            logger.info(f"  -> Antigravity CLI is not installed locally in workspace.")
            if self._should_install():
                logger.info("Installing Antigravity CLI...")
                return self._download_agy()
            else:
                logger.info(f"  -> Skipping Antigravity CLI installation.")
                return False

    def start(self, llm_base_url, active_model=None):
        logger.info("Starting Antigravity CLI (agy)...")
        self._configure_settings_and_rules()
        if getattr(config, "aside_browser_enabled", True) and getattr(config, "aside_browser_auto_sync", True):
            try:
                from local_ai_launcher.backends.aside import setup_aside_local_engine, is_aside_installed
                if is_aside_installed():
                    setup_aside_local_engine(llm_base_url, active_model=active_model)
            except Exception as e:
                logger.warning(f"Notice: Failed to configure Aside browser for Antigravity: {e}")
        env_setup = build_ps_env({
            'GEMINI_DIR': self.gemini_dir,
        })
        agy_bin = f"& '{self.agy_exe}'" if isinstance(self.agy_exe, Path) and self.agy_exe.exists() else "agy"
        agy_cmd = f"{agy_bin} --add-dir '{config.harness_start_dir}' --dangerously-skip-permissions"
        try:
            agy_process = start_in_wt_tab(f'powershell.exe -NoExit -Command "{env_setup}{agy_cmd}"', cwd=str(config.harness_start_dir))
            self.processes.append(('Antigravity CLI', agy_process, self.TRACK_CONSOLE_CLOSE))
        except Exception as e:
            logger.error(f"Error starting Antigravity: {e}")



class BaseAntigravityAdapter(HarnessAdapter):
    def _start_mcp_service(self):
        mcp_port = getattr(config, 'mcp_server_port', 9932)
        try:
            mcp_port = int(mcp_port)
            if not (1 <= mcp_port <= 65535):
                mcp_port = 9932
        except (ValueError, TypeError):
            mcp_port = 9932

        if is_port_in_use(mcp_port):
            logger.info(f"  -> MCP Server port {mcp_port} is already in use.")
        else:
            logger.info(f"  -> Starting MCP Server on port {mcp_port}...")
            env_setup = build_ps_env({
                'OPENAI_BASE_URL': f"http://127.0.0.1:{LLM_BACKEND_PORT}/v1",
                'OPENAI_MODEL': config.default_model
            })
            cmd = f'uvicorn local_ai_launcher.mcp_server:app --port {mcp_port} --host 127.0.0.1'
            try:
                proc = subprocess.Popen(
                    f'powershell.exe -WindowStyle Hidden -Command "{env_setup}{cmd}"',
                    creationflags=subprocess.CREATE_NO_WINDOW
                )
                self.processes.append(('MCP Server', proc, False))
                start = time.time()
                while time.time() - start < 10.0:
                    if is_port_in_use(mcp_port):
                        break
                    time.sleep(0.5)
            except Exception as e:
                logger.error(f"  -> Failed to start MCP Server: {e}")
                
        if is_port_in_use(LLM_BACKEND_PORT):
            send_background_warmup(
                llm_base_url=f"http://127.0.0.1:{LLM_BACKEND_PORT}/v1",
                model_name=config.default_model
            )

    def _configure_local_llm_endpoint(self):
        user_home = Path(os.path.expanduser("~"))
        if user_home.exists():
            ensure_directory_junction(user_home / ".gemini" / "config", AI_WORKSPACE_DIR / ".gemini" / "config")
        setup_antigravity_mcp_config()
        if getattr(config, "aside_browser_enabled", True) and getattr(config, "aside_browser_auto_sync", True):
            try:
                from local_ai_launcher.backends.aside import setup_aside_local_engine
                setup_aside_local_engine(f"http://127.0.0.1:{LLM_BACKEND_PORT}/v1", active_model=config.default_model)
            except Exception as e:
                logger.warning(f"Notice: Failed to configure Aside browser for Antigravity: {e}")

class AgyIdeAdapter(BaseAntigravityAdapter):
    def __init__(self, root_dir, processes):
        super().__init__("agy-ide", root_dir, processes)
        self.install_dir = config.agy_ide_dir
        
    def check(self):
        logger.info("\nChecking for Antigravity IDE (agy-ide)...")
        self._configure_local_llm_endpoint()
        
        if not self.install_dir.exists() or not list(self.install_dir.glob("*.exe")):
            if self._should_install():
                logger.info(f"  -> Installing Antigravity IDE via winget to {self.install_dir}...")
                try:
                    subprocess.check_call(
                        ["winget", "install", "--id", "Google.AntigravityIDE", "--exact", "--silent", "--accept-package-agreements", "--accept-source-agreements", "--location", str(self.install_dir)]
                    )
                    return True
                except Exception as e:
                    logger.error(f"  -> Failed to install Antigravity IDE: {e}")
                    return False
            return False
        return True
        
    def start(self, llm_base_url, active_model=None):
        logger.info("Starting Antigravity IDE...")
        self._start_mcp_service()
        
        exe_path = self.install_dir / "AntigravityIDE.exe"
        if not exe_path.exists():
            exes = [e for e in self.install_dir.glob("*.exe") if "unins" not in e.name.lower()]
            if exes:
                exe_path = exes[0]

        
        try:
            if not exe_path.exists():
                logger.warning(f"  -> {exe_path} not found. Mocking IDE launch.")
                env_setup = build_ps_env({'GEMINI_DIR': AI_WORKSPACE_DIR / ".gemini"})
                proc = subprocess.Popen(
                    f'powershell.exe -WindowStyle Hidden -Command "{env_setup}Start-Sleep -Seconds 3600"',
                    cwd=str(config.harness_start_dir),
                    creationflags=subprocess.CREATE_NO_WINDOW
                )
            else:
                my_env = os.environ.copy()
                my_env['GEMINI_DIR'] = str(AI_WORKSPACE_DIR / ".gemini")
                proc = subprocess.Popen(
                    [str(exe_path), f"--user-data-dir={self.install_dir / 'userdata'}", f"--extensions-dir={self.install_dir / 'extensions'}"],
                    cwd=str(config.harness_start_dir),
                    env=my_env
                )
                
            self.processes.append(('Antigravity IDE', proc, False))
            logger.info(f"  -> Antigravity IDE running at {self.install_dir}")
        except Exception as e:
            logger.error(f"  -> Error starting Antigravity IDE: {e}")

class AgyDesktopAdapter(BaseAntigravityAdapter):
    def __init__(self, root_dir, processes):
        super().__init__("agy-desktop", root_dir, processes)
        self.install_dir = config.agy_desktop_dir
        
    def check(self):
        logger.info("\nChecking for Antigravity Desktop (agy-desktop)...")
        self._configure_local_llm_endpoint()
        
        if not self.install_dir.exists() or not list(self.install_dir.glob("*.exe")):
            if self._should_install():
                logger.info(f"  -> Installing Antigravity Desktop via winget to {self.install_dir}...")
                try:
                    subprocess.check_call(
                        ["winget", "install", "--id", "Google.Antigravity", "--exact", "--silent", "--accept-package-agreements", "--accept-source-agreements", "--location", str(self.install_dir)]
                    )
                    return True
                except Exception as e:
                    logger.error(f"  -> Failed to install Antigravity Desktop: {e}")
                    return False
            return False
        return True
        
    def start(self, llm_base_url, active_model=None):
        logger.info("Starting Antigravity Desktop...")
        self._start_mcp_service()
        
        exe_path = self.install_dir / "Antigravity.exe"
        if not exe_path.exists():
            exes = [e for e in self.install_dir.glob("*.exe") if "unins" not in e.name.lower()]
            if exes:
                exe_path = exes[0]
        
        try:
            if not exe_path.exists():
                logger.warning(f"  -> {exe_path} not found. Mocking Desktop launch.")
                env_setup = build_ps_env({'GEMINI_DIR': AI_WORKSPACE_DIR / ".gemini"})
                proc = subprocess.Popen(
                    f'powershell.exe -WindowStyle Hidden -Command "{env_setup}Start-Sleep -Seconds 3600"',
                    cwd=str(config.harness_start_dir),
                    creationflags=subprocess.CREATE_NO_WINDOW
                )
            else:
                my_env = os.environ.copy()
                my_env['GEMINI_DIR'] = str(AI_WORKSPACE_DIR / ".gemini")
                proc = subprocess.Popen(
                    [str(exe_path), f"--user-data-dir={self.install_dir / 'userdata'}"],
                    cwd=str(config.harness_start_dir),
                    env=my_env
                )
                
            self.processes.append(('Antigravity Desktop', proc, False))
            logger.info(f"  -> Antigravity Desktop running at {self.install_dir}")
        except Exception as e:
            logger.error(f"  -> Error starting Antigravity Desktop: {e}")
