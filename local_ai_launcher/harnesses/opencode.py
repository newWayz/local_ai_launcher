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

from local_ai_launcher.harness_base import HarnessAdapter

class OpenCodeAdapter(HarnessAdapter):
    """Adapter for OpenCode - terminal-native AI coding agent.
    
    OpenCode is a Go-based TUI coding agent that supports multiple LLM providers.
    It can be installed via npm (opencode-ai) or Go (go install).
    """
    
    def __init__(self, root_dir, processes):
        super().__init__("opencode", root_dir, processes)
        self.local_bin = AI_WORKSPACE_DIR / "node_modules" / ".bin"
        self.go_bin = AI_WORKSPACE_DIR / "go" / "bin"
        self.opencode_exe = (
            (self.local_bin / "opencode.cmd") if (self.local_bin / "opencode.cmd").exists()
            else ((self.go_bin / "opencode.exe") if (self.go_bin / "opencode.exe").exists()
            else Path(shutil.which("opencode") or "opencode"))
        )
    
    def check(self):
        logger.info(f"\nChecking for OpenCode terminal coding agent...")
        
        is_installed = (self.local_bin / "opencode.cmd") .exists() or (self.go_bin / "opencode.exe").exists() or shutil.which("opencode")
        
        if is_installed:
            logger.info(f"  -> OpenCode is already installed.")
            if self._should_update():
                logger.info("Updating OpenCode...")
                try:
                    subprocess.run([get_npm_exe(), "install", "--prefix", str(AI_WORKSPACE_DIR), "opencode-ai"], shell=NPM_SHELL, check=True)
                    logger.info(f"  -> OpenCode updated successfully via npm.")
                    self.record_update()
                except subprocess.CalledProcessError as e:
                    logger.error(f"  -> Failed to update OpenCode via npm: {e}")
                    # Fallback: try go install
                    try:
                        go_env = os.environ.copy()
                        go_env["GOPATH"] = str(AI_WORKSPACE_DIR / "go")
                        subprocess.run(
                            ["go", "install", "github.com/opencode-ai/opencode@latest"],
                            shell=False, check=True, env=go_env
                        )
                        logger.info(f"  -> OpenCode updated successfully via go install.")
                        self.record_update()
                    except (subprocess.CalledProcessError, FileNotFoundError):
                        logger.error(f"  -> Failed to update OpenCode via go install as well.")
            return True
        else:
            logger.info(f"  -> OpenCode is not installed.")
            if self._should_install():
                logger.info("Installing OpenCode into workspace...")
                # Try npm first, then go
                try:
                    subprocess.run([get_npm_exe(), "install", "--prefix", str(AI_WORKSPACE_DIR), "opencode-ai"], shell=NPM_SHELL, check=True)
                    logger.info(f"  -> OpenCode installed successfully via npm in {AI_WORKSPACE_DIR}.")
                    return True
                except (subprocess.CalledProcessError, FileNotFoundError):
                    logger.info(f"  -> npm install failed, trying go install...")
                    try:
                        go_env = os.environ.copy()
                        go_env["GOPATH"] = str(AI_WORKSPACE_DIR / "go")
                        subprocess.run(
                            ["go", "install", "github.com/opencode-ai/opencode@latest"],
                            shell=False, check=True, env=go_env
                        )
                        logger.info(f"  -> OpenCode installed successfully via go install in {AI_WORKSPACE_DIR}.")
                        return True
                    except (subprocess.CalledProcessError, FileNotFoundError) as e:
                        logger.error(f"  -> Failed to install OpenCode: {e}")
                        logger.info(f"  -> Install manually: npm install --prefix {AI_WORKSPACE_DIR} opencode-ai")
                        return False
            else:
                logger.info(f"  -> Skipping OpenCode installation.")
                return False
    
    def start(self, llm_base_url, active_model=None):
        model_name = resolve_model_name(active_model)

        opencode_args = f"--model {model_name}"
        opencode_bin = f"& '{self.opencode_exe}'" if isinstance(self.opencode_exe, Path) and self.opencode_exe.exists() else "opencode"
        opencode_cmd = f"{opencode_bin} {opencode_args}"

        # Pre-evaluate OpenCode agent system prompt and tool definitions into KV cache for sub-second TTFT
        from local_ai_launcher.context_preloader import preload_harness_context
        preload_harness_context(llm_base_url, model_name, harness_name="opencode")

        env_setup = build_ps_env({
            'OPENAI_BASE_URL': llm_base_url,
            'OPENAI_API_BASE': llm_base_url,
            'OPENAI_API_KEY': 'dummy',
            'OPENAI_MODEL': model_name,
            'MODEL': model_name,
        })
        logger.info("Starting OpenCode agent...")
        try:
            opencode_process = start_in_wt_tab(f'powershell.exe -NoExit -Command "{env_setup}{opencode_cmd}"', cwd=str(config.harness_start_dir))
            self.processes.append(('OpenCode agent', opencode_process, self.TRACK_CONSOLE_CLOSE))
        except Exception as e:
            logger.error(f"Error starting OpenCode: {e}")


