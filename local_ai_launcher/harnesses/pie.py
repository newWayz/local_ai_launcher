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

class PieAdapter(HarnessAdapter):
    """Adapter for Pie - the Rust rewrite of Pi coding harness.
    
    Pie is a high-performance, single-binary Rust implementation of the Pi agent.
    It requires Rust/Cargo to be installed for building from source,
    or can use a pre-built binary if available.
    """
    
    def __init__(self, root_dir, processes):
        super().__init__("pie", root_dir, processes)
        self.pie_install_dir = AI_WORKSPACE_DIR / "pie"
        self.pie_exe = self.pie_install_dir / "target" / "release" / "pie.exe"
        # Also check if 'pie' is globally available (e.g. via cargo install)
        self.pie_global = shutil.which("pie")
    
    def check(self):
        logger.info(f"\nChecking for Pie coding harness (Rust)...")
        
        is_installed = self.pie_exe.exists() or self.pie_global
        
        if is_installed:
            location = str(self.pie_exe) if self.pie_exe.exists() else self.pie_global
            logger.info(f"  -> Pie coding harness is already installed at: {location}")
            if self._should_update():
                logger.info("Updating Pie coding harness...")
                if self.pie_install_dir.exists() and (self.pie_install_dir / "Cargo.toml").exists():
                    try:
                        res = subprocess.run(["git", "pull", "--autostash"], cwd=str(self.pie_install_dir), timeout=120, check=True, capture_output=True, text=True)
                        sys.stdout.write(res.stdout)
                        if res.stderr: sys.stderr.write(res.stderr)
                        if "Already up to date." not in res.stdout:
                            subprocess.run(["cargo", "build", "--release"], cwd=str(self.pie_install_dir), timeout=600, check=True)
                        logger.info(f"  -> Pie coding harness updated successfully.")
                        self.record_update()
                    except (subprocess.CalledProcessError, FileNotFoundError) as e:
                        logger.error(f"  -> Failed to update Pie: {e}")
                else:
                    logger.info(f"  -> Cannot update: source directory not found at {self.pie_install_dir}")
            return True
        else:
            logger.info(f"  -> Pie coding harness is not installed.")
            if self._should_install():
                logger.info("Installing Pie coding harness from source...")
                try:
                    # Check for Rust/Cargo
                    cargo_check = shutil.which("cargo")
                    if not cargo_check:
                        logger.error("  -> Error: Rust/Cargo is not installed. Install from https://rustup.rs/")
                        return False
                    
                    # Clone and build
                    if not self.pie_install_dir.exists():
                        subprocess.run(
                            ["git", "clone", "https://github.com/c4pt0r/pie.git", str(self.pie_install_dir)],
                            check=True
                        )
                    subprocess.run(
                        ["cargo", "build", "--release"],
                        cwd=str(self.pie_install_dir),
                        check=True
                    )
                    logger.info(f"  -> Pie coding harness installed successfully at: {self.pie_exe}")
                    return True
                except (subprocess.CalledProcessError, FileNotFoundError) as e:
                    logger.error(f"  -> Failed to install Pie: {e}")
                    return False
            else:
                logger.info(f"  -> Skipping Pie coding harness installation.")
                return False
    
    def start(self, llm_base_url, active_model=None):
        model_name = resolve_model_name(active_model)

        pie_args = f"--model {model_name}"
        pie_bin = f"& '{self.pie_exe}'" if self.pie_exe.exists() else "pie"
        pie_cmd = f"{pie_bin} {pie_args}"

        # Pre-evaluate Pi/Pie agent system prompt and tool definitions into KV cache for sub-second TTFT
        from local_ai_launcher.context_preloader import preload_harness_context
        preload_harness_context(llm_base_url, model_name, harness_name="pi")

        env_setup = build_ps_env({
            'OPENAI_BASE_URL': llm_base_url,
            'OPENAI_API_BASE': llm_base_url,
            'OPENAI_API_KEY': 'dummy',
            'PIE_MODEL': model_name,
            'OPENAI_MODEL': model_name,
        })

        logger.info("Starting Pie coding agent...")
        try:
            pie_process = start_in_wt_tab(f'powershell.exe -NoExit -Command "{env_setup}{pie_cmd}"', cwd=str(config.harness_start_dir))
            self.processes.append(('Pie coding agent', pie_process, self.TRACK_CONSOLE_CLOSE))
        except Exception as e:
            logger.error(f"Error starting Pie: {e}")


