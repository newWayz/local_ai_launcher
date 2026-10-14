from local_ai_launcher.config import prompt_choice
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

class HarnessAdapter(ABC):
    """Base class for all coding harness adapters.
    
    To add a new harness:
    1. Create a subclass of HarnessAdapter
    2. Implement check(), start(), and get_needed_libs()
    3. Register it in HARNESS_REGISTRY at the bottom of this section
    """
    
    REQUIRED_LIBS = []
    TRACK_CONSOLE_CLOSE = True
    
    def __init__(self, name, root_dir, processes):
        self.name = name
        self.root_dir = root_dir
        self.processes = processes
        self.install_behavior = config.harness_install_behaviors.get(name, 'existing')
        self.update_behavior = config.harness_update_behaviors.get(name, 'never_update')
        self.update_frequency_hrs = getattr(config, 'harness_update_frequency_hrs', {}).get(
            name, getattr(config, 'update_frequency_hrs', 168)
        )
    
    @abstractmethod
    def check(self):
        """Check if the harness is installed. Install/update as needed.
        Returns True if the harness is ready to use, False otherwise."""
        pass
    
    @abstractmethod
    def start(self, llm_base_url, active_model=None):
        """Start the harness, pointing it at the given LLM base URL.
        
        Args:
            llm_base_url: The OpenAI-compatible base URL to use.
                          When OmniRoute proxy is enabled, this will be
                          http://localhost:20128/v1 (OmniRoute).
                          Otherwise, http://localhost:8080/v1 (LocalAI unified server direct).
            active_model: The name or alias of the model to configure by default.
        """
        pass
    
    def get_needed_libs(self):
        """Return a list of pip libraries needed in the workspace venv.
        Override in subclasses that need additional libraries."""
        return list(self.REQUIRED_LIBS)
    
    def _should_install(self):
        """Determine if we should install based on configured behavior."""
        if self.install_behavior == 'never_install':
            logger.info(f"  -> Skipping {self.name} installation as per configuration.")
            return False
        elif self.install_behavior == 'always_install':
            return True
        else:  # 'existing' - ask user
            return prompt_choice(f"Do you want to install {self.name}? (y/n): ", default='n')
    
    def _should_update(self):
        """Determine if we should update based on configured behavior and frequency schedule."""
        from local_ai_launcher.update_frequency import should_update_component
        force = getattr(config, 'force_update', False)
        return should_update_component(
            component_name=self.name,
            category="harness",
            behavior=self.update_behavior,
            frequency=self.update_frequency_hrs,
            force=force,
            prompt_if_existing=True,
        )

    def record_update(self):
        """Record successful update timestamp for this harness in update_state.json."""
        from local_ai_launcher.update_frequency import record_successful_update
        record_successful_update(f"harness:{self.name}")


def escape_ps_arg(val) -> str:
    """Safely escape single quotes for PowerShell single-quoted string literals."""
    return str(val).replace("'", "''")


def send_background_warmup(llm_base_url: str, model_name: str, prompt: Optional[str] = None):
    """Fire off a silent background thread to warm up the LLM KV cache without spawning powershell."""
    warmup_prompt = prompt or "ping"

    def _worker():
        try:
            time.sleep(1.0)
            url = f"{llm_base_url.rstrip('/')}/chat/completions"
            data = json.dumps({
                "model": model_name,
                "messages": [{"role": "user", "content": warmup_prompt}],
                "max_tokens": 1
            }).encode("utf-8")
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", "User-Agent": "HarnessWarmup"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                resp.read()
        except Exception:
            pass

    threading.Thread(target=_worker, daemon=True).start()



