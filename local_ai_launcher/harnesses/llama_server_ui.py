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

class LsAdapter(HarnessAdapter):
    """Adapter for opening llama-server UI in browser."""
    
    TRACK_CONSOLE_CLOSE = False
    
    def __init__(self, root_dir, processes):
        super().__init__("ls", root_dir, processes)
        
    def check(self):
        return True
        
    def start(self, llm_base_url, active_model=None):
        logger.info("Waiting for llama-server to be available...")
        if wait_for_http_service(f"http://127.0.0.1:{LLM_BACKEND_PORT}/health", "llama-server", timeout=120):
            logger.info("Waiting for the model to be fully loaded before opening UI...")
            
            model_name = resolve_model_name(active_model, False, None)
            chat_url = f"http://127.0.0.1:{LLM_BACKEND_PORT}/v1/chat/completions"
            payload = json.dumps({
                "model": model_name,
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 1,
                "temperature": 0.0
            }).encode("utf-8")
            
            start_time = time.time()
            model_loaded = False
            while time.time() - start_time < 300: # Wait up to 5 minutes
                try:
                    req = urllib.request.Request(
                        chat_url,
                        data=payload,
                        headers={"Content-Type": "application/json", "User-Agent": "LsHarness-Wait"},
                        method="POST"
                    )
                    with urllib.request.urlopen(req, timeout=30) as response:
                        if response.status == 200:
                            model_loaded = True
                            break
                except Exception:
                    pass
                time.sleep(2)
                
            if model_loaded:
                logger.info("Model is fully loaded.")
            else:
                logger.warning("Timed out waiting for model to load.")
                
            logger.info(f"Opening llama-server UI in browser at http://127.0.0.1:{LLM_BACKEND_PORT}")
            webbrowser.open(f"http://127.0.0.1:{LLM_BACKEND_PORT}")


