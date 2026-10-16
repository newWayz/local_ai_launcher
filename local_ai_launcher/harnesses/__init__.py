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

from local_ai_launcher.harnesses.hermes import HermesAdapter
from local_ai_launcher.harnesses.pi import PiAdapter, setup_pi_mcp_config
from local_ai_launcher.harnesses.omp import OmpAdapter, setup_omp_models_and_auth
from local_ai_launcher.harnesses.pie import PieAdapter
from local_ai_launcher.harnesses.opencode import OpenCodeAdapter
from local_ai_launcher.harnesses.antigravity import AntigravityAdapter, AgyIdeAdapter, AgyDesktopAdapter, setup_antigravity_mcp_config
from local_ai_launcher.harnesses.llama_server_ui import LsAdapter
from local_ai_launcher.harnesses.deepseek import DeepseekAdapter
from local_ai_launcher.harnesses.orca import OrcaAdapter
from local_ai_launcher.harnesses.t3 import T3Adapter
from local_ai_launcher.harnesses.openlumara import OpenLumaraAdapter
from local_ai_launcher.harnesses.claude import (
    ClaudeAdapter,
    ClaudeLocalAdapter,
    ClaudeOfficialAdapter,
    ClaudeDesktopAdapter,
    ClaudeDesktopOfficialAdapter,
    ClaudeDesktopGatewayAdapter,
    setup_claude_cli_mcp_config,
    setup_claude_desktop_config,
)
from local_ai_launcher.harnesses.aside import AsideAdapter

HARNESS_REGISTRY = {
    "hermes": HermesAdapter,
    "pi": PiAdapter,
    "pie": PieAdapter,
    "omp": OmpAdapter,
    "oh-my-pi": OmpAdapter,
    "ohmypi": OmpAdapter,
    "opencode": OpenCodeAdapter,
    "agy": AntigravityAdapter,
    "agy-ide": AgyIdeAdapter,
    "agy-desktop": AgyDesktopAdapter,
    "ls": LsAdapter,
    "deepseek": DeepseekAdapter,
    "dsh": DeepseekAdapter,
    "orca": OrcaAdapter,
    "t3": T3Adapter,
    "t3code": T3Adapter,
    "openlumara": OpenLumaraAdapter,
    "lumara": OpenLumaraAdapter,
    "antigravity": AntigravityAdapter,
    "claude": ClaudeAdapter,
    "claude-local": ClaudeLocalAdapter,
    "claude-official": ClaudeOfficialAdapter,
    "claude-code": ClaudeAdapter,
    "claude-code-local": ClaudeLocalAdapter,
    "claude-code-official": ClaudeOfficialAdapter,
    "claudecode": ClaudeAdapter,
    "claude-desktop": ClaudeDesktopAdapter,
    "claude-desktop-official": ClaudeDesktopOfficialAdapter,
    "claude-desktop-mcp": ClaudeDesktopOfficialAdapter,
    "claude-desktop-gateway": ClaudeDesktopGatewayAdapter,
    "claude-desktop-local": ClaudeDesktopGatewayAdapter,
    "claudedesktop": ClaudeDesktopAdapter,
    "aside": AsideAdapter,
    "aside-browser": AsideAdapter,
    "aside-cli": AsideAdapter,
}


ONLINE_ONLY_HARNESSES = {
    "agy", "agy-ide", "agy-desktop", "antigravity",
    "claude-official", "claude-code-official"
}


