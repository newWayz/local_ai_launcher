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

def check_omniroute():
    """Check if omniroute is installed and provide options to install or update."""
    logger.info(f"\nChecking for omniroute...")
    
    local_bin = AI_WORKSPACE_DIR / "node_modules" / ".bin"
    omniroute_cmd = local_bin / "omniroute.cmd"
    is_installed = omniroute_cmd.exists() or shutil.which("omniroute")
    
    if is_installed:
        logger.info(f"  -> omniroute is already installed.")
        # Determine update behavior
        update_behavior = config.omniroute_update_behavior if config.omniroute_update_behavior != 'existing' else 'existing'
        update_frequency_hrs = getattr(config, 'engine_update_frequency_hrs', {}).get(
            'omniroute', getattr(config, 'update_frequency_hrs', 168)
        )
        force = getattr(config, 'force_update', False)

        from local_ai_launcher.update_frequency import should_update_component, record_successful_update
        do_update = should_update_component(
            component_name="omniroute",
            category="engine",
            behavior=update_behavior,
            frequency=update_frequency_hrs,
            force=force,
            prompt_if_existing=True,
        )

        if do_update:
            logger.info("Updating omniroute...")
            try:
                subprocess.run([get_npm_exe(), "install", "--prefix", str(AI_WORKSPACE_DIR), "omniroute"], shell=NPM_SHELL, check=True)
                logger.info(f"  -> omniroute updated successfully.")
                record_successful_update("engine:omniroute")
                return True
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                logger.error(f"  -> Failed to update omniroute: {e}")
                return False
        return True
    else:
        logger.info(f"  -> omniroute is not installed.")
        # Determine install behavior
        install_behavior = config.omniroute_install_behavior if config.omniroute_install_behavior != 'existing' else 'existing'
        
        # Handle installation based on behavior
        if install_behavior == 'never_install':
            logger.info(f"  -> Skipping omniroute installation as per configuration.")
            return False
        elif install_behavior == 'always_install':
            logger.info("Installing omniroute...")
            try:
                subprocess.run([get_npm_exe(), "install", "--prefix", str(AI_WORKSPACE_DIR), "omniroute"], shell=NPM_SHELL, check=True)
                logger.info(f"  -> omniroute installed successfully in {AI_WORKSPACE_DIR}.")
                return True
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                logger.error(f"  -> Failed to install omniroute: {e}")
                return False
        else:  # existing behavior
            # Ask if user wants to install
            if prompt_choice("Do you want to install omniroute? (y/n): ", default='n'):
                logger.info("Installing omniroute...")
                try:
                    subprocess.run([get_npm_exe(), "install", "--prefix", str(AI_WORKSPACE_DIR), "omniroute"], shell=NPM_SHELL, check=True)
                    logger.info(f"  -> omniroute installed successfully in {AI_WORKSPACE_DIR}.")
                    return True
                except (subprocess.CalledProcessError, FileNotFoundError) as e:
                    logger.error(f"  -> Failed to install omniroute: {e}")
                    return False
            else:
                logger.info(f"  -> Skipping omniroute installation.")
                return False


