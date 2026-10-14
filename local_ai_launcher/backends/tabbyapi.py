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

def resolve_exl3_model_folder(active_model: str) -> Optional[Path]:
    """
    Resolves an EXL3 model name or path into its physical directory under models/exl3/,
    properly discovering nested author directories like 'turboderp/Qwen3.8-27B-exl3-SC-3.0bpw-H4'.
    """
    if not active_model:
        return None
    exl3_dir = AI_WORKSPACE_DIR / "models" / "exl3"
    if not exl3_dir.exists():
        return None

    clean_name = str(active_model).replace('\\', '/').strip()
    if clean_name.lower().startswith("exl3/"):
        clean_name = clean_name[5:]

    # 1. Direct path check
    direct_path = exl3_dir / clean_name
    if direct_path.is_dir() and any(direct_path.glob("*.safetensors")):
        return direct_path

    clean_lower = clean_name.lower()
    clean_base = clean_lower.split('/')[-1]

    # 2. Search all scanned model directories
    candidate_dirs = scan_model_directories(exl3_dir)
    # Pass 1: exact matching on relative path or folder name
    for folder in candidate_dirs:
        rel_name = folder.relative_to(exl3_dir).as_posix().lower()
        fn_lower = folder.name.lower()
        if clean_lower in (fn_lower, rel_name) or clean_base == fn_lower:
            if any(folder.glob("*.safetensors")):
                return folder

    # Pass 2: prefix or substring matching
    for folder in candidate_dirs:
        rel_name = folder.relative_to(exl3_dir).as_posix().lower()
        fn_lower = folder.name.lower()
        if fn_lower.startswith(clean_base) or clean_base.startswith(fn_lower) or clean_base in fn_lower:
            if any(folder.glob("*.safetensors")):
                return folder

    return None


def setup_tabbyapi_config(active_model: str = None):
    """Ensure TabbyAPI global config.yml exists with inline_model_loading enabled and injects per-model sampler configs."""

    tabby_dir = AI_WORKSPACE_DIR / "tabbyapi"
    config_yml = tabby_dir / "config.yml"
    
    if not config_yml.exists():
        tabby_dir.mkdir(parents=True, exist_ok=True)
        model_dir_str = str(AI_WORKSPACE_DIR / 'models' / 'exl3').replace('\\', '/')
        default_config = f"""model:
  model_dir: "{model_dir_str}"
  inline_model_loading: true
network:
  host: 127.0.0.1
  port: {LLM_BACKEND_PORT}
  disable_auth: true
"""
        config_yml.write_text(default_config, encoding='utf-8')
        logger.info(f"  -> Created default TabbyAPI config at {config_yml}")
    
    try:
        import yaml
        with open(config_yml, 'r', encoding='utf-8') as f:
            cfg_data = yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning(f"Notice: Failed to read TabbyAPI config: {e}")
        cfg_data = {}

    if not isinstance(cfg_data, dict):
        cfg_data = {}

    changed = False
    if "network" not in cfg_data or not isinstance(cfg_data["network"], dict):
        cfg_data["network"] = {}
    if not cfg_data["network"].get("disable_auth"):
        cfg_data["network"]["disable_auth"] = True
        changed = True

    if "model" not in cfg_data or not isinstance(cfg_data["model"], dict):
        cfg_data["model"] = {}
    if not cfg_data["model"].get("inline_model_loading"):
        cfg_data["model"]["inline_model_loading"] = True
        changed = True

    if active_model:
        model_folder = resolve_exl3_model_folder(active_model) or (AI_WORKSPACE_DIR / "models" / "exl3" / active_model)
        
        # Round max_seq_len and cache_size to nearest 256 in active model's tabby_config.yml
        active_conf = model_folder / "tabby_config.yml"
        if not active_conf.exists() and model_folder.parent.exists():
            parent_conf = model_folder.parent / "tabby_config.yml"
            if parent_conf.exists():
                active_conf = parent_conf

        if active_conf.exists():
            try:
                with open(active_conf, 'r', encoding='utf-8') as f:
                    yml_data = yaml.safe_load(f)
                if yml_data and "model" in yml_data:
                    model_changed = False
                    for key in ["max_seq_len", "cache_size"]:
                        if key in yml_data["model"]:
                            try:
                                val = max(256, int(yml_data["model"][key]))
                            except (ValueError, TypeError):
                                val = 2048
                            remainder = val % 256
                            if remainder != 0:
                                rounded = val - remainder
                                if remainder >= 128:
                                    rounded += 256
                                yml_data["model"][key] = rounded
                                model_changed = True
                                logger.info(f"  -> Rounded {key} from {val} to {rounded} in {active_model} (must be multiple of 256)")
                            elif val != yml_data["model"][key]:
                                yml_data["model"][key] = val
                                model_changed = True
                    if model_changed:
                        with open(active_conf, 'w', encoding='utf-8') as f:
                            yaml.safe_dump(yml_data, f, sort_keys=False)
            except Exception as e:
                logger.warning(f"  -> Failed to apply 256-rounding to {active_model} tabby_config.yml: {e}")

        sampler_yml = model_folder / "sampler_config.yml"
        if not sampler_yml.exists() and model_folder.parent.exists():
            parent_sampler = model_folder.parent / "sampler_config.yml"
            if parent_sampler.exists():
                sampler_yml = parent_sampler
        
        if sampler_yml.exists():
            overrides_dir = tabby_dir / "sampler_overrides"
            overrides_dir.mkdir(parents=True, exist_ok=True)
            target_preset = overrides_dir / "active_model_preset.yml"
            shutil.copy2(sampler_yml, target_preset)
            logger.info(f"  -> Copied sampler_config.yml from {sampler_yml} to TabbyAPI overrides.")
            
            if "sampling" not in cfg_data or not isinstance(cfg_data["sampling"], dict):
                cfg_data["sampling"] = {}
            if cfg_data["sampling"].get("override_preset") != "active_model_preset":
                cfg_data["sampling"]["override_preset"] = "active_model_preset"
                changed = True
                logger.info(f"  -> TabbyAPI override_preset set to active_model_preset for {active_model}.")
        else:
            if "sampling" in cfg_data and isinstance(cfg_data["sampling"], dict) and "override_preset" in cfg_data["sampling"]:
                if cfg_data["sampling"]["override_preset"] is not None:
                    cfg_data["sampling"]["override_preset"] = None
                    changed = True
                    logger.info(f"  -> Cleared TabbyAPI override_preset (no sampler_config.yml found in {active_model}).")
            logger.info(
                f"  -> [Notice] No 'sampler_config.yml' found in {model_folder}.\n"
                f"     You can place a 'sampler_config.yml' there to automatically configure sampling parameters without reconfiguring harnesses.\n"
                f"     Supported options per TabbyAPI & OpenAI API sampler specs:\n"
                f"       - Common: temperature (e.g. 0.7), top_p (e.g. 0.95), top_k (e.g. 50), max_tokens\n"
                f"       - Penalties: repetition_penalty (e.g. 1.05), frequency_penalty (e.g. 0.0), presence_penalty (e.g. 0.0)\n"
                f"       - Advanced: min_p (e.g. 0.05), top_a (e.g. 0.0), typical_p (e.g. 1.0), tfs (Tail-Free: 1.0)\n"
                f"       - DRY Sampler: dry_multiplier (e.g. 0.8), dry_base (e.g. 1.75), dry_allowed_length (e.g. 2)"
            )
                
    if changed:
        try:
            with open(config_yml, 'w', encoding='utf-8') as f:
                yaml.safe_dump(cfg_data, f, sort_keys=False)
            logger.info(f"  -> Updated TabbyAPI config at {config_yml}")
        except Exception as e:
            logger.warning(f"Notice: Failed to save TabbyAPI config: {e}")


def check_tabbyapi():
    """Check if TabbyAPI is installed and provide options to install or update."""
    logger.info("\nChecking for TabbyAPI...")
    tabby_dir = AI_WORKSPACE_DIR / "tabbyapi"
    tabby_main = tabby_dir / "main.py"
    is_installed = tabby_main.exists()
    
    if is_installed:
        logger.info("  -> TabbyAPI is already installed.")

        # Verify TabbyAPI environment health (self-healing dependency check)
        venv_python = str(tabby_dir / "venv" / "Scripts" / "python.exe") if sys.platform == 'win32' else str(tabby_dir / "venv" / "bin" / "python")
        if Path(venv_python).exists():
            try:
                chk = subprocess.run([venv_python, "-c", "import backends.exllamav3.sampler"], cwd=str(tabby_dir), capture_output=True, text=True, timeout=15)
                if chk.returncode != 0:
                    logger.warning(f"  -> TabbyAPI health probe reported dependency issue: {chk.stderr.strip()}. Auto-repairing dependencies...")
                    subprocess.run(["uv", "pip", "install", "-e", ".[cu12]", "--python", venv_python], cwd=str(tabby_dir), timeout=300, check=False)
            except Exception as e:
                logger.debug(f"  -> TabbyAPI health probe notice: {e}")

        update_behavior = config.tabbyapi_update_behavior if config.tabbyapi_update_behavior != 'existing' else 'existing'
        update_frequency_hrs = getattr(config, 'engine_update_frequency_hrs', {}).get(
            'tabbyapi', getattr(config, 'update_frequency_hrs', 168)
        )
        force = getattr(config, 'force_update', False)

        from local_ai_launcher.update_frequency import should_update_component, record_successful_update
        do_update = should_update_component(
            component_name="tabbyapi",
            category="engine",
            behavior=update_behavior,
            frequency=update_frequency_hrs,
            force=force,
            prompt_if_existing=True,
        )

        if do_update:
            logger.info("Updating TabbyAPI...")
            try:
                res = subprocess.run(["git", "pull", "--autostash"], cwd=str(tabby_dir), timeout=300, check=True, capture_output=True, text=True)
                sys.stdout.write(res.stdout)
                if res.stderr: sys.stderr.write(res.stderr)
                if "Already up to date." not in res.stdout:
                    venv_python = str(tabby_dir / "venv" / "Scripts" / "python.exe") if sys.platform == 'win32' else str(tabby_dir / "venv" / "bin" / "python")
                    subprocess.run(["uv", "pip", "install", "-U", ".", "--python", venv_python], timeout=300, cwd=str(tabby_dir), check=True)
                    if sys.platform == 'win32':
                        update_script = tabby_dir / "update_scripts" / "update_deps.bat"
                        if update_script.exists():
                            logger.info("  -> Running update_deps.bat...")
                            subprocess.run([str(update_script)], cwd=str(tabby_dir), stdin=subprocess.DEVNULL, check=False)
                    else:
                        update_script = tabby_dir / "update_scripts" / "update_deps.sh"
                        if update_script.exists():
                            logger.info("  -> Running update_deps.sh...")
                            subprocess.run(["bash", str(update_script)], cwd=str(tabby_dir), stdin=subprocess.DEVNULL, check=False)
                record_successful_update("engine:tabbyapi")
                return True
            except Exception as e:
                logger.warning(f"  -> Failed to update TabbyAPI ({e}). Falling back to existing installed version.")
                return True
        return True
    else:
        logger.info("  -> TabbyAPI is not installed.")
        install_behavior = config.tabbyapi_install_behavior if config.tabbyapi_install_behavior != 'existing' else 'existing'
        if install_behavior == 'never_install':
            logger.info("  -> Skipping TabbyAPI installation as per configuration.")
            return False
        
        do_install = install_behavior == 'always_install'
        if not do_install:
            do_install = prompt_choice("Do you want to install TabbyAPI? (y/n): ", default='n')
            
        if do_install:
            logger.info("Installing TabbyAPI...")
            try:
                subprocess.run(["git", "clone", "https://github.com/theroyallab/tabbyAPI.git", "tabbyapi"], cwd=str(AI_WORKSPACE_DIR), check=True)
                subprocess.run(["uv", "venv", "--python", config.python_version, "venv"], cwd=str(tabby_dir), check=True)
                venv_python = str(tabby_dir / "venv" / "Scripts" / "python.exe") if sys.platform == 'win32' else str(tabby_dir / "venv" / "bin" / "python")
                subprocess.run(["uv", "pip", "install", ".", "--python", venv_python], timeout=300, cwd=str(tabby_dir), check=True)
                if sys.platform == 'win32':
                    update_script = tabby_dir / "update_scripts" / "update_deps.bat"
                    if update_script.exists():
                        logger.info("  -> Running update_deps.bat to install extra dependencies...")
                        subprocess.run([str(update_script)], cwd=str(tabby_dir), stdin=subprocess.DEVNULL, check=False)
                else:
                    update_script = tabby_dir / "update_scripts" / "update_deps.sh"
                    if update_script.exists():
                        logger.info("  -> Running update_deps.sh to install extra dependencies...")
                        subprocess.run(["bash", str(update_script)], cwd=str(tabby_dir), stdin=subprocess.DEVNULL, check=False)
                logger.info(f"  -> TabbyAPI installed successfully in {tabby_dir}.")
                return True
            except Exception as e:
                logger.error(f"  -> Failed to install TabbyAPI: {e}")
                return False
        else:
            return False


