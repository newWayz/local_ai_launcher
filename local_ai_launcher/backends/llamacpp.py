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

FORK_INFO_FILE = os.path.join(INSTALL_DIR_LLAMA_CPP, ".fork_info")

def get_installed_fork_info():
    """Reads installed fork and version metadata from .fork_info if present."""
    if not os.path.exists(FORK_INFO_FILE):
        return None, None
    try:
        with open(FORK_INFO_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("fork"), data.get("version")
    except Exception:
        return None, None

def get_local_llama_cpp_version():
    """Checks if llama-server.exe exists and extracts its version build tag."""
    if not os.path.exists(EXE_PATH_LLAMA_CPP):
        return None
    
    try:
        # Runs 'llama-server.exe --version' and captures the text output
        result = subprocess.run(
            [EXE_PATH_LLAMA_CPP, "--version"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True, 
            encoding='utf-8',
            errors='replace',
            check=True,
            timeout=15
        )
        # Combines outputs (version text formatting can vary depending on stdout/stderr output)
        output = result.stdout + result.stderr
        
        # Looks for the build number (e.g., 'b3152' or 'version 3152')
        match = re.search(r'\b(b\d+)\b', output)
        if match:
            return match.group(1)
        
        # Fallback regex if it just outputs numbers
        match_alt = re.search(r'\b(\d+)\b', output)
        if match_alt:
            return f"b{match_alt.group(1)}"
            
        return "unknown_version"
    except Exception:
        return "corrupt_or_unresponsive"


def check_llamacpp():
    if sys.platform != 'win32':
        logger.info("Status: Automatic llama.cpp updates are only supported on Windows. Skipping.")
        return True

    target_fork = getattr(config, 'llamacpp_fork', 'atomic').lower().strip()
    if 'blackwell' in target_fork or 'qwen38' in target_fork:
        from local_ai_launcher.backends.blackwell_builder import check_blackwell_llamacpp
        return check_blackwell_llamacpp()

    is_atomic = 'atomic' in target_fork or 'turbo' in target_fork
    fork_name = 'atomic' if is_atomic else 'official'
    repo_name = "AtomicBot-ai/atomic-llama-cpp-turboquant" if is_atomic else "ggml-org/llama.cpp"

    # 1. Check local file status
    local_version = get_local_llama_cpp_version()
    installed_fork, stored_version = get_installed_fork_info()
    if local_version is not None and installed_fork is None:
        installed_fork = "official"

    if local_version is None:
        logger.info(f"Status: llama-server.exe is NOT installed. Installing ({fork_name} fork)")
        if config.llamacpp_install_behavior == 'never_install':
            logger.info("  -> Skipping llama.cpp installation as per configuration.")
            return False
        elif config.llamacpp_install_behavior == 'existing':
            if not prompt_choice(f"Do you want to install llama.cpp ({fork_name})? (y/n): ", default='y'):
                logger.info("  -> Skipping llama.cpp installation.")
                return False
    elif local_version in ["unknown_version", "corrupt_or_unresponsive"]:
        fatal_error("Status: llama-server.exe is present but version could not be verified.")
    elif installed_fork != fork_name:
        logger.info(f"Status: Found local llama-server.exe ({installed_fork}, Version: {stored_version or local_version}), but configured fork is '{fork_name}'.")
        if config.llamacpp_install_behavior == 'never_install':
            logger.info(f"  -> Skipping switch to {fork_name} as per configuration.")
            return True
        elif config.llamacpp_install_behavior == 'existing':
            if not prompt_choice(f"Switch llama-server.exe to '{fork_name}' fork (CUDA 13.x)? (y to switch): ", default='y'):
                logger.info(f"  -> Keeping existing {installed_fork} version.")
                return True
    else:  # installed_fork == fork_name and present
        logger.info(f"Status: Found local llama-server.exe ({fork_name}, Version: {stored_version or local_version})")
        update_behavior = config.llamacpp_update_behavior if config.llamacpp_update_behavior != 'existing' else 'existing'
        update_frequency_hrs = getattr(config, 'engine_update_frequency_hrs', {}).get(
            'llamacpp', getattr(config, 'update_frequency_hrs', 168)
        )
        force = getattr(config, 'force_update', False)

        from local_ai_launcher.update_frequency import should_update_component, record_successful_update
        do_update = should_update_component(
            component_name="llamacpp",
            category="engine",
            behavior=update_behavior,
            frequency=update_frequency_hrs,
            force=force,
            prompt_if_existing=True,
        )
        if not do_update:
            return True

        logger.info("Updating llama.cpp...")

    logger.info(f"Checking GitHub for latest release from {repo_name}...")
    
    # 2. Fetch release metadata from GitHub
    api_url = f"https://api.github.com/repos/{repo_name}/releases?per_page=5"
    try:
        req = urllib.request.Request(api_url, headers={'User-Agent': 'LlamaCppUpdater'})
        with urllib.request.urlopen(req, timeout=30) as response:
            releases_data = json.loads(response.read().decode('utf-8'))
    except Exception as e:
        logger.error(f"Error fetching release data from GitHub: {e}")
        if local_version and local_version not in ["unknown_version", "corrupt_or_unresponsive"]:
            logger.info("Falling back to existing local version.")
            return True
        fatal_error()

    # Pick the latest release that contains binary archives
    release_data = None
    if isinstance(releases_data, list):
        for r in releases_data:
            assets = r.get("assets", [])
            if any('win' in a.get('name', '').lower() and a.get('name', '').endswith('.zip') for a in assets):
                release_data = r
                break
    elif isinstance(releases_data, dict):
        release_data = releases_data

    if not release_data:
        logger.error(f"No release with Windows binaries found on {repo_name}.")
        if local_version:
            return True
        fatal_error()

    remote_version = release_data.get("tag_name", "unknown")
    logger.info(f"Latest online version ({fork_name}): {remote_version}")

    # 3. Compare versions if already on same fork
    effective_local_version = stored_version if installed_fork == fork_name else None
    if effective_local_version and (effective_local_version == remote_version or remote_version.startswith(effective_local_version)):
        logger.info(f"? llama-server.exe ({fork_name}) is already up to date!")
        return True
        
    logger.info(f"Proceeding with installation/update to {remote_version} ({fork_name})...")

    # 4. Match the correct assets based on hardware choice (prioritize CUDA 13.x)
    assets = release_data.get("assets", [])
    main_matched_asset = None
    cudart_matched_asset = None
    
    if config.use_cuda:
        # Prioritize CUDA 13.x over CUDA 12.x
        cuda_priority = ["13.3", "13.1", "13.4", "13.2", "13.0", "13", "12.4", "12.2", "12", ""]
        for ver in cuda_priority:
            for asset in assets:
                name = asset.get("name", "")
                name_lower = name.lower()
                if not name_lower.endswith(".zip") or "arm64" in name_lower or not ("x64" in name_lower or "win64" in name_lower or "amd64" in name_lower):
                    continue
                if is_atomic:
                    if "win" in name_lower and "cuda" in name_lower and (f"cuda-{ver}" in name_lower if ver else True):
                        main_matched_asset = asset
                        break
                else:
                    if name_lower.startswith("llama-") and "win-cuda-" in name_lower and (f"cuda-{ver}" in name_lower if ver else True):
                        main_matched_asset = asset
                        ver_match = re.search(r'bin-win-cuda-([\d\.]+)', name)
                        if ver_match:
                            cudart_keyword = f"cudart-llama-bin-win-cuda-{ver_match.group(1)}"
                            cudart_matched_asset = next((a for a in assets if cudart_keyword in a.get("name", "") and a.get("name", "").endswith("-x64.zip")), None)
                        break
            if main_matched_asset:
                break
    else:
        # CPU path
        for asset in assets:
            name_lower = asset.get("name", "").lower()
            if not name_lower.endswith(".zip") or "arm64" in name_lower:
                continue
            if is_atomic and "win" in name_lower and "cpu" in name_lower:
                main_matched_asset = asset
                break
            elif not is_atomic and name_lower.startswith("llama-") and "win-cpu" in name_lower:
                main_matched_asset = asset
                break

    if not main_matched_asset:
        target_keyword = "CUDA 13.x/12.x" if config.use_cuda else "CPU"
        logger.info(f"Could not find a matching Windows archive ({target_keyword}) for version {remote_version}.")
        if local_version and local_version not in ["unknown_version", "corrupt_or_unresponsive"]:
            logger.info("Falling back to existing local version.")
            return True
        fatal_error()
        
    def flatten_extracted_dir(target_dir):
        """If llama-server.exe was placed in a subdirectory (e.g. build/bin), move contents to target_dir root."""
        exe_file = os.path.join(target_dir, "llama-server.exe")
        if not os.path.exists(exe_file):
            for root, dirs, files in os.walk(target_dir):
                if "llama-server.exe" in files:
                    for item in os.listdir(root):
                        src = os.path.join(root, item)
                        dst = os.path.join(target_dir, item)
                        try:
                            if os.path.exists(dst):
                                if os.path.isdir(dst):
                                    shutil.rmtree(dst, ignore_errors=True)
                                else:
                                    os.remove(dst)
                            shutil.move(src, dst)
                        except PermissionError as pe:
                            logger.error(f"Cannot overwrite {dst} (file locked by running process?): {pe}")
                            raise
                    break
            for root, dirs, files in os.walk(target_dir, topdown=False):
                if root != target_dir and not os.listdir(root):
                    try:
                        os.rmdir(root)
                    except Exception:
                        pass

    def download_and_extract(asset):
        download_url = asset["browser_download_url"]
        logger.info(f"Downloading {asset['name']}...")
        tmp_path = None
        try:
            req = urllib.request.Request(download_url, headers={'User-Agent': 'LlamaCppUpdater'})
            with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_file:
                tmp_path = Path(tmp_file.name)
                with urllib.request.urlopen(req, timeout=180) as response:
                    shutil.copyfileobj(response, tmp_file)
            
            # Ensure target folder exists
            if not os.path.exists(INSTALL_DIR_LLAMA_CPP):
                os.makedirs(INSTALL_DIR_LLAMA_CPP)
                
            logger.info(f"Extracting files to {INSTALL_DIR_LLAMA_CPP}...")
            with zipfile.ZipFile(tmp_path) as z:
                z.extractall(INSTALL_DIR_LLAMA_CPP)
            flatten_extracted_dir(INSTALL_DIR_LLAMA_CPP)
            return True
        except Exception as e:
            logger.error(f"Failed to download or extract {asset['name']}: {e}")
            return False
        finally:
            if tmp_path and tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass

    # Clean old DLLs and executables before extracting to avoid mixing DLLs from different CUDA versions or forks
    if os.path.exists(INSTALL_DIR_LLAMA_CPP):
        locked_files = []
        for fname in os.listdir(INSTALL_DIR_LLAMA_CPP):
            if fname.lower().endswith(('.dll', '.exe', '.pdb')) or fname == '.fork_info':
                fpath = os.path.join(INSTALL_DIR_LLAMA_CPP, fname)
                try:
                    os.remove(fpath)
                except PermissionError:
                    locked_files.append(fname)
                except Exception:
                    pass
        if locked_files:
            logger.error(f"Cannot safely update llama.cpp: The following files are locked by a running process: {', '.join(locked_files[:5])}. Please stop llama-server before updating.")
            if local_version and local_version not in ["unknown_version", "corrupt_or_unresponsive"]:
                logger.info("Falling back to existing local version.")
                return True
            return False
        for sdir in ["build"]:
            sdir_path = os.path.join(INSTALL_DIR_LLAMA_CPP, sdir)
            if os.path.exists(sdir_path) and os.path.isdir(sdir_path):
                try:
                    shutil.rmtree(sdir_path)
                except Exception:
                    pass

    # Download main binary asset
    if not download_and_extract(main_matched_asset):
        if local_version and local_version not in ["unknown_version", "corrupt_or_unresponsive"]:
            logger.info("Falling back to existing local version.")
            return True
        fatal_error()

    # Download companion CUDA runtime DLLs if applicable
    if config.use_cuda and cudart_matched_asset:
        logger.info(f"Downloading required CUDA runtime DLL dependency ({cudart_matched_asset['name']})...")
        if not download_and_extract(cudart_matched_asset):
            logger.error("Warning: Failed to extract CUDA runtime DLLs. If they aren't installed globally, llama-server may fail to launch.")

    # Record installed fork metadata
    try:
        with open(FORK_INFO_FILE, "w", encoding="utf-8") as f:
            json.dump({
                "fork": fork_name,
                "version": remote_version,
                "asset": main_matched_asset.get("name"),
                "date": datetime.now().isoformat()
            }, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to write .fork_info: {e}")

    try:
        from local_ai_launcher.update_frequency import record_successful_update
        record_successful_update("engine:llamacpp")
    except Exception:
        pass

    logger.info(f"Successfully updated llama.cpp ({fork_name}) to {remote_version}!")
    return True


