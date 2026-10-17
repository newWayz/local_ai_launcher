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

def safe_extract_zip(zf: zipfile.ZipFile, target_dir: Path) -> None:
    """Extract zip file securely preventing zip-slip path traversal."""
    target_path = Path(target_dir).resolve()
    target_norm = Path(os.path.normcase(str(target_path)))
    for member in zf.infolist():
        member_path = (target_path / member.filename).resolve()
        member_norm = Path(os.path.normcase(str(member_path)))
        if not member_norm.is_relative_to(target_norm):
            raise RuntimeError(f"Zip slip attempt detected in archive: {member.filename}")
    zf.extractall(target_path)

_scripts_or_bin = "Scripts" if sys.platform == "win32" else "bin"
WORKSPACE_BIN_DIRS = [
    str(AI_WORKSPACE_DIR / "uv-bin"),
    str(AI_WORKSPACE_DIR / "bin"),
    str(AI_WORKSPACE_DIR / "node"),
    str(AI_WORKSPACE_DIR / "node_modules" / ".bin"),
    str(AI_WORKSPACE_DIR / ".venv" / _scripts_or_bin),
    str(config.hermes_dir / "bin"),
    str(config.hermes_dir / "hermes-agent" / "venv" / _scripts_or_bin),
    str(AI_WORKSPACE_DIR / "llama-cpp"),
    str(AI_WORKSPACE_DIR / "go" / "bin"),
    str(AI_WORKSPACE_DIR / "github" / "bin"),
    str(AI_WORKSPACE_DIR / ".herdr" / "bin"),
    str(AI_WORKSPACE_DIR / "tools" / "herdr"),
]


def ensure_directory_junction(link_path: Path, target_path: Path) -> bool:
    """Ensures link_path is an NTFS directory junction pointing to target_path on Windows."""
    if sys.platform != 'win32' or os.environ.get("PYTEST_CURRENT_TEST"):
        return True

    import stat
    try:
        target_path = Path(target_path).resolve()
        target_path.mkdir(parents=True, exist_ok=True)
        link_path = Path(link_path)

        if os.path.lexists(link_path):
            st = link_path.lstat()
            is_reparse = bool(st.st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)
            if is_reparse:
                try:
                    if link_path.resolve() == target_path:
                        return True
                except Exception:
                    pass
                try:
                    os.rmdir(link_path)
                except OSError:
                    link_path.unlink(missing_ok=True)
            else:
                if link_path.is_dir():
                    for item in link_path.rglob("*"):
                        if item.is_file():
                            rel = item.relative_to(link_path)
                            dest = target_path / rel
                            if not dest.exists():
                                dest.parent.mkdir(parents=True, exist_ok=True)
                                shutil.copy2(item, dest)
                bak_dir = link_path.parent / f"{link_path.name}_pre_junction_bak"
                if os.path.lexists(bak_dir):
                    try:
                        st_bak = bak_dir.lstat()
                        if bool(st_bak.st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT):
                            os.rmdir(bak_dir)
                        else:
                            shutil.rmtree(bak_dir, ignore_errors=True)
                    except Exception:
                        pass
                try:
                    shutil.move(str(link_path), str(bak_dir))
                except Exception:
                    pass

        link_path.parent.mkdir(parents=True, exist_ok=True)
        cmd_str = f'mklink /J "{link_path}" "{target_path}"'
        res = subprocess.run(
            f'cmd.exe /c "{cmd_str}"',
            capture_output=True,
            text=True,
            check=False,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
        )
        if res.returncode == 0:
            logger.info(f"  -> Verified workspace storage junction: {link_path} -> {target_path}")
            return True
        return False
    except Exception as e:
        logger.warning(f"Could not maintain junction {link_path} -> {target_path}: {e}")
        return False


def heal_workspace_bin_trampolines(bin_dirs: Optional[List[str]] = None) -> List[Path]:
    """
    Scans workspace bin and Scripts directories for broken uv/pip launcher trampolines
    (which fail on Windows with 'error: uv trampoline failed to canonicalize script path').
    Automatically detects and unlinks corrupted trampolines so Windows cleanly falls back
    to companion scripts (.cmd / .bat) or active venv targets without requiring AI or manual intervention.
    """
    if sys.platform != 'win32':
        return []
    if os.environ.get("PYTEST_CURRENT_TEST") and bin_dirs is None:
        # Fast-path skip global directory scan during unit tests unless custom directories were passed
        return []

    if bin_dirs is None:
        bin_dirs = list(WORKSPACE_BIN_DIRS)
        local_appdata = os.environ.get("LOCALAPPDATA")
        if local_appdata:
            bin_dirs.append(str(Path(local_appdata) / "hermes" / "bin"))
        bin_dirs.append(str(Path.home() / ".hermes" / "bin"))

    healed: List[Path] = []
    for d_str in bin_dirs:
        d = Path(d_str)
        if not d.exists() or not d.is_dir():
            continue
        try:
            for exe_file in d.glob("*.exe"):
                # uv trampolines on Windows are small stub executables (< 2MB)
                try:
                    size = exe_file.stat().st_size
                    if size > 2 * 1024 * 1024 or size == 0:
                        continue
                except OSError:
                    continue

                # Fast binary signature filter: avoid spawning subprocesses for non-uv executables.
                # uv trampoline stubs embed 'uv trampoline' and 'canonicalize' in their binary data.
                try:
                    with open(exe_file, "rb") as f:
                        header = f.read(131072)
                        if b"uv trampoline" not in header and b"canonicalize" not in header:
                            continue
                except OSError:
                    continue

                try:
                    res = subprocess.run(
                        [str(exe_file), "--version"],
                        capture_output=True,
                        text=True,
                        encoding='utf-8',
                        errors='replace',
                        timeout=2,
                        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
                    )
                    combined_err = ((res.stderr or "") + (res.stdout or "")).lower()
                    if res.returncode != 0 and ("uv trampoline failed" in combined_err or "failed to canonicalize script path" in combined_err):
                        logger.warning(f"  -> [Self-Heal] Detected broken uv trampoline: {exe_file}. Auto-purging stale binary...")
                        try:
                            exe_file.unlink(missing_ok=True)
                            healed.append(exe_file)
                        except (PermissionError, OSError) as e:
                            logger.debug(f"Could not unlink locked trampoline {exe_file}: {e}")
                except Exception as e:
                    logger.debug(f"Trampoline probe notice for {exe_file}: {e}")
        except Exception as e:
            logger.debug(f"Error scanning trampolines in {d}: {e}")

    return healed


def _setup_workspace_path():
    """Add workspace bin directories to PATH so workspace tools take precedence."""
    workspace_bin_dirs = WORKSPACE_BIN_DIRS
    for d in workspace_bin_dirs:
        if ".venv" not in d:
            os.makedirs(d, exist_ok=True)
    heal_workspace_bin_trampolines(workspace_bin_dirs)
    os.environ["PATH"] = os.path.pathsep.join(workspace_bin_dirs) + os.path.pathsep + os.environ.get("PATH", "")


def setup_windows_environment_variables():
    """If running on Windows, set persistent User environment variables and update PATH for workspace."""
    if sys.platform != 'win32':
        return

    workspace_str = str(AI_WORKSPACE_DIR)
    pi_home_str = str(AI_WORKSPACE_DIR / ".pi")
    hermes_home_str = str(config.hermes_dir)
    gopath_str = str(AI_WORKSPACE_DIR / "go")
    cargo_home_str = str(AI_WORKSPACE_DIR / "cargo")
    uv_cache_str = str(AI_WORKSPACE_DIR / "uv-cache")
    uv_python_dir_str = str(AI_WORKSPACE_DIR / "uv-cache" / "python")
    gemini_dir_str = str(AI_WORKSPACE_DIR / ".gemini")
    gh_config_dir_str = str(AI_WORKSPACE_DIR / ".gh")
    gh_data_dir_str = str(AI_WORKSPACE_DIR / ".gh" / "data")
    hf_home_str = str(AI_WORKSPACE_DIR / "models" / "huggingface")
    
    user_vars = {
        'HOME': workspace_str,
        'HERMES_HOME': hermes_home_str,
        'PI_HOME': pi_home_str,
        'OMP_HOME': str(AI_WORKSPACE_DIR / ".omp"),
        'PI_CODING_AGENT_DIR': str(AI_WORKSPACE_DIR / ".pi" / "agent"),
        'GOPATH': gopath_str,
        'CARGO_HOME': cargo_home_str,
        'UV_CACHE_DIR': uv_cache_str,
        'UV_PYTHON_INSTALL_DIR': uv_python_dir_str,
        'GH_CONFIG_DIR': gh_config_dir_str,
        'GH_DATA_DIR': gh_data_dir_str,
        'HF_HOME': hf_home_str,
        'HF_HUB_CACHE': hf_home_str,
        'AGENT_BROWSER_HOME': str(AI_WORKSPACE_DIR / '.agent-browser'),
        'AGENT_BROWSER_DOWNLOAD_PATH': str(AI_WORKSPACE_DIR / 'downloads'),
        'CODEX_HOME': str(AI_WORKSPACE_DIR / '.codex'),
        'CLAUDE_CONFIG_DIR': str(AI_WORKSPACE_DIR / '.claude'),
        'ORCA_USER_DATA_PATH': str(AI_WORKSPACE_DIR / 'orca' / 'data'),
        'HERDR_HOME': str(AI_WORKSPACE_DIR / '.herdr'),
        'HERDR_CONFIG_PATH': str(AI_WORKSPACE_DIR / '.herdr' / 'config.toml'),
        'HERDR_DATA_PATH': str(AI_WORKSPACE_DIR / '.herdr' / 'data'),
        'GEMINI_DIR': gemini_dir_str,
        'MY_PI_PROJECT_SKILLS': '0',
    }
    
    _setup_workspace_path()

    user_home = Path(os.path.expanduser("~"))
    if user_home.exists():
        ensure_directory_junction(user_home / ".gemini" / "antigravity-cli", AI_WORKSPACE_DIR / ".gemini" / "antigravity-cli")
        ensure_directory_junction(user_home / ".gemini" / "config", AI_WORKSPACE_DIR / ".gemini" / "config")
        ensure_directory_junction(user_home / ".herdr", AI_WORKSPACE_DIR / ".herdr")
        ensure_directory_junction(user_home / ".config" / "herdr", AI_WORKSPACE_DIR / ".herdr")
    appdata_env = os.environ.get("APPDATA")
    if appdata_env:
        ensure_directory_junction(Path(appdata_env) / "herdr", AI_WORKSPACE_DIR / ".herdr")
    
    ps_commands = []
    for var, val in user_vars.items():
        os.environ[var] = val
        escaped_val = val.replace("'", "''")
        ps_commands.append(f"[System.Environment]::SetEnvironmentVariable('{var}', '{escaped_val}', 'User')")
    
    new_bin_dirs = WORKSPACE_BIN_DIRS
    
    ps_path_script = (
        "$oldPath = [System.Environment]::GetEnvironmentVariable('PATH', 'User'); "
        "$paths = if ($oldPath) { $oldPath -split ';' } else { @() }; "
    )
    for bdir in new_bin_dirs:
        escaped_bdir = bdir.replace("'", "''")
        ps_path_script += f"if ($paths -notcontains '{escaped_bdir}') {{ $paths = @('{escaped_bdir}') + $paths }}; "
    
    ps_path_script += "$newPath = ($paths | Where-Object { $_ -ne '' }) -join ';'; [System.Environment]::SetEnvironmentVariable('PATH', $newPath, 'User');"
    ps_commands.append(ps_path_script)
    
    # Modernize legacy Windows Console globally for this user (ANSI, QuickEdit, Ctrl-keys)
    ps_commands.append("[void](New-ItemProperty -Path 'HKCU:\\Console' -Name 'VirtualTerminalLevel' -Value 1 -PropertyType DWORD -Force -ErrorAction SilentlyContinue)")
    ps_commands.append("[void](New-ItemProperty -Path 'HKCU:\\Console' -Name 'CtrlKeyShortcutsEnabled' -Value 1 -PropertyType DWORD -Force -ErrorAction SilentlyContinue)")
    ps_commands.append("[void](New-ItemProperty -Path 'HKCU:\\Console' -Name 'QuickEdit' -Value 1 -PropertyType DWORD -Force -ErrorAction SilentlyContinue)")
    
    full_script = "; ".join(ps_commands)
    try:
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = 0
        subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", full_script],
            creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP,
            startupinfo=startupinfo,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False
        )
    except Exception as e:
        logger.warning(f"Warning: Could not update Windows persistent User environment variables: {e}")


def setup_windows_user_account():
    """Create the sandbox Windows normal user account if it doesn't exist, and update 1_run_agent.cmd to use it."""
    if sys.platform != 'win32':
        return
        
    if not getattr(config, 'create_ai_bot_user', False):
        return
        
    ai_bot_exists = False
    sandbox_user = getattr(config, 'ai_bot_username', 'ai_bot')
    sandbox_pass = getattr(config, 'ai_bot_password', 'ai_bot')
    try:
        # Check if the user already exists
        res = subprocess.run(["net", "user", sandbox_user], capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
        if res.returncode == 0:
            ai_bot_exists = True
        elif getattr(config, 'create_ai_bot_user', False):
            logger.info(f"Creating Windows normal user account '{sandbox_user}'...")
            # Attempt to create the user. Note: This requires the script to be run with Administrator privileges.
            res_add = subprocess.run(["net", "user", sandbox_user, "/add"], capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
            if res_add.returncode == 0:
                logger.info(f"  -> Successfully created '{sandbox_user}' user account.")
                ai_bot_exists = True
                
                logger.info(f"Setting password for newly created '{sandbox_user}' account...")
                res_pwd = subprocess.run(["net", "user", sandbox_user, sandbox_pass], capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
                if res_pwd.returncode != 0:
                    logger.warning(f"  -> Failed to set password for '{sandbox_user}' (Administrator privileges may be required). Error: {res_pwd.stderr.strip() or res_pwd.stdout.strip()}")
            else:
                logger.warning(f"  -> Failed to create '{sandbox_user}' user account (Administrator privileges are likely required). Error: {res_add.stderr.strip() or res_add.stdout.strip()}")
                
        if ai_bot_exists:
            # The previous explicit DENY rule applied to the system and workspace drives caused
            # cmd.exe and subprocess.Popen(shell=True) to crash with Access is Denied due to
            # inherited SYNCHRONIZE and GENERIC_WRITE denials. We will remove it.
            system_drive = os.environ.get('SystemDrive', 'C:') + '\\'
            workspace_drive = AI_WORKSPACE_DIR.anchor
            target_drives = list({system_drive, workspace_drive})  # Use a set to remove duplicates
            
            for drive in target_drives:
                logger.info(f"Ensuring no explicit deny permissions exist for '{sandbox_user}' on {drive}...")
                subprocess.run(
                    ["icacls", drive, "/remove:d", sandbox_user],
                    capture_output=True,
                    creationflags=subprocess.CREATE_NO_WINDOW
                )

            # Explicitly grant sandbox_user full control over the AI_WORKSPACE_DIR
            logger.info(f"Ensuring '{sandbox_user}' has full access to {AI_WORKSPACE_DIR}...")
            # Break inheritance from C:\ so the DENY rule doesn't propagate to newly created files
            subprocess.run(
                ["icacls", str(AI_WORKSPACE_DIR), "/inheritance:d"],
                capture_output=True,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
            # Remove any copied DENY rules for the sandbox user
            subprocess.run(
                ["icacls", str(AI_WORKSPACE_DIR), "/remove:d", sandbox_user, "/T"],
                capture_output=True,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
            subprocess.run(
                ["icacls", str(AI_WORKSPACE_DIR), "/grant", f"{sandbox_user}:(OI)(CI)F", "/T"],
                capture_output=True,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
            
            import getpass
            current_user = getpass.getuser()
            
            if current_user.lower() != sandbox_user.lower():
                uv_cache_dir = AI_WORKSPACE_DIR / "uv-cache"
                if uv_cache_dir.exists():
                    logger.info(f"  -> Wiping '{uv_cache_dir.name}' to prevent permission locks for the new sandboxed user...")
                    subprocess.run(["cmd", "/c", "rmdir", "/s", "/q", str(uv_cache_dir)], creationflags=subprocess.CREATE_NO_WINDOW)

                logger.warning(f"\n*** Relaunching services securely as '{sandbox_user}'... ***\n")
                
                # Relaunch the script using runas
                python_exe = sys.executable
                args = [python_exe] + sys.argv
                cmd_str = subprocess.list2cmdline(args)
                
                subprocess.Popen(["runas", f"/user:{sandbox_user}", "/savecred", cmd_str])
                
                # Exit the original non-sandboxed process
                sys.exit(0)

    except Exception as e:
        logger.warning(f"  -> Exception while checking/creating '{sandbox_user}' user account: {e}")


def _find_npm_executable(node_dir):
    if sys.platform == 'win32':
        if (node_dir / "npm.cmd").exists():
            return str(node_dir / "npm.cmd")
        return shutil.which("npm.cmd") or shutil.which("npm")
    else:
        if (node_dir / "npm").exists():
            return str(node_dir / "npm")
        return shutil.which("npm")


def ensure_node_and_npm():
    """
    Ensure Node.js and NPM are installed in workspace root (AI_WORKSPACE_DIR / 'node').
    Downloads portable Node.js into the root 'node' directory if not available.
    """
    node_dir = AI_WORKSPACE_DIR / "node"
    
    # Ensure node_dir is in PATH if it exists
    if str(node_dir) not in os.environ.get("PATH", ""):
        os.environ["PATH"] = str(node_dir) + os.path.pathsep + os.environ.get("PATH", "")
    
    # Check if npm is available in workspace node dir or PATH
    npm_path = _find_npm_executable(node_dir)

    if not npm_path:
        # If missing, download portable Node.js into AI_WORKSPACE_DIR / 'node'
        logger.info(f"\nNode.js / NPM not found in PATH or workspace. Installing portable Node.js into {node_dir}...")
        node_dir.mkdir(parents=True, exist_ok=True)
        
        if sys.platform == 'win32':
            node_url = "https://nodejs.org/dist/v24.19.0/node-v24.19.0-win-x64.zip"
            zip_path = AI_WORKSPACE_DIR / "node_temp.zip"
            logger.info(f"Downloading Node.js portable zip from {node_url} (this may take a moment)...")
            try:
                req = urllib.request.Request(node_url, headers={'User-Agent': 'HealthCheck'})
                with urllib.request.urlopen(req, timeout=30) as resp, open(zip_path, 'wb') as out_file:
                    shutil.copyfileobj(resp, out_file)
                logger.info("Extracting Node.js zip...")
                temp_extract_dir = AI_WORKSPACE_DIR / "node_extract_temp"
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    safe_extract_zip(zip_ref, temp_extract_dir)
                    
                extracted_subdirs = list(temp_extract_dir.glob("node-*"))
                if extracted_subdirs:
                    source_dir = extracted_subdirs[0]
                    for item in source_dir.iterdir():
                        target_item = node_dir / item.name
                        if item.is_dir():
                            if target_item.exists():
                                shutil.rmtree(target_item)
                            shutil.move(str(item), str(target_item))
                        else:
                            if target_item.exists():
                                target_item.unlink()
                            shutil.move(str(item), str(target_item))
                shutil.rmtree(temp_extract_dir, ignore_errors=True)
                zip_path.unlink(missing_ok=True)
                logger.info(f"  -> Node.js / NPM installed successfully in {node_dir}.")
            except Exception as e:
                logger.error(f"  -> Failed to download/install Node.js: {e}")
                if zip_path.exists():
                    zip_path.unlink(missing_ok=True)
        else:
            logger.info("Please install Node.js / npm using your package manager.")

        # Re-check npm path after installation
        npm_path = _find_npm_executable(node_dir)

        if npm_path:
            logger.info("Updating npm to latest version (npm@latest)...")
            try:
                subprocess.run([npm_path, "install", "-g", "npm@latest"], shell=(sys.platform == 'win32'), check=True)
                logger.info("  -> npm updated to latest version successfully.")
            except Exception as e:
                logger.warning(f"  -> Warning: Could not update npm to latest: {e}")

    # Ensure git-based npm packages clone without restrictions across all sub-installs and allow specific install scripts
    if npm_path:
        try:
            subprocess.run([npm_path, "config", "set", "allow-git", "all"], shell=(sys.platform == 'win32'), capture_output=True)
            subprocess.run([npm_path, "config", "set", "allow-scripts", "@google/genai,protobufjs,koffi,donsetch,pnpm"], shell=(sys.platform == 'win32'), capture_output=True)
        except Exception:
            pass

    return npm_path or "npm"


def get_npm_exe():
    """Lazily ensure Node.js and NPM are installed in workspace root when required by a tool."""
    return ensure_node_and_npm()


NPM_SHELL = (sys.platform == 'win32')


def ensure_bun() -> Optional[str]:
    """Ensure Bun executable is available in workspace or PATH for Bun-based agents (e.g. Oh My Pi)."""
    exe_name = "bun.exe" if sys.platform == 'win32' else "bun"
    node_dir = AI_WORKSPACE_DIR / "node"
    bin_dir = AI_WORKSPACE_DIR / "bin"
    node_modules_bin = AI_WORKSPACE_DIR / "node_modules" / ".bin"

    def _sync_destinations(src: Path) -> str:
        for t_dir in [node_dir, node_modules_bin, bin_dir]:
            if t_dir.exists():
                dest = t_dir / exe_name
                if not dest.exists():
                    try:
                        shutil.copy2(src, dest)
                        if sys.platform != 'win32':
                            os.chmod(dest, 0o755)
                    except Exception:
                        pass
                elif sys.platform != 'win32':
                    try:
                        os.chmod(dest, 0o755)
                    except Exception:
                        pass
        return str(src)

    for candidate in [node_dir / exe_name, bin_dir / exe_name, node_modules_bin / exe_name]:
        if candidate.exists():
            return _sync_destinations(candidate)

    bun_path = shutil.which(exe_name) or shutil.which("bun")
    if bun_path:
        return _sync_destinations(Path(bun_path))

    import platform
    machine = platform.machine().lower()
    bun_url = None
    if sys.platform == 'win32' and machine in ('amd64', 'x86_64'):
        bun_url = "https://github.com/oven-sh/bun/releases/download/bun-v1.4.3/bun-windows-x64.zip"
    elif sys.platform == 'linux':
        if machine in ('amd64', 'x86_64'):
            bun_url = "https://github.com/oven-sh/bun/releases/download/bun-v1.4.3/bun-linux-x64.zip"
        elif machine in ('aarch64', 'arm64'):
            bun_url = "https://github.com/oven-sh/bun/releases/download/bun-v1.4.3/bun-linux-aarch64.zip"
    elif sys.platform == 'darwin':
        if machine in ('aarch64', 'arm64'):
            bun_url = "https://github.com/oven-sh/bun/releases/download/bun-v1.4.3/bun-darwin-aarch64.zip"
        elif machine in ('amd64', 'x86_64'):
            bun_url = "https://github.com/oven-sh/bun/releases/download/bun-v1.4.3/bun-darwin-x64.zip"

    if bun_url:
        logger.info(f"\nBun not found in PATH or workspace. Installing portable Bun for {sys.platform} ({machine})...")
        node_dir.mkdir(parents=True, exist_ok=True)
        zip_path = AI_WORKSPACE_DIR / "bun_temp.zip"
        temp_extract_dir = AI_WORKSPACE_DIR / "bun_extract_temp"
        try:
            req = urllib.request.Request(bun_url, headers={'User-Agent': 'HealthCheck'})
            with urllib.request.urlopen(req, timeout=45) as resp, open(zip_path, 'wb') as out_file:
                shutil.copyfileobj(resp, out_file)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                safe_extract_zip(zip_ref, temp_extract_dir)

            extracted_buns = list(temp_extract_dir.glob(f"**/{exe_name}"))
            if extracted_buns:
                target_bun = node_dir / exe_name
                shutil.copy2(extracted_buns[0], target_bun)
                if sys.platform != 'win32':
                    os.chmod(target_bun, 0o755)
                _sync_destinations(target_bun)
                logger.info(f"  -> Portable Bun installed successfully in {target_bun}.")
                return str(target_bun)
        except Exception as e:
            logger.warning(f"  -> Warning: Failed to download portable Bun: {e}")
        finally:
            shutil.rmtree(temp_extract_dir, ignore_errors=True)
            if zip_path.exists():
                zip_path.unlink(missing_ok=True)

    return None



def ensure_uv():
    """Ensure uv is available in PATH. If missing, attempt to install uv via pip."""
    uv_path = shutil.which("uv")
    if uv_path:
        return uv_path
    
    logger.info("Status: 'uv' not found in PATH. Attempting to install uv via pip...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "uv"], timeout=300, check=True)
        uv_path = shutil.which("uv")
        if uv_path:
            logger.info("  -> 'uv' installed successfully.")
            return uv_path
    except Exception as e:
        logger.info(f"  -> Could not install uv via pip: {e}")
    
    return None


def ensure_uv_python() -> bool:
    """Ensure that the Python version configured in LauncherConfig is installed via uv."""
    uv_exe = ensure_uv()
    if not uv_exe:
        return False
    try:
        logger.info(f"Ensuring Python {config.python_version} (latest) is installed via uv...")
        subprocess.run([uv_exe, "python", "install", config.python_version], check=True)
        return True
    except Exception as e:
        logger.warning(f"Notice: uv could not install Python {config.python_version}: {e}")
        return False


def ensure_hf(venv_python: Optional[Path] = None) -> bool:
    """Ensures Hugging Face library (huggingface_hub, hf_transfer) is installed in the workspace venv using uv."""
    if venv_python is None:
        venv_python = AI_WORKSPACE_DIR / ".venv" / "Scripts" / "python.exe"
    
    if not Path(venv_python).exists():
        logger.warning(f"Workspace venv Python not found at: {venv_python}")
        return False
    
    uv_exe = ensure_uv()
    hf_libs = ["huggingface_hub", "hf_transfer"]
    missing_libs = []
    
    for lib in hf_libs:
        check_res = subprocess.run([str(venv_python), "-c", f"import {lib}"], capture_output=True)
        if check_res.returncode != 0:
            missing_libs.append(lib)
            
    if missing_libs:
        logger.info(f"Status: Hugging Face libraries ({', '.join(missing_libs)}) not found in venv. Installing via uv...")
        try:
            if uv_exe:
                subprocess.run([uv_exe, "pip", "install", "--python", str(venv_python)] + missing_libs, timeout=300, check=True)
            else:
                subprocess.run([str(venv_python), "-m", "pip", "install"] + missing_libs, check=True)
            logger.info(f"  -> Successfully installed Hugging Face libraries ({', '.join(missing_libs)}).")
            return True
        except Exception as e:
            logger.error(f"  -> Error installing Hugging Face libraries: {e}")
            return False
    return True


def setup_workspace_venv():
    """Ensures the workspace venv exists at AI_WORKSPACE_DIR\\.venv, matches config.python_version, and has required libraries."""
    workspace_dir = AI_WORKSPACE_DIR
    venv_dir = workspace_dir / ".venv"
    venv_python = (venv_dir / "Scripts" / "python.exe") if sys.platform == "win32" else (venv_dir / "bin" / "python")
    
    # Get needed libs from active harness adapter(s) and core requirements
    from local_ai_launcher.harnesses import HARNESS_REGISTRY
    needed_libs = [
        "requests", "psutil", "huggingface_hub", "hf_transfer",
        "fastapi", "uvicorn", "httpx", "pillow", "pymupdf", "python-multipart",
        "pyyaml", "grpcio", "protobuf", "jinja2"
    ]
    if getattr(config, 'voice_to_text', 'parakeet_tdt_v3') == 'parakeet_tdt_v3':
        needed_libs.extend(["openvino", "soundfile", "sounddevice", "pyperclip"])
    harnesses = [config.choose_harness] if isinstance(config.choose_harness, str) else config.choose_harness
    for h_name in harnesses:
        if h_name in HARNESS_REGISTRY:
            adapter_cls = HARNESS_REGISTRY[h_name]
            harness_libs = getattr(adapter_cls, 'REQUIRED_LIBS', [])
            needed_libs.extend(harness_libs)
    # Deduplicate while preserving order
    needed_libs = list(dict.fromkeys(needed_libs))
        
    uv_exe = ensure_uv()
    if uv_exe:
        ensure_uv_python()

    # Check if existing venv is healthy and matches config.python_version
    is_venv_valid = False
    if venv_python.exists():
        try:
            res = subprocess.run(
                [str(venv_python), "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if res.returncode == 0:
                installed_ver = res.stdout.strip()
                py_ver_prefix = ".".join(config.python_version.split(".")[:2])
                if installed_ver == py_ver_prefix:
                    is_venv_valid = True
                else:
                    logger.info(f"Existing venv Python version ({installed_ver}) does not match configured version ({config.python_version}). Rebuilding venv...")
        except Exception:
            is_venv_valid = False

    if not is_venv_valid:
        logger.info(f"Status: Workspace venv python at {venv_python} is missing, invalid, or version-mismatched. Creating venv with Python {config.python_version}...")
        if venv_dir.exists():
            shutil.rmtree(venv_dir, ignore_errors=True)
        try:
            if uv_exe:
                subprocess.run([uv_exe, "venv", "--python", config.python_version, str(venv_dir)], check=True)
                logger.info(f"  -> Workspace venv created successfully with uv (Python {config.python_version}).")
                logger.info(f"Installing libraries with uv: {', '.join(needed_libs)}...")
                subprocess.run([uv_exe, "pip", "install", "--python", str(venv_python)] + needed_libs, timeout=300, check=True)
            else:
                logger.info("  -> Falling back to standard Python venv creation...")
                subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
                subprocess.run([str(venv_python), "-m", "pip", "install"] + needed_libs, check=True)
            logger.info("  -> Libraries installed successfully.")
        except Exception as e:
            fatal_error(f"Error setting up workspace venv: {e}")
    else:
        # Venv exists. Check if any needed libraries are missing in the venv
        _IMPORT_NAME_MAP = {
            "python-multipart": "multipart",
            "pymupdf": "fitz",
            "pillow": "PIL",
            "huggingface_hub": "huggingface_hub",
            "hf_transfer": "hf_transfer",
            "openvino": "openvino",
            "soundfile": "soundfile",
            "sounddevice": "sounddevice",
            "pyperclip": "pyperclip",
            "pyyaml": "yaml",
            "grpcio": "grpc",
            "protobuf": "google.protobuf",
            "jinja2": "jinja2",
        }
        missing_libs = []
        for lib in needed_libs:
            import_name = _IMPORT_NAME_MAP.get(lib, lib)
            check_res = subprocess.run([str(venv_python), "-c", f"import {import_name}"], capture_output=True)
            if check_res.returncode != 0:
                missing_libs.append(lib)

        if missing_libs:
            logger.info(f"Installing missing libraries into workspace venv: {', '.join(missing_libs)}...")
            try:
                if uv_exe:
                    subprocess.run([uv_exe, "pip", "install", "--python", str(venv_python)] + missing_libs, timeout=300, check=True)
                else:
                    subprocess.run([str(venv_python), "-m", "pip", "install"] + missing_libs, check=True)
                logger.info("  -> Missing libraries installed successfully.")
            except Exception as e:
                logger.error(f"Error installing missing libraries: {e}")

        # Check if we should update libraries
        update_behaviors = [config.harness_update_behaviors.get(h, 'never_update') for h in harnesses]
        if 'always_update' in update_behaviors:
            behavior_var = 'always_update'
        elif 'existing' in update_behaviors:
            behavior_var = 'existing'
        else:
            behavior_var = 'never_update'
            
        from local_ai_launcher.update_frequency import should_update_component, record_successful_update
        should_update = should_update_component(
            component_name="workspace_libraries",
            category="workspace",
            behavior=behavior_var,
            frequency=getattr(config, 'update_frequency_hrs', 168),
            force=getattr(config, 'force_update', False),
            prompt_if_existing=True,
        )
          
        if should_update:
            logger.info(f"Updating workspace venv libraries: {', '.join(needed_libs)}...")
            try:
                if uv_exe:
                    subprocess.run([uv_exe, "pip", "install", "--python", str(venv_python), "--upgrade"] + needed_libs, timeout=300, check=True)
                else:
                    subprocess.run([str(venv_python), "-m", "pip", "install", "--upgrade"] + needed_libs, check=True)
                logger.info("  -> Libraries updated successfully.")
                record_successful_update("workspace:workspace_libraries")
            except Exception as e:
                logger.error(f"Error updating libraries: {e}")





def setup_workspace_git():
    # Add safe.directory to avoid dubious ownership errors when running as sandboxed user (ai_bot)
    try:
        subprocess.run(["git", "config", "--global", "--add", "safe.directory", str(AI_WORKSPACE_DIR).replace("\\", "/")], capture_output=True)
    except Exception:
        pass

    if not (AI_WORKSPACE_DIR / ".git").exists():
        logger.info("Initializing git repository in workspace...")
        try:
            subprocess.run(["git", "init"], cwd=str(AI_WORKSPACE_DIR), check=True)
            exclude_path = AI_WORKSPACE_DIR / ".git" / "info" / "exclude"
            exclude_path.parent.mkdir(parents=True, exist_ok=True)
            with open(exclude_path, "w") as f:
                f.write("# Ignore everything by default\n")
                f.write("*\n\n")
                f.write("# Allow git to search inside directories\n")
                f.write("!*/\n\n")
                f.write("# Explicitly ignore __pycache__ folders\n")
                f.write("__pycache__/\n\n")
                f.write("# Explicitly exclude secrets/PII from git tracking\n")
                f.write("config/accounts.ini\n")
                f.write(".gh/\n\n")
                f.write("# Explicitly include specific files and directories\n")
                f.write("!config/USER.md\n")
                f.write("!local_ai_launcher/**\n")
                f.write("!local_ai_launcher/freetokens_models.ini\n")
                f.write("!local_ai_launcher/models.ini\n")
                f.write("!localai/**\n")
                f.write("!freetokens_models.ini\n")
                f.write("!models.ini\n")
                f.write("!models/exl3/turboderp/Muse-Glimmer-30B-exl3-3.0bpw/sampler_config.yml\n")
                f.write("!models/exl3/turboderp/Muse-Glimmer-30B-exl3-3.0bpw/tabby_config.yml\n")
                f.write("!models/exl3/turboderp/Qwen3.8-27B-exl3-3.0bpw/sampler_config.yml\n")
                f.write("!models/exl3/turboderp/Qwen3.8-27B-exl3-3.0bpw/tabby_config.yml\n")
        except Exception as e:
            logger.error(f"Failed to initialize git repository: {e}")

    # Wire GitHub if requested
    if getattr(config, "github_sync_mode", "auto") == "auto":
        try:
            remotes = subprocess.check_output(["git", "remote"], cwd=str(AI_WORKSPACE_DIR), text=True).strip()
            logger.info("  -> Attempting to auto-wire the remote GitHub repository.")
            # We import prompt_string here since environment.py likely doesn't have it imported
            from local_ai_launcher.config import prompt_string
            prompt_timeout = 10
            remote_url = prompt_string(f"Enter the GitHub repository URL to wire it (or press Enter to skip) [{prompt_timeout}s timeout]: ", timeout=prompt_timeout)
            if remote_url:
                subprocess.run(["git", "remote", "add", "origin", remote_url], cwd=str(AI_WORKSPACE_DIR), check=True)
                subprocess.run(["git", "branch", "-M", "main"], cwd=str(AI_WORKSPACE_DIR), check=True)
                print(f"Successfully wired GitHub remote 'origin' to {remote_url}\n")
        except Exception as e:
            logger.error(f"Failed to wire GitHub: {e}")


def get_local_github_version(gh_exe: Path) -> Optional[str]:
    """Checks if gh.exe exists in the workspace and extracts its version string."""
    if not gh_exe.exists():
        return None
    try:
        res = subprocess.run(
            [str(gh_exe), "--version"],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=10
        )
        match = re.search(r'gh version ([\d\.]+)', res.stdout)
        if match:
            return f"v{match.group(1)}"
        return "unknown"
    except Exception:
        return "corrupt"


def get_local_git_version(git_cmd: Optional[str] = None) -> Optional[str]:
    """Returns the installed Git version string, or None if git is not found."""
    target_git = git_cmd or shutil.which("git")
    if not target_git:
        return None
    try:
        res = subprocess.run(
            [str(target_git), "--version"],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=10,
        )
        match = re.search(r'git version ([\w\.\-]+)', res.stdout)
        if match:
            return match.group(1)
        return "unknown"
    except Exception:
        return "corrupt"


def get_remote_git_version() -> Optional[str]:
    """Check the latest available Git release version for the current platform."""
    if sys.platform == "win32":
        try:
            req = urllib.request.Request(
                "https://gitforwindows.org/latest-tag.txt",
                headers={"User-Agent": "GitUpdater"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                tag = resp.read().decode("utf-8").strip()
                return tag.lstrip("v")
        except Exception as e:
            logger.debug(f"Could not fetch Git for Windows latest tag: {e}")
    return None


def check_git() -> bool:
    """Check local Git installation and update Git if configured and due."""
    local_version = get_local_git_version()
    if local_version is None:
        logger.warning("\nGit is not detected on the system PATH.")
        return False

    update_behavior = getattr(config, "git_update_behavior", "never_update")
    auto_update = getattr(config, "git_auto_update", False)
    if auto_update and update_behavior == "never_update":
        update_behavior = "always_update"

    from local_ai_launcher.update_frequency import should_update_component, record_successful_update
    do_update = should_update_component(
        component_name="git",
        category="cli",
        behavior=update_behavior,
        frequency=getattr(config, "git_update_frequency_hrs", getattr(config, "update_frequency_hrs", 168)),
        force=getattr(config, "force_update", False),
        prompt_if_existing=True,
    )
    if not do_update:
        return True

    logger.info(f"\nStatus: Found local Git (Version: {local_version})")
    logger.info("Checking for the latest Git release...")
    remote_version = get_remote_git_version()

    if remote_version and local_version == remote_version:
        logger.info(f"  -> Git is already up to date ({local_version}).")
        record_successful_update("cli:git")
        return True

    if sys.platform == "win32":
        target_info = f" -> {remote_version}" if remote_version else ""
        logger.info(f"Updating Git for Windows ({local_version}{target_info})...")
        try:
            res = subprocess.run(
                ["git", "update-git-for-windows", "-y"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=300,
            )
            if res.returncode in (0, 2):
                logger.info(f"  -> Git update successfully initiated ({res.stdout.strip() or 'installer started'}).")
                record_successful_update("cli:git")
                return True
            else:
                logger.warning(f"  -> Git update returned status {res.returncode}: {res.stderr.strip() or res.stdout.strip()}")
                return False
        except Exception as e:
            logger.error(f"Failed to update Git for Windows: {e}")
            return False
    else:
        logger.info(f"  -> Git {local_version} is managed by system package manager on this platform.")
        record_successful_update("cli:git")
        return True


ensure_git = check_git


def check_github() -> bool:
    """Ensure GitHub CLI (gh) is installed in the workspace and configured for workspace-local storage."""
    github_dir = getattr(config, 'github_dir', AI_WORKSPACE_DIR / "github")
    gh_bin_dir = github_dir / "bin"
    gh_exe = gh_bin_dir / "gh.exe"

    gh_home = AI_WORKSPACE_DIR / ".gh"
    gh_home.mkdir(parents=True, exist_ok=True)
    os.environ["GH_CONFIG_DIR"] = str(gh_home)
    os.environ["GH_DATA_DIR"] = str(gh_home / "data")
    if str(gh_bin_dir) not in os.environ.get("PATH", ""):
        os.environ["PATH"] = str(gh_bin_dir) + os.path.pathsep + os.environ.get("PATH", "")

    local_version = get_local_github_version(gh_exe)

    if local_version is None:
        logger.info("\nChecking for GitHub CLI (gh)...")
        logger.info(f"  -> GitHub CLI is not installed locally in workspace ({github_dir}).")
        install_behavior = getattr(config, 'github_install_behavior', 'always_install')
        if install_behavior == 'never_install':
            logger.info("  -> Skipping GitHub CLI installation as per configuration.")
            return False
        elif install_behavior == 'existing':
            if not prompt_choice("Do you want to install GitHub CLI into workspace? (y/n): ", default='y'):
                logger.info("  -> Skipping GitHub CLI installation.")
                return False
        logger.info(f"Installing GitHub CLI into workspace ({github_dir})...")
    else:
        logger.info(f"\nStatus: Found local GitHub CLI (Version: {local_version}) in {github_dir}")
        update_behavior = getattr(config, 'github_update_behavior', 'never_update')
        from local_ai_launcher.update_frequency import should_update_component, record_successful_update
        do_update = should_update_component(
            component_name="github_cli",
            category="cli",
            behavior=update_behavior,
            frequency=getattr(config, 'update_frequency_hrs', 168),
            force=getattr(config, 'force_update', False),
            prompt_if_existing=True,
        )
        if not do_update:
            return True
        logger.info("Updating GitHub CLI in workspace...")

    logger.info("Checking for the latest GitHub CLI release...")
    remote_version = None
    download_url = None

    try:
        req = urllib.request.Request("https://github.com/cli/cli/releases/latest", headers={'User-Agent': 'GitHubCliUpdater'})
        with urllib.request.urlopen(req, timeout=15) as resp:
            tag_match = re.search(r'/tag/(v[\d\.]+)', resp.geturl())
            if tag_match:
                remote_version = tag_match.group(1)
                clean_ver = remote_version.lstrip('v')
                download_url = f"https://github.com/cli/cli/releases/download/{remote_version}/gh_{clean_ver}_windows_amd64.zip"
    except Exception as e:
        logger.warning(f"Could not resolve latest release via redirect: {e}")

    if not remote_version:
        try:
            api_url = "https://api.github.com/repos/cli/cli/releases/latest"
            req = urllib.request.Request(api_url, headers={'User-Agent': 'GitHubCliUpdater'})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                remote_version = data.get("tag_name")
                for asset in data.get("assets", []):
                    name = asset.get("name", "")
                    if "windows" in name and "amd64" in name and name.endswith(".zip"):
                        download_url = asset.get("browser_download_url")
                        break
        except Exception as e:
            logger.error(f"Error fetching GitHub CLI release metadata: {e}")
            if local_version:
                return True
            return False

    if local_version and remote_version and local_version == remote_version:
        logger.info(f"  -> GitHub CLI is already up to date ({local_version}).")
        return True

    if not download_url:
        logger.error("Could not find matching Windows download archive for GitHub CLI.")
        return local_version is not None

    logger.info(f"Downloading GitHub CLI {remote_version} to workspace ({github_dir})...")
    try:
        req = urllib.request.Request(download_url, headers={'User-Agent': 'GitHubCliUpdater'})
        with urllib.request.urlopen(req, timeout=120) as resp:
            zip_buffer = io.BytesIO(resp.read())

        github_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_buffer) as z:
            safe_extract_zip(z, github_dir)

        # Create wrapper so both `gh` and `github` are runnable
        cmd_file = gh_bin_dir / "github.cmd"
        cmd_file.write_text('@echo off\n"%~dp0gh.exe" %*\n', encoding='utf-8')
        try:
            shutil.copy2(gh_exe, gh_bin_dir / "github.exe")
        except Exception:
            pass

        logger.info(f"  -> GitHub CLI {remote_version} installed successfully to {gh_exe}")
        try:
            record_successful_update("cli:github_cli")
        except Exception:
            pass
        return True
    except Exception as e:
        logger.error(f"Failed to install GitHub CLI: {e}")
        return local_version is not None


ensure_github = check_github


def ensure_running_in_venv():
    # Guard against infinite re-launch loops (e.g. path casing/symlink mismatches)
    if os.environ.get('_IN_WORKSPACE_VENV') == '1':
        return

    workspace_dir = AI_WORKSPACE_DIR
    venv_dir = workspace_dir / ".venv"
    venv_python = venv_dir / "Scripts" / "python.exe"
    
    setup_workspace_git()
    setup_workspace_venv()
    
    current_exe = Path(sys.executable).resolve()
    target_exe = venv_python.resolve()
    
    if current_exe != target_exe:
        logger.info(f"Relaunching script using workspace venv Python: {target_exe}")
        
        # Set guard to prevent infinite re-launch loops
        os.environ['_IN_WORKSPACE_VENV'] = '1'
        
        # Ensure we pass the correct arguments to the new python interpreter
        if sys.argv[0].endswith("__main__.py"):
            new_args = ["-m", "local_ai_launcher"] + sys.argv[1:]
        else:
            new_args = [os.path.abspath(sys.argv[0])] + sys.argv[1:]
        
        if sys.platform == 'win32':
            ret = subprocess.call([str(target_exe)] + new_args)
            sys.exit(ret)
        else:
            os.execv(str(target_exe), [str(target_exe)] + new_args)


def _get_cuda_build_env():
    """Ensure CUDA_PATH and CUDA_HOME point to a valid CUDA installation with include headers."""
    env = os.environ.copy()
    if sys.platform == 'win32':
        candidates = [
            env.get("CUDA_PATH_V13_4"),
            env.get("CUDA_PATH_V13_3"),
            env.get("CUDA_PATH_V13_0"),
            env.get("CUDA_PATH"),
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8",
        ]
        for candidate in candidates:
            if candidate and (Path(candidate) / "include" / "cuda_runtime_api.h").exists():
                env["CUDA_PATH"] = candidate
                env["CUDA_HOME"] = candidate
                break
    return env


def is_genuine_intel_processor() -> bool:
    import platform
    import sys
    if sys.platform != 'win32':
        return False
    processor = platform.processor().lower()
    return 'intel' in processor or 'genuineintel' in processor


def set_pcore_affinity(process_id: Optional[int] = None, num_cores: int = 8) -> bool:
    """
    Pins thread/process affinity to physical P-cores (cores 0..num_cores-1).
    On Intel Core Ultra 9 275HX, cores 0-7 are Lion Cove P-cores, preventing
    thread demotion to Skymont E-cores and eliminating OpenMP straggler stalls.
    """
    if sys.platform != "win32":
        return True
    try:
        import psutil
        proc = psutil.Process(process_id) if process_id else psutil.Process()
        avail_logical = psutil.cpu_count(logical=True) or num_cores
        target_cores = list(range(min(num_cores, avail_logical)))
        proc.cpu_affinity(target_cores)
        logger.info(f"Set CPU affinity to P-cores {target_cores} for PID {proc.pid}")
        return True
    except Exception as e:
        logger.debug(f"Could not set P-core affinity: {e}")
        return False


def configure_directx_gpu_preferences() -> bool:
    """
    Registers discrete high-performance GPU preference (GpuPreference=2) in Windows Registry
    (HKCU\\Software\\Microsoft\\DirectX\\UserGpuPreferences) for all local AI runtime binaries.
    Prevents Windows WDDM hybrid graphics / Optimus from vacillating between iGPU and dGPU.
    """
    if sys.platform != "win32":
        return True
    try:
        import winreg
        key_path = r"Software\Microsoft\DirectX\UserGpuPreferences"
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, key_path) as key:
            target_binaries = [
                str(AI_WORKSPACE_DIR / "llama-cpp" / "llama-server.exe"),
                str(AI_WORKSPACE_DIR / "llama-cpp-blackwell" / "llama-server.exe"),
                str(AI_WORKSPACE_DIR / "llama-cpp-blackwell" / "llama-server-mtpctx.exe"),
                str(AI_WORKSPACE_DIR / "llama-cpp-huang" / "llama-server.exe"),
                str(AI_WORKSPACE_DIR / "llama-cpp-huang" / "llama-server-mtpctx.exe"),
                str(AI_WORKSPACE_DIR / "localai" / "local-ai.exe"),
            ]
            for binary_path in target_binaries:
                p = Path(binary_path)
                if p.exists() or "llama" in binary_path or "local-ai" in binary_path:
                    try:
                        winreg.SetValueEx(key, str(p.resolve()), 0, winreg.REG_SZ, "GpuPreference=2;")
                    except Exception:
                        pass

            # Ensure general python.exe is NOT forced onto discrete GPU to preserve laptop battery
            py_binaries = [
                str(Path(AI_WORKSPACE_DIR / ".venv" / "Scripts" / "python.exe").resolve()),
                str(Path(sys.executable).resolve()),
            ]
            for py_bin in py_binaries:
                try:
                    winreg.DeleteValue(key, py_bin)
                except Exception:
                    pass
        return True
    except Exception as e:
        logger.debug(f"DirectX GPU preference registry notice: {e}")
        return False


def warmup_gpu_for_hybrid_mode(timeout: float = 10.0) -> dict:
    """
    Safely wakes up discrete NVIDIA GPU from D3cold (0W sleep) on hybrid (Optimus) laptops
    before launching heavy CUDA model loading or inference.
    Wakes up the PCIe link, ensures P0/P2 power state, and prevents BugCheck 0x00020001 (HYPERVISOR_ERROR 0x28).
    """
    result = {"awake": False, "device_name": None, "pstate": None, "temp_c": None, "power_w": None}
    if not getattr(config, "use_cuda", True):
        return result

    # 1. Register DirectX preferences so Windows knows these binaries target the discrete GPU
    configure_directx_gpu_preferences()

    # 2. Native CUDA Driver API probe (Zero dependency: uses nvcuda.dll directly)
    if sys.platform == "win32":
        try:
            import ctypes
            nvcuda = ctypes.WinDLL("nvcuda.dll")
            if nvcuda.cuInit(0) == 0:
                dev = ctypes.c_int()
                if nvcuda.cuDeviceGet(ctypes.byref(dev), 0) == 0:
                    name_buf = ctypes.create_string_buffer(256)
                    nvcuda.cuDeviceGetName(name_buf, 256, dev)
                    dev_name = name_buf.value.decode("utf-8", errors="ignore")
                    ctx = ctypes.c_void_p()
                    # cuCtxCreate_v2 wakes up hardware compute engine and trains PCIe link to P0
                    if nvcuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev) == 0:
                        nvcuda.cuCtxDestroy_v2(ctx)
                        result["awake"] = True
                        result["device_name"] = dev_name
                        logger.info(f"  -> Discrete GPU native CUDA context initialized: {dev_name}")
        except Exception as e:
            logger.debug(f"Native nvcuda driver wake-up notice: {e}")

    # 3. Gentle NVML / nvidia-smi wake-up probe for telemetry (PState, temp, power)
    if shutil.which("nvidia-smi"):
        try:
            t0 = time.time()
            creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,pstate,temperature.gpu,power.draw", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=timeout,
                creationflags=creation_flags
            )
            if res.returncode == 0 and res.stdout.strip():
                lines = res.stdout.strip().splitlines()
                if lines:
                    parts = [p.strip() for p in lines[0].split(",")]
                    if len(parts) >= 4:
                        result["awake"] = True
                        if not result["device_name"]:
                            result["device_name"] = parts[0]
                        result["pstate"] = parts[1]
                        try:
                            result["temp_c"] = float(parts[2])
                        except Exception:
                            pass
                        try:
                            result["power_w"] = float(parts[3])
                        except Exception:
                            pass
                        wake_dur = time.time() - t0
                        logger.info(
                            f"  -> Discrete GPU telemetry: {result['device_name']} "
                            f"(PState: {result['pstate']}, Temp: {result['temp_c']}C, Power: {result['power_w']}W, wake latency: {wake_dur:.1f}s)"
                        )
        except Exception as e:
            logger.debug(f"nvidia-smi wake-up notice: {e}")

    # 4. Settle duration for PCIe bus clock and voltage rails
    if result["awake"]:
        time.sleep(0.3)

    return result


