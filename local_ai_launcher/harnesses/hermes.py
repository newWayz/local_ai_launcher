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

def _run_silent(cmd: List[str] | str, **kwargs) -> subprocess.CompletedProcess:
    """Run a subprocess silently on Windows without allocating or flashing console windows."""
    if sys.platform == 'win32':
        kwargs.setdefault('creationflags', subprocess.CREATE_NO_WINDOW)
        startupinfo = kwargs.get('startupinfo')
        if startupinfo is None:
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = 0  # SW_HIDE
            kwargs['startupinfo'] = startupinfo
    kwargs.setdefault('stdout', subprocess.DEVNULL)
    kwargs.setdefault('stderr', subprocess.DEVNULL)
    return subprocess.run(cmd, **kwargs)


class HermesAdapter(HarnessAdapter):
    """Adapter for the Hermes coding agent stack."""
    
    def __init__(self, root_dir, processes):
        super().__init__("hermes", root_dir, processes)
        self.hermes_home = config.hermes_dir
        self.hermes_agent_dir = self.hermes_home / "hermes-agent"

    def _ensure_user_profile_junction(self):
        """
        Ensures ~/.hermes in the user profile directory is an NTFS Junction
        pointing to the workspace hermes_home (C:/aiWorkspace/.hermes).
        This maintains a single canonical Hermes configuration and skill directory.
        """
        if sys.platform != 'win32':
            return
        user_hermes = Path.home() / ".hermes"
        target_hermes = self.hermes_home.resolve()
        ensure_directory_junction(user_hermes, target_hermes)

    def _sanitize_install_state(self) -> None:
        """
        Detects and heals corrupted or broken PM install state in .hermes/installs.
        If facts.json points to a missing or broken virtual environment, redirects it
        to a healthy generation or purges the stale record so install.ps1/source_completion.py
        does not crash with [WinError 2] (executable not found).
        """
        installs_dir = self.hermes_home / "installs"
        if not installs_dir.exists():
            return

        for install_dir in installs_dir.iterdir():
            if not install_dir.is_dir():
                continue
            facts_file = install_dir / "facts.json"
            environments_dir = install_dir / "environments"
            if not facts_file.exists():
                continue

            try:
                data = json.loads(facts_file.read_text(encoding="utf-8-sig"))
                env_str = data.get("packages", {}).get("venv", {}).get("environment")
                env_path = Path(env_str) if env_str else None
                py_path = (env_path / "Scripts" / "python.exe") if (env_path and sys.platform == 'win32') else (env_path / "bin" / "python") if env_path else None

                if py_path and py_path.exists():
                    continue

                # The recorded environment is broken/missing. Search for an alternative healthy generation.
                logger.warning(f"  -> [Self-Heal] Detected broken Hermes environment in {facts_file}: {env_path}. Searching for healthy generation...")
                healthy_venv = None
                if environments_dir.exists():
                    for gen_dir in environments_dir.iterdir():
                        cand_venv = gen_dir / "venv"
                        cand_py = (cand_venv / "Scripts" / "python.exe") if sys.platform == 'win32' else (cand_venv / "bin" / "python")
                        if cand_py.exists():
                            healthy_venv = cand_venv
                            break

                if healthy_venv:
                    logger.info(f"  -> [Self-Heal] Recovered Hermes dependency environment: {healthy_venv}")
                    data.setdefault("packages", {}).setdefault("venv", {})["environment"] = str(healthy_venv.resolve())
                    cand_lock = healthy_venv.parent / "workspace" / "uv.lock"
                    if cand_lock.exists():
                        data["packages"]["venv"]["resolved_lock"] = str(cand_lock.resolve())
                    facts_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
                else:
                    logger.warning(f"  -> [Self-Heal] No healthy generation found; purging stale facts.json to allow clean bootstrap.")
                    facts_file.unlink(missing_ok=True)
            except Exception as e:
                logger.debug(f"Notice while sanitizing Hermes install state {facts_file}: {e}")

    def _resolve_hermes_venv(self) -> Optional[Path]:
        """Find the active virtual environment for Hermes (via PM generation or local venv)."""
        self._sanitize_install_state()
        # 1. Check PM facts.json in hermes_home / installs
        try:
            installs_dir = self.hermes_home / "installs"
            if installs_dir.exists():
                for facts_file in installs_dir.glob("*/facts.json"):
                    data = json.loads(facts_file.read_text(encoding="utf-8-sig"))
                    env_str = data.get("packages", {}).get("venv", {}).get("environment")
                    if env_str:
                        env_path = Path(env_str)
                        if env_path.exists() and (env_path / "Scripts" / "python.exe").exists():
                            return env_path
        except Exception:
            pass

        # 2. Check local venv
        local_venv = self.hermes_agent_dir / "venv"
        if local_venv.exists() and (local_venv / "Scripts" / "python.exe").exists():
            return local_venv
        return None

    def _ensure_venv_junction(self) -> None:
        """Ensure self.hermes_agent_dir / 'venv' points to the active PM environment generation."""
        active_venv = self._resolve_hermes_venv()
        if not active_venv or not self.hermes_agent_dir.exists():
            return
        local_venv = self.hermes_agent_dir / "venv"
        try:
            if local_venv.exists() and local_venv.resolve() == active_venv.resolve():
                return
            ensure_directory_junction(local_venv, active_venv)
        except Exception as e:
            logger.debug(f"Hermes venv junction notice: {e}")

    def _test_executable(self, exe: Path | str) -> bool:
        """Test if a CLI binary runs cleanly without throwing errors or flashing console windows."""
        if not exe:
            return False
        path_obj = Path(exe)
        if not path_obj.exists():
            return False
        try:
            startupinfo = None
            if sys.platform == 'win32':
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = 0
            res = subprocess.run(
                [str(path_obj), "--version"],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=15,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0,
                startupinfo=startupinfo
            )
            return res.returncode == 0
        except Exception:
            return False

    def _clean_broken_trampolines(self) -> None:
        """Detect and remove corrupted or broken uv trampolines in bin that shadow commands."""
        bin_dir = self.hermes_home / "bin"
        venv_scripts = self.hermes_agent_dir / "venv" / "Scripts"
        heal_workspace_bin_trampolines([str(bin_dir), str(venv_scripts)])

    def _resolve_hermes_cli(self) -> Optional[Path]:
        """Resolve the functional hermes CLI launcher executable in preference order."""
        candidates = [
            self.hermes_home / "bin" / ("hermes.exe" if sys.platform == 'win32' else "hermes"),
            self.hermes_home / "bin" / "hermes.cmd",
            self.hermes_agent_dir / "venv" / "Scripts" / "hermes.exe",
        ]
        for c in candidates:
            if self._test_executable(c):
                return c

        which_path = shutil.which("hermes")
        if which_path and self._test_executable(which_path):
            return Path(which_path)
        return None

    def _repair_launchers(self, hermes_venv_py: Path) -> None:
        """Repair launcher entry points and trampolines."""
        logger.info("  -> Repairing Hermes launchers and entry points...")
        launchers_py = self.hermes_agent_dir / "hermes_cli" / "_launchers.py"
        bin_dir = self.hermes_home / "bin"
        bin_dir.mkdir(parents=True, exist_ok=True)
        if launchers_py.exists():
            try:
                _run_silent(
                    [str(hermes_venv_py), "-I", "-X", "utf8", str(launchers_py), str(bin_dir)],
                    check=True,
                    timeout=60,
                    capture_output=True
                )
                logger.info("  -> Re-minted Hermes launcher commands in bin.")
            except Exception as e:
                logger.debug(f"Notice: Failed to re-mint launchers via _launchers.py: {e}")

        try:
            uv_exe = ensure_uv()
            if uv_exe:
                _run_silent(
                    [uv_exe, "pip", "install", "--python", str(hermes_venv_py), "-e", str(self.hermes_agent_dir)],
                    timeout=300,
                    check=True,
                    capture_output=True
                )
                logger.info("  -> Hermes entry point trampolines repaired successfully.")
        except Exception as e:
            logger.warning(f"  -> Warning: Could not repair Hermes entry points: {e}")

    def _install_hermes(self) -> bool:
        """Clone and install Hermes Agent into the workspace."""
        repo_url = getattr(config, 'hermes_repo_url', "https://github.com/NousResearch/hermes-agent.git")
        try:
            # 1. Clean broken trampolines and sanitize install state before starting install
            self._clean_broken_trampolines()
            self._sanitize_install_state()

            # 2. Check if hermes_agent_dir is an incomplete or non-git folder
            if self.hermes_agent_dir.exists() and not (self.hermes_agent_dir / ".git").exists():
                logger.warning(f"  -> {self.hermes_agent_dir} exists but is not a valid git repository. Cleaning up...")
                shutil.rmtree(self.hermes_agent_dir, ignore_errors=True)

            if not self.hermes_agent_dir.exists():
                logger.info(f"  -> Cloning Hermes Agent from {repo_url}...")
                _run_silent(["git", "clone", repo_url, str(self.hermes_agent_dir)], check=True)
                logger.info(f"  -> Cloned Hermes Agent into {self.hermes_agent_dir}")

            # 3. Run official installer
            if sys.platform == 'win32':
                install_script = self.hermes_agent_dir / "scripts" / "install.ps1"
                if install_script.exists():
                    logger.info("  -> Running Hermes Agent bootstrap installer (scripts/install.ps1)...")
                    _run_silent([
                        "powershell.exe",
                        "-ExecutionPolicy", "Bypass",
                        "-File", str(install_script),
                        "-HermesHome", str(self.hermes_home),
                        "-InstallDir", str(self.hermes_agent_dir),
                        "-NonInteractive",
                    ], check=True, env=os.environ)
            else:
                install_script = self.hermes_agent_dir / "scripts" / "install.sh"
                if install_script.exists():
                    subprocess.run(["bash", str(install_script), "--skip-setup"], check=True, env=os.environ)

            # 4. Ensure venv junction is in place
            self._ensure_venv_junction()

            # 5. Verify CLI works
            cli_path = self._resolve_hermes_cli()
            return cli_path is not None
        except Exception as e:
            logger.error(f"  -> Error installing Hermes Agent: {e}")
            return False

    def check(self):
        logger.info(f"\nChecking for Hermes coding agent...")
        os.environ['HERMES_HOME'] = str(self.hermes_home)

        # Unify user profile ~/.hermes with workspace .hermes
        self._ensure_user_profile_junction()

        # Ensure builtin skills (deep-solve, cpu-tools, wipe-agy-chat, update-skills) are provisioned for Hermes
        try:
            from local_ai_launcher.skills_manager import SkillManager
            SkillManager().ensure_builtin_skills([self.hermes_home / "skills"])
        except Exception as e:
            logger.warning(f"  -> Warning: Failed to provision skills for Hermes: {e}")

        # Clean any broken trampolines and sync venv junction
        self._clean_broken_trampolines()
        self._ensure_venv_junction()

        hermes_venv_py = self.hermes_agent_dir / "venv" / "Scripts" / "python.exe"
        is_installed = self.hermes_agent_dir.exists() and hermes_venv_py.exists()

        if is_installed:
            logger.info(f"  -> Hermes coding agent is already installed at {self.hermes_agent_dir}.")
            # Test if launcher works cleanly
            cli_path = self._resolve_hermes_cli()
            if not cli_path:
                self._repair_launchers(hermes_venv_py)
                cli_path = self._resolve_hermes_cli()

            if self._should_update():
                logger.info("Updating Hermes coding agent...")
                try:
                    h_cmd = str(cli_path) if cli_path else "hermes"
                    _run_silent([h_cmd, "update"], shell=False, check=True, env=os.environ)
                    logger.info(f"  -> Hermes coding agent updated successfully.")
                    self._ensure_venv_junction()
                    self.record_update()
                except (subprocess.CalledProcessError, FileNotFoundError) as e:
                    logger.error(f"  -> Failed to update Hermes coding agent: {e}")
            return True
        else:
            logger.info(f"  -> Hermes coding agent is not installed.")
            if self._should_install():
                logger.info("Installing Hermes coding agent...")
                if self._install_hermes():
                    logger.info(f"  -> Hermes coding agent installed successfully.")
                    return True
                else:
                    logger.error(f"  -> Failed to install Hermes coding agent.")
                    logger.info("  -> Please install Hermes manually from: https://github.com/NousResearch/hermes-agent")
                    return False
            else:
                logger.info(f"  -> Skipping Hermes coding agent installation.")
                return False
    
    def start(self, llm_base_url, active_model=None, launch_tui=True, open_browser=True, auxiliary_only=False):
        model_name = resolve_model_name(active_model)
        os.environ['HERMES_HOME'] = str(self.hermes_home)
        os.environ['PYTHONIOENCODING'] = 'utf-8'
        os.environ['HERMES_GATEWAY_DETACHED'] = '1'
        os.environ['VIRTUAL_ENV'] = str(self.hermes_agent_dir / "venv")

        # Unify user profile ~/.hermes with workspace .hermes
        self._ensure_user_profile_junction()

        # Ensure builtin skills are provisioned for Hermes
        try:
            from local_ai_launcher.skills_manager import SkillManager
            SkillManager().ensure_builtin_skills([self.hermes_home / "skills"])
        except Exception as e:
            logger.warning(f"  -> Warning: Failed to provision skills for Hermes: {e}")
        
        if not self.hermes_agent_dir.exists() or not (self.hermes_agent_dir / "venv").exists():
            logger.error("Error: Hermes coding agent is not installed or not properly set up.")
            logger.info("Please ensure Hermes is installed before running with config.choose_harness='hermes'.")
            return
        
        # Point Hermes at the LLM endpoint (OmniRoute or direct llama-server)
        os.environ['OPENAI_BASE_URL'] = llm_base_url
        os.environ['OPENAI_API_BASE'] = llm_base_url

        
        # Stop any existing Hermes Gateway and Dashboard processes before starting clean instances
        logger.info("Ensuring previous Hermes background instances are stopped...")
        try:
            has_running_hermes = False
            pid_file = self.hermes_home / "gateway.pid"
            if pid_file.exists():
                has_running_hermes = True
            elif psutil is not None:
                for proc in psutil.process_iter(['name', 'cmdline']):
                    try:
                        p_name = (proc.info['name'] or '').lower()
                        if p_name in ('python.exe', 'pythonw.exe', 'hermes.exe'):
                            cmdline = " ".join(proc.cmdline() or []).lower()
                            if 'hermes_cli' in cmdline or str(self.hermes_home).lower() in cmdline:
                                has_running_hermes = True
                                break
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass

            if has_running_hermes:
                # Console python.exe (NOT pythonw.exe): hidden-console design per hermes-agent
                # aa2ae36c3f (#54220/#56747) — a console-less pythonw parent makes every
                # descendant spawn (cmd/git/npm/taskkill) flash a visible conhost window.
                pyw_exe = self.hermes_agent_dir / "venv" / "Scripts" / "python.exe"
                if not pyw_exe.exists():
                    pyw_exe = self.hermes_agent_dir / "venv" / "Scripts" / "pythonw.exe"
                cli_main = self.hermes_agent_dir / "hermes_cli" / "main.py"
                if pyw_exe.exists() and cli_main.exists():
                    _run_silent([str(pyw_exe), str(cli_main), "dashboard", "--stop"])
                    _run_silent([str(pyw_exe), str(cli_main), "gateway", "stop"])
                    time.sleep(1)
        except Exception as e:
            logger.info(f"Notice: Cleanup of previous Hermes processes: {e}")

        # Kill previous Hermes process from PID file first if present
        pid_file = self.hermes_home / "gateway.pid"
        if pid_file.exists() and psutil is not None:
            try:
                pid_text = pid_file.read_text(encoding="utf-8").strip()
                if pid_text.isdigit():
                    old_pid = int(pid_text)
                    if psutil.pid_exists(old_pid):
                        p = psutil.Process(old_pid)
                        p.terminate()
            except Exception:
                pass

        # Kill any remaining orphaned hermes python processes to release locks
        try:
            if psutil is not None:
                for proc in psutil.process_iter(['pid', 'name']):
                    try:
                        p_name = (proc.info['name'] or '').lower()
                        if p_name in ('python.exe', 'pythonw.exe', 'hermes.exe'):
                            cmdline = " ".join(proc.cmdline()).lower()
                            if 'hermes' in cmdline or str(self.hermes_home).lower() in cmdline:
                                proc.terminate()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
        except Exception as e:
            logger.warning(f"Failed to cleanup orphaned hermes processes: {e}")

        # Clean up stale locks and active session state files so Hermes starts completely anew
        stale_files = [
            self.hermes_home / "gateway.lock",
            self.hermes_home / "gateway.pid",
            self.hermes_home / "gateway_state.json",
            self.hermes_home / "auth.lock",
            self.hermes_home / "runtime" / "active_sessions.json",
            self.hermes_home / "runtime" / "active_sessions.lock",
        ]
        for sf in stale_files:
            try:
                if sf.exists():
                    sf.unlink()
            except Exception:
                pass

        # Update config.yaml directly without spawning verbose CLI subcommands
        config_file = self.hermes_home / "config.yaml"
        if config_file.exists():
            try:
                content = config_file.read_text(encoding="utf-8")
                target_model = resolve_model_name(active_model)

                content = re.sub(r'(model:\s*\n\s*default:\s*).*', lambda m: m.group(1) + target_model, content)
                if "base_url:" in content:
                    content = re.sub(r'(base_url:\s*).*', lambda m: m.group(1) + llm_base_url, content)
                if "SIGNAL_URL:" in content:
                    signal_port = getattr(config, 'signal_daemon_port', 8085)
                    content = re.sub(r'(SIGNAL_URL:\s*).*', lambda m: m.group(1) + f'http://127.0.0.1:{signal_port}', content)
                
                if getattr(config, "searxng_enabled", True) and str(getattr(config, "searxng_mode", "wsl_docker")).lower() != "disabled":
                    searxng_url = getattr(config, "searxng_effective_url", "http://127.0.0.1:8088")
                    if "search_backend:" in content:
                        content = re.sub(r'(search_backend:\s*).*', r'\1"searxng"', content)
                    if "SEARXNG_URL:" in content:
                        content = re.sub(r'(SEARXNG_URL:\s*).*', lambda m: m.group(1) + searxng_url, content)
                
                # Fetch discord token if any
                if config.start_discord_gateway:
                    accounts_file = AI_WORKSPACE_DIR / "config" / "accounts.ini"
                    conf = configparser.ConfigParser()
                    if accounts_file.exists():
                        conf.read(accounts_file, encoding='utf-8')
                    
                    discord_token = conf.get("Discord", "token", fallback=None) if conf.has_section("Discord") else None
                    
                    if not discord_token:
                        from local_ai_launcher.config import prompt_choice, prompt_string
                        print("\n[Discord Gateway Setup]")
                        if prompt_choice("No Discord token found. Would you like to set one up now? (y/n): "):
                            discord_token = prompt_string("Enter your Discord Bot Token: ")
                            if discord_token:
                                if not conf.has_section("Discord"):
                                    conf.add_section("Discord")
                                conf.set("Discord", "token", discord_token)
                                accounts_file.parent.mkdir(parents=True, exist_ok=True)
                                with open(accounts_file, 'w', encoding='utf-8') as f:
                                    conf.write(f)
                                logger.info("  -> Discord token saved to config/accounts.ini")
                                
                    if discord_token:
                        os.environ['DISCORD_TOKEN'] = discord_token
                        if "DISCORD_TOKEN:" in content:
                            content = re.sub(r'(DISCORD_TOKEN:\s*).*', lambda m: m.group(1) + discord_token, content)
                            
                config_file.write_text(content, encoding="utf-8")
            except Exception as e:
                logger.info(f"Notice: Direct config update failed: {e}")

        if auxiliary_only:
            from local_ai_launcher.context_preloader import preload_harness_context
            preload_harness_context(llm_base_url, model_name, harness_name="hermes")
            logger.info("  -> Hermes environment and configuration prepared for Orca auxiliary usage (no daemons spawned).")
            return

        # Start Signal CLI Daemon first if present
        signal_dir = AI_WORKSPACE_DIR / "signal-cli-0.14.4.1"
        signal_bat = signal_dir / "bin" / "signal-cli.bat"
        if config.start_signal_daemon and signal_bat.exists():
            accounts_file = AI_WORKSPACE_DIR / "config" / "accounts.ini"
            signal_account = None
            if accounts_file.exists():
                conf = configparser.ConfigParser()
                conf.read(accounts_file, encoding='utf-8')
                signal_account = conf.get("Signal", "account_number", fallback=None)
            
            if signal_account:
                signal_port = getattr(config, 'signal_daemon_port', 8085)
                signal_process = start_process(
                    "Signal CLI Daemon", 
                    f'& "{signal_bat}" --account {signal_account} daemon --http 127.0.0.1:{signal_port}',
                    str(signal_dir),
                    show_window=False
                )
                if signal_process:
                    self.processes.append(('Signal CLI', signal_process))
            else:
                logger.warning("Signal CLI found, but no account_number set in config/accounts.ini. Skipping Signal start.")

        # Start Hermes Gateway
        # Console python.exe under CREATE_NO_WINDOW owns ONE hidden console that all Hermes
        # descendants inherit (upstream fix aa2ae36c3f); pythonw.exe has no console and made
        # every child spawn flash a window. Also restores stdout/stderr so daemon logs flow.
        pyw_exe = self.hermes_agent_dir / "venv" / "Scripts" / "python.exe"
        if not pyw_exe.exists():
            pyw_exe = self.hermes_agent_dir / "venv" / "Scripts" / "pythonw.exe"
        cli_main = self.hermes_agent_dir / "hermes_cli" / "main.py"

        gateway_cmd = [str(pyw_exe), str(cli_main), "gateway", "run"]
        gateway_process = start_process(
            "Hermes Gateway", 
            gateway_cmd,
            str(self.hermes_agent_dir),
            show_window=False,
            use_shell=False
        )
        if gateway_process:
            self.processes.append(('Hermes Gateway', gateway_process))
        
        no_open_arg = ["--no-open"] if not open_browser else []
        # Start Hermes Dashboard (builds web UI automatically on launch if config.skip_hermes_web_ui_build is False)
        dashboard_cmd = [str(pyw_exe), str(cli_main), "dashboard", "--host", "127.0.0.1", "--port", "9119"] + no_open_arg
        if config.skip_hermes_web_ui_build:
            dashboard_cmd.append("--skip-build")
            logger.info("Starting Hermes Dashboard (skipping web UI build)...")
        else:
            logger.info("Starting Hermes Dashboard (building web UI on launch in background)...")

        dashboard_process = start_process(
            "Hermes Dashboard", 
            dashboard_cmd,
            str(self.hermes_agent_dir),
            show_window=False,
            use_shell=False
        )
        if dashboard_process:
            self.processes.append(('Hermes Dashboard', dashboard_process))

        # Pre-evaluate Hermes agent system prompt and tool definitions into KV cache for sub-second TTFT
        from local_ai_launcher.context_preloader import preload_harness_context
        preload_harness_context(llm_base_url, model_name, harness_name="hermes")

        if not launch_tui:
            logger.info("  -> Hermes services (gateway & dashboard) configured and started in background (headless mode).")
            return
        
        # Launch interactive Hermes TUI chat in a new PowerShell window (matching Pi, Pie, and OpenCode adapters)
        logger.info("Starting Hermes interactive TUI session in new PowerShell window...")
        cli_path = self._resolve_hermes_cli()
        hermes_bin = f"& '{cli_path}'" if cli_path else "hermes"

        env_setup = build_ps_env({
            'HERMES_HOME': self.hermes_home,
            'OPENAI_BASE_URL': llm_base_url,
            'OPENAI_API_BASE': llm_base_url,
            'PYTHONIOENCODING': 'utf-8',
        })

        try:
            # Launch the interactive session normally in the visible console
            hermes_process = start_in_wt_tab(f'powershell.exe -NoExit -Command "chcp 65001 | Out-Null; [console]::InputEncoding=[console]::OutputEncoding=[System.Text.Encoding]::UTF8; {env_setup}{hermes_bin} chat"', cwd=str(config.harness_start_dir))
            self.processes.append(('Hermes coding agent', hermes_process, self.TRACK_CONSOLE_CLOSE))
        except Exception as e:
            logger.error(f"Error starting Hermes: {e}")


