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

USER_MD_BOILERPLATE = """# USER.md - Universal Preferences

## Personal Context
- Operating System: Windows / PowerShell
- Preferred Output: Concise, high-density, minimal fluff.

## General Coding Standards
- Python: Python 3.11+, explicit type hints, `ruff` for formatting.
- TypeScript: Strict mode, preference for functional patterns over OOP.
- Shell: Write POSIX-compliant scripts where possible; prefer `uv` over standard `pip`.
"""


class PreferenceManager:
    """Manages the verification and setup of preference files like USER.md."""
    
    def __init__(self, config: MnemosyneAppConfig):
        self.config = config

    def verify_master_file(self) -> bool:
        """Verify master preference file exists. Returns True if file already existed, False if newly created."""
        if not self.config.master_path.exists():
            self.config.master_path.parent.mkdir(parents=True, exist_ok=True)
            self.config.master_path.write_text(USER_MD_BOILERPLATE, encoding="utf-8")
            mnemosyne_logger.info(f"[SETUP] Master preference file not found. Created boilerplate at: {self.config.master_path}")
            return False
        else:
            mnemosyne_logger.info(f"[VERIFY] Master preference file present at: {self.config.master_path}")
            return True


class BackupService:
    """Manages backups of config files with retention policy enforcement."""
    
    def __init__(self, config: MnemosyneAppConfig):
        self.config = config

    def handle_backups(self) -> None:
        self.config.backup_dir.mkdir(parents=True, exist_ok=True)
        all_backups = sorted(
            self.config.backup_dir.glob("USER_*.md"),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )

        should_backup = True
        if all_backups:
            latest_backup = all_backups[0]
            mtime = datetime.fromtimestamp(latest_backup.stat().st_mtime)
            days_since = (datetime.now() - mtime).total_seconds() / 86400.0
            
            if days_since < self.config.backup_interval_days:
                should_backup = False
                days_remaining = round(self.config.backup_interval_days - days_since, 1)
                mnemosyne_logger.info(f"[BACKUP] Skipping (Last backup: {latest_backup.name}. Next in ~{days_remaining} days)")

        if should_backup:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            backup_file_name = f"USER_{timestamp}.md"
            backup_file_path = self.config.backup_dir / backup_file_name

            shutil.copy2(self.config.master_path, backup_file_path)
            mnemosyne_logger.info(f"[BACKUP] Created copy: {backup_file_name}")

            updated_backups = sorted(
                self.config.backup_dir.glob("USER_*.md"),
                key=lambda f: f.stat().st_mtime,
                reverse=True,
            )

            if len(updated_backups) > self.config.max_backups_to_keep:
                files_to_delete = updated_backups[self.config.max_backups_to_keep:]
                for old_file in files_to_delete:
                    old_file.unlink(missing_ok=True)
                    mnemosyne_logger.info(f"  [CLEANUP] Deleted old backup: {old_file.name}")


class DependencyInstaller:
    """Manages virtual environments and dependency installations for Agents."""
    
    AGENT_PACKAGE_MAP = {
        "Hermes Agent": ["mnemosyne-hermes"],
        "Pi Coding Agent": ["mnemosyne-memory"],
    }

    def __init__(self, config: MnemosyneAppConfig):
        self.config = config

    def get_venv_python(self, agent_dir: Path) -> str:
        """Returns the platform-agnostic venv python executable if available."""
        win_py = agent_dir / "venv" / "Scripts" / "python.exe"
        posix_py = agent_dir / "venv" / "bin" / "python"
        
        if win_py.exists():
            return str(win_py)
        if posix_py.exists():
            return str(posix_py)
        return sys.executable

    def check_and_install_mnemosyne(self, agent_name: str, agent_dir: Path) -> None:
        if not agent_dir.exists():
            mnemosyne_logger.info(f"[MNEMOSYNE] Directory for {agent_name} not found at {agent_dir}. Skipping.")
            return

        python_bin = self.get_venv_python(agent_dir)
        required_packages = self.AGENT_PACKAGE_MAP.get(agent_name, ["mnemosyne-memory"])

        for pkg in required_packages:
            check_cmd = ["uv", "pip", "show", pkg, "--python", python_bin]
            res = subprocess.run(check_cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)

            if res.returncode == 0:
                mnemosyne_logger.info(f"[MNEMOSYNE] Confirmed '{pkg}' installed in {agent_name}.")
            elif self.config.install_mnemosyne == "install":
                mnemosyne_logger.info(f"  [INSTALL] Installing '{pkg}' for {agent_name}...")
                try:
                    subprocess.run(
                        ["uv", "pip", "install", pkg, "--python", python_bin],
                        cwd=str(agent_dir),
                        check=True,
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace"
                    )
                    mnemosyne_logger.info(f"  [INSTALL] Successfully installed '{pkg}'.")
                except subprocess.CalledProcessError as e:
                    mnemosyne_logger.error(f"  [ERROR] Failed to install '{pkg}': {e.stderr.strip()}")


class AgentConfigurator:
    """Manages the creation and updates of agent-specific config files."""
    
    def __init__(self, config: MnemosyneAppConfig):
        self.config = config

    def setup_agent_configs(self) -> None:
        mnemosyne_logger.info("Checking Agent Configurations...")
        params = compute_compaction_params(config.default_model)
        pi_reserve_tokens = params.pi_reserve_tokens
        pi_keep_tokens = params.pi_keep_tokens
        hermes_prune_tokens = params.hermes_prune_tokens
        protect_first_n = params.protect_first_n
        protect_last_n = params.protect_last_n
        compaction_enabled_str = params.compaction_enabled_str

        # --- Hermes Config ---
        self.config.hermes_config_dir.mkdir(parents=True, exist_ok=True)
        hermes_config_file = self.config.hermes_config_dir / "config.yaml"
        hermes_boilerplate = f"""memory:
  backend: mnemosyne
  db_path: {self.config.mnemosyne_db_path.as_posix()}
  sqlite_options:
    journal_mode: WAL
  compression:
    enabled: {compaction_enabled_str}
    threshold: {config.hermes_compression_threshold_ratio}
    protect_first_n: {protect_first_n}
    protect_last_n: {protect_last_n}
    proactive_prune_tokens: {hermes_prune_tokens}
    idle_compact_after_seconds: 1800

  ingestion:
    system_files:
      - {self.config.master_path.as_posix()}

    tags:
      - user-preference
      - hermes:task-history
      - hermes:tool-quirk
"""
        if not hermes_config_file.exists():
            hermes_config_file.write_text(hermes_boilerplate, encoding="utf-8")
            mnemosyne_logger.info(f"[CONFIG] Created Hermes configuration boilerplate at: {hermes_config_file}")
        else:
            content = hermes_config_file.read_text(encoding="utf-8")
            if "backend: mnemosyne" not in content:
                import re
                escaped_boilerplate = hermes_boilerplate.replace('\\', '\\\\')
                new_content = re.sub(r"^memory:\n(?:  .*?\n)*", escaped_boilerplate, content, flags=re.MULTILINE)
                if new_content != content:
                    hermes_config_file.write_text(new_content, encoding="utf-8")
                    mnemosyne_logger.info(f"[CONFIG] Updated Hermes configuration with Mnemosyne settings at: {hermes_config_file}")
                else:
                    hermes_config_file.write_text(content + "\n" + hermes_boilerplate, encoding="utf-8")
                    mnemosyne_logger.info(f"[CONFIG] Appended Mnemosyne settings to Hermes configuration at: {hermes_config_file}")
            else:
                mnemosyne_logger.info("[CONFIG] Hermes configuration present.")

        # --- Pi Coding Agent Config ---
        self.config.pi_config_dir.mkdir(parents=True, exist_ok=True)
        pi_config_file = self.config.pi_config_dir / "config.json"
        master_path_escaped = str(self.config.master_path).replace("\\", "\\\\")
        db_path_escaped = str(self.config.mnemosyne_db_path).replace("\\", "\\\\")
        pi_boilerplate = f"""{{
  "systemFiles": [
    "{master_path_escaped}"
  ],
  "compaction": {{
    "enabled": {compaction_enabled_str},
    "reserveTokens": {pi_reserve_tokens},
    "keepRecentTokens": {pi_keep_tokens}
  }},
  "mnemosyne": {{
    "dbPath": "{db_path_escaped}",
    "journalMode": "WAL",
    "filterTags": [
      "user-preference",
      "pi:*",
      "coding-convention"
    ],
    "excludeTags": [
      "hermes:task-history"
    ]
  }}
}}
"""
        if not pi_config_file.exists():
            pi_config_file.write_text(pi_boilerplate, encoding="utf-8")
            mnemosyne_logger.info(f"[CONFIG] Created Pi configuration boilerplate at: {pi_config_file}")
        else:
            try:
                import json
                pi_data = json.loads(pi_config_file.read_text(encoding="utf-8"))
                if "mnemosyne" not in pi_data:
                    pi_data["mnemosyne"] = {
                        "dbPath": str(self.config.mnemosyne_db_path),
                        "journalMode": "WAL",
                        "filterTags": ["user-preference", "pi:*", "coding-convention"],
                        "excludeTags": ["hermes:task-history"]
                    }
                    pi_data["compaction"] = {
                        "enabled": compaction_enabled_str == "true",
                        "reserveTokens": pi_reserve_tokens,
                        "keepRecentTokens": pi_keep_tokens
                    }
                    if "systemFiles" not in pi_data:
                        pi_data["systemFiles"] = []
                    if str(self.config.master_path) not in pi_data["systemFiles"]:
                        pi_data["systemFiles"].append(str(self.config.master_path))
                    pi_config_file.write_text(json.dumps(pi_data, indent=2), encoding="utf-8")
                    mnemosyne_logger.info(f"[CONFIG] Updated Pi configuration with Mnemosyne settings at: {pi_config_file}")
                else:
                    mnemosyne_logger.info("[CONFIG] Pi configuration present.")
            except Exception as e:
                mnemosyne_logger.error(f"[CONFIG] Failed to update Pi configuration: {e}")

        # --- T3 Code Harness Config ---
        try:
            from local_ai_launcher.harnesses.t3 import setup_t3_config
            setup_t3_config(
                self.config.t3_config_dir,
                f"http://localhost:{LLM_BACKEND_PORT}/v1",
                config.default_model
            )
            mnemosyne_logger.info("[CONFIG] T3 Code configuration verified.")
        except ImportError:
            mnemosyne_logger.debug("[CONFIG] T3 harness module not available, skipping.")
        except Exception as e:
            mnemosyne_logger.warning(f"[CONFIG] Failed to configure T3 Code: {e}")

        # --- Orca ADE Harness Config ---
        try:
            from local_ai_launcher.harnesses.orca import setup_orca_config
            setup_orca_config(
                self.config.orca_config_dir,
                f"http://localhost:{LLM_BACKEND_PORT}/v1",
                config.default_model
            )
            mnemosyne_logger.info("[CONFIG] Orca configuration verified.")
        except ImportError:
            mnemosyne_logger.debug("[CONFIG] Orca harness module not available, skipping.")
        except Exception as e:
            mnemosyne_logger.warning(f"[CONFIG] Failed to configure Orca: {e}")


class DatabaseManager:
    """Manages SQLite properties and system databases."""
    
    def __init__(self, config: MnemosyneAppConfig):
        self.config = config

    def enforce_sqlite_wal(self) -> None:
        mnemosyne_logger.info("Verifying SQLite WAL Mode...")

        try:
            self.config.mnemosyne_db_path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(str(self.config.mnemosyne_db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute("PRAGMA journal_mode=WAL;")
                result = cursor.fetchone()
                if result:
                    mnemosyne_logger.info(f"[DATABASE] SQLite WAL mode enabled on {self.config.mnemosyne_db_path} (Mode: {result[0]}).")
                else:
                    mnemosyne_logger.warning("[DATABASE] SQLite execution returned no results for WAL mode command.")
        except Exception as e:
            mnemosyne_logger.error(f"[DATABASE] Failed to set WAL mode via sqlite3 module: {e}")


class ProjectSyncService:
    """Synchronizes user preference files across active projects, sets up project-level Mnemosyne, and manages Git exclusions."""
    
    def __init__(self, config: MnemosyneAppConfig):
        self.config = config

    def ensure_project_mnemosyne(self, project: Path) -> None:
        """Verifies if a project has project-level Mnemosyne setup, creating it if missing."""
        mnemosyne_dir = project / ".mnemosyne"
        pi_dir = project / ".pi"
        pi_db_path = mnemosyne_dir / "memory_pi.db"
        hermes_db_path = mnemosyne_dir / "memory_hermes.db"
        pi_config_path = pi_dir / "config.json"
        hermes_config_path = mnemosyne_dir / "config.yaml"

        mnemosyne_dir.mkdir(parents=True, exist_ok=True)
        pi_dir.mkdir(parents=True, exist_ok=True)
        
        params = compute_compaction_params(config.default_model)
        pi_reserve_tokens = params.pi_reserve_tokens
        pi_keep_tokens = params.pi_keep_tokens
        hermes_prune_tokens = params.hermes_prune_tokens
        protect_first_n = params.protect_first_n
        protect_last_n = params.protect_last_n
        compaction_enabled_str = params.compaction_enabled_str

        for db_path in [pi_db_path, hermes_db_path]:
            if not db_path.exists():
                mnemosyne_logger.info(f"  [MNEMOSYNE] Initializing project-level Mnemosyne DB: {db_path}")
                try:
                    with sqlite3.connect(str(db_path)) as conn:
                        cursor = conn.cursor()
                        cursor.execute("PRAGMA journal_mode=WAL;")
                        conn.commit()
                    mnemosyne_logger.info(f"  [MNEMOSYNE] SQLite WAL mode initialized for {project.name} at {db_path.name}")
                except Exception as e:
                    mnemosyne_logger.error(f"  [ERROR] Failed to initialize project-level Mnemosyne DB for {project.name} ({db_path.name}): {e}")
            else:
                mnemosyne_logger.info(f"  [MNEMOSYNE] Project-level Mnemosyne DB present for {project.name} ({db_path.name})")

        if not pi_config_path.exists():
            db_path_escaped = str(pi_db_path).replace("\\", "\\\\")
            pi_config_boilerplate = f"""{{
  "projectName": "{project.name}",
  "compaction": {{
    "enabled": {compaction_enabled_str},
    "reserveTokens": {pi_reserve_tokens},
    "keepRecentTokens": {pi_keep_tokens}
  }},
  "mnemosyne": {{
    "dbPath": "{db_path_escaped}",
    "journalMode": "WAL",
    "filterTags": [
      "project:{project.name}",
      "coding-convention"
    ]
  }}
}}
"""
            pi_config_path.write_text(pi_config_boilerplate, encoding="utf-8")
            mnemosyne_logger.info(f"  [CONFIG] Created Pi project-level Mnemosyne config at: {pi_config_path}")

        if not hermes_config_path.exists():
            hermes_config_boilerplate = f"""memory:
  backend: mnemosyne
  db_path: {str(hermes_db_path)}
  sqlite_options:
    journal_mode: WAL
  compression:
    enabled: {compaction_enabled_str}
    threshold: {config.hermes_compression_threshold_ratio}
    protect_first_n: {protect_first_n}
    protect_last_n: {protect_last_n}
    proactive_prune_tokens: {hermes_prune_tokens}
    idle_compact_after_seconds: 1800
  ingestion:
    tags:
      - project:{project.name}
      - coding-convention
"""
            hermes_config_path.write_text(hermes_config_boilerplate, encoding="utf-8")
            mnemosyne_logger.info(f"  [CONFIG] Created Hermes project-level Mnemosyne config at: {hermes_config_path}")

    def sync_projects(self) -> None:
        if not self.config.projects_dir.exists():
            mnemosyne_logger.warning(f"[WARNING] Projects directory does not exist at: {self.config.projects_dir}")
            return

        projects = [p for p in self.config.projects_dir.iterdir() if p.is_dir() and not p.name.startswith('.')]
        if not projects:
            mnemosyne_logger.warning(f"[WARNING] No project directories found in {self.config.projects_dir}")
            return

        mnemosyne_logger.info(f"Starting sync across {len(projects)} projects...")

        for project in projects:
            target_user_md = project / "USER.md"
            shutil.copy2(self.config.master_path, target_user_md)
            mnemosyne_logger.info(f"[SYNCED] USER.md -> {project.name}")

            # Ensure project-level Mnemosyne is present (kept trackable in Git)
            self.ensure_project_mnemosyne(project)

            git_dir = project / ".git"
            if git_dir.exists() and git_dir.is_dir():
                info_folder = git_dir / "info"
                info_folder.mkdir(parents=True, exist_ok=True)
                local_exclude_path = info_folder / "exclude"

                existing_excludes = []
                if local_exclude_path.exists():
                    existing_excludes = local_exclude_path.read_text(encoding="utf-8").splitlines()

                entries_to_exclude = ["USER.md", ".t3", ".orca", ".worktreeinclude"]
                new_excludes = [e for e in entries_to_exclude if e not in existing_excludes]
                if new_excludes:
                    with open(local_exclude_path, "a", encoding="utf-8") as f:
                        f.write("\n" + "\n".join(new_excludes) + "\n")
                    mnemosyne_logger.info(f"  [GIT] Excluded {', '.join(new_excludes)} in local .git/info/exclude")


def run_mnemosyne_verifier() -> None:
    """Executes all Mnemosyne verification and project synchronization tasks."""
    config = MnemosyneAppConfig()

    preference_manager = PreferenceManager(config)
    backup_service = BackupService(config)
    dependency_installer = DependencyInstaller(config)
    agent_configurator = AgentConfigurator(config)
    database_manager = DatabaseManager(config)
    project_sync = ProjectSyncService(config)

    file_existed = preference_manager.verify_master_file()
    if file_existed:
        backup_service.handle_backups()

    dependency_installer.check_and_install_mnemosyne("Hermes Agent", config.hermes_dir)
    dependency_installer.check_and_install_mnemosyne("Pi Coding Agent", config.pi_dir)

    agent_configurator.setup_agent_configs()
    database_manager.enforce_sqlite_wal()
    project_sync.sync_projects()

    mnemosyne_logger.info("All Mnemosyne verification tasks completed successfully!")


