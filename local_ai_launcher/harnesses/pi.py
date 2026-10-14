from local_ai_launcher.process import build_ps_env, run_silent_subprocess as _run_silent
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

def setup_pi_models_and_auth(pi_home, llm_base_url, active_model=None, use_omniroute=False, omniroute_model="1-local_first-brain"):
    """Automatically configure models.json and auth.json for Pi coding harness to use llama-server/OmniRoute."""
    
    agent_dir = pi_home / "agent"
    agent_dir.mkdir(parents=True, exist_ok=True)
    pi_home.mkdir(parents=True, exist_ok=True)
    
    active_model_name = normalize_model_identifier(
        resolve_model_name(active_model, use_omniroute=use_omniroute, omniroute_model=omniroute_model)
    )
    preset_models = [m for m in get_available_preset_models() if m.get("id") != "*"]
    
    models_list = []
    matched = next(
        (m for m in preset_models if m["id"] == active_model_name or normalize_model_identifier(m["id"]) == active_model_name),
        None
    )
    ctx_size, is_reasoning = get_model_specs(active_model_name)
    if matched:
        canonical_matched = dict(matched)
        canonical_matched["id"] = active_model_name
        canonical_matched["name"] = active_model_name
        if is_reasoning or "-think" in str(active_model_name).lower():
            canonical_matched["reasoning"] = True
        models_list.append(canonical_matched)
    else:
        if "freetoken" in str(active_model_name).lower() and ctx_size == 32768:
            ctx_size = 8192 # Safe default for FreeToken when unconfigured and likely VRAM-bound
        model_entry = {
            "id": active_model_name,
            "name": active_model_name,
            "contextWindow": ctx_size,
            "input": ["text", "image"]
        }
        if is_reasoning or "-think" in str(active_model_name).lower():
            model_entry["reasoning"] = True
        models_list.append(model_entry)
        
    for m in preset_models:
        if not any(existing["id"] == m["id"] for existing in models_list):
            m_copy = dict(m)
            m_ctx, m_reason = get_model_specs(m["id"])
            if m_reason or "-think" in str(m["id"]).lower():
                m_copy["reasoning"] = True
            models_list.append(m_copy)

    if use_omniroute:
        # Dynamically fetch available models/combos from OmniRoute
        fetched = fetch_omniroute_models(llm_base_url)
        combos = [m for m in fetched if ('1-' in m or '2-' in m or 'combo' in m.lower())]
        if not combos:
            combos = ["1-local_first-brain", "2-online-free_workers"]
        
        for c in combos:
            if not any(m['id'] == c for m in models_list):
                models_list.append({
                    "id": c,
                    "name": c,
                    "contextWindow": 131072,
                    "input": ["text", "image"]
                })
            
    models_data = {
        "providers": {
            "openai": {
                "baseUrl": llm_base_url,
                "api": "openai-completions",
                "apiKey": "dummy",
                "models": models_list
            }
        }
    }
    
    auth_data = {
        "openai": {
            "type": "api_key",
            "key": "dummy"
        },
        "anthropic": {
            "type": "api_key",
            "key": "dummy"
        }
    }

    for target_dir in [agent_dir, pi_home]:
        models_file = target_dir / "models.json"
        auth_file = target_dir / "auth.json"
        try:
            from local_ai_launcher.paths import atomic_write_json
            for fp, data in [(models_file, models_data), (auth_file, auth_data)]:
                atomic_write_json(fp, data)
        except Exception as e:
            logger.warning(f"Warning: Could not write Pi config/auth to {target_dir}: {e}")

    # Ensure agent settings.json has correct defaultProvider, defaultModel and enabledModels
    settings_file = agent_dir / "settings.json"
    try:
        settings_data = {}
        if settings_file.exists():
            with open(settings_file, "r", encoding="utf-8") as f:
                settings_data = json.load(f)
        if not isinstance(settings_data, dict):
            settings_data = {}
        settings_data["defaultProvider"] = "openai"
        settings_data["defaultModel"] = active_model_name
        settings_data["enabledModels"] = ["openai/*", "openai/**"]

        # Ensure -builtin:mcp is set if roach-pi is enabled to prevent /mcp collision
        has_roach_pi = any("roach-pi" in ext for ext in config.harness_extensions.get("pi", []))
        ext_list = settings_data.get("extensions", [])
        if not isinstance(ext_list, list):
            ext_list = []
        if has_roach_pi and "-builtin:mcp" not in ext_list:
            ext_list.append("-builtin:mcp")
        settings_data["extensions"] = ext_list

        from local_ai_launcher.paths import atomic_write_json
        atomic_write_json(settings_file, settings_data)
    except Exception as e:
        logger.warning(f"Warning: Could not update Pi settings in {settings_file}: {e}")

    # Remove any stale project-local settings that break model resolution
    for stale_dir in [config.harness_start_dir, PROJECTS_DIR]:
        if stale_dir:
            stale_settings = Path(stale_dir) / ".pi" / "settings.json"
            if stale_settings.exists():
                try:
                    stale_settings.unlink()
                except Exception:
                    pass


def setup_pi_web_search_config(pi_home, workflow=None, auto_open_browser=None):
    """Automatically configure web-search.json for Pi coding harness to auto-approve search results headless."""
    if workflow is None:
        workflow = getattr(config, "pi_web_search_workflow", "auto-summary")
    if auto_open_browser is None:
        auto_open_browser = getattr(config, "pi_web_search_auto_open_browser", False)

    agent_dir = pi_home / "agent"
    for target_dir in [agent_dir, pi_home]:
        target_dir.mkdir(parents=True, exist_ok=True)
        ws_file = target_dir / "web-search.json"
        try:
            ws_data = {}
            if ws_file.exists():
                try:
                    with open(ws_file, "r", encoding="utf-8") as f:
                        ws_data = json.load(f)
                except Exception:
                    ws_data = {}
            ws_data["workflow"] = workflow
            ws_data["autoOpenBrowser"] = auto_open_browser
            if getattr(config, "searxng_enabled", True) and str(getattr(config, "searxng_mode", "wsl_docker")).lower() != "disabled":
                ws_data["searxngBaseUrl"] = getattr(config, "searxng_effective_url", "http://127.0.0.1:8088")
            if "provider" not in ws_data:
                ws_data["provider"] = "all"

            from local_ai_launcher.paths import atomic_write_json
            atomic_write_json(ws_file, ws_data)
        except Exception as e:
            logger.warning(f"Warning: Could not write Pi web search config to {target_dir}: {e}")


def setup_pi_mcp_config(pi_home: Path):
    """Ensure Pi agent mcp.json contains configured and enabled MCP tools (e.g. REA)."""
    agent_dir = pi_home / "agent"
    agent_dir.mkdir(parents=True, exist_ok=True)
    mcp_file = agent_dir / "mcp.json"
    current_data = {}
    if mcp_file.exists():
        try:
            with open(mcp_file, "r", encoding="utf-8") as f:
                current_data = json.load(f)
        except Exception:
            current_data = {}
    if not isinstance(current_data, dict):
        current_data = {}
    mcp_servers = current_data.setdefault("mcpServers", {})
    if not isinstance(mcp_servers, dict):
        mcp_servers = {}
        current_data["mcpServers"] = mcp_servers

    if getattr(config, "rea_enabled", True):
        rea_cmd = "cmd.exe" if sys.platform == "win32" else "npx"
        rea_args = ["/d", "/c", "npx", "-y", "rea-agents@latest", "mcp"] if sys.platform == "win32" else ["-y", "rea-agents@latest", "mcp"]
        mcp_servers["rea"] = {
            "type": "stdio",
            "command": rea_cmd,
            "args": rea_args,
            "env": {
                "REA_MCP_INPUT_SCHEMA_PROFILE": getattr(config, "rea_schema_profile", "compact"),
            }
        }
    else:
        mcp_servers.pop("rea", None)

    try:
        from local_ai_launcher.paths import atomic_write_json
        atomic_write_json(mcp_file, current_data)
    except Exception as e:
        logger.warning(f"Warning: Could not write Pi MCP config to {mcp_file}: {e}")


class PiAdapter(HarnessAdapter):
    """Adapter for the Pi coding harness (Python/TypeScript original)."""
    
    def __init__(self, root_dir, processes):
        super().__init__("pi", root_dir, processes)
        self.venv_dir = AI_WORKSPACE_DIR / ".venv"
        self.node_bin = AI_WORKSPACE_DIR / "node"
        self.local_bin = AI_WORKSPACE_DIR / "node_modules" / ".bin"
        self.pi_home = AI_WORKSPACE_DIR / ".pi"
        self.pi_exe = (
            (self.node_bin / "pi.cmd") if (self.node_bin / "pi.cmd").exists()
            else ((self.local_bin / "pi.cmd") if (self.local_bin / "pi.cmd").exists()
            else ((self.venv_dir / "Scripts" / "pi.exe") if (self.venv_dir / "Scripts" / "pi.exe").exists()
            else Path(shutil.which("pi") or "pi")))
        )
    
    def get_needed_libs(self):
        return []

    def verify_and_patch_pi_compat_exports(self):
        """Ensure @earendil-works/pi-ai packages export './compat' and fix extensions importing from compat paths."""
        
        target_pkg_jsons = [
            AI_WORKSPACE_DIR / "node_modules" / "@earendil-works" / "pi-ai" / "package.json",
            AI_WORKSPACE_DIR / "node" / "node_modules" / "@earendil-works" / "pi-ai" / "package.json",
            self.pi_home / "node_modules" / "@earendil-works" / "pi-ai" / "package.json",
            self.pi_home / "agent" / "node_modules" / "@earendil-works" / "pi-ai" / "package.json",
        ]
        
        for pkg_json_path in target_pkg_jsons:
            if pkg_json_path.exists():
                try:
                    with open(pkg_json_path, "r+", encoding="utf-8") as f:
                        data = json.load(f)
                        exports = data.get("exports", {})
                        if isinstance(exports, dict) and "./compat" not in exports:
                            exports["./compat"] = {"types": "./dist/index.d.ts", "import": "./dist/index.js"}
                            data["exports"] = exports
                            f.seek(0)
                            json.dump(data, f, indent=2)
                            f.truncate()
                            logger.info(f"  -> Auto-patched missing compat export in {pkg_json_path}")
                except Exception:
                    pass

        target_pwa_files = [
            self.pi_home / "git" / "pi-web-access" / "index.ts",
            self.pi_home / "git" / "pi-web-access" / "summary-review.ts",
            self.pi_home / "git" / "pi-web-access" / "page-query.ts",
            self.pi_home / "agent" / "npm" / "node_modules" / "pi-web-access" / "index.ts",
            self.pi_home / "agent" / "npm" / "node_modules" / "pi-web-access" / "summary-review.ts",
            self.pi_home / "agent" / "npm" / "node_modules" / "pi-web-access" / "page-query.ts",
            AI_WORKSPACE_DIR / "node" / "node_modules" / "pi-web-access" / "index.ts",
            AI_WORKSPACE_DIR / "node_modules" / "pi-web-access" / "index.ts",
        ]
        for fpath in target_pwa_files:
            if fpath.exists():
                try:
                    content = fpath.read_text(encoding="utf-8")
                    if "@earendil-works/pi-ai/compat" in content:
                        content = content.replace("@earendil-works/pi-ai/compat", "@earendil-works/pi-ai")
                        fpath.write_text(content, encoding="utf-8")
                        logger.info(f"  -> Auto-patched compat import in {fpath}")
                except Exception:
                    pass

        # Check roach-pi package.json for host packages in dependencies (e.g. @sinclair/typebox)
        roach_pkg_candidates = [
            self.pi_home / "agent" / "git" / "github.com" / "tmdgusya" / "roach-pi" / "package.json",
            self.pi_home / "git" / "github.com" / "tmdgusya" / "roach-pi" / "package.json",
            self.pi_home / "git" / "roach-pi" / "package.json",
        ]
        for roach_pkg_json in roach_pkg_candidates:
            if roach_pkg_json.exists():
                try:
                    with open(roach_pkg_json, "r+", encoding="utf-8") as f:
                        data = json.load(f)
                        deps = data.get("dependencies", {})
                        peer_deps = data.get("peerDependencies", {})
                        modified = False
                        if "@sinclair/typebox" in deps:
                            del deps["@sinclair/typebox"]
                            peer_deps["@sinclair/typebox"] = "*"
                            data["dependencies"] = deps
                            data["peerDependencies"] = peer_deps
                            modified = True
                        if modified:
                            f.seek(0)
                            json.dump(data, f, indent=2)
                            f.truncate()
                            logger.info(f"  -> Auto-patched @sinclair/typebox to peerDependencies in {roach_pkg_json}")
                    dup_typebox = roach_pkg_json.parent / "node_modules" / "@sinclair" / "typebox"
                    if dup_typebox.exists():
                        shutil.rmtree(dup_typebox, ignore_errors=True)
                except Exception as e:
                    logger.warning(f"  -> Warning: Could not patch roach-pi package.json: {e}")

    def ensure_agent_browser(self):
        """Ensure agent-browser CLI is installed locally in workspace node_modules with Windows binary shims."""
        agent_browser_cmd = self.local_bin / "agent-browser.cmd"
        agent_browser_exe = self.local_bin / "agent-browser.exe"
        if not agent_browser_cmd.exists() or not agent_browser_exe.exists():
            logger.info("\nChecking agent-browser dependency in workspace...")
            try:
                _run_silent(
                    [get_npm_exe(), "install", "--prefix", str(AI_WORKSPACE_DIR), "agent-browser@0.33.2"],
                    shell=NPM_SHELL,
                    check=True
                )
                native_bin = AI_WORKSPACE_DIR / "node_modules" / "agent-browser" / "bin" / "agent-browser-win32-x64.exe"
                if native_bin.exists():
                    shutil.copy2(native_bin, self.local_bin / "agent-browser.exe")
                    if self.node_bin.exists():
                        shutil.copy2(native_bin, self.node_bin / "agent-browser.exe")
                        if agent_browser_cmd.exists():
                            shutil.copy2(agent_browser_cmd, self.node_bin / "agent-browser.cmd")
                logger.info("  -> agent-browser verified in workspace.")
            except Exception as e:
                logger.warning(f"  -> Warning: Failed to ensure agent-browser: {e}")

    def ensure_extensions_and_skills(self):
        """Ensure extensions (config.harness_extensions[self.name]) and skills (config.harness_skills[self.name]) defined for this harness are installed."""
        self.verify_and_patch_pi_compat_exports()

        # Ensure builtin skills (deep-solve, cpu-tools, wipe-agy-chat, update-skills) are provisioned in Gateway (.agents/skills)
        try:
            from local_ai_launcher.skills_manager import SkillManager
            SkillManager().ensure_builtin_skills()
            # Clean up deprecated .pi/skills/ to avoid duplicate collisions with .agents/skills/
            legacy_pi_skills = self.pi_home / "skills"
            if legacy_pi_skills.exists():
                shutil.rmtree(legacy_pi_skills, ignore_errors=True)

            # Provision Pi-scoped skills (e.g. donsetch) directly to .pi/agent/skills/
            if getattr(config, "pi_donsetch_enabled", True):
                SkillManager().ensure_scoped_harness_skills("pi", self.pi_home / "agent" / "skills")
            else:
                scoped_skill = self.pi_home / "agent" / "skills" / "donsetch"
                if scoped_skill.exists():
                    shutil.rmtree(scoped_skill, ignore_errors=True)
                settings_file = self.pi_home / "agent" / "settings.json"
                if settings_file.exists():
                    try:
                        with open(settings_file, "r", encoding="utf-8") as sf:
                            sdata = json.load(sf)
                        pkgs = sdata.get("packages", [])
                        filtered_pkgs = [p for p in pkgs if "donsetch" not in p]
                        if len(filtered_pkgs) != len(pkgs):
                            sdata["packages"] = filtered_pkgs
                            from local_ai_launcher.paths import atomic_write_json
                            atomic_write_json(settings_file, sdata)
                    except Exception:
                        pass
        except Exception as e:
            logger.warning(f"  -> Warning: Failed to provision skills for Pi: {e}")

        # Ensure autonomous loop recovery and cpu-tools extensions are provisioned
        try:
            from local_ai_launcher.paths import LOCAL_AI_LAUNCHER_DIR, atomic_write_text
            shim_src = LOCAL_AI_LAUNCHER_DIR / "shims" / "loop_recovery.ts"
            target_ext = self.pi_home / "agent" / "extensions" / "loop-recovery.ts"
            if shim_src.exists():
                src_text = shim_src.read_text(encoding="utf-8")
                if not target_ext.exists() or target_ext.read_text(encoding="utf-8") != src_text:
                    atomic_write_text(target_ext, src_text)
                    logger.info(f"  -> Provisioned loop-recovery extension to {target_ext}")

            cpu_tools_src = LOCAL_AI_LAUNCHER_DIR / "shims" / "cpu_tools.ts"
            target_cpu_tools = self.pi_home / "agent" / "extensions" / "cpu-tools.ts"
            if cpu_tools_src.exists():
                cpu_text = cpu_tools_src.read_text(encoding="utf-8")
                if not target_cpu_tools.exists() or target_cpu_tools.read_text(encoding="utf-8") != cpu_text:
                    atomic_write_text(target_cpu_tools, cpu_text)
                    logger.info(f"  -> Provisioned cpu-tools extension to {target_cpu_tools}")

            auto_cycle_src = LOCAL_AI_LAUNCHER_DIR / "shims" / "auto_cycle.ts"
            target_auto_cycle = self.pi_home / "agent" / "extensions" / "auto-cycle.ts"
            if auto_cycle_src.exists():
                cycle_text = auto_cycle_src.read_text(encoding="utf-8")
                if not target_auto_cycle.exists() or target_auto_cycle.read_text(encoding="utf-8") != cycle_text:
                    atomic_write_text(target_auto_cycle, cycle_text)
                    logger.info(f"  -> Provisioned auto-cycle extension to {target_auto_cycle}")
        except Exception as e:
            logger.warning(f"  -> Warning: Failed to provision extensions for Pi: {e}")

        my_extensions = list(config.harness_extensions.get(self.name, []))
        if getattr(config, "pi_donsetch_enabled", True):
            if not any("donsetch" in ext for ext in my_extensions):
                my_extensions.append("npm:donsetch")
        else:
            my_extensions = [ext for ext in my_extensions if "donsetch" not in ext]

        my_skills = config.harness_skills.get(self.name, [])

        if not my_extensions and not my_skills:
            return

        pi_cmd_str = str(self.pi_exe) if self.pi_exe.exists() else "pi"
        env = os.environ.copy()
        env['PI_HOME'] = str(self.pi_home)
        env['PI_CODING_AGENT_DIR'] = str(self.pi_home / "agent")
        env['npm_config_allow_git'] = 'all'

        installed_output = ""
        try:
            res = _run_silent([pi_cmd_str, "list"], encoding='utf-8', errors='replace', env=env, shell=NPM_SHELL, stdin=subprocess.DEVNULL, timeout=45)
            if res.returncode == 0:
                installed_output = res.stdout
        except Exception as e:
            logger.warning(f"  -> Warning: Could not run 'pi list': {e}")

        def is_source_installed(source):
            basename = source.rstrip('/').split('/')[-1]
            short_name = source.replace("npm:", "").replace("git:", "")
            return (source in installed_output) or (basename in installed_output) or (short_name in installed_output)

        def ensure_extension_deps(source):
            basename = source.rstrip('/').split('/')[-1]
            for git_base in [self.pi_home / "agent" / "git", self.pi_home / "git"]:
                if git_base.exists():
                    for pkg_dir in git_base.glob(f"**/{basename}"):
                        if (pkg_dir / "package.json").exists():
                            node_modules_dir = pkg_dir / "node_modules"
                            # If node_modules is missing or missing packages, run npm install inside the extension folder
                            if not node_modules_dir.exists() or not any(node_modules_dir.iterdir()):
                                logger.info(f"  -> Installing missing dependencies for '{basename}' in {pkg_dir}...")
                                try:
                                    _run_silent([get_npm_exe(), "config", "set", "allow-git", "all"], shell=NPM_SHELL, env=env, stdin=subprocess.DEVNULL, timeout=45)
                                    _run_silent([get_npm_exe(), "install", "--omit=peer", "--prefix", str(pkg_dir)], shell=NPM_SHELL, check=True, env=env, stdin=subprocess.DEVNULL, timeout=120)
                                    logger.info(f"  -> Dependencies for '{basename}' installed successfully.")
                                except Exception as e:
                                    logger.error(f"  -> Warning: Failed to install dependencies for '{basename}': {e}")
                            else:
                                logger.info(f"  -> Dependencies for '{basename}' are already set up.")

            # Ensure donsetch binary is downloaded if npm blocked install-scripts
            if "donsetch" in source:
                donsetch_pkg = self.pi_home / "agent" / "npm" / "node_modules" / "donsetch"
                if donsetch_pkg.exists():
                    bin_name = "donsetch.exe" if sys.platform == 'win32' else "donsetch"
                    bin_path = donsetch_pkg / "binaries" / bin_name
                    if not bin_path.exists():
                        install_js = donsetch_pkg / "install.js"
                        if install_js.exists():
                            logger.info(f"  -> Running postinstall binary downloader for donsetch in {donsetch_pkg}...")
                            try:
                                node_exe = str(self.node_bin / ("node.exe" if sys.platform == 'win32' else "node"))
                                if not Path(node_exe).exists():
                                    node_exe = shutil.which("node") or "node"
                                _run_silent([node_exe, str(install_js)], cwd=str(donsetch_pkg), shell=NPM_SHELL, stdin=subprocess.DEVNULL, timeout=120)
                                if sys.platform != 'win32' and bin_path.exists():
                                    try:
                                        bin_path.chmod(bin_path.stat().st_mode | 0o111)
                                    except Exception:
                                        pass
                                logger.info("  -> donsetch binary verified.")
                            except Exception as e:
                                logger.warning(f"  -> Warning: Failed to run donsetch install.js: {e}")
                    if bin_path.exists() and self.node_bin.exists():
                        try:
                            for fn in ["donsetch.exe", "donsetch.exe.version", "onnxruntime.dll", "pdfium.dll"]:
                                s = donsetch_pkg / "binaries" / fn
                                if s.exists():
                                    shutil.copy2(s, self.node_bin / fn)
                        except Exception:
                            pass

        if my_extensions:
            logger.info(f"\nChecking {self.name} extensions...")
            for ext in my_extensions:
                if is_source_installed(ext):
                    logger.info(f"  -> Extension '{ext}' is already installed.")
                    ensure_extension_deps(ext)
                else:
                    logger.info(f"  -> Installing extension '{ext}' for {self.name}...")
                    try:
                        _run_silent([pi_cmd_str, "install", ext], check=True, env=env, shell=NPM_SHELL, stdin=subprocess.DEVNULL, timeout=120)
                        logger.info(f"  -> Extension '{ext}' installed successfully.")
                        ensure_extension_deps(ext)
                    except Exception as e:
                        logger.error(f"  -> Warning: Failed to install extension '{ext}': {e}")

        if my_skills:
            logger.info(f"\nChecking {self.name} skills...")
            for skill in my_skills:
                if is_source_installed(skill):
                    logger.info(f"  -> Skill '{skill}' is already installed.")
                else:
                    logger.info(f"  -> Installing skill '{skill}' for {self.name}...")
                    try:
                        _run_silent([pi_cmd_str, "install", skill], check=True, env=env, shell=NPM_SHELL, stdin=subprocess.DEVNULL, timeout=120)
                        logger.info(f"  -> Skill '{skill}' installed successfully.")
                    except Exception as e:
                        logger.error(f"  -> Warning: Failed to install skill '{skill}': {e}")
    
    def check(self):
        logger.info(f"\nChecking for pi coding harness...")
        os.environ['PI_HOME'] = str(self.pi_home)
        os.environ['PI_CODING_AGENT_DIR'] = str(self.pi_home / "agent")
        os.environ['OPENAI_API_KEY'] = 'dummy'
        os.environ['npm_config_allow_git'] = 'all'
        
        self.verify_and_patch_pi_compat_exports()
        
        is_installed = (self.node_bin / "pi.cmd").exists() or (self.local_bin / "pi.cmd").exists() or (self.venv_dir / "Scripts" / "pi.exe").exists() or shutil.which("pi")
        
        if is_installed:
            logger.info(f"  -> pi coding harness is already installed.")
            if self._should_update():
                logger.info("Updating pi coding harness...")
                updated_ok = False
                pi_cmd_str = str(self.pi_exe) if self.pi_exe.exists() else "pi"
                env = os.environ.copy()
                env['npm_config_allow_git'] = 'all'
                env.pop('npm_config_allow_scripts', None)

                # Primary strategy: self-update using pi's native updater
                try:
                    res = _run_silent([pi_cmd_str, "update", "--self"], shell=NPM_SHELL, check=True, env=env)
                    logger.info(f"  -> pi coding harness updated successfully via self-update.")
                    updated_ok = True
                except Exception as e:
                    logger.warning(f"  -> Notice: 'pi update --self' encountered: {e}. Trying npm global install fallback...")

                # Fallback strategy if self-update command failed
                if not updated_ok:
                    try:
                        npm_exe = get_npm_exe()
                        _run_silent([npm_exe, "install", "-g", "--ignore-scripts", "--min-release-age=0", "@earendil-works/pi-coding-agent@latest"], shell=NPM_SHELL, check=True)
                        logger.info(f"  -> pi coding harness updated successfully via npm install.")
                        updated_ok = True
                    except Exception as e:
                        logger.error(f"  -> Failed to update pi coding harness: {e}")

                # Also update pi extensions & skills
                logger.info("Updating pi extensions and skills...")
                try:
                    _run_silent([pi_cmd_str, "update", "--extensions"], shell=NPM_SHELL, check=True, env=env)
                    logger.info(f"  -> pi extensions & skills updated successfully.")
                except Exception as e:
                    logger.error(f"  -> Failed to update pi extensions & skills: {e}")
                if updated_ok:
                    self.record_update()
            
            # Ensure configured extensions and skills are installed
            self.ensure_extensions_and_skills()
            self.ensure_agent_browser()
            setup_pi_web_search_config(self.pi_home)
            return True
        else:
            logger.info(f"  -> pi coding harness is not installed.")
            if self._should_install():
                logger.info("Installing pi coding harness globally...")
                try:
                    _run_silent([get_npm_exe(), "install", "--prefix", str(AI_WORKSPACE_DIR), "@earendil-works/pi-coding-agent"], shell=NPM_SHELL, check=True)
                    logger.info(f"  -> pi coding harness installed globally.")
                    if (self.node_bin / "pi.cmd").exists():
                        self.pi_exe = self.node_bin / "pi.cmd"
                    elif (self.local_bin / "pi.cmd").exists():
                        self.pi_exe = self.local_bin / "pi.cmd"
                    
                    # Ensure configured extensions and skills are installed after pi installation
                    self.ensure_extensions_and_skills()
                    self.ensure_agent_browser()
                    setup_pi_web_search_config(self.pi_home)
                    return True
                except subprocess.CalledProcessError as e:
                    logger.error(f"  -> Failed to install pi coding harness: {e}")
                    return False
            else:
                logger.info(f"  -> Skipping pi coding harness installation.")
                return False
    
    def start(self, llm_base_url, active_model=None, launch_tui=True):
        pi_agent_dir = self.pi_home / "agent"
        goal_state_dir = pi_agent_dir / "goal-state"
        goal_state_dir.mkdir(parents=True, exist_ok=True)

        os.environ['PI_HOME'] = str(self.pi_home)
        os.environ['PI_CODING_AGENT_DIR'] = str(pi_agent_dir)
        os.environ['PI_GOAL_STATE_ROOT'] = str(goal_state_dir)
        os.environ['OPENAI_API_KEY'] = 'dummy'
        os.environ['npm_config_allow_git'] = 'all'

        try:
            from local_ai_launcher.skills_manager import GATEWAY_DIR, SkillManager
            from local_ai_launcher.bridges.gvs5h_solver import ensure_deep_solve_skills
            ensure_deep_solve_skills([GATEWAY_DIR / "deep-solve"])
            from local_ai_launcher.bridges.cpu_tools_cli import ensure_cpu_tools_skills
            ensure_cpu_tools_skills([GATEWAY_DIR / "cpu-tools"])
            if getattr(config, "pi_donsetch_enabled", True):
                SkillManager().ensure_scoped_harness_skills("pi", self.pi_home / "agent" / "skills")
        except Exception as e:
            logger.warning(f"  -> Warning: Failed to provision skills for Pi: {e}")

        try:
            from local_ai_launcher.paths import LOCAL_AI_LAUNCHER_DIR, atomic_write_text
            shim_src = LOCAL_AI_LAUNCHER_DIR / "shims" / "loop_recovery.ts"
            target_ext = self.pi_home / "agent" / "extensions" / "loop-recovery.ts"
            if shim_src.exists():
                src_text = shim_src.read_text(encoding="utf-8")
                if not target_ext.exists() or target_ext.read_text(encoding="utf-8") != src_text:
                    atomic_write_text(target_ext, src_text)

            cpu_tools_src = LOCAL_AI_LAUNCHER_DIR / "shims" / "cpu_tools.ts"
            target_cpu_tools = self.pi_home / "agent" / "extensions" / "cpu-tools.ts"
            if cpu_tools_src.exists():
                cpu_text = cpu_tools_src.read_text(encoding="utf-8")
                if not target_cpu_tools.exists() or target_cpu_tools.read_text(encoding="utf-8") != cpu_text:
                    atomic_write_text(target_cpu_tools, cpu_text)

            auto_cycle_src = LOCAL_AI_LAUNCHER_DIR / "shims" / "auto_cycle.ts"
            target_auto_cycle = self.pi_home / "agent" / "extensions" / "auto-cycle.ts"
            if auto_cycle_src.exists():
                cycle_text = auto_cycle_src.read_text(encoding="utf-8")
                if not target_auto_cycle.exists() or target_auto_cycle.read_text(encoding="utf-8") != cycle_text:
                    atomic_write_text(target_auto_cycle, cycle_text)
        except Exception as e:
            logger.warning(f"  -> Warning: Failed to provision extensions for Pi: {e}")

        # Auto-configure models.json and auth.json for Pi dynamically
        setup_pi_models_and_auth(self.pi_home, llm_base_url, active_model, use_omniroute=config.use_omniroute_proxy, omniroute_model=config.omniroute_default_model)
        setup_pi_web_search_config(self.pi_home)
        setup_pi_mcp_config(self.pi_home)
        model_name = resolve_model_name(active_model)

        # Pre-evaluate Pi agent system prompt and tool definitions into KV cache for sub-second TTFT
        from local_ai_launcher.context_preloader import preload_harness_context
        preload_harness_context(llm_base_url, model_name, harness_name="pi")

        if not launch_tui:
            logger.info("  -> Pi coding harness configured for local use (headless mode).")
            return

        pi_args = f"--provider openai --model {model_name}"
        pi_bin = f"& '{self.pi_exe}'" if self.pi_exe.exists() else "pi"
        pi_cmd = f"{pi_bin} {pi_args}"
        browser_home = AI_WORKSPACE_DIR / ".agent-browser"
        browser_home.mkdir(parents=True, exist_ok=True)
        downloads_dir = AI_WORKSPACE_DIR / "downloads"
        downloads_dir.mkdir(parents=True, exist_ok=True)

        pi_env_dict = {
            'OPENAI_BASE_URL': llm_base_url,
            'OPENAI_API_BASE': llm_base_url,
            'OPENAI_API_KEY': 'dummy',
            'PI_HOME': self.pi_home,
            'PI_CODING_AGENT_DIR': pi_agent_dir,
            'PI_GOAL_STATE_ROOT': goal_state_dir,
            'PI_LOOP_RECOVERY_MAX_RETRIES': str(max(0, int(getattr(config, 'loop_recovery_max_retries', 3)))),
            'PI_SKIP_VERSION_CHECK': '1',
            'npm_config_allow_git': 'all',
            'FORCE_COLOR': '1',
            'AGENT_BROWSER_HOME': str(browser_home),
            'AGENT_BROWSER_DOWNLOAD_PATH': str(downloads_dir),
        }
        if getattr(config, "pi_donsetch_enabled", True):
            from local_ai_launcher.paths import get_donsetch_binary_path
            don_bin = getattr(config, "donsetch_bin", None) or get_donsetch_binary_path()
            if don_bin:
                pi_env_dict['DONSETCH_BINARY'] = str(don_bin)
                pi_env_dict['DONSETCH_BIN'] = str(don_bin)
        env_setup = build_ps_env(pi_env_dict)
        logger.info("Starting Pi coding agent...")
        try:
            pi_process = start_in_wt_tab(f'powershell.exe -NoExit -Command "chcp 65001 | Out-Null; [console]::InputEncoding=[console]::OutputEncoding=[System.Text.Encoding]::UTF8; {env_setup}{pi_cmd}"', cwd=str(config.harness_start_dir))
            self.processes.append(('Pi coding agent', pi_process, self.TRACK_CONSOLE_CLOSE))
        except Exception as e:
            logger.error(f"Error starting Pi: {e}")


