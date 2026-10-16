"""
Adapter for Oh My Pi (omp) coding harness.

Oh My Pi is an advanced terminal AI coding agent and software architect
forked and expanded from Pi, featuring multi-provider support, LSP/AST integrations,
and built-in search scrapers. It is completely separate and distinct from the standard
Pi harness, maintaining its own configuration and state under .omp.
"""

import subprocess
import time
import sys
import os
import shutil
from pathlib import Path
import json
import logging
from typing import List, Optional, Dict

try:
    import yaml
except ImportError:
    yaml = None

from local_ai_launcher.config import (
    AI_WORKSPACE_DIR,
    PROJECTS_DIR,
    config,
)
from local_ai_launcher.environment import (
    get_npm_exe,
    ensure_bun,
    NPM_SHELL,
)
from local_ai_launcher.models import (
    get_available_preset_models,
    get_model_specs,
    normalize_model_identifier,
    resolve_model_name,
    fetch_omniroute_models,
)
from local_ai_launcher.process import (
    build_ps_env,
    run_silent_subprocess as _run_silent,
    start_in_wt_tab,
)
from local_ai_launcher.paths import (
    OMP_DIR,
    atomic_write_json,
    atomic_write_text,
)
from local_ai_launcher.harness_base import HarnessAdapter

logger = logging.getLogger("service_launcher")


def setup_omp_models_and_auth(
    omp_home: Path,
    llm_base_url: str,
    active_model: Optional[str] = None,
    use_omniroute: bool = False,
    omniroute_model: str = "1-local_first-brain"
) -> None:
    """Automatically configure models.json, models.yml, settings.json, settings.yml,
    and auth.json for Oh My Pi (omp) coding harness to use the unified LocalAI / OmniRoute gateway.

    Oh My Pi is separate and distinct from Pi and maintains its isolated state in .omp.
    """
    agent_dir = omp_home / "agent"
    agent_dir.mkdir(parents=True, exist_ok=True)
    omp_home.mkdir(parents=True, exist_ok=True)

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
            ctx_size = 8192
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

    yaml_text = None
    if yaml is not None:
        try:
            yaml_text = yaml.dump(models_data, sort_keys=False, indent=2)
        except Exception:
            yaml_text = None

    for target_dir in [agent_dir, omp_home]:
        models_json_file = target_dir / "models.json"
        models_yml_file = target_dir / "models.yml"
        auth_file = target_dir / "auth.json"
        try:
            atomic_write_json(models_json_file, models_data)
            atomic_write_json(auth_file, auth_data)
            if yaml_text:
                atomic_write_text(models_yml_file, yaml_text)
        except Exception as e:
            logger.warning(f"Warning: Could not write OMP config/auth to {target_dir}: {e}")

    # Ensure agent settings.json and settings.yml has correct defaultProvider, defaultModel and enabledModels
    settings_data = {
        "defaultProvider": "openai",
        "defaultModel": active_model_name,
        "enabledModels": ["openai/*", "openai/**"]
    }
    settings_file = agent_dir / "settings.json"
    settings_yml_file = agent_dir / "settings.yml"
    try:
        existing = {}
        if settings_file.exists():
            try:
                with open(settings_file, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            except Exception:
                existing = {}
        if isinstance(existing, dict):
            existing.update(settings_data)
            settings_data = existing
        atomic_write_json(settings_file, settings_data)
        if yaml is not None:
            atomic_write_text(settings_yml_file, yaml.dump(settings_data, sort_keys=False, indent=2))
    except Exception as e:
        logger.warning(f"Warning: Could not write OMP settings in {settings_file}: {e}")

    # Remove stale project-local settings that break model resolution
    for stale_dir in [config.harness_start_dir, PROJECTS_DIR]:
        if stale_dir:
            stale_settings = Path(stale_dir) / ".omp" / "settings.json"
            if stale_settings.exists():
                try:
                    stale_settings.unlink()
                except Exception:
                    pass


def setup_omp_mcp_config(omp_home: Path) -> None:
    """Ensure Oh My Pi agent mcp.json contains configured and enabled MCP tools (e.g. REA and Aside)."""
    agent_dir = omp_home / "agent"
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

    if getattr(config, "aside_browser_enabled", True):
        try:
            from local_ai_launcher.backends.aside import get_aside_mcp_server_config
            aside_cfg = get_aside_mcp_server_config()
            if aside_cfg:
                mcp_servers["aside"] = aside_cfg
            else:
                mcp_servers.pop("aside", None)
        except Exception as e:
            logger.warning(f"Notice: Failed to configure Aside MCP for OMP: {e}")
    else:
        mcp_servers.pop("aside", None)

    for target_dir in [agent_dir, omp_home]:
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_json(target_dir / "mcp.json", current_data)
        except Exception as e:
            logger.warning(f"Warning: Could not write OMP MCP config to {target_dir}: {e}")


class OmpAdapter(HarnessAdapter):
    """Adapter for Oh My Pi (omp) - the distinct terminal coding agent."""

    def __init__(self, root_dir, processes):
        super().__init__("omp", root_dir, processes)
        self.venv_dir = AI_WORKSPACE_DIR / ".venv"
        self.node_bin = AI_WORKSPACE_DIR / "node"
        self.local_bin = AI_WORKSPACE_DIR / "node_modules" / ".bin"
        self.omp_home = getattr(config, "omp_dir", OMP_DIR)
        self.omp_exe = self._resolve_executable()

    def _resolve_executable(self) -> Path:
        if (self.node_bin / "omp.cmd").exists():
            return self.node_bin / "omp.cmd"
        if (self.local_bin / "omp.cmd").exists():
            return self.local_bin / "omp.cmd"
        if (self.venv_dir / "Scripts" / "omp.exe").exists():
            return self.venv_dir / "Scripts" / "omp.exe"
        which_path = shutil.which("omp")
        if which_path:
            return Path(which_path)
        return Path("omp")

    def get_needed_libs(self):
        return []

    def ensure_extensions_and_skills(self):
        """Ensure extensions and skills defined for Oh My Pi are provisioned in .omp/agent/."""
        # Ensure builtin skills (deep-solve, cpu-tools, wipe-agy-chat, update-skills) are provisioned in Gateway (.agents/skills)
        try:
            from local_ai_launcher.skills_manager import SkillManager
            SkillManager().ensure_builtin_skills()
            legacy_omp_skills = self.omp_home / "skills"
            if legacy_omp_skills.exists():
                shutil.rmtree(legacy_omp_skills, ignore_errors=True)
        except Exception as e:
            logger.warning(f"  -> Warning: Failed to provision skills for Oh My Pi: {e}")

        # Provision shims to .omp/agent/extensions/
        try:
            from local_ai_launcher.paths import LOCAL_AI_LAUNCHER_DIR, atomic_write_text
            ext_dir = self.omp_home / "agent" / "extensions"
            ext_dir.mkdir(parents=True, exist_ok=True)

            shim_src = LOCAL_AI_LAUNCHER_DIR / "shims" / "loop_recovery.ts"
            target_ext = ext_dir / "loop-recovery.ts"
            if shim_src.exists():
                src_text = shim_src.read_text(encoding="utf-8")
                if not target_ext.exists() or target_ext.read_text(encoding="utf-8") != src_text:
                    atomic_write_text(target_ext, src_text)
                    logger.info(f"  -> Provisioned loop-recovery extension to {target_ext}")

            cpu_tools_src = LOCAL_AI_LAUNCHER_DIR / "shims" / "cpu_tools.ts"
            target_cpu_tools = ext_dir / "cpu-tools.ts"
            if cpu_tools_src.exists():
                cpu_text = cpu_tools_src.read_text(encoding="utf-8")
                if not target_cpu_tools.exists() or target_cpu_tools.read_text(encoding="utf-8") != cpu_text:
                    atomic_write_text(target_cpu_tools, cpu_text)
                    logger.info(f"  -> Provisioned cpu-tools extension to {target_cpu_tools}")

            auto_cycle_src = LOCAL_AI_LAUNCHER_DIR / "shims" / "auto_cycle.ts"
            target_auto_cycle = ext_dir / "auto-cycle.ts"
            if auto_cycle_src.exists():
                cycle_text = auto_cycle_src.read_text(encoding="utf-8")
                if not target_auto_cycle.exists() or target_auto_cycle.read_text(encoding="utf-8") != cycle_text:
                    atomic_write_text(target_auto_cycle, cycle_text)
                    logger.info(f"  -> Provisioned auto-cycle extension to {target_auto_cycle}")
        except Exception as e:
            logger.warning(f"  -> Warning: Failed to provision extensions for Oh My Pi: {e}")

    def check(self):
        logger.info("\nChecking for Oh My Pi (omp) coding harness...")
        ensure_bun()

        # Build scoped subprocess environment without permanently mutating process-global os.environ
        omp_home_resolved = self.omp_home.resolve()
        sub_env = os.environ.copy()
        sub_env['HOME'] = str(AI_WORKSPACE_DIR)
        sub_env['USERPROFILE'] = str(omp_home_resolved.parent)
        sub_env['OMP_HOME'] = str(omp_home_resolved)
        sub_env['PI_CONFIG_DIR'] = omp_home_resolved.name
        sub_env['PI_CODING_AGENT_DIR'] = str(omp_home_resolved / "agent")
        sub_env['OPENAI_API_KEY'] = 'dummy'
        sub_env['npm_config_allow_git'] = 'all'

        is_installed = (
            (self.node_bin / "omp.cmd").exists()
            or (self.local_bin / "omp.cmd").exists()
            or (self.venv_dir / "Scripts" / "omp.exe").exists()
            or (shutil.which("omp") is not None)
        )

        if is_installed:
            logger.info("  -> Oh My Pi (omp) coding harness is already installed.")
            if self._should_update():
                logger.info("Updating Oh My Pi (omp) coding harness...")
                updated_ok = False
                omp_cmd_str = str(self.omp_exe) if self.omp_exe.exists() else "omp"
                env = sub_env.copy()
                env.pop('npm_config_allow_scripts', None)

                # Primary: native self-update
                try:
                    res = _run_silent([omp_cmd_str, "update", "--self"], shell=NPM_SHELL, check=True, env=env)
                    logger.info("  -> Oh My Pi updated successfully via self-update.")
                    updated_ok = True
                except Exception as e:
                    logger.warning(f"  -> Notice: 'omp update --self' encountered: {e}. Trying npm update fallback...")

                if not updated_ok:
                    try:
                        npm_exe = get_npm_exe()
                        _run_silent([npm_exe, "install", "-g", "--ignore-scripts", "@oh-my-pi/pi-coding-agent@latest"], shell=NPM_SHELL, check=True, env=env)
                        logger.info("  -> Oh My Pi updated successfully via npm global install.")
                        updated_ok = True
                    except Exception as e:
                        logger.error(f"  -> Failed to update Oh My Pi: {e}")

                if updated_ok:
                    self.record_update()

            setup_omp_mcp_config(self.omp_home)
            self.ensure_extensions_and_skills()
            return True
        else:
            logger.info("  -> Oh My Pi (omp) coding harness is not installed.")
            if self._should_install():
                logger.info("Installing Oh My Pi (omp) coding harness...")
                try:
                    npm_exe = get_npm_exe()
                    _run_silent(
                        [npm_exe, "install", "--prefix", str(AI_WORKSPACE_DIR), "@oh-my-pi/pi-coding-agent"],
                        shell=NPM_SHELL,
                        check=True,
                        env=sub_env
                    )
                    logger.info("  -> Oh My Pi coding harness installed successfully.")
                    # Sync trampolines to node_bin if available
                    if (self.local_bin / "omp.cmd").exists() and self.node_bin.exists():
                        try:
                            shutil.copy2(self.local_bin / "omp.cmd", self.node_bin / "omp.cmd")
                            if (self.local_bin / "omp.ps1").exists():
                                shutil.copy2(self.local_bin / "omp.ps1", self.node_bin / "omp.ps1")
                        except Exception:
                            pass
                    self.omp_exe = self._resolve_executable()
                    setup_omp_mcp_config(self.omp_home)
                    self.ensure_extensions_and_skills()
                    return True
                except subprocess.CalledProcessError as e:
                    logger.error(f"  -> Failed to install Oh My Pi: {e}")
                    return False
            else:
                logger.info("  -> Skipping Oh My Pi installation.")
                return False

    def start(self, llm_base_url, active_model=None, launch_tui=True):
        omp_home_resolved = self.omp_home.resolve()
        omp_agent_dir = omp_home_resolved / "agent"
        omp_agent_dir.mkdir(parents=True, exist_ok=True)

        # Auto-configure models, mcp, extensions, and auth specifically for OMP
        setup_omp_models_and_auth(
            omp_home_resolved,
            llm_base_url,
            active_model,
            use_omniroute=config.use_omniroute_proxy,
            omniroute_model=config.omniroute_default_model
        )
        setup_omp_mcp_config(omp_home_resolved)
        self.ensure_extensions_and_skills()

        model_name = resolve_model_name(active_model)

        # Pre-evaluate OMP context into KV cache
        from local_ai_launcher.context_preloader import preload_harness_context
        preload_harness_context(llm_base_url, model_name, harness_name="omp")

        if not launch_tui:
            logger.info("  -> Oh My Pi coding harness configured for local use (headless mode).")
            return

        omp_args = f"--provider openai --model {model_name}"
        omp_bin = f"& '{self.omp_exe}'" if self.omp_exe.exists() else "omp"
        omp_cmd = f"{omp_bin} {omp_args}"

        omp_env_dict = {
            'OPENAI_BASE_URL': llm_base_url,
            'OPENAI_API_BASE': llm_base_url,
            'OPENAI_API_KEY': 'dummy',
            'HOME': str(AI_WORKSPACE_DIR),
            'USERPROFILE': str(omp_home_resolved.parent),
            'OMP_HOME': str(omp_home_resolved),
            'PI_CONFIG_DIR': omp_home_resolved.name,
            'PI_CODING_AGENT_DIR': str(omp_agent_dir),
            'PI_SKIP_VERSION_CHECK': '1',
            'npm_config_allow_git': 'all',
            'FORCE_COLOR': '1',
        }
        env_setup = build_ps_env(omp_env_dict, harness_name="omp")
        logger.info("Starting Oh My Pi coding agent...")
        try:
            omp_process = start_in_wt_tab(
                f'powershell.exe -NoExit -Command "chcp 65001 | Out-Null; [console]::InputEncoding=[console]::OutputEncoding=[System.Text.Encoding]::UTF8; {env_setup}{omp_cmd}"',
                cwd=str(config.harness_start_dir),
                title="Oh My Pi"
            )
            self.processes.append(('Oh My Pi coding agent', omp_process, self.TRACK_CONSOLE_CLOSE))
        except Exception as e:
            logger.error(f"Error starting Oh My Pi: {e}")

