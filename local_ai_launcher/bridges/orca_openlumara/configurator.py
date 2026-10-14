import os
import json
import shutil
import logging
import tempfile
import socket
import urllib.request
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger("service_launcher")

try:
    from local_ai_launcher.config import AI_WORKSPACE_DIR
    DEFAULT_WORKSPACE_DIR = AI_WORKSPACE_DIR
except Exception:
    DEFAULT_WORKSPACE_DIR = Path(os.environ.get("AI_WORKSPACE_DIR", Path(__file__).resolve().parents[3]))
DEFAULT_OPENLUMARA_DIR = DEFAULT_WORKSPACE_DIR / "openlumara"


def get_orca_profile_path() -> Optional[Path]:
    """Resolve the active Orca ADE profile settings file, prioritizing workspace-contained paths."""
    appdata = os.environ.get("APPDATA")
    if os.environ.get("PYTEST_CURRENT_TEST") and appdata:
        return Path(appdata) / "Orca" / "profiles" / "local-default" / "orca-data.json"

    candidates = [
        DEFAULT_WORKSPACE_DIR / "orca" / "data" / "profiles" / "local-default" / "orca-data.json",
        DEFAULT_WORKSPACE_DIR / "AppData" / "Roaming" / "orca" / "profiles" / "local-default" / "orca-data.json",
        DEFAULT_WORKSPACE_DIR / "AppData" / "Roaming" / "Orca" / "profiles" / "local-default" / "orca-data.json",
    ]

    if appdata:
        candidates.append(Path(appdata) / "orca" / "profiles" / "local-default" / "orca-data.json")
        candidates.append(Path(appdata) / "Orca" / "profiles" / "local-default" / "orca-data.json")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0] if candidates else None


def is_openlumara_configured_in_orca(profile_path: Optional[Path] = None) -> bool:
    """Check whether OpenLumara is already configured in Orca settings."""
    target_path = profile_path or get_orca_profile_path()
    if not target_path or not target_path.exists():
        return False

    try:
        data = json.loads(target_path.read_text(encoding="utf-8"))
        settings = data.get("settings", {})
        overrides = settings.get("agentCmdOverrides", {})
        return "openlumara" in overrides
    except Exception:
        return False


def setup_openlumara_config(
    lumara_dir: Optional[Path] = None,
    llm_base_url: Optional[str] = None,
    active_model: Optional[str] = None,
    mode: str = "cli",
) -> bool:
    """Configure OpenLumara's on-disk config.yml/config.yaml for local LLM engines (llama-server, OmniRoute, TabbyAPI, FreeToken).
    Mirrors hermes.py and pi.py config injection to ensure zero code modification to OpenLumara.
    """
    target_dir = Path(lumara_dir) if lumara_dir else DEFAULT_OPENLUMARA_DIR
    if not target_dir.exists():
        return False

    # 1. Resolve LLM base URL
    resolved_url = llm_base_url or os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
    if not resolved_url:
        def _port_open(port: int) -> bool:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(0.2)
                    return s.connect_ex(("127.0.0.1", port)) == 0
            except Exception:
                return False

        try:
            from local_ai_launcher.config import config
            localai_port = config.localai_port
        except Exception:
            localai_port = 8080

        if _port_open(20128):
            resolved_url = "http://localhost:20128/v1"
        elif _port_open(localai_port):
            resolved_url = f"http://localhost:{localai_port}/v1"
        elif _port_open(9931):
            resolved_url = "http://localhost:9931/v1"
        else:
            resolved_url = f"http://localhost:{localai_port}/v1"

    # 2. Resolve active model name
    model_name = active_model or os.environ.get("OPENAI_MODEL") or os.environ.get("MODEL")
    try:
        from local_ai_launcher.models import resolve_model_name
        model_name = resolve_model_name(model_name)
    except Exception:
        pass

    # If model name is not determined, try discovering currently loaded model from endpoint
    if not model_name or model_name in ("default", "*"):
        try:
            req = urllib.request.Request(f"{resolved_url}/models", headers={"User-Agent": "OpenLumara-Configurator"})
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                m_data = json.loads(resp.read().decode("utf-8"))
                if "data" in m_data and len(m_data["data"]) > 0:
                    model_name = m_data["data"][0].get("id") or model_name
        except Exception:
            pass

    if not model_name or model_name in ("default", "*"):
        try:
            from local_ai_launcher.config import config
            model_name = config.default_model
        except Exception:
            model_name = "default"

    # 3. Determine model specifications (context window, reasoning mode)
    ctx_size = 32768
    is_reasoning = False
    try:
        from local_ai_launcher.models import get_model_specs
        ctx_size, is_reasoning = get_model_specs(model_name)
    except Exception:
        pass

    if "freetoken" in str(model_name).lower() and ctx_size == 32768:
        ctx_size = 8192
    if any(k in str(model_name).lower() for k in ("think", "deepseek-r1", "reason")):
        is_reasoning = True
    if ctx_size <= 0:
        ctx_size = 32768

    max_output = min(ctx_size, max(8192, ctx_size // 8))

    # 4. Read existing configuration if present
    config_yml = target_dir / "config.yml"
    config_yaml = target_dir / "config.yaml"
    cfg: Dict[str, Any] = {}

    import yaml
    for f in [config_yml, config_yaml]:
        if f.exists():
            try:
                loaded = yaml.safe_load(f.read_text(encoding="utf-8"))
                if isinstance(loaded, dict) and loaded:
                    cfg = loaded
                    break
            except Exception as e:
                logger.warning(f"[BRIDGE] Notice loading {f}: {e}")

    # 5. Core settings
    core_sec = cfg.setdefault("core", {})
    core_sec.setdefault("data_folder", "data")
    core_sec.setdefault("auto_resume_chats", True)
    core_sec.setdefault("cmd_prefix", "/")
    core_sec.setdefault("tool_timeout", 30)

    # 6. API settings (conforming to local engine)
    api_sec = cfg.setdefault("api", {})
    api_sec["url"] = resolved_url
    api_sec["key"] = os.environ.get("OPENAI_API_KEY", "dummy")
    api_sec["max_context"] = ctx_size
    api_sec["max_output_tokens"] = max_output
    api_sec.setdefault("max_messages", 200)
    api_sec.setdefault("use_developer_role", False)
    api_sec.setdefault("custom_fields", {})

    # 7. Model settings
    model_sec = cfg.setdefault("model", {})
    model_sec["name"] = model_name
    model_sec["enable_thinking"] = is_reasoning
    model_sec["use_tools"] = True
    model_sec.setdefault("temperature", 0.7)
    model_sec.setdefault("keep_reasoning_in_context", True)
    model_sec.setdefault("only_preserve_reasoning_for_current_agentic_loop", True)
    model_sec.setdefault("reasoning_effort", "none")

    # 8. Channel settings: ensure cli and logger are enabled so self.channel is never None
    chan_sec = cfg.setdefault("channels", {})
    enabled_chans = chan_sec.get("enabled", [])
    if not isinstance(enabled_chans, list):
        enabled_chans = []
    disabled_chans = chan_sec.get("disabled", [])
    if not isinstance(disabled_chans, list):
        disabled_chans = []

    for req_chan in ["cli", "logger"]:
        if req_chan not in enabled_chans:
            enabled_chans.append(req_chan)
        if req_chan in disabled_chans:
            disabled_chans.remove(req_chan)

    if mode == "webui":
        if "webui" not in enabled_chans:
            enabled_chans.append("webui")
        if "webui" in disabled_chans:
            disabled_chans.remove("webui")
    else:
        if "webui" in enabled_chans:
            enabled_chans.remove("webui")
        if "webui" not in disabled_chans:
            disabled_chans.append("webui")

    chan_sec["enabled"] = enabled_chans
    chan_sec["disabled"] = disabled_chans

    chan_settings = chan_sec.setdefault("settings", {})
    logger_settings = chan_settings.setdefault("logger", {})
    logger_settings.setdefault("path", "lumara.log")

    # 9. Module settings: disable auto_backup to prevent upstream NoneType.log exception
    mod_sec = cfg.setdefault("modules", {})
    enabled_mods = mod_sec.get("enabled", [])
    if not isinstance(enabled_mods, list):
        enabled_mods = []
    disabled_mods = mod_sec.get("disabled", [])
    if not isinstance(disabled_mods, list):
        disabled_mods = []

    if "auto_backup" in enabled_mods:
        enabled_mods.remove("auto_backup")
    if "auto_backup" not in disabled_mods:
        disabled_mods.append("auto_backup")

    mod_sec["enabled"] = enabled_mods
    mod_sec["disabled"] = disabled_mods

    # 10. Atomic disk write
    try:
        yaml_content = yaml.safe_dump(cfg, default_flow_style=False, sort_keys=False)
        fd, tmp_file = tempfile.mkstemp(dir=str(target_dir), text=True)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(yaml_content)
        os.replace(tmp_file, config_yml)

        if config_yaml.exists():
            try:
                config_yaml.write_text(yaml_content, encoding="utf-8")
            except Exception:
                pass

        logger.info(f"[BRIDGE] Injected OpenLumara config -> URL: {resolved_url}, Model: {model_name}, Ctx: {ctx_size}, Thinking: {is_reasoning}")
        return True
    except Exception as e:
        logger.error(f"[BRIDGE] Error writing OpenLumara config: {e}")
        return False


def configure_orca_profile_for_openlumara(
    profile_path: Optional[Path] = None,
    llm_base_url: Optional[str] = None,
    active_model: Optional[str] = None,
    set_default_agent: Optional[bool] = None,
) -> bool:
    """Safely register OpenLumara into Orca ADE profile settings without touching other agents."""
    target_path = profile_path or get_orca_profile_path()
    if not target_path:
        logger.warning("[BRIDGE] Could not determine Orca profile path.")
        return False

    if set_default_agent is None:
        try:
            from local_ai_launcher.config import config
            set_default_agent = any(h in getattr(config, 'choose_harness', []) for h in ("openlumara", "lumara"))
        except Exception:
            set_default_agent = True

    try:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        data: Dict[str, Any] = {}

        if target_path.exists():
            try:
                data = json.loads(target_path.read_text(encoding="utf-8"))
            except Exception as read_err:
                logger.warning(f"[BRIDGE] Failed to parse existing orca-data.json: {read_err}. Creating fresh structure.")
                data = {}

        if not isinstance(data, dict):
            data = {}

        data.setdefault("schemaVersion", 1)
        settings = data.setdefault("settings", {})

        if set_default_agent:
            settings["defaultTuiAgent"] = "trae"
            disabled_tui = settings.get("disabledTuiAgents", [])
            if isinstance(disabled_tui, list) and "trae" in disabled_tui:
                disabled_tui.remove("trae")
                settings["disabledTuiAgents"] = disabled_tui
        elif settings.get("defaultTuiAgent") == "trae":
            settings.pop("defaultTuiAgent", None)

        cmd_overrides = settings.setdefault("agentCmdOverrides", {})
        cmd_overrides["openlumara"] = "openlumara --cli --disable_auto_installer"
        cmd_overrides["trae"] = "openlumara --cli --disable_auto_installer"

        default_args = settings.setdefault("agentDefaultArgs", {})
        default_args["openlumara"] = "--cli --disable_auto_installer"
        default_args["trae"] = ""

        # Resolve endpoint and model if not explicitly provided
        resolved_url = llm_base_url or os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
        if not resolved_url:
            def _port_open(port: int) -> bool:
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.settimeout(0.2)
                        return s.connect_ex(("127.0.0.1", port)) == 0
                except Exception:
                    return False
            try:
                from local_ai_launcher.config import config
                localai_port = config.localai_port
            except Exception:
                localai_port = 8080

            if _port_open(20128):
                resolved_url = "http://localhost:20128/v1"
            elif _port_open(localai_port):
                resolved_url = f"http://localhost:{localai_port}/v1"
            elif _port_open(9931):
                resolved_url = "http://localhost:9931/v1"
            else:
                resolved_url = f"http://localhost:{localai_port}/v1"

        resolved_model = active_model or os.environ.get("OPENAI_MODEL") or os.environ.get("MODEL")
        if not resolved_model:
            try:
                from local_ai_launcher.config import config
                resolved_model = config.default_model
            except Exception:
                resolved_model = "default"

        default_env = settings.setdefault("agentDefaultEnv", {})
        for agent_key in ["openlumara", "trae"]:
            lumara_env = default_env.setdefault(agent_key, {})
            lumara_env["OPENAI_BASE_URL"] = resolved_url
            lumara_env["OPENAI_API_BASE"] = resolved_url
            if resolved_model:
                lumara_env["OPENAI_MODEL"] = resolved_model
                lumara_env["MODEL"] = resolved_model
            lumara_env["HOME"] = str(DEFAULT_WORKSPACE_DIR)
            lumara_env["PI_HOME"] = str(DEFAULT_WORKSPACE_DIR / ".pi")
            lumara_env["PI_CODING_AGENT_DIR"] = str(DEFAULT_WORKSPACE_DIR / ".pi" / "agent")
            lumara_env["HF_HOME"] = str(DEFAULT_WORKSPACE_DIR / "models" / "huggingface")
            lumara_env["HF_HUB_CACHE"] = str(DEFAULT_WORKSPACE_DIR / "models" / "huggingface")
            lumara_env["npm_config_allow_git"] = "all"

        # Create backup before writing
        if target_path.exists():
            backup_path = target_path.with_suffix(".json.bak")
            try:
                shutil.copy2(target_path, backup_path)
            except Exception:
                pass

        from local_ai_launcher.paths import atomic_write_json
        atomic_write_json(target_path, data)
        logger.info(f"[BRIDGE] Successfully registered OpenLumara in Orca ADE profile: {target_path}")

        # Also keep OpenLumara on-disk configuration synced with current endpoint/model
        setup_openlumara_config(
            lumara_dir=DEFAULT_OPENLUMARA_DIR,
            llm_base_url=llm_base_url,
            active_model=active_model
        )

        return True

    except Exception as e:
        logger.error(f"[BRIDGE] Error configuring Orca profile: {e}")
        return False


def install_openlumara_wrappers(workspace_dir: Optional[Path] = None) -> bool:
    """Deploy executable wrapper scripts for OpenLumara on system and workspace PATH."""
    ws = workspace_dir or DEFAULT_WORKSPACE_DIR
    target_dirs = [
        ws / ".venv" / "Scripts",
        ws / "openlumara" / "venv" / "Scripts",
    ]

    cmd_script = f"""@echo off
setlocal enabledelayedexpansion
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
chcp 65001 >nul 2>&1

:: Prefer OpenLumara venv python if available
if exist "{ws}\\openlumara\\venv\\Scripts\\python.exe" (
    set "PY_BIN={ws}\\openlumara\\venv\\Scripts\\python.exe"
) else if exist "{ws}\\.venv\\Scripts\\python.exe" (
    set "PY_BIN={ws}\\.venv\\Scripts\\python.exe"
) else (
    set "PY_BIN=python"
)

"%PY_BIN%" -m local_ai_launcher.bridges.orca_openlumara.shim %*
"""

    ps1_script = f"""# OpenLumara PowerShell wrapper for Orca ADE & Windows Terminal
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::InputEncoding = [System.Text.Encoding]::UTF8

$pyBin = "{ws}\\openlumara\\venv\\Scripts\\python.exe"
if (-not (Test-Path $pyBin)) {{
    $pyBin = "{ws}\\.venv\\Scripts\\python.exe"
    if (-not (Test-Path $pyBin)) {{
        $pyBin = "python"
    }}
}}

& $pyBin -m local_ai_launcher.bridges.orca_openlumara.shim @args
"""

    success = False
    for d in target_dirs:
        try:
            d.mkdir(parents=True, exist_ok=True)
            (d / "openlumara.cmd").write_text(cmd_script, encoding="utf-8")
            (d / "openlumara.ps1").write_text(ps1_script, encoding="utf-8")
            (d / "openlumara.bat").write_text(cmd_script, encoding="utf-8")
            # Also deploy trae wrappers for Orca ADE integration
            (d / "trae.cmd").write_text(cmd_script, encoding="utf-8")
            (d / "trae.ps1").write_text(ps1_script, encoding="utf-8")
            (d / "trae.bat").write_text(cmd_script, encoding="utf-8")
            success = True
        except Exception as e:
            logger.warning(f"[BRIDGE] Could not write wrapper in {d}: {e}")

    if success:
        logger.info("[BRIDGE] OpenLumara and Trae wrapper scripts deployed successfully to workspace PATH.")
    return success
