"""
Aside AI Browser Backend & Engine Configuration Bridge.

Connects the Aside browser (aside.com) to the local AI engine (Unified Gateway on Port 8080)
and provisions MCP server integration, CLI routing, and session state for all coding agent harnesses.
Enables interactive web browsing, DOM manipulation, form submission, authenticated site access,
and visual problem-solving without GPU token leakage.
"""

import os
import sys
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional

from local_ai_launcher.paths import (
    AI_WORKSPACE_DIR,
    get_aside_binary_path,
    get_aside_app_path,
    atomic_write_json,
)
from local_ai_launcher.models import resolve_model_name
from local_ai_launcher.process import start_process

logger = logging.getLogger("aside_backend")

ASIDE_USER_HOME = Path.home() / ".aside"


def get_aside_config_dir(account_id: str = "0") -> Path:
    """Retrieves the configuration and settings directory for the specified Aside account ID."""
    return ASIDE_USER_HOME / "u" / str(account_id)


def get_aside_accounts() -> List[str]:
    """Retrieves all registered Aside account IDs, defaulting to ['0']."""
    accounts_file = ASIDE_USER_HOME / "accounts.json"
    if accounts_file.exists():
        try:
            with open(accounts_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                accounts = data.get("accounts", [])
                acc_ids = [str(a.get("id")) for a in accounts if "id" in a]
                if acc_ids:
                    return acc_ids
        except Exception as e:
            logger.debug(f"Aside accounts.json read notice: {e}")
    return ["0"]


def setup_aside_local_engine(
    llm_base_url: str,
    active_model: Optional[str] = None,
    ctx_size: Optional[int] = None,
    account_ids: Optional[List[str]] = None,
) -> bool:
    """
    Configures Aside browser profiles to connect directly to the running Local AI engine.
    Registers a custom model endpoint in settings.json pointing to the Unified Gateway (/v1)
    and ensures matching credentials.json entries.
    """
    if not llm_base_url:
        logger.warning("No LLM base URL provided for Aside engine configuration.")
        return False

    if not is_aside_installed() and account_ids is None and not (ASIDE_USER_HOME / "accounts.json").exists():
        logger.debug("Aside browser is not installed; skipping local engine configuration.")
        return False

    norm_url = str(llm_base_url).rstrip("/")
    if not norm_url.endswith("/v1"):
        norm_url = f"{norm_url}/v1"

    model_name = resolve_model_name(active_model) if active_model else "default"

    # Context window calculation
    if ctx_size is None:
        try:
            from local_ai_launcher.models import compute_compaction_params
            compaction = compute_compaction_params(model_name)
            ctx_size = max(512, int(compaction.ctx_size))
        except Exception:
            ctx_size = 131072

    max_tokens = min(ctx_size, max(4096, ctx_size // 8))

    # Detect vision capability for multimodal DOM/screenshot analysis
    is_vision = False
    try:
        from local_ai_launcher.models import get_available_preset_models
        models_info = get_available_preset_models(only_installed=False, active_model=model_name)
        matched = next((m for m in models_info if m.get("id") == model_name), None)
        if matched and "image" in matched.get("input", []):
            is_vision = True
    except Exception:
        pass
    if not is_vision:
        is_vision = any(k in str(model_name).lower() for k in ("vision", "-vl", "_vl", "mmproj", "gemma-4"))

    model_input = ["text", "image"] if is_vision else ["text"]

    target_accounts = account_ids or get_aside_accounts()
    success = False

    for acc_id in target_accounts:
        account_dir = get_aside_config_dir(acc_id)
        account_dir.mkdir(parents=True, exist_ok=True)

        settings_file = account_dir / "settings.json"
        credentials_file = account_dir / "credentials.json"

        settings_data: Dict[str, Any] = {}
        if settings_file.exists():
            try:
                with open(settings_file, "r", encoding="utf-8") as f:
                    settings_data = json.load(f)
            except Exception as e:
                logger.warning(f"Could not parse Aside settings.json for account {acc_id}: {e}")
                settings_data = {}

        if not isinstance(settings_data, dict):
            settings_data = {}

        custom_models: List[Dict[str, Any]] = settings_data.setdefault("customModels", [])
        if not isinstance(custom_models, list):
            custom_models = []
            settings_data["customModels"] = custom_models

        # Model specification conforming to Aside's custom provider schema
        local_model_entry = {
            "id": "local-engine",
            "name": f"Local AI Engine ({model_name})",
            "model": model_name,
            "api": "openai-completions",
            "provider": "local",
            "reasoning": False,
            "contextWindow": ctx_size,
            "maxTokens": max_tokens,
            "baseUrl": norm_url,
            "input": model_input,
            "cost": {
                "input": 0,
                "output": 0,
                "cacheRead": 0,
                "cacheWrite": 0,
            },
            "headers": {
                "Authorization": "Bearer dummy",
            },
        }

        # Replace existing local provider entry or append
        existing_idx = next(
            (i for i, m in enumerate(custom_models) if isinstance(m, dict) and (m.get("provider") == "local" or m.get("id") == "local-engine")),
            None
        )
        if existing_idx is not None:
            custom_models[existing_idx] = local_model_entry
        else:
            custom_models.append(local_model_entry)

        # Set defaultModel to the local engine
        settings_data["defaultModel"] = {
            "provider": "local",
            "modelId": "local-engine",
            "thinkingLevel": "off",
            "fastMode": False,
        }

        # Keep CLI executions ephemeral so they do not clutter interactive tabs
        cli_settings = settings_data.setdefault("cli", {})
        if isinstance(cli_settings, dict):
            cli_settings["ephemeral"] = True

        try:
            atomic_write_json(settings_file, settings_data)
            logger.info(f"Configured Aside browser account '{acc_id}' -> Local Engine {norm_url} ({model_name}).")
            success = True
        except Exception as e:
            logger.warning(f"Failed to update Aside settings.json for account {acc_id}: {e}")

        # Update credentials.json to recognize local provider
        cred_data: Dict[str, Any] = {}
        if credentials_file.exists():
            try:
                with open(credentials_file, "r", encoding="utf-8") as f:
                    cred_data = json.load(f)
            except Exception:
                cred_data = {}
        if not isinstance(cred_data, dict):
            cred_data = {}
        cred_data["local"] = "dummy"
        try:
            atomic_write_json(credentials_file, cred_data)
        except Exception as e:
            logger.debug(f"Credentials write notice for account {acc_id}: {e}")

    return success


def get_aside_mcp_server_config(require_installed: bool = True) -> Optional[Dict[str, Any]]:
    """
    Returns standard MCP server definition for Aside browser over stdio,
    compatible with Pi agent mcp.json, Claude Code .mcp.json, and Claude Desktop.
    Returns None if Aside is not installed and require_installed is True.
    """
    bin_path = get_aside_binary_path()
    if require_installed and not bin_path:
        which_bin = shutil.which("aside") or shutil.which("aside.exe")
        if not which_bin:
            return None
        bin_path = Path(which_bin)

    bin_cmd = str(bin_path) if bin_path else "aside"

    if sys.platform == "win32":
        return {
            "type": "stdio",
            "command": "cmd.exe",
            "args": ["/d", "/c", bin_cmd, "mcp"],
            "env": {},
        }
    return {
        "type": "stdio",
        "command": bin_cmd,
        "args": ["mcp"],
        "env": {},
    }


def is_aside_installed() -> bool:
    """Returns True if Aside CLI or Aside Desktop Application is installed."""
    return get_aside_binary_path() is not None or get_aside_app_path() is not None


def start_aside_browser(
    processes: List,
    llm_base_url: str,
    active_model: Optional[str] = None,
    open_window: bool = True,
) -> bool:
    """
    Starts the Aside Browser desktop application or CLI connected to the local AI engine.
    """
    setup_aside_local_engine(llm_base_url, active_model=active_model)

    app_exe = get_aside_app_path()
    cli_exe = get_aside_binary_path()

    if app_exe and app_exe.exists():
        logger.info(f"Starting Aside Browser application ({app_exe})...")
        proc = start_process(
            "Aside Browser",
            f'& "{app_exe}"',
            str(AI_WORKSPACE_DIR),
            show_window=open_window
        )
        if proc:
            processes.append(("Aside Browser", proc, False))
            return True

    if cli_exe and cli_exe.exists():
        logger.info(f"Aside desktop app not found; Aside CLI available at {cli_exe}.")
        return True

    logger.warning("Aside Browser executable not found on system.")
    return False
