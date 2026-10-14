"""
LiteLLM Universal Proxy Engine.
Translates Anthropic Messages API specification (/v1/messages) to OpenAI API
specification (/v1/chat/completions) across all local engines (llama-server,
TabbyAPI, FreeToken, Strata, OmniRoute, LocalAI).
"""

import os
import sys
import time
import json
import shutil
import logging
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

from local_ai_launcher.config import (
    config,
    AI_WORKSPACE_DIR,
    CONFIG_DIR,
    prompt_choice,
)
from local_ai_launcher.paths import LITELLM_PORT, LITELLM_CONFIG_FILE
from local_ai_launcher.process import (
    force_kill_port,
    wait_for_port,
    is_port_in_use,
    start_process,
    kill_process_tree,
)
from local_ai_launcher.update_frequency import (
    should_update_component,
    record_successful_update,
)

logger = logging.getLogger("service_launcher")

_active_litellm_process = None


def _get_workspace_python() -> str:
    """Finds the most appropriate python interpreter in workspace venvs or current sys.executable."""
    exe_name = "python.exe" if os.name == "nt" else "python"
    bin_sub = "Scripts" if os.name == "nt" else "bin"

    # 1. Active sys.prefix virtual environment
    active_py = Path(sys.prefix) / bin_sub / exe_name
    if active_py.exists():
        return str(active_py)

    # 2. Workspace .venv or venv
    for v_name in (".venv", "venv"):
        v_py = AI_WORKSPACE_DIR / v_name / bin_sub / exe_name
        if v_py.exists():
            return str(v_py)

    return sys.executable


def is_litellm_installed() -> bool:
    """Checks if litellm and litellm proxy dependencies are installed."""
    try:
        import litellm.proxy.proxy_server  # noqa: F401
        return True
    except (ImportError, ModuleNotFoundError):
        pass

    exe_name = "litellm.exe" if os.name == "nt" else "litellm"
    bin_sub = "Scripts" if os.name == "nt" else "bin"

    # Active environment
    if (Path(sys.prefix) / bin_sub / exe_name).exists():
        return True

    # Workspace virtual environments
    for v_name in (".venv", "venv"):
        v_path = AI_WORKSPACE_DIR / v_name
        if (v_path / bin_sub / exe_name).exists():
            return True
        lib_dir = v_path / ("Lib" if os.name == "nt" else "lib")
        if (lib_dir / "site-packages" / "litellm").exists():
            return True
        for p in lib_dir.glob("python*/site-packages/litellm"):
            if p.exists():
                return True

    return bool(shutil.which("litellm"))


def get_litellm_version() -> Optional[str]:
    """Retrieves installed LiteLLM version."""
    try:
        import litellm
        return getattr(litellm, "__version__", None)
    except Exception:
        return None


def get_pip_install_cmd(*args: str) -> List[str]:
    """Resolves fast uv pip or standard python pip install command."""
    uv_exe = "uv.exe" if os.name == "nt" else "uv"
    uv_bin = AI_WORKSPACE_DIR / "uv-bin" / uv_exe
    uv_which = shutil.which("uv")
    uv_path = str(uv_bin) if uv_bin.exists() else (uv_which or None)

    py_exe = _get_workspace_python()

    if uv_path:
        cmd = [uv_path, "pip", "install"]
        if py_exe:
            cmd.extend(["--python", py_exe])
        cmd.extend(args)
        return cmd
    cmd = [py_exe, "-m", "pip", "install"]
    cmd.extend(args)
    return cmd


def check_litellm() -> bool:
    """Verifies LiteLLM installation and executes updates according to schedule."""
    logger.info("\nChecking for LiteLLM universal proxy engine...")
    installed = is_litellm_installed()

    if installed:
        v_str = get_litellm_version() or "installed"
        logger.info(f"  -> LiteLLM is already installed ({v_str}).")

        update_behavior = getattr(config, "litellm_update_behavior", "never_update")
        update_frequency_hrs = getattr(config, "engine_update_frequency_hrs", {}).get(
            "litellm", getattr(config, "update_frequency_hrs", 168)
        )
        force = getattr(config, "force_update", False)

        do_update = should_update_component(
            component_name="litellm",
            category="engine",
            behavior=update_behavior,
            frequency=update_frequency_hrs,
            force=force,
            prompt_if_existing=True,
        )

        if do_update:
            logger.info("Updating LiteLLM via pip...")
            cmd = get_pip_install_cmd("--upgrade")
            cmd.append("litellm[proxy]")
            try:
                subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                logger.info("  -> LiteLLM updated successfully.")
                record_successful_update("engine:litellm")
                return True
            except Exception as e:
                logger.error(f"  -> Failed to update LiteLLM: {e}")
                return False
        return True
    else:
        logger.info("  -> LiteLLM is not installed.")
        install_behavior = getattr(config, "litellm_install_behavior", "always_install")

        if install_behavior == "never_install":
            logger.info("  -> Skipping LiteLLM installation as per configuration.")
            return False
        elif install_behavior == "always_install":
            logger.info("Installing LiteLLM into workspace environment...")
            cmd = get_pip_install_cmd("litellm[proxy]")
            try:
                subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                logger.info("  -> LiteLLM installed successfully.")
                return True
            except Exception as e:
                logger.error(f"  -> Failed to install LiteLLM: {e}")
                return False
        else:  # 'existing'
            if prompt_choice("Do you want to install LiteLLM universal proxy? (y/n): ", default="y"):
                logger.info("Installing LiteLLM...")
                cmd = get_pip_install_cmd("litellm[proxy]")
                try:
                    subprocess.run(
                        cmd,
                        check=True,
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                    )
                    logger.info("  -> LiteLLM installed successfully.")
                    return True
                except Exception as e:
                    logger.error(f"  -> Failed to install LiteLLM: {e}")
                    return False
            else:
                logger.info("  -> Skipping LiteLLM installation.")
                return False


def generate_litellm_config(
    config_path: Optional[Path] = None,
    target_llm_url: str = "http://127.0.0.1:8080",
    active_model: Optional[str] = None
) -> Path:
    """Generates a dynamic LiteLLM YAML configuration file routing Anthropic models to OpenAI backend."""
    cfg_file = Path(config_path) if config_path else (getattr(config, "litellm_config_file", None) or LITELLM_CONFIG_FILE)
    cfg_file.parent.mkdir(parents=True, exist_ok=True)

    base = target_llm_url.rstrip("/")
    api_base = base if base.endswith("/v1") else f"{base}/v1"
    target_model = active_model or getattr(config, "default_model", "default") or "default"
    dummy_key = getattr(config, "claude_api_key", "dummy-token") or "dummy-token"

    configured_model = getattr(config, "claude_model", "claude-sonnet-4-5") or "claude-sonnet-4-5"

    model_list = [
        {
            "model_name": configured_model,
            "litellm_params": {
                "model": f"openai/{target_model}",
                "api_base": api_base,
                "api_key": dummy_key,
            }
        },
        {
            "model_name": "claude-sonnet-4-5",
            "litellm_params": {
                "model": f"openai/{target_model}",
                "api_base": api_base,
                "api_key": dummy_key,
            }
        },
        {
            "model_name": "claude-3-7-sonnet",
            "litellm_params": {
                "model": f"openai/{target_model}",
                "api_base": api_base,
                "api_key": dummy_key,
            }
        },
        {
            "model_name": "claude-3-5-sonnet",
            "litellm_params": {
                "model": f"openai/{target_model}",
                "api_base": api_base,
                "api_key": dummy_key,
            }
        },
        {
            "model_name": "claude-*",
            "litellm_params": {
                "model": f"openai/{target_model}",
                "api_base": api_base,
                "api_key": dummy_key,
            }
        },
        {
            "model_name": "*",
            "litellm_params": {
                "model": f"openai/{target_model}",
                "api_base": api_base,
                "api_key": dummy_key,
            }
        }
    ]

    config_dict = {
        "model_list": model_list,
        "general_settings": {
            "master_key": dummy_key,
        },
        "litellm_settings": {
            "drop_params": True,
            "set_verbose": False,
            "use_chat_completions_url_for_anthropic_messages": True,
        }
    }

    try:
        import yaml
        content = yaml.safe_dump(config_dict, sort_keys=False)
    except Exception:
        content = json.dumps(config_dict, indent=2)

    # Atomic write
    tmp_path = cfg_file.with_suffix(".yaml.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(content)
    try:
        os.replace(tmp_path, cfg_file)
    except Exception:
        shutil.copyfile(tmp_path, cfg_file)
        try:
            tmp_path.unlink()
        except OSError:
            pass

    return cfg_file


def start_litellm_proxy(
    processes: list,
    target_llm_url: str = "http://127.0.0.1:8080",
    active_model: Optional[str] = None,
    port: Optional[int] = None
) -> Optional[str]:
    """Starts LiteLLM universal proxy translating Anthropic Messages API -> OpenAI API."""
    global _active_litellm_process

    if not getattr(config, "use_litellm_proxy", True):
        logger.debug("LiteLLM proxy disabled by configuration.")
        return None

    effective_port = port or getattr(config, "litellm_port", LITELLM_PORT)
    proxy_url = f"http://127.0.0.1:{effective_port}"

    # Check if LiteLLM is already listening and responsive on this port
    if is_port_in_use(effective_port):
        if _active_litellm_process and _active_litellm_process.poll() is None:
            logger.info(f"  -> LiteLLM proxy already running on {proxy_url}.")
            return proxy_url
        logger.info(f"Port {effective_port} in use; terminating stale process...")
        force_kill_port(effective_port)
        time.sleep(0.5)

    if not is_litellm_installed():
        if not check_litellm():
            logger.warning("LiteLLM is not available; falling back to direct engine routing.")
            return None

    cfg_file = generate_litellm_config(
        target_llm_url=target_llm_url,
        active_model=active_model
    )

    py_bin = _get_workspace_python()

    cmd = [
        py_bin,
        "-m",
        "litellm.proxy.proxy_cli",
        "--config",
        str(cfg_file),
        "--port",
        str(effective_port),
        "--host",
        "127.0.0.1",
    ]
    if getattr(config, "show_launcher_logging", False):
        cmd.append("--detailed_debug")

    logger.info(f"Starting LiteLLM Universal Proxy on port {effective_port} (Backend: {target_llm_url})...")
    try:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        env["LITELLM_USE_CHAT_COMPLETIONS_URL_FOR_ANTHROPIC_MESSAGES"] = "True"
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0

        proc = subprocess.Popen(
            cmd,
            cwd=str(AI_WORKSPACE_DIR),
            env=env,
            creationflags=creationflags,
            stdout=subprocess.DEVNULL if not getattr(config, "show_launcher_logging", False) else None,
            stderr=subprocess.DEVNULL if not getattr(config, "show_launcher_logging", False) else None,
        )

        _active_litellm_process = proc
        processes.append(("LiteLLM Proxy", proc, False))

        if wait_for_port(effective_port, timeout=12.0):
            logger.info(f"  -> LiteLLM Universal Proxy active at {proxy_url}.")
            return proxy_url
        else:
            logger.warning(f"  -> LiteLLM proxy started but port {effective_port} did not respond within 12s.")
            return None
    except Exception as e:
        logger.error(f"Error starting LiteLLM proxy: {e}")
        return None


def stop_litellm_proxy() -> None:
    """Stops the active LiteLLM proxy process and cleans up port."""
    global _active_litellm_process
    if _active_litellm_process is not None:
        try:
            kill_process_tree(_active_litellm_process)
        except Exception:
            pass
        _active_litellm_process = None

    port = getattr(config, "litellm_port", LITELLM_PORT)
    try:
        force_kill_port(port)
    except Exception:
        pass
