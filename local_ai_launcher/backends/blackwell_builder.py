"""
Blackwell Engine Unified Manager & Dispatcher.
Dispatches modularly between Raymond Huang's Adaptive-KV engine (default)
and Bruno Passini's 256K Hybrid-KV engine based on config.blackwell_engine.
"""
import os
import sys
import logging
from typing import Optional, Dict

from local_ai_launcher.config import config, logger
from local_ai_launcher.backends.bruno_builder import (
    BRUNO_REPO_URL,
    BRUNO_BRANCH,
    BLACKWELL_REPO_URL,
    BLACKWELL_BRANCH,
    FORK_INFO_BRUNO_FILE,
    FORK_INFO_BLACKWELL_FILE,
    BRUNO_RUNTIME_ENV,
    BLACKWELL_RUNTIME_ENV,
    get_bruno_runtime_env,
    get_blackwell_runtime_env,
    resolve_bruno_server_exe,
    get_installed_bruno_fork_info,
    get_installed_blackwell_fork_info,
    get_local_bruno_version,
    get_local_blackwell_version,
    check_build_prerequisites,
    patch_bruno_source,
    patch_blackwell_source,
    build_bruno_llamacpp,
    build_blackwell_llamacpp,
    check_bruno_llamacpp,
    is_bruno_model_allowed,
)
from local_ai_launcher.backends.huang_builder import (
    HUANG_REPO_URL,
    HUANG_BRANCH,
    FORK_INFO_HUANG_FILE,
    resolve_huang_server_exe,
    get_installed_huang_fork_info,
    get_local_huang_version,
    build_huang_llamacpp,
    check_huang_llamacpp,
)


def resolve_blackwell_engine() -> str:
    """Returns the configured Blackwell engine ('huang' or 'bruno'). Defaults to 'huang'."""
    engine = getattr(config, "blackwell_engine", "huang")
    if not engine:
        engine = "huang"
    clean = str(engine).lower().strip()
    return "bruno" if "bruno" in clean else "huang"


def resolve_blackwell_server_exe() -> Optional[str]:
    """Resolves llama-server executable for the currently active Blackwell engine."""
    engine = resolve_blackwell_engine()
    if engine == "huang":
        exe = resolve_huang_server_exe()
        if exe and os.path.exists(exe):
            return exe
        # Fallback to Bruno binary if Huang is not yet compiled
        return resolve_bruno_server_exe()
    else:
        return resolve_bruno_server_exe()


def check_blackwell_llamacpp() -> bool:
    """Checks or builds the currently active Blackwell engine according to configuration."""
    engine = resolve_blackwell_engine()
    if engine == "huang":
        # Check Huang engine; if Huang binary is not compiled yet, check if Bruno binary is available and compatible
        huang_ok = check_huang_llamacpp()
        if not huang_ok:
            active_mod = (
                getattr(config, "active_model", None)
                or getattr(config, "requested_model", None)
                or getattr(config, "default_model", "")
            )
            if active_mod and not is_bruno_model_allowed(active_mod):
                logger.info(f"  -> Model '{active_mod}' requires Huang engine (unsupported on Bruno). Fallback will use safe official llama-server.")
                return False
            if resolve_bruno_server_exe():
                logger.info("  -> Falling back to existing Bruno Blackwell llama-server binary.")
                return True
        return huang_ok
    else:
        return check_bruno_llamacpp()


__all__ = [
    "resolve_blackwell_engine",
    "resolve_blackwell_server_exe",
    "check_blackwell_llamacpp",
    "BLACKWELL_REPO_URL",
    "BLACKWELL_BRANCH",
    "FORK_INFO_BLACKWELL_FILE",
    "BLACKWELL_RUNTIME_ENV",
    "get_blackwell_runtime_env",
    "get_installed_blackwell_fork_info",
    "get_local_blackwell_version",
    "check_build_prerequisites",
    "patch_blackwell_source",
    "build_blackwell_llamacpp",
    "resolve_bruno_server_exe",
    "check_bruno_llamacpp",
    "build_bruno_llamacpp",
    "resolve_huang_server_exe",
    "check_huang_llamacpp",
    "build_huang_llamacpp",
]
