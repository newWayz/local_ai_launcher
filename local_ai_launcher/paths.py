"""
Filesystem path resolution and topology constants for local_ai_launcher.
Decoupled from user configuration to avoid circular dependencies.
"""
import os
import sys
import shutil
from pathlib import Path
from typing import Optional

AI_WORKSPACE_DIR = Path(os.environ.get("AI_WORKSPACE_DIR", Path(__file__).resolve().parent.parent))
LOCAL_AI_LAUNCHER_DIR = Path(__file__).resolve().parent

# Consolidated workspace paths
CONFIG_DIR = AI_WORKSPACE_DIR / "config"
PROJECTS_DIR = AI_WORKSPACE_DIR / "projects"
HERMES_DIR = AI_WORKSPACE_DIR / ".hermes"
PI_DIR = AI_WORKSPACE_DIR / ".pi"
DSH_DIR = AI_WORKSPACE_DIR / ".dsh"
T3_DIR = AI_WORKSPACE_DIR / ".t3"
ORCA_DIR = AI_WORKSPACE_DIR / ".orca"
ORCA_DATA_DIR = AI_WORKSPACE_DIR / ".orca" / "data"
CLAUDE_DIR = AI_WORKSPACE_DIR / ".claude"
CLAUDE_DESKTOP_DIR = AI_WORKSPACE_DIR / "claude-desktop"
GITHUB_DIR = AI_WORKSPACE_DIR / "github"
AGY_IDE_DIR = AI_WORKSPACE_DIR / "agy-ide"
AGY_DESKTOP_DIR = AI_WORKSPACE_DIR / "agy-desktop"
OPENLUMARA_DIR = AI_WORKSPACE_DIR / "openlumara"
LOCALAI_MODELS_DIR = AI_WORKSPACE_DIR / "localai" / "models"
LOCALAI_REPO_DIR = AI_WORKSPACE_DIR / "localai_repo"
LOCALAI_BIN_DIR = AI_WORKSPACE_DIR / "tools" / "localai"
LOCALAI_EXE = LOCALAI_BIN_DIR / ("local-ai.exe" if sys.platform == 'win32' else "local-ai")

STRATA_DIR = AI_WORKSPACE_DIR / "strata"
STRATA_ENGINE_DIR = STRATA_DIR / "engine"
STRATA_EXE = STRATA_ENGINE_DIR / ("strata.exe" if sys.platform == 'win32' else "strata")

_LLAMA_EXE_NAME = "llama-server.exe" if sys.platform == 'win32' else "llama-server"
SCRIPT_DIR = AI_WORKSPACE_DIR
INSTALL_DIR_LLAMA_CPP = str(AI_WORKSPACE_DIR / "llama-cpp")
EXE_PATH_LLAMA_CPP = os.path.join(INSTALL_DIR_LLAMA_CPP, _LLAMA_EXE_NAME)
INSTALL_DIR_LLAMA_CPP_BLACKWELL = str(AI_WORKSPACE_DIR / "llama-cpp-blackwell")
EXE_PATH_LLAMA_CPP_BLACKWELL = os.path.join(INSTALL_DIR_LLAMA_CPP_BLACKWELL, _LLAMA_EXE_NAME)
INSTALL_DIR_LLAMA_CPP_BRUNO = str(AI_WORKSPACE_DIR / "llama-cpp-bruno")
EXE_PATH_LLAMA_CPP_BRUNO = os.path.join(INSTALL_DIR_LLAMA_CPP_BRUNO, _LLAMA_EXE_NAME)
INSTALL_DIR_LLAMA_CPP_HUANG = str(AI_WORKSPACE_DIR / "llama-cpp-huang")
EXE_PATH_LLAMA_CPP_HUANG = os.path.join(INSTALL_DIR_LLAMA_CPP_HUANG, _LLAMA_EXE_NAME)

_candidate_models_ini = LOCAL_AI_LAUNCHER_DIR / "models.ini"
MODELS_PRESET_FILE = _candidate_models_ini if _candidate_models_ini.exists() else (AI_WORKSPACE_DIR / "models.ini")

_candidate_ft_ini = LOCAL_AI_LAUNCHER_DIR / "freetokens_models.ini"
FREETOKENS_MODELS_PRESET_FILE = _candidate_ft_ini if _candidate_ft_ini.exists() else (AI_WORKSPACE_DIR / "freetokens_models.ini")

_candidate_strata_json = LOCAL_AI_LAUNCHER_DIR / "strata_models.json"
STRATA_MODELS_PRESET_FILE = _candidate_strata_json if _candidate_strata_json.exists() else (AI_WORKSPACE_DIR / "strata_models.json")

TMP_DIR = AI_WORKSPACE_DIR / "tmp"
MODELS_LOCAL_OVERRIDE_FILE = AI_WORKSPACE_DIR / "models.local.ini"
RESOLVED_MODELS_PRESET_FILE = TMP_DIR / "resolved_models.ini"

root_dir = AI_WORKSPACE_DIR
OMNIROUTE_PORT = 20128
LITELLM_PORT = 4000
LITELLM_CONFIG_FILE = CONFIG_DIR / "litellm_config.yaml"

def get_basename(s: str) -> str:
    """Extract clean basename from a model identifier or path."""
    return str(s).replace('\\', '/').split('/')[-1]


def atomic_write_json(path, data, indent: int = 2) -> None:
    """Atomically write data as JSON using a temporary file in the same directory."""
    import json
    import tempfile
    target_path = Path(path).resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(target_path.parent), text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent)
        os.replace(tmp_path, target_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def atomic_write_text(path, text: str, encoding: str = "utf-8") -> None:
    """Atomically write text content using a temporary file in the same directory."""
    import tempfile
    target_path = Path(path).resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(target_path.parent), text=True)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(text)
        os.replace(tmp_path, target_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def get_donsetch_binary_path() -> Optional[Path]:
    """Portably locate the donsetch binary across Pi npm package, workspace bins, or PATH."""
    bin_name = "donsetch.exe" if sys.platform == 'win32' else "donsetch"
    candidates = [
        PI_DIR / "agent" / "npm" / "node_modules" / "donsetch" / "binaries" / bin_name,
        AI_WORKSPACE_DIR / "node" / bin_name,
        AI_WORKSPACE_DIR / "node_modules" / ".bin" / bin_name,
        DSH_DIR / "cache" / "donsetch" / "dsh" / "bin" / "4.7.4" / bin_name,
        DSH_DIR / "cache" / "donsetch" / bin_name,
    ]
    try:
        candidates.append(Path.home() / ".cargo" / "bin" / bin_name)
    except Exception:
        pass

    for c in candidates:
        if c.exists() and c.is_file():
            return c
    which_bin = shutil.which(bin_name)
    if which_bin:
        return Path(which_bin)
    return None


__all__ = [
    "AI_WORKSPACE_DIR",
    "LOCAL_AI_LAUNCHER_DIR",
    "CONFIG_DIR",
    "PROJECTS_DIR",
    "HERMES_DIR",
    "PI_DIR",
    "DSH_DIR",
    "T3_DIR",
    "ORCA_DIR",
    "ORCA_DATA_DIR",
    "CLAUDE_DIR",
    "CLAUDE_DESKTOP_DIR",
    "GITHUB_DIR",
    "AGY_IDE_DIR",
    "AGY_DESKTOP_DIR",
    "OPENLUMARA_DIR",
    "LOCALAI_MODELS_DIR",
    "LOCALAI_REPO_DIR",
    "LOCALAI_BIN_DIR",
    "LOCALAI_EXE",
    "SCRIPT_DIR",
    "INSTALL_DIR_LLAMA_CPP",
    "EXE_PATH_LLAMA_CPP",
    "INSTALL_DIR_LLAMA_CPP_BLACKWELL",
    "EXE_PATH_LLAMA_CPP_BLACKWELL",
    "INSTALL_DIR_LLAMA_CPP_BRUNO",
    "EXE_PATH_LLAMA_CPP_BRUNO",
    "INSTALL_DIR_LLAMA_CPP_HUANG",
    "EXE_PATH_LLAMA_CPP_HUANG",
    "MODELS_PRESET_FILE",
    "FREETOKENS_MODELS_PRESET_FILE",
    "STRATA_MODELS_PRESET_FILE",
    "MODELS_LOCAL_OVERRIDE_FILE",
    "RESOLVED_MODELS_PRESET_FILE",
    "TMP_DIR",
    "root_dir",
    "OMNIROUTE_PORT",
    "LITELLM_PORT",
    "LITELLM_CONFIG_FILE",
    "get_basename",
    "atomic_write_json",
    "atomic_write_text",
    "get_donsetch_binary_path",
]
