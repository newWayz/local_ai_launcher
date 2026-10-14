import subprocess
import time
import sys
import os
import shutil
from pathlib import Path
import re
import json
import logging
from datetime import datetime
from typing import Optional, Dict

from local_ai_launcher.config import (
    config,
    logger,
    prompt_choice,
    fatal_error,
    INSTALL_DIR_LLAMA_CPP_HUANG,
    EXE_PATH_LLAMA_CPP_HUANG,
    INSTALL_DIR_LLAMA_CPP_BLACKWELL,
    EXE_PATH_LLAMA_CPP_BLACKWELL,
)

import tempfile

HUANG_REPO_URL = "https://github.com/RaymondHuang210129/llama.cpp-adaptive-kv-streaming.git"
HUANG_BRANCH = "feature/adaptive-kv-stream-v2"
FORK_INFO_HUANG_FILE = os.path.join(INSTALL_DIR_LLAMA_CPP_HUANG, ".fork_info")


def detect_cuda_architectures() -> str:
    """
    Detects target CUDA architectures, defaulting to 120a-real for Blackwell RTX 50-series if supported,
    or falling back to native / user-configured architecture.
    """
    configured = getattr(config, "cuda_architectures", None)
    if configured:
        return str(configured)

    try:
        res = subprocess.run(["nvcc", "--help"], capture_output=True, text=True, timeout=5)
        if "120a" in res.stdout or "120" in res.stdout:
            return "120a-real"
        res_v = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, timeout=5)
        if "V13." in res_v.stdout or "release 13." in res_v.stdout:
            return "120a-real"
    except Exception:
        pass
    return "native"


def resolve_huang_server_exe(allow_fallback: bool = True) -> Optional[str]:
    """
    Returns the resolved path to llama-server binary for Raymond Huang's Adaptive-KV engine.
    If not yet compiled and allow_fallback is True, falls back to installed Blackwell/Bruno binary if present.
    """
    server_bin = "llama-server.exe" if sys.platform == 'win32' else "llama-server"
    alt_bin = "llama-server-mtpctx.exe" if sys.platform == 'win32' else "llama-server-mtpctx"
    candidates = [
        os.path.join(INSTALL_DIR_LLAMA_CPP_HUANG, server_bin),
        os.path.join(INSTALL_DIR_LLAMA_CPP_HUANG, alt_bin),
    ]
    for cand in candidates:
        if os.path.exists(cand):
            return cand

    # Fallback to existing Blackwell/Bruno binary if Huang is not yet compiled and fallback requested
    if allow_fallback:
        fallback_candidates = [
            os.path.join(INSTALL_DIR_LLAMA_CPP_BLACKWELL, server_bin),
            os.path.join(INSTALL_DIR_LLAMA_CPP_BLACKWELL, alt_bin),
        ]
        for cand in fallback_candidates:
            if os.path.exists(cand):
                logger.debug(f"[huang_builder] Huang binary missing; using fallback binary at {cand}")
                return cand
    return None


def get_installed_huang_fork_info():
    """Reads installed fork and version metadata from .fork_info if present."""
    if not os.path.exists(FORK_INFO_HUANG_FILE):
        return None, None
    try:
        with open(FORK_INFO_HUANG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("fork"), data.get("version", data.get("branch"))
    except Exception:
        return None, None


def get_local_huang_version():
    """Checks if Huang llama-server binary exists in its dedicated install directory and extracts version build tag."""
    dedicated_exe = os.path.join(INSTALL_DIR_LLAMA_CPP_HUANG, "llama-server.exe")
    if not os.path.exists(dedicated_exe):
        return None

    try:
        result = subprocess.run(
            [dedicated_exe, "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace',
            check=True,
            timeout=15
        )
        output = result.stdout + result.stderr
        match = re.search(r'\b(b\d+)\b', output)
        if match:
            return match.group(1)
        match_alt = re.search(r'\b(\d+)\b', output)
        if match_alt:
            return f"b{match_alt.group(1)}"
        return "unknown_version"
    except Exception:
        return "corrupt_or_unresponsive"


def check_build_prerequisites() -> tuple[bool, str]:
    """Check that git, cmake, and nvcc (CUDA 13.x) are available on PATH."""
    missing = []
    if shutil.which("git") is None:
        missing.append("git")
    if shutil.which("cmake") is None:
        missing.append("cmake")
    if shutil.which("nvcc") is None:
        missing.append("nvcc (CUDA Toolkit)")

    if missing:
        msg = f"Missing required build tool(s): {', '.join(missing)}."
        return False, msg
    return True, "All build prerequisites detected."


def build_huang_llamacpp(build_root: str = None) -> bool:
    """Clones RaymondHuang210129/llama.cpp-adaptive-kv-streaming and compiles with sm_120a CUDA support."""
    ok, reason = check_build_prerequisites()
    if not ok:
        logger.error(f"Cannot build Huang Adaptive-KV llama.cpp: {reason}")
        return False

    if build_root is None:
        build_root = os.path.join(tempfile.gettempdir(), "build_huang_llamacpp")

    src_dir = os.path.join(build_root, "src")
    cmake_build_dir = os.path.join(build_root, "build")

    os.makedirs(build_root, exist_ok=True)

    logger.info(f"Preparing source in {src_dir}...")
    if not os.path.exists(src_dir):
        clone_cmd = [
            "git", "clone",
            "--depth", "1",
            "-b", HUANG_BRANCH,
            HUANG_REPO_URL,
            src_dir
        ]
        logger.info(f"Cloning {HUANG_BRANCH} from {HUANG_REPO_URL}...")
        res = subprocess.run(clone_cmd, text=True, encoding='utf-8', errors='replace')
        if res.returncode != 0:
            logger.error("Git clone failed.")
            return False
    else:
        logger.info(f"Existing source directory found at {src_dir}. Fetching latest...")
        subprocess.run(["git", "fetch", "--depth", "1", "origin", HUANG_BRANCH], cwd=src_dir)
        subprocess.run(["git", "reset", "--hard", f"origin/{HUANG_BRANCH}"], cwd=src_dir)

    is_windows = sys.platform == 'win32'
    arch = detect_cuda_architectures()
    logger.info(f"Configuring CMake for Raymond Huang Adaptive-KV llama.cpp with CUDA arch '{arch}'...")
    cmake_config_cmd = [
        "cmake",
        "-B", cmake_build_dir,
        "-S", src_dir,
    ]
    if is_windows:
        cmake_config_cmd.extend(["-G", "Visual Studio 17 2022", "-A", "x64"])
    elif shutil.which("ninja"):
        cmake_config_cmd.extend(["-G", "Ninja"])

    cmake_config_cmd.extend([
        "-DGGML_CUDA=ON",
        "-DGGML_CUDA_FA_ALL_QUANTS=ON",
        f"-DCMAKE_CUDA_ARCHITECTURES={arch}",
        "-DCMAKE_BUILD_TYPE=Release"
    ])
    res = subprocess.run(cmake_config_cmd, text=True, encoding='utf-8', errors='replace')
    if res.returncode != 0:
        logger.error("CMake configuration failed.")
        return False

    # Build llama-server target
    logger.info("Compiling llama-server target (this may take several minutes)...")
    cpu_count = os.cpu_count() or 8
    cmake_build_cmd = [
        "cmake",
        "--build", cmake_build_dir,
        "--config", "Release",
        "--target", "llama-server",
        "-j", str(max(4, min(cpu_count, 16)))
    ]
    res = subprocess.run(cmake_build_cmd, text=True, encoding='utf-8', errors='replace')
    if res.returncode != 0:
        logger.error("Build failed.")
        return False

    server_bin = "llama-server.exe" if is_windows else "llama-server"
    alt_bin = "llama-server-mtpctx.exe" if is_windows else "llama-server-mtpctx"

    # Locate generated binaries and copy to INSTALL_DIR_LLAMA_CPP_HUANG
    bin_candidates = [
        os.path.join(cmake_build_dir, "bin", "Release"),
        os.path.join(cmake_build_dir, "bin"),
        os.path.join(cmake_build_dir, "Release"),
        os.path.join(cmake_build_dir, "tools", "server", "Release"),
        os.path.join(cmake_build_dir, "tools", "server")
    ]
    found_bin_dir = None
    for cand in bin_candidates:
        if os.path.exists(os.path.join(cand, server_bin)) or os.path.exists(os.path.join(cand, alt_bin)):
            found_bin_dir = cand
            break

    if not found_bin_dir:
        logger.error("Could not locate compiled llama-server binary in build artifacts.")
        return False

    logger.info(f"Installing compiled binaries to {INSTALL_DIR_LLAMA_CPP_HUANG}...")
    os.makedirs(INSTALL_DIR_LLAMA_CPP_HUANG, exist_ok=True)

    # Clean old binaries
    locked_files = []
    for fname in os.listdir(INSTALL_DIR_LLAMA_CPP_HUANG):
        if fname.lower().endswith(('.exe', '.dll', '.pdb', '.so', '.dylib')) or fname == '.fork_info':
            fpath = os.path.join(INSTALL_DIR_LLAMA_CPP_HUANG, fname)
            try:
                os.remove(fpath)
            except PermissionError:
                locked_files.append(fname)
            except Exception:
                pass

    if locked_files:
        logger.error(f"Cannot install compiled binaries: Target files are locked: {', '.join(locked_files[:5])}. Please stop the running llama-server.")
        return False

    valid_exts = ('.exe', '.dll') if is_windows else ('', '.so', '.dylib')
    for item in os.listdir(found_bin_dir):
        if is_windows and item.lower().endswith(valid_exts):
            src = os.path.join(found_bin_dir, item)
            dst = os.path.join(INSTALL_DIR_LLAMA_CPP_HUANG, item)
            shutil.copy2(src, dst)
        elif not is_windows and (item in (server_bin, alt_bin) or item.endswith(('.so', '.dylib'))):
            src = os.path.join(found_bin_dir, item)
            dst = os.path.join(INSTALL_DIR_LLAMA_CPP_HUANG, item)
            shutil.copy2(src, dst)
            try:
                os.chmod(dst, 0o755)
            except Exception:
                pass

    for root, _, files in os.walk(cmake_build_dir):
        for item in files:
            if item.lower().endswith(('.dll', '.so')):
                src = os.path.join(root, item)
                dst = os.path.join(INSTALL_DIR_LLAMA_CPP_HUANG, item)
                if not os.path.exists(dst):
                    try:
                        shutil.copy2(src, dst)
                    except Exception:
                        pass

    # Save fork info
    try:
        with open(FORK_INFO_HUANG_FILE, "w", encoding="utf-8") as f:
            json.dump({
                "fork": "huang_adaptive_kv",
                "engine": "huang",
                "repo": HUANG_REPO_URL,
                "branch": HUANG_BRANCH,
                "target": arch,
                "binary": server_bin,
                "date": datetime.now().isoformat()
            }, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to write .fork_info: {e}")

    logger.info(f"[SUCCESS] Huang Adaptive-KV llama.cpp successfully built and installed at {INSTALL_DIR_LLAMA_CPP_HUANG}")
    return True


def check_huang_llamacpp() -> bool:
    """Verifies that the Huang Adaptive-KV specialized llama-server is present or builds it according to configuration."""
    if sys.platform != 'win32':
        logger.info("Huang Adaptive-KV llama.cpp is only configured for Windows x64. Skipping.")
        return False

    local_version = get_local_huang_version()
    installed_fork, stored_version = get_installed_huang_fork_info()

    if local_version is not None and local_version not in ["unknown_version", "corrupt_or_unresponsive"]:
        logger.info(f"Status: Found local Huang Adaptive-KV llama-server.exe ({stored_version or local_version}) [sm_120a optimized].")
        return True

    update_behavior = getattr(config, 'llamacpp_huang_update_behavior', 'never_update')
    if update_behavior == 'never_update' and local_version is not None:
        logger.info("  -> Skipping Huang llama.cpp update.")
        return True

    logger.info("Status: Huang Adaptive-KV llama-server.exe is NOT installed in llama-cpp-huang.")
    behavior = getattr(config, 'llamacpp_huang_install_behavior', 'existing')
    if behavior == 'never_install':
        logger.info("  -> Skipping Huang llama.cpp build as per configuration.")
        return False
    elif behavior == 'always_install':
        logger.info("Proceeding with Huang Adaptive-KV compilation (behavior='always_install')...")
        return build_huang_llamacpp()

    # Check build prerequisites portably
    from local_ai_launcher.backends.bruno_builder import check_build_prerequisites
    prereqs_ok, prereq_msg = check_build_prerequisites()
    if not prereqs_ok:
        logger.warning(f"  -> Build prerequisites missing ({prereq_msg}). Skipping Huang compilation.")
        logger.warning("  -> Safe dynamic VRAM limits will be applied to fallback llama-server.")
        return False

    if behavior == 'existing':
        import os
        is_ci = any(os.environ.get(v) for v in ("CI", "GITHUB_ACTIONS", "CONTINUOUS_INTEGRATION"))
        is_noninteractive = is_ci or not (sys.stdin and sys.stdin.isatty())
        if is_noninteractive:
            logger.info("  -> Non-interactive/CI environment detected and Huang binary missing. Skipping auto-compilation to prevent pipeline stalls.")
            logger.info("  -> Safe dynamic VRAM limits will be applied to fallback llama-server.")
            return False

        if not prompt_choice("Do you want to compile Huang Adaptive-KV llama.cpp (RaymondHuang210129)? (y/n): ", default='y'):
            logger.info("  -> Skipping Huang build (will use fallback llama-server with safe dynamic VRAM limit).")
            return False

    logger.info("Proceeding with Huang Adaptive-KV llama.cpp compilation...")
    return build_huang_llamacpp()
