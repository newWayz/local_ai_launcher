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
from typing import Optional, Dict, Tuple, List

from local_ai_launcher.config import (
    config,
    logger,
    prompt_choice,
    fatal_error,
    INSTALL_DIR_LLAMA_CPP_BLACKWELL,
    EXE_PATH_LLAMA_CPP_BLACKWELL,
    INSTALL_DIR_LLAMA_CPP_BRUNO,
    EXE_PATH_LLAMA_CPP_BRUNO,
)

BRUNO_REPO_URL = "https://github.com/BrunoPPassini/llama.cpp.git"
BRUNO_BRANCH = "qwen38-blackwell-256k"
BLACKWELL_REPO_URL = BRUNO_REPO_URL
BLACKWELL_BRANCH = BRUNO_BRANCH

FORK_INFO_BRUNO_FILE = os.path.join(INSTALL_DIR_LLAMA_CPP_BRUNO, ".fork_info")
FORK_INFO_BLACKWELL_FILE = os.path.join(INSTALL_DIR_LLAMA_CPP_BLACKWELL, ".fork_info")

# Canonical 36 environment variables required by the BrunoPPassini Blackwell runtime
# to activate the 64K hot VRAM / 192K cold host DDR5 ring buffer, sparse CUDA VMM,
# and triple-buffered DMA/compute pipeline as defined in start-server.ps1.
BRUNO_RUNTIME_ENV: Dict[str, str] = {
    "LLAMA_MTP_SHARED_COMPUTE": "1",
    "LLAMA_MTP_MMA_FIXED_SHAPE": "1",
    "LLAMA_ARG_JINJA": "1",
    "LLAMA_ARG_CHAT_TEMPLATE_KWARGS": '{"enable_thinking":true,"preserve_thinking":true}',
    "LLAMA_QWEN35_BLACKWELL_ENGINE": "1",
    "LLAMA_KV_SPARSE_VMM": "1",
    "LLAMA_KV_SPARSE_INITIAL_TOKENS": "8192",
    "LLAMA_KV_SPARSE_CHUNK_TOKENS": "4096",
    "LLAMA_KV_SPARSE_PREFETCH_TOKENS": "4096",
    "LLAMA_KV_SPARSE_HOST_FALLBACK": "1",
    "LLAMA_KV_SPARSE_DEVICE_RESERVE_MIB": "512",
    "LLAMA_KV_SPARSE_TRACE": "1",
    "LLAMA_KV_SPARSE_DEVICE_PREFIX_TOKENS": "16384",
    "LLAMA_KV_RESERVE_TOKENS": "16384",
    "LLAMA_KV_HOST_STAGE_COLD_FA": "1",
    "LLAMA_KV_HOST_RING_FA": "1",
    "LLAMA_KV_HOST_RING_FORCE_VEC": "0",
    "LLAMA_KV_HOST_RING_TILE_TOKENS": "8192",
    "LLAMA_KV_HOST_RING_MMA": "1",
    "LLAMA_KV_HOST_RING_MMA_COLD_ONLY": "1",
    "LLAMA_KV_HOST_RING_MMA_TILE_TOKENS": "8192",
    "LLAMA_KV_HOST_RING_MMA_PARTITIONS": "8",
    "LLAMA_KV_HOST_RING_MMA_DIRECT_HOT": "1",
    "LLAMA_KV_HOST_RING_MMA_PIPELINE": "1",
    "LLAMA_KV_HOST_RING_MMA_PREFILL_PIPELINE": "1",
    "LLAMA_KV_HOST_RING_MMA_TRIPLE_PIPELINE": "1",
    "LLAMA_KV_HOST_RING_MMA_FUSED_Q4_DEQUANT": "1",
    "LLAMA_KV_HOST_RING_MMA_Q4_NARROW_BLOCK": "1",
    "LLAMA_KV_HOST_RING_MMA_PREFILL_FSM": "0",
    "LLAMA_KV_HOST_RING_MMA_INLINE_Q4": "0",
    "LLAMA_RS_TRANSACTION_LOG": "1",
    "LLAMA_RS_PHASE_ARENA": "1",
    "LLAMA_COMPUTE_SPARSE_VMM": "1",
    "LLAMA_COMPUTE_SPARSE_TRIM": "1",
    "LLAMA_COMPUTE_SPARSE_TRIM_WINDOW": "8",
    "LLAMA_COMPUTE_SPARSE_TRIM_HEADROOM_MIB": "32",
    "LLAMA_COMPUTE_SPARSE_TRIM_FLOOR_MIB": "160",
    "LLAMA_COMPUTE_SPARSE_TRIM_THRESHOLD_MIB": "64",
    "LLAMA_COMPUTE_SPARSE_REQUEST_RESET": "0",
    "LLAMA_QWEN35_LAYER_MAJOR_WINDOW": "0",
    "LLAMA_KV_HOST_RING_MMA_LAYER_CACHE_MIB": "0",
    "LLAMA_DISABLE_FUSED_GDN_CH": "1",
}
BLACKWELL_RUNTIME_ENV = BRUNO_RUNTIME_ENV

# Strict allowed models for Bruno Passini hybrid-KV on 16GB VRAM
# 1. Unsloth Qwen3.8-27B-UD-IQ4_XS.gguf (unpruned base)
# 2. Huihui-ai Huihui-Qwen3.8-27B-abliterated-UD-IQ4_XS.gguf (unpruned base)
# Note: ASCII variants and ISTA-DASLab IQ3_S models are explicitly disallowed on Bruno.
BRUNO_ALLOWED_STEMS: List[str] = [
    "qwen3.8-27b-ud-iq4_xs",
    "huihui-qwen3.8-27b-abliterated-ud-iq4_xs",
]

BRUNO_ALLOWED_FILES = {
    "qwen3.8-27b-ud-iq4_xs.gguf",
    "huihui-qwen3.8-27b-abliterated-ud-iq4_xs.gguf",
}

BRUNO_ALLOWED_PRESETS = {
    "qwen-38_27b-blackwell-256k-think",
    "qwen-38_27b-blackwell-256k-nothink",
    "qwen-38_27b-blackwell-256k-abli-think",
    "qwen-38_27b-blackwell-256k-abli-nothink",
}


def is_bruno_model_allowed(model_name: str, model_path: str = "") -> bool:
    """
    Validates if a model is permitted under the Bruno Passini engine.
    Bruno is strictly tuned to:
      1. Qwen3.8-27B-UD-IQ4_XS.gguf
      2. Huihui-Qwen3.8-27B-abliterated-UD-IQ4_XS.gguf
    ASCII models and ISTA-DASLab IQ3_S models are explicitly disallowed.
    """
    combined = f"{model_name} {model_path}".lower().replace("\\", "/")

    # ASCII variants are explicitly disallowed for Bruno
    if "ascii" in combined:
        return False

    # ISTA-DASLab / GSQ / IQ3_S are explicitly disallowed for Bruno
    if "gsq" in combined or "iq3_s" in combined or "iq3s" in combined:
        return False

    # Huang-specific presets are explicitly disallowed for Bruno
    if "huang" in combined:
        return False

    name_clean = model_name.strip().lower()
    if name_clean in BRUNO_ALLOWED_PRESETS:
        return True

    path_name = Path(model_path).name.lower() if model_path else ""
    if path_name in BRUNO_ALLOWED_FILES:
        return True

    name_as_file = Path(model_name).name.lower()
    if name_as_file in BRUNO_ALLOWED_FILES:
        return True

    path_stem = Path(model_path).stem.lower() if model_path else ""
    name_stem = Path(model_name).stem.lower()
    if path_stem in BRUNO_ALLOWED_STEMS or name_stem in BRUNO_ALLOWED_STEMS:
        return True

    return False


def resolve_bruno_kv_cache_types(
    k_type: Optional[str],
    v_type: Optional[str],
) -> Tuple[str, str]:
    """
    Enforces Bruno Passini Blackwell KV cache stability invariants:
    - Must ALWAYS run in symmetric q8_0/q8_0 or q4_0/q4_0.
    - If mixed (e.g. q8 / q4), choose the lower (q4_0 / q4_0).
    - Any other format (f16, bf16, q5_0, etc.) defaults to q4_0 / q4_0 to avoid crashes.
    """
    k = (k_type or "").strip().lower()
    v = (v_type or "").strip().lower()

    def _is_q8(val: str) -> bool:
        return val in ("q8_0", "q8", "q8_1", "q8_k")

    def _is_q4(val: str) -> bool:
        return val in ("q4_0", "q4", "q4_1", "q4_k", "q4_k_m", "q4_k_s", "iq4_xs", "iq4_nl")

    # If both are q8, run symmetric q8_0 / q8_0
    if _is_q8(k) and _is_q8(v):
        return "q8_0", "q8_0"

    # In all other cases (both q4, mixed q8/q4, or unsupported formats), clamp to safe symmetric q4_0
    return "q4_0", "q4_0"


def get_bruno_runtime_env() -> Dict[str, str]:
    """Returns a copy of the current environment with Bruno Blackwell runtime variables merged in."""
    env = os.environ.copy()
    env.update(BRUNO_RUNTIME_ENV)
    return env


get_blackwell_runtime_env = get_bruno_runtime_env


def resolve_bruno_server_exe() -> Optional[str]:
    """Returns the resolved path to either llama-server.exe or llama-server-mtpctx.exe for Bruno's engine."""
    candidates = [
        os.path.join(INSTALL_DIR_LLAMA_CPP_BRUNO, "llama-server.exe"),
        os.path.join(INSTALL_DIR_LLAMA_CPP_BRUNO, "llama-server-mtpctx.exe"),
        os.path.join(INSTALL_DIR_LLAMA_CPP_BLACKWELL, "llama-server.exe"),
        os.path.join(INSTALL_DIR_LLAMA_CPP_BLACKWELL, "llama-server-mtpctx.exe"),
    ]
    for cand in candidates:
        if os.path.exists(cand):
            return cand
    return None


resolve_blackwell_server_exe = resolve_bruno_server_exe


def get_installed_bruno_fork_info():
    """Reads installed fork and version metadata from .fork_info if present."""
    for info_path in [FORK_INFO_BRUNO_FILE, FORK_INFO_BLACKWELL_FILE]:
        if os.path.exists(info_path):
            try:
                with open(info_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data.get("fork"), data.get("version", data.get("branch"))
            except Exception:
                pass
    return None, None


get_installed_blackwell_fork_info = get_installed_bruno_fork_info


def get_local_bruno_version():
    """Checks if Bruno Blackwell llama-server binary exists and extracts its version build tag."""
    exe_path = resolve_bruno_server_exe()
    if not exe_path:
        return None

    try:
        result = subprocess.run(
            [exe_path, "--version"],
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


get_local_blackwell_version = get_local_bruno_version


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


def patch_bruno_source(src_dir: str) -> bool:
    """
    Patches BrunoPPassini/llama.cpp source code to allow RTX 5080 Laptop GPUs
    (Compute Capability 12.0 sm_120) to pass the Blackwell runtime architecture gate.
    """
    model_cpp = os.path.join(src_dir, "src", "llama-model.cpp")
    if not os.path.exists(model_cpp):
        logger.warning(f"Could not find {model_cpp} to apply Blackwell architecture patch.")
        return False

    try:
        with open(model_cpp, "r", encoding="utf-8") as f:
            content = f.read()

        target = 'dev_desc.find("RTX 5070 Ti") != std::string::npos;'
        replacement = '(dev_desc.find("RTX 5070 Ti") != std::string::npos || dev_desc.find("RTX 5080") != std::string::npos || dev_desc.find("RTX 50") != std::string::npos);'

        if target in content:
            content = content.replace(target, replacement)
            with open(model_cpp, "w", encoding="utf-8") as f:
                f.write(content)
            logger.info("  -> Successfully patched src/llama-model.cpp to support RTX 5080 Laptop GPU architecture gate.")
            return True
        elif replacement in content:
            logger.info("  -> src/llama-model.cpp already contains RTX 5080 architecture patch.")
            return True
        else:
            logger.warning("  -> Target check 'RTX 5070 Ti' not found in src/llama-model.cpp; skipping patch.")
            return False
    except Exception as e:
        logger.error(f"Error while patching src/llama-model.cpp: {e}")
        return False


patch_blackwell_source = patch_bruno_source


def build_bruno_llamacpp(build_root: str = None, install_dir: str = None) -> bool:
    """Clones BrunoPPassini/llama.cpp (branch qwen38-blackwell-256k) and compiles with sm_120a CUDA support."""
    ok, reason = check_build_prerequisites()
    if not ok:
        logger.error(f"Cannot build Bruno Blackwell llama.cpp: {reason}")
        return False

    if build_root is None:
        import tempfile
        build_root = os.path.join(tempfile.gettempdir(), "build_bruno_llamacpp")

    if install_dir is None:
        install_dir = INSTALL_DIR_LLAMA_CPP_BRUNO

    src_dir = os.path.join(build_root, "src")
    cmake_build_dir = os.path.join(build_root, "build")

    os.makedirs(build_root, exist_ok=True)

    logger.info(f"Preparing source in {src_dir}...")
    if not os.path.exists(src_dir):
        clone_cmd = [
            "git", "clone",
            "--depth", "1",
            "-b", BRUNO_BRANCH,
            BRUNO_REPO_URL,
            src_dir
        ]
        logger.info(f"Cloning {BRUNO_BRANCH} from {BRUNO_REPO_URL}...")
        res = subprocess.run(clone_cmd, text=True, encoding='utf-8', errors='replace')
        if res.returncode != 0:
            logger.error("Git clone failed.")
            return False
    else:
        logger.info(f"Existing source directory found at {src_dir}. Fetching latest...")
        subprocess.run(["git", "fetch", "--depth", "1", "origin", BRUNO_BRANCH], cwd=src_dir)
        subprocess.run(["git", "reset", "--hard", f"origin/{BRUNO_BRANCH}"], cwd=src_dir)

    # Apply architecture gate patch to support RTX 5080 Laptop GPU
    patch_bruno_source(src_dir)

    # Configure with CMake
    logger.info("Configuring CMake with CUDA Blackwell sm_120a architecture...")
    is_windows = sys.platform == 'win32'
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
        "-DGGML_CUDA_GRAPHS=ON",
        "-DGGML_CUDA_FORCE_CUBLAS=OFF",
        "-DGGML_CUDA_FORCE_MMQ=OFF",
        "-DCMAKE_CUDA_ARCHITECTURES=120a-real",
        "-DCMAKE_BUILD_TYPE=Release"
    ])
    res = subprocess.run(cmake_config_cmd, text=True, encoding='utf-8', errors='replace')
    if res.returncode != 0:
        logger.error("CMake configuration failed.")
        return False

    # Build llama-server target
    logger.info("Compiling llama-server target with MSVC (this may take several minutes)...")
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

    bin_candidates = [
        os.path.join(cmake_build_dir, "bin", "Release"),
        os.path.join(cmake_build_dir, "bin"),
        os.path.join(cmake_build_dir, "Release"),
        os.path.join(cmake_build_dir, "tools", "server", "Release"),
        os.path.join(cmake_build_dir, "tools", "server")
    ]
    found_bin_dir = None
    for cand in bin_candidates:
        if os.path.exists(os.path.join(cand, "llama-server-mtpctx.exe")) or os.path.exists(os.path.join(cand, "llama-server.exe")):
            found_bin_dir = cand
            break

    if not found_bin_dir:
        logger.error("Could not locate compiled llama-server binary in build artifacts.")
        return False

    logger.info(f"Installing compiled binaries to {install_dir}...")
    os.makedirs(install_dir, exist_ok=True)

    locked_files = []
    for fname in os.listdir(install_dir):
        if fname.lower().endswith(('.exe', '.dll', '.pdb')) or fname == '.fork_info':
            fpath = os.path.join(install_dir, fname)
            try:
                os.remove(fpath)
            except PermissionError:
                locked_files.append(fname)
            except Exception:
                pass

    if locked_files:
        logger.error(f"Cannot install compiled binaries: Target files are locked by a running process: {', '.join(locked_files[:5])}. Please stop the running llama-server and re-run.")
        return False

    for item in os.listdir(found_bin_dir):
        if item.lower().endswith(('.exe', '.dll')):
            src = os.path.join(found_bin_dir, item)
            dst = os.path.join(install_dir, item)
            shutil.copy2(src, dst)

    for root, _, files in os.walk(cmake_build_dir):
        for item in files:
            if item.lower().endswith('.dll'):
                src = os.path.join(root, item)
                dst = os.path.join(install_dir, item)
                if not os.path.exists(dst):
                    try:
                        shutil.copy2(src, dst)
                    except Exception:
                        pass

    mtp_exe = os.path.join(install_dir, "llama-server-mtpctx.exe")
    std_exe = os.path.join(install_dir, "llama-server.exe")
    if os.path.exists(mtp_exe) and not os.path.exists(std_exe):
        shutil.copy2(mtp_exe, std_exe)
    elif os.path.exists(std_exe) and not os.path.exists(mtp_exe):
        shutil.copy2(std_exe, mtp_exe)

    fork_file = os.path.join(install_dir, ".fork_info")
    try:
        with open(fork_file, "w", encoding="utf-8") as f:
            json.dump({
                "fork": "blackwell_qwen38",
                "engine": "bruno",
                "repo": BRUNO_REPO_URL,
                "branch": BRUNO_BRANCH,
                "target": "120a-real",
                "binary": "llama-server-mtpctx.exe",
                "date": datetime.now().isoformat()
            }, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to write .fork_info: {e}")

    logger.info(f"[SUCCESS] Bruno Blackwell llama.cpp successfully built and installed at {install_dir}")
    return True


build_blackwell_llamacpp = build_bruno_llamacpp


def check_bruno_llamacpp() -> bool:
    """Verifies that the Bruno Blackwell specialized llama-server is present or builds it according to configuration."""
    if sys.platform != 'win32':
        logger.info("Bruno Blackwell llama.cpp is only configured for Windows x64. Skipping.")
        return False

    local_version = get_local_bruno_version()
    installed_fork, stored_version = get_installed_bruno_fork_info()

    if local_version is not None and local_version not in ["unknown_version", "corrupt_or_unresponsive"]:
        logger.info(f"Status: Found local Bruno Blackwell llama-server.exe ({stored_version or local_version}) [sm_120a RTX 5080 Laptop optimized] - updates permanently disabled.")
        return True

    update_behavior = getattr(config, 'llamacpp_blackwell_update_behavior', 'never_update')
    if update_behavior == 'never_update' and local_version is not None:
        logger.info("  -> Skipping Bruno Blackwell llama.cpp update (locked to custom sm_120a build).")
        return True

    logger.info("Status: Bruno Blackwell llama-server.exe is NOT installed.")
    behavior = getattr(config, 'llamacpp_blackwell_install_behavior', 'existing')
    if behavior == 'never_install':
        logger.info("  -> Skipping Bruno Blackwell llama.cpp build as per configuration.")
        return False
    elif behavior == 'existing':
        if not prompt_choice("Do you want to compile Bruno Blackwell llama.cpp (qwen38-blackwell-256k)? (y/n): ", default='n'):
            logger.info("  -> Skipping Bruno build.")
            return False

    logger.info("Proceeding with Bruno Blackwell llama.cpp compilation...")
    return build_bruno_llamacpp()


check_blackwell_llamacpp = check_bruno_llamacpp
