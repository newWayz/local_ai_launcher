"""
ASCII vocabulary pruner and manager for Blackwell Huang Adaptive-KV runtime.

Manages pruning byte-level BPE GGUF models down to an ASCII / P1M keep-set
using https://github.com/bsaleh03/ASCII-Condensed-prune-tools.
Reduces model weight footprint from ~14.25 GB down to ~12.25 GB so that
Huang's bounded device phase arena (--kv-stream-arena-mib 2240) fits
within a 16 GB VRAM budget alongside full GPU layer offloading (ngl = -1).
"""
import os
import sys
import re
import subprocess
import logging
from pathlib import Path
from typing import Optional, Tuple, List

from local_ai_launcher.paths import AI_WORKSPACE_DIR, LOCAL_AI_LAUNCHER_DIR

logger = logging.getLogger("ascii_pruner")

PRUNE_REPO_URL = "https://github.com/bsaleh03/ASCII-Condensed-prune-tools"
PRUNE_TOOLS_DIR = AI_WORKSPACE_DIR / "tools" / "ASCII-Condensed-prune-tools"

# Strict allowed models for Huang Adaptive-KV on 16GB VRAM
# 1. ISTA-DASLab IQ3_S (unchanged)
# 2. Unsloth ASCII IQ4_XS (pruned)
# 3. Huihui ASCII IQ4_XS (pruned)
HUANG_ALLOWED_STEMS = [
    "qwen3.8-27b-gsq-rco-iq3_s",
    "qwen3.8-27b-ud-ascii-iq4_xs",
    "huihui-qwen3.8-27b-abliterated-ud-ascii-iq4_xs",
]

# Mapping of standard unpruned base models to their ASCII counterparts
HUANG_ASCII_MAPPINGS = {
    "qwen3.8-27b-ud-iq4_xs.gguf": "Qwen3.8-27B-UD-ASCII-IQ4_XS.gguf",
    "huihui-qwen3.8-27b-abliterated-ud-iq4_xs.gguf": "Huihui-Qwen3.8-27B-abliterated-UD-ASCII-IQ4_XS.gguf",
}

# Canonical Hugging Face base filenames for ASCII models
HUANG_BASE_MAPPINGS = {
    "qwen3.8-27b-ud-ascii-iq4_xs.gguf": "Qwen3.8-27B-UD-IQ4_XS.gguf",
    "huihui-qwen3.8-27b-abliterated-ud-ascii-iq4_xs.gguf": "Huihui-Qwen3.8-27B-abliterated-UD-IQ4_XS.gguf",
}


def ensure_prune_dependencies() -> None:
    """Verifies that numpy and gguf are installed in the python environment."""
    try:
        import numpy  # noqa: F401
        import gguf  # noqa: F401
    except ImportError:
        logger.info("[ascii_pruner] Required dependencies (numpy, gguf) not found. Installing via pip...")
        cmd = [sys.executable, "-m", "pip", "install", "numpy>=1.26.0", "gguf>=0.10.0"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"Failed to auto-install 'numpy' and 'gguf': {res.stderr}")
        logger.info("[ascii_pruner] Successfully installed numpy and gguf.")


def derive_base_model_filename(ascii_model_path_or_name: Path | str) -> str:
    """
    Derives the canonical base model filename on Hugging Face for a given ASCII model name/path.
    E.g. Qwen3.8-27B-UD-ASCII-IQ4_XS.gguf -> Qwen3.8-27B-UD-IQ4_XS.gguf
    """
    name = Path(ascii_model_path_or_name).name
    lower_name = name.lower()

    if lower_name in HUANG_BASE_MAPPINGS:
        return HUANG_BASE_MAPPINGS[lower_name]

    for base_k, ascii_v in HUANG_ASCII_MAPPINGS.items():
        if ascii_v.lower() == lower_name:
            if "unsloth" in lower_name or "qwen3.8" in lower_name:
                return "Qwen3.8-27B-UD-IQ4_XS.gguf"
            if "huihui" in lower_name:
                return "Huihui-Qwen3.8-27B-abliterated-UD-IQ4_XS.gguf"
            return base_k

    if "ud-ascii-iq4_xs" in lower_name:
        return name.replace("UD-ASCII-IQ4_XS", "UD-IQ4_XS").replace("ud-ascii-iq4_xs", "ud-iq4_xs")
    elif "ascii-iq4_xs" in lower_name:
        return name.replace("ASCII-IQ4_XS", "IQ4_XS").replace("ascii-iq4_xs", "iq4_xs")
    elif "-ascii.gguf" in lower_name:
        return name.replace("-ASCII.gguf", ".gguf").replace("-ascii.gguf", ".gguf")
    elif "-ascii" in lower_name:
        return name.replace("-ASCII", "").replace("-ascii", "")
    return name


def resolve_prune_vocab_script() -> Optional[Path]:
    """Finds prune_vocab.py across candidate locations."""
    candidates = [
        PRUNE_TOOLS_DIR / "prune_vocab.py",
        AI_WORKSPACE_DIR / "ASCII-Condensed-prune-tools" / "prune_vocab.py",
        LOCAL_AI_LAUNCHER_DIR / "tools" / "ASCII-Condensed-prune-tools" / "prune_vocab.py",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def ensure_prune_tools() -> Path:
    """
    Ensures ASCII-Condensed-prune-tools repository is cloned and returns the path to prune_vocab.py.
    """
    ensure_prune_dependencies()
    script_path = resolve_prune_vocab_script()
    if script_path and script_path.is_file():
        return script_path

    target_dir = PRUNE_TOOLS_DIR
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"[ascii_pruner] Cloning {PRUNE_REPO_URL} into {target_dir}...")

    cmd = ["git", "clone", PRUNE_REPO_URL, str(target_dir)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Failed to clone {PRUNE_REPO_URL}: {res.stderr}")

    script_path = target_dir / "prune_vocab.py"
    if not script_path.is_file():
        raise FileNotFoundError(f"prune_vocab.py not found at {script_path} after cloning")

    return script_path


def derive_ascii_model_path(base_model_path: Path | str) -> Path:
    """
    Derives the canonical ASCII output path for a given base GGUF model.
    E.g. Qwen3.8-27B-UD-IQ4_XS.gguf -> Qwen3.8-27B-UD-ASCII-IQ4_XS.gguf
    """
    base_p = Path(base_model_path)
    base_name = base_p.name
    lower_name = base_name.lower()

    for k, v in HUANG_ASCII_MAPPINGS.items():
        if lower_name == k:
            return base_p.parent / v

    if "ud-iq4_xs" in lower_name:
        new_name = base_name.replace("UD-IQ4_XS", "UD-ASCII-IQ4_XS").replace("ud-iq4_xs", "ud-ascii-iq4_xs")
        return base_p.parent / new_name
    elif "iq4_xs" in lower_name:
        new_name = base_name.replace("IQ4_XS", "ASCII-IQ4_XS").replace("iq4_xs", "ascii-iq4_xs")
        return base_p.parent / new_name
    elif lower_name.endswith(".gguf"):
        new_name = base_name[:-5] + "-ASCII.gguf"
        return base_p.parent / new_name
    return base_p.parent / (base_name + "-ASCII.gguf")


def is_ascii_model(model_path_or_name: Path | str) -> bool:
    """Checks if a model name or path represents an ASCII-pruned variant."""
    return "ascii" in str(model_path_or_name).lower()


def get_system_vram_mb() -> float:
    """Returns the total GPU VRAM in MiB, inspecting nvidia-smi/CUDA or hardware profile."""
    try:
        from local_ai_launcher.vram_coordinator import get_vram_info
        vinfo = get_vram_info()
        if vinfo.get("available") and vinfo.get("total_mb", 0) > 0:
            return float(vinfo["total_mb"])
    except Exception:
        pass

    try:
        from local_ai_launcher.config import get_active_hardware_profile
        profile = get_active_hardware_profile()
        if profile.get("vram_ceiling_mb", 0) > 0:
            return float(profile["vram_ceiling_mb"])
    except Exception:
        pass

    return 16384.0  # 16 GB default


def estimate_model_and_mmproj_size_mb(
    model_path_or_name: str,
    mmproj_path: Optional[str] = None,
    no_mmproj: bool = False,
) -> Tuple[float, float]:
    """
    Estimates or reads the disk size (in MiB) for the model weight file and mmproj file.
    Inspects disk files first, then presets, then falls back to parameter heuristics.
    """
    model_size_mb = 0.0
    mmproj_size_mb = 0.0

    p = Path(model_path_or_name)
    found_file = None
    if p.is_file():
        found_file = p
    elif not p.is_absolute():
        for cand in [
            AI_WORKSPACE_DIR / "models" / "llama-server" / p,
            AI_WORKSPACE_DIR / "models" / p,
            AI_WORKSPACE_DIR / "models" / "gguf" / p,
            AI_WORKSPACE_DIR / p,
        ]:
            if cand.is_file():
                found_file = cand
                break
        if not found_file:
            models_dir = AI_WORKSPACE_DIR / "models"
            if models_dir.is_dir():
                for cand in models_dir.glob(f"**/{p.name}"):
                    if cand.is_file():
                        found_file = cand
                        break

    # Preset lookup if not directly located
    if not found_file:
        try:
            from local_ai_launcher.presets import load_effective_models_parser
            cp = load_effective_models_parser()
            lookup_keys = [k.lower() for k in [model_path_or_name, p.name] if k]
            for s in cp.sections():
                aliases = [a.strip().lower() for a in cp.get(s, "alias", fallback="").split(",") if a.strip()]
                if s.lower() in lookup_keys or any(k in aliases for k in lookup_keys):
                    m_opt = cp.get(s, "model", fallback=None)
                    if m_opt:
                        for cand in [
                            Path(m_opt),
                            AI_WORKSPACE_DIR / m_opt,
                            AI_WORKSPACE_DIR / "models" / "llama-server" / m_opt,
                        ]:
                            if cand.is_file():
                                found_file = cand
                                break
                    if not mmproj_path and not no_mmproj:
                        mm_opt = cp.get(s, "mmproj", fallback=None)
                        if mm_opt:
                            mmproj_path = mm_opt
                    break
        except Exception:
            pass

    if found_file and found_file.is_file():
        model_size_mb = os.path.getsize(found_file) / (1024 * 1024)
    else:
        # Heuristic estimation based on model parameter count & quantization
        ident = str(model_path_or_name).lower()
        if "qwen3.8-27b-gsq-rco-iq3_s" in ident or "huang-gsq" in ident:
            model_size_mb = 11226.0
        elif "ud-ascii-iq4_xs" in ident or "ascii" in ident:
            model_size_mb = 12937.0
        elif "ud-iq4_xs" in ident:
            model_size_mb = 13592.0
        else:
            m = re.search(r"(\d+(?:\.\d+)?)\s*[bB]", ident)
            if m:
                param_b = float(m.group(1))
                if param_b >= 60:
                    model_size_mb = param_b * 550.0
                elif param_b >= 30:
                    model_size_mb = param_b * 570.0
                elif param_b >= 20:
                    model_size_mb = param_b * 500.0
                elif param_b >= 10:
                    model_size_mb = param_b * 600.0
                else:
                    model_size_mb = param_b * 625.0
                if "iq3" in ident or "q3" in ident:
                    model_size_mb *= 0.75
                elif "q8" in ident or "fp8" in ident:
                    model_size_mb *= 1.8
                elif "f16" in ident or "bf16" in ident:
                    model_size_mb *= 3.5

    # MMProj size
    if not no_mmproj and mmproj_path:
        mm_p = Path(mmproj_path)
        found_mm = None
        if mm_p.is_file():
            found_mm = mm_p
        elif not mm_p.is_absolute():
            for cand in [
                AI_WORKSPACE_DIR / "models" / "llama-server" / mm_p,
                AI_WORKSPACE_DIR / "models" / mm_p,
                AI_WORKSPACE_DIR / mm_p,
            ]:
                if cand.is_file():
                    found_mm = cand
                    break
        if found_mm and found_mm.is_file():
            mmproj_size_mb = os.path.getsize(found_mm) / (1024 * 1024)
        else:
            mm_ident = str(mmproj_path).lower()
            if "bf16" in mm_ident or "f16" in mm_ident:
                mmproj_size_mb = 1200.0
            else:
                mmproj_size_mb = 800.0

    return model_size_mb, mmproj_size_mb


def is_huang_vram_eligible(
    model_path_or_name: str,
    mmproj_path: Optional[str] = None,
    arena_mib: Optional[int] = None,
    no_mmproj: Optional[bool] = None,
    total_vram_mb: Optional[float] = None,
    ngl: Optional[int] = None,
    n_cpu_moe: Optional[int] = None,
) -> Tuple[bool, str]:
    """
    Evaluates dynamic VRAM eligibility for Raymond Huang's Adaptive-KV streaming engine.
    Zero static whitelist:
      Condition 1: Model weight (+ mmproj file size if present) exceeds 62% of total GPU VRAM.
      Condition 2: Settings are configured to fully fit within GPU VRAM
                   (model + mmproj + kv-stream-arena-mib + CUDA runtime headroom <= total VRAM,
                    with no CPU MoE offloading and ngl != 0).
    """
    total_vram = float(total_vram_mb) if total_vram_mb is not None else get_system_vram_mb()
    threshold_mb = 0.62 * total_vram

    # Extract preset defaults if model matches a preset
    if arena_mib is None or no_mmproj is None or ngl is None or n_cpu_moe is None:
        try:
            from local_ai_launcher.presets import load_effective_models_parser
            cp = load_effective_models_parser()
            p_name = Path(model_path_or_name).name
            lookup_keys = [k.lower() for k in [model_path_or_name, p_name] if k]
            for s in cp.sections():
                aliases = [a.strip().lower() for a in cp.get(s, "alias", fallback="").split(",") if a.strip()]
                if s.lower() in lookup_keys or any(k in aliases for k in lookup_keys):
                    if arena_mib is None and cp.has_option(s, "kv-stream-arena-mib"):
                        arena_mib = int(cp.get(s, "kv-stream-arena-mib"))
                    if no_mmproj is None and cp.has_option(s, "no-mmproj"):
                        no_mmproj = cp.get(s, "no-mmproj").lower() in ("true", "1", "yes", "on")
                    if ngl is None and cp.has_option(s, "ngl"):
                        ngl = int(cp.get(s, "ngl"))
                    if n_cpu_moe is None and cp.has_option(s, "n-cpu-moe"):
                        n_cpu_moe = int(cp.get(s, "n-cpu-moe"))
                    if not mmproj_path and cp.has_option(s, "mmproj"):
                        mmproj_path = cp.get(s, "mmproj")
                    break
        except Exception:
            pass

    if arena_mib is None:
        from local_ai_launcher.config import config
        arena_mib = getattr(config, "huang_arena_mib", 2240)
    if no_mmproj is None:
        from local_ai_launcher.config import config
        no_mmproj = getattr(config, "huang_no_mmproj", True)
    if ngl is None:
        ngl = -1
    if n_cpu_moe is None:
        n_cpu_moe = 0

    model_size_mb, mmproj_size_mb = estimate_model_and_mmproj_size_mb(
        model_path_or_name=model_path_or_name,
        mmproj_path=mmproj_path,
        no_mmproj=no_mmproj,
    )
    weight_mb = model_size_mb + mmproj_size_mb

    # Condition 1: Exceeds 62% of total GPU VRAM
    if weight_mb <= threshold_mb:
        return False, (
            f"Model footprint ({weight_mb:.1f} MiB) does not exceed 62% of total GPU VRAM "
            f"({total_vram:.1f} MiB, threshold: {threshold_mb:.1f} MiB); Adaptive-KV streaming not required"
        )

    # Condition 2: Settings configured to fully fit within GPU VRAM
    if ngl == 0:
        return False, "Model settings set ngl=0 (CPU-only execution); cannot run on GPU VRAM"

    if n_cpu_moe > 0:
        return False, f"Model settings require CPU MoE offloading (n-cpu-moe={n_cpu_moe}); cannot fully fit within GPU VRAM"

    # Effective footprint accounting for auto-ASCII pruning if Qwen 3.8 27B IQ4_XS base
    effective_weight_mb = weight_mb
    ident = str(model_path_or_name).lower()
    if ("qwen3.8-27b" in ident or "qwen-38_27b" in ident) and "iq4_xs" in ident and "ascii" not in ident and "gsq" not in ident:
        # Base IQ4_XS auto-diverts to ASCII (~12.94 GB) to fit within 16GB VRAM
        effective_weight_mb = 12937.0 + mmproj_size_mb

    headroom_mb = 384.0  # CUDA runtime context, activations, scratch space
    total_required_mb = effective_weight_mb + arena_mib + headroom_mb

    if total_required_mb > total_vram:
        return False, (
            f"Model settings do not fully fit within GPU VRAM: requires {total_required_mb:.1f} MiB "
            f"({effective_weight_mb:.1f} MiB weights + {arena_mib} MiB arena + {headroom_mb:.0f} MiB headroom) "
            f"which exceeds total GPU VRAM ({total_vram:.1f} MiB)"
        )

    return True, (
        f"Model footprint ({weight_mb:.1f} MiB) exceeds 62% of total VRAM ({threshold_mb:.1f} MiB) "
        f"and settings fully fit within GPU VRAM ({total_required_mb:.1f} MiB <= {total_vram:.1f} MiB with {arena_mib} MiB arena)"
    )


def is_huang_model_allowed(
    model_identifier: str,
    mmproj_path: Optional[str] = None,
    arena_mib: Optional[int] = None,
) -> bool:
    """
    Checks if a model qualifies for the Raymond Huang Adaptive-KV runtime under the dynamic VRAM rule:
    1. Model weight (+ mmproj file if present) exceeds 62% of total GPU VRAM.
    2. Settings allow it to fully fit within GPU VRAM alongside the Adaptive-KV streaming arena.
    Zero static whitelist: any model satisfying the dynamic VRAM criteria qualifies.
    """
    eligible, _ = is_huang_vram_eligible(
        model_path_or_name=model_identifier,
        mmproj_path=mmproj_path,
        arena_mib=arena_mib,
    )
    return eligible


def ensure_ascii_variant(
    base_model_path: Path | str,
    target_ascii_path: Optional[Path | str] = None,
    policy: str = "P1M",
) -> Path:
    """
    Ensures that the ASCII variant for the given base GGUF model exists.
    If it already exists and is non-empty (>1GB), returns the target path immediately.
    If not, clones ASCII-Condensed-prune-tools (if needed) and prunes the model.
    """
    base_p = Path(base_model_path)
    if not base_p.is_file():
        raise FileNotFoundError(f"Base model not found for ASCII pruning: {base_p}")

    target_p = Path(target_ascii_path) if target_ascii_path else derive_ascii_model_path(base_p)

    # If target already exists and is sufficiently large (> 10 GB), reuse it
    if target_p.is_file() and target_p.stat().st_size > 10 * 1024 * 1024 * 1024:
        logger.info(f"[ascii_pruner] Reusing existing ASCII variant: {target_p} ({target_p.stat().st_size / 1e9:.2f} GB)")
        return target_p

    prune_script = ensure_prune_tools()

    logger.info(
        f"[ascii_pruner] Pruning base model '{base_p.name}' to ASCII variant '{target_p.name}' "
        f"using {policy} policy (~12.25 GB target)..."
    )

    staging_p = target_p.with_suffix(".prune_staging.tmp")
    if staging_p.is_file():
        try:
            staging_p.unlink()
        except Exception:
            pass

    cmd = [
        sys.executable,
        str(prune_script),
        str(base_p),
        str(staging_p),
        "--policy",
        policy,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # Clean up partial staging output if any
        if staging_p.is_file():
            try:
                staging_p.unlink()
            except Exception:
                pass
        raise RuntimeError(
            f"ASCII pruning failed with return code {result.returncode}:\n"
            f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )

    if not staging_p.is_file() or staging_p.stat().st_size < 10 * 1024 * 1024 * 1024:
        if staging_p.is_file():
            try:
                staging_p.unlink()
            except Exception:
                pass
        raise RuntimeError(
            f"ASCII pruning finished but output file is missing or too small: {staging_p}"
        )

    # Atomic rename to final target path
    os.replace(staging_p, target_p)

    logger.info(
        f"[ascii_pruner] Successfully produced ASCII variant: {target_p} "
        f"({target_p.stat().st_size / 1e9:.2f} GB)"
    )
    return target_p


def resolve_huang_model_selection(
    model_name: str,
    model_path: str,
) -> Tuple[str, str]:
    """
    Validates and resolves the model requested for the Huang Adaptive-KV runtime.
    Enforces the user rule:
      1. C:\\aiWorkspace\\models\\llama-server\\ISTA-DASLab\\Qwen3.8-27B-GSQ-RCO\\Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf unchanged.
      2. Unsloth Qwen3.8-27B-UD-IQ4_XS.gguf -> auto-pruned ASCII variant only.
      3. Huihui-ai Huihui-Qwen3.8-27B-abliterated-UD-IQ4_XS.gguf -> auto-pruned ASCII variant only.
    Raises ValueError if an unpermitted model is selected.
    """
    model_key = f"{model_name} {model_path}".lower().replace("\\", "/")

    # Case 1: ISTA-DASLab IQ3_S (unchanged)
    if "qwen3.8-27b-gsq-rco-iq3_s" in model_key:
        return model_name, model_path

    # Case 2: Already points to an ASCII variant
    if "ascii" in model_key:
        # Ensure it exists or auto-prune
        p = Path(model_path)
        if not p.is_file():
            base_fname = derive_base_model_filename(p.name)
            base_candidate = p.parent / base_fname
            if base_candidate.is_file():
                ensure_ascii_variant(base_candidate, p)
            else:
                try:
                    from local_ai_launcher.backends.model_installer import ensure_single_model_installed
                    ensure_single_model_installed(model_name)
                except Exception as e:
                    logger.debug(f"[ascii_pruner] Auto-install attempt for '{model_name}': {e}")
        return model_name, model_path

    # Case 3: Unsloth base IQ4_XS -> Auto-divert to ASCII variant
    if "unsloth" in model_key and "iq4_xs" in model_key:
        p = Path(model_path)
        if not p.is_absolute():
            p = AI_WORKSPACE_DIR / "models" / "llama-server" / model_path
        if not p.is_file():
            p = AI_WORKSPACE_DIR / "models" / "llama-server" / "unsloth" / "Qwen3.8-27B-GGUF" / "Qwen3.8-27B-UD-IQ4_XS.gguf"

        target_name = "qwen-38_27b-huang-ascii-nothink" if "nothink" in model_key else "qwen-38_27b-huang-ascii-think"
        ascii_path = derive_ascii_model_path(p)
        if not ascii_path.is_file():
            if not p.is_file():
                try:
                    from local_ai_launcher.backends.model_installer import ensure_single_model_installed
                    ensure_single_model_installed(target_name)
                except Exception as e:
                    logger.debug(f"[ascii_pruner] Auto-install attempt for '{target_name}': {e}")
            if p.is_file():
                ascii_path = ensure_ascii_variant(p, ascii_path)

        logger.info(f"[ascii_pruner] Diverted Huang engine from base IQ4_XS to ASCII variant: {ascii_path.name}")
        return target_name, str(ascii_path)

    # Case 4: Huihui base IQ4_XS -> Auto-divert to ASCII variant
    if "huihui" in model_key and "iq4_xs" in model_key:
        p = Path(model_path)
        if not p.is_absolute():
            p = AI_WORKSPACE_DIR / "models" / "llama-server" / model_path
        if not p.is_file():
            p = AI_WORKSPACE_DIR / "models" / "llama-server" / "huihui-ai" / "Huihui-Qwen3.8-27B-abliterated-GGUF" / "Huihui-Qwen3.8-27B-abliterated-UD-IQ4_XS.gguf"

        target_name = "qwen-38_27b-huang-abli-ascii-nothink" if "nothink" in model_key else "qwen-38_27b-huang-abli-ascii-think"
        ascii_path = derive_ascii_model_path(p)
        if not ascii_path.is_file():
            if not p.is_file():
                try:
                    from local_ai_launcher.backends.model_installer import ensure_single_model_installed
                    ensure_single_model_installed(target_name)
                except Exception as e:
                    logger.debug(f"[ascii_pruner] Auto-install attempt for '{target_name}': {e}")
            if p.is_file():
                ascii_path = ensure_ascii_variant(p, ascii_path)

        logger.info(f"[ascii_pruner] Diverted Huang engine from base abliterated IQ4_XS to ASCII variant: {ascii_path.name}")
        return target_name, str(ascii_path)

    # Case 5: Known Blackwell preset alias redirection to ASCII presets
    known_bw = {
        "qwen-38_27b-blackwell-256k-think",
        "qwen-38_27b-blackwell-256k-nothink",
        "qwen-38_27b-blackwell-256k-abli-think",
        "qwen-38_27b-blackwell-256k-abli-nothink",
    }
    if model_name.lower() in known_bw:
        is_abli = "abli" in model_name.lower()
        is_nothink = "nothink" in model_name.lower()
        if is_abli:
            p = AI_WORKSPACE_DIR / "models" / "llama-server" / "huihui-ai" / "Huihui-Qwen3.8-27B-abliterated-GGUF" / "Huihui-Qwen3.8-27B-abliterated-UD-IQ4_XS.gguf"
            target_name = "qwen-38_27b-huang-abli-ascii-nothink" if is_nothink else "qwen-38_27b-huang-abli-ascii-think"
        else:
            p = AI_WORKSPACE_DIR / "models" / "llama-server" / "unsloth" / "Qwen3.8-27B-GGUF" / "Qwen3.8-27B-UD-IQ4_XS.gguf"
            target_name = "qwen-38_27b-huang-ascii-nothink" if is_nothink else "qwen-38_27b-huang-ascii-think"

        ascii_path = derive_ascii_model_path(p)
        if not ascii_path.is_file():
            if not p.is_file():
                try:
                    from local_ai_launcher.backends.model_installer import ensure_single_model_installed
                    ensure_single_model_installed(target_name)
                except Exception as e:
                    logger.debug(f"[ascii_pruner] Auto-install attempt for '{target_name}': {e}")
            if p.is_file():
                ascii_path = ensure_ascii_variant(p, ascii_path)

        logger.info(f"[ascii_pruner] Diverted Huang engine for '{model_name}' to ASCII preset '{target_name}': {ascii_path.name}")
        return target_name, str(ascii_path)

    # Dynamic VRAM eligibility evaluation (Zero static whitelist)
    eligible, reason = is_huang_vram_eligible(model_path_or_name=model_path or model_name)
    if eligible:
        logger.info(f"[ascii_pruner] Qualified model '{model_name or model_path}' for Huang Adaptive-KV: {reason}")
        return model_name, model_path

    raise ValueError(
        f"Model '{model_name or model_path}' is not eligible for the Raymond Huang Adaptive-KV runtime: {reason}. "
        f"Huang Adaptive-KV requires that the model (+ mmproj) weight exceeds 62% of total GPU VRAM "
        f"and settings are configured to fully fit within GPU VRAM with the streaming arena."
    )
