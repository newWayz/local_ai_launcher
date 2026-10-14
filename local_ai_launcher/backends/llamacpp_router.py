"""
Unified llama-server Variant Router & Priority Decision Engine.
Coordinates between:
  1. Bruno Passini Fork (16GB Blackwell, Qwen3.8-27B-UD-IQ4_XS base/abliterated, 256K context, 36 VMM env vars)
  2. Raymond Huang Fork (16GB NVIDIA, 63%~96.8% VRAM, MTP only, zero CPU offload, circular arena)
  3. Official Upstream llama.cpp (universal fallback, clamped context, non-duplicated --fit off)

Preserves non-llama engines (TabbyAPI, FreeToken, Strata).
"""
import os
import sys
import re
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

from local_ai_launcher.config import (
    AI_WORKSPACE_DIR,
    EXE_PATH_LLAMA_CPP,
    EXE_PATH_LLAMA_CPP_HUANG,
    EXE_PATH_LLAMA_CPP_BLACKWELL,
    config,
    logger,
    get_active_hardware_profile,
    load_effective_models_parser,
    resolve_effective_models_preset,
)
from local_ai_launcher.backends.bruno_builder import is_bruno_model_allowed

router_logger = logging.getLogger("llamacpp_router")

HUANG_VRAM_MIN_RATIO = 0.63
HUANG_VRAM_MAX_RATIO = 0.968
VRAM_HEADROOM_MB = 500.0


def find_model_file(model_path_or_name: Path | str) -> Optional[Path]:
    """Resolves a model path or identifier to an existing on-disk Path."""
    if not model_path_or_name:
        return None
    p = Path(model_path_or_name)
    if p.is_file():
        return p
    if not p.is_absolute():
        candidates = [
            AI_WORKSPACE_DIR / "models" / "llama-server" / p,
            AI_WORKSPACE_DIR / "models" / p,
            AI_WORKSPACE_DIR / "models" / "gguf" / p,
            AI_WORKSPACE_DIR / p,
        ]
        for cand in candidates:
            if cand.is_file():
                return cand
        # Glob search by filename under models directory
        models_dir = AI_WORKSPACE_DIR / "models"
        if models_dir.is_dir():
            for cand in models_dir.glob(f"**/{p.name}"):
                if cand.is_file():
                    return cand
    return None


def resolve_preset_data(model_name: str, model_path: str = "") -> Tuple[Optional[str], Dict[str, Any]]:
    """Looks up preset section name and options from effective models.ini."""
    try:
        cp = load_effective_models_parser()
        p_name = Path(model_path).name if model_path else ""
        lookup_keys = [k.lower() for k in [model_name, model_path, p_name] if k]
        for s in cp.sections():
            if s.lower() in lookup_keys:
                return s, dict(cp.items(s))
            aliases = [a.strip().lower() for a in cp.get(s, "alias", fallback="").split(",") if a.strip()]
            if any(k in aliases for k in lookup_keys):
                return s, dict(cp.items(s))
    except Exception as e:
        router_logger.debug(f"Preset lookup notice: {e}")
    return None, {}


def get_system_vram_mb() -> float:
    """Returns total GPU VRAM in MiB from VRAM coordinator or hardware profile."""
    try:
        from local_ai_launcher.vram_coordinator import get_vram_info
        vinfo = get_vram_info()
        if vinfo.get("available") and vinfo.get("total_mb", 0) > 0:
            return float(vinfo["total_mb"])
    except Exception:
        pass
    try:
        profile = get_active_hardware_profile()
        if profile.get("vram_ceiling_mb", 0) > 0:
            return float(profile["vram_ceiling_mb"])
    except Exception:
        pass
    return 16384.0


def get_model_size_mb(model_path_or_name: Path | str, preset_opts: Optional[Dict[str, Any]] = None) -> float:
    """Measures model file size in MiB directly from disk or preset reference."""
    file_path = find_model_file(model_path_or_name)
    if not file_path and preset_opts and "model" in preset_opts:
        file_path = find_model_file(preset_opts["model"])
    if file_path and file_path.is_file():
        try:
            return os.path.getsize(str(file_path)) / (1024 * 1024)
        except Exception:
            pass
    # Fallback to estimate_model_and_mmproj_size_mb when file is not on disk (CI/testing)
    try:
        from local_ai_launcher.backends.ascii_pruner import estimate_model_and_mmproj_size_mb
        msize, _ = estimate_model_and_mmproj_size_mb(model_path_or_name=model_path_or_name)
        if msize > 0:
            return msize
        if preset_opts and "model" in preset_opts:
            msize, _ = estimate_model_and_mmproj_size_mb(model_path_or_name=preset_opts["model"])
            if msize > 0:
                return msize
    except Exception:
        pass
    return 0.0


def get_model_size_and_vram_threshold(model_path_or_name: Path | str) -> Tuple[float, float]:
    """
    Measures model file size in MiB and calculates the VRAM threshold (VRAM - 500 MB).
    Returns (model_size_mb, threshold_mb).
    """
    total_vram = get_system_vram_mb()
    threshold_mb = total_vram - VRAM_HEADROOM_MB
    model_size_mb = get_model_size_mb(model_path_or_name)
    return model_size_mb, threshold_mb


def check_model_exceeds_vram_threshold(model_path_or_name: Path | str) -> Tuple[bool, float, float]:
    """
    Evaluates whether model file size exceeds (VRAM size - 500 MB).
    Returns (exceeds, model_size_mb, threshold_mb).
    """
    model_size_mb, threshold_mb = get_model_size_and_vram_threshold(model_path_or_name)
    exceeds = (model_size_mb > 0) and (model_size_mb > threshold_mb)
    return exceeds, model_size_mb, threshold_mb


def check_model_supports_mtp(model_path_or_name: Path | str, preset_opts: Optional[Dict[str, Any]] = None) -> bool:
    """
    Checks if a model has verified MTP support:
      1. -mtp in the filename
      2. Companion -mtp.gguf exists on disk
      3. GGUF header contains 'nextn' or 'mtp' tensor metadata
    Does NOT use raw 'ascii' heuristics to avoid false positives.
    """
    if not model_path_or_name and not (preset_opts and preset_opts.get("model")):
        return False

    candidates = []
    if model_path_or_name:
        candidates.append(str(model_path_or_name))
    if preset_opts and preset_opts.get("model"):
        candidates.append(str(preset_opts["model"]))

    for cand in candidates:
        lower_str = cand.lower()
        if "-mtp" in lower_str or "nextn" in lower_str:
            return True

        file_p = find_model_file(cand)
        if file_p and file_p.is_file():
            # Check companion -mtp.gguf
            companion = file_p.with_name(file_p.stem + "-mtp" + file_p.suffix)
            if companion.is_file():
                return True

            # Inspect GGUF metadata
            try:
                import gguf
                reader = gguf.GGUFReader(str(file_p))
                for k in reader.fields:
                    k_lower = k.lower()
                    if "nextn" in k_lower or "mtp" in k_lower:
                        val = reader.fields[k].parts[-1]
                        if hasattr(val, "__iter__") and len(val) > 0 and val[0] > 0:
                            return True
                        elif isinstance(val, (int, float)) and val > 0:
                            return True
            except Exception:
                pass

    return False


def has_cpu_offload_settings(opts: Dict[str, Any], req_obj: Any) -> Tuple[bool, str]:
    """
    Determines if the model configuration offloads layers to CPU RAM.
    Returns (has_offload, reason).
    """
    # 1. Check n-cpu-moe
    n_cpu_moe = 0
    if "n-cpu-moe" in opts:
        try:
            n_cpu_moe = int(opts["n-cpu-moe"])
        except Exception:
            pass
    if req_obj and getattr(req_obj, "EngineArgs", None):
        try:
            ea = json.loads(req_obj.EngineArgs)
            if isinstance(ea, dict) and "n-cpu-moe" in ea:
                n_cpu_moe = int(ea["n-cpu-moe"])
        except Exception:
            pass
    if n_cpu_moe > 0:
        return True, f"CPU MoE offloading enabled (n-cpu-moe={n_cpu_moe})"

    # 2. Check ngl / n-gpu-layers partial offload
    ngl = None
    if "ngl" in opts:
        try:
            ngl = int(opts["ngl"])
        except Exception:
            pass
    elif "n-gpu-layers" in opts:
        try:
            ngl = int(opts["n-gpu-layers"])
        except Exception:
            pass
    elif req_obj and getattr(req_obj, "NGPULayers", 0) != 0:
        ngl = req_obj.NGPULayers

    if ngl is not None and 0 <= ngl < 99:
        return True, f"Partial GPU offload configured (ngl={ngl})"

    return False, "Full GPU execution"


def should_force_fit_off(
    model_path_or_name: Path | str,
    opts_dict: Optional[Dict[str, Any]] = None,
    req_obj: Any = None,
) -> Tuple[bool, float, float]:
    """
    Evaluates whether --fit off should be explicitly forced:
    True if the model has no CPU-offload settings and model file size > (VRAM size - 500 MB).
    Returns (should_force, model_size_mb, vram_threshold_mb).
    """
    opts = opts_dict or {}
    has_cpu_offload, _ = has_cpu_offload_settings(opts, req_obj)
    model_size_mb, threshold_mb = get_model_size_and_vram_threshold(model_path_or_name)
    should_force = (not has_cpu_offload) and (model_size_mb > threshold_mb)
    return should_force, model_size_mb, threshold_mb


def enforce_cmd_arg(cmd_list: List[str], flag: str, value: str) -> None:
    """Enforce a flag and its value in cmd_list without duplicating the flag."""
    i = 0
    while i < len(cmd_list):
        if cmd_list[i] == flag:
            del cmd_list[i]
            if i < len(cmd_list) and not cmd_list[i].startswith("--"):
                del cmd_list[i]
        else:
            i += 1
    cmd_list.extend([flag, value])


def enforce_fallback_fit_off_if_needed(
    cmd: List[str],
    model_path_or_name: Path | str,
    opts_dict: Optional[Dict[str, Any]] = None,
    req_obj: Any = None,
) -> bool:
    """
    Enforces '--fit off' non-duplicated exclusively for models that undergo
    Automatic Fallback to Official Upstream llama-server when the model has no
    CPU-offload settings and exceeds the VRAM threshold (VRAM - 500MB).
    """
    force_fit_off, m_size, v_thresh = should_force_fit_off(model_path_or_name, opts_dict, req_obj)
    if force_fit_off:
        enforce_cmd_arg(cmd, "--fit", "off")
        fit_msg = (
            f"[llama-server] Model size ({m_size:.1f} MB) exceeds VRAM ceiling threshold "
            f"({v_thresh:.1f} MB) without CPU-offload settings. Explicitly setting non-duplicated --fit off."
        )
        router_logger.info(fit_msg)
        print(fit_msg, flush=True)
        return True
    return False


def resolve_llamacpp_variant(
    model_name: str,
    model_path: str = "",
    preset_opts: Optional[Dict[str, Any]] = None,
    req_obj: Any = None,
) -> Tuple[str, str, Dict[str, Any]]:
    """
    Unified 3-Tier Priority Decision Engine:
      Tier 1: Bruno Passini Fork (16GB Blackwell, Qwen3.8-27B-UD-IQ4_XS base/abliterated)
      Tier 2: Raymond Huang Fork (63%~96.8% VRAM, MTP verified, zero CPU offload)
      Tier 3: Official Upstream llama.cpp (Universal fallback)

    Returns (variant, reason, metadata) where variant is 'bruno', 'huang', or 'official'.
    """
    # Resolve preset data if not supplied
    matched_sec, discovered_opts = resolve_preset_data(model_name, model_path)
    opts = dict(discovered_opts)
    if preset_opts:
        opts.update(preset_opts)

    total_vram = get_system_vram_mb()
    preset_model_path = opts.get("model") if opts else None
    if model_path and find_model_file(model_path):
        model_target = model_path
    elif preset_model_path:
        model_target = preset_model_path
    else:
        model_target = model_path or model_name

    model_size = get_model_size_mb(model_target, opts)
    vram_ratio = (model_size / total_vram) if total_vram > 0 else 0.0
    vram_threshold = total_vram - VRAM_HEADROOM_MB

    profile = get_active_hardware_profile()
    profile_name = profile.get("name", "blackwell_16gb")
    is_blackwell_hardware = profile_name == "blackwell_16gb"

    has_cpu_offload, cpu_reason = has_cpu_offload_settings(opts, req_obj)
    has_mtp = check_model_supports_mtp(model_target, opts)

    # Check explicit spec-type
    spec_type = opts.get("spec-type") or getattr(config, "huang_spec_type", "draft-mtp")
    if req_obj and getattr(req_obj, "EngineArgs", None):
        try:
            ea = json.loads(req_obj.EngineArgs)
            if isinstance(ea, dict) and "spec-type" in ea:
                spec_type = ea["spec-type"]
        except Exception:
            pass
    spec_disabled = str(spec_type).lower() in ("none", "off", "0", "false")

    metadata: Dict[str, Any] = {
        "model_size_mb": model_size,
        "total_vram_mb": total_vram,
        "vram_ratio": vram_ratio,
        "has_cpu_offload": has_cpu_offload,
        "cpu_reason": cpu_reason,
        "has_mtp": has_mtp,
        "spec_disabled": spec_disabled,
        "is_fit_off_needed": (not has_cpu_offload) and (model_size > vram_threshold),
        "vram_threshold_mb": vram_threshold,
        "matched_section": matched_sec,
        "effective_opts": opts,
    }

    # Check explicit hints
    raw_ident = f"{model_name} {model_path} {opts.get('alias', '')} {opts.get('engine', '')}".lower()

    # -------------------------------------------------------------------------
    # Tier 0: Check Explicit Official Upstream Selection
    # -------------------------------------------------------------------------
    official_requested = (
        "official" in raw_ident or
        "upstream" in raw_ident or
        opts.get("engine", "").lower() in ("official", "upstream", "llama-cpp", "llama-server") or
        getattr(config, "blackwell_engine", "") in ("official", "upstream")
    )
    if official_requested:
        return (
            "official",
            f"Official upstream llama.cpp explicitly requested for '{model_name or model_path}'",
            metadata,
        )

    # -------------------------------------------------------------------------
    # Tier 1: Evaluate Bruno Passini Fork
    # -------------------------------------------------------------------------
    bruno_allowed = is_bruno_model_allowed(model_name, model_path) or (
        bool(opts.get("model")) and is_bruno_model_allowed(model_name, opts["model"])
    )
    huang_requested = (
        "huang" in raw_ident or
        opts.get("engine", "").lower() == "huang"
    )
    bruno_requested = (
        "bruno" in raw_ident or
        "blackwell-256k" in raw_ident or
        opts.get("engine", "").lower() == "bruno" or
        getattr(config, "blackwell_engine", "huang") == "bruno"
    )

    if is_blackwell_hardware and bruno_allowed and not huang_requested:
        return (
            "bruno",
            f"Model '{model_name or model_path}' qualifies for Bruno Passini 256K Hybrid-KV on Blackwell 16GB",
            metadata,
        )

    # -------------------------------------------------------------------------
    # Tier 2: Evaluate Raymond Huang Fork
    # -------------------------------------------------------------------------
    # Requirements:
    #   1. MTP enabled and verified present
    #   2. Zero CPU offload in settings
    #   3. Model size within 63% ~ 96.8% of total GPU VRAM
    if not spec_disabled and has_mtp:
        if not has_cpu_offload:
            if (HUANG_VRAM_MIN_RATIO <= vram_ratio <= HUANG_VRAM_MAX_RATIO) or (model_size == 0 and huang_requested):
                return (
                    "huang",
                    f"Model footprint ({model_size:.1f} MB, {vram_ratio*100:.1f}% VRAM) within 63%~96.8% sweet spot with MTP and full GPU execution",
                    metadata,
                )
            elif vram_ratio < HUANG_VRAM_MIN_RATIO and model_size > 0:
                reason = f"Model size ({model_size:.1f} MB, {vram_ratio*100:.1f}% VRAM) below 63% threshold; standard KV cache fits in VRAM without circular streaming arena"
            elif vram_ratio > HUANG_VRAM_MAX_RATIO or model_size > vram_threshold:
                reason = f"Model size ({model_size:.1f} MB, {vram_ratio*100:.1f}% VRAM) exceeds VRAM ceiling threshold ({vram_threshold:.1f} MB); Raymond Huang streaming arena cannot fit in remaining VRAM"
            else:
                reason = "Model footprint outside sweet spot"
        else:
            reason = f"Model has CPU offload settings ({cpu_reason}); Raymond Huang streaming arena requires text-only GPU VRAM execution"
    else:
        if spec_disabled:
            reason = "Speculative execution explicitly disabled (spec-type=none)"
        else:
            reason = "Model lacks MTP tensors or companion MTP file; Raymond Huang engine provides no benefit without MTP"

    # -------------------------------------------------------------------------
    # Tier 3: Official Upstream llama.cpp Fallback
    # -------------------------------------------------------------------------
    return (
        "official",
        f"Falling back to Official Upstream llama-server: {reason}",
        metadata,
    )
