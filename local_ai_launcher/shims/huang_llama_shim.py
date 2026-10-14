import sys
import os
import time
import json
import socket
import logging
import subprocess
import atexit
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional, Dict, Tuple, Any, List

import configparser
import httpx
import grpc

from local_ai_launcher.config import (
    AI_WORKSPACE_DIR,
    EXE_PATH_LLAMA_CPP_HUANG,
    EXE_PATH_LLAMA_CPP_BLACKWELL,
    EXE_PATH_LLAMA_CPP,
    MODELS_PRESET_FILE,
    config,
    resolve_effective_models_preset,
    load_effective_models_parser,
    get_active_hardware_profile,
)
from local_ai_launcher.localai.proto import backend_pb2, backend_pb2_grpc
from local_ai_launcher.shims.base_shim import BaseBackendServicer, serve_shim, build_common_llama_payload
from local_ai_launcher.backends.huang_builder import resolve_huang_server_exe
from local_ai_launcher.backends.bruno_builder import is_bruno_model_allowed
from local_ai_launcher.backends.llamacpp_router import (
    find_model_file,
    resolve_preset_data,
    get_system_vram_mb,
    get_model_size_mb,
    get_model_size_and_vram_threshold as _router_get_vram_thresh,
    check_model_exceeds_vram_threshold as _router_check_exceeds,
    check_model_supports_mtp as _router_check_mtp,
    has_cpu_offload_settings,
    should_force_fit_off as _router_should_force_fit_off,
    enforce_cmd_arg as _router_enforce_cmd_arg,
    enforce_fallback_fit_off_if_needed as _router_enforce_fallback,
    resolve_llamacpp_variant,
    HUANG_VRAM_MIN_RATIO,
    HUANG_VRAM_MAX_RATIO,
    VRAM_HEADROOM_MB,
)

logger = logging.getLogger("huang_llama_shim")


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def check_model_supports_mtp(model_path: Path | str) -> bool:
    """
    Checks whether a given GGUF model contains an embedded MTP (Multi-Token Prediction / nextn) layer.
    First checks filename patterns; if ambiguous, checks GGUF metadata fields if the file exists.
    """
    return _router_check_mtp(model_path)


def get_model_size_and_vram_threshold(model_path_or_name: Path | str) -> Tuple[float, float]:
    """
    Measures model file size in MiB and calculates the VRAM threshold (VRAM - 500 MB).
    Returns (model_size_mb, threshold_mb).
    """
    return _router_get_vram_thresh(model_path_or_name)


def check_model_exceeds_vram_threshold(model_path_or_name: Path | str) -> Tuple[bool, float, float]:
    """
    Evaluates whether model file size exceeds (VRAM size - 500 MB).
    Returns (exceeds, model_size_mb, threshold_mb).
    """
    return _router_check_exceeds(model_path_or_name)


def should_force_fit_off(model_path_or_name: Path | str, opts_dict: Optional[Dict[str, Any]] = None, req_obj: Any = None) -> Tuple[bool, float, float]:
    """
    Evaluates whether --fit off should be explicitly forced:
    True if the model has no CPU-offload settings and model file size > (VRAM size - 500 MB).
    Returns (should_force, model_size_mb, vram_threshold_mb).
    """
    return _router_should_force_fit_off(model_path_or_name, opts_dict, req_obj)


def enforce_cmd_arg(cmd_list: list, flag: str, value: str) -> None:
    """Enforce a flag and its value in cmd_list without duplicating the flag."""
    _router_enforce_cmd_arg(cmd_list, flag, value)


def enforce_fallback_fit_off_if_needed(cmd: list, model_path_or_name: Path | str, opts_dict: Optional[Dict[str, Any]] = None, req_obj: Any = None) -> bool:
    """
    Enforces '--fit off' non-duplicated exclusively for models that undergo
    Automatic Fallback to Official Upstream llama-server when the model has no
    CPU-offload settings and exceeds the VRAM threshold (VRAM - 500MB).
    """
    force_fit_off, m_size, v_thresh = should_force_fit_off(model_path_or_name, opts_dict, req_obj)
    if force_fit_off:
        enforce_cmd_arg(cmd, "--fit", "off")
        fit_msg = (
            f"[huang-llama] Model size ({m_size:.1f} MB) exceeds VRAM ceiling threshold "
            f"({v_thresh:.1f} MB) without CPU-offload settings. Explicitly setting non-duplicated --fit off."
        )
        logger.info(fit_msg)
        print(fit_msg, flush=True)
        return True
    return False


class HuangLlamaServicer(BaseBackendServicer):
    """
    gRPC shim wrapping Raymond Huang's Adaptive-KV Streaming V2 llama-server.
    Features:
      - Bounded device phase arena (--kv-stream-arena-mib)
      - Attached Layer-17 Multi-Token Prediction (--spec-type draft-mtp --spec-draft-n-max 3)
      - High fidelity Q8_0 Key / Q4_0 Value caching
      - Zero WDDM cuCtxSynchronize barrier issues (no dynamic sparse VMM page-table mutations)
    Fulfills LocalAI's backend.proto contract.
    """
    def __init__(self):
        super().__init__("huang-llama")
        global _active_servicer
        _active_servicer = self
        self.process: Optional[subprocess.Popen] = None
        self.http_port: Optional[int] = None
        self.http_base: Optional[str] = None
        self.active_model_path: Optional[str] = None
        self.current_server_exe: Optional[str] = None
        self.current_is_blackwell: Optional[bool] = None
        self._log_file = None
        self._delegate = None

    def _wait_for_health(self, timeout: int = 60) -> bool:
        if not self.http_base:
            return False
        try:
            timeout = max(1, int(timeout))
        except (ValueError, TypeError):
            timeout = 60
        health_url = f"{self.http_base}/health"
        start = time.time()
        while time.time() - start < timeout:
            if self.process and self.process.poll() is not None:
                logger.error(f"[huang-llama] Subprocess exited early with code {self.process.returncode}. Check logs/llama_server_engine.log")
                return False
            try:
                req = urllib.request.Request(health_url, headers={"User-Agent": "ShimHealthCheck"})
                with urllib.request.urlopen(req, timeout=1.5) as resp:
                    if resp.status in (200, 204, 302):
                        return True
            except Exception:
                pass
            time.sleep(0.5)
        return False

    def LoadModel(self, request, context):
        """Launches Raymond Huang's Adaptive-KV llama-server process with the requested model."""
        self.Free(None, None)  # Evict any existing process first

        model_name = request.Model or ""
        model_path = request.ModelFile or request.Model or ""
        if not model_name and not model_path:
            return backend_pb2.Result(success=False, message="No model specified in ModelOptions")

        self.http_port = find_free_port()
        self.http_base = f"http://127.0.0.1:{self.http_port}"

        # Check if the model is defined in effective models preset
        matched_section = None
        matched_opts = {}
        effective_preset_file, is_overridden = resolve_effective_models_preset()
        try:
            cp = load_effective_models_parser()
            lookup_keys = [k.lower() for k in [model_name, model_path, Path(model_path).name] if k]
            for s in cp.sections():
                if s.lower() in lookup_keys:
                    matched_section = s
                    matched_opts = dict(cp.items(s))
                    break
                aliases = [a.strip().lower() for a in cp.get(s, "alias", fallback="").split(",") if a.strip()]
                if any(k in aliases for k in lookup_keys):
                    matched_section = s
                    matched_opts = dict(cp.items(s))
                    break
        except Exception as e:
            logger.warning(f"[huang-llama] Failed reading effective models preset: {e}")

        # Check active hardware profile
        active_profile = get_active_hardware_profile()
        profile_name = active_profile.get("name", "blackwell_16gb")

        # Check if targeting Blackwell / Adaptive-KV
        matched_aliases = [a.strip().lower() for a in matched_opts.get("alias", "").split(",") if a.strip()]
        from local_ai_launcher.backends.ascii_pruner import is_huang_vram_eligible
        vram_eligible, _ = is_huang_vram_eligible(
            model_path_or_name=model_path or model_name,
            mmproj_path=request.MMProj or matched_opts.get("mmproj"),
            no_mmproj=str(matched_opts.get("no-mmproj", "")).lower() in ("true", "1", "yes", "on"),
        )
        is_blackwell_target = (
            profile_name == "blackwell_16gb" and (
                "blackwell" in model_name.lower() or
                "blackwell" in model_path.lower() or
                "huang" in model_name.lower() or
                "huang" in model_path.lower() or
                "qwen3.8-27b" in model_name.lower() or
                "qwen3.8-27b" in model_path.lower() or
                "qwen-38_27b" in model_name.lower() or
                "qwen-38_27b" in model_path.lower() or
                (matched_section and ("blackwell" in matched_section.lower() or "huang" in matched_section.lower() or "qwen-38_27b" in matched_section.lower())) or
                any("blackwell" in a or "huang" in a or "qwen-38_27b" in a for a in matched_aliases) or
                (matched_opts.get("engine", "").lower() in ("blackwell", "huang")) or
                (getattr(config, "blackwell_engine", "huang") == "huang" and vram_eligible)
            )
        )

        is_genuine_huang = False
        is_bruno_fallback = False

        if is_blackwell_target:
            huang_exe = resolve_huang_server_exe(allow_fallback=False)
            if huang_exe and os.path.exists(huang_exe):
                server_exe = huang_exe
                is_genuine_huang = True
                engine_label = "Raymond Huang Adaptive-KV llama-server"
            elif os.path.exists(EXE_PATH_LLAMA_CPP_HUANG):
                server_exe = EXE_PATH_LLAMA_CPP_HUANG
                is_genuine_huang = True
                engine_label = "Raymond Huang Adaptive-KV llama-server"
            elif os.path.exists(EXE_PATH_LLAMA_CPP_BLACKWELL) and is_bruno_model_allowed(model_name or model_path):
                server_exe = EXE_PATH_LLAMA_CPP_BLACKWELL
                is_bruno_fallback = True
                engine_label = "Bruno Passini Blackwell fallback llama-server"
                logger.warning(
                    f"[huang-llama] Notice: Raymond Huang adaptive KV binary not found at {EXE_PATH_LLAMA_CPP_HUANG}. "
                    f"Operating in Bruno Passini fallback mode with sparse hybrid-KV environment."
                )
            elif os.path.exists(EXE_PATH_LLAMA_CPP):
                server_exe = EXE_PATH_LLAMA_CPP
                engine_label = "official llama-server"
                logger.warning(
                    f"[huang-llama] Notice: Huang binary not found at {EXE_PATH_LLAMA_CPP_HUANG}. Falling back to official llama-server at {EXE_PATH_LLAMA_CPP}."
                )
            else:
                from local_ai_launcher.backends.llamacpp import check_llamacpp
                logger.info(f"[huang-llama] Notice: llama-server.exe missing. Attempting automatic installation from GitHub releases...")
                if check_llamacpp() and os.path.exists(EXE_PATH_LLAMA_CPP):
                    server_exe = EXE_PATH_LLAMA_CPP
                    engine_label = "official llama-server"
                else:
                    server_exe = EXE_PATH_LLAMA_CPP_HUANG
                    is_genuine_huang = True
                    engine_label = "Raymond Huang Adaptive-KV llama-server"
        else:
            server_exe = EXE_PATH_LLAMA_CPP if os.path.exists(EXE_PATH_LLAMA_CPP) else (resolve_huang_server_exe() or EXE_PATH_LLAMA_CPP_HUANG)
            fork_label = getattr(config, 'llamacpp_fork', 'official')
            engine_label = f"{fork_label} llama-server"

        if is_bruno_fallback:
            from local_ai_launcher.backends.bruno_builder import get_bruno_runtime_env
            env = get_bruno_runtime_env()
        else:
            env = os.environ.copy()

        if is_blackwell_target and is_genuine_huang:
            from local_ai_launcher.backends.ascii_pruner import resolve_huang_model_selection
            try:
                resolved_name, resolved_path = resolve_huang_model_selection(model_name, model_path)
                model_name, model_path = resolved_name, resolved_path
            except ValueError as ve:
                logger.info(
                    f"[huang-llama] Model '{model_name or model_path}' does not qualify for Huang Adaptive-KV ({ve}). "
                    f"Falling back to official llama-server at {EXE_PATH_LLAMA_CPP}."
                )
                is_blackwell_target = False
                is_genuine_huang = False
                server_exe = EXE_PATH_LLAMA_CPP if os.path.exists(EXE_PATH_LLAMA_CPP) else (resolve_huang_server_exe() or EXE_PATH_LLAMA_CPP_HUANG)
                engine_label = "official llama-server"
                env = os.environ.copy()

            # Re-evaluate matched preset section with the resolved model name/path
            if cp:
                lookup_keys = [k.lower() for k in [model_name, model_path, Path(model_path).name] if k]
                matched_section = None
                matched_opts = {}
                for s in cp.sections():
                    if s.lower() in lookup_keys:
                        matched_section = s
                        matched_opts = dict(cp.items(s))
                        break
                    aliases = [a.strip().lower() for a in cp.get(s, "alias", fallback="").split(",") if a.strip()]
                    if any(k in aliases for k in lookup_keys):
                        matched_section = s
                        matched_opts = dict(cp.items(s))
                        break

        hf_home = str(AI_WORKSPACE_DIR / "models" / "huggingface")
        env["HOME"] = str(AI_WORKSPACE_DIR)
        env["HF_HOME"] = hf_home
        env["HF_HUB_CACHE"] = hf_home

        arena_mib = getattr(config, "huang_arena_mib", 2240)
        auxiliary_layers = getattr(config, "huang_auxiliary_layers", 1)
        draft_max = getattr(config, "huang_draft_max", 3)
        draft_p_min = getattr(config, "huang_draft_p_min", 0.20)
        draft_p_split = getattr(config, "huang_draft_p_split", 0.10)
        spec_type = getattr(config, "huang_spec_type", "draft-mtp")
        batch_size = getattr(config, "huang_batch_size", 512)
        ubatch_size = getattr(config, "huang_ubatch_size", 256)
        parallel = getattr(config, "huang_parallel", 1)
        no_mmproj = getattr(config, "huang_no_mmproj", True)

        if matched_opts:
            if "kv-stream-arena-mib" in matched_opts:
                try:
                    arena_mib = int(matched_opts["kv-stream-arena-mib"])
                except Exception:
                    pass
            if "kv-stream-auxiliary-layers" in matched_opts:
                try:
                    auxiliary_layers = int(matched_opts["kv-stream-auxiliary-layers"])
                except Exception:
                    pass
            if "spec-draft-n-max" in matched_opts:
                try:
                    draft_max = int(matched_opts["spec-draft-n-max"])
                except Exception:
                    pass
            if "spec-draft-p-min" in matched_opts:
                try:
                    draft_p_min = float(matched_opts["spec-draft-p-min"])
                except Exception:
                    pass
            if "spec-draft-p-split" in matched_opts:
                try:
                    draft_p_split = float(matched_opts["spec-draft-p-split"])
                except Exception:
                    pass
            if "spec-type" in matched_opts:
                spec_type = matched_opts["spec-type"]
            if "batch-size" in matched_opts:
                try:
                    batch_size = int(matched_opts["batch-size"])
                except Exception:
                    pass
            if "ubatch-size" in matched_opts:
                try:
                    ubatch_size = int(matched_opts["ubatch-size"])
                except Exception:
                    pass
            if "parallel" in matched_opts:
                try:
                    parallel = int(matched_opts["parallel"])
                except Exception:
                    pass
            if "no-mmproj" in matched_opts:
                no_mmproj = str(matched_opts["no-mmproj"]).lower() in ("true", "1", "yes", "on")

        self.current_server_exe = str(server_exe)
        self.current_is_blackwell = is_blackwell_target

        if not os.path.exists(server_exe):
            return backend_pb2.Result(success=False, message=f"llama-server binary not found at {server_exe}")

        if matched_section:
            overlay_msg = " (with local overlay)" if is_overridden else ""
            logger.info(f"[huang-llama] Matched preset section '{matched_section}' in {effective_preset_file.name}{overlay_msg}")
            cmd = [
                str(server_exe),
                "--port", str(self.http_port),
                "--models-preset", str(effective_preset_file),
                "--models-max", "1",
                "--host", "127.0.0.1",
                "--jinja",
            ]
            if is_blackwell_target and is_genuine_huang:
                # Resolve preset model to verify MTP support
                preset_model_val = matched_opts.get("model", "")
                resolved_preset_model = Path(preset_model_val)
                if not resolved_preset_model.is_absolute():
                    resolved_preset_model = AI_WORKSPACE_DIR / preset_model_val

                mtp_requested = (
                    str(spec_type).lower() not in ("none", "off", "0", "false")
                    and arena_mib > 0
                )
                has_mtp = False
                if mtp_requested:
                    has_mtp = check_model_supports_mtp(resolved_preset_model)
                    if not has_mtp:
                        companion = resolved_preset_model.with_name(
                            resolved_preset_model.stem + "-mtp" + resolved_preset_model.suffix
                        )
                        if os.path.exists(str(companion)) and check_model_supports_mtp(companion):
                            resolved_preset_model = companion
                            has_mtp = True

                exceeds_vram, m_size, v_thresh = check_model_exceeds_vram_threshold(resolved_preset_model)
                total_vram = get_system_vram_mb()
                vram_ratio = (m_size / total_vram) if total_vram > 0 else 0.0
                below_min_vram = (vram_ratio < HUANG_VRAM_MIN_RATIO) and (m_size > 0)
                has_cpu_offload, cpu_reason = has_cpu_offload_settings(matched_opts, request)

                if mtp_requested and has_mtp and not exceeds_vram and not below_min_vram and not has_cpu_offload:
                    cmd.extend([
                        "--kv-stream-arena-mib", str(arena_mib),
                        "--kv-stream-auxiliary-layers", str(auxiliary_layers),
                        "--spec-type", str(spec_type),
                        "--spec-draft-n-max", str(draft_max),
                        "--spec-draft-p-min", str(draft_p_min),
                        "--spec-draft-p-split", str(draft_p_split),
                        "--batch-size", str(batch_size),
                        "--ubatch-size", str(ubatch_size),
                        "--parallel", str(parallel),
                    ])
                    if no_mmproj:
                        cmd.append("--no-mmproj")

                    # Explicitly guarantee context size on CLI from preset or request
                    target_ctx = matched_opts.get("ctx-size") or (request.ContextSize if request.ContextSize > 0 else 262000)
                    try:
                        if int(target_ctx) > 0:
                            cmd.extend(["--ctx-size", str(target_ctx)])
                    except Exception:
                        pass
                else:
                    # Fault tolerance: Huang fork provides no benefit without MTP or when model size exceeds VRAM - 500MB.
                    # Fall back to Official Upstream llama-server and inform user.
                    official_server_exe = EXE_PATH_LLAMA_CPP
                    if not os.path.exists(official_server_exe):
                        from local_ai_launcher.backends.llamacpp import check_llamacpp
                        check_llamacpp()
                    if os.path.exists(official_server_exe):
                        server_exe = official_server_exe
                        self.current_server_exe = str(server_exe)
                        engine_label = "official llama-server"
                        cmd[0] = str(server_exe)

                    if exceeds_vram:
                        fallback_msg = (
                            f"[huang-llama] Model size ({m_size:.1f} MB) exceeds VRAM ceiling threshold "
                            f"({v_thresh:.1f} MB). Huang Adaptive-KV streaming arena cannot fit in remaining VRAM. "
                            f"Falling back to Official Upstream llama-server at {server_exe}."
                        )
                    elif below_min_vram:
                        fallback_msg = (
                            f"[huang-llama] Model size ({m_size:.1f} MB, {vram_ratio*100:.1f}% VRAM) below 63% threshold. "
                            f"Standard KV cache fits in VRAM without circular streaming arena. "
                            f"Falling back to Official Upstream llama-server at {server_exe}."
                        )
                    elif has_cpu_offload:
                        fallback_msg = (
                            f"[huang-llama] Model has CPU offload settings ({cpu_reason}). "
                            f"Huang Adaptive-KV requires text-only GPU VRAM execution. "
                            f"Falling back to Official Upstream llama-server at {server_exe}."
                        )
                    else:
                        fallback_msg = (
                            f"[huang-llama] MTP is disabled or absent for preset '{matched_section}'. "
                            f"Huang Adaptive-KV streaming provides no benefit without MTP. "
                            f"Falling back to Official Upstream llama-server at {server_exe}."
                        )
                    logger.warning(fallback_msg)
                    print(fallback_msg, flush=True)

                    from local_ai_launcher.vram_coordinator import calculate_safe_context_limit
                    target_ctx = int(matched_opts.get("ctx-size") or (request.ContextSize if request.ContextSize > 0 else 262000))
                    safe_fallback_ctx = calculate_safe_context_limit(
                        resolved_preset_model,
                        requested_ctx=target_ctx,
                        engine="official",
                        preset_opts=matched_opts
                    )
                    cmd.extend([
                        "--spec-type", "none",
                        "--parallel", str(parallel),
                    ])
                    if safe_fallback_ctx < target_ctx:
                        cmd.extend(["--ctx-size", str(safe_fallback_ctx)])

                    # For models undergoing Automatic Fallback to Official Upstream:
                    # Enforce --fit off non-duplicated if size > VRAM - 500MB without CPU offload
                    enforce_fallback_fit_off_if_needed(cmd, resolved_preset_model, matched_opts, request)
            elif not is_genuine_huang:
                from local_ai_launcher.vram_coordinator import calculate_safe_context_limit
                model_str = None
                try:
                    from configparser import ConfigParser
                    p = ConfigParser()
                    p.read(effective_preset_file, encoding="utf-8")
                    if p.has_section("default") and p.has_option("default", "model"):
                        model_str = p.get("default", "model")
                except Exception:
                    pass
                safe_fallback_ctx = calculate_safe_context_limit(
                    model_str,
                    requested_ctx=262000,
                    engine="official",
                    preset_opts=matched_opts
                )
                if safe_fallback_ctx < 262000:
                    cmd.extend(["--ctx-size", str(safe_fallback_ctx)])
                logger.warning(
                    f"[huang-llama] CAUTION: Running official llama-server fallback without streaming phase arena. "
                    f"Safe context limit evaluated at {safe_fallback_ctx} tokens to maintain hardware VRAM safety."
                )
                enforce_fallback_fit_off_if_needed(
                    cmd,
                    model_str or (matched_opts.get("model") if matched_opts else None),
                    matched_opts,
                    request,
                )
        else:
            model_p = Path(model_path)
            if not model_p.is_absolute():
                candidates = [
                    AI_WORKSPACE_DIR / "models" / "llama-server" / model_path,
                    AI_WORKSPACE_DIR / "models" / model_path,
                    AI_WORKSPACE_DIR / "models" / "gguf" / model_path,
                    AI_WORKSPACE_DIR / model_path,
                ]
                for c in candidates:
                    if c.exists():
                        model_p = c
                        break

            self.active_model_path = str(model_p)
            ctx_size = request.ContextSize if request.ContextSize > 0 else 262144
            if not is_genuine_huang:
                from local_ai_launcher.vram_coordinator import calculate_safe_context_limit
                req_opts = {}
                if request.EngineArgs:
                    try:
                        ea = json.loads(request.EngineArgs)
                        if isinstance(ea, dict):
                            req_opts.update(ea)
                    except Exception:
                        pass
                if request.NGPULayers != 0:
                    req_opts["ngl"] = request.NGPULayers
                if request.CacheTypeKey:
                    req_opts["ctk"] = request.CacheTypeKey
                if request.CacheTypeValue:
                    req_opts["ctv"] = request.CacheTypeValue
                if request.MMProj:
                    req_opts["mmproj"] = request.MMProj
                ctx_size = calculate_safe_context_limit(
                    model_p,
                    requested_ctx=ctx_size,
                    engine="official",
                    preset_opts=req_opts
                )
            gpu_layers = request.NGPULayers if request.NGPULayers != 0 else -1

            cmd = [
                str(server_exe),
                "--port", str(self.http_port),
                "--model", str(model_p),
                "--ctx-size", str(ctx_size),
                "--n-gpu-layers", str(gpu_layers),
                "--host", "127.0.0.1",
                "--jinja",
            ]

            if is_blackwell_target:
                cmd.extend(["--flash-attn", "on"])
                if is_genuine_huang:
                    mtp_requested = (
                        str(spec_type).lower() not in ("none", "off", "0", "false")
                        and arena_mib > 0
                    )
                    has_mtp = False
                    if mtp_requested:
                        has_mtp = check_model_supports_mtp(model_p)
                        if not has_mtp:
                            companion_mtp = model_p.with_name(model_p.stem + "-mtp" + model_p.suffix)
                            if os.path.exists(str(companion_mtp)) and check_model_supports_mtp(companion_mtp):
                                model_p = companion_mtp
                                self.active_model_path = str(model_p)
                                cmd[cmd.index("--model") + 1] = str(model_p)
                                has_mtp = True

                    exceeds_vram, m_size, v_thresh = check_model_exceeds_vram_threshold(model_p)
                    total_vram = get_system_vram_mb()
                    vram_ratio = (m_size / total_vram) if total_vram > 0 else 0.0
                    below_min_vram = (vram_ratio < HUANG_VRAM_MIN_RATIO) and (m_size > 0)
                    has_cpu_offload, cpu_reason = has_cpu_offload_settings({}, request)

                    if mtp_requested and has_mtp and not exceeds_vram and not below_min_vram and not has_cpu_offload:
                        cmd.extend([
                            "--kv-stream-arena-mib", str(arena_mib),
                            "--kv-stream-auxiliary-layers", str(auxiliary_layers),
                            "--spec-type", str(spec_type),
                            "--spec-draft-n-max", str(draft_max),
                            "--spec-draft-p-min", str(draft_p_min),
                            "--spec-draft-p-split", str(draft_p_split),
                            "--batch-size", str(batch_size),
                            "--ubatch-size", str(ubatch_size),
                            "--parallel", str(parallel),
                        ])
                        if no_mmproj:
                            cmd.append("--no-mmproj")
                    else:
                        # Fault tolerance: Huang fork provides no benefit without MTP or when model size exceeds VRAM - 500MB.
                        # Fall back to Official Upstream llama-server and inform user.
                        official_server_exe = EXE_PATH_LLAMA_CPP
                        if not os.path.exists(official_server_exe):
                            from local_ai_launcher.backends.llamacpp import check_llamacpp
                            check_llamacpp()
                        if os.path.exists(official_server_exe):
                            server_exe = official_server_exe
                            self.current_server_exe = str(server_exe)
                            engine_label = "official llama-server"
                            cmd[0] = str(server_exe)

                        if exceeds_vram:
                            fallback_msg = (
                                f"[huang-llama] Model size ({m_size:.1f} MB) exceeds VRAM ceiling threshold "
                                f"({v_thresh:.1f} MB). Huang Adaptive-KV streaming arena cannot fit in remaining VRAM. "
                                f"Falling back to Official Upstream llama-server at {server_exe}."
                            )
                        elif below_min_vram:
                            fallback_msg = (
                                f"[huang-llama] Model size ({m_size:.1f} MB, {vram_ratio*100:.1f}% VRAM) below 63% threshold. "
                                f"Standard KV cache fits in VRAM without circular streaming arena. "
                                f"Falling back to Official Upstream llama-server at {server_exe}."
                            )
                        elif has_cpu_offload:
                            fallback_msg = (
                                f"[huang-llama] Model has CPU offload settings ({cpu_reason}). "
                                f"Huang Adaptive-KV requires text-only GPU VRAM execution. "
                                f"Falling back to Official Upstream llama-server at {server_exe}."
                            )
                        else:
                            fallback_msg = (
                                f"[huang-llama] MTP is disabled or absent for model '{model_p.name}'. "
                                f"Huang Adaptive-KV streaming provides no benefit without MTP. "
                                f"Falling back to Official Upstream llama-server at {server_exe}."
                            )
                        logger.warning(fallback_msg)
                        print(fallback_msg, flush=True)

                        from local_ai_launcher.vram_coordinator import calculate_safe_context_limit
                        safe_ctx = calculate_safe_context_limit(model_p, requested_ctx=ctx_size, engine="official")
                        cmd[cmd.index("--ctx-size") + 1] = str(safe_ctx)
                        cmd.extend([
                            "--spec-type", "none",
                            "--parallel", str(parallel),
                        ])

                        # For models undergoing Automatic Fallback to Official Upstream:
                        # Enforce --fit off non-duplicated if size > VRAM - 500MB without CPU offload
                        enforce_fallback_fit_off_if_needed(cmd, model_p, {}, request)
            elif not is_genuine_huang:
                enforce_fallback_fit_off_if_needed(cmd, model_p, {}, request)

            if request.Threads > 0:
                cmd.extend(["--threads", str(request.Threads)])
            if request.NBatch > 0 and not (is_blackwell_target and is_genuine_huang):
                cmd.extend(["--batch-size", str(request.NBatch)])
            if request.FlashAttention and not is_blackwell_target:
                cmd.extend(["--flash-attn", request.FlashAttention])

            # KV Cache precision
            ctk = request.CacheTypeKey or ("q8_0" if is_blackwell_target else None)
            ctv = request.CacheTypeValue or ("q4_0" if is_blackwell_target else None)
            if ctk:
                cmd.extend(["-ctk", ctk])
            if ctv:
                cmd.extend(["-ctv", ctv])

            if request.DraftModel and os.path.exists(request.DraftModel):
                cmd.extend(["--model-draft", str(request.DraftModel)])
            if request.MLock:
                cmd.append("--mlock")
            if request.MMap:
                cmd.append("--mmap")
            if request.MMProj and os.path.exists(request.MMProj) and not (is_blackwell_target and is_genuine_huang and no_mmproj):
                cmd.extend(["--mmproj", str(request.MMProj)])

            if request.EngineArgs:
                try:
                    ea = json.loads(request.EngineArgs)
                    if isinstance(ea, dict):
                        for k, v in ea.items():
                            if k not in (
                                "model", "port", "host", "n-gpu-layers", "ngl", "ctx-size",
                                "threads", "batch-size", "ubatch-size", "flash-attn", "ctk", "ctv",
                                "model-draft", "mmproj", "alias", "kv-stream-arena-mib",
                                "kv-stream-auxiliary-layers", "spec-type", "spec-draft-n-max",
                                "spec-draft-p-min", "spec-draft-p-split",
                                "parallel", "no-mmproj"
                            ):
                                flag = f"--{k}"
                                if str(v).lower() in ("true", "yes", "on"):
                                    cmd.append(flag)
                                elif str(v).lower() not in ("false", "no", "off") and v:
                                    cmd.extend([flag, str(v)])
                except Exception:
                    pass

        if request.EnvVars:
            env.update(request.EnvVars)

        logger.info(f"[huang-llama] Launching {engine_label} ({Path(server_exe).name}) on port {self.http_port}...")
        try:
            creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            logs_dir = AI_WORKSPACE_DIR / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            self._log_file = open(logs_dir / "llama_server_engine.log", "a", encoding="utf-8")
            self.process = subprocess.Popen(
                cmd,
                cwd=str(AI_WORKSPACE_DIR),
                env=env,
                stdout=self._log_file,
                stderr=self._log_file,
                creationflags=creationflags
            )
            try:
                from local_ai_launcher.process import assign_process_to_job
                assign_process_to_job(self.process)
            except Exception:
                pass
            ready = self._wait_for_health(timeout=90)
            if not ready:
                self.Free(None, None)
                return backend_pb2.Result(success=False, message="Timed out waiting for Huang Adaptive-KV llama-server health")

            if matched_section:
                target_id = matched_section
                logger.info(f"[huang-llama] Triggering warmup load for preset '{target_id}'...")
                warmup_payload = {
                    "model": target_id,
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 1,
                    "temperature": 0.0
                }
                warmup_req = urllib.request.Request(
                    f"{self.http_base}/v1/chat/completions",
                    data=json.dumps(warmup_payload).encode("utf-8"),
                    headers={"Content-Type": "application/json", "User-Agent": "HuangWarmup"}
                )
                try:
                    with urllib.request.urlopen(warmup_req, timeout=120) as resp:
                        if resp.status == 200:
                            logger.info(f"[huang-llama] Preset '{target_id}' successfully preloaded into GPU VRAM!")
                except Exception as we:
                    logger.warning(f"[huang-llama] Warmup ping notice: {we}")

            self.is_ready = True
            self.current_model = matched_section or request.Model
            logger.info(f"[huang-llama] Model '{self.current_model}' ready on {self.http_base}")
            return backend_pb2.Result(success=True, message=f"Model loaded on {self.http_base}")
        except Exception as e:
            logger.error(f"[huang-llama] Failed to spawn llama-server: {e}")
            self.Free(None, None)
            return backend_pb2.Result(success=False, message=str(e))

    def Predict(self, request, context):
        if getattr(self, "_delegate", None):
            return self._delegate.Predict(request, context)

        if not self.is_ready or not self.http_base:
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            context.set_details("Huang Adaptive-KV llama backend not initialized")
            return backend_pb2.Reply(message=b"Backend not initialized")

        payload = self._build_request_payload(request, stream=False)
        if self.current_model and "model" not in payload:
            payload["model"] = self.current_model
        url = f"{self.http_base}/v1/chat/completions" if "messages" in payload else f"{self.http_base}/v1/completions"
        try:
            with httpx.Client(timeout=600.0) as client:
                resp = client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()

            if "choices" in data and len(data["choices"]) > 0:
                choice = data["choices"][0]
                msg = choice.get("message", {})
                content = msg.get("content") or choice.get("text", "")
                usage = data.get("usage") or {}
                tokens = usage.get("completion_tokens", 0) if isinstance(usage, dict) else 0
                prompt_tokens = usage.get("prompt_tokens", 0) if isinstance(usage, dict) else 0

                reply = backend_pb2.Reply(
                    message=json.dumps(data).encode("utf-8"),
                    tokens=tokens,
                    prompt_tokens=prompt_tokens
                )

                cd = backend_pb2.ChatDelta(
                    content=content or "",
                    reasoning_content=msg.get("reasoning_content") or ""
                )
                if msg.get("tool_calls"):
                    for idx, tc in enumerate(msg["tool_calls"]):
                        fn = tc.get("function", {})
                        cd.tool_calls.append(backend_pb2.ToolCallDelta(
                            index=tc.get("index", idx),
                            id=tc.get("id", "") or "",
                            name=fn.get("name", "") or "",
                            arguments=fn.get("arguments", "") or ""
                        ))
                reply.chat_deltas.append(cd)
                return reply
            return backend_pb2.Reply(message=b"")
        except Exception as e:
            logger.error(f"[huang-llama] Predict HTTP request failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return backend_pb2.Reply(message=b"")

    def PredictStream(self, request, context):
        if getattr(self, "_delegate", None):
            yield from self._delegate.PredictStream(request, context)
            return

        if not self.is_ready or not self.http_base:
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            context.set_details("Huang Adaptive-KV llama backend not initialized")
            return

        payload = self._build_request_payload(request, stream=True)
        if self.current_model and "model" not in payload:
            payload["model"] = self.current_model
        url = f"{self.http_base}/v1/chat/completions" if "messages" in payload else f"{self.http_base}/v1/completions"
        try:
            with httpx.Client(timeout=600.0) as client:
                with client.stream("POST", url, json=payload) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        if not line:
                            continue
                        if line.startswith("data: "):
                            raw_data = line[6:].strip()
                            if raw_data == "[DONE]":
                                break
                            try:
                                chunk = json.loads(raw_data)
                                if "choices" in chunk and len(chunk["choices"]) > 0:
                                    choice = chunk["choices"][0]
                                    delta_msg = choice.get("delta", {})
                                    content = delta_msg.get("content") or choice.get("text", "")
                                    tool_calls = delta_msg.get("tool_calls")

                                    reply = backend_pb2.Reply(message=raw_data.encode("utf-8"))
                                    cd = backend_pb2.ChatDelta(
                                        content=content or "",
                                        reasoning_content=delta_msg.get("reasoning_content") or ""
                                    )
                                    if tool_calls and isinstance(tool_calls, list):
                                        for idx, tc in enumerate(tool_calls):
                                            fn = tc.get("function", {})
                                            cd.tool_calls.append(backend_pb2.ToolCallDelta(
                                                index=tc.get("index", idx),
                                                id=tc.get("id", "") or "",
                                                name=fn.get("name", "") or "",
                                                arguments=fn.get("arguments", "") or ""
                                            ))
                                    reply.chat_deltas.append(cd)
                                    yield reply
                                elif "usage" in chunk or ("choices" in chunk and len(chunk["choices"]) == 0):
                                    yield backend_pb2.Reply(message=raw_data.encode("utf-8"))
                            except json.JSONDecodeError:
                                pass
        except Exception as e:
            logger.error(f"[huang-llama] PredictStream HTTP request failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))

    def _build_request_payload(self, request, stream: bool = False) -> dict:
        return build_common_llama_payload(request, current_model=self.current_model, stream=stream)

    def Free(self, request, context):
        if getattr(self, "_delegate", None):
            try:
                res = self._delegate.Free(request, context)
            except Exception as e:
                logger.warning(f"[huang-llama] Error freeing delegate: {e}")
                res = backend_pb2.Result(success=True, message=str(e))
            self._delegate = None
            self.process = None
            self.http_port = None
            self.http_base = None
            self.is_ready = False
            self.current_model = None
            self.active_model_path = None
            self.current_server_exe = None
            self.current_is_blackwell = None
            if self._log_file:
                try:
                    self._log_file.close()
                except Exception:
                    pass
                self._log_file = None
            return res

        if self.process:
            logger.info(f"[huang-llama] Terminating llama-server process (PID: {self.process.pid})...")
            try:
                from local_ai_launcher.process import kill_process_tree
                kill_process_tree(self.process)
                if hasattr(self.process, "wait"):
                    try:
                        self.process.wait(timeout=3.0)
                    except Exception:
                        pass
            except Exception:
                try:
                    self.process.terminate()
                    self.process.wait(timeout=3.0)
                except Exception:
                    self.process.kill()
            self.process = None

        if self._log_file:
            try:
                self._log_file.close()
            except Exception:
                pass
            self._log_file = None

        self.http_port = None
        self.http_base = None
        self.is_ready = False
        self.current_model = None
        self.active_model_path = None
        self.current_server_exe = None
        self.current_is_blackwell = None
        return backend_pb2.Result(success=True, message="Freed Huang Adaptive-KV llama-server process")


_active_servicer = None


def _cleanup_llama_on_exit():
    servicer = _active_servicer
    if servicer:
        try:
            servicer.Free(None, None)
        except Exception:
            pass
        if servicer.process and servicer.process.poll() is None:
            try:
                from local_ai_launcher.process import kill_process_tree
                kill_process_tree(servicer.process)
            except Exception:
                pass


_cleanup_huang_on_exit = _cleanup_llama_on_exit
atexit.register(_cleanup_llama_on_exit)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Raymond Huang Adaptive-KV llama-server gRPC Shim")
    parser.add_argument("--addr", default=f"127.0.0.1:{config.blackwell_shim_port}", help="gRPC address to bind")
    parser.add_argument("--port", type=int, help="Alternative port flag.")
    args, _ = parser.parse_known_args()

    port = args.port or int(args.addr.split(":")[-1])
    logger.info(f"Starting Raymond Huang Adaptive-KV llama-server gRPC shim on port {port}...")
    serve_shim(HuangLlamaServicer(), default_addr=args.addr, port=port)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
