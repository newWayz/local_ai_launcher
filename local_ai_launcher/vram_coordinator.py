import sys
import time
import json
import argparse
import logging
import urllib.request
import urllib.error
from typing import Optional, Dict, Any

from local_ai_launcher.config import config, LLM_BACKEND_PORT

logger = logging.getLogger("vram_coordinator")


def get_vram_info() -> Dict[str, Any]:
    """Inspect GPU VRAM usage if torch / nvidia-smi is available."""
    info = {"available": False, "allocated_mb": 0, "reserved_mb": 0, "free_mb": 0, "total_mb": 0}
    try:
        import torch
        if torch.cuda.is_available():
            info["available"] = True
            info["allocated_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
            info["reserved_mb"] = torch.cuda.memory_reserved() / (1024 * 1024)
            free, total = torch.cuda.mem_get_info()
            info["free_mb"] = free / (1024 * 1024)
            info["total_mb"] = total / (1024 * 1024)
            return info
    except Exception:
        pass

    # Fallback to nvidia-smi via subprocess
    try:
        import subprocess
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=3
        )
        if res.returncode == 0:
            lines = res.stdout.strip().splitlines()
            if lines:
                parts = [p.strip() for p in lines[0].split(",")]
                if len(parts) >= 3:
                    info["available"] = True
                    info["total_mb"] = float(parts[0])
                    info["allocated_mb"] = float(parts[1])
                    info["free_mb"] = float(parts[2])
    except Exception:
        pass

    return info


def evict_llm_vram(localai_url: Optional[str] = None, timeout: int = 15, force: bool = False, required_mb: int = 12288) -> bool:
    """
    Evicts all active LLM backends from GPU VRAM via LocalAI shutdown endpoint.
    Used before ComfyUI Gate 2 (FLUX image gen) and Gate 3 (Wan 14B video gen).
    Respects hardware profile policy:
      - Strict eviction (16GB / 12GB): always frees VRAM for diffusion.
      - Opportunistic eviction (24GB+ rigs): if free_mb >= required_mb, skips eviction.
    """
    from local_ai_launcher.config import get_active_hardware_profile
    active_profile = get_active_hardware_profile()
    is_strict = active_profile.get("strict_vram_eviction", True)

    if not force and not is_strict:
        vinfo = get_vram_info()
        free_mb = vinfo.get("free_mb", 0)
        if free_mb >= required_mb:
            logger.info(
                f"[VRAM Coordinator] Sufficient free VRAM ({free_mb:.0f} MB >= {required_mb} MB) "
                f"for profile '{active_profile.get('name')}'; keeping LLM loaded in VRAM."
            )
            return True

    base_url = localai_url or f"http://{config.localai_host}:{config.localai_port}"
    shutdown_url = f"{base_url}/backend/shutdown"

    logger.info(f"[VRAM Coordinator] Evicting LLM from VRAM via {shutdown_url}...")
    success = False

    try:
        req = urllib.request.Request(
            shutdown_url,
            data=b"{}",
            headers={"Content-Type": "application/json", "User-Agent": "VRAMCoordinator"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status in (200, 204):
                logger.info("[VRAM Coordinator] LocalAI accepted backend shutdown request.")
                success = True
            else:
                logger.warning(f"[VRAM Coordinator] Received status code {resp.status} on shutdown.")
    except urllib.error.HTTPError as e:
        if e.code in (200, 204, 404):
            # 404 might mean no backend is currently running or endpoint unsupported
            success = True
        else:
            logger.warning(f"[VRAM Coordinator] HTTP error on shutdown: {e}")
    except Exception as e:
        logger.warning(f"[VRAM Coordinator] Connection error calling shutdown: {e}")

    # Allow 2 seconds for GPU driver to release allocations
    time.sleep(2.0)

    vram = get_vram_info()
    if vram["available"]:
        logger.info(f"[VRAM Coordinator] VRAM Status after eviction: Free: {vram['free_mb']:.0f} MB / Total: {vram['total_mb']:.0f} MB")

    return success


def restore_llm(model_name: Optional[str] = None, localai_url: Optional[str] = None) -> bool:
    """
    Optional: Trigger lazy reload of default LLM model after ComfyUI completes.
    In LocalAI architecture, this is usually automatic on the next agent prompt.
    """
    base_url = localai_url or f"http://{config.localai_host}:{config.localai_port}"
    target_model = model_name or config.default_model
    chat_url = f"{base_url}/v1/chat/completions"

    logger.info(f"[VRAM Coordinator] Triggering lazy-load of model '{target_model}'...")
    payload = json.dumps({
        "model": target_model,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
        "temperature": 0.0
    }).encode("utf-8")

    try:
        req = urllib.request.Request(
            chat_url,
            data=payload,
            headers={"Content-Type": "application/json", "User-Agent": "VRAMCoordinator"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status == 200
    except Exception as e:
        logger.warning(f"[VRAM Coordinator] Restore ping notification: {e}")
        return False


def calculate_safe_context_limit(
    model_path_or_size,
    requested_ctx: int = 262000,
    engine: str = "official",
    hardware_profile: Optional[Dict[str, Any]] = None,
    preset_opts: Optional[Dict[str, Any]] = None,
) -> int:
    """
    Portably calculates the maximum safe context token limit to prevent VRAM oversubscription.
    - If engine natively supports streaming phase arena (Huang Adaptive-KV with bounded arena)
      or sparse host DDR5 ring (Bruno Passini): returns requested_ctx unchanged.
    - If engine allocates static KV cache in VRAM (standard official llama-server):
      calculates physical VRAM headroom taking into account CPU MoE and layer offloading,
      and clamps ctx_size so total allocation never exceeds safe hardware ceiling (~80-85%
      of physical VRAM), preventing WDDM paging deadlocks and BSODs.
    """
    clean_engine = str(engine or "").lower().strip()
    if clean_engine in ("huang", "huang-llama", "bruno", "bruno-llama", "blackwell-llama"):
        return requested_ctx

    from pathlib import Path
    from local_ai_launcher.config import get_active_hardware_profile, AI_WORKSPACE_DIR
    profile = hardware_profile or get_active_hardware_profile()
    vram_ceiling = float(profile.get("vram_ceiling_mb", 16384))

    # Resolve preset options if not explicitly supplied
    opts = dict(preset_opts or {})
    if not opts and isinstance(model_path_or_size, (str, Path)):
        try:
            from local_ai_launcher.backends.llamacpp_router import resolve_preset_data
            _, resolved_opts = resolve_preset_data(str(model_path_or_size))
            if resolved_opts:
                opts = dict(resolved_opts)
        except Exception:
            pass

    # 1. Determine model weight size in MiB
    model_mb = 12000.0
    if isinstance(model_path_or_size, (int, float)):
        model_mb = float(model_path_or_size)
    elif isinstance(model_path_or_size, (str, Path)):
        p = Path(model_path_or_size)
        if not p.is_absolute():
            candidates = [
                AI_WORKSPACE_DIR / "models" / "llama-server" / p,
                AI_WORKSPACE_DIR / "models" / p,
                AI_WORKSPACE_DIR / p,
            ]
            for c in candidates:
                if c.is_file():
                    p = c
                    break
        if p.is_file():
            try:
                model_mb = p.stat().st_size / (1024 * 1024)
            except Exception:
                pass
        else:
            name_lower = str(model_path_or_size).lower()
            if any(k in name_lower for k in ("70b", "72b")):
                model_mb = 38000.0
            elif any(k in name_lower for k in ("35b", "36b")):
                model_mb = 16000.0
            elif any(k in name_lower for k in ("14b", "12b")):
                model_mb = 8500.0
            elif any(k in name_lower for k in ("7b", "8b")):
                model_mb = 4500.0
            elif any(k in name_lower for k in ("3b", "1.5b")):
                model_mb = 2000.0

    # 1b. Check for CPU offloading settings (n-cpu-moe, ngl / n-gpu-layers)
    n_cpu_moe = 0
    if "n-cpu-moe" in opts:
        try:
            n_cpu_moe = int(opts["n-cpu-moe"])
        except Exception:
            pass

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

    is_moe = n_cpu_moe > 0
    if n_cpu_moe > 0:
        # In MoE models, active dense base is ~10-15% of total parameters.
        # Offloading n_cpu_moe expert layers places the vast majority of weights in host DDR RAM.
        total_moe_layers = max(n_cpu_moe + 2, 28)
        cpu_offload_ratio = min(0.85, n_cpu_moe / total_moe_layers)
        gpu_resident_mb = model_mb * (1.0 - cpu_offload_ratio)
        logger.debug(
            f"[calculate_safe_context_limit] MoE offloading active (n-cpu-moe={n_cpu_moe}). "
            f"Estimated GPU-resident model weights: {gpu_resident_mb:.0f}MB out of {model_mb:.0f}MB."
        )
        model_mb = gpu_resident_mb
    elif ngl is not None:
        if ngl == 0:
            # 100% CPU execution
            model_mb = 0.0
        elif 0 < ngl < 99:
            # Partial GPU offload
            total_layers = max(ngl + 5, 36)
            gpu_fraction = min(1.0, max(0.05, ngl / total_layers))
            gpu_resident_mb = model_mb * gpu_fraction
            logger.debug(
                f"[calculate_safe_context_limit] Partial GPU offloading active (ngl={ngl}). "
                f"Estimated GPU-resident model weights: {gpu_resident_mb:.0f}MB out of {model_mb:.0f}MB."
            )
            model_mb = gpu_resident_mb

    # Account for multimodal vision projector if configured
    if "mmproj" in opts and opts["mmproj"]:
        mm_mb = 1500.0
        mm_p = Path(opts["mmproj"])
        if not mm_p.is_absolute():
            mm_candidates = [
                AI_WORKSPACE_DIR / "models" / "llama-server" / mm_p,
                AI_WORKSPACE_DIR / "models" / mm_p,
                AI_WORKSPACE_DIR / mm_p,
            ]
            for mc in mm_candidates:
                if mc.is_file():
                    mm_p = mc
                    break
        if mm_p.is_file():
            try:
                mm_mb = mm_p.stat().st_size / (1024 * 1024)
            except Exception:
                pass
        model_mb += mm_mb

    # 2. Reserve safety cushion for Windows DWM compositor, display, and compute scratch
    compositor_and_scratch_mb = 1500.0
    usable_kv_vram_mb = max(512.0, (vram_ceiling * 0.95) - model_mb - compositor_and_scratch_mb)

    # 3. Estimate KV bytes per token based on architecture layers and cache types (q4_0, q8_0, etc.)
    ctk = str(opts.get("ctk", "q8_0")).lower().strip()
    ctv = str(opts.get("ctv", "q8_0")).lower().strip()

    # MoE models with 28-32 layers consume ~56 KB/tok (Q8) vs dense 64-layer models at ~96 KB/tok (Q8)
    base_kb = 56.0 if is_moe else 96.0
    if ("q4" in ctk or "turbo4" in ctk) and ("q4" in ctv or "turbo4" in ctv):
        kb_per_token = (base_kb * 0.5) / 1024.0
    elif "q4" in ctk or "q4" in ctv or "turbo4" in ctk or "turbo4" in ctv:
        kb_per_token = (base_kb * 0.75) / 1024.0
    else:
        kb_per_token = base_kb / 1024.0

    max_safe_tokens = int(usable_kv_vram_mb / kb_per_token)

    for threshold in (262144, 131072, 65536, 32768, 16384, 8192, 4096):
        if max_safe_tokens >= threshold:
            max_safe_tokens = threshold
            break
    else:
        max_safe_tokens = max(2048, (max_safe_tokens // 1024) * 1024)

    if requested_ctx > max_safe_tokens:
        logger.warning(
            f"[VRAM Safety Gate] Engine '{engine}' lacks streaming KV arena on {vram_ceiling:.0f}MB hardware. "
            f"Clamped context from {requested_ctx} to safe limit {max_safe_tokens} tokens "
            f"(GPU weights: {model_mb:.0f}MB, Available KV: {usable_kv_vram_mb:.0f}MB) to prevent system crash."
        )
        return max_safe_tokens

    return requested_ctx


def main():
    parser = argparse.ArgumentParser(description="VRAM Coordinator for LocalAI & ComfyUI")
    parser.add_argument("--evict", action="store_true", help="Evict active LLM backend from VRAM before ComfyUI Gate 2/3")
    parser.add_argument("--restore", action="store_true", help="Warm up/restore default model in LocalAI after ComfyUI")
    parser.add_argument("--status", action="store_true", help="Check current GPU VRAM status")
    parser.add_argument("--model", type=str, help="Model name for restore operation")
    parser.add_argument("--url", type=str, help="LocalAI base URL (default: http://127.0.0.1:8080)")

    args = parser.parse_args()

    if args.evict:
        ok = evict_llm_vram(args.url)
        sys.exit(0 if ok else 1)
    elif args.restore:
        ok = restore_llm(args.model, args.url)
        sys.exit(0 if ok else 1)
    elif args.status:
        v = get_vram_info()
        print(json.dumps(v, indent=2))
        sys.exit(0)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
