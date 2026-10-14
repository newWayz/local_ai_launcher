"""
CPU-only Decision & Gating Subsystem.
====================================
Wires all discrete agent routing, gating, and triage decisions directly into the
canonical CPU von System One Gater (wfzyx/von) on Port 8083.
Legacy deprecated model names (open-jev-deberta-v3-large, clm-8b) are accepted
as aliases and resolve to the von engine.
Executes single-forward-pass typed decisions in ~15ms with 0 autoregressive token generation.

Topology Mapping:
- Port 8081: Unified CPU-only llama-server inference worker (-ngl 0).
             Hosts Decider-4B C++ worker by default; also hosts Gemma 4 Vision Director
             when media production directing is active (strictly isolated, non-overlapping workloads).
- Port 8082: Dedicated CPU Reranker (bge-reranker-v2-m3)
- Port 8083: Dedicated CPU von System One Gater (wfzyx/von; legacy aliases: open-jev, clm-8b)
- Port 8084: Dedicated CPU Decider-4B FastAPI Engine (Mapika/decider-4b v2.1 cognitive endpoints)
"""

from __future__ import annotations

import os
import sys
import math
import time
import socket
import logging
import subprocess
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple

from local_ai_launcher.config import (
    AI_WORKSPACE_DIR,
    INSTALL_DIR_LLAMA_CPP,
    EXE_PATH_LLAMA_CPP,
    MODELS_PRESET_FILE,
    config,
)
from local_ai_launcher.process import start_process, kill_process_tree, wait_for_port
from local_ai_launcher.backends.cpu_onnx_gater import (
    get_or_load_onnx_gater,
    evaluate_onnx_decision_sync,
    evaluate_onnx_decision_async,
    start_cpu_onnx_gater_server,
    stop_cpu_onnx_gater_server,
    DEFAULT_ONNX_GATER_MODEL,
    DEFAULT_ONNX_GATER_PORT,
)

logger = logging.getLogger("cpu_decision")

DEFAULT_DECISION_MODEL = "wfzyx/von"
DEFAULT_CPU_LLAMA_SERVER_PORT = 8081
_active_cpu_decision_proc = None


def is_port_open(port: int, host: str = "127.0.0.1") -> bool:
    """Checks if a TCP port is open and listening."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex((host, port)) == 0


def resolve_decision_model_path(model_name: Optional[str] = None) -> Tuple[Path, Optional[str], int]:
    """
    Resolves model path for CPU operations.
    - For ONNX gater, returns path to models/onnx-gaters/von
      (legacy aliases open-jev/clm-8b normalize to von upstream in config).
    - For CPU llama-server, returns path to Gemma 4 Director GGUF in models/llama-server.
    """
    target = (model_name or getattr(config, 'cpu_decision_model', DEFAULT_DECISION_MODEL)).lower()
    if any(k in target for k in ("gemma", "director", "media", "vision", "llama-server")):
        try:
            from local_ai_launcher.backends.cpu_director import resolve_director_model_paths
            p, _, ctx = resolve_director_model_paths(workspace_dir=AI_WORKSPACE_DIR)
            if p and p.exists():
                return p, "HauhauCS/Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced", ctx
        except Exception:
            pass
        fallback_gguf = (
            AI_WORKSPACE_DIR
            / "models"
            / "llama-server"
            / "HauhauCS"
            / "Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced"
            / "Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf"
        )
        return fallback_gguf, "HauhauCS/Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced", 16384

    # Legacy deprecated names (open-jev, deberta, clm-8b) kept as pure aliases:
    # they resolve to the von engine. Alias normalization also happens at config
    # ingest (LauncherConfig.__setattr__), so old wire requests keep working.
    if any(k in target for k in ("open-jev", "openjev", "deberta", "clm-8b", "clm-v0.1", "contrastive-lm", "stanford-nvidia", "von")):
        p = AI_WORKSPACE_DIR / "models" / "onnx-gaters" / "von"
        return p, "wfzyx/von", 768

    # Default: Dedicated CPU ONNX von Gater directory
    p = AI_WORKSPACE_DIR / "models" / "onnx-gaters" / "von"
    return p, "wfzyx/von", 768


def ensure_decision_model(model_name: Optional[str] = None) -> Path:
    """
    Verifies that the target decision model exists locally.
    Delegates to ensure_onnx_gater_model for the ONNX decision engine.
    """
    target = model_name or getattr(config, 'cpu_decision_model', DEFAULT_DECISION_MODEL)
    from local_ai_launcher.backends.cpu_onnx_gater import ensure_onnx_gater_model
    return ensure_onnx_gater_model(target)


def build_cpu_decision_cmd(
    model_name: Optional[str] = None,
    port: Optional[int] = None,
    threads: Optional[int] = None,
    ctx_size: Optional[int] = None
) -> List[str]:
    """
    Constructs CPU llama-server command line for Port 8081 (Media Director / Gemma 4).
    Distinct reference to CPU-only llama-server preserving -ngl 0 execution.
    """
    target_port = port or getattr(config, 'cpu_llama_server_port', 8081)
    target_threads = threads or getattr(config, 'cpu_llama_server_threads', 4)
    target_ctx = ctx_size or getattr(config, 'cpu_llama_server_context_size', 16384)

    try:
        from local_ai_launcher.backends.cpu_director import build_cpu_director_cmd
        cmd = build_cpu_director_cmd(
            port=target_port,
            threads=target_threads,
            ctx_size=target_ctx,
            workspace_dir=AI_WORKSPACE_DIR
        )
        if cmd:
            return cmd
    except Exception:
        pass

    exe = AI_WORKSPACE_DIR / "llama-cpp" / "llama-server.exe"
    if not exe.exists():
        exe = Path("llama-server")

    model_path = (
        AI_WORKSPACE_DIR
        / "models"
        / "llama-server"
        / "HauhauCS"
        / "Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced"
        / "Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf"
    )

    return [
        str(exe),
        "-m", str(model_path),
        "--port", str(target_port),
        "-ngl", "0",
        "-t", str(target_threads),
        "-c", str(target_ctx),
        "--host", "127.0.0.1",
        "--no-webui"
    ]


def start_cpu_decision_server(processes: Optional[list] = None):
    """
    Starts the dedicated CPU ONNX Decision Gater server on Port 8083.
    """
    return start_cpu_onnx_gater_server(processes=processes)


def stop_cpu_decision_server():
    """
    Stops the dedicated CPU ONNX Decision Gater server.
    """
    stop_cpu_onnx_gater_server()


def _process_logprobs(
    data: Dict[str, Any],
    candidate_options: List[str],
    temperature: float = 1.45
) -> Dict[str, Any]:
    """
    Extracts log probabilities from completion response,
    applies temperature scaling (default T=1.45 for Becker calibration),
    and returns normalized choice distribution.
    """
    choices = data.get("choices", [])
    if not choices:
        return {
            "choice": candidate_options[0] if candidate_options else None,
            "confidence": 0.5,
            "distribution": {c: 1.0 / len(candidate_options) for c in candidate_options} if candidate_options else {}
        }

    content_logprobs = choices[0].get("logprobs", {}).get("content", [])
    if not content_logprobs:
        text = choices[0].get("message", {}).get("content", "").strip()
        matched = None
        for opt in candidate_options:
            if opt.lower() == text.lower() or opt.lower() in text.lower():
                matched = opt
                break
        choice = matched or (candidate_options[0] if candidate_options else None)
        return {
            "choice": choice,
            "confidence": 0.7 if matched else 0.5,
            "distribution": {c: (1.0 if c == choice else 0.0) for c in candidate_options} if candidate_options else {}
        }

    top_logprobs = content_logprobs[0].get("top_logprobs", [])
    extracted: Dict[str, float] = {}
    options_map = {opt.lower(): opt for opt in candidate_options}

    for entry in top_logprobs:
        raw_tok = entry.get("token", "").strip().strip("\"'.,:;`")
        clean_tok_lower = raw_tok.lower()
        if clean_tok_lower in options_map:
            canonical = options_map[clean_tok_lower]
            lp = float(entry.get("logprob", -99.0))
            if canonical not in extracted or lp > extracted[canonical]:
                extracted[canonical] = lp

    if not extracted:
        first_token = content_logprobs[0].get("token", "").strip().strip("\"'.,:;`")
        for opt in candidate_options:
            if opt.lower() == first_token.lower():
                return {
                    "choice": opt,
                    "confidence": 0.8,
                    "distribution": {c: (1.0 if c == opt else 0.0) for c in candidate_options}
                }
        return {
            "choice": candidate_options[0] if candidate_options else None,
            "confidence": 0.5,
            "distribution": {c: 1.0 / len(candidate_options) for c in candidate_options} if candidate_options else {}
        }

    min_found_lp = min(extracted.values())
    floor_lp = min_found_lp - 5.0
    for opt in candidate_options:
        if opt not in extracted:
            extracted[opt] = floor_lp

    temp = max(0.01, temperature)
    scaled_exp = {k: math.exp(v / temp) for k, v in extracted.items()}
    total_sum = sum(scaled_exp.values()) if scaled_exp else 1.0
    probabilities = {k: round(v / total_sum, 4) for k, v in scaled_exp.items()}

    best_choice = max(probabilities, key=probabilities.get) if probabilities else None
    confidence = probabilities.get(best_choice, 0.0) if best_choice else 0.0

    return {
        "choice": best_choice,
        "confidence": confidence,
        "distribution": probabilities
    }


def evaluate_decision_sync(
    prompt: str,
    candidate_options: List[str],
    port: Optional[int] = None,
    temperature: float = 1.45,
    system_prompt: str = "You are a fast-path decision engine. Reply ONLY with exactly one choice token.",
    cyclic_permutation: bool = False,
    timeout: float = 5.0
) -> Dict[str, Any]:
    """
    Synchronously evaluates a discrete decision via the dedicated CPU ONNX Gater.
    Runs single-pass in ~15ms without autoregressive token generation.
    """
    target_port = port or getattr(config, 'cpu_onnx_gater_port', DEFAULT_ONNX_GATER_PORT)
    return evaluate_onnx_decision_sync(
        prompt=prompt,
        candidate_options=candidate_options,
        port=target_port,
        temperature=temperature
    )


async def evaluate_decision_async(
    prompt: str,
    candidate_options: List[str],
    port: Optional[int] = None,
    temperature: float = 1.45,
    system_prompt: str = "You are a fast-path decision engine. Reply ONLY with exactly one choice token.",
    cyclic_permutation: bool = False,
    timeout: float = 5.0
) -> Dict[str, Any]:
    """
    Asynchronously evaluates a discrete decision via the dedicated CPU ONNX Gater on Port 8083.
    Non-blocking async query with zero autoregressive generation.
    """
    target_port = port or getattr(config, 'cpu_onnx_gater_port', DEFAULT_ONNX_GATER_PORT)
    return await evaluate_onnx_decision_async(
        prompt=prompt,
        candidate_options=candidate_options,
        port=target_port,
        temperature=temperature,
        timeout=timeout
    )


def check_cpu_engine_health(port: Optional[int] = None, timeout: float = 1.0) -> bool:
    """Fast health check for CPU decision server (Port 8083)."""
    target_port = port or getattr(config, 'cpu_onnx_gater_port', DEFAULT_ONNX_GATER_PORT)
    return is_port_open(target_port)


async def compact_text_async(
    text: str,
    max_summary_tokens: int = 500,
    port: Optional[int] = None,
    timeout: float = 12.0
) -> Optional[str]:
    """
    Asynchronously compresses verbose context (code diffs, compiler diagnostics, logs)
    using the CPU llama-server (Port 8081) or primary LocalAI gateway (Port 8080).
    """
    llama_port = port or (8081 if is_port_open(8081) else getattr(config, 'localai_port', 8080))
    url = f"http://127.0.0.1:{llama_port}/v1/chat/completions"

    system_prompt = (
        "You are an expert context compaction engine. Extract the critical facts, error causes, "
        "modified symbols, and invariant violations. Return ONLY the high-density extractive summary."
    )
    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Compress this context:\n\n{text}"}
        ],
        "max_tokens": max_summary_tokens,
        "temperature": 0.2
    }

    try:
        from local_ai_launcher.dispatcher import get_shared_async_client
        client = get_shared_async_client(timeout=timeout)
        resp = await client.post(url, json=payload)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    except Exception as e:
        logger.debug(f"Compaction notice: {e}")

    # Fallback local deterministic compaction if no completion server is online
    return _local_extractive_compaction(text, max_lines=25)


def compact_text_sync(
    text: str,
    max_summary_tokens: int = 500,
    port: Optional[int] = None,
    timeout: float = 12.0
) -> Optional[str]:
    """
    Synchronously compresses verbose context using the CPU llama-server (Port 8081) or primary gateway.
    """
    import urllib.request
    import json

    llama_port = port or (8081 if is_port_open(8081) else getattr(config, 'localai_port', 8080))
    url = f"http://127.0.0.1:{llama_port}/v1/chat/completions"

    system_prompt = (
        "You are an expert context compaction engine. Extract the critical facts, error causes, "
        "modified symbols, and invariant violations. Return ONLY the high-density extractive summary."
    )
    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Compress this context:\n\n{text}"}
        ],
        "max_tokens": max_summary_tokens,
        "temperature": 0.2
    }

    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    except Exception as e:
        logger.debug(f"Compaction sync notice: {e}")

    return _local_extractive_compaction(text, max_lines=25)


def _local_extractive_compaction(text: str, max_lines: int = 25) -> str:
    """Deterministic local compaction extracting essential lines without LLM overhead."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) <= max_lines:
        return text.strip()

    critical = []
    keywords = ("error", "failed", "exception", "traceback", "def ", "class ", "diff --git", "@@", "+", "-")
    for line in lines:
        if any(line.lower().startswith(k) or f" {k}" in line.lower() for k in keywords):
            critical.append(line)
        if len(critical) >= max_lines:
            break

    if not critical:
        critical = lines[:max_lines]

    return "\n".join(critical)
