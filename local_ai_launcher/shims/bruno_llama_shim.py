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
    EXE_PATH_LLAMA_CPP_BLACKWELL,
    EXE_PATH_LLAMA_CPP_BRUNO,
    EXE_PATH_LLAMA_CPP,
    MODELS_PRESET_FILE,
    config,
    resolve_effective_models_preset,
    load_effective_models_parser,
    get_active_hardware_profile,
)
from local_ai_launcher.localai.proto import backend_pb2, backend_pb2_grpc
from local_ai_launcher.shims.base_shim import BaseBackendServicer, serve_shim, build_common_llama_payload
from local_ai_launcher.backends.bruno_builder import (
    resolve_bruno_server_exe,
    get_bruno_runtime_env,
    BRUNO_RUNTIME_ENV,
    is_bruno_model_allowed,
    resolve_bruno_kv_cache_types,
)
from local_ai_launcher.backends.llamacpp_router import (
    enforce_fallback_fit_off_if_needed,
    should_force_fit_off,
)

logger = logging.getLogger("bruno_llama_shim")


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


class BrunoLlamaServicer(BaseBackendServicer):
    """
    gRPC shim wrapping Bruno Passini's 256K Hybrid-KV Blackwell llama-server
    (with 36 hybrid-KV runtime env vars and sm_120a MMA tensor cores).
    Fulfills LocalAI's backend.proto contract.
    """
    def __init__(self):
        super().__init__("bruno-llama")
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
                logger.error(f"[bruno-llama] Subprocess exited early with code {self.process.returncode}. Check logs/llama_server_engine.log")
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
        """Launches Bruno Passini's specialized Blackwell llama-server process with the requested model."""
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
            logger.warning(f"[bruno-llama] Failed reading effective models preset: {e}")

        # Check active hardware profile
        active_profile = get_active_hardware_profile()
        profile_name = active_profile.get("name", "blackwell_16gb")

        # Check if targeting Blackwell specialized fork
        matched_aliases = [a.strip().lower() for a in matched_opts.get("alias", "").split(",") if a.strip()]
        has_blackwell_cue = (
            profile_name == "blackwell_16gb" and (
                "blackwell" in model_name.lower() or
                "blackwell" in model_path.lower() or
                "bruno" in model_name.lower() or
                "huang" in model_name.lower() or
                is_bruno_model_allowed(model_name, model_path) or
                (matched_section and ("blackwell" in matched_section.lower() or "bruno" in matched_section.lower() or "huang" in matched_section.lower())) or
                any("blackwell" in a or "bruno" in a or "huang" in a for a in matched_aliases) or
                (matched_opts.get("engine", "").lower() in ("blackwell", "bruno", "huang"))
            )
        )

        is_blackwell_target = False
        if has_blackwell_cue:
            from local_ai_launcher.backends.llamacpp_router import resolve_llamacpp_variant
            variant, route_reason, route_meta = resolve_llamacpp_variant(
                model_name=model_name,
                model_path=model_path,
                preset_opts=matched_opts,
                req_obj=request,
            )
            if variant == "bruno":
                is_blackwell_target = True
            elif variant == "huang":
                logger.warning(
                    f"[bruno-llama] Notice: Model '{model_name or model_path}' routed to Raymond Huang Adaptive-KV "
                    f"({route_reason}); delegating to Huang engine..."
                )
                from local_ai_launcher.shims.huang_llama_shim import HuangLlamaServicer
                self._delegate = HuangLlamaServicer()
                res = self._delegate.LoadModel(request, context)
                if not res.success:
                    logger.error(f"[bruno-llama] Delegate HuangLlamaServicer failed to load model: {res.message}")
                    self.Free(None, None)
                    return res
                self.process = self._delegate.process
                self.http_port = self._delegate.http_port
                self.http_base = self._delegate.http_base
                self.is_ready = self._delegate.is_ready
                self.current_model = self._delegate.current_model
                self._log_file = self._delegate._log_file
                self.active_model_path = self._delegate.active_model_path
                self.current_server_exe = self._delegate.current_server_exe
                self.current_is_blackwell = self._delegate.current_is_blackwell
                return res
            else:
                logger.info(
                    f"[bruno-llama] Model '{model_name or model_path}' routed to Official Upstream ({route_reason}). "
                    f"Falling back to official llama-server at {EXE_PATH_LLAMA_CPP}."
                )
                is_blackwell_target = False

        if is_blackwell_target:
            server_exe = resolve_bruno_server_exe() or EXE_PATH_LLAMA_CPP_BLACKWELL
            if not os.path.exists(server_exe):
                if os.path.exists(EXE_PATH_LLAMA_CPP):
                    logger.warning(f"[bruno-llama] Bruno binary missing at {server_exe}; falling back to {EXE_PATH_LLAMA_CPP}")
                    server_exe = EXE_PATH_LLAMA_CPP
                else:
                    from local_ai_launcher.backends.llamacpp import check_llamacpp
                    logger.info(f"[bruno-llama] Notice: llama-server.exe missing. Attempting automatic installation from GitHub releases...")
                    if check_llamacpp() and os.path.exists(EXE_PATH_LLAMA_CPP):
                        server_exe = EXE_PATH_LLAMA_CPP
            env = get_bruno_runtime_env()
            engine_label = "Bruno Blackwell hybrid-KV llama-server"
        else:
            if os.path.exists(EXE_PATH_LLAMA_CPP):
                server_exe = EXE_PATH_LLAMA_CPP
            elif os.path.exists(EXE_PATH_LLAMA_CPP_BLACKWELL):
                server_exe = EXE_PATH_LLAMA_CPP_BLACKWELL
            else:
                from local_ai_launcher.backends.llamacpp import check_llamacpp
                logger.info(f"[bruno-llama] Notice: llama-server.exe missing. Attempting automatic installation from GitHub releases...")
                if check_llamacpp() and os.path.exists(EXE_PATH_LLAMA_CPP):
                    server_exe = EXE_PATH_LLAMA_CPP
                else:
                    server_exe = resolve_bruno_server_exe() or EXE_PATH_LLAMA_CPP_BLACKWELL
            env = os.environ.copy()
            fork_label = getattr(config, 'llamacpp_fork', 'official')
            engine_label = f"{fork_label} llama-server"

        hf_home = str(AI_WORKSPACE_DIR / "models" / "huggingface")
        env["HOME"] = str(AI_WORKSPACE_DIR)
        env["HF_HOME"] = hf_home
        env["HF_HUB_CACHE"] = hf_home

        self.current_server_exe = str(server_exe)
        self.current_is_blackwell = is_blackwell_target

        if not os.path.exists(server_exe):
            return backend_pb2.Result(success=False, message=f"llama-server binary not found at {server_exe}")

        # Resolve symmetric KV cache types (q8/q8 or q4/q4, choosing lower if mixed)
        raw_k = matched_opts.get("ctk") or request.CacheTypeKey or "q4_0"
        raw_v = matched_opts.get("ctv") or request.CacheTypeValue or "q4_0"
        k_cache, v_cache = resolve_bruno_kv_cache_types(raw_k, raw_v)

        raw_k_draft = matched_opts.get("cache-type-k-draft") or "q4_0"
        raw_v_draft = matched_opts.get("cache-type-v-draft") or "q4_0"
        k_draft, v_draft = resolve_bruno_kv_cache_types(raw_k_draft, raw_v_draft)

        if is_blackwell_target and (raw_k != k_cache or raw_v != v_cache):
            logger.info(
                f"[bruno-llama] Normalized KV cache from ({raw_k}, {raw_v}) to "
                f"symmetric ({k_cache}, {v_cache}) for Bruno engine stability."
            )

        if matched_section:
            overlay_msg = " (with local overlay)" if is_overridden else ""
            logger.info(f"[bruno-llama] Matched preset section '{matched_section}' in {effective_preset_file.name}{overlay_msg}")
            cmd = [
                str(server_exe),
                "--port", str(self.http_port),
                "--models-preset", str(effective_preset_file),
                "--models-max", "1",
                "--host", "127.0.0.1",
                "--jinja",
            ]
            if is_blackwell_target:
                cmd.extend([
                    "--cache-type-k", k_cache,
                    "--cache-type-v", v_cache,
                    "--cache-type-k-draft", k_draft,
                    "--cache-type-v-draft", v_draft,
                ])
            else:
                from local_ai_launcher.vram_coordinator import calculate_safe_context_limit
                preset_m = matched_opts.get("model", "")
                target_ctx = int(matched_opts.get("ctx-size", 262000))
                safe_fallback_ctx = calculate_safe_context_limit(
                    preset_m,
                    requested_ctx=target_ctx,
                    engine="official",
                    preset_opts=matched_opts
                )
                if safe_fallback_ctx < target_ctx:
                    cmd.extend(["--ctx-size", str(safe_fallback_ctx)])
                cmd.extend(["--spec-type", "none"])
                enforce_fallback_fit_off_if_needed(cmd, preset_m, matched_opts, request)
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
            ctx_size = request.ContextSize if request.ContextSize > 0 else 262000
            if not is_blackwell_target:
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

            if request.Threads > 0:
                cmd.extend(["--threads", str(request.Threads)])
            if request.NBatch > 0:
                cmd.extend(["--batch-size", str(request.NBatch)])
            if request.FlashAttention:
                cmd.extend(["--flash-attn", request.FlashAttention])

            if is_blackwell_target:
                cmd.extend([
                    "--cache-type-k", k_cache,
                    "--cache-type-v", v_cache,
                    "--cache-type-k-draft", k_draft,
                    "--cache-type-v-draft", v_draft,
                ])
            else:
                if request.CacheTypeKey:
                    cmd.extend(["--cache-type-k", request.CacheTypeKey])
                if request.CacheTypeValue:
                    cmd.extend(["--cache-type-v", request.CacheTypeValue])
                cmd.extend(["--spec-type", "none"])
                enforce_fallback_fit_off_if_needed(cmd, model_p, {}, request)

            if request.DraftModel and os.path.exists(request.DraftModel):
                cmd.extend(["--model-draft", str(request.DraftModel)])
            if request.MLock:
                cmd.append("--mlock")
            if request.MMap:
                cmd.append("--mmap")
            if request.MMProj and os.path.exists(request.MMProj):
                cmd.extend(["--mmproj", str(request.MMProj)])

            if request.EngineArgs:
                try:
                    ea = json.loads(request.EngineArgs)
                    if isinstance(ea, dict):
                        for k, v in ea.items():
                            if k not in ("model", "port", "host", "n-gpu-layers", "ngl", "ctx-size", "threads", "batch-size", "flash-attn", "ctk", "ctv", "model-draft", "mmproj", "alias"):
                                flag = f"--{k}"
                                if str(v).lower() in ("true", "yes", "on"):
                                    cmd.append(flag)
                                elif str(v).lower() not in ("false", "no", "off") and v:
                                    cmd.extend([flag, str(v)])
                except Exception:
                    pass

        if request.EnvVars:
            env.update(request.EnvVars)

        logger.info(f"[bruno-llama] Launching {engine_label} ({Path(server_exe).name}) on port {self.http_port}...")
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
                return backend_pb2.Result(success=False, message="Timed out waiting for Bruno Blackwell llama-server health")

            if matched_section:
                target_id = matched_section
                logger.info(f"[bruno-llama] Triggering warmup load for preset '{target_id}'...")
                warmup_payload = {
                    "model": target_id,
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 1,
                    "temperature": 0.0
                }
                warmup_req = urllib.request.Request(
                    f"{self.http_base}/v1/chat/completions",
                    data=json.dumps(warmup_payload).encode("utf-8"),
                    headers={"Content-Type": "application/json", "User-Agent": "BrunoWarmup"}
                )
                try:
                    with urllib.request.urlopen(warmup_req, timeout=120) as resp:
                        if resp.status == 200:
                            logger.info(f"[bruno-llama] Preset '{target_id}' successfully preloaded into GPU VRAM!")
                except Exception as we:
                    logger.warning(f"[bruno-llama] Warmup ping notice: {we}")

            self.is_ready = True
            self.current_model = matched_section or request.Model
            logger.info(f"[bruno-llama] Model '{self.current_model}' ready on {self.http_base}")
            return backend_pb2.Result(success=True, message=f"Model loaded on {self.http_base}")
        except Exception as e:
            logger.error(f"[bruno-llama] Failed to spawn llama-server: {e}")
            self.Free(None, None)
            return backend_pb2.Result(success=False, message=str(e))

    def Predict(self, request, context):
        if getattr(self, "_delegate", None):
            return self._delegate.Predict(request, context)

        if not self.is_ready or not self.http_base:
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            context.set_details("Bruno Blackwell llama backend not initialized")
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
            logger.error(f"[bruno-llama] Predict HTTP request failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return backend_pb2.Reply(message=b"")

    def PredictStream(self, request, context):
        if getattr(self, "_delegate", None):
            yield from self._delegate.PredictStream(request, context)
            return

        if not self.is_ready or not self.http_base:
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            context.set_details("Bruno Blackwell llama backend not initialized")
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
            logger.error(f"[bruno-llama] PredictStream HTTP request failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))

    def _build_request_payload(self, request, stream: bool = False) -> dict:
        return build_common_llama_payload(request, current_model=self.current_model, stream=stream)

    def Free(self, request, context):
        if getattr(self, "_delegate", None):
            try:
                res = self._delegate.Free(request, context)
            except Exception as e:
                logger.warning(f"[bruno-llama] Error freeing delegate: {e}")
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
            logger.info(f"[bruno-llama] Terminating llama-server process (PID: {self.process.pid})...")
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
        return backend_pb2.Result(success=True, message="Freed Bruno Blackwell llama-server process")


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


_cleanup_bruno_on_exit = _cleanup_llama_on_exit
atexit.register(_cleanup_llama_on_exit)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Bruno Passini Blackwell llama-server gRPC Shim")
    parser.add_argument("--addr", default=f"127.0.0.1:{config.blackwell_shim_port}", help="gRPC address to bind")
    parser.add_argument("--port", type=int, help="Alternative port flag.")
    args, _ = parser.parse_known_args()

    port = args.port or int(args.addr.split(":")[-1])
    logger.info(f"Starting Bruno Passini Blackwell llama-server gRPC shim on port {port}...")
    serve_shim(BrunoLlamaServicer(), default_addr=args.addr, port=port)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
