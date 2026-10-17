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
from typing import Optional

import httpx
import grpc

from local_ai_launcher.config import AI_WORKSPACE_DIR, STRATA_DIR, config
from local_ai_launcher.localai.proto import backend_pb2, backend_pb2_grpc
from local_ai_launcher.shims.base_shim import BaseBackendServicer, serve_shim, build_common_llama_payload
from local_ai_launcher.backends.strata import (
    check_strata,
    resolve_strata_model_files,
    ensure_strata_pack,
    build_strata_config,
    load_strata_settings
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger("strata_grpc_shim")


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


class StrataServicer(BaseBackendServicer):
    """
    gRPC shim wrapping Strata local inference engine (C++/CUDA server for Qwen3.8-Flash-Next).
    Proxies LocalAI gRPC Predict/PredictStream requests to Strata's native OpenAI-compatible API.
    """
    def __init__(self):
        super().__init__("strata")
        self.process: Optional[subprocess.Popen] = None
        self.log_file = None
        self.http_port: Optional[int] = None
        self.http_base: Optional[str] = None
        self.strata_dir = STRATA_DIR

    def _wait_for_health(self, timeout: int = 180) -> bool:
        if not self.http_base:
            return False
        try:
            timeout = max(10, int(timeout))
        except (ValueError, TypeError):
            timeout = 180
        start = time.time()
        while time.time() - start < timeout:
            if self.process and self.process.poll() is not None:
                err_snippet = ""
                try:
                    log_path = self.strata_dir / "strata_engine.log"
                    if log_path.exists():
                        lines = log_path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
                        err_snippet = "\n".join(lines[-15:])
                except Exception:
                    pass
                logger.error(f"[strata] Subprocess exited early with code {self.process.returncode}" + (f":\n{err_snippet}" if err_snippet else ""))
                return False
            for check_path in ("/v1/models", "/health", "/v1/props"):
                try:
                    req = urllib.request.Request(f"{self.http_base}{check_path}", headers={"User-Agent": "StrataHealthCheck"})
                    with urllib.request.urlopen(req, timeout=2.0) as resp:
                        if resp.status in (200, 204, 302):
                            return True
                except Exception:
                    pass
            time.sleep(1.5)
        return False

    def LoadModel(self, request, context):
        self.Free(None, None)

        model_name = request.Model or ""
        model_file = request.ModelFile or ""
        engine_args = {}
        if getattr(request, "EngineArgs", None):
            try:
                engine_args = json.loads(request.EngineArgs)
            except Exception:
                pass

        target_path = model_file or engine_args.get("model") or model_name
        target_mtp = engine_args.get("mtp")
        if not target_path:
            return backend_pb2.Result(success=False, message="No model specified in ModelOptions")

        if not check_strata():
            return backend_pb2.Result(success=False, message="Strata engine installation check failed")

        resolved = resolve_strata_model_files(target_path, mtp_path_str=target_mtp)
        first_shard = resolved["first_shard"]
        ple_shard = resolved["ple_shard"]
        mtp_dir = resolved["mtp_dir"]
        pack_dir = resolved["pack_dir"]

        if not first_shard or not first_shard.exists():
            return backend_pb2.Result(
                success=False,
                message=f"Model shard file not found: {first_shard}"
            )

        # Prepare pack if not built
        ok_pack = ensure_strata_pack(first_shard, pack_dir)
        if not ok_pack:
            return backend_pb2.Result(
                success=False,
                message=f"Failed to build Strata expert pack at {pack_dir}"
            )

        self.http_port = find_free_port()
        self.http_base = f"http://127.0.0.1:{self.http_port}"

        # Load user strata settings from local_ai_launcher/strata_settings.json
        strata_user_cfg = load_strata_settings()

        # Resolve mmap-experts memory policy (default True = dynamic OS page cache; False = pure resident RAM)
        mmap_experts_val = engine_args.get("mmap_experts")
        if mmap_experts_val is None:
            mmap_experts_val = strata_user_cfg.get("mmap_experts", getattr(config, "strata_mmap_experts", True))
        is_mmap = bool(mmap_experts_val)

        # Resolve KV cache quantization mode (default "int8", options: "int8", "q4_0", "k8v4", "fp16")
        kv_mode_val = engine_args.get("kv_mode") or engine_args.get("kv")
        if not kv_mode_val:
            kv_mode_val = strata_user_cfg.get("kv_mode", getattr(config, "strata_kv_mode", "int8"))

        # Resolve resident expert budget in GiB (default 56.0 GiB)
        resident_budget_val = engine_args.get("resident_budget_gib") or engine_args.get("resident_budget")
        if resident_budget_val is None:
            resident_budget_val = strata_user_cfg.get("resident_budget_gib", getattr(config, "strata_resident_budget_gib", 52.0))
        try:
            resident_budget_f = float(resident_budget_val)
        except (ValueError, TypeError):
            resident_budget_f = 0.0

        # Execute system RAM & VRAM clawback across tiers: "off", "light", "max"
        clawback_mode = engine_args.get("clawback") or engine_args.get("brutal_ram_clawback") or strata_user_cfg.get("brutal_ram_clawback", getattr(config, "strata_brutal_ram_clawback", "light"))
        try:
            from local_ai_launcher.memory_clawback import clawback_system_memory
            clawback_system_memory(mode=clawback_mode)
        except Exception as e:
            logger.debug(f"[strata] Memory clawback notification: {e}")

        # Resolve MTP spec
        spec_val = engine_args.get("spec") or strata_user_cfg.get("spec", 4)
        try:
            spec_i = int(spec_val)
        except (ValueError, TypeError):
            spec_i = 4

        # Resolve custom chat template and whether to use it
        target_chat_template = (
            engine_args.get("chat_template_file")
            or engine_args.get("chat-template-file")
            or resolved.get("chat_template_file")
        )
        use_custom_tmpl = engine_args.get("use_custom_chat_template")
        if use_custom_tmpl is None:
            use_custom_tmpl = getattr(config, "strata_use_custom_chat_template", True)
        use_custom_tmpl = bool(use_custom_tmpl)

        kv_resident_val = engine_args.get("kv_resident") or strata_user_cfg.get("kv_resident")
        vram_reserve_val = engine_args.get("vram_reserve_mib") or strata_user_cfg.get("vram_reserve_mib")
        sys_ram_kv_val = (
            engine_args.get("system_ram_kv_cache")
            or engine_args.get("ram_only_kv_cache")
            or strata_user_cfg.get("system_ram_kv_cache")
            or getattr(config, "strata_system_ram_kv_cache", False)
        )

        reasoning_effort_val = (
            engine_args.get("reasoning_effort")
            or (engine_args.get("reasoning", {}).get("effort") if isinstance(engine_args.get("reasoning"), dict) else None)
            or strata_user_cfg.get("reasoning_effort")
        )

        ctx_size = getattr(request, "ContextSize", 262144) or strata_user_cfg.get("max_context", 262144) or 262144
        cfg_path = build_strata_config(
            model_name=model_name,
            first_shard=first_shard,
            ple_shard=ple_shard,
            pack_dir=pack_dir,
            mtp_dir=mtp_dir,
            port=self.http_port,
            max_context=ctx_size,
            kv_mode=str(kv_mode_val),
            spec=spec_i,
            mmap_experts=is_mmap,
            resident_budget_gib=resident_budget_f,
            chat_template_file=target_chat_template,
            use_custom_chat_template=use_custom_tmpl,
            kv_resident=kv_resident_val,
            vram_reserve_mib=vram_reserve_val,
            system_ram_kv_cache=bool(sys_ram_kv_val),
            reasoning_effort=reasoning_effort_val
        )

        try:
            cmd = [
                sys.executable,
                str(self.strata_dir / "serve" / "server.py"),
                "--engine", "strata",
                "--config", str(cfg_path),
                "--port", str(self.http_port)
            ]
            logger.info(f"[strata] Launching Strata server on port {self.http_port} (PID will be monitored)...")

            env = os.environ.copy()
            # Under Windows WDDM, cap host pinned memory to 8 GiB so WDDM does not reject subsequent cudaMalloc calls
            env.setdefault("STRATA_ARENA_PIN_GIB", "8")
            creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            log_path = self.strata_dir / "strata_engine.log"
            self.log_file = open(log_path, "w", encoding="utf-8", errors="replace")

            self.process = subprocess.Popen(
                cmd,
                cwd=str(self.strata_dir),
                env=env,
                stdout=self.log_file,
                stderr=subprocess.STDOUT,
                creationflags=creationflags
            )

            try:
                from local_ai_launcher.process import assign_process_to_job
                assign_process_to_job(self.process)
            except Exception:
                pass

            # Strata can take 60-120 seconds on initial RAM pinning
            ready = self._wait_for_health(timeout=180)
            if not ready:
                self.Free(None, None)
                return backend_pb2.Result(success=False, message="Timed out waiting for Strata engine to become healthy")

            self.is_ready = True
            self.current_model = model_name
            logger.info(f"[strata] Model '{model_name}' loaded successfully on {self.http_base}!")
            return backend_pb2.Result(success=True, message=f"Strata ready on {self.http_base}")
        except Exception as e:
            logger.error(f"[strata] Failed to launch Strata: {e}")
            self.Free(None, None)
            return backend_pb2.Result(success=False, message=str(e))

    def _build_request_payload(self, request, stream: bool = False) -> dict:
        return build_common_llama_payload(request, current_model=self.current_model or "strata", stream=stream, chat_only=True)

    def Predict(self, request, context):
        if not self.is_ready or not self.http_base:
            if context:
                context.set_code(grpc.StatusCode.UNAVAILABLE)
                context.set_details("Strata backend not initialized")
            return backend_pb2.Reply(message=b"Backend not initialized")

        payload = self._build_request_payload(request, stream=False)
        try:
            with httpx.Client(timeout=180.0) as client:
                url = f"{self.http_base}/v1/chat/completions"
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
            logger.error(f"[strata] Predict error: {e}")
            if context:
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(str(e))
            return backend_pb2.Reply(message=str(e).encode("utf-8"))

    def PredictStream(self, request, context):
        if not self.is_ready or not self.http_base:
            if context:
                context.set_code(grpc.StatusCode.UNAVAILABLE)
                context.set_details("Strata backend not initialized")
            yield backend_pb2.Reply(message=b"Backend not initialized")
            return

        payload = self._build_request_payload(request, stream=True)
        url = f"{self.http_base}/v1/chat/completions"
        try:
            with httpx.Client(timeout=180.0) as client:
                with client.stream("POST", url, json=payload) as response:
                    if response.status_code != 200:
                        err_text = response.read().decode("utf-8", errors="replace")
                        logger.error(f"[strata] Upstream HTTP error {response.status_code}: {err_text}")
                        if context:
                            context.set_code(grpc.StatusCode.INTERNAL)
                            context.set_details(f"Upstream HTTP error {response.status_code}: {err_text}")
                        return
                    for line in response.iter_lines():
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
                            except json.JSONDecodeError:
                                yield backend_pb2.Reply(message=raw_data.encode("utf-8"))
        except Exception as e:
            logger.error(f"[strata] PredictStream error: {e}")
            if context:
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(str(e))
            yield backend_pb2.Reply(message=str(e).encode("utf-8"))

    def Free(self, request, context):
        """Releases GPU VRAM and terminates the Strata process."""
        self.is_ready = False
        self.current_model = None
        self.http_base = None

        if self.process:
            logger.info("[strata] Terminating Strata process to free GPU VRAM...")
            try:
                from local_ai_launcher.process import kill_process_tree
                kill_process_tree(self.process)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
            self.process = None

        # Clean up any orphaned strata.exe instances if port was bound
        if self.http_port:
            try:
                from local_ai_launcher.process import force_kill_port
                force_kill_port(self.http_port)
            except Exception:
                pass

        if self.log_file:
            try:
                self.log_file.close()
            except Exception:
                pass
            self.log_file = None

        # Short driver stabilization delay to ensure CUDA page tables are completely unmapped
        time.sleep(0.5)

        return backend_pb2.Result(success=True, message="Strata engine freed from VRAM")


if __name__ == "__main__":
    servicer = StrataServicer()
    default_port = getattr(config, "strata_shim_port", 9943)
    serve_shim(servicer, default_addr=f"127.0.0.1:{default_port}")
