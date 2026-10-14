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
import configparser
from pathlib import Path
from typing import Optional

import httpx
import grpc

from local_ai_launcher.config import AI_WORKSPACE_DIR, FREETOKENS_MODELS_PRESET_FILE
from local_ai_launcher.localai.proto import backend_pb2, backend_pb2_grpc
from local_ai_launcher.shims.base_shim import BaseBackendServicer, serve_shim
from local_ai_launcher.backends.freetoken import check_freetoken, _patch_freetoken_windows_hotfixes

logger = logging.getLogger("freetoken_grpc_shim")


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


class FreeTokenServicer(BaseBackendServicer):
    """
    gRPC shim wrapping FreeToken (NVFP4 inference server).
    Handles PyTorch CUDA 13, Windows hotfixes, checkpoint conversion, and HTTP-to-gRPC proxying.
    """
    def __init__(self):
        super().__init__("freetoken-nvfp4")
        self.process: Optional[subprocess.Popen] = None
        self.http_port: Optional[int] = None
        self.http_base: Optional[str] = None
        self.freetoken_dir = AI_WORKSPACE_DIR / "freetoken"

    def _wait_for_health(self, timeout: int = 120) -> bool:
        if not self.http_base:
            return False
        try:
            timeout = max(1, int(timeout))
        except (ValueError, TypeError):
            timeout = 120
        health_url = f"{self.http_base}/health"
        start = time.time()
        while time.time() - start < timeout:
            if self.process and self.process.poll() is not None:
                logger.error(f"[freetoken-nvfp4] Subprocess exited early with code {self.process.returncode}")
                return False
            try:
                req = urllib.request.Request(health_url, headers={"User-Agent": "ShimHealthCheck"})
                with urllib.request.urlopen(req, timeout=2.0) as resp:
                    if resp.status in (200, 204):
                        return True
            except urllib.error.HTTPError as e:
                # 503 means model is still loading weights into VRAM
                if e.code == 503:
                    time.sleep(2.0)
                    continue
                if e.code in (200, 204):
                    return True
            except Exception:
                pass
            time.sleep(1.0)
        return False

    def LoadModel(self, request, context):
        self.Free(None, None)

        model_name = request.Model or request.ModelFile
        if not model_name:
            return backend_pb2.Result(success=False, message="No model specified in ModelOptions")

        clean_dm = model_name
        if clean_dm.lower().startswith("freetoken/") or clean_dm.lower().startswith("freetoken\\"):
            clean_dm = clean_dm[10:].replace("\\", "/")

        installed = check_freetoken()
        if not installed:
            return backend_pb2.Result(success=False, message="FreeToken installation check failed")

        _patch_freetoken_windows_hotfixes(self.freetoken_dir)

        freetoken_exe = (self.freetoken_dir / "venv" / "Scripts" / "ft.exe") if sys.platform == 'win32' else (self.freetoken_dir / "venv" / "bin" / "ft")
        self.http_port = find_free_port()
        self.http_base = f"http://127.0.0.1:{self.http_port}"

        # Resolve ftw model folder
        safe_repo = clean_dm.replace('/', '_')
        ftw_dir = AI_WORKSPACE_DIR / "models" / "freetoken" / safe_repo
        model_target = ftw_dir if ftw_dir.exists() else (AI_WORKSPACE_DIR / "models" / "freetoken" / clean_dm)

        cmd = [
            str(freetoken_exe),
            "serve",
            "--model", str(model_target),
            "--port", str(self.http_port),
            "--host", "127.0.0.1"
        ]

        # Apply flags from freetokens_models.ini if configured
        if FREETOKENS_MODELS_PRESET_FILE.exists():
            conf = configparser.ConfigParser(strict=False)
            conf.read(FREETOKENS_MODELS_PRESET_FILE, encoding='utf-8')
            for sec in conf.sections():
                sec_alias = conf.get(sec, 'alias', fallback='')
                if sec == model_name or sec_alias == model_name or sec == clean_dm:
                    for k, v in conf.items(sec):
                        if k not in ('alias', 'model', 'mmproj', 'chat-template-file'):
                            if str(v).lower() in ("true", "yes", "on"):
                                cmd.append(f"--{k}")
                            elif str(v).lower() not in ("false", "no", "off") and v:
                                cmd.extend([f"--{k}", str(v)])
                    break

        logger.info(f"[freetoken-nvfp4] Launching FreeToken for '{clean_dm}' on port {self.http_port}...")
        try:
            env = os.environ.copy()
            hf_home = str(AI_WORKSPACE_DIR / "models" / "huggingface")
            env["HOME"] = str(AI_WORKSPACE_DIR)
            env["HF_HOME"] = hf_home
            env["HF_HUB_CACHE"] = hf_home
            creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            self.process = subprocess.Popen(
                cmd,
                cwd=str(self.freetoken_dir),
                env=env,
                creationflags=creationflags
            )
            try:
                from local_ai_launcher.process import assign_process_to_job
                assign_process_to_job(self.process)
            except Exception:
                pass
            ready = self._wait_for_health(timeout=180)
            if not ready:
                self.Free(None, None)
                return backend_pb2.Result(success=False, message="Timed out waiting for FreeToken to load weights")

            self.is_ready = True
            self.current_model = model_name
            logger.info(f"[freetoken-nvfp4] Model '{clean_dm}' loaded successfully into VRAM!")
            return backend_pb2.Result(success=True, message=f"FreeToken ready on {self.http_base}")
        except Exception as e:
            logger.error(f"[freetoken-nvfp4] Failed to launch FreeToken: {e}")
            self.Free(None, None)
            return backend_pb2.Result(success=False, message=str(e))

    def Predict(self, request, context):
        if not self.is_ready or not self.http_base:
            if context:
                context.set_code(grpc.StatusCode.UNAVAILABLE)
                context.set_details("FreeToken backend not initialized")
            return backend_pb2.Reply(message=b"Backend not initialized")

        payload = self._build_request_payload(request, stream=False)
        try:
            with httpx.Client(timeout=120.0) as client:
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
            logger.error(f"[freetoken-nvfp4] Predict error: {e}")
            if context:
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(str(e))
            return backend_pb2.Reply(message=str(e).encode("utf-8"))

    def PredictStream(self, request, context):
        if not self.is_ready or not self.http_base:
            if context:
                context.set_code(grpc.StatusCode.UNAVAILABLE)
                context.set_details("FreeToken backend not initialized")
            yield backend_pb2.Reply(message=b"Backend not initialized")
            return

        payload = self._build_request_payload(request, stream=True)
        url = f"{self.http_base}/v1/chat/completions"
        try:
            with httpx.Client(timeout=120.0) as client:
                with client.stream("POST", url, json=payload) as response:
                    if response.status_code != 200:
                        err_text = response.read().decode("utf-8", errors="replace")
                        logger.error(f"[freetoken-nvfp4] Upstream HTTP error {response.status_code}: {err_text}")
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
                                elif "usage" in chunk:
                                    yield backend_pb2.Reply(message=raw_data.encode("utf-8"))
                            except json.JSONDecodeError:
                                pass
        except Exception as e:
            logger.error(f"[freetoken-nvfp4] Stream error: {e}")
            if context:
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(str(e))
            return

    def _build_request_payload(self, request, stream: bool = False) -> dict:
        payload = {
            "model": self.current_model or "default",
            "stream": stream,
        }
        if request.Temperature > 0:
            payload["temperature"] = max(0.0, min(2.0, float(request.Temperature)))
        elif request.Metadata and request.Metadata.get("explicit_temperature") in ("0", "0.0", "0.00"):
            payload["temperature"] = 0.0
        if request.TopP > 0:
            payload["top_p"] = max(0.0, min(1.0, float(request.TopP)))
        if request.TopK > 0:
            payload["top_k"] = max(0, int(request.TopK))
        if request.MinP > 0:
            payload["min_p"] = max(0.0, min(1.0, float(request.MinP)))
        if request.Penalty > 0:
            payload["repetition_penalty"] = max(0.0, float(request.Penalty))
        if request.PresencePenalty != 0.0:
            payload["presence_penalty"] = max(-2.0, min(2.0, float(request.PresencePenalty)))
        if request.FrequencyPenalty != 0.0:
            payload["frequency_penalty"] = max(-2.0, min(2.0, float(request.FrequencyPenalty)))
        if request.Seed > 0:
            payload["seed"] = max(0, int(request.Seed))

        if request.Metadata:
            if "response_format" in request.Metadata:
                try:
                    payload["response_format"] = json.loads(request.Metadata["response_format"])
                except Exception:
                    payload["response_format"] = request.Metadata["response_format"]
            if "stream_options" in request.Metadata:
                try:
                    payload["stream_options"] = json.loads(request.Metadata["stream_options"])
                except Exception:
                    payload["stream_options"] = request.Metadata["stream_options"]
            if "extra_body_params" in request.Metadata:
                try:
                    extra = json.loads(request.Metadata["extra_body_params"])
                    if isinstance(extra, dict):
                        for k, v in extra.items():
                            if k not in payload and k not in ("messages", "prompt", "model", "stream"):
                                payload[k] = v
                except Exception:
                    pass

        if request.Grammar:
            payload["grammar"] = request.Grammar
        if request.TypicalP > 0:
            payload["typical_p"] = max(0.0, min(1.0, float(request.TypicalP)))
        if request.LogitBias:
            try:
                payload["logit_bias"] = json.loads(request.LogitBias)
            except Exception:
                pass
        if request.Logprobs > 0:
            payload["logprobs"] = True
            if request.TopLogprobs > 0:
                payload["top_logprobs"] = max(0, int(request.TopLogprobs))

        if request.Tokens > 0:
            payload["max_tokens"] = max(1, int(request.Tokens))
        if request.StopPrompts:
            payload["stop"] = list(request.StopPrompts)

        if request.Messages:
            msgs = []
            for m in request.Messages:
                msg_obj = {"role": m.role}
                if m.content:
                    msg_obj["content"] = m.content
                elif m.tool_calls:
                    msg_obj["content"] = None
                else:
                    msg_obj["content"] = ""
                if m.name:
                    msg_obj["name"] = m.name
                if m.tool_call_id:
                    msg_obj["tool_call_id"] = m.tool_call_id
                if m.tool_calls:
                    try:
                        msg_obj["tool_calls"] = json.loads(m.tool_calls)
                    except Exception:
                        msg_obj["tool_calls"] = m.tool_calls
                msgs.append(msg_obj)

            if request.Images:
                for msg in reversed(msgs):
                    if msg.get("role") == "user":
                        c_val = msg.get("content") or ""
                        parts = []
                        if isinstance(c_val, list):
                            parts = list(c_val)
                        elif c_val:
                            parts = [{"type": "text", "text": c_val}]
                        for img_url in request.Images:
                            parts.append({"type": "image_url", "image_url": {"url": img_url}})
                        msg["content"] = parts
                        break

            payload["messages"] = msgs
        else:
            payload["messages"] = [{"role": "user", "content": request.Prompt or ""}]

        if request.Tools:
            try:
                payload["tools"] = json.loads(request.Tools)
            except Exception:
                payload["tools"] = request.Tools
        if request.ToolChoice:
            try:
                payload["tool_choice"] = json.loads(request.ToolChoice)
            except Exception:
                payload["tool_choice"] = request.ToolChoice

        return payload

    def Free(self, request, context):
        if self.process:
            logger.info("[freetoken-nvfp4] Freeing VRAM and terminating FreeToken process...")
            try:
                from local_ai_launcher.process import kill_process_tree
                kill_process_tree(self.process)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
            self.process = None

        self.is_ready = False
        self.current_model = None
        self.http_port = None
        self.http_base = None
        return backend_pb2.Result(success=True, message="FreeToken backend freed")


_active_servicer = None


def _cleanup_freetoken_on_exit():
    if _active_servicer and _active_servicer.process and _active_servicer.process.poll() is None:
        try:
            from local_ai_launcher.process import kill_process_tree
            kill_process_tree(_active_servicer.process)
        except Exception:
            pass


atexit.register(_cleanup_freetoken_on_exit)

if sys.platform == 'win32':
    try:
        import ctypes
        from ctypes import wintypes
        def _shim_ctrl_handler(ctrl_type):
            _cleanup_freetoken_on_exit()
            return False
        _HandlerRoutine = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)
        _shim_ctrl_handler._routine = _HandlerRoutine(_shim_ctrl_handler)
        ctypes.windll.kernel32.SetConsoleCtrlHandler(_shim_ctrl_handler._routine, True)
    except Exception:
        pass


if __name__ == "__main__":
    _active_servicer = FreeTokenServicer()
    serve_shim(_active_servicer, "127.0.0.1:9942")
