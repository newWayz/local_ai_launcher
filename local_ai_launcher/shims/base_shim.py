import sys
import os
import json
import argparse
import logging
import asyncio
from concurrent import futures
from pathlib import Path
from typing import Optional, Generator, AsyncGenerator

from local_ai_launcher.localai.proto import backend_pb2, backend_pb2_grpc
import grpc

logger = logging.getLogger("grpc_shim")

class BaseBackendServicer(backend_pb2_grpc.BackendServicer):
    """
    Base implementation of LocalAI BackendServicer providing default handlers
    and standard lifecycle methods for external inference backends.
    """
    def __init__(self, name: str):
        self.name = name
        self.is_ready = False
        self.is_busy = False
        self.current_model = None

    def Health(self, request, context):
        return backend_pb2.Reply(message=b"OK")

    def Status(self, request, context):
        if not self.is_ready:
            state = backend_pb2.StatusResponse.UNINITIALIZED
        elif self.is_busy:
            state = backend_pb2.StatusResponse.BUSY
        else:
            state = backend_pb2.StatusResponse.READY
        return backend_pb2.StatusResponse(state=state)

    def LoadModel(self, request, context):
        """Override in subclasses to initialize backend process/weights."""
        self.current_model = request.Model
        self.is_ready = True
        return backend_pb2.Result(success=True, message=f"Loaded {request.Model}")

    def Predict(self, request, context):
        """Override in subclasses for single-shot completion."""
        return backend_pb2.Reply(message=b"Not implemented")

    def PredictStream(self, request, context):
        """Override in subclasses for streaming completions."""
        yield backend_pb2.Reply(message=b"Not implemented")

    def TokenizeString(self, request, context):
        """Tokenize string and return token length."""
        text = request.Prompt or ""
        tokens = [ord(c) for c in text[:100]] # basic fallback
        return backend_pb2.TokenizationResponse(length=len(tokens), tokens=tokens)

    def Free(self, request, context):
        """Override in subclasses to release GPU VRAM and terminate subprocesses."""
        self.is_ready = False
        self.current_model = None
        return backend_pb2.Result(success=True, message="Freed")


def build_common_llama_payload(request, current_model: Optional[str] = None, stream: bool = False, chat_only: bool = False) -> dict:
    """Build unified HTTP request payload for llama-server compatible shims."""
    payload = {
        "stream": stream,
    }
    if current_model:
        payload["model"] = current_model

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
        payload["repeat_penalty"] = max(0.0, float(request.Penalty))
    if request.PresencePenalty != 0.0:
        payload["presence_penalty"] = max(-2.0, min(2.0, float(request.PresencePenalty)))
    if request.FrequencyPenalty != 0.0:
        payload["frequency_penalty"] = max(-2.0, min(2.0, float(request.FrequencyPenalty)))
    if request.Seed > 0:
        payload["seed"] = max(0, int(request.Seed))

    # Forward reasoning parameters, response_format, stream_options, and extra client parameters from Metadata
    if request.Metadata:
        if "reasoning_budget" in request.Metadata:
            try:
                payload["reasoning_budget"] = max(0, int(request.Metadata["reasoning_budget"]))
            except Exception:
                pass
        if "reasoning_format" in request.Metadata:
            payload["reasoning_format"] = request.Metadata["reasoning_format"]
        if "reasoning_effort" in request.Metadata:
            payload["reasoning_effort"] = request.Metadata["reasoning_effort"]
        if "chat_template_kwargs" in request.Metadata:
            try:
                payload["chat_template_kwargs"] = json.loads(request.Metadata["chat_template_kwargs"])
            except Exception:
                pass
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

        # Re-inject multimodal images into user message if present
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
        prompt_text = request.Prompt or ""
        try:
            parsed = json.loads(prompt_text)
            if isinstance(parsed, list):
                payload["messages"] = parsed
            elif isinstance(parsed, dict) and "messages" in parsed:
                payload["messages"] = parsed["messages"]
            elif chat_only:
                payload["messages"] = [{"role": "user", "content": prompt_text}]
            else:
                payload["prompt"] = prompt_text
        except Exception:
            if chat_only:
                payload["messages"] = [{"role": "user", "content": prompt_text}]
            else:
                payload["prompt"] = prompt_text

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


def serve_shim(
    servicer: BaseBackendServicer,
    default_addr: str = "127.0.0.1:50051",
    port: Optional[int] = None,
):
    """Standard gRPC server entrypoint accepting --addr CLI parameter (LocalAI contract)."""
    parser = argparse.ArgumentParser(description=f"LocalAI gRPC Shim for {servicer.name}")
    parser.add_argument("--addr", default=default_addr, help="The address (host:port) to bind the gRPC server to.")
    parser.add_argument("--port", type=int, help="Alternative port flag.")
    args, _ = parser.parse_known_args()

    addr = args.addr
    if port is not None:
        host = str(args.addr).rsplit(":", 1)[0] if ":" in str(args.addr) else "127.0.0.1"
        addr = f"{host}:{port}"
    elif args.port is not None:
        try:
            port_num = int(args.port)
            if 1 <= port_num <= 65535:
                addr = f"127.0.0.1:{port_num}"
            else:
                logger.warning(f"[{servicer.name}] Invalid port {args.port}, falling back to {default_addr}")
                addr = default_addr
        except (ValueError, TypeError):
            logger.warning(f"[{servicer.name}] Unparseable port {args.port}, falling back to {default_addr}")
            addr = default_addr
    else:
        try:
            if ":" in str(addr):
                _, port_str = str(addr).rsplit(":", 1)
                port_val = int(port_str)
                if not (1 <= port_val <= 65535):
                    logger.warning(f"[{servicer.name}] Port {port_val} in addr out of range, using {default_addr}")
                    addr = default_addr
        except Exception:
            addr = default_addr

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4), options=[
        ('grpc.max_message_length', 50 * 1024 * 1024),
        ('grpc.max_send_message_length', 50 * 1024 * 1024),
        ('grpc.max_receive_message_length', 50 * 1024 * 1024),
    ])
    backend_pb2_grpc.add_BackendServicer_to_server(servicer, server)
    server.add_insecure_port(addr)
    server.start()
    logger.info(f"[{servicer.name}] gRPC shim started and listening on {addr}")
    print(f"[{servicer.name}] gRPC shim started and listening on {addr}", file=sys.stderr, flush=True)

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info(f"[{servicer.name}] Shutting down gRPC shim...")
        servicer.Free(None, None)
        server.stop(grace=1)
