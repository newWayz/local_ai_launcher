#!/usr/bin/env python3
"""
Claude Desktop Model Context Protocol (MCP) Bridge.
Zero-dependency stdio JSON-RPC 2.0 server connecting Claude Desktop to the local
Strata / LocalAI inference gateway (Port 8080).

Exposes:
- query_local_model: Delegate prompts to local Qwen 3.8 Flash MoE / LocalAI models.
- get_local_model_status: Probe health and loaded model topology on localhost.
"""
import sys
import os
import json
import urllib.request
import urllib.error
from typing import Dict, Any, Optional

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "local-ai-bridge"
SERVER_VERSION = "1.0.0"

DEFAULT_GATEWAY_URL = os.environ.get("LOCAL_LLM_URL") or os.environ.get("LOCAL_LLM_BASE_URL") or "http://127.0.0.1:8080/v1"
DEFAULT_MODEL = os.environ.get("LOCAL_LLM_MODEL") or "default"
API_KEY = os.environ.get("LOCAL_LLM_API_KEY") or "dummy-token"


def log_debug(msg: str):
    print(f"[{SERVER_NAME}] {msg}", file=sys.stderr, flush=True)


def handle_initialize(req_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "result": {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {
                "tools": {}
            },
            "serverInfo": {
                "name": SERVER_NAME,
                "version": SERVER_VERSION
            }
        }
    }


def handle_tools_list(req_id: Any) -> Dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "result": {
            "tools": [
                {
                    "name": "query_local_model",
                    "description": (
                        "Query the high-speed local AI model running on localhost (e.g. Strata Qwen 3.8 Flash MoE "
                        "or LocalAI Unified Gateway). Useful for private, offline, or intensive code tasks "
                        "without transmitting repository data to external cloud APIs."
                    ),
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "The prompt, instruction, or code to execute on the local model."
                            },
                            "system_prompt": {
                                "type": "string",
                                "description": "Optional system prompt steering the local model."
                            },
                            "temperature": {
                                "type": "number",
                                "description": "Sampling temperature (default: 0.7)."
                            },
                            "max_tokens": {
                                "type": "integer",
                                "description": "Maximum tokens to generate (default: 2048)."
                            }
                        },
                        "required": ["prompt"]
                    }
                },
                {
                    "name": "get_local_model_status",
                    "description": "Check the health, loaded model, and endpoint status of the local AI launcher.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {}
                    }
                }
            ]
        }
    }


def _get_candidate_endpoints(endpoint_type: str = "chat") -> list:
    """Resolves primary and fallback HTTP endpoints for local AI inference / health probes.

    Enables seamless failover between LiteLLM (Port 4000) and LocalAI / Strata (Port 8080)
    so MCP tool queries survive dynamic harness restarts and proxy lifecycle changes.
    """
    primary_base = DEFAULT_GATEWAY_URL.rstrip("/")
    bases = [primary_base]

    # If primary is port 4000 (LiteLLM), fallback to port 8080 (LocalAI / Strata)
    # If primary is port 8080, fallback to port 4000 (LiteLLM)
    if ":4000" in primary_base:
        bases.append("http://127.0.0.1:8080/v1")
    elif ":8080" in primary_base:
        bases.append("http://127.0.0.1:4000/v1")
    else:
        bases.append("http://127.0.0.1:8080/v1")
        bases.append("http://127.0.0.1:4000/v1")

    seen = set()
    unique_bases = []
    for b in bases:
        b_norm = b.rstrip("/")
        if b_norm not in seen:
            seen.add(b_norm)
            unique_bases.append(b_norm)

    endpoints = []
    for b in unique_bases:
        if endpoint_type == "chat":
            ep = f"{b}/chat/completions" if b.endswith("/v1") else f"{b}/v1/chat/completions"
        elif endpoint_type == "models":
            ep = f"{b}/models" if b.endswith("/v1") else f"{b}/v1/models"
        else:
            ep = b
        endpoints.append(ep)
    return endpoints


def _execute_local_query(prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.7, max_tokens: int = 2048) -> str:
    endpoints = _get_candidate_endpoints("chat")

    model_to_use = DEFAULT_MODEL
    if not model_to_use or model_to_use == "default":
        try:
            from local_ai_launcher.config import config
            if getattr(config, "default_model", None):
                model_to_use = config.default_model
        except Exception:
            pass

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model_to_use,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False
    }

    req_data = json.dumps(payload).encode("utf-8")
    last_error = None

    for idx, endpoint in enumerate(endpoints):
        req = urllib.request.Request(
            endpoint,
            data=req_data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {API_KEY}",
                "User-Agent": f"ClaudeDesktopBridge/{SERVER_VERSION}"
            }
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                choices = data.get("choices", [])
                if choices:
                    return choices[0].get("message", {}).get("content", "")
                return "Local model returned no choices."
        except urllib.error.HTTPError as e:
            err_body = ""
            try:
                err_body = e.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            if (e.code in (404, 502, 503, 504)) and idx < len(endpoints) - 1:
                log_debug(f"HTTP {e.code} at {endpoint}. Retrying fallback {endpoints[idx + 1]}...")
                last_error = f"HTTP {e.code}: {err_body or e.reason}"
                continue
            return f"HTTP error {e.code} from local AI engine at {endpoint}: {err_body or e.reason}. Ensure Local AI Launcher is running."
        except urllib.error.URLError as e:
            last_error = str(e)
            if idx < len(endpoints) - 1:
                log_debug(f"Connection failed at {endpoint} ({e}). Retrying fallback {endpoints[idx + 1]}...")
                continue
            return f"Error connecting to local AI engine (attempted {', '.join(endpoints)}): {e}. Ensure Local AI Launcher is running."
        except Exception as e:
            last_error = str(e)
            if idx < len(endpoints) - 1:
                continue
            return f"Error executing local query: {e}"

    return f"Error executing local query: {last_error}"


def _get_status() -> str:
    endpoints = _get_candidate_endpoints("models")
    errors = []
    max_ctx = os.environ.get("CLAUDE_CODE_MAX_CONTEXT_TOKENS")
    compact_win = os.environ.get("CLAUDE_CODE_AUTO_COMPACT_WINDOW")
    ctx_info = ""
    if max_ctx:
        ctx_formatted = f"{int(max_ctx):,}" if max_ctx.isdigit() else max_ctx
        ctx_info = f"\nContext Window: {ctx_formatted} tokens"
        if compact_win:
            comp_formatted = f"{int(compact_win):,}" if compact_win.isdigit() else compact_win
            ctx_info += f" (Auto-compact: {comp_formatted} tokens)"

    for endpoint in endpoints:
        req = urllib.request.Request(endpoint, headers={"Authorization": f"Bearer {API_KEY}"})
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = [m.get("id") for m in data.get("data", [])]
                base_part = endpoint.replace("/models", "").replace("/v1/models", "")
                return f"Local AI Launcher is healthy and online.\nEndpoint: {base_part}\nAvailable Models: {', '.join(models) if models else 'default'}{ctx_info}"
        except Exception as e:
            errors.append(f"{endpoint}: {e}")
    return f"Local AI Launcher status probe failed (tried {', '.join(endpoints)}): {'; '.join(errors)}{ctx_info}"


def handle_tools_call(req_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    name = params.get("name")
    args = params.get("arguments") or {}

    if name == "query_local_model":
        prompt = str(args.get("prompt") or "")
        system_prompt = args.get("system_prompt")
        if system_prompt is not None:
            system_prompt = str(system_prompt)
        raw_temp = args.get("temperature")
        try:
            temp = float(raw_temp) if raw_temp is not None else 0.7
        except (ValueError, TypeError):
            temp = 0.7

        raw_max_tok = args.get("max_tokens")
        try:
            max_tok = int(raw_max_tok) if raw_max_tok is not None else 2048
        except (ValueError, TypeError):
            max_tok = 2048

        result_text = _execute_local_query(prompt, system_prompt, temp, max_tok)
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": result_text
                    }
                ]
            }
        }
    elif name == "get_local_model_status":
        status_text = _get_status()
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": status_text
                    }
                ]
            }
        }
    else:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {
                "code": -32601,
                "message": f"Tool '{name}' not found"
            }
        }


def main():
    if hasattr(sys.stdin, "reconfigure"):
        try:
            sys.stdin.reconfigure(encoding="utf-8", errors="replace")
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue

            try:
                msg = json.loads(line)
            except Exception:
                continue

            if not isinstance(msg, dict):
                continue

            method = msg.get("method")
            req_id = msg.get("id")
            params = msg.get("params") or {}

            if method == "initialize":
                resp = handle_initialize(req_id, params)
            elif method == "notifications/initialized":
                continue
            elif method == "tools/list":
                resp = handle_tools_list(req_id)
            elif method == "tools/call":
                resp = handle_tools_call(req_id, params)
            elif method == "ping":
                resp = {"jsonrpc": "2.0", "id": req_id, "result": {}}
            else:
                if req_id is not None:
                    resp = {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {
                            "code": -32601,
                            "message": f"Method '{method}' not found"
                        }
                    }
                else:
                    continue

            try:
                out_str = json.dumps(resp, ensure_ascii=False) + "\n"
                sys.stdout.write(out_str)
            except (UnicodeEncodeError, UnicodeError):
                out_str = json.dumps(resp, ensure_ascii=True) + "\n"
                sys.stdout.write(out_str)
            sys.stdout.flush()

        except (KeyboardInterrupt, SystemExit):
            break
        except Exception as e:
            log_debug(f"Unhandled loop error: {e}")


if __name__ == "__main__":
    main()
