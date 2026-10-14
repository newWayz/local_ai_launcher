import os
import json
import logging
from typing import Optional, Dict, Any, AsyncGenerator

from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
import uvicorn

from starlette.background import BackgroundTask

app = FastAPI(title="Antigravity MCP Server")

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HOP_BY_HOP_HEADERS = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "host", "content-length"
}

logger = logging.getLogger("mcp_server")
logging.basicConfig(level=logging.INFO)

try:
    from local_ai_launcher.config import config
    _default_llm_port = config.localai_port
except Exception:
    _default_llm_port = 8080

OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", f"http://127.0.0.1:{_default_llm_port}/v1")
MODEL = os.environ.get("OPENAI_MODEL", "default")

@app.get("/health")
async def health():
    return {"status": "ok"}

async def forward_to_llm(prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.7, max_tokens: int = 2048, stream: bool = True) -> AsyncGenerator[str, None]:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream
    }
    
    async with httpx.AsyncClient() as client:
        if stream:
            async with client.stream("POST", f"{OPENAI_BASE_URL}/chat/completions", json=payload, timeout=300.0) as response:
                if response.status_code != 200:
                    yield f"Error: {response.status_code} - {await response.aread()}"
                    return
                async for chunk in response.aiter_lines():
                    if chunk.startswith("data: "):
                        data = chunk[6:]
                        if data == "[DONE]":
                            break
                        try:
                            parsed = json.loads(data)
                            if "choices" in parsed and len(parsed["choices"]) > 0:
                                delta = parsed["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    yield content
                        except json.JSONDecodeError:
                            pass
        else:
            response = await client.post(f"{OPENAI_BASE_URL}/chat/completions", json=payload, timeout=300.0)
            if response.status_code != 200:
                yield f"Error: {response.status_code} - {response.text}"
                return
            data = response.json()
            if "choices" in data and len(data["choices"]) > 0:
                yield data["choices"][0].get("message", {}).get("content", "")

@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_v1(path: str, request: Request):
    if path.startswith("audio/"):
        target_url = f"http://127.0.0.1:8000/v1/{path}"
    else:
        target_url = f"{OPENAI_BASE_URL}/{path}"

    client = httpx.AsyncClient()
    body = await request.body()
    try:
        req = client.build_request(
            request.method,
            target_url,
            headers={k: v for k, v in request.headers.items() if k.lower() not in HOP_BY_HOP_HEADERS},
            content=body,
            timeout=300.0
        )
        response = await client.send(req, stream=True)
    except Exception as e:
        await client.aclose()
        raise HTTPException(status_code=502, detail=f"Proxy error: {e}")

    async def stream_generator():
        try:
            async for chunk in response.aiter_raw():
                yield chunk
        finally:
            await response.aclose()
            await client.aclose()

    return StreamingResponse(
        stream_generator(),
        status_code=response.status_code,
        headers={k: v for k, v in response.headers.items() if k.lower() not in HOP_BY_HOP_HEADERS and k.lower() != "content-encoding"},
        background=BackgroundTask(client.aclose)
    )

@app.post("/mcp")
async def mcp_endpoint(request: Request):
    mcp_method = request.headers.get("Mcp-Method")
    mcp_version = request.headers.get("MCP-Protocol-Version")
    
    if mcp_version != "2026-07-28":
        logger.warning(f"Unsupported MCP protocol version: {mcp_version}")
        
    try:
        body = await request.json()
    except Exception:
        body = {}
        
    if mcp_method == "tools/list" or body.get("method") == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": body.get("id"),
            "result": {
                "tools": [
                    {
                        "name": "query_local_model",
                        "description": "Send a prompt to the local LLM",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "prompt": {"type": "string"},
                                "system_prompt": {"type": "string"},
                                "temperature": {"type": "number"},
                                "max_tokens": {"type": "integer"},
                                "stream": {"type": "boolean"}
                            },
                            "required": ["prompt"]
                        }
                    }
                ]
            }
        }
    
    if mcp_method == "tools/call" or body.get("method") == "tools/call":
        params = body.get("params", {})
        name = params.get("name") or request.headers.get("Mcp-Name")
        args = params.get("arguments", {})
        
        if name == "query_local_model":
            prompt = args.get("prompt")
            stream = args.get("stream", True)
            
            if not prompt:
                raise HTTPException(status_code=400, detail="prompt is required")
                
            if stream:
                async def stream_generator():
                    async for chunk in forward_to_llm(
                        prompt, 
                        args.get("system_prompt"), 
                        args.get("temperature", 0.7), 
                        args.get("max_tokens", 2048),
                        stream=True
                    ):
                        yield chunk
                return StreamingResponse(stream_generator(), media_type="text/event-stream")
            else:
                chunks = []
                async for chunk in forward_to_llm(
                    prompt, 
                    args.get("system_prompt"), 
                    args.get("temperature", 0.7), 
                    args.get("max_tokens", 2048),
                    stream=False
                ):
                    chunks.append(chunk)
                return {
                    "jsonrpc": "2.0",
                    "id": body.get("id"),
                    "result": {
                        "content": [{"type": "text", "text": "".join(chunks)}]
                    }
                }
                
    if mcp_method == "server/discover" or body.get("method") == "server/discover":
         return {
            "jsonrpc": "2.0",
            "id": body.get("id"),
            "result": {
                "capabilities": {
                    "tools": {}
                }
            }
         }

    raise HTTPException(status_code=404, detail=f"Method not found: {mcp_method or body.get('method')}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9932)
    args = parser.parse_args()
    
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
