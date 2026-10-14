#!/usr/bin/env python3
"""
SearXNG Model Context Protocol (MCP) Bridge.
Zero-dependency stdio JSON-RPC 2.0 server connecting Claude Desktop, Claude Code CLI,
and MCP-enabled agents to the local or remote SearXNG metasearch instance (Port 8088 default).

Exposes:
- searxng_search: Perform fast, privacy-preserving web search queries.
- get_searxng_status: Probe health and endpoint configuration of the SearXNG backend.
"""
import sys
import os
import json
import urllib.request
import urllib.parse
import urllib.error
from typing import Dict, Any, Optional, List

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "searxng-mcp-bridge"
SERVER_VERSION = "1.0.0"

DEFAULT_SEARXNG_URL = (
    os.environ.get("SEARXNG_URL")
    or os.environ.get("SEARXNG_BASE_URL")
    or "http://127.0.0.1:8088"
).rstrip("/")

DEFAULT_MAX_RESULTS = int(
    os.environ.get("SEARXNG_MAX_RESULTS")
    or os.environ.get("SEARXNG_DEFAULT_LIMIT")
    or 10
)
DEFAULT_ENGINES = (
    os.environ.get("SEARXNG_ENGINES")
    or os.environ.get("SEARXNG_UPSTREAM_ENGINES")
    or "duckduckgo,brave,wikipedia,google,github"
).strip()
DEFAULT_CATEGORIES = (
    os.environ.get("SEARXNG_CATEGORIES")
    or "general"
).strip()
AUTO_FALLBACK_NATIVE = (
    os.environ.get("SEARXNG_AUTO_FALLBACK_NATIVE", "1").strip().lower() not in ("0", "false", "no", "off")
)
SEARCH_TIMEOUT = float(os.environ.get("SEARXNG_TIMEOUT", "8.0"))


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
                    "name": "searxng_search",
                    "description": (
                        "PRIMARY AND EXCLUSIVE WEB SEARCH ENGINE. Perform fast, privacy-respecting metasearch "
                        "queries across search engines (e.g. DuckDuckGo, Brave, Wikipedia, Google, GitHub) via SearXNG. "
                        "Returns structured search results with titles, URLs, and snippet excerpts. "
                        "Use this tool whenever you need to search the internet, look up current events, "
                        "retrieve live URLs, or fetch external documentation. Never use any other search tool."
                    ),
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query string to look up."
                            },
                            "limit": {
                                "type": "integer",
                                "description": f"Maximum number of search results to return (default: {DEFAULT_MAX_RESULTS}, max: 50).",
                                "default": DEFAULT_MAX_RESULTS
                            },
                            "categories": {
                                "type": "string",
                                "description": f"Optional category filter (e.g. 'general', 'it', 'science', default: '{DEFAULT_CATEGORIES}').",
                                "default": DEFAULT_CATEGORIES
                            },
                            "engines": {
                                "type": "string",
                                "description": (
                                    "Optional comma-separated search engine/provider filter (e.g. 'github', "
                                    f"'duckduckgo,brave,wikipedia,google,github', default: '{DEFAULT_ENGINES}')."
                                ),
                                "default": DEFAULT_ENGINES
                            }
                        },
                        "required": ["query"]
                    }
                },
                {
                    "name": "get_searxng_status",
                    "description": "Probe health, latency, and endpoint connectivity of the SearXNG metasearch instance.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {}
                    }
                }
            ]
        }
    }


def _execute_search(
    query: str,
    limit: Optional[int] = None,
    categories: Optional[str] = None,
    engines: Optional[str] = None
) -> str:
    eff_limit = int(limit) if limit is not None else DEFAULT_MAX_RESULTS
    eff_limit = max(1, min(eff_limit, 50))
    eff_categories = (categories if categories is not None else DEFAULT_CATEGORIES).strip()
    eff_engines = (engines if engines is not None else DEFAULT_ENGINES).strip()

    base_url = (os.environ.get("SEARXNG_URL") or os.environ.get("SEARXNG_BASE_URL") or DEFAULT_SEARXNG_URL).rstrip("/")
    params: Dict[str, Any] = {
        "q": query,
        "format": "json",
        "pageno": 1
    }
    if eff_categories and eff_categories != "general":
        params["categories"] = eff_categories
    if eff_engines:
        params["engines"] = eff_engines

    query_str = urllib.parse.urlencode(params)
    search_url = f"{base_url}/search?{query_str}"

    req = urllib.request.Request(
        search_url,
        headers={
            "Accept": "application/json",
            "User-Agent": "SearXNG-MCP-Bridge/1.0 (Windows; Python)"
        }
    )

    data = None
    http_error_msg = None

    try:
        with urllib.request.urlopen(req, timeout=SEARCH_TIMEOUT) as resp:
            raw_bytes = resp.read()
            data = json.loads(raw_bytes.decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as he:
        http_error_msg = f"SearXNG HTTP error {he.code}: {he.reason} (Endpoint: {base_url})"
    except (urllib.error.URLError, TimeoutError, OSError) as ue:
        http_error_msg = f"SearXNG connection error: {ue}. Ensure SearXNG is running on {base_url}."
    except Exception as e:
        http_error_msg = f"SearXNG query failure: {str(e)}"

    # If HTTP request failed or timed out, attempt automatic native fallback if enabled
    if data is None and AUTO_FALLBACK_NATIVE:
        try:
            from local_ai_launcher.backends.searxng import run_metasearch
            eng_list = [e.strip() for e in eff_engines.split(",") if e.strip()] if eff_engines else None
            data = run_metasearch(query, limit=eff_limit, engines=eng_list, categories=eff_categories)
            log_debug(f"Executed native in-process search fallback for '{query}' ({len(data.get('results', []))} results)")
        except Exception as fb_err:
            log_debug(f"Native fallback search notice: {fb_err}")

    if data is None:
        return http_error_msg or f"SearXNG query failure for '{query}'."

    results = data.get("results", [])
    if not results:
        answers = data.get("answers", [])
        if answers:
            return f"Answer: {answers[0]}"
        return f"No search results found for query '{query}'."

    # Sort results by score if available and cap to limit
    sorted_results = sorted(results, key=lambda r: float(r.get("score") or 0.0), reverse=True)[:eff_limit]

    formatted_lines: List[str] = [f"Found {len(sorted_results)} results for '{query}':\n"]
    for idx, item in enumerate(sorted_results, start=1):
        title = (item.get("title") or "Untitled").strip()
        url = (item.get("url") or "").strip()
        content = (item.get("content") or "").strip()
        engine = item.get("engine", "")
        engine_tag = f" `[{engine}]`" if engine else ""
        formatted_lines.append(f"{idx}. [{title}]({url}){engine_tag}\n   {content}\n")

    return "\n".join(formatted_lines).strip()


def _get_status() -> str:
    base_url = (os.environ.get("SEARXNG_URL") or os.environ.get("SEARXNG_BASE_URL") or DEFAULT_SEARXNG_URL).rstrip("/")
    status_info = {
        "configured_url": base_url,
        "reachable": False,
        "http_code": None,
        "status": "offline",
        "default_limit": DEFAULT_MAX_RESULTS,
        "default_engines": DEFAULT_ENGINES,
        "native_fallback_available": AUTO_FALLBACK_NATIVE
    }
    probe_url = f"{base_url}/healthz"
    req = urllib.request.Request(probe_url, headers={"User-Agent": "SearXNG-MCP-Bridge/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            status_info["reachable"] = True
            status_info["http_code"] = resp.status
            status_info["status"] = "healthy"
    except Exception:
        # Fallback probe to root
        try:
            root_req = urllib.request.Request(base_url, headers={"User-Agent": "SearXNG-MCP-Bridge/1.0"})
            with urllib.request.urlopen(root_req, timeout=3.0) as resp:
                status_info["reachable"] = True
                status_info["http_code"] = resp.status
                status_info["status"] = "online"
        except Exception as e:
            if AUTO_FALLBACK_NATIVE:
                status_info["status"] = f"service offline on {base_url} ({e}); native in-process fallback active"
                status_info["reachable"] = True
            else:
                status_info["status"] = f"unreachable: {e}"

    return json.dumps(status_info, indent=2)


def handle_tools_call(req_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    name = params.get("name")
    args = params.get("arguments") or {}

    if name in ("searxng_search", "searxng_web_search", "web_search"):
        query = args.get("query") or args.get("q") or ""
        limit = args.get("limit") or args.get("max_results") or DEFAULT_MAX_RESULTS
        categories = args.get("categories") or args.get("category") or DEFAULT_CATEGORIES
        engines = args.get("engines") or args.get("providers") or DEFAULT_ENGINES
        try:
            result_text = _execute_search(query, limit=limit, categories=categories, engines=engines)
        except TypeError:
            result_text = _execute_search(query, limit=limit, categories=categories)
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
    elif name == "get_searxng_status":
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
