"""
SearXNG Portable Metasearch Backend Service for local_ai_launcher.
Coordinates multi-tier metasearch engine deployment:
- Tier 3 (Default): Headless Docker Engine inside WSL2 (-p 127.0.0.1:8088:8080, UWSGI_WORKERS=1, ~400MB RAM, zero Docker Desktop bloat).
- Tier 2: Windows-Native Portable Python Metasearch Engine (~25MB RAM, 0 Docker/WSL, 0 MB GPU VRAM).
- Tier 1: Zero-Footprint Remote Mode (0 MB local RAM, public/private instance).

SearXNG is classified as an essential harness service (like Parakeet NPU on Port 8000)
and is strictly protected from memory clawback evictions when Strata is active.
"""

import os
import sys
import time
import shutil
import secrets
import logging
import argparse
import subprocess
import urllib.request
import urllib.parse
import urllib.error
import re
import html
import threading
from pathlib import Path
from typing import List, Optional, Dict, Any

from local_ai_launcher.config import AI_WORKSPACE_DIR, config
from local_ai_launcher.process import run_silent_subprocess, start_process

logger = logging.getLogger("searxng_backend")

DEFAULT_SEARXNG_PORT = 8088
DEFAULT_SEARXNG_DIR = AI_WORKSPACE_DIR / "tools" / "searxng"


# =============================================================================
# Native Metasearch Engine (DuckDuckGo Lite + Wikipedia Opensearch)
# Provides a 100% compliant SearXNG JSON API format with 0 dependencies.
# =============================================================================

def fetch_duckduckgo_lite(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Fetches and parses search results from DuckDuckGo Lite."""
    results = []
    try:
        url = "https://lite.duckduckgo.com/lite/"
        data = urllib.parse.urlencode({"q": query}).encode("utf-8")
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        req = urllib.request.Request(url, data=data, headers=headers)
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            page_html = resp.read().decode("utf-8", errors="ignore")

        raw_links = re.findall(
            r"<a\s+[^>]*href=['\"]([^'\"]+)['\"][^>]*class=['\"]result-link['\"][^>]*>(.*?)</a>",
            page_html,
            re.IGNORECASE
        )
        if not raw_links:
            raw_links = re.findall(
                r"<a\s+[^>]*class=['\"]result-link['\"][^>]*href=['\"]([^'\"]+)['\"][^>]*>(.*?)</a>",
                page_html,
                re.IGNORECASE
            )
        snippets = re.findall(
            r"<td\s+[^>]*class=['\"]result-snippet['\"][^>]*>(.*?)</td>",
            page_html,
            re.DOTALL | re.IGNORECASE
        )

        for i in range(min(len(raw_links), len(snippets), limit)):
            link_href = raw_links[i][0].strip()
            # Unescape and strip HTML tags from title and snippet
            title_text = html.unescape(re.sub(r'<[^>]+>', '', raw_links[i][1])).strip()
            snippet_text = html.unescape(re.sub(r'<[^>]+>', '', snippets[i])).strip()

            # Handle DuckDuckGo redirect wrapper if present
            if "duckduckgo.com/l/?uddg=" in link_href:
                parsed_url = urllib.parse.urlparse(link_href)
                query_params = urllib.parse.parse_qs(parsed_url.query)
                if "uddg" in query_params:
                    link_href = query_params["uddg"][0]

            if title_text and link_href.startswith("http"):
                results.append({
                    "title": title_text,
                    "url": link_href,
                    "content": snippet_text,
                    "engine": "duckduckgo",
                    "score": 1.0 - (i * 0.05)
                })
    except Exception as e:
        logger.debug(f"DuckDuckGo Lite search notice: {e}")

    return results


def fetch_duckduckgo_api(query: str, limit: int = 3) -> List[Dict[str, Any]]:
    """Fetches instant answers and related topic links from official DuckDuckGo API."""
    results = []
    try:
        url = f"https://api.duckduckgo.com/?q={urllib.parse.quote_plus(query)}&format=json&no_html=1&skip_disambig=0"
        req = urllib.request.Request(url, headers={"User-Agent": "LocalAI-SearXNG/1.0 (Windows)"})
        with urllib.request.urlopen(req, timeout=4.0) as resp:
            import json
            data = json.loads(resp.read().decode("utf-8", errors="ignore"))
            if data.get("AbstractURL") and data.get("AbstractText"):
                results.append({
                    "title": data.get("Heading") or query.title(),
                    "url": data.get("AbstractURL"),
                    "content": data.get("AbstractText"),
                    "engine": "duckduckgo",
                    "score": 1.0
                })
            for topic in data.get("RelatedTopics", []):
                if len(results) >= limit:
                    break
                if isinstance(topic, dict) and topic.get("FirstURL") and topic.get("Text"):
                    text = topic.get("Text", "")
                    title = text.split(" - ")[0] if " - " in text else text[:60]
                    results.append({
                        "title": title,
                        "url": topic.get("FirstURL"),
                        "content": text,
                        "engine": "duckduckgo",
                        "score": 0.9
                    })
    except Exception as e:
        logger.debug(f"DuckDuckGo API search notice: {e}")

    return results


def fetch_wikipedia_opensearch(query: str, limit: int = 3) -> List[Dict[str, Any]]:
    """Fetches encyclopedic summaries from Wikipedia Opensearch API."""
    results = []
    try:
        url = f"https://en.wikipedia.org/w/api.php?action=opensearch&search={urllib.parse.quote_plus(query)}&limit={limit}&namespace=0&format=json"
        req = urllib.request.Request(url, headers={"User-Agent": "LocalAI-SearXNG/1.0 (Windows)"})
        with urllib.request.urlopen(req, timeout=4.0) as resp:
            import json
            data = json.loads(resp.read().decode("utf-8", errors="ignore"))
            if isinstance(data, list) and len(data) >= 4:
                titles = data[1]
                snippets = data[2]
                urls = data[3]
                for i in range(min(len(titles), len(urls))):
                    if urls[i]:
                        results.append({
                            "title": titles[i],
                            "url": urls[i],
                            "content": snippets[i] if i < len(snippets) else "",
                            "engine": "wikipedia",
                            "score": 0.95
                        })
    except Exception as e:
        logger.debug(f"Wikipedia search notice: {e}")

    return results


def fetch_github_search(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Fetches repository search results directly from GitHub API and site search."""
    results = []
    # 1. Official GitHub Search API
    try:
        clean_q = re.sub(r'site:github\.com', '', query, flags=re.IGNORECASE).strip()
        url = f"https://api.github.com/search/repositories?q={urllib.parse.quote_plus(clean_q)}&per_page={max(1, limit)}"
        headers = {
            "User-Agent": "LocalAI-SearXNG/1.0 (Windows; Python)",
            "Accept": "application/vnd.github.v3+json",
        }
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            import json
            data = json.loads(resp.read().decode("utf-8", errors="ignore"))
            items = data.get("items", [])
            for i, item in enumerate(items[:limit]):
                full_name = item.get("full_name") or item.get("name") or "GitHub Repo"
                html_url = item.get("html_url") or ""
                desc = item.get("description") or ""
                stars = item.get("stargazers_count", 0)
                lang = item.get("language") or ""
                meta_parts = []
                if lang:
                    meta_parts.append(f"Lang: {lang}")
                if stars:
                    meta_parts.append(f"Stars: {stars}")
                meta_str = f" ({', '.join(meta_parts)})" if meta_parts else ""
                content = f"{desc}{meta_str}".strip() or f"GitHub repository {full_name}"

                if html_url:
                    results.append({
                        "title": full_name,
                        "url": html_url,
                        "content": content,
                        "engine": "github",
                        "score": 0.98 - (i * 0.05)
                    })
    except Exception as e:
        logger.debug(f"GitHub API search notice: {e}")

    # 2. If API returned few results or encountered rate limit, query DuckDuckGo Lite scoped to github.com
    if len(results) < limit:
        try:
            ddg_q = f"site:github.com {query}"
            needed = limit - len(results)
            ddg_hits = fetch_duckduckgo_lite(ddg_q, limit=needed)
            existing_urls = {r["url"] for r in results}
            for hit in ddg_hits:
                if hit["url"] not in existing_urls and "github.com" in hit["url"]:
                    hit["engine"] = "github"
                    hit["score"] = 0.90
                    results.append(hit)
        except Exception as e:
            logger.debug(f"DuckDuckGo site:github.com search notice: {e}")

    return results[:limit]


def run_metasearch(
    query: str,
    limit: int = 10,
    engines: Optional[List[str]] = None,
    categories: Optional[str] = None
) -> Dict[str, Any]:
    """Aggregates results across engines and packages into standard SearXNG JSON schema."""
    aggregated: List[Dict[str, Any]] = []
    seen_urls = set()

    active_engines = [
        str(e).strip().lower()
        for e in (engines or getattr(config, "searxng_upstream_engines", ["duckduckgo", "brave", "wikipedia", "google", "github"]))
    ]

    # 1. Query GitHub if requested or query explicitly targets github
    if "github" in active_engines or "github" in query.lower():
        github_hits = fetch_github_search(query, limit=min(5, limit))
        for hit in github_hits:
            if hit["url"] not in seen_urls:
                seen_urls.add(hit["url"])
                aggregated.append(hit)

    # 2. Query DuckDuckGo API for instant answers and related topics
    if any(e in active_engines for e in ("duckduckgo", "ddg")):
        ddg_api_hits = fetch_duckduckgo_api(query, limit=limit)
        for hit in ddg_api_hits:
            if hit["url"] not in seen_urls:
                seen_urls.add(hit["url"])
                aggregated.append(hit)

    # 3. Query Wikipedia for direct encyclopedic knowledge hits
    if "wikipedia" in active_engines:
        wiki_hits = fetch_wikipedia_opensearch(query, limit=3)
        for hit in wiki_hits:
            if hit["url"] not in seen_urls:
                seen_urls.add(hit["url"])
                aggregated.append(hit)

    # 4. Query DuckDuckGo Lite for additional web results
    if any(e in active_engines for e in ("duckduckgo", "ddg", "google", "brave")):
        ddg_hits = fetch_duckduckgo_lite(query, limit=limit)
        for hit in ddg_hits:
            if hit["url"] not in seen_urls:
                seen_urls.add(hit["url"])
                aggregated.append(hit)

    # Cap to requested limit
    capped_results = aggregated[:limit]

    return {
        "query": query,
        "number_of_results": len(capped_results),
        "results": capped_results,
        "answers": [],
        "corrections": [],
        "infoboxes": [],
        "suggestions": [],
        "unresponsive_engines": []
    }


def create_fastapi_app():
    """Builds FastAPI application exposing standard SearXNG endpoints."""
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    app = FastAPI(title="SearXNG Native Compatible Metasearch Service", version="1.0.0")

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok", "service": "searxng-native"}

    @app.get("/")
    async def root():
        return {"status": "ok", "service": "searxng-native", "message": "SearXNG Compatible Metasearch Ready"}

    @app.get("/search")
    async def search(
        request: Request,
        q: Optional[str] = None,
        query: Optional[str] = None,
        format: Optional[str] = "json",
        limit: int = 10,
        engines: Optional[str] = None,
        categories: Optional[str] = None,
    ):
        target_query = (q or query or "").strip()
        if not target_query:
            return JSONResponse({
                "query": "",
                "number_of_results": 0,
                "results": [],
                "answers": []
            })

        eng_list = [e.strip() for e in engines.split(",") if e.strip()] if engines else None
        data = run_metasearch(target_query, limit=limit, engines=eng_list, categories=categories)
        return JSONResponse(data)

    return app


# =============================================================================
# Lifecycle & Process Management
# =============================================================================

def is_searxng_healthy(url: Optional[str] = None, timeout: float = 2.5) -> bool:
    """Probes SearXNG instance health check endpoint or root endpoint."""
    base_url = (url or getattr(config, "searxng_effective_url", f"http://127.0.0.1:{DEFAULT_SEARXNG_PORT}")).rstrip("/")
    probe_url = f"{base_url}/healthz"
    req = urllib.request.Request(probe_url, headers={"User-Agent": "SearXNG-HealthProbe/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status in (200, 204, 302)
    except Exception:
        # Fallback probe to root index
        try:
            req_root = urllib.request.Request(base_url, headers={"User-Agent": "SearXNG-HealthProbe/1.0"})
            with urllib.request.urlopen(req_root, timeout=timeout) as resp:
                return resp.status in (200, 302)
        except Exception:
            return False


def ensure_searxng_settings_file(
    target_path: Optional[Path] = None,
    port: int = DEFAULT_SEARXNG_PORT,
    engines: Optional[List[str]] = None,
    limiter: bool = False,
) -> Path:
    """Generates or updates a hardened minimal settings.yml for SearXNG."""
    out_file = target_path or (DEFAULT_SEARXNG_DIR / "searx" / "settings.yml")
    out_file.parent.mkdir(parents=True, exist_ok=True)

    secret_key = secrets.token_hex(32)
    if out_file.exists():
        try:
            content = out_file.read_text(encoding="utf-8")
            for line in content.splitlines():
                if "secret_key:" in line:
                    parts = line.split(":", 1)
                    if len(parts) == 2 and parts[1].strip() and parts[1].strip() != '""':
                        secret_key = parts[1].strip().strip('"').strip("'")
                        break
        except Exception:
            pass

    selected_engines = engines or getattr(
        config, "searxng_upstream_engines", ["duckduckgo", "brave", "wikipedia", "google", "github"]
    )
    engine_yaml_blocks = []
    for eng in selected_engines:
        engine_yaml_blocks.append(f"""  - name: {eng}
    engine: {eng}
    enable: true""")

    yaml_content = f"""# SearXNG Minimal Configuration generated by local_ai_launcher
use_default_settings: true

server:
  port: {port}
  bind_address: "0.0.0.0"
  secret_key: "{secret_key}"
  limiter: {'true' if limiter else 'false'}
  image_proxy: false

search:
  safe_search: 0
  autocomplete: ""
  formats:
    - html
    - json

outgoing:
  request_timeout: 6.0
  max_request_timeout: 8.0

engines:
{chr(10).join(engine_yaml_blocks)}
"""
    try:
        from local_ai_launcher.paths import atomic_write_text
        atomic_write_text(out_file, yaml_content)
    except Exception:
        out_file.write_text(yaml_content, encoding="utf-8")

    limiter_file = out_file.parent / "limiter.toml"
    if not limiter_file.exists():
        limiter_content = """# Minimal limiter configuration disabling rate limits for local harness queries
[botdetection.ip_limit]
default = 0
link_token = false
"""
        try:
            from local_ai_launcher.paths import atomic_write_text
            atomic_write_text(limiter_file, limiter_content)
        except Exception:
            limiter_file.write_text(limiter_content, encoding="utf-8")

    return out_file


def _check_wsl_status() -> tuple[bool, Optional[str]]:
    """Checks if WSL2 is installed and has at least one distribution available."""
    if sys.platform != "win32":
        return False, None
    try:
        res = run_silent_subprocess(["wsl.exe", "-l", "-q"], timeout=5)
        if res.returncode == 0:
            lines = [l.strip().replace("\x00", "") for l in (res.stdout or "").splitlines() if l.strip()]
            if lines:
                return True, lines[0]
        return False, None
    except Exception:
        return False, None


def to_wsl_path(path: Path) -> str:
    """
    Converts any Windows Path (e.g. C:\\aiWorkspace or D:\\custom\\dir) to its WSL mount path (/mnt/c/...).
    Safely handles non-C: drives, relative paths, and cross-platform environments.
    """
    try:
        if sys.platform != "win32":
            return path.as_posix()
        resolved = Path(path).resolve()
        drive = resolved.drive.rstrip(":").lower()
        if drive:
            rel = resolved.relative_to(resolved.anchor).as_posix()
            return f"/mnt/{drive}/{rel}" if rel else f"/mnt/{drive}"
        return resolved.as_posix()
    except Exception:
        raw = str(path).replace("\\", "/").lstrip("/")
        if len(raw) >= 2 and raw[1] == ":":
            drive = raw[0].lower()
            return f"/mnt/{drive}/{raw[2:].lstrip('/')}"
        return f"/mnt/c/{raw}"


def ensure_wsl_keepalive_config() -> bool:
    """
    Ensures ~/.wslconfig has vmIdleTimeout=-1 configured so background daemons
    (like Docker Engine and SearXNG) do not suspend when idle.
    Preserves all existing user settings in .wslconfig.
    """
    if sys.platform != "win32":
        return False
    try:
        user_home = Path.home()
        wslconfig_path = user_home / ".wslconfig"
        if not wslconfig_path.exists():
            wslconfig_content = "[wsl2]\nvmIdleTimeout=-1\n"
            wslconfig_path.write_text(wslconfig_content, encoding="utf-8")
            logger.info("  -> Created ~/.wslconfig with vmIdleTimeout=-1 for WSL daemon persistence.")
            return True

        content = wslconfig_path.read_text(encoding="utf-8")
        if "vmidletimeout" in content.lower():
            return True

        lines = content.splitlines()
        new_lines = []
        found_wsl2 = False
        inserted = False
        for line in lines:
            new_lines.append(line)
            if line.strip().lower() == "[wsl2]":
                found_wsl2 = True
                new_lines.append("vmIdleTimeout=-1")
                inserted = True

        if not inserted:
            if not found_wsl2:
                new_lines.append("\n[wsl2]")
            new_lines.append("vmIdleTimeout=-1")

        wslconfig_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        logger.info("  -> Added vmIdleTimeout=-1 to ~/.wslconfig.")
        return True
    except Exception as e:
        logger.debug(f"Notice: .wslconfig check: {e}")
        return False


def ensure_wsl_distribution(distro: str = "Ubuntu-26.04") -> bool:
    """
    Ensures a valid WSL2 distribution exists. If zero distributions are installed,
    portably provisions Ubuntu-26.04 LTS via wsl.exe --install with online fallbacks.
    """
    if sys.platform != "win32":
        return False
    if not shutil.which("wsl.exe"):
        logger.info("  -> [WSL Provisioning] wsl.exe not found on system. WSL2 is not available.")
        return False

    has_wsl, existing_distro = _check_wsl_status()
    if has_wsl:
        return True

    candidates = [distro]
    if distro != "Ubuntu" and not distro.startswith("Ubuntu-26"):
        candidates.append("Ubuntu-26.04")
    if "Ubuntu" not in candidates:
        candidates.append("Ubuntu")

    for target in candidates:
        logger.info(f"  -> [WSL Provisioning] No WSL distributions found. Automatically installing {target}...")
        try:
            install_res = run_silent_subprocess(
                ["wsl.exe", "--install", target, "--no-launch", "--web-download"],
                timeout=180
            )
            if install_res.returncode == 0:
                logger.info(f"  -> [WSL Provisioning] {target} installed successfully.")
                return True
            else:
                logger.warning(f"  -> [WSL Provisioning] {target} install notice: {install_res.stderr or install_res.stdout}")
        except Exception as e:
            logger.warning(f"  -> [WSL Provisioning] Failed to provision {target}: {e}")

    return False


def start_searxng_native(processes: List, port: int = DEFAULT_SEARXNG_PORT) -> bool:
    """
    Tier 2: Starts Windows-Native Metasearch Engine on Port 8088.
    Zero-docker, zero-WSL, ~25MB RAM, 0 MB GPU VRAM.
    """
    if is_searxng_healthy(f"http://127.0.0.1:{port}", timeout=1.0):
        logger.info(f"  -> SearXNG is already running on http://127.0.0.1:{port}.")
        return True

    try:
        from local_ai_launcher.process import is_port_in_use, force_kill_port
        if is_port_in_use(port) and not is_searxng_healthy(f"http://127.0.0.1:{port}", timeout=1.0):
            logger.warning(f"Port {port} is occupied by an unrecognized process. Reclaiming port safely...")
            force_kill_port(port)
            time.sleep(0.5)
    except Exception as e:
        logger.debug(f"Port pre-flight notice: {e}")

    python_exe = sys.executable

    logger.info(f"Starting Windows-Native SearXNG (Port {port}, FastAPI/Uvicorn ~25MB RAM)...")

    proc = start_process(
        "SearXNG Metasearch",
        f'& "{python_exe}" -m local_ai_launcher.backends.searxng --port {port}',
        str(AI_WORKSPACE_DIR),
        show_window=False
    )
    if proc:
        processes.append(("SearXNG Metasearch", proc, True))

    for _ in range(12):
        if is_searxng_healthy(f"http://127.0.0.1:{port}", timeout=1.0):
            logger.info(f"  -> SearXNG (Native Tier 2) ready on http://127.0.0.1:{port}")
            return True
        time.sleep(0.5)

    return False


def start_searxng_wsl_docker(
    processes: List,
    port: int = DEFAULT_SEARXNG_PORT,
    distro: str = "",
    container_name: str = "searxng",
    workers: int = 1,
    threads: int = 4,
) -> bool:
    """
    Tier 3 (Default): Starts SearXNG inside a headless WSL2 Docker Engine without Docker Desktop bloat.
    Auto-provisions Ubuntu-26.04 LTS if zero distros are installed, and falls back to Tier 2 if needed.
    """
    if is_searxng_healthy(f"http://127.0.0.1:{port}", timeout=1.0):
        logger.info(f"  -> SearXNG is already running and healthy at http://127.0.0.1:{port}.")
        return True

    # Ensure WSL background persistence so Windows does not idle-suspend background Docker daemons
    ensure_wsl_keepalive_config()

    has_wsl, default_distro = _check_wsl_status()
    if not has_wsl:
        target_distro = distro or getattr(config, "searxng_wsl_distro", "") or "Ubuntu-26.04"
        def _bg_provision():
            try:
                ensure_wsl_distribution(distro=target_distro)
            except Exception as e:
                logger.debug(f"Background WSL provisioning notice: {e}")
        provision_thread = threading.Thread(target=_bg_provision, daemon=True, name="searxng-wsl-provision")
        provision_thread.start()

        logger.info(
            f"  -> [SearXNG Tier 3] WSL2 has no installed distributions. "
            f"Activating Tier 2 Windows-Native search instantly (<0.5s) while provisioning {target_distro} in background..."
        )
        return start_searxng_native(processes, port)

    target_distro = distro or getattr(config, "searxng_wsl_distro", "") or default_distro
    wsl_cmd_base = ["wsl.exe"]
    if target_distro:
        wsl_cmd_base.extend(["-d", target_distro])
    wsl_cmd_base.extend(["-u", "root"])

    logger.info(f"Checking SearXNG Docker container in WSL ({target_distro or 'default'})...")

    try:
        check_docker = run_silent_subprocess(wsl_cmd_base + ["docker", "info"], timeout=8)
        if check_docker.returncode != 0:
            has_docker_bin = run_silent_subprocess(wsl_cmd_base + ["which", "docker"], timeout=5).returncode == 0
            if not has_docker_bin:
                logger.info(f"  -> [WSL Provisioning] Docker not found in {target_distro or 'default'}. Auto-installing docker.io...")
                run_silent_subprocess(wsl_cmd_base + ["apt-get", "update", "-y"], timeout=60)
                run_silent_subprocess(wsl_cmd_base + ["apt-get", "install", "-y", "docker.io"], timeout=120)
            run_silent_subprocess(wsl_cmd_base + ["service", "docker", "start"], timeout=10)
    except Exception as e:
        logger.debug(f"Notice: WSL Docker check: {e}")

    try:
        ps_check = run_silent_subprocess(
            wsl_cmd_base + ["docker", "ps", "-a", "--filter", f"name={container_name}", "--format", "{{.Status}}"],
            timeout=8
        )
        status_output = (ps_check.stdout or "").strip()
        if ps_check.returncode == 0 and status_output:
            if "Up" in status_output:
                logger.info(f"  -> SearXNG container '{container_name}' is already running in WSL.")
            else:
                logger.info(f"  -> Starting stopped SearXNG container '{container_name}' in WSL...")
                run_silent_subprocess(wsl_cmd_base + ["docker", "start", container_name], timeout=15)
            # Ensure container resource limits are enforced (512MB RAM, 2 CPUs)
            run_silent_subprocess(
                wsl_cmd_base + ["docker", "update", "--memory", "512m", "--memory-swap", "1g", "--cpus", "2", container_name],
                timeout=8
            )
        else:
            ensure_searxng_settings_file(port=8080)
            wsl_mount_dir = to_wsl_path(DEFAULT_SEARXNG_DIR / "searx")

            logger.info(f"  -> Launching new SearXNG container '{container_name}' (Port {port}:8080, workers={workers}, cap=512MB RAM)...")
            run_cmd = wsl_cmd_base + [
                "docker", "run", "-d",
                "--name", container_name,
                "--restart", "unless-stopped",
                "--memory", "512m",
                "--memory-swap", "1g",
                "--cpus", "2",
                "-p", f"127.0.0.1:{port}:8080",
                "-v", f"{wsl_mount_dir}:/etc/searxng:rw",
                "-e", f"UWSGI_WORKERS={workers}",
                "-e", f"UWSGI_THREADS={threads}",
                "searxng/searxng:latest"
            ]
            run_res = run_silent_subprocess(run_cmd, timeout=30)
            if run_res.returncode != 0:
                logger.warning(f"  -> Notice launching SearXNG in WSL: {run_res.stderr or run_res.stdout}")
                logger.info("  -> Falling back to Tier 2 Windows-Native Metasearch Engine...")
                return start_searxng_native(processes, port)
    except Exception as e:
        logger.warning(f"  -> Notice configuring SearXNG in WSL: {e}")
        return start_searxng_native(processes, port)

    for _ in range(12):
        if is_searxng_healthy(f"http://127.0.0.1:{port}", timeout=1.0):
            logger.info(f"  -> SearXNG (WSL2 Docker Tier 3) ready on http://127.0.0.1:{port}")
            return True
        time.sleep(0.5)

    # If container failed to bind, fall back to native
    logger.info("  -> Container did not respond on Port 8088. Falling back to Tier 2 Windows-Native...")
    return start_searxng_native(processes, port)


DONSETCH_HARNESSES = {"pi", "deepseek", "dsh"}


def should_start_searxng(selected_harnesses: Optional[Any] = None) -> bool:
    """
    Determines whether the SearXNG metasearch service should be started.

    SearXNG should NOT be started if:
    1. searxng_enabled is False or searxng_mode is 'disabled' / 'off' / 'none'.
    2. No harnesses are started / selected.
    3. The ONLY harnesses started are Pi and/or DeepSeek (dsh), which use DonSeTch
       (unless DonSeTch is explicitly disabled for that harness, requiring SearXNG fallback).
    """
    if not getattr(config, "searxng_enabled", True):
        return False

    mode = str(getattr(config, "searxng_mode", "wsl_docker")).strip().lower()
    if mode in ("disabled", "off", "none"):
        return False

    if selected_harnesses is not None:
        raw = selected_harnesses
    else:
        raw = getattr(config, "choose_harness", [])

    if isinstance(raw, str):
        raw = [raw]
    elif not isinstance(raw, (list, tuple, set)):
        raw = []

    active = [
        str(h).strip().lower() for h in raw
        if h and str(h).strip().lower() not in ("none", "off", "disabled", "no", "false")
    ]
    if not active:
        logger.info("Notice: No active coding harness selected; skipping SearXNG metasearch service startup.")
        return False

    # Check if any active harness requires SearXNG:
    # Pi and DeepSeek (dsh) use DonSeTch by default.
    # Other harnesses (claude, hermes, orca, t3, opencode, etc.) use SearXNG.
    for h in active:
        if h == "pi":
            if not getattr(config, "pi_donsetch_enabled", True):
                return True
        elif h in ("deepseek", "dsh"):
            if not getattr(config, "deepseek_donsetch_enabled", True):
                return True
        else:
            return True

    logger.info("Notice: Only DonSeTch harnesses selected (pi/dsh); skipping SearXNG metasearch service startup.")
    return False


def start_searxng_server(processes: List, selected_harnesses: Optional[Any] = None) -> bool:
    """
    Main entry point for starting the SearXNG metasearch service based on configured tier.
    """
    if selected_harnesses is not None and not should_start_searxng(selected_harnesses):
        return True

    if not getattr(config, "searxng_enabled", True):
        return True

    mode = str(getattr(config, "searxng_mode", "wsl_docker")).strip().lower()
    if mode in ("disabled", "off", "none"):
        return True

    port = int(getattr(config, "searxng_port", DEFAULT_SEARXNG_PORT))

    if mode == "remote":
        remote_url = getattr(config, "searxng_remote_url", "https://searx.be")
        logger.info(f"SearXNG configured in Tier 1 (Remote Mode: {remote_url}). Local processes skipped (0 MB RAM).")
        return True

    elif mode == "wsl_docker":
        workers = int(getattr(config, "searxng_uwsgi_workers", 1))
        threads = int(getattr(config, "searxng_uwsgi_threads", 4))
        container_name = getattr(config, "searxng_wsl_container_name", "searxng")
        distro = getattr(config, "searxng_wsl_distro", "")
        return start_searxng_wsl_docker(
            processes=processes,
            port=port,
            distro=distro,
            container_name=container_name,
            workers=workers,
            threads=threads,
        )

    elif mode == "native":
        return start_searxng_native(processes=processes, port=port)

    else:
        logger.warning(f"Unknown SearXNG mode '{mode}'. Defaulting to Tier 3 (wsl_docker).")
        return start_searxng_wsl_docker(processes=processes, port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SearXNG Native Compatible Metasearch Service")
    parser.add_argument("--port", type=int, default=DEFAULT_SEARXNG_PORT, help="Port to listen on (default: 8088)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    args = parser.parse_args()

    import uvicorn
    app = create_fastapi_app()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
