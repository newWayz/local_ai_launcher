"""
Canonical CPU-only Media Production Director Backend.
=====================================================
Runs Gemma 4 / lightweight vision-language models strictly via llama-server with -ngl 0,
pinned to CPU cores so GPU VRAM is 100% available for video diffusion.
Exposes standard OpenAI-compatible REST API at http://127.0.0.1:8081/v1

Architecture Note (Port 8081):
Port 8081 is the unified CPU llama-server.exe worker port on this system.
It hosts the Gemma 4 Director during media production workflows and the Decider-4B
inference worker during fast-path agent triage. These two workloads are strictly
isolated, mutually exclusive, and never overlap.

Fully portable across new machines via:
- Dynamic AI Workspace discovery (--ai-workspace, AI_WORKSPACE_DIR, or .env)
- Explicit model overrides (--model-path or DIRECTOR_MODEL_PATH)
- Graceful standalone degradation when models are missing
"""

from __future__ import annotations

import argparse
import configparser
import logging
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Locate studio_addon root
BACKENDS_DIR = Path(__file__).resolve().parent
STUDIO_DIR = BACKENDS_DIR.parent
ROOT_DIR = STUDIO_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from studio_addon.config import (
        AI_WORKSPACE_DIR as DEFAULT_WORKSPACE,
        CPU_DIRECTOR_URL,
        DEFAULT_DIRECTOR_CTX,
        DIRECTOR_MODEL_PATH as DEFAULT_MODEL_PATH,
        resolve_ai_workspace,
    )
except ImportError:
    DEFAULT_WORKSPACE = Path(os.environ.get("AI_WORKSPACE_DIR", Path(__file__).resolve().parent.parent.parent))
    CPU_DIRECTOR_URL = "http://127.0.0.1:8081/v1"
    DEFAULT_MODEL_PATH = None
    DEFAULT_DIRECTOR_CTX = 65536
    def resolve_ai_workspace(ws=None):
        p = Path(ws or os.environ.get("AI_WORKSPACE_DIR", Path(__file__).resolve().parent.parent.parent))
        return p, p / "local_ai_launcher", sys.executable, None

logger = logging.getLogger("cpu_director")

DEFAULT_DIRECTOR_MODEL = "gemma-4-12b-director"
DEFAULT_DIRECTOR_PORT = 8081


def is_port_open(port: int, host: str = "127.0.0.1") -> bool:
    """Check if TCP port is currently listening."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex((host, port)) == 0


def find_llama_server(workspace_dir: Path) -> Optional[Path]:
    """Finds llama-server executable in workspace or system PATH."""
    candidates = [
        workspace_dir / "llama-cpp" / "llama-server.exe",
        workspace_dir / "llama-cpp" / "llama-server",
        workspace_dir / "bin" / "llama-server.exe",
        workspace_dir / "bin" / "llama-server",
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            return c

    in_path = shutil.which("llama-server") or shutil.which("llama-server.exe")
    if in_path:
        return Path(in_path)

    return None


def resolve_director_model_paths(
    model_name: str = DEFAULT_DIRECTOR_MODEL,
    workspace_dir: Optional[Path] = None,
    explicit_model_path: Optional[str | Path] = None,
) -> Tuple[Optional[Path], Optional[Path], int]:
    """Resolves the model, mmproj path, and context size dynamically."""
    ws = workspace_dir or DEFAULT_WORKSPACE

    # 1. Direct explicit override
    if explicit_model_path:
        p = Path(explicit_model_path).resolve()
        if p.exists():
            # Check for sibling mmproj
            mm_candidates = list(p.parent.glob("mmproj*.gguf"))
            mm_proj = mm_candidates[0] if mm_candidates else None
            return p, mm_proj, DEFAULT_DIRECTOR_CTX

    # 2. Check environment variable override
    env_path = os.environ.get("DIRECTOR_MODEL_PATH") or DEFAULT_MODEL_PATH
    if env_path:
        p = Path(env_path).resolve()
        if p.exists():
            mm_candidates = list(p.parent.glob("mmproj*.gguf"))
            mm_proj = mm_candidates[0] if mm_candidates else None
            return p, mm_proj, DEFAULT_DIRECTOR_CTX

    # 3. Consult models.ini if available in workspace
    models_ini_candidates = [
        ws / "local_ai_launcher" / "models.ini",
        ws / "models.ini",
        ws / "config" / "models.ini",
        STUDIO_DIR / "models.ini",
    ]
    models_ini = next((p for p in models_ini_candidates if p.exists()), None)
    target_sec = None
    if models_ini:
        cp = configparser.ConfigParser()
        try:
            cp.read(models_ini, encoding="utf-8")
            for sec in cp.sections():
                if sec.lower() == model_name.lower():
                    target_sec = sec
                    break
            if not target_sec:
                for sec in cp.sections():
                    aliases = [a.strip().lower() for a in cp[sec].get("alias", "").split(",") if a.strip()]
                    if model_name.lower() in aliases:
                        target_sec = sec
                        break
            if not target_sec and any(k in model_name.lower() for k in ("director", "cpu", "gemma")):
                for sec in cp.sections():
                    if "director" in sec.lower():
                        target_sec = sec
                        break
        except Exception:
            pass

    if target_sec:
        opts = dict(cp[target_sec])
        model_rel = opts.get("model", "")
        mmproj_rel = opts.get("mmproj", "")
        ctx = max(int(opts.get("ctx-size", DEFAULT_DIRECTOR_CTX)), DEFAULT_DIRECTOR_CTX)
    else:
        model_rel = "models/llama-server/HauhauCS/Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced/Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf"
        mmproj_rel = "models/llama-server/HauhauCS/Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced/mmproj-Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced-BF16.gguf"
        ctx = DEFAULT_DIRECTOR_CTX

    def _resolve_candidate(rel: str) -> Optional[Path]:
        if not rel:
            return None
        candidate = Path(rel)
        if candidate.is_absolute() and candidate.exists():
            return candidate
        for base in (ws, ws.parent, STUDIO_DIR, ROOT_DIR):
            cand = base / rel
            if cand.exists():
                return cand.resolve()
        return (ws / rel).resolve()

    model_path = _resolve_candidate(model_rel) if model_rel else None
    mmproj_path = _resolve_candidate(mmproj_rel) if mmproj_rel else None
    return model_path, mmproj_path, ctx


def build_cpu_director_cmd(
    model_name: str = DEFAULT_DIRECTOR_MODEL,
    port: int = DEFAULT_DIRECTOR_PORT,
    threads: int = 4,
    ctx_size: Optional[int] = DEFAULT_DIRECTOR_CTX,
    workspace_dir: Optional[Path] = None,
    explicit_model_path: Optional[str | Path] = None,
) -> Optional[List[str]]:
    """Constructs the llama-server launch command line pinned strictly to CPU (-ngl 0)."""
    ws = workspace_dir or DEFAULT_WORKSPACE
    exe_path = find_llama_server(ws)
    if not exe_path:
        return None

    model_path, mmproj_path, default_ctx = resolve_director_model_paths(
        model_name, workspace_dir=ws, explicit_model_path=explicit_model_path
    )
    if not model_path or not model_path.exists():
        return None

    target_ctx = ctx_size if ctx_size is not None else default_ctx

    cmd = [
        str(exe_path),
        "-m", str(model_path),
        "--port", str(port),
        "-ngl", "0",
        "-t", str(threads),
        "-c", str(target_ctx),
        "--jinja",
        "--reasoning", "off",
        "--temp", "0.8",
        "--top-p", "0.95",
        "--top-k", "64",
        "--min-p", "0.05",
        "--host", "127.0.0.1",
        "--no-webui",
    ]
    if mmproj_path and mmproj_path.exists():
        cmd.extend(["--mmproj", str(mmproj_path)])

    return cmd


def sync_to_localai_workspace(workspace_dir: Path) -> None:
    """Ensures external local_ai_launcher/backends directory receives updated copy if present."""
    target_dir = workspace_dir / "local_ai_launcher" / "backends"
    if target_dir.exists() and target_dir.is_dir():
        try:
            target_file = target_dir / "cpu_director.py"
            shutil.copy2(__file__, target_file)
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Portable CPU-only Media Production Director Server")
    parser.add_argument("--port", type=int, default=DEFAULT_DIRECTOR_PORT, help="Port to bind (default: 8081)")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads (default: 4)")
    parser.add_argument("--model", type=str, default=DEFAULT_DIRECTOR_MODEL, help="Model preset from models.ini")
    parser.add_argument("--model-path", type=str, default=None, help="Explicit path to any GGUF model file")
    parser.add_argument("--ctx", type=int, default=DEFAULT_DIRECTOR_CTX, help=f"Context size (default: {DEFAULT_DIRECTOR_CTX})")
    parser.add_argument("--ai-workspace", type=str, default=None, help="Path to AI workspace")
    args = parser.parse_args()

    # 1. Resolve workspace
    ws_dir, localai_dir, py_exe, _ = resolve_ai_workspace(args.ai_workspace)
    sync_to_localai_workspace(ws_dir)

    # 2. Check port collision
    if is_port_open(args.port):
        print(f"[CPU Director] Server is already running on port {args.port} (reusing instance).")
        sys.exit(0)

    # 3. Locate llama-server
    exe_path = find_llama_server(ws_dir)
    if not exe_path:
        print(f"[CPU Director] Notice: 'llama-server' executable not found in workspace ({ws_dir}) or system PATH.")
        print("              CPU Director server will not run, but Studio Addon will operate normally")
        print("              using built-in zero-failure prompt compilers and lexical matchers.")
        sys.exit(0)

    # 4. Locate model weights
    model_path, mmproj_path, ctx = resolve_director_model_paths(
        args.model, workspace_dir=ws_dir, explicit_model_path=args.model_path
    )
    if not model_path or not model_path.exists():
        print(f"[CPU Director] Notice: No local model weights found for CPU Director.")
        print(f"              Checked target path: {model_path}")
        print("              To run local Gemma 4 Director on CPU, specify your GGUF path:")
        print("                Option 1: In .env set DIRECTOR_MODEL_PATH=<path_to_gguf>")
        print(f"                Option 2: Place model in {ws_dir / 'models' / 'llama-server'}")
        print("                Option 3: Pass --model-path <path_to_gguf>")
        print("              Studio Addon will continue seamlessly with deterministic compilers.")
        sys.exit(0)

    # 5. Build and launch command
    cmd = build_cpu_director_cmd(
        model_name=args.model,
        port=args.port,
        threads=args.threads,
        ctx_size=args.ctx,
        workspace_dir=ws_dir,
        explicit_model_path=args.model_path,
    )
    print("=" * 75)
    print(f"  LAUNCHING CPU DIRECTOR (PORT {args.port}) - 0 MB GPU VRAM")
    print("=" * 75)
    print(f"  - Host Binary : {exe_path.name}")
    print(f"  - Model File  : {model_path.name}")
    print(f"  - Context Size: {args.ctx} tokens")
    print(f"  - CPU Threads : {args.threads}")
    print(f"  - Workspace   : {ws_dir}")
    print("=" * 75)

    proc = subprocess.Popen(cmd, cwd=str(ws_dir))
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()


if __name__ == "__main__":
    main()
