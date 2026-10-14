import os
import sys
import json
import socket
import subprocess
import urllib.request
from pathlib import Path

try:
    from local_ai_launcher.config import AI_WORKSPACE_DIR
    WORKSPACE_DIR = AI_WORKSPACE_DIR
except Exception:
    WORKSPACE_DIR = Path(os.environ.get("AI_WORKSPACE_DIR", Path(__file__).resolve().parents[3]))
OPENLUMARA_DIR = WORKSPACE_DIR / "openlumara"
OPENLUMARA_PYTHON = OPENLUMARA_DIR / "venv" / "Scripts" / "python.exe"


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """Quick socket check to see if a port is bound."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.2)
            return s.connect_ex((host, port)) == 0
    except Exception:
        return False


def resolve_llm_endpoint() -> str:
    """Resolve the active OpenAI-compatible endpoint."""
    env_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
    if env_url:
        return env_url

    # Check OmniRoute proxy first (port 20128)
    if is_port_in_use(20128):
        return "http://localhost:20128/v1"

    # Check LocalAI native server (config.localai_port, default 8080)
    try:
        from local_ai_launcher.config import config
        localai_port = config.localai_port
    except Exception:
        localai_port = 8080

    if is_port_in_use(localai_port):
        return f"http://localhost:{localai_port}/v1"

    # Fallback to direct llama-server / backend (port 9931) if running
    if is_port_in_use(9931):
        return "http://localhost:9931/v1"

    return f"http://localhost:{localai_port}/v1"


def resolve_active_model(llm_url: str) -> str:
    """Resolve active model name from environment, live endpoint, or config."""
    model = os.environ.get("OPENAI_MODEL") or os.environ.get("MODEL")
    if model and model not in ("default", "*"):
        return model

    # Query live endpoint to discover active model loaded in local engine
    try:
        req = urllib.request.Request(f"{llm_url}/models", headers={"User-Agent": "OpenLumara-Shim"})
        with urllib.request.urlopen(req, timeout=1.0) as resp:
            m_data = json.loads(resp.read().decode("utf-8"))
            if "data" in m_data and len(m_data["data"]) > 0:
                discovered = m_data["data"][0].get("id")
                if discovered:
                    return discovered
    except Exception:
        pass

    # Try reading from local_ai_launcher config if available
    try:
        from local_ai_launcher.config import config
        return config.default_model
    except Exception:
        pass

    return "default"


def main():
    if not OPENLUMARA_DIR.exists():
        print(f"[ERROR] OpenLumara directory not found at {OPENLUMARA_DIR}", file=sys.stderr)
        sys.exit(1)

    python_exe = OPENLUMARA_PYTHON if OPENLUMARA_PYTHON.exists() else Path(sys.executable)
    main_script = OPENLUMARA_DIR / "main.py"
    config_file = OPENLUMARA_DIR / "config.yml"

    llm_url = resolve_llm_endpoint()
    model_name = resolve_active_model(llm_url)

    # Automatically adapt and inject on-disk config.yml before startup (conforming to local engine)
    try:
        from local_ai_launcher.bridges.orca_openlumara.configurator import setup_openlumara_config
        setup_openlumara_config(
            lumara_dir=OPENLUMARA_DIR,
            llm_base_url=llm_url,
            active_model=model_name,
            mode="cli"
        )
    except Exception as cfg_err:
        print(f"[WARN] OpenLumara pre-launch config sync notice: {cfg_err}", file=sys.stderr)

    incoming_args = list(sys.argv[1:])

    has_cli = "--cli" in incoming_args
    has_help = any(arg in incoming_args for arg in ("-h", "--help"))
    has_config = "--config" in incoming_args
    has_disable_installer = "--disable_auto_installer" in incoming_args

    augmented_args = []
    if not has_cli and not has_help:
        augmented_args.append("--cli")

    if not has_disable_installer and not has_help:
        augmented_args.append("--disable_auto_installer")

    if not has_config and config_file.exists() and not has_help:
        augmented_args.extend(["--config", str(config_file)])

    # Add API endpoint if not explicitly provided
    if not any(arg.startswith("--api.url") for arg in incoming_args):
        augmented_args.extend(["--api.url", llm_url])

    # Add API key if not provided
    if not any(arg.startswith("--api.key") for arg in incoming_args):
        api_key = os.environ.get("OPENAI_API_KEY", "dummy")
        augmented_args.extend(["--api.key", api_key])

    # Add model name if not provided
    if not any(arg.startswith("--model.name") for arg in incoming_args):
        augmented_args.extend(["--model.name", model_name])

    final_args = [str(python_exe), str(main_script)] + incoming_args + augmented_args

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    env["OPENAI_BASE_URL"] = llm_url
    env["OPENAI_API_BASE"] = llm_url
    env["OPENAI_MODEL"] = model_name
    env["MODEL"] = model_name

    cwd = Path.cwd()
    try:
        ret = subprocess.call(final_args, cwd=str(cwd), env=env)
        sys.exit(ret)
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
