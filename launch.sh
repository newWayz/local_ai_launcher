#!/usr/bin/env bash
# =============================================================================
# Local AI Launcher (POSIX Entrypoint for Linux, macOS, and WSL)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}"

# Locate Python binary: prefer workspace .venv if present, otherwise system python3
if [ -f "${SCRIPT_DIR}/.venv/bin/python" ]; then
    PYTHON_BIN="${SCRIPT_DIR}/.venv/bin/python"
elif [ -f "${SCRIPT_DIR}/.venv/Scripts/python.exe" ]; then
    PYTHON_BIN="${SCRIPT_DIR}/.venv/Scripts/python.exe"
else
    PYTHON_BIN="python3"
fi

exec "${PYTHON_BIN}" -m local_ai_launcher "$@"
