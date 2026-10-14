@echo off
set "SCRIPT_DIR=%~dp0"
set "WORKSPACE_DIR=%SCRIPT_DIR%..\..\"
if exist "%WORKSPACE_DIR%.venv\Scripts\python.exe" (
    "%WORKSPACE_DIR%.venv\Scripts\python.exe" -m local_ai_launcher.shims.blackwell_llama_shim %*
) else (
    python -m local_ai_launcher.shims.blackwell_llama_shim %*
)
