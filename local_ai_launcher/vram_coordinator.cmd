@echo off
set "SCRIPT_DIR=%~dp0"
set "WORKSPACE_DIR=%SCRIPT_DIR%..\"
set "PYTHONPATH=%WORKSPACE_DIR%;%PYTHONPATH%"
if exist "%WORKSPACE_DIR%.venv\Scripts\python.exe" (
    "%WORKSPACE_DIR%.venv\Scripts\python.exe" -m local_ai_launcher.vram_coordinator %*
) else (
    python -m local_ai_launcher.vram_coordinator %*
)
