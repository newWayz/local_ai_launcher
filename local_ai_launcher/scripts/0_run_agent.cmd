@echo off
set "SCRIPT_DIR=%~dp0"
set "WORKSPACE_DIR=%SCRIPT_DIR%..\..\"
set "PROJECTS_DIR=%WORKSPACE_DIR%projects"
pushd "%WORKSPACE_DIR%" >nul 2>&1
WHERE wt.exe >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    start "" wt.exe -w "AiLauncher_860a1b27" nt -d "%PROJECTS_DIR%" --title "Local AI Cockpit" cmd.exe /c "cd /d "%WORKSPACE_DIR%" && "%WORKSPACE_DIR%local_ai_launcher\1_launch.cmd" --unified"
) else (
    cd /d "%WORKSPACE_DIR%"
    "%WORKSPACE_DIR%local_ai_launcher\1_launch.cmd"
)
popd >nul 2>&1
