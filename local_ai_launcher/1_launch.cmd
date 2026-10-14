@ECHO off
set "ROOT=%~dp0.."
pushd "%ROOT%" >nul 2>&1
if "%1"=="--unified" goto :run
if "%AI_LAUNCHER_UNIFIED%"=="1" goto :run
WHERE wt.exe >nul 2>&1
IF %ERRORLEVEL% EQU 0 (
    set "AI_LAUNCHER_UNIFIED=1"
    set "AI_LAUNCHER_WINDOW_NAME=AiLauncher_860a1b27"
    start "" wt.exe -w "AiLauncher_860a1b27" nt -d "%ROOT%" --title "Local AI Cockpit" cmd.exe /c "cd /d "%ROOT%" && "%~f0" --unified %*"
    popd >nul 2>&1
    exit /b
)
:run
SETLOCAL EnableDelayedExpansion
set "PYTHONPATH=%ROOT%"
if exist "%ROOT%\.venv\Scripts\python.exe" (
    "%ROOT%\.venv\Scripts\python.exe" -m local_ai_launcher %*
) else (
    python -m local_ai_launcher %*
)
pause
popd >nul 2>&1
