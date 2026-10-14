@echo off
set "SCRIPT_DIR=%~dp0"
set "PROJECTS_DIR=%SCRIPT_DIR%..\..\projects"
agy --add-dir "%PROJECTS_DIR%" --dangerously-skip-permissions
