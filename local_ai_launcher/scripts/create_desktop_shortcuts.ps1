# Generates dynamic desktop shortcuts for Windows File Explorer / Desktop
$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$workspaceDir = (Resolve-Path "$scriptDir\..\..").Path
$desktop = [Environment]::GetFolderPath("Desktop")

$wsh = New-Object -ComObject WScript.Shell

# 1. Local AI Cockpit Shortcut
$cockpitShortcut = $wsh.CreateShortcut("$desktop\Local AI Cockpit.lnk")
$cockpitShortcut.TargetPath = "$workspaceDir\local_ai_launcher\scripts\0_run_agent.cmd"
$cockpitShortcut.WorkingDirectory = "$workspaceDir\projects"
$cockpitShortcut.Description = "Launch Local AI Serving & Coding Agent Cockpit"
$cockpitShortcut.Save()

# 2. Antigravity CLI Shortcut
$agyShortcut = $wsh.CreateShortcut("$desktop\Start Antigravity CLI.lnk")
$agyShortcut.TargetPath = "$workspaceDir\local_ai_launcher\scripts\4_start_agy_cli.cmd"
$agyShortcut.WorkingDirectory = "$workspaceDir\projects"
$agyShortcut.Description = "Start Google Antigravity CLI in Projects Directory"
$agyShortcut.Save()

Write-Host "[SUCCESS] Created desktop shortcuts: 'Local AI Cockpit' and 'Start Antigravity CLI' on $desktop" -ForegroundColor Green
