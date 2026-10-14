<#
.SYNOPSIS
    Wipes Antigravity CLI (agy) chat history, conversations, brain logs, and cache.
.PARAMETER KeepCurrent
    If specified, preserves the currently active session so an active CLI instance is not killed.
#>
param(
    [switch]$KeepCurrent
)

$cliDir = "C:\aiWorkspace\.gemini\antigravity-cli"

Write-Host "===================================================" -ForegroundColor Cyan
Write-Host "             Antigravity Chat Wiper                " -ForegroundColor Cyan
Write-Host "===================================================" -ForegroundColor Cyan

$activeId = $null
if ($KeepCurrent) {
    $locks = Get-ChildItem -Path "$cliDir\presence\*.lock" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending
    if ($locks) {
        $activeId = $locks[0].BaseName
        Write-Host "Preserving active session: $activeId" -ForegroundColor Yellow
    }
}

Write-Host "Wiping conversations..."
Get-ChildItem -Path "$cliDir\conversations" -File -ErrorAction SilentlyContinue | ForEach-Object {
    if (-not $activeId -or -not $_.Name.StartsWith($activeId)) {
        Remove-Item -Path $_.FullName -Force -ErrorAction SilentlyContinue
    }
}

Write-Host "Wiping brain logs and transcripts..."
Get-ChildItem -Path "$cliDir\brain" -Directory -ErrorAction SilentlyContinue | ForEach-Object {
    if (-not $activeId -or $_.Name -ne $activeId) {
        Remove-Item -Path $_.FullName -Recurse -Force -ErrorAction SilentlyContinue
    }
}

Write-Host "Wiping annotations..."
Get-ChildItem -Path "$cliDir\annotations" -File -ErrorAction SilentlyContinue | ForEach-Object {
    if (-not $activeId -or -not $_.Name.StartsWith($activeId)) {
        Remove-Item -Path $_.FullName -Force -ErrorAction SilentlyContinue
    }
}

Write-Host "Wiping presence tokens..."
Get-ChildItem -Path "$cliDir\presence" -File -ErrorAction SilentlyContinue | ForEach-Object {
    if (-not $activeId -or -not $_.Name.StartsWith($activeId)) {
        Remove-Item -Path $_.FullName -Force -ErrorAction SilentlyContinue
    }
}

Write-Host "Cleaning conversation summaries..."
$summariesDb = "$cliDir\conversation_summaries.db"
if (Test-Path $summariesDb) {
    if ($activeId) {
        python -c "import sqlite3; conn = sqlite3.connect(r'$summariesDb'); conn.execute('DELETE FROM conversation_summaries WHERE conversation_id != ?', ('$activeId',)); conn.commit(); conn.close()" 2>$null
    } else {
        Remove-Item -Path "$cliDir\conversation_summaries.db*" -Force -ErrorAction SilentlyContinue
    }
}

Write-Host "Clearing command history..."
$historyFile = "$cliDir\history.jsonl"
if (Test-Path $historyFile) {
    if ($activeId) {
        (Get-Content -Path $historyFile | Where-Object { $_ -match $activeId }) | Set-Content -Path $historyFile -Encoding UTF8
    } else {
        Clear-Content -Path $historyFile -ErrorAction SilentlyContinue
    }
}

Write-Host "`nComplete wipe accomplished. Antigravity chat history cleared." -ForegroundColor Green
Write-Host "===================================================" -ForegroundColor Cyan
