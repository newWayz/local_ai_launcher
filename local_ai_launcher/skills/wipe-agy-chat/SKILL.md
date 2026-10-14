---
name: wipe-agy-chat
description: Wipes Antigravity CLI (agy) chat history, previous conversation sessions, brain logs, transcripts, and command logs. Activate when the user asks to wipe, clear, or reset agy chat history or mentions wipe-agy-chat.
---

# Wipe Antigravity Chat (wipe-agy-chat)

This skill provides maintenance automation to wipe Antigravity CLI (`agy`) conversation databases, brain traces, summaries, and history.

## Script Location
The PowerShell maintenance script is self-contained within this skill at:
`scripts/wipe-agy-chat.ps1`

## Usage
- **From within an active session** (preserves the current session so the running CLI is not terminated):
  ```powershell
  powershell -ExecutionPolicy Bypass -File "%~dp0scripts\wipe-agy-chat.ps1" -KeepCurrent
  ```
  Or from workspace root:
  ```powershell
  powershell -ExecutionPolicy Bypass -File .agents\skills\wipe-agy-chat\scripts\wipe-agy-chat.ps1 -KeepCurrent
  ```

- **Outside an active session** (complete wipe of all conversation DBs, summaries, brain logs, and history):
  ```powershell
  powershell -ExecutionPolicy Bypass -File .agents\skills\wipe-agy-chat\scripts\wipe-agy-chat.ps1
  ```
