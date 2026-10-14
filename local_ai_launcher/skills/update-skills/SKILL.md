---
name: update-skills
description: Pulls latest updates from upstream Git for workspace skill suites (gstack, hyperframes, remotion, superpowers, etc.), maintaining clean nested routing and low token overhead. Activate when the user asks to "update skills", "upgrade tools", "sync skills", or mentions update-skills.
---

# Update Workspace Skills (update-skills)

This skill provides fast, automated updates for all vendor skill suites tracked in `C:\aiWorkspace\.agents\vendor\`.

## What This Does
1. Checks and fast-forward pulls upstream Git changes for every installed suite (`gstack`, `hyperframes`, `remotion`, `superpowers`).
2. Runs suite-specific post-update setup hooks (e.g. `setup` via Git Bash).
3. Preserves clean nested routing so prompt context remains under 2,000 tokens (preventing flat clutter).
4. Clears stale harness caches (such as Hermes's `.skills_prompt_snapshot.json`).

## Usage

Run the workspace updater via PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File C:\aiWorkspace\.agents\update-skills.ps1
```

To perform a dry-run check without pulling changes:

```powershell
powershell -ExecutionPolicy Bypass -File C:\aiWorkspace\.agents\update-skills.ps1 -CheckOnly
```

Or invoke via `local_ai_launcher`:

```bash
python -m local_ai_launcher --update-skills
```
