---
name: herdr
description: Operate Herdr terminal multiplexer, session manager, and multi-agent coordination. Inspect workspaces, tabs, panes, and manage concurrent coding agents.
---

# Herdr — Terminal Multiplexer & Multi-Agent Operations Layer

Herdr is an agent-forward terminal runtime and session manager built specifically for managing and running multiple AI coding agents (Claude Code, Pi, Hermes, OpenCode, Trae, etc.) concurrently.

## Safety Guardrail: HERDR_ENV

Before running any command to inspect, split, or control Herdr panes from the terminal, always verify that the environment variable `HERDR_ENV` is set:

```bash
test "${HERDR_ENV:-}" = 1
```

If `HERDR_ENV` is not `1`, you are running in an ordinary terminal session outside Herdr. Halt execution and notify the user.

## Core Capabilities

1. **Agent State Awareness**:
   Herdr displays the live status of each agent (working, blocked, idle, done) in the sidebar.
   Agents can report their status to Herdr using:
   ```bash
   "$HERDR_BIN_PATH" pane report-agent
   ```

2. **Pane & Workspace Inspection**:
   Inspect active sessions, tabs, and panes:
   ```bash
   herdr status
   herdr pane list
   ```

3. **Background Persistence**:
   Sessions run inside a background server daemon. Terminals and agent tasks survive window closes and reconnections.
   To reload configuration safely without stopping active panes:
   ```bash
   herdr server reload-config
   ```

4. **Multi-Agent Coordination**:
   Launch auxiliary worker agents or test runners in sibling panes without stealing focus from the primary agent:
   ```bash
   herdr pane split
   ```
