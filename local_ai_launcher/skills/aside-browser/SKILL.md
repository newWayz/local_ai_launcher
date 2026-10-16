---
name: aside-browser
description: Interactive web browsing, automated web tasks, logged-in session workflows, Playwright JS DOM actions, and visual problem-solving via Aside Browser connected to the local AI engine.
---

# Aside Browser — Interactive Web Navigation & Problem-Solving

Aside (`aside.com`) is a Chromium-based AI browser pre-connected to the Local AI Engine (`http://127.0.0.1:8080/v1`).
Use this skill whenever a task requires interactive web browsing, authenticated dashboard actions, form filling, SPA DOM navigation, or visual problem-solving beyond static search engines.

## When to Use

1. **Interactive Web Navigation & Actions**:
   - Clicking buttons, filling text forms, handling dropdowns, toggling checkboxes, solving cookie/modal dialogues.
   - Multi-step navigation across single-page applications (SPAs) requiring JavaScript execution.
2. **Authenticated Workflows**:
   - Accessing user-authenticated web services (Slack, Gmail, GitHub, AWS, Datadog, Jira, Linear, internal developer portals).
   - Performing tasks without exposing or asking for plaintext user passwords.
3. **Deterministic DOM & Scripting (Playwright JS)**:
   - Extracting structured data from rendered pages, capturing screenshots, inspecting network requests or console logs.
4. **User Context & Personal Memory**:
   - Querying Aside's local memory (`aside memory search "<query>"`) for user preferences, previous research, and site history.

---

## Invocation Interfaces

### 1. Model Context Protocol (MCP) Tools
When running inside harnesses with MCP support (Pi, Claude, OpenCode), the `aside` MCP server exposes:

* **`aside__exec`** (or `exec`):
  Spawns an autonomous browser agent task in Aside.
  ```json
  {
    "prompt": "Open http://localhost:3000, verify the login form is visible, and return the page title"
  }
  ```
* **`aside__repl`** (or `repl`):
  Executes Playwright-compatible JavaScript directly on open browser tabs:
  ```javascript
  const page = await openTab('https://news.ycombinator.com');
  const titles = await page.$$eval('.titleline > a', els => els.slice(0, 5).map(e => e.textContent));
  console.log(JSON.stringify(titles));
  ```
* **`aside__memory_search`**:
  Queries distilled local markdown memory files.

---

### 2. Command Line Interface (`aside` CLI)
Available directly in PowerShell and Terminal across all harnesses (`pi`, `dsh`, `claude`, `hermes`, `opencode`):

#### Run a Browser Task
```powershell
aside "Open localhost:3000 and run smoke verification"
aside "Go to GitHub notifications and summarize any failed CI runs"
```

#### Playwright-Compatible REPL
```powershell
aside repl "const p = await openTab('https://example.com'); console.log(await p.title())"
```

#### Session Steering & Continuation
```powershell
aside session list
aside session resume <session-id> "Click the second link and verify the download"
aside session steer <session-id> "Switch to checking the staging server instead"
aside session stop <session-id>
```

#### Query Personal Memory
```powershell
aside memory search "OAuth debugging" --json
```

---

## Local AI Engine Integration

Aside Browser is automatically pre-configured by `local_ai_launcher` to route model inferences to the unified local gateway:
- **Base URL**: `http://127.0.0.1:8080/v1`
- **Provider**: `local` (OpenAI-compatible)
- **Model**: Matches the active local engine preset (e.g. `local-engine`)
- **Privacy & Safety**: All browser reasoning, element targeting, and DOM planning stay strictly local on host DDR5/GPU hardware with 0 cloud token leakage.
