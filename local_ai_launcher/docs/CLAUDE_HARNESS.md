# Claude Code / Claude Desktop Harness Family

> Code-verified 2026-10-14 against `harnesses/claude.py`, `harnesses/__init__.py`, `config.py`, `harness_model.py`, `bridges/claude_desktop_mcp.py`, `backends/litellm.py`.

The largest harness adapter family in the launcher: `harnesses/claude.py` (~2,100 lines) registers **30 keys** in `HARNESS_REGISTRY` — more than any other module.

## 1. Registry Keys

| Group | Keys | Class |
|---|---|---|
| Claude Code CLI | `claude`, `claude-code`, `claudecode` | `ClaudeAdapter` |
| Claude Code (offline gateway) | `claude-local`, `claude-code-local` | `ClaudeLocalAdapter` |
| Claude Code (Anthropic cloud) | `claude-official`, `claude-code-official` | `ClaudeOfficialAdapter` — **online-only** |
| Claude Desktop | `claude-desktop`, `claudedesktop` | `ClaudeDesktopAdapter` |
| Desktop official GUI + MCP | `claude-desktop-official`, `claude-desktop-mcp` | `ClaudeDesktopOfficialAdapter` |
| Desktop direct-local gateway | `claude-desktop-gateway`, `claude-desktop-local` | `ClaudeDesktopGatewayAdapter` |

`ONLINE_ONLY_HARNESSES` (`harnesses/__init__.py`) contains `claude-official` and `claude-code-official` (plus the Antigravity family).

## 2. Routing Modes

`--claude-mode {local | official | local_claude_native}` (sets `config.claude_mode`; `--claude-desktop-mode {mcp | gateway | disabled | off}` for Desktop):

| Mode | Path | Translation needed? |
|---|---|---|
| `local` | Claude CLI → gateway **8080** OpenAI endpoints | Yes — Anthropic↔OpenAI by the **LiteLLM proxy (Port 4000)** or gateway-side `/v1/messages` shim |
| `official` | Claude CLI → Anthropic cloud | n/a (online-only) |
| `local_claude_native` | Claude CLI → gateway **native Anthropic Messages API** (`/v1/messages`, `/v1/messages/count_tokens`) | **No** — engine speaks Claude protocol directly |

Native mode membership is governed by `config.local_claude_native_engines` (default `["strata"]`, expandable via `--local-claude-native-engine(s)`) and the capability registry in `harness_model.py` (`is_local_claude_native_engine`). When a Strata (or other native) model is active, `is_local_claude_native_active()` in `harnesses/claude.py` short-circuits all translation bridges.

Strata-specific native options: `local_claude_native_use_custom_chat_template`, `local_claude_native_chat_template_file` (mirrors of the `strata_*` settings, `config.py`).

## 3. Claude Desktop

- **`mcp` mode**: official Anthropic GUI kept, local model reached through the zero-dependency stdio MCP bridge `bridges/claude_desktop_mcp.py` (`query_local_model` → gateway 8080).
- **`gateway` mode**: developer-mode direct local endpoint (no cloud GUI expectations).
- **`disabled` / `off`**: adapter skipped.

## 4. Offline Answer Contract

The local profile's isolated `CLAUDE.md` is seeded with the *Offline Answer Contract* persona (`OFFLINE_PERSONA_CONTRACT_MD` in `harnesses/claude.py`) to suppress model-side content deflection on the offline box. The contract text is shared with the chat template `models/chat_template_qwen38_sharp_v22.4.0_medium-convergent.jinja` and both copies are fingerprint-pinned by `tests/test_claude_searxng_offline_policy.py` — edit both or neither.

## 5. Ports Involved

| Port | Role |
|---|---|
| 8080 | Unified gateway (`/v1/messages` native, `/v1/chat/completions` translated) |
| 4000 | LiteLLM Anthropic↔OpenAI proxy (`backends/litellm.py`) for non-native engines |
| 9932 | MCP tool proxy (Antigravity/Claude Desktop tooling) |
