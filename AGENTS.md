# Workspace Directives & Path Conventions

## 1. Environment & Workspace Layout
- **Operating System**: Windows 11 (Shell: PowerShell 7 / Windows Terminal, UTF-8).
- **Workspace Root**: `C:/aiWorkspace`
  - Core launcher & runtime engines: `C:/aiWorkspace/local_ai_launcher/` (See `C:/aiWorkspace/local_ai_launcher/AGENTS.md` for daemon topology, port allocations, and engine guardrails)
  - Full test suite: `C:/aiWorkspace/local_ai_launcher/tests/`
  - Workspace configurations & prompts: `C:/aiWorkspace/config/`
  - Active development subprojects: `C:/aiWorkspace/projects/`
  - Pi harness configuration & state: `C:/aiWorkspace/.pi/` (Agent dir: `C:/aiWorkspace/.pi/agent/`)
    - Shims & extensions: Tracked in `local_ai_launcher/shims/` (`loop_recovery.ts`, `cpu_tools.ts`, `auto_cycle.ts`), auto-provisioned to `.pi/agent/extensions/`.
    - Extension compatibility: Host-provided packages (`@sinclair/typebox`) must use `peerDependencies: {"*"}`, and `builtin:mcp` is auto-suppressed via `"-builtin:mcp"` in `settings.json` when custom adapters (`roach-pi`) are configured.
    - Skills & Gateway: Shared workspace skills live in `C:/aiWorkspace/.agents/skills/` (Gateway) and are natively auto-discovered by Pi across ancestor directories. Skills must NEVER be duplicated into `.pi/skills/`. Only harness-specific customizations reside in `.pi/agent/skills/` (`clear-pi-history`, `ledger-discipline`, `senior-thinking-skills`).
  - Temporary workspace scratch: `C:/aiWorkspace/tmp/` (Windows has no `/tmp/`; use `C:/aiWorkspace/tmp/` when temporary files are needed)
  - Destructive commands: bash deletions/kills outside `tmp/` (`rm -rf`, `Remove-Item`, cleanup scripts) require user confirmation first — never run them as "verification tests"; prefer DRY-RUN defaults (e.g. `clear_pi_history.ps1` deletes only with `-Force`).
- **Current Working Directory (CWD)**:
  - When launched, Pi's `cwd` is typically `C:/aiWorkspace/projects`.
  - When you are in `projects/`, relative paths such as `local_ai_launcher/config.py` or `tests/` do NOT exist in `.`.
  - Always use workspace-absolute paths (`C:/aiWorkspace/...`) or parent-relative paths (`../...`) to access repository-wide files.

## 2. File Path Formatting & Tool Rules
- **Forward Slashes**: Always use forward slashes in file paths across all tools (`read`, `write`, `edit`, `find`, `grep`, `ls`):
  - Example: `C:/aiWorkspace/local_ai_launcher/config.py`
  - Forward slashes are natively resolved by Node.js and prevent JSON string escape errors (`\t`, `\n`, `\a`, etc.).
- **Native Tools Directly**:
  - Always use Pi's native tools (`read`, `find`, `grep`, `ls`, `edit`, `write`).
  - Do NOT call Unix commands like `realpath`, `sed`, or `cat` in PowerShell.
  - Do NOT attempt to copy files to Unix `/tmp/`.
- **Files vs Directories (No EISDIR)**:
  - Never pass a directory path (e.g. `local_ai_launcher`) to `read` or `edit`. Node.js will throw `EISDIR: illegal operation on a directory`.
  - Only pass files to `read` and `edit`. Use `ls` or `find` to explore directories first.

## 3. File Discovery Strategy
When finding files or resolving `ENOENT` (file not found):
1. **Locate by Pattern (`find`)**:
   - Explicitly specify `path: "C:/aiWorkspace"` to search the entire workspace:
     `find(pattern="*config*.py", path="C:/aiWorkspace")`
   - To search only within the current project directory:
     `find(pattern="*.py", path=".")`
2. **Search by Content (`grep`)**:
   - Run `grep` with `path: "C:/aiWorkspace"` across all files:
     `grep(pattern="LauncherConfig", path="C:/aiWorkspace")`
3. **Inspect Directory Contents (`ls`)**:
   - Check directory contents before assuming a path exists:
     `ls(path="C:/aiWorkspace")` or `ls(path="C:/aiWorkspace/projects")`
4. **Unicode / Special Character Paths**:
   - If a path provided in user messages contains fullwidth brackets or slashes (e.g. `「` `」` `【` `】` `／`), replace with ASCII equivalents (`[` `]` `[` `]` `/`) and call `read` directly.

## 4. Coding & Architecture Standards
- **Role**: Senior Software Architect & QA Engineer.
- **Python**: Python 3.11+, explicit type hints, `pytest` for verification, `uv` for dependency management.
- **Tests**: Always run `pytest` to verify changes before concluding tasks.
- **CPU Auxiliary Microservices**:
  - Discrete decisions / evaluations: `POST http://127.0.0.1:8080/v1/decide` (CPU ONNX Gater on Port 8083, ~15ms).
  - Code & document passage reranking: `POST http://127.0.0.1:8080/v1/rerank` (CPU Reranker on Port 8082, ~25ms).
  - **Pi Tools**: Use native `cpu_rerank` (query, documents, top_n) and `cpu_decide` (prompt, candidate_options) tools directly in Pi sessions.
  - **Hermes / Shell CLI**: Run `python -m local_ai_launcher.bridges.cpu_tools_cli rerank` or `decide` via the `cpu-tools` skill.
  - **Model Alias**: Discrete gating requests can target model `cpu-gater` or `open-jev-deberta-v3-large` without loading/evicting GPU models.
- **Launcher Engine Directives**: When modifying `local_ai_launcher/`, inspect `C:/aiWorkspace/local_ai_launcher/AGENTS.md` for daemon topology, port allocations, Blackwell preset protections, and sequential test execution requirements.
