# Documentation Update Analysis — `local_ai_launcher`

> **Date:** 2026-10-14
> **Purpose:** Audit of every project document vs. the current code, so another AI can execute the updates.
> **Method:** Static analysis only (launcher and pytest were NOT run — a launcher/test run triggers `taskkill`). The single executed command was `pytest --collect-only`, which collected **471 tests** in 1.9 s. Every claim below cites a file:line from the current tree.

---

## 0. Ground-truth facts (use these when rewriting any doc)

These were verified directly against code. Treat this section as the source of truth for the update pass.

| Fact | Verified value | Evidence |
|---|---|---|
| Package version | `2.0.0` | `__init__.py:6` |
| Test count | **471 collected** (27 test files + `conftest.py`) | `pytest --collect-only` output; `ls tests/` |
| Unified gateway | Port **8080**, native FastAPI (`localai/native_server.py`) | `config.py:699` |
| Gateway endpoints | `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/messages` + `/messages` (Anthropic native), `/v1/messages/count_tokens`, `/v1/models`, `/v1/rerank` + `/rerank`, `/v1/decide`, `/v1/decider/slots`, `/v1/decider/dom`, `/backend/shutdown`, `/v1/global/health`, WebUI at `/` and `/models` | `native_server.py:473–2654` (decorator lines listed in §1.6) |
| **No `/v1/embeddings` endpoint exists** | only an internal "tool action embeddings" pre-registration endpoint | `native_server.py:2654` (no `@app.post("/v1/embeddings")` anywhere) |
| CPU llama-server | **8081** — unified CPU worker; runs Decider-4B C++ inference backend AND Gemma-4 director workload | `config.py:961,1009–1017`, `AGENTS.md:10` |
| CPU reranker | **8082** (`/v1/rerank`, `/rerank`) | `config.py:974`, `cpu_reranker.py:300–301` |
| CPU gater | **8083** — default model is **`wfzyx/von`** (ModernBERT "System One"); `open-jev-deberta-v3-large` and `clm-8b` are **deprecated/replaced** | `cpu_onnx_gater.py:2–3,31–33`; `config.py:987,995,1002` |
| Gater microservice routes | `/v1/decide`, `/v1/systemone`, `/v1/tripwire`, `/v1/chat/completions`, `/v1/shortlist`, `/v1/tools/route`, `/v1/tools/register`, `/v1/cache/actions`, `/v1/clm/info`, `/health` | `cpu_onnx_gater.py:482–594` |
| Decider-4B | **8084** — `Mapika/decider-4b-v2.1`; routes `/v1/decider/slots`, `/v1/decider/dom`, `/v1/decider/policy_prior`, `/v1/decider/info`; raw inference proxied to 8081; **port 9935 retired** | `config.py:1005–1017`, `cpu_decider.py:492–584` |
| Signal daemon | 8085 | `config.py:802` |
| SearXNG metasearch | **8088** (portable; WSL2-Docker/Native/Remote tiers) | `config.py:1105–1128`, `backends/searxng.py` |
| LiteLLM proxy | **4000** (`LITELLM_PORT`) — Anthropic↔OpenAI translation proxy | `paths.py:64`, `config.py:936`, `backends/litellm.py` docstring |
| NPU speech-to-text | 8000 (Parakeet TDT V3) | `parakeet_npu.py:693` |
| OmniRoute proxy | 20128 | AGENTS.md:17 (unchanged) |
| Shim ports | 9940 blackwell, 9941 tabby, 9942 freetoken, 9943 strata; raw engines: llama-server 9931, freetoken 9933, strata 9934 | `config.py:721–725` |
| MCP server | 9932; T3 web 3773 | `config.py:896–897` |
| Harness registry | keys include: `hermes, pi, pie, opencode, agy, agy-ide, agy-desktop, antigravity, ls, deepseek, dsh, orca, t3, t3code, openlumara, lumara`, plus the whole **Claude family**: `claude, claude-local, claude-official, claude-code, claude-code-local, claude-code-official, claudecode, claude-desktop, claude-desktop-official, claude-desktop-mcp, claude-desktop-gateway, claude-desktop-local, claudedesktop` | `harnesses/__init__.py:58–94` |
| Online-only harnesses | `agy, agy-ide, agy-desktop, antigravity, claude-official, claude-code-official` | `harnesses/__init__.py:97–101` |
| Loop detector — code defaults | `ngram_size=8`, `ngram_threshold=3`, `cycle_similarity=0.82`, `min_sentence_len=35`, `max_history_words=1500` | `config.py:778–780,1064–1066`, `loop_detector.py:35–55` |
| Loop detector — ACTIVE tuned values (launcher.toml) | `ngram_size=12`, `ngram_threshold=5` ("do not go 3 or below"), `cycle_similarity=0.93`, `min_sentence_len=40` | `launcher.toml:138–142` |
| Loop code-block guard | inside code blocks the effective threshold is `ngram_threshold + 2` | `loop_detector.py:141–143` |
| GVS5H solver defaults | `--max-iters 6`, `--max-seconds 180`, `--test-timeout 10.0`, `--max-log-runs 100`, model/endpoint overridable via `GVS5H_MODEL`/`GVS5H_ENDPOINT` env | `gvs5h_solver.py:35–46,932–943` |
| Hardware profiles | defined in **`config.py:535`** (`HARDWARE_PROFILES: Dict[str, Dict[str, Any]]`), NOT in `presets.py`, and they are plain dicts, not a `HardwareProfile` dataclass. Members: `blackwell_16gb`, `ada_24gb`, `standard_16gb`, `budget_12gb` | `config.py:535–617` |
| Model presets | `models.ini` has **36 sections** (incl. new `tiel-coder-35b*`, `qwen-38_27b-huang-*` Huang adaptive-KV presets, `muse-glimmer-30B-q3-uncen*`, `gemma-4-12B-it-QAT-audio*`) | `models.ini` section grep |
| `localai/models/` | **empty at rest** — YAML configs are auto-generated at runtime (gitignored); "33 configs" is stale | `ls localai/models/` → 0 |
| Local overrides | `models.local.ini` merged with top priority by `presets.py` | `presets.py:31–35` |
| Clawback "light" evictions | Reranker 8082, ONNX Gater 8083, **Decider-4B 8084**, CPU Director 8081 — Parakeet is NOT evicted | `memory_clawback.py:45–80` |
| Git repo root | the **workspace** `C:/aiWorkspace` is the git root; remote `github.com/newWayz/local_ai_launcher.git`; `local_ai_launcher/` is whitelisted wholesale in the root `.gitignore`; `launcher.toml`/`launcher.json` ignored | `git rev-parse`, root `.gitignore:154–158` |
| CI | `.github/workflows/ci.yml` exists at workspace root | `ls ../.github/workflows` |
| `pyproject.toml` | lives at **workspace root** `C:/aiWorkspace/pyproject.toml`, not inside `local_ai_launcher/` | `ls` |

### Undocumented modules (exist in code, missing from every doc's tree)

| Module | What it is (from its own docstring) |
|---|---|
| `harness_model.py` | Harness/Engine characteristics registry & routing model (offline capability, meta-harness, preload priority) |
| `update_frequency.py` | Update-schedule manager; state in `config/update_state.json` |
| `prompts.py` | Terminal prompting & interactive helpers (re-exported by `config.py:66`) |
| `mcp_server.py` | Tool proxy MCP server (port 9932) — in port tables but not in any directory tree |
| `backends/llamacpp_router.py` | **Unified llama-server variant router**: Bruno fork / Huang fork / official upstream, priority decision engine |
| `backends/bruno_builder.py` | Bruno fork builder (`BrunoPPassini/llama.cpp`, branch `qwen38-blackwell-256k`) — same repo/branch as "blackwell" builder |
| `backends/huang_builder.py` | Huang fork builder (`RaymondHuang210129/llama.cpp-adaptive-kv-streaming`, branch `feature/adaptive-kv-stream-v2`) |
| `backends/ascii_pruner.py` | ASCII BPE-vocabulary pruner (14.25 GB → 12.25 GB) for Huang adaptive-KV runtime |
| `backends/cpu_decider.py` | Decider-4B FastAPI engine (8084) |
| `backends/cpu_director.py` | CPU media-production director (Gemma 4 VLM on 8081) |
| `backends/litellm.py` | LiteLLM Anthropic↔OpenAI proxy supervisor (port 4000) |
| `backends/searxng.py` | SearXNG portable metasearch supervisor (8088, 3 tiers) |
| `backends/localai.py` | Official LocalAI install/update supervisor |
| `bridges/safe_run.py` | Windows Job Object containment runner (`KILL_ON_JOB_CLOSE`) |
| `bridges/dom_navigator.py` | DOM/accessibility pruning bridge for Decider-4B (Mind2Web/MiniWoB++) |
| `bridges/searxng_mcp.py` | stdio MCP bridge → SearXNG |
| `bridges/claude_desktop_mcp.py` | stdio MCP bridge → Claude Desktop |
| `shims/bruno_llama_shim.py` + `.cmd`, `shims/huang_llama_shim.py` + `.cmd` | gRPC shims for Bruno and Huang llama forks |
| `shims/loop_recovery.ts`, `cpu_tools.ts`, `auto_cycle.ts` | Pi harness extensions (auto-provisioned to `.pi/agent/extensions/` per workspace AGENTS.md) |
| `vendor/von/` | vendored von backend |
| `skills/reliable-file-read/`, `skills/scoped/{pi,deepseek}/` | additional builtin skills + harness-scoped skill deployment (`skills_manager.py:212,236–243`) |
| `external_sources.toml` 11th source | `[skills.sources.rea]` (docs list only 10) |
| `strata_models.json(.example)`, `strata_settings.json(.example)`, `strata-qwen-38-flash-strata-iq3s.json` etc. | Strata pack/model settings files (`backends/strata.py:load_strata_settings`, `config.py:656`) |
| `hermes_system_prompt.txt`, `hermes_system_prompt_strata.txt`, `hermes_tools.json` | Hermes prompt/tool assets |

---

## 1. Per-document update plan

Priority scale: **P0** = factually wrong today and misleading to a reader/agent; **P1** = incomplete/stale inventory; **P2** = cosmetic/historical, annotate only.

### 1.1 `README.MD` — P0 (flagship doc, several wrong facts)

**Errors to fix:**
1. **§1 + §2 claim `/v1/embeddings` is exposed.** It is not — no such route exists in `native_server.py`. Replace with the real endpoint set: `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/messages` (+ `/messages`, `/v1/messages/count_tokens` — Anthropic Messages API, Strata-native), `/v1/models`, `/v1/rerank`, `/v1/decide`, `/v1/decider/*`.
2. **§2 port table is missing rows:** 8084 (Decider-4B), 8088 (SearXNG), 4000 (LiteLLM proxy). Add them (ground-truth table §0).
3. **§2 row 8083** still names "CPU CLM / Decision Gater — `open-jev-deberta-v3-large` or `clm-8b`". Default is now **`wfzyx/von`** (ModernBERT System One); both old models are deprecated (`cpu_onnx_gater.py:2–3,31`). Same error in **§5.2**, which even describes clm-8b as "High-Capacity default alternative" — delete/replace with von description and note the deprecation (see `docs/analysis-local_ai_launcher-skill_routers.md:37–39` for the replacement wording).
4. **§1 loop-detection numbers are wrong:** "N=16, threshold ≥4, p∈{1,2,3}, similarity ≥0.93". Code defaults are 8/3/0.82; the active tuned `launcher.toml` uses **12/5/0.93** with `min_sentence_len=40`. Rewrite: state defaults, state tuned values, and note the code-block +2 threshold rule (`loop_detector.py:141–143`). The 2026-10-09 tuning-rationale paragraph (3→4) is superseded by the launcher.toml comment ("default: 5, do not go 3 or below") — update to reflect 12-gram/5-repeat tuning.
5. **§3 CLI flags table is a small subset.** Add the missing flags from `__main__.py:45–74`: `--port`, `--start-dir`, `--no-verify`, `--build-ui`, `--pi-search-workflow`, `--pi-search-browser`, `--download-model`/`--no-download-model`, `--install-localai`/`--no-install-localai`, `--update-localai`/`--no-update-localai`, `--sync-skills`, `--update-skills`, `--force-update`, `--update-frequency-hrs`, `--update-git`/`--no-update-git`, `--claude-mode {local,official,local_claude_native}`, `--claude-desktop-mode {mcp,gateway,disabled,off}`, `--local-claude-native-engine`, `--local-claude-native-engines`.
6. **§3 harness list omits the Claude family.** Add Claude Code CLI + Claude Desktop adapters (`--harness claude`, `claude-local`, `claude-official`, `claude-desktop*` — see ground-truth registry) and the `dsh`/`t3code`/`lumara`/`antigravity` aliases. Claude is the single largest harness adapter (`harnesses/claude.py`, ~2100 lines) — its absence is the biggest doc gap in the project.
7. **§8 "All 223 tests pass" → 471 collected.** (Don't re-run pytest here; use the collect count. Keep the "pass" claim only if the implementing AI is allowed to run the suite elsewhere.)
8. **§1/§5 title "CPU Auxiliary Subsystem (Ports 8081–8083)"** → 8081–8085 (+8088). Add Decider-4B (8084) and SearXNG (8088) bullets.
9. **§7 component-architecture tree** is missing: `harness_model.py`, `update_frequency.py`, `prompts.py`, `mcp_server.py`, `memory_clawback.py`, `mnemosyne.py`, `presets.py`, `paths.py`, `process.py`, `environment.py`, `dispatcher.py` partially, and the new backends/bridges/shims listed in §0. Update it.
10. **§4 tip box** "loop_recovery_temp_floor default 0.15" — verified OK (`config.py:784–785` shows `loop_recovery_max_retries=3`, `temp_reduce=0.1`; confirm floor default before editing; the value 0.15 appears in ARCHITECTURE §3.8 — cross-check `config.py` `loop_recovery_temp_floor` and align both docs to one value).

**Additions:**
- Mention the **Strata + Claude-native Anthropic routing** (`--claude-mode local_claude_native`, engines supporting `/v1/messages` natively — `config.py:739–771`, `harness_model.py`).
- Mention **SearXNG offline web search** (8088 + `bridges/searxng_mcp.py`) and **LiteLLM proxy** (4000) in §1 capabilities.
- §2 table: note port **9935 is retired** in favor of 8081 (`config.py:1013`).

### 1.2 `docs/ARCHITECTURE.md` — P0 (most stale doc)

**Errors to fix:**
1. **Header:** "223/223 Tests Passing" and §4 transcript "collected 169 items / 169 passed in 18.13s" with the 17-file list → replace with 471 collected / 27 test files; list the 10 newer test files (`test_ascii_pruner`, `test_claude_searxng_offline_policy`, `test_context_preloader`, `test_hardware_profiles_and_overlays`, `test_harness_characteristics`, `test_memory_clawback`, `test_safe_run`, `test_searxng_wsl`, `test_skills_manager`, `test_update_frequency`). Also `test_audit_fixes.py` and `test_headless_config.py` no longer exist — remove them.
2. **§2 directory tree** — rewrite against the ground-truth inventory:
   - `launch.cmd` → **`1_launch.cmd`**.
   - shims: add `bruno_llama_shim.py/.cmd`, `huang_llama_shim.py/.cmd`, and the three `.ts` Pi extensions.
   - backends: add `ascii_pruner`, `bruno_builder`, `cpu_decider`, `cpu_director`, `huang_builder`, `litellm`, `llamacpp_router`, `localai`, `searxng`; keep `cpu_decision` (present).
   - harnesses: add `claude.py`; header says "11 Coding Agent Harness Adapters" → 12 files / 30 registry keys (count the family).
   - bridges: add `claude_desktop_mcp.py`, `cpu_tools_cli.py`, `dom_navigator.py`, `safe_run.py`, `searxng_mcp.py` (tree lists only `gvs5h_solver.py` + `orca_openlumara/`).
   - root: add `harness_model.py`, `update_frequency.py`, `prompts.py`, `mcp_server.py`, `environment.py` (present), `hermes_system_prompt*.txt`, `hermes_tools.json`, `strata_models.json(.example)`, `strata_settings.json(.example)`, `models.local.ini` (gitignored override layer).
   - `localai/models/` "(33 configs)" → "auto-generated at runtime (gitignored; empty at rest)".
   - skills: add `reliable-file-read/` and `scoped/{pi,deepseek}/`.
   - `scripts/` contents differ: `0_run_agent.cmd`, `4_start_agy_cli.cmd`, `create_desktop_shortcuts.ps1`, `setup_strata_iq3s.py`, two `.url` webui shortcuts, `66-webui-dsh.url`.
   - tests: full 27-file list.
3. **§3.5 "33 model configurations"** → 36 `models.ini` presets (and growing with Huang/ascii-abli variants); also mention `freetokens_models.ini` and `models.local.ini` overlay.
4. **§3.6 CPU subsystem:** port 8081 description "Gemma 4 VLM director" is half-stale — 8081 is the **unified CPU llama-server worker** hosting Decider-4B C++ inference AND the Gemma director (AGENTS.md:10 has current wording; copy it). Add **8084 Decider-4B** and **8088 SearXNG** bullets. Port 8085 text is fine.
5. **§3.8 loop detection:** "N-grams older than max_history_words (1500)" OK; but the section should reflect tuned 12/5/0.93 vs defaults 8/3/0.82 (see §0) and the code-block +2 guard.
6. **§5 port table:** add 8084, 8088, 4000; 9935 retired note; row "Intel NPU STT / TabbyAPI direct" — TabbyAPI-direct-on-8000 claim is unverified in config (only `parakeet_npu.py:693` default 8000 found); drop or verify the TabbyAPI half.
7. **§5 CLI examples** use `C:\aiWorkspace\local_ai_launcher\launch.cmd` → `1_launch.cmd`.
8. **§6.1** references `local_ai_launcher\launch.cmd` — same rename.
9. **§7.2/§7.3 skills:** add `reliable-file-read` to builtin insurance; add `[skills.sources.rea]` to the Tier-2 registry diagram (11 sources, not 10); mention `ensure_scoped_harness_skills` (harness-scoped skills for pi/deepseek, `skills_manager.py:236`).
10. **§1 executive summary** harness list: add **Claude Code / Claude Desktop** to the unified-harness sentence.
11. Add a short subsection (or extend §3.2) on the **Anthropic Messages API surface** (`/v1/messages`, `/v1/messages/count_tokens`, `--claude-mode local_claude_native`, LiteLLM proxy on 4000) — this is currently only in AGENTS.md/README hints, not in ARCHITECTURE at all.
12. **§3.7 dispatcher / §8 environment** claims (trampoline healing, junctions) verified accurate vs `environment.py:59,125,202,230–231` — keep as-is.

### 1.3 `CHANGELOG.md` — P0 (version mismatch)

1. `__init__.py:6` says **2.0.0** but the changelog stops at 1.1.0 (2026-10-03). Add a `[2.0.0]` section (date it from git when the implementing AI commits — do not run the launcher). Content to include (all verified present in code):
   - Claude Code + Claude Desktop harness family, Claude-native Anthropic `/v1/messages` routing (`--claude-mode`, `--local-claude-native-engine(s)`), `bridges/claude_desktop_mcp.py`.
   - von System One gater replacing `open-jev-deberta-v3-large`/`clm-8b` (8083; `/v1/systemone`, `/v1/tripwire`, `/v1/shortlist`, `/v1/tools/route`).
   - Decider-4B v2.1 subsystem (8084 FastAPI + 8081 C++ worker, 9935 retired), `bridges/dom_navigator.py`.
   - SearXNG portable metasearch (8088, WSL2/native/remote tiers) + `bridges/searxng_mcp.py` + offline-policy tests.
   - LiteLLM Anthropic↔OpenAI proxy (4000).
   - Huang adaptive-KV-stream llama fork (builder+shim+6 `qwen-38_27b-huang-*` presets), ASCII-vocabulary pruner backend, Bruno/Huang/official **llamacpp_router**.
   - `update_frequency.py` scheduling, `--update-frequency-hrs`, `--force-update`.
   - `safe_run.py` Job Object containment.
   - Loop-detector retuning (12-gram/5-repeat/0.93 in launcher.toml; code-block +2 guard).
   - `harness_model.py` characteristics registry; harness-scoped skills; `reliable-file-read` skill; `[skills.sources.rea]`.
   - Memory clawback tiers (`memory_clawback.py`) + Strata config keys.
2. Fix 1.0.0 entry's port-8083 line (it says `open-jev-deberta-v3-large / clm-8b` — that was true then, leave historical entries alone, but do NOT backfill wrong history into 2.0.0).
3. Note: 1.0.0 claims `pyproject.toml` + CI "Added" — both live at the **workspace root**, not in the package dir; add a clarifying line.

### 1.4 `CONTRIBUTING.md` — P0 (one wrong file reference)

1. **§3** says add profiles to `HARDWARE_PROFILES` in **`presets.py`** and construct a `HardwareProfile(...)` dataclass. Reality: `HARDWARE_PROFILES: Dict[str, Dict[str, Any]]` in **`config.py:535`**, entries are plain dicts with keys `name`, plus tuning keys (see `config.py:536–606` — read the exact dict keys there and copy them into the example; the shown `vram_budget_mb/flash_attention/compute_precision/kv_cache_precision/description` shape must be validated against the actual dict contents before rewriting).
2. **§2** model-preset example: `jinja = true` — `models.ini` uses `jinja = on` (e.g. `models.ini:10,121`); align. Also mention `models.local.ini` as the preferred place for machine-specific tuning (AGENTS.md guardrail #2), and that `generate_all_model_configs()` signature is `generate_all_model_configs(target_dir=None, force=False)` (`model_configs.py:166`).
3. **§4** harness registration: registry lives in `harnesses/__init__.py:58` (correct) — add that adapters may register **multiple alias keys** (see claude family) and that `ONLINE_ONLY_HARNESSES` (`harnesses/__init__.py:97`) must be updated for cloud-only agents.
4. **§1** clone URL is right but the **repo root is `C:/aiWorkspace`** (package is a subdirectory; root `.gitignore` whitelists `!local_ai_launcher/`); `pyproject.toml` and `pytest` rootdir are at workspace root. Adjust setup commands (`uv sync`/`pytest local_ai_launcher/tests` from workspace root, `PYTHONPATH=C:/aiWorkspace`).
5. **§5** "run pytest" — add the AGENTS.md sequential-test guardrail (never parallel; port-cleanup fixtures bind identical sockets) and the note that test runs must not be launched while the live launcher holds ports.

### 1.5 `docs/STRATA.md` — P1 (one wrong port + minor)

1. **§5 light mode** lists evicted services as "BGE Reranker (8082), ONNX Gater (8083), **Parakeet Voice (Port 8084)**, CPU Director (8081)". Wrong: 8084 is **Decider-4B**; Parakeet (8000) is not evicted at all. Correct list per `memory_clawback.py:45–80`: reranker 8082, ONNX gater 8083, Decider-4B 8084, CPU Director 8081.
2. Config keys `strata_mmap_experts` / `strata_brutal_ram_clawback` / `strata_resident_budget_gib` and `[strata]` table — verified present (`config.py:257–263,726`) — keep.
3. `strata_brutal_ram_clawback` accepts aliases `brutal|full|heavy|true|1 → max`, `off|false|0` variants (`memory_clawback.py:190–201`) — optionally document accepted spellings.
4. VRAM/pack JSON files (`strata_settings.json`, `strata_models.json`, `strata-*.json` presets) are not mentioned — add a pointer (`backends/strata.py:load_strata_settings`, `scripts/setup_strata_iq3s.py`).

### 1.6 `docs/GVS5H_SOLVER.md` — P1 (new flags missing)

1. Verified accurate: all three invocation methods, `--max-iters 6`, `--max-seconds 180`, default model/endpoint (`gvs5h_solver.py:35–44`).
2. Missing from the args table: `--test-timeout` (default 10.0 s), `--max-log-runs` (default 100), `--prune-logs`, `--quiet`, `--ensure-skills` (provisions deep-solve SKILL.md into Pi/Hermes/workspace). Add them (`gvs5h_solver.py:939–943`).
3. Add note: `--model`/`--endpoint` defaults can be overridden via `GVS5H_MODEL` / `GVS5H_ENDPOINT` env vars (`gvs5h_solver.py:35–36`).

### 1.7 `docs/BLACKWELL_REFERENCE.md` — P1 (engine landscape changed)

1. **§7 "Alternative Engine Options"** lists only official + atomic-turboquant + tabby + freetoken. Reality now has **three llama.cpp forks routed by `backends/llamacpp_router.py`**: Bruno (blackwell, same repo/branch — `bruno_builder.py:26–29`), **Huang adaptive-KV-stream** (`huang_builder.py:24–25`, presets `qwen-38_27b-huang-{gsq,ascii,abli-ascii}-{think,nothink}`), official fallback. Rewrite §7 around the router; verify whether `llamacpp_fork = "atomic"` is still a valid config option before keeping that bullet (grep `llamacpp_fork` in config.py/models.py during the update pass; if absent, delete the bullet).
2. **§5 wiring reference** — re-verify `BLACKWELL_RUNTIME_ENV` still exists with `LLAMA_ARG_JINJA` (note: the 36-flag dict now lives in `bruno_builder.py:29+` as `BRUNO_RUNTIME_ENV`; `blackwell_builder.py` may alias it — check and cite the actual symbol/file).
3. **§8 mirror paths** — the third mirror is listed as `local_ai_launcher\BLACKWELL_REFERENCE.md` (package root) but the file lives at `local_ai_launcher\docs\BLACKWELL_REFERENCE.md`. Fix path (and re-check whether the other two mirrors still exist before asserting them).
4. Header date "September 2026" — add "last code-verified 2026-10-14".

### 1.8 `docs/UV_CACHE_PRUNE.md` — P2 (accurate; annotate only)

Verified against `environment.py:59,125,202,230–231` (`UV_CACHE_DIR`, `UV_PYTHON_INSTALL_DIR`, `heal_workspace_bin_trampolines` run inside `_setup_workspace_path`, `ensure_directory_junction`). No factual changes required. Optional: add "code-verified 2026-10-14" stamp.

### 1.9 `docs/analysis-local_ai_launcher-skill_routers.md` — P2 (proposal doc, mark status)

Port map and von/Decider-4B descriptions match current code (8081–8085, 9940–9943, 20128). It is a **proposal**, not a spec of current state — add a one-line status header: which routers (if any) were actually implemented as of 2026-10-14 (implementing AI: check `.agents/skills/` for `admin/`, `debug/`, `plan-and-build/` etc. junctions before claiming; do not assume). Its "26 skills" inventory is a point-in-time snapshot — label it as such.

### 1.10 `AGENTS.md` (package root) — P2 (mostly current)

1. Verified current: von on 8083, Decider-4B on 8084 with 8081 as C++ worker, 9935 retired, SearXNG 8088, `/v1/decide` + `/v1/rerank` on gateway.
2. Missing from §1 topology: **port 4000 (LiteLLM proxy)**, **9931/9933/9934 raw-engine ports**, **9932 MCP**, **3773 T3**, **8000 NPU STT** (8000 and 20128 are listed; add the others for completeness).
3. Guardrail note: mention `harness_model.py` as the seam for harness/engine capability routing, and `update_frequency.py` for update cadence, so AI modifiers don't re-implement them.

### 1.11 `README` vs `launcher.toml.example` cross-check — P1

`launcher.toml.example:189–193` documents loop defaults (8/3/0.82) while live `launcher.toml:138–142` ships 12/5/0.93 with different inline "default:" comments (threshold comment says "default: 5" in launcher.toml but the code default is 3). Decide one convention: example file = code defaults (keep), live file = tuned values (keep), but fix the misleading "default: 5" comment in `launcher.toml:139` to "code default: 3; tuned: 5" — this is a config-comment fix, include it in the plan so the implementing AI touches it too.

---

## 2. New documents to create

1. **`docs/CLAUDE_HARNESS.md` (P1)** — the Claude Code/Desktop family is the largest undocumented subsystem: `harnesses/claude.py` (30 registry keys, modes `local|official|local_claude_native`, desktop modes `mcp|gateway|disabled|off`), Anthropic `/v1/messages` native engine support (`config.py:739–771`), `bridges/claude_desktop_mcp.py`, LiteLLM 4000 translation path, `--claude-*` CLI flags. One page: mode matrix + port map + routing decision flow.
2. **`docs/LLAMACPP_ROUTERS.md` (P2, optional)** — Bruno vs Huang vs official fork selection (`llamacpp_router.py` docstring is the seed), Huang ascii-pruner pipeline, per-preset env differences. Only if BLACKWELL_REFERENCE.md §7 rewrite grows too large; otherwise fold into it.
3. **`docs/SEARXNG.md` (P2, optional)** — 3-tier architecture (WSL2 Docker / native Waitress / remote), 8088, MCP bridge, offline-policy test (`test_claude_searxng_offline_policy.py`). A section in ARCHITECTURE §3 may suffice.

---

## 3. Execution order & verification for the implementing AI

1. Order: **README → ARCHITECTURE → CHANGELOG → CONTRIBUTING → STRATA → GVS5H → BLACKWELL → AGENTS → analysis-doc stamp → new docs.**
2. Do not run the launcher or pytest (taskkill hazard). Verify statically: `grep` each port/model/flag name you write into a doc back to the `file:line` in §0; re-list `tests/`, `harnesses/__init__.py:58`, `models.ini` sections at edit time (they drift).
3. Keep historical claims (ARCHITECTURE §1 "legacy ai_launcher problem", CHANGELOG 1.0.0/1.1.0 bodies) as history — only add, never rewrite past entries, except where they were wrong at write time (none found).
4. Test-count phrasing everywhere: prefer "**471 tests collected** (as of 2026-10-14)" over "N tests pass" unless the implementer actually runs the suite.
5. Diagrams (`docs/*.svg`) are referenced only by prose; check whether the "4 Dedicated Engines" topology still matches (now 5+ GPU engines incl. Strata/Bruno/Huang) — if stale, note it under the SVG filename in a `docs/README.md` index (create one; it doesn't exist).

## 4. Items deliberately NOT flagged (verified accurate)

- `environment.py` trampoline healing + junction handling (ARCHITECTURE §8, UV_CACHE_PRUNE §2).
- Hermes multi-tier CLI resolution + bootstrap (ARCHITECTURE §8.3 — code present in `harnesses/hermes.py`).
- vram_coordinator `--evict/--restore/--status/--model` (README §6, `vram_coordinator.py:323–326`).
- GVS5H defaults table (except the missing new flags).
- STRATA §2–§4 budget/mmap mode content and config keys.
- Skills CLI command table (`skills_manager status/sync/update/clean` — README §7).
- Lion Cove P-core affinity & whitelisting claims (README §5.3–5.4) — not contradicted by code; left untouched.
