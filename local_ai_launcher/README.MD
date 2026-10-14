# Local AI Launcher

> **Unified Local AI Serving, Coding Harness Orchestrator & Autonomous Solver**  
> Optimized for Windows 11 on NVIDIA RTX 5080 Laptop GPU (16 GB VRAM) & Intel Core Ultra NPU.

`local_ai_launcher` consolidates local LLM serving, multi-engine quantization backends, real-time loop mitigation, speech-to-text dictation, the GVS5H deep-solve autonomous algorithm solver, and multiple coding agent harnesses under a single OpenAI-compatible API on port **8080** (`http://localhost:8080/v1`).

---

## 1. Architecture & Core Capabilities

- **Unified API Gateway**: Exposes OpenAI-compatible endpoints (`/v1/chat/completions`, `/v1/models`, `/v1/embeddings`) backed by [LocalAI](https://github.com/mudler/LocalAI) architecture. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).
- **Dynamic Mutual Exclusion (`SINGLE_ACTIVE_BACKEND=true`)**: Automatically swaps and evicts models between Blackwell GGUF (`sm_120`), TabbyAPI EXL3, FreeToken NVFP4, and Strata (125B MoE) without VRAM collisions or manual restarts. See [`docs/BLACKWELL_REFERENCE.md`](docs/BLACKWELL_REFERENCE.md) and [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).
- **Real-Time Model Loop Mitigation**: Gateway-level engine in [`localai/loop_detector.py`](localai/loop_detector.py) evaluates n-gram repetition ($N=16$, threshold $\ge 4$) and cyclic autocorrelation ($p \in \{1, 2, 3\}$, similarity $\ge 0.93$). Immediately calls `call.cancel()` on upstream gRPC streams to halt GPU token generation, synthesizes closing `</think>` tags, and injects redirection notices.
  - **Tuning rationale (2026-10-09, see `C:/aiWorkspace/tmp/local_llm_harness_stall_analysis.md`)**: n-gram threshold raised 3 -> 4 because exact 16-word runs repeated 3x occur legitimately in code comments (triple-backtick / parametrized tables); the 4th repetition still catches true loops ~0.6-2 s later at ~50 tok/s. Cycle similarity raised 0.85 -> 0.93 because near-identical code/JSON "sentences" (one identifier differing, Jaccard $\approx (n-1)/(n+1)$ for $n$ unique words) false-positived at 0.85 (2 logged trips: DataFrame rows, libuv boilerplate); 0.93 requires $\ge 23$ unique shared words, evicting the 13-20-word boilerplate band while verbatim prose loops ($\ge 0.95$) remain caught. Net effect: 0 demonstrated true positives lost (all loop classes are caught by the n-gram path), false-positive cost (cancelled 60k-token turn + 180 s sampler perturbation) eliminated.
- **Dual-Tier CPU / GPU Dispatcher**: CPU auxiliary microservices handle sub-15ms discrete decisions and passage reranking with **0 MB GPU VRAM**, freeing the RTX 5080 for deep reasoning models (e.g. Qwen 3.8 27B / Qwen 3.8 Flash-Next 125B MoE).
- **Sub-30ms TTFT Preloader**: [`context_preloader.py`](context_preloader.py) pre-evaluates system prompts and tool schemas into the local LLM KV cache at launch (`max_tokens=1`), slashing Time To First Token from ~2.5s to **<30ms** for Pi, Hermes, and T3.
- **Skill Suite Gateway**: [`skills_manager.py`](skills_manager.py) manages upstream Git skill suites (`gstack`, `superpowers`, etc.) via shallow clones and exposes clean router junctions to harnesses, cutting prompt overhead from **~11.4k to ~1.2k tokens (90% reduction)**.

---

## 2. Ports & Endpoints Matrix

| Port | Service | Role & Default Endpoint |
| :---: | :--- | :--- |
| **8080** | **LocalAI Core Gateway** | Primary API for all coding agents (`http://localhost:8080/v1`) |
| **8081** | **CPU llama-server** | CPU-only Media Director (`-ngl 0`, Lion Cove P-cores, Gemma 4 Vision / text) |
| **8082** | **CPU Reranker** | Cross-encoder passage reranker (`bge-reranker-v2-m3` on `/v1/rerank`) |
| **8083** | **CPU CLM / Decision Gater** | Sub-15ms choices & tool shortlisting (`open-jev-deberta-v3-large` or `clm-8b` on `/v1/decide`, `/v1/shortlist`) |
| **8085** | **Hermes Signal Daemon** | Dedicated Signal CLI gateway daemon |
| **8000** | **Intel NPU Speech-to-Text** | Offline Parakeet TDT V3 transcription API (`/v1/audio/transcriptions`) |
| **20128** | **OmniRoute Proxy** | Token compression proxy routing to 8080 |
| **9940** | **Blackwell / llama.cpp Shim** | Internal gRPC shim for GGUF backends (raw `llama-server` on `9931`) |
| **9941** | **TabbyAPI gRPC Shim** | Internal gRPC shim for ExLlamaV3 (EXL3) backends |
| **9942** | **FreeToken gRPC Shim** | Internal gRPC shim for NVFP4 backends (direct HTTP on `9933`) |
| **9943** | **Strata gRPC Shim** | Internal gRPC shim for Strata 125B MoE (raw engine on `9934`) |

---

## 3. Quick Start & CLI Options

```bat
:: Launch default configuration
1_launch.cmd

:: Or directly with Python
python -m local_ai_launcher
```

### Essential CLI Flags

| CLI Flag | Description | Example |
| :--- | :--- | :--- |
| `--harness [names...]` | Coding harnesses to launch into Windows Terminal | `--harness orca pi`<br>`--harness agy` (online only)<br>`--harness none` (voice-to-text only) |
| `--model [alias]` | Set initial active model alias | `--model qwen-38_27b-blackwell-256k-think`<br>`--model strata-iq3s` |
| `--gui` / `--headless` | Open browser WebUI cockpit or run silently | `python -m local_ai_launcher --gui` |
| `--no-omniroute` | Direct connection to LocalAI (bypasses compression proxy) | `--no-omniroute` |
| `--no-preload` | Skip initial model preloading into VRAM | `--no-preload` |
| `--cpu-only` | Force backend inference onto CPU without CUDA | `--cpu-only` |

### Supported Coding Agent Harnesses
- **Offline Agents**: Orca ADE (`orca`), Pi (`pi`), Hermes (`hermes`), DeepSeek (`deepseek`), OpenCode (`opencode`), Pie (`pie`), T3 (`t3`), OpenLumara (`openlumara`), Llama Server WebUI (`ls`).
- **Online-Only Agents**: Google Antigravity CLI (`agy`), Antigravity IDE (`agy-ide`), Antigravity Desktop (`agy-desktop`).
- **Standalone Dictation**: `none` (runs Intel NPU voice daemon with 0 MB LLM VRAM consumption).

---

## 4. Configuration & Headache Prevention

Create a `launcher.toml` file in `local_ai_launcher/` or in the workspace root to override defaults without editing git-tracked files (see [`launcher.toml.example`](launcher.toml.example)):

```toml
# launcher.toml
default_model = "qwen-38_27b-blackwell-256k-think"
choose_harness = ["orca", "pi"]
enable_loop_detection = true
loop_recovery_temp_floor = 0.15
voice_dictation_hotkey = "f8"

# Hardware Profile override:
# [hardware]
# profile = "blackwell_16gb"
```

> [!WARNING]
> **TOML Windows Path Syntax**: Always use **single quotes** for Windows paths containing backslashes in TOML (e.g. `'freetoken\nvidia_Qwen3.6-35B-NVFP4'`). Python raw strings `r"..."` are **invalid in TOML** and will crash the parser on launch.

> [!IMPORTANT]
> **UV Cache & Python Trampoline Maintenance**: `UV_PYTHON_INSTALL_DIR` is isolated inside `uv-cache/python`. When clearing disk space, **never delete `uv-cache/python/` or break its directory junctions**. The launcher features autonomous startup self-healing (`heal_workspace_bin_trampolines`) to automatically detect and purge broken `uv` PE trampolines (`failed to canonicalize script path`) and repair dangling NTFS junctions. See [`docs/UV_CACHE_PRUNE.md`](docs/UV_CACHE_PRUNE.md) for details and maintenance procedures.

> [!TIP]
> **Loop Recovery Temperature Floor**: `loop_recovery_temp_floor` (default: `0.15`) prevents over-cooling during consecutive recovery streaks while ensuring reasoning stays coherent. When a loop is detected, the gateway halts the stream via `call.cancel()` and cools the temperature incrementally without dropping below the floor.

> [!IMPORTANT]
> **Strata Tool-Call Parsing & Backtick Stalls**: The strata backend renders its OWN chat template from the model pack (`strata/data/packs/<pack>/tokenizer/chat_template.jinja`); `chat-template-file` / `jinja` in `models.ini` are ignored for strata (they apply to llama-server/blackwell presets only). Its serve-layer parser (`strata/serve/frontend.py`, `OutputParser`) matches the tool-call open tag ANYWHERE in the streamed text, so when a model quotes the call format (e.g. inside a triple-backtick fence or inline code span) the parser fabricates a bogus tool call or raises `malformed tool call` and aborts the stream — harnesses (pi/hermes/t3code) then receive a stop-terminated, tool-call-free response and stall. Aggravating factor: the gateway loop detector previously false-positived on near-identical code/JSON sentences (see tuned values above). Recommended parser fix (not yet applied): in `OutputParser.feed` content state, combine **V1 fence parity** (track triple-backtick fence parity; ignore the open tag inside fences — WHY: harness prompts teach models to quote the format inside fences, and the parser fires on the first tag it sees) with **weak-V4 shape check** (after the open tag, require `function=` + identifier before entering call state — WHY: rejects JSON-format examples and prose garbage, the two observed crash signatures, while keeping hallucinated-name calls intact and avoiding backtick-span parity fragility and the line-start contract the pack template never states). Companion: gate `native_server._extract_text_tool_calls` (non-stream path) with the same shape check. Full evidence and regression analysis: `C:/aiWorkspace/tmp/local_llm_harness_stall_analysis.md`.

---

## 5. CPU Auxiliary Subsystem (Ports 8081–8083)

The launcher implements a **System 1 (CPU Fast-Path) / System 2 (GPU Deep Reasoning)** dual-tier architecture:

1. **CPU Reranker (Port 8082)**: Runs `BAAI/bge-reranker-v2-m3` via FastEmbed/OpenVINO CPU for cross-encoder passage ranking (`/v1/rerank`), dropping irrelevant context before GPU invocation.
2. **CPU Decision Gater & CLM (Port 8083)**: Executes zero-generation typed decisions, tool shortlisting, and security triage (`/v1/decide`, `/v1/shortlist`):
   - **`com-kotobalabs/open-jev-deberta-v3-large`** *(Default)*: 435M bidirectional dual-encoder, **~14ms** latency, **~450 MB** RAM footprint (INT8).
   - **`stanford-nvidia/clm-8b`** *(High-Capacity)*: 8.03B causal decoder with action vector caching, **~35ms** latency on DDR5-6400, **~8.2 GB** RAM footprint (INT8).
3. **Core Tool Whitelisting**: Foundational tools (`read`, `write`, `edit`, `find`, `grep`, `bash`, `terminal`) are **strictly whitelisted** and never evicted during tool shortlisting, preventing agent hallucination and broken workflows.
4. **Lion Cove P-Core Affinity**: CPU microservices automatically pin threads to physical P-cores 0–7 (affinity mask `0x00FF`), preventing Windows Thread Director from demoting matrix operations to Skymont E-cores.

---

## 6. Autonomous Solvers & VRAM Coordination

### GVS5H Deep-Solve Engine
A 5-phase sequential ledger-based optimization loop for complex algorithms and dense modules. See [`docs/GVS5H_SOLVER.md`](docs/GVS5H_SOLVER.md).

- **In Pi or Hermes**: Simply prompt the agent:
  > *"Use deep-solve to implement an efficient A\* pathfinding algorithm in `src/pathfinder.py` and test it with `pytest tests/test_pathfinder.py`."*
- **Via Terminal CLI**:
  ```powershell
  python -m local_ai_launcher.bridges.gvs5h_solver `
    --task "Implement thread-safe LRU cache" `
    --target "src/cache.py" `
    --test-cmd "pytest tests/test_cache.py"
  ```

### VRAM Coordination (Diffusion Gates)
Coexists with heavy image/video generation tools (ComfyUI, FLUX.2, Wan 14B) without memory crashes:

```bat
:: Evict LLM from GPU memory before diffusion generation
vram_coordinator.cmd --evict

:: Inspect current VRAM allocation
vram_coordinator.cmd --status

:: Restore LLM into GPU VRAM
vram_coordinator.cmd --restore
```

---

## 7. Skill Suite Manager & Directory Layout

### Skills CLI Commands

```powershell
# Inspect builtin insurance and external suites
python -m local_ai_launcher.skills_manager status

# Synchronize router junctions into .agents/skills/ (non-destructive)
python -m local_ai_launcher.skills_manager sync

# Pull latest upstream Git commits for all suites
python -m local_ai_launcher.skills_manager update

# Prune legacy flattened clutter from .agents/skills/
python -m local_ai_launcher.skills_manager clean
```

### Component Architecture

```
local_ai_launcher/
├── __main__.py          # Entrypoint, CLI parsing, service orchestration
├── config.py            # Global configuration defaults, ports, model catalogs
├── dispatcher.py        # Smart Agent Dispatcher (CPU fast-path, circuit breaking)
├── context_preloader.py # Sub-30ms TTFT prompt preloading engine
├── vram_coordinator.py  # VRAM monitoring & ComfyUI eviction coordinator
├── skills_manager.py    # Git vendor manager & NTFS junction router
├── backends/            # Engine drivers (Blackwell sm_120, TabbyAPI, FreeToken, NPU voice, CPU microservices)
├── localai/             # LocalAI gateway core, loop detector, and Cockpit WebUI
├── shims/               # gRPC shims conforming to backend.proto
├── harnesses/           # Agent adapters (orca, pi, hermes, agy, t3, opencode, etc.)
├── bridges/             # GVS5H deep-solve solver and CPU tools CLI bridge
├── skills/              # Git-tracked builtin custom skills (cpu-tools, deep-solve, wipe-agy-chat)
├── scripts/             # Troubleshooting launchers, WebUI shortcuts, and desktop generator
├── docs/                # Architecture, Blackwell tuning, GVS5H guide, and UV prune manual
└── tests/               # Pytest verification test suite
```

---

## 8. Testing & Verification

Run the automated test suite with `PYTHONPATH` set to the workspace root:

```powershell
$env:PYTHONPATH="."; pytest tests
# Or from workspace root:
# $env:PYTHONPATH="C:\aiWorkspace"; pytest local_ai_launcher/tests
```

All **223 unit and integration tests** pass, verifying model routing, gRPC shims, loop detection & cooling recovery, CPU auxiliary services (ports 8081–8083), CLM tool shortlisting guardrails, GVS5H deep-solve, VRAM coordination, skills management, and Windows trampoline self-healing.
