# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-10-14

### Added
- **Claude Code / Claude Desktop Harness Family** (`harnesses/claude.py`, 30 `HARNESS_REGISTRY` keys):
  - Routing modes via `--claude-mode {local, official, local_claude_native}`; `local_claude_native` lets native Anthropic Messages engines (default member: **Strata**) serve Claude Code/Desktop directly over `/v1/messages` without a translation bridge.
  - Claude Desktop modes via `--claude-desktop-mode {mcp, gateway, disabled, off}`; stdio MCP bridge `bridges/claude_desktop_mcp.py`.
  - Native Anthropic endpoints on the gateway: `/v1/messages`, `/messages`, `/v1/messages/count_tokens`.
  - Offline Answer Contract persona seeded into the isolated local profile (fingerprint-pinned by `tests/test_claude_searxng_offline_policy.py`).
- **von System One Gater (Port 8083)**: `wfzyx/von` ModernBERT non-autoregressive encoder replacing the deprecated `open-jev-deberta-v3-large` and `CLM-8B`; endpoints `/v1/decide`, `/v1/systemone`, `/v1/tripwire`, `/v1/shortlist`, `/v1/tools/route`.
- **Decider-4B v2.1 Subsystem (Port 8084)**: `Mapika/decider-4b-v2.1` FastAPI engine (`/v1/decider/slots`, `/v1/decider/dom`, `/v1/decider/policy_prior`) with raw C++ inference on the unified CPU worker (8081); **port 9935 retired**. DOM bridge `bridges/dom_navigator.py` (Mind2Web / MiniWoB++).
- **SearXNG Portable Metasearch (Port 8088)**: three service tiers (WSL2 Docker default / native Waitress / remote), MCP bridge `bridges/searxng_mcp.py`, offline web-search policy for agents.
- **LiteLLM Proxy (Port 4000)**: Anthropic Messages ↔ OpenAI translation for engines without native `/v1/messages`.
- **Huang Adaptive-KV-Stream llama fork**: builder + gRPC shim (`RaymondHuang210129/llama.cpp-adaptive-kv-streaming`), six `qwen-38_27b-huang-*` presets, ASCII-vocabulary pruner backend (`backends/ascii_pruner.py`, 14.25 GB → 12.25 GB), and the unified **llama-server variant router** (`backends/llamacpp_router.py`: Bruno / Huang / official priority engine; `blackwell_engine = "huang"` default).
- **Update Frequency Manager** (`update_frequency.py`): `--update-frequency-hrs`, `--force-update`, state in `config/update_state.json`.
- **`safe_run.py`**: Windows Job Object process containment runner (`KILL_ON_JOB_CLOSE`) reaping detached/grandchild processes.
- **Memory Clawback Tiers** (`memory_clawback.py`): `off` / `light` (evict aux microservices 8081–8084 + working-set trim) / `max` (terminate non-essential user apps), wired to Strata `strata_brutal_ram_clawback`.
- **Loop-Detector Retuning**: code defaults 8-gram / 3-repeat / 0.82; shipped `launcher.toml` tunes to 12 / 5 / 0.93 with code-block +2 threshold guard (eliminates `'''`-comment false positives).
- **Harness/Engine Characteristics Registry** (`harness_model.py`): offline capability, meta-harness designation, preload/inner-CLI resolution ordering.
- **Skills**: `reliable-file-read` builtin skill, harness-scoped skill deployment (`skills/scoped/{pi,deepseek}`), `[skills.sources.rea]` registry entry.

### Changed
- Gateway endpoint surface documented: `/v1/responses`, `/v1/rerank`, `/v1/decide`, `/v1/decider/*` (there is **no** `/v1/embeddings` endpoint).
- Test suite grew from 169 → **471 collected tests** across 27 files.

> Note: `pyproject.toml` and the GitHub Actions CI live at the **workspace root** (`C:/aiWorkspace`), which is the git repository root; `local_ai_launcher/` is a whitelisted subdirectory.

## [1.1.0] - 2026-10-03

### Added
- **Strata Inference Engine Backend**:
  - Purpose-built C++/CUDA inference engine integration for **Qwen3.8-Flash-Next (125B MoE)** on consumer gaming hardware (RTX 5080 Laptop GPU with 16 GB VRAM + 64 GB DDR5 RAM).
  - High-speed token generation (~60–90+ tokens/s) via dense VRAM layer pinning and host DDR5 PCIe sparse expert streaming.
  - Native speculative decoding via Multi-Token Prediction (MTP) draft head (`strata/rt`).
  - Dedicated gRPC shim ([`local_ai_launcher/shims/strata_grpc_shim.py`](shims/strata_grpc_shim.py)) on port **9943** wrapping internal Strata server on port **9934**.
  - Dynamic mutual exclusion and VRAM coordinator support (`Free()`) with 500ms driver memory unmapping stabilization.
  - Windows Job Object binding (`assign_process_to_job`) and `force_kill_port` process whitelist registration for orphan prevention.
  - Model preset `[qwen-38-flash-strata-iq3s]` (aliases: `strata-iq3s`, `qwen-strata-iq3s`) with 64K context budget and INT8 KV cache.
  - Automated shard downloader and pack builder script ([`local_ai_launcher/scripts/setup_strata_iq3s.py`](scripts/setup_strata_iq3s.py)).

## [1.0.0] - 2026-10-02

### Added
- **Unified Gateway**: OpenAI-compatible REST API on port 8080 powered by LocalAI supervisor.
- **Dynamic Mutual Exclusion (`SINGLE_ACTIVE_BACKEND=true`)**: Automatic engine swapping between Blackwell GGUF, TabbyAPI EXL3, and FreeToken NVFP4 without VRAM collisions.
- **Hardware Profile System**: Modular presets (`blackwell_16gb`, `ada_24gb`, `standard_16gb`, `budget_12gb`) in `launcher.toml`.
- **Zero-Loss Local Overrides (`models.local.ini`)**: Dynamic runtime INI overlays for local machine tuning without polluting Git commits.
- **Blackwell sm_120 Optimization**: Custom llama.cpp MSVC runtime with 36 hybrid-KV environment variables and WDDM BAR1 aperture exhaustion protections.
- **CPU Auxiliary Microservices (0 MB GPU VRAM)**:
  - Port 8081: CPU llama-server (Gemma 4 Vision Director)
  - Port 8082: CPU cross-encoder reranker (`BAAI/bge-reranker-v2-m3`)
  - Port 8083: CPU ONNX discrete decision gater (`open-jev-deberta-v3-large` / `clm-8b`)
- **GVS5H Zero-Shot Deep-Solve Engine**: Autonomous multi-agent algorithm solver with pre-flight backup, automated test execution, and rollback.
- **Voice Dictation**: Intel NPU-accelerated speech-to-text with OpenVINO Parakeet TDT V3 and global hotkey.
- **Harness Integration**: Out-of-the-box adapters for Orca ADE, Pi, Hermes, DeepSeek, OpenCode, Pie, T3, OpenLumara, and Google Antigravity.
- **Windows Environment Self-Healing**:
  - Autonomous `uv` PE trampoline healing (`heal_workspace_bin_trampolines`) detecting and unlinking broken stubs (`error: uv trampoline failed to canonicalize script path`) using sub-millisecond binary signature pre-filtering.
  - Hardened NTFS directory junction management (`ensure_directory_junction`) resilient against dangling reparse points and Windows `mklink` collision errors.
  - Multi-tier Hermes CLI candidate probing (`hermes.exe` -> `hermes.cmd` -> internal venv scripts) and automated Nous Research PowerShell bootstrap.
- **Cross-Platform Support**: Added `launch.sh` for Linux/macOS/WSL alongside Windows `1_launch.cmd`.
- **Standard Packaging**: Added `pyproject.toml` with `uv`/`pip` dependencies and GitHub Actions CI.
