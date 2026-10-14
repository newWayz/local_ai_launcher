# Local AI Launcher — Agent Directives & Architecture Seams

## 1. System Topology & Port Allocations
- **Port 8080 (Unified Gateway)**: Native FastAPI server in `localai/native_server.py`. Exposes OpenAI-compatible endpoints (`/v1/chat/completions`, `/v1/models`, `/v1/embeddings`, `/v1/decide`, `/v1/rerank`). All agent harnesses connect here.
- **Ports 9940–9942 (Inference gRPC Shims)**: Managed via `shims/base_shim.py`:
  - `9940`: `blackwell_llama_shim.py` (Blackwell `sm_120` llama-server with 36 hybrid-KV env vars)
  - `9941`: `tabby_grpc_shim.py` (TabbyAPI EXL3 with 256-sequence rounding)
  - `9942`: `freetoken_grpc_shim.py` (FreeToken NVFP4)
- **Ports 8081–8085, 8088 (CPU Auxiliary Microservices & Daemons, 0 MB GPU VRAM)**:
  - `8081`: Unified CPU `llama-server.exe` worker (`-ngl 0`). Runs Decider-4B C++ inference worker by default; also hosts Gemma 4 Vision CPU Director when media production directing is active (strictly isolated, non-overlapping workloads). Port `9935` is retired in favor of `8081`.
  - `8082`: CPU Reranker (`BAAI/bge-reranker-v2-m3` via FastEmbed/FastAPI)
  - `8083`: CPU ONNX Gater (`wfzyx/von` ModernBERT System One Reflex Gater)
  - `8084`: CPU Decider-4B FastAPI Engine (`Mapika/decider-4b-v2.1` multi-slot extraction, Mind2Web DOM actions, MCTS policy priors; proxies raw inference to Port 8081)
  - `8085`: Signal CLI Daemon
  - `8088`: SearXNG Portable Metasearch Service (Tier 3 WSL2 Docker Engine default / Tier 2 Native Waitress / Tier 1 Remote; 0 MB GPU VRAM)
- **Port 8000**: Intel NPU Speech-to-Text (Parakeet TDT V3 transcription endpoint)
- **Port 4000**: LiteLLM Proxy (Anthropic Messages <-> OpenAI translation for Claude routing)
- **Port 20128**: OmniRoute Token Compression Proxy
- **Raw Engine Ports** (behind the gRPC shims): `9931` internal `llama-server`, `9933` FreeToken direct HTTP, `9934` Strata native engine. `9935` is retired (use 8081).
- **Port 9932**: MCP tool proxy (Antigravity Desktop/IDE); **Port 3773**: T3 internal web interface

## 2. Core Guardrails for AI Modifiers
1. **Preserve Blackwell Flagship Tuning**: Never lower or dilute context lengths, threads, or hybrid-KV cache allocations for the flagship reference profile (`blackwell_16gb`) in `models.ini` or `presets.py`. Local hardware changes belong in `launcher.toml` or `models.local.ini`.
2. **Local Overrides Precedence**: Machine-specific customizations belong in `models.local.ini` (automatically merged with top priority via `presets.py` and ignored by Git).
3. **Dynamic Path Hygiene**: Never hardcode absolute filesystem paths. Always import `LOCAL_AI_LAUNCHER_DIR` or `AI_WORKSPACE_DIR` from `local_ai_launcher.paths`.
4. **Sequential Test Discipline**: Run `pytest local_ai_launcher/tests` to verify modifications. Never launch parallel test executions; port cleanup fixtures bind to identical TCP sockets.
5. **Git Cleanliness**: Never commit machine-specific state (`launcher.toml`, `models.local.ini`, `.env`, or `.lnk` binary shortcuts).
6. **Existing Seams — Reuse, Don't Re-implement**: harness/engine capability routing (offline capability, meta-harness, preload priority) lives in `harness_model.py`; update cadence/scheduling lives in `update_frequency.py` (`config/update_state.json`); llama fork variant selection lives in `backends/llamacpp_router.py` (never hardcode a single fork).
