# Strata 125B MoE Inference Engine & Memory Architecture

## 1. Overview
[Strata](https://github.com/Niko1221/Strata) is a specialized speculative decoding C++/CUDA inference engine for large Mixture-of-Experts (MoE) models such as **Qwen3.8-Flash-Next 125B** (IQ3_S quantization). It runs within `local_ai_launcher` via an isolated gRPC shim ([`strata_grpc_shim.py`](file:///C:/aiWorkspace/local_ai_launcher/shims/strata_grpc_shim.py)) on port **9943**, supervising the native engine process on port **9934** and routing through LocalAI on port **8080**.

---

## 2. VRAM & Cache Budgeting (16 GB RTX 5080 Laptop GPU)

Operating a 125B parameter MoE model on a 16 GB GPU requires strict memory budgeting across dense base weights, multi-token prediction (MTP) draft heads, the KV cache, and dynamic expert cache slots:

| Component | Allocation | Details |
|---|---|---|
| **Dense Weights & PLE** | ~4.20 GiB | Shared attention, routers, and PLE dense tensors resident in VRAM |
| **MTP Speculative Head** | ~0.80 GiB | Multi-token prediction draft verification head |
| **Hybrid KV Cache** | ~3.19 GiB | `--max-context 262144 --kv int8` (12 QSA layers @ 13.0 KB/token) |
| **Expert GPU Cache** | ~5.39 GiB | 4,190 dynamic expert cache slots (`--expert-cache auto`) |
| **Activation & CUDA Buffers** | ~0.80 GiB | Scratchpad, GEMM workspace, and cuBLAS handles |
| **Total VRAM Allocated** | **~14.30 GiB** | **1.70 GiB headroom** under Windows WDDM 16.0 GiB ceiling |

### KV Cache Sizing & Quantization
Qwen3.8-Flash-Next utilizes a hybrid attention architecture (36 GDN linear recurrence layers with constant 117.7 MB state, and 12 QSA attention layers).
- **Author's Recommended Precision (`--kv int8`)**: Grouped-64 INT8 KV storage. Full floating-point precision matching all published benchmarks. At 262,144 (256K) native context, occupies **3.19 GiB**.
- **Alternative 4-Bit Mode (`--kv q4_0`)**: Hadamard-rotated 4-bit KV cache. Occupies **1.59 GiB** at 256K context (~4% faster, ~5,420 expert cache slots), but slightly less precise on long-context needles.
- **KV Streaming (`--kv-resident 32768`)**: Streams older KV blocks from 64 GB DDR5 RAM when running extreme context windows (>256K).

---

## 3. Storage & Disk Write Analysis
- **Zero Disk Writes**: GGUF shards are mapped read-only via Windows `MapViewOfFile` with `PAGE_READONLY` and `FILE_MAP_READ` (see `src/core/expert_source.cpp`). 
- **No Weight Swapping to Disk**: No temporary swap files or intermediate weight files are written to SSD during inference or caching.
- **Pack Preparation**: Strata's `tools/iq_pack.py` generates native pack artifacts (`dense.bin`, `index.txt`, `tokenizer/`) once at initial setup inside `strata/data/packs/`.

---

## 4. `mmap-experts`: Modes & Performance

Configured cleanly in `launcher.toml` (or via `LauncherConfig` in `config.py`):
```toml
# Top-level in launcher.toml:
strata_mmap_experts = true         # true: OS page cache (mmap), false: resident RAM arena
strata_brutal_ram_clawback = "light" # "off", "light" (default: trim & aux eviction), "max" (kill user apps)
strata_resident_budget_gib = 52    # Explicit RAM budget in GiB for resident experts (engine 0.1.31+)

# Or via [strata] table:
[strata]
mmap_experts = true
brutal_ram_clawback = "light"
resident_budget_gib = 52
```
*(Note: `mmap-experts` and `resident-budget-gib` are intentionally not placed in `models.ini` to ensure compatibility with standard `llama-server` engine variants that do not recognize those flags).*

Engine-level defaults additionally load from `strata_settings.json` / `strata_models.json` (see `backends/strata.py:load_strata_settings` and the `.example` templates; per-model pack presets ship as `strata-*.json`). Initial shard download + pack build: `python local_ai_launcher/scripts/setup_strata_iq3s.py`.

### Mode A: `strata_resident_budget_gib = 52` (Hybrid Resident Budget + Lookahead Prefetch - Recommended)
- **Mechanism**: Strata copies up to 52 GiB of the hottest experts (ranked by `expert-profile.bin`) directly into pinned host RAM, streams cold experts via `MapViewOfFile` (mmap), and activates background routing lookahead prefetch (`PrefetchVirtualMemory`).
- **Memory Footprint**: Clamped cleanly to available RAM minus 4 GiB headroom; requires no application force-termination.
- **Throughput**: ~**30 – 45 tokens/sec**.

### Mode B: `strata_mmap_experts = true, strata_resident_budget_gib = 0` (Pure Dynamic OS Page Cache)
- **Mechanism**: The 50.3 GB expert arena is memory-mapped as virtual memory without a pinned RAM budget.
- **Memory Footprint**: Minimal (~4-8 GB working RAM). Fits cleanly on machines under heavy background load.
- **Throughput**: ~**6.5 – 8.6 tokens/sec**.

### Mode C: `strata_mmap_experts = false` (100% Physical RAM Resident Arena)
- **Mechanism**: Strata pre-allocates and commits the full 50.3 GB expert arena directly into physical RAM at startup.
- **Throughput**: **40 – 53 tokens/sec** peak throughput.
- **Requirement**: Requires 55–60 GB of contiguous physical RAM available immediately.

---

## 5. Memory Clawback Tiers (`memory_clawback.py`)

Configured via `strata_brutal_ram_clawback = "off" | "light" | "max"` (accepted spellings also include `brutal`/`full`/`heavy`/`true`/`1` → `max`, and `false`/`0` → `off`; see `memory_clawback.py`):

1. **`"off"` Mode**:
   - Skips all memory reclamation steps.
2. **`"light"` Mode (Default)**:
   - **Suppression of Auxiliary Microservices**: Evicts BGE Reranker (Port 8082), ONNX Decision Gater (Port 8083), Decider-4B Engine (Port 8084), and CPU Director (Port 8081). Reclaims **~4.0 – 5.0 GB** of RAM. (Parakeet voice on Port 8000 is not part of the clawback set.)
   - **Working Set Trimming (`K32EmptyWorkingSet`)**: Flushes unreferenced private working set pages to the standby list. Reclaims **~1.0 – 2.5 GB** in <500 ms.
   - **Zero Process Termination**: User programs (browsers, Discord, Steam, IDEs) remain open.
3. **`"max"` Mode (Brutal Clawback)**:
   - Includes everything in `"light"`.
   - **Aggressive Process Termination**: Issues a loud console warning and terminates non-essential memory-heavy user applications: Chrome, Edge, Firefox, Discord, Slack, Steam, Spotify, etc.
   - **Strict Protection**: Never terminates `explorer.exe`, Windows critical system binaries, Windows Terminal, PowerShell, or active agent execution trees.
