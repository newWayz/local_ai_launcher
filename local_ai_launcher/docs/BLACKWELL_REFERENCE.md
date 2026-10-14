# Blackwell Llama-Server Optimization & Porting Reference Guide
**Target System**: NVIDIA GeForce RTX 5080 Laptop GPU (16 GB GDDR7, WDDM 3.x, Optimus)  
**Target Model**: `Qwen3.8-27B-UD-IQ4_XS.gguf` (14.25 GB, hybrid GDN + 16-layer attention)  
**Source Implementation**: `BrunoPPassini/llama.cpp` (branch: `qwen38-blackwell-256k`, commit: `1a63cbf`)  
**Date**: September 2026 (engine-landscape sections re-verified against code 2026-10-14)

---

## 1. Executive Summary & Root Cause Diagnostic

### The Issue
Both `ai_launcher` and `local_ai_launcher` previously hung indefinitely near initialization or on initial token evaluation with the final log message referencing `ggml_backend_cuda_sparse_ensure_range`.

### The Core Root Cause
1. **Context-wide VMM Barrier (`cuCtxSynchronize`)**:
   In `ggml/src/ggml-cuda/ggml-cuda.cu:906`, the runtime executes a context-wide synchronization barrier `CU_CHECK(cuCtxSynchronize())` inside `sparse_mutex` before modifying GPU virtual memory page tables.
2. **Missing Proactive Host Fallback (`LLAMA_KV_SPARSE_DEVICE_RESERVE_MIB`)**:
   In the author's desktop launcher, `LLAMA_KV_SPARSE_DEVICE_RESERVE_MIB` was undefined (defaulting to 0). This forced the engine to always attempt physical device VRAM allocations (`cuMemCreate` with `CU_MEM_LOCATION_TYPE_DEVICE`) for the first 65,536 tokens (`LLAMA_KV_SPARSE_DEVICE_PREFIX_TOKENS=65536`, ~1,152 MiB).
3. **Laptop WDDM VRAM Exhaustion & BAR1 Aperture Wall**:
   - On the desktop RTX 5070 Ti, free VRAM after loading 13,061 MiB of model weights was ~3,242 MiB (resting with ~100–140 MiB free under full context).
   - On the RTX 5080 Laptop, Windows DWM and display compositor reservations leave only ~1,617 MiB free VRAM.
   - Live hardware telemetry revealed:
     ```text
     FB Memory Usage:   Total: 16303 MiB | Free: 16003 MiB (unloaded)
     BAR1 Memory Usage: Total: 16384 MiB | Used: 16355 MiB | Free: 29 MiB
     ```
   - With only **29 MiB of BAR1 aperture free** and VRAM saturated, device allocations triggered an unrecoverable WDDM paging deadlock inside `cuCtxSynchronize()`.

---

## 2. Architectural Comparison of the 3 Strategic Paths

```
                  ┌─────────────────────────────────────────────────────────┐
                  │ Strategic Paths for Blackwell Llama-Server on RTX 5080  │
                  └───────────────────────────┬─────────────────────────────┘
                                              │
         ┌────────────────────────────────────┼───────────────────────────────────┐
         ▼                                    ▼                                   ▼
┌───────────────────┐               ┌───────────────────┐               ┌───────────────────┐
│      Path A       │               │      Path B       │               │      Path C       │
│  Fork Upstream    │               │ Fork BrunoPassini │               │ Runtime Config &  │
│  llama.cpp master │               │  Blackwell Branch │               │ Existing Binaries │
├───────────────────┤               ├───────────────────┤               ├───────────────────┤
│ High Risk / Weeks │               │ Medium / Rebuild  │               │ Lowest Risk / 0s  │
│ 77 files conflict │               │ Patch mobile WDDM │               │ Tuned via EnvVars │
│ Upstream refactor │               │ Keep sm_120a core │               │ VERIFIED WORKING  │
└───────────────────┘               └───────────────────┘               └───────────────────┘
```

---

### Path A: Forking Upstream `master` (`ggml-org/llama.cpp`)

* **Concept**: Fork the official `ggml-org/llama.cpp` repository and port Bruno Passini's Blackwell features onto the latest master codebase.
* **Potential Advantages**:
  - Incorporates latest upstream CUDA 13.4/WDDM driver 616 patches.
  - Native multi-token prediction (MTP) drafting improvements and cleaner memory pools.
* **Blockers & Severe Risks**:
  - **77 Modified Files**: The Blackwell fork made extensive, invasive modifications across `ggml-cuda.cu`, `fattn.cu`, `mmq.cu`, `gated_delta_net.cu`, `llama-kv-cache.cpp`, `llama-context.cpp`, and `llama-memory-hybrid.cpp`.
  - **Architectural Divergence**: Upstream `llama.cpp` has completely refactored its memory architecture since base commit `d775b8967` (introducing `llama-memory-hybrid-iswa`, `llama-kv-cache-dsa`, and split buffer allocators).
  - Porting the custom sparse VMM and MMA host ring into upstream's new memory model requires thousands of lines of manual C++/CUDA rewrites.
* **Verdict**: **Not recommended.** Extremely high engineering effort with a high probability of introducing subtle numerical or concurrency regressions.

---

### Path B: Forking Bruno Passini's Branch (`qwen38-blackwell-256k`)

* **Concept**: Fork `https://github.com/BrunoPPassini/llama.cpp.git` directly on branch `qwen38-blackwell-256k` and maintain a custom build specifically tailored for mobile Blackwell.
* **Potential Advantages**:
  - **Preserves Core Blackwell Optimizations**: Retains all custom `sm_120a` MMA kernels, fused Q4 dequantization, P8 online-softmax partitioning, and recurrent GDN transaction logs without merge conflicts.
  - **Enables Direct C++ Driver Patches**:
    1. Replace context-wide `cuCtxSynchronize()` barriers in `ggml-cuda.cu` with fine-grained CUDA event waits (`cudaEventSynchronize`).
    2. Add hardware checks querying free BAR1 aperture via `cuDeviceGetAttribute` before attempting host VMM mapping.
    3. Remove the hardcoded `RTX 5070 Ti` device name string check in `src/llama-model.cpp`.
* **Build Configuration**:
  ```powershell
  cmake -S . -B build-blackwell `
    -G "Visual Studio 17 2022" -A x64 `
    -DGGML_CUDA=ON `
    -DGGML_CUDA_FA_ALL_QUANTS=ON `
    -DGGML_CUDA_GRAPHS=ON `
    -DGGML_CUDA_FORCE_CUBLAS=OFF `
    -DGGML_CUDA_FORCE_MMQ=OFF `
    -DCMAKE_CUDA_ARCHITECTURES=120a-real

  cmake --build build-blackwell --config Release --target llama-server --parallel 4
  ```
* **Verdict**: **The superior source-code path.** Only required if runtime environment tuning (Path C) cannot satisfy edge-case stability requirements.

---

### Path C: Runtime Configuration of Existing Binaries (VERIFIED & PROMOTED)

* **Concept**: Leverage the 36 built-in runtime environment switches already compiled into `C:\aiWorkspace\llama-cpp-blackwell\ggml-cuda.dll` without modifying or rebuilding C++ code.

---

## 3. Detailed Breakdown of Path C Phases

### Phase 1: Minimal Impact & Proactive Headroom (VERIFIED WORKING)

* **Objective**: Enable proactive host fallback before VRAM is exhausted; shrink the device prefix to fit laptop VRAM while keeping first 32K tokens at full GPU speed.
* **Configuration**:
  ```ini
  LLAMA_KV_SPARSE_DEVICE_RESERVE_MIB = 512
  LLAMA_KV_SPARSE_DEVICE_PREFIX_TOKENS = 16384
  LLAMA_KV_RESERVE_TOKENS = 16384
  LLAMA_KV_SPARSE_TRACE = 1
  LLAMA_KV_HOST_RING_MMA_TRIPLE_PIPELINE = 1
  LLAMA_COMPUTE_SPARSE_VMM = 1
  LLAMA_COMPUTE_SPARSE_TRIM = 1

  # Model-specific thread restriction (prevents E-core stalls on Intel Arrow Lake-HX)
  threads = 8
  threads-batch = 8
  ```
* **How It Works**:
  - Setting `DEVICE_RESERVE_MIB = 512` ensures that initial KV allocation uses native GPU VRAM (since free VRAM is ~1,475 MiB > 512 MiB floor). Setting it higher (e.g. 1536 MiB) caused token-zero host fallback where all tokens crawled across PCIe.
  - Setting `PREFIX_TOKENS = 16384` uses ~280 MiB of VRAM for KV, leaving ~795 MiB of physical VRAM cushion for compute and WDDM.
  - Restricting compute to `threads = 8` keeps worker threads pinned strictly to the 8 Performance cores (P-cores) on the Intel Core Ultra 9 275HX, eliminating E-core synchronization stalls.
* **Verification Evidence**:
  - Inference response time dropped from 4.1s to **0.99 seconds** (prompt prefill at **340 t/s**, decode at **67.2 t/s**).
  - No premature host fallback on token 0; GPU remains responsive without desktop lag.

---

### Phase 2: Compute-Arena Barrier Elimination & Prefix Shrink

* **When to Apply**: If deep context prefill (e.g. 80K+ tokens) or large prompt bursts re-trigger a driver stall.
* **Configuration Additions**:
  ```ini
  LLAMA_COMPUTE_SPARSE_VMM = 0
  LLAMA_COMPUTE_SPARSE_TRIM = 0
  LLAMA_KV_SPARSE_DEVICE_PREFIX_TOKENS = 16384
  LLAMA_KV_RESERVE_TOKENS = 16384
  ```
* **How It Works**:
  - The Blackwell fork applies sparse VMM to the compute scratch arena (`ggml_backend_cuda_buffer_init_tensor` calls `ensure_range` on graph execution).
  - Setting `COMPUTE_SPARSE_VMM=0` allocates the ~400 MiB compute arena once statically via `cudaMalloc`, eliminating 90% of runtime `sparse_ensure_range` barrier calls during inference.
  - Dropping the device prefix to 16,384 tokens recovers **~864 MiB of physical VRAM**.
* **Trade-offs / Side Effects**:
  - Tokens 16,384 to 65,536 enter the cold ring earlier, dropping decode speed in that window from ~75 tg/s to ~45 tg/s.
  - Resting baseline VRAM consumption increases by ~300 MiB because compute scratch is never trimmed.

---

### Phase 3: Maximum Concurrency Relaxation & Watchdog Tuning

* **When to Apply**: Nuclear fallback if Windows WDDM TDR watchdog continues to kill long prefills.
* **Configuration Additions**:
  ```ini
  LLAMA_KV_HOST_RING_MMA_TRIPLE_PIPELINE = 0
  LLAMA_SPARSE_VMM_MAX_RUN_MIB = 16
  LLAMA_KV_HOST_RING_MMA_PREFILL_FSM = 0
  ```
  *Windows Registry*:
  ```text
  Key   : HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\GraphicsDrivers
  Value : TdrDelay (DWORD) = 60 (decimal)
  ```
* **How It Works**:
  - Disabling the triple pipeline (`TRIPLE_PIPELINE=0`) falls back to double-buffering or serial DMA staging, eliminating inter-stream CUDA event race conditions.
  - Setting `MAX_RUN_MIB=16` throttles mapping slices so they never exceed the remaining 29 MiB BAR1 aperture.
  - `TdrDelay=60` extends the Windows GPU watchdog timeout from 2 seconds to 60 seconds.
* **Trade-offs / Side Effects**:
  - Cold-tail decode speed (>16K/32K tokens) drops by ~10% to 12% due to lack of three-way concurrency overlap.
  - Massive prompt expansions may experience prefill stutter as memory maps in 16 MiB increments.

---

## 4. Side Effects & Operational Trade-offs Matrix

| Parameter Change | Positive Stability Effect | Latency / Throughput Trade-off | Resource Impact |
|---|---|---|---|
| `DEVICE_RESERVE_MIB = 1536` | Prevents WDDM 0-byte VRAM paging crash | None | Cold tokens route to host DDR5 |
| `PREFIX_TOKENS: 64K -> 32K` | Frees ~576 MiB physical VRAM | Tokens 32K–64K decode at ~45 tg/s instead of ~75 tg/s | Less VRAM consumed, ~500 MB more host RAM |
| `PREFIX_TOKENS: 32K -> 16K` | Frees ~864 MiB physical VRAM | Tokens 16K–32K also drop to ~45 tg/s | Lowest VRAM pressure |
| `COMPUTE_SPARSE_VMM = 0` | Eliminates compute-time sync barriers | None during inference | Resting VRAM floor is ~300 MiB higher |
| `TRIPLE_PIPELINE = 0` | Eliminates stream event deadlocks | ~10–12% slower cold decode | Slightly less GPU compute/copy concurrency |
| `MAX_RUN_MIB = 16` | Fits inside 29 MiB BAR1 aperture | Minor prefill latency stutter on large initial prompts | Smaller contiguous driver mappings |
| `TdrDelay = 60` | Prevents Windows watchdog kill | OS takes 60s to recover if GPU genuinely crashes | Mild impact (mitigated by Optimus iGPU display) |

---

## 5. Wiring Reference for Launchers

### For `ai_launcher` (Atomic Sandbox)
File: `C:\aiWorkspace\ai_launcher\backends\blackwell_builder.py`
```python
BLACKWELL_RUNTIME_ENV: Dict[str, str] = {
    # ... standard flags ...
    "LLAMA_KV_SPARSE_HOST_FALLBACK": "1",
    "LLAMA_KV_SPARSE_DEVICE_RESERVE_MIB": "512",
    "LLAMA_KV_SPARSE_TRACE": "1",
    "LLAMA_KV_SPARSE_DEVICE_PREFIX_TOKENS": "16384",
    "LLAMA_KV_RESERVE_TOKENS": "16384",
    # ... remaining flags ...
}
```
*Atomic Test Probe*: `C:\aiWorkspace\ai_launcher\test_blackwell_phase1.ps1`

### For `local_ai_launcher` (Production Deployment)
Promotion status: **COMPLETED & ACTIVE** (re-wired since; code-verified 2026-10-14).
1. The canonical 36-flag runtime dictionary (including `"LLAMA_ARG_JINJA": "1"`) is defined as `BRUNO_RUNTIME_ENV` in `C:\aiWorkspace\local_ai_launcher\backends\bruno_builder.py`; `BLACKWELL_RUNTIME_ENV` is an alias to it (same file) re-exported through `blackwell_builder.py`.
2. Fork selection is now layered: `config.py` keeps `llamacpp_blackwell_update_behavior = "never_update"`, while the active llama variant is decided per-request by `backends/llamacpp_router.py` (`bruno` / `huang` / `official`). `llamacpp_fork` defaults to `"official"` and the `blackwell_engine` knob selects `"huang"` (default) or `"bruno"` for Blackwell-class models; the older `llamacpp_fork = "blackwell_qwen38"` value no longer exists.
3. `C:\aiWorkspace\local_ai_launcher\models.ini` has `[qwen-38_27b-blackwell-256k-think]` configured with:
   ```ini
   batch-size = 1024
   ubatch-size = 512
   ctk = q4_0
   ctv = q4_0
   ctx-size = 262000
   jinja = on
   threads = 8
   threads-batch = 8
   cache-reuse = 0
   ```

---

## 6. Historical Context & Detailed Optimization Changelog

### 6.1 Original Upstream State & Design Intent
* **Upstream Fork**: `BrunoPPassini/llama.cpp` (branch `qwen38-blackwell-256k`, base commit `1a63cbf`).
* **Original Intended Hardware**: NVIDIA GeForce RTX 5070 Ti Desktop GPU (16 GB dedicated GDDR7, desktop motherboards with full BAR1 aperture support, no Windows DWM/Optimus display compositor shared-memory contention).
* **Original Memory Allocation Model**:
  - Assumed ~3,242 MiB of free physical VRAM after loading model weights.
  - Hardcoded device prefix allocation of 65,536 tokens (`LLAMA_KV_SPARSE_DEVICE_PREFIX_TOKENS=65536`, ~1,152 MiB).
  - Defaulted `LLAMA_KV_SPARSE_DEVICE_RESERVE_MIB` to 0 (allocating physical VRAM until completely exhausted before attempting host paging).
* **Original Architecture Gate**:
  - `src/llama-model.cpp` contained a string filter strictly matching `dev_desc.find("RTX 5070 Ti") != std::string::npos` to activate Blackwell MMA features.
* **Original Prompt / Template Handling**:
  - Relied on llama-server's native C++ `peg` template parser, without forcing Jinja template evaluation.

---

### 6.2 Failure Modes Encountered on RTX 5080 Laptop GPU
1. **Architecture Gate Rejection**:
   - The laptop GPU reports its name as `NVIDIA GeForce RTX 5080 Laptop GPU`.
   - The original upstream binary rejected the GPU during model weight loading and refused to initialize the custom `sm_120a` kernels.
2. **0-Byte WDDM Memory Deadlock inside `cuCtxSynchronize()`**:
   - Unlike the desktop RTX 5070 Ti, the RTX 5080 Laptop runs under Windows DWM with Optimus hybrid graphics.
   - Post-load free VRAM was only ~1,617 MiB, and free BAR1 aperture was critically depleted to just **29 MiB** (`Used: 16355 MiB | Free: 29 MiB`).
   - The 64K token prefix allocation (~1,152 MiB) exhausted remaining physical VRAM to near 0 bytes. When the runtime attempted to map host memory, `CU_CHECK(cuCtxSynchronize())` at `ggml-cuda.cu:906` deadlocked the Windows WDDM 3.x driver.
3. **Turn-by-Turn Prefix Cache Eviction (Token 0 Reprocessing)**:
   - In multi-turn chat sessions (e.g. at 81,241+ tokens), the native C++ template parser introduced whitespace divergence on `<think>` boundaries (e.g., `\n</think>\n\n` vs `\n\n</think>\n\n`).
   - Because the Blackwell fork uses non-sliceable Gated Delta Net (GDN) recurrence layers alongside standard attention, any single-token divergence invalidated the entire cache prefix, resulting in:
     `W slot operator (): id 0 | task 9513 | cached history cannot be truncated exactly; reprocessing prompt from token zero`.

---

### 6.3 Complete Record of Modifications & Specific Purposes

| Component / Setting | Original Value | Optimized Local Value | Purpose & Architectural Rationale |
|---|---|---|---|
| **Architecture Gate** (`src/llama-model.cpp`) | Match `"RTX 5070 Ti"` only | Match `"RTX 5070 Ti"`, `"RTX 5080"`, or `"RTX 50"` | Enables Compute Capability 12.0 `sm_120` detection for mobile RTX 5080 / Blackwell laptop chips. |
| **Compilation Architecture** | Unspecified / Desktop default | `-DCMAKE_CUDA_ARCHITECTURES=120a-real` | Targets native Blackwell MMA instruction set with real PTX generation. |
| **`LLAMA_KV_SPARSE_DEVICE_RESERVE_MIB`** | `0` (undefined) | `512` | Guarantees a 512 MiB VRAM safety cushion for Windows WDDM compositor and compute scratch, preventing 0-byte paging deadlock. |
| **`LLAMA_KV_SPARSE_DEVICE_PREFIX_TOKENS`** | `65536` (~1,152 MiB) | `16384` (~280 MiB) | Reduces hot KV VRAM footprint by ~864 MiB, fitting comfortably into laptop VRAM without triggering BAR1 exhaustion. |
| **`LLAMA_KV_RESERVE_TOKENS`** | `65536` | `16384` | Synchronizes KV reservation floor with prefix size for deterministic host DDR5 ring fallback. |
| **CPU Worker Threads** (`threads`, `threads-batch`) | `16` (All host cores) | `8` | Pins execution strictly to the 8 Performance cores (P-cores) of Intel Core Ultra 9 275HX, eliminating E-core stalls. |
| **`LLAMA_ARG_JINJA` & `--jinja`** | Unset (off) | `"1"` (`jinja = on`) | Forces deterministic Jinja2 chat template rendering, eliminating 1-token whitespace drift at `<think>` block boundaries. |
| **`cache-reuse`** | `256` | `0` | Bypasses speculative prefix slicing on non-sliceable GDN recurrence state, ensuring clean append without token-0 rollback. |
| **Launcher Wiring** (`llamacpp_fork`) | `'official'` | `'blackwell_qwen38'` | Promotes this locally optimized sm_120a build as the active llama engine across `ai_launcher` and `local_ai_launcher`. |
| **Update Lock** (`llamacpp_blackwell_update_behavior`) | Unset / auto-update | `'never_update'` | Permanently locks the custom compiled sm_120a binary against git pull / release downloads. |

---

### 6.4 Rationale for Disabling & Permanently Locking Update Functionality
The update functionality for `blackwell_qwen38` was explicitly removed and permanently locked (`never_update`) for the following critical technical reasons:
1. **Upstream Incompatibility**: The upstream repository `BrunoPPassini/llama.cpp:qwen38-blackwell-256k` does not contain the RTX 5080 Laptop device name patch in `llama-model.cpp` nor the mobile WDDM headroom defaults.
2. **Binary Destruction Risk**: A `git pull`, `git reset`, or automated binary downloader would overwrite the locally compiled `120a-real` binary (`llama-server-mtpctx.exe` / `llama-server.exe`) and wipe out the custom patches.
3. **Driver / Environment Coupling**: This build is tuned specifically for the CUDA 13.x / sm_120a toolchain and the laptop's BAR1/WDDM limits. Automated updates would break this verified working state.

---

## 7. Remaining Alternative Engine Options & Strategic Trade-offs

If requirements shift or a different model format is needed, the following alternative backends are supported and available in the workspace:

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│        Alternative Inference Engines Overview (router-managed, code-verified 2026-10-14)    │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
  │
  ├─► [1] Blackwell-class llama.cpp variants — selected per request by backends/llamacpp_router.py
  │     ├─ Bruno fork  (blackwell_engine = "bruno")
  │     │    • Engine: BrunoPPassini/llama.cpp branch qwen38-blackwell-256k, sm_120a custom build
  │     │    • Best For: Hybrid GDN architectures (Qwen 3.8 / Qwen 3.5), extreme context (up to 256K)
  │     │    • Strengths: 64K VRAM hot / 192K DDR5 cold ring, native sm_120a MMA tensor cores,
  │     │      canonical 36-flag BRUNO_RUNTIME_ENV.
  │     ├─ Huang fork  (blackwell_engine = "huang", DEFAULT)
  │     │    • Engine: RaymondHuang210129/llama.cpp-adaptive-kv-streaming branch feature/adaptive-kv-stream-v2
  │     │    • Best For: 63–96.8% VRAM utilization with MTP only, zero CPU offload, circular KV arena
  │     │    • Companions: ASCII-vocabulary pruner (backends/ascii_pruner.py, 14.25 GB -> ~12.25 GB)
  │     │      enabling full GPU offload (ngl = -1) within 16 GB; presets
  │     │      qwen-38_27b-huang-{gsq,ascii,abli-ascii}-{think,nothink}.
  │     └─ Official upstream (ggml-org/llama.cpp, router fallback)
  │          • Best For: Standard Transformer GGUFs (Llama 3, Mistral, Command-R, DeepSeek non-GDN)
  │          • Strengths: Rock-solid stability, standard CUDA graph execution, upstream bugfixes.
  │          • Limitations: Lacks the custom host ring / sm_120a MMA kernels and adaptive KV streaming.
  │
  ├─► [2] TabbyAPI / ExLlamaV3 (`exllamav3`)
  │     • Model Path: `models/exl3/*` (e.g. `Muse-Glimmer-30B-exl3-3.0bpw`, `Qwen3.8-27B-exl3`)
  │     • Config Switch: `default_model = "Muse-Glimmer-30B-exl3-3.0bpw"` in config.py
  │     • Strengths: Industry-leading token-generation throughput (100+ tg/s), dedicated PyTorch
  │       allocator, zero WDDM barrier synchronization issues, native flash-attn.
  │     • Limitations: Requires EXL3 format models; does not support standard GGUF files.
  │
  ├─► [3] FreeToken NVFP4 (`freetoken` gRPC Shim)
  │     • Model Path: `models/freetoken/*` (e.g. `nvidia_Qwen3.6-35B-A3B-NVFP4`)
  │     • Config Switch: `default_model = 'freetoken\nvidia_Qwen3.6-35B-A3B-NVFP4'`
  │     • Strengths: Native NVFP4 hardware acceleration on Blackwell Tensor Cores (sm_120).
  │     • Limitations: Fixed FP4 precision; higher VRAM baseline for model parameters.
  │
  └─► [4] Strata 125B MoE (`strata` gRPC Shim, port 9943)
        • Model Path: IQ3_S packs built by scripts/setup_strata_iq3s.py
        • Strengths: 125B MoE on 16 GB VRAM via expert streaming; native Anthropic Messages API.
        • Details: docs/STRATA.md.
```
*(The former "Atomic TurboQuant" fork option (`llamacpp_fork = "atomic"`) is no longer present in the codebase.)*

---

## 8. Persistence Architecture & Safe Mirror Locations

To ensure that future launcher updates, repository re-clones, or automated maintenance scripts do not accidentally delete or overwrite this reference and documentation, copies are permanently mirrored across three isolated, git-ignored tiers:

1. **Primary Workspace Reference** (Permanent models directory, gitignored):
   `C:\aiWorkspace\models\blackwell_optimization_paths_reference.md`
2. **Binary-Adjacent Reference** (Inside the update-locked Blackwell binary directory):
   `C:\aiWorkspace\llama-cpp-blackwell\README_BLACKWELL_OPTIMIZATION.md`
3. **Launcher Architecture Reference** (Directly inside the local production launcher package):
   `C:\aiWorkspace\local_ai_launcher\docs\BLACKWELL_REFERENCE.md`

Future AI assistants encountering Blackwell, sm_120a, or long-context prefix-caching behavior should consult this guide before altering build flags, environment variables, or update behavior.

