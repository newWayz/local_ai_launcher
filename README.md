# Local AI Launcher

> **Unified Local AI Serving & Coding Harness Orchestrator**  
> Engineered for maximum performance and zero-loss precision on the flagship reference rig (NVIDIA Blackwell RTX 5080 Laptop GPU 16 GB VRAM & Intel Core Ultra NPU), with modular hardware profile adaptability across Ada, Ampere, and modern accelerators.

`local_ai_launcher` consolidates local LLM serving, specialized quantization backends (llama.cpp GGUF with hybrid KV cache, TabbyAPI EXL3, FreeToken NVFP4), speech-to-text dictation, and multiple autonomous coding agent harnesses under a single OpenAI-compatible API on port **8080** (`http://localhost:8080/v1`).

---

## 1. Core Architectural Capabilities

1. **Unified API Gateway**: Exposes standard OpenAI-compatible REST endpoints (`/v1/chat/completions`, `/v1/models`, `/v1/embeddings`) backed by [LocalAI](https://github.com/mudler/LocalAI) supervisor architecture.
2. **Hardware Profiles & Zero-Loss Overlays**:
   - **Flagship Reference (`blackwell_16gb`)**: Retains 100% of tested hybrid-KV memory parameters, 36 CUDA environment variables, 256k context limits, and MTP draft acceleration tuned specifically for NVIDIA Blackwell RTX 5080 Laptop (16 GB VRAM).
   - **Modular Profiles**: Built-in selectable profiles for `ada_24gb` (RTX 4090 / RTX 6000 Ada with 32k FP16 KV), `standard_16gb` (RTX 4080 / 3090 with balanced flash attention), and `budget_12gb` (RTX 4070 / 3060 with Q4_0 KV cache).
   - **Local Overrides (`models.local.ini`)**: User-defined local overrides merge cleanly over hardware presets without polluting Git commits or dropping performance fidelity.
3. **Dynamic Engine Mutual Exclusion (`SINGLE_ACTIVE_BACKEND=true`)**: Automatically swaps and evicts models between Blackwell GGUF, TabbyAPI EXL3, and FreeToken NVFP4 backends on the fly without VRAM collisions or manual restarts.
4. **Smart Online vs. Offline Harness Detection**:
   - When only **online-only** harnesses (`agy`, `agy-ide`, `agy-desktop`) are selected, the offline LLM engine and OmniRoute proxy are **completely bypassed** to conserve power and VRAM.
   - The Intel NPU speech-to-text engine starts independently regardless of harness mode.
5. **Diffusion Co-existence (ComfyUI / OmniStudio)**: Provides `vram_coordinator.py` to immediately evict the LLM from GPU memory during heavy image/video generation gates (FLUX.2, Wan 14B), lazy-reloading automatically upon the next coding agent prompt.
6. **Integrated Voice Dictation**: Background speech-to-text service accelerated on Intel NPU using OpenVINO Parakeet TDT 0.6B V3 with global hotkey (`Right Alt`) and REST API on port `8000`.

---

## 2. Tools & Repositories Used

| Tool / Repository | Role / Purpose | Default Endpoint / Port |
| :--- | :--- | :--- |
| **[LocalAI](https://github.com/mudler/LocalAI)** | Unified API supervisor & model dispatcher (`native_server.py`) | `http://localhost:8080/v1` |
| **[llama.cpp](https://github.com/ggerganov/llama.cpp)** (Blackwell `sm_120`) | Custom build with 36 hybrid-KV environment variables for RTX 5080 | Managed via gRPC shim on port `9940` (raw `9931`) |
| **[TabbyAPI](https://github.com/theroyallab/tabbyAPI)** | ExLlamaV3 (EXL3) runtime with 256-multiple sequence constraints | Managed via gRPC shim on port `9941` |
| **FreeToken** | NVFP4 runtime with CUDA 13.x and PyTorch Windows hotfixes | Managed via gRPC shim on port `9942` |
| **[OmniRoute](https://github.com/dnakov/omniroute)** | Token compression proxy for prompt optimization | `http://localhost:20128` |
| **Intel OpenVINO + Parakeet TDT V3** | Offline NPU-accelerated speech-to-text voice dictation | `http://127.0.0.1:8000/v1/audio/transcriptions` |
| **Windows Terminal (`wt.exe`)** | Launches each coding agent into dedicated, named tabs | Native Windows Terminal window |

### Supported Coding Agent Harnesses
- **Offline LLM Agents**: Orca ADE (`orca`), Pi (`pi`), Hermes (`hermes`), DeepSeek (`deepseek`), OpenCode (`opencode`), Pie (`pie`), T3 (`t3`), OpenLumara (`openlumara` / `lumara`), Llama Server WebUI (`ls`).
- **Online-Only Agents**: Google Antigravity CLI (`agy` / `antigravity`), Antigravity IDE (`agy-ide`), Antigravity Desktop (`agy-desktop`).

---

## 3. Quick Start & Installation

### Option A: One-Click Launch Scripts (Recommended)

**Windows 11 (PowerShell / Command Prompt)**:
```bat
local_ai_launcher\1_launch.cmd
```

**Linux / macOS / WSL**:
```bash
chmod +x launch.sh
./launch.sh
```

The launcher automatically verifies or provisions the isolated `.venv` environment using `uv`, installs missing dependencies, checks backend compilers, and launches the orchestration cockpit.

### Option B: Modern Package Installation (`uv` or `pip`)

```bash
# Clone the repository
git clone https://github.com/newWayz/local_ai.git
cd local_ai

# Using uv (fastest)
uv sync --all-extras

# Or using pip in editable mode
pip install -e ".[all]"

# Launch the orchestrator
python -m local_ai_launcher
```

---

## 4. Hardware Profiles & Customization

The system configures itself through `local_ai_launcher/launcher.toml` (copy from `launcher.toml.example`):

```toml
[hardware]
# Active hardware profile: "blackwell_16gb", "ada_24gb", "standard_16gb", "budget_12gb"
profile = "blackwell_16gb"

# Target GPU device index
gpu_device_id = 0

# Maximum VRAM budget in megabytes
vram_budget_mb = 16384

# Enable flash attention kernel acceleration
flash_attention = true

# Primary tensor compute precision
compute_precision = "fp16"

# Key-Value cache quantization format
kv_cache_precision = "q8_0"
```

### Zero-Loss Local Overrides (`models.local.ini`)

To customize context length, threads, or paths for your specific machine without touching tracked Git files, create `models.local.ini` in the workspace root or inside `local_ai_launcher/`:

```ini
[default]
threads = 16
ctx-size = 32768

[qwen-38_27b-blackwell-256k-think]
ctx-size = 65536
threads = 14
```

`models.local.ini` is automatically ignored by Git and merged with top priority at runtime, allowing you to fine-tune settings without merge conflicts.

---

## 5. Command-Line Reference

| CLI Argument | Description | Example |
| :--- | :--- | :--- |
| `--harness [names...]` | Specify one or more coding harnesses to launch | `--harness agy` (online only)<br>`--harness orca pi` (offline) |
| `--model [name]` | Set starting model alias | `--model qwen-38_27b-blackwell-256k-think` |
| `--headless` / `--no-gui` | Run LocalAI in headless mode (no browser popup) | `python -m local_ai_launcher --headless` |
| `--gui` | Open LocalAI WebUI dashboard in default browser | `python -m local_ai_launcher --gui` |
| `--port [port]` | Change LocalAI API port (default: 8080) | `--port 8085` |
| `--no-omniroute` | Bypass OmniRoute proxy, connects directly to LocalAI | `--no-omniroute` |
| `--no-preload` | Do not preload default model into VRAM at startup | `--no-preload` |
| `--download-model` | Automatically download missing Hugging Face files | `--download-model` |
| `--cpu-only` | Run backend models without CUDA acceleration | `--cpu-only` |

### VRAM Coordination (ComfyUI / Diffusion Gates)

```bat
:: Evict LLM from VRAM before heavy image/video generation
local_ai_launcher\vram_coordinator.cmd --evict

:: Check current VRAM usage
local_ai_launcher\vram_coordinator.cmd --status

:: Manually restore LLM into VRAM
local_ai_launcher\vram_coordinator.cmd --restore
```

---

## 6. System Ports Reference

| Port | Service | Notes |
| :--- | :--- | :--- |
| **8080** | LocalAI Core Gateway | All coding agents connect here (`http://localhost:8080/v1`) |
| **8000** | Intel NPU Speech-to-Text | Parakeet TDT V3 transcription endpoint (`/v1/audio/transcriptions`) |
| **20128** | OmniRoute Proxy | Token compression proxy (routes to 8080) |
| **9931** | Internal `llama-server` | Direct backend port managed by Blackwell shim |
| **9932** | MCP Server | Tool proxy for Antigravity Desktop / IDE |
| **9940** | Blackwell gRPC Shim | LocalAI internal gRPC backend connection |
| **9941** | TabbyAPI gRPC Shim | LocalAI internal gRPC backend connection |
| **9942** | FreeToken gRPC Shim | LocalAI internal gRPC backend connection |
| **3773** | T3 Code Server | Internal web interface for T3 |

---

## 7. Testing & Verification

Run the full automated test suite using pytest with `PYTHONPATH` set to the repository root:

**Windows (PowerShell)**:
```powershell
$env:PYTHONPATH="."; pytest local_ai_launcher/tests
```

**Linux / macOS / WSL (Bash)**:
```bash
PYTHONPATH=. pytest local_ai_launcher/tests
```

All 216 unit, integration, and property tests execute in **< 3.5 seconds**.

---

## 8. License

This project is licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
