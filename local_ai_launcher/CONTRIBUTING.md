# Contributing to Local AI Launcher

## 1. Development Setup
The git repository root is the **workspace** (`C:/aiWorkspace`); `local_ai_launcher/` is a git-tracked subdirectory of it, and `pyproject.toml` plus the GitHub Actions CI (`ci.yml`) live at that root. Clone the repo and work from the checkout root (this is the workspace root; the Python package lives in the `local_ai_launcher/` subdirectory):
```bash
git clone https://github.com/newWayz/local_ai_launcher.git
cd local_ai_launcher   # the checkout root, NOT the package subdirectory
```
Provision virtual environment with dependencies using `uv` (recommended) or `pip`:
```bash
# Using uv (from workspace root)
uv sync --all-extras

# Or using pip in editable mode
pip install -e ".[all]"
```

## 2. Adding a Model Preset (`models.ini`)
Add a new INI section under `local_ai_launcher/models.ini` (machine-specific tuning of *existing* presets belongs in the gitignored `models.local.ini`, which `presets.py` merges with top priority):
```ini
[my-model-alias]
model = models/llama-server/owner/repo/model.gguf
threads = 8
ctx-size = 32768
temperature = 0.7
top_k = 40
top_p = 0.95
jinja = on
chat-template-file = models/my-template.jinja
```
After editing, regenerate LocalAI definitions (the generated YAML lands in the gitignored `localai/models/` directory):
```bash
python -c "from local_ai_launcher.localai.model_configs import generate_all_model_configs; generate_all_model_configs()"
```

## 3. Adding a Hardware Profile (`config.py`)
Hardware profiles are plain dicts in `HARDWARE_PROFILES` inside **`local_ai_launcher/config.py`** (not `presets.py`). Add your profile there, mirroring the existing key set:
```python
"my_gpu_profile": {
    "name": "my_gpu_profile",
    "description": "Custom profile for 12 GB GPUs",
    "engine_binary": "llama-cpp/llama-server.exe",
    "vram_ceiling_mb": 12288,
    "vram_evict_threshold_mb": 11000,
    "strict_vram_eviction": False,
    "default_n_cpu_moe": 0,
    "ctk": "q8_0",
    "ctv": "q8_0",
    "batch_size": 4096,
    "ubatch_size": 4096,
},
```
Select it via `hardware_profile = "my_gpu_profile"` in `launcher.toml`.

## 4. Adding an Agent Harness (`harnesses/`)
1. Create `local_ai_launcher/harnesses/my_agent.py` subclassing `HarnessAdapter`.
2. Implement `launch()`, `is_installed()`, and `setup_config()`.
3. Register your class in `HARNESS_REGISTRY` inside `local_ai_launcher/harnesses/__init__.py`. One adapter class may register **multiple alias keys** (see the Claude family — 30 keys).
4. If the agent requires cloud connectivity, add its keys to `ONLINE_ONLY_HARNESSES` in the same file, and describe its capabilities (offline model support, meta-harness, CLI vs GUI) in `harness_model.py`.

## 5. Verification & Testing
Before submitting a pull request, run the full test suite **sequentially** from the workspace root (never in parallel — port-cleanup fixtures bind identical TCP sockets; and do not run it while the live launcher holds the ports):
```bash
$env:PYTHONPATH="C:\aiWorkspace"; pytest local_ai_launcher/tests
```
All tests must pass with zero warnings or port leaks.
