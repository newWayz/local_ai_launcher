# Contributing to Local AI Launcher

## 1. Development Setup
1. Clone the repository and navigate to the project directory:
   ```bash
   git clone https://github.com/newWayz/local_ai_launcher.git
   cd local_ai_launcher
   ```
2. Provision virtual environment with dependencies using `uv` (recommended) or `pip`:
   ```bash
   # Using uv
   uv sync --all-extras

   # Or using pip in editable mode
   pip install -e ".[all]"
   ```

## 2. Adding a Model Preset (`models.ini`)
Add a new INI section under `local_ai_launcher/models.ini`:
```ini
[my-model-alias]
model = models/llama-server/owner/repo/model.gguf
threads = 8
ctx-size = 32768
temperature = 0.7
top_k = 40
top_p = 0.95
jinja = true
chat-template-file = models/my-template.jinja
```
After editing, regenerate LocalAI definitions:
```bash
python -c "from local_ai_launcher.localai.model_configs import generate_all_model_configs; generate_all_model_configs()"
```

## 3. Adding a Hardware Profile (`presets.py`)
Add your profile definition to `HARDWARE_PROFILES` in `local_ai_launcher/presets.py`:
```python
"my_gpu_profile": HardwareProfile(
    name="my_gpu_profile",
    vram_budget_mb=12288,
    flash_attention=True,
    compute_precision="fp16",
    kv_cache_precision="q4_0",
    description="Custom profile for 12 GB GPUs",
)
```

## 4. Adding an Agent Harness (`harnesses/`)
1. Create `local_ai_launcher/harnesses/my_agent.py` subclassing `HarnessAdapter`.
2. Implement `launch()`, `is_installed()`, and `setup_config()`.
3. Register your class in `HARNESS_REGISTRY` inside `local_ai_launcher/harnesses/__init__.py`.

## 5. Verification & Testing
Before submitting a pull request, run the full test suite:
```bash
pytest local_ai_launcher/tests
```
All tests must pass with zero warnings or port leaks.
