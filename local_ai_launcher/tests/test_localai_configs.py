import pytest
from pathlib import Path
import yaml

from local_ai_launcher.localai.model_configs import (
    sanitize_filename,
    generate_localai_yaml,
    generate_all_model_configs,
    get_model_backend_mapping
)


def test_sanitize_filename():
    assert sanitize_filename("qwen-38_27b") == "qwen-38_27b"
    assert sanitize_filename("nvidia/Qwen3.6-35B") == "nvidia_Qwen3.6-35B"
    assert sanitize_filename("model*name:v1") == "model_name_v1"


def test_generate_localai_yaml():
    doc = generate_localai_yaml(
        name="test-model",
        backend="blackwell-llama",
        model_path="models/test.gguf",
        context_size=65536,
        gpu_layers=-1,
        parameters={"threads": 8}
    )
    assert doc["name"] == "test-model"
    assert doc["backend"] == "blackwell-llama"
    assert doc["parameters"]["model"] == "models/test.gguf"
    assert doc["parameters"]["threads"] == 8
    assert doc["context_size"] == 65536
    assert doc["gpu_layers"] == -1


def test_generate_all_model_configs(tmp_path):
    out_files = generate_all_model_configs(target_dir=tmp_path)
    assert len(out_files) > 0

    # Verify director model config exists
    director_file = tmp_path / "gemma-4-director.yaml"
    assert director_file.exists()
    data = yaml.safe_load(director_file.read_text(encoding="utf-8"))
    assert data["name"] == "gemma-4-director"
    assert data["gpu_layers"] == 0  # CPU-only to preserve VRAM for diffusion

    mapping = get_model_backend_mapping(models_dir=tmp_path)
    assert "gemma-4-director" in mapping
    assert mapping["gemma-4-director"]["backend"] in ("llama-cpp", "blackwell-llama")

    # Verify primary Huang and Strata configs exist and map correctly
    huang_file = tmp_path / "qwen-38_27b-huang-gsq-think.yaml"
    assert huang_file.exists()
    assert "qwen-38_27b-huang-gsq-think" in mapping
    assert mapping["qwen-38_27b-huang-gsq-think"]["backend"] == "blackwell-llama"

    strata_file = tmp_path / "qwen-38-flash-strata-iq3s.yaml"
    assert strata_file.exists()
    assert "qwen-38-flash-strata-iq3s" in mapping
    assert mapping["qwen-38-flash-strata-iq3s"]["backend"] == "strata"


def test_smart_caching_and_stale_pruning(tmp_path):
    """Verify generate_all_model_configs prunes deleted models and uses hash caching."""
    # 1. Initial generation
    out_files = generate_all_model_configs(target_dir=tmp_path)
    assert (tmp_path / ".source_manifest.hash").exists()

    # 2. Simulate an orphaned/deleted model YAML from past run
    stale_file = tmp_path / "deleted_legacy_model.yaml"
    stale_file.write_text("name: deleted_legacy_model\nbackend: unknown", encoding="utf-8")
    assert stale_file.exists()

    # 3. Running with force=True must prune the stale file
    updated_files = generate_all_model_configs(target_dir=tmp_path, force=True)
    assert not stale_file.exists()
    assert (tmp_path / "gemma-4-director.yaml").exists()

    # 4. Running with force=False returns cached list without error
    cached_files = generate_all_model_configs(target_dir=tmp_path, force=False)
    assert len(cached_files) == len(updated_files)


def test_generate_localai_yaml_bounds_hardening():
    """Verify generate_localai_yaml safely clamps negative context_size and gpu_layers."""
    doc = generate_localai_yaml(
        name="hardening-model",
        backend="llama-cpp",
        model_path="models/test.gguf",
        context_size=-8192,
        gpu_layers=-99
    )
    assert doc["context_size"] >= 512
    # gpu_layers < -1 is clamped to -1 (offload all)
    assert doc["gpu_layers"] == -1

