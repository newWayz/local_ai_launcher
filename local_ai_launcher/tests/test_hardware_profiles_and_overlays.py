import os
import pytest
import tempfile
import configparser
from pathlib import Path
from unittest.mock import patch, MagicMock

from local_ai_launcher.config import (
    config,
    LauncherConfig,
    HARDWARE_PROFILES,
    get_active_hardware_profile,
    MODELS_PRESET_FILE,
    _flatten_overrides,
)
from local_ai_launcher.presets import (
    load_effective_models_parser,
    resolve_effective_models_preset,
    get_candidate_override_file,
)
from local_ai_launcher.localai.model_configs import get_source_manifest_hash
from local_ai_launcher.vram_coordinator import evict_llm_vram


def test_default_hardware_profile():
    """Verify default hardware profile is blackwell_16gb with expected tuning parameters."""
    cfg = LauncherConfig()
    assert cfg.hardware_profile == "blackwell_16gb"
    profile = get_active_hardware_profile(cfg)
    assert profile["name"] == "blackwell_16gb"
    assert profile["vram_ceiling_mb"] == 16384
    assert profile["default_n_cpu_moe"] == 25
    assert profile["strict_vram_eviction"] is True
    assert any(k in profile["engine_binary"].lower() for k in ("blackwell", "huang"))
    cfg_bruno = LauncherConfig()
    cfg_bruno.blackwell_engine = "bruno"
    profile_bruno = get_active_hardware_profile(cfg_bruno)
    assert "blackwell" in profile_bruno["engine_binary"].lower()


def test_hardware_profiles_registry():
    """Verify all defined hardware profiles contain complete and valid specifications."""
    required_keys = {
        "name", "description", "engine_binary", "vram_ceiling_mb",
        "vram_evict_threshold_mb", "strict_vram_eviction", "default_n_cpu_moe",
        "ctk", "ctv", "batch_size", "ubatch_size", "clm_pcore_affinity"
    }
    for name, p in HARDWARE_PROFILES.items():
        assert required_keys.issubset(set(p.keys())), f"Profile '{name}' missing required keys"
        assert p["vram_ceiling_mb"] > 0
        assert p["batch_size"] > 0

    # Verify 24GB profile has opportunistic eviction (strict=False) and 0 MoE offload
    ada = HARDWARE_PROFILES["ada_24gb"]
    assert ada["strict_vram_eviction"] is False
    assert ada["default_n_cpu_moe"] == 0
    assert ada["vram_ceiling_mb"] == 24576


def test_hardware_profile_override_from_toml():
    """Verify [hardware] section in TOML correctly maps to LauncherConfig attributes."""
    raw_toml = {
        "hardware": {
            "profile": "ada_24gb",
            "models_preset": "custom_models.ini",
            "override": "my_override.ini",
        }
    }
    flattened = _flatten_overrides(raw_toml)
    assert flattened["hardware_profile"] == "ada_24gb"
    assert flattened["models_preset_file"] == "custom_models.ini"
    assert flattened["models_local_override_file"] == "my_override.ini"


def test_load_effective_models_parser_no_override(tmp_path):
    """Verify base preset is loaded cleanly when no override file exists."""
    base_ini = tmp_path / "base.ini"
    base_ini.write_text("[*]\nthreads = 16\n\n[model-a]\nctx-size = 32768\n", encoding="utf-8")

    parser = load_effective_models_parser(base_file=base_ini, override_file=tmp_path / "nonexistent.ini")
    assert parser.has_section("*")
    assert parser.get("*", "threads") == "16"
    assert parser.get("model-a", "ctx-size") == "32768"


def test_load_effective_models_parser_with_override(tmp_path):
    """Verify local override values take precedence while preserving untouched base values."""
    base_ini = tmp_path / "base.ini"
    base_ini.write_text(
        "[*]\nthreads = 16\nctk = q8_0\n\n"
        "[tiel-coder-35b-think]\nn-cpu-moe = 25\nctx-size = 262000\nreasoning = on\n",
        encoding="utf-8"
    )

    override_ini = tmp_path / "models.local.ini"
    override_ini.write_text(
        "[*]\nthreads = 32\n\n"
        "[tiel-coder-35b-think]\nn-cpu-moe = 0\n\n"
        "[custom-model]\nctx-size = 65536\n",
        encoding="utf-8"
    )

    parser = load_effective_models_parser(base_file=base_ini, override_file=override_ini)

    # 1. Global section [*] has overridden threads and preserved ctk
    assert parser.get("*", "threads") == "32"
    assert parser.get("*", "ctk") == "q8_0"

    # 2. Existing model has overridden n-cpu-moe but preserved ctx-size and reasoning
    assert parser.get("tiel-coder-35b-think", "n-cpu-moe") == "0"
    assert parser.get("tiel-coder-35b-think", "ctx-size") == "262000"
    assert parser.get("tiel-coder-35b-think", "reasoning") == "on"

    # 3. New section from override was cleanly added
    assert parser.has_section("custom-model")
    assert parser.get("custom-model", "ctx-size") == "65536"


def test_load_effective_models_parser_malformed_syntax(tmp_path):
    """Verify malformed override syntax does not crash and safely falls back to base preset."""
    base_ini = tmp_path / "base.ini"
    base_ini.write_text("[*]\nthreads = 16\n", encoding="utf-8")

    corrupted_ini = tmp_path / "corrupted.ini"
    corrupted_ini.write_text("This is not valid INI syntax without section headers :(", encoding="utf-8")

    # Should not raise exception; must log warning and return base parser
    parser = load_effective_models_parser(base_file=base_ini, override_file=corrupted_ini)
    assert parser.has_section("*")
    assert parser.get("*", "threads") == "16"


def test_resolve_effective_models_preset_atomic_file(tmp_path):
    """Verify resolve_effective_models_preset writes a consolidated atomic file when override exists."""
    base_ini = tmp_path / "base.ini"
    base_ini.write_text("[*]\nthreads = 16\n", encoding="utf-8")

    override_ini = tmp_path / "models.local.ini"
    override_ini.write_text("[*]\nthreads = 24\n", encoding="utf-8")

    effective_path, is_overridden = resolve_effective_models_preset(base_file=base_ini, override_file=override_ini)
    assert is_overridden is True
    assert effective_path.is_file()
    assert effective_path.name == "resolved_models.ini"

    # Verify contents of generated resolved file
    cp = configparser.ConfigParser(strict=False, interpolation=None)
    cp.read(str(effective_path), encoding="utf-8")
    assert cp.get("*", "threads") == "24"


def test_manifest_hash_sensitive_to_profile():
    """Verify changing hardware profile changes source manifest hash for LocalAI regeneration."""
    with patch.object(config, "hardware_profile", "blackwell_16gb"):
        hash_blackwell = get_source_manifest_hash()

    with patch.object(config, "hardware_profile", "ada_24gb"):
        hash_ada = get_source_manifest_hash()

    assert hash_blackwell != hash_ada


def test_vram_coordinator_profile_aware_eviction():
    """Verify 24GB opportunistic profile skips eviction when enough VRAM is free, but 16GB strictly evicts."""
    # Mock VRAM info to have 16GB free (enough for 12GB diffusion)
    mock_vram = {"available": True, "allocated_mb": 4000, "free_mb": 16000, "total_mb": 24576}

    # 1. On 24GB profile with strict_vram_eviction=False, eviction is skipped
    with patch("local_ai_launcher.vram_coordinator.get_vram_info", return_value=mock_vram), \
         patch("local_ai_launcher.config.config.hardware_profile", "ada_24gb"), \
         patch("urllib.request.urlopen") as mock_url:
        result = evict_llm_vram(force=False, required_mb=12288)
        assert result is True
        mock_url.assert_not_called()

    # 2. On 16GB profile with strict_vram_eviction=True, eviction is ALWAYS invoked
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.__enter__.return_value = mock_resp
    with patch("local_ai_launcher.vram_coordinator.get_vram_info", return_value=mock_vram), \
         patch("local_ai_launcher.config.config.hardware_profile", "blackwell_16gb"), \
         patch("urllib.request.urlopen", return_value=mock_resp) as mock_url, \
         patch("time.sleep"):
        result = evict_llm_vram(force=False, required_mb=12288)
        assert result is True
        mock_url.assert_called_once()


def test_resolve_effective_models_preset_sanitization(tmp_path):
    """Verify resolve_effective_models_preset strips unsupported keys and excludes non-llama sections."""
    base_ini = tmp_path / "base.ini"
    base_ini.write_text(
        "[*]\nthreads = 16\n\n"
        "[model-llama]\n"
        "model = models/llama-server/test.gguf\n"
        "ctx-size = 32768\n"
        "kv-stream-arena-mib = 2240\n"
        "kv-stream-auxiliary-layers = 1\n\n"
        "[qwen-strata-moe]\n"
        "engine = strata\n"
        "model = models/strata/test.gguf\n"
        "mtp = models/strata/rt\n",
        encoding="utf-8"
    )

    effective_path, is_overridden = resolve_effective_models_preset(base_file=base_ini)
    assert is_overridden is False
    assert effective_path.is_file()

    cp = configparser.ConfigParser(strict=False, interpolation=None)
    cp.read(str(effective_path), encoding="utf-8")

    # model-llama kept, but custom kv-stream options stripped
    assert cp.has_section("model-llama")
    assert cp.get("model-llama", "ctx-size") == "32768"
    assert not cp.has_option("model-llama", "kv-stream-arena-mib")
    assert not cp.has_option("model-llama", "kv-stream-auxiliary-layers")

    # strata non-llama engine section excluded from llama-server preset
    assert not cp.has_section("qwen-strata-moe")


def test_production_models_preset_sanitized():
    """Verify the production models preset resolves cleanly without invalid llama-server options."""
    effective_path, _ = resolve_effective_models_preset()
    assert effective_path.is_file()

    cp = configparser.ConfigParser(strict=False, interpolation=None)
    cp.read(str(effective_path), encoding="utf-8")

    from local_ai_launcher.presets import UNRECOGNIZED_LLAMA_SERVER_KEYS
    for s in cp.sections():
        assert cp.get(s, "engine", fallback="").lower() != "strata"
        for k in UNRECOGNIZED_LLAMA_SERVER_KEYS:
            assert not cp.has_option(s, k), f"Option '{k}' found in section '{s}'"
        for opt in cp.options(s):
            assert not opt.startswith("kv-stream-"), f"kv-stream option '{opt}' found in section '{s}'"

