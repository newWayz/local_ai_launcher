import pytest
import yaml
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from local_ai_launcher.config import AI_WORKSPACE_DIR, MODELS_PRESET_FILE, FREETOKENS_MODELS_PRESET_FILE
from local_ai_launcher.localai.model_configs import generate_all_model_configs, get_model_backend_mapping
from local_ai_launcher.backends.tabbyapi import resolve_exl3_model_folder, setup_tabbyapi_config
from local_ai_launcher.shims.blackwell_llama_shim import BlackwellLlamaServicer
from local_ai_launcher.shims.huang_llama_shim import HuangLlamaServicer
from local_ai_launcher.shims.bruno_llama_shim import BrunoLlamaServicer
from local_ai_launcher.shims.strata_grpc_shim import StrataServicer
from local_ai_launcher.shims.tabby_grpc_shim import TabbyAPIServicer
from local_ai_launcher.shims.freetoken_grpc_shim import FreeTokenServicer
from local_ai_launcher.localai.proto import backend_pb2


def test_models_ini_global_inheritance_and_overwrites(tmp_path):
    """Verify [*] global settings are inherited and per-model settings overwrite them."""
    yamls = generate_all_model_configs(target_dir=tmp_path, force=True)
    assert len(yamls) > 0

    # 1. Test Qwen 3.8 Blackwell Think model
    bw_yaml = tmp_path / "qwen-38_27b-blackwell-256k-think.yaml"
    assert bw_yaml.exists(), "Blackwell think model config YAML must exist"
    
    with open(bw_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # Global [*] inherited
    assert data["backend"] == "blackwell-llama"
    assert data["parameters"]["flash_attn"] == "on"

    # Model-specific overrides over [*]
    assert data["parameters"]["threads"] == 8
    assert data["parameters"]["threads_batch"] == 8
    assert data["parameters"]["cache_reuse"] == 0  # Overrides [*] 256 to disable unsupported DeltaNet KV shifting
    assert data["parameters"]["ctk"] == "q4_0"  # Overrides [*] q8_0
    assert data["parameters"]["ctv"] == "q4_0"  # Overrides [*] q8_0
    assert data["parameters"]["batch_size"] == 1024  # Overrides [*] 4096
    assert data["parameters"]["ubatch_size"] == 512  # Overrides [*] 4096

    # Model-specific speculative decoding & checkpoints (if enabled in models.ini)
    if "spec_type" in data["parameters"]:
        assert data["parameters"]["spec_type"] == "draft-mtp"
        assert int(data["parameters"]["spec_draft_n_max"]) >= 1
    assert data["parameters"]["ctx_checkpoints"] == 32
    assert data["context_size"] == 262000

    # Sampling parameters
    assert "sampling" in data
    assert data["sampling"]["temperature"] == 0.6
    assert data["sampling"]["top_p"] == 0.95
    assert data["sampling"]["top_k"] == 20
    assert data["sampling"]["min_p"] == 0.0

    # Reasoning parameters
    assert "reasoning" in data
    assert data["reasoning"]["enabled"] is True
    assert data["reasoning"]["format"] == "deepseek"
    assert data["reasoning"]["budget_tokens"] == 8192
    assert "Wait, I'm overthinking this" in data["reasoning"]["budget_message"]

    # 2. Test Qwen 3.8 Huang GSQ Adaptive-KV Think model (primary default model)
    huang_yaml = tmp_path / "qwen-38_27b-huang-gsq-think.yaml"
    assert huang_yaml.exists(), "Huang GSQ think model config YAML must exist"
    with open(huang_yaml, "r", encoding="utf-8") as f:
        huang_data = yaml.safe_load(f)

    assert huang_data["backend"] == "blackwell-llama"
    assert huang_data["context_size"] == 262000
    assert huang_data["parameters"]["threads"] == 8
    assert huang_data["parameters"]["threads_batch"] == 8
    assert huang_data["parameters"]["batch_size"] == 512
    assert huang_data["parameters"]["ubatch_size"] == 256
    assert huang_data["parameters"]["cache_reuse"] == 0
    assert huang_data["parameters"]["ctk"] == "q8_0"
    assert huang_data["parameters"]["ctv"] == "q4_0"
    assert huang_data["parameters"]["kv_stream_arena_mib"] == 2240
    assert huang_data["parameters"]["kv_stream_auxiliary_layers"] == 1
    assert huang_data["parameters"]["spec_type"] == "draft-mtp"
    assert huang_data["parameters"]["spec_draft_n_max"] == 3
    assert huang_data["parameters"]["spec_draft_p_min"] == pytest.approx(0.20)
    assert huang_data["parameters"]["spec_draft_p_split"] == pytest.approx(0.1)
    assert huang_data["parameters"]["ctx_checkpoints"] == 32
    assert huang_data["parameters"]["no_mmproj"] is True
    assert huang_data["sampling"]["temperature"] == 0.6
    assert huang_data["sampling"]["top_p"] == 0.95
    assert huang_data["sampling"]["top_k"] == 20
    assert huang_data["sampling"]["min_p"] == 0.0
    assert huang_data["reasoning"]["enabled"] is True
    assert huang_data["reasoning"]["budget_tokens"] == 8192

    # 3. Test Qwen 3.8 Strata 125B MoE model
    strata_yaml = tmp_path / "qwen-38-flash-strata-iq3s.yaml"
    if strata_yaml.exists():
        with open(strata_yaml, "r", encoding="utf-8") as f:
            strata_data = yaml.safe_load(f)
        assert strata_data["backend"] == "strata"
        assert strata_data["context_size"] == 262144
        assert strata_data["sampling"]["temperature"] == 0.6
        assert strata_data["sampling"]["top_p"] == 0.95
        assert strata_data["sampling"]["top_k"] == 20
        assert strata_data["sampling"]["min_p"] == 0.05
        assert strata_data["reasoning"]["enabled"] is True
        assert strata_data["reasoning"]["effort"] == "xhigh"
        assert strata_data["reasoning"]["budget_tokens"] == 16384

    # 4. Test non-Qwen transformer model inherits global cache_reuse = 256
    tiel_yaml = tmp_path / "tiel-coder-35b-think.yaml"
    if tiel_yaml.exists():
        with open(tiel_yaml, "r", encoding="utf-8") as f:
            tiel_data = yaml.safe_load(f)
        assert tiel_data["parameters"]["cache_reuse"] == 256


def test_tabbyapi_author_directory_resolution(tmp_path, monkeypatch):
    """Verify resolve_exl3_model_folder locates models nested in author directories."""
    monkeypatch.setattr("local_ai_launcher.backends.tabbyapi.AI_WORKSPACE_DIR", tmp_path)
    m_dir = tmp_path / "models" / "exl3" / "turboderp" / "Qwen3.8-27B-exl3-SC-3.0bpw-H4"
    m_dir.mkdir(parents=True, exist_ok=True)
    (m_dir / "model.safetensors").write_text("dummy", encoding="utf-8")

    resolved = resolve_exl3_model_folder("Qwen3.8-27B-exl3-SC-3.0bpw-H4")
    assert resolved is not None
    assert "Qwen3.8-27B-exl3-SC-3.0bpw-H4" in resolved.name
    assert "turboderp" in resolved.as_posix()

    # Also works when prefixed with author
    resolved_full = resolve_exl3_model_folder("turboderp/Qwen3.8-27B-exl3-SC-3.0bpw-H4")
    assert resolved_full is not None
    assert resolved_full == resolved


def test_tabbyapi_config_and_sampler_overrides(tmp_path, monkeypatch):
    """Verify setup_tabbyapi_config copies sampler_config.yml and sets override_preset."""
    test_tabby_dir = tmp_path / "tabbyapi"
    test_tabby_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("local_ai_launcher.backends.tabbyapi.AI_WORKSPACE_DIR", tmp_path)

    # Create dummy models/exl3/turboderp/Qwen3.8-27B-exl3-SC-3.0bpw-H4
    m_dir = tmp_path / "models" / "exl3" / "turboderp" / "Qwen3.8-27B-exl3-SC-3.0bpw-H4"
    m_dir.mkdir(parents=True, exist_ok=True)
    (m_dir / "model.safetensors").write_text("dummy", encoding="utf-8")
    (m_dir / "tabby_config.yml").write_text("model:\n  max_seq_len: 65535\n  cache_size: 65535\n", encoding="utf-8")
    (m_dir / "sampler_config.yml").write_text("temperature:\n  override: 0.6\ntop_k:\n  override: 20\n", encoding="utf-8")

    setup_tabbyapi_config("Qwen3.8-27B-exl3-SC-3.0bpw-H4")

    # Verify 256-rounding on tabby_config.yml
    updated_tabby = yaml.safe_load((m_dir / "tabby_config.yml").read_text(encoding="utf-8"))
    assert updated_tabby["model"]["max_seq_len"] % 256 == 0
    assert updated_tabby["model"]["cache_size"] % 256 == 0

    # Verify sampler_config.yml copied to active_model_preset.yml
    preset_file = test_tabby_dir / "sampler_overrides" / "active_model_preset.yml"
    assert preset_file.exists()
    preset_data = yaml.safe_load(preset_file.read_text(encoding="utf-8"))
    assert preset_data["temperature"]["override"] == 0.6

    # Verify config.yml updated with override_preset
    cfg_yml = test_tabby_dir / "config.yml"
    assert cfg_yml.exists()
    cfg_content = cfg_yml.read_text(encoding="utf-8")
    assert "override_preset: active_model_preset" in cfg_content


def test_exl3_yaml_generation_fidelity(tmp_path):
    """Verify generated EXL3 YAML contains detailed tabby and sampler config parameters."""
    yamls = generate_all_model_configs(target_dir=tmp_path, force=True)
    qwen_yaml = tmp_path / "Qwen3.8-27B-exl3-SC-3.0bpw-H4.yaml"
    assert qwen_yaml.exists()

    with open(qwen_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert data["backend"] == "tabbyapi-exl3"
    assert data["parameters"]["cache_mode"] == "8,8"
    assert data["parameters"]["draft_model"]["draft_mode"] == "mtp"
    assert data["sampling"]["temperature"] == 0.6
    assert data["sampling"]["top_k"] == 20
    assert data["sampling"]["min_p"] == 0.05
    assert data["reasoning"]["enabled"] is True
    assert data["reasoning"]["budget_tokens"] == 4096


def test_freetoken_yaml_generation_fidelity(tmp_path):
    """Verify FreeToken YAML preserves all specialized flags from freetokens_models.ini."""
    yamls = generate_all_model_configs(target_dir=tmp_path, force=True)
    ft_yaml = tmp_path / "nvidia_Qwen3.6-35B-A3B-NVFP4.yaml"
    assert ft_yaml.exists()

    with open(ft_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert data["backend"] == "freetoken-nvfp4"
    assert data["parameters"]["max_seq_len_override"] == 131072
    assert data["parameters"]["memory_ratio"] == 0.88
    assert data["parameters"]["cuda_graph_max_bs"] == 1
    assert data["parameters"]["moe_backend"] == "offload"
    assert data["parameters"]["nvfp4_backend"] == "auto"
    assert data["reasoning"]["parser"] == "qwen3"


def test_shims_forward_all_sampling_and_reasoning_parameters():
    """Verify gRPC shims forward sampling, penalties, and reasoning to backend HTTP engines."""
    # 1. Blackwell Llama Shim
    bw_servicer = BlackwellLlamaServicer()
    bw_req = backend_pb2.PredictOptions(
        Temperature=0.6,
        TopP=0.95,
        TopK=20,
        MinP=0.05,
        Penalty=1.05,
        PresencePenalty=1.5,
        FrequencyPenalty=0.2,
        Seed=424242,
        Tokens=2048,
        Metadata={
            "reasoning_budget": "8192",
            "reasoning_format": "deepseek",
            "reasoning_effort": "high"
        }
    )
    bw_payload = bw_servicer._build_request_payload(bw_req)
    assert bw_payload["temperature"] == pytest.approx(0.6)
    assert bw_payload["top_p"] == pytest.approx(0.95)
    assert bw_payload["top_k"] == 20
    assert bw_payload["min_p"] == pytest.approx(0.05)
    assert bw_payload["repeat_penalty"] == pytest.approx(1.05)
    assert bw_payload["presence_penalty"] == pytest.approx(1.5)
    assert bw_payload["frequency_penalty"] == pytest.approx(0.2)
    assert bw_payload["seed"] == 424242
    assert bw_payload["max_tokens"] == 2048
    assert bw_payload["reasoning_budget"] == 8192
    assert bw_payload["reasoning_format"] == "deepseek"
    assert bw_payload["reasoning_effort"] == "high"

    # 2. TabbyAPI Shim
    tabby_servicer = TabbyAPIServicer()
    tabby_payload = tabby_servicer._build_request_payload(bw_req)
    assert tabby_payload["temperature"] == pytest.approx(0.6)
    assert tabby_payload["top_p"] == pytest.approx(0.95)
    assert tabby_payload["top_k"] == 20
    assert tabby_payload["min_p"] == pytest.approx(0.05)
    assert tabby_payload["repetition_penalty"] == pytest.approx(1.05)
    assert tabby_payload["reasoning_budget_tokens"] == 8192
    assert tabby_payload["template_vars"] == {"reasoning_effort": "high"}

    # 3. FreeToken Shim
    ft_servicer = FreeTokenServicer()
    ft_payload = ft_servicer._build_request_payload(bw_req)
    assert ft_payload["temperature"] == pytest.approx(0.6)
    assert ft_payload["top_p"] == pytest.approx(0.95)
    assert ft_payload["top_k"] == 20
    assert ft_payload["min_p"] == pytest.approx(0.05)
    assert ft_payload["repetition_penalty"] == pytest.approx(1.05)

    # 4. Huang Llama Shim (Raymond Huang Adaptive-KV V2)
    huang_servicer = HuangLlamaServicer()
    huang_payload = huang_servicer._build_request_payload(bw_req)
    assert huang_payload["temperature"] == pytest.approx(0.6)
    assert huang_payload["top_p"] == pytest.approx(0.95)
    assert huang_payload["top_k"] == 20
    assert huang_payload["min_p"] == pytest.approx(0.05)
    assert huang_payload["repeat_penalty"] == pytest.approx(1.05)
    assert huang_payload["presence_penalty"] == pytest.approx(1.5)
    assert huang_payload["frequency_penalty"] == pytest.approx(0.2)
    assert huang_payload["seed"] == 424242
    assert huang_payload["max_tokens"] == 2048
    assert huang_payload["reasoning_budget"] == 8192
    assert huang_payload["reasoning_format"] == "deepseek"
    assert huang_payload["reasoning_effort"] == "high"

    # 5. Bruno Llama Shim (Bruno Passini 256K Hybrid-KV)
    bruno_servicer = BrunoLlamaServicer()
    bruno_payload = bruno_servicer._build_request_payload(bw_req)
    assert bruno_payload["temperature"] == pytest.approx(0.6)
    assert bruno_payload["top_p"] == pytest.approx(0.95)
    assert bruno_payload["top_k"] == 20
    assert bruno_payload["min_p"] == pytest.approx(0.05)
    assert bruno_payload["repeat_penalty"] == pytest.approx(1.05)
    assert bruno_payload["presence_penalty"] == pytest.approx(1.5)
    assert bruno_payload["frequency_penalty"] == pytest.approx(0.2)
    assert bruno_payload["seed"] == 424242
    assert bruno_payload["max_tokens"] == 2048
    assert bruno_payload["reasoning_budget"] == 8192
    assert bruno_payload["reasoning_format"] == "deepseek"
    assert bruno_payload["reasoning_effort"] == "high"

    # 6. Strata Shim (125B MoE Native Engine)
    strata_servicer = StrataServicer()
    strata_payload = strata_servicer._build_request_payload(bw_req)
    assert strata_payload["temperature"] == pytest.approx(0.6)
    assert strata_payload["top_p"] == pytest.approx(0.95)
    assert strata_payload["top_k"] == 20
    assert strata_payload["min_p"] == pytest.approx(0.05)
    assert strata_payload["repeat_penalty"] == pytest.approx(1.05)
    assert strata_payload["presence_penalty"] == pytest.approx(1.5)
    assert strata_payload["frequency_penalty"] == pytest.approx(0.2)
    assert strata_payload["seed"] == 424242
    assert strata_payload["max_tokens"] == 2048
    assert strata_payload["reasoning_budget"] == 8192
    assert strata_payload["reasoning_format"] == "deepseek"
    assert strata_payload["reasoning_effort"] == "high"


def test_strata_models_json_decoupling(tmp_path):
    """Verify strata_models.json decoupling generates LocalAI YAMLs and resolves aliases without models.ini."""
    from local_ai_launcher.backends.strata import load_strata_model_registry, resolve_strata_model_files
    from local_ai_launcher.localai.model_configs import generate_all_model_configs

    # 1. Registry loads from strata_models.json
    registry = load_strata_model_registry()
    assert "qwen-38-flash-strata-iq3s" in registry
    assert "qwen-38-flash-strata-iq3s-vision" in registry
    assert "swift-qwen-38-flash-strata-iq3s" in registry
    assert "swift-qwen-38-flash-strata-iq3s-vision" in registry

    # 2. Alias resolution works without models.ini
    res = resolve_strata_model_files("strata-iq3s")
    assert res["first_shard"] is not None
    assert "Qwen3.8-Flash-Next" in res["first_shard"].name
    assert res.get("chat_template_file") is not None
    assert "chat_template" in res["chat_template_file"]

    res_vision = resolve_strata_model_files("strata-vision")
    assert res_vision["first_shard"] is not None
    assert res_vision.get("chat_template_file") is not None

    res_swift = resolve_strata_model_files("swift-strata-iq3s")
    assert res_swift["first_shard"] is not None
    assert "Swift-Qwen3.8-Flash-Next" in res_swift["first_shard"].name
    assert res_swift.get("chat_template_file") is not None

    res_swift_vis = resolve_strata_model_files("swift-vision")
    assert res_swift_vis["first_shard"] is not None
    assert res_swift_vis.get("mmproj") is not None
    assert "mmproj-Swift" in res_swift_vis["mmproj"]

    # 3. YAML generation creates primary model and aliases
    yamls = generate_all_model_configs(target_dir=tmp_path, force=True)
    yaml_names = {y.stem for y in yamls}
    assert "qwen-38-flash-strata-iq3s" in yaml_names
    assert "strata-iq3s" in yaml_names
    assert "qwen-38-flash-strata-iq3" in yaml_names
    assert "qwen-38-flash-strata-iq3s-vision" in yaml_names
    assert "swift-qwen-38-flash-strata-iq3s" in yaml_names
    assert "swift-strata-iq3s" in yaml_names
    assert "swift-qwen-38-flash-strata-iq3s-vision" in yaml_names
    assert "swift-vision" in yaml_names

    primary_yaml = tmp_path / "swift-qwen-38-flash-strata-iq3s.yaml"
    with open(primary_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert data["backend"] == "strata"
    assert data["context_size"] == 262144
    assert data["sampling"]["temperature"] == 0.6
    assert data["sampling"]["top_p"] == 0.95
    assert data["sampling"]["top_k"] == 20
    assert data["sampling"]["min_p"] == 0.05
    assert data["reasoning"]["enabled"] is True
    assert data["reasoning"]["budget_tokens"] == 16384


def test_launcher_config_strata_use_custom_chat_template(tmp_path):
    """Verify strata_use_custom_chat_template configuration default and overrides."""
    from local_ai_launcher.config import LauncherConfig, _flatten_overrides

    # 1. Default is True
    cfg = LauncherConfig()
    assert cfg.strata_use_custom_chat_template is True

    # 2. _flatten_overrides parses [strata] table
    flat = _flatten_overrides({"strata": {"use_custom_chat_template": False}})
    assert flat.get("strata_use_custom_chat_template") is False

    # 3. Direct kwarg override
    cfg_direct = LauncherConfig(strata_use_custom_chat_template=False)
    assert cfg_direct.strata_use_custom_chat_template is False

    # 4. Override via TOML file
    toml_path = tmp_path / "launcher.toml"
    toml_path.write_text("[strata]\nuse_custom_chat_template = false\n", encoding="utf-8")
    cfg_file = LauncherConfig(config_file=str(toml_path))
    assert cfg_file.strata_use_custom_chat_template is False


def test_launcher_config_claude_context_and_compaction_settings(tmp_path):
    """Verify claude context and compaction window settings in LauncherConfig and TOML overrides."""
    from local_ai_launcher.config import LauncherConfig, _flatten_overrides

    # 1. Defaults
    cfg = LauncherConfig()
    assert cfg.claude_auto_compact_ratio == pytest.approx(0.91)
    assert cfg.claude_max_context_tokens is None
    assert cfg.claude_auto_compact_window is None

    # 2. _flatten_overrides handles [claude] table
    flat = _flatten_overrides({
        "claude": {
            "auto_compact_ratio": 0.88,
            "max_context_tokens": 262144,
            "auto_compact_window": 230000,
        }
    })
    assert flat["claude_auto_compact_ratio"] == pytest.approx(0.88)
    assert flat["claude_max_context_tokens"] == 262144
    assert flat["claude_auto_compact_window"] == 230000

    # 3. TOML loading
    toml_path = tmp_path / "launcher.toml"
    toml_path.write_text(
        "[claude]\n"
        "auto_compact_ratio = 0.85\n"
        "max_context_tokens = 262144\n"
        "auto_compact_window = 220000\n",
        encoding="utf-8"
    )
    cfg_file = LauncherConfig(config_file=str(toml_path))
    assert cfg_file.claude_auto_compact_ratio == pytest.approx(0.85)
    assert cfg_file.claude_max_context_tokens == 262144
    assert cfg_file.claude_auto_compact_window == 220000


def test_launcher_config_claude_desktop_context_and_compaction_settings(tmp_path):
    """Verify claude desktop context and compaction window settings in LauncherConfig and TOML overrides."""
    from local_ai_launcher.config import LauncherConfig, _flatten_overrides
    from local_ai_launcher.harnesses.claude import resolve_claude_context_and_compact

    # 1. Defaults
    cfg = LauncherConfig()
    assert cfg.claude_desktop_auto_compact_ratio is None
    assert cfg.claude_desktop_max_context_tokens is None
    assert cfg.claude_desktop_auto_compact_window is None

    # 2. _flatten_overrides handles [claude_desktop] table
    flat = _flatten_overrides({
        "claude_desktop": {
            "auto_compact_ratio": 0.82,
            "max_context_tokens": 131072,
            "auto_compact_window": 105000,
        }
    })
    assert flat["claude_desktop_auto_compact_ratio"] == pytest.approx(0.82)
    assert flat["claude_desktop_max_context_tokens"] == 131072
    assert flat["claude_desktop_auto_compact_window"] == 105000

    # 3. TOML loading with dedicated [claude_desktop] table
    toml_path = tmp_path / "launcher.toml"
    toml_path.write_text(
        "[claude]\n"
        "auto_compact_ratio = 0.9155\n"
        "max_context_tokens = 262144\n"
        "auto_compact_window = 240000\n\n"
        "[claude_desktop]\n"
        "auto_compact_ratio = 0.80\n"
        "max_context_tokens = 65536\n"
        "auto_compact_window = 52000\n",
        encoding="utf-8"
    )
    cfg_file = LauncherConfig(config_file=str(toml_path))
    assert cfg_file.claude_desktop_auto_compact_ratio == pytest.approx(0.80)
    assert cfg_file.claude_desktop_max_context_tokens == 65536
    assert cfg_file.claude_desktop_auto_compact_window == 52000

    # General claude settings are isolated
    assert cfg_file.claude_max_context_tokens == 262144
    assert cfg_file.claude_auto_compact_window == 240000

    # 4. Desktop context resolution: returns None for MCP mode, resolves for Gateway mode
    mcp_ctx, mcp_win = resolve_claude_context_and_compact("custom-model", is_desktop=True, desktop_mode="mcp")
    assert mcp_ctx is None
    assert mcp_win is None

    gw_ctx, gw_win = resolve_claude_context_and_compact("strata-iq3s", is_desktop=True, desktop_mode="gateway")
    assert gw_ctx == 262144
    assert gw_win == 239000


def test_strata_kv_resident_and_vram_reserve_tuning(tmp_path):
    """Verify kv_resident and vram_reserve_mib are assembled with clamp guardrails in build_strata_config."""
    from local_ai_launcher.backends.strata import build_strata_config

    pack_dir = tmp_path / "dummy_pack"
    pack_dir.mkdir(parents=True, exist_ok=True)
    first_shard = tmp_path / "dummy-00001-of-00002.gguf"
    first_shard.touch()

    # 1. Verify kv_resident clamp (minimum 20480 cells per QSA layer) and vram_reserve_mib
    cfg_path = build_strata_config(
        model_name="test_dummy_stream",
        first_shard=first_shard,
        ple_shard=None,
        pack_dir=pack_dir,
        mtp_dir=None,
        port=9999,
        kv_resident=8192,  # Below minimum -> must clamp to 20480
        vram_reserve_mib=700
    )
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    args = data["args"]
    assert "--kv-resident" in args
    assert args[args.index("--kv-resident") + 1] == "20480"
    assert "--vram-reserve-mib" in args
    assert args[args.index("--vram-reserve-mib") + 1] == "700"

    # 2. Verify explicit valid window (e.g. 32768)
    cfg_path2 = build_strata_config(
        model_name="test_dummy_stream2",
        first_shard=first_shard,
        ple_shard=None,
        pack_dir=pack_dir,
        mtp_dir=None,
        port=9999,
        kv_resident=32768
    )
    with open(cfg_path2, "r", encoding="utf-8") as f:
        data2 = json.load(f)
    args2 = data2["args"]
    assert "--kv-resident" in args2
    assert args2[args2.index("--kv-resident") + 1] == "32768"
    assert "--vram-reserve-mib" not in args2


def test_strata_system_ram_kv_cache_toggle(tmp_path):
    """Verify system_ram_kv_cache toggle works cleanly across config and build_strata_config."""
    from local_ai_launcher.config import LauncherConfig, _flatten_overrides
    from local_ai_launcher.backends.strata import build_strata_config

    # 1. Config TOML reading and flattening under [strata]
    flat = _flatten_overrides({"strata": {"system_ram_kv_cache": True}})
    assert flat.get("strata_system_ram_kv_cache") is True

    flat_alias = _flatten_overrides({"strata": {"ram_only_kv_cache": True}})
    assert flat_alias.get("strata_system_ram_kv_cache") is True

    flat_top = _flatten_overrides({"system_ram_kv_cache": True})
    assert flat_top.get("strata_system_ram_kv_cache") is True

    toml_path = tmp_path / "launcher.toml"
    toml_path.write_text("[strata]\nsystem_ram_kv_cache = true\n", encoding="utf-8")
    cfg_file = LauncherConfig(config_file=str(toml_path))
    assert cfg_file.strata_system_ram_kv_cache is True

    # Default is False (standard behavior)
    cfg_default = LauncherConfig()
    assert cfg_default.strata_system_ram_kv_cache is False

    pack_dir = tmp_path / "dummy_pack"
    pack_dir.mkdir(parents=True, exist_ok=True)
    first_shard = tmp_path / "dummy-00001-of-00002.gguf"
    first_shard.touch()

    # 2. When system_ram_kv_cache is ON (True):
    # Streaming is activated, --kv-resident defaults to 20480 (safe min), and vram_reserve defaults to 700
    cfg_on_path = build_strata_config(
        model_name="test_sys_ram_on",
        first_shard=first_shard,
        ple_shard=None,
        pack_dir=pack_dir,
        mtp_dir=None,
        port=9998,
        system_ram_kv_cache=True
    )
    with open(cfg_on_path, "r", encoding="utf-8") as f:
        data_on = json.load(f)
    assert data_on["system_ram_kv_cache"] is True
    assert "--kv-resident" in data_on["args"]
    assert data_on["args"][data_on["args"].index("--kv-resident") + 1] == "20480"
    assert "--vram-reserve-mib" in data_on["args"]
    assert data_on["args"][data_on["args"].index("--vram-reserve-mib") + 1] == "700"

    # 3. When system_ram_kv_cache is OFF (False) - Standard Behavior:
    # No --kv-resident flag injected; full context in VRAM
    cfg_off_path = build_strata_config(
        model_name="test_sys_ram_off",
        first_shard=first_shard,
        ple_shard=None,
        pack_dir=pack_dir,
        mtp_dir=None,
        port=9997,
        system_ram_kv_cache=False
    )
    with open(cfg_off_path, "r", encoding="utf-8") as f:
        data_off = json.load(f)
    assert data_off["system_ram_kv_cache"] is False
    assert "--kv-resident" not in data_off["args"]
    assert "--vram-reserve-mib" not in data_off["args"]


def test_claude_permission_mode_fidelity(tmp_path):
    """Verify claude_permission_mode and claude_desktop_permission_mode TOML parsing and defaults."""
    from local_ai_launcher.config import LauncherConfig, _flatten_overrides

    # 1. Defaults: Claude CLI defaults to 'smart', desktop defaults to None (inherits 'smart' at runtime)
    cfg_default = LauncherConfig()
    assert cfg_default.claude_permission_mode == "smart"
    assert cfg_default.claude_permission_mode_explicit is False
    assert cfg_default.claude_desktop_permission_mode is None
    assert cfg_default.claude_desktop_permission_mode_explicit is False

    # 2. Flatten overrides under [claude]
    flat_cli_smart = _flatten_overrides({"claude": {"permission_mode": "smart"}})
    assert flat_cli_smart["claude_permission_mode"] == "smart"
    assert flat_cli_smart["claude_permission_mode_explicit"] is True

    flat_cli_yolo = _flatten_overrides({"claude": {"permissions_mode": "yolo"}})
    assert flat_cli_yolo["claude_permission_mode"] == "yolo"

    flat_cli_default = _flatten_overrides({"claude": {"permission": "default"}})
    assert flat_cli_default["claude_permission_mode"] == "default"

    # 3. Flatten overrides under [claude_desktop]
    flat_desk_smart = _flatten_overrides({"claude_desktop": {"permission_mode": "smart"}})
    assert flat_desk_smart["claude_desktop_permission_mode"] == "smart"
    assert flat_desk_smart["claude_desktop_permission_mode_explicit"] is True

    flat_desk_yolo = _flatten_overrides({"claude_desktop": {"permissions": "yolo"}})
    assert flat_desk_yolo["claude_desktop_permission_mode"] == "yolo"

    # 4. End-to-end TOML loading via LauncherConfig
    toml_path = tmp_path / "launcher.toml"
    toml_path.write_text(
        """
[claude]
permission_mode = "yolo"

[claude_desktop]
permission_mode = "default"
""",
        encoding="utf-8",
    )
    cfg_loaded = LauncherConfig(config_file=str(toml_path))
    assert cfg_loaded.claude_permission_mode == "yolo"
    assert cfg_loaded.claude_permission_mode_explicit is True
    assert cfg_loaded.claude_desktop_permission_mode == "default"
    assert cfg_loaded.claude_desktop_permission_mode_explicit is True


def test_claude_thinking_mode_and_contract_fidelity(tmp_path):
    """Verify claude_thinking_mode and claude_offline_inject_contract TOML parsing and defaults."""
    from local_ai_launcher.config import LauncherConfig, _flatten_overrides

    # 1. Defaults
    cfg_default = LauncherConfig()
    assert cfg_default.claude_thinking_mode == "adaptive"
    assert cfg_default.claude_thinking_mode_explicit is False
    assert cfg_default.claude_offline_inject_contract is True
    assert cfg_default.claude_desktop_thinking_mode is None
    assert cfg_default.claude_desktop_thinking_mode_explicit is False

    # 2. Flatten overrides under [claude]
    flat_think_disabled = _flatten_overrides({"claude": {"thinking_mode": "disabled", "offline_inject_contract": False}})
    assert flat_think_disabled["claude_thinking_mode"] == "disabled"
    assert flat_think_disabled["claude_thinking_mode_explicit"] is True
    assert flat_think_disabled["claude_offline_inject_contract"] is False

    flat_think_alias = _flatten_overrides({"claude": {"thinking": "high"}})
    assert flat_think_alias["claude_thinking_mode"] == "high"

    # 3. Flatten overrides under [claude_desktop]
    flat_desk = _flatten_overrides({"claude_desktop": {"thinking_mode": "low", "offline_inject_contract": True}})
    assert flat_desk["claude_desktop_thinking_mode"] == "low"
    assert flat_desk["claude_desktop_thinking_mode_explicit"] is True
    assert flat_desk["claude_offline_inject_contract"] is True

    # 4. End-to-end TOML loading via LauncherConfig
    toml_path = tmp_path / "launcher.toml"
    toml_path.write_text(
        """
[claude]
thinking_mode = "disabled"
offline_inject_contract = true

[claude_desktop]
thinking_mode = "high"
""",
        encoding="utf-8",
    )
    cfg_loaded = LauncherConfig(config_file=str(toml_path))
    assert cfg_loaded.claude_thinking_mode == "disabled"
    assert cfg_loaded.claude_thinking_mode_explicit is True
    assert cfg_loaded.claude_offline_inject_contract is True
    assert cfg_loaded.claude_desktop_thinking_mode == "high"
    assert cfg_loaded.claude_desktop_thinking_mode_explicit is True


def test_strata_reasoning_effort_xhigh_fidelity(tmp_path):
    """Verify Strata model settings default to xhigh effort across registry, config, and YAML generation."""
    from local_ai_launcher.backends.strata import load_strata_model_registry, build_strata_config
    import json

    # 1. strata_models.json registry has effort: xhigh
    reg = load_strata_model_registry()
    assert "qwen-38-flash-strata-iq3s" in reg
    iq3s_spec = reg["qwen-38-flash-strata-iq3s"]
    assert iq3s_spec["reasoning"]["effort"] == "xhigh"

    # 2. build_strata_config populates reasoning_effort: xhigh
    dummy_shard = tmp_path / "dummy-00001-of-00002.gguf"
    dummy_shard.write_text("dummy", encoding="utf-8")
    pack_dir = tmp_path / "pack"
    pack_dir.mkdir(parents=True, exist_ok=True)
    cfg_file = build_strata_config(
        model_name="qwen-38-flash-strata-iq3s",
        first_shard=dummy_shard,
        ple_shard=None,
        pack_dir=pack_dir,
        mtp_dir=None,
        port=56000
    )
    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg_data = json.load(f)
    assert cfg_data.get("reasoning_effort") == "xhigh"


def test_ensure_strata_model_installed_portable_download(tmp_path, monkeypatch):
    """Verify ensure_single_model_installed and ensure_strata_model_installed portably download missing Strata shards."""
    from local_ai_launcher.backends.model_installer import (
        find_strata_model_spec,
        ensure_strata_model_installed,
        ensure_single_model_installed
    )
    from local_ai_launcher.config import config
    from unittest.mock import patch, MagicMock

    # 1. Spec lookup works for primary name and aliases
    spec_tuple = find_strata_model_spec("swift-strata")
    assert spec_tuple is not None
    assert spec_tuple[0] == "swift-qwen-38-flash-strata-iq3s"

    spec_vision = find_strata_model_spec("swift-vision")
    assert spec_vision is not None
    assert spec_vision[0] == "swift-qwen-38-flash-strata-iq3s-vision"

    # 2. Mock download_model_file and ensure_strata_pack to verify missing file handling
    downloaded_calls = []
    def mock_download(repo_id, filename, target_dir):
        downloaded_calls.append((repo_id, filename, target_dir))
        # Create a dummy file so target_path exists
        out_f = Path(target_dir) / filename
        out_f.parent.mkdir(parents=True, exist_ok=True)
        out_f.write_text("dummy", encoding="utf-8")
        return True

    monkeypatch.setattr(config, "models_install_behavior", "always_install")

    with patch("local_ai_launcher.backends.model_installer.AI_WORKSPACE_DIR", tmp_path), \
         patch("local_ai_launcher.backends.strata.AI_WORKSPACE_DIR", tmp_path), \
         patch("local_ai_launcher.backends.model_installer.download_model_file", side_effect=mock_download), \
         patch("local_ai_launcher.backends.strata.ensure_strata_pack", return_value=True) as mock_pack:
        
        ok = ensure_single_model_installed("swift-strata")
        assert ok is True
        assert len(downloaded_calls) >= 2
        repos = {call[0] for call in downloaded_calls}
        filenames = {call[1] for call in downloaded_calls}
        assert "ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF" in repos
        assert "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf" in filenames
        assert "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf" in filenames
        assert mock_pack.called


def test_strata_shard_companion_discovery_when_dir_exists(tmp_path, monkeypatch):
    """Verify resolve_strata_model_files infers companion shards even when model_dir exists without files."""
    from local_ai_launcher.backends.strata import resolve_strata_model_files

    # 1. Directory exists, but no shard files exist on disk
    target_dir = tmp_path / "models" / "llama-server" / "ukisai" / "Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF"
    target_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr("local_ai_launcher.backends.strata.AI_WORKSPACE_DIR", tmp_path)

    resolved = resolve_strata_model_files("swift-strata")
    shards = resolved.get("shards", [])
    assert len(shards) == 2
    shard_names = [s.name for s in shards]
    assert "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf" in shard_names
    assert "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf" in shard_names
    assert resolved["first_shard"].name == "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf"
    # For Swift 1.5, PLE table (per_layer_token_embd.weight) resides in shard 1
    assert resolved["ple_shard"].name == "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf"

    # For standard Qwen MoE (SC117), PLE table resides in shard 2
    resolved_sc117 = resolve_strata_model_files("strata-iq3s")
    assert resolved_sc117["ple_shard"].name == "Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00002-of-00002.gguf"

    # 2. Only shard 1 exists on disk, shard 2 missing
    s1 = target_dir / "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf"
    s1.write_text("dummy", encoding="utf-8")

    resolved_partial = resolve_strata_model_files("swift-strata")
    shards_partial = resolved_partial.get("shards", [])
    assert len(shards_partial) == 2
    partial_names = [s.name for s in shards_partial]
    assert "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf" in partial_names


def test_find_strata_model_spec_by_path_and_filename():
    """Verify find_strata_model_spec resolves by preset name, alias, full path, and shard filename."""
    from local_ai_launcher.backends.model_installer import find_strata_model_spec

    # By preset key
    match_key = find_strata_model_spec("swift-qwen-38-flash-strata-iq3s")
    assert match_key is not None
    assert match_key[0] == "swift-qwen-38-flash-strata-iq3s"

    # By alias
    match_alias = find_strata_model_spec("swift-strata")
    assert match_alias is not None
    assert match_alias[0] == "swift-qwen-38-flash-strata-iq3s"

    # By shard filename
    match_file = find_strata_model_spec("Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf")
    assert match_file is not None
    assert match_file[0] == "swift-qwen-38-flash-strata-iq3s"

    # By relative path
    match_path = find_strata_model_spec("models/llama-server/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf")
    assert match_path is not None
    assert match_path[0] == "swift-qwen-38-flash-strata-iq3s"


def test_ensure_strata_model_installed_cleans_zero_byte_files(tmp_path, monkeypatch):
    """Verify ensure_strata_model_installed deletes 0-byte corrupt shards before invoking download."""
    from local_ai_launcher.backends.model_installer import ensure_strata_model_installed
    from local_ai_launcher.config import config
    from unittest.mock import patch

    monkeypatch.setattr(config, "models_install_behavior", "always_install")

    target_dir = tmp_path / "models" / "llama-server" / "ukisai" / "Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF"
    target_dir.mkdir(parents=True, exist_ok=True)
    corrupted_s1 = target_dir / "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf"
    corrupted_s1.write_bytes(b"")  # 0-byte truncated file
    assert corrupted_s1.exists() and corrupted_s1.stat().st_size == 0

    downloaded = []
    def mock_download(repo_id, filename, target_dir):
        downloaded.append(filename)
        out_f = Path(target_dir) / filename
        out_f.write_text("downloaded_content", encoding="utf-8")
        return True

    with patch("local_ai_launcher.backends.model_installer.AI_WORKSPACE_DIR", tmp_path), \
         patch("local_ai_launcher.backends.strata.AI_WORKSPACE_DIR", tmp_path), \
         patch("local_ai_launcher.backends.model_installer.download_model_file", side_effect=mock_download), \
         patch("local_ai_launcher.backends.strata.ensure_strata_pack", return_value=True):

        ok = ensure_strata_model_installed("swift-strata")
        assert ok is True
        assert corrupted_s1.name in downloaded
        assert corrupted_s1.stat().st_size > 0


