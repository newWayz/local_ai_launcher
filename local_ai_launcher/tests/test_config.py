import os
import sys
import pytest
from pathlib import Path
from local_ai_launcher.config import LauncherConfig, config, MnemosyneAppConfig
from local_ai_launcher.__main__ import parse_cli_args


def test_headless_config_defaults_and_property():
    cfg = LauncherConfig()
    assert cfg.localai_headless is False
    assert cfg.open_localai_gui is True

    cfg.open_localai_gui = False
    assert cfg.localai_headless is True

    cfg.localai_headless = False
    assert cfg.open_localai_gui is True


def test_headless_env_var_override(monkeypatch):
    monkeypatch.setenv("LOCALAI_HEADLESS", "1")
    cfg = LauncherConfig()
    assert cfg.localai_headless is True
    assert cfg.open_localai_gui is False

    monkeypatch.setenv("LOCALAI_HEADLESS", "0")
    cfg2 = LauncherConfig()
    assert cfg2.localai_headless is False
    assert cfg2.open_localai_gui is True


def test_cli_headless_flags(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["local_ai_launcher", "--headless"])
    parse_cli_args()
    assert config.localai_headless is True

    monkeypatch.setattr(sys, "argv", ["local_ai_launcher", "--gui"])
    parse_cli_args()
    assert config.localai_headless is False

    monkeypatch.setattr(sys, "argv", ["local_ai_launcher", "--no-gui"])
    parse_cli_args()
    assert config.localai_headless is True

    monkeypatch.setattr(sys, "argv", ["local_ai_launcher", "--no-headless"])
    parse_cli_args()
    assert config.localai_headless is False


def test_paths_resolution_and_independence():
    """Verify paths.py resolves filesystem anchors without circular dependencies."""
    from local_ai_launcher import paths
    assert isinstance(paths.AI_WORKSPACE_DIR, Path)
    assert isinstance(paths.LOCAL_AI_LAUNCHER_DIR, Path)
    assert paths.PROJECTS_DIR == paths.AI_WORKSPACE_DIR / "projects"
    assert paths.CONFIG_DIR == paths.AI_WORKSPACE_DIR / "config"
    assert paths.HERMES_DIR == paths.AI_WORKSPACE_DIR / ".hermes"
    assert paths.ORCA_DIR == paths.AI_WORKSPACE_DIR / ".orca"


def test_prompts_non_interactive_fallbacks(monkeypatch):
    """Verify prompts.py helpers safely fallback to defaults when run non-interactively."""
    from local_ai_launcher import prompts

    # In non-interactive mode, choices and strings return their defaults immediately
    assert prompts.prompt_choice("Non interactive?", default='n') is False
    assert prompts.prompt_choice("Non interactive?", default='y') is True
    assert prompts.prompt_string("Enter name:", default="test_bot") == "test_bot"

    # Verify behavior resolver
    assert prompts._resolve_behavior("never_install", "install", "comp") is False
    assert prompts._resolve_behavior("always_install", "install", "comp") is True


def test_config_facade_backward_compatibility():
    """Verify config.py re-exports all expected path, prompt, and engine symbols."""
    from local_ai_launcher import config as cfg_mod

    assert hasattr(cfg_mod, "AI_WORKSPACE_DIR")
    assert hasattr(cfg_mod, "MODELS_PRESET_FILE")
    assert hasattr(cfg_mod, "FREETOKENS_MODELS_PRESET_FILE")
    assert hasattr(cfg_mod, "prompt_choice")
    assert hasattr(cfg_mod, "prompt_string")
    assert hasattr(cfg_mod, "LLM_BACKEND_PORT")
    assert hasattr(cfg_mod, "OPENAI_API_SPEC_PORT")
    assert hasattr(cfg_mod, "LauncherConfig")
    assert hasattr(cfg_mod, "MnemosyneAppConfig")
    assert hasattr(cfg_mod, "config")


def test_launcher_toml_override_loading(tmp_path):
    """Verify LauncherConfig loads user overrides from an external TOML file."""
    toml_file = tmp_path / "launcher.toml"
    toml_file.write_text(
        'default_model = "test-custom-override-model"\n'
        'localai_port = 9099\n'
        'choose_harness = ["pi", "hermes"]\n'
        'use_omniroute_proxy = true\n',
        encoding="utf-8"
    )

    cfg = LauncherConfig(config_file=toml_file)
    assert cfg.default_model == "test-custom-override-model"
    assert cfg.localai_port == 9099
    assert cfg.choose_harness == ["pi", "hermes"]
    assert cfg.use_omniroute_proxy is True


def test_launcher_toml_strata_override_loading(tmp_path):
    """Verify strata_mmap_experts and strata_brutal_ram_clawback load correctly from launcher.toml."""
    # Test top-level keys
    toml_top = tmp_path / "launcher_top.toml"
    toml_top.write_text(
        "strata_mmap_experts = false\n"
        'strata_brutal_ram_clawback = "off"\n',
        encoding="utf-8"
    )
    cfg1 = LauncherConfig(config_file=toml_top)
    assert cfg1.strata_mmap_experts is False
    assert cfg1.strata_brutal_ram_clawback == "off"

    # Test [strata] table
    toml_sec = tmp_path / "launcher_sec.toml"
    toml_sec.write_text(
        "[strata]\n"
        "mmap_experts = false\n"
        'brutal_ram_clawback = "max"\n',
        encoding="utf-8"
    )
    cfg2 = LauncherConfig(config_file=toml_sec)
    assert cfg2.strata_mmap_experts is False
    assert cfg2.strata_brutal_ram_clawback == "max"

    # Test default is light
    cfg_def = LauncherConfig()
    assert cfg_def.strata_brutal_ram_clawback == "light"


def test_config_negative_and_out_of_bounds_sanitization():
    """Verify LauncherConfig and MnemosyneAppConfig sanitize negative numbers and bounds."""
    cfg = LauncherConfig()

    # 1. Ports: negative or out-of-range (1..65535) reset safely to known defaults
    cfg.localai_port = -8080
    assert cfg.localai_port == 8080
    cfg.localai_port = 70000
    assert cfg.localai_port == 8080
    cfg.blackwell_shim_port = -9940
    assert cfg.blackwell_shim_port == 9940
    cfg.tabbyapi_shim_port = 70000
    assert cfg.tabbyapi_shim_port == 9941
    cfg.freetoken_shim_port = 0
    assert cfg.freetoken_shim_port == 9942
    cfg.strata_shim_port = -9943
    assert cfg.strata_shim_port == 9943
    cfg.mcp_server_port = -10
    assert cfg.mcp_server_port == 9932
    cfg.t3_port = -3773
    assert cfg.t3_port == 3773

    # 2. Loop recovery: negative values clamped to safe minimums
    cfg.loop_recovery_max_retries = -5
    assert cfg.loop_recovery_max_retries == 0
    cfg.loop_recovery_temp_boost = -0.5
    assert cfg.loop_recovery_temp_boost == 0.0
    cfg.loop_recovery_presence_penalty = -0.5
    assert cfg.loop_recovery_presence_penalty == 0.0

    # 3. Loop detector: positive lower bounds enforced (resets to defaults if invalid)
    cfg.loop_detector_ngram_size = -3
    assert cfg.loop_detector_ngram_size == 8
    cfg.loop_detector_ngram_threshold = -1
    assert cfg.loop_detector_ngram_threshold == 3
    cfg.loop_detector_min_sentence_len = -20
    assert cfg.loop_detector_min_sentence_len == 0
    cfg.loop_detector_cycle_similarity = -0.5
    assert cfg.loop_detector_cycle_similarity == 0.82
    cfg.loop_detector_cycle_similarity = 1.5
    assert cfg.loop_detector_cycle_similarity == 0.82

    # 4. Booleans: avoid python bool(-1) == True trap
    cfg.enable_loop_detection = -1
    assert cfg.enable_loop_detection is False
    cfg.enable_loop_detection = "-1"
    assert cfg.enable_loop_detection is False
    cfg.enable_loop_detection = 1
    assert cfg.enable_loop_detection is True
    cfg.show_llama_server_console = -2
    assert cfg.show_llama_server_console is False
    cfg.use_omniroute_proxy = -1
    assert cfg.use_omniroute_proxy is False

    # 5. Ratios clamped to valid range
    cfg.pi_compaction_keep_ratio = -0.4
    assert cfg.pi_compaction_keep_ratio == 0.0
    cfg.pi_compaction_keep_ratio = 1.4
    assert cfg.pi_compaction_keep_ratio == 1.0
    cfg.hermes_compression_threshold_ratio = -0.1
    assert cfg.hermes_compression_threshold_ratio == 0.0
    cfg.hermes_compression_threshold_ratio = 1.9
    assert cfg.hermes_compression_threshold_ratio == 1.0

    # 6. MnemosyneAppConfig intervals and retention
    mcfg = MnemosyneAppConfig()
    mcfg.backup_interval_days = -5
    assert mcfg.backup_interval_days == 5
    mcfg.max_backups_to_keep = 0
    assert mcfg.max_backups_to_keep == 10


def test_cpu_auxiliary_toggles_and_loop_detector_customization():
    cfg = LauncherConfig()
    assert cfg.enable_cpu_reranker is True
    assert cfg.enable_cpu_onnx_gater is True
    assert cfg.enable_cpu_decision is True
    assert cfg.enable_cpu_llama_server is True
    assert cfg.enable_auto_cpu_dispatch is True
    assert cfg.use_cpu_reranker is True
    assert cfg.use_cpu_onnx_gater is True
    assert cfg.use_cpu_decision is True
    assert cfg.use_cpu_llama_server is True

    # Test toggles
    cfg.enable_cpu_reranker = False
    assert cfg.use_cpu_reranker is False
    cfg.enable_cpu_reranker = True
    assert cfg.use_cpu_reranker is True

    cfg.enable_cpu_onnx_gater = False
    assert cfg.use_cpu_onnx_gater is False
    assert cfg.use_cpu_decision is False
    cfg.enable_cpu_onnx_gater = True
    assert cfg.use_cpu_onnx_gater is True
    assert cfg.use_cpu_decision is True

    cfg.enable_cpu_llama_server = False
    assert cfg.use_cpu_llama_server is False
    cfg.enable_cpu_llama_server = True
    assert cfg.use_cpu_llama_server is True

    # Ngram token count customization
    cfg.loop_detector_ngram_size = 12
    assert cfg.loop_detector_ngram_size == 12
    cfg.loop_detector_ngram_size = -5
    assert cfg.loop_detector_ngram_size == 8


def test_gvs5h_log_retention_config():
    cfg = LauncherConfig()
    assert cfg.gvs5h_max_log_runs == 100

    # Customization
    cfg.gvs5h_max_log_runs = 50
    assert cfg.gvs5h_max_log_runs == 50

    # Hardening: negative or zero values reset to default 100
    cfg.gvs5h_max_log_runs = -10
    assert cfg.gvs5h_max_log_runs == 100
    cfg.gvs5h_max_log_runs = 0
    assert cfg.gvs5h_max_log_runs == 100


def test_donsetch_config_defaults():
    cfg = LauncherConfig()
    assert cfg.pi_donsetch_enabled is True
    assert cfg.deepseek_donsetch_enabled is True
    assert cfg.donsetch_tool_prefix == "donsetch"
    assert cfg.donsetch_auto_update is True
    assert cfg.donsetch_channel == "stable"
    assert cfg.donsetch_bin is None

    # Verify native/first-class integration lists
    assert "npm:donsetch" in cfg.harness_extensions.get("pi", [])
    assert "github:dondai44423/donsetch-dsh" in cfg.harness_plugins.get("deepseek", [])

    # Verify other harnesses do NOT have donsetch extensions or plugins
    for harness in ("hermes", "claude", "opencode", "orca", "t3", "agy"):
        assert not any("donsetch" in ext for ext in cfg.harness_extensions.get(harness, []))
        assert not any("donsetch" in p for p in cfg.harness_plugins.get(harness, []))


def test_donsetch_toml_flattening():
    from local_ai_launcher.config import _flatten_overrides

    # 1. Dedicated [donsetch] section
    overrides = {
        "donsetch": {
            "enabled": False,
            "tool_prefix": "web",
            "channel": "latest",
            "auto_update": False,
            "bin": "C:/custom/donsetch.exe",
        }
    }
    flat = _flatten_overrides(overrides)
    assert flat["pi_donsetch_enabled"] is False
    assert flat["deepseek_donsetch_enabled"] is False
    assert flat["donsetch_tool_prefix"] == "web"
    assert flat["donsetch_channel"] == "latest"
    assert flat["donsetch_auto_update"] is False
    assert flat["donsetch_bin"] == Path("C:/custom/donsetch.exe")

    # 2. Harness-scoped section overrides ([pi] and [deepseek])
    harness_overrides = {
        "pi": {
            "donsetch": False,
            "extensions": ["custom-ext"],
        },
        "deepseek": {
            "donsetch": True,
            "plugins": ["custom-plugin"],
            "tool_prefix": "custom_search",
        }
    }
    flat2 = _flatten_overrides(harness_overrides)
    assert flat2["pi_donsetch_enabled"] is False
    assert "custom-ext" in flat2["harness_extensions"]["pi"]
    assert flat2["deepseek_donsetch_enabled"] is True
    assert "custom-plugin" in flat2["harness_plugins"]["deepseek"]
    assert flat2["donsetch_tool_prefix"] == "custom_search"


def test_update_frequency_hrs_config_semantics():
    """Verify update_frequency_hrs semantics, values (1, 24, 0, negative), and string parsing."""
    cfg = LauncherConfig()
    assert cfg.update_frequency_hrs == 168
    assert getattr(cfg, "update_frequency", None) is None

    # Value of 1: hourly
    cfg.update_frequency_hrs = 1
    assert cfg.update_frequency_hrs == 1

    # Value of 24: daily
    cfg.update_frequency_hrs = 24
    assert cfg.update_frequency_hrs == 24

    # Value of 0: always
    cfg.update_frequency_hrs = 0
    assert cfg.update_frequency_hrs == 0

    # Negative value: never
    cfg.update_frequency_hrs = -1
    assert cfg.update_frequency_hrs == float("inf")

    # Setting string representation into update_frequency_hrs sanitizes into numeric hours
    cfg.update_frequency_hrs = "daily"
    assert cfg.update_frequency_hrs == 24

    cfg.update_frequency_hrs = "always"
    assert cfg.update_frequency_hrs == 0

    cfg.update_frequency_hrs = "48h"
    assert cfg.update_frequency_hrs == 48

    cfg.update_frequency_hrs = "never"
    assert cfg.update_frequency_hrs == float("inf")


def test_searxng_config_defaults_and_properties():
    """Verify SearXNG default configuration settings, port default 8088, and effective URL property."""
    from local_ai_launcher.config import _flatten_overrides
    cfg = LauncherConfig()
    assert cfg.searxng_enabled is True
    assert cfg.searxng_mode == "wsl_docker"  # Tier 3 user selected default
    assert cfg.searxng_port == 8088
    assert cfg.searxng_uwsgi_workers == 1
    assert cfg.searxng_limiter is False
    assert cfg.searxng_wsl_distro == "Ubuntu-26.04"
    assert cfg.searxng_effective_url == "http://127.0.0.1:8088"

    # Switching to remote mode reflects remote URL
    cfg.searxng_mode = "remote"
    cfg.searxng_remote_url = "https://custom.searx.space"
    assert cfg.searxng_effective_url == "https://custom.searx.space"


def test_searxng_overrides_flattening_and_port_sanitization():
    """Verify [searxng] table in overrides is properly flattened and port is validated."""
    from local_ai_launcher.config import _flatten_overrides
    raw = {
        "searxng": {
            "mode": "native",
            "port": 8888,
            "remote_url": "https://searx.be",
            "engines": ["duckduckgo", "brave"],
            "workers": 2,
            "limiter": False
        }
    }
    flat = _flatten_overrides(raw)
    assert flat["searxng_mode"] == "native"
    assert flat["searxng_port"] == 8888
    assert flat["searxng_remote_url"] == "https://searx.be"
    assert flat["searxng_upstream_engines"] == ["duckduckgo", "brave"]
    assert flat["searxng_uwsgi_workers"] == 2
    assert flat["searxng_limiter"] is False

    # Port clamping on invalid value
    cfg = LauncherConfig()
    sanitized_port = cfg._sanitize_field("searxng_port", 999999)
    assert sanitized_port == 8088


def test_rea_config_and_overrides():
    """Verify REA default configuration and table/key override flattening."""
    from local_ai_launcher.config import LauncherConfig, _flatten_overrides
    cfg = LauncherConfig()
    assert cfg.rea_enabled is True
    assert cfg.rea_schema_profile == "compact"

    # 1. Table override [rea]
    raw_table = {
        "rea": {
            "enabled": False,
            "input_schema_profile": "full",
        }
    }
    flat = _flatten_overrides(raw_table)
    assert flat["rea_enabled"] is False
    assert flat["rea_schema_profile"] == "full"

    # 2. Key override rea_enabled
    raw_keys = {
        "rea_enabled": True,
        "rea_schema_profile": "compact",
    }
    flat2 = _flatten_overrides(raw_keys)
    assert flat2["rea_enabled"] is True
    assert flat2["rea_schema_profile"] == "compact"


def test_git_config_and_overrides():
    """Verify Git client default configuration and table/key override flattening."""
    from local_ai_launcher.config import LauncherConfig, _flatten_overrides
    cfg = LauncherConfig()
    assert cfg.git_update_behavior == "never_update"
    assert cfg.git_auto_update is False
    assert cfg.git_update_frequency_hrs == 168

    # 1. Table override [git] with auto_update
    raw_table = {
        "git": {
            "auto_update": True,
            "update_behavior": "always_update",
            "update_frequency_hrs": 24,
        }
    }
    flat = _flatten_overrides(raw_table)
    assert flat["git_auto_update"] is True
    assert flat["git_update_behavior"] == "always_update"
    assert flat["git_update_frequency_hrs"] == 24

    # 2. Key override
    raw_keys = {
        "git_auto_update": True,
    }
    flat2 = _flatten_overrides(raw_keys)
    assert flat2["git_auto_update"] is True
    assert flat2["git_update_behavior"] == "always_update"


def test_local_claude_native_config_and_overrides():
    """Verify local_claude_native options, Strata engine membership, and expandability to new engines."""
    from local_ai_launcher.config import LauncherConfig, _flatten_overrides
    cfg = LauncherConfig()

    # 1. Defaults: strata is a member of local_claude_native
    assert "strata" in cfg.local_claude_native_engines
    assert cfg.is_local_claude_native_engine("strata") is True
    assert cfg.is_local_claude_native_engine("non_existent_engine") is False
    assert cfg.local_claude_native_use_custom_chat_template is True
    assert cfg.strata_use_custom_chat_template is True

    # 2. Bidirectional sync between local_claude_native and strata chat template
    cfg.local_claude_native_use_custom_chat_template = False
    assert cfg.strata_use_custom_chat_template is False
    cfg.strata_use_custom_chat_template = True
    assert cfg.local_claude_native_use_custom_chat_template is True

    # 3. Expandability to future engines
    cfg.local_claude_native_engines = ["strata", "vllm", "aphrodite"]
    assert cfg.is_local_claude_native_engine("vllm") is True
    assert cfg.is_local_claude_native_engine("aphrodite") is True
    assert cfg.is_local_claude_native_engine("strata") is True

    # 4. Table override [local_claude_native] in TOML
    toml_dict = {
        "local_claude_native": {
            "engines": ["strata", "sglang"],
            "use_custom_chat_template": False,
            "chat_template_file": "custom_claude.jinja",
        }
    }
    flat = _flatten_overrides(toml_dict)
    assert flat["local_claude_native_engines"] == ["strata", "sglang"]
    assert flat["local_claude_native_use_custom_chat_template"] is False
    assert flat["strata_use_custom_chat_template"] is False
    assert flat["local_claude_native_chat_template_file"] == "custom_claude.jinja"

    # 5. Strata table retains engine-specific options while syncing shared Claude option
    strata_toml = {
        "strata": {
            "mmap_experts": True,
            "resident_budget_gib": 56,
            "brutal_ram_clawback": "light",
            "kv_mode": "int8",
            "use_custom_chat_template": True,
        }
    }
    flat_strata = _flatten_overrides(strata_toml)
    assert flat_strata["strata_mmap_experts"] is True
    assert flat_strata["strata_resident_budget_gib"] == 56
    assert flat_strata["strata_brutal_ram_clawback"] == "light"
    assert flat_strata["strata_kv_mode"] == "int8"
    assert flat_strata["strata_use_custom_chat_template"] is True
    assert flat_strata["local_claude_native_use_custom_chat_template"] is True





