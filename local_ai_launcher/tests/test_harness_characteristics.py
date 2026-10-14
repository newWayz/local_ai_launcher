"""
Unit tests for harness and engine characteristics registry, priority routing,
and context preloading resolution.
"""

import json
from unittest.mock import MagicMock, patch
import pytest

from local_ai_launcher.config import config
from local_ai_launcher.harness_model import (
    HARNESS_SPECS,
    ENGINE_SPECS,
    get_harness_characteristics,
    is_meta_harness,
    is_cli_harness,
    is_offline_capable,
    normalize_harness_list,
    resolve_server_preloaded_harness,
    resolve_meta_harness_inner_cli_harness,
)
from local_ai_launcher.context_preloader import (
    clear_preloaded_cache,
    mark_context_preloaded,
    get_preloaded_harness_for_model,
    preload_harness_context,
)
from local_ai_launcher.process import preload_default_model_async


def test_harness_specs_registry_integrity():
    """Verify all canonical harnesses are registered with required traits."""
    for required in ("pi", "hermes", "deepseek", "claude", "claude-desktop", "opencode", "openlumara", "orca", "t3", "antigravity"):
        spec = get_harness_characteristics(required)
        assert spec is not None, f"Harness '{required}' must be registered"
        assert spec.canonical_name


def test_harness_aliases_resolution():
    """Verify alias mapping to canonical names."""
    assert get_harness_characteristics("dsh").canonical_name == "deepseek"
    assert get_harness_characteristics("deepseek").canonical_name == "deepseek"
    assert get_harness_characteristics("t3code").canonical_name == "t3"
    assert get_harness_characteristics("t3").canonical_name == "t3"
    assert get_harness_characteristics("claudedesktop").canonical_name == "claude-desktop"
    assert get_harness_characteristics("claude-desktop").canonical_name == "claude-desktop"
    assert get_harness_characteristics("claudecode").canonical_name == "claude"
    assert get_harness_characteristics("pie").canonical_name == "pi"
    assert get_harness_characteristics("lumara").canonical_name == "openlumara"
    assert get_harness_characteristics("agy").canonical_name == "antigravity"
    assert get_harness_characteristics("agy-desktop").canonical_name == "antigravity"


def test_meta_harness_identification():
    """Verify meta-harnesses (can host other harnesses) are properly identified."""
    assert is_meta_harness("orca") is True
    assert is_meta_harness("t3") is True
    assert is_meta_harness("t3code") is True

    # Non-meta harnesses
    assert is_meta_harness("pi") is False
    assert is_meta_harness("hermes") is False
    assert is_meta_harness("deepseek") is False
    assert is_meta_harness("dsh") is False
    assert is_meta_harness("claude") is False
    assert is_meta_harness("claude-desktop") is False
    assert is_meta_harness("claudedesktop") is False
    assert is_meta_harness("antigravity") is False


def test_cli_harness_identification():
    """Verify CLI/TUI agents vs full GUI desktop apps."""
    # CLI agents
    assert is_cli_harness("pi") is True
    assert is_cli_harness("hermes") is True
    assert is_cli_harness("deepseek") is True
    assert is_cli_harness("dsh") is True
    assert is_cli_harness("claude") is True
    assert is_cli_harness("openlumara") is True
    assert is_cli_harness("opencode") is True

    # GUI applications
    assert is_cli_harness("claudedesktop") is False
    assert is_cli_harness("claude-desktop") is False
    assert is_cli_harness("orca") is False
    assert is_cli_harness("t3") is False
    assert is_cli_harness("t3code") is False
    assert is_cli_harness("agy-desktop") is False


def test_offline_capability_across_engines():
    """Verify offline capability rules: Pi/Hermes/DSH are always offline, Claude requires Strata."""
    # Always offline capable
    for h in ("pi", "hermes", "deepseek", "dsh", "opencode", "orca", "t3"):
        assert is_offline_capable(h, model_name="any-model") is True

    # Cloud only
    assert is_offline_capable("antigravity", model_name="strata-iq3s") is False
    assert is_offline_capable("agy-desktop", model_name="strata-iq3s") is False

    # Claude Code CLI: offline under Strata, not under standard models
    assert is_offline_capable("claude", model_name="qwen-38-flash-strata-iq3s") is True
    assert is_offline_capable("claude", model_name="cyber-tiel-coder-35b") is False

    # Claude Desktop: offline under Strata, not under standard models
    assert is_offline_capable("claudedesktop", model_name="qwen-38-flash-strata-iq3s") is True
    assert is_offline_capable("claudedesktop", model_name="cyber-tiel-coder-35b") is False


def test_resolve_server_preloaded_harness_user_example_strata():
    """Verify the exact user scenario:
    choose_harness = ["orca", "t3", "claudedesktop", "pi", "hermes"]
    with Strata active -> 'claudedesktop' is preloaded into the LLM server KV cache.
    """
    selected = ["orca", "t3", "claudedesktop", "pi", "hermes"]
    target = resolve_server_preloaded_harness(
        selected,
        model_name="qwen-38-flash-strata-iq3s"
    )
    # Leftmost non-meta offline-capable harness is claude-desktop
    assert target in ("claudedesktop", "claude-desktop")


def test_resolve_server_preloaded_harness_user_example_non_strata():
    """Verify that when Strata is not active, claudedesktop is not offline capable,
    so the server preloads the next leftmost offline non-meta harness ('pi').
    """
    selected = ["orca", "t3", "claudedesktop", "pi", "hermes"]
    target = resolve_server_preloaded_harness(
        selected,
        model_name="qwen-38_27b-huang-gsq-iq3s-think"
    )
    assert target == "pi"


def test_resolve_server_preloaded_harness_meta_only():
    """Verify that if only meta-harnesses are defined, it defaults to the leftmost meta-harness inner CLI agent ('pi')."""
    assert resolve_server_preloaded_harness(["orca", "t3"]) == "pi"
    assert resolve_server_preloaded_harness(["orca"]) == "pi"
    assert resolve_server_preloaded_harness(["t3code"]) == "pi"


def test_resolve_server_preloaded_harness_meta_and_single_cli():
    """Verify leftmost non-meta offline harness is selected when placed after meta-harnesses."""
    assert resolve_server_preloaded_harness(["orca", "hermes"]) == "hermes"
    assert resolve_server_preloaded_harness(["t3", "orca", "dsh", "pi"]) == "deepseek"


def test_resolve_meta_harness_inner_cli_user_example():
    """Verify that in the leftmost meta-harness, it loads the leftmost non-meta offline CLI harness:
    choose_harness = ["orca", "t3", "claudedesktop", "pi", "hermes"]
    -> returns 'pi' (because claudedesktop is a GUI app, not a CLI).
    """
    selected = ["orca", "t3", "claudedesktop", "pi", "hermes"]
    inner = resolve_meta_harness_inner_cli_harness(
        selected,
        model_name="qwen-38-flash-strata-iq3s"
    )
    assert inner == "pi"


def test_resolve_meta_harness_inner_cli_hermes_first():
    """Verify that if hermes is before pi, hermes is selected for the meta-harness inner agent."""
    selected = ["orca", "t3", "claudedesktop", "hermes", "pi"]
    inner = resolve_meta_harness_inner_cli_harness(
        selected,
        model_name="qwen-38-flash-strata-iq3s"
    )
    assert inner == "hermes"


def test_resolve_meta_harness_inner_cli_defaults_to_pi():
    """Verify that if no offline CLI harness is defined in choose_harness, defaults to 'pi' for both Orca and T3."""
    assert resolve_meta_harness_inner_cli_harness(["orca"]) == "pi"
    assert resolve_meta_harness_inner_cli_harness(["t3code"]) == "pi"
    assert resolve_meta_harness_inner_cli_harness(["orca", "t3"]) == "pi"
    assert resolve_meta_harness_inner_cli_harness(["orca", "claudedesktop"]) == "pi"


def test_launcher_config_helper_methods(monkeypatch):
    """Verify LauncherConfig helper methods reflect priority resolution."""
    monkeypatch.setattr(config, "choose_harness", ["orca", "t3", "claudedesktop", "pi", "hermes"])
    monkeypatch.setattr(config, "default_model", "qwen-38-flash-strata-iq3s")

    assert config.get_server_preloaded_harness() in ("claudedesktop", "claude-desktop")
    assert config.get_meta_harness_inner_cli_harness() == "pi"


def test_preload_default_model_async_targets_leftmost_non_meta(monkeypatch):
    """Verify preload_default_model_async preloads the leftmost non-meta harness into LLM server."""
    clear_preloaded_cache()
    monkeypatch.setattr(config, "choose_harness", ["orca", "t3", "claudedesktop", "pi", "hermes"])
    monkeypatch.setattr(config, "default_model", "qwen-38-flash-strata-iq3s")

    with patch("urllib.request.urlopen") as mock_urlopen, patch("time.sleep"):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = b"{}"
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        thread = preload_default_model_async(
            port=8080,
            model_name="qwen-38-flash-strata-iq3s",
            timeout=10
        )
        thread.join(timeout=2.0)

        assert mock_urlopen.called
        call_reqs = [c[0][0] for c in mock_urlopen.call_args_list if hasattr(c[0][0], "data") and c[0][0].data]
        assert len(call_reqs) >= 1
        comp_req = call_reqs[-1]
        assert "claudedesktop" in comp_req.get_header("User-agent").lower() or "claude" in comp_req.get_header("User-agent").lower()
        payload = json.loads(comp_req.data.decode("utf-8"))
        assert any("Claude" in m.get("content", "") for m in payload.get("messages", []) if m.get("role") == "system")


def test_kv_cache_preservation_skips_secondary_harness_prefill():
    """Verify that once a designated harness is preloaded into the server KV cache,
    secondary harness prefill calls for other harnesses are skipped to prevent cache pollution.
    """
    clear_preloaded_cache()
    base_url = "http://127.0.0.1:8080/v1"
    model_name = "test-kv-model"

    # Mark claudedesktop as preloaded
    mark_context_preloaded(base_url, model_name, "claudedesktop")
    assert get_preloaded_harness_for_model(base_url, model_name) == "claudedesktop"

    with patch("urllib.request.urlopen") as mock_urlopen:
        # Attempt to preload t3 or pi for the same model
        t_t3 = preload_harness_context(base_url, model_name, "t3")
        assert t_t3 is None  # Skipped!
        assert not mock_urlopen.called

        t_pi = preload_harness_context(base_url, model_name, "pi")
        assert t_pi is None  # Skipped!
        assert not mock_urlopen.called


def test_orca_config_inner_cli_agent_selection(tmp_path, monkeypatch):
    """Verify setup_orca_config sets defaultTuiAgent based on resolve_meta_harness_inner_cli_harness."""
    from local_ai_launcher.harnesses.orca import setup_orca_config

    orca_home = tmp_path / ".orca"
    monkeypatch.setenv("APPDATA", str(tmp_path / "AppData"))
    monkeypatch.setattr(config, "choose_harness", ["orca", "t3", "claudedesktop", "pi", "hermes"])
    monkeypatch.setattr(config, "default_model", "qwen-38-flash-strata-iq3s")
    monkeypatch.setattr(config, "orca_data_dir", tmp_path / "orca_data")

    setup_orca_config(orca_home, "http://127.0.0.1:8080/v1", "qwen-38-flash-strata-iq3s")

    orca_data_file = tmp_path / "AppData" / "Orca" / "profiles" / "local-default" / "orca-data.json"
    assert orca_data_file.exists()
    data = json.loads(orca_data_file.read_text(encoding="utf-8"))
    # In ["orca", "t3", "claudedesktop", "pi", "hermes"], claudedesktop is GUI, so pi is selected!
    assert data["settings"]["defaultTuiAgent"] == "pi"


def test_t3_config_inner_cli_agent_selection(tmp_path, monkeypatch):
    """Verify setup_t3_config sets defaultHarness based on resolve_meta_harness_inner_cli_harness."""
    from local_ai_launcher.harnesses.t3 import setup_t3_config

    t3_home = tmp_path / ".t3"
    monkeypatch.setenv("T3CODE_BASE_DIR", str(tmp_path / "t3_userdata"))
    monkeypatch.setattr(config, "choose_harness", ["t3", "claudedesktop", "hermes", "pi"])
    monkeypatch.setattr(config, "default_model", "qwen-38-flash-strata-iq3s")

    setup_t3_config(t3_home, "http://127.0.0.1:8080/v1", "qwen-38-flash-strata-iq3s")

    config_file = t3_home / "config.json"
    assert config_file.exists()
    data = json.loads(config_file.read_text(encoding="utf-8"))
    # In ["t3", "claudedesktop", "hermes", "pi"], hermes is the leftmost offline CLI harness!
    assert data["defaultHarness"] == "hermes"


def test_meta_harness_defaults_to_claude_under_strata(tmp_path, monkeypatch):
    """Verify that when only meta-harnesses are defined:
    - Under Strata engine: Orca sets defaultTuiAgent to claude-local and T3 sets defaultHarness to claude
    - Under non-Strata engine: Orca sets defaultTuiAgent to pi and T3 sets defaultHarness to pi
    """
    from local_ai_launcher.harnesses.orca import setup_orca_config
    from local_ai_launcher.harnesses.t3 import setup_t3_config

    # 1. Under Strata Engine
    orca_home = tmp_path / ".orca_strata"
    t3_home = tmp_path / ".t3_strata"
    monkeypatch.setenv("APPDATA", str(tmp_path / "AppData_strata"))
    monkeypatch.setenv("T3CODE_BASE_DIR", str(tmp_path / "t3_userdata_strata"))
    monkeypatch.setattr(config, "choose_harness", ["orca", "t3"])
    monkeypatch.setattr(config, "default_model", "qwen-38-flash-strata-iq3s")
    monkeypatch.setattr(config, "orca_data_dir", tmp_path / "orca_data_strata")

    setup_orca_config(orca_home, "http://127.0.0.1:8080/v1", "qwen-38-flash-strata-iq3s")
    setup_t3_config(t3_home, "http://127.0.0.1:8080/v1", "qwen-38-flash-strata-iq3s")

    orca_data_file = tmp_path / "AppData_strata" / "Orca" / "profiles" / "local-default" / "orca-data.json"
    data_orca = json.loads(orca_data_file.read_text(encoding="utf-8"))
    assert data_orca["settings"]["defaultTuiAgent"] == "claude-local"

    data_t3 = json.loads((t3_home / "config.json").read_text(encoding="utf-8"))
    assert data_t3["defaultHarness"] == "claude"

    # 2. Under non-Strata Engine (Huang / llama.cpp)
    orca_home_non = tmp_path / ".orca_non"
    t3_home_non = tmp_path / ".t3_non"
    monkeypatch.setenv("APPDATA", str(tmp_path / "AppData_non"))
    monkeypatch.setenv("T3CODE_BASE_DIR", str(tmp_path / "t3_userdata_non"))
    monkeypatch.setattr(config, "choose_harness", ["orca", "t3"])
    monkeypatch.setattr(config, "default_model", "qwen-38_27b-huang-gsq-iq3s-think")
    monkeypatch.setattr(config, "orca_data_dir", tmp_path / "orca_data_non")

    setup_orca_config(orca_home_non, "http://127.0.0.1:8080/v1", "qwen-38_27b-huang-gsq-iq3s-think")
    setup_t3_config(t3_home_non, "http://127.0.0.1:8080/v1", "qwen-38_27b-huang-gsq-iq3s-think")

    orca_data_file_non = tmp_path / "AppData_non" / "Orca" / "profiles" / "local-default" / "orca-data.json"
    data_orca_non = json.loads(orca_data_file_non.read_text(encoding="utf-8"))
    assert data_orca_non["settings"]["defaultTuiAgent"] == "pi"

    data_t3_non = json.loads((t3_home_non / "config.json").read_text(encoding="utf-8"))
    assert data_t3_non["defaultHarness"] == "pi"


def test_local_claude_native_engine_characteristics_and_expandability():
    """Verify local_claude_native engine traits, Strata membership, and dynamic engine expandability."""
    from local_ai_launcher.harness_model import (
        HARNESS_SPECS,
        ENGINE_SPECS,
        EngineCharacteristics,
        get_local_claude_native_engines,
        is_local_claude_native_engine,
        register_local_claude_native_engine,
        is_offline_capable,
    )

    # 1. Claude harnesses specify local_claude_native
    assert HARNESS_SPECS["claude"].offline_support == "local_claude_native"
    assert HARNESS_SPECS["claude-desktop"].offline_support == "local_claude_native"

    # 2. Strata is a canonical member of local_claude_native
    assert ENGINE_SPECS["strata"].is_local_claude_native is True
    assert ENGINE_SPECS["strata"].supports_anthropic_messages is True
    assert "strata" in get_local_claude_native_engines()
    assert is_local_claude_native_engine("strata") is True
    assert is_local_claude_native_engine("strata-engine") is True

    # 3. Non-Claude-native engines are False
    assert is_local_claude_native_engine("huang") is False
    assert is_local_claude_native_engine("bruno") is False
    assert is_local_claude_native_engine("official") is False

    # 4. Expandability to future engines: register a new engine
    new_engine = "vllm_anthropic"
    assert is_local_claude_native_engine(new_engine) is False
    assert is_offline_capable("claude", engine=new_engine) is False

    register_local_claude_native_engine(
        new_engine,
        EngineCharacteristics(
            engine_name=new_engine,
            aliases=("vllm", "vllm-native"),
            supports_anthropic_messages=True,
            is_local_claude_native=True,
            description="vLLM native Anthropic server"
        )
    )

    # 5. The new engine is immediately recognized as local_claude_native
    assert is_local_claude_native_engine(new_engine) is True
    assert is_local_claude_native_engine("vllm") is True
    assert new_engine in get_local_claude_native_engines()

    # 6. Claude harnesses can now run offline against the new future engine!
    assert is_offline_capable("claude", engine=new_engine) is True
    assert is_offline_capable("claude-desktop", engine=new_engine) is True
    assert is_offline_capable("claude", engine="vllm") is True


def test_is_local_claude_native_active_helper():
    """Verify is_local_claude_native_active helper and is_strata_active backward-compat alias."""
    from local_ai_launcher.harnesses.claude import is_local_claude_native_active, is_strata_active

    # Strata model is active
    assert is_local_claude_native_active("qwen-38-flash-strata-iq3s") is True
    assert is_strata_active("qwen-38-flash-strata-iq3s") is True

    # Standard model is not active
    assert is_local_claude_native_active("cyber-tiel-coder-35b") is False
    assert is_strata_active("cyber-tiel-coder-35b") is False


