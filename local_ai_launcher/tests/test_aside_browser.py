"""
Comprehensive test suite for Aside AI Browser integration with local AI launcher.
Verifies binary discovery, engine configuration, MCP server registration across harnesses,
PowerShell environment path injection, and harness registry characteristics.
"""

import json
import os
import sys
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from local_ai_launcher.config import LauncherConfig, _flatten_overrides, config
from local_ai_launcher.paths import (
    get_aside_binary_path,
    get_aside_app_path,
    AI_WORKSPACE_DIR,
)
from local_ai_launcher.backends.aside import (
    setup_aside_local_engine,
    get_aside_mcp_server_config,
    is_aside_installed,
    get_aside_config_dir,
)
from local_ai_launcher.process import build_ps_env
from local_ai_launcher.harness_model import HARNESS_SPECS, HARNESS_ALIAS_MAP
from local_ai_launcher.harnesses import HARNESS_REGISTRY
from local_ai_launcher.harnesses.pi import setup_pi_mcp_config
from local_ai_launcher.harnesses.claude import setup_claude_cli_mcp_config, setup_claude_desktop_config
from local_ai_launcher.harnesses.antigravity import setup_antigravity_mcp_config
from local_ai_launcher.harnesses.orca import setup_orca_config
from local_ai_launcher.harnesses.t3 import setup_t3_config
from local_ai_launcher.harnesses.aside import AsideAdapter


@pytest.fixture
def temp_dir():
    d = Path(tempfile.mkdtemp(prefix="aside_test_"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_aside_binary_paths(temp_dir):
    """Verify Aside binary and app path resolution."""
    bin_path = get_aside_binary_path()
    app_path = get_aside_app_path()

    # If installed on host, paths should be resolved and valid
    if bin_path:
        assert bin_path.exists()
        assert bin_path.name.lower().startswith("aside")

    # Verify custom / fallback path resolution when LOCALAPPDATA has no Aside
    fake_exe = Path(r"C:\fake\Aside\CLI\current\aside.exe")
    with patch.dict(os.environ, {"LOCALAPPDATA": str(temp_dir)}):
        with patch("shutil.which", return_value=str(fake_exe)):
            resolved = get_aside_binary_path()
            assert resolved == fake_exe


def test_setup_aside_local_engine(temp_dir):
    """Verify setup_aside_local_engine writes settings.json and credentials.json correctly."""
    user_dir = temp_dir / ".aside" / "u" / "0"
    user_dir.mkdir(parents=True, exist_ok=True)

    with patch("local_ai_launcher.backends.aside.get_aside_config_dir", return_value=user_dir), \
         patch("local_ai_launcher.backends.aside.get_aside_accounts", return_value=["0"]):
        success = setup_aside_local_engine(
            llm_base_url="http://127.0.0.1:8080/v1",
            active_model="qwen-38_27b-blackwell-256k-think",
            ctx_size=65536,
        )
        assert success is True

        settings_file = user_dir / "settings.json"
        creds_file = user_dir / "credentials.json"
        assert settings_file.exists()
        assert creds_file.exists()

        settings_data = json.loads(settings_file.read_text(encoding="utf-8"))
        assert settings_data["defaultModel"]["modelId"] == "local-engine"
        assert settings_data["defaultModel"]["provider"] == "local"
        assert "customModels" in settings_data

        local_model = next((m for m in settings_data["customModels"] if m.get("id") == "local-engine"), None)
        assert local_model is not None
        assert local_model["provider"] == "local"
        assert local_model["api"] == "openai-completions"
        assert local_model["model"] == "qwen-38_27b-blackwell-256k-think"
        assert local_model["baseUrl"] == "http://127.0.0.1:8080/v1"
        assert local_model["contextWindow"] == 65536
        assert local_model["headers"]["Authorization"] == "Bearer dummy"

        creds_data = json.loads(creds_file.read_text(encoding="utf-8"))
        assert creds_data.get("local") == "dummy"


def test_get_aside_mcp_server_config():
    """Verify Aside MCP stdio configuration schema."""
    fake_exe = Path(r"C:\Aside\CLI\current\aside.exe")
    with patch("local_ai_launcher.backends.aside.get_aside_binary_path", return_value=fake_exe):
        mcp_cfg = get_aside_mcp_server_config()
        assert mcp_cfg is not None
        assert mcp_cfg["type"] == "stdio"
        if sys.platform == "win32":
            assert mcp_cfg["command"] == "cmd.exe"
            assert str(fake_exe) in mcp_cfg["args"]
            assert "mcp" in mcp_cfg["args"]
        else:
            assert mcp_cfg["command"] == str(fake_exe)
            assert mcp_cfg["args"] == ["mcp"]


def test_build_ps_env_aside_path_injection():
    """Verify build_ps_env adds Aside CLI directory to $env:PATH when enabled."""
    fake_exe = Path(r"C:\Users\test\AppData\Local\Aside\CLI\current\aside.exe")
    fake_dir = r"C:\Users\test\AppData\Local\Aside\CLI\current"

    with patch.object(config, "aside_browser_enabled", True), \
         patch("local_ai_launcher.paths.get_aside_binary_path", return_value=fake_exe), \
         patch("pathlib.Path.is_file", return_value=True):
        env_vars = {}
        cmd = build_ps_env(env_vars, harness_name="pi")

        assert env_vars.get("ASIDE_CLI_DIR") == fake_dir
        assert env_vars.get("ASIDE_BINARY") == str(fake_exe)
        assert fake_dir in cmd
        assert "$env:PATH" in cmd

    # Disabled case
    with patch.object(config, "aside_browser_enabled", False):
        env_vars = {}
        cmd = build_ps_env(env_vars, harness_name="pi")
        assert "ASIDE_CLI_DIR" not in env_vars
        assert "Aside" not in cmd


def test_pi_setup_mcp_aside(temp_dir):
    """Verify setup_pi_mcp_config injects Aside MCP into Pi's mcp.json."""
    fake_exe = Path(r"C:\Aside\CLI\current\aside.exe")

    with patch.object(config, "aside_browser_enabled", True), \
         patch("local_ai_launcher.backends.aside.get_aside_binary_path", return_value=fake_exe):
        setup_pi_mcp_config(temp_dir)

        mcp_file = temp_dir / "agent" / "mcp.json"
        assert mcp_file.exists()
        data = json.loads(mcp_file.read_text(encoding="utf-8"))
        assert "aside" in data.get("mcpServers", {})
        if sys.platform == "win32":
            assert data["mcpServers"]["aside"]["command"] == "cmd.exe"
            assert str(fake_exe) in data["mcpServers"]["aside"]["args"]
        else:
            assert data["mcpServers"]["aside"]["command"] == str(fake_exe)

    # Disable Aside MCP
    with patch.object(config, "aside_browser_enabled", False):
        setup_pi_mcp_config(temp_dir)
        data = json.loads(mcp_file.read_text(encoding="utf-8"))
        assert "aside" not in data.get("mcpServers", {})


def test_claude_setup_mcp_aside(temp_dir):
    """Verify setup_claude_cli_mcp_config and setup_claude_desktop_config inject Aside MCP."""
    fake_exe = Path(r"C:\Aside\CLI\current\aside.exe")

    with patch.object(config, "aside_browser_enabled", True), \
         patch("local_ai_launcher.backends.aside.get_aside_binary_path", return_value=fake_exe):
        # 1. Claude Code CLI (.mcp.json)
        mcp_path = setup_claude_cli_mcp_config(
            project_dir=temp_dir,
            llm_base_url="http://127.0.0.1:8080",
            mode="local",
        )
        assert mcp_path.exists()
        cli_data = json.loads(mcp_path.read_text(encoding="utf-8"))
        assert "aside" in cli_data.get("mcpServers", {})
        if sys.platform == "win32":
            assert cli_data["mcpServers"]["aside"]["command"] == "cmd.exe"
            assert str(fake_exe) in cli_data["mcpServers"]["aside"]["args"]
        else:
            assert cli_data["mcpServers"]["aside"]["command"] == str(fake_exe)

        # Settings local approval
        settings_file = temp_dir / ".claude" / "settings.local.json"
        assert settings_file.exists()
        sdata = json.loads(settings_file.read_text(encoding="utf-8"))
        assert "aside" in sdata.get("enabledMcpjsonServers", [])

        # 2. Claude Desktop
        desktop_cfg_path = temp_dir / "claude_desktop_config.json"
        setup_claude_desktop_config(
            config_path=desktop_cfg_path,
            llm_base_url="http://127.0.0.1:8080",
            desktop_mode="gateway",
        )
        assert desktop_cfg_path.exists()
        desk_data = json.loads(desktop_cfg_path.read_text(encoding="utf-8"))
        assert "aside" in desk_data.get("mcpServers", {})
        if sys.platform == "win32":
            assert desk_data["mcpServers"]["aside"]["command"] == "cmd.exe"
            assert str(fake_exe) in desk_data["mcpServers"]["aside"]["args"]
        else:
            assert desk_data["mcpServers"]["aside"]["command"] == str(fake_exe)


def test_antigravity_setup_mcp_aside(temp_dir):
    """Verify setup_antigravity_mcp_config injects Aside MCP into mcp_config.json."""
    fake_exe = Path(r"C:\Aside\CLI\current\aside.exe")

    with patch.object(config, "aside_browser_enabled", True), \
         patch("local_ai_launcher.backends.aside.get_aside_binary_path", return_value=fake_exe):
        mcp_cfg_file = setup_antigravity_mcp_config(gemini_dir=temp_dir)
        assert mcp_cfg_file.exists()
        data = json.loads(mcp_cfg_file.read_text(encoding="utf-8"))
        assert "aside" in data.get("mcpServers", {})
        if sys.platform == "win32":
            assert data["mcpServers"]["aside"]["command"] == "cmd.exe"
            assert str(fake_exe) in data["mcpServers"]["aside"]["args"]
        else:
            assert data["mcpServers"]["aside"]["command"] == str(fake_exe)


def test_orca_and_t3_aside_env(temp_dir):
    """Verify Orca ADE and T3 Code receive Aside CLI path in their runtime environment."""
    fake_exe = Path(r"C:\Aside\CLI\current\aside.exe")
    fake_dir = r"C:\Aside\CLI\current"

    with patch.object(config, "aside_browser_enabled", True), \
         patch("local_ai_launcher.paths.get_aside_binary_path", return_value=fake_exe), \
         patch("pathlib.Path.is_file", return_value=True):
        # Orca
        with patch.dict(os.environ, {"APPDATA": str(temp_dir)}):
            setup_orca_config(temp_dir, "http://127.0.0.1:8080", "test-model")
            settings_file = temp_dir / "Orca" / "profiles" / "local-default" / "orca-data.json"
            assert settings_file.exists()
            orca_data = json.loads(settings_file.read_text(encoding="utf-8"))
            pi_env = orca_data["settings"]["agentDefaultEnv"]["pi"]
            assert pi_env.get("ASIDE_CLI_DIR") == fake_dir
            assert pi_env.get("ASIDE_BINARY") == str(fake_exe)

        # T3
        setup_t3_config(temp_dir, "http://127.0.0.1:8080", "test-model")
        t3_cfg = temp_dir / "opencode_pi.json"
        assert t3_cfg.exists()


def test_harness_model_and_registry_aside():
    """Verify harness characteristics and registry mapping for Aside."""
    assert "aside" in HARNESS_SPECS
    spec = HARNESS_SPECS["aside"]
    assert spec.canonical_name == "aside"
    assert spec.is_cli is True
    assert spec.is_meta_harness is False
    assert spec.offline_support == "always"
    assert spec.is_offline_capable() is True

    # Test alias mappings
    assert HARNESS_ALIAS_MAP["aside"] == "aside"
    assert HARNESS_ALIAS_MAP["aside-browser"] == "aside"
    assert HARNESS_ALIAS_MAP["aside-cli"] == "aside"

    # Test HARNESS_REGISTRY
    assert "aside" in HARNESS_REGISTRY
    assert HARNESS_REGISTRY["aside"] == AsideAdapter
    assert HARNESS_REGISTRY["aside-browser"] == AsideAdapter
    assert HARNESS_REGISTRY["aside-cli"] == AsideAdapter


def test_config_aside_parsing(temp_dir):
    """Verify LauncherConfig parses [aside] section and boolean coercion."""
    cfg = LauncherConfig()
    assert cfg.aside_browser_enabled is True
    assert cfg.aside_browser_model == "local-engine"
    assert cfg.aside_browser_auto_sync is True
    assert cfg.aside_browser_headless is False

    # 1. Table override [aside] via _flatten_overrides
    raw_table = {
        "aside": {
            "enable": False,
            "model": "custom-aside-model",
            "auto_sync": False,
            "headless": True,
        }
    }
    flat = _flatten_overrides(raw_table)
    assert flat["aside_browser_enabled"] is False
    assert flat["aside_browser_model"] == "custom-aside-model"
    assert flat["aside_browser_auto_sync"] is False
    assert flat["aside_browser_headless"] is True

    # 2. TOML file load
    toml_file = temp_dir / "launcher.toml"
    toml_file.write_text("""
[aside]
enable = false
model = "custom-aside-model"
auto_sync = false
headless = true
""", encoding="utf-8")
    cfg_toml = LauncherConfig(config_file=toml_file)
    assert cfg_toml.aside_browser_enabled is False
    assert cfg_toml.aside_browser_model == "custom-aside-model"
    assert cfg_toml.aside_browser_auto_sync is False
    assert cfg_toml.aside_browser_headless is True

    # 3. Boolean sanitization against truthiness strings and negative numbers
    assert cfg._sanitize_field("aside_browser_enabled", "off") is False
    assert cfg._sanitize_field("aside_browser_auto_sync", -1) is False
    assert cfg._sanitize_field("aside_browser_headless", "yes") is True


def test_donsetch_and_searxng_routing_preserved():
    """Verify that DonSeTch is preserved for Pi and DeepSeek while SearXNG is preserved for other harnesses."""
    with patch.object(config, "pi_donsetch_enabled", True), \
         patch.object(config, "deepseek_donsetch_enabled", True), \
         patch.object(config, "searxng_enabled", True), \
         patch.object(config, "searxng_mode", "wsl_docker"):
        # Pi should NOT have SEARXNG_URL
        pi_env = {}
        build_ps_env(pi_env, harness_name="pi")
        assert "SEARXNG_URL" not in pi_env

        # DeepSeek should NOT have SEARXNG_URL
        dsh_env = {}
        build_ps_env(dsh_env, harness_name="deepseek")
        assert "SEARXNG_URL" not in dsh_env

        # Hermes SHOULD have SEARXNG_URL
        hermes_env = {}
        build_ps_env(hermes_env, harness_name="hermes")
        assert "SEARXNG_URL" in hermes_env
        assert hermes_env["SEARXNG_URL"] == config.searxng_effective_url

        # Claude SHOULD have SEARXNG_URL
        claude_env = {}
        build_ps_env(claude_env, harness_name="claude")
        assert "SEARXNG_URL" in claude_env


def test_aside_not_installed_handling(temp_dir):
    """Verify Aside does not pollute MCP servers or create files when Aside is not installed."""
    with patch("local_ai_launcher.backends.aside.get_aside_binary_path", return_value=None), \
         patch("local_ai_launcher.paths.get_aside_binary_path", return_value=None), \
         patch("local_ai_launcher.backends.aside.get_aside_app_path", return_value=None), \
         patch("shutil.which", return_value=None), \
         patch.object(config, "aside_browser_enabled", True):
        # 1. get_aside_mcp_server_config returns None when require_installed=True
        assert get_aside_mcp_server_config(require_installed=True) is None

        # 2. setup_pi_mcp_config does NOT register aside
        setup_pi_mcp_config(temp_dir)
        pi_mcp = temp_dir / "agent" / "mcp.json"
        data = json.loads(pi_mcp.read_text(encoding="utf-8"))
        assert "aside" not in data.get("mcpServers", {})

        # 3. setup_claude_cli_mcp_config does NOT register aside
        mcp_path = setup_claude_cli_mcp_config(
            project_dir=temp_dir,
            llm_base_url="http://127.0.0.1:8080",
            mode="local",
        )
        c_data = json.loads(mcp_path.read_text(encoding="utf-8"))
        assert "aside" not in c_data.get("mcpServers", {})

        settings_local = temp_dir / ".claude" / "settings.local.json"
        if settings_local.exists():
            s_data = json.loads(settings_local.read_text(encoding="utf-8"))
            assert "aside" not in s_data.get("enabledMcpjsonServers", [])

        # 4. setup_antigravity_mcp_config does NOT register aside
        ag_mcp = setup_antigravity_mcp_config(gemini_dir=temp_dir)
        ag_data = json.loads(ag_mcp.read_text(encoding="utf-8"))
        assert "aside" not in ag_data.get("mcpServers", {})

        # 5. setup_aside_local_engine cleanly skips and returns False
        empty_home = temp_dir / "empty_user_home"
        with patch("local_ai_launcher.backends.aside.ASIDE_USER_HOME", empty_home):
            res = setup_aside_local_engine("http://127.0.0.1:8080/v1")
            assert res is False
            assert not (empty_home / "u").exists()

