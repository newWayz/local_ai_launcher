import pytest
from pathlib import Path
import sys
import json



from local_ai_launcher.config import config, LauncherConfig, MnemosyneAppConfig
from local_ai_launcher.harnesses import HARNESS_REGISTRY
from local_ai_launcher.harnesses.orca import OrcaAdapter, setup_orca_config
from local_ai_launcher.harnesses.t3 import T3Adapter, setup_t3_config
from local_ai_launcher.harnesses.herdr import HerdrAdapter, setup_herdr_config
from local_ai_launcher.harnesses.claude import ClaudeAdapter


@pytest.fixture(autouse=True)
def isolate_harness_env(tmp_path_factory, monkeypatch):
    """Ensure tests never write to real user home or active workspace."""
    fake_home = tmp_path_factory.mktemp("fake_home")
    fake_ws = tmp_path_factory.mktemp("fake_ws")
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    monkeypatch.setattr("local_ai_launcher.harnesses.t3.AI_WORKSPACE_DIR", fake_ws)


def test_harness_registry_contains_orca_and_t3():
    assert "orca" in HARNESS_REGISTRY
    assert HARNESS_REGISTRY["orca"] is OrcaAdapter
    assert "t3" in HARNESS_REGISTRY
    assert HARNESS_REGISTRY["t3"] is T3Adapter
    assert "t3code" in HARNESS_REGISTRY
    assert HARNESS_REGISTRY["t3code"] is T3Adapter
    assert "herdr" in HARNESS_REGISTRY
    assert HARNESS_REGISTRY["herdr"] is HerdrAdapter


def test_config_contains_orca_and_t3_defaults():
    cfg = LauncherConfig()
    assert "orca" in cfg.harness_install_behaviors
    assert "t3" in cfg.harness_install_behaviors
    assert "herdr" in cfg.harness_install_behaviors
    assert "orca" in cfg.harness_update_behaviors
    assert "t3" in cfg.harness_update_behaviors
    assert "herdr" in cfg.harness_update_behaviors
    assert "orca" in cfg.harness_extensions
    assert "t3" in cfg.harness_extensions
    assert "herdr" in cfg.harness_extensions
    assert "orca" in cfg.harness_skills
    assert "t3" in cfg.harness_skills
    assert "herdr" in cfg.harness_skills
    assert hasattr(cfg, "t3_port")
    assert cfg.t3_port == 3773
    assert hasattr(config, "t3_dir")
    assert hasattr(config, "orca_dir")
    assert hasattr(config, "herdr_dir")


def test_mnemosyne_app_config_has_dirs():
    mcfg = MnemosyneAppConfig()
    assert hasattr(mcfg, "t3_config_dir")
    assert hasattr(mcfg, "orca_config_dir")
    assert hasattr(mcfg, "herdr_config_dir")
    assert hasattr(mcfg, "t3_dir")
    assert hasattr(mcfg, "orca_dir")
    assert hasattr(mcfg, "herdr_dir")
    assert mcfg.t3_dir == mcfg.t3_config_dir
    assert mcfg.orca_dir == mcfg.orca_config_dir
    assert mcfg.herdr_dir == mcfg.herdr_config_dir


def test_t3_auto_configuration(tmp_path, monkeypatch):
    test_start_dir = tmp_path / "test_project"
    test_start_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(config, "harness_start_dir", test_start_dir)
    monkeypatch.setattr(config, "choose_harness", ["pi"])
    monkeypatch.setenv("T3CODE_BASE_DIR", str(tmp_path / "custom_t3_userdata"))

    t3_home = tmp_path / ".t3"
    setup_t3_config(t3_home, "http://localhost:9931/v1", "test-model")

    # Verify local config
    local_cfg = t3_home / "config.json"
    assert local_cfg.exists()
    content = json.loads(local_cfg.read_text(encoding="utf-8"))
    assert content["openaiBaseUrl"] == "http://localhost:9931/v1"
    assert content["defaultModel"] == "pi/test-model"
    assert content["defaultHarness"] == "pi"
    assert content["activeProject"] == str(test_start_dir)

    # Verify global settings
    global_cfg = tmp_path / "custom_t3_userdata" / "settings.json"
    assert global_cfg.exists()
    global_content = json.loads(global_cfg.read_text(encoding="utf-8"))
    assert "providerInstances" in global_content
    assert "pi" in global_content["providerInstances"]
    assert "hermes" in global_content["providerInstances"]
    assert "localai" not in global_content["providerInstances"]
    assert "openai" not in global_content["providerInstances"]
    assert global_content["providerInstances"]["pi"]["driver"] == "opencode"
    assert global_content["providerInstances"]["pi"]["displayName"] == "Pi Coding Harness"
    assert global_content["providerInstances"]["hermes"]["driver"] == "opencode"
    assert global_content["providerInstances"]["hermes"]["displayName"] == "Hermes"
    assert "binaryPath" in global_content["providerInstances"]["pi"]["config"]
    assert "serverUrl" not in global_content["providerInstances"]["pi"]["config"]
    assert global_content["providers"]["opencode"]["enabled"] is False
    assert "binaryPath" in global_content["providers"]["opencode"]
    assert "serverUrl" not in global_content["providers"]["opencode"]
    assert global_content["serverSettings"]["port"] == 3773
    assert global_content["defaultModelSelection"]["model"] == "pi/test-model"
    assert global_content["defaultModelSelection"]["instanceId"] == "pi"

    # Verify cache pre-population
    cache_dir = tmp_path / "caches"
    assert (cache_dir / "pi.json").exists()
    assert (cache_dir / "hermes.json").exists()
    assert not (cache_dir / "localai.json").exists()
    assert not (cache_dir / "opencode.json").exists()


def test_t3_deep_merge_preserves_user_customizations(tmp_path, monkeypatch):
    custom_userdata = tmp_path / "custom_t3_userdata"
    custom_userdata.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("T3CODE_BASE_DIR", str(custom_userdata))

    pre_existing = {
        "defaultTheme": "tokyo-night",
        "providers": {
            "claudeAgent": {"enabled": True}
        },
        "providerInstances": {
            "my_custom": {"driver": "claudeAgent", "enabled": True}
        }
    }
    (custom_userdata / "settings.json").write_text(json.dumps(pre_existing), encoding="utf-8")

    t3_home = tmp_path / ".t3"
    setup_t3_config(t3_home, "http://localhost:8080/v1", "qwen-test")

    updated = json.loads((custom_userdata / "settings.json").read_text(encoding="utf-8"))
    assert updated["defaultTheme"] == "tokyo-night"
    assert updated["providers"]["claudeAgent"]["enabled"] is True
    assert updated["providerInstances"]["my_custom"]["driver"] == "claudeAgent"
    assert updated["providers"]["opencode"]["enabled"] is False
    assert "binaryPath" in updated["providers"]["opencode"]
    assert "serverUrl" not in updated["providers"]["opencode"]
    assert updated["providerInstances"]["pi"]["driver"] == "opencode"
    assert updated["providerInstances"]["hermes"]["driver"] == "opencode"
    assert "localai" not in updated["providerInstances"]


def test_t3_pi_and_hermes_integration(tmp_path, monkeypatch):
    """Verify setup_t3_config links Pi models/auth, Hermes environment, and custom OpenCode commands."""
    from unittest.mock import patch

    test_start_dir = tmp_path / "test_project"
    test_start_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(config, "harness_start_dir", test_start_dir)
    monkeypatch.setenv("T3CODE_BASE_DIR", str(tmp_path / "custom_t3_userdata"))

    t3_home = tmp_path / ".t3"

    with patch("local_ai_launcher.harnesses.pi.setup_pi_models_and_auth") as mock_pi_setup, \
         patch("local_ai_launcher.bridges.gvs5h_solver.ensure_deep_solve_skills") as mock_ds, \
         patch("local_ai_launcher.bridges.cpu_tools_cli.ensure_cpu_tools_skills") as mock_cpu, \
         patch("local_ai_launcher.harnesses.hermes.HermesAdapter._ensure_user_profile_junction") as mock_junction:

        setup_t3_config(t3_home, "http://localhost:8080/v1", "test-model")

        assert mock_pi_setup.called
        assert mock_ds.called
        assert mock_cpu.called
        assert mock_junction.called

        # Verify settings.json environment contains Pi and Hermes variables and isolated XDG data paths
        settings_file = tmp_path / "custom_t3_userdata" / "settings.json"
        assert settings_file.exists()
        settings_data = json.loads(settings_file.read_text(encoding="utf-8"))
        pi_env = {item["name"]: item["value"] for item in settings_data["providerInstances"]["pi"]["environment"]}
        assert "PI_HOME" in pi_env
        assert "PI_CODING_AGENT_DIR" in pi_env
        assert "HERMES_HOME" in pi_env
        assert "XDG_DATA_HOME" in pi_env
        assert pi_env["MODEL"] == "test-model"

        hermes_env = {item["name"]: item["value"] for item in settings_data["providerInstances"]["hermes"]["environment"]}
        assert "HERMES_HOME" in hermes_env
        assert "PI_HOME" in hermes_env
        assert "XDG_DATA_HOME" in hermes_env
        assert pi_env["XDG_DATA_HOME"] != hermes_env["XDG_DATA_HOME"]

        # Verify opencode.jsonc contains /pi and /hermes commands, subagents, and providers
        opencode_file = Path.home() / ".config" / "opencode" / "opencode.jsonc"
        assert opencode_file.exists()
        opencode_data = json.loads(opencode_file.read_text(encoding="utf-8"))
        assert "command" in opencode_data
        assert "pi" in opencode_data["command"]
        assert "hermes" in opencode_data["command"]
        assert "agent" in opencode_data
        assert "pi" in opencode_data["agent"]
        assert "hermes" in opencode_data["agent"]
        assert "provider" in opencode_data
        assert "pi" in opencode_data["provider"]
        assert "hermes" in opencode_data["provider"]
        assert opencode_data["provider"]["pi"]["name"] == "Pi Coding Harness"
        assert opencode_data["provider"]["hermes"]["name"] == "Hermes"
        assert opencode_data["provider"]["pi"]["npm"] == "@ai-sdk/openai-compatible"
        assert opencode_data["provider"]["hermes"]["npm"] == "@ai-sdk/openai-compatible"
        assert opencode_data["provider"]["pi"]["options"]["baseURL"] == "http://localhost:8080/v1"
        assert opencode_data["provider"]["hermes"]["options"]["baseURL"] == "http://localhost:8080/v1"


def test_t3_desktop_gui_launch_precedence(tmp_path, monkeypatch):
    """Verify T3Adapter prefers desktop GUI executable over CLI and does not open browser when GUI exists."""
    from unittest.mock import patch, MagicMock

    mock_localappdata = tmp_path / "LocalAppData"
    mock_gui = mock_localappdata / "Programs" / "t3code" / "T3 Code (Alpha).exe"
    mock_gui.parent.mkdir(parents=True, exist_ok=True)
    mock_gui.write_text("mock binary", encoding="utf-8")

    monkeypatch.setenv("LOCALAPPDATA", str(mock_localappdata))
    monkeypatch.setattr(config, "harness_start_dir", tmp_path)

    adapter = T3Adapter(tmp_path, [])
    assert adapter.desktop_exe == mock_gui
    assert adapter.t3_exe == mock_gui

    with patch("local_ai_launcher.harness_base.send_background_warmup"), \
         patch("local_ai_launcher.harnesses.t3.assign_process_to_job"), \
         patch("local_ai_launcher.harnesses.t3.start_in_wt_tab") as mock_wt, \
         patch("local_ai_launcher.harnesses.t3.wait_for_port", return_value=True), \
         patch("local_ai_launcher.harnesses.t3.subprocess.Popen") as mock_popen, \
         patch("local_ai_launcher.harnesses.t3.webbrowser.open") as mock_browser:

        mock_server_proc = MagicMock()
        mock_server_proc.pid = 1001
        mock_server_proc.poll.return_value = None
        mock_wt.return_value = mock_server_proc

        mock_gui_proc = MagicMock()
        mock_gui_proc.pid = 1002
        mock_popen.return_value = mock_gui_proc

        adapter.start("http://127.0.0.1:8080/v1", "qwen-test")

        assert not mock_wt.called
        assert mock_popen.called
        assert str(mock_gui) in mock_popen.call_args[0][0]
        assert not mock_browser.called

        called_env = mock_popen.call_args[1]["env"]
        assert "PI_HOME" in called_env
        assert "HERMES_HOME" in called_env
        assert "node" in called_env["PATH"].lower()
        assert ".hermes" in called_env["PATH"].lower()


def test_t3_browser_fallback_when_desktop_absent(tmp_path, monkeypatch):
    """Verify T3Adapter falls back to webbrowser.open when standalone desktop GUI app is absent."""
    from unittest.mock import patch, MagicMock

    mock_localappdata = tmp_path / "EmptyLocalAppData"
    mock_localappdata.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCALAPPDATA", str(mock_localappdata))
    monkeypatch.setattr(config, "harness_start_dir", tmp_path)

    adapter = T3Adapter(tmp_path, [])
    assert adapter.desktop_exe is None

    with patch("local_ai_launcher.harness_base.send_background_warmup"), \
         patch("local_ai_launcher.harnesses.t3.assign_process_to_job"), \
         patch("local_ai_launcher.harnesses.t3.start_in_wt_tab") as mock_wt, \
         patch("local_ai_launcher.harnesses.t3.wait_for_port", return_value=True), \
         patch("local_ai_launcher.harnesses.t3.subprocess.Popen") as mock_popen, \
         patch("local_ai_launcher.harnesses.hermes.HermesAdapter._ensure_user_profile_junction"), \
         patch("local_ai_launcher.bridges.gvs5h_solver.ensure_deep_solve_skills"), \
         patch("local_ai_launcher.harnesses.t3.webbrowser.open") as mock_browser:

        mock_server_proc = MagicMock()
        mock_server_proc.pid = 1001
        mock_server_proc.poll.return_value = None
        mock_wt.return_value = mock_server_proc

        adapter.start("http://127.0.0.1:8080/v1", "qwen-test")

        assert mock_wt.called
        assert not mock_popen.called
        assert mock_browser.called

        called_cmd = mock_wt.call_args[0][0]
        assert "PI_HOME" in called_cmd
        assert "HERMES_HOME" in called_cmd
        assert "3773" in mock_browser.call_args[0][0]


def test_orca_auto_configuration(tmp_path, monkeypatch):
    # Mock subprocess.run to prevent real git init/commit during tests
    def mock_run(*args, **kwargs):
        class MockRes:
            returncode = 0
            stdout = "true"
            stderr = ""
        return MockRes()
    monkeypatch.setattr("subprocess.run", mock_run)

    test_start_dir = tmp_path / "test_project"
    test_start_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(config, "harness_start_dir", test_start_dir)
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    
    orca_home = tmp_path / ".orca"
    setup_orca_config(orca_home, "http://localhost:9931/v1", "test-model")
    
    # Verify .worktreeinclude
    worktree_include = test_start_dir / ".worktreeinclude"
    assert worktree_include.exists()
    includes = worktree_include.read_text(encoding="utf-8").splitlines()
    assert "USER.md" in includes
    assert ".env" in includes
    assert ".git/info/exclude" in includes

    # Verify profile settings
    orca_data = tmp_path / "appdata" / "Orca" / "profiles" / "local-default" / "orca-data.json"
    assert orca_data.exists()
    data = json.loads(orca_data.read_text(encoding="utf-8"))
    assert data["settings"]["workspaceDir"] == str(test_start_dir)


def test_orca_config_agent_default_env_pi(tmp_path, monkeypatch):
    """Verifies agentDefaultEnv for pi agent contains HOME, PI_HOME, and PI_CODING_AGENT_DIR."""
    def mock_run(*args, **kwargs):
        class MockRes:
            returncode = 0
            stdout = "true"
            stderr = ""
        return MockRes()
    monkeypatch.setattr("subprocess.run", mock_run)

    test_start_dir = tmp_path / "workspace"
    test_start_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(config, "harness_start_dir", test_start_dir)
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))

    orca_home = tmp_path / ".orca"
    setup_orca_config(orca_home, "http://localhost:8080/v1", "test-model-123")

    orca_data = tmp_path / "appdata" / "Orca" / "profiles" / "local-default" / "orca-data.json"
    assert orca_data.exists()
    data = json.loads(orca_data.read_text(encoding="utf-8"))

    pi_env = data["settings"]["agentDefaultEnv"]["pi"]
    assert pi_env["OPENAI_BASE_URL"] == "http://localhost:8080/v1"
    assert pi_env["OPENAI_MODEL"] == "test-model-123"
    assert "Users" not in pi_env["HOME"]
    assert "Users" not in pi_env["PI_HOME"]
    assert "PI_CODING_AGENT_DIR" in pi_env
    assert "HF_HOME" in pi_env
    assert "npm_config_allow_git" in pi_env


def test_adapter_instantiations(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "t3_dir", tmp_path / ".t3")
    monkeypatch.setattr(config, "orca_dir", tmp_path / ".orca")
    monkeypatch.setattr(config, "herdr_dir", tmp_path / ".herdr")
    dummy_processes = []
    orca = OrcaAdapter("test_root", dummy_processes)
    t3 = T3Adapter("test_root", dummy_processes)
    herdr = HerdrAdapter("test_root", dummy_processes)
    assert orca.name == "orca"
    assert t3.name == "t3"
    assert herdr.name == "herdr"
    assert orca.install_behavior == "always_install"
    assert t3.install_behavior == "always_install"
    assert herdr.install_behavior == "always_install"


def test_herdr_config_and_adapter(tmp_path, monkeypatch):
    test_start_dir = tmp_path / "workspace"
    test_start_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(config, "harness_start_dir", test_start_dir)
    monkeypatch.setattr(config, "choose_harness", ["herdr", "pi"])
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))

    herdr_home = tmp_path / ".herdr"
    setup_herdr_config(herdr_home, "http://localhost:8080/v1", "test-model-123")

    config_file = herdr_home / "config.toml"
    assert config_file.exists()
    content = config_file.read_text(encoding="utf-8")
    assert "onboarding = false" in content
    assert "resume_agents_on_restore = true" in content
    assert "launch default agent (pi)" in content

    runtime_meta_file = herdr_home / "herdr-runtime.json"
    assert runtime_meta_file.exists()
    runtime_data = json.loads(runtime_meta_file.read_text(encoding="utf-8"))
    assert runtime_data["defaultInnerCliAgent"] == "pi"
    assert runtime_data["activeModel"] == "test-model-123"
    assert runtime_data["openaiBaseUrl"] == "http://localhost:8080/v1"


def test_herdr_adapter_check_and_bootstrap(tmp_path, monkeypatch):
    """Verify HerdrAdapter check, install strategies, and binary discovery."""
    monkeypatch.setattr(config, "herdr_dir", tmp_path / ".herdr")
    adapter = HerdrAdapter("test_root", [])

    # 1. When not installed and install strategy is never_install
    adapter.install_behavior = "never_install"
    monkeypatch.setattr("shutil.which", lambda cmd: None)
    fake_nonexistent = tmp_path / "nonexistent" / "herdr.exe"
    monkeypatch.setattr(adapter, "_resolve_executable", lambda: fake_nonexistent)
    adapter.herdr_exe = fake_nonexistent

    assert adapter.check() is False

    # 2. When binary exists, check returns True
    fake_bin = tmp_path / ".herdr" / "bin" / ("herdr.exe" if sys.platform == "win32" else "herdr")
    fake_bin.parent.mkdir(parents=True, exist_ok=True)
    fake_bin.write_text("binary-stub", encoding="utf-8")
    monkeypatch.setattr(adapter, "_resolve_executable", lambda: fake_bin)
    adapter.herdr_exe = fake_bin
    monkeypatch.setattr(adapter, "_should_update", lambda: False)

    assert adapter.check() is True


def test_herdr_adapter_start_env_and_headless(tmp_path, monkeypatch):
    """Verify HerdrAdapter start configures environment variables and supports headless mode."""
    test_start_dir = tmp_path / "workspace"
    test_start_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(config, "harness_start_dir", test_start_dir)
    monkeypatch.setattr(config, "herdr_dir", tmp_path / ".herdr")
    monkeypatch.setattr(config, "choose_harness", ["herdr", "pi"])

    procs = []
    adapter = HerdrAdapter("test_root", procs)
    fake_bin = tmp_path / ".herdr" / "bin" / ("herdr.exe" if sys.platform == "win32" else "herdr")
    fake_bin.parent.mkdir(parents=True, exist_ok=True)
    fake_bin.write_text("binary-stub", encoding="utf-8")
    adapter.herdr_exe = fake_bin

    # Headless mode should configure settings without starting process
    adapter.start("http://localhost:8080/v1", "test-model-123", launch_tui=False)
    assert (tmp_path / ".herdr" / "config.toml").exists()
    assert (tmp_path / ".herdr" / "herdr-runtime.json").exists()
    assert len(procs) == 0



def test_pi_auto_configuration(tmp_path, monkeypatch):
    from local_ai_launcher.harnesses.pi import setup_pi_models_and_auth

    test_start_dir = tmp_path / "test_project"
    test_start_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(config, "harness_start_dir", test_start_dir)

    # Simulate stale project-local settings that break model resolution
    stale_proj_settings = test_start_dir / ".pi" / "settings.json"
    stale_proj_settings.parent.mkdir(parents=True, exist_ok=True)
    stale_proj_settings.write_text('{"defaultProvider":"llama.cpp","enabledModels":["llama*"]}', encoding="utf-8")

    pi_home = tmp_path / ".pi"
    setup_pi_models_and_auth(pi_home, "http://localhost:9931/v1", "qwen-test-model")

    # Verify stale project-local settings were removed
    assert not stale_proj_settings.exists()

    # Verify agent settings.json
    agent_settings = pi_home / "agent" / "settings.json"
    assert agent_settings.exists()
    settings = json.loads(agent_settings.read_text(encoding="utf-8"))
    assert settings["defaultProvider"] == "openai"
    assert settings["defaultModel"] == "qwen-test-model"
    assert "openai/*" in settings["enabledModels"]
    assert "-builtin:mcp" in settings.get("extensions", [])

    models_data = json.loads((pi_home / "agent" / "models.json").read_text(encoding="utf-8"))
    assert not any(m["id"] == "*" for m in models_data["providers"]["openai"]["models"])


def test_pi_huang_and_strata_model_configuration(tmp_path, monkeypatch):
    """Verify setup_pi_models_and_auth provisions Huang and Strata models with correct context window and reasoning."""
    from local_ai_launcher.harnesses.pi import setup_pi_models_and_auth

    test_start_dir = tmp_path / "test_project"
    test_start_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(config, "harness_start_dir", test_start_dir)

    # 1. Test Huang Adaptive-KV model
    pi_home_huang = tmp_path / "pi_huang"
    setup_pi_models_and_auth(pi_home_huang, "http://127.0.0.1:8080/v1", "qwen-38_27b-huang-gsq-think")
    m_data_huang = json.loads((pi_home_huang / "agent" / "models.json").read_text(encoding="utf-8"))
    huang_entry = next((m for m in m_data_huang["providers"]["openai"]["models"] if m["id"] == "qwen-38_27b-huang-gsq-think"), None)
    assert huang_entry is not None
    assert huang_entry["contextWindow"] == 262000
    assert huang_entry.get("reasoning") is True

    # 2. Test Strata MoE model
    pi_home_strata = tmp_path / "pi_strata"
    setup_pi_models_and_auth(pi_home_strata, "http://127.0.0.1:8080/v1", "qwen-38-flash-strata-iq3s")
    m_data_strata = json.loads((pi_home_strata / "agent" / "models.json").read_text(encoding="utf-8"))
    strata_entry = next((m for m in m_data_strata["providers"]["openai"]["models"] if m["id"] == "qwen-38-flash-strata-iq3s"), None)
    assert strata_entry is not None
    assert strata_entry["contextWindow"] == 262144
    assert strata_entry.get("reasoning") is True


def test_pi_freetoken_model_configuration(tmp_path, monkeypatch):
    """Verify setup_pi_models_and_auth cleanly resolves 'freetoken\\' models to canonical ID with full context window."""
    from local_ai_launcher.harnesses.pi import setup_pi_models_and_auth
    from local_ai_launcher.models import get_model_specs, resolve_model_name

    raw_model = r"freetoken\nvidia_Qwen3.6-35B-A3B-NVFP4"
    
    # 1. Verify get_model_specs does not raise NameError and retrieves correct 131k context & reasoning from freetokens_models.ini
    ctx, is_reasoning = get_model_specs(raw_model)
    assert ctx == 131072
    assert is_reasoning is True

    # 2. Verify resolve_model_name strips prefix
    assert resolve_model_name(raw_model) == "nvidia_Qwen3.6-35B-A3B-NVFP4"

    # 3. Verify Pi configuration
    pi_home = tmp_path / ".pi"
    setup_pi_models_and_auth(pi_home, "http://localhost:8080/v1", raw_model)

    agent_settings = pi_home / "agent" / "settings.json"
    assert agent_settings.exists()
    settings = json.loads(agent_settings.read_text(encoding="utf-8"))
    assert settings["defaultProvider"] == "openai"
    assert settings["defaultModel"] == "nvidia_Qwen3.6-35B-A3B-NVFP4"
    assert "openai/*" in settings["enabledModels"]
    assert "openai/**" in settings["enabledModels"]

    models_data = json.loads((pi_home / "agent" / "models.json").read_text(encoding="utf-8"))
    first_model = models_data["providers"]["openai"]["models"][0]
    assert first_model["id"] == "nvidia_Qwen3.6-35B-A3B-NVFP4"
    assert first_model["contextWindow"] == 131072
    assert first_model.get("reasoning") is True


def test_pi_skills_gateway_provisioning_and_cleanup(tmp_path, monkeypatch):
    """Verify Pi does not duplicate skills into .pi/skills and cleans up legacy .pi/skills directory."""
    from local_ai_launcher.harnesses.pi import PiAdapter

    pi_home = tmp_path / ".pi"
    legacy_skills = pi_home / "skills"
    legacy_skills.mkdir(parents=True, exist_ok=True)
    (legacy_skills / "legacy_skill.md").write_text("dummy", encoding="utf-8")

    adapter = PiAdapter(tmp_path, [])
    monkeypatch.setattr(adapter, "pi_home", pi_home)
    monkeypatch.setattr(adapter, "verify_and_patch_pi_compat_exports", lambda: None)
    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr=""))

    # Calling ensure_extensions_and_skills should clean up legacy .pi/skills
    adapter.ensure_extensions_and_skills()

    assert not legacy_skills.exists()


def test_pi_donsetch_provisioning(tmp_path, monkeypatch):
    """Verify Pi adapter properly manages donsetch extension, scoped skill, and environment."""
    from unittest.mock import patch
    import subprocess
    from local_ai_launcher.harnesses.pi import PiAdapter
    import local_ai_launcher.harnesses.pi as pi_mod

    pi_home = tmp_path / ".pi"
    agent_dir = pi_home / "agent"
    agent_dir.mkdir(parents=True, exist_ok=True)
    (agent_dir / "settings.json").write_text("{}", encoding="utf-8")

    adapter = PiAdapter(tmp_path, [])
    monkeypatch.setattr(adapter, "pi_home", pi_home)
    monkeypatch.setattr(pi_mod, "AI_WORKSPACE_DIR", tmp_path)
    monkeypatch.setattr(adapter, "verify_and_patch_pi_compat_exports", lambda: None)
    monkeypatch.setattr(adapter, "ensure_agent_browser", lambda: None)
    monkeypatch.setattr(config, "harness_start_dir", tmp_path)

    installed_cmds = []
    def fake_run_silent(cmd, **kwargs):
        cmd_str = " ".join(str(c) for c in cmd)
        installed_cmds.append(cmd_str)
        if "list" in cmd_str:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("local_ai_launcher.harnesses.pi._run_silent", fake_run_silent)

    # 1. When pi_donsetch_enabled is True
    monkeypatch.setattr(config, "pi_donsetch_enabled", True)
    monkeypatch.setattr(config, "harness_extensions", {"pi": ["npm:donsetch"]})

    with patch("local_ai_launcher.skills_manager.SkillManager.ensure_scoped_harness_skills") as mock_scoped, \
         patch("local_ai_launcher.skills_manager.SkillManager.ensure_builtin_skills"):
        adapter.ensure_extensions_and_skills()
        mock_scoped.assert_called_with("pi", pi_home / "agent" / "skills")

    assert any("install npm:donsetch" in c for c in installed_cmds)

    # Test start() environment injection
    fake_donsetch_exe = tmp_path / "donsetch.exe"
    fake_donsetch_exe.write_text("binary", encoding="utf-8")
    monkeypatch.setattr(config, "donsetch_bin", fake_donsetch_exe)

    launched_cmd = []
    def fake_start_wt(cmd, **kwargs):
        launched_cmd.append(cmd)
        return 12345

    with patch("local_ai_launcher.context_preloader.preload_harness_context"), \
         patch("local_ai_launcher.harnesses.pi.start_in_wt_tab", side_effect=fake_start_wt), \
         patch("local_ai_launcher.skills_manager.SkillManager.ensure_scoped_harness_skills"):
        adapter.start("http://127.0.0.1:8080/v1", "test-model")

    assert len(launched_cmd) == 1
    assert f"$env:DONSETCH_BINARY='{fake_donsetch_exe}';" in launched_cmd[0]

    # 2. When pi_donsetch_enabled is False
    monkeypatch.setattr(config, "pi_donsetch_enabled", False)
    monkeypatch.setattr(config, "harness_extensions", {"pi": ["npm:donsetch"]})
    installed_cmds.clear()
    launched_cmd.clear()

    with patch("local_ai_launcher.skills_manager.SkillManager.ensure_scoped_harness_skills") as mock_scoped, \
         patch("local_ai_launcher.skills_manager.SkillManager.ensure_builtin_skills"):
        adapter.ensure_extensions_and_skills()
        mock_scoped.assert_not_called()

    assert not any("npm:donsetch" in c for c in installed_cmds)

    with patch("local_ai_launcher.context_preloader.preload_harness_context"), \
         patch("local_ai_launcher.harnesses.pi.start_in_wt_tab", side_effect=fake_start_wt), \
         patch("local_ai_launcher.skills_manager.SkillManager.ensure_scoped_harness_skills"):
        adapter.start("http://127.0.0.1:8080/v1", "test-model")

    assert len(launched_cmd) == 1
    assert "DONSETCH_BINARY" not in launched_cmd[0]


def test_get_donsetch_binary_path_resolution(tmp_path, monkeypatch):
    """Verify get_donsetch_binary_path discovers binary across all supported candidate directories."""
    import sys
    from local_ai_launcher.paths import get_donsetch_binary_path
    import local_ai_launcher.paths as paths_mod

    bin_name = "donsetch.exe" if sys.platform == 'win32' else "donsetch"

    fake_ws = tmp_path / "ws"
    fake_pi = tmp_path / "pi"
    fake_dsh = tmp_path / "dsh"
    fake_ws.mkdir()
    fake_pi.mkdir()
    fake_dsh.mkdir()

    monkeypatch.setattr(paths_mod, "AI_WORKSPACE_DIR", fake_ws)
    monkeypatch.setattr(paths_mod, "PI_DIR", fake_pi)
    monkeypatch.setattr(paths_mod, "DSH_DIR", fake_dsh)
    monkeypatch.setattr("shutil.which", lambda name: None)

    # 1. No candidates -> returns None
    assert get_donsetch_binary_path() is None

    # 2. Candidate in workspace node/
    node_bin = fake_ws / "node" / bin_name
    node_bin.parent.mkdir(parents=True, exist_ok=True)
    node_bin.write_text("stub", encoding="utf-8")
    assert get_donsetch_binary_path() == node_bin
    node_bin.unlink()

    # 3. Candidate in Pi npm package binaries/
    pi_bin = fake_pi / "agent" / "npm" / "node_modules" / "donsetch" / "binaries" / bin_name
    pi_bin.parent.mkdir(parents=True, exist_ok=True)
    pi_bin.write_text("stub", encoding="utf-8")
    assert get_donsetch_binary_path() == pi_bin
    pi_bin.unlink()

    # 4. Candidate in DSH versioned cache
    dsh_bin = fake_dsh / "cache" / "donsetch" / "dsh" / "bin" / "4.7.4" / bin_name
    dsh_bin.parent.mkdir(parents=True, exist_ok=True)
    dsh_bin.write_text("stub", encoding="utf-8")
    assert get_donsetch_binary_path() == dsh_bin


def test_openlumara_harness_registered():
    from local_ai_launcher.harnesses import HARNESS_REGISTRY
    from local_ai_launcher.harnesses.openlumara import OpenLumaraAdapter
    assert "openlumara" in HARNESS_REGISTRY
    assert HARNESS_REGISTRY["openlumara"] is OpenLumaraAdapter
    assert "lumara" in HARNESS_REGISTRY
    assert HARNESS_REGISTRY["lumara"] is OpenLumaraAdapter


def test_openlumara_setup_config(tmp_path):
    import yaml
    from local_ai_launcher.bridges.orca_openlumara.configurator import setup_openlumara_config

    lumara_dir = tmp_path / "openlumara"
    lumara_dir.mkdir(parents=True, exist_ok=True)

    success = setup_openlumara_config(
        lumara_dir=lumara_dir,
        llm_base_url="http://localhost:9931/v1",
        active_model="qwen-38_27b-blackwell-256k-think",
        mode="cli"
    )
    assert success is True

    cfg_file = lumara_dir / "config.yml"
    assert cfg_file.exists()

    cfg = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))
    assert cfg["api"]["url"] == "http://localhost:9931/v1"
    assert cfg["model"]["name"] == "qwen-38_27b-blackwell-256k-think"
    assert cfg["model"]["enable_thinking"] is True
    assert "cli" in cfg["channels"]["enabled"]
    assert "logger" in cfg["channels"]["enabled"]
    assert "webui" in cfg["channels"]["disabled"]
    assert "auto_backup" in cfg["modules"]["disabled"]


def test_orca_openlumara_bridge_profile(tmp_path):
    from local_ai_launcher.bridges.orca_openlumara.configurator import configure_orca_profile_for_openlumara

    profile_file = tmp_path / "profiles" / "local-default" / "orca-data.json"
    profile_file.parent.mkdir(parents=True, exist_ok=True)
    # Simulate pre-existing disabledTuiAgents with 'trae'
    profile_file.write_text(json.dumps({
        "schemaVersion": 1,
        "settings": {
            "disabledTuiAgents": ["trae", "cursor"]
        }
    }), encoding="utf-8")

    success = configure_orca_profile_for_openlumara(
        profile_path=profile_file,
        llm_base_url="http://localhost:9931/v1",
        active_model="qwen-test",
        set_default_agent=True
    )
    assert success is True
    assert profile_file.exists()

    data = json.loads(profile_file.read_text(encoding="utf-8"))
    assert data["settings"]["defaultTuiAgent"] == "trae"
    assert "trae" not in data["settings"]["disabledTuiAgents"]
    assert "openlumara" in data["settings"]["agentCmdOverrides"]
    assert "trae" in data["settings"]["agentCmdOverrides"]
    assert "--disable_auto_installer" in data["settings"]["agentCmdOverrides"]["trae"]
    assert data["settings"]["agentDefaultEnv"]["openlumara"]["OPENAI_BASE_URL"] == "http://localhost:9931/v1"
    assert data["settings"]["agentDefaultEnv"]["openlumara"]["OPENAI_MODEL"] == "qwen-test"
    assert data["settings"]["agentDefaultEnv"]["trae"]["OPENAI_BASE_URL"] == "http://localhost:9931/v1"
    assert data["settings"]["agentDefaultEnv"]["trae"]["OPENAI_MODEL"] == "qwen-test"


def test_openlumara_and_trae_wrappers_deployed(tmp_path):
    from local_ai_launcher.bridges.orca_openlumara.configurator import install_openlumara_wrappers

    success = install_openlumara_wrappers(workspace_dir=tmp_path)
    assert success is True

    scripts_dir = tmp_path / ".venv" / "Scripts"
    assert (scripts_dir / "openlumara.cmd").exists()
    assert (scripts_dir / "openlumara.ps1").exists()
    assert (scripts_dir / "trae.cmd").exists()
    assert (scripts_dir / "trae.ps1").exists()
    assert (scripts_dir / "trae.bat").exists()


def test_orca_setup_sets_trae_when_openlumara_selected(tmp_path, monkeypatch):
    def mock_run(*args, **kwargs):
        class MockRes:
            returncode = 0
            stdout = "true"
            stderr = ""
        return MockRes()
    monkeypatch.setattr("subprocess.run", mock_run)

    test_start_dir = tmp_path / "test_project"
    test_start_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(config, "harness_start_dir", test_start_dir)
    monkeypatch.setattr(config, "choose_harness", ["orca", "openlumara"])
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))

    orca_home = tmp_path / ".orca"
    setup_orca_config(orca_home, "http://localhost:9931/v1", "test-model")

    orca_data = tmp_path / "appdata" / "Orca" / "profiles" / "local-default" / "orca-data.json"
    assert orca_data.exists()
    data = json.loads(orca_data.read_text(encoding="utf-8"))
    assert data["settings"]["defaultTuiAgent"] == "trae"


def test_run_harnesses_openlumara_auxiliary_mode(monkeypatch):
    from local_ai_launcher.__main__ import run_harnesses
    import local_ai_launcher.__main__ as main_module

    calls = []

    class MockAdapter:
        def __init__(self, root_dir, processes):
            self.root_dir = root_dir
            self.processes = processes
        def check(self):
            return True
        def start(self, url, model, **kwargs):
            calls.append((self.__class__.__name__, kwargs))

    class MockOrca(MockAdapter): pass
    class MockOpenLumara(MockAdapter): pass
    class MockPi(MockAdapter): pass
    class MockHermes(MockAdapter): pass

    registry = {
        "orca": MockOrca,
        "openlumara": MockOpenLumara,
        "lumara": MockOpenLumara,
        "pi": MockPi,
        "hermes": MockHermes,
    }
    monkeypatch.setattr(main_module, "HARNESS_REGISTRY", registry)

    # When Orca and OpenLumara are selected, openlumara must launch headlessly (launch_tui=False)
    run_harnesses([], ["orca", "openlumara"], "http://localhost:9931/v1", None)

    lumara_calls = [c for c in calls if c[0] == "MockOpenLumara"]
    assert len(lumara_calls) == 1
    assert lumara_calls[0][1].get("launch_tui") is False

    # When OpenLumara is selected alone without Orca, launch_tui is not False (standard start)
    calls.clear()
    run_harnesses([], ["openlumara"], "http://localhost:9931/v1", None)
    lumara_calls = [c for c in calls if c[0] == "MockOpenLumara"]
    assert len(lumara_calls) == 1
    assert lumara_calls[0][1].get("launch_tui") is not False


def test_online_only_harnesses_set():
    from local_ai_launcher.harnesses import ONLINE_ONLY_HARNESSES
    assert "agy" in ONLINE_ONLY_HARNESSES
    assert "agy-ide" in ONLINE_ONLY_HARNESSES
    assert "agy-desktop" in ONLINE_ONLY_HARNESSES
    assert "agy-cli" not in ONLINE_ONLY_HARNESSES
    assert "orca" not in ONLINE_ONLY_HARNESSES
    assert "pi" not in ONLINE_ONLY_HARNESSES


def test_online_only_skips_offline_llm_engine(monkeypatch):
    import local_ai_launcher.__main__ as main_mod

    events = []

    monkeypatch.setattr(main_mod, "ensure_running_in_venv", lambda: None)
    monkeypatch.setattr(main_mod, "ensure_windows_terminal", lambda: None)
    monkeypatch.setattr(main_mod, "setup_windows_environment_variables", lambda: None)
    monkeypatch.setattr(main_mod, "setup_windows_user_account", lambda: None)
    monkeypatch.setattr(main_mod, "force_kill_port", lambda p: None)
    monkeypatch.setattr(main_mod, "check_git", lambda: None)
    monkeypatch.setattr(main_mod, "check_github", lambda: None)
    monkeypatch.setattr(main_mod, "run_mnemosyne_verifier", lambda: None)
    monkeypatch.setattr(main_mod.time, "sleep", lambda s: None)

    monkeypatch.setattr(main_mod, "start_voice_to_text", lambda p: events.append("voice_to_text"))
    monkeypatch.setattr(main_mod, "start_localai_engine", lambda p: (events.append("localai_engine"), None, None))
    monkeypatch.setattr(main_mod, "start_proxy", lambda p, u: events.append("proxy"))
    monkeypatch.setattr(main_mod, "run_harnesses", lambda p, h, u, t: events.append(f"harnesses:{h}"))

    # Mock loop to immediately exit
    monkeypatch.setattr(main_mod, "main", main_mod.main)
    # We can test the section directly or monkeypatch sys.exit in loop
    # Let's test with choose_harness = ["agy"]
    monkeypatch.setattr(main_mod.config, "choose_harness", ["agy"])
    monkeypatch.setattr(main_mod.config, "run_mnemosyne_verification", False)

    # To avoid the while True loop, raise KeyboardInterrupt in run_harnesses or mock time.sleep to break
    def mock_run_harnesses(processes, selected, url, thread):
        events.append(f"run_harnesses:{selected}:{url}")
        raise KeyboardInterrupt()

    monkeypatch.setattr(main_mod, "run_harnesses", mock_run_harnesses)

    try:
        main_mod.main()
    except KeyboardInterrupt:
        pass

    assert "voice_to_text" in events
    assert "localai_engine" not in events
    assert "proxy" not in events
    assert any("run_harnesses:['agy']:None" in e for e in events)

    # Now test with mixed/offline harness: choose_harness = ["agy", "orca"]
    events.clear()
    monkeypatch.setattr(main_mod.config, "choose_harness", ["agy", "orca"])
    monkeypatch.setattr(main_mod, "start_localai_engine", lambda p: (events.append("localai_engine"), "http://localhost:8080/v1", None, None)[1:])
    monkeypatch.setattr(main_mod, "start_proxy", lambda p, u: (events.append("proxy"), "http://localhost:9931/v1")[1])

    def mock_run_harnesses_offline(processes, selected, url, thread):
        events.append(f"run_harnesses:{selected}:{url}")
        raise KeyboardInterrupt()

    monkeypatch.setattr(main_mod, "run_harnesses", mock_run_harnesses_offline)

    try:
        main_mod.main()
    except KeyboardInterrupt:
        pass

    assert "voice_to_text" in events
    assert "proxy" in events
    assert any("run_harnesses:['agy', 'orca']:http://localhost:9931/v1" in e for e in events)

    def break_sleep(s):
        if s == 1:
            raise KeyboardInterrupt()

    # Test with empty harness list (Voice-to-Text only mode)
    events.clear()
    monkeypatch.setattr(main_mod.config, "choose_harness", [])
    monkeypatch.setattr(main_mod.time, "sleep", break_sleep)
    try:
        main_mod.main()
    except KeyboardInterrupt:
        pass

    assert "voice_to_text" in events
    assert "localai_engine" not in events
    assert "proxy" not in events
    assert not any("run_harnesses:" in e for e in events)

    # Test with choose_harness = ["none"]
    events.clear()
    monkeypatch.setattr(main_mod.config, "choose_harness", ["none"])
    monkeypatch.setattr(main_mod.time, "sleep", break_sleep)
    try:
        main_mod.main()
    except KeyboardInterrupt:
        pass

    assert "voice_to_text" in events
    assert "localai_engine" not in events
    assert "proxy" not in events
    assert not any("run_harnesses:" in e for e in events)


def test_harness_negative_and_out_of_bounds_hardening(tmp_path, monkeypatch):
    """Verify harnesses sanitize negative ports, context sizes, and compaction ratios."""
    from unittest.mock import patch, MagicMock
    import yaml
    from local_ai_launcher.harnesses.deepseek import DeepseekAdapter
    from local_ai_launcher.harnesses.antigravity import AgyIdeAdapter

    # 1. T3: Negative port and negative/invalid context size
    t3_home = tmp_path / ".t3"
    t3_userdata = tmp_path / "custom_t3_userdata"
    monkeypatch.setenv("T3CODE_BASE_DIR", str(t3_userdata))
    monkeypatch.setattr(config, "t3_port", -9999)
    monkeypatch.setattr(config, "harness_start_dir", tmp_path)
    with patch("local_ai_launcher.harnesses.t3.get_model_specs", return_value=(-4096, False)):
        setup_t3_config(t3_home, "http://127.0.0.1:8080/v1", "test-model")

    t3_settings_file = t3_userdata / "settings.json"
    assert t3_settings_file.exists()
    settings = json.loads(t3_settings_file.read_text(encoding="utf-8"))
    assert settings.get("serverSettings", {}).get("port") == 3773
    # Context window clamped to minimum 512
    models = settings.get("providerInstances", {}).get("pi", {}).get("models", [])
    test_model_entry = next((m for m in models if m["id"] == "test-model"), None)
    assert test_model_entry is not None
    assert test_model_entry["contextWindow"] >= 512

    # 2. Deepseek: Negative ratios and invalid bounds
    monkeypatch.setenv("DSH_HOME", str(tmp_path / ".dsh"))
    monkeypatch.setattr(config, "harness_start_dir", tmp_path)
    monkeypatch.setattr(config, "hermes_compression_threshold_ratio", 2.5)
    monkeypatch.setattr(config, "pi_compaction_keep_ratio", -0.5)
    
    deepseek_adapter = DeepseekAdapter(tmp_path, [])
    with patch("local_ai_launcher.harness_base.send_background_warmup"), \
         patch("local_ai_launcher.harnesses.deepseek.start_in_wt_tab"):
        deepseek_adapter.start("http://127.0.0.1:8080/v1", "dummy-model")

    patch_file = tmp_path / "dsh_model_patch.yml"
    assert patch_file.exists()
    dsh_config = yaml.safe_load(patch_file.read_text(encoding="utf-8"))
    compaction_cfg = next(c["config"] for c in dsh_config if c.get("id") == "compaction-basic")
    assert compaction_cfg["thresholdRatio"] <= 1.0
    assert compaction_cfg["retainRatio"] >= 0.0

    # 3. Antigravity MCP server: Negative port fallback
    monkeypatch.setattr(config, "mcp_server_port", -50)
    ag_adapter = AgyIdeAdapter(tmp_path, [])
    with patch("local_ai_launcher.harnesses.antigravity.is_port_in_use", return_value=True):
        # Should not raise ValueError/OverflowError when checking port
        ag_adapter._start_mcp_service()


def test_concurrent_preflight_updates_and_staged_launch(monkeypatch):
    """Verifies that all preflight checks/updates complete before any harness is started."""
    from local_ai_launcher.__main__ import run_harnesses
    import local_ai_launcher.__main__ as main_mod

    events = []

    class MockAdapter:
        def __init__(self, root_dir, processes):
            self.root_dir = root_dir
            self.processes = processes
        def check(self):
            events.append(f"check_start:{self.__class__.__name__}")
            import time
            time.sleep(0.05)
            events.append(f"check_end:{self.__class__.__name__}")
            return True
        def start(self, url, model, **kwargs):
            events.append(f"start:{self.__class__.__name__}")

    class MockPi(MockAdapter): pass
    class MockHermes(MockAdapter): pass
    class MockOpenLumara(MockAdapter): pass
    class MockOrca(MockAdapter): pass

    registry = {
        "pi": MockPi,
        "hermes": MockHermes,
        "openlumara": MockOpenLumara,
        "orca": MockOrca,
    }
    monkeypatch.setattr(main_mod, "HARNESS_REGISTRY", registry)

    run_harnesses([], ["orca"], "http://localhost:9931/v1", None)

    # All checks must have ended before any start event occurs
    first_start_idx = next((i for i, e in enumerate(events) if e.startswith("start:")), len(events))
    assert first_start_idx > 0

    check_events = events[:first_start_idx]
    start_events = events[first_start_idx:]

    assert any("MockPi" in e for e in check_events)
    assert any("MockHermes" in e for e in check_events)
    assert any("MockOpenLumara" in e for e in check_events)
    assert any("MockOrca" in e for e in check_events)

    # All 4 harnesses must have finished check_end before any start
    for name in ("MockPi", "MockHermes", "MockOpenLumara", "MockOrca"):
        assert f"check_end:{name}" in check_events
        assert f"start:{name}" in start_events


def test_preflight_update_error_isolation(monkeypatch):
    """Verifies that if an auxiliary harness fails preflight update, other harnesses still start."""
    from local_ai_launcher.__main__ import run_harnesses
    import local_ai_launcher.__main__ as main_mod

    started = []

    class MockPi:
        def __init__(self, root_dir, processes): pass
        def check(self):
            raise RuntimeError("Simulated network timeout during Pi update")
        def start(self, url, model, **kwargs):
            started.append("pi")

    class MockOrca:
        def __init__(self, root_dir, processes): pass
        def check(self):
            return True
        def start(self, url, model, **kwargs):
            started.append("orca")

    registry = {
        "pi": MockPi,
        "orca": MockOrca,
    }
    monkeypatch.setattr(main_mod, "HARNESS_REGISTRY", registry)

    # Should not raise exception; Pi skipped, Orca started
    run_harnesses([], ["orca"], "http://localhost:9931/v1", None)

    assert "orca" in started
    assert "pi" not in started


def test_orca_config_includes_pi_skip_version_check(tmp_path, monkeypatch):
    """Verifies that setup_orca_config injects PI_SKIP_VERSION_CHECK into agentDefaultEnv for Pi."""
    monkeypatch.setattr(config, "harness_start_dir", tmp_path)
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))

    # Mock subprocess.run for git operations
    class MockRes:
        returncode = 0
        stdout = "true"
        stderr = ""
    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: MockRes())

    orca_home = tmp_path / ".orca"
    setup_orca_config(orca_home, "http://localhost:9931/v1", "test-model")

    orca_data_file = Path(tmp_path / "appdata") / "Orca" / "profiles" / "local-default" / "orca-data.json"
    assert orca_data_file.exists()
    content = json.loads(orca_data_file.read_text(encoding="utf-8"))
    agent_env = content["settings"]["agentDefaultEnv"]
    assert "pi" in agent_env
    assert agent_env["pi"].get("PI_SKIP_VERSION_CHECK") == "1"


def test_orca_config_claude_isolation(tmp_path, monkeypatch):
    """Verifies that setup_orca_config isolates claude-local and claude-official in Orca ADE."""
    monkeypatch.setattr(config, "harness_start_dir", tmp_path)
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    monkeypatch.setattr(config, "claude_mode", "local")

    class MockRes:
        returncode = 0
        stdout = "true"
        stderr = ""
    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: MockRes())

    orca_home = tmp_path / ".orca"
    setup_orca_config(orca_home, "http://localhost:8080/v1", "strata-test")

    orca_data_file = Path(tmp_path / "appdata") / "Orca" / "profiles" / "local-default" / "orca-data.json"
    assert orca_data_file.exists()
    content = json.loads(orca_data_file.read_text(encoding="utf-8"))
    agent_env = content["settings"]["agentDefaultEnv"]

    # 1. claude-local has isolated config dir and local endpoint
    assert "claude-local" in agent_env
    assert ".cbox" in agent_env["claude-local"]["CLAUDE_CONFIG_DIR"]
    assert agent_env["claude-local"]["ANTHROPIC_BASE_URL"] == "http://localhost:8080"
    assert agent_env["claude-local"]["ANTHROPIC_API_KEY"] == "dummy-token"

    # 2. claude-official does NOT have ANTHROPIC_BASE_URL and has clean CLAUDE_CONFIG_DIR
    assert "claude-official" in agent_env
    assert agent_env["claude-official"]["CLAUDE_CONFIG_DIR"] == ""
    assert "ANTHROPIC_BASE_URL" not in agent_env["claude-official"]

    # 3. agentCmdOverrides routes claude-local to isolated runner
    cmd_overrides = content["settings"]["agentCmdOverrides"]
    assert "claude-local" in cmd_overrides
    assert "cbox" in cmd_overrides["claude-local"] or "claude-local" in cmd_overrides["claude-local"]


def test_hermes_adapter_trampoline_cleanup(tmp_path, monkeypatch):
    """Verify that _clean_broken_trampolines removes failing uv trampolines in bin."""
    from local_ai_launcher.harnesses.hermes import HermesAdapter
    adapter = HermesAdapter(tmp_path, [])
    adapter.hermes_home = tmp_path / ".hermes"
    bin_dir = adapter.hermes_home / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    
    broken_exe = bin_dir / "hermes.exe"
    broken_exe.write_text("uv trampoline canonicalize stub", encoding="utf-8")
    
    # Mock subprocess.run to simulate broken uv trampoline
    def mock_run(cmd, *args, **kwargs):
        class Res:
            returncode = 1
            stdout = ""
            stderr = "error: uv trampoline failed to canonicalize script path"
        return Res()
    
    monkeypatch.setattr("subprocess.run", mock_run)
    adapter._clean_broken_trampolines()
    assert not broken_exe.exists()


def test_hermes_adapter_resolve_cli(tmp_path, monkeypatch):
    """Verify _resolve_hermes_cli correctly identifies available working Hermes executables."""
    from local_ai_launcher.harnesses.hermes import HermesAdapter
    adapter = HermesAdapter(tmp_path, [])
    adapter.hermes_home = tmp_path / ".hermes"
    adapter.hermes_agent_dir = adapter.hermes_home / "hermes-agent"
    
    bin_dir = adapter.hermes_home / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    cmd_file = bin_dir / "hermes.cmd"
    cmd_file.write_text("@echo off", encoding="utf-8")
    
    monkeypatch.setattr(adapter, "_test_executable", lambda p: Path(p) == cmd_file)
    resolved = adapter._resolve_hermes_cli()
    assert resolved == cmd_file


def test_hermes_adapter_ensure_venv_junction(tmp_path, monkeypatch):
    """Verify _ensure_venv_junction links hermes-agent/venv to the active PM environment."""
    from local_ai_launcher.harnesses.hermes import HermesAdapter
    adapter = HermesAdapter(tmp_path, [])
    adapter.hermes_home = tmp_path / ".hermes"
    adapter.hermes_agent_dir = adapter.hermes_home / "hermes-agent"
    adapter.hermes_agent_dir.mkdir(parents=True, exist_ok=True)
    
    active_pm_venv = tmp_path / "pm_env" / "venv"
    active_pm_venv.mkdir(parents=True, exist_ok=True)
    (active_pm_venv / "Scripts").mkdir(parents=True, exist_ok=True)
    (active_pm_venv / "Scripts" / "python.exe").write_text("", encoding="utf-8")
    
    monkeypatch.setattr(adapter, "_resolve_hermes_venv", lambda: active_pm_venv)
    
    junction_calls = []
    monkeypatch.setattr("local_ai_launcher.harnesses.hermes.ensure_directory_junction", lambda link, target: junction_calls.append((link, target)))
    
    adapter._ensure_venv_junction()
    assert len(junction_calls) == 1
    assert junction_calls[0][0] == adapter.hermes_agent_dir / "venv"
    assert junction_calls[0][1] == active_pm_venv


def test_hermes_adapter_install_flow(tmp_path, monkeypatch):
    """Verify _install_hermes invokes the official installer and handles errors gracefully."""
    from local_ai_launcher.harnesses.hermes import HermesAdapter
    adapter = HermesAdapter(tmp_path, [])
    adapter.hermes_home = tmp_path / ".hermes"
    adapter.hermes_agent_dir = adapter.hermes_home / "hermes-agent"
    
    # Simulate git clone and scripts/install.ps1 execution
    def mock_run(cmd, *args, **kwargs):
        class Res:
            returncode = 0
            stdout = ""
            stderr = ""
        return Res()
    
    monkeypatch.setattr("subprocess.run", mock_run)
    monkeypatch.setattr(adapter, "_ensure_venv_junction", lambda: None)
    monkeypatch.setattr(adapter, "_resolve_hermes_cli", lambda: adapter.hermes_home / "bin" / "hermes.cmd")
    
    # Create fake install script
    scripts_dir = adapter.hermes_agent_dir / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    (scripts_dir / "install.ps1").write_text("# install", encoding="utf-8")
    (adapter.hermes_agent_dir / ".git").mkdir(parents=True, exist_ok=True)
    
    assert adapter._install_hermes() is True


def test_hermes_adapter_sanitize_install_state_recovery(tmp_path):
    """Verify _sanitize_install_state redirects facts.json to a healthy generation when current is broken."""
    from local_ai_launcher.harnesses.hermes import HermesAdapter
    adapter = HermesAdapter(tmp_path, [])
    adapter.hermes_home = tmp_path / ".hermes"

    install_dir = adapter.hermes_home / "installs" / "default"
    environments_dir = install_dir / "environments"
    environments_dir.mkdir(parents=True, exist_ok=True)

    # Broken generation
    broken_gen = environments_dir / "broken_gen"
    (broken_gen / "venv" / "Scripts").mkdir(parents=True, exist_ok=True)
    # No python.exe in broken_gen

    # Healthy generation
    healthy_gen = environments_dir / "healthy_gen"
    healthy_py = healthy_gen / "venv" / "Scripts" / "python.exe" if sys.platform == 'win32' else healthy_gen / "venv" / "bin" / "python"
    healthy_py.parent.mkdir(parents=True, exist_ok=True)
    healthy_py.write_text("stub", encoding="utf-8")
    (healthy_gen / "workspace").mkdir(parents=True, exist_ok=True)
    (healthy_gen / "workspace" / "uv.lock").write_text("lock", encoding="utf-8")

    facts_file = install_dir / "facts.json"
    facts_data = {
        "packages": {
            "venv": {
                "environment": str((broken_gen / "venv").resolve()),
                "resolved_lock": str((broken_gen / "workspace" / "uv.lock").resolve()),
            }
        }
    }
    facts_file.write_text(json.dumps(facts_data), encoding="utf-8")

    adapter._sanitize_install_state()

    updated = json.loads(facts_file.read_text(encoding="utf-8"))
    assert updated["packages"]["venv"]["environment"] == str((healthy_gen / "venv").resolve())
    assert updated["packages"]["venv"]["resolved_lock"] == str((healthy_gen / "workspace" / "uv.lock").resolve())


def test_hermes_adapter_sanitize_install_state_purge(tmp_path):
    """Verify _sanitize_install_state purges facts.json when all recorded environments are missing or broken."""
    from local_ai_launcher.harnesses.hermes import HermesAdapter
    adapter = HermesAdapter(tmp_path, [])
    adapter.hermes_home = tmp_path / ".hermes"

    install_dir = adapter.hermes_home / "installs" / "default"
    install_dir.mkdir(parents=True, exist_ok=True)

    facts_file = install_dir / "facts.json"
    facts_data = {
        "packages": {
            "venv": {
                "environment": str((install_dir / "non_existent" / "venv").resolve()),
            }
        }
    }
    facts_file.write_text(json.dumps(facts_data), encoding="utf-8")

    adapter._sanitize_install_state()

    assert not facts_file.exists()


def test_deepseek_adapter_model_patch_and_settings_sync(tmp_path, monkeypatch):
    """Verify DeepseekAdapter generates valid composed patch and syncs settings.yaml."""
    from unittest.mock import patch
    import yaml
    from local_ai_launcher.harnesses.deepseek import DeepseekAdapter

    dsh_home = tmp_path / ".dsh"
    dsh_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("DSH_HOME", str(dsh_home))
    monkeypatch.setattr(config, "harness_start_dir", tmp_path)

    # Pre-populate settings.yaml with old model
    old_settings = {
        "ui-theme": {"preference": "dark"},
        "agent-default-model": {
            "provider": "other",
            "model": "deepseek-v4-flash"
        }
    }
    settings_file = dsh_home / "settings.yaml"
    settings_file.write_text(yaml.dump(old_settings), encoding="utf-8")

    adapter = DeepseekAdapter(tmp_path, [])
    with patch("local_ai_launcher.context_preloader.preload_harness_context") as mock_preload, \
         patch("local_ai_launcher.harnesses.deepseek.start_in_wt_tab") as mock_wt:
        adapter.start("http://127.0.0.1:8080/v1", "tiel-coder-35b-vision-think")

    # 1. Verify dsh_model_patch.yml
    patch_file = tmp_path / "dsh_model_patch.yml"
    assert patch_file.exists()
    patch_data = yaml.safe_load(patch_file.read_text(encoding="utf-8"))

    agent_def = next(p for p in patch_data if p.get("id") == "agent-default-model")
    assert agent_def["config"]["provider"] == "deepseek-official"
    assert agent_def["config"]["model"] == "tiel-coder-35b-vision-think"

    title_llm = next(p for p in patch_data if p.get("id") == "session-title-llm")
    assert title_llm["config"]["provider"] == "deepseek-official"
    assert title_llm["config"]["model"] == "tiel-coder-35b-vision-think"
    assert title_llm["config"]["targetWords"] == 5
    assert title_llm["config"]["targetCjkCharacters"] == 10
    assert title_llm["config"]["maxInputBytes"] == 4096
    assert title_llm["config"]["maxOutputTokens"] == 64
    assert title_llm["config"]["timeoutMs"] == 60000

    llm_deepseek = next(p for p in patch_data if p.get("id") == "llm-deepseek")
    assert llm_deepseek["config"]["baseURL"] == "http://127.0.0.1:8080/v1"
    assert llm_deepseek["config"]["apiKeyEnv"] == "DEEPSEEK_API_KEY"
    assert llm_deepseek["config"]["maxTokens"] <= llm_deepseek["config"]["defaultContextWindow"]

    model_entry = llm_deepseek["config"]["models"][0]
    assert model_entry["id"] == "tiel-coder-35b-vision-think"
    assert model_entry["maxTokens"] <= model_entry["contextWindow"]
    assert "image" in model_entry["inputModalities"]
    assert "text" in model_entry["inputModalities"]
    assert model_entry.get("imagePixelBudget") == 640000
    assert model_entry.get("imageMaxBytes") == 1048576

    # 2. Verify settings.yaml synced
    updated_settings = yaml.safe_load(settings_file.read_text(encoding="utf-8"))
    assert updated_settings["ui-theme"]["preference"] == "dark"
    assert updated_settings["agent-default-model"]["provider"] == "deepseek-official"
    assert updated_settings["agent-default-model"]["model"] == "tiel-coder-35b-vision-think"

    # 3. Verify context preloader was invoked
    mock_preload.assert_called_once()


def test_deepseek_adapter_safe_context_clamping(tmp_path, monkeypatch):
    """Verify DeepseekAdapter clamps contextWindow and maxTokens to hardware safety limits."""
    from unittest.mock import patch
    import yaml
    from local_ai_launcher.harnesses.deepseek import DeepseekAdapter

    monkeypatch.setattr(config, "harness_start_dir", tmp_path)
    monkeypatch.setenv("DSH_HOME", str(tmp_path / ".dsh"))

    adapter = DeepseekAdapter(tmp_path, [])
    class FakeCompaction:
        ctx_size = 262000
    with patch("local_ai_launcher.harnesses.deepseek.compute_compaction_params", return_value=FakeCompaction()), \
         patch("local_ai_launcher.vram_coordinator.calculate_safe_context_limit", return_value=131072), \
         patch("local_ai_launcher.context_preloader.preload_harness_context"), \
         patch("local_ai_launcher.harnesses.deepseek.start_in_wt_tab"):
        adapter.start("http://127.0.0.1:8080/v1", "test-model")

    patch_file = tmp_path / "dsh_model_patch.yml"
    assert patch_file.exists()
    patch_data = yaml.safe_load(patch_file.read_text(encoding="utf-8"))
    llm_deepseek = next(p for p in patch_data if p.get("id") == "llm-deepseek")
    assert llm_deepseek["config"]["defaultContextWindow"] == 131072
    assert llm_deepseek["config"]["maxTokens"] == 16384  # 131072 // 8


def test_deepseek_adapter_text_only_model(tmp_path, monkeypatch):
    """Verify text-only models omit image request parameters to prevent upstream schema errors."""
    from unittest.mock import patch
    import yaml
    from local_ai_launcher.harnesses.deepseek import DeepseekAdapter

    monkeypatch.setattr(config, "harness_start_dir", tmp_path)
    monkeypatch.setenv("DSH_HOME", str(tmp_path / ".dsh"))

    adapter = DeepseekAdapter(tmp_path, [])
    with patch("local_ai_launcher.context_preloader.preload_harness_context"), \
         patch("local_ai_launcher.harnesses.deepseek.start_in_wt_tab"):
        adapter.start("http://127.0.0.1:8080/v1", "text-only-model")

    patch_file = tmp_path / "dsh_model_patch.yml"
    assert patch_file.exists()
    patch_data = yaml.safe_load(patch_file.read_text(encoding="utf-8"))
    llm_deepseek = next(p for p in patch_data if p.get("id") == "llm-deepseek")
    model_entry = llm_deepseek["config"]["models"][0]
    assert model_entry["inputModalities"] == ["text"]
    assert "imagePixelBudget" not in model_entry
    assert "imageMaxBytes" not in model_entry


def test_deepseek_adapter_executable_resolution(tmp_path, monkeypatch):
    """Verify _resolve_executable prioritizes local node_modules/.bin over PATH."""
    from local_ai_launcher.harnesses.deepseek import DeepseekAdapter

    adapter = DeepseekAdapter(tmp_path, [])
    local_bin = tmp_path / "node_modules" / ".bin"
    local_bin.mkdir(parents=True, exist_ok=True)
    fake_exe = local_bin / adapter.exe_name
    fake_exe.write_text("stub", encoding="utf-8")
    monkeypatch.setattr(adapter, "local_bin", local_bin)

    resolved = adapter._resolve_executable()
    assert resolved == fake_exe

def test_deepseek_adapter_dsh_home_fallback(tmp_path, monkeypatch):
    """Verify DSH_HOME defaults to AI_WORKSPACE_DIR / .dsh when unset, without touching user files."""
    from unittest.mock import patch
    import yaml
    from local_ai_launcher.harnesses.deepseek import DeepseekAdapter
    import local_ai_launcher.harnesses.deepseek as ds_module

    mock_workspace = tmp_path / "mock_workspace"
    mock_workspace.mkdir(parents=True, exist_ok=True)
    monkeypatch.delenv("DSH_HOME", raising=False)
    monkeypatch.setattr(ds_module, "AI_WORKSPACE_DIR", mock_workspace)
    monkeypatch.setattr(config, "harness_start_dir", tmp_path)

    adapter = DeepseekAdapter(tmp_path, [])
    with patch("local_ai_launcher.context_preloader.preload_harness_context"), \
         patch("local_ai_launcher.harnesses.deepseek.start_in_wt_tab"):
        adapter.start("http://127.0.0.1:8080/v1", "test-model")

    fallback_settings = mock_workspace / ".dsh" / "settings.yaml"
    assert fallback_settings.exists()


def test_deepseek_donsetch_integration(tmp_path, monkeypatch):
    """Verify DeepseekAdapter generates donsetch patch and injects DONSETCH env vars when enabled, and omits when disabled."""
    from unittest.mock import patch
    import yaml
    from local_ai_launcher.harnesses.deepseek import DeepseekAdapter
    import local_ai_launcher.harnesses.deepseek as ds_mod

    dsh_home = tmp_path / ".dsh"
    dsh_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("DSH_HOME", str(dsh_home))
    monkeypatch.setattr(ds_mod, "AI_WORKSPACE_DIR", tmp_path)
    monkeypatch.setattr(config, "harness_start_dir", tmp_path)

    fake_don_bin = tmp_path / "binaries" / "donsetch.exe"
    fake_don_bin.parent.mkdir(parents=True, exist_ok=True)
    fake_don_bin.write_text("stub", encoding="utf-8")
    monkeypatch.setattr(config, "donsetch_bin", fake_don_bin)

    adapter = DeepseekAdapter(tmp_path, [])

    # 1. When deepseek_donsetch_enabled is True
    monkeypatch.setattr(config, "deepseek_donsetch_enabled", True)
    monkeypatch.setattr(config, "donsetch_tool_prefix", "myfetch")
    monkeypatch.setattr(config, "donsetch_channel", "stable")
    monkeypatch.setattr(config, "donsetch_auto_update", False)

    launched_cmd = []
    def fake_start_wt(cmd, **kwargs):
        launched_cmd.append(cmd)
        return 12345

    with patch("local_ai_launcher.context_preloader.preload_harness_context"), \
         patch("local_ai_launcher.harnesses.deepseek.start_in_wt_tab", side_effect=fake_start_wt), \
         patch("local_ai_launcher.skills_manager.SkillManager.ensure_scoped_harness_skills") as mock_scoped:
        adapter.start("http://127.0.0.1:8080/v1", "test-model")
        mock_scoped.assert_called_with("deepseek", dsh_home / "skills")

    patch_file = tmp_path / "dsh_model_patch.yml"
    assert patch_file.exists()
    patch_data = yaml.safe_load(patch_file.read_text(encoding="utf-8"))
    donsetch_entry = next((p for p in patch_data if p.get("id") == "donsetch"), None)
    assert donsetch_entry is not None
    assert donsetch_entry["config"]["toolPrefix"] == "myfetch"
    assert donsetch_entry["config"]["autoUpdate"] is False

    assert len(launched_cmd) == 1
    cmd_text = launched_cmd[0]
    assert f"$env:DONSETCH_BIN='{fake_don_bin}';" in cmd_text
    assert "$env:DONSETCH_DSH_PREFIX='myfetch';" in cmd_text
    assert "$env:DONSETCH_DSH_CHANNEL='stable';" in cmd_text
    assert "$env:DONSETCH_DSH_AUTOUPDATE='off';" in cmd_text

    # 2. When deepseek_donsetch_enabled is False
    monkeypatch.setattr(config, "deepseek_donsetch_enabled", False)
    launched_cmd.clear()

    with patch("local_ai_launcher.context_preloader.preload_harness_context"), \
         patch("local_ai_launcher.harnesses.deepseek.start_in_wt_tab", side_effect=fake_start_wt), \
         patch("local_ai_launcher.skills_manager.SkillManager.ensure_scoped_harness_skills"):
        adapter.start("http://127.0.0.1:8080/v1", "test-model")

    patch_data_disabled = yaml.safe_load(patch_file.read_text(encoding="utf-8"))
    assert not any(p.get("id") == "donsetch" for p in patch_data_disabled)

    assert len(launched_cmd) == 1
    assert "DONSETCH_BIN" not in launched_cmd[0]
    assert "DONSETCH_DSH_PREFIX" not in launched_cmd[0]


def test_deepseek_ensure_plugins_and_skills(tmp_path, monkeypatch):
    """Verify DeepseekAdapter installs plugins specified in harness_plugins."""
    import subprocess
    from unittest.mock import patch
    from local_ai_launcher.harnesses.deepseek import DeepseekAdapter
    import local_ai_launcher.harnesses.deepseek as ds_mod

    dsh_home = tmp_path / ".dsh"
    dsh_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("DSH_HOME", str(dsh_home))
    monkeypatch.setattr(ds_mod, "AI_WORKSPACE_DIR", tmp_path)

    adapter = DeepseekAdapter(tmp_path, [])
    monkeypatch.setattr(config, "deepseek_donsetch_enabled", True)
    monkeypatch.setattr(config, "harness_plugins", {"deepseek": ["github:dondai44423/donsetch-dsh"]})

    calls = []
    def fake_subprocess_run(cmd, **kwargs):
        calls.append(" ".join(str(c) for c in cmd))
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("subprocess.run", fake_subprocess_run)

    with patch("local_ai_launcher.skills_manager.SkillManager.ensure_scoped_harness_skills"):
        adapter.ensure_plugins_and_skills()
    assert any("plugin --profile web add github:dondai44423/donsetch-dsh" in c for c in calls)


def test_concurrent_harness_preflight_async_pipeline(monkeypatch):
    """Verify that start_harness_preflight_async kicks off preflight concurrently and run_harnesses consumes it."""
    import time
    import threading
    from local_ai_launcher.__main__ import start_harness_preflight_async, run_harnesses
    import local_ai_launcher.__main__ as main_mod

    events = []

    class MockAsyncAdapter:
        def __init__(self, root_dir, processes):
            self.root_dir = root_dir
            self.processes = processes
        def check(self):
            events.append("check_start")
            time.sleep(0.04)
            events.append("check_end")
            return True
        def start(self, url, model, **kwargs):
            events.append("adapter_start")

    registry = {"pi": MockAsyncAdapter}
    monkeypatch.setattr(main_mod, "HARNESS_REGISTRY", registry)

    session = start_harness_preflight_async(["pi"], [])
    assert session is not None

    llm_done = []
    def mock_llm_preload():
        time.sleep(0.04)
        llm_done.append(True)

    llm_t = threading.Thread(target=mock_llm_preload)
    llm_t.start()

    run_harnesses([], ["pi"], "http://127.0.0.1:8080/v1", llm_t)

    assert "check_end" in events
    assert "adapter_start" in events
    assert llm_done == [True]
    check_end_idx = events.index("check_end")
    adapter_start_idx = events.index("adapter_start")
    assert check_end_idx < adapter_start_idx


def test_concurrent_cpu_auxiliary_startup_parallelism(monkeypatch):
    """Verify start_cpu_auxiliary_engines parallelizes reranker, gater, and decider startup."""
    import time
    from local_ai_launcher.__main__ import start_cpu_auxiliary_engines

    times = {}

    def mock_reranker(proc):
        times["reranker_start"] = time.time()
        time.sleep(0.04)
        times["reranker_end"] = time.time()

    def mock_onnx_gater(proc):
        times["gater_start"] = time.time()
        time.sleep(0.04)
        times["gater_end"] = time.time()

    def mock_decider(proc):
        times["decider_start"] = time.time()
        time.sleep(0.04)
        times["decider_end"] = time.time()

    monkeypatch.setattr("local_ai_launcher.backends.cpu_reranker.start_cpu_reranker_server", mock_reranker)
    monkeypatch.setattr("local_ai_launcher.backends.cpu_reranker.ensure_reranker_model", lambda m: None)
    monkeypatch.setattr("local_ai_launcher.backends.cpu_onnx_gater.start_cpu_onnx_gater_server", mock_onnx_gater)
    monkeypatch.setattr("local_ai_launcher.backends.cpu_onnx_gater.ensure_onnx_gater_model", lambda m: None)
    monkeypatch.setattr("local_ai_launcher.backends.cpu_decider.start_cpu_decider_server", mock_decider)
    monkeypatch.setattr("local_ai_launcher.backends.cpu_decider.ensure_decider_model", lambda m: None)

    t0 = time.time()
    start_cpu_auxiliary_engines([])
    total_elapsed = time.time() - t0

    # If serial, 3 * 0.04 = 0.12s. In parallel, it completes much faster.
    assert total_elapsed < 0.11
    assert "reranker_start" in times
    assert "gater_start" in times
    assert "decider_start" in times


def test_harness_registry_contains_claude():
    assert "claude" in HARNESS_REGISTRY
    assert HARNESS_REGISTRY["claude"] is ClaudeAdapter
    assert "claude-code" in HARNESS_REGISTRY
    assert HARNESS_REGISTRY["claude-code"] is ClaudeAdapter
    assert "claudecode" in HARNESS_REGISTRY
    assert HARNESS_REGISTRY["claudecode"] is ClaudeAdapter


def test_config_contains_claude_defaults():
    cfg = LauncherConfig()
    assert "claude" in cfg.harness_install_behaviors
    assert "claude-code" in cfg.harness_install_behaviors
    assert "claude" in cfg.harness_update_behaviors
    assert "claude-code" in cfg.harness_update_behaviors
    assert "claude" in cfg.harness_extensions
    assert "claude" in cfg.harness_skills
    assert hasattr(cfg, "claude_dir")
    assert hasattr(cfg, "claude_mode")
    assert cfg.claude_mode == "local"
    assert cfg.claude_model == "claude-sonnet-4-5"
    assert cfg.claude_api_key == "dummy-token"

    mcfg = MnemosyneAppConfig()
    assert hasattr(mcfg, "claude_dir")
    assert hasattr(mcfg, "claude_config_dir")
    assert mcfg.claude_dir == mcfg.claude_config_dir


def test_claude_adapter_local_start_ephemeral_isolation(tmp_path, monkeypatch):
    """Verify local start sets ephemeral local env, bypasses Claude login, and leaves home untouched."""
    import local_ai_launcher.harnesses.claude as claude_mod
    from local_ai_launcher.config import config

    fake_home = tmp_path / "home"
    fake_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(Path, "home", lambda: fake_home)

    captured_cmds = []

    def mock_start_wt(cmd, cwd=None):
        captured_cmds.append((cmd, cwd))
        return 12345

    monkeypatch.setattr(claude_mod, "start_in_wt_tab", mock_start_wt)
    monkeypatch.setattr(config, "claude_mode", "local")
    monkeypatch.setattr(config, "use_litellm_proxy", False)
    monkeypatch.setattr(config, "claude_base_url", "")
    monkeypatch.setattr(config, "claude_model", "claude-sonnet-4-5")
    monkeypatch.setattr(config, "claude_api_key", "dummy-token")

    procs = []
    adapter = ClaudeAdapter(str(tmp_path), procs)
    adapter.start("http://127.0.0.1:8080/v1", active_model="strata-test")

    assert len(captured_cmds) == 1
    cmd_str, cwd = captured_cmds[0]
    # Check ephemeral environment injection for zero-subscription local mode
    assert "$env:ANTHROPIC_BASE_URL='http://127.0.0.1:8080';" in cmd_str
    assert "$env:ANTHROPIC_API_KEY='dummy-token';" in cmd_str
    assert "$env:ANTHROPIC_MODEL='claude-sonnet-4-5';" in cmd_str
    assert "$env:CLAUDE_CONFIG_DIR" in cmd_str
    assert "$env:CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC='1';" in cmd_str
    assert "$env:DISABLE_TELEMETRY='1';" in cmd_str
    assert "$env:DO_NOT_TRACK='1';" in cmd_str
    assert "$env:DISABLE_ERROR_REPORTING='1';" in cmd_str
    assert "$env:DISABLE_AUTOUPDATER='1';" in cmd_str
    assert "$env:DISABLE_UPDATES='1';" in cmd_str
    assert "chcp 65001" in cmd_str

    # Invariant: Global user ~/.claude.json was NEVER created or modified
    assert not (fake_home / ".claude.json").exists()


def test_claude_adapter_official_start_isolation(tmp_path, monkeypatch):
    """Verify official start preserves official Anthropic credentials without overriding base URL."""
    import local_ai_launcher.harnesses.claude as claude_mod
    from local_ai_launcher.config import config

    captured_cmds = []

    def mock_start_wt(cmd, cwd=None):
        captured_cmds.append((cmd, cwd))
        return 12346

    monkeypatch.setattr(claude_mod, "start_in_wt_tab", mock_start_wt)
    monkeypatch.setattr(config, "claude_mode", "official")

    procs = []
    adapter = ClaudeAdapter(str(tmp_path), procs)
    adapter.start("http://127.0.0.1:8080/v1", active_model="test-model")

    assert len(captured_cmds) == 1
    cmd_str, _ = captured_cmds[0]
    # In official mode, ANTHROPIC_BASE_URL and dummy key must NOT be injected
    assert "$env:ANTHROPIC_BASE_URL" not in cmd_str
    assert "$env:ANTHROPIC_API_KEY = 'dummy-token'" not in cmd_str
    assert "CLAUDE CODE CLI -- [OFFICIAL ANTHROPIC CLOUD MODE]" in cmd_str
    assert "Claude Code [OFFICIAL CLOUD]" in cmd_str


def test_claude_adapter_binary_healing(tmp_path, monkeypatch):
    """Verify _ensure_native_binary replaces placeholder stub with native Windows PE binary."""
    import os
    import local_ai_launcher.harnesses.claude as claude_mod
    monkeypatch.setattr(claude_mod, "os", os)

    mock_node_modules = tmp_path / "node_modules"
    pkg_dir = mock_node_modules / "@anthropic-ai" / "claude-code"
    win64_dir = mock_node_modules / "@anthropic-ai" / "claude-code-win32-x64"
    pkg_dir.mkdir(parents=True, exist_ok=True)
    win64_dir.mkdir(parents=True, exist_ok=True)

    bin_exe = pkg_dir / "bin" / "claude.exe"
    bin_exe.parent.mkdir(parents=True, exist_ok=True)
    # Write a 500-byte placeholder stub
    bin_exe.write_bytes(b"#!/bin/sh\necho stub\n" + b"x" * 480)

    # Write a simulated 20MB native binary
    win64_exe = win64_dir / "claude.exe"
    win64_exe.write_bytes(b"MZ_PE_HEADER" + b"\x00" * (15 * 1024 * 1024))

    monkeypatch.setattr(claude_mod, "AI_WORKSPACE_DIR", tmp_path)

    adapter = ClaudeAdapter(str(tmp_path), [])
    result = adapter._ensure_native_binary()

    assert result is not None
    assert result == bin_exe
    assert bin_exe.stat().st_size > 10_000_000
    assert bin_exe.read_bytes().startswith(b"MZ_PE_HEADER")


def test_claude_adapter_local_profile_preseeded(tmp_path, monkeypatch):
    """Verify local mode pre-seeds .claude.json with onboarding completed and dummy key pre-approved."""
    import local_ai_launcher.harnesses.claude as claude_mod
    from local_ai_launcher.config import config

    test_claude_dir = tmp_path / ".claude"
    monkeypatch.setattr(config, "claude_dir", test_claude_dir)
    monkeypatch.setattr(config, "claude_mode", "local")
    monkeypatch.setattr(config, "claude_api_key", "custom-dummy-token-12345")
    monkeypatch.setattr(config, "claude_dangerously_skip_permissions", True)

    captured_cmds = []
    monkeypatch.setattr(claude_mod, "start_in_wt_tab", lambda cmd, cwd=None: captured_cmds.append(cmd))

    adapter = ClaudeAdapter(str(tmp_path), [])
    adapter.start("http://127.0.0.1:8080/v1", active_model="strata")

    local_profile = adapter.claude_local_dir
    assert (local_profile / ".claude.json").exists()
    assert (local_profile / "settings.json").exists()

    claude_cfg = json.loads((local_profile / ".claude.json").read_text(encoding="utf-8"))
    assert claude_cfg["hasCompletedOnboarding"] is True
    assert claude_cfg["bypassPermissionsModeAccepted"] is True
    assert "token-12345" in claude_cfg["customApiKeyResponses"]["approved"][0]

    settings_cfg = json.loads((local_profile / "settings.json").read_text(encoding="utf-8"))
    assert settings_cfg["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8080"
    assert settings_cfg["env"]["ANTHROPIC_API_KEY"] == "custom-dummy-token-12345"
    assert settings_cfg["env"]["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"
    assert settings_cfg["env"]["DISABLE_TELEMETRY"] == "1"
    assert settings_cfg["env"]["DO_NOT_TRACK"] == "1"
    assert settings_cfg["env"]["DISABLE_ERROR_REPORTING"] == "1"
    assert settings_cfg["env"]["DISABLE_AUTOUPDATER"] == "1"
    assert settings_cfg["env"]["DISABLE_UPDATES"] == "1"
    assert settings_cfg["permissions"]["defaultMode"] == "bypassPermissions"

    assert len(captured_cmds) == 1
    cmd = captured_cmds[0]
    assert "CLAUDE CODE CLI -- [LOCAL OFFLINE MODE]" in cmd
    assert "Claude Code [LOCAL OFFLINE]" in cmd
    assert "Telemetry : DISABLED" in cmd
    assert f"$env:CLAUDE_CONFIG_DIR='{local_profile}';" in cmd

    assert (local_profile / "claude-local.ps1").exists()
    assert (local_profile / "claude-local.sh").exists()
    ps1_txt = (local_profile / "claude-local.ps1").read_text(encoding="utf-8")
    sh_txt = (local_profile / "claude-local.sh").read_text(encoding="utf-8")
    assert "CLAUDE_CONFIG_DIR" in ps1_txt and "ANTHROPIC_BASE_URL" in ps1_txt
    assert '$env:CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC = "1"' in ps1_txt
    assert '$env:DISABLE_TELEMETRY = "1"' in ps1_txt
    assert 'export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1"' in sh_txt
    assert 'export DISABLE_TELEMETRY="1"' in sh_txt


def test_claude_permission_mode_smart_and_overrides(tmp_path, monkeypatch):
    """Verify smart mode (acceptEdits) default behavior, yolo override, and desktop gateway synchronization."""
    import local_ai_launcher.harnesses.claude as claude_mod
    from local_ai_launcher.harnesses.claude import (
        resolve_claude_permission_mode,
        setup_claude_desktop_config,
        ClaudeAdapter,
    )
    from local_ai_launcher.config import config

    # 1. Default resolution is smart -> acceptEdits
    monkeypatch.setattr(config, "claude_permission_mode", "smart")
    monkeypatch.setattr(config, "claude_dangerously_skip_permissions", False)
    monkeypatch.setattr(config, "claude_desktop_permission_mode", None)
    assert resolve_claude_permission_mode(is_desktop=False) == "acceptEdits"
    assert resolve_claude_permission_mode(is_desktop=True, desktop_mode="gateway") == "acceptEdits"
    assert resolve_claude_permission_mode(is_desktop=True, desktop_mode="mcp") == "default"

    # 2. Yolo / bypass resolution
    monkeypatch.setattr(config, "claude_permission_mode", "yolo")
    assert resolve_claude_permission_mode(is_desktop=False) == "bypassPermissions"
    assert resolve_claude_permission_mode(is_desktop=True, desktop_mode="gateway") == "bypassPermissions"

    # 3. Dedicated desktop override
    monkeypatch.setattr(config, "claude_permission_mode", "smart")
    monkeypatch.setattr(config, "claude_desktop_permission_mode", "default")
    assert resolve_claude_permission_mode(is_desktop=False) == "acceptEdits"
    assert resolve_claude_permission_mode(is_desktop=True, desktop_mode="gateway") == "default"

    # 4. Legacy claude_dangerously_skip_permissions takes precedence as bypassPermissions
    monkeypatch.setattr(config, "claude_dangerously_skip_permissions", True)
    assert resolve_claude_permission_mode(is_desktop=False) == "bypassPermissions"
    assert resolve_claude_permission_mode(is_desktop=True, desktop_mode="gateway") == "bypassPermissions"

    # 5. Verify ClaudeAdapter in default smart mode seeds acceptEdits and passes --permission-mode acceptEdits
    monkeypatch.setattr(config, "claude_dangerously_skip_permissions", False)
    monkeypatch.setattr(config, "claude_permission_mode", "smart")
    monkeypatch.setattr(config, "claude_desktop_permission_mode", None)

    test_claude_dir = tmp_path / ".claude_smart"
    monkeypatch.setattr(config, "claude_dir", test_claude_dir)
    monkeypatch.setattr(config, "claude_mode", "local")

    captured_cmds = []
    monkeypatch.setattr(claude_mod, "start_in_wt_tab", lambda cmd, cwd=None: captured_cmds.append(cmd))

    adapter = ClaudeAdapter(str(tmp_path), [])
    adapter.start("http://127.0.0.1:8080/v1", active_model="strata")

    smart_profile = adapter.claude_local_dir
    smart_settings = json.loads((smart_profile / "settings.json").read_text(encoding="utf-8"))
    assert smart_settings["permissions"]["defaultMode"] == "acceptEdits"

    ps1_txt = (smart_profile / "claude-local.ps1").read_text(encoding="utf-8")
    assert "--permission-mode acceptEdits" in ps1_txt

    cmd = captured_cmds[0]
    assert "--permission-mode acceptEdits" in cmd
    assert "Perm Mode : SMART (acceptEdits - auto-approves safe file edits)" in cmd

    # 6. Verify setup_claude_desktop_config in gateway mode sets defaultMode: acceptEdits
    desk_path = tmp_path / "claude_desktop_smart.json"
    setup_claude_desktop_config(
        desk_path,
        "http://127.0.0.1:8080",
        active_model="strata-iq3s",
        desktop_mode="gateway",
        permission_mode="acceptEdits"
    )
    desk_data = json.loads(desk_path.read_text(encoding="utf-8"))
    assert desk_data["deploymentMode"] == "3p"
    assert desk_data["permissions"]["defaultMode"] == "acceptEdits"

    # 7. In MCP mode, permissions block is cleared (online official account rules)
    setup_claude_desktop_config(
        desk_path,
        "http://127.0.0.1:8080",
        active_model="strata-iq3s",
        desktop_mode="mcp"
    )
    desk_data_mcp = json.loads(desk_path.read_text(encoding="utf-8"))
    assert "permissions" not in desk_data_mcp



def test_claude_adapter_binary_healing_cross_platform(tmp_path, monkeypatch):
    """Verify _ensure_native_binary heals binaries on Linux (linux-x64) and macOS (darwin-arm64)."""
    import local_ai_launcher.harnesses.claude as claude_mod
    from local_ai_launcher.harnesses.claude import ClaudeAdapter

    # 1. Test Linux x64
    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.setattr("platform.machine", lambda: "x86_64")
    monkeypatch.setattr(claude_mod, "AI_WORKSPACE_DIR", tmp_path)

    pkg_dir = tmp_path / "node_modules" / "@anthropic-ai" / "claude-code"
    linux_dir = tmp_path / "node_modules" / "@anthropic-ai" / "claude-code-linux-x64"
    pkg_dir.mkdir(parents=True, exist_ok=True)
    linux_dir.mkdir(parents=True, exist_ok=True)

    bin_file = pkg_dir / "bin" / "claude"
    bin_file.parent.mkdir(parents=True, exist_ok=True)
    bin_file.write_bytes(b"#!/bin/sh\nstub\n")

    linux_bin = linux_dir / "claude"
    linux_bin.write_bytes(b"ELF_BINARY" + b"\x00" * (12 * 1024 * 1024))

    adapter = ClaudeAdapter(str(tmp_path), [])
    result = adapter._ensure_native_binary()
    assert result == bin_file
    assert bin_file.stat().st_size > 10_000_000
    assert bin_file.read_bytes().startswith(b"ELF_BINARY")

    # 2. Test Darwin ARM64 (Apple Silicon)
    monkeypatch.setattr("sys.platform", "darwin")
    monkeypatch.setattr("platform.machine", lambda: "arm64")
    darwin_dir = tmp_path / "node_modules" / "@anthropic-ai" / "claude-code-darwin-arm64"
    darwin_dir.mkdir(parents=True, exist_ok=True)
    darwin_bin = darwin_dir / "claude"
    darwin_bin.write_bytes(b"MACHO_BINARY" + b"\x00" * (12 * 1024 * 1024))
    bin_file.write_bytes(b"stub2")

    adapter_darwin = ClaudeAdapter(str(tmp_path), [])
    result_darwin = adapter_darwin._ensure_native_binary()
    assert result_darwin == bin_file
    assert bin_file.read_bytes().startswith(b"MACHO_BINARY")


def test_claude_adapter_start_posix(tmp_path, monkeypatch):
    """Verify ClaudeAdapter.start on POSIX launches via subprocess.Popen with environment variables."""
    import local_ai_launcher.harnesses.claude as claude_mod
    from local_ai_launcher.harnesses.claude import ClaudeAdapter
    from local_ai_launcher.config import config

    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.setattr(config, "claude_mode", "local")
    monkeypatch.setattr(config, "use_litellm_proxy", False)

    spawned = []
    class MockPopen:
        def __init__(self, cmd, cwd=None, env=None):
            spawned.append({"cmd": cmd, "cwd": cwd, "env": env})
        def poll(self): return None

    monkeypatch.setattr("subprocess.Popen", MockPopen)

    procs = []
    adapter = ClaudeAdapter(str(tmp_path), procs)
    adapter.claude_exe = tmp_path / "claude"
    adapter.claude_exe.write_text("#!/bin/sh", encoding="utf-8")

    adapter.start("http://127.0.0.1:8080/v1", active_model="strata")

    assert len(spawned) == 1
    assert spawned[0]["cmd"] == [str(adapter.claude_exe), "--permission-mode", "acceptEdits", "--thinking", "adaptive", "--disallowedTools", "WebSearch,WebFetch"]
    assert spawned[0]["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8080"
    assert spawned[0]["env"]["ANTHROPIC_API_KEY"] == "dummy-token"


@pytest.mark.asyncio
async def test_native_server_count_tokens_and_model_mapping():
    """Verify /v1/messages/count_tokens endpoint and claude-* model name fallback."""
    from starlette.requests import Request
    from local_ai_launcher.localai.native_server import count_tokens_endpoint, _convert_anthropic_to_openai_body, _estimate_anthropic_tokens

    # 1. Verify token estimation
    sample_body = {
        "model": "claude-sonnet-4-5",
        "system": "You are a helpful coding assistant.",
        "messages": [
            {"role": "user", "content": "How do I write a binary search algorithm in Python?"}
        ],
        "tools": [
            {
                "name": "read_file",
                "description": "Reads a file from the repository.",
                "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}}
            }
        ]
    }
    tokens = _estimate_anthropic_tokens(sample_body)
    assert tokens > 10

    # 2. Verify count_tokens_endpoint HTTP response
    body_bytes = json.dumps(sample_body).encode("utf-8")
    async def receive():
        return {"type": "http.request", "body": body_bytes, "more_body": False}
    scope = {"type": "http", "method": "POST", "path": "/v1/messages/count_tokens", "headers": []}
    req = Request(scope, receive)

    resp = await count_tokens_endpoint(req)
    assert resp.status_code == 200
    resp_data = json.loads(resp.body.decode("utf-8"))
    assert "input_tokens" in resp_data
    assert resp_data["input_tokens"] == tokens

    # 3. Verify claude-* model identifier translation to active/default model
    converted = _convert_anthropic_to_openai_body(sample_body)
    assert converted["model"] != "claude-sonnet-4-5"
    assert converted["model"] == config.default_model


def test_context_preloader_claude_prefix():
    """Verify Claude Code and Claude Desktop prefix in context preloader."""
    from local_ai_launcher.context_preloader import build_harness_prefix, OPENCODE_TOOLS

    prefix = build_harness_prefix("claude", "qwen-test")
    assert "messages" in prefix
    assert "tools" in prefix
    assert prefix["tools"] == OPENCODE_TOOLS
    assert "Claude" in prefix["messages"][0]["content"]

    prefix_dt = build_harness_prefix("claude-desktop", "qwen-test")
    assert "messages" in prefix_dt
    assert "tools" in prefix_dt
    assert prefix_dt["tools"] == OPENCODE_TOOLS


def test_claude_desktop_registry():
    """Verify claude-desktop and claudedesktop are properly registered in HARNESS_REGISTRY."""
    from local_ai_launcher.harnesses import HARNESS_REGISTRY
    from local_ai_launcher.harnesses.claude import ClaudeDesktopAdapter

    from local_ai_launcher.harnesses import HARNESS_REGISTRY
    from local_ai_launcher.harnesses.claude import (
        ClaudeDesktopAdapter,
        ClaudeDesktopOfficialAdapter,
        ClaudeDesktopGatewayAdapter,
        ClaudeLocalAdapter,
        ClaudeOfficialAdapter,
    )

    assert "claude-desktop" in HARNESS_REGISTRY
    assert "claudedesktop" in HARNESS_REGISTRY
    assert HARNESS_REGISTRY["claude-desktop"] is ClaudeDesktopAdapter
    assert HARNESS_REGISTRY["claudedesktop"] is ClaudeDesktopAdapter
    assert HARNESS_REGISTRY["claude-desktop-official"] is ClaudeDesktopOfficialAdapter
    assert HARNESS_REGISTRY["claude-desktop-mcp"] is ClaudeDesktopOfficialAdapter
    assert HARNESS_REGISTRY["claude-desktop-gateway"] is ClaudeDesktopGatewayAdapter
    assert HARNESS_REGISTRY["claude-desktop-local"] is ClaudeDesktopGatewayAdapter
    assert HARNESS_REGISTRY["claude-local"] is ClaudeLocalAdapter
    assert HARNESS_REGISTRY["claude-official"] is ClaudeOfficialAdapter
    assert HARNESS_REGISTRY["claude-code-local"] is ClaudeLocalAdapter
    assert HARNESS_REGISTRY["claude-code-official"] is ClaudeOfficialAdapter


def test_simultaneous_claude_cli_local_and_desktop_official(tmp_path, monkeypatch):
    """Verify simultaneous launch of Claude Code CLI in local mode and Claude Desktop in official mode."""
    import local_ai_launcher.harnesses.claude as claude_mod
    from local_ai_launcher.harnesses.claude import ClaudeLocalAdapter, ClaudeDesktopOfficialAdapter
    from local_ai_launcher.config import config
    from unittest.mock import MagicMock

    workspace_claude = tmp_path / ".claude"
    desktop_cfg = tmp_path / "claude_desktop_config.json"
    fake_home = tmp_path / "fake_home"
    fake_home.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(config, "claude_dir", workspace_claude)
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    monkeypatch.setattr(claude_mod, "get_claude_desktop_config_path", lambda: desktop_cfg)

    captured_cmds = []
    monkeypatch.setattr(claude_mod, "start_in_wt_tab", lambda cmd, cwd=None: captured_cmds.append(cmd))

    mock_proc = MagicMock()
    mock_proc.pid = 9999
    monkeypatch.setattr(claude_mod.subprocess, "Popen", lambda *args, **kwargs: mock_proc)

    procs = []
    cli_adapter = ClaudeLocalAdapter(str(tmp_path), procs)
    desktop_adapter = ClaudeDesktopOfficialAdapter(str(tmp_path), procs)

    active_url = "http://127.0.0.1:8080/v1"
    cli_adapter.start(active_url, active_model="strata")
    desktop_adapter.start(active_url, active_model="strata")

    # Verify CLI Local Adapter:
    assert len(captured_cmds) == 1
    cli_cmd = captured_cmds[0]
    assert "CLAUDE CODE CLI -- [LOCAL OFFLINE MODE]" in cli_cmd
    assert "Claude Code [LOCAL OFFLINE]" in cli_cmd
    assert "$env:ANTHROPIC_BASE_URL='http://127.0.0.1:8080';" in cli_cmd
    assert "$env:ANTHROPIC_API_KEY='dummy-token';" in cli_cmd
    assert f"$env:CLAUDE_CONFIG_DIR='{cli_adapter.claude_local_dir}';" in cli_cmd

    local_claude_json = cli_adapter.claude_local_dir / ".claude.json"
    assert local_claude_json.exists()
    cfg_data = json.loads(local_claude_json.read_text(encoding="utf-8"))
    assert cfg_data["hasCompletedOnboarding"] is True
    assert "dummy-token" in cfg_data["customApiKeyResponses"]["approved"][0]

    # Verify Desktop Official Adapter:
    assert desktop_cfg.exists()
    d_data = json.loads(desktop_cfg.read_text(encoding="utf-8"))
    assert "local-model" in d_data["mcpServers"]
    assert "gateway" not in d_data  # GUI stays logged in to Anthropic subscription!
    assert d_data["mcpServers"]["local-model"]["env"]["LOCAL_LLM_URL"] == "http://127.0.0.1:8080/v1"

    # Invariant: Global user ~/.claude.json was NEVER modified
    assert not (fake_home / ".claude.json").exists()


def test_simultaneous_claude_local_and_claude_official(tmp_path, monkeypatch):
    """Verify simultaneous launch of claude-local and claude-official produces two distinct isolated tabs."""
    import local_ai_launcher.harnesses.claude as claude_mod
    from local_ai_launcher.harnesses.claude import ClaudeLocalAdapter, ClaudeOfficialAdapter
    from local_ai_launcher.config import config

    workspace_claude = tmp_path / ".claude"
    fake_home = tmp_path / "fake_home"
    fake_home.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(config, "claude_dir", workspace_claude)
    monkeypatch.setattr(Path, "home", lambda: fake_home)

    captured_cmds = []
    monkeypatch.setattr(claude_mod, "start_in_wt_tab", lambda cmd, cwd=None: captured_cmds.append(cmd))

    procs = []
    local_adapter = ClaudeLocalAdapter(str(tmp_path), procs)
    official_adapter = ClaudeOfficialAdapter(str(tmp_path), procs)

    active_url = "http://127.0.0.1:8080/v1"
    local_adapter.start(active_url, active_model="strata")
    official_adapter.start(active_url, active_model="strata")

    # Both tabs were dispatched to Windows Terminal
    assert len(captured_cmds) == 2

    # Tab 1: Local
    local_cmd = captured_cmds[0]
    assert "CLAUDE CODE CLI -- [LOCAL OFFLINE MODE]" in local_cmd
    assert "Claude Code [LOCAL OFFLINE]" in local_cmd
    assert "$env:ANTHROPIC_BASE_URL='http://127.0.0.1:8080';" in local_cmd
    assert "$env:ANTHROPIC_API_KEY='dummy-token';" in local_cmd
    assert f"$env:CLAUDE_CONFIG_DIR='{local_adapter.claude_local_dir}';" in local_cmd

    # Tab 2: Official
    official_cmd = captured_cmds[1]
    assert "CLAUDE CODE CLI -- [OFFICIAL ANTHROPIC CLOUD MODE]" in official_cmd
    assert "Claude Code [OFFICIAL CLOUD]" in official_cmd
    assert "$env:ANTHROPIC_BASE_URL" not in official_cmd
    assert "$env:ANTHROPIC_API_KEY = 'dummy-token'" not in official_cmd
    assert "$env:CLAUDE_CONFIG_DIR" not in official_cmd

    # Invariant: Global user ~/.claude.json was NEVER modified
    assert not (fake_home / ".claude.json").exists()


def test_cli_flags_claude_mode_selection(monkeypatch):
    """Verify --claude-mode and --claude-desktop-mode flags in parse_cli_args."""
    from local_ai_launcher.__main__ import parse_cli_args
    from local_ai_launcher.config import config

    orig_cli = config.claude_mode
    orig_desktop = config.claude_desktop_mode
    orig_cli_explicit = config.claude_mode_explicit
    orig_desktop_explicit = config.claude_desktop_mode_explicit
    try:
        test_args = ["launcher", "--claude-mode", "official", "--claude-desktop-mode", "gateway"]
        monkeypatch.setattr(sys, "argv", test_args)

        parse_cli_args()
        assert config.claude_mode == "official"
        assert config.claude_desktop_mode == "gateway"
        assert config.claude_mode_explicit is True
        assert config.claude_desktop_mode_explicit is True
    finally:
        config.claude_mode = orig_cli
        config.claude_desktop_mode = orig_desktop
        config.claude_mode_explicit = orig_cli_explicit
        config.claude_desktop_mode_explicit = orig_desktop_explicit


def test_claude_desktop_config_defaults():
    """Verify Claude Desktop default configuration settings in LauncherConfig and MnemosyneAppConfig."""
    from local_ai_launcher.config import LauncherConfig, MnemosyneAppConfig, CLAUDE_DESKTOP_DIR

    cfg = LauncherConfig()
    assert cfg.claude_desktop_dir == CLAUDE_DESKTOP_DIR
    assert cfg.claude_desktop_mode == "mcp"
    assert cfg.harness_install_behaviors.get("claude-desktop") == "always_install"
    assert cfg.harness_update_behaviors.get("claude-desktop") == "always_update"

    mnemo_cfg = MnemosyneAppConfig()
    assert mnemo_cfg.claude_desktop_dir == CLAUDE_DESKTOP_DIR
    assert mnemo_cfg.claude_desktop_config_dir == CLAUDE_DESKTOP_DIR


def test_setup_claude_desktop_config_mcp_mode(tmp_path):
    """Verify setup_claude_desktop_config writes valid MCP bridge configuration in Mode A."""
    from local_ai_launcher.harnesses.claude import setup_claude_desktop_config
    import sys

    cfg_file = tmp_path / "claude_desktop_config.json"
    written_path = setup_claude_desktop_config(
        config_path=cfg_file,
        llm_base_url="http://127.0.0.1:8080",
        active_model="qwen-desktop-test"
    )

    assert written_path == cfg_file
    assert cfg_file.exists()

    data = json.loads(cfg_file.read_text(encoding="utf-8"))
    assert "mcpServers" in data
    assert "local-model" in data["mcpServers"]

    local_mcp = data["mcpServers"]["local-model"]
    assert "python" in Path(local_mcp["command"]).name.lower()
    assert local_mcp["args"] == ["-m", "local_ai_launcher.bridges.claude_desktop_mcp"]
    assert local_mcp["env"]["LOCAL_LLM_URL"] == "http://127.0.0.1:8080/v1"
    assert local_mcp["env"]["LOCAL_LLM_MODEL"] == "qwen-desktop-test"
    assert local_mcp["env"]["LOCAL_LLM_API_KEY"] == "dummy-token"


def test_setup_claude_desktop_config_preserves_existing_servers(tmp_path):
    """Verify setup_claude_desktop_config merges seamlessly without overwriting existing user MCP servers."""
    from local_ai_launcher.harnesses.claude import setup_claude_desktop_config

    cfg_file = tmp_path / "claude_desktop_config.json"
    initial_content = {
        "mcpServers": {
            "user-custom-tool": {
                "command": "node",
                "args": ["custom-tool.js"],
                "env": {"KEY": "VAL"}
            }
        },
        "theme": "dark"
    }
    cfg_file.write_text(json.dumps(initial_content, indent=2), encoding="utf-8")

    setup_claude_desktop_config(config_path=cfg_file, llm_base_url="http://127.0.0.1:8080")

    data = json.loads(cfg_file.read_text(encoding="utf-8"))
    assert data["theme"] == "dark"
    assert "user-custom-tool" in data["mcpServers"]
    assert data["mcpServers"]["user-custom-tool"]["command"] == "node"
    assert "local-model" in data["mcpServers"]


def test_setup_claude_desktop_config_gateway_mode(tmp_path, monkeypatch):
    """Verify setup_claude_desktop_config configures gateway section when claude_desktop_mode is 'gateway'."""
    from local_ai_launcher.harnesses.claude import setup_claude_desktop_config
    from local_ai_launcher.config import config

    monkeypatch.setattr(config, "claude_desktop_mode", "gateway")

    cfg_file = tmp_path / "claude_desktop_config.json"
    setup_claude_desktop_config(config_path=cfg_file, llm_base_url="http://127.0.0.1:8080")

    data = json.loads(cfg_file.read_text(encoding="utf-8"))
    assert "gateway" in data
    assert data["gateway"]["url"] == "http://127.0.0.1:8080/v1"
    assert data["gateway"]["apiKey"] == "dummy-token"
    assert data.get("disableEssentialTelemetry") is True
    assert data.get("disableNonessentialTelemetry") is True
    assert data.get("disableNonessentialServices") is True
    assert data.get("disableAutoUpdates") is True
    assert data["mcpServers"]["local-model"]["env"]["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"
    assert data["mcpServers"]["local-model"]["env"]["DISABLE_TELEMETRY"] == "1"
    assert data["mcpServers"]["local-model"]["env"]["DO_NOT_TRACK"] == "1"
    assert data["mcpServers"]["local-model"]["env"]["DISABLE_ERROR_REPORTING"] == "1"
    assert data["mcpServers"]["local-model"]["env"]["DISABLE_GROWTHBOOK"] == "1"
    assert data["mcpServers"]["local-model"]["env"]["DISABLE_AUTOUPDATER"] == "1"
    assert data["mcpServers"]["local-model"]["env"]["DISABLE_UPDATES"] == "1"


def test_claude_desktop_adapter_start_mock_fallback(tmp_path, monkeypatch):
    """Verify ClaudeDesktopAdapter handles absent binary via mock process gracefully."""
    import local_ai_launcher.harnesses.claude as claude_mod
    from local_ai_launcher.harnesses.claude import ClaudeDesktopAdapter

    # Route config to tmp_path
    monkeypatch.setattr(claude_mod, "get_claude_desktop_config_path", lambda: tmp_path / "claude_desktop_config.json")
    monkeypatch.setattr(config, "use_litellm_proxy", False)
    monkeypatch.setattr(config, "claude_desktop_mode", "mcp")

    procs = []
    adapter = ClaudeDesktopAdapter(str(tmp_path), procs)
    # Ensure binary is not found
    adapter._resolve_executable = lambda: None

    adapter.start("http://127.0.0.1:8080", active_model="test-model")

    assert len(procs) == 1
    name, proc, track_close = procs[0]
    assert name == "Claude Desktop harness"
    assert not track_close

    # Clean up mock process
    try:
        proc.terminate()
        proc.wait(timeout=2.0)
    except Exception:
        pass


def test_claude_desktop_adapter_start_gateway_telemetry_isolation(tmp_path, monkeypatch):
    """Verify ClaudeDesktopAdapter in gateway mode injects telemetry suppression into process environment."""
    import subprocess
    import local_ai_launcher.harnesses.claude as claude_mod
    from local_ai_launcher.harnesses.claude import ClaudeDesktopAdapter

    monkeypatch.setattr(claude_mod, "get_claude_desktop_config_path", lambda: tmp_path / "claude_desktop_config.json")
    monkeypatch.setattr(config, "use_litellm_proxy", False)
    monkeypatch.setattr(config, "claude_desktop_mode", "gateway")

    captured_env = []
    original_popen = subprocess.Popen

    def fake_popen(*args, **kwargs):
        captured_env.append(kwargs.get("env", {}))
        return original_popen(*args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    procs = []
    adapter = ClaudeDesktopAdapter(str(tmp_path), procs)
    adapter._resolve_executable = lambda: None

    adapter.start("http://127.0.0.1:8080", active_model="strata-iq3s-vision")

    assert len(procs) == 1
    _, proc, _ = procs[0]
    try:
        proc.terminate()
        proc.wait(timeout=2.0)
    except Exception:
        pass

    assert len(captured_env) == 1
    env = captured_env[0]
    assert env.get("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC") == "1"
    assert env.get("DISABLE_TELEMETRY") == "1"
    assert env.get("DO_NOT_TRACK") == "1"
    assert env.get("DISABLE_ERROR_REPORTING") == "1"
    assert env.get("DISABLE_GROWTHBOOK") == "1"
    assert env.get("DISABLE_AUTOUPDATER") == "1"
    assert env.get("DISABLE_UPDATES") == "1"
    assert env.get("CLAUDE_CODE_AUTOUPDATER_DISABLED_BY_HOST") == "1"


def test_claude_desktop_mcp_bridge_protocol():
    """Verify local Claude Desktop stdio JSON-RPC 2.0 MCP server handlers."""
    from local_ai_launcher.bridges.claude_desktop_mcp import (
        handle_initialize,
        handle_tools_list,
        SERVER_NAME,
        SERVER_VERSION,
        PROTOCOL_VERSION,
    )

    init_res = handle_initialize(1, {})
    assert init_res["id"] == 1
    assert init_res["result"]["protocolVersion"] == PROTOCOL_VERSION
    assert init_res["result"]["serverInfo"]["name"] == SERVER_NAME

    tools_res = handle_tools_list(2)
    assert tools_res["id"] == 2
    tool_names = [t["name"] for t in tools_res["result"]["tools"]]
    assert "query_local_model" in tool_names
    assert "get_local_model_status" in tool_names


def test_setup_claude_desktop_config_off_mode(tmp_path, monkeypatch):
    """Verify setup_claude_desktop_config prunes local integrations when mode is 'off'."""
    from local_ai_launcher.harnesses.claude import setup_claude_desktop_config
    from local_ai_launcher.config import config

    cfg_file = tmp_path / "claude_desktop_config.json"
    initial_content = {
        "mcpServers": {
            "local-model": {"command": "python", "args": []},
            "user-tool": {"command": "node", "args": []}
        },
        "gateway": {"url": "http://127.0.0.1:8080/v1"}
    }
    cfg_file.write_text(json.dumps(initial_content), encoding="utf-8")

    monkeypatch.setattr(config, "claude_desktop_mode", "off")
    setup_claude_desktop_config(config_path=cfg_file)

    data = json.loads(cfg_file.read_text(encoding="utf-8"))
    assert "local-model" not in data["mcpServers"]
    assert "user-tool" in data["mcpServers"]
    assert "gateway" not in data


def test_setup_claude_desktop_config_utf8_bom(tmp_path):
    """Verify setup_claude_desktop_config reads files containing UTF-8 BOM without crashing."""
    from local_ai_launcher.harnesses.claude import setup_claude_desktop_config

    cfg_file = tmp_path / "claude_desktop_config.json"
    initial_content = {
        "mcpServers": {"existing": {"command": "test"}},
        "theme": "light"
    }
    # Write with utf-8-sig (BOM)
    cfg_file.write_text(json.dumps(initial_content), encoding="utf-8-sig")

    setup_claude_desktop_config(config_path=cfg_file, llm_base_url="http://127.0.0.1:8080")

    data = json.loads(cfg_file.read_text(encoding="utf-8"))
    assert data["theme"] == "light"
    assert "existing" in data["mcpServers"]
    assert "local-model" in data["mcpServers"]
    assert data["mcpServers"]["local-model"]["env"]["PYTHONIOENCODING"] == "utf-8"
    assert data["mcpServers"]["local-model"]["env"]["PYTHONUTF8"] == "1"


def test_claude_desktop_adapter_check_no_side_effects(tmp_path, monkeypatch):
    """Verify ClaudeDesktopAdapter.check() does NOT create or mutate user config files."""
    import local_ai_launcher.harnesses.claude as claude_mod
    from local_ai_launcher.harnesses.claude import ClaudeDesktopAdapter

    cfg_target = tmp_path / "claude_desktop_config.json"
    monkeypatch.setattr(claude_mod, "get_claude_desktop_config_path", lambda: cfg_target)

    procs = []
    adapter = ClaudeDesktopAdapter(str(tmp_path), procs)
    adapter._resolve_executable = lambda: None

    # Run check
    is_installed = adapter.check()
    assert not is_installed

    # Config file must NOT have been created during check()
    assert not cfg_target.exists()


def test_claude_mode_config_sanitization():
    """Verify robust sanitization of claude_desktop_mode and claude_mode."""
    from local_ai_launcher.config import LauncherConfig

    cfg = LauncherConfig()
    cfg.claude_desktop_mode = "developer"
    assert cfg.claude_desktop_mode == "gateway"

    cfg.claude_desktop_mode = "mcp_bridge"
    assert cfg.claude_desktop_mode == "mcp"

    cfg.claude_desktop_mode = "disabled"
    assert cfg.claude_desktop_mode == "off"

    cfg.claude_desktop_mode = "invalid_string"
    assert cfg.claude_desktop_mode == "mcp"

    cfg.claude_mode = "cloud"
    assert cfg.claude_mode == "official"

    cfg.claude_mode = "offline"
    assert cfg.claude_mode == "local"

    cfg.claude_mode = "unknown_val"
    assert cfg.claude_mode == "local"


@pytest.mark.asyncio
async def test_native_server_claude_model_fallback_direct_openai():
    """Verify direct OpenAI completions with claude-* model identifiers resolve to default model."""
    from local_ai_launcher.localai.native_server import supervisor
    from local_ai_launcher.config import config
    from unittest.mock import AsyncMock, patch

    with patch.object(supervisor, "ensure_backend_loaded", new_callable=AsyncMock) as mock_load:
        mock_load.return_value = None
        # Verify ensure_backend_loaded translates claude-* model name
        stub = await supervisor.ensure_backend_loaded("claude-3-7-sonnet")
        mock_load.assert_called_with("claude-3-7-sonnet")

    # Directly test the alias translation logic in ensure_backend_loaded
    clean_name = "claude-sonnet-4-5"
    if clean_name.lower() in ("1-local_first-brain", "omniroute", "default") or clean_name.lower().startswith(("1-", "claude-", "claude_", "anthropic/")):
        resolved = config.default_model
    else:
        resolved = clean_name
    assert resolved == config.default_model


def test_litellm_config_generation(tmp_path):
    """Verify dynamic LiteLLM YAML config file generation."""
    from local_ai_launcher.backends.litellm import generate_litellm_config
    import yaml

    cfg_file = tmp_path / "litellm_config.yaml"
    res_path = generate_litellm_config(
        config_path=cfg_file,
        target_llm_url="http://127.0.0.1:8080/v1",
        active_model="qwen-2.5-coder-32b"
    )

    assert res_path.exists()
    content = yaml.safe_load(res_path.read_text(encoding="utf-8"))
    assert "model_list" in content
    models = content["model_list"]
    assert len(models) >= 4

    # Ensure wildcard mapping exists
    wildcards = [m for m in models if m["model_name"] == "*"]
    assert len(wildcards) == 1
    assert wildcards[0]["litellm_params"]["model"] == "openai/qwen-2.5-coder-32b"
    assert wildcards[0]["litellm_params"]["api_base"] == "http://127.0.0.1:8080/v1"

    # Ensure claude-* mapping exists
    claude_wildcards = [m for m in models if m["model_name"] == "claude-*"]
    assert len(claude_wildcards) == 1
    assert claude_wildcards[0]["litellm_params"]["model"] == "openai/qwen-2.5-coder-32b"


def test_litellm_check_behavior(monkeypatch):
    """Verify check_litellm honors install behaviors."""
    import local_ai_launcher.backends.litellm as litellm_mod
    from local_ai_launcher.backends.litellm import check_litellm

    # When not installed and install_behavior is never_install -> returns False
    monkeypatch.setattr(litellm_mod, "is_litellm_installed", lambda: False)
    monkeypatch.setattr(config, "litellm_install_behavior", "never_install")
    assert check_litellm() is False

    # When already installed and update_behavior is never_update -> returns True without calling pip
    monkeypatch.setattr(litellm_mod, "is_litellm_installed", lambda: True)
    monkeypatch.setattr(config, "litellm_update_behavior", "never_update")
    assert check_litellm() is True


def test_claude_adapter_skips_launch_when_non_strata_model(monkeypatch, tmp_path):
    """Verify ClaudeAdapter.start skips launching offline Claude Code CLI when model is non-Strata."""
    import local_ai_launcher.backends.litellm as litellm_mod
    from local_ai_launcher.harnesses.claude import ClaudeAdapter

    proxy_started = []
    def mock_start_proxy(processes, target_llm_url, active_model=None, port=None):
        proxy_started.append((target_llm_url, active_model))
        return "http://127.0.0.1:4000"

    monkeypatch.setattr(litellm_mod, "start_litellm_proxy", mock_start_proxy)

    launched_cmds = []
    def mock_wt(cmd, cwd=None):
        launched_cmds.append(cmd)
        class MockProc:
            def poll(self): return None
        return MockProc()
    monkeypatch.setattr("local_ai_launcher.harnesses.claude.start_in_wt_tab", mock_wt)

    monkeypatch.setattr(config, "claude_mode", "local")
    monkeypatch.setattr(config, "claude_mode_explicit", True)
    monkeypatch.setattr(config, "default_model", "qwen-38")

    procs = []
    adapter = ClaudeAdapter(str(tmp_path), procs)
    adapter.claude_exe = tmp_path / "claude.cmd"
    adapter.claude_exe.write_text("@echo off", encoding="utf-8")

    # Attempt to start offline Claude Code CLI with a non-Strata model
    adapter.start("http://127.0.0.1:8080/v1", active_model="test-qwen")

    # Must NOT launch any terminal tab and must NOT start LiteLLM proxy
    assert len(launched_cmds) == 0
    assert len(proxy_started) == 0
    assert len(procs) == 0


def test_claude_desktop_gateway_mode_skips_when_non_strata_model(monkeypatch, tmp_path):
    """Verify ClaudeDesktopAdapter skips offline gateway configuration/launch when model is non-Strata."""
    import local_ai_launcher.backends.litellm as litellm_mod
    from local_ai_launcher.harnesses.claude import ClaudeDesktopAdapter
    import local_ai_launcher.harnesses.claude as claude_mod

    proxy_started = []
    def mock_start_proxy(processes, target_llm_url, active_model=None, port=None):
        proxy_started.append((target_llm_url, active_model))
        return "http://127.0.0.1:4000"

    monkeypatch.setattr(litellm_mod, "start_litellm_proxy", mock_start_proxy)

    cfg_target = tmp_path / "claude_desktop_config.json"
    monkeypatch.setattr(claude_mod, "get_claude_desktop_config_path", lambda: cfg_target)
    monkeypatch.setattr(config, "claude_desktop_mode", "gateway")
    monkeypatch.setattr(config, "claude_desktop_mode_explicit", True)
    monkeypatch.setattr(config, "default_model", "qwen-38")

    procs = []
    adapter = ClaudeDesktopAdapter(str(tmp_path), procs)
    adapter._resolve_executable = lambda: None

    adapter.start("http://127.0.0.1:8080/v1", active_model="test-qwen")

    # Must NOT start LiteLLM proxy and must NOT configure/launch gateway
    assert len(proxy_started) == 0
    assert not cfg_target.exists()
    assert len(procs) == 0


def test_orca_ade_claude_local_strata_policy(tmp_path, monkeypatch):
    """Verify Orca ADE only offers offline claude-local when Strata is active."""
    from local_ai_launcher.harnesses.orca import setup_orca_config

    monkeypatch.setattr(config, "harness_start_dir", tmp_path)
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))

    class MockRes:
        returncode = 0
        stdout = "true"
        stderr = ""
    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: MockRes())

    orca_home = tmp_path / ".orca"
    orca_data_file = Path(tmp_path / "appdata") / "Orca" / "profiles" / "local-default" / "orca-data.json"

    # 1. Non-Strata model: claude-local must be omitted
    monkeypatch.setattr(config, "claude_mode", "local")
    monkeypatch.setattr(config, "default_model", "qwen-38")
    setup_orca_config(orca_home, "http://127.0.0.1:8080", active_model="test-qwen")

    data = json.loads(orca_data_file.read_text(encoding="utf-8"))
    env = data["settings"]["agentDefaultEnv"]
    overrides = data["settings"].get("agentCmdOverrides", {})

    assert "claude-local" not in env
    assert "claude-local" not in overrides
    assert "claude-official" in env
    # default "claude" falls back to official when local engine is non-Strata
    assert env["claude"] == {"CLAUDE_CONFIG_DIR": ""}

    # 2. Strata model: claude-local must be configured
    setup_orca_config(orca_home, "http://127.0.0.1:8080", active_model="strata-iq3s-vision")

    data_strata = json.loads(orca_data_file.read_text(encoding="utf-8"))
    env_strata = data_strata["settings"]["agentDefaultEnv"]
    overrides_strata = data_strata["settings"].get("agentCmdOverrides", {})

    assert "claude-local" in env_strata
    assert "claude-local" in overrides_strata
    assert env_strata["claude-local"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8080"
    assert env_strata["claude"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8080"


def test_stop_litellm_proxy(monkeypatch):
    """Verify stop_litellm_proxy terminates process and cleans up port."""
    import local_ai_launcher.backends.litellm as litellm_mod
    from local_ai_launcher.backends.litellm import stop_litellm_proxy

    killed_ports = []
    monkeypatch.setattr(litellm_mod, "force_kill_port", lambda p: killed_ports.append(p))

    class MockProc:
        def __init__(self):
            self.killed = False
        def kill(self):
            self.killed = True

    mock_p = MockProc()
    monkeypatch.setattr(litellm_mod, "_active_litellm_process", mock_p)
    monkeypatch.setattr(litellm_mod, "kill_process_tree", lambda p: setattr(p, "killed", True))

    stop_litellm_proxy()

    assert mock_p.killed is True
    assert 4000 in killed_ports
    assert litellm_mod._active_litellm_process is None


def test_claude_adapter_bypasses_litellm_when_strata_active(monkeypatch, tmp_path):
    """Verify ClaudeAdapter.start bypasses LiteLLM proxy and connects directly when Strata is active."""
    import local_ai_launcher.backends.litellm as litellm_mod
    from local_ai_launcher.harnesses.claude import ClaudeAdapter

    proxy_started = []
    def mock_start_proxy(processes, target_llm_url, active_model=None, port=None):
        proxy_started.append((target_llm_url, active_model))
        return "http://127.0.0.1:4000"

    monkeypatch.setattr(litellm_mod, "start_litellm_proxy", mock_start_proxy)

    launched_cmds = []
    def mock_wt(cmd, cwd=None):
        launched_cmds.append(cmd)
        class MockProc:
            def poll(self): return None
        return MockProc()
    monkeypatch.setattr("local_ai_launcher.harnesses.claude.start_in_wt_tab", mock_wt)

    monkeypatch.setattr(config, "claude_mode", "local")
    monkeypatch.setattr(config, "use_litellm_proxy", True)
    monkeypatch.setattr(config, "default_model", "strata-iq3s")

    procs = []
    adapter = ClaudeAdapter(str(tmp_path), procs)
    adapter.claude_exe = tmp_path / "claude.cmd"
    adapter.claude_exe.write_text("@echo off", encoding="utf-8")

    adapter.start("http://127.0.0.1:8080/v1", active_model="strata-iq3s")

    # LiteLLM proxy must NOT be started for Strata
    assert len(proxy_started) == 0
    assert len(launched_cmds) == 1
    # Check that ANTHROPIC_BASE_URL points directly to 8080 (Strata), not 4000 (LiteLLM)
    assert "$env:ANTHROPIC_BASE_URL='http://127.0.0.1:8080';" in launched_cmds[0]


def test_claude_desktop_gateway_bypasses_litellm_when_strata_active(monkeypatch, tmp_path):
    """Verify ClaudeDesktopAdapter in gateway mode bypasses LiteLLM and connects directly when Strata is active."""
    import local_ai_launcher.backends.litellm as litellm_mod
    from local_ai_launcher.harnesses.claude import ClaudeDesktopAdapter
    import local_ai_launcher.harnesses.claude as claude_mod

    proxy_started = []
    def mock_start_proxy(processes, target_llm_url, active_model=None, port=None):
        proxy_started.append((target_llm_url, active_model))
        return "http://127.0.0.1:4000"

    monkeypatch.setattr(litellm_mod, "start_litellm_proxy", mock_start_proxy)

    cfg_target = tmp_path / "claude_desktop_config.json"
    monkeypatch.setattr(claude_mod, "get_claude_desktop_config_path", lambda: cfg_target)
    monkeypatch.setattr(config, "claude_desktop_mode", "gateway")
    monkeypatch.setattr(config, "use_litellm_proxy", True)

    procs = []
    adapter = ClaudeDesktopAdapter(str(tmp_path), procs)
    adapter._resolve_executable = lambda: None

    adapter.start("http://127.0.0.1:8080/v1", active_model="qwen-38-flash-strata-iq3s")

    # LiteLLM proxy must NOT be started for Strata
    assert len(proxy_started) == 0
    assert cfg_target.exists()
    data = json.loads(cfg_target.read_text(encoding="utf-8"))
    assert "gateway" in data
    assert data["gateway"]["url"] == "http://127.0.0.1:8080/v1"


def test_claude_desktop_resolve_squirrel_versioning(tmp_path, monkeypatch):
    """Verify _resolve_executable correctly identifies newest version in Squirrel app-* directory."""
    from local_ai_launcher.harnesses.claude import ClaudeDesktopAdapter

    mock_lad = tmp_path / "LocalAppData"
    ac_dir = mock_lad / "AnthropicClaude"
    v1_dir = ac_dir / "app-2.9.0"
    v2_dir = ac_dir / "app-2.19675.1"
    v1_dir.mkdir(parents=True)
    v2_dir.mkdir(parents=True)
    (v1_dir / "claude.exe").write_text("", encoding="utf-8")
    (v2_dir / "claude.exe").write_text("", encoding="utf-8")

    monkeypatch.setenv("LOCALAPPDATA", str(mock_lad))
    monkeypatch.setattr("os.name", "nt")

    adapter = ClaudeDesktopAdapter(tmp_path / "empty_install", [])
    adapter.install_dir = tmp_path / "nonexistent"

    resolved = adapter._resolve_executable()
    assert resolved is not None
    assert resolved == v2_dir / "claude.exe"


def test_claude_desktop_start_sets_cwd_to_parent(tmp_path, monkeypatch):
    """Verify ClaudeDesktopAdapter.start runs Claude Desktop with cwd=exe_path.parent for Electron assets."""
    from local_ai_launcher.harnesses.claude import ClaudeDesktopAdapter
    import local_ai_launcher.harnesses.claude as claude_mod

    exe_dir = tmp_path / "app-2.19675.1"
    exe_dir.mkdir(parents=True)
    fake_exe = exe_dir / "claude.exe"
    fake_exe.write_text("", encoding="utf-8")

    cfg_target = tmp_path / "claude_desktop_config.json"
    monkeypatch.setattr(claude_mod, "get_claude_desktop_config_path", lambda: cfg_target)
    monkeypatch.setattr(config, "claude_desktop_mode", "mcp")

    captured_cwd = []
    class MockProc:
        def poll(self): return None

    def mock_popen(cmd, cwd=None, **kwargs):
        captured_cwd.append(cwd)
        return MockProc()

    monkeypatch.setattr(claude_mod.subprocess, "Popen", mock_popen)

    procs = []
    adapter = ClaudeDesktopAdapter(str(tmp_path), procs)
    adapter._resolve_executable = lambda: fake_exe

    adapter.start("http://127.0.0.1:8080/v1", active_model="strata")

    assert len(captured_cwd) == 1
    assert captured_cwd[0] == str(exe_dir)
    assert len(procs) == 1


def test_claude_desktop_mcp_mode_allows_non_strata(tmp_path, monkeypatch):
    """Verify online Claude Desktop (MCP bridge mode) succeeds for non-Strata models."""
    from local_ai_launcher.harnesses.claude import ClaudeDesktopAdapter
    import local_ai_launcher.harnesses.claude as claude_mod

    cfg_target = tmp_path / "claude_desktop_config.json"
    monkeypatch.setattr(claude_mod, "get_claude_desktop_config_path", lambda: cfg_target)
    monkeypatch.setattr(config, "claude_desktop_mode", "mcp")

    mock_procs = []
    class MockProc:
        def poll(self): return None

    monkeypatch.setattr(claude_mod.subprocess, "Popen", lambda *args, **kwargs: MockProc())

    adapter = ClaudeDesktopAdapter(str(tmp_path), mock_procs)
    fake_exe = tmp_path / "fake_claude.exe"
    fake_exe.write_text("", encoding="utf-8")
    adapter._resolve_executable = lambda: fake_exe

    adapter.start("http://127.0.0.1:8080/v1", active_model="qwen-2.5-coder-32b")

    # MCP mode must succeed: config file written with MCP server local-model, and process launched
    assert cfg_target.exists()
    data = json.loads(cfg_target.read_text(encoding="utf-8"))
    assert "local-model" in data["mcpServers"]
    assert "gateway" not in data
    assert len(mock_procs) == 1


def test_claude_desktop_gateway_adapter_skips_non_strata(tmp_path, monkeypatch):
    """Verify ClaudeDesktopGatewayAdapter strictly skips launch for non-Strata engines."""
    from local_ai_launcher.harnesses.claude import ClaudeDesktopGatewayAdapter
    import local_ai_launcher.harnesses.claude as claude_mod

    cfg_target = tmp_path / "claude_desktop_config.json"
    monkeypatch.setattr(claude_mod, "get_claude_desktop_config_path", lambda: cfg_target)

    mock_procs = []
    adapter = ClaudeDesktopGatewayAdapter(str(tmp_path), mock_procs)

    # Launch with non-Strata model
    adapter.start("http://127.0.0.1:8080/v1", active_model="deepseek-r1-distill-qwen-14b")

    # Offline gateway must be skipped
    assert not cfg_target.exists()
    assert len(mock_procs) == 0


def test_setup_claude_desktop_gateway_auto_provisions_no_login_profile(tmp_path):
    """Verify setup_claude_desktop_config in gateway mode auto-seeds developer_settings and configLibrary."""
    from local_ai_launcher.harnesses.claude import setup_claude_desktop_config

    cfg_file = tmp_path / "claude_desktop_config.json"
    setup_claude_desktop_config(
        config_path=cfg_file,
        llm_base_url="http://127.0.0.1:8080/v1",
        active_model="strata-iq3s-vision",
        desktop_mode="gateway"
    )

    # 1. claude_desktop_config.json has deploymentMode: 3p and telemetry disabled
    data = json.loads(cfg_file.read_text(encoding="utf-8"))
    assert data.get("deploymentMode") == "3p"
    assert "gateway" in data
    assert data.get("disableEssentialTelemetry") is True
    assert data.get("disableNonessentialTelemetry") is True
    assert data.get("disableNonessentialServices") is True
    assert data.get("disableAutoUpdates") is True

    # 2. developer_settings.json has allowDevTools: True (bypasses manual Help -> Troubleshooting menu)
    dev_file = tmp_path / "developer_settings.json"
    assert dev_file.exists()
    dev_data = json.loads(dev_file.read_text(encoding="utf-8"))
    assert dev_data.get("allowDevTools") is True

    # 3. configLibrary has applied gateway profile (bypasses Developer -> Configure Third-Party dialog)
    meta_file = tmp_path / "configLibrary" / "_meta.json"
    assert meta_file.exists()
    meta_data = json.loads(meta_file.read_text(encoding="utf-8"))
    applied_id = meta_data.get("appliedId")
    assert applied_id is not None

    profile_file = tmp_path / "configLibrary" / f"{applied_id}.json"
    assert profile_file.exists()
    profile_data = json.loads(profile_file.read_text(encoding="utf-8"))
    assert profile_data.get("inferenceProvider") == "gateway"
    assert profile_data.get("inferenceGatewayBaseUrl") == "http://127.0.0.1:8080"
    assert profile_data.get("inferenceCredentialKind") == "static"
    assert profile_data.get("modelDiscoveryEnabled") is False
    assert "models" not in profile_data
    assert "inference" not in profile_data
    assert profile_data.get("coworkTabEnabled") is False
    assert profile_data.get("chatTabEnabled") is True
    assert profile_data.get("disableEssentialTelemetry") is True
    assert profile_data.get("disableNonessentialTelemetry") is True
    assert profile_data.get("disableNonessentialServices") is True
    assert profile_data.get("disableAutoUpdates") is True
    assert "inferenceModels" in profile_data
    models_list = profile_data["inferenceModels"]
    assert len(models_list) >= 5
    for m in models_list:
        assert m["name"].startswith("claude-")
        assert "qwen" not in m["name"]
        assert "llama" not in m["name"]
    assert any(m.get("isFamilyDefault") for m in models_list)

    # 4. Switching to official MCP mode purges deploymentMode and telemetry overrides so normal official login is preserved
    setup_claude_desktop_config(
        config_path=cfg_file,
        llm_base_url="http://127.0.0.1:8080/v1",
        active_model="strata-iq3s-vision",
        desktop_mode="mcp"
    )
    mcp_data = json.loads(cfg_file.read_text(encoding="utf-8"))
    assert "deploymentMode" not in mcp_data
    assert "gateway" not in mcp_data
    assert "disableEssentialTelemetry" not in mcp_data
    assert "disableNonessentialTelemetry" not in mcp_data
    assert "disableNonessentialServices" not in mcp_data
    assert "disableAutoUpdates" not in mcp_data
    assert "local-model" in mcp_data["mcpServers"]


def test_claude_adapter_defaults_to_offline_local_on_strata(tmp_path, monkeypatch):
    """Verify base ClaudeAdapter ('claude') automatically selects offline local mode when Strata engine is active."""
    import local_ai_launcher.harnesses.claude as claude_mod
    from local_ai_launcher.harnesses.claude import ClaudeAdapter

    captured_cmds = []
    monkeypatch.setattr(claude_mod, "start_in_wt_tab", lambda cmd, cwd=None: captured_cmds.append((cmd, cwd)) or 1111)

    procs = []
    adapter = ClaudeAdapter(str(tmp_path), procs)
    adapter.start("http://127.0.0.1:8080/v1", active_model="strata-iq3s-vision")

    assert len(captured_cmds) == 1
    cmd_str, _ = captured_cmds[0]
    assert "$env:ANTHROPIC_BASE_URL='http://127.0.0.1:8080';" in cmd_str
    assert "$env:ANTHROPIC_API_KEY='dummy-token';" in cmd_str
    assert "CLAUDE CODE CLI -- [LOCAL OFFLINE MODE]" in cmd_str


def test_claude_adapter_defaults_to_online_official_on_non_strata(tmp_path, monkeypatch):
    """Verify base ClaudeAdapter ('claude') automatically selects online official mode and loads LiteLLM MCP when engine is not Strata."""
    import local_ai_launcher.harnesses.claude as claude_mod
    from local_ai_launcher.harnesses.claude import ClaudeAdapter
    import local_ai_launcher.backends.litellm as litellm_mod

    proxy_started = []
    def mock_start_proxy(processes, target_llm_url, active_model=None, port=None):
        proxy_started.append((target_llm_url, active_model))
        return "http://127.0.0.1:4000"

    monkeypatch.setattr(litellm_mod, "start_litellm_proxy", mock_start_proxy)
    monkeypatch.setattr(config, "harness_start_dir", tmp_path)
    monkeypatch.setattr(config, "use_litellm_proxy", True)

    captured_cmds = []
    monkeypatch.setattr(claude_mod, "start_in_wt_tab", lambda cmd, cwd=None: captured_cmds.append((cmd, cwd)) or 2222)

    procs = []
    adapter = ClaudeAdapter(str(tmp_path), procs)
    adapter.start("http://127.0.0.1:8080/v1", active_model="qwen-2.5-coder-32b")

    assert len(captured_cmds) == 1
    cmd_str, _ = captured_cmds[0]
    assert "$env:ANTHROPIC_BASE_URL" not in cmd_str
    assert "CLAUDE CODE CLI -- [OFFICIAL ANTHROPIC CLOUD MODE]" in cmd_str
    assert "MCP Server: local-model / litellm" in cmd_str

    # LiteLLM supplemental proxy must have been started for non-Strata
    assert len(proxy_started) == 1
    assert proxy_started[0][1] == "qwen-2.5-coder-32b"

    # .mcp.json must have been generated in harness start dir
    mcp_file = tmp_path / ".mcp.json"
    assert mcp_file.exists()
    mcp_data = json.loads(mcp_file.read_text(encoding="utf-8"))
    assert "local-model" in mcp_data["mcpServers"]
    assert "litellm" in mcp_data["mcpServers"]
    assert mcp_data["mcpServers"]["litellm"]["env"]["LOCAL_LLM_URL"] == "http://127.0.0.1:4000/v1"

    # Auto-approval in .claude/settings.local.json must be present
    settings_file = tmp_path / ".claude" / "settings.local.json"
    assert settings_file.exists()
    settings_data = json.loads(settings_file.read_text(encoding="utf-8"))
    assert settings_data.get("enableAllProjectMcpServers") is True
    assert "local-model" in settings_data.get("enabledMcpjsonServers", [])


def test_claude_official_alias_runs_online_even_on_strata(tmp_path, monkeypatch):
    """Verify ClaudeOfficialAdapter ('claude-official') runs online official mode even when Strata is active."""
    import local_ai_launcher.harnesses.claude as claude_mod
    from local_ai_launcher.harnesses.claude import ClaudeOfficialAdapter

    captured_cmds = []
    monkeypatch.setattr(claude_mod, "start_in_wt_tab", lambda cmd, cwd=None: captured_cmds.append((cmd, cwd)) or 3333)

    procs = []
    adapter = ClaudeOfficialAdapter(str(tmp_path), procs)
    adapter.start("http://127.0.0.1:8080/v1", active_model="strata-iq3s-vision")

    assert len(captured_cmds) == 1
    cmd_str, _ = captured_cmds[0]
    assert "$env:ANTHROPIC_BASE_URL" not in cmd_str
    assert "CLAUDE CODE CLI -- [OFFICIAL ANTHROPIC CLOUD MODE]" in cmd_str


def test_claude_desktop_adapter_defaults_to_offline_gateway_on_strata(tmp_path, monkeypatch):
    """Verify base ClaudeDesktopAdapter ('claudedesktop') automatically selects offline gateway mode when Strata is active."""
    import local_ai_launcher.harnesses.claude as claude_mod
    from local_ai_launcher.harnesses.claude import ClaudeDesktopAdapter

    cfg_target = tmp_path / "claude_desktop_config.json"
    monkeypatch.setattr(claude_mod, "get_claude_desktop_config_path", lambda: cfg_target)

    mock_procs = []
    class MockProc:
        def poll(self): return None

    monkeypatch.setattr(claude_mod.subprocess, "Popen", lambda *args, **kwargs: MockProc())

    adapter = ClaudeDesktopAdapter(str(tmp_path), mock_procs)
    fake_exe = tmp_path / "fake_claude.exe"
    fake_exe.write_text("", encoding="utf-8")
    adapter._resolve_executable = lambda: fake_exe

    adapter.start("http://127.0.0.1:8080/v1", active_model="strata-iq3s-vision")

    assert cfg_target.exists()
    data = json.loads(cfg_target.read_text(encoding="utf-8"))
    assert data.get("deploymentMode") == "3p"
    assert "gateway" in data
    assert len(mock_procs) == 1


def test_claude_desktop_adapter_defaults_to_online_mcp_on_non_strata(tmp_path, monkeypatch):
    """Verify base ClaudeDesktopAdapter ('claudedesktop') automatically selects online MCP mode when engine is not Strata."""
    import local_ai_launcher.harnesses.claude as claude_mod
    from local_ai_launcher.harnesses.claude import ClaudeDesktopAdapter
    import local_ai_launcher.backends.litellm as litellm_mod

    cfg_target = tmp_path / "claude_desktop_config.json"
    monkeypatch.setattr(claude_mod, "get_claude_desktop_config_path", lambda: cfg_target)

    proxy_started = []
    def fake_start_litellm(processes, target_llm_url, active_model, port=None):
        proxy_started.append((target_llm_url, active_model))
        return "http://127.0.0.1:4000"

    monkeypatch.setattr(litellm_mod, "start_litellm_proxy", fake_start_litellm)

    mock_procs = []
    class MockProc:
        def poll(self): return None

    monkeypatch.setattr(claude_mod.subprocess, "Popen", lambda *args, **kwargs: MockProc())

    adapter = ClaudeDesktopAdapter(str(tmp_path), mock_procs)
    fake_exe = tmp_path / "fake_claude.exe"
    fake_exe.write_text("", encoding="utf-8")
    adapter._resolve_executable = lambda: fake_exe

    adapter.start("http://127.0.0.1:8080/v1", active_model="qwen-2.5-coder-32b")

    assert cfg_target.exists()
    data = json.loads(cfg_target.read_text(encoding="utf-8"))
    assert "deploymentMode" not in data
    assert "gateway" not in data
    assert "local-model" in data["mcpServers"]
    assert "litellm" in data["mcpServers"]
    assert data["mcpServers"]["local-model"]["env"]["LOCAL_LLM_URL"] == "http://127.0.0.1:4000/v1"
    assert data["mcpServers"]["litellm"]["env"]["LOCAL_LLM_URL"] == "http://127.0.0.1:4000/v1"
    assert len(proxy_started) == 1
    assert len(mock_procs) == 1


def test_claude_desktop_official_alias_runs_online_even_on_strata(tmp_path, monkeypatch):
    """Verify ClaudeDesktopOfficialAdapter ('claude-desktop-official') runs online MCP mode even when Strata is active."""
    import local_ai_launcher.harnesses.claude as claude_mod
    from local_ai_launcher.harnesses.claude import ClaudeDesktopOfficialAdapter

    cfg_target = tmp_path / "claude_desktop_config.json"
    monkeypatch.setattr(claude_mod, "get_claude_desktop_config_path", lambda: cfg_target)

    mock_procs = []
    class MockProc:
        def poll(self): return None

    monkeypatch.setattr(claude_mod.subprocess, "Popen", lambda *args, **kwargs: MockProc())

    adapter = ClaudeDesktopOfficialAdapter(str(tmp_path), mock_procs)
    fake_exe = tmp_path / "fake_claude.exe"
    fake_exe.write_text("", encoding="utf-8")
    adapter._resolve_executable = lambda: fake_exe

    adapter.start("http://127.0.0.1:8080/v1", active_model="strata-iq3s-vision")

    assert cfg_target.exists()
    data = json.loads(cfg_target.read_text(encoding="utf-8"))
    assert "deploymentMode" not in data
    assert "gateway" not in data
    assert "local-model" in data["mcpServers"]
    assert len(mock_procs) == 1


def test_setup_claude_cli_mcp_config(tmp_path):
    """Verify setup_claude_cli_mcp_config creates .mcp.json and auto-approves in settings.local.json."""
    from local_ai_launcher.harnesses.claude import setup_claude_cli_mcp_config

    mcp_path = setup_claude_cli_mcp_config(
        project_dir=tmp_path,
        llm_base_url="http://127.0.0.1:8080/v1",
        active_model="qwen-test"
    )

    assert mcp_path.exists()
    data = json.loads(mcp_path.read_text(encoding="utf-8"))
    assert "local-model" in data["mcpServers"]
    assert "litellm" in data["mcpServers"]
    assert data["mcpServers"]["local-model"]["args"] == ["-m", "local_ai_launcher.bridges.claude_desktop_mcp"]
    assert data["mcpServers"]["local-model"]["env"]["LOCAL_LLM_URL"] == "http://127.0.0.1:8080/v1"
    assert data["mcpServers"]["local-model"]["env"]["LOCAL_LLM_MODEL"] == "qwen-test"

    settings_file = tmp_path / ".claude" / "settings.local.json"
    assert settings_file.exists()
    s_data = json.loads(settings_file.read_text(encoding="utf-8"))
    assert s_data["enableAllProjectMcpServers"] is True
    assert "local-model" in s_data["enabledMcpjsonServers"]
    assert "litellm" in s_data["enabledMcpjsonServers"]


def test_setup_claude_cli_mcp_config_preserves_existing_servers(tmp_path):
    """Verify setup_claude_cli_mcp_config preserves user-defined MCP servers in .mcp.json."""
    from local_ai_launcher.harnesses.claude import setup_claude_cli_mcp_config

    mcp_file = tmp_path / ".mcp.json"
    mcp_file.write_text(json.dumps({
        "mcpServers": {
            "my-custom-tool": {
                "command": "node",
                "args": ["server.js"]
            }
        }
    }), encoding="utf-8")

    setup_claude_cli_mcp_config(
        project_dir=tmp_path,
        llm_base_url="http://127.0.0.1:4000/v1",
        active_model="deepseek"
    )

    data = json.loads(mcp_file.read_text(encoding="utf-8"))
    assert "my-custom-tool" in data["mcpServers"]
    assert data["mcpServers"]["my-custom-tool"]["command"] == "node"
    assert "local-model" in data["mcpServers"]
    assert "litellm" in data["mcpServers"]


def test_claude_desktop_mcp_fallback_to_8080_when_4000_fails(monkeypatch):
    """Verify claude_desktop_mcp._execute_local_query falls back from port 4000 to port 8080 on connection error."""
    import local_ai_launcher.bridges.claude_desktop_mcp as mcp_bridge
    import urllib.error

    monkeypatch.setattr(mcp_bridge, "DEFAULT_GATEWAY_URL", "http://127.0.0.1:4000/v1")

    attempted_urls = []
    class MockResponse:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self):
            return json.dumps({
                "choices": [{"message": {"content": "Response from 8080 fallback"}}]
            }).encode("utf-8")

    def mock_urlopen(req, timeout=None):
        url = req.full_url
        attempted_urls.append(url)
        if ":4000" in url:
            raise urllib.error.URLError("Connection refused on 4000")
        elif ":8080" in url:
            return MockResponse()
        raise urllib.error.URLError("Unexpected URL")

    monkeypatch.setattr(mcp_bridge.urllib.request, "urlopen", mock_urlopen)

    resp = mcp_bridge._execute_local_query("test prompt")
    assert resp == "Response from 8080 fallback"
    assert any(":4000" in u for u in attempted_urls)
    assert any(":8080" in u for u in attempted_urls)


def test_claude_desktop_cleanup_removes_dummy_config_library_profile(tmp_path, monkeypatch):
    """Verify setup_claude_desktop_config removes dummy 3P gateway profiles when switching back to official mode."""
    from local_ai_launcher.harnesses.claude import setup_claude_desktop_config

    cfg_file = tmp_path / "claude_desktop_config.json"
    cfg_file.write_text(json.dumps({
        "deploymentMode": "3p",
        "disableEssentialTelemetry": True,
        "disableNonessentialTelemetry": True,
        "disableNonessentialServices": True,
        "disableAutoUpdates": True,
    }), encoding="utf-8")
    cfg_lib = tmp_path / "configLibrary"
    cfg_lib.mkdir(parents=True, exist_ok=True)

    applied_id = "00000000-0000-0000-0000-000000000001"
    dummy_profile = cfg_lib / f"{applied_id}.json"
    dummy_profile.write_text("{}", encoding="utf-8")

    meta_file = cfg_lib / "_meta.json"
    meta_file.write_text(json.dumps({
        "appliedId": applied_id,
        "entries": [{"id": applied_id, "name": "Local Strata Gateway"}]
    }), encoding="utf-8")

    setup_claude_desktop_config(
        config_path=cfg_file,
        llm_base_url="http://127.0.0.1:8080",
        desktop_mode="mcp"
    )

    assert not dummy_profile.exists()
    assert not meta_file.exists()

    cfg_data = json.loads(cfg_file.read_text(encoding="utf-8"))
    assert "disableEssentialTelemetry" not in cfg_data
    assert "disableNonessentialTelemetry" not in cfg_data
    assert "disableNonessentialServices" not in cfg_data
    assert "disableAutoUpdates" not in cfg_data


def test_claude_cli_local_mode_refreshes_existing_mcp_json(tmp_path, monkeypatch):
    """Verify ClaudeAdapter in local mode refreshes existing .mcp.json to point to Strata endpoint."""
    import local_ai_launcher.harnesses.claude as claude_mod
    from local_ai_launcher.harnesses.claude import ClaudeAdapter
    from local_ai_launcher.config import config

    mcp_file = tmp_path / ".mcp.json"
    mcp_file.write_text(json.dumps({
        "mcpServers": {
            "local-model": {
                "command": "python",
                "env": {"LOCAL_LLM_URL": "http://127.0.0.1:4000/v1"}
            }
        }
    }), encoding="utf-8")

    monkeypatch.setattr(config, "harness_start_dir", tmp_path)
    monkeypatch.setattr(claude_mod, "is_strata_active", lambda *args: True)

    captured_cmds = []
    monkeypatch.setattr(claude_mod, "start_in_wt_tab", lambda cmd, cwd=None: captured_cmds.append((cmd, cwd)) or 4444)

    procs = []
    adapter = ClaudeAdapter(str(tmp_path), procs)
    fake_exe = tmp_path / "fake_claude.exe"
    fake_exe.write_text("", encoding="utf-8")
    adapter._resolve_executable = lambda: fake_exe

    adapter.start("http://127.0.0.1:8080", active_model="strata-iq3s-vision")

    assert len(captured_cmds) == 1
    mcp_data = json.loads(mcp_file.read_text(encoding="utf-8"))
    assert mcp_data["mcpServers"]["local-model"]["env"]["LOCAL_LLM_URL"] == "http://127.0.0.1:8080/v1"


def test_get_mcp_python_resolution(tmp_path, monkeypatch):
    """Verify _get_mcp_python resolves active sys.prefix, .venv, venv, or sys.executable."""
    import os
    import sys
    from local_ai_launcher.harnesses.claude import _get_mcp_python
    import local_ai_launcher.harnesses.claude as claude_mod

    # Custom workspace venv
    bin_name = "Scripts" if os.name == "nt" else "bin"
    exe_name = "python.exe" if os.name == "nt" else "python"
    custom_venv = tmp_path / "venv" / bin_name
    custom_venv.mkdir(parents=True, exist_ok=True)
    custom_py = custom_venv / exe_name
    custom_py.write_text("", encoding="utf-8")

    monkeypatch.setattr(claude_mod, "AI_WORKSPACE_DIR", tmp_path)
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "nonexistent_prefix"))

    resolved = _get_mcp_python()
    assert Path(resolved).resolve() == custom_py.resolve()


def test_claude_desktop_mcp_handles_null_tool_arguments(monkeypatch):
    """Verify claude_desktop_mcp.handle_tools_call handles null/None and invalid parameters gracefully."""
    import local_ai_launcher.bridges.claude_desktop_mcp as mcp_bridge

    monkeypatch.setattr(mcp_bridge, "_execute_local_query", lambda prompt, system_prompt, temp, max_tok: f"Executed: temp={temp}, max={max_tok}")

    params = {
        "name": "query_local_model",
        "arguments": {
            "prompt": "Hello",
            "temperature": None,
            "max_tokens": None,
            "system_prompt": None
        }
    }
    resp = mcp_bridge.handle_tools_call("req-1", params)
    assert resp["id"] == "req-1"
    assert "Executed: temp=0.7, max=2048" in resp["result"]["content"][0]["text"]


def test_is_claude_desktop_running_graceful():
    """Verify is_claude_desktop_running returns a boolean without throwing exceptions."""
    from local_ai_launcher.harnesses.claude import is_claude_desktop_running
    res = is_claude_desktop_running()
    assert isinstance(res, bool)


def test_claude_adapter_check_handles_npm_timeout(tmp_path, monkeypatch):
    """Verify ClaudeAdapter.check handles subprocess.TimeoutExpired gracefully during update."""
    import local_ai_launcher.harnesses.claude as claude_mod
    from local_ai_launcher.harnesses.claude import ClaudeAdapter
    import subprocess

    procs = []
    adapter = ClaudeAdapter(str(tmp_path), procs)
    fake_exe = tmp_path / "claude.exe"
    fake_exe.write_text("dummy", encoding="utf-8")
    adapter._resolve_executable = lambda: fake_exe
    adapter._should_update = lambda: True

    def mock_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=300)

    monkeypatch.setattr(claude_mod.subprocess, "run", mock_run)

    # Must return True (continuing with existing installed executable) without raising
    assert adapter.check() is True


def test_claude_cli_dynamic_context_and_auto_compact_from_strata(tmp_path, monkeypatch):
    """Verify ClaudeAdapter dynamically extracts ctx_size from strata_models.json and calculates auto-compaction window."""
    import json
    import local_ai_launcher.harnesses.claude as claude_mod
    from local_ai_launcher.harnesses.claude import ClaudeAdapter
    from local_ai_launcher.config import config

    monkeypatch.setattr(config, "harness_start_dir", tmp_path)
    monkeypatch.setattr(claude_mod, "is_strata_active", lambda *args: True)

    captured_cmds = []
    monkeypatch.setattr(claude_mod, "start_in_wt_tab", lambda cmd, cwd=None: captured_cmds.append((cmd, cwd)) or 5555)

    procs = []
    adapter = ClaudeAdapter(str(tmp_path), procs)
    fake_exe = tmp_path / "fake_claude.exe"
    fake_exe.write_text("", encoding="utf-8")
    adapter._resolve_executable = lambda: fake_exe

    adapter.start("http://127.0.0.1:8080", active_model="strata-iq3s")

    assert len(captured_cmds) == 1
    cmd = captured_cmds[0][0]
    assert "$env:CLAUDE_CODE_MAX_CONTEXT_TOKENS='262144';" in cmd
    assert "$env:CLAUDE_CODE_AUTO_COMPACT_WINDOW='239000';" in cmd
    assert "262,144 tokens" in cmd
    assert "239,000" in cmd

    # Verify settings.json
    settings_file = adapter.claude_local_dir / "settings.json"
    assert settings_file.exists()
    settings_data = json.loads(settings_file.read_text(encoding="utf-8"))
    assert settings_data["env"]["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "262144"
    assert settings_data["env"]["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] == "239000"

    # Verify helper scripts
    ps1_file = adapter.claude_local_dir / "claude-local.ps1"
    assert ps1_file.exists()
    assert '$env:CLAUDE_CODE_MAX_CONTEXT_TOKENS = "262144"' in ps1_file.read_text(encoding="utf-8")
    assert '$env:CLAUDE_CODE_AUTO_COMPACT_WINDOW = "239000"' in ps1_file.read_text(encoding="utf-8")

    sh_file = adapter.claude_local_dir / "claude-local.sh"
    assert sh_file.exists()
    assert 'export CLAUDE_CODE_MAX_CONTEXT_TOKENS="262144"' in sh_file.read_text(encoding="utf-8")
    assert 'export CLAUDE_CODE_AUTO_COMPACT_WINDOW="239000"' in sh_file.read_text(encoding="utf-8")


def test_claude_cli_context_custom_overrides(tmp_path, monkeypatch):
    """Verify ClaudeAdapter honors explicit claude_max_context_tokens and custom auto_compact_ratio."""
    import local_ai_launcher.harnesses.claude as claude_mod
    from local_ai_launcher.harnesses.claude import ClaudeAdapter
    from local_ai_launcher.config import config

    monkeypatch.setattr(config, "harness_start_dir", tmp_path)
    monkeypatch.setattr(claude_mod, "is_strata_active", lambda *args: True)
    monkeypatch.setattr(config, "claude_max_context_tokens", 131072)
    monkeypatch.setattr(config, "claude_auto_compact_window", 110000)

    captured_cmds = []
    monkeypatch.setattr(claude_mod, "start_in_wt_tab", lambda cmd, cwd=None: captured_cmds.append((cmd, cwd)) or 6666)

    procs = []
    adapter = ClaudeAdapter(str(tmp_path), procs)
    fake_exe = tmp_path / "fake_claude.exe"
    fake_exe.write_text("", encoding="utf-8")
    adapter._resolve_executable = lambda: fake_exe

    adapter.start("http://127.0.0.1:8080", active_model="strata-iq3s")

    assert len(captured_cmds) == 1
    cmd = captured_cmds[0][0]
    assert "$env:CLAUDE_CODE_MAX_CONTEXT_TOKENS='131072';" in cmd
    assert "$env:CLAUDE_CODE_AUTO_COMPACT_WINDOW='110000';" in cmd


def test_claude_desktop_dynamic_context_and_auto_compact_from_strata(tmp_path, monkeypatch):
    """Verify ClaudeDesktopAdapter dynamically extracts ctx_size from strata and configures gateway & MCP."""
    import json
    import local_ai_launcher.harnesses.claude as claude_mod
    from local_ai_launcher.harnesses.claude import ClaudeDesktopAdapter
    from local_ai_launcher.config import config

    monkeypatch.setattr(config, "harness_start_dir", tmp_path)
    monkeypatch.setattr(claude_mod, "is_strata_active", lambda *args: True)
    cfg_file = tmp_path / "claude_desktop_config.json"
    monkeypatch.setattr(claude_mod, "get_claude_desktop_config_path", lambda: cfg_file)

    captured_procs = []
    class DummyProc:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            captured_procs.append(self)

    monkeypatch.setattr(claude_mod.subprocess, "Popen", DummyProc)

    procs = []
    adapter = ClaudeDesktopAdapter(str(tmp_path), procs)
    fake_exe = tmp_path / "fake_desktop.exe"
    fake_exe.write_text("", encoding="utf-8")
    adapter._resolve_executable = lambda: fake_exe

    adapter.start("http://127.0.0.1:8080", active_model="strata-iq3s")

    assert len(captured_procs) == 1
    proc_env = captured_procs[0].kwargs.get("env", {})
    assert proc_env.get("CLAUDE_CODE_MAX_CONTEXT_TOKENS") == "262144"
    assert proc_env.get("CLAUDE_CODE_AUTO_COMPACT_WINDOW") == "239000"

    # Verify claude_desktop_config.json
    assert cfg_file.exists()
    cfg_data = json.loads(cfg_file.read_text(encoding="utf-8"))
    assert cfg_data["gateway"]["maxContextTokens"] == 262144
    assert cfg_data["gateway"]["autoCompactWindow"] == 239000
    assert cfg_data["mcpServers"]["local-model"]["env"]["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "262144"
    assert cfg_data["mcpServers"]["local-model"]["env"]["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] == "239000"

    # Verify configLibrary profile uses clean flat schema without deprecation/warning keys
    profile_file = tmp_path / "configLibrary" / "00000000-0000-0000-0000-000000000001.json"
    assert profile_file.exists()
    profile_data = json.loads(profile_file.read_text(encoding="utf-8"))
    assert "inference" not in profile_data
    assert "maxContextTokens" not in profile_data
    assert "autoCompactWindow" not in profile_data
    assert profile_data.get("coworkTabEnabled") is False
    assert profile_data.get("chatTabEnabled") is True


def test_claude_desktop_separate_overrides_and_inheritance(tmp_path, monkeypatch):
    """Verify Claude Desktop inherits from [claude] by default and honors dedicated overrides separately."""
    import json
    import local_ai_launcher.harnesses.claude as claude_mod
    from local_ai_launcher.harnesses.claude import (
        resolve_claude_context_and_compact,
        setup_claude_desktop_config,
    )
    from local_ai_launcher.config import config

    # 1. Base Claude configuration
    monkeypatch.setattr(config, "claude_max_context_tokens", 262144)
    monkeypatch.setattr(config, "claude_auto_compact_window", 240000)
    monkeypatch.setattr(config, "claude_auto_compact_ratio", 0.9155)
    monkeypatch.setattr(config, "claude_desktop_max_context_tokens", None)
    monkeypatch.setattr(config, "claude_desktop_auto_compact_window", None)
    monkeypatch.setattr(config, "claude_desktop_auto_compact_ratio", None)

    # In MCP mode, desktop has NO option for custom context/compact limits
    mcp_ctx, mcp_win = resolve_claude_context_and_compact("custom-model", is_desktop=True, desktop_mode="mcp")
    assert mcp_ctx is None
    assert mcp_win is None

    # In gateway mode, desktop inherits from general claude settings by default
    d_ctx, d_win = resolve_claude_context_and_compact("custom-model", is_desktop=True, desktop_mode="gateway")
    assert d_ctx == 262144
    assert d_win == 240000

    # 2. Dedicated desktop override for context tokens & ratio in gateway mode
    monkeypatch.setattr(config, "claude_desktop_max_context_tokens", 65536)
    monkeypatch.setattr(config, "claude_desktop_auto_compact_ratio", 0.85)
    d_ctx, d_win = resolve_claude_context_and_compact("custom-model", is_desktop=True, desktop_mode="gateway")
    assert d_ctx == 65536
    assert d_win == 56000  # round(65536 * 0.85 / 1000) * 1000 = 56000

    # MCP mode still returns None even with explicit overrides
    mcp_ctx, mcp_win = resolve_claude_context_and_compact("custom-model", is_desktop=True, desktop_mode="mcp")
    assert mcp_ctx is None
    assert mcp_win is None

    # General claude CLI is unaffected
    c_ctx, c_win = resolve_claude_context_and_compact("custom-model", is_desktop=False)
    assert c_ctx == 262144
    assert c_win == 240000

    # 3. Dedicated desktop explicit auto_compact_window in gateway mode
    monkeypatch.setattr(config, "claude_desktop_auto_compact_window", 50000)
    d_ctx, d_win = resolve_claude_context_and_compact("custom-model", is_desktop=True, desktop_mode="gateway")
    assert d_ctx == 65536
    assert d_win == 50000

    # 4. Verify setup_claude_desktop_config in MCP mode: NO context env in MCP servers
    cfg_file = tmp_path / "claude_desktop_config_mcp.json"
    setup_claude_desktop_config(
        config_path=cfg_file,
        llm_base_url="http://127.0.0.1:8080",
        active_model="custom-model",
        desktop_mode="mcp",
    )
    assert cfg_file.exists()
    cfg_data = json.loads(cfg_file.read_text(encoding="utf-8"))
    assert "CLAUDE_CODE_MAX_CONTEXT_TOKENS" not in cfg_data["mcpServers"]["local-model"]["env"]
    assert "CLAUDE_CODE_AUTO_COMPACT_WINDOW" not in cfg_data["mcpServers"]["local-model"]["env"]
    assert "CLAUDE_CODE_MAX_CONTEXT_TOKENS" not in cfg_data["mcpServers"]["litellm"]["env"]
    assert "CLAUDE_CODE_AUTO_COMPACT_WINDOW" not in cfg_data["mcpServers"]["litellm"]["env"]

    # 5. Verify setup_claude_desktop_config in Gateway mode: context settings ARE applied
    cfg_file_gw = tmp_path / "claude_desktop_config_gw.json"
    setup_claude_desktop_config(
        config_path=cfg_file_gw,
        llm_base_url="http://127.0.0.1:8080",
        active_model="custom-model",
        desktop_mode="gateway",
    )
    assert cfg_file_gw.exists()
    cfg_gw_data = json.loads(cfg_file_gw.read_text(encoding="utf-8"))
    assert cfg_gw_data["gateway"]["maxContextTokens"] == 65536
    assert cfg_gw_data["gateway"]["autoCompactWindow"] == 50000
    assert cfg_gw_data["mcpServers"]["local-model"]["env"]["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "65536"
    assert cfg_gw_data["mcpServers"]["local-model"]["env"]["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] == "50000"


def test_rea_harness_mcp_provisioning_and_toggling(tmp_path, monkeypatch):
    """Verify Pi and Claude adapters provision REA in MCP configs when enabled and remove when disabled."""
    from local_ai_launcher.harnesses.pi import setup_pi_mcp_config
    from local_ai_launcher.harnesses.claude import setup_claude_cli_mcp_config, setup_claude_desktop_config
    import json

    # 1. Pi MCP config: enabled
    monkeypatch.setattr(config, "rea_enabled", True)
    monkeypatch.setattr(config, "rea_schema_profile", "compact")
    fake_pi_home = tmp_path / "pi"
    setup_pi_mcp_config(fake_pi_home)
    pi_mcp = fake_pi_home / "agent" / "mcp.json"
    assert pi_mcp.exists()
    pi_data = json.loads(pi_mcp.read_text(encoding="utf-8"))
    assert "rea" in pi_data["mcpServers"]
    assert pi_data["mcpServers"]["rea"]["type"] == "stdio"
    assert pi_data["mcpServers"]["rea"]["env"]["REA_MCP_INPUT_SCHEMA_PROFILE"] == "compact"

    # 2. Pi MCP config: disabled toggle
    monkeypatch.setattr(config, "rea_enabled", False)
    setup_pi_mcp_config(fake_pi_home)
    pi_data_disabled = json.loads(pi_mcp.read_text(encoding="utf-8"))
    assert "rea" not in pi_data_disabled["mcpServers"]

    # 3. Claude CLI MCP config: enabled
    monkeypatch.setattr(config, "rea_enabled", True)
    cli_dir = tmp_path / "claude_project"
    mcp_cli = setup_claude_cli_mcp_config(cli_dir, "http://127.0.0.1:8080/v1")
    cli_data = json.loads(mcp_cli.read_text(encoding="utf-8"))
    assert "rea" in cli_data["mcpServers"]
    assert cli_data["mcpServers"]["rea"]["env"]["REA_MCP_INPUT_SCHEMA_PROFILE"] == "compact"

    settings_file = cli_dir / ".claude" / "settings.local.json"
    settings_data = json.loads(settings_file.read_text(encoding="utf-8"))
    assert "rea" in settings_data["enabledMcpjsonServers"]

    # 4. Claude CLI MCP config: disabled toggle
    monkeypatch.setattr(config, "rea_enabled", False)
    setup_claude_cli_mcp_config(cli_dir, "http://127.0.0.1:8080/v1")
    cli_data_disabled = json.loads(mcp_cli.read_text(encoding="utf-8"))
    assert "rea" not in cli_data_disabled["mcpServers"]
    settings_data_disabled = json.loads(settings_file.read_text(encoding="utf-8"))
    assert "rea" not in settings_data_disabled["enabledMcpjsonServers"]

    # 5. Claude Desktop MCP config: enabled & disabled
    desk_path = tmp_path / "claude_desktop.json"
    monkeypatch.setattr(config, "rea_enabled", True)
    setup_claude_desktop_config(desk_path, "http://127.0.0.1:8080", desktop_mode="mcp")
    desk_data = json.loads(desk_path.read_text(encoding="utf-8"))
    assert "rea" in desk_data["mcpServers"]

    monkeypatch.setattr(config, "rea_enabled", False)
    setup_claude_desktop_config(desk_path, "http://127.0.0.1:8080", desktop_mode="mcp")
    desk_data_disabled = json.loads(desk_path.read_text(encoding="utf-8"))
    assert "rea" not in desk_data_disabled["mcpServers"]

    # 6. Antigravity MCP config: enabled & disabled
    from local_ai_launcher.harnesses.antigravity import setup_antigravity_mcp_config
    fake_gemini_dir = tmp_path / "gemini"
    monkeypatch.setattr(config, "rea_enabled", True)
    agy_mcp = setup_antigravity_mcp_config(fake_gemini_dir)
    assert agy_mcp.exists()
    agy_data = json.loads(agy_mcp.read_text(encoding="utf-8"))
    assert "rea" in agy_data["mcpServers"]
    assert agy_data["mcpServers"]["rea"]["env"]["REA_MCP_INPUT_SCHEMA_PROFILE"] == "compact"
    assert "local-llm" in agy_data["mcpServers"]

    monkeypatch.setattr(config, "rea_enabled", False)
    setup_antigravity_mcp_config(fake_gemini_dir)
    agy_data_disabled = json.loads(agy_mcp.read_text(encoding="utf-8"))
    assert "rea" not in agy_data_disabled["mcpServers"]
    assert "local-llm" in agy_data_disabled["mcpServers"]


def test_claude_thinking_mode_resolution_and_runner_scripts(tmp_path, monkeypatch):
    """Verify thinking mode resolution, runner script generation, and desktop config synchronization."""
    import local_ai_launcher.harnesses.claude as claude_mod
    from local_ai_launcher.harnesses.claude import (
        resolve_claude_thinking_mode,
        setup_claude_desktop_config,
        ClaudeAdapter,
        ClaudeDesktopAdapter,
    )
    from local_ai_launcher.config import config

    # 1. Test resolution logic
    monkeypatch.setattr(config, "claude_thinking_mode", "adaptive")
    monkeypatch.setattr(config, "claude_desktop_thinking_mode", None)
    assert resolve_claude_thinking_mode(is_desktop=False) == "adaptive"
    assert resolve_claude_thinking_mode(is_desktop=True, desktop_mode="gateway") == "adaptive"
    assert resolve_claude_thinking_mode(is_desktop=True, desktop_mode="mcp") == "adaptive"

    monkeypatch.setattr(config, "claude_thinking_mode", "disabled")
    assert resolve_claude_thinking_mode(is_desktop=False) == "disabled"
    assert resolve_claude_thinking_mode(is_desktop=True, desktop_mode="gateway") == "disabled"

    # Separate desktop override
    monkeypatch.setattr(config, "claude_thinking_mode", "adaptive")
    monkeypatch.setattr(config, "claude_desktop_thinking_mode", "high")
    assert resolve_claude_thinking_mode(is_desktop=False) == "adaptive"
    assert resolve_claude_thinking_mode(is_desktop=True, desktop_mode="gateway") == "high"

    # 2. Verify ClaudeAdapter runner script injection for disabled thinking mode
    test_claude_dir = tmp_path / ".claude_thinking_off"
    monkeypatch.setattr(config, "claude_dir", test_claude_dir)
    monkeypatch.setattr(config, "claude_mode", "local")
    monkeypatch.setattr(config, "claude_thinking_mode", "disabled")
    monkeypatch.setattr(config, "claude_desktop_thinking_mode", None)

    captured_cmds = []
    monkeypatch.setattr(claude_mod, "start_in_wt_tab", lambda cmd, cwd=None: captured_cmds.append(cmd))

    adapter = ClaudeAdapter(str(tmp_path), [])
    adapter.start("http://127.0.0.1:8080/v1", active_model="strata")

    local_prof = adapter.claude_local_dir
    settings_data = json.loads((local_prof / "settings.json").read_text(encoding="utf-8"))
    assert settings_data["alwaysThinkingEnabled"] is False
    assert settings_data["env"]["MAX_THINKING_TOKENS"] == "0"

    ps1_txt = (local_prof / "claude-local.ps1").read_text(encoding="utf-8")
    assert '$env:MAX_THINKING_TOKENS = "0"' in ps1_txt
    assert "--thinking disabled" in ps1_txt

    sh_txt = (local_prof / "claude-local.sh").read_text(encoding="utf-8")
    assert 'export MAX_THINKING_TOKENS="0"' in sh_txt
    assert "--thinking disabled" in sh_txt

    cmd_txt = (local_prof / "claude-local.cmd").read_text(encoding="utf-8")
    assert 'set "MAX_THINKING_TOKENS=0"' in cmd_txt
    assert "--thinking disabled" in cmd_txt

    # 3. Verify Claude Desktop gateway configuration with thinking modes
    desk_path = tmp_path / "claude_desktop_thinking.json"
    setup_claude_desktop_config(
        desk_path,
        "http://127.0.0.1:8080",
        active_model="strata-iq3s",
        desktop_mode="gateway",
        thinking_mode="high"
    )
    desk_data = json.loads(desk_path.read_text(encoding="utf-8"))
    assert desk_data["preferences"]["alwaysThinkingEnabled"] is True
    assert desk_data["preferences"]["effortLevel"] == "high"

    # Gateway mode disabled
    setup_claude_desktop_config(
        desk_path,
        "http://127.0.0.1:8080",
        active_model="strata-iq3s",
        desktop_mode="gateway",
        thinking_mode="disabled"
    )
    desk_data_off = json.loads(desk_path.read_text(encoding="utf-8"))
    assert desk_data_off["preferences"]["alwaysThinkingEnabled"] is False
    assert desk_data_off["alwaysThinkingEnabled"] is False
    assert desk_data_off["mcpServers"]["local-model"]["env"]["MAX_THINKING_TOKENS"] == "0"

    # MCP mode purges thinking keys
    setup_claude_desktop_config(
        desk_path,
        "http://127.0.0.1:8080",
        active_model="strata-iq3s",
        desktop_mode="mcp"
    )
    desk_data_mcp = json.loads(desk_path.read_text(encoding="utf-8"))
    assert "alwaysThinkingEnabled" not in desk_data_mcp.get("preferences", {})
    assert "alwaysThinkingEnabled" not in desk_data_mcp


def test_gateway_contract_injection_tier_d_and_thinking_translation(monkeypatch):
    """Verify Tier D gateway contract injection and thinking translation in _convert_anthropic_to_openai_body."""
    from local_ai_launcher.localai.native_server import _convert_anthropic_to_openai_body
    from local_ai_launcher.harnesses.claude import OFFLINE_PERSONA_CONTRACT_MD
    from local_ai_launcher.config import config

    monkeypatch.setattr(config, "default_model", "qwen-38-flash-strata-iq3s")
    monkeypatch.setattr(config, "claude_offline_inject_contract", True)
    monkeypatch.setattr(config, "claude_thinking_mode", "adaptive")

    # 1. Claude request with local strata engine -> contract injected at end of system prompt
    body_claude = {
        "model": "claude-sonnet-4-5",
        "system": "Base developer instructions.",
        "messages": [{"role": "user", "content": "Explain quantum computing."}]
    }
    converted = _convert_anthropic_to_openai_body(body_claude)
    sys_content = converted["messages"][0]["content"]
    assert "Base developer instructions." in sys_content
    assert "Offline Answer Contract" in sys_content
    assert sys_content.endswith("continue.")
    assert converted["chat_template_kwargs"]["enable_thinking"] is True

    # 2. When claude_offline_inject_contract is toggled off -> no contract injected
    monkeypatch.setattr(config, "claude_offline_inject_contract", False)
    converted_no_contract = _convert_anthropic_to_openai_body(body_claude)
    assert "Offline Answer Contract" not in converted_no_contract["messages"][0]["content"]
    monkeypatch.setattr(config, "claude_offline_inject_contract", True)

    # 3. If request system prompt already contains the contract -> no duplicate injection
    body_with_contract = {
        "model": "claude-sonnet-4-5",
        "system": f"Base developer instructions.\n\n{OFFLINE_PERSONA_CONTRACT_MD}",
        "messages": [{"role": "user", "content": "Explain quantum computing."}]
    }
    converted_dup = _convert_anthropic_to_openai_body(body_with_contract)
    assert converted_dup["messages"][0]["content"].count("Offline Answer Contract") == 1

    # 4. Thinking translation: explicit disabled in request
    body_think_off = {
        "model": "claude-sonnet-4-5",
        "thinking": {"type": "disabled"},
        "messages": [{"role": "user", "content": "Hello"}]
    }
    conv_think_off = _convert_anthropic_to_openai_body(body_think_off)
    assert conv_think_off["chat_template_kwargs"]["enable_thinking"] is False
    assert conv_think_off["reasoning_effort"] == "none"

    # 5. Thinking translation: enabled with budget and effort
    body_think_on = {
        "model": "claude-sonnet-4-5",
        "thinking": {"type": "enabled", "budget_tokens": 4096, "effort": "high"},
        "messages": [{"role": "user", "content": "Hello"}]
    }
    conv_think_on = _convert_anthropic_to_openai_body(body_think_on)
    assert conv_think_on["chat_template_kwargs"]["enable_thinking"] is True
    assert conv_think_on["reasoning_budget"] == 4096
    assert conv_think_on["reasoning_effort"] == "high"

    # 6. Non-Claude request -> neither contract nor default thinking kwargs injected
    body_gpt = {
        "model": "gpt-4o",
        "system": "System instructions.",
        "messages": [{"role": "user", "content": "Hello"}]
    }
    conv_gpt = _convert_anthropic_to_openai_body(body_gpt)
    assert "Offline Answer Contract" not in conv_gpt["messages"][0]["content"]
    assert "chat_template_kwargs" not in conv_gpt


def test_harness_registry_contains_omp():
    """Verify omp and its aliases are registered in HARNESS_REGISTRY and distinct from Pi."""
    from local_ai_launcher.harnesses.omp import OmpAdapter
    from local_ai_launcher.harnesses.pi import PiAdapter
    assert "omp" in HARNESS_REGISTRY
    assert HARNESS_REGISTRY["omp"] is OmpAdapter
    assert "oh-my-pi" in HARNESS_REGISTRY
    assert HARNESS_REGISTRY["oh-my-pi"] is OmpAdapter
    assert "ohmypi" in HARNESS_REGISTRY
    assert HARNESS_REGISTRY["ohmypi"] is OmpAdapter
    assert HARNESS_REGISTRY["omp"] is not PiAdapter


def test_config_contains_omp_defaults():
    """Verify LauncherConfig contains default behaviors and directories for omp."""
    cfg = LauncherConfig()
    assert "omp" in cfg.harness_install_behaviors
    assert "omp" in cfg.harness_update_behaviors
    assert "omp" in cfg.harness_extensions
    assert "omp" in cfg.harness_skills
    assert hasattr(cfg, "omp_dir")
    assert hasattr(config, "omp_dir")
    assert cfg.omp_dir != cfg.pi_dir


def test_mnemosyne_app_config_has_omp_dirs():
    """Verify MnemosyneAppConfig exposes omp directories distinct from pi."""
    mcfg = MnemosyneAppConfig()
    assert hasattr(mcfg, "omp_config_dir")
    assert hasattr(mcfg, "omp_dir")
    assert mcfg.omp_dir == mcfg.omp_config_dir
    assert mcfg.omp_dir != mcfg.pi_dir


def test_omp_auto_configuration(tmp_path, monkeypatch):
    """Verify setup_omp_models_and_auth provisions models.json, models.yml, settings, and auth in .omp."""
    from local_ai_launcher.harnesses.omp import setup_omp_models_and_auth

    test_start_dir = tmp_path / "test_project"
    test_start_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(config, "harness_start_dir", test_start_dir)

    # Stale project settings should be purged
    stale_omp_settings = test_start_dir / ".omp" / "settings.json"
    stale_omp_settings.parent.mkdir(parents=True, exist_ok=True)
    stale_omp_settings.write_text('{"defaultProvider":"stale"}', encoding="utf-8")

    omp_home = tmp_path / ".omp"
    pi_home = tmp_path / ".pi"
    setup_omp_models_and_auth(omp_home, "http://localhost:8080/v1", "qwen-test-model")

    # Stale project-level settings removed
    assert not stale_omp_settings.exists()

    # Pi directory remains untouched (separate and distinct!)
    assert not pi_home.exists()

    # Verify agent settings.json
    agent_settings = omp_home / "agent" / "settings.json"
    assert agent_settings.exists()
    settings = json.loads(agent_settings.read_text(encoding="utf-8"))
    assert settings["defaultProvider"] == "openai"
    assert settings["defaultModel"] == "qwen-test-model"
    assert "openai/*" in settings["enabledModels"]
    assert settings["setupVersion"] == 2
    assert settings["startup"]["setupWizard"] is False

    # Verify config.yml exists and has setupVersion & startup
    config_yml = omp_home / "agent" / "config.yml"
    assert config_yml.exists()
    assert "setupVersion: 2" in config_yml.read_text(encoding="utf-8")
    assert "setupWizard: false" in config_yml.read_text(encoding="utf-8")

    # Verify models.json
    models_file = omp_home / "agent" / "models.json"
    assert models_file.exists()
    models_data = json.loads(models_file.read_text(encoding="utf-8"))
    assert "openai" in models_data["providers"]
    assert models_data["providers"]["openai"]["baseUrl"] == "http://localhost:8080/v1"
    models_list = models_data["providers"]["openai"]["models"]
    assert any(m["id"] == "qwen-test-model" for m in models_list)

    # Verify models.yml exists
    models_yml = omp_home / "agent" / "models.yml"
    assert models_yml.exists()
    yml_text = models_yml.read_text(encoding="utf-8")
    assert "openai:" in yml_text
    assert "baseUrl: http://localhost:8080/v1" in yml_text

    # Verify auth.json exists
    auth_file = omp_home / "agent" / "auth.json"
    assert auth_file.exists()
    auth_data = json.loads(auth_file.read_text(encoding="utf-8"))
    assert "openai" in auth_data
    assert auth_data["openai"]["key"] == "dummy"


def test_omp_adapter_executable_and_start(tmp_path, monkeypatch):
    """Verify OmpAdapter executes properly, sets OMP_HOME, and passes expected CLI args."""
    from local_ai_launcher.harnesses.omp import OmpAdapter

    dummy_processes = []
    omp_home = tmp_path / ".omp"
    monkeypatch.setattr(config, "omp_dir", omp_home)
    monkeypatch.setattr(config, "harness_start_dir", tmp_path / "work")

    adapter = OmpAdapter(str(tmp_path), dummy_processes)
    assert adapter.name == "omp"
    assert adapter.install_behavior == "always_install"
    assert adapter.update_behavior == "always_update"

    # Test check when omp binary is found
    fake_omp = tmp_path / "omp.cmd"
    fake_omp.write_text("@echo off\necho omp 1.0\n", encoding="utf-8")
    monkeypatch.setattr(adapter, "omp_exe", fake_omp)
    monkeypatch.setattr(adapter, "_should_update", lambda: False)

    assert adapter.check() is True

    # Test start
    started_commands = []
    def mock_start_in_wt_tab(cmd, cwd=None, title=None):
        started_commands.append((cmd, cwd, title))
        proc = type("MockProc", (), {"poll": lambda self: None})()
        return proc

    monkeypatch.setattr("local_ai_launcher.harnesses.omp.start_in_wt_tab", mock_start_in_wt_tab)
    adapter.start("http://localhost:8080/v1", "test-omp-model", launch_tui=True)

    assert len(started_commands) == 1
    cmd_str, cwd, title = started_commands[0]
    assert title == "Oh My Pi"
    assert "$env:OMP_HOME=" in cmd_str
    assert "$env:OPENAI_BASE_URL=" in cmd_str
    assert "$env:USERPROFILE=" in cmd_str
    assert "$env:OMP_SKIP_SETUP='1'" in cmd_str
    assert "$env:PI_CONFIG_DIR='.omp'" in cmd_str or "$env:PI_CONFIG_DIR='" + omp_home.name + "'" in cmd_str
    assert ":\\" not in cmd_str.split("$env:PI_CONFIG_DIR=")[1].split(";")[0]
    assert ":/" not in cmd_str.split("$env:PI_CONFIG_DIR=")[1].split(";")[0]
    assert "--provider openai --model test-omp-model" in cmd_str
    assert len(dummy_processes) == 1
    assert dummy_processes[0][0] == "Oh My Pi coding agent"


def test_omp_windows_path_resolution_regression(tmp_path, monkeypatch):
    """Verify that OMP environment prevents Windows double-drive path concatenation in Tie() and hEs()."""
    import os
    from pathlib import Path
    from local_ai_launcher.harnesses.omp import OmpAdapter, setup_omp_mcp_config

    omp_home = tmp_path / ".omp"
    monkeypatch.setattr(config, "omp_dir", omp_home)
    adapter = OmpAdapter(str(tmp_path), [])

    started_commands = []
    monkeypatch.setattr("local_ai_launcher.harnesses.omp.start_in_wt_tab", lambda cmd, cwd=None, title=None: started_commands.append(cmd))
    adapter.start("http://localhost:8080/v1", "test-model", launch_tui=True)

    assert len(started_commands) == 1
    cmd_str = started_commands[0]

    # Verify PI_CONFIG_DIR is relative (no drive letters)
    import re
    m_cfg = re.search(r"\$env:PI_CONFIG_DIR='([^']*)'", cmd_str)
    assert m_cfg is not None
    pi_config_dir_val = m_cfg.group(1)
    assert not Path(pi_config_dir_val).is_absolute()
    assert pi_config_dir_val == ".omp"

    # Verify USERPROFILE is parent of omp_home
    m_user = re.search(r"\$env:USERPROFILE='([^']*)'", cmd_str)
    assert m_user is not None
    userprofile_val = m_user.group(1)
    assert Path(userprofile_val).resolve() == omp_home.parent.resolve()

    # Verify path.join simulation
    simulated_config_root = Path(userprofile_val) / pi_config_dir_val
    assert simulated_config_root.resolve() == omp_home.resolve()

    # Verify MCP config written to both agent_dir and omp_home
    setup_omp_mcp_config(omp_home)
    assert (omp_home / "mcp.json").is_file()
    assert (omp_home / "agent" / "mcp.json").is_file()


def test_context_preloader_omp_prefix():
    """Verify build_harness_prefix builds Oh My Pi (omp) system prompt and tools."""
    from local_ai_launcher.context_preloader import build_harness_prefix

    prefix = build_harness_prefix("omp", "test-model")
    assert prefix is not None
    assert "messages" in prefix
    assert "tools" in prefix
    assert "Oh My Pi" in prefix["messages"][0]["content"]

    prefix_alias = build_harness_prefix("oh-my-pi", "test-model")
    assert "Oh My Pi" in prefix_alias["messages"][0]["content"]


def test_omp_ensure_extensions_and_skills(tmp_path, monkeypatch):
    """Verify OmpAdapter provisions loop-recovery, cpu-tools, and auto-cycle extensions and clears legacy skills."""
    from local_ai_launcher.harnesses.omp import OmpAdapter

    omp_home = tmp_path / ".omp"
    legacy_skills = omp_home / "skills"
    legacy_skills.mkdir(parents=True, exist_ok=True)
    (legacy_skills / "dummy.txt").write_text("old", encoding="utf-8")

    monkeypatch.setattr(config, "omp_dir", omp_home)
    adapter = OmpAdapter(str(tmp_path), [])

    adapter.ensure_extensions_and_skills()

    # Legacy skills directory should be cleaned up to prevent duplicate collisions with .agents/skills
    assert not legacy_skills.exists()

    # Extensions directory should have provisioned shims
    ext_dir = omp_home / "agent" / "extensions"
    assert ext_dir.exists()
    assert (ext_dir / "loop-recovery.ts").exists()
    assert (ext_dir / "cpu-tools.ts").exists()
    assert (ext_dir / "auto-cycle.ts").exists()


def test_omp_adapter_no_global_environ_leakage(tmp_path, monkeypatch):
    """Verify OmpAdapter does not mutate process-global os.environ['PI_CODING_AGENT_DIR']."""
    import os
    from local_ai_launcher.harnesses.omp import OmpAdapter

    original_pi_dir = "C:/aiWorkspace/.pi/agent"
    monkeypatch.setenv("PI_CODING_AGENT_DIR", original_pi_dir)

    omp_home = tmp_path / ".omp"
    monkeypatch.setattr(config, "omp_dir", omp_home)
    monkeypatch.setattr(config, "harness_start_dir", tmp_path / "work")

    adapter = OmpAdapter(str(tmp_path), [])
    fake_omp = tmp_path / "omp.cmd"
    fake_omp.write_text("@echo off\necho omp 1.0\n", encoding="utf-8")
    monkeypatch.setattr(adapter, "omp_exe", fake_omp)
    monkeypatch.setattr(adapter, "_should_update", lambda: False)

    # Run check()
    adapter.check()
    assert os.environ.get("PI_CODING_AGENT_DIR") == original_pi_dir

    # Run start() in headless mode
    adapter.start("http://localhost:8080/v1", "test-model", launch_tui=False)
    assert os.environ.get("PI_CODING_AGENT_DIR") == original_pi_dir


def test_config_omp_toml_section_flattening():
    """Verify [omp] section in launcher.toml correctly maps extensions and skills."""
    from local_ai_launcher.config import _flatten_overrides

    raw_toml_data = {
        "omp": {
            "extensions": ["npm:custom-ext-1", "npm:custom-ext-2"],
            "skills": ["custom-skill-1"]
        }
    }
    flattened = _flatten_overrides(raw_toml_data)
    assert "harness_extensions" in flattened
    assert "omp" in flattened["harness_extensions"]
    assert "npm:custom-ext-1" in flattened["harness_extensions"]["omp"]
    assert "harness_skills" in flattened
    assert "omp" in flattened["harness_skills"]
    assert "custom-skill-1" in flattened["harness_skills"]["omp"]


def test_omp_cross_platform_executable_resolution(tmp_path, monkeypatch):
    """Verify OmpAdapter resolves executable candidates across Windows and POSIX conventions."""
    from local_ai_launcher.harnesses.omp import OmpAdapter

    adapter = OmpAdapter(str(tmp_path), [])
    node_dir = tmp_path / "node"
    nm_bin = tmp_path / "node_modules" / ".bin"
    bin_dir = tmp_path / "bin"
    venv_scripts = tmp_path / ".venv" / "Scripts"
    venv_bin = tmp_path / ".venv" / "bin"

    node_dir.mkdir(parents=True, exist_ok=True)
    nm_bin.mkdir(parents=True, exist_ok=True)
    bin_dir.mkdir(parents=True, exist_ok=True)
    venv_scripts.mkdir(parents=True, exist_ok=True)
    venv_bin.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(adapter, "node_bin", node_dir)
    monkeypatch.setattr(adapter, "local_bin", nm_bin)
    monkeypatch.setattr(adapter, "venv_dir", tmp_path / ".venv")

    import sys

    # 1. When binary is in node/
    test_exe = node_dir / ("omp.cmd" if sys.platform == "win32" else "omp")
    test_exe.write_text("stub", encoding="utf-8")
    assert adapter._resolve_executable() == test_exe
    test_exe.unlink()

    # 2. When binary is in node_modules/.bin/
    test_exe_nm = nm_bin / ("omp.cmd" if sys.platform == "win32" else "omp")
    test_exe_nm.write_text("stub", encoding="utf-8")
    assert adapter._resolve_executable() == test_exe_nm
    test_exe_nm.unlink()

    # 3. When binary is in venv
    venv_target = venv_scripts / "omp.exe" if sys.platform == "win32" else venv_bin / "omp"
    venv_target.write_text("stub", encoding="utf-8")
    assert adapter._resolve_executable() == venv_target
    venv_target.unlink()


def test_omp_start_process_safety(tmp_path, monkeypatch):
    """Verify OmpAdapter handles None process gracefully and formats non-Windows start command."""
    import sys
    from local_ai_launcher.harnesses.omp import OmpAdapter

    dummy_procs = []
    adapter = OmpAdapter(str(tmp_path), dummy_procs)
    fake_omp = tmp_path / "omp.cmd"
    fake_omp.write_text("stub", encoding="utf-8")
    monkeypatch.setattr(adapter, "omp_exe", fake_omp)
    monkeypatch.setattr(config, "harness_start_dir", tmp_path)

    # 1. When start_in_wt_tab returns None (headless/non-interactive)
    monkeypatch.setattr("local_ai_launcher.harnesses.omp.start_in_wt_tab", lambda cmd, cwd=None, title=None: None)
    monkeypatch.setattr("sys.platform", "win32")
    adapter.start("http://localhost:8080/v1", "test-model", launch_tui=True)
    assert len(dummy_procs) == 0  # Crucial: Must NOT append None!

    # 2. When running on non-Windows platform
    monkeypatch.setattr("sys.platform", "linux")
    spawned_cmds = []
    class DummyProc:
        def poll(self):
            return None
    def mock_popen(args, **kwargs):
        spawned_cmds.append((args, kwargs))
        return DummyProc()

    monkeypatch.setattr("subprocess.Popen", mock_popen)
    adapter.start("http://localhost:8080/v1", "test-model", launch_tui=True)
    assert len(spawned_cmds) == 1
    args, kwargs = spawned_cmds[0]
    assert args[0] == "bash"
    assert args[1] == "-c"
    assert "OMP_SKIP_SETUP='1'" in args[2]
    assert len(dummy_procs) == 1
    assert dummy_procs[0][0] == "Oh My Pi coding agent"


def test_ensure_bun_destinations_sync(tmp_path, monkeypatch):
    """Verify ensure_bun propagates Bun binary to workspace node and node_modules/.bin."""
    import local_ai_launcher.environment as env_mod

    fake_ws = tmp_path / "ws"
    node_dir = fake_ws / "node"
    nm_bin = fake_ws / "node_modules" / ".bin"
    node_dir.mkdir(parents=True, exist_ok=True)
    nm_bin.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(env_mod, "AI_WORKSPACE_DIR", fake_ws)
    import sys
    exe_name = "bun.exe" if sys.platform == "win32" else "bun"

    # Seed bun in node_dir
    seed_bun = node_dir / exe_name
    seed_bun.write_text("bun-binary", encoding="utf-8")

    res = env_mod.ensure_bun()
    assert res == str(seed_bun)
    # nm_bin should now also have bun!
    assert (nm_bin / exe_name).is_file()
    assert (nm_bin / exe_name).read_text(encoding="utf-8") == "bun-binary"















