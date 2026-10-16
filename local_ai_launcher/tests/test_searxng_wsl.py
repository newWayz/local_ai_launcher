"""
Unit tests for WSL distribution auto-provisioning and graceful fallbacks in SearXNG backend.
"""
import sys
from unittest.mock import patch, MagicMock
from subprocess import CompletedProcess

from pathlib import Path
from local_ai_launcher.backends.searxng import (
    _check_wsl_status,
    ensure_wsl_distribution,
    start_searxng_wsl_docker,
    to_wsl_path,
    ensure_wsl_keepalive_config,
)


def test_check_wsl_status_with_installed_distro():
    with patch("sys.platform", "win32"), \
         patch("local_ai_launcher.backends.searxng.run_silent_subprocess") as mock_sub:
        mock_sub.return_value = CompletedProcess(
            args=["wsl.exe", "-l", "-q"],
            returncode=0,
            stdout="Ubuntu-26.04\n\x00Debian\n"
        )
        has_wsl, distro = _check_wsl_status()
        assert has_wsl is True
        assert distro == "Ubuntu-26.04"


def test_check_wsl_status_with_zero_installs():
    with patch("sys.platform", "win32"), \
         patch("local_ai_launcher.backends.searxng.run_silent_subprocess") as mock_sub:
        mock_sub.return_value = CompletedProcess(
            args=["wsl.exe", "-l", "-q"],
            returncode=1,
            stdout="Windows Subsystem for Linux has no installed distributions.\n"
        )
        has_wsl, distro = _check_wsl_status()
        assert has_wsl is False
        assert distro is None


def test_ensure_wsl_distribution_noop_when_distro_exists():
    with patch("sys.platform", "win32"), \
         patch("shutil.which", return_value="C:\\Windows\\System32\\wsl.exe"), \
         patch("local_ai_launcher.backends.searxng._check_wsl_status", return_value=(True, "Ubuntu-26.04")), \
         patch("local_ai_launcher.backends.searxng.run_silent_subprocess") as mock_sub:
        res = ensure_wsl_distribution("Ubuntu-26.04")
        assert res is True
        mock_sub.assert_not_called()


def test_ensure_wsl_distribution_installs_ubuntu_when_zero_installs():
    with patch("sys.platform", "win32"), \
         patch("shutil.which", return_value="C:\\Windows\\System32\\wsl.exe"), \
         patch("local_ai_launcher.backends.searxng._check_wsl_status", return_value=(False, None)), \
         patch("local_ai_launcher.backends.searxng.run_silent_subprocess") as mock_sub:
        mock_sub.return_value = CompletedProcess(
            args=["wsl.exe", "--install", "Ubuntu-26.04", "--no-launch", "--web-download"],
            returncode=0,
            stdout="Distribution successfully installed."
        )
        res = ensure_wsl_distribution("Ubuntu-26.04")
        assert res is True
        mock_sub.assert_called_once_with(
            ["wsl.exe", "--install", "Ubuntu-26.04", "--no-launch", "--web-download"],
            timeout=180
        )


def test_ensure_wsl_distribution_fallback_to_generic_ubuntu():
    with patch("sys.platform", "win32"), \
         patch("shutil.which", return_value="C:\\Windows\\System32\\wsl.exe"), \
         patch("local_ai_launcher.backends.searxng._check_wsl_status", return_value=(False, None)), \
         patch("local_ai_launcher.backends.searxng.run_silent_subprocess") as mock_sub:
        # First call fails, second call succeeds
        mock_sub.side_effect = [
            CompletedProcess(args=[], returncode=1, stderr="Distro not found"),
            CompletedProcess(args=[], returncode=0, stdout="Installed generic Ubuntu")
        ]
        res = ensure_wsl_distribution("Ubuntu-26.04")
        assert res is True
        assert mock_sub.call_count == 2


def test_ensure_wsl_distribution_fails_gracefully_when_all_fail():
    with patch("sys.platform", "win32"), \
         patch("shutil.which", return_value="C:\\Windows\\System32\\wsl.exe"), \
         patch("local_ai_launcher.backends.searxng._check_wsl_status", return_value=(False, None)), \
         patch("local_ai_launcher.backends.searxng.run_silent_subprocess") as mock_sub:
        mock_sub.return_value = CompletedProcess(args=[], returncode=1, stderr="No network")
        res = ensure_wsl_distribution("Ubuntu-26.04")
        assert res is False


def test_start_searxng_wsl_docker_fallbacks_to_native_when_wsl_unusable():
    procs = []
    with patch("local_ai_launcher.backends.searxng.is_searxng_healthy", side_effect=[False, True]), \
         patch("local_ai_launcher.backends.searxng.ensure_wsl_keepalive_config"), \
         patch("local_ai_launcher.backends.searxng._check_wsl_status", return_value=(False, None)), \
         patch("local_ai_launcher.backends.searxng.ensure_wsl_distribution", return_value=False), \
         patch("local_ai_launcher.backends.searxng.start_searxng_native", return_value=True) as mock_native:
        ok = start_searxng_wsl_docker(procs, port=8088)
        assert ok is True
        mock_native.assert_called_once_with(procs, 8088)


def test_to_wsl_path():
    with patch("sys.platform", "win32"):
        p_c = Path("C:/aiWorkspace/tools/searxng")
        assert to_wsl_path(p_c) == "/mnt/c/aiWorkspace/tools/searxng"

        p_d = Path("D:/data/searxng")
        assert to_wsl_path(p_d) == "/mnt/d/data/searxng"

    with patch("sys.platform", "linux"):
        p_linux = Path("/var/lib/searxng")
        assert to_wsl_path(p_linux) == "/var/lib/searxng"


def test_ensure_wsl_keepalive_config_creates_new(tmp_path):
    with patch("sys.platform", "win32"), \
         patch("pathlib.Path.home", return_value=tmp_path):
        res = ensure_wsl_keepalive_config()
        assert res is True
        cfg = tmp_path / ".wslconfig"
        assert cfg.exists()
        content = cfg.read_text(encoding="utf-8")
        assert "[wsl2]" in content
        assert "vmIdleTimeout=-1" in content


def test_ensure_wsl_keepalive_config_updates_existing(tmp_path):
    cfg = tmp_path / ".wslconfig"
    cfg.write_text("[wsl2]\nmemory=8GB\nguiApplications=false\n", encoding="utf-8")

    with patch("sys.platform", "win32"), \
         patch("pathlib.Path.home", return_value=tmp_path):
        res = ensure_wsl_keepalive_config()
        assert res is True
        content = cfg.read_text(encoding="utf-8")
        assert "memory=8GB" in content
        assert "vmIdleTimeout=-1" in content


def test_ensure_wsl_keepalive_config_noop_if_present(tmp_path):
    cfg = tmp_path / ".wslconfig"
    cfg.write_text("[wsl2]\nmemory=8GB\nvmIdleTimeout=-1\n", encoding="utf-8")

    with patch("sys.platform", "win32"), \
         patch("pathlib.Path.home", return_value=tmp_path):
        res = ensure_wsl_keepalive_config()
        assert res is True
        content = cfg.read_text(encoding="utf-8")
        assert content.count("vmIdleTimeout") == 1


def test_start_searxng_wsl_docker_launches_with_resource_limits():
    procs = []
    with patch("local_ai_launcher.backends.searxng.is_searxng_healthy", side_effect=[False, True]), \
         patch("local_ai_launcher.backends.searxng.ensure_wsl_keepalive_config"), \
         patch("local_ai_launcher.backends.searxng._check_wsl_status", return_value=(True, "Ubuntu-26.04")), \
         patch("local_ai_launcher.backends.searxng.run_silent_subprocess") as mock_sub, \
         patch("local_ai_launcher.backends.searxng.ensure_searxng_settings_file"):
        
        # 1. docker info -> ok (0)
        # 2. docker ps -a -> empty (not created yet)
        # 3. docker run -> ok (0)
        mock_sub.side_effect = [
            CompletedProcess(args=[], returncode=0, stdout=""),
            CompletedProcess(args=[], returncode=0, stdout=""),
            CompletedProcess(args=[], returncode=0, stdout="container-id-123"),
        ]
        
        ok = start_searxng_wsl_docker(procs, port=8088, distro="Ubuntu-26.04")
        assert ok is True
        
        # Verify docker run command arguments
        run_call_args = mock_sub.call_args_list[2][0][0]
        assert "docker" in run_call_args
        assert "run" in run_call_args
        assert "--memory" in run_call_args
        assert "512m" in run_call_args
        assert "--cpus" in run_call_args
        assert "2" in run_call_args
        assert "-e" in run_call_args
        assert "UWSGI_WORKERS=1" in run_call_args
        assert "/etc/searxng:rw" in " ".join(run_call_args)


def test_start_searxng_wsl_docker_updates_running_container_limits():
    procs = []
    with patch("local_ai_launcher.backends.searxng.is_searxng_healthy", side_effect=[False, True]), \
         patch("local_ai_launcher.backends.searxng.ensure_wsl_keepalive_config"), \
         patch("local_ai_launcher.backends.searxng._check_wsl_status", return_value=(True, "Ubuntu-26.04")), \
         patch("local_ai_launcher.backends.searxng.run_silent_subprocess") as mock_sub:
        
        # 1. docker info -> ok (0)
        # 2. docker ps -a -> Up (container running)
        # 3. docker update -> ok (0)
        mock_sub.side_effect = [
            CompletedProcess(args=[], returncode=0, stdout=""),
            CompletedProcess(args=[], returncode=0, stdout="Up 2 hours"),
            CompletedProcess(args=[], returncode=0, stdout="searxng"),
        ]
        
        ok = start_searxng_wsl_docker(procs, port=8088, distro="Ubuntu-26.04")
        assert ok is True
        
        # Verify docker update command was issued
        update_call_args = mock_sub.call_args_list[2][0][0]
        assert "docker" in update_call_args
        assert "update" in update_call_args
        assert "--memory" in update_call_args
        assert "512m" in update_call_args
        assert "--cpus" in update_call_args
        assert "2" in update_call_args


def test_should_start_searxng_policy(monkeypatch):
    """Verify should_start_searxng only starts SearXNG when a harness requiring SearXNG is active."""
    from local_ai_launcher.config import config
    from local_ai_launcher.backends.searxng import should_start_searxng

    monkeypatch.setattr(config, "searxng_enabled", True)
    monkeypatch.setattr(config, "searxng_mode", "wsl_docker")
    monkeypatch.setattr(config, "pi_donsetch_enabled", True)
    monkeypatch.setattr(config, "deepseek_donsetch_enabled", True)

    # 1. Solely Pi or DSH or both -> SearXNG should NOT be started
    assert should_start_searxng(["pi"]) is False
    assert should_start_searxng(["dsh"]) is False
    assert should_start_searxng(["deepseek"]) is False
    assert should_start_searxng(["pi", "dsh"]) is False
    assert should_start_searxng(["pi", "deepseek"]) is False
    assert should_start_searxng(["dsh", "pi"]) is False

    # 2. Other harnesses -> SearXNG SHOULD be started
    assert should_start_searxng(["claude"]) is True
    assert should_start_searxng(["hermes"]) is True
    assert should_start_searxng(["orca"]) is True
    assert should_start_searxng(["t3"]) is True
    assert should_start_searxng(["pi", "claude"]) is True
    assert should_start_searxng(["dsh", "hermes"]) is True

    # 3. Empty list -> No harness running, SearXNG should NOT be started
    assert should_start_searxng([]) is False

    # 4. Fallback to config.choose_harness when selected_harnesses is None
    monkeypatch.setattr(config, "choose_harness", ["pi"])
    assert should_start_searxng(None) is False

    monkeypatch.setattr(config, "choose_harness", ["pi", "dsh"])
    assert should_start_searxng(None) is False

    monkeypatch.setattr(config, "choose_harness", ["claude"])
    assert should_start_searxng(None) is True

    # 5. When DonSeTch is disabled for Pi or DSH -> SearXNG is started as fallback
    monkeypatch.setattr(config, "pi_donsetch_enabled", False)
    assert should_start_searxng(["pi"]) is True
    monkeypatch.setattr(config, "pi_donsetch_enabled", True)

    monkeypatch.setattr(config, "deepseek_donsetch_enabled", False)
    assert should_start_searxng(["dsh"]) is True
    assert should_start_searxng(["deepseek"]) is True
    monkeypatch.setattr(config, "deepseek_donsetch_enabled", True)

    # 6. Global toggles (searxng_enabled=False or searxng_mode='disabled')
    monkeypatch.setattr(config, "searxng_enabled", False)
    assert should_start_searxng(["claude"]) is False
    monkeypatch.setattr(config, "searxng_enabled", True)

    monkeypatch.setattr(config, "searxng_mode", "disabled")
    assert should_start_searxng(["claude"]) is False


def test_start_searxng_service_bypassed_for_pi_and_dsh(monkeypatch):
    """Verify start_searxng_service skips start_searxng_server when only pi/dsh are active."""
    from local_ai_launcher.config import config
    from local_ai_launcher.__main__ import start_searxng_service
    import local_ai_launcher.backends.searxng as searxng_mod

    started_calls = []
    monkeypatch.setattr(searxng_mod, "start_searxng_server", lambda procs, **kw: started_calls.append(procs))
    monkeypatch.setattr(config, "searxng_enabled", True)
    monkeypatch.setattr(config, "searxng_mode", "wsl_docker")
    monkeypatch.setattr(config, "pi_donsetch_enabled", True)
    monkeypatch.setattr(config, "deepseek_donsetch_enabled", True)

    # Pi only -> bypassed
    start_searxng_service([], selected_harnesses=["pi"])
    assert len(started_calls) == 0

    # DSH only -> bypassed
    start_searxng_service([], selected_harnesses=["dsh"])
    assert len(started_calls) == 0

    # Both Pi and DSH -> bypassed
    start_searxng_service([], selected_harnesses=["pi", "dsh"])
    assert len(started_calls) == 0

    # Claude -> started
    start_searxng_service([], selected_harnesses=["claude"])
    assert len(started_calls) == 1

    # Pi and Claude -> started
    start_searxng_service([], selected_harnesses=["pi", "claude"])
    assert len(started_calls) == 2


def test_build_ps_env_searxng_omission_for_donsetch():
    """Verify build_ps_env omits SEARXNG env vars for Pi/DSH but includes them for other harnesses."""
    from local_ai_launcher.process import build_ps_env
    from local_ai_launcher.config import config

    # For Pi with DonSeTch -> SEARXNG_URL should NOT be present
    pi_env = build_ps_env({}, harness_name="pi")
    assert "SEARXNG_URL" not in pi_env
    assert "SEARXNG_BASE_URL" not in pi_env

    # For DeepSeek / DSH with DonSeTch -> SEARXNG_URL should NOT be present
    dsh_env = build_ps_env({}, harness_name="dsh")
    assert "SEARXNG_URL" not in dsh_env
    assert "SEARXNG_BASE_URL" not in dsh_env

    ds_env = build_ps_env({}, harness_name="deepseek")
    assert "SEARXNG_URL" not in ds_env
    assert "SEARXNG_BASE_URL" not in ds_env

    # When DONSETCH_BINARY is present in env_vars -> SEARXNG_URL should NOT be present
    explicit_don = build_ps_env({"DONSETCH_BINARY": "C:/custom/donsetch.exe"})
    assert "SEARXNG_URL" not in explicit_don

    # For Claude -> SEARXNG_URL SHOULD be present
    claude_env = build_ps_env({}, harness_name="claude")
    assert "SEARXNG_URL" in claude_env
    assert "SEARXNG_BASE_URL" in claude_env

    # For Hermes -> SEARXNG_URL SHOULD be present
    hermes_env = build_ps_env({}, harness_name="hermes")
    assert "SEARXNG_URL" in hermes_env
    assert "SEARXNG_BASE_URL" in hermes_env


def test_pi_web_search_config_donsetch_omission(tmp_path, monkeypatch):
    """Verify setup_pi_web_search_config removes searxngBaseUrl when DonSeTch is active."""
    import json
    from local_ai_launcher.harnesses.pi import setup_pi_web_search_config
    from local_ai_launcher.config import config

    # Pre-populate web-search.json with a stale searxngBaseUrl
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir(parents=True, exist_ok=True)
    ws_file = agent_dir / "web-search.json"
    ws_file.write_text(json.dumps({"searxngBaseUrl": "http://127.0.0.1:8088"}), encoding="utf-8")

    # 1. When DonSeTch is enabled -> searxngBaseUrl must be removed
    monkeypatch.setattr(config, "pi_donsetch_enabled", True)
    setup_pi_web_search_config(tmp_path)
    data = json.loads(ws_file.read_text(encoding="utf-8"))
    assert "searxngBaseUrl" not in data

    # 2. When DonSeTch is disabled -> searxngBaseUrl must be populated
    monkeypatch.setattr(config, "pi_donsetch_enabled", False)
    monkeypatch.setattr(config, "searxng_enabled", True)
    monkeypatch.setattr(config, "searxng_mode", "wsl_docker")
    setup_pi_web_search_config(tmp_path)
    data2 = json.loads(ws_file.read_text(encoding="utf-8"))
    assert "searxngBaseUrl" in data2
    assert data2["searxngBaseUrl"] == "http://127.0.0.1:8088"


def test_deepseek_searxng_omitted_when_donsetch_enabled(tmp_path, monkeypatch):
    """Verify DeepseekAdapter does not inject SEARXNG env vars when DonSeTch is active, only on fallback."""
    from unittest.mock import patch
    from local_ai_launcher.harnesses.deepseek import DeepseekAdapter
    import local_ai_launcher.harnesses.deepseek as ds_mod
    from local_ai_launcher.config import config

    dsh_home = tmp_path / ".dsh"
    dsh_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("DSH_HOME", str(dsh_home))
    monkeypatch.setattr(ds_mod, "AI_WORKSPACE_DIR", tmp_path)
    monkeypatch.setattr(config, "harness_start_dir", tmp_path)
    monkeypatch.setattr(config, "searxng_enabled", True)
    monkeypatch.setattr(config, "searxng_mode", "wsl_docker")

    adapter = DeepseekAdapter(tmp_path, [])
    launched_cmd = []

    def fake_start_wt(cmd, **kwargs):
        launched_cmd.append(cmd)
        return 12345

    # 1. When deepseek_donsetch_enabled is True -> no SEARXNG env vars
    monkeypatch.setattr(config, "deepseek_donsetch_enabled", True)
    with patch("local_ai_launcher.context_preloader.preload_harness_context"), \
         patch("local_ai_launcher.harnesses.deepseek.start_in_wt_tab", side_effect=fake_start_wt), \
         patch("local_ai_launcher.skills_manager.SkillManager.ensure_scoped_harness_skills"):
        adapter.start("http://127.0.0.1:8080/v1", "test-model")

    assert len(launched_cmd) == 1
    assert "SEARXNG_URL" not in launched_cmd[0]
    assert "SEARXNG_BASE_URL" not in launched_cmd[0]

    # 2. When deepseek_donsetch_enabled is False -> fallback to SEARXNG
    monkeypatch.setattr(config, "deepseek_donsetch_enabled", False)
    launched_cmd.clear()
    with patch("local_ai_launcher.context_preloader.preload_harness_context"), \
         patch("local_ai_launcher.harnesses.deepseek.start_in_wt_tab", side_effect=fake_start_wt), \
         patch("local_ai_launcher.skills_manager.SkillManager.ensure_scoped_harness_skills"):
        adapter.start("http://127.0.0.1:8080/v1", "test-model")

    assert len(launched_cmd) == 1
    assert "SEARXNG_URL" in launched_cmd[0]
    assert "SEARXNG_BASE_URL" in launched_cmd[0]


