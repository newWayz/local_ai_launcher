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

