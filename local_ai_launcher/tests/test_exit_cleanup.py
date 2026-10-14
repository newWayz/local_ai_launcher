import sys
import subprocess
import time
from unittest.mock import patch, MagicMock

import pytest

from local_ai_launcher.process import (
    assign_process_to_job,
    kill_process_tree,
    kill_all_launcher_processes,
    _tracked_spawned_processes,
    force_kill_port,
    get_or_create_job_object,
    purge_stale_orca_locks,
    kill_workspace_orca_processes
)
from local_ai_launcher.localai.native_server import BackendSupervisor
from local_ai_launcher.shims.blackwell_llama_shim import _cleanup_llama_on_exit
import local_ai_launcher.shims.blackwell_llama_shim as llama_shim_mod
import local_ai_launcher.shims.huang_llama_shim as huang_shim_mod
import local_ai_launcher.shims.bruno_llama_shim as bruno_shim_mod
from local_ai_launcher.shims.tabby_grpc_shim import _cleanup_tabby_on_exit
import local_ai_launcher.shims.tabby_grpc_shim as tabby_shim_mod
from local_ai_launcher.shims.freetoken_grpc_shim import _cleanup_freetoken_on_exit
import local_ai_launcher.shims.freetoken_grpc_shim as ft_shim_mod


def test_assign_process_to_job():
    """Verify assign_process_to_job adds process to tracking and invokes Win32 Job Object."""
    mock_proc = MagicMock()
    mock_proc.pid = 12345
    
    initial_len = len(_tracked_spawned_processes)
    assign_process_to_job(mock_proc)
    assert len(_tracked_spawned_processes) == initial_len + 1
    assert _tracked_spawned_processes[-1] == mock_proc
    _tracked_spawned_processes.pop()


def test_kill_process_tree_live_process():
    """Verify kill_process_tree terminates a live spawned subprocess."""
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
    )
    assert proc.poll() is None
    kill_process_tree(proc)
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()
    assert proc.poll() is not None


def test_kill_all_launcher_processes():
    """Verify kill_all_launcher_processes purges tracked processes, ports, and binaries."""
    orig_tracked = list(_tracked_spawned_processes)
    try:
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        _tracked_spawned_processes.clear()
        _tracked_spawned_processes.append(mock_proc)

        killed_ports = []
        with patch("local_ai_launcher.process.kill_process_tree") as mock_kill_tree, \
             patch("local_ai_launcher.process.force_kill_port", side_effect=lambda p, allowed=None: killed_ports.append(p)), \
             patch("subprocess.run") as mock_subproc:
            kill_all_launcher_processes()
            
            mock_kill_tree.assert_called_with(mock_proc)
            assert len(_tracked_spawned_processes) == 0
            # Check all critical ports are cleaned up
            assert 8080 in killed_ports  # LocalAI
            assert 9931 in killed_ports  # llama-server direct
            assert 9940 in killed_ports  # Blackwell shim
            assert 9941 in killed_ports  # Tabby shim
            assert 9942 in killed_ports  # FreeToken shim
            assert 9932 in killed_ports  # MCP server
            assert 20128 in killed_ports # OmniRoute
            assert 8000 in killed_ports  # TabbyAPI direct
            assert 9943 in killed_ports  # FreeToken direct
            assert 8081 in killed_ports  # CPU Decision llama-server
            assert 8082 in killed_ports  # CPU Reranker microservice
            assert 8083 in killed_ports  # CPU ONNX Decision Gater
            assert 8085 in killed_ports  # Hermes Signal CLI Daemon
            assert 3773 in killed_ports  # T3 Code server
            assert 8088 in killed_ports  # SearXNG Metasearch service
    finally:
        _tracked_spawned_processes.clear()
        _tracked_spawned_processes.extend(orig_tracked)


def test_native_server_shutdown_active_backend_sync():
    """Verify BackendSupervisor.shutdown_active_backend_sync cleanly terminates active shim."""
    sup = BackendSupervisor()
    mock_stub = MagicMock()
    mock_shim = MagicMock()
    mock_shim.poll.return_value = None

    sup.grpc_stub = mock_stub
    sup.active_shim_proc = mock_shim
    sup.active_backend_name = "blackwell-llama"
    sup.active_model_name = "qwen-38_27b"

    with patch("local_ai_launcher.process.kill_process_tree") as mock_kill:
        sup.shutdown_active_backend_sync()
        mock_stub.Free.assert_called_once()
        mock_kill.assert_called_once_with(mock_shim)
        assert sup.active_shim_proc is None
        assert sup.active_backend_name is None
        assert sup.active_model_name is None


def test_shim_cleanup_on_exit(monkeypatch):
    """Verify shim module exit hooks kill active engine subprocess."""
    mock_servicer = MagicMock()
    mock_engine = MagicMock()
    mock_engine.poll.return_value = None
    mock_servicer.process = mock_engine

    # Test Blackwell llama shim
    monkeypatch.setattr(llama_shim_mod, "_active_servicer", mock_servicer)
    with patch("local_ai_launcher.process.kill_process_tree") as mock_kill:
        _cleanup_llama_on_exit()
        mock_kill.assert_called_once_with(mock_engine)

    # Test Tabby shim
    monkeypatch.setattr(tabby_shim_mod, "_active_servicer", mock_servicer)
    with patch("local_ai_launcher.process.kill_process_tree") as mock_kill:
        _cleanup_tabby_on_exit()
        mock_kill.assert_called_once_with(mock_engine)

    # Test FreeToken shim
    monkeypatch.setattr(ft_shim_mod, "_active_servicer", mock_servicer)
    with patch("local_ai_launcher.process.kill_process_tree") as mock_kill:
        _cleanup_freetoken_on_exit()
        mock_kill.assert_called_once_with(mock_engine)

    # Test Huang llama shim
    monkeypatch.setattr(huang_shim_mod, "_active_servicer", mock_servicer)
    with patch("local_ai_launcher.process.kill_process_tree") as mock_kill:
        huang_shim_mod._cleanup_llama_on_exit()
        mock_kill.assert_called_once_with(mock_engine)

    # Test Bruno llama shim
    monkeypatch.setattr(bruno_shim_mod, "_active_servicer", mock_servicer)
    with patch("local_ai_launcher.process.kill_process_tree") as mock_kill:
        bruno_shim_mod._cleanup_llama_on_exit()
        mock_kill.assert_called_once_with(mock_engine)


def test_purge_stale_orca_locks(tmp_path):
    """Verify purge_stale_orca_locks removes dead PID files while preserving active ones."""
    import os
    import json
    
    daemon_dir = tmp_path / "daemon"
    daemon_dir.mkdir(parents=True, exist_ok=True)
    
    # Stale files (dead PID 99999999)
    stale_pid_file = daemon_dir / "daemon-v36.pid"
    stale_token_file = daemon_dir / "daemon-v36.token"
    stale_pid_file.write_text(json.dumps({"pid": 99999999}), encoding="utf-8")
    stale_token_file.write_text("dummy-token", encoding="utf-8")

    stale_runtime_file = tmp_path / "orca-runtime.json"
    stale_runtime_file.write_text(json.dumps({"pid": 99999999}), encoding="utf-8")

    purged = purge_stale_orca_locks(tmp_path)
    assert len(purged) == 3
    assert not stale_pid_file.exists()
    assert not stale_token_file.exists()
    assert not stale_runtime_file.exists()

    # Active file (current process PID)
    live_pid_file = daemon_dir / "daemon-v36.pid"
    live_pid_file.write_text(json.dumps({"pid": os.getpid()}), encoding="utf-8")
    
    purged_active = purge_stale_orca_locks(tmp_path)
    assert len(purged_active) == 0
    assert live_pid_file.exists()


def test_kill_workspace_orca_processes():
    """Verify kill_workspace_orca_processes targets only workspace-local or daemon-host processes."""
    workspace_orca = MagicMock()
    workspace_orca.info = {
        'name': 'Orca.exe',
        'exe': r'C:\aiWorkspace\orca\Orca.exe',
        'cmdline': [r'C:\aiWorkspace\orca\Orca.exe']
    }

    daemon_host_orca = MagicMock()
    daemon_host_orca.info = {
        'name': 'Orca.exe',
        'exe': r'C:\Users\user\AppData\Local\Orca\daemon-host\1.4.209\Orca.exe',
        'cmdline': [r'C:\Users\user\AppData\Local\Orca\daemon-host\1.4.209\Orca.exe']
    }

    external_orca = MagicMock()
    external_orca.info = {
        'name': 'Orca.exe',
        'exe': r'D:\OtherProjects\SomeApp\Orca.exe',
        'cmdline': [r'D:\OtherProjects\SomeApp\Orca.exe']
    }

    with patch("psutil.process_iter", return_value=[workspace_orca, daemon_host_orca, external_orca]), \
         patch("local_ai_launcher.process.kill_process_tree") as mock_kill:
        kill_workspace_orca_processes()
        assert mock_kill.call_count == 2
        mock_kill.assert_any_call(workspace_orca)
        mock_kill.assert_any_call(daemon_host_orca)


def test_force_kill_port_netstat_allowed_names():
    """Verify that netstat fallback does not kill processes outside allowed_names."""
    from local_ai_launcher.process import force_kill_port

    with patch("subprocess.check_output") as mock_output, \
         patch("local_ai_launcher.process.kill_process_tree") as mock_kill, \
         patch("psutil.Process") as mock_psutil_proc, \
         patch("sys.platform", "win32"):

        # Mock netstat output showing port 9999 is held by PID 12345
        mock_output.return_value = "  TCP    127.0.0.1:9999         0.0.0.0:0              LISTENING       12345\n".encode("utf-8")

        # Scenario A: PID 12345 is an unrelated process (e.g. spotify.exe)
        proc_instance_evil = MagicMock()
        proc_instance_evil.name.return_value = "spotify.exe"
        mock_psutil_proc.return_value = proc_instance_evil

        force_kill_port(9999, allowed_names={"llama-server", "python"})
        assert not mock_kill.called

        # Scenario B: PID 12345 is an allowed process (e.g. llama-server.exe)
        proc_instance_good = MagicMock()
        proc_instance_good.name.return_value = "llama-server.exe"
        mock_psutil_proc.return_value = proc_instance_good

        force_kill_port(9999, allowed_names={"llama-server", "python"})
        assert mock_kill.called
        assert mock_kill.call_args[0][0] == 12345


def test_wait_for_port():
    """Verify wait_for_port successfully detects open port and times out appropriately."""
    from local_ai_launcher.process import wait_for_port

    with patch("local_ai_launcher.process.is_port_in_use") as mock_in_use:
        # Case 1: Port becomes available on second poll
        mock_in_use.side_effect = [False, True]
        assert wait_for_port(8080, timeout=1.0) is True

        # Case 2: Port never opens
        mock_in_use.side_effect = None
        mock_in_use.return_value = False
        assert wait_for_port(8080, timeout=0.5) is False


def test_send_background_warmup():
    """Verify send_background_warmup dispatches HTTP request via urllib in background."""
    from local_ai_launcher.harness_base import send_background_warmup

    with patch("urllib.request.urlopen") as mock_urlopen, \
         patch("threading.Thread") as mock_thread, \
         patch("time.sleep"):
        mock_resp = MagicMock()
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        def _run_inline(target=None, daemon=True):
            mock_inst = MagicMock()
            mock_inst.start.side_effect = target
            return mock_inst

        mock_thread.side_effect = _run_inline

        send_background_warmup("http://127.0.0.1:8080/v1", "test-model", "test prompt")

        assert mock_urlopen.called
        req = mock_urlopen.call_args[0][0]
        assert req.full_url == "http://127.0.0.1:8080/v1/chat/completions"
        assert req.get_header("Content-type") == "application/json"


