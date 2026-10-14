import os
import sys
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from local_ai_launcher.localai.native_server import app, WEB_DIR, STATIC_DIR
from local_ai_launcher.process import start_in_wt_tab, AI_WORKSPACE_DIR
import hashlib
from unittest.mock import MagicMock, patch


def test_web_gui_static_and_html_served():
    client = TestClient(app)
    
    # Test Root GUI
    res_root = client.get("/")
    assert res_root.status_code == 200
    assert "LocalAI Unified Cockpit" in res_root.text
    assert "text/html" in res_root.headers.get("content-type", "")

    # Test Alternate Routes
    for route in ("/chat", "/models", "/backends", "/app"):
        res = client.get(route)
        assert res.status_code == 200
        assert "LocalAI Unified Cockpit" in res.text

    # Test Static Asset (Favicon or Logo)
    res_logo = client.get("/static/logo_horizontal.png")
    assert res_logo.status_code == 200

    # Test API VRAM
    res_vram = client.get("/api/vram")
    assert res_vram.status_code == 200
    vram_data = res_vram.json()
    assert "total_mb" in vram_data
    assert "free_mb" in vram_data
    assert vram_data.get("single_active_backend") is True

    # Test API Models Details
    res_models = client.get("/api/models/details")
    assert res_models.status_code == 200
    models_data = res_models.json()
    assert "models" in models_data
    assert len(models_data["models"]) > 0


def test_terminal_tabs_window_name_and_command(monkeypatch):
    workspace_id = hashlib.md5(str(AI_WORKSPACE_DIR).encode('utf-8')).hexdigest()[:8]
    expected_win_name = f"AiLauncher_{workspace_id}"
    assert expected_win_name == f"AiLauncher_{workspace_id}"
    assert expected_win_name.startswith("AiLauncher_") and len(expected_win_name) == 19
    
    # Ensure environment or hash targets the exact workspace window
    monkeypatch.setenv("AI_LAUNCHER_WINDOW_NAME", expected_win_name)
    monkeypatch.setenv("AI_LAUNCHER_UNIFIED", "1")
    
    def _mock_popen(cmd, *args, **kwargs):
        cmd_str = str(cmd)
        import re, base64
        m = re.search(r'-EncodedCommand\s+([A-Za-z0-9+/=]+)', cmd_str)
        if m:
            try:
                decoded = base64.b64decode(m.group(1)).decode('utf-16le', errors='ignore')
                m_file = re.search(r"-FilePath '([^']+)'", decoded)
                if m_file:
                    Path(m_file.group(1)).write_text("99999\n", encoding="ascii")
            except Exception:
                pass
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.wait.return_value = 0
        return mock_proc

    from unittest.mock import patch
    with patch("subprocess.Popen", side_effect=_mock_popen) as mock_popen, patch("shutil.which", return_value="C:\\wt.exe"), patch("local_ai_launcher.process.is_interactive_desktop", return_value=True):
        start_in_wt_tab("echo hello", title="TestTab")
        
        assert mock_popen.called
        call_args = mock_popen.call_args[0][0]
        assert f'-w "{expected_win_name}" nt' in call_args
        assert '--title "TestTab"' in call_args
        assert '-EncodedCommand' in call_args


def test_start_in_wt_tab_includes_directory_flag():
    """Verifies start_in_wt_tab includes -d flag so terminal opens in workspace dir."""
    def _mock_popen(cmd, *args, **kwargs):
        cmd_str = str(cmd)
        import re, base64
        m = re.search(r'-EncodedCommand\s+([A-Za-z0-9+/=]+)', cmd_str)
        if m:
            try:
                decoded = base64.b64decode(m.group(1)).decode('utf-16le', errors='ignore')
                m_file = re.search(r"-FilePath '([^']+)'", decoded)
                if m_file:
                    Path(m_file.group(1)).write_text("99999\n", encoding="ascii")
            except Exception:
                pass
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.wait.return_value = 0
        return mock_proc

    from unittest.mock import patch
    with patch("subprocess.Popen", side_effect=_mock_popen) as mock_popen, \
         patch("shutil.which", return_value="C:\\wt.exe"), \
         patch("local_ai_launcher.process.is_interactive_desktop", return_value=True):

        start_in_wt_tab("echo hello", title="DirTest")

        assert mock_popen.called
        call_args = mock_popen.call_args[0][0]
        # The -d flag must be present to prevent WT from defaulting to %USERPROFILE%
        assert '-d "' in call_args or "-d '" in call_args or "-d " in call_args

