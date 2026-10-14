import os
import sys
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from local_ai_launcher.config import config, AI_WORKSPACE_DIR, LOCALAI_REPO_DIR
from local_ai_launcher.backends.localai import check_localai, _sync_web_assets


def test_localai_config_defaults():
    """Verify default LocalAI repository and install/update configuration."""
    assert config.localai_repo_url == "https://github.com/mudler/LocalAI.git"
    assert config.localai_install_behavior in ("existing", "always_install", "never_install")
    assert config.localai_update_behavior in ("never_update", "always_update", "existing")
    assert hasattr(config, "localai_repo_dir")
    assert hasattr(config, "localai_bin_dir")
    assert hasattr(config, "localai_exe")


def test_check_localai_never_install():
    """Verify never_install skips git clone."""
    with patch.object(config, "localai_install_behavior", "never_install"), \
         patch.object(Path, "exists", return_value=False), \
         patch("subprocess.run") as mock_run:
        res = check_localai()
        assert res is True
        mock_run.assert_not_called()


def test_check_localai_existing_portable_assets():
    """Verify existing portable assets allow running without re-cloning."""
    with patch.object(config, "localai_install_behavior", "existing"), \
         patch.object(Path, "exists", return_value=True), \
         patch("subprocess.run") as mock_run:
        res = check_localai()
        assert res is True
        mock_run.assert_not_called()


def test_check_localai_always_install():
    """Verify always_install triggers git clone of LocalAI repo."""
    def mock_exists(self=None, *args):
        if self and "localai_repo" in str(self):
            return False
        return True

    with patch.object(config, "localai_install_behavior", "always_install"), \
         patch.object(Path, "exists", mock_exists), \
         patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        res = check_localai()
        assert res is True
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert call_args[0] == "git"
        assert call_args[1] == "clone"
        assert "https://github.com/mudler/LocalAI.git" in call_args


def test_check_localai_always_update():
    """Verify always_update triggers git pull on existing repo."""
    def mock_exists(self=None, *args):
        return True

    with patch.object(config, "localai_update_behavior", "always_update"), \
         patch.object(config, "update_frequency_hrs", 0), \
         patch.object(config, "engine_update_frequency_hrs", {}), \
         patch.object(Path, "exists", mock_exists), \
         patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="Already up to date.")
        res = check_localai()
        assert res is True
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert call_args[0] == "git"
        assert call_args[1] == "pull"


def test_sync_web_assets(tmp_path):
    """Verify _sync_web_assets copies files recursively."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "test.html").write_text("hello", encoding="utf-8")
    sub = src / "sub"
    sub.mkdir()
    (sub / "app.js").write_text("console.log(1)", encoding="utf-8")

    dst = tmp_path / "dst"
    _sync_web_assets(src, dst)

    assert (dst / "test.html").read_text(encoding="utf-8") == "hello"
    assert (dst / "sub" / "app.js").read_text(encoding="utf-8") == "console.log(1)"
