import os
import sys
import pytest
from pathlib import Path

from local_ai_launcher.environment import (
    ensure_directory_junction,
    setup_windows_environment_variables,
    safe_extract_zip,
)
from local_ai_launcher.backends.parakeet_npu import DEFAULT_MODEL_DIR


def test_ensure_directory_junction_recovery(tmp_path, monkeypatch):
    """Verifies ensure_directory_junction migrates files from normal dirs and triggers mklink."""
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

    target_dir = tmp_path / "workspace_target"
    link_dir = tmp_path / "user_link"
    link_dir.mkdir(parents=True)
    (link_dir / "keybindings.json").write_text('{"test": true}', encoding="utf-8")

    commands = []
    def mock_run(cmd, *args, **kwargs):
        commands.append(cmd)
        class Res:
            returncode = 0
        return Res()
    monkeypatch.setattr("subprocess.run", mock_run)

    success = ensure_directory_junction(link_dir, target_dir)
    assert success is True
    assert (target_dir / "keybindings.json").exists()
    assert (tmp_path / "user_link_pre_junction_bak" / "keybindings.json").exists()
    assert any("mklink /J" in str(c) for c in commands)


def test_setup_windows_environment_variables_sets_home(monkeypatch):
    """Verifies setup_windows_environment_variables puts HOME, PI_HOME, HF_HOME into os.environ."""
    if sys.platform != 'win32':
        pytest.skip("Windows-only test")

    # Capture PowerShell calls without running them
    ps_calls = []
    def mock_run(*args, **kwargs):
        cmd_str = args[0] if args else kwargs.get("args", "")
        if isinstance(cmd_str, list):
            cmd_str = " ".join(cmd_str)
        ps_calls.append(cmd_str)
        class MockRes:
            returncode = 0
            stdout = ""
            stderr = ""
        return MockRes()
    monkeypatch.setattr("subprocess.run", mock_run)

    orig_env = dict(os.environ)
    try:
        setup_windows_environment_variables()

        assert "HOME" in os.environ
        assert "Users" not in os.environ["HOME"]
        assert "PI_HOME" in os.environ
        assert "HF_HOME" in os.environ
        assert "HF_HUB_CACHE" in os.environ
        assert "GEMINI_DIR" in os.environ
        assert "Users" not in os.environ["GEMINI_DIR"]
    finally:
        os.environ.clear()
        os.environ.update(orig_env)


def test_antigravity_directory_junctions(tmp_path, monkeypatch):
    """Verifies Antigravity harness ensures directory junctions for cli and config."""
    from local_ai_launcher.harnesses.antigravity import AntigravityAdapter

    junction_calls = []
    def mock_junction(link, target):
        junction_calls.append((Path(link), Path(target)))
        return True
    monkeypatch.setattr("local_ai_launcher.harnesses.antigravity.ensure_directory_junction", mock_junction)

    adapter = AntigravityAdapter("test_root", [])
    adapter._configure_settings_and_rules()

    assert any("antigravity-cli" in str(link) for link, target in junction_calls)
    assert any("config" in str(link) for link, target in junction_calls)


def test_parakeet_model_dir_resolution():
    """Verifies Parakeet OpenVINO model directory resolves with expected naming."""
    assert "openvino" in str(DEFAULT_MODEL_DIR).lower()
    assert DEFAULT_MODEL_DIR.name == "parakeet-tdt-v3-ov"
    assert isinstance(DEFAULT_MODEL_DIR, Path)


def test_safe_extract_zip_security(tmp_path):
    """Verify safe_extract_zip allows valid members and prevents zip-slip prefix collisions."""
    import io
    import zipfile

    target_path = tmp_path / "extract_target"
    target_path.mkdir(parents=True, exist_ok=True)

    # 1. Valid zip archive
    valid_bio = io.BytesIO()
    with zipfile.ZipFile(valid_bio, "w") as zf:
        zf.writestr("subdir/valid.txt", "hello world")
    valid_bio.seek(0)

    with zipfile.ZipFile(valid_bio, "r") as zf:
        safe_extract_zip(zf, target_path)

    assert (target_path / "subdir" / "valid.txt").exists()
    assert (target_path / "subdir" / "valid.txt").read_text(encoding="utf-8") == "hello world"

    # 2. Path traversal attack (prefix collision or relative escape)
    evil_bio = io.BytesIO()
    with zipfile.ZipFile(evil_bio, "w") as zf:
        zf.writestr("../escaped.txt", "malicious payload")
    evil_bio.seek(0)

    with zipfile.ZipFile(evil_bio, "r") as zf:
        with pytest.raises(RuntimeError, match="Zip slip attempt detected"):
            safe_extract_zip(zf, target_path)


def test_heal_workspace_bin_trampolines(tmp_path, monkeypatch):
    """Verifies heal_workspace_bin_trampolines detects and purges broken uv trampolines while preserving healthy binaries."""
    from local_ai_launcher.environment import heal_workspace_bin_trampolines
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)

    broken_exe = bin_dir / "broken_tool.exe"
    broken_exe.write_bytes(b"uv trampoline canonicalize stub")

    healthy_exe = bin_dir / "healthy_tool.exe"
    healthy_exe.write_bytes(b"uv trampoline canonicalize stub")

    non_uv_exe = bin_dir / "non_uv_tool.exe"
    non_uv_exe.write_bytes(b"standard generic executable content")

    spawned_cmds = []

    def mock_run(cmd, *args, **kwargs):
        spawned_cmds.append(cmd)
        class Res:
            pass
        res = Res()
        if "broken_tool" in str(cmd[0]):
            res.returncode = 1
            res.stdout = ""
            res.stderr = "error: uv trampoline failed to canonicalize script path\n"
        else:
            res.returncode = 0
            res.stdout = "tool 1.0.0"
            res.stderr = ""
        return res

    monkeypatch.setattr("subprocess.run", mock_run)

    healed = heal_workspace_bin_trampolines([str(bin_dir)])

    assert broken_exe not in bin_dir.glob("*.exe")
    assert healthy_exe in bin_dir.glob("*.exe")
    assert non_uv_exe in bin_dir.glob("*.exe")
    assert broken_exe in healed
    assert healthy_exe not in healed
    # Verify non_uv_exe was never even executed via subprocess
    assert not any("non_uv_tool" in str(c[0]) for c in spawned_cmds)


def test_git_version_helpers(monkeypatch):
    """Verify get_local_git_version and get_remote_git_version parsing."""
    from local_ai_launcher.environment import get_local_git_version, get_remote_git_version

    class MockRes:
        returncode = 0
        stdout = "git version 2.56.0.windows.2\n"
        stderr = ""

    monkeypatch.setattr("shutil.which", lambda cmd: "C:\\Program Files\\Git\\cmd\\git.exe" if cmd == "git" else None)
    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: MockRes())

    ver = get_local_git_version()
    assert ver == "2.56.0.windows.2"

    class MockResp:
        def read(self):
            return b"v2.56.0.windows.2\n"
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    monkeypatch.setattr("urllib.request.urlopen", lambda req, **kwargs: MockResp())
    rver = get_remote_git_version()
    if sys.platform == "win32":
        assert rver == "2.56.0.windows.2"


def test_check_git_lifecycle(monkeypatch):
    """Verify check_git respecting never_update, auto_update, and update execution."""
    from local_ai_launcher.environment import check_git
    from local_ai_launcher.config import config

    # 1. When git is missing
    monkeypatch.setattr("local_ai_launcher.environment.get_local_git_version", lambda: None)
    assert check_git() is False

    # 2. When git exists but behavior is never_update and auto_update is False
    monkeypatch.setattr("local_ai_launcher.environment.get_local_git_version", lambda: "2.55.0.windows.5")
    monkeypatch.setattr(config, "git_update_behavior", "never_update")
    monkeypatch.setattr(config, "git_auto_update", False)
    monkeypatch.setattr(config, "force_update", False)
    assert check_git() is True

    # 3. When auto_update is True and already up to date
    monkeypatch.setattr(config, "git_auto_update", True)
    monkeypatch.setattr(config, "git_update_behavior", "always_update")
    monkeypatch.setattr("local_ai_launcher.environment.get_remote_git_version", lambda: "2.55.0.windows.5")
    assert check_git() is True

    # 4. When auto_update is True and update is needed
    monkeypatch.setattr(config, "force_update", True)
    monkeypatch.setattr("local_ai_launcher.environment.get_remote_git_version", lambda: "2.56.0.windows.2")
    executed_cmds = []
    class MockRes:
        returncode = 2
        stdout = "started"
        stderr = ""
    def mock_run(cmd, *args, **kwargs):
        executed_cmds.append(cmd)
        return MockRes()
    monkeypatch.setattr("subprocess.run", mock_run)

    assert check_git() is True
    if sys.platform == "win32":
        assert any("update-git-for-windows" in " ".join(c) for c in executed_cmds)


