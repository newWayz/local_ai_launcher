import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from local_ai_launcher.paths import AI_WORKSPACE_DIR, EXE_PATH_LLAMA_CPP
from local_ai_launcher.backends.ascii_pruner import (
    derive_ascii_model_path,
    is_ascii_model,
    is_huang_model_allowed,
    is_huang_vram_eligible,
    estimate_model_and_mmproj_size_mb,
    get_system_vram_mb,
    resolve_huang_model_selection,
    HUANG_ALLOWED_STEMS,
)
from local_ai_launcher.shims.huang_llama_shim import HuangLlamaServicer


def test_derive_ascii_model_path():
    # Test unsloth path
    p1 = Path("models/llama-server/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ4_XS.gguf")
    ascii_p1 = derive_ascii_model_path(p1)
    assert ascii_p1.name == "Qwen3.8-27B-UD-ASCII-IQ4_XS.gguf"

    # Test huihui path
    p2 = Path("models/llama-server/huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF/Huihui-Qwen3.8-27B-abliterated-UD-IQ4_XS.gguf")
    ascii_p2 = derive_ascii_model_path(p2)
    assert ascii_p2.name == "Huihui-Qwen3.8-27B-abliterated-UD-ASCII-IQ4_XS.gguf"

    # Test generic path
    p3 = Path("models/my_model.gguf")
    ascii_p3 = derive_ascii_model_path(p3)
    assert ascii_p3.name == "my_model-ASCII.gguf"


def test_is_ascii_model():
    assert is_ascii_model("Qwen3.8-27B-UD-ASCII-IQ4_XS.gguf") is True
    assert is_ascii_model("Huihui-Qwen3.8-27B-abliterated-UD-ASCII-IQ4_XS.gguf") is True
    assert is_ascii_model("Qwen3.8-27B-UD-IQ4_XS.gguf") is False
    assert is_ascii_model("Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf") is False


def test_is_huang_model_allowed():
    # 1. ISTA-DASLab IQ3_S (unchanged)
    assert is_huang_model_allowed("C:/aiWorkspace/models/llama-server/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO/Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf") is True
    # 2. Unsloth ASCII IQ4_XS
    assert is_huang_model_allowed("C:/aiWorkspace/models/llama-server/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-ASCII-IQ4_XS.gguf") is True
    # 3. Huihui ASCII IQ4_XS
    assert is_huang_model_allowed("C:/aiWorkspace/models/llama-server/huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF/Huihui-Qwen3.8-27B-abliterated-UD-ASCII-IQ4_XS.gguf") is True
    # Base models that auto-resolve to ASCII
    assert is_huang_model_allowed("Qwen3.8-27B-UD-IQ4_XS.gguf") is True
    assert is_huang_model_allowed("Huihui-Qwen3.8-27B-abliterated-UD-IQ4_XS.gguf") is True
    # Allowed presets
    assert is_huang_model_allowed("qwen-38_27b-huang-gsq-iq3s-think") is True
    assert is_huang_model_allowed("qwen-38_27b-blackwell-256k-think") is True

    # Disallowed models
    assert is_huang_model_allowed("gemma-4-12b") is False
    assert is_huang_model_allowed("tiel-coder-35b") is False
    assert is_huang_model_allowed("llama-3.3-70b") is False


def test_resolve_huang_model_selection_ista_unchanged():
    name, path = resolve_huang_model_selection(
        "qwen-38_27b-huang-gsq-iq3s-think",
        r"C:\aiWorkspace\models\llama-server\ISTA-DASLab\Qwen3.8-27B-GSQ-RCO\Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf"
    )
    assert "IQ3_S.gguf" in path
    assert "ASCII" not in path


def test_resolve_huang_model_selection_unsloth_diverts_to_ascii():
    name, path = resolve_huang_model_selection(
        "qwen-38_27b-blackwell-256k-think",
        r"C:\aiWorkspace\models\llama-server\unsloth\Qwen3.8-27B-GGUF\Qwen3.8-27B-UD-IQ4_XS.gguf"
    )
    assert name == "qwen-38_27b-huang-ascii-think"
    assert "UD-ASCII-IQ4_XS.gguf" in path


def test_resolve_huang_model_selection_huihui_diverts_to_ascii():
    name, path = resolve_huang_model_selection(
        "qwen-38_27b-blackwell-256k-abli-think",
        r"C:\aiWorkspace\models\llama-server\huihui-ai\Huihui-Qwen3.8-27B-abliterated-GGUF\Huihui-Qwen3.8-27B-abliterated-UD-IQ4_XS.gguf"
    )
    assert name == "qwen-38_27b-huang-abli-ascii-think"
    assert "UD-ASCII-IQ4_XS.gguf" in path


def test_resolve_huang_model_selection_unpermitted_raises():
    with pytest.raises(ValueError, match="not eligible for the Raymond Huang Adaptive-KV runtime"):
        resolve_huang_model_selection("some-unsupported-model", "models/other/model.gguf")


def test_huang_servicer_routes_other_models_to_official_llama_server():
    servicer = HuangLlamaServicer()
    req = MagicMock()
    req.Model = "qwen-38_27b-blackwell-unsupported"
    req.ModelFile = "models/unsupported.gguf"
    req.ContextSize = 0
    req.NGPULayers = 0
    req.Threads = 0
    req.NBatch = 0
    req.FlashAttention = ""
    req.CacheTypeKey = ""
    req.CacheTypeValue = ""
    req.DraftModel = ""
    req.MLock = False
    req.MMap = False
    req.MMProj = ""
    req.EngineArgs = ""
    req.EnvVars = {}

    with patch("subprocess.Popen") as mock_popen, \
         patch.object(servicer, "_wait_for_health", return_value=True), \
         patch("local_ai_launcher.process.assign_process_to_job"), \
         patch("local_ai_launcher.shims.huang_llama_shim.get_active_hardware_profile", return_value={"name": "blackwell_16gb"}), \
         patch("local_ai_launcher.shims.huang_llama_shim.resolve_huang_server_exe", return_value="dummy/llama-server.exe"), \
         patch("os.path.exists", return_value=True):
        mock_proc = MagicMock()
        mock_proc.pid = 40001
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        res = servicer.LoadModel(req, None)
        assert res.success is True
        assert servicer.current_is_blackwell is False
        assert servicer.current_server_exe == str(EXE_PATH_LLAMA_CPP)

        cmd = mock_popen.call_args[0][0]
        assert "--kv-stream-arena-mib" not in cmd


def test_is_huang_vram_eligible_dynamic_criteria(tmp_path):
    # Case 1: Model weight > 62% of 16GB VRAM and fits with 2240 arena -> Eligible!
    # 16,384 * 0.62 = 10,158.08 MiB.
    # Dummy file of 11,000 MiB (11.0 GB)
    model_file = tmp_path / "custom-dense-22b.gguf"
    model_file.write_bytes(b"dummy")

    with patch("os.path.getsize", return_value=11000 * 1024 * 1024):  # 11,000 MiB (67.1% of 16GB)
        eligible, reason = is_huang_vram_eligible(
            str(model_file),
            total_vram_mb=16384.0,
            arena_mib=2240,
            no_mmproj=True
        )
        assert eligible is True
        assert "exceeds 62%" in reason
        assert "fully fit" in reason

    # Case 2: Model weight <= 62% of 16GB VRAM -> Not eligible (Adaptive KV not needed)
    with patch("os.path.getsize", return_value=8000 * 1024 * 1024):  # 8,000 MiB (48.8% of 16GB)
        eligible, reason = is_huang_vram_eligible(
            str(model_file),
            total_vram_mb=16384.0,
            arena_mib=2240,
            no_mmproj=True
        )
        assert eligible is False
        assert "does not exceed 62%" in reason

    # Case 3: Model weight exceeds total GPU VRAM -> Not eligible (does not fit)
    with patch("os.path.getsize", return_value=15000 * 1024 * 1024):  # 15,000 MiB + 2240 arena + 384 = 17,624 > 16,384
        eligible, reason = is_huang_vram_eligible(
            str(model_file),
            total_vram_mb=16384.0,
            arena_mib=2240,
            no_mmproj=True
        )
        assert eligible is False
        assert "exceeds total GPU VRAM" in reason

    # Case 4: Setting ngl=0 (CPU-only) -> Not eligible
    with patch("os.path.getsize", return_value=11000 * 1024 * 1024):
        eligible, reason = is_huang_vram_eligible(
            str(model_file),
            total_vram_mb=16384.0,
            ngl=0
        )
        assert eligible is False
        assert "ngl=0" in reason

    # Case 5: Setting requiring CPU MoE offload -> Not eligible
    with patch("os.path.getsize", return_value=11000 * 1024 * 1024):
        eligible, reason = is_huang_vram_eligible(
            str(model_file),
            total_vram_mb=16384.0,
            n_cpu_moe=25
        )
        assert eligible is False
        assert "CPU MoE" in reason


def test_is_huang_vram_eligible_with_mmproj(tmp_path):
    model_file = tmp_path / "vision-base.gguf"
    model_file.write_bytes(b"model")
    mmproj_file = tmp_path / "mmproj-vision.gguf"
    mmproj_file.write_bytes(b"mmproj")

    # Model is 9,500 MiB (< 62% of 16GB, which is 10,158 MiB)
    # MMProj is 1,200 MiB
    # Combined = 10,700 MiB (> 62% of 16GB)
    def mock_getsize(path):
        if "mmproj" in str(path):
            return 1200 * 1024 * 1024
        return 9500 * 1024 * 1024

    with patch("os.path.getsize", side_effect=mock_getsize):
        # Without mmproj (or with no_mmproj=True), it is 9500 MiB <= 62% -> False
        eligible_no_mm, _ = is_huang_vram_eligible(
            str(model_file),
            mmproj_path=str(mmproj_file),
            no_mmproj=True,
            total_vram_mb=16384.0,
        )
        assert eligible_no_mm is False

        # With mmproj included, 9500 + 1200 = 10,700 MiB > 62% and fits -> True!
        eligible_with_mm, _ = is_huang_vram_eligible(
            str(model_file),
            mmproj_path=str(mmproj_file),
            no_mmproj=False,
            total_vram_mb=16384.0,
        )
        assert eligible_with_mm is True


def test_resolve_huang_model_selection_arbitrary_model_allowed(tmp_path):
    # Verify that there is NO static whitelist: any arbitrary model meeting the dynamic
    # VRAM condition (> 62% VRAM and fitting in VRAM) is accepted and tried out!
    arbitrary_model = tmp_path / "arbitrary-new-architecture-24b.gguf"
    arbitrary_model.write_bytes(b"weights")

    with patch("os.path.getsize", return_value=11500 * 1024 * 1024):  # 11,500 MiB > 62% of 16GB and fits
        name, path = resolve_huang_model_selection("arbitrary-24b", str(arbitrary_model))
        assert name == "arbitrary-24b"
        assert path == str(arbitrary_model)


def test_derive_base_model_filename():
    from local_ai_launcher.backends.ascii_pruner import derive_base_model_filename
    assert derive_base_model_filename("Qwen3.8-27B-UD-ASCII-IQ4_XS.gguf") == "Qwen3.8-27B-UD-IQ4_XS.gguf"
    assert derive_base_model_filename("Huihui-Qwen3.8-27B-abliterated-UD-ASCII-IQ4_XS.gguf") == "Huihui-Qwen3.8-27B-abliterated-UD-IQ4_XS.gguf"
    assert derive_base_model_filename("my-custom-model-ASCII.gguf") == "my-custom-model.gguf"


def test_find_model_section_huang_gsq_iq3s():
    from local_ai_launcher.backends.model_installer import find_model_section
    from local_ai_launcher.presets import load_effective_models_parser
    parser = load_effective_models_parser()
    match = find_model_section(parser, "qwen-38_27b-huang-gsq-iq3s-think")
    assert match is not None
    sec_name, sec_opts = match
    assert "huang-gsq" in sec_name
    assert "ISTA-DASLab" in sec_opts.get("model", "")


def test_ensure_single_model_installed_ascii_triggers_download_and_prune(monkeypatch, tmp_path):
    from local_ai_launcher.backends.model_installer import ensure_single_model_installed
    from local_ai_launcher.config import config

    monkeypatch.setattr(config, "models_install_behavior", "always_install")

    downloaded_files = []
    pruned_pairs = []

    def mock_download(repo_id, filename, target_dir):
        downloaded_files.append((repo_id, filename))
        # Create dummy file so stat().st_size > 0 passes
        dummy_file = target_dir / filename
        dummy_file.parent.mkdir(parents=True, exist_ok=True)
        dummy_file.write_bytes(b"dummy_base_content")
        return True

    def mock_ensure_ascii(base_p, target_p):
        pruned_pairs.append((str(base_p), str(target_p)))
        # Create dummy target file
        target_path = Path(target_p)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(b"dummy_ascii_content")
        return target_path

    # Pre-create chat-template file in tmp_path so it passes
    chat_tmpl = tmp_path / "models" / "peculiar-ragdoll-Qwen-Sharp-Chat-Templates-modded.txt"
    chat_tmpl.parent.mkdir(parents=True, exist_ok=True)
    chat_tmpl.write_text("dummy template")
    (tmp_path / "models" / "chat_template_qwen38_sharp_v22.4.0_medium-convergent.jinja").write_text("dummy template")

    with patch("local_ai_launcher.backends.model_installer.AI_WORKSPACE_DIR", tmp_path), \
         patch("local_ai_launcher.backends.model_installer.download_model_file", side_effect=mock_download), \
         patch("local_ai_launcher.backends.ascii_pruner.ensure_ascii_variant", side_effect=mock_ensure_ascii):
        
        # When target ASCII model doesn't exist, it should download base model and call ensure_ascii_variant
        result = ensure_single_model_installed("qwen-38_27b-huang-ascii-think")
        assert result is True
        assert any("UD-IQ4_XS.gguf" in f[1] and "unsloth" in f[0] for f in downloaded_files)
        assert any("UD-ASCII-IQ4_XS.gguf" in p[1] for p in pruned_pairs)


def test_ensure_single_model_installed_huihui_ascii_triggers_download_and_prune(monkeypatch, tmp_path):
    from local_ai_launcher.backends.model_installer import ensure_single_model_installed
    from local_ai_launcher.config import config

    monkeypatch.setattr(config, "models_install_behavior", "always_install")

    downloaded_files = []
    pruned_pairs = []

    def mock_download(repo_id, filename, target_dir):
        downloaded_files.append((repo_id, filename))
        dummy_file = target_dir / filename
        dummy_file.parent.mkdir(parents=True, exist_ok=True)
        dummy_file.write_bytes(b"dummy_base_content")
        return True

    def mock_ensure_ascii(base_p, target_p):
        pruned_pairs.append((str(base_p), str(target_p)))
        target_path = Path(target_p)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(b"dummy_ascii_content")
        return target_path

    chat_tmpl = tmp_path / "models" / "peculiar-ragdoll-Qwen-Sharp-Chat-Templates-modded.txt"
    chat_tmpl.parent.mkdir(parents=True, exist_ok=True)
    chat_tmpl.write_text("dummy template")
    (tmp_path / "models" / "chat_template_qwen38_sharp_v22.4.0_medium-convergent.jinja").write_text("dummy template")

    with patch("local_ai_launcher.backends.model_installer.AI_WORKSPACE_DIR", tmp_path), \
         patch("local_ai_launcher.backends.model_installer.download_model_file", side_effect=mock_download), \
         patch("local_ai_launcher.backends.ascii_pruner.ensure_ascii_variant", side_effect=mock_ensure_ascii):
        
        result = ensure_single_model_installed("qwen-38_27b-huang-abli-ascii-think")
        assert result is True
        assert any("Huihui-Qwen3.8-27B-abliterated-UD-IQ4_XS.gguf" in f[1] and "huihui-ai" in f[0] for f in downloaded_files)
        assert any("Huihui-Qwen3.8-27B-abliterated-UD-ASCII-IQ4_XS.gguf" in p[1] for p in pruned_pairs)
