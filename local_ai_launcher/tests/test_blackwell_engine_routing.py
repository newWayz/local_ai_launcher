import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from local_ai_launcher.config import (
    config,
    LauncherConfig,
    MODELS_PRESET_FILE,
    EXE_PATH_LLAMA_CPP_BLACKWELL,
    EXE_PATH_LLAMA_CPP_BRUNO,
    EXE_PATH_LLAMA_CPP_HUANG,
    EXE_PATH_LLAMA_CPP,
)
from local_ai_launcher.localai.model_configs import get_model_backend_mapping, generate_all_model_configs
from local_ai_launcher.backends.blackwell_builder import (
    resolve_blackwell_server_exe,
    resolve_blackwell_engine,
    BLACKWELL_RUNTIME_ENV,
)
from local_ai_launcher.backends.bruno_builder import (
    BRUNO_RUNTIME_ENV,
    resolve_bruno_server_exe,
    is_bruno_model_allowed,
    resolve_bruno_kv_cache_types,
)
from local_ai_launcher.backends.huang_builder import resolve_huang_server_exe
from local_ai_launcher.shims.blackwell_llama_shim import BlackwellLlamaServicer
from local_ai_launcher.shims.bruno_llama_shim import BrunoLlamaServicer
from local_ai_launcher.shims.huang_llama_shim import HuangLlamaServicer
from local_ai_launcher.localai.proto import backend_pb2


def test_blackwell_default_engine():
    """Verify that Huang engine is the default blackwell engine."""
    assert config.blackwell_engine == "huang"
    assert resolve_blackwell_engine() == "huang"
    assert config.huang_arena_mib == 2240
    assert config.huang_draft_max == 3
    assert config.huang_spec_type == "draft-mtp"


def test_blackwell_model_mapping(tmp_path):
    """Verify Blackwell and Huang models map to blackwell-llama backend, Strata maps to strata backend, and non-blackwell models map to llama-cpp."""
    generate_all_model_configs(target_dir=tmp_path, force=True)
    mapping = get_model_backend_mapping(models_dir=tmp_path)
    assert "qwen-38_27b-blackwell-256k-think" in mapping
    info = mapping["qwen-38_27b-blackwell-256k-think"]
    assert info["backend"] == "blackwell-llama"
    assert info["data"]["context_size"] == 262000

    # Huang model maps to blackwell-llama
    assert "qwen-38_27b-huang-gsq-think" in mapping
    info_huang = mapping["qwen-38_27b-huang-gsq-think"]
    assert info_huang["backend"] == "blackwell-llama"
    assert info_huang["data"]["context_size"] == 262000

    # Strata model maps to strata
    assert "qwen-38-flash-strata-iq3s" in mapping
    info_strata = mapping["qwen-38-flash-strata-iq3s"]
    assert info_strata["backend"] == "strata"
    assert info_strata["data"]["context_size"] == 262144

    # Non-blackwell models must map to llama-cpp (governed by llamacpp_fork)
    assert mapping["qwen-38_27b-think"]["backend"] == "llama-cpp"
    assert mapping["tiel-coder-35b-think"]["backend"] == "llama-cpp"
    assert config.llamacpp_fork in ("official", "atomic")


def test_blackwell_runtime_env_count():
    """Verify canonical 36 hybrid-KV environment variables exist in Bruno builder."""
    assert len(BLACKWELL_RUNTIME_ENV) >= 36
    assert BLACKWELL_RUNTIME_ENV["LLAMA_QWEN35_BLACKWELL_ENGINE"] == "1"
    assert BLACKWELL_RUNTIME_ENV["LLAMA_ARG_JINJA"] == "1"
    assert BLACKWELL_RUNTIME_ENV["LLAMA_KV_SPARSE_DEVICE_PREFIX_TOKENS"] in ("16384", "65536")
    assert BLACKWELL_RUNTIME_ENV["LLAMA_KV_RESERVE_TOKENS"] in ("16384", "65536")
    assert len(BRUNO_RUNTIME_ENV) >= 36


def test_blackwell_binary_resolution():
    """Verify resolve_blackwell_server_exe returns valid path if installed."""
    exe = resolve_blackwell_server_exe()
    if os.path.exists(EXE_PATH_LLAMA_CPP_HUANG) or os.path.exists(EXE_PATH_LLAMA_CPP_BLACKWELL):
        assert exe is not None
        assert os.path.exists(exe)


def test_blackwell_preset_section_found():
    """Verify Blackwell Qwen3.8, Huang, and Strata preset sections are present in models.ini."""
    import configparser
    cp = configparser.ConfigParser(strict=False)
    cp.read(str(MODELS_PRESET_FILE), encoding="utf-8")
    blackwell_sec = "qwen-38_27b-blackwell-256k-think"
    assert cp.has_section(blackwell_sec)
    sec = dict(cp.items(blackwell_sec))
    assert "model" in sec
    assert "unsloth" in sec["model"].lower()
    assert sec.get("jinja") == "on"
    assert sec.get("cache-reuse") == "0"

    # Huang primary model preset
    huang_sec = "qwen-38_27b-huang-gsq-think"
    assert cp.has_section(huang_sec)
    h_sec = dict(cp.items(huang_sec))
    assert "ista-daslab" in h_sec.get("model", "").lower()
    assert h_sec.get("spec-type") == "draft-mtp"
    assert int(h_sec.get("spec-draft-n-max", 1)) == 3
    assert int(h_sec.get("kv-stream-arena-mib", 0)) == 2240

    # Strata MoE model preset (decoupled into native strata_models.json)
    strata_sec = "qwen-38-flash-strata-iq3s"
    from local_ai_launcher.backends.strata import load_strata_model_registry
    strata_reg = load_strata_model_registry()
    assert strata_sec in strata_reg
    s_sec = strata_reg[strata_sec]
    assert "sc117" in s_sec.get("model", "").lower()


def test_blackwell_update_locked(monkeypatch):
    """Verify that Blackwell forks have updates permanently locked."""
    from local_ai_launcher.backends.blackwell_builder import check_blackwell_llamacpp
    assert config.llamacpp_blackwell_update_behavior == "never_update"
    assert config.llamacpp_huang_update_behavior == "never_update"
    monkeypatch.setattr(config, "blackwell_engine", "bruno")
    assert check_blackwell_llamacpp() is True


def test_huang_engine_shim_routing():
    """Verify HuangLlamaServicer synthesizes adaptive KV streaming flags without Bruno env vars."""
    servicer = HuangLlamaServicer()

    with patch("subprocess.Popen") as mock_popen, \
         patch.object(servicer, "_wait_for_health", return_value=True), \
         patch("local_ai_launcher.process.assign_process_to_job"), \
         patch("local_ai_launcher.shims.huang_llama_shim.resolve_huang_server_exe", return_value=str(EXE_PATH_LLAMA_CPP_HUANG)), \
         patch("os.path.exists", return_value=True), \
         patch("urllib.request.urlopen"):
        mock_proc = MagicMock()
        mock_proc.pid = 20001
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        req_bw = backend_pb2.ModelOptions(Model="qwen-38_27b-blackwell-256k-think")
        res_bw = servicer.LoadModel(req_bw, None)
        assert res_bw.success is True
        assert servicer.current_is_blackwell is True

        cmd_bw = mock_popen.call_args[0][0]
        env_bw = mock_popen.call_args[1]["env"]

        # Verify Huang adaptive KV CLI flags are present and semantically valid
        assert "--kv-stream-arena-mib" in cmd_bw
        arena_idx = cmd_bw.index("--kv-stream-arena-mib")
        assert int(cmd_bw[arena_idx + 1]) >= 128

        assert "--kv-stream-auxiliary-layers" in cmd_bw
        aux_idx = cmd_bw.index("--kv-stream-auxiliary-layers")
        assert int(cmd_bw[aux_idx + 1]) >= 0

        assert "--spec-type" in cmd_bw
        assert cmd_bw[cmd_bw.index("--spec-type") + 1] in ("draft-mtp", "draft", "none")

        assert "--spec-draft-n-max" in cmd_bw
        assert int(cmd_bw[cmd_bw.index("--spec-draft-n-max") + 1]) >= 1

        assert "--spec-draft-p-min" in cmd_bw
        pmin_val = float(cmd_bw[cmd_bw.index("--spec-draft-p-min") + 1])
        assert 0.0 <= pmin_val <= 1.0

        assert "--spec-draft-p-split" in cmd_bw
        psplit_val = float(cmd_bw[cmd_bw.index("--spec-draft-p-split") + 1])
        assert 0.0 <= psplit_val <= 1.0

        assert "--batch-size" in cmd_bw
        bs_val = int(cmd_bw[cmd_bw.index("--batch-size") + 1])
        assert bs_val >= 1

        assert "--ubatch-size" in cmd_bw
        ubs_val = int(cmd_bw[cmd_bw.index("--ubatch-size") + 1])
        assert 1 <= ubs_val <= bs_val

        assert "--parallel" in cmd_bw
        assert int(cmd_bw[cmd_bw.index("--parallel") + 1]) >= 1

        assert "--no-mmproj" in cmd_bw

        # Verify clean env without Bruno's sparse VMM env vars
        assert "LLAMA_QWEN35_BLACKWELL_ENGINE" not in env_bw

    # Switch to standard model (tiel-coder-35b-think)
    with patch("subprocess.Popen") as mock_popen, \
         patch.object(servicer, "_wait_for_health", return_value=True), \
         patch("local_ai_launcher.process.assign_process_to_job"), \
         patch("os.path.exists", return_value=True), \
         patch("urllib.request.urlopen"):
        mock_proc = MagicMock()
        mock_proc.pid = 20002
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        req_std = backend_pb2.ModelOptions(Model="tiel-coder-35b-think")
        res_std = servicer.LoadModel(req_std, None)
        assert res_std.success is True
        assert servicer.current_is_blackwell is False
        assert servicer.current_server_exe == str(EXE_PATH_LLAMA_CPP)

        cmd_std = mock_popen.call_args[0][0]
        assert "--kv-stream-arena-mib" not in cmd_std


def test_huang_shim_graceful_fallback_to_bruno():
    """Verify that when Huang binary is missing, HuangLlamaServicer falls back to Bruno safely without invalid flags."""
    servicer = HuangLlamaServicer()

    def fake_exists(path):
        p = str(path).lower()
        if "huang" in p:
            return False
        if "blackwell" in p or "models" in p:
            return True
        return False

    with patch("subprocess.Popen") as mock_popen, \
         patch.object(servicer, "_wait_for_health", return_value=True), \
         patch("local_ai_launcher.process.assign_process_to_job"), \
         patch("local_ai_launcher.shims.huang_llama_shim.resolve_huang_server_exe", return_value=None), \
         patch("os.path.exists", side_effect=fake_exists), \
         patch("urllib.request.urlopen"):
        mock_proc = MagicMock()
        mock_proc.pid = 20005
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        req_bw = backend_pb2.ModelOptions(Model="qwen-38_27b-blackwell-256k-think")
        res_bw = servicer.LoadModel(req_bw, None)
        assert res_bw.success is True
        assert servicer.current_is_blackwell is True

        cmd_bw = mock_popen.call_args[0][0]
        env_bw = mock_popen.call_args[1]["env"]

        # Crucial: verify --kv-stream-arena-mib is NOT passed to Bruno's binary
        assert "--kv-stream-arena-mib" not in cmd_bw
        # Crucial: verify Bruno's sparse VMM env vars ARE passed so fallback works
        assert env_bw.get("LLAMA_QWEN35_BLACKWELL_ENGINE") == "1"


def test_huang_engine_shim_loads_huang_gsq_preset():
    """Verify HuangLlamaServicer directly loads the primary qwen-38_27b-huang-gsq-think preset with adaptive KV flags."""
    servicer = HuangLlamaServicer()

    with patch("subprocess.Popen") as mock_popen, \
         patch.object(servicer, "_wait_for_health", return_value=True), \
         patch("local_ai_launcher.process.assign_process_to_job"), \
         patch("local_ai_launcher.shims.huang_llama_shim.resolve_huang_server_exe", return_value=str(EXE_PATH_LLAMA_CPP_HUANG)), \
         patch("os.path.exists", return_value=True), \
         patch("urllib.request.urlopen"):
        mock_proc = MagicMock()
        mock_proc.pid = 20099
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        req = backend_pb2.ModelOptions(Model="qwen-38_27b-huang-gsq-think")
        res = servicer.LoadModel(req, None)
        assert res.success is True
        assert servicer.current_is_blackwell is True

        cmd = mock_popen.call_args[0][0]
        assert "--kv-stream-arena-mib" in cmd
        assert "--kv-stream-auxiliary-layers" in cmd
        assert "--spec-type" in cmd
        assert "--spec-draft-n-max" in cmd
        assert cmd[cmd.index("--spec-draft-n-max") + 1] == "3"
        assert "--batch-size" in cmd
        assert cmd[cmd.index("--batch-size") + 1] == "512"
        assert "--ubatch-size" in cmd
        assert cmd[cmd.index("--ubatch-size") + 1] == "256"
        assert "--no-mmproj" in cmd
        assert "--fit" not in cmd


def test_config_sanitization_huang_fields():
    """Verify configuration hardening clamps negative or out-of-bounds Huang tunables."""
    cfg = LauncherConfig()
    assert cfg._sanitize_field("huang_arena_mib", -500) == 128
    assert cfg._sanitize_field("huang_arena_mib", 100000) == 65536
    assert cfg._sanitize_field("huang_draft_max", -1) == 1
    assert cfg._sanitize_field("huang_draft_max", 32) == 16
    assert cfg._sanitize_field("huang_spec_type", "invalid_type") == "draft-mtp"
    assert cfg._sanitize_field("huang_spec_type", "draft-mtp") == "draft-mtp"
    assert cfg._sanitize_field("huang_draft_p_min", -0.5) == 0.0
    assert cfg._sanitize_field("huang_draft_p_min", 1.5) == 1.0
    assert cfg._sanitize_field("huang_draft_p_min", 0.20) == 0.20
    assert cfg._sanitize_field("huang_draft_p_split", -0.1) == 0.0
    assert cfg._sanitize_field("huang_draft_p_split", 1.2) == 1.0
    assert cfg._sanitize_field("huang_draft_p_split", 0.10) == 0.10
    assert cfg._sanitize_field("huang_auxiliary_layers", -1) == 0
    assert cfg._sanitize_field("huang_auxiliary_layers", 20) == 16
    assert cfg._sanitize_field("huang_auxiliary_layers", 1) == 1
    assert cfg._sanitize_field("huang_batch_size", 0) == 512
    assert cfg._sanitize_field("huang_batch_size", 32768) == 16384
    assert cfg._sanitize_field("huang_batch_size", 512) == 512
    assert cfg._sanitize_field("huang_ubatch_size", -10) == 256
    assert cfg._sanitize_field("huang_ubatch_size", 256) == 256
    assert cfg._sanitize_field("huang_parallel", 0) == 1
    assert cfg._sanitize_field("huang_parallel", 32) == 16
    assert cfg._sanitize_field("huang_parallel", 1) == 1
    assert cfg._sanitize_field("huang_no_mmproj", "true") is True
    assert cfg._sanitize_field("huang_no_mmproj", "false") is False
    assert cfg._sanitize_field("huang_no_mmproj", True) is True


def test_bruno_engine_shim_routing():
    """Verify BrunoLlamaServicer injects the 36 sparse VMM runtime env vars."""
    servicer = BrunoLlamaServicer()

    with patch("subprocess.Popen") as mock_popen, \
         patch.object(servicer, "_wait_for_health", return_value=True), \
         patch("local_ai_launcher.process.assign_process_to_job"), \
         patch("urllib.request.urlopen"):
        mock_proc = MagicMock()
        mock_proc.pid = 20003
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        req_bw = backend_pb2.ModelOptions(Model="qwen-38_27b-blackwell-256k-think")
        res_bw = servicer.LoadModel(req_bw, None)
        assert res_bw.success is True
        assert servicer.current_is_blackwell is True

        cmd_bw = mock_popen.call_args[0][0]
        env_bw = mock_popen.call_args[1]["env"]

        # Verify Bruno's runtime env is passed
        assert env_bw.get("LLAMA_QWEN35_BLACKWELL_ENGINE") == "1"
        assert env_bw.get("LLAMA_KV_SPARSE_DEVICE_PREFIX_TOKENS") in ("16384", "65536")

    # Switch to standard model (tiel-coder-35b-think)
    with patch("subprocess.Popen") as mock_popen, \
         patch.object(servicer, "_wait_for_health", return_value=True), \
         patch("local_ai_launcher.process.assign_process_to_job"), \
         patch("urllib.request.urlopen"):
        mock_proc = MagicMock()
        mock_proc.pid = 20004
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        req_std = backend_pb2.ModelOptions(Model="tiel-coder-35b-think")
        res_std = servicer.LoadModel(req_std, None)
        assert res_std.success is True
        assert servicer.current_is_blackwell is False
        assert servicer.current_server_exe == str(EXE_PATH_LLAMA_CPP)

        env_std = mock_popen.call_args[1]["env"]
        assert "LLAMA_QWEN35_BLACKWELL_ENGINE" not in env_std


def test_blackwell_facade_dispatcher(monkeypatch):
    """Verify BlackwellLlamaServicer facade dynamically yields Huang or Bruno instance based on config."""
    monkeypatch.setattr(config, "blackwell_engine", "huang")
    servicer_huang = BlackwellLlamaServicer()
    assert isinstance(servicer_huang, HuangLlamaServicer)

    monkeypatch.setattr(config, "blackwell_engine", "bruno")
    servicer_bruno = BlackwellLlamaServicer()
    assert isinstance(servicer_bruno, BrunoLlamaServicer)


def test_config_blackwell_engine_selection(tmp_path):
    """Verify loading blackwell engine selection from launcher.toml overrides."""
    toml_file = tmp_path / "launcher.toml"
    toml_file.write_text(
        """
[blackwell]
engine = "bruno"
arena_mib = 3000
auxiliary_layers = 2
draft_max = 5
draft_p_min = 0.25
draft_p_split = 0.15
spec_type = "draft-mtp"
batch_size = 512
ubatch_size = 256
parallel = 1
no_mmproj = true
        """,
        encoding="utf-8",
    )

    test_cfg = LauncherConfig(config_file=toml_file)
    assert test_cfg.blackwell_engine == "bruno"
    assert test_cfg.huang_arena_mib == 3000
    assert test_cfg.huang_auxiliary_layers == 2
    assert test_cfg.huang_draft_max == 5
    assert test_cfg.huang_draft_p_min == 0.25
    assert test_cfg.huang_draft_p_split == 0.15
    assert test_cfg.huang_batch_size == 512
    assert test_cfg.huang_ubatch_size == 256
    assert test_cfg.huang_parallel == 1
    assert test_cfg.huang_no_mmproj is True


def test_bruno_kv_cache_symmetry_and_clamping():
    """Verify Bruno Passini engine enforces symmetric q8_0/q8_0 or q4_0/q4_0 and clamps mixes/unsupported to lower."""
    assert resolve_bruno_kv_cache_types("q8_0", "q8_0") == ("q8_0", "q8_0")
    assert resolve_bruno_kv_cache_types("q8", "q8") == ("q8_0", "q8_0")
    assert resolve_bruno_kv_cache_types("q4_0", "q4_0") == ("q4_0", "q4_0")
    assert resolve_bruno_kv_cache_types("q4", "q4") == ("q4_0", "q4_0")

    # Mixed q8 / q4 must choose lower (q4_0 / q4_0)
    assert resolve_bruno_kv_cache_types("q8_0", "q4_0") == ("q4_0", "q4_0")
    assert resolve_bruno_kv_cache_types("q4_0", "q8_0") == ("q4_0", "q4_0")
    assert resolve_bruno_kv_cache_types("q8", "q4") == ("q4_0", "q4_0")

    # Any other format (f16, bf16, q5_0, etc.) clamps to q4_0 / q4_0
    assert resolve_bruno_kv_cache_types("f16", "f16") == ("q4_0", "q4_0")
    assert resolve_bruno_kv_cache_types("q5_0", "q5_0") == ("q4_0", "q4_0")
    assert resolve_bruno_kv_cache_types("bf16", "q8_0") == ("q4_0", "q4_0")
    assert resolve_bruno_kv_cache_types(None, None) == ("q4_0", "q4_0")


def test_is_bruno_model_allowed():
    """Verify Bruno engine strictly permits only unpruned Qwen3.8-27B-UD-IQ4_XS and Huihui abliterated UD-IQ4_XS."""
    # Permitted models
    assert is_bruno_model_allowed("qwen-38_27b-blackwell-256k-think") is True
    assert is_bruno_model_allowed("qwen-38_27b-blackwell-256k-nothink") is True
    assert is_bruno_model_allowed("qwen-38_27b-blackwell-256k-abli-think") is True
    assert is_bruno_model_allowed("qwen-38_27b-blackwell-256k-abli-nothink") is True
    assert is_bruno_model_allowed("", "C:/aiWorkspace/models/llama-server/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ4_XS.gguf") is True
    assert is_bruno_model_allowed("", "C:/aiWorkspace/models/llama-server/huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF/Huihui-Qwen3.8-27B-abliterated-UD-IQ4_XS.gguf") is True

    # Explicitly disallowed: ASCII variants
    assert is_bruno_model_allowed("qwen-38_27b-huang-ascii-think") is False
    assert is_bruno_model_allowed("", "C:/aiWorkspace/models/llama-server/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-ASCII-IQ4_XS.gguf") is False
    assert is_bruno_model_allowed("", "Huihui-Qwen3.8-27B-abliterated-UD-ASCII-IQ4_XS.gguf") is False

    # Explicitly disallowed: ISTA-DASLab / GSQ / IQ3_S variants
    assert is_bruno_model_allowed("qwen-38_27b-huang-gsq-iq3s-think") is False
    assert is_bruno_model_allowed("", "Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf") is False

    # Disallowed: Other models with accidental blackwell aliases
    assert is_bruno_model_allowed("gemma-4-blackwell-fake") is False
    assert is_bruno_model_allowed("tiel-coder-35b-think") is False


def test_bruno_servicer_diverts_huang_model_to_huang_engine():
    """Verify that if a Huang model is requested under Bruno, it automatically delegates to Huang engine."""
    servicer = BrunoLlamaServicer()

    with patch("subprocess.Popen") as mock_popen, \
         patch.object(HuangLlamaServicer, "_wait_for_health", return_value=True), \
         patch("local_ai_launcher.process.assign_process_to_job"), \
         patch("urllib.request.urlopen"):
        mock_proc = MagicMock()
        mock_proc.pid = 30001
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        req = backend_pb2.ModelOptions(Model="qwen-38_27b-huang-gsq-iq3s-think")
        res = servicer.LoadModel(req, None)
        assert res.success is True
        assert servicer._delegate is not None
        assert isinstance(servicer._delegate, HuangLlamaServicer)
        assert servicer.process == mock_proc

        # Verify Free delegates cleanly
        res_free = servicer.Free(None, None)
        assert res_free.success is True
        assert servicer._delegate is None


def test_bruno_servicer_runs_unrelated_model_in_official_llama_server():
    """Verify that models not qualifying for Bruno or Huang run in official llama-server.exe rather than crashing."""
    servicer = BrunoLlamaServicer()

    with patch("subprocess.Popen") as mock_popen, \
         patch.object(servicer, "_wait_for_health", return_value=True), \
         patch("local_ai_launcher.process.assign_process_to_job"), \
         patch("os.path.exists", return_value=True), \
         patch("urllib.request.urlopen"), \
         patch("local_ai_launcher.shims.bruno_llama_shim.get_active_hardware_profile", return_value={"name": "blackwell_16gb"}):
        mock_proc = MagicMock()
        mock_proc.pid = 30005
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        req = backend_pb2.ModelOptions(Model="fake-unrelated-blackwell-model")
        res = servicer.LoadModel(req, None)
        assert res.success is True
        assert servicer.current_is_blackwell is False
        assert servicer.current_server_exe == str(EXE_PATH_LLAMA_CPP)

        env_call = mock_popen.call_args[1].get("env", {})
        assert "LLAMA_QWEN35_BLACKWELL_ENGINE" not in env_call


def test_bruno_servicer_enforces_symmetric_kv_cache_cli():
    """Verify that Bruno Passini engine always applies symmetric --cache-type-k/--cache-type-v on CLI even if mixed in options."""
    servicer = BrunoLlamaServicer()

    with patch("subprocess.Popen") as mock_popen, \
         patch.object(servicer, "_wait_for_health", return_value=True), \
         patch("local_ai_launcher.process.assign_process_to_job"), \
         patch("urllib.request.urlopen"):
        mock_proc = MagicMock()
        mock_proc.pid = 30002
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        req = backend_pb2.ModelOptions(
            Model="qwen-38_27b-blackwell-256k-think",
            CacheTypeKey="q8_0",
            CacheTypeValue="q4_0"  # Mixed!
        )
        res = servicer.LoadModel(req, None)
        assert res.success is True

        cmd = mock_popen.call_args[0][0]
        # In cmd, --cache-type-k and --cache-type-v must be symmetric q4_0 because q8_0/q4_0 mixed clamps to lower
        assert "--cache-type-k" in cmd
        assert "--cache-type-v" in cmd
        ctk_idx = cmd.index("--cache-type-k")
        ctv_idx = cmd.index("--cache-type-v")
        assert cmd[ctk_idx + 1] == "q4_0"
        assert cmd[ctv_idx + 1] == "q4_0"


def test_serve_shim_accepts_port_kwarg():
    """Verify that serve_shim accepts port keyword argument without TypeError."""
    from local_ai_launcher.shims.base_shim import serve_shim, BaseBackendServicer
    servicer = BaseBackendServicer("dummy")
    with patch("argparse.ArgumentParser.parse_known_args", return_value=(MagicMock(addr="127.0.0.1:50051", port=None), [])), \
         patch("grpc.server") as mock_grpc_server, \
         patch("local_ai_launcher.localai.proto.backend_pb2_grpc.add_BackendServicer_to_server"):
        server_inst = MagicMock()
        mock_grpc_server.return_value = server_inst
        # Must not raise TypeError: serve_shim() got an unexpected keyword argument 'port'
        serve_shim(servicer, port=9940)
        server_inst.add_insecure_port.assert_called_with("127.0.0.1:9940")


def test_hermes_adapter_start_resolves_model_name():
    """Verify HermesAdapter.start defines model_name and passes it cleanly to preload_harness_context."""
    from local_ai_launcher.harnesses.hermes import HermesAdapter
    adapter = HermesAdapter(Path("C:/aiWorkspace"), [])

    with patch.object(adapter, "_ensure_user_profile_junction"), \
         patch.object(adapter, "_resolve_hermes_cli", return_value="C:/aiWorkspace/.hermes/bin/hermes.exe"), \
         patch("local_ai_launcher.skills_manager.SkillManager.ensure_builtin_skills"), \
         patch("local_ai_launcher.harnesses.hermes.start_in_wt_tab", return_value=MagicMock()), \
         patch("local_ai_launcher.harnesses.hermes.start_process", return_value=MagicMock()), \
         patch("local_ai_launcher.harnesses.hermes.subprocess.run", return_value=MagicMock()), \
         patch("local_ai_launcher.harnesses.hermes.time.sleep", return_value=None), \
         patch("local_ai_launcher.harnesses.hermes.config") as mock_config, \
         patch("local_ai_launcher.context_preloader.preload_harness_context") as mock_preload:
        mock_config.skip_hermes_web_ui_build = True
        mock_config.start_discord_gateway = False
        mock_config.start_signal_daemon = False
        mock_config.harness_start_dir = Path("C:/aiWorkspace")
        # Must not raise NameError: name 'model_name' is not defined
        adapter.start("http://127.0.0.1:8080", active_model="qwen-38_27b-huang-gsq-think", launch_tui=True)
        assert mock_preload.called
        call_args = mock_preload.call_args[0]
        assert call_args[0] == "http://127.0.0.1:8080"
        assert "qwen" in call_args[1].lower()
        assert mock_preload.call_args[1].get("harness_name") == "hermes"


def test_huang_shim_drops_arena_when_spec_type_none():
    """Verify HuangLlamaServicer decouples --kv-stream-arena-mib when spec-type is 'none' and runs official binary."""
    servicer = HuangLlamaServicer()

    with patch("subprocess.Popen") as mock_popen, \
         patch.object(servicer, "_wait_for_health", return_value=True), \
         patch("local_ai_launcher.process.assign_process_to_job"), \
         patch("local_ai_launcher.shims.huang_llama_shim.resolve_huang_server_exe", return_value=str(EXE_PATH_LLAMA_CPP_HUANG)), \
         patch("os.path.exists", return_value=True), \
         patch("urllib.request.urlopen"):
        mock_proc = MagicMock()
        mock_proc.pid = 20101
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        # Request with explicit EngineArgs disabling speculation: spec-type = none
        req = backend_pb2.ModelOptions(
            Model="qwen-38_27b-huang-gsq-think",
            EngineArgs='{"spec-type": "none"}'
        )
        # Mock matched_opts with spec-type = none
        with patch("local_ai_launcher.shims.huang_llama_shim.load_effective_models_parser") as mock_parser:
            mock_cp = MagicMock()
            mock_cp.sections.return_value = ["qwen-38_27b-huang-gsq-think"]
            mock_cp.items.return_value = [
                ("model", "models/llama-server/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf"),
                ("spec-type", "none"),
                ("ctx-size", "262000"),
                ("kv-stream-arena-mib", "2240"),
            ]
            mock_cp.get.return_value = "qwen-38_27b-huang-gsq-think"
            mock_parser.return_value = mock_cp

            res = servicer.LoadModel(req, None)
            assert res.success is True

            cmd = mock_popen.call_args[0][0]
            # When spec-type is none, fallback MUST route to official upstream binary
            assert cmd[0] == str(EXE_PATH_LLAMA_CPP)
            assert servicer.current_server_exe == str(EXE_PATH_LLAMA_CPP)
            assert "--kv-stream-arena-mib" not in cmd
            assert "--kv-stream-auxiliary-layers" not in cmd
            assert "--spec-type" in cmd
            assert cmd[cmd.index("--spec-type") + 1] == "none"


def test_huang_shim_drops_arena_when_model_lacks_mtp():
    """Verify HuangLlamaServicer decouples --kv-stream-arena-mib when loading a direct non-MTP model and runs official binary."""
    servicer = HuangLlamaServicer()

    with patch("subprocess.Popen") as mock_popen, \
         patch.object(servicer, "_wait_for_health", return_value=True), \
         patch("local_ai_launcher.process.assign_process_to_job"), \
         patch("local_ai_launcher.shims.huang_llama_shim.resolve_huang_server_exe", return_value=str(EXE_PATH_LLAMA_CPP_HUANG)), \
         patch("os.path.exists", return_value=True), \
         patch("urllib.request.urlopen"), \
         patch("local_ai_launcher.shims.huang_llama_shim.check_model_supports_mtp", return_value=False):
        mock_proc = MagicMock()
        mock_proc.pid = 20102
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        req = backend_pb2.ModelOptions(
            ModelFile="C:/aiWorkspace/models/llama-server/unsloth/qwen-38_27b-test-no-mtp.gguf",
            ContextSize=65536
        )
        res = servicer.LoadModel(req, None)
        assert res.success is True

        cmd = mock_popen.call_args[0][0]
        # Must fall back to official upstream binary, omit --kv-stream-arena-mib and pass --spec-type none
        assert cmd[0] == str(EXE_PATH_LLAMA_CPP)
        assert servicer.current_server_exe == str(EXE_PATH_LLAMA_CPP)
        assert "--kv-stream-arena-mib" not in cmd
        assert "--kv-stream-auxiliary-layers" not in cmd
        assert "--spec-type" in cmd
        assert cmd[cmd.index("--spec-type") + 1] == "none"


def test_huang_shim_forces_fit_off_when_model_exceeds_vram_threshold():
    """Verify HuangLlamaServicer explicitly forces --fit off when model size > (VRAM - 500MB) without CPU offload."""
    servicer = HuangLlamaServicer()

    with patch("subprocess.Popen") as mock_popen, \
         patch.object(servicer, "_wait_for_health", return_value=True), \
         patch("local_ai_launcher.process.assign_process_to_job"), \
         patch("local_ai_launcher.shims.huang_llama_shim.resolve_huang_server_exe", return_value=str(EXE_PATH_LLAMA_CPP_HUANG)), \
         patch("os.path.exists", return_value=True), \
         patch("urllib.request.urlopen"), \
         patch("local_ai_launcher.shims.huang_llama_shim.check_model_supports_mtp", return_value=False), \
         patch("local_ai_launcher.shims.huang_llama_shim.should_force_fit_off", return_value=(True, 16000.0, 15884.0)):
        mock_proc = MagicMock()
        mock_proc.pid = 20103
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        req = backend_pb2.ModelOptions(
            ModelFile="C:/aiWorkspace/models/llama-server/unsloth/qwen-38_27b-test-no-mtp.gguf",
            ContextSize=65536
        )
        res = servicer.LoadModel(req, None)
        assert res.success is True

        cmd = mock_popen.call_args[0][0]
        assert cmd[0] == str(EXE_PATH_LLAMA_CPP)
        assert "--fit" in cmd
        assert cmd.count("--fit") == 1
        assert cmd[cmd.index("--fit") + 1] == "off"


def test_huang_shim_enforces_non_duplicated_fit_off():
    """Verify that when fallback enforces --fit off, the --fit flag is never duplicated even if previously present."""
    from local_ai_launcher.shims.huang_llama_shim import enforce_cmd_arg, enforce_fallback_fit_off_if_needed

    cmd = ["llama-server.exe", "--port", "8080", "--fit", "on", "--ctx-size", "4096"]
    enforce_cmd_arg(cmd, "--fit", "off")
    assert cmd.count("--fit") == 1
    assert cmd[cmd.index("--fit") + 1] == "off"

    # Calling again should still produce exactly one --fit off
    enforce_cmd_arg(cmd, "--fit", "off")
    assert cmd.count("--fit") == 1
    assert cmd[cmd.index("--fit") + 1] == "off"


def test_huang_engine_never_adds_fit_off_for_mtp_model():
    """Verify that genuine Huang execution for an MTP-enabled model never appends --fit off."""
    servicer = HuangLlamaServicer()

    with patch("subprocess.Popen") as mock_popen, \
         patch.object(servicer, "_wait_for_health", return_value=True), \
         patch("local_ai_launcher.process.assign_process_to_job"), \
         patch("local_ai_launcher.shims.huang_llama_shim.resolve_huang_server_exe", return_value=str(EXE_PATH_LLAMA_CPP_HUANG)), \
         patch("os.path.exists", return_value=True), \
         patch("urllib.request.urlopen"), \
         patch("local_ai_launcher.shims.huang_llama_shim.check_model_supports_mtp", return_value=True):
        mock_proc = MagicMock()
        mock_proc.pid = 20104
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        req = backend_pb2.ModelOptions(
            ModelFile="C:/aiWorkspace/models/llama-server/unsloth/qwen-38_27b-test-mtp.gguf",
            ContextSize=65536
        )
        res = servicer.LoadModel(req, None)
        assert res.success is True

        cmd = mock_popen.call_args[0][0]
        assert cmd[0] == str(EXE_PATH_LLAMA_CPP_HUANG)
        assert "--kv-stream-arena-mib" in cmd
        assert "--fit" not in cmd


def test_huang_shim_falls_back_to_official_when_model_exceeds_vram_even_with_mtp():
    """Verify HuangLlamaServicer falls back to official llama-server when model size > VRAM - 500MB even if MTP is present."""
    servicer = HuangLlamaServicer()

    with patch("subprocess.Popen") as mock_popen, \
         patch.object(servicer, "_wait_for_health", return_value=True), \
         patch("local_ai_launcher.process.assign_process_to_job"), \
         patch("local_ai_launcher.shims.huang_llama_shim.resolve_huang_server_exe", return_value=str(EXE_PATH_LLAMA_CPP_HUANG)), \
         patch("os.path.exists", return_value=True), \
         patch("urllib.request.urlopen"), \
         patch("local_ai_launcher.shims.huang_llama_shim.check_model_supports_mtp", return_value=True), \
         patch("local_ai_launcher.shims.huang_llama_shim.check_model_exceeds_vram_threshold", return_value=(True, 16000.0, 15884.0)), \
         patch("local_ai_launcher.shims.huang_llama_shim.should_force_fit_off", return_value=(True, 16000.0, 15884.0)):
        mock_proc = MagicMock()
        mock_proc.pid = 20105
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        req = backend_pb2.ModelOptions(
            ModelFile="C:/aiWorkspace/models/llama-server/unsloth/qwen-38_27b-test-mtp.gguf",
            ContextSize=65536
        )
        res = servicer.LoadModel(req, None)
        assert res.success is True

        cmd = mock_popen.call_args[0][0]
        # Must fall back to official upstream binary even though MTP is supported
        assert cmd[0] == str(EXE_PATH_LLAMA_CPP)
        assert servicer.current_server_exe == str(EXE_PATH_LLAMA_CPP)
        assert "--kv-stream-arena-mib" not in cmd
        assert "--spec-type" in cmd
        assert cmd[cmd.index("--spec-type") + 1] == "none"
        # Non-duplicated --fit off enforced on official fallback
        assert "--fit" in cmd
        assert cmd.count("--fit") == 1
        assert cmd[cmd.index("--fit") + 1] == "off"


def test_huang_shim_preset_falls_back_to_official_when_model_exceeds_vram():
    """Verify HuangLlamaServicer preset load falls back to official llama-server when preset model > VRAM - 500MB."""
    servicer = HuangLlamaServicer()

    with patch("subprocess.Popen") as mock_popen, \
         patch.object(servicer, "_wait_for_health", return_value=True), \
         patch("local_ai_launcher.process.assign_process_to_job"), \
         patch("local_ai_launcher.shims.huang_llama_shim.resolve_huang_server_exe", return_value=str(EXE_PATH_LLAMA_CPP_HUANG)), \
         patch("os.path.exists", return_value=True), \
         patch("urllib.request.urlopen"), \
         patch("local_ai_launcher.shims.huang_llama_shim.check_model_supports_mtp", return_value=True), \
         patch("local_ai_launcher.shims.huang_llama_shim.check_model_exceeds_vram_threshold", return_value=(True, 16000.0, 15884.0)), \
         patch("local_ai_launcher.shims.huang_llama_shim.should_force_fit_off", return_value=(True, 16000.0, 15884.0)):
        mock_proc = MagicMock()
        mock_proc.pid = 20106
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        req = backend_pb2.ModelOptions(Model="qwen-38_27b-huang-gsq-think")
        res = servicer.LoadModel(req, None)
        assert res.success is True

        cmd = mock_popen.call_args[0][0]
        assert cmd[0] == str(EXE_PATH_LLAMA_CPP)
        assert servicer.current_server_exe == str(EXE_PATH_LLAMA_CPP)
        assert "--kv-stream-arena-mib" not in cmd
        assert "--fit" in cmd
        assert cmd.count("--fit") == 1
        assert cmd[cmd.index("--fit") + 1] == "off"


def test_router_model_below_63_percent_vram_routes_to_official():
    """Verify router diverts model below 63% VRAM threshold to Official Upstream."""
    from local_ai_launcher.backends.llamacpp_router import resolve_llamacpp_variant

    with patch("local_ai_launcher.backends.llamacpp_router.get_system_vram_mb", return_value=16384.0), \
         patch("local_ai_launcher.backends.llamacpp_router.get_model_size_mb", return_value=5000.0), \
         patch("local_ai_launcher.backends.llamacpp_router.check_model_supports_mtp", return_value=True):
        variant, reason, meta = resolve_llamacpp_variant("small-model-mtp.gguf", preset_opts={})
        assert variant == "official"
        assert "below 63% threshold" in reason
        assert meta["is_fit_off_needed"] is False


def test_router_model_with_cpu_offload_routes_to_official():
    """Verify router diverts models with CPU MoE offload or partial GPU offload to Official Upstream."""
    from local_ai_launcher.backends.llamacpp_router import resolve_llamacpp_variant

    with patch("local_ai_launcher.backends.llamacpp_router.get_system_vram_mb", return_value=16384.0), \
         patch("local_ai_launcher.backends.llamacpp_router.get_model_size_mb", return_value=12000.0), \
         patch("local_ai_launcher.backends.llamacpp_router.check_model_supports_mtp", return_value=True):
        # Case A: n-cpu-moe > 0
        variant, reason, meta = resolve_llamacpp_variant(
            "model-moe.gguf",
            preset_opts={"n-cpu-moe": "4"}
        )
        assert variant == "official"
        assert "CPU MoE offloading enabled" in reason
        assert meta["has_cpu_offload"] is True

        # Case B: partial ngl (e.g. ngl = 25)
        variant_b, reason_b, meta_b = resolve_llamacpp_variant(
            "model-moe.gguf",
            preset_opts={"ngl": "25"}
        )
        assert variant_b == "official"
        assert "Partial GPU offload configured" in reason_b
        assert meta_b["has_cpu_offload"] is True


def test_router_model_in_sweet_spot_routes_to_huang():
    """Verify router selects Huang for models in 63%~96.8% VRAM with verified MTP and zero CPU offload."""
    from local_ai_launcher.backends.llamacpp_router import resolve_llamacpp_variant

    with patch("local_ai_launcher.backends.llamacpp_router.get_system_vram_mb", return_value=16384.0), \
         patch("local_ai_launcher.backends.llamacpp_router.get_model_size_mb", return_value=12000.0), \
         patch("local_ai_launcher.backends.llamacpp_router.check_model_supports_mtp", return_value=True), \
         patch("local_ai_launcher.backends.llamacpp_router.is_bruno_model_allowed", return_value=False):
        variant, reason, meta = resolve_llamacpp_variant("mid-model-mtp.gguf", preset_opts={})
        assert variant == "huang"
        assert "sweet spot" in reason
        assert meta["is_fit_off_needed"] is False


def test_router_model_exceeding_ceiling_forces_fit_off_on_official():
    """Verify router selects Official and marks fit_off_needed for models exceeding VRAM - 500MB without CPU offload."""
    from local_ai_launcher.backends.llamacpp_router import resolve_llamacpp_variant

    with patch("local_ai_launcher.backends.llamacpp_router.get_system_vram_mb", return_value=16384.0), \
         patch("local_ai_launcher.backends.llamacpp_router.get_model_size_mb", return_value=16000.0), \
         patch("local_ai_launcher.backends.llamacpp_router.check_model_supports_mtp", return_value=True), \
         patch("local_ai_launcher.backends.llamacpp_router.is_bruno_model_allowed", return_value=False):
        variant, reason, meta = resolve_llamacpp_variant("huge-model-mtp.gguf", preset_opts={})
        assert variant == "official"
        assert "exceeds" in reason
        assert meta["is_fit_off_needed"] is True


def test_router_bruno_priority_on_blackwell():
    """Verify router selects Bruno on 16GB Blackwell hardware for permitted Bruno models."""
    from local_ai_launcher.backends.llamacpp_router import resolve_llamacpp_variant

    with patch("local_ai_launcher.backends.llamacpp_router.get_system_vram_mb", return_value=16384.0), \
         patch("local_ai_launcher.backends.llamacpp_router.get_active_hardware_profile", return_value={"name": "blackwell_16gb"}):
        variant, reason, meta = resolve_llamacpp_variant("qwen-38_27b-blackwell-256k-think", preset_opts={})
        assert variant == "bruno"
        assert "Bruno Passini" in reason


def test_router_explicit_official_hint_bypasses_forks():
    """Verify explicit official hint in preset or alias routes directly to Official Upstream."""
    from local_ai_launcher.backends.llamacpp_router import resolve_llamacpp_variant

    with patch("local_ai_launcher.backends.llamacpp_router.get_system_vram_mb", return_value=16384.0):
        # Via engine option
        variant, reason, meta = resolve_llamacpp_variant(
            "qwen-38_27b-blackwell-256k-think",
            preset_opts={"engine": "official"}
        )
        assert variant == "official"
        assert "explicitly requested" in reason

        # Via alias hint
        variant_b, reason_b, _ = resolve_llamacpp_variant(
            "custom-model",
            preset_opts={"alias": "custom-official-upstream"}
        )
        assert variant_b == "official"
        assert "explicitly requested" in reason_b


def test_huang_servicer_delegate_methods():
    """Verify HuangLlamaServicer delegates Predict, PredictStream, and Free cleanly to _delegate."""
    servicer = HuangLlamaServicer()
    mock_delegate = MagicMock()
    mock_delegate.Predict.return_value = backend_pb2.Reply(message=b"delegated_predict")
    mock_delegate.PredictStream.return_value = iter([backend_pb2.Reply(message=b"delegated_stream")])
    mock_delegate.Free.return_value = backend_pb2.Result(success=True, message="delegated_free")

    servicer._delegate = mock_delegate

    # Test Predict
    req = backend_pb2.PredictOptions(Prompt="hello")
    res_p = servicer.Predict(req, None)
    assert res_p.message == b"delegated_predict"
    mock_delegate.Predict.assert_called_once()

    # Test PredictStream
    res_s = list(servicer.PredictStream(req, None))
    assert len(res_s) == 1
    assert res_s[0].message == b"delegated_stream"
    mock_delegate.PredictStream.assert_called_once()

    # Test Free
    res_f = servicer.Free(None, None)
    assert res_f.success is True
    assert servicer._delegate is None
    mock_delegate.Free.assert_called_once()





