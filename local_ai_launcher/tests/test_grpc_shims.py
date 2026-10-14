import pytest
from unittest.mock import MagicMock
from local_ai_launcher.localai.proto import backend_pb2
from local_ai_launcher.shims.base_shim import BaseBackendServicer
from local_ai_launcher.shims.blackwell_llama_shim import BlackwellLlamaServicer
from local_ai_launcher.shims.huang_llama_shim import HuangLlamaServicer
from local_ai_launcher.shims.bruno_llama_shim import BrunoLlamaServicer
from local_ai_launcher.shims.tabby_grpc_shim import TabbyAPIServicer
from local_ai_launcher.shims.freetoken_grpc_shim import FreeTokenServicer
from local_ai_launcher.shims.strata_grpc_shim import StrataServicer


def test_base_shim_servicer():
    servicer = BaseBackendServicer("test-backend")
    health = servicer.Health(backend_pb2.HealthMessage(), None)
    assert health.message == b"OK"

    status = servicer.Status(backend_pb2.HealthMessage(), None)
    assert status.state == backend_pb2.StatusResponse.UNINITIALIZED

    res = servicer.LoadModel(backend_pb2.ModelOptions(Model="test"), None)
    assert res.success is True

    status = servicer.Status(backend_pb2.HealthMessage(), None)
    assert status.state == backend_pb2.StatusResponse.READY

    free_res = servicer.Free(None, None)
    assert free_res.success is True
    assert servicer.is_ready is False


def test_blackwell_llama_servicer_payload_generation():
    servicer = BlackwellLlamaServicer()
    req = backend_pb2.PredictOptions(
        Prompt="Hello world",
        Temperature=0.8,
        TopP=0.9,
        Tokens=256,
        StopPrompts=["<|end|>"]
    )
    payload = servicer._build_request_payload(req, stream=False)
    assert payload["prompt"] == "Hello world"
    assert payload["temperature"] == pytest.approx(0.8, 0.001)
    assert payload["top_p"] == pytest.approx(0.9, 0.001)
    assert payload["max_tokens"] == 256
    assert payload["stop"] == ["<|end|>"]
    assert payload["stream"] is False

    # Messages format
    req_chat = backend_pb2.PredictOptions(
        Messages=[backend_pb2.Message(role="user", content="ping")]
    )
    payload_chat = servicer._build_request_payload(req_chat, stream=True)
    assert len(payload_chat["messages"]) == 1
    assert payload_chat["messages"][0]["role"] == "user"
    assert payload_chat["messages"][0]["content"] == "ping"
    assert payload_chat["stream"] is True


def test_tabby_servicer_payload_generation():
    servicer = TabbyAPIServicer()
    servicer.current_model = "test-exl3"
    req = backend_pb2.PredictOptions(
        Prompt="test prompt",
        Temperature=0.5,
        Tokens=100
    )
    payload = servicer._build_request_payload(req)
    assert payload["model"] == "test-exl3"
    assert len(payload["messages"]) == 1
    assert payload["messages"][0]["content"] == "test prompt"
    assert payload["temperature"] == 0.5


def test_freetoken_servicer_payload_generation():
    servicer = FreeTokenServicer()
    servicer.current_model = "test-nvfp4"
    req = backend_pb2.PredictOptions(
        Prompt="test prompt",
        Tokens=50
    )
    payload = servicer._build_request_payload(req)
    assert payload["model"] == "test-nvfp4"
    assert payload["max_tokens"] == 50


def test_blackwell_shim_env_contains_home_and_hf(monkeypatch):
    """Verifies blackwell_llama_shim injects HOME and HF_HOME into env for subprocess.Popen."""
    from unittest.mock import patch, MagicMock
    import os

    servicer = BlackwellLlamaServicer()
    request = backend_pb2.ModelOptions(Model="test-model")

    captured_env = {}

    def mock_popen(cmd, **kwargs):
        captured_env.update(kwargs.get("env", {}))
        proc = MagicMock()
        proc.pid = 12345
        proc.poll.return_value = None
        return proc

    # Prevent actual llama-server launch but capture the env dict
    monkeypatch.setattr("os.path.exists", lambda p: True)
    with patch("subprocess.Popen", side_effect=mock_popen), \
         patch.object(servicer, "_wait_for_health", return_value=True), \
         patch("local_ai_launcher.shims.blackwell_llama_shim.resolve_blackwell_server_exe", return_value=None), \
         patch("local_ai_launcher.process.assign_process_to_job", return_value=None):
        servicer.LoadModel(request, None)

    assert "HOME" in captured_env
    assert "Users" not in captured_env["HOME"]
    assert "HF_HOME" in captured_env
    assert "HF_HUB_CACHE" in captured_env


def test_tabby_shim_env_contains_home_and_hf(monkeypatch):
    """Verifies tabby_grpc_shim passes HOME and HF_HOME in env to subprocess.Popen."""
    from unittest.mock import patch, MagicMock

    servicer = TabbyAPIServicer()
    request = backend_pb2.ModelOptions(Model="test-exl3-model")

    captured_env = {}

    def mock_popen(cmd, **kwargs):
        captured_env.update(kwargs.get("env", {}))
        proc = MagicMock()
        proc.pid = 12345
        proc.poll.return_value = None
        return proc

    with patch("subprocess.Popen", side_effect=mock_popen), \
         patch.object(servicer, "_wait_for_health", return_value=True), \
         patch("local_ai_launcher.shims.tabby_grpc_shim.setup_tabbyapi_config", return_value=None), \
         patch("local_ai_launcher.process.assign_process_to_job", return_value=None):
        servicer.LoadModel(request, None)

    assert "HOME" in captured_env
    assert "Users" not in captured_env["HOME"]
    assert "HF_HOME" in captured_env
    assert "HF_HUB_CACHE" in captured_env


def test_freetoken_shim_env_contains_home_and_hf(monkeypatch):
    """Verifies freetoken_grpc_shim passes HOME and HF_HOME in env to subprocess.Popen."""
    from unittest.mock import patch, MagicMock

    servicer = FreeTokenServicer()
    request = backend_pb2.ModelOptions(Model="test-nvfp4-model")

    captured_env = {}

    def mock_popen(cmd, **kwargs):
        captured_env.update(kwargs.get("env", {}))
        proc = MagicMock()
        proc.pid = 12345
        proc.poll.return_value = None
        return proc

    with patch("subprocess.Popen", side_effect=mock_popen), \
         patch.object(servicer, "_wait_for_health", return_value=True), \
         patch("local_ai_launcher.shims.freetoken_grpc_shim.check_freetoken", return_value=True), \
         patch("local_ai_launcher.shims.freetoken_grpc_shim._patch_freetoken_windows_hotfixes", return_value=None), \
         patch("local_ai_launcher.process.assign_process_to_job", return_value=None):
        servicer.LoadModel(request, None)

    assert "HOME" in captured_env
    assert "Users" not in captured_env["HOME"]
    assert "HF_HOME" in captured_env
    assert "HF_HUB_CACHE" in captured_env


def test_serve_shim_port_sanitization(monkeypatch):
    """Verify serve_shim safely falls back to default_addr when invalid or negative ports are passed."""
    import sys
    from unittest.mock import patch, MagicMock
    from local_ai_launcher.shims.base_shim import serve_shim

    servicer = BaseBackendServicer("test-hardening-backend")
    mock_server = MagicMock()

    # Case 1: Negative port flag --port -8080
    monkeypatch.setattr(sys, "argv", ["shim", "--port", "-8080"])
    with patch("grpc.server", return_value=mock_server), \
         patch("local_ai_launcher.shims.base_shim.backend_pb2_grpc.add_BackendServicer_to_server"):
        mock_server.wait_for_termination.side_effect = KeyboardInterrupt()
        serve_shim(servicer, default_addr="127.0.0.1:50051")
        mock_server.add_insecure_port.assert_called_with("127.0.0.1:50051")

    # Case 2: Out of range port in --addr 127.0.0.1:90000
    mock_server.reset_mock()
    monkeypatch.setattr(sys, "argv", ["shim", "--addr", "127.0.0.1:90000"])
    with patch("grpc.server", return_value=mock_server), \
         patch("local_ai_launcher.shims.base_shim.backend_pb2_grpc.add_BackendServicer_to_server"):
        mock_server.wait_for_termination.side_effect = KeyboardInterrupt()
        serve_shim(servicer, default_addr="127.0.0.1:50051")
        mock_server.add_insecure_port.assert_called_with("127.0.0.1:50051")


def test_shims_negative_and_out_of_bounds_sampling():
    """Verify all gRPC shims clamp negative or out-of-range sampling and penalty parameters."""
    # 1. Blackwell Llama Servicer
    b_servicer = BlackwellLlamaServicer()
    req_b = backend_pb2.PredictOptions(
        Temperature=3.5,
        TopP=1.5,
        MinP=1.2,
        PresencePenalty=-3.5,
        FrequencyPenalty=3.5,
        Seed=-100,
        Tokens=-50
    )
    b_payload = b_servicer._build_request_payload(req_b)
    assert b_payload["temperature"] <= 2.0
    assert b_payload["top_p"] <= 1.0
    assert b_payload["min_p"] <= 1.0
    assert b_payload["presence_penalty"] >= -2.0
    assert b_payload["frequency_penalty"] <= 2.0
    assert "max_tokens" not in b_payload or b_payload.get("max_tokens", 1) >= 1

    # 2. FreeToken Servicer
    ft_servicer = FreeTokenServicer()
    req_ft = backend_pb2.PredictOptions(
        Temperature=-0.5,
        TopP=-0.2,
        PresencePenalty=-4.0,
        FrequencyPenalty=4.0
    )
    ft_payload = ft_servicer._build_request_payload(req_ft)
    if "presence_penalty" in ft_payload:
        assert ft_payload["presence_penalty"] >= -2.0
    if "frequency_penalty" in ft_payload:
        assert ft_payload["frequency_penalty"] <= 2.0

    # 3. TabbyAPI Servicer
    tabby_servicer = TabbyAPIServicer()
    req_tabby = backend_pb2.PredictOptions(
        Temperature=3.0,
        TopP=1.8,
        TopK=-10
    )
    tabby_payload = tabby_servicer._build_request_payload(req_tabby)
    assert tabby_payload["temperature"] <= 2.0
    assert tabby_payload["top_p"] <= 1.0

    # 4. Huang Llama Servicer
    h_servicer = HuangLlamaServicer()
    h_payload = h_servicer._build_request_payload(req_b)
    assert h_payload["temperature"] <= 2.0
    assert h_payload["top_p"] <= 1.0
    assert h_payload["min_p"] <= 1.0
    assert h_payload["presence_penalty"] >= -2.0
    assert h_payload["frequency_penalty"] <= 2.0

    # 5. Bruno Llama Servicer
    br_servicer = BrunoLlamaServicer()
    br_payload = br_servicer._build_request_payload(req_b)
    assert br_payload["temperature"] <= 2.0
    assert br_payload["top_p"] <= 1.0
    assert br_payload["min_p"] <= 1.0
    assert br_payload["presence_penalty"] >= -2.0
    assert br_payload["frequency_penalty"] <= 2.0

    # 6. Strata Servicer
    st_servicer = StrataServicer()
    st_payload = st_servicer._build_request_payload(req_b)
    assert st_payload["temperature"] <= 2.0
    assert st_payload["top_p"] <= 1.0
    assert st_payload["min_p"] <= 1.0
    assert st_payload["presence_penalty"] >= -2.0
    assert st_payload["frequency_penalty"] <= 2.0


def test_greedy_decoding_with_explicit_temperature():
    """Verify that temperature=0.0 is forwarded when explicitly set, but not on default proto."""
    for ServicerClass in [
        BlackwellLlamaServicer,
        FreeTokenServicer,
        TabbyAPIServicer,
        HuangLlamaServicer,
        BrunoLlamaServicer,
        StrataServicer,
    ]:
        servicer = ServicerClass()

        # Case 1: Unset temperature in protobuf (default 0.0 without metadata) -> omitted to preserve backend default
        req_default = backend_pb2.PredictOptions(Prompt="test")
        payload_default = servicer._build_request_payload(req_default)
        assert "temperature" not in payload_default

        # Case 2: Explicit greedy decoding (temperature=0.0 with explicit_temperature in Metadata)
        req_greedy = backend_pb2.PredictOptions(
            Prompt="test",
            Temperature=0.0,
            Metadata={"explicit_temperature": "0.0"}
        )
        payload_greedy = servicer._build_request_payload(req_greedy)
        assert "temperature" in payload_greedy
        assert payload_greedy["temperature"] == 0.0

        # Case 3: Standard non-zero temperature
        req_normal = backend_pb2.PredictOptions(
            Prompt="test",
            Temperature=0.7
        )
        payload_normal = servicer._build_request_payload(req_normal)
        assert payload_normal["temperature"] == pytest.approx(0.7, 0.001)


def test_strata_servicer_payload_generation():
    """Verify StrataServicer correctly formats OpenAI-compatible chat payload."""
    servicer = StrataServicer()
    servicer.current_model = "qwen-38-flash-strata-iq3s"

    req = backend_pb2.PredictOptions(
        Prompt="Hello Strata",
        Temperature=0.6,
        TopP=0.95,
        Tokens=512
    )
    payload = servicer._build_request_payload(req, stream=False)
    assert payload["model"] == "qwen-38-flash-strata-iq3s"
    assert payload["messages"] == [{"role": "user", "content": "Hello Strata"}]
    assert payload["temperature"] == pytest.approx(0.6, 0.001)
    assert payload["top_p"] == pytest.approx(0.95, 0.001)
    assert payload["max_tokens"] == 512
    assert payload["stream"] is False


def test_strata_model_alias_resolution():
    """Verify resolve_strata_model_files successfully maps aliases like strata-iq3s to physical GGUF shards."""
    from local_ai_launcher.backends.strata import resolve_strata_model_files
    res = resolve_strata_model_files("strata-iq3s")
    assert res["first_shard"] is not None
    assert "Qwen3.8-Flash-Next" in res["first_shard"].name
    if res["first_shard"].exists():
        assert res["ple_shard"] is not None
        assert res["ple_shard"].exists()
    assert res["mtp_dir"] is not None
    assert res.get("chat_template_file") is not None
    assert "chat_template" in res["chat_template_file"]


def test_strata_servicer_load_model_with_alias():
    """Verify StrataServicer.LoadModel properly resolves model alias and builds server config."""
    from unittest.mock import patch, MagicMock
    servicer = StrataServicer()
    req = backend_pb2.ModelOptions(Model="strata-iq3s")

    with patch.object(servicer, "_wait_for_health", return_value=True), \
         patch("local_ai_launcher.process.assign_process_to_job"), \
         patch("subprocess.Popen") as mock_popen, \
         patch("local_ai_launcher.backends.strata.ensure_strata_pack", return_value=True):
        mock_proc = MagicMock()
        mock_proc.pid = 40001
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        res = servicer.LoadModel(req, None)
        assert res.success is True
        assert servicer.is_ready is True
        assert servicer.current_model == "strata-iq3s"
        assert mock_popen.called
        cmd = mock_popen.call_args[0][0]
        assert "--engine" in cmd
        assert "strata" in cmd
        servicer.Free(None, None)


def test_strata_servicer_load_model_resident_mode():
    """Verify StrataServicer invokes clawback_system_memory and omits --mmap-experts in resident mode."""
    from unittest.mock import patch, MagicMock
    from local_ai_launcher.config import config
    servicer = StrataServicer()
    req = backend_pb2.ModelOptions(Model="strata-iq3s")

    with patch.object(servicer, "_wait_for_health", return_value=True), \
         patch("local_ai_launcher.process.assign_process_to_job"), \
         patch("subprocess.Popen") as mock_popen, \
         patch("local_ai_launcher.backends.strata.ensure_strata_pack", return_value=True), \
         patch("local_ai_launcher.memory_clawback.clawback_system_memory") as mock_clawback, \
         patch.object(config, "strata_mmap_experts", False):

        mock_proc = MagicMock()
        mock_proc.pid = 40002
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        res = servicer.LoadModel(req, None)
        assert res.success is True
        assert mock_clawback.called
        assert mock_popen.called
        servicer.Free(None, None)


def test_strata_build_config_chat_template_preservation(tmp_path):
    """Verify build_strata_config backs up original chat template and toggles cleanly."""
    import json
    from local_ai_launcher.backends.strata import build_strata_config

    pack_dir = tmp_path / "mock_pack"
    tok_dir = pack_dir / "tokenizer"
    tok_dir.mkdir(parents=True, exist_ok=True)
    orig_tmpl_content = "{% stock original template %}"
    custom_tmpl_content = "{% custom tuned template %}"

    (tok_dir / "chat_template.jinja").write_text(orig_tmpl_content, encoding="utf-8")
    custom_file = tmp_path / "custom.jinja"
    custom_file.write_text(custom_tmpl_content, encoding="utf-8")

    dummy_shard = tmp_path / "model.gguf"
    dummy_shard.write_text("gguf", encoding="utf-8")

    # 1. First run with use_custom_chat_template=True -> backs up original and writes custom
    cfg_p1 = build_strata_config(
        model_name="test-strata",
        first_shard=dummy_shard,
        ple_shard=dummy_shard,
        pack_dir=pack_dir,
        mtp_dir=tmp_path,
        port=8089,
        chat_template_file=custom_file,
        use_custom_chat_template=True
    )
    assert (tok_dir / "chat_template.original.jinja").exists()
    assert (tok_dir / "chat_template.original.jinja").read_text(encoding="utf-8") == orig_tmpl_content
    assert (tok_dir / "chat_template.jinja").read_text(encoding="utf-8") == custom_tmpl_content
    cfg1 = json.loads(cfg_p1.read_text(encoding="utf-8"))
    assert cfg1["use_custom_chat_template"] is True
    assert cfg1["chat_template_file"] == custom_file.as_posix()

    # 2. Second run with use_custom_chat_template=False -> restores original
    cfg_p2 = build_strata_config(
        model_name="test-strata",
        first_shard=dummy_shard,
        ple_shard=dummy_shard,
        pack_dir=pack_dir,
        mtp_dir=tmp_path,
        port=8089,
        chat_template_file=custom_file,
        use_custom_chat_template=False
    )
    assert (tok_dir / "chat_template.original.jinja").exists()
    assert (tok_dir / "chat_template.original.jinja").read_text(encoding="utf-8") == orig_tmpl_content
    assert (tok_dir / "chat_template.jinja").read_text(encoding="utf-8") == orig_tmpl_content
    cfg2 = json.loads(cfg_p2.read_text(encoding="utf-8"))
    assert cfg2["use_custom_chat_template"] is False




