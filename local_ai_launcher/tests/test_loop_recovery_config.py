import time
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from local_ai_launcher.config import LauncherConfig, config
from local_ai_launcher.harnesses.pi import PiAdapter
from local_ai_launcher.localai.native_server import (
    app,
    _recent_loop_sessions,
    _recent_loop_streaks,
    _record_loop_event,
    _prune_expired_loop_state,
    _resolve_session_key
)
from local_ai_launcher.localai.proto import backend_pb2


@pytest.fixture(autouse=True)
def clean_loop_state():
    """Ensure in-memory loop tracking dictionaries are pristine before and after tests."""
    _recent_loop_sessions.clear()
    _recent_loop_streaks.clear()
    yield
    _recent_loop_sessions.clear()
    _recent_loop_streaks.clear()


def test_loop_recovery_config_defaults():
    """Verify default values for autonomous loop recovery."""
    cfg = LauncherConfig()
    assert cfg.loop_recovery_max_retries == 3
    assert cfg.loop_recovery_temp_reduce == 0.1
    assert cfg.loop_recovery_temp_boost == 0.1
    assert cfg.loop_recovery_presence_penalty == 0.0


def test_loop_recovery_config_toml_flattening(tmp_path):
    """Verify both flat and nested TOML tables load correctly into LauncherConfig."""
    # Test flat keys
    flat_toml = tmp_path / "launcher_flat.toml"
    flat_toml.write_text(
        """
        loop_recovery_max_retries = 5
        loop_recovery_temp_boost = 0.25
        loop_recovery_presence_penalty = 0.35
        """,
        encoding="utf-8"
    )
    cfg_flat = LauncherConfig()
    cfg_flat.load_overrides(flat_toml)
    assert cfg_flat.loop_recovery_max_retries == 5
    assert cfg_flat.loop_recovery_temp_boost == 0.25
    assert cfg_flat.loop_recovery_presence_penalty == 0.35

    # Test [loop_recovery] table
    nested_toml = tmp_path / "launcher_nested.toml"
    nested_toml.write_text(
        """
        [loop_recovery]
        max_retries = 7
        temp_boost = 0.30
        presence_penalty = 0.40
        """,
        encoding="utf-8"
    )
    cfg_nested = LauncherConfig()
    cfg_nested.load_overrides(nested_toml)
    assert cfg_nested.loop_recovery_max_retries == 7
    assert cfg_nested.loop_recovery_temp_boost == 0.30
    assert cfg_nested.loop_recovery_presence_penalty == 0.40

    # Test [loop_detection] table with prefix
    table_toml = tmp_path / "launcher_table.toml"
    table_toml.write_text(
        """
        [loop_detection]
        recovery_max_retries = 6
        recovery_temp_boost = 0.22
        recovery_presence_penalty = 0.28
        """,
        encoding="utf-8"
    )
    cfg_table = LauncherConfig()
    cfg_table.load_overrides(table_toml)
    assert cfg_table.loop_recovery_max_retries == 6
    assert cfg_table.loop_recovery_temp_boost == 0.22
    assert cfg_table.loop_recovery_presence_penalty == 0.28


def test_pi_adapter_loop_recovery_provisioning(tmp_path, monkeypatch):
    """Verify Pi adapter provisions loop-recovery.ts and exports PI_LOOP_RECOVERY_MAX_RETRIES."""
    monkeypatch.setattr(config, "loop_recovery_max_retries", 5)
    monkeypatch.setattr(config, "harness_extensions", {})
    monkeypatch.setattr(config, "harness_skills", {})
    processes = []
    adapter = PiAdapter(tmp_path, processes)
    adapter.pi_home = tmp_path / ".pi"

    # Verify ensure_extensions_and_skills copies loop-recovery.ts
    adapter.ensure_extensions_and_skills()
    target_ext = adapter.pi_home / "agent" / "extensions" / "loop-recovery.ts"
    assert target_ext.exists()
    content = target_ext.read_text(encoding="utf-8")
    assert "PI_LOOP_RECOVERY_MAX_RETRIES" in content
    assert "deliverAs" in content
    assert "compact" in content

    # Verify start() embeds PI_LOOP_RECOVERY_MAX_RETRIES in the launch command
    with patch("local_ai_launcher.harnesses.pi.start_in_wt_tab") as mock_start_tab, \
         patch("local_ai_launcher.harnesses.pi.subprocess.Popen"):
        adapter.start("http://127.0.0.1:9931/v1", "test-model", launch_tui=True)
        assert mock_start_tab.called
        cmd_arg = mock_start_tab.call_args[0][0]
        assert "PI_LOOP_RECOVERY_MAX_RETRIES" in cmd_arg
        assert "5" in cmd_arg


def test_native_server_sampler_perturbation():
    """Verify that native_server perturbs temperature and presence penalty for looped sessions."""
    client = TestClient(app)
    session_id = "test-session-perturb-42"
    _recent_loop_sessions[session_id] = time.time()

    mock_reply = backend_pb2.Reply(
        message=b'{"choices": [{"message": {"role": "assistant", "content": "Recovered response"}}]}'
    )
    mock_stub = MagicMock()
    mock_stub.Predict.return_value = mock_reply

    with patch("local_ai_launcher.localai.native_server.supervisor.ensure_backend_loaded") as mock_ensure, \
         patch("local_ai_launcher.localai.native_server.config") as mock_config:
        
        async def _fake_ensure(m):
            return mock_stub
        mock_ensure.side_effect = _fake_ensure

        mock_config.loop_recovery_temp_reduce = 0.25
        mock_config.loop_recovery_temp_boost = 0.25
        mock_config.loop_recovery_presence_penalty = 0.35
        mock_config.enable_loop_detection = True

        payload = {
            "model": "qwen-38_27b-blackwell-256k-think",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.7,
            "presence_penalty": 0.0,
            "stream": False
        }
        headers = {"x-session-id": session_id}

        resp = client.post("/v1/chat/completions", json=payload, headers=headers)
        assert resp.status_code == 200

        # Verify Predict was called with perturbed Temperature and PresencePenalty
        predict_args = mock_stub.Predict.call_args[0][0]
        # Base 0.7 - 0.25 = 0.45
        assert pytest.approx(predict_args.Temperature, 0.01) == 0.45
        # Max(0.0, 0.35) = 0.35
        assert pytest.approx(predict_args.PresencePenalty, 0.01) == 0.35

        # Verify session was popped from _recent_loop_sessions
        assert session_id not in _recent_loop_sessions


def test_negative_number_hardening_variables_and_flags():
    """Verify that LauncherConfig hardens against negative numbers across variables and flags."""
    from local_ai_launcher.config import MnemosyneAppConfig

    # 1. User sample: direct instantiation with negative numbers
    cfg = LauncherConfig(
        loop_recovery_max_retries=-3,
        loop_recovery_temp_boost=-0.15,
        loop_recovery_presence_penalty=-0.2,
    )
    assert cfg.loop_recovery_max_retries == 0
    assert cfg.loop_recovery_temp_boost == 0.0
    assert cfg.loop_recovery_presence_penalty == 0.0

    # 2. Direct attribute assignments
    cfg.loop_recovery_max_retries = -5
    cfg.loop_recovery_temp_boost = -0.5
    cfg.loop_recovery_presence_penalty = -0.3
    assert cfg.loop_recovery_max_retries == 0
    assert cfg.loop_recovery_temp_boost == 0.0
    assert cfg.loop_recovery_presence_penalty == 0.0

    # 3. String numeric inputs
    cfg.loop_recovery_max_retries = "-3"
    cfg.loop_recovery_temp_boost = "-0.15"
    cfg.loop_recovery_presence_penalty = "-0.2"
    assert cfg.loop_recovery_max_retries == 0
    assert cfg.loop_recovery_temp_boost == 0.0
    assert cfg.loop_recovery_presence_penalty == 0.0

    # 4. Port numbers
    cfg.localai_port = -8080
    assert cfg.localai_port == 8080  # reset to safe default
    cfg.mcp_server_port = 0
    assert cfg.mcp_server_port == 9932

    # 5. Loop detector settings
    cfg.loop_detector_ngram_size = -8
    assert cfg.loop_detector_ngram_size == 8
    cfg.loop_detector_ngram_threshold = -3
    assert cfg.loop_detector_ngram_threshold == 3
    cfg.loop_detector_cycle_similarity = -0.82
    assert cfg.loop_detector_cycle_similarity == 0.82
    cfg.loop_detector_min_sentence_len = -35
    assert cfg.loop_detector_min_sentence_len == 0

    # 6. Boolean flags: negative numbers coerced to False (avoiding Python bool(-1) == True trap)
    cfg.enable_loop_detection = -1
    assert cfg.enable_loop_detection is False
    cfg.enable_loop_detection = "-1"
    assert cfg.enable_loop_detection is False
    cfg.show_llama_server_console = -3
    assert cfg.show_llama_server_console is False
    cfg.use_omniroute_proxy = -1
    assert cfg.use_omniroute_proxy is False

    # 7. Ratios clamped to [0.0, 1.0]
    cfg.pi_compaction_keep_ratio = -0.4
    assert cfg.pi_compaction_keep_ratio == 0.0
    cfg.hermes_compression_threshold_ratio = 1.5
    assert cfg.hermes_compression_threshold_ratio == 1.0

    # 8. MnemosyneAppConfig
    mcfg = MnemosyneAppConfig()
    mcfg.backup_interval_days = -5
    assert mcfg.backup_interval_days == 5
    mcfg.max_backups_to_keep = 0
    assert mcfg.max_backups_to_keep == 10


def test_negative_toml_overrides(tmp_path):
    """Verify that loading negative numbers from TOML files safely clamps them."""
    neg_toml = tmp_path / "launcher_negative.toml"
    neg_toml.write_text(
        """
        loop_recovery_max_retries = -3
        loop_recovery_temp_boost = -0.15
        loop_recovery_presence_penalty = -0.2
        localai_port = -8080
        enable_loop_detection = -1
        pi_compaction_keep_ratio = -0.4
        """,
        encoding="utf-8"
    )
    cfg = LauncherConfig()
    cfg.load_overrides(neg_toml)
    assert cfg.loop_recovery_max_retries == 0
    assert cfg.loop_recovery_temp_boost == 0.0
    assert cfg.loop_recovery_presence_penalty == 0.0
    assert cfg.localai_port == 8080
    assert cfg.enable_loop_detection is False
    assert cfg.pi_compaction_keep_ratio == 0.0


def test_hardening_across_all_modules(tmp_path):
    """Verify hardening against negative numbers and invalid bounds across all subsystems."""
    # 1. GVS5HSolver
    from local_ai_launcher.bridges.gvs5h_solver import GVS5HSolver
    target_file = tmp_path / "solution.py"
    target_file.write_text("print('hello')", encoding="utf-8")
    solver = GVS5HSolver(
        task="test",
        target_file=str(target_file),
        max_iters=-10,
        max_seconds=-100,
        test_timeout=-5.0
    )
    assert solver.max_iters >= 1
    assert solver.max_seconds >= 1
    assert solver.test_timeout >= 0.1

    # 2. StreamingLoopDetector
    from local_ai_launcher.localai.loop_detector import StreamingLoopDetector
    detector = StreamingLoopDetector(
        ngram_size=-4,
        ngram_threshold=-2,
        cycle_sim_threshold=-0.5,
        min_sentence_len=-10,
        max_history_words=-500,
        enabled=-1
    )
    assert detector.ngram_size >= 1
    assert detector.ngram_threshold >= 1
    assert 0.01 <= detector.cycle_sim_threshold <= 1.0
    assert detector.min_sentence_len >= 0
    assert detector.max_history_words >= 50
    assert detector.enabled is False

    # 3. LocalAIManager
    from local_ai_launcher.localai.manager import LocalAIManager
    mgr = LocalAIManager(port=-8080)
    assert mgr.port == 8080
    assert "8080" in mgr.base_url

    # 4. Blackwell llama shim _build_request_payload
    from local_ai_launcher.shims.blackwell_llama_shim import BlackwellLlamaServicer
    b_shim = BlackwellLlamaServicer()
    req = backend_pb2.PredictOptions(
        Temperature=2.5,
        TopP=1.5,
        TopK=10,
        MinP=1.2,
        Penalty=1.5,
        PresencePenalty=-3.0,
        FrequencyPenalty=-3.0,
        Seed=-100,
        Tokens=-50
    )
    payload = b_shim._build_request_payload(req)
    assert payload["temperature"] <= 2.0
    assert payload["top_p"] <= 1.0
    assert payload["min_p"] <= 1.0
    assert payload["presence_penalty"] >= -2.0
    assert payload["frequency_penalty"] >= -2.0

    # 5. FreeToken shim _build_request_payload
    from local_ai_launcher.shims.freetoken_grpc_shim import FreeTokenServicer
    ft_shim = FreeTokenServicer()
    req_ft = backend_pb2.PredictOptions(
        Temperature=-0.5,
        TopP=-0.8,
        PresencePenalty=-4.0,
        FrequencyPenalty=4.0
    )
    ft_payload = ft_shim._build_request_payload(req_ft)
    if "presence_penalty" in ft_payload:
        assert ft_payload["presence_penalty"] >= -2.0
    if "frequency_penalty" in ft_payload:
        assert ft_payload["frequency_penalty"] <= 2.0

    # 6. TabbyAPI shim _build_request_payload
    from local_ai_launcher.shims.tabby_grpc_shim import TabbyAPIServicer
    tabby_shim = TabbyAPIServicer()
    req_tabby = backend_pb2.PredictOptions(
        Temperature=3.0,
        TopP=2.0,
        TopK=-5
    )
    tabby_payload = tabby_shim._build_request_payload(req_tabby)
    assert tabby_payload["temperature"] <= 2.0
    assert tabby_payload["top_p"] <= 1.0

    # 7. Model Configs generate_localai_yaml
    from local_ai_launcher.localai.model_configs import generate_localai_yaml
    doc = generate_localai_yaml("test", "llama-cpp", "test.gguf", context_size=-5000, gpu_layers=-100)
    assert doc["context_size"] >= 512
    assert doc["gpu_layers"] == -1

    # 8. T3 setup config
    from local_ai_launcher.harnesses.t3 import setup_t3_config
    t3_home = tmp_path / ".t3"
    with patch("local_ai_launcher.harnesses.t3.config") as mock_cfg:
        mock_cfg.t3_port = -3773
        mock_cfg.harness_start_dir = tmp_path
        setup_t3_config(t3_home, "http://127.0.0.1:8080/v1", "dummy-model")


def test_loop_recovery_temp_reduce_stacking_and_floor():
    """Verify that consecutive loops increment streak, multiply temp reduction, and clamp to floor."""
    client = TestClient(app)
    session_id = "test-session-stacking-99"

    mock_reply = backend_pb2.Reply(
        message=b'{"choices": [{"message": {"role": "assistant", "content": "Clean text"}}]}'
    )
    mock_stub = MagicMock()
    mock_stub.Predict.return_value = mock_reply

    with patch("local_ai_launcher.localai.native_server.supervisor.ensure_backend_loaded") as mock_ensure, \
         patch("local_ai_launcher.localai.native_server.config") as mock_config:

        async def _fake_ensure(m):
            return mock_stub
        mock_ensure.side_effect = _fake_ensure

        mock_config.loop_recovery_temp_reduce = 0.1
        mock_config.loop_recovery_temp_floor = 0.2
        mock_config.loop_recovery_presence_penalty = 0.0
        mock_config.enable_loop_detection = False

        headers = {"x-session-id": session_id}
        payload = {
            "model": "qwen-38_27b-blackwell-256k-think",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.7,
            "stream": False
        }

        # Consecutive loop events increment the streak
        _record_loop_event(session_id)
        assert _recent_loop_streaks[session_id][1] == 1

        # Turn with streak 1 cools temp by 0.1 (0.7 -> 0.6)
        resp = client.post("/v1/chat/completions", json=payload, headers=headers)
        assert resp.status_code == 200
        predict_args = mock_stub.Predict.call_args[0][0]
        assert pytest.approx(predict_args.Temperature, 0.01) == 0.6

        # Two consecutive loop detections occur (e.g. retries looped again)
        _record_loop_event(session_id)
        assert _recent_loop_streaks[session_id][1] == 1
        _record_loop_event(session_id)
        assert _recent_loop_streaks[session_id][1] == 2

        # Turn with streak 2 cools temp by 0.2 (0.7 -> 0.5)
        resp = client.post("/v1/chat/completions", json=payload, headers=headers)
        assert resp.status_code == 200
        predict_args = mock_stub.Predict.call_args[0][0]
        assert pytest.approx(predict_args.Temperature, 0.01) == 0.5

        # Three consecutive loops occur
        _record_loop_event(session_id)
        _record_loop_event(session_id)
        _record_loop_event(session_id)
        assert _recent_loop_streaks[session_id][1] == 3

        # Turn with streak 3 cools temp by 0.3 (0.7 -> 0.4)
        resp = client.post("/v1/chat/completions", json=payload, headers=headers)
        assert resp.status_code == 200
        predict_args = mock_stub.Predict.call_args[0][0]
        assert pytest.approx(predict_args.Temperature, 0.01) == 0.4

        # 6 consecutive loops -> reaches floor (0.7 - 0.6 = 0.1, clamped to floor 0.2)
        for _ in range(6):
            _record_loop_event(session_id)
        assert _recent_loop_streaks[session_id][1] == 6

        resp = client.post("/v1/chat/completions", json=payload, headers=headers)
        assert resp.status_code == 200
        predict_args = mock_stub.Predict.call_args[0][0]
        assert pytest.approx(predict_args.Temperature, 0.01) == 0.2


def test_loop_recovery_streak_reset_on_clean_completion():
    """Verify that clean non-streaming and streaming completions reset the loop streak."""
    client = TestClient(app)
    session_id = "test-session-reset-12"

    # Set up a prior loop streak
    _record_loop_event(session_id)
    assert session_id in _recent_loop_streaks

    mock_reply = backend_pb2.Reply(
        message=b'{"choices": [{"message": {"role": "assistant", "content": "Clean without loops"}}]}'
    )
    mock_stub = MagicMock()
    mock_stub.Predict.return_value = mock_reply

    with patch("local_ai_launcher.localai.native_server.supervisor.ensure_backend_loaded") as mock_ensure, \
         patch("local_ai_launcher.localai.native_server.config") as mock_config:

        async def _fake_ensure(m):
            return mock_stub
        mock_ensure.side_effect = _fake_ensure

        mock_config.loop_recovery_temp_reduce = 0.1
        mock_config.loop_recovery_temp_floor = 0.15
        mock_config.enable_loop_detection = True

        headers = {"x-session-id": session_id}
        payload = {
            "model": "qwen-38_27b-blackwell-256k-think",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.7,
            "stream": False
        }

        # Non-streaming clean response resets streak
        resp = client.post("/v1/chat/completions", json=payload, headers=headers)
        assert resp.status_code == 200
        assert session_id not in _recent_loop_streaks


def test_loop_recovery_config_aliasing_and_floor_toml(tmp_path):
    """Verify bi-directional aliasing of temp_reduce / temp_boost and TOML loading for temp_floor."""
    cfg = LauncherConfig()
    cfg.loop_recovery_temp_reduce = 0.2
    assert cfg.loop_recovery_temp_boost == 0.2

    cfg.loop_recovery_temp_boost = 0.3
    assert cfg.loop_recovery_temp_reduce == 0.3

    cfg.loop_recovery_temp_floor = -0.5
    assert cfg.loop_recovery_temp_floor == 0.0

    # TOML table [loop_recovery] with floor and reduce
    toml_path = tmp_path / "launcher_loop.toml"
    toml_path.write_text(
        """
        [loop_recovery]
        temp_reduce = 0.12
        temp_floor = 0.18
        max_retries = 4
        """,
        encoding="utf-8"
    )
    cfg2 = LauncherConfig()
    cfg2.load_overrides(toml_path)
    assert cfg2.loop_recovery_temp_reduce == 0.12
    assert cfg2.loop_recovery_temp_boost == 0.12
    assert cfg2.loop_recovery_temp_floor == 0.18
    assert cfg2.loop_recovery_max_retries == 4


def test_low_temp_recovery_never_heats_up():
    """Verify that when temperature is 0.0 or below temp_floor, loop recovery never raises it to temp_floor."""
    client = TestClient(app)
    session_id = "test-session-greedy-00"

    mock_reply = backend_pb2.Reply(
        message=b'{"choices": [{"message": {"role": "assistant", "content": "Deterministic text"}}]}'
    )
    mock_stub = MagicMock()
    mock_stub.Predict.return_value = mock_reply

    with patch("local_ai_launcher.localai.native_server.supervisor.ensure_backend_loaded") as mock_ensure, \
         patch("local_ai_launcher.localai.native_server.config") as mock_config:

        async def _fake_ensure(m):
            return mock_stub
        mock_ensure.side_effect = _fake_ensure

        mock_config.loop_recovery_temp_reduce = 0.1
        mock_config.loop_recovery_temp_floor = 0.15
        mock_config.loop_recovery_presence_penalty = 0.0
        mock_config.enable_loop_detection = False

        headers = {"x-session-id": session_id}

        # 1. Test greedy decoding: temp = 0.0
        _record_loop_event(session_id)
        payload_greedy = {
            "model": "qwen-38_27b-blackwell-256k-think",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.0,
            "stream": False
        }
        resp = client.post("/v1/chat/completions", json=payload_greedy, headers=headers)
        assert resp.status_code == 200
        predict_args = mock_stub.Predict.call_args[0][0]
        # Temperature must remain 0.0, NEVER boosted to temp_floor 0.15
        assert predict_args.Temperature == 0.0

        # 2. Test low temp below floor: temp = 0.05
        _record_loop_event(session_id)
        payload_low = {
            "model": "qwen-38_27b-blackwell-256k-think",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.05,
            "stream": False
        }
        resp = client.post("/v1/chat/completions", json=payload_low, headers=headers)
        assert resp.status_code == 200
        predict_args = mock_stub.Predict.call_args[0][0]
        # Temperature must be <= 0.05 (clamped to effective_floor 0.05 or 0.0), NEVER boosted to 0.15
        assert round(predict_args.Temperature, 4) <= 0.05


def test_loop_state_ttl_pruning_and_capacity():
    """Verify that expired loop entries are pruned and dictionary size is capped."""
    now = time.time()
    
    # 1. Populate with expired items (> 180s ago)
    _recent_loop_sessions["old_sess_1"] = now - 200.0
    _recent_loop_sessions["old_sess_2"] = now - 300.0
    _recent_loop_sessions["live_sess"] = now - 10.0

    _recent_loop_streaks["old_sess_1"] = (now - 200.0, 3)
    _recent_loop_streaks["live_sess"] = (now - 10.0, 1)

    _prune_expired_loop_state(now, max_entries=500)

    assert "old_sess_1" not in _recent_loop_sessions
    assert "old_sess_2" not in _recent_loop_sessions
    assert "live_sess" in _recent_loop_sessions
    assert "old_sess_1" not in _recent_loop_streaks
    assert "live_sess" in _recent_loop_streaks

    # 2. Capacity cap test
    for i in range(20):
        _recent_loop_sessions[f"cap_sess_{i}"] = now - (20 - i)
        _recent_loop_streaks[f"cap_sess_{i}"] = (now - (20 - i), 1)

    # Prune with max_entries=5
    _prune_expired_loop_state(now, max_entries=5)
    assert len(_recent_loop_sessions) <= 5
    assert len(_recent_loop_streaks) <= 5


def test_resolve_session_key_ignores_request_id():
    """Verify that _resolve_session_key does not bind to per-request x-request-id and honors true session headers."""
    from starlette.datastructures import Headers

    class DummyClient:
        host = "192.168.1.50"

    class DummyRequest:
        def __init__(self, headers_dict, host="192.168.1.50"):
            self.headers = Headers(headers_dict)
            self.client = DummyClient()
            self.client.host = host

    # 1. Request with only x-request-id should NOT use x-request-id (falls back to client host)
    req1 = DummyRequest({"x-request-id": "req-uuid-12345"})
    key1 = _resolve_session_key({}, req1)
    assert key1 == "client_192.168.1.50"
    assert "req-uuid" not in key1

    # 2. Request with x-session-id
    req2 = DummyRequest({"x-session-id": "sess-pi-abc", "x-request-id": "req-uuid-9999"})
    assert _resolve_session_key({}, req2) == "sess-pi-abc"

    # 3. Request with x-conversation-id
    req3 = DummyRequest({"x-conversation-id": "conv-agent-42"})
    assert _resolve_session_key({}, req3) == "conv-agent-42"

    # 4. Request with x-thread-id
    req4 = DummyRequest({"x-thread-id": "thread-777"})
    assert _resolve_session_key({}, req4) == "thread-777"




