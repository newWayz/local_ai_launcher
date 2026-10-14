import os
from unittest.mock import MagicMock, patch
import pytest

from local_ai_launcher.memory_clawback import (
    evict_auxiliary_services,
    trim_process_working_sets,
    terminate_non_essential_applications,
    clawback_system_memory,
)
from local_ai_launcher.backends.strata import build_strata_config
from local_ai_launcher.presets import load_effective_models_parser
from local_ai_launcher.localai.model_configs import generate_all_model_configs


def test_trim_process_working_sets_safe_execution():
    """Verify trim_process_working_sets runs safely on Windows without unhandled exceptions."""
    trimmed = trim_process_working_sets()
    assert isinstance(trimmed, int)
    assert trimmed >= 0


def test_evict_auxiliary_services_targets_matching_ports():
    """Verify evict_auxiliary_services triggers stop hooks and force_kill_port for all aux ports."""
    with patch("local_ai_launcher.backends.cpu_reranker.stop_cpu_reranker_server") as mock_stop_reranker, \
         patch("local_ai_launcher.backends.cpu_onnx_gater.stop_cpu_onnx_gater_server") as mock_stop_gater, \
         patch("local_ai_launcher.process.force_kill_port") as mock_kill_port:

        res = evict_auxiliary_services()
        assert res.get("cpu_reranker") is True
        assert res.get("cpu_onnx_gater") is True
        mock_stop_reranker.assert_called_once()
        mock_stop_gater.assert_called_once()
        # Verify force_kill_port was invoked on ports 8082, 8083, 8084, 8081
        called_ports = [call[0][0] for call in mock_kill_port.call_args_list]
        assert 8082 in called_ports
        assert 8083 in called_ports
        assert 8084 in called_ports
        assert 8081 in called_ports
        # Verify Parakeet NPU (Port 8000) and SearXNG (Port 8088) are preserved and never killed
        assert 8000 not in called_ports
        assert 8088 not in called_ports


def test_terminate_non_essential_applications_protection_and_termination():
    """Verify protected applications (explorer, python, system) are preserved while target bloat apps are terminated."""
    mock_explorer = MagicMock()
    mock_explorer.info = {"pid": 1001, "name": "explorer.exe"}

    mock_current_python = MagicMock()
    mock_current_python.info = {"pid": os.getpid(), "name": "python.exe"}

    mock_chrome = MagicMock()
    mock_chrome.info = {"pid": 2002, "name": "chrome.exe"}

    mock_discord = MagicMock()
    mock_discord.info = {"pid": 2003, "name": "discord.exe"}

    with patch("psutil.process_iter", return_value=[mock_explorer, mock_current_python, mock_chrome, mock_discord]), \
         patch("psutil.Process") as mock_proc_cls:
        mock_curr = MagicMock()
        mock_curr.parents.return_value = []
        mock_curr.children.return_value = []
        mock_proc_cls.return_value = mock_curr

        terminated = terminate_non_essential_applications()
        assert len(terminated) == 2
        assert any("chrome.exe" in t for t in terminated)
        assert any("discord.exe" in t for t in terminated)
        mock_chrome.kill.assert_called_once()
        mock_discord.kill.assert_called_once()
        mock_explorer.kill.assert_not_called()
        mock_current_python.kill.assert_not_called()


def test_clawback_system_memory_modes():
    """Verify brutal=False does not terminate non-essential apps, while brutal=True does."""
    with patch("local_ai_launcher.memory_clawback.evict_auxiliary_services", return_value={"cpu_reranker": True}) as mock_evict, \
         patch("local_ai_launcher.memory_clawback.terminate_non_essential_applications", return_value=["chrome.exe (PID: 2002)"]) as mock_terminate, \
         patch("local_ai_launcher.memory_clawback.trim_process_working_sets", return_value=12) as mock_trim:

        # Non-brutal / Light clawback
        res_gentle = clawback_system_memory(brutal=False)
        assert res_gentle["trimmed_count"] == 12
        assert res_gentle["terminated_apps"] == []
        mock_terminate.assert_not_called()

        # Explicit light mode
        res_light = clawback_system_memory(mode="light")
        assert res_light["trimmed_count"] == 12
        assert res_light["terminated_apps"] == []
        mock_terminate.assert_not_called()

        # Brutal / Max clawback
        res_brutal = clawback_system_memory(brutal=True)
        assert res_brutal["trimmed_count"] == 12
        assert len(res_brutal["terminated_apps"]) == 1
        mock_terminate.assert_called_once()

        # Explicit max mode
        mock_terminate.reset_mock()
        res_max = clawback_system_memory(mode="max")
        assert res_max["trimmed_count"] == 12
        assert len(res_max["terminated_apps"]) == 1
        mock_terminate.assert_called_once()

        # Off mode
        mock_terminate.reset_mock()
        res_off = clawback_system_memory(mode="off")
        assert res_off["trimmed_count"] == 0
        assert res_off["terminated_apps"] == []
        mock_terminate.assert_not_called()


def test_build_strata_config_mmap_flag_toggle(tmp_path):
    """Verify build_strata_config correctly toggles --mmap-experts based on argument."""
    import json
    shard1 = tmp_path / "model.gguf"
    shard1.touch()
    ple = tmp_path / "ple.gguf"
    ple.touch()
    pack_dir = tmp_path / "pack"
    pack_dir.mkdir()
    mtp = tmp_path / "mtp"
    mtp.mkdir()

    # mmap_experts = True
    cfg_file_mmap = build_strata_config(
        model_name="test-model",
        first_shard=shard1,
        ple_shard=ple,
        pack_dir=pack_dir,
        mtp_dir=mtp,
        port=8080,
        mmap_experts=True,
    )
    with open(cfg_file_mmap, "r", encoding="utf-8") as f:
        data_mmap = json.load(f)
    assert "--mmap-experts" in data_mmap["args"]

    # mmap_experts = False
    cfg_file_resident = build_strata_config(
        model_name="test-model",
        first_shard=shard1,
        ple_shard=ple,
        pack_dir=pack_dir,
        mtp_dir=mtp,
        port=8080,
        mmap_experts=False,
    )
    with open(cfg_file_resident, "r", encoding="utf-8") as f:
        data_resident = json.load(f)
    assert "--mmap-experts" not in data_resident["args"]


def test_models_ini_and_config_strata_settings(tmp_path):
    """Verify models.ini is decoupled from Strata, strata_models.json contains presets, and config/TOML handles strata settings."""
    from local_ai_launcher.config import LauncherConfig
    from local_ai_launcher.backends.strata import load_strata_model_registry

    # 1. Verify models.ini does NOT contain legacy Strata engine sections (clean decoupling)
    parser = load_effective_models_parser()
    assert not parser.has_section("qwen-38-flash-strata-iq3s"), "models.ini must not contain Strata sections"

    # 2. Verify native Strata model registry loads the model presets cleanly
    registry = load_strata_model_registry()
    assert "qwen-38-flash-strata-iq3s" in registry, "Strata model registry must contain qwen-38-flash-strata-iq3s"
    sec = registry["qwen-38-flash-strata-iq3s"]
    assert "mmap-experts" not in sec
    assert "kv" not in sec
    assert "kv-mode" not in sec

    # Verify default LauncherConfig settings
    cfg = LauncherConfig()
    assert cfg.strata_mmap_experts is True
    assert cfg.strata_brutal_ram_clawback == "light"
    assert cfg.strata_kv_mode == "int8"

    # Verify YAML generation does not pollute engine_args with mmap-experts or strata-specific kv
    yamls = generate_all_model_configs(target_dir=tmp_path, force=True)
    strata_yaml = next((y for y in yamls if "strata" in y.stem.lower()), None)
    assert strata_yaml is not None
    import yaml
    with open(strata_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert "mmap-experts" not in data.get("engine_args", {})
    assert "kv" not in data.get("engine_args", {})
    assert data.get("parameters", {}).get("mmap_experts") is True


def test_strata_kv_mode_configuration_and_sanitization(tmp_path):
    """Verify strata_kv_mode defaults to int8, parses TOML table/top-level overrides, and rejects invalid modes."""
    import json
    from local_ai_launcher.config import LauncherConfig
    from local_ai_launcher.backends.strata import build_strata_config

    # 1. Default is int8
    cfg = LauncherConfig()
    assert cfg.strata_kv_mode == "int8"

    # 2. Top-level TOML override
    toml_top = tmp_path / "top.toml"
    toml_top.write_text('strata_kv_mode = "q4_0"\n', encoding="utf-8")
    cfg_top = LauncherConfig(config_file=toml_top)
    assert cfg_top.strata_kv_mode == "q4_0"

    # 3. [strata] table override with kv_mode
    toml_sec = tmp_path / "sec.toml"
    toml_sec.write_text('[strata]\nkv_mode = "k8v4"\n', encoding="utf-8")
    cfg_sec = LauncherConfig(config_file=toml_sec)
    assert cfg_sec.strata_kv_mode == "k8v4"

    # 4. [strata] table override with kv
    toml_kv = tmp_path / "kv.toml"
    toml_kv.write_text('[strata]\nkv = "fp16"\n', encoding="utf-8")
    cfg_kv = LauncherConfig(config_file=toml_kv)
    assert cfg_kv.strata_kv_mode == "fp16"

    # 5. Invalid mode falls back safely to int8 (e.g., fp8 is not supported by Strata C++/CUDA engine)
    toml_invalid = tmp_path / "invalid.toml"
    toml_invalid.write_text('strata_kv_mode = "fp8"\n', encoding="utf-8")
    cfg_invalid = LauncherConfig(config_file=toml_invalid)
    assert cfg_invalid.strata_kv_mode == "int8"

    # 6. Verify build_strata_config injects --kv <mode> into args
    dummy_first = tmp_path / "model-00001-of-00002.gguf"
    dummy_first.write_text("dummy", encoding="utf-8")
    pack_dir = tmp_path / "pack"
    pack_dir.mkdir()
    (pack_dir / "tokenizer").mkdir()

    cfg_file = build_strata_config(
        model_name="test-strata",
        first_shard=dummy_first,
        ple_shard=None,
        pack_dir=pack_dir,
        mtp_dir=None,
        port=9999,
        max_context=262144,
        kv_mode="q4_0"
    )
    with open(cfg_file, "r", encoding="utf-8") as f:
        built_cfg = json.load(f)
    assert "--kv" in built_cfg["args"]
    kv_idx = built_cfg["args"].index("--kv")
    assert built_cfg["args"][kv_idx + 1] == "q4_0"


def test_strata_resident_budget_gib(tmp_path):
    """Verify strata_resident_budget_gib defaults to 52.0, handles TOML overrides, and injects flags into build_strata_config."""
    import json
    from local_ai_launcher.config import LauncherConfig
    from local_ai_launcher.backends.strata import build_strata_config

    # 1. Default is 52.0
    cfg = LauncherConfig()
    assert cfg.strata_resident_budget_gib == 52.0

    # 2. Top-level TOML override
    toml_top = tmp_path / "top_budget.toml"
    toml_top.write_text('strata_resident_budget_gib = 40\n', encoding="utf-8")
    cfg_top = LauncherConfig(config_file=toml_top)
    assert cfg_top.strata_resident_budget_gib == 40.0

    # 3. [strata] table override
    toml_sec = tmp_path / "sec_budget.toml"
    toml_sec.write_text('[strata]\nresident_budget_gib = 36\n', encoding="utf-8")
    cfg_sec = LauncherConfig(config_file=toml_sec)
    assert cfg_sec.strata_resident_budget_gib == 36.0

    # 4. [strata] table override with resident_budget alias
    toml_alias = tmp_path / "alias_budget.toml"
    toml_alias.write_text('[strata]\nresident_budget = 48.5\n', encoding="utf-8")
    cfg_alias = LauncherConfig(config_file=toml_alias)
    assert cfg_alias.strata_resident_budget_gib == 48.5

    # 5. Disable budget (0.0)
    toml_zero = tmp_path / "zero_budget.toml"
    toml_zero.write_text('strata_resident_budget_gib = 0\n', encoding="utf-8")
    cfg_zero = LauncherConfig(config_file=toml_zero)
    assert cfg_zero.strata_resident_budget_gib == 0.0

    # 6. Verify build_strata_config injects --resident-budget-gib 52 and --mmap-experts
    dummy_first = tmp_path / "model2-00001-of-00002.gguf"
    dummy_first.write_text("dummy", encoding="utf-8")
    pack_dir = tmp_path / "pack2"
    pack_dir.mkdir()
    (pack_dir / "tokenizer").mkdir()

    cfg_file = build_strata_config(
        model_name="test-strata-budget",
        first_shard=dummy_first,
        ple_shard=None,
        pack_dir=pack_dir,
        mtp_dir=None,
        port=9998,
        mmap_experts=True,
        resident_budget_gib=52.0
    )
    with open(cfg_file, "r", encoding="utf-8") as f:
        built_cfg = json.load(f)
    assert "--resident-budget-gib" in built_cfg["args"]
    idx = built_cfg["args"].index("--resident-budget-gib")
    assert built_cfg["args"][idx + 1] == "52"
    assert "--mmap-experts" in built_cfg["args"]

    # 7. When resident_budget_gib is 0, --resident-budget-gib is omitted
    cfg_file_zero = build_strata_config(
        model_name="test-strata-zero",
        first_shard=dummy_first,
        ple_shard=None,
        pack_dir=pack_dir,
        mtp_dir=None,
        port=9997,
        mmap_experts=True,
        resident_budget_gib=0.0
    )
    with open(cfg_file_zero, "r", encoding="utf-8") as f:
        built_zero = json.load(f)
    assert "--resident-budget-gib" not in built_zero["args"]
    assert "--mmap-experts" in built_zero["args"]

    # 8. Safe fallback if expert-profile.bin is missing
    import local_ai_launcher.backends.strata as strata_mod
    orig_strata_dir = strata_mod.STRATA_DIR
    try:
        fake_dir = tmp_path / "fake_strata"
        fake_dir.mkdir()
        (fake_dir / "engine").mkdir()
        fake_exe = fake_dir / "engine" / "strata.exe"
        fake_exe.touch()
        strata_mod.STRATA_DIR = fake_dir
        strata_mod.STRATA_EXE = fake_exe
        cfg_file_no_profile = build_strata_config(
            model_name="test-strata-no-profile",
            first_shard=dummy_first,
            ple_shard=None,
            pack_dir=pack_dir,
            mtp_dir=None,
            port=9996,
            mmap_experts=True,
            resident_budget_gib=52.0
        )
        with open(cfg_file_no_profile, "r", encoding="utf-8") as f:
            built_no_profile = json.load(f)
        assert "--resident-budget-gib" not in built_no_profile["args"]
        assert "--mmap-experts" in built_no_profile["args"]
    finally:
        strata_mod.STRATA_DIR = orig_strata_dir
        strata_mod.STRATA_EXE = orig_strata_dir / "engine" / "strata.exe"


def test_strata_settings_json_overrides(tmp_path):
    """Verify strata_settings.json overrides MTP spec, draft probability, and memory tunables in build_strata_config."""
    import json
    from local_ai_launcher.backends.strata import build_strata_config, load_strata_settings
    shard1 = tmp_path / "model.gguf"
    shard1.touch()
    pack_dir = tmp_path / "pack"
    pack_dir.mkdir()
    mtp = tmp_path / "mtp"
    mtp.mkdir()

    custom_json = tmp_path / "strata_settings.json"
    custom_json.write_text(json.dumps({
        "spec": 6,
        "spec_min_p": 0.35,
        "suffix_draft": 2,
        "coupled_draft": True,
        "kv_mode": "q4_0",
        "resident_budget_gib": 48
    }), encoding="utf-8")

    loaded = load_strata_settings(settings_file=custom_json)
    assert loaded["spec"] == 6
    assert loaded["spec_min_p"] == 0.35
    assert loaded["kv_mode"] == "q4_0"

    cfg_file = build_strata_config(
        model_name="test-custom-strata",
        first_shard=shard1,
        ple_shard=None,
        pack_dir=pack_dir,
        mtp_dir=mtp,
        port=8089,
        settings_file=custom_json
    )
    with open(cfg_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    args = data["args"]
    assert "--spec" in args
    spec_idx = args.index("--spec")
    assert args[spec_idx + 1] == "6"

    assert "--spec-min-p" in args
    min_p_idx = args.index("--spec-min-p")
    assert args[min_p_idx + 1] == "0.35"

    assert "--kv" in args
    kv_idx = args.index("--kv")
    assert args[kv_idx + 1] == "q4_0"

    assert "--suffix-draft" in args
    assert "--coupled-draft" in args


def test_is_strata_active_probes(monkeypatch):
    """Verify is_strata_active correctly detects active Strata state across supervisor, ports, and processes."""
    from local_ai_launcher.memory_clawback import is_strata_active
    from local_ai_launcher.config import config

    # 1. Bypass disabled
    monkeypatch.setattr(config, "strata_bypass_aux_models", False)
    assert is_strata_active() is False

    monkeypatch.setattr(config, "strata_bypass_aux_models", True)

    # 2. Config primary engine set to strata
    monkeypatch.setattr(config, "blackwell_engine", "strata")
    assert is_strata_active() is True
    monkeypatch.setattr(config, "blackwell_engine", "huang")

    # 3. Active supervisor backend name
    from local_ai_launcher.localai.native_server import supervisor
    monkeypatch.setattr(supervisor, "active_backend_name", "strata")
    assert is_strata_active() is True
    monkeypatch.setattr(supervisor, "active_backend_name", "blackwell-llama")

    # 4. Inactive when engine is blackwell and no strata process/port
    with patch("socket.socket") as mock_sock, patch("psutil.process_iter", return_value=[]):
        mock_instance = MagicMock()
        mock_instance.connect_ex.return_value = 111  # Connection refused
        mock_sock.return_value.__enter__.return_value = mock_instance
        assert is_strata_active() is False


def test_strata_bypasses_cpu_models_in_gateway():
    """Verify gateway endpoints refuse or pass-through CPU models when Strata is active."""
    from fastapi.testclient import TestClient
    from local_ai_launcher.localai.native_server import app

    client = TestClient(app)

    with patch("local_ai_launcher.localai.native_server.is_strata_active", return_value=True):
        # 1. Rerank returns unranked pass-through documents with warning
        resp_rerank = client.post(
            "/v1/rerank",
            json={"query": "test query", "documents": ["doc1", "doc2", "doc3"], "top_n": 2}
        )
        assert resp_rerank.status_code == 200
        data_rerank = resp_rerank.json()
        assert len(data_rerank["results"]) == 2
        assert "warning" in data_rerank

        # 2. Decide returns 503 (CPU Gater offline for Strata)
        resp_decide = client.post("/v1/decide", json={"prompt": "should we proceed?"})
        assert resp_decide.status_code == 503

        # 3. Tripwire returns ALLOW pass-through
        resp_tripwire = client.post("/v1/tripwire", json={"prompt": "echo hello"})
        assert resp_tripwire.status_code == 200
        assert resp_tripwire.json()["action"] == "ALLOW"

        # 4. Decider endpoints return 503
        resp_slots = client.post("/v1/decider/slots", json={"context": "task", "questions": {}})
        assert resp_slots.status_code == 503

        resp_dom = client.post("/v1/decider/dom", json={"dom_state": "<button>", "instruction": "click"})
        assert resp_dom.status_code == 503

        resp_prior = client.post("/v1/decider/policy_prior", json={"state": "state", "candidate_actions": ["a"]})
        assert resp_prior.status_code == 503

        # 5. Decider info returns offline status
        resp_info = client.get("/v1/decider/info")
        assert resp_info.status_code == 200
        assert resp_info.json()["status"] == "disabled_by_strata_memory_policy"


def test_strata_bypasses_cpu_in_dispatcher():
    """Verify dispatcher skips reranking, slots, and ONNX gating when Strata is active."""
    from local_ai_launcher.dispatcher import _build_dispatch_context, smart_agent_dispatch

    with patch("local_ai_launcher.dispatcher.is_strata_active", return_value=True), \
         patch("local_ai_launcher.backends.cpu_reranker.rerank") as mock_rerank:

        # 1. _build_dispatch_context bypasses reranker
        raw_docs = ["doc A", "doc B", "doc C"]
        ctx, filtered, rerank_ms = _build_dispatch_context(query="test", raw_documents=raw_docs, diff_context="")
        mock_rerank.assert_not_called()
        assert len(filtered) == 3
        assert rerank_ms >= 0.0

        # 2. smart_agent_dispatch bypasses CPU tiers and routes directly to GPU deep reasoning
        with patch("local_ai_launcher.dispatcher.get_shared_sync_client") as mock_client_cls, \
             patch("local_ai_launcher.backends.cpu_onnx_gater.evaluate_onnx_decision_sync") as mock_onnx, \
             patch("local_ai_launcher.backends.cpu_decider.evaluate_decider_slots_sync") as mock_decider:

            mock_client = MagicMock()
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"choices": [{"message": {"content": "GPU reasoning response"}}]}
            mock_client.post.return_value = mock_resp
            mock_client_cls.return_value = mock_client

            res = smart_agent_dispatch(
                task_type="classification",
                query="classify this",
                slots={"intent": ["A", "B"]},
                candidate_options=["PASS", "FAIL"]
            )
            mock_onnx.assert_not_called()
            mock_decider.assert_not_called()
            assert res["action"] == "GPU reasoning response"



