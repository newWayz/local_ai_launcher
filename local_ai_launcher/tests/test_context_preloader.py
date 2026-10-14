import json
import time
import urllib.error
from unittest.mock import MagicMock, patch
import pytest

from local_ai_launcher.context_preloader import (
    build_harness_prefix,
    preload_harness_context,
    is_local_endpoint,
    OPENCODE_TOOLS,
    PI_TOOLS,
    HERMES_TOOLS,
    DEEPSEEK_TOOLS
)


def test_is_local_endpoint():
    """Verify local loopback detection for preloading safety."""
    assert is_local_endpoint("http://127.0.0.1:8080/v1") is True
    assert is_local_endpoint("http://localhost:9931/v1") is True
    assert is_local_endpoint("http://0.0.0.0:20128/v1") is True
    assert is_local_endpoint("https://api.openai.com/v1") is False
    assert is_local_endpoint("https://api.anthropic.com") is False
    assert is_local_endpoint(None) is False


def test_build_harness_prefix_opencode_t3():
    """Verify OpenCode / T3 prefix includes expected system message and tools."""
    for h_name in ("t3", "opencode", "t3code"):
        prefix = build_harness_prefix(h_name, "test-model")
        assert "messages" in prefix
        assert len(prefix["messages"]) >= 1
        assert prefix["messages"][0]["role"] == "system"
        assert "OpenCode" in prefix["messages"][0]["content"]

        tool_names = [t["function"]["name"] for t in prefix["tools"]]
        assert "bash" in tool_names
        assert "view" in tool_names
        assert "edit" in tool_names
        assert "patch" in tool_names
        assert "glob" in tool_names
        assert "grep" in tool_names
        assert prefix["tools"] == OPENCODE_TOOLS


def test_build_harness_prefix_pi():
    """Verify Pi prefix includes expected system message and tools."""
    prefix = build_harness_prefix("pi", "test-model")
    assert "messages" in prefix
    assert prefix["messages"][0]["role"] == "system"
    assert "Pi" in prefix["messages"][0]["content"]

    tool_names = [t["function"]["name"] for t in prefix["tools"]]
    assert "read" in tool_names
    assert "write" in tool_names
    assert "edit" in tool_names
    assert "find" in tool_names
    assert "grep" in tool_names
    assert "ls" in tool_names
    assert "bash" in tool_names
    assert prefix["tools"] == PI_TOOLS


def test_build_harness_prefix_hermes():
    """Verify Hermes prefix includes expected system message and tools."""
    for h_name in ("hermes", "hermes-agent"):
        prefix = build_harness_prefix(h_name, "test-model")
        assert "messages" in prefix
        assert prefix["messages"][0]["role"] == "system"
        assert "Hermes" in prefix["messages"][0]["content"]

        tool_names = [t["function"]["name"] for t in prefix["tools"]]
        assert "terminal" in tool_names
        assert ("write_file" in tool_names) or ("file_editor" in tool_names)
        assert prefix["tools"] == HERMES_TOOLS


def test_build_harness_prefix_deepseek():
    """Verify Deepseek / DSH prefix includes expected system message and tools."""
    for h_name in ("deepseek", "dsh"):
        prefix = build_harness_prefix(h_name, "test-model")
        assert "messages" in prefix
        assert prefix["messages"][0]["role"] == "system"
        assert "DeepSeek" in prefix["messages"][0]["content"]

        tool_names = [t["function"]["name"] for t in prefix["tools"]]
        assert "read" in tool_names
        assert "write" in tool_names
        assert "edit" in tool_names
        assert "glob" in tool_names
        assert "grep" in tool_names
        assert "pwsh" in tool_names
        assert "read_image" in tool_names
        assert "ask_user_question" in tool_names
        assert "skill" in tool_names
        assert "todo_write" in tool_names
        assert prefix["tools"] == DEEPSEEK_TOOLS


def test_preload_harness_context_dispatches_http_request():
    """Verify preload_harness_context sends an async HTTP completion request with max_tokens=1."""
    with patch("urllib.request.urlopen") as mock_urlopen, \
         patch("time.sleep"):  # Bypass delay
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"{}"
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        thread = preload_harness_context(
            llm_base_url="http://127.0.0.1:8080/v1",
            model_name="cyber-tiel-coder-35b",
            harness_name="t3",
            delay_seconds=0.0
        )
        assert thread is not None
        thread.join(timeout=2.0)

        assert mock_urlopen.called
        req = mock_urlopen.call_args[0][0]
        assert req.full_url == "http://127.0.0.1:8080/v1/chat/completions"
        assert req.get_header("Content-type") == "application/json"
        assert "HarnessContextPreloader/t3" in req.get_header("User-agent")

        payload = json.loads(req.data.decode("utf-8"))
        assert payload["model"] == "cyber-tiel-coder-35b"
        assert payload["max_tokens"] == 1
        assert payload["temperature"] == 0.0
        assert any(m["role"] == "system" for m in payload["messages"])
        assert any(m["role"] == "user" for m in payload["messages"])
        assert len(payload["tools"]) == len(OPENCODE_TOOLS)


def test_preload_harness_context_skips_remote_endpoints():
    """Verify non-local endpoints are skipped to protect API credits."""
    with patch("urllib.request.urlopen") as mock_urlopen:
        thread = preload_harness_context(
            llm_base_url="https://api.openai.com/v1",
            model_name="gpt-4o",
            harness_name="pi"
        )
        assert thread is None
        assert not mock_urlopen.called


def test_preload_harness_context_fallback_on_400_with_tools():
    """Verify that when backend returns 400 with tools, it retries with messages-only."""
    with patch("urllib.request.urlopen") as mock_urlopen, \
         patch("time.sleep"):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"{}"

        # First call raises HTTPError 400, second call succeeds
        err_400 = urllib.error.HTTPError(
            url="http://127.0.0.1:8080/v1/chat/completions",
            code=400,
            msg="tools not supported",
            hdrs={},
            fp=None
        )
        mock_urlopen.side_effect = [err_400, mock_resp]

        thread = preload_harness_context(
            llm_base_url="http://127.0.0.1:8080/v1",
            model_name="test-model",
            harness_name="t3",
            delay_seconds=0.0
        )
        assert thread is not None
        thread.join(timeout=2.0)

        assert mock_urlopen.call_count == 2
        second_call_req = mock_urlopen.call_args_list[1][0][0]
        second_payload = json.loads(second_call_req.data.decode("utf-8"))
        assert "tools" not in second_payload
        assert any(m["role"] == "system" for m in second_payload["messages"])


def test_preload_harness_context_normalizes_url_without_v1():
    """Verify preload_harness_context automatically appends /v1 when missing from base URL."""
    from local_ai_launcher.context_preloader import clear_preloaded_cache
    clear_preloaded_cache()
    with patch("urllib.request.urlopen") as mock_urlopen, \
         patch("time.sleep"):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"{}"
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        thread = preload_harness_context(
            llm_base_url="http://127.0.0.1:8080",
            model_name="test-norm-model",
            harness_name="pi",
            delay_seconds=0.0
        )
        assert thread is not None
        thread.join(timeout=2.0)

        assert mock_urlopen.called
        req = mock_urlopen.call_args[0][0]
        assert req.full_url == "http://127.0.0.1:8080/v1/chat/completions"


def test_preload_harness_context_maps_omniroute_alias():
    """Verify OmniRoute virtual combo alias ('1-local_first-brain') resolves to physical model."""
    from local_ai_launcher.context_preloader import clear_preloaded_cache
    from local_ai_launcher.config import config
    clear_preloaded_cache()
    with patch("urllib.request.urlopen") as mock_urlopen, \
         patch("time.sleep"):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"{}"
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        thread = preload_harness_context(
            llm_base_url="http://127.0.0.1:8080/v1",
            model_name="1-local_first-brain",
            harness_name="pi",
            delay_seconds=0.0
        )
        assert thread is not None
        thread.join(timeout=2.0)

        assert mock_urlopen.called
        req = mock_urlopen.call_args[0][0]
        payload = json.loads(req.data.decode("utf-8"))
        assert payload["model"] == config.default_model


def test_preload_harness_context_skips_redundant_calls():
    """Verify duplicate preloading for same model & harness is skipped to prevent wasted inference turns."""
    from local_ai_launcher.context_preloader import clear_preloaded_cache
    clear_preloaded_cache()
    with patch("urllib.request.urlopen") as mock_urlopen, \
         patch("time.sleep"):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"{}"
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        thread1 = preload_harness_context(
            llm_base_url="http://127.0.0.1:8080/v1",
            model_name="dedup-model",
            harness_name="hermes",
            delay_seconds=0.0
        )
        assert thread1 is not None
        thread1.join(timeout=2.0)
        assert mock_urlopen.call_count == 1

        thread2 = preload_harness_context(
            llm_base_url="http://127.0.0.1:8080/v1",
            model_name="dedup-model",
            harness_name="hermes",
            delay_seconds=0.0
        )
        assert thread2 is None
        assert mock_urlopen.call_count == 1


def test_preload_default_model_async_embeds_harness_context():
    """Verify process.preload_default_model_async preloads with target harness system prompt & tools."""
    from local_ai_launcher.process import preload_default_model_async
    from local_ai_launcher.context_preloader import clear_preloaded_cache
    clear_preloaded_cache()

    with patch("urllib.request.urlopen") as mock_urlopen, \
         patch("time.sleep"):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = b"{}"
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        preload_thread = preload_default_model_async(
            port=8080,
            model_name="warmup-model",
            timeout=10,
            harness_name="pi"
        )
        preload_thread.join(timeout=2.0)

        assert mock_urlopen.called
        # Find completion request
        call_reqs = [call[0][0] for call in mock_urlopen.call_args_list if hasattr(call[0][0], "data") and call[0][0].data]
        assert len(call_reqs) >= 1
        comp_req = call_reqs[-1]
        payload = json.loads(comp_req.data.decode("utf-8"))
        assert payload["model"] == "warmup-model"
        assert len(payload["tools"]) > 0
        assert any("Pi" in m.get("content", "") for m in payload["messages"] if m.get("role") == "system")
 
 
def test_is_strata_model_detection():
    """Verify detection of Strata engine from model names and presets."""
    from local_ai_launcher.context_preloader import is_strata_model
    assert is_strata_model("qwen-38-flash-strata-iq3s") is True
    assert is_strata_model("strata-iq3s") is True
    assert is_strata_model("qwen-strata-iq3s") is True
    assert is_strata_model("cyber-tiel-coder-35b") is False
    assert is_strata_model(None) is False


def test_preload_harness_context_strata_dynamic_timeout():
    """Verify that Strata models receive an extended timeout (>= 180s) to survive cold arena boot."""
    from local_ai_launcher.context_preloader import clear_preloaded_cache
    clear_preloaded_cache()

    with patch("urllib.request.urlopen") as mock_urlopen, \
         patch("time.sleep"):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"{}"
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        thread = preload_harness_context(
            llm_base_url="http://127.0.0.1:8080/v1",
            model_name="qwen-38-flash-strata-iq3s",
            harness_name="hermes",
            timeout=20.0,
            delay_seconds=0.0
        )
        assert thread is not None
        thread.join(timeout=2.0)

        assert mock_urlopen.called
        timeout_arg = mock_urlopen.call_args[1].get("timeout")
        assert timeout_arg >= 180.0


def test_hermes_prefix_strata_omits_cpu_aux_directives():
    """Verify Hermes prefix for Strata models omits CPU auxiliary microservices directives."""
    prefix_strata = build_harness_prefix("hermes", "qwen-38-flash-strata-iq3s")
    sys_content = prefix_strata["messages"][0]["content"]
    assert "Use auxiliary CPU microservices" not in sys_content

