import json
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from local_ai_launcher.localai.native_server import app, _normalize_content
from local_ai_launcher.localai.proto import backend_pb2
from local_ai_launcher.bridges.orca_openlumara.shim import resolve_llm_endpoint
from local_ai_launcher.config import config


def test_normalize_content():
    # String
    assert _normalize_content("hello") == "hello"

    # None
    assert _normalize_content(None) == ""

    # Pi multimodal list format
    pi_blocks = [
        {"type": "text", "text": "What is the capital of France?"}
    ]
    assert _normalize_content(pi_blocks) == "What is the capital of France?"

    # Mixed text and image blocks
    images = []
    mixed_blocks = [
        {"type": "text", "text": "Describe this image:"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo..."}},
        {"type": "text", "text": "in detail."}
    ]
    res = _normalize_content(mixed_blocks, all_images=images)
    assert "Describe this image:" in res
    assert "in detail." in res
    assert len(images) == 1
    assert images[0] == "data:image/png;base64,iVBORw0KGgo..."

    # Pure strings in list
    assert _normalize_content(["line 1", "line 2"]) == "line 1\nline 2"


def test_protobuf_message_creation_with_pi_payload():
    """Verify that protobuf backend_pb2.Message can be instantiated without TypeError."""
    pi_messages = [
        {"role": "system", "content": "You are Pi."},
        {"role": "user", "content": [{"type": "text", "text": "Hello!"}]},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "read"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "file data"}
    ]

    proto_msgs = []
    for m in pi_messages:
        content_str = _normalize_content(m.get("content"))
        name = str(m.get("name") or "")
        tool_call_id = str(m.get("tool_call_id") or "")
        reasoning_content = str(m.get("reasoning_content") or "")
        raw_tc = m.get("tool_calls")
        tc_str = json.dumps(raw_tc) if isinstance(raw_tc, (list, dict)) else (str(raw_tc) if raw_tc else "")

        msg = backend_pb2.Message(
            role=str(m.get("role") or "user"),
            content=content_str,
            name=name,
            tool_call_id=tool_call_id,
            reasoning_content=reasoning_content,
            tool_calls=tc_str
        )
        proto_msgs.append(msg)

    assert len(proto_msgs) == 4
    po = backend_pb2.PredictOptions(Messages=proto_msgs)
    serialized = po.SerializeToString()
    assert len(serialized) > 0


def test_native_server_chat_completions_pi_payload():
    """Test POST /v1/chat/completions with Pi's multimodal list payload and verify no 500 error."""
    client = TestClient(app)

    # Mock supervisor.ensure_backend_loaded to return a mock stub
    mock_stub = MagicMock()
    mock_reply = backend_pb2.Reply(
        message=b"Paris is the capital of France.",
        tokens=8,
        prompt_tokens=15
    )
    mock_stub.Predict.return_value = mock_reply

    with patch("local_ai_launcher.localai.native_server.supervisor.ensure_backend_loaded") as mock_ensure:
        async def _fake_ensure(model_name):
            return mock_stub
        mock_ensure.side_effect = _fake_ensure

        # Pi ADE request payload
        payload = {
            "model": "qwen-38_27b-blackwell-256k-think",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": [{"type": "text", "text": "What is the capital of France?"}]},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "search"}}]}
            ],
            "temperature": 0.7,
            "max_tokens": None,  # Should not cause TypeError
            "stream": False
        }

        resp = client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert data["choices"][0]["message"]["content"] == "Paris is the capital of France."
        assert data["usage"]["completion_tokens"] == 8


def test_resolve_llm_endpoint_uses_localai_port(monkeypatch):
    """Test that shim and bridge fall back to config.localai_port (8080) instead of 9931."""
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    with patch("local_ai_launcher.bridges.orca_openlumara.shim.is_port_in_use") as mock_port:
        def side_effect(port):
            return port == config.localai_port  # 8080 is open

        mock_port.side_effect = side_effect
        endpoint = resolve_llm_endpoint()
        assert endpoint == f"http://localhost:{config.localai_port}/v1"


def test_native_server_chat_completions_tool_calling_non_streaming():
    """Verify non-streaming tool calling returns OpenAI function calling format with finish_reason: 'tool_calls'."""
    client = TestClient(app)
    mock_stub = MagicMock()

    # Raw response from backend (e.g. llama-server / tabby / freetoken)
    raw_llama_resp = {
        "id": "chatcmpl-test1234",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_abc123",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": "{\"location\": \"Tokyo\"}"
                            }
                        }
                    ]
                },
                "finish_reason": "tool_calls"
            }
        ],
        "usage": {"prompt_tokens": 20, "completion_tokens": 15, "total_tokens": 35}
    }

    mock_reply = backend_pb2.Reply(
        message=json.dumps(raw_llama_resp).encode("utf-8"),
        tokens=15,
        prompt_tokens=20
    )
    mock_stub.Predict.return_value = mock_reply

    with patch("local_ai_launcher.localai.native_server.supervisor.ensure_backend_loaded") as mock_ensure:
        async def _fake_ensure(model_name):
            return mock_stub
        mock_ensure.side_effect = _fake_ensure

        payload = {
            "model": "qwen-38_27b-blackwell-256k-think",
            "messages": [{"role": "user", "content": "What's the weather in Tokyo?"}],
            "tools": [{"type": "function", "function": {"name": "get_weather"}}],
            "stream": False
        }
        resp = client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["finish_reason"] == "tool_calls"
        assert len(data["choices"][0]["message"]["tool_calls"]) == 1
        tc = data["choices"][0]["message"]["tool_calls"][0]
        assert tc["id"] == "call_abc123"
        assert tc["function"]["name"] == "get_weather"
        assert json.loads(tc["function"]["arguments"]) == {"location": "Tokyo"}


def test_native_server_chat_completions_tool_calling_streaming():
    """Verify streaming tool calling yields tool_calls deltas and finish_reason: 'tool_calls' without duplicate stop."""
    client = TestClient(app)
    mock_stub = MagicMock()

    chunk1 = json.dumps({
        "id": "chatcmpl-stream1",
        "object": "chat.completion.chunk",
        "choices": [{
            "index": 0,
            "delta": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "index": 0,
                    "id": "call_stream99",
                    "type": "function",
                    "function": {"name": "run_command", "arguments": ""}
                }]
            },
            "finish_reason": None
        }]
    })

    chunk2 = json.dumps({
        "id": "chatcmpl-stream1",
        "object": "chat.completion.chunk",
        "choices": [{
            "index": 0,
            "delta": {
                "tool_calls": [{
                    "index": 0,
                    "function": {"arguments": "{\"cmd\": \"dir\"}"}
                }]
            },
            "finish_reason": None
        }]
    })

    chunk3 = json.dumps({
        "id": "chatcmpl-stream1",
        "object": "chat.completion.chunk",
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "tool_calls"
        }]
    })

    replies = [
        backend_pb2.Reply(message=chunk1.encode("utf-8")),
        backend_pb2.Reply(message=chunk2.encode("utf-8")),
        backend_pb2.Reply(message=chunk3.encode("utf-8")),
    ]

    mock_stub.PredictStream.return_value = iter(replies)

    with patch("local_ai_launcher.localai.native_server.supervisor.ensure_backend_loaded") as mock_ensure:
        async def _fake_ensure(model_name):
            return mock_stub
        mock_ensure.side_effect = _fake_ensure

        payload = {
            "model": "qwen-38_27b-blackwell-256k-think",
            "messages": [{"role": "user", "content": "List files"}],
            "tools": [{"type": "function", "function": {"name": "run_command"}}],
            "stream": True
        }
        resp = client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 200
        lines = [line for line in resp.text.split("\n") if line.startswith("data: ")]
        assert lines[-1] == "data: [DONE]"

        parsed_chunks = [json.loads(line[6:]) for line in lines if line != "data: [DONE]"]
        assert len(parsed_chunks) == 3
        # Check finish reason of last chunk
        assert parsed_chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"
        # Check tool_calls delta in chunk 0
        tc_delta = parsed_chunks[0]["choices"][0]["delta"]["tool_calls"][0]
        assert tc_delta["id"] == "call_stream99"
        assert tc_delta["function"]["name"] == "run_command"


def test_shim_payload_builders_preserve_tool_call_conventions():
    """Verify that all shims correctly set content=None on assistant tool_call messages."""
    from local_ai_launcher.shims.blackwell_llama_shim import BlackwellLlamaServicer
    from local_ai_launcher.shims.tabby_grpc_shim import TabbyAPIServicer
    from local_ai_launcher.shims.freetoken_grpc_shim import FreeTokenServicer

    msgs = [
        backend_pb2.Message(role="user", content="run tool"),
        backend_pb2.Message(
            role="assistant",
            content="",
            tool_calls=json.dumps([{"id": "call_1", "type": "function", "function": {"name": "tool1", "arguments": "{}"}}])
        ),
        backend_pb2.Message(role="tool", tool_call_id="call_1", content="ok")
    ]
    req = backend_pb2.PredictOptions(
        Messages=msgs,
        Tools=json.dumps([{"type": "function", "function": {"name": "tool1"}}])
    )

    for servicer_cls in (BlackwellLlamaServicer, TabbyAPIServicer, FreeTokenServicer):
        servicer = servicer_cls()
        payload = servicer._build_request_payload(req)
        assert "messages" in payload
        assert len(payload["messages"]) == 3
        # Assistant tool call message must have content None
        assert payload["messages"][1]["content"] is None
        assert payload["messages"][1]["tool_calls"][0]["id"] == "call_1"
        assert payload["messages"][2]["tool_call_id"] == "call_1"
        assert payload["messages"][2]["content"] == "ok"
        assert len(payload["tools"]) == 1


def test_native_server_chat_completions_tool_calling_delta_fallback():
    """Verify non-streaming fallback when shim populates ChatDelta with tool_calls instead of raw JSON."""
    client = TestClient(app)
    mock_stub = MagicMock()

    cd = backend_pb2.ChatDelta(content="")
    cd.tool_calls.append(backend_pb2.ToolCallDelta(
        index=0,
        id="call_fallback123",
        name="read_file",
        arguments="{\"path\": \"foo.txt\"}"
    ))
    mock_reply = backend_pb2.Reply(
        message=b"",
        tokens=10,
        prompt_tokens=15,
        chat_deltas=[cd]
    )
    mock_stub.Predict.return_value = mock_reply

    with patch("local_ai_launcher.localai.native_server.supervisor.ensure_backend_loaded") as mock_ensure:
        async def _fake_ensure(model_name):
            return mock_stub
        mock_ensure.side_effect = _fake_ensure

        payload = {
            "model": "qwen-38_27b-blackwell-256k-think",
            "messages": [{"role": "user", "content": "read foo.txt"}],
            "tools": [{"type": "function", "function": {"name": "read_file"}}],
            "stream": False
        }
        resp = client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["finish_reason"] == "tool_calls"
        assert len(data["choices"][0]["message"]["tool_calls"]) == 1
        tc = data["choices"][0]["message"]["tool_calls"][0]
        assert tc["id"] == "call_fallback123"
        assert tc["function"]["name"] == "read_file"
        assert json.loads(tc["function"]["arguments"]) == {"path": "foo.txt"}


def test_response_format_and_stream_options_forwarding():
    """Verify that response_format and stream_options are preserved and forwarded by native_server and shims."""
    from local_ai_launcher.shims.blackwell_llama_shim import BlackwellLlamaServicer
    client = TestClient(app)
    mock_stub = MagicMock()

    captured_opts = []

    with patch("local_ai_launcher.localai.native_server.supervisor.ensure_backend_loaded") as mock_ensure:
        async def _fake_ensure(model_name):
            def _fake_predict(opts):
                captured_opts.append(opts)
                return backend_pb2.Reply(message=b'{"choices":[{"message":{"content":"{}"}}]}')
            mock_stub.Predict.side_effect = _fake_predict
            return mock_stub
        mock_ensure.side_effect = _fake_ensure

        payload = {
            "model": "qwen-38_27b-blackwell-256k-think",
            "messages": [{"role": "user", "content": "Return json"}],
            "response_format": {"type": "json_object"},
            "stream_options": {"include_usage": True},
            "dry_multiplier": 0.8,
            "samplers": ["top_k", "temperature"],
            "stream": False
        }
        resp = client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 200
        assert len(captured_opts) == 1
        opts = captured_opts[0]
        assert "response_format" in opts.Metadata
        assert "stream_options" in opts.Metadata
        assert "extra_body_params" in opts.Metadata
        extra = json.loads(opts.Metadata["extra_body_params"])
        assert extra.get("dry_multiplier") == 0.8
        assert extra.get("samplers") == ["top_k", "temperature"]

        # Test that BlackwellLlamaServicer builds payload with these
        servicer = BlackwellLlamaServicer()
        built = servicer._build_request_payload(opts)
        assert built.get("response_format") == {"type": "json_object"}
        assert built.get("stream_options") == {"include_usage": True}
        assert built.get("dry_multiplier") == 0.8
        assert built.get("samplers") == ["top_k", "temperature"]


def test_multimodal_vision_image_forwarding():
    """Verify multimodal vision requests inject image_url blocks into messages."""
    from local_ai_launcher.shims.blackwell_llama_shim import BlackwellLlamaServicer

    msgs = [
        backend_pb2.Message(role="user", content="Look at this picture")
    ]
    opts = backend_pb2.PredictOptions(
        Messages=msgs,
        Images=["data:image/jpeg;base64,123456789"]
    )
    servicer = BlackwellLlamaServicer()
    built = servicer._build_request_payload(opts)
    assert len(built["messages"]) == 1
    user_msg = built["messages"][0]
    assert isinstance(user_msg["content"], list)
    assert user_msg["content"][0] == {"type": "text", "text": "Look at this picture"}
    assert user_msg["content"][1] == {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,123456789"}}


def test_single_model_endpoint_retrieve():
    """Verify GET /v1/models/{model_id} returns model metadata."""
    client = TestClient(app)
    resp = client.get("/v1/models/qwen-38_27b-blackwell-256k-think")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "model"
    assert data["id"] == "qwen-38_27b-blackwell-256k-think"


def test_huang_and_bruno_llama_shims_tool_calling_predict():
    """Verify Huang and Bruno shims preserve tool_calls and json data in Predict."""
    from local_ai_launcher.shims.huang_llama_shim import HuangLlamaServicer
    from local_ai_launcher.shims.bruno_llama_shim import BrunoLlamaServicer

    mock_resp_data = {
        "id": "chatcmpl-test",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "Executing command",
                "reasoning_content": "Thinking about command",
                "tool_calls": [{
                    "index": 0,
                    "id": "call_123",
                    "type": "function",
                    "function": {"name": "bash", "arguments": "{\"command\": \"ls\"}"}
                }]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20}
    }

    req = backend_pb2.PredictOptions(
        Messages=[backend_pb2.Message(role="user", content="run ls")]
    )

    for servicer in (HuangLlamaServicer(), BrunoLlamaServicer()):
        servicer.is_ready = True
        servicer.current_model = "qwen-38_27b-blackwell-256k-think"
        servicer.http_base = "http://127.0.0.1:9999"
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_resp_data

        with patch("httpx.Client.post", return_value=mock_resp):
            reply = servicer.Predict(req, None)
            assert reply.tokens == 20
            assert reply.prompt_tokens == 10
            # Raw JSON payload must be in reply.message for passthrough
            raw_parsed = json.loads(reply.message.decode("utf-8"))
            assert raw_parsed["choices"][0]["message"]["tool_calls"][0]["id"] == "call_123"
            # chat_deltas must be populated with tool_calls and reasoning
            assert len(reply.chat_deltas) == 1
            cd = reply.chat_deltas[0]
            assert cd.content == "Executing command"
            assert cd.reasoning_content == "Thinking about command"
            assert len(cd.tool_calls) == 1
            assert cd.tool_calls[0].name == "bash"
            assert cd.tool_calls[0].id == "call_123"


def test_huang_and_bruno_llama_shims_tool_calling_predict_stream():
    """Verify Huang and Bruno shims stream raw_data passthrough chunks and ChatDelta tool_calls."""
    from local_ai_launcher.shims.huang_llama_shim import HuangLlamaServicer
    from local_ai_launcher.shims.bruno_llama_shim import BrunoLlamaServicer

    chunk1 = "data: " + json.dumps({
        "id": "chatcmpl-test",
        "choices": [{
            "delta": {
                "role": "assistant",
                "content": "Let me run this",
                "reasoning_content": "Planning bash call"
            },
            "finish_reason": None
        }]
    })
    chunk2 = "data: " + json.dumps({
        "id": "chatcmpl-test",
        "choices": [{
            "delta": {
                "tool_calls": [{
                    "index": 0,
                    "id": "call_stream123",
                    "type": "function",
                    "function": {"name": "bash", "arguments": "{\"command\": \"dir\"}"}
                }]
            },
            "finish_reason": None
        }]
    })
    chunk3 = "data: " + json.dumps({
        "id": "chatcmpl-test",
        "choices": [{
            "delta": {},
            "finish_reason": "tool_calls"
        }]
    })
    chunk_done = "data: [DONE]"

    lines = [chunk1, chunk2, chunk3, chunk_done]

    req = backend_pb2.PredictOptions(
        Messages=[backend_pb2.Message(role="user", content="dir")]
    )

    for servicer in (HuangLlamaServicer(), BrunoLlamaServicer()):
        servicer.is_ready = True
        servicer.current_model = "qwen-38_27b-blackwell-256k-think"
        servicer.http_base = "http://127.0.0.1:9999"

        class MockStreamContext:
            def __enter__(self):
                mock_resp = MagicMock()
                mock_resp.iter_lines.return_value = lines
                return mock_resp
            def __exit__(self, *args):
                pass

        with patch("httpx.Client.stream", return_value=MockStreamContext()):
            replies = list(servicer.PredictStream(req, None))
            assert len(replies) == 3
            # Check chunk 1
            c1_raw = json.loads(replies[0].message.decode("utf-8"))
            assert c1_raw["choices"][0]["delta"]["content"] == "Let me run this"
            assert replies[0].chat_deltas[0].reasoning_content == "Planning bash call"

            # Check chunk 2 has tool_calls
            c2_raw = json.loads(replies[1].message.decode("utf-8"))
            assert c2_raw["choices"][0]["delta"]["tool_calls"][0]["id"] == "call_stream123"
            assert len(replies[1].chat_deltas[0].tool_calls) == 1
            assert replies[1].chat_deltas[0].tool_calls[0].name == "bash"

            # Check chunk 3 has finish_reason tool_calls
            c3_raw = json.loads(replies[2].message.decode("utf-8"))
            assert c3_raw["choices"][0]["finish_reason"] == "tool_calls"


def test_native_server_non_streaming_text_tool_call_extraction():
    """Verify native_server extracts text <tool_call> tags when model emits raw XML."""
    client = TestClient(app)
    mock_stub = MagicMock()

    raw_text = 'I will run the command:\n<tool_call>\n{"name": "bash", "arguments": {"command": "git status"}}\n</tool_call>'
    mock_reply = backend_pb2.Reply(
        message=raw_text.encode("utf-8"),
        tokens=12,
        prompt_tokens=8
    )
    mock_stub.Predict.return_value = mock_reply

    with patch("local_ai_launcher.localai.native_server.supervisor.ensure_backend_loaded") as mock_ensure:
        async def _fake_ensure(model_name):
            return mock_stub
        mock_ensure.side_effect = _fake_ensure

        payload = {
            "model": "qwen-38_27b-blackwell-256k-think",
            "messages": [{"role": "user", "content": "git status"}],
            "stream": False
        }
        resp = client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["finish_reason"] == "tool_calls"
        assert len(data["choices"][0]["message"]["tool_calls"]) == 1
        tc = data["choices"][0]["message"]["tool_calls"][0]
        assert tc["function"]["name"] == "bash"
        assert json.loads(tc["function"]["arguments"]) == {"command": "git status"}
        assert "<tool_call>" not in (data["choices"][0]["message"]["content"] or "")


def test_native_server_streaming_tool_calls_overrides_stop_finish_reason():
    """Verify native_server overrides finish_reason 'stop' to 'tool_calls' when stream emitted tool_calls."""
    client = TestClient(app)
    mock_stub = MagicMock()

    chunk1 = json.dumps({
        "id": "chatcmpl-stream-tc",
        "object": "chat.completion.chunk",
        "choices": [{
            "index": 0,
            "delta": {
                "role": "assistant",
                "tool_calls": [{
                    "index": 0,
                    "id": "call_12345",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{\"path\": \"foo.txt\"}"}
                }]
            },
            "finish_reason": None
        }]
    })

    chunk2 = json.dumps({
        "id": "chatcmpl-stream-tc",
        "object": "chat.completion.chunk",
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"  # Upstream model sent stop instead of tool_calls
        }]
    })

    replies = [
        backend_pb2.Reply(message=chunk1.encode("utf-8")),
        backend_pb2.Reply(message=chunk2.encode("utf-8")),
    ]
    mock_stub.PredictStream.return_value = iter(replies)

    with patch("local_ai_launcher.localai.native_server.supervisor.ensure_backend_loaded") as mock_ensure:
        async def _fake_ensure(model_name):
            return mock_stub
        mock_ensure.side_effect = _fake_ensure

        payload = {
            "model": "qwen-38_27b-blackwell-256k-think",
            "messages": [{"role": "user", "content": "Read foo.txt"}],
            "stream": True
        }
        resp = client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 200

        data_lines = [l for l in resp.text.split("\n") if l.startswith("data: ") and not l.endswith("[DONE]")]
        chunks = [json.loads(l[6:]) for l in data_lines]
        final_chunk = chunks[-1]
        assert final_chunk["choices"][0]["finish_reason"] == "tool_calls"


def test_bruno_and_huang_shims_endpoint_dispatch_and_empty_choices():
    """Verify Bruno and Huang shims dynamically dispatch to /v1/chat/completions vs /v1/completions and handle empty choices."""
    from local_ai_launcher.shims.bruno_llama_shim import BrunoLlamaServicer
    from local_ai_launcher.shims.huang_llama_shim import HuangLlamaServicer

    for ServicerCls in (BrunoLlamaServicer, HuangLlamaServicer):
        servicer = ServicerCls()
        servicer.is_ready = True
        servicer.http_base = "http://127.0.0.1:8081"
        servicer.current_model = "test-model"

        # Case 1: Chat completion with messages
        req_chat = backend_pb2.PredictOptions(
            Messages=[backend_pb2.Message(role="user", content="hello")]
        )
        with patch("httpx.Client.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"choices": [{"message": {"role": "assistant", "content": "hi"}}]}
            mock_post.return_value = mock_resp

            servicer.Predict(req_chat, MagicMock())
            assert mock_post.call_args[0][0] == "http://127.0.0.1:8081/v1/chat/completions"

        # Case 2: Text completion with raw prompt
        req_text = backend_pb2.PredictOptions(
            Prompt="Once upon a time"
        )
        with patch("httpx.Client.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"choices": [{"text": " there was a..."}]}
            mock_post.return_value = mock_resp

            servicer.Predict(req_text, MagicMock())
            assert mock_post.call_args[0][0] == "http://127.0.0.1:8081/v1/completions"

        # Case 3: PredictStream handles empty choices and usage chunks without crashing
        req_stream = backend_pb2.PredictOptions(
            Messages=[backend_pb2.Message(role="user", content="test stream")]
        )
        fake_sse_lines = [
            'data: {"choices": []}',
            'data: {"usage": {"prompt_tokens": 10, "completion_tokens": 20}}',
            'data: {"choices": [{"delta": {"content": "ok"}}]}',
            'data: [DONE]'
        ]
        with patch("httpx.Client.stream") as mock_stream_ctx:
            mock_stream_resp = MagicMock()
            mock_stream_resp.iter_lines.return_value = iter(fake_sse_lines)
            mock_stream_ctx.return_value.__enter__.return_value = mock_stream_resp

            replies = list(servicer.PredictStream(req_stream, MagicMock()))
            assert len(replies) >= 3
            # The chunk with delta content should have yielded a chat delta
            delta_replies = [r for r in replies if r.chat_deltas]
            assert len(delta_replies) == 1
            assert delta_replies[0].chat_deltas[0].content == "ok"


def test_strata_shim_endpoint_dispatch_and_stream():
    """Verify Strata shim dispatches chat completions and handles empty streaming choices cleanly."""
    from local_ai_launcher.shims.strata_grpc_shim import StrataServicer

    servicer = StrataServicer()
    servicer.is_ready = True
    servicer.http_base = "http://127.0.0.1:8081"
    servicer.current_model = "qwen-38-flash-strata-iq3s"

    # 1. Chat completion
    req_chat = backend_pb2.PredictOptions(
        Messages=[backend_pb2.Message(role="user", content="hello strata")]
    )
    with patch("httpx.Client.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"choices": [{"message": {"role": "assistant", "content": "hi from strata"}}]}
        mock_post.return_value = mock_resp

        reply = servicer.Predict(req_chat, MagicMock())
        assert mock_post.call_args[0][0] == "http://127.0.0.1:8081/v1/chat/completions"
        assert reply.chat_deltas[0].content == "hi from strata"

    # 2. PredictStream
    fake_sse_lines = [
        'data: {"choices": []}',
        'data: {"choices": [{"delta": {"content": "ok strata"}}]}',
        'data: [DONE]'
    ]
    with patch("httpx.Client.stream") as mock_stream_ctx:
        mock_stream_resp = MagicMock()
        mock_stream_resp.status_code = 200
        mock_stream_resp.iter_lines.return_value = iter(fake_sse_lines)
        mock_stream_ctx.return_value.__enter__.return_value = mock_stream_resp

        replies = list(servicer.PredictStream(req_chat, MagicMock()))
        delta_replies = [r for r in replies if r.chat_deltas]
        assert len(delta_replies) == 1
        assert delta_replies[0].chat_deltas[0].content == "ok strata"


def test_anthropic_messages_endpoint_non_streaming():
    """Verify POST /v1/messages translates Anthropic request and returns Anthropic message envelope."""
    client = TestClient(app)

    mock_stub = MagicMock()
    mock_reply = backend_pb2.Reply(
        message=b"Hello from local model!",
        tokens=5,
        prompt_tokens=12
    )
    mock_stub.Predict.return_value = mock_reply

    with patch("local_ai_launcher.localai.native_server.supervisor.ensure_backend_loaded") as mock_ensure:
        async def _fake_ensure(model_name):
            return mock_stub
        mock_ensure.side_effect = _fake_ensure

        payload = {
            "model": "qwen-38_27b-blackwell-256k-think",
            "messages": [
                {"role": "user", "content": "Hi there"}
            ],
            "system": "You are a helpful coding agent.",
            "max_tokens": 4096,
            "stream": False
        }

        resp = client.post("/v1/messages", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "message"
        assert data["role"] == "assistant"
        assert len(data["content"]) >= 1
        assert data["content"][0]["type"] == "text"
        assert data["content"][0]["text"] == "Hello from local model!"
        assert data["stop_reason"] == "end_turn"
        assert data["usage"]["input_tokens"] == 12
        assert data["usage"]["output_tokens"] == 5


def test_anthropic_messages_endpoint_streaming():
    """Verify POST /v1/messages translates streaming responses into valid Anthropic SSE events."""
    client = TestClient(app)

    mock_stub = MagicMock()
    # Emulate PredictStream yielding token chunks
    chunk1 = backend_pb2.Reply(message=b"data: {\"choices\": [{\"delta\": {\"content\": \"Part 1 \"}}]}\n\n")
    chunk2 = backend_pb2.Reply(message=b"data: {\"choices\": [{\"delta\": {\"content\": \"Part 2\"}, \"finish_reason\": \"stop\"}], \"usage\": {\"prompt_tokens\": 10, \"completion_tokens\": 6}}\n\n")
    mock_stub.PredictStream.return_value = iter([chunk1, chunk2])

    with patch("local_ai_launcher.localai.native_server.supervisor.ensure_backend_loaded") as mock_ensure:
        async def _fake_ensure(model_name):
            return mock_stub
        mock_ensure.side_effect = _fake_ensure

        payload = {
            "model": "qwen-38_27b-blackwell-256k-think",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Stream test"}]}
            ],
            "max_tokens": 1024,
            "stream": True
        }

        resp = client.post("/v1/messages", json=payload)
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

        body_text = resp.text
        assert "event: message_start" in body_text
        assert "event: content_block_start" in body_text
        assert "event: content_block_delta" in body_text
        assert "Part 1 " in body_text
        assert "Part 2" in body_text
        assert "event: content_block_stop" in body_text
        assert "event: message_delta" in body_text
        assert "event: message_stop" in body_text


def test_anthropic_messages_endpoint_tool_calling():
    """Verify POST /v1/messages correctly formats tool calls and tools parameter."""
    client = TestClient(app)

    mock_stub = MagicMock()
    # Emulate Predict returning function call JSON
    tc_data = [{
        "id": "call_abc123",
        "type": "function",
        "function": {"name": "read_file", "arguments": "{\"path\": \"test.py\"}"}
    }]
    mock_reply = backend_pb2.Reply(
        message=b"",
        tokens=15,
        prompt_tokens=25
    )
    # Inject tool_calls into the completion response
    with patch("local_ai_launcher.localai.native_server.supervisor.ensure_backend_loaded") as mock_ensure, \
         patch("local_ai_launcher.localai.native_server.chat_completions") as mock_chat:
        from fastapi.responses import JSONResponse
        mock_chat.return_value = JSONResponse({
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1234567,
            "model": "test-model",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": tc_data
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 25, "completion_tokens": 15}
        })

        payload = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Read test.py"}],
            "tools": [{
                "name": "read_file",
                "description": "Read local file",
                "input_schema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"]
                }
            }],
            "stream": False
        }

        resp = client.post("/v1/messages", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "message"
        assert data["stop_reason"] == "tool_use"
        assert len(data["content"]) == 1
        tool_block = data["content"][0]
        assert tool_block["type"] == "tool_use"
        assert tool_block["id"] == "call_abc123"
        assert tool_block["name"] == "read_file"
        assert tool_block["input"] == {"path": "test.py"}


def test_anthropic_messages_endpoint_tool_result_no_empty_user():
    """Verify tool_result without text does not append a spurious empty user message."""
    client = TestClient(app)

    with patch("local_ai_launcher.localai.native_server.supervisor.ensure_backend_loaded") as mock_ensure, \
         patch("local_ai_launcher.localai.native_server.chat_completions") as mock_chat:
        from fastapi.responses import JSONResponse
        captured_messages = []

        async def _fake_chat(req):
            b = json.loads((await req.body()).decode("utf-8"))
            captured_messages.extend(b.get("messages", []))
            return JSONResponse({
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1234567,
                "model": "test-model",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "Done processing tool result."},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5}
            })
        mock_chat.side_effect = _fake_chat

        payload = {
            "model": "test-model",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_123",
                            "content": "File contents here",
                            "is_error": False
                        }
                    ]
                }
            ],
            "stream": False
        }

        resp = client.post("/v1/messages", json=payload)
        assert resp.status_code == 200
        # Verify that ONLY the tool message was translated, and NO empty user message was appended
        assert len(captured_messages) == 1
        assert captured_messages[0]["role"] == "tool"
        assert captured_messages[0]["tool_call_id"] == "call_123"
        assert captured_messages[0]["content"] == "File contents here"


def test_anthropic_messages_endpoint_streaming_multichunk_thinking():
    """Verify streaming <think> tags spanning multiple chunks preserve thinking block continuity."""
    client = TestClient(app)

    mock_stub = MagicMock()
    # Emulate chunks: <think>, thinking body, </think>, and final answer
    c1 = backend_pb2.Reply(message=b"data: {\"choices\": [{\"delta\": {\"content\": \"<think>\\nStep 1: check\"}}]}\n\n")
    c2 = backend_pb2.Reply(message=b"data: {\"choices\": [{\"delta\": {\"content\": \"\\nStep 2: verify\"}}]}\n\n")
    c3 = backend_pb2.Reply(message=b"data: {\"choices\": [{\"delta\": {\"content\": \"\\n</think>\\nAnswer is 42.\"}}]}\n\n")
    c4 = backend_pb2.Reply(message=b"data: {\"choices\": [{\"delta\": {\"content\": \"\"}, \"finish_reason\": \"stop\"}], \"usage\": {\"prompt_tokens\": 10, \"completion_tokens\": 8}}\n\n")
    mock_stub.PredictStream.return_value = iter([c1, c2, c3, c4])

    with patch("local_ai_launcher.localai.native_server.supervisor.ensure_backend_loaded") as mock_ensure:
        async def _fake_ensure(model_name):
            return mock_stub
        mock_ensure.side_effect = _fake_ensure

        payload = {
            "model": "qwen-38_27b-blackwell-256k-think",
            "messages": [
                {"role": "user", "content": "What is 2+2?"}
            ],
            "stream": True
        }

        resp = client.post("/v1/messages", json=payload)
        assert resp.status_code == 200
        lines = [line.strip() for line in resp.text.split("\n") if line.strip()]

        # Parse SSE events
        events = []
        for l in lines:
            if l.startswith("data:"):
                try:
                    events.append(json.loads(l[5:].strip()))
                except Exception:
                    pass

        # Verify block sequence:
        # block 0 must be 'thinking'
        start_events = [e for e in events if e.get("type") == "content_block_start"]
        assert len(start_events) == 2
        assert start_events[0]["content_block"]["type"] == "thinking"
        assert start_events[1]["content_block"]["type"] == "text"

        deltas = [e.get("delta", {}) for e in events if e.get("type") == "content_block_delta"]
        thinking_text = "".join(d.get("thinking", "") for d in deltas if d.get("type") == "thinking_delta")
        text_content = "".join(d.get("text", "") for d in deltas if d.get("type") == "text_delta")

        assert "Step 1: check" in thinking_text
        assert "Step 2: verify" in thinking_text
        assert "<think>" not in thinking_text
        assert "</think>" not in thinking_text
        assert "Answer is 42." in text_content








