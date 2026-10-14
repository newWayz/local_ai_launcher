import json
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from local_ai_launcher.localai.loop_detector import (
    StreamingLoopDetector,
    detect_and_truncate_loop
)
from local_ai_launcher.localai.native_server import app
from local_ai_launcher.localai.proto import backend_pb2
from local_ai_launcher.config import config


USER_SAMPLE_TRACE = """
I'm hitting the same error persistently, which suggests my struct definitions are still misaligned. Looking at the actual Windows job object structures, I need to verify the exact layout of the basic limit information and extended
limit information structures to find where the mismatch is occurring. I'm realizing the struct layout might be the issue — JOBOBJECT_EXTENDED_LIMIT_INFORMATION should contain the basic limit info followed by IO counters and then
the memory limits, but I need to verify the field names and sizes are matching what Windows actually expects. The ERROR_MORE_DATA error suggests the structure size or alignment is off, so I'm checking whether
JOBOBJECT_EXTENDED_LIMIT_INFORMATION is the right structure type and if my field definitions match the actual Windows API layout. I'm reconsidering whether the struct layout is actually the problem here. The
JOBOBJECT_EXTENDED_LIMIT_INFORMATION structure should be 168 bytes on 64-bit systems, and if my ctypes definition doesn't match that exactly, SetInformationJobObject would return ERROR_MORE_DATA. Let me verify the actual structure
definition from the Windows headers to make sure I have the right fields in the right order. I'm reconsidering the struct definition and the error code. The issue might be that I'm using the wrong information class number or the
struct layout doesn't match what the API expects. Let me verify the exact JOBOBJECT_EXTENDED_LIMIT_INFORMATION structure and double-check that JobObjectExtendedLimitInformation = 9 is correct, since ERROR_MORE_DATA specifically
indicates a size mismatch between what I'm passing and what the API needs. I'm second-guessing the struct layout — maybe the issue is that JOBOBJECT_EXTENDED_LIMIT_INFORMATION has a different field order or padding than I'm
assuming, or perhaps I'm using the wrong information class entirely. Let me reconsider whether the problem is actually the struct size or something else about how I'm calling the API. I'm second-guessing the struct layout — the
error code 24 (ERROR_MORE_DATA) keeps appearing, which suggests the buffer I'm passing is the wrong size. Let me verify the actual field order and sizes in JOBOBJECT_BASIC_LIMIT_INFORMATION against the Windows headers to make sure
my ctypes structure matches exactly. I'm second-guessing whether my struct layout is actually correct — the error code 24 (ERROR_MORE_DATA) is suspicious and suggests the buffer size might be off. Let me verify by checking the
actual Windows header definition of JOBOBJECT_BASIC_LIMIT_INFORMATION to make sure my ctypes structure matches the expected 88 bytes on x64. I'm second-guessing the struct layout and whether the information class constant is
correct. The error code 24 (ERROR_MORE_DATA) is suspicious — it suggests a size mismatch between what I'm sending and what the API expects. Let me verify the actual Windows header definition for JOBOBJECT_BASIC_LIMIT_INFORMATION
to make sure my ctypes structure matches exactly, since that's likely where the discrepancy is. I'm second-guessing the struct layout — the error code 24 (ERROR_MORE_DATA) suggests the buffer size is wrong, which points to my
_ExtendedLimit structure not matching what the API expects. Let me verify the actual Windows header definition for JOBOBJECT_EXTENDED_LIMIT_INFORMATION to make sure the field order and sizes are correct, since that's likely where
the mismatch is. I'm second-guessing the struct layout — the error code 24 (ERROR_MORE_DATA) suggests the buffer size is wrong, which points to my _ExtendedLimit structure not matching what the API expects. Let me verify the
actual Windows header definition for JOBOBJECT_EXTENDED_LIMIT_INFORMATION to make sure the field order and sizes are correct, since that's likely where the mismatch is. I'm second-guessing the struct layout — the error code 24
(ERROR_MORE_DATA) suggests the buffer size is wrong, which points to my _ExtendedLimit structure not matching what the API expects. Let me verify the actual Windows header definition for JOBOBJECT_EXTENDED_LIMIT_INFORMATION to
make sure the field order and sizes are correct, since that's likely where the mismatch is.
"""


def test_verbatim_loop_detection():
    """Verify that verbatim repeated sentences are caught quickly."""
    detector = StreamingLoopDetector(ngram_size=8, ngram_threshold=3)
    sentence = "Let us verify the exact memory structure and boundary offsets in the header file. "
    full_text = sentence * 4

    triggered = False
    for i in range(0, len(full_text), 20):
        if detector.feed(full_text[i:i+20]):
            triggered = True
            break
    assert triggered is True
    assert detector.loop_detected is True


def test_user_sample_pi_loop_detection():
    """Verify that the user's exact Pi harness trace is caught deterministically across varying chunk sizes."""
    for chunk_size in [15, 30, 60]:
        detector = StreamingLoopDetector(
            ngram_size=8,
            ngram_threshold=3,
            cycle_sim_threshold=0.82,
            min_sentence_len=35
        )
        triggered = False
        for i in range(0, len(USER_SAMPLE_TRACE), chunk_size):
            chunk = USER_SAMPLE_TRACE[i:i+chunk_size]
            if detector.feed(chunk, is_reasoning=True):
                triggered = True
                break
        assert triggered is True
        assert detector.loop_detected is True
        assert "what the api expects let me verify the" in detector.reason or "cycle" in detector.loop_type.lower() or "ngram" in detector.loop_type.lower()


def test_alternating_cycle_with_lexical_jitter():
    """Verify detection of alternating sentences with near-duplicate word variations."""
    s1 = "We should inspect the kernel job object information structure to make sure the buffer size is aligned with Windows.\n"
    s2 = "Let me re-examine the kernel job object information layout to ensure the memory bounds match the OS specification.\n"
    
    text = (s1 + s2) * 3
    detector = StreamingLoopDetector(cycle_sim_threshold=0.80, min_sentence_len=35)
    triggered = False
    for i in range(0, len(text), 25):
        if detector.feed(text[i:i+25]):
            triggered = True
            break
    assert triggered is True
    assert detector.loop_type == "cycle"


def test_period_3_cycle_detection():
    """Verify period-3 alternating loop (A-B-C-A-B-C) is detected."""
    s1 = "First step is to check whether the process memory quota is exceeded by the job object.\n"
    s2 = "Second step is to verify if the security descriptor permissions allow opening the target process handle.\n"
    s3 = "Third step is to inspect the basic limit flags to confirm the extended limits structure is required.\n"

    text = (s1 + s2 + s3) * 3
    detector = StreamingLoopDetector(cycle_sim_threshold=0.80, min_sentence_len=35)
    triggered = False
    for i in range(0, len(text), 25):
        if detector.feed(text[i:i+25]):
            triggered = True
            break
    assert triggered is True


def test_negative_code_and_asserts():
    """Ensure standard Python code, unit test asserts, and natural reasoning do not trigger false alarms."""
    code_text = """
import unittest

class TestJobLimits(unittest.TestCase):
    def test_limit_01(self):
        self.assertEqual(get_limit(1), 100)
    def test_limit_02(self):
        self.assertEqual(get_limit(2), 200)
    def test_limit_03(self):
        self.assertEqual(get_limit(3), 300)
    def test_limit_04(self):
        self.assertEqual(get_limit(4), 400)
    def test_limit_05(self):
        self.assertEqual(get_limit(5), 500)
"""
    detector = StreamingLoopDetector()
    for i in range(0, len(code_text), 20):
        assert detector.feed(code_text[i:i+20]) is False
    assert detector.loop_detected is False


def test_code_block_fence_exemption():
    """Ensure repetitive statements inside markdown code fences do not trigger sentence cyclic alarms."""
    fenced_code = """```python
for item in data_stream:
    if item is None:
        continue
    process(item)
    save(item)
    commit()
for item in data_stream:
    if item is None:
        continue
    process(item)
    save(item)
    commit()
```"""
    detector = StreamingLoopDetector()
    for i in range(0, len(fenced_code), 15):
        detector.feed(fenced_code[i:i+15])
    assert detector.loop_detected is False


def test_build_halt_chunk_thinking_synthesis():
    """Verify that build_halt_chunk properly closes </think> tags when interrupted in reasoning monologue."""
    detector = StreamingLoopDetector()
    detector.in_thinking_block = True
    detector.loop_detected = True
    chunk = detector.build_halt_chunk("req-123", "qwen-38_27b", 12345678)
    
    assert chunk["id"] == "req-123"
    assert chunk["choices"][0]["finish_reason"] == "stop"
    delta = chunk["choices"][0]["delta"]
    assert "</think>" in delta["content"]
    assert "Reasoning halted" in delta["reasoning_content"]


def test_detect_and_truncate_loop_non_streaming():
    """Verify that non-streaming detect_and_truncate_loop cleans repetitive text."""
    sentence = "We must recheck the exact structure alignment in the Windows system headers. "
    repeated = sentence * 6
    cleaned, looped, reason = detect_and_truncate_loop(repeated, ngram_size=8, ngram_threshold=3)
    assert looped is True
    assert "[Response truncated: repetition loop detected" in cleaned
    assert len(cleaned) < len(repeated)


def test_native_server_streaming_loop_interception():
    """Test that POST /v1/chat/completions halts stream and cancels gRPC on model loop."""
    client = TestClient(app)

    repetitive_chunk = (
        "I'm second-guessing the struct layout — the error code 24 (ERROR_MORE_DATA) suggests the buffer size is wrong, "
        "which points to my _ExtendedLimit structure not matching what the API expects. "
        "Let me verify the actual Windows header definition for JOBOBJECT_EXTENDED_LIMIT_INFORMATION to make sure the field order and sizes are correct. "
    )
    
    # Create fake gRPC stream yielding repetitive chunks
    mock_replies = []
    for _ in range(6):
        mock_reply = backend_pb2.Reply(
            chat_deltas=[
                backend_pb2.ChatDelta(content=repetitive_chunk, reasoning_content="")
            ]
        )
        mock_replies.append(mock_reply)

    mock_call = MagicMock()
    mock_call.__iter__.return_value = iter(mock_replies)

    mock_stub = MagicMock()
    mock_stub.PredictStream.return_value = mock_call

    with patch("local_ai_launcher.localai.native_server.supervisor.ensure_backend_loaded") as mock_ensure:
        async def _fake_ensure(m):
            return mock_stub
        mock_ensure.side_effect = _fake_ensure

        payload = {
            "model": "qwen-38_27b-blackwell-256k-think",
            "messages": [{"role": "user", "content": "Debug this struct"}],
            "stream": True
        }

        resp = client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 200
        content = resp.text
        
        # Verify that halt chunk was delivered and stream terminated with [DONE]
        assert "Response halted: repetition loop detected" in content or "data: [DONE]" in content
        # Verify call.cancel() was called to protect GPU resources
        assert mock_call.cancel.called


def test_code_fence_slicing_and_multi_fence_recovery():
    """Verify that multi-fence deltas and split fences across chunk boundaries do not desync in_code_block."""
    detector = StreamingLoopDetector()

    # 1. Multiple fences inside one delta: opening and closing together
    delta1 = "Here is a quick snippet: ```python\nx = 1\n``` And here is normal prose afterwards.\n"
    detector.feed(delta1)
    # in_code_block must NOT get stuck at True
    assert detector.in_code_block is False

    # 2. Split fence across chunks: first chunk has "\n``", second chunk has "`python\n"
    detector.feed("\n``")
    assert detector.in_code_block is False
    detector.feed("`python\nprint('inside code block')\n")
    assert detector.in_code_block is True

    # 3. Closing fence split across chunks
    detector.feed("``")
    assert detector.in_code_block is True
    detector.feed("`\n")
    assert detector.in_code_block is False

    # 4. Now verify that normal sentence loops outside code block are detected
    repeated_sentence = "We must recheck the exact memory layout in the kernel system headers.\n"
    triggered = False
    for _ in range(4):
        if detector.feed(repeated_sentence):
            triggered = True
            break
    assert triggered is True
    assert detector.loop_detected is True


def test_thinking_tag_slicing_robustness():
    """Verify that split <think> and </think> tags across streaming chunks do not desynchronize thinking block state."""
    detector = StreamingLoopDetector()

    # Split open tag: "<thi" + "nk>Thinking monologue started"
    detector.feed("<thi")
    assert detector.in_thinking_block is False
    detector.feed("nk>This is deep internal reasoning monologue.\n")
    assert detector.in_thinking_block is True

    # Split close tag: "</thi" + "nk>Final answer follows."
    detector.feed("</thi")
    assert detector.in_thinking_block is True
    detector.feed("nk>\nFinal answer follows here.\n")
    assert detector.in_thinking_block is False


def test_sliding_window_prunes_distant_ngrams():
    """Verify that n-grams separated by more than max_history_words are evicted and do not cause false positives."""
    # Small history window for deterministic test
    detector = StreamingLoopDetector(ngram_size=8, ngram_threshold=3, max_history_words=50)

    boilerplate = "for more information regarding this setup consult official system reference documentation "
    # Occurrence 1
    detector.feed(boilerplate)
    assert detector.loop_detected is False

    # Feed 60 unique filler words to push occurrence 1 out of the 50-word sliding window
    filler = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega "
    detector.feed(filler)
    detector.feed(filler)

    # Occurrence 2 (previously occurrence 1 was evicted, so count is now at 1, not 2)
    detector.feed(boilerplate)
    assert detector.loop_detected is False

    # Occurrence 3 (count becomes 2, which is still < threshold 3)
    detector.feed(boilerplate)
    assert detector.loop_detected is False

