import pytest
from unittest.mock import patch, MagicMock
import urllib.request
import urllib.error

from local_ai_launcher.vram_coordinator import get_vram_info, evict_llm_vram, restore_llm


def test_get_vram_info():
    info = get_vram_info()
    assert isinstance(info, dict)
    assert "available" in info
    assert "free_mb" in info
    assert "total_mb" in info


def test_evict_llm_vram_success(monkeypatch):
    import time
    monkeypatch.setattr(time, "sleep", lambda s: None)
    class MockResponse:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *args): pass

    def mock_urlopen(req, timeout=15):
        return MockResponse()

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)
    ok = evict_llm_vram(localai_url="http://127.0.0.1:8080", timeout=5)
    assert ok is True


def test_restore_llm_success(monkeypatch):
    class MockResponse:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *args): pass

    def mock_urlopen(req, timeout=60):
        return MockResponse()

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)
    ok = restore_llm(model_name="test-model", localai_url="http://127.0.0.1:8080")
    assert ok is True
