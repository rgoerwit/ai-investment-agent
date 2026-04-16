import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src import cleanup


@pytest.fixture(autouse=True)
def cleanup_registry_isolated():
    original = list(cleanup._cleanup_functions)
    cleanup._cleanup_functions.clear()
    try:
        yield
    finally:
        cleanup._cleanup_functions.clear()
        cleanup._cleanup_functions.extend(original)


@pytest.mark.asyncio
async def test_cleanup_genai_clients_times_out_and_clears_client(monkeypatch):
    client = SimpleNamespace()

    async def _slow_close():
        await asyncio.sleep(1)

    client.aclose = _slow_close
    llm = SimpleNamespace(async_client=client)
    fake_llms = SimpleNamespace(get_all_llm_instances=lambda: {"quick": llm})

    monkeypatch.setitem(sys.modules, "src.llms", fake_llms)
    monkeypatch.setattr(cleanup, "GENAI_CLIENT_CLOSE_TIMEOUT_SECONDS", 0.01)

    with patch("src.cleanup.logger") as mock_logger:
        await cleanup._cleanup_genai_clients()

    assert llm.async_client is None
    mock_logger.warning.assert_called_once()
    assert mock_logger.warning.call_args.args == ("cleanup_timeout",)
    assert mock_logger.warning.call_args.kwargs == {
        "resource": "genai_async_client_quick",
        "timeout_seconds": 0.01,
    }


@pytest.mark.asyncio
async def test_cleanup_genai_clients_logs_close_failure_and_clears_client(monkeypatch):
    class ExplodingClient:
        async def aclose(self):
            raise RuntimeError("close boom")

    llm = SimpleNamespace(async_client=ExplodingClient())
    fake_llms = SimpleNamespace(get_all_llm_instances=lambda: {"deep": llm})

    monkeypatch.setitem(sys.modules, "src.llms", fake_llms)

    with patch("src.cleanup.logger") as mock_logger:
        await cleanup._cleanup_genai_clients()

    assert llm.async_client is None
    mock_logger.debug.assert_called_once()
    assert mock_logger.debug.call_args.args == ("cleanup_error",)
    assert mock_logger.debug.call_args.kwargs == {
        "resource": "genai_deep",
        "error": "close boom",
    }


@pytest.mark.asyncio
async def test_cleanup_async_resources_continues_after_registered_cleanup_error(
    monkeypatch,
):
    calls: list[str] = []

    async def _bad_cleanup():
        calls.append("bad")
        raise RuntimeError("cleanup failed")

    async def _good_cleanup():
        calls.append("good")

    cleanup.register_cleanup(_bad_cleanup)
    cleanup.register_cleanup(_good_cleanup)
    monkeypatch.setattr(cleanup, "_cleanup_data_fetchers", _good_cleanup)
    monkeypatch.setattr(cleanup, "_cleanup_genai_clients", _good_cleanup)

    with patch("src.cleanup.logger") as mock_logger:
        await cleanup.cleanup_async_resources()

    assert calls == ["bad", "good", "good", "good"]
    mock_logger.debug.assert_called_once_with(
        "cleanup_error",
        function="_bad_cleanup",
        error="cleanup failed",
    )
