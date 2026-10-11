"""The async Tavily entrypoints bound search *and* inspection with one deadline.

``tavily_search_with_timeout``, ``search_tavily_inspected`` and
``extract_tavily_inspected`` check the time left after the search, then give
inspection only that. A result arriving with nothing left is discarded before
inspection starts; an inspection that overruns -- even one that ignores
cancellation -- returns ``None``, never the uninspected payload. This matches
the sync worker in ``search_tavily_sync_inspected``.

The macro-context path calls ``get_macroeconomic_news`` outside the tool
service, so this bound is the only one on its inspection.

Fake deadline clock for sizing; releasable stalls for overruns. Nothing touches
the network.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from types import SimpleNamespace
from typing import Any

import pytest

import src.async_utils as async_utils
import src.tavily_utils as tavily_utils
import src.tooling.inspection_service as inspection_service
from src.config import config
from src.llm_runtime import timeouts as llm_timeouts
from src.llm_runtime.timeouts import enclosing_deadline
from tests.helpers.tavily_inspection import inspect_tavily_with

PAYLOAD = {"results": [{"content": "RAW_UNINSPECTED_PAYLOAD"}]}
SAFETY_NET_SECONDS = 2.0


@pytest.fixture
def fake_clock(monkeypatch) -> list[float]:
    clock = [1_000.0]
    monkeypatch.setattr(llm_timeouts, "now", lambda: clock[0])
    return clock


class _Search:
    """A fake Tavily search/extract tool whose request runs *on_call* first."""

    def __init__(self, on_call: Callable[[], None] | None = None) -> None:
        self.on_call = on_call
        self.calls = 0

    def ainvoke(self, payload: dict[str, Any]) -> Awaitable[Any]:
        del payload
        self.calls += 1
        return self._respond()

    async def _respond(self) -> Any:
        if self.on_call is not None:
            self.on_call()
        return PAYLOAD


@pytest.fixture
def install(monkeypatch):
    """Install one fake tool behind all three async entrypoints."""
    import langchain_tavily
    import langchain_tavily._utilities as tavily_utilities

    def _install(tool: _Search) -> None:
        monkeypatch.setattr(tavily_utils, "_tavily_tool", tool)
        monkeypatch.setattr(
            tavily_utils,
            "_build_tavily_tool",
            lambda profile, include_answer=False: tool,
        )
        monkeypatch.setattr(langchain_tavily, "TavilyExtract", lambda **kwargs: tool)
        monkeypatch.setattr(
            tavily_utilities, "TavilyExtractAPIWrapper", lambda **kwargs: object()
        )
        monkeypatch.setattr(
            tavily_utils,
            "config",
            SimpleNamespace(
                get_tavily_api_key=lambda: "test-key",
                tavily_extract_timeout_seconds=config.tavily_extract_timeout_seconds,
                tavily_search_timeout_seconds=config.tavily_search_timeout_seconds,
            ),
        )

    return _install


ENTRYPOINTS: dict[str, Callable[..., Awaitable[Any]]] = {
    "search_with_timeout": lambda **kw: tavily_utils.tavily_search_with_timeout(
        {"query": "q"}, **kw
    ),
    "search_inspected": lambda **kw: tavily_utils.search_tavily_inspected(
        "q", profile="news_basic", **kw
    ),
    "extract_inspected": lambda **kw: tavily_utils.extract_tavily_inspected(
        ["https://example.com/ir"], query="q", **kw
    ),
}


def _count_inspections(monkeypatch) -> list[Any]:
    inspected: list[Any] = []

    async def _record(raw: Any, query_text: str) -> Any:
        del query_text
        inspected.append(raw)
        return {"inspected": True}

    inspect_tavily_with(monkeypatch, _record)
    return inspected


@pytest.mark.asyncio
@pytest.mark.parametrize("entrypoint", sorted(ENTRYPOINTS))
async def test_inspection_gets_only_the_time_the_search_left(
    monkeypatch, fake_clock, install, entrypoint
):
    inspected = _count_inspections(monkeypatch)
    seen: list[float] = []
    real = async_utils.run_with_hard_timeout

    async def spy(coro, *, timeout, label):
        seen.append(timeout)
        return await real(coro, timeout=timeout, label=label)

    monkeypatch.setattr(async_utils, "run_with_hard_timeout", spy)
    # Inspection is bounded inside InspectionService.check_within.
    monkeypatch.setattr(inspection_service, "run_with_hard_timeout", spy)

    def _spend_three_seconds() -> None:
        fake_clock[0] += 3.0

    install(_Search(_spend_three_seconds))
    with enclosing_deadline(4.0):
        result = await ENTRYPOINTS[entrypoint]()

    assert result == {"inspected": True}
    assert inspected == [PAYLOAD]
    assert seen == [pytest.approx(4.0), pytest.approx(1.0)]


@pytest.mark.asyncio
@pytest.mark.parametrize("entrypoint", sorted(ENTRYPOINTS))
async def test_late_search_result_is_discarded_before_inspection(
    monkeypatch, fake_clock, install, entrypoint
):
    inspected = _count_inspections(monkeypatch)

    def _arrive_after_the_deadline() -> None:
        fake_clock[0] += 31.0  # past the 30 s default bound

    tool = _Search(_arrive_after_the_deadline)
    install(tool)

    assert await ENTRYPOINTS[entrypoint]() is None
    assert tool.calls == 1
    assert inspected == []


@pytest.mark.asyncio
@pytest.mark.parametrize("entrypoint", sorted(ENTRYPOINTS))
async def test_stalled_inspection_returns_none_within_the_bound(
    monkeypatch, install, entrypoint
):
    release = asyncio.Event()
    finished = asyncio.Event()

    async def _stubborn_inspection(raw: Any, query_text: str) -> Any:
        del query_text
        try:
            await release.wait()
        except asyncio.CancelledError:
            # Ignore cancellation, as a judge awaiting a slow client can.
            await release.wait()
        finished.set()
        return raw

    inspect_tavily_with(monkeypatch, _stubborn_inspection)
    install(_Search())
    bound = 0.05

    result = await async_utils.run_with_hard_timeout(
        ENTRYPOINTS[entrypoint](timeout=bound),
        timeout=SAFETY_NET_SECONDS,
        label="test_safety_net",
    )

    assert result is None
    # Returned while inspection was still stalled.
    assert not finished.is_set()
    release.set()
    await asyncio.wait_for(finished.wait(), timeout=SAFETY_NET_SECONDS)


@pytest.mark.asyncio
async def test_inspection_failure_still_propagates(monkeypatch, install):
    async def _broken(raw: Any, query_text: str) -> Any:
        raise RuntimeError("inspector misconfigured")

    inspect_tavily_with(monkeypatch, _broken)
    install(_Search())

    with pytest.raises(RuntimeError, match="inspector misconfigured"):
        await tavily_utils.search_tavily_inspected("q", profile="news_basic")


@pytest.mark.asyncio
async def test_an_enforced_inspection_failure_returns_none_like_a_timeout(
    monkeypatch, install
):
    """A fail-closed backend error is "could not inspect": no result, never
    the placeholder or the raw payload."""
    inspected: list[Any] = []

    async def _failed(raw: Any, query_text: str) -> Any:
        del query_text
        inspected.append(raw)
        return "TOOL_BLOCKED: INSPECTION_BACKEND_ERROR"

    inspect_tavily_with(monkeypatch, _failed, enforced="inspection_failed")
    install(_Search())

    assert await tavily_utils.search_tavily_inspected("q", profile="news_basic") is None
    assert len(inspected) == 1


@pytest.mark.asyncio
async def test_empty_search_result_skips_inspection(monkeypatch, install):
    inspected = _count_inspections(monkeypatch)

    class _Empty(_Search):
        async def _respond(self) -> Any:
            return None

    install(_Empty())

    assert await tavily_utils.search_tavily_inspected("q", profile="news_basic") is None
    assert inspected == []
