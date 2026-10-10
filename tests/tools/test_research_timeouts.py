"""Research deadlines: one ``Settings`` field per scope, clamped to the caller.

Covers web search (Tavily async, profile, extract and sync; DuckDuckGo), the
filing lookup and per-fetcher filing fetch, the official-document download, DNS
check and PDF extraction, and the editor's reference fetch. Each scope is
checked four ways: the shipped default (read from the field definition, never
by instantiating ``Settings``), an override reaching the real request or
blocking call, a caller's enclosing deadline winning over a looser default, and
a spent or fired deadline taking the call site's existing fallback without
starting work. Fake clocks, spies and tiny timeouts only; nothing touches the
network.
"""

from __future__ import annotations

import asyncio
import importlib
import threading
import time
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

import src.async_utils as async_utils
import src.blocking_io as blocking_io
import src.editor_tools as editor_tools
import src.tavily_utils as tavily_utils
from src.blocking_io import InFlightLimit
from src.config import Settings, config
from src.data import gap_fill
from src.data.filings.base import FilingFetcher
from src.forensic_budget import AuditorBudgetPolicy
from src.llm_runtime import timeouts as llm_timeouts
from src.llm_runtime.timeouts import enclosing_deadline
from src.tools import official_documents as documents
from src.tools import research, shared

# ``src.data.filings`` re-exports its singleton as ``registry``, shadowing the
# submodule attribute; import_module returns the module itself.
registry_module = importlib.import_module("src.data.filings.registry")

_NOW = 1_000.0

SHIPPED_DEFAULTS = {
    "tavily_search_timeout_seconds": 30.0,
    "tavily_extract_timeout_seconds": 30.0,
    "gap_fill_search_timeout_seconds": 5.0,
    "ddg_search_timeout_seconds": 8.0,
    "ddg_http_timeout_seconds": 5,
    "official_filings_lookup_timeout_seconds": 20.0,
    "filing_fetch_timeout_seconds": 30.0,
    "official_document_download_timeout_seconds": 35.0,
    "official_document_dns_timeout_seconds": 5.0,
    "official_document_extract_timeout_seconds": 20.0,
    "official_document_download_total_timeout_seconds": 35.0,
    "subprocess_cleanup_timeout_seconds": 5.0,
    "reference_fetch_timeout_seconds": 10.0,
}


@pytest.fixture
def fake_clock(monkeypatch) -> list[float]:
    """A frozen monotonic clock for the deadline helpers; advance via clock[0]."""
    clock = [_NOW]
    monkeypatch.setattr(llm_timeouts, "now", lambda: clock[0])
    return clock


def _spy_hard_timeout(monkeypatch, result: Any = None) -> list[float]:
    """Replace ``run_with_hard_timeout`` where it is imported at call time.

    Records each timeout and never runs the work: a coroutine is closed, an
    executor future cancelled.
    """
    seen: list[float] = []

    async def spy(awaitable, *, timeout, label):
        del label
        seen.append(timeout)
        if asyncio.iscoroutine(awaitable):
            awaitable.close()
        else:
            awaitable.cancel()
        return result

    monkeypatch.setattr(async_utils, "run_with_hard_timeout", spy)
    return seen


class _Inert:
    """A fake Tavily tool that records calls and returns a fixed payload.

    ``ainvoke`` counts when the request coroutine is *created*, not when it
    runs, so a spent-deadline test also catches a coroutine built and dropped.
    """

    def __init__(self, payload: Any = None, *, delay: float = 0.0) -> None:
        self.payload = payload
        self.delay = delay
        self.async_calls = 0
        self.sync_calls = 0

    def ainvoke(self, payload: dict[str, Any]) -> Any:
        del payload
        self.async_calls += 1
        return self._respond()

    async def _respond(self) -> Any:
        if self.delay:
            await asyncio.sleep(self.delay)
        return self.payload

    def invoke(self, payload: dict[str, Any]) -> Any:
        del payload
        self.sync_calls += 1
        if self.delay:
            time.sleep(self.delay)
        return self.payload


@pytest.fixture
def passthrough_inspection(monkeypatch) -> list[Any]:
    inspected: list[Any] = []

    async def _record(raw: Any, query_text: str) -> Any:
        del query_text
        inspected.append(raw)
        return raw

    monkeypatch.setattr(tavily_utils, "_inspect_tavily_result", _record)
    return inspected


# --- Defaults --------------------------------------------------------------


@pytest.mark.parametrize("field, expected", sorted(SHIPPED_DEFAULTS.items()))
def test_shipped_default_is_unchanged(field, expected):
    assert Settings.model_fields[field].default == expected


@pytest.mark.parametrize("field", sorted(SHIPPED_DEFAULTS))
def test_a_non_positive_override_is_rejected(monkeypatch, field):
    # A zero bound would make every call a timeout, which reads as no results.
    monkeypatch.setenv(Settings.model_fields[field].validation_alias, "0")
    with pytest.raises(ValidationError):
        Settings()


@pytest.mark.parametrize(
    "policy, field",
    [
        (tavily_utils._TAVILY_SYNC_POLICY, "tavily_search_timeout_seconds"),
        (documents._PDF_POLICY, "official_document_extract_timeout_seconds"),
        (documents._DNS_POLICY, "official_document_dns_timeout_seconds"),
    ],
    ids=lambda value: getattr(value, "label", value),
)
def test_policy_constant_matches_its_setting_default(policy, field):
    # The constant's number is only a default; the call site applies the field.
    assert policy.hard_timeout_seconds == Settings.model_fields[field].default


# --- Tavily async search ---------------------------------------------------


@pytest.fixture
def tavily_tool(monkeypatch) -> _Inert:
    tool = _Inert({"results": []})
    monkeypatch.setattr(tavily_utils, "_tavily_tool", tool)
    monkeypatch.setattr(
        tavily_utils, "_build_tavily_tool", lambda profile, include_answer=False: tool
    )
    return tool


async def _search_all_async(**kwargs: Any) -> None:
    await tavily_utils.tavily_search_with_timeout({"query": "q"}, **kwargs)
    await tavily_utils.search_tavily_inspected("q", profile="news_basic", **kwargs)


@pytest.mark.asyncio
async def test_tavily_search_override_reaches_hard_timeout(
    monkeypatch, tavily_tool, passthrough_inspection
):
    seen = _spy_hard_timeout(monkeypatch)
    monkeypatch.setattr(config, "tavily_search_timeout_seconds", 12.5)

    await _search_all_async()

    assert seen == [12.5, 12.5]


@pytest.mark.asyncio
async def test_tavily_search_caller_deadline_wins(
    monkeypatch, fake_clock, tavily_tool, passthrough_inspection
):
    seen = _spy_hard_timeout(monkeypatch)

    with enclosing_deadline(4.0):
        await _search_all_async()
        # An explicit per-call bound is clamped too, not only the default.
        await _search_all_async(timeout=9.0)
        # A tighter explicit bound stands: the clamp is min(), not "outer wins".
        await _search_all_async(timeout=1.5)

    assert seen == [pytest.approx(v) for v in (4.0, 4.0, 4.0, 4.0, 1.5, 1.5)]


@pytest.mark.asyncio
async def test_tavily_search_timeout_returns_none(monkeypatch, passthrough_inspection):
    tool = _Inert({"results": []}, delay=1.0)
    monkeypatch.setattr(tavily_utils, "_tavily_tool", tool)
    monkeypatch.setattr(
        tavily_utils, "_build_tavily_tool", lambda profile, include_answer=False: tool
    )
    monkeypatch.setattr(config, "tavily_search_timeout_seconds", 0.01)

    assert await tavily_utils.tavily_search_with_timeout({"query": "q"}) is None
    assert (
        await tavily_utils.search_tavily_inspected("q", profile="finance_deep") is None
    )
    assert passthrough_inspection == []


@pytest.mark.asyncio
async def test_tavily_search_spent_deadline_creates_no_request(
    monkeypatch, fake_clock, tavily_tool, passthrough_inspection
):
    seen = _spy_hard_timeout(monkeypatch)

    with enclosing_deadline(1.0):
        fake_clock[0] += 5.0
        assert await tavily_utils.tavily_search_with_timeout({"query": "q"}) is None
        assert (
            await tavily_utils.search_tavily_inspected("q", profile="news_basic")
            is None
        )

    assert seen == []
    assert tavily_tool.async_calls == 0
    assert passthrough_inspection == []


@pytest.mark.asyncio
async def test_gap_fill_passes_its_own_search_setting(monkeypatch):
    monkeypatch.setattr(config, "gap_fill_search_timeout_seconds", 2.5)
    calls: list[dict[str, Any]] = []

    async def search_fn(query, **kwargs):
        del query
        calls.append(kwargs)
        return {}

    await gap_fill._fetch_tavily_gaps_impl(
        SimpleNamespace(tavily_client=object()),
        "7203.T",
        ["revenueGrowth"],
        yf_module=SimpleNamespace(Ticker=lambda symbol: SimpleNamespace(info={})),
        search_fn=search_fn,
        query_builder=lambda symbol, name, terms: "q",
        info_policy=gap_fill.GAP_FILL_YFINANCE_INFO_POLICY,
    )

    assert calls == [{"profile": "finance_deep", "timeout": 2.5}]


@pytest.mark.asyncio
async def test_gap_fill_search_bound_is_clamped_by_the_real_search(
    monkeypatch, fake_clock, tavily_tool, passthrough_inspection
):
    # Gap fill hands its 5 s to search_tavily_inspected, which shrinks it.
    seen = _spy_hard_timeout(monkeypatch)

    with enclosing_deadline(0.75):
        await tavily_utils.search_tavily_inspected(
            "q",
            profile="finance_deep",
            timeout=config.gap_fill_search_timeout_seconds,
        )

    assert seen == [pytest.approx(0.75)]


# --- Tavily extract --------------------------------------------------------


class _FakeExtract:
    instances: list[_FakeExtract] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.calls = 0
        _FakeExtract.instances.append(self)

    def ainvoke(self, payload: dict[str, Any]) -> Any:
        # Counted at coroutine creation, like _Inert.ainvoke.
        del payload
        self.calls += 1
        return self._respond()

    async def _respond(self) -> Any:
        return {"results": []}


@pytest.fixture
def fake_extract(monkeypatch) -> type[_FakeExtract]:
    import langchain_tavily
    import langchain_tavily._utilities as tavily_utilities

    _FakeExtract.instances = []
    monkeypatch.setattr(langchain_tavily, "TavilyExtract", _FakeExtract)
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
    return _FakeExtract


async def _extract() -> Any:
    return await tavily_utils.extract_tavily_inspected(
        ["https://example.com/ir"], query="guidance"
    )


@pytest.mark.asyncio
async def test_tavily_extract_override_is_separate_from_search(
    monkeypatch, fake_extract, passthrough_inspection
):
    seen = _spy_hard_timeout(monkeypatch)
    monkeypatch.setattr(tavily_utils.config, "tavily_extract_timeout_seconds", 6.0)
    monkeypatch.setattr(tavily_utils.config, "tavily_search_timeout_seconds", 99.0)

    await _extract()

    assert seen == [6.0]


@pytest.mark.asyncio
async def test_tavily_extract_caller_deadline_wins(
    monkeypatch, fake_clock, fake_extract, passthrough_inspection
):
    seen = _spy_hard_timeout(monkeypatch)

    with enclosing_deadline(2.0):
        await _extract()

    assert seen == [pytest.approx(2.0)]


@pytest.mark.asyncio
async def test_tavily_extract_timeout_returns_none(
    monkeypatch, fake_extract, passthrough_inspection
):
    async def stall(self, payload):
        del self, payload
        await asyncio.sleep(1)

    monkeypatch.setattr(_FakeExtract, "ainvoke", stall)
    monkeypatch.setattr(tavily_utils.config, "tavily_extract_timeout_seconds", 0.01)

    assert await _extract() is None
    assert passthrough_inspection == []


@pytest.mark.asyncio
async def test_tavily_extract_spent_deadline_creates_no_request(
    monkeypatch, fake_clock, fake_extract, passthrough_inspection
):
    seen = _spy_hard_timeout(monkeypatch)

    with enclosing_deadline(1.0):
        fake_clock[0] += 5.0
        assert await _extract() is None

    assert seen == []
    assert [instance.calls for instance in fake_extract.instances] == [0]


# --- Tavily sync search ----------------------------------------------------


@pytest.fixture
def sync_limit(monkeypatch) -> InFlightLimit:
    """A fresh limit so no test inherits another file's orphaned workers."""
    fresh = InFlightLimit(2)
    monkeypatch.setattr(tavily_utils, "_TAVILY_SYNC_IN_FLIGHT", fresh)
    monkeypatch.setattr(
        tavily_utils,
        "_TAVILY_SYNC_POLICY",
        tavily_utils._TAVILY_SYNC_POLICY.__class__(
            "tavily-research-test",
            tavily_utils._TAVILY_SYNC_POLICY.hard_timeout_seconds,
            in_flight=fresh,
        ),
    )
    return fresh


def _spy_sync_call(monkeypatch, *, run: bool) -> list[Any]:
    """Replace ``run_blocking_call_sync``; record each policy.

    With *run*, the worker body is executed inline (no caller bound), which
    exposes the worker's own shared-deadline check.
    """
    seen: list[Any] = []

    def spy(policy, fn):
        seen.append(policy)
        return fn() if run else None

    monkeypatch.setattr(tavily_utils, "run_blocking_call_sync", spy)
    return seen


def test_sync_search_override_reaches_blocking_policy(
    monkeypatch, sync_limit, tavily_tool
):
    seen = _spy_sync_call(monkeypatch, run=False)
    monkeypatch.setattr(config, "tavily_search_timeout_seconds", 7.0)

    tavily_utils.search_tavily_sync_inspected("q", profile="news_basic")

    assert [policy.hard_timeout_seconds for policy in seen] == [7.0]
    # The in-flight cap survives with_timeout().
    assert seen[0].in_flight is sync_limit


def test_sync_search_caller_deadline_wins(
    monkeypatch, fake_clock, sync_limit, tavily_tool
):
    seen = _spy_sync_call(monkeypatch, run=False)

    with enclosing_deadline(3.0):
        tavily_utils.search_tavily_sync_inspected("q", profile="news_basic")

    assert [policy.hard_timeout_seconds for policy in seen] == [pytest.approx(3.0)]


def test_sync_worker_deadline_starts_from_the_clamped_bound(
    monkeypatch, sync_limit, passthrough_inspection
):
    # Real clock: the search takes 0.3 s, well inside the 30 s default but
    # past a 0.1 s caller deadline. Run inline, the worker must see its shared
    # deadline already gone and discard the result before inspection; with a
    # deadline taken from the unclamped 30 s it would inspect it.
    tool = _Inert({"results": ["late"]}, delay=0.3)
    monkeypatch.setattr(
        tavily_utils, "_build_tavily_tool", lambda profile, include_answer=False: tool
    )
    _spy_sync_call(monkeypatch, run=True)

    with enclosing_deadline(0.1):
        result = tavily_utils.search_tavily_sync_inspected("q", profile="news_basic")

    assert result is None
    assert tool.sync_calls == 1
    assert passthrough_inspection == []


def test_sync_worker_inspects_within_an_unspent_clamped_bound(
    monkeypatch, sync_limit, passthrough_inspection
):
    # Control for the test above: same inline run, ample caller deadline.
    tool = _Inert({"results": ["fresh"]})
    monkeypatch.setattr(
        tavily_utils, "_build_tavily_tool", lambda profile, include_answer=False: tool
    )
    _spy_sync_call(monkeypatch, run=True)

    with enclosing_deadline(5.0):
        result = tavily_utils.search_tavily_sync_inspected("q", profile="news_basic")

    assert result == {"results": ["fresh"]}
    assert passthrough_inspection == [{"results": ["fresh"]}]


def test_sync_search_timeout_returns_none(monkeypatch, sync_limit):
    release = threading.Event()
    tool = _Inert()
    tool.invoke = lambda payload: release.wait()  # type: ignore[method-assign]
    monkeypatch.setattr(
        tavily_utils, "_build_tavily_tool", lambda profile, include_answer=False: tool
    )
    monkeypatch.setattr(config, "tavily_search_timeout_seconds", 0.05)
    try:
        assert (
            tavily_utils.search_tavily_sync_inspected("q", profile="news_basic") is None
        )
    finally:
        release.set()


def test_sync_search_spent_deadline_starts_no_worker(
    monkeypatch, fake_clock, sync_limit, tavily_tool, passthrough_inspection
):
    seen = _spy_sync_call(monkeypatch, run=True)

    with enclosing_deadline(1.0):
        fake_clock[0] += 5.0
        assert (
            tavily_utils.search_tavily_sync_inspected("q", profile="news_basic") is None
        )

    assert seen == []
    assert tavily_tool.sync_calls == 0
    assert sync_limit.in_flight == 0
    assert passthrough_inspection == []


# --- DuckDuckGo ------------------------------------------------------------


class _RecordingDDGS:
    timeouts: list[Any] = []

    def __init__(self, *args: Any, timeout: Any = None, **kwargs: Any) -> None:
        del args, kwargs
        _RecordingDDGS.timeouts.append(timeout)

    def text(self, query: str, max_results: int = 5) -> list[dict[str, str]]:
        return [{"title": query, "href": "https://example.com", "body": "b"}]


@pytest.fixture
def ddgs(monkeypatch) -> type[_RecordingDDGS]:
    _RecordingDDGS.timeouts = []
    monkeypatch.setattr(shared, "DDGS_AVAILABLE", True)
    monkeypatch.setattr("ddgs.DDGS", _RecordingDDGS)
    return _RecordingDDGS


def _passthrough_hard_timeout(monkeypatch) -> list[float]:
    """Record each ``run_with_hard_timeout`` bound, then really run it."""
    seen: list[float] = []
    real = async_utils.run_with_hard_timeout

    async def spy(awaitable, *, timeout, label):
        seen.append(timeout)
        return await real(awaitable, timeout=timeout, label=label)

    monkeypatch.setattr(async_utils, "run_with_hard_timeout", spy)
    return seen


@pytest.mark.asyncio
async def test_ddg_overrides_reach_the_wait_and_the_client(monkeypatch, ddgs):
    seen = _passthrough_hard_timeout(monkeypatch)
    monkeypatch.setattr(config, "ddg_search_timeout_seconds", 6.5)
    monkeypatch.setattr(config, "ddg_http_timeout_seconds", 3)

    result = await shared._ddg_search("acme")

    assert result and result[0]["title"] == "acme"
    assert seen == [6.5]
    assert ddgs.timeouts == [3]


@pytest.mark.asyncio
async def test_ddg_caller_deadline_wins_for_both_scopes(monkeypatch, fake_clock, ddgs):
    # The fake clock drives only the clamp; the real wait uses the loop clock
    # and the fake client answers at once.
    seen = _passthrough_hard_timeout(monkeypatch)

    with enclosing_deadline(2.4):
        result = await shared._ddg_search("acme")

    assert result and result[0]["title"] == "acme"
    assert seen == [pytest.approx(2.4)]
    # ddgs takes whole seconds: the clamped 2.4 s rounds up, never the 5 s.
    assert ddgs.timeouts == [3]


@pytest.mark.asyncio
async def test_ddg_spent_deadline_submits_no_search(monkeypatch, fake_clock, ddgs):
    seen = _spy_hard_timeout(monkeypatch, result=[{"never": True}])

    with enclosing_deadline(1.0):
        fake_clock[0] += 5.0
        assert await shared._ddg_search("acme") == []

    assert seen == []
    assert ddgs.timeouts == []


# (A stalled DDG search returning [] at the configured bound is covered by
# tests/financial/test_filings.py and tests/tools/test_ddg_concurrency.py.)


# --- Filing lookup and filing fetch ----------------------------------------


class _Fetcher(FilingFetcher):
    def __init__(self, *, delay: float = 0.0) -> None:
        self.delay = delay
        self.started = 0

    @property
    def supported_suffixes(self) -> list[str]:
        return ["T"]

    @property
    def source_name(self) -> str:
        return "fake"

    def is_available(self) -> bool:
        return True

    def get_filing_data(self, ticker: str) -> Any:  # type: ignore[override]
        # Counted at coroutine creation, so a dropped coroutine is caught too.
        del ticker
        self.started += 1
        return self._respond()

    async def _respond(self) -> Any:
        if self.delay:
            await asyncio.sleep(self.delay)
        return SimpleNamespace(to_report_string=lambda: "report")


class _WaitForSpy:
    """Stands in for ``asyncio`` inside ``src.tools.research``."""

    def __init__(self) -> None:
        self.timeouts: list[float] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(asyncio, name)

    async def wait_for(self, coro, timeout):
        self.timeouts.append(timeout)
        return await asyncio.wait_for(coro, timeout)


def _lookup_registry(monkeypatch, fetch) -> list[str]:
    started: list[str] = []

    def _fetch(ticker: str) -> Any:
        # Counted at coroutine creation, so a dropped coroutine is caught too.
        started.append(ticker)
        return fetch(ticker)

    monkeypatch.setattr(
        "src.data.filings.registry", SimpleNamespace(fetch=_fetch), raising=True
    )
    return started


async def _no_filing(_ticker: str) -> None:
    return None


@pytest.mark.asyncio
async def test_filing_lookup_override_reaches_wait_for(monkeypatch):
    spy = _WaitForSpy()
    monkeypatch.setattr(research, "asyncio", spy)
    monkeypatch.setattr(config, "official_filings_lookup_timeout_seconds", 11.0)
    _lookup_registry(monkeypatch, _no_filing)

    await research.get_official_filings.ainvoke({"ticker": "7203.T"})

    assert spy.timeouts == [11.0]


@pytest.mark.asyncio
async def test_filing_lookup_caller_deadline_wins(monkeypatch, fake_clock):
    spy = _WaitForSpy()
    monkeypatch.setattr(research, "asyncio", spy)
    _lookup_registry(monkeypatch, _no_filing)

    with enclosing_deadline(4.0):
        await research.get_official_filings.ainvoke({"ticker": "7203.T"})

    assert spy.timeouts == [pytest.approx(4.0)]


@pytest.mark.asyncio
async def test_filing_lookup_spent_deadline_reports_timeout_unstarted(
    monkeypatch, fake_clock
):
    started = _lookup_registry(monkeypatch, _no_filing)

    with enclosing_deadline(1.0):
        fake_clock[0] += 5.0
        result = await research.get_official_filings.ainvoke({"ticker": "7203.T"})

    assert "REASON: LOOKUP_TIMEOUT" in result
    assert started == []


def _registry_with(fetcher: _Fetcher) -> Any:
    registry = registry_module.FilingRegistry()
    registry.register(fetcher)
    return registry


@pytest.mark.asyncio
async def test_filing_fetch_override_reaches_hard_timeout(monkeypatch):
    seen: list[float] = []

    async def spy(coro, *, timeout, label):
        del label
        seen.append(timeout)
        return await coro

    monkeypatch.setattr(registry_module, "run_with_hard_timeout", spy)
    monkeypatch.setattr(config, "filing_fetch_timeout_seconds", 9.0)

    assert await _registry_with(_Fetcher()).fetch("7203.T") is not None
    assert seen == [9.0]


@pytest.mark.asyncio
async def test_filing_fetch_caller_deadline_wins(monkeypatch, fake_clock):
    seen: list[float] = []

    async def spy(coro, *, timeout, label):
        del label
        seen.append(timeout)
        return await coro

    monkeypatch.setattr(registry_module, "run_with_hard_timeout", spy)

    with enclosing_deadline(6.0):
        await _registry_with(_Fetcher()).fetch("7203.T")

    assert seen == [pytest.approx(6.0)]


@pytest.mark.asyncio
async def test_filing_fetch_timeout_returns_none(monkeypatch):
    monkeypatch.setattr(config, "filing_fetch_timeout_seconds", 0.01)

    assert await _registry_with(_Fetcher(delay=1.0)).fetch("7203.T") is None


@pytest.mark.asyncio
async def test_filing_fetch_spent_deadline_starts_no_fetcher(monkeypatch, fake_clock):
    fetcher = _Fetcher()

    with enclosing_deadline(1.0):
        fake_clock[0] += 5.0
        assert await _registry_with(fetcher).fetch("7203.T") is None

    assert fetcher.started == 0


# --- Official document: download, DNS check, PDF extraction ----------------

_DOC_URL = "https://www.sec.gov/Archives/report.pdf"


class _RecordingClient:
    """Fake ``httpx.AsyncClient``: records its timeout, then fails on entry."""

    def __init__(self, timeouts: list[Any], error: BaseException) -> None:
        self._timeouts = timeouts
        self._error = error

    def __call__(self, *, timeout: Any, **kwargs: Any) -> _RecordingClient:
        del kwargs
        self._timeouts.append(timeout)
        return self

    async def __aenter__(self) -> Any:
        raise self._error

    async def __aexit__(self, *exc: Any) -> None:
        return None


class _Halt(Exception):
    pass


def _patch_client(monkeypatch, module, error: BaseException) -> list[Any]:
    timeouts: list[Any] = []
    monkeypatch.setattr(module.httpx, "AsyncClient", _RecordingClient(timeouts, error))
    return timeouts


@pytest.mark.asyncio
async def test_document_download_override_reaches_the_client(monkeypatch):
    timeouts = _patch_client(monkeypatch, documents, _Halt())
    monkeypatch.setattr(config, "official_document_download_timeout_seconds", 12.0)

    with pytest.raises(_Halt):
        await documents._download_official(_DOC_URL, 1_000)

    assert timeouts == [12.0]


@pytest.mark.asyncio
async def test_document_download_caller_deadline_wins(monkeypatch, fake_clock):
    timeouts = _patch_client(monkeypatch, documents, _Halt())

    # The download gets the caller's 27 s less the 20 s extraction reserve.
    with enclosing_deadline(27.0), pytest.raises(_Halt):
        await documents._download_official(_DOC_URL, 1_000)

    assert timeouts == [pytest.approx(7.0)]


@pytest.mark.asyncio
async def test_document_download_timeout_is_retrieval_failed(monkeypatch):
    _patch_client(monkeypatch, documents, httpx.ReadTimeout("slow"))

    result = await documents.get_official_document.ainvoke({"url": _DOC_URL})

    assert result == "STATUS: INSUFFICIENT_DATA\nREASON: DOCUMENT_RETRIEVAL_FAILED"


@pytest.mark.asyncio
async def test_document_download_spent_deadline_opens_no_client(
    monkeypatch, fake_clock
):
    timeouts = _patch_client(monkeypatch, documents, _Halt())

    with enclosing_deadline(1.0):
        fake_clock[0] += 5.0
        result = await documents.get_official_document.ainvoke({"url": _DOC_URL})

    # The same reply an httpx timeout gets.
    assert result == "STATUS: INSUFFICIENT_DATA\nREASON: DOCUMENT_RETRIEVAL_FAILED"
    assert timeouts == []


def _spy_documents_blocking(monkeypatch, result: Any = None) -> list[Any]:
    """Replace the module's ``run_blocking_call``; record each policy, never run.

    The real call's own deadline clamp is applied first, so what is recorded
    is the bound a worker would get, and a spent deadline raises unrecorded.
    """
    seen: list[Any] = []

    async def spy(policy, fn):
        del fn
        seen.append(blocking_io._within_deadline(policy))
        if isinstance(result, BaseException):
            raise result
        return result

    monkeypatch.setattr(documents, "run_blocking_call", spy)
    return seen


@pytest.mark.asyncio
async def test_dns_check_override_reaches_blocking_policy(monkeypatch):
    seen = _spy_documents_blocking(monkeypatch, result={"8.8.8.8"})
    monkeypatch.setattr(config, "official_document_dns_timeout_seconds", 2.0)

    await documents._ensure_public_hostname(_DOC_URL)

    assert [policy.hard_timeout_seconds for policy in seen] == [2.0]
    assert seen[0].label == documents._DNS_POLICY.label


@pytest.mark.asyncio
async def test_dns_check_caller_deadline_wins(monkeypatch, fake_clock):
    seen = _spy_documents_blocking(monkeypatch, result={"8.8.8.8"})

    with enclosing_deadline(0.5):
        await documents._ensure_public_hostname(_DOC_URL)

    assert [policy.hard_timeout_seconds for policy in seen] == [pytest.approx(0.5)]


@pytest.mark.asyncio
async def test_dns_check_timeout_fails_the_document(monkeypatch):
    _spy_documents_blocking(monkeypatch, result=TimeoutError("dns"))

    with pytest.raises(documents.DocumentExtractionError) as exc_info:
        await documents._ensure_public_hostname(_DOC_URL)

    assert exc_info.value.reason == "DOCUMENT_DNS_FAILED"


@pytest.mark.asyncio
async def test_dns_check_spent_deadline_resolves_nothing(monkeypatch, fake_clock):
    seen = _spy_documents_blocking(monkeypatch, result={"8.8.8.8"})

    with enclosing_deadline(1.0):
        fake_clock[0] += 5.0
        with pytest.raises(documents.DocumentExtractionError) as exc_info:
            await documents._ensure_public_hostname(_DOC_URL)

    assert exc_info.value.reason == "DOCUMENT_DNS_FAILED"
    assert seen == []


def _audit_policy() -> AuditorBudgetPolicy:
    return AuditorBudgetPolicy(
        search_calls=1,
        document_calls=1,
        filing_calls=1,
        metrics_calls=1,
        news_calls=1,
        calculation_calls=1,
        max_document_bytes=1_000_000,
        max_document_pages=10,
        max_selected_pages=2,
        max_evidence_chars=1000,
        max_tool_iterations=1,
        max_llm_calls=1,
    )


_USEFUL = documents.ExtractedDocument("filing text " * 40, "pypdf", 1, (0,))


@pytest.mark.asyncio
async def test_pdf_extract_override_reaches_pypdf_policy(monkeypatch):
    seen = _spy_documents_blocking(monkeypatch, result=_USEFUL)
    monkeypatch.setattr(config, "official_document_extract_timeout_seconds", 4.0)

    assert await documents.extract_pdf_text(b"%PDF", _audit_policy(), "") is _USEFUL
    assert [policy.hard_timeout_seconds for policy in seen] == [4.0]
    assert seen[0].label == documents._PDF_POLICY.label


@pytest.mark.asyncio
async def test_pdf_extract_caller_deadline_wins(monkeypatch, fake_clock):
    seen = _spy_documents_blocking(monkeypatch, result=_USEFUL)

    with enclosing_deadline(3.0):
        await documents.extract_pdf_text(b"%PDF", _audit_policy(), "")

    assert [policy.hard_timeout_seconds for policy in seen] == [pytest.approx(3.0)]


class _FakeProcess:
    """A child that runs (``returncode is None``) until it finishes or is killed."""

    def __init__(self, stdout: bytes = b"", *, stall: bool = False) -> None:
        self._stdout = stdout
        self._stall = stall
        self.returncode: int | None = None
        self.killed = False
        self.reaped = False

    async def communicate(self, payload: bytes) -> tuple[bytes, bytes]:
        del payload
        if self._stall:
            await asyncio.sleep(1)
        self.returncode = 0
        return self._stdout, b""

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    async def wait(self) -> int:
        self.reaped = True
        return self.returncode if self.returncode is not None else 0


class _SubprocessAsyncio:
    """Stands in for ``asyncio`` in ``official_documents`` for the pdftotext path."""

    def __init__(self, process: _FakeProcess) -> None:
        self.process = process
        self.spawned = 0
        self.timeouts: list[float] = []
        self.subprocess = asyncio.subprocess

    def __getattr__(self, name: str) -> Any:
        return getattr(asyncio, name)

    async def create_subprocess_exec(self, *args: Any, **kwargs: Any) -> _FakeProcess:
        del args, kwargs
        self.spawned += 1
        return self.process

    async def wait_for(self, coro, timeout):
        self.timeouts.append(timeout)
        return await asyncio.wait_for(coro, timeout)


def _pdftotext(monkeypatch, process: _FakeProcess) -> _SubprocessAsyncio:
    fake = _SubprocessAsyncio(process)
    monkeypatch.setattr(documents, "asyncio", fake)
    monkeypatch.setattr(documents.shutil, "which", lambda name: "/usr/bin/pdftotext")
    return fake


async def _run_pdftotext() -> Any:
    return await documents._extract_with_pdftotext(
        b"%PDF", max_pages=5, selected_pages=2, keywords=()
    )


@pytest.mark.asyncio
async def test_pdftotext_override_reaches_its_wait(monkeypatch):
    fake = _pdftotext(monkeypatch, _FakeProcess(b"page one\f"))
    monkeypatch.setattr(config, "official_document_extract_timeout_seconds", 4.0)

    extracted = await _run_pdftotext()

    assert extracted.backend == "pdftotext"
    assert fake.timeouts == [4.0]


@pytest.mark.asyncio
async def test_pdftotext_caller_deadline_wins(monkeypatch, fake_clock):
    fake = _pdftotext(monkeypatch, _FakeProcess(b"page one\f"))

    with enclosing_deadline(1.5):
        await _run_pdftotext()

    assert fake.timeouts == [pytest.approx(1.5)]


@pytest.mark.asyncio
async def test_pdftotext_timeout_kills_and_reports_extraction_timeout(monkeypatch):
    process = _FakeProcess(stall=True)
    _pdftotext(monkeypatch, process)
    monkeypatch.setattr(config, "official_document_extract_timeout_seconds", 0.01)

    with pytest.raises(documents.DocumentExtractionError) as exc_info:
        await _run_pdftotext()

    assert exc_info.value.reason == "EXTRACTION_TIMEOUT"
    assert process.killed
    assert process.reaped


@pytest.mark.asyncio
async def test_pdf_extract_timeout_still_reports_extraction_timeout(monkeypatch):
    # pypdf times out; pdftotext is missing; today's reason is kept.
    _spy_documents_blocking(monkeypatch, result=TimeoutError("pypdf"))
    monkeypatch.setattr(documents.shutil, "which", lambda name: None)

    with pytest.raises(documents.DocumentExtractionError) as exc_info:
        await documents.extract_pdf_text(b"%PDF", _audit_policy(), "")

    assert exc_info.value.reason == "EXTRACTION_TIMEOUT"


@pytest.mark.asyncio
async def test_pdf_extract_spent_deadline_runs_neither_backend(monkeypatch, fake_clock):
    seen = _spy_documents_blocking(monkeypatch, result=_USEFUL)
    fake = _pdftotext(monkeypatch, _FakeProcess(b"page one\f"))

    with enclosing_deadline(1.0):
        fake_clock[0] += 5.0
        with pytest.raises(documents.DocumentExtractionError) as exc_info:
            await documents.extract_pdf_text(b"%PDF", _audit_policy(), "")

    assert exc_info.value.reason == "EXTRACTION_TIMEOUT"
    assert seen == []
    assert fake.spawned == 0


# --- Editor reference fetch ------------------------------------------------

_REF_URL = "https://example.com/article"


@pytest.mark.asyncio
async def test_reference_fetch_override_reaches_the_client(monkeypatch):
    timeouts = _patch_client(monkeypatch, editor_tools, httpx.ReadTimeout("slow"))
    monkeypatch.setattr(config, "reference_fetch_timeout_seconds", 4.0)

    result = await editor_tools.fetch_reference_content.ainvoke({"url": _REF_URL})

    assert timeouts == [4.0]
    assert result == "FETCH_FAILED: Request timed out after 4 seconds"


@pytest.mark.asyncio
async def test_reference_fetch_default_reply_is_unchanged(monkeypatch):
    timeouts = _patch_client(monkeypatch, editor_tools, httpx.ReadTimeout("slow"))

    result = await editor_tools.fetch_reference_content.ainvoke({"url": _REF_URL})

    assert timeouts == [10.0]
    assert result == "FETCH_FAILED: Request timed out after 10 seconds"


@pytest.mark.asyncio
async def test_reference_fetch_caller_deadline_wins(monkeypatch, fake_clock):
    timeouts = _patch_client(monkeypatch, editor_tools, httpx.ReadTimeout("slow"))

    with enclosing_deadline(2.5):
        await editor_tools.fetch_reference_content.ainvoke({"url": _REF_URL})

    assert timeouts == [pytest.approx(2.5)]


@pytest.mark.asyncio
async def test_reference_fetch_spent_deadline_opens_no_client(monkeypatch, fake_clock):
    timeouts = _patch_client(monkeypatch, editor_tools, _Halt())

    with enclosing_deadline(1.0):
        fake_clock[0] += 5.0
        result = await editor_tools.fetch_reference_content.ainvoke({"url": _REF_URL})

    assert result.startswith("FETCH_FAILED: Request timed out")
    assert timeouts == []
