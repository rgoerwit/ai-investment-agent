"""Official documents: how the time is shared, and the extraction child's cleanup.

The whole sequence -- download, extraction, inspection -- runs inside the tool's
allowance and is shared by fractions of it, so the split holds for any runner
value: the download may use ``OFFICIAL_DOCUMENT_DOWNLOAD_FRACTION`` (a trickling
body stops there); of the rest, ``OFFICIAL_DOCUMENT_INSPECTION_FRACTION`` is held
back from extraction for content inspection, which gets whatever is then left
and fails the document closed when it cannot finish. The pdftotext child's own
cleanup is tested with the spawn helper (``tests/test_subprocesses.py``). Fake
clocks, fake transports and releasable stalls. Nothing touches the network.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from src import blocking_io
from src.async_utils import run_with_hard_timeout
from src.config import Settings, config
from src.forensic_budget import AuditorBudgetPolicy
from src.llm_runtime import timeouts as llm_timeouts
from src.llm_runtime.timeouts import (
    deadline_margin,
    enclosing_deadline,
    tool_call_deadline,
)
from src.tooling.inspection_service import InspectionService
from src.tooling.inspector import InspectionDecision
from src.tooling.runtime import ToolCallFailed, ToolExecutionService, ToolInvocation
from src.tools import official_documents as documents

_NOW = 1_000.0
_DOC_URL = "https://www.sec.gov/Archives/report.pdf"
_USEFUL = documents.ExtractedDocument("filing text " * 40, "pypdf", 1, (0,))
_PDF = b"%PDF-1.4 fake"
_HTML = b"<html><body>" + b"annual report figures " * 20 + b"</body></html>"
# A test-side guard so a broken bound fails instead of hanging the suite. It
# is never what a passing test waits for.
_GUARD_SECONDS = 5.0
_REAL_ASYNC_CLIENT = httpx.AsyncClient
DOWNLOAD = "official_document_download_fraction"
INSPECTION = "official_document_inspection_fraction"
RUNNER = "tool_execution_timeout_seconds"


class _LogRecorder:
    """Stands in for the module logger (capture_logs misses cached loggers)."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str, dict[str, Any]]] = []

    def __getattr__(self, level: str) -> Any:
        def record(event: str, **fields: Any) -> None:
            self.events.append((level, event, fields))

        return record


def _record_logs(monkeypatch) -> _LogRecorder:
    recorder = _LogRecorder()
    monkeypatch.setattr(documents, "logger", recorder)
    return recorder


@pytest.fixture
def fake_clock(monkeypatch) -> list[float]:
    """A frozen monotonic clock for the deadline helpers; advance via clock[0]."""
    clock = [_NOW]
    monkeypatch.setattr(llm_timeouts, "now", lambda: clock[0])
    return clock


@pytest.fixture(autouse=True)
def _no_dns(monkeypatch) -> None:
    async def _public(url: str) -> None:
        del url

    monkeypatch.setattr(documents, "_ensure_public_hostname", _public)


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


def _spy_pypdf(monkeypatch, result: Any = _USEFUL, *, spend: Any = None) -> list[float]:
    """Record each pypdf allowance at the thread boundary; never start a thread.

    Spies below ``run_blocking_call``'s own deadline clamp, so what is recorded
    is the bound the worker would really get, and a spent allowance records
    nothing. ``spend(timeout)``, when given, runs as if pypdf used that time.
    """
    seen: list[float] = []

    async def spy(coro, *, timeout, label):
        del label
        coro.close()  # the to_thread coroutine: closed unstarted, no thread
        seen.append(timeout)
        if spend is not None:
            return spend(timeout)
        return result

    monkeypatch.setattr(blocking_io, "run_with_hard_timeout", spy)
    return seen


def _mock_client(monkeypatch, handler) -> list[float]:
    """Route ``httpx.AsyncClient`` through a ``MockTransport``; record timeouts."""
    timeouts: list[float] = []

    def factory(*, timeout: float, **kwargs: Any) -> httpx.AsyncClient:
        del kwargs
        timeouts.append(timeout)
        return _REAL_ASYNC_CLIENT(
            timeout=timeout, transport=httpx.MockTransport(handler)
        )

    monkeypatch.setattr(documents.httpx, "AsyncClient", factory)
    return timeouts


def _respond(content: bytes, content_type: str, clock: list[float] | None = None):
    """A handler that, given a clock, spends the whole download share first."""

    def handler(request: httpx.Request) -> httpx.Response:
        if clock is not None:
            clock[0] += request.extensions["timeout"]["read"]
        return httpx.Response(
            200, headers={"content-type": content_type}, content=content
        )

    return handler


class _Inspection(InspectionService):
    """The real ``check_within`` bound around a stand-in ``evaluate``."""

    def __init__(self, *, stall: bool = False) -> None:
        super().__init__()
        self.stall = stall
        self.calls = 0
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.bounds: list[float] = []

    async def check_within(self, envelope, *, seconds, label):
        self.bounds.append(seconds)
        return await super().check_within(envelope, seconds=seconds, label=label)

    async def evaluate(self, envelope: Any) -> tuple[InspectionDecision, Any]:
        self.calls += 1
        self.entered.set()
        if self.stall:
            await self.release.wait()
        decision = InspectionDecision(
            action="allow", threat_level="safe", enforced="allowed"
        )
        return decision, envelope.content_text


def _use_inspection(monkeypatch, inspection: _Inspection | None = None) -> _Inspection:
    inspection = inspection or _Inspection()
    monkeypatch.setattr(documents, "get_current_inspection_service", lambda: inspection)
    return inspection


async def _guarded(awaitable: Any) -> Any:
    """Deadline-only safety net: a broken bound fails here, never hangs."""
    return await run_with_hard_timeout(
        awaitable, timeout=_GUARD_SECONDS, label="test_guard"
    )


def _shares(allowance: float) -> tuple[float, float]:
    """(download, inspection reserve) for an allowance, at the shipped fractions."""
    download = allowance * 0.55
    return download, (allowance - download) * 0.25


# --- Settings --------------------------------------------------------------


def test_shipped_fractions():
    assert Settings.model_fields[DOWNLOAD].default == 0.55
    assert Settings.model_fields[INSPECTION].default == 0.25


@pytest.mark.parametrize("field", [DOWNLOAD, INSPECTION])
@pytest.mark.parametrize("value", [0.0, -0.1, 1.0, 1.5])
def test_a_fraction_outside_zero_to_one_is_rejected(field, value):
    with pytest.raises(ValidationError):
        Settings(**{field: value})


@pytest.mark.parametrize("field", [DOWNLOAD, INSPECTION])
@pytest.mark.parametrize("value", [1e-6, 0.999])
def test_a_fraction_just_inside_the_bounds_is_accepted(field, value):
    assert getattr(Settings(**{field: value}), field) == value


@pytest.mark.parametrize(
    "removed",
    [
        "pdf_extraction_reserve_seconds",
        "official_document_download_total_timeout_seconds",
        "official_document_download_timeout_seconds",
    ],
)
def test_the_absolute_document_settings_are_gone(removed):
    assert removed not in Settings.model_fields


# --- Allocation --------------------------------------------------------------


def test_with_no_deadline_the_runner_setting_is_the_allowance(fake_clock):
    del fake_clock
    allocation = documents._allocate()
    download, reserve = _shares(45.0)
    assert allocation.deadline == _NOW + 45.0
    assert allocation.download_seconds == pytest.approx(download)
    assert allocation.inspection_reserve == pytest.approx(reserve)


def test_a_caller_deadline_shrinks_every_share(fake_clock):
    del fake_clock
    with enclosing_deadline(30.0):
        allocation = documents._allocate()
    assert allocation.download_seconds == pytest.approx(_shares(30.0)[0])
    assert allocation.inspection_reserve == pytest.approx(_shares(30.0)[1])


def test_under_the_runner_the_allowance_is_kept_back_by_its_margin(fake_clock):
    del fake_clock
    with tool_call_deadline(45.0):
        allocation = documents._allocate()
    allowance = 45.0 - deadline_margin(45.0)
    assert allocation.deadline == pytest.approx(_NOW + allowance)
    assert allocation.download_seconds == pytest.approx(_shares(allowance)[0])


@pytest.mark.parametrize("runner", [300.0, 45.0, 20.0, 10.0, 1.0, 0.01])
def test_every_share_is_positive_and_fits_for_any_runner(
    monkeypatch, fake_clock, runner
):
    """By construction: no runner value starves the download or extraction.

    20 s is the old extraction reserve: a runner at or below it once left the
    download nothing, so no document was ever fetched.
    """
    del fake_clock
    monkeypatch.setattr(config, RUNNER, runner)
    allocation = documents._allocate()
    extraction = runner - allocation.download_seconds - allocation.inspection_reserve
    assert allocation.download_seconds > 0
    assert allocation.inspection_reserve > 0
    assert extraction > 0
    assert allocation.download_seconds + allocation.inspection_reserve < runner


@pytest.mark.asyncio
async def test_a_runner_at_the_old_reserve_still_downloads(monkeypatch, fake_clock):
    del fake_clock
    monkeypatch.setattr(config, RUNNER, 20.0)
    timeouts = _mock_client(monkeypatch, _respond(_PDF, "application/pdf"))
    _spy_pypdf(monkeypatch)
    _use_inspection(monkeypatch)

    result = await documents.get_official_document.ainvoke({"url": _DOC_URL})

    assert result.startswith("STATUS: EVIDENCE_FOUND")
    assert timeouts == [pytest.approx(_shares(20.0)[0])]


@pytest.mark.asyncio
async def test_a_spent_allowance_opens_no_client(monkeypatch, fake_clock):
    timeouts = _mock_client(monkeypatch, _respond(_PDF, "application/pdf"))
    inspection = _use_inspection(monkeypatch)

    with enclosing_deadline(1.0):
        fake_clock[0] += 1.0  # exactly at the deadline
        result = await documents.get_official_document.ainvoke({"url": _DOC_URL})

    assert result == "STATUS: INSUFFICIENT_DATA\nREASON: DOCUMENT_ALLOWANCE_SPENT"
    assert timeouts == []
    assert inspection.calls == 0


@pytest.mark.asyncio
async def test_a_download_using_its_whole_share_leaves_extraction_and_inspection_theirs(
    monkeypatch, fake_clock
):
    timeouts = _mock_client(monkeypatch, _respond(_PDF, "application/pdf", fake_clock))
    seen = _spy_pypdf(monkeypatch)
    inspection = _use_inspection(monkeypatch)

    with tool_call_deadline(45.0):
        result = await documents.get_official_document.ainvoke({"url": _DOC_URL})

    allowance = 45.0 - deadline_margin(45.0)
    download, reserve = _shares(allowance)
    assert result.startswith("STATUS: EVIDENCE_FOUND")
    assert timeouts == [pytest.approx(download)]
    # pypdf: what the download left, less the inspection reserve (below its cap).
    assert seen == [pytest.approx(allowance - download - reserve)]
    assert inspection.bounds == [pytest.approx(allowance - download)]


@pytest.mark.asyncio
async def test_extraction_that_spends_its_bound_still_leaves_inspection_the_reserve(
    monkeypatch, fake_clock
):
    _mock_client(monkeypatch, _respond(_PDF, "application/pdf"))

    def spend(timeout: float) -> Any:
        # A stall ends a moment after its bound, as a real timer does.
        fake_clock[0] += timeout + 1e-3
        raise TimeoutError("pypdf stalled")

    seen = _spy_pypdf(monkeypatch, spend=spend)
    monkeypatch.setattr(documents, "resolve_program", lambda name: "/usr/bin/pdftotext")

    async def spawn(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("pdftotext must not start: only the reserve is left")

    monkeypatch.setattr(documents, "run_program", spawn)
    logs = _record_logs(monkeypatch)
    allocation = documents._allocate()
    fake_clock[0] += allocation.download_seconds  # the download used its share

    with pytest.raises(documents.DocumentExtractionError) as exc_info:
        await documents.extract_pdf_text(_PDF, _audit_policy(), "", allocation)

    assert exc_info.value.reason == "EXTRACTION_TIMEOUT"
    assert len(seen) == 1
    # What is left is the inspection reserve, to the timer's overrun.
    assert allocation.left() == pytest.approx(allocation.inspection_reserve, abs=2e-3)
    assert (
        "info",
        "official_document_extraction_skipped",
        {"backend": "pdftotext", "reason": "EXTRACTION_TIMEOUT"},
    ) in logs.events


@pytest.mark.asyncio
async def test_spent_tool_runner_starts_neither_extractor(monkeypatch, fake_clock):
    seen = _spy_pypdf(monkeypatch)
    spawned: list[Any] = []

    async def spawn(*args: Any, **kwargs: Any) -> Any:
        spawned.append(args)
        raise AssertionError("pdftotext must not start")

    monkeypatch.setattr(documents, "run_program", spawn)
    monkeypatch.setattr(documents, "resolve_program", lambda name: "/usr/bin/pdftotext")

    logs = _record_logs(monkeypatch)
    with tool_call_deadline(10.0):
        fake_clock[0] += 15.0
        with pytest.raises(documents.DocumentExtractionError) as exc_info:
            await documents.extract_pdf_text(b"%PDF", _audit_policy(), "")

    assert exc_info.value.reason == "EXTRACTION_TIMEOUT"
    assert seen == []
    assert spawned == []
    assert (
        "info",
        "official_document_extraction_skipped",
        {"backend": "pdftotext", "reason": "EXTRACTION_TIMEOUT"},
    ) in logs.events


# --- Inspection inside the allocation -------------------------------------------


@pytest.mark.asyncio
async def test_inspection_with_nothing_left_never_starts_and_fails_closed(
    monkeypatch, fake_clock
):
    # Plain text needs no extraction worker; the download spends the whole allowance.
    def handler(request: httpx.Request) -> httpx.Response:
        fake_clock[0] += 100.0
        return httpx.Response(
            200,
            headers={"content-type": "text/plain"},
            content=b"annual report figures " * 20,
        )

    _mock_client(monkeypatch, handler)
    inspection = _use_inspection(monkeypatch)
    logs = _record_logs(monkeypatch)

    with pytest.raises(ToolCallFailed) as raised:
        await documents.get_official_document.ainvoke({"url": _DOC_URL})

    assert raised.value.reason == "INSPECTION_TIMEOUT"
    assert inspection.calls == 0
    assert ("warning", "official_document_inspection_timeout") in {
        (level, event) for level, event, _ in logs.events
    }


@pytest.mark.asyncio
async def test_a_stalled_inspection_fails_closed_within_the_allowance(monkeypatch):
    monkeypatch.setattr(config, RUNNER, 0.2)
    _mock_client(monkeypatch, _respond(_HTML, "text/html"))
    inspection = _use_inspection(monkeypatch, _Inspection(stall=True))

    with pytest.raises(ToolCallFailed) as raised:
        await _guarded(documents.get_official_document.ainvoke({"url": _DOC_URL}))

    assert raised.value.reason == "INSPECTION_TIMEOUT"
    assert inspection.entered.is_set()
    assert inspection.bounds[0] <= 0.2
    inspection.release.set()
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_a_stalled_extractor_reports_its_own_timeout_not_the_runners(monkeypatch):
    """The equal-timer race: under the real tool runner, extraction (bounded inside
    the allowance, less the inspection reserve) fires first, so the reply is a
    typed EXTRACTION_TIMEOUT, never the runner's bare timeout."""
    monkeypatch.setattr(config, RUNNER, 0.3)
    _mock_client(monkeypatch, _respond(_PDF, "application/pdf"))
    _use_inspection(monkeypatch)
    monkeypatch.setattr(documents, "resolve_program", lambda name: None)
    release = threading.Event()
    monkeypatch.setattr(
        documents,
        "_extract_with_pypdf",
        lambda payload, **kwargs: (release.wait(_GUARD_SECONDS), _USEFUL)[1],
    )

    async def runner(args: dict[str, Any]) -> Any:
        return await documents.get_official_document.ainvoke(args)

    try:
        result = await _guarded(
            ToolExecutionService().execute(
                ToolInvocation(
                    name="get_official_document",
                    args={"url": _DOC_URL},
                    source="preflight",
                ),
                runner=runner,
            )
        )
    finally:
        release.set()

    assert result.value == "STATUS: INSUFFICIENT_DATA\nREASON: EXTRACTION_TIMEOUT"


class _Trickle(httpx.AsyncByteStream):
    """A body that sends its first bytes, then stalls until released."""

    def __init__(self) -> None:
        self.release = asyncio.Event()
        self.started = False
        self.finished = False

    async def __aiter__(self):
        self.started = True
        yield b"%PDF-1.4 "
        await self.release.wait()
        self.finished = True
        yield b"rest"


@pytest.mark.asyncio
async def test_trickling_download_stops_at_its_share(monkeypatch):
    body = _Trickle()
    _mock_client(
        monkeypatch,
        lambda request: httpx.Response(
            200, headers={"content-type": "application/pdf"}, stream=body
        ),
    )
    monkeypatch.setattr(config, RUNNER, 0.1)

    result = await _guarded(documents.get_official_document.ainvoke({"url": _DOC_URL}))

    assert result == "STATUS: INSUFFICIENT_DATA\nREASON: DOCUMENT_DOWNLOAD_TIMEOUT"
    assert body.started
    assert not body.release.is_set()
    assert not body.finished


@pytest.mark.asyncio
async def test_httpx_gets_the_download_share_per_phase(monkeypatch, fake_clock):
    del fake_clock
    timeouts = _mock_client(
        monkeypatch, lambda request: httpx.Response(200, text="x" * 300)
    )
    monkeypatch.setattr(config, DOWNLOAD, 0.4)

    await documents._download_official(
        _DOC_URL, 1_000, allocation=documents._allocate()
    )

    assert timeouts == [pytest.approx(45.0 * 0.4)]


@pytest.mark.asyncio
async def test_redirect_and_size_guards_hold_inside_the_total_bound(monkeypatch):
    def redirect(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "https://evil.example/x.pdf"})

    _mock_client(monkeypatch, redirect)
    with pytest.raises(documents.DocumentExtractionError) as exc_info:
        await documents._download_official(
            _DOC_URL, 1_000, allocation=documents._allocate()
        )
    assert exc_info.value.reason == "UNAPPROVED_DOCUMENT_HOST"

    _mock_client(monkeypatch, lambda request: httpx.Response(200, content=b"x" * 50))
    with pytest.raises(documents.DocumentExtractionError) as exc_info:
        await documents._download_official(
            _DOC_URL, 10, allocation=documents._allocate()
        )
    assert exc_info.value.reason == "DOCUMENT_SIZE_LIMIT"
