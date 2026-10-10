"""Official documents: how the time is shared, and the extraction child's cleanup.

The download may use what is left (caller deadline and tool runner) less the
PDF extraction reserve, and never more than its total cap, so a trickling body
stops there. Each extractor then gets what is left. A pdftotext child is killed
and reaped on every abnormal exit -- its own timeout, an error, or the caller's
cancellation -- with the reap bounded, shielded and its failure logged, never
leaked. Fake clocks, fake transports and releasable stalls; one real child
(``/bin/sleep``) for the cancellation path. Nothing touches the network.
"""

from __future__ import annotations

import asyncio
import gc
import os
import signal
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import httpx
import pytest

from src.config import Settings, config
from src.forensic_budget import AuditorBudgetPolicy
from src.llm_runtime import timeouts as llm_timeouts
from src.llm_runtime.timeouts import enclosing_deadline
from src.tooling.runtime import tool_call_deadline
from src.tools import official_documents as documents

_NOW = 1_000.0
_DOC_URL = "https://www.sec.gov/Archives/report.pdf"
_USEFUL = documents.ExtractedDocument("filing text " * 40, "pypdf", 1, (0,))
# A test-side guard so a broken bound fails instead of hanging the suite. It
# is never what a passing test waits for.
_GUARD_SECONDS = 5.0
_REAL_ASYNC_CLIENT = httpx.AsyncClient


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


def _spy_pypdf(monkeypatch, result: Any = _USEFUL) -> list[float]:
    """Record each pypdf allowance; never start a thread."""
    seen: list[float] = []

    async def spy(policy, fn):
        del fn
        seen.append(policy.hard_timeout_seconds)
        return result

    monkeypatch.setattr(documents, "run_blocking_call", spy)
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


def _passthrough_inspection(monkeypatch) -> None:
    class _Inspection:
        async def check(self, envelope: Any) -> Any:
            return envelope.content_text

    monkeypatch.setattr(
        documents, "get_current_inspection_service", lambda: _Inspection()
    )


# --- Settings --------------------------------------------------------------


def test_reserve_default_is_unset_and_derives_from_the_extract_cap(monkeypatch):
    assert Settings.model_fields["pdf_extraction_reserve_seconds"].default is None
    monkeypatch.setattr(config, "pdf_extraction_reserve_seconds", None)
    monkeypatch.setattr(config, "official_document_extract_timeout_seconds", 13.0)

    assert documents.pdf_extraction_reserve_seconds() == 13.0


@pytest.mark.parametrize("value", [0.0, 7.5])
def test_reserve_override_wins(monkeypatch, value):
    monkeypatch.setattr(config, "pdf_extraction_reserve_seconds", value)

    assert documents.pdf_extraction_reserve_seconds() == value


def test_negative_reserve_is_rejected(monkeypatch):
    monkeypatch.setenv("PDF_EXTRACTION_RESERVE_SECONDS", "-1")
    with pytest.raises(ValueError):
        Settings()


# --- Allocation --------------------------------------------------------------


def test_download_without_any_deadline_gets_its_total_cap(fake_clock):
    del fake_clock
    assert documents._download_allowance() == 35.0


def test_download_gets_caller_deadline_less_reserve(fake_clock):
    del fake_clock
    with enclosing_deadline(30.0):
        assert documents._download_allowance() == pytest.approx(10.0)


def test_download_gets_tool_runner_less_reserve(fake_clock):
    del fake_clock
    with tool_call_deadline(45.0):
        assert documents._download_allowance() == pytest.approx(25.0)


@pytest.mark.asyncio
async def test_less_than_the_reserve_left_opens_no_client(monkeypatch, fake_clock):
    del fake_clock
    timeouts = _mock_client(monkeypatch, lambda request: httpx.Response(200))

    with tool_call_deadline(15.0):
        result = await documents.get_official_document.ainvoke({"url": _DOC_URL})

    assert result == "STATUS: INSUFFICIENT_DATA\nREASON: DOCUMENT_RETRIEVAL_FAILED"
    assert timeouts == []


@pytest.mark.asyncio
async def test_a_download_using_its_whole_allowance_leaves_extraction_the_reserve(
    monkeypatch, fake_clock
):
    """Under the 45 s runner the download gets 25 s; extraction then gets 20 s."""

    def handler(request: httpx.Request) -> httpx.Response:
        # The download takes all the time it was given.
        fake_clock[0] += request.extensions["timeout"]["read"]
        return httpx.Response(
            200,
            headers={"content-type": "application/pdf"},
            content=b"%PDF-1.4 fake",
        )

    timeouts = _mock_client(monkeypatch, handler)
    seen = _spy_pypdf(monkeypatch)
    _passthrough_inspection(monkeypatch)

    with tool_call_deadline(45.0):
        result = await documents.get_official_document.ainvoke({"url": _DOC_URL})

    assert result.startswith("STATUS: EVIDENCE_FOUND")
    assert timeouts == [pytest.approx(25.0)]
    assert seen == [pytest.approx(documents.pdf_extraction_reserve_seconds())]


@pytest.mark.asyncio
async def test_spent_tool_runner_starts_neither_extractor(monkeypatch, fake_clock):
    seen = _spy_pypdf(monkeypatch)
    spawned: list[Any] = []

    async def spawn(*args: Any, **kwargs: Any) -> Any:
        spawned.append(args)
        raise AssertionError("pdftotext must not start")

    monkeypatch.setattr(documents.asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(documents.shutil, "which", lambda name: "/usr/bin/pdftotext")

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
async def test_trickling_download_stops_at_the_total_deadline(monkeypatch):
    body = _Trickle()
    _mock_client(
        monkeypatch,
        lambda request: httpx.Response(
            200, headers={"content-type": "application/pdf"}, stream=body
        ),
    )
    monkeypatch.setattr(
        config, "official_document_download_total_timeout_seconds", 0.05
    )

    async with asyncio.timeout(_GUARD_SECONDS):
        result = await documents.get_official_document.ainvoke({"url": _DOC_URL})

    assert result == "STATUS: INSUFFICIENT_DATA\nREASON: DOCUMENT_RETRIEVAL_FAILED"
    assert body.started
    assert not body.release.is_set()
    assert not body.finished


@pytest.mark.asyncio
async def test_per_phase_timeout_is_unchanged_inside_a_looser_total(monkeypatch):
    timeouts = _mock_client(
        monkeypatch, lambda request: httpx.Response(200, text="x" * 300)
    )
    monkeypatch.setattr(config, "official_document_download_timeout_seconds", 12.0)

    await documents._download_official(_DOC_URL, 1_000)

    assert timeouts == [12.0]


@pytest.mark.asyncio
async def test_redirect_and_size_guards_hold_inside_the_total_bound(monkeypatch):
    def redirect(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "https://evil.example/x.pdf"})

    _mock_client(monkeypatch, redirect)
    with pytest.raises(documents.DocumentExtractionError) as exc_info:
        await documents._download_official(_DOC_URL, 1_000)
    assert exc_info.value.reason == "UNAPPROVED_DOCUMENT_HOST"

    _mock_client(monkeypatch, lambda request: httpx.Response(200, content=b"x" * 50))
    with pytest.raises(documents.DocumentExtractionError) as exc_info:
        await documents._download_official(_DOC_URL, 10)
    assert exc_info.value.reason == "DOCUMENT_SIZE_LIMIT"


# --- Subprocess cleanup ------------------------------------------------------


class _Child:
    """A fake pdftotext child with releasable stalls in communicate() and wait()."""

    def __init__(
        self,
        *,
        stdout: bytes = b"page one " * 40 + b"\f",
        stall: bool = False,
        stall_reap: bool = False,
        exits_on_its_own: bool = False,
        reap_error: BaseException | None = None,
    ) -> None:
        self.stdout = stdout
        self.stall = stall
        self.returncode: int | None = None
        self.in_communicate = asyncio.Event()
        self.reap_started = asyncio.Event()
        self.release_reap = asyncio.Event()
        if not stall_reap:
            self.release_reap.set()
        self.exits_on_its_own = exits_on_its_own
        self.reap_error = reap_error
        self.kill_calls = 0
        self.reaped = False

    async def communicate(self, payload: bytes) -> tuple[bytes, bytes]:
        del payload
        self.in_communicate.set()
        if self.stall:
            await asyncio.Event().wait()
        self.returncode = 0
        return self.stdout, b""

    def kill(self) -> None:
        self.kill_calls += 1
        if self.exits_on_its_own:
            # Exited between the returncode check and the kill.
            self.returncode = 0
            raise ProcessLookupError
        self.returncode = -9

    async def wait(self) -> int:
        self.reap_started.set()
        await self.release_reap.wait()
        if self.reap_error is not None:
            raise self.reap_error
        self.reaped = True
        assert self.returncode is not None
        return self.returncode


def _install_child(monkeypatch, child: _Child) -> list[Any]:
    spawned: list[Any] = []

    async def spawn(*args: Any, **kwargs: Any) -> _Child:
        del kwargs
        spawned.append(args)
        return child

    monkeypatch.setattr(documents.asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(documents.shutil, "which", lambda name: "/usr/bin/pdftotext")
    return spawned


async def _pdftotext() -> documents.ExtractedDocument:
    return await documents._extract_with_pdftotext(
        b"%PDF", max_pages=5, selected_pages=2, keywords=()
    )


@contextmanager
def _loop_errors() -> Iterator[list[dict[str, Any]]]:
    """Errors the running loop reports, e.g. "Task exception was never retrieved"."""
    errors: list[dict[str, Any]] = []
    loop = asyncio.get_running_loop()
    previous = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: errors.append(context))
    try:
        yield errors
    finally:
        loop.set_exception_handler(previous)


@pytest.mark.asyncio
async def test_normal_completion_neither_kills_nor_leaves_a_reaper(monkeypatch):
    child = _Child()
    _install_child(monkeypatch, child)

    extracted = await _pdftotext()

    assert extracted.backend == "pdftotext"
    assert child.kill_calls == 0
    assert documents._PENDING_REAPS == set()


@pytest.mark.asyncio
async def test_cancel_during_communicate_kills_and_reaps(monkeypatch):
    child = _Child(stall=True)
    _install_child(monkeypatch, child)

    task = asyncio.ensure_future(_pdftotext())
    await child.in_communicate.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert child.kill_calls == 1
    assert child.reaped
    assert documents._PENDING_REAPS == set()


@pytest.mark.asyncio
async def test_cancel_again_during_cleanup_still_reaps(monkeypatch):
    child = _Child(stall=True, stall_reap=True)
    _install_child(monkeypatch, child)

    with _loop_errors() as loop_errors:
        task = asyncio.ensure_future(_pdftotext())
        await child.in_communicate.wait()
        task.cancel()
        await child.reap_started.wait()
        task.cancel()  # Interrupts only the wait: the reap is shielded.

        with pytest.raises(asyncio.CancelledError):
            await task
        assert child.kill_calls == 1
        assert not child.reaped

        child.release_reap.set()
        while documents._PENDING_REAPS:
            await asyncio.sleep(0)

        assert child.reaped
        assert loop_errors == []


@pytest.mark.asyncio
async def test_child_exiting_concurrently_is_still_reaped(monkeypatch):
    child = _Child(stall=True, exits_on_its_own=True)
    _install_child(monkeypatch, child)

    task = asyncio.ensure_future(_pdftotext())
    await child.in_communicate.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert child.kill_calls == 1
    assert child.reaped


@pytest.mark.asyncio
async def test_local_timeout_kills_reaps_and_reports_extraction_timeout(monkeypatch):
    child = _Child(stall=True)
    _install_child(monkeypatch, child)
    monkeypatch.setattr(config, "official_document_extract_timeout_seconds", 0.01)

    async with asyncio.timeout(_GUARD_SECONDS):
        with pytest.raises(documents.DocumentExtractionError) as exc_info:
            await _pdftotext()

    assert exc_info.value.reason == "EXTRACTION_TIMEOUT"
    assert child.kill_calls == 1
    assert child.reaped


@pytest.mark.asyncio
async def test_stalled_reap_is_bounded_and_its_late_failure_logged_not_leaked(
    monkeypatch,
):
    child = _Child(stall=True, stall_reap=True, reap_error=OSError("reap failed"))
    _install_child(monkeypatch, child)
    monkeypatch.setattr(config, "subprocess_cleanup_timeout_seconds", 0.01)

    task = asyncio.ensure_future(_pdftotext())
    await child.in_communicate.wait()
    logs = _record_logs(monkeypatch)
    with _loop_errors() as loop_errors:
        task.cancel()
        async with asyncio.timeout(_GUARD_SECONDS):
            with pytest.raises(asyncio.CancelledError):
                await task
        # The caller was released with the reap still pending.
        assert len(documents._PENDING_REAPS) == 1

        child.release_reap.set()
        while documents._PENDING_REAPS:
            await asyncio.sleep(0)
        gc.collect()
        await asyncio.sleep(0)

    events = [event for _, event, _ in logs.events]
    assert "official_document_subprocess_reap_pending" in events
    assert "official_document_subprocess_reap_failed" in events
    assert loop_errors == []


@pytest.mark.asyncio
async def test_real_child_is_killed_and_reaped_on_cancellation():
    """The same create_subprocess_exec + cleanup path, with a harmless real child.

    ``close_fds=False`` and no ``cwd=`` keep the runtime on ``posix_spawn``
    (see .claude/rules/tests.md).
    """
    process = await asyncio.create_subprocess_exec(
        "/bin/sleep",
        "30",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        close_fds=False,
    )
    try:
        task = asyncio.ensure_future(
            documents._communicate_bounded(
                process, b"", timeout=config.official_document_extract_timeout_seconds
            )
        )
        await asyncio.sleep(0)  # Let it enter communicate().
        task.cancel()

        async with asyncio.timeout(_GUARD_SECONDS):
            with pytest.raises(asyncio.CancelledError):
                await task

        assert process.returncode == -signal.SIGKILL
        with pytest.raises(ChildProcessError):
            os.waitpid(process.pid, os.WNOHANG)
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
