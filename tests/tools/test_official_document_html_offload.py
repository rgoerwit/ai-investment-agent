"""Official documents: HTML parsing runs on a bounded worker, not the event loop.

``BeautifulSoup`` parsing, link ranking and text flattening used to run inline in
``get_official_document``, so an 80 ms parse could overrun a 20 ms runner bound
and froze every other task meanwhile. They now run as one ``run_blocking_call``
under the extraction allowance and the in-flight limit the PDF path shares.
The download is stubbed at ``_download_official``; nothing touches the network.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.blocking_io import DEFAULT_WORKER_SLOTS, BlockingCallPolicy, InFlightLimit
from src.config import config
from src.tooling.inspector import InspectionDecision
from src.tools import official_documents as documents
from tests.helpers.stalled_workers import stalled_worker

_URL = "https://www.sec.gov/Archives/edgar/data/1/index.htm"
_LABEL = "official_document_html_extract"
_BODY = "Annual report statement of financial position figures. " * 12
_HTML = (
    "<html><body><p>"
    + _BODY
    + '</p><a href="/Archives/edgar/data/1/annual-report-2025.pdf">Annual report'
    + ' 2025</a><a href="https://elsewhere.example/x.pdf">other host</a>'
    + "</body></html>"
).encode()
_BOUND = 0.2
_GUARD_SECONDS = 5.0
_STALL_CEILING = 3.0


@pytest.fixture(autouse=True)
def _stubbed(monkeypatch) -> MagicMock:
    async def download(*args, **kwargs):
        del args, kwargs
        return _HTML, "text/html", _URL

    monkeypatch.setattr(documents, "_download_official", download)
    monkeypatch.setattr(config, "official_document_extract_timeout_seconds", _BOUND)
    service = MagicMock()
    service.check_within = AsyncMock(
        side_effect=lambda envelope, **_: _verdict(envelope)
    )
    monkeypatch.setattr(documents, "get_current_inspection_service", lambda: service)
    return service


def _verdict(envelope):
    return (
        InspectionDecision(action="allow", threat_level="safe", enforced="allowed"),
        envelope.content_text,
    )


async def _fetch() -> str:
    return await asyncio.wait_for(
        documents.get_official_document.ainvoke({"url": _URL}), _GUARD_SECONDS
    )


def _reply(reason: str) -> str:
    return f"STATUS: INSUFFICIENT_DATA\nREASON: {reason}"


def _stalling_text(monkeypatch, release: threading.Event, started: threading.Event):
    """Make the flattening step park until *release*; return the real one."""
    real = documents._html_document_text

    def parked(soup):
        started.set()
        release.wait(_STALL_CEILING)
        return real(soup)

    monkeypatch.setattr(documents, "_html_document_text", parked)


async def _heartbeat(ticks: list[float]) -> None:
    while True:
        ticks.append(time.monotonic())
        await asyncio.sleep(0.005)


@pytest.mark.asyncio
async def test_valid_html_keeps_its_text_and_candidate_paths(_stubbed):
    reply = await _fetch()

    assert reply.startswith("STATUS: EVIDENCE_FOUND\n")
    metadata = json.loads(reply.split("DOCUMENT_METADATA: ", 1)[1].split("\n", 1)[0])
    assert metadata["backend"] == "html"
    assert metadata["candidate_paths"] == [
        "/Archives/edgar/data/1/annual-report-2025.pdf"
    ]
    assert "statement of financial position" in reply
    sent = _stubbed.check_within.await_args.args[0]
    assert sent.content_text.startswith("Annual report statement")


@pytest.mark.asyncio
async def test_extract_html_is_the_pure_parse_rank_flatten_unit():
    text, paths = documents._extract_html(_HTML, _URL, "")

    assert text.startswith("Annual report statement of financial position")
    assert paths == ("/Archives/edgar/data/1/annual-report-2025.pdf",)


@pytest.mark.asyncio
async def test_a_stalled_parse_times_out_while_the_event_loop_keeps_ticking(
    monkeypatch, _stubbed
):
    started = threading.Event()
    ticks: list[float] = []
    with stalled_worker(_LABEL) as release:
        _stalling_text(monkeypatch, release, started)
        beat = asyncio.create_task(_heartbeat(ticks))
        try:
            began = time.monotonic()
            reply = await _fetch()
            elapsed = time.monotonic() - began
        finally:
            beat.cancel()

        assert reply == _reply("EXTRACTION_TIMEOUT")
        assert started.is_set()
        assert not release.is_set()
        assert elapsed < _STALL_CEILING
        # The loop ran during the bound; inline it would have been frozen.
        assert sum(1 for t in ticks if began <= t <= began + elapsed) >= 5
        _stubbed.check_within.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_saturated_limit_refuses_promptly_with_the_typed_reason(
    monkeypatch, _stubbed
):
    limit = InFlightLimit(1)
    assert limit.try_acquire()
    monkeypatch.setattr(
        documents,
        "_HTML_POLICY",
        BlockingCallPolicy(_LABEL, 20.0, in_flight=limit),
        raising=False,
    )
    began = time.monotonic()
    try:
        reply = await _fetch()
    finally:
        limit.release()

    assert reply == _reply("EXTRACTION_TIMEOUT")
    assert time.monotonic() - began < _BOUND
    assert limit.in_flight == 0
    _stubbed.check_within.assert_not_awaited()


@pytest.mark.asyncio
async def test_the_worker_slot_stays_held_until_the_stalled_worker_finishes(
    monkeypatch, _stubbed
):
    limit = InFlightLimit(1)
    monkeypatch.setattr(
        documents,
        "_HTML_POLICY",
        BlockingCallPolicy(_LABEL, 20.0, in_flight=limit),
        raising=False,
    )
    started = threading.Event()
    with stalled_worker(_LABEL) as release:
        _stalling_text(monkeypatch, release, started)
        assert await _fetch() == _reply("EXTRACTION_TIMEOUT")
        assert limit.in_flight == 1
        # A second document is refused while the stalled worker holds the slot.
        assert await _fetch() == _reply("EXTRACTION_TIMEOUT")
        assert limit.in_flight == 1
    assert limit.in_flight == 0


@pytest.mark.asyncio
async def test_a_late_result_is_never_inspected(monkeypatch, _stubbed):
    started = threading.Event()
    with stalled_worker(_LABEL) as release:
        _stalling_text(monkeypatch, release, started)
        assert await _fetch() == _reply("EXTRACTION_TIMEOUT")
    # The worker finished after the release, with a full result.
    _stubbed.check_within.assert_not_awaited()


def test_pdf_and_html_extraction_share_one_limit_sized_by_the_named_constant():
    limit = documents._HTML_POLICY.in_flight
    assert limit is not None
    assert limit.max_in_flight == DEFAULT_WORKER_SLOTS
    assert documents._PDF_POLICY.in_flight is limit
    assert documents._HTML_POLICY.label != documents._PDF_POLICY.label


@pytest.mark.asyncio
async def test_a_stalled_pypdf_worker_keeps_its_slot_and_saturation_is_refused(
    monkeypatch,
):
    """PDF extraction is bounded like HTML: a stalled worker holds a slot."""
    limit = InFlightLimit(1)
    monkeypatch.setattr(
        documents,
        "_PDF_POLICY",
        BlockingCallPolicy("official_document_pdf_extract", 0.05, in_flight=limit),
    )
    started = threading.Event()
    with stalled_worker("official_document_pdf_extract") as release:

        def stall(*args, **kwargs):
            started.set()
            release.wait()
            raise AssertionError("released stall must not be inspected")

        monkeypatch.setattr(documents, "_extract_with_pypdf", stall)
        monkeypatch.setattr(documents, "resolve_program", lambda name: None)
        for _ in range(2):
            with pytest.raises(documents.DocumentExtractionError) as raised:
                await documents.extract_pdf_text(
                    b"%PDF",
                    MagicMock(max_document_pages=10, max_selected_pages=2),
                    "",
                )
            assert raised.value.reason == "EXTRACTION_TIMEOUT"
        assert limit.in_flight == 1
        assert started.is_set()
    assert limit.in_flight == 0


@pytest.mark.asyncio
async def test_an_html_page_with_no_allowance_left_is_refused_before_a_worker(
    monkeypatch, _stubbed
):
    monkeypatch.setattr(config, "tool_execution_timeout_seconds", 1e-9)
    monkeypatch.setattr(
        documents,
        "_download_official",
        AsyncMock(return_value=(_HTML, "text/html", _URL)),
    )
    limit = documents._HTML_POLICY.in_flight
    assert limit is not None

    assert await _fetch() == _reply("EXTRACTION_TIMEOUT")
    assert limit.in_flight == 0
    _stubbed.check_within.assert_not_awaited()
