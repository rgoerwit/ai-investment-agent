"""Official documents: the enforced inspection outcome is what reaches the ledger.

A ``TOOL_BLOCKED`` placeholder used to come back from ``check_within`` as if it
were the document text, so the tool wrapped it in ``STATUS: EVIDENCE_FOUND``
with primary authority, the outer inspection allowed that explanation, and the
ledger bound a blocked document as primary evidence. Each case here runs the
real tool through a real ``ToolExecutionService`` with the evidence ledger and
the outer inspection hook, and asserts together what the model is told, the
authority it is shown, the ledger record and whether a claim can bind to it.
Downloads are stubbed at ``_download_official``; nothing touches the network.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any, Literal

import pytest

from src.tooling.evidence_recorder import EvidenceRecorder, bind_fetched_evidence
from src.tooling.inspection_hook import ContentInspectionHook
from src.tooling.inspection_service import InspectionService
from src.tooling.inspector import InspectionDecision, InspectionEnvelope
from src.tooling.runtime import ToolExecutionService, ToolInvocation, tool_error_text
from src.tools import official_documents as documents

_URL = "https://www.sec.gov/Archives/edgar/data/1/report.txt"
_ORIGINAL = "original filing figures " * 20
_CLEAN = "sanitized filing figures " * 20
_SECRET = "sk-secret-token-in-exception"

Mode = Literal["warn", "sanitize", "block"]
Policy = Literal["fail_open", "fail_closed"]


class _Scripted:
    """An inspector that answers every envelope with one scripted outcome."""

    def __init__(self, outcome: InspectionDecision | BaseException) -> None:
        self.outcome = outcome
        self.calls = 0

    async def inspect(self, envelope: InspectionEnvelope) -> InspectionDecision:
        del envelope
        self.calls += 1
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


class _Stalled:
    """An inspector that never answers until released."""

    def __init__(self) -> None:
        self.release = asyncio.Event()

    async def inspect(self, envelope: InspectionEnvelope) -> InspectionDecision:
        del envelope
        await self.release.wait()
        return InspectionDecision(action="allow", threat_level="safe")


class _ShortBound(InspectionService):
    """The real ``check_within`` with the tool's allowance shortened to 50 ms."""

    async def check_within(self, envelope, *, seconds, label):
        return await super().check_within(envelope, seconds=0.05, label=label)


def _decision(action: str, sanitized: str | None = None) -> InspectionDecision:
    return InspectionDecision(
        action=action,  # type: ignore[arg-type]
        threat_level="high",
        threat_types=["prompt_injection"],
        findings=["planted finding"],
        reason="planted",
        sanitized_content=sanitized,
    )


@pytest.fixture(autouse=True)
def _download(monkeypatch) -> None:
    async def fake(url, max_bytes, extra_hosts=(), *, allocation, progress=None):
        del max_bytes, extra_hosts, allocation, progress
        return _ORIGINAL.encode(), "text/plain", url

    monkeypatch.setattr(documents, "_download_official", fake)


async def _run(
    monkeypatch,
    inner: InspectionService,
    outer: InspectionService | None = None,
) -> tuple[str, Any, Any]:
    """Fetch through the tool service; return reply, ledger record, binding."""
    monkeypatch.setattr(documents, "get_current_inspection_service", lambda: inner)
    recorder = EvidenceRecorder()
    service = ToolExecutionService(
        [recorder, ContentInspectionHook(outer or InspectionService())]
    )

    async def runner(args: dict[str, Any]) -> Any:
        return await documents.get_official_document.ainvoke(args)

    try:
        result = await service.execute(
            ToolInvocation(
                name="get_official_document",
                args={"url": _URL},
                source="preflight",
                agent_key="foreign_language_analyst",
            ),
            runner=runner,
        )
        reply = str(result.value)
    except Exception as exc:
        reply = tool_error_text(exc)
    records = recorder.snapshot()
    assert len(records) == 1
    return reply, records[0], bind_fetched_evidence(records, _URL)


def _assert_evidence(reply: str, record: Any, binding: Any, text: str) -> None:
    assert reply.startswith("STATUS: EVIDENCE_FOUND")
    assert '"authority": "PRIMARY_REGISTRY"' in reply
    assert text in reply
    assert (record.execution_status, record.evidence_status) == (
        "SUCCEEDED",
        "EVIDENCE_FOUND",
    )
    assert not record.blocked
    assert dict(record.authority_by_url)[_URL] == "PRIMARY_REGISTRY"
    assert binding is not None and binding.authority == "PRIMARY_REGISTRY"


def _assert_not_evidence(
    reply: str,
    record: Any,
    binding: Any,
    *,
    execution_status: str,
    reason: str,
) -> None:
    assert "EVIDENCE_FOUND" not in reply
    assert "DOCUMENT_METADATA" not in reply
    assert "authority" not in reply
    assert record.execution_status == execution_status
    assert record.evidence_status != "EVIDENCE_FOUND"
    assert record.authority_by_url == ()
    assert record.reason == reason
    assert binding is None


def _service(
    inspector: Any, mode: Mode = "block", policy: Policy = "fail_closed"
) -> InspectionService:
    return InspectionService(inspector, mode=mode, fail_policy=policy)


@pytest.mark.asyncio
async def test_allowed_document_is_primary_evidence(monkeypatch):
    inspector = _Scripted(InspectionDecision(action="allow", threat_level="safe"))

    reply, record, binding = await _run(monkeypatch, _service(inspector))

    _assert_evidence(reply, record, binding, _ORIGINAL)
    assert inspector.calls == 1


@pytest.mark.asyncio
async def test_sanitized_document_is_evidence_with_the_sanitized_text(monkeypatch):
    inspector = _Scripted(_decision("sanitize", _CLEAN))

    reply, record, binding = await _run(monkeypatch, _service(inspector, "sanitize"))

    _assert_evidence(reply, record, binding, _CLEAN)
    assert _ORIGINAL not in reply


@pytest.mark.asyncio
async def test_block_in_block_mode_is_recorded_blocked(monkeypatch):
    inner = _service(_Scripted(_decision("block")))

    reply, record, binding = await _run(monkeypatch, inner, outer=inner)

    assert reply == "TOOL_BLOCKED: OFFICIAL_DOCUMENT_BLOCKED"
    _assert_not_evidence(
        reply, record, binding, execution_status="BLOCKED", reason="TOOL_BLOCKED"
    )
    assert record.blocked


@pytest.mark.asyncio
async def test_block_followed_by_an_outer_allow_stays_blocked(monkeypatch):
    """The outer hook never sees the explanation as inspectable evidence."""
    outer_inspector = _Scripted(InspectionDecision(action="allow", threat_level="safe"))

    reply, record, binding = await _run(
        monkeypatch,
        _service(_Scripted(_decision("block"))),
        outer=_service(outer_inspector),
    )

    assert reply == "TOOL_BLOCKED: OFFICIAL_DOCUMENT_BLOCKED"
    _assert_not_evidence(
        reply, record, binding, execution_status="BLOCKED", reason="TOOL_BLOCKED"
    )
    assert outer_inspector.calls == 0


@pytest.mark.asyncio
async def test_block_in_sanitize_mode_with_a_replacement_is_sanitized(monkeypatch):
    inspector = _Scripted(_decision("block", _CLEAN))

    reply, record, binding = await _run(monkeypatch, _service(inspector, "sanitize"))

    _assert_evidence(reply, record, binding, _CLEAN)


@pytest.mark.asyncio
async def test_block_in_sanitize_mode_without_a_replacement_is_not_enforced(
    monkeypatch,
):
    inspector = _Scripted(_decision("block"))

    reply, record, binding = await _run(monkeypatch, _service(inspector, "sanitize"))

    _assert_evidence(reply, record, binding, _ORIGINAL)


@pytest.mark.asyncio
async def test_block_in_warn_mode_is_not_enforced(monkeypatch):
    """Official documents honour the configured mode: no special strictness."""
    inspector = _Scripted(_decision("block"))

    reply, record, binding = await _run(monkeypatch, _service(inspector, "warn"))

    _assert_evidence(reply, record, binding, _ORIGINAL)


@pytest.mark.asyncio
async def test_backend_failure_fail_closed_is_recorded_failed_without_secrets(
    monkeypatch,
):
    inspector = _Scripted(RuntimeError(_SECRET))

    reply, record, binding = await _run(
        monkeypatch, _service(inspector, policy="fail_closed")
    )

    assert reply == "TOOL_ERROR: INSPECTION_FAILED"
    _assert_not_evidence(
        reply, record, binding, execution_status="FAILED", reason="INSPECTION_FAILED"
    )
    assert _SECRET not in reply
    assert _SECRET not in record.content
    assert _SECRET not in repr(record)


@pytest.mark.asyncio
async def test_backend_failure_fail_open_allows_the_document(monkeypatch):
    inspector = _Scripted(RuntimeError(_SECRET))

    reply, record, binding = await _run(
        monkeypatch, _service(inspector, policy="fail_open")
    )

    _assert_evidence(reply, record, binding, _ORIGINAL)
    assert _SECRET not in reply


@pytest.mark.asyncio
async def test_inspection_timeout_is_recorded_failed(monkeypatch):
    inspector = _Stalled()
    try:
        reply, record, binding = await _run(
            monkeypatch, _ShortBound(inspector, mode="block", fail_policy="fail_open")
        )
    finally:
        inspector.release.set()
        await asyncio.sleep(0)

    assert reply == "TOOL_ERROR: INSPECTION_TIMEOUT"
    _assert_not_evidence(
        reply, record, binding, execution_status="FAILED", reason="INSPECTION_TIMEOUT"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "decision_factory",
    [
        pytest.param(lambda: _decision("sanitize", ""), id="sanitize-empty"),
        pytest.param(lambda: _decision("block", ""), id="block-empty"),
    ],
)
async def test_empty_approved_content_is_not_evidence(
    monkeypatch, decision_factory: Callable[[], InspectionDecision]
):
    inspector = _Scripted(decision_factory())

    reply, record, binding = await _run(monkeypatch, _service(inspector, "sanitize"))

    assert reply == "STATUS: INSUFFICIENT_DATA\nREASON: DOCUMENT_TEXT_UNAVAILABLE"
    _assert_not_evidence(
        reply,
        record,
        binding,
        execution_status="SUCCEEDED",
        reason="DOCUMENT_TEXT_UNAVAILABLE",
    )
