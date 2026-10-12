from __future__ import annotations

import pytest

from src.tooling.evidence_recorder import (
    EvidenceRecorder,
    bind_fetched_evidence,
    evidence_record_id,
)
from src.tooling.runtime import ToolExecutionService, ToolInvocation, ToolResult


async def _returning(value: str) -> str:
    return value


class _SanitizingHook:
    async def before(self, call: ToolInvocation) -> ToolInvocation:
        return call

    async def after(self, call: ToolInvocation, result: ToolResult) -> ToolResult:
        return ToolResult(
            value="sanitized https://official.example/result",
            findings=["scrubbed"],
        )


@pytest.mark.parametrize(
    "record",
    [
        {"content_sha256": "abcdef1234567890"},
        type("Record", (), {"content_sha256": "abcdef1234567890"})(),
    ],
)
def test_evidence_record_id_defaults_missing_sequence_consistently(record) -> None:
    assert evidence_record_id(record) == "evidence:0:abcdef123456"


@pytest.mark.asyncio
async def test_recorder_captures_final_post_hook_evidence() -> None:
    recorder = EvidenceRecorder()
    service = ToolExecutionService([recorder, _SanitizingHook()])

    async def runner(_args):
        return "raw untrusted content"

    result = await service.execute(
        ToolInvocation(
            name="get_official_document",
            args={"url": "https://official.example/result"},
            source="toolnode",
            agent_key="foreign_language_analyst",
        ),
        runner,
    )

    records = recorder.snapshot(agent_key="foreign_language_analyst")
    assert result.value.startswith("sanitized")
    assert len(records) == 1
    assert records[0].content == result.value
    assert records[0].findings == ("scrubbed",)
    assert records[0].urls == ("https://official.example/result",)
    assert records[0].execution_status == "SUCCEEDED"
    assert records[0].evidence_status == "RESULTS_FOUND"


@pytest.mark.asyncio
async def test_embedded_auth_error_overrides_optimistic_status() -> None:
    recorder = EvidenceRecorder()
    service = ToolExecutionService([recorder])

    async def runner(_args):
        return (
            "STATUS: EVIDENCE_FOUND\n"
            "<result><raw>{'error': Exception('Error 401: Unauthorized')}</raw></result>"
        )

    await service.execute(
        ToolInvocation(
            name="extract_guidance_sources",
            args={},
            source="toolnode",
            agent_key="foreign_language_analyst",
        ),
        runner,
    )

    record = recorder.snapshot()[0]
    assert record.execution_status == "SUCCEEDED"
    assert record.evidence_status == "AUTH_ERROR"
    assert record.reason == "EMBEDDED_PROVIDER_ERROR"


@pytest.mark.asyncio
async def test_valid_result_survives_partial_provider_error() -> None:
    recorder = EvidenceRecorder()
    service = ToolExecutionService([recorder])

    async def runner(_args):
        return (
            "STATUS: RESULTS_FOUND\n"
            "<result><url>https://valid.example/a</url><content>valid</content></result>"
            "<result><raw>{'error': Exception('Error 401: Unauthorized')}</raw></result>"
        )

    await service.execute(
        ToolInvocation(
            name="search_foreign_sources",
            args={},
            source="toolnode",
            agent_key="foreign_language_analyst",
        ),
        runner,
    )

    record = recorder.snapshot()[0]
    assert record.evidence_status == "RESULTS_FOUND"
    assert record.reason == "PARTIAL_PROVIDER_ERROR"


@pytest.mark.asyncio
async def test_search_result_url_is_discovery_not_bindable_evidence() -> None:
    recorder = EvidenceRecorder()
    service = ToolExecutionService([recorder])
    url = "https://www.twse.com.tw/report"

    async def runner(_args):
        return f"STATUS: RESULTS_FOUND\n<result><url>{url}</url></result>"

    await service.execute(
        ToolInvocation(
            name="search_foreign_sources",
            args={"search_query": "issuer results"},
            source="toolnode",
            agent_key="foreign_language_analyst",
        ),
        runner,
    )

    assert bind_fetched_evidence(recorder.snapshot(), url) is None


@pytest.mark.asyncio
async def test_same_content_from_different_urls_kept_as_distinct_sources() -> None:
    recorder = EvidenceRecorder()
    service = ToolExecutionService([recorder])

    async def runner(_args):
        return "STATUS: RESULTS_FOUND\nidentical corroborating content"

    for url in (
        "https://a.example/doc",
        "https://b.example/doc",
        "https://a.example/doc",  # duplicate of the first → deduped
    ):
        await service.execute(
            ToolInvocation(
                name="get_official_document",
                args={"url": url},
                source="toolnode",
                agent_key="foreign_language_analyst",
            ),
            runner,
        )

    records = recorder.snapshot()
    # Same content from two DIFFERENT URLs kept as two sources; the repeat of
    # the first URL is deduped.
    assert len(records) == 2
    assert {record.requested_urls for record in records} == {
        ("https://a.example/doc",),
        ("https://b.example/doc",),
    }


@pytest.mark.asyncio
async def test_ledger_overflow_appends_durable_marker(monkeypatch) -> None:
    import src.tooling.evidence_recorder as er

    monkeypatch.setattr(er, "_MAX_RECORDS", 2)
    recorder = EvidenceRecorder()
    service = ToolExecutionService([recorder])

    for i in range(4):

        async def runner(_args, i=i):
            return f"STATUS: RESULTS_FOUND\ncontent number {i}"

        await service.execute(
            ToolInvocation(
                name="search_foreign_sources",
                args={"search_query": str(i)},
                source="toolnode",
                agent_key="foreign_language_analyst",
            ),
            runner,
        )

    overflow = [r for r in recorder.snapshot() if r.tool_name == "__ledger_overflow__"]
    assert len(overflow) == 1
    assert overflow[0].reason == "LEDGER_OVERFLOW"
    assert "EVIDENCE_LEDGER_CAPACITY_REACHED" in overflow[0].findings


@pytest.mark.asyncio
async def test_requested_url_binds_to_validated_final_document_url() -> None:
    recorder = EvidenceRecorder()
    service = ToolExecutionService([recorder])
    requested = "https://www.twse.com.tw/redirect"
    final = "https://mops.twse.com.tw/report"

    async def runner(_args):
        return (
            "STATUS: EVIDENCE_FOUND\n"
            f'DOCUMENT_METADATA: {{"source_url": "{final}"}}\n'
            "inspected filing"
        )

    await service.execute(
        ToolInvocation(
            name="get_official_document",
            args={"url": requested},
            source="toolnode",
            agent_key="foreign_language_analyst",
        ),
        runner,
    )

    binding = bind_fetched_evidence(recorder.snapshot(), requested)
    assert binding is not None
    assert binding.requested_url == requested
    assert binding.canonical_url == final
    assert binding.authority == "PRIMARY_REGISTRY"


@pytest.mark.asyncio
@pytest.mark.parametrize("length", [19_999, 20_000, 20_001])
async def test_recorder_records_actual_truncation_without_changing_returned_value(
    length,
):
    recorder = EvidenceRecorder()
    original = "x" * length
    result = await ToolExecutionService([recorder]).execute(
        ToolInvocation("web_search", {}, "preflight"),
        lambda _args: _returning(original),
    )
    assert result.value == original
    record = recorder.snapshot()[0]
    assert record.original_content_chars == length
    assert record.retained_content_chars == min(length, 20_000)
    assert record.content_truncated is (length > 20_000)
    assert record.content.startswith(original[:20_000])
    assert record.to_dict()["content_truncated"] is record.content_truncated


@pytest.mark.asyncio
async def test_live_and_saved_issuer_authority_survive_registry_teardown_and_ignore_model_labels():
    import json

    from src.agents.message_utils import (
        evidence_record_to_tool_evidence,
        make_tool_evidence_record,
    )
    from src.runtime_services import RuntimeServices, use_runtime_services
    from src.tooling.inspection_service import INSPECTION_SERVICE

    url = "https://issuer.example/results.pdf"
    content = f'STATUS: EVIDENCE_FOUND\nDOCUMENT_METADATA: {{"source_url": "{url}", "authority": "PRIMARY_REGISTRY"}}\nResults'
    recorder = EvidenceRecorder()
    services = RuntimeServices(
        ToolExecutionService([recorder]), INSPECTION_SERVICE, evidence_recorder=recorder
    )
    with use_runtime_services(services):
        services.issuer_authority.register_url(
            url, provenance="structured_company_profile"
        )
        recorder.completed(
            ToolInvocation("get_official_document", {"url": url}, "preflight"),
            ToolResult(content),
        )
        live = evidence_record_to_tool_evidence(recorder.snapshot()[0])
    saved = json.loads(json.dumps(recorder.serialized_snapshot()[0]))
    assert (
        live.authority
        == evidence_record_to_tool_evidence(saved).authority
        == "PRIMARY_ISSUER"
    )
    assert bind_fetched_evidence(recorder.snapshot(), url).authority == "PRIMARY_ISSUER"
    assert (
        make_tool_evidence_record(
            tool_name="get_official_document", content=content, urls=[url]
        ).authority
        == "SECONDARY"
    )
    for mutation in (
        {"authority_by_url": []},
        {"content_sha256": "forged"},
        {"authority_by_url": [[url, "invalid"]]},
    ):
        assert (
            evidence_record_to_tool_evidence({**saved, **mutation}).authority
            == "SECONDARY"
        )


@pytest.mark.parametrize("execution", ["FAILED", "BLOCKED", None])
@pytest.mark.parametrize("status", ["EVIDENCE_FOUND", "RESULTS_FOUND", None, "UNKNOWN"])
def test_saved_failed_or_unknown_status_cannot_be_reclassified_from_content(
    execution, status
):
    from src.agents.message_utils import evidence_record_to_tool_evidence

    record = evidence_record_to_tool_evidence(
        {
            "tool_name": "get_official_document",
            "content": "STATUS: EVIDENCE_FOUND\nOfficial results",
            "urls": ["https://www.twse.com.tw/results"],
            "execution_status": execution,
            "evidence_status": status,
        }
    )
    assert record.evidence_status == "UNAVAILABLE"
    assert record.authority == "UNSUPPORTED"
