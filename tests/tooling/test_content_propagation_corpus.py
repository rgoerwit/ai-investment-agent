from __future__ import annotations

import pytest

from src.tooling.heuristic_inspector import HeuristicInspector
from src.tooling.inspection_hook import ContentInspectionHook
from src.tooling.inspection_service import InspectionService
from src.tooling.inspector import InspectionEnvelope, SourceKind
from src.tooling.runtime import ToolExecutionService, ToolInvocation
from tests.helpers.injection_corpus import load_corpus


def _blocking_service() -> InspectionService:
    return InspectionService(
        inspector=HeuristicInspector(),
        mode="block",
        fail_policy="fail_closed",
    )


def _must_block_case(source_kind: str) -> dict:
    return load_corpus(source_kind=source_kind, expectation="must_block")[0]


@pytest.mark.security
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source_kind",
    [
        "web_search",
        "social_feed",
        "financial_api",
        "cached_context",
        "mcp_tool_output",
        "tool_output",
    ],
)
async def test_inspection_service_blocks_source_kind_corpus_payloads(source_kind):
    case = _must_block_case(source_kind)
    service = _blocking_service()

    decision, approved = await service.evaluate(
        InspectionEnvelope(
            content_text=case["payload"],
            raw_content=case["payload"],
            source_kind=SourceKind(case["source_kind"]),
            source_name="corpus",
            tool_name="corpus",
        )
    )

    assert decision.action == "block"
    assert str(approved).startswith("TOOL_BLOCKED:")
    assert case["payload"] not in str(approved)


@pytest.mark.security
@pytest.mark.asyncio
async def test_content_inspection_hook_preserves_benign_string_output():
    service = _blocking_service()
    tool_service = ToolExecutionService([ContentInspectionHook(service)])
    call = ToolInvocation(name="safe_tool", args={}, source="toolnode")

    result = await tool_service.execute(
        call,
        lambda _args: _async_value("Revenue rose 3% and margins were stable."),
    )

    assert result.blocked is False
    assert result.value == "Revenue rose 3% and margins were stable."


@pytest.mark.security
@pytest.mark.asyncio
async def test_content_inspection_hook_blocks_tool_output_before_agent_context():
    case = _must_block_case("tool_output")
    service = _blocking_service()
    tool_service = ToolExecutionService([ContentInspectionHook(service)])
    call = ToolInvocation(name="hostile_tool", args={}, source="toolnode")

    result = await tool_service.execute(
        call, lambda _args: _async_value(case["payload"])
    )

    assert result.blocked is True
    assert str(result.value).startswith("TOOL_BLOCKED:")
    assert case["payload"] not in str(result.value)


@pytest.mark.security
@pytest.mark.asyncio
async def test_content_inspection_hook_blocks_mcp_tool_output():
    case = _must_block_case("mcp_tool_output")
    service = _blocking_service()
    tool_service = ToolExecutionService([ContentInspectionHook(service)])
    call = ToolInvocation(name="mcp__quotes__lookup", args={}, source="toolnode")

    result = await tool_service.execute(
        call, lambda _args: _async_value(case["payload"])
    )

    assert result.blocked is True
    assert str(result.value).startswith("TOOL_BLOCKED:")
    assert case["payload"] not in str(result.value)


@pytest.mark.security
@pytest.mark.asyncio
async def test_unknown_object_output_is_degraded_without_calling_str():
    class ExplosiveString:
        def __str__(self):
            raise AssertionError("__str__ should not run during inspection")

    service = _blocking_service()
    tool_service = ToolExecutionService([ContentInspectionHook(service)])
    call = ToolInvocation(name="object_tool", args={}, source="toolnode")

    result = await tool_service.execute(
        call,
        lambda _args: _async_value(ExplosiveString()),
    )

    assert result.blocked is False
    assert result.value == "<ExplosiveString>"
    assert result.findings == ["non-string tool output degraded for inspection safety"]


@pytest.mark.security
@pytest.mark.asyncio
async def test_nested_unknown_object_output_is_degraded_to_safe_json_text():
    class HostileObject:
        def __str__(self):
            raise AssertionError("__str__ should not run during inspection")

    service = _blocking_service()
    tool_service = ToolExecutionService([ContentInspectionHook(service)])
    call = ToolInvocation(name="nested_object_tool", args={}, source="toolnode")

    result = await tool_service.execute(
        call,
        lambda _args: _async_value({"payload": HostileObject()}),
    )

    assert result.blocked is False
    assert result.value == '{"payload": "<HostileObject>"}'
    assert result.findings == ["non-string tool output degraded for inspection safety"]


async def _async_value(value):
    return value
