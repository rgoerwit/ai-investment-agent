from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.evidence_preflight import run_preflight_calls
from src.tooling.runtime import ToolCallFailed


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (
            "STATUS: UNAVAILABLE\nREASON: ADAPTER_UNAVAILABLE\nNo adapter.",
            ("SUCCEEDED", "UNAVAILABLE", "INSUFFICIENT_DATA"),
        ),
        (
            "STATUS: NO_RESULTS\nREASON: NO_RESULTS\nNo results.",
            ("SUCCEEDED", "NO_RESULTS", "INSUFFICIENT_DATA"),
        ),
        (
            "STATUS: RESULTS_FOUND\n<search_results></search_results>",
            ("SUCCEEDED", "RESULTS_FOUND", "COMPLETED"),
        ),
        (
            "STATUS: EVIDENCE_FOUND\n"
            "<result><raw>{'error': Exception('Error 401: Unauthorized')}</raw></result>",
            ("SUCCEEDED", "AUTH_ERROR", "INSUFFICIENT_DATA"),
        ),
    ],
)
async def test_preflight_separates_execution_from_evidence(value, expected):
    tool = SimpleNamespace(name="search_foreign_sources", ainvoke=AsyncMock())
    service = SimpleNamespace(
        execute=AsyncMock(return_value=SimpleNamespace(value=value, blocked=False))
    )

    with patch(
        "src.runtime_services.get_current_tool_service",
        return_value=service,
    ):
        outcomes, _ = await run_preflight_calls(
            [("search", tool, {})],
            agent_key="test",
            source="preflight",
            ticker="TEST",
            failure_event="test_failure",
            logger=MagicMock(),
        )

    outcome = outcomes[0]
    assert (
        outcome.execution_status,
        outcome.evidence_status,
        outcome.legacy_status,
    ) == expected
    assert f"EXECUTION_STATUS: {expected[0]}" in outcome.render()
    assert f"EVIDENCE_STATUS: {expected[1]}" in outcome.render()
    invocation = service.execute.await_args.args[0]
    assert invocation.source == "preflight"
    assert invocation.agent_key == "test"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (ToolCallFailed("INSPECTION_TIMEOUT"), "INSPECTION_TIMEOUT"),
        (ValueError("prose https://x?token=1"), "VALUEERROR"),
    ],
)
async def test_a_failed_call_carries_its_typed_reason_and_counts_as_a_failure(
    error, reason
):
    tool = SimpleNamespace(name="get_official_document", ainvoke=AsyncMock())
    service = SimpleNamespace(execute=AsyncMock(side_effect=error))
    budget = MagicMock()
    budget.authorize_tool.return_value = None

    with patch(
        "src.runtime_services.get_current_tool_service",
        return_value=service,
    ):
        outcomes, _ = await run_preflight_calls(
            [("document", tool, {"url": "https://www.sec.gov/x"})],
            agent_key="test",
            source="preflight",
            ticker="TEST",
            failure_event="test_failure",
            logger=MagicMock(),
            budget=budget,
        )

    (outcome,) = outcomes
    assert (outcome.execution_status, outcome.reason) == ("FAILED", reason)
    assert "token" not in outcome.render()
    budget.record_tool_failure.assert_called_once_with("get_official_document")
