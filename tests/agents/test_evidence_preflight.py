from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.evidence_preflight import run_preflight_calls


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
            source="toolnode",
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
