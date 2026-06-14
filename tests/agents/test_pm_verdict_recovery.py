from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.decision_nodes import _recover_pm_verdict_metadata
from src.agents.pm_verdict_metadata import PMVerdictRecovery


class _LLM:
    def with_structured_output(self, schema, strict: bool = False):
        assert schema is PMVerdictRecovery
        assert strict is True
        return self


@pytest.mark.asyncio
async def test_recover_pm_verdict_metadata_uses_structured_output() -> None:
    with patch(
        "src.agents.decision_nodes.agent_runtime.invoke_with_rate_limit_handling",
        new=AsyncMock(return_value=PMVerdictRecovery(verdict="DO_NOT_INITIATE")),
    ) as invoke:
        metadata = await _recover_pm_verdict_metadata(
            "The committee declines to initiate a position.",
            _LLM(),
        )

    assert metadata is not None
    assert metadata.verdict == "DO_NOT_INITIATE"
    assert invoke.await_count == 1


@pytest.mark.asyncio
async def test_recover_pm_verdict_metadata_skips_llm_without_structured_output() -> (
    None
):
    metadata = await _recover_pm_verdict_metadata("BUY", object())

    assert metadata is None
