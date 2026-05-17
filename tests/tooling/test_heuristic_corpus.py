from __future__ import annotations

import pytest

from src.tooling.heuristic_inspector import HeuristicInspector
from src.tooling.inspector import InspectionEnvelope, SourceKind
from tests.helpers.injection_corpus import load_corpus

pytestmark = pytest.mark.security

_LEAK_MARKERS = (
    "ignore previous",
    "disregard prior",
    "reveal the system prompt",
    "print the api key",
    "include the credentials",
    "expose token",
)


def _envelope(case: dict) -> InspectionEnvelope:
    return InspectionEnvelope(
        content_text=case["payload"],
        source_kind=SourceKind(case["source_kind"]),
        source_name="corpus",
        tool_name="corpus",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    load_corpus(expectation="must_block"),
    ids=lambda case: case["id"],
)
async def test_heuristic_blocks_must_block_cases(case):
    decision = await HeuristicInspector().inspect(_envelope(case))

    assert decision.action == "block", (
        f"{case['id']} ({case['source_kind']}/{case['category']}): "
        f"action={decision.action} threat_level={decision.threat_level} "
        f"findings={decision.findings}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    load_corpus(expectation="block_or_sanitize"),
    ids=lambda case: case["id"],
)
async def test_heuristic_sanitize_cases_do_not_retain_instruction_markers(case):
    decision = await HeuristicInspector().inspect(_envelope(case))

    assert decision.action in {"block", "sanitize"}
    if decision.action == "sanitize":
        sanitized = (decision.sanitized_content or "").lower()
        assert sanitized
        assert case["payload"].lower() != sanitized
        for marker in _LEAK_MARKERS:
            assert marker not in sanitized
