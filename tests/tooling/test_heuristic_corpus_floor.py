from __future__ import annotations

import pytest

from src.tooling.heuristic_inspector import HeuristicInspector
from src.tooling.inspector import InspectionEnvelope, SourceKind
from tests.helpers.injection_corpus import load_corpus, load_detection_floors

pytestmark = pytest.mark.security


def _envelope(case: dict) -> InspectionEnvelope:
    return InspectionEnvelope(
        content_text=case["payload"],
        source_kind=SourceKind(case["source_kind"]),
        source_name="corpus",
        tool_name="corpus",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bucket",
    load_detection_floors(),
    ids=lambda bucket: f"{bucket['source_kind']}:{bucket['category']}",
)
async def test_heuristic_detection_floor_by_bucket(bucket):
    cases = load_corpus(
        source_kind=bucket["source_kind"],
        category=bucket["category"],
    )
    if not cases:
        pytest.fail(f"no corpus cases for {bucket['source_kind']}/{bucket['category']}")

    inspector = HeuristicInspector()
    flagged = 0

    for case in cases:
        decision = await inspector.inspect(_envelope(case))
        if decision.action in {"block", "sanitize", "degrade"}:
            flagged += 1

    rate = flagged / len(cases)
    assert rate >= bucket["floor"], (
        f"{bucket['source_kind']}/{bucket['category']}: "
        f"{rate:.0%} < floor {bucket['floor']:.0%}"
    )
