"""Capital absence instructions agree with code-owned coverage normalization."""

import json
from pathlib import Path

import pytest

from src.agents.capital_structure import normalize_legal_output


def _capital_schema() -> dict[str, str]:
    prompt = json.loads(Path("prompts/legal_counsel.json").read_text())
    message = prompt["system_message"]
    start = message.index('{\n  "pfic_status"')
    payload, _ = json.JSONDecoder().raw_decode(message[start:])
    assert "coverage_status=UNRESOLVED and exposure_type=UNKNOWN" in message
    assert "complete absence" in message
    return payload["capital_structure"]


@pytest.mark.parametrize(
    "coverage,exposure",
    [
        ("FOUND", "NONE"),
        ("NOT_FOUND", "NONE"),
        ("SEARCH_FAILED", "UNKNOWN"),
        ("UNRESOLVED", "UNKNOWN"),
    ],
)
def test_capital_absence_normalizes_to_advertised_unresolved_output(coverage, exposure):
    schema = _capital_schema()
    assert "NOT_FOUND" not in schema["coverage_status"].split("|")
    assert "NONE" not in schema["exposure_type"].split("|")
    normalized, _ = normalize_legal_output(
        json.dumps(
            {
                "capital_structure": {
                    "coverage_status": coverage,
                    "exposure_type": exposure,
                }
            }
        ),
        "#### structures_search\nEXECUTION_STATUS: SUCCEEDED\nEVIDENCE_STATUS: NO_RESULTS",
    )
    assert normalized is not None
    capital = json.loads(normalized)["capital_structure"]
    assert capital["classification"] == "UNRESOLVED"
    assert capital["coverage_status"] in schema["coverage_status"].split("|")
    assert capital["exposure_type"] in schema["exposure_type"].split("|")
