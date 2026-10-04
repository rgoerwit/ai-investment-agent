"""A sourced holder name can survive when the source omits its percentage."""

from src.agents.foreign_language_evidence import normalize_foreign_language_evidence
from src.agents.message_utils import make_tool_evidence_record


def _report(holder: str) -> str:
    return f"""**OWNERSHIP STRUCTURE**
- Largest Shareholder: {holder}
- Control Status: CONTROLLED
- Control Basis: MAJORITY_VOTING_RIGHTS
- Ownership Evidence Status: CITED
- Ownership Source URL: https://issuer.example/holders
"""


def test_sourced_holder_without_percentage_keeps_name_but_not_control() -> None:
    evidence = make_tool_evidence_record(
        tool_name="get_official_filings",
        content="Acme Holdings is the largest shareholder. https://issuer.example/holders",
        urls={"https://issuer.example/holders"},
    )

    normalized = normalize_foreign_language_evidence(
        _report("Acme Holdings"), [], ticker="TEST.T", additional_records=[evidence]
    )

    assert "Largest Shareholder: Acme Holdings" in normalized
    assert "Ownership Evidence Status: VERIFIED_URL" in normalized
    assert "Control Status: UNKNOWN" in normalized
    assert "Control Basis: UNKNOWN" in normalized


def test_name_without_largest_holder_evidence_is_rejected() -> None:
    evidence = make_tool_evidence_record(
        tool_name="get_official_filings",
        content="Acme Holdings is a shareholder. https://issuer.example/holders",
        urls={"https://issuer.example/holders"},
    )

    normalized = normalize_foreign_language_evidence(
        _report("Acme Holdings"), [], ticker="TEST.T", additional_records=[evidence]
    )

    assert "Largest Shareholder: UNKNOWN" in normalized
    assert "Ownership Evidence Status: REJECTED" in normalized
