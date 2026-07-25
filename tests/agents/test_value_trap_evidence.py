"""Regression tests for Value Trap acquisition-context provenance."""

from langchain_core.messages import ToolMessage

from src.agents.evidence_constraints import downstream_evidence_constraints
from src.agents.value_trap_evidence import normalize_value_trap_m_and_a_evidence
from src.data_block_utils import extract_block_field
from src.validators.supplemental_extractors import extract_value_trap_score
from tests.helpers.frozen_regressions import load_frozen_regression


def _report(*, status: str, url: str, context: str) -> str:
    return f"""### --- START VALUE_TRAP_BLOCK ---
SCORE: 55
VERDICT: CAUTIOUS
TRAP_RISK: MEDIUM
M&A_CONTEXT_EVIDENCE: {status}
M&A_CONTEXT_SOURCE_URL: {url}
M&A_CONTEXT: {context}
### --- END VALUE_TRAP_BLOCK ---"""


def _field(report: str, name: str) -> str | None:
    return extract_block_field(report, "VALUE_TRAP_BLOCK", name)


def test_cited_context_survives_when_url_is_in_agent_tool_output():
    url = "https://example.com/visco-acquisition"
    messages = [
        ToolMessage(
            content=f"Company filing: {url}",
            tool_call_id="call-1",
            additional_kwargs={"agent_key": "value_trap_detector"},
        )
    ]

    normalized = normalize_value_trap_m_and_a_evidence(
        _report(
            status="CITED",
            url=url,
            context="Acquired Mingdar in a board-approved transaction.",
        ),
        messages,
        ticker="6782.TW",
    )

    assert _field(normalized, "M&A_CONTEXT_EVIDENCE") == "CITED"
    assert _field(normalized, "M&A_CONTEXT_SOURCE_URL") == url
    assert "Mingdar" in (_field(normalized, "M&A_CONTEXT") or "")
    metrics = extract_value_trap_score(normalized)
    assert metrics["m_and_a_context_evidence"] == "CITED"
    assert metrics["m_and_a_context_source_url"] == url


def test_unseen_citation_is_downgraded_and_named_context_removed():
    regression = load_frozen_regression("6782_TW_regression.json")
    normalized = normalize_value_trap_m_and_a_evidence(
        regression["value_trap_output"],
        [],
        ticker=regression["ticker"],
    )

    assert _field(normalized, "M&A_CONTEXT_EVIDENCE") == "UNKNOWN"
    assert _field(normalized, "M&A_CONTEXT_SOURCE_URL") is None
    assert _field(normalized, "M&A_CONTEXT") == "UNKNOWN"
    assert "From-eyes" not in normalized


def test_not_found_is_preserved_without_inventing_context():
    normalized = normalize_value_trap_m_and_a_evidence(
        _report(
            status="NOT_FOUND",
            url="N/A",
            context="No acquisition record located.",
        ),
        [],
        ticker="6782.TW",
    )

    assert _field(normalized, "M&A_CONTEXT_EVIDENCE") == "NOT_FOUND"
    assert _field(normalized, "M&A_CONTEXT") == "UNKNOWN"


def test_downstream_constraint_blocks_unverified_m_and_a_narrative():
    state = {
        "value_trap_report": _report(
            status="UNKNOWN",
            url="N/A",
            context="UNKNOWN",
        )
    }

    constraints = downstream_evidence_constraints(state)

    assert "Value Trap M&A context is not source-verified" in constraints
    assert "infer acquisition-led growth" in constraints
