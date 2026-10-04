"""Regression tests for Value Trap acquisition-context provenance."""

import re

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
    value = extract_block_field(report, "VALUE_TRAP_BLOCK", name)
    return None if value in {"N/A", "NONE"} else value


def _prompt_report(*, status: str, url: str, context: str) -> str:
    return f"""### --- START VALUE_TRAP_BLOCK ---
SCORE: 55
VERDICT: CAUTIOUS
CAPITAL_ALLOCATION:
  RATING: MIXED
  M&A_CONTEXT_EVIDENCE: {status}
  M&A_CONTEXT_SOURCE_URL: {url}
  M&A_CONTEXT: {context}
  BUYBACK_CONTEXT: UNKNOWN
CATALYSTS:
  INDEX_CANDIDATE: NONE
### --- END VALUE_TRAP_BLOCK ---"""


def _assert_one_m_and_a_field_set(report: str) -> None:
    for field in ("M&A_CONTEXT_EVIDENCE", "M&A_CONTEXT_SOURCE_URL", "M&A_CONTEXT"):
        assert len(re.findall(rf"(?m)^{field}:", report)) == 1
        assert len(re.findall(rf"(?m)^[ \t]+{field}:", report)) == 0


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


def test_prompt_indented_citation_survives_without_duplicate_fields():
    url = "https://example.com/filing"
    report = _prompt_report(status="CITED", url=url, context="Acquired a distributor.")
    messages = [ToolMessage(content=f"Official filing: {url}", tool_call_id="call-1")]

    normalized = normalize_value_trap_m_and_a_evidence(report, messages, ticker="TEST")

    _assert_one_m_and_a_field_set(normalized)
    assert _field(normalized, "M&A_CONTEXT_EVIDENCE") == "CITED"
    assert _field(normalized, "M&A_CONTEXT_SOURCE_URL") == url
    assert extract_value_trap_score(normalized)["m_and_a_context_evidence"] == "CITED"
    assert (
        normalize_value_trap_m_and_a_evidence(normalized, messages, ticker="TEST")
        == normalized
    )


def test_normalized_citation_is_visible_to_strict_downstream_reader():
    url = "https://example.com/filing"
    report = _prompt_report(status="CITED", url=url, context="Acquired a distributor.")
    messages = [ToolMessage(content=f"Official filing: {url}", tool_call_id="call-1")]

    normalized = normalize_value_trap_m_and_a_evidence(report, messages, ticker="TEST")
    assert (
        extract_block_field(normalized, "VALUE_TRAP_BLOCK", "M&A_CONTEXT_EVIDENCE")
        == "CITED"
    )


def test_prompt_indented_unseen_citation_is_downgraded_once():
    report = _prompt_report(
        status="CITED",
        url="https://unseen.example/deal",
        context="Acquired a distributor.",
    )

    normalized = normalize_value_trap_m_and_a_evidence(report, [], ticker="TEST")

    _assert_one_m_and_a_field_set(normalized)
    assert _field(normalized, "M&A_CONTEXT_EVIDENCE") == "UNKNOWN"
    assert "unseen.example" not in normalized
    assert "Acquired a distributor" not in normalized


def test_duplicate_m_and_a_fields_fail_closed():
    url = "https://example.com/filing"
    report = _prompt_report(status="CITED", url=url, context="Acquired a distributor.")
    report = report.replace(
        "### --- END VALUE_TRAP_BLOCK ---",
        "M&A_CONTEXT_EVIDENCE: UNKNOWN\nM&A_CONTEXT_SOURCE_URL: N/A\n"
        "M&A_CONTEXT: UNKNOWN\n### --- END VALUE_TRAP_BLOCK ---",
    )
    messages = [ToolMessage(content=f"Official filing: {url}", tool_call_id="call-1")]

    normalized = normalize_value_trap_m_and_a_evidence(report, messages, ticker="TEST")

    _assert_one_m_and_a_field_set(normalized)
    assert _field(normalized, "M&A_CONTEXT_EVIDENCE") == "UNKNOWN"
    assert _field(normalized, "M&A_CONTEXT_SOURCE_URL") is None
    assert "Acquired a distributor" not in normalized


def test_single_duplicate_identical_field_also_fails_closed():
    url = "https://example.com/filing"
    report = _report(status="CITED", url=url, context="Acquired a distributor.")
    report = report.replace(
        "### --- END VALUE_TRAP_BLOCK ---",
        "M&A_CONTEXT_EVIDENCE: CITED\n### --- END VALUE_TRAP_BLOCK ---",
    )
    messages = [ToolMessage(content=url, tool_call_id="call-1")]

    normalized = normalize_value_trap_m_and_a_evidence(report, messages, ticker="TEST")

    _assert_one_m_and_a_field_set(normalized)
    assert _field(normalized, "M&A_CONTEXT_EVIDENCE") == "UNKNOWN"


def test_missing_m_and_a_fields_receive_one_conservative_set():
    report = _prompt_report(status="CITED", url="https://example.com", context="Deal")
    report = "\n".join(
        line
        for line in report.splitlines()
        if not line.lstrip().startswith("M&A_CONTEXT")
    )

    normalized = normalize_value_trap_m_and_a_evidence(report, [], ticker="TEST")

    _assert_one_m_and_a_field_set(normalized)
    assert _field(normalized, "M&A_CONTEXT_EVIDENCE") == "UNKNOWN"
    assert _field(normalized, "M&A_CONTEXT") == "UNKNOWN"


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


def test_source_checked_m_and_a_does_not_get_false_downstream_constraint():
    url = "https://example.com/filing"
    messages = [ToolMessage(content=url, tool_call_id="call-1")]
    report = normalize_value_trap_m_and_a_evidence(
        _prompt_report(status="CITED", url=url, context="Acquired a distributor."),
        messages,
        ticker="TEST",
    )
    state = {
        "value_trap_report": report,
        "artifact_statuses": {
            "value_trap_report": {"complete": True, "ok": True, "content": report}
        },
    }

    constraints = downstream_evidence_constraints(state)

    assert "Value Trap M&A context is not source-verified" not in constraints


def test_supplemental_m_and_a_reader_ignores_prose_outside_block():
    report = (
        "M&A_CONTEXT_EVIDENCE: CITED\nM&A_CONTEXT_SOURCE_URL: https://example.com/false\n"
        "M&A_CONTEXT: Acquired a distributor.\n"
        + _report(status="UNKNOWN", url="N/A", context="UNKNOWN")
    )

    metrics = extract_value_trap_score(report)

    assert metrics["m_and_a_context_evidence"] == "UNKNOWN"
    assert metrics["m_and_a_context_source_url"] is None
    assert metrics["m_and_a_context"] == "UNKNOWN"
