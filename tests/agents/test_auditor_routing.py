from src.agents.consultant_nodes import (
    _auditor_should_escalate,
    _budget_exhausted_report,
)


def test_complete_complex_audit_escalates_to_sol() -> None:
    report = """FORENSIC_DATA_BLOCK:
STATUS: CONCERN
ANOMALIES: Related-party acquisition accounting requires review.
VERDICT: Review.
"""
    assert _auditor_should_escalate(report)


def test_incomplete_audit_never_spends_sol_budget() -> None:
    report = """FORENSIC_DATA_BLOCK:
STATUS: INSUFFICIENT_DATA
REASON: STATEMENT_TRIAD_INCOMPLETE
ANOMALIES: Related-party acquisition unverified.
VERDICT: Unable.
"""
    assert not _auditor_should_escalate(report)


def test_complete_ordinary_audit_stays_on_terra() -> None:
    report = "FORENSIC_DATA_BLOCK:\nSTATUS: CLEAN\nVERDICT: No anomalies.\n"
    assert not _auditor_should_escalate(report)


def test_budget_fallback_uses_exact_reason_code() -> None:
    report = _budget_exhausted_report("LLM_CALL_BUDGET_EXHAUSTED", "AGS.SI")
    assert "STATUS: INSUFFICIENT_DATA" in report
    assert "REASON: LLM_CALL_BUDGET_EXHAUSTED" in report
