from src.forensic_budget import AuditorBudgetLedger, AuditorBudgetPolicy


def _policy(**overrides) -> AuditorBudgetPolicy:
    values = {
        "search_calls": 1,
        "document_calls": 1,
        "filing_calls": 1,
        "metrics_calls": 1,
        "news_calls": 1,
        "calculation_calls": 1,
        "max_document_bytes": 1000,
        "max_document_pages": 10,
        "max_selected_pages": 2,
        "max_evidence_chars": 10,
        "max_tool_iterations": 1,
        "max_llm_calls": 2,
    }
    values.update(overrides)
    return AuditorBudgetPolicy(**values)


def test_tool_budget_exhaustion_is_distinct_from_missing_data() -> None:
    ledger = AuditorBudgetLedger(_policy())

    assert ledger.consume_tool("search_foreign_sources") is None
    assert ledger.consume_tool("search_foreign_sources") == (
        "TOOL_CALL_BUDGET_EXHAUSTED"
    )
    assert ledger.telemetry()["outcomes"] == ["TOOL_CALL_BUDGET_EXHAUSTED"]


def test_evidence_cap_is_cumulative() -> None:
    ledger = AuditorBudgetLedger(_policy())

    assert ledger.cap_evidence("123456") == "123456"
    assert ledger.cap_evidence("abcdef") == "abcd\nREASON: EVIDENCE_CHAR_LIMIT"
    assert ledger.evidence_chars == 10
    assert ledger.evidence_truncated is True


def test_llm_budget_includes_repairs_and_escalations() -> None:
    ledger = AuditorBudgetLedger(_policy(max_llm_calls=2))

    assert ledger.consume_llm() is None
    assert ledger.consume_llm() is None
    assert ledger.consume_llm() == "LLM_CALL_BUDGET_EXHAUSTED"
