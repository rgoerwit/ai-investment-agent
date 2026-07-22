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


def test_loop_control_telemetry_records_forced_synthesis_and_failure() -> None:
    ledger = AuditorBudgetLedger(_policy())
    ledger.cap_evidence("filing")
    ledger.record_tool_round(["get_official_document"])
    ledger.record_tool_failure("get_official_document")
    ledger.record_forced_synthesis()
    ledger.record_repair_input("malformed draft")

    telemetry = ledger.telemetry()

    assert telemetry["tool_rounds_used"] == 1
    assert telemetry["forced_synthesis_used"] is True
    assert telemetry["stop_reason"] == "TOOL_ROUND_LIMIT"
    assert telemetry["final_tool_names"] == ["get_official_document"]
    assert telemetry["failed_tools"] == ["get_official_document"]
    assert telemetry["synthesis_evidence_chars"] == len("filing")
    assert telemetry["repair_input_chars"] == len("malformed draft")


def test_typed_insufficient_tool_result_is_recorded_separately() -> None:
    ledger = AuditorBudgetLedger(_policy())

    ledger.record_tool_result(
        "get_official_document",
        "STATUS: INSUFFICIENT_DATA\nREASON: UNAPPROVED_DOCUMENT_HOST",
    )

    assert ledger.insufficient_tools == ["get_official_document"]
    assert ledger.failed_tools == []


def test_blocked_and_failed_tool_results_are_distinct() -> None:
    ledger = AuditorBudgetLedger(_policy())

    ledger.record_tool_result("get_official_document", "TOOL_BLOCKED: policy")
    ledger.record_tool_result("get_news", "TOOL_ERROR: TimeoutError")

    assert ledger.blocked_tools == ["get_official_document"]
    assert ledger.failed_tools == ["get_news"]
    assert ledger.insufficient_tools == []


def test_successful_tool_result_is_not_recorded_as_failure() -> None:
    ledger = AuditorBudgetLedger(_policy())

    ledger.record_tool_result("get_official_document", "STATUS: FOUND\nRevenue: 10")

    assert ledger.failed_tools == []
