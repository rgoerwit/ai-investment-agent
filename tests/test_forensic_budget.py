import pytest

from src.forensic_budget import (
    AuditorBudgetLedger,
    AuditorBudgetPolicy,
    ForeignLanguageBudgetPolicy,
    ResearchBudgetLedger,
    graph_research_budget_policies,
)


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
    # No REJECTED_HOST line in this result — must not fire on unrelated
    # insufficient-data reasons (e.g. a plain DOCUMENT_SIZE_LIMIT rejection).
    assert ledger.rejected_hosts == []
    assert ledger.tool_outcome_events == {"ordinary_insufficient": 1}


def test_rejected_host_is_recorded_and_deduplicated() -> None:
    ledger = AuditorBudgetLedger(_policy())

    ledger.record_tool_result(
        "get_official_document",
        "STATUS: INSUFFICIENT_DATA\nREASON: UNAPPROVED_DOCUMENT_HOST\n"
        "REJECTED_HOST: example.com\nAPPROVED_HOSTS: a.com, b.com",
    )
    ledger.record_tool_result(
        "get_official_document",
        "STATUS: INSUFFICIENT_DATA\nREASON: UNAPPROVED_DOCUMENT_HOST\n"
        "REJECTED_HOST: example.com\nAPPROVED_HOSTS: a.com, b.com",
    )

    assert ledger.rejected_hosts == ["example.com"]
    assert ledger.telemetry()["rejected_hosts"] == ["example.com"]


def test_blocked_and_failed_tool_results_are_distinct() -> None:
    ledger = AuditorBudgetLedger(_policy())

    ledger.record_tool_result("get_official_document", "TOOL_BLOCKED: policy")
    ledger.record_tool_result("get_news", "TOOL_ERROR: TimeoutError")

    assert ledger.blocked_tools == ["get_official_document"]
    assert ledger.failed_tools == ["get_news"]
    assert ledger.insufficient_tools == []
    assert ledger.tool_outcome_events == {"execution_error": 1}


def test_block_name_dedup_does_not_erase_block_event_count() -> None:
    ledger = ResearchBudgetLedger(_foreign_policy())

    ledger.block_tool("search_foreign_sources", "TOOL_ROUND_LIMIT")
    ledger.block_tool("search_foreign_sources", "TOOL_ROUND_LIMIT")

    assert ledger.blocked_tools == ["search_foreign_sources"]
    assert ledger.blocked_reasons == {"TOOL_ROUND_LIMIT": 2}


def test_acquisition_failure_is_not_ordinary_insufficient() -> None:
    ledger = AuditorBudgetLedger(_policy())

    ledger.record_tool_result(
        "get_official_document",
        "STATUS: INSUFFICIENT_DATA\nREASON: DOCUMENT_DNS_FAILED",
    )

    assert ledger.insufficient_tools == ["get_official_document"]
    assert ledger.tool_outcome_events == {"evidence_acquisition_failure": 1}
    assert ledger.tool_outcome_events_by_name == {
        "evidence_acquisition_failure": {"get_official_document": 1}
    }


def test_successful_tool_result_is_not_recorded_as_failure() -> None:
    ledger = AuditorBudgetLedger(_policy())

    ledger.record_tool_result("get_official_document", "STATUS: FOUND\nRevenue: 10")

    assert ledger.failed_tools == []


def _foreign_policy(**overrides) -> ForeignLanguageBudgetPolicy:
    values = {
        "search_calls": 3,
        "document_calls": 2,
        "filing_calls": 1,
        "guidance_calls": 1,
        "max_tool_iterations": 2,
        "max_llm_calls": 4,
        "max_tool_calls_per_turn": 3,
        "purpose_call_limit": 2,
    }
    values.update(overrides)
    return ForeignLanguageBudgetPolicy(**values)


def test_shared_research_ledger_blocks_equivalent_calls() -> None:
    ledger = ResearchBudgetLedger(_foreign_policy())
    args = {
        "ticker": "TEST",
        "search_query": "issuer latest results",
        "purpose": "latest_results",
    }

    assert ledger.authorize_tool("search_foreign_sources", args) is None
    assert ledger.authorize_tool("search_foreign_sources", args) == (
        "DUPLICATE_TOOL_CALL"
    )
    assert ledger.telemetry()["blocked_reasons"] == {"DUPLICATE_TOOL_CALL": 1}


def test_shared_research_ledger_caps_refinements_by_purpose() -> None:
    ledger = ResearchBudgetLedger(_foreign_policy(purpose_call_limit=2))
    for query in ("latest results", "latest official income statement"):
        assert (
            ledger.authorize_tool(
                "search_foreign_sources",
                {
                    "ticker": "TEST",
                    "search_query": query,
                    "purpose": "latest_results",
                },
            )
            is None
        )
    assert (
        ledger.authorize_tool(
            "search_foreign_sources",
            {
                "ticker": "TEST",
                "search_query": "latest filing revenue",
                "purpose": "latest_results",
            },
        )
        == "PURPOSE_CALL_BUDGET_EXHAUSTED"
    )


def test_shared_research_ledger_opens_failed_host_circuit_after_threshold() -> None:
    ledger = ResearchBudgetLedger(_foreign_policy(host_failure_limit=2))
    first = {"url": "https://ir.example.com/a.pdf"}
    second = {"url": "https://ir.example.com/b.pdf"}
    third = {"url": "https://ir.example.com/c.pdf"}
    for args in (first, second):
        assert ledger.authorize_tool("get_official_document", args) is None
        ledger.record_tool_result(
            "get_official_document",
            "STATUS: INSUFFICIENT_DATA\nREASON: DOCUMENT_DNS_FAILED",
            args=args,
        )

    assert ledger.authorize_tool("get_official_document", third) == (
        "HOST_FAILURE_CIRCUIT_OPEN"
    )
    assert ledger.telemetry()["host_failures"] == {"ir.example.com": 2}


def test_shared_research_ledger_round_trips_persisted_telemetry() -> None:
    policy = _foreign_policy()
    ledger = ResearchBudgetLedger(policy)
    args = {
        "ticker": "TEST",
        "search_query": "issuer guidance",
        "purpose": "management_guidance",
    }
    assert ledger.authorize_tool("search_foreign_sources", args) is None
    ledger.record_tool_round(["search_foreign_sources"])
    ledger.record_forced_synthesis("TOOL_ROUND_LIMIT")

    restored = ResearchBudgetLedger.from_telemetry(policy, ledger.telemetry())

    assert restored.authorize_tool("search_foreign_sources", args) == (
        "DUPLICATE_TOOL_CALL"
    )
    assert restored.stop_reason == "TOOL_ROUND_LIMIT"


def test_value_trap_policy_bounds_the_observed_reformulation_pathology() -> None:
    policy = graph_research_budget_policies(quick_mode=True)["value_trap_detector"]

    assert policy.tool_limit("get_ownership_structure") == 1
    assert policy.tool_limit("get_official_filings") == 1
    assert policy.tool_limit("get_news") == 2
    assert policy.tool_limit("search_foreign_sources") == 6
    assert policy.max_tool_iterations == 4
    assert policy.max_llm_calls == 6


def test_junior_fundamentals_budget_is_one_parallel_tool_round_plus_synthesis() -> None:
    policy = graph_research_budget_policies(quick_mode=True)[
        "junior_fundamentals_analyst"
    ]

    assert policy.max_tool_iterations == 1
    assert policy.max_tool_calls_per_turn == 2
    assert policy.tool_limit("get_financial_metrics") == 1
    assert policy.tool_limit("get_fundamental_analysis") == 1
    assert policy.max_llm_calls == 3


@pytest.mark.parametrize("quick_mode", (False, True))
def test_graph_research_policies_reserve_one_structural_recovery_turn(
    quick_mode: bool,
) -> None:
    policies = graph_research_budget_policies(quick_mode=quick_mode)

    assert len(policies) == 6
    for agent, policy in policies.items():
        assert policy.max_llm_calls == policy.max_tool_iterations + 2, agent
