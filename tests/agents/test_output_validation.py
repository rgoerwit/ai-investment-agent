from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.agents.forensic_repair import canonicalize_forensic_auditor_output
from src.agents.output_validation import (
    extract_completion_tokens,
    get_configured_output_cap,
    log_output_diagnostics,
    log_truncation_diagnostic,
    should_fail_closed,
    validate_required_output,
)


def _with_latest_results(content: str) -> str:
    return (
        content
        + """
### --- START LATEST_RESULTS ---
LATEST_RESULTS_COVERAGE_STATUS: NOT_FOUND
LATEST_RESULTS_PERIOD: N/A
LATEST_RESULTS_PERIOD_END: N/A
LATEST_RESULTS_PRIOR_PERIOD: N/A
LATEST_RESULTS_PRIOR_PERIOD_END: N/A
LATEST_RESULTS_PERIOD_MONTHS: N/A
LATEST_RESULTS_CURRENCY: N/A
LATEST_RESULTS_REPORTING_UNIT: N/A
LATEST_RESULTS_REVENUE: N/A
LATEST_RESULTS_PRIOR_REVENUE: N/A
LATEST_RESULTS_EARNINGS: N/A
LATEST_RESULTS_PRIOR_EARNINGS: N/A
LATEST_RESULTS_EARNINGS_SCOPE: N/A
LATEST_RESULTS_SOURCE_URL: N/A
### --- END LATEST_RESULTS ---
"""
    )


def test_validate_required_output_accepts_parseable_data_block():
    content = """
### --- START DATA_BLOCK ---
RAW_HEALTH_SCORE: 7/12
ADJUSTED_HEALTH_SCORE: 58%
GUIDANCE_COVERAGE_STATUS: NOT_DISCLOSED_AFTER_TARGETED_SEARCH
MATERIAL_NONOPERATING_DRIVER: UNKNOWN
EARNINGS_BASELINE_STATUS: UNKNOWN
NORMALIZED_EARNINGS_AVAILABLE: UNKNOWN
GUIDANCE_BRIDGE_STATUS: NOT_APPLICABLE
### --- END DATA_BLOCK ---
"""

    validation = validate_required_output("fundamentals_analyst", content)

    assert validation["ok"] is True
    assert validation["missing"] == []


def test_fundamentals_validation_rejects_silently_dropped_guidance_fields():
    content = """
### --- START DATA_BLOCK ---
RAW_HEALTH_SCORE: 7/12
ADJUSTED_HEALTH_SCORE: 58%
### --- END DATA_BLOCK ---
"""

    validation = validate_required_output("fundamentals_analyst", content)

    assert validation["ok"] is False
    assert "promoted_management_guidance" in validation["missing"]


def test_fundamentals_validation_identifies_invalid_guidance_enum_value():
    content = """
### --- START DATA_BLOCK ---
GUIDANCE_COVERAGE_STATUS: MISSING
MATERIAL_NONOPERATING_DRIVER: UNKNOWN
EARNINGS_BASELINE_STATUS: UNKNOWN
NORMALIZED_EARNINGS_AVAILABLE: UNKNOWN
GUIDANCE_BRIDGE_STATUS: UNRESOLVED
### --- END DATA_BLOCK ---
"""

    validation = validate_required_output("fundamentals_analyst", content)

    assert validation["ok"] is False
    assert (
        validation["issues"]["promoted_management_guidance"]
        == "GUIDANCE_COVERAGE_STATUS=MISSING; expected one of: FOUND, "
        "NOT_APPLICABLE, NOT_DISCLOSED_AFTER_TARGETED_SEARCH, SEARCH_FAILED, "
        "UNRESOLVED_AFTER_TARGETED_SEARCH"
    )


def test_foreign_language_validation_accepts_sourced_guidance_block():
    content = """
### --- START MANAGEMENT_GUIDANCE ---
COVERAGE_STATUS: FOUND
SOURCE_DATE: 2026-05-08
SOURCE_URL: https://example.com/results
SEARCHES_COMPLETED: results release, presentation, transcript, filing
SEARCH_PROVENANCE: CODE_OWNED_PREFLIGHT
OPERATING_VS_NET_DIRECTION: OP_UP_NET_DOWN
MATERIAL_NONOPERATING_DRIVER: YES
DRIVER_TYPE: TAX_CREDIT
DRIVER_PERSISTENCE: EXPIRING
EARNINGS_BASELINE_STATUS: TEMPORARILY_BOOSTED
GUIDANCE_BRIDGE_STATUS: RECONCILED
### --- END MANAGEMENT_GUIDANCE ---
"""

    validation = validate_required_output(
        "foreign_language_analyst", _with_latest_results(content)
    )

    assert validation["ok"] is True


def test_foreign_language_validation_requires_explicit_negative_search_coverage():
    content = """
### --- START MANAGEMENT_GUIDANCE ---
COVERAGE_STATUS: NOT_DISCLOSED_AFTER_TARGETED_SEARCH
SOURCE_DATE: N/A
SOURCE_URL: N/A
SEARCHES_COMPLETED: N/A
SEARCH_PROVENANCE: CODE_OWNED_PREFLIGHT
### --- END MANAGEMENT_GUIDANCE ---
"""

    validation = validate_required_output(
        "foreign_language_analyst", _with_latest_results(content)
    )

    assert validation["ok"] is False
    assert validation["missing"] == ["management_guidance_block"]
    assert should_fail_closed(
        "foreign_language_analyst",
        validation=validation,
        truncated=False,
        content=content,
    )


def test_foreign_language_validation_accepts_targeted_unresolved_status():
    content = """
### --- START MANAGEMENT_GUIDANCE ---
COVERAGE_STATUS: UNRESOLVED_AFTER_TARGETED_SEARCH
SEARCHES_COMPLETED: results_package=COMPLETED; earnings_bridge=INSUFFICIENT_DATA
SEARCH_PROVENANCE: CODE_OWNED_PREFLIGHT
EARNINGS_BASELINE_STATUS: UNKNOWN
GUIDANCE_BRIDGE_STATUS: UNRESOLVED
### --- END MANAGEMENT_GUIDANCE ---
"""

    validation = validate_required_output(
        "foreign_language_analyst", _with_latest_results(content)
    )

    assert validation["ok"] is True


def test_foreign_language_validation_rejects_false_durable_divergence():
    content = """
### --- START MANAGEMENT_GUIDANCE ---
COVERAGE_STATUS: FOUND
SOURCE_DATE: 2026-05-08
SOURCE_URL: https://example.com/results
SEARCHES_COMPLETED: results_package=COMPLETED; earnings_bridge=COMPLETED
SEARCH_PROVENANCE: CODE_OWNED_PREFLIGHT
OPERATING_VS_NET_DIRECTION: OP_UP_NET_DOWN
MATERIAL_NONOPERATING_DRIVER: NO
DRIVER_TYPE: NONE
DRIVER_PERSISTENCE: N/A
EARNINGS_BASELINE_STATUS: DURABLE
GUIDANCE_BRIDGE_STATUS: UNRESOLVED
### --- END MANAGEMENT_GUIDANCE ---
"""

    validation = validate_required_output(
        "foreign_language_analyst", _with_latest_results(content)
    )

    assert validation["ok"] is False


def test_validate_required_output_detects_missing_pm_sections():
    content = """
### PORTFOLIO MANAGER VERDICT: BUY
### THESIS COMPLIANCE SUMMARY
"""

    validation = validate_required_output("portfolio_manager", content)

    assert validation["ok"] is False
    assert "position_section" in validation["missing"]
    assert "decision_facts" in validation["missing"]
    assert "decision_gates" in validation["missing"]


def test_validate_required_output_accepts_complete_pm_trace():
    content = """
### PORTFOLIO MANAGER VERDICT: HOLD
### THESIS COMPLIANCE SUMMARY
### --- START PM_BLOCK ---
VERDICT: HOLD
DECISION_FACTS: claim:pe_ratio_ttm:123
DECISION_GATES: NONE
### --- END PM_BLOCK ---
"""

    validation = validate_required_output("portfolio_manager", content)

    assert validation["ok"] is True


def test_validate_required_output_accepts_consultant_structure():
    content = """
### CONSULTANT REVIEW: APPROVED
### FINAL CONSULTANT VERDICT
Overall Assessment: APPROVED
"""

    validation = validate_required_output("consultant", content)

    assert validation["ok"] is True


def test_extract_completion_tokens_tolerates_mock_usage_metadata():
    response = Mock()
    response.usage_metadata = Mock()
    response.response_metadata = {}

    assert extract_completion_tokens(response) == 0


def test_extract_completion_tokens_reads_response_metadata_usage():
    response = Mock()
    response.usage_metadata = None
    response.response_metadata = {"usage": {"output_tokens": 52}}

    assert extract_completion_tokens(response) == 52


def test_get_configured_output_cap_ignores_non_numeric_mock_attrs():
    runnable = Mock()

    assert get_configured_output_cap(runnable) is None


def test_consultant_validation_does_not_fail_closed_on_short_nontruncated_output():
    validation = {
        "ok": False,
        "checks": [("final_verdict", False)],
        "missing": ["final_verdict"],
    }

    assert (
        should_fail_closed(
            "consultant",
            validation=validation,
            truncated=False,
            content="CONSULTANT REVIEW: APPROVED",
        )
        is False
    )


def test_auditor_validation_rejects_status_only_stub():
    content = "STATUS: REVIEW"

    validation = validate_required_output("global_forensic_auditor", content)

    assert validation["ok"] is False
    assert "forensic_block" in validation["missing"]


def test_auditor_validation_accepts_legacy_forensic_block():
    content = "FORENSIC_DATA_BLOCK:\n" "STATUS: CLEAN\n" "VERDICT: RELY_ON_DATA_BLOCK\n"

    validation = validate_required_output("global_forensic_auditor", content)

    assert validation["ok"] is True


def test_auditor_validation_accepts_markdown_verdict_variant():
    content = (
        "FORENSIC_DATA_BLOCK:\n"
        "STATUS: INSUFFICIENT_DATA\n"
        "**Verdict**: Unable to complete forensic audit from verified filings.\n"
    )

    validation = validate_required_output("global_forensic_auditor", content)

    assert validation["ok"] is True


def test_auditor_validation_accepts_bold_colon_verdict_variant():
    content = (
        "```\n"
        "FORENSIC_DATA_BLOCK:\n"
        "STATUS: INSUFFICIENT_DATA\n"
        "META: N/A\n"
        "```\n"
        "**Verdict:** Unable to complete forensic audit from verified filings.\n"
    )

    validation = validate_required_output("global_forensic_auditor", content)

    assert validation["ok"] is True


def test_auditor_validation_accepts_fenced_forensic_block():
    content = (
        "### --- START FORENSIC_DATA_BLOCK ---\n"
        "STATUS: CLEAN\n"
        "VERDICT: RELY_ON_DATA_BLOCK\n"
        "### --- END FORENSIC_DATA_BLOCK ---"
    )

    validation = validate_required_output("global_forensic_auditor", content)

    assert validation["ok"] is True


def test_auditor_validation_accepts_canonicalized_skt_style_fallback():
    content = (
        "## FORENSIC AUDITOR REPORT\n\n"
        "**STATUS**: INSUFFICIENT_DATA\n\n"
        "FORENSIC_DATA_BLOCK:\n"
        "STATUS: INSUFFICIENT_DATA\n"
        "META: UNKNOWN | Report_Date: UNKNOWN\n"
    )

    validation = validate_required_output(
        "global_forensic_auditor",
        canonicalize_forensic_auditor_output(content),
    )

    assert validation["ok"] is True


def test_auditor_validation_accepts_canonicalized_inline_stub():
    content = (
        "FORENSIC_DATA_BLOCK: STATUS=INSUFFICIENT_DATA, "
        "REASON=STALE_DATA, REPORT_DATE=2025-06-30, AGE=9 months"
    )

    validation = validate_required_output(
        "global_forensic_auditor",
        canonicalize_forensic_auditor_output(content),
    )

    assert validation["ok"] is True


def test_auditor_validation_rejects_prose_only_output():
    content = "I could not verify the statements or auditor report for this ticker."

    validation = validate_required_output(
        "global_forensic_auditor",
        canonicalize_forensic_auditor_output(content),
    )

    assert validation["ok"] is False
    assert "forensic_block" in validation["missing"]


def test_auditor_validation_fails_closed_for_invalid_structure():
    validation = {
        "ok": False,
        "checks": [("forensic_block", False)],
        "missing": ["forensic_block"],
    }

    assert (
        should_fail_closed(
            "global_forensic_auditor",
            validation=validation,
            truncated=False,
            content="STATUS: REVIEW",
        )
        is True
    )


def test_portfolio_manager_validation_fails_closed_when_required_structure_missing():
    validation = {
        "ok": False,
        "checks": [("execution_section", False)],
        "missing": ["execution_section"],
    }

    assert (
        should_fail_closed(
            "portfolio_manager",
            validation=validation,
            truncated=False,
            content="### PORTFOLIO MANAGER VERDICT: BUY",
        )
        is True
    )


def test_log_truncation_diagnostic_warns_for_code_truncation():
    runnable = Mock()
    response = Mock()
    response.usage_metadata = {}
    response.response_metadata = {}

    with patch("src.agents.output_validation.logger") as mock_logger:
        log_truncation_diagnostic(
            agent_key="consultant",
            ticker="TEST",
            runnable=runnable,
            response=response,
            content="Some content\n[...TRUNCATED 5000 chars...]",
            trunc_info={
                "truncated": True,
                "source": "code",
                "marker": "[...TRUNCATED",
                "confidence": "high",
            },
        )

    mock_logger.warning.assert_called_once()
    assert mock_logger.warning.call_args[0][0] == "agent_output_truncated"


def test_log_truncation_diagnostic_warns_near_output_cap_with_upgrade_suggestion():
    runnable = Mock()
    runnable._configured_max_completion_tokens = 1000
    runnable._configured_api_completion_tokens = 1000
    response = Mock()
    response.usage_metadata = {"completion_tokens": 950}
    response.response_metadata = {}

    with patch("src.agents.output_validation.logger") as mock_logger:
        log_truncation_diagnostic(
            agent_key="news_analyst",
            ticker="TEST",
            runnable=runnable,
            response=response,
            content="OPPORTUNITY: Benefiting from",
            trunc_info={
                "truncated": True,
                "source": "llm",
                "marker": "ends with: 'OPPORTUNITY: Benefiting from'",
                "confidence": "medium",
            },
        )

    mock_logger.warning.assert_called_once()
    assert mock_logger.warning.call_args[0][0] == "agent_output_truncated"
    assert (
        mock_logger.warning.call_args[1]["suggestion"]
        == "consider increasing max output tokens for this agent"
    )
    assert mock_logger.warning.call_args[1]["utilization_ratio"] == 0.95
    assert mock_logger.warning.call_args[1]["intent_utilization_ratio"] is None
    assert mock_logger.warning.call_args[1]["api_utilization_ratio"] == 0.95


def test_log_truncation_diagnostic_downgrades_heuristic_low_utilization_to_info():
    runnable = Mock()
    runnable._configured_max_completion_tokens = 1000
    runnable._configured_api_completion_tokens = 1000
    response = Mock()
    response.usage_metadata = {"completion_tokens": 200}
    response.response_metadata = {}

    with patch("src.agents.output_validation.logger") as mock_logger:
        log_truncation_diagnostic(
            agent_key="news_analyst",
            ticker="TEST",
            runnable=runnable,
            response=response,
            content="The company remains exposed to",
            trunc_info={
                "truncated": True,
                "source": "llm",
                "marker": "ends with: 'The company remains exposed to'",
                "confidence": "medium",
            },
        )

    mock_logger.info.assert_called_once()
    assert mock_logger.info.call_args[0][0] == "agent_output_truncation_suspected"
    mock_logger.warning.assert_not_called()


def test_log_truncation_diagnostic_warns_for_incomplete_required_block():
    runnable = Mock()
    response = Mock()
    response.usage_metadata = {}
    response.response_metadata = {}

    with patch("src.agents.output_validation.logger") as mock_logger:
        log_truncation_diagnostic(
            agent_key="portfolio_manager",
            ticker="TEST",
            runnable=runnable,
            response=response,
            content="PM_BLOCK:\nTICKER: TEST",
            trunc_info={
                "truncated": True,
                "source": "llm",
                "marker": "incomplete PM_BLOCK block (missing ('VERDICT:', 'RISK_ZONE:', 'ZONE:'))",
                "confidence": "medium",
            },
        )

    mock_logger.warning.assert_called_once()
    assert mock_logger.warning.call_args[0][0] == "agent_output_truncated"


def test_log_truncation_diagnostic_prefers_reserve_suggestion_when_thinking_consumes_cap():
    runnable = Mock()
    runnable._configured_max_output_tokens = 2048
    runnable._configured_api_output_tokens = 4096
    response = Mock()
    response.usage_metadata = {
        "output_tokens": 3900,
        "output_token_details": {"reasoning": 3700},
    }
    response.response_metadata = {}

    with patch("src.agents.output_validation.logger") as mock_logger:
        log_truncation_diagnostic(
            agent_key="market_analyst",
            ticker="Y92.SI",
            runnable=runnable,
            response=response,
            content="### LIQUIDITY ASSESSMENT\n**Trading Regularity",
            trunc_info={
                "truncated": True,
                "source": "llm",
                "marker": "ends with: 'Trading Regularity'",
                "confidence": "medium",
            },
        )

    payload = mock_logger.warning.call_args[1]
    assert (
        payload["suggestion"]
        == "consider increasing reasoning reserve / API output cap"
    )
    assert payload["thinking_tokens"] == 3700
    assert payload["visible_output_tokens"] == 200
    assert payload["intent_utilization_ratio"] == 0.0977
    assert payload["api_utilization_ratio"] == 0.9521


def test_log_output_diagnostics_reads_openai_object_metadata_on_final_response():
    runnable = Mock()
    runnable._configured_max_completion_tokens = 8192
    runnable._configured_api_completion_tokens = 10240
    response = Mock()
    response.usage_metadata = None
    response.response_metadata = SimpleNamespace(
        token_usage=SimpleNamespace(
            prompt_tokens=1077,
            completion_tokens=834,
            total_tokens=1911,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=516),
        )
    )

    with patch("src.agents.output_validation.logger") as mock_logger:
        log_output_diagnostics(
            agent_key="global_forensic_auditor",
            ticker="Y92.SI",
            runnable=runnable,
            response=response,
            content="FORENSIC_DATA_BLOCK:\nSTATUS: CLEAN",
            truncated=False,
            validation={"ok": True, "missing": []},
        )

    payload = mock_logger.debug.call_args[1]
    assert payload["completion_tokens_total"] == 834
    assert payload["thinking_tokens"] == 516
    assert payload["visible_output_tokens"] == 318
    assert payload["intent_utilization_ratio"] == 0.0388
    assert payload["api_utilization_ratio"] == 0.0814
