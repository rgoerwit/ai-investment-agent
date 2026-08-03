from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from src.agents.forensic_repair import canonicalize_forensic_auditor_output
from src.agents.management_guidance import (
    GUIDANCE_PROMOTION_FIELDS,
    _build_unresolved_guidance_block,
    backfill_guidance_contract,
)
from src.agents.output_validation import (
    extract_completion_tokens,
    get_configured_output_cap,
    log_output_diagnostics,
    log_truncation_diagnostic,
    should_fail_closed,
    validate_required_output,
)
from src.data_block_utils import extract_block_text_value
from src.earnings_baseline import (
    REQUIRED_GUIDANCE_CONTRACT_ENUMS,
    REQUIRED_GUIDANCE_CONTRACT_FIELDS,
    guidance_contract_value_is_uninterpretable,
    requires_eps_growth_withholding,
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


def test_every_required_guidance_field_has_a_code_owned_producer():
    """A validator-required field the code cannot produce is an unfixable rejection.

    GUIDANCE_BRIDGE_STATUS became unconditionally required while appearing in zero
    prompts and having no Senior-side producer, so any run whose Foreign Language
    Analyst degraded lost its entire Portfolio Manager verdict. This asserts the
    structural property that made that possible can never recur: every required
    field must be promotable *and* emitted by the conservative fallback block.
    """
    required = set(REQUIRED_GUIDANCE_CONTRACT_FIELDS)

    assert required == set(REQUIRED_GUIDANCE_CONTRACT_ENUMS), (
        "REQUIRED_GUIDANCE_CONTRACT_FIELDS and _ENUMS disagree; both are consumed "
        "by _promoted_management_guidance_issue"
    )
    assert required <= set(GUIDANCE_PROMOTION_FIELDS.values()), (
        "a required guidance field is not promotable from the FLA block: "
        f"{sorted(required - set(GUIDANCE_PROMOTION_FIELDS.values()))}"
    )

    fallback = _build_unresolved_guidance_block({}, "")
    for source_field, target_field in GUIDANCE_PROMOTION_FIELDS.items():
        if target_field not in required:
            continue
        value = extract_block_text_value(fallback, source_field)
        assert value, f"{target_field} has no conservative fallback value"
        assert (
            value.strip().upper() in REQUIRED_GUIDANCE_CONTRACT_ENUMS[target_field]
        ), f"{target_field} fallback {value!r} is outside its own enum"


def test_absent_foreign_language_analyst_still_yields_a_valid_contract():
    """The SAP.DE-class failure: a degraded FLA must not destroy the analysis."""
    body = "SECTOR: Financials\nPE_RATIO_TTM: 13.30"

    updated, backfilled = backfill_guidance_contract(body, "")
    content = f"### --- START DATA_BLOCK ---\n{updated}\n### --- END DATA_BLOCK ---\n"

    assert set(backfilled) == set(REQUIRED_GUIDANCE_CONTRACT_FIELDS)
    assert validate_required_output("fundamentals_analyst", content)["ok"] is True
    # Conservative, not merely valid: trailing EPS growth must stay unusable.
    assert "GUIDANCE_BRIDGE_STATUS: UNRESOLVED" in updated
    assert "GUIDANCE_COVERAGE_STATUS: SEARCH_FAILED" in updated
    assert requires_eps_growth_withholding("UNKNOWN", "UNRESOLVED") is True


def test_backfill_never_overwrites_promoted_guidance():
    """Filling an absent field must never downgrade a sourced one."""
    promoted = (
        "GUIDANCE_COVERAGE_STATUS: FOUND\n"
        "MATERIAL_NONOPERATING_DRIVER: NO\n"
        "EARNINGS_BASELINE_STATUS: DURABLE\n"
        "NORMALIZED_EARNINGS_AVAILABLE: YES\n"
        "GUIDANCE_BRIDGE_STATUS: RECONCILED"
    )

    updated, backfilled = backfill_guidance_contract(promoted, "")

    assert backfilled == ()
    assert updated == promoted


def test_backfill_replaces_an_out_of_enum_guidance_value():
    """Models emit tokens the contract does not define; uninterpretable == absent.

    Measured on the persisted corpus: five runs (6831.HK, AGS.BR, 6782.TW, GTT.PA,
    2458.TW, all 2026-07-28) lost their Portfolio Manager to a literal
    `GUIDANCE_COVERAGE_STATUS: MISSING` — present, so an absence-only predicate
    skipped it, and out-of-enum, so the validator rejected it.
    """
    emitted = (
        "GUIDANCE_COVERAGE_STATUS: MISSING\n"
        "MATERIAL_NONOPERATING_DRIVER: UNKNOWN\n"
        "EARNINGS_BASELINE_STATUS: UNKNOWN\n"
        "NORMALIZED_EARNINGS_AVAILABLE: UNKNOWN\n"
        "GUIDANCE_BRIDGE_STATUS: UNRESOLVED"
    )

    updated, backfilled = backfill_guidance_contract(emitted, "")
    content = f"### --- START DATA_BLOCK ---\n{updated}\n### --- END DATA_BLOCK ---\n"

    assert backfilled == ("GUIDANCE_COVERAGE_STATUS",)
    assert "GUIDANCE_COVERAGE_STATUS: SEARCH_FAILED" in updated
    assert "MISSING" not in updated
    assert validate_required_output("fundamentals_analyst", content)["ok"] is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        # Null tokens the validator folds to UNKNOWN must survive untouched.
        ("MATERIAL_NONOPERATING_DRIVER", "N/A"),
        ("EARNINGS_BASELINE_STATUS", "N/A"),
        # N/A is a first-class member of this field's enum.
        ("NORMALIZED_EARNINGS_AVAILABLE", "N/A"),
    ],
)
def test_backfill_preserves_values_the_validator_accepts(field: str, value: str):
    """Replacement may only add meaning, never overwrite an acceptable value."""
    assert guidance_contract_value_is_uninterpretable(field, value) is False


def test_backfill_fills_only_the_missing_guidance_field():
    """The retry path emits the four fields its correction names, and not the fifth."""
    partial = (
        "GUIDANCE_COVERAGE_STATUS: FOUND\n"
        "MATERIAL_NONOPERATING_DRIVER: YES\n"
        "EARNINGS_BASELINE_STATUS: TEMPORARILY_BOOSTED\n"
        "NORMALIZED_EARNINGS_AVAILABLE: NO"
    )

    updated, backfilled = backfill_guidance_contract(partial, "")

    assert backfilled == ("GUIDANCE_BRIDGE_STATUS",)
    assert "GUIDANCE_COVERAGE_STATUS: FOUND" in updated
    assert "EARNINGS_BASELINE_STATUS: TEMPORARILY_BOOSTED" in updated
    assert "GUIDANCE_BRIDGE_STATUS: UNRESOLVED" in updated


def test_backfill_reports_a_completed_search_without_inventing_one():
    """Coverage status must reflect whether a search actually ran."""
    ran = """
### --- START MANAGEMENT_GUIDANCE ---
COVERAGE_STATUS: NOT_DISCLOSED_AFTER_TARGETED_SEARCH
SEARCHES_COMPLETED: results_package=SUCCEEDED/RESULTS_FOUND; earnings_bridge=SUCCEEDED/NO_RESULTS
### --- END MANAGEMENT_GUIDANCE ---
"""
    prose = """
### --- START MANAGEMENT_GUIDANCE ---
COVERAGE_STATUS: FOUND
SEARCHES_COMPLETED: everything imaginable
### --- END MANAGEMENT_GUIDANCE ---
"""

    ran_body, _ = backfill_guidance_contract("SECTOR: Industrials", ran)
    prose_body, _ = backfill_guidance_contract("SECTOR: Industrials", prose)

    assert "GUIDANCE_COVERAGE_STATUS: UNRESOLVED_AFTER_TARGETED_SEARCH" in ran_body
    # A self-attested prose value is not evidence a search ran.
    assert "GUIDANCE_COVERAGE_STATUS: SEARCH_FAILED" in prose_body


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


def _guidance_data_block(
    normalized_available: str,
    *,
    coverage: str = "NOT_DISCLOSED_AFTER_TARGETED_SEARCH",
    material_driver: str = "UNKNOWN",
    baseline_status: str = "UNKNOWN",
) -> str:
    return f"""
### --- START DATA_BLOCK ---
RAW_HEALTH_SCORE: 7/12
ADJUSTED_HEALTH_SCORE: 58%
GUIDANCE_COVERAGE_STATUS: {coverage}
MATERIAL_NONOPERATING_DRIVER: {material_driver}
EARNINGS_BASELINE_STATUS: {baseline_status}
NORMALIZED_EARNINGS_AVAILABLE: {normalized_available}
GUIDANCE_BRIDGE_STATUS: NOT_APPLICABLE
### --- END DATA_BLOCK ---
"""


def test_guidance_validation_accepts_present_na_normalized_earnings():
    # 6831.HK regression: NORMALIZED_EARNINGS_AVAILABLE's allowed set includes the
    # literal "N/A", but the normalized reader stripped that token to None (rendered
    # "" by canonical_enum), so a contractually-valid present N/A was reported as
    # <missing> and the artifact was non-publishable. The raw read fixes this.
    validation = validate_required_output(
        "fundamentals_analyst", _guidance_data_block("N/A")
    )

    assert validation["ok"] is True
    assert "promoted_management_guidance" not in validation.get("issues", {})


@pytest.mark.parametrize("token", ["N/A", "YES", "NO", "UNKNOWN"])
def test_guidance_validation_accepts_valid_normalized_earnings_tokens(token):
    validation = validate_required_output(
        "fundamentals_analyst", _guidance_data_block(token)
    )

    assert validation["ok"] is True


@pytest.mark.parametrize("token", ["n/a", " N/A ", "n/A"])
def test_guidance_validation_normalizes_na_case_and_whitespace(token):
    validation = validate_required_output(
        "fundamentals_analyst", _guidance_data_block(token)
    )

    assert validation["ok"] is True


def test_guidance_validation_accepts_na_as_semantically_unknown():
    validation = validate_required_output(
        "fundamentals_analyst",
        _guidance_data_block(
            "N/A",
            material_driver="N/A",
            baseline_status="N/A",
        ),
    )

    assert validation["ok"] is True


def test_guidance_validation_rejects_absent_normalized_earnings():
    block = """
### --- START DATA_BLOCK ---
RAW_HEALTH_SCORE: 7/12
ADJUSTED_HEALTH_SCORE: 58%
GUIDANCE_COVERAGE_STATUS: NOT_DISCLOSED_AFTER_TARGETED_SEARCH
MATERIAL_NONOPERATING_DRIVER: UNKNOWN
EARNINGS_BASELINE_STATUS: UNKNOWN
GUIDANCE_BRIDGE_STATUS: NOT_APPLICABLE
### --- END DATA_BLOCK ---
"""

    validation = validate_required_output("fundamentals_analyst", block)

    assert validation["ok"] is False
    assert validation["issues"]["promoted_management_guidance"] == (
        "NORMALIZED_EARNINGS_AVAILABLE=<missing>; expected one of: "
        "N/A, NO, UNKNOWN, YES"
    )


def test_guidance_validation_rejects_invalid_normalized_earnings_token():
    validation = validate_required_output(
        "fundamentals_analyst", _guidance_data_block("MAYBE")
    )

    assert validation["ok"] is False
    assert (
        "NORMALIZED_EARNINGS_AVAILABLE=MAYBE"
        in validation["issues"]["promoted_management_guidance"]
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
