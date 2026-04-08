from unittest.mock import Mock

from src.agents.output_validation import (
    extract_completion_tokens,
    get_configured_output_cap,
    should_fail_closed,
    validate_required_output,
)


def test_validate_required_output_accepts_parseable_data_block():
    content = """
### --- START DATA_BLOCK ---
RAW_HEALTH_SCORE: 7/12
ADJUSTED_HEALTH_SCORE: 58%
### --- END DATA_BLOCK ---
"""

    validation = validate_required_output("fundamentals_analyst", content)

    assert validation["ok"] is True
    assert validation["missing"] == []


def test_validate_required_output_detects_missing_pm_sections():
    content = """
### PORTFOLIO MANAGER VERDICT: BUY
### THESIS COMPLIANCE SUMMARY
"""

    validation = validate_required_output("portfolio_manager", content)

    assert validation["ok"] is False
    assert "execution_section" in validation["missing"]


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


def test_auditor_validation_accepts_fenced_forensic_block():
    content = (
        "### --- START FORENSIC_DATA_BLOCK ---\n"
        "STATUS: CLEAN\n"
        "VERDICT: RELY_ON_DATA_BLOCK\n"
        "### --- END FORENSIC_DATA_BLOCK ---"
    )

    validation = validate_required_output("global_forensic_auditor", content)

    assert validation["ok"] is True


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
