"""
Tests for FORENSIC_DATA_BLOCK prompt additions.

Verifies that:
1. Auditor prompt contains FORENSIC_DATA_BLOCK template and instructions
2. Consultant prompt contains forensic validation logic
3. Portfolio Manager prompt contains forensic penalties
"""

import pytest

from src.prompts import get_prompt


class TestAuditorForensicDataBlock:
    """Test auditor prompt contains FORENSIC_DATA_BLOCK instructions."""

    def test_auditor_has_forensic_data_block_template(self):
        """Verify auditor prompt contains the FORENSIC_DATA_BLOCK template."""
        auditor = get_prompt("global_forensic_auditor")
        assert auditor is not None, "Auditor prompt not found"

        system_msg = auditor.system_message
        assert (
            "FORENSIC_DATA_BLOCK:" in system_msg
        ), "FORENSIC_DATA_BLOCK template missing"
        assert "META:" in system_msg, "META field missing from template"
        assert "EARNINGS_QUALITY:" in system_msg, "EARNINGS_QUALITY field missing"
        assert "CASH_CYCLE:" in system_msg, "CASH_CYCLE field missing"
        assert "SOFT_ASSETS:" in system_msg, "SOFT_ASSETS field missing"
        assert "SOLVENCY:" in system_msg, "SOLVENCY field missing"
        assert "CASH_INTEGRITY:" in system_msg, "CASH_INTEGRITY field missing"
        assert "ANOMALIES:" in system_msg, "ANOMALIES field missing"

    def test_auditor_has_international_terminology(self):
        """Verify auditor prompt includes multilingual field names."""
        auditor = get_prompt("global_forensic_auditor")
        system_msg = auditor.system_message

        # Check for Japanese terms
        assert "総資産" in system_msg, "Japanese term for Total Assets missing"
        assert "売掛金" in system_msg, "Japanese term for AR missing"
        assert "売上高" in system_msg, "Japanese term for Revenue missing"

        # Check for Chinese terms
        assert "总资产" in system_msg, "Chinese term for Total Assets missing"
        assert "应收账款" in system_msg, "Chinese term for AR missing"
        assert "其他应收款" in system_msg, "Chinese term for Other Receivables missing"

        # Check for Korean terms
        assert "자산총계" in system_msg, "Korean term for Total Assets missing"
        assert "매출채권" in system_msg, "Korean term for AR missing"

        # Check for German terms
        assert "Bilanz" in system_msg, "German term for Balance Sheet missing"
        assert "Aktiva" in system_msg, "German term for Assets missing"

    def test_auditor_has_formula_definitions(self):
        """Verify auditor prompt contains calculation formulas."""
        auditor = get_prompt("global_forensic_auditor")
        system_msg = auditor.system_message

        # Check for key formula definitions
        assert "NI_TO_OCF = Operating Cash Flow / Net Income" in system_msg
        assert "PAPER_PROFIT = (Net Income - OCF) / Total Assets" in system_msg
        assert "DSO = (Accounts Receivable / Revenue)" in system_msg
        assert "ZOMBIE_RATIO = EBIT / Interest Expense" in system_msg
        assert "ALTMAN_Z =" in system_msg
        assert "GHOST_YIELD = (Interest Income / Cash" in system_msg
        assert "TRASH_BIN = Other Receivables / Total Assets" in system_msg

    def test_auditor_has_threshold_guidance(self):
        """Verify auditor prompt contains threshold guidance for flags."""
        auditor = get_prompt("global_forensic_auditor")
        system_msg = auditor.system_message

        assert "RED_FLAG if Paper Profit >10%" in system_msg
        assert "RED_FLAG if TOTAL >40%" in system_msg  # Soft assets
        assert "HIGH_RISK if Zombie <1.0" in system_msg
        assert "RED_FLAG if >15%" in system_msg  # Trash bin

    def test_auditor_has_date_criticality_warning(self):
        """Verify auditor prompt emphasizes using actual statement dates."""
        auditor = get_prompt("global_forensic_auditor")
        system_msg = auditor.system_message

        assert "Report_Date = actual financial statement date" in system_msg
        assert "Do NOT use today's date" in system_msg

    def test_auditor_has_accounting_standard_notes(self):
        """Verify auditor prompt includes notes on different accounting standards."""
        auditor = get_prompt("global_forensic_auditor")
        system_msg = auditor.system_message

        assert "IFRS:" in system_msg
        assert "Japanese GAAP:" in system_msg
        assert "Chinese GAAP:" in system_msg
        assert "US GAAP:" in system_msg

    def test_auditor_version_updated(self):
        """Verify auditor version was incremented."""
        auditor = get_prompt("global_forensic_auditor")
        # Version should be 2.2 or higher
        version_parts = auditor.version.split(".")
        assert int(version_parts[0]) >= 2, "Major version should be at least 2"
        assert int(version_parts[1]) >= 2, "Minor version should be at least 2"


class TestConsultantForensicValidation:
    """Test consultant prompt contains forensic validation logic."""

    def test_consultant_has_forensic_validation_section(self):
        """Verify consultant prompt has FORENSIC VALIDATION section."""
        consultant = get_prompt("consultant")
        assert consultant is not None, "Consultant prompt not found"

        system_msg = consultant.system_message
        assert (
            "FORENSIC VALIDATION" in system_msg
        ), "FORENSIC VALIDATION section missing"
        assert "state contains FORENSIC_DATA_BLOCK" in system_msg

    def test_consultant_checks_report_date(self):
        """Verify consultant checks REPORT_DATE age."""
        consultant = get_prompt("consultant")
        system_msg = consultant.system_message

        assert "Analysis_Date - REPORT_DATE = Age_In_Months" in system_msg
        assert "OPINION = QUALIFIED" in system_msg or "OPINION = ADVERSE" in system_msg

    def test_consultant_compares_with_fundamentals(self):
        """Verify consultant compares forensic flags with Senior Fundamentals."""
        consultant = get_prompt("consultant")
        system_msg = consultant.system_message

        assert "Forensic metrics conflict with Fundamentals DATA_BLOCK" in system_msg
        assert "Fundamentals DATA_BLOCK" in system_msg
        assert "Data Conflict" in system_msg or "MEDIUM RISK" in system_msg

    def test_consultant_checks_flag_discrepancies(self):
        """Verify consultant checks for flag discrepancies."""
        consultant = get_prompt("consultant")
        system_msg = consultant.system_message

        assert (
            "Material Red Flags" in system_msg
            or "Qualified/Adverse opinion" in system_msg
        )
        assert "conflict" in system_msg

    def test_consultant_outputs_forensic_assessment(self):
        """Verify consultant outputs FORENSIC ASSESSMENT section."""
        consultant = get_prompt("consultant")
        system_msg = consultant.system_message

        assert "FORENSIC ASSESSMENT" in system_msg

    def test_consultant_uses_judgment_not_hard_rules(self):
        """Verify consultant uses judgment, not automatic rejections."""
        consultant = get_prompt("consultant")
        system_msg = consultant.system_message

        assert 'Reserve "MAJOR CONCERNS" for decision-changing issues' in system_msg
        assert "Threshold Calibration" in system_msg

    def test_consultant_version_updated(self):
        """Verify consultant version was incremented."""
        consultant = get_prompt("consultant")
        # Version should be 1.1 or higher
        version_parts = consultant.version.split(".")
        assert int(version_parts[0]) >= 1, "Major version should be at least 1"
        assert int(version_parts[1]) >= 1, "Minor version should be at least 1"


class TestPortfolioManagerForensicPenalties:
    """Test portfolio manager prompt contains forensic penalties."""

    def test_pm_has_forensic_penalties_section(self):
        """Verify PM prompt has Forensic Penalties section."""
        pm = get_prompt("portfolio_manager")
        assert pm is not None, "Portfolio Manager prompt not found"

        system_msg = pm.system_message
        assert "Forensic Penalties" in system_msg, "Forensic Penalties section missing"
        assert "FORENSIC_DATA_BLOCK present" in system_msg

    def test_pm_has_auditor_opinion_penalties(self):
        """Verify PM has penalties for auditor opinions."""
        pm = get_prompt("portfolio_manager")
        system_msg = pm.system_message

        assert "OPINION = ADVERSE" in system_msg and "+2.0" in system_msg
        assert "OPINION = QUALIFIED" in system_msg and "+1.0" in system_msg

    def test_pm_has_red_flag_penalties(self):
        """Verify PM has penalties for RED_FLAG findings."""
        pm = get_prompt("portfolio_manager")
        system_msg = pm.system_message

        assert "RED_FLAG in EARNINGS_QUALITY" in system_msg and "+1.5" in system_msg
        assert "RED_FLAG in SOFT_ASSETS" in system_msg or "RED_FLAG in" in system_msg
        assert "CASH_INTEGRITY" in system_msg and "+1.0" in system_msg

    def test_pm_has_altman_z_penalty(self):
        """Verify PM has penalty for Altman Z-Score distress."""
        pm = get_prompt("portfolio_manager")
        system_msg = pm.system_message

        assert "ALTMAN_Z < 1.8" in system_msg
        assert "+1.0" in system_msg  # Should have the penalty value

    def test_pm_has_concern_flag_penalties(self):
        """Verify PM has penalties for CONCERN flags."""
        pm = get_prompt("portfolio_manager")
        system_msg = pm.system_message

        assert "CONCERN flags" in system_msg and "+0.5" in system_msg

    def test_pm_has_stale_data_penalty(self):
        """Verify PM has penalty for stale data."""
        pm = get_prompt("portfolio_manager")
        system_msg = pm.system_message

        assert ">18 months old" in system_msg
        assert (
            "FORENSIC DATA STALE" in system_msg
            or "Forensic data is stale" in system_msg
        )

    def test_pm_has_consultant_conflict_penalty(self):
        """Verify PM has penalty for consultant-detected conflicts."""
        pm = get_prompt("portfolio_manager")
        system_msg = pm.system_message

        assert (
            "Consultant Classification" in system_msg and "Data Conflict" in system_msg
        )
        assert "CRITICAL metric" in system_msg or "SECONDARY metric" in system_msg
        assert "+1.5" in system_msg or "+0.5" in system_msg

    def test_pm_no_hard_fail_for_forensic(self):
        """Verify forensic findings are advisory (no hard fails mentioned)."""
        pm = get_prompt("portfolio_manager")
        system_msg = pm.system_message

        # Check that forensic penalties section doesn't contain hard fail language
        forensic_section_start = system_msg.find("**Forensic Penalties")
        if forensic_section_start != -1:
            # Get next 500 chars after forensic penalties section starts
            forensic_section = system_msg[
                forensic_section_start : forensic_section_start + 500
            ]

            # Should not have HARD FAIL or MANDATORY SELL in forensic section
            assert (
                "HARD FAIL" not in forensic_section
            ), "Forensic should not trigger HARD FAIL"
            assert (
                "MANDATORY SELL" not in forensic_section
            ), "Forensic should not trigger MANDATORY SELL"

    def test_pm_version_updated(self):
        """Verify PM version was incremented."""
        pm = get_prompt("portfolio_manager")
        # Version should be 7.2 or higher
        version_parts = pm.version.split(".")
        assert int(version_parts[0]) >= 7, "Major version should be at least 7"
        assert int(version_parts[1]) >= 2, "Minor version should be at least 2"


class TestForensicDataBlockIntegration:
    """Test that all three prompts work together coherently."""

    def test_all_three_prompts_reference_forensic_data_block(self):
        """Verify auditor, consultant, and PM all reference FORENSIC_DATA_BLOCK."""
        auditor = get_prompt("global_forensic_auditor")
        consultant = get_prompt("consultant")
        pm = get_prompt("portfolio_manager")

        assert "FORENSIC_DATA_BLOCK" in auditor.system_message
        assert "FORENSIC_DATA_BLOCK" in consultant.system_message
        assert "FORENSIC_DATA_BLOCK" in pm.system_message

    def test_consistent_flag_terminology(self):
        """Verify consistent use of RED_FLAG, CONCERN, CLEAN across prompts."""
        auditor = get_prompt("global_forensic_auditor")
        consultant = get_prompt("consultant")
        pm = get_prompt("portfolio_manager")

        # All should reference RED_FLAG or equivalent terminology
        assert "RED_FLAG" in auditor.system_message
        assert (
            "Material Red Flags" in consultant.system_message
            or "Qualified/Adverse opinion" in consultant.system_message
        )
        assert "RED_FLAG" in pm.system_message

        # Auditor should define CONCERN and CLEAN
        assert "CONCERN" in auditor.system_message
        assert "CLEAN" in auditor.system_message

    def test_consistent_field_names(self):
        """Verify consistent field names across prompts."""
        auditor = get_prompt("global_forensic_auditor")
        consultant = get_prompt("consultant")
        pm = get_prompt("portfolio_manager")

        # Key fields should be mentioned consistently
        assert "EARNINGS_QUALITY" in auditor.system_message
        assert "EARNINGS_QUALITY" in pm.system_message

        assert "SOFT_ASSETS" in auditor.system_message
        assert "SOFT_ASSETS" in pm.system_message

        assert "CASH_INTEGRITY" in auditor.system_message
        assert "CASH_INTEGRITY" in pm.system_message

    def test_workflow_consistency(self):
        """Verify workflow: Auditor creates -> Consultant validates -> PM applies penalties."""
        auditor = get_prompt("global_forensic_auditor")
        consultant = get_prompt("consultant")
        pm = get_prompt("portfolio_manager")

        # Auditor should create the block
        assert "Append this block at end of report" in auditor.system_message

        # Consultant should validate it
        assert "FORENSIC VALIDATION" in consultant.system_message
        assert (
            "Forensic metrics conflict with Fundamentals DATA_BLOCK"
            in consultant.system_message
            or "Hierarchy of Truth" in consultant.system_message
        )

        # PM should apply penalties
        assert "Forensic Penalties" in pm.system_message
        assert (
            "+1.5" in pm.system_message or "+1.0" in pm.system_message
        )  # Has numeric penalties
