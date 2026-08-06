"""
Tests for Legal Counsel Agent and Legal Flag Detection

This module tests the Legal Counsel agent that detects PFIC, VIE, CMIC,
and other regulatory risks for US investors in ex-US equities.

Test categories:
1. Legal risk extraction from JSON output (PFIC, VIE, CMIC, other)
2. Legal flag detection (PFIC/VIE/CMIC warnings)
3. Integration with red flag detector

Run with: pytest tests/test_legal_counsel.py -v
"""

import json

from src.runtime_diagnostics import ArtifactStatus
from src.validators.red_flag_detector import RedFlagDetector


class TestLegalRiskExtraction:
    """Test extraction of legal/tax risks from Legal Counsel output."""

    def test_extract_clean_json(self):
        """Test extraction from clean JSON output."""
        legal_report = json.dumps(
            {
                "pfic_status": "PROBABLE",
                "pfic_evidence": "Company states it may be classified as PFIC",
                "pfic_source": "20-F 2024",
                "vie_structure": "YES",
                "vie_evidence": "Uses contractual VIE arrangements for China operations",
                "withholding_rate": "10%",
                "country": "China",
                "sector": "Technology",
            }
        )

        risks = RedFlagDetector.extract_legal_risks(legal_report)

        assert risks["pfic_status"] == "PROBABLE"
        assert risks["pfic_evidence"] == "Company states it may be classified as PFIC"
        assert risks["vie_structure"] == "YES"
        assert (
            risks["vie_evidence"]
            == "Uses contractual VIE arrangements for China operations"
        )
        assert risks["country"] == "China"
        assert risks["sector"] == "Technology"

    def test_extract_clean_pfic_status(self):
        """Test PFIC status CLEAN extraction."""
        legal_report = json.dumps(
            {
                "pfic_status": "CLEAN",
                "pfic_evidence": "Company explicitly states it is not a PFIC",
                "vie_structure": "N/A",
                "vie_evidence": None,
                "country": "Japan",
                "sector": "Automotive",
            }
        )

        risks = RedFlagDetector.extract_legal_risks(legal_report)

        assert risks["pfic_status"] == "CLEAN"
        assert risks["vie_structure"] == "N/A"

    def test_extract_uncertain_pfic_status(self):
        """Test PFIC status UNCERTAIN extraction."""
        legal_report = json.dumps(
            {
                "pfic_status": "UNCERTAIN",
                "pfic_evidence": "Uses hedge language: 'we believe but no assurance'",
                "vie_structure": "NO",
                "country": "Hong Kong",
                "sector": "Finance",
            }
        )

        risks = RedFlagDetector.extract_legal_risks(legal_report)

        assert risks["pfic_status"] == "UNCERTAIN"
        assert risks["vie_structure"] == "NO"

    def test_extract_na_pfic_status(self):
        """Test PFIC status N/A extraction for non-high-risk sectors."""
        legal_report = json.dumps(
            {
                "pfic_status": "N/A",
                "pfic_evidence": "No PFIC disclosure found",
                "vie_structure": "N/A",
                "country": "Germany",
                "sector": "Automotive",
            }
        )

        risks = RedFlagDetector.extract_legal_risks(legal_report)

        assert risks["pfic_status"] == "N/A"
        assert risks["vie_structure"] == "N/A"

    def test_extract_from_markdown_code_block(self):
        """Test extraction when JSON is wrapped in markdown code block."""
        legal_report = """```json
{
    "pfic_status": "PROBABLE",
    "pfic_evidence": "PFIC warning in 20-F",
    "vie_structure": "NO"
}
```"""

        risks = RedFlagDetector.extract_legal_risks(legal_report)

        assert risks["pfic_status"] == "PROBABLE"
        assert risks["vie_structure"] == "NO"

    def test_extract_fallback_regex(self):
        """Test regex fallback when JSON parsing fails."""
        legal_report = """
        Based on my analysis:
        pfic_status: PROBABLE
        vie_structure: YES
        The company has significant PFIC concerns.
        """

        risks = RedFlagDetector.extract_legal_risks(legal_report)

        assert risks["pfic_status"] == "PROBABLE"
        assert risks["vie_structure"] == "YES"

    def test_extract_empty_report(self):
        """Test extraction from empty report."""
        risks = RedFlagDetector.extract_legal_risks("")

        assert risks["pfic_status"] is None
        assert risks["vie_structure"] is None

    def test_extract_none_report(self):
        """Test extraction from None report."""
        risks = RedFlagDetector.extract_legal_risks(None)

        assert risks["pfic_status"] is None
        assert risks["vie_structure"] is None

    def test_extract_cmic_flagged(self):
        """Test CMIC FLAGGED extraction."""
        legal_report = json.dumps(
            {
                "pfic_status": "N/A",
                "vie_structure": "N/A",
                "cmic_status": "FLAGGED",
                "cmic_evidence": "Company appears on OFAC NS-CMIC list",
                "country": "China",
                "sector": "Defense",
            }
        )

        risks = RedFlagDetector.extract_legal_risks(legal_report)

        assert risks["cmic_status"] == "FLAGGED"
        assert "NS-CMIC" in risks["cmic_evidence"]

    def test_malformed_json_fallback_recovers_cmic_evidence(self):
        legal_report = (
            '{"pfic_status":"N/A","vie_structure":"N/A",'
            '"cmic_status":"FLAGGED",'
            '"cmic_evidence":"Company appears on OFAC NS-CMIC list",}'
        )

        risks = RedFlagDetector.extract_legal_risks(legal_report)

        assert risks["cmic_status"] == "FLAGGED"
        assert risks["cmic_evidence"] == "Company appears on OFAC NS-CMIC list"

    def test_extract_cmic_uncertain(self):
        """Test CMIC UNCERTAIN extraction."""
        legal_report = json.dumps(
            {
                "pfic_status": "N/A",
                "vie_structure": "N/A",
                "cmic_status": "UNCERTAIN",
                "cmic_evidence": "State-owned enterprise in sensitive sector",
                "country": "China",
                "sector": "Semiconductors",
            }
        )

        risks = RedFlagDetector.extract_legal_risks(legal_report)

        assert risks["cmic_status"] == "UNCERTAIN"

    def test_extract_cmic_clear(self):
        """Test CMIC CLEAR extraction for non-defense Chinese company."""
        legal_report = json.dumps(
            {
                "pfic_status": "N/A",
                "vie_structure": "YES",
                "cmic_status": "CLEAR",
                "cmic_evidence": None,
                "country": "China",
                "sector": "Consumer",
            }
        )

        risks = RedFlagDetector.extract_legal_risks(legal_report)

        assert risks["cmic_status"] == "CLEAR"

    def test_extract_other_regulatory_risks(self):
        """Test extraction of other_regulatory_risks array."""
        legal_report = json.dumps(
            {
                "pfic_status": "N/A",
                "vie_structure": "YES",
                "cmic_status": "CLEAR",
                "other_regulatory_risks": [
                    {
                        "risk_type": "HFCAA",
                        "description": "Pending PCAOB audit compliance",
                        "severity": "HIGH",
                    },
                    {
                        "risk_type": "SDN",
                        "description": "Minor Russia exposure in supply chain",
                        "severity": "LOW",
                    },
                ],
                "country": "China",
                "sector": "Technology",
            }
        )

        risks = RedFlagDetector.extract_legal_risks(legal_report)

        assert len(risks["other_regulatory_risks"]) == 2
        assert risks["other_regulatory_risks"][0]["risk_type"] == "HFCAA"
        assert risks["other_regulatory_risks"][0]["severity"] == "HIGH"

    def test_extract_cmic_regex_fallback(self):
        """Test CMIC extraction via regex fallback."""
        legal_report = """
        Based on my analysis:
        cmic_status: FLAGGED
        The company appears on defense blacklist.
        """

        risks = RedFlagDetector.extract_legal_risks(legal_report)

        assert risks["cmic_status"] == "FLAGGED"

    def test_extract_nested_capital_structure(self):
        capital = {
            "coverage_status": "FOUND",
            "exposure_type": "GUARANTEE_BACKSTOP",
            "classification": "BLOCK_BUY",
        }
        legal_report = json.dumps(
            {
                "pfic_status": "N/A",
                "vie_structure": "N/A",
                "capital_structure": capital,
            }
        )

        risks = RedFlagDetector.extract_legal_risks(legal_report)

        assert risks["capital_structure"] == capital


class TestLegalFlagDetection:
    """Test detection of legal/tax warning flags."""

    def test_pfic_probable_flag(self):
        """Test PFIC_PROBABLE warning flag detection."""
        legal_risks = {
            "pfic_status": "PROBABLE",
            "pfic_evidence": "Company acknowledges PFIC classification",
            "vie_structure": "NO",
            "vie_evidence": None,
        }

        warnings = RedFlagDetector.detect_legal_flags(legal_risks, "TEST.HK")

        assert len(warnings) == 1
        assert warnings[0]["type"] == "PFIC_PROBABLE"
        assert warnings[0]["severity"] == "WARNING"
        assert warnings[0]["action"] == "RISK_PENALTY"
        assert warnings[0]["risk_penalty"] == 1.0

    def test_failed_artifact_is_coverage_gap_not_substantive_legal_risk(self):
        legal_risks = {
            "pfic_status": "UNCERTAIN",
            "pfic_evidence": "Fallback text",
            "vie_structure": "YES",
            "cmic_status": "FLAGGED",
            "other_regulatory_risks": [{"risk_type": "SANCTIONS", "severity": "HIGH"}],
            "capital_structure": {
                "coverage_status": "SEARCH_FAILED",
                "classification": "UNRESOLVED",
            },
        }
        status = ArtifactStatus(
            complete=True,
            ok=False,
            content="fallback",
            error_kind="timeout",
            provider="google",
        )

        warnings = RedFlagDetector.detect_legal_flags(
            legal_risks,
            "TEST",
            artifact_status=status,
        )

        assert [warning["type"] for warning in warnings] == [
            "LEGAL_COUNSEL_UNAVAILABLE"
        ]
        assert warnings[0]["risk_penalty"] == 0.0
        assert warnings[0]["blocks_buy"] is True
        assert "google, timeout" in warnings[0]["detail"]

    def test_successful_artifact_preserves_substantive_legal_flags(self):
        legal_risks = {
            "pfic_status": "PROBABLE",
            "pfic_evidence": "Issuer disclosure",
            "vie_structure": "YES",
            "vie_evidence": "Contractual control",
            "cmic_status": "FLAGGED",
            "cmic_evidence": "Current OFAC match",
        }
        status = ArtifactStatus(
            complete=True,
            ok=True,
            content="valid legal report",
            provider="google",
        )

        warnings = RedFlagDetector.detect_legal_flags(
            legal_risks,
            "TEST",
            artifact_status=status,
        )

        assert {warning["type"] for warning in warnings} == {
            "PFIC_PROBABLE",
            "VIE_STRUCTURE",
            "CMIC_FLAGGED",
        }
        assert sum(warning["risk_penalty"] for warning in warnings) == 3.5

    def test_ordinary_commitment_qualifies_ratios_without_blocking_buy(self):
        legal_risks = {
            "pfic_status": "CLEAN",
            "vie_structure": "NO",
            "capital_structure": {
                "coverage_status": "FOUND",
                "exposure_type": "LEASE_COMMITMENT",
                "classification": "QUALIFY_RATIOS",
                "entity": "Data Center A",
                "amount": "USD 2.0 billion",
                "amount_basis": "UNDISCOUNTED",
                "source_url": "https://example.com/filing",
                "evidence": "Uncommenced leases were disclosed.",
            },
        }

        warnings = RedFlagDetector.detect_legal_flags(legal_risks, "TEST")

        flag = next(flag for flag in warnings if flag["type"] == "DEBT_LIKE_COMMITMENT")
        assert flag["risk_penalty"] == 0.0
        assert flag["blocks_buy"] is False

    def test_material_parent_recourse_blocks_buy(self):
        legal_risks = {
            "pfic_status": "CLEAN",
            "vie_structure": "NO",
            "capital_structure": {
                "coverage_status": "FOUND",
                "exposure_type": "GUARANTEE_BACKSTOP",
                "classification": "BLOCK_BUY",
                "entity": "Unconsolidated JV",
                "amount": "USD 1.2 billion",
                "amount_basis": "MAXIMUM_EXPOSURE",
                "source_url": "https://example.com/filing",
                "evidence": "Parent guarantees the JV borrowing.",
            },
        }

        warnings = RedFlagDetector.detect_legal_flags(legal_risks, "TEST")

        flag = next(
            flag for flag in warnings if flag["type"] == "OFF_BALANCE_SHEET_RECOURSE"
        )
        assert flag["risk_penalty"] == 0.0
        assert flag["blocks_buy"] is True

    def test_retrieval_failure_is_visible_but_does_not_block_buy(self):
        legal_risks = {
            "pfic_status": "CLEAN",
            "vie_structure": "NO",
            "capital_structure": {
                "coverage_status": "SEARCH_FAILED",
                "exposure_type": "UNKNOWN",
                "classification": "UNRESOLVED",
            },
        }

        warnings = RedFlagDetector.detect_legal_flags(legal_risks, "TEST")

        flag = next(
            flag
            for flag in warnings
            if flag["type"] == "CAPITAL_STRUCTURE_EVIDENCE_GAP"
        )
        assert flag["blocks_buy"] is False

    def test_zero_debt_scale_formats_as_minor_warning_without_error(self):
        legal_risks = {
            "pfic_status": "CLEAN",
            "vie_structure": "NO",
            "capital_structure": {
                "coverage_status": "FOUND",
                "exposure_type": "GUARANTEE_BACKSTOP",
                "classification": "BLOCK_BUY",
                "entity": "Small JV",
                "amount": "USD 5 million",
                "amount_basis": "MAXIMUM_EXPOSURE",
                "source_url": "https://example.com/filing",
                "evidence": "Parent guarantee disclosed.",
                "scale_assessment": {
                    "status": "MEASURABLE",
                    "exposure_to_debt_pct": None,
                    "exposure_to_equity_pct": 5.0,
                    "exposure_to_revenue_pct": None,
                    "reported_de_pct": 0.0,
                    "adjusted_de_pct": 5.0,
                    "decision_material": False,
                },
            },
        }

        warnings = RedFlagDetector.detect_legal_flags(legal_risks, "TEST")

        flag = next(
            flag
            for flag in warnings
            if flag["type"] == "OFF_BALANCE_SHEET_RECOURSE_MINOR"
        )
        assert flag["blocks_buy"] is False
        assert "debt comparison N/A" in flag["detail"]

    def test_pfic_uncertain_flag(self):
        """Test PFIC_UNCERTAIN warning flag detection."""
        legal_risks = {
            "pfic_status": "UNCERTAIN",
            "pfic_evidence": "Hedge language used in disclosures",
            "vie_structure": "NO",
            "vie_evidence": None,
        }

        warnings = RedFlagDetector.detect_legal_flags(legal_risks, "TEST.HK")

        assert len(warnings) == 1
        assert warnings[0]["type"] == "PFIC_UNCERTAIN"
        assert warnings[0]["severity"] == "WARNING"
        assert warnings[0]["action"] == "RISK_PENALTY"
        assert warnings[0]["risk_penalty"] == 0.5

    def test_vie_structure_flag(self):
        """Test VIE_STRUCTURE warning flag detection."""
        legal_risks = {
            "pfic_status": "CLEAN",
            "pfic_evidence": "Not a PFIC",
            "vie_structure": "YES",
            "vie_evidence": "Uses VIE contractual structure for mainland operations",
        }

        warnings = RedFlagDetector.detect_legal_flags(legal_risks, "BABA")

        assert len(warnings) == 1
        assert warnings[0]["type"] == "VIE_STRUCTURE"
        assert warnings[0]["severity"] == "WARNING"
        assert warnings[0]["action"] == "RISK_PENALTY"
        assert warnings[0]["risk_penalty"] == 0.5

    def test_combined_pfic_and_vie_flags(self):
        """Test both PFIC and VIE warnings together."""
        legal_risks = {
            "pfic_status": "PROBABLE",
            "pfic_evidence": "PFIC warning in 20-F",
            "vie_structure": "YES",
            "vie_evidence": "VIE structure for China internet operations",
        }

        warnings = RedFlagDetector.detect_legal_flags(legal_risks, "PDD")

        assert len(warnings) == 2
        types = [w["type"] for w in warnings]
        assert "PFIC_PROBABLE" in types
        assert "VIE_STRUCTURE" in types

        total_penalty = sum(w["risk_penalty"] for w in warnings)
        assert total_penalty == 1.5  # 1.0 + 0.5

    def test_clean_no_warnings(self):
        """Test CLEAN status generates no warnings."""
        legal_risks = {
            "pfic_status": "CLEAN",
            "pfic_evidence": "Company explicitly not a PFIC",
            "vie_structure": "NO",
            "vie_evidence": None,
        }

        warnings = RedFlagDetector.detect_legal_flags(legal_risks, "7203.T")

        assert len(warnings) == 0

    def test_na_status_no_warnings(self):
        """Test N/A status generates no warnings."""
        legal_risks = {
            "pfic_status": "N/A",
            "pfic_evidence": None,
            "vie_structure": "N/A",
            "vie_evidence": None,
        }

        warnings = RedFlagDetector.detect_legal_flags(legal_risks, "SAP.DE")

        assert len(warnings) == 0

    def test_warning_not_auto_reject(self):
        """Test that legal warnings use RISK_PENALTY, not AUTO_REJECT."""
        legal_risks = {
            "pfic_status": "PROBABLE",
            "pfic_evidence": "PFIC warning",
            "vie_structure": "YES",
            "vie_evidence": "VIE structure",
        }

        warnings = RedFlagDetector.detect_legal_flags(legal_risks, "TEST")

        for warning in warnings:
            assert warning["action"] == "RISK_PENALTY"
            assert warning["action"] != "AUTO_REJECT"

    def test_cmic_flagged_warning(self):
        """Test CMIC_FLAGGED warning flag detection with high penalty."""
        legal_risks = {
            "pfic_status": "N/A",
            "vie_structure": "N/A",
            "cmic_status": "FLAGGED",
            "cmic_evidence": "Company on NS-CMIC list",
            "other_regulatory_risks": [],
        }

        warnings = RedFlagDetector.detect_legal_flags(legal_risks, "0001.SS")

        assert len(warnings) == 1
        assert warnings[0]["type"] == "CMIC_FLAGGED"
        assert warnings[0]["severity"] == "HIGH"
        assert warnings[0]["action"] == "RISK_PENALTY"
        assert warnings[0]["risk_penalty"] == 2.0  # Highest legal penalty
        # A US person is legally prohibited from initiating this position, so the
        # flag must carry mechanical force, not just a tally the PM can reason past.
        assert warnings[0]["blocks_buy"] is True

    def test_cmic_flagged_demotes_a_buy_verdict(self):
        """End-to-end: the blocks_buy chain turns a PM BUY into HOLD."""
        from src.agents.verdict_policy import maybe_demote_buy_on_blocking_flags

        warnings = RedFlagDetector.detect_legal_flags(
            {"cmic_status": "FLAGGED", "cmic_evidence": "NS-CMIC listed"}, "0001.SS"
        )
        pm_text = (
            "### PORTFOLIO MANAGER VERDICT: BUY\n\n"
            "### --- START PM_BLOCK ---\nVERDICT: BUY\n### --- END PM_BLOCK ---\n"
        )

        out, demoted = maybe_demote_buy_on_blocking_flags(
            pm_text, red_flags=warnings, ticker="0001.SS"
        )

        assert demoted is True
        assert "VERDICT: HOLD" in out
        assert "VERDICT: BUY" not in out

    def test_cmic_uncertain_warning(self):
        """Test CMIC_UNCERTAIN warning flag detection."""
        legal_risks = {
            "pfic_status": "N/A",
            "vie_structure": "N/A",
            "cmic_status": "UNCERTAIN",
            "cmic_evidence": "State-owned enterprise in sensitive sector",
            "other_regulatory_risks": [],
        }

        warnings = RedFlagDetector.detect_legal_flags(legal_risks, "600000.SS")

        assert len(warnings) == 1
        assert warnings[0]["type"] == "CMIC_UNCERTAIN"
        assert warnings[0]["risk_penalty"] == 1.0
        # An unconfirmed connection is a penalty, not a prohibition.
        assert warnings[0].get("blocks_buy") is not True

    def test_cmic_clear_no_warning(self):
        """Test CMIC CLEAR status generates no CMIC warning."""
        legal_risks = {
            "pfic_status": "N/A",
            "vie_structure": "N/A",
            "cmic_status": "CLEAR",
            "cmic_evidence": None,
            "other_regulatory_risks": [],
        }

        warnings = RedFlagDetector.detect_legal_flags(legal_risks, "BABA")

        # No CMIC warning for CLEAR status
        cmic_warnings = [w for w in warnings if "CMIC" in w["type"]]
        assert len(cmic_warnings) == 0

    def test_other_regulatory_risks_high_severity(self):
        """Test other_regulatory_risks with HIGH severity."""
        legal_risks = {
            "pfic_status": "N/A",
            "vie_structure": "N/A",
            "cmic_status": "N/A",
            "other_regulatory_risks": [
                {
                    "risk_type": "HFCAA",
                    "description": "Failing PCAOB audit requirements",
                    "severity": "HIGH",
                }
            ],
        }

        warnings = RedFlagDetector.detect_legal_flags(legal_risks, "LK")

        assert len(warnings) == 1
        assert warnings[0]["type"] == "REGULATORY_HFCAA"
        assert warnings[0]["risk_penalty"] == 1.5  # HIGH = 1.5

    def test_other_regulatory_risks_multiple(self):
        """Test multiple other_regulatory_risks with different severities."""
        legal_risks = {
            "pfic_status": "N/A",
            "vie_structure": "N/A",
            "cmic_status": "N/A",
            "other_regulatory_risks": [
                {"risk_type": "HFCAA", "description": "Audit risk", "severity": "HIGH"},
                {
                    "risk_type": "SDN",
                    "description": "Minor exposure",
                    "severity": "LOW",
                },
                {
                    "risk_type": "ENTITY_LIST",
                    "description": "Export controls",
                    "severity": "MEDIUM",
                },
            ],
        }

        warnings = RedFlagDetector.detect_legal_flags(legal_risks, "TEST")

        assert len(warnings) == 3
        total_penalty = sum(w["risk_penalty"] for w in warnings)
        assert total_penalty == 3.0  # 1.5 + 0.5 + 1.0

    def test_combined_cmic_and_other_risks(self):
        """Test CMIC + other regulatory risks combined."""
        legal_risks = {
            "pfic_status": "PROBABLE",
            "pfic_evidence": "PFIC warning",
            "vie_structure": "YES",
            "vie_evidence": "VIE structure",
            "cmic_status": "UNCERTAIN",
            "cmic_evidence": "Possible defense ties",
            "other_regulatory_risks": [
                {
                    "risk_type": "HFCAA",
                    "description": "Audit risk",
                    "severity": "MEDIUM",
                }
            ],
        }

        warnings = RedFlagDetector.detect_legal_flags(legal_risks, "TEST.HK")

        # Should have: PFIC_PROBABLE, VIE, CMIC_UNCERTAIN, REGULATORY_HFCAA
        assert len(warnings) == 4
        types = [w["type"] for w in warnings]
        assert "PFIC_PROBABLE" in types
        assert "VIE_STRUCTURE" in types
        assert "CMIC_UNCERTAIN" in types
        assert "REGULATORY_HFCAA" in types

        # Total penalty: 1.0 + 0.5 + 1.0 + 1.0 = 3.5
        total_penalty = sum(w["risk_penalty"] for w in warnings)
        assert total_penalty == 3.5


class TestLegalFlagIntegration:
    """Test integration of legal flags with financial red flags."""

    def test_legal_warnings_do_not_cause_reject(self):
        """Test that legal warnings alone don't cause REJECT status."""
        # Create a report with healthy financials
        report = """
### --- START DATA_BLOCK ---
ADJUSTED_HEALTH_SCORE: 75%
PE_RATIO_TTM: 12.00
### --- END DATA_BLOCK ---

**Leverage (2/2 pts)**:
- D/E: 50: 1 pts

**Interest Coverage**: 5.0x
**Free Cash Flow**: $500M
**Net Income**: $400M
"""

        metrics = RedFlagDetector.extract_metrics(report)
        red_flags, result = RedFlagDetector.detect_red_flags(metrics, "TEST")

        # Healthy financials should PASS
        assert result == "PASS"
        assert len(red_flags) == 0

        # Now add legal warnings
        legal_risks = {
            "pfic_status": "PROBABLE",
            "pfic_evidence": "PFIC warning",
            "vie_structure": "YES",
            "vie_evidence": "VIE structure",
        }
        legal_warnings = RedFlagDetector.detect_legal_flags(legal_risks, "TEST")

        # Legal warnings exist but don't trigger reject
        assert len(legal_warnings) == 2
        for warning in legal_warnings:
            assert warning["action"] == "RISK_PENALTY"

    def test_legal_warnings_combined_with_financial_flags(self):
        """Test legal warnings combined with financial red flags."""
        # Create a report with extreme leverage
        report = """
### --- START DATA_BLOCK ---
ADJUSTED_HEALTH_SCORE: 30%
PE_RATIO_TTM: 25.00
### --- END DATA_BLOCK ---

**Leverage (0/2 pts)**:
- D/E: 600: 0 pts

**Interest Coverage**: 5.0x
**Free Cash Flow**: $500M
**Net Income**: $400M
"""

        metrics = RedFlagDetector.extract_metrics(report)
        red_flags, result = RedFlagDetector.detect_red_flags(metrics, "TEST")

        # Extreme leverage should REJECT
        assert result == "REJECT"
        assert len(red_flags) == 1
        assert red_flags[0]["type"] == "EXTREME_LEVERAGE"
        assert red_flags[0]["action"] == "AUTO_REJECT"

        # Add legal warnings
        legal_risks = {
            "pfic_status": "PROBABLE",
            "pfic_evidence": "PFIC warning",
            "vie_structure": "NO",
            "vie_evidence": None,
        }
        legal_warnings = RedFlagDetector.detect_legal_flags(legal_risks, "TEST")

        # Combined flags
        all_flags = red_flags + legal_warnings
        assert len(all_flags) == 2

        # Still should be REJECT due to financial issues
        has_auto_reject = any(f["action"] == "AUTO_REJECT" for f in all_flags)
        assert has_auto_reject


class TestWithholdingTaxRates:
    """Test withholding tax rate lookup in toolkit."""

    def test_withholding_rates_exist(self):
        """Test that WITHHOLDING_TAX_RATES is populated."""
        from src.tools.legal import WITHHOLDING_TAX_RATES

        # Check key countries exist
        assert "japan" in WITHHOLDING_TAX_RATES
        assert "hong kong" in WITHHOLDING_TAX_RATES
        assert "china" in WITHHOLDING_TAX_RATES
        assert "germany" in WITHHOLDING_TAX_RATES

    def test_known_withholding_rates(self):
        """Test known withholding rates are correct."""
        from src.tools.legal import WITHHOLDING_TAX_RATES

        # Japan has treaty rate of 15%
        assert WITHHOLDING_TAX_RATES["japan"] == "15%"

        # Hong Kong has 0% withholding
        assert WITHHOLDING_TAX_RATES["hong kong"] == "0%"

        # China has 10%
        assert WITHHOLDING_TAX_RATES["china"] == "10%"


class _RecordingLogger:
    """Minimal structlog stand-in (repo idiom; capture_logs is config-sensitive)."""

    def __init__(self):
        self.events = []

    def debug(self, event, **kwargs):
        self.events.append((event, kwargs))

    def warning(self, event, **kwargs):
        self.events.append((event, kwargs))

    def error(self, event, **kwargs):
        self.events.append((event, kwargs))


class TestLegalJsonFallbackVisibility:
    """Malformed legal JSON must warn operators, then still extract via regex."""

    def test_malformed_json_warns_and_extracts_fallback_fields(self, monkeypatch):
        from src.validators import supplemental_extractors

        recorder = _RecordingLogger()
        monkeypatch.setattr(supplemental_extractors, "logger", recorder)

        malformed = '{"pfic_status": "PROBABLE", "vie_structure": "YES",}'
        risks = RedFlagDetector.extract_legal_risks(malformed)

        assert risks["pfic_status"] == "PROBABLE"
        assert risks["vie_structure"] == "YES"
        warnings = [
            kwargs
            for event, kwargs in recorder.events
            if event == "legal_report_json_parse_failed_using_regex_fallback"
        ]
        assert len(warnings) == 1
        assert "report_prefix" in warnings[0]

    def test_malformed_json_ignores_prefixed_and_suffixed_decoy_keys(self):
        malformed = """{
            non_pfic_status: "PROBABLE",
            pfic_status_note: "UNCERTAIN",
            previous_vie_structure: "YES",
            vie_structure_detail: "YES",
            non_cmic_status: "FLAGGED",
            cmic_status_source: "UNCERTAIN",
            non_pfic_evidence: "wrong prefix",
            pfic_evidence_note: "wrong suffix",
        }"""

        risks = RedFlagDetector.extract_legal_risks(malformed)

        assert risks["pfic_status"] is None
        assert risks["vie_structure"] is None
        assert risks["cmic_status"] is None
        assert risks["pfic_evidence"] is None

    def test_malformed_json_recovers_exact_keys_after_decoys_and_decodes_escapes(self):
        malformed = r"""{
            non_pfic_status: "PROBABLE",
            pfic_status: "CLEAN",
            pfic_evidence_note: "wrong field",
            pfic_evidence: "Issuer said \"not a PFIC\".\nSee filing.",
            vie_structure: "NO",
            cmic_status: "CLEAR",
        }"""

        risks = RedFlagDetector.extract_legal_risks(malformed)

        assert risks["pfic_status"] == "CLEAN"
        assert risks["pfic_evidence"] == 'Issuer said "not a PFIC".\nSee filing.'
        assert risks["vie_structure"] == "NO"
        assert risks["cmic_status"] == "CLEAR"

    def test_report_prefix_in_warning_is_redacted(self, monkeypatch):
        from src.validators import supplemental_extractors

        recorder = _RecordingLogger()
        monkeypatch.setattr(supplemental_extractors, "logger", recorder)

        malformed = '{"pfic_evidence": "see https://x.test/doc?apikey=TOPSECRET456",}'
        RedFlagDetector.extract_legal_risks(malformed)

        warnings = [
            kwargs
            for event, kwargs in recorder.events
            if event == "legal_report_json_parse_failed_using_regex_fallback"
        ]
        assert len(warnings) == 1
        assert "TOPSECRET456" not in warnings[0]["report_prefix"]

    def test_valid_json_does_not_warn(self, monkeypatch):
        from src.validators import supplemental_extractors

        recorder = _RecordingLogger()
        monkeypatch.setattr(supplemental_extractors, "logger", recorder)

        valid = '{"pfic_status": "CLEAN", "vie_structure": "NO", "country": "Japan"}'
        risks = RedFlagDetector.extract_legal_risks(valid)

        assert risks["pfic_status"] == "CLEAN"
        assert not any(
            event == "legal_report_json_parse_failed_using_regex_fallback"
            for event, _ in recorder.events
        )
