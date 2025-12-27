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
        from src.toolkit import WITHHOLDING_TAX_RATES

        # Check key countries exist
        assert "japan" in WITHHOLDING_TAX_RATES
        assert "hong kong" in WITHHOLDING_TAX_RATES
        assert "china" in WITHHOLDING_TAX_RATES
        assert "germany" in WITHHOLDING_TAX_RATES

    def test_known_withholding_rates(self):
        """Test known withholding rates are correct."""
        from src.toolkit import WITHHOLDING_TAX_RATES

        # Japan has treaty rate of 15%
        assert WITHHOLDING_TAX_RATES["japan"] == "15%"

        # Hong Kong has 0% withholding
        assert WITHHOLDING_TAX_RATES["hong kong"] == "0%"

        # China has 10%
        assert WITHHOLDING_TAX_RATES["china"] == "10%"
