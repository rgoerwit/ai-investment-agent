"""Focused metric-extraction coverage for the red-flag validator."""

from tests.validators.red_flag_validator_cases import (
    TestDataBlockMarkerVariants,
    TestDebtToEquityNormalization,
    TestMetricExtraction,
    TestSegmentOwnershipOCFFields,
)

__all__ = [
    "TestMetricExtraction",
    "TestDataBlockMarkerVariants",
    "TestSegmentOwnershipOCFFields",
    "TestDebtToEquityNormalization",
]

from src.validators.metric_extractor import (
    extract_debt_to_equity,
    extract_interest_coverage,
    extract_metrics,
)


def _block(*lines: str) -> str:
    body = "\n".join(lines)
    return f"### --- START DATA_BLOCK ---\n{body}\n### --- END DATA_BLOCK ---"


class TestNewDataBlockFields:
    """Parser contract for ASSET_TURNOVER / INVENTORY_TURNOVER_TREND /
    CAPACITY_UTILIZATION / FACILITY_BUILDOUT_STATUS (added for the APR mitigation)."""

    def test_positive_parse(self):
        m = extract_metrics(
            _block(
                "ASSET_TURNOVER: 1.92",
                "INVENTORY_TURNOVER_TREND: RISING",
                "CAPACITY_UTILIZATION: 78.5%",
                "FACILITY_BUILDOUT_STATUS: RAMPING",
            )
        )
        assert m["asset_turnover"] == 1.92
        assert m["inventory_turnover_trend"] == "RISING"
        assert m["capacity_utilization"] == 78.5
        assert m["facility_buildout_status"] == "RAMPING"

    def test_na_stays_none(self):
        m = extract_metrics(
            _block(
                "ASSET_TURNOVER: N/A",
                "INVENTORY_TURNOVER_TREND: N/A",
                "CAPACITY_UTILIZATION: N/A",
                "FACILITY_BUILDOUT_STATUS: N/A",
            )
        )
        assert m["asset_turnover"] is None
        assert m["inventory_turnover_trend"] is None
        assert m["capacity_utilization"] is None
        assert m["facility_buildout_status"] is None

    def test_facility_none_is_preserved_as_real_state(self):
        # NONE = "no buildout disclosed" — a real state, distinct from N/A (unknown).
        m = extract_metrics(_block("FACILITY_BUILDOUT_STATUS: NONE"))
        assert m["facility_buildout_status"] == "NONE"

    def test_malformed_capacity_does_not_crash(self):
        m = extract_metrics(_block("CAPACITY_UTILIZATION: high"))
        assert m["capacity_utilization"] is None

    def test_absent_fields_default_none(self):
        m = extract_metrics(_block("NET_MARGIN: 4.9%"))
        assert m["asset_turnover"] is None
        assert m["inventory_turnover_trend"] is None
        assert m["capacity_utilization"] is None
        assert m["facility_buildout_status"] is None


class TestTrailingPeriodNumbers:
    """A DATA_BLOCK value followed by a sentence period (`D/E: 0.30.`) must not
    crash the parser. Regression for the 102260.KS / 1818.HK pipeline FAILs where
    the loose ``[0-9.]+`` class captured the trailing dot and ``float('0.30.')``
    raised, killing the Portfolio Manager node."""

    def test_debt_to_equity_trailing_period(self):
        # 0.30 is a ratio (<10) → normalized to a percentage (×100).
        assert extract_debt_to_equity("D/E: 0.30.") == 30.0
        assert extract_debt_to_equity("Debt/Equity: 0.54.") == 54.0

    def test_debt_to_equity_semantics_preserved(self):
        assert extract_debt_to_equity("D/E: 6.92%") == 6.92  # explicit percent
        assert extract_debt_to_equity("D/E: 6.92") == 692.0  # ratio → percent
        assert extract_debt_to_equity("D/E: 120") == 120.0  # already a percent
        assert extract_debt_to_equity("D/E: N/A") is None

    def test_interest_coverage_trailing_period(self):
        assert extract_interest_coverage("Interest Coverage: 2.5.") == 2.5

    def test_ratios_trailing_period(self):
        m = extract_metrics(
            _block(
                "PE_RATIO_TTM: 6.19.",
                "PB_RATIO: 1.2.",
                "PEG_RATIO: 0.07.",
                "SECTOR_MEDIAN_PE: 12.4.",
                "PE_VS_SECTOR: 0.5.",
            )
        )
        assert m["pe_ratio"] == 6.19
        assert m["pb_ratio"] == 1.2
        assert m["peg_ratio"] == 0.07
        assert m["sector_median_pe"] == 12.4
        assert m["pe_vs_sector"] == 0.5

    def test_currency_trailing_period_and_multiplier(self):
        m = extract_metrics(
            _block(
                "OPERATING_CASH_FLOW: ¥13.39B.",
                "FREE_CASH_FLOW: 1,234.56M.",
            )
        )
        assert m["ocf"] == 13.39e9
        assert m["fcf"] == 1_234.56e6
