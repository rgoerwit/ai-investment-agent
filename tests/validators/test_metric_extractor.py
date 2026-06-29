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

from src.validators.metric_extractor import extract_metrics


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
