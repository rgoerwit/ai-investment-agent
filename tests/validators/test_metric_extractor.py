"""Focused metric-extraction coverage for the red-flag validator."""

import pytest

from src.monetary import parse_monetary_amount
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
    extract_operating_cash_flow,
)
from src.validators.supplemental_flags import detect_return_quality_fragility_flags


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


class TestManagementGuidanceFields:
    def test_material_tax_driver_is_extracted(self):
        m = extract_metrics(
            _block(
                "GUIDANCE_COVERAGE_STATUS: FOUND",
                "GUIDANCE_SOURCE_DATE: 2026-05-08",
                "GUIDANCE_SOURCE_URL: https://finance.logmi.jp/articles/384869",
                "GUIDANCE_PERIOD: FY3/27",
                "GUIDANCE_REVENUE: ¥110.0B",
                "GUIDANCE_OPERATING_PROFIT: ¥12.3B (+5%)",
                "GUIDANCE_ORDINARY_OR_PRETAX_PROFIT: ¥12.5B",
                "GUIDANCE_NET_INCOME: ¥9.0B (-4%)",
                "OPERATING_VS_NET_DIRECTION: OP_UP_NET_DOWN",
                "MATERIAL_NONOPERATING_DRIVER: YES",
                "DRIVER_TYPE: TAX_CREDIT",
                "DRIVER_PERSISTENCE: EXPIRING",
                "DRIVER_MATERIALITY: MATERIAL",
                "DRIVER_AFFECTED_PERIOD: FY3/26",
                "EARNINGS_BASELINE_STATUS: TEMPORARILY_BOOSTED",
                "NORMALIZED_EARNINGS_AVAILABLE: NO",
            )
        )

        assert m["guidance_coverage_status"] == "FOUND"
        assert m["guidance_source_url"] == "https://finance.logmi.jp/articles/384869"
        assert m["guidance_period"] == "FY3/27"
        assert m["operating_vs_net_direction"] == "OP_UP_NET_DOWN"
        assert m["driver_type"] == "TAX_CREDIT"
        assert m["driver_persistence"] == "EXPIRING"
        assert m["earnings_baseline_status"] == "TEMPORARILY_BOOSTED"
        assert m["normalized_earnings_available"] == "NO"

    def test_absent_guidance_fields_remain_none_for_legacy_reports(self):
        m = extract_metrics(_block("NET_MARGIN: 4.9%"))

        assert m["guidance_coverage_status"] is None
        assert m["driver_type"] is None
        assert m["earnings_baseline_status"] is None


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


class TestSignedDecisionMetrics:
    def test_negative_return_metrics_preserve_sign(self):
        metrics = extract_metrics(
            _block(
                "ROA_PERCENT: 13.41%",
                "ROA_5Y_AVG: -12.88%",
                "ROE_5Y_AVG: -4.2%",
                "PEG_RATIO: -0.7",
            )
        )

        assert metrics["roa_current"] == 13.41
        assert metrics["roa_5y_avg"] == -12.88
        assert metrics["roe_5y_avg"] == -4.2
        assert metrics["peg_ratio"] == -0.7

    def test_negative_return_metrics_feed_fragility_rule(self):
        report = _block("ROA_PERCENT: 13.41%", "ROA_5Y_AVG: -12.88%")

        flags = detect_return_quality_fragility_flags(
            report,
            base_metrics=extract_metrics(report),
        )

        assert [flag["type"] for flag in flags] == ["RETURN_QUALITY_FRAGILITY"]

    def test_negative_debt_to_equity_is_signed(self):
        assert extract_debt_to_equity("D/E: -0.30") == -30.0


@pytest.mark.parametrize(
    "amount,expected",
    [
        ("8176419328 KRW", 8176419328),
        ("1,234 BRL", 1234),
        ("-1,234 MYR", -1234),
        ("+1.25B KRW", 1.25e9),
        ("2.5m BRL", 2.5e6),
        ("3K MYR", 3000),
    ],
)
def test_currency_suffix_is_not_a_magnitude(amount, expected):
    metrics = extract_metrics(
        _block(f"OPERATING_CASH_FLOW: {amount}", f"FREE_CASH_FLOW: {amount}")
        + f"\n**Net Income**: {amount}"
    )
    assert metrics["ocf"] == expected
    assert metrics["fcf"] == expected
    assert metrics["net_income"] == expected


def test_korean_cash_conversion_is_not_inflated_by_currency_suffix():
    metrics = extract_metrics(
        _block("OPERATING_CASH_FLOW: 8176419328 KRW") + "\n**Net Income**: 7.00B KRW"
    )
    assert metrics["ocf"] / metrics["net_income"] == pytest.approx(1.168059904)


@pytest.mark.parametrize(
    "amount,expected",
    [
        ("100K KRW", 100000),
        ("100 KRW", 100),
        ("-100K KRW", -100000),
        ("100 million MYR", 100e6),
        ("100bn BRL", 100e9),
        ("1.25T TWD", 1.25e12),
        ("+1,234 CAD", 1234),
        ("100KXYZ", None),
        ("100MMXYZ", None),
    ],
)
def test_shared_monetary_tokens_preserve_sign_scale_and_currency_boundary(
    amount, expected
):
    assert parse_monetary_amount(amount) == expected
    assert extract_operating_cash_flow("Operating Cash Flow: " + amount) == expected


def test_shared_money_parser_rejects_nonfinite_amount():
    assert parse_monetary_amount("9" * 400 + "B") is None


@pytest.mark.parametrize(
    "amount", ["USD-100K", "KRW-100K", "--100K", "100..5M", "1,2,3M", "123,45"]
)
def test_shared_money_parser_rejects_ambiguous_or_malformed_tokens(amount):
    assert parse_monetary_amount(amount) is None


def test_shared_money_parser_preserves_indian_grouping():
    assert parse_monetary_amount("1,23,456.78 INR") == 123456.78


@pytest.mark.parametrize(
    "amount",
    [
        "€1.2 Mrd.",
        "CHF1’234 Mio",
        "R$4.1 bi",
        "100 億",
        "100 万",
        "USD 1,5",
        "CHF1’234",
    ],
)
def test_unsupported_monetary_units_fail_closed(amount):
    assert parse_monetary_amount(amount) is None
    assert extract_operating_cash_flow(f"OPERATING_CASH_FLOW: {amount}") is None


def test_magnitude_followed_by_prose_remains_supported():
    assert parse_monetary_amount("1.2B compared with 1.1B") == 1.2e9


@pytest.mark.parametrize(
    "amount",
    [
        "€1,234 Mio.",
        "EUR 2,500 Mrd",
        "Operating cash flow: 4,800 Mio. EUR",
        "1,200 crore",
        "3,000 employees",
    ],
)
def test_grouped_unsupported_amount_cannot_backtrack_to_numeric_prefix(amount):
    from src.validators.metric_extractor import extract_free_cash_flow

    assert parse_monetary_amount(amount) is None
    assert extract_operating_cash_flow(f"Operating Cash Flow: {amount}") is None
    assert extract_free_cash_flow(f"Free Cash Flow: {amount}") is None
    assert extract_metrics(_block(f"OPERATING_CASH_FLOW: {amount}"))["ocf"] is None
