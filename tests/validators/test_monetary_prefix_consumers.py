"""Sanitized Oct-5 report fragments exercise the shared monetary consumers."""

import json
from pathlib import Path

import pytest

from src.agents.capital_structure import _amount_supported, _parse_exposure_amount
from src.agents.support import compute_data_conflicts
from src.exchange_metadata import CURRENCY_DISPLAY_FORMATS
from src.monetary import has_valid_monetary_grouping, parse_monetary_amount
from src.validators.financial_rules import (
    detect_red_flags,
    extract_datablock_ocf_observation,
    parse_ocf_amount,
)
from src.validators.metric_extractor import extract_metrics
from src.validators.sector_classifier import Sector

_REPORTS = json.loads(
    (Path(__file__).parents[1] / "fixtures/monetary_prefix_reports.json").read_text()
)
_EXPECTED = [
    (266.69e6, 33.31e6, 76.66e6),
    (3.62e9, 1.67e9, 2131.41e6),
    (403.14e6, 268.93e6, 276.08e6),
    (2.14e9, 593.86e6, 178.90e6),
]


@pytest.mark.parametrize(
    "number,expected",
    [
        ("1,234.125", 1234.125),
        ("-1,234.125", -1234.125),
        ("+1,234.125", 1234.125),
        ("1,23,456.125", 123456.125),
        ("-1,23,456.125", -123456.125),
        ("+1,23,456.125", 123456.125),
    ],
)
def test_shared_grouping_preserves_supported_signed_money(number, expected):
    assert has_valid_monetary_grouping(number)
    assert parse_monetary_amount(f"{number} USD") == expected


@pytest.mark.parametrize("row,expected", list(zip(_REPORTS, _EXPECTED, strict=True)))
def test_retained_prefix_reports_reach_metric_and_comparison_consumers(row, expected):
    report = row["report_fragment"]
    lines = report.splitlines()
    metrics = extract_metrics(
        "### --- START DATA_BLOCK ---\n"
        + "\n".join(lines[:2])
        + "\n### --- END DATA_BLOCK ---\n"
        + "\n".join(lines[2:])
    )
    assert tuple(metrics[key] for key in ("ocf", "fcf", "net_income")) == pytest.approx(
        expected
    )
    assert metrics["ocf"] / metrics["net_income"] == pytest.approx(
        expected[0] / expected[2]
    )
    amount = report.splitlines()[0].partition(":")[2].strip()
    assert parse_ocf_amount(amount) == pytest.approx(expected[0])
    conflicts = compute_data_conflicts(
        json.dumps({"operatingCashflow": expected[0] / 2}),
        f"Operating Cash Flow (Filing): {amount}\nPeriod: FY2026",
    )
    assert "INVESTIGATE" in conflicts


@pytest.mark.parametrize("amount", ["-SGD  5.2M", "(SGD 5.2M)", "(5.2M SGD)"])
def test_signed_prefix_and_accounting_amounts_remain_negative(amount):
    assert parse_monetary_amount(amount) == -5.2e6
    assert (
        extract_metrics(
            f"### --- START DATA_BLOCK ---\nOPERATING_CASH_FLOW: {amount}\n### --- END DATA_BLOCK ---"
        )["ocf"]
        == -5.2e6
    )
    assert _parse_exposure_amount(amount) is None
    assert not _amount_supported(f"Exposure: {amount}", "SGD 5.2M")


@pytest.mark.parametrize("currency,amount", [("AUD", "266.69M"), ("CAD", "2.14B")])
def test_capital_source_matching_still_binds_positive_amount_and_currency(
    currency, amount
):
    expected = f"{currency} {amount}"
    assert _amount_supported(f"Guarantee: {expected}", expected)
    assert not _amount_supported(f"Guarantee: USD {amount}", expected)
    assert not _amount_supported(f"Guarantee: -{expected}", expected)


@pytest.mark.parametrize("amount", ["AUD 1,5M", "R$4.1 bi", "NT$1’234M", "USD-100K"])
def test_malformed_first_field_cannot_fall_through_to_later_value(amount):
    assert (
        extract_metrics(
            f"### --- START DATA_BLOCK ---\nOPERATING_CASH_FLOW: {amount}; prior: CAD 2M\n### --- END DATA_BLOCK ---"
        )["ocf"]
        is None
    )


@pytest.mark.parametrize(
    "ocf,sector,flagged",
    [
        ("AUD 266.69M", Sector.INDUSTRIALS, True),
        ("AUD 229.98M", Sector.INDUSTRIALS, False),
        ("AUD 229.9801M", Sector.INDUSTRIALS, True),
        ("AUD 266.69M", Sector.FINANCIALS, False),
    ],
)
def test_prefix_amounts_execute_cash_conversion_review_with_controlled_common_basis(
    ocf, sector, flagged
):
    # Same FY2026 consolidated AUD basis is controlled here; the retained report
    # fragments alone do not establish NI's accounting period compatibility.
    metrics = extract_metrics(
        f"### --- START DATA_BLOCK ---\nOPERATING_CASH_FLOW: {ocf}\nOCF_PERIOD: FY\nMETRIC_SCOPE_OCF: CONSOLIDATED\n### --- END DATA_BLOCK ---\n**Net Income**: AUD 76.66M"
    )
    flags, _ = detect_red_flags(metrics, ticker="TEST", sector=sector)
    cash_flags = [flag for flag in flags if flag["type"] == "SUSPICIOUS_OCF_NI_RATIO"]
    assert bool(cash_flags) is flagged
    if flagged:
        assert cash_flags[0]["risk_penalty"] == 0.0
        assert cash_flags[0]["action"] == "REVIEW"


@pytest.mark.parametrize(
    "amount", ["--SGD 5.2M", "(SGD 5.2M", "- SGD -5.2M", "USD 5.2M BRL"]
)
def test_free_scanner_cannot_reenter_malformed_signed_prefix(amount):
    assert parse_monetary_amount(amount) is None
    assert parse_ocf_amount(amount) is None
    assert parse_ocf_amount(f"{amount}; prior CAD 2M") is None


@pytest.mark.parametrize(
    "amount,currency",
    [
        ("R$3.62B", "BRL"),
        ("NT$403.14M", "TWD"),
        ("A$266.69M", "AUD"),
        ("HK$5.2M", "HKD"),
        ("RMB5.2M", "CNY"),
        ("CNY5.2M", "CNY"),
        ("Rp5.2M", "IDR"),
        ("CAD$5.2M", "CAD"),
        ("$5.2M", None),
        ("¥5.2M", None),
    ],
)
def test_shared_currency_identity_reaches_ocf_observations(amount, currency):
    report = f"### --- START DATA_BLOCK ---\nOPERATING_CASH_FLOW: {amount}\nOCF_PERIOD: FY\nLATEST_QUARTER_DATE: 2026-06-30\nMETRIC_SCOPE_OCF: CONSOLIDATED\n### --- END DATA_BLOCK ---"
    observation = extract_datablock_ocf_observation(report)
    assert observation.amount == parse_monetary_amount(amount)
    assert observation.currency == currency
    assert observation.scope == "CONSOLIDATED"


@pytest.mark.parametrize(
    "amount,currency",
    [("GBp100", "GBp"), ("GBP100", "GBP"), ("GBX100", "GBX"), ("gbx100", "GBX")],
)
def test_monetary_identity_preserves_minor_denomination(amount, currency):
    from src.monetary import extract_monetary_currency

    assert parse_monetary_amount(amount) == 100
    assert extract_monetary_currency(amount) == currency


def test_shared_display_symbols_and_currency_only_metadata_agree():
    from src.exchange_metadata import CURRENCY_SYMBOL_TO_CODE
    from src.monetary import extract_monetary_currency

    for symbol, currency in CURRENCY_SYMBOL_TO_CODE.items():
        assert extract_monetary_currency(f"{symbol}100M") == currency
    assert extract_monetary_currency("Currency: S$") == "SGD"
    assert extract_monetary_currency("Currency: SGD") == "SGD"
    assert extract_monetary_currency("Currency: $") is None


@pytest.mark.parametrize(
    "amount,currency,measurable",
    [
        ("GBp100", "GBp", True),
        ("GBp100", "GBP", False),
        ("GBP100", "GBp", False),
        ("RMB100", "CNY", True),
    ],
)
def test_capital_ratio_requires_identical_denominations(amount, currency, measurable):
    from src.agents.capital_structure import assess_capital_structure_scale

    result = assess_capital_structure_scale(
        {"amount": amount, "amount_basis": "MAXIMUM_EXPOSURE"},
        {"financialCurrency": currency, "totalDebt": 200, "stockholdersEquity": 500},
        leverage_threshold=100,
    )
    assert (result["status"] == "MEASURABLE") is measurable


@pytest.mark.parametrize(
    "first,second,blocked",
    [("GBp", "GBP", True), ("GBP", "GBp", True), ("RMB", "CNY", False)],
)
def test_ocf_comparison_preserves_currency_and_minor_unit_basis(first, second, blocked):
    from datetime import date

    from src.validators.financial_rules import OcfObservation, _ocf_comparison_blocker

    def observation(currency):
        return OcfObservation(
            amount=100_000_000,
            period="FY",
            period_end=date(2025, 12, 31),
            currency=currency,
            scope="CONSOLIDATED",
        )

    result = _ocf_comparison_blocker(observation(first), observation(second))
    assert bool(result and result.startswith("CURRENCY_MISMATCH")) is blocked


@pytest.mark.parametrize("currency", CURRENCY_DISPLAY_FORMATS)
@pytest.mark.parametrize("template", ["{currency}100M", "100M {currency}"])
def test_shared_currency_registry_codes_are_recognized(currency, template):
    from src.monetary import extract_monetary_currency

    text = template.format(currency=currency)
    assert parse_monetary_amount(text) == 100_000_000
    assert extract_monetary_currency(text) == currency
