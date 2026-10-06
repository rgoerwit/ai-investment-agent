"""Behavioral currency contracts across report, FX, and broker boundaries.

Expected values are explicit examples, not computed by the parser under test.
Nominal extraction preserves denominations; only conversion applies FX/unit scale.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.agents.capital_structure import _amount_supported, _parse_exposure_amount
from src.data.pattern_extraction import FinancialPatternExtractor
from src.fx_normalization import (
    FxRateCache,
    canonical_currency_code,
    get_fx_rate,
    get_fx_rate_fallback,
    normalize_financial_dict,
    normalize_to_usd,
)
from src.ibkr.reconciliation_rules import _resolve_fx
from src.monetary import extract_monetary_currency, parse_monetary_amount
from src.validators.financial_rules import (
    extract_datablock_ocf_observation,
    parse_ocf_amount,
)
from src.validators.metric_extractor import extract_metrics
from tests.ibkr.reconciler_cases import _make_analysis


def _report(amount: str) -> str:
    return (
        "### --- START DATA_BLOCK ---\n"
        f"OPERATING_CASH_FLOW: {amount}\nFREE_CASH_FLOW: {amount}\n"
        "OCF_PERIOD: FY\nLATEST_QUARTER_DATE: 2026-06-30\n"
        "METRIC_SCOPE_OCF: CONSOLIDATED\n### --- END DATA_BLOCK ---"
    )


@pytest.mark.parametrize(
    "currency,canonical",
    [
        ("KRW", "KRW"),
        ("krw", "KRW"),
        ("RMB", "CNY"),
        ("CNY", "CNY"),
        ("GBp", "GBp"),
        ("GBP", "GBP"),
        ("GBX", "GBX"),
        ("CAD", "CAD"),
        ("NZD", "NZD"),
    ],
)
@pytest.mark.parametrize(
    "number,expected",
    [
        ("100K", 100_000),
        ("100M", 100_000_000),
        ("1.25B", 1_250_000_000),
        ("1,23,456", 123_456),
    ],
)
@pytest.mark.parametrize("template", ["{currency}{number}", "{number} {currency}"])
def test_same_nominal_amount_and_identity_across_consumers(
    currency, canonical, number, expected, template
):
    amount = template.format(currency=currency, number=number)
    report = _report(amount)
    metrics = extract_metrics(report)
    observation = extract_datablock_ocf_observation(report)
    assert parse_monetary_amount(amount) == expected
    assert parse_ocf_amount(amount) == expected
    assert metrics["ocf"] == expected
    assert metrics["fcf"] == expected
    assert observation.amount == expected
    assert observation.currency == canonical
    assert extract_monetary_currency(amount) == canonical
    assert canonical_currency_code(currency) == canonical
    assert _parse_exposure_amount(amount) == (expected, canonical)
    assert (
        FinancialPatternExtractor().extract_from_text(f"Market Cap (2025): {amount}")[
            "marketCap"
        ]
        == expected
    )


@pytest.mark.parametrize("spelling", ["NZ$", "NZD "])
@pytest.mark.parametrize(
    "number,expected",
    [("83.57M", 83_570_000), ("100K", 100_000), ("1.25B", 1_250_000_000)],
)
def test_observed_new_zealand_symbol_matches_iso_currency(spelling, number, expected):
    amount = f"{spelling}{number}"
    report = _report(amount)
    assert parse_monetary_amount(amount) == expected
    assert extract_monetary_currency(amount) == "NZD"
    assert extract_metrics(report)["ocf"] == expected
    assert extract_metrics(report)["fcf"] == expected
    observation = extract_datablock_ocf_observation(report)
    assert (observation.amount, observation.currency) == (expected, "NZD")
    # Capital exposure requires ISO currency tokens, even for known symbols.
    assert _parse_exposure_amount(amount) == (
        None if spelling == "NZ$" else (expected, "NZD")
    )
    assert (
        FinancialPatternExtractor().extract_from_text(f"Market Cap: {amount}")[
            "marketCap"
        ]
        == expected
    )


@pytest.mark.parametrize(
    "amount",
    [
        "€1,234 Mio.",
        "EUR 2,500 Mrd",
        "4,800 Mio. EUR",
        "1,200 crore",
        "3,000 employees",
        "USD 1,5",
        "KRW 1,,000K",
        "KRW 12,34K",
        "CHF 1’234 Mio.",
        "R$4.1 bi",
        "USD 100K KRW",
        "--KRW 100K",
        "(KRW 100K",
        "KRW 100K..",
    ],
)
def test_invalid_money_never_becomes_a_partial_or_comparative_value(amount):
    assert parse_monetary_amount(amount) is None
    assert parse_ocf_amount(amount) is None
    assert _parse_exposure_amount(amount) is None
    assert parse_ocf_amount(f"{amount}; prior CAD 2M") is None
    for text in [amount, f"{amount}; prior CAD 2M"]:
        metrics = extract_metrics(_report(text))
        assert metrics["ocf"] is None
        assert metrics["fcf"] is None
        assert extract_datablock_ocf_observation(_report(text)) is None
        assert "marketCap" not in FinancialPatternExtractor().extract_from_text(
            f"Market Cap: {text}"
        )


@pytest.mark.parametrize(
    "amount,expected", [("-100K KRW", -100_000), ("(KRW 100K)", -100_000), ("0 KRW", 0)]
)
def test_signed_and_zero_values_preserve_cashflow_but_not_positive_only_claims(
    amount, expected
):
    assert parse_monetary_amount(amount) == expected
    assert extract_metrics(_report(amount))["ocf"] == expected
    observation = extract_datablock_ocf_observation(_report(amount))
    if expected == 0:
        # The OCF comparison contract requires a nonzero observation; nominal
        # extraction still retains zero rather than inventing a missing metric.
        assert observation is None
    else:
        assert observation.amount == expected
    assert _parse_exposure_amount(amount) is None
    assert "marketCap" not in FinancialPatternExtractor().extract_from_text(
        f"Market Cap: {amount}"
    )


@pytest.fixture
def offline_rates(monkeypatch):
    # Controlled USD-per-major-unit rates, independent of the repository's
    # manually maintained production fallback table or live market conditions.
    monkeypatch.setattr(
        "src.fx_normalization.FALLBACK_RATES_TO_USD",
        {
            "USD": 1.0,
            "GBP": 1.25,
            "KRW": 0.001,
            "CNY": 0.125,
            "CAD": 0.8,
        },
    )
    live = AsyncMock(return_value=None)
    monkeypatch.setattr("src.fx_normalization.get_fx_rate_yfinance", live)
    return live


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "currency,expected_rate",
    [
        ("KRW", 0.001),
        (" krw ", 0.001),
        ("RMB", 0.125),
        ("cny", 0.125),
        ("GBP", 1.25),
        ("gbp", 1.25),
        ("GBp", 0.0125),
        ("gbx", 0.0125),
    ],
)
@pytest.mark.parametrize("nominal", [100_000, 0, -100_000])
async def test_scalar_dict_cache_and_broker_apply_one_identical_conversion(
    currency, expected_rate, nominal, offline_rates, monkeypatch
):
    assert get_fx_rate_fallback(currency) == pytest.approx(expected_rate)
    rate, source = await get_fx_rate(currency)
    assert rate == pytest.approx(expected_rate)
    assert source == "fallback"
    parsed = parse_monetary_amount(f"{nominal} {currency.strip()}")
    assert parsed == nominal
    converted, metadata = await normalize_to_usd(parsed, currency)
    assert converted == pytest.approx(nominal * expected_rate)
    assert metadata["original_value"] == nominal
    assert metadata["original_currency"] == currency
    assert metadata["fx_rate"] == pytest.approx(expected_rate)
    normalized = await normalize_financial_dict(
        {
            "currency": currency,
            "operatingCashflow": nominal,
            "freeCashflow": nominal,
            "marketCap": nominal,
            "trailingPE": 12,
            "revenueGrowth": 0.2,
        }
    )
    for field in ("operatingCashflow", "freeCashflow", "marketCap"):
        assert normalized[field] == pytest.approx(converted)
    assert normalized["trailingPE"] == 12
    assert normalized["revenueGrowth"] == 0.2
    assert normalized["currency"] == "USD"
    cache = FxRateCache()
    assert (await cache.get_rate(currency))[0] == pytest.approx(expected_rate)
    assert cache.peek_cached_rate(currency)[0] == pytest.approx(expected_rate)
    monkeypatch.setattr(
        "src.ibkr.reconciliation_rules.get_fx_rate_cache", lambda: cache
    )
    analysis = _make_analysis(ticker="TEST.L", current_price=100)
    analysis.currency = currency
    analysis.fx_rate_to_usd = expected_rate
    assert _resolve_fx(analysis) == pytest.approx(expected_rate)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source,target,expected",
    [
        ("GBp", "GBP", 0.01),
        ("GBP", "GBp", 100),
        ("GBX", "GBp", 1),
        ("USD", "GBp", 80),
        ("GBp", "USD", 0.0125),
        ("RMB", "CAD", 0.15625),
    ],
)
async def test_cross_rates_are_reciprocal_and_cache_target_denomination_separately(
    source, target, expected, offline_rates
):
    cache = FxRateCache()
    assert (await get_fx_rate(source, target))[0] == pytest.approx(expected)
    assert get_fx_rate_fallback(source, target) == pytest.approx(expected)
    assert (await cache.get_rate(source, target))[0] == pytest.approx(expected)
    assert (await cache.get_rate(target, source))[0] == pytest.approx(1 / expected)
    assert cache.peek_cached_rate(source, target)[0] == pytest.approx(expected)


@pytest.mark.asyncio
async def test_cache_deduplicates_aliases_without_merging_pence_pounds_or_targets(
    offline_rates,
):
    cache = FxRateCache()
    rates = await cache.get_rates(["GBp", "GBP", "gbp", "RMB", "CNY", "GBp"])
    assert set(rates) == {"GBp", "GBP", "CNY"}
    assert rates["GBp"][0] == 0.0125
    assert rates["GBP"][0] == 1.25
    assert offline_rates.await_count == 3
    await cache.get_rates(["GBp", "GBP", "RMB"])
    assert offline_rates.await_count == 3
    assert (await cache.get_rate("USD", "GBP"))[0] == 0.8
    assert (await cache.get_rate("USD", "GBp"))[0] == 80
    assert cache.peek_cached_rate("USD", "GBP")[0] == 0.8
    assert cache.peek_cached_rate("USD", "GBp")[0] == 80


@pytest.mark.asyncio
async def test_unavailable_currency_is_not_cached_as_usd_and_can_recover(offline_rates):
    cache = FxRateCache()
    assert await cache.get_rate("ZZZ") == (None, "unavailable")
    assert cache.peek_cached_rate("ZZZ") is None
    original, metadata = await normalize_to_usd(100_000, "ZZZ")
    assert original == 100_000
    assert metadata["normalized"] is False
    assert metadata["fx_source"] == "unavailable"
    offline_rates.return_value = 0.5
    assert await cache.get_rate("ZZZ") == (0.5, "yfinance")
    assert cache.peek_cached_rate("ZZZ") == (0.5, "yfinance")


@pytest.mark.asyncio
async def test_cache_expiry_refreshes_only_the_expired_currency_pair(
    offline_rates, monkeypatch
):
    clock = [100.0]
    monkeypatch.setattr(
        "src.fx_normalization.time", SimpleNamespace(monotonic=lambda: clock[0])
    )
    cache = FxRateCache(cache_ttl_secs=10)
    assert (await cache.get_rate("GBp"))[0] == 0.0125
    clock[0] = 105
    assert (await cache.get_rate("GBP"))[0] == 1.25
    clock[0] = 110
    assert cache.peek_cached_rate("GBp") is None
    assert cache.peek_cached_rate("GBP")[0] == 1.25
    offline_rates.return_value = 0.013
    assert (await cache.get_rate("GBp"))[0] == 0.013
    assert (await cache.get_rate("GBP"))[0] == 1.25
    assert offline_rates.await_count == 3


@pytest.mark.asyncio
async def test_partial_batch_failure_preserves_known_rates_and_retries_only_failure(
    offline_rates,
):
    cache = FxRateCache()
    rates = await cache.get_rates(["KRW", "GBp", "ZZZ"])
    assert set(rates) == {"KRW", "GBp"}
    assert rates["KRW"][0] == 0.001
    assert rates["GBp"][0] == 0.0125
    assert cache.peek_cached_rate("ZZZ") is None
    assert offline_rates.await_count == 3
    offline_rates.return_value = 0.5
    recovered = await cache.get_rates(["KRW", "GBp", "ZZZ"])
    assert recovered["KRW"] == rates["KRW"]
    assert recovered["GBp"] == rates["GBp"]
    assert recovered["ZZZ"] == (0.5, "yfinance")
    assert offline_rates.await_count == 4


@pytest.mark.parametrize(
    "amount", ["NZ$100K USD", "USD 100K NZD", "NZ$1,5M", "--NZ$100K"]
)
def test_new_zealand_invalid_tokens_cannot_supply_amount_or_currency_authority(amount):
    assert parse_monetary_amount(amount) is None
    assert extract_monetary_currency(amount) is None
    assert extract_metrics(_report(amount))["ocf"] is None
    assert not _amount_supported(f"Guarantee: {amount}", "NZD 100K")


def test_symbol_source_supports_iso_exposure_without_cross_currency_matching():
    assert _amount_supported("Guarantee: NZ$100K", "NZD 100K")
    assert not _amount_supported("Guarantee: A$100K", "NZD 100K")
    assert extract_monetary_currency("Currency: NZ$") == "NZD"
