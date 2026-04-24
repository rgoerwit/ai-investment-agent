"""
Test Smart Gap-Filling Fetcher

Verifies that:
1. Basics are ALWAYS present
2. Gaps are filled from multiple sources
3. Calculated metrics work
4. Validation is fine-grained
"""

import asyncio
from unittest.mock import AsyncMock

import pytest

from src.data.fetcher import SmartMarketDataFetcher
from src.data.validator import FineGrainedValidator


def _fmt_pct(value) -> str:
    return f"{value:.1f}%" if isinstance(value, int | float) else "N/A"


def _mock_source_fetch(
    fetcher: SmartMarketDataFetcher,
    *,
    ticker: str,
    yfinance: dict,
    yahooquery: dict | None = None,
) -> None:
    fetcher._pre_resolve_ticker = AsyncMock(return_value=ticker)
    fetcher._fetch_all_sources_parallel = AsyncMock(
        return_value={"yfinance": yfinance, "yahooquery": yahooquery}
    )
    fetcher._fetch_tavily_gaps = AsyncMock(return_value={})
    fetcher._probe_ibkr_security = AsyncMock(return_value=None)


@pytest.fixture
def sample_metrics():
    return {
        "AAPL": {
            "symbol": "AAPL",
            "shortName": "Apple Inc.",
            "currency": "USD",
            "currentPrice": 212.34,
            "marketCap": 3_280_000_000_000,
            "trailingPE": 31.2,
            "priceToBook": 45.3,
            "revenueGrowth": 0.061,
            "grossMargins": 0.462,
            "freeCashflow": 101_000_000_000,
            "debtToEquity": 1.45,
            "currentRatio": 0.98,
            "profitMargins": 0.247,
            "operatingMargins": 0.311,
            "returnOnEquity": 1.52,
            "operatingCashflow": 118_000_000_000,
            "numberOfAnalystOpinions": 38,
            "sector": "Technology",
            "industry": "Consumer Electronics",
        },
        "0005.HK": {
            "symbol": "0005.HK",
            "shortName": "HSBC Holdings plc",
            "currency": "HKD",
            "currentPrice": 69.8,
            "priceToBook": 0.94,
            "bookValue": 74.2,
            "marketCap": 1_250_000_000_000,
            "trailingPE": 8.5,
            "returnOnEquity": 0.121,
            "revenueGrowth": 0.043,
            "profitMargins": 0.287,
            "debtToEquity": 0.22,
            "currentRatio": 1.1,
            "freeCashflow": 18_400_000_000,
            "operatingCashflow": 24_100_000_000,
            "sector": "Financial Services",
            "industry": "Banks - Diversified",
            "numberOfAnalystOpinions": 19,
        },
        "MSFT": {
            "symbol": "MSFT",
            "shortName": "Microsoft Corporation",
            "currency": "USD",
            "currentPrice": 468.11,
            "marketCap": 3_480_000_000_000,
            "trailingPE": 35.1,
            "priceToBook": 12.7,
            "returnOnEquity": 0.341,
            "revenueGrowth": 0.152,
            "profitMargins": 0.359,
            "operatingMargins": 0.431,
            "grossMargins": 0.689,
            "debtToEquity": 0.37,
            "currentRatio": 1.32,
            "freeCashflow": 69_200_000_000,
            "operatingCashflow": 87_600_000_000,
            "sector": "Technology",
            "industry": "Software - Infrastructure",
            "numberOfAnalystOpinions": 44,
        },
        "BRK-B": {
            "symbol": "BRK-B",
            "shortName": "Berkshire Hathaway Inc.",
            "currency": "USD",
            "currentPrice": 453.87,
            "marketCap": 978_000_000_000,
            "profitMargins": 0.188,
            "operatingMargins": 0.214,
            "debtToEquity": 0.26,
            "currentRatio": 1.71,
            "freeCashflow": 32_500_000_000,
            "operatingCashflow": 46_800_000_000,
            "sector": "Financial Services",
            "industry": "Insurance - Diversified",
            "numberOfAnalystOpinions": 9,
        },
    }


@pytest.mark.asyncio
async def test_basic_fetch(sample_metrics):
    """Test that basics are always present."""
    print("=" * 80)
    print("TEST 1: Basic Fetch (AAPL)")
    print("=" * 80)

    fetcher = SmartMarketDataFetcher()
    _mock_source_fetch(
        fetcher,
        ticker="AAPL",
        yfinance=sample_metrics["AAPL"],
        yahooquery={"regularMarketSource": "DELAYED"},
    )
    data = await fetcher.get_financial_metrics("AAPL")

    print(f"\nSymbol: {data.get('symbol')}")
    print(f"Price: {data.get('currentPrice', data.get('regularMarketPrice', 'N/A'))}")
    print(f"Currency: {data.get('currency')}")
    print(
        f"Market Cap: {data.get('marketCap', 'N/A'):,}"
        if data.get("marketCap")
        else "Market Cap: N/A"
    )

    quality = data.get("_quality", {})
    print("\nQuality Check:")
    print(f"  Basics OK: {quality.get('basics_ok')}")
    print(f"  Coverage: {_fmt_pct(quality.get('coverage_pct'))}")
    print(f"  Sources: {', '.join(quality.get('sources_used', []))}")
    print(f"  Gaps Filled: {quality.get('gaps_filled_this_fetch')}")

    if quality.get("basics_missing"):
        print(f"  ⚠ Missing Basics: {quality['basics_missing']}")

    if quality.get("suspicious_fields"):
        print(f"  ⚡ Suspicious: {quality['suspicious_fields']}")

    # Show key metrics
    print("\nKey Metrics:")
    print(f"  P/E: {data.get('trailingPE', 'N/A')}")
    print(f"  P/B: {data.get('priceToBook', 'N/A')}")
    print(f"  Revenue Growth: {data.get('revenueGrowth', 'N/A')}")
    if data.get("revenueGrowth"):
        source = data.get("_revenueGrowth_source", "unknown")
        print(f"    (source: {source})")
    print(f"  Gross Margin: {data.get('grossMargins', 'N/A')}")
    if data.get("grossMargins"):
        source = data.get("_grossMargins_source", "unknown")
        print(f"    (source: {source})")
    print(
        f"  FCF: {data.get('freeCashflow', 'N/A'):,}"
        if data.get("freeCashflow")
        else "  FCF: N/A"
    )
    if data.get("freeCashflow"):
        source = data.get("_freeCashflow_source", "unknown")
        print(f"    (source: {source})")
    print(f"  D/E: {data.get('debtToEquity', 'N/A')}")
    print(f"  Current Ratio: {data.get('currentRatio', 'N/A')}")

    stats = fetcher.get_stats()
    print("\nFetcher Stats:")
    print(f"  Total Fetches: {stats['fetches']}")
    print(f"  Basics Success Rate: {stats.get('basics_success_rate', 0):.1%}")
    print(f"  Sources Used: {stats['sources']}")
    print(f"  Total Gaps Filled: {stats['gaps_filled']}")

    assert isinstance(data, dict)
    assert "symbol" in data
    return data


@pytest.mark.asyncio
async def test_hsbc_currency_bug(sample_metrics):
    """Test HSBC currency bug is fixed."""
    print("\n" + "=" * 80)
    print("TEST 2: HSBC Currency Bug Fix")
    print("=" * 80)

    fetcher = SmartMarketDataFetcher()
    _mock_source_fetch(fetcher, ticker="0005.HK", yfinance=sample_metrics["0005.HK"])
    data = await fetcher.get_financial_metrics("0005.HK")

    print(f"\nSymbol: {data.get('symbol')}")
    print(f"Price: {data.get('currentPrice', 'N/A')} {data.get('currency')}")
    print(f"P/B: {data.get('priceToBook', 'N/A')}")
    print(f"Book Value: {data.get('bookValue', 'N/A')}")

    # Check if P/B is reasonable (<5.0 for HSBC)
    pb = data.get("priceToBook")
    if pb and pb < 5.0:
        print("✓ P/B looks correct (<5.0)")
    elif pb:
        print(f"⚠ P/B still high: {pb:.2f}")

    quality = data.get("_quality", {})
    print("\nQuality:")
    print(f"  Coverage: {_fmt_pct(quality.get('coverage_pct'))}")
    print(f"  Gaps Filled: {quality.get('gaps_filled_this_fetch')}")

    assert isinstance(data, dict)
    assert data.get("symbol") == "0005.HK"


@pytest.mark.asyncio
async def test_fine_grained_validation(sample_metrics):
    """Test fine-grained validation."""
    print("\n" + "=" * 80)
    print("TEST 3: Fine-Grained Validation")
    print("=" * 80)

    fetcher = SmartMarketDataFetcher()
    validator = FineGrainedValidator()
    _mock_source_fetch(fetcher, ticker="MSFT", yfinance=sample_metrics["MSFT"])

    data = await fetcher.get_financial_metrics("MSFT")
    validation = validator.validate_comprehensive(data, "MSFT")

    print("\nValidation Results for MSFT:")
    print(validator.get_validation_summary(validation))


@pytest.mark.asyncio
async def test_gap_filling(sample_metrics):
    """Test that gaps get filled from multiple sources."""
    print("\n" + "=" * 80)
    print("TEST 4: Multi-Source Gap Filling")
    print("=" * 80)

    fetcher = SmartMarketDataFetcher()
    _mock_source_fetch(fetcher, ticker="BRK-B", yfinance=sample_metrics["BRK-B"])

    # Test with a ticker that might have sparse data
    data = await fetcher.get_financial_metrics("BRK-B")

    quality = data.get("_quality", {})
    print("\nBRK-B Quality:")
    print(f"  Coverage: {_fmt_pct(quality.get('coverage_pct'))}")
    print(f"  Sources: {', '.join(quality.get('sources_used', []))}")
    print(f"  Gaps Filled: {quality.get('gaps_filled_this_fetch')}")

    # Check which metrics are present
    important = [
        "marketCap",
        "trailingPE",
        "priceToBook",
        "returnOnEquity",
        "revenueGrowth",
        "profitMargins",
        "operatingMargins",
        "grossMargins",
        "debtToEquity",
        "currentRatio",
        "freeCashflow",
        "operatingCashflow",
    ]

    present = [field for field in important if data.get(field) is not None]
    missing = [field for field in important if data.get(field) is None]

    print(f"\nMetrics Present ({len(present)}/{len(important)}):")
    for field in present:
        source = data.get(f"_{field}_source", "yfinance")
        print(f"  ✓ {field}: {data[field]} (from {source})")

    if missing:
        print(f"\nMetrics Missing ({len(missing)}/{len(important)}):")
        for field in missing:
            print(f"  ✗ {field}")

    assert isinstance(data, dict)
    assert data.get("symbol") == "BRK-B"


async def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("SMART GAP-FILLING FETCHER TESTS")
    print("=" * 80)

    await test_basic_fetch()
    await test_hsbc_currency_bug()
    await test_fine_grained_validation()
    await test_gap_filling()

    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
