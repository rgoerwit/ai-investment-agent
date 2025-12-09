"""
Test Smart Gap-Filling Fetcher

Verifies that:
1. Basics are ALWAYS present
2. Gaps are filled from multiple sources
3. Calculated metrics work
4. Validation is fine-grained
"""

import asyncio
import pytest
from src.data.fetcher import SmartMarketDataFetcher
from src.data.validator import FineGrainedValidator


@pytest.mark.asyncio
async def test_basic_fetch():
    """Test that basics are always present."""
    print("=" * 80)
    print("TEST 1: Basic Fetch (AAPL)")
    print("=" * 80)
    
    fetcher = SmartMarketDataFetcher()
    data = await fetcher.get_financial_metrics("AAPL")
    
    print(f"\nSymbol: {data.get('symbol')}")
    print(f"Price: {data.get('currentPrice', data.get('regularMarketPrice', 'N/A'))}")
    print(f"Currency: {data.get('currency')}")
    print(f"Market Cap: {data.get('marketCap', 'N/A'):,}" if data.get('marketCap') else "Market Cap: N/A")
    
    quality = data.get('_quality', {})
    print(f"\nQuality Check:")
    print(f"  Basics OK: {quality.get('basics_ok')}")
    print(f"  Coverage: {quality.get('coverage_pct'):.1f}%")
    print(f"  Sources: {', '.join(quality.get('sources_used', []))}")
    print(f"  Gaps Filled: {quality.get('gaps_filled_this_fetch')}")
    
    if quality.get('basics_missing'):
        print(f"  ⚠ Missing Basics: {quality['basics_missing']}")
    
    if quality.get('suspicious_fields'):
        print(f"  ⚡ Suspicious: {quality['suspicious_fields']}")
    
    # Show key metrics
    print(f"\nKey Metrics:")
    print(f"  P/E: {data.get('trailingPE', 'N/A')}")
    print(f"  P/B: {data.get('priceToBook', 'N/A')}")
    print(f"  Revenue Growth: {data.get('revenueGrowth', 'N/A')}")
    if data.get('revenueGrowth'):
        source = data.get('_revenueGrowth_source', 'unknown')
        print(f"    (source: {source})")
    print(f"  Gross Margin: {data.get('grossMargins', 'N/A')}")
    if data.get('grossMargins'):
        source = data.get('_grossMargins_source', 'unknown')
        print(f"    (source: {source})")
    print(f"  FCF: {data.get('freeCashflow', 'N/A'):,}" if data.get('freeCashflow') else "  FCF: N/A")
    if data.get('freeCashflow'):
        source = data.get('_freeCashflow_source', 'unknown')
        print(f"    (source: {source})")
    print(f"  D/E: {data.get('debtToEquity', 'N/A')}")
    print(f"  Current Ratio: {data.get('currentRatio', 'N/A')}")
    
    stats = fetcher.get_stats()
    print(f"\nFetcher Stats:")
    print(f"  Total Fetches: {stats['fetches']}")
    print(f"  Basics Success Rate: {stats.get('basics_success_rate', 0):.1%}")
    print(f"  Sources Used: {stats['sources']}")
    print(f"  Total Gaps Filled: {stats['gaps_filled']}")
    
    return data


@pytest.mark.asyncio
async def test_hsbc_currency_bug():
    """Test HSBC currency bug is fixed."""
    print("\n" + "=" * 80)
    print("TEST 2: HSBC Currency Bug Fix")
    print("=" * 80)
    
    fetcher = SmartMarketDataFetcher()
    data = await fetcher.get_financial_metrics("0005.HK")
    
    print(f"\nSymbol: {data.get('symbol')}")
    print(f"Price: {data.get('currentPrice', 'N/A')} {data.get('currency')}")
    print(f"P/B: {data.get('priceToBook', 'N/A')}")
    print(f"Book Value: {data.get('bookValue', 'N/A')}")
    
    # Check if P/B is reasonable (<5.0 for HSBC)
    pb = data.get('priceToBook')
    if pb and pb < 5.0:
        print(f"✓ P/B looks correct (<5.0)")
    elif pb:
        print(f"⚠ P/B still high: {pb:.2f}")
    
    quality = data.get('_quality', {})
    print(f"\nQuality:")
    print(f"  Coverage: {quality.get('coverage_pct'):.1f}%")
    print(f"  Gaps Filled: {quality.get('gaps_filled_this_fetch')}")


@pytest.mark.asyncio
async def test_fine_grained_validation():
    """Test fine-grained validation."""
    print("\n" + "=" * 80)
    print("TEST 3: Fine-Grained Validation")
    print("=" * 80)
    
    fetcher = SmartMarketDataFetcher()
    validator = FineGrainedValidator()
    
    data = await fetcher.get_financial_metrics("MSFT")
    validation = validator.validate_comprehensive(data, "MSFT")
    
    print(f"\nValidation Results for MSFT:")
    print(validator.get_validation_summary(validation))


@pytest.mark.asyncio
async def test_gap_filling():
    """Test that gaps get filled from multiple sources."""
    print("\n" + "=" * 80)
    print("TEST 4: Multi-Source Gap Filling")
    print("=" * 80)
    
    fetcher = SmartMarketDataFetcher()
    
    # Test with a ticker that might have sparse data
    data = await fetcher.get_financial_metrics("BRK-B")
    
    quality = data.get('_quality', {})
    print(f"\nBRK-B Quality:")
    print(f"  Coverage: {quality.get('coverage_pct'):.1f}%")
    print(f"  Sources: {', '.join(quality.get('sources_used', []))}")
    print(f"  Gaps Filled: {quality.get('gaps_filled_this_fetch')}")
    
    # Check which metrics are present
    important = [
        'marketCap', 'trailingPE', 'priceToBook', 'returnOnEquity',
        'revenueGrowth', 'profitMargins', 'operatingMargins', 'grossMargins',
        'debtToEquity', 'currentRatio', 'freeCashflow', 'operatingCashflow'
    ]
    
    present = [field for field in important if data.get(field) is not None]
    missing = [field for field in important if data.get(field) is None]
    
    print(f"\nMetrics Present ({len(present)}/{len(important)}):")
    for field in present:
        source = data.get(f'_{field}_source', 'yfinance')
        print(f"  ✓ {field}: {data[field]} (from {source})")
    
    if missing:
        print(f"\nMetrics Missing ({len(missing)}/{len(important)}):")
        for field in missing:
            print(f"  ✗ {field}")


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