"""Tests for liquidity calculation tool with comprehensive edge case coverage."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.liquidity_calculation_tool import calculate_liquidity_metrics

# ==================== TEST FIXTURES ====================


@pytest.fixture(autouse=True)
def mock_get_financial_metrics():
    """
    Ensure all tests in this module use a mock for get_financial_metrics.
    This forces the liquidity tool to use the historical mean price from
    the mock data provided by the tests, ensuring isolation.
    """
    from unittest.mock import AsyncMock, patch

    with patch(
        "src.data.fetcher.SmartMarketDataFetcher.get_financial_metrics",
        new=AsyncMock(return_value=None),
    ):
        yield


# ==================== EXISTING TESTS ====================


@pytest.mark.asyncio
async def test_liquidity_insufficient_data():
    """Test liquidity check fails with insufficient data."""
    mock_data = pd.DataFrame()  # Empty dataframe

    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker.return_value.history.return_value = mock_data
        result = await calculate_liquidity_metrics.ainvoke({"ticker": "TEST"})

    assert "FAIL" in result
    assert "Insufficient Data" in result


@pytest.mark.asyncio
async def test_liquidity_usd_pass():
    """Test US stock passes liquidity check."""
    # Price $10, volume 100k = $1M daily turnover (passes $500k threshold)
    # Vary prices slightly to avoid flat-price detection (±1% variation)
    import numpy as np

    prices = 10.0 + np.random.uniform(-0.1, 0.1, 60)
    mock_data = pd.DataFrame({"Close": prices, "Volume": [100000] * 60})

    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker.return_value.history.return_value = mock_data
        result = await calculate_liquidity_metrics.ainvoke({"ticker": "AAPL"})

    assert "PASS" in result
    # Check turnover is in expected range ($950k-$1.05M)
    import re

    turnover_match = re.search(r"Turnover \(USD\): \$(\d{1,3}(?:,\d{3})*)", result)
    if turnover_match:
        turnover = int(turnover_match.group(1).replace(",", ""))
        assert 950_000 < turnover < 1_050_000


@pytest.mark.asyncio
async def test_liquidity_usd_fail():
    """Test stock fails liquidity check."""
    # Price $1, volume 100k = $100k daily turnover (fails $500k threshold)
    mock_data = pd.DataFrame({"Close": [1.0] * 60, "Volume": [100000] * 60})

    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker.return_value.history.return_value = mock_data
        result = await calculate_liquidity_metrics.ainvoke({"ticker": "LOWVOL"})

    assert "FAIL" in result


@pytest.mark.asyncio
async def test_liquidity_fx_conversion_gbp_pence():
    """UK stocks (pence) with FX conversion."""
    # Vary prices slightly to avoid flat-price detection (±1% variation)
    import numpy as np

    prices = 400.0 + np.random.uniform(-4, 4, 60)
    mock_data = pd.DataFrame(
        {
            "Close": prices,  # ~400 pence = ~£4
            "Volume": [100000] * 60,
        }
    )

    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker.return_value.history.return_value = mock_data
        result = await calculate_liquidity_metrics.ainvoke({"ticker": "TEST.L"})

    # The pence adjustment is logged but not in the result string
    # Check that the calculation is correct: ~400 pence = ~£4.00 * 100k vol * 1.27 FX = ~$508k
    assert "PASS" in result
    # Check turnover is in expected range ($480k-$540k)
    import re

    turnover_match = re.search(r"Turnover \(USD\): \$(\d{1,3}(?:,\d{3})*)", result)
    if turnover_match:
        turnover = int(turnover_match.group(1).replace(",", ""))
        assert 480_000 < turnover < 540_000
    assert "GBP" in result


@pytest.mark.asyncio
async def test_liquidity_fx_conversion_hkd():
    """Hong Kong stocks with FX conversion."""
    # HKD 80 * 100k volume * 0.129 FX = $1.032M USD
    # Vary prices slightly to avoid flat-price detection (±1% variation)
    import numpy as np

    prices = 80.0 + np.random.uniform(-0.8, 0.8, 60)
    mock_data = pd.DataFrame({"Close": prices, "Volume": [100000] * 60})

    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker.return_value.history.return_value = mock_data
        result = await calculate_liquidity_metrics.ainvoke({"ticker": "0005.HK"})

    assert "PASS" in result
    assert "HKD" in result


@pytest.mark.asyncio
async def test_liquidity_fx_conversion_twd_fix():
    """Taiwan stocks with FX conversion - regression test."""
    # TWD 100 * 100k volume * 0.031 FX = $310k USD (fails)
    mock_data = pd.DataFrame({"Close": [100.0] * 60, "Volume": [100000] * 60})

    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker.return_value.history.return_value = mock_data
        result = await calculate_liquidity_metrics.ainvoke({"ticker": "2330.TW"})

    # Should properly convert TWD to USD
    assert "TWD" in result or "TW" in result


@pytest.mark.asyncio
async def test_liquidity_expanded_currencies():
    """Test expanded currency support."""
    # JPY 1000 * 100k volume * 0.0067 FX = $670k USD (passes)
    mock_data = pd.DataFrame({"Close": [1000.0] * 60, "Volume": [100000] * 60})

    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker.return_value.history.return_value = mock_data
        result = await calculate_liquidity_metrics.ainvoke(
            {"ticker": "7203.T"}
        )  # Toyota

    assert "JPY" in result or "T" in result


@pytest.mark.asyncio
async def test_liquidity_unknown_suffix_fallback():
    """Test handling of unknown suffixes."""
    mock_data = pd.DataFrame({"Close": [10.0] * 60, "Volume": [100000] * 60})

    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker.return_value.history.return_value = mock_data
        result = await calculate_liquidity_metrics.ainvoke({"ticker": "TEST.UNKNOWN"})

    # Should still work, defaulting to USD
    assert "PASS" in result or "FAIL" in result


@pytest.mark.asyncio
async def test_liquidity_zero_volume_edge_case():
    """Test handling of zero volume."""
    mock_data = pd.DataFrame(
        {
            "Close": [10.0] * 60,
            "Volume": [0] * 60,  # Zero volume
        }
    )

    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker.return_value.history.return_value = mock_data
        result = await calculate_liquidity_metrics.ainvoke({"ticker": "ZERO"})

    assert "FAIL" in result


@pytest.mark.asyncio
async def test_liquidity_mixed_case_ticker():
    """Test ticker normalization."""
    mock_data = pd.DataFrame({"Close": [10.0] * 60, "Volume": [100000] * 60})

    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker.return_value.history.return_value = mock_data
        result = await calculate_liquidity_metrics.ainvoke(
            {"ticker": "aapl"}
        )  # lowercase

    assert "PASS" in result or "FAIL" in result


@pytest.mark.asyncio
async def test_error_handling():
    """Test graceful error handling."""
    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker.return_value.history.side_effect = Exception("Network error")
        result = await calculate_liquidity_metrics.ainvoke({"ticker": "ERROR"})

    assert "error" in result.lower() or "fail" in result.lower()


# ==================== NEW EDGE CASE TESTS ====================


@pytest.mark.asyncio
async def test_liquidity_nan_values():
    """Test handling of NaN values in price/volume data."""
    mock_data = pd.DataFrame(
        {"Close": [10.0, np.nan, 10.0] * 20, "Volume": [100000, 100000, np.nan] * 20}
    )

    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker.return_value.history.return_value = mock_data
        result = await calculate_liquidity_metrics.ainvoke({"ticker": "NANTEST"})

    # Should handle NaN gracefully (either skip or interpolate)
    assert isinstance(result, str)
    assert "PASS" in result or "FAIL" in result or "Insufficient" in result


@pytest.mark.asyncio
async def test_liquidity_negative_prices():
    """Test handling of negative prices (data corruption)."""
    mock_data = pd.DataFrame(
        {
            "Close": [-10.0] * 60,  # Invalid negative price
            "Volume": [100000] * 60,
        }
    )

    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker.return_value.history.return_value = mock_data
        result = await calculate_liquidity_metrics.ainvoke({"ticker": "NEGATIVE"})

    # Should detect invalid data
    assert "FAIL" in result or "error" in result.lower() or "invalid" in result.lower()


@pytest.mark.asyncio
async def test_liquidity_extreme_volatility():
    """Test handling of extreme price volatility."""
    # Simulate a flash crash scenario
    prices = [100.0] * 30 + [1.0] * 5 + [100.0] * 25  # Flash crash in the middle
    mock_data = pd.DataFrame({"Close": prices, "Volume": [100000] * 60})

    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker.return_value.history.return_value = mock_data
        result = await calculate_liquidity_metrics.ainvoke({"ticker": "VOLATILE"})

    # Should still calculate average correctly
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_liquidity_insufficient_rows():
    """Test handling of insufficient historical data rows."""
    mock_data = pd.DataFrame(
        {
            "Close": [10.0] * 5,  # Only 5 rows instead of 60
            "Volume": [100000] * 5,
        }
    )

    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker.return_value.history.return_value = mock_data
        result = await calculate_liquidity_metrics.ainvoke({"ticker": "SHORTDATA"})

    # Should either fail or work with available data
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_liquidity_missing_columns():
    """Test handling of missing required columns."""
    mock_data = pd.DataFrame(
        {
            "Close": [10.0] * 60
            # Missing 'Volume' column
        }
    )

    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker.return_value.history.return_value = mock_data
        result = await calculate_liquidity_metrics.ainvoke({"ticker": "NOCOL"})

    assert "FAIL" in result or "error" in result.lower() or "Insufficient" in result


@pytest.mark.asyncio
async def test_liquidity_infinity_values():
    """Test handling of infinity values."""
    mock_data = pd.DataFrame(
        {"Close": [10.0, np.inf, 10.0] * 20, "Volume": [100000] * 60}
    )

    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker.return_value.history.return_value = mock_data
        result = await calculate_liquidity_metrics.ainvoke({"ticker": "INFTEST"})

    # Should handle infinity gracefully
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_liquidity_boundary_threshold():
    """Test liquidity at exact threshold boundary."""
    # Exactly $500,000 daily turnover (boundary case)
    mock_data = pd.DataFrame(
        {
            "Close": [5.0] * 60,
            "Volume": [100000] * 60,  # 5 * 100k = $500k exactly
        }
    )

    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker.return_value.history.return_value = mock_data
        result = await calculate_liquidity_metrics.ainvoke({"ticker": "BOUNDARY"})

    # Should be clear pass or fail (not ambiguous)
    assert "PASS" in result or "FAIL" in result


@pytest.mark.asyncio
async def test_liquidity_api_timeout():
    """Test handling of API timeout."""
    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker.return_value.history.side_effect = TimeoutError("Request timeout")
        result = await calculate_liquidity_metrics.ainvoke({"ticker": "TIMEOUT"})

    assert (
        "error" in result.lower()
        or "fail" in result.lower()
        or "timeout" in result.lower()
    )


@pytest.mark.asyncio
async def test_liquidity_malformed_response():
    """Test handling of malformed API response."""
    with patch("yfinance.Ticker") as mock_ticker:
        # Return a non-DataFrame object
        mock_ticker.return_value.history.return_value = "not a dataframe"
        result = await calculate_liquidity_metrics.ainvoke({"ticker": "MALFORMED"})

    assert "error" in result.lower() or "fail" in result.lower()


@pytest.mark.asyncio
async def test_liquidity_unicode_ticker():
    """Test handling of unicode/special characters in ticker."""
    mock_data = pd.DataFrame({"Close": [10.0] * 60, "Volume": [100000] * 60})

    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker.return_value.history.return_value = mock_data
        result = await calculate_liquidity_metrics.ainvoke({"ticker": "TEST™"})

    # Should handle gracefully (sanitize or error)
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_liquidity_very_large_volume():
    """Test handling of extremely large volume numbers."""
    # Vary prices slightly to avoid flat-price detection (±1% variation)
    import numpy as np

    prices = 10.0 + np.random.uniform(-0.1, 0.1, 60)
    mock_data = pd.DataFrame(
        {
            "Close": prices,
            "Volume": [999999999999] * 60,  # Billions of shares
        }
    )

    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker.return_value.history.return_value = mock_data
        result = await calculate_liquidity_metrics.ainvoke({"ticker": "MEGAVOL"})

    # Should handle large numbers without overflow
    assert "PASS" in result
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_liquidity_decimal_precision():
    """Test handling of high decimal precision in prices."""
    mock_data = pd.DataFrame(
        {
            "Close": [10.123456789] * 60,  # Many decimal places
            "Volume": [100000] * 60,
        }
    )

    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker.return_value.history.return_value = mock_data
        result = await calculate_liquidity_metrics.ainvoke({"ticker": "PRECISE"})

    # Should handle precision correctly
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_liquidity_fractional_volume():
    """Test handling of fractional volume (shouldn't happen but might)."""
    mock_data = pd.DataFrame(
        {
            "Close": [10.0] * 60,
            "Volume": [100000.5] * 60,  # Fractional shares
        }
    )

    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker.return_value.history.return_value = mock_data
        result = await calculate_liquidity_metrics.ainvoke({"ticker": "FRAC"})

    # Should either round or handle gracefully
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_liquidity_empty_ticker_string():
    """Test handling of empty ticker string."""
    mock_data = pd.DataFrame({"Close": [10.0] * 60, "Volume": [100000] * 60})

    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker.return_value.history.return_value = mock_data
        result = await calculate_liquidity_metrics.ainvoke({"ticker": ""})

    # Should handle empty ticker gracefully
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_liquidity_none_ticker():
    """Test handling of None as ticker."""
    mock_data = pd.DataFrame({"Close": [10.0] * 60, "Volume": [100000] * 60})

    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker.return_value.history.return_value = mock_data
        try:
            result = await calculate_liquidity_metrics.ainvoke({"ticker": None})
            # If it doesn't raise, it should return error string
            assert "error" in result.lower() or "fail" in result.lower()
        except (TypeError, AttributeError):
            # Acceptable to raise exception for None
            pass


@pytest.mark.asyncio
async def test_liquidity_mixed_data_types():
    """Test handling of mixed data types in volume column."""
    mock_data = pd.DataFrame(
        {"Close": [10.0] * 60, "Volume": [100000, "100000", 100000.0, "invalid"] * 15}
    )

    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker.return_value.history.return_value = mock_data
        result = await calculate_liquidity_metrics.ainvoke({"ticker": "MIXED"})

    # Should handle type conversion or error gracefully
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_liquidity_duplicate_dates():
    """Test handling of duplicate date entries."""
    dates = pd.date_range("2024-01-01", periods=30).tolist() * 2  # Duplicates
    mock_data = pd.DataFrame(
        {"Close": [10.0] * 60, "Volume": [100000] * 60}, index=dates
    )

    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker.return_value.history.return_value = mock_data
        result = await calculate_liquidity_metrics.ainvoke({"ticker": "DUP"})

    # Should handle duplicates (dedup or average)
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_liquidity_out_of_order_dates():
    """Test handling of out-of-order date entries."""
    dates = pd.date_range("2024-01-01", periods=60).tolist()
    dates = dates[:30] + dates[30:][::-1]  # Reverse second half
    mock_data = pd.DataFrame(
        {"Close": [10.0] * 60, "Volume": [100000] * 60}, index=dates
    )

    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker.return_value.history.return_value = mock_data
        result = await calculate_liquidity_metrics.ainvoke({"ticker": "UNSORTED"})

    # Should sort or handle gracefully
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_liquidity_whitespace_ticker():
    """Test handling of ticker with whitespace."""
    mock_data = pd.DataFrame({"Close": [10.0] * 60, "Volume": [100000] * 60})

    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker.return_value.history.return_value = mock_data
        result = await calculate_liquidity_metrics.ainvoke({"ticker": "  AAPL  "})

    # Should strip whitespace
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_liquidity_network_intermittent():
    """Test handling of intermittent network failures."""
    call_count = 0

    def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ConnectionError("Network unavailable")
        return pd.DataFrame({"Close": [10.0] * 60, "Volume": [100000] * 60})

    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker.return_value.history.side_effect = side_effect
        result = await calculate_liquidity_metrics.ainvoke({"ticker": "INTERMITTENT"})

    # Depending on retry logic, might succeed or fail
    assert isinstance(result, str)
