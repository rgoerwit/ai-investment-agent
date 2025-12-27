from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from src.data.alpha_vantage_fetcher import AlphaVantageFetcher
from src.data.eodhd_fetcher import EODHDFetcher
from src.data.fmp_fetcher import FMPFetcher
from src.data.interfaces import FinancialFetcher


@pytest.fixture
def mock_aioresponse():
    with patch("aiohttp.ClientSession.get") as mock_get:
        yield mock_get


@pytest.mark.asyncio
async def test_fmp_fetcher_compliance(mock_aioresponse):
    """Test FMPFetcher implements FinancialFetcher and returns standard keys."""
    fetcher = FMPFetcher(api_key="test_key")
    assert isinstance(fetcher, FinancialFetcher)

    # Mock API Response
    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.json.return_value = [
        {"priceToEarningsRatio": 15.5, "returnOnEquity": 0.12, "debtToEquityRatio": 0.5}
    ]
    mock_aioresponse.return_value.__aenter__.return_value = mock_resp

    # Test Metrics
    data = await fetcher.get_financial_metrics("AAPL")
    assert data is not None
    assert data["trailingPE"] == 15.5  # Standardized key
    assert data["returnOnEquity"] == 0.12
    assert data["_source"] == "fmp"

    # Test Price History (Stub)
    hist = await fetcher.get_price_history("AAPL")
    assert isinstance(hist, pd.DataFrame)
    assert hist.empty


@pytest.mark.asyncio
async def test_alpha_vantage_fetcher_compliance(mock_aioresponse):
    """Test AlphaVantageFetcher implements FinancialFetcher."""
    fetcher = AlphaVantageFetcher(api_key="test_key")
    assert isinstance(fetcher, FinancialFetcher)

    # Mock API Response
    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.json.return_value = {
        "Symbol": "IBM",
        "PERatio": "20.5",
        "ReturnOnEquityTTM": "0.15",
    }
    mock_aioresponse.return_value.__aenter__.return_value = mock_resp

    # Test Metrics
    data = await fetcher.get_financial_metrics("IBM")
    assert data is not None
    assert data["trailingPE"] == 20.5
    assert data["returnOnEquity"] == 0.15


@pytest.mark.asyncio
async def test_eodhd_fetcher_compliance(mock_aioresponse):
    """Test EODHDFetcher implements FinancialFetcher."""
    fetcher = EODHDFetcher(api_key="test_key")
    assert isinstance(fetcher, FinancialFetcher)

    # Mock API Response
    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.json.return_value = {
        "Highlights": {"PERatio": 10.0, "ReturnOnEquityTTM": 0.08}
    }
    mock_aioresponse.return_value.__aenter__.return_value = mock_resp

    # Test Metrics
    data = await fetcher.get_financial_metrics("TSLA.US")
    assert data is not None
    assert data["trailingPE"] == 10.0
    assert data["returnOnEquity"] == 0.08
