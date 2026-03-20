"""Tests for _fetch_market_context() in src/main.py."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


def _make_hist(prices: list[float], dates: list[str] | None = None) -> pd.DataFrame:
    """Build a minimal yfinance-style history DataFrame."""
    if dates is None:
        dates = [f"2026-03-0{i+1}" for i in range(len(prices))]
    idx = pd.DatetimeIndex(dates)
    return pd.DataFrame({"Close": prices}, index=idx)


class TestFetchMarketContext:
    """Tests for the _fetch_market_context helper."""

    @patch("yfinance.Ticker")
    def test_returns_formatted_note_for_japanese_stock(self, mock_ticker_cls):
        """7203.T maps to Nikkei-225; returns formatted down-note."""
        hist = _make_hist([40000.0, 38320.0], ["2026-03-04", "2026-03-05"])
        mock_ticker_cls.return_value.history.return_value = hist

        from src.main import _fetch_market_context

        result = asyncio.run(_fetch_market_context("7203.T", "2026-03-05"))

        assert "MARKET NOTE:" in result
        assert "Nikkei-225" in result
        assert "down" in result
        assert "4.2" in result  # (40000→38320) / 40000 * 100 = 4.2%
        assert "2026-03-05" in result

    @patch("yfinance.Ticker")
    def test_returns_formatted_note_for_hk_stock(self, mock_ticker_cls):
        """0005.HK maps to Hang Seng."""
        hist = _make_hist([20000.0, 20600.0], ["2026-03-04", "2026-03-05"])
        mock_ticker_cls.return_value.history.return_value = hist

        from src.main import _fetch_market_context

        result = asyncio.run(_fetch_market_context("0005.HK", "2026-03-05"))

        assert "Hang Seng" in result
        assert "up" in result
        assert "3.0" in result  # (20600-20000)/20000*100 = 3.0%

    @patch("yfinance.Ticker")
    def test_non_target_exchange_uses_sp500(self, mock_ticker_cls):
        """Ticker without recognized suffix falls back to S&P 500."""
        hist = _make_hist([5000.0, 5050.0], ["2026-03-04", "2026-03-05"])
        mock_ticker_cls.return_value.history.return_value = hist

        from src.main import _fetch_market_context

        result = asyncio.run(_fetch_market_context("AAPL", "2026-03-05"))

        assert "S&P 500" in result
        assert "up" in result

    @patch("yfinance.Ticker")
    def test_graceful_fallback_on_yfinance_error(self, mock_ticker_cls):
        """yfinance failure returns empty string, no exception."""
        mock_ticker_cls.return_value.history.side_effect = RuntimeError("network error")

        from src.main import _fetch_market_context

        result = asyncio.run(_fetch_market_context("7203.T", "2026-03-05"))
        assert result == ""

    @patch("yfinance.Ticker")
    def test_graceful_fallback_on_insufficient_history(self, mock_ticker_cls):
        """Only 1 row of history (no change computable) → empty string."""
        hist = _make_hist([40000.0], ["2026-03-05"])
        mock_ticker_cls.return_value.history.return_value = hist

        from src.main import _fetch_market_context

        result = asyncio.run(_fetch_market_context("7203.T", "2026-03-05"))
        assert result == ""

    @patch("yfinance.Ticker")
    def test_graceful_fallback_on_empty_history(self, mock_ticker_cls):
        """Empty history DataFrame → empty string."""
        hist = pd.DataFrame({"Close": []})
        mock_ticker_cls.return_value.history.return_value = hist

        from src.main import _fetch_market_context

        result = asyncio.run(_fetch_market_context("7203.T", "2026-03-05"))
        assert result == ""

    @patch("yfinance.Ticker")
    def test_taiwan_stock_uses_taiwan_weighted(self, mock_ticker_cls):
        """2330.TW maps to Taiwan Weighted index."""
        hist = _make_hist([19000.0, 18810.0], ["2026-03-04", "2026-03-05"])
        mock_ticker_cls.return_value.history.return_value = hist

        from src.main import _fetch_market_context

        result = asyncio.run(_fetch_market_context("2330.TW", "2026-03-05"))
        assert "Taiwan Weighted" in result

    @patch("yfinance.Ticker")
    def test_european_stock_uses_correct_benchmark(self, mock_ticker_cls):
        """ASML.AS maps to AEX."""
        hist = _make_hist([900.0, 891.0], ["2026-03-04", "2026-03-05"])
        mock_ticker_cls.return_value.history.return_value = hist

        from src.main import _fetch_market_context

        result = asyncio.run(_fetch_market_context("ASML.AS", "2026-03-05"))
        assert "AEX" in result
