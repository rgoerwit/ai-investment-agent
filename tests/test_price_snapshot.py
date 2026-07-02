"""Tests for _fetch_price_snapshot() in src/main.py and the news-analyst
drawdown-context injection built from it."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd


def _make_hist(closes: list[float]) -> pd.DataFrame:
    dates = pd.date_range("2025-07-01", periods=len(closes), freq="B")
    return pd.DataFrame(
        {
            "Close": closes,
            "High": [c * 1.02 for c in closes],
            "Low": [c * 0.98 for c in closes],
        },
        index=dates,
    )


class TestFetchPriceSnapshot:
    @patch("yfinance.Ticker")
    def test_returns_snapshot_with_sma_and_52wk_fields(self, mock_ticker_cls):
        closes = [10.0] * 200 + [8.0] * 50
        mock_ticker_cls.return_value.history.return_value = _make_hist(closes)

        from src.main import _fetch_price_snapshot

        snapshot = asyncio.run(_fetch_price_snapshot("6831.HK"))

        assert snapshot is not None
        assert snapshot["current"] == 8.0
        assert snapshot["high_52w"] == 10.0 * 1.02
        assert snapshot["low_52w"] == 8.0 * 0.98
        assert snapshot["sma50"] == 8.0
        assert 8.0 < snapshot["sma200"] < 10.0

    @patch("yfinance.Ticker")
    def test_short_history_falls_back_to_full_window_sma200(self, mock_ticker_cls):
        closes = [10.0] * 30
        mock_ticker_cls.return_value.history.return_value = _make_hist(closes)

        from src.main import _fetch_price_snapshot

        snapshot = asyncio.run(_fetch_price_snapshot("NEW.T"))

        assert snapshot is not None
        assert snapshot["sma200"] == 10.0

    @patch("yfinance.Ticker")
    def test_too_few_closes_returns_none(self, mock_ticker_cls):
        mock_ticker_cls.return_value.history.return_value = _make_hist([10.0] * 5)

        from src.main import _fetch_price_snapshot

        assert asyncio.run(_fetch_price_snapshot("THIN.T")) is None

    @patch("yfinance.Ticker")
    def test_fetch_error_returns_none(self, mock_ticker_cls):
        mock_ticker_cls.return_value.history.side_effect = RuntimeError("boom")

        from src.main import _fetch_price_snapshot

        assert asyncio.run(_fetch_price_snapshot("ERR.T")) is None


class TestNewsDrawdownContext:
    @staticmethod
    def _context(**snapshot: float):
        return SimpleNamespace(price_snapshot=snapshot or None)

    def test_triggered_below_52wk_ratio(self):
        from src.agents.analyst_nodes import _build_news_price_drawdown_context

        block = _build_news_price_drawdown_context(
            "6831.HK",
            self._context(current=5.60, high_52w=9.81, sma200=6.47),
        )

        assert "### PRICE DRAWDOWN CONTEXT" in block
        assert "43% below" in block
        assert "PRICE DRAWDOWN PROTOCOL" in block

    def test_triggered_below_sma200_floor_even_above_52wk_ratio(self):
        from src.agents.analyst_nodes import _build_news_price_drawdown_context

        block = _build_news_price_drawdown_context(
            "X.T",
            self._context(current=7.0, high_52w=10.0, sma200=9.5),
        )

        assert "### PRICE DRAWDOWN CONTEXT" in block

    def test_not_triggered_when_healthy(self):
        from src.agents.analyst_nodes import _build_news_price_drawdown_context

        assert (
            _build_news_price_drawdown_context(
                "OK.T",
                self._context(current=9.0, high_52w=10.0, sma200=8.5),
            )
            == ""
        )

    def test_missing_snapshot_or_fields_yields_empty(self):
        from src.agents.analyst_nodes import _build_news_price_drawdown_context

        assert _build_news_price_drawdown_context("A.T", self._context()) == ""
        assert _build_news_price_drawdown_context("B.T", None) == ""
        assert (
            _build_news_price_drawdown_context("C.T", SimpleNamespace()) == ""
        )  # older contexts without the attribute

    def test_zero_or_negative_high_no_zerodivision(self):
        from src.agents.analyst_nodes import _build_news_price_drawdown_context

        assert (
            _build_news_price_drawdown_context(
                "Z.T", self._context(current=5.0, high_52w=0.0)
            )
            == ""
        )
