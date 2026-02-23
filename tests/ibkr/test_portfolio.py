"""Tests for portfolio reading and normalization."""

import pytest

from src.ibkr.models import NormalizedPosition, PortfolioSummary
from src.ibkr.portfolio import build_portfolio_summary, normalize_positions


class TestNormalizePositions:
    """Test conversion of raw IBKR positions to NormalizedPosition."""

    def test_standard_position(self):
        raw = [
            {
                "conid": 123456,
                "contractDesc": "7203",
                "listingExchange": "TSE",
                "position": 100,
                "avgCost": 2000.0,
                "mktValue": 1400.0,
                "unrealizedPnl": 67.0,
                "currency": "JPY",
                "mktPrice": 2100.0,
            }
        ]
        positions = normalize_positions(raw)
        assert len(positions) == 1
        p = positions[0]
        assert p.yf_ticker == "7203.T"
        assert p.quantity == 100
        assert p.avg_cost_local == 2000.0
        assert p.current_price_local == 2100.0
        assert p.currency == "JPY"

    def test_hk_zero_padding(self):
        raw = [
            {
                "conid": 789,
                "contractDesc": "5",
                "listingExchange": "SEHK",
                "position": 400,
                "avgCost": 58.0,
                "mktValue": 2960.0,
                "currency": "HKD",
                "mktPrice": 59.0,
            }
        ]
        positions = normalize_positions(raw)
        assert positions[0].yf_ticker == "0005.HK"

    def test_empty_symbol_skipped(self):
        raw = [{"conid": 0, "contractDesc": "", "listingExchange": ""}]
        positions = normalize_positions(raw)
        assert len(positions) == 0

    def test_multiple_positions(self):
        raw = [
            {
                "conid": 1,
                "contractDesc": "7203",
                "listingExchange": "TSE",
                "position": 100,
                "mktPrice": 2100,
            },
            {
                "conid": 2,
                "contractDesc": "ASML",
                "listingExchange": "AEB",
                "position": 50,
                "mktPrice": 600,
            },
        ]
        positions = normalize_positions(raw)
        assert len(positions) == 2
        tickers = {p.yf_ticker for p in positions}
        assert "7203.T" in tickers
        assert "ASML.AS" in tickers

    def test_alternative_field_names(self):
        """Test fallback field names (qty, avgPrice, lastPrice)."""
        raw = [
            {
                "conid": 1,
                "contractDesc": "AAPL",
                "listingExchange": "SMART",
                "qty": 25,
                "avgPrice": 150.0,
                "marketValue": 3900.0,
                "lastPrice": 156.0,
            }
        ]
        positions = normalize_positions(raw)
        assert positions[0].quantity == 25
        assert positions[0].avg_cost_local == 150.0
        assert positions[0].current_price_local == 156.0


class TestBuildPortfolioSummary:
    """Test portfolio summary construction."""

    def test_from_base_ledger(self):
        ledger = {
            "BASE": {
                "cashbalance": 18200.0,
                "netliquidationvalue": 125430.0,
            }
        }
        positions = [
            NormalizedPosition(
                conid=1,
                yf_ticker="7203.T",
                quantity=100,
                market_value_usd=14000,
                currency="JPY",
            ),
        ]
        summary = build_portfolio_summary(ledger, positions, "U1234567")
        assert summary.account_id == "U1234567"
        assert summary.portfolio_value_usd == 125430.0
        assert summary.cash_balance_usd == 18200.0
        assert summary.position_count == 1
        # Available cash = 18200 - (125430 * 0.05) = 18200 - 6271.5 = 11928.5
        assert summary.available_cash_usd == pytest.approx(11928.5, rel=0.01)

    def test_fallback_to_positions_sum(self):
        ledger = {}  # No ledger data
        positions = [
            NormalizedPosition(
                conid=1, yf_ticker="A", quantity=10, market_value_usd=5000
            ),
            NormalizedPosition(
                conid=2, yf_ticker="B", quantity=20, market_value_usd=8000
            ),
        ]
        summary = build_portfolio_summary(ledger, positions, "U999")
        assert summary.portfolio_value_usd == 13000.0

    def test_zero_cash_buffer(self):
        ledger = {"BASE": {"cashbalance": 10000, "netliquidationvalue": 100000}}
        summary = build_portfolio_summary(ledger, [], "U1", cash_buffer_pct=0.0)
        assert summary.available_cash_usd == 10000.0

    def test_high_cash_buffer(self):
        ledger = {"BASE": {"cashbalance": 5000, "netliquidationvalue": 100000}}
        summary = build_portfolio_summary(ledger, [], "U1", cash_buffer_pct=0.10)
        # available = 5000 - 10000 = negative → clamped to 0
        assert summary.available_cash_usd == 0.0
