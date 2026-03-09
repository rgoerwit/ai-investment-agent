"""Tests for portfolio reading and normalization."""

import pytest

from src.ibkr.models import NormalizedPosition, PortfolioSummary
from src.ibkr.portfolio import build_portfolio_summary, normalize_positions
from src.ibkr.ticker import Ticker


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

    def test_jpy_market_value_converted_to_usd(self):
        """JPY mktValue is converted to USD using FALLBACK_RATES_TO_USD (0.0067)."""
        raw = [
            {
                "conid": 1,
                "contractDesc": "7203",
                "listingExchange": "TSE",
                "position": 100,
                "mktValue": 210_000.0,  # ¥210,000
                "currency": "JPY",
            }
        ]
        positions = normalize_positions(raw)
        # ¥210,000 × 0.0067 = $1,407
        assert positions[0].market_value_usd == pytest.approx(1407.0, rel=0.01)

    def test_usd_market_value_unchanged(self):
        """USD positions are not double-converted (rate = 1.0)."""
        raw = [
            {
                "conid": 2,
                "contractDesc": "AAPL",
                "listingExchange": "SMART",
                "position": 10,
                "mktValue": 1800.0,
                "currency": "USD",
            }
        ]
        positions = normalize_positions(raw)
        assert positions[0].market_value_usd == pytest.approx(1800.0)

    def test_unknown_currency_falls_back_to_1x(self):
        """Unknown currency code is treated as 1.0 (no conversion)."""
        raw = [
            {
                "conid": 3,
                "contractDesc": "XYZ",
                "listingExchange": "SMART",
                "position": 1,
                "mktValue": 500.0,
                "currency": "ZZZ",  # fictitious currency
            }
        ]
        positions = normalize_positions(raw)
        assert positions[0].market_value_usd == pytest.approx(500.0)

    def test_lse_price_converted_gbp_to_gbx(self):
        """IBKR reports .L prices in GBP; normalize_positions multiplies by 100 → GBX
        and updates currency field to 'GBX' to reflect the actual denomination."""
        raw = [
            {
                "conid": 101,
                "contractDesc": "GAMA",
                "listingExchange": "LSE",
                "position": 200,
                "mktValue": 1788.0,
                "currency": "GBP",
                "mktPrice": 8.94,  # IBKR: £8.94 GBP
            }
        ]
        positions = normalize_positions(raw)
        assert positions[0].yf_ticker == "GAMA.L"
        # current_price_local must be 894.0 GBX so stop comparisons work correctly
        assert positions[0].current_price_local == pytest.approx(894.0)
        # currency field updated to reflect actual denomination of *_local fields
        assert positions[0].currency == "GBX"

    def test_lse_currency_defaults_to_gbx(self):
        """When IBKR omits currency for a .L ticker, it defaults to 'GBP' initially,
        but normalize_positions converts prices to GBX (pence) and updates currency to 'GBX'."""
        raw = [
            {
                "conid": 102,
                "contractDesc": "KLR",
                "listingExchange": "LSE",
                "position": 50,
                "mktValue": 1101.0,
                # currency field intentionally absent
                "mktPrice": 22.02,
            }
        ]
        positions = normalize_positions(raw)
        assert positions[0].yf_ticker == "KLR.L"
        # After GBP→GBX conversion, currency field reflects actual denomination
        assert positions[0].currency == "GBX"
        # Price must be in pence (×100) to match analysis/yfinance convention
        assert positions[0].current_price_local == pytest.approx(2202.0)

    def test_lse_currency_field_is_gbx_after_normalisation(self):
        """NormalizedPosition.currency must be 'GBX' (not 'GBP') for .L stocks so that
        any code calling _resolve_fx(analysis) or displaying the price symbol gets GBX."""
        raw = [
            {
                "conid": 103,
                "contractDesc": "GAMA",
                "listingExchange": "LSE",
                "position": 100,
                "mktValue": 894.0,
                "currency": "GBP",
                "mktPrice": 8.94,
            }
        ]
        positions = normalize_positions(raw)
        p = positions[0]
        # Currency must reflect the unit of current_price_local (GBX = pence)
        assert p.currency == "GBX", (
            "currency should be GBX after GBP→GBX conversion so downstream "
            "FX lookups use the correct pence rate (0.0127) not the pound rate (1.27)"
        )
        assert p.current_price_local == pytest.approx(894.0)
        # market_value_usd computed from GBP mktValue before ×100 — must NOT be affected
        assert p.market_value_usd == pytest.approx(894.0 * 1.27, rel=0.05)


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
                ticker=Ticker.from_yf("7203.T", currency="JPY"),
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
                conid=1, ticker=Ticker.from_yf("A"), quantity=10, market_value_usd=5000
            ),
            NormalizedPosition(
                conid=2, ticker=Ticker.from_yf("B"), quantity=20, market_value_usd=8000
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
