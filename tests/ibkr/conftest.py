"""Shared fixtures for IBKR tests."""

from unittest.mock import MagicMock

import pytest

from src.ibkr.models import (
    AnalysisRecord,
    NormalizedPosition,
    PortfolioSummary,
    TradeBlockData,
)


@pytest.fixture
def mock_ibkr_client():
    """Mock IbkrClient that returns sample data."""
    client = MagicMock()
    client._settings = MagicMock()
    client._settings.ibkr_account_id = "U1234567"

    # Sample positions
    client.get_positions.return_value = [
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
        },
        {
            "conid": 789012,
            "contractDesc": "5",
            "listingExchange": "SEHK",
            "position": 400,
            "avgCost": 58.0,
            "mktValue": 2960.0,
            "unrealizedPnl": 120.0,
            "currency": "HKD",
            "mktPrice": 59.0,
        },
    ]

    # Sample ledger
    client.get_ledger.return_value = {
        "BASE": {
            "cashbalance": 18200.0,
            "netliquidationvalue": 125430.0,
        }
    }

    return client


@pytest.fixture
def sample_positions():
    """Sample normalized positions for reconciliation tests."""
    return [
        NormalizedPosition(
            conid=123456,
            yf_ticker="7203.T",
            symbol="7203",
            exchange="TSE",
            quantity=100,
            avg_cost_local=2000.0,
            market_value_usd=1400.0,
            currency="JPY",
            current_price_local=2100.0,
        ),
        NormalizedPosition(
            conid=789012,
            yf_ticker="0005.HK",
            symbol="5",
            exchange="SEHK",
            quantity=400,
            avg_cost_local=58.0,
            market_value_usd=2960.0,
            currency="HKD",
            current_price_local=59.0,
        ),
    ]


@pytest.fixture
def sample_portfolio():
    """Sample portfolio summary."""
    return PortfolioSummary(
        account_id="U1234567",
        portfolio_value_usd=125430.0,
        cash_balance_usd=18200.0,
        cash_pct=14.5,
        position_count=2,
        available_cash_usd=11928.5,
    )


@pytest.fixture
def sample_analysis_buy():
    """Sample BUY analysis record."""
    return AnalysisRecord(
        ticker="7203.T",
        analysis_date="2026-02-17",
        verdict="BUY",
        health_adj=88.0,
        growth_adj=83.0,
        zone="LOW",
        current_price=2100.0,
        currency="JPY",
        entry_price=2100.0,
        stop_price=1900.0,
        target_1_price=2500.0,
        target_2_price=3000.0,
        conviction="Medium",
        trade_block=TradeBlockData(
            action="BUY",
            size_pct=5.0,
            conviction="Medium",
            entry_price=2100.0,
            stop_price=1900.0,
            target_1_price=2500.0,
            target_2_price=3000.0,
        ),
    )
