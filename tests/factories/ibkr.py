from __future__ import annotations

from datetime import datetime, timedelta

from src.ibkr.models import (
    AnalysisRecord,
    NormalizedPosition,
    PortfolioSummary,
    TradeBlockData,
)
from src.ibkr.ticker import Ticker


def make_position(
    ticker: str = "7203.T",
    quantity: float = 100,
    avg_cost: float = 2000,
    current_price: float = 2100,
    market_value_usd: float = 1400,
    currency: str = "JPY",
    conid: int = 123456,
) -> NormalizedPosition:
    return NormalizedPosition(
        conid=conid,
        ticker=Ticker.from_yf(ticker, currency=currency),
        quantity=quantity,
        avg_cost_local=avg_cost,
        market_value_usd=market_value_usd,
        currency=currency,
        current_price_local=current_price,
    )


def make_analysis(
    ticker: str = "7203.T",
    verdict: str = "BUY",
    age_days: int = 5,
    entry_price: float = 2100.0,
    stop_price: float = 1900.0,
    target_1: float = 2500.0,
    target_2: float = 3000.0,
    conviction: str = "Medium",
    size_pct: float = 5.0,
    current_price: float = 2100.0,
) -> AnalysisRecord:
    analysis_date = (datetime.now() - timedelta(days=age_days)).strftime("%Y-%m-%d")
    return AnalysisRecord(
        ticker=ticker,
        analysis_date=analysis_date,
        verdict=verdict,
        current_price=current_price,
        entry_price=entry_price,
        stop_price=stop_price,
        target_1_price=target_1,
        target_2_price=target_2,
        conviction=conviction,
        currency="JPY",
        trade_block=TradeBlockData(
            action=verdict,
            size_pct=size_pct,
            conviction=conviction,
            entry_price=entry_price,
            stop_price=stop_price,
            target_1_price=target_1,
            target_2_price=target_2,
        ),
    )


def make_portfolio(
    value: float = 100000,
    cash: float = 15000,
    cash_buffer_pct: float = 0.05,
) -> PortfolioSummary:
    return PortfolioSummary(
        account_id="U1234567",
        portfolio_value_usd=value,
        cash_balance_usd=cash,
        cash_pct=cash / value if value > 0 else 0,
        available_cash_usd=cash - (value * cash_buffer_pct),
    )
