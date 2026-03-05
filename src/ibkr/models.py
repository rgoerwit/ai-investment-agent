"""
Pydantic models for IBKR portfolio management.

Typed data contracts for positions, analysis records, and reconciliation items.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class NormalizedPosition(BaseModel):
    """A normalized IBKR portfolio position."""

    conid: int
    yf_ticker: str
    symbol: str = ""
    exchange: str = ""
    quantity: float  # Can be negative (short)
    avg_cost_local: float = 0.0
    market_value_usd: float = 0.0
    unrealized_pnl_usd: float = 0.0
    currency: str = "USD"
    current_price_local: float = 0.0


class TradeBlockData(BaseModel):
    """Parsed TRADE_BLOCK fields from an analysis."""

    action: str = ""  # BUY/SELL/HOLD/REJECT
    size_pct: float = 0.0
    conviction: str = ""  # High/Medium/Low
    entry_price: float | None = None
    stop_price: float | None = None
    target_1_price: float | None = None
    target_2_price: float | None = None
    risk_reward: str = ""
    special: str = ""


class AnalysisRecord(BaseModel):
    """A loaded analysis record from a results JSON file."""

    ticker: str
    analysis_date: str  # YYYY-MM-DD
    file_path: str = ""
    verdict: str = ""  # BUY/SELL/HOLD/DO_NOT_INITIATE
    health_adj: float | None = None
    growth_adj: float | None = None
    zone: str = ""  # HIGH/MODERATE/LOW
    position_size: float | None = None
    current_price: float | None = None
    currency: str = "USD"
    fx_rate_to_usd: float | None = None
    trade_block: TradeBlockData = Field(default_factory=TradeBlockData)
    # Snapshot fields (may be missing in older analyses)
    entry_price: float | None = None
    stop_price: float | None = None
    target_1_price: float | None = None
    target_2_price: float | None = None
    conviction: str = ""
    sector: str = ""  # GICS sector (e.g. "Industrials"), if available in snapshot
    exchange: str = ""  # Exchange suffix (e.g. "HK", "T"), inferred from ticker

    @property
    def age_days(self) -> int:
        """Days since analysis."""
        try:
            analysis_dt = datetime.strptime(self.analysis_date, "%Y-%m-%d")
            return (datetime.now() - analysis_dt).days
        except (ValueError, TypeError):
            return 9999


class ReconciliationItem(BaseModel):
    """A single reconciliation item comparing IBKR position vs evaluator recommendation."""

    ticker: str
    action: Literal["BUY", "SELL", "TRIM", "ADD", "HOLD", "REVIEW"]
    reason: str
    urgency: Literal["HIGH", "MEDIUM", "LOW"]
    ibkr_position: NormalizedPosition | None = None
    analysis: AnalysisRecord | None = None
    suggested_quantity: int | None = None
    suggested_price: float | None = None
    suggested_order_type: str = "LMT"  # LMT or MKT
    cash_impact_usd: float = 0.0  # negative = cost, positive = proceeds
    settlement_date: str | None = None  # for sells/trims: "YYYY-MM-DD"


class PortfolioSummary(BaseModel):
    """Summary of the IBKR portfolio for display."""

    account_id: str = ""
    portfolio_value_usd: float = 0.0
    cash_balance_usd: float = 0.0  # total cash incl. unsettled
    settled_cash_usd: float = 0.0  # T+0 spendable cash
    cash_pct: float = 0.0
    position_count: int = 0
    available_cash_usd: float = 0.0  # settled_cash minus cash_buffer
    # Concentration weights (% of portfolio value) — populated by reconcile()
    sector_weights: dict[str, float] = Field(default_factory=dict)
    exchange_weights: dict[str, float] = Field(default_factory=dict)
