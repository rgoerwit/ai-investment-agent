"""
Pydantic models for IBKR portfolio management.

Typed data contracts for positions, analysis records, and reconciliation items.

## Ticker Format Convention
----------------------------------------------------------------------
Two distinct ticker formats flow through this system:

  yfinance format  → has an exchange suffix: "7203.T", "0005.HK", "ASML.AS", "GAMA.L"
                     HK codes are zero-padded to 4 digits: "0005", not "5".
                     This is the CANONICAL format used throughout the analysis system,
                     analysis file names, and all dict keys.

  IBKR format      → exchange suffix-free, as IBKR stores it internally:
                     "7203", "5", "ASML", "GAMA".
                     HK codes are NOT zero-padded.
                     Only appears in raw IBKR API responses.

IBKR-layer objects (NormalizedPosition, ReconciliationItem) carry a
`ticker: Ticker` value object (src/ibkr/ticker.py) that derives both
representations on demand:
  ticker.yf    → yfinance format — used for analysis dict lookups, run commands
  ticker.ibkr  → IBKR raw symbol — used for display in format_report()

Outside the IBKR layer (analysis pipeline, AnalysisRecord, analyses dicts)
plain yfinance strings are used exclusively.  The boundary crossing is
always via ticker.yf.

Backward-compatible properties:
  NormalizedPosition.yf_ticker → ticker.yf
  NormalizedPosition.symbol    → ticker.ibkr
  NormalizedPosition.exchange  → ticker.exchange
  ReconciliationItem.ibkr_symbol → ticker.ibkr

## Currency Convention
----------------------------------------------------------------------
All monetary fields follow a strict naming convention to prevent
cross-currency arithmetic bugs:

  _local  suffix → value is in the stock's TRADING currency
                   (JPY for 7203.T, HKD for 0005.HK, GBX for .L, etc.)
  _usd    suffix → value has been converted to USD (using fx_rate_to_usd)
  (no suffix)    → either a ratio / percentage, or clearly context-only

Fields without a suffix that store prices (e.g. entry_price, stop_price)
are ALWAYS in LOCAL currency.  The `currency` field on AnalysisRecord
records which local currency that is.

The single source of truth for the FX conversion rate is
`AnalysisRecord.fx_rate_to_usd` (saved at analysis time).  When it is
missing, `src.ibkr.reconciliation_rules._resolve_fx()` applies a hardcoded
fallback from `src.fx_normalization.FALLBACK_RATES_TO_USD`, logging a warning.

## GBX (British pence) note
----------------------------------------------------------------------
IBKR reports London Stock Exchange prices in GBP (pounds).  The
portfolio reader (`src/ibkr/portfolio.py`) multiplies them by 100 so
that `current_price_local` and `avg_cost_local` on NormalizedPosition
are in GBX (pence), matching the prices yfinance and the Trader agent
emit.  After the conversion `NormalizedPosition.currency` is set to
"GBX", not "GBP".  `market_value_usd` is computed from IBKR's
GBP-denominated `mktValue` (before the ×100) so it is correct.
----------------------------------------------------------------------
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.ibkr.ticker import Ticker


class NormalizedPosition(BaseModel):
    """A normalized IBKR portfolio position.

    Denomination contract
    ─────────────────────
    • avg_cost_local     – average cost in LOCAL currency (GBX for .L stocks)
    • current_price_local – latest price in LOCAL currency
    • market_value_usd   – position value in USD (converted via IBKR/fallback FX)
    • unrealized_pnl_usd – running P&L in USD
    • currency           – ISO code for the LOCAL currency ("GBX" not "GBP" for .L)

    The `ticker` field carries symbol, exchange, and currency together.
    Backward-compatible properties (yf_ticker, symbol, exchange) are preserved.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    conid: int
    ticker: Ticker
    quantity: float  # Can be negative (short)
    avg_cost_local: float = 0.0  # LOCAL currency — e.g. JPY, GBX, HKD
    market_value_usd: float = 0.0  # USD (FX-converted)
    unrealized_pnl_usd: float = 0.0  # USD (FX-converted)
    currency: str = "USD"  # ISO code for the LOCAL currency above
    current_price_local: float = 0.0  # LOCAL currency

    @field_validator("ticker", mode="before")
    @classmethod
    def _parse_ticker(cls, v: object) -> Ticker:
        if isinstance(v, str):
            return Ticker.from_yf(v)
        return v  # type: ignore[return-value]

    # Backward-compatible properties so existing call sites keep working
    @property
    def yf_ticker(self) -> str:
        return self.ticker.yf

    @property
    def symbol(self) -> str:
        return self.ticker.ibkr

    @property
    def exchange(self) -> str:
        return self.ticker.exchange


class TradeBlockData(BaseModel):
    """Parsed TRADE_BLOCK fields from an analysis.

    All price fields are in LOCAL currency (the stock's trading currency).
    """

    action: str = ""  # BUY/SELL/HOLD/REJECT
    size_pct: float = 0.0
    conviction: str = ""  # High/Medium/Low
    entry_price: float | None = None  # LOCAL currency
    stop_price: float | None = None  # LOCAL currency
    target_1_price: float | None = None  # LOCAL currency
    target_2_price: float | None = None  # LOCAL currency
    risk_reward: str = ""
    special: str = ""


class AnalysisRecord(BaseModel):
    """A loaded analysis record from a results JSON file.

    Denomination contract
    ─────────────────────
    All price fields (entry_price, stop_price, target_*_price, current_price)
    are in LOCAL currency — i.e. the currency recorded in the `currency` field.
    Example: a 7203.T record with currency="JPY" has entry_price in JPY.

    `fx_rate_to_usd` converts LOCAL → USD.  It is saved at analysis time from
    a yfinance FX fetch.  When it is None (e.g. older snapshots, offline
    analyses), `src.ibkr.reconciliation_rules._resolve_fx()` supplies a hardcoded fallback.

    Do NOT compare price fields from two different AnalysisRecords without
    first checking that their currencies match.
    """

    ticker: str
    analysis_date: str  # YYYY-MM-DD
    file_path: str = ""
    verdict: str = ""  # BUY/SELL/HOLD/DO_NOT_INITIATE
    health_adj: float | None = None
    growth_adj: float | None = None
    zone: str = ""  # HIGH/MODERATE/LOW
    position_size: float | None = None
    current_price: float | None = None  # LOCAL currency (see class docstring)
    currency: str = "USD"  # ISO code for all price fields above
    fx_rate_to_usd: float | None = None  # LOCAL → USD conversion rate at analysis time
    trade_block: TradeBlockData = Field(default_factory=TradeBlockData)
    # Snapshot fields (may be missing in older analyses)
    entry_price: float | None = None  # LOCAL currency
    stop_price: float | None = None  # LOCAL currency
    target_1_price: float | None = None  # LOCAL currency
    target_2_price: float | None = None  # LOCAL currency
    conviction: str = ""
    sector: str = ""  # GICS sector (e.g. "Industrials"), if available in snapshot
    exchange: str = ""  # Exchange suffix (e.g. "HK", "T"), inferred from ticker
    is_quick_mode: bool = False  # True if analysis was run with --quick (less thorough)

    @property
    def age_days(self) -> int:
        """Days since analysis."""
        try:
            analysis_dt = datetime.strptime(self.analysis_date, "%Y-%m-%d")
            return (datetime.now() - analysis_dt).days
        except (ValueError, TypeError):
            return 9999


class ReconciliationItem(BaseModel):
    """A single reconciliation item comparing IBKR position vs evaluator recommendation.

    Denomination contract
    ─────────────────────
    • suggested_price    – limit-order price in LOCAL currency (same as ibkr_position.currency
                           or analysis.currency).  Passed directly to IBKR as the order price.
    • suggested_quantity – number of shares (dimensionless)
    • cash_impact_usd    – estimated cash change in USD (negative = cost, positive = proceeds).
                           Already FX-converted using analysis.fx_rate_to_usd at reconcile time.

    Ticker fields
    ─────────────
    • ticker      – Ticker value object; .yf gives yfinance format ("MEGP.L", "7203.T") —
                    used for analysis lookup and run commands.  .ibkr gives IBKR raw format
                    ("MEGP", "7203") — used for display in format_report().
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    ticker: Ticker
    action: Literal["BUY", "SELL", "TRIM", "ADD", "HOLD", "REVIEW", "REMOVE"]
    reason: str
    urgency: Literal["HIGH", "MEDIUM", "LOW"]
    ibkr_position: NormalizedPosition | None = None
    analysis: AnalysisRecord | None = None
    suggested_quantity: int | None = None
    suggested_price: float | None = None  # LOCAL currency (see class docstring)
    suggested_order_type: str = "LMT"  # LMT or MKT
    cash_impact_usd: float = 0.0  # USD (negative = cost, positive = proceeds)
    settlement_date: str | None = None  # for sells/trims: "YYYY-MM-DD"
    is_watchlist: bool = False  # True when sourced from IBKR watchlist (zero holdings)
    sell_type: str | None = None  # "STOP_BREACH" | "HARD_REJECT" | "SOFT_REJECT" | None

    @field_validator("ticker", mode="before")
    @classmethod
    def _parse_ticker(cls, v: object) -> Ticker:
        if isinstance(v, str):
            return Ticker.from_yf(v)
        return v  # type: ignore[return-value]

    @property
    def ibkr_symbol(self) -> str:
        """Backward-compatible property: IBKR raw symbol for display."""
        return self.ticker.ibkr


class PortfolioSummary(BaseModel):
    """Summary of the IBKR portfolio for display.

    All monetary fields are in USD (account base currency).
    The IBKR ledger BASE section reports values in the account's base
    currency; this system assumes USD as the account base currency.
    """

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
