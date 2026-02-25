"""
Portfolio reading and normalization.

Reads raw IBKR positions and converts them to NormalizedPosition models
with yfinance ticker mapping and FX normalization.
"""

from __future__ import annotations

import structlog

from src.ibkr.client import IbkrClient
from src.ibkr.models import NormalizedPosition, PortfolioSummary
from src.ibkr.ticker_mapper import resolve_yf_ticker_from_position

logger = structlog.get_logger(__name__)


def normalize_positions(raw_positions: list[dict]) -> list[NormalizedPosition]:
    """
    Convert raw IBKR position dicts to NormalizedPosition models.

    Maps IBKR symbols to yfinance tickers for reconciliation against
    evaluator analyses.

    Args:
        raw_positions: List of raw IBKR position dicts

    Returns:
        List of NormalizedPosition models (skips positions that can't be mapped)
    """
    positions: list[NormalizedPosition] = []

    for raw in raw_positions:
        yf_ticker = resolve_yf_ticker_from_position(raw)
        if not yf_ticker:
            logger.warning(
                "position_unmapped",
                raw_symbol=raw.get("contractDesc", "?"),
                exchange=raw.get("listingExchange", "?"),
            )
            continue

        position = NormalizedPosition(
            conid=raw.get("conid", 0),
            yf_ticker=yf_ticker,
            symbol=raw.get("contractDesc", ""),
            exchange=raw.get("listingExchange", ""),
            quantity=float(raw.get("position", 0) or raw.get("qty", 0)),
            avg_cost_local=float(raw.get("avgCost", 0) or raw.get("avgPrice", 0)),
            market_value_usd=float(raw.get("mktValue", 0) or raw.get("marketValue", 0)),
            unrealized_pnl_usd=float(raw.get("unrealizedPnl", 0)),
            currency=raw.get("currency", "USD"),
            current_price_local=float(
                raw.get("mktPrice", 0) or raw.get("lastPrice", 0)
            ),
        )
        positions.append(position)

    logger.info(
        "positions_normalized",
        count=len(positions),
        skipped=len(raw_positions) - len(positions),
    )
    return positions


def build_portfolio_summary(
    ledger: dict,
    positions: list[NormalizedPosition],
    account_id: str = "",
    cash_buffer_pct: float = 0.05,
) -> PortfolioSummary:
    """
    Build portfolio summary from IBKR ledger and normalized positions.

    Args:
        ledger: Raw IBKR ledger dict
        positions: Normalized positions
        account_id: IBKR account ID
        cash_buffer_pct: Cash buffer fraction (don't deploy into new BUYs)

    Returns:
        PortfolioSummary model
    """
    # IBKR ledger structure: {"BASE": {"cashbalance": X, "netliquidationvalue": Y, ...}}
    base = ledger.get("BASE", ledger)
    if isinstance(base, dict):
        cash = float(base.get("cashbalance", 0) or base.get("totalcashvalue", 0))
        portfolio_value = float(
            base.get("netliquidationvalue", 0) or base.get("netLiquidation", 0)
        )
        # IBKR ledger BASE section contains "settledcash" as a separate field
        settled_cash = float(
            base.get("settledcash", 0) or base.get("settledBalance", 0)
        )
        if settled_cash <= 0:
            settled_cash = cash  # fallback: if IBKR doesn't separate it, use total cash
    else:
        cash = 0.0
        settled_cash = 0.0
        portfolio_value = sum(p.market_value_usd for p in positions)

    # Fallback portfolio value from positions
    if portfolio_value <= 0:
        portfolio_value = sum(p.market_value_usd for p in positions) + max(cash, 0)

    cash_pct = (cash / portfolio_value * 100) if portfolio_value > 0 else 0.0
    # available_cash derived from settled_cash (not total cash) — only spendable funds
    available_cash = max(0, settled_cash - (portfolio_value * cash_buffer_pct))

    return PortfolioSummary(
        account_id=account_id,
        portfolio_value_usd=portfolio_value,
        cash_balance_usd=cash,
        settled_cash_usd=settled_cash,
        cash_pct=cash_pct,
        position_count=len(positions),
        available_cash_usd=available_cash,
    )


def read_portfolio(
    client: IbkrClient,
    account_id: str | None = None,
    cash_buffer_pct: float = 0.05,
) -> tuple[list[NormalizedPosition], PortfolioSummary]:
    """
    Read and normalize portfolio from IBKR.

    Convenience function that combines position reading, normalization,
    and portfolio summary in one call.

    Args:
        client: Connected IbkrClient
        account_id: IBKR account ID (uses default from settings if None)
        cash_buffer_pct: Cash reserve fraction

    Returns:
        Tuple of (normalized_positions, portfolio_summary)
    """
    acct = account_id or client.account_id

    raw_positions = client.get_positions(acct)
    positions = normalize_positions(raw_positions)

    ledger = client.get_ledger(acct)
    summary = build_portfolio_summary(ledger, positions, acct, cash_buffer_pct)

    logger.info(
        "portfolio_read",
        account=acct,
        positions=summary.position_count,
        value=f"${summary.portfolio_value_usd:,.0f}",
        cash=f"${summary.cash_balance_usd:,.0f}",
        cash_pct=f"{summary.cash_pct:.1f}%",
    )

    return positions, summary
