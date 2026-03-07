"""
Portfolio reading and normalization.

Reads raw IBKR positions and converts them to NormalizedPosition models
with yfinance ticker mapping and FX normalization.
"""

from __future__ import annotations

import structlog

from src.fx_normalization import FALLBACK_RATES_TO_USD
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

        currency = raw.get("currency") or ("GBP" if yf_ticker.endswith(".L") else "USD")
        raw_market_value = float(raw.get("mktValue", 0) or raw.get("marketValue", 0))
        fx_rate = FALLBACK_RATES_TO_USD.get(currency.upper(), 1.0)
        market_value_usd = raw_market_value * fx_rate

        # IBKR reports LSE (.L) prices in GBP; yfinance and analysis stop/target
        # prices use GBX (pence). Multiply by 100 so all downstream comparisons
        # (stop breach, target hit, drift, P&L) use consistent GBX units.
        current_price_local = float(raw.get("mktPrice", 0) or raw.get("lastPrice", 0))
        avg_cost_local = float(raw.get("avgCost", 0) or raw.get("avgPrice", 0))
        if yf_ticker.endswith(".L"):
            current_price_local *= 100
            avg_cost_local *= 100  # GBP → GBX, consistent with current_price_local

        position = NormalizedPosition(
            conid=raw.get("conid", 0),
            yf_ticker=yf_ticker,
            symbol=raw.get("contractDesc", ""),
            exchange=raw.get("listingExchange", ""),
            quantity=float(raw.get("position", 0) or raw.get("qty", 0)),
            avg_cost_local=avg_cost_local,
            market_value_usd=market_value_usd,
            unrealized_pnl_usd=float(raw.get("unrealizedPnl", 0)),
            currency=currency,
            current_price_local=current_price_local,
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


def read_watchlist(
    client: IbkrClient | None,
    name_hint: str = "default watchlist",
) -> set[str] | None:
    """
    Read IBKR watchlist and return a set of yfinance tickers.

    Maps each watchlist row through the same symbol→yf-ticker resolver used
    for live positions.  Rows that cannot be mapped are skipped with a debug log.

    Args:
        client: Connected IbkrClient (returns empty set if None)
        name_hint: Passed to client.get_watchlist()

    Returns:
        Set of yfinance ticker strings (e.g. {"0005.HK", "7203.T"}).
        None if the named watchlist was not found (distinct from an empty watchlist).
        Empty set if client is None, watchlist exists but is empty, or API error.
    """
    if client is None:
        return set()

    rows = client.get_watchlist(name_hint)
    if rows is None:
        return None  # watchlist not found
    if not rows:
        return set()  # watchlist found but empty

    tickers: set[str] = set()
    for row in rows:
        # Map watchlist row to the same dict shape that resolve_yf_ticker_from_position
        # understands ("contractDesc" + "listingExchange").
        mapped = {
            "contractDesc": row.get("symbol", ""),
            "listingExchange": row.get("exchange", ""),
            "currency": row.get("currency", ""),
        }
        yf_ticker = resolve_yf_ticker_from_position(mapped)
        if yf_ticker:
            tickers.add(yf_ticker)
        else:
            logger.debug(
                "watchlist_row_unmapped",
                symbol=row.get("symbol", "?"),
                exchange=row.get("exchange", "?"),
            )

    logger.info("watchlist_tickers_resolved", count=len(tickers), total_rows=len(rows))
    return tickers


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

    # IBKR CP API requires portfolio_accounts() to be called before any /portfolio/
    # endpoints to initialise the session for that account. Without it, positions and
    # ledger calls may return empty results. Failure is logged but non-fatal — the
    # subsequent calls may still succeed (e.g. in certain OAuth configurations).
    try:
        client.get_accounts()
    except Exception as e:
        logger.warning("portfolio_accounts_preflight_failed", error=str(e))

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
