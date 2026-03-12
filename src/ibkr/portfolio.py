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
from src.ibkr.ticker import Ticker
from src.ibkr.ticker_mapper import (
    _yf_search_ticker,
    cache_conid_mapping,
    ibkr_symbol_to_yf,
    yf_ticker_from_conid,
)

# IBKR exchange codes for US venues — these never need a yfinance suffix search
_US_EXCHANGES: frozenset[str] = frozenset(
    {"NASDAQ", "NYSE", "ARCA", "AMEX", "SMART", "IEXG", "CBOE", ""}
)

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
        # Extract raw IBKR fields
        raw_symbol = (raw.get("contractDesc", "") or raw.get("ticker", "")).strip()
        if "-" in raw_symbol:
            raw_symbol = raw_symbol.split("-")[0]
        raw_exchange = (
            raw.get("listingExchange", "") or raw.get("exchange", "")
        ).strip()
        raw_currency = (raw.get("currency", "") or "").strip()

        if not raw_symbol:
            logger.warning(
                "position_unmapped",
                raw_symbol="(empty)",
                exchange=raw_exchange,
            )
            continue

        # Build Ticker from IBKR fields — this is the authoritative conversion point.
        ticker_obj = Ticker.from_ibkr(raw_symbol, raw_exchange, raw_currency)

        # Network fallback: for non-US positions where the exchange code is unknown
        # (not in IBKR_TO_YFINANCE), attempt a yfinance.Search to resolve the suffix.
        # The network call and result caching live in ticker_mapper._yf_search_ticker.
        if (
            not ticker_obj.has_suffix
            and raw_exchange
            and raw_exchange not in _US_EXCHANGES
        ):
            yf_str = _yf_search_ticker(raw_symbol, raw_exchange, raw_currency)
            if yf_str:
                ticker_obj = Ticker.from_yf(yf_str, currency=raw_currency)

        raw_market_value = float(raw.get("mktValue", 0) or raw.get("marketValue", 0))
        currency = raw_currency or ("GBP" if ticker_obj.suffix == ".L" else "USD")
        fx_rate = FALLBACK_RATES_TO_USD.get(currency.upper(), 1.0)
        market_value_usd = raw_market_value * fx_rate

        # IBKR reports LSE (.L) prices in GBP; yfinance and analysis stop/target
        # prices use GBX (pence). Multiply by 100 so all downstream comparisons
        # (stop breach, target hit, drift, P&L) use consistent GBX units.
        # NOTE: market_value_usd is computed from IBKR's GBP mktValue (before ×100)
        # using the GBP FX rate, so it is correct — do NOT re-apply FX on GBX prices.
        current_price_local = float(raw.get("mktPrice", 0) or raw.get("lastPrice", 0))
        avg_cost_local = float(raw.get("avgCost", 0) or raw.get("avgPrice", 0))
        if ticker_obj.suffix == ".L" and currency.upper() == "GBP":
            current_price_local *= 100
            avg_cost_local *= 100  # GBP → GBX, consistent with analysis/yfinance prices
            currency = "GBX"  # Reflect actual denomination of *_local fields
            # Re-build Ticker so its currency field is "GBX" (used in suffix fallback)
            ticker_obj = Ticker(
                symbol=ticker_obj.symbol,
                exchange=ticker_obj.exchange,
                currency="GBX",
            )

        position = NormalizedPosition(
            conid=raw.get("conid", 0),
            ticker=ticker_obj,
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


def _resolve_watchlist_conid(conid: int, client: IbkrClient | None) -> str:
    """Resolve a watchlist conid to a yfinance ticker.

    Checks the local conid cache first (instant, no API call).  On a miss,
    calls /iserver/contract/{conid}/info via the client, maps to a yfinance
    ticker using the same IBKR→yfinance table as live positions, and caches
    the result so subsequent runs are instant.

    Returns the yfinance ticker string, or "" if resolution fails.
    """
    # Fast path: reverse-lookup in local cache.
    # A bare cached value (no ".") may be a correctly-resolved US ticker OR a
    # previously failed resolution for a non-US stock where the exchange was
    # "SMART" and the currency was ambiguous.  If a client is available, bypass
    # the cache for bare entries so ibkr_symbol_to_yf can try the yfinance
    # search fallback (which is now enabled for SMART + non-USD currency).
    cached = yf_ticker_from_conid(conid)
    if cached and ("." in cached or client is None):
        logger.debug("watchlist_conid_cache_hit", conid=conid, yf_ticker=cached)
        return cached
    if cached:
        logger.debug(
            "watchlist_conid_bare_cache_bypass",
            conid=conid,
            cached=cached,
            reason="retrying to resolve exchange suffix",
        )

    # Slow path: ask IBKR for contract details
    if client is None:
        return ""

    info = client.get_contract_info(conid)
    if not info:
        logger.debug("watchlist_conid_no_info", conid=conid)
        return cached or ""  # fall back to bare cached value if API fails

    symbol = info.get("symbol", "") or info.get("ticker", "")
    exchange = info.get("listingExchange", "") or info.get("exchange", "")
    currency = info.get("currency", "")

    if not symbol:
        logger.debug("watchlist_conid_no_symbol", conid=conid, info=info)
        return cached or ""

    yf_ticker = ibkr_symbol_to_yf(symbol, exchange, currency)
    if yf_ticker:
        cache_conid_mapping(yf_ticker, conid, symbol, exchange)
        logger.info(
            "watchlist_conid_resolved",
            conid=conid,
            symbol=symbol,
            exchange=exchange,
            yf_ticker=yf_ticker,
        )
    return yf_ticker or cached or ""


def read_watchlist(
    client: IbkrClient | None,
    name_hint: str = "",
) -> set[str] | None:
    """
    Read IBKR watchlist and return a set of yfinance tickers.

    IBKR watchlist rows contain only the conid (field "C").  This function
    resolves each conid to a yfinance ticker via the local cache (fast) or
    the /iserver/contract/{conid}/info API (on first encounter), then caches
    the result for subsequent runs.

    Args:
        client: Connected IbkrClient (returns empty set if None)
        name_hint: Case-insensitive substring of the watchlist name to load.
            Empty string (default) → uses the first watchlist found.

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
    skipped = 0
    if rows:
        logger.debug("watchlist_first_row", row=rows[0])
    for row in rows:
        # IBKR watchlist rows: {"C": conid} for securities, {"H": "1"} for blank spacers
        # Some API versions use "conid" instead of "C"
        raw_conid = row.get("C") or row.get("conid") or row.get("conId")
        if not raw_conid:
            continue  # blank spacer row or unexpected format

        try:
            conid = int(raw_conid)
        except (TypeError, ValueError):
            logger.debug("watchlist_bad_conid", raw=raw_conid)
            continue

        yf_ticker = _resolve_watchlist_conid(conid, client)
        if yf_ticker:
            tickers.add(yf_ticker)
        else:
            skipped += 1
            logger.debug("watchlist_row_unresolved", conid=conid)

    logger.info(
        "watchlist_tickers_resolved",
        count=len(tickers),
        skipped=skipped,
        total_rows=len(rows),
    )
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
