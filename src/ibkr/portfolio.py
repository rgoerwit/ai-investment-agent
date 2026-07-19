"""
Portfolio reading and normalization.

Reads raw IBKR positions and converts them to NormalizedPosition models
with yfinance ticker mapping and FX normalization.
"""

from __future__ import annotations

import structlog

from src.error_safety import summarize_exception
from src.fx_normalization import FALLBACK_RATES_TO_USD
from src.ibkr.client import IbkrClient, mask_account
from src.ibkr.exceptions import IBKRError
from src.ibkr.models import NormalizedPosition, PortfolioSummary
from src.ibkr.portfolio_defaults import DEFAULT_CASH_BUFFER_PCT
from src.ibkr.ticker import Ticker
from src.ibkr.ticker_mapper import (
    TickerResolution,
    _yf_search_ticker,
    cache_conid_mapping,
    resolve_ibkr_ticker,
    yf_ticker_from_conid,
)
from src.ticker_corrections import apply_operator_override

# IBKR exchange codes for US venues — these never need a yfinance suffix search
_US_EXCHANGES: frozenset[str] = frozenset(
    {"NASDAQ", "NYSE", "ARCA", "AMEX", "SMART", "IEXG", "CBOE", ""}
)
_MULTI_EXCHANGE_CURRENCIES: frozenset[str] = frozenset(
    {"TWD", "KRW", "INR", "CNY", "CAD"}
)

logger = structlog.get_logger(__name__)


def normalize_positions(
    raw_positions: list[dict],
    *,
    client: IbkrClient | None = None,
) -> list[NormalizedPosition]:
    """
    Convert raw IBKR position dicts to NormalizedPosition models.

    Maps IBKR symbols to yfinance tickers for reconciliation against
    evaluator analyses.

    Args:
        raw_positions: List of raw IBKR position dicts
        client: Connected IBKR client. When available, held positions in
            ambiguous multi-exchange markets are resolved from their conid.

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
        smart_non_usd = raw_exchange.upper() in {
            "",
            "SMART",
        } and raw_currency.upper() not in {
            "",
            "USD",
        }
        ticker_identity_verified = ticker_obj.exchange_resolved and not smart_non_usd
        if ticker_identity_verified:
            ticker_resolution_source = "exchange_map"
        elif ticker_obj.has_suffix:
            ticker_resolution_source = "currency_fallback"
        else:
            ticker_resolution_source = "unresolved"

        conid = _parse_conid(raw.get("conid"))
        if (
            conid is not None
            and client is not None
            and _should_resolve_position_conid(
                ticker_obj,
                raw_exchange=raw_exchange,
                raw_currency=raw_currency,
            )
        ):
            resolution = _resolve_conid_ticker(
                conid,
                client,
                force_live=True,
                context="position",
            )
            if resolution.yf_ticker:
                ticker_obj = Ticker.from_yf(
                    resolution.yf_ticker,
                    currency=raw_currency,
                )
                ticker_identity_verified = resolution.exchange_verified
                ticker_resolution_source = resolution.source

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
                ticker_identity_verified = False
                ticker_resolution_source = "yfinance_search"

        # Operator-confirmed listing migrations (config/ticker_overrides.json):
        # keep position keys aligned with the analysis side until IBKR's own
        # exchange metadata catches up with the move.
        overridden_yf, was_overridden = apply_operator_override(ticker_obj.yf)
        if was_overridden:
            ticker_obj = Ticker.from_yf(overridden_yf, currency=raw_currency)
            ticker_identity_verified = True
            ticker_resolution_source = "operator_override"

        raw_market_value = float(raw.get("mktValue", 0) or raw.get("marketValue", 0))
        currency = raw_currency or ("GBP" if ticker_obj.suffix == ".L" else "USD")
        fx_rate = FALLBACK_RATES_TO_USD.get(currency.upper(), 1.0)
        market_value_usd = raw_market_value * fx_rate

        # IBKR reports LSE (.L) prices in GBP; yfinance and saved downside/base-case
        # reference prices use GBX (pence). Multiply by 100 so review-level,
        # valuation-reference, drift, and P&L comparisons use consistent GBX units.
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
            conid=conid or 0,
            ticker=ticker_obj,
            quantity=float(raw.get("position", 0) or raw.get("qty", 0)),
            avg_cost_local=avg_cost_local,
            market_value_usd=market_value_usd,
            unrealized_pnl_usd=float(raw.get("unrealizedPnl", 0)),
            currency=currency,
            current_price_local=current_price_local,
            ticker_identity_verified=ticker_identity_verified,
            ticker_resolution_source=ticker_resolution_source,
        )
        positions.append(position)

    logger.info(
        "positions_normalized",
        count=len(positions),
        skipped=len(raw_positions) - len(positions),
    )
    return positions


def _parse_conid(raw_conid: object) -> int | None:
    """Return a valid IBKR conid, or None when the raw payload is not usable."""
    if isinstance(raw_conid, bool):
        return None
    try:
        conid = int(raw_conid)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return conid if conid > 0 else None


def _should_resolve_position_conid(
    ticker_obj: Ticker,
    *,
    raw_exchange: str,
    raw_currency: str,
) -> bool:
    """Whether a held position should ask IBKR contract details for its conid."""
    exchange = raw_exchange.strip().upper()
    currency = raw_currency.strip().upper()

    if currency == "USD" and exchange in _US_EXCHANGES:
        return False
    if currency not in {"", "USD"} and exchange in {"", "SMART"}:
        return True
    if currency in _MULTI_EXCHANGE_CURRENCIES:
        return True
    return not ticker_obj.exchange_resolved


def build_portfolio_summary(
    ledger: dict,
    positions: list[NormalizedPosition],
    account_id: str = "",
    cash_buffer_pct: float = DEFAULT_CASH_BUFFER_PCT,
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


def _resolve_conid_ticker(
    conid: int,
    client: IbkrClient | None,
    *,
    force_live: bool = False,
    context: str = "watchlist",
) -> TickerResolution:
    """Resolve an IBKR conid to a yfinance ticker.

    Checks the local conid cache first (instant, no API call).  On a miss,
    calls /iserver/contract/{conid}/info via the client, maps to a yfinance
    ticker using the same IBKR→yfinance table as live positions, and caches
    the result so subsequent runs are instant.

    Returns the ticker plus resolution provenance. Inferred mappings remain
    usable for research lookup but cannot authorize an order.
    """
    # Fast path: reverse-lookup in local cache.
    # A bare cached value (no ".") may be a correctly-resolved US ticker OR a
    # previously failed resolution for a non-US stock where the exchange was
    # "SMART" and the currency was ambiguous.  If a client is available, bypass
    # the cache for bare entries so ibkr_symbol_to_yf can try the yfinance
    # search fallback (which is now enabled for SMART + non-USD currency).
    cached = None if force_live else yf_ticker_from_conid(conid)
    if cached and ("." in cached or client is None):
        logger.debug(
            "conid_cache_hit",
            context=context,
            conid=conid,
            yf_ticker=cached,
        )
        return TickerResolution(cached, "unresolved", False)
    if cached:
        logger.debug(
            "conid_bare_cache_bypass",
            context=context,
            conid=conid,
            cached=cached,
            reason="retrying to resolve exchange suffix",
        )

    # Slow path: ask IBKR for contract details
    if client is None:
        return TickerResolution(cached or "", "unresolved", False)

    try:
        info = client.get_contract_info(conid, compete=False)
    except Exception as exc:
        summary = summarize_exception(exc, operation="conid_contract_info")
        summary.pop("message_preview", None)
        logger.warning(
            "conid_contract_info_failed",
            context=context,
            conid=conid,
            **summary,
        )
        return TickerResolution(cached or "", "unresolved", False)

    if not info:
        logger.debug("conid_no_contract_info", context=context, conid=conid)
        try:
            info = client.get_security_definition(conid)
        except AttributeError:
            info = {}
        except Exception as exc:
            summary = summarize_exception(exc, operation="conid_security_definition")
            summary.pop("message_preview", None)
            logger.warning(
                "conid_security_definition_failed",
                context=context,
                conid=conid,
                **summary,
            )
            info = {}
        if not info:
            return TickerResolution(cached or "", "unresolved", False)

    symbol = (info.get("symbol", "") or info.get("ticker", "") or "").strip()
    exchange = (
        info.get("primaryExch", "")
        or info.get("listingExchange", "")
        or info.get("exchange", "")
        or info.get("allExchanges", "")
        or ""
    ).strip()
    currency = (info.get("currency", "") or "").strip()

    if not symbol:
        logger.debug("conid_no_symbol", context=context, conid=conid)
        return TickerResolution(cached or "", "unresolved", False)

    resolution = resolve_ibkr_ticker(symbol, exchange, currency)
    if resolution.yf_ticker and resolution.exchange_verified:
        cache_conid_mapping(resolution.yf_ticker, conid, symbol, exchange)
    if resolution.yf_ticker:
        logger.debug(
            "conid_resolved",
            context=context,
            conid=conid,
            symbol=symbol,
            exchange=exchange,
            currency=currency,
            yf_ticker=resolution.yf_ticker,
            resolution_source=resolution.source,
            exchange_verified=resolution.exchange_verified,
        )
    if resolution.yf_ticker:
        return resolution
    return TickerResolution(cached or "", "unresolved", False)


def _resolve_conid_to_yf(
    conid: int,
    client: IbkrClient | None,
    *,
    force_live: bool = False,
    context: str = "watchlist",
) -> str:
    """Backward-compatible string projection of conid ticker resolution."""
    return _resolve_conid_ticker(
        conid,
        client,
        force_live=force_live,
        context=context,
    ).yf_ticker


def _resolve_watchlist_conid(conid: int, client: IbkrClient | None) -> str:
    """Resolve a watchlist conid to a yfinance ticker."""
    return _resolve_conid_to_yf(conid, client, context="watchlist")


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
        Empty set if client is None, the watchlist exists but is empty, or a
        *default* (unnamed) discovery hit an API error.

    Raises:
        IBKRError: when an *explicitly named* watchlist fetch fails (API/auth
            error) — fail closed so the caller does not act on a phantom-empty list.
    """
    if client is None:
        return set()

    try:
        rows = client.get_watchlist(name_hint)
    except IBKRError:
        if name_hint:
            # Explicitly requested watchlist: fail closed rather than silently
            # degrade to "empty" and produce a misleading zero-candidate report.
            raise
        # Default (unnamed) discovery is best-effort — soft-fail to empty.
        logger.warning("watchlist_default_fetch_failed", reason="api_error")
        return set()
    if rows is None:
        return None  # watchlist not found
    if not rows:
        return set()  # watchlist found but empty

    tickers: set[str] = set()
    skipped = 0
    logger.debug("watchlist_first_row", row=rows[0])
    for row in rows:
        # IBKR watchlist rows — two known formats:
        #   Legacy: {"C": conid_int}  e.g. {"C": 12345678}
        #   New:    {"C": "conid@EXCHANGE", "conid": conid_int}  e.g. {"C": "39131511@TWSE", "conid": 39131511}
        #   Spacer: {"H": "1"}  — no conid, skip
        #
        # Priority: "conid" (clean int) > "conId" > numeric part of "C" (strip @exchange suffix)
        raw_conid = (
            row.get("conid") or row.get("conId") or str(row.get("C", "")).split("@")[0]
        )
        if not raw_conid:
            # Known spacers: {"H": "1"} or similar header rows with no security data.
            # Anything else is an unexpected format — warn so API changes are visible.
            if "H" not in row and row:
                logger.warning(
                    "watchlist_row_unknown_format",
                    row=row,
                    note=(
                        "No 'conid', 'conId', or 'C' field found; row skipped. "
                        "IBKR may have changed the watchlist API response format."
                    ),
                )
            continue

        try:
            conid = int(raw_conid)
        except (TypeError, ValueError):
            logger.warning(
                "watchlist_bad_conid",
                raw=raw_conid,
                row=row,
                note=(
                    "Could not parse conid as integer; row skipped. "
                    "IBKR may have changed the watchlist API response format."
                ),
            )
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
    cash_buffer_pct: float = DEFAULT_CASH_BUFFER_PCT,
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
        logger.warning(
            "portfolio_accounts_preflight_failed",
            **summarize_exception(e, operation="portfolio_accounts_preflight_failed"),
        )

    raw_positions = client.get_positions(acct)
    positions = normalize_positions(raw_positions, client=client)

    ledger = client.get_ledger(acct)
    summary = build_portfolio_summary(ledger, positions, acct, cash_buffer_pct)

    logger.info(
        "portfolio_read",
        account=mask_account(acct),
        positions=summary.position_count,
        value=f"${summary.portfolio_value_usd:,.0f}",
        cash=f"${summary.cash_balance_usd:,.0f}",
        cash_pct=f"{summary.cash_pct:.1f}%",
    )

    return positions, summary
