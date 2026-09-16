"""
Portfolio reading and normalization.

Reads raw IBKR positions and converts them to NormalizedPosition models
with yfinance ticker mapping and FX normalization.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import structlog

from src.error_safety import summarize_exception
from src.fx_normalization import get_fx_rate_cache
from src.ibkr.client import IbkrClient, mask_account
from src.ibkr.exceptions import IBKRError
from src.ibkr.models import (
    NormalizedPosition,
    PortfolioSummary,
    UnresolvedBrokerPosition,
)
from src.ibkr.portfolio_defaults import DEFAULT_CASH_BUFFER_PCT
from src.ibkr.position_values import (
    NormalizedPositionValues,
    normalize_position_values,
)
from src.ibkr.ticker import Ticker, classify_ibkr_symbol
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


def _parse_position_number(
    value: object, *, default: float = 0.0
) -> tuple[float, bool]:
    """Parse one broker number without letting a malformed row abort the snapshot."""
    if value is None or value == "":
        return default, True
    if isinstance(value, bool):
        return default, False
    try:
        return float(value), True  # type: ignore[arg-type]
    except (OverflowError, TypeError, ValueError):
        return default, False


def _position_field(raw: dict, primary: str, fallback: str) -> object:
    """Select an IBKR field without hiding malformed falsey primary values."""
    value = raw.get(primary)
    if value is None or value == "":
        return raw.get(fallback, 0)
    return value


@dataclass(frozen=True)
class _BrokerNumerics:
    raw_market_value: float
    quantity: float
    current_price_local: float
    avg_cost_local: float
    raw_unrealized_pnl: float | None
    malformed_fields: tuple[str, ...]


@dataclass(frozen=True)
class _ResolvedPositionIdentity:
    ticker: Ticker
    verified: bool
    source: str


_PositionIdentityStatus = Literal["resolved", "unresolved", "excluded"]


@dataclass(frozen=True)
class _PositionIdentityOutcome:
    status: _PositionIdentityStatus
    conid: int | None
    resolved: _ResolvedPositionIdentity | None = None
    reason: str | None = None
    instrument_kind: str | None = None


@dataclass(frozen=True)
class _PendingPosition:
    """One parsed broker row awaiting only its batched FX rate."""

    conid: int | None
    broker_token: str
    currency: str
    numerics: _BrokerNumerics
    identity: _ResolvedPositionIdentity | None
    unresolved_reason: str | None = None


@dataclass
class PositionNormalizationResult:
    """Resolved positions plus broker holdings quarantined without a ticker."""

    resolved: list[NormalizedPosition] = field(default_factory=list)
    unresolved: list[UnresolvedBrokerPosition] = field(default_factory=list)
    excluded_non_security_count: int = 0


def _resolve_position_identity(
    symbol: str,
    exchange: str,
    currency: str,
    conid: int | None,
    client: IbkrClient | None,
) -> _PositionIdentityOutcome:
    """Resolve one broker identity before it can acquire a research ticker."""
    classification = classify_ibkr_symbol(symbol.partition("-")[0])
    if classification.remedy == "drop":
        return _PositionIdentityOutcome(
            "excluded", conid, instrument_kind=classification.kind
        )

    if classification.remedy == "recover_from_conid" or not symbol:
        embedded_conid = classification.contract_id
        # Conflicting identifiers are not grounds to choose either instrument.
        if conid is not None and embedded_conid is not None and embedded_conid != conid:
            logger.warning(
                "position_contract_identifier_mismatch",
                conid=conid,
                embedded_conid=embedded_conid,
            )
            return _PositionIdentityOutcome(
                "unresolved",
                conid,
                reason="broker contract identifier does not match conid",
            )
        conid = conid or embedded_conid
        # A placeholder provides no competing market identity: use a safe cached
        # conid observation before requiring an available brokerage session.
        resolution = (
            _resolve_conid_ticker(conid, client, context="position")
            if conid is not None
            else TickerResolution("", "unresolved", False)
        )
        if not resolution.yf_ticker:
            return _PositionIdentityOutcome(
                "unresolved", conid, reason="canonical security identity unavailable"
            )
        # The resolved market key already owns its venue. Raw broker currency
        # must not add a different suffix to a deliberately bare US identity.
        ticker = Ticker.from_yf(resolution.yf_ticker)
    else:
        ticker = Ticker.from_ibkr(classification.symbol, exchange, currency)
        smart_non_usd = exchange.upper() in {"", "SMART"} and currency.upper() not in {
            "",
            "USD",
        }
        verified = ticker.exchange_resolved and not smart_non_usd
        resolution = TickerResolution(
            ticker.yf,
            "exchange_map"
            if verified
            else ("currency_fallback" if ticker.has_suffix else "unresolved"),
            verified,
        )
        recovered_from_conid = False
        if (
            conid is not None
            and client is not None
            and _should_resolve_position_conid(
                ticker, raw_exchange=exchange, raw_currency=currency
            )
        ):
            # Ordinary symbols can carry ambiguous venue metadata, so consult
            # the live contract while retaining safe cache fallback.
            contract_resolution = _resolve_conid_ticker(
                conid, client, force_live=True, context="position"
            )
            if contract_resolution.source == "non_analyzable":
                return _PositionIdentityOutcome(
                    "unresolved",
                    conid,
                    reason="broker contract is not a market security",
                )
            if contract_resolution.yf_ticker:
                resolution = contract_resolution
                ticker = Ticker.from_yf(resolution.yf_ticker)
                recovered_from_conid = True

        # Search can fill an unknown raw-symbol listing, but cannot replace a
        # recovered contract identity with a weaker guess.
        if (
            not recovered_from_conid
            and not ticker.has_suffix
            and exchange
            and exchange.upper() not in _US_EXCHANGES
        ):
            searched = _yf_search_ticker(classification.symbol, exchange, currency)
            if searched:
                ticker = Ticker.from_yf(searched)
                resolution = TickerResolution(searched, "yfinance_search", False)

    overridden, was_overridden = apply_operator_override(ticker.yf)
    if was_overridden:
        ticker = Ticker.from_yf(overridden)
        resolution = TickerResolution(overridden, "operator_override", True)
    return _PositionIdentityOutcome(
        "resolved",
        conid,
        resolved=_ResolvedPositionIdentity(
            ticker, resolution.exchange_verified, resolution.source
        ),
    )


def _parse_broker_numerics(raw: dict) -> _BrokerNumerics:
    """Parse the same monetary inputs for resolved and quarantined holdings."""
    market_value, market_value_valid = _parse_position_number(
        _position_field(raw, "mktValue", "marketValue")
    )
    quantity, quantity_valid = _parse_position_number(
        _position_field(raw, "position", "qty")
    )
    price, price_valid = _parse_position_number(
        _position_field(raw, "mktPrice", "lastPrice")
    )
    cost, cost_valid = _parse_position_number(
        _position_field(raw, "avgCost", "avgPrice")
    )
    raw_pnl = raw.get("unrealizedPnl")
    pnl, pnl_valid = _parse_position_number(raw_pnl)
    return _BrokerNumerics(
        raw_market_value=market_value,
        quantity=quantity,
        current_price_local=price,
        avg_cost_local=cost,
        raw_unrealized_pnl=None if raw_pnl is None else pnl,
        malformed_fields=tuple(
            name
            for name, valid in (
                ("quantity", quantity_valid),
                ("market_value", market_value_valid),
                ("current_price", price_valid),
                ("avg_cost", cost_valid),
                ("unrealized_pnl", pnl_valid),
            )
            if not valid
        ),
    )


def _value_broker_position(
    numerics: _BrokerNumerics, currency: str, fx_rate: float | None
) -> NormalizedPositionValues:
    """Apply one valuation policy independently of the identity outcome."""
    if numerics.malformed_fields:
        return NormalizedPositionValues(
            market_value_usd=0.0,
            unrealized_pnl_usd=0.0,
            fx_rate_to_usd=None,
            market_value_basis="UNAVAILABLE",
            unrealized_pnl_basis="UNAVAILABLE",
            valuation_valid=False,
            valuation_issue=(
                "Malformed broker numeric field(s): "
                + ", ".join(numerics.malformed_fields)
            ),
        )
    return normalize_position_values(
        quantity=numerics.quantity,
        current_price_local=numerics.current_price_local,
        avg_cost_local=numerics.avg_cost_local,
        raw_market_value=numerics.raw_market_value,
        raw_unrealized_pnl=numerics.raw_unrealized_pnl,
        currency=currency,
        fx_rate=fx_rate,
    )


def normalize_positions(
    raw_positions: list[dict],
    *,
    client: IbkrClient | None = None,
) -> PositionNormalizationResult:
    """Resolve broker identities and retain unresolved holdings for accounting.

    Callers must explicitly consume the resolved and unresolved collections.
    Identity resolution precedes a single batched FX lookup and valuation path.
    """
    pending: list[_PendingPosition] = []
    result = PositionNormalizationResult()
    for raw in raw_positions:
        symbol = (raw.get("contractDesc", "") or raw.get("ticker", "")).strip()
        exchange = (raw.get("listingExchange", "") or raw.get("exchange", "")).strip()
        currency = (raw.get("currency", "") or "").strip()
        outcome = _resolve_position_identity(
            symbol, exchange, currency, _parse_conid(raw.get("conid")), client
        )
        if outcome.status == "excluded":
            result.excluded_non_security_count += 1
            logger.info(
                "position_non_analyzable_skipped",
                instrument_kind=outcome.instrument_kind,
            )
            continue
        if outcome.status == "unresolved":
            logger.warning("position_identity_unresolved", conid=outcome.conid)
        identity = outcome.resolved
        pending.append(
            _PendingPosition(
                conid=outcome.conid,
                broker_token=symbol,
                currency=currency
                or (
                    "GBP"
                    if identity is not None and identity.ticker.suffix == ".L"
                    else "USD"
                ),
                numerics=_parse_broker_numerics(raw),
                identity=identity,
                unresolved_reason=outcome.reason,
            )
        )

    fx_rates = get_fx_rate_cache().resolve_rates_sync({p.currency for p in pending})
    for position in pending:
        numerics = position.numerics
        rate_info = fx_rates.get(position.currency.strip().upper())
        values = _value_broker_position(
            numerics, position.currency, rate_info[0] if rate_info else None
        )
        if not values.valuation_valid:
            logger.warning(
                "position_valuation_unavailable",
                conid=position.conid,
                currency=position.currency,
                reason=values.valuation_issue,
            )
        if position.identity is None:
            assert position.unresolved_reason is not None
            result.unresolved.append(
                UnresolvedBrokerPosition(
                    conid=position.conid or 0,
                    broker_token=position.broker_token,
                    quantity=numerics.quantity,
                    currency=position.currency,
                    market_value_usd=values.market_value_usd,
                    valuation_valid=values.valuation_valid,
                    valuation_issue=values.valuation_issue,
                    reason=position.unresolved_reason,
                )
            )
            continue

        identity = position.identity
        # Preserve broker currency units. Comparisons convert both price sides
        # by currency code; exchange suffix alone must never rescale GBP prices.
        result.resolved.append(
            NormalizedPosition(
                conid=position.conid or 0,
                ticker=identity.ticker,
                quantity=numerics.quantity,
                avg_cost_local=numerics.avg_cost_local,
                market_value_usd=values.market_value_usd,
                unrealized_pnl_usd=values.unrealized_pnl_usd,
                fx_rate_to_usd=values.fx_rate_to_usd,
                market_value_basis=values.market_value_basis,
                unrealized_pnl_basis=values.unrealized_pnl_basis,
                valuation_valid=values.valuation_valid,
                valuation_issue=values.valuation_issue,
                position_flat=values.position_flat,
                currency=position.currency,
                current_price_local=numerics.current_price_local,
                ticker_identity_verified=identity.verified,
                ticker_resolution_source=identity.source,
            )
        )
    logger.info(
        "positions_normalized",
        count=len(result.resolved),
        unresolved=len(result.unresolved),
        excluded_non_security=result.excluded_non_security_count,
    )
    return result


def _parse_conid(raw_conid: object) -> int | None:
    """Return a valid IBKR conid, or None when the raw payload is not usable."""
    if isinstance(raw_conid, bool):
        return None
    if not isinstance(raw_conid, str | int | float):
        return None
    try:
        conid = int(raw_conid)
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
    positions: Sequence[NormalizedPosition],
    account_id: str = "",
    cash_buffer_pct: float = DEFAULT_CASH_BUFFER_PCT,
    *,
    unresolved_positions: Sequence[UnresolvedBrokerPosition] = (),
) -> PortfolioSummary:
    """
    Build portfolio summary from IBKR ledger and normalized positions.

    Args:
        ledger: Raw IBKR ledger dict
        positions: Normalized positions
        account_id: IBKR account ID
        cash_buffer_pct: Cash buffer fraction (don't deploy into new BUYs)
        unresolved_positions: Holdings retained for accounting without a safe ticker

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
        portfolio_value = sum(p.market_value_usd for p in positions) + sum(
            p.market_value_usd for p in unresolved_positions
        )

    # Fallback portfolio value from positions
    if portfolio_value <= 0:
        portfolio_value = (
            sum(p.market_value_usd for p in positions)
            + sum(p.market_value_usd for p in unresolved_positions)
            + max(cash, 0)
        )

    cash_pct = (cash / portfolio_value * 100) if portfolio_value > 0 else 0.0
    # available_cash derived from settled_cash (not total cash) — only spendable funds
    available_cash = max(0, settled_cash - (portfolio_value * cash_buffer_pct))

    return PortfolioSummary(
        account_id=account_id,
        portfolio_value_usd=portfolio_value,
        cash_balance_usd=cash,
        settled_cash_usd=settled_cash,
        cash_pct=cash_pct,
        position_count=len(positions) + len(unresolved_positions),
        available_cash_usd=available_cash,
        unresolved_positions=list(unresolved_positions),
    )


def _resolve_conid_ticker(
    conid: int,
    client: IbkrClient | None,
    *,
    force_live: bool = False,
    context: str = "watchlist",
) -> TickerResolution:
    """Resolve an IBKR conid to a yfinance ticker.

    Checks the local conid cache first. Unless ``force_live`` is set, a suffixed
    cached identity is returned immediately. Live resolution can improve or
    invalidate it; transient live failures retain the cached mapping.

    Returns the ticker plus resolution provenance. Inferred mappings remain
    usable for research lookup but cannot authorize an order.
    """
    # Fast path: reverse-lookup in local cache.
    # A bare cached value (no ".") may be a correctly-resolved US ticker OR a
    # previously failed resolution for a non-US stock where the exchange was
    # "SMART" and the currency was ambiguous.  If a client is available, bypass
    # the cache for bare entries so ibkr_symbol_to_yf can try the yfinance
    # search fallback (which is now enabled for SMART + non-USD currency).
    cached = yf_ticker_from_conid(conid)
    fast_path = None if force_live else cached
    if fast_path and ("." in fast_path or client is None):
        logger.debug(
            "conid_cache_hit",
            context=context,
            conid=conid,
            yf_ticker=fast_path,
        )
        return TickerResolution(fast_path, "conid_cache", False)
    if fast_path:
        logger.debug(
            "conid_bare_cache_bypass",
            context=context,
            conid=conid,
            cached=fast_path,
            reason="retrying to resolve exchange suffix",
        )

    # Slow path: ask IBKR for contract details
    if client is None:
        return TickerResolution(
            cached or "",
            "conid_cache" if cached else "unresolved",
            False,
        )

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
        return TickerResolution(
            cached or "",
            "conid_cache" if cached else "unresolved",
            False,
        )

    symbol = (info.get("symbol", "") or info.get("ticker", "") or "").strip()
    classification = classify_ibkr_symbol(symbol)
    if not info or classification.kind == "contract_identifier":
        event = "conid_no_contract_info" if not info else "conid_placeholder_symbol"
        logger.debug(event, context=context, conid=conid)
        try:
            security_definition = client.get_security_definition(conid)
        except AttributeError:
            security_definition = {}
        except Exception as exc:
            summary = summarize_exception(exc, operation="conid_security_definition")
            summary.pop("message_preview", None)
            logger.warning(
                "conid_security_definition_failed",
                context=context,
                conid=conid,
                **summary,
            )
            security_definition = {}
        if security_definition:
            info = security_definition
        elif not info:
            return TickerResolution(
                cached or "",
                "conid_cache" if cached else "unresolved",
                False,
            )
        elif classification.kind == "contract_identifier":
            # A placeholder returned from both live symbol-bearing fields is
            # not evidence that a previously verified mapping became wrong.
            return TickerResolution(
                cached or "",
                "conid_cache" if cached else "unresolved",
                False,
            )

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
        return TickerResolution(
            cached or "",
            "conid_cache" if cached else "unresolved",
            False,
        )

    resolution = resolve_ibkr_ticker(symbol, exchange, currency)
    if resolution.source == "non_analyzable":
        # Placeholder metadata is an availability failure, not evidence that a
        # validated cached security identity became false. A real non-security
        # token such as .REC remains authoritative negative evidence.
        if classify_ibkr_symbol(symbol).kind == "contract_identifier" and cached:
            return TickerResolution(cached, "conid_cache", False)
        return resolution
    if resolution.yf_ticker and resolution.exchange_verified:
        cache_conid_mapping(
            resolution.yf_ticker,
            conid,
            symbol,
            exchange,
            source="contract_info",
            confidence="verified",
        )
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
    return TickerResolution(
        cached or "",
        "conid_cache" if cached else "unresolved",
        False,
    )


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
    normalized = normalize_positions(raw_positions, client=client)
    positions = normalized.resolved

    ledger = client.get_ledger(acct)
    summary = build_portfolio_summary(
        ledger,
        positions,
        acct,
        cash_buffer_pct,
        unresolved_positions=normalized.unresolved,
    )

    logger.info(
        "portfolio_read",
        account=mask_account(acct),
        positions=summary.position_count,
        value=f"${summary.portfolio_value_usd:,.0f}",
        cash=f"${summary.cash_balance_usd:,.0f}",
        cash_pct=f"{summary.cash_pct:.1f}%",
    )

    return positions, summary
