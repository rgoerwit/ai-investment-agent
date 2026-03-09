"""
Ticker mapping between IBKR conid/symbol and yfinance ticker format.

Wraps the existing TickerFormatter from src/ticker_utils.py with:
- IBKR API calls for conid resolution
- Local JSON cache with TTL
- HK zero-padding normalization
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import structlog

from src.ibkr.exceptions import IBKRTickerResolutionError
from src.ibkr.order_builder import parse_price
from src.ibkr.ticker import _CURRENCY_TO_SUFFIX  # noqa: F401 — re-exported for compat
from src.ticker_utils import TickerFormatter

logger = structlog.get_logger(__name__)

CACHE_FILE = Path("scratch/conid_map.json")
CACHE_TTL_SECONDS = 30 * 24 * 3600  # 30 days

_cache: dict | None = None  # Module-level session cache; loaded once per process


def _get_cache() -> dict:
    """Return session cache, loading from disk on first call."""
    global _cache
    if _cache is None:
        _cache = _load_cache()
    return _cache


def _flush_cache() -> None:
    """Write session cache to disk (only if it was loaded)."""
    if _cache is not None:
        _save_cache(_cache)


def _load_cache() -> dict:
    """Load conid cache from disk."""
    if not CACHE_FILE.exists():
        return {}
    try:
        with open(CACHE_FILE) as f:
            data = json.load(f)
        # Evict stale entries
        now = time.time()
        return {
            k: v for k, v in data.items() if now - v.get("ts", 0) < CACHE_TTL_SECONDS
        }
    except (json.JSONDecodeError, OSError):
        return {}


def _save_cache(cache: dict) -> None:
    """Save conid cache to disk."""
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)
    except OSError as e:
        logger.warning("conid_cache_save_failed", error=str(e))


# Venues that should be excluded from yfinance.Search fallback results
_OTC_VENUES: frozenset[str] = frozenset({"PNK", "OTC", "PINX", "GREY"})


def ibkr_symbol_to_yf(symbol: str, exchange: str, currency: str = "") -> str:
    """
    Convert an IBKR symbol + exchange to yfinance ticker format.

    Uses TickerFormatter.IBKR_TO_YFINANCE mapping.
    Handles HK zero-padding (e.g., IBKR "5" on SEHK → "0005.HK").

    Args:
        symbol: IBKR symbol (e.g., "5", "7203", "ASML")
        exchange: IBKR exchange code (e.g., "SEHK", "TSE", "AEB")
        currency: IBKR currency code (e.g., "HKD", "JPY") — used as fallback
                  when exchange is unknown. Unambiguous single-exchange currencies only.

    Returns:
        yfinance ticker string (e.g., "0005.HK", "7203.T", "ASML.AS")
    """
    suffix = TickerFormatter.IBKR_TO_YFINANCE.get(exchange, "")

    # Fallback: derive suffix from unambiguous currency when exchange is unknown
    if not suffix and currency:
        suffix = _CURRENCY_TO_SUFFIX.get(currency.upper(), "")
        if suffix:
            logger.debug(
                "exchange_suffix_from_currency",
                symbol=symbol,
                exchange=exchange,
                currency=currency,
                suffix=suffix,
            )

    if suffix == ".HK":
        # HK stocks: pad to 4 digits (IBKR strips leading zeros)
        symbol = symbol.lstrip("0") or "0"
        symbol = symbol.zfill(4)

    if suffix:
        return f"{symbol}{suffix}"

    # Returning bare symbol — warn if exchange was specified but unrecognised
    if exchange and exchange not in ("", "SMART", "NASDAQ", "NYSE", "ARCA", "AMEX"):
        # Final fallback: search yfinance before giving up
        yf_ticker = _yf_search_ticker(symbol, exchange, currency)
        if yf_ticker:
            return yf_ticker
        logger.warning(
            "ibkr_exchange_unmapped",
            symbol=symbol,
            exchange=exchange,
            currency=currency,
            note="No yfinance suffix found; position will use bare ticker (may be incorrect for non-US stocks)",
        )
    return symbol


def _yf_search_ticker(symbol: str, exchange: str, currency: str) -> str:
    """
    Resolve an unmapped (symbol, exchange) pair via yfinance.Search.

    Filters out OTC/pink-sheet results.  If the IBKR-reported currency maps
    unambiguously to a yfinance suffix (via _CURRENCY_TO_SUFFIX), only
    candidates with that suffix are accepted.  Otherwise the first remaining
    equity result is used and logged as a best-guess warning.

    Result is cached in conid_map.json (positive and negative) to avoid
    repeat network calls.

    Args:
        symbol:   IBKR symbol (e.g. "ANDR")
        exchange: IBKR exchange code not in IBKR_TO_YFINANCE (e.g. "VSE")
        currency: IBKR-reported currency (e.g. "EUR", "PLN")

    Returns:
        yfinance ticker string (e.g. "ANDR.VI"), or "" if unresolvable.
    """
    cache_key = f"ibkr:{symbol.upper()}:{exchange.upper()}"
    cache = _get_cache()
    if cache_key in cache:
        cached = cache[cache_key].get("yf_ticker", "")
        logger.debug(
            "yf_search_cache_hit",
            symbol=symbol,
            exchange=exchange,
            yf_ticker=cached or "(none)",
        )
        return cached  # "" = negative cache (previous search found nothing)

    try:
        import yfinance as yf

        results = yf.Search(symbol, max_results=10)
        quotes = getattr(results, "quotes", []) or []

        # Keep only equities from non-OTC venues
        candidates = [
            q
            for q in quotes
            if q.get("quoteType") == "EQUITY"
            and q.get("exchange", "") not in _OTC_VENUES
        ]

        matched = ""
        if candidates:
            # If currency maps unambiguously to a suffix, prefer that suffix
            expected_suffix = _CURRENCY_TO_SUFFIX.get(currency.upper(), "")
            if expected_suffix:
                suffix_matches = [
                    q
                    for q in candidates
                    if q.get("symbol", "").endswith(expected_suffix)
                ]
                if suffix_matches:
                    matched = suffix_matches[0].get("symbol", "")

            # No unambiguous suffix match — take first remaining equity (may be ambiguous
            # for multi-country currencies like EUR).  Logged as a best-guess warning so
            # the operator can override via the static table if wrong.
            if not matched:
                matched = candidates[0].get("symbol", "")
                if matched:
                    logger.warning(
                        "yf_search_best_guess",
                        symbol=symbol,
                        exchange=exchange,
                        yf_ticker=matched,
                        note="Ambiguous currency; verify this mapping or add exchange to IBKR_TO_YFINANCE",
                    )

        # Cache result — empty string is a negative cache entry (don't re-search)
        cache[cache_key] = {"yf_ticker": matched, "ts": time.time()}
        _flush_cache()

        if matched:
            logger.info(
                "yf_search_resolved",
                symbol=symbol,
                exchange=exchange,
                yf_ticker=matched,
                currency=currency,
            )
        else:
            logger.debug(
                "yf_search_no_match",
                symbol=symbol,
                exchange=exchange,
                currency=currency,
            )
        return matched

    except Exception as e:
        logger.debug("yf_search_failed", symbol=symbol, exchange=exchange, error=str(e))
        return ""


def yf_to_ibkr_format(yf_ticker: str) -> tuple[str, str]:
    """
    Convert yfinance ticker to IBKR symbol + exchange.

    Args:
        yf_ticker: yfinance ticker (e.g., "0005.HK", "7203.T")

    Returns:
        Tuple of (symbol, ibkr_exchange_code)
    """
    normalized, metadata = TickerFormatter.normalize_ticker(
        yf_ticker, target_format="ibkr"
    )
    symbol = metadata.get("symbol", yf_ticker.split(".")[0])
    ibkr_exchange = metadata.get("ibkr_exchange", "SMART")
    return symbol, ibkr_exchange


def resolve_conid(yf_ticker: str, client: object | None = None) -> int | None:
    """
    Resolve IBKR conid for a yfinance ticker.

    Checks local cache first, then queries IBKR API via client.

    Args:
        yf_ticker: yfinance ticker (e.g., "7203.T")
        client: IbkrClient instance (optional — returns None if not provided)

    Returns:
        IBKR conid integer, or None if not resolved

    Raises:
        IBKRTickerResolutionError: If API is available but resolution fails
    """
    cache = _get_cache()
    cache_key = yf_ticker.upper()

    # Check cache
    if cache_key in cache:
        cached = cache[cache_key]
        conid = cached.get("conid")
        if conid:
            logger.debug("conid_cache_hit", ticker=yf_ticker, conid=conid)
            return conid

    if client is None:
        return None

    # Query IBKR API
    symbol, exchange = yf_to_ibkr_format(yf_ticker)

    try:
        # IBind's stock_conid_by_symbol returns {symbol: [{conid, exchange, ...}]}.
        # default_filtering=False required: the library default applies {isUS: True}
        # which drops all non-US listed contracts (i.e. every stock this system trades).
        result = client.stock_conid_by_symbol(symbol, default_filtering=False)
        if not result or symbol.upper() not in result:
            raise IBKRTickerResolutionError(yf_ticker)

        candidates = result[symbol.upper()]
        if not candidates:
            raise IBKRTickerResolutionError(yf_ticker)

        # Find matching exchange
        conid = None
        for candidate in candidates:
            if candidate.get("exchange") == exchange:
                conid = candidate.get("conid")
                break

        # Fallback: first candidate
        if conid is None:
            conid = candidates[0].get("conid")

        if conid is None:
            raise IBKRTickerResolutionError(yf_ticker)

        # Cache it (write-through: update memory cache and flush to disk)
        cache[cache_key] = {
            "conid": conid,
            "symbol": symbol,
            "exchange": exchange,
            "ts": time.time(),
        }
        _flush_cache()

        logger.info("conid_resolved", ticker=yf_ticker, conid=conid, exchange=exchange)
        return conid

    except IBKRTickerResolutionError:
        raise
    except Exception as e:
        logger.warning("conid_resolution_failed", ticker=yf_ticker, error=str(e))
        raise IBKRTickerResolutionError(yf_ticker, str(e)) from e


def yf_ticker_from_conid(conid: int) -> str | None:
    """Reverse-lookup: find yf_ticker for a known conid from the local cache.

    Checks entries stored by resolve_conid() (key = yf_ticker, value has "conid" field).
    Ignores yfinance.Search cache entries (key prefix "ibkr:").

    Returns yf_ticker string, or None if not found.
    """
    cache = _get_cache()
    for key, entry in cache.items():
        if (
            not key.startswith("ibkr:")
            and isinstance(entry, dict)
            and entry.get("conid") == conid
        ):
            return key
    return None


def cache_conid_mapping(yf_ticker: str, conid: int, symbol: str, exchange: str) -> None:
    """Store a conid↔yf_ticker mapping so future lookups skip the API call."""
    cache = _get_cache()
    cache[yf_ticker.upper()] = {
        "conid": conid,
        "symbol": symbol,
        "exchange": exchange,
        "ts": time.time(),
    }
    _flush_cache()


def resolve_yf_ticker_from_position(position: dict) -> str:
    """
    Extract yfinance ticker from an IBKR position dict.

    IBKR position dicts typically contain:
    - "conid": contract ID
    - "contractDesc" or "ticker": symbol
    - "listingExchange": exchange code

    Args:
        position: Raw IBKR position dict

    Returns:
        yfinance ticker string
    """
    symbol = position.get("contractDesc", "") or position.get("ticker", "")
    exchange = position.get("listingExchange", "") or position.get("exchange", "")
    currency = position.get("currency", "")

    # Clean up symbol (remove trailing spaces, exchange suffixes)
    symbol = symbol.strip()
    # Some IBKR symbols include exchange suffix like "7203-TSE"
    if "-" in symbol:
        parts = symbol.split("-")
        symbol = parts[0]
        if not exchange:
            exchange = parts[1] if len(parts) > 1 else ""

    if not symbol:
        return ""

    return ibkr_symbol_to_yf(symbol, exchange, currency)


def parse_trade_block_price(price_str: str) -> float | None:
    """Parse a price value from TRADE_BLOCK format. See order_builder.parse_price."""
    return parse_price(price_str)
