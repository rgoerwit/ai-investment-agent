"""
International Ticker Utilities
Updated: Removed brittle hardcoded maps in favor of dynamic name normalization
and strict search query generation.
"""

import asyncio
import re
from dataclasses import dataclass

import structlog

from src.blocking_io import (
    YAHOOQUERY_QUOTE_TYPE_POLICY,
    YFINANCE_INFO_POLICY,
    run_blocking_call,
)
from src.exchange_metadata import (
    EXCHANGES_BY_SUFFIX,
    IBKR_TO_YFINANCE,
    canonical_suffix_for_reuters_exchange,
    canonical_suffix_for_token,
    format_ibkr_symbol,
    format_yahoo_symbol,
)

logger = structlog.get_logger(__name__)
_ibkr_name_service = None

_COMPANY_NAME_BASE_FALLBACK_SUFFIXES = frozenset({".ST"})
_NORMALIZED_SHARE_CLASS_TICKER_PATTERN = re.compile(
    r"^([A-Z0-9][A-Z0-9-]*)-([A-Z])(\.[A-Z]+)$"
)

# Legal entity suffixes to strip for cleaner search queries
LEGAL_SUFFIXES = [
    r"\s+Company\s+Limited",
    r"\s+Co\.,?\s+Ltd\.?",
    r"\s+Ltd\.?",
    r"\s+Limited",
    r"\s+Corp\.?",
    r"\s+Corporation",
    r"\s+Inc\.?",
    r"\s+Incorporated",
    r"\s+PLC",
    r"\s+Public\s+Limited\s+Company",
    r"\s+S\.A\.",
    r"\s+AG",
    r"\s+SE",
    r"\s+Group",
    r"\s+Holdings?",
    r"\s+\(Holdings?\)",
    r"\s+NV",
    r"\s+BV",
    r"\s+GmbH",
    r"\s+K\.K\.",
    r"\s+Kabushiki\s+Kaisha",
    r"\s+Pty",
    r"\s+Pte",
    r"\s+S\.p\.A\.",
    r"\s+SA\/NV",
]


def normalize_company_name(raw_name: str) -> str:
    """
    Dynamically strips legal fluff to isolate the 'Semantic Core' of the name.

    Example:
    "China Resources Beer (Holdings) Company Limited" -> "China Resources Beer"
    "Samsung Electronics Co., Ltd." -> "Samsung Electronics"

    This allows for quoted searches like "China Resources Beer" which excludes "Cement".
    """
    if not raw_name:
        return ""

    clean_name = raw_name.strip()

    # 1. Remove text inside parentheses (often legal descriptors or stock codes)
    # e.g. "Tencent Holdings (0700)" -> "Tencent Holdings"
    clean_name = re.sub(r"\s*\(.*?\)", "", clean_name)

    # 2. Iteratively strip legal suffixes (case insensitive)
    # We loop because sometimes they stack (e.g. "Group Holdings Ltd")
    # We sort suffixes by length (desc) to catch "Public Limited Company" before "Company"
    sorted_suffixes = sorted(LEGAL_SUFFIXES, key=len, reverse=True)

    original = clean_name
    for _ in range(2):  # Run twice to catch stacked suffixes
        for suffix in sorted_suffixes:
            clean_name = re.sub(suffix + "$", "", clean_name, flags=re.IGNORECASE)

    clean_name = clean_name.strip().rstrip(",").strip()

    # Safety valve: If we stripped everything (e.g. company was just named "Holdings"), revert
    if len(clean_name) < 2:
        return original

    return clean_name


def generate_strict_search_query(ticker: str, raw_name: str, topic: str) -> str:
    """
    Generates a search query that enforces exact name matching to prevent hallucinations.

    Format: '"Semantic Core Name" {ticker} {topic}'
    """
    core_name = normalize_company_name(raw_name)

    # If the name is very short (e.g. "BP"), quotes might be too restrictive,
    # but for Asian multi-word names, quotes are essential.
    if len(core_name.split()) > 1:
        query = f'"{core_name}" {ticker} {topic}'
    else:
        query = f"{core_name} {ticker} {topic}"

    return query


class TickerFormatter:
    """Handles international ticker format conversion and validation."""

    # Alternative ticker format patterns
    TICKER_PATTERNS = {
        "reuters": re.compile(r"^([A-Z0-9]+)\.([A-Z]+)-([A-Z]{2})$"),
        "standard": re.compile(r"^([A-Z0-9][A-Z0-9-]*)\.([A-Z]+)$"),
        "plain": re.compile(r"^([A-Z0-9]+)$"),
        "ibkr": re.compile(r"^([A-Z0-9]+):([A-Z]+)$"),
    }

    @staticmethod
    def _format_symbol_for_target(symbol: str, suffix: str, target_format: str) -> str:
        if target_format == "ibkr":
            return format_ibkr_symbol(symbol, suffix)
        return format_yahoo_symbol(symbol, suffix)

    @classmethod
    def normalize_ticker(
        cls, ticker: str, target_format: str = "yfinance"
    ) -> tuple[str, dict[str, str]]:
        """
        Normalize ticker to target format and extract metadata.
        """
        original_ticker = ticker.strip().upper()
        ticker = original_ticker

        # FIRST: Apply known corrections from ticker_corrections module
        try:
            from src.ticker_corrections import TickerCorrector

            corrected, was_corrected, company_name = TickerCorrector.apply_correction(
                ticker
            )
            if was_corrected:
                logger.info(
                    "ticker_pre_corrected",
                    original=ticker,
                    corrected=corrected,
                    company=company_name,
                )
                ticker = corrected
        except ImportError:
            logger.debug("ticker_corrections_module_not_available")

        # Normalize multi-dot share-class tickers: NIL.B.ST → NIL-B.ST
        # Scandinavian and other exchanges use a dot before the share-class letter
        # (A, B, C …); yfinance expects a hyphen in that position instead.
        # Only fires when the final segment is a known exchange suffix — US and
        # unknown-suffix tickers are unaffected.
        _parts = ticker.split(".")
        if len(_parts) > 2 and canonical_suffix_for_token(_parts[-1]):
            _rejoined = "-".join(_parts[:-1]) + "." + _parts[-1]
            logger.debug(
                "multi_dot_ticker_normalised", original=ticker, normalised=_rejoined
            )
            ticker = _rejoined

        # Try IBKR format (e.g., "NOVN:SWX")
        ibkr_match = cls.TICKER_PATTERNS["ibkr"].match(ticker)
        if ibkr_match:
            symbol, exchange = ibkr_match.groups()
            return cls._convert_from_ibkr(
                symbol, exchange, target_format, original_ticker
            )

        # Try Reuters format (e.g., "NOV.N-CH")
        reuters_match = cls.TICKER_PATTERNS["reuters"].match(ticker)
        if reuters_match:
            symbol, reuters_code, country_code = reuters_match.groups()
            canonical_suffix = cls._map_reuters_to_exchange(reuters_code, country_code)

            if canonical_suffix:
                exchange_info = EXCHANGES_BY_SUFFIX[canonical_suffix]
                formatted_symbol = cls._format_symbol_for_target(
                    symbol, exchange_info.yf_suffix, target_format
                )
                if target_format == "yfinance":
                    normalized = f"{formatted_symbol}{exchange_info.yf_suffix}"
                elif target_format == "ibkr":
                    normalized = f"{formatted_symbol}:{exchange_info.ibkr_code}"
                else:
                    normalized = ticker

                metadata = {
                    "original": original_ticker,
                    "symbol": formatted_symbol,
                    "exchange_suffix": exchange_info.yf_suffix,
                    "exchange_name": exchange_info.exchange_name,
                    "country": exchange_info.country,
                    "ibkr_exchange": exchange_info.ibkr_code,
                    "format": "reuters",
                }
                return normalized, metadata

        # Try standard format (e.g., "NOVN.SW")
        standard_match = cls.TICKER_PATTERNS["standard"].match(ticker)
        if standard_match:
            symbol, suffix = standard_match.groups()

            canonical_suffix = canonical_suffix_for_token(suffix)
            if canonical_suffix:
                exchange_info = EXCHANGES_BY_SUFFIX[canonical_suffix]
                formatted_symbol = cls._format_symbol_for_target(
                    symbol, exchange_info.yf_suffix, target_format
                )

                if target_format == "yfinance":
                    normalized = f"{formatted_symbol}{exchange_info.yf_suffix}"
                elif target_format == "ibkr":
                    normalized = f"{formatted_symbol}:{exchange_info.ibkr_code}"
                else:
                    normalized = ticker

                metadata = {
                    "original": original_ticker,
                    "symbol": formatted_symbol,
                    "exchange_suffix": exchange_info.yf_suffix,
                    "exchange_name": exchange_info.exchange_name,
                    "country": exchange_info.country,
                    "ibkr_exchange": exchange_info.ibkr_code,
                    "format": "standard",
                }
                return normalized, metadata
            else:
                # 1-char suffix not recognised as an exchange → US share-class separator.
                # e.g. PBR.A → PBR-A, BRK.B → BRK-B (yfinance hyphen convention for
                # preferred/class shares and ADRs).  Known single-letter exchange codes
                # (T=Tokyo, L=London, V=TSX Venture, …) are caught by the
                # canonical exchange-suffix branch above and never reach here.
                if len(suffix) == 1:
                    share_class_base = f"{symbol}-{suffix}"
                    normalized = (
                        share_class_base
                        if target_format == "yfinance"
                        else f"{share_class_base}:SMART"
                    )
                    metadata = {
                        "original": original_ticker,
                        "symbol": symbol,
                        "exchange_suffix": f".{suffix}",
                        "exchange_name": "US Exchange (assumed)",
                        "country": "United States",
                        "ibkr_exchange": "SMART",
                        "format": "share_class",
                    }
                    return normalized, metadata

                normalized = ticker
                metadata = {
                    "original": original_ticker,
                    "symbol": symbol,
                    "exchange_suffix": f".{suffix}",
                    "exchange_name": "Unknown",
                    "country": "Unknown",
                    "ibkr_exchange": "SMART",
                    "format": "unknown",
                }
                return normalized, metadata

        # Plain ticker (assume US if no suffix)
        plain_match = cls.TICKER_PATTERNS["plain"].match(ticker)
        if plain_match:
            normalized = ticker if target_format == "yfinance" else f"{ticker}:SMART"
            metadata = {
                "original": original_ticker,
                "symbol": ticker,
                "exchange_suffix": "",
                "exchange_name": "US Exchange (assumed)",
                "country": "United States",
                "ibkr_exchange": "SMART",
                "format": "plain",
            }
            return normalized, metadata

        # Unable to parse
        metadata = {
            "original": original_ticker,
            "symbol": ticker,
            "exchange_suffix": "",
            "exchange_name": "Unknown",
            "country": "Unknown",
            "ibkr_exchange": "SMART",
            "format": "invalid",
        }
        return ticker, metadata

    @classmethod
    def _convert_from_ibkr(
        cls, symbol: str, exchange: str, target_format: str, original_ticker: str
    ) -> tuple[str, dict[str, str]]:
        """Convert from IBKR format to target format."""
        suffix = IBKR_TO_YFINANCE.get(exchange)
        if suffix:
            info = EXCHANGES_BY_SUFFIX[suffix]
            formatted_symbol = cls._format_symbol_for_target(
                symbol, info.yf_suffix, target_format
            )
            if target_format == "yfinance":
                normalized = f"{formatted_symbol}{info.yf_suffix}"
            else:
                normalized = f"{formatted_symbol}:{exchange}"

            metadata = {
                "original": original_ticker,
                "symbol": formatted_symbol,
                "exchange_suffix": info.yf_suffix,
                "exchange_name": info.exchange_name,
                "country": info.country,
                "ibkr_exchange": exchange,
                "format": "ibkr",
            }
            return normalized, metadata
        if suffix == "":
            normalized = (
                symbol if target_format == "yfinance" else f"{symbol}:{exchange}"
            )
            metadata = {
                "original": original_ticker,
                "symbol": symbol,
                "exchange_suffix": "",
                "exchange_name": "US Exchange (assumed)",
                "country": "United States",
                "ibkr_exchange": exchange,
                "format": "ibkr",
            }
            return normalized, metadata

        if target_format == "yfinance":
            normalized = symbol
        else:
            normalized = f"{symbol}:{exchange}"

        metadata = {
            "original": original_ticker,
            "symbol": symbol,
            "exchange_suffix": "",
            "exchange_name": f"Exchange {exchange}",
            "country": "United States"
            if exchange in ["NASDAQ", "NYSE", "SMART"]
            else "Unknown",
            "ibkr_exchange": exchange,
            "format": "ibkr",
        }
        return normalized, metadata

    @classmethod
    def _map_reuters_to_exchange(
        cls, reuters_code: str, country_code: str
    ) -> str | None:
        """Return the canonical yfinance suffix for a Reuters exchange/country pair."""
        return canonical_suffix_for_reuters_exchange(reuters_code, country_code)

    @classmethod
    def to_yfinance(cls, ticker: str) -> str:
        """Convert ticker to yfinance format."""
        normalized, _ = cls.normalize_ticker(ticker, target_format="yfinance")
        return normalized

    @classmethod
    def to_ibkr(cls, ticker: str) -> str:
        """Convert any ticker format to IBKR format."""
        normalized, metadata = cls.normalize_ticker(ticker, target_format="ibkr")
        return normalized

    @classmethod
    def get_exchange_info(cls, ticker: str) -> dict[str, str]:
        """Get exchange information for a ticker."""
        _, metadata = cls.normalize_ticker(ticker)
        return metadata

    @classmethod
    def is_international(cls, ticker: str) -> bool:
        """Check if ticker is for a non-US exchange."""
        _, metadata = cls.normalize_ticker(ticker)
        return metadata.get("country", "United States") != "United States"


# Convenience functions
def normalize_ticker(ticker: str, target_format: str = "yfinance") -> str:
    """Normalize ticker to target format."""
    normalized, _ = TickerFormatter.normalize_ticker(
        ticker, target_format=target_format
    )
    return normalized


def to_yfinance(ticker: str) -> str:
    """Convert any ticker format to yfinance format."""
    return TickerFormatter.to_yfinance(ticker)


def to_ibkr(ticker: str) -> str:
    """Convert any ticker format to IBKR format."""
    return TickerFormatter.to_ibkr(ticker)


def get_ticker_info(ticker: str) -> dict[str, str]:
    """Get complete ticker information."""
    _, metadata = TickerFormatter.normalize_ticker(ticker)
    return metadata


# --- Company Name Resolution ---


@dataclass
class CompanyNameResult:
    """Result of multi-source company name resolution."""

    name: str  # Resolved name or ticker as fallback
    source: str  # Which source resolved it ("yfinance", "yahooquery", "fmp", "eodhd", "unresolved")
    is_resolved: bool  # True if a real name was found (not just ticker echoed back)


def _company_name_lookup_candidates(ticker: str) -> list[tuple[str, str]]:
    """Return ordered lookup aliases for company-name resolution."""
    cleaned = ticker.strip().upper()
    candidates: list[tuple[str, str]] = []
    seen: set[str] = set()

    def add(symbol: str, strategy: str) -> None:
        if symbol and symbol not in seen:
            candidates.append((symbol, strategy))
            seen.add(symbol)

    add(cleaned, "exact")

    normalized = normalize_ticker(cleaned)
    if normalized != cleaned:
        add(normalized, "normalized_alias")

    match = _NORMALIZED_SHARE_CLASS_TICKER_PATTERN.match(normalized)
    if match:
        base_symbol, _share_class, exchange_suffix = match.groups()
        if exchange_suffix in _COMPANY_NAME_BASE_FALLBACK_SUFFIXES:
            add(f"{base_symbol}{exchange_suffix}", "base_ticker_fallback")

    return candidates


def _is_valid_company_name(name: str | None, ticker: str) -> bool:
    """Check if a resolved name is valid (not empty, not just the ticker echoed back)."""
    if not name or not name.strip():
        return False
    cleaned = name.strip()
    # Reject if name is just the ticker string (some APIs echo ticker as name)
    if cleaned.upper() == ticker.upper():
        return False
    # Reject if name is just the ticker base (without exchange suffix)
    ticker_base = ticker.split(".")[0]
    if cleaned.upper() == ticker_base.upper():
        return False
    return True


async def _try_yfinance(ticker: str) -> str | None:
    """Attempt company name resolution via yfinance."""
    try:
        import yfinance as yf

        info = await run_blocking_call(
            YFINANCE_INFO_POLICY.with_label(f"yfinance.info:{ticker}"),
            lambda: yf.Ticker(ticker).info,
        )
        if info:
            name = info.get("longName") or info.get("shortName")
            return name if isinstance(name, str) else None
    except (asyncio.TimeoutError, Exception) as e:
        logger.debug("company_name_yfinance_failed", ticker=ticker, error=str(e))
    return None


async def _try_yahooquery(ticker: str) -> str | None:
    """Attempt company name resolution via yahooquery."""
    try:
        from yahooquery import Ticker as YQTicker

        result = await run_blocking_call(
            YAHOOQUERY_QUOTE_TYPE_POLICY.with_label(f"yahooquery.quote_type:{ticker}"),
            lambda: YQTicker(ticker).quote_type,
        )
        if isinstance(result, dict) and ticker in result:
            data = result[ticker]
            if isinstance(data, dict):
                name = data.get("longName") or data.get("shortName")
                return name if isinstance(name, str) else None
    except (asyncio.TimeoutError, Exception) as e:
        logger.debug("company_name_yahooquery_failed", ticker=ticker, error=str(e))
    return None


async def _try_fmp(ticker: str) -> str | None:
    """Attempt company name resolution via FMP profile endpoint."""
    try:
        from src.data.fmp_fetcher import get_fmp_fetcher

        fmp = get_fmp_fetcher()
        if not fmp.is_available():
            return None
        name = await asyncio.wait_for(fmp.get_company_name(ticker), timeout=5)
        return name
    except (asyncio.TimeoutError, Exception) as e:
        logger.debug("company_name_fmp_failed", ticker=ticker, error=str(e))
    return None


async def _try_eodhd(ticker: str) -> str | None:
    """Attempt company name resolution via EODHD General endpoint."""
    try:
        from src.data.eodhd_fetcher import get_eodhd_fetcher

        eodhd = get_eodhd_fetcher()
        if not eodhd.is_available():
            return None
        name = await asyncio.wait_for(eodhd.get_company_name(ticker), timeout=5)
        return name
    except (asyncio.TimeoutError, Exception) as e:
        logger.debug("company_name_eodhd_failed", ticker=ticker, error=str(e))
    return None


async def _try_ibkr(ticker: str) -> str | None:
    """Attempt company name resolution via the sparse IBKR security probe."""
    try:
        probe = await _get_ibkr_name_service().probe_security(ticker)
        if probe.identity_confidence == "VERIFIED":
            return probe.company_name if isinstance(probe.company_name, str) else None
    except Exception as e:
        logger.debug("company_name_ibkr_failed", ticker=ticker, error=str(e))
    return None


def _get_ibkr_name_service():
    global _ibkr_name_service
    if _ibkr_name_service is None:
        from src.ibkr.security_data_service import IbkrSecurityDataService

        _ibkr_name_service = IbkrSecurityDataService()
    return _ibkr_name_service


async def resolve_company_name(
    ticker: str,
    *,
    allow_ibkr_probe: bool = False,
) -> CompanyNameResult:
    """
    Resolve company name from multiple sources with fallback chain.

    Tries sources in order (stops at first success):
    1. yfinance (free, cached)
    2. yahooquery (free, different backend)
    3. FMP (paid, lightweight profile call)
    4. EODHD (paid, filtered General endpoint)

    The optional IBKR probe is disabled by default because its brokerage
    connection path can perform synchronous retries that are too expensive for
    normal analysis startup.

    Each source has a 5-second timeout. Names are validated to ensure
    they aren't just the ticker string echoed back.

    Returns:
        CompanyNameResult with resolved name, source, and is_resolved flag.
    """
    sources = [
        ("yfinance", _try_yfinance),
        ("yahooquery", _try_yahooquery),
        ("fmp", _try_fmp),
        ("eodhd", _try_eodhd),
    ]
    lookup_candidates = _company_name_lookup_candidates(ticker)

    for lookup_ticker, lookup_strategy in lookup_candidates:
        for source_name, resolver in sources:
            try:
                raw_name = await resolver(lookup_ticker)
                if isinstance(raw_name, str) and _is_valid_company_name(
                    raw_name, lookup_ticker
                ):
                    normalized = normalize_company_name(raw_name)
                    logger.debug(
                        "company_name_resolved",
                        ticker=ticker,
                        requested_ticker=ticker,
                        lookup_ticker=lookup_ticker,
                        lookup_strategy=lookup_strategy,
                        name=normalized,
                        source=source_name,
                    )
                    return CompanyNameResult(
                        name=normalized, source=source_name, is_resolved=True
                    )
                if raw_name:
                    logger.debug(
                        "company_name_rejected",
                        ticker=ticker,
                        requested_ticker=ticker,
                        lookup_ticker=lookup_ticker,
                        lookup_strategy=lookup_strategy,
                        raw_name=raw_name,
                        source=source_name,
                        reason="name matches lookup ticker string",
                    )
            except Exception as e:
                logger.debug(
                    "company_name_source_error",
                    ticker=ticker,
                    requested_ticker=ticker,
                    lookup_ticker=lookup_ticker,
                    lookup_strategy=lookup_strategy,
                    source=source_name,
                    error=str(e),
                )

    if allow_ibkr_probe:
        try:
            raw_name = await _try_ibkr(ticker)
            if isinstance(raw_name, str) and _is_valid_company_name(raw_name, ticker):
                normalized = normalize_company_name(raw_name)
                logger.debug(
                    "company_name_resolved",
                    ticker=ticker,
                    requested_ticker=ticker,
                    lookup_ticker=ticker,
                    lookup_strategy="ibkr_probe",
                    name=normalized,
                    source="ibkr",
                )
                return CompanyNameResult(
                    name=normalized, source="ibkr", is_resolved=True
                )
            if raw_name:
                logger.debug(
                    "company_name_rejected",
                    ticker=ticker,
                    requested_ticker=ticker,
                    lookup_ticker=ticker,
                    lookup_strategy="ibkr_probe",
                    raw_name=raw_name,
                    source="ibkr",
                    reason="name matches lookup ticker string",
                )
        except Exception as e:
            logger.debug(
                "company_name_source_error",
                ticker=ticker,
                requested_ticker=ticker,
                lookup_ticker=ticker,
                lookup_strategy="ibkr_probe",
                source="ibkr",
                error=str(e),
            )

    logger.debug(
        "company_name_unresolved",
        ticker=ticker,
        requested_ticker=ticker,
        lookup_candidates=[symbol for symbol, _strategy in lookup_candidates],
    )
    return CompanyNameResult(name=ticker, source="unresolved", is_resolved=False)
