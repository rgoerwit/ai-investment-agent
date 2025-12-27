"""
International Ticker Utilities
Updated: Removed brittle hardcoded maps in favor of dynamic name normalization
and strict search query generation.
"""

import re

import structlog

logger = structlog.get_logger(__name__)

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

    clean_name = clean_name.strip()

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

    # Common exchange suffix mappings
    # Format: "exchange_code": ("yfinance_suffix", "exchange_name", "country", "ibkr_code")
    EXCHANGE_SUFFIXES = {
        # European exchanges
        "SW": (".SW", "SIX Swiss Exchange", "Switzerland", "SWX"),
        "SWX": (".SW", "SIX Swiss Exchange", "Switzerland", "SWX"),
        "VX": (".SW", "SIX Swiss Exchange", "Switzerland", "SWX"),
        "DE": (".DE", "XETRA", "Germany", "IBIS"),
        "F": (".F", "Frankfurt Stock Exchange", "Germany", "FWB"),
        "PA": (".PA", "Euronext Paris", "France", "SBF"),
        "AS": (".AS", "Euronext Amsterdam", "Netherlands", "AEB"),
        "BR": (".BR", "Euronext Brussels", "Belgium", "EBR"),
        "LS": (".LS", "Euronext Lisbon", "Portugal", "BVLP"),
        "MI": (".MI", "Borsa Italiana", "Italy", "BVME"),
        "MC": (".MC", "Bolsa de Madrid", "Spain", "BM"),
        "L": (".L", "London Stock Exchange", "UK", "LSE"),
        # Asian exchanges
        "T": (".T", "Tokyo Stock Exchange", "Japan", "TSE"),
        "HK": (".HK", "Hong Kong Stock Exchange", "Hong Kong", "SEHK"),
        "SS": (".SS", "Shanghai Stock Exchange", "China", "SSE"),
        "SZ": (".SZ", "Shenzhen Stock Exchange", "China", "SZSE"),
        "KS": (".KS", "Korea Stock Exchange", "South Korea", "KRX"),
        "KQ": (".KQ", "KOSDAQ", "South Korea", "KOSDAQ"),
        "TW": (".TW", "Taiwan Stock Exchange", "Taiwan", "TWSE"),
        "SI": (".SI", "Singapore Exchange", "Singapore", "SGX"),
        "BO": (".BO", "Bombay Stock Exchange", "India", "BSE"),
        "NS": (".NS", "National Stock Exchange of India", "India", "NSE"),
        # Other major exchanges
        "TO": (".TO", "Toronto Stock Exchange", "Canada", "TSX"),
        "V": (".V", "TSX Venture Exchange", "Canada", "VENTURE"),
        "AX": (".AX", "Australian Securities Exchange", "Australia", "ASX"),
        "NZ": (".NZ", "New Zealand Exchange", "New Zealand", "NZE"),
        "SA": (".SA", "B3 (Brazil)", "Brazil", "BVMF"),
        "MX": (".MX", "Bolsa Mexicana de Valores", "Mexico", "MEXI"),
        "JK": (".JK", "Indonesia Stock Exchange", "Indonesia", "IDX"),
        "KL": (".KL", "Bursa Malaysia", "Malaysia", "KLSE"),
        "BK": (".BK", "Stock Exchange of Thailand", "Thailand", "SET"),
    }

    # IBKR exchange code to yfinance suffix mapping
    IBKR_TO_YFINANCE = {
        "SWX": ".SW",
        "IBIS": ".DE",
        "FWB": ".F",
        "SBF": ".PA",
        "AEB": ".AS",
        "EBR": ".BR",
        "BVLP": ".LS",
        "BVME": ".MI",
        "BM": ".MC",
        "LSE": ".L",
        "TSE": ".T",
        "SEHK": ".HK",
        "SSE": ".SS",
        "SZSE": ".SZ",
        "KRX": ".KS",
        "KOSDAQ": ".KQ",
        "TWSE": ".TW",
        "SGX": ".SI",
        "BSE": ".BO",
        "NSE": ".NS",
        "TSX": ".TO",
        "VENTURE": ".V",
        "ASX": ".AX",
        "NZE": ".NZ",
        "BVMF": ".SA",
        "MEXI": ".MX",
        "IDX": ".JK",
        "KLSE": ".KL",
        "SET": ".BK",
        "NASDAQ": "",
        "NYSE": "",
        "ARCA": "",
        "AMEX": "",
        "SMART": "",
    }

    # Reverse mapping: yfinance to IBKR
    YFINANCE_TO_IBKR = {v: k for k, v in IBKR_TO_YFINANCE.items() if v}
    YFINANCE_TO_IBKR[""] = "SMART"

    # Alternative ticker format patterns
    TICKER_PATTERNS = {
        "reuters": re.compile(r"^([A-Z0-9]+)\.([A-Z]+)-([A-Z]{2})$"),
        "standard": re.compile(r"^([A-Z0-9]+)\.([A-Z]+)$"),
        "plain": re.compile(r"^([A-Z0-9]+)$"),
        "ibkr": re.compile(r"^([A-Z0-9]+):([A-Z]+)$"),
    }

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
            exchange_mapping = cls._map_reuters_to_exchange(reuters_code, country_code)

            if exchange_mapping:
                if target_format == "yfinance":
                    normalized = f"{symbol}{exchange_mapping[0]}"
                elif target_format == "ibkr":
                    normalized = f"{symbol}:{exchange_mapping[3]}"
                else:
                    normalized = ticker

                metadata = {
                    "original": original_ticker,
                    "symbol": symbol,
                    "exchange_suffix": exchange_mapping[0],
                    "exchange_name": exchange_mapping[1],
                    "country": exchange_mapping[2],
                    "ibkr_exchange": exchange_mapping[3],
                    "format": "reuters",
                }
                return normalized, metadata

        # Try standard format (e.g., "NOVN.SW")
        standard_match = cls.TICKER_PATTERNS["standard"].match(ticker)
        if standard_match:
            symbol, suffix = standard_match.groups()

            if suffix in cls.EXCHANGE_SUFFIXES:
                exchange_info = cls.EXCHANGE_SUFFIXES[suffix]

                if target_format == "yfinance":
                    normalized = f"{symbol}{exchange_info[0]}"
                elif target_format == "ibkr":
                    normalized = f"{symbol}:{exchange_info[3]}"
                else:
                    normalized = ticker

                metadata = {
                    "original": original_ticker,
                    "symbol": symbol,
                    "exchange_suffix": exchange_info[0],
                    "exchange_name": exchange_info[1],
                    "country": exchange_info[2],
                    "ibkr_exchange": exchange_info[3],
                    "format": "standard",
                }
                return normalized, metadata
            else:
                normalized = ticker if target_format == "yfinance" else ticker
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
        for _suffix_key, info in cls.EXCHANGE_SUFFIXES.items():
            if info[3] == exchange:
                if target_format == "yfinance":
                    normalized = f"{symbol}{info[0]}"
                elif target_format == "ibkr":
                    normalized = f"{symbol}:{exchange}"
                else:
                    normalized = f"{symbol}:{exchange}"

                metadata = {
                    "original": original_ticker,
                    "symbol": symbol,
                    "exchange_suffix": info[0],
                    "exchange_name": info[1],
                    "country": info[2],
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
    ) -> tuple[str, str, str, str] | None:
        """Map Reuters exchange codes to exchange info (yfinance_suffix, name, country, ibkr_code)."""
        reuters_mapping = {
            "N": {
                "CH": ("SW", "SIX Swiss Exchange", "Switzerland", "SWX"),
                "DE": ("DE", "XETRA", "Germany", "IBIS"),
                "FR": ("PA", "Euronext Paris", "France", "SBF"),
                "JP": ("T", "Tokyo Stock Exchange", "Japan", "TSE"),
                "GB": ("L", "London Stock Exchange", "UK", "LSE"),
            },
            "O": {
                "CH": ("SW", "SIX Swiss Exchange", "Switzerland", "SWX"),
            },
            "S": {
                "CH": ("SW", "SIX Swiss Exchange", "Switzerland", "SWX"),
            },
            "VX": {
                "CH": ("SW", "SIX Swiss Exchange", "Switzerland", "SWX"),
            },
        }

        if (
            reuters_code in reuters_mapping
            and country_code in reuters_mapping[reuters_code]
        ):
            suffix, name, country, ibkr = reuters_mapping[reuters_code][country_code]
            return (
                f".{suffix}" if not suffix.startswith(".") else suffix,
                name,
                country,
                ibkr,
            )

        return None

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
