"""Canonical exchange metadata for ticker normalization and IBKR/yfinance mapping."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExchangeInfo:
    """Stable facts about a supported exchange suffix."""

    yf_suffix: str
    exchange_name: str
    country: str
    ibkr_code: str
    currency: str


EXCHANGES_BY_SUFFIX: dict[str, ExchangeInfo] = {
    ".SW": ExchangeInfo(".SW", "SIX Swiss Exchange", "Switzerland", "SWX", "CHF"),
    ".DE": ExchangeInfo(".DE", "XETRA", "Germany", "IBIS", "EUR"),
    ".F": ExchangeInfo(".F", "Frankfurt Stock Exchange", "Germany", "FWB", "EUR"),
    ".PA": ExchangeInfo(".PA", "Euronext Paris", "France", "SBF", "EUR"),
    ".AS": ExchangeInfo(".AS", "Euronext Amsterdam", "Netherlands", "AEB", "EUR"),
    ".BR": ExchangeInfo(".BR", "Euronext Brussels", "Belgium", "EBR", "EUR"),
    ".LS": ExchangeInfo(".LS", "Euronext Lisbon", "Portugal", "BVLP", "EUR"),
    ".MI": ExchangeInfo(".MI", "Borsa Italiana", "Italy", "BVME", "EUR"),
    ".MC": ExchangeInfo(".MC", "Bolsa de Madrid", "Spain", "BM", "EUR"),
    ".L": ExchangeInfo(".L", "London Stock Exchange", "UK", "LSE", "GBP"),
    ".T": ExchangeInfo(".T", "Tokyo Stock Exchange", "Japan", "TSE", "JPY"),
    ".HK": ExchangeInfo(".HK", "Hong Kong Stock Exchange", "Hong Kong", "SEHK", "HKD"),
    ".SS": ExchangeInfo(".SS", "Shanghai Stock Exchange", "China", "SSE", "CNY"),
    ".SZ": ExchangeInfo(".SZ", "Shenzhen Stock Exchange", "China", "SZSE", "CNY"),
    ".KS": ExchangeInfo(".KS", "Korea Stock Exchange", "South Korea", "KRX", "KRW"),
    ".KQ": ExchangeInfo(".KQ", "KOSDAQ", "South Korea", "KOSDAQ", "KRW"),
    ".TW": ExchangeInfo(".TW", "Taiwan Stock Exchange", "Taiwan", "TWSE", "TWD"),
    ".SI": ExchangeInfo(".SI", "Singapore Exchange", "Singapore", "SGX", "SGD"),
    ".BO": ExchangeInfo(".BO", "Bombay Stock Exchange", "India", "BSE", "INR"),
    ".NS": ExchangeInfo(
        ".NS", "National Stock Exchange of India", "India", "NSE", "INR"
    ),
    ".TO": ExchangeInfo(".TO", "Toronto Stock Exchange", "Canada", "TSX", "CAD"),
    ".V": ExchangeInfo(".V", "TSX Venture Exchange", "Canada", "VENTURE", "CAD"),
    ".AX": ExchangeInfo(
        ".AX", "Australian Securities Exchange", "Australia", "ASX", "AUD"
    ),
    ".NZ": ExchangeInfo(".NZ", "New Zealand Exchange", "New Zealand", "NZE", "NZD"),
    ".SA": ExchangeInfo(".SA", "B3 (Brazil)", "Brazil", "BVMF", "BRL"),
    ".MX": ExchangeInfo(".MX", "Bolsa Mexicana de Valores", "Mexico", "MEXI", "MXN"),
    ".JK": ExchangeInfo(".JK", "Indonesia Stock Exchange", "Indonesia", "IDX", "IDR"),
    ".KL": ExchangeInfo(".KL", "Bursa Malaysia", "Malaysia", "KLSE", "MYR"),
    ".BK": ExchangeInfo(".BK", "Stock Exchange of Thailand", "Thailand", "SET", "THB"),
    ".OL": ExchangeInfo(".OL", "Oslo Børs", "Norway", "OSL", "NOK"),
    ".ST": ExchangeInfo(".ST", "Nasdaq Stockholm", "Sweden", "STO", "SEK"),
    ".HE": ExchangeInfo(".HE", "Nasdaq Helsinki", "Finland", "HEL", "EUR"),
    ".CO": ExchangeInfo(".CO", "Nasdaq Copenhagen", "Denmark", "CPH", "DKK"),
    ".VI": ExchangeInfo(".VI", "Wiener Börse", "Austria", "WBAG", "EUR"),
    ".WA": ExchangeInfo(".WA", "Warsaw Stock Exchange", "Poland", "WSE", "PLN"),
    ".PR": ExchangeInfo(".PR", "Prague Stock Exchange", "Czech Republic", "PSE", "CZK"),
    ".BD": ExchangeInfo(".BD", "Budapest Stock Exchange", "Hungary", "BSE2", "HUF"),
    ".RO": ExchangeInfo(".RO", "Bucharest Stock Exchange", "Romania", "BVB", "RON"),
    ".TWO": ExchangeInfo(".TWO", "Taipei Exchange", "Taiwan", "TPEX", "TWD"),
}

NORMALIZATION_SUFFIX_ALIASES: dict[str, str] = {
    **{suffix.lstrip("."): suffix for suffix in EXCHANGES_BY_SUFFIX},
    "SWX": ".SW",
    "VX": ".SW",
}

IBKR_EXCHANGE_ALIASES: dict[str, str] = {
    "IBIS2": ".DE",
    "FWB2": ".F",
    "EBS": ".SW",
    "SIBE": ".MC",
    "ENEXT.BE": ".BR",
    "LSEETF": ".L",
    "OSE.JPN": ".T",
    "VSE": ".VI",
}

US_IBKR_EXCHANGES: dict[str, str] = {
    "NASDAQ": "",
    "NYSE": "",
    "ARCA": "",
    "AMEX": "",
    "SMART": "",
    "IEXG": "",
    "CBOE": "",
}

SUFFIX_TO_CURRENCY_CODE: dict[str, str] = {
    suffix: info.currency for suffix, info in EXCHANGES_BY_SUFFIX.items()
}

YFINANCE_TO_IBKR: dict[str, str] = {
    suffix: info.ibkr_code for suffix, info in EXCHANGES_BY_SUFFIX.items()
}
YFINANCE_TO_IBKR[""] = "SMART"

IBKR_TO_YFINANCE: dict[str, str] = (
    {info.ibkr_code: suffix for suffix, info in EXCHANGES_BY_SUFFIX.items()}
    | IBKR_EXCHANGE_ALIASES
    | US_IBKR_EXCHANGES
)


def canonical_suffix_for_token(token: str) -> str | None:
    """Return the canonical yfinance suffix for a suffix token like 'SWX' or 'TWO'."""
    cleaned = token.strip().upper().lstrip(".")
    if not cleaned:
        return None
    return NORMALIZATION_SUFFIX_ALIASES.get(cleaned)


def _validate_exchange_metadata() -> None:
    canonical_codes = [info.ibkr_code for info in EXCHANGES_BY_SUFFIX.values()]
    if len(canonical_codes) != len(set(canonical_codes)):
        raise ValueError("Duplicate canonical ibkr_code values")

    for suffix, info in EXCHANGES_BY_SUFFIX.items():
        if not suffix.startswith("."):
            raise ValueError(f"Malformed suffix key: {suffix}")
        if suffix != info.yf_suffix:
            raise ValueError(
                f"Key {suffix} does not match record suffix {info.yf_suffix}"
            )
        if not info.ibkr_code or info.ibkr_code != info.ibkr_code.upper():
            raise ValueError(f"Canonical ibkr_code must be uppercase: {info.ibkr_code}")
        if not info.currency:
            raise ValueError(f"Missing currency for canonical exchange {suffix}")

    overlap = set(IBKR_EXCHANGE_ALIASES) & set(canonical_codes)
    if overlap:
        raise ValueError(f"IBKR alias overlaps canonical codes: {sorted(overlap)}")

    for key, suffix in NORMALIZATION_SUFFIX_ALIASES.items():
        if key != key.upper():
            raise ValueError(f"Normalization suffix alias must be uppercase: {key}")
        if suffix not in EXCHANGES_BY_SUFFIX:
            raise ValueError(f"Alias {key} points to unknown suffix {suffix}")

    for key, suffix in IBKR_EXCHANGE_ALIASES.items():
        if key != key.upper():
            raise ValueError(f"IBKR alias must be uppercase: {key}")
        if suffix not in EXCHANGES_BY_SUFFIX:
            raise ValueError(f"IBKR alias {key} points to unknown suffix {suffix}")

    for key, suffix in US_IBKR_EXCHANGES.items():
        if key != key.upper():
            raise ValueError(f"US IBKR exchange must be uppercase: {key}")
        if suffix != "":
            raise ValueError(f"US IBKR exchange {key} must map to empty suffix")


_validate_exchange_metadata()
