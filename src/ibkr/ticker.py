"""
Ticker value object for the IBKR integration layer.

Carries (symbol, exchange, currency) together and derives both IBKR and
yfinance string representations on demand.  Lives exclusively inside
src/ibkr/ — everything outside this layer (agents, analysis pipeline,
AnalysisRecord, analyses dicts) continues to use plain yfinance strings.

Boundary rule: when IBKR-layer code needs to look up an AnalysisRecord,
it calls ticker.yf to get the string key.  That is the only crossing point.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.exchange_metadata import (
    IBKR_TO_YFINANCE,
    SUFFIX_TO_CURRENCY_CODE,
    YFINANCE_TO_IBKR,
)

_NUMERIC_SYMBOL_WIDTH_BY_SUFFIX: dict[str, int] = {
    ".HK": 4,
    ".KS": 6,
    ".KQ": 6,
}


def _pad_numeric_symbol_for_suffix(symbol: str, suffix: str) -> str:
    """Return the exchange-canonical numeric symbol, leaving mixed codes intact."""
    cleaned = symbol.strip()
    width = _NUMERIC_SYMBOL_WIDTH_BY_SUFFIX.get(suffix)
    # Some exchange instruments are not pure numeric; never pad those.
    if width and cleaned.isdigit():
        return cleaned.zfill(width)
    return cleaned


def _strip_storage_padding_for_suffix(symbol: str, suffix: str) -> str:
    """Return the display/lookup symbol used inside the IBKR layer."""
    cleaned = symbol.strip()
    if suffix == ".HK" and cleaned.isdigit():
        return cleaned.lstrip("0") or "0"
    if suffix in {".KS", ".KQ"}:
        return _pad_numeric_symbol_for_suffix(cleaned, suffix)
    return cleaned


def _build_currency_to_suffix() -> dict[str, str]:
    """Build the IBKR currency fallback map from canonical exchange facts."""
    grouped: dict[str, set[str]] = {}
    for suffix, currency in SUFFIX_TO_CURRENCY_CODE.items():
        grouped.setdefault(currency, set()).add(suffix)

    derived = {
        currency: sorted(suffixes)[0]
        for currency, suffixes in grouped.items()
        if len(suffixes) == 1
    }

    # Shared-currency exchanges need explicit policy rather than arbitrary inversion.
    return {
        **derived,
        "TWD": ".TW",
        "KRW": ".KS",
        "GBP": ".L",
        "GBX": ".L",
    }


_CURRENCY_TO_SUFFIX: dict[str, str] = _build_currency_to_suffix()


def _suffix_for_exchange_currency(exchange: str, currency: str) -> str:
    """Resolve yfinance suffix from exact exchange first, then currency fallback."""
    sfx = IBKR_TO_YFINANCE.get(exchange)
    if sfx is not None:
        if sfx == "" and exchange in ("", "SMART") and currency not in ("", "USD"):
            # IBKR contract-info endpoints can report SMART for non-US watchlist
            # contracts; keep this aligned with ibkr_symbol_to_yf() fallback behavior.
            return _CURRENCY_TO_SUFFIX.get(currency, "")
        return sfx
    if currency:
        return _CURRENCY_TO_SUFFIX.get(currency, "")
    return ""


@dataclass(frozen=True, slots=True)
class Ticker:
    """Immutable value object representing an equity ticker.

    Carries the three fields that unambiguously identify an IBKR position:
      symbol   — IBKR symbol (e.g. "7203", "MEGP", "5", "005930")
      exchange — IBKR exchange code (e.g. "TSE", "LSE", "SEHK", "SMART", "")
      currency — ISO currency code (e.g. "JPY", "GBX", "HKD", "") — optional fallback

    Derived properties (.yf, .ibkr, .suffix, .has_suffix) are computed
    on-demand from these three fields.  No network calls are ever made inside
    this class — that is the caller's responsibility.
    """

    symbol: str  # IBKR symbol — no exchange suffix
    exchange: str  # IBKR exchange code (upper-case)
    currency: str  # ISO currency code (upper-case) — used only as suffix fallback

    @property
    def suffix(self) -> str:
        """Return the yfinance exchange suffix (e.g. '.HK', '.T', '').

        Lookup order:
        1. IBKR_TO_YFINANCE[exchange] — static, authoritative.
           Returns "" for US venues (NASDAQ, NYSE, SMART, …) — that is a valid
           result meaning "no suffix".  Returns None (missing key) for completely
           unknown exchange codes → fall through to step 2.
        2. _CURRENCY_TO_SUFFIX[currency] — fallback for unambiguous single-country
           currencies when the exchange code is unknown.
        3. "" — US/ADR or genuinely unresolvable.
        """
        return _suffix_for_exchange_currency(self.exchange, self.currency.upper())

    @property
    def yf(self) -> str:
        """Return yfinance-format ticker string.

        HK stocks are zero-padded to 4 digits ("0005.HK").
        Korean stocks are zero-padded to 6 digits ("005930.KS").
        US/ADR stocks have no suffix ("AAPL").
        """
        sfx = self.suffix
        return f"{_pad_numeric_symbol_for_suffix(self.symbol, sfx)}{sfx}"

    @property
    def ibkr(self) -> str:
        """Return the IBKR symbol used for display and lookup."""
        return self.symbol

    @property
    def has_suffix(self) -> bool:
        """True when the ticker has a non-empty yfinance exchange suffix."""
        return bool(self.suffix)

    @property
    def exchange_resolved(self) -> bool:
        """True when the exchange code is explicitly in the IBKR→yfinance map.

        A US ticker (SMART / NASDAQ / NYSE) resolves to suffix "" — that is a
        *known* result, not a missing one.  This property distinguishes between
        "intentionally no suffix (US stock)" and "suffix unknown (unrecognised
        exchange code)".  Use it to suppress false ⚠ suffix warnings for US
        equities.
        """
        return IBKR_TO_YFINANCE.get(self.exchange) is not None

    def __str__(self) -> str:
        return self.yf

    # ── Constructors ──────────────────────────────────────────────────────────

    @classmethod
    def from_ibkr(
        cls,
        symbol: str,
        exchange: str = "",
        currency: str = "",
    ) -> Ticker:
        """Build a Ticker from raw IBKR API fields.

        Args:
            symbol:   IBKR bare symbol (e.g. "5", "7203", "ASML").
                      Pre-padded numeric symbols for HK/Korea (e.g. "0005",
                      "005930") have leading zeros stripped; .yf re-applies
                      exchange-canonical zero-padding.
            exchange: IBKR exchange code (e.g. "SEHK", "TSE", "LSE", "SMART").
                      Normalised to upper-case.
            currency: ISO currency code (e.g. "HKD", "JPY", "GBP").
                      Used as suffix fallback when exchange is unknown.
                      Normalised to upper-case.
        """
        sym = symbol.strip()
        exch = exchange.strip().upper() if exchange else ""
        ccy = currency.strip().upper() if currency else ""

        # IBKR can occasionally send pre-padded numeric symbols; store the
        # bare form and let .yf re-apply exchange-canonical padding.
        sfx = _suffix_for_exchange_currency(exch, ccy)
        sym = _strip_storage_padding_for_suffix(sym, sfx)

        return cls(symbol=sym, exchange=exch, currency=ccy)

    @classmethod
    def from_yf(cls, yf_str: str, currency: str = "") -> Ticker:
        """Parse a yfinance-format ticker string into a Ticker.

        Normalizes exchange-specific numeric symbols for IBKR display/lookup.
        HK uses the bare symbol ("0005.HK" → "5"); Korea uses fixed-width symbols
        ("5930.KS" → "005930"). Round-trips correctly:
        Ticker.from_yf("0005.HK").yf == "0005.HK".

        Args:
            yf_str:   yfinance ticker (e.g. "7203.T", "0005.HK", "AAPL").
            currency: Optional ISO currency code to attach (used as fallback
                      if the exchange suffix is later unknown).
        """
        yf_str = yf_str.strip()
        if "." in yf_str:
            sym_part, sfx_part = yf_str.rsplit(".", 1)
            suffix = f".{sfx_part}"
            ibkr_exchange = YFINANCE_TO_IBKR.get(suffix, "SMART")
            symbol = _strip_storage_padding_for_suffix(sym_part, suffix)
        else:
            symbol = yf_str
            ibkr_exchange = "SMART"
        return cls(
            symbol=symbol,
            exchange=ibkr_exchange,
            currency=currency.strip().upper() if currency else "",
        )
