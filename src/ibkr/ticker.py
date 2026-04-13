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
        "GBP": ".L",
        "GBX": ".L",
    }


_CURRENCY_TO_SUFFIX: dict[str, str] = _build_currency_to_suffix()


@dataclass(frozen=True, slots=True)
class Ticker:
    """Immutable value object representing an equity ticker.

    Carries the three fields that unambiguously identify an IBKR position:
      symbol   — IBKR bare symbol (e.g. "7203", "MEGP", "5") — never zero-padded
      exchange — IBKR exchange code (e.g. "TSE", "LSE", "SEHK", "SMART", "")
      currency — ISO currency code (e.g. "JPY", "GBX", "HKD", "") — optional fallback

    Derived properties (.yf, .ibkr, .suffix, .has_suffix) are computed
    on-demand from these three fields.  No network calls are ever made inside
    this class — that is the caller's responsibility.
    """

    symbol: str  # IBKR bare symbol — no exchange suffix, not zero-padded
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
        sfx = IBKR_TO_YFINANCE.get(self.exchange)
        if sfx is not None:
            # Explicit entry: "" means US (no suffix), non-empty means the exchange.
            return sfx
        # Exchange not in static map → try currency fallback
        if self.currency:
            return _CURRENCY_TO_SUFFIX.get(self.currency.upper(), "")
        return ""

    @property
    def yf(self) -> str:
        """Return yfinance-format ticker string.

        HK stocks are zero-padded to 4 digits ("0005.HK").
        US/ADR stocks have no suffix ("AAPL").
        """
        sfx = self.suffix
        if sfx == ".HK":
            bare = self.symbol.lstrip("0") or "0"
            return f"{bare.zfill(4)}.HK"
        return f"{self.symbol}{sfx}"

    @property
    def ibkr(self) -> str:
        """Return IBKR bare symbol (no suffix, no zero-padding)."""
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
                      Pre-padded HK symbols (e.g. "0005") have leading zeros
                      stripped → stored as "5"; .yf re-applies zero-padding.
            exchange: IBKR exchange code (e.g. "SEHK", "TSE", "LSE", "SMART").
                      Normalised to upper-case.
            currency: ISO currency code (e.g. "HKD", "JPY", "GBP").
                      Used as suffix fallback when exchange is unknown.
                      Normalised to upper-case.
        """
        sym = symbol.strip()
        exch = exchange.strip().upper() if exchange else ""
        ccy = currency.strip().upper() if currency else ""

        # Determine if this is an HK stock so we can strip zero-padding.
        # IBKR can occasionally send "0005" instead of "5" for SEHK positions.
        sfx = IBKR_TO_YFINANCE.get(exch)
        if sfx is None and ccy:
            sfx = _CURRENCY_TO_SUFFIX.get(ccy, "")
        if sfx == ".HK":
            sym = sym.lstrip("0") or "0"

        return cls(symbol=sym, exchange=exch, currency=ccy)

    @classmethod
    def from_yf(cls, yf_str: str, currency: str = "") -> Ticker:
        """Parse a yfinance-format ticker string into a Ticker.

        Strips HK zero-padding from the symbol component so that the stored
        symbol is always the bare IBKR form (e.g. "0005.HK" → symbol="5").
        Round-trips correctly: Ticker.from_yf("0005.HK").yf == "0005.HK".

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
            # Strip HK zero-padding: "0005" → "5" (re-applied by .yf)
            symbol = (sym_part.lstrip("0") or "0") if suffix == ".HK" else sym_part
        else:
            symbol = yf_str
            ibkr_exchange = "SMART"
        return cls(
            symbol=symbol,
            exchange=ibkr_exchange,
            currency=currency.strip().upper() if currency else "",
        )
