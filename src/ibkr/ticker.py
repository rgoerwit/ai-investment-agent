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

from src.ticker_utils import TickerFormatter

# Single-exchange currencies that unambiguously resolve to one yfinance suffix.
# EUR, CHF, CAD are intentionally omitted — each spans multiple exchanges.
_CURRENCY_TO_SUFFIX: dict[str, str] = {
    "HKD": ".HK",  # Hong Kong dollar → SEHK
    "JPY": ".T",  # Japanese yen → TSE
    "TWD": ".TW",  # Taiwan dollar → TWSE
    "KRW": ".KS",  # Korean won → KRX
    "SGD": ".SI",  # Singapore dollar → SGX
    "AUD": ".AX",  # Australian dollar → ASX
    "NZD": ".NZ",  # New Zealand dollar → NZX
    "BRL": ".SA",  # Brazilian real → B3
    "MXN": ".MX",  # Mexican peso → BMV
    "MYR": ".KL",  # Malaysian ringgit → Bursa Malaysia
    "PLN": ".WA",  # Polish złoty → Warsaw Stock Exchange
    "SEK": ".ST",  # Swedish krona → Nasdaq Stockholm
    "NOK": ".OL",  # Norwegian krone → Oslo Børs
    "DKK": ".CO",  # Danish krone → Nasdaq Copenhagen
    # GBX and GBP: London Stock Exchange only. IBKR stores LSE prices in GBP
    # (pounds); portfolio.py multiplies by 100 to convert to GBX (pence) and
    # sets currency="GBX".  Both map to ".L" — GBP is included as defence-in-
    # depth in case a position arrives before the ×100 conversion.
    "GBX": ".L",  # British pence (IBKR-normalized) → LSE
    "GBP": ".L",  # British pound → LSE
}


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
        1. TickerFormatter.IBKR_TO_YFINANCE[exchange] — static, authoritative.
           Returns "" for US venues (NASDAQ, NYSE, SMART, …) — that is a valid
           result meaning "no suffix".  Returns None (missing key) for completely
           unknown exchange codes → fall through to step 2.
        2. _CURRENCY_TO_SUFFIX[currency] — fallback for unambiguous single-country
           currencies when the exchange code is unknown.
        3. "" — US/ADR or genuinely unresolvable.
        """
        sfx = TickerFormatter.IBKR_TO_YFINANCE.get(self.exchange)
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
        sfx = TickerFormatter.IBKR_TO_YFINANCE.get(exch)
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
            ibkr_exchange = TickerFormatter.YFINANCE_TO_IBKR.get(suffix, "SMART")
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
