"""Shared reconciliation rules and ticker/FX helper logic."""

from __future__ import annotations

import structlog

from src.exchange_metadata import IBKR_TO_YFINANCE
from src.fx_normalization import get_fx_rate_fallback
from src.ibkr.models import AnalysisRecord, NormalizedPosition

logger = structlog.get_logger(__name__)


def _resolve_fx(analysis: AnalysisRecord) -> float:
    """Return FX rate (local → USD) for an analysis, with fallback chain."""
    currency = (analysis.currency or "USD").strip().upper()
    saved = analysis.fx_rate_to_usd

    if saved is not None:
        if saved == 1.0 and currency not in ("USD", ""):
            fallback = get_fx_rate_fallback(currency)
            if fallback is not None:
                logger.warning(
                    "fx_rate_saved_1_overridden",
                    ticker=analysis.ticker,
                    currency=currency,
                    fallback_rate=fallback,
                    msg="Saved fx_rate=1.0 for non-USD currency replaced with fallback "
                    "(legacy snapshot; re-run analysis to persist correct rate)",
                )
                return fallback
        return saved

    if currency in ("USD", ""):
        return 1.0
    rate = get_fx_rate_fallback(currency)
    if rate is not None:
        logger.warning(
            "fx_rate_missing_using_fallback",
            ticker=analysis.ticker,
            currency=currency,
            fallback_rate=rate,
            msg="Analysis snapshot missing fx_rate_to_usd — cost/quantity estimates approximate",
        )
        return rate
    logger.error(
        "fx_rate_unknown",
        ticker=analysis.ticker,
        currency=currency,
        msg="No FX rate available; cost/quantity will be wrong — re-run analysis to fix",
    )
    return 1.0


_MIN_ORDER_USD: float = 200.0

_EXCHANGE_LONG_NAMES: dict[str, str] = {
    "HK": "Hong Kong",
    "T": "Japan",
    "KS": "Korea",
    "KQ": "Korea KOSDAQ",
    "TW": "Taiwan",
    "TWO": "Taiwan OTC",
    "AS": "Amsterdam",
    "DE": "Germany",
    "PA": "France",
    "L": "UK",
    "SS": "Shanghai",
    "SZ": "Shenzhen",
    "SI": "Singapore",
    "US": "United States",
    "MX": "Mexico",
    "MC": "Madrid",
    "AX": "Australia",
    "KL": "Malaysia",
    "VI": "Vienna",
    "WA": "Poland",
    "ST": "Sweden",
    "OL": "Norway",
    "CO": "Denmark",
    "SA": "Brazil",
    "JK": "Indonesia",
    "BK": "Thailand",
    "BO": "India BSE",
    "NS": "India NSE",
    "TO": "Canada",
    "V": "Canada TSXV",
    "NZ": "New Zealand",
    "SW": "Switzerland",
    "F": "Frankfurt",
    "BR": "Belgium",
    "LS": "Lisbon",
    "MI": "Milan",
    "EUR": "Europe (EUR)",
}


def _exchange_from_ticker(ticker: str) -> str:
    """Infer exchange code from yfinance ticker suffix (e.g. '0005.HK' → 'HK')."""
    if "." not in ticker:
        return "US"
    return ticker.rsplit(".", 1)[-1].upper()


def _exchange_from_position(pos: NormalizedPosition) -> str:
    """Derive a short exchange code from a NormalizedPosition."""
    yf_str = pos.ticker.yf
    if "." in yf_str:
        return yf_str.rsplit(".", 1)[-1].upper()

    if pos.ticker.exchange:
        ibkr_suffix = IBKR_TO_YFINANCE.get(pos.ticker.exchange, None)
        if ibkr_suffix is not None:
            return ibkr_suffix.lstrip(".") if ibkr_suffix else "US"

    currency_to_exchange: dict[str, str] = {
        "HKD": "HK",
        "JPY": "T",
        "TWD": "TW",
        "KRW": "KS",
        "SGD": "SI",
        "AUD": "AX",
        "NZD": "NZ",
        "BRL": "SA",
        "MXN": "MX",
        "MYR": "KL",
        "PLN": "WA",
        "SEK": "ST",
        "NOK": "OL",
        "DKK": "CO",
        "GBP": "L",
        "GBX": "L",
        "CAD": "TO",
        "CHF": "SW",
        "EUR": "EUR",
    }
    if pos.currency:
        code = currency_to_exchange.get(pos.currency.upper(), "")
        if code:
            return code

    return "US"


def _normalize_verdict(raw: str) -> str:
    """Normalise a verdict string to canonical UPPER_SNAKE_CASE."""
    normed = raw.strip().replace(" ", "_").upper()
    if normed == "DO":
        return "DO_NOT_INITIATE"
    return normed


_REJECT_VERDICTS = frozenset({"DO_NOT_INITIATE", "SELL", "REJECT"})


def check_staleness(
    analysis: AnalysisRecord,
    current_price_local: float | None = None,
    max_age_days: int = 14,
    drift_threshold_pct: float = 15.0,
    structural_macro_events: list | None = None,
) -> tuple[bool, str]:
    """Check if an analysis is stale and should be reviewed."""
    reasons = []

    if analysis.age_days > max_age_days:
        age_str = "no date" if analysis.age_days >= 9999 else f"{analysis.age_days}d"
        reasons.append(f"age {age_str} > {max_age_days}d limit")

    entry_price = analysis.entry_price or analysis.current_price
    if entry_price and current_price_local and entry_price > 0:
        drift_pct = abs((current_price_local - entry_price) / entry_price) * 100
        if drift_pct > drift_threshold_pct:
            direction = "up" if current_price_local > entry_price else "down"
            reasons.append(f"price drift {drift_pct:.1f}% {direction}")

    if structural_macro_events and analysis.analysis_date:
        for event in structural_macro_events:
            if event.event_date > analysis.analysis_date:
                if event.scope == "GLOBAL":
                    reasons.append(
                        f"STRUCTURAL macro event ({event.news_headline[:40]!r}) "
                        f"detected after analysis"
                    )
                    break
                ticker = getattr(analysis, "ticker", "") or ""
                dot = ticker.rfind(".")
                suffix = ticker[dot:] if dot >= 0 else ""
                if suffix and suffix == event.primary_region:
                    reasons.append(
                        f"STRUCTURAL macro event ({event.news_headline[:40]!r}) "
                        f"in your region ({suffix}) after analysis"
                    )
                    break

    if reasons:
        return True, "; ".join(reasons)
    return False, ""


def check_stop_breach(
    analysis: AnalysisRecord,
    current_price_local: float,
) -> bool:
    """Check if current price has breached the stop-loss level."""
    stop = analysis.stop_price
    if stop and current_price_local > 0:
        ratio = stop / current_price_local
        if ratio > 50 or ratio < 0.02:
            logger.warning(
                "stop_price_ratio_suspicious",
                ticker=analysis.ticker,
                stop=stop,
                current_price=current_price_local,
                ratio=f"{ratio:.1f}x",
                hint="Possible currency-unit mismatch or stale stop from different analysis",
            )
            return False
        return current_price_local < stop
    return False


def check_target_hit(
    analysis: AnalysisRecord,
    current_price_local: float,
) -> bool:
    """Check if current price has hit or exceeded TARGET_1."""
    target = analysis.target_1_price
    if target and current_price_local > 0:
        return current_price_local >= target
    return False


def _settlement_date(business_days: int) -> str:
    """Return settlement date as YYYY-MM-DD, skipping weekends."""
    from datetime import date, timedelta

    d = date.today()
    added = 0
    while added < business_days:
        d += timedelta(days=1)
        if d.weekday() < 5:
            added += 1
    return d.isoformat()


def _classify_sell_type(analysis: AnalysisRecord | None, stop_breached: bool) -> str:
    """Classify why a position is being sold."""
    if stop_breached:
        return "STOP_BREACH"
    if analysis is None:
        return "HARD_REJECT"
    health_ok = (analysis.health_adj or 0.0) >= 50.0
    growth_ok = (analysis.growth_adj or 0.0) >= 50.0
    return "SOFT_REJECT" if (health_ok and growth_ok) else "HARD_REJECT"
