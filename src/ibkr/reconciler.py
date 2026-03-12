"""
Reconciler: Compare IBKR positions vs evaluator recommendations.

The equity evaluator produces one-off BUY/SELL/DNI verdicts per ticker,
unaware of existing positions. This module bridges that gap by:

1. Loading latest analysis JSONs from results/
2. Reading live IBKR positions
3. Generating position-aware actions:
   - Evaluator says BUY + not held       → BUY (new position)
   - Evaluator says BUY + already held    → HOLD or ADD (if underweight)
   - Evaluator says DNI/SELL + held       → SELL/CLOSE (exit position)
   - Evaluator says BUY + stale analysis  → REVIEW (needs re-analysis)
   - Held position + no analysis          → REVIEW (unknown status)
   - Position overweight vs target        → TRIM
   - Target hit                           → REVIEW (take profit?)
   - Stop breached                        → SELL (urgent)
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import structlog

from src.fx_normalization import get_fx_rate_fallback
from src.ibkr.models import (
    AnalysisRecord,
    NormalizedPosition,
    PortfolioSummary,
    ReconciliationItem,
    TradeBlockData,
)
from src.ibkr.order_builder import (
    calculate_quantity,
    parse_trade_block,
    round_to_lot_size,
)
from src.ibkr.ticker import Ticker
from src.ticker_utils import TickerFormatter

logger = structlog.get_logger(__name__)


def _resolve_fx(analysis: AnalysisRecord) -> float:
    """Return FX rate (local → USD) for an analysis, with fallback chain.

    Priority:
    1. Saved fx_rate_to_usd from the analysis snapshot — UNLESS it equals exactly
       1.0 for a non-USD currency, which indicates the legacy bogus fallback that was
       stored before SEK/NOK/PLN/etc. were added to FALLBACK_RATES_TO_USD.  In that
       case, fall through to the fallback table so old snapshots are self-healing.
    2. Hardcoded fallback table (get_fx_rate_fallback) keyed by analysis.currency.
    3. 1.0 (last resort — logs an error so the user knows cost figures may be wrong).
    """
    currency = (analysis.currency or "USD").strip().upper()
    saved = analysis.fx_rate_to_usd

    if saved is not None:
        # Guard against legacy bogus rate: before the fix, unknown currencies (SEK,
        # NOK, PLN, etc.) were saved as 1.0 via `or 1.0` in retrospective.py.
        # For any non-USD currency, 1.0 is implausible — prefer the fallback table.
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


# Minimum USD order value — orders below this are not worth recommending.
_MIN_ORDER_USD: float = 200.0

# Human-readable exchange names for the concentration section.
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
    """
    Derive a short exchange code (e.g. 'T', 'HK', 'KL') from a NormalizedPosition.

    Priority:
    1. ticker.yf suffix — always correct when normalize_positions() set it via
       Ticker.from_ibkr() (which already consulted IBKR_TO_YFINANCE).
    2. IBKR listingExchange (ticker.exchange) via TickerFormatter.IBKR_TO_YFINANCE —
       corrects cases where an analysis was run without an exchange suffix, causing
       the ticker-based inference to fall through to the wrong "US" default.
    3. Currency heuristic — handles IBKR exchanges not in the static map (e.g. some
       regional venues) when the currency is unambiguous (SEK→ST, MYR→KL, etc.).
    4. "US" fallback.

    No network calls are made.
    """
    yf_str = pos.ticker.yf
    # 1. Suffix present on ticker.yf — most positions satisfy this condition
    if "." in yf_str:
        return yf_str.rsplit(".", 1)[-1].upper()

    # 2. IBKR listingExchange via static map
    if pos.ticker.exchange:
        ibkr_suffix = TickerFormatter.IBKR_TO_YFINANCE.get(pos.ticker.exchange, None)
        if ibkr_suffix is not None:
            # Empty string means a US venue (NASDAQ, NYSE, etc.)
            return ibkr_suffix.lstrip(".") if ibkr_suffix else "US"

    # 3. Currency heuristic.
    #    Single-exchange currencies (HKD, JPY, etc.) are truly unambiguous.
    #    Regionally-specific currencies (GBP, CAD, CHF) have multiple exchanges
    #    within the same country but are geographically unambiguous — far better
    #    to label them as UK/Canada/Switzerland than to miscount them as US.
    #    EUR is multi-country but unmistakably non-US; grouped as "EUR".
    _CURRENCY_TO_EXCHANGE: dict[str, str] = {
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
        # Regionally-specific (non-US) currencies — group by country/region
        "GBP": "L",  # UK (LSE / AIM)
        "GBX": "L",  # UK pence
        "CAD": "TO",  # Canada (TSX / TSXV)
        "CHF": "SW",  # Switzerland (SIX)
        "EUR": "EUR",  # Europe (multi-country; better than "US")
    }
    if pos.currency:
        code = _CURRENCY_TO_EXCHANGE.get(pos.currency.upper(), "")
        if code:
            return code

    return "US"


# ══════════════════════════════════════════════════════════════════════════════
# Analysis Loading
# ══════════════════════════════════════════════════════════════════════════════


def _normalize_verdict(raw: str) -> str:
    """Normalise a verdict string to canonical UPPER_SNAKE_CASE.

    The PM occasionally outputs "DO NOT INITIATE" (spaces) instead of
    "DO_NOT_INITIATE".  Both forms must compare equal in Phase 1.5 checks.

    It also sometimes writes "DO\nNOT INITIATE" (newline between words),
    causing the verdict regex to capture only the first word "DO".  Map that
    truncation back to the full form so routing and display stay correct.
    """
    normed = raw.strip().replace(" ", "_").upper()
    if normed == "DO":
        return "DO_NOT_INITIATE"
    return normed


_REJECT_VERDICTS = frozenset({"DO_NOT_INITIATE", "SELL", "REJECT"})


def _parse_scores_from_final_decision(text: str) -> dict:
    """Extract health_adj, growth_adj, verdict, zone from a PM final_decision narrative.

    Handles two legacy text formats for analyses that predate prediction_snapshot:

    Mid-era (structured fields in text):
        HEALTH_ADJ: 79  /  GROWTH_ADJ: 83  /  VERDICT: BUY  /  ZONE: MODERATE

    Old-era (prose narrative):
        Financial Health: 70.8% (Adjusted)  /  Growth Transition: 66.7% (Adjusted)
        **Action**: **BUY**
    """
    result: dict = {}

    # ── Health score ──────────────────────────────────────────────────────────
    m = re.search(r"\bHEALTH_ADJ[:\s]+([0-9.]+)", text, re.IGNORECASE)
    if not m:
        # Handles "Financial Health: 70.8%" and "**Financial Health**: 70.8%"
        # [^0-9\n]+ stops at the first digit so the number isn't partially consumed
        m = re.search(r"Financial Health[^0-9\n]+([\d.]+)%", text, re.IGNORECASE)
    if m:
        try:
            result["health_adj"] = float(m.group(1))
        except ValueError:
            pass

    # ── Growth score ──────────────────────────────────────────────────────────
    m = re.search(r"\bGROWTH_ADJ[:\s]+([0-9.]+)", text, re.IGNORECASE)
    if not m:
        m = re.search(r"Growth Transition[^0-9\n]+([\d.]+)%", text, re.IGNORECASE)
    if m:
        try:
            result["growth_adj"] = float(m.group(1))
        except ValueError:
            pass

    # ── Verdict ───────────────────────────────────────────────────────────────
    # Capture 1-to-4 uppercase/underscore tokens separated by spaces or tabs
    # (NOT newlines) so that "DO NOT INITIATE" is captured whole rather than
    # truncated to "DO".  After capture, spaces are normalised to underscores.
    _VERDICT_TOKEN = r"[A-Z_]+(?:[ \t][A-Z_]+)*"
    for pat in (
        rf"\bVERDICT[:\s]+({_VERDICT_TOKEN})",
        rf"PORTFOLIO MANAGER VERDICT:\s*({_VERDICT_TOKEN})",
        r"\*\*Action\*\*:\s*\*\*(\w[\w_ ]*)\*\*",
    ):
        m = re.search(pat, text)
        if m:
            result["verdict"] = m.group(1).strip().replace(" ", "_").upper()
            break

    # ── Risk zone ─────────────────────────────────────────────────────────────
    m = re.search(r"\bZONE[:\s]+(HIGH|MODERATE|LOW)\b", text, re.IGNORECASE)
    if m:
        result["zone"] = m.group(1).upper()

    return result


def load_latest_analyses(results_dir: Path) -> dict[str, AnalysisRecord]:
    """
    Load the most recent analysis JSON for each ticker from results_dir.

    Returns dict of yf_ticker -> AnalysisRecord (most recent only).
    """
    if not results_dir.exists():
        logger.warning("results_dir_not_found", path=str(results_dir))
        return {}

    analyses: dict[str, AnalysisRecord] = {}

    for filepath in sorted(results_dir.glob("*_analysis.json"), reverse=True):
        try:
            with open(filepath) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.debug("analysis_load_failed", file=filepath.name, error=str(e))
            continue

        snapshot = data.get("prediction_snapshot", {})

        # For analyses that predate prediction_snapshot, extract scores from the
        # PM final_decision text so that health/growth appear in the SELL output.
        if snapshot.get("health_adj") is None or not snapshot.get("verdict"):
            fd_text = (data.get("final_decision") or {}).get("decision", "") or ""
            if fd_text:
                fb = _parse_scores_from_final_decision(fd_text)
                if snapshot.get("health_adj") is None:
                    snapshot = {**snapshot, "health_adj": fb.get("health_adj")}
                if snapshot.get("growth_adj") is None:
                    snapshot = {**snapshot, "growth_adj": fb.get("growth_adj")}
                if not snapshot.get("verdict"):
                    snapshot = {
                        **snapshot,
                        "verdict": _normalize_verdict(fb.get("verdict") or ""),
                    }
                if not snapshot.get("zone"):
                    snapshot = {**snapshot, "zone": fb.get("zone") or ""}

        ticker = snapshot.get("ticker") or data.get("ticker", "")
        if not ticker:
            # Try to extract from filename: TICKER_YYYY-MM-DD_analysis.json
            parts = filepath.stem.split("_")
            if len(parts) >= 3:
                ticker = parts[0].replace("_", ".")
            if not ticker:
                continue

        # Skip if we already have a more recent analysis for this ticker
        if ticker in analyses:
            continue

        # Parse TRADE_BLOCK from trader output
        trader_plan = data.get("investment_analysis", {}).get("trader_plan", "") or ""
        trade_block = parse_trade_block(trader_plan) or TradeBlockData()

        # Build the analysis record
        record = AnalysisRecord(
            ticker=ticker,
            analysis_date=snapshot.get("analysis_date", "")
            or _extract_date_from_filename(filepath.name),
            file_path=str(filepath),
            verdict=_normalize_verdict(snapshot.get("verdict", "") or ""),
            health_adj=snapshot.get("health_adj"),
            growth_adj=snapshot.get("growth_adj"),
            zone=snapshot.get("zone") or "",
            position_size=snapshot.get("position_size"),
            current_price=snapshot.get("current_price"),
            currency=snapshot.get("currency") or "USD",
            fx_rate_to_usd=snapshot.get("fx_rate_to_usd"),
            trade_block=trade_block,
            # Structured TRADE_BLOCK fields from snapshot (if present)
            entry_price=snapshot.get("entry_price") or trade_block.entry_price,
            stop_price=snapshot.get("stop_price") or trade_block.stop_price,
            target_1_price=snapshot.get("target_1_price") or trade_block.target_1_price,
            target_2_price=snapshot.get("target_2_price") or trade_block.target_2_price,
            conviction=snapshot.get("conviction") or trade_block.conviction,
            # Concentration metadata (sector may be absent in older snapshots)
            sector=snapshot.get("sector") or "",
            exchange=snapshot.get("exchange") or _exchange_from_ticker(ticker),
            is_quick_mode=bool(snapshot.get("is_quick_mode", False)),
        )

        analyses[ticker] = record
        logger.debug(
            "analysis_loaded",
            ticker=ticker,
            verdict=record.verdict,
            date=record.analysis_date,
            age_days=record.age_days,
        )

    logger.info("analyses_loaded", count=len(analyses))
    return analyses


def _extract_date_from_filename(filename: str) -> str:
    """Extract YYYY-MM-DD from filename like '7203_T_2026-02-15_analysis.json'."""
    match = re.search(r"(\d{4}-\d{2}-\d{2})", filename)
    return match.group(1) if match else ""


# ══════════════════════════════════════════════════════════════════════════════
# Staleness Detection
# ══════════════════════════════════════════════════════════════════════════════


def check_staleness(
    analysis: AnalysisRecord,
    current_price_local: float | None = None,
    max_age_days: int = 14,
    drift_threshold_pct: float = 15.0,
    structural_macro_events: list | None = None,
) -> tuple[bool, str]:
    """
    Check if an analysis is stale and should be reviewed.

    Args:
        analysis: The analysis record to check
        current_price_local: Live price from IBKR (in local currency)
        max_age_days: Maximum age before considered stale
        drift_threshold_pct: Price movement threshold for staleness
        structural_macro_events: Optional list of MacroEvent (STRUCTURAL only).
            If an event occurred AFTER this analysis was written, the analysis
            is considered stale (thesis may no longer be valid).

    Returns:
        Tuple of (is_stale, reason)
    """
    reasons = []

    # Age check
    if analysis.age_days > max_age_days:
        age_str = "no date" if analysis.age_days >= 9999 else f"{analysis.age_days}d"
        reasons.append(f"age {age_str} > {max_age_days}d limit")

    # Price drift check (requires both analysis price and current price)
    entry_price = analysis.entry_price or analysis.current_price
    if entry_price and current_price_local and entry_price > 0:
        drift_pct = abs((current_price_local - entry_price) / entry_price) * 100
        if drift_pct > drift_threshold_pct:
            direction = "up" if current_price_local > entry_price else "down"
            reasons.append(f"price drift {drift_pct:.1f}% {direction}")

    # Structural macro event check: if a STRUCTURAL event occurred AFTER this
    # analysis was written, the thesis may no longer be valid — force re-analysis.
    if structural_macro_events and analysis.analysis_date:
        for event in structural_macro_events:
            if event.event_date > analysis.analysis_date:
                if event.scope == "GLOBAL":
                    reasons.append(
                        f"STRUCTURAL macro event ({event.news_headline[:40]!r}) "
                        f"detected after analysis"
                    )
                    break
                # For REGIONAL: check if this analysis ticker matches the event region
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
            return False  # suppress — don't trigger a stop on a likely data error
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


# ══════════════════════════════════════════════════════════════════════════════
# Settlement Helpers
# ══════════════════════════════════════════════════════════════════════════════


def _settlement_date(business_days: int) -> str:
    """Return settlement date as YYYY-MM-DD, skipping weekends."""
    from datetime import date, timedelta

    d = date.today()
    added = 0
    while added < business_days:
        d += timedelta(days=1)
        if d.weekday() < 5:  # Mon–Fri
            added += 1
    return d.isoformat()


# ══════════════════════════════════════════════════════════════════════════════
# Sell Classification
# ══════════════════════════════════════════════════════════════════════════════


def _classify_sell_type(analysis: AnalysisRecord | None, stop_breached: bool) -> str:
    """
    Classify why a position is being sold.

    Returns:
        "STOP_BREACH"  — mechanical stop-loss trigger
        "HARD_REJECT"  — fundamental failure (health_adj < 50 OR growth_adj < 50)
        "SOFT_REJECT"  — passed hard checks, rejected on soft tally / macro fear
    """
    if stop_breached:
        return "STOP_BREACH"
    if analysis is None:
        return "HARD_REJECT"  # no data → treat conservatively
    health_ok = (analysis.health_adj or 0.0) >= 50.0
    growth_ok = (analysis.growth_adj or 0.0) >= 50.0
    return "SOFT_REJECT" if (health_ok and growth_ok) else "HARD_REJECT"


# ══════════════════════════════════════════════════════════════════════════════
# Position-Aware Reconciliation
# ══════════════════════════════════════════════════════════════════════════════


def reconcile(
    positions: list[NormalizedPosition],
    analyses: dict[str, AnalysisRecord],
    portfolio: PortfolioSummary,
    max_age_days: int = 14,
    drift_threshold_pct: float = 15.0,
    overweight_threshold_pct: float = 20.0,
    underweight_threshold_pct: float = 20.0,
    sector_limit_pct: float = 30.0,
    exchange_limit_pct: float = 40.0,
    watchlist_tickers: set[str] | None = None,
) -> list[ReconciliationItem]:
    """
    Compare IBKR positions against evaluator recommendations.

    This is the core function that translates one-off evaluator verdicts
    into position-aware actions. The evaluator doesn't know about existing
    positions — this function does.

    Args:
        positions: Live IBKR positions (normalized). Each NormalizedPosition has
            a yf_ticker (yfinance format, e.g. "7203.T") and a symbol (IBKR raw, e.g. "7203").
        analyses: dict keyed by **yfinance ticker** (e.g. "7203.T", "0005.HK") →
            AnalysisRecord. All keys must be in yf format — this is what
            load_latest_analyses() returns and what watchlist_tickers contains.
        portfolio: Portfolio summary (cash, value); sector_weights and
            exchange_weights fields are populated as a side-effect.
        max_age_days: Max analysis age for staleness
        drift_threshold_pct: Price drift threshold
        overweight_threshold_pct: % overweight before suggesting TRIM
        underweight_threshold_pct: % shortfall vs target before suggesting ADD
        sector_limit_pct: Warn when an ADD/BUY would push any sector above this %
        exchange_limit_pct: Warn when an ADD/BUY would push any exchange above this %
        watchlist_tickers: Optional set of **yfinance tickers** from the IBKR watchlist.
            Conversion from IBKR watchlist format to yf format is the caller's
            responsibility (see ibkr_symbol_to_yf in ticker_mapper.py).
            Tickers not currently held are evaluated in Phase 1.5 and surfaced as
            BUY/HOLD/REMOVE/REVIEW based on their analysis verdict.

    Returns:
        List of ReconciliationItems with position-aware actions
    """
    items: list[ReconciliationItem] = []
    held_tickers: set[str] = set()
    # Track remaining settled cash across Phase 1 ADDs and Phase 2 BUYs
    remaining_cash = portfolio.available_cash_usd

    # ── Secondary lookup: base symbol → AnalysisRecord (alphabetic tickers only) ──
    # Enables cross-format matching when a position's yf_ticker doesn't find an
    # exact key in analyses, in either direction:
    #   "MEGP" pos   → analysis stored as "MEGP.L"   (ibkr_symbol_to_yf failed)
    #   "MEGP.L" pos → analysis stored as "MEGP"     (was re-run without suffix)
    # Purely numeric bases (e.g. "7203", "0005") are excluded because the same
    # number exists on Tokyo, HK, and TW — we cannot safely cross-match them.
    #
    # Suffixed analyses always win over bare ones in this lookup, so even when
    # both "CEK" and "CEK.DE" exist in analyses (different run dates), the lookup
    # always returns the canonical suffixed form "CEK.DE".
    _alpha_base_lookup: dict[str, AnalysisRecord] = {}
    _alpha_base_to_key: dict[str, str] = {}  # base → the analyses key that was chosen
    for _yf_t, _rec in analyses.items():
        _base = (_yf_t.rsplit(".", 1)[0] if "." in _yf_t else _yf_t).upper()
        if re.match(
            r"^[A-Z][A-Z0-9]*$", _base
        ):  # starts with a letter → not purely numeric
            if "." in _yf_t:
                # Suffixed entry is more specific — always overwrite any bare entry
                _alpha_base_lookup[_base] = _rec
                _alpha_base_to_key[_base] = _yf_t
            else:
                # Bare (no suffix) — only use if no suffixed entry exists yet
                _alpha_base_lookup.setdefault(_base, _rec)
                _alpha_base_to_key.setdefault(_base, _yf_t)

    # Pre-fetch structural macro events for staleness invalidation (fail-safe).
    _structural_events: list = []
    try:
        from datetime import datetime as _dt
        from datetime import timedelta

        from src.memory import create_macro_events_store

        _mstore = create_macro_events_store()
        if _mstore.available:
            _structural_events = _mstore.get_structural_events_since(
                (_dt.now() - timedelta(days=180)).strftime("%Y-%m-%d")
            )
    except Exception:
        pass

    # ── Pre-compute concentration weights from currently held positions ──
    # These are stored on the portfolio object for use by format_report().
    # Use sum of position values as denominator so weights always sum to 100%,
    # even when IBKR ledger total and position market values use different FX rates.
    # Use pos.ticker.yf (with alpha-base fallback) so bare-ticker positions (e.g. "CEK")
    # find the right analysis (e.g. "CEK.DE") for sector classification.
    _sector_weights: dict[str, float] = {}
    _exchange_weights: dict[str, float] = {}
    _total_pos_value = sum(p.market_value_usd for p in positions)
    if _total_pos_value > 0:
        for pos in positions:
            _cticker = pos.ticker.yf
            _analysis = analyses.get(_cticker)
            # Alpha-base fallback: bare ticker position (e.g. "CEK") → suffixed analysis
            if (
                _analysis is None
                and not pos.ticker.has_suffix
                and not pos.ticker.ibkr.isdigit()
            ):
                _best = _alpha_base_lookup.get(pos.ticker.ibkr.upper())
                if _best and "." in _best.ticker:
                    _cticker = _best.ticker
                    _analysis = _best
            _sector = (_analysis.sector if _analysis else "") or "Unknown"
            # Use IBKR-authoritative exchange inference: yf_ticker suffix → IBKR
            # listingExchange → currency heuristic.  Avoids misclassifying stocks
            # that were analysed without an exchange suffix as "US".
            _exchange = _exchange_from_position(pos)
            _w = pos.market_value_usd / _total_pos_value * 100
            _sector_weights[_sector] = _sector_weights.get(_sector, 0.0) + _w
            _exchange_weights[_exchange] = _exchange_weights.get(_exchange, 0.0) + _w
    portfolio.sector_weights = _sector_weights
    portfolio.exchange_weights = _exchange_weights

    # ── Phase 1: Evaluate existing positions ──
    for pos in positions:
        # IBKR occasionally returns positions with quantity=0 for a short period
        # after a fill clears (the position is in the process of being removed).
        # Skip these — they are not held; treating them as held would trigger
        # spurious stop-breach SELLs or verdict-conflict REVIEWs with no shares.
        if pos.quantity <= 0:
            continue

        # Resolve canonical yfinance ticker BEFORE the analyses.get() call.
        # For bare tickers (no exchange suffix), consult _alpha_base_lookup first:
        # it always returns the suffixed analysis when one exists, so a bare "CEK"
        # position finds "CEK.DE" analysis rather than the bare "CEK" one.
        # Numeric symbols excluded — same number appears on multiple exchanges.
        yf_key = pos.ticker.yf
        analysis: AnalysisRecord | None = None

        if (
            not pos.ticker.has_suffix
            and pos.ticker.ibkr
            and not pos.ticker.ibkr.isdigit()
        ):
            # Bare ticker: alpha-base lookup wins over exact-match to prefer suffixed form
            _best = _alpha_base_lookup.get(pos.ticker.ibkr.upper())
            if _best:
                yf_key = (
                    _best.ticker
                )  # e.g. "CEK.DE" or bare "MEGP" if no suffixed version
                analysis = _best
                logger.debug(
                    "analysis_found_via_alpha_base",
                    pos_yf=pos.ticker.yf,
                    ibkr_symbol=pos.ticker.ibkr,
                    found_as=_best.ticker,
                )

        # Direct lookup: suffixed tickers, or bare tickers with no alpha-base match
        if analysis is None:
            analysis = analyses.get(yf_key)

        # Fallback B: suffixed pos but analysis stored under bare ticker
        # (e.g. pos.ticker.yf = "MEGP.L" but analysis stored as "MEGP").
        if analysis is None and pos.ticker.ibkr and not pos.ticker.ibkr.isdigit():
            analysis = _alpha_base_lookup.get(pos.ticker.ibkr.upper())
            if analysis:
                logger.debug(
                    "analysis_found_via_base_symbol",
                    yf_ticker=pos.ticker.yf,
                    ibkr_symbol=pos.ticker.ibkr,
                    found_as=analysis.ticker,
                )

        # Canonical Ticker for the item (upgraded if fallback fired):
        item_ticker = Ticker.from_yf(yf_key) if yf_key != pos.ticker.yf else pos.ticker
        ticker = yf_key
        held_tickers.add(ticker)

        # For bare tickers (no exchange suffix), also block every suffixed
        # analysis key with the same base symbol so Phase 2 doesn't surface
        # them as untracked BUY candidates.  The alpha_base_lookup above handles
        # alphabetic tickers when an analysis exists, but numeric tickers are
        # explicitly excluded from that lookup — this covers the gap.
        # Example: holding bare "5434" while the analysis is "5434.TW" would
        # otherwise produce a false WATCHLIST CANDIDATE for the held position.
        if "." not in ticker:
            _held_base = ticker.upper()
            for _ak in analyses:
                if "." in _ak and _ak.split(".")[0].upper() == _held_base:
                    held_tickers.add(_ak)

        if analysis is None:
            # Position held but NO analysis exists → needs review
            items.append(
                ReconciliationItem(
                    ticker=item_ticker,
                    action="REVIEW",
                    reason="Position held but no evaluator analysis found",
                    urgency="MEDIUM",
                    ibkr_position=pos,
                )
            )
            continue

        current_price = pos.current_price_local

        # Check stop breach (URGENT)
        if check_stop_breach(analysis, current_price):
            items.append(
                ReconciliationItem(
                    ticker=item_ticker,
                    action="SELL",
                    reason=f"Stop breached: price {current_price:.2f} < stop {analysis.stop_price:.2f}",
                    urgency="HIGH",
                    ibkr_position=pos,
                    analysis=analysis,
                    suggested_quantity=abs(int(pos.quantity)),
                    suggested_price=current_price,
                    suggested_order_type="LMT",
                    cash_impact_usd=pos.market_value_usd,
                    settlement_date=_settlement_date(2),
                    sell_type="STOP_BREACH",
                )
            )
            continue

        # Check verdict conflict: we hold but evaluator says don't
        verdict_upper = _normalize_verdict(analysis.verdict or "")
        if verdict_upper in _REJECT_VERDICTS:
            items.append(
                ReconciliationItem(
                    ticker=item_ticker,
                    action="SELL",
                    reason=f"Verdict → {analysis.verdict}  ({analysis.analysis_date})",
                    urgency="HIGH",
                    ibkr_position=pos,
                    analysis=analysis,
                    suggested_quantity=abs(int(pos.quantity)),
                    suggested_order_type="LMT",
                    suggested_price=current_price,
                    cash_impact_usd=pos.market_value_usd,
                    settlement_date=_settlement_date(2),
                    sell_type=_classify_sell_type(analysis, stop_breached=False),
                )
            )
            continue

        # Check target hit (profit-taking review — more specific than staleness)
        if check_target_hit(analysis, current_price):
            items.append(
                ReconciliationItem(
                    ticker=item_ticker,
                    action="REVIEW",
                    reason=f"Target hit: price {current_price:.2f} >= target {analysis.target_1_price:.2f}",
                    urgency="LOW",
                    ibkr_position=pos,
                    analysis=analysis,
                )
            )
            continue

        # Check staleness (after target hit, which is more specific)
        is_stale, stale_reason = check_staleness(
            analysis,
            current_price,
            max_age_days,
            drift_threshold_pct,
            structural_macro_events=_structural_events,
        )
        if is_stale:
            items.append(
                ReconciliationItem(
                    ticker=item_ticker,
                    action="REVIEW",
                    reason=f"Stale analysis: {stale_reason}",
                    urgency="MEDIUM",
                    ibkr_position=pos,
                    analysis=analysis,
                )
            )
            continue

        # Check overweight / underweight
        target_size_pct = analysis.trade_block.size_pct or (analysis.position_size or 0)
        if target_size_pct > 0 and portfolio.portfolio_value_usd > 0:
            actual_pct = (pos.market_value_usd / portfolio.portfolio_value_usd) * 100
            excess_pct = actual_pct - target_size_pct
            if excess_pct > overweight_threshold_pct:
                target_value_usd = portfolio.portfolio_value_usd * (
                    target_size_pct / 100
                )
                trim_value_usd = pos.market_value_usd - target_value_usd
                # Use implied USD price per share for cross-currency accuracy
                price_usd_per_share = (
                    pos.market_value_usd / abs(pos.quantity)
                    if pos.quantity != 0
                    else 1.0
                )
                trim_qty = round_to_lot_size(
                    int(trim_value_usd / (price_usd_per_share or 1.0)), ticker
                )
                items.append(
                    ReconciliationItem(
                        ticker=item_ticker,
                        action="TRIM",
                        reason=f"Overweight: {actual_pct:.1f}% vs target {target_size_pct:.1f}% (+{excess_pct:.1f}%)",
                        urgency="MEDIUM",
                        ibkr_position=pos,
                        analysis=analysis,
                        suggested_quantity=trim_qty,
                        suggested_price=pos.current_price_local,
                        suggested_order_type="LMT",
                        cash_impact_usd=trim_value_usd,
                        settlement_date=_settlement_date(2),
                    )
                )
                continue

            # Check underweight — only for BUY-verdict positions with sufficient shortfall
            shortfall_pct = target_size_pct - actual_pct
            verdict_upper = _normalize_verdict(analysis.verdict or "")
            if (
                shortfall_pct > underweight_threshold_pct
                and verdict_upper == "BUY"
                and remaining_cash > 0
            ):
                target_value_usd = portfolio.portfolio_value_usd * (
                    target_size_pct / 100
                )
                add_value_usd = min(
                    target_value_usd - pos.market_value_usd, remaining_cash
                )
                price_usd_per_share = (
                    pos.market_value_usd / abs(pos.quantity)
                    if pos.quantity != 0
                    else 1.0
                )
                add_qty = round_to_lot_size(
                    int(add_value_usd / (price_usd_per_share or 1.0)), ticker
                )
                # Skip only when we CAN calculate a quantity but the cost is trivially
                # small. If add_qty=0 due to lot-size rounding, fall through to HOLD.
                actual_add_cost = add_qty * price_usd_per_share
                if add_qty == 0 or (add_qty > 0 and actual_add_cost < _MIN_ORDER_USD):
                    # Fall through to HOLD — not worth placing
                    pass
                else:
                    remaining_cash -= add_value_usd
                    # Check concentration limits
                    add_reason = f"Underweight: {actual_pct:.1f}% vs target {target_size_pct:.1f}% (-{shortfall_pct:.1f}%)"
                    if portfolio.portfolio_value_usd > 0:
                        exch = analysis.exchange or _exchange_from_ticker(ticker)
                        sect = analysis.sector or "Unknown"
                        proj_exch = (
                            _exchange_weights.get(exch, 0.0)
                            + add_value_usd / portfolio.portfolio_value_usd * 100
                        )
                        proj_sect = (
                            _sector_weights.get(sect, 0.0)
                            + add_value_usd / portfolio.portfolio_value_usd * 100
                        )
                        conc_warns = []
                        if proj_exch > exchange_limit_pct:
                            conc_warns.append(
                                f"⚠ {exch} → {proj_exch:.0f}% (limit {exchange_limit_pct:.0f}%)"
                            )
                        if proj_sect > sector_limit_pct:
                            conc_warns.append(
                                f"⚠ {sect} sector → {proj_sect:.0f}% (limit {sector_limit_pct:.0f}%)"
                            )
                        if conc_warns:
                            add_reason += "  " + "; ".join(conc_warns)
                    items.append(
                        ReconciliationItem(
                            ticker=item_ticker,
                            action="ADD",
                            reason=add_reason,
                            urgency="LOW",
                            ibkr_position=pos,
                            analysis=analysis,
                            suggested_quantity=add_qty,
                            # Prefer live IBKR price for ADD (we already own it)
                            suggested_price=pos.current_price_local
                            or analysis.entry_price,
                            suggested_order_type="LMT",
                            cash_impact_usd=-add_value_usd,
                        )
                    )
                    continue

        # All clear — HOLD
        status_parts = []
        if analysis.entry_price and current_price:
            gain_pct = (
                (current_price - analysis.entry_price) / analysis.entry_price
            ) * 100
            status_parts.append(
                f"entry {analysis.entry_price:.2f} → {current_price:.2f} ({gain_pct:+.1f}%)"
            )
        if analysis.stop_price:
            status_parts.append(f"stop {analysis.stop_price:.2f}")
        if analysis.target_1_price:
            status_parts.append(f"target {analysis.target_1_price:.2f}")

        items.append(
            ReconciliationItem(
                ticker=item_ticker,
                action="HOLD",
                reason=f"Within targets — {'; '.join(status_parts)}"
                if status_parts
                else "Position OK",
                urgency="LOW",
                ibkr_position=pos,
                analysis=analysis,
            )
        )

    # ── Phase 1.5: Watchlist tickers not currently held ──
    # Evaluates each watchlist ticker that isn't an open position.
    # After processing, all watchlist tickers are added to held_tickers so that
    # Phase 2 doesn't surface them a second time as plain BUY recommendations.
    watchlist_set = (watchlist_tickers or set()) - held_tickers
    # Collects the analyses keys (e.g. "WDO.TO") resolved via base-symbol lookup
    # for watchlist entries stored without a suffix (e.g. "WDO").  These must also
    # be added to held_tickers after Phase 1.5 so Phase 2 doesn't re-surface them
    # as untracked BUY candidates.
    _watchlist_resolved_keys: set[str] = set()

    for ticker in sorted(watchlist_set):
        analysis = analyses.get(ticker)

        # ALWAYS check _alpha_base_to_key for the base symbol — even when the bare
        # analysis was found directly (e.g. analyses holds both "WDO" DNI and
        # "WDO.TO" BUY).  Two goals:
        # (a) Add the suffixed key to _watchlist_resolved_keys so Phase 2 doesn't
        #     re-surface it as an untracked BUY candidate.
        # (b) Prefer the more-specific suffixed analysis and ticker form so that
        #     run commands (`--ticker WDO.TO`) are generated correctly.
        _wl_base = (ticker.rsplit(".", 1)[0] if "." in ticker else ticker).upper()
        if re.match(r"^[A-Z][A-Z0-9]*$", _wl_base):
            _resolved_key = _alpha_base_to_key.get(_wl_base)
            if _resolved_key and _resolved_key != ticker:
                _watchlist_resolved_keys.add(_resolved_key)
                _resolved_analysis = analyses.get(_resolved_key)
                if _resolved_analysis:
                    # Prefer the more-specific suffixed form
                    analysis = _resolved_analysis
                    ticker = _resolved_key  # e.g. upgrade "WDO" → "WDO.TO"
        else:
            # Numeric base (e.g. "5434"): _alpha_base_to_key skips numeric symbols.
            # Manually cross-reference all suffixed analysis keys so Phase 2 doesn't
            # re-surface them as untracked BUY candidates.
            for _ak in analyses:
                if "." in _ak and _ak.split(".")[0].upper() == _wl_base:
                    _watchlist_resolved_keys.add(_ak)
                    if analysis is None:
                        _resolved_analysis = analyses.get(_ak)
                        if _resolved_analysis:
                            analysis = _resolved_analysis
                            ticker = _ak

        if analysis is None:
            items.append(
                ReconciliationItem(
                    ticker=ticker,
                    action="REVIEW",
                    reason="Watchlist: no analysis found",
                    urgency="MEDIUM",
                    is_watchlist=True,
                )
            )
            continue

        is_stale, stale_reason = check_staleness(
            analysis,
            None,
            max_age_days,
            drift_threshold_pct,
            structural_macro_events=_structural_events,
        )
        if is_stale:
            items.append(
                ReconciliationItem(
                    ticker=ticker,
                    action="REVIEW",
                    reason=f"Watchlist: stale analysis ({stale_reason})",
                    urgency="MEDIUM",
                    analysis=analysis,
                    is_watchlist=True,
                )
            )
            continue

        verdict_upper = _normalize_verdict(analysis.verdict or "")

        if verdict_upper in _REJECT_VERDICTS:
            items.append(
                ReconciliationItem(
                    ticker=ticker,
                    action="REMOVE",
                    reason=f"Remove from watchlist — verdict → {analysis.verdict}  ({analysis.analysis_date})",
                    urgency="MEDIUM",
                    analysis=analysis,
                    is_watchlist=True,
                )
            )

        elif verdict_upper == "BUY":
            has_portfolio = portfolio.portfolio_value_usd > 0
            if has_portfolio and remaining_cash <= 0:
                # No cash — still surface the BUY as informational
                items.append(
                    ReconciliationItem(
                        ticker=ticker,
                        action="BUY",
                        reason=f"Watchlist BUY ({analysis.analysis_date}) — no cash available",
                        urgency="MEDIUM",
                        analysis=analysis,
                        is_watchlist=True,
                    )
                )
            else:
                entry_price = analysis.entry_price or analysis.current_price
                conviction = (
                    analysis.conviction or analysis.trade_block.conviction or ""
                )
                size_pct = analysis.trade_block.size_pct or (
                    analysis.position_size or 0
                )
                _fx = _resolve_fx(analysis)
                buy_qty = calculate_quantity(
                    available_cash_usd=remaining_cash,
                    entry_price_local=entry_price or 0.0,
                    fx_rate_to_usd=_fx,
                    size_pct=size_pct,
                    portfolio_value_usd=portfolio.portfolio_value_usd,
                    yf_ticker=ticker,
                )
                buy_cost_usd = buy_qty * (entry_price or 0.0) * _fx
                if buy_qty > 0 and buy_cost_usd < _MIN_ORDER_USD:
                    # Too small to place; skip (same rule as Phase 2)
                    pass
                else:
                    remaining_cash -= buy_cost_usd
                    buy_reason = f"Watchlist BUY ({analysis.analysis_date}) — {conviction} conviction, target {size_pct:.1f}%"
                    if portfolio.portfolio_value_usd > 0:
                        exch = analysis.exchange or _exchange_from_ticker(ticker)
                        sect = analysis.sector or "Unknown"
                        proj_exch = (
                            _exchange_weights.get(exch, 0.0)
                            + buy_cost_usd / portfolio.portfolio_value_usd * 100
                        )
                        proj_sect = (
                            _sector_weights.get(sect, 0.0)
                            + buy_cost_usd / portfolio.portfolio_value_usd * 100
                        )
                        conc_warns = []
                        if proj_exch > exchange_limit_pct:
                            conc_warns.append(
                                f"⚠ {exch} → {proj_exch:.0f}% (limit {exchange_limit_pct:.0f}%)"
                            )
                        if proj_sect > sector_limit_pct:
                            conc_warns.append(
                                f"⚠ {sect} sector → {proj_sect:.0f}% (limit {sector_limit_pct:.0f}%)"
                            )
                        if conc_warns:
                            buy_reason += "  " + "; ".join(conc_warns)
                    items.append(
                        ReconciliationItem(
                            ticker=ticker,
                            action="BUY",
                            reason=buy_reason,
                            urgency="MEDIUM",
                            analysis=analysis,
                            suggested_quantity=buy_qty if buy_qty > 0 else None,
                            suggested_price=entry_price,
                            suggested_order_type="LMT",
                            cash_impact_usd=-buy_cost_usd if buy_cost_usd > 0 else 0.0,
                            is_watchlist=True,
                        )
                    )

        else:
            # HOLD or any other verdict: keep watching, no position action
            items.append(
                ReconciliationItem(
                    ticker=ticker,
                    action="HOLD",
                    reason=f"Watchlist: monitoring — verdict {analysis.verdict}  ({analysis.analysis_date})",
                    urgency="LOW",
                    analysis=analysis,
                    is_watchlist=True,
                )
            )

    # Prevent Phase 2 from re-processing watchlist tickers.
    # Also block any suffixed keys resolved via base-symbol lookup (e.g. "WDO.TO"
    # when the watchlist entry was "WDO") so Phase 2 doesn't surface them again.
    held_tickers.update(watchlist_set)
    held_tickers.update(_watchlist_resolved_keys)

    # ── Phase 2: Find BUY recommendations not yet held ──
    for ticker, analysis in analyses.items():
        if ticker in held_tickers:
            continue  # Already handled above

        verdict_upper = _normalize_verdict(analysis.verdict or "")
        if verdict_upper not in ("BUY",):
            continue  # Only surface new BUY recommendations

        # Skip stale analyses for new buys
        is_stale, stale_reason = check_staleness(
            analysis,
            None,
            max_age_days,
            drift_threshold_pct,
            structural_macro_events=_structural_events,
        )
        if is_stale:
            continue

        # Only skip new buys if we have a live portfolio but no cash
        # (in --read-only mode portfolio_value_usd=0, so buys still surface)
        has_portfolio = portfolio.portfolio_value_usd > 0
        if has_portfolio and remaining_cash <= 0:
            continue

        entry_price = analysis.entry_price or analysis.current_price
        conviction = analysis.conviction or analysis.trade_block.conviction or ""
        size_pct = analysis.trade_block.size_pct or (analysis.position_size or 0)

        _fx = _resolve_fx(analysis)
        buy_qty = calculate_quantity(
            available_cash_usd=remaining_cash,
            entry_price_local=entry_price or 0.0,
            fx_rate_to_usd=_fx,
            size_pct=size_pct,
            portfolio_value_usd=portfolio.portfolio_value_usd,
            yf_ticker=ticker,
        )
        buy_cost_usd = buy_qty * (entry_price or 0.0) * _fx

        # Skip only when we CAN calculate a quantity but the cost is trivially small.
        # If buy_qty=0 due to lot-size rounding (e.g. can't afford a full lot yet),
        # still surface the BUY as an informational recommendation (quantity=None).
        if buy_qty > 0 and buy_cost_usd < _MIN_ORDER_USD:
            continue

        remaining_cash -= buy_cost_usd

        # Check concentration limits
        buy_reason = f"New BUY ({analysis.analysis_date}) — {conviction} conviction, target {size_pct:.1f}%"
        if portfolio.portfolio_value_usd > 0:
            exch = analysis.exchange or _exchange_from_ticker(ticker)
            sect = analysis.sector or "Unknown"
            proj_exch = (
                _exchange_weights.get(exch, 0.0)
                + buy_cost_usd / portfolio.portfolio_value_usd * 100
            )
            proj_sect = (
                _sector_weights.get(sect, 0.0)
                + buy_cost_usd / portfolio.portfolio_value_usd * 100
            )
            conc_warns = []
            if proj_exch > exchange_limit_pct:
                conc_warns.append(
                    f"⚠ {exch} → {proj_exch:.0f}% (limit {exchange_limit_pct:.0f}%)"
                )
            if proj_sect > sector_limit_pct:
                conc_warns.append(
                    f"⚠ {sect} sector → {proj_sect:.0f}% (limit {sector_limit_pct:.0f}%)"
                )
            if conc_warns:
                buy_reason += "  " + "; ".join(conc_warns)

        items.append(
            ReconciliationItem(
                ticker=ticker,
                action="BUY",
                reason=buy_reason,
                urgency="MEDIUM",
                analysis=analysis,
                suggested_quantity=buy_qty if buy_qty > 0 else None,
                suggested_price=entry_price,
                suggested_order_type="LMT",
                cash_impact_usd=-buy_cost_usd if buy_cost_usd > 0 else 0.0,
            )
        )

    # Sort: HIGH urgency first, then MEDIUM, then LOW
    urgency_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    items.sort(key=lambda x: urgency_order.get(x.urgency, 9))

    logger.info(
        "reconciliation_complete",
        total_items=len(items),
        sells=sum(1 for i in items if i.action == "SELL"),
        trims=sum(1 for i in items if i.action == "TRIM"),
        adds=sum(1 for i in items if i.action == "ADD"),
        buys=sum(1 for i in items if i.action == "BUY"),
        holds=sum(1 for i in items if i.action == "HOLD"),
        reviews=sum(1 for i in items if i.action == "REVIEW"),
        removes=sum(1 for i in items if i.action == "REMOVE"),
    )

    return items


# ══════════════════════════════════════════════════════════════════════════════
# Portfolio-Level Health Check
# ══════════════════════════════════════════════════════════════════════════════


def compute_portfolio_health(
    positions: list[NormalizedPosition],
    analyses: dict[str, AnalysisRecord],
    portfolio: PortfolioSummary,
    max_age_days: int = 14,
    reconciliation_items: list | None = None,
    correlated_window_days: int = 7,
) -> list[str]:
    """
    Compute portfolio-level health flags using data already in held analyses.

    Returns a list of human-readable flag strings. Empty list = no flags.
    Call AFTER reconcile() so that portfolio.exchange_weights is already set.

    When reconciliation_items is provided, also detects CORRELATED_SELL_EVENT
    and demotes SOFT_REJECT SELLs to REVIEW in-place on correlated days.

    Args:
        correlated_window_days: Window width for grouping nearby sell dates.
            Sells whose analysis_date falls within this many days of an anchor date
            are counted as a single correlated event.  Default 7 prevents batch
            re-analyses run over consecutive nights from fragmenting the same event.

    Flags:
        LOW_HEALTH_AVERAGE      Weighted avg health_adj < 60 across holdings
        LOW_GROWTH_AVERAGE      Weighted avg growth_adj < 55 across holdings
        CURRENCY_CONCENTRATION  >50% of portfolio in a single currency
        STALE_ANALYSIS_RATIO    >30% of positions have analyses older than max_age_days
        CORRELATED_SELL_EVENT   ≥5 and ≥25% of held positions flipped verdict within
                                correlated_window_days of the same anchor date
    """
    if not positions or portfolio.portfolio_value_usd <= 0:
        return []

    flags: list[str] = []
    total_weight = 0.0
    weighted_health = 0.0
    weighted_growth = 0.0
    health_count = 0
    growth_count = 0
    stale_count = 0
    currency_weights: dict[str, float] = {}
    # Per-position scores for detail lines: (ticker, score, is_stale)
    scored_health: list[tuple[str, float, bool]] = []
    scored_growth: list[tuple[str, float, bool]] = []

    for pos in positions:
        weight = pos.market_value_usd / portfolio.portfolio_value_usd
        total_weight += weight

        analysis = analyses.get(pos.ticker.yf)
        is_stale = analysis is not None and analysis.age_days > max_age_days
        if analysis:
            if analysis.health_adj is not None:
                weighted_health += analysis.health_adj * weight
                health_count += 1
                scored_health.append((pos.ticker.yf, analysis.health_adj, is_stale))
            if analysis.growth_adj is not None:
                weighted_growth += analysis.growth_adj * weight
                growth_count += 1
                scored_growth.append((pos.ticker.yf, analysis.growth_adj, is_stale))
            if is_stale:
                stale_count += 1

        ccy = (pos.currency or "USD").upper()
        currency_weights[ccy] = currency_weights.get(ccy, 0.0) + weight * 100

    def _worst_detail(
        scored: list[tuple[str, float, bool]],
        max_age_days: int,
        n: int = 5,
    ) -> str:
        """Build a detail sub-line showing the N lowest-scoring positions."""
        worst = sorted(scored, key=lambda x: x[1])[:n]
        items_str = "  ".join(f"{t}({s:.0f}{'†' if st else ''})" for t, s, st in worst)
        stale_n = sum(1 for _, _, st in scored if st)
        lines = [f"       Lowest: {items_str}"]
        if stale_n > 0:
            lines.append(
                f"       (†= stale >{max_age_days}d — {stale_n}/{len(scored)} scored"
                f" analyses; scores may not reflect recent conditions)"
            )
        return "\n".join(lines)

    if total_weight > 0:
        if health_count > 0 and (weighted_health / total_weight) < 60:
            detail = _worst_detail(scored_health, max_age_days)
            flags.append(
                f"LOW_HEALTH_AVERAGE: weighted avg health {weighted_health / total_weight:.0f} < 60"
                " — portfolio skewing toward distressed names"
                f"\n{detail}"
            )
        if growth_count > 0 and (weighted_growth / total_weight) < 55:
            detail = _worst_detail(scored_growth, max_age_days)
            flags.append(
                f"LOW_GROWTH_AVERAGE: weighted avg growth {weighted_growth / total_weight:.0f} < 55"
                " — GARP thesis eroding"
                f"\n{detail}"
            )

    for ccy, pct in sorted(currency_weights.items(), key=lambda x: -x[1]):
        if pct > 50:
            flags.append(
                f"CURRENCY_CONCENTRATION: {pct:.1f}% in {ccy}"
                " — FX risk amplification"
            )

    if positions:
        stale_pct = stale_count / len(positions) * 100
        if stale_pct > 30:
            flags.append(
                f"STALE_ANALYSIS_RATIO: {stale_count}/{len(positions)} positions"
                f" ({stale_pct:.0f}%) have analyses older than {max_age_days}d"
                " — flying blind on significant chunk of portfolio"
                " (re-run with --refresh-stale to update)"
            )

    # Geography concentration is already surfaced via portfolio.exchange_weights;
    # flag here only if it exceeds the standard 40% exchange limit.
    for exch, pct in portfolio.exchange_weights.items():
        if pct > 40:
            long_name = _EXCHANGE_LONG_NAMES.get(exch, exch)
            flags.append(
                f"GEOGRAPHY_CONCENTRATION: {pct:.1f}% in {exch} ({long_name})"
                " — single-exchange concentration"
            )

    # ── CORRELATED_SELL_EVENT ─────────────────────────────────────────────────
    # Detect when many held positions change verdict within a short window — a
    # sign of macro noise rather than company-specific thesis failures.  Uses a
    # sliding date window (correlated_window_days, default 7) so that batch
    # re-analyses spread over consecutive nights don't fragment the same macro
    # event across multiple exact dates.  Only verdict-driven SELLs
    # (HARD_REJECT + SOFT_REJECT) are counted; mechanical stop breaches are
    # excluded.
    if reconciliation_items is not None:
        from datetime import date as _date
        from datetime import timedelta as _td

        verdict_sells = [
            item
            for item in reconciliation_items
            if item.action == "SELL"
            and item.sell_type in ("HARD_REJECT", "SOFT_REJECT")
        ]
        total_held = sum(
            1 for item in reconciliation_items if item.ibkr_position is not None
        )

        if verdict_sells:
            # Parse analysis dates; skip sells with unparseable dates.
            dated: list[tuple] = []
            for item in verdict_sells:
                if item.analysis and item.analysis.analysis_date:
                    try:
                        d = _date.fromisoformat(item.analysis.analysis_date)
                        dated.append((item, d))
                    except ValueError:
                        pass

            if dated:
                all_dates = [d for _, d in dated]
                peak_count = 0
                peak_anchor = None

                # Sliding window: for each sell date as anchor, count sells
                # whose date falls within [anchor, anchor + window).
                for anchor in all_dates:
                    window_end = anchor + _td(days=correlated_window_days - 1)
                    count = sum(1 for d in all_dates if anchor <= d <= window_end)
                    if count > peak_count:
                        peak_count = count
                        peak_anchor = anchor

                correlated_event = (
                    peak_count >= 5
                    and total_held > 0
                    and peak_count / total_held >= 0.25
                )
                if correlated_event and peak_anchor is not None:
                    flags.append(
                        f"CORRELATED_SELL_EVENT: {peak_count} positions changed verdict"
                        f" within {correlated_window_days}d of {peak_anchor.isoformat()}"
                        f" ({peak_count / total_held:.0%} of held"
                        f" positions) — probable macro event. Execute stop-breach SELLs"
                        f" only; review verdict-change SELLs before acting."
                    )
                    # Demote SOFT_REJECT from SELL to REVIEW — these passed all hard
                    # fundamental checks; the SELL verdict came from the soft-tally
                    # (geopolitical/macro points), not from a thesis failure.
                    for item in reconciliation_items:
                        if item.action == "SELL" and item.sell_type == "SOFT_REJECT":
                            item.action = "REVIEW"
                            item.urgency = "MEDIUM"
                            item.reason += (
                                "  [MACRO_WATCH: demoted from SELL — correlated"
                                " event detected]"
                            )
                    logger.info(
                        "correlated_sell_event_detected",
                        peak_date=peak_anchor.isoformat(),
                        window_days=correlated_window_days,
                        peak_count=peak_count,
                        total_held=total_held,
                        pct=f"{peak_count / total_held:.0%}",
                    )

    return flags
