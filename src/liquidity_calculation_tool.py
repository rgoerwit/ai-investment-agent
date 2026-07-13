import math
from dataclasses import dataclass
from typing import Annotated, Literal

import structlog
from langchain_core.tools import tool

from src.error_safety import summarize_exception
from src.fx_normalization import (
    get_fx_rate,
    is_near_minor_unit_ratio,
    normalize_minor_unit_currency,
)
from src.runtime_services import get_current_market_data_fetcher
from src.thesis_constants import LIQUIDITY_MIN_USD, LIQUIDITY_PASS_USD
from src.ticker_utils import normalize_ticker

logger = structlog.get_logger(__name__)

from src.currency_resolver import resolve_local_trading_currency
from src.ticker_policy import get_ticker_suffix


def _market_data_fetcher():
    """Resolve the active market-data fetcher at call time."""
    return get_current_market_data_fetcher()


# Provenance of the per-share price used for turnover. Typed so the
# string-membership checks downstream (and the Details: line) can't silently
# drift if a label is renamed.
TurnoverPriceSource = Literal["anchor", "history"]
TurnoverPriceReason = Literal[
    "anchor_already_major",
    "anchor_minor_scaled",
    "anchor_major",
    "lse_history_pence",
    "history_major_assumed",
]


@dataclass(frozen=True)
class TurnoverPrice:
    """Per-share price for turnover, already scaled to the major currency unit."""

    value: float
    currency: str | None
    source: TurnoverPriceSource
    reason: TurnoverPriceReason


def _coerce_positive_float(value: object) -> float | None:
    """Best-effort positive, finite float coercion.

    Returns None for junk, non-positive, or non-finite (``nan``/``inf``) inputs
    so a bad quote can never flow into turnover math as a spurious PASS.
    """
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and parsed > 0 else None


def _resolve_turnover_price(
    anchor_data: dict | None,
    hist_close_mean: float,
    normalized_symbol: str,
) -> TurnoverPrice:
    """Resolve the major-unit per-share price for turnover from data provenance.

    The fetcher converts a triangulated minor-unit quote (e.g. GBp/GBX -> GBP)
    and records it under ``info["_unit_normalization"]`` with
    ``kind == "quote_minor_to_major"`` (set on the *merged* dict in
    ``_normalize_data_integrity``). When that marker is present the anchor price
    is ALREADY in major units and must not be divided again — divining the scale
    from the bare ``.L`` suffix is the GBp->GBP double-normalization bug. A
    still-minor anchor (currency ``GBp``/``GBX``) is scaled via
    ``normalize_minor_unit_currency``. Only the raw-history fallback keeps the
    blunt ``.L`` pence assumption, because LSE history stays in exchange quote
    (minor) units and carries no currency label to key off.
    """
    if isinstance(anchor_data, dict):
        anchor_price = _coerce_positive_float(anchor_data.get("currentPrice"))
        if anchor_price is not None:
            unit_meta = anchor_data.get("_unit_normalization")
            if (
                isinstance(unit_meta, dict)
                and unit_meta.get("kind") == "quote_minor_to_major"
            ):
                currency = unit_meta.get("to_currency") or anchor_data.get("currency")
                return TurnoverPrice(
                    anchor_price,
                    str(currency) if currency else None,
                    "anchor",
                    "anchor_already_major",
                )
            anchor_currency = anchor_data.get("currency")
            major_currency, scale = normalize_minor_unit_currency(
                anchor_currency if isinstance(anchor_currency, str) else None
            )
            reason: TurnoverPriceReason = (
                "anchor_minor_scaled" if scale != 1.0 else "anchor_major"
            )
            return TurnoverPrice(anchor_price * scale, major_currency, "anchor", reason)

    # Historical close fallback is still provider quote-unit data. LSE (.L)
    # history is quoted in pence -> scale to pounds; other venues assumed major.
    if normalized_symbol.endswith(".L"):
        return TurnoverPrice(
            hist_close_mean * 0.01, "GBP", "history", "lse_history_pence"
        )
    return TurnoverPrice(hist_close_mean, None, "history", "history_major_assumed")


def _near_100x_mismatch(left: float | None, right: float | None) -> bool:
    """True when two prices differ by ~100x — a minor/major unit-scale bug."""
    if not left or not right or left <= 0 or right <= 0:
        return False
    ratio = max(left, right) / min(left, right)
    return is_near_minor_unit_ratio(ratio)


@tool
async def calculate_liquidity_metrics(
    ticker: Annotated[str | None, "Stock ticker symbol"] = None,
) -> str:
    """
    Calculate liquidity metrics using the robust MarketDataFetcher.
    Checks 3-month average volume and turnover.
    Handles global currency conversion automatically.
    """
    if not ticker:
        return "Error: No ticker symbol provided."

    normalized_symbol = normalize_ticker(ticker)

    try:
        # Step 1: Fetch ROBUST anchor price (multi-source validated)
        # This prevents "cent-scaling" bugs where history data is 1/100th of reality
        # Anchoring to the validated spot price ensures turnover reflects actual market value.
        anchor_data = await _market_data_fetcher().get_financial_metrics(
            normalized_symbol
        )

        # Step 2: Fetch volume history
        hist = await _market_data_fetcher().get_historical_prices(
            normalized_symbol, period="3mo"
        )

        if hist.empty:
            logger.warning("no_history_found", ticker=ticker)
            return f"""Liquidity Analysis for {ticker}:
Status: FAIL - Insufficient Data
Avg Daily Volume (3mo): N/A
Avg Daily Turnover (USD): N/A
"""

        # Calculate metrics
        avg_volume = hist["Volume"].mean()
        hist_close_mean = hist["Close"].mean()

        # Resolve the per-share price for turnover from data provenance, not from
        # a bare ``.L`` suffix. The fetcher may already have normalized a
        # minor-unit quote (GBp -> GBP); dividing by 100 again is the historical
        # double-normalization bug that 100x-undercounted UK turnover.
        turnover_price = _resolve_turnover_price(
            anchor_data, hist_close_mean, normalized_symbol
        )
        price_for_turnover = turnover_price.value
        if turnover_price.reason in ("anchor_minor_scaled", "lse_history_pence"):
            logger.info(
                "minor_unit_price_scaled",
                ticker=ticker,
                source=turnover_price.source,
                reason=turnover_price.reason,
            )

        # --- UNIT-MISMATCH SENTINEL (regression guard) ---
        # After provenance resolution the anchor price and the raw history mean
        # (scaled by the same minor-unit rule) agree to within ordinary price
        # drift. A ~100x gap means a unit-scale bug slipped through; surface a
        # loud ERROR rather than letting it masquerade as a liquidity hard fail.
        if turnover_price.source == "anchor":
            raw_history_mean = _coerce_positive_float(hist_close_mean)
            if raw_history_mean is not None:
                history_scaled = raw_history_mean * (
                    0.01 if normalized_symbol.endswith(".L") else 1.0
                )
                if _near_100x_mismatch(turnover_price.value, history_scaled):
                    logger.error(
                        "liquidity_unit_mismatch_suspected",
                        ticker=ticker,
                        anchor_price=turnover_price.value,
                        history_scaled=history_scaled,
                    )
                    return f"""Liquidity Analysis for {ticker}:
Status: ERROR
Error: liquidity unit mismatch detected (turnover off by ~100x); rerun after unit reconciliation.
"""

        # --- HEARTBEAT CHECK (Trap A: Liquidity Distortion) ---
        # Detect irregular trading patterns that create stale/manipulated pricing
        total_days = len(hist)
        zero_vol_days = (hist["Volume"] == 0).sum()
        flat_days = (hist["Close"] == hist["Close"].shift(1)).sum()
        pct_zero = (zero_vol_days / total_days) * 100 if total_days > 0 else 0
        pct_flat = (flat_days / total_days) * 100 if total_days > 0 else 0

        # Local turnover (price already in major currency units).
        avg_turnover_local = avg_volume * price_for_turnover

        # Determine currency and FX rate based on resolution
        suffix = get_ticker_suffix(normalized_symbol)
        res = resolve_local_trading_currency(ticker=normalized_symbol)
        resolution_source: str
        if res.code:
            currency = res.code
            resolution_source = res.source
        elif not suffix or suffix == ".US":
            currency = "USD"
            resolution_source = "fallback_us_listing"
        else:
            logger.warning(
                "liquidity_currency_unresolved",
                ticker=ticker,
                resolution_source=res.source,
            )
            return f"""Liquidity Analysis for {ticker}:
Status: ERROR
Error: Could not determine trading currency for turnover conversion.
"""

        # Get FX rate dynamically (with fallback to static rates)
        fx_rate, fx_source = await get_fx_rate(currency, "USD", allow_fallback=True)

        if fx_rate is None:
            # Total FX failure - assume 1.0 and flag as uncertain
            fx_rate = 1.0
            fx_source = "assumed"
            logger.warning(
                "fx_rate_unavailable_using_1.0", ticker=ticker, currency=currency
            )

        logger.info(
            "liquidity_fx_conversion",
            ticker=ticker,
            resolution_source=resolution_source,
            currency=currency,
            fx_rate=fx_rate,
            source=fx_source,
        )

        avg_turnover_usd = avg_turnover_local * fx_rate

        # Thresholds (Aligned with Portfolio Manager Prompt)
        THRESHOLD_PASS = LIQUIDITY_PASS_USD
        THRESHOLD_MARGINAL = LIQUIDITY_MIN_USD

        # Failure Conditions
        fails_zero_vol = pct_zero > 15.0
        fails_flat_price = pct_flat > 30.0

        # Determine status with priority: heartbeat issues > insufficient liquidity > marginal > pass
        status = "PASS"
        reasons = []

        if fails_zero_vol or fails_flat_price:
            status = "FAIL (Irregular Trading)"
            if fails_zero_vol:
                reasons.append(f"{int(pct_zero)}% zero-volume days")
            if fails_flat_price:
                reasons.append(f"{int(pct_flat)}% flat-price days")

        elif avg_turnover_usd < THRESHOLD_MARGINAL:
            status = "FAIL (Insufficient Liquidity)"
            reasons.append(
                f"${int(avg_turnover_usd):,} < ${THRESHOLD_MARGINAL:,} minimum"
            )

        elif avg_turnover_usd < THRESHOLD_PASS:
            status = "MARGINAL"
            reasons.append(f"Low liquidity (${int(avg_turnover_usd):,})")

        # else: status remains "PASS"

        # Build status line with reasons if applicable
        status_line = status
        if reasons:
            status_line = f"{status} - {'; '.join(reasons)}"

        agent_note = ""
        if fails_zero_vol or fails_flat_price:
            agent_note = (
                "\nAGENT INSTRUCTION: This is a definitive hard fail based on measured "
                "exchange data. Do not substitute volume, turnover, or liquidity figures "
                "from another listing, a different exchange, or your training knowledge. "
                "Report this ticker as illiquid on this exchange."
            )

        return f"""Liquidity Analysis for {ticker}:
Status: {status_line}
Avg Daily Volume (3mo): {int(avg_volume):,}
Avg Daily Turnover (USD): ${int(avg_turnover_usd):,}
Trading Regularity: {pct_zero:.0f}% zero-volume days, {pct_flat:.0f}% flat-price days (last 3mo)
Details: {currency} turnover converted at FX rate {fx_rate:.6f} (source: {fx_source}); price_source={turnover_price.source}, unit={turnover_price.reason}
Thresholds: ${LIQUIDITY_MIN_USD:,} USD minimum (MARGINAL), ${LIQUIDITY_PASS_USD:,} USD recommended (PASS), <15% zero-volume days, <30% flat-price days{agent_note}
"""

    except Exception as e:
        summary = summarize_exception(e, operation="liquidity_calculation")
        logger.error(
            "liquidity_calculation_failed", ticker=ticker, exc_info=True, **summary
        )
        return f"""Liquidity Analysis for {ticker}:
Status: ERROR
Error: {summary["error_type"]} (details in operator logs)
"""
