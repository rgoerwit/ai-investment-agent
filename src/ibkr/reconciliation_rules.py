"""Shared reconciliation rules and ticker/FX helper logic."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Literal

import structlog

from src.exchange_metadata import IBKR_TO_YFINANCE
from src.fx_normalization import get_fx_rate_fallback
from src.ibkr.models import ActionBasis, AnalysisRecord, NormalizedPosition
from src.ibkr.portfolio_defaults import (
    DEFAULT_DRIFT_PCT,
    DEFAULT_MAX_AGE_DAYS,
    DEFAULT_SELL_CONFIRMATION_MIN_SPACING_DAYS,
)

if TYPE_CHECKING:
    from src.ibkr.buy_stability import PriorVerdict

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

_CURRENCY_EQUIVALENTS = {"GBX": "GBP", "GBP": "GBX"}  # same economy, unit scaled


def base_match_allowed(pos: NormalizedPosition, analysis: AnalysisRecord) -> bool:
    """Whether a base-symbol fallback match is safe for this position.

    A position may borrow the single base-matched analysis only when the
    analysis currency agrees — otherwise it is a different listing entirely
    (the AGS.BR-shows-AGS.SI bug). This applies to suffix-less positions too:
    a Brussels AGS that IBKR reports as SMART/EUR must not adopt an SGD
    ``AGS.SI`` analysis just because no ``.BR`` analysis exists to poison the
    base. Unknown currency on either side fails closed: losing an analysis
    match is safer than attaching another issuer's research to a live holding.
    """
    a_ccy = (getattr(analysis, "currency", "") or "").upper()
    p_ccy = (pos.currency or "").upper()
    if not a_ccy or not p_ccy:
        return False
    return a_ccy == p_ccy or _CURRENCY_EQUIVALENTS.get(a_ccy) == p_ccy


def analysis_identity_verified(
    pos: NormalizedPosition, analysis: AnalysisRecord
) -> bool:
    """Return whether a held position is safely linked to its analysis.

    A conid proves which IBKR contract is held; an exact, exchange-qualified
    yfinance key proves which saved analysis is attached. SMART plus a non-USD
    currency is only a currency-derived suffix guess and therefore cannot
    authorize a sale. US SMART/USD positions are the intentional suffix-less
    exception.
    """
    if (
        pos.conid <= 0
        or not pos.ticker_identity_verified
        or analysis.ticker.upper() != pos.ticker.yf.upper()
    ):
        return False
    if not base_match_allowed(pos, analysis):
        return False
    exchange = (pos.ticker.exchange or "").upper()
    currency = (pos.currency or pos.ticker.currency or "").upper()
    if exchange in {"", "SMART"}:
        return currency in {"", "USD"} and not pos.ticker.has_suffix
    return pos.ticker.exchange_resolved


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


def _normalize_zone(raw: str | None) -> str:
    """Normalize a PM risk-zone string to its canonical uppercase token."""
    return (raw or "").strip().upper()


_REJECT_VERDICTS = frozenset({"DO_NOT_INITIATE", "SELL", "REJECT"})
SCREEN_REVIEW_DNI_ZONES = frozenset({"LOW", "MODERATE"})
_PROFIT_TAKE_MIN_GAIN_PCT = 25.0
_PROFIT_TAKE_RISK_LARGE_GAIN_PCT = 50.0
_CAPITAL_IDLE_CASH_RISK = "CAPITAL_IDLE_CASH_RISK"
_CAPITAL_IDLE_CASH_SEVERE = "CAPITAL_IDLE_CASH_SEVERE"


@dataclass(frozen=True)
class ProfitTakeDecision:
    """Advisory profit-take classification — ``action`` is always "REVIEW".

    The Literal retains "SELL" only for legacy artifact compatibility;
    classify_profit_take stopped granting executable sell authority in the
    July 2026 retail alignment (selling an intact winner is a capital-gains
    decision the operator makes, not the reconciler).
    """

    qualifies: bool
    action: Literal["SELL", "REVIEW"] | None = None
    reasons: tuple[str, ...] = ()
    cost_basis_return_pct: float | None = None


def check_staleness(
    analysis: AnalysisRecord,
    current_price_local: float | None = None,
    max_age_days: int = DEFAULT_MAX_AGE_DAYS,
    drift_threshold_pct: float = DEFAULT_DRIFT_PCT,
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


def check_review_level_breach(
    analysis: AnalysisRecord,
    current_price_local: float,
) -> bool:
    """Check whether price crossed the legacy downside review level.

    This is a refresh/urgency signal only. It never grants sell authority.
    ``stop_price`` remains the persisted field name for old TRADE_BLOCK data.
    """
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
                hint=(
                    "Possible currency-unit mismatch or stale downside level "
                    "from a different analysis"
                ),
            )
            return False
        return current_price_local < stop
    return False


# Backward-compatible name for existing callers and saved-contract tests.
check_stop_breach = check_review_level_breach


def check_base_case_reference_reached(
    analysis: AnalysisRecord,
    current_price_local: float,
) -> bool:
    """Check whether price reached the legacy base-case valuation reference."""
    target = analysis.target_1_price
    if target and current_price_local > 0:
        return current_price_local >= target
    return False


# Backward-compatible name for existing callers and saved-contract tests.
check_target_hit = check_base_case_reference_reached


def _cost_basis_return_pct(position: NormalizedPosition) -> float | None:
    """Return current price return vs IBKR average cost, guarding bad cost data."""
    if position.avg_cost_local <= 0 or position.current_price_local <= 0:
        return None
    return (
        (position.current_price_local - position.avg_cost_local)
        / position.avg_cost_local
        * 100
    )


def classify_profit_take(
    *,
    analysis: AnalysisRecord,
    position: NormalizedPosition,
    target_hit: bool,
) -> ProfitTakeDecision:
    """Classify advisory capital-allocation reviews for appreciated holdings."""
    health_ok = (analysis.health_adj or 0.0) >= 50.0
    growth_ok = (analysis.growth_adj or 0.0) >= 50.0
    if not (health_ok and growth_ok):
        return ProfitTakeDecision(False)

    gain_pct = _cost_basis_return_pct(position)
    if gain_pct is None or gain_pct < _PROFIT_TAKE_MIN_GAIN_PCT:
        return ProfitTakeDecision(False, cost_basis_return_pct=gain_pct)

    capital_flags = set(analysis.capital_flag_types)
    reasons: list[str] = []
    severe_idle_cash = _CAPITAL_IDLE_CASH_SEVERE in capital_flags
    if severe_idle_cash:
        reasons.append("capital_idle_cash_severe")
    elif _CAPITAL_IDLE_CASH_RISK in capital_flags and target_hit:
        reasons.append("capital_idle_cash_risk_plus_target_hit")
    elif (
        _CAPITAL_IDLE_CASH_RISK in capital_flags
        and gain_pct >= _PROFIT_TAKE_RISK_LARGE_GAIN_PCT
    ):
        reasons.append("capital_idle_cash_risk_plus_large_gain")

    if not reasons:
        return ProfitTakeDecision(False, cost_basis_return_pct=gain_pct)

    # Profit-taking is ALWAYS advisory (July 2026, retail alignment): selling an
    # intact winner is a capital-gains event and a portfolio choice, not a
    # thesis conclusion — the system surfaces the case (gain, idle-cash flags,
    # tax context) and the operator decides. tax_term is annotated, never used
    # to grant sell authority: IBKR lot data is not ingested, so the field is
    # effectively UNKNOWN in practice and must not pretend precision.
    tax_term = position.tax_term
    if tax_term == "SHORT_TERM":
        reasons.append("short_term_tax")
    elif tax_term == "UNKNOWN":
        reasons.append("unknown_tax_term")
    return ProfitTakeDecision(
        True,
        action="REVIEW",
        reasons=tuple(reasons),
        cost_basis_return_pct=gain_pct,
    )


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
    """Classify a legacy sell/review reason for compatibility and grouping."""
    if stop_breached:
        return "STOP_BREACH"
    if analysis is None:
        return "HARD_REJECT"
    # M&A take-private is event-driven, not fundamental — surface the deal
    # context to the operator regardless of zone/score routing below.
    if analysis.m_and_a_status == "ACTIVE_TENDER":
        return "SPECIAL_SITUATION_EXIT"
    health_ok = (analysis.health_adj or 0.0) >= 50.0
    growth_ok = (analysis.growth_adj or 0.0) >= 50.0
    return "SOFT_REJECT" if (health_ok and growth_ok) else "HARD_REJECT"


# A legacy downside level this close to price sits inside routine daily noise.
REVIEW_LEVEL_NOISE_BAND_PCT = 5.0


def review_level_note(
    analysis: AnalysisRecord | None, current_price_local: float | None
) -> str | None:
    """Flag a legacy downside level too close to price to be decision-useful.

    The level does not ratchet after gains and never acts as a sale trigger;
    raising it mechanically would recreate trailing-stop behavior.
    """
    if analysis is None or not analysis.stop_price or not current_price_local:
        return None
    if not 0 < analysis.stop_price < current_price_local:
        return None
    stop_gap_pct = (
        (current_price_local - analysis.stop_price) / current_price_local * 100
    )
    if stop_gap_pct < REVIEW_LEVEL_NOISE_BAND_PCT:
        return (
            f"⚠ review level within {stop_gap_pct:.1f}% of price — inside daily "
            "noise; treat it as context, not an action trigger"
        )
    return None


# Backward-compatible alias; new code uses the retail-framed name above.
stop_staleness_note = review_level_note


@dataclass(frozen=True)
class HeldDisposition:
    """The portfolio-level disposition of a held position with a reject verdict."""

    action: Literal["SELL", "REVIEW"]
    basis: ActionBasis
    executable: bool = False
    detail: str = ""


def _gate_scores_intact(analysis: AnalysisRecord) -> bool:
    """Both thesis gate scores at/above the 50% hard gates.

    Deliberately narrow: this tests only health_adj/growth_adj, not general
    quality — an intact-score reject loses automatic sale authority. Mandatory
    restrictions retain sale authority; downside-price breaches, tender
    mechanics, and data-quality concerns remain review-only.
    """
    return (analysis.health_adj or 0.0) >= 50.0 and (analysis.growth_adj or 0.0) >= 50.0


def reject_confirmed(
    analysis: AnalysisRecord,
    prior_history: Sequence[PriorVerdict],
    *,
    min_spacing_days: int = DEFAULT_SELL_CONFIRMATION_MIN_SPACING_DAYS,
) -> bool:
    """Whether a held-position reject verdict is confirmed by prior history.

    Confirmation requires provably full-mode analyses on BOTH sides: the
    current analysis must be full-mode (a screening-tier verdict never
    executes a sell), and the most recent *provably full-mode* prior must also
    reject, at least ``min_spacing_days`` before the current analysis — one
    bad data day re-analyzed twice must not self-confirm. Mode-unknown
    records (``is_quick_mode is None``, legacy artifacts) carry no sell
    authority on either side. An intervening full-mode non-reject breaks
    confirmation (the thesis recovered in between).
    """
    if analysis.is_quick_mode is not False:
        return False
    fulls = [
        record
        for record in prior_history
        if record.is_quick_mode is False and record.verdict
    ]
    if not fulls:
        return False
    latest = max(fulls, key=lambda record: record.analysis_dt)
    if _normalize_verdict(latest.verdict) not in _REJECT_VERDICTS:
        return False
    try:
        current_date = date.fromisoformat(analysis.analysis_date)
    except (ValueError, TypeError):
        return False
    return (current_date - latest.analysis_dt.date()).days >= min_spacing_days


def classify_disposition(
    analysis: AnalysisRecord,
    *,
    current_price_local: float,
    prior_history: Sequence[PriorVerdict],
    min_spacing_days: int = DEFAULT_SELL_CONFIRMATION_MIN_SPACING_DAYS,
) -> HeldDisposition:
    """Classify the portfolio disposition of a held position whose verdict rejects.

    A portfolio action requires portfolio evidence: a stock-level rejection may
    trigger refresh, review, replacement analysis, or an exit, but it must not
    decide among those alone. A legacy downside-level breach may raise urgency
    in the caller but has no sell authority. This function decides SELL vs
    REVIEW for reject verdicts, most-structural evidence first.
    """
    evidence = analysis.evidence

    if evidence.mandatory_exit_flag_types:
        return HeldDisposition(
            "SELL",
            "MANDATORY_EXIT",
            executable=True,
            detail=f"Restriction: {', '.join(evidence.mandatory_exit_flag_types)}",
        )

    # An active tender is a deal-mechanics decision (tender vs market sale vs
    # hold for a bump), not a verdict consequence. Deliberate widening vs the
    # legacy path, which auto-sold these inside the reject branch.
    if analysis.m_and_a_status == "ACTIVE_TENDER":
        return HeldDisposition(
            "REVIEW",
            "SPECIAL_SITUATION_REVIEW",
            detail="Active tender — decide on deal premium/mechanics, not verdict",
        )

    # Buy-blocking flags mean the gate arithmetic is indeterminate — the reject
    # cannot be trusted in either direction. Review, never exit, on model doubt.
    if evidence.buy_blocking_flag_types:
        return HeldDisposition(
            "REVIEW",
            "DATA_QUALITY",
            detail=(
                "Gate scores unreliable "
                f"({', '.join(evidence.buy_blocking_flag_types)}) — "
                "re-run before acting"
            ),
        )

    # Intact gate scores strip the verdict of automatic SELL authority — the
    # check deliberately precedes confirmation, otherwise every persistent
    # entry-screen rejection of a winner would escalate to SELL on its next
    # refresh ≥ min_spacing_days later (the 2026-07-14 nine-sell day). Exits
    # for this class require a future fundamental failure or mandatory
    # restriction — never price levels or repeated entry-screen rejects alone.
    if _gate_scores_intact(analysis):
        entry = analysis.entry_price or analysis.current_price
        appreciated = bool(
            entry and current_price_local > 0 and current_price_local >= entry
        )
        if appreciated or evidence.dni_review_candidate:
            # "Wouldn't initiate at this price" at/above entry is the GARP
            # screen rejecting its own winner, not exit evidence.
            return HeldDisposition(
                "REVIEW",
                "ENTRY_CONSTRAINT",
                detail=(
                    "Fundamentals intact; verdict reflects entry screen, "
                    "not thesis failure"
                ),
            )
        return HeldDisposition(
            "REVIEW",
            "THESIS_REASSESSMENT",
            detail=(
                "Gate scores intact — price weakness alone is not exit "
                "evidence; watch fundamentals, not price"
            ),
        )

    if reject_confirmed(analysis, prior_history, min_spacing_days=min_spacing_days):
        return HeldDisposition(
            "SELL",
            "CONFIRMED_THESIS_FAILURE",
            executable=True,
            detail="Reject confirmed by prior full-mode analysis",
        )

    return HeldDisposition(
        "REVIEW",
        "THESIS_REASSESSMENT",
        detail="Unconfirmed reject — refresh analysis before exiting",
    )
