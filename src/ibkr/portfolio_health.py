"""Portfolio-level health checks and correlated-sell handling."""

from __future__ import annotations

import structlog

from src.fx_normalization import comparable_prices
from src.ibkr.models import AnalysisRecord, NormalizedPosition, PortfolioSummary
from src.ibkr.portfolio_defaults import (
    DEFAULT_EXCHANGE_LIMIT_PCT,
    DEFAULT_MAX_AGE_DAYS,
)
from src.ibkr.reconciliation_rules import _EXCHANGE_LONG_NAMES

logger = structlog.get_logger(__name__)

# The macro override (demote SELLs→REVIEW + enable buy-the-dip) is a TRANSIENT-shock
# protection: it applies only this many days after the event onset, then lapses so
# impaired positions (incl. structural events) can exit and dip-buys stop. Kept short
# (≈3 weeks) regardless of the event record's longer expiry, which still drives the
# ongoing-awareness banner and forced re-analysis.
_DEFAULT_MACRO_OVERRIDE_MAX_AGE_DAYS = 21

# Canonical matcher for the CORRELATED_SELL_EVENT flag this module emits (see the
# per-trigger `evidence` strings in compute_portfolio_health). It lives beside the
# emitter so the shape and its readers move together.
#
# THREE trigger phrasings must all parse:
#   window           -> "N positions changed verdict within 14d of DATE (P% of held positions)"
#   cumulative       -> "N positions changed verdict across the held book as of DATE (P% ...)"
#   drawdown_breadth -> "N positions currently trading >=10% below entry as of DATE (P% ...)"
#
# `window` is therefore None for two of the three -- those phrasings carry no window, and
# a reader that requires one silently drops the flag. That is exactly what happened: the
# dashboard hardcoded the `window` phrasing and rendered an empty macro alert on the other
# two triggers. Consumers: web/ibkr_dashboard/macro_alerts.py,
# scripts/portfolio_manager.py, ibkr/portfolio_report_status.py.
#
# The percentage is emitted with `:.0%`, so it is always integral.
CORRELATED_EVENT_EVIDENCE_PATTERN = (
    r"(?P<count>\d+) positions"
    r".*?(?:within (?P<window>\d+)d of|as of) (?P<date>\d{4}-\d{2}-\d{2})"
    r".*?\((?P<pct>\d+)% of held positions\)"
)


def _entry_and_current(item) -> tuple[float | None, float | None]:
    """Return (analysis entry, current position price) in a common denomination.

    Returns ``(None, None)`` when the two sides are not comparable — a different
    economy, or an unlabelled currency on either side. Both callers treat that as
    "no signal", which is correct: an incomparable pair is not evidence a
    position is down, and reading it as such produced the GAMA.L/MEGP.L
    fabricated ~99% drawdowns.
    """
    analysis = getattr(item, "analysis", None)
    pos = getattr(item, "ibkr_position", None)
    if analysis is None or pos is None:
        return None, None
    pair = comparable_prices(
        analysis.entry_price or analysis.current_price,
        getattr(analysis, "currency", None),
        pos.current_price_local,
        getattr(pos, "currency", None),
    )
    if pair is None:
        return None, None
    return pair.left, pair.right


def _below_entry(item) -> bool:
    entry, current = _entry_and_current(item)
    return entry is not None and current is not None and current < entry


def is_macro_event_evidence(item) -> bool:
    """Whether a reconciliation item is evidence of a correlated market event.

    Keys on ``action_basis`` (the decision layer), not the final action —
    permanent reviews must not erase event breadth. ENTRY_CONSTRAINT items are
    deliberately excluded: a winner appreciating out of the entry screen is a
    verdict flip with the price UP, which is screen behavior, not distress.
    Price-down thesis reassessments count; price-up/flat ones do not.
    Legacy items without a basis fall back to the pre-basis SELL predicate.

    This is the single canonical predicate — the CLI report must import it,
    not re-declare the tuple (the scripts/portfolio_manager.py:959 duplicate).
    """
    basis = getattr(item, "action_basis", None)
    if basis is not None:
        if basis in ("STOP_LOSS", "CONFIRMED_THESIS_FAILURE"):
            return True
        return basis == "THESIS_REASSESSMENT" and _below_entry(item)
    return item.action == "SELL" and item.sell_type in (
        "HARD_REJECT",
        "SOFT_REJECT",
        "STOP_BREACH",
    )


def _apply_macro_demotions(
    reconciliation_items: list, soft_tag: str, stop_tag_template: str
) -> int:
    """Demote macro-driven SELLs to REVIEW; returns number demoted.

    Fundamentals-intact rejections and price-level breaches are REVIEW at
    source. This compatibility path also demotes every legacy STOP_BREACH sell,
    regardless of score: old artifacts must not recover price-only authority.
    CONFIRMED_THESIS_FAILURE never demotes because two spaced full analyses
    agreed on a fundamental failure.
    """
    demoted = 0
    for item in reconciliation_items:
        if (
            item.action == "SELL"
            and item.sell_type == "SOFT_REJECT"
            and getattr(item, "action_basis", None) is None
        ):
            item.action = "REVIEW"
            item.urgency = "MEDIUM"
            item.reason += soft_tag
            demoted += 1
        elif item.action == "SELL" and item.sell_type == "STOP_BREACH":
            analysis = item.analysis
            item.action = "REVIEW"
            item.urgency = "HIGH"
            item.suggested_quantity = None
            item.cash_impact_usd = 0.0
            item.settlement_date = None
            if analysis is not None:
                item.reason += stop_tag_template.format(
                    health=analysis.health_adj or 0.0,
                    growth=analysis.growth_adj or 0.0,
                )
            else:
                item.reason += (
                    "  [MACRO_PRICE: legacy price-trigger sale downgraded — "
                    "refresh fundamentals before acting]"
                )
            demoted += 1
    return demoted


def compute_portfolio_health(
    positions: list[NormalizedPosition],
    analyses: dict[str, AnalysisRecord],
    portfolio: PortfolioSummary,
    max_age_days: int = DEFAULT_MAX_AGE_DAYS,
    reconciliation_items: list | None = None,
    correlated_window_days: int = 14,
    drawdown_pct: float = 10.0,
    drawdown_breadth_ratio: float = 0.35,
    cumulative_fallback_ratio: float = 0.35,
    active_macro_events: list | None = None,
    macro_override_max_age_days: int = _DEFAULT_MACRO_OVERRIDE_MAX_AGE_DAYS,
    exchange_limit_pct: float = DEFAULT_EXCHANGE_LIMIT_PCT,
) -> list[str]:
    """Compute portfolio-level health flags using data already in held analyses."""
    if not positions or portfolio.portfolio_value_usd <= 0:
        return []

    flags: list[str] = []
    total_weight = 0.0
    weighted_health = 0.0
    weighted_growth = 0.0
    health_count = 0
    growth_count = 0
    stale_count = 0
    stale_in_queue_count = 0
    stale_need_refresh_count = 0
    currency_weights: dict[str, float] = {}
    scored_health: list[tuple[str, float, bool]] = []
    scored_growth: list[tuple[str, float, bool]] = []
    reconciliation_by_ticker: dict[str, tuple[str, str | None]] = {}

    if reconciliation_items:
        for item in reconciliation_items:
            ticker = getattr(getattr(item, "ticker", None), "yf", None)
            if not ticker or ticker in reconciliation_by_ticker:
                continue
            reconciliation_by_ticker[ticker] = (
                getattr(item, "action", ""),
                getattr(item, "sell_type", None),
            )

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
                action, sell_type = reconciliation_by_ticker.get(
                    pos.ticker.yf, ("", None)
                )
                if action in {"SELL", "TRIM"} or sell_type == "SOFT_REJECT":
                    stale_in_queue_count += 1
                else:
                    stale_need_refresh_count += 1

        ccy = (pos.currency or "USD").upper()
        currency_weights[ccy] = currency_weights.get(ccy, 0.0) + weight * 100

    def _worst_detail(
        scored: list[tuple[str, float, bool]],
        max_age_days: int,
        n: int = 5,
    ) -> str:
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
                f"CURRENCY_CONCENTRATION: {pct:.1f}% in {ccy} — FX risk amplification"
            )

    if positions:
        stale_pct = stale_count / len(positions) * 100
        if stale_pct > 30:
            stale_message = (
                f"STALE_ANALYSIS_RATIO: {stale_count}/{len(positions)} positions"
                f" ({stale_pct:.0f}%) have analyses older than {max_age_days}d"
            )
            if reconciliation_items is not None:
                stale_message += (
                    f" — {stale_in_queue_count} already in sell/review queue,"
                    f" {stale_need_refresh_count} still need refreshed analysis"
                    " before action (see ANALYSIS FRESHNESS section)"
                )
            else:
                stale_message += (
                    " — flying blind on significant chunk of portfolio"
                    " (re-run with --refresh-stale to update)"
                )
            flags.append(stale_message)

    for exch, pct in portfolio.exchange_weights.items():
        if pct > exchange_limit_pct:
            long_name = _EXCHANGE_LONG_NAMES.get(exch, exch)
            flags.append(
                f"GEOGRAPHY_CONCENTRATION: {pct:.1f}% in {exch} ({long_name})"
                " — single-exchange concentration"
            )

    if reconciliation_items is not None:
        from datetime import date as _date
        from datetime import timedelta as _td

        # Event evidence: thesis/verdict failures plus downside-review breaches;
        # a burst of price-level breaches is the clearest same-time shock signal.
        # PROFIT_TAKE stays excluded (capital-allocation exits are firm-specific),
        # as are ENTRY_CONSTRAINT reviews (winners appreciating out of the entry
        # screen are not distress). Keyed on action_basis via the canonical
        # predicate so permanent reviews still count as breadth evidence.
        event_sells = [
            item for item in reconciliation_items if is_macro_event_evidence(item)
        ]
        total_held = sum(
            1 for item in reconciliation_items if item.ibkr_position is not None
        )

        correlated_event = False
        peak_count = 0
        peak_anchor = None
        trigger = ""

        if event_sells and total_held > 0:
            dated: list[tuple] = []
            for item in event_sells:
                if item.analysis and item.analysis.analysis_date:
                    try:
                        d = _date.fromisoformat(item.analysis.analysis_date)
                        dated.append((item, d))
                    except ValueError:
                        pass

            if dated:
                all_dates = [d for _, d in dated]
                for anchor in all_dates:
                    window_end = anchor + _td(days=correlated_window_days - 1)
                    count = sum(1 for d in all_dates if anchor <= d <= window_end)
                    if count > peak_count:
                        peak_count = count
                        peak_anchor = anchor

                correlated_event = peak_count >= 5 and peak_count / total_held >= 0.25
                trigger = "window" if correlated_event else ""

                if not correlated_event:
                    # Cumulative fallback: refresh throttling smears verdict
                    # flips across months, so the window can stay sparse while
                    # the book fills with macro-driven sells.
                    total_ratio = len(event_sells) / total_held
                    if (
                        len(event_sells) >= 8
                        and total_ratio >= cumulative_fallback_ratio
                    ):
                        correlated_event = True
                        trigger = "cumulative"
                        peak_count = len(event_sells)
                        peak_anchor = max(all_dates)

                if correlated_event:
                    # Model-shift vs market-event discriminator: a real macro
                    # event pushes prices DOWN and verdicts follow. When the
                    # majority of verdict-flip evidence trades at/above its
                    # analysis entry, the model got more bearish while the
                    # market did not — analyzer-side re-rating, not distress.
                    # A genuine selloff still fires via the drawdown-breadth
                    # trigger below, which is inherently price-down.
                    priced = [
                        item
                        for item in event_sells
                        if _entry_and_current(item) != (None, None)
                    ]
                    up_count = sum(1 for item in priced if not _below_entry(item))
                    if priced and up_count / len(priced) > 0.5:
                        flags.append(
                            f"MODEL_REGIME_SHIFT: {up_count} of {len(priced)}"
                            " verdict-flip positions trade at/above their"
                            " analysis entry — flips are analyzer-side"
                            " (re-rating), not market distress. Audit recent"
                            " prompt/model/threshold changes before acting;"
                            " macro SELL demotions and dip-buying are NOT"
                            " enabled."
                        )
                        logger.info(
                            "model_regime_shift_detected",
                            trigger=trigger,
                            evidence_count=len(event_sells),
                            priced_count=len(priced),
                            at_or_above_entry=up_count,
                        )
                        correlated_event = False
                        trigger = ""
                        peak_count = 0
                        peak_anchor = None

        if not correlated_event and total_held > 0:
            # Drawdown breadth: refresh-schedule-independent price evidence —
            # how much of the held book trades well below its analysis entry
            # right now. Shares `_entry_and_current` with the staleness drift
            # check rather than re-deriving the pair: this loop used to inline
            # its own comparison and carried a stale comment claiming prices were
            # "GBX-normalized upstream", describing a x100 that the Aug 2026
            # denomination work deleted.
            drawdown_count = 0
            for item in reconciliation_items:
                entry, current = _entry_and_current(item)
                if entry is None or current is None:
                    continue
                if (entry - current) / entry * 100 >= drawdown_pct:
                    drawdown_count += 1
            if (
                drawdown_count >= 8
                and drawdown_count / total_held >= drawdown_breadth_ratio
            ):
                correlated_event = True
                trigger = "drawdown_breadth"
                peak_count = drawdown_count
                peak_anchor = _date.today()

        # The macro override (demote SELLs→REVIEW + enable buy-the-dip) protects
        # against panic-selling a TRANSIENT shock — so it applies only briefly after
        # onset, then LAPSES. Fresh verdict/cumulative evidence is evaluated apart
        # from stored events: an old lapsed event must not suppress a genuinely new
        # selloff. Drawdown-breadth evidence is different because it re-anchors to
        # today every run; if an old active event already explains the drawdown, do
        # not restart the override from drawdown alone.
        today = _date.today()
        fresh_onset = (
            peak_anchor if correlated_event and peak_anchor is not None else None
        )
        fresh_within_window = (
            fresh_onset is not None
            and (today - fresh_onset).days <= macro_override_max_age_days
        )

        active_event_onsets: list[tuple[object, _date]] = []
        undated_active_event = None
        for ev in active_macro_events or []:
            raw = getattr(ev, "event_date", "") or getattr(ev, "detected_date", "")
            try:
                active_event_onsets.append((ev, _date.fromisoformat(raw)))
            except (ValueError, TypeError):
                if undated_active_event is None:
                    undated_active_event = ev

        active_within_window_event = None
        lapsed_onsets: list[tuple[object | None, _date]] = []
        for ev, active_onset in active_event_onsets:
            if (today - active_onset).days <= macro_override_max_age_days:
                if active_within_window_event is None:
                    active_within_window_event = ev
            else:
                lapsed_onsets.append((ev, active_onset))

        has_lapsed_active_event = bool(lapsed_onsets)
        fresh_override_allowed = bool(
            fresh_within_window
            and (trigger != "drawdown_breadth" or not has_lapsed_active_event)
        )
        if fresh_onset is not None and not fresh_override_allowed:
            lapsed_onsets.append((None, fresh_onset))
        lapsed_event, lapsed_onset = (
            min(lapsed_onsets, key=lambda item: item[1])
            if lapsed_onsets
            else (None, None)
        )
        # A stored macro signal we can't date (degenerate/malformed event record)
        # is not evidence the override has lapsed. Preserve the previous defensive
        # behavior only when there is no dated active/fresh evidence to use.
        fallback_active_event = (
            undated_active_event
            if undated_active_event is not None
            and not active_event_onsets
            and fresh_onset is None
            else None
        )

        if fresh_override_allowed and correlated_event and peak_anchor is not None:
            # Truthful per-trigger phrasing. The "(within Nd of|as of) DATE"
            # shape is a parsing contract with _store_macro_event_if_detected
            # and the report banner — keep them in sync.
            if trigger == "window":
                evidence = (
                    f"changed verdict within {correlated_window_days}d"
                    f" of {peak_anchor.isoformat()}"
                )
            elif trigger == "cumulative":
                evidence = (
                    "changed verdict across the held book"
                    f" as of {peak_anchor.isoformat()}"
                )
            else:  # drawdown_breadth
                evidence = (
                    f"currently trading ≥{drawdown_pct:.0f}% below entry"
                    f" as of {peak_anchor.isoformat()}"
                )
            flags.append(
                f"CORRELATED_SELL_EVENT: {peak_count} positions {evidence}"
                f" ({peak_count / total_held:.0%} of held"
                f" positions) — probable macro event [{trigger}]. Do not sell"
                f" into a correlated drop: exit only on confirmed fundamental"
                f" failure; review everything else."
            )
            demoted = _apply_macro_demotions(
                reconciliation_items,
                soft_tag=(
                    "  [MACRO_WATCH: demoted from SELL — correlated event detected]"
                ),
                stop_tag_template=(
                    "  [MACRO_PRICE: price-drop review during correlated event"
                    " — fundamentals intact (health {health:.0f}%, growth"
                    " {growth:.0f}%); no sale without fundamental failure]"
                ),
            )
            logger.info(
                "correlated_sell_event_detected",
                trigger=trigger,
                peak_date=peak_anchor.isoformat(),
                window_days=correlated_window_days,
                peak_count=peak_count,
                total_held=total_held,
                demoted=demoted,
                pct=f"{peak_count / total_held:.0%}",
            )
        elif (
            active_within_window_event is not None or fallback_active_event is not None
        ):
            # No fresh detection, but a previously detected event is still within
            # the brief override window — sustain the demotion so it doesn't vanish
            # the day after detection.
            event = active_within_window_event or fallback_active_event
            event_type = getattr(event, "event_type", "MACRO")
            expiry = getattr(event, "expiry", "?")
            demoted = _apply_macro_demotions(
                reconciliation_items,
                soft_tag=(
                    f"  [MACRO_WATCH: active {event_type} event"
                    f" until {expiry} — demoted from SELL]"
                ),
                stop_tag_template=(
                    f"  [MACRO_PRICE: price-drop review during active {event_type}"
                    " event — fundamentals intact (health {health:.0f}%,"
                    " growth {growth:.0f}%); no sale without fundamental failure]"
                ),
            )
            if demoted:
                flags.append(
                    f"ACTIVE_MACRO_EVENT: {event_type} event active until"
                    f" {expiry} — {demoted} SELL(s) demoted to REVIEW"
                    " (sustained macro override)."
                )
                logger.info(
                    "active_macro_event_demotions_sustained",
                    event_type=event_type,
                    expiry=expiry,
                    demoted=demoted,
                )
        elif lapsed_onset is not None:
            # Event ongoing but PAST the brief override window: stop demoting so
            # impaired positions can exit, and stop buy-the-dip (no transient
            # mean-reversion to lean on). A distinct flag (not CORRELATED_/
            # ACTIVE_MACRO_EVENT) keeps the dip-buy gate and panic banner off while
            # still flagging that the event is live and verdicts are re-priced.
            days = (today - lapsed_onset).days
            event_type = (
                getattr(lapsed_event, "event_type", "MACRO")
                if lapsed_event is not None
                else "MACRO"
            )
            flags.append(
                f"MACRO_EVENT_ONGOING: {event_type} event onset ~{days}d ago — brief"
                f" macro override (≤{macro_override_max_age_days}d) has lapsed;"
                " confirmed fundamental exits can flow normally, and"
                " buy-the-dip is suppressed (no transient mean-reversion assumed for"
                " a sustained event)."
            )
            logger.info(
                "macro_override_lapsed",
                onset=lapsed_onset.isoformat(),
                days=days,
                event_type=event_type,
            )

    return flags
