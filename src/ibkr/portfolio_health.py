"""Portfolio-level health checks and correlated-sell handling."""

from __future__ import annotations

import structlog

from src.ibkr.models import AnalysisRecord, NormalizedPosition, PortfolioSummary
from src.ibkr.portfolio_defaults import DEFAULT_MAX_AGE_DAYS
from src.ibkr.reconciliation_rules import _EXCHANGE_LONG_NAMES

logger = structlog.get_logger(__name__)


def _apply_macro_demotions(
    reconciliation_items: list, soft_tag: str, stop_tag_template: str
) -> int:
    """Demote macro-driven SELLs to REVIEW; returns number demoted.

    SOFT_REJECT sells always demote. STOP_BREACH sells demote only when
    fundamentals are intact (health and growth both >= 50%) — weak positions
    keep their executable stop.
    """
    demoted = 0
    for item in reconciliation_items:
        if item.action == "SELL" and item.sell_type == "SOFT_REJECT":
            item.action = "REVIEW"
            item.urgency = "MEDIUM"
            item.reason += soft_tag
            demoted += 1
        elif item.action == "SELL" and item.sell_type == "STOP_BREACH":
            analysis = item.analysis
            if (
                analysis is not None
                and (analysis.health_adj or 0.0) >= 50.0
                and (analysis.growth_adj or 0.0) >= 50.0
            ):
                item.action = "REVIEW"
                item.urgency = "MEDIUM"
                item.reason += stop_tag_template.format(
                    health=analysis.health_adj, growth=analysis.growth_adj
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
                f"CURRENCY_CONCENTRATION: {pct:.1f}% in {ccy}"
                " — FX risk amplification"
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
        if pct > 40:
            long_name = _EXCHANGE_LONG_NAMES.get(exch, exch)
            flags.append(
                f"GEOGRAPHY_CONCENTRATION: {pct:.1f}% in {exch} ({long_name})"
                " — single-exchange concentration"
            )

    if reconciliation_items is not None:
        from datetime import date as _date
        from datetime import timedelta as _td

        # Event evidence: thesis/verdict failures PLUS stop breaches — a burst
        # of breached stops is the purest same-time price-shock signal.
        # PROFIT_TAKE stays excluded (capital-allocation exits are firm-specific).
        event_sells = [
            item
            for item in reconciliation_items
            if item.action == "SELL"
            and item.sell_type in ("HARD_REJECT", "SOFT_REJECT", "STOP_BREACH")
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

        if not correlated_event and total_held > 0:
            # Drawdown breadth: refresh-schedule-independent price evidence —
            # how much of the held book trades well below its analysis entry
            # right now. Uses the same entry/current fields as the staleness
            # drift check (both LOCAL currency, GBX-normalized upstream).
            drawdown_count = 0
            for item in reconciliation_items:
                pos = item.ibkr_position
                analysis = item.analysis
                if pos is None or analysis is None:
                    continue
                entry = analysis.entry_price or analysis.current_price
                current = pos.current_price_local
                if not entry or not current or entry <= 0:
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

        if correlated_event and peak_anchor is not None:
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
                f" positions) — probable macro event [{trigger}]. Execute"
                f" stop-breach SELLs on fundamentally weak positions only;"
                f" review others before acting."
            )
            demoted = _apply_macro_demotions(
                reconciliation_items,
                soft_tag=(
                    "  [MACRO_WATCH: demoted from SELL — correlated" " event detected]"
                ),
                stop_tag_template=(
                    "  [MACRO_STOP: stop breach during correlated event"
                    " — fundamentals intact (health {health:.0f}%, growth"
                    " {growth:.0f}%); review before executing]"
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
        elif active_macro_events:
            # No fresh detection, but a previously detected event is still
            # active (unexpired) — sustain the demotion across runs so the
            # override doesn't vanish the day after detection.
            event = active_macro_events[0]
            event_type = getattr(event, "event_type", "MACRO")
            expiry = getattr(event, "expiry", "?")
            demoted = _apply_macro_demotions(
                reconciliation_items,
                soft_tag=(
                    f"  [MACRO_WATCH: active {event_type} event"
                    f" until {expiry} — demoted from SELL]"
                ),
                stop_tag_template=(
                    f"  [MACRO_STOP: stop breach during active {event_type}"
                    " event — fundamentals intact (health {health:.0f}%,"
                    " growth {growth:.0f}%); review before executing]"
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

    return flags
