"""Portfolio-level health checks and correlated-sell handling."""

from __future__ import annotations

import structlog

from src.ibkr.models import AnalysisRecord, NormalizedPosition, PortfolioSummary
from src.ibkr.reconciliation_rules import _EXCHANGE_LONG_NAMES

logger = structlog.get_logger(__name__)


def compute_portfolio_health(
    positions: list[NormalizedPosition],
    analyses: dict[str, AnalysisRecord],
    portfolio: PortfolioSummary,
    max_age_days: int = 14,
    reconciliation_items: list | None = None,
    correlated_window_days: int = 14,
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
                if not correlated_event and total_held > 0:
                    total_verdict_ratio = len(verdict_sells) / total_held
                    correlated_event = (
                        len(verdict_sells) >= 8 and total_verdict_ratio >= 0.40
                    )
                    if correlated_event:
                        peak_count = len(verdict_sells)
                        peak_anchor = max(all_dates)

                if correlated_event and peak_anchor is not None:
                    flags.append(
                        f"CORRELATED_SELL_EVENT: {peak_count} positions changed verdict"
                        f" within {correlated_window_days}d of {peak_anchor.isoformat()}"
                        f" ({peak_count / total_held:.0%} of held"
                        f" positions) — probable macro event. Execute stop-breach SELLs"
                        f" on fundamentally weak positions only; review others before acting."
                    )
                    for item in reconciliation_items:
                        if item.action == "SELL" and item.sell_type == "SOFT_REJECT":
                            item.action = "REVIEW"
                            item.urgency = "MEDIUM"
                            item.reason += (
                                "  [MACRO_WATCH: demoted from SELL — correlated"
                                " event detected]"
                            )
                        elif item.action == "SELL" and item.sell_type == "STOP_BREACH":
                            analysis = item.analysis
                            if (
                                analysis is not None
                                and (analysis.health_adj or 0.0) >= 50.0
                                and (analysis.growth_adj or 0.0) >= 50.0
                            ):
                                item.action = "REVIEW"
                                item.urgency = "MEDIUM"
                                item.reason += (
                                    "  [MACRO_STOP: stop breach during correlated event"
                                    " — fundamentals intact (health"
                                    f" {analysis.health_adj:.0f}%, growth"
                                    f" {analysis.growth_adj:.0f}%); review before executing]"
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
