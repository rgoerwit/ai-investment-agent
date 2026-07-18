"""Position, watchlist, and concentration sections for the portfolio report."""

from __future__ import annotations

from collections.abc import Callable

from src.ibkr.dip_watch import (
    DIP_CONCENTRATION_MIN_SCORE,
    dip_watch_source,
    score_dip_watch_item,
)
from src.ibkr.models import ReconciliationItem
from src.ibkr.portfolio_presentation import (
    SELL_RECOMMENDATIONS_TITLE,
    SELL_RELATED_REVIEWS_TITLE,
)
from src.ibkr.portfolio_report import (
    PortfolioReportContext,
    render_watchlist_optimization,
    select_report_dip_candidates,
)
from src.ibkr.portfolio_report_formatting import (
    DIVIDER,
    ReportBuffer,
    as_of_date,
    bar_chart,
    currency_prefix,
    display_ticker,
    item_currency,
    normalize_reason,
)
from src.ibkr.reconciliation_rules import _EXCHANGE_LONG_NAMES, stop_staleness_note
from src.ibkr.refresh_service import run_ticker_for
from src.sector_normalization import aggregate_sector_weights


def _render_dip_watch(
    candidates: list[ReconciliationItem],
    *,
    analysis_command: Callable[[str], str],
) -> tuple[str, ...]:
    lines = [
        DIVIDER,
        "  DIP WATCH  (existing positions — consider adding)",
        DIVIDER,
        "",
        "  Ranked by fundamental quality × dip depth × risk/reward:",
        "",
    ]
    for item in candidates:
        analysis = item.analysis
        pos = item.ibkr_position
        score = score_dip_watch_item(item)
        stars = (
            "★★★"
            if score >= DIP_CONCENTRATION_MIN_SCORE
            else "★★ "
            if score >= 60
            else "★  "
        )
        symbol = currency_prefix(item_currency(item))
        health = (
            f"{analysis.health_adj:.0f}"
            if analysis and analysis.health_adj is not None
            else "?"
        )
        growth = (
            f"{analysis.growth_adj:.0f}"
            if analysis and analysis.growth_adj is not None
            else "?"
        )
        entry = (
            analysis.entry_price if analysis and analysis.entry_price else None
        ) or (pos.avg_cost_local if pos and pos.avg_cost_local else None)
        entry_label = (
            "analysis entry" if analysis and analysis.entry_price else "IBKR cost basis"
        )
        if entry and pos and pos.current_price_local and entry > 0:
            change = (pos.current_price_local - entry) / entry * 100
            suffix = (
                "  (⚠ unit mismatch?)" if abs(change) >= 90.0 else f"  ({change:+.1f}%)"
            )
            entry_text = (
                f"{entry_label} {symbol}{entry:,.2f}  now "
                f"{symbol}{pos.current_price_local:,.2f}{suffix}"
            )
        else:
            entry_text = "(no entry price recorded)"
        risk_reward = "—"
        if (
            analysis
            and analysis.target_1_price
            and analysis.stop_price
            and pos
            and pos.current_price_local
        ):
            current = pos.current_price_local
            if current > 0 and current > analysis.stop_price:
                upside = (analysis.target_1_price - current) / current * 100
                downside = max((current - analysis.stop_price) / current * 100, 0.001)
                risk_reward = (
                    f"R/R {upside / downside:.1f}×  "
                    f"(target +{upside:.0f}% / stop -{downside:.0f}%)"
                )
        lines.append(
            f"  {stars}  {display_ticker(item):<12}  Health:{health}%  "
            f"Growth:{growth}%  |  {entry_text}  |  {risk_reward}"
        )
        if dip_watch_source(item) == "macro_review":
            as_of = analysis.analysis_date if analysis else "?"
            lines.append(
                "             macro dip — fundamentals intact, review "
                f"{as_of}; standalone verdict was REJECT (often valuation, which "
                "the dip improves) — review before adding"
            )
    lines.extend(("", "  → Re-run before acting:"))
    for item in candidates:
        ticker = run_ticker_for(item)
        suffix_warning = (
            "  ← ⚠ verify exchange suffix (no '.' in ticker)"
            if "." not in ticker
            else ""
        )
        lines.append(f"      {analysis_command(ticker)}{suffix_warning}")
    lines.append("")
    return tuple(lines)


def render_position_and_risk_sections(
    context: PortfolioReportContext,
    *,
    analysis_command: Callable[[str], str],
) -> tuple[str, ...]:
    """Render position-derived sections without recomputing action policy."""
    lines: list[str] = []
    writer = ReportBuffer(
        lines=lines,
        show_recommendations=context.show_recommendations,
        settled_cash_usd=context.cash_summary.settled_cash_usd,
        live_orders=list(context.live_orders),
    )
    groups = context.plan.groups
    sell_recommendations = [
        *groups.stop_sells,
        *groups.hard_sells,
        *groups.profit_take_sells,
        *groups.soft_sells,
    ]
    if sell_recommendations:
        writer.section(
            SELL_RECOMMENDATIONS_TITLE,
            "review sell reason labels before executing",
        )
        for item in sell_recommendations:
            currency = item_currency(item)
            reason = (
                as_of_date(item.reason)
                if item.sell_type == "HARD_REJECT"
                else normalize_reason(item.reason)
            )
            lines.append(
                f"{writer.order_line(item, currency)}  "
                f"[{writer.sell_type_label(item)}] {reason}"
            )
            if item.sell_type in ("STOP_BREACH", "HARD_REJECT"):
                score_line = writer.score_line(item)
                if score_line:
                    lines.append(score_line)
                writer.append_pnl_proceeds(item, currency)
            elif item.sell_type == "SOFT_REJECT":
                details = writer.soft_rejection_score_segments(item)
                thesis = writer.soft_rejection_thesis_segment(item)
                if thesis:
                    details.append(thesis)
                writer.append_wrapped_segments(details)
                writer.append_wrapped_segments(writer.soft_rejection_pnl_segments(item))
            elif item.sell_type == "PROFIT_TAKE":
                writer.append_wrapped_segments(writer.profit_take_segments(item))
            note = writer.order_note(item)
            if note:
                lines.append(note)
            lines.append("")

    sell_reviews = [
        *groups.macro_stop_reviews,
        *groups.macro_reviews,
        *groups.profit_take_reviews,
    ]
    if sell_reviews:
        writer.section(
            SELL_RELATED_REVIEWS_TITLE,
            "tax, macro, or intact-thesis review before acting",
        )
        for item in sell_reviews:
            currency = item_currency(item)
            reason = normalize_reason(
                item.reason.split("  [MACRO_STOP:")[0].split("  [MACRO_WATCH:")[0]
            )
            lines.append(
                f"{writer.order_line(item, currency)}  "
                f"[{writer.sell_type_label(item)}] {reason}"
            )
            if item.sell_type == "PROFIT_TAKE":
                writer.append_wrapped_segments(writer.profit_take_segments(item))
            elif item.sell_type == "SOFT_REJECT":
                details = writer.soft_rejection_score_segments(item)
                thesis = writer.soft_rejection_thesis_segment(item)
                if thesis:
                    details.append(thesis)
                writer.append_wrapped_segments(details)
                writer.append_wrapped_segments(writer.soft_rejection_pnl_segments(item))
            else:
                score_line = writer.score_line(item)
                if score_line:
                    lines.append(score_line)
                writer.append_pnl_proceeds(item, currency)
            if item.action == "REVIEW":
                holding = writer.holding_line(item, currency)
                if holding:
                    lines.append(holding)
                if item.action_basis == "ENTRY_CONSTRAINT":
                    pos = item.ibkr_position
                    ratchet = stop_staleness_note(
                        item.analysis, pos.current_price_local if pos else None
                    )
                    if ratchet:
                        lines.append("             " + ratchet)
            note = writer.order_note(item)
            if note:
                lines.append(note)
            lines.append("")

    dip_candidates = list(select_report_dip_candidates(context.plan))
    if dip_candidates:
        lines.extend(
            _render_dip_watch(dip_candidates, analysis_command=analysis_command)
        )
    withheld_dips = context.plan.concentration_withheld_dips
    if withheld_dips:
        count = len(withheld_dips)
        symbols = ", ".join(display_ticker(item) for item in withheld_dips)
        lines.extend(
            (
                f"  ({count} dip candidate{'s' if count != 1 else ''} withheld "
                f"— overweight bucket, below ★★★: {symbols})",
                "",
            )
        )

    if groups.trims:
        writer.section("TRIMS", "reduce to target weight")
        for item in groups.trims:
            currency = item_currency(item)
            lines.append(
                f"{writer.order_line(item, currency)}  {normalize_reason(item.reason)}"
            )
            writer.append_pnl_proceeds(item, currency)
            note = writer.order_note(item)
            if note:
                lines.append(note)
            lines.append("")

    if groups.adds:
        writer.section("ADDS", "increase underweight positions")
        for item in groups.adds:
            currency = item_currency(item)
            lines.append(
                f"{writer.order_line(item, currency)}  {normalize_reason(item.reason)}"
            )
            pos = item.ibkr_position
            if pos and pos.quantity:
                symbol = currency_prefix(currency)
                average = (
                    f" @ avg {symbol}{pos.avg_cost_local:,.2f}"
                    if pos.avg_cost_local
                    else ""
                )
                lines.append(
                    f"             [upping position — currently hold "
                    f"{pos.quantity:,.0f} shares{average}]"
                )
            cost = writer.cost_line(item)
            if cost:
                lines.append(cost)
            note = writer.order_note(item)
            if note:
                lines.append(note)
            lines.append("")

    lines.extend(
        render_watchlist_optimization(
            context.plan,
            portfolio_value_usd=context.portfolio.portfolio_value_usd,
            settled_cash_usd=context.cash_summary.settled_cash_usd,
            show_recommendations=context.show_recommendations,
            portfolio_data_loaded=context.portfolio_data_loaded,
            watchlist_candidates_blocked_by_cash=(
                context.watchlist_candidates_blocked_by_cash
            ),
            live_orders=list(context.live_orders),
        )
    )

    if groups.holds_real:
        writer.section("HOLDS", "no action")
        for item in groups.holds_real:
            pos = item.ibkr_position
            analysis = item.analysis
            symbol = currency_prefix(item_currency(item))
            weight = ""
            if pos and context.portfolio.portfolio_value_usd > 0:
                weight = f"{pos.market_value_usd / context.portfolio.portfolio_value_usd * 100:.1f}%"
            price = ""
            if pos and pos.current_price_local:
                entry = (
                    analysis.entry_price if analysis and analysis.entry_price else None
                ) or (pos.avg_cost_local if pos.avg_cost_local else None)
                label = (
                    "analysis entry"
                    if analysis and analysis.entry_price
                    else "IBKR cost basis"
                )
                if entry:
                    gain = (pos.current_price_local - entry) / entry * 100
                    price = (
                        f"{label} {symbol}{entry:,.2f}  now "
                        f"{symbol}{pos.current_price_local:,.2f}  ({gain:+.1f}%)"
                    )
            stop = (
                f"stop {symbol}{analysis.stop_price:,.2f}"
                if analysis and analysis.stop_price
                else ""
            )
            target = (
                f"target {symbol}{analysis.target_1_price:,.2f}"
                if analysis and analysis.target_1_price
                else ""
            )
            note = ""
            if item.action_basis == "DE_MINIMIS":
                note = "de-minimis — monitor only"
            elif analysis and pos:
                note = stop_staleness_note(analysis, pos.current_price_local) or ""
            row = "  ".join(
                part for part in (weight, price, stop, target, note) if part
            )
            lines.append(f"  {'HOLD':<6}  {display_ticker(item):<12}  {row}")
        lines.append("")

    if groups.holds_watch:
        writer.section("WATCHLIST — MONITORING", "on watchlist, not yet a buy")
        for item in groups.holds_watch:
            analysis = item.analysis
            verdict = (
                f"Last analysis ({analysis.analysis_date}): {analysis.verdict} — not initiated"
                if analysis
                else "no analysis"
            )
            lines.append(f"  {'WATCH':<6}  {display_ticker(item):<12}  {verdict}")
        lines.append("")

    if groups.reviews:
        writer.section("REVIEWS", "analysis not decision-safe — refresh before acting")
        for item in groups.reviews:
            reason = item.reason.replace("Stale analysis: ", "").replace(
                "Position held but no evaluator analysis found", "no analysis found"
            )
            ticker = run_ticker_for(item)
            suffix_warning = (
                "  ← ⚠ exchange unknown, verify suffix" if "." not in ticker else ""
            )
            lines.append(
                f"  {'REVIEW':<6}  {display_ticker(item):<12}  {reason}  →  "
                f"{analysis_command(ticker)}{suffix_warning}"
            )
        lines.append("")

    if not context.items:
        lines.extend(("  No reconciliation items.", ""))

    sector_weights = aggregate_sector_weights(context.portfolio.sector_weights)
    exchange_weights = context.portfolio.exchange_weights
    currency_weights = context.portfolio.currency_weights
    if sector_weights or exchange_weights or currency_weights:
        writer.section("CONCENTRATION")
        if sector_weights:
            lines.append("  Sector:")
            for sector, pct in sorted(sector_weights.items(), key=lambda row: -row[1]):
                lines.append(
                    f"    {sector:<22} {pct:>5.1f}%  "
                    f"{bar_chart(pct, context.sector_limit_pct)}"
                )
            lines.append("")
        if exchange_weights:
            lines.append("  Exchange:")
            for exchange, pct in sorted(
                exchange_weights.items(), key=lambda row: -row[1]
            ):
                name = _EXCHANGE_LONG_NAMES.get(exchange, exchange)
                lines.append(
                    f"    {exchange:<5} ({name:<13}) {pct:>5.1f}%  "
                    f"{bar_chart(pct, context.exchange_limit_pct)}"
                )
            lines.append("")
        if currency_weights:
            lines.append("  Currency:")
            for currency, pct in sorted(
                currency_weights.items(), key=lambda row: -row[1]
            ):
                lines.append(f"    {currency:<22} {pct:>5.1f}%  {bar_chart(pct, 50.0)}")
            lines.append("")

    if context.portfolio_health_flags:
        writer.section("PORTFOLIO HEALTH", "cross-portfolio signals")
        for flag in context.portfolio_health_flags:
            first, *continuation = flag.split("\n")
            lines.append(f"  !! {first}")
            lines.extend(f"  {line}" for line in continuation)
        lines.append("")
    return tuple(lines)
