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
    DETAIL_INDENT,
    DIVIDER,
    ReportBuffer,
    as_of_date,
    bar_chart,
    currency_prefix,
    display_ticker,
    item_currency,
    normalize_reason,
    split_reason,
)
from src.ibkr.reconciliation_rules import _EXCHANGE_LONG_NAMES
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
    ]
    if not candidates:
        lines.extend(("  No dip-buy candidates this run.", ""))
        return tuple(lines)
    lines.extend(
        (
            "  Ranked by fundamental quality × dip depth × valuation upside:",
            "",
        )
    )
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
        upside_text = "—"
        if (
            analysis
            and analysis.target_1_price
            and pos
            and pos.current_price_local
            and pos.current_price_local > 0
        ):
            current = pos.current_price_local
            upside = (analysis.target_1_price - current) / current * 100
            upside_text = f"upside +{upside:.0f}% to base-case reference"
        lines.append(
            f"  {stars}  {display_ticker(item):<12}  Health:{health}%  "
            f"Growth:{growth}%  |  {upside_text}"
        )
        lines.append(f"{DETAIL_INDENT}{entry_text}")
        source = dip_watch_source(item)
        if source == "macro_review":
            as_of = analysis.analysis_date if analysis else "?"
            lines.extend(
                (
                    f"{DETAIL_INDENT}macro dip — fundamentals intact, review "
                    f"{as_of}; standalone verdict was REJECT",
                    f"{DETAIL_INDENT}(often valuation, which the dip improves) "
                    "— review before adding",
                )
            )
        elif source == "held_thesis_dip":
            as_of = analysis.analysis_date if analysis else "?"
            lines.append(
                f"{DETAIL_INDENT}intact-thesis drawdown — scores hold the "
                f"gates ({as_of}); verdict reflects price/entry screen, "
                "not deterioration — refresh before adding"
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
                f"{writer.order_line(item, currency)}  [{writer.sell_type_label(item)}]"
            )
            writer.append_reason_detail(reason)
            if item.sell_type in ("STOP_BREACH", "HARD_REJECT"):
                score_line = writer.score_line(item)
                if score_line:
                    lines.append(score_line)
                writer.append_pnl_proceeds(item, currency)
            elif item.sell_type == "SOFT_REJECT":
                writer.append_soft_rejection_details(item)
            elif item.sell_type == "PROFIT_TAKE":
                writer.append_wrapped_segments(writer.profit_take_segments(item))
            writer.append_thesis_break_line(item)
            writer.append_sale_tax_note(item)
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
                item.reason.split("  [MACRO_PRICE:")[0]
                .split("  [MACRO_STOP:")[0]
                .split("  [MACRO_WATCH:")[0]
            )
            lines.append(
                f"{writer.order_line(item, currency)}  [{writer.sell_type_label(item)}]"
            )
            writer.append_reason_detail(reason)
            if item.sell_type == "PROFIT_TAKE":
                writer.append_wrapped_segments(writer.profit_take_segments(item))
            elif item.sell_type == "SOFT_REJECT":
                writer.append_soft_rejection_details(item)
            else:
                score_line = writer.score_line(item)
                if score_line:
                    lines.append(score_line)
                writer.append_pnl_proceeds(item, currency)
            if item.action == "REVIEW":
                holding = writer.holding_line(item, currency)
                if holding:
                    lines.append(holding)
            writer.append_thesis_break_line(item)
            note = writer.order_note(item)
            if note:
                lines.append(note)
            lines.append("")

    dip_candidates = list(select_report_dip_candidates(context.plan))
    lines.extend(_render_dip_watch(dip_candidates, analysis_command=analysis_command))
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
            # Routine HOLD rows emphasize ownership context and local return;
            # legacy downside/target levels stay in drilldown, not the action table.
            currency = item_currency(item) or "?"
            weight = ""
            if pos and context.portfolio.portfolio_value_usd > 0:
                weight = f"{pos.market_value_usd / context.portfolio.portfolio_value_usd * 100:.1f}%"
            label = ""
            entry_text = now_text = gain_text = "—"
            if pos and pos.current_price_local:
                entry = (
                    analysis.entry_price if analysis and analysis.entry_price else None
                ) or (pos.avg_cost_local if pos.avg_cost_local else None)
                label = "entry" if analysis and analysis.entry_price else "cost"
                if entry:
                    gain = (pos.current_price_local - entry) / entry * 100
                    entry_text = f"{entry:,.2f}"
                    now_text = f"{pos.current_price_local:,.2f}"
                    gain_text = f"({gain:+.1f}%)"
            note = ""
            if item.action_basis == "DE_MINIMIS":
                note = "de-minimis — monitor only"
            scores = ""
            if (
                analysis
                and analysis.health_adj is not None
                and analysis.growth_adj is not None
            ):
                scores = f"  H:{analysis.health_adj:.0f} G:{analysis.growth_adj:.0f}"
            lines.append(
                f"  {'HOLD':<6}  {display_ticker(item):<12} {weight:>5}  "
                f"{currency:<4} {label:<5} {entry_text:>10} → {now_text:>10}"
                f"  {gain_text:>8}{scores}"
            )
            if note:
                lines.append(f"{DETAIL_INDENT}{note}")
            writer.append_fx_split_line(item)
            writer.append_thesis_break_line(item)
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
            head, reason_detail = split_reason(reason)
            ticker = run_ticker_for(item)
            suffix_warning = (
                "  ← ⚠ exchange unknown, verify suffix" if "." not in ticker else ""
            )
            lines.append(
                f"  {'REVIEW':<6}  {display_ticker(item):<12}  {head}  "
                f"[{writer.sell_type_label(item)}]"
            )
            if reason_detail:
                lines.append(f"{DETAIL_INDENT}{reason_detail}")
            holding = writer.holding_line(item, item_currency(item))
            if holding:
                lines.append(holding)
            writer.append_thesis_break_line(item)
            note = writer.order_note(item)
            if note:
                lines.append(note)
            lines.append(
                f"{DETAIL_INDENT}→  {analysis_command(ticker)}{suffix_warning}"
            )
        lines.append("")

    if not context.items:
        lines.extend(("  No reconciliation items.", ""))

    sector_weights = aggregate_sector_weights(context.portfolio.sector_weights)
    exchange_weights = context.portfolio.exchange_weights
    if sector_weights or exchange_weights:
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

    if context.portfolio_health_flags:
        writer.section("PORTFOLIO HEALTH", "cross-portfolio signals")
        for flag in context.portfolio_health_flags:
            first, *continuation = flag.split("\n")
            lines.extend(
                ReportBuffer.wrap_banner_value("  !! ", first, width=110, max_lines=4)
            )
            lines.extend(f"  {line}" for line in continuation)
        lines.append("")
    return tuple(lines)
