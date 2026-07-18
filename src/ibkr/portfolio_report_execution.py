"""Cash, execution-plan, and summary sections for the portfolio text report."""

from __future__ import annotations

from collections.abc import Callable

from src.ibkr.dip_watch import DIP_CONCENTRATION_MIN_SCORE, score_dip_watch_item
from src.ibkr.portfolio_presentation import (
    build_action_summary_counts,
    is_executable_buy,
)
from src.ibkr.portfolio_report import (
    PortfolioReportContext,
    select_report_dip_candidates,
)
from src.ibkr.portfolio_report_formatting import (
    ReportBuffer,
    currency_prefix,
    display_ticker,
    item_currency,
)
from src.ibkr.refresh_service import run_ticker_for
from src.ibkr.watchlist_optimization import build_watchlist_optimization_summary


def _render_cash_summary(context: PortfolioReportContext) -> tuple[str, ...]:
    if not context.show_recommendations:
        return ()
    lines: list[str] = []
    writer = ReportBuffer(
        lines=lines,
        show_recommendations=context.show_recommendations,
        settled_cash_usd=context.cash_summary.settled_cash_usd,
        live_orders=list(context.live_orders),
    )
    writer.section("CASH SUMMARY")
    portfolio = context.portfolio
    cash = context.cash_summary
    buy_items = [
        item
        for item in context.items
        if is_executable_buy(item, context.plan.executable_buy_ids)
        and item.cash_impact_usd < 0
    ]
    lines.append(
        f"  Settled cash:                                ${portfolio.settled_cash_usd:>7,.0f}"
    )
    if cash.buffer_reserve_usd > 0:
        lines.append(
            "  Cash buffer (held back, not for new buys):  "
            f"-${cash.buffer_reserve_usd:>7,.0f}"
        )
    lines.append(
        f"  Deployable into new buys:                    ${portfolio.available_cash_usd:>7,.0f}"
    )
    total_cost = 0.0
    for item in buy_items:
        cost = abs(item.cash_impact_usd)
        total_cost += cost
        quantity = (
            f"({abs(item.suggested_quantity)} sh)" if item.suggested_quantity else ""
        )
        label = f"  {item.action}  {display_ticker(item)}  {quantity}:"
        lines.append(f"{label:<46}- ${cost:>6,.0f}")
    if buy_items:
        remaining = portfolio.settled_cash_usd - total_cost
        lines.extend(
            (
                "  " + "─" * 48,
                f"  Settled cash after recommended buys:         ${remaining:>7,.0f}",
                "",
            )
        )
    if cash.pending_inflows:
        settlement = cash.next_settlement_date or "in 2 business days"
        lines.append(f"  Pending inflows (sale proceeds, clears {settlement}):")
        for row in cash.pending_inflows:
            quantity = f"({abs(row.quantity)} sh)" if row.quantity else ""
            label = f"    {row.action}  {row.ticker_ibkr}  {quantity}:"
            lines.append(f"{label:<46}+ ${row.cash_impact_usd:>6,.0f}")
        lines.append(
            f"{'  Total pending:':<46}  ${cash.pending_inflows_total_usd:>6,.0f}"
        )
        if cash.conditional_proceeds_usd > 0:
            lines.append(
                f"{'  Conditional (soft-sell reviews):':<46}"
                f"  ${cash.conditional_proceeds_usd:>6,.0f}"
            )
        lines.extend(
            (
                "",
                "  ⚠  Do NOT spend sale proceeds today — they have not settled yet.",
                f"     If orders fill by market close, funds clear {settlement}.",
                "     Place additional BUYs only after that settlement date.",
                "",
            )
        )
    elif cash.conditional_proceeds_usd > 0:
        lines.extend(
            (
                "  No confirmed sale proceeds pending.",
                "  Conditional (soft-sell reviews if executed):"
                f"  ~${cash.conditional_proceeds_usd:>6,.0f}",
                "",
            )
        )
    return tuple(lines)


def _settlement_groups(
    context: PortfolioReportContext,
) -> tuple[dict[str, float], dict[str, float]]:
    confirmed: dict[str, float] = {}
    conditional: dict[str, float] = {}
    for item in context.items:
        if (
            item.action not in ("SELL", "TRIM")
            or not item.settlement_date
            or item.cash_impact_usd <= 0
        ):
            continue
        target = conditional if item.sell_type == "SOFT_REJECT" else confirmed
        target[item.settlement_date] = (
            target.get(item.settlement_date, 0.0) + item.cash_impact_usd
        )
    return confirmed, conditional


def _append_today_actions(
    lines: list[str], context: PortfolioReportContext, writer: ReportBuffer
) -> None:
    action_today = [item for item in context.items if item.action in ("SELL", "TRIM")]
    funded_today = [
        item
        for item in context.items
        if is_executable_buy(item, context.plan.executable_buy_ids)
        and item.cash_impact_usd < 0
    ]
    if not (action_today or funded_today):
        return
    lines.append(f"  TODAY ({context.today_iso}):")
    for item in action_today:
        quantity = (
            f"  {abs(item.suggested_quantity)} shares"
            if item.suggested_quantity
            else ""
        )
        price = (
            f"  @ {currency_prefix(item_currency(item))}{item.suggested_price:,.2f}"
            if item.suggested_price
            else ""
        )
        existing = writer.find_live_order(item)
        recommended_side = "SELL" if item.action in ("SELL", "TRIM") else "BUY"
        if existing and existing[1] == recommended_side:
            raw_quantity = existing[0].get("remainingSize") or existing[0].get(
                "totalSize"
            )
            if raw_quantity is None:
                order_quantity: int | None = None
            else:
                try:
                    order_quantity = int(raw_quantity)
                except (TypeError, ValueError):
                    order_quantity = None
            if (
                order_quantity is not None
                and item.suggested_quantity is not None
                and order_quantity < item.suggested_quantity
            ):
                needed = item.suggested_quantity - order_quantity
                lines.append(
                    f"    → {item.action}  {display_ticker(item)}  {needed} more shares"
                    f"{price}  {item.suggested_order_type or 'LMT'}  "
                    f"({order_quantity} of {item.suggested_quantity} already submitted)"
                )
            else:
                lines.append(
                    f"    ✓ {item.action}  {display_ticker(item)}{quantity}{price}  "
                    f"{item.suggested_order_type or 'LMT'}  "
                    "(order already submitted — verify in IBKR)"
                )
        else:
            lines.append(
                f"    → {item.action}  {display_ticker(item)}{quantity}{price}  "
                f"{item.suggested_order_type or 'LMT'}"
            )

    buys_in_flight: list[str] = []
    for item in funded_today:
        existing = writer.find_live_order(item)
        if existing and existing[1] == "BUY":
            buys_in_flight.append(display_ticker(item))
            continue
        quantity = (
            f"  {abs(item.suggested_quantity)} shares"
            if item.suggested_quantity
            else ""
        )
        price = (
            f"  @ {currency_prefix(item_currency(item))}{item.suggested_price:,.2f}"
            if item.suggested_price
            else ""
        )
        cost = f"  (~${abs(item.cash_impact_usd):,.0f})" if item.cash_impact_usd else ""
        quantity_note = (
            ""
            if item.suggested_quantity
            else (
                "  [portfolio/cash not loaded — inspect before placing order]"
                if not context.portfolio_data_loaded
                else "  [quantity unavailable — inspect before placing order]"
            )
        )
        lines.append(
            f"    → {item.action}  {display_ticker(item)}{quantity}{price}{cost}"
            f"{quantity_note}  {writer.buy_pos_tag(item)}  — use already-settled cash"
        )
    if buys_in_flight:
        lines.append(
            f"    ✓ {len(buys_in_flight)} already in flight"
            f" ({', '.join(buys_in_flight)}) — verify in IBKR"
        )
    lines.append("")


def _append_dip_actions(
    lines: list[str],
    context: PortfolioReportContext,
    writer: ReportBuffer,
    *,
    analysis_command: Callable[[str], str],
) -> None:
    candidates = select_report_dip_candidates(context.plan)
    if not candidates:
        return
    lines.append(f"  DIP OPPORTUNITIES ({context.today_iso}):")
    in_flight: list[str] = []
    for item in candidates:
        existing = writer.find_live_order(item)
        if existing and existing[1] == "BUY":
            in_flight.append(display_ticker(item))
            continue
        analysis = item.analysis
        position = item.ibkr_position
        score = score_dip_watch_item(item)
        stars = (
            "★★★"
            if score >= DIP_CONCENTRATION_MIN_SCORE
            else "★★ "
            if score >= 60
            else "★  "
        )
        quantity = (
            f"{position.quantity:,.0f} sh held"
            if position and position.quantity
            else "held"
        )
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
        lines.append(
            f"    → DIP ADD  {display_ticker(item)}  {stars}  score {score:.0f}"
            f"  H:{health}% G:{growth}%  [{quantity}]"
            f"  →  {analysis_command(run_ticker_for(item))}"
        )
    if in_flight:
        lines.append(
            f"    ✓ {len(in_flight)} already in flight"
            f" ({', '.join(in_flight)}) — verify in IBKR"
        )
    lines.append("")


def _render_action_plan(
    context: PortfolioReportContext,
    *,
    analysis_command: Callable[[str], str],
    recommend_command: Callable[..., str],
) -> tuple[str, ...]:
    lines: list[str] = []
    writer = ReportBuffer(
        lines=lines,
        show_recommendations=context.show_recommendations,
        settled_cash_usd=context.cash_summary.settled_cash_usd,
        live_orders=list(context.live_orders),
    )
    action_today = [item for item in context.items if item.action in ("SELL", "TRIM")]
    funded_today = [
        item
        for item in context.items
        if is_executable_buy(item, context.plan.executable_buy_ids)
        and item.cash_impact_usd < 0
    ]
    dip_candidates = select_report_dip_candidates(context.plan)
    confirmed, conditional = _settlement_groups(context)
    activity = context.refresh_activity
    if not (
        action_today
        or funded_today
        or dip_candidates
        or confirmed
        or conditional
        or activity.refreshed
        or activity.failed
        or activity.skipped_due_to_limit
    ):
        return ()
    writer.section(
        "ACTION PLAN", "execution orders · settlement · refresh follow-through"
    )
    _append_today_actions(lines, context, writer)
    _append_dip_actions(lines, context, writer, analysis_command=analysis_command)
    for settlement in sorted(set(confirmed) | set(conditional)):
        confirmed_amount = confirmed.get(settlement, 0.0)
        conditional_amount = conditional.get(settlement, 0.0)
        lines.append(f"  {settlement} — sale proceeds from today's sells/trims clear:")
        if confirmed_amount > 0:
            lines.append(f"    → ${confirmed_amount:,.0f} available on this date")
        if conditional_amount > 0:
            lines.append(
                f"    → ~${conditional_amount:,.0f} additional if soft-sell reviews are executed"
            )
        if confirmed_amount == 0 and conditional_amount > 0:
            lines.append(
                "    → No confirmed proceeds — review soft sells before counting on this cash"
            )
        if dip_candidates:
            tickers = "  ".join(display_ticker(item) for item in dip_candidates[:3])
            lines.append(
                f"    → Top dip candidates for deployment: {tickers}"
                "  (see DIP OPPORTUNITIES above)"
            )
        lines.extend(
            (
                "    → Run before placing any additional buys:",
                f"        {recommend_command(watchlist_name=context.watchlist_name)}",
                "",
            )
        )
    if activity.refreshed or activity.failed or activity.skipped_due_to_limit:
        lines.append("  ANALYSIS REFRESH:")
        if activity.refreshed:
            lines.append(f"    ✓ Refreshed this run: {', '.join(activity.refreshed)}")
        if activity.failed:
            lines.append(f"    → Retry failed refreshes: {', '.join(activity.failed)}")
        if activity.skipped_due_to_limit:
            lines.append(
                "    → Remaining after limit: "
                + ", ".join(activity.skipped_due_to_limit)
            )
        lines.append("")
    return tuple(lines)


def _render_summary(context: PortfolioReportContext) -> tuple[str, ...]:
    lines: list[str] = []
    counts = build_action_summary_counts(context.plan.groups)
    counts.pop("BUY", None)
    counts.pop("CANDIDATES", None)
    counts.pop("REMOVE", None)
    counts.update(build_watchlist_optimization_summary(context.plan.optimization))
    order = (
        "SELL",
        "TRIM",
        "ADD",
        "HOLD",
        "REVIEW",
        "MACRO_WATCH",
        "WATCHLIST_KEEP",
        "WATCHLIST_ADD",
        "WATCHLIST_REMOVE",
        "WATCHLIST_MONITOR",
        "WATCHLIST_REVIEW",
    )
    summary = [f"{counts[action]} {action}" for action in order if action in counts]
    sell_notional = sum(
        item.cash_impact_usd
        for item in context.items
        if item.action in ("SELL", "TRIM") and item.cash_impact_usd > 0
    )
    buy_notional = sum(
        abs(item.cash_impact_usd)
        for item in context.items
        if is_executable_buy(item, context.plan.executable_buy_ids)
        and item.cash_impact_usd
    )
    if sell_notional or buy_notional:
        nav = context.portfolio.portfolio_value_usd
        sell_pct = f" ({sell_notional / nav * 100:.1f}% of NAV)" if nav > 0 else ""
        buy_pct = f" ({buy_notional / nav * 100:.1f}% of NAV)" if nav > 0 else ""
        lines.append(
            f"  Plan turnover:  executable sells ~${sell_notional:,.0f}{sell_pct}"
            f"  ·  buys ~${buy_notional:,.0f}{buy_pct}"
        )
    lines.append(f"  Summary:  {'  ·  '.join(summary) or 'empty'}")
    return tuple(lines)


def render_cash_execution_and_summary_sections(
    context: PortfolioReportContext,
    *,
    analysis_command: Callable[[str], str],
    recommend_command: Callable[..., str],
) -> tuple[str, ...]:
    """Render cash availability, sequenced actions, and aggregate counts."""
    return (
        *_render_cash_summary(context),
        *_render_action_plan(
            context,
            analysis_command=analysis_command,
            recommend_command=recommend_command,
        ),
        *_render_summary(context),
    )
