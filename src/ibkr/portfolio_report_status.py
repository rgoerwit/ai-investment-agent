"""Account, alert, and freshness sections for the portfolio text report."""

from __future__ import annotations

import re
from collections.abc import Callable

from src.ibkr.portfolio_report import PortfolioReportContext
from src.ibkr.portfolio_report_formatting import ReportBuffer


def _append_account_header(lines: list[str], context: PortfolioReportContext) -> None:
    portfolio = context.portfolio
    cash = context.cash_summary
    lines.extend(
        (
            f"=== IBKR Portfolio Reconciliation  {context.generated_at[:16].replace('T', ' ')} ===",
            "",
        )
    )
    if not context.portfolio_data_loaded:
        lines.extend(
            (
                "⚠ READ-ONLY — no IBKR connection; holdings, watchlist, and cash were NOT loaded.",
                "  BUYs below come from saved analyses; whether you already own/watch them is UNKNOWN.",
                "",
            )
        )
    if context.errors.get("watchlist"):
        lines.extend(
            (
                "⚠ WATCHLIST UNAVAILABLE — could not read your IBKR watchlist "
                "(brokerage session unavailable).",
                "  Holdings/SELL/HOLD analysis below is unaffected. Watchlist "
                "filtering is unavailable;",
                "  unheld BUY analyses are shown as BUY CANDIDATES and should be "
                "verified in IBKR before acting.",
                "",
            )
        )
    if context.show_recommendations and context.errors.get("live_orders"):
        lines.extend(
            (
                "⚠ LIVE ORDERS UNAVAILABLE — open-order dedup is disabled; an "
                "order you already have working",
                "  may be re-suggested. Verify in IBKR before placing orders.",
                "",
            )
        )
    if not context.portfolio_data_loaded:
        lines.extend(
            (
                "  Account:          not loaded (read-only)",
                "  Net liquidation:  not loaded",
                "  Cash (total):     not loaded",
                "  Settled cash:     not loaded",
                "  Available:        not loaded",
                "",
            )
        )
        return

    nlv = portfolio.portfolio_value_usd
    lines.extend(
        (
            f"  Account:          {portfolio.account_id or 'N/A'}",
            f"  Net liquidation:  ${nlv:>10,.0f}",
        )
    )
    if nlv > 0:
        if cash.unsettled_cash_usd > 0:
            cash_note = (
                f"  includes ${cash.unsettled_cash_usd:,.0f} of unsettled sale proceeds "
                "(not yet spendable)"
            )
        else:
            cash_note = "  all shown cash is settled"
        lines.extend(
            (
                f"  Cash (total):     ${portfolio.cash_balance_usd:>10,.0f}   "
                f"({portfolio.cash_balance_usd / nlv * 100:.1f}%){cash_note}",
                f"  Settled cash:     ${portfolio.settled_cash_usd:>10,.0f}   "
                f"({portfolio.settled_cash_usd / nlv * 100:.1f}%)  fully settled",
                f"  Buffer reserve:   ${cash.buffer_reserve_usd:>10,.0f}   "
                f"({cash.buffer_reserve_usd / nlv * 100:.1f}%)"
                "  cash buffer — not deployed into new buys",
                f"  Available:        ${portfolio.available_cash_usd:>10,.0f}"
                "           deployable into new buys (settled − buffer)",
            )
        )
    else:
        lines.extend(
            (
                f"  Cash (total):     ${portfolio.cash_balance_usd:,.0f}",
                f"  Settled cash:     ${portfolio.settled_cash_usd:,.0f}",
                f"  Available:        ${portfolio.available_cash_usd:,.0f}",
            )
        )
    lines.append("")


def _append_macro_banner(
    lines: list[str], context: PortfolioReportContext, writer: ReportBuffer
) -> None:
    flags = context.portfolio_health_flags
    correlated = next((flag for flag in flags if "CORRELATED_SELL_EVENT" in flag), None)
    active = next((flag for flag in flags if "ACTIVE_MACRO_EVENT" in flag), None)
    width = 52
    if correlated:
        match = re.search(
            r"(\d+) positions.*?of (\d{4}-\d{2}-\d{2}) \((\d+)%", correlated
        )
        count, event_date, percent = (
            (match.group(1), match.group(2), f"{match.group(3)}%")
            if match
            else ("?", "?", "?%")
        )
        failure_count = sum(
            1
            for item in context.items
            if item.action == "SELL" and item.action_basis == "CONFIRMED_THESIS_FAILURE"
        )
        guidance = ["Pause thesis changes; re-evaluate intrinsic value."]
        if failure_count:
            guidance.append(f"{failure_count} confirmed thesis failure(s) remain live.")
        if not failure_count:
            guidance.append("No executable SELLs — all demoted to review.")
        banner = [
            "╔" + "═" * 54 + "╗",
            f"║  {'!! MACRO ALERT':<{width}}║",
            f"║  {f'{count} positions impacted (as of {event_date})':<{width}}║",
            f"║  {f'({percent} of held positions) — probable macro event':<{width}}║",
            f"║  {'Likely macro event, not individual thesis failure.':<{width}}║",
            *(f"║  {line:<{width}}║" for line in guidance),
            "╚" + "═" * 54 + "╝",
        ]
        try:
            event = context.current_macro_event
            if event is None:
                from src.memory import create_macro_events_store

                store = create_macro_events_store()
                if store.available:
                    active_events = store.get_active_events()
                    event = active_events[0] if active_events else None
            if event and event.news_headline != "unknown":
                for field in (
                    f"Macro driver: {event.event_type}",
                    f"Impact: {event.impact}",
                ):
                    banner.insert(-1, f"║  {field:<{width}}║")
                for wrapped in writer.wrap_banner_value(
                    "Headline: ", event.news_headline, width=width, max_lines=2
                ):
                    banner.insert(-1, f"║  {wrapped:<{width}}║")
        except Exception:
            pass
        lines.extend((*banner, ""))
    elif active:
        match = re.search(
            r"ACTIVE_MACRO_EVENT:\s*(\S+) event active until (\S+)\s*—\s*(\d+)",
            active,
        )
        event_type, until, count = match.groups() if match else ("MACRO", "?", "?")
        lines.extend(
            (
                "╔" + "═" * 54 + "╗",
                f"║  {'!! MACRO OVERRIDE ACTIVE':<{width}}║",
                f"║  {f'{event_type} event active until {until}':<{width}}║",
                f"║  {f'{count} SELL(s) held in REVIEW (sustained override)':<{width}}║",
                "╚" + "═" * 54 + "╝",
                "",
            )
        )
    elif any("MODEL_REGIME_SHIFT" in flag for flag in flags):
        lines.extend(
            (
                "╔" + "═" * 54 + "╗",
                f"║  {'!! MODEL RE-RATING DETECTED':<{width}}║",
                f"║  {'Broad verdict flips with prices at/above entry':<{width}}║",
                f"║  {'— analyzer-side re-rating, not market distress.':<{width}}║",
                f"║  {'Audit recent prompt/model/threshold changes':<{width}}║",
                f"║  {'before acting on the flipped verdicts.':<{width}}║",
                "╚" + "═" * 54 + "╝",
                "",
            )
        )


def _append_screening_freshness(
    lines: list[str], context: PortfolioReportContext, writer: ReportBuffer
) -> None:
    freshness = context.screening_freshness
    if freshness.status == "fresh":
        return
    writer.section("SCREENING FRESHNESS", "last completed broad market sweep")
    if freshness.status == "missing":
        lines.extend(
            (
                "  No broad-screen completion recorded.",
                "  → Run: ./scripts/run_pipeline.sh",
            )
        )
    else:
        lines.append(
            "  Last completed sweep: "
            f"{freshness.screening_date or 'unknown'}  ({freshness.age_days} days ago)"
        )
        if freshness.candidate_count is not None or freshness.buy_count is not None:
            candidate_count = (
                freshness.candidate_count
                if freshness.candidate_count is not None
                else "—"
            )
            buy_count = freshness.buy_count if freshness.buy_count is not None else "—"
            lines.append(
                f"  Candidates screened: {candidate_count}  ·  BUYs found: {buy_count}"
            )
        lines.append("  → Consider re-running: ./scripts/run_pipeline.sh")
    lines.append("")


def _append_analysis_freshness(
    lines: list[str],
    context: PortfolioReportContext,
    writer: ReportBuffer,
    *,
    analysis_command: Callable[[str], str],
    user_action: str,
) -> None:
    summary = context.freshness_summary
    activity = context.refresh_activity
    if not (
        summary.blocking_now
        or summary.stale_in_queue
        or summary.due_soon
        or summary.candidate_blocked
        or activity.refreshed
        or activity.failed
        or activity.skipped_due_to_policy
        or activity.skipped_due_to_limit
        or activity.skipped_read_only
    ):
        return
    writer.section(
        "ANALYSIS FRESHNESS", "what is stale, what is queued, what happens next"
    )
    lines.append("  Needs review before action:")
    if summary.blocking_now:
        for row in summary.blocking_now:
            details = [row.reason_family]
            if row.age_days is not None:
                details.append(f"{row.age_days}d old")
            if row.expires_date:
                details.append(f"expires {row.expires_date}")
            lines.append(
                f"    {row.display_ticker:<12} {'  ·  '.join(details)}"
                f"  →  {analysis_command(row.run_ticker)}"
            )
    else:
        lines.append("    None")
    lines.append("")

    if summary.candidate_blocked:
        lines.append("  Candidates needing full refresh:")
        for row in summary.candidate_blocked:
            details = [row.reason_family]
            if row.age_days is not None:
                details.append(f"{row.age_days}d old")
            lines.append(
                f"    {row.display_ticker:<12} {'  ·  '.join(details)}"
                f"  →  {analysis_command(row.run_ticker)}"
            )
        lines.append("")

    lines.append("  Already in refresh queue:")
    if summary.stale_in_queue:
        for row in summary.stale_in_queue:
            details = [f"already in {row.action} queue"]
            if row.age_days is not None:
                details.append(f"{row.age_days}d old")
            lines.append(f"    {row.display_ticker:<12} {'  ·  '.join(details)}")
    else:
        lines.append("    None")
    lines.append("")

    lines.append("  Due soon:")
    if summary.due_soon:
        for row in sorted(
            summary.due_soon,
            key=lambda current: (
                current.days_until_due if current.days_until_due is not None else 9999
            ),
        ):
            due_details: list[str] = []
            if row.expires_date:
                due_details.append(f"expires {row.expires_date}")
            if row.days_until_due is not None:
                due_details.append(f"{row.days_until_due}d remaining")
            lines.append(
                f"    {row.display_ticker:<12} {'  ·  '.join(due_details)}"
                f"  →  {analysis_command(row.run_ticker)}"
            )
    else:
        lines.append("    None")
    lines.extend(("", "  Refresh activity this run:"))
    lines.append(
        f"    Policy: {activity.policy}"
        + (f"  ·  limit {activity.limit}" if activity.limit else "")
    )
    if activity.refreshed:
        lines.append(f"    Refreshed: {', '.join(activity.refreshed)}")
    if activity.failed:
        lines.append(f"    Failed: {', '.join(activity.failed)}")
    if activity.skipped_read_only:
        lines.append(
            f"    Skipped (read-only): {', '.join(activity.skipped_read_only)}"
        )
    if activity.skipped_due_to_policy:
        lines.append(
            f"    Skipped (policy): {', '.join(activity.skipped_due_to_policy)}"
        )
    if activity.skipped_due_to_limit:
        lines.append(
            "    Deferred by refresh limit: will be retried on the next "
            "refresh-enabled run: " + ", ".join(activity.skipped_due_to_limit)
        )
    if not (
        activity.refreshed
        or activity.failed
        or activity.skipped_read_only
        or activity.skipped_due_to_policy
        or activity.skipped_due_to_limit
    ):
        lines.append("    No refresh actions were needed.")
    lines.extend(("", f"  User action: {user_action}", ""))


def render_header_and_status_sections(
    context: PortfolioReportContext,
    *,
    analysis_command: Callable[[str], str],
    freshness_user_action: str,
) -> tuple[str, ...]:
    """Render account state, macro alerts, and data-freshness guidance."""
    lines: list[str] = []
    writer = ReportBuffer(
        lines=lines,
        show_recommendations=context.show_recommendations,
        settled_cash_usd=context.cash_summary.settled_cash_usd,
        live_orders=list(context.live_orders),
    )
    _append_account_header(lines, context)
    _append_macro_banner(lines, context, writer)
    _append_screening_freshness(lines, context, writer)
    _append_analysis_freshness(
        lines,
        context,
        writer,
        analysis_command=analysis_command,
        user_action=freshness_user_action,
    )
    return tuple(lines)
