"""Focused text renderers for the portfolio-manager CLI report."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from src.ibkr.dip_watch import dip_watch_source
from src.ibkr.models import PortfolioSummary, ReconciliationItem
from src.ibkr.order_presentation import build_live_order_note
from src.ibkr.portfolio_action_plan import PortfolioActionPlan
from src.ibkr.portfolio_presentation import CashSummaryView
from src.ibkr.portfolio_report_formatting import ReportBuffer, wrap_listing
from src.ibkr.refresh_service import AnalysisFreshnessSummary, RefreshActivity
from src.ibkr.screening_freshness import ScreeningFreshnessSummary
from src.ibkr.ticker import Ticker
from src.ibkr.watchlist_optimization import (
    CONCENTRATION_EXCEPTION_MIN_SCORE,
    CONCENTRATION_INCUMBENT_MIN_SCORE,
    ConcentrationNote,
    watchlist_candidate_conviction,
    watchlist_candidate_score,
)
from src.memory import MacroEvent

_DIVIDER = "═" * 54


def _breach_bullets(note: ConcentrationNote, indent: str) -> list[str]:
    return [
        f"{indent}· overweight {breach.dimension} {breach.key} "
        f"(projected {breach.projected_pct:.1f}% > {breach.limit_pct:.0f}%)"
        for breach in note.breaches
    ]


def _breach_category(note: ConcentrationNote) -> str:
    """Reason label without per-candidate projections, for grouping."""
    return " + ".join(
        f"{breach.dimension} {breach.key} > {breach.limit_pct:.0f}%"
        for breach in note.breaches
    )


@dataclass(frozen=True)
class PortfolioReportContext:
    """Prepared report inputs; renderers must not recompute recommendation policy."""

    items: tuple[ReconciliationItem, ...]
    portfolio: PortfolioSummary
    plan: PortfolioActionPlan
    cash_summary: CashSummaryView
    portfolio_health_flags: tuple[str, ...]
    max_age_days: int
    live_orders: tuple[dict, ...]
    errors: Mapping[str, str]
    watchlist_name: str | None
    watchlist_total: int | None
    watchlist_candidates_blocked_by_cash: int
    freshness_summary: AnalysisFreshnessSummary
    refresh_activity: RefreshActivity
    screening_freshness: ScreeningFreshnessSummary
    portfolio_data_loaded: bool
    current_macro_event: MacroEvent | None
    exchange_limit_pct: float
    sector_limit_pct: float
    show_recommendations: bool
    generated_at: str
    today_iso: str


def select_report_dip_candidates(
    plan: PortfolioActionPlan,
) -> tuple[ReconciliationItem, ...]:
    """Return dip candidates the report may display under the current regime."""
    return tuple(
        item
        for item in plan.groups.dip_candidates
        if dip_watch_source(item) == "held_buy_pullback"
        or (plan.macro_event_active and dip_watch_source(item) == "macro_review")
    )


def _item_currency(item: ReconciliationItem) -> str | None:
    if item.analysis and item.analysis.currency:
        return item.analysis.currency
    if item.ibkr_position and item.ibkr_position.currency:
        return item.ibkr_position.currency
    return None


def _currency_prefix(currency: str | None) -> str:
    return f"{currency.upper()} " if currency else "? "


def _watchlist_symbol(item: ReconciliationItem) -> str:
    yf_hint = f" ({item.ticker.yf})" if "." in item.ticker.yf else ""
    return f"{item.ticker.ibkr}{yf_hint}"


def render_watchlist_optimization(
    plan: PortfolioActionPlan,
    *,
    portfolio_value_usd: float,
    settled_cash_usd: float,
    show_recommendations: bool,
    portfolio_data_loaded: bool,
    watchlist_candidates_blocked_by_cash: int,
    live_orders: list[dict] | None,
) -> list[str]:
    """Render the action plan's watchlist section without recomputing policy."""
    optimization = plan.optimization
    lines: list[str] = []
    concentration_screened = len(optimization.withheld_candidates) + sum(
        1 for move in optimization.remove if move.reason == "concentration_displaced"
    )
    partial_fill_subtitle = (
        f"optimal BUY-ready set under-filled ({len(optimization.optimal)}"
        f" of {optimization.target_size})"
    )
    if concentration_screened:
        partial_fill_subtitle = (
            f"{partial_fill_subtitle[:-1]} — {concentration_screened} withheld by "
            "concentration)"
        )
    case_subtitles = {
        "no_watchlist": "no watchlist loaded — additions only",
        "watchlist_unavailable": (
            "watchlist status UNKNOWN — additions only; confirm watchlist status "
            "and re-check IBKR before acting"
        ),
        "nothing_actionable": "no medium-or-better candidates",
        "empty_pool": (
            "no medium-or-better candidates; existing entries retained for review"
        ),
        "partial_fill": partial_fill_subtitle,
        "aligned": "current watchlist already matches the BUY-ready target",
        "full_optimize": (
            f"top {optimization.target_size} medium-or-higher BUY-ready slots"
        ),
    }
    subtitle = case_subtitles[optimization.case.value]
    header = f"  WATCHLIST OPTIMIZATION  ({subtitle})"
    if len(header) <= 100:
        header_lines: list[str] = [header]
    else:
        header_lines = [
            "  WATCHLIST OPTIMIZATION",
            *ReportBuffer.wrap_banner_value(
                "  ", f"({subtitle})", width=100, max_lines=3
            ),
        ]
    lines.extend((_DIVIDER, *header_lines, _DIVIDER, ""))

    def render_candidate(item: ReconciliationItem, label: str) -> None:
        analysis = item.analysis
        conviction = watchlist_candidate_conviction(item) or "unspecified"
        score = watchlist_candidate_score(item)
        lines.append(
            f"  {label:<6}  {_watchlist_symbol(item):<20}  "
            f"{conviction} conviction  score {score:.0f}"
        )
        detail: list[str] = []
        if analysis:
            size_pct = analysis.trade_block.size_pct or analysis.position_size or 0
            if size_pct and portfolio_value_usd > 0:
                target = portfolio_value_usd * size_pct / 100
                detail.append(f"target {size_pct:.1f}% (${target:,.0f})")
        if not portfolio_data_loaded:
            detail.append("[own/watchlist status unknown]")
        if item.is_cash_blocked:
            detail.append("no cash — inspect before ordering; no order ticket")
        elif show_recommendations and item.suggested_quantity:
            detail.append(
                f"future BUY: {item.suggested_quantity} shares"
                + (
                    f" @ {_currency_prefix(_item_currency(item))}"
                    f"{item.suggested_price:,.2f}"
                    if item.suggested_price
                    else ""
                )
            )
        elif not portfolio_data_loaded:
            detail.append("portfolio/cash not loaded — inspect before ordering")
        elif show_recommendations and not item.suggested_quantity:
            detail.append("quantity unavailable — inspect before placing order")
        if not item.suggested_price:
            detail.append("no entry price — re-run analysis")
        if detail:
            lines.append(f"             {'  ·  '.join(detail)}")
        if (
            item.is_watchlist
            and not item.is_cash_blocked
            and show_recommendations
            and item.cash_impact_usd
        ):
            cost = abs(item.cash_impact_usd)
            funded = (
                f"use already-settled cash (${settled_cash_usd:,.0f} available)"
                if settled_cash_usd > 0
                else "use already-settled cash"
            )
            lines.append(f"             Cost: ~${cost:,.0f} USD  ·  {funded}")

    for item in optimization.add:
        render_candidate(item, "+ ADD")
    if optimization.add:
        lines.append("")

    keep_ids = {item.ticker.yf.upper() for item in optimization.keep}
    for note in optimization.admitted_over_limit:
        threshold = (
            f"{CONCENTRATION_INCUMBENT_MIN_SCORE:.0f} (incumbent)"
            if note.item.ticker.yf.upper() in keep_ids
            else f"{CONCENTRATION_EXCEPTION_MIN_SCORE:.0f}"
        )
        lines.append(
            f"  ⚠ over-limit admit  {_watchlist_symbol(note.item)}  — "
            f"high conviction, score "
            f"{watchlist_candidate_score(note.item):.0f}/200 ≥ {threshold}"
        )
        lines.extend(_breach_bullets(note, "        "))
    if optimization.admitted_over_limit:
        lines.append("")

    must_remove = [
        move for move in optimization.remove if move.reason == "verdict_reject"
    ]
    optional_remove = [
        move for move in optimization.remove if move.reason != "verdict_reject"
    ]
    if must_remove:
        lines.append("  MUST REMOVE (verdict reject):")
        for move in must_remove:
            verdict = move.item.analysis.verdict if move.item.analysis else "REJECT"
            lines.append(
                f"    − REMOVE FROM WATCHLIST  {_watchlist_symbol(move.item)}  "
                f"— verdict {verdict}"
            )
        lines.append("")
    if optional_remove:
        lines.append(
            "  OPTIONAL OPTIMIZATION (retain if you disagree with the ranking):"
        )
        for move in optional_remove:
            if move.reason == "concentration_displaced" and move.note is not None:
                lines.append(
                    f"    − REMOVE FROM WATCHLIST  {_watchlist_symbol(move.item)}"
                )
                lines.extend(_breach_bullets(move.note, "        "))
                lines.append(
                    "        · below retention bar (needs high conviction + "
                    f"score ≥ {CONCENTRATION_INCUMBENT_MIN_SCORE:.0f})"
                )
            else:
                lines.append(
                    f"    − REMOVE FROM WATCHLIST  {_watchlist_symbol(move.item)}  "
                    f"— {move.reason.replace('_', ' ')}"
                )
        lines.append("")

    if optimization.keep:
        lines.extend(
            wrap_listing(
                f"  KEEPING ACTIVE ({len(optimization.keep)}): ",
                [_watchlist_symbol(item) for item in optimization.keep],
            )
        )
    if optimization.monitors:
        lines.extend(
            wrap_listing(
                f"  KEEPING MONITORS ({len(optimization.monitors)}): ",
                [_watchlist_symbol(item) for item in optimization.monitors],
            )
        )
    if optimization.reviews:
        quick_review_count = sum(
            1
            for item in optimization.reviews
            if item.analysis and item.analysis.is_quick_mode
        )
        lines.extend(
            wrap_listing(
                f"  KEEPING REVIEWS ({len(optimization.reviews)}): ",
                [_watchlist_symbol(item) for item in optimization.reviews],
            )
        )
        if quick_review_count:
            lines.append(f"    ({quick_review_count} quick — re-run full)")
    if optimization.protected_tickers:
        lines.extend(
            wrap_listing(
                f"  KEEPING PROTECTED ({len(optimization.protected_tickers)}): ",
                list(optimization.protected_tickers),
            )
        )
        lines.append("    (held or unresolved; not changed automatically)")
    if (
        optimization.keep
        or optimization.monitors
        or optimization.reviews
        or optimization.protected_tickers
    ):
        lines.append("")

    if optimization.excluded_low_conviction:
        lines.append(
            "  Excluded below medium conviction: "
            f"{len(optimization.excluded_low_conviction)}"
        )
    if optimization.withheld_candidates:
        lines.append(
            f"  Withheld by concentration ({len(optimization.withheld_candidates)}):"
        )
        grouped: dict[str, list[ConcentrationNote]] = {}
        for note in optimization.withheld_candidates:
            grouped.setdefault(_breach_category(note), []).append(note)
        for notes in grouped.values():
            # Same breach shape across the group; keep the worst projection
            # per dimension so magnitude survives the grouping.
            label = " + ".join(
                f"{group[0].dimension} {group[0].key} up to "
                f"{max(breach.projected_pct for breach in group):.1f}%"
                f" > {group[0].limit_pct:.0f}%"
                for group in zip(*(note.breaches for note in notes), strict=True)
            )
            symbols = [_watchlist_symbol(note.item) for note in notes]
            lines.extend(wrap_listing(f"    · {label}:  ", symbols))
    if watchlist_candidates_blocked_by_cash:
        lines.append(
            "  Cash-blocked candidates retained for ranking: "
            f"{watchlist_candidates_blocked_by_cash}"
        )
        if not optimization.optimal and not portfolio_data_loaded:
            lines.append(
                "  No candidates shown — holdings and cash were not loaded "
                "(read-only run), so deployable cash is unknown."
            )
    if plan.in_flight_buys:
        lines.append(
            "  ✓ "
            f"{len(plan.in_flight_buys)} BUY order"
            f"{'s' if len(plan.in_flight_buys) != 1 else ''} already in flight "
            f"({', '.join(_watchlist_symbol(item) for item in plan.in_flight_buys)})"
            " — excluded from changes"
        )
    if plan.live_order_items:
        lines.append("  LIVE ORDER STATUS:")
        for item in plan.live_order_items:
            order_note = build_live_order_note(item, live_orders)
            if order_note:
                lines.append(f"    {_watchlist_symbol(item)}  {order_note.strip()}")

    replacement_entries = [
        *optimization.optimal,
        *optimization.monitors,
        *optimization.reviews,
    ]
    replacement_symbols = [item.ticker.ibkr for item in replacement_entries]
    replacement_symbols.extend(
        Ticker.from_yf(ticker).ibkr for ticker in optimization.protected_tickers
    )
    if (
        optimization.watchlist_supplied
        and replacement_symbols
        and not optional_remove
        and len(replacement_symbols) == len(set(replacement_symbols))
    ):
        lines.extend(
            wrap_listing("  [Update IBKR watchlist to]: ", replacement_symbols)
        )
    elif optional_remove:
        lines.append(
            "  Exact replacement list withheld — decide optional removals first."
        )
    elif len(replacement_symbols) != len(set(replacement_symbols)):
        lines.append(
            "  Exact replacement list withheld — raw IBKR symbols are "
            "exchange-ambiguous."
        )
    lines.append("")
    return lines
