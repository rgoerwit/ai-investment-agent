from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast

from src.ibkr.dip_watch import DipWatchCandidate, build_dip_watch_candidates
from src.ibkr.models import (
    AnalysisRecord,
    NormalizedPosition,
    ReconciliationItem,
)
from src.ibkr.order_presentation import build_live_order_note
from src.ibkr.portfolio_action_plan import (
    PortfolioActionPlan,
    build_action_plan_counts,
    build_portfolio_action_plan,
    has_active_macro_event,
)
from src.ibkr.portfolio_presentation import (
    PortfolioActionGroups,
    build_action_display_sections,
    build_action_summary_counts,
    build_cash_summary,
    build_freshness_overview,
    fx_return_split_diagnostic,
    get_sell_type_label,
    retail_safe_action,
)
from src.ibkr.recommendation_service import PortfolioRecommendationBundle
from src.ibkr.screening_freshness import ScreeningFreshnessSummary
from src.ibkr.watchlist_optimization import (
    ConcentrationNote,
    WatchlistMove,
    concentration_breach_summary,
)
from src.sector_normalization import aggregate_sector_weights
from src.web.ibkr_dashboard.drilldown_service import build_structured_sections


def _build_dashboard_action_view(
    bundle: PortfolioRecommendationBundle,
) -> PortfolioActionPlan:
    return build_portfolio_action_plan(
        bundle.items,
        bundle.portfolio,
        watchlist_tickers=bundle.watchlist_tickers,
        watchlist_supplied=(
            bundle.watchlist_total is not None and not bundle.watchlist_unavailable
        ),
        watchlist_unavailable=bundle.watchlist_unavailable,
        live_orders=bundle.live_orders,
        macro_event_active=has_active_macro_event(bundle.health_flags),
        exchange_limit_pct=bundle.exchange_limit_pct,
        sector_limit_pct=bundle.sector_limit_pct,
    )


def serialize_dashboard_snapshot(
    bundle: PortfolioRecommendationBundle,
    *,
    status: str = "ready",
    fetched_at: str | None = None,
    cache_hit: bool = False,
    refreshing: bool = False,
    load_error: str | None = None,
    macro_alert: dict[str, Any] | None = None,
    read_only: bool = False,
) -> dict[str, Any]:
    view = _build_dashboard_action_view(bundle)
    cash_summary = _serialize_cash_summary(bundle, view)
    return {
        "status": status,
        "as_of": fetched_at or datetime.now(UTC).isoformat(),
        "cache_hit": cache_hit,
        "refreshing": refreshing,
        "load_error": load_error,
        "read_only": read_only,
        "portfolio": _serialize_portfolio(bundle),
        "overview": _serialize_overview(bundle, view),
        "macro_alert": macro_alert,
        "screening_freshness": _serialize_screening_freshness(
            bundle.screening_freshness
        ),
        "freshness": _serialize_freshness(bundle),
        "freshness_overview": _serialize_freshness_overview(bundle),
        "actions": _serialize_actions(
            view,
            live_orders=bundle.live_orders,
        ),
        "watchlist": {
            "name": bundle.watchlist_name,
            "total": bundle.watchlist_total,
            "tickers": sorted(bundle.watchlist_tickers),
            "status": _watchlist_status(bundle),
        },
        "orders": bundle.live_orders,
        "health_flags": list(bundle.health_flags),
        "positions": [
            _serialize_position_row(item, live_orders=bundle.live_orders)
            for item in bundle.items
            if item.ibkr_position
        ],
        "summary_counts": _summary_counts(bundle.items, view),
        "cash_summary": cash_summary,
        "cash_timeline": list(cash_summary["pending_inflows"]),
        # Limits let the concentration cards render "limit 40%" and warn near cap.
        "concentration_limits": {
            "sector": bundle.sector_limit_pct,
            "exchange": bundle.exchange_limit_pct,
        },
        # Non-fatal data-source failures (e.g. {"live_orders": "..."}) so the UI
        # can distinguish "no live orders" from "live orders could not load".
        "errors": dict(bundle.errors),
    }


def _serialize_breaches(note: ConcentrationNote) -> list[dict[str, Any]]:
    return [
        {
            "dimension": breach.dimension,
            "key": breach.key,
            "projected_pct": breach.projected_pct,
            "limit_pct": breach.limit_pct,
        }
        for breach in note.breaches
    ]


def _watchlist_status(bundle: PortfolioRecommendationBundle) -> str:
    if bundle.watchlist_unavailable:
        return "unavailable"
    if bundle.watchlist_total is None:
        return "not_loaded"
    return "loaded"


def serialize_equity_drilldown(
    item: ReconciliationItem,
    *,
    live_orders: list[dict] | None = None,
    analysis_json: dict[str, Any] | None,
    report_markdown_html: str | None,
    report_markdown_path: str | None,
    article_markdown_html: str | None,
    article_markdown_path: str | None,
) -> dict[str, Any]:
    payload = serialize_item(item, live_orders=live_orders)
    payload["structured"] = build_structured_sections(analysis_json)
    payload["report_markdown_html"] = report_markdown_html
    payload["report_markdown_path"] = report_markdown_path
    payload["article_markdown_html"] = article_markdown_html
    payload["article_markdown_path"] = article_markdown_path
    payload["note"] = (
        "no markdown report saved"
        if report_markdown_html is None and article_markdown_html is None
        else None
    )
    return payload


def serialize_item(
    item: ReconciliationItem,
    *,
    live_orders: list[dict] | None = None,
) -> dict[str, Any]:
    item = retail_safe_action(item)
    return {
        "ticker_yf": item.ticker.yf,
        "ticker_ibkr": item.ticker.ibkr,
        "action": item.action,
        "sell_type": item.sell_type,
        "sell_type_label": get_sell_type_label(item.sell_type),
        "action_basis": item.action_basis,
        "reason": item.reason,
        "urgency": item.urgency,
        "is_watchlist": item.is_watchlist,
        "suggested_quantity": item.suggested_quantity,
        "suggested_price": item.suggested_price,
        "suggested_order_type": item.suggested_order_type,
        "cash_impact_usd": item.cash_impact_usd,
        "settlement_date": item.settlement_date,
        "cost_basis_return_pct": item.cost_basis_return_pct,
        "profit_take_reasons": list(item.profit_take_reasons),
        "live_order_note": build_live_order_note(item, live_orders),
        "position": _serialize_position(item.ibkr_position),
        "analysis": _serialize_analysis(item.analysis),
    }


def _serialize_portfolio(bundle: PortfolioRecommendationBundle) -> dict[str, Any]:
    portfolio = bundle.portfolio
    buffer_reserve = max(portfolio.settled_cash_usd - portfolio.available_cash_usd, 0.0)
    return {
        "account_id": portfolio.account_id,
        "net_liquidation_usd": portfolio.portfolio_value_usd,
        "cash_balance_usd": portfolio.cash_balance_usd,
        "settled_cash_usd": portfolio.settled_cash_usd,
        "available_cash_usd": portfolio.available_cash_usd,
        "buffer_reserve_usd": buffer_reserve,
        "cash_pct": portfolio.cash_pct,
        "position_count": portfolio.position_count,
        "sector_weights": aggregate_sector_weights(portfolio.sector_weights),
        "exchange_weights": portfolio.exchange_weights,
    }


def _serialize_freshness(bundle: PortfolioRecommendationBundle) -> dict[str, Any]:
    summary = bundle.freshness_summary
    return {
        "blocking_now": [_serialize_freshness_row(row) for row in summary.blocking_now],
        "stale_in_queue": [
            _serialize_freshness_row(row) for row in summary.stale_in_queue
        ],
        "due_soon": [_serialize_freshness_row(row) for row in summary.due_soon],
        "candidate_blocked": [
            _serialize_freshness_row(row) for row in summary.candidate_blocked
        ],
        "fresh_count": len(summary.fresh),
        "refresh_activity": {
            "policy": bundle.refresh_activity.policy,
            "limit": bundle.refresh_activity.limit,
            "queued": list(bundle.refresh_activity.queued),
            "refreshed": list(bundle.refresh_activity.refreshed),
            "failed": list(bundle.refresh_activity.failed),
            "skipped_due_to_policy": list(
                bundle.refresh_activity.skipped_due_to_policy
            ),
            "skipped_due_to_limit": list(bundle.refresh_activity.skipped_due_to_limit),
            "skipped_read_only": list(bundle.refresh_activity.skipped_read_only),
        },
    }


def _serialize_screening_freshness(
    summary: ScreeningFreshnessSummary,
) -> dict[str, Any]:
    return {
        "status": summary.status,
        "screening_date": summary.screening_date,
        "completed_at": summary.completed_at,
        "age_days": summary.age_days,
        "stale_after_days": summary.stale_after_days,
        "candidate_count": summary.candidate_count,
        "buy_count": summary.buy_count,
    }


def _serialize_freshness_overview(
    bundle: PortfolioRecommendationBundle,
) -> dict[str, Any]:
    overview = build_freshness_overview(
        bundle.freshness_summary,
        bundle.refresh_activity,
    )
    return {
        "blocking_now": overview.blocking_now,
        "stale_in_queue": overview.stale_in_queue,
        "due_soon": overview.due_soon,
        "candidate_blocked": overview.candidate_blocked,
        "fresh_count": overview.fresh_count,
        "refreshed_count": overview.refreshed_count,
        "failed_count": overview.failed_count,
        "queued_count": overview.queued_count,
        "skipped_due_to_limit": overview.skipped_due_to_limit,
        "skipped_read_only": overview.skipped_read_only,
    }


def _serialize_overview(
    bundle: PortfolioRecommendationBundle,
    view: PortfolioActionPlan,
) -> dict[str, Any]:
    counts = build_action_summary_counts(view.groups)
    # Buy/candidate chips come from the optimizer view, not the raw groups —
    # the header must agree with the filtered action lists below it.
    new_buys = len(view.optimization.keep)
    candidates = len(view.optimization.add)
    return {
        "sells": counts.get("SELL", 0),
        "reviews": counts.get("REVIEW", 0),
        "holds": counts.get("HOLD", 0),
        "macro_watch": counts.get("MACRO_WATCH", 0),
        "new_buys": new_buys,
        "candidates": candidates,
        "total_items": len(bundle.items),
        "position_count": bundle.portfolio.position_count,
        "has_live_positions": bundle.portfolio.position_count > 0,
        "is_candidate_heavy": bundle.portfolio.position_count == 0
        and (new_buys > 0 or candidates > 0),
    }


def _serialize_watchlist_move(
    move: WatchlistMove,
    *,
    live_orders: list[dict] | None,
    reason_key: str = "removal_reason",
) -> dict[str, Any]:
    row = serialize_item(move.item, live_orders=live_orders)
    row[reason_key] = move.reason
    if move.note is not None:
        row["concentration"] = concentration_breach_summary(move.note)
        row["breaches"] = _serialize_breaches(move.note)
    return row


def _serialize_actions(
    view: PortfolioActionPlan,
    *,
    live_orders: list[dict] | None,
) -> dict[str, Any]:
    groups = view.groups
    optimization = view.optimization
    dip_watch = [
        _serialize_dip_watch(candidate)
        for candidate in build_dip_watch_candidates(list(groups.dip_candidates))
    ]

    return {
        "sell_stop_breach": [
            serialize_item(item, live_orders=live_orders) for item in groups.stop_sells
        ],
        "sell_hard": [
            serialize_item(item, live_orders=live_orders) for item in groups.hard_sells
        ],
        "sell_profit_take": [
            serialize_item(item, live_orders=live_orders)
            for item in groups.profit_take_sells
        ],
        "sell_soft_review": [
            serialize_item(item, live_orders=live_orders) for item in groups.soft_sells
        ],
        "review_profit_take": [
            serialize_item(item, live_orders=live_orders)
            for item in groups.profit_take_reviews
        ],
        "review_macro": [
            serialize_item(item, live_orders=live_orders)
            for item in groups.macro_reviews
        ],
        "review_stop_breach": [
            serialize_item(item, live_orders=live_orders)
            for item in groups.macro_stop_reviews
        ],
        "review": [
            serialize_item(item, live_orders=live_orders) for item in groups.reviews
        ],
        "hold": [
            serialize_item(item, live_orders=live_orders) for item in groups.holds_real
        ],
        "add": [serialize_item(item, live_orders=live_orders) for item in groups.adds],
        "trim": [
            serialize_item(item, live_orders=live_orders) for item in groups.trims
        ],
        "dip_watch": dip_watch,
        # Watchlist lists mirror the CLI's WATCHLIST ADDITION REVIEW section:
        # selected keeps/adds, removals, the non-empty floor, and
        # concentration-withheld off-watch names — never the raw groups.
        "watchlist_buy": [
            serialize_item(item, live_orders=live_orders) for item in optimization.keep
        ],
        "watchlist_candidate": [
            serialize_item(item, live_orders=live_orders) for item in optimization.add
        ],
        "watchlist_monitor": [
            serialize_item(item, live_orders=live_orders)
            for item in optimization.monitors
        ],
        "watchlist_remove": [
            _serialize_watchlist_move(move, live_orders=live_orders)
            for move in optimization.remove
        ],
        "watchlist_floor_retained": [
            _serialize_watchlist_move(
                move,
                live_orders=live_orders,
                reason_key="retention_reason",
            )
            for move in optimization.retained_for_watchlist_floor
        ],
        "watchlist_withheld": [
            {
                **serialize_item(note.item, live_orders=live_orders),
                "concentration": concentration_breach_summary(note),
                "breaches": _serialize_breaches(note),
            }
            for note in optimization.withheld_candidates
        ],
        "watchlist_capacity_limited": [
            serialize_item(item, live_orders=live_orders)
            for item in optimization.capacity_limited_candidates
        ],
        "watchlist_below_conviction": [
            serialize_item(item, live_orders=live_orders)
            for item in optimization.excluded_low_conviction
        ],
        "watchlist_in_flight": [
            {
                **serialize_item(item, live_orders=live_orders),
                "watchlist_membership": (
                    "not_on_loaded_watchlist"
                    if optimization.watchlist_supplied
                    else "unknown"
                ),
            }
            for item in view.in_flight_buys
        ],
        "action_sections": _serialize_action_sections(
            groups,
            dip_watch=tuple(dip_watch),
            live_orders=live_orders,
        ),
        "macro_event_detected": view.macro_event_active,
    }


def _serialize_action_sections(
    groups: PortfolioActionGroups,
    *,
    dip_watch: tuple[dict[str, Any], ...] | tuple[Any, ...],
    live_orders: list[dict] | None,
) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for section in build_action_display_sections(groups, dip_watch_items=dip_watch):
        if section.kind == "dip_watch":
            items = [
                candidate
                if isinstance(candidate, dict)
                else _serialize_dip_watch(cast(DipWatchCandidate, candidate))
                for candidate in section.items
            ]
        else:
            items = [
                serialize_item(cast(ReconciliationItem, item), live_orders=live_orders)
                for item in section.items
            ]
        payload.append(
            {
                "key": section.key,
                "title": section.title,
                "kind": section.kind,
                "items": items,
            }
        )
    return payload


def _serialize_dip_watch(candidate: DipWatchCandidate) -> dict[str, Any]:
    return {
        "ticker_yf": candidate.ticker_yf,
        "ticker_ibkr": candidate.ticker_ibkr,
        "score": candidate.score,
        "stars": candidate.stars,
        "dip_pct": candidate.dip_pct,
        "upside_pct": candidate.upside_pct,
        "held_quantity": candidate.held_quantity,
        "health_adj": candidate.health_adj,
        "growth_adj": candidate.growth_adj,
        "entry_price": candidate.entry_price,
        "current_price": candidate.current_price,
        "currency": candidate.currency,
        "run_ticker": candidate.run_ticker,
        "source": candidate.source,
    }


def _serialize_position(position: NormalizedPosition | None) -> dict[str, Any] | None:
    if position is None:
        return None
    split, fx_return_issue = fx_return_split_diagnostic(position)
    local_pct, fx_pct, usd_pct = split if split is not None else (None, None, None)
    return {
        "ticker_yf": position.ticker.yf,
        "ticker_ibkr": position.ticker.ibkr,
        "ticker_identity_verified": position.ticker_identity_verified,
        "ticker_resolution_source": position.ticker_resolution_source,
        "quantity": position.quantity,
        "avg_cost_local": position.avg_cost_local,
        "current_price_local": position.current_price_local,
        "currency": position.currency,
        "market_value_usd": position.market_value_usd,
        "unrealized_pnl_usd": position.unrealized_pnl_usd,
        "fx_rate_to_usd": position.fx_rate_to_usd,
        "market_value_basis": position.market_value_basis,
        "unrealized_pnl_basis": position.unrealized_pnl_basis,
        "valuation_valid": position.valuation_valid,
        "valuation_issue": position.valuation_issue,
        # Local-price vs implied FX/basis decomposition (multiplicative
        # residual; None when unavailable).
        "local_return_pct": local_pct,
        "fx_effect_pct": fx_pct,
        "usd_return_pct": usd_pct,
        "fx_return_issue": fx_return_issue,
        "acquired_date": position.acquired_date,
        "holding_period_days": position.holding_period_days,
        "tax_term": position.tax_term,
    }


def _serialize_analysis(analysis: AnalysisRecord | None) -> dict[str, Any] | None:
    if analysis is None:
        return None
    return {
        "ticker": analysis.ticker,
        "analysis_date": analysis.analysis_date,
        "age_days": analysis.age_days,
        "verdict": analysis.verdict,
        "health_adj": analysis.health_adj,
        "growth_adj": analysis.growth_adj,
        "zone": analysis.zone,
        "position_size": analysis.position_size,
        "current_price": analysis.current_price,
        "currency": analysis.currency,
        "currency_source": analysis.currency_source,
        "currency_repaired": analysis.currency_repaired,
        "currency_repair_reason": analysis.currency_repair_reason,
        "entry_price": analysis.entry_price,
        "stop_price": analysis.stop_price,
        "target_1_price": analysis.target_1_price,
        "target_2_price": analysis.target_2_price,
        "conviction": analysis.conviction,
        "sector": analysis.sector,
        "exchange": analysis.exchange,
        "is_quick_mode": analysis.is_quick_mode,
        "capital_flag_types": list(analysis.capital_flag_types),
        # Fundamental thesis-break triggers (bear KILL_CRITERIA) — the exit
        # conditions the dashboard shows ahead of legacy downside-price context.
        "kill_criteria": list(analysis.kill_criteria),
        "trade_block": {
            "action": analysis.trade_block.action,
            "size_pct": analysis.trade_block.size_pct,
            "conviction": analysis.trade_block.conviction,
            "entry_price": analysis.trade_block.entry_price,
            "stop_price": analysis.trade_block.stop_price,
            "target_1_price": analysis.trade_block.target_1_price,
            "target_2_price": analysis.trade_block.target_2_price,
            "risk_reward": analysis.trade_block.risk_reward,
            "special": analysis.trade_block.special,
        },
    }


def _serialize_position_row(
    item: ReconciliationItem,
    *,
    live_orders: list[dict] | None,
) -> dict[str, Any]:
    payload = serialize_item(item, live_orders=live_orders)
    payload["display_group"] = "watchlist" if item.is_watchlist else "position"
    return payload


def _serialize_freshness_row(row: Any) -> dict[str, Any]:
    return {
        "display_ticker": row.display_ticker,
        "run_ticker": row.run_ticker,
        "bucket": row.bucket,
        "reason_family": row.reason_family,
        "reason_text": row.reason_text,
        "action": row.action,
        "age_days": row.age_days,
        "expires_date": row.expires_date,
        "days_until_due": row.days_until_due,
    }


def _summary_counts(
    items: list[ReconciliationItem],
    view: PortfolioActionPlan,
) -> dict[str, int]:
    return build_action_plan_counts(view, items)


def _serialize_cash_summary(
    bundle: PortfolioRecommendationBundle,
    view: PortfolioActionPlan,
) -> dict[str, Any]:
    summary = build_cash_summary(
        bundle.items,
        bundle.portfolio,
        executable_buy_ids=view.executable_buy_ids,
    )
    return {
        "total_cash_usd": summary.total_cash_usd,
        "settled_cash_usd": summary.settled_cash_usd,
        "available_cash_usd": summary.available_cash_usd,
        "buffer_reserve_usd": summary.buffer_reserve_usd,
        "unsettled_cash_usd": summary.unsettled_cash_usd,
        "recommended_buy_cost_usd": summary.recommended_buy_cost_usd,
        "settled_cash_after_recommended_buys_usd": (
            summary.settled_cash_after_recommended_buys_usd
        ),
        "pending_inflows_total_usd": summary.pending_inflows_total_usd,
        "next_settlement_date": summary.next_settlement_date,
        "pending_inflows": [
            {
                "ticker_yf": row.ticker_yf,
                "ticker_ibkr": row.ticker_ibkr,
                "action": row.action,
                "quantity": row.quantity,
                "cash_impact_usd": row.cash_impact_usd,
                "settlement_date": row.settlement_date,
            }
            for row in summary.pending_inflows
        ],
    }
