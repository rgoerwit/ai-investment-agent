from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from src.ibkr.dip_watch import DipWatchCandidate, build_dip_watch_candidates
from src.ibkr.models import (
    AnalysisRecord,
    NormalizedPosition,
    ReconciliationItem,
)
from src.ibkr.portfolio_presentation import (
    aggregate_sector_weights,
    build_action_summary_counts,
    build_cash_summary,
    build_freshness_overview,
    build_live_order_note,
    build_portfolio_overview,
    group_portfolio_actions,
)
from src.ibkr.recommendation_service import PortfolioRecommendationBundle
from src.ibkr.screening_freshness import ScreeningFreshnessSummary
from src.web.ibkr_dashboard.drilldown_service import build_structured_sections

_DIP_WATCH_LIMIT = 7


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
    return {
        "status": status,
        "as_of": fetched_at or datetime.now(UTC).isoformat(),
        "cache_hit": cache_hit,
        "refreshing": refreshing,
        "load_error": load_error,
        "read_only": read_only,
        "portfolio": _serialize_portfolio(bundle),
        "overview": _serialize_overview(bundle),
        "macro_alert": macro_alert,
        "screening_freshness": _serialize_screening_freshness(
            bundle.screening_freshness
        ),
        "freshness": _serialize_freshness(bundle),
        "freshness_overview": _serialize_freshness_overview(bundle),
        "actions": _serialize_actions(
            bundle.items,
            bundle.health_flags,
            watchlist_tickers=bundle.watchlist_tickers,
            live_orders=bundle.live_orders,
        ),
        "watchlist": {
            "name": bundle.watchlist_name,
            "total": bundle.watchlist_total,
            "tickers": sorted(bundle.watchlist_tickers),
        },
        "orders": bundle.live_orders,
        "health_flags": list(bundle.health_flags),
        "positions": [
            _serialize_position_row(item, live_orders=bundle.live_orders)
            for item in bundle.items
            if item.ibkr_position
        ],
        "summary_counts": _summary_counts(
            bundle.items,
            watchlist_tickers=bundle.watchlist_tickers,
        ),
        "cash_summary": _serialize_cash_summary(bundle),
        "cash_timeline": _serialize_cash_timeline(bundle),
    }


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
    return {
        "ticker_yf": item.ticker.yf,
        "ticker_ibkr": item.ticker.ibkr,
        "action": item.action,
        "sell_type": item.sell_type,
        "reason": item.reason,
        "urgency": item.urgency,
        "is_watchlist": item.is_watchlist,
        "suggested_quantity": item.suggested_quantity,
        "suggested_price": item.suggested_price,
        "suggested_order_type": item.suggested_order_type,
        "cash_impact_usd": item.cash_impact_usd,
        "settlement_date": item.settlement_date,
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


def _serialize_overview(bundle: PortfolioRecommendationBundle) -> dict[str, Any]:
    overview = build_portfolio_overview(
        bundle.items,
        bundle.portfolio,
        watchlist_tickers=bundle.watchlist_tickers,
    )
    return {
        "sells": overview.sell_count,
        "reviews": overview.review_count,
        "holds": overview.hold_count,
        "macro_watch": overview.macro_watch_count,
        "new_buys": overview.new_buy_count,
        "candidates": overview.candidate_count,
        "total_items": overview.total_items,
        "position_count": overview.position_count,
        "has_live_positions": overview.has_live_positions,
        "is_candidate_heavy": overview.is_candidate_heavy,
    }


def _serialize_actions(
    items: list[ReconciliationItem],
    health_flags: list[str],
    *,
    watchlist_tickers: set[str] | None,
    live_orders: list[dict] | None,
) -> dict[str, Any]:
    groups = group_portfolio_actions(items, watchlist_tickers=watchlist_tickers)
    dip_watch = [
        _serialize_dip_watch(candidate)
        for candidate in build_dip_watch_candidates(
            list(groups.macro_reviews),
            limit=_DIP_WATCH_LIMIT,
        )
    ]

    return {
        "sell_stop_breach": [
            serialize_item(item, live_orders=live_orders) for item in groups.stop_sells
        ],
        "sell_hard": [
            serialize_item(item, live_orders=live_orders) for item in groups.hard_sells
        ],
        "sell_soft_review": [
            serialize_item(item, live_orders=live_orders) for item in groups.soft_sells
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
        "watchlist_buy": [
            serialize_item(item, live_orders=live_orders) for item in groups.new_buys
        ],
        "watchlist_candidate": [
            serialize_item(item, live_orders=live_orders)
            for item in groups.watchlist_candidates
        ],
        "watchlist_monitor": [
            serialize_item(item, live_orders=live_orders) for item in groups.holds_watch
        ],
        "watchlist_remove": [
            serialize_item(item, live_orders=live_orders) for item in groups.removes
        ],
        "macro_event_detected": any(
            "CORRELATED_SELL_EVENT" in flag for flag in health_flags
        ),
    }


def _serialize_dip_watch(candidate: DipWatchCandidate) -> dict[str, Any]:
    return {
        "ticker_yf": candidate.ticker_yf,
        "ticker_ibkr": candidate.ticker_ibkr,
        "score": candidate.score,
        "stars": candidate.stars,
        "dip_pct": candidate.dip_pct,
        "risk_reward": candidate.risk_reward,
        "held_quantity": candidate.held_quantity,
        "health_adj": candidate.health_adj,
        "growth_adj": candidate.growth_adj,
        "entry_price": candidate.entry_price,
        "current_price": candidate.current_price,
        "currency": candidate.currency,
        "run_ticker": candidate.run_ticker,
    }


def _serialize_position(position: NormalizedPosition | None) -> dict[str, Any] | None:
    if position is None:
        return None
    return {
        "ticker_yf": position.ticker.yf,
        "ticker_ibkr": position.ticker.ibkr,
        "quantity": position.quantity,
        "avg_cost_local": position.avg_cost_local,
        "current_price_local": position.current_price_local,
        "currency": position.currency,
        "market_value_usd": position.market_value_usd,
        "unrealized_pnl_usd": position.unrealized_pnl_usd,
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
        "entry_price": analysis.entry_price,
        "stop_price": analysis.stop_price,
        "target_1_price": analysis.target_1_price,
        "target_2_price": analysis.target_2_price,
        "conviction": analysis.conviction,
        "sector": analysis.sector,
        "exchange": analysis.exchange,
        "is_quick_mode": analysis.is_quick_mode,
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
    *,
    watchlist_tickers: set[str] | None,
) -> dict[str, int]:
    groups = group_portfolio_actions(items, watchlist_tickers=watchlist_tickers)
    counts = build_action_summary_counts(groups)
    return {
        "buys": counts.get("BUY", 0),
        "candidates": counts.get("CANDIDATES", 0),
        "sells": counts.get("SELL", 0),
        "reviews": counts.get("REVIEW", 0),
        "holds": counts.get("HOLD", 0),
        "macro_watch": counts.get("MACRO_WATCH", 0),
        "watchlist": sum(1 for item in items if item.is_watchlist),
        "total": len(items),
    }


def _serialize_cash_summary(bundle: PortfolioRecommendationBundle) -> dict[str, Any]:
    summary = build_cash_summary(bundle.items, bundle.portfolio)
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


def _serialize_cash_timeline(
    bundle: PortfolioRecommendationBundle,
) -> list[dict[str, Any]]:
    return list(_serialize_cash_summary(bundle)["pending_inflows"])
