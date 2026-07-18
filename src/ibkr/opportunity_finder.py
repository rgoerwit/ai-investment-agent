"""Phase 2 off-watchlist BUY opportunity discovery."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from src.config import config
from src.ibkr.concentration import (
    canonical_exchange_bucket,
    canonical_sector_bucket,
    format_concentration_warnings,
    project_concentration_breaches,
)
from src.ibkr.models import (
    AnalysisRecord,
    PortfolioSummary,
    ReconciliationItem,
)

logger = structlog.get_logger(__name__)
from src.ibkr.order_builder import calculate_quantity
from src.ibkr.reconciliation_rules import (
    _MIN_ORDER_USD,
    _normalize_verdict,
    _resolve_fx,
    check_staleness,
)
from src.ibkr.ticker import Ticker

if TYPE_CHECKING:
    from src.ibkr.reconciler import ReconciliationDiagnostics


def find_opportunities(
    analyses: dict[str, AnalysisRecord],
    held_tickers: set[str],
    portfolio: PortfolioSummary,
    *,
    diagnostics: ReconciliationDiagnostics | None,
    structural_macro_events: list,
    max_age_days: int,
    drift_threshold_pct: float,
    sector_limit_pct: float,
    exchange_limit_pct: float,
    sector_weights: dict[str, float],
    exchange_weights: dict[str, float],
    remaining_cash: float,
) -> tuple[list[ReconciliationItem], float]:
    """Find new BUY recommendations not already held or handled by watchlist."""
    items: list[ReconciliationItem] = []
    withheld_unstable: list[str] = []

    for ticker, analysis in analyses.items():
        if ticker in held_tickers:
            continue
        if _normalize_verdict(analysis.verdict or "") != "BUY":
            continue

        # Quick-mode BUYs are screening candidates, not investable signals — surface as
        # REVIEW so the off-watchlist finder never proposes an order off a fast-tier
        # verdict (the analysis itself carries a QUICK-MODE QUALIFICATION note).
        if getattr(analysis, "is_quick_mode", False):
            items.append(
                ReconciliationItem(
                    ticker=Ticker.from_yf(ticker),
                    action="REVIEW",
                    reason=(
                        f"Quick-mode screening BUY ({analysis.analysis_date}) — "
                        "re-run full analysis before acting"
                    ),
                    urgency="LOW",
                    analysis=analysis,
                )
            )
            continue

        # Opt-in BUY stability gate. Withhold a fresh BUY that is either
        # contradicted by recent same-ticker runs (verdict-noise defense) or
        # marginal (risk_tally >= margin) with an unresolved peak/transient
        # quality flag. The gate is agents-free (src.ibkr.buy_stability +
        # neutral parser); the lazy import keeps it off the import path entirely
        # when disabled. risk_tally + quality_flag_types are persisted on
        # AnalysisRecord by the analysis index, so both branches are live.
        if getattr(config, "buy_stability_enabled", False):
            from src.ibkr.buy_stability import (
                BuyStabilityConfig,
                assess_buy_stability,
                load_recent_same_ticker_verdicts,
            )

            stability_cfg = BuyStabilityConfig.from_config(config)
            prior_verdicts = load_recent_same_ticker_verdicts(
                ticker,
                lookback_days=stability_cfg.lookback_days,
                results_dir=config.results_dir,
                exclude_path=analysis.file_path or None,
            )
            withhold_reason = assess_buy_stability(
                analysis.verdict,
                prior_verdicts,
                risk_tally=analysis.risk_tally,
                active_flags=analysis.quality_flag_types,
                cfg=stability_cfg,
            )
            if withhold_reason:
                logger.debug(
                    "offwatch_buy_withheld_unstable",
                    ticker=ticker,
                    reason=withhold_reason,
                )
                withheld_unstable.append(ticker)
                continue

        is_stale, _stale_reason = check_staleness(
            analysis,
            None,
            max_age_days,
            drift_threshold_pct,
            structural_macro_events=structural_macro_events,
        )
        if is_stale:
            continue

        has_portfolio = portfolio.portfolio_value_usd > 0
        if has_portfolio and remaining_cash <= 0:
            entry_price = analysis.entry_price or analysis.current_price
            conviction = analysis.conviction or analysis.trade_block.conviction or ""
            size_pct = analysis.trade_block.size_pct or (analysis.position_size or 0)
            items.append(
                ReconciliationItem(
                    ticker=Ticker.from_yf(ticker),
                    action="BUY",
                    reason=(
                        f"New BUY ({analysis.analysis_date}) — {conviction} conviction, "
                        f"target {size_pct:.1f}% — no cash available"
                    ),
                    urgency="MEDIUM",
                    analysis=analysis,
                    suggested_price=entry_price,
                    suggested_order_type="LMT",
                    cash_impact_usd=0.0,
                    is_cash_blocked=True,
                )
            )
            if diagnostics is not None:
                diagnostics.cash_blocked_offwatch_buy_count += 1
            continue

        entry_price = analysis.entry_price or analysis.current_price
        conviction = analysis.conviction or analysis.trade_block.conviction or ""
        size_pct = analysis.trade_block.size_pct or (analysis.position_size or 0)

        fx_rate = _resolve_fx(analysis)
        buy_qty = calculate_quantity(
            available_cash_usd=remaining_cash,
            entry_price_local=entry_price or 0.0,
            fx_rate_to_usd=fx_rate,
            size_pct=size_pct,
            portfolio_value_usd=portfolio.portfolio_value_usd,
            yf_ticker=ticker,
        )
        buy_cost_usd = buy_qty * (entry_price or 0.0) * fx_rate

        if buy_qty > 0 and buy_cost_usd < _MIN_ORDER_USD:
            continue

        remaining_cash -= buy_cost_usd

        buy_reason = f"New BUY ({analysis.analysis_date}) — {conviction} conviction, target {size_pct:.1f}%"
        if portfolio.portfolio_value_usd > 0:
            projected_weight = buy_cost_usd / portfolio.portfolio_value_usd * 100
            breaches = project_concentration_breaches(
                exchange_key=canonical_exchange_bucket(
                    ticker,
                    analysis_exchange=analysis.exchange,
                ),
                sector_key=canonical_sector_bucket(analysis.sector),
                candidate_pct=projected_weight,
                exchange_weights=exchange_weights,
                sector_weights=sector_weights,
                exchange_limit_pct=exchange_limit_pct,
                sector_limit_pct=sector_limit_pct,
            )
            conc_warns = format_concentration_warnings(breaches)
            if conc_warns:
                buy_reason += "  " + "; ".join(conc_warns)

        items.append(
            ReconciliationItem(
                ticker=Ticker.from_yf(ticker),
                action="BUY",
                reason=buy_reason,
                urgency="MEDIUM",
                analysis=analysis,
                suggested_quantity=buy_qty if buy_qty > 0 else None,
                suggested_price=entry_price,
                suggested_order_type="LMT",
                cash_impact_usd=-buy_cost_usd if buy_cost_usd > 0 else 0.0,
            )
        )

    if withheld_unstable:
        # Often dozens across a large index — one operator-visible summary, per-ticker
        # detail at debug (mirrors reconciler.alpha_base_ambiguous_summary).
        logger.info(
            "offwatch_buy_withheld_unstable_summary",
            count=len(withheld_unstable),
            sample=sorted(withheld_unstable)[:8],
            reason=(
                "marginal BUY with unresolved peak/transient flag — "
                "withheld pending stability"
            ),
        )

    return items, remaining_cash
