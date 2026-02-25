"""
Reconciler: Compare IBKR positions vs evaluator recommendations.

The equity evaluator produces one-off BUY/SELL/DNI verdicts per ticker,
unaware of existing positions. This module bridges that gap by:

1. Loading latest analysis JSONs from results/
2. Reading live IBKR positions
3. Generating position-aware actions:
   - Evaluator says BUY + not held       → BUY (new position)
   - Evaluator says BUY + already held    → HOLD or ADD (if underweight)
   - Evaluator says DNI/SELL + held       → SELL/CLOSE (exit position)
   - Evaluator says BUY + stale analysis  → REVIEW (needs re-analysis)
   - Held position + no analysis          → REVIEW (unknown status)
   - Position overweight vs target        → TRIM
   - Target hit                           → REVIEW (take profit?)
   - Stop breached                        → SELL (urgent)
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import structlog

from src.ibkr.models import (
    AnalysisRecord,
    NormalizedPosition,
    PortfolioSummary,
    ReconciliationItem,
    TradeBlockData,
)
from src.ibkr.order_builder import (
    calculate_quantity,
    parse_trade_block,
    round_to_lot_size,
)

logger = structlog.get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Analysis Loading
# ══════════════════════════════════════════════════════════════════════════════


def load_latest_analyses(results_dir: Path) -> dict[str, AnalysisRecord]:
    """
    Load the most recent analysis JSON for each ticker from results_dir.

    Returns dict of yf_ticker -> AnalysisRecord (most recent only).
    """
    if not results_dir.exists():
        logger.warning("results_dir_not_found", path=str(results_dir))
        return {}

    analyses: dict[str, AnalysisRecord] = {}

    for filepath in sorted(results_dir.glob("*_analysis.json"), reverse=True):
        try:
            with open(filepath) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.debug("analysis_load_failed", file=filepath.name, error=str(e))
            continue

        snapshot = data.get("prediction_snapshot", {})
        ticker = snapshot.get("ticker") or data.get("ticker", "")
        if not ticker:
            # Try to extract from filename: TICKER_YYYY-MM-DD_analysis.json
            parts = filepath.stem.split("_")
            if len(parts) >= 3:
                ticker = parts[0].replace("_", ".")
            if not ticker:
                continue

        # Skip if we already have a more recent analysis for this ticker
        if ticker in analyses:
            continue

        # Parse TRADE_BLOCK from trader output
        trader_plan = data.get("investment_analysis", {}).get("trader_plan", "") or ""
        trade_block = parse_trade_block(trader_plan) or TradeBlockData()

        # Build the analysis record
        record = AnalysisRecord(
            ticker=ticker,
            analysis_date=snapshot.get("analysis_date", "")
            or _extract_date_from_filename(filepath.name),
            file_path=str(filepath),
            verdict=snapshot.get("verdict", "") or "",
            health_adj=snapshot.get("health_adj"),
            growth_adj=snapshot.get("growth_adj"),
            zone=snapshot.get("zone", ""),
            position_size=snapshot.get("position_size"),
            current_price=snapshot.get("current_price"),
            currency=snapshot.get("currency", "USD"),
            fx_rate_to_usd=snapshot.get("fx_rate_to_usd"),
            trade_block=trade_block,
            # Structured TRADE_BLOCK fields from snapshot (if present)
            entry_price=snapshot.get("entry_price") or trade_block.entry_price,
            stop_price=snapshot.get("stop_price") or trade_block.stop_price,
            target_1_price=snapshot.get("target_1_price") or trade_block.target_1_price,
            target_2_price=snapshot.get("target_2_price") or trade_block.target_2_price,
            conviction=snapshot.get("conviction", "") or trade_block.conviction,
        )

        analyses[ticker] = record
        logger.debug(
            "analysis_loaded",
            ticker=ticker,
            verdict=record.verdict,
            date=record.analysis_date,
            age_days=record.age_days,
        )

    logger.info("analyses_loaded", count=len(analyses))
    return analyses


def _extract_date_from_filename(filename: str) -> str:
    """Extract YYYY-MM-DD from filename like '7203_T_2026-02-15_analysis.json'."""
    match = re.search(r"(\d{4}-\d{2}-\d{2})", filename)
    return match.group(1) if match else ""


# ══════════════════════════════════════════════════════════════════════════════
# Staleness Detection
# ══════════════════════════════════════════════════════════════════════════════


def check_staleness(
    analysis: AnalysisRecord,
    current_price_local: float | None = None,
    max_age_days: int = 14,
    drift_threshold_pct: float = 15.0,
) -> tuple[bool, str]:
    """
    Check if an analysis is stale and should be reviewed.

    Args:
        analysis: The analysis record to check
        current_price_local: Live price from IBKR (in local currency)
        max_age_days: Maximum age before considered stale
        drift_threshold_pct: Price movement threshold for staleness

    Returns:
        Tuple of (is_stale, reason)
    """
    reasons = []

    # Age check
    if analysis.age_days > max_age_days:
        reasons.append(f"age {analysis.age_days}d > {max_age_days}d limit")

    # Price drift check (requires both analysis price and current price)
    entry_price = analysis.entry_price or analysis.current_price
    if entry_price and current_price_local and entry_price > 0:
        drift_pct = abs((current_price_local - entry_price) / entry_price) * 100
        if drift_pct > drift_threshold_pct:
            direction = "up" if current_price_local > entry_price else "down"
            reasons.append(f"price drift {drift_pct:.1f}% {direction}")

    if reasons:
        return True, "; ".join(reasons)
    return False, ""


def check_stop_breach(
    analysis: AnalysisRecord,
    current_price_local: float,
) -> bool:
    """Check if current price has breached the stop-loss level."""
    stop = analysis.stop_price
    if stop and current_price_local > 0:
        return current_price_local < stop
    return False


def check_target_hit(
    analysis: AnalysisRecord,
    current_price_local: float,
) -> bool:
    """Check if current price has hit or exceeded TARGET_1."""
    target = analysis.target_1_price
    if target and current_price_local > 0:
        return current_price_local >= target
    return False


# ══════════════════════════════════════════════════════════════════════════════
# Settlement Helpers
# ══════════════════════════════════════════════════════════════════════════════


def _settlement_date(business_days: int) -> str:
    """Return settlement date as YYYY-MM-DD, skipping weekends."""
    from datetime import date, timedelta

    d = date.today()
    added = 0
    while added < business_days:
        d += timedelta(days=1)
        if d.weekday() < 5:  # Mon–Fri
            added += 1
    return d.isoformat()


# ══════════════════════════════════════════════════════════════════════════════
# Position-Aware Reconciliation
# ══════════════════════════════════════════════════════════════════════════════


def reconcile(
    positions: list[NormalizedPosition],
    analyses: dict[str, AnalysisRecord],
    portfolio: PortfolioSummary,
    max_age_days: int = 14,
    drift_threshold_pct: float = 15.0,
    overweight_threshold_pct: float = 20.0,
    underweight_threshold_pct: float = 20.0,
) -> list[ReconciliationItem]:
    """
    Compare IBKR positions against evaluator recommendations.

    This is the core function that translates one-off evaluator verdicts
    into position-aware actions. The evaluator doesn't know about existing
    positions — this function does.

    Args:
        positions: Live IBKR positions (normalized)
        analyses: Latest analysis per ticker (from load_latest_analyses)
        portfolio: Portfolio summary (cash, value)
        max_age_days: Max analysis age for staleness
        drift_threshold_pct: Price drift threshold
        overweight_threshold_pct: % overweight before suggesting TRIM
        underweight_threshold_pct: % shortfall vs target before suggesting ADD

    Returns:
        List of ReconciliationItems with position-aware actions
    """
    items: list[ReconciliationItem] = []
    held_tickers: set[str] = set()
    # Track remaining settled cash across Phase 1 ADDs and Phase 2 BUYs
    remaining_cash = portfolio.available_cash_usd

    # ── Phase 1: Evaluate existing positions ──
    for pos in positions:
        ticker = pos.yf_ticker
        held_tickers.add(ticker)
        analysis = analyses.get(ticker)

        if analysis is None:
            # Position held but NO analysis exists → needs review
            items.append(
                ReconciliationItem(
                    ticker=ticker,
                    action="REVIEW",
                    reason="Position held but no evaluator analysis found",
                    urgency="MEDIUM",
                    ibkr_position=pos,
                )
            )
            continue

        current_price = pos.current_price_local

        # Check stop breach (URGENT)
        if check_stop_breach(analysis, current_price):
            items.append(
                ReconciliationItem(
                    ticker=ticker,
                    action="SELL",
                    reason=f"Stop breached: price {current_price:.2f} < stop {analysis.stop_price:.2f}",
                    urgency="HIGH",
                    ibkr_position=pos,
                    analysis=analysis,
                    suggested_quantity=abs(int(pos.quantity)),
                    suggested_price=current_price,
                    suggested_order_type="LMT",
                    cash_impact_usd=pos.market_value_usd,
                    settlement_date=_settlement_date(2),
                )
            )
            continue

        # Check verdict conflict: we hold but evaluator says don't
        verdict_upper = (analysis.verdict or "").upper()
        if verdict_upper in ("DO_NOT_INITIATE", "SELL", "REJECT"):
            items.append(
                ReconciliationItem(
                    ticker=ticker,
                    action="SELL",
                    reason=f"Verdict → {analysis.verdict}  ({analysis.analysis_date})",
                    urgency="HIGH",
                    ibkr_position=pos,
                    analysis=analysis,
                    suggested_quantity=abs(int(pos.quantity)),
                    suggested_order_type="LMT",
                    suggested_price=current_price,
                    cash_impact_usd=pos.market_value_usd,
                    settlement_date=_settlement_date(2),
                )
            )
            continue

        # Check target hit (profit-taking review — more specific than staleness)
        if check_target_hit(analysis, current_price):
            items.append(
                ReconciliationItem(
                    ticker=ticker,
                    action="REVIEW",
                    reason=f"Target hit: price {current_price:.2f} >= T1 {analysis.target_1_price:.2f}",
                    urgency="LOW",
                    ibkr_position=pos,
                    analysis=analysis,
                )
            )
            continue

        # Check staleness (after target hit, which is more specific)
        is_stale, stale_reason = check_staleness(
            analysis, current_price, max_age_days, drift_threshold_pct
        )
        if is_stale:
            items.append(
                ReconciliationItem(
                    ticker=ticker,
                    action="REVIEW",
                    reason=f"Stale analysis: {stale_reason}",
                    urgency="MEDIUM",
                    ibkr_position=pos,
                    analysis=analysis,
                )
            )
            continue

        # Check overweight / underweight
        target_size_pct = analysis.trade_block.size_pct or (analysis.position_size or 0)
        if target_size_pct > 0 and portfolio.portfolio_value_usd > 0:
            actual_pct = (pos.market_value_usd / portfolio.portfolio_value_usd) * 100
            excess_pct = actual_pct - target_size_pct
            if excess_pct > overweight_threshold_pct:
                target_value_usd = portfolio.portfolio_value_usd * (
                    target_size_pct / 100
                )
                trim_value_usd = pos.market_value_usd - target_value_usd
                # Use implied USD price per share for cross-currency accuracy
                price_usd_per_share = (
                    pos.market_value_usd / abs(pos.quantity)
                    if pos.quantity != 0
                    else 1.0
                )
                trim_qty = round_to_lot_size(
                    int(trim_value_usd / (price_usd_per_share or 1.0)), ticker
                )
                items.append(
                    ReconciliationItem(
                        ticker=ticker,
                        action="TRIM",
                        reason=f"Overweight: {actual_pct:.1f}% vs target {target_size_pct:.1f}% (+{excess_pct:.1f}%)",
                        urgency="MEDIUM",
                        ibkr_position=pos,
                        analysis=analysis,
                        suggested_quantity=trim_qty,
                        suggested_price=pos.current_price_local,
                        suggested_order_type="LMT",
                        cash_impact_usd=trim_value_usd,
                        settlement_date=_settlement_date(2),
                    )
                )
                continue

            # Check underweight — only for BUY-verdict positions with sufficient shortfall
            shortfall_pct = target_size_pct - actual_pct
            verdict_upper = (analysis.verdict or "").upper()
            if (
                shortfall_pct > underweight_threshold_pct
                and verdict_upper == "BUY"
                and remaining_cash > 0
            ):
                target_value_usd = portfolio.portfolio_value_usd * (
                    target_size_pct / 100
                )
                add_value_usd = min(
                    target_value_usd - pos.market_value_usd, remaining_cash
                )
                price_usd_per_share = (
                    pos.market_value_usd / abs(pos.quantity)
                    if pos.quantity != 0
                    else 1.0
                )
                add_qty = round_to_lot_size(
                    int(add_value_usd / (price_usd_per_share or 1.0)), ticker
                )
                remaining_cash -= add_value_usd
                items.append(
                    ReconciliationItem(
                        ticker=ticker,
                        action="ADD",
                        reason=f"Underweight: {actual_pct:.1f}% vs target {target_size_pct:.1f}% (-{shortfall_pct:.1f}%)",
                        urgency="LOW",
                        ibkr_position=pos,
                        analysis=analysis,
                        suggested_quantity=add_qty,
                        suggested_price=analysis.entry_price or pos.current_price_local,
                        suggested_order_type="LMT",
                        cash_impact_usd=-add_value_usd,
                    )
                )
                continue

        # All clear — HOLD
        status_parts = []
        if analysis.entry_price and current_price:
            gain_pct = (
                (current_price - analysis.entry_price) / analysis.entry_price
            ) * 100
            status_parts.append(
                f"entry {analysis.entry_price:.2f} → {current_price:.2f} ({gain_pct:+.1f}%)"
            )
        if analysis.stop_price:
            status_parts.append(f"stop {analysis.stop_price:.2f}")
        if analysis.target_1_price:
            status_parts.append(f"T1 {analysis.target_1_price:.2f}")

        items.append(
            ReconciliationItem(
                ticker=ticker,
                action="HOLD",
                reason=f"Within targets — {'; '.join(status_parts)}"
                if status_parts
                else "Position OK",
                urgency="LOW",
                ibkr_position=pos,
                analysis=analysis,
            )
        )

    # ── Phase 2: Find BUY recommendations not yet held ──
    for ticker, analysis in analyses.items():
        if ticker in held_tickers:
            continue  # Already handled above

        verdict_upper = (analysis.verdict or "").upper()
        if verdict_upper not in ("BUY",):
            continue  # Only surface new BUY recommendations

        # Skip stale analyses for new buys
        is_stale, stale_reason = check_staleness(
            analysis, None, max_age_days, drift_threshold_pct
        )
        if is_stale:
            continue

        # Only skip new buys if we have a live portfolio but no cash
        # (in --read-only mode portfolio_value_usd=0, so buys still surface)
        has_portfolio = portfolio.portfolio_value_usd > 0
        if has_portfolio and remaining_cash <= 0:
            continue

        entry_price = analysis.entry_price or analysis.current_price
        conviction = analysis.conviction or analysis.trade_block.conviction or ""
        size_pct = analysis.trade_block.size_pct or (analysis.position_size or 0)

        buy_qty = calculate_quantity(
            available_cash_usd=remaining_cash,
            entry_price_local=entry_price or 0.0,
            fx_rate_to_usd=analysis.fx_rate_to_usd,
            size_pct=size_pct,
            portfolio_value_usd=portfolio.portfolio_value_usd,
            yf_ticker=ticker,
        )
        buy_cost_usd = buy_qty * (entry_price or 0.0) * (analysis.fx_rate_to_usd or 1.0)
        remaining_cash -= buy_cost_usd

        items.append(
            ReconciliationItem(
                ticker=ticker,
                action="BUY",
                reason=f"New BUY ({analysis.analysis_date}) — {conviction} conviction, target {size_pct:.1f}%",
                urgency="MEDIUM",
                analysis=analysis,
                suggested_quantity=buy_qty if buy_qty > 0 else None,
                suggested_price=entry_price,
                suggested_order_type="LMT",
                cash_impact_usd=-buy_cost_usd if buy_cost_usd > 0 else 0.0,
            )
        )

    # Sort: HIGH urgency first, then MEDIUM, then LOW
    urgency_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    items.sort(key=lambda x: urgency_order.get(x.urgency, 9))

    logger.info(
        "reconciliation_complete",
        total_items=len(items),
        sells=sum(1 for i in items if i.action == "SELL"),
        trims=sum(1 for i in items if i.action == "TRIM"),
        adds=sum(1 for i in items if i.action == "ADD"),
        buys=sum(1 for i in items if i.action == "BUY"),
        holds=sum(1 for i in items if i.action == "HOLD"),
        reviews=sum(1 for i in items if i.action == "REVIEW"),
    )

    return items
