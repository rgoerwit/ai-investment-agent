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

# Minimum USD order value — orders below this are not worth recommending.
_MIN_ORDER_USD: float = 200.0

# Human-readable exchange names for the concentration section.
_EXCHANGE_LONG_NAMES: dict[str, str] = {
    "HK": "Hong Kong",
    "T": "Japan",
    "KS": "Korea",
    "KQ": "Korea KOSDAQ",
    "TW": "Taiwan",
    "TWO": "Taiwan OTC",
    "AS": "Amsterdam",
    "DE": "Germany",
    "PA": "France",
    "L": "UK",
    "SS": "Shanghai",
    "SZ": "Shenzhen",
    "SI": "Singapore",
    "US": "United States",
    "MX": "Mexico",
    "MC": "Madrid",
    "AX": "Australia",
    "KL": "Malaysia",
    "VI": "Vienna",
}


def _exchange_from_ticker(ticker: str) -> str:
    """Infer exchange code from yfinance ticker suffix (e.g. '0005.HK' → 'HK')."""
    if "." not in ticker:
        return "US"
    return ticker.rsplit(".", 1)[-1].upper()


# ══════════════════════════════════════════════════════════════════════════════
# Analysis Loading
# ══════════════════════════════════════════════════════════════════════════════


def _parse_scores_from_final_decision(text: str) -> dict:
    """Extract health_adj, growth_adj, verdict, zone from a PM final_decision narrative.

    Handles two legacy text formats for analyses that predate prediction_snapshot:

    Mid-era (structured fields in text):
        HEALTH_ADJ: 79  /  GROWTH_ADJ: 83  /  VERDICT: BUY  /  ZONE: MODERATE

    Old-era (prose narrative):
        Financial Health: 70.8% (Adjusted)  /  Growth Transition: 66.7% (Adjusted)
        **Action**: **BUY**
    """
    result: dict = {}

    # ── Health score ──────────────────────────────────────────────────────────
    m = re.search(r"\bHEALTH_ADJ[:\s]+([0-9.]+)", text, re.IGNORECASE)
    if not m:
        # Handles "Financial Health: 70.8%" and "**Financial Health**: 70.8%"
        # [^0-9\n]+ stops at the first digit so the number isn't partially consumed
        m = re.search(r"Financial Health[^0-9\n]+([\d.]+)%", text, re.IGNORECASE)
    if m:
        try:
            result["health_adj"] = float(m.group(1))
        except ValueError:
            pass

    # ── Growth score ──────────────────────────────────────────────────────────
    m = re.search(r"\bGROWTH_ADJ[:\s]+([0-9.]+)", text, re.IGNORECASE)
    if not m:
        m = re.search(r"Growth Transition[^0-9\n]+([\d.]+)%", text, re.IGNORECASE)
    if m:
        try:
            result["growth_adj"] = float(m.group(1))
        except ValueError:
            pass

    # ── Verdict ───────────────────────────────────────────────────────────────
    # Try structured field, then PM header, then action markdown
    for pat in (
        r"\bVERDICT[:\s]+([A-Z][A-Z_]*)",
        r"PORTFOLIO MANAGER VERDICT:\s*([A-Z][A-Z_]*)",
        r"\*\*Action\*\*:\s*\*\*(\w[\w_]*)\*\*",
    ):
        m = re.search(pat, text)
        if m:
            result["verdict"] = m.group(1).upper()
            break

    # ── Risk zone ─────────────────────────────────────────────────────────────
    m = re.search(r"\bZONE[:\s]+(HIGH|MODERATE|LOW)\b", text, re.IGNORECASE)
    if m:
        result["zone"] = m.group(1).upper()

    return result


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

        # For analyses that predate prediction_snapshot, extract scores from the
        # PM final_decision text so that health/growth appear in the SELL output.
        if snapshot.get("health_adj") is None or not snapshot.get("verdict"):
            fd_text = (data.get("final_decision") or {}).get("decision", "") or ""
            if fd_text:
                fb = _parse_scores_from_final_decision(fd_text)
                if snapshot.get("health_adj") is None:
                    snapshot = {**snapshot, "health_adj": fb.get("health_adj")}
                if snapshot.get("growth_adj") is None:
                    snapshot = {**snapshot, "growth_adj": fb.get("growth_adj")}
                if not snapshot.get("verdict"):
                    snapshot = {**snapshot, "verdict": fb.get("verdict") or ""}
                if not snapshot.get("zone"):
                    snapshot = {**snapshot, "zone": fb.get("zone") or ""}

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
            zone=snapshot.get("zone") or "",
            position_size=snapshot.get("position_size"),
            current_price=snapshot.get("current_price"),
            currency=snapshot.get("currency") or "USD",
            fx_rate_to_usd=snapshot.get("fx_rate_to_usd"),
            trade_block=trade_block,
            # Structured TRADE_BLOCK fields from snapshot (if present)
            entry_price=snapshot.get("entry_price") or trade_block.entry_price,
            stop_price=snapshot.get("stop_price") or trade_block.stop_price,
            target_1_price=snapshot.get("target_1_price") or trade_block.target_1_price,
            target_2_price=snapshot.get("target_2_price") or trade_block.target_2_price,
            conviction=snapshot.get("conviction") or trade_block.conviction,
            # Concentration metadata (sector may be absent in older snapshots)
            sector=snapshot.get("sector") or "",
            exchange=snapshot.get("exchange") or _exchange_from_ticker(ticker),
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
        age_str = "no date" if analysis.age_days >= 9999 else f"{analysis.age_days}d"
        reasons.append(f"age {age_str} > {max_age_days}d limit")

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
        ratio = stop / current_price_local
        if ratio > 50 or ratio < 0.02:
            logger.warning(
                "stop_price_ratio_suspicious",
                ticker=analysis.ticker,
                stop=stop,
                current_price=current_price_local,
                ratio=f"{ratio:.1f}x",
                hint="Possible currency-unit mismatch or stale stop from different analysis",
            )
            return False  # suppress — don't trigger a stop on a likely data error
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
# Sell Classification
# ══════════════════════════════════════════════════════════════════════════════


def _classify_sell_type(analysis: AnalysisRecord | None, stop_breached: bool) -> str:
    """
    Classify why a position is being sold.

    Returns:
        "STOP_BREACH"  — mechanical stop-loss trigger
        "HARD_REJECT"  — fundamental failure (health_adj < 50 OR growth_adj < 50)
        "SOFT_REJECT"  — passed hard checks, rejected on soft tally / macro fear
    """
    if stop_breached:
        return "STOP_BREACH"
    if analysis is None:
        return "HARD_REJECT"  # no data → treat conservatively
    health_ok = (analysis.health_adj or 0.0) >= 50.0
    growth_ok = (analysis.growth_adj or 0.0) >= 50.0
    return "SOFT_REJECT" if (health_ok and growth_ok) else "HARD_REJECT"


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
    sector_limit_pct: float = 30.0,
    exchange_limit_pct: float = 40.0,
    watchlist_tickers: set[str] | None = None,
) -> list[ReconciliationItem]:
    """
    Compare IBKR positions against evaluator recommendations.

    This is the core function that translates one-off evaluator verdicts
    into position-aware actions. The evaluator doesn't know about existing
    positions — this function does.

    Args:
        positions: Live IBKR positions (normalized)
        analyses: Latest analysis per ticker (from load_latest_analyses)
        portfolio: Portfolio summary (cash, value); sector_weights and
            exchange_weights fields are populated as a side-effect.
        max_age_days: Max analysis age for staleness
        drift_threshold_pct: Price drift threshold
        overweight_threshold_pct: % overweight before suggesting TRIM
        underweight_threshold_pct: % shortfall vs target before suggesting ADD
        sector_limit_pct: Warn when an ADD/BUY would push any sector above this %
        exchange_limit_pct: Warn when an ADD/BUY would push any exchange above this %
        watchlist_tickers: Optional set of yfinance tickers from the IBKR watchlist.
            Tickers not currently held are evaluated in Phase 1.5 and surfaced as
            BUY/HOLD/REMOVE/REVIEW based on their analysis verdict.

    Returns:
        List of ReconciliationItems with position-aware actions
    """
    items: list[ReconciliationItem] = []
    held_tickers: set[str] = set()
    # Track remaining settled cash across Phase 1 ADDs and Phase 2 BUYs
    remaining_cash = portfolio.available_cash_usd

    # ── Pre-compute concentration weights from currently held positions ──
    # These are stored on the portfolio object for use by format_report().
    # Use sum of position values as denominator so weights always sum to 100%,
    # even when IBKR ledger total and position market values use different FX rates.
    _sector_weights: dict[str, float] = {}
    _exchange_weights: dict[str, float] = {}
    _total_pos_value = sum(p.market_value_usd for p in positions)
    if _total_pos_value > 0:
        for pos in positions:
            _analysis = analyses.get(pos.yf_ticker)
            _sector = (_analysis.sector if _analysis else "") or "Unknown"
            _exchange = (
                (_analysis.exchange if _analysis else "")
                or _exchange_from_ticker(pos.yf_ticker)
                or pos.exchange
                or "Unknown"
            )
            _w = pos.market_value_usd / _total_pos_value * 100
            _sector_weights[_sector] = _sector_weights.get(_sector, 0.0) + _w
            _exchange_weights[_exchange] = _exchange_weights.get(_exchange, 0.0) + _w
    portfolio.sector_weights = _sector_weights
    portfolio.exchange_weights = _exchange_weights

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
                    sell_type="STOP_BREACH",
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
                    sell_type=_classify_sell_type(analysis, stop_breached=False),
                )
            )
            continue

        # Check target hit (profit-taking review — more specific than staleness)
        if check_target_hit(analysis, current_price):
            items.append(
                ReconciliationItem(
                    ticker=ticker,
                    action="REVIEW",
                    reason=f"Target hit: price {current_price:.2f} >= target {analysis.target_1_price:.2f}",
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
                # Skip only when we CAN calculate a quantity but the cost is trivially
                # small. If add_qty=0 due to lot-size rounding, fall through to HOLD.
                actual_add_cost = add_qty * price_usd_per_share
                if add_qty == 0 or (add_qty > 0 and actual_add_cost < _MIN_ORDER_USD):
                    # Fall through to HOLD — not worth placing
                    pass
                else:
                    remaining_cash -= add_value_usd
                    # Check concentration limits
                    add_reason = f"Underweight: {actual_pct:.1f}% vs target {target_size_pct:.1f}% (-{shortfall_pct:.1f}%)"
                    if portfolio.portfolio_value_usd > 0:
                        exch = analysis.exchange or _exchange_from_ticker(ticker)
                        sect = analysis.sector or "Unknown"
                        proj_exch = (
                            _exchange_weights.get(exch, 0.0)
                            + add_value_usd / portfolio.portfolio_value_usd * 100
                        )
                        proj_sect = (
                            _sector_weights.get(sect, 0.0)
                            + add_value_usd / portfolio.portfolio_value_usd * 100
                        )
                        conc_warns = []
                        if proj_exch > exchange_limit_pct:
                            conc_warns.append(
                                f"⚠ {exch} → {proj_exch:.0f}% (limit {exchange_limit_pct:.0f}%)"
                            )
                        if proj_sect > sector_limit_pct:
                            conc_warns.append(
                                f"⚠ {sect} sector → {proj_sect:.0f}% (limit {sector_limit_pct:.0f}%)"
                            )
                        if conc_warns:
                            add_reason += "  " + "; ".join(conc_warns)
                    items.append(
                        ReconciliationItem(
                            ticker=ticker,
                            action="ADD",
                            reason=add_reason,
                            urgency="LOW",
                            ibkr_position=pos,
                            analysis=analysis,
                            suggested_quantity=add_qty,
                            # Prefer live IBKR price for ADD (we already own it)
                            suggested_price=pos.current_price_local
                            or analysis.entry_price,
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
            status_parts.append(f"target {analysis.target_1_price:.2f}")

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

    # ── Phase 1.5: Watchlist tickers not currently held ──
    # Evaluates each watchlist ticker that isn't an open position.
    # After processing, all watchlist tickers are added to held_tickers so that
    # Phase 2 doesn't surface them a second time as plain BUY recommendations.
    watchlist_set = (watchlist_tickers or set()) - held_tickers
    for ticker in sorted(watchlist_set):
        analysis = analyses.get(ticker)

        if analysis is None:
            items.append(
                ReconciliationItem(
                    ticker=ticker,
                    action="REVIEW",
                    reason="Watchlist: no analysis found",
                    urgency="MEDIUM",
                    is_watchlist=True,
                )
            )
            continue

        is_stale, stale_reason = check_staleness(
            analysis, None, max_age_days, drift_threshold_pct
        )
        if is_stale:
            items.append(
                ReconciliationItem(
                    ticker=ticker,
                    action="REVIEW",
                    reason=f"Watchlist: stale analysis ({stale_reason})",
                    urgency="MEDIUM",
                    analysis=analysis,
                    is_watchlist=True,
                )
            )
            continue

        verdict_upper = (analysis.verdict or "").upper()

        if verdict_upper in ("DO_NOT_INITIATE", "SELL", "REJECT"):
            items.append(
                ReconciliationItem(
                    ticker=ticker,
                    action="REMOVE",
                    reason=f"Remove from watchlist — verdict → {analysis.verdict}  ({analysis.analysis_date})",
                    urgency="MEDIUM",
                    analysis=analysis,
                    is_watchlist=True,
                )
            )

        elif verdict_upper == "BUY":
            has_portfolio = portfolio.portfolio_value_usd > 0
            if has_portfolio and remaining_cash <= 0:
                # No cash — still surface the BUY as informational
                items.append(
                    ReconciliationItem(
                        ticker=ticker,
                        action="BUY",
                        reason=f"Watchlist BUY ({analysis.analysis_date}) — no cash available",
                        urgency="MEDIUM",
                        analysis=analysis,
                        is_watchlist=True,
                    )
                )
            else:
                entry_price = analysis.entry_price or analysis.current_price
                conviction = (
                    analysis.conviction or analysis.trade_block.conviction or ""
                )
                size_pct = analysis.trade_block.size_pct or (
                    analysis.position_size or 0
                )
                buy_qty = calculate_quantity(
                    available_cash_usd=remaining_cash,
                    entry_price_local=entry_price or 0.0,
                    fx_rate_to_usd=analysis.fx_rate_to_usd,
                    size_pct=size_pct,
                    portfolio_value_usd=portfolio.portfolio_value_usd,
                    yf_ticker=ticker,
                )
                buy_cost_usd = (
                    buy_qty * (entry_price or 0.0) * (analysis.fx_rate_to_usd or 1.0)
                )
                if buy_qty > 0 and buy_cost_usd < _MIN_ORDER_USD:
                    # Too small to place; skip (same rule as Phase 2)
                    pass
                else:
                    remaining_cash -= buy_cost_usd
                    buy_reason = f"Watchlist BUY ({analysis.analysis_date}) — {conviction} conviction, target {size_pct:.1f}%"
                    if portfolio.portfolio_value_usd > 0:
                        exch = analysis.exchange or _exchange_from_ticker(ticker)
                        sect = analysis.sector or "Unknown"
                        proj_exch = (
                            _exchange_weights.get(exch, 0.0)
                            + buy_cost_usd / portfolio.portfolio_value_usd * 100
                        )
                        proj_sect = (
                            _sector_weights.get(sect, 0.0)
                            + buy_cost_usd / portfolio.portfolio_value_usd * 100
                        )
                        conc_warns = []
                        if proj_exch > exchange_limit_pct:
                            conc_warns.append(
                                f"⚠ {exch} → {proj_exch:.0f}% (limit {exchange_limit_pct:.0f}%)"
                            )
                        if proj_sect > sector_limit_pct:
                            conc_warns.append(
                                f"⚠ {sect} sector → {proj_sect:.0f}% (limit {sector_limit_pct:.0f}%)"
                            )
                        if conc_warns:
                            buy_reason += "  " + "; ".join(conc_warns)
                    items.append(
                        ReconciliationItem(
                            ticker=ticker,
                            action="BUY",
                            reason=buy_reason,
                            urgency="MEDIUM",
                            analysis=analysis,
                            suggested_quantity=buy_qty if buy_qty > 0 else None,
                            suggested_price=entry_price,
                            suggested_order_type="LMT",
                            cash_impact_usd=-buy_cost_usd if buy_cost_usd > 0 else 0.0,
                            is_watchlist=True,
                        )
                    )

        else:
            # HOLD or any other verdict: keep watching, no position action
            items.append(
                ReconciliationItem(
                    ticker=ticker,
                    action="HOLD",
                    reason=f"Watchlist: monitoring — verdict {analysis.verdict}  ({analysis.analysis_date})",
                    urgency="LOW",
                    analysis=analysis,
                    is_watchlist=True,
                )
            )

    # Prevent Phase 2 from re-processing watchlist tickers
    held_tickers.update(watchlist_set)

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

        # Skip only when we CAN calculate a quantity but the cost is trivially small.
        # If buy_qty=0 due to lot-size rounding (e.g. can't afford a full lot yet),
        # still surface the BUY as an informational recommendation (quantity=None).
        if buy_qty > 0 and buy_cost_usd < _MIN_ORDER_USD:
            continue

        remaining_cash -= buy_cost_usd

        # Check concentration limits
        buy_reason = f"New BUY ({analysis.analysis_date}) — {conviction} conviction, target {size_pct:.1f}%"
        if portfolio.portfolio_value_usd > 0:
            exch = analysis.exchange or _exchange_from_ticker(ticker)
            sect = analysis.sector or "Unknown"
            proj_exch = (
                _exchange_weights.get(exch, 0.0)
                + buy_cost_usd / portfolio.portfolio_value_usd * 100
            )
            proj_sect = (
                _sector_weights.get(sect, 0.0)
                + buy_cost_usd / portfolio.portfolio_value_usd * 100
            )
            conc_warns = []
            if proj_exch > exchange_limit_pct:
                conc_warns.append(
                    f"⚠ {exch} → {proj_exch:.0f}% (limit {exchange_limit_pct:.0f}%)"
                )
            if proj_sect > sector_limit_pct:
                conc_warns.append(
                    f"⚠ {sect} sector → {proj_sect:.0f}% (limit {sector_limit_pct:.0f}%)"
                )
            if conc_warns:
                buy_reason += "  " + "; ".join(conc_warns)

        items.append(
            ReconciliationItem(
                ticker=ticker,
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
        removes=sum(1 for i in items if i.action == "REMOVE"),
    )

    return items


# ══════════════════════════════════════════════════════════════════════════════
# Portfolio-Level Health Check
# ══════════════════════════════════════════════════════════════════════════════


def compute_portfolio_health(
    positions: list[NormalizedPosition],
    analyses: dict[str, AnalysisRecord],
    portfolio: PortfolioSummary,
    max_age_days: int = 14,
    reconciliation_items: list | None = None,
) -> list[str]:
    """
    Compute portfolio-level health flags using data already in held analyses.

    Returns a list of human-readable flag strings. Empty list = no flags.
    Call AFTER reconcile() so that portfolio.exchange_weights is already set.

    When reconciliation_items is provided, also detects CORRELATED_SELL_EVENT
    and demotes SOFT_REJECT SELLs to REVIEW in-place on correlated days.

    Flags:
        LOW_HEALTH_AVERAGE      Weighted avg health_adj < 60 across holdings
        LOW_GROWTH_AVERAGE      Weighted avg growth_adj < 55 across holdings
        CURRENCY_CONCENTRATION  >50% of portfolio in a single currency
        STALE_ANALYSIS_RATIO    >30% of positions have analyses older than max_age_days
        CORRELATED_SELL_EVENT   ≥5 and ≥25% of held positions flipped verdict same date
    """
    if not positions or portfolio.portfolio_value_usd <= 0:
        return []

    flags: list[str] = []
    total_weight = 0.0
    weighted_health = 0.0
    weighted_growth = 0.0
    health_count = 0
    growth_count = 0
    stale_count = 0
    currency_weights: dict[str, float] = {}
    # Per-position scores for detail lines: (ticker, score, is_stale)
    scored_health: list[tuple[str, float, bool]] = []
    scored_growth: list[tuple[str, float, bool]] = []

    for pos in positions:
        weight = pos.market_value_usd / portfolio.portfolio_value_usd
        total_weight += weight

        analysis = analyses.get(pos.yf_ticker)
        is_stale = analysis is not None and analysis.age_days > max_age_days
        if analysis:
            if analysis.health_adj is not None:
                weighted_health += analysis.health_adj * weight
                health_count += 1
                scored_health.append((pos.yf_ticker, analysis.health_adj, is_stale))
            if analysis.growth_adj is not None:
                weighted_growth += analysis.growth_adj * weight
                growth_count += 1
                scored_growth.append((pos.yf_ticker, analysis.growth_adj, is_stale))
            if is_stale:
                stale_count += 1

        ccy = (pos.currency or "USD").upper()
        currency_weights[ccy] = currency_weights.get(ccy, 0.0) + weight * 100

    def _worst_detail(
        scored: list[tuple[str, float, bool]],
        max_age_days: int,
        n: int = 5,
    ) -> str:
        """Build a detail sub-line showing the N lowest-scoring positions."""
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
            flags.append(
                f"STALE_ANALYSIS_RATIO: {stale_count}/{len(positions)} positions"
                f" ({stale_pct:.0f}%) have analyses older than {max_age_days}d"
                " — flying blind on significant chunk of portfolio"
                " (re-run with --refresh-stale to update)"
            )

    # Geography concentration is already surfaced via portfolio.exchange_weights;
    # flag here only if it exceeds the standard 40% exchange limit.
    for exch, pct in portfolio.exchange_weights.items():
        if pct > 40:
            long_name = _EXCHANGE_LONG_NAMES.get(exch, exch)
            flags.append(
                f"GEOGRAPHY_CONCENTRATION: {pct:.1f}% in {exch} ({long_name})"
                " — single-exchange concentration"
            )

    # ── CORRELATED_SELL_EVENT ─────────────────────────────────────────────────
    # Detect when many held positions flip verdict on the same date — a sign of
    # macro noise rather than company-specific thesis failures.  Only verdict-
    # driven SELLs (HARD_REJECT + SOFT_REJECT) are counted; mechanical stop
    # breaches are excluded.
    if reconciliation_items is not None:
        from collections import Counter

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
            date_counts = Counter(
                item.analysis.analysis_date
                for item in verdict_sells
                if item.analysis and item.analysis.analysis_date
            )
            if date_counts:
                peak_date, peak_count = date_counts.most_common(1)[0]
                correlated_event = (
                    peak_count >= 5
                    and total_held > 0
                    and peak_count / total_held >= 0.25
                )
                if correlated_event:
                    flags.append(
                        f"CORRELATED_SELL_EVENT: {peak_count} positions changed verdict"
                        f" on {peak_date} ({peak_count / total_held:.0%} of held"
                        f" positions) — probable macro event. Execute stop-breach SELLs"
                        f" only; review verdict-change SELLs before acting."
                    )
                    # Demote SOFT_REJECT from SELL to REVIEW — these passed all hard
                    # fundamental checks; the SELL verdict came from the soft-tally
                    # (geopolitical/macro points), not from a thesis failure.
                    for item in reconciliation_items:
                        if item.action == "SELL" and item.sell_type == "SOFT_REJECT":
                            item.action = "REVIEW"
                            item.urgency = "MEDIUM"
                            item.reason += (
                                "  [MACRO_WATCH: demoted from SELL — correlated"
                                " event detected]"
                            )
                    logger.info(
                        "correlated_sell_event_detected",
                        peak_date=peak_date,
                        peak_count=peak_count,
                        total_held=total_held,
                        pct=f"{peak_count / total_held:.0%}",
                    )

    return flags
