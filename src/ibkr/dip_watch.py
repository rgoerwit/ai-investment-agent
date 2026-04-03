from __future__ import annotations

from dataclasses import dataclass

from src.ibkr.models import ReconciliationItem
from src.ibkr.refresh_service import run_ticker_for


@dataclass(frozen=True)
class DipWatchCandidate:
    ticker_yf: str
    ticker_ibkr: str
    score: float
    stars: str
    dip_pct: float
    risk_reward: float | None
    held_quantity: float
    health_adj: float | None
    growth_adj: float | None
    entry_price: float | None
    current_price: float | None
    currency: str
    run_ticker: str


def compute_dip_score(item: ReconciliationItem) -> float:
    """Return the current CLI dip-watch score for one reconciliation item."""
    analysis = item.analysis
    position = item.ibkr_position
    if analysis is None:
        return 0.0

    health = analysis.health_adj or 0.0
    growth = analysis.growth_adj or 0.0
    base = health * 0.4 + growth * 0.4

    price_bonus = 0.0
    if analysis.entry_price and position and position.current_price_local:
        if analysis.entry_price > 0:
            dip_pct = (
                (analysis.entry_price - position.current_price_local)
                / analysis.entry_price
                * 100
            )
            if dip_pct > 0:
                price_bonus = min(dip_pct * 1.5, 12.0)

    rr_bonus = 0.0
    if analysis.target_1_price and analysis.stop_price and position:
        current = position.current_price_local
        if current > 0 and current > analysis.stop_price:
            upside = (analysis.target_1_price - current) / current
            downside = max((current - analysis.stop_price) / current, 0.001)
            rr_bonus = min((upside / downside) * 2.5, 8.0)

    return base + price_bonus + rr_bonus


def risk_reward_ratio(item: ReconciliationItem) -> float | None:
    """Return upside/downside from the current price, or None when unavailable."""
    analysis = item.analysis
    position = item.ibkr_position
    if analysis is None or position is None:
        return None
    current = position.current_price_local
    if (
        analysis.target_1_price is None
        or analysis.stop_price is None
        or current <= 0
        or current <= analysis.stop_price
    ):
        return None
    upside = (analysis.target_1_price - current) / current
    downside = max((current - analysis.stop_price) / current, 0.001)
    return round(upside / downside, 1)


def select_dip_watch_candidates(
    items: list[ReconciliationItem],
    *,
    min_health: float = 55.0,
    min_growth: float = 55.0,
    min_score: float = 50.0,
    limit: int | None = None,
) -> list[ReconciliationItem]:
    """Return items eligible for DIP WATCH using the current CLI rules."""
    ranked = [
        item
        for item in items
        if item.analysis is not None
        and (item.analysis.health_adj or 0.0) >= min_health
        and (item.analysis.growth_adj or 0.0) >= min_growth
        and compute_dip_score(item) >= min_score
    ]
    ranked.sort(key=compute_dip_score, reverse=True)
    if limit is not None:
        return ranked[:limit]
    return ranked


def build_dip_watch_candidates(
    items: list[ReconciliationItem],
    *,
    min_health: float = 55.0,
    min_growth: float = 55.0,
    min_score: float = 50.0,
    limit: int | None = None,
) -> list[DipWatchCandidate]:
    """Return serializable dip-watch candidates derived from reconciliation items."""
    candidates = select_dip_watch_candidates(
        items,
        min_health=min_health,
        min_growth=min_growth,
        min_score=min_score,
        limit=limit,
    )
    rows: list[DipWatchCandidate] = []
    for item in candidates:
        analysis = item.analysis
        position = item.ibkr_position
        if analysis is None or position is None:
            continue
        entry_price = analysis.entry_price
        current_price = position.current_price_local
        dip_pct = 0.0
        if entry_price and entry_price > 0 and current_price > 0:
            dip_pct = (entry_price - current_price) / entry_price * 100
        score = compute_dip_score(item)
        rows.append(
            DipWatchCandidate(
                ticker_yf=item.ticker.yf,
                ticker_ibkr=item.ticker.ibkr,
                score=round(score, 1),
                stars="★★★" if score >= 75 else ("★★" if score >= 60 else "★"),
                dip_pct=round(dip_pct, 1),
                risk_reward=risk_reward_ratio(item),
                held_quantity=position.quantity,
                health_adj=analysis.health_adj,
                growth_adj=analysis.growth_adj,
                entry_price=analysis.entry_price,
                current_price=current_price,
                currency=position.currency,
                run_ticker=run_ticker_for(item),
            )
        )
    return rows
