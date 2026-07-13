from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.ibkr.models import ReconciliationItem
from src.ibkr.reconciliation_rules import _normalize_verdict, _normalize_zone
from src.ibkr.refresh_service import run_ticker_for

_DEFAULT_DIP_WATCH_MAX_AGE_DAYS = 30
_DEFAULT_DIP_WATCH_EXCLUDED_ZONES = frozenset({"HIGH"})
_DEFAULT_DIP_WATCH_MIN_DIP_PCT = 5.0
_DIP_POSTURE_PRICE_MULTIPLIERS = {
    "BUYABLE": 1.0,
    "SCALE_SLOWLY": 0.85,
    "WAIT_FOR_CONFIRMATION": 0.60,
    "AVOID": 0.30,
}
DipWatchSource = Literal["held_buy_pullback", "macro_review"]


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
    source: DipWatchSource


def dip_pct(item: ReconciliationItem) -> float:
    """Return percent pullback from analysis entry to current position price."""
    analysis = item.analysis
    position = item.ibkr_position
    if (
        analysis is None
        or position is None
        or analysis.entry_price is None
        or analysis.entry_price <= 0
        or position.current_price_local <= 0
    ):
        return 0.0
    return (
        (analysis.entry_price - position.current_price_local)
        / analysis.entry_price
        * 100
    )


def dip_watch_source(item: ReconciliationItem) -> DipWatchSource | None:
    """Return the dip-watch source class for structurally eligible held items."""
    if item.ibkr_position is None:
        return None
    if (
        item.action == "HOLD"
        and _normalize_verdict(item.analysis.verdict if item.analysis else "") == "BUY"
    ):
        return "held_buy_pullback"
    if item.action == "REVIEW" and item.sell_type == "SOFT_REJECT":
        return "macro_review"
    return None


def collect_dip_watch_source_items(
    items: list[ReconciliationItem],
) -> list[ReconciliationItem]:
    """Return held items that may be evaluated for dip-watch eligibility."""
    return [item for item in items if dip_watch_source(item) is not None]


def macro_regime_price_multiplier(item: ReconciliationItem) -> float:
    """Return the regime multiplier for the price-pullback portion of DIP WATCH."""
    analysis = getattr(item, "analysis", None)
    if analysis is None:
        return 1.0
    regime = getattr(analysis, "macro_regime", None) or {}
    if not regime.get("present") or regime.get("confidence") == "LOW":
        return 1.0
    return _DIP_POSTURE_PRICE_MULTIPLIERS.get(regime.get("dip_posture"), 1.0)


def compute_dip_score(
    item: ReconciliationItem,
    *,
    regime_multiplier: float = 1.0,
) -> float:
    """Return the current CLI dip-watch score for one reconciliation item."""
    analysis = item.analysis
    position = item.ibkr_position
    if analysis is None:
        return 0.0

    health = analysis.health_adj or 0.0
    growth = analysis.growth_adj or 0.0
    base = health * 0.4 + growth * 0.4

    price_bonus = 0.0
    current_dip_pct = dip_pct(item)
    if current_dip_pct > 0:
        price_bonus = min(current_dip_pct * 1.5, 12.0) * regime_multiplier

    rr_bonus = 0.0
    if analysis.target_1_price and analysis.stop_price and position:
        current = position.current_price_local
        if current > 0 and current > analysis.stop_price:
            upside = (analysis.target_1_price - current) / current
            downside = max((current - analysis.stop_price) / current, 0.001)
            rr_bonus = min((upside / downside) * 2.5, 8.0)

    return base + price_bonus + rr_bonus


def score_dip_watch_item(item: ReconciliationItem) -> float:
    """Return the DIP WATCH score after applying macro regime price discipline."""
    return compute_dip_score(
        item,
        regime_multiplier=macro_regime_price_multiplier(item),
    )


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


def is_dip_watch_eligible(
    item: ReconciliationItem,
    *,
    min_health: float = 55.0,
    min_growth: float = 55.0,
    min_score: float = 50.0,
    max_age_days: int = _DEFAULT_DIP_WATCH_MAX_AGE_DAYS,
    excluded_zones: frozenset[str] = _DEFAULT_DIP_WATCH_EXCLUDED_ZONES,
    min_dip_pct: float = _DEFAULT_DIP_WATCH_MIN_DIP_PCT,
    macro_event_active: bool = False,
) -> bool:
    """Return True when a held item is safe to surface as a dip-watch candidate.

    During an active macro event, a fundamentally-intact ``macro_review`` position
    that dipped is a dip-buy candidate even though the selloff pushed it to HIGH
    zone and its standalone verdict is a (usually valuation-driven) REJECT. The
    safeguards that DON'T relax: a **recent review** (``max_age_days``) and sound
    **health/growth** — a stale rating is not trusted, so a position only qualifies
    once its analysis has been refreshed.
    """
    analysis = item.analysis
    if analysis is None:
        return False
    source = dip_watch_source(item)
    if source is None:
        return False
    macro_dip = macro_event_active and source == "macro_review"
    if not macro_dip:
        if _normalize_verdict(analysis.verdict or "") != "BUY":
            return False
        if _normalize_zone(analysis.zone) in excluded_zones:
            return False
    if analysis.age_days > max_age_days:
        return False
    if (analysis.health_adj or 0.0) < min_health:
        return False
    if (analysis.growth_adj or 0.0) < min_growth:
        return False
    if dip_pct(item) < min_dip_pct:
        return False
    return score_dip_watch_item(item) >= min_score


def select_dip_watch_candidates(
    items: list[ReconciliationItem],
    *,
    min_health: float = 55.0,
    min_growth: float = 55.0,
    min_score: float = 50.0,
    max_age_days: int = _DEFAULT_DIP_WATCH_MAX_AGE_DAYS,
    excluded_zones: frozenset[str] = _DEFAULT_DIP_WATCH_EXCLUDED_ZONES,
    min_dip_pct: float = _DEFAULT_DIP_WATCH_MIN_DIP_PCT,
    macro_event_active: bool = False,
    limit: int | None = None,
) -> list[ReconciliationItem]:
    """Return items eligible for DIP WATCH using the current CLI rules."""
    ranked = [
        item
        for item in items
        if is_dip_watch_eligible(
            item,
            min_health=min_health,
            min_growth=min_growth,
            min_score=min_score,
            max_age_days=max_age_days,
            excluded_zones=excluded_zones,
            min_dip_pct=min_dip_pct,
            macro_event_active=macro_event_active,
        )
    ]
    ranked.sort(key=score_dip_watch_item, reverse=True)
    if limit is not None:
        return ranked[:limit]
    return ranked


def build_dip_watch_candidates(
    items: list[ReconciliationItem],
    *,
    min_health: float = 55.0,
    min_growth: float = 55.0,
    min_score: float = 50.0,
    max_age_days: int = _DEFAULT_DIP_WATCH_MAX_AGE_DAYS,
    excluded_zones: frozenset[str] = _DEFAULT_DIP_WATCH_EXCLUDED_ZONES,
    min_dip_pct: float = _DEFAULT_DIP_WATCH_MIN_DIP_PCT,
    limit: int | None = None,
) -> list[DipWatchCandidate]:
    """Return serializable dip-watch candidates derived from reconciliation items."""
    candidates = select_dip_watch_candidates(
        items,
        min_health=min_health,
        min_growth=min_growth,
        min_score=min_score,
        max_age_days=max_age_days,
        excluded_zones=excluded_zones,
        min_dip_pct=min_dip_pct,
        limit=limit,
    )
    rows: list[DipWatchCandidate] = []
    for item in candidates:
        analysis = item.analysis
        position = item.ibkr_position
        if analysis is None or position is None:
            continue
        source = dip_watch_source(item)
        if source is None:
            continue
        current_price = position.current_price_local
        score = score_dip_watch_item(item)
        rows.append(
            DipWatchCandidate(
                ticker_yf=item.ticker.yf,
                ticker_ibkr=item.ticker.ibkr,
                score=round(score, 1),
                stars="★★★" if score >= 75 else ("★★" if score >= 60 else "★"),
                dip_pct=round(dip_pct(item), 1),
                risk_reward=risk_reward_ratio(item),
                held_quantity=position.quantity,
                health_adj=analysis.health_adj,
                growth_adj=analysis.growth_adj,
                entry_price=analysis.entry_price,
                current_price=current_price,
                currency=position.currency,
                run_ticker=run_ticker_for(item),
                source=source,
            )
        )
    return rows
