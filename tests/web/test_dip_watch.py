from __future__ import annotations

from src.ibkr.dip_watch import (
    DIP_CONCENTRATION_MIN_SCORE,
    build_dip_watch_candidates,
    collect_dip_watch_source_items,
    compute_dip_score,
    dip_pct,
    dip_watch_source,
    is_dip_watch_eligible,
    macro_regime_price_multiplier,
    risk_reward_ratio,
    score_dip_watch_item,
    screen_dip_candidates_by_concentration,
    select_dip_watch_candidates,
)
from src.ibkr.models import ReconciliationItem
from tests.factories.ibkr import make_analysis, make_position


def _dip_item(
    *,
    ticker: str = "7203.T",
    verdict: str = "BUY",
    zone: str = "MODERATE",
    age_days: int = 5,
    health_adj: float | None = 75.0,
    growth_adj: float | None = 72.0,
    entry_price: float | None = 2100.0,
    current_price: float = 1800.0,
    stop_price: float | None = 1700.0,
    target_1: float | None = 2600.0,
    action: str = "REVIEW",
    sell_type: str | None = "SOFT_REJECT",
    held: bool = True,
) -> ReconciliationItem:
    analysis = make_analysis(
        ticker=ticker,
        verdict=verdict,
        zone=zone,
        age_days=age_days,
        health_adj=health_adj,
        growth_adj=growth_adj,
        entry_price=entry_price or 0.0,
        current_price=current_price,
        stop_price=stop_price or 0.0,
        target_1=target_1 or 0.0,
    )
    analysis.entry_price = entry_price
    analysis.stop_price = stop_price
    analysis.target_1_price = target_1
    return ReconciliationItem(
        ticker=ticker,
        action=action,
        urgency="MEDIUM",
        reason="Macro review",
        ibkr_position=(
            make_position(ticker=ticker, current_price=current_price) if held else None
        ),
        analysis=analysis,
        sell_type=sell_type,
    )


def test_compute_dip_score_returns_zero_without_analysis():
    item = ReconciliationItem(
        ticker="7203.T",
        action="REVIEW",
        urgency="MEDIUM",
        reason="No analysis",
        ibkr_position=make_position(),
    )
    assert compute_dip_score(item) == 0.0
    assert score_dip_watch_item(item) == 0.0


def test_select_dip_watch_candidates_filters_and_ranks(sample_bundle):
    review_items = [item for item in sample_bundle.items if item.action == "REVIEW"]
    selected = select_dip_watch_candidates(review_items)
    assert selected == []


def test_build_dip_watch_candidates_includes_star_thresholds(sample_bundle):
    rows = build_dip_watch_candidates([_dip_item()])
    assert rows[0].ticker_yf == "7203.T"
    assert rows[0].stars in {"★★★", "★★", "★"}
    assert rows[0].run_ticker == "7203.T"
    assert rows[0].source == "macro_review"
    assert rows[0].dip_pct == 14.3


def test_risk_reward_ratio_returns_none_when_stop_or_target_missing():
    analysis = make_analysis()
    analysis.target_1_price = None
    item = ReconciliationItem(
        ticker="7203.T",
        action="REVIEW",
        urgency="MEDIUM",
        reason="Missing target",
        ibkr_position=make_position(),
        analysis=analysis,
    )
    assert risk_reward_ratio(item) is None


def test_compute_dip_score_prefers_better_dip():
    strong = ReconciliationItem(
        ticker="7203.T",
        action="REVIEW",
        urgency="MEDIUM",
        reason="Better dip",
        ibkr_position=make_position(current_price=1800),
        analysis=make_analysis(entry_price=2100, current_price=1800),
    )
    weak = ReconciliationItem(
        ticker="7203.T",
        action="REVIEW",
        urgency="MEDIUM",
        reason="At entry",
        ibkr_position=make_position(current_price=2100),
        analysis=make_analysis(entry_price=2100, current_price=2100),
    )
    assert compute_dip_score(strong) > compute_dip_score(weak)


def test_compute_dip_score_regime_multiplier_changes_only_price_bonus():
    item = _dip_item(
        health_adj=75.0,
        growth_adj=75.0,
        entry_price=2000.0,
        current_price=1800.0,
        stop_price=None,
        target_1=None,
    )

    baseline = compute_dip_score(item)
    suppressed = compute_dip_score(item, regime_multiplier=0.3)

    assert baseline == 75.0 * 0.4 + 75.0 * 0.4 + 12.0
    assert suppressed == 75.0 * 0.4 + 75.0 * 0.4 + (12.0 * 0.3)


def test_macro_regime_multiplier_defaults_to_one_when_absent_or_low_confidence():
    absent = _dip_item()
    low = _dip_item()
    low.analysis.macro_regime = {
        "present": True,
        "confidence": "LOW",
        "dip_posture": "AVOID",
    }

    assert macro_regime_price_multiplier(absent) == 1.0
    assert macro_regime_price_multiplier(low) == 1.0


def test_partial_block_default_dip_posture_yields_sixty_percent_multiplier():
    item = _dip_item()
    item.analysis.macro_regime = {
        "present": True,
        "confidence": "HIGH",
        "dip_posture": "WAIT_FOR_CONFIRMATION",
    }

    assert macro_regime_price_multiplier(item) == 0.60


def test_select_dip_watch_candidates_ranking_uses_regime_adjusted_score():
    buyable = _dip_item(ticker="BUYABLE.T", health_adj=70.0, growth_adj=70.0)
    buyable.analysis.macro_regime = {
        "present": True,
        "confidence": "HIGH",
        "dip_posture": "BUYABLE",
    }
    waiting = _dip_item(ticker="WAIT.T", health_adj=71.0, growth_adj=71.0)
    waiting.analysis.macro_regime = {
        "present": True,
        "confidence": "HIGH",
        "dip_posture": "WAIT_FOR_CONFIRMATION",
    }

    selected = select_dip_watch_candidates([waiting, buyable])

    assert [item.ticker.yf for item in selected] == ["BUYABLE.T", "WAIT.T"]


def test_select_dip_watch_candidates_includes_fresh_buy():
    selected = select_dip_watch_candidates([_dip_item()])
    assert [item.ticker.yf for item in selected] == ["7203.T"]


def test_source_collector_includes_held_buy_pullback_and_macro_review():
    held_buy = _dip_item(ticker="HELD.T", action="HOLD", sell_type=None)
    macro_review = _dip_item(ticker="MACRO.T")
    rejected_macro = _dip_item(ticker="DNI.T", verdict="DO_NOT_INITIATE")
    stop_review = _dip_item(ticker="STOP.T", action="REVIEW", sell_type="STOP_BREACH")
    profit_review = _dip_item(
        ticker="PROFIT.T",
        action="REVIEW",
        sell_type="PROFIT_TAKE",
    )
    plain_review = _dip_item(ticker="PLAIN.T", action="REVIEW", sell_type=None)
    sell = _dip_item(ticker="SELL.T", action="SELL", sell_type=None)
    trim = _dip_item(ticker="TRIM.T", action="TRIM", sell_type=None)
    unheld = _dip_item(ticker="UNHELD.T", action="HOLD", sell_type=None, held=False)

    source_items = collect_dip_watch_source_items(
        [
            held_buy,
            macro_review,
            rejected_macro,
            stop_review,
            profit_review,
            plain_review,
            sell,
            trim,
            unheld,
        ]
    )

    assert [item.ticker.yf for item in source_items] == ["HELD.T", "MACRO.T", "DNI.T"]
    assert dip_watch_source(held_buy) == "held_buy_pullback"
    assert dip_watch_source(macro_review) == "macro_review"
    assert dip_watch_source(stop_review) is None


def test_held_buy_pullback_must_clear_eligibility_gate():
    held_buy = _dip_item(action="HOLD", sell_type=None)
    rejected_macro = _dip_item(verdict="DO_NOT_INITIATE")

    selected = select_dip_watch_candidates(
        collect_dip_watch_source_items([held_buy, rejected_macro])
    )

    assert [item.ticker.yf for item in selected] == ["7203.T"]


def test_select_dip_watch_candidates_filters_and_limits_after_eligibility():
    low_score = _dip_item(
        ticker="LOW.T",
        health_adj=58.0,
        growth_adj=56.0,
        entry_price=2000.0,
        current_price=1980.0,
        stop_price=1900.0,
        target_1=2100.0,
    )
    high_score = _dip_item(
        ticker="HIGH.T",
        health_adj=80.0,
        growth_adj=78.0,
        entry_price=2000.0,
        current_price=1800.0,
        stop_price=1700.0,
        target_1=2600.0,
    )
    rejected = _dip_item(ticker="DNI.T", verdict="DO_NOT_INITIATE")

    selected = select_dip_watch_candidates(
        [low_score, high_score, rejected],
        limit=1,
    )

    assert [item.ticker.yf for item in selected] == ["HIGH.T"]


def test_dip_watch_excludes_do_not_initiate_verdict():
    item = _dip_item(verdict="DO_NOT_INITIATE", zone="HIGH")
    assert is_dip_watch_eligible(item) is False
    assert select_dip_watch_candidates([item]) == []


def test_dip_watch_excludes_non_buy_verdicts():
    for verdict in ("HOLD", "SELL", "REJECT", "", "DO NOT INITIATE"):
        assert is_dip_watch_eligible(_dip_item(verdict=verdict)) is False


def test_dip_watch_excludes_high_zone_even_when_buy():
    assert is_dip_watch_eligible(_dip_item(verdict="BUY", zone="HIGH")) is False


def test_dip_watch_freshness_boundary_is_30_days():
    assert is_dip_watch_eligible(_dip_item(age_days=30)) is True
    assert is_dip_watch_eligible(_dip_item(age_days=31)) is False


def test_dip_watch_requires_minimum_five_percent_dip():
    shallow = _dip_item(entry_price=100.0, current_price=96.0)
    boundary = _dip_item(entry_price=100.0, current_price=95.0)

    assert dip_pct(shallow) == 4.0
    assert is_dip_watch_eligible(shallow) is False
    assert dip_pct(boundary) == 5.0
    assert is_dip_watch_eligible(boundary) is True


def test_dip_watch_excludes_invalid_or_missing_analysis_date():
    invalid = _dip_item()
    invalid.analysis.analysis_date = "not-a-date"
    missing = _dip_item()
    missing.analysis.analysis_date = ""

    assert is_dip_watch_eligible(invalid) is False
    assert is_dip_watch_eligible(missing) is False


def test_dip_watch_missing_inputs_do_not_raise():
    item = _dip_item(
        health_adj=None,
        growth_adj=None,
        entry_price=None,
        stop_price=None,
        target_1=None,
    )
    assert is_dip_watch_eligible(item) is False
    assert compute_dip_score(item) == 0.0


# ── Concentration screen (overweight bucket ⇒ ★★★-only dips) ─────────────────
# _dip_item fixtures: dip ≈14.3% → price bonus capped at 12; R/R capped at 8.
# health/growth 75/72 → score ≈78.8 (★★★); 60/60 → 68 (sub-★★★, still eligible).


def _held_buy_dip(ticker: str = "7203.T", *, health: float, growth: float):
    return _dip_item(
        ticker=ticker,
        health_adj=health,
        growth_adj=growth,
        action="HOLD",
        sell_type=None,
    )


def test_screen_withholds_sub_star3_dip_in_overweight_exchange():
    item = _held_buy_dip(health=60, growth=60)
    kept, withheld = screen_dip_candidates_by_concentration(
        [item], exchange_weights={"T": 45.0}
    )
    assert kept == []
    assert withheld == [item]
    assert score_dip_watch_item(item) < DIP_CONCENTRATION_MIN_SCORE


def test_screen_keeps_star3_dip_in_overweight_exchange():
    item = _held_buy_dip(health=75, growth=72)
    kept, withheld = screen_dip_candidates_by_concentration(
        [item], exchange_weights={"T": 45.0}
    )
    assert kept == [item]
    assert withheld == []
    assert score_dip_watch_item(item) >= DIP_CONCENTRATION_MIN_SCORE


def test_select_dip_watch_returns_kept_and_withheld_in_one_pass():
    from src.ibkr.dip_watch import DipWatchSelection, select_dip_watch

    keep = _held_buy_dip(ticker="7203.T", health=75, growth=72)  # ★★★
    withhold = _held_buy_dip(ticker="6758.T", health=60, growth=60)  # sub-★★★

    selection = select_dip_watch([keep, withhold], exchange_weights={"T": 45.0})

    assert isinstance(selection, DipWatchSelection)
    assert list(selection.kept) == [keep]
    assert list(selection.withheld) == [withhold]
    # The thin list wrapper agrees with the kept slice.
    assert select_dip_watch_candidates(
        [keep, withhold], exchange_weights={"T": 45.0}
    ) == [keep]


def test_group_portfolio_actions_exposes_dip_withheld():
    from src.ibkr.portfolio_presentation import group_portfolio_actions

    keep = _held_buy_dip(ticker="7203.T", health=75, growth=72)
    withhold = _held_buy_dip(ticker="6758.T", health=60, growth=60)

    groups = group_portfolio_actions([keep, withhold], exchange_weights={"T": 45.0})

    assert list(groups.dip_candidates) == [keep]
    assert list(groups.dip_withheld) == [withhold]


def test_screen_inactive_without_weights():
    item = _held_buy_dip(health=60, growth=60)
    for weights in (None, {}):
        kept, withheld = screen_dip_candidates_by_concentration(
            [item], exchange_weights=weights, sector_weights=weights
        )
        assert kept == [item]
        assert withheld == []


def test_bucket_exactly_at_limit_is_not_overweight():
    item = _held_buy_dip(health=60, growth=60)
    kept, withheld = screen_dip_candidates_by_concentration(
        [item], exchange_weights={"T": 40.0}
    )
    assert kept == [item]
    assert withheld == []


def test_sector_only_overweight_screens():
    item = _held_buy_dip(health=60, growth=60)
    item.analysis.sector = "Industrials"
    kept, withheld = screen_dip_candidates_by_concentration(
        [item],
        exchange_weights={"T": 10.0},
        sector_weights={"Industrials": 35.0},
    )
    assert withheld == [item]


def test_unknown_sector_skips_sector_dimension():
    item = _held_buy_dip(health=60, growth=60)  # analysis.sector unset
    kept, withheld = screen_dip_candidates_by_concentration(
        [item],
        sector_weights={"Industrials": 45.0},
    )
    assert kept == [item]


def test_select_applies_screen_before_limit_and_backfills():
    """The withheld top scorer's slot goes to the next under-limit name —
    the display limit is still filled instead of shipping short."""
    top_but_overweight = _held_buy_dip("7203.T", health=60, growth=60)
    second = _held_buy_dip("0005.HK", health=58, growth=58)
    third = _held_buy_dip("0700.HK", health=56, growth=56)

    selected = select_dip_watch_candidates(
        [top_but_overweight, second, third],
        limit=2,
        exchange_weights={"T": 45.0},
    )

    assert [item.ticker.yf for item in selected] == ["0005.HK", "0700.HK"]


def test_macro_event_does_not_waive_screen():
    """A macro event widens dip candidacy but never the concentration bar."""
    macro_dip = _dip_item(
        verdict="DO_NOT_INITIATE",
        zone="HIGH",
        health_adj=60,
        growth_adj=60,
    )  # REVIEW + SOFT_REJECT: macro_review source, eligible only during events

    without_weights = select_dip_watch_candidates([macro_dip], macro_event_active=True)
    with_weights = select_dip_watch_candidates(
        [macro_dip],
        macro_event_active=True,
        exchange_weights={"T": 45.0},
    )

    assert without_weights == [macro_dip]
    assert with_weights == []


def test_screen_without_analysis_withholds_conservatively_without_crash():
    item = ReconciliationItem(
        ticker="7203.T",
        action="REVIEW",
        urgency="MEDIUM",
        reason="No analysis",
        ibkr_position=make_position(),
    )
    kept, withheld = screen_dip_candidates_by_concentration(
        [item], exchange_weights={"T": 45.0}
    )
    assert withheld == [item]


# ── Intact-thesis drawdown promotion (July 2026) ──────────────────────────────


def _thesis_dip_item(
    *,
    action_basis: str = "THESIS_REASSESSMENT",
    verdict: str = "DO_NOT_INITIATE",
    current_price: float = 1680.0,  # -20% vs entry 2100
    health_adj: float | None = 75.0,
    growth_adj: float | None = 72.0,
    age_days: int = 5,
) -> ReconciliationItem:
    item = _dip_item(
        verdict=verdict,
        current_price=current_price,
        health_adj=health_adj,
        growth_adj=growth_adj,
        age_days=age_days,
        action="REVIEW",
        sell_type="SOFT_REJECT",
    )
    item.action_basis = action_basis
    return item


def test_thesis_reassessment_review_is_thesis_dip_source():
    item = _thesis_dip_item()
    assert dip_watch_source(item) == "held_thesis_dip"


def test_entry_constraint_review_is_thesis_dip_source():
    item = _thesis_dip_item(action_basis="ENTRY_CONSTRAINT")
    assert dip_watch_source(item) == "held_thesis_dip"


def test_intact_thesis_deep_dip_is_eligible_without_macro_event():
    """The dead-money fix: a -20% intact-score reject reaches DIP WATCH with
    no macro event required and no BUY verdict required."""
    item = _thesis_dip_item()
    assert is_dip_watch_eligible(item, macro_event_active=False)


def test_intact_thesis_shallow_dip_below_15pct_not_promoted():
    item = _thesis_dip_item(current_price=1953.0)  # -7% vs entry
    assert not is_dip_watch_eligible(item, macro_event_active=False)


def test_weak_score_thesis_reassessment_not_promoted():
    """The weak-score unconfirmed reject also carries THESIS_REASSESSMENT —
    the 55/55 eligibility floor keeps it out of the dip queue."""
    item = _thesis_dip_item(health_adj=48.0, growth_adj=45.0)
    assert not is_dip_watch_eligible(item, macro_event_active=False)


def test_stale_thesis_dip_not_promoted():
    item = _thesis_dip_item(age_days=45)
    assert not is_dip_watch_eligible(item, macro_event_active=False)


def test_compute_dip_score_is_stop_independent():
    """Stop distance must not influence the dip score (July 2026)."""
    near_stop = _dip_item(stop_price=1750.0)
    far_stop = _dip_item(stop_price=900.0)
    assert compute_dip_score(near_stop) == compute_dip_score(far_stop)
