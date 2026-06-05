"""Collected reconciliation-rules tests extracted from reconciler cases."""

import pytest

from src.ibkr.models import AnalysisRecord, NormalizedPosition
from src.ibkr.reconciliation_rules import (
    SCREEN_REVIEW_DNI_ZONES,
    _classify_sell_type,
    _cost_basis_return_pct,
    _normalize_zone,
)
from tests.ibkr.reconciler_cases import (
    TestCheckStaleness,
    TestCheckStopBreach,
    TestCheckTargetHit,
    TestExchangeFromPosition,
    TestResolveFx,
    TestStalenessDisplay,
)


def _make_position(avg_cost: float, current_price: float) -> NormalizedPosition:
    return NormalizedPosition(
        conid=12345,
        ticker="7203.T",
        quantity=100,
        avg_cost_local=avg_cost,
        current_price_local=current_price,
    )


def _make_analysis(health: float, growth: float) -> AnalysisRecord:
    return AnalysisRecord(
        ticker="7203.T",
        analysis_date="2026-01-01",
        health_adj=health,
        growth_adj=growth,
    )


class TestCostBasisReturnPct:
    def test_profit(self):
        pos = _make_position(avg_cost=100.0, current_price=150.0)
        assert _cost_basis_return_pct(pos) == pytest.approx(50.0)

    def test_loss(self):
        pos = _make_position(avg_cost=200.0, current_price=100.0)
        assert _cost_basis_return_pct(pos) == pytest.approx(-50.0)

    def test_breakeven(self):
        pos = _make_position(avg_cost=100.0, current_price=100.0)
        assert _cost_basis_return_pct(pos) == pytest.approx(0.0)

    def test_zero_avg_cost_returns_none(self):
        pos = _make_position(avg_cost=0.0, current_price=100.0)
        assert _cost_basis_return_pct(pos) is None

    def test_zero_current_price_returns_none(self):
        pos = _make_position(avg_cost=100.0, current_price=0.0)
        assert _cost_basis_return_pct(pos) is None

    def test_negative_avg_cost_returns_none(self):
        pos = _make_position(avg_cost=-50.0, current_price=100.0)
        assert _cost_basis_return_pct(pos) is None


class TestNormalizeZone:
    def test_none_returns_empty_string(self):
        assert _normalize_zone(None) == ""

    def test_strips_and_upcases(self):
        assert _normalize_zone(" moderate ") == "MODERATE"

    def test_screen_review_dni_zones_are_low_and_moderate(self):
        assert SCREEN_REVIEW_DNI_ZONES == frozenset({"LOW", "MODERATE"})


class TestClassifySellType:
    def test_stop_breached_always_returns_stop_breach(self):
        assert _classify_sell_type(None, stop_breached=True) == "STOP_BREACH"
        assert (
            _classify_sell_type(_make_analysis(60.0, 60.0), stop_breached=True)
            == "STOP_BREACH"
        )

    def test_no_analysis_returns_hard_reject(self):
        assert _classify_sell_type(None, stop_breached=False) == "HARD_REJECT"

    def test_both_scores_ok_returns_soft_reject(self):
        analysis = _make_analysis(health=55.0, growth=60.0)
        assert _classify_sell_type(analysis, stop_breached=False) == "SOFT_REJECT"

    def test_low_health_returns_hard_reject(self):
        analysis = _make_analysis(health=40.0, growth=60.0)
        assert _classify_sell_type(analysis, stop_breached=False) == "HARD_REJECT"

    def test_low_growth_returns_hard_reject(self):
        analysis = _make_analysis(health=60.0, growth=45.0)
        assert _classify_sell_type(analysis, stop_breached=False) == "HARD_REJECT"

    def test_boundary_exactly_50_is_ok(self):
        analysis = _make_analysis(health=50.0, growth=50.0)
        assert _classify_sell_type(analysis, stop_breached=False) == "SOFT_REJECT"

    def test_none_scores_returns_hard_reject(self):
        analysis = AnalysisRecord(
            ticker="7203.T",
            analysis_date="2026-01-01",
            health_adj=None,
            growth_adj=None,
        )
        assert _classify_sell_type(analysis, stop_breached=False) == "HARD_REJECT"
