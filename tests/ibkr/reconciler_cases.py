"""Shared reconciler test cases and helpers for IBKR split test modules."""

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import ANY, MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]

from src.fx_normalization import set_fx_rate_cache
from src.ibkr.analysis_index import (
    AnalysisLoadProgress,
    _analysis_index_lock,
    _analysis_index_lock_path,
    _analysis_index_path,
    _build_analysis_record_from_data,
    _parse_scores_from_final_decision,
    load_latest_analyses,
    update_latest_analyses_index,
)
from src.ibkr.models import (
    AnalysisRecord,
    NormalizedPosition,
    PortfolioSummary,
    ReconciliationItem,
    TradeBlockData,
)
from src.ibkr.portfolio_health import compute_portfolio_health
from src.ibkr.reconciler import ReconciliationDiagnostics, reconcile
from src.ibkr.reconciliation_rules import (
    _exchange_from_position,
    _resolve_fx,
    check_staleness,
    check_stop_breach,
    check_target_hit,
)
from src.ibkr.ticker import Ticker


class _FakeFxRateCache:
    """Deterministic, network-free stand-in for FxRateCache.

    Most IBKR reconciliation/portfolio tests exercise unit-classification or
    repair logic, not live FX fetching — pinning fixed rates here keeps them
    independent of both yfinance availability and future
    FALLBACK_RATES_TO_USD refreshes. Currencies not listed here resolve to
    None, matching production behavior for a genuinely unresolvable currency.
    """

    _RATES = {
        "JPY": 0.0067,
        "KRW": 0.00075,
        "TWD": 0.032,
        "HKD": 0.128,
        "GBP": 1.27,
        "SEK": 0.093,
        "NOK": 0.092,
        "DKK": 0.144,
    }

    def peek_cached_rate(self, currency: str, to_currency: str = "USD"):
        currency = currency.strip().upper()
        if currency == to_currency:
            return 1.0, "identity"
        rate = self._RATES.get(currency)
        return (rate, "fallback") if rate is not None else None

    def resolve_rates_sync(self, currencies, to_currency: str = "USD"):
        to_currency = to_currency.strip().upper()
        result = {}
        for currency in currencies:
            currency = currency.strip().upper()
            if currency == to_currency:
                continue
            rate = self._RATES.get(currency)
            if rate is not None:
                result[currency] = (rate, "fallback")
        return result


def _recent(days_ago: int) -> str:
    """ISO date `days_ago` before today — for macro tests that must fall
    within the (recent) macro-override window regardless of the run date."""
    from datetime import date, timedelta

    return (date.today() - timedelta(days=days_ago)).isoformat()


# ── Fixtures ──


def _make_position(
    ticker: str = "7203.T",
    quantity: float = 100,
    avg_cost: float = 2000,
    current_price: float = 2100,
    market_value_usd: float = 1400,
    currency: str = "JPY",
    conid: int = 123456,
    tax_term: str = "UNKNOWN",
) -> NormalizedPosition:
    from src.ibkr.ticker import Ticker

    return NormalizedPosition(
        conid=conid,
        ticker=Ticker.from_yf(ticker, currency=currency),
        quantity=quantity,
        avg_cost_local=avg_cost,
        market_value_usd=market_value_usd,
        currency=currency,
        current_price_local=current_price,
        tax_term=tax_term,
        ticker_identity_verified=True,
        ticker_resolution_source="exchange_map",
    )


def _make_analysis(
    ticker: str = "7203.T",
    verdict: str = "BUY",
    age_days: int = 5,
    entry_price: float = 2100.0,
    stop_price: float = 1900.0,
    target_1: float = 2500.0,
    target_2: float = 3000.0,
    conviction: str = "Medium",
    size_pct: float = 5.0,
    current_price: float = 2100.0,
    capital_flag_types: tuple[str, ...] = (),
    risk_tally: float | None = None,
    quality_flag_types: tuple[str, ...] = (),
) -> AnalysisRecord:
    from datetime import datetime, timedelta

    analysis_date = (datetime.now() - timedelta(days=age_days)).strftime("%Y-%m-%d")
    return AnalysisRecord(
        ticker=ticker,
        analysis_date=analysis_date,
        verdict=verdict,
        current_price=current_price,
        health_adj=70.0,
        growth_adj=65.0,
        entry_price=entry_price,
        stop_price=stop_price,
        target_1_price=target_1,
        target_2_price=target_2,
        conviction=conviction,
        currency="JPY",
        trade_block=TradeBlockData(
            action=verdict,
            size_pct=size_pct,
            conviction=conviction,
            entry_price=entry_price,
            stop_price=stop_price,
            target_1_price=target_1,
            target_2_price=target_2,
        ),
        capital_flag_types=capital_flag_types,
        risk_tally=risk_tally,
        quality_flag_types=quality_flag_types,
    )


def _make_portfolio(
    value: float = 100000,
    cash: float = 15000,
    cash_buffer_pct: float = 0.05,
) -> PortfolioSummary:
    return PortfolioSummary(
        account_id="U1234567",
        portfolio_value_usd=value,
        cash_balance_usd=cash,
        cash_pct=cash / value if value > 0 else 0,
        available_cash_usd=cash - (value * cash_buffer_pct),
    )


# ── Staleness Tests ──


class TestCheckStaleness:
    def test_fresh_analysis(self):
        analysis = _make_analysis(age_days=5)
        is_stale, reason = check_staleness(analysis, current_price_local=2100)
        assert not is_stale

    def test_old_analysis(self):
        analysis = _make_analysis(age_days=20)
        is_stale, reason = check_staleness(analysis, max_age_days=14)
        assert is_stale
        assert "age 20d" in reason

    def test_price_drift_up(self):
        analysis = _make_analysis(entry_price=100.0)
        is_stale, reason = check_staleness(
            analysis, current_price_local=120.0, drift_threshold_pct=15.0
        )
        assert is_stale
        assert "drift" in reason
        assert "up" in reason

    def test_price_drift_down(self):
        analysis = _make_analysis(entry_price=100.0)
        is_stale, reason = check_staleness(
            analysis, current_price_local=80.0, drift_threshold_pct=15.0
        )
        assert is_stale
        assert "down" in reason

    def test_small_drift_ok(self):
        analysis = _make_analysis(entry_price=100.0)
        is_stale, _ = check_staleness(
            analysis, current_price_local=108.0, drift_threshold_pct=15.0
        )
        assert not is_stale

    def test_no_current_price(self):
        analysis = _make_analysis(age_days=5)
        is_stale, _ = check_staleness(analysis, current_price_local=None)
        assert not is_stale

    def test_global_structural_event_after_analysis_forces_stale(self):
        """GLOBAL STRUCTURAL event after analysis date → stale."""
        from src.memory import MacroEvent

        analysis = _make_analysis(ticker="7203.T", age_days=3)
        # event_date is 1 day after analysis_date → fires
        from datetime import datetime, timedelta

        event_date = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")
        event = MacroEvent(
            event_date=event_date,
            detected_date=event_date,
            expiry="2026-06-01",
            impact="STRUCTURAL",
            event_type="REGULATORY_SHIFT",
            scope="GLOBAL",
            primary_region="GLOBAL",
            primary_sector="",
            severity="HIGH",
            correlation_pct=0.45,
            peak_count=10,
            total_held=22,
            news_headline="New legislation enacted",
            news_detail="",
        )
        is_stale, reason = check_staleness(analysis, structural_macro_events=[event])
        assert is_stale is True
        assert "STRUCTURAL macro event" in reason

    def test_regional_structural_event_does_not_invalidate_different_exchange(self):
        """STRUCTURAL event scoped to .T must not invalidate a .HK ticker."""
        from src.memory import MacroEvent

        analysis = _make_analysis(ticker="0005.HK", age_days=3)
        from datetime import datetime, timedelta

        event_date = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")
        event = MacroEvent(
            event_date=event_date,
            detected_date=event_date,
            expiry="2026-06-01",
            impact="STRUCTURAL",
            event_type="REGULATORY_SHIFT",
            scope="REGIONAL",
            primary_region=".T",
            primary_sector="",
            severity="HIGH",
            correlation_pct=0.45,
            peak_count=8,
            total_held=20,
            news_headline="Japan regulation change",
            news_detail="",
        )
        is_stale, _ = check_staleness(analysis, structural_macro_events=[event])
        assert is_stale is False


class TestAlphaBaseFallback:
    def test_alphanumeric_base_matches_suffixed_analysis(self):
        pos = _make_position(ticker="262A.T", currency="JPY")
        pos = NormalizedPosition(
            conid=pos.conid,
            ticker=Ticker.from_ibkr("262A", "UNKNOWN", "JPY"),
            quantity=pos.quantity,
            avg_cost_local=pos.avg_cost_local,
            market_value_usd=pos.market_value_usd,
            currency=pos.currency,
            current_price_local=pos.current_price_local,
        )
        analyses = {"262A.T": _make_analysis(ticker="262A.T")}

        items = reconcile(
            positions=[pos], analyses=analyses, portfolio=_make_portfolio()
        )

        assert len(items) == 1
        assert items[0].ticker.yf == "262A.T"

    def test_suffixed_analysis_wins_over_bare_for_alphanumeric_base(self):
        pos = NormalizedPosition(
            conid=123456,
            ticker=Ticker.from_ibkr("CEK", "UNKNOWN", "EUR"),
            quantity=100,
            avg_cost_local=10,
            market_value_usd=1000,
            currency="EUR",
            current_price_local=10,
        )
        # Currency must agree with the EUR position — the base-match guard
        # blocks cross-currency borrows even for suffix-less positions.
        cek = _make_analysis(ticker="CEK", verdict="HOLD")
        cek.currency = "EUR"
        cek_de = _make_analysis(ticker="CEK.DE", verdict="BUY")
        cek_de.currency = "EUR"
        analyses = {"CEK": cek, "CEK.DE": cek_de}

        items = reconcile(
            positions=[pos], analyses=analyses, portfolio=_make_portfolio()
        )

        assert len(items) == 1
        assert items[0].ticker.yf == "CEK.DE"

    def test_numeric_base_does_not_crossmatch_across_exchanges(self):
        pos = NormalizedPosition(
            conid=123456,
            ticker=Ticker.from_ibkr("2628", "UNKNOWN", ""),
            quantity=100,
            avg_cost_local=10,
            market_value_usd=1000,
            currency="USD",
            current_price_local=10,
        )
        analyses = {
            "2628.HK": _make_analysis(
                ticker="2628.HK", verdict="BUY", current_price=10
            ),
            "2628.TW": _make_analysis(
                ticker="2628.TW", verdict="BUY", current_price=10
            ),
            "2628.T": _make_analysis(ticker="2628.T", verdict="BUY", current_price=10),
        }

        items = reconcile(
            positions=[pos], analyses=analyses, portfolio=_make_portfolio()
        )

        assert len(items) == 1
        assert items[0].action == "REVIEW"
        assert "no evaluator analysis" in items[0].reason.lower()

    def test_regional_structural_event_invalidates_matching_exchange(self):
        """STRUCTURAL event scoped to .T invalidates a .T ticker."""
        from src.memory import MacroEvent

        analysis = _make_analysis(ticker="7203.T", age_days=3)
        from datetime import datetime, timedelta

        event_date = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")
        event = MacroEvent(
            event_date=event_date,
            detected_date=event_date,
            expiry="2026-06-01",
            impact="STRUCTURAL",
            event_type="REGULATORY_SHIFT",
            scope="REGIONAL",
            primary_region=".T",
            primary_sector="",
            severity="HIGH",
            correlation_pct=0.45,
            peak_count=8,
            total_held=20,
            news_headline="Japan regulation change",
            news_detail="",
        )
        is_stale, reason = check_staleness(analysis, structural_macro_events=[event])
        assert is_stale is True
        assert "STRUCTURAL macro event" in reason


class TestCheckStopBreach:
    def test_stop_breached(self):
        analysis = _make_analysis(stop_price=1900.0)
        assert check_stop_breach(analysis, 1850.0) is True

    def test_above_stop(self):
        analysis = _make_analysis(stop_price=1900.0)
        assert check_stop_breach(analysis, 2000.0) is False

    def test_no_stop(self):
        analysis = _make_analysis(stop_price=None)
        assert check_stop_breach(analysis, 1500.0) is False


class TestCheckTargetHit:
    def test_target_hit(self):
        analysis = _make_analysis(target_1=2500.0)
        assert check_target_hit(analysis, 2600.0) is True

    def test_below_target(self):
        analysis = _make_analysis(target_1=2500.0)
        assert check_target_hit(analysis, 2300.0) is False

    def test_no_target(self):
        analysis = _make_analysis(target_1=None)
        assert check_target_hit(analysis, 5000.0) is False


# ── Reconciliation Tests ──


class TestReconcile:
    """Test the core reconciliation logic — position-aware action generation."""

    def test_held_buy_within_targets(self):
        """Held + evaluator says BUY + within targets → HOLD."""
        pos = _make_position(current_price=2100)
        analysis = _make_analysis(
            verdict="BUY", entry_price=2000, stop_price=1800, target_1=2500
        )
        items = reconcile(
            [pos],
            {"7203.T": analysis},
            _make_portfolio(),
        )
        assert len(items) == 1
        assert items[0].action == "HOLD"

    def test_held_but_verdict_dni(self):
        """Held + unconfirmed DO_NOT_INITIATE → REVIEW, never unattended SELL.

        A stock-level rejection may trigger refresh, review, replacement
        analysis, or an exit — it must not decide among those alone.
        """
        pos = _make_position(current_price=2100)
        analysis = _make_analysis(verdict="DO_NOT_INITIATE")
        items = reconcile(
            [pos],
            {"7203.T": analysis},
            _make_portfolio(),
        )
        assert len(items) == 1
        assert items[0].action == "REVIEW"
        assert items[0].urgency == "MEDIUM"
        assert "do_not_initiate" in items[0].reason.lower()

    def test_held_but_verdict_sell(self):
        """Held + unconfirmed SELL verdict → REVIEW with a typed basis
        (intact fundamentals at entry price → entry constraint)."""
        pos = _make_position()
        analysis = _make_analysis(verdict="SELL")
        items = reconcile([pos], {"7203.T": analysis}, _make_portfolio())
        assert items[0].action == "REVIEW"
        assert items[0].action_basis == "ENTRY_CONSTRAINT"

    def test_held_stop_breached(self):
        """Held + price below the review level → urgent non-executable REVIEW
        (July 2026: a price break has no sell authority)."""
        pos = _make_position(current_price=1700)
        analysis = _make_analysis(stop_price=1900)
        items = reconcile([pos], {"7203.T": analysis}, _make_portfolio())
        assert items[0].action == "REVIEW"
        assert items[0].urgency == "HIGH"
        assert "review level" in items[0].reason
        assert items[0].suggested_quantity is None
        assert items[0].cash_impact_usd == 0.0

    def test_zero_quantity_position_with_stop_breach_ignored(self):
        """IBKR position with quantity=0 (just sold) must not generate a SELL.

        Regression: IBKR briefly retains zero-quantity positions after a fill.
        Previously this produced a spurious SELL with no share count or proceeds.
        Use DO_NOT_INITIATE verdict so Phase 2 does not regenerate a BUY either.
        """
        pos = _make_position(quantity=0, current_price=1700)
        analysis = _make_analysis(verdict="DO_NOT_INITIATE", stop_price=1900)
        items = reconcile([pos], {"7203.T": analysis}, _make_portfolio())
        assert items == []

    def test_zero_quantity_position_with_verdict_conflict_ignored(self):
        """IBKR position with quantity=0 must not generate a REVIEW/SELL for verdict conflict."""
        pos = _make_position(quantity=0)
        analysis = _make_analysis(verdict="DO_NOT_INITIATE")
        items = reconcile([pos], {"7203.T": analysis}, _make_portfolio())
        assert items == []

    def test_held_base_case_reference_reached(self):
        """Held + price at base-case valuation reference → REVIEW."""
        pos = _make_position(current_price=2600)
        analysis = _make_analysis(target_1=2500)
        items = reconcile([pos], {"7203.T": analysis}, _make_portfolio())
        assert items[0].action == "REVIEW"
        assert "base-case valuation reference" in items[0].reason.lower()

    def test_held_stale_analysis(self):
        """Held + stale analysis → REVIEW."""
        pos = _make_position()
        analysis = _make_analysis(age_days=20)
        items = reconcile(
            [pos],
            {"7203.T": analysis},
            _make_portfolio(),
            max_age_days=14,
        )
        assert items[0].action == "REVIEW"
        assert "stale" in items[0].reason.lower()

    def test_held_no_analysis(self):
        """Held but no analysis exists → REVIEW."""
        pos = _make_position()
        items = reconcile([pos], {}, _make_portfolio())
        assert items[0].action == "REVIEW"
        assert "no evaluator analysis" in items[0].reason.lower()

    def test_not_held_buy_recommendation(self):
        """Not held + evaluator says BUY + fresh analysis → BUY."""
        analysis = _make_analysis(verdict="BUY", age_days=3)
        items = reconcile(
            [],
            {"7203.T": analysis},
            _make_portfolio(cash=15000),
        )
        assert len(items) == 1
        assert items[0].action == "BUY"

    def test_not_held_buy_but_stale(self):
        """Not held + evaluator says BUY but stale → skipped."""
        analysis = _make_analysis(verdict="BUY", age_days=20)
        items = reconcile(
            [],
            {"7203.T": analysis},
            _make_portfolio(),
            max_age_days=14,
        )
        assert len(items) == 0

    def test_not_held_dni_ignored(self):
        """Not held + evaluator says DNI → no action (we don't show these)."""
        analysis = _make_analysis(verdict="DO_NOT_INITIATE")
        items = reconcile([], {"7203.T": analysis}, _make_portfolio())
        assert len(items) == 0

    def test_no_cash_retains_non_executable_new_buy_for_watchlist_ranking(self):
        """No available cash retains a non-executable candidate for watchlist merit."""
        analysis = _make_analysis(verdict="BUY", age_days=3)
        items = reconcile(
            [],
            {"7203.T": analysis},
            _make_portfolio(cash=0, value=100000),
        )
        assert len(items) == 1
        assert items[0].is_cash_blocked is True
        assert items[0].suggested_quantity is None
        assert items[0].cash_impact_usd == 0.0

    def test_overweight_position(self):
        """Overweight position → advisory REVIEW with trim math (July 2026:
        a drift trim is a capital-gains sale the operator decides)."""
        pos = _make_position(market_value_usd=30000)  # 30% of 100k
        analysis = _make_analysis(verdict="BUY", size_pct=5.0)  # Target 5%
        items = reconcile(
            [pos],
            {"7203.T": analysis},
            _make_portfolio(value=100000),
            overweight_threshold_pct=20.0,
        )
        assert items[0].action == "REVIEW"
        assert items[0].action_basis == "OVERWEIGHT"
        assert "Overweight" in items[0].reason
        assert "advisory" in items[0].reason
        assert items[0].suggested_quantity is None
        assert items[0].cash_impact_usd == 0.0

    def test_urgency_sorting(self):
        """HIGH urgency items should come first."""
        pos_stop = _make_position(ticker="0005.HK", current_price=50, conid=1)
        pos_ok = _make_position(ticker="7203.T", current_price=2100, conid=2)
        analyses = {
            "0005.HK": _make_analysis(ticker="0005.HK", stop_price=55),  # breached
            "7203.T": _make_analysis(ticker="7203.T", stop_price=1800),  # ok
        }
        items = reconcile([pos_stop, pos_ok], analyses, _make_portfolio())
        assert items[0].urgency == "HIGH"
        assert items[0].ticker.yf == "0005.HK"

    def test_mixed_positions_and_recommendations(self):
        """Multiple positions + new buy recs should all appear."""
        held = _make_position(ticker="7203.T", current_price=2100, conid=1)
        analyses = {
            "7203.T": _make_analysis(ticker="7203.T", verdict="BUY"),
            "0005.HK": _make_analysis(ticker="0005.HK", verdict="BUY", age_days=3),
        }
        items = reconcile([held], analyses, _make_portfolio(cash=15000))
        tickers = {i.ticker.yf for i in items}
        assert "7203.T" in tickers
        assert "0005.HK" in tickers
        # 7203.T should be HOLD (already held), 0005.HK should be BUY (not held)
        actions = {i.ticker.yf: i.action for i in items}
        assert actions["7203.T"] == "HOLD"
        assert actions["0005.HK"] == "BUY"


# ── Watchlist Phase 1.5 verdict routing tests ──


class TestWatchlistVerdictRouting:
    """Watchlist tickers with space-separated verdicts must route to REMOVE, not MONITORING."""

    def _run(self, verdict: str) -> list[ReconciliationItem]:
        """Reconcile a single watchlist ticker (not held) with the given verdict."""
        analysis = _make_analysis(ticker="SOP", verdict=verdict, age_days=0)
        portfolio = _make_portfolio(value=100_000, cash=50_000)
        portfolio.exchange_weights = {}
        portfolio.sector_weights = {}
        return reconcile(
            positions=[],
            analyses={"SOP": analysis},
            portfolio=portfolio,
            watchlist_tickers={"SOP"},
        )

    def test_do_not_initiate_with_underscores_routes_to_remove(self):
        """Canonical 'DO_NOT_INITIATE' → REMOVE action, is_watchlist=True."""
        items = self._run("DO_NOT_INITIATE")
        assert len(items) == 1
        assert items[0].action == "REMOVE"
        assert items[0].is_watchlist

    def test_do_not_initiate_with_spaces_routes_to_remove(self):
        """Space-variant 'DO NOT INITIATE' → same REMOVE routing as underscore form."""
        items = self._run("DO NOT INITIATE")
        assert len(items) == 1
        assert items[0].action == "REMOVE"
        assert items[0].is_watchlist

    def test_do_not_initiate_mixed_case_routes_to_remove(self):
        """Lowercase variant 'do_not_initiate' → normalised and routed to REMOVE."""
        items = self._run("do_not_initiate")
        assert len(items) == 1
        assert items[0].action == "REMOVE"

    def test_hold_verdict_routes_to_monitoring(self):
        """HOLD verdict → HOLD action (monitoring), not REMOVE."""
        items = self._run("HOLD")
        assert len(items) == 1
        assert items[0].action == "HOLD"
        assert items[0].is_watchlist


class TestWatchlistSuffixDedup:
    """Phase 1.5: bare watchlist ticker must block the suffixed form from Phase 2."""

    def _make_port(self) -> PortfolioSummary:
        p = _make_portfolio(value=100_000, cash=50_000)
        p.exchange_weights = {}
        p.sector_weights = {}
        return p

    def test_bare_watchlist_blocks_suffixed_when_both_analyses_exist(self):
        """Watchlist 'WDO' + analyses {'WDO': DNI, 'WDO.TO': BUY}.

        'WDO.TO' must NOT appear as a Phase 2 untracked BUY candidate.
        The watchlist item should prefer the suffixed BUY analysis.
        """
        bare_dni = _make_analysis(ticker="WDO", verdict="DO_NOT_INITIATE", age_days=5)
        suffixed_buy = _make_analysis(ticker="WDO.TO", verdict="BUY", age_days=1)
        items = reconcile(
            positions=[],
            analyses={"WDO": bare_dni, "WDO.TO": suffixed_buy},
            portfolio=self._make_port(),
            watchlist_tickers={"WDO"},
        )
        # Must NOT produce a Phase 2 (non-watchlist) BUY for "WDO.TO"
        phase2_buys = [i for i in items if i.action == "BUY" and not i.is_watchlist]
        assert (
            not phase2_buys
        ), f"Unexpected Phase 2 BUY: {[i.ticker.yf for i in phase2_buys]}"
        # Watchlist item must use the suffixed ticker and BUY verdict
        wl = [i for i in items if i.is_watchlist]
        assert len(wl) == 1
        assert wl[0].action == "BUY"
        assert wl[0].ticker.yf == "WDO.TO"
        assert wl[0].ticker.ibkr == "WDO"

    def test_bare_watchlist_blocks_suffixed_when_only_suffixed_analysis_exists(self):
        """Watchlist 'WDO' + analyses {'WDO.TO': BUY}.

        Regression: the existing fallback path must still work when only the
        suffixed analysis exists (bare analysis absent).
        """
        suffixed_buy = _make_analysis(ticker="WDO.TO", verdict="BUY", age_days=1)
        items = reconcile(
            positions=[],
            analyses={"WDO.TO": suffixed_buy},
            portfolio=self._make_port(),
            watchlist_tickers={"WDO"},
        )
        phase2_buys = [i for i in items if i.action == "BUY" and not i.is_watchlist]
        assert (
            not phase2_buys
        ), f"Unexpected Phase 2 BUY: {[i.ticker.yf for i in phase2_buys]}"
        wl = [i for i in items if i.is_watchlist and i.action == "BUY"]
        assert len(wl) == 1
        assert wl[0].ticker.yf == "WDO.TO"

    def test_suffixed_watchlist_ticker_not_duplicated_in_phase2(self):
        """Watchlist 'WDO.TO' (already suffixed) must not produce a Phase 2 BUY."""
        buy = _make_analysis(ticker="WDO.TO", verdict="BUY", age_days=1)
        items = reconcile(
            positions=[],
            analyses={"WDO.TO": buy},
            portfolio=self._make_port(),
            watchlist_tickers={"WDO.TO"},
        )
        phase2_buys = [i for i in items if i.action == "BUY" and not i.is_watchlist]
        assert not phase2_buys

    def test_numeric_bare_watchlist_blocks_suffixed_candidate(self):
        """Watchlist '5434' (numeric base, no exchange suffix) must block '5434.TW' in Phase 2.

        _alpha_base_to_key skips numeric symbols so the 'else' branch must handle this.
        """
        buy = _make_analysis(ticker="5434.TW", verdict="BUY", age_days=1)
        items = reconcile(
            positions=[],
            analyses={"5434.TW": buy},
            portfolio=self._make_port(),
            watchlist_tickers={"5434"},
        )
        phase2_buys = [i for i in items if i.action == "BUY" and not i.is_watchlist]
        assert not phase2_buys, f"Numeric watchlist produced false candidate: {[i.ticker.yf for i in phase2_buys]}"
        wl = [i for i in items if i.is_watchlist and i.action == "BUY"]
        assert len(wl) == 1
        assert wl[0].ticker.yf == "5434.TW"

    def test_numeric_bare_watchlist_does_not_block_different_exchange(self):
        """Watchlist '5434' must NOT block a different stock like '5400.TW'."""
        buy = _make_analysis(ticker="5400.TW", verdict="BUY", age_days=1)
        items = reconcile(
            positions=[],
            analyses={"5400.TW": buy},
            portfolio=self._make_port(),
            watchlist_tickers={"5434"},
        )
        phase2_buys = [i for i in items if i.action == "BUY" and not i.is_watchlist]
        assert len(phase2_buys) == 1
        assert phase2_buys[0].ticker.yf == "5400.TW"


class TestHeldPositionBlocksCandidate:
    """Phase 1 held position must prevent the same ticker from appearing as a Phase 2 BUY candidate."""

    def _make_port(self) -> PortfolioSummary:
        p = _make_portfolio(value=100_000, cash=50_000)
        p.exchange_weights = {}
        p.sector_weights = {}
        return p

    def test_numeric_held_bare_blocks_suffixed_candidate(self):
        """Held bare '5434' (IBKR numeric, no exchange resolved) must block '5434.TW' in Phase 2.

        The alpha_base_lookup is explicitly skipped for numeric tickers, so the
        bare-ticker cross-reference added after held_tickers.add() must cover this gap.
        """
        pos = _make_position(
            ticker="5434", quantity=1000, currency="TWD", current_price=334.5
        )
        buy_analysis = _make_analysis(ticker="5434.TW", verdict="BUY", age_days=1)
        items = reconcile(
            positions=[pos],
            analyses={"5434.TW": buy_analysis},
            portfolio=self._make_port(),
        )
        phase2_buys = [i for i in items if i.action == "BUY" and not i.is_watchlist]
        assert not phase2_buys, f"Held position produced false candidate: {[i.ticker.yf for i in phase2_buys]}"

    def test_alpha_held_bare_blocks_suffixed_candidate(self):
        """Held bare 'WDO' (alphabetic, no exchange resolved) must block 'WDO.TO' in Phase 2.

        alpha_base_lookup handles this when the analysis exists, but the bare-ticker
        cross-reference provides a safety net when the lookup misses.
        """
        pos = _make_position(
            ticker="WDO", quantity=100, currency="CAD", current_price=22.95
        )
        buy_analysis = _make_analysis(ticker="WDO.TO", verdict="BUY", age_days=1)
        items = reconcile(
            positions=[pos],
            analyses={"WDO.TO": buy_analysis},
            portfolio=self._make_port(),
        )
        phase2_buys = [i for i in items if i.action == "BUY" and not i.is_watchlist]
        assert not phase2_buys, f"Held position produced false candidate: {[i.ticker.yf for i in phase2_buys]}"

    def test_suffixed_held_still_blocks_same_key(self):
        """Held '5434.TW' (suffix already resolved) blocks the same analysis normally."""
        pos = _make_position(
            ticker="5434.TW", quantity=1000, currency="TWD", current_price=334.5
        )
        buy_analysis = _make_analysis(ticker="5434.TW", verdict="BUY", age_days=1)
        items = reconcile(
            positions=[pos],
            analyses={"5434.TW": buy_analysis},
            portfolio=self._make_port(),
        )
        phase2_buys = [i for i in items if i.action == "BUY" and not i.is_watchlist]
        assert not phase2_buys

    def test_different_base_candidate_not_blocked(self):
        """Holding '5434' does not block a candidate with a different base (e.g. 'WDO.TO')."""
        pos = _make_position(
            ticker="5434", quantity=1000, currency="TWD", current_price=334.5
        )
        buy_analysis = _make_analysis(ticker="WDO.TO", verdict="BUY", age_days=1)
        items = reconcile(
            positions=[pos],
            analyses={"WDO.TO": buy_analysis},
            portfolio=self._make_port(),
        )
        phase2_buys = [i for i in items if i.action == "BUY" and not i.is_watchlist]
        assert len(phase2_buys) == 1
        assert phase2_buys[0].ticker.yf == "WDO.TO"


class TestPhase2CashBlockedCandidates:
    def test_offwatch_buy_is_retained_when_cash_policy_blocks_it(self):
        portfolio = _make_portfolio(value=100_000, cash=0)
        portfolio.exchange_weights = {}
        portfolio.sector_weights = {}
        diagnostics = ReconciliationDiagnostics()
        buy = _make_analysis(ticker="WDO.TO", verdict="BUY", age_days=1)

        items = reconcile(
            positions=[],
            analyses={"WDO.TO": buy},
            portfolio=portfolio,
            diagnostics=diagnostics,
        )

        phase2_buys = [i for i in items if i.action == "BUY" and not i.is_watchlist]
        assert len(phase2_buys) == 1
        assert phase2_buys[0].is_cash_blocked is True
        assert phase2_buys[0].suggested_quantity is None
        assert phase2_buys[0].cash_impact_usd == 0.0
        assert diagnostics.cash_blocked_offwatch_buy_count == 1


# ── Analysis Loading Tests ──


class TestLoadLatestAnalyses:
    def test_loads_from_json_files(self, tmp_path):
        """Correctly loads analysis JSON with prediction_snapshot."""
        analysis_data = {
            "prediction_snapshot": {
                "ticker": "7203.T",
                "analysis_date": "2026-02-20",
                "verdict": "BUY",
                "current_price": 2100.0,
                "currency": "JPY",
            },
            "investment_analysis": {
                "trader_plan": (
                    "TRADE_BLOCK:\n"
                    "ACTION: BUY\n"
                    "SIZE: 5.0%\n"
                    "CONVICTION: High\n"
                    "ENTRY: 2,100 (Limit)\n"
                    "STOP: 1,900 (-9.5%)\n"
                    "TARGET_1: 2,500 (+19.0%)\n"
                    "TARGET_2: 3,000 (+42.9%)\n"
                    "R:R: 4.2:1\n"
                    "SPECIAL: JPY exposure\n"
                ),
            },
        }
        filepath = tmp_path / "7203_T_2026-02-20_analysis.json"
        filepath.write_text(json.dumps(analysis_data))

        result = load_latest_analyses(tmp_path)
        assert "7203.T" in result
        record = result["7203.T"]
        assert record.verdict == "BUY"
        assert record.trade_block.entry_price == 2100.0
        assert record.trade_block.stop_price == 1900.0

    def test_empty_dir(self, tmp_path):
        result = load_latest_analyses(tmp_path)
        assert result == {}

    def test_nonexistent_dir(self):
        result = load_latest_analyses(Path("/nonexistent/path"))
        assert result == {}

    def test_malformed_json_skipped(self, tmp_path):
        (tmp_path / "bad_analysis.json").write_text("not json{{{")
        result = load_latest_analyses(tmp_path)
        assert result == {}

    def test_most_recent_wins(self, tmp_path):
        """If multiple analyses for same ticker, most recent file wins."""
        for date in ["2026-02-15", "2026-02-20"]:
            data = {
                "prediction_snapshot": {
                    "ticker": "7203.T",
                    "analysis_date": date,
                    "verdict": "BUY" if date == "2026-02-20" else "HOLD",
                },
                "investment_analysis": {},
            }
            filepath = tmp_path / f"7203_T_{date}_analysis.json"
            filepath.write_text(json.dumps(data))

        result = load_latest_analyses(tmp_path)
        # Files are sorted reverse by name, so 2026-02-20 should win
        assert result["7203.T"].verdict == "BUY"

    def test_loads_sector_and_exchange(self, tmp_path):
        """sector and exchange are loaded from snapshot; exchange inferred from ticker if absent."""
        data_with_sector = {
            "prediction_snapshot": {
                "ticker": "7203.T",
                "analysis_date": "2026-02-20",
                "verdict": "BUY",
                "sector": "Consumer Discretionary",
                "exchange": "T",
            },
            "investment_analysis": {},
        }
        (tmp_path / "7203_T_2026-02-20_analysis.json").write_text(
            json.dumps(data_with_sector)
        )
        result = load_latest_analyses(tmp_path)
        assert result["7203.T"].sector == "Consumer Discretionary"
        assert result["7203.T"].exchange == "T"

    def test_derives_idle_cash_capital_flags_from_fundamentals_report(self, tmp_path):
        """Analysis loading bridges saved fundamentals into IBKR profit-take inputs."""
        data = {
            "prediction_snapshot": {
                "ticker": "7203.T",
                "analysis_date": "2026-02-20",
                "verdict": "BUY",
            },
            "reports": {
                "fundamentals_report": """
### --- START DATA_BLOCK ---
SECTOR: Industrials
RAW_HEALTH_SCORE: 72%
ADJUSTED_HEALTH_SCORE: 72%
RAW_GROWTH_SCORE: 65%
ADJUSTED_GROWTH_SCORE: 65%
US_REVENUE_PERCENT: 0%
ANALYST_COVERAGE_ENGLISH: LOW
PE_RATIO_TTM: 12.0
ROIC_QUALITY: WEAK
LEVERAGE_QUALITY: CONSERVATIVE
ROIC_PERCENT: 4%
NET_CASH_TO_MARKET_CAP: 45%
CASH_TO_ASSETS: 28%
PAYOUT_RATIO: 0%
CAPEX_TO_DA_STATUS: MAINTENANCE
CAPITAL_PLAN_STATUS: NONE
### --- END DATA_BLOCK ---
""",
            },
            "investment_analysis": {},
        }
        (tmp_path / "7203_T_2026-02-20_analysis.json").write_text(json.dumps(data))

        result = load_latest_analyses(tmp_path)

        assert result["7203.T"].capital_flag_types == ("CAPITAL_IDLE_CASH_SEVERE",)

    def test_exchange_inferred_from_ticker_when_absent(self, tmp_path):
        """If exchange absent from snapshot, infer from ticker suffix."""
        data = {
            "prediction_snapshot": {
                "ticker": "0005.HK",
                "analysis_date": "2026-02-20",
                "verdict": "BUY",
            },
            "investment_analysis": {},
        }
        (tmp_path / "0005_HK_2026-02-20_analysis.json").write_text(json.dumps(data))
        result = load_latest_analyses(tmp_path)
        assert result["0005.HK"].exchange == "HK"

    def test_legacy_filename_fallback_recovers_hk_ticker_suffix(self, tmp_path):
        """Older files without snapshot ticker should recover suffix from filename."""
        data = {
            "final_decision": {
                "decision": "VERDICT: BUY\nHEALTH_ADJ: 80\nGROWTH_ADJ: 70"
            },
            "investment_analysis": {},
        }

        filepath = tmp_path / "0005_HK_2026-02-20_analysis.json"
        record = _build_analysis_record_from_data(filepath, data)

        assert record is not None
        assert record.ticker == "0005.HK"
        assert record.exchange == "HK"

    def test_legacy_filename_fallback_recovers_tse_ticker_suffix(self, tmp_path):
        """Older files without snapshot ticker should recover TSE suffix from filename."""
        data = {
            "final_decision": {
                "decision": "VERDICT: HOLD\nHEALTH_ADJ: 62\nGROWTH_ADJ: 55"
            },
            "investment_analysis": {},
        }

        filepath = tmp_path / "7203_T_2026-02-20_analysis.json"
        record = _build_analysis_record_from_data(filepath, data)

        assert record is not None
        assert record.ticker == "7203.T"
        assert record.exchange == "T"

    def test_is_quick_mode_loaded_from_snapshot(self, tmp_path):
        """is_quick_mode is extracted from prediction_snapshot when present."""
        data = {
            "prediction_snapshot": {
                "ticker": "7203.T",
                "analysis_date": "2026-03-01",
                "verdict": "BUY",
                "is_quick_mode": True,
            },
            "investment_analysis": {},
        }
        (tmp_path / "7203_T_2026-03-01_analysis.json").write_text(json.dumps(data))
        result = load_latest_analyses(tmp_path)
        assert result["7203.T"].is_quick_mode is True

    def test_is_quick_mode_unknown_when_absent(self, tmp_path):
        """Snapshots predating is_quick_mode load as mode-unknown (None).

        The legacy False default gave pre-field artifacts full-mode authority
        in the SELL confirmation gate; None is falsy, so quick-BUY gates
        behave exactly as before."""
        data = {
            "prediction_snapshot": {
                "ticker": "7203.T",
                "analysis_date": "2026-03-01",
                "verdict": "BUY",
                # no is_quick_mode field — older snapshot
            },
            "investment_analysis": {},
        }
        (tmp_path / "7203_T_2026-03-01_analysis.json").write_text(json.dumps(data))
        result = load_latest_analyses(tmp_path)
        assert result["7203.T"].is_quick_mode is None

    def test_progress_callback_receives_discovered_and_complete_events(self, tmp_path):
        """Progress callback gets start/end events plus parsing updates."""
        events: list[AnalysisLoadProgress] = []

        for ticker in ("7203.T", "0005.HK"):
            data = {
                "prediction_snapshot": {
                    "ticker": ticker,
                    "analysis_date": "2026-03-01",
                    "verdict": "BUY",
                },
                "investment_analysis": {},
            }
            filename = ticker.replace(".", "_")
            (tmp_path / f"{filename}_2026-03-01_analysis.json").write_text(
                json.dumps(data)
            )

        analyses = load_latest_analyses(tmp_path, progress=events.append)

        assert analyses.keys() == {"7203.T", "0005.HK"}
        assert events[0] == AnalysisLoadProgress(
            phase="discovered",
            total_files=2,
            processed_files=0,
            loaded_analyses=0,
            current_file=None,
        )
        assert any(event.phase == "parsing" for event in events)
        assert events[-1] == AnalysisLoadProgress(
            phase="complete",
            total_files=2,
            processed_files=2,
            loaded_analyses=2,
            current_file="0005_HK_2026-03-01_analysis.json",
        )

    def test_progress_callback_reports_duplicate_history_scans(self, tmp_path):
        """Older duplicate files still advance progress, even if they are skipped."""
        events: list[AnalysisLoadProgress] = []

        for date in ("2026-03-01", "2026-02-20"):
            data = {
                "prediction_snapshot": {
                    "ticker": "7203.T",
                    "analysis_date": date,
                    "verdict": "BUY",
                },
                "investment_analysis": {},
            }
            (tmp_path / f"7203_T_{date}_analysis.json").write_text(json.dumps(data))

        load_latest_analyses(tmp_path, progress=events.append)

        parsing_events = [event for event in events if event.phase == "parsing"]
        assert [event.processed_files for event in parsing_events] == [1, 2]

    def test_large_scan_progress_is_coarse_grained(self, tmp_path):
        """Larger directories emit early heartbeat updates, then coarse progress."""
        events: list[AnalysisLoadProgress] = []

        for i in range(30):
            data = {
                "prediction_snapshot": {
                    "ticker": f"TICK{i:02d}.TO",
                    "analysis_date": "2026-03-01",
                    "verdict": "BUY",
                },
                "investment_analysis": {},
            }
            (tmp_path / f"TICK{i:02d}_TO_2026-03-01_analysis.json").write_text(
                json.dumps(data)
            )

        load_latest_analyses(tmp_path, progress=events.append)

        parsing_events = [event for event in events if event.phase == "parsing"]
        assert [event.processed_files for event in parsing_events] == [1, 5, 10, 25, 30]

    def test_very_large_scan_progress_stays_visible(self, tmp_path):
        """Very large directories emit early and periodic heartbeat updates."""
        events: list[AnalysisLoadProgress] = []

        for i in range(1001):
            data = {
                "prediction_snapshot": {
                    "ticker": f"TICK{i:04d}.TO",
                    "analysis_date": "2026-03-01",
                    "verdict": "BUY",
                },
                "investment_analysis": {},
            }
            (tmp_path / f"TICK{i:04d}_TO_2026-03-01_analysis.json").write_text(
                json.dumps(data)
            )

        load_latest_analyses(tmp_path, progress=events.append)

        parsing_events = [
            event.processed_files for event in events if event.phase == "parsing"
        ]
        assert parsing_events[:6] == [1, 5, 10, 25, 50, 100]
        assert 250 in parsing_events
        assert 500 in parsing_events
        assert 750 in parsing_events
        assert 1000 in parsing_events
        assert parsing_events[-1] == 1001

    def test_default_loader_logging_is_aggregate_not_per_file(self, tmp_path):
        """Default load path emits one aggregate debug summary, not one log per file."""
        for ticker in ("7203.T", "0005.HK"):
            data = {
                "prediction_snapshot": {
                    "ticker": ticker,
                    "analysis_date": "2026-03-01",
                    "verdict": "BUY",
                },
                "investment_analysis": {},
            }
            filename = ticker.replace(".", "_")
            (tmp_path / f"{filename}_2026-03-01_analysis.json").write_text(
                json.dumps(data)
            )

        with patch("src.ibkr.analysis_index.logger", new=MagicMock()) as mock_logger:
            load_latest_analyses(tmp_path)

        debug_events = [call.args[0] for call in mock_logger.debug.call_args_list]
        assert "analysis_loaded" not in debug_events
        assert "analyses_scan_complete" in debug_events

    def test_load_latest_analyses_creates_index_automatically(self, tmp_path):
        """First full scan writes a sibling latest-analyses index automatically."""
        data = {
            "prediction_snapshot": {
                "ticker": "7203.T",
                "analysis_date": "2026-03-01",
                "verdict": "BUY",
            },
            "investment_analysis": {},
        }
        (tmp_path / "7203_T_2026-03-01_analysis.json").write_text(json.dumps(data))

        analyses = load_latest_analyses(tmp_path)

        index_path = _analysis_index_path(tmp_path)
        assert "7203.T" in analyses
        assert index_path.exists()

    def test_load_latest_analyses_uses_index_when_results_unchanged(
        self, tmp_path, monkeypatch
    ):
        """Second load reuses the cache index instead of reparsing analysis files."""
        data = {
            "prediction_snapshot": {
                "ticker": "7203.T",
                "analysis_date": "2026-03-01",
                "verdict": "BUY",
            },
            "investment_analysis": {},
        }
        (tmp_path / "7203_T_2026-03-01_analysis.json").write_text(json.dumps(data))

        first = load_latest_analyses(tmp_path)
        assert "7203.T" in first

        monkeypatch.setattr(
            "src.ibkr.analysis_index._build_analysis_record_from_file",
            lambda filepath: (_ for _ in ()).throw(AssertionError("should use index")),
        )

        second = load_latest_analyses(tmp_path)
        assert second["7203.T"].ticker == "7203.T"

    def test_load_latest_analyses_accepts_benign_dir_mtime_mismatch(
        self, tmp_path, monkeypatch
    ):
        """Directory mtime churn alone should not force a full rebuild."""
        data = {
            "prediction_snapshot": {
                "ticker": "7203.T",
                "analysis_date": "2026-03-01",
                "verdict": "BUY",
            },
            "investment_analysis": {},
        }
        (tmp_path / "7203_T_2026-03-01_analysis.json").write_text(json.dumps(data))

        load_latest_analyses(tmp_path)
        (tmp_path / "notes.txt").write_text("benign churn")

        monkeypatch.setattr(
            "src.ibkr.analysis_index._build_analysis_record_from_file",
            lambda filepath: (_ for _ in ()).throw(AssertionError("should use index")),
        )

        with patch("src.ibkr.analysis_index.logger", new=MagicMock()) as mock_logger:
            second = load_latest_analyses(tmp_path)

        assert second["7203.T"].ticker == "7203.T"
        # The accept is logged at debug (intentionally quiet — it fires on every
        # load given full-ns dir mtime never matches second-granular reads).
        debug_events = [call.args[0] for call in mock_logger.debug.call_args_list]
        assert "analysis_index_mtime_mismatch_accepted" in debug_events

    def test_load_latest_analyses_rebuilds_when_analysis_file_count_changes(
        self, tmp_path
    ):
        """Adding a new analysis file should still invalidate the stale index."""
        first = {
            "prediction_snapshot": {
                "ticker": "7203.T",
                "analysis_date": "2026-03-01",
                "verdict": "BUY",
            },
            "investment_analysis": {},
        }
        second = {
            "prediction_snapshot": {
                "ticker": "0005.HK",
                "analysis_date": "2026-03-01",
                "verdict": "HOLD",
            },
            "investment_analysis": {},
        }
        (tmp_path / "7203_T_2026-03-01_analysis.json").write_text(json.dumps(first))
        load_latest_analyses(tmp_path)
        (tmp_path / "0005_HK_2026-03-01_analysis.json").write_text(json.dumps(second))

        events: list[AnalysisLoadProgress] = []
        analyses = load_latest_analyses(tmp_path, progress=events.append)

        assert analyses.keys() == {"7203.T", "0005.HK"}
        assert any(event.phase == "rebuilding_index" for event in events)

    def test_load_latest_analyses_rebuilds_when_index_missing(self, tmp_path):
        """If the cache index is removed, the loader rescans and recreates it."""
        data = {
            "prediction_snapshot": {
                "ticker": "7203.T",
                "analysis_date": "2026-03-01",
                "verdict": "BUY",
            },
            "investment_analysis": {},
        }
        (tmp_path / "7203_T_2026-03-01_analysis.json").write_text(json.dumps(data))

        load_latest_analyses(tmp_path)
        index_path = _analysis_index_path(tmp_path)
        index_path.unlink()

        analyses = load_latest_analyses(tmp_path)
        assert analyses["7203.T"].ticker == "7203.T"
        assert index_path.exists()

    def test_load_latest_analyses_rebuilds_when_index_is_corrupt(self, tmp_path):
        """Corrupt cache index falls back to a full scan and is rewritten."""
        events: list[AnalysisLoadProgress] = []
        data = {
            "prediction_snapshot": {
                "ticker": "7203.T",
                "analysis_date": "2026-03-01",
                "verdict": "BUY",
            },
            "investment_analysis": {},
        }
        (tmp_path / "7203_T_2026-03-01_analysis.json").write_text(json.dumps(data))

        load_latest_analyses(tmp_path)
        index_path = _analysis_index_path(tmp_path)
        index_path.write_text("{not valid json")

        analyses = load_latest_analyses(tmp_path, progress=events.append)
        assert analyses["7203.T"].ticker == "7203.T"
        payload = json.loads(index_path.read_text())
        assert payload["version"] >= 1
        assert any(event.phase == "rebuilding_index" for event in events)

    def test_load_latest_analyses_rebuilds_when_index_is_truncated(self, tmp_path):
        """Truncated index file is treated as corrupt and rebuilt automatically."""
        data = {
            "prediction_snapshot": {
                "ticker": "7203.T",
                "analysis_date": "2026-03-01",
                "verdict": "BUY",
            },
            "investment_analysis": {},
        }
        (tmp_path / "7203_T_2026-03-01_analysis.json").write_text(json.dumps(data))

        load_latest_analyses(tmp_path)
        index_path = _analysis_index_path(tmp_path)
        index_path.write_text('{"version": 1, "results_dir": ')

        analyses = load_latest_analyses(tmp_path)
        assert analyses["7203.T"].ticker == "7203.T"
        assert json.loads(index_path.read_text())["version"] >= 1

    def test_load_latest_analyses_rebuilds_when_index_entry_source_changes(
        self, tmp_path
    ):
        """If a cached entry's source file changes, the whole index is rebuilt."""
        events: list[AnalysisLoadProgress] = []
        data = {
            "prediction_snapshot": {
                "ticker": "7203.T",
                "analysis_date": "2026-03-01",
                "verdict": "BUY",
            },
            "investment_analysis": {},
        }
        filepath = tmp_path / "7203_T_2026-03-01_analysis.json"
        filepath.write_text(json.dumps(data))
        load_latest_analyses(tmp_path)

        data["prediction_snapshot"]["verdict"] = "SELL"
        filepath.write_text(json.dumps(data))

        analyses = load_latest_analyses(tmp_path, progress=events.append)
        assert analyses["7203.T"].verdict == "SELL"
        assert any(event.phase == "rebuilding_index" for event in events)

    def test_load_latest_analyses_rebuilds_when_index_entry_source_missing(
        self, tmp_path
    ):
        """If an indexed source file disappears, the cache is rebuilt from remaining files."""
        events: list[AnalysisLoadProgress] = []
        first = {
            "prediction_snapshot": {
                "ticker": "7203.T",
                "analysis_date": "2026-03-01",
                "verdict": "BUY",
            },
            "investment_analysis": {},
        }
        second = {
            "prediction_snapshot": {
                "ticker": "6758.T",
                "analysis_date": "2026-03-02",
                "verdict": "BUY",
            },
            "investment_analysis": {},
        }
        first_path = tmp_path / "7203_T_2026-03-01_analysis.json"
        second_path = tmp_path / "6758_T_2026-03-02_analysis.json"
        first_path.write_text(json.dumps(first))
        second_path.write_text(json.dumps(second))
        load_latest_analyses(tmp_path)

        first_path.unlink()

        analyses = load_latest_analyses(tmp_path, progress=events.append)
        assert set(analyses) == {"6758.T"}
        assert any(event.phase == "rebuilding_index" for event in events)

    def test_load_latest_analyses_rebuilds_when_index_entry_metadata_missing(
        self, tmp_path
    ):
        """Missing per-entry source metadata invalidates the cache cleanly."""
        events: list[AnalysisLoadProgress] = []
        data = {
            "prediction_snapshot": {
                "ticker": "7203.T",
                "analysis_date": "2026-03-01",
                "verdict": "BUY",
            },
            "investment_analysis": {},
        }
        filepath = tmp_path / "7203_T_2026-03-01_analysis.json"
        filepath.write_text(json.dumps(data))
        load_latest_analyses(tmp_path)

        index_path = _analysis_index_path(tmp_path)
        payload = json.loads(index_path.read_text())
        payload["analyses"]["7203.T"].pop("source_mtime_ns")
        index_path.write_text(json.dumps(payload))

        analyses = load_latest_analyses(tmp_path, progress=events.append)
        assert analyses["7203.T"].ticker == "7203.T"
        assert any(event.phase == "rebuilding_index" for event in events)

    def test_load_latest_analyses_rebuilds_when_index_entry_source_size_mismatches(
        self, tmp_path
    ):
        """Wrong cached source_size invalidates the index entry and forces rebuild."""
        events: list[AnalysisLoadProgress] = []
        data = {
            "prediction_snapshot": {
                "ticker": "7203.T",
                "analysis_date": "2026-03-01",
                "verdict": "BUY",
            },
            "investment_analysis": {},
        }
        filepath = tmp_path / "7203_T_2026-03-01_analysis.json"
        filepath.write_text(json.dumps(data))
        load_latest_analyses(tmp_path)

        index_path = _analysis_index_path(tmp_path)
        payload = json.loads(index_path.read_text())
        payload["analyses"]["7203.T"]["source_size"] += 1
        index_path.write_text(json.dumps(payload))

        analyses = load_latest_analyses(tmp_path, progress=events.append)
        assert analyses["7203.T"].ticker == "7203.T"
        assert any(event.phase == "rebuilding_index" for event in events)

    def test_load_latest_analyses_ignores_stale_index_after_results_change(
        self, tmp_path
    ):
        """Adding a new analysis invalidates the old index and rebuilds it."""
        events: list[AnalysisLoadProgress] = []
        first = {
            "prediction_snapshot": {
                "ticker": "7203.T",
                "analysis_date": "2026-03-01",
                "verdict": "BUY",
            },
            "investment_analysis": {},
        }
        second = {
            "prediction_snapshot": {
                "ticker": "6758.T",
                "analysis_date": "2026-03-02",
                "verdict": "BUY",
            },
            "investment_analysis": {},
        }
        (tmp_path / "7203_T_2026-03-01_analysis.json").write_text(json.dumps(first))
        initial = load_latest_analyses(tmp_path)
        assert set(initial) == {"7203.T"}

        (tmp_path / "6758_T_2026-03-02_analysis.json").write_text(json.dumps(second))
        rebuilt = load_latest_analyses(tmp_path, progress=events.append)
        assert set(rebuilt) == {"7203.T", "6758.T"}
        assert any(
            event.phase == "rebuilding_index"
            and "stale_directory_state" in (event.current_file or "")
            for event in events
        )

    def test_load_latest_analyses_rebuilds_when_index_path_mismatches(self, tmp_path):
        """Path mismatch in the index payload is treated as untrusted and rebuilt."""
        events: list[AnalysisLoadProgress] = []
        data = {
            "prediction_snapshot": {
                "ticker": "7203.T",
                "analysis_date": "2026-03-01",
                "verdict": "BUY",
            },
            "investment_analysis": {},
        }
        filepath = tmp_path / "7203_T_2026-03-01_analysis.json"
        filepath.write_text(json.dumps(data))
        load_latest_analyses(tmp_path)

        index_path = _analysis_index_path(tmp_path)
        payload = json.loads(index_path.read_text())
        payload["results_dir"] = "/tmp/not-the-same-results-dir"
        index_path.write_text(json.dumps(payload))

        analyses = load_latest_analyses(tmp_path, progress=events.append)
        assert analyses["7203.T"].ticker == "7203.T"
        assert any(
            event.phase == "rebuilding_index"
            and "path_mismatch" in (event.current_file or "")
            for event in events
        )

    def test_load_latest_analyses_rebuilds_when_index_has_mixed_valid_and_invalid_entries(
        self, tmp_path
    ):
        """A single invalid entry invalidates the cache and full rebuild restores all entries."""
        events: list[AnalysisLoadProgress] = []
        for ticker, prefix in [("7203.T", "7203_T"), ("6758.T", "6758_T")]:
            data = {
                "prediction_snapshot": {
                    "ticker": ticker,
                    "analysis_date": "2026-03-01",
                    "verdict": "BUY",
                },
                "investment_analysis": {},
            }
            (tmp_path / f"{prefix}_2026-03-01_analysis.json").write_text(
                json.dumps(data)
            )

        load_latest_analyses(tmp_path)
        index_path = _analysis_index_path(tmp_path)
        payload = json.loads(index_path.read_text())
        payload["analyses"]["6758.T"]["source_file"] = str(tmp_path / "missing.json")
        index_path.write_text(json.dumps(payload))

        analyses = load_latest_analyses(tmp_path, progress=events.append)
        assert set(analyses) == {"7203.T", "6758.T"}
        assert any(event.phase == "rebuilding_index" for event in events)

    def test_incremental_index_update_after_save(self, tmp_path):
        """A valid cache index can be updated atomically after one new save."""
        first = {
            "prediction_snapshot": {
                "ticker": "7203.T",
                "analysis_date": "2026-03-01",
                "verdict": "BUY",
            },
            "investment_analysis": {},
        }
        first_path = tmp_path / "7203_T_2026-03-01_analysis.json"
        first_path.write_text(json.dumps(first))
        load_latest_analyses(tmp_path)

        previous_dir_mtime_ns = tmp_path.stat().st_mtime_ns

        second = {
            "prediction_snapshot": {
                "ticker": "6758.T",
                "analysis_date": "2026-03-02",
                "verdict": "BUY",
            },
            "investment_analysis": {},
        }
        second_path = tmp_path / "6758_T_2026-03-02_analysis.json"
        second_path.write_text(json.dumps(second))

        record = _build_analysis_record_from_data(second_path, second)
        assert record is not None
        assert (
            update_latest_analyses_index(
                tmp_path,
                record,
                previous_dir_mtime_ns=previous_dir_mtime_ns,
                analysis_file_count_before_save=1,
            )
            is True
        )

        rebuilt = load_latest_analyses(tmp_path)
        assert set(rebuilt) == {"7203.T", "6758.T"}

    def test_incremental_index_update_logs_success(self, tmp_path):
        """Successful incremental updates emit an explicit info event."""
        first = {
            "prediction_snapshot": {
                "ticker": "7203.T",
                "analysis_date": "2026-03-01",
                "verdict": "BUY",
            },
            "investment_analysis": {},
        }
        second = {
            "prediction_snapshot": {
                "ticker": "6758.T",
                "analysis_date": "2026-03-02",
                "verdict": "BUY",
            },
            "investment_analysis": {},
        }
        (tmp_path / "7203_T_2026-03-01_analysis.json").write_text(json.dumps(first))
        load_latest_analyses(tmp_path)

        previous_dir_mtime_ns = tmp_path.stat().st_mtime_ns
        second_path = tmp_path / "6758_T_2026-03-02_analysis.json"
        second_path.write_text(json.dumps(second))
        record = _build_analysis_record_from_data(second_path, second)
        assert record is not None

        with patch("src.ibkr.analysis_index.logger") as mock_logger:
            updated = update_latest_analyses_index(
                tmp_path,
                record,
                previous_dir_mtime_ns=previous_dir_mtime_ns,
                analysis_file_count_before_save=1,
            )

        assert updated is True
        mock_logger.info.assert_any_call(
            "analysis_index_incremental_updated",
            ticker="6758.T",
            path=str(_analysis_index_path(tmp_path)),
            source_file=str(second_path),
        )

    @pytest.mark.parametrize(
        ("reason", "previous_dir_mtime_ns", "mutate_index"),
        [
            ("missing_previous_dir_mtime", None, None),
            ("index_missing", 1, None),
            (
                "version_mismatch",
                "FROM_DIR",
                lambda payload, tmp_path: payload.__setitem__("version", -1),
            ),
            (
                "results_dir_mismatch",
                "FROM_DIR",
                lambda payload, tmp_path: payload.__setitem__(
                    "results_dir", str((tmp_path / "other").resolve())
                ),
            ),
            (
                "stale_directory_state",
                "FROM_DIR",
                lambda payload, tmp_path: payload.__setitem__(
                    "results_dir_mtime_ns",
                    int(payload["results_dir_mtime_ns"]) - 1,
                ),
            ),
        ],
    )
    def test_incremental_index_update_logs_skip_reasons(
        self,
        tmp_path,
        reason,
        previous_dir_mtime_ns,
        mutate_index,
    ):
        """Each guarded skip path emits a specific reason code."""
        seed = {
            "prediction_snapshot": {
                "ticker": "7203.T",
                "analysis_date": "2026-03-01",
                "verdict": "BUY",
            },
            "investment_analysis": {},
        }
        new_data = {
            "prediction_snapshot": {
                "ticker": "6758.T",
                "analysis_date": "2026-03-02",
                "verdict": "BUY",
            },
            "investment_analysis": {},
        }
        seed_path = tmp_path / "7203_T_2026-03-01_analysis.json"
        seed_path.write_text(json.dumps(seed))
        load_latest_analyses(tmp_path)

        if mutate_index is not None:
            index_path = _analysis_index_path(tmp_path)
            payload = json.loads(index_path.read_text())
            mutate_index(payload, tmp_path)
            index_path.write_text(json.dumps(payload))

        new_path = tmp_path / "6758_T_2026-03-02_analysis.json"
        new_path.write_text(json.dumps(new_data))
        record = _build_analysis_record_from_data(new_path, new_data)
        assert record is not None

        if reason == "index_missing":
            _analysis_index_path(tmp_path).unlink()
            effective_previous_dir_mtime_ns = tmp_path.stat().st_mtime_ns
        elif previous_dir_mtime_ns == "FROM_DIR":
            effective_previous_dir_mtime_ns = tmp_path.stat().st_mtime_ns
        else:
            effective_previous_dir_mtime_ns = previous_dir_mtime_ns

        with patch("src.ibkr.analysis_index.logger") as mock_logger:
            updated = update_latest_analyses_index(
                tmp_path,
                record,
                previous_dir_mtime_ns=effective_previous_dir_mtime_ns,
                analysis_file_count_before_save=(
                    0 if reason == "stale_directory_state" else None
                ),
            )

        assert updated is False
        skip_calls = [
            call
            for call in mock_logger.info.call_args_list
            if call.args and call.args[0] == "analysis_index_incremental_update_skipped"
        ]
        assert skip_calls
        matched = next(
            (
                call
                for call in skip_calls
                if call.kwargs["ticker"] == "6758.T"
                and call.kwargs["path"] == str(_analysis_index_path(tmp_path))
                and call.kwargs["reason"] == reason
            ),
            None,
        )
        assert matched is not None
        if reason == "stale_directory_state":
            assert (
                matched.kwargs["expected_previous_dir_mtime_ns"]
                == effective_previous_dir_mtime_ns
            )
            assert "index_dir_mtime_ns" in matched.kwargs
            assert "current_dir_mtime_ns" in matched.kwargs
            assert matched.kwargs["source_file"] == str(new_path)

    def test_leftover_lock_file_does_not_block_index_use(self, tmp_path):
        """An empty leftover lock file is harmless when no process holds the lock."""
        data = {
            "prediction_snapshot": {
                "ticker": "7203.T",
                "analysis_date": "2026-03-01",
                "verdict": "BUY",
            },
            "investment_analysis": {},
        }
        (tmp_path / "7203_T_2026-03-01_analysis.json").write_text(json.dumps(data))
        load_latest_analyses(tmp_path)

        lock_path = _analysis_index_lock_path(tmp_path)
        lock_path.write_text("")

        analyses = load_latest_analyses(tmp_path)
        assert analyses["7203.T"].ticker == "7203.T"

    def test_incremental_index_update_waits_for_existing_lock(self, tmp_path):
        """Concurrent writers serialize on the whole-index lock."""
        first = {
            "prediction_snapshot": {
                "ticker": "7203.T",
                "analysis_date": "2026-03-01",
                "verdict": "BUY",
            },
            "investment_analysis": {},
        }
        second = {
            "prediction_snapshot": {
                "ticker": "6758.T",
                "analysis_date": "2026-03-02",
                "verdict": "BUY",
            },
            "investment_analysis": {},
        }
        (tmp_path / "7203_T_2026-03-01_analysis.json").write_text(json.dumps(first))
        load_latest_analyses(tmp_path)

        previous_dir_mtime_ns = tmp_path.stat().st_mtime_ns
        second_path = tmp_path / "6758_T_2026-03-02_analysis.json"
        second_path.write_text(json.dumps(second))
        record = _build_analysis_record_from_data(second_path, second)
        assert record is not None

        # Launch the lock-holder as a posix_spawn subprocess (close_fds=False, no
        # cwd=) rather than a multiprocessing "spawn" Process: the latter calls
        # fork(), which SIGSEGVs in Apple's Network.framework atfork handler on
        # macOS once the gRPC stack is loaded. Readiness is signaled via a marker
        # file. See CLAUDE.md (macOS-Specific Issues).
        ready_path = tmp_path / ".lock_holder_ready"
        child_env = {**os.environ}
        child_env["PYTHONPATH"] = (
            str(_REPO_ROOT) + os.pathsep + child_env["PYTHONPATH"]
            if child_env.get("PYTHONPATH")
            else str(_REPO_ROOT)
        )
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "tests.ibkr.lock_helpers",
                str(tmp_path),
                "0.4",
                str(ready_path),
            ],
            env=child_env,
            close_fds=False,
        )
        try:
            deadline = time.time() + 2.0
            while not ready_path.exists() and time.time() < deadline:
                time.sleep(0.01)
            assert ready_path.exists(), "lock holder did not signal readiness"

            start = time.perf_counter()
            with patch("src.ibkr.analysis_index.logger") as mock_logger:
                updated = update_latest_analyses_index(
                    tmp_path,
                    record,
                    previous_dir_mtime_ns=previous_dir_mtime_ns,
                    analysis_file_count_before_save=1,
                )
            elapsed = time.perf_counter() - start
        finally:
            proc.wait(timeout=2.0)
        assert proc.returncode == 0
        assert updated is True
        assert elapsed >= 0.25
        mock_logger.debug.assert_any_call(
            "analysis_index_lock_waiting",
            path=str(_analysis_index_lock_path(tmp_path)),
        )
        acquire_calls = [
            call
            for call in mock_logger.debug.call_args_list
            if call.args and call.args[0] == "analysis_index_lock_acquired"
        ]
        assert acquire_calls
        assert acquire_calls[-1].kwargs["path"] == str(
            _analysis_index_lock_path(tmp_path)
        )
        assert acquire_calls[-1].kwargs["wait_secs"] >= 0.0

    def test_incremental_index_update_accepts_mtime_mismatch_when_file_count_matches(
        self, tmp_path
    ):
        """Append-only saves can proceed when file count proves the index is current."""
        first = {
            "prediction_snapshot": {
                "ticker": "7203.T",
                "analysis_date": "2026-03-01",
                "verdict": "BUY",
            },
            "investment_analysis": {},
        }
        second = {
            "prediction_snapshot": {
                "ticker": "6758.T",
                "analysis_date": "2026-03-02",
                "verdict": "BUY",
            },
            "investment_analysis": {},
        }
        (tmp_path / "7203_T_2026-03-01_analysis.json").write_text(json.dumps(first))
        load_latest_analyses(tmp_path)

        index_path = _analysis_index_path(tmp_path)
        payload = json.loads(index_path.read_text())
        payload["results_dir_mtime_ns"] = int(payload["results_dir_mtime_ns"]) - 1
        index_path.write_text(json.dumps(payload))

        previous_dir_mtime_ns = tmp_path.stat().st_mtime_ns
        second_path = tmp_path / "6758_T_2026-03-02_analysis.json"
        second_path.write_text(json.dumps(second))
        record = _build_analysis_record_from_data(second_path, second)
        assert record is not None

        with patch("src.ibkr.analysis_index.logger") as mock_logger:
            updated = update_latest_analyses_index(
                tmp_path,
                record,
                previous_dir_mtime_ns=previous_dir_mtime_ns,
                analysis_file_count_before_save=1,
            )

        assert updated is True
        # The accept is logged at debug (intentionally quiet); the index still
        # updates incrementally because the pre-save file count proves it current.
        mock_logger.debug.assert_any_call(
            "analysis_index_incremental_update_mtime_mismatch_accepted",
            ticker="6758.T",
            path=str(index_path),
            expected_previous_dir_mtime_ns=previous_dir_mtime_ns,
            index_dir_mtime_ns=payload["results_dir_mtime_ns"],
            analysis_file_count_before_save=1,
            indexed_total_files=1,
            source_file=str(second_path),
        )

    def test_malformed_index_logs_exception_details(self, tmp_path):
        """Index load failures should preserve exception type information."""
        index_path = _analysis_index_path(tmp_path)
        index_path.write_text("{not-json")

        with patch("src.ibkr.analysis_index.logger") as mock_logger:
            analyses = load_latest_analyses(tmp_path)

        assert analyses == {}
        mock_logger.warning.assert_any_call(
            "analysis_index_load_failed",
            path=str(index_path),
            operation="loading latest analyses index",
            error_type="JSONDecodeError",
            root_cause_type="JSONDecodeError",
            failure_kind=ANY,
            retryable=ANY,
            host=ANY,
            message_preview=ANY,
        )

    def test_unparseable_analysis_log_redacts_sensitive_exception_text(self, tmp_path):
        """Snapshot parse failures must not log raw provider keys or bearer tokens."""
        secret_text = (
            "api_key=sk-testsecret1234567890 Bearer ya29.fakeSensitiveToken123456"
        )
        analysis_path = tmp_path / "7203.T_2026-03-01_analysis.json"
        analysis_path.write_text("{}")

        with (
            patch(
                "src.ibkr.analysis_index._build_analysis_record_from_file",
                side_effect=OSError(secret_text),
            ),
            patch("src.ibkr.analysis_index.logger") as mock_logger,
        ):
            analyses = load_latest_analyses(tmp_path)

        assert analyses == {}
        warning_calls = [
            call
            for call in mock_logger.warning.call_args_list
            if call.args and call.args[0] == "analysis_file_unparseable"
        ]
        assert warning_calls
        logged_text = repr(warning_calls[0])
        assert "sk-testsecret" not in logged_text
        assert "fakeSensitiveToken" not in logged_text
        assert "[REDACTED]" in logged_text

    def test_filename_duplicate_history_is_skipped_before_json_load(self, tmp_path):
        """Older files with the same filename-level ticker key are skipped early."""
        newest = {
            "prediction_snapshot": {
                "ticker": "7203.T",
                "analysis_date": "2026-03-02",
                "verdict": "BUY",
            },
            "investment_analysis": {},
        }
        older = {
            "prediction_snapshot": {
                "ticker": "7203.T",
                "analysis_date": "2026-03-01",
                "verdict": "SELL",
            },
            "investment_analysis": {},
        }
        (tmp_path / "7203_T_2026-03-02_analysis.json").write_text(json.dumps(newest))
        (tmp_path / "7203_T_2026-03-01_analysis.json").write_text(json.dumps(older))

        original_json_load = json.load

        with patch("src.ibkr.analysis_index.json.load") as mock_json_load:
            mock_json_load.side_effect = original_json_load
            analyses = load_latest_analyses(tmp_path)

        assert mock_json_load.call_count == 1
        assert analyses["7203.T"].verdict == "BUY"

    def test_filename_dedupe_does_not_hide_older_valid_file_when_newest_is_bad(
        self, tmp_path
    ):
        """A malformed newest file must not block a valid older file with the same key."""
        valid = {
            "prediction_snapshot": {
                "ticker": "7203.T",
                "analysis_date": "2026-03-01",
                "verdict": "BUY",
            },
            "investment_analysis": {},
        }
        (tmp_path / "7203_T_2026-03-02_analysis.json").write_text("not json{{{")
        (tmp_path / "7203_T_2026-03-01_analysis.json").write_text(json.dumps(valid))

        analyses = load_latest_analyses(tmp_path)

        assert analyses["7203.T"].verdict == "BUY"


# ── Concentration Tests ──


class TestConcentration:
    """Test sector/exchange concentration tracking."""

    def test_exchange_weights_populated_after_reconcile(self):
        """reconcile() populates portfolio.exchange_weights from held positions.

        Weights are % of deployed equity (sum of position values), not % of
        total portfolio value (which includes cash and may use a different FX rate).
        Positions: HK=$40k, JP=$20k → total $60k → HK=66.7%, JP=33.3%.
        """
        pos_hk = _make_position(
            ticker="0005.HK", market_value_usd=40000, currency="HKD"
        )
        pos_jp = _make_position(ticker="7203.T", market_value_usd=20000, currency="JPY")
        analyses = {
            "0005.HK": _make_analysis(ticker="0005.HK"),
            "7203.T": _make_analysis(ticker="7203.T"),
        }
        portfolio = _make_portfolio(value=100000, cash=15000)
        reconcile([pos_hk, pos_jp], analyses, portfolio)

        assert "HK" in portfolio.exchange_weights
        assert "T" in portfolio.exchange_weights
        # Denominator is sum of positions ($60k), not ledger value ($100k)
        assert abs(portfolio.exchange_weights["HK"] - 66.7) < 0.5
        assert abs(portfolio.exchange_weights["T"] - 33.3) < 0.5
        # Weights must sum to 100%
        assert abs(sum(portfolio.exchange_weights.values()) - 100.0) < 0.1

    def test_concentration_warning_in_buy_reason(self):
        """BUY reason includes ⚠ when projected exchange weight exceeds limit."""
        # Already have 35% in HK; adding another HK stock would exceed 40% limit
        pos_hk = _make_position(
            ticker="0005.HK", market_value_usd=35000, currency="HKD"
        )
        analysis_existing = _make_analysis(ticker="0005.HK", verdict="BUY")
        # New BUY candidate also on HK
        analysis_new = _make_analysis(ticker="2388.HK", verdict="BUY", age_days=3)
        analysis_new.exchange = "HK"

        portfolio = _make_portfolio(value=100000, cash=20000)
        items = reconcile(
            [pos_hk],
            {"0005.HK": analysis_existing, "2388.HK": analysis_new},
            portfolio,
            exchange_limit_pct=40.0,
        )
        buy_items = [i for i in items if i.action == "BUY" and i.ticker.yf == "2388.HK"]
        if buy_items:
            # If a BUY item was generated, it should have a concentration warning
            assert "⚠" in buy_items[0].reason or "HK" in buy_items[0].reason

    def test_no_concentration_when_portfolio_empty(self):
        """Empty portfolio → no exchange weights."""
        portfolio = _make_portfolio(value=0, cash=0)
        reconcile([], {}, portfolio)
        assert portfolio.exchange_weights == {}
        assert portfolio.sector_weights == {}

    def test_concentration_sums_to_100(self):
        """
        Sector and exchange weights always sum to ~100% even when position market
        values and the ledger portfolio_value_usd use different FX rates (stale rates
        vs live rates). The denominator is now sum-of-positions, not ledger total.
        """
        # Simulate stale-FX scenario: positions sum to $60k but ledger says $71k
        pos_hk = _make_position(
            ticker="0005.HK", market_value_usd=40000, currency="HKD", conid=1
        )
        pos_jp = _make_position(
            ticker="7203.T", market_value_usd=20000, currency="JPY", conid=2
        )
        analyses = {
            "0005.HK": _make_analysis(ticker="0005.HK", verdict="BUY", size_pct=5.0),
            "7203.T": _make_analysis(ticker="7203.T", verdict="BUY", size_pct=5.0),
        }
        # Ledger value intentionally higher than sum of positions (simulates FX drift)
        portfolio = _make_portfolio(value=71000, cash=15000)
        reconcile([pos_hk, pos_jp], analyses, portfolio)

        total_exchange = sum(portfolio.exchange_weights.values())
        total_sector = sum(portfolio.sector_weights.values())
        assert total_exchange == pytest.approx(100.0, abs=0.1)
        assert total_sector == pytest.approx(100.0, abs=0.1)


class TestExchangeFromPosition:
    """Unit tests for _exchange_from_position() helper."""

    def _pos(self, yf_ticker: str, ibkr_exchange: str = "", currency: str = "USD"):
        if "." in yf_ticker:
            ticker = Ticker.from_yf(yf_ticker, currency=currency)
        else:
            ticker = Ticker.from_ibkr(
                yf_ticker, exchange=ibkr_exchange, currency=currency
            )
        return NormalizedPosition(
            conid=1,
            ticker=ticker,
            quantity=1,
            avg_cost_local=100.0,
            market_value_usd=100.0,
            currency=currency,
            current_price_local=100.0,
        )

    def test_suffix_present_takes_priority(self):
        """yf_ticker with .T suffix → 'T', regardless of IBKR exchange field."""
        pos = self._pos("7203.T", ibkr_exchange="KLSE", currency="MYR")
        assert _exchange_from_position(pos) == "T"

    def test_ibkr_exchange_used_when_no_suffix(self):
        """No suffix but IBKR listingExchange=KLSE → 'KL', not 'US'."""
        pos = self._pos("MEGP", ibkr_exchange="KLSE", currency="MYR")
        assert _exchange_from_position(pos) == "KL"

    def test_ibkr_tsej_maps_to_t(self):
        """IBKR Client Portal Tokyo code 'TSEJ' → 'T'."""
        pos = self._pos("7203", ibkr_exchange="TSEJ", currency="JPY")
        assert _exchange_from_position(pos) == "T"

    def test_ibkr_tse_maps_to_to(self):
        """IBKR Client Portal Toronto code 'TSE' → 'TO' (not Tokyo's 'T')."""
        pos = self._pos("PEY", ibkr_exchange="TSE", currency="CAD")
        assert _exchange_from_position(pos) == "TO"

    def test_ibkr_wse_maps_to_wa(self):
        """IBKR WSE (Warsaw Stock Exchange) → 'WA'."""
        pos = self._pos("FPE3", ibkr_exchange="WSE", currency="PLN")
        assert _exchange_from_position(pos) == "WA"

    def test_ibkr_nasdaq_maps_to_us(self):
        """IBKR NASDAQ → 'US' (US exchange, empty suffix)."""
        pos = self._pos("AAPL", ibkr_exchange="NASDAQ", currency="USD")
        assert _exchange_from_position(pos) == "US"

    def test_currency_fallback_sek_maps_to_st(self):
        """No suffix, no known IBKR exchange, but currency=SEK → 'ST'."""
        pos = self._pos("KRN", ibkr_exchange="SFB", currency="SEK")
        # SFB is not in IBKR_TO_YFINANCE; should fall through to currency heuristic
        assert _exchange_from_position(pos) == "ST"

    def test_currency_fallback_myr_maps_to_kl(self):
        """No suffix, unknown IBKR exchange, currency=MYR → 'KL'."""
        pos = self._pos("MEGP", ibkr_exchange="MESDAQ", currency="MYR")
        assert _exchange_from_position(pos) == "KL"

    def test_unknown_everything_returns_us(self):
        """No suffix, unknown exchange, USD currency → 'US' as safe default."""
        pos = self._pos("THING", ibkr_exchange="", currency="USD")
        assert _exchange_from_position(pos) == "US"

    def test_gbp_maps_to_l(self):
        """GBP currency → 'L' (UK), not 'US'. MEGP, GAMA, KLR should not appear in US bucket."""
        pos = self._pos("MEGP", ibkr_exchange="", currency="GBP")
        assert _exchange_from_position(pos) == "L"

    def test_gbx_maps_to_l(self):
        """GBX (pence, after LSE conversion) → 'L'."""
        pos = self._pos("GAMA", ibkr_exchange="", currency="GBX")
        assert _exchange_from_position(pos) == "L"

    def test_cad_maps_to_to(self):
        """CAD → 'TO' (Canada TSX). MTL, NWC, TXG must not appear in US bucket."""
        pos = self._pos("MTL", ibkr_exchange="", currency="CAD")
        assert _exchange_from_position(pos) == "TO"

    def test_chf_maps_to_sw(self):
        """CHF → 'SW' (Switzerland SIX)."""
        pos = self._pos("NESN", ibkr_exchange="", currency="CHF")
        assert _exchange_from_position(pos) == "SW"

    def test_eur_maps_to_eur(self):
        """EUR → 'EUR' (European bucket). Better than 'US' for German/French/etc. stocks."""
        pos = self._pos("FPE3", ibkr_exchange="", currency="EUR")
        assert _exchange_from_position(pos) == "EUR"

    def test_exchange_weight_no_us_for_myr_position(self):
        """Regression: suffix-less Malaysian position must NOT count toward 'US' weight.

        Before fix: _analysis.exchange='US' (stored wrong) + no suffix on yf_ticker
        → misclassified as US.  After fix: currency=MYR → 'KL'.
        """
        pos = NormalizedPosition(
            conid=99,
            ticker=Ticker.from_ibkr("MEGP", exchange="KLSE", currency="MYR"),
            quantity=100,
            avg_cost_local=1.4,
            market_value_usd=140.0,
            currency="MYR",
            current_price_local=1.35,
        )
        analysis = _make_analysis(ticker="MEGP")
        analysis.exchange = "US"  # wrong value stored in snapshot

        portfolio = _make_portfolio(value=1000, cash=860)
        reconcile([pos], {"MEGP": analysis}, portfolio)

        assert (
            "KL" in portfolio.exchange_weights
        ), "Malaysian position must be KL, not US"
        assert "US" not in portfolio.exchange_weights, "Must not be misclassified as US"


class TestStalenessDisplay:
    """Test that staleness reason strings are human-readable."""

    def test_normal_age_shows_days(self):
        analysis = _make_analysis(age_days=20)
        is_stale, reason = check_staleness(analysis, max_age_days=14)
        assert is_stale
        assert "20d" in reason
        assert "9999" not in reason

    def test_missing_date_shows_no_date(self):
        """age_days=9999 sentinel (missing/malformed analysis_date) → 'no date'."""
        analysis = AnalysisRecord(
            ticker="TEST.T",
            analysis_date="",  # missing date → age_days = 9999
        )
        assert analysis.age_days >= 9999
        is_stale, reason = check_staleness(analysis, max_age_days=14)
        assert is_stale
        assert "no date" in reason
        assert "9999" not in reason


# ── Portfolio Health Tests ──


class TestComputePortfolioHealth:
    """Test portfolio-level health flag computation."""

    def test_no_flags_for_healthy_portfolio(self):
        """Healthy scores, fresh analyses, diversified currency → no flags."""
        pos = _make_position(market_value_usd=10000, currency="JPY")
        analysis = _make_analysis(age_days=5)
        analysis.health_adj = 75.0
        analysis.growth_adj = 70.0
        portfolio = _make_portfolio(value=100000)
        portfolio.exchange_weights = {"T": 10.0}
        flags = compute_portfolio_health([pos], {"7203.T": analysis}, portfolio)
        assert flags == []

    def test_low_health_average_flag(self):
        """Weighted avg health < 60 → LOW_HEALTH_AVERAGE flag."""
        pos = _make_position(market_value_usd=10000, currency="JPY")
        analysis = _make_analysis(age_days=5)
        analysis.health_adj = 45.0  # below 60
        analysis.growth_adj = 70.0
        portfolio = _make_portfolio(value=100000)
        portfolio.exchange_weights = {}
        flags = compute_portfolio_health([pos], {"7203.T": analysis}, portfolio)
        assert any("LOW_HEALTH" in f for f in flags)

    def test_low_growth_average_flag(self):
        """Weighted avg growth < 55 → LOW_GROWTH_AVERAGE flag."""
        pos = _make_position(market_value_usd=10000, currency="JPY")
        analysis = _make_analysis(age_days=5)
        analysis.health_adj = 70.0
        analysis.growth_adj = 40.0  # below 55
        portfolio = _make_portfolio(value=100000)
        portfolio.exchange_weights = {}
        flags = compute_portfolio_health([pos], {"7203.T": analysis}, portfolio)
        assert any("LOW_GROWTH" in f for f in flags)

    def test_stale_analysis_ratio_flag(self):
        """More than 30% of positions stale → STALE_ANALYSIS_RATIO flag."""
        pos1 = _make_position(
            ticker="7203.T", market_value_usd=5000, currency="JPY", conid=1
        )
        pos2 = _make_position(
            ticker="0005.HK", market_value_usd=5000, currency="HKD", conid=2
        )
        old_analysis = _make_analysis(ticker="0005.HK", age_days=20)
        fresh_analysis = _make_analysis(ticker="7203.T", age_days=3)
        portfolio = _make_portfolio(value=100000)
        portfolio.exchange_weights = {}
        analyses = {"7203.T": fresh_analysis, "0005.HK": old_analysis}
        flags = compute_portfolio_health(
            [pos1, pos2], analyses, portfolio, max_age_days=14
        )
        assert any("STALE" in f for f in flags)

    def test_stale_analysis_ratio_with_reconciliation_breakdown(self):
        """When reconciliation items are present, stale flags distinguish queue vs blocking."""
        pos1 = _make_position(
            ticker="7203.T", market_value_usd=5000, currency="JPY", conid=1
        )
        pos2 = _make_position(
            ticker="0005.HK", market_value_usd=5000, currency="HKD", conid=2
        )
        stale_a = _make_analysis(ticker="7203.T", age_days=20)
        stale_b = _make_analysis(ticker="0005.HK", age_days=20)
        portfolio = _make_portfolio(value=100000)
        portfolio.exchange_weights = {}
        items = [
            ReconciliationItem(
                ticker="7203.T",
                action="REVIEW",
                reason="Stale analysis: age 20d > max_age_days 14",
                urgency="MEDIUM",
                ibkr_position=pos1,
                analysis=stale_a,
            ),
            ReconciliationItem(
                ticker="0005.HK",
                action="REVIEW",
                reason="Verdict → DO_NOT_INITIATE  (2026-03-05)",
                urgency="MEDIUM",
                ibkr_position=pos2,
                analysis=stale_b,
                sell_type="SOFT_REJECT",
            ),
        ]
        flags = compute_portfolio_health(
            [pos1, pos2],
            {"7203.T": stale_a, "0005.HK": stale_b},
            portfolio,
            max_age_days=14,
            reconciliation_items=items,
        )
        stale_flag = next(flag for flag in flags if "STALE_ANALYSIS_RATIO" in flag)
        assert "1 already in sell/review queue" in stale_flag
        assert "1 still need refreshed analysis before action" in stale_flag

    def test_stale_analysis_ratio_without_reconciliation_keeps_legacy_message(self):
        """Older call sites without reconciliation items keep the legacy wording."""
        pos1 = _make_position(
            ticker="7203.T", market_value_usd=5000, currency="JPY", conid=1
        )
        pos2 = _make_position(
            ticker="0005.HK", market_value_usd=5000, currency="HKD", conid=2
        )
        stale_a = _make_analysis(ticker="7203.T", age_days=20)
        stale_b = _make_analysis(ticker="0005.HK", age_days=20)
        portfolio = _make_portfolio(value=100000)
        portfolio.exchange_weights = {}
        flags = compute_portfolio_health(
            [pos1, pos2],
            {"7203.T": stale_a, "0005.HK": stale_b},
            portfolio,
            max_age_days=14,
        )
        stale_flag = next(flag for flag in flags if "STALE_ANALYSIS_RATIO" in flag)
        assert "flying blind on significant chunk of portfolio" in stale_flag

    def test_currency_concentration_flag(self):
        """More than 50% in a single currency → CURRENCY_CONCENTRATION flag."""
        pos = _make_position(market_value_usd=60000, currency="HKD")
        analysis = _make_analysis(age_days=5)
        analysis.health_adj = 70.0
        analysis.growth_adj = 70.0
        portfolio = _make_portfolio(value=100000)
        portfolio.exchange_weights = {}
        flags = compute_portfolio_health([pos], {"7203.T": analysis}, portfolio)
        assert any("CURRENCY_CONCENTRATION" in f for f in flags)

    def test_empty_positions_returns_no_flags(self):
        """No positions → no flags."""
        portfolio = _make_portfolio(value=0)
        flags = compute_portfolio_health([], {}, portfolio)
        assert flags == []

    def test_low_health_flag_includes_worst_contributors(self):
        """LOW_HEALTH_AVERAGE flag lists the tickers with the lowest health scores."""
        pos1 = _make_position(ticker="7203.T", market_value_usd=5000, conid=1)
        pos2 = _make_position(ticker="0005.HK", market_value_usd=5000, conid=2)
        a1 = _make_analysis(ticker="7203.T", age_days=3)
        a1.health_adj = 20.0  # very low
        a2 = _make_analysis(ticker="0005.HK", age_days=3)
        a2.health_adj = 30.0  # also low
        portfolio = _make_portfolio(value=100000)
        portfolio.exchange_weights = {}
        flags = compute_portfolio_health(
            [pos1, pos2], {"7203.T": a1, "0005.HK": a2}, portfolio
        )
        health_flag = next(f for f in flags if "LOW_HEALTH" in f)
        # Flag must be multi-line with ticker scores embedded
        assert "7203.T" in health_flag
        assert "0005.HK" in health_flag
        assert "Lowest:" in health_flag

    def test_stale_positions_marked_in_health_flag(self):
        """Stale analyses are marked with † and a caveat is appended."""
        pos = _make_position(ticker="7203.T", market_value_usd=10000, conid=1)
        analysis = _make_analysis(ticker="7203.T", age_days=20)  # stale
        analysis.health_adj = 25.0
        portfolio = _make_portfolio(value=100000)
        portfolio.exchange_weights = {}
        flags = compute_portfolio_health(
            [pos], {"7203.T": analysis}, portfolio, max_age_days=14
        )
        health_flag = next(f for f in flags if "LOW_HEALTH" in f)
        assert "†" in health_flag
        assert "stale" in health_flag


# ── Sell Type Classification Tests ──


class TestSellTypeTagging:
    """Test that SELL items are tagged with the correct sell_type."""

    def test_stop_breach_tagged_stop_breach(self):
        """Price below the review level → sell_type=STOP_BREACH, review-only."""
        pos = _make_position(current_price=1700)
        analysis = _make_analysis(stop_price=1900)
        items = reconcile([pos], {"7203.T": analysis}, _make_portfolio())
        assert items[0].action == "REVIEW"
        assert items[0].sell_type == "STOP_BREACH"

    def test_fundamental_failure_tagged_hard_reject(self):
        """Verdict DO_NOT_INITIATE + health_adj=40 → sell_type HARD_REJECT;
        unconfirmed, so the disposition is REVIEW (thesis reassessment)."""
        pos = _make_position(current_price=2100)
        analysis = _make_analysis(verdict="DO_NOT_INITIATE")
        analysis.health_adj = 40.0  # below 50 → hard fail
        analysis.growth_adj = 60.0
        items = reconcile([pos], {"7203.T": analysis}, _make_portfolio())
        item = next(i for i in items if i.sell_type == "HARD_REJECT")
        assert item.action == "REVIEW"
        assert item.action_basis == "THESIS_REASSESSMENT"

    def test_soft_failure_tagged_soft_reject(self):
        """Intact fundamentals at/above entry → sell_type SOFT_REJECT with an
        ENTRY_CONSTRAINT review — the screen rejected the price, not the thesis."""
        pos = _make_position(current_price=2100)
        analysis = _make_analysis(verdict="DO_NOT_INITIATE")
        analysis.health_adj = 65.0  # passes hard check
        analysis.growth_adj = 60.0  # passes hard check
        items = reconcile([pos], {"7203.T": analysis}, _make_portfolio())
        item = next(i for i in items if i.sell_type == "SOFT_REJECT")
        assert item.action == "REVIEW"
        assert item.action_basis == "ENTRY_CONSTRAINT"

    def test_no_analysis_tagged_hard_reject(self):
        """Held position with no analysis → REVIEW (not SELL), so no sell_type."""
        pos = _make_position(current_price=2100)
        items = reconcile([pos], {}, _make_portfolio())
        # No analysis means REVIEW, not SELL
        assert items[0].action == "REVIEW"
        assert items[0].sell_type is None

    def test_hold_has_no_sell_type(self):
        """HOLD items have sell_type=None."""
        pos = _make_position(current_price=2100)
        analysis = _make_analysis(verdict="BUY")
        items = reconcile([pos], {"7203.T": analysis}, _make_portfolio())
        assert items[0].action == "HOLD"
        assert items[0].sell_type is None


class TestProfitTakeClassification:
    """Profit-taking is capital-allocation driven, not a fundamentals-failure sell."""

    def test_severe_idle_cash_long_term_is_advisory_review(self):
        """Profit-taking is always advisory (July 2026) — even a long-term lot
        with severe idle-cash flags surfaces as REVIEW, never an order."""
        pos = _make_position(
            avg_cost=2000,
            current_price=2600,
            tax_term="LONG_TERM",
        )
        analysis = _make_analysis(
            target_1=2500,
            capital_flag_types=("CAPITAL_IDLE_CASH_SEVERE",),
        )

        items = reconcile([pos], {"7203.T": analysis}, _make_portfolio())

        item = items[0]
        assert item.action == "REVIEW"
        assert item.sell_type == "PROFIT_TAKE"
        assert item.cost_basis_return_pct == pytest.approx(30.0)
        assert "capital_idle_cash_severe" in item.profit_take_reasons
        assert item.suggested_quantity is None
        assert item.cash_impact_usd == 0.0
        assert "advisory" in item.reason

    def test_unknown_holding_period_defaults_to_review(self):
        pos = _make_position(avg_cost=2000, current_price=2600)
        analysis = _make_analysis(
            target_1=2500,
            capital_flag_types=("CAPITAL_IDLE_CASH_SEVERE",),
        )

        items = reconcile([pos], {"7203.T": analysis}, _make_portfolio())

        item = items[0]
        assert item.action == "REVIEW"
        assert item.sell_type == "PROFIT_TAKE"
        assert "unknown_tax_term" in item.profit_take_reasons
        assert item.suggested_quantity is None
        assert item.cash_impact_usd == 0.0

    def test_unknown_holding_period_severe_large_gain_is_advisory_review(self):
        """No severity override anymore — an unknown holding period with a
        severe flag and a large gain is still an advisory REVIEW (July 2026)."""
        pos = _make_position(avg_cost=2000, current_price=3400)
        analysis = _make_analysis(
            target_1=4000,
            capital_flag_types=("CAPITAL_IDLE_CASH_SEVERE",),
        )

        items = reconcile([pos], {"7203.T": analysis}, _make_portfolio())

        item = items[0]
        assert item.action == "REVIEW"
        assert item.sell_type == "PROFIT_TAKE"
        assert item.cost_basis_return_pct == pytest.approx(70.0)
        assert "capital_idle_cash_severe" in item.profit_take_reasons
        assert "unknown_tax_term" in item.profit_take_reasons
        assert item.suggested_quantity is None
        assert item.cash_impact_usd == 0.0

    def test_short_term_holding_period_defaults_to_review(self):
        pos = _make_position(
            avg_cost=2000,
            current_price=2600,
            tax_term="SHORT_TERM",
        )
        analysis = _make_analysis(
            target_1=2500,
            capital_flag_types=("CAPITAL_IDLE_CASH_SEVERE",),
        )

        items = reconcile([pos], {"7203.T": analysis}, _make_portfolio())

        item = items[0]
        assert item.action == "REVIEW"
        assert item.sell_type == "PROFIT_TAKE"
        assert "short_term_tax" in item.profit_take_reasons

    def test_reference_reached_without_capital_flag_keeps_existing_review(self):
        pos = _make_position(avg_cost=2000, current_price=2600)
        analysis = _make_analysis(target_1=2500)

        items = reconcile([pos], {"7203.T": analysis}, _make_portfolio())

        item = items[0]
        assert item.action == "REVIEW"
        assert item.sell_type is None
        assert item.reason.startswith("Base-case valuation reference reached:")

    def test_idle_cash_risk_needs_target_hit_or_large_gain(self):
        # entry_price near current isolates this from drift-staleness: the point is
        # that a RISK flag with neither a target hit nor a large (>=50%) gain does not
        # profit-take — gain here is +27.5% off cost, below the large-gain bar.
        pos = _make_position(avg_cost=2000, current_price=2550)
        analysis = _make_analysis(
            entry_price=2500,
            target_1=3000,
            capital_flag_types=("CAPITAL_IDLE_CASH_RISK",),
        )

        items = reconcile([pos], {"7203.T": analysis}, _make_portfolio())

        assert items[0].action == "HOLD"
        assert items[0].sell_type is None

    def test_idle_cash_risk_with_large_gain_qualifies_without_target_hit(self):
        pos = _make_position(
            avg_cost=2000,
            current_price=3100,
            tax_term="LONG_TERM",
        )
        analysis = _make_analysis(
            target_1=3500,
            capital_flag_types=("CAPITAL_IDLE_CASH_RISK",),
        )

        items = reconcile([pos], {"7203.T": analysis}, _make_portfolio())

        item = items[0]
        assert item.action == "REVIEW"
        assert item.sell_type == "PROFIT_TAKE"
        assert "capital_idle_cash_risk_plus_large_gain" in item.profit_take_reasons

    def test_profit_take_requires_intact_business_quality(self):
        pos = _make_position(
            avg_cost=2000,
            current_price=2600,
            tax_term="LONG_TERM",
        )
        analysis = _make_analysis(
            target_1=2500,
            capital_flag_types=("CAPITAL_IDLE_CASH_SEVERE",),
        )
        analysis.growth_adj = 45.0

        items = reconcile([pos], {"7203.T": analysis}, _make_portfolio())

        assert items[0].action == "REVIEW"
        assert items[0].sell_type is None

    def test_price_break_priority_beats_profit_take(self):
        """A price break outranks the profit-take path — the urgent review
        fires before any capital-allocation advisory."""
        pos = _make_position(
            avg_cost=1000,
            current_price=1800,
            tax_term="LONG_TERM",
        )
        analysis = _make_analysis(
            stop_price=1900,
            target_1=1700,
            capital_flag_types=("CAPITAL_IDLE_CASH_SEVERE",),
        )

        items = reconcile([pos], {"7203.T": analysis}, _make_portfolio())

        assert items[0].action == "REVIEW"
        assert items[0].sell_type == "STOP_BREACH"


# ── Correlated Sell Event Detection Tests ──


def _make_multi_sell_scenario(
    n_soft_sells: int,
    n_stop_breaches: int,
    n_hard_rejects: int,
    n_holds: int,
    sell_date: str | None = None,
) -> tuple[list, dict, "PortfolioSummary"]:
    """
    Build positions + analyses for a correlated-sell scenario.

    Returns (positions, analyses, portfolio) ready for reconcile().

    ``sell_date`` defaults to a recent date (within the macro override window) so
    the demotion fires; pass an older date to exercise the lapsed-override path.
    """
    from datetime import datetime, timedelta

    if sell_date is None:
        sell_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")

    positions = []
    analyses = {}
    conid = 1

    # SOFT_REJECT rejections — passed both hard checks, rejected on soft tally.
    # Priced BELOW entry (2100 → 1950, above the 1900 stop): genuine macro
    # evidence is price-down; at/above-entry rejections classify as
    # ENTRY_CONSTRAINT and are deliberately excluded from event breadth.
    for i in range(n_soft_sells):
        ticker = f"SOFT{i:02d}.T"
        pos = _make_position(
            ticker=ticker, current_price=1950, market_value_usd=1000, conid=conid
        )
        a = _make_analysis(ticker=ticker, verdict="DO_NOT_INITIATE", age_days=0)
        a.analysis_date = sell_date
        a.health_adj = 65.0
        a.growth_adj = 60.0
        positions.append(pos)
        analyses[ticker] = a
        conid += 1

    # STOP_BREACH SELLs
    for i in range(n_stop_breaches):
        ticker = f"STOP{i:02d}.T"
        pos = _make_position(
            ticker=ticker, current_price=1500, market_value_usd=1000, conid=conid
        )
        a = _make_analysis(ticker=ticker, verdict="BUY", stop_price=1800, age_days=0)
        a.analysis_date = sell_date
        a.health_adj = 70.0
        a.growth_adj = 65.0
        positions.append(pos)
        analyses[ticker] = a
        conid += 1

    # HARD_REJECT rejections — failed fundamental checks (price-down, as above)
    for i in range(n_hard_rejects):
        ticker = f"HARD{i:02d}.T"
        pos = _make_position(
            ticker=ticker, current_price=1950, market_value_usd=1000, conid=conid
        )
        a = _make_analysis(ticker=ticker, verdict="DO_NOT_INITIATE", age_days=0)
        a.analysis_date = sell_date
        a.health_adj = 35.0  # fails hard check
        a.growth_adj = 40.0
        positions.append(pos)
        analyses[ticker] = a
        conid += 1

    # HOLDs — normal positions
    for i in range(n_holds):
        ticker = f"HOLD{i:02d}.T"
        pos = _make_position(
            ticker=ticker, current_price=2100, market_value_usd=1000, conid=conid
        )
        a = _make_analysis(ticker=ticker, verdict="BUY", age_days=0)
        a.analysis_date = sell_date
        a.health_adj = 70.0
        a.growth_adj = 65.0
        positions.append(pos)
        analyses[ticker] = a
        conid += 1

    total_value = len(positions) * 1000.0
    portfolio = _make_portfolio(value=total_value, cash=0)
    portfolio.exchange_weights = {}
    portfolio.sector_weights = {}
    return positions, analyses, portfolio


def _make_sell_item_on_date(
    ticker: str,
    date_str: str,
    conid: int = 99999,
    sell_type: str = "SOFT_REJECT",
    current_price: float = 1800.0,
) -> ReconciliationItem:
    """Build a SELL ReconciliationItem with a specific analysis_date.

    Defaults to price BELOW the analysis entry (2100 → 1800): genuine macro
    event evidence is price-down. Pass current_price >= 2100 to model the
    analyzer-side re-rating case (MODEL_REGIME_SHIFT discriminator).
    """
    pos = _make_position(
        ticker=ticker,
        market_value_usd=1000,
        conid=conid,
        current_price=current_price,
    )
    a = _make_analysis(ticker=ticker, verdict="DO_NOT_INITIATE", age_days=0)
    a.analysis_date = date_str
    if sell_type == "SOFT_REJECT":
        a.health_adj = 65.0
        a.growth_adj = 60.0
    else:
        a.health_adj = 35.0
        a.growth_adj = 40.0
    return ReconciliationItem(
        ticker=ticker,
        action="SELL",
        reason="Verdict → DO_NOT_INITIATE",
        urgency="HIGH",
        ibkr_position=pos,
        analysis=a,
        sell_type=sell_type,
    )


def _make_hold_item_for_health(ticker: str, conid: int = 88888) -> ReconciliationItem:
    """Build a HOLD ReconciliationItem (counts toward total_held)."""
    pos = _make_position(ticker=ticker, market_value_usd=1000, conid=conid)
    a = _make_analysis(ticker=ticker, verdict="BUY", age_days=0)
    a.health_adj = 70.0
    a.growth_adj = 65.0
    return ReconciliationItem(
        ticker=ticker,
        action="HOLD",
        reason="Within targets",
        urgency="LOW",
        ibkr_position=pos,
        analysis=a,
    )


class TestCorrelatedSellDetection:
    """Test CORRELATED_SELL_EVENT detection and SOFT_REJECT demotion."""

    def test_correlated_sell_event_triggered(self):
        """6 verdict-change SELLs, 8 total held → 75% → CORRELATED_SELL_EVENT fires."""
        positions, analyses, portfolio = _make_multi_sell_scenario(
            n_soft_sells=6, n_stop_breaches=0, n_hard_rejects=0, n_holds=2
        )
        items = reconcile(positions, analyses, portfolio)
        flags = compute_portfolio_health(
            positions, analyses, portfolio, reconciliation_items=items
        )
        assert any("CORRELATED_SELL_EVENT" in f for f in flags)

    def test_correlated_sell_event_not_triggered_when_sparse(self):
        """3 verdict-change SELLs, 20 held → 15% → no correlated event."""
        positions, analyses, portfolio = _make_multi_sell_scenario(
            n_soft_sells=3, n_stop_breaches=0, n_hard_rejects=0, n_holds=17
        )
        items = reconcile(positions, analyses, portfolio)
        flags = compute_portfolio_health(
            positions, analyses, portfolio, reconciliation_items=items
        )
        assert not any("CORRELATED_SELL_EVENT" in f for f in flags)

    def test_correlated_sell_event_not_triggered_when_fewer_than_5(self):
        """4 verdict-change SELLs on same date, 4 held → 100% but count < 5 → no event."""
        positions, analyses, portfolio = _make_multi_sell_scenario(
            n_soft_sells=4, n_stop_breaches=0, n_hard_rejects=0, n_holds=0
        )
        items = reconcile(positions, analyses, portfolio)
        flags = compute_portfolio_health(
            positions, analyses, portfolio, reconciliation_items=items
        )
        assert not any("CORRELATED_SELL_EVENT" in f for f in flags)

    def test_stop_breaches_count_toward_correlation(self):
        """10 stop-breach SELLs ARE event evidence (June 2026 Hormuz fix).

        A burst of breached stops is the purest same-time price-shock signal;
        the old behavior excluded them, which let a 29-sell macro event miss
        the cumulative trigger by one position.
        """
        positions, analyses, portfolio = _make_multi_sell_scenario(
            n_soft_sells=0, n_stop_breaches=10, n_hard_rejects=0, n_holds=2
        )
        items = reconcile(positions, analyses, portfolio)
        flags = compute_portfolio_health(
            positions, analyses, portfolio, reconciliation_items=items
        )
        assert any("CORRELATED_SELL_EVENT" in f for f in flags)
        # Price breaks are REVIEW at source (July 2026 — no sell authority);
        # they still count as event evidence via action_basis=STOP_LOSS.
        stop_reviews = [
            i for i in items if i.sell_type == "STOP_BREACH" and i.action == "REVIEW"
        ]
        assert len(stop_reviews) == 10
        assert all("review level" in i.reason for i in stop_reviews)

    def test_profit_take_not_counted_in_correlation(self):
        """PROFIT_TAKE exits stay excluded from event evidence."""
        items = [
            _make_hold_item_for_health(f"H{i}.T", conid=200 + i) for i in range(10)
        ]
        for item in items[:6]:
            item.action = "SELL"
            item.sell_type = "PROFIT_TAKE"
        portfolio = _make_portfolio(value=10_000, cash=0)
        portfolio.exchange_weights = {}
        positions = [i.ibkr_position for i in items if i.ibkr_position]
        flags = compute_portfolio_health(
            positions, {}, portfolio, reconciliation_items=items
        )
        assert not any("CORRELATED_SELL_EVENT" in f for f in flags)

    def test_soft_reject_is_review_at_source_and_counts_as_evidence(self):
        """Fundamentals-intact rejections are REVIEW at the disposition layer —
        permanently, not only during a macro window — and still count toward
        correlated-event breadth via action_basis when priced below entry."""
        positions, analyses, portfolio = _make_multi_sell_scenario(
            n_soft_sells=6, n_stop_breaches=0, n_hard_rejects=0, n_holds=2
        )
        items = reconcile(positions, analyses, portfolio)
        soft = [i for i in items if i.sell_type == "SOFT_REJECT"]
        assert len(soft) == 6
        assert all(i.action == "REVIEW" for i in soft)
        assert all(i.action_basis == "THESIS_REASSESSMENT" for i in soft)

        flags = compute_portfolio_health(
            positions, analyses, portfolio, reconciliation_items=items
        )
        # Permanent reviews did not erase event breadth evidence
        assert any("CORRELATED_SELL_EVENT" in f for f in flags)
        assert all(i.action == "REVIEW" for i in soft)

    def test_unconfirmed_hard_reject_is_review_not_sell(self):
        """A single unconfirmed reject — even with weak scores — is a REVIEW
        with a refresh requirement, never an unattended executable SELL."""
        positions, analyses, portfolio = _make_multi_sell_scenario(
            n_soft_sells=0, n_stop_breaches=0, n_hard_rejects=3, n_holds=3
        )
        with patch("src.ibkr.position_evaluator._load_prior_history", return_value=[]):
            items = reconcile(positions, analyses, portfolio)
        hard = [i for i in items if i.sell_type == "HARD_REJECT"]
        assert len(hard) == 3
        assert all(i.action == "REVIEW" for i in hard)
        assert all(i.action_basis == "THESIS_REASSESSMENT" for i in hard)

    def test_confirmed_hard_reject_stays_sell_on_correlated_day(self):
        """A reject CONFIRMED by a prior full-mode reject is an executable SELL
        and survives the macro demotion pass (two independent analyses agreed —
        the forcing function must outlive macro windows)."""
        from datetime import datetime, timedelta

        from src.ibkr.buy_stability import PriorVerdict

        positions, analyses, portfolio = _make_multi_sell_scenario(
            n_soft_sells=5,  # enough to trigger event
            n_stop_breaches=0,
            n_hard_rejects=3,
            n_holds=3,
        )

        def fake_history(analysis):
            if analysis.ticker.startswith("HARD"):
                return [
                    PriorVerdict(
                        verdict="DO_NOT_INITIATE",
                        analysis_dt=datetime.now() - timedelta(days=20),
                        is_quick_mode=False,
                        file_path="prior.json",
                    )
                ]
            return []

        with patch(
            "src.ibkr.position_evaluator._load_prior_history",
            side_effect=fake_history,
        ):
            items = reconcile(positions, analyses, portfolio)
        compute_portfolio_health(
            positions, analyses, portfolio, reconciliation_items=items
        )

        hard_sells = [
            i for i in items if i.sell_type == "HARD_REJECT" and i.action == "SELL"
        ]
        assert len(hard_sells) == 3  # confirmed — survives the macro pass
        assert all(i.action_basis == "CONFIRMED_THESIS_FAILURE" for i in hard_sells)

    def test_quick_mode_reject_never_confirms(self):
        """A quick-mode screening reject cannot confirm a thesis failure."""
        from datetime import datetime, timedelta

        from src.ibkr.buy_stability import PriorVerdict

        positions, analyses, portfolio = _make_multi_sell_scenario(
            n_soft_sells=0, n_stop_breaches=0, n_hard_rejects=1, n_holds=3
        )
        quick_prior = [
            PriorVerdict(
                verdict="DO_NOT_INITIATE",
                analysis_dt=datetime.now() - timedelta(days=20),
                is_quick_mode=True,
                file_path="prior.json",
            )
        ]
        with patch(
            "src.ibkr.position_evaluator._load_prior_history",
            return_value=quick_prior,
        ):
            items = reconcile(positions, analyses, portfolio)
        hard = [i for i in items if i.sell_type == "HARD_REJECT"]
        assert all(i.action == "REVIEW" for i in hard)

    def test_same_week_reject_does_not_self_confirm(self):
        """Two rejects inside the min-spacing window (one bad data day,
        re-analyzed) must not self-confirm into an executable SELL."""
        from datetime import datetime, timedelta

        from src.ibkr.buy_stability import PriorVerdict

        positions, analyses, portfolio = _make_multi_sell_scenario(
            n_soft_sells=0, n_stop_breaches=0, n_hard_rejects=1, n_holds=3
        )
        close_prior = [
            PriorVerdict(
                verdict="DO_NOT_INITIATE",
                # scenario analysis_date is now-5d; 2 days before it → spacing 2d
                analysis_dt=datetime.now() - timedelta(days=7),
                is_quick_mode=False,
                file_path="prior.json",
            )
        ]
        with patch(
            "src.ibkr.position_evaluator._load_prior_history",
            return_value=close_prior,
        ):
            items = reconcile(positions, analyses, portfolio)
        hard = [i for i in items if i.sell_type == "HARD_REJECT"]
        assert all(i.action == "REVIEW" for i in hard)

    def test_legacy_stop_sell_demotion_still_guards_hand_built_items(self):
        """reconcile() no longer emits SELL+STOP_BREACH, but the macro demotion
        pass still guards legacy/hand-built items with intact fundamentals."""
        positions, analyses, portfolio = _make_multi_sell_scenario(
            n_soft_sells=5,  # triggers event
            n_stop_breaches=2,
            n_hard_rejects=0,
            n_holds=3,
        )
        items = reconcile(positions, analyses, portfolio)
        # Simulate legacy artifacts: force the price-break reviews back to SELL.
        for item in items:
            if item.sell_type == "STOP_BREACH":
                item.action = "SELL"
        compute_portfolio_health(
            positions, analyses, portfolio, reconciliation_items=items
        )

        # Intact fundamentals (70/65 from the fixture) → demoted with the tag.
        stop_reviews = [
            i for i in items if i.sell_type == "STOP_BREACH" and i.action == "REVIEW"
        ]
        assert len(stop_reviews) == 2
        assert all("MACRO_PRICE" in i.reason for i in stop_reviews)

    def test_stop_breach_is_review_at_source_on_correlated_day(self):
        """Price breaks are REVIEW at source (July 2026) — the macro demotion
        pass finds no SELL+STOP_BREACH items to touch, and the review framing
        already carries the fundamentals context."""
        positions, analyses, portfolio = _make_multi_sell_scenario(
            n_soft_sells=5,  # triggers event
            n_stop_breaches=2,
            n_hard_rejects=0,
            n_holds=3,
        )
        items = reconcile(positions, analyses, portfolio)
        # Price breaks never start as SELL — review at source.
        stop_before = [
            i for i in items if i.sell_type == "STOP_BREACH" and i.action == "SELL"
        ]
        assert len(stop_before) == 0

        compute_portfolio_health(
            positions, analyses, portfolio, reconciliation_items=items
        )

        stop_reviews = [
            i for i in items if i.sell_type == "STOP_BREACH" and i.action == "REVIEW"
        ]
        stop_sells = [
            i for i in items if i.sell_type == "STOP_BREACH" and i.action == "SELL"
        ]
        assert len(stop_reviews) == 2
        assert len(stop_sells) == 0
        # Reason carries the fundamentals context (scores from the analysis).
        assert all("H:" in i.reason and "G:" in i.reason for i in stop_reviews)

    def test_legacy_stop_sell_without_analysis_is_downgraded(self):
        """Unknown fundamentals cannot turn a price trigger into sale authority."""
        positions, analyses, portfolio = _make_multi_sell_scenario(
            n_soft_sells=5,  # triggers event
            n_stop_breaches=1,
            n_hard_rejects=0,
            n_holds=3,
        )
        items = reconcile(positions, analyses, portfolio)
        # Simulate a legacy SELL-at-source artifact with no analysis attached
        for item in items:
            if item.sell_type == "STOP_BREACH":
                item.action = "SELL"
                item.analysis = None

        compute_portfolio_health(
            positions, analyses, portfolio, reconciliation_items=items
        )

        stop_reviews = [
            i for i in items if i.sell_type == "STOP_BREACH" and i.action == "REVIEW"
        ]
        assert len(stop_reviews) == 1

    def test_price_break_is_review_without_correlated_event(self):
        """A price break is REVIEW at source even with no macro event — the
        review framing does not depend on correlated-event demotion."""
        positions, analyses, portfolio = _make_multi_sell_scenario(
            n_soft_sells=2,  # NOT enough to trigger event (need ≥5)
            n_stop_breaches=1,
            n_hard_rejects=0,
            n_holds=10,
        )
        items = reconcile(positions, analyses, portfolio)
        compute_portfolio_health(
            positions, analyses, portfolio, reconciliation_items=items
        )

        stop_items = [i for i in items if i.sell_type == "STOP_BREACH"]
        assert len(stop_items) == 1
        assert stop_items[0].action == "REVIEW"
        # No macro tag without an event — the review stands on its own reason.
        assert "MACRO_PRICE" not in stop_items[0].reason

    # ── Boundary / threshold tests ───────────────────────────────────────────

    def _make_stop_breach_at_correlated_event(
        self, health: float | None, growth: float | None
    ) -> list:
        """Return items after compute_portfolio_health, with a STOP_BREACH whose
        analysis has the specified health_adj and growth_adj values.

        reconcile() emits price breaks as REVIEW at source (July 2026); these
        boundary tests cover the legacy demotion pass in _apply_macro_demotions,
        so the item is forced back to SELL to simulate a legacy artifact.
        """
        positions, analyses, portfolio = _make_multi_sell_scenario(
            n_soft_sells=5,  # enough to trigger correlated event
            n_stop_breaches=1,
            n_hard_rejects=0,
            n_holds=3,
        )
        items = reconcile(positions, analyses, portfolio)
        for item in items:
            if item.sell_type == "STOP_BREACH" and item.analysis is not None:
                item.action = "SELL"  # simulate legacy SELL-at-source artifact
                item.analysis.health_adj = health
                item.analysis.growth_adj = growth
        compute_portfolio_health(
            positions, analyses, portfolio, reconciliation_items=items
        )
        return items

    def test_stop_breach_demoted_at_exact_health_boundary(self):
        """STOP_BREACH with health_adj=50.0 exactly is demoted (>= not >)."""
        items = self._make_stop_breach_at_correlated_event(health=50.0, growth=65.0)
        reviews = [
            i for i in items if i.sell_type == "STOP_BREACH" and i.action == "REVIEW"
        ]
        assert len(reviews) == 1

    def test_stop_breach_demoted_at_exact_growth_boundary(self):
        """STOP_BREACH with growth_adj=50.0 exactly is demoted (>= not >)."""
        items = self._make_stop_breach_at_correlated_event(health=65.0, growth=50.0)
        reviews = [
            i for i in items if i.sell_type == "STOP_BREACH" and i.action == "REVIEW"
        ]
        assert len(reviews) == 1

    def test_stop_breach_is_review_health_just_below_threshold(self):
        """A weak score does not let a legacy price trigger bypass confirmation."""
        items = self._make_stop_breach_at_correlated_event(health=49.9, growth=65.0)
        reviews = [
            i for i in items if i.sell_type == "STOP_BREACH" and i.action == "REVIEW"
        ]
        assert len(reviews) == 1

    def test_stop_breach_is_review_growth_just_below_threshold(self):
        """A weak score does not let a legacy price trigger bypass confirmation."""
        items = self._make_stop_breach_at_correlated_event(health=65.0, growth=49.9)
        reviews = [
            i for i in items if i.sell_type == "STOP_BREACH" and i.action == "REVIEW"
        ]
        assert len(reviews) == 1

    def test_stop_breach_is_review_when_health_weak_growth_strong(self):
        items = self._make_stop_breach_at_correlated_event(health=35.0, growth=65.0)
        reviews = [
            i for i in items if i.sell_type == "STOP_BREACH" and i.action == "REVIEW"
        ]
        assert len(reviews) == 1

    def test_stop_breach_is_review_when_health_strong_growth_weak(self):
        items = self._make_stop_breach_at_correlated_event(health=65.0, growth=35.0)
        reviews = [
            i for i in items if i.sell_type == "STOP_BREACH" and i.action == "REVIEW"
        ]
        assert len(reviews) == 1

    def test_stop_breach_is_review_when_health_adj_none(self):
        items = self._make_stop_breach_at_correlated_event(health=None, growth=65.0)
        reviews = [
            i for i in items if i.sell_type == "STOP_BREACH" and i.action == "REVIEW"
        ]
        assert len(reviews) == 1

    def test_stop_breach_is_review_when_growth_adj_none(self):
        items = self._make_stop_breach_at_correlated_event(health=65.0, growth=None)
        reviews = [
            i for i in items if i.sell_type == "STOP_BREACH" and i.action == "REVIEW"
        ]
        assert len(reviews) == 1

    def test_stop_breach_demotion_ignores_score_strength(self):
        """All legacy price-trigger sales downgrade regardless of score state."""
        positions, analyses, portfolio = _make_multi_sell_scenario(
            n_soft_sells=5,  # triggers event
            n_stop_breaches=5,
            n_hard_rejects=0,
            n_holds=3,
        )
        items = reconcile(positions, analyses, portfolio)
        stop_items = [i for i in items if i.sell_type == "STOP_BREACH"]
        assert len(stop_items) == 5
        # Simulate legacy SELL-at-source; give 3 strong, 2 weak fundamentals.
        for i, item in enumerate(stop_items):
            item.action = "SELL"
            if item.analysis is not None:
                if i < 3:
                    item.analysis.health_adj = 70.0
                    item.analysis.growth_adj = 65.0
                else:
                    item.analysis.health_adj = 30.0
                    item.analysis.growth_adj = 35.0
        compute_portfolio_health(
            positions, analyses, portfolio, reconciliation_items=items
        )
        demoted = [
            i for i in items if i.sell_type == "STOP_BREACH" and i.action == "REVIEW"
        ]
        remaining_sells = [
            i for i in items if i.sell_type == "STOP_BREACH" and i.action == "SELL"
        ]
        assert len(demoted) == 5
        assert len(remaining_sells) == 0
        assert all("MACRO_PRICE" in i.reason for i in demoted)

    def test_no_reconciliation_items_no_crash(self):
        """compute_portfolio_health with reconciliation_items=None doesn't crash."""
        pos = _make_position(market_value_usd=10000)
        analysis = _make_analysis()
        analysis.health_adj = 70.0
        portfolio = _make_portfolio(value=100000)
        portfolio.exchange_weights = {}
        # Should not raise
        flags = compute_portfolio_health(
            [pos], {"7203.T": analysis}, portfolio, reconciliation_items=None
        )
        assert isinstance(flags, list)


class TestParseScoresFromFinalDecision:
    """Unit tests for the legacy analysis text extractor."""

    def test_structured_fields(self):
        """Mid-era format: HEALTH_ADJ / GROWTH_ADJ / VERDICT / ZONE structured fields."""
        text = """
=== DECISION LOGIC ===
ZONE: LOW (< 1.0)
HEALTH_ADJ: 79
GROWTH_ADJ: 83
VERDICT: BUY
"""
        result = _parse_scores_from_final_decision(text)
        assert result["health_adj"] == 79.0
        assert result["growth_adj"] == 83.0
        assert result["verdict"] == "BUY"
        assert result["zone"] == "LOW"

    def test_narrative_format(self):
        """Old-era format: prose Financial Health / Growth Transition percentages."""
        text = """
**Action**: **BUY**

- **Financial Health**: 70.8% (Adjusted) - **PASS**
- **Growth Transition**: 66.7% (Adjusted) - **PASS**
"""
        result = _parse_scores_from_final_decision(text)
        assert result["health_adj"] == 70.8
        assert result["growth_adj"] == 66.7
        assert result["verdict"] == "BUY"

    def test_pm_verdict_header(self):
        """PORTFOLIO MANAGER VERDICT header pattern."""
        text = "### PORTFOLIO MANAGER VERDICT: HOLD\nHEALTH_ADJ: 55\nGROWTH_ADJ: 60"
        result = _parse_scores_from_final_decision(text)
        assert result["verdict"] == "HOLD"
        assert result["health_adj"] == 55.0

    def test_structured_takes_precedence_over_narrative(self):
        """Structured HEALTH_ADJ wins over the prose Financial Health value."""
        text = "Financial Health: 60.0% (Adjusted)\nHEALTH_ADJ: 79"
        result = _parse_scores_from_final_decision(text)
        assert result["health_adj"] == 79.0

    def test_missing_fields_return_empty_dict(self):
        """Text with no recognisable patterns → empty dict (no KeyError)."""
        result = _parse_scores_from_final_decision("Nothing useful here.")
        assert result == {}

    def test_do_not_initiate_verdict(self):
        """DO_NOT_INITIATE verdict parsed correctly (underscore in name)."""
        text = "VERDICT: DO_NOT_INITIATE\nHEALTH_ADJ: 38\nGROWTH_ADJ: 32"
        result = _parse_scores_from_final_decision(text)
        assert result["verdict"] == "DO_NOT_INITIATE"

    def test_do_not_initiate_with_spaces_normalised(self):
        """'DO NOT INITIATE' (space-separated PM output) → normalised to 'DO_NOT_INITIATE'."""
        text = "VERDICT: DO NOT INITIATE\nHEALTH_ADJ: 38\nGROWTH_ADJ: 32"
        result = _parse_scores_from_final_decision(text)
        assert result["verdict"] == "DO_NOT_INITIATE"

    def test_verdict_with_trailing_slash_not_captured(self):
        """Verdict stops before ' / ZONE' — doesn't bleed into next field."""
        text = "VERDICT: BUY / ZONE: MODERATE"
        result = _parse_scores_from_final_decision(text)
        assert result["verdict"] == "BUY"

    def test_load_latest_analyses_normalises_spaced_verdict(self, tmp_path):
        """Snapshot with 'DO NOT INITIATE' verdict → loaded as 'DO_NOT_INITIATE'."""
        import json as _json

        from src.ibkr.analysis_index import load_latest_analyses

        analysis_json = {
            "prediction_snapshot": {
                "ticker": "SOP",
                "verdict": "DO NOT INITIATE",
                "analysis_date": "2026-03-07",
            },
            "investment_analysis": {"trader_plan": ""},
            "final_decision": {"decision": ""},
        }
        f = tmp_path / "SOP_20260307_120000_analysis.json"
        f.write_text(_json.dumps(analysis_json))
        analyses = load_latest_analyses(tmp_path)
        r = analyses.get("SOP")
        assert r is not None
        assert r.verdict == "DO_NOT_INITIATE"

    def test_load_latest_analyses_uses_fallback(self, tmp_path):
        """load_latest_analyses fills health/growth from final_decision text."""
        analysis_json = {
            "prediction_snapshot": {},
            "investment_analysis": {"trader_plan": ""},
            "final_decision": {
                "decision": ("HEALTH_ADJ: 79\nGROWTH_ADJ: 83\nVERDICT: BUY\nZONE: LOW")
            },
        }
        f = tmp_path / "9201.T_20260210_220052_analysis.json"
        f.write_text(json.dumps(analysis_json))

        analyses = load_latest_analyses(tmp_path)
        r = analyses.get("9201.T") or analyses.get("9201_T")
        assert r is not None, "9201.T should be loaded"
        assert r.health_adj == 79.0
        assert r.growth_adj == 83.0
        assert r.verdict == "BUY"
        assert r.zone == "LOW"


# ── Window-Based Correlated Sell Detection Tests ──


class TestCorrelatedSellDetectionWindow:
    """
    Test that the sliding date window correctly groups sells from nearby dates.

    Scenario that motivated the fix:
        22:07 run — 22 sells all dated 2026-03-05 → 22/49 = 45% → event fires,
            24 positions demoted to MACRO_WATCH
        23:10 run — same portfolio, ~4 analyses re-run with new date 2026-03-06
            → 2026-03-05 cluster drops from 22 to 18, still 18/49 = 37% but with
            exact-date matching the Counter only sees 18 on 03-05 vs 4 on 03-06,
            which still fires.  In practice, enough re-runs fragmented the cluster
            below the threshold.  The 7-day window prevents this fragility.
    """

    def test_sells_within_window_grouped_as_correlated(self):
        """
        3 sells on 2026-03-05 + 3 sells on 2026-03-10 (5 days apart, within 7-day
        window) + 2 holds = 8 total held.
        6 within window / 8 held = 75% and ≥5 → fires.
        Without the window fix the per-date max is 3 < 5 → would NOT fire.
        """
        items = (
            [
                _make_sell_item_on_date(f"S{i}.T", _recent(18), conid=100 + i)
                for i in range(3)
            ]
            + [
                _make_sell_item_on_date(f"T{i}.T", _recent(13), conid=200 + i)
                for i in range(3)
            ]
            + [_make_hold_item_for_health(f"H{i}.T", conid=300 + i) for i in range(2)]
        )
        positions = [i.ibkr_position for i in items if i.ibkr_position]
        portfolio = _make_portfolio(value=len(positions) * 1000, cash=0)
        portfolio.exchange_weights = {}
        flags = compute_portfolio_health(
            positions, {}, portfolio, reconciliation_items=items
        )
        assert any("CORRELATED_SELL_EVENT" in f for f in flags)

    def test_sells_beyond_window_not_grouped(self):
        """
        3 sells on 2026-03-05 + 3 sells on 2026-03-20 (15 days apart, beyond
        14-day window) + 2 holds = 8 total held.
        Neither window contains ≥5 sells → primary does NOT fire.
        6/8 total ratio = 75% would trigger secondary, but count < 8 → no fire.
        """
        items = (
            [
                _make_sell_item_on_date(f"S{i}.T", _recent(18), conid=100 + i)
                for i in range(3)
            ]
            + [
                _make_sell_item_on_date(f"T{i}.T", _recent(3), conid=200 + i)
                for i in range(3)
            ]
            + [_make_hold_item_for_health(f"H{i}.T", conid=300 + i) for i in range(2)]
        )
        positions = [i.ibkr_position for i in items if i.ibkr_position]
        portfolio = _make_portfolio(value=len(positions) * 1000, cash=0)
        portfolio.exchange_weights = {}
        flags = compute_portfolio_health(
            positions, {}, portfolio, reconciliation_items=items
        )
        assert not any("CORRELATED_SELL_EVENT" in f for f in flags)

    def test_reanalysis_date_shift_does_not_suppress_alarm(self):
        """
        Real-world regression: 22 sells on 2026-03-05, then 4 re-analysed on
        2026-03-06 (same SELL verdict, new date) — should still fire.
        With exact-date matching: 18 on 03-05, 4 on 03-06, peak=18, 18/49=37% → fires.
        With window: 22 within [03-05, 03-11], 22/49=45% → fires.
        Both methods agree here; test documents expected behaviour.
        """
        sells_d0 = [
            _make_sell_item_on_date(f"A{i}.T", _recent(18), conid=100 + i)
            for i in range(18)
        ]
        sells_d1 = [
            _make_sell_item_on_date(f"B{i}.T", _recent(17), conid=200 + i)
            for i in range(4)
        ]
        holds = [
            _make_hold_item_for_health(f"H{i}.T", conid=300 + i) for i in range(27)
        ]
        items = sells_d0 + sells_d1 + holds
        positions = [i.ibkr_position for i in items if i.ibkr_position]
        portfolio = _make_portfolio(value=len(positions) * 1000, cash=0)
        portfolio.exchange_weights = {}
        flags = compute_portfolio_health(
            positions, {}, portfolio, reconciliation_items=items
        )
        assert any("CORRELATED_SELL_EVENT" in f for f in flags)

    def test_custom_window_days_respected(self):
        """correlated_window_days=3: sells 2 days apart group; 4 days apart do not."""
        # 3 sells on day 0 + 3 sells on day 2 = within 3-day window → 6 total
        items_close = (
            [
                _make_sell_item_on_date(f"S{i}.T", _recent(18), conid=100 + i)
                for i in range(3)
            ]
            + [
                _make_sell_item_on_date(f"T{i}.T", _recent(16), conid=200 + i)
                for i in range(3)
            ]
            + [_make_hold_item_for_health(f"H{i}.T", conid=300 + i) for i in range(2)]
        )
        positions_close = [i.ibkr_position for i in items_close if i.ibkr_position]
        portfolio_close = _make_portfolio(value=len(positions_close) * 1000, cash=0)
        portfolio_close.exchange_weights = {}
        flags_close = compute_portfolio_health(
            positions_close,
            {},
            portfolio_close,
            reconciliation_items=items_close,
            correlated_window_days=3,
        )
        assert any("CORRELATED_SELL_EVENT" in f for f in flags_close)

        # 3 sells on day 0 + 3 sells on day 4 = outside 3-day window → max 3 < 5
        items_far = (
            [
                _make_sell_item_on_date(f"S{i}.T", _recent(18), conid=400 + i)
                for i in range(3)
            ]
            + [
                _make_sell_item_on_date(f"T{i}.T", _recent(14), conid=500 + i)
                for i in range(3)
            ]
            + [_make_hold_item_for_health(f"H{i}.T", conid=600 + i) for i in range(2)]
        )
        positions_far = [i.ibkr_position for i in items_far if i.ibkr_position]
        portfolio_far = _make_portfolio(value=len(positions_far) * 1000, cash=0)
        portfolio_far.exchange_weights = {}
        flags_far = compute_portfolio_health(
            positions_far,
            {},
            portfolio_far,
            reconciliation_items=items_far,
            correlated_window_days=3,
        )
        assert not any("CORRELATED_SELL_EVENT" in f for f in flags_far)


class TestSecondaryRatioCorrelatedSell:
    """Test that gradual accumulation of sells (spread over weeks) triggers
    CORRELATED_SELL_EVENT via the secondary absolute-ratio check."""

    def test_gradual_accumulation_triggers_secondary(self):
        """10 verdict sells across 4+ weeks out of 20 held (50%) → secondary fires."""
        items = (
            # Sells spread across multiple weeks — no 14-day window catches ≥5
            [
                _make_sell_item_on_date(f"A{i}.T", "2026-02-16", conid=100 + i)
                for i in range(2)
            ]
            + [
                _make_sell_item_on_date(f"B{i}.T", _recent(18), conid=200 + i)
                for i in range(2)
            ]
            + [
                _make_sell_item_on_date(f"C{i}.T", _recent(3), conid=300 + i)
                for i in range(3)
            ]
            + [
                _make_sell_item_on_date(f"D{i}.T", "2026-04-10", conid=400 + i)
                for i in range(3)
            ]
            + [_make_hold_item_for_health(f"H{i}.T", conid=500 + i) for i in range(10)]
        )
        positions = [i.ibkr_position for i in items if i.ibkr_position]
        portfolio = _make_portfolio(value=len(positions) * 1000, cash=0)
        portfolio.exchange_weights = {}
        flags = compute_portfolio_health(
            positions, {}, portfolio, reconciliation_items=items
        )
        # 10 sells / 20 total = 50% ≥ 40% and count=10 ≥ 8 → secondary fires
        assert any("CORRELATED_SELL_EVENT" in f for f in flags)

    def test_secondary_does_not_fire_below_threshold(self):
        """7 verdict sells out of 50 held (14%) → too low for secondary."""
        items = [
            _make_sell_item_on_date(f"S{i}.T", f"2026-0{i + 1}-05", conid=100 + i)
            for i in range(7)
        ] + [_make_hold_item_for_health(f"H{i}.T", conid=200 + i) for i in range(43)]
        positions = [i.ibkr_position for i in items if i.ibkr_position]
        portfolio = _make_portfolio(value=len(positions) * 1000, cash=0)
        portfolio.exchange_weights = {}
        flags = compute_portfolio_health(
            positions, {}, portfolio, reconciliation_items=items
        )
        # 7/50 = 14% < 40% → secondary does NOT fire; also count < 8
        assert not any("CORRELATED_SELL_EVENT" in f for f in flags)

    def test_secondary_requires_minimum_count(self):
        """6 verdict sells out of 10 held (60%) → ratio passes but count < 8 → no fire."""
        items = [
            _make_sell_item_on_date(f"S{i}.T", f"2026-0{i + 1}-15", conid=100 + i)
            for i in range(6)
        ] + [_make_hold_item_for_health(f"H{i}.T", conid=200 + i) for i in range(4)]
        positions = [i.ibkr_position for i in items if i.ibkr_position]
        portfolio = _make_portfolio(value=len(positions) * 1000, cash=0)
        portfolio.exchange_weights = {}
        flags = compute_portfolio_health(
            positions, {}, portfolio, reconciliation_items=items
        )
        # 6/10 = 60% ≥ 40% but count=6 < 8 → secondary does NOT fire
        assert not any("CORRELATED_SELL_EVENT" in f for f in flags)

    def test_secondary_demotes_soft_rejects_to_review(self):
        """When secondary fires, SOFT_REJECT sells are demoted to REVIEW."""
        items = [
            _make_sell_item_on_date(
                f"S{i}.T", f"2026-0{(i % 4) + 1}-{10 + i}", conid=100 + i
            )
            for i in range(10)
        ] + [_make_hold_item_for_health(f"H{i}.T", conid=200 + i) for i in range(10)]
        positions = [i.ibkr_position for i in items if i.ibkr_position]
        portfolio = _make_portfolio(value=len(positions) * 1000, cash=0)
        portfolio.exchange_weights = {}

        # Verify sells exist before
        soft_before = [
            i for i in items if i.sell_type == "SOFT_REJECT" and i.action == "SELL"
        ]
        assert len(soft_before) == 10

        compute_portfolio_health(
            positions,
            {},
            portfolio,
            reconciliation_items=items,
            macro_override_max_age_days=10**6,
        )

        # After: demoted to REVIEW
        soft_after = [
            i for i in items if i.sell_type == "SOFT_REJECT" and i.action == "SELL"
        ]
        reviews = [
            i for i in items if i.sell_type == "SOFT_REJECT" and i.action == "REVIEW"
        ]
        assert len(soft_after) == 0
        assert len(reviews) == 10


class TestCurrencyAccuracy:
    """End-to-end tests verifying cost/quantity calculations are plausible for
    various currencies.  These are regression tests for the bug where a missing
    fx_rate_to_usd caused local-currency prices to be treated as USD (e.g.
    kr104.50 SEK reported as Cost ~$314 instead of ~$29-35).

    All 'plausible' bounds assume rough FX rates and round-trip through
    _resolve_fx → calculate_quantity → buy_cost_usd.
    """

    def _buy_analysis(
        self, currency: str, entry_price: float, fx_rate: float | None = None
    ) -> AnalysisRecord:
        """Make a BUY AnalysisRecord with the given currency and price."""
        from datetime import datetime, timedelta

        return AnalysisRecord(
            ticker="TEST.XX",
            analysis_date=(datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"),
            verdict="BUY",
            currency=currency,
            fx_rate_to_usd=fx_rate,
            entry_price=entry_price,
            current_price=entry_price,
            conviction="Medium",
            trade_block=TradeBlockData(
                action="BUY",
                size_pct=3.0,
                conviction="Medium",
                entry_price=entry_price,
            ),
        )

    def _run_reconcile(
        self, ticker: str, analysis: AnalysisRecord
    ) -> list[ReconciliationItem]:
        """Run reconciler with no positions and a single watchlist analysis."""
        portfolio = _make_portfolio(value=100_000, cash=30_000)
        return reconcile(
            positions=[],
            analyses={ticker: analysis},
            portfolio=portfolio,
            watchlist_tickers={ticker},
        )

    def test_sek_cost_is_in_dollars_not_sek(self):
        """Regression: CAG.ST 3-share buy @ kr104.50/share.
        Cost must be ~$29 (≈ SEK 313.50 * 0.093) not ~$314 (SEK treated as USD)."""
        analysis = self._buy_analysis("SEK", entry_price=104.50)  # no saved FX rate
        items = self._run_reconcile("CAG.ST", analysis)
        buys = [i for i in items if i.action == "BUY"]
        assert buys, "Expected a BUY item for SEK stock"
        cost = abs(buys[0].cash_impact_usd)
        # 3% of $100k = $3000 target; SEK rate ~0.093 → $3000 / (104.50*0.093) ≈ 309 shares
        # Lot size 1 → qty ≈ 309; cost ≈ 309 * 104.50 * 0.093 ≈ $3005
        # Key check: cost must be in dollar range, NOT 100× inflated
        assert cost < 10_000, (
            f"Cost ${cost:.0f} is implausibly high — "
            "FX rate may have defaulted to 1.0 (treating SEK as USD)"
        )
        assert cost > 100, f"Cost ${cost:.0f} is implausibly low"

    def test_jpy_cost_is_in_dollars_not_yen(self):
        """JPY stock: 3% of $100k at ¥2000/share should cost ~$3000 in USD."""
        analysis = self._buy_analysis("JPY", entry_price=2000.0)
        items = self._run_reconcile("7203.T", analysis)
        buys = [i for i in items if i.action == "BUY"]
        assert buys, "Expected a BUY item for JPY stock"
        cost = abs(buys[0].cash_impact_usd)
        assert 500 < cost < 5_000, (
            f"JPY buy cost ${cost:.0f} is wrong — "
            "should be ~$3000 (3% of $100k), not ¥3000×1.0"
        )

    def test_usd_cost_is_plausible(self):
        """USD stock needs no conversion — cost should be directly proportional."""
        analysis = self._buy_analysis("USD", entry_price=50.0, fx_rate=1.0)
        items = self._run_reconcile("AAPL", analysis)
        buys = [i for i in items if i.action == "BUY"]
        assert buys, "Expected a BUY item for USD stock"
        cost = abs(buys[0].cash_impact_usd)
        # 3% of $100k = $3000 target; $3000 / $50 = 60 shares; 60 * $50 = $3000
        assert 2000 < cost < 4000, f"USD buy cost ${cost:.0f} is implausible"

    def test_hkd_cost_is_in_dollars_not_hkd(self):
        """HKD stock: 3% of $100k at HK$58/share → ~$3000, not HK$3000."""
        analysis = self._buy_analysis("HKD", entry_price=58.0)
        items = self._run_reconcile("0005.HK", analysis)
        buys = [i for i in items if i.action == "BUY"]
        assert buys, "Expected a BUY item for HKD stock"
        cost = abs(buys[0].cash_impact_usd)
        # HKD rate ~0.128; $3000 / (58 * 0.128) ≈ 404 shares → round lot 400
        # cost ≈ 400 * 58 * 0.128 ≈ $2969
        assert (
            1000 < cost < 5000
        ), f"HKD buy cost ${cost:.0f} is wrong — should be ~$3000, not HK$3000"

    def test_saved_fx_rate_takes_precedence_over_fallback(self):
        """If the analysis snapshot has a saved fx_rate, use it — not the fallback table."""
        # Use an exotic hypothetical rate to distinguish from table
        analysis = self._buy_analysis("SEK", entry_price=100.0, fx_rate=0.100)
        items = self._run_reconcile("TEST.ST", analysis)
        buys = [i for i in items if i.action == "BUY"]
        assert buys
        # With rate=0.100: $3000 / (100 * 0.100) = 300 shares; cost = 300 * 100 * 0.100 = $3000
        # With fallback ~0.093: $3000 / (100 * 0.093) ≈ 322 shares; cost ≈ $2994
        cost = abs(buys[0].cash_impact_usd)
        qty = buys[0].suggested_quantity
        # Check that the saved rate (0.100) was used: qty should be exactly 300
        assert (
            qty == 300
        ), f"qty={qty} suggests fallback rate was used instead of saved rate 0.100"


class TestResolveFx:
    """Unit tests for _resolve_fx() — FX rate fallback chain."""

    @pytest.fixture(autouse=True)
    def _deterministic_fx_cache(self):
        """_resolve_fx() now resolves live-first via the shared FxRateCache —
        pin a fixed, network-free cache so exact-rate assertions below don't
        depend on yfinance availability."""
        set_fx_rate_cache(_FakeFxRateCache())
        yield
        set_fx_rate_cache(None)

    def _analysis(self, currency: str, fx_rate: float | None) -> AnalysisRecord:
        return AnalysisRecord(
            ticker="TEST.ST",
            analysis_date="2026-03-07",
            verdict="BUY",
            currency=currency,
            fx_rate_to_usd=fx_rate,
        )

    def test_saved_rate_returned_as_is(self):
        """When fx_rate_to_usd is already set in the snapshot, return it directly."""
        a = self._analysis("SEK", 0.097)
        assert _resolve_fx(a) == pytest.approx(0.097)

    def test_usd_currency_no_rate_returns_1(self):
        """USD with no saved rate should return 1.0 without hitting fallback table."""
        a = self._analysis("USD", None)
        assert _resolve_fx(a) == 1.0

    def test_sek_no_rate_uses_fallback(self):
        """SEK with fx_rate_to_usd=None must hit the fallback table, not return 1.0."""
        a = self._analysis("SEK", None)
        rate = _resolve_fx(a)
        # SEK fallback should be around 0.093 — definitely NOT 1.0
        assert rate != 1.0, "SEK should not fall back to 1.0 (that's the bug)"
        assert 0.07 < rate < 0.12, f"SEK fallback rate {rate} looks implausible"

    def test_nok_no_rate_uses_fallback(self):
        """NOK with no saved rate should use hardcoded fallback."""
        a = self._analysis("NOK", None)
        rate = _resolve_fx(a)
        assert rate != 1.0
        assert 0.07 < rate < 0.12

    def test_dkk_no_rate_uses_fallback(self):
        """DKK with no saved rate should use hardcoded fallback."""
        a = self._analysis("DKK", None)
        rate = _resolve_fx(a)
        assert rate != 1.0
        assert 0.12 < rate < 0.18

    def test_unknown_currency_fails_closed(self):
        """Unknown currency must disable sizing rather than masquerade as USD."""
        a = self._analysis("ZZZ", None)
        assert _resolve_fx(a) is None

    def test_sek_cost_calculation_is_plausible(self):
        """Regression for CAG.ST: 3 shares @ kr104.50 should cost ~$29-35, not ~$314."""
        a = self._analysis("SEK", None)
        fx = _resolve_fx(a)
        cost = 3 * 104.50 * fx
        assert (
            cost < 100
        ), f"Cost ${cost:.2f} is too high — FX rate is wrong (1.0 fallback?)"
        assert cost > 10, f"Cost ${cost:.2f} is suspiciously low"

    def test_legacy_1_overridden_for_sek(self):
        """Regression: CAG.ST snapshot stored fx_rate_to_usd=1.0 (legacy bogus fallback).

        Before fix: _resolve_fx returned 1.0 directly (not None, so trusted).
        After fix: 1.0 for non-USD currency is overridden with fallback table.
        3 shares @ kr104.50 cost ≈$29, NOT $314.
        """
        a = self._analysis("SEK", 1.0)  # legacy snapshot value
        rate = _resolve_fx(a)
        assert rate != 1.0, "Legacy 1.0 for SEK must be overridden"
        cost = 3 * 104.50 * rate
        assert cost < 100, f"CAG.ST cost ${cost:.2f} — should be ~$29, not ~$314"

    def test_legacy_1_overridden_for_jpy(self):
        """JPY snapshot with fx_rate_to_usd=1.0 is also a legacy bogus value."""
        a = self._analysis("JPY", 1.0)
        rate = _resolve_fx(a)
        assert rate < 0.05, f"JPY rate {rate} must be ~0.007, not 1.0"

    def test_inverse_jpy_quote_is_rejected_as_unit_mismatch(self):
        """USD/JPY (~150) must not be accepted where JPY/USD (~0.007) is required."""
        a = self._analysis("JPY", 150.0)
        rate = _resolve_fx(a)
        assert rate == pytest.approx(0.0067)

    def test_non_finite_saved_rate_uses_fallback(self):
        a = self._analysis("JPY", float("nan"))
        assert _resolve_fx(a) == pytest.approx(0.0067)

    def test_usd_rate_1_is_trusted(self):
        """For USD, fx_rate_to_usd=1.0 is correct and must NOT be overridden."""
        a = self._analysis("USD", 1.0)
        assert _resolve_fx(a) == 1.0

    def test_plausible_eur_rate_is_trusted(self):
        """A non-1.0 rate for EUR is trusted as-is (e.g. 1.08 is a real EUR/USD rate)."""
        a = self._analysis("EUR", 1.08)
        assert _resolve_fx(a) == pytest.approx(1.08)


# ── IBKR Symbol / Alpha Base Lookup Tests ──


class TestIbkrSymbol:
    """Phase 1 ReconciliationItems must carry ibkr_symbol = pos.symbol."""

    def _reconcile_one(self, pos, analysis=None):
        analyses = {analysis.ticker: analysis} if analysis else {}
        portfolio = _make_portfolio()
        return reconcile([pos], analyses, portfolio)

    def test_hold_has_ibkr_symbol(self):
        """HOLD item carries the IBKR raw symbol from the position."""
        pos = NormalizedPosition(
            conid=1,
            ticker=Ticker.from_yf("7203.T", currency="JPY"),
            quantity=100,
            market_value_usd=1400,
            currency="JPY",
        )
        a = _make_analysis(ticker="7203.T", verdict="BUY", age_days=3)
        items = self._reconcile_one(pos, a)
        hold = next(i for i in items if i.action == "HOLD")
        assert hold.ibkr_symbol == "7203"
        assert hold.ticker.yf == "7203.T"  # yf format preserved

    def test_sell_has_ibkr_symbol(self):
        """SELL item (stop breach) carries ibkr_symbol."""
        pos = NormalizedPosition(
            conid=2,
            ticker=Ticker.from_yf("MEGP.L", currency="GBX"),
            quantity=50,
            market_value_usd=500,
            currency="GBX",
            current_price_local=100.0,
            avg_cost_local=200.0,
        )
        a = _make_analysis(
            ticker="MEGP.L",
            verdict="BUY",
            stop_price=150.0,
            entry_price=200.0,
            current_price=100.0,
        )
        items = self._reconcile_one(pos, a)
        # Price break → urgent REVIEW (no sell authority); symbol still carried.
        review = next(
            i for i in items if i.action == "REVIEW" and i.sell_type == "STOP_BREACH"
        )
        assert review.ibkr_symbol == "MEGP"

    def test_review_no_analysis_has_ibkr_symbol(self):
        """REVIEW (no analysis) item carries ibkr_symbol."""
        pos = NormalizedPosition(
            conid=3,
            ticker=Ticker.from_yf("CAG.ST", currency="SEK"),
            quantity=3,
            market_value_usd=90,
            currency="SEK",
        )
        items = self._reconcile_one(pos, analysis=None)
        review = next(i for i in items if i.action == "REVIEW")
        assert review.ibkr_symbol == "CAG"

    def test_phase2_buy_has_no_ibkr_symbol(self):
        """Phase 2 BUY (new position, not held) has no ibkr_position."""
        pos = NormalizedPosition(
            conid=4,
            ticker=Ticker.from_yf("EXISTING.T", currency="JPY"),
            quantity=10,
            market_value_usd=100,
            currency="JPY",
        )
        a_held = _make_analysis(ticker="EXISTING.T", verdict="BUY", age_days=3)
        a_new = AnalysisRecord(
            ticker="NEW.T",
            analysis_date=a_held.analysis_date,
            verdict="BUY",
            currency="JPY",
            entry_price=500.0,
            fx_rate_to_usd=0.007,
            trade_block=TradeBlockData(action="BUY", size_pct=5.0),
        )
        analyses = {"EXISTING.T": a_held, "NEW.T": a_new}
        portfolio = _make_portfolio()
        items = reconcile([pos], analyses, portfolio)
        buy = next(
            (i for i in items if i.action == "BUY" and i.ticker.yf == "NEW.T"), None
        )
        if buy:
            assert buy.ibkr_position is None


class TestAlphaBaseLookup:
    """Reconciler finds analyses via base-symbol fallback when yf_ticker suffix mismatches."""

    def test_held_ticker_with_suffix_finds_analysis_stored_without_suffix(self):
        """Position yf_ticker='MEGP.L' finds analysis stored under key 'MEGP'."""
        pos = NormalizedPosition(
            conid=1,
            ticker=Ticker.from_yf("MEGP.L", currency="GBX"),
            quantity=50,
            market_value_usd=500,
            currency="GBX",
            current_price_local=200.0,
        )
        a = _make_analysis(
            ticker="MEGP",
            verdict="BUY",
            age_days=3,
            entry_price=200.0,
            stop_price=150.0,
        )
        a.currency = "GBX"  # realistic: records carry the local currency
        analyses = {"MEGP": a}
        portfolio = _make_portfolio()
        items = reconcile([pos], analyses, portfolio)
        # Should NOT be REVIEW with "no evaluator analysis found"
        non_hold = [i for i in items if i.action != "HOLD"]
        assert not any(
            "no evaluator analysis found" in i.reason for i in non_hold
        ), "Base-symbol fallback should have found the MEGP analysis"

    def test_numeric_ticker_does_not_use_base_lookup(self):
        """Numeric IBKR symbol (e.g. '7203') must NOT cross-match across exchanges."""
        pos = NormalizedPosition(
            conid=2,
            ticker=Ticker.from_yf("7203.T", currency="JPY"),
            quantity=100,
            market_value_usd=1000,
            currency="JPY",
            current_price_local=2100.0,
        )
        # Analysis stored under HK key with same numeric base — must NOT match
        a_hk = _make_analysis(ticker="7203.HK", verdict="BUY", age_days=3)
        analyses = {"7203.HK": a_hk}
        portfolio = _make_portfolio()
        items = reconcile([pos], analyses, portfolio)
        # 7203.T position should get REVIEW "no analysis found", not the HK analysis
        assert any(
            i.action == "REVIEW" and "no evaluator analysis found" in i.reason
            for i in items
        ), "Numeric ticker must not match cross-exchange analysis via base lookup"

    def test_suffixed_analysis_takes_priority_over_bare(self):
        """When both 'MEGP.L' and 'MEGP' analyses exist, suffixed one is preferred."""
        pos = NormalizedPosition(
            conid=3,
            ticker=Ticker.from_yf("MEGP.L", currency="GBX"),
            quantity=50,
            market_value_usd=500,
            currency="GBX",
            current_price_local=200.0,
        )
        a_bare = _make_analysis(ticker="MEGP", verdict="SELL", age_days=3)
        a_suffixed = _make_analysis(
            ticker="MEGP.L",
            verdict="BUY",
            age_days=3,
            entry_price=200.0,
            stop_price=150.0,
        )
        analyses = {"MEGP": a_bare, "MEGP.L": a_suffixed}
        portfolio = _make_portfolio()
        items = reconcile([pos], analyses, portfolio)
        # Should find the BUY (suffixed) analysis, not the SELL (bare) analysis
        assert not any(
            i.action == "SELL" for i in items
        ), "Suffixed MEGP.L analysis (BUY) should take priority over bare MEGP (SELL)"

    def test_bare_position_with_both_bare_and_suffixed_analyses_uses_suffixed(self):
        """CEK-style: position yf_ticker='CEK' (bare), analyses has both 'CEK' (bare,
        most recent) and 'CEK.DE' (suffixed, older).  Alpha-base lookup must resolve
        to 'CEK.DE' BEFORE analyses.get() is called, so the correct suffixed analysis
        is used and the reconciled item carries ticker='CEK.DE'."""
        pos = NormalizedPosition(
            conid=4,
            ticker=Ticker.from_ibkr("CEK", currency="EUR"),
            quantity=100,
            market_value_usd=2000,
            currency="EUR",
            current_price_local=20.0,
        )
        # Bare analysis is "newer" (age_days=1); suffixed is "older" (age_days=5).
        # load_latest_analyses() would return analyses={"CEK": a_bare, "CEK.DE": a_suffixed}.
        a_bare = _make_analysis(ticker="CEK", verdict="SELL", age_days=1)
        a_bare.currency = "EUR"  # currency must agree with the EUR position
        a_suffixed = _make_analysis(
            ticker="CEK.DE",
            verdict="BUY",
            age_days=5,
            entry_price=20.0,
            stop_price=15.0,
        )
        a_suffixed.currency = "EUR"
        analyses = {"CEK": a_bare, "CEK.DE": a_suffixed}
        portfolio = _make_portfolio()
        items = reconcile([pos], analyses, portfolio)
        # Reconciler must use the suffixed analysis (BUY), not the bare one (SELL)
        assert not any(
            i.action == "SELL" for i in items
        ), "Alpha-base lookup must prefer suffixed 'CEK.DE' over bare 'CEK'"
        # The reconciled item's ticker must be the canonical suffixed form
        cek_items = [i for i in items if "CEK" in i.ticker.yf]
        assert cek_items, "Expected a reconciliation item for CEK"
        assert (
            cek_items[0].ticker.yf == "CEK.DE"
        ), f"Expected ticker.yf='CEK.DE', got '{cek_items[0].ticker.yf}'"


class TestMacroEventTriggersJune2026:
    """Regression suite for the June 2026 Hormuz-event detector fixes:
    stop-breach counting, 35% cumulative fallback, drawdown-breadth trigger,
    and sustained demotion from stored active events.
    """

    @staticmethod
    def _spread_date(i: int) -> str:
        from datetime import date, timedelta

        return (date(2025, 6, 1) + timedelta(days=20 * i)).isoformat()

    def _hormuz_shape(self):
        """29 event sells (27 soft + 2 strong-fundamentals stops) + 39 holds = 68.

        Analysis dates spread 20 days apart — the refresh throttle smear that
        kept the old detector blind (peak 14d window count = 1).
        """
        items = []
        for i in range(27):
            items.append(
                _make_sell_item_on_date(
                    f"SOFT{i:02d}.T", self._spread_date(i), conid=100 + i
                )
            )
        for i in range(2):
            stop = _make_sell_item_on_date(
                f"STOP{i}.T",
                self._spread_date(27 + i),
                conid=400 + i,
                sell_type="STOP_BREACH",
            )
            stop.analysis.health_adj = 87.0  # HERDEZ-like: fundamentals intact
            stop.analysis.growth_adj = 83.0
            items.append(stop)
        items += [
            _make_hold_item_for_health(f"H{i:02d}.T", conid=600 + i) for i in range(39)
        ]
        positions = [i.ibkr_position for i in items if i.ibkr_position]
        portfolio = _make_portfolio(value=len(positions) * 1000, cash=0)
        portfolio.exchange_weights = {}
        return items, positions, portfolio

    def test_hormuz_shape_fires_cumulative_trigger(self):
        items, positions, portfolio = self._hormuz_shape()
        flags = compute_portfolio_health(
            positions, {}, portfolio, reconciliation_items=items
        )
        event_flags = [f for f in flags if "CORRELATED_SELL_EVENT" in f]
        assert event_flags, flags
        assert "[cumulative]" in event_flags[0]
        # 29/68 = 43% — readable in the flag for the operator banner regex
        assert "43%" in event_flags[0] or "42%" in event_flags[0]

    def test_hormuz_shape_demotes_softs_and_strong_stops(self):
        items, positions, portfolio = self._hormuz_shape()
        compute_portfolio_health(positions, {}, portfolio, reconciliation_items=items)
        soft_sells_left = [
            i for i in items if i.sell_type == "SOFT_REJECT" and i.action == "SELL"
        ]
        assert not soft_sells_left
        demoted_stops = [
            i for i in items if i.sell_type == "STOP_BREACH" and i.action == "REVIEW"
        ]
        assert len(demoted_stops) == 2

    def test_cumulative_boundary_just_below_35pct_does_not_fire(self):
        # 8 spread-out sells / 23 held = 34.8% < 35%
        items = [
            _make_sell_item_on_date(f"S{i}.T", self._spread_date(i), conid=100 + i)
            for i in range(8)
        ] + [_make_hold_item_for_health(f"H{i}.T", conid=300 + i) for i in range(15)]
        positions = [i.ibkr_position for i in items if i.ibkr_position]
        portfolio = _make_portfolio(value=len(positions) * 1000, cash=0)
        portfolio.exchange_weights = {}
        flags = compute_portfolio_health(
            positions, {}, portfolio, reconciliation_items=items
        )
        assert not any("CORRELATED_SELL_EVENT" in f for f in flags)

    def test_drawdown_breadth_fires_without_any_sells(self):
        """Price evidence alone: 10 of 24 holds trade 14% below entry."""
        items = [
            _make_hold_item_for_health(f"H{i:02d}.T", conid=100 + i) for i in range(24)
        ]
        for item in items[:10]:
            item.ibkr_position.current_price_local = 1800.0  # entry 2100 → -14.3%
        positions = [i.ibkr_position for i in items if i.ibkr_position]
        portfolio = _make_portfolio(value=len(positions) * 1000, cash=0)
        portfolio.exchange_weights = {}
        flags = compute_portfolio_health(
            positions, {}, portfolio, reconciliation_items=items
        )
        event_flags = [f for f in flags if "CORRELATED_SELL_EVENT" in f]
        assert event_flags, flags
        assert "[drawdown_breadth]" in event_flags[0]
        assert "below entry" in event_flags[0]

    def test_active_stored_event_sustains_demotion(self):
        """No fresh detection, but an unexpired stored event keeps demoting."""
        from types import SimpleNamespace

        items = [
            _make_sell_item_on_date(f"S{i}.T", self._spread_date(i), conid=100 + i)
            for i in range(2)
        ] + [_make_hold_item_for_health(f"H{i}.T", conid=300 + i) for i in range(18)]
        positions = [i.ibkr_position for i in items if i.ibkr_position]
        portfolio = _make_portfolio(value=len(positions) * 1000, cash=0)
        portfolio.exchange_weights = {}

        event = SimpleNamespace(event_type="GEOPOLITICAL", expiry="2026-12-01")
        flags = compute_portfolio_health(
            positions,
            {},
            portfolio,
            reconciliation_items=items,
            active_macro_events=[event],
        )

        assert any("ACTIVE_MACRO_EVENT" in f for f in flags)
        demoted = [i for i in items if i.action == "REVIEW"]
        assert len(demoted) == 2
        assert all("active GEOPOLITICAL event" in i.reason for i in demoted)

    def test_active_event_within_window_demotes(self):
        """Within the brief override window, a recent active event still demotes."""
        from datetime import date, timedelta
        from types import SimpleNamespace

        items = [
            _make_sell_item_on_date(f"S{i}.T", self._spread_date(i), conid=100 + i)
            for i in range(2)
        ] + [_make_hold_item_for_health(f"H{i}.T", conid=300 + i) for i in range(18)]
        positions = [i.ibkr_position for i in items if i.ibkr_position]
        portfolio = _make_portfolio(value=len(positions) * 1000, cash=0)
        portfolio.exchange_weights = {}
        recent = (date.today() - timedelta(days=3)).isoformat()
        event = SimpleNamespace(
            event_type="GEOPOLITICAL",
            expiry="2026-12-01",
            event_date=recent,
            detected_date=recent,
        )
        flags = compute_portfolio_health(
            positions,
            {},
            portfolio,
            reconciliation_items=items,
            active_macro_events=[event],
        )
        assert any("ACTIVE_MACRO_EVENT" in f for f in flags)
        assert len([i for i in items if i.action == "REVIEW"]) == 2

    def test_active_event_lapses_past_override_window(self):
        """Past the brief override window, demotion lapses (impaired positions can
        exit) and a distinct MACRO_EVENT_ONGOING flag keeps the dip-buy/panic banner
        off while signalling the event is still live."""
        from datetime import date, timedelta
        from types import SimpleNamespace

        items = [
            _make_sell_item_on_date(f"S{i}.T", self._spread_date(i), conid=100 + i)
            for i in range(2)
        ] + [_make_hold_item_for_health(f"H{i}.T", conid=300 + i) for i in range(18)]
        positions = [i.ibkr_position for i in items if i.ibkr_position]
        portfolio = _make_portfolio(value=len(positions) * 1000, cash=0)
        portfolio.exchange_weights = {}
        old = (date.today() - timedelta(days=40)).isoformat()
        event = SimpleNamespace(
            event_type="GEOPOLITICAL",
            expiry="2026-12-01",
            event_date=old,
            detected_date=old,
        )
        flags = compute_portfolio_health(
            positions,
            {},
            portfolio,
            reconciliation_items=items,
            active_macro_events=[event],
        )
        assert any("MACRO_EVENT_ONGOING" in f for f in flags)
        assert not any("ACTIVE_MACRO_EVENT" in f for f in flags)
        assert all(i.action == "SELL" for i in items[:2])  # not demoted — can exit

    def test_fresh_window_detection_wins_over_old_lapsed_active_event(self):
        """A new verdict-cluster macro signal must not be suppressed by an older
        unexpired event whose brief override window has already lapsed."""
        from datetime import date, timedelta
        from types import SimpleNamespace

        recent = (date.today() - timedelta(days=3)).isoformat()
        items = [
            _make_sell_item_on_date(f"S{i}.T", recent, conid=100 + i) for i in range(6)
        ] + [_make_hold_item_for_health(f"H{i}.T", conid=300 + i) for i in range(14)]
        positions = [i.ibkr_position for i in items if i.ibkr_position]
        portfolio = _make_portfolio(value=len(positions) * 1000, cash=0)
        portfolio.exchange_weights = {}
        old = (date.today() - timedelta(days=45)).isoformat()
        old_event = SimpleNamespace(
            event_type="GEOPOLITICAL",
            expiry="2026-12-01",
            event_date=old,
            detected_date=old,
        )

        flags = compute_portfolio_health(
            positions,
            {},
            portfolio,
            reconciliation_items=items,
            active_macro_events=[old_event],
        )

        assert any("CORRELATED_SELL_EVENT" in f and "[window]" in f for f in flags)
        assert not any("MACRO_EVENT_ONGOING" in f for f in flags)
        assert len([i for i in items[:6] if i.action == "REVIEW"]) == 6

    def test_fresh_cumulative_detection_wins_over_old_lapsed_active_event(self):
        """Fresh cumulative sell pressure is separate from a prior lapsed event."""
        from datetime import date, timedelta
        from types import SimpleNamespace

        items = [
            _make_sell_item_on_date(
                f"S{i}.T",
                (date.today() - timedelta(days=13 * i)).isoformat(),
                conid=100 + i,
            )
            for i in range(9)
        ] + [_make_hold_item_for_health(f"H{i}.T", conid=300 + i) for i in range(11)]
        positions = [i.ibkr_position for i in items if i.ibkr_position]
        portfolio = _make_portfolio(value=len(positions) * 1000, cash=0)
        portfolio.exchange_weights = {}
        old = (date.today() - timedelta(days=45)).isoformat()
        old_event = SimpleNamespace(
            event_type="GEOPOLITICAL",
            expiry="2026-12-01",
            event_date=old,
            detected_date=old,
        )

        flags = compute_portfolio_health(
            positions,
            {},
            portfolio,
            reconciliation_items=items,
            active_macro_events=[old_event],
        )

        assert any("CORRELATED_SELL_EVENT" in f and "[cumulative]" in f for f in flags)
        assert not any("MACRO_EVENT_ONGOING" in f for f in flags)
        assert len([i for i in items[:9] if i.action == "REVIEW"]) == 9

    def test_old_active_event_blocks_drawdown_only_reanchoring(self):
        """Drawdown breadth alone can be the same old structural impairment, so an
        old active event should not let it restart the brief override every run."""
        from datetime import date, timedelta
        from types import SimpleNamespace

        items = [
            _make_hold_item_for_health(f"H{i:02d}.T", conid=100 + i) for i in range(24)
        ]
        for item in items[:10]:
            item.ibkr_position.current_price_local = 1800.0  # entry 2100 → -14.3%
        positions = [i.ibkr_position for i in items if i.ibkr_position]
        portfolio = _make_portfolio(value=len(positions) * 1000, cash=0)
        portfolio.exchange_weights = {}
        old = (date.today() - timedelta(days=45)).isoformat()
        old_event = SimpleNamespace(
            event_type="GEOPOLITICAL",
            expiry="2026-12-01",
            event_date=old,
            detected_date=old,
        )

        flags = compute_portfolio_health(
            positions,
            {},
            portfolio,
            reconciliation_items=items,
            active_macro_events=[old_event],
        )

        assert any("MACRO_EVENT_ONGOING" in f for f in flags)
        assert not any("CORRELATED_SELL_EVENT" in f for f in flags)
        assert not any("ACTIVE_MACRO_EVENT" in f for f in flags)

    def test_recent_active_event_wins_when_older_active_event_is_first(self):
        """If multiple stored events are active, use a within-window event for the
        sustained override instead of letting an older first event force lapse."""
        from datetime import date, timedelta
        from types import SimpleNamespace

        items = [
            _make_sell_item_on_date(f"S{i}.T", self._spread_date(i), conid=100 + i)
            for i in range(2)
        ] + [_make_hold_item_for_health(f"H{i}.T", conid=300 + i) for i in range(18)]
        positions = [i.ibkr_position for i in items if i.ibkr_position]
        portfolio = _make_portfolio(value=len(positions) * 1000, cash=0)
        portfolio.exchange_weights = {}
        old = (date.today() - timedelta(days=45)).isoformat()
        recent = (date.today() - timedelta(days=3)).isoformat()
        events = [
            SimpleNamespace(
                event_type="OLD_EVENT",
                expiry="2026-12-01",
                event_date=old,
                detected_date=old,
            ),
            SimpleNamespace(
                event_type="RECENT_EVENT",
                expiry="2026-09-01",
                event_date=recent,
                detected_date=recent,
            ),
        ]

        flags = compute_portfolio_health(
            positions,
            {},
            portfolio,
            reconciliation_items=items,
            active_macro_events=events,
        )

        assert any("ACTIVE_MACRO_EVENT: RECENT_EVENT" in f for f in flags)
        assert all("active RECENT_EVENT event" in i.reason for i in items[:2])

    def test_no_active_event_no_fresh_detection_no_demotion(self):
        items = [
            _make_sell_item_on_date(f"S{i}.T", self._spread_date(i), conid=100 + i)
            for i in range(2)
        ] + [_make_hold_item_for_health(f"H{i}.T", conid=300 + i) for i in range(18)]
        positions = [i.ibkr_position for i in items if i.ibkr_position]
        portfolio = _make_portfolio(value=len(positions) * 1000, cash=0)
        portfolio.exchange_weights = {}
        flags = compute_portfolio_health(
            positions, {}, portfolio, reconciliation_items=items
        )
        assert not any("ACTIVE_MACRO_EVENT" in f for f in flags)
        assert all(i.action == "SELL" for i in items[:2])


class TestAmbiguousBaseGuards:
    """June 2026 AGS fix: base-symbol cross-matching must never attach a
    different exchange's analysis to a suffixed position."""

    @staticmethod
    def _pos(yf: str, currency: str, conid: int = 555001) -> NormalizedPosition:
        return NormalizedPosition(
            conid=conid,
            ticker=Ticker.from_yf(yf, currency=currency),
            quantity=10,
            avg_cost_local=68.0,
            market_value_usd=400,
            currency=currency,
            current_price_local=66.0,
        )

    def test_suffixed_position_does_not_borrow_other_exchange_analysis(self):
        """AGS.BR (EUR) + only AGS.SI (SGD) analysis → no cross-match."""
        ags_si = _make_analysis(ticker="AGS.SI")
        ags_si.currency = "SGD"
        items = reconcile(
            positions=[self._pos("AGS.BR", "EUR")],
            analyses={"AGS.SI": ags_si},
            portfolio=_make_portfolio(),
        )
        held = [i for i in items if i.ibkr_position is not None]
        assert len(held) == 1
        assert held[0].ticker.yf == "AGS.BR"
        assert items[0].analysis is None or items[0].analysis.ticker != "AGS.SI"

    def test_ambiguous_base_poisoned_each_position_gets_own_analysis(self):
        """Both AGS.SI and AGS.BR analyzed → base poisoned, direct matches only."""
        ags_si = _make_analysis(ticker="AGS.SI")
        ags_si.currency = "SGD"
        ags_br = _make_analysis(ticker="AGS.BR")
        ags_br.currency = "EUR"
        items = reconcile(
            positions=[
                self._pos("AGS.SI", "SGD", conid=555001),
                self._pos("AGS.BR", "EUR", conid=555002),
            ],
            analyses={"AGS.SI": ags_si, "AGS.BR": ags_br},
            portfolio=_make_portfolio(),
        )
        by_yf = {i.ticker.yf: i for i in items}
        assert by_yf["AGS.SI"].analysis.ticker == "AGS.SI"
        assert by_yf["AGS.BR"].analysis.ticker == "AGS.BR"

    def test_suffixed_position_borrows_bare_analysis_when_currency_agrees(self):
        """KRN.DE (EUR) + bare 'KRN' analysis with EUR currency → match allowed."""
        bare = _make_analysis(ticker="KRN")
        bare.currency = "EUR"
        items = reconcile(
            positions=[self._pos("KRN.DE", "EUR")],
            analyses={"KRN": bare},
            portfolio=_make_portfolio(),
        )
        held = [i for i in items if i.ibkr_position is not None]
        assert len(held) == 1
        assert held[0].analysis is not None
        assert held[0].analysis.ticker == "KRN"

    def test_suffixed_position_blocks_bare_analysis_on_currency_mismatch(self):
        bare = _make_analysis(ticker="KRN")
        bare.currency = "CHF"
        items = reconcile(
            positions=[self._pos("KRN.DE", "EUR")],
            analyses={"KRN": bare},
            portfolio=_make_portfolio(),
        )
        held = [i for i in items if i.ibkr_position is not None]
        assert len(held) == 1
        assert held[0].analysis is None or held[0].analysis.ticker != "KRN"


class TestMacroDetectorInputRobustness:
    """Malformed/missing inputs must never crash or silently mask detection."""

    @staticmethod
    def _spread_date(i: int) -> str:
        from datetime import date, timedelta

        return (date(2025, 6, 1) + timedelta(days=20 * i)).isoformat()

    @staticmethod
    def _book(items):
        positions = [i.ibkr_position for i in items if i.ibkr_position]
        portfolio = _make_portfolio(value=max(len(positions), 1) * 1000, cash=0)
        portfolio.exchange_weights = {}
        return positions, portfolio

    def test_malformed_dates_dont_crash_or_mask_detection(self):
        items = [
            _make_sell_item_on_date(f"S{i}.T", _recent(18), conid=100 + i)
            for i in range(5)
        ]
        bad_dates = [None, "", "not-a-date", "2026-13-45"]
        for i, bad in enumerate(bad_dates):
            item = _make_sell_item_on_date(f"B{i}.T", _recent(18), conid=200 + i)
            item.analysis.analysis_date = bad
            items.append(item)
        items += [
            _make_hold_item_for_health(f"H{i}.T", conid=300 + i) for i in range(7)
        ]
        positions, portfolio = self._book(items)

        flags = compute_portfolio_health(
            positions, {}, portfolio, reconciliation_items=items
        )
        # 5 valid clustered / 16 held = 31% >= 25% with count 5 — still fires
        assert any("CORRELATED_SELL_EVENT" in f for f in flags)

    def test_all_malformed_dates_fall_through_to_drawdown(self):
        items = []
        for i in range(9):
            item = _make_sell_item_on_date(f"S{i}.T", _recent(18), conid=100 + i)
            item.analysis.analysis_date = "garbage"
            items.append(item)
        items += [
            _make_hold_item_for_health(f"H{i:02d}.T", conid=300 + i) for i in range(31)
        ]
        for item in items[9 : 9 + 14]:
            item.ibkr_position.current_price_local = 1800.0  # entry 2100 → -14.3%
        positions, portfolio = self._book(items)

        flags = compute_portfolio_health(
            positions, {}, portfolio, reconciliation_items=items
        )
        event_flags = [f for f in flags if "CORRELATED_SELL_EVENT" in f]
        assert event_flags and "[drawdown_breadth]" in event_flags[0]

    def test_none_analysis_and_none_position_items_skipped_safely(self):
        items = [
            _make_sell_item_on_date(f"S{i}.T", _recent(18), conid=100 + i)
            for i in range(6)
        ] + [_make_hold_item_for_health(f"H{i}.T", conid=300 + i) for i in range(4)]
        no_analysis = _make_hold_item_for_health("NA.T", conid=400)
        no_analysis.analysis = None
        no_position = _make_hold_item_for_health("NP.T", conid=401)
        no_position.ibkr_position = None
        items += [no_analysis, no_position]
        positions, portfolio = self._book(items)

        flags = compute_portfolio_health(
            positions, {}, portfolio, reconciliation_items=items
        )
        assert any("CORRELATED_SELL_EVENT" in f for f in flags)  # no crash, fires

    def test_zero_negative_none_entry_prices_no_division_error(self):
        items = [
            _make_hold_item_for_health(f"H{i:02d}.T", conid=100 + i) for i in range(12)
        ]
        items[0].analysis.entry_price = 0.0
        items[1].analysis.entry_price = -5.0
        items[2].analysis.entry_price = None
        items[2].analysis.current_price = None  # fallback also absent
        items[3].ibkr_position.current_price_local = 0.0
        positions, portfolio = self._book(items)

        flags = compute_portfolio_health(
            positions, {}, portfolio, reconciliation_items=items
        )
        assert not any("CORRELATED_SELL_EVENT" in f for f in flags)

    def test_entry_price_none_falls_back_to_analysis_current_price(self):
        items = [
            _make_hold_item_for_health(f"H{i:02d}.T", conid=100 + i) for i in range(20)
        ]
        for item in items[:8]:
            item.analysis.entry_price = None
            item.analysis.current_price = 2100.0  # fallback entry
            item.ibkr_position.current_price_local = 1800.0  # -14.3%
        positions, portfolio = self._book(items)

        flags = compute_portfolio_health(
            positions, {}, portfolio, reconciliation_items=items
        )
        event_flags = [f for f in flags if "CORRELATED_SELL_EVENT" in f]
        assert event_flags and "[drawdown_breadth]" in event_flags[0]

    def test_future_dated_analyses_no_crash(self):
        from datetime import date, timedelta

        tomorrow = (date.today() + timedelta(days=1)).isoformat()
        items = [
            _make_sell_item_on_date(f"S{i}.T", tomorrow, conid=100 + i)
            for i in range(6)
        ] + [_make_hold_item_for_health(f"H{i}.T", conid=300 + i) for i in range(4)]
        positions, portfolio = self._book(items)

        flags = compute_portfolio_health(
            positions, {}, portfolio, reconciliation_items=items
        )
        assert any("CORRELATED_SELL_EVENT" in f for f in flags)

    def test_year_boundary_window_grouping(self):
        items = (
            [
                _make_sell_item_on_date(f"S{i}.T", "2025-12-28", conid=100 + i)
                for i in range(3)
            ]
            + [
                _make_sell_item_on_date(f"T{i}.T", "2026-01-04", conid=200 + i)
                for i in range(3)
            ]
            + [_make_hold_item_for_health(f"H{i}.T", conid=300 + i) for i in range(2)]
        )
        positions, portfolio = self._book(items)

        flags = compute_portfolio_health(
            positions,
            {},
            portfolio,
            reconciliation_items=items,
            macro_override_max_age_days=10**6,
        )
        assert any("CORRELATED_SELL_EVENT" in f for f in flags)

    def test_empty_reconciliation_items_no_flags_no_errors(self):
        holds = [_make_hold_item_for_health(f"H{i}.T", conid=100 + i) for i in range(3)]
        positions, portfolio = self._book(holds)

        flags = compute_portfolio_health(
            positions, {}, portfolio, reconciliation_items=[]
        )
        assert not any("CORRELATED_SELL_EVENT" in f for f in flags)

    def test_all_position_none_items_zero_held_no_division_error(self):
        holds = [_make_hold_item_for_health(f"H{i}.T", conid=100 + i) for i in range(3)]
        positions, portfolio = self._book(holds)
        orphan = _make_sell_item_on_date("S0.T", _recent(18), conid=200)
        orphan.ibkr_position = None

        flags = compute_portfolio_health(
            positions, {}, portfolio, reconciliation_items=[orphan]
        )
        assert not any("CORRELATED_SELL_EVENT" in f for f in flags)

    def test_tiny_book_never_fires(self):
        items = [_make_sell_item_on_date("S0.T", _recent(18), conid=100)]
        positions, portfolio = self._book(items)

        flags = compute_portfolio_health(
            positions, {}, portfolio, reconciliation_items=items
        )
        # 1/1 = 100% but count floors (>=5 window, >=8 cumulative/drawdown) hold
        assert not any("CORRELATED_SELL_EVENT" in f for f in flags)

    def test_duplicate_ticker_items_counted_per_item(self):
        # The AGS-dup shape: same ticker appears as two SELL items. Current
        # contract counts per ITEM (not per ticker) in both numerator and
        # denominator — pinned deliberately.
        items = []
        for i in range(4):
            for dup in range(2):
                items.append(
                    _make_sell_item_on_date(
                        f"DUP{i}.T",
                        self._spread_date(i * 2 + dup),
                        conid=100 + i * 2 + dup,
                    )
                )
        items += [
            _make_hold_item_for_health(f"H{i:02d}.T", conid=300 + i) for i in range(12)
        ]
        positions, portfolio = self._book(items)

        flags = compute_portfolio_health(
            positions,
            {},
            portfolio,
            reconciliation_items=items,
            macro_override_max_age_days=10**6,
        )
        # 8 evidence items / 20 held = 40% >= 35%, count 8 >= 8 → cumulative fires
        event_flags = [f for f in flags if "CORRELATED_SELL_EVENT" in f]
        assert event_flags and "[cumulative]" in event_flags[0]

    def test_trim_and_untyped_items_are_not_evidence_nor_demoted(self):
        items = []
        for i in range(9):
            item = _make_hold_item_for_health(f"T{i}.T", conid=100 + i)
            item.action = "TRIM"
            item.sell_type = None
            items.append(item)
        untyped_sell = _make_hold_item_for_health("U0.T", conid=200)
        untyped_sell.action = "SELL"
        untyped_sell.sell_type = None
        items.append(untyped_sell)
        items += [
            _make_hold_item_for_health(f"H{i}.T", conid=300 + i) for i in range(2)
        ]
        positions, portfolio = self._book(items)

        flags = compute_portfolio_health(
            positions, {}, portfolio, reconciliation_items=items
        )
        assert not any("CORRELATED_SELL_EVENT" in f for f in flags)
        assert all(i.action == "TRIM" for i in items[:9])  # untouched
        assert untyped_sell.action == "SELL"


class TestMacroDetectorBoundaries:
    """Exact threshold edges for all three triggers."""

    @staticmethod
    def _spread_date(i: int) -> str:
        from datetime import date, timedelta

        return (date(2025, 6, 1) + timedelta(days=20 * i)).isoformat()

    @staticmethod
    def _book(items):
        positions = [i.ibkr_position for i in items if i.ibkr_position]
        portfolio = _make_portfolio(value=max(len(positions), 1) * 1000, cash=0)
        portfolio.exchange_weights = {}
        return positions, portfolio

    def _window_case(self, holds: int) -> list[str]:
        items = [
            _make_sell_item_on_date(f"S{i}.T", _recent(18), conid=100 + i)
            for i in range(5)
        ] + [
            _make_hold_item_for_health(f"H{i:02d}.T", conid=300 + i)
            for i in range(holds)
        ]
        positions, portfolio = self._book(items)
        return compute_portfolio_health(
            positions, {}, portfolio, reconciliation_items=items
        )

    def test_window_ratio_exact_25pct_fires(self):
        assert any("CORRELATED_SELL_EVENT" in f for f in self._window_case(15))  # 5/20

    def test_window_ratio_just_below_25pct_does_not_fire(self):
        flags = self._window_case(16)  # 5/21 = 23.8%
        assert not any("CORRELATED_SELL_EVENT" in f for f in flags)

    def test_cumulative_exact_35pct_fires(self):
        items = [
            _make_sell_item_on_date(f"S{i:02d}.T", self._spread_date(i), conid=100 + i)
            for i in range(14)
        ] + [
            _make_hold_item_for_health(f"H{i:02d}.T", conid=300 + i) for i in range(26)
        ]
        positions, portfolio = self._book(items)
        flags = compute_portfolio_health(
            positions,
            {},
            portfolio,
            reconciliation_items=items,
            macro_override_max_age_days=10**6,
        )
        event_flags = [f for f in flags if "CORRELATED_SELL_EVENT" in f]
        assert event_flags and "[cumulative]" in event_flags[0]  # 14/40 = 35.0%

    def test_cumulative_ratio_passes_but_count_floor_blocks(self):
        items = [
            _make_sell_item_on_date(f"S{i}.T", self._spread_date(i), conid=100 + i)
            for i in range(7)
        ] + [
            _make_hold_item_for_health(f"H{i:02d}.T", conid=300 + i) for i in range(13)
        ]
        positions, portfolio = self._book(items)
        flags = compute_portfolio_health(
            positions, {}, portfolio, reconciliation_items=items
        )
        # 7/20 = 35% but count 7 < 8
        assert not any("CORRELATED_SELL_EVENT" in f for f in flags)

    def _drawdown_case(self, current: float, n_down: int, n_total: int) -> list[str]:
        items = [
            _make_hold_item_for_health(f"H{i:02d}.T", conid=100 + i)
            for i in range(n_total)
        ]
        for item in items[:n_down]:
            item.ibkr_position.current_price_local = current  # entry 2100
        positions, portfolio = self._book(items)
        return compute_portfolio_health(
            positions, {}, portfolio, reconciliation_items=items
        )

    def test_drawdown_exact_10pct_counts(self):
        flags = self._drawdown_case(
            current=1890.0, n_down=8, n_total=22
        )  # 10.0%, 36.4%
        assert any("drawdown_breadth" in f for f in flags)

    def test_drawdown_just_below_10pct_does_not_count(self):
        flags = self._drawdown_case(current=1892.0, n_down=8, n_total=22)  # 9.90%
        assert not any("CORRELATED_SELL_EVENT" in f for f in flags)

    def test_drawdown_breadth_just_below_35pct_does_not_fire(self):
        flags = self._drawdown_case(current=1800.0, n_down=8, n_total=23)  # 34.8%
        assert not any("CORRELATED_SELL_EVENT" in f for f in flags)

    def test_drawdown_count_floor_blocks_high_ratio(self):
        flags = self._drawdown_case(
            current=1800.0, n_down=7, n_total=10
        )  # 70%, count 7
        assert not any("CORRELATED_SELL_EVENT" in f for f in flags)

    def test_gbx_consistent_units_no_phantom_drawdown(self):
        # .L positions store BOTH entry and current in GBX (normalized
        # upstream): a 5% dip must not be misread as a -99% GBP/GBX phantom.
        items = []
        for i in range(8):
            item = _make_hold_item_for_health(f"LON{i}.L", conid=100 + i)
            item.analysis.entry_price = 200.0  # GBX
            item.ibkr_position.current_price_local = 190.0  # GBX, -5%
            items.append(item)
        items += [
            _make_hold_item_for_health(f"H{i}.T", conid=300 + i) for i in range(2)
        ]
        positions, portfolio = self._book(items)
        flags = compute_portfolio_health(
            positions, {}, portfolio, reconciliation_items=items
        )
        assert not any("CORRELATED_SELL_EVENT" in f for f in flags)


class TestMacroTriggerPrecedence:
    """Trigger ordering, idempotency, and path exclusivity."""

    @staticmethod
    def _book(items):
        positions = [i.ibkr_position for i in items if i.ibkr_position]
        portfolio = _make_portfolio(value=max(len(positions), 1) * 1000, cash=0)
        portfolio.exchange_weights = {}
        return positions, portfolio

    def _all_triggers_scenario(self):
        # 8 same-day sells / 12 held: window (67%), cumulative (67%), AND
        # broad drawdown all satisfied — window must win the label.
        items = [
            _make_sell_item_on_date(f"S{i}.T", _recent(18), conid=100 + i)
            for i in range(8)
        ] + [_make_hold_item_for_health(f"H{i}.T", conid=300 + i) for i in range(4)]
        for item in items:
            if item.ibkr_position is not None:
                item.ibkr_position.current_price_local = 1700.0  # -19%
        return items

    def test_window_label_wins_when_all_triggers_satisfied(self):
        items = self._all_triggers_scenario()
        positions, portfolio = self._book(items)
        flags = compute_portfolio_health(
            positions, {}, portfolio, reconciliation_items=items
        )
        event_flags = [f for f in flags if "CORRELATED_SELL_EVENT" in f]
        assert len(event_flags) == 1
        assert "[window]" in event_flags[0]

    def test_second_invocation_does_not_double_tag(self):
        items = self._all_triggers_scenario()
        positions, portfolio = self._book(items)
        compute_portfolio_health(positions, {}, portfolio, reconciliation_items=items)
        compute_portfolio_health(positions, {}, portfolio, reconciliation_items=items)
        for item in items:
            assert item.reason.count("[MACRO_WATCH") <= 1

    def test_fresh_detection_suppresses_sustained_path(self):
        from types import SimpleNamespace

        items = self._all_triggers_scenario()
        positions, portfolio = self._book(items)
        event = SimpleNamespace(event_type="GEOPOLITICAL", expiry="2026-12-01")
        flags = compute_portfolio_health(
            positions,
            {},
            portfolio,
            reconciliation_items=items,
            active_macro_events=[event],
        )
        assert any("CORRELATED_SELL_EVENT" in f for f in flags)
        assert not any("ACTIVE_MACRO_EVENT" in f for f in flags)
        demoted = [i for i in items if i.action == "REVIEW"]
        assert demoted
        assert all("correlated event detected" in i.reason for i in demoted)

    def test_weak_fundamentals_price_trigger_is_review_on_both_paths(self):
        from types import SimpleNamespace

        def weak_stop(conid):
            item = _make_sell_item_on_date(
                f"W{conid}.T", "2026-03-05", conid=conid, sell_type="STOP_BREACH"
            )
            item.analysis.health_adj = 35.0
            item.analysis.growth_adj = 40.0
            return item

        # Path 1: fresh detection (8 soft same-day + weak stop / 12 held)
        items = [
            _make_sell_item_on_date(f"S{i}.T", _recent(18), conid=100 + i)
            for i in range(8)
        ] + [weak_stop(900)]
        items += [
            _make_hold_item_for_health(f"H{i}.T", conid=300 + i) for i in range(3)
        ]
        positions, portfolio = self._book(items)
        compute_portfolio_health(positions, {}, portfolio, reconciliation_items=items)
        assert items[8].action == "REVIEW"

        # Path 2: sustained-only override
        items2 = [weak_stop(901)] + [
            _make_hold_item_for_health(f"H{i}.T", conid=400 + i) for i in range(9)
        ]
        positions2, portfolio2 = self._book(items2)
        event = SimpleNamespace(event_type="GEOPOLITICAL", expiry="2026-12-01")
        compute_portfolio_health(
            positions2,
            {},
            portfolio2,
            reconciliation_items=items2,
            active_macro_events=[event],
        )
        assert items2[0].action == "REVIEW"


class TestSustainedOverrideEdgeCases:
    @staticmethod
    def _book(items):
        positions = [i.ibkr_position for i in items if i.ibkr_position]
        portfolio = _make_portfolio(value=max(len(positions), 1) * 1000, cash=0)
        portfolio.exchange_weights = {}
        return positions, portfolio

    def test_degenerate_event_object_uses_defaults(self):
        from types import SimpleNamespace

        items = [_make_sell_item_on_date("S0.T", _recent(18), conid=100)] + [
            _make_hold_item_for_health(f"H{i}.T", conid=300 + i) for i in range(9)
        ]
        positions, portfolio = self._book(items)
        flags = compute_portfolio_health(
            positions,
            {},
            portfolio,
            reconciliation_items=items,
            active_macro_events=[SimpleNamespace()],  # no event_type / expiry
        )
        active = [f for f in flags if "ACTIVE_MACRO_EVENT" in f]
        assert active and "MACRO" in active[0] and "?" in active[0]
        assert items[0].action == "REVIEW"

    def test_multiple_active_events_first_wins(self):
        from types import SimpleNamespace

        items = [_make_sell_item_on_date("S0.T", _recent(18), conid=100)] + [
            _make_hold_item_for_health(f"H{i}.T", conid=300 + i) for i in range(9)
        ]
        positions, portfolio = self._book(items)
        events = [
            SimpleNamespace(event_type="COMMODITY_SHOCK", expiry="2026-09-01"),
            SimpleNamespace(event_type="GEOPOLITICAL", expiry="2026-12-01"),
        ]
        compute_portfolio_health(
            positions,
            {},
            portfolio,
            reconciliation_items=items,
            active_macro_events=events,
        )
        assert "COMMODITY_SHOCK" in items[0].reason

    def test_nothing_to_demote_appends_no_flag(self):
        from types import SimpleNamespace

        items = [
            _make_hold_item_for_health(f"H{i}.T", conid=300 + i) for i in range(10)
        ]
        positions, portfolio = self._book(items)
        event = SimpleNamespace(event_type="GEOPOLITICAL", expiry="2026-12-01")
        flags = compute_portfolio_health(
            positions,
            {},
            portfolio,
            reconciliation_items=items,
            active_macro_events=[event],
        )
        assert not any("ACTIVE_MACRO_EVENT" in f for f in flags)


class TestGeographyConcentration:
    """GEOGRAPHY_CONCENTRATION honors the configured exchange limit."""

    def _flags(self, weight: float, **kwargs) -> list[str]:
        pos = _make_position(market_value_usd=10000, currency="JPY")
        analysis = _make_analysis(age_days=5)
        analysis.health_adj = 75.0
        analysis.growth_adj = 70.0
        portfolio = _make_portfolio(value=100000)
        portfolio.exchange_weights = {"T": weight}
        return compute_portfolio_health(
            [pos], {"7203.T": analysis}, portfolio, **kwargs
        )

    def test_flags_above_default_limit(self):
        flags = self._flags(45.0)
        assert any("GEOGRAPHY_CONCENTRATION: 45.0% in T" in f for f in flags)

    def test_configured_limit_respected(self):
        flags = self._flags(45.0, exchange_limit_pct=50.0)
        assert not any("GEOGRAPHY_CONCENTRATION" in f for f in flags)

    def test_exact_limit_is_silent(self):
        flags = self._flags(40.0)
        assert not any("GEOGRAPHY_CONCENTRATION" in f for f in flags)

    def test_empty_exchange_weights_no_flag_no_crash(self):
        pos = _make_position(market_value_usd=10000, currency="JPY")
        analysis = _make_analysis(age_days=5)
        analysis.health_adj = 75.0
        analysis.growth_adj = 70.0
        portfolio = _make_portfolio(value=100000)
        portfolio.exchange_weights = {}
        flags = compute_portfolio_health([pos], {"7203.T": analysis}, portfolio)
        assert not any("GEOGRAPHY_CONCENTRATION" in f for f in flags)
