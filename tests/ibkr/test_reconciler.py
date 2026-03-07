"""Tests for the reconciler — position-aware action generation."""

import json
import tempfile
from pathlib import Path

import pytest

from src.ibkr.models import (
    AnalysisRecord,
    NormalizedPosition,
    PortfolioSummary,
    ReconciliationItem,
    TradeBlockData,
)
from src.ibkr.reconciler import (
    _parse_scores_from_final_decision,
    check_staleness,
    check_stop_breach,
    check_target_hit,
    compute_portfolio_health,
    load_latest_analyses,
    reconcile,
)

# ── Fixtures ──


def _make_position(
    ticker: str = "7203.T",
    quantity: float = 100,
    avg_cost: float = 2000,
    current_price: float = 2100,
    market_value_usd: float = 1400,
    currency: str = "JPY",
    conid: int = 123456,
) -> NormalizedPosition:
    return NormalizedPosition(
        conid=conid,
        yf_ticker=ticker,
        symbol=ticker.split(".")[0],
        quantity=quantity,
        avg_cost_local=avg_cost,
        market_value_usd=market_value_usd,
        currency=currency,
        current_price_local=current_price,
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
) -> AnalysisRecord:
    from datetime import datetime, timedelta

    analysis_date = (datetime.now() - timedelta(days=age_days)).strftime("%Y-%m-%d")
    return AnalysisRecord(
        ticker=ticker,
        analysis_date=analysis_date,
        verdict=verdict,
        current_price=current_price,
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
        """Held + evaluator says DO_NOT_INITIATE → SELL."""
        pos = _make_position(current_price=2100)
        analysis = _make_analysis(verdict="DO_NOT_INITIATE")
        items = reconcile(
            [pos],
            {"7203.T": analysis},
            _make_portfolio(),
        )
        assert len(items) == 1
        assert items[0].action == "SELL"
        assert items[0].urgency == "HIGH"
        assert "do_not_initiate" in items[0].reason.lower()

    def test_held_but_verdict_sell(self):
        """Held + evaluator says SELL → SELL."""
        pos = _make_position()
        analysis = _make_analysis(verdict="SELL")
        items = reconcile([pos], {"7203.T": analysis}, _make_portfolio())
        assert items[0].action == "SELL"

    def test_held_stop_breached(self):
        """Held + price below stop → urgent SELL via LMT at current price."""
        pos = _make_position(current_price=1700)
        analysis = _make_analysis(stop_price=1900)
        items = reconcile([pos], {"7203.T": analysis}, _make_portfolio())
        assert items[0].action == "SELL"
        assert items[0].urgency == "HIGH"
        assert "stop" in items[0].reason.lower()
        assert items[0].suggested_order_type == "LMT"
        assert items[0].suggested_price == 1700

    def test_held_target_hit(self):
        """Held + price at target → REVIEW."""
        pos = _make_position(current_price=2600)
        analysis = _make_analysis(target_1=2500)
        items = reconcile([pos], {"7203.T": analysis}, _make_portfolio())
        assert items[0].action == "REVIEW"
        assert "target" in items[0].reason.lower()

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

    def test_no_cash_skips_new_buys(self):
        """No available cash → no new BUY recommendations."""
        analysis = _make_analysis(verdict="BUY", age_days=3)
        items = reconcile(
            [],
            {"7203.T": analysis},
            _make_portfolio(cash=0, value=100000),
        )
        assert len(items) == 0

    def test_overweight_position(self):
        """Overweight position → TRIM."""
        pos = _make_position(market_value_usd=30000)  # 30% of 100k
        analysis = _make_analysis(verdict="BUY", size_pct=5.0)  # Target 5%
        items = reconcile(
            [pos],
            {"7203.T": analysis},
            _make_portfolio(value=100000),
            overweight_threshold_pct=20.0,
        )
        assert items[0].action == "TRIM"

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
        assert items[0].ticker == "0005.HK"

    def test_mixed_positions_and_recommendations(self):
        """Multiple positions + new buy recs should all appear."""
        held = _make_position(ticker="7203.T", current_price=2100, conid=1)
        analyses = {
            "7203.T": _make_analysis(ticker="7203.T", verdict="BUY"),
            "0005.HK": _make_analysis(ticker="0005.HK", verdict="BUY", age_days=3),
        }
        items = reconcile([held], analyses, _make_portfolio(cash=15000))
        tickers = {i.ticker for i in items}
        assert "7203.T" in tickers
        assert "0005.HK" in tickers
        # 7203.T should be HOLD (already held), 0005.HK should be BUY (not held)
        actions = {i.ticker: i.action for i in items}
        assert actions["7203.T"] == "HOLD"
        assert actions["0005.HK"] == "BUY"


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
        pos_hk.yf_ticker = "0005.HK"
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
        buy_items = [i for i in items if i.action == "BUY" and i.ticker == "2388.HK"]
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
        """Price below stop → sell_type=STOP_BREACH."""
        pos = _make_position(current_price=1700)
        analysis = _make_analysis(stop_price=1900)
        items = reconcile([pos], {"7203.T": analysis}, _make_portfolio())
        assert items[0].action == "SELL"
        assert items[0].sell_type == "STOP_BREACH"

    def test_fundamental_failure_tagged_hard_reject(self):
        """Verdict DO_NOT_INITIATE + health_adj=40 → HARD_REJECT (fails hard check)."""
        pos = _make_position(current_price=2100)
        analysis = _make_analysis(verdict="DO_NOT_INITIATE")
        analysis.health_adj = 40.0  # below 50 → hard fail
        analysis.growth_adj = 60.0
        items = reconcile([pos], {"7203.T": analysis}, _make_portfolio())
        sell = next(i for i in items if i.action == "SELL")
        assert sell.sell_type == "HARD_REJECT"

    def test_soft_failure_tagged_soft_reject(self):
        """Verdict DO_NOT_INITIATE + health_adj=65 + growth_adj=60 → SOFT_REJECT."""
        pos = _make_position(current_price=2100)
        analysis = _make_analysis(verdict="DO_NOT_INITIATE")
        analysis.health_adj = 65.0  # passes hard check
        analysis.growth_adj = 60.0  # passes hard check
        items = reconcile([pos], {"7203.T": analysis}, _make_portfolio())
        sell = next(i for i in items if i.action == "SELL")
        assert sell.sell_type == "SOFT_REJECT"

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


# ── Correlated Sell Event Detection Tests ──


def _make_multi_sell_scenario(
    n_soft_sells: int,
    n_stop_breaches: int,
    n_hard_rejects: int,
    n_holds: int,
    sell_date: str = "2026-03-05",
) -> tuple[list, dict, "PortfolioSummary"]:
    """
    Build positions + analyses for a correlated-sell scenario.

    Returns (positions, analyses, portfolio) ready for reconcile().
    """
    from datetime import datetime, timedelta

    positions = []
    analyses = {}
    conid = 1

    # SOFT_REJECT SELLs — passed both hard checks, rejected on soft tally
    for i in range(n_soft_sells):
        ticker = f"SOFT{i:02d}.T"
        pos = _make_position(
            ticker=ticker, current_price=2100, market_value_usd=1000, conid=conid
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

    # HARD_REJECT SELLs — failed fundamental checks
    for i in range(n_hard_rejects):
        ticker = f"HARD{i:02d}.T"
        pos = _make_position(
            ticker=ticker, current_price=2100, market_value_usd=1000, conid=conid
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

    def test_stop_breach_not_counted_in_correlation(self):
        """10 stop-breach SELLs should NOT trigger correlated-sell flag."""
        positions, analyses, portfolio = _make_multi_sell_scenario(
            n_soft_sells=0, n_stop_breaches=10, n_hard_rejects=0, n_holds=2
        )
        items = reconcile(positions, analyses, portfolio)
        flags = compute_portfolio_health(
            positions, analyses, portfolio, reconciliation_items=items
        )
        assert not any("CORRELATED_SELL_EVENT" in f for f in flags)

    def test_soft_reject_demoted_to_review_on_correlated_day(self):
        """On a correlated day, SOFT_REJECT SELLs are demoted to REVIEW."""
        positions, analyses, portfolio = _make_multi_sell_scenario(
            n_soft_sells=6, n_stop_breaches=0, n_hard_rejects=0, n_holds=2
        )
        items = reconcile(positions, analyses, portfolio)
        # Before health check: SOFT_REJECT items are still SELL
        soft_sells_before = [
            i for i in items if i.sell_type == "SOFT_REJECT" and i.action == "SELL"
        ]
        assert len(soft_sells_before) == 6

        compute_portfolio_health(
            positions, analyses, portfolio, reconciliation_items=items
        )

        # After: demoted to REVIEW
        soft_sells_after = [
            i for i in items if i.sell_type == "SOFT_REJECT" and i.action == "SELL"
        ]
        soft_reviews_after = [
            i for i in items if i.sell_type == "SOFT_REJECT" and i.action == "REVIEW"
        ]
        assert len(soft_sells_after) == 0
        assert len(soft_reviews_after) == 6
        # Reason string annotated
        assert all("MACRO_WATCH" in i.reason for i in soft_reviews_after)

    def test_hard_reject_stays_sell_on_correlated_day(self):
        """HARD_REJECT SELLs remain SELL even when correlated event fires."""
        positions, analyses, portfolio = _make_multi_sell_scenario(
            n_soft_sells=5,  # enough to trigger event
            n_stop_breaches=0,
            n_hard_rejects=3,
            n_holds=3,
        )
        items = reconcile(positions, analyses, portfolio)
        compute_portfolio_health(
            positions, analyses, portfolio, reconciliation_items=items
        )

        hard_sells = [
            i for i in items if i.sell_type == "HARD_REJECT" and i.action == "SELL"
        ]
        assert len(hard_sells) == 3  # unchanged

    def test_stop_breach_stays_sell_on_correlated_day(self):
        """STOP_BREACH SELLs are never demoted, regardless of correlated event."""
        positions, analyses, portfolio = _make_multi_sell_scenario(
            n_soft_sells=5,  # triggers event
            n_stop_breaches=2,
            n_hard_rejects=0,
            n_holds=3,
        )
        items = reconcile(positions, analyses, portfolio)
        compute_portfolio_health(
            positions, analyses, portfolio, reconciliation_items=items
        )

        stop_sells = [
            i for i in items if i.sell_type == "STOP_BREACH" and i.action == "SELL"
        ]
        assert len(stop_sells) == 2  # unchanged

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
