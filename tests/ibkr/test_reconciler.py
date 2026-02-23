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
    check_staleness,
    check_stop_breach,
    check_target_hit,
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
        """Held + evaluator says DO_NOT_INITIATE → SELL (conflict)."""
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
        assert "conflict" in items[0].reason.lower()

    def test_held_but_verdict_sell(self):
        """Held + evaluator says SELL → SELL."""
        pos = _make_position()
        analysis = _make_analysis(verdict="SELL")
        items = reconcile([pos], {"7203.T": analysis}, _make_portfolio())
        assert items[0].action == "SELL"

    def test_held_stop_breached(self):
        """Held + price below stop → urgent SELL."""
        pos = _make_position(current_price=1700)
        analysis = _make_analysis(stop_price=1900)
        items = reconcile([pos], {"7203.T": analysis}, _make_portfolio())
        assert items[0].action == "SELL"
        assert items[0].urgency == "HIGH"
        assert "stop" in items[0].reason.lower()
        assert items[0].suggested_order_type == "MKT"

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
