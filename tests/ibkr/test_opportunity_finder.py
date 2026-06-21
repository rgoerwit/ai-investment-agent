"""Collected opportunity-finder tests extracted from reconciler cases."""

import json
from datetime import datetime, timedelta

from src.config import config
from src.ibkr.reconciler import reconcile
from tests.ibkr.reconciler_cases import (
    TestCurrencyAccuracy,
    TestPhase2CashBlockedCandidates,
    _make_analysis,
    _make_portfolio,
)


def _write_prior(results_dir, ticker: str, verdict: str, *, days_ago: int = 1) -> None:
    dt = datetime.now() - timedelta(days=days_ago)
    name = f"{ticker}_{dt.strftime('%Y%m%d_%H%M%S')}_analysis.json"
    pm_text = (
        f"### --- START PM_BLOCK ---\nVERDICT: {verdict}\n### --- END PM_BLOCK ---"
    )
    (results_dir / name).write_text(
        json.dumps({"final_decision": {"decision": pm_text}})
    )


class TestBuyStabilityGateSeam:
    """End-to-end: the opt-in BUY stability gate withholds unstable off-watchlist BUYs."""

    def _buy_count(self, items) -> int:
        return sum(1 for it in items if it.action == "BUY")

    def test_contradicting_history_withholds_buy_when_enabled(
        self, tmp_path, monkeypatch
    ):
        _write_prior(tmp_path, "7203.T", "HOLD")
        monkeypatch.setattr(config, "buy_stability_enabled", True, raising=False)
        monkeypatch.setattr(config, "results_dir", str(tmp_path), raising=False)

        items = reconcile(
            [],
            {"7203.T": _make_analysis(verdict="BUY", age_days=3)},
            _make_portfolio(cash=15000),
        )
        assert self._buy_count(items) == 0  # withheld as unstable

    def test_consistent_history_allows_buy_when_enabled(self, tmp_path, monkeypatch):
        _write_prior(tmp_path, "7203.T", "BUY")
        monkeypatch.setattr(config, "buy_stability_enabled", True, raising=False)
        monkeypatch.setattr(config, "results_dir", str(tmp_path), raising=False)

        items = reconcile(
            [],
            {"7203.T": _make_analysis(verdict="BUY", age_days=3)},
            _make_portfolio(cash=15000),
        )
        assert self._buy_count(items) == 1

    def test_gate_disabled_ignores_history(self, tmp_path, monkeypatch):
        _write_prior(tmp_path, "7203.T", "HOLD")
        monkeypatch.setattr(config, "buy_stability_enabled", False, raising=False)
        monkeypatch.setattr(config, "results_dir", str(tmp_path), raising=False)

        items = reconcile(
            [],
            {"7203.T": _make_analysis(verdict="BUY", age_days=3)},
            _make_portfolio(cash=15000),
        )
        assert self._buy_count(items) == 1  # explicit opt-out restores prior behavior

    def test_default_on_withholds_contradicted_buy_without_explicit_flag(
        self, tmp_path, monkeypatch
    ):
        # Phase 2: the gate is default-ON. With BUY_STABILITY_ENABLED unset, a
        # contradicted off-watchlist BUY must be withheld without any opt-in.
        assert config.buy_stability_enabled is True  # the flipped default
        _write_prior(tmp_path, "7203.T", "HOLD")
        monkeypatch.setattr(config, "results_dir", str(tmp_path), raising=False)
        # buy_stability_enabled intentionally NOT patched — relies on the default.

        items = reconcile(
            [],
            {"7203.T": _make_analysis(verdict="BUY", age_days=3)},
            _make_portfolio(cash=15000),
        )
        assert self._buy_count(items) == 0  # withheld by default-on gate

    def test_marginal_peak_flag_withholds_buy_without_history(
        self, tmp_path, monkeypatch
    ):
        # No prior runs in tmp results_dir → contradiction branch inert. The
        # marginal-tally + unresolved peak flag branch must still withhold.
        monkeypatch.setattr(config, "buy_stability_enabled", True, raising=False)
        monkeypatch.setattr(config, "results_dir", str(tmp_path), raising=False)

        analysis = _make_analysis(
            verdict="BUY",
            age_days=3,
            risk_tally=0.75,
            quality_flag_types=("CYCLICAL_PEAK_WARNING",),
        )
        items = reconcile([], {"7203.T": analysis}, _make_portfolio(cash=15000))
        assert self._buy_count(items) == 0  # withheld via marginal-quality branch

    def test_marginal_without_quality_flag_allows_buy(self, tmp_path, monkeypatch):
        # Marginal tally but no peak/transient flag and no history → BUY proceeds.
        monkeypatch.setattr(config, "buy_stability_enabled", True, raising=False)
        monkeypatch.setattr(config, "results_dir", str(tmp_path), raising=False)

        analysis = _make_analysis(
            verdict="BUY", age_days=3, risk_tally=0.75, quality_flag_types=()
        )
        items = reconcile([], {"7203.T": analysis}, _make_portfolio(cash=15000))
        assert self._buy_count(items) == 1
