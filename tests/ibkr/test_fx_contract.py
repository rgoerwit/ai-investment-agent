"""Repository-level contracts for FX direction, fallback, and action safety."""

from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest

from src.config import config
from src.fx_normalization import set_fx_rate_cache
from src.ibkr.models import AnalysisRecord, NormalizedPosition, PortfolioSummary
from src.ibkr.reconciler import _populate_portfolio_weights, reconcile
from src.ibkr.reconciliation_rules import _resolve_fx
from src.ibkr.ticker import Ticker
from tests.ibkr.reconciler_cases import (
    _FakeFxRateCache,
    _make_analysis,
    _make_portfolio,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _deterministic_fx_cache():
    """These tests exercise _resolve_fx's sentinel/unit-mismatch repair
    logic, not live FX fetching — pin a fixed, network-free cache."""
    set_fx_rate_cache(_FakeFxRateCache())
    yield
    set_fx_rate_cache(None)


def _analysis(currency: str, rate: float | None) -> AnalysisRecord:
    analysis = _make_analysis(ticker="TEST.ST", verdict="BUY", current_price=100.0)
    analysis.currency = currency
    analysis.fx_rate_to_usd = rate
    analysis.entry_price = 100.0
    analysis.trade_block.entry_price = 100.0
    return analysis


@pytest.mark.parametrize(
    ("currency", "inverse_rate", "expected"),
    [
        ("JPY", 150.0, 0.0067),
        ("KRW", 1_330.0, 0.00075),
        ("TWD", 31.0, 0.032),
        ("HKD", 7.8, 0.128),
    ],
)
def test_inverse_usd_local_quotes_are_replaced(currency, inverse_rate, expected):
    assert _resolve_fx(_analysis(currency, inverse_rate)) == pytest.approx(expected)


@pytest.mark.parametrize("bad_rate", [0.0, -1.0, math.nan, math.inf, -math.inf])
def test_invalid_saved_rate_uses_known_currency_fallback(bad_rate):
    assert _resolve_fx(_analysis("JPY", bad_rate)) == pytest.approx(0.0067)


def test_currency_code_is_normalized_before_resolution():
    assert _resolve_fx(_analysis(" jpy ", None)) == pytest.approx(0.0067)


def test_explicit_rate_for_currency_outside_fallback_table_is_preserved():
    assert _resolve_fx(_analysis("KWD", 3.25)) == pytest.approx(3.25)


@pytest.mark.parametrize("rate", [None, 1.0, 0.0, math.nan])
def test_unknown_currency_without_trustworthy_explicit_rate_returns_none(rate):
    assert _resolve_fx(_analysis("ZZZ", rate)) is None


@pytest.mark.parametrize("saved_rate", [None, 1.0, 0.0, math.nan, 150.0])
def test_usd_always_resolves_to_identity(saved_rate):
    assert _resolve_fx(_analysis("USD", saved_rate)) == 1.0


def test_unknown_fx_blocks_watchlist_buy_as_data_quality_review(monkeypatch):
    monkeypatch.setattr(config, "buy_stability_enabled", False, raising=False)
    analysis = _analysis("ZZZ", None)

    items = reconcile(
        [],
        {analysis.ticker: analysis},
        _make_portfolio(),
        watchlist_tickers={analysis.ticker},
    )

    item = next(item for item in items if item.ticker.yf == analysis.ticker)
    assert item.action == "REVIEW"
    assert item.action_basis == "DATA_QUALITY"
    assert item.suggested_quantity is None
    assert item.cash_impact_usd == 0.0
    assert "FX rate" in item.reason


def test_unknown_fx_blocks_offwatch_buy_as_data_quality_review(monkeypatch):
    monkeypatch.setattr(config, "buy_stability_enabled", False, raising=False)
    analysis = _analysis("ZZZ", None)

    items = reconcile([], {analysis.ticker: analysis}, _make_portfolio())

    item = next(item for item in items if item.ticker.yf == analysis.ticker)
    assert item.action == "REVIEW"
    assert item.action_basis == "DATA_QUALITY"
    assert item.suggested_quantity is None
    assert item.cash_impact_usd == 0.0
    assert "FX rate" in item.reason


def test_invalid_position_is_excluded_from_all_concentration_weights():
    valid = NormalizedPosition(
        conid=1,
        ticker=Ticker.from_yf("AAPL"),
        quantity=10,
        market_value_usd=1_000.0,
        currency="USD",
        valuation_valid=True,
    )
    invalid = NormalizedPosition(
        conid=2,
        ticker=Ticker.from_yf("7203.T", currency="JPY"),
        quantity=100,
        market_value_usd=100_000.0,
        currency="JPY",
        valuation_valid=False,
        valuation_issue="unit mismatch",
    )
    portfolio = PortfolioSummary(
        portfolio_value_usd=101_000.0,
        cash_balance_usd=0.0,
    )
    analyses = {
        "AAPL": _make_analysis(ticker="AAPL"),
        "7203.T": _make_analysis(ticker="7203.T"),
    }

    sector_weights, exchange_weights = _populate_portfolio_weights(
        [valid, invalid],
        analyses,
        portfolio,
        {},
    )

    assert sum(sector_weights.values()) == pytest.approx(100.0)
    assert exchange_weights == {"US": pytest.approx(100.0)}
    assert portfolio.currency_weights == {"USD": pytest.approx(100.0)}


def test_no_fx_fallback_lookup_in_repo_silently_defaults_to_one():
    offenders: list[str] = []
    paths = [*_REPO_ROOT.glob("src/**/*.py"), *_REPO_ROOT.glob("scripts/**/*.py")]
    for path in paths:
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(
                node.func, ast.Attribute
            ):
                continue
            if node.func.attr != "get" or len(node.args) < 2:
                continue
            owner = node.func.value
            default = node.args[1]
            if (
                isinstance(owner, ast.Name)
                and owner.id == "FALLBACK_RATES_TO_USD"
                and isinstance(default, ast.Constant)
                and default.value == 1.0
            ):
                offenders.append(f"{path.relative_to(_REPO_ROOT)}:{node.lineno}")

    assert offenders == [], (
        "Unknown currencies must return None, never silently use USD identity: "
        + ", ".join(offenders)
    )


class TestFxResolutionInsideAnEventLoop:
    """A cold FX cache inside a running loop must degrade, not abort the report.

    `recommendation_service.build_bundle` is driven by `asyncio.run`, and the
    reconciler reaches `_fx_rate_for_currency` synchronously from inside it.
    `resolve_rates_sync` is `asyncio.run` and raises there by its own documented
    contract. The miss is normally impossible because `normalize_positions`
    batch-warms every held currency first -- but a `--read-only` run has no
    positions to normalize, so an opportunity in an unseen currency arrives cold.
    That aborted the entire portfolio report with
    "asyncio.run() cannot be called from a running event loop".
    """

    @staticmethod
    def _cold_cache():
        from src.fx_normalization import FxRateCache

        return FxRateCache()

    def test_cold_miss_inside_a_loop_falls_back_instead_of_raising(self, monkeypatch):
        import asyncio

        from src.fx_normalization import set_fx_rate_cache
        from src.ibkr.reconciliation_rules import _fx_rate_for_currency

        set_fx_rate_cache(self._cold_cache())

        async def _resolve_under_a_running_loop():
            return _fx_rate_for_currency("JPY")

        rate = asyncio.run(_resolve_under_a_running_loop())

        assert rate is not None
        assert 0 < rate < 1  # JPY->USD, from the static table

    def test_warm_cache_is_still_served_without_io(self):
        import asyncio

        from src.fx_normalization import set_fx_rate_cache
        from src.ibkr.reconciliation_rules import _fx_rate_for_currency

        set_fx_rate_cache(_FakeFxRateCache())

        async def _resolve_under_a_running_loop():
            return _fx_rate_for_currency("JPY")

        assert asyncio.run(_resolve_under_a_running_loop()) is not None

    def test_outside_a_loop_the_live_first_path_is_unchanged(self, monkeypatch):
        from src.fx_normalization import set_fx_rate_cache
        from src.ibkr.reconciliation_rules import _fx_rate_for_currency

        cache = self._cold_cache()
        calls: list[list[str]] = []

        def _fake_sync(currencies, to_currency="USD"):
            calls.append(list(currencies))
            return {"JPY": (0.0065, "test")}

        monkeypatch.setattr(cache, "resolve_rates_sync", _fake_sync)
        set_fx_rate_cache(cache)

        assert _fx_rate_for_currency("JPY") == 0.0065
        assert calls == [["JPY"]]
