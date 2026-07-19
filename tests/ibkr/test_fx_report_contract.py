"""Rendered portfolio-report contracts for FX and valuation language."""

from __future__ import annotations

from scripts.portfolio_manager import format_report
from src.config import config
from src.ibkr.models import (
    AnalysisRecord,
    NormalizedPosition,
    ReconciliationItem,
    TradeBlockData,
)
from src.ibkr.portfolio import normalize_positions
from src.ibkr.portfolio_report_formatting import ReportBuffer
from src.ibkr.reconciler import reconcile
from src.ibkr.ticker import Ticker
from tests.ibkr.reconciler_cases import _make_portfolio


def _position(**overrides) -> NormalizedPosition:
    values = {
        "conid": 1,
        "ticker": Ticker.from_yf("HERDEZ.MX", currency="MXN"),
        "quantity": 100.0,
        "avg_cost_local": 100.0,
        "current_price_local": 105.5,
        "currency": "MXN",
        "market_value_usd": 800.0,
        "unrealized_pnl_usd": -205.0,
        "market_value_basis": "BROKER_USD",
        "unrealized_pnl_basis": "BROKER_USD",
        "valuation_valid": True,
    }
    values.update(overrides)
    return NormalizedPosition(**values)


def _item(position: NormalizedPosition, *, action: str = "HOLD"):
    return ReconciliationItem(
        ticker=position.ticker,
        action=action,
        urgency="LOW",
        reason="Position monitored",
        ibkr_position=position,
    )


def _fx_lines(position: NormalizedPosition) -> list[str]:
    writer = ReportBuffer(
        lines=[],
        show_recommendations=False,
        settled_cash_usd=0.0,
        live_orders=[],
    )
    writer.append_fx_split_line(_item(position))
    return writer.lines


def test_valid_split_is_explicitly_labeled_as_implied_fx_and_basis():
    rendered = "\n".join(_fx_lines(_position()))

    assert "local-price +5.5%" in rendered
    assert "implied FX/basis" in rendered
    assert "USD -20.4%" in rendered
    assert " · FX " not in rendered
    assert "FX effect" not in rendered


def test_local_converted_pnl_omits_unavailable_historical_fx_split():
    assert _fx_lines(_position(unrealized_pnl_basis="LOCAL_CONVERTED")) == []


def test_missing_pnl_omits_return_line_instead_of_printing_zero_return():
    assert (
        _fx_lines(
            _position(
                unrealized_pnl_usd=0.0,
                unrealized_pnl_basis="UNAVAILABLE",
            )
        )
        == []
    )


def test_invalid_valuation_prints_diagnostic_not_numeric_return():
    rendered = "\n".join(
        _fx_lines(
            _position(
                market_value_usd=0.0,
                unrealized_pnl_usd=0.0,
                market_value_basis="UNAVAILABLE",
                unrealized_pnl_basis="UNAVAILABLE",
                valuation_valid=False,
                valuation_issue="Broker market value units could not be verified",
            )
        )
    )

    assert "Broker market value units could not be verified" in rendered
    assert "local-price" not in rendered
    assert "USD +0.0%" not in rendered


def test_implausible_residual_is_rendered_as_withheld_not_as_fx_return():
    rendered = "\n".join(
        _fx_lines(
            _position(
                current_price_local=98.5,
                market_value_usd=312.0,
                unrealized_pnl_usd=-688.0,
            )
        )
    )
    normalized = " ".join(rendered.split())

    assert "FX decomposition withheld" in normalized
    assert "implausible" in normalized
    assert "verify value units or entry FX" in normalized
    assert "implied FX/basis" not in rendered


def test_complete_hold_report_uses_conservative_fx_vocabulary():
    report = format_report(
        [_item(_position())],
        _make_portfolio(),
        show_recommendations=False,
    )

    assert "implied FX/basis" in report
    assert " · FX " not in report
    assert "FX effect" not in report


def test_raw_broker_usd_values_flow_through_to_formatted_implied_split():
    position = normalize_positions(
        [
            {
                "conid": 1,
                "contractDesc": "7203",
                "listingExchange": "TSEJ",
                "position": 100,
                "avgCost": 2_000.0,
                "mktPrice": 2_100.0,
                "mktValue": 1_407.0,
                "unrealizedPnl": 67.0,
                "currency": "JPY",
            }
        ]
    )[0]

    report = format_report(
        [_item(position)],
        _make_portfolio(),
        show_recommendations=False,
    )

    assert position.market_value_basis == "BROKER_USD"
    assert position.unrealized_pnl_basis == "BROKER_USD"
    assert "local-price +5.0%" in report
    assert "implied FX/basis" in report


def test_raw_local_values_do_not_create_formatted_historical_fx_claim():
    position = normalize_positions(
        [
            {
                "conid": 1,
                "contractDesc": "7203",
                "listingExchange": "TSEJ",
                "position": 100,
                "avgCost": 2_000.0,
                "mktPrice": 2_100.0,
                "mktValue": 210_000.0,
                "unrealizedPnl": 10_000.0,
                "currency": "JPY",
            }
        ]
    )[0]

    report = format_report(
        [_item(position)],
        _make_portfolio(),
        show_recommendations=False,
    )

    assert position.unrealized_pnl_basis == "LOCAL_CONVERTED"
    assert "implied FX/basis" not in report
    assert "return:" not in report


def test_unknown_fx_buy_renders_as_review_not_order(monkeypatch):
    monkeypatch.setattr(config, "buy_stability_enabled", False, raising=False)
    analysis = AnalysisRecord(
        ticker="TEST.ST",
        analysis_date="2026-07-19",
        verdict="BUY",
        currency="ZZZ",
        entry_price=100.0,
        current_price=100.0,
        conviction="High",
        trade_block=TradeBlockData(
            action="BUY",
            size_pct=3.0,
            conviction="High",
            entry_price=100.0,
        ),
    )
    portfolio = _make_portfolio()
    items = reconcile(
        [],
        {analysis.ticker: analysis},
        portfolio,
        watchlist_tickers={analysis.ticker},
    )

    report = format_report(items, portfolio, show_recommendations=True)

    assert "REVIEWS" in report
    assert "Watchlist BUY blocked" in report
    assert "no trustworthy local-to-USD FX rate is available" in report
    assert "BUY ORDERS" not in report
