from __future__ import annotations

from src.ibkr.portfolio_presentation import build_cash_summary, build_live_order_note
from src.web.ibkr_dashboard.serializers import (
    serialize_dashboard_snapshot,
    serialize_equity_drilldown,
)


def test_serialize_dashboard_snapshot_shapes_payload(sample_bundle):
    macro_alert = {"detected": True, "headline": "macro headline"}
    payload = serialize_dashboard_snapshot(
        sample_bundle,
        status="ready",
        fetched_at="2026-03-28T12:00:00Z",
        refreshing=True,
        macro_alert=macro_alert,
    )
    assert payload["status"] == "ready"
    assert (
        payload["portfolio"]["net_liquidation_usd"]
        == sample_bundle.portfolio.portfolio_value_usd
    )
    assert "sell_hard" in payload["actions"]
    assert "watchlist_candidate" in payload["actions"]
    assert payload["summary_counts"]["buys"] == 1
    assert payload["summary_counts"]["candidates"] == 1
    assert "ticker_yf" in payload["actions"]["sell_hard"][0]
    assert isinstance(payload["portfolio"]["settled_cash_usd"], int | float)
    assert payload["macro_alert"] == macro_alert
    assert payload["refreshing"] is True
    assert payload["actions"]["watchlist_candidate"][0]["ticker_yf"] == "BMW.DE"
    assert payload["overview"]["candidates"] == 1
    assert payload["freshness_overview"]["blocking_now"] == 1


def test_serialize_dashboard_snapshot_handles_empty_lists(sample_bundle):
    sample_bundle.live_orders = []
    payload = serialize_dashboard_snapshot(sample_bundle)
    assert payload["orders"] == []
    assert payload["freshness"]["candidate_blocked"] == []
    assert payload["macro_alert"] is None


def test_serialize_dashboard_snapshot_uses_shared_cash_summary(sample_bundle):
    payload = serialize_dashboard_snapshot(sample_bundle)
    shared = build_cash_summary(sample_bundle.items, sample_bundle.portfolio)

    assert (
        payload["cash_summary"]["pending_inflows_total_usd"]
        == shared.pending_inflows_total_usd
    )
    assert (
        payload["cash_summary"]["pending_inflows"][0]["ticker_yf"]
        == shared.pending_inflows[0].ticker_yf
    )


def test_serialize_dashboard_snapshot_uses_shared_live_order_annotations(sample_bundle):
    payload = serialize_dashboard_snapshot(sample_bundle)
    sell_item = next(item for item in sample_bundle.items if item.ticker.yf == "7203.T")

    assert payload["actions"]["sell_hard"][0][
        "live_order_note"
    ] == build_live_order_note(
        sell_item,
        sample_bundle.live_orders,
    )


def test_serialize_equity_drilldown_includes_structured_and_markdown(sample_bundle):
    item = next(item for item in sample_bundle.items if item.ticker.yf == "MEGP.L")
    payload = serialize_equity_drilldown(
        item,
        live_orders=sample_bundle.live_orders,
        analysis_json={"prediction_snapshot": {"ticker": "MEGP.L"}},
        report_markdown_html="<p>report</p>",
        report_markdown_path="results/MEGP.L.md",
        article_markdown_html=None,
        article_markdown_path=None,
    )
    assert payload["structured"]["prediction_snapshot"]["ticker"] == "MEGP.L"
    assert payload["report_markdown_html"] == "<p>report</p>"
    assert payload["analysis"]["ticker"] == "MEGP.L"
    assert "file_path" not in payload["analysis"]
    assert payload["note"] is None
    assert payload["live_order_note"] is None
