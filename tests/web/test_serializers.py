from __future__ import annotations

from src.ibkr.models import (
    AnalysisRecord,
    NormalizedPosition,
    PortfolioSummary,
    ReconciliationItem,
    TradeBlockData,
)
from src.ibkr.portfolio_presentation import build_cash_summary, build_live_order_note
from src.ibkr.recommendation_service import PortfolioRecommendationBundle
from src.ibkr.screening_freshness import ScreeningFreshnessSummary
from src.ibkr.ticker import Ticker
from src.web.ibkr_dashboard.serializers import (
    serialize_dashboard_snapshot,
    serialize_equity_drilldown,
)
from tests.factories.ibkr import make_analysis, make_position


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
    assert payload["actions"]["dip_watch"][0]["ticker_yf"] == "MEGP.L"
    assert payload["actions"]["dip_watch"][0]["source"] == "held_buy_pullback"
    assert payload["overview"]["candidates"] == 1
    assert payload["freshness_overview"]["blocking_now"] == 1
    assert payload["actions"]["action_sections"][0]["key"] == "sell_recommendations"
    assert payload["actions"]["action_sections"][1]["key"] == "sell_related_reviews"
    assert (
        payload["actions"]["action_sections"][0]["items"][0]["sell_type_label"]
        == "FUNDAMENTAL FAILURE"
    )


def test_serialize_dashboard_snapshot_handles_empty_lists(sample_bundle):
    sample_bundle.live_orders = []
    payload = serialize_dashboard_snapshot(sample_bundle)
    assert payload["orders"] == []
    assert payload["freshness"]["candidate_blocked"] == []
    assert payload["macro_alert"] is None
    assert payload["screening_freshness"]["status"] == "missing"


def test_serialize_dashboard_snapshot_includes_screening_freshness(sample_bundle):
    sample_bundle.screening_freshness = ScreeningFreshnessSummary(
        status="stale",
        screening_date="2026-01-05",
        completed_at="2026-01-05T10:30:00Z",
        age_days=90,
        stale_after_days=90,
        candidate_count=245,
        buy_count=12,
    )
    payload = serialize_dashboard_snapshot(sample_bundle)
    assert payload["screening_freshness"]["status"] == "stale"
    assert payload["screening_freshness"]["screening_date"] == "2026-01-05"
    assert payload["screening_freshness"]["candidate_count"] == 245


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


def test_serialize_dashboard_snapshot_canonicalizes_sector_weights(sample_bundle):
    sample_bundle.portfolio.sector_weights = {
        "Technology": 8.0,
        "Information Technology": 1.5,
        "Healthcare": 3.0,
        "Health Care": 1.0,
    }

    payload = serialize_dashboard_snapshot(sample_bundle)

    assert payload["portfolio"]["sector_weights"] == {
        "Information Technology": 9.5,
        "Health Care": 4.0,
    }


def test_serialize_dashboard_snapshot_uses_shared_live_order_annotations(sample_bundle):
    payload = serialize_dashboard_snapshot(sample_bundle)
    sell_item = next(item for item in sample_bundle.items if item.ticker.yf == "7203.T")

    assert payload["actions"]["sell_hard"][0][
        "live_order_note"
    ] == build_live_order_note(
        sell_item,
        sample_bundle.live_orders,
    )


def test_serialize_dashboard_snapshot_includes_profit_take_fields(sample_bundle):
    sample_bundle.items.append(
        ReconciliationItem(
            ticker=Ticker.from_yf("6758.T"),
            action="SELL",
            reason="Profit take",
            urgency="LOW",
            sell_type="PROFIT_TAKE",
            cost_basis_return_pct=42.5,
            profit_take_reasons=("capital_idle_cash_severe",),
            ibkr_position=NormalizedPosition(
                conid=1,
                ticker=Ticker.from_yf("6758.T"),
                quantity=100,
                avg_cost_local=1000,
                current_price_local=1425,
                tax_term="LONG_TERM",
            ),
        )
    )

    payload = serialize_dashboard_snapshot(sample_bundle)
    assert payload["actions"]["sell_profit_take"] == []
    row = payload["actions"]["review_profit_take"][0]

    assert row["action"] == "REVIEW"
    assert row["sell_type"] == "PROFIT_TAKE"
    assert row["cost_basis_return_pct"] == 42.5
    assert row["profit_take_reasons"] == ["capital_idle_cash_severe"]


def test_serialize_dashboard_snapshot_includes_currency_repair_metadata(sample_bundle):
    sample_bundle.items[0].analysis = AnalysisRecord(
        ticker="7203.T",
        analysis_date="2026-03-01",
        verdict="BUY",
        currency="JPY",
        currency_source="repair_on_load",
        currency_repaired=True,
        currency_repair_reason="legacy_snapshot_usd_default",
        trade_block=TradeBlockData(),
    )

    payload = serialize_dashboard_snapshot(sample_bundle)
    analysis = payload["actions"]["sell_hard"][0]["analysis"]

    assert analysis["currency"] == "JPY"
    assert analysis["currency_source"] == "repair_on_load"
    assert analysis["currency_repaired"] is True
    assert analysis["currency_repair_reason"] == "legacy_snapshot_usd_default"


def test_serialize_dashboard_snapshot_exposes_ticker_resolution_provenance(
    sample_bundle,
):
    sample_bundle.items[0].ibkr_position = sample_bundle.items[
        0
    ].ibkr_position.model_copy(
        update={
            "ticker_identity_verified": False,
            "ticker_resolution_source": "yfinance_search",
        }
    )

    payload = serialize_dashboard_snapshot(sample_bundle)
    row = payload["actions"]["review"][0]
    position = row["position"]

    assert row["action"] == "REVIEW"
    assert position["ticker_identity_verified"] is False
    assert position["ticker_resolution_source"] == "yfinance_search"


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


# ── Concentration parity: dashboard payload mirrors the CLI optimizer ─────────


def _watch_buy(ticker: str, conviction: str = "Medium") -> ReconciliationItem:
    analysis = make_analysis(ticker=ticker, conviction=conviction, size_pct=4.0)
    return ReconciliationItem(
        ticker=ticker,
        action="BUY",
        urgency="MEDIUM",
        reason=f"Watchlist BUY — {conviction} conviction",
        analysis=analysis,
        suggested_quantity=100,
        suggested_price=100.0,
        cash_impact_usd=-1752.0,
        is_watchlist=True,
    )


def _offwatch_buy(ticker: str) -> ReconciliationItem:
    analysis = make_analysis(ticker=ticker, conviction="Medium", size_pct=4.0)
    return ReconciliationItem(
        ticker=ticker,
        action="BUY",
        urgency="MEDIUM",
        reason="New BUY — Medium conviction",
        analysis=analysis,
        suggested_quantity=100,
        suggested_price=100.0,
        cash_impact_usd=-1500.0,
        is_watchlist=False,
    )


def _dip_hold(ticker: str) -> ReconciliationItem:
    analysis = make_analysis(
        ticker=ticker,
        verdict="BUY",
        health_adj=60.0,
        growth_adj=60.0,
        entry_price=2100.0,
        stop_price=1700.0,
        target_1=2600.0,
        current_price=1800.0,
    )
    return ReconciliationItem(
        ticker=ticker,
        action="HOLD",
        urgency="LOW",
        reason="Held — dip",
        ibkr_position=make_position(ticker=ticker, current_price=1800.0),
        analysis=analysis,
    )


def _concentration_bundle(**overrides) -> PortfolioRecommendationBundle:
    """T exchange already over the 40% limit: the on-watch T buy is screened,
    the off-watch T buy is withheld, and the sub-★★★ T dip is dropped."""
    portfolio = PortfolioSummary(
        portfolio_value_usd=100000,
        cash_balance_usd=15000,
        settled_cash_usd=10000,
        available_cash_usd=8000,
        position_count=1,
        exchange_weights={"T": 45.0},
    )
    values: dict = {
        "portfolio": portfolio,
        "items": [
            _watch_buy("7203.T"),
            _watch_buy("MEGP.L"),
            _offwatch_buy("9984.T"),
            _dip_hold("6758.T"),
        ],
        "watchlist_tickers": {"7203.T", "MEGP.L"},
        "watchlist_total": 2,
    }
    values.update(overrides)
    return PortfolioRecommendationBundle(**values)


def test_dashboard_hides_concentration_screened_watchlist_buy():
    payload = serialize_dashboard_snapshot(_concentration_bundle())
    actions = payload["actions"]

    buy_tickers = [row["ticker_yf"] for row in actions["watchlist_buy"]]
    assert buy_tickers == ["MEGP.L"]
    removes = {row["ticker_yf"]: row for row in actions["watchlist_remove"]}
    assert removes["7203.T"]["removal_reason"] == "concentration_displaced"
    assert "overweight exchange T" in removes["7203.T"]["concentration"]
    # Header chips and cash agree with the filtered lists (no split-brain).
    assert payload["overview"]["new_buys"] == 1
    assert payload["summary_counts"]["buys"] == 1
    assert payload["cash_summary"]["recommended_buy_cost_usd"] == 1752.0


def test_dashboard_withholds_offwatch_and_screens_dip():
    payload = serialize_dashboard_snapshot(_concentration_bundle())
    actions = payload["actions"]

    assert [row["ticker_yf"] for row in actions["watchlist_withheld"]] == ["9984.T"]
    assert "overweight exchange T" in actions["watchlist_withheld"][0]["concentration"]
    assert "9984.T" not in [row["ticker_yf"] for row in actions["watchlist_candidate"]]
    assert "6758.T" not in [row["ticker_yf"] for row in actions["dip_watch"]]
    assert payload["summary_counts"]["watchlist_withheld"] == 1


def test_dashboard_screen_inactive_without_weights():
    bundle = _concentration_bundle()
    bundle.portfolio.exchange_weights = {}
    payload = serialize_dashboard_snapshot(bundle)
    actions = payload["actions"]

    assert sorted(row["ticker_yf"] for row in actions["watchlist_buy"]) == [
        "7203.T",
        "MEGP.L",
    ]
    assert [row["ticker_yf"] for row in actions["watchlist_candidate"]] == ["9984.T"]
    assert "6758.T" in [row["ticker_yf"] for row in actions["dip_watch"]]
    assert actions["watchlist_remove"] == []
    assert actions["watchlist_withheld"] == []


def test_dashboard_unavailable_watchlist_is_additions_only():
    bundle = _concentration_bundle(watchlist_unavailable=True)
    payload = serialize_dashboard_snapshot(bundle)
    actions = payload["actions"]

    assert actions["watchlist_buy"] == []  # no keep authority
    assert actions["watchlist_remove"] == []  # no removal claims
    assert payload["overview"]["new_buys"] == 0


def test_dashboard_custom_bundle_limits_honored():
    bundle = _concentration_bundle(exchange_limit_pct=60.0)
    payload = serialize_dashboard_snapshot(bundle)
    actions = payload["actions"]

    assert "7203.T" in [row["ticker_yf"] for row in actions["watchlist_buy"]]
    assert actions["watchlist_remove"] == []
    assert actions["watchlist_withheld"] == []
    assert "6758.T" in [row["ticker_yf"] for row in actions["dip_watch"]]


def test_dashboard_emits_structured_breaches_alongside_string():
    payload = serialize_dashboard_snapshot(_concentration_bundle())
    actions = payload["actions"]

    withheld = actions["watchlist_withheld"][0]
    assert isinstance(withheld["breaches"], list) and withheld["breaches"]
    breach = withheld["breaches"][0]
    assert set(breach) == {"dimension", "key", "projected_pct", "limit_pct"}
    assert breach["dimension"] == "exchange"
    assert breach["key"] == "T"
    assert breach["limit_pct"] == 40.0

    removed = {row["ticker_yf"]: row for row in actions["watchlist_remove"]}
    assert removed["7203.T"]["breaches"][0]["dimension"] == "exchange"


def test_dashboard_snapshot_includes_concentration_limits(sample_bundle):
    payload = serialize_dashboard_snapshot(sample_bundle)

    assert payload["concentration_limits"] == {
        "sector": sample_bundle.sector_limit_pct,
        "exchange": sample_bundle.exchange_limit_pct,
    }


def test_dashboard_snapshot_passes_through_errors(sample_bundle):
    sample_bundle.errors["live_orders"] = "IBKR session not authenticated"
    payload = serialize_dashboard_snapshot(sample_bundle)

    assert payload["errors"]["live_orders"] == "IBKR session not authenticated"
