from __future__ import annotations

import pytest

from src.ibkr.models import ReconciliationItem
from src.ibkr.portfolio_presentation import (
    SELL_RECOMMENDATIONS_TITLE,
    SELL_RELATED_REVIEWS_TITLE,
    build_action_display_sections,
    build_action_summary_counts,
    get_sell_type_label,
    group_portfolio_actions,
)
from src.ibkr.ticker import Ticker
from src.sector_normalization import aggregate_sector_weights


def test_group_portfolio_actions_matches_cli_buckets(sample_bundle):
    groups = group_portfolio_actions(
        sample_bundle.items,
        watchlist_tickers=sample_bundle.watchlist_tickers,
    )

    assert [item.ticker.yf for item in groups.hard_sells] == ["7203.T"]
    assert [item.ticker.yf for item in groups.macro_reviews] == ["5285.T"]
    assert [item.ticker.yf for item in groups.holds_real] == ["MEGP.L"]
    assert [item.ticker.yf for item in groups.new_buys] == ["ASML.AS"]
    assert [item.ticker.yf for item in groups.watchlist_candidates] == ["BMW.DE"]
    assert [item.ticker.yf for item in groups.dip_candidates] == ["MEGP.L"]


def test_build_action_summary_counts_separates_buys_from_candidates(sample_bundle):
    groups = group_portfolio_actions(
        sample_bundle.items,
        watchlist_tickers=sample_bundle.watchlist_tickers,
    )
    counts = build_action_summary_counts(groups)

    assert counts["SELL"] == 1
    assert counts["BUY"] == 1
    assert counts["CANDIDATES"] == 1
    assert counts["HOLD"] == 1
    assert counts["MACRO_WATCH"] == 1


def test_legacy_profit_take_sell_is_downgraded_to_review():
    sell = ReconciliationItem(
        ticker=Ticker.from_yf("7203.T"),
        action="SELL",
        reason="Profit take",
        urgency="LOW",
        sell_type="PROFIT_TAKE",
    )
    review = ReconciliationItem(
        ticker=Ticker.from_yf("6758.T"),
        action="REVIEW",
        reason="Profit take review",
        urgency="MEDIUM",
        sell_type="PROFIT_TAKE",
    )

    groups = group_portfolio_actions([sell, review])
    counts = build_action_summary_counts(groups)

    assert groups.profit_take_sells == ()
    assert [item.ticker.yf for item in groups.profit_take_reviews] == [
        "7203.T",
        "6758.T",
    ]
    assert "SELL" not in counts
    assert counts["REVIEW"] == 2


def test_inferred_listing_cannot_survive_presentation_as_executable_sell(
    sample_bundle,
):
    item = sample_bundle.items[0]
    item.ibkr_position = item.ibkr_position.model_copy(
        update={
            "ticker_identity_verified": False,
            "ticker_resolution_source": "yfinance_search",
        }
    )

    groups = group_portfolio_actions([item])

    assert groups.hard_sells == ()
    assert len(groups.reviews) == 1
    assert groups.reviews[0].action == "REVIEW"
    assert groups.reviews[0].suggested_quantity is None
    assert "sale downgraded" in groups.reviews[0].reason
    assert "listing mapping is unverified" in groups.reviews[0].reason


@pytest.mark.parametrize(
    ("action", "sell_type", "expected_basis"),
    [
        ("SELL", "STOP_BREACH", "STOP_LOSS"),
        ("SELL", "HARD_REJECT", "THESIS_REASSESSMENT"),
        ("TRIM", None, "OVERWEIGHT"),
    ],
)
def test_legacy_position_reductions_are_non_executable(
    action, sell_type, expected_basis
):
    item = ReconciliationItem(
        ticker=Ticker.from_yf("7203.T"),
        action=action,
        reason="Legacy action",
        urgency="HIGH",
        sell_type=sell_type,
        suggested_quantity=100,
        suggested_price=1900,
        cash_impact_usd=1200,
        settlement_date="2026-07-20",
    )

    groups = group_portfolio_actions([item])
    surfaced = (
        groups.macro_stop_reviews or groups.profit_take_reviews or groups.reviews
    )[0]

    assert surfaced.action == "REVIEW"
    assert surfaced.action_basis == expected_basis
    assert surfaced.suggested_quantity is None
    assert surfaced.cash_impact_usd == 0
    assert groups.stop_sells == ()
    assert groups.hard_sells == ()
    assert groups.trims == ()


def test_screen_reject_review_routes_to_generic_reviews():
    item = ReconciliationItem(
        ticker=Ticker.from_yf("4396.T"),
        action="REVIEW",
        reason="Screen-threshold DNI",
        urgency="MEDIUM",
        sell_type="SCREEN_REJECT",
    )

    groups = group_portfolio_actions([item])
    counts = build_action_summary_counts(groups)

    assert groups.hard_sells == ()
    assert groups.soft_sells == ()
    assert groups.stop_sells == ()
    assert groups.reviews == (item,)
    assert counts["REVIEW"] == 1


def test_aggregate_sector_weights_normalizes_equivalent_labels():
    weights = aggregate_sector_weights({"Healthcare": 12.5, "Health Care": 7.5})
    assert weights == {"Health Care": 20.0}


def test_get_sell_type_label_uses_shared_backend_labels():
    assert get_sell_type_label("STOP_BREACH") == "PRICE-DROP REVIEW"
    assert get_sell_type_label("HARD_REJECT") == "FUNDAMENTAL FAILURE"
    assert get_sell_type_label("SOFT_REJECT") == "SOFT REJECTION"
    assert get_sell_type_label("SCREEN_REJECT") == "SCREEN REVIEW"
    assert get_sell_type_label("DATA_QUALITY_REVIEW") == "DATA REVIEW"
    assert get_sell_type_label("SPECIAL_SITUATION_EXIT") == "M&A EXIT"
    assert get_sell_type_label("PROFIT_TAKE") == "PROFIT TAKE"
    assert get_sell_type_label("UNKNOWN") == "SELL"


def test_build_action_display_sections_matches_cli_contract(sample_bundle):
    groups = group_portfolio_actions(
        sample_bundle.items,
        watchlist_tickers=sample_bundle.watchlist_tickers,
    )

    sections = build_action_display_sections(groups)

    assert [section.key for section in sections] == [
        "sell_recommendations",
        "sell_related_reviews",
        "dip_watch",
        "hold",
    ]
    assert sections[0].title == SELL_RECOMMENDATIONS_TITLE
    assert sections[1].title == SELL_RELATED_REVIEWS_TITLE
    assert [item.ticker.yf for item in sections[0].items] == ["7203.T"]
    assert [item.ticker.yf for item in sections[1].items] == ["5285.T"]


class TestMacroDemotedItemBuckets:
    """A macro-demoted item must leave the executable SELL plan but keep its
    potential proceeds visible in the conditional ("soft-sell reviews") bucket.
    """

    @staticmethod
    def _demoted_item():
        from tests.ibkr.reconciler_cases import _make_sell_item_on_date

        item = _make_sell_item_on_date("DEMO.T", "2026-03-05", conid=777)
        item.action = "REVIEW"  # as _apply_macro_demotions leaves it
        item.reason += "  [MACRO_WATCH: demoted from SELL — correlated event detected]"
        item.cash_impact_usd = 1234.0
        return item

    def test_demoted_item_moves_to_macro_review_bucket(self):
        from src.ibkr.portfolio_presentation import group_portfolio_actions

        item = self._demoted_item()
        groups = group_portfolio_actions([item])
        assert item in groups.macro_reviews
        assert item not in groups.soft_sells

    def test_demoted_item_proceeds_stay_in_conditional_bucket(self):
        from src.ibkr.models import PortfolioSummary
        from src.ibkr.portfolio_presentation import build_cash_summary

        item = self._demoted_item()
        summary = build_cash_summary(
            [item], PortfolioSummary(portfolio_value_usd=10_000)
        )
        assert summary.conditional_proceeds_usd == 1234.0
        # And NOT in confirmed pending inflows
        assert summary.pending_inflows_total_usd == 0.0


class TestFxReturnSplit:
    """Local-price vs FX decomposition — observed values only, never fabricated."""

    @staticmethod
    def _pos(**overrides):
        from src.ibkr.models import NormalizedPosition
        from src.ibkr.ticker import Ticker

        defaults = {
            "conid": 1,
            "ticker": Ticker.from_yf("HERDEZ.MX", currency="MXN"),
            "quantity": 100,
            "avg_cost_local": 100.0,
            "current_price_local": 105.5,  # local-price +5.5%
            "currency": "MXN",
            "market_value_usd": 800.0,
            "unrealized_pnl_usd": -205.0,  # USD cost basis 1005 → -20.4%
        }
        defaults.update(overrides)
        return NormalizedPosition(**defaults)

    def test_herdez_style_local_gain_usd_loss(self):
        from src.ibkr.portfolio_presentation import fx_return_split

        split = fx_return_split(self._pos())
        assert split is not None
        local_pct, fx_pct, usd_pct = split
        assert local_pct == pytest.approx(5.5, abs=0.01)
        assert usd_pct == pytest.approx(-20.4, abs=0.1)
        # Multiplicative reconciliation: (1+usd) == (1+local) × (1+fx)
        assert (1 + usd_pct / 100) == pytest.approx(
            (1 + local_pct / 100) * (1 + fx_pct / 100)
        )
        assert fx_pct < 0  # FX drag

    def test_usd_position_has_no_split(self):
        from src.ibkr.models import NormalizedPosition
        from src.ibkr.portfolio_presentation import fx_return_split
        from src.ibkr.ticker import Ticker

        pos = NormalizedPosition(
            conid=2,
            ticker=Ticker.from_yf("AAPL"),
            quantity=10,
            avg_cost_local=100.0,
            current_price_local=110.0,
            currency="USD",
            market_value_usd=1100.0,
            unrealized_pnl_usd=100.0,
        )
        assert fx_return_split(pos) is None

    def test_zero_pnl_is_a_valid_observation(self):
        from src.ibkr.portfolio_presentation import fx_return_split

        split = fx_return_split(
            self._pos(market_value_usd=1005.0, unrealized_pnl_usd=0.0)
        )
        assert split is not None
        assert split[2] == pytest.approx(0.0)

    def test_real_large_return_is_not_misclassified_as_unit_mismatch(self):
        from src.ibkr.portfolio_presentation import fx_return_split

        split = fx_return_split(
            self._pos(
                current_price_local=200.0,
                market_value_usd=1600.0,
                unrealized_pnl_usd=595.0,
            )
        )
        assert split is not None
        assert split[0] == pytest.approx(100.0)

    def test_degenerate_or_extreme_ratio_inputs_yield_none(self):
        from src.ibkr.portfolio_presentation import fx_return_split

        assert fx_return_split(None) is None
        assert fx_return_split(self._pos(avg_cost_local=0.0)) is None
        # A 100x local-price ratio is a likely pence/pounds or feed mismatch.
        assert fx_return_split(self._pos(current_price_local=10000.0)) is None
