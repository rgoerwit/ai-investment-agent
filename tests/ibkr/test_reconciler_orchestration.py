"""Collected reconciler-orchestration tests extracted from reconciler cases."""

import pytest

from src.ibkr.models import UnresolvedBrokerPosition
from src.ibkr.portfolio_health import compute_portfolio_health
from src.ibkr.reconciler import _populate_portfolio_weights
from tests.factories.ibkr import make_analysis, make_portfolio, make_position
from tests.ibkr.reconciler_cases import (
    TestAlphaBaseFallback,
    TestAlphaBaseLookup,
    TestAmbiguousBaseGuards,
    TestIbkrSymbol,
)


def test_populate_portfolio_weights_canonicalizes_sector_labels():
    positions = [
        make_position(ticker="7203.T", market_value_usd=600.0),
        make_position(ticker="6758.T", market_value_usd=400.0, conid=654321),
    ]
    analyses = {
        "7203.T": make_analysis(ticker="7203.T"),
        "6758.T": make_analysis(ticker="6758.T"),
    }
    analyses["7203.T"].sector = "Technology"
    analyses["6758.T"].sector = "Information Technology"
    portfolio = make_portfolio(value=1000.0, cash=100.0)

    sector_weights, _exchange_weights = _populate_portfolio_weights(
        positions,
        analyses,
        portfolio,
        alpha_base_lookup={},
    )

    assert sector_weights == {"Information Technology": 100.0}
    assert portfolio.sector_weights == {"Information Technology": 100.0}


def test_populate_portfolio_weights_keeps_identity_coverage_out_of_buckets():
    positions = [make_position(ticker="7203.T", market_value_usd=750.0)]
    analyses = {"7203.T": make_analysis(ticker="7203.T")}
    portfolio = make_portfolio(value=1000.0, cash=0.0)
    portfolio.unresolved_positions = [
        UnresolvedBrokerPosition(
            conid=17_382_285,
            broker_token="IBCID17382285",
            quantity=3,
            currency="KRW",
            market_value_usd=250,
            reason="canonical security identity unavailable",
        )
    ]

    sector_weights, exchange_weights = _populate_portfolio_weights(
        positions, analyses, portfolio, alpha_base_lookup={}
    )

    assert "Unresolved identity" not in sector_weights
    assert "Unresolved identity" not in exchange_weights
    assert sum(sector_weights.values()) == pytest.approx(100.0)
    assert portfolio.currency_weights == {"JPY": pytest.approx(100.0)}


def test_all_unresolved_portfolio_retains_known_currency_risk() -> None:
    portfolio = make_portfolio(value=1000.0, cash=0.0)
    portfolio.unresolved_positions = [
        UnresolvedBrokerPosition(
            conid=17_382_285,
            broker_token="IBCID17382285",
            quantity=3,
            currency="KRW",
            market_value_usd=600,
            reason="canonical security identity unavailable",
        )
    ]

    flags = compute_portfolio_health([], {}, portfolio)

    assert any("CURRENCY_CONCENTRATION: 60.0% in KRW" in flag for flag in flags)
    assert not any("GEOGRAPHY_CONCENTRATION" in flag for flag in flags)
