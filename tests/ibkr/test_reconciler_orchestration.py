"""Collected reconciler-orchestration tests extracted from reconciler cases."""

from src.ibkr.reconciler import _populate_portfolio_weights
from tests.factories.ibkr import make_analysis, make_portfolio, make_position
from tests.ibkr.reconciler_cases import (
    TestAlphaBaseFallback,
    TestAlphaBaseLookup,
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
