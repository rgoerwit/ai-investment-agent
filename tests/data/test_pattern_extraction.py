"""Monetary web extraction uses the same unit contract as report extraction."""

import pytest

from src.data.pattern_extraction import FinancialPatternExtractor


@pytest.mark.parametrize(
    "amount,expected",
    [
        ("100 BRL", 100),
        ("-100M USD", None),
        ("100K KRW", 100000),
        ("1.2 billion USD", 1.2e9),
        ("1,234.5M", 1.2345e9),
        ("1.2 Mrd.", None),
        ("€1,234 Mio.", None),
        ("EUR 2,500 Mrd", None),
        ("Operating cash flow: 4,800 Mio. EUR", None),
        ("1,200 crore", None),
        ("3,000 employees", None),
        ("1,5B", None),
    ],
)
def test_market_cap_shared_monetary_contract(amount, expected):
    result = FinancialPatternExtractor().extract_from_text(f"Market Cap: {amount}")
    assert result.get("marketCap") == expected
    if expected is not None:
        assert result["_marketCap_source"] == "web_search_extraction"
    assert (
        FinancialPatternExtractor().extract_from_text(
            f"Market Cap: {amount}", {"marketCap"}
        )
        == {}
    )


def test_nonmonetary_locale_ratio_contract_is_preserved():
    result = FinancialPatternExtractor().extract_from_text("P/E Ratio (TTM): 12,5")
    assert result["trailingPE"] == 12.5
