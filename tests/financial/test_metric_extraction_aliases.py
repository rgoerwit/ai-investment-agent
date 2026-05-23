from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from src.data.metric_extraction import (
    extract_from_financial_statements,
    statement_value,
)


class _Ticker:
    def __init__(
        self,
        *,
        financials: pd.DataFrame | None = None,
        cashflow: pd.DataFrame | None = None,
        balance_sheet: pd.DataFrame | None = None,
    ) -> None:
        self.financials = financials if financials is not None else pd.DataFrame()
        self.cashflow = cashflow if cashflow is not None else pd.DataFrame()
        self.balance_sheet = (
            balance_sheet if balance_sheet is not None else pd.DataFrame()
        )
        self.quarterly_financials = pd.DataFrame()
        self.quarterly_cashflow = pd.DataFrame()


class _BrokenTicker:
    @property
    def financials(self):
        raise RuntimeError("statement endpoint failed")


def _fetcher():
    return SimpleNamespace(stats={"sources": {"statements": 0}})


def test_statement_value_reads_canonical_and_alias_rows() -> None:
    df = pd.DataFrame(
        {"2024": [100.0, 42.0]},
        index=["Operating Revenue", "Operating Income Loss"],
    )

    assert statement_value(df, "total_revenue") == 100.0
    assert statement_value(df, "operating_income") == 42.0


def test_statement_aliases_feed_extracted_metrics() -> None:
    financials = pd.DataFrame(
        {
            "2024": [200.0, 80.0, 40.0, 30.0],
            "2023": [100.0, 35.0, 20.0, 10.0],
        },
        index=[
            "Operating Revenue",
            "Gross Profit",
            "Operating Income Loss",
            "Net Income Common Stockholders",
        ],
    )
    cashflow = pd.DataFrame(
        {"2024": [50.0, -10.0]},
        index=[
            "Cash Flow From Continuing Operating Activities",
            "Capital Expenditures",
        ],
    )
    balance_sheet = pd.DataFrame(
        {"2024": [100.0, 50.0, 20.0, 5.0, 200.0, 40.0, 10.0]},
        index=[
            "Total Current Assets",
            "Total Current Liabilities",
            "Long Term Debt And Capital Lease Obligation",
            "Current Debt And Capital Lease Obligation",
            "Total Stockholder Equity",
            "Cash",
            "Short Term Investments",
        ],
    )

    result = extract_from_financial_statements(
        _fetcher(),
        _Ticker(
            financials=financials,
            cashflow=cashflow,
            balance_sheet=balance_sheet,
        ),
        "TEST",
    )

    assert result["revenueGrowth"] == 1.0
    assert result["grossMargins"] == 0.4
    assert result["operatingMargins"] == 0.2
    assert result["profitMargins"] == 0.15
    assert result["operatingCashflow"] == 50.0
    assert result["freeCashflow"] == 40.0
    assert result["currentRatio"] == 2.0
    assert result["debtToEquity"] == 0.125
    assert result["cashAndShortTermInvestments"] == 50.0


def test_bundled_cash_and_short_term_investments_is_not_double_counted() -> None:
    balance_sheet = pd.DataFrame(
        {"2024": [100.0, 25.0, 10.0]},
        index=[
            "Total Assets",
            "Cash And Short Term Investments",
            "Short Term Investments",
        ],
    )

    result = extract_from_financial_statements(
        _fetcher(),
        _Ticker(balance_sheet=balance_sheet),
        "TEST",
    )

    assert result["cashAndShortTermInvestments"] == 25.0


def test_statement_access_failure_degrades_to_empty_metrics() -> None:
    result = extract_from_financial_statements(_fetcher(), _BrokenTicker(), "TEST")

    assert result["graham_test"] == "INSUFFICIENT_DATA"
    assert "revenueGrowth" not in result
    assert "currentRatio" not in result
