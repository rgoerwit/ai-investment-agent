from __future__ import annotations

from datetime import date, timedelta
from types import SimpleNamespace

import pandas as pd
import pytest

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


# --- FY EPS growth derived from filed statements (not the stale info scalar) ---


def test_earnings_growth_from_diluted_eps() -> None:
    financials = pd.DataFrame(
        {"2025": [200.0, 70.7], "2024": [180.0, 59.01]},
        index=["Total Revenue", "Diluted EPS"],
    )
    result = extract_from_financial_statements(
        _fetcher(), _Ticker(financials=financials), "TEST"
    )
    assert result["earningsGrowth"] == pytest.approx((70.7 - 59.01) / 59.01)
    assert result["_earningsGrowth_source"] == "calculated_from_statement_diluted_eps"


def test_earnings_growth_falls_back_to_basic_eps() -> None:
    financials = pd.DataFrame(
        {"2025": [200.0, 5.2], "2024": [180.0, 5.0]},
        index=["Total Revenue", "Basic EPS"],
    )
    result = extract_from_financial_statements(
        _fetcher(), _Ticker(financials=financials), "TEST"
    )
    assert result["earningsGrowth"] == pytest.approx(0.04)
    assert result["_earningsGrowth_source"] == "calculated_from_statement_basic_eps"


def test_earnings_growth_ni_proxy_when_no_eps_rows() -> None:
    financials = pd.DataFrame(
        {"2025": [200.0, 1100.0], "2024": [180.0, 1000.0]},
        index=["Total Revenue", "Net Income"],
    )
    result = extract_from_financial_statements(
        _fetcher(), _Ticker(financials=financials), "TEST"
    )
    assert result["earningsGrowth"] == pytest.approx(0.1)
    assert (
        result["_earningsGrowth_source"] == "calculated_from_statement_net_income_proxy"
    )


def test_earnings_growth_prefers_diluted_eps_over_net_income_on_buyback() -> None:
    # Net income flat (0%), but diluted EPS +5% because the share count fell
    # (buyback). EPS growth must win — this is the NI != EPS caveat.
    financials = pd.DataFrame(
        {"2025": [200.0, 1000.0, 52.5], "2024": [180.0, 1000.0, 50.0]},
        index=["Total Revenue", "Net Income", "Diluted EPS"],
    )
    result = extract_from_financial_statements(
        _fetcher(), _Ticker(financials=financials), "TEST"
    )
    assert result["earningsGrowth"] == pytest.approx(0.05)
    assert result["_earningsGrowth_source"] == "calculated_from_statement_diluted_eps"


def test_earnings_growth_skipped_when_prior_eps_nonpositive() -> None:
    # A loss-year base makes a YoY % meaningless; leave N/A rather than fabricate.
    financials = pd.DataFrame(
        {"2025": [200.0, 5.0], "2024": [180.0, -2.0]},
        index=["Total Revenue", "Diluted EPS"],
    )
    result = extract_from_financial_statements(
        _fetcher(), _Ticker(financials=financials), "TEST"
    )
    assert "earningsGrowth" not in result


# --- Stale-annual-statements flag (yfinance can lag a full FY for ex-US names) ---


def test_statements_stale_flag_for_old_annual() -> None:
    latest = pd.Timestamp(date.today() - timedelta(days=540))
    prior = pd.Timestamp(date.today() - timedelta(days=540 + 365))
    financials = pd.DataFrame(
        {latest: [200.0, 70.0], prior: [180.0, 60.0]},
        index=["Total Revenue", "Diluted EPS"],
    )
    result = extract_from_financial_statements(
        _fetcher(), _Ticker(financials=financials), "TEST"
    )
    assert result["statements_stale"] is True
    assert result["_income_statement_date"] == latest.date().isoformat()


def test_statements_not_stale_for_recent_annual() -> None:
    latest = pd.Timestamp(date.today() - timedelta(days=180))
    prior = pd.Timestamp(date.today() - timedelta(days=180 + 365))
    financials = pd.DataFrame(
        {latest: [200.0, 70.0], prior: [180.0, 60.0]},
        index=["Total Revenue", "Diluted EPS"],
    )
    result = extract_from_financial_statements(
        _fetcher(), _Ticker(financials=financials), "TEST"
    )
    assert "statements_stale" not in result
    assert result["_statements_age_days"] < 457
