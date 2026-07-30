from __future__ import annotations

import pandas as pd
import pytest

from src.data.metric_extraction import (
    calculate_derived_metrics,
    extract_from_financial_statements,
)


class _Fetcher:
    def __init__(self) -> None:
        self.stats = {"sources": {"statements": 0}}


class _Ticker:
    def __init__(self) -> None:
        cols = pd.to_datetime(["2025-12-31", "2024-12-31", "2023-12-31", "2022-12-31"])
        self.financials = pd.DataFrame(
            [[1331.0, 1210.0, 1100.0, 1000.0], [200.0, 180.0, 160.0, 140.0]],
            index=["Total Revenue", "Net Income"],
            columns=cols,
        )
        self.cashflow = pd.DataFrame(
            [[220.0, 200.0, 180.0, 160.0], [-70.0, -65.0, -60.0, -55.0]],
            index=["Operating Cash Flow", "Capital Expenditure"],
            columns=cols,
        )
        self.balance_sheet = pd.DataFrame(
            [[1000.0, 2000.0, 2100.0, 2200.0], [500.0, 520.0, 540.0, 560.0]],
            index=["Total Assets", "Stockholders Equity"],
            columns=cols,
        )
        self.quarterly_financials = pd.DataFrame()
        self.quarterly_cashflow = pd.DataFrame()


class _TickerWithMisalignedCashFlow:
    def __init__(self) -> None:
        cols = pd.to_datetime(["2025-12-31", "2024-12-31", "2023-12-31", "2022-12-31"])
        self.financials = pd.DataFrame(
            [[1331.0, 1210.0, 1100.0, 1000.0], [200.0, 180.0, 160.0, 140.0]],
            index=["Total Revenue", "Net Income"],
            columns=cols,
        )
        self.cashflow = pd.DataFrame(
            [[220.0, None, 180.0, 160.0], [-70.0, -65.0, None, -55.0]],
            index=["Operating Cash Flow", "Capital Expenditure"],
            columns=cols,
        )
        self.balance_sheet = pd.DataFrame(
            [[1000.0, 2000.0, 2100.0, 2200.0], [500.0, 520.0, 540.0, 560.0]],
            index=["Total Assets", "Stockholders Equity"],
            columns=cols,
        )
        self.quarterly_financials = pd.DataFrame()
        self.quarterly_cashflow = pd.DataFrame()


def test_statement_extraction_adds_3y_cagr_and_cycle_position() -> None:
    result = extract_from_financial_statements(_Fetcher(), _Ticker(), "TEST")

    assert result["revenue_cagr_3y"] == pytest.approx(0.10, abs=0.001)
    assert result["fcf_cagr_3y"] == pytest.approx(0.126, abs=0.001)
    assert result["cycle_position"] == "PEAK"


def test_fcf_cagr_requires_same_period_ocf_and_capex() -> None:
    result = extract_from_financial_statements(
        _Fetcher(),
        _TickerWithMisalignedCashFlow(),
        "TEST",
    )

    assert result["freeCashflow"] == pytest.approx(150.0)
    assert result["revenue_cagr_3y"] == pytest.approx(0.10, abs=0.001)
    assert "fcf_cagr_3y" not in result


def test_calculate_derived_metrics_adds_pe_vs_sector() -> None:
    result = calculate_derived_metrics(
        {"sector": "Financials", "trailingPE": 14.0},
        "TEST",
    )

    assert result["sectorMedianPE"] == pytest.approx(12.0)
    assert result["peVsSector"] == pytest.approx(1.17)
    assert result["sectorPeReferenceType"] == "STATIC_POLICY_REFERENCE"
    assert result["sectorPeReferenceAsOf"] == "N/A"
    assert result["_sectorMedianPE_source"] == "static_gics_sector_median"
    assert result["_peVsSector_source"] == "static_gics_sector_median"


@pytest.mark.parametrize(
    "sector",
    ["Financial Services", " financial   services ", "Financials"],
)
def test_calculate_derived_metrics_normalizes_vendor_sector_aliases(
    sector: str,
) -> None:
    result = calculate_derived_metrics(
        {"sector": sector, "trailingPE": 16.554346},
        "B3SA3.SA",
    )

    assert result["sectorMedianPE"] == pytest.approx(12.0)
    assert result["peVsSector"] == pytest.approx(1.38)


@pytest.mark.parametrize("sector", [None, "", "Unknown", "Aerospace", 123])
def test_calculate_derived_metrics_keeps_unknown_sector_na(sector: object) -> None:
    result = calculate_derived_metrics(
        {"sector": sector, "trailingPE": 16.554346},
        "TEST",
    )

    assert "sectorMedianPE" not in result
    assert "peVsSector" not in result
