"""Statement-derived FY growth must override stale Yahoo `info` scalars, and the
provenance tag must ride only with the accepted value.

Regression coverage for the bug where `extract_from_financial_statements` computed
the correct FY growth (e.g. 4396.T revenue +22.3%) but `fetch_yfinance_enhanced`
kept the stale `info` scalar (+5.3%) while still copying the `calculated_from_statements`
source tag — inflating the stale value's merge quality to 10 so it beat even paid feeds.
"""

from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace

import pytest

import src.data.source_fetchers as sf
from src.data.merge_policy import (
    quarantine_forward_pe_outlier,
    smart_merge_with_quality,
)


class _FakeTicker:
    def __init__(self, info: dict) -> None:
        self._info = info

    @property
    def info(self) -> dict:
        return self._info


def _fake_fetcher(statement_data: dict) -> SimpleNamespace:
    return SimpleNamespace(
        stats={"sources": defaultdict(int)},
        _extract_from_financial_statements=lambda ticker, symbol: dict(statement_data),
    )


@pytest.mark.asyncio
async def test_statement_growth_overrides_stale_scalar(monkeypatch) -> None:
    info = {
        "currentPrice": 957.0,
        "currency": "JPY",
        "revenueGrowth": 0.053,
        "earningsGrowth": 0.105,
    }
    statement_data = {
        "revenueGrowth": 0.223,
        "_revenueGrowth_source": "calculated_from_statements",
        "earningsGrowth": 0.198,
        "_earningsGrowth_source": "calculated_from_statement_diluted_eps",
        "_income_statement_date": "2025-06-30",
    }
    monkeypatch.setattr(sf.yf, "Ticker", lambda symbol: _FakeTicker(info))

    out = await sf.fetch_yfinance_enhanced(_fake_fetcher(statement_data), "4396.T")

    assert out["revenueGrowth"] == 0.223
    assert out["_revenueGrowth_source"] == "calculated_from_statements"
    assert out["earningsGrowth"] == 0.198
    assert out["_earningsGrowth_source"] == "calculated_from_statement_diluted_eps"
    assert out["_statement_overrides"]["revenueGrowth"] == {
        "scalar": 0.053,
        "statement": 0.223,
    }
    assert out["_income_statement_date"] == "2025-06-30"


@pytest.mark.asyncio
async def test_stale_statements_keep_info_scalar(monkeypatch) -> None:
    # When the statements themselves are stale (a missing latest FY), the info
    # scalar may be the fresher figure — keep it, and surface statements_stale.
    info = {
        "currentPrice": 448.0,
        "currency": "MXN",
        "revenueGrowth": 0.071,
        "earningsGrowth": -0.077,
    }
    statement_data = {
        "revenueGrowth": 0.099,
        "_revenueGrowth_source": "calculated_from_statements",
        "earningsGrowth": 0.089,
        "_earningsGrowth_source": "calculated_from_statement_net_income_proxy",
        "statements_stale": True,
        "_income_statement_date": "2024-12-31",
    }
    monkeypatch.setattr(sf.yf, "Ticker", lambda symbol: _FakeTicker(info))

    out = await sf.fetch_yfinance_enhanced(_fake_fetcher(statement_data), "FRAGUAB.MX")

    assert out["revenueGrowth"] == 0.071
    assert out["earningsGrowth"] == -0.077
    assert out.get("_revenueGrowth_source") is None
    assert out.get("_earningsGrowth_source") is None
    assert out.get("_statement_overrides") is None
    assert out["statements_stale"] is True
    assert out["_income_statement_date"] == "2024-12-31"


@pytest.mark.asyncio
async def test_no_false_source_tag_when_statement_value_absent(monkeypatch) -> None:
    # A statement field with no value must not leave its source tag on the
    # surviving info scalar (this is what inflated merge quality before).
    info = {"currentPrice": 100.0, "currency": "USD", "grossMargins": 0.28}
    statement_data = {
        "grossMargins": None,
        "_grossMargins_source": "calculated_from_statements",
    }
    monkeypatch.setattr(sf.yf, "Ticker", lambda symbol: _FakeTicker(info))

    out = await sf.fetch_yfinance_enhanced(_fake_fetcher(statement_data), "TEST")

    assert out["grossMargins"] == 0.28
    assert out.get("_grossMargins_source") is None


def _merge(sources: dict) -> tuple:
    merged, meta = smart_merge_with_quality(
        sources, "TEST", quarantine_forward_pe_outlier
    )
    return merged.get("revenueGrowth"), meta.get("field_sources", {}).get(
        "revenueGrowth"
    )


def test_merge_prefers_statement_growth_over_paid_feeds() -> None:
    value, source = _merge(
        {
            "yfinance": {
                "revenueGrowth": 0.223,
                "_revenueGrowth_source": "calculated_from_statements",
            },
            "eodhd": {"revenueGrowth": 0.058},
            "fmp": {"revenueGrowth": 0.05},
        }
    )
    assert value == 0.223
    assert source == "yfinance"


def test_eodhd_fills_growth_when_yfinance_untagged() -> None:
    # Stale yfinance info scalar (no statement tag → quality 9) loses to EODHD (9.5).
    value, source = _merge(
        {
            "yfinance": {"revenueGrowth": 0.071},
            "eodhd": {"revenueGrowth": 0.058},
        }
    )
    assert value == 0.058
    assert source == "eodhd"


# --- Live canary: yfinance annual statements must still yield FY growth ---------
# Mirrors tests/scripts/test_find_gems.py::TestExchangeMetricCanary. Catches the
# bug class this work addresses: yfinance row-label drift (e.g. "Diluted EPS"
# renamed) silently dropping FY earnings growth back to the stale info scalar.

_GROWTH_CANARIES = {".T": "4396.T", ".HK": "3306.HK"}


@pytest.mark.parametrize(
    ("suffix", "ticker"),
    list(_GROWTH_CANARIES.items()),
    ids=list(_GROWTH_CANARIES),
)
def test_statement_growth_canary_non_null(suffix, ticker) -> None:
    yf = pytest.importorskip("yfinance")
    import requests

    from src.data.fetcher import SmartMarketDataFetcher

    fetcher = SmartMarketDataFetcher()
    try:
        data = fetcher._extract_from_financial_statements(yf.Ticker(ticker), ticker)
    except requests.exceptions.RequestException as exc:
        pytest.skip(f"{ticker}: yfinance unreachable: {exc}")

    if not data or data.get("revenueGrowth") is None:
        pytest.skip(f"{ticker}: no statement data (yfinance may be down/rate-limited)")

    assert data.get("earningsGrowth") is not None, (
        f"{ticker}: FY earnings growth missing while revenue growth present — "
        "yfinance EPS rows may have drifted (Diluted/Basic EPS labels changed)"
    )
    assert str(data.get("_earningsGrowth_source", "")).startswith(
        "calculated_from_statement"
    )
