"""Regression tests for yfinance statement extraction thread offloading."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.data.fetcher import SmartMarketDataFetcher


@pytest.mark.asyncio
async def test_fetch_yfinance_enhanced_offloads_statement_extraction(mocker) -> None:
    fetcher = SmartMarketDataFetcher()
    fake_ticker = SimpleNamespace()
    mocker.patch("src.data.fetcher.yf.Ticker", return_value=fake_ticker)

    calls: list[tuple[str, tuple[object, ...]]] = []

    async def fake_to_thread(fn, *args, **kwargs):
        del kwargs
        calls.append((getattr(fn, "__name__", type(fn).__name__), args))
        if getattr(fn, "__name__", "") == "<lambda>":
            return {
                "currentPrice": 10.0,
                "currency": "USD",
                "financialCurrency": "USD",
            }
        if getattr(fn, "__name__", "") == "_extract_from_financial_statements":
            return {
                "freeCashflow": 1000.0,
                "revenueGrowth_TTM": 0.12,
            }
        raise AssertionError(f"Unexpected to_thread target: {fn!r}")

    mocker.patch("src.data.fetcher.asyncio.to_thread", side_effect=fake_to_thread)

    result = await fetcher._fetch_yfinance_enhanced("TEST")

    assert result is not None
    assert result["revenueGrowth_TTM"] == 0.12
    assert any(name == "_extract_from_financial_statements" for name, _ in calls)


@pytest.mark.asyncio
async def test_fetch_yfinance_enhanced_returns_none_when_statement_offload_raises(
    mocker,
) -> None:
    fetcher = SmartMarketDataFetcher()
    fake_ticker = SimpleNamespace()
    mocker.patch("src.data.fetcher.yf.Ticker", return_value=fake_ticker)

    async def fake_to_thread(fn, *args, **kwargs):
        del args, kwargs
        if getattr(fn, "__name__", "") == "<lambda>":
            return {
                "currentPrice": 10.0,
                "currency": "USD",
                "financialCurrency": "USD",
            }
        if getattr(fn, "__name__", "") == "_extract_from_financial_statements":
            raise RuntimeError("statement fetch failed")
        raise AssertionError(f"Unexpected to_thread target: {fn!r}")

    mocker.patch("src.data.fetcher.asyncio.to_thread", side_effect=fake_to_thread)

    result = await fetcher._fetch_yfinance_enhanced("TEST")

    assert result is None


@pytest.mark.asyncio
async def test_fetch_yfinance_enhanced_adds_cross_listing_note_for_currency_mismatch(
    mocker,
) -> None:
    fetcher = SmartMarketDataFetcher()
    fake_ticker = SimpleNamespace()
    mocker.patch("src.data.fetcher.yf.Ticker", return_value=fake_ticker)

    async def fake_to_thread(fn, *args, **kwargs):
        del kwargs
        if getattr(fn, "__name__", "") == "<lambda>":
            return {
                "currentPrice": 10.0,
                "currency": "USD",
                "financialCurrency": "HKD",
                "exchange": "NYSE",
            }
        if getattr(fn, "__name__", "") == "_extract_from_financial_statements":
            return {"freeCashflow": 1000.0}
        raise AssertionError(f"Unexpected to_thread target: {fn!r}")

    mocker.patch("src.data.fetcher.asyncio.to_thread", side_effect=fake_to_thread)

    result = await fetcher._fetch_yfinance_enhanced("TEST")

    assert result is not None
    assert "cross_listing_note" in result
    assert "USD" in result["cross_listing_note"]
    assert "HKD" in result["cross_listing_note"]


@pytest.mark.asyncio
async def test_fetch_yfinance_enhanced_skips_cross_listing_note_when_currencies_match(
    mocker,
) -> None:
    fetcher = SmartMarketDataFetcher()
    fake_ticker = SimpleNamespace()
    mocker.patch("src.data.fetcher.yf.Ticker", return_value=fake_ticker)

    async def fake_to_thread(fn, *args, **kwargs):
        del kwargs
        if getattr(fn, "__name__", "") == "<lambda>":
            return {
                "currentPrice": 10.0,
                "currency": "USD",
                "financialCurrency": "USD",
            }
        if getattr(fn, "__name__", "") == "_extract_from_financial_statements":
            return {"freeCashflow": 1000.0}
        raise AssertionError(f"Unexpected to_thread target: {fn!r}")

    mocker.patch("src.data.fetcher.asyncio.to_thread", side_effect=fake_to_thread)

    result = await fetcher._fetch_yfinance_enhanced("TEST")

    assert result is not None
    assert "cross_listing_note" not in result


@pytest.mark.asyncio
async def test_fetch_all_sources_parallel_marks_empty_sources_as_non_network_like(
    mocker,
) -> None:
    fetcher = SmartMarketDataFetcher()
    mocker.patch.object(
        fetcher, "_fetch_yfinance_enhanced", AsyncMock(return_value=None)
    )
    mocker.patch.object(fetcher, "_fetch_yahooquery_fallback", return_value=None)
    mocker.patch.object(fetcher, "_fetch_fmp_fallback", AsyncMock(return_value=None))
    mocker.patch.object(fetcher, "_fetch_eodhd_fallback", AsyncMock(return_value=None))
    mocker.patch.object(fetcher, "_fetch_av_fallback", AsyncMock(return_value=None))
    mock_logger = mocker.patch("src.data.fetcher.logger")

    async def fake_to_thread(fn, *args, **kwargs):
        del kwargs
        return fn(*args)

    mocker.patch("src.data.fetcher.asyncio.to_thread", side_effect=fake_to_thread)

    results = await fetcher._fetch_all_sources_parallel("TEST")

    assert all(value is None for value in results.values())
    aggregate_call = next(
        call
        for call in mock_logger.warning.call_args_list
        if call.args and call.args[0] == "all_data_sources_failed"
    )
    assert aggregate_call.kwargs["suspected_cause"] == "no_data_or_provider_unavailable"
    assert aggregate_call.kwargs["source_outcomes"] == {
        "yfinance": "empty",
        "yahooquery": "empty",
        "fmp": "empty",
        "eodhd": "empty",
        "alpha_vantage": "empty",
    }


@pytest.mark.asyncio
async def test_fetch_all_sources_parallel_marks_transient_failures_conservatively(
    mocker,
) -> None:
    fetcher = SmartMarketDataFetcher()

    async def raise_timeout(*args, **kwargs):
        del args, kwargs
        raise asyncio.TimeoutError

    async def raise_connection_error(*args, **kwargs):
        del args, kwargs
        raise ConnectionError("proxy disconnected")

    mocker.patch.object(fetcher, "_fetch_yfinance_enhanced", side_effect=raise_timeout)
    mocker.patch.object(
        fetcher,
        "_fetch_yahooquery_fallback",
        side_effect=ConnectionError("proxy disconnected"),
    )
    mocker.patch.object(fetcher, "_fetch_fmp_fallback", side_effect=raise_timeout)
    mocker.patch.object(
        fetcher, "_fetch_eodhd_fallback", side_effect=raise_connection_error
    )
    mocker.patch.object(fetcher, "_fetch_av_fallback", side_effect=raise_timeout)
    mock_logger = mocker.patch("src.data.fetcher.logger")

    async def fake_to_thread(fn, *args, **kwargs):
        del kwargs
        return fn(*args)

    mocker.patch("src.data.fetcher.asyncio.to_thread", side_effect=fake_to_thread)

    results = await fetcher._fetch_all_sources_parallel("TEST")

    assert all(value is None for value in results.values())
    aggregate_call = next(
        call
        for call in mock_logger.warning.call_args_list
        if call.args and call.args[0] == "all_data_sources_failed"
    )
    assert (
        aggregate_call.kwargs["suspected_cause"] == "connectivity_or_provider_transient"
    )
    assert aggregate_call.kwargs["source_outcomes"] == {
        "yfinance": "timeout",
        "yahooquery": "error:ConnectionError",
        "fmp": "timeout",
        "eodhd": "error:ConnectionError",
        "alpha_vantage": "timeout",
    }


@pytest.mark.asyncio
async def test_fetch_all_sources_parallel_skips_aggregate_warning_when_one_source_succeeds(
    mocker,
) -> None:
    fetcher = SmartMarketDataFetcher()
    mocker.patch.object(
        fetcher, "_fetch_yfinance_enhanced", AsyncMock(return_value={"symbol": "TEST"})
    )
    mocker.patch.object(fetcher, "_fetch_yahooquery_fallback", return_value=None)
    mocker.patch.object(fetcher, "_fetch_fmp_fallback", AsyncMock(return_value=None))
    mocker.patch.object(fetcher, "_fetch_eodhd_fallback", AsyncMock(return_value=None))
    mocker.patch.object(fetcher, "_fetch_av_fallback", AsyncMock(return_value=None))
    mock_logger = mocker.patch("src.data.fetcher.logger")

    async def fake_to_thread(fn, *args, **kwargs):
        del kwargs
        return fn(*args)

    mocker.patch("src.data.fetcher.asyncio.to_thread", side_effect=fake_to_thread)

    results = await fetcher._fetch_all_sources_parallel("TEST")

    assert results["yfinance"] == {"symbol": "TEST"}
    assert not any(
        call.args and call.args[0] == "all_data_sources_failed"
        for call in mock_logger.warning.call_args_list
    )


@pytest.mark.asyncio
async def test_get_financial_metrics_uncached_logs_total_timeout_and_returns_empty(
    mocker,
) -> None:
    fetcher = SmartMarketDataFetcher()
    mocker.patch.object(
        fetcher, "_fetch_metrics_inner", AsyncMock(side_effect=asyncio.TimeoutError)
    )
    mock_logger = mocker.patch("src.data.fetcher.logger")

    result = await fetcher._get_financial_metrics_uncached("TEST", timeout=1)

    assert result == {}
    mock_logger.warning.assert_any_call(
        "get_financial_metrics_total_timeout", ticker="TEST", timeout=1
    )
