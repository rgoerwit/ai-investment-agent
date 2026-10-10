"""Market-data deadlines: one ``Settings`` field per scope, clamped to the caller.

Each scope is checked four ways: the shipped default (read from the real field
definition, never by instantiating ``Settings``), an override reaching the
actual client object or blocking call, a caller's enclosing deadline winning
over a looser default, and a spent or fired deadline taking the call site's
existing fallback. Fake clocks and tiny timeouts only; nothing touches the
network.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest

import src.blocking_io as blocking_io
import src.consultant_tools as consultant_tools
import src.data.fetcher as fetcher_module
import src.data.gap_fill as gap_fill
import src.fx_normalization as fx_normalization
import src.ticker_utils as ticker_utils
from src.blocking_io import (
    FX_RATE_POLICY,
    GAP_FILL_YFINANCE_INFO_POLICY,
    YAHOOQUERY_QUOTE_TYPE_POLICY,
    YFINANCE_INFO_POLICY,
)
from src.config import Settings, config
from src.data import alpha_vantage_fetcher, eodhd_fetcher, fmp_fetcher
from src.data.fetcher import SmartMarketDataFetcher
from src.data.source_fetchers import fetch_all_sources_parallel
from src.llm_runtime import timeouts as llm_timeouts
from src.llm_runtime.timeouts import (
    clamp_to_operational_deadline,
    enclosing_deadline,
    require_allowance,
)

_NOW = 1_000.0

SHIPPED_DEFAULTS = {
    "market_data_aggregate_fetch_timeout_seconds": 30.0,
    "market_data_per_source_timeout_seconds": 15.0,
    "market_data_vendor_http_timeout_seconds": 10.0,
    "market_data_anchor_lookup_timeout_seconds": 5.0,
    "market_data_ticker_resolution_timeout_seconds": 10.0,
    "market_data_price_history_timeout_seconds": 15.0,
    "company_name_lookup_timeout_seconds": 5.0,
    "fx_rate_fetch_timeout_seconds": 8.0,
    "consultant_spot_check_timeout_seconds": 8.0,
}


@pytest.fixture
def fake_clock(monkeypatch) -> list[float]:
    """A frozen monotonic clock for the deadline helpers; advance via clock[0]."""
    clock = [_NOW]
    monkeypatch.setattr(llm_timeouts, "now", lambda: clock[0])
    return clock


def _spy_hard_timeout(monkeypatch, module, result: Any = None) -> list[float]:
    """Replace ``module.run_with_hard_timeout``; record each timeout, never run."""
    seen: list[float] = []

    async def spy(coro, *, timeout, label):
        del label
        seen.append(timeout)
        coro.close()
        return result

    monkeypatch.setattr(module, "run_with_hard_timeout", spy)
    return seen


def _spy_blocking_call(monkeypatch, module, result: Any = None) -> list[Any]:
    """Replace ``module.run_blocking_call``; record each policy, never run.

    The real call's own deadline clamp is applied first, so what is recorded
    is the bound a worker would get, and a spent deadline raises unrecorded.
    """
    seen: list[Any] = []

    async def spy(policy, fn):
        del fn
        seen.append(blocking_io._within_deadline(policy))
        return result

    monkeypatch.setattr(module, "run_blocking_call", spy)
    return seen


# --- Defaults --------------------------------------------------------------


@pytest.mark.parametrize("field, expected", sorted(SHIPPED_DEFAULTS.items()))
def test_shipped_default_is_unchanged(field, expected):
    assert Settings.model_fields[field].default == expected


@pytest.mark.parametrize(
    "policy, field",
    [
        (YFINANCE_INFO_POLICY, "company_name_lookup_timeout_seconds"),
        (YAHOOQUERY_QUOTE_TYPE_POLICY, "company_name_lookup_timeout_seconds"),
        (GAP_FILL_YFINANCE_INFO_POLICY, "company_name_lookup_timeout_seconds"),
        (FX_RATE_POLICY, "fx_rate_fetch_timeout_seconds"),
    ],
    ids=lambda value: getattr(value, "label", value),
)
def test_policy_constant_matches_its_setting_default(policy, field):
    # The constant's number is only a default; the call site applies the field.
    # Pinning them stops a reader of src/blocking_io.py being misled.
    assert policy.hard_timeout_seconds == Settings.model_fields[field].default


# --- The clamp itself ------------------------------------------------------


def test_clamp_without_caller_deadline_keeps_default():
    assert clamp_to_operational_deadline(15) == 15.0


def test_clamp_takes_the_tighter_of_default_and_remaining(fake_clock):
    with enclosing_deadline(4.0):
        assert clamp_to_operational_deadline(15.0) == pytest.approx(4.0)
        # Not inner < outer: a looser caller leaves the inner default alone.
        assert clamp_to_operational_deadline(2.0) == pytest.approx(2.0)


def test_require_allowance_raises_timeout_once_spent(fake_clock):
    with enclosing_deadline(1.0):
        fake_clock[0] += 5.0
        assert clamp_to_operational_deadline(10.0) < 0
        with pytest.raises(TimeoutError, match="deadline already passed"):
            require_allowance(10.0, label="probe")


# --- Aggregate metrics fetch -----------------------------------------------


class _AsyncioWaitForSpy:
    """Stands in for ``asyncio`` inside ``src.data.fetcher`` to record wait_for."""

    def __init__(self) -> None:
        self.timeouts: list[float] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(asyncio, name)

    async def wait_for(self, coro, timeout):
        self.timeouts.append(timeout)
        return await asyncio.wait_for(coro, timeout)


@pytest.mark.asyncio
async def test_aggregate_override_reaches_wait_for(monkeypatch):
    spy = _AsyncioWaitForSpy()
    monkeypatch.setattr(fetcher_module, "asyncio", spy)
    monkeypatch.setattr(config, "market_data_aggregate_fetch_timeout_seconds", 12.5)
    fetcher = SmartMarketDataFetcher()
    monkeypatch.setattr(fetcher, "_fetch_metrics_inner", AsyncMock(return_value={}))

    await fetcher._get_financial_metrics_uncached("TEST")

    assert spy.timeouts == [12.5]


@pytest.mark.asyncio
async def test_aggregate_caller_deadline_wins(monkeypatch, fake_clock):
    spy = _AsyncioWaitForSpy()
    monkeypatch.setattr(fetcher_module, "asyncio", spy)
    fetcher = SmartMarketDataFetcher()
    monkeypatch.setattr(fetcher, "_fetch_metrics_inner", AsyncMock(return_value={}))

    with enclosing_deadline(3.0):
        await fetcher._get_financial_metrics_uncached("TEST")

    assert spy.timeouts == [pytest.approx(3.0)]


@pytest.mark.asyncio
async def test_aggregate_timeout_returns_empty_payload(monkeypatch):
    monkeypatch.setattr(config, "market_data_aggregate_fetch_timeout_seconds", 0.01)
    fetcher = SmartMarketDataFetcher()

    async def stall(_ticker):
        await asyncio.sleep(1)

    monkeypatch.setattr(fetcher, "_fetch_metrics_inner", stall)

    assert await fetcher._get_financial_metrics_uncached("TEST") == {}


@pytest.mark.asyncio
async def test_aggregate_spent_deadline_never_starts_fetch(monkeypatch, fake_clock):
    fetcher = SmartMarketDataFetcher()
    inner = AsyncMock(return_value={"never": True})
    monkeypatch.setattr(fetcher, "_fetch_metrics_inner", inner)

    with enclosing_deadline(1.0):
        fake_clock[0] += 5.0
        assert await fetcher._get_financial_metrics_uncached("TEST") == {}

    inner.assert_not_called()


# --- Per-source fetch ------------------------------------------------------


def _source_fetcher() -> SimpleNamespace:
    return SimpleNamespace(
        _fetch_yfinance_enhanced=AsyncMock(return_value=None),
        _fetch_yahooquery_fallback=MagicMock(return_value=None),
        _fetch_fmp_fallback=AsyncMock(return_value=None),
        _fetch_eodhd_fallback=AsyncMock(return_value=None),
        _fetch_av_fallback=AsyncMock(return_value=None),
    )


@pytest.mark.asyncio
async def test_per_source_override_reaches_fetch_all_sources(monkeypatch):
    impl = AsyncMock(return_value={})
    monkeypatch.setattr(fetcher_module, "fetch_all_sources_parallel_impl", impl)
    monkeypatch.setattr(config, "market_data_per_source_timeout_seconds", 6.5)

    await SmartMarketDataFetcher()._fetch_all_sources_parallel("TEST")

    assert impl.await_args.args[2] == 6.5


@pytest.mark.asyncio
async def test_per_source_caller_deadline_wins(monkeypatch, fake_clock):
    import src.async_utils as async_utils

    seen = _spy_hard_timeout(monkeypatch, async_utils)
    monkeypatch.setattr(config, "ibkr_data_source_enabled", False)

    with enclosing_deadline(4.0):
        await fetch_all_sources_parallel(_source_fetcher(), "TEST", 15.0)

    assert seen == [pytest.approx(4.0)] * 5


@pytest.mark.asyncio
async def test_per_source_timeout_records_timeout_outcome(monkeypatch):
    monkeypatch.setattr(config, "ibkr_data_source_enabled", False)
    fetcher = _source_fetcher()

    async def stall(_symbol):
        await asyncio.sleep(1)

    fetcher._fetch_fmp_fallback = stall
    fetcher._fetch_yfinance_enhanced = AsyncMock(return_value={"currentPrice": 1.0})

    results = await fetch_all_sources_parallel(fetcher, "TEST", 0.01)

    assert results["fmp"] is None
    assert results["yfinance"] == {"currentPrice": 1.0}


@pytest.mark.asyncio
async def test_per_source_spent_deadline_starts_no_source(monkeypatch, fake_clock):
    monkeypatch.setattr(config, "ibkr_data_source_enabled", False)
    fetcher = _source_fetcher()

    with enclosing_deadline(1.0):
        fake_clock[0] += 5.0
        results = await fetch_all_sources_parallel(fetcher, "TEST", 15.0)

    assert set(results.values()) == {None}
    fetcher._fetch_yfinance_enhanced.assert_not_called()
    fetcher._fetch_fmp_fallback.assert_not_called()
    fetcher._fetch_yahooquery_fallback.assert_not_called()


# --- Vendor HTTP total and anchor lookup -----------------------------------


class _FakeResponse:
    status = 404

    async def text(self) -> str:
        return ""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeSession:
    """Records the ClientTimeout handed to the session and to each request."""

    def __init__(self, seen: list[tuple[str, float]], raise_on_get=None, **kwargs):
        seen.append(("session", kwargs["timeout"].total))
        self._seen = seen
        self._raise_on_get = raise_on_get

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def get(self, url, params=None, timeout=None):
        del url, params
        self._seen.append(("request", timeout.total))
        if self._raise_on_get is not None:
            raise self._raise_on_get
        return _FakeResponse()


def _patch_session(monkeypatch, module, raise_on_get=None) -> list[tuple[str, float]]:
    seen: list[tuple[str, float]] = []
    monkeypatch.setattr(
        module.aiohttp,
        "ClientSession",
        lambda **kwargs: _FakeSession(seen, raise_on_get=raise_on_get, **kwargs),
    )
    return seen


def _vendor_calls():
    return {
        "fmp": (
            fmp_fetcher,
            lambda: fmp_fetcher.FMPFetcher(api_key="k")._get("profile", {}),
        ),
        "eodhd": (
            eodhd_fetcher,
            lambda: eodhd_fetcher.EODHDFetcher(api_key="k").get_financial_metrics(
                "7203.T"
            ),
        ),
        "alpha_vantage": (
            alpha_vantage_fetcher,
            lambda: alpha_vantage_fetcher.AlphaVantageFetcher(
                api_key="k"
            ).get_financial_metrics("7203.T"),
        ),
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("vendor", ["fmp", "eodhd", "alpha_vantage"])
async def test_vendor_http_override_reaches_session_and_request(monkeypatch, vendor):
    module, call = _vendor_calls()[vendor]
    seen = _patch_session(monkeypatch, module)
    monkeypatch.setattr(config, "market_data_vendor_http_timeout_seconds", 7.5)

    assert await call() is None  # the fake answers 404

    assert seen == [("session", 7.5), ("request", 7.5)]


@pytest.mark.asyncio
@pytest.mark.parametrize("vendor", ["fmp", "eodhd", "alpha_vantage"])
async def test_vendor_http_caller_deadline_wins(monkeypatch, fake_clock, vendor):
    module, call = _vendor_calls()[vendor]
    seen = _patch_session(monkeypatch, module)

    with enclosing_deadline(2.5):
        await call()

    assert seen == [("session", pytest.approx(2.5)), ("request", pytest.approx(2.5))]


@pytest.mark.asyncio
@pytest.mark.parametrize("vendor", ["fmp", "eodhd", "alpha_vantage"])
async def test_vendor_http_timeout_returns_none(monkeypatch, vendor):
    module, call = _vendor_calls()[vendor]
    _patch_session(monkeypatch, module, raise_on_get=asyncio.TimeoutError())

    assert await call() is None


@pytest.mark.asyncio
@pytest.mark.parametrize("vendor", ["fmp", "eodhd", "alpha_vantage"])
async def test_vendor_http_spent_deadline_opens_no_session(
    monkeypatch, fake_clock, vendor
):
    module, call = _vendor_calls()[vendor]
    seen = _patch_session(monkeypatch, module)

    with enclosing_deadline(1.0):
        fake_clock[0] += 5.0
        assert await call() is None

    assert seen == []


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["verify_anchor_metrics", "get_company_name"])
async def test_anchor_override_is_separate_from_vendor_http(monkeypatch, method):
    seen = _patch_session(monkeypatch, eodhd_fetcher)
    monkeypatch.setattr(config, "market_data_anchor_lookup_timeout_seconds", 3.5)
    monkeypatch.setattr(config, "market_data_vendor_http_timeout_seconds", 99.0)

    fetcher = eodhd_fetcher.EODHDFetcher(api_key="k")
    assert await getattr(fetcher, method)("7203.T") is None

    assert seen == [("session", 3.5), ("request", 3.5)]


@pytest.mark.asyncio
async def test_anchor_caller_deadline_wins(monkeypatch, fake_clock):
    seen = _patch_session(monkeypatch, eodhd_fetcher)

    with enclosing_deadline(1.5):
        await eodhd_fetcher.EODHDFetcher(api_key="k").verify_anchor_metrics("7203.T")

    assert seen == [("session", pytest.approx(1.5)), ("request", pytest.approx(1.5))]


@pytest.mark.asyncio
async def test_anchor_timeout_returns_none(monkeypatch):
    _patch_session(monkeypatch, eodhd_fetcher, raise_on_get=asyncio.TimeoutError())

    fetcher = eodhd_fetcher.EODHDFetcher(api_key="k")
    assert await fetcher.verify_anchor_metrics("7203.T") is None


# --- Ticker resolution -----------------------------------------------------


def _resolving_fetcher(monkeypatch) -> SmartMarketDataFetcher:
    fetcher = SmartMarketDataFetcher()
    fetcher._mnemonic_cache = {}
    fetcher.tavily_client = MagicMock()
    monkeypatch.setattr(fetcher_module, "allows_search_resolution", lambda _s: True)
    return fetcher


async def _resolve_both(fetcher: SmartMarketDataFetcher) -> tuple[str, str | None]:
    pre = await fetcher._pre_resolve_ticker("SCIENTX.KL")
    via_search = await fetcher._resolve_ticker_via_search("PADINI.KL")
    return pre, via_search


@pytest.mark.asyncio
async def test_ticker_resolution_override_reaches_both_lookups(monkeypatch):
    fetcher = _resolving_fetcher(monkeypatch)
    seen = _spy_hard_timeout(monkeypatch, fetcher_module, result={})
    monkeypatch.setattr(config, "market_data_ticker_resolution_timeout_seconds", 4.25)

    await _resolve_both(fetcher)

    assert seen == [4.25, 4.25]


@pytest.mark.asyncio
async def test_ticker_resolution_caller_deadline_wins(monkeypatch, fake_clock):
    fetcher = _resolving_fetcher(monkeypatch)
    seen = _spy_hard_timeout(monkeypatch, fetcher_module, result={})

    with enclosing_deadline(2.0):
        await _resolve_both(fetcher)

    assert seen == [pytest.approx(2.0)] * 2


@pytest.mark.asyncio
async def test_ticker_resolution_spent_deadline_keeps_fallbacks(
    monkeypatch, fake_clock
):
    fetcher = _resolving_fetcher(monkeypatch)
    seen = _spy_hard_timeout(monkeypatch, fetcher_module, result={})

    with enclosing_deadline(1.0):
        fake_clock[0] += 5.0
        pre, via_search = await _resolve_both(fetcher)

    # Today's timeout fallbacks: the original ticker, and no rescue.
    assert (pre, via_search) == ("SCIENTX.KL", None)
    assert seen == []


# --- Price history ---------------------------------------------------------


_HISTORY = pd.DataFrame({"Close": [1.0, 2.0]})


@pytest.mark.asyncio
async def test_price_history_override_reaches_hard_timeout(monkeypatch):
    monkeypatch.setattr(fetcher_module.yf, "Ticker", MagicMock())
    seen = _spy_hard_timeout(monkeypatch, fetcher_module, result=_HISTORY)
    monkeypatch.setattr(config, "market_data_price_history_timeout_seconds", 9.5)

    hist = await SmartMarketDataFetcher()._get_price_history_uncached("TEST")

    assert not hist.empty
    assert seen == [9.5]


@pytest.mark.asyncio
async def test_price_history_caller_deadline_wins(monkeypatch, fake_clock):
    monkeypatch.setattr(fetcher_module.yf, "Ticker", MagicMock())
    seen = _spy_hard_timeout(monkeypatch, fetcher_module, result=_HISTORY)

    with enclosing_deadline(6.0):
        await SmartMarketDataFetcher()._get_price_history_uncached("TEST")

    assert seen == [pytest.approx(6.0)]


@pytest.mark.asyncio
async def test_price_history_timeout_returns_empty_frame(monkeypatch):
    monkeypatch.setattr(fetcher_module.yf, "Ticker", MagicMock())

    async def fires(coro, *, timeout, label):
        del timeout, label
        coro.close()
        raise asyncio.TimeoutError("fired")

    monkeypatch.setattr(fetcher_module, "run_with_hard_timeout", fires)

    hist = await SmartMarketDataFetcher()._get_price_history_uncached("TEST")

    assert isinstance(hist, pd.DataFrame) and hist.empty


@pytest.mark.asyncio
async def test_price_history_spent_deadline_never_downloads(monkeypatch, fake_clock):
    monkeypatch.setattr(fetcher_module.yf, "Ticker", MagicMock())
    seen = _spy_hard_timeout(monkeypatch, fetcher_module, result=_HISTORY)

    with enclosing_deadline(1.0):
        fake_clock[0] += 5.0
        hist = await SmartMarketDataFetcher()._get_price_history_uncached("TEST")

    assert hist.empty
    assert seen == []


# --- Company-name lookup (ticker_utils and gap fill) -----------------------


@pytest.mark.asyncio
async def test_company_name_override_reaches_blocking_policies(monkeypatch):
    seen = _spy_blocking_call(monkeypatch, ticker_utils, result=None)
    monkeypatch.setattr(config, "company_name_lookup_timeout_seconds", 2.75)

    await ticker_utils._try_yfinance("7203.T")
    await ticker_utils._try_yahooquery("7203.T")

    assert [policy.hard_timeout_seconds for policy in seen] == [2.75, 2.75]
    assert seen[0].label == "yfinance.info:7203.T"


@pytest.mark.asyncio
async def test_company_name_caller_deadline_wins(monkeypatch, fake_clock):
    seen = _spy_blocking_call(monkeypatch, ticker_utils, result=None)

    with enclosing_deadline(1.25):
        await ticker_utils._try_yfinance("7203.T")

    assert seen[0].hard_timeout_seconds == pytest.approx(1.25)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "resolver, factory",
    [
        (ticker_utils._try_fmp, "src.data.fmp_fetcher.get_fmp_fetcher"),
        (ticker_utils._try_eodhd, "src.data.eodhd_fetcher.get_eodhd_fetcher"),
    ],
    ids=["fmp", "eodhd"],
)
async def test_company_name_vendor_timeout_falls_through(
    monkeypatch, resolver, factory
):
    async def stall(_ticker):
        await asyncio.sleep(1)

    vendor = SimpleNamespace(is_available=lambda: True, get_company_name=stall)
    monkeypatch.setattr(factory, lambda: vendor)
    monkeypatch.setattr(config, "company_name_lookup_timeout_seconds", 0.01)

    assert await resolver("7203.T") is None


@pytest.mark.asyncio
async def test_company_name_spent_deadline_creates_no_request(monkeypatch, fake_clock):
    get_company_name = MagicMock()
    vendor = SimpleNamespace(
        is_available=lambda: True, get_company_name=get_company_name
    )
    monkeypatch.setattr("src.data.fmp_fetcher.get_fmp_fetcher", lambda: vendor)
    seen = _spy_blocking_call(monkeypatch, ticker_utils, result=None)

    with enclosing_deadline(1.0):
        fake_clock[0] += 5.0
        assert await ticker_utils._try_yfinance("7203.T") is None
        assert await ticker_utils._try_fmp("7203.T") is None

    assert seen == []
    get_company_name.assert_not_called()


async def _gap_fill(monkeypatch, seen_names: list[str]) -> None:
    def query_builder(symbol, company_name, terms):
        del symbol, terms
        seen_names.append(company_name)
        return "q"

    await gap_fill._fetch_tavily_gaps_impl(
        SimpleNamespace(tavily_client=object()),
        "7203.T",
        ["revenueGrowth"],
        yf_module=MagicMock(),
        search_fn=AsyncMock(return_value={}),
        query_builder=query_builder,
    )


@pytest.mark.asyncio
async def test_gap_fill_default_policy_takes_setting_and_keeps_shared_cap(
    monkeypatch,
):
    seen = _spy_blocking_call(monkeypatch, gap_fill, result={"longName": "Toyota"})
    monkeypatch.setattr(config, "company_name_lookup_timeout_seconds", 3.0)
    names: list[str] = []

    await _gap_fill(monkeypatch, names)

    assert seen[0].hard_timeout_seconds == 3.0
    assert seen[0].in_flight is GAP_FILL_YFINANCE_INFO_POLICY.in_flight
    assert names == ["Toyota"]


@pytest.mark.asyncio
async def test_gap_fill_spent_deadline_falls_back_to_symbol(monkeypatch, fake_clock):
    seen = _spy_blocking_call(monkeypatch, gap_fill, result={"longName": "Toyota"})
    names: list[str] = []

    with enclosing_deadline(1.0):
        fake_clock[0] += 5.0
        await _gap_fill(monkeypatch, names)

    assert seen == []
    assert names == ["7203.T"]


# --- FX rate fetch ---------------------------------------------------------


@pytest.fixture
def quiet_yfinance(monkeypatch):
    monkeypatch.setattr(
        "src.yfinance_runtime.configure_yfinance_defaults", lambda: None
    )


@pytest.mark.asyncio
async def test_fx_override_reaches_blocking_policy(monkeypatch, quiet_yfinance):
    seen = _spy_blocking_call(monkeypatch, fx_normalization, result=0.0067)
    monkeypatch.setattr(config, "fx_rate_fetch_timeout_seconds", 4.0)

    rate = await fx_normalization.get_fx_rate_yfinance("JPY", "USD")

    assert rate == pytest.approx(0.0067)
    assert seen[0].hard_timeout_seconds == 4.0
    assert seen[0].label == "fx_rate:JPYUSD=X"


@pytest.mark.asyncio
async def test_fx_caller_deadline_wins(monkeypatch, fake_clock, quiet_yfinance):
    seen = _spy_blocking_call(monkeypatch, fx_normalization, result=0.0067)

    with enclosing_deadline(0.5):
        await fx_normalization.get_fx_rate_yfinance("JPY", "USD")

    assert seen[0].hard_timeout_seconds == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_fx_spent_deadline_returns_none_for_static_fallback(
    monkeypatch, fake_clock, quiet_yfinance
):
    seen = _spy_blocking_call(monkeypatch, fx_normalization, result=0.0067)

    with enclosing_deadline(1.0):
        fake_clock[0] += 5.0
        assert await fx_normalization.get_fx_rate_yfinance("JPY", "USD") is None

    assert seen == []


# --- Consultant spot check -------------------------------------------------


async def _spot_check() -> dict[str, Any]:
    import json

    raw = await consultant_tools.spot_check_metric.ainvoke(
        {"ticker": "7203.T", "metric": "trailingPE"}
    )
    return json.loads(raw)


@pytest.mark.asyncio
async def test_spot_check_override_reaches_hard_timeout(monkeypatch):
    monkeypatch.setattr(consultant_tools.yf, "Ticker", MagicMock())
    seen = _spy_hard_timeout(monkeypatch, consultant_tools, result={"trailingPE": 9})
    monkeypatch.setattr(config, "consultant_spot_check_timeout_seconds", 3.5)

    result = await _spot_check()

    assert result["value"] == 9
    assert seen == [3.5]


@pytest.mark.asyncio
async def test_spot_check_caller_deadline_wins(monkeypatch, fake_clock):
    monkeypatch.setattr(consultant_tools.yf, "Ticker", MagicMock())
    seen = _spy_hard_timeout(monkeypatch, consultant_tools, result={"trailingPE": 9})

    with enclosing_deadline(2.0):
        await _spot_check()

    assert seen == [pytest.approx(2.0)]


@pytest.mark.asyncio
async def test_spot_check_spent_deadline_returns_timeout_payload(
    monkeypatch, fake_clock
):
    ticker_cls = MagicMock()
    monkeypatch.setattr(consultant_tools.yf, "Ticker", ticker_cls)
    seen = _spy_hard_timeout(monkeypatch, consultant_tools, result={"trailingPE": 9})

    with enclosing_deadline(1.0):
        fake_clock[0] += 5.0
        result = await _spot_check()

    assert "timed out" in result.get("error", "").lower()
    assert seen == []
    ticker_cls.assert_not_called()
