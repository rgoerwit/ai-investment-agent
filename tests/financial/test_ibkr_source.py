"""Tests for the optional IBKR advisory market-data source in the analysis pipeline.

Covers: probe fundamental enrichment, the shared service accessor, source gating and
optionality (including partial / absent market-data entitlement), gap-fill vs override
quality, and the advisory-conflict surfacing. No live IBKR creds are used — a fake
config/client is injected exactly like ``tests/ibkr/test_security_data_service.py``.

The load-bearing invariant under test: nothing depends on IBKR being available, opted
in, or fully entitled. With the flag off the pipeline is unchanged; with the flag on but
unavailable/partial, only the fields IBKR actually returns are used.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from src.config import config
from src.data.fetcher import SmartMarketDataFetcher
from src.data.merge_policy import SOURCE_QUALITY, collect_ibkr_advisory_conflicts
from src.ibkr.security_data_service import (
    IbkrSecurityDataService,
    IbkrSecurityProbe,
    get_security_data_service,
    set_security_data_service,
)


class _FakeConfig:
    def __init__(self, configured: bool = True) -> None:
        self._configured = configured

    def is_configured(self) -> bool:
        return self._configured


class _FundamentalsClient:
    """Fake IBKR client whose snapshot returns identity + fundamental field codes."""

    snapshot: dict[str, str] = {
        "31": "4.56",  # last price
        "55": "3600",  # symbol
        "6509": "R",  # market-data availability
        "7051": "Modern Dental Group",
        "7289": "5000",  # market cap (candidate code)
        "7290": "12.5",  # trailing P/E
        "7291": "0.36",  # EPS
        "7287": "3.2",  # dividend yield
        "7293": "6.10",  # 52-week high
        "7294": "3.80",  # 52-week low
    }

    def __init__(self, _config) -> None:
        pass

    def connect(
        self, brokerage_session: bool = False, *, maintain: bool = False
    ) -> None:
        pass

    def close(self) -> None:
        pass

    def logout(self) -> None:
        pass

    def stock_conid_by_symbol(self, symbol: str, default_filtering: bool = False):
        return {
            symbol: [
                {"conid": 3600, "exchange": "SEHK", "symbol": "3600", "currency": "HKD"}
            ]
        }

    def get_contract_info(self, conid: int, *, compete: bool = True):
        return {
            "symbol": "3600",
            "exchange": "SEHK",
            "primaryExch": "SEHK",
            "currency": "HKD",
            "companyName": "Modern Dental",
        }

    def get_marketdata_snapshot(
        self, conid: int, *, fields: str = "", compete: bool = False
    ):
        return dict(self.snapshot)


class _PartialFundamentalsClient(_FundamentalsClient):
    """Entitled for only some fields — others come back blank (no subscription)."""

    snapshot = {
        "31": "4.56",
        "55": "3600",
        "6509": "R",
        "7051": "Modern Dental Group",
        "7290": "12.5",  # only P/E entitled
        "7289": "",  # market cap blank
    }


class _NoFundamentalsClient(_FundamentalsClient):
    """Identity resolves but no fundamental fields return at all."""

    snapshot = {"31": "4.56", "55": "3600", "6509": "R", "7051": "Modern Dental Group"}


def _verified_probe(**ratios) -> IbkrSecurityProbe:
    return IbkrSecurityProbe(
        configured=True,
        requested_ticker="3600.HK",
        identity_confidence="VERIFIED",
        last_price=ratios.get("last_price"),
        trailing_pe=ratios.get("trailing_pe"),
        eps=ratios.get("eps"),
        market_cap=ratios.get("market_cap"),
        fifty_two_week_high=ratios.get("fifty_two_week_high"),
        fifty_two_week_low=ratios.get("fifty_two_week_low"),
        fundamentals_status="OK",
    )


class _StubService:
    """Duck-typed stand-in for the shared IBKR service."""

    def __init__(self, probe: IbkrSecurityProbe) -> None:
        self._probe = probe

    async def probe_security(self, ticker: str) -> IbkrSecurityProbe:
        return self._probe


# --------------------------------------------------------------------------- #
# Probe fundamental enrichment
# --------------------------------------------------------------------------- #


def test_probe_populates_fundamental_fields():
    service = IbkrSecurityDataService(
        config=_FakeConfig(True), client_cls=_FundamentalsClient
    )
    with patch("src.ibkr.security_data_service.cache_conid_mapping"):
        probe = service._probe_security_sync("3600.HK")
    assert probe.identity_confidence == "VERIFIED"
    assert probe.trailing_pe == 12.5
    assert probe.eps == 0.36
    assert probe.market_cap == 5000
    assert probe.dividend_yield == 3.2
    assert probe.fifty_two_week_high == 6.10
    assert probe.fifty_two_week_low == 3.80
    assert probe.fundamentals_status == "OK"


def test_probe_partial_entitlement_keeps_present_fields_only():
    service = IbkrSecurityDataService(
        config=_FakeConfig(True), client_cls=_PartialFundamentalsClient
    )
    with patch("src.ibkr.security_data_service.cache_conid_mapping"):
        probe = service._probe_security_sync("3600.HK")
    assert probe.trailing_pe == 12.5
    assert probe.market_cap is None  # blank → None, not an error
    assert probe.fundamentals_status == "OK"  # at least one field present


def test_probe_no_fundamentals_sets_status_no_fields():
    service = IbkrSecurityDataService(
        config=_FakeConfig(True), client_cls=_NoFundamentalsClient
    )
    with patch("src.ibkr.security_data_service.cache_conid_mapping"):
        probe = service._probe_security_sync("3600.HK")
    assert probe.identity_confidence == "VERIFIED"
    assert probe.trailing_pe is None
    assert probe.fundamentals_status == "NO_FIELDS"


# --------------------------------------------------------------------------- #
# Shared accessor / consolidation
# --------------------------------------------------------------------------- #


def test_shared_accessor_returns_singleton():
    set_security_data_service(None)
    try:
        assert get_security_data_service() is get_security_data_service()
    finally:
        set_security_data_service(None)


def test_fetcher_and_ticker_utils_share_one_service():
    from src import ticker_utils

    sentinel = IbkrSecurityDataService(config=_FakeConfig(False))
    set_security_data_service(sentinel)
    try:
        assert SmartMarketDataFetcher()._get_ibkr_security_service() is sentinel
        assert ticker_utils._get_ibkr_name_service() is sentinel
    finally:
        set_security_data_service(None)


# --------------------------------------------------------------------------- #
# Source gating / optionality
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_fallback_noop_when_flag_off():
    # Default flag is False — no IBKR access attempted at all.
    assert await SmartMarketDataFetcher()._fetch_ibkr_fallback("3600.HK") is None


@pytest.mark.asyncio
async def test_fallback_noop_when_not_configured(monkeypatch):
    monkeypatch.setattr(config, "ibkr_data_source_enabled", True)
    monkeypatch.setattr(
        "src.ibkr_config.IbkrSettings.is_configured", lambda self: False
    )
    assert await SmartMarketDataFetcher()._fetch_ibkr_fallback("3600.HK") is None


@pytest.mark.asyncio
async def test_fallback_noop_when_identity_unverified(monkeypatch):
    monkeypatch.setattr(config, "ibkr_data_source_enabled", True)
    monkeypatch.setattr("src.ibkr_config.IbkrSettings.is_configured", lambda self: True)
    probe = IbkrSecurityProbe(
        configured=True, requested_ticker="X", identity_confidence="AMBIGUOUS"
    )
    set_security_data_service(_StubService(probe))
    try:
        assert await SmartMarketDataFetcher()._fetch_ibkr_fallback("X") is None
    finally:
        set_security_data_service(None)


@pytest.mark.asyncio
async def test_fallback_maps_ratios_and_increments_stats(monkeypatch):
    monkeypatch.setattr(config, "ibkr_data_source_enabled", True)
    monkeypatch.setattr("src.ibkr_config.IbkrSettings.is_configured", lambda self: True)
    probe = _verified_probe(
        last_price=4.56,
        trailing_pe=12.5,
        market_cap=5000.0,
        eps=0.36,
        fifty_two_week_high=6.1,
        fifty_two_week_low=3.8,
    )
    set_security_data_service(_StubService(probe))
    try:
        fetcher = SmartMarketDataFetcher()
        result = await fetcher._fetch_ibkr_fallback("3600.HK")
        assert result == {
            "currentPrice": 4.56,
            "trailingPE": 12.5,
            "marketCap": 5000.0,
            "trailingEps": 0.36,
            "fiftyTwoWeekHigh": 6.1,
            "fiftyTwoWeekLow": 3.8,
        }
        assert fetcher.stats["sources"]["ibkr"] == 1
    finally:
        set_security_data_service(None)


@pytest.mark.asyncio
async def test_fallback_partial_returns_only_present_fields(monkeypatch):
    monkeypatch.setattr(config, "ibkr_data_source_enabled", True)
    monkeypatch.setattr("src.ibkr_config.IbkrSettings.is_configured", lambda self: True)
    set_security_data_service(_StubService(_verified_probe(trailing_pe=12.5)))
    try:
        result = await SmartMarketDataFetcher()._fetch_ibkr_fallback("3600.HK")
        assert result == {"trailingPE": 12.5}  # nothing else assumed present
    finally:
        set_security_data_service(None)


@pytest.mark.asyncio
async def test_fallback_returns_none_when_no_ratios(monkeypatch):
    monkeypatch.setattr(config, "ibkr_data_source_enabled", True)
    monkeypatch.setattr("src.ibkr_config.IbkrSettings.is_configured", lambda self: True)
    set_security_data_service(_StubService(_verified_probe()))  # all None
    try:
        assert await SmartMarketDataFetcher()._fetch_ibkr_fallback("3600.HK") is None
    finally:
        set_security_data_service(None)


@pytest.mark.asyncio
async def test_fallback_swallows_exceptions(monkeypatch):
    monkeypatch.setattr(config, "ibkr_data_source_enabled", True)
    monkeypatch.setattr("src.ibkr_config.IbkrSettings.is_configured", lambda self: True)

    class _Boom:
        async def probe_security(self, ticker: str):
            raise RuntimeError("ibkr down")

    set_security_data_service(_Boom())
    try:
        assert await SmartMarketDataFetcher()._fetch_ibkr_fallback("3600.HK") is None
    finally:
        set_security_data_service(None)


# --------------------------------------------------------------------------- #
# Parallel fetch registration (flag-gated)
# --------------------------------------------------------------------------- #


def _patch_core_sources(fetcher, monkeypatch):
    monkeypatch.setattr(
        fetcher, "_fetch_yfinance_enhanced", AsyncMock(return_value={"symbol": "X"})
    )
    monkeypatch.setattr(fetcher, "_fetch_yahooquery_fallback", lambda s: None)
    monkeypatch.setattr(fetcher, "_fetch_fmp_fallback", AsyncMock(return_value=None))
    monkeypatch.setattr(fetcher, "_fetch_eodhd_fallback", AsyncMock(return_value=None))
    monkeypatch.setattr(fetcher, "_fetch_av_fallback", AsyncMock(return_value=None))


@pytest.mark.asyncio
async def test_ibkr_absent_from_fetch_when_flag_off(monkeypatch):
    fetcher = SmartMarketDataFetcher()
    _patch_core_sources(fetcher, monkeypatch)
    results = await fetcher._fetch_all_sources_parallel("X")
    assert "ibkr" not in results  # pipeline byte-for-byte unchanged when opted out


@pytest.mark.asyncio
async def test_ibkr_present_in_fetch_when_flag_on(monkeypatch):
    monkeypatch.setattr(config, "ibkr_data_source_enabled", True)
    fetcher = SmartMarketDataFetcher()
    _patch_core_sources(fetcher, monkeypatch)
    monkeypatch.setattr(
        fetcher, "_fetch_ibkr_fallback", AsyncMock(return_value={"trailingPE": 12.0})
    )
    results = await fetcher._fetch_all_sources_parallel("X")
    assert results.get("ibkr") == {"trailingPE": 12.0}


# --------------------------------------------------------------------------- #
# Merge precedence (IBKR > Yahoo/FMP, IBKR < EODHD/statements)
# --------------------------------------------------------------------------- #


def test_ibkr_overrides_yahoo_quality():
    merged, metadata = SmartMarketDataFetcher()._smart_merge_with_quality(
        {"yfinance": {"trailingPE": 20.0}, "ibkr": {"trailingPE": 12.0}}, "TEST"
    )
    assert merged["trailingPE"] == 12.0
    assert metadata["field_sources"]["trailingPE"] == "ibkr"


def test_ibkr_defers_to_eodhd():
    merged, metadata = SmartMarketDataFetcher()._smart_merge_with_quality(
        {"eodhd": {"trailingPE": 20.0}, "ibkr": {"trailingPE": 12.0}}, "TEST"
    )
    assert merged["trailingPE"] == 20.0
    assert metadata["field_sources"]["trailingPE"] == "eodhd"


def test_ibkr_gap_fills_null():
    merged, metadata = SmartMarketDataFetcher()._smart_merge_with_quality(
        {
            "yfinance": {"trailingPE": None, "marketCap": 1e9},
            "ibkr": {"trailingPE": 12.0},
        },
        "TEST",
    )
    assert merged["trailingPE"] == 12.0
    assert metadata["field_sources"]["trailingPE"] == "ibkr"


def test_ibkr_divergence_recorded_in_source_conflicts():
    _merged, metadata = SmartMarketDataFetcher()._smart_merge_with_quality(
        {"eodhd": {"trailingPE": 20.0}, "ibkr": {"trailingPE": 12.0}}, "TEST"
    )
    conflict = metadata["source_conflicts"]["trailingPE"]
    assert conflict["new_source"] == "ibkr"
    assert conflict["old_source"] == "eodhd"


# --------------------------------------------------------------------------- #
# Advisory-conflict surfacing
# --------------------------------------------------------------------------- #


def test_collect_ibkr_advisory_conflicts_collects_ibkr_party_only():
    merged = {
        "_source_conflicts": {
            "trailingPE": {
                "old": 20.0,
                "old_source": "eodhd",
                "new": 12.0,
                "new_source": "ibkr",
                "variance_pct": 40.0,
            },
            "marketCap": {
                "old": 1.0,
                "old_source": "yfinance",
                "new": 1.2,
                "new_source": "fmp",
                "variance_pct": 20.0,
            },
        }
    }
    collect_ibkr_advisory_conflicts(merged)
    advisory = merged["_ibkr_advisory_conflicts"]
    assert len(advisory) == 1
    assert advisory[0]["field"] == "trailingPE"
    assert advisory[0]["advisory"] is True


def test_collect_ibkr_advisory_conflicts_noop_without_ibkr():
    merged = {
        "_source_conflicts": {
            "marketCap": {"new_source": "fmp", "old_source": "yfinance"}
        }
    }
    collect_ibkr_advisory_conflicts(merged)
    assert "_ibkr_advisory_conflicts" not in merged


# --------------------------------------------------------------------------- #
# Registration canary
# --------------------------------------------------------------------------- #


def test_ibkr_registered_in_quality_and_stats():
    assert SOURCE_QUALITY["ibkr"] == 9.4
    assert "ibkr" in SmartMarketDataFetcher().stats["sources"]


# --------------------------------------------------------------------------- #
# Contract: neutered IBKR adds no value; improved entitlement auto-recovers
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_source_reattempts_after_unavailable_no_permanent_latch(monkeypatch):
    """A neutered IBKR must NOT latch off. If entitlement/availability later improves
    (vendors change this), the next probe picks up the now-available data — there is no
    circuit-breaker that permanently disables the source after an empty result."""
    monkeypatch.setattr(config, "ibkr_data_source_enabled", True)
    monkeypatch.setattr("src.ibkr_config.IbkrSettings.is_configured", lambda self: True)
    fetcher = SmartMarketDataFetcher()
    try:
        # Run 1 — not entitled: probe verifies identity but returns no ratios → no-op.
        set_security_data_service(_StubService(_verified_probe()))
        assert await fetcher._fetch_ibkr_fallback("3600.HK") is None
        # Run 2 — user adds a market-data subscription: same ticker now returns data,
        # and the source uses it with no intervention.
        set_security_data_service(_StubService(_verified_probe(trailing_pe=12.5)))
        assert await fetcher._fetch_ibkr_fallback("3600.HK") == {"trailingPE": 12.5}
    finally:
        set_security_data_service(None)


def test_empty_ibkr_leaves_merged_data_identical():
    """IBKR present-but-empty in the source results yields merged DATA identical to a
    merge with no IBKR at all — nothing downstream expects value from a neutered IBKR."""
    fetcher = SmartMarketDataFetcher()
    base = {"yfinance": {"trailingPE": 20.0, "marketCap": 1e9}}
    merged_without, _ = fetcher._smart_merge_with_quality(dict(base), "TEST")
    merged_with_empty, _ = fetcher._smart_merge_with_quality(
        {**base, "ibkr": None}, "TEST"
    )
    assert merged_with_empty == merged_without


# --------------------------------------------------------------------------- #
# PHASE-6 advisory layer — the branch that fires the day entitlement starts
# serving (never exercised by the live UNAVAILABLE run; cover it directly).
# --------------------------------------------------------------------------- #


def test_apply_ibkr_advisory_ok_branch_populates_everything(monkeypatch):
    monkeypatch.setattr(config, "ibkr_data_source_enabled", True)
    fetcher = SmartMarketDataFetcher()
    merged = {
        "trailingPE": 12.0,
        "_field_sources": {"trailingPE": "ibkr"},
        "_source_conflicts": {
            "trailingPE": {
                "old": 20.0,
                "old_source": "eodhd",
                "new": 12.0,
                "new_source": "ibkr",
                "variance_pct": 40.0,
            }
        },
    }
    source_results = {
        "ibkr": {"trailingPE": 12.0, "marketCap": 5e8, "currentPrice": None}
    }
    fetcher._apply_ibkr_advisory(merged, source_results)
    assert merged["_ibkr_advisory_status"] == "OK"
    # None-valued fields dropped from the retained alternative set
    assert merged["_ibkr_metrics"] == {"trailingPE": 12.0, "marketCap": 5e8}
    advisory = merged["_ibkr_advisory_conflicts"]
    assert advisory[0]["field"] == "trailingPE"
    assert advisory[0]["advisory"] is True


def test_apply_ibkr_advisory_unavailable_branch(monkeypatch):
    monkeypatch.setattr(config, "ibkr_data_source_enabled", True)
    fetcher = SmartMarketDataFetcher()
    merged = {"trailingPE": 20.0}
    fetcher._apply_ibkr_advisory(merged, {"ibkr": None})
    assert merged["_ibkr_advisory_status"] == "UNAVAILABLE"
    assert "_ibkr_metrics" not in merged


def test_apply_ibkr_advisory_adds_nothing_when_flag_off():
    fetcher = SmartMarketDataFetcher()  # flag defaults off
    merged = {"trailingPE": 20.0}
    fetcher._apply_ibkr_advisory(merged, {"ibkr": {"trailingPE": 12.0}})
    assert merged == {"trailingPE": 20.0}  # opted out => completely untouched
