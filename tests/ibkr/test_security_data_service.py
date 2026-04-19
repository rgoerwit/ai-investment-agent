from __future__ import annotations

from unittest.mock import patch

import pytest

from src.ibkr.security_data_service import IbkrSecurityDataService


class _FakeConfig:
    def __init__(self, configured: bool = True):
        self._configured = configured

    def is_configured(self) -> bool:
        return self._configured


class _FakeClient:
    def __init__(self, _config):
        self.connected = False

    def connect(self, brokerage_session: bool = False) -> None:
        self.connected = True

    def close(self) -> None:
        self.connected = False

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
        self,
        conid: int,
        *,
        fields: str = "",
        compete: bool = False,
    ):
        return {
            "31": "4.56",
            "84": "4.55",
            "86": "4.57",
            "87": "12345",
            "6509": "R",
            "7051": "Modern Dental Group",
            "55": "3600",
        }


class _AmbiguousClient(_FakeClient):
    def stock_conid_by_symbol(self, symbol: str, default_filtering: bool = False):
        return {
            symbol: [
                {"conid": 1, "exchange": "SGX", "symbol": symbol, "currency": "SGD"},
                {"conid": 2, "exchange": "SEHK", "symbol": symbol, "currency": "HKD"},
            ]
        }


class _WrongExchangeSingleClient(_FakeClient):
    def stock_conid_by_symbol(self, symbol: str, default_filtering: bool = False):
        return {
            symbol: [
                {"conid": 99, "exchange": "NYSE", "symbol": symbol, "currency": "USD"}
            ]
        }


def test_probe_returns_neutral_when_ibkr_not_configured():
    service = IbkrSecurityDataService(config=_FakeConfig(configured=False))

    probe = service._probe_security_sync("3600.HK")

    assert probe.configured is False
    assert probe.identity_confidence == "UNVERIFIED"
    assert probe.error_kind == "NOT_CONFIGURED"


def test_probe_returns_verified_identity_and_quote():
    service = IbkrSecurityDataService(
        config=_FakeConfig(configured=True),
        client_cls=_FakeClient,
    )

    with patch("src.ibkr.security_data_service.cache_conid_mapping") as mock_cache:
        probe = service._probe_security_sync("3600.HK")

    assert probe.configured is True
    assert probe.identity_confidence == "VERIFIED"
    assert probe.company_name == "Modern Dental Group"
    assert probe.resolved_yf_ticker == "3600.HK"
    assert probe.last_price == 4.56
    assert probe.market_data_availability == "R"
    mock_cache.assert_called_once()


def test_probe_marks_multiple_unmatched_candidates_as_ambiguous():
    service = IbkrSecurityDataService(
        config=_FakeConfig(configured=True),
        client_cls=_AmbiguousClient,
    )

    probe = service._probe_security_sync("BEC.SG")

    assert probe.configured is True
    assert probe.identity_confidence == "AMBIGUOUS"
    assert probe.error_kind == "AMBIGUOUS"
    assert probe.last_price is None


def test_probe_marks_single_wrong_exchange_candidate_as_ambiguous():
    service = IbkrSecurityDataService(
        config=_FakeConfig(configured=True),
        client_cls=_WrongExchangeSingleClient,
    )

    with patch("src.ibkr.security_data_service.cache_conid_mapping") as mock_cache:
        probe = service._probe_security_sync("BEC.SG")

    assert probe.configured is True
    assert probe.identity_confidence == "AMBIGUOUS"
    assert probe.error_kind == "AMBIGUOUS"
    mock_cache.assert_not_called()


def test_select_candidate_single_no_expected_exchange_is_verified():
    result, confidence = IbkrSecurityDataService._select_candidate(
        [{"conid": 1, "exchange": "NYSE", "symbol": "AAPL"}],
        expected_exchange="",
    )

    assert result is not None
    assert confidence == "VERIFIED"


def test_snapshot_number_handles_ibkr_prefixes():
    assert IbkrSecurityDataService._snapshot_number("C123.45") == pytest.approx(123.45)
    assert IbkrSecurityDataService._snapshot_number("H9.87") == pytest.approx(9.87)
    assert IbkrSecurityDataService._snapshot_number("") is None
    assert IbkrSecurityDataService._snapshot_number(None) is None


@pytest.mark.asyncio
async def test_probe_async_cache_reuses_previous_result():
    service = IbkrSecurityDataService(
        config=_FakeConfig(configured=True),
        client_cls=_FakeClient,
    )

    with patch.object(
        service, "_probe_security_sync", wraps=service._probe_security_sync
    ) as wrapped:
        first = await service.probe_security("3600.HK")
        second = await service.probe_security("3600.HK")

    assert first == second
    assert wrapped.call_count == 1
