from __future__ import annotations

import pytest

from src.ibkr.account_service import IbkrAccountService


class FakeConfig:
    ibkr_account_id = "U123456"


class FakeClient:
    def __init__(self, config):
        self.config = config
        self.connected = False
        self.closed = False

    def connect(self, *, brokerage_session: bool) -> None:
        assert brokerage_session is False
        self.connected = True

    def close(self) -> None:
        self.closed = True

    def get_accounts(self) -> list[str]:
        assert self.connected is True
        return ["U123456", "U654321"]

    def get_ledger(self, account_id: str) -> dict:
        assert account_id == "U123456"
        return {"BASE": {"cashbalance": 1000.0, "netliquidationvalue": 5000.0}}

    def get_positions(self, account_id: str) -> list[dict]:
        assert account_id == "U123456"
        return [{"conid": 1}, {"conid": 2}]


@pytest.mark.asyncio
async def test_verify_connection_returns_current_auth_summary_shape():
    checked: list[object] = []
    validated: list[object] = []
    prompted: list[object] = []

    service = IbkrAccountService(
        config=FakeConfig(),
        client_cls=FakeClient,
        check_config_fn=lambda config: checked.append(config),
        validate_key_files_fn=lambda config: (
            validated.append(config) or {"signature_key": "ok", "encryption_key": "ok"}
        ),
        prompt_for_missing_secret_fn=lambda config: prompted.append(config),
    )

    status = await service.verify_connection(account_id="U123456")

    assert checked and validated and prompted
    assert status.account_id == "U123456"
    assert status.visible_accounts == ["U123456", "U654321"]
    assert status.key_info == {"signature_key": "ok", "encryption_key": "ok"}
    assert status.portfolio_summary.portfolio_value_usd == 5000.0
    assert status.raw_position_count == 2


@pytest.mark.asyncio
async def test_fetch_account_ids_uses_read_only_connection():
    service = IbkrAccountService(config=FakeConfig(), client_cls=FakeClient)
    assert await service.fetch_account_ids() == ["U123456", "U654321"]


@pytest.mark.asyncio
async def test_fetch_ledger_uses_configured_account():
    service = IbkrAccountService(config=FakeConfig(), client_cls=FakeClient)
    ledger = await service.fetch_ledger()
    assert ledger["BASE"]["cashbalance"] == 1000.0
