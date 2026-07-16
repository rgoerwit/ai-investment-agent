from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

from src.ibkr.portfolio_defaults import (
    DEFAULT_CASH_BUFFER_PCT,
    DEFAULT_DRIFT_PCT,
    DEFAULT_EXCHANGE_LIMIT_PCT,
    DEFAULT_MAX_AGE_DAYS,
    DEFAULT_REFRESH_LIMIT,
    DEFAULT_SECTOR_LIMIT_PCT,
)


class DashboardSettings(BaseSettings):
    results_dir: Path = Path("results")
    account_id: str | None = None
    watchlist_name: str | None = None

    # Safe default: disk-only mode. Live IBKR portfolio access is opt-in via
    # IBKR_DASHBOARD_READ_ONLY=false or saved preferences.
    read_only: bool = True
    cash_buffer: float = DEFAULT_CASH_BUFFER_PCT
    max_age_days: int = DEFAULT_MAX_AGE_DAYS
    drift_pct: float = DEFAULT_DRIFT_PCT
    sector_limit_pct: float = DEFAULT_SECTOR_LIMIT_PCT
    exchange_limit_pct: float = DEFAULT_EXCHANGE_LIMIT_PCT

    host: str = "127.0.0.1"
    port: int = 5050
    debug: bool = False

    # Live IBKR snapshots can take about a minute on first load when positions,
    # watchlists, and live orders all need brokerage-session calls and retries.
    snapshot_timeout_seconds: int = 180
    default_refresh_limit: int = DEFAULT_REFRESH_LIMIT
    runtime_dir: Path = Path("runtime") / "ibkr_dashboard"

    model_config = {
        "env_prefix": "IBKR_DASHBOARD_",
        "env_file": ".env",
        "extra": "ignore",
    }


class DashboardPreferences(BaseModel):
    account_id: str | None = None
    read_only: bool = True
    watchlist_name: str | None = None
    max_age_days: int = DEFAULT_MAX_AGE_DAYS
    quick_mode_default: bool = True
    refresh_limit: int = DEFAULT_REFRESH_LIMIT
    risk_thresholds: dict[str, float] = Field(default_factory=dict)
    notes: str = ""


class DashboardPreferencesStore:
    def __init__(self, path: Path) -> None:
        self._path = path

    def load(self, defaults: DashboardSettings) -> DashboardPreferences:
        payload: dict[str, Any] = {
            "account_id": defaults.account_id,
            "read_only": defaults.read_only,
            "watchlist_name": defaults.watchlist_name,
            "max_age_days": defaults.max_age_days,
            "quick_mode_default": True,
            "refresh_limit": defaults.default_refresh_limit,
            "risk_thresholds": {},
            "notes": "",
        }
        if self._path.exists():
            payload.update(json.loads(self._path.read_text(encoding="utf-8")))
        return DashboardPreferences.model_validate(payload)

    def save(
        self,
        data: dict[str, Any],
        defaults: DashboardSettings,
        *,
        base_preferences: DashboardPreferences | None = None,
    ) -> DashboardPreferences:
        current = (
            base_preferences.model_dump()
            if base_preferences is not None
            else self.load(defaults).model_dump()
        )
        current.update(data)
        preferences = DashboardPreferences.model_validate(current)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(preferences.model_dump(), indent=2),
            encoding="utf-8",
        )
        return preferences
