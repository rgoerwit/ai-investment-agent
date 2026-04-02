from __future__ import annotations

from src.web.ibkr_dashboard.app import create_app
from src.web.ibkr_dashboard.settings import DashboardSettings


def test_create_app_registers_dashboard_services(tmp_path):
    settings = DashboardSettings(runtime_dir=tmp_path / "runtime")
    app = create_app(settings)
    assert "SNAPSHOT_SERVICE" in app.config
    assert "JOB_STORE" in app.config
    assert "MACRO_ALERT_SERVICE" in app.config
    assert app.config["JOB_STORE"].list_jobs() == []
    assert app.config["DASHBOARD_SETTINGS"].read_only is False
    assert app.config["SEND_FILE_MAX_AGE_DEFAULT"] == 0


def test_create_app_applies_startup_preference_overrides(tmp_path):
    settings = DashboardSettings(runtime_dir=tmp_path / "runtime")
    app = create_app(
        settings,
        preferences_override={
            "account_id": "U20958465",
            "watchlist_name": "default watchlist",
            "read_only": True,
        },
    )

    preferences = app.config["SNAPSHOT_SERVICE"].current_preferences()
    assert preferences.account_id == "U20958465"
    assert preferences.watchlist_name == "default watchlist"
    assert preferences.read_only is True
