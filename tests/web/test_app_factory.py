from __future__ import annotations

from src.web.ibkr_dashboard.app import create_app
from src.web.ibkr_dashboard.settings import DashboardPreferencesStore, DashboardSettings


def test_create_app_registers_dashboard_services(tmp_path):
    settings = DashboardSettings(runtime_dir=tmp_path / "runtime")
    app = create_app(settings)
    assert "RUNTIME_SERVICES" in app.config
    assert "PROVIDER_RUNTIME" in app.config
    assert "SNAPSHOT_SERVICE" in app.config
    assert "JOB_STORE" in app.config
    assert "MACRO_ALERT_SERVICE" in app.config
    assert app.config["JOB_STORE"].list_jobs() == []
    assert app.config["DASHBOARD_SETTINGS"].read_only is True
    assert app.config["SEND_FILE_MAX_AGE_DEFAULT"] == 0


def test_read_only_default_is_safe_and_env_overridable(tmp_path, monkeypatch):
    """Default is disk-only mode; live access is an explicit opt-in."""
    monkeypatch.delenv("IBKR_DASHBOARD_READ_ONLY", raising=False)
    assert DashboardSettings(runtime_dir=tmp_path).read_only is True

    monkeypatch.setenv("IBKR_DASHBOARD_READ_ONLY", "false")
    assert DashboardSettings(runtime_dir=tmp_path).read_only is False


def test_create_app_applies_startup_preference_overrides(tmp_path):
    settings = DashboardSettings(runtime_dir=tmp_path / "runtime")
    app = create_app(
        settings,
        preferences_override={
            "account_id": "U1234567",
            "watchlist_name": "default watchlist",
            "read_only": True,
        },
    )

    preferences = app.config["SNAPSHOT_SERVICE"].current_preferences()
    assert preferences.account_id == "U1234567"
    assert preferences.watchlist_name == "default watchlist"
    assert preferences.read_only is True


def test_create_app_preference_precedence_saved_over_defaults_startup_over_saved(
    tmp_path,
):
    """Dashboard precedence is startup args > saved prefs > env/default settings."""
    runtime_dir = tmp_path / "runtime"
    settings = DashboardSettings(
        runtime_dir=runtime_dir,
        read_only=False,
        account_id="DEFAULT",
        max_age_days=30,
        default_refresh_limit=12,
    )
    DashboardPreferencesStore(runtime_dir / "settings.json").save(
        {
            "read_only": True,
            "account_id": "SAVED",
            "max_age_days": 21,
            "refresh_limit": 9,
        },
        settings,
    )

    app = create_app(
        settings,
        preferences_override={
            "read_only": False,
            "max_age_days": 7,
        },
    )

    preferences = app.config["SNAPSHOT_SERVICE"].current_preferences()
    assert preferences.read_only is False
    assert preferences.max_age_days == 7
    assert preferences.account_id == "SAVED"
    assert preferences.refresh_limit == 9


def test_create_app_saved_preferences_override_dashboard_settings(tmp_path):
    """Without startup overrides, saved preferences beat DashboardSettings values."""
    runtime_dir = tmp_path / "runtime"
    settings = DashboardSettings(
        runtime_dir=runtime_dir,
        read_only=False,
        account_id="DEFAULT",
        max_age_days=30,
    )
    DashboardPreferencesStore(runtime_dir / "settings.json").save(
        {
            "read_only": True,
            "account_id": "SAVED",
            "max_age_days": 21,
        },
        settings,
    )

    app = create_app(settings)

    preferences = app.config["SNAPSHOT_SERVICE"].current_preferences()
    assert preferences.read_only is True
    assert preferences.account_id == "SAVED"
    assert preferences.max_age_days == 21


def test_dashboard_defaults_match_ibkr_config_defaults():
    """The dashboard duplicates three IBKR portfolio defaults; keep them in sync.

    DashboardSettings deliberately does not import IbkrSettings (process
    isolation), so this parity test is the only drift guard.
    """
    from src.ibkr_config import IbkrSettings

    dashboard_fields = DashboardSettings.model_fields
    ibkr_fields = IbkrSettings.model_fields

    pairs = [
        ("max_age_days", "ibkr_max_analysis_age_days"),
        ("drift_pct", "ibkr_drift_threshold_pct"),
        ("cash_buffer", "ibkr_cash_buffer_pct"),
    ]
    for dash_name, ibkr_name in pairs:
        assert dashboard_fields[dash_name].default == ibkr_fields[ibkr_name].default, (
            f"{dash_name} drifted from {ibkr_name}"
        )
