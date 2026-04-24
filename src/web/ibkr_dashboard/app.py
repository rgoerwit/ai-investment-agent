from __future__ import annotations

import argparse
from typing import Any

from flask import Flask

from src.config import config
from src.ibkr.cli_options import (
    add_common_portfolio_request_args,
    dashboard_preferences_overrides_from_args,
    dashboard_settings_overrides_from_args,
    validate_common_portfolio_request_args,
)
from src.runtime_services import (
    build_provider_runtime,
    build_runtime_services_from_config,
)
from src.web.ibkr_dashboard.api import api_bp
from src.web.ibkr_dashboard.job_store import RefreshJobStore
from src.web.ibkr_dashboard.macro_alerts import MacroAlertService
from src.web.ibkr_dashboard.settings import (
    DashboardPreferencesStore,
    DashboardSettings,
)
from src.web.ibkr_dashboard.snapshot_service import DashboardSnapshotService
from src.web.ibkr_dashboard.views import views_bp


def create_app(
    settings: DashboardSettings | None = None,
    *,
    preferences_override: dict[str, Any] | None = None,
) -> Flask:
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    resolved_settings = settings or DashboardSettings()
    resolved_settings.runtime_dir.mkdir(parents=True, exist_ok=True)
    provider_runtime = build_provider_runtime(explicit=True)
    runtime_services = build_runtime_services_from_config(
        config,
        enable_tool_audit=False,
        provider_runtime=provider_runtime,
    )
    preferences_store = DashboardPreferencesStore(
        resolved_settings.runtime_dir / "settings.json"
    )
    preferences = preferences_store.load(resolved_settings)
    if preferences_override:
        preferences = preferences.model_copy(update=preferences_override)

    app.config["DASHBOARD_SETTINGS"] = resolved_settings
    app.config["PROVIDER_RUNTIME"] = provider_runtime
    app.config["RUNTIME_SERVICES"] = runtime_services
    app.config["SNAPSHOT_SERVICE"] = DashboardSnapshotService(
        resolved_settings,
        preferences=preferences,
        runtime_services=runtime_services,
    )
    app.config["JOB_STORE"] = RefreshJobStore(
        resolved_settings.runtime_dir / "jobs.sqlite"
    )
    app.config["MACRO_ALERT_SERVICE"] = MacroAlertService()
    app.config["PREFERENCES_STORE"] = preferences_store
    app.json.sort_keys = False

    app.register_blueprint(api_bp)
    app.register_blueprint(views_bp)
    return app


def main() -> None:
    settings = DashboardSettings()
    parser = argparse.ArgumentParser(description="IBKR dashboard")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    add_common_portfolio_request_args(
        parser,
        mode_flag_style="dual",
        results_dir_default=None,
        max_age_default=None,
        cash_buffer_default=None,
        drift_pct_default=None,
        refresh_limit_default=None,
        sector_limit_default=None,
        exchange_limit_default=None,
        read_only_default=None,
        read_only_help="Use saved analysis results only and skip live IBKR portfolio data.",
        live_help="Use live IBKR portfolio data for this dashboard session.",
        account_id_help="Override the IBKR account ID for this dashboard session.",
        results_dir_help="Override the results directory for this dashboard session.",
        watchlist_help=(
            "Name of the IBKR watchlist to evaluate (case-insensitive substring match)."
        ),
    )
    args = parser.parse_args()
    validate_common_portfolio_request_args(parser, args)

    overrides = dashboard_settings_overrides_from_args(args)
    if args.host:
        overrides["host"] = args.host
    if args.port:
        overrides["port"] = args.port
    if args.debug:
        overrides["debug"] = True
    resolved_settings = settings.model_copy(update=overrides)

    preferences_override = dashboard_preferences_overrides_from_args(args)

    app = create_app(
        resolved_settings,
        preferences_override=preferences_override or None,
    )
    app.run(
        host=resolved_settings.host,
        port=resolved_settings.port,
        debug=resolved_settings.debug,
    )


if __name__ == "__main__":
    main()
