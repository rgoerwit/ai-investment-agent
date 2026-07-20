from __future__ import annotations

from pathlib import Path

from flask import Blueprint, render_template

views_bp = Blueprint("ibkr_dashboard_views", __name__)

DASHBOARD_SCRIPTS = (
    "dashboard_state.js",
    "dashboard_render_shared.js",
    "dashboard_render_portfolio.js",
    "dashboard_render_operations.js",
    "dashboard_render_detail.js",
    "dashboard.js",
)


def _asset_version(filename: str) -> str:
    asset_path = Path(__file__).with_name("static") / filename
    return str(int(asset_path.stat().st_mtime))


def _asset_bundle_version(*filenames: str) -> str:
    return str(max(int(_asset_version(filename)) for filename in filenames))


@views_bp.get("/")
def index():
    return render_template(
        "index.html",
        dashboard_css_version=_asset_version("dashboard.css"),
        dashboard_js_version=_asset_bundle_version(*DASHBOARD_SCRIPTS),
        dashboard_scripts=DASHBOARD_SCRIPTS,
    )
