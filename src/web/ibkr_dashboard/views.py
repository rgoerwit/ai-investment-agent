from __future__ import annotations

from pathlib import Path

from flask import Blueprint, render_template

views_bp = Blueprint("ibkr_dashboard_views", __name__)


def _asset_version(filename: str) -> str:
    asset_path = Path(__file__).with_name("static") / filename
    return str(int(asset_path.stat().st_mtime))


@views_bp.get("/")
def index():
    return render_template(
        "index.html",
        dashboard_css_version=_asset_version("dashboard.css"),
        dashboard_js_version=_asset_version("dashboard.js"),
    )
