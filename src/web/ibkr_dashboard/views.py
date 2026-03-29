from __future__ import annotations

from flask import Blueprint, render_template

views_bp = Blueprint("ibkr_dashboard_views", __name__)


@views_bp.get("/")
def index():
    return render_template("index.html")
