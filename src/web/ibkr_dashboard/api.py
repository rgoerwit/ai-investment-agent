from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from flask import Blueprint, current_app, jsonify, request

from src.web.ibkr_dashboard.drilldown_service import (
    DrilldownLoadError,
    find_markdown_artifacts,
    find_reconciliation_item,
    load_analysis_json,
    render_markdown_file,
)
from src.web.ibkr_dashboard.job_store import RefreshJobRequest
from src.web.ibkr_dashboard.serializers import (
    serialize_dashboard_snapshot,
    serialize_equity_drilldown,
)

api_bp = Blueprint("ibkr_dashboard_api", __name__, url_prefix="/api")


def _snapshot_service():
    return current_app.config["SNAPSHOT_SERVICE"]


def _job_store():
    return current_app.config["JOB_STORE"]


def _settings():
    return current_app.config["DASHBOARD_SETTINGS"]


def _macro_alert_service():
    return current_app.config["MACRO_ALERT_SERVICE"]


def _preferences_store():
    return current_app.config["PREFERENCES_STORE"]


def _preferences():
    return _snapshot_service().current_preferences()


def _load_bundle(*, force: bool = False):
    return _snapshot_service().load_snapshot_sync(force=force)


def _load_snapshot_or_response(*, force: bool = False):
    bundle, metadata = _load_bundle(force=force)
    metadata_payload = asdict(metadata)

    if bundle is None and metadata.status == "loading":
        return None, metadata, (jsonify(metadata_payload), 202)

    if bundle is None:
        return (
            None,
            metadata,
            (
                jsonify({"error": "snapshot_load_failed", **metadata_payload}),
                503,
            ),
        )

    return bundle, metadata, None


@api_bp.get("/portfolio")
def get_portfolio():
    force = request.args.get("refresh") in {"1", "true", "yes"}
    bundle, metadata, response = _load_snapshot_or_response(force=force)
    if response is not None:
        return response
    payload = serialize_dashboard_snapshot(
        bundle,
        status=metadata.status,
        fetched_at=metadata.fetched_at,
        cache_hit=metadata.cache_hit,
        refreshing=metadata.refreshing,
        load_error=metadata.last_error,
        macro_alert=_macro_alert_service().build_alert(bundle.health_flags),
        read_only=_preferences().read_only,
    )
    return jsonify(payload)


@api_bp.get("/orders")
def get_orders():
    bundle, metadata, response = _load_snapshot_or_response(force=False)
    if response is not None:
        return response
    return jsonify(
        {
            "status": metadata.status,
            "as_of": metadata.fetched_at,
            "refreshing": metadata.refreshing,
            "orders": bundle.live_orders,
        }
    )


@api_bp.get("/watchlist")
def get_watchlist():
    bundle, metadata, response = _load_snapshot_or_response(force=False)
    if response is not None:
        return response
    watchlist_items = [
        {
            "ticker_yf": item.ticker.yf,
            "ticker_ibkr": item.ticker.ibkr,
            "action": item.action,
            "reason": item.reason,
        }
        for item in bundle.items
        if item.is_watchlist
    ]
    return jsonify(
        {
            "status": metadata.status,
            "as_of": metadata.fetched_at,
            "refreshing": metadata.refreshing,
            "name": bundle.watchlist_name,
            "total": bundle.watchlist_total,
            "tickers": sorted(bundle.watchlist_tickers),
            "items": watchlist_items,
        }
    )


@api_bp.get("/equities/<ticker>")
def get_equity_drilldown(ticker: str):
    bundle, metadata, response = _load_snapshot_or_response(force=False)
    if response is not None:
        return response
    item = find_reconciliation_item(bundle.items, ticker)
    if item is None:
        return jsonify(
            {"error": "not_found", "message": f"Unknown ticker: {ticker}"}
        ), 404

    analysis_json = None
    report_markdown_html = None
    article_markdown_html = None
    report_markdown_path = None
    article_markdown_path = None

    if item.analysis and item.analysis.file_path:
        try:
            analysis_path = Path(item.analysis.file_path)
            analysis_json = load_analysis_json(analysis_path)
            artifacts = find_markdown_artifacts(item.analysis)
            report_markdown_path = artifacts["report_markdown_path"]
            article_markdown_path = artifacts["article_markdown_path"]
            if report_markdown_path:
                report_markdown_html = render_markdown_file(Path(report_markdown_path))
            if article_markdown_path:
                article_markdown_html = render_markdown_file(
                    Path(article_markdown_path)
                )
        except DrilldownLoadError as exc:
            return (
                jsonify(
                    {
                        "error": "drilldown_load_failed",
                        "message": str(exc),
                    }
                ),
                500,
            )

    payload = serialize_equity_drilldown(
        item,
        live_orders=bundle.live_orders,
        analysis_json=analysis_json,
        report_markdown_html=report_markdown_html,
        report_markdown_path=report_markdown_path,
        article_markdown_html=article_markdown_html,
        article_markdown_path=article_markdown_path,
    )
    payload["status"] = metadata.status
    payload["as_of"] = metadata.fetched_at
    payload["refreshing"] = metadata.refreshing
    return jsonify(payload)


@api_bp.get("/refresh/jobs")
def list_refresh_jobs():
    return jsonify({"jobs": _job_store().list_jobs()})


@api_bp.get("/refresh/jobs/<job_id>")
def get_refresh_job(job_id: str):
    job = _job_store().get_job(job_id)
    if job is None:
        return jsonify({"error": "not_found", "message": f"Unknown job: {job_id}"}), 404
    return jsonify(job)


@api_bp.post("/refresh/jobs")
def create_refresh_job():
    payload = request.get_json(silent=True) or {}
    scope = str(payload.get("scope") or "").strip()
    if scope not in {"stale_positions", "due_soon", "ticker_list"}:
        return (
            jsonify(
                {
                    "error": "invalid_request",
                    "message": "scope must be stale_positions, due_soon, or ticker_list",
                }
            ),
            400,
        )

    quick_mode = bool(payload.get("quick_mode", True))
    preferences = _preferences()
    refresh_limit = int(payload.get("refresh_limit") or preferences.refresh_limit)
    max_age_days = int(payload.get("max_age_days") or preferences.max_age_days)
    watchlist_name = payload.get("watchlist_name") or preferences.watchlist_name
    try:
        tickers = _resolve_refresh_tickers(scope, payload)
    except ValueError as exc:
        return jsonify({"error": "invalid_request", "message": str(exc)}), 400
    except RuntimeError as exc:
        return jsonify({"error": "snapshot_required", "message": str(exc)}), 409
    if scope == "ticker_list" and not tickers:
        return (
            jsonify(
                {
                    "error": "invalid_request",
                    "message": "tickers must contain at least one ticker for ticker_list jobs",
                }
            ),
            400,
        )
    if scope != "ticker_list" and not tickers:
        return (
            jsonify(
                {
                    "error": "no_refresh_candidates",
                    "message": (
                        "No eligible tickers matched that refresh scope in the current data."
                    ),
                }
            ),
            409,
        )

    request_row = RefreshJobRequest(
        scope=scope,
        tickers=tickers,
        results_dir=str(_settings().results_dir),
        watchlist_name=watchlist_name,
        quick_mode=quick_mode,
        refresh_limit=refresh_limit,
        max_age_days=max_age_days,
    )
    job_id = _job_store().enqueue(request_row)
    return jsonify({"accepted": True, "job_id": job_id, "tickers": list(tickers)}), 202


@api_bp.get("/settings")
def get_settings():
    preferences = _preferences()
    return jsonify(preferences.model_dump())


@api_bp.post("/settings")
def save_settings():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return (
            jsonify(
                {
                    "error": "invalid_request",
                    "message": "settings payload must be a JSON object",
                }
            ),
            400,
        )
    preferences = _preferences_store().save(
        payload,
        _settings(),
        base_preferences=_preferences(),
    )
    snapshot_reload_required = bool(_snapshot_service().apply_preferences(preferences))
    return jsonify(
        {
            **preferences.model_dump(),
            "saved": True,
            "snapshot_reload_required": snapshot_reload_required,
            "note": (
                "Settings saved. Data-affecting changes apply to the next reload."
                if snapshot_reload_required
                else "Settings saved. The current data already matches these settings."
            ),
        }
    )


def _resolve_refresh_tickers(scope: str, payload: dict[str, Any]) -> tuple[str, ...]:
    if scope == "ticker_list":
        raw_tickers = payload.get("tickers") or []
        if not isinstance(raw_tickers, list):
            raise ValueError("tickers must be a list for ticker_list jobs")
        return tuple(
            str(ticker).strip() for ticker in raw_tickers if str(ticker).strip()
        )

    bundle = _snapshot_service().get_cached_snapshot()
    if bundle is None:
        raise RuntimeError(
            "Load the portfolio data first before creating a scope-based refresh job."
        )

    freshness = bundle.freshness_summary
    if scope == "stale_positions":
        ordered = [row.run_ticker for row in freshness.blocking_now]
        ordered.extend(row.run_ticker for row in freshness.stale_in_queue)
        return tuple(dict.fromkeys(ordered))
    if scope == "due_soon":
        return tuple(dict.fromkeys(row.run_ticker for row in freshness.due_soon))
    return ()
