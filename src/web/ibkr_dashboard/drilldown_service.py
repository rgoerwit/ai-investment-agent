from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import nh3
from markdown import markdown

from src.ibkr.models import AnalysisRecord, ReconciliationItem

_ALLOWED_TAGS = {
    "a",
    "blockquote",
    "br",
    "code",
    "em",
    "h1",
    "h2",
    "h3",
    "hr",
    "li",
    "ol",
    "p",
    "pre",
    "strong",
    "table",
    "tbody",
    "td",
    "th",
    "thead",
    "tr",
    "ul",
}
_ALLOWED_ATTRIBUTES = {
    "a": {"href", "title"},
    "th": {"colspan", "rowspan"},
    "td": {"colspan", "rowspan"},
}


class DrilldownLoadError(RuntimeError):
    """Raised when a saved analysis artifact cannot be loaded safely."""


def find_reconciliation_item(
    items: list[ReconciliationItem],
    ticker: str,
) -> ReconciliationItem | None:
    ticker_upper = ticker.upper()
    for item in items:
        if (
            item.ticker.yf.upper() == ticker_upper
            or item.ticker.ibkr.upper() == ticker_upper
        ):
            return item
    return None


def load_analysis_json(path: Path) -> dict[str, Any] | None:
    if not path.exists() or path.suffix.lower() != ".json":
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DrilldownLoadError(f"Malformed analysis JSON at {path}") from exc


def render_markdown_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    raw = path.read_text(encoding="utf-8")
    html = markdown(raw, extensions=["tables", "fenced_code"])
    return nh3.clean(
        html,
        tags=_ALLOWED_TAGS,
        attributes=_ALLOWED_ATTRIBUTES,
        url_schemes={"http", "https"},
    )


def find_markdown_artifacts(analysis: AnalysisRecord) -> dict[str, str | None]:
    if not analysis.file_path:
        return {"report_markdown_path": None, "article_markdown_path": None}
    source_path = Path(analysis.file_path)
    results_dir = source_path.parent

    report_path = (
        source_path.with_suffix(".md") if source_path.suffix == ".json" else None
    )
    if report_path is not None and not report_path.exists():
        report_path = None

    article_path = results_dir / f"{analysis.ticker}_article.md"
    if not article_path.exists():
        article_path = None

    return {
        "report_markdown_path": str(report_path) if report_path else None,
        "article_markdown_path": str(article_path) if article_path else None,
    }


def build_structured_sections(analysis_json: dict[str, Any] | None) -> dict[str, Any]:
    if analysis_json is None:
        return {}
    return {
        "prediction_snapshot": analysis_json.get("prediction_snapshot"),
        "final_decision": analysis_json.get("final_decision"),
        "investment_analysis": analysis_json.get("investment_analysis"),
        "risk_analysis": analysis_json.get("risk_analysis"),
        "reports": analysis_json.get("reports"),
        "artifact_statuses": analysis_json.get("artifact_statuses"),
        "analysis_validity": analysis_json.get("analysis_validity"),
    }
