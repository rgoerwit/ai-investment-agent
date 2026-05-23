"""Source-confidence / data-quality rows for the investment memo.

The table answers "where did each load-bearing claim come from?" without
asking the reader to dig through the appendix. It is rendered as a small
markdown table inside the InvestmentMemo (see `memo.py`).

Each row is a tuple ``(claim, source, confidence)``:

- ``claim`` — what the row covers (Core financials, Forensic check, …).
- ``source`` — agent or feed that produced the claim, or ``"Not run"``.
- ``confidence`` — ``HIGH`` / ``MEDIUM`` / ``LOW`` / ``—``.

The builder accepts either the live AgentState or the saved analysis JSON
shape, so quality-judge tooling that walks ``results/*.json`` can call it.
"""

from __future__ import annotations

import structlog

from src.data_block_utils import extract_data_block_field
from src.reporting.state_access import (
    get_apac_regional_report,
    get_auditor_report,
    get_consultant_review,
    get_fundamentals_report,
)

logger = structlog.get_logger(__name__)


SourceRow = tuple[str, str, str]


def _run_summary(state: dict) -> dict:
    summary = state.get("run_summary")
    return summary if isinstance(summary, dict) else {}


def _apac_status(state: dict) -> str | None:
    """Return APAC verdict tag (SUPPORT / CAUTION / OVERRIDE) when discoverable."""
    apac = get_apac_regional_report(state)
    if not apac:
        return None
    if "NO_MATERIAL_APAC_CONNECTION" in apac:
        return "SILENT"
    if "APAC_SPECIALIST_UNAVAILABLE" in apac:
        return "UNAVAILABLE"
    for tag in ("OVERRIDE", "CAUTION", "SUPPORT"):
        if tag in apac:
            return tag
    return "RAN"


def build_source_confidence_rows(state: dict) -> list[SourceRow]:
    """Compose the source-confidence rows for the memo."""
    rows: list[SourceRow] = []

    fundamentals = get_fundamentals_report(state)
    ocf_source = (
        (extract_data_block_field(fundamentals, "OPERATING_CASH_FLOW_SOURCE") or "")
        .strip()
        .upper()
    )
    if ocf_source == "FILING":
        rows.append(("Core financials", "Filing-level OCF (ground truth)", "HIGH"))
    elif ocf_source in {"JUNIOR", "AGGREGATOR"}:
        rows.append(("Core financials", "Aggregator (yfinance / yahooquery)", "MEDIUM"))
    elif fundamentals:
        rows.append(("Core financials", "Aggregator (source unspecified)", "MEDIUM"))
    else:
        rows.append(("Core financials", "Not available", "LOW"))

    summary = _run_summary(state)
    auditor_ran = bool(summary.get("auditor_completed")) or bool(
        get_auditor_report(state)
    )
    auditor_clean = bool(summary.get("auditor_successful"))
    if auditor_clean:
        rows.append(("Forensic check", "Auditor (gpt-5.4-mini)", "HIGH"))
    elif auditor_ran:
        rows.append(("Forensic check", "Auditor ran with caveats", "MEDIUM"))
    else:
        rows.append(("Forensic check", "Not run", "—"))

    consultant_ran = (
        bool(summary.get("consultant_completed"))
        or bool(summary.get("consultant_finished"))
        or bool(get_consultant_review(state))
    )
    consultant_ok = bool(summary.get("consultant_successful"))
    if consultant_ok:
        rows.append(("Cross-model review", "Consultant (gpt-5.4)", "HIGH"))
    elif consultant_ran:
        rows.append(
            ("Cross-model review", "Consultant ran with reservations", "MEDIUM")
        )
    else:
        rows.append(("Cross-model review", "Not run", "—"))

    apac_status = _apac_status(state)
    if apac_status in {"SUPPORT", "CAUTION", "OVERRIDE", "RAN"}:
        rows.append(("Regional context", f"APAC Specialist ({apac_status})", "MEDIUM"))
    elif apac_status == "SILENT":
        rows.append(
            ("Regional context", "APAC Specialist (no material APAC link)", "—")
        )
    elif apac_status == "UNAVAILABLE":
        rows.append(("Regional context", "APAC Specialist unavailable", "—"))
    else:
        rows.append(("Regional context", "Not applicable", "—"))

    return rows


def render_source_confidence_markdown(rows: list[SourceRow]) -> str:
    """Render confidence rows as a compact 3-column markdown table.

    Returns an empty string when there are no rows so the caller can skip the
    section entirely.
    """
    if not rows:
        return ""
    lines = [
        "| Claim | Source | Confidence |",
        "| --- | --- | --- |",
    ]
    for claim, source, confidence in rows:
        lines.append(f"| {claim} | {source} | {confidence} |")
    return "\n".join(lines) + "\n"
