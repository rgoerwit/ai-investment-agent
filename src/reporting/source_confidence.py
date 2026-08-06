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

import re
from dataclasses import dataclass

import structlog

from src.agents.fundamentals_reconciler import extract_raw_metrics_payload
from src.data_block_utils import extract_data_block_field
from src.reporting.state_access import (
    get_apac_regional_report,
    get_auditor_report,
    get_consultant_review,
    get_fundamentals_report,
    get_raw_fundamentals_data,
)
from src.validators.financial_rules import parse_ocf_amount

logger = structlog.get_logger(__name__)


SourceRow = tuple[str, str, str]
_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


@dataclass(frozen=True)
class MetricDisplaySource:
    """Provenance for a rendered DATA_BLOCK metric.

    ``selected_key`` is the raw metric key chosen for display; ``derivation_label``
    says how that key was derived; ``provider_source`` says which provider won
    the merge. These are separate axes and should not be collapsed.
    """

    display_field: str
    selected_key: str
    derivation_label: str | None
    provider_source: str | None


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


def _safe_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, str | int | float):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _relative_match(left: float, right: float) -> bool:
    if left == right:
        return True
    baseline = max(abs(left), abs(right), 1.0)
    return abs(left - right) / baseline <= 0.02


def _select_amount_key(
    raw: dict,
    display_value: str | None,
    candidates: tuple[str, ...],
) -> str | None:
    parsed = parse_ocf_amount(display_value)
    if parsed is not None:
        for key in candidates:
            candidate = _safe_float(raw.get(key))
            if candidate is not None and _relative_match(parsed, candidate):
                return key
    for key in candidates:
        if raw.get(key) is not None:
            return key
    return None


def _select_pe_key(raw: dict, display_value: str | None) -> str | None:
    match = _NUMBER_RE.search(display_value or "")
    displayed = _safe_float(match.group(0).replace(",", "")) if match else None
    trailing_pe = _safe_float(raw.get("trailingPE"))
    if trailing_pe is not None and (
        displayed is None or _relative_match(displayed, trailing_pe)
    ):
        return "trailingPE"

    market_cap = _safe_float(raw.get("marketCap"))
    net_income = _safe_float(raw.get("netIncomeToCommon"))
    if market_cap is not None and net_income is not None and net_income != 0:
        calculated = market_cap / net_income
        if displayed is None or _relative_match(displayed, calculated):
            return "marketCap/netIncomeToCommon"
    return None


def _metric_source(
    raw: dict,
    display_field: str,
    selected_key: str | None,
) -> MetricDisplaySource | None:
    if not selected_key:
        return None
    field_sources = raw.get("_field_sources")
    field_sources = field_sources if isinstance(field_sources, dict) else {}
    if "/" in selected_key:
        parts = selected_key.split("/")
        providers = sorted(
            {str(field_sources.get(part)) for part in parts if field_sources.get(part)}
        )
        return MetricDisplaySource(
            display_field=display_field,
            selected_key=selected_key,
            derivation_label="calculated_from_market_cap_net_income",
            provider_source="+".join(providers) if providers else None,
        )
    return MetricDisplaySource(
        display_field=display_field,
        selected_key=selected_key,
        derivation_label=raw.get(f"_{selected_key}_source"),
        provider_source=field_sources.get(selected_key),
    )


def _metric_display_sources(
    state: dict, fundamentals: str
) -> list[MetricDisplaySource]:
    raw = extract_raw_metrics_payload(get_raw_fundamentals_data(state))
    if not raw:
        return []
    rows: list[MetricDisplaySource] = []
    selections = (
        (
            "OPERATING_CASH_FLOW",
            _select_amount_key(
                raw,
                extract_data_block_field(fundamentals, "OPERATING_CASH_FLOW"),
                ("operatingCashflow_TTM", "operatingCashflow"),
            ),
        ),
        (
            "FREE_CASH_FLOW",
            _select_amount_key(
                raw,
                extract_data_block_field(fundamentals, "FREE_CASH_FLOW"),
                ("freeCashflow_TTM", "freeCashflow"),
            ),
        ),
        (
            "PE_RATIO_TTM",
            _select_pe_key(raw, extract_data_block_field(fundamentals, "PE_RATIO_TTM")),
        ),
    )
    for display_field, selected_key in selections:
        source = _metric_source(raw, display_field, selected_key)
        if source is not None:
            rows.append(source)
    return rows


def _format_metric_source(source: MetricDisplaySource) -> str:
    details = [f"selected {source.selected_key}"]
    if source.derivation_label:
        details.append(f"derivation {source.derivation_label}")
    else:
        details.append("derivation unavailable")
    if source.provider_source:
        details.append(f"provider {source.provider_source}")
    else:
        details.append("provider unavailable")
    return "; ".join(details)


def _quarterly_diagnostics(raw: dict) -> str | None:
    diagnostics = raw.get("_quarterly_diagnostics")
    if not isinstance(diagnostics, list):
        return None
    parts: list[str] = []
    for item in diagnostics:
        if not isinstance(item, dict):
            continue
        field = item.get("field")
        reason = item.get("reason")
        if field and reason:
            parts.append(f"{field}: {reason}")
        if len(parts) >= 4:
            break
    if not parts:
        return None
    suffix = "…" if len(diagnostics) > len(parts) else ""
    return "; ".join(parts) + suffix


def build_source_confidence_rows(state: dict) -> list[SourceRow]:
    """Compose the source-confidence rows for the memo."""
    rows: list[SourceRow] = []

    fundamentals = get_fundamentals_report(state)
    ocf_source = (
        (extract_data_block_field(fundamentals, "OPERATING_CASH_FLOW_SOURCE") or "")
        .strip()
        .upper()
    )
    ocf_reason = (
        (extract_data_block_field(fundamentals, "OCF_FILING_REASON") or "")
        .strip()
        .upper()
    )
    if ocf_source == "FILING" and ocf_reason == "DISCREPANCY":
        # Filing and aggregator OCF materially diverged: not "ground truth".
        # Verify against the actual cash-flow statement line (KTY.WA 2026-06-27).
        rows.append(
            (
                "Core financials",
                "Filing/API OCF conflict — verify statement line",
                "MEDIUM",
            )
        )
    elif ocf_source == "FILING":
        rows.append(("Core financials", "Filing-level OCF (ground truth)", "HIGH"))
    elif ocf_source in {"JUNIOR", "AGGREGATOR"}:
        rows.append(("Core financials", "Aggregator (yfinance / yahooquery)", "MEDIUM"))
    elif fundamentals:
        rows.append(("Core financials", "Aggregator (source unspecified)", "MEDIUM"))
    else:
        rows.append(("Core financials", "Not available", "LOW"))

    for source in _metric_display_sources(state, fundamentals):
        rows.append(
            (
                f"Metric provenance: {source.display_field}",
                _format_metric_source(source),
                "MEDIUM",
            )
        )

    raw_payload = extract_raw_metrics_payload(get_raw_fundamentals_data(state))
    quarterly_note = _quarterly_diagnostics(raw_payload)
    if quarterly_note:
        rows.append(("Quarterly/TTM diagnostics", quarterly_note, "MEDIUM"))

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
    # `consultant_successful` only means the consultant returned a parseable
    # review — not that it approved. Branch on the derived verdict; fall back to
    # the legacy "ran ok → HIGH" path only for pre-change saved JSON (verdict None).
    verdict = summary.get("consultant_verdict")
    if verdict == "CLEAN" or (verdict is None and summary.get("consultant_successful")):
        rows.append(
            (
                "Cross-model review",
                "Consultant — no material concerns in bounded review",
                "HIGH",
            )
        )
    elif verdict == "CONDITIONAL":
        rows.append(("Cross-model review", "Consultant — conditional", "MEDIUM"))
    elif verdict == "MAJOR_CONCERNS":
        rows.append(("Cross-model review", "Consultant — major concerns", "LOW"))
    elif verdict == "REJECTED":
        rows.append(("Cross-model review", "Consultant — not approved", "LOW"))
    elif verdict == "ERROR":
        rows.append(
            ("Cross-model review", "Consultant review failed validation", "LOW")
        )
    elif verdict == "SKIPPED":
        rows.append(("Cross-model review", "Consultant bypassed (quick screen)", "—"))
    elif verdict == "UNPARSED":
        rows.append(("Cross-model review", "Consultant review unparsed", "LOW"))
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
