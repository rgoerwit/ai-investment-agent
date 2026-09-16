"""Render the canonical claims and evidence constraints used by a decision."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.reporting.state_access import get_effective_red_flags
from src.tooling.evidence_recorder import evidence_record_id, normalize_http_url


def _cell(value: object, *, limit: int = 80) -> str:
    text = " ".join(str(value or "N/A").split()).replace("|", "\\|")
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _record_field(record: Any, field: str, default: Any = None) -> Any:
    return (
        record.get(field, default)
        if isinstance(record, Mapping)
        else getattr(record, field, default)
    )


def _inspected_source(source: Mapping[str, Any], claim: Mapping[str, Any]) -> str:
    if not claim.get("decision_eligible"):
        return "—"
    evidence_id = str(claim.get("evidence_id") or "")
    url = normalize_http_url(str(claim.get("source_url") or ""))
    if not evidence_id.startswith("evidence:") or not url:
        return "—"
    records = source.get("evidence_records")
    if not isinstance(records, list | tuple):
        return "—"
    matched = False
    for record in records:
        if evidence_record_id(record) != evidence_id:
            continue
        if (
            bool(_record_field(record, "blocked", False))
            or _record_field(record, "execution_status") != "SUCCEEDED"
            or _record_field(record, "evidence_status") != "EVIDENCE_FOUND"
        ):
            continue
        known_urls = {
            normalized
            for field in ("requested_urls", "urls")
            for value in (_record_field(record, field, ()) or ())
            if (normalized := normalize_http_url(str(value)))
        }
        if url in known_urls:
            matched = True
            break
    if not matched:
        return "—"
    safe_url = url.replace(")", "%29")
    return f"[inspected source]({safe_url})"


def render_decision_evidence_markdown(source: Any) -> str:
    """Render decision-trace facts and BUY-blocking evidence gaps."""
    if not isinstance(source, Mapping):
        return ""
    snapshot = source.get("analysis_snapshot")
    trace = source.get("decision_trace")
    if not isinstance(snapshot, Mapping) or not isinstance(trace, Mapping):
        return ""
    claims = snapshot.get("claims")
    claims = claims if isinstance(claims, Mapping) else {}
    raw_ids = trace.get("decision_facts")
    fact_ids = raw_ids if isinstance(raw_ids, list | tuple) else []

    rows: list[str] = []
    external_count = 0
    for claim_id in fact_ids[:20]:
        claim = claims.get(str(claim_id))
        if not isinstance(claim, Mapping):
            rows.append(
                f"| `{_cell(claim_id, limit=60)}` | Missing canonical claim | — | — | — |"
            )
            continue
        source_cell = _inspected_source(source, claim)
        if source_cell != "—":
            external_count += 1
        rows.append(
            "| "
            + " | ".join(
                (
                    _cell(claim.get("field")),
                    _cell(claim.get("value")),
                    _cell(claim.get("authority")),
                    _cell(claim.get("coverage")),
                    source_cell,
                )
            )
            + " |"
        )

    parts = ["## Decision Evidence\n\n"]
    if rows:
        parts.append("| Claim | Value | Authority | Coverage | Source |\n")
        parts.append("|---|---:|---|---|---|\n")
        parts.extend(f"{row}\n" for row in rows)
        parts.append("\n")
    else:
        parts.append("The decision trace references no canonical decision facts.\n\n")
    run_summary = source.get("run_summary")
    is_quick = bool(
        isinstance(run_summary, Mapping) and run_summary.get("quick_mode") is True
    )
    if external_count == 0 and is_quick:
        parts.append(
            "*Quick screening intentionally disables external-document extraction; "
            "the screen relies on aggregator and code-owned lineage and must be "
            "promoted to full analysis before action.*\n\n"
        )
    elif external_count == 0:
        parts.append(
            "*No decision fact is backed by an inspected external document; any "
            "facts above rely on aggregator or code-owned lineage. Specialist and "
            "consultant narrative outside this table is unverified review context, "
            "not external verification, and may only qualify the decision "
            "conservatively.*\n\n"
        )
    else:
        parts.append(
            f"*{external_count} decision fact(s) are bound to inspected external "
            "documents.*\n\n"
        )

    blockers = [
        flag
        for flag in get_effective_red_flags(source)
        if isinstance(flag, Mapping) and flag.get("blocks_buy") is True
    ]
    if blockers:
        parts.append("**Decision constraints.**\n\n")
        for flag in blockers[:10]:
            parts.append(
                f"- `{_cell(flag.get('type'), limit=60)}`: "
                f"{_cell(flag.get('detail'), limit=180)}\n"
            )
        parts.append("\n")
    parts.append("---\n\n")
    return "".join(parts)
