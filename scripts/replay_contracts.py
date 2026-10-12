"""Offline contract replay; counts describe coverage, not correctness targets.

Run with ``poetry run python -m scripts.replay_contracts --input-dir results``.
Checked expectations belong in the committed deterministic replay tests.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, get_args

from src.agents.foreign_language_evidence import (
    _CONTROL_BASIS_TERMS,
    _NON_CONTROL_RELATIONSHIPS,
    _field,
    normalize_foreign_language_evidence,
    unique_latest_results_block,
)
from src.agents.message_utils import (
    ToolEvidenceRecord,
    evidence_record_to_tool_evidence,
)
from src.data_block_utils import extract_block_field_from_text_raw
from src.monetary import exact_decimal
from src.pm_claim_audit import (
    normalize_decision_trace_citations,
    validate_decision_trace,
)
from src.score_lineage import add_validated_derivations
from src.tooling.evidence_recorder import DOCUMENT_ACQUISITION_TOOLS, normalize_http_url
from src.validators.entity_governance_card import ControlStatus
from src.validators.metric_extractor import extract_metrics


def _acquisition_state(
    evidence: Sequence[object],
    records: Sequence[ToolEvidenceRecord],
    url: str | None,
    reason: str,
    complete: bool,
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    """Classify retained execution, never infer why a source lacks binding."""

    def urls(record: dict[str, Any], field: str) -> list[str]:
        values = record.get(field, [])
        return (
            [
                normalized
                for value in values
                if isinstance(value, str) and (normalized := normalize_http_url(value))
            ]
            if isinstance(values, list)
            else []
        )

    cited = [
        r for r in evidence if isinstance(r, dict) and url and url in urls(r, "urls")
    ]
    attempts = [
        r
        for r in evidence
        if isinstance(r, dict)
        and url
        and isinstance(r.get("tool_name"), str)
        and r.get("tool_name") in DOCUMENT_ACQUISITION_TOOLS
        and url in [*urls(r, "requested_urls"), *urls(r, "urls")]
    ]
    if not url:
        status = "invalid_citation"
    elif not complete:
        status = "incomplete_found_claim"
    elif reason in {"SUPPORTED_PRIMARY", "SUPPORTED_SECONDARY"}:
        status = reason.lower()
    elif attempts:
        status = (
            "fetched_but_unbound"
            if any(
                r.get("execution_status") == "SUCCEEDED"
                and not r.get("blocked")
                and isinstance(r.get("evidence_status"), str)
                and r.get("evidence_status") in {"EVIDENCE_FOUND", "RESULTS_FOUND"}
                for r in attempts
            )
            else "attempted_failed_or_blocked"
        )
    elif any(
        bound_url == url and authority in {"PRIMARY_ISSUER", "PRIMARY_REGISTRY"}
        for record in records
        for bound_url, authority in record.authority_by_url
    ):
        status = "approved_citation_not_fetched"
    else:
        status = "citation_authority_unavailable"
    return status, attempts, cited


def _successful_evidence_records(evidence: list[object]) -> list[ToolEvidenceRecord]:
    """One retained-success filter shared by ownership and results replay."""
    return [
        evidence_record_to_tool_evidence(
            {
                **r,
                "urls": [
                    value for value in r.get("urls", []) if isinstance(value, str)
                ],
            }
        )
        for r in evidence
        if isinstance(r, dict)
        and r.get("execution_status") == "SUCCEEDED"
        and not r.get("blocked")
        and isinstance(r.get("content"), str)
        and isinstance(r.get("tool_name"), str)
        and isinstance(r.get("urls", []), list)
        and isinstance(r.get("evidence_status"), str)
        and r.get("evidence_status") in {"EVIDENCE_FOUND", "RESULTS_FOUND"}
    ]


def _replay_ownership(before: str, after: str, counts: Counter[str]) -> None:
    """Separate changes in control/basis from evidence-only normalization."""
    classifications = []
    for stage, report in (("before", before), ("after", after)):
        fields = tuple(
            _field(report, label) or "UNKNOWN"
            for label in (
                "Control Status",
                "Control Basis",
                "Ownership Evidence Status",
            )
        )
        classifications.append(fields)
        # Classification labels are finite; never print free-form report values.
        allowed = (
            set(get_args(ControlStatus)),
            {
                *_CONTROL_BASIS_TERMS,
                *_NON_CONTROL_RELATIONSHIPS.values(),
                "MAJORITY_VOTING_RIGHTS",
                "UNKNOWN",
                "CONFLICTING_CONTROL_EVIDENCE",
            },
            {
                "CITED",
                "VERIFIED_URL",
                "VERIFIED_OFFICIAL_FILING",
                "DISCLOSED_UNVERIFIED",
                "REJECTED",
                "NOT_FOUND",
                "UNKNOWN",
            },
        )
        for category, value, labels in zip(
            ("control", "basis", "evidence"), fields, allowed, strict=True
        ):
            counts[f"{stage}_{category}:{value if value in labels else 'OTHER'}"] += 1
    counts["replayed"] += 1
    if classifications[0][:2] != classifications[1][:2]:
        counts["substantive_control_changed"] += 1
    elif before != after and any(
        _field(before, label) != _field(after, label)
        for label in (
            "Largest Shareholder",
            "Controlling Shareholder",
            "Relationship",
            "Influential Entity",
            "Ownership Evidence Status",
        )
    ):
        counts["relationship_or_evidence_only_changed"] += 1


def _replay_eps(snapshot: object, report: object, counts: Counter[str]) -> None:
    """Replay evidence lineage; unsupported EPS awards are coverage, not risk."""
    if snapshot is None:
        counts["missing_snapshot"] += 1
        return
    if not isinstance(snapshot, dict) or any(
        not isinstance(snapshot.get(key, {}), dict) for key in ("claims", "scorecards")
    ):
        counts["malformed_snapshot"] += 1
        return
    if (
        snapshot.get("contract_status") != "VALID"
        or not isinstance(report, str)
        or not report
    ):
        counts["unavailable_contract_or_report"] += 1
        return
    try:
        derived = add_validated_derivations(snapshot, report)
        cards = [
            item.get("scorecards", {}).get("GROWTH", {}) for item in (snapshot, derived)
        ]
        criteria = [card.get("criteria", {}).get("EPS_GROWTH", {}) for card in cards]
        dependencies = [criterion.get("derived_from", []) for criterion in criteria]
        if any(
            not isinstance(value, list)
            or any(not isinstance(item, str) for item in value)
            for value in dependencies
        ) or any(
            criterion.get("award", "MISSING") not in {"0", "1", "N/A", "MISSING"}
            or not isinstance(card.get("lineage_gaps", []), list)
            for card, criterion in zip(cards, criteria, strict=True)
        ):
            raise ValueError("invalid lineage shape")
        for stage, card, criterion in zip(
            ("before", "after"), cards, criteria, strict=True
        ):
            counts[
                f"{stage}_growth:{'eligible' if card.get('decision_eligible') is True else 'ineligible' if card.get('decision_eligible') is False else 'unavailable'}"
            ] += 1
            counts[f"{stage}_eps_award:{criterion.get('award', 'MISSING')}"] += 1
        counts["replayed"] += 1
        if dependencies[0] != dependencies[1]:
            counts["eps_dependencies_changed"] += 1
        if (
            cards[0].get("decision_eligible") is True
            and cards[1].get("decision_eligible") is False
        ):
            counts["growth_eligibility_lost"] += 1
        if criteria[1].get("award") == "1" and "EPS_GROWTH" in cards[1].get(
            "lineage_gaps", []
        ):
            counts["unsupported_positive_eps_after"] += 1
    except (ValueError, TypeError, KeyError, AttributeError):
        counts["malformed_snapshot"] += 1


def replay_contracts(paths: Iterable[Path]) -> dict[str, dict[str, int]]:
    """Replay saved successful evidence without models, network, or file writes."""
    counts: dict[str, Counter[str]] = {
        key: Counter()
        for key in (
            "artifacts",
            "ocf",
            "latest_results",
            "latest_results_acquisition",
            "latest_results_claims",
            "pm_trace",
            "ownership",
            "eps_lineage",
        )
    }
    for path in sorted({p.resolve() for p in paths}):
        counts["artifacts"]["seen"] += 1
        try:
            data = json.loads(path.read_text())
            if not isinstance(data, dict):
                raise ValueError("invalid artifact shape")
            reports = data.get("reports", {})
            sources = data.get("source_artifacts", {})
            evidence = data.get("evidence_records", [])
            final = data.get("final_decision", {})
            if not all(
                isinstance(value, dict) for value in (reports, sources, final)
            ) or not isinstance(evidence, list):
                raise ValueError("invalid artifact fields")
            report = reports.get("fundamentals_report")
            if isinstance(report, str) and report:
                metric = extract_metrics(report).get("ocf")
                counts["ocf"]["present" if metric is not None else "missing"] += 1
            foreign = sources.get("foreign_language_report")
            if isinstance(foreign, str) and foreign:
                records = _successful_evidence_records(evidence)
                normalized = normalize_foreign_language_evidence(
                    foreign, [], ticker="REPLAY", additional_records=records
                )
                _replay_ownership(foreign, normalized, counts["ownership"])
                block = unique_latest_results_block(foreign)
                coverage = extract_block_field_from_text_raw(
                    block or "", "LATEST_RESULTS_COVERAGE_STATUS"
                )
                if coverage == "FOUND":
                    normalized_block = unique_latest_results_block(normalized) or ""
                    reason = (
                        extract_block_field_from_text_raw(
                            normalized_block, "LATEST_RESULTS_NORMALIZATION_REASON"
                        )
                        or "UNKNOWN"
                    )
                    counts["latest_results"][reason] += 1
                    # Presence diagnostics never grant source authority or repair
                    # a claim. Keep incomplete FOUND distinct from reader gaps.
                    figures = [
                        exact_decimal(
                            extract_block_field_from_text_raw(
                                block or "", f"LATEST_RESULTS_{field}"
                            )
                            or ""
                        )
                        for field in (
                            "REVENUE",
                            "PRIOR_REVENUE",
                            "EARNINGS",
                            "PRIOR_EARNINGS",
                        )
                    ]
                    complete = all(figure is not None for figure in figures)
                    counts["latest_results_claims"][
                        "complete_figures" if complete else "missing_or_invalid_figures"
                    ] += 1
                    url = normalize_http_url(
                        extract_block_field_from_text_raw(
                            block or "", "LATEST_RESULTS_SOURCE_URL"
                        )
                        or ""
                    )
                    acquisition, attempts, cited = _acquisition_state(
                        evidence, records, url, reason, complete
                    )
                    counts["latest_results_acquisition"][acquisition] += 1
                    if len(attempts) > 1:
                        counts["latest_results_claims"][
                            "multiple_document_records"
                        ] += 1
                    if any(r.get("content_truncated") for r in cited):
                        counts["latest_results_claims"][
                            "retained_evidence_truncated"
                        ] += 1
            _replay_eps(data.get("analysis_snapshot"), report, counts["eps_lineage"])
            decision = final.get("decision")
            if isinstance(decision, str) and decision:
                snapshot = data.get("analysis_snapshot")
                flags = data.get("red_flags", [])
                before = validate_decision_trace(decision, snapshot, flags)
                cleaned = normalize_decision_trace_citations(decision, snapshot, flags)
                after = validate_decision_trace(cleaned, snapshot, flags)
                counts["pm_trace"][
                    "valid_before" if before["status"] == "VALID" else "invalid_before"
                ] += 1
                counts["pm_trace"][
                    "valid_after" if after["status"] == "VALID" else "invalid_after"
                ] += 1
            counts["artifacts"]["processed"] += 1
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            # Never print raw exceptions or retained model/tool content.
            counts["artifacts"]["malformed_or_unreadable"] += 1
    return {key: dict(sorted(value.items())) for key, value in counts.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.input_dir.is_dir():
        parser.error("input directory does not exist or is not a directory")
    result: dict[str, Any] = replay_contracts(args.input_dir.glob("*_analysis.json"))
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
