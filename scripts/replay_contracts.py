"""Offline contract replay; counts describe coverage, not correctness targets.

Run with ``poetry run python -m scripts.replay_contracts --input-dir results``.
Checked expectations belong in the committed deterministic replay tests.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from src.agents.foreign_language_evidence import (
    normalize_foreign_language_evidence,
    unique_latest_results_block,
)
from src.agents.message_utils import make_tool_evidence_record
from src.data_block_utils import extract_block_field_from_text_raw
from src.pm_claim_audit import (
    normalize_decision_trace_citations,
    validate_decision_trace,
)
from src.validators.metric_extractor import extract_metrics


def replay_contracts(paths: Iterable[Path]) -> dict[str, dict[str, int]]:
    """Replay saved successful evidence without models, network, or file writes."""
    counts: dict[str, Counter[str]] = {
        key: Counter() for key in ("artifacts", "ocf", "latest_results", "pm_trace")
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
                block = unique_latest_results_block(foreign)
                coverage = extract_block_field_from_text_raw(
                    block or "", "LATEST_RESULTS_COVERAGE_STATUS"
                )
                if coverage == "FOUND":
                    records = [
                        make_tool_evidence_record(
                            tool_name=r["tool_name"],
                            content=r["content"],
                            urls=r.get("urls", []),
                            evidence_status=r.get("evidence_status"),
                        )
                        for r in evidence
                        if isinstance(r, dict)
                        and r.get("execution_status") == "SUCCEEDED"
                        and not r.get("blocked")
                        and isinstance(r.get("content"), str)
                        and isinstance(r.get("tool_name"), str)
                        and isinstance(r.get("urls", []), list)
                        and r.get("evidence_status")
                        in {"EVIDENCE_FOUND", "RESULTS_FOUND"}
                    ]
                    normalized = normalize_foreign_language_evidence(
                        foreign, [], ticker="REPLAY", additional_records=records
                    )
                    normalized_block = unique_latest_results_block(normalized) or ""
                    reason = (
                        extract_block_field_from_text_raw(
                            normalized_block, "LATEST_RESULTS_NORMALIZATION_REASON"
                        )
                        or "UNKNOWN"
                    )
                    counts["latest_results"][reason] += 1
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
