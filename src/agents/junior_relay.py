"""Pure assembly of the Junior's current inspected acquisition round."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import ToolMessage

from src.agents.output_limits import cap_state_value
from src.runtime_diagnostics import (
    ArtifactErrorKind,
    failure_artifact,
    success_artifact,
)
from src.tooling.evidence_recorder import classify_evidence_value
from src.tooling.runtime import TOOL_TIMEOUT, tool_error_message

# Required acquisition sections are a contract, independent of tool permissions.
JUNIOR_REQUIRED_TOOLS = ("get_financial_metrics", "get_fundamental_analysis")


def build_junior_relay(
    ticker: str, outcomes: list[tuple[ToolMessage, dict[str, dict[str, Any]]]]
) -> dict[str, Any]:
    """Relay inspected originals before display caps; never acquire or infer facts."""
    sections = []
    failures: list[str] = []
    error_kind: ArtifactErrorKind = "data_unavailable"
    for number, name in enumerate(JUNIOR_REQUIRED_TOOLS, 1):
        owned = [
            (message, records) for message, records in outcomes if message.name == name
        ]
        reason: str | None = "MISSING_OR_DUPLICATE_ACQUISITION"
        payload = ""
        if len(owned) == 1:
            message, records = owned[0]
            execution, evidence, reason, _ = classify_evidence_value(
                name,
                message.content,
                blocked=bool(message.additional_kwargs.get("blocked")),
            )
            metrics_ok = (
                name != "get_financial_metrics"
                or records.get("raw_financial_metrics", {}).get("status") == "VALID"
            )
            if (
                message.status != "error"
                and execution == "SUCCEEDED"
                and evidence
                in {"EVIDENCE_FOUND", "RESULTS_FOUND", "COVERAGE_COMPLETE_NO_MATCH"}
                and metrics_ok
            ):
                payload = str(message.content)
                if reason == "PARTIAL_PROVIDER_ERROR":
                    payload += "\n\n[CODE ACQUISITION NOTE: Partial provider failure; successful inspected results retained. Error envelopes are unavailable evidence.]"
            elif message.content == tool_error_message(TOOL_TIMEOUT):
                reason = TOOL_TIMEOUT
                error_kind = "timeout"
            elif (
                not metrics_ok
                and records.get("raw_financial_metrics", {}).get("status") == "INVALID"
            ):
                error_kind = "output_contract_violation"
        if not payload:
            reason = reason or "ACQUISITION_UNUSABLE"
            payload = "STATUS: UNAVAILABLE\nREASON: " + reason
            failures.append(f"{name}: {reason}")
        sections.append(f"### TOOL {number}: {name}\n{payload}")
    relay = (
        f"=== RAW FINANCIAL DATA FOR {ticker} ===\n\n"
        + "\n\n".join(sections)
        + "\n\n=== END RAW DATA ==="
    )
    bounded = cap_state_value(relay, "raw_fundamentals_data")
    if bounded != relay:
        failures.append("ARTIFACT_LIMIT_EXCEEDED")
        error_kind = "output_cap_exhausted"
    return (
        failure_artifact(
            "raw_fundamentals_data",
            "; ".join(failures),
            provider="code",
            fallback_content=bounded,
            error_kind=error_kind,
        )
        if failures
        else success_artifact("raw_fundamentals_data", relay, provider="code")
    )
