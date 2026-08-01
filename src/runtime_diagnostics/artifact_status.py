from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import structlog

from src.data_block_utils import has_parseable_data_block
from src.provenance_schema import DecisionTrace, SchemaDecodeError
from src.runtime_diagnostics.failure_classification import (
    ArtifactErrorKind,
    classify_failure,
)

logger = structlog.get_logger(__name__)


def _decode_snapshot_status(snapshot: Any) -> str | None:
    """Contract status via the typed codec, fail-closed on a corrupt/future shape.

    ``None`` when no snapshot is present (grandfathered legacy path). A
    ``SchemaDecodeError`` (future schema_version or a type-corrupt
    contract_status) yields ``DECODE_FAILED`` → a required failure downstream,
    so a payload current code cannot safely read is never treated as VALID.
    """
    if not isinstance(snapshot, dict):
        return None
    # Lazy import avoids any chance of an analysis_snapshot import cycle.
    from src.analysis_snapshot import AnalysisSnapshot

    try:
        return AnalysisSnapshot.from_dict(snapshot).contract_status or "INVALID"
    except SchemaDecodeError as exc:
        logger.warning("analysis_snapshot_schema_decode_failed", reason=str(exc))
        return "DECODE_FAILED"


def _decode_trace_status(decision_trace: Any) -> str | None:
    if not isinstance(decision_trace, dict):
        return None
    try:
        return DecisionTrace.from_dict(decision_trace).status or "INVALID"
    except SchemaDecodeError as exc:
        logger.warning("decision_trace_schema_decode_failed", reason=str(exc))
        return "DECODE_FAILED"


FUNDAMENTALS_SYNC_FIELDS = frozenset(
    {"raw_fundamentals_data", "foreign_language_report", "legal_report"}
)
SYNC_CHECK_FIELDS = frozenset(
    {
        "market_report",
        "sentiment_report",
        "news_report",
        "value_trap_report",
        "auditor_report",
    }
)
REQUIRED_PUBLISHABLE_ARTIFACTS = frozenset(
    {
        "market_report",
        "sentiment_report",
        "news_report",
        "value_trap_report",
        "fundamentals_report",
        "final_trade_decision",
    }
)
QUICK_REQUIRED_PUBLISHABLE_ARTIFACTS = REQUIRED_PUBLISHABLE_ARTIFACTS - frozenset(
    {"value_trap_report"}
)
OPTIONAL_PUBLISHABLE_ARTIFACTS = frozenset(
    {"auditor_report", "consultant_review", "valuation_params", "apac_regional_report"}
)
QUICK_OPTIONAL_PUBLISHABLE_ARTIFACTS = OPTIONAL_PUBLISHABLE_ARTIFACTS | frozenset(
    {"value_trap_report"}
)

# Provenance publication contract. Any result stamped with this version is
# produced by current code and therefore MUST carry a present-and-VALID canonical
# analysis snapshot and final decision trace to be publishable — absence is a
# failure, not a pass. Legacy/frozen artifacts (produced before the stamp existed)
# carry no version and are grandfathered: their snapshot/trace are only checked
# when present. Bump when the required-provenance shape changes.
PROVENANCE_CONTRACT_VERSION = 1


def stamp_provenance_contract(result: dict[str, Any]) -> None:
    """Stamp the current provenance contract onto a live-run result.

    The single stamping seam for every current-run entry point (main analyzer,
    portfolio_manager refresh persistence). Sets — not setdefault — so a stale or
    lower version can never survive on a run current code produced.
    """
    result["provenance_contract_version"] = PROVENANCE_CONTRACT_VERSION


def has_provenance_contract(result: Mapping[str, Any]) -> bool:
    """Whether ``result`` declares the current provenance publication contract.

    True only for artifacts stamped by current code; legacy artifacts (no stamp)
    return False and are grandfathered through the present-but-invalid checks.
    """
    version = result.get("provenance_contract_version")
    return isinstance(version, int) and version >= PROVENANCE_CONTRACT_VERSION


def _missing_provenance_failure(message: str) -> dict[str, Any]:
    return {
        "complete": False,
        "ok": False,
        "content": None,
        "error_kind": "data_contract_error",
        "provider": "deterministic",
        "message": message,
        "retryable": False,
    }


@dataclass(frozen=True)
class ArtifactStatus:
    complete: bool
    ok: bool
    content: str | None
    error_kind: ArtifactErrorKind | None = None
    provider: str | None = None
    message: str | None = None
    retryable: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def success_artifact(
    field: str, content: str, *, provider: str | None = None
) -> dict[str, Any]:
    return {
        field: content,
        "artifact_statuses": {
            field: ArtifactStatus(
                complete=True,
                ok=True,
                content=content,
                provider=provider,
            ).as_dict()
        },
    }


def failure_artifact(
    field: str,
    exc: BaseException | str,
    *,
    provider: str | None = None,
    fallback_content: str = "",
    error_kind: ArtifactErrorKind | None = None,
) -> dict[str, Any]:
    if isinstance(exc, BaseException):
        details = classify_failure(exc, provider=provider)
        status = ArtifactStatus(
            complete=True,
            ok=False,
            content=fallback_content or None,
            error_kind=details.kind,
            provider=details.provider,
            message=details.message,
            retryable=details.retryable,
        )
    else:
        status = ArtifactStatus(
            complete=True,
            ok=False,
            content=fallback_content or None,
            error_kind=error_kind or "application_error",
            provider=provider,
            message=str(exc)[:400],
            retryable=False,
        )

    return {
        field: fallback_content,
        "artifact_statuses": {field: status.as_dict()},
    }


def get_artifact_status(state: Mapping[str, Any], field: str) -> ArtifactStatus:
    statuses = state.get("artifact_statuses", {}) or {}
    raw = statuses.get(field)
    if isinstance(raw, dict):
        return ArtifactStatus(
            complete=bool(raw.get("complete", True)),
            ok=bool(raw.get("ok")),
            content=raw.get("content"),
            error_kind=raw.get("error_kind"),
            provider=raw.get("provider"),
            message=raw.get("message"),
            retryable=bool(raw.get("retryable")),
        )

    content = state.get(field)
    normalized = content if isinstance(content, str) else None
    complete = bool(normalized)
    return ArtifactStatus(
        complete=complete,
        ok=bool(normalized),
        content=normalized,
    )


def is_artifact_complete(state: Mapping[str, Any], field: str) -> bool:
    return get_artifact_status(state, field).complete


def is_artifact_valid(state: Mapping[str, Any], field: str) -> bool:
    return get_artifact_status(state, field).ok


def get_valid_artifact_content(
    state: Mapping[str, Any], field: str, default: str = ""
) -> str:
    status = get_artifact_status(state, field)
    if not status.ok or not status.content:
        return default
    return status.content


def _is_quick_mode_result(result: dict[str, Any]) -> bool:
    run_summary = result.get("run_summary")
    if isinstance(run_summary, dict):
        return bool(run_summary.get("quick_mode", False))

    metadata = result.get("metadata")
    if isinstance(metadata, dict):
        return bool(metadata.get("quick_mode", False))

    return bool(result.get("quick_mode", False))


def get_required_publishable_artifacts(result: dict[str, Any]) -> frozenset[str]:
    return (
        QUICK_REQUIRED_PUBLISHABLE_ARTIFACTS
        if _is_quick_mode_result(result)
        else REQUIRED_PUBLISHABLE_ARTIFACTS
    )


def get_optional_publishable_artifacts(result: dict[str, Any]) -> frozenset[str]:
    return (
        QUICK_OPTIONAL_PUBLISHABLE_ARTIFACTS
        if _is_quick_mode_result(result)
        else OPTIONAL_PUBLISHABLE_ARTIFACTS
    )


def build_analysis_validity(result: dict[str, Any]) -> dict[str, Any]:
    fundamentals = get_valid_artifact_content(result, "fundamentals_report")
    pm_decision = get_valid_artifact_content(result, "final_trade_decision")
    data_block_present = has_parseable_data_block(fundamentals)
    required_failures: dict[str, Any] = {}
    optional_failures: dict[str, Any] = {}
    required_artifacts = get_required_publishable_artifacts(result)
    optional_artifacts = get_optional_publishable_artifacts(result)
    snapshot = result.get("analysis_snapshot")
    snapshot_status = _decode_snapshot_status(snapshot)
    decision_trace = result.get("decision_trace")
    decision_trace_status = _decode_trace_status(decision_trace)

    for field in required_artifacts:
        status = get_artifact_status(result, field)
        if not status.ok:
            required_failures[field] = status.as_dict()

    for field in optional_artifacts:
        status = get_artifact_status(result, field)
        if status.complete and not status.ok:
            optional_failures[field] = status.as_dict()

    pre_screening = result.get("pre_screening_result")
    has_valid_pre_screening = pre_screening in {"PASS", "REJECT"}
    if not has_valid_pre_screening:
        required_failures["pre_screening_result"] = {
            "complete": bool(pre_screening),
            "ok": False,
            "content": pre_screening if isinstance(pre_screening, str) else None,
            "error_kind": "application_error",
            "provider": "unknown",
            "message": "Pre-screening result missing or invalid",
            "retryable": False,
        }
    provenance_required = has_provenance_contract(result)
    if provenance_required and not isinstance(snapshot, dict):
        required_failures["analysis_snapshot"] = _missing_provenance_failure(
            "Canonical analysis snapshot is missing under the provenance contract"
        )
    elif snapshot_status is not None and snapshot_status != "VALID":
        required_failures["analysis_snapshot"] = {
            "complete": bool(snapshot),
            "ok": False,
            "content": None,
            "error_kind": "data_contract_error",
            "provider": "deterministic",
            "message": f"Canonical analysis snapshot is {snapshot_status}",
            "retryable": False,
        }
    if provenance_required and not isinstance(decision_trace, dict):
        required_failures["decision_trace"] = _missing_provenance_failure(
            "Final decision trace is missing under the provenance contract"
        )
    elif decision_trace_status is not None and decision_trace_status != "VALID":
        required_failures["decision_trace"] = {
            "complete": bool(decision_trace),
            "ok": False,
            "content": None,
            "error_kind": "data_contract_error",
            "provider": "deterministic",
            "message": f"Final decision trace is {decision_trace_status}",
            "retryable": False,
        }

    publishable = bool(
        pm_decision
        and fundamentals
        and data_block_present
        and has_valid_pre_screening
        and not required_failures
    )
    return {
        "publishable": publishable,
        "has_valid_pm_decision": bool(pm_decision),
        "has_valid_fundamentals": bool(fundamentals),
        "has_data_block": data_block_present,
        "has_valid_pre_screening": has_valid_pre_screening,
        "analysis_snapshot_status": snapshot_status,
        "decision_trace_status": decision_trace_status,
        "required_artifacts": sorted(required_artifacts),
        "optional_artifacts": sorted(optional_artifacts),
        "required_failures": required_failures,
        "optional_failures": optional_failures,
        # Backward-compatible alias for older callers/tests.
        "fatal_failures": required_failures,
    }


def is_publishable_analysis(result: dict[str, Any]) -> bool:
    validity = result.get("analysis_validity")
    if isinstance(validity, dict) and "publishable" in validity:
        return bool(validity["publishable"])
    return bool(build_analysis_validity(result)["publishable"])
