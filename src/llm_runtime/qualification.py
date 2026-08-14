"""Evidence levels for provider support; construction is not qualification."""

from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import Any

from src.llm_runtime.bindings import BindingPlan
from src.llm_runtime.capabilities import Capability
from src.llm_runtime.seats import SEATS, SeatId


class QualificationLevel(StrEnum):
    CONSTRUCTIBLE = "constructible"
    CONTRACT_CAPABLE = "contract_capable"
    PRODUCTION_QUALIFIED = "production_qualified"


@dataclass(frozen=True)
class QualificationEvidence:
    provider: str
    level: QualificationLevel
    seats: tuple[SeatId, ...]
    verified_on: date
    evidence_ref: str
    judge_model: str | None = None


def offline_contract_findings(plan: BindingPlan) -> tuple[str, ...]:
    """Check reviewed capability facts for every reachable normal/quick seat."""

    findings: list[str] = []
    for seat_id, spec in SEATS.items():
        if not plan.statuses[seat_id].enabled:
            continue
        for label, binding in (
            ("normal", plan.bindings[seat_id]),
            ("quick", plan.quick_bindings[seat_id]),
        ):
            missing = spec.requires - binding.profile.capabilities
            if missing:
                findings.append(
                    f"{seat_id.value} ({label}) lacks "
                    + ", ".join(sorted(capability.value for capability in missing))
                )
    return tuple(findings)


def validate_live_contract_result(
    *,
    seat_id: SeatId,
    tool_calls_valid: bool,
    structured_output_valid: bool,
    artifact_valid: bool,
    usage_recorded: bool,
) -> tuple[str, ...]:
    """Normalize live/recorded conformance results without weakening seat rules."""

    required = SEATS[seat_id].requires
    failures: list[str] = []
    if Capability.TOOL_CALLING in required and not tool_calls_valid:
        failures.append("tool_calling")
    if Capability.STRUCTURED_OUTPUT in required and not structured_output_valid:
        failures.append("structured_output")
    if not artifact_valid:
        failures.append("artifact_contract")
    if not usage_recorded:
        failures.append("usage_telemetry")
    return tuple(failures)


def evidence_to_dict(evidence: QualificationEvidence) -> dict[str, Any]:
    return {
        "provider": evidence.provider,
        "level": evidence.level.value,
        "seats": [seat.value for seat in evidence.seats],
        "verified_on": evidence.verified_on.isoformat(),
        "evidence_ref": evidence.evidence_ref,
        "judge_model": evidence.judge_model,
    }
