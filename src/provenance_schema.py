"""Typed, versioned codecs for the gate-critical provenance payloads.

Stage 6 of the provenance-architecture completion. The analysis snapshot,
decision trace, and score scorecard travel through state and persistence as
``dict[str, Any]`` bags read by magic string keys. This module gives the two
lowest-read payloads (``Scorecard``, ``DecisionTrace``) a single typed
definition with an explicit ``to_dict`` wire shape and a fail-closed
``from_dict`` loader. ``AnalysisSnapshot`` (co-located with ``ClaimRecord`` in
``analysis_snapshot``) reuses the ``SchemaStatus`` machinery here.

Design contract:
- **Immutable**: frozen dataclasses hold tuples internally; ``to_dict``
  materializes the exact existing wire shape (lists, key set, key order).
- **Additive versioning**: ``to_dict`` appends ``schema_version`` last; nothing
  else in the wire shape changes.
- **Fail-closed decoding**: ``from_dict`` raises ``SchemaDecodeError`` on a
  *future* (``seen > current``) or non-integer ``schema_version`` and on a
  present-but-type-invalid *required* field. A missing version is legacy and
  loads. Boundary callers translate the exception into a non-publishable /
  ineligible outcome — never a silent default of a corrupted value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar


class SchemaDecodeError(Exception):
    """A payload could not be safely decoded (future schema or corrupt field)."""


@dataclass(frozen=True, slots=True)
class SchemaStatus:
    compatible: bool
    legacy: bool
    future: bool
    seen: int | None


def classify_schema_version(seen: Any, current: int) -> SchemaStatus:
    """Classify an observed ``schema_version`` against the current one.

    - ``None`` → legacy (all pre-Stage-6 artifacts): compatible, loads.
    - ``<= current`` integer → compatible.
    - ``> current`` integer → future: incompatible (unknown newer shape).
    - non-integer → incompatible (corrupt).
    """
    if seen is None:
        return SchemaStatus(compatible=True, legacy=True, future=False, seen=None)
    if isinstance(seen, bool) or not isinstance(seen, int):
        return SchemaStatus(compatible=False, legacy=False, future=False, seen=None)
    if seen > current:
        return SchemaStatus(compatible=False, legacy=False, future=True, seen=seen)
    return SchemaStatus(compatible=True, legacy=False, future=False, seen=seen)


def _check_version(cls_name: str, seen: Any, current: int) -> None:
    status = classify_schema_version(seen, current)
    if not status.compatible:
        raise SchemaDecodeError(
            f"{cls_name}: incompatible schema_version {seen!r} (current {current})"
        )


def _coerce_number(value: Any, field: str) -> float:
    """Preserve an existing int/float as-is; raise on a non-numeric required field.

    Numeric type is preserved (int stays int, float stays float) so a wire
    round-trip is byte-identical. ``bool`` is rejected (it is an ``int``
    subclass but never a legitimate score value).
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise SchemaDecodeError(f"{field} is not numeric: {value!r}")
    return value


def _optional_number(value: Any, field: str, default: float) -> float:
    """A defaulted numeric: absent → default; present-but-non-numeric → fail closed.

    Lets a minimal/legacy scorecard (only ``percentage`` + ``decision_eligible``,
    as several internal consumers build) decode, while a *present* corrupt value
    still raises.
    """
    if value is None:
        return default
    return _coerce_number(value, field)


def _require_str_or_default(value: Any, default: str, field: str) -> str:
    if value is None:
        return default
    if not isinstance(value, str):
        raise SchemaDecodeError(f"{field} must be a string, got {type(value).__name__}")
    return value


def _str_tuple(value: Any) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(str(item) for item in value)


@dataclass(frozen=True, slots=True)
class ScorecardCriterion:
    award: str
    max_points: float
    derived_from: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "award": self.award,
            "max_points": self.max_points,
            "derived_from": list(self.derived_from),
        }

    @classmethod
    def from_dict(cls, d: Any) -> ScorecardCriterion:
        award = d.get("award") if isinstance(d, dict) else None
        return cls(
            award=str(award) if award is not None else "N/A",
            max_points=_coerce_number(
                d.get("max_points") if isinstance(d, dict) else None, "max_points"
            ),
            derived_from=_str_tuple(
                d.get("derived_from") if isinstance(d, dict) else None
            ),
        )


@dataclass(frozen=True, slots=True)
class Scorecard:
    """A HEALTH/GROWTH rubric scorecard (lives under ``snapshot["scorecards"]``)."""

    SCHEMA_VERSION: ClassVar[int] = 1

    criteria: tuple[tuple[str, ScorecardCriterion], ...]
    earned: float
    available: float
    rubric_total: float
    percentage: float
    advisory_percentage: float
    advisory_only_awards: tuple[str, ...]
    decision_eligible: bool
    lineage_gaps: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "criteria": {name: crit.to_dict() for name, crit in self.criteria},
            "earned": self.earned,
            "available": self.available,
            "rubric_total": self.rubric_total,
            "percentage": self.percentage,
            "advisory_percentage": self.advisory_percentage,
            "advisory_only_awards": list(self.advisory_only_awards),
            "decision_eligible": self.decision_eligible,
            "lineage_gaps": list(self.lineage_gaps),
            "schema_version": self.SCHEMA_VERSION,
        }

    @classmethod
    def from_dict(cls, d: Any) -> Scorecard:
        if not isinstance(d, dict):
            raise SchemaDecodeError(
                f"scorecard must be a mapping, got {type(d).__name__}"
            )
        _check_version("Scorecard", d.get("schema_version"), cls.SCHEMA_VERSION)
        raw_criteria = d.get("criteria") or {}
        if not isinstance(raw_criteria, dict):
            raise SchemaDecodeError("scorecard.criteria must be a mapping")
        percentage = _coerce_number(d.get("percentage"), "percentage")
        advisory = d.get("advisory_percentage")
        return cls(
            criteria=tuple(
                (str(name), ScorecardCriterion.from_dict(component))
                for name, component in raw_criteria.items()
            ),
            earned=_optional_number(d.get("earned"), "earned", 0.0),
            available=_optional_number(d.get("available"), "available", 0.0),
            rubric_total=_optional_number(d.get("rubric_total"), "rubric_total", 0.0),
            percentage=percentage,
            advisory_percentage=(
                _coerce_number(advisory, "advisory_percentage")
                if advisory is not None
                else percentage
            ),
            advisory_only_awards=_str_tuple(d.get("advisory_only_awards")),
            decision_eligible=bool(d.get("decision_eligible")),
            lineage_gaps=_str_tuple(d.get("lineage_gaps")),
        )


@dataclass(frozen=True, slots=True)
class DecisionTrace:
    """The PM final decision trace (``result["decision_trace"]``).

    ``source_families`` is the single canonical field: the legacy wire carried
    ``untraced_source_families`` and ``advisory_source_families`` identical by
    construction, so ``to_dict`` emits both from this one field for
    compatibility.
    """

    SCHEMA_VERSION: ClassVar[int] = 1

    status: str
    verdict: str
    decision_facts: tuple[str, ...] = ()
    decision_gates: tuple[str, ...] = ()
    support_facts: tuple[str, ...] = ()
    thesis_support_facts: tuple[str, ...] = ()
    invalid_facts: tuple[str, ...] = ()
    invalid_gates: tuple[str, ...] = ()
    missing_gates: tuple[str, ...] = ()
    missing_fields: tuple[str, ...] = ()
    source_families: tuple[str, ...] = ()
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        families = list(self.source_families)
        return {
            "status": self.status,
            "verdict": self.verdict,
            "decision_facts": list(self.decision_facts),
            "decision_gates": list(self.decision_gates),
            "support_facts": list(self.support_facts),
            "thesis_support_facts": list(self.thesis_support_facts),
            "invalid_facts": list(self.invalid_facts),
            "invalid_gates": list(self.invalid_gates),
            "missing_gates": list(self.missing_gates),
            "missing_fields": list(self.missing_fields),
            "untraced_source_families": families,
            "advisory_source_families": list(families),
            "reason": self.reason,
            "schema_version": self.SCHEMA_VERSION,
        }

    @classmethod
    def from_dict(cls, d: Any) -> DecisionTrace:
        if not isinstance(d, dict):
            raise SchemaDecodeError(
                f"decision_trace must be a mapping, got {type(d).__name__}"
            )
        _check_version("DecisionTrace", d.get("schema_version"), cls.SCHEMA_VERSION)
        families = d.get("untraced_source_families")
        if families is None:
            families = d.get("advisory_source_families")
        reason = d.get("reason")
        return cls(
            status=_require_str_or_default(d.get("status"), "INVALID", "status"),
            verdict=_require_str_or_default(d.get("verdict"), "UNPARSEABLE", "verdict"),
            decision_facts=_str_tuple(d.get("decision_facts")),
            decision_gates=_str_tuple(d.get("decision_gates")),
            support_facts=_str_tuple(d.get("support_facts")),
            thesis_support_facts=_str_tuple(d.get("thesis_support_facts")),
            invalid_facts=_str_tuple(d.get("invalid_facts")),
            invalid_gates=_str_tuple(d.get("invalid_gates")),
            missing_gates=_str_tuple(d.get("missing_gates")),
            missing_fields=_str_tuple(d.get("missing_fields")),
            source_families=_str_tuple(families),
            reason=reason if reason is None or isinstance(reason, str) else str(reason),
        )
