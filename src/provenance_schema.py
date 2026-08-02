"""Typed, versioned codecs for the gate-critical provenance payloads.

Stage 6 of the provenance-architecture completion. The analysis snapshot,
decision trace, and score scorecard travel through state and persistence as
``dict[str, Any]`` bags read by magic string keys. This module gives the two
lowest-read payloads (``Scorecard``, ``DecisionTrace``) a single typed
definition with an explicit ``to_dict`` wire shape and a fail-closed
``from_dict`` loader. ``AnalysisSnapshot`` (co-located with ``ClaimRecord`` in
``analysis_snapshot``) reuses this module's schema-version machinery.

Design contract:
- **Frozen fields, tuple internals.** ``Scorecard`` / ``DecisionTrace`` hold
  tuples, and every ``to_dict`` builds fresh dicts/lists — they are genuinely
  value-immutable. ``AnalysisSnapshot`` is a *shallow* wire adapter: it is a
  frozen dataclass (no field reassignment) but its ``claims`` / ``conflicts`` are
  shared by reference with the wire dict, so it is a transient codec, not a
  deep-immutable value (documented on the class).
- **Additive versioning.** ``to_dict`` appends ``schema_version`` last. That is
  the only additive wire change for the scorecard, the full decision trace, and
  the full/reduced snapshot. The PM_BLOCK-missing decision trace additionally
  gains the previously-omitted (empty) list keys, because both trace shapes are
  now one unified model.
- **Fail-closed, transport/type-only decoding.** ``from_dict`` raises
  ``SchemaDecodeError`` — never a raw ``TypeError``/``OverflowError`` and never a
  silently-defaulted corrupt value — on a *future* or non-integer
  ``schema_version``, or a present-but-type-invalid field: a non-numeric/non-finite
  number, a non-``bool`` eligibility, a non-string status, a wrong-typed
  collection (``criteria``/``scorecards`` not a mapping), or a list holding a
  non-string element. A missing ``schema_version`` is legacy and loads; a missing
  optional field defaults. The codecs validate *shape and type* only — payload
  *semantics* (a trace's facts/thesis-support invariants, a score's range) are
  owned by their producers (``validate_decision_trace``, the score reconciler),
  not re-derived here. Boundary callers translate the exception into a
  non-publishable / ineligible outcome.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

# Wire numbers preserve their JSON type (int stays int, float stays float) so a
# round-trip is byte-identical; the fields below are therefore int-or-float.
Number = int | float


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


def require_schema_compatible(cls_name: str, seen: Any, current: int) -> None:
    """Raise ``SchemaDecodeError`` unless ``seen`` is a compatible schema version.

    Shared by every payload codec (scorecard, decision trace, analysis snapshot)
    so the future/corrupt-version rule cannot drift between them.
    """
    status = classify_schema_version(seen, current)
    if not status.compatible:
        raise SchemaDecodeError(
            f"{cls_name}: incompatible schema_version {seen!r} (current {current})"
        )


def _coerce_number(value: Any, field: str) -> Number:
    """Preserve an existing finite int/float as-is; raise on anything else.

    Numeric type is preserved (int stays int, float stays float) so a wire
    round-trip is byte-identical. ``bool`` is rejected (an ``int`` subclass but
    never a legitimate score value); NaN/inf are rejected so a corrupt score
    cannot slip past the gate comparisons (``nan >= 50`` and ``nan < 50`` are
    both False).
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise SchemaDecodeError(f"{field} is not numeric: {value!r}")
    # Only floats can be non-finite. ``math.isfinite`` on a huge int itself raises
    # OverflowError (a raw exception that would escape the boundary), so the
    # finiteness check is guarded behind ``isinstance(float)`` — ints are always
    # finite and pass through unchanged.
    if isinstance(value, float) and not math.isfinite(value):
        raise SchemaDecodeError(f"{field} is not finite: {value!r}")
    return value


def _optional_number(value: Any, field: str, default: Number) -> Number:
    """A defaulted numeric: absent → default; present-but-invalid → fail closed.

    Lets a minimal/legacy scorecard (only ``percentage`` + ``decision_eligible``,
    as several internal consumers build) decode, while a *present* corrupt value
    still raises.
    """
    if value is None:
        return default
    return _coerce_number(value, field)


def _require_bool(value: Any, field: str) -> bool:
    """A strict boolean: absent/null → False (legacy); present-non-bool → fail closed.

    Guards against ``bool("false") is True`` making a corrupt scorecard
    decision-eligible. A JSON ``null`` decodes to ``None`` and is treated as the
    absent/legacy default (``False``).
    """
    if value is None:
        return False
    if not isinstance(value, bool):
        raise SchemaDecodeError(
            f"{field} must be a boolean, got {type(value).__name__}"
        )
    return value


def _require_str_or_default(value: Any, default: str, field: str) -> str:
    if value is None:
        return default
    if not isinstance(value, str):
        raise SchemaDecodeError(f"{field} must be a string, got {type(value).__name__}")
    return value


def _str_tuple(value: Any, field: str = "value") -> tuple[str, ...]:
    """Decode a wire list of strings into a tuple; anything else fails closed.

    The historical helper assumed every truthy value was iterable and stringified
    each element, so a corrupt scalar (``"decision_facts": 1``) raised a raw
    ``TypeError`` that escaped the ``SchemaDecodeError`` boundary, and a corrupt
    element (``[1]``) was silently coerced to ``"1"``. Both now fail closed.
    """
    if value is None:
        return ()
    if not isinstance(value, list | tuple):
        raise SchemaDecodeError(f"{field} must be a list, got {type(value).__name__}")
    for item in value:
        if not isinstance(item, str):
            raise SchemaDecodeError(
                f"{field} must contain only strings, got {type(item).__name__}"
            )
    return tuple(value)


@dataclass(frozen=True, slots=True)
class ScorecardCriterion:
    award: str
    max_points: Number
    derived_from: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "award": self.award,
            "max_points": self.max_points,
            "derived_from": list(self.derived_from),
        }

    @classmethod
    def from_dict(cls, d: Any) -> ScorecardCriterion:
        if not isinstance(d, dict):
            raise SchemaDecodeError(
                f"scorecard criterion must be a mapping, got {type(d).__name__}"
            )
        award = d.get("award")
        return cls(
            award=str(award) if award is not None else "N/A",
            max_points=_coerce_number(d.get("max_points"), "max_points"),
            derived_from=_str_tuple(d.get("derived_from"), "derived_from"),
        )


@dataclass(frozen=True, slots=True)
class Scorecard:
    """A HEALTH/GROWTH rubric scorecard (lives under ``snapshot["scorecards"]``)."""

    SCHEMA_VERSION: ClassVar[int] = 1

    criteria: tuple[tuple[str, ScorecardCriterion], ...]
    earned: Number
    available: Number
    rubric_total: Number
    percentage: Number
    advisory_percentage: Number
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
        require_schema_compatible(
            "Scorecard", d.get("schema_version"), cls.SCHEMA_VERSION
        )
        raw_criteria = d.get("criteria")
        if raw_criteria is None:
            raw_criteria = {}
        elif not isinstance(raw_criteria, dict):
            # Explicitly type-checked so a wrong-typed value (e.g. []) fails
            # closed rather than being silently normalized to {} by `or {}`.
            raise SchemaDecodeError(
                f"scorecard.criteria must be a mapping, got {type(raw_criteria).__name__}"
            )
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
            advisory_only_awards=_str_tuple(
                d.get("advisory_only_awards"), "advisory_only_awards"
            ),
            decision_eligible=_require_bool(
                d.get("decision_eligible"), "decision_eligible"
            ),
            lineage_gaps=_str_tuple(d.get("lineage_gaps"), "lineage_gaps"),
        )

    @classmethod
    def decode_or_none(cls, raw: Any) -> Scorecard | None:
        """Decode a scorecard mapping, fail-closed to ``None`` (ineligible path).

        The shared entry point for the decision consumers (score projection, PM
        trace reconciliation, ``DecisionInputs``) so their fail-closed handling
        cannot drift. A non-mapping or an undecodable payload yields ``None`` —
        the conservative "no usable score" outcome.
        """
        if not isinstance(raw, Mapping):
            return None
        try:
            return cls.from_dict(dict(raw))
        except SchemaDecodeError:
            return None


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
        require_schema_compatible(
            "DecisionTrace", d.get("schema_version"), cls.SCHEMA_VERSION
        )
        # Transport/type-only: this codec validates shape, type, and schema
        # version. Trace *semantic* validity — facts/gates present, thesis
        # support for a BUY, no invalid/missing references — is owned solely by
        # validate_decision_trace at construction (the single place that computes
        # `status`), so it is not re-derived here (that would couple the wire
        # codec to the business formula and drift from it). A well-typed but
        # semantically-impossible VALID trace is an accepted residual, addressed
        # by the provenance contract, not by the codec.
        status = _require_str_or_default(d.get("status"), "INVALID", "status")
        verdict = _require_str_or_default(d.get("verdict"), "UNPARSEABLE", "verdict")
        families = d.get("untraced_source_families")
        if families is None:
            families = d.get("advisory_source_families")
        reason = d.get("reason")
        return cls(
            status=status,
            verdict=verdict,
            decision_facts=_str_tuple(d.get("decision_facts"), "decision_facts"),
            decision_gates=_str_tuple(d.get("decision_gates"), "decision_gates"),
            support_facts=_str_tuple(d.get("support_facts"), "support_facts"),
            thesis_support_facts=_str_tuple(
                d.get("thesis_support_facts"), "thesis_support_facts"
            ),
            invalid_facts=_str_tuple(d.get("invalid_facts"), "invalid_facts"),
            invalid_gates=_str_tuple(d.get("invalid_gates"), "invalid_gates"),
            missing_gates=_str_tuple(d.get("missing_gates"), "missing_gates"),
            missing_fields=_str_tuple(d.get("missing_fields"), "missing_fields"),
            source_families=_str_tuple(families, "source_families"),
            reason=reason if reason is None or isinstance(reason, str) else str(reason),
        )
