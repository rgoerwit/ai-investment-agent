"""Pure contracts for management-guidance and earnings-baseline handling."""

from __future__ import annotations

from src.evidence_disposition import AwardDisposition

GUIDANCE_COVERAGE_STATUSES = frozenset(
    {
        "FOUND",
        "NOT_DISCLOSED_AFTER_TARGETED_SEARCH",
        "SEARCH_FAILED",
        "UNRESOLVED_AFTER_TARGETED_SEARCH",
        "NOT_APPLICABLE",
    }
)
EARNINGS_BASELINE_STATUSES = frozenset(
    {
        "DURABLE",
        "MIXED",
        "TEMPORARILY_BOOSTED",
        "TEMPORARILY_DEPRESSED",
        "REGIME_DEPENDENT",
        "UNKNOWN",
    }
)
GUIDANCE_BRIDGE_STATUSES = frozenset({"RECONCILED", "UNRESOLVED", "NOT_APPLICABLE"})
MATERIAL_NONOPERATING_DRIVER_STATUSES = frozenset({"YES", "NO", "UNKNOWN"})
NORMALIZED_EARNINGS_AVAILABLE_STATUSES = frozenset({"YES", "NO", "UNKNOWN", "N/A"})
# The Senior DATA_BLOCK must carry every one of these or the fundamentals artifact
# fails closed and the Portfolio Manager is skipped. Order is load-bearing: the
# validator reports the first offender, so it doubles as diagnostic priority.
# Every entry must have a guaranteed code-owned producer — see
# tests/agents/test_output_validation.py::test_every_required_guidance_field_has_a_code_owned_producer
REQUIRED_GUIDANCE_CONTRACT_FIELDS: tuple[str, ...] = (
    "GUIDANCE_COVERAGE_STATUS",
    "MATERIAL_NONOPERATING_DRIVER",
    "EARNINGS_BASELINE_STATUS",
    "NORMALIZED_EARNINGS_AVAILABLE",
    "GUIDANCE_BRIDGE_STATUS",
)
DISTORTED_EARNINGS_BASELINE_STATUSES = frozenset(
    {
        "MIXED",
        "TEMPORARILY_BOOSTED",
        "TEMPORARILY_DEPRESSED",
        "REGIME_DEPENDENT",
    }
)
"""Positive diagnoses that trailing earnings are an unsafe growth baseline."""

UNCLASSIFIED_EARNINGS_BASELINE_STATUSES = frozenset({"UNKNOWN"})
"""The *absence* of such a diagnosis.

Kept apart from the diagnosed set because the two must not score alike, and the
distinction is easy to lose: ``canonical_guidance_enum`` below manufactures
``UNKNOWN`` from null tokens, so an omitted field arrives here indistinguishable
from a considered "I could not tell" — and both are the absence of evidence, not
evidence of distortion. Scoring them as a rubric failure penalizes exactly the
thinly-covered ex-US names this system exists to find. See
:mod:`src.evidence_disposition`.
"""

UNUSABLE_EARNINGS_BASELINE_STATUSES = (
    DISTORTED_EARNINGS_BASELINE_STATUSES | UNCLASSIFIED_EARNINGS_BASELINE_STATUSES
)
"""Every status that is not ``DURABLE``.

Retained as the union because ``is_unusable_earnings_baseline`` has callers that
legitimately mean "not durable" regardless of why. Callers deciding a *score*
must use the two sets above instead.
"""


REQUIRED_GUIDANCE_CONTRACT_ENUMS: dict[str, frozenset[str]] = {
    "GUIDANCE_COVERAGE_STATUS": GUIDANCE_COVERAGE_STATUSES,
    "MATERIAL_NONOPERATING_DRIVER": MATERIAL_NONOPERATING_DRIVER_STATUSES,
    "EARNINGS_BASELINE_STATUS": EARNINGS_BASELINE_STATUSES,
    "NORMALIZED_EARNINGS_AVAILABLE": NORMALIZED_EARNINGS_AVAILABLE_STATUSES,
    "GUIDANCE_BRIDGE_STATUS": GUIDANCE_BRIDGE_STATUSES,
}


def canonical_enum(value: object) -> str:
    """Return the canonical representation used by structured guidance fields."""
    return str(value or "").strip().upper()


_GUIDANCE_NULL_TO_UNKNOWN_FIELDS = frozenset(
    {
        "OPERATING_VS_NET_DIRECTION",
        "MATERIAL_NONOPERATING_DRIVER",
        "EARNINGS_BASELINE_STATUS",
    }
)
_GUIDANCE_NULL_TOKENS = frozenset({"N/A", "NA", "NONE", "NOT APPLICABLE"})
_GUIDANCE_ENUM_FIELDS = frozenset(
    {
        "COVERAGE_STATUS",
        "SOURCE_TYPE",
        "OPERATING_VS_NET_DIRECTION",
        "MATERIAL_NONOPERATING_DRIVER",
        "DRIVER_TYPE",
        "DRIVER_PERSISTENCE",
        "DRIVER_MATERIALITY",
        "MANAGEMENT_IDENTIFIED",
        "EARNINGS_BASELINE_STATUS",
        "NORMALIZED_EARNINGS_AVAILABLE",
        "GUIDANCE_BRIDGE_STATUS",
    }
)


def canonical_guidance_enum(field: str, value: object) -> str:
    """Canonicalize semantically unknown guidance enums without hiding omissions."""
    if field not in _GUIDANCE_ENUM_FIELDS:
        return str(value or "").strip()
    token = canonical_enum(value)
    if field in _GUIDANCE_NULL_TO_UNKNOWN_FIELDS and token in _GUIDANCE_NULL_TOKENS:
        return "UNKNOWN"
    return token


def guidance_contract_value_is_uninterpretable(field: str, value: object) -> bool:
    """Return whether a required guidance value carries no contract meaning.

    Absent and out-of-enum are the same condition for decision purposes: a token
    the contract does not define (models have emitted a literal ``MISSING`` for
    coverage) cannot be reasoned about, so the deterministic layer is entitled to
    replace it with a conservative value. This never fires on a value the
    validator would accept — including the null tokens it folds to ``UNKNOWN`` —
    so replacement can only add meaning, never remove it.
    """
    allowed = REQUIRED_GUIDANCE_CONTRACT_ENUMS.get(field)
    if allowed is None:
        return False
    token = canonical_enum(value)
    if not token:
        return True
    if token in allowed:
        return False
    return not (token in _GUIDANCE_NULL_TOKENS and "UNKNOWN" in allowed)


def is_unusable_earnings_baseline(value: object) -> bool:
    """Return whether trailing earnings must not be treated as durable."""
    return canonical_enum(value) in UNUSABLE_EARNINGS_BASELINE_STATUSES


def eps_growth_award_disposition(
    *,
    baseline_status: object,
    bridge_status: object,
) -> AwardDisposition:
    """Return *why* sustained EPS-growth credit is withheld, not merely whether.

    The distinction is the whole point. A diagnosed distortion
    (``TEMPORARILY_BOOSTED`` and friends) is evidence the criterion fails, so it
    scores ``0``. An unclassifiable baseline, or an unresolved guidance bridge,
    is the absence of evidence — a rubric criterion is a MERIT use, so it scores
    ``N/A`` and leaves the denominator. Returning a bool made those two
    indistinguishable to the caller, which is how the punitive reading survived.

    Precedence is deliberate: a positive diagnosis outranks an unresolved
    bridge, since a bridge cannot un-diagnose a distortion the analyst found.

    Both parameters are keyword-only. ``EARNINGS_BASELINE_STATUSES`` and
    ``GUIDANCE_BRIDGE_STATUSES`` are disjoint, so transposing them raises
    nothing and type-checks fine — it just falls through every branch and
    returns ``KEEP``, silently restoring the punitive-vs-neutral defect this
    function exists to prevent, in the *permissive* direction. Making the swap
    a ``TypeError`` is cheaper than a test that notices it after the fact.
    """
    baseline = canonical_enum(baseline_status)
    if baseline in DISTORTED_EARNINGS_BASELINE_STATUSES:
        return AwardDisposition.REFUTED
    if baseline in UNCLASSIFIED_EARNINGS_BASELINE_STATUSES:
        return AwardDisposition.UNRESOLVED
    if canonical_enum(bridge_status) == "UNRESOLVED":
        return AwardDisposition.UNRESOLVED
    return AwardDisposition.KEEP


def requires_eps_growth_withholding(
    baseline_status: object,
    bridge_status: object,
) -> bool:
    """Return whether sustained EPS-growth credit must be withheld.

    Retained positional for its existing callers, which only need the boolean;
    the *token* to write is :func:`eps_growth_award_disposition`, and scoring
    sites must use that so they can tell ``0`` from ``N/A``.
    """
    return (
        eps_growth_award_disposition(
            baseline_status=baseline_status, bridge_status=bridge_status
        )
        is not AwardDisposition.KEEP
    )


def is_material_baseline_distortion(value: object) -> bool:
    """Return whether a classified baseline explicitly carries distortion."""
    return canonical_enum(value) in (UNUSABLE_EARNINGS_BASELINE_STATUSES - {"UNKNOWN"})
