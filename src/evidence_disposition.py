"""How a decision resolves evidence it could not obtain.

Every deterministic gate in this repo eventually meets an input it cannot
resolve — a metric the sources did not carry, a status the analyst could not
classify, a flag ledger that failed to load. What happens next is a *policy*
choice, and for a long time it was made ad hoc at each site with the reasoning
living only in prose. That produced two live defects with opposite signs:

* ``EPS_GROWTH`` was scored ``0`` when ``EARNINGS_BASELINE_STATUS`` came back
  ``UNKNOWN``. ``0`` stays in the rubric denominator where ``N/A`` leaves it, so
  "we could not classify this" landed as "this failed" — on a criterion feeding
  the hard ``GROWTH < 50%`` gate. Measured 2026-08-15: it fired on 6 of 12
  tickers and pushed 4 of 6 quick names below a gate they previously cleared.
  On a screener that is a false negative, and a false negative is never revisited.
* The Legal Counsel fallback substituted ``cmic_status: "N/A"`` when the agent
  was unavailable, so an *unreachable* compliance check produced a CMIC-**clean**
  record — inverting that prompt's own "when uncertain, report UNCERTAIN".

Both are the same category error. The rule that resolves it:

    Missing evidence about MERIT resolves NEUTRAL.
    Missing evidence about HAZARD or AUTHORITY resolves CLOSED.

Scoring merit punitively invents a failure; screening hazard permissively
invents an all-clear. The direction is not a matter of taste and it is not
uniform — which is exactly why picking one has to be an explicit act rather
than a default that falls out of whichever sentinel happened to be in scope.

This module is deliberately tiny and dependency-free so that any layer may
import it. It does not *apply* the policy — callers do, because only they know
what "neutral" means in their own vocabulary (``N/A`` in a rubric, an
``UNCERTAIN`` token in a compliance field, a withheld action in the reconciler).
It exists so the choice is named, greppable, and testable in one place.

The cross-layer contracts are pinned in
``tests/test_authority_contracts.py::TestMissingEvidenceDisposition`` — one row
per consumer, because a per-consumer unit test cannot catch a consumer that
never asked the question.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = [
    "AbsentEvidence",
    "AwardDisposition",
    "EvidenceUse",
    "resolve_absent_evidence",
    "rubric_award_token",
]


class EvidenceUse(StrEnum):
    """What a decision is using a piece of evidence *for*.

    This is the only input to the policy, and it is a property of the consumer
    rather than of the datum: the same ``UNKNOWN`` earnings baseline is a merit
    input to the growth rubric and would be an authority input to a sell gate.
    """

    MERIT = "merit"
    """Scores a positive attribute (growth, health, quality). Absence must not
    be scored as a failure — the stock is not penalized for the pipeline's
    inability to measure it."""

    HAZARD = "hazard"
    """Screens for a risk (PFIC, VIE, CMIC, sanctions, accounting distortion).
    Absence must not be reported as an all-clear."""

    AUTHORITY = "authority"
    """Gates an action with real-world consequence (initiate, sell, bind an
    identity to a position). Absence must not authorize the action."""


class AbsentEvidence(StrEnum):
    """The resolved policy for an input the pipeline could not obtain."""

    NEUTRAL = "neutral"
    """Remove the input from the decision entirely: it must neither help nor
    hurt. In a scored rubric this is ``N/A`` (out of numerator *and*
    denominator), never ``0`` (in the denominator, i.e. a failure)."""

    CLOSED = "closed"
    """Assume the unfavourable reading: flag the hazard as uncertain, or
    withhold the action, until evidence actually arrives."""


_ABSENT_EVIDENCE_POLICY: dict[EvidenceUse, AbsentEvidence] = {
    EvidenceUse.MERIT: AbsentEvidence.NEUTRAL,
    EvidenceUse.HAZARD: AbsentEvidence.CLOSED,
    EvidenceUse.AUTHORITY: AbsentEvidence.CLOSED,
}


def resolve_absent_evidence(use: EvidenceUse) -> AbsentEvidence:
    """Return how an unobtainable input must be resolved for this use.

    Total over ``EvidenceUse`` by construction: adding a use without a policy
    is a ``KeyError`` at import-adjacent test time rather than a silent default,
    because a silent default is the failure mode this module exists to remove.
    """
    return _ABSENT_EVIDENCE_POLICY[use]


class AwardDisposition(StrEnum):
    """Why a rubric criterion is being withheld — evidence against, or no evidence.

    Separated from the raw token because the two collapse to visibly different
    arithmetic and the collapse is easy to make by accident: the withholding
    site that motivated this module wrote a bare ``"0"`` for both cases.
    """

    KEEP = "keep"
    """Evidence supports the award; leave it alone."""

    REFUTED = "refuted"
    """Positive evidence that the criterion fails (a *diagnosed* earnings
    distortion, not an unclassifiable one). Scores ``0``."""

    UNRESOLVED = "unresolved"
    """No evidence either way. Scores ``N/A`` — a rubric criterion is a MERIT
    use, so absence is neutral."""


_AWARD_TOKENS: dict[AwardDisposition, str] = {
    AwardDisposition.REFUTED: "0",
    AwardDisposition.UNRESOLVED: "N/A",
}


def rubric_award_token(disposition: AwardDisposition) -> str:
    """Return the breakdown token for a withheld criterion.

    ``"0"`` keeps the criterion in the rubric denominator; ``"N/A"`` removes it.
    That single character decides whether a data gap reads as a failure, which
    is why the mapping is here rather than inline at a call site.

    Raises:
        ValueError: for :attr:`AwardDisposition.KEEP` — there is no withheld
            token for an award that is not being withheld, and silently
            returning one would let a caller zero a criterion it meant to keep.
    """
    try:
        return _AWARD_TOKENS[disposition]
    except KeyError:
        raise ValueError(
            f"{disposition!r} does not withhold an award; nothing to serialize"
        ) from None
