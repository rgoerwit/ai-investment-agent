"""What a retrospective outcome may produce, decided deterministically.

Every lesson-quality defect chased through August 2026 had one shape: the
generator was asked for content the evidence could not support, and the response
was another rule policing the output. A prompt rule, a decline token, a grounding
gate, an evidence recovery — each correct, none sufficient, because the request
itself was impossible. The offending instruction was::

    If the lesson scope is UNRESOLVED ... write what should be CHECKED next time.

For an unexplained residual with only price data, *what should be checked next
time* has no grounded answer. So the model invented one: FX exposure for a
German listing whose bear case never mentions currency (2PP.DE), liquidity and
momentum screens for a case about analyst coverage and margins (3008.TW).

This module states the rule that ends it:

    Never ask a generator for content the evidence cannot support. Where the
    evidence determines the content, decide deterministically.

Two axes, deliberately separate, because one boolean was answering both:

* **Capability** — what the *decision-time record* can support. A recorded bear
  risk licenses a review question about that risk. A recorded macro regime
  licenses an observation about the regime. Neither licenses the other, and this
  conflation is what let a regime-only snapshot produce company-mechanism prose.
* **Disposition** — what may be produced, given the capability *and* the measured
  outcome. Computable only after pricing, which is why it is a separate stage:
  ``has_grounding_context`` runs in the candidate loop, before any fetch.

Pure: no LLM, no I/O, no mutation, primitive inputs. The whole policy is
matrix-testable without a snapshot fixture or a network call.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

__all__ = [
    "ACTIVE_TENDER_STATUS",
    "EvidenceCapability",
    "LessonDisposition",
    "DispositionVerdict",
    "DRIVER_MARKET",
    "DRIVER_RESIDUAL",
    "DRIVER_MIXED",
    "DRIVER_UNKNOWN",
    "ATTRIBUTION_DOMINANCE_RATIO",
    "derive_disposition",
    "OutcomeFacts",
    "render_record",
    "HYPOTHESIS_TOPIC_BEAR_CASE",
    "HYPOTHESIS_TOPIC_RED_FLAGS",
    "HYPOTHESIS_TOPIC_KILL_CRITERIA",
    "HYPOTHESIS_TOPIC_NONE",
    "MATERIALIZATION_NOT_EVALUATED",
]

# How much larger one leg must be than the other to be called the driver. Lives
# here rather than in `retrospective` because dominance is a disposition input,
# and this module must import nothing from its consumer.
ATTRIBUTION_DOMINANCE_RATIO = 1.5

DRIVER_MARKET = "MARKET"
DRIVER_RESIDUAL = "RESIDUAL"
DRIVER_MIXED = "MIXED"
DRIVER_UNKNOWN = "UNKNOWN"

# A live deal prices the stock against its terms. `RUMORED` is deliberately not
# included: a rumour does not pin the price, so attribution still means something.
ACTIVE_TENDER_STATUS = "ACTIVE_TENDER"


class EvidenceCapability(StrEnum):
    """What the decision-time record can license — never what happened.

    Both may hold at once. Neither substitutes for the other: this is the
    distinction a single ``has_grounding_context`` boolean could not express, and
    its absence is why regime-only snapshots produced company-mechanism prose.
    """

    #: A decision-time claim about the company — bear risks, red flags, or
    #: pre-registered thesis-break triggers. Licenses a *review question* about
    #: that specific claim. Never establishes that the claim materialized.
    HYPOTHESIS = "HYPOTHESIS"

    #: A recorded macro regime. Licenses an observation about the regime, and
    #: nothing about the company.
    CONTEXT = "CONTEXT"


class LessonDisposition(StrEnum):
    """What this outcome may produce. Total over the inputs; no implicit default."""

    #: Nothing recorded to reason from. Declined before pricing.
    SKIP_NO_EVIDENCE = "SKIP_NO_EVIDENCE"

    #: A live deal priced the stock against its terms, so the benchmark
    #: decomposition is an artefact of deal mechanics however the legs fall.
    SPECIAL_SITUATION_REVIEW = "SPECIAL_SITUATION_REVIEW"

    #: Market-dominated outcome in a recorded, demonstrably unchanged regime.
    #: The only injectable disposition.
    CONTEXTUAL_OBSERVATION = "CONTEXTUAL_OBSERVATION"

    #: An unattributed outcome against a recorded company hypothesis. Produces a
    #: review record quoting that hypothesis — never a cause, never a suggested
    #: check the record does not contain.
    REVIEW_HYPOTHESIS = "REVIEW_HYPOTHESIS"

    #: An unattributed outcome with context but no company hypothesis. **Produces
    #: no record at all.** A stored document saying "cause unresolved" would fill
    #: a durable store with repetitive non-actionable text; the honest artifact
    #: is the run counter, not a lesson.
    WITHHOLD_UNRESOLVED = "WITHHOLD_UNRESOLVED"

    @property
    def is_injectable(self) -> bool:
        """Only a contextual observation may ever reach a live analysis."""
        return self is LessonDisposition.CONTEXTUAL_OBSERVATION

    @property
    def produces_record(self) -> bool:
        """Whether anything is written to the durable store."""
        return self in {
            LessonDisposition.CONTEXTUAL_OBSERVATION,
            LessonDisposition.REVIEW_HYPOTHESIS,
            LessonDisposition.SPECIAL_SITUATION_REVIEW,
        }


@dataclass(frozen=True, slots=True)
class DispositionVerdict:
    """The disposition plus a machine-readable reason.

    ``reason_code`` exists so consumers and audits can branch or aggregate
    without parsing prose — the mistake the earlier eligibility layer made by
    returning only a sentence.
    """

    disposition: LessonDisposition
    reason_code: str
    reason: str

    @property
    def is_injectable(self) -> bool:
        return self.disposition.is_injectable


def derive_disposition(
    capabilities: frozenset[EvidenceCapability],
    *,
    dominant_driver: str,
    regime_shifted: bool | None,
    m_and_a_status: str | None = None,
    regime_shift_reason: str = "",
) -> DispositionVerdict:
    """Decide what this outcome may produce.

    Precedence is explicit and ordered, because several rules can apply at once:

    1. **Active tender wins over attribution.** The decomposition still computes
       and means nothing, so reporting ``driver MIXED`` would name a symptom and
       hide the cause.
    2. **Market-dominated, in a recorded and unchanged regime** is the sole
       injectable path. ``regime_shifted is False`` and not merely "not True":
       ``None`` means the comparison could not be made, and an unestablished
       stability is not a stable regime.
    3. **A recorded company hypothesis** makes anything else a review record.
    4. **Context alone** produces nothing — see ``WITHHOLD_UNRESOLVED``.
    5. **Neither capability** is skipped before pricing.

    Note rule 2 before rule 3 deliberately. A snapshot may hold *both*
    capabilities; when the outcome is market-dominated it stays a contextual
    observation, and the bear hypothesis is not promoted into a cause by the
    coincidence of also being on file.
    """
    if (m_and_a_status or "").strip().upper() == ACTIVE_TENDER_STATUS:
        return DispositionVerdict(
            LessonDisposition.SPECIAL_SITUATION_REVIEW,
            "active_tender",
            "an active tender priced the stock against deal terms, not the market",
        )

    if EvidenceCapability.CONTEXT in capabilities:
        if dominant_driver == DRIVER_MARKET and regime_shifted is False:
            return DispositionVerdict(
                LessonDisposition.CONTEXTUAL_OBSERVATION,
                "market_dominated_stable_regime",
                "market-dominated outcome in a recorded, unchanged regime",
            )

    # One explanation, shared by both remaining branches. They differ in what
    # they may produce, not in why the contextual path was unavailable — and
    # deriving it twice is how the two would drift into disagreeing.
    blocker_code, blocker = _why_not_contextual(
        capabilities, dominant_driver, regime_shifted
    )
    # The shift detail is a fact the policy layer cannot derive — "the cache was
    # 40d old" describes the run, "RISK_OFF -> RISK_ON" describes the world — and
    # it belongs on *either* regime-blocked disposition, not only one.
    detail = regime_shift_reason.strip()
    if detail and blocker_code.startswith("regime_"):
        blocker = f"{blocker} ({detail})"

    if EvidenceCapability.HYPOTHESIS in capabilities:
        return DispositionVerdict(
            LessonDisposition.REVIEW_HYPOTHESIS,
            f"review:{blocker_code}",
            f"{blocker}; a decision-time hypothesis is on record to review",
        )

    if EvidenceCapability.CONTEXT in capabilities:
        return DispositionVerdict(
            LessonDisposition.WITHHOLD_UNRESOLVED,
            f"withhold:{blocker_code}",
            f"{blocker}; and no company hypothesis on record — nothing can be said",
        )

    return DispositionVerdict(
        LessonDisposition.SKIP_NO_EVIDENCE,
        "no_evidence",
        "no bear risks, flags, triggers or regime recorded at decision time",
    )


def _why_not_contextual(
    capabilities: frozenset[EvidenceCapability],
    dominant_driver: str,
    regime_shifted: bool | None,
) -> tuple[str, str]:
    """The single blocking reason the contextual path was unavailable.

    Four distinct causes, and conflating them misinforms an operator. Only the
    first is a property of the *outcome*; the rest are properties of what the
    decision recorded or of the run that evaluated it. An earlier version wrote
    "not attributed to the market" unconditionally and so reported exactly that
    of a MARKET-dominated outcome — a sentence contradicting itself.
    """
    if dominant_driver != DRIVER_MARKET:
        return (
            f"non_market:{dominant_driver.lower()}",
            f"driver {dominant_driver}: the move was not attributed to the market",
        )
    if EvidenceCapability.CONTEXT not in capabilities:
        return (
            "no_regime_recorded",
            "market-dominated, but no decision-time regime was recorded, so no "
            "regime can ever match it",
        )
    if regime_shifted is None:
        return (
            "regime_comparison_unknown",
            "market-dominated, but the regime comparison could not be made",
        )
    return (
        "regime_shifted",
        "market-dominated, but the regime shifted after the decision",
    )


# ══════════════════════════════════════════════════════════════════════════════
# Rendering — deterministic, one template per disposition
# ══════════════════════════════════════════════════════════════════════════════
#
# The generator is not asked for these. That is the whole point: a template has
# no slot for FX exposure the bear case never mentions, nor for "rather than
# relying on fundamentals". The failures of August 2026 become unrepresentable
# rather than prohibited.
#
# Every template states the measurement, quotes what was recorded, and stops.
# None proposes a check, names a cause, or suggests what to do next time.

#: Where a review record's hypothesis came from. Deterministic — it reports the
#: *source* of the recorded claim, never an interpretation of it. Asking a model
#: to infer a "topic" is how a causal vocabulary re-enters by the back door.
HYPOTHESIS_TOPIC_BEAR_CASE = "recorded_bear_case"
HYPOTHESIS_TOPIC_RED_FLAGS = "recorded_red_flags"
HYPOTHESIS_TOPIC_KILL_CRITERIA = "recorded_kill_criteria"
HYPOTHESIS_TOPIC_NONE = ""

#: What a price-only retrospective may say about whether a hypothesis came true.
MATERIALIZATION_NOT_EVALUATED = (
    "Whether this materialized was NOT_EVALUATED: no post-decision company "
    "evidence was gathered."
)


@dataclass(frozen=True, slots=True)
class OutcomeFacts:
    """The measured facts a template may state. All deterministic."""

    ticker: str
    days_elapsed: int
    price_return_pct: float
    benchmark_return_pct: float
    excess_return_pct: float
    benchmark_used: str
    market_return_pct: float | None = None
    residual_return_pct: float | None = None
    regime_label: str = ""


def _pct(value: float | None) -> str:
    return "unknown" if value is None else f"{value:+.1f}%"


#: Why a review record exists, keyed on the verdict's reason code. Rendering
#: reads this rather than assuming: `REVIEW_HYPOTHESIS` covers *both* an
#: unattributed outcome and a market-dominated one whose regime cannot authorize
#: an observation, and a fixed sentence for both said "not attributed to the
#: market" of a MARKET-dominated outcome — the same self-contradiction the reason
#: codes were introduced to remove, reappearing one layer down.
_REVIEW_PREAMBLE = {
    "no_regime_recorded": (
        "The outcome was market-dominated, but no decision-time regime was "
        "recorded, so it cannot support a contextual observation."
    ),
    "regime_comparison_unknown": (
        "The outcome was market-dominated, but the regime comparison could not "
        "be made, so it cannot support a contextual observation."
    ),
    "regime_shifted": (
        "The outcome was market-dominated, but the regime shifted afterwards, so "
        "it cannot support a contextual observation."
    ),
}
_REVIEW_PREAMBLE_NON_MARKET = "The outcome is not attributed to the market."

#: Blocker codes whose preamble is the non-market sentence. Enumerated rather
#: than defaulted: `.get(code, NON_MARKET)` would hand the contradictory prose to
#: any *future* market-blocking reason, which is precisely how "not attributed to
#: the market" came to be printed of a MARKET-dominated outcome in the first
#: place. Unknown codes now raise, and the matrix test exercises every code
#: `_why_not_contextual` can emit.
_REVIEW_PREAMBLE_NON_MARKET_CODES = frozenset({"non_market"})


def render_record(
    verdict: DispositionVerdict,
    facts: OutcomeFacts,
    *,
    hypothesis: str = "",
    hypothesis_topic: str = HYPOTHESIS_TOPIC_NONE,
) -> str:
    """The record text for a disposition that produces one.

    Takes the whole verdict, not a bare disposition, so the rendering cannot
    contradict the classification that produced it — which it did when it
    received only the disposition and had to assume why.

    Raises for a disposition that produces nothing, rather than returning "" —
    a silent empty string would let a caller write a blank record and never
    notice. Total over `produces_record`, so a new disposition cannot be added
    without deciding what it says.
    """
    disposition = verdict.disposition
    if not disposition.produces_record:
        raise ValueError(f"{disposition} produces no record; do not render one")

    measured = (
        f"{facts.ticker} returned {_pct(facts.price_return_pct)} against "
        f"{facts.benchmark_used} at {_pct(facts.benchmark_return_pct)} over "
        f"{facts.days_elapsed}d (excess {_pct(facts.excess_return_pct)})."
    )

    if disposition is LessonDisposition.REVIEW_HYPOTHESIS:
        recorded = hypothesis.strip() or "(no hypothesis text recorded)"
        blocker = verdict.reason_code.removeprefix("review:").split(":")[0]
        if blocker in _REVIEW_PREAMBLE_NON_MARKET_CODES:
            preamble = _REVIEW_PREAMBLE_NON_MARKET
        elif blocker in _REVIEW_PREAMBLE:
            preamble = _REVIEW_PREAMBLE[blocker]
        else:
            raise ValueError(
                f"no review preamble for blocker {blocker!r}; add one rather than "
                f"letting it default — a market-blocking reason rendered as "
                f"'not attributed to the market' is the contradiction this "
                f"mapping exists to prevent"
            )
        return (
            f"REVIEW — {measured} {preamble} "
            f"Recorded at decision time ({hypothesis_topic or 'source unrecorded'}): "
            f"{recorded} {MATERIALIZATION_NOT_EVALUATED}"
        )

    if disposition is LessonDisposition.CONTEXTUAL_OBSERVATION:
        regime = facts.regime_label or "an unlabelled regime"
        return (
            f"CONTEXT — under a recorded {regime} regime unchanged across the "
            f"period, {measured} Market leg {_pct(facts.market_return_pct)}, "
            f"residual {_pct(facts.residual_return_pct)}. This describes the "
            f"regime relationship only; it establishes nothing about the company."
        )

    # SPECIAL_SITUATION_REVIEW
    return (
        f"SPECIAL SITUATION — an active tender was on record at decision time, "
        f"so benchmark attribution is not diagnostic for this outcome. {measured}"
    )
