"""The decision matrix: what each combination of evidence and outcome may produce.

Four rounds of lesson-quality fixes each added a rule in a different place —
`lesson_scope_for`, `lesson_eligibility`, `has_grounding_context`, the tender
check, and nine prompt rules — so no single artifact stated the policy and each
new edge case found a gap between them. This file is the policy, exhaustively.

Two stages, and the split is not stylistic. `evidence_capabilities` runs in the
candidate loop **before any price fetch**, so it cannot depend on attribution or
the regime comparison; `derive_disposition` runs after. A single function taking
both would be unimplementable at the first call site.

The named cases are real, from the 2026-08-17 probes, and they are the regression
contract: `2PP.DE` invented FX exposure and `3008.TW` invented momentum and
technical screens, neither appearing anywhere in their recorded evidence.
"""

from __future__ import annotations

import pytest

from src.lesson_disposition import (
    DRIVER_MARKET,
    DRIVER_MIXED,
    DRIVER_RESIDUAL,
    DRIVER_UNKNOWN,
    EvidenceCapability,
    LessonDisposition,
    derive_disposition,
)

HYPOTHESIS = frozenset({EvidenceCapability.HYPOTHESIS})
CONTEXT = frozenset({EvidenceCapability.CONTEXT})
BOTH = frozenset({EvidenceCapability.HYPOTHESIS, EvidenceCapability.CONTEXT})
NONE: frozenset[EvidenceCapability] = frozenset()


def _verdict(capabilities, driver=DRIVER_RESIDUAL, shifted=None, tender=None):
    return derive_disposition(
        capabilities,
        dominant_driver=driver,
        regime_shifted=shifted,
        m_and_a_status=tender,
    )


# ══════════════════════════════════════════════════════════════════════════════
# The matrix
# ══════════════════════════════════════════════════════════════════════════════

_MATRIX = [
    # (label, capabilities, driver, shifted, tender, expected)
    (
        "no evidence at all",
        NONE,
        DRIVER_RESIDUAL,
        None,
        None,
        LessonDisposition.SKIP_NO_EVIDENCE,
    ),
    (
        "no evidence, market-dominated",
        NONE,
        DRIVER_MARKET,
        False,
        None,
        LessonDisposition.SKIP_NO_EVIDENCE,
    ),
    (
        "hypothesis, residual outcome",
        HYPOTHESIS,
        DRIVER_RESIDUAL,
        None,
        None,
        LessonDisposition.REVIEW_HYPOTHESIS,
    ),
    (
        "hypothesis, mixed outcome",
        HYPOTHESIS,
        DRIVER_MIXED,
        None,
        None,
        LessonDisposition.REVIEW_HYPOTHESIS,
    ),
    (
        "hypothesis, market outcome but no regime",
        HYPOTHESIS,
        DRIVER_MARKET,
        None,
        None,
        LessonDisposition.REVIEW_HYPOTHESIS,
    ),
    (
        "context only, residual outcome",
        CONTEXT,
        DRIVER_RESIDUAL,
        False,
        None,
        LessonDisposition.WITHHOLD_UNRESOLVED,
    ),
    (
        "context only, market outcome, stable regime",
        CONTEXT,
        DRIVER_MARKET,
        False,
        None,
        LessonDisposition.CONTEXTUAL_OBSERVATION,
    ),
    (
        "context only, market outcome, shifted regime",
        CONTEXT,
        DRIVER_MARKET,
        True,
        None,
        LessonDisposition.WITHHOLD_UNRESOLVED,
    ),
    (
        "context only, market outcome, unknown shift",
        CONTEXT,
        DRIVER_MARKET,
        None,
        None,
        LessonDisposition.WITHHOLD_UNRESOLVED,
    ),
    (
        "both, market outcome, stable regime",
        BOTH,
        DRIVER_MARKET,
        False,
        None,
        LessonDisposition.CONTEXTUAL_OBSERVATION,
    ),
    (
        "both, residual outcome",
        BOTH,
        DRIVER_RESIDUAL,
        False,
        None,
        LessonDisposition.REVIEW_HYPOTHESIS,
    ),
    (
        "both, market outcome, shifted regime",
        BOTH,
        DRIVER_MARKET,
        True,
        None,
        LessonDisposition.REVIEW_HYPOTHESIS,
    ),
    (
        "tender outranks a clean contextual case",
        BOTH,
        DRIVER_MARKET,
        False,
        "ACTIVE_TENDER",
        LessonDisposition.SPECIAL_SITUATION_REVIEW,
    ),
    (
        "a rumour does not pin the price",
        BOTH,
        DRIVER_MARKET,
        False,
        "RUMORED",
        LessonDisposition.CONTEXTUAL_OBSERVATION,
    ),
    (
        "unknown driver with a hypothesis",
        HYPOTHESIS,
        DRIVER_UNKNOWN,
        None,
        None,
        LessonDisposition.REVIEW_HYPOTHESIS,
    ),
]


@pytest.mark.parametrize(
    ("capabilities", "driver", "shifted", "tender", "expected"),
    [row[1:] for row in _MATRIX],
    ids=[row[0] for row in _MATRIX],
)
def test_the_matrix(capabilities, driver, shifted, tender, expected):
    assert _verdict(capabilities, driver, shifted, tender).disposition is expected


def test_the_matrix_covers_every_disposition():
    """A policy with an unreachable state is a policy with a dead branch."""
    covered = {row[5] for row in _MATRIX}
    assert covered == set(LessonDisposition), (
        f"unexercised dispositions: {set(LessonDisposition) - covered}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# The properties the matrix exists to protect
# ══════════════════════════════════════════════════════════════════════════════


class TestInjectabilityIsNarrow:
    def test_only_a_stable_market_observation_is_injectable(self):
        injectable = [
            row[0]
            for row in _MATRIX
            if _verdict(row[1], row[2], row[3], row[4]).is_injectable
        ]
        assert injectable == [
            "context only, market outcome, stable regime",
            "both, market outcome, stable regime",
            "a rumour does not pin the price",
        ]

    @pytest.mark.parametrize("shifted", [True, None])
    def test_an_unestablished_regime_is_not_a_stable_one(self, shifted):
        """`is False`, never `is not True`.

        `None` means the comparison could not be made — a stale macro cache, a
        changed summarizer prompt. Reading that as stability authorizes guidance
        about a regime nobody verified still holds.
        """
        assert not _verdict(BOTH, DRIVER_MARKET, shifted).is_injectable


class TestCapabilitiesDoNotSubstituteForEachOther:
    """The conflation a single grounding boolean could not express.

    A recorded macro regime licenses an observation about the regime. It does not
    license a claim about the company — which is how a regime-only snapshot came
    to produce company-mechanism prose.
    """

    def test_context_alone_never_yields_a_company_review(self):
        for driver in (DRIVER_RESIDUAL, DRIVER_MIXED, DRIVER_UNKNOWN):
            assert (
                _verdict(CONTEXT, driver).disposition
                is LessonDisposition.WITHHOLD_UNRESOLVED
            )

    def test_hypothesis_alone_never_yields_a_contextual_observation(self):
        for shifted in (True, False, None):
            assert not _verdict(HYPOTHESIS, DRIVER_MARKET, shifted).is_injectable

    def test_holding_both_does_not_promote_the_hypothesis_into_a_cause(self):
        """Precedence documented: contextual outcomes stay contextual.

        A snapshot can carry a bear case *and* a regime. When the outcome is
        market-dominated, the bear case being coincidentally on file must not
        convert it into an explanation.
        """
        assert (
            _verdict(BOTH, DRIVER_MARKET, False).disposition
            is LessonDisposition.CONTEXTUAL_OBSERVATION
        )


class TestWithholdingWritesNothing:
    """A durable store full of "cause unresolved" documents is noise.

    The honest artifact for an unattributed outcome with no company hypothesis is
    the run counter, not a lesson — otherwise a legacy sweep fills the corpus
    with repetitive non-actionable text.
    """

    def test_withheld_and_skipped_produce_no_record(self):
        for disposition in (
            LessonDisposition.WITHHOLD_UNRESOLVED,
            LessonDisposition.SKIP_NO_EVIDENCE,
        ):
            assert not disposition.produces_record

    def test_the_three_review_or_observation_kinds_do(self):
        for disposition in (
            LessonDisposition.CONTEXTUAL_OBSERVATION,
            LessonDisposition.REVIEW_HYPOTHESIS,
            LessonDisposition.SPECIAL_SITUATION_REVIEW,
        ):
            assert disposition.produces_record


class TestReasonsAreMachineReadableAndHonest:
    def test_every_verdict_carries_a_code_and_prose(self):
        for row in _MATRIX:
            verdict = _verdict(row[1], row[2], row[3], row[4])
            assert verdict.reason_code and verdict.reason, row[0]

    def test_a_market_outcome_is_never_described_as_non_market(self):
        """The bug the matrix caught.

        The review branch assumed non-market attribution, but it is also reached
        when the outcome *was* market-dominated and the regime could not
        authorize an observation — so it reported "not attributed to the market"
        of a MARKET-dominated outcome, a sentence contradicting itself.
        """
        for shifted in (True, None):
            verdict = _verdict(BOTH, DRIVER_MARKET, shifted)
            assert "not attributed to the market" not in verdict.reason
            assert "market-dominated" in verdict.reason

    def test_the_four_blocking_causes_are_distinguishable(self):
        codes = {
            _verdict(BOTH, DRIVER_RESIDUAL, False).reason_code,
            _verdict(HYPOTHESIS, DRIVER_MARKET, False).reason_code,
            _verdict(BOTH, DRIVER_MARKET, None).reason_code,
            _verdict(BOTH, DRIVER_MARKET, True).reason_code,
        }
        assert len(codes) == 4, f"blocking causes collapsed: {codes}"

    def test_the_shift_detail_reaches_both_regime_blocked_dispositions(self):
        """ "The cache was 40d old" describes the run, not the snapshot.

        It belongs on the review record *and* the withheld one; an earlier
        version attached it to only the latter.
        """
        for capabilities in (BOTH, CONTEXT):
            verdict = derive_disposition(
                capabilities,
                dominant_driver=DRIVER_MARKET,
                regime_shifted=None,
                regime_shift_reason="cached macro brief is 40d old",
            )
            assert "40d old" in verdict.reason


class TestTheProbeCasesAreTheRegressionContract:
    """The two lessons that failed the acceptance gate on 2026-08-17.

    Neither may reach a free-form generator again. Under this policy both are
    REVIEW_HYPOTHESIS — a record that may quote the recorded bear case and
    nothing else. Step C makes that structural by rendering from a template;
    this asserts the classification that routes them there.
    """

    def test_2pp_de_is_a_review_record_not_a_lesson(self):
        """Residual-dominated, bear case on record, no regime recorded.

        Its recorded evidence is US revenue 56.88%, analyst coverage, ADR
        listing, an eroding checkout moat. It invented FX and sector momentum.
        """
        verdict = _verdict(HYPOTHESIS, DRIVER_RESIDUAL, None)
        assert verdict.disposition is LessonDisposition.REVIEW_HYPOTHESIS
        assert not verdict.is_injectable

    def test_3008_tw_is_a_review_record_not_a_lesson(self):
        """Its evidence covers analyst coverage, growth, margins, governance.

        It invented liquidity screens, momentum screens and a technical
        breakout — none of which appear in the record.
        """
        verdict = _verdict(HYPOTHESIS, DRIVER_RESIDUAL, None)
        assert verdict.disposition is LessonDisposition.REVIEW_HYPOTHESIS

    def test_2727_tw_the_first_market_dominated_outcome_stays_review_only(self):
        """MARKET-dominated, but legacy: no decision-time regime was recorded.

        The first such outcome in four probes, and it must not become injectable
        on attribution alone.
        """
        verdict = _verdict(HYPOTHESIS, DRIVER_MARKET, None)
        assert not verdict.is_injectable
        assert "no decision-time regime" in verdict.reason


class TestStageOneGatesWhatStageTwoWouldOtherwiseDecide:
    """The policy is total; the pipeline is not. Both facts must be recorded.

    `derive_disposition` answers correctly for any input, including an active
    tender with no other evidence. But `has_grounding_context` — stage 1 — runs
    in the candidate loop before pricing and does not treat a tender as evidence,
    so that combination never reaches stage 2 in the live flow.

    Measured 2026-08-17: **0 of 7,952 snapshots** are an active tender with no
    bear text, flags, triggers or regime; all 10 tender snapshots carry other
    evidence and are classified normally. So this is documented rather than
    engineered around — promoting a tender to a third capability would add a
    pre-pricing path for a case that has never occurred, and would price a
    snapshot to emit a record whose content is fixed before the price is known.

    If it ever occurs, the one-line change is to add SPECIAL_SITUATION to
    `evidence_capabilities`. This test is what will fail first.
    """

    def test_the_policy_answers_in_isolation(self):
        assert (
            _verdict(NONE, DRIVER_RESIDUAL, None, "ACTIVE_TENDER").disposition
            is LessonDisposition.SPECIAL_SITUATION_REVIEW
        )

    def test_but_stage_one_never_admits_it(self):
        from src.retrospective import evidence_capabilities, has_grounding_context

        snapshot = {"m_and_a_status": "ACTIVE_TENDER"}
        assert evidence_capabilities(snapshot) == frozenset()
        assert not has_grounding_context(snapshot), (
            "a tender is not, by itself, evidence a lesson can be drawn from"
        )

    def test_a_tender_with_evidence_does_reach_stage_two(self):
        """The 10 real cases: classified by tender, not by attribution."""
        from src.retrospective import has_grounding_context

        snapshot = {
            "m_and_a_status": "ACTIVE_TENDER",
            "bear_risks_excerpt": "1. Deal break risk.",
        }
        assert has_grounding_context(snapshot)
        assert (
            _verdict(HYPOTHESIS, DRIVER_RESIDUAL, None, "ACTIVE_TENDER").disposition
            is LessonDisposition.SPECIAL_SITUATION_REVIEW
        )


class TestParityWithThePreConsolidationLogic:
    """B was worth doing separately only if it changed no classification.

    The pre-refactor decision was spread across `lesson_eligibility`'s own
    inlined checks. That logic is reproduced here verbatim — the standard parity
    shape, as `tests/llm_runtime/test_legacy_parity.py` does for the two LLM
    configuration schemas — so "consolidation preserved behaviour" is a committed
    assertion rather than a one-off claim someone made once in a terminal.

    Synthetic shapes rather than the operator's corpus, because CI has neither
    `results/` nor the archive. `scripts/retrospective_evidence_audit.py
    --parity` runs the same comparison over real artifacts.
    """

    @staticmethod
    def _pre_consolidation_eligibility(comparison) -> str:
        from src.retrospective import (
            LESSON_ELIGIBILITY_INJECTABLE,
            LESSON_ELIGIBILITY_REVIEW_ONLY,
            REGIME_COMPARED_FIELDS,
        )

        if str(comparison.get("m_and_a_status") or "").strip().upper() == (
            "ACTIVE_TENDER"
        ):
            return LESSON_ELIGIBILITY_REVIEW_ONLY
        attribution = comparison.get("attribution")
        attribution = attribution if isinstance(attribution, dict) else {}
        if str(attribution.get("dominant_driver") or DRIVER_UNKNOWN) != DRIVER_MARKET:
            return LESSON_ELIGIBILITY_REVIEW_ONLY
        regime = comparison.get("regime_at_decision")
        regime = regime if isinstance(regime, dict) else {}
        if not any(
            str(regime.get(field) or "").strip() for field in REGIME_COMPARED_FIELDS
        ):
            return LESSON_ELIGIBILITY_REVIEW_ONLY
        delta = comparison.get("cached_regime_delta")
        delta = delta if isinstance(delta, dict) else {}
        if delta.get("shifted") is not False:
            return LESSON_ELIGIBILITY_REVIEW_ONLY
        return LESSON_ELIGIBILITY_INJECTABLE

    def test_every_shape_agrees(self):
        from src.retrospective import lesson_eligibility

        regimes = [
            {"risk_appetite": "RISK_ON", "shock_type": "NONE"},
            {"risk_appetite": "", "shock_type": ""},
            None,
            "corrupted",
        ]
        bears = ["1. Cyclical exposure.", "BEAR CASE SUMMARY**:", "", None]
        combinations = 0
        for driver in (DRIVER_MARKET, DRIVER_RESIDUAL, DRIVER_MIXED, DRIVER_UNKNOWN):
            for shifted in (True, False, None, "not-a-bool"):
                for regime in regimes:
                    for bear in bears:
                        for tender in (None, "ACTIVE_TENDER", "RUMORED", "NONE"):
                            comparison = {
                                "attribution": {"dominant_driver": driver},
                                "regime_at_decision": regime,
                                "cached_regime_delta": {"shifted": shifted},
                                "bear_risks_excerpt": bear,
                                "m_and_a_status": tender,
                            }
                            combinations += 1
                            assert lesson_eligibility(comparison)[0] == (
                                self._pre_consolidation_eligibility(comparison)
                            ), comparison
        assert combinations == 1024, "the shape matrix shrank; parity is weaker"
