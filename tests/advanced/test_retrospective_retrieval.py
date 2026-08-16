"""Step 9: retrieve a lesson only where it applies, and label what it cannot claim.

Three defects in the retrieval path:

* A ``CONTEXTUAL`` lesson asserts something about a *particular* regime. Nothing
  checked the regime before injecting it — which is how a rule induced from a
  20% benchmark crash would have been handed to an analysis running in a calm
  market.
* ``n_results=5`` against a top-3 cut left the boost/floor machinery almost
  nothing to rank; and ``get_relevant_lessons`` accepted a ``ticker`` it never
  used.
* "Have I screened this ticker out before?" is an exact-match fact, and it was
  left to embedding similarity, where it can simply lose a slot.

``UNRESOLVED`` lessons are **stored but never injected**. An earlier revision
injected them with a caveat appended; measured against a real batch, 32 of 47
stored lessons (68%) were UNRESOLVED, and a caveat beside an LLM-authored
imperative does not stop it acting as one. They stay retrievable for review and
for promotion once an evidence-backed post-mortem exists.
"""

from __future__ import annotations

import pytest

from src.retrospective import (
    LESSON_QUERY_CANDIDATES,
    SCOPE_CONTEXTUAL,
    SCOPE_UNRESOLVED,
    UNRESOLVED_LESSON_MARKER,
    _regime_matches,
    format_lessons_for_injection,
    get_relevant_lessons,
)
from tests.advanced.retrospective_fakes import FakeLessonsMemory


class _QueryableMemory(FakeLessonsMemory):
    """Adds the async vector-query surface `get_relevant_lessons` calls."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.query_calls: list[dict] = []

    async def query_similar_situations(self, query_text: str, n_results: int = 5):
        self.query_calls.append({"query_text": query_text, "n_results": n_results})
        payload = self.situation_collection.query(n_results=n_results)
        return [
            {"document": document, "metadata": metadata, "distance": 0.1}
            for document, metadata in zip(
                payload["documents"][0], payload["metadatas"][0], strict=False
            )
        ]


def _lesson(**meta) -> dict:
    base = {
        "ticker": "OTHER.T",
        "sector": "Industrials",
        "exchange": "T",
        "currency": "JPY",
        "lesson_type": "missed_risk",
        "failure_mode": "MACRO_REGIME",
        "confidence_weight": 0.8,
    }
    base.update(meta)
    return base


RISK_OFF_NOW = {
    "risk_appetite": "RISK_OFF",
    "shock_type": "RATES",
    "confidence": "HIGH",
}
RISK_ON_NOW = {"risk_appetite": "RISK_ON", "shock_type": "NONE", "confidence": "HIGH"}


# ══════════════════════════════════════════════════════════════════════════════
# The regime predicate
# ══════════════════════════════════════════════════════════════════════════════


class TestRegimeMatching:
    def test_matching_risk_appetite_is_enough(self):
        assert _regime_matches(RISK_OFF_NOW, {"regime_risk_appetite": "RISK_OFF"})

    def test_matching_shock_type_is_enough(self):
        assert _regime_matches(RISK_OFF_NOW, {"regime_shock_type": "RATES"})

    def test_neither_matching_is_no_match(self):
        assert not _regime_matches(
            RISK_OFF_NOW,
            {"regime_risk_appetite": "RISK_ON", "regime_shock_type": "NONE"},
        )

    def test_an_unknown_current_regime_never_matches(self):
        """No regime to compare against is no basis to apply a scoped lesson."""
        assert not _regime_matches(None, {"regime_risk_appetite": "RISK_OFF"})
        assert not _regime_matches({}, {"regime_risk_appetite": "RISK_OFF"})

    def test_a_lesson_with_no_regime_metadata_never_matches(self):
        assert not _regime_matches(RISK_OFF_NOW, {})

    def test_matching_is_case_and_whitespace_insensitive(self):
        assert _regime_matches(
            {"risk_appetite": " risk_off "}, {"regime_risk_appetite": "RISK_OFF"}
        )

    def test_malformed_metadata_does_not_raise(self):
        assert not _regime_matches(RISK_OFF_NOW, {"regime_risk_appetite": 12345})


# ══════════════════════════════════════════════════════════════════════════════
# Scoped injection
# ══════════════════════════════════════════════════════════════════════════════


class TestContextualLessonsAreRegimeScoped:
    @pytest.mark.asyncio
    async def test_a_matching_regime_admits_the_lesson(self):
        memory = _QueryableMemory()
        memory.seed(
            "Rate shocks compress multiples first.",
            **_lesson(lesson_scope=SCOPE_CONTEXTUAL, regime_risk_appetite="RISK_OFF"),
        )
        text = await format_lessons_for_injection(
            memory, "7203.T", "Industrials", current_regime=RISK_OFF_NOW
        )
        assert "Rate shocks compress multiples first." in text

    @pytest.mark.asyncio
    async def test_a_different_regime_skips_it(self):
        """The 001060.KS class of harm: a crash-era rule in a calm market."""
        memory = _QueryableMemory()
        memory.seed(
            "Defensive names outperform, so relax valuation discipline.",
            **_lesson(lesson_scope=SCOPE_CONTEXTUAL, regime_risk_appetite="RISK_OFF"),
        )
        text = await format_lessons_for_injection(
            memory, "7203.T", "Industrials", current_regime=RISK_ON_NOW
        )
        assert text == ""

    @pytest.mark.asyncio
    async def test_no_current_regime_skips_contextual_lessons(self):
        memory = _QueryableMemory()
        memory.seed(
            "Rate shocks compress multiples.",
            **_lesson(lesson_scope=SCOPE_CONTEXTUAL, regime_risk_appetite="RISK_OFF"),
        )
        assert (
            await format_lessons_for_injection(
                memory, "7203.T", "Industrials", current_regime=None
            )
            == ""
        )

    @pytest.mark.asyncio
    async def test_a_legacy_lesson_without_a_scope_is_unaffected(self):
        """Scope-less records must behave exactly as before."""
        memory = _QueryableMemory()
        memory.seed("An older lesson.", **_lesson())
        text = await format_lessons_for_injection(
            memory, "7203.T", "Industrials", current_regime=RISK_ON_NOW
        )
        assert "An older lesson." in text

    @pytest.mark.asyncio
    async def test_a_malformed_scope_is_treated_as_legacy(self):
        memory = _QueryableMemory()
        memory.seed("An odd lesson.", **_lesson(lesson_scope=12345))
        text = await format_lessons_for_injection(
            memory, "7203.T", "Industrials", current_regime=None
        )
        assert "An odd lesson." in text


class TestUnresolvedLessonsAreStoredButNeverInjected:
    """Retained for review; withheld from live guidance.

    An earlier revision injected these with a caveat appended, reasoning that
    this repo labels unknowns rather than deleting them. Measured against a real
    batch that reasoning does not survive: **32 of 47 stored lessons (68%) were
    UNRESOLVED**, and the stored text is an LLM-authored imperative that a
    neighbouring caveat does not neutralize — while the underlying move may have
    been an earnings surprise, a takeover rumour, sector rotation or a data
    error. The repo's actual precedent is narrower than "label everything":
    `*_SCORE_UNRELIABLE` withholds authority, and so does this.
    """

    @pytest.mark.asyncio
    async def test_it_is_never_injected(self):
        memory = _QueryableMemory()
        memory.seed(
            "Avoid early margin-recovery stories.",
            **_lesson(lesson_scope=SCOPE_UNRESOLVED, confidence_weight=0.95),
        )
        text = await format_lessons_for_injection(
            memory, "7203.T", "Industrials", current_regime=RISK_ON_NOW
        )
        assert text == ""

    @pytest.mark.asyncio
    async def test_not_even_under_a_matching_regime(self):
        """Regime match is irrelevant — the *cause* is what is unknown."""
        memory = _QueryableMemory()
        memory.seed(
            "Avoid early margin-recovery stories.",
            **_lesson(lesson_scope=SCOPE_UNRESOLVED, regime_risk_appetite="RISK_ON"),
        )
        text = await format_lessons_for_injection(
            memory, "7203.T", "Industrials", current_regime=RISK_ON_NOW
        )
        assert text == ""

    @pytest.mark.asyncio
    async def test_it_is_still_retrievable_for_review(self):
        """Withheld from injection is not deleted — promotion stays possible."""
        memory = _QueryableMemory()
        memory.seed(
            "Avoid early margin-recovery stories.",
            **_lesson(lesson_scope=SCOPE_UNRESOLVED),
        )
        results = await get_relevant_lessons(memory, "Industrials", "7203.T")
        assert len(results) == 1
        assert results[0]["metadata"]["lesson_scope"] == SCOPE_UNRESOLVED

    @pytest.mark.asyncio
    async def test_it_does_not_crowd_out_an_injectable_lesson(self):
        """Filtered before ranking, so it cannot consume a top-3 slot."""
        memory = _QueryableMemory()
        for i in range(5):
            memory.seed(
                f"Unresolved {i}.",
                **_lesson(lesson_scope=SCOPE_UNRESOLVED, confidence_weight=0.99),
            )
        memory.seed(
            "The one that applies.",
            **_lesson(
                lesson_scope=SCOPE_CONTEXTUAL,
                regime_risk_appetite="RISK_ON",
                confidence_weight=0.5,
            ),
        )
        text = await format_lessons_for_injection(
            memory, "7203.T", "Industrials", current_regime=RISK_ON_NOW
        )
        assert "The one that applies." in text
        assert "Unresolved" not in text

    @pytest.mark.asyncio
    async def test_a_contextual_lesson_is_unaffected(self):
        memory = _QueryableMemory()
        memory.seed(
            "A regime lesson.",
            **_lesson(lesson_scope=SCOPE_CONTEXTUAL, regime_risk_appetite="RISK_ON"),
        )
        text = await format_lessons_for_injection(
            memory, "7203.T", "Industrials", current_regime=RISK_ON_NOW
        )
        assert "A regime lesson." in text
        assert UNRESOLVED_LESSON_MARKER not in text

    @pytest.mark.asyncio
    async def test_a_prior_rejection_is_unaffected(self):
        """The exact-match screening fact is not a generalized rule."""
        memory = _QueryableMemory()
        memory.seed(
            "PRIOR SCREENING RECORD: 7203.T — DO_NOT_INITIATE.",
            **_lesson(ticker="7203.T", lesson_type="prior_rejection"),
        )
        text = await format_lessons_for_injection(
            memory, "7203.T", "Industrials", current_regime=RISK_ON_NOW
        )
        assert "PRIOR REJECTION (7203.T)" in text


# ══════════════════════════════════════════════════════════════════════════════
# Candidate sourcing
# ══════════════════════════════════════════════════════════════════════════════


class TestCandidateSourcing:
    @pytest.mark.asyncio
    async def test_the_ticker_reaches_the_query(self):
        """It was accepted and discarded; a sector-only query is cross-listing."""
        memory = _QueryableMemory()
        await get_relevant_lessons(memory, "Industrials", "7203.T")
        assert "7203.T" in memory.query_calls[0]["query_text"]

    @pytest.mark.asyncio
    async def test_the_candidate_pool_is_wide_enough_to_rank(self):
        memory = _QueryableMemory()
        await get_relevant_lessons(memory, "Industrials", "7203.T")
        assert memory.query_calls[0]["n_results"] == LESSON_QUERY_CANDIDATES
        assert LESSON_QUERY_CANDIDATES > 3, "top-3 of 3 is not a ranking"

    @pytest.mark.asyncio
    async def test_a_same_ticker_rejection_is_fetched_deterministically(self):
        """It must arrive even when the vector query returns nothing."""
        memory = _QueryableMemory()
        memory.seed(
            "PRIOR SCREENING RECORD: 7203.T — DO_NOT_INITIATE.",
            **_lesson(ticker="7203.T", lesson_type="prior_rejection"),
        )

        async def _empty(query_text: str, n_results: int = 5):
            return []

        memory.query_similar_situations = _empty
        results = await get_relevant_lessons(memory, "Industrials", "7203.T")
        assert len(results) == 1
        assert results[0]["metadata"]["ticker"] == "7203.T"

    @pytest.mark.asyncio
    async def test_it_is_not_duplicated_when_the_vector_query_also_returns_it(self):
        memory = _QueryableMemory()
        memory.seed(
            "PRIOR SCREENING RECORD: 7203.T — DO_NOT_INITIATE.",
            **_lesson(ticker="7203.T", lesson_type="prior_rejection"),
        )
        results = await get_relevant_lessons(memory, "Industrials", "7203.T")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_another_tickers_rejection_is_not_force_fetched(self):
        memory = _QueryableMemory()
        memory.seed(
            "PRIOR SCREENING RECORD: OTHER.T",
            **_lesson(ticker="OTHER.T", lesson_type="prior_rejection"),
        )

        async def _empty(query_text: str, n_results: int = 5):
            return []

        memory.query_similar_situations = _empty
        assert await get_relevant_lessons(memory, "Industrials", "7203.T") == []

    @pytest.mark.asyncio
    async def test_a_rejection_record_reaches_injection(self):
        memory = _QueryableMemory()
        memory.seed(
            "PRIOR SCREENING RECORD: 7203.T — DO_NOT_INITIATE on 2026-01-01.",
            **_lesson(
                ticker="7203.T", lesson_type="prior_rejection", confidence_weight=0.3
            ),
        )
        text = await format_lessons_for_injection(
            memory, "7203.T", "Industrials", current_regime=RISK_ON_NOW
        )
        assert "PRIOR REJECTION (7203.T)" in text

    @pytest.mark.asyncio
    async def test_a_raising_metadata_fetch_degrades_to_the_vector_query(self):
        memory = _QueryableMemory()
        memory.seed("A vector lesson.", **_lesson())

        def _boom(*_a, **_k):
            raise RuntimeError("chroma is down")

        memory.situation_collection.get = _boom
        results = await get_relevant_lessons(memory, "Industrials", "7203.T")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_an_empty_store_is_not_an_error(self):
        memory = _QueryableMemory()
        assert await get_relevant_lessons(memory, "Industrials", "7203.T") == []

    @pytest.mark.asyncio
    async def test_an_unavailable_store_returns_nothing(self):
        memory = _QueryableMemory(available=False)
        assert await get_relevant_lessons(memory, "Industrials", "7203.T") == []


class TestNoDoubleCountedScoringTerms:
    """Quick/strict and signal strength are priced at write time already."""

    @pytest.mark.asyncio
    async def test_quick_mode_is_not_penalized_a_second_time(self):
        memory = _QueryableMemory()
        memory.seed("Lesson A.", **_lesson(confidence_weight=0.8, is_quick_mode=True))
        memory.seed("Lesson B.", **_lesson(confidence_weight=0.8, is_quick_mode=False))
        text = await format_lessons_for_injection(
            memory, "7203.T", "Industrials", current_regime=RISK_ON_NOW
        )
        assert "conf: 1.05" in text
        assert text.count("conf: 1.05") == 2, (
            "identical stored weights must rank identically; a retrieval-time "
            "quick-mode penalty would double-count save_rejection_record's"
        )
