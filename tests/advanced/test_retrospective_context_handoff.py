"""Every stage must hand the next one its context, or a field silently vanishes.

The individual stages are each well covered, and that is exactly how the FX
defect survived: `attribute_return` and `_render_attribution` were tested in
isolation while the site that *decided* the value was not, so reverting the
initializer to `0.0` left the whole suite green.

This drives the full chain for one analysis:

    result.red_flags
      -> extract_snapshot()          red_flags_at_decision
      -> compare_to_reality()        carried through **snapshot
      -> store_lesson()              flat ChromaDB metadata
      -> format_lessons_for_injection()   rendered observation text

and asserts a field present at the first stage is readable at the last. The
fixture deliberately carries the awkward combinations — an active tender, a
non-USD currency whose FX cannot be determined, a fallback currency resolution,
and a macro region — because those are the ones a refactor drops.
"""

from __future__ import annotations

import pytest

from src.retrospective import (
    FX_UNAVAILABLE,
    LESSON_ELIGIBILITY_REVIEW_ONLY,
    THESIS_NOT_EVALUATED,
    _observation_lines,
    compare_to_reality,
    extract_snapshot,
    format_lessons_for_injection,
    lesson_eligibility,
    store_lesson,
)
from tests.advanced.retrospective_fakes import (
    FakeLessonsMemory,
    make_snapshot,
    yfinance_ticker_stub,
)
from tests.advanced.test_retrospective_retrieval import _QueryableMemory

_FLAGS = [
    {"type": "CMIC_FLAGGED", "severity": "CRITICAL", "detail": "long prose"},
    {"type": "PFIC_UNCERTAIN", "severity": "WARNING", "rationale": "more prose"},
    # Duplicate + malformed entries must not reach the record.
    {"type": "CMIC_FLAGGED"},
    {"severity": "HIGH"},
    "not a mapping",
]


def _graph_result() -> dict:
    return {
        "final_trade_decision": "PORTFOLIO MANAGER VERDICT: BUY",
        "fundamentals_report": "",
        "red_flags": _FLAGS,
        "macro_context_region": "APAC",
    }


class TestFlagsSurviveEveryStage:
    def test_extract_captures_types_only_and_deduplicates(self):
        snapshot = extract_snapshot(_graph_result(), "7203.T")
        assert snapshot["red_flags_at_decision"] == [
            "CMIC_FLAGGED",
            "PFIC_UNCERTAIN",
        ], "types only, sorted, deduplicated, malformed entries dropped"

    def test_the_explanatory_prose_is_not_carried(self):
        """A hazard is a recorded fact; its explanation is not one.

        Retaining `detail`/`rationale` would smuggle the most causally-loaded
        sentence in the codebase into a lesson record as though it were evidence.
        """
        snapshot = extract_snapshot(_graph_result(), "7203.T")
        blob = repr(snapshot["red_flags_at_decision"])
        assert "long prose" not in blob and "more prose" not in blob

    @pytest.mark.asyncio
    async def test_flags_and_context_reach_the_rendered_observation(self, monkeypatch):
        # --- stage 1: extract ------------------------------------------------
        snapshot = make_snapshot(age_days=200)
        # `extract_snapshot` stamps `analysis_date` with *today*, so the aged date
        # has to be restored after merging or the snapshot falls under
        # MINIMUM_DAYS_ELAPSED and never triggers — which is a fixture bug that
        # reads exactly like a broken handoff, hence the assert below.
        aged_date = snapshot["analysis_date"]
        snapshot.update(extract_snapshot(_graph_result(), "7203.T"))
        snapshot.update(
            {
                "ticker": "7203.T",
                "analysis_date": aged_date,
                "verdict": "BUY",
                "currency": "JPY",
                "currency_source": "fallback_bare_ticker",
                "fx_rate_to_usd": None,  # -> FX_UNAVAILABLE
                "benchmark_index": "^N225",
                "macro_region": "APAC",
                "m_and_a_status": "ACTIVE_TENDER",
                "sector": "Consumer Discretionary",
            }
        )

        # --- stage 2: compare ------------------------------------------------
        # `import yfinance as yf` is inside the fetch function, so
        # `src.retrospective.yf` is not a seam: patching it does nothing and this
        # test was reaching the live network, passing only because 7203.T is a
        # real ticker with real data.
        import yfinance

        monkeypatch.setattr(
            yfinance,
            "Ticker",
            yfinance_ticker_stub(stock=(1000.0, 500.0), benchmark=(30000.0, 29000.0)),
        )
        comparison = await compare_to_reality(snapshot)
        assert comparison is not None, "fixture must trigger or this asserts nothing"
        assert comparison["red_flags_at_decision"] == ["CMIC_FLAGGED", "PFIC_UNCERTAIN"]
        assert comparison["fx_observation"] == FX_UNAVAILABLE
        assert comparison["macro_region"] == "APAC"
        assert comparison["currency_source"] == "fallback_bare_ticker"

        # An active tender prices against deal terms, so the outcome is not a
        # market observation however the legs decompose.
        eligibility, reason = lesson_eligibility(comparison)
        assert eligibility == LESSON_ELIGIBILITY_REVIEW_ONLY
        assert "tender" in reason

        # --- stage 3: store --------------------------------------------------
        memory = _QueryableMemory()
        stored = await store_lesson(
            "Verify the ownership chain before trusting a low multiple.",
            "missed_risk",
            "REGULATORY_SHIFT",
            comparison,
            0.8,
            memory,
        )
        assert stored is True
        meta = memory.metadatas()[0]
        assert meta["red_flags_at_decision"] == "CMIC_FLAGGED,PFIC_UNCERTAIN"
        assert meta["fx_observation"] == FX_UNAVAILABLE
        assert meta["macro_region"] == "APAC"
        assert meta["m_and_a_status"] == "ACTIVE_TENDER"
        assert meta["currency_source"] == "fallback_bare_ticker"
        assert meta["benchmark_used"] == "^N225"
        assert meta["thesis_validation_status"] == THESIS_NOT_EVALUATED
        # ChromaDB metadata takes no nested values.
        assert all(
            isinstance(value, str | int | float | bool)
            for value in meta.values()
            if value is not None
        )

        # --- stage 4: render -------------------------------------------------
        # Fed the real stored metadata rather than driven through
        # `format_lessons_for_injection`. This record is correctly withheld by
        # *two* independent gates (UNRESOLVED scope and REVIEW_ONLY eligibility),
        # so routing through retrieval would assert an empty string and prove
        # nothing about the handoff. Those gates have their own suites; this
        # asserts the last hop — stored metadata to rendered text.
        text = "\n".join(_observation_lines(dict(meta)))
        assert "CMIC_FLAGGED,PFIC_UNCERTAIN" in text
        assert "not evidence they occurred or caused this outcome" in text
        assert "vs ^N225" in text
        assert FX_UNAVAILABLE in text
        assert "region APAC" in text
        assert "currency source fallback_bare_ticker" in text
        assert "special situation ACTIVE_TENDER" in text
        assert "unvalidated price-only classification" in text, (
            "FAILURE_MODE renders beside the prose and must not read as a finding"
        )


class TestALegacyRecordStaysQuiet:
    """Absent context must render nothing, never a default that asserts something."""

    @pytest.mark.asyncio
    async def test_no_invented_defaults(self):
        memory = _QueryableMemory()
        memory.seed(
            "An older lesson.",
            ticker="9984.T",
            sector="Industrials",
            exchange="T",
            currency="JPY",
            lesson_type="missed_risk",
            failure_mode="MACRO_REGIME",
            confidence_weight=0.8,
            lesson_eligibility="INJECTABLE",
        )
        text = await format_lessons_for_injection(
            memory,
            "7203.T",
            "Industrials",
            current_regime={"risk_appetite": "RISK_ON", "confidence": "HIGH"},
        )
        assert "An older lesson." in text
        assert "decision context:" not in text
        assert "hazards recorded" not in text
        assert "observed over" not in text
        assert "region " not in text
