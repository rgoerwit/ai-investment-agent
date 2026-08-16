"""Step 8: show the lesson LLM what the deterministic layer already knows.

``regime_at_decision`` was recorded in every snapshot and never passed to the
model — which was nonetheless asked to choose between ``MACRO_REGIME`` and
``OPERATIONAL_MISS``. That is how ``001060.KS`` produced a rule to relax
valuation discipline out of a 20% benchmark crash.

Every assertion here reads the **captured prompt string**, because that is the
artifact under test: what the model was actually told. A mock that only records
"generate_lesson was called" cannot express any of it.

Also closes a real gap: ``compute_confidence`` applied a quick-mode factor but no
strict-mode factor, while ``save_rejection_record`` had discounted strict runs
since it was written. The two paths disagreed silently.
"""

from __future__ import annotations

import pytest

from src.retrospective import (
    DRIVER_MARKET,
    DRIVER_RESIDUAL,
    SCOPE_CONTEXTUAL,
    SCOPE_UNRESOLVED,
    STRICT_MODE_CONFIDENCE_FACTOR,
    THESIS_NOT_EVALUATED,
    compute_confidence,
    generate_lesson,
    store_lesson,
)
from tests.advanced.retrospective_fakes import FakeLessonLLM, FakeLessonsMemory


def _comparison(**overrides):
    base = {
        "ticker": "001060.KS",
        "analysis_date": "2026-06-16",
        "sector": "Health Care",
        "exchange": "KS",
        "currency": "KRW",
        "verdict": "DO_NOT_INITIATE",
        "days_elapsed": 180,
        "price_return_pct": 5.87,
        "benchmark_return_pct": -20.04,
        "excess_return_pct": 25.91,
        "benchmark_used": "^KS11",
        "bear_risks_excerpt": "Saturated domestic market.",
        "attribution": {
            "market_return_pct": -20.04,
            "residual_return_pct": 25.91,
            "fx_return_pct": -5.57,
            "usd_investor_return_pct": 0.0,
            "dominant_driver": DRIVER_MARKET,
            "benchmark_available": True,
        },
        "lesson_scope": SCOPE_CONTEXTUAL,
        "thesis_validation_status": THESIS_NOT_EVALUATED,
        "kill_criteria": [
            "Payout ratio remains 0.0% for two consecutive fiscal years.",
            "MRQ earnings growth remains negative for three quarters.",
        ],
        "regime_at_decision": {
            "risk_appetite": "RISK_OFF",
            "shock_type": "GEOPOLITICAL",
            "shock_phase": "ACUTE",
            "confidence": "HIGH",
        },
        "regime_confidence": "HIGH",
        "cached_regime_delta": {
            "shifted": True,
            "shift_reason": "risk appetite: RISK_OFF -> RISK_ON",
            "regime_now": {
                "risk_appetite": "RISK_ON",
                "shock_type": "NONE",
                "shock_phase": "NONE",
            },
            "staleness_days": 2,
        },
    }
    base.update(overrides)
    return base


async def _prompt_for(comparison, monkeypatch) -> str:
    llm = FakeLessonLLM()
    monkeypatch.setattr(
        "src.llm_runtime.construction.build_required_model_for_seat",
        lambda *_a, **_k: llm,
    )
    await generate_lesson(comparison)
    return llm.last_prompt


class TestThePromptCarriesTheDeterministicAnswer:
    @pytest.mark.asyncio
    async def test_the_attribution_is_present(self, monkeypatch):
        prompt = await _prompt_for(_comparison(), monkeypatch)
        assert "RETURN ATTRIBUTION" in prompt
        assert "-20.0%" in prompt and "+25.9%" in prompt
        assert f"Dominant driver: {DRIVER_MARKET}" in prompt

    @pytest.mark.asyncio
    async def test_the_residual_is_not_called_alpha(self, monkeypatch):
        prompt = await _prompt_for(_comparison(), monkeypatch)
        assert "not stock-specific alpha" in prompt
        assert "sector-wide" in prompt

    @pytest.mark.asyncio
    async def test_both_regimes_are_present(self, monkeypatch):
        """The omission that produced the harmful lesson."""
        prompt = await _prompt_for(_comparison(), monkeypatch)
        assert "REGIME AT DECISION (T0): RISK_OFF / GEOPOLITICAL / ACUTE" in prompt
        assert "CACHED REGIME NOW (T1):  RISK_ON / NONE / NONE" in prompt
        assert "[shifted: yes]" in prompt

    @pytest.mark.asyncio
    async def test_the_kill_criteria_are_present_and_not_adjudicated(self, monkeypatch):
        prompt = await _prompt_for(_comparison(), monkeypatch)
        assert "1. Payout ratio remains 0.0%" in prompt
        assert "Do NOT" in prompt and "whether these fired" in prompt
        assert THESIS_NOT_EVALUATED in prompt

    @pytest.mark.asyncio
    async def test_the_regime_conditioning_rule_is_stated(self, monkeypatch):
        prompt = await _prompt_for(_comparison(), monkeypatch)
        assert "REGIME CONDITIONING" in prompt
        assert "A market-wide move is not evidence the screen was wrong." in prompt

    @pytest.mark.asyncio
    async def test_the_unresolved_rule_is_stated(self, monkeypatch):
        prompt = await _prompt_for(
            _comparison(lesson_scope=SCOPE_UNRESOLVED), monkeypatch
        )
        assert f"If the lesson scope is {SCOPE_UNRESOLVED}" in prompt
        assert "what should be CHECKED next time" in prompt

    @pytest.mark.asyncio
    async def test_the_tooling_comparison_is_present(self, monkeypatch):
        prompt = await _prompt_for(_comparison(), monkeypatch)
        assert "TOOLING BETWEEN RUNS:" in prompt

    @pytest.mark.asyncio
    async def test_the_lesson_scope_is_shown(self, monkeypatch):
        prompt = await _prompt_for(
            _comparison(lesson_scope=SCOPE_UNRESOLVED), monkeypatch
        )
        assert f"Lesson scope: {SCOPE_UNRESOLVED}" in prompt


class TestDegradedInputsRenderAsProseNotNone:
    """A literal "None" in a prompt reads to the model as a value."""

    @pytest.mark.asyncio
    async def test_a_legacy_comparison_without_attribution(self, monkeypatch):
        comparison = _comparison()
        comparison.pop("attribution")
        prompt = await _prompt_for(comparison, monkeypatch)
        assert "unavailable for this analysis" in prompt
        assert "None" not in prompt

    @pytest.mark.asyncio
    async def test_no_regime_at_decision(self, monkeypatch):
        comparison = _comparison(regime_at_decision=None, regime_confidence=None)
        prompt = await _prompt_for(comparison, monkeypatch)
        assert "REGIME AT DECISION (T0): not recorded" in prompt
        assert "confidence unknown" in prompt

    @pytest.mark.asyncio
    async def test_an_unknown_regime_delta_says_why(self, monkeypatch):
        comparison = _comparison(
            cached_regime_delta={
                "shifted": None,
                "shift_reason": "cached macro brief is 40d old (> 14d); not 'now'",
                "regime_now": None,
            }
        )
        prompt = await _prompt_for(comparison, monkeypatch)
        assert "[shifted: unknown]" in prompt
        assert "40d old" in prompt, "naming the cause stops the model guessing"

    @pytest.mark.asyncio
    async def test_no_regime_delta_at_all(self, monkeypatch):
        comparison = _comparison()
        comparison.pop("cached_regime_delta")
        prompt = await _prompt_for(comparison, monkeypatch)
        assert "not resolved [shifted: unknown]" in prompt

    @pytest.mark.asyncio
    async def test_empty_kill_criteria(self, monkeypatch):
        prompt = await _prompt_for(_comparison(kill_criteria=[]), monkeypatch)
        assert "none recorded" in prompt

    @pytest.mark.asyncio
    async def test_malformed_kill_criteria(self, monkeypatch):
        prompt = await _prompt_for(_comparison(kill_criteria="not a list"), monkeypatch)
        assert "none recorded" in prompt

    @pytest.mark.asyncio
    async def test_a_missing_market_leg_renders_unknown(self, monkeypatch):
        comparison = _comparison(
            attribution={
                "market_return_pct": None,
                "residual_return_pct": None,
                "fx_return_pct": None,
                "usd_investor_return_pct": None,
                "dominant_driver": "UNKNOWN",
                "benchmark_available": False,
            }
        )
        prompt = await _prompt_for(comparison, monkeypatch)
        assert "Market (^KS11): unknown" in prompt
        assert "Residual: unknown" in prompt

    @pytest.mark.asyncio
    async def test_a_legacy_comparison_still_generates(self, monkeypatch):
        """Old artifacts must not break lesson generation."""
        llm = FakeLessonLLM()
        monkeypatch.setattr(
            "src.llm_runtime.construction.build_required_model_for_seat",
            lambda *_a, **_k: llm,
        )
        result = await generate_lesson(
            {"ticker": "X.T", "analysis_date": "2026-01-01", "verdict": "HOLD"}
        )
        assert result is not None
        assert result[1] == "missed_risk"


class TestConfidenceAccountsForStrictMode:
    def _base(self, **overrides):
        base = {
            "days_elapsed": 180,
            "excess_return_pct": 40.0,
            "decision_intent": "reasoning",
            "is_quick_mode": False,
            "is_strict_mode": False,
        }
        base.update(overrides)
        return base

    def test_strict_mode_discounts_the_weight(self):
        """The gap: only rejection records applied this factor."""
        normal = compute_confidence(self._base())
        strict = compute_confidence(self._base(is_strict_mode=True))
        assert strict < normal
        assert strict == pytest.approx(normal * STRICT_MODE_CONFIDENCE_FACTOR, abs=0.01)

    def test_quick_and_strict_compound(self):
        both = compute_confidence(self._base(is_quick_mode=True, is_strict_mode=True))
        plain = compute_confidence(self._base())
        assert both == pytest.approx(
            plain * 0.7 * STRICT_MODE_CONFIDENCE_FACTOR, abs=0.01
        )

    def test_a_legacy_comparison_without_the_flag_is_unchanged(self):
        legacy = self._base()
        legacy.pop("is_strict_mode")
        assert compute_confidence(legacy) == compute_confidence(self._base())

    def test_the_factor_is_shared_with_rejection_records(self):
        """One constant, so the two paths cannot drift again."""
        import inspect

        from src import retrospective

        source = inspect.getsource(retrospective.save_rejection_record)
        assert "STRICT_MODE_CONFIDENCE_FACTOR" in source


class TestStoredMetadata:
    @pytest.mark.asyncio
    async def test_every_new_field_round_trips_as_a_flat_scalar(self):
        memory = FakeLessonsMemory()
        await store_lesson(
            "a lesson",
            "missed_risk",
            "MACRO_REGIME",
            _comparison(comparison_context="CHANGED", decision_intent="reasoning"),
            0.7,
            memory,
        )
        stored = memory.metadatas()[0]

        assert stored["lesson_scope"] == SCOPE_CONTEXTUAL
        assert stored["dominant_driver"] == DRIVER_MARKET
        assert stored["market_return_pct"] == pytest.approx(-20.04)
        assert stored["residual_return_pct"] == pytest.approx(25.91)
        assert stored["benchmark_available"] is True
        assert stored["regime_shifted"] == "YES"
        assert stored["comparison_context"] == "CHANGED"
        assert stored["decision_intent"] == "reasoning"
        assert stored["is_strict_mode"] is False
        assert stored["thesis_validation_status"] == THESIS_NOT_EVALUATED
        assert stored["regime_equity_transmission"] == ""

        for key, value in stored.items():
            assert isinstance(value, str | int | float | bool), (
                f"{key} is {type(value).__name__}; ChromaDB metadata must be flat"
            )

    @pytest.mark.asyncio
    async def test_an_unresolved_regime_delta_is_not_stored_as_no(self):
        """`NO` would claim the regime demonstrably held still."""
        memory = FakeLessonsMemory()
        await store_lesson(
            "a lesson",
            "missed_risk",
            "MACRO_REGIME",
            _comparison(cached_regime_delta={"shifted": None, "shift_reason": "stale"}),
            0.7,
            memory,
        )
        assert memory.metadatas()[0]["regime_shifted"] == "UNKNOWN"

    @pytest.mark.asyncio
    async def test_a_genuine_no_shift_is_stored_as_no(self):
        memory = FakeLessonsMemory()
        await store_lesson(
            "a lesson",
            "missed_risk",
            "MACRO_REGIME",
            _comparison(cached_regime_delta={"shifted": False, "shift_reason": "none"}),
            0.7,
            memory,
        )
        assert memory.metadatas()[0]["regime_shifted"] == "NO"

    @pytest.mark.asyncio
    async def test_a_legacy_comparison_stores_safe_defaults(self):
        memory = FakeLessonsMemory()
        await store_lesson(
            "a lesson",
            "missed_risk",
            "OPERATIONAL_MISS",
            {"ticker": "X.T", "analysis_date": "2026-01-01"},
            0.5,
            memory,
        )
        stored = memory.metadatas()[0]
        assert stored["dominant_driver"] == "UNKNOWN"
        assert stored["regime_shifted"] == "UNKNOWN"
        assert stored["benchmark_available"] is False
        assert stored["thesis_validation_status"] == THESIS_NOT_EVALUATED


class TestTheParseContractIsUnchanged:
    @pytest.mark.asyncio
    async def test_the_response_markers_still_parse(self, monkeypatch):
        llm = FakeLessonLLM(
            "LESSON: Watch regime before valuation.\n"
            "TYPE: false_positive\n"
            "FAILURE_MODE: MACRO_REGIME"
        )
        monkeypatch.setattr(
            "src.llm_runtime.construction.build_required_model_for_seat",
            lambda *_a, **_k: llm,
        )
        text, lesson_type, failure_mode = await generate_lesson(_comparison())
        assert text == "Watch regime before valuation."
        assert lesson_type == "false_positive"
        assert failure_mode == "MACRO_REGIME"

    @pytest.mark.asyncio
    async def test_an_out_of_enum_failure_mode_still_coerces(self, monkeypatch):
        llm = FakeLessonLLM(
            "LESSON: Something.\nTYPE: nonsense\nFAILURE_MODE: INVENTED_MODE"
        )
        monkeypatch.setattr(
            "src.llm_runtime.construction.build_required_model_for_seat",
            lambda *_a, **_k: llm,
        )
        _text, lesson_type, failure_mode = await generate_lesson(_comparison())
        assert lesson_type == "missed_risk"
        assert failure_mode == "OPERATIONAL_MISS"
