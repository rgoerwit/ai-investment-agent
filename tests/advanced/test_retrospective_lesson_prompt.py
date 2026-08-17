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
    BEAR_EXCERPT_PROMPT_CHARS,
    DRIVER_MARKET,
    DRIVER_RESIDUAL,
    FAILURE_MODES,
    SCOPE_CONTEXTUAL,
    SCOPE_UNRESOLVED,
    STRICT_MODE_CONFIDENCE_FACTOR,
    THESIS_NOT_EVALUATED,
    UNRESOLVED_PRICE_ONLY,
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
        # Out-of-vocabulary now resolves to the value that claims nothing.
        # OPERATIONAL_MISS asserted an operational cause from a parse failure.
        assert failure_mode == UNRESOLVED_PRICE_ONLY


class TestThePromptForbidsTheConstructionsTheModelActuallyUsed:
    """Two rules aimed at observed evasions, not at the abstract principle.

    Measured 2026-08-16 across 67 stored lessons, both patterns recurring:

    * a cause invented from price alone — "prioritize vetting data quality
      discrepancies, specifically cash flow reconciliations" (7047.T), where no
      input mentions cash flow at all;
    * the regime-conditioning rule satisfied in the clause the model wrote first,
      then undone by a trailing contrast — "…rather than assuming fundamental
      undervaluation alone is a sufficient margin of safety" (7638.T).

    The second is why restating the principle would not have helped: the model
    obeyed the rule as written and smuggled the prohibited half back afterwards.
    """

    @pytest.mark.asyncio
    async def test_inventing_a_mechanism_is_forbidden(self, monkeypatch):
        prompt = await _prompt_for(_comparison(), monkeypatch)
        assert "failure mechanism that appears nowhere above" in prompt
        # Wording deliberately generalized: the rule used to enumerate "recorded
        # bear risks, thesis-break triggers, or the regime", two of which are
        # empty on essentially every legacy snapshot.
        assert "follows from something actually shown to you" in prompt

    @pytest.mark.asyncio
    async def test_demoting_the_screen_by_contrast_is_forbidden(self, monkeypatch):
        prompt = await _prompt_for(_comparison(), monkeypatch)
        assert "rather than" in prompt and "instead of" in prompt
        assert "Write what to ADD, never what to trust less." in prompt

    @pytest.mark.asyncio
    async def test_type_is_scoped_to_the_price_outcome(self, monkeypatch):
        prompt = await _prompt_for(_comparison(), monkeypatch)
        assert "TYPE labels the prediction against the price only" in prompt
        assert THESIS_NOT_EVALUATED in prompt

    @pytest.mark.asyncio
    async def test_the_output_template_stays_clean(self, monkeypatch):
        """Explanatory prose belongs in the rules, never in the format block.

        The model copies the template, so a clarification written into the
        `TYPE:` line would be echoed into the parsed value.
        """
        prompt = await _prompt_for(_comparison(), monkeypatch)
        assert (
            "TYPE: missed_risk | false_positive | missed_opportunity | correct_call"
            in prompt
        )


class TestTheScopeStampGatesTheAntiDiagnosisRule:
    """The mapping fix and the prompt fix are one change, observed here.

    `_render_attribution` prints the scope into the prompt, and the "unexplained,
    not diagnosed" rule keys on it — so while MIXED resolved to CONTEXTUAL, that
    rule was never in force for the 67 records that most needed it.
    """

    @pytest.mark.asyncio
    async def test_a_mixed_outcome_now_renders_as_unresolved(self, monkeypatch):
        comparison = _comparison(
            attribution={
                "market_return_pct": 21.0,
                "residual_return_pct": -18.0,
                "fx_return_pct": 0.0,
                "usd_investor_return_pct": 0.0,
                "dominant_driver": "MIXED",
                "benchmark_available": True,
            },
            lesson_scope=SCOPE_UNRESOLVED,
        )
        prompt = await _prompt_for(comparison, monkeypatch)
        assert f"Lesson scope: {SCOPE_UNRESOLVED}" in prompt
        assert "the residual is unexplained, not" in prompt

    @pytest.mark.asyncio
    async def test_a_comparison_with_no_scope_renders_the_humbler_one(
        self, monkeypatch
    ):
        """Fail closed: an absent scope reaches the prompt as UNRESOLVED.

        The old `or SCOPE_CONTEXTUAL` default asserted the opposite in the very
        place the anti-diagnosis rule is read.
        """
        comparison = _comparison()
        comparison.pop("lesson_scope")
        prompt = await _prompt_for(comparison, monkeypatch)
        assert f"Lesson scope: {SCOPE_UNRESOLVED}" in prompt
        assert f"Lesson scope: {SCOPE_CONTEXTUAL}" not in prompt


class TestTheGroundingRuleIsSatisfiable:
    """An unsatisfiable instruction is ignored, not obeyed.

    Measured on the 2026-08-17 probe: 5 of the 7 lessons whose named mechanism
    could be checked cited something present in NO input. The cause was not model
    misbehaviour — the rule pointed at three grounding sources, and across all
    7,952 snapshots the thesis-break triggers are absent in 100% (the field
    postdates every legacy snapshot), the regime in 84%, and the bear excerpt was
    clipped to 300 chars while 55% are longer. Told to ground its answer and
    handed almost nothing, the model invents.

    Two repairs: show it the material that already exists on disk, and give it a
    legal way to decline.
    """

    @pytest.mark.asyncio
    async def test_the_full_stored_bear_excerpt_reaches_the_prompt(self, monkeypatch):
        excerpt = "".join(f"risk{i} " for i in range(80))[:500]
        prompt = await _prompt_for(_comparison(bear_risks_excerpt=excerpt), monkeypatch)
        assert excerpt[:BEAR_EXCERPT_PROMPT_CHARS] in prompt
        assert len(excerpt) > 300, "fixture must exceed the old clip or this is vacuous"

    def test_the_prompt_budget_matches_what_the_snapshot_stores(self):
        """A clip below the stored length withholds evidence the rule demands.

        `_extract_bear_risks` keeps ~500 chars; showing the model fewer means the
        anti-invention rule binds while the material to satisfy it sits unused.
        """
        assert BEAR_EXCERPT_PROMPT_CHARS >= 500

    @pytest.mark.asyncio
    async def test_declining_to_name_a_mechanism_is_offered_as_valid(self, monkeypatch):
        prompt = await _prompt_for(_comparison(), monkeypatch)
        assert "then SAY SO" in prompt
        assert "complete and correct lesson" in prompt

    @pytest.mark.asyncio
    async def test_the_rule_no_longer_cites_a_universally_absent_source(
        self, monkeypatch
    ):
        """It may describe triggers as *possibly* absent; it may not require them.

        The earlier wording said a check must follow from "the recorded bear
        risks, the thesis-break triggers, or the regime" — naming as a permitted
        source a field that is empty in every snapshot on disk.
        """
        prompt = await _prompt_for(_comparison(), monkeypatch)
        assert "follows from something actually shown to you" in prompt
        assert (
            "You may only\n  name a check that follows from the recorded" not in prompt
        )


class TestDecliningIsReachableInTheResponseFormat:
    """An escape hatch the output format cannot express is not an escape hatch.

    The prose rule said "if nothing supports a specific check, say so" while
    every one of the twelve FAILURE_MODE values named a mechanism — so a model
    that declined in the LESSON line still had to assert a cause one field down,
    and that field renders in the header beside the prose. The rule was
    unsatisfiable one field over from where it was written.
    """

    @pytest.mark.asyncio
    async def test_the_prompt_offers_the_token_in_the_enum(self, monkeypatch):
        prompt = await _prompt_for(_comparison(), monkeypatch)
        assert f"FAILURE_MODE: {UNRESOLVED_PRICE_ONLY} |" in prompt

    @pytest.mark.asyncio
    async def test_the_prompt_ties_declining_to_the_token(self, monkeypatch):
        """Offering the value is not enough; the model must be told when to use it."""
        prompt = await _prompt_for(_comparison(), monkeypatch)
        assert f"FAILURE_MODE must be {UNRESOLVED_PRICE_ONLY}" in prompt

    @pytest.mark.asyncio
    @pytest.mark.parametrize("emitted", [UNRESOLVED_PRICE_ONLY, "GOVERNANCE_BLEED"])
    async def test_the_parser_preserves_what_the_model_emitted(
        self, monkeypatch, emitted
    ):
        """Both rows are needed, and the second is why.

        Asserting only that a declining response yields the declining token
        proves nothing: that token is *also* the out-of-vocabulary default, so
        the assertion holds even if it were deleted from FAILURE_MODES. Pairing
        it with a causal mode shows the parser is preserving what was emitted
        rather than collapsing everything to the default.
        """
        llm = FakeLessonLLM(
            reply=(
                "LESSON: The inputs identify no mechanism; record what evidence "
                "would settle it.\n"
                "TYPE: missed_risk\n"
                f"FAILURE_MODE: {emitted}"
            )
        )
        monkeypatch.setattr(
            "src.llm_runtime.construction.build_required_model_for_seat",
            lambda *_a, **_k: llm,
        )
        result = await generate_lesson(_comparison())
        assert result is not None
        assert result[2] == emitted

    @pytest.mark.asyncio
    async def test_an_unparseable_mode_no_longer_invents_a_cause(self, monkeypatch):
        llm = FakeLessonLLM(
            reply="LESSON: Something.\nTYPE: missed_risk\nFAILURE_MODE: NONSENSE"
        )
        monkeypatch.setattr(
            "src.llm_runtime.construction.build_required_model_for_seat",
            lambda *_a, **_k: llm,
        )
        result = await generate_lesson(_comparison())
        assert result is not None
        assert result[2] == UNRESOLVED_PRICE_ONLY

    def test_it_is_the_only_member_that_names_no_mechanism(self):
        assert UNRESOLVED_PRICE_ONLY in FAILURE_MODES
        others = FAILURE_MODES - {UNRESOLVED_PRICE_ONLY}
        assert len(others) == 12, "the causal vocabulary must be unchanged"
