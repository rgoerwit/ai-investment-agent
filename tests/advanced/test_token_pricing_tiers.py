"""Tests for the July 2026 pricing-table rewrite and flex-tier cost tracking.

Guards against the blind spot that caused 3-4x cost underreporting: every
current model must have an explicit pricing entry (no silent default-pricing
fallback), and per-call service tiers must halve the estimate for flex calls.
"""

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from src.token_tracker import (
    CACHED_PROMPT_MULTIPLIER,
    DEFAULT_PRICING_PER_1M,
    FLEX_TIER_MULTIPLIER,
    MODEL_PRICING_PER_1M,
    TokenTrackingCallback,
    TokenUsage,
    _lookup_model_pricing,
    _warned_unknown_pricing_models,
)


def _usage(model: str, tier: str | None = None, cached: int = 0) -> TokenUsage:
    return TokenUsage(
        timestamp="2026-07-03T12:00:00",
        agent_name="test",
        model_name=model,
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
        total_tokens=2_000_000,
        service_tier=tier,
        cached_prompt_tokens=cached,
    )


class TestCurrentModelPricing:
    """One assertion per model actually configured in this repo."""

    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("gemini-3.5-flash", 1.50 + 9.00),
            ("gemini-3.1-flash-lite", 0.25 + 1.50),
            ("gemini-3.1-pro-preview", 2.00 + 12.00),
            ("gemini-3-flash-preview", 0.50 + 3.00),
            ("gemini-3-pro-preview", 2.00 + 12.00),
            ("gpt-5.4", 2.50 + 15.00),
            ("gpt-5.4-mini", 0.75 + 4.50),
            ("gpt-5.5", 5.00 + 30.00),
            ("claude-opus-4-6", 5.00 + 25.00),
            ("deepseek-v4-pro", 0.435 + 0.87),
        ],
    )
    def test_standard_tier_cost(self, model, expected):
        assert _usage(model).estimated_cost_usd == pytest.approx(expected)

    def test_no_current_model_hits_default_fallback(self):
        # These are the models wired via config defaults / .env — each must
        # prefix-match an explicit entry, never the default fallback.
        for model in (
            "gemini-3.5-flash",
            "gemini-3.1-flash-lite",
            "gemini-3.1-pro-preview",
            "gemini-3-flash-preview",
            "gpt-5.4",
            "gpt-5.4-mini",
            "claude-opus-4-6",
            "deepseek-v4-pro",
        ):
            assert _lookup_model_pricing(model) is not DEFAULT_PRICING_PER_1M, model

    def test_mini_and_lite_variants_match_before_parents(self):
        assert _lookup_model_pricing("gpt-5.4-mini")["completion"] == 4.50
        assert _lookup_model_pricing("gemini-2.5-flash-lite")["prompt"] == 0.10


class TestFlexTierPricing:
    def test_flex_halves_the_cost(self):
        standard = _usage("gpt-5.4").estimated_cost_usd
        flex = _usage("gpt-5.4", tier="flex").estimated_cost_usd
        assert flex == pytest.approx(standard * FLEX_TIER_MULTIPLIER)

    def test_fallback_to_standard_prices_full(self):
        # A flex-configured run whose call fell back reports its real tier
        assert _usage("gemini-3.5-flash", tier="standard").estimated_cost_usd == (
            _usage("gemini-3.5-flash").estimated_cost_usd
        )

    def test_auto_tier_prices_full(self):
        assert _usage("gpt-5.4", tier="auto").estimated_cost_usd == (
            _usage("gpt-5.4").estimated_cost_usd
        )


class TestCachedPromptPricing:
    """Cached prompt-prefix tokens bill at 10% of the input rate (both
    vendors, verified July 2026); flex does not stack on the cached portion."""

    def test_cached_tokens_reduce_prompt_cost(self):
        # gpt-5.4: 1M prompt (600k cached) + 1M completion
        # = 400k*$2.50 + 600k*$0.25 + 1M*$15 per-1M rates
        expected = 0.4 * 2.50 + 0.6 * 2.50 * CACHED_PROMPT_MULTIPLIER + 15.00
        assert _usage("gpt-5.4", cached=600_000).estimated_cost_usd == pytest.approx(
            expected
        )

    def test_fully_cached_prompt(self):
        expected = 2.50 * CACHED_PROMPT_MULTIPLIER + 15.00
        assert _usage("gpt-5.4", cached=1_000_000).estimated_cost_usd == pytest.approx(
            expected
        )

    def test_cached_exceeding_prompt_clamps(self):
        # Defensive: a provider quirk reporting cached > prompt must not
        # produce a negative uncached component.
        capped = _usage("gpt-5.4", cached=5_000_000).estimated_cost_usd
        assert capped == _usage("gpt-5.4", cached=1_000_000).estimated_cost_usd

    def test_flex_does_not_discount_cached_portion(self):
        # Uncached prompt + completion halve under flex; cached portion stays
        # at the standard cached rate (OpenAI: discounts don't stack).
        cost = _usage("gpt-5.4", tier="flex", cached=600_000).estimated_cost_usd
        expected = (
            0.4 * 2.50 + 15.00
        ) * FLEX_TIER_MULTIPLIER + 0.6 * 2.50 * CACHED_PROMPT_MULTIPLIER
        assert cost == pytest.approx(expected)

    def test_zero_cached_matches_previous_behavior(self):
        assert _usage("gemini-3.1-pro-preview").estimated_cost_usd == pytest.approx(
            2.00 + 12.00
        )


class TestCachedTokenExtraction:
    """End-to-end: provider usage metadata → breakdown → tracker record."""

    def _result_with_usage(self, usage_metadata) -> LLMResult:
        message = AIMessage(content="ok", usage_metadata=usage_metadata)
        return LLMResult(generations=[[ChatGeneration(message=message)]])

    def test_langchain_cache_read_detail_extracted(self):
        from src.llm_usage import extract_token_usage_breakdown

        result = self._result_with_usage(
            {
                "input_tokens": 10_000,
                "output_tokens": 500,
                "total_tokens": 10_500,
                "input_token_details": {"cache_read": 8_000},
            }
        )
        breakdown = extract_token_usage_breakdown(result)
        assert breakdown.input_tokens == 10_000
        assert breakdown.cached_input_tokens == 8_000

    def test_openai_raw_prompt_tokens_details_extracted(self):
        from src.llm_usage import extract_token_usage_breakdown

        message = AIMessage(content="ok")
        message.response_metadata["token_usage"] = {
            "prompt_tokens": 10_000,
            "completion_tokens": 500,
            "total_tokens": 10_500,
            "prompt_tokens_details": {"cached_tokens": 8_000},
        }
        result = LLMResult(generations=[[ChatGeneration(message=message)]])
        breakdown = extract_token_usage_breakdown(result)
        assert breakdown.cached_input_tokens == 8_000

    def test_absent_details_yield_none(self):
        from src.llm_usage import extract_token_usage_breakdown

        result = self._result_with_usage(
            {"input_tokens": 10_000, "output_tokens": 500, "total_tokens": 10_500}
        )
        assert extract_token_usage_breakdown(result).cached_input_tokens is None

    def test_callback_records_cached_tokens(self):
        recorded = {}

        class _FakeTracker:
            def record_usage(self, **kwargs):
                recorded.update(kwargs)

        callback = TokenTrackingCallback("Consultant", tracker=_FakeTracker())
        callback.on_llm_end(
            self._result_with_usage(
                {
                    "input_tokens": 10_000,
                    "output_tokens": 500,
                    "total_tokens": 10_500,
                    "input_token_details": {"cache_read": 8_000},
                }
            )
        )
        assert recorded["prompt_tokens"] == 10_000
        assert recorded["cached_prompt_tokens"] == 8_000


class TestUnknownModelWarning:
    def test_unknown_model_uses_default_and_warns_once(self, caplog):
        _warned_unknown_pricing_models.discard("totally-new-model")
        assert _lookup_model_pricing("totally-new-model") is DEFAULT_PRICING_PER_1M
        assert "totally-new-model" in _warned_unknown_pricing_models
        # Second lookup does not duplicate the warning entry
        _lookup_model_pricing("totally-new-model")
        assert (
            len([m for m in _warned_unknown_pricing_models if m == "totally-new-model"])
            == 1
        )

    def test_retired_models_are_absent_from_table(self):
        # Deliberate policy: retired/deprecated models are not carried
        for retired in ("gpt-4-turbo", "gpt-4-32k", "gemini-2.0-flash-exp"):
            assert not any(
                retired == key for key in MODEL_PRICING_PER_1M
            ), f"{retired} should not have a pricing entry"


class TestServiceTierExtraction:
    def _llm_result(self, llm_output=None, resp_meta=None) -> LLMResult:
        message = AIMessage(content="ok")
        if resp_meta:
            message.response_metadata.update(resp_meta)
        return LLMResult(
            generations=[[ChatGeneration(message=message)]],
            llm_output=llm_output,
        )

    def test_tier_from_llm_output(self):
        result = self._llm_result(llm_output={"service_tier": "flex"})
        assert TokenTrackingCallback._extract_service_tier(result) == "flex"

    def test_tier_from_response_metadata(self):
        result = self._llm_result(resp_meta={"service_tier": "standard"})
        assert TokenTrackingCallback._extract_service_tier(result) == "standard"

    def test_llm_output_wins_over_metadata(self):
        result = self._llm_result(
            llm_output={"service_tier": "flex"},
            resp_meta={"service_tier": "standard"},
        )
        assert TokenTrackingCallback._extract_service_tier(result) == "flex"

    def test_no_tier_returns_none(self):
        assert TokenTrackingCallback._extract_service_tier(self._llm_result()) is None
