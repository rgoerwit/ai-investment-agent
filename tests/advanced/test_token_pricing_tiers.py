"""Tests for the July 2026 pricing-table rewrite and flex-tier cost tracking.

Guards against the blind spot that caused 3-4x cost underreporting: every
current model must have an explicit pricing entry (no silent default-pricing
fallback), and per-call service tiers must halve the estimate for flex calls.
"""

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from src.token_tracker import (
    CACHE_WRITE_PROMPT_MULTIPLIER,
    CACHED_PROMPT_MULTIPLIER,
    DEFAULT_PRICING_PER_1M,
    FLEX_TIER_MULTIPLIER,
    GPT56_LONG_CONTEXT_COMPLETION_MULTIPLIER,
    GPT56_LONG_CONTEXT_INPUT_THRESHOLD,
    GPT56_LONG_CONTEXT_PROMPT_MULTIPLIER,
    MODEL_PRICING_PER_1M,
    TokenTrackingCallback,
    TokenUsage,
    _lookup_model_pricing,
    _warned_unknown_pricing_models,
)


def _usage(
    model: str,
    tier: str | None = None,
    cached: int = 0,
    *,
    prompt: int = 1_000_000,
    completion: int = 1_000_000,
    cache_write: int = 0,
) -> TokenUsage:
    return TokenUsage(
        timestamp="2026-07-03T12:00:00",
        agent_name="test",
        model_name=model,
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
        service_tier=tier,
        cached_prompt_tokens=cached,
        cache_write_prompt_tokens=cache_write,
    )


class TestCurrentModelPricing:
    """One assertion per model actually configured in this repo."""

    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("gemini-3.6-flash", 1.50 + 7.50),
            ("gemini-3.6-flash-002", 1.50 + 7.50),
            ("gemini-3.5-flash", 1.50 + 9.00),
            ("gemini-3.1-flash-lite", 0.25 + 1.50),
            ("gemini-3.1-pro-preview", 2.00 + 12.00),
            ("gemini-3-flash-preview", 0.50 + 3.00),
            ("gemini-3-pro-preview", 2.00 + 12.00),
            ("gpt-5.4", 2.50 + 15.00),
            ("gpt-5.4-mini", 0.75 + 4.50),
            ("gpt-5.5", 5.00 + 30.00),
            ("gpt-5.6-sol", 5.00 * 2.00 + 30.00 * 1.50),
            ("gpt-5.6-terra", 2.50 * 2.00 + 15.00 * 1.50),
            ("gpt-5.6-luna", 1.00 * 2.00 + 6.00 * 1.50),
            ("gpt-5.6", 5.00 * 2.00 + 30.00 * 1.50),
            ("claude-opus-4-6", 5.00 + 25.00),
            ("glm-5.2", 1.40 + 4.40),
            ("deepseek-v4-pro", 0.435 + 0.87),
            ("kimi-k3", 0.95 + 4.00),
        ],
    )
    def test_standard_tier_cost(self, model, expected):
        assert _usage(model).estimated_cost_usd == pytest.approx(expected)

    def test_no_current_model_hits_default_fallback(self):
        # Configured and adoption-ready models must prefix-match an explicit
        # entry, never the default fallback.
        for model in (
            "gemini-3.6-flash",
            "gemini-3.5-flash",
            "gemini-3.1-flash-lite",
            "gemini-3.1-pro-preview",
            "gemini-3-flash-preview",
            "gpt-5.4",
            "gpt-5.4-mini",
            "gpt-5.6-sol",
            "gpt-5.6-terra",
            "gpt-5.6-luna",
            "gpt-5.6",
            "claude-opus-4-6",
            "glm-5.2",
            "deepseek-v4-pro",
            "kimi-k3",
        ):
            assert _lookup_model_pricing(model) is not DEFAULT_PRICING_PER_1M, model

    def test_mini_and_lite_variants_match_before_parents(self):
        assert _lookup_model_pricing("gpt-5.4-mini")["completion"] == 4.50
        assert _lookup_model_pricing("gemini-2.5-flash-lite")["prompt"] == 0.10


class TestMatcherOrderIndependence:
    """A1b: exact-match then longest-prefix, insensitive to dict insertion order."""

    def test_longest_prefix_wins_regardless_of_insertion_order(self, monkeypatch):
        # Parent listed BEFORE the more-specific variant — the old first-match
        # loop would have mispriced the variant; longest-prefix must not.
        shuffled = {
            "gpt-5.4": {"prompt": 2.50, "completion": 15.00},
            "gpt-5.4-mini": {"prompt": 0.75, "completion": 4.50},
        }
        monkeypatch.setattr(
            "src.token_tracker.MODEL_PRICING_PER_1M", shuffled, raising=True
        )
        assert _lookup_model_pricing("gpt-5.4-mini-2026")["completion"] == 4.50
        assert _lookup_model_pricing("gpt-5.4")["completion"] == 15.00

    def test_exact_match_preferred(self):
        # Bare "gpt-5.6" must resolve to its own entry, not a longer sibling.
        assert _lookup_model_pricing("gpt-5.6")["prompt"] == 5.00
        assert _lookup_model_pricing("gpt-5.6-terra")["prompt"] == 2.50

    def test_vendor_prefix_is_stripped(self):
        assert _lookup_model_pricing("moonshot/kimi-k3") is _lookup_model_pricing(
            "kimi-k3"
        )
        assert _lookup_model_pricing("google/gemini-3.6-flash")["completion"] == 7.50

    def test_current_matches_unchanged_by_rewrite(self):
        # Every table key resolves to itself (regression pin for the rewrite).
        for key, prices in MODEL_PRICING_PER_1M.items():
            assert _lookup_model_pricing(key) is prices, key

    @pytest.mark.parametrize(
        ("model", "prompt_rate", "completion_rate"),
        [
            ("gpt-5.6-sol", 5.00, 30.00),
            ("gpt-5.6-terra", 2.50, 15.00),
            ("gpt-5.6-luna", 1.00, 6.00),
        ],
    )
    def test_gpt56_long_context_surcharge_starts_above_threshold(
        self, model, prompt_rate, completion_rate
    ):
        completion = 10_000
        at_threshold = _usage(
            model,
            prompt=GPT56_LONG_CONTEXT_INPUT_THRESHOLD,
            completion=completion,
        )
        expected_base = (
            GPT56_LONG_CONTEXT_INPUT_THRESHOLD / 1_000_000 * prompt_rate
            + completion / 1_000_000 * completion_rate
        )
        assert at_threshold.estimated_cost_usd == pytest.approx(expected_base)

        above_threshold = _usage(
            model,
            prompt=GPT56_LONG_CONTEXT_INPUT_THRESHOLD + 1,
            completion=completion,
        )
        expected_surcharged = (
            (GPT56_LONG_CONTEXT_INPUT_THRESHOLD + 1)
            / 1_000_000
            * prompt_rate
            * GPT56_LONG_CONTEXT_PROMPT_MULTIPLIER
            + completion
            / 1_000_000
            * completion_rate
            * GPT56_LONG_CONTEXT_COMPLETION_MULTIPLIER
        )
        assert above_threshold.estimated_cost_usd == pytest.approx(expected_surcharged)

    def test_gpt56_cache_writes_use_published_multiplier(self):
        usage = _usage(
            "gpt-5.6-luna",
            prompt=100_000,
            completion=10_000,
            cache_write=40_000,
        )
        expected = (
            60_000 / 1_000_000 * 1.00
            + 40_000 / 1_000_000 * 1.00 * CACHE_WRITE_PROMPT_MULTIPLIER
            + 10_000 / 1_000_000 * 6.00
        )
        assert usage.estimated_cost_usd == pytest.approx(expected)


class TestFlexTierPricing:
    def test_flex_halves_the_cost(self):
        standard = _usage("gpt-5.4").estimated_cost_usd
        flex = _usage("gpt-5.4", tier="flex").estimated_cost_usd
        assert flex == pytest.approx(standard * FLEX_TIER_MULTIPLIER)

    def test_gemini_36_flex_halves_published_rates(self):
        assert _usage(
            "gemini-3.6-flash", tier="flex"
        ).estimated_cost_usd == pytest.approx((1.50 + 7.50) * FLEX_TIER_MULTIPLIER)

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
    """Cached prompt-prefix tokens use provider-published rates."""

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

    def test_glm_uses_model_specific_cached_input_rate(self):
        # 1M prompt (600k cached) + 1M completion
        expected = 0.4 * 1.40 + 0.6 * 0.26 + 4.40
        assert _usage("glm-5.2", cached=600_000).estimated_cost_usd == pytest.approx(
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

    def test_langchain_flex_cache_and_reasoning_details_extracted(self):
        from src.llm_usage import extract_token_usage_breakdown

        result = self._result_with_usage(
            {
                "input_tokens": 10_000,
                "output_tokens": 500,
                "total_tokens": 10_500,
                "input_token_details": {
                    "flex_cache_read": 7_000,
                    "flex_cache_creation": 1_000,
                },
                "output_token_details": {"flex_reasoning": 300},
            }
        )
        breakdown = extract_token_usage_breakdown(result)
        assert breakdown.cached_input_tokens == 7_000
        assert breakdown.cache_write_input_tokens == 1_000
        assert breakdown.thinking_tokens == 300

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
                    "input_token_details": {
                        "cache_read": 7_000,
                        "cache_creation": 1_000,
                    },
                }
            )
        )
        assert recorded["prompt_tokens"] == 10_000
        assert recorded["cached_prompt_tokens"] == 7_000
        assert recorded["cache_write_prompt_tokens"] == 1_000


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
