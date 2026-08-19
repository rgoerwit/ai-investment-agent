import pytest

from src.llm_runtime.capabilities import Capability
from src.llm_runtime.identities import sanitize_endpoint_host
from src.llm_runtime.profiles import (
    ModelProfile,
    TemperaturePolicy,
    TokenParameter,
    UnsupportedModelCapability,
    adjust_reasoning,
    resolve_profile,
)


def test_shipped_models_resolve_to_commercial_identity() -> None:
    assert resolve_profile("gemini-3.1-pro-preview").identity.vendor_id == "google"
    assert resolve_profile("gpt-5.4-mini").identity.vendor_id == "openai"
    assert resolve_profile("claude-opus-4-6").identity.model_lineage == "claude"
    assert resolve_profile("claude-opus-4-8").reasoning_ladder == (
        "low",
        "medium",
        "high",
        "max",
    )
    assert (
        resolve_profile("deepseek-v4-pro").identity.adapter_kind == "openai_compatible"
    )
    assert Capability.TOOL_CALLING in resolve_profile("kimi-k3").capabilities
    assert Capability.STRUCTURED_OUTPUT in resolve_profile("kimi-k3").capabilities
    claude = resolve_profile("claude-opus-4-6")
    assert Capability.TOOL_CALLING in claude.capabilities
    assert Capability.STRUCTURED_OUTPUT in claude.capabilities
    assert claude.reasoning_ladder == ("low", "medium", "high", "max")
    assert resolve_profile("claude-haiku-4-5").identity.vendor_id == "anthropic"


def test_unknown_future_claude_version_does_not_inherit_a_broad_old_profile() -> None:
    profile = resolve_profile("claude-opus-4-9")
    assert profile.identity.vendor_id == "unknown"
    assert profile.capabilities == frozenset({Capability.TEXT_GENERATION})


def test_longest_prefix_wins_independent_of_order() -> None:
    broad = ModelProfile(
        "model-",
        resolve_profile("gpt-5.4").identity,
        frozenset(),
        (),
        TokenParameter.MAX_TOKENS,
        TemperaturePolicy.OMIT,
    )
    narrow = ModelProfile(
        "model-pro",
        resolve_profile("gemini-3-flash-preview").identity,
        frozenset(),
        (),
        TokenParameter.MAX_OUTPUT_TOKENS,
        TemperaturePolicy.SUPPORTED,
    )
    assert resolve_profile("model-pro-v2", (narrow, broad)) is narrow
    assert resolve_profile("model-pro-v2", (broad, narrow)) is narrow


def test_unknown_model_is_conservative_and_unpriced() -> None:
    profile = resolve_profile("new-vendor/unknown-9000")
    assert profile.identity.vendor_id == "unknown"
    assert profile.capabilities == frozenset({Capability.TEXT_GENERATION})
    assert profile.reasoning_ladder == ()
    assert profile.pricing_key is None


def test_reasoning_adjustment_uses_documented_ladder_and_clamps() -> None:
    google = resolve_profile("gemini-3-flash-preview")
    openai = resolve_profile("gpt-5.6-sol")
    assert adjust_reasoning(google, "low") == "medium"
    assert adjust_reasoning(google, "high") == "high"
    assert adjust_reasoning(openai, "xhigh") == "max"


def test_reasoning_adjustment_rejects_incapable_or_unknown_value() -> None:
    with pytest.raises(UnsupportedModelCapability):
        adjust_reasoning(resolve_profile("claude-haiku-4-5"), "low")
    with pytest.raises(UnsupportedModelCapability):
        adjust_reasoning(resolve_profile("gpt-5.4"), "max")


def test_endpoint_identity_retains_host_only() -> None:
    assert (
        sanitize_endpoint_host("https://user:secret@API.Z.AI/v1?q=secret") == "api.z.ai"
    )
    with pytest.raises(ValueError):
        sanitize_endpoint_host("api.z.ai/v1")


def test_equal_rank_profiles_are_rejected() -> None:
    profile = resolve_profile("gpt-5.4")
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_profile("gpt-5.4", (profile, profile))


def test_new_and_compat_openai_reasoning_registries_cannot_drift() -> None:
    from src import llms
    from src.llm_runtime.profiles import MODEL_PROFILES

    for profile in MODEL_PROFILES:
        if (
            profile.identity.vendor_id == "openai"
            and Capability.REASONING_CONTROL in profile.capabilities
        ):
            assert llms._openai_supported_reasoning_efforts(  # noqa: SLF001
                profile.prefix
            ) == frozenset(profile.reasoning_ladder)
