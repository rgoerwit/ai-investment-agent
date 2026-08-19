"""Reviewed model-family facts used for safe binding validation."""

import re
from dataclasses import dataclass
from enum import StrEnum

from src.llm_runtime.capabilities import Capability
from src.llm_runtime.identities import ModelIdentity


class TokenParameter(StrEnum):
    MAX_OUTPUT_TOKENS = "max_output_tokens"
    MAX_COMPLETION_TOKENS = "max_completion_tokens"
    MAX_TOKENS = "max_tokens"


class TemperaturePolicy(StrEnum):
    SUPPORTED = "supported"
    OMIT = "omit"
    FIXED_ONE = "fixed_one"


class ReasoningApiMode(StrEnum):
    NONE = "none"
    MANUAL = "manual"
    ADAPTIVE = "adaptive"


@dataclass(frozen=True)
class ModelProfile:
    prefix: str
    identity: ModelIdentity
    capabilities: frozenset[Capability]
    reasoning_ladder: tuple[str, ...]
    token_parameter: TokenParameter
    temperature_policy: TemperaturePolicy
    reasoning_api_mode: ReasoningApiMode = ReasoningApiMode.NONE
    service_tiers: frozenset[str] = frozenset({"standard"})
    pricing_key: str | None = None
    priority: int = 0


class UnsupportedModelCapability(ValueError):
    pass


_TEXT = frozenset({Capability.TEXT_GENERATION})
_NATIVE_ANALYSIS = frozenset(
    {
        Capability.TEXT_GENERATION,
        Capability.TOOL_CALLING,
        Capability.STRUCTURED_OUTPUT,
        Capability.REASONING_CONTROL,
    }
)
_OPENAI_ANALYSIS = frozenset({*_NATIVE_ANALYSIS, Capability.RESPONSES_API})
_ANTHROPIC_AGENTIC = frozenset(
    {
        Capability.TEXT_GENERATION,
        Capability.TOOL_CALLING,
        Capability.STRUCTURED_OUTPUT,
    }
)
_ANTHROPIC_REASONING = frozenset({*_ANTHROPIC_AGENTIC, Capability.REASONING_CONTROL})
_ANTHROPIC_TOOL_USE = frozenset({Capability.TEXT_GENERATION, Capability.TOOL_CALLING})


MODEL_PROFILES: tuple[ModelProfile, ...] = (
    ModelProfile(
        prefix="gemini-3.1-pro",
        identity=ModelIdentity("google", "gemini", "google_native"),
        capabilities=_NATIVE_ANALYSIS,
        reasoning_ladder=("low", "medium", "high"),
        token_parameter=TokenParameter.MAX_OUTPUT_TOKENS,
        temperature_policy=TemperaturePolicy.SUPPORTED,
        service_tiers=frozenset({"standard", "flex"}),
        pricing_key="gemini-3.1-pro",
    ),
    ModelProfile(
        prefix="gemini-3",
        identity=ModelIdentity("google", "gemini", "google_native"),
        capabilities=_NATIVE_ANALYSIS,
        reasoning_ladder=("low", "medium", "high"),
        token_parameter=TokenParameter.MAX_OUTPUT_TOKENS,
        temperature_policy=TemperaturePolicy.SUPPORTED,
        service_tiers=frozenset({"standard", "flex"}),
        pricing_key="gemini-3",
    ),
    ModelProfile(
        prefix="gpt-5.6",
        identity=ModelIdentity("openai", "gpt", "openai_native"),
        capabilities=_OPENAI_ANALYSIS,
        reasoning_ladder=("none", "low", "medium", "high", "xhigh", "max"),
        token_parameter=TokenParameter.MAX_COMPLETION_TOKENS,
        temperature_policy=TemperaturePolicy.OMIT,
        service_tiers=frozenset({"standard", "flex"}),
        pricing_key="gpt-5.6",
    ),
    ModelProfile(
        prefix="gpt-5.4",
        identity=ModelIdentity("openai", "gpt", "openai_native"),
        capabilities=_OPENAI_ANALYSIS,
        reasoning_ladder=("none", "low", "medium", "high", "xhigh"),
        token_parameter=TokenParameter.MAX_COMPLETION_TOKENS,
        temperature_policy=TemperaturePolicy.OMIT,
        service_tiers=frozenset({"standard", "flex"}),
        pricing_key="gpt-5.4",
    ),
    ModelProfile(
        prefix="gpt-5.1",
        identity=ModelIdentity("openai", "gpt", "openai_native"),
        capabilities=_OPENAI_ANALYSIS,
        reasoning_ladder=("none", "low", "medium", "high"),
        token_parameter=TokenParameter.MAX_COMPLETION_TOKENS,
        temperature_policy=TemperaturePolicy.OMIT,
        service_tiers=frozenset({"standard", "flex"}),
        pricing_key="gpt-5.1",
    ),
    ModelProfile(
        prefix="gpt-5",
        identity=ModelIdentity("openai", "gpt", "openai_native"),
        capabilities=_OPENAI_ANALYSIS,
        reasoning_ladder=("minimal", "low", "medium", "high"),
        token_parameter=TokenParameter.MAX_COMPLETION_TOKENS,
        temperature_policy=TemperaturePolicy.OMIT,
        service_tiers=frozenset({"standard", "flex"}),
        pricing_key="gpt-5",
    ),
    ModelProfile(
        prefix="gpt-4o",
        identity=ModelIdentity("openai", "gpt", "openai_native"),
        capabilities=frozenset(
            {
                Capability.TEXT_GENERATION,
                Capability.TOOL_CALLING,
                Capability.STRUCTURED_OUTPUT,
                Capability.RESPONSES_API,
            }
        ),
        reasoning_ladder=(),
        token_parameter=TokenParameter.MAX_COMPLETION_TOKENS,
        temperature_policy=TemperaturePolicy.SUPPORTED,
        pricing_key="gpt-4o",
    ),
    ModelProfile(
        prefix="claude-opus-4-8",
        identity=ModelIdentity("anthropic", "claude", "anthropic_native"),
        capabilities=_ANTHROPIC_REASONING,
        reasoning_ladder=("low", "medium", "high", "max"),
        token_parameter=TokenParameter.MAX_TOKENS,
        temperature_policy=TemperaturePolicy.OMIT,
        reasoning_api_mode=ReasoningApiMode.ADAPTIVE,
        pricing_key="claude-opus-4",
    ),
    ModelProfile(
        prefix="claude-opus-4-6",
        identity=ModelIdentity("anthropic", "claude", "anthropic_native"),
        capabilities=_ANTHROPIC_REASONING,
        reasoning_ladder=("low", "medium", "high", "max"),
        token_parameter=TokenParameter.MAX_TOKENS,
        temperature_policy=TemperaturePolicy.OMIT,
        reasoning_api_mode=ReasoningApiMode.ADAPTIVE,
        pricing_key="claude-opus-4",
    ),
    ModelProfile(
        prefix="claude-sonnet-4-6",
        identity=ModelIdentity("anthropic", "claude", "anthropic_native"),
        capabilities=_ANTHROPIC_REASONING,
        reasoning_ladder=("low", "medium", "high", "max"),
        token_parameter=TokenParameter.MAX_TOKENS,
        temperature_policy=TemperaturePolicy.OMIT,
        reasoning_api_mode=ReasoningApiMode.ADAPTIVE,
        pricing_key="claude-sonnet-4",
    ),
    ModelProfile(
        prefix="claude-opus-4-5",
        identity=ModelIdentity("anthropic", "claude", "anthropic_native"),
        capabilities=_ANTHROPIC_AGENTIC,
        reasoning_ladder=(),
        token_parameter=TokenParameter.MAX_TOKENS,
        temperature_policy=TemperaturePolicy.OMIT,
        reasoning_api_mode=ReasoningApiMode.MANUAL,
        pricing_key="claude-opus-4",
    ),
    ModelProfile(
        prefix="claude-sonnet-4-5",
        identity=ModelIdentity("anthropic", "claude", "anthropic_native"),
        capabilities=_ANTHROPIC_AGENTIC,
        reasoning_ladder=(),
        token_parameter=TokenParameter.MAX_TOKENS,
        temperature_policy=TemperaturePolicy.OMIT,
        reasoning_api_mode=ReasoningApiMode.MANUAL,
        pricing_key="claude-sonnet-4",
    ),
    ModelProfile(
        prefix="claude-haiku-4-5",
        identity=ModelIdentity("anthropic", "claude", "anthropic_native"),
        capabilities=_ANTHROPIC_AGENTIC,
        reasoning_ladder=(),
        token_parameter=TokenParameter.MAX_TOKENS,
        temperature_policy=TemperaturePolicy.OMIT,
        pricing_key="claude-haiku-4",
    ),
    ModelProfile(
        prefix="claude-opus-4",
        identity=ModelIdentity("anthropic", "claude", "anthropic_native"),
        capabilities=_ANTHROPIC_TOOL_USE,
        reasoning_ladder=(),
        token_parameter=TokenParameter.MAX_TOKENS,
        temperature_policy=TemperaturePolicy.OMIT,
        reasoning_api_mode=ReasoningApiMode.MANUAL,
        pricing_key="claude-opus-4",
    ),
    ModelProfile(
        prefix="claude-sonnet-4",
        identity=ModelIdentity("anthropic", "claude", "anthropic_native"),
        capabilities=_ANTHROPIC_TOOL_USE,
        reasoning_ladder=(),
        token_parameter=TokenParameter.MAX_TOKENS,
        temperature_policy=TemperaturePolicy.OMIT,
        reasoning_api_mode=ReasoningApiMode.MANUAL,
        pricing_key="claude-sonnet-4",
    ),
    ModelProfile(
        prefix="deepseek-v4",
        identity=ModelIdentity("deepseek", "deepseek", "openai_compatible"),
        capabilities=frozenset(
            {Capability.TEXT_GENERATION, Capability.REASONING_CONTROL}
        ),
        reasoning_ladder=("low", "high", "max"),
        token_parameter=TokenParameter.MAX_COMPLETION_TOKENS,
        temperature_policy=TemperaturePolicy.OMIT,
        pricing_key="deepseek-v4",
    ),
    ModelProfile(
        prefix="glm-5",
        identity=ModelIdentity("zai", "glm", "openai_compatible"),
        capabilities=frozenset(
            {Capability.TEXT_GENERATION, Capability.REASONING_CONTROL}
        ),
        reasoning_ladder=("low", "high", "max"),
        token_parameter=TokenParameter.MAX_COMPLETION_TOKENS,
        temperature_policy=TemperaturePolicy.OMIT,
        pricing_key="glm-5",
    ),
    ModelProfile(
        prefix="kimi-k3",
        identity=ModelIdentity("moonshot", "kimi", "openai_compatible"),
        # This is deliberately family-specific. The repository's existing Kimi
        # review deployment exercises tool calls and strict editor output; an
        # arbitrary compatible endpoint must not inherit these capabilities.
        capabilities=_NATIVE_ANALYSIS,
        reasoning_ladder=("low", "high", "max"),
        token_parameter=TokenParameter.MAX_COMPLETION_TOKENS,
        temperature_policy=TemperaturePolicy.OMIT,
        pricing_key="kimi-k3",
    ),
    ModelProfile(
        # Prefix is version-specific, not the bare "grok-4" family: 4.5 documents
        # only low|medium|high, so a family-wide row would promise ``xhigh`` on a
        # model that rejects it. An unregistered Grok therefore fails closed at
        # capability validation rather than inheriting 4.6's ladder.
        #
        # The ladder is load-bearing rather than descriptive. xAI documents that
        # reasoning_effort defaults to "high" and that reasoning *cannot be
        # disabled*, while `budgets.resolve_generation_budget` enables a
        # reasoning reserve only when an effort was resolved. An empty ladder
        # would therefore pair guaranteed deep reasoning with zero reserve --
        # the 1088.HK Consultant starvation, as a certainty rather than a risk.
        prefix="grok-4.6",
        identity=ModelIdentity("xai", "grok", "openai_compatible"),
        capabilities=_NATIVE_ANALYSIS,
        reasoning_ladder=("low", "medium", "high", "xhigh"),
        token_parameter=TokenParameter.MAX_COMPLETION_TOKENS,
        temperature_policy=TemperaturePolicy.OMIT,
        pricing_key="grok-4.6",
    ),
)


def conservative_unknown_profile(model: str) -> ModelProfile:
    return ModelProfile(
        prefix=model,
        identity=ModelIdentity("unknown", "unknown", "openai_compatible"),
        capabilities=_TEXT,
        reasoning_ladder=(),
        token_parameter=TokenParameter.MAX_COMPLETION_TOKENS,
        temperature_policy=TemperaturePolicy.OMIT,
    )


def resolve_profile(
    model: str, profiles: tuple[ModelProfile, ...] = MODEL_PROFILES
) -> ModelProfile:
    normalized = model.strip().lower().split("/", 1)[-1]
    matches = [profile for profile in profiles if normalized.startswith(profile.prefix)]
    if not matches:
        return conservative_unknown_profile(normalized)
    ranked = sorted(
        matches, key=lambda profile: (len(profile.prefix), profile.priority)
    )
    winner = ranked[-1]
    if re.fullmatch(
        r"claude-(?:opus|sonnet)-4-\d+(?:-.*)?", normalized
    ) and winner.prefix in {
        "claude-opus-4",
        "claude-sonnet-4",
    }:
        return conservative_unknown_profile(normalized)
    if len(ranked) > 1:
        previous = ranked[-2]
        if (
            len(previous.prefix) == len(winner.prefix)
            and previous.priority == winner.priority
        ):
            raise ValueError(f"ambiguous model profiles for {model!r}")
    return winner


def resolve_sampling_temperature(
    profile: ModelProfile, requested: float | None
) -> float | None:
    """Translate a seat's sampling preference into provider-safe API data."""

    if requested is None or profile.temperature_policy is TemperaturePolicy.OMIT:
        return None
    if profile.temperature_policy is TemperaturePolicy.FIXED_ONE:
        return 1.0
    return requested


def adjust_reasoning(profile: ModelProfile, baseline: str, steps: int = 1) -> str:
    if Capability.REASONING_CONTROL not in profile.capabilities:
        raise UnsupportedModelCapability("model has no reviewed reasoning control")
    try:
        current = profile.reasoning_ladder.index(baseline)
    except ValueError as exc:
        raise UnsupportedModelCapability(
            f"reasoning value {baseline!r} is not supported by {profile.prefix}"
        ) from exc
    target = min(max(current + steps, 0), len(profile.reasoning_ladder) - 1)
    return profile.reasoning_ladder[target]
