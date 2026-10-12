"""Anthropic-native editorial adapter."""

from typing import Any

from langchain_core.language_models import BaseChatModel

from src.llm_runtime.adapters import transports
from src.llm_runtime.adapters.base import SeatModelRequest
from src.llm_runtime.profiles import ReasoningApiMode
from src.llm_runtime.rate_limits import limiter_for_binding

# Long-form default for a seat that expresses no preference. Anthropic counts
# thinking tokens against max_tokens, so this must stay generous enough for an
# article plus its reasoning reserve.
_DEFAULT_ANTHROPIC_OUTPUT_TOKENS = 16384


class AnthropicAdapter:
    kind = "anthropic_native"

    def build(self, request: SeatModelRequest) -> BaseChatModel:
        from src.config import config

        settings = request.settings or config
        policy = request.seat.execution_policy
        api_key = settings.get_claude_api_key()
        if not api_key:
            raise ValueError("Anthropic writer binding requires ANTHROPIC_API_KEY")

        # Honour the cross-adapter output-budget contract, as the OpenAI adapter
        # does. This was hardcoded, which happened to be invisible because the
        # only Anthropic seat (the writer) asks for exactly this default — so a
        # seat requesting a smaller cap would have silently received 16384.
        # Anthropic counts thinking tokens against max_tokens, hence the
        # long-form default when a seat expresses no preference.
        kwargs: dict[str, Any] = {
            "model": request.binding.model,
            "max_tokens": request.output_tokens or _DEFAULT_ANTHROPIC_OUTPUT_TOKENS,
            # The seat's explicit client ceiling when it has one.
            "timeout": float(
                policy.client_timeout_seconds
                if policy.client_timeout_seconds is not None
                else settings.api_timeout
            ),
            "callbacks": list(request.callbacks),
            "anthropic_api_key": api_key,
            # The binding's provider, explicitly: it sizes every request.
            "dispatch_provider": request.binding.provider,
            "budget_timeout_class": request.seat.timeout_class.value,
        }
        reasoning_mode = request.binding.profile.reasoning_api_mode
        if reasoning_mode is ReasoningApiMode.ADAPTIVE:
            kwargs["thinking"] = {"type": "adaptive"}
            kwargs["effort"] = request.reasoning_value or "high"
        elif reasoning_mode is ReasoningApiMode.MANUAL:
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": 8192}
        else:
            # Anthropic rejects a temperature other than 1.0 with thinking, so
            # the seat's temperature applies only to a model without it.
            kwargs["temperature"] = (
                policy.sampling_temperature
                if policy.sampling_temperature is not None
                else 0.7
            )
        model = transports.new_anthropic_chat(**kwargs)
        thinking = kwargs.get("thinking")
        reserve = (
            int(thinking.get("budget_tokens", 0)) if isinstance(thinking, dict) else 0
        )
        model._configured_reasoning_reserve_tokens = reserve  # type: ignore[attr-defined]
        model.rate_limiter = limiter_for_binding(
            settings,
            request.binding.identity.vendor_id,
            request.binding.endpoint_host,
        )
        return model
