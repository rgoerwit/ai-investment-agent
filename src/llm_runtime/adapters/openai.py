"""OpenAI-native chat adapter."""

from typing import Any

from langchain_core.language_models import BaseChatModel

from src.llm_runtime.adapters.base import SeatModelRequest
from src.llm_runtime.budgets import resolve_generation_budget, stamp_budget_metadata
from src.llm_runtime.profiles import resolve_sampling_temperature
from src.llm_runtime.rate_limits import limiter_for_binding
from src.llm_runtime.seats import SeatSpec


class OpenAIAdapter:
    kind = "openai_native"

    def build(self, request: SeatModelRequest) -> BaseChatModel:
        from src.config import config

        settings = request.settings or config
        policy = request.seat.execution_policy
        if request.service_tier == "flex":
            # Keep the incident-tested flex retry/downgrade transport intact
            # during facade extraction. Standard OpenAI construction below is
            # provider-local and does not import the Google SDK-heavy facade.
            from src import llms

            preference = (
                (request.reasoning_value,) if request.reasoning_value else ("medium",)
            )
            model = llms._build_openai_chat(
                request.binding.model,
                api_key=settings.get_openai_api_key(),
                callbacks=list(request.callbacks),
                max_completion_tokens=request.output_tokens or 8192,
                service_tier_label=f"openai_sdk_timeout:{request.seat.seat_id.value}",
                service_tier=request.service_tier,
                unthrottled_kind=request.seat.seat_id.value,
                effort_preference=preference,
                settings=settings,
            )
        else:
            from langchain_openai import ChatOpenAI

            budget = resolve_generation_budget(
                settings,
                intent_tokens=request.output_tokens or 8192,
                reasoning_value=request.reasoning_value,
            )
            kwargs: dict[str, Any] = {
                "model": request.binding.model,
                "api_key": settings.get_openai_api_key(),
                "callbacks": list(request.callbacks),
                "timeout": policy.client_timeout_seconds or 120,
                "max_retries": (
                    policy.sdk_max_retries if policy.sdk_max_retries is not None else 3
                ),
                "max_completion_tokens": budget.api_cap_tokens,
                "streaming": False,
                "use_responses_api": True,
                "output_version": "responses/v1",
            }
            base_url = settings.get_openai_api_base()
            if base_url:
                # Binding validation has already proved that this is an
                # OpenAI-owned endpoint under the provider-scoped schema.
                kwargs["base_url"] = base_url
            if request.reasoning_value is not None:
                kwargs["reasoning_effort"] = request.reasoning_value
            temperature = resolve_sampling_temperature(
                request.binding.profile,
                policy.sampling_temperature,
            )
            if temperature is not None:
                kwargs["temperature"] = temperature
            model = ChatOpenAI(**kwargs)
            stamp_budget_metadata(
                model,
                callbacks=request.callbacks,
                budget=budget,
                intent_attr="_configured_max_completion_tokens",
                api_attr="_configured_api_completion_tokens",
            )
        model.rate_limiter = limiter_for_binding(
            settings,
            request.binding.identity.vendor_id,
            request.binding.endpoint_host,
        )
        return model

    def build_transport_fallback(self, model_name: str) -> BaseChatModel:
        """Build the Consultant's alternate OpenAI transport in one owned seam."""

        from langchain_openai import ChatOpenAI
        from pydantic import SecretStr

        from src.config import config

        raw_api_key = config.get_openai_api_key()
        api_key = SecretStr(raw_api_key) if raw_api_key else None
        base_url = config.get_openai_api_base()
        kwargs: dict[str, Any] = {
            "model": model_name,
            "timeout": 120,
            "max_retries": 3,
            "streaming": False,
            "api_key": api_key,
        }
        if base_url:
            kwargs["base_url"] = base_url
        else:
            kwargs.update(
                use_responses_api=True,
                output_version="responses/v1",
            )
        return ChatOpenAI(**kwargs)

    def prepare_messages(self, messages: list[Any], *, seat: SeatSpec) -> list[Any]:
        del seat
        return list(messages)
