"""OpenAI-native chat adapter."""

from typing import Any

from langchain_core.language_models import BaseChatModel

from src.llm_runtime.adapters.base import SeatModelRequest
from src.llm_runtime.budgets import resolve_generation_budget, stamp_budget_metadata
from src.llm_runtime.profiles import resolve_sampling_temperature
from src.llm_runtime.rate_limits import limiter_for_binding
from src.llm_runtime.seats import SEATS, SeatId
from src.llm_runtime.timeouts import (
    construction_transport_attempts,
    policy_for_seat,
    sdk_max_retries,
)


class OpenAIAdapter:
    kind = "openai_native"

    def build(self, request: SeatModelRequest) -> BaseChatModel:
        from src.config import config

        settings = request.settings or config
        policy = request.seat.execution_policy
        # The OpenAI SDK fixes retries on the client (no per-request override),
        # so the client is built with the seat's policy attempts: one when the
        # runtime loop owns retries. The request timeout is the policy's
        # standard ceiling; a tiered transport re-sizes each request at dispatch.
        attempts = construction_transport_attempts(
            request.seat, vendor="openai", cfg=settings
        )
        request_cap = policy_for_seat(
            request.seat, provider="openai", cfg=settings
        ).request_cap
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
                intent=request.binding.intent,
                settings=settings,
                include_reasoning_output=request.include_reasoning_output,
                request_timeout_seconds=request_cap,
                transport_attempts=attempts,
                timeout_class=request.seat.timeout_class,
            )
        else:
            from langchain_openai import ChatOpenAI

            budget = resolve_generation_budget(
                settings,
                intent_tokens=request.output_tokens or 8192,
                reasoning_value=request.reasoning_value,
                intent=request.binding.intent,
            )
            kwargs: dict[str, Any] = {
                "model": request.binding.model,
                "api_key": settings.get_openai_api_key(),
                "callbacks": list(request.callbacks),
                "timeout": request_cap,
                "max_retries": sdk_max_retries("openai", attempts),
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
            if request.include_reasoning_output:
                reasoning: dict[str, Any] = {"summary": "auto"}
                if request.reasoning_value is not None:
                    reasoning["effort"] = request.reasoning_value
                kwargs["reasoning"] = reasoning
            elif request.reasoning_value is not None:
                kwargs["reasoning_effort"] = request.reasoning_value
            temperature = resolve_sampling_temperature(
                request.binding.profile,
                policy.sampling_temperature,
            )
            if temperature is not None:
                kwargs["temperature"] = temperature
            if request.binding.failover_model:
                # Only the tiered transport can serve a failover attempt; with
                # service_tier="auto" it behaves exactly like ChatOpenAI.
                from src import llms

                kwargs["budget_timeout_class"] = request.seat.timeout_class.value
                model = llms._get_flex_fallback_chat_openai_cls()(**kwargs)
            else:
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
        consultant = SEATS[SeatId.CONSULTANT]
        kwargs: dict[str, Any] = {
            "model": model_name,
            "timeout": policy_for_seat(
                consultant, provider="openai", cfg=config
            ).request_cap,
            # The Consultant's policy: one request per dispatch (was 3 retries).
            "max_retries": sdk_max_retries(
                "openai",
                construction_transport_attempts(
                    consultant, vendor="openai", cfg=config
                ),
            ),
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
