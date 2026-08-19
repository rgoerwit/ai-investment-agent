"""Google-native chat adapter."""

from typing import Any

from langchain_core.language_models import BaseChatModel

from src.llm_runtime.adapters.base import SeatModelRequest
from src.llm_runtime.rate_limits import limiter_for_binding
from src.llm_runtime.seats import ModelIntent, SeatSpec


class GoogleAdapter:
    kind = "google_native"

    def build(self, request: SeatModelRequest) -> BaseChatModel:
        # The compatibility facade still owns the incident-tested tiered transport
        # subclass during extraction. Its provider SDK import moves here when the
        # old public factories are removed.
        from src import llms
        from src.config import config
        from src.llm_runtime.seats import ReasoningAdjustment, SeatId

        settings = request.settings or config
        policy = request.seat.execution_policy

        if not policy.reasoning_control_enabled:
            llm = llms.create_gemini_model(
                request.binding.model,
                temperature=policy.sampling_temperature or 0.0,
                timeout=int(policy.client_timeout_seconds or settings.api_timeout),
                max_retries=(
                    policy.sdk_max_retries
                    if policy.sdk_max_retries is not None
                    else settings.api_retry_attempts
                ),
                callbacks=list(request.callbacks),
                max_output_tokens=request.output_tokens,
                service_tier=request.service_tier,
                api_key=settings.get_google_api_key(),
                settings=settings,
            )
            llm.rate_limiter = limiter_for_binding(
                settings,
                request.binding.identity.vendor_id,
                request.binding.endpoint_host,
            )
            return llm

        if request.seat.seat_id is SeatId.ARTICLE_WRITER_BASE_FALLBACK:
            return llms.create_writer_fallback_llm(
                callbacks=list(request.callbacks),
                model=request.binding.model,
                service_tier=request.service_tier,
                api_key=settings.get_google_api_key(),
                settings=settings,
            )
        if request.binding.intent in {
            ModelIntent.FAST,
            ModelIntent.CLASSIFIER,
        }:
            return llms.create_quick_thinking_llm(
                temperature=(
                    policy.sampling_temperature
                    if policy.sampling_temperature is not None
                    else 0.3
                ),
                model=request.binding.model,
                timeout=(
                    int(policy.client_timeout_seconds)
                    if policy.client_timeout_seconds is not None
                    else None
                ),
                max_retries=policy.sdk_max_retries,
                callbacks=list(request.callbacks),
                max_output_tokens=request.output_tokens,
                service_tier=request.service_tier,
                thinking_level_bump=(
                    request.seat.normal_reasoning_adjustment
                    is ReasoningAdjustment.ONE_STEP
                    and not request.quick_mode
                ),
                api_key=settings.get_google_api_key(),
                settings=settings,
            )

        reasoning = request.reasoning_value
        if reasoning is None and request.binding.profile.reasoning_ladder:
            reasoning = (
                "low"
                if request.quick_mode or request.binding.intent is ModelIntent.FAST
                else "high"
            )
        llm = llms.create_gemini_model(
            request.binding.model,
            temperature=(
                policy.sampling_temperature
                if policy.sampling_temperature is not None
                else 0.1
                if request.binding.intent is not ModelIntent.FAST
                else 0.3
            ),
            timeout=int(policy.client_timeout_seconds or settings.api_timeout),
            max_retries=(
                policy.sdk_max_retries
                if policy.sdk_max_retries is not None
                else settings.api_retry_attempts
            ),
            callbacks=list(request.callbacks),
            thinking_level=reasoning,
            max_output_tokens=request.output_tokens,
            reserve_class=(
                "deep"
                if request.binding.intent
                in {ModelIntent.REASONING, ModelIntent.CRITICAL, ModelIntent.ESCALATION}
                else "default"
            ),
            service_tier=request.service_tier,
            api_key=settings.get_google_api_key(),
            settings=settings,
        )
        llm.rate_limiter = limiter_for_binding(
            settings,
            request.binding.identity.vendor_id,
            request.binding.endpoint_host,
        )
        return llm

    def prepare_messages(self, messages: list[Any], *, seat: SeatSpec) -> list[Any]:
        del seat
        from src.agents.message_utils import filter_messages_for_gemini

        return filter_messages_for_gemini(messages)
