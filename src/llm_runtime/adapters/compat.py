"""Restricted OpenAI-compatible adapter for qualified single-shot seats."""

from typing import Any

from langchain_core.language_models import BaseChatModel

from src.llm_runtime.adapters.base import SeatModelRequest
from src.llm_runtime.budgets import resolve_generation_budget, stamp_budget_metadata
from src.llm_runtime.profiles import resolve_sampling_temperature
from src.llm_runtime.provider_policy import (
    is_provider_qualified,
    provider_default_headers,
)
from src.llm_runtime.rate_limits import limiter_for_binding
from src.llm_runtime.seats import SeatId, SeatSpec

# Providers whose effort comes from the seat's resolved intent rather than an
# explicit per-binding override.
_SEAT_RESOLVED_EFFORT_PROVIDERS = frozenset({"moonshot", "xai"})


class CompatibleAdapter:
    kind = "openai_compatible"

    _REVIEW_SEATS = {
        SeatId.CONSULTANT,
        SeatId.AUDITOR,
        SeatId.AUDITOR_ESCALATION,
        SeatId.EDITOR,
        SeatId.ARTICLE_WRITER_REVIEW_FALLBACK,
    }
    _APAC_SEATS = {SeatId.APAC, SeatId.APAC_DIRECT_RETRY}

    def build(self, request: SeatModelRequest) -> BaseChatModel | None:
        seat_id = request.seat.seat_id
        if seat_id not in self._APAC_SEATS | self._REVIEW_SEATS:
            raise ValueError(
                "compatible transport is restricted to reviewed APAC/Moonshot seats"
            )
        from langchain_openai import ChatOpenAI

        from src.config import config

        settings = request.settings or config
        policy = request.seat.execution_policy

        provider = request.binding.provider
        if not is_provider_qualified(provider, request.seat.binding_group):
            raise ValueError(
                f"provider {provider!r} is not qualified for "
                f"{request.seat.binding_group.value!r} compatible seats"
            )

        base_url, api_key = {
            "deepseek": (settings.deepseek_api_base, settings.deepseek_api_key),
            "zai": (settings.zai_api_base, settings.zai_api_key),
            "moonshot": (settings.moonshot_api_base, settings.moonshot_api_key),
            "xai": (settings.xai_api_base, settings.xai_api_key),
        }[provider]
        thinking = seat_id is SeatId.APAC
        # Review-plane providers take the seat-resolved effort; the APAC pair
        # keeps its explicit override. xAI must be in this set: its documented
        # default is "high" and reasoning cannot be disabled, so sending no
        # effort would pair the deepest-but-one reasoning with the *default*
        # reserve rather than the deep one.
        reasoning_value = (
            request.reasoning_value
            if provider in _SEAT_RESOLVED_EFFORT_PROVIDERS
            else request.binding.reasoning_value_override
        )
        if thinking and reasoning_value is None:
            # Preserve the established deepest default while honoring an explicit
            # per-seat override that binding validation already checked.
            reasoning_value = "max"
        budget = resolve_generation_budget(
            settings,
            intent_tokens=request.output_tokens or 8192,
            reasoning_value=reasoning_value,
        )
        kwargs: dict[str, Any] = {
            "model": request.binding.model,
            "base_url": base_url,
            "api_key": api_key.get_secret_value(),
            # One knob for "how long may a compatible vendor's client wait",
            # shared with the legacy path's _apply_openai_api_base. APAC keeps
            # its own longer allowance for the deepest-reasoning re-issue.
            "timeout": policy.client_timeout_seconds
            or (
                240
                if seat_id in self._APAC_SEATS
                else settings.openai_compatible_client_timeout_seconds
            ),
            "max_retries": (
                policy.sdk_max_retries
                if policy.sdk_max_retries is not None
                else 1
                if thinking
                else 3
            ),
            "callbacks": list(request.callbacks),
            "max_completion_tokens": budget.api_cap_tokens,
            "streaming": False,
        }
        if seat_id in self._APAC_SEATS:
            kwargs["extra_body"] = {
                "thinking": {"type": "enabled" if thinking else "disabled"}
            }
        default_headers = provider_default_headers(provider)
        if default_headers:
            kwargs["default_headers"] = default_headers
        if reasoning_value is not None:
            kwargs["reasoning_effort"] = reasoning_value
        temperature = resolve_sampling_temperature(
            request.binding.profile,
            policy.sampling_temperature,
        )
        if temperature is not None:
            kwargs["temperature"] = temperature
        llm = ChatOpenAI(**kwargs)
        llm.rate_limiter = limiter_for_binding(
            settings,
            request.binding.identity.vendor_id,
            request.binding.endpoint_host,
        )
        stamp_budget_metadata(
            llm,
            callbacks=list(request.callbacks),
            budget=budget,
            intent_attr="_configured_max_completion_tokens",
            api_attr="_configured_api_completion_tokens",
        )
        return llm

    def prepare_messages(self, messages: list[Any], *, seat: SeatSpec) -> list[Any]:
        del seat
        return list(messages)
