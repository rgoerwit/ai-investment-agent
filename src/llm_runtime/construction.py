"""Application-owned seat construction shared by graph and auxiliary workflows."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel

from src.config import Settings, config
from src.llm_runtime.adapters.base import SeatModelRequest
from src.llm_runtime.bindings import BindingPlan, resolve_binding_plan
from src.llm_runtime.factory import SeatModelFactory
from src.llm_runtime.profiles import ModelProfile, adjust_reasoning, resolve_profile
from src.llm_runtime.seats import (
    SEATS,
    ModelIntent,
    ReasoningAdjustment,
    SeatId,
)


@dataclass(frozen=True)
class LegacySeatRequest:
    seat_id: SeatId
    settings: Settings
    quick_mode: bool
    callbacks: tuple[BaseCallbackHandler, ...]
    output_tokens: int | None
    model_override: str | None
    resolved_model: str


@dataclass(frozen=True)
class LegacyGraphFactories:
    quick: Callable[..., BaseChatModel]
    deep: Callable[..., BaseChatModel]
    apex: Callable[..., BaseChatModel]
    consultant: Callable[..., BaseChatModel | None]
    auditor: Callable[..., BaseChatModel | None]
    apac: Callable[..., BaseChatModel | None]


LegacyBuilder = Callable[[LegacySeatRequest], BaseChatModel | None]


@dataclass(frozen=True)
class WriterSeatTier:
    label: str
    build: Callable[[], BaseChatModel]


def reasoning_value_for_seat(
    profile: ModelProfile,
    intent: ModelIntent,
    *,
    adjust: bool,
) -> str | None:
    ladder = profile.reasoning_ladder
    if not ladder:
        return None
    prose_preferences = (
        ("high", "medium", "low")
        if profile.identity.vendor_id == "anthropic"
        else ("low", "minimal", "none")
    )
    reasoning_preferences = (
        ("high", "medium", "low")
        if profile.identity.vendor_id == "google"
        else ("medium", "high", "low")
    )
    preferences = {
        ModelIntent.FAST: ("low", "minimal", "none"),
        ModelIntent.CLASSIFIER: ("low", "minimal", "none"),
        ModelIntent.PROSE: prose_preferences,
        ModelIntent.REASONING: reasoning_preferences,
        ModelIntent.CRITICAL: ("high", "medium"),
        ModelIntent.ESCALATION: ("max", "xhigh", "high"),
    }[intent]
    baseline = next((value for value in preferences if value in ladder), ladder[-1])
    return adjust_reasoning(profile, baseline) if adjust else baseline


def build_legacy_model(
    request: LegacySeatRequest,
    *,
    graph_factories: LegacyGraphFactories | None = None,
) -> BaseChatModel | None:
    """Construct any legacy seat through one compatibility dispatcher."""

    from src import llms

    seat_id = request.seat_id
    settings = request.settings
    quick_mode = request.quick_mode
    callbacks = list(request.callbacks)
    output_tokens = request.output_tokens
    model_override = request.model_override
    quick_factory = (
        graph_factories.quick if graph_factories else llms.create_quick_thinking_llm
    )
    deep_factory = (
        graph_factories.deep if graph_factories else llms.create_deep_thinking_llm
    )
    apex_factory = graph_factories.apex if graph_factories else llms.create_apex_llm
    consultant_factory = (
        graph_factories.consultant if graph_factories else llms.get_consultant_llm
    )
    auditor_factory = (
        graph_factories.auditor if graph_factories else llms.create_auditor_llm
    )
    apac_factory = (
        graph_factories.apac if graph_factories else llms.create_apac_specialist_llm
    )

    if seat_id is SeatId.ARTICLE_WRITER:
        return llms.create_writer_llm(
            callbacks=callbacks,
            model=request.resolved_model,
            api_key_override=settings.get_claude_api_key(),
            settings=settings,
        )
    if seat_id is SeatId.ARTICLE_WRITER_REVIEW_FALLBACK:
        return llms.create_writer_openai_fallback_llm(
            callbacks=callbacks,
            model=request.resolved_model,
            settings=settings,
        )
    if seat_id is SeatId.ARTICLE_WRITER_BASE_FALLBACK:
        return llms.create_writer_fallback_llm(
            callbacks=callbacks,
            model=request.resolved_model,
            api_key=settings.get_google_api_key(),
            settings=settings,
        )
    if seat_id is SeatId.EDITOR:
        return llms.create_editor_llm(
            callbacks=callbacks,
            model=request.resolved_model,
            settings=settings,
        )
    if seat_id is SeatId.CONSULTANT:
        return consultant_factory(
            callbacks=callbacks,
            quick_mode=quick_mode,
            max_completion_tokens=output_tokens,
            model=request.resolved_model,
            settings=settings,
        )
    if seat_id in {SeatId.AUDITOR, SeatId.AUDITOR_ESCALATION}:
        return auditor_factory(
            callbacks=callbacks,
            quick_mode=quick_mode,
            max_completion_tokens=output_tokens,
            model_name_override=request.resolved_model,
            settings=settings,
        )
    if seat_id in {SeatId.APAC, SeatId.APAC_DIRECT_RETRY}:
        return apac_factory(
            callbacks=callbacks,
            quick_mode=quick_mode,
            max_completion_tokens=output_tokens,
            thinking_enabled=seat_id is SeatId.APAC,
            settings=settings,
        )
    if seat_id is SeatId.SEMANTIC_JUDGE and model_override:
        profile = resolve_profile(model_override)
        if profile.identity.adapter_kind == "openai_native":
            return llms.create_consultant_llm(
                model=model_override,
                quick_mode=True,
                max_completion_tokens=output_tokens,
                callbacks=callbacks,
            )
    if seat_id is SeatId.HEALTH_CHECK:
        from src.runtime_config import get_runtime_config

        runtime = get_runtime_config(settings)
        return llms.create_gemini_model(
            request.resolved_model or runtime.quick_think_llm,
            temperature=0.0,
            timeout=10,
            max_retries=1,
            service_tier="standard",
            api_key=settings.get_google_api_key(),
            settings=settings,
        )
    if seat_id is SeatId.PORTFOLIO_MACRO_CLASSIFIER:
        factory = quick_factory if quick_mode else deep_factory
        return factory(
            model=request.resolved_model,
            callbacks=callbacks,
            max_output_tokens=output_tokens,
            settings=settings,
        )
    if seat_id in {SeatId.SENIOR_FUNDAMENTALS, SeatId.PORTFOLIO_MANAGER}:
        return apex_factory(
            (
                "senior_fundamentals"
                if seat_id is SeatId.SENIOR_FUNDAMENTALS
                else "portfolio_manager"
            ),
            quick_mode=quick_mode,
            callbacks=callbacks,
            max_output_tokens=output_tokens,
            settings=settings,
        )
    if seat_id is SeatId.ANALYST_RETRY:
        return deep_factory(
            model=request.resolved_model,
            callbacks=callbacks,
            max_output_tokens=output_tokens,
            settings=settings,
        )
    policy = SEATS[seat_id].execution_policy
    spec = SEATS[seat_id]
    if quick_mode or spec.normal_intent in {ModelIntent.FAST, ModelIntent.CLASSIFIER}:
        return quick_factory(
            temperature=(
                policy.sampling_temperature
                if policy.sampling_temperature is not None
                else 0.3
            ),
            model=model_override or request.resolved_model,
            timeout=(
                int(policy.client_timeout_seconds)
                if policy.client_timeout_seconds is not None
                else None
            ),
            max_retries=policy.sdk_max_retries,
            callbacks=callbacks,
            max_output_tokens=output_tokens,
            service_tier=("standard" if policy.standard_tier_only else None),
            thinking_level_bump=(seat_id is SeatId.VALUE_TRAP and not quick_mode),
            api_key=settings.get_google_api_key(),
            settings=settings,
        )
    return deep_factory(
        model=model_override or request.resolved_model,
        timeout=(
            int(policy.client_timeout_seconds)
            if policy.client_timeout_seconds is not None
            else None
        ),
        max_retries=policy.sdk_max_retries,
        callbacks=callbacks,
        max_output_tokens=output_tokens,
        api_key=settings.get_google_api_key(),
        settings=settings,
    )


def _default_legacy_builder(request: LegacySeatRequest) -> BaseChatModel | None:
    return build_legacy_model(request)


def build_model_for_seat(
    seat_id: SeatId,
    *,
    settings: Settings = config,
    plan: BindingPlan | None = None,
    factory: SeatModelFactory | None = None,
    quick_mode: bool = False,
    callbacks: Sequence[BaseCallbackHandler] = (),
    output_tokens: int | None = None,
    service_tier: str | None = None,
    model_override: str | None = None,
    legacy_builder: LegacyBuilder | None = None,
) -> BaseChatModel | None:
    """Construct one fresh client from a canonical seat binding."""

    resolved_plan = plan or resolve_binding_plan(settings)
    callback_list = list(callbacks)
    status = resolved_plan.status_for(seat_id, quick_mode=quick_mode)
    if not status.enabled and (
        resolved_plan.schema == "new"
        or (quick_mode and SEATS[seat_id].disabled_in_quick_mode)
    ):
        return None
    if resolved_plan.schema == "legacy":
        builder = legacy_builder or _default_legacy_builder
        binding = (
            resolved_plan.quick_bindings if quick_mode else resolved_plan.bindings
        )[seat_id]
        return builder(
            LegacySeatRequest(
                seat_id=seat_id,
                settings=settings,
                quick_mode=quick_mode,
                callbacks=tuple(callback_list),
                output_tokens=output_tokens,
                model_override=model_override,
                resolved_model=binding.model,
            )
        )

    binding = resolved_plan.for_seat(seat_id, quick_mode=quick_mode)
    if model_override and model_override != binding.model:
        raise ValueError(
            f"{seat_id.value} model override {model_override!r} differs from the "
            f"resolved binding {binding.model!r}; pin it in environment settings"
        )
    spec = SEATS[seat_id]
    if spec.execution_policy.standard_tier_only or (
        quick_mode and spec.execution_policy.standard_tier_in_quick_mode
    ):
        service_tier = "standard"
    elif service_tier is None:
        if binding.provider == "google":
            service_tier = settings.google_service_tier
        elif binding.provider == "openai":
            service_tier = settings.openai_service_tier
    for callback in callback_list:
        bind_identity = getattr(callback, "bind_identity", None)
        if callable(bind_identity):
            bind_identity(
                seat_id=seat_id.value,
                binding_group=spec.binding_group.value,
                vendor_id=binding.identity.vendor_id,
                model_lineage=binding.identity.model_lineage,
                adapter_kind=binding.identity.adapter_kind,
                endpoint_host=binding.endpoint_host,
            )
    reasoning_value = binding.reasoning_value_override
    if not spec.execution_policy.reasoning_control_enabled:
        reasoning_value = None
    elif reasoning_value is None:
        reasoning_value = reasoning_value_for_seat(
            binding.profile,
            binding.intent,
            adjust=(
                spec.normal_reasoning_adjustment is ReasoningAdjustment.ONE_STEP
                and not quick_mode
            ),
        )
    model = (factory or SeatModelFactory()).build(
        SeatModelRequest(
            binding=binding,
            seat=spec,
            quick_mode=quick_mode,
            callbacks=tuple(callback_list),
            output_tokens=(
                output_tokens
                if spec.execution_policy.output_token_override_enabled
                else None
            ),
            reasoning_value=reasoning_value,
            service_tier=service_tier,
            settings=settings,
        )
    )
    if model is None and status.enabled:
        raise RuntimeError(f"active seat {seat_id.value} returned no model")
    return model


def build_required_model_for_seat(*args: Any, **kwargs: Any) -> BaseChatModel:
    model = build_model_for_seat(*args, **kwargs)
    if model is None:
        seat_id = args[0] if args else kwargs.get("seat_id", "unknown")
        raise RuntimeError(f"required seat {seat_id} is unavailable")
    return model


def writer_seat_fallback_chain(
    *,
    settings: Settings = config,
    callbacks: Sequence[BaseCallbackHandler] = (),
    plan: BindingPlan | None = None,
    factory: SeatModelFactory | None = None,
) -> list[WriterSeatTier]:
    """Return lazy, provider-neutral editorial fallback tiers."""

    resolved_plan = plan or resolve_binding_plan(settings)
    if resolved_plan.schema == "legacy":
        from src.llms import writer_fallback_chain

        return [
            WriterSeatTier(tier.label, tier.build)
            for tier in writer_fallback_chain(callbacks=list(callbacks))
        ]

    model_factory = factory or SeatModelFactory()
    tiers: list[WriterSeatTier] = []
    candidates = (
        ("review_group", SeatId.ARTICLE_WRITER_REVIEW_FALLBACK),
        ("base_group", SeatId.ARTICLE_WRITER_BASE_FALLBACK),
    )
    for label, seat_id in candidates:
        if not resolved_plan.statuses[seat_id].enabled:
            continue

        def build(seat: SeatId = seat_id) -> BaseChatModel:
            return build_required_model_for_seat(
                seat,
                settings=settings,
                plan=resolved_plan,
                factory=model_factory,
                callbacks=callbacks,
                output_tokens=16_384,
            )

        tiers.append(WriterSeatTier(label, build))
    return tiers
