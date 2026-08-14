"""Resolve typed settings into an immutable, validated seat binding plan."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from src.llm_runtime.capabilities import Capability
from src.llm_runtime.identities import (
    ModelIdentity,
    vendor_for_endpoint_host,
)
from src.llm_runtime.profiles import ModelProfile, resolve_profile
from src.llm_runtime.provider_policy import (
    is_provider_qualified,
    provider_credential,
    provider_endpoint_host,
    provider_for_group,
)
from src.llm_runtime.seats import (
    SEATS,
    BindingGroup,
    ModelIntent,
    ReasoningAdjustment,
    SeatId,
    SeatSpec,
)


class BindingConfigurationError(ValueError):
    def __init__(self, errors: list[str]) -> None:
        self.errors = tuple(errors)
        super().__init__("Invalid LLM binding configuration:\n- " + "\n- ".join(errors))


@dataclass(frozen=True)
class ResolvedBinding:
    seat_id: SeatId
    provider: str
    model: str
    identity: ModelIdentity
    intent: ModelIntent
    required_capabilities: frozenset[Capability]
    endpoint_host: str | None
    profile: ModelProfile
    reasoning_value_override: str | None = None


@dataclass(frozen=True)
class SeatStatus:
    enabled: bool
    reason: str | None = None
    mode: str = "required"
    provider_ready: bool = True

    @classmethod
    def resolve(
        cls, mode: str, *, provider_ready: bool, reason: str | None
    ) -> "SeatStatus":
        if mode == "off":
            return cls(False, "configured off", mode, provider_ready)
        if provider_ready:
            return cls(True, None, mode, True)
        return cls(False, reason or "provider unavailable", mode, False)


@dataclass(frozen=True)
class BindingPlan:
    bindings: Mapping[SeatId, ResolvedBinding]
    quick_bindings: Mapping[SeatId, ResolvedBinding]
    statuses: Mapping[SeatId, SeatStatus]
    schema: str

    def status_for(self, seat_id: SeatId, *, quick_mode: bool = False) -> SeatStatus:
        """Return effective availability for the requested execution mode."""

        status = self.statuses[seat_id]
        if quick_mode and status.enabled and SEATS[seat_id].disabled_in_quick_mode:
            return SeatStatus(
                enabled=False,
                reason="disabled in quick mode",
                mode=status.mode,
                provider_ready=status.provider_ready,
            )
        return status

    def for_seat(self, seat_id: SeatId, *, quick_mode: bool = False) -> ResolvedBinding:
        status = self.status_for(seat_id, quick_mode=quick_mode)
        if not status.enabled:
            raise BindingConfigurationError(
                [f"{seat_id.value} is unavailable: {status.reason or 'disabled'}"]
            )
        return (self.quick_bindings if quick_mode else self.bindings)[seat_id]

    def reachable_bindings(
        self, *, quick_mode: bool = False
    ) -> tuple[ResolvedBinding, ...]:
        bindings = self.quick_bindings if quick_mode else self.bindings
        return tuple(
            binding
            for seat_id, binding in bindings.items()
            if self.status_for(seat_id, quick_mode=quick_mode).enabled
        )

    def telemetry(self, settings: Any) -> dict[str, Any]:
        """Return a secret-free snapshot suitable for persisted artifacts."""

        seats: dict[str, dict[str, Any]] = {}
        for seat_id, binding in self.bindings.items():
            spec = SEATS[seat_id]
            quick = self.quick_bindings[seat_id]
            status = self.statuses[seat_id]
            quick_status = self.status_for(seat_id, quick_mode=True)
            seats[seat_id.value] = {
                "binding_group": spec.binding_group.value,
                "authority_stage": spec.authority_stage.value,
                "enabled": status.enabled,
                "mode": status.mode,
                "provider_ready": status.provider_ready,
                "unavailable_reason": status.reason,
                "quick_enabled": quick_status.enabled,
                "quick_unavailable_reason": quick_status.reason,
                "vendor": binding.identity.vendor_id,
                "lineage": binding.identity.model_lineage,
                "adapter": binding.identity.adapter_kind,
                "endpoint_host": binding.endpoint_host,
                "model": binding.model,
                "quick_model": quick.model,
                "normal_intent": binding.intent.value,
                "quick_intent": quick.intent.value,
                "reasoning_override": binding.reasoning_value_override,
                "quick_reasoning_override": quick.reasoning_value_override,
                "required_capabilities": sorted(
                    capability.value for capability in binding.required_capabilities
                ),
            }
        return {
            "schema": self.schema,
            "seats": seats,
            "independence": self.independence_telemetry(settings),
        }

    def independence_telemetry(self, settings: Any) -> dict[str, Any]:
        base = self.bindings[SeatId.PORTFOLIO_MANAGER].identity

        def row(seat_id: SeatId, required: bool, waiver_reason: str) -> dict[str, Any]:
            identity = self.bindings[seat_id].identity
            independent = (
                base.vendor_id != identity.vendor_id
                and base.model_lineage != identity.model_lineage
            )
            return {
                "required": required,
                "satisfied": independent,
                "waiver_reason": None if required else waiver_reason,
            }

        return {
            "review": row(
                SeatId.CONSULTANT,
                bool(getattr(settings, "llm_require_review_independence", False)),
                str(getattr(settings, "llm_review_independence_waiver_reason", "")),
            ),
            "regional": row(
                SeatId.APAC,
                bool(getattr(settings, "llm_require_regional_independence", False)),
                str(getattr(settings, "llm_regional_independence_waiver_reason", "")),
            ),
        }


_NEW_SELECTOR_FIELDS = {
    "llm_base_provider",
    "llm_review_provider",
    "llm_regional_provider",
    "llm_writer_provider",
    "llm_operational_provider",
    "llm_judge_provider",
}
_LEGACY_BINDING_FIELDS = {
    "llm_provider",
    "deep_think_llm",
    "quick_think_llm",
    "apex_model",
    "apex_quick_model",
    "consultant_model",
    "consultant_quick_model",
    "auditor_model",
    "auditor_quick_model",
    "auditor_escalation_model",
    "editor_model",
    "writer_model",
    "apac_specialist_model",
    "apac_specialist_base_url",
}
_PROVIDER_MODEL_FIELDS = {
    ("google", "fast"): "google_llm_fast_model",
    ("google", "reasoning"): "google_llm_reasoning_model",
    ("google", "critical"): "google_llm_critical_model",
    ("openai", "fast"): "openai_llm_fast_model",
    ("openai", "reasoning"): "openai_llm_reasoning_model",
    ("openai", "critical"): "openai_llm_critical_model",
    ("openai", "escalation"): "openai_llm_escalation_model",
    ("moonshot", "fast"): "moonshot_llm_fast_model",
    ("moonshot", "reasoning"): "moonshot_llm_reasoning_model",
    ("moonshot", "critical"): "moonshot_llm_critical_model",
    ("moonshot", "escalation"): "moonshot_llm_escalation_model",
    # No ("xai", "escalation") row: the suffix map below routes ESCALATION to
    # "critical" for every non-OpenAI provider, so such a row would be dead.
    ("xai", "fast"): "xai_llm_fast_model",
    ("xai", "reasoning"): "xai_llm_reasoning_model",
    ("xai", "critical"): "xai_llm_critical_model",
    ("anthropic", "prose"): "anthropic_llm_prose_model",
    ("deepseek", "reasoning"): "deepseek_llm_reasoning_model",
    ("zai", "reasoning"): "zai_llm_reasoning_model",
}


def _schema(settings: Any) -> str:
    explicit = set(settings.model_fields_set)
    # An empty selector is *unset*, not "selected as blank". `_provider_for_group`
    # already falls back to the default provider on a falsy value, so counting a
    # blank `LLM_BASE_PROVIDER=` as an opt-in would silently activate the new
    # schema from a placeholder line — and then reject the operator's real legacy
    # keys as "mixed". The two readers must agree on what empty means.
    new = {
        field
        for field in explicit & _NEW_SELECTOR_FIELDS
        if str(getattr(settings, field, "") or "").strip()
    }
    legacy: set[str] = set()
    for field in explicit & _LEGACY_BINDING_FIELDS:
        value = getattr(settings, field, None)
        default = type(settings).model_fields[field].default
        if value in {None, ""} and default in {None, ""}:
            continue
        if value != default:
            legacy.add(field)
    if new and legacy:
        conflicts = ", ".join(sorted(new | legacy))
        raise BindingConfigurationError(
            [f"new and legacy LLM keys are mixed: {conflicts}"]
        )
    return "new" if new else "legacy"


def _base_group_cli_model(
    settings: Any, spec: SeatSpec, intent: ModelIntent
) -> str | None:
    """Resolve `--quick-model` / `--deep-model` for one seat, or ``None``.

    Scope is deliberately narrow. Both flags apply only to the **base** group —
    the research fleet they were written for — so they cannot silently re-point
    the adversarial review plane, the writer, or the operational helpers, whose
    whole purpose is to be bound elsewhere.

    ``--quick-model`` drives the ``fast`` intent and ``--deep-model`` the
    ``reasoning`` intent, and **nothing else**. In particular ``critical`` (the
    two gate-critical APEX seats) is untouched: under the legacy schema
    ``APEX_MODEL`` already superseded ``DEEP_MODEL`` for those seats, so
    extending the flag to ``critical`` would grant it authority it never had over
    the seats with the densest incident history in the repo. Pin those with
    ``LLM_SEAT_MODEL_OVERRIDES`` instead.
    """
    if spec.binding_group is not BindingGroup.BASE:
        return None
    from src.runtime_config import get_runtime_config

    runtime = get_runtime_config(settings)
    if intent is ModelIntent.FAST:
        value = runtime.base_fast_model_override
    elif intent is ModelIntent.REASONING:
        value = runtime.base_reasoning_model_override
    else:
        return None
    return str(value).strip() if value and str(value).strip() else None


def _model_for(
    settings: Any,
    schema: str,
    spec: SeatSpec,
    provider: str,
    *,
    quick_mode: bool,
) -> str:
    if schema == "legacy":
        if spec.seat_id is SeatId.ARTICLE_WRITER:
            return str(settings.writer_model)
        if spec.seat_id is SeatId.CONSULTANT:
            return str(
                settings.consultant_quick_model
                if quick_mode
                else settings.consultant_model
            )
        if spec.seat_id in {SeatId.EDITOR, SeatId.ARTICLE_WRITER_REVIEW_FALLBACK}:
            return str(settings.editor_model or settings.consultant_model)
        if spec.seat_id is SeatId.AUDITOR:
            return str(
                settings.auditor_quick_model
                if quick_mode
                else settings.auditor_model or settings.consultant_model
            )
        if spec.seat_id is SeatId.AUDITOR_ESCALATION:
            return str(
                settings.auditor_escalation_model
                or settings.auditor_model
                or settings.consultant_model
            )
        if spec.seat_id in {SeatId.APAC, SeatId.APAC_DIRECT_RETRY}:
            return str(settings.apac_specialist_model)
        if spec.seat_id is SeatId.ARTICLE_WRITER_BASE_FALLBACK:
            return str(settings.deep_think_llm)
        if spec.seat_id is SeatId.SENIOR_FUNDAMENTALS:
            if quick_mode:
                return str(settings.apex_quick_model or settings.quick_think_llm)
            return str(settings.apex_model or settings.quick_think_llm)
        if spec.seat_id is SeatId.PORTFOLIO_MANAGER:
            if quick_mode:
                return str(settings.apex_quick_model or settings.quick_think_llm)
            return str(settings.apex_model or settings.deep_think_llm)
        intent = spec.quick_intent if quick_mode else spec.normal_intent
        if not quick_mode and intent in {
            ModelIntent.REASONING,
            ModelIntent.CRITICAL,
            ModelIntent.ESCALATION,
        }:
            return str(settings.deep_think_llm)
        return str(settings.quick_think_llm)
    intent = spec.quick_intent if quick_mode else spec.normal_intent
    cli_model = _base_group_cli_model(settings, spec, intent)
    if cli_model:
        # Documented precedence is CLI > shell env > .env > default, so a
        # command-line model outranks an `LLM_SEAT_MODEL_OVERRIDES` pin. The
        # resulting model is still vendor-checked against the group's provider
        # by `_resolve_mode_binding`, so a cross-vendor flag fails loudly.
        return cli_model
    overrides = (
        settings.llm_seat_quick_model_overrides
        if quick_mode
        else settings.llm_seat_model_overrides
    )
    override = overrides.get(spec.seat_id.value)
    if override:
        return str(override).strip()
    suffix = {
        ModelIntent.FAST: "fast",
        ModelIntent.CLASSIFIER: "fast",
        ModelIntent.REASONING: "reasoning",
        ModelIntent.PROSE: "prose" if provider == "anthropic" else "reasoning",
        ModelIntent.CRITICAL: "critical",
        ModelIntent.ESCALATION: "escalation" if provider == "openai" else "critical",
    }[intent]
    field = _PROVIDER_MODEL_FIELDS.get((provider, suffix))
    if field is None:
        raise BindingConfigurationError(
            [f"provider {provider!r} has no configured {suffix!r} model"]
        )
    return str(getattr(settings, field)).strip()


def _reasoning_override(
    settings: Any,
    schema: str,
    spec: SeatSpec,
    profile: ModelProfile,
    *,
    quick_mode: bool,
) -> str | None:
    if schema == "legacy":
        return None
    overrides = (
        settings.llm_seat_quick_reasoning_overrides
        if quick_mode
        else settings.llm_seat_reasoning_overrides
    )
    raw_value = overrides.get(spec.seat_id.value)
    if not raw_value:
        return None
    value = str(raw_value).strip().lower()
    if value not in profile.reasoning_ladder:
        supported = ", ".join(profile.reasoning_ladder) or "none"
        mode = "quick" if quick_mode else "normal"
        raise BindingConfigurationError(
            [
                f"{spec.seat_id.value} ({mode}): reasoning override {value!r} "
                f"is unsupported by {profile.prefix!r}; supported values: {supported}"
            ]
        )
    return value


def _endpoint(settings: Any, schema: str, provider: str, seat_id: SeatId) -> str | None:
    try:
        return provider_endpoint_host(settings, schema, provider, seat_id)
    except ValueError as exc:
        raise BindingConfigurationError([f"{seat_id.value}: {exc}"]) from exc


def _mode(settings: Any, schema: str, spec: SeatSpec) -> str:
    if spec.seat_id in {
        SeatId.ARTICLE_WRITER,
        SeatId.ARTICLE_WRITER_REVIEW_FALLBACK,
        SeatId.ARTICLE_WRITER_BASE_FALLBACK,
    }:
        return "auto"
    if spec.optional_mode_field is None:
        return "required"
    if schema == "legacy":
        if spec.seat_id in {SeatId.APAC, SeatId.APAC_DIRECT_RETRY}:
            return "auto" if settings.enable_apac_specialist else "off"
        return "auto" if settings.enable_consultant else "off"
    return str(getattr(settings, spec.optional_mode_field))


def _resolve_mode_binding(
    settings: Any,
    schema: str,
    spec: SeatSpec,
    provider: str,
    endpoint_host: str | None,
    endpoint_vendor: str | None,
    *,
    quick_mode: bool,
    errors: list[str],
    unknown_bindings: dict[str, set[str]],
) -> ResolvedBinding:
    """Resolve one normal/quick binding through the same validation path."""

    model = _model_for(settings, schema, spec, provider, quick_mode=quick_mode)
    profile = resolve_profile(model)
    reasoning_override = _reasoning_override(
        settings, schema, spec, profile, quick_mode=quick_mode
    )
    identity = profile.identity.at_endpoint(
        f"https://{endpoint_host}" if endpoint_host else None
    )
    if endpoint_vendor and endpoint_vendor != identity.vendor_id:
        identity = ModelIdentity(
            endpoint_vendor,
            identity.model_lineage,
            "openai_compatible",
            endpoint_host,
        )

    mode_label = f"{spec.seat_id.value} (quick)" if quick_mode else spec.seat_id.value
    if schema == "new":
        if profile.identity.vendor_id == "unknown":
            unknown_bindings.setdefault(model, set()).add(mode_label)
        else:
            if profile.identity.vendor_id != provider:
                errors.append(
                    f"{mode_label}: model {model!r} belongs to "
                    f"{profile.identity.vendor_id!r}, not {provider!r}"
                )
            missing = spec.requires - profile.capabilities
            if missing:
                errors.append(
                    f"{mode_label}: model {model!r} lacks {', '.join(sorted(missing))}"
                )
            if (
                not quick_mode
                and spec.normal_reasoning_adjustment is ReasoningAdjustment.ONE_STEP
                and Capability.REASONING_CONTROL not in profile.capabilities
            ):
                errors.append(
                    f"{mode_label}: model {model!r} cannot honor the required "
                    "one-step reasoning adjustment"
                )

    intent = spec.quick_intent if quick_mode else spec.normal_intent
    return ResolvedBinding(
        spec.seat_id,
        provider,
        model,
        identity,
        intent,
        spec.requires,
        endpoint_host,
        profile,
        reasoning_override,
    )


def resolve_binding_plan(settings: Any) -> BindingPlan:
    schema = _schema(settings)
    bindings: dict[SeatId, ResolvedBinding] = {}
    quick_bindings: dict[SeatId, ResolvedBinding] = {}
    statuses: dict[SeatId, SeatStatus] = {}
    errors: list[str] = []
    unknown_bindings: dict[str, set[str]] = {}
    rejected_provider_groups: set[tuple[str, BindingGroup]] = set()
    for seat_id, spec in SEATS.items():
        try:
            provider = provider_for_group(settings, schema, spec.binding_group)
            provider_group = (provider, spec.binding_group)
            if schema == "new" and not is_provider_qualified(*provider_group):
                if provider_group not in rejected_provider_groups:
                    errors.append(
                        f"provider {provider!r} is not application-qualified for "
                        f"binding group {spec.binding_group.value!r}; transport "
                        "capability alone is insufficient"
                    )
                    rejected_provider_groups.add(provider_group)
                continue
            endpoint_host = _endpoint(settings, schema, provider, seat_id)
            endpoint_vendor = vendor_for_endpoint_host(endpoint_host)
            if schema == "new" and endpoint_host and endpoint_vendor is None:
                errors.append(
                    f"{seat_id.value}: endpoint host {endpoint_host!r} has no reviewed provider identity"
                )
            if schema == "new" and endpoint_vendor and endpoint_vendor != provider:
                errors.append(
                    f"{seat_id.value}: endpoint host {endpoint_host!r} belongs to "
                    f"{endpoint_vendor!r}, not {provider!r}"
                )
            bindings[seat_id] = _resolve_mode_binding(
                settings,
                schema,
                spec,
                provider,
                endpoint_host,
                endpoint_vendor,
                quick_mode=False,
                errors=errors,
                unknown_bindings=unknown_bindings,
            )
            quick_bindings[seat_id] = _resolve_mode_binding(
                settings,
                schema,
                spec,
                provider,
                endpoint_host,
                endpoint_vendor,
                quick_mode=True,
                errors=errors,
                unknown_bindings=unknown_bindings,
            )
            mode = _mode(settings, schema, spec)
            ready = bool(provider_credential(settings, provider))
            if mode == "off":
                statuses[seat_id] = SeatStatus.resolve(
                    mode, provider_ready=ready, reason="configured off"
                )
            elif ready:
                statuses[seat_id] = SeatStatus.resolve(
                    mode, provider_ready=True, reason=None
                )
            elif mode == "required" and schema == "new":
                errors.append(
                    f"{seat_id.value}: missing credential for provider {provider!r}"
                )
                statuses[seat_id] = SeatStatus.resolve(
                    mode, provider_ready=False, reason="missing credential"
                )
            else:
                statuses[seat_id] = SeatStatus.resolve(
                    mode, provider_ready=False, reason="missing credential"
                )
        except BindingConfigurationError as exc:
            errors.extend(exc.errors)

    for model, affected in sorted(unknown_bindings.items()):
        errors.append(
            f"model {model!r} has no reviewed capability profile; affected "
            f"bindings: {', '.join(sorted(affected))}"
        )

    independence_seats = {
        SeatId.PORTFOLIO_MANAGER,
        SeatId.CONSULTANT,
        SeatId.APAC,
    }
    if independence_seats <= bindings.keys() and independence_seats <= statuses.keys():
        _validate_independence(bindings, statuses, settings, schema, errors)
    if errors:
        raise BindingConfigurationError(errors)
    return BindingPlan(
        MappingProxyType(bindings),
        MappingProxyType(quick_bindings),
        MappingProxyType(statuses),
        schema,
    )


def _validate_independence(
    bindings: dict[SeatId, ResolvedBinding],
    statuses: dict[SeatId, SeatStatus],
    settings: Any,
    schema: str,
    errors: list[str],
) -> None:
    if schema == "legacy":
        return
    base = bindings[SeatId.PORTFOLIO_MANAGER].identity
    review_active = any(
        statuses[seat_id].enabled
        for seat_id in (SeatId.CONSULTANT, SeatId.AUDITOR, SeatId.EDITOR)
    )
    checks = (
        (
            "review",
            bindings[SeatId.CONSULTANT].identity,
            bool(settings.llm_require_review_independence),
            settings.llm_review_independence_waiver_reason,
            review_active,
        ),
        (
            "regional",
            bindings[SeatId.APAC].identity,
            bool(settings.llm_require_regional_independence),
            settings.llm_regional_independence_waiver_reason,
            statuses[SeatId.APAC].enabled,
        ),
    )
    for label, other, required, reason, active in checks:
        reason = str(reason).strip()
        if not required and not reason:
            errors.append(
                f"{label} independence waiver reason is required when the "
                "independence requirement is false"
            )
        collapsed = (
            base.vendor_id == other.vendor_id
            or base.model_lineage == other.model_lineage
        )
        if active and required and collapsed:
            errors.append(
                f"{label} binding must differ from base in vendor and model lineage"
            )
