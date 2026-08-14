"""Provider selection and application policy, separate from transport support."""

from typing import Any

from src.llm_runtime.identities import sanitize_endpoint_host
from src.llm_runtime.seats import BindingGroup, SeatId

PROVIDER_GROUP_QUALIFICATIONS: dict[str, frozenset[BindingGroup]] = {
    # Google and OpenAI have offline contracts for reversible base/review use and
    # established operational, judge, and article fallback paths.
    "google": frozenset(
        {
            BindingGroup.BASE,
            BindingGroup.REVIEW,
            BindingGroup.WRITER,
            BindingGroup.OPERATIONAL,
            BindingGroup.JUDGE,
        }
    ),
    "openai": frozenset(
        {
            BindingGroup.BASE,
            BindingGroup.REVIEW,
            BindingGroup.WRITER,
            BindingGroup.OPERATIONAL,
            BindingGroup.JUDGE,
        }
    ),
    # Anthropic's API has broader mechanical capability, but this application has
    # only qualified its editorial path. Capability support is not evaluation.
    "anthropic": frozenset({BindingGroup.WRITER}),
    "deepseek": frozenset({BindingGroup.REGIONAL_REVIEW}),
    "zai": frozenset({BindingGroup.REGIONAL_REVIEW}),
    "moonshot": frozenset({BindingGroup.REVIEW}),
}


def is_provider_qualified(provider: str, group: BindingGroup) -> bool:
    """Return whether repository evidence qualifies *provider* for *group*."""

    return group in PROVIDER_GROUP_QUALIFICATIONS.get(provider, frozenset())


_LEGACY_GROUP_PROVIDERS = {
    BindingGroup.BASE: "google",
    BindingGroup.REVIEW: "openai",
    BindingGroup.REGIONAL_REVIEW: "deepseek",
    BindingGroup.WRITER: "anthropic",
    BindingGroup.OPERATIONAL: "google",
    BindingGroup.JUDGE: "google",
}

_GROUP_SELECTOR_FIELDS = {
    BindingGroup.BASE: "llm_base_provider",
    BindingGroup.REVIEW: "llm_review_provider",
    BindingGroup.REGIONAL_REVIEW: "llm_regional_provider",
    BindingGroup.WRITER: "llm_writer_provider",
    BindingGroup.OPERATIONAL: "llm_operational_provider",
    BindingGroup.JUDGE: "llm_judge_provider",
}


def provider_for_group(settings: Any, schema: str, group: BindingGroup) -> str:
    """Resolve a binding group's provider without inspecting model names."""

    if schema == "legacy":
        return _LEGACY_GROUP_PROVIDERS[group]
    value = getattr(settings, _GROUP_SELECTOR_FIELDS[group])
    return str(value).strip().lower() if value else _LEGACY_GROUP_PROVIDERS[group]


def provider_credential(settings: Any, provider: str) -> str:
    """Return the provider credential as plain text for construction checks."""

    field = {
        "google": "google_api_key",
        "openai": "openai_api_key",
        "anthropic": "claude_api_key",
        "moonshot": "moonshot_api_key",
        "deepseek": "deepseek_api_key",
        "zai": "zai_api_key",
    }.get(provider)
    if field is None:
        return ""
    value = getattr(settings, field)
    return value.get_secret_value() if value is not None else ""


def provider_endpoint_host(
    settings: Any, schema: str, provider: str, seat_id: SeatId
) -> str | None:
    """Resolve and sanitize a binding endpoint without exposing credentials."""

    setting_name: str | None = None
    raw_url: str | None = None
    if schema == "legacy" and seat_id in {SeatId.APAC, SeatId.APAC_DIRECT_RETRY}:
        setting_name = "APAC_SPECIALIST_BASE_URL"
        raw_url = settings.apac_specialist_base_url
    elif provider == "deepseek":
        setting_name, raw_url = "DEEPSEEK_API_BASE", settings.deepseek_api_base
    elif provider == "zai":
        setting_name, raw_url = "ZAI_API_BASE", settings.zai_api_base
    elif provider == "moonshot":
        setting_name, raw_url = "MOONSHOT_API_BASE", settings.moonshot_api_base
    elif provider == "openai":
        setting_name = "OPENAI_API_BASE"
        raw_url = settings.openai_api_base or None
    try:
        return sanitize_endpoint_host(raw_url)
    except ValueError as exc:
        raise ValueError(f"{setting_name}: {exc}") from exc
