"""Provider selection and application policy, separate from transport support."""

import threading
import uuid
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
    # xAI Grok 4.6 documents tool calling, structured output, and a four-step
    # reasoning ladder, which covers every review seat's declared requirement.
    # Review-only: the compatible transport serves no base seat, and live
    # qualification evidence is still pending (see docs/LLM_PROVIDERS.md).
    "xai": frozenset({BindingGroup.REVIEW}),
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
        "xai": "xai_api_key",
    }.get(provider)
    if field is None:
        return ""
    value = getattr(settings, field)
    # Stripped so a whitespace-only or stray-newline ``.env`` value reads as
    # absent here rather than passing the readiness check and failing later as
    # an opaque 401 from the vendor.
    return value.get_secret_value().strip() if value is not None else ""


# Providers reached only through the OpenAI-compatible transport. Unlike
# ``openai`` — where an empty base URL correctly means "use the vendor's own
# default" — a blank value here resolves to ``base_url=None``, which the OpenAI
# SDK fills in with ``api.openai.com``. That would quietly point one vendor's
# credential at another vendor's endpoint, so the URL is mandatory.
_ENDPOINT_REQUIRED_PROVIDERS = frozenset({"deepseek", "zai", "moonshot", "xai"})


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
    elif provider == "xai":
        setting_name, raw_url = "XAI_API_BASE", settings.xai_api_base
    elif provider == "openai":
        setting_name = "OPENAI_API_BASE"
        raw_url = settings.openai_api_base or None
    if provider in _ENDPOINT_REQUIRED_PROVIDERS and not str(raw_url or "").strip():
        raise ValueError(
            f"{setting_name}: a base URL is required for provider {provider!r}; "
            "an empty value makes the OpenAI-compatible client fall back to "
            "OpenAI's default endpoint, sending this vendor's key elsewhere"
        )
    try:
        return sanitize_endpoint_host(raw_url)
    except ValueError as exc:
        raise ValueError(f"{setting_name}: {exc}") from exc


_cache_affinity_lock = threading.Lock()
_cache_affinity_id: str | None = None


def cache_affinity_id() -> str:
    """Return a stable per-process token routing one run's calls to one server.

    Deliberately *not* the persisted ``run_summary.run_id``: that is derived at
    persistence time from an optional Langfuse trace, long after seat models are
    constructed at graph-build time. This is a cache-locality hint only and must
    never be read as a correlation key. Per-process is the correct granularity
    because every entry point (``run_tickers.sh``, the pipeline, the CLI) spawns
    one process per ticker, so per-process is per-run without any plumbing.
    """

    global _cache_affinity_id
    with _cache_affinity_lock:
        if _cache_affinity_id is None:
            _cache_affinity_id = uuid.uuid4().hex
        return _cache_affinity_id


def provider_default_headers(provider: str) -> dict[str, str]:
    """Return a provider's cache-affinity hints; empty for those wanting none.

    Which vendor wants which hint is a policy fact about the provider, not a
    property of the transport, so it belongs beside the credential and endpoint
    resolvers rather than as a vendor conditional inside an adapter.
    """

    if provider == "xai":
        # xAI documents that without this header related requests land on
        # cache-cold servers and bill the full input rate rather than the
        # discounted cached-input rate.
        return {"x-grok-conv-id": cache_affinity_id()}
    return {}


def _reset_cache_affinity_for_tests() -> None:
    """Clear the process-global affinity token so tests stay deterministic."""

    global _cache_affinity_id
    with _cache_affinity_lock:
        _cache_affinity_id = None
