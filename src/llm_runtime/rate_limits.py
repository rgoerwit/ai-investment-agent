"""Provider-neutral application rate-limit construction and fallback ownership."""

from threading import Lock
from typing import Any

import structlog
from langchain_core.rate_limiters import BaseRateLimiter, InMemoryRateLimiter

logger = structlog.get_logger(__name__)

# Google is absent: its field depends on the schema (``google_rpm_field``).
_PROVIDER_RPM_FIELDS = {
    "openai": "openai_rpm_limit",
    "anthropic": "anthropic_rpm_limit",
    "deepseek": "deepseek_rpm_limit",
    "zai": "zai_rpm_limit",
    "moonshot": "moonshot_rpm_limit",
    "xai": "xai_rpm_limit",
}
# Every vendor with an application RPM ceiling other than Google.
NON_GOOGLE_RATE_LIMITED_VENDORS = tuple(_PROVIDER_RPM_FIELDS)
_FALLBACK_LIMITERS: dict[tuple[str, str | None, int], BaseRateLimiter] = {}
_FALLBACK_LIMITERS_LOCK = Lock()


def google_rpm_field(settings: Any) -> str:
    """The Settings field holding the Google RPM ceiling under this schema.

    The multi-provider schema (``LLM_BASE_PROVIDER`` set) reads
    ``GOOGLE_RPM_LIMIT``; the legacy schema keeps ``GEMINI_RPM_LIMIT``.
    """
    if settings.llm_base_provider is not None:
        return "google_rpm_limit"
    return "gemini_rpm_limit"


def configured_google_rpm(settings: Any) -> int:
    """The configured Google RPM ceiling, before any run-mode clamp."""
    return int(getattr(settings, google_rpm_field(settings)))


def provider_rpm(settings: Any, vendor_id: str) -> int | None:
    """The configured RPM ceiling for ``vendor_id``, or None when it has none."""
    if vendor_id == "google":
        return configured_google_rpm(settings)
    field_name = _PROVIDER_RPM_FIELDS.get(vendor_id)
    rpm = getattr(settings, field_name, None) if field_name else None
    return None if rpm is None else int(rpm)


def create_process_rate_limiter(rpm: int) -> InMemoryRateLimiter:
    """Create one independently owned limiter from a provider RPM ceiling.

    The bucket starts holding one token, so a process's first request is sent
    at once. ``InMemoryRateLimiter`` starts empty and refills only after a
    whole ``1 / rps`` interval, so every process otherwise waited that long
    (about 5 s at 15 RPM) before its first request. Later requests are paced
    exactly as before: the refill clock starts at the first acquisition.
    """

    safety_factor = 0.8
    requests_per_second = (rpm / 60.0) * safety_factor
    max_bucket_size = max(5, int(rpm * 0.1))
    logger.info(
        "rate_limiter_configured",
        rpm=rpm,
        rps=round(requests_per_second, 2),
        max_bucket=max_bucket_size,
    )
    limiter = InMemoryRateLimiter(
        requests_per_second=requests_per_second,
        check_every_n_seconds=0.1,
        max_bucket_size=max_bucket_size,
    )
    limiter.available_tokens = 1.0
    return limiter


def limiter_for_binding(
    settings: Any,
    vendor_id: str,
    endpoint_host: str | None,
) -> BaseRateLimiter | None:
    """Return the scoped limiter, or a safe process fallback for direct callers."""

    from src.runtime_services import get_current_provider_runtime

    runtime = get_current_provider_runtime()
    if runtime is not None:
        return runtime.limiter_for(vendor_id, endpoint_host)

    rpm = provider_rpm(settings, vendor_id)
    if rpm is None:
        return None
    key = (vendor_id, endpoint_host, rpm)
    with _FALLBACK_LIMITERS_LOCK:
        limiter = _FALLBACK_LIMITERS.get(key)
        if limiter is None:
            limiter = create_process_rate_limiter(rpm)
            _FALLBACK_LIMITERS[key] = limiter
        return limiter


def reset_fallback_limiters_for_tests() -> None:
    """Clear process fallbacks so tests can assert construction deterministically."""

    with _FALLBACK_LIMITERS_LOCK:
        _FALLBACK_LIMITERS.clear()
