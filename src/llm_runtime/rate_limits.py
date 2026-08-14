"""Provider-neutral application rate-limit construction and fallback ownership."""

from threading import Lock
from typing import Any

import structlog
from langchain_core.rate_limiters import BaseRateLimiter, InMemoryRateLimiter

logger = structlog.get_logger(__name__)

_PROVIDER_RPM_FIELDS = {
    "google": "google_rpm_limit",
    "openai": "openai_rpm_limit",
    "anthropic": "anthropic_rpm_limit",
    "deepseek": "deepseek_rpm_limit",
    "zai": "zai_rpm_limit",
    "moonshot": "moonshot_rpm_limit",
}
_FALLBACK_LIMITERS: dict[tuple[str, str | None, int], BaseRateLimiter] = {}
_FALLBACK_LIMITERS_LOCK = Lock()


def create_process_rate_limiter(rpm: int) -> InMemoryRateLimiter:
    """Create one independently owned limiter from a provider RPM ceiling."""

    safety_factor = 0.8
    requests_per_second = (rpm / 60.0) * safety_factor
    max_bucket_size = max(5, int(rpm * 0.1))
    logger.info(
        "rate_limiter_configured",
        rpm=rpm,
        rps=round(requests_per_second, 2),
        max_bucket=max_bucket_size,
    )
    return InMemoryRateLimiter(
        requests_per_second=requests_per_second,
        check_every_n_seconds=0.1,
        max_bucket_size=max_bucket_size,
    )


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

    field_name = _PROVIDER_RPM_FIELDS.get(vendor_id)
    rpm = getattr(settings, field_name, None) if field_name else None
    if rpm is None:
        return None
    key = (vendor_id, endpoint_host, int(rpm))
    with _FALLBACK_LIMITERS_LOCK:
        limiter = _FALLBACK_LIMITERS.get(key)
        if limiter is None:
            limiter = create_process_rate_limiter(int(rpm))
            _FALLBACK_LIMITERS[key] = limiter
        return limiter


def reset_fallback_limiters_for_tests() -> None:
    """Clear process fallbacks so tests can assert construction deterministically."""

    with _FALLBACK_LIMITERS_LOCK:
        _FALLBACK_LIMITERS.clear()
