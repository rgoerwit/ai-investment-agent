import time
from types import SimpleNamespace

import langchain_core.rate_limiters as langchain_rate_limiters
import pytest

from src.config import Settings
from src.llm_runtime import rate_limits

# This module tests the real limiter's construction and pacing.
pytestmark = pytest.mark.real_rate_limiter


def test_all_shipped_providers_have_safe_default_rpm_ceilings() -> None:
    settings = Settings(_env_file=None)
    assert settings.google_rpm_limit == 15
    assert settings.openai_rpm_limit == 120
    assert settings.anthropic_rpm_limit == 60
    assert settings.deepseek_rpm_limit == 30
    assert settings.zai_rpm_limit == 30
    assert settings.moonshot_rpm_limit == 60


def test_direct_construction_reuses_provider_specific_process_fallback(
    monkeypatch,
) -> None:
    created = []

    def fake_create(rpm: int):
        limiter = SimpleNamespace(rpm=rpm)
        created.append(limiter)
        return limiter

    rate_limits.reset_fallback_limiters_for_tests()
    monkeypatch.setattr(rate_limits, "create_process_rate_limiter", fake_create)
    settings = Settings(_env_file=None)

    first = rate_limits.limiter_for_binding(settings, "openai", None)
    second = rate_limits.limiter_for_binding(settings, "openai", None)
    anthropic = rate_limits.limiter_for_binding(settings, "anthropic", None)

    assert first is second
    assert first.rpm == 120
    assert anthropic.rpm == 60
    assert len(created) == 2


# --- The Google ceiling follows the schema, in one place -----------------------


@pytest.mark.parametrize(
    ("base_provider", "field", "rpm"),
    [(None, "gemini_rpm_limit", 33), ("google", "google_rpm_limit", 44)],
    ids=["legacy_schema", "multi_provider_schema"],
)
def test_configured_google_rpm_follows_the_schema(base_provider, field, rpm) -> None:
    settings = SimpleNamespace(
        llm_base_provider=base_provider, gemini_rpm_limit=33, google_rpm_limit=44
    )
    assert rate_limits.google_rpm_field(settings) == field
    assert rate_limits.configured_google_rpm(settings) == rpm
    assert rate_limits.provider_rpm(settings, "google") == rpm


def test_settings_without_the_schema_selector_read_the_legacy_gemini_ceiling() -> None:
    settings = Settings(_env_file=None, gemini_rpm_limit=12, google_rpm_limit=99)
    assert settings.llm_base_provider is None
    assert rate_limits.configured_google_rpm(settings) == 12


@pytest.mark.parametrize(
    ("vendor", "expected"),
    [("openai", 120), ("xai", 7), ("deepseek", None), ("unknown-vendor", None)],
)
def test_provider_rpm(vendor, expected) -> None:
    settings = SimpleNamespace(openai_rpm_limit=120, xai_rpm_limit=7)
    assert rate_limits.provider_rpm(settings, vendor) == expected


def test_unscoped_google_fallback_reads_the_legacy_field(monkeypatch) -> None:
    created: list[int] = []
    monkeypatch.setattr(
        rate_limits,
        "create_process_rate_limiter",
        lambda rpm: created.append(rpm) or SimpleNamespace(rpm=rpm),
    )
    rate_limits.reset_fallback_limiters_for_tests()
    settings = SimpleNamespace(
        llm_base_provider=None, gemini_rpm_limit=33, google_rpm_limit=44
    )
    try:
        assert rate_limits.limiter_for_binding(settings, "google", None).rpm == 33
    finally:
        rate_limits.reset_fallback_limiters_for_tests()
    assert created == [33]


# --- A fresh limiter sends its first request at once ---------------------------


@pytest.fixture
def limiter_clock(monkeypatch) -> list[float]:
    """Drive ``InMemoryRateLimiter``'s token clock by hand."""
    clock = [100.0]
    monkeypatch.setattr(
        langchain_rate_limiters,
        "time",
        SimpleNamespace(monotonic=lambda: clock[0], sleep=time.sleep),
    )
    return clock


def _acquired_at(limiter, clock: list[float], instants: list[float]) -> list[bool]:
    sent = []
    for instant in instants:
        clock[0] = 100.0 + instant
        sent.append(limiter.acquire(blocking=False))
    return sent


def test_the_first_request_needs_no_wait_and_the_pace_is_unchanged(
    limiter_clock,
) -> None:
    # 15 RPM x 0.8 = one token every 5 s.
    limiter = rate_limits.create_process_rate_limiter(15)
    instants = [0.0, 0.0, 4.99, 5.0, 9.99, 10.0]

    assert _acquired_at(limiter, limiter_clock, instants) == [
        True,  # the one starting token
        False,  # no burst beyond it
        False,
        True,  # then one token per interval, as before
        False,
        True,
    ]


def test_the_library_default_starts_empty(limiter_clock) -> None:
    # The control: the same bucket as constructed by langchain waits a full
    # interval before its first request.
    control = langchain_rate_limiters.InMemoryRateLimiter(
        requests_per_second=0.2, check_every_n_seconds=0.1, max_bucket_size=5
    )
    assert _acquired_at(control, limiter_clock, [0.0, 4.99, 5.0]) == [
        False,
        False,
        True,
    ]


@pytest.mark.asyncio
async def test_the_first_async_acquire_does_not_sleep(limiter_clock) -> None:
    limiter = rate_limits.create_process_rate_limiter(15)
    start = time.monotonic()
    assert await limiter.aacquire(blocking=True) is True
    assert time.monotonic() - start < 0.05
