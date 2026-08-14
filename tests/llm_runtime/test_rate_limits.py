from types import SimpleNamespace

from src.config import Settings
from src.llm_runtime import rate_limits


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
