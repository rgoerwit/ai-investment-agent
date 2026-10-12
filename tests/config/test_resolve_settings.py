"""``resolve_settings``: the one "explicit settings or the live singleton" reader.

Four modules each kept a private copy (``timeouts._settings``,
``service_tiers._cfg``, ``llms._settings_or_default``, ``run_deadline._settings``);
one of them read a name bound at import, which is why tests patched two places.
"""

from __future__ import annotations

import pytest

import src.config as config_module
from src import llms, service_tiers
from src.config import Settings, resolve_settings
from src.llm_runtime import run_deadline, timeouts
from src.llm_runtime.timeouts import TimeoutClass


def _other(**update: object) -> Settings:
    return Settings(_env_file=None, **update)


def test_an_explicit_settings_object_wins_over_the_singleton(monkeypatch):
    explicit = _other(api_timeout=111)
    monkeypatch.setattr(config_module, "config", _other(api_timeout=222))
    assert resolve_settings(explicit) is explicit


def test_none_reads_the_live_singleton():
    assert resolve_settings() is config_module.config
    assert resolve_settings(None) is config_module.config


def test_a_patched_singleton_is_seen_without_reimporting(monkeypatch):
    patched = _other(api_timeout=222)
    monkeypatch.setattr(config_module, "config", patched)
    assert resolve_settings() is patched


@pytest.mark.parametrize(
    "module, name",
    [
        (timeouts, "_settings"),
        (service_tiers, "_cfg"),
        (llms, "_settings_or_default"),
        (run_deadline, "_settings"),
    ],
)
def test_the_private_copies_are_gone(module, name):
    assert not hasattr(module, name)


def test_the_policy_resolver_sees_a_patched_singleton(monkeypatch):
    monkeypatch.setattr(
        config_module,
        "config",
        _other(quick_llm_call_hard_timeout_seconds=17.0),
    )
    assert timeouts.quick_invocation_cap(TimeoutClass.STANDARD) == 17.0


def test_an_explicit_cfg_beats_the_patched_singleton_in_the_policy_resolver(
    monkeypatch,
):
    monkeypatch.setattr(
        config_module,
        "config",
        _other(quick_llm_call_hard_timeout_seconds=17.0),
    )
    explicit = _other(quick_llm_call_hard_timeout_seconds=23.0)
    assert timeouts.quick_invocation_cap(TimeoutClass.STANDARD, explicit) == 23.0


def test_service_tiers_reads_the_patched_singleton(monkeypatch):
    monkeypatch.setattr(config_module, "config", _other(gemini_service_tier="flex"))
    assert service_tiers.gemini_flex_active() is True
    assert service_tiers.gemini_flex_active(_other()) is False


def test_run_deadline_reserve_follows_the_patched_singleton(monkeypatch):
    monkeypatch.setattr(config_module, "_cached_env_file_values", lambda: {})
    monkeypatch.setenv(run_deadline.RUN_DEADLINE_EPOCH_ENV, "1000600")

    def deadline(cfg=None):
        return run_deadline.run_deadline(
            cfg, wall_clock=lambda: 1_000_000.0, clock=lambda: 50.0
        )

    monkeypatch.setattr(
        config_module, "config", _other(publication_reserve_seconds=41.0)
    )
    assert deadline() == pytest.approx(50.0 + 600 - 41.0)
    assert deadline(_other(publication_reserve_seconds=5.0)) == pytest.approx(
        50.0 + 600 - 5.0
    )


def test_llms_defaults_follow_the_patched_singleton(monkeypatch):
    monkeypatch.setattr(config_module, "config", _other(llm_base_output_tokens=4096))
    assert llms.resolve_settings().llm_base_output_tokens == 4096
