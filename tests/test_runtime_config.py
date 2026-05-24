from __future__ import annotations

import asyncio
from types import SimpleNamespace

from src.runtime_config import (
    RuntimeConfig,
    bind_runtime_config,
    build_runtime_config,
    get_runtime_config,
    quick_runtime_clamp_changes,
    use_runtime_config,
)


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        quick_think_llm="quick-old",
        deep_think_llm="deep-old",
        enable_memory=True,
        langfuse_enabled=False,
        api_retry_attempts=5,
        gemini_rpm_limit=1000,
        llm_call_hard_timeout_seconds=600.0,
        quick_mode_active=False,
    )


def _args(**overrides) -> SimpleNamespace:
    values = {
        "quick": False,
        "quick_model": None,
        "deep_model": None,
        "no_memory": False,
        "enable_langfuse": False,
        "trace_langfuse": False,
        "ticker": "TEST",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_build_runtime_config_applies_cli_overrides_without_mutating_base() -> None:
    base = _config()

    runtime_config = build_runtime_config(
        _args(
            quick=True,
            quick_model="quick-new",
            deep_model="deep-new",
            no_memory=True,
            trace_langfuse=True,
        ),
        base,
    )

    assert runtime_config == RuntimeConfig(
        quick_think_llm="quick-new",
        deep_think_llm="deep-new",
        enable_memory=False,
        langfuse_enabled=True,
        api_retry_attempts=2,
        gemini_rpm_limit=360,
        llm_call_hard_timeout_seconds=120.0,
        quick_mode_active=True,
    )
    assert base.quick_think_llm == "quick-old"
    assert base.deep_think_llm == "deep-old"
    assert base.enable_memory is True
    assert base.langfuse_enabled is False
    assert base.api_retry_attempts == 5
    assert base.gemini_rpm_limit == 1000
    assert base.llm_call_hard_timeout_seconds == 600.0
    assert base.quick_mode_active is False


def test_quick_mode_does_not_raise_tighter_existing_values() -> None:
    base = _config()
    base.api_retry_attempts = 1
    base.gemini_rpm_limit = 60
    base.llm_call_hard_timeout_seconds = 45.0

    runtime_config = build_runtime_config(_args(quick=True), base)

    assert runtime_config.api_retry_attempts == 1
    assert runtime_config.gemini_rpm_limit == 60
    assert runtime_config.llm_call_hard_timeout_seconds == 45.0
    assert runtime_config.quick_mode_active is True


def test_no_quick_mode_does_not_clamp() -> None:
    base = _config()

    runtime_config = build_runtime_config(_args(quick=False), base)

    assert runtime_config.api_retry_attempts == 5
    assert runtime_config.gemini_rpm_limit == 1000
    assert runtime_config.llm_call_hard_timeout_seconds == 600.0
    assert runtime_config.quick_mode_active is False


def test_clamp_change_summary_only_reports_lowered_values() -> None:
    base = _config()
    runtime_config = build_runtime_config(_args(quick=True), base)

    assert quick_runtime_clamp_changes(_args(quick=True), base, runtime_config) == {
        "api_retry_attempts": {"from": 5, "to": 2},
        "gemini_rpm_limit": {"from": 1000, "to": 360},
        "llm_call_hard_timeout_seconds": {"from": 600.0, "to": 120.0},
    }


def test_unbound_runtime_config_falls_back_to_base_config() -> None:
    base = _config()

    assert get_runtime_config(base) == RuntimeConfig.from_config(base)


def test_runtime_config_binding_restore_is_idempotent() -> None:
    base = _config()
    override = RuntimeConfig.from_config(base).with_overrides(enable_memory=False)
    restore = bind_runtime_config(override)

    assert get_runtime_config(base).enable_memory is False

    restore()
    restore()

    assert get_runtime_config(base).enable_memory is True


def test_runtime_config_nested_bindings_restore_outer_value() -> None:
    base = _config()
    outer = RuntimeConfig.from_config(base).with_overrides(enable_memory=False)
    inner = RuntimeConfig.from_config(base).with_overrides(langfuse_enabled=True)

    with use_runtime_config(outer):
        assert get_runtime_config(base).enable_memory is False
        with use_runtime_config(inner):
            assert get_runtime_config(base).langfuse_enabled is True
            assert get_runtime_config(base).enable_memory is True
        assert get_runtime_config(base).enable_memory is False

    assert get_runtime_config(base) == RuntimeConfig.from_config(base)


def test_runtime_config_bindings_are_task_local() -> None:
    base = _config()
    first = RuntimeConfig.from_config(base).with_overrides(quick_think_llm="first")
    second = RuntimeConfig.from_config(base).with_overrides(quick_think_llm="second")

    async def read_bound_model(runtime_config: RuntimeConfig) -> str:
        with use_runtime_config(runtime_config):
            await asyncio.sleep(0)
            return get_runtime_config(base).quick_think_llm

    async def run_both() -> list[str]:
        return await asyncio.gather(
            read_bound_model(first),
            read_bound_model(second),
        )

    assert asyncio.run(run_both()) == ["first", "second"]
