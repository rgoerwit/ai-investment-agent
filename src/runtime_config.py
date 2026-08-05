from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RuntimeConfig:
    quick_think_llm: str
    deep_think_llm: str
    enable_memory: bool
    langfuse_enabled: bool
    api_retry_attempts: int
    gemini_rpm_limit: int
    llm_call_hard_timeout_seconds: float
    quick_mode_active: bool
    images_dir: Path
    quiet_mode: bool

    @classmethod
    def from_config(cls, base_config: Any) -> RuntimeConfig:
        return cls(
            quick_think_llm=base_config.quick_think_llm,
            deep_think_llm=base_config.deep_think_llm,
            enable_memory=base_config.enable_memory,
            langfuse_enabled=base_config.langfuse_enabled,
            api_retry_attempts=base_config.api_retry_attempts,
            gemini_rpm_limit=base_config.gemini_rpm_limit,
            llm_call_hard_timeout_seconds=base_config.llm_call_hard_timeout_seconds,
            quick_mode_active=getattr(base_config, "quick_mode_active", False),
            images_dir=Path(base_config.images_dir),
            quiet_mode=getattr(base_config, "quiet_mode", False),
        )

    def with_overrides(self, **changes: Any) -> RuntimeConfig:
        return replace(self, **changes)


_CURRENT_RUNTIME_CONFIG: ContextVar[RuntimeConfig | None] = ContextVar(
    "current_runtime_config",
    default=None,
)


def build_runtime_config(args: Any, base_config: Any) -> RuntimeConfig:
    runtime_config = RuntimeConfig.from_config(base_config)
    if getattr(args, "quick", False):
        runtime_config = runtime_config.with_overrides(
            quick_mode_active=True,
            api_retry_attempts=min(runtime_config.api_retry_attempts, 2),
            gemini_rpm_limit=min(runtime_config.gemini_rpm_limit, 360),
            llm_call_hard_timeout_seconds=min(
                runtime_config.llm_call_hard_timeout_seconds,
                120.0,
            ),
        )
    if getattr(args, "quick_model", None):
        runtime_config = runtime_config.with_overrides(quick_think_llm=args.quick_model)
    if getattr(args, "deep_model", None):
        runtime_config = runtime_config.with_overrides(deep_think_llm=args.deep_model)
    if getattr(args, "no_memory", False):
        runtime_config = runtime_config.with_overrides(enable_memory=False)
    if getattr(args, "quiet", False) or getattr(args, "brief", False):
        runtime_config = runtime_config.with_overrides(quiet_mode=True)
    if getattr(args, "enable_langfuse", False) or getattr(
        args, "trace_langfuse", False
    ):
        runtime_config = runtime_config.with_overrides(langfuse_enabled=True)
    return runtime_config


def quick_runtime_clamp_changes(
    args: Any, base_config: Any, runtime_config: RuntimeConfig
) -> dict[str, dict[str, Any]]:
    if not getattr(args, "quick", False):
        return {}
    changes: dict[str, dict[str, Any]] = {}
    for field in (
        "api_retry_attempts",
        "gemini_rpm_limit",
        "llm_call_hard_timeout_seconds",
    ):
        before = getattr(base_config, field)
        after = getattr(runtime_config, field)
        if after < before:
            changes[field] = {"from": before, "to": after}
    return changes


def get_runtime_config(base_config: Any) -> RuntimeConfig:
    return _CURRENT_RUNTIME_CONFIG.get() or RuntimeConfig.from_config(base_config)


def bind_runtime_config(runtime_config: RuntimeConfig) -> Callable[[], None]:
    token: Token[RuntimeConfig | None] = _CURRENT_RUNTIME_CONFIG.set(runtime_config)
    restored = False

    def restore() -> None:
        nonlocal restored
        if restored:
            return
        _CURRENT_RUNTIME_CONFIG.reset(token)
        restored = True

    return restore


@contextmanager
def use_runtime_config(runtime_config: RuntimeConfig) -> Iterator[RuntimeConfig]:
    restore = bind_runtime_config(runtime_config)
    try:
        yield runtime_config
    finally:
        restore()
