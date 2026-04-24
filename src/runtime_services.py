"""Runtime-scoped service and provider ownership.

This module provides the process/run-scoped container used by the CLI, worker,
and dashboard app to keep tool execution, content inspection, and provider
ownership explicit without forcing broad signature churn through the codebase.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from langchain_core.rate_limiters import BaseRateLimiter

from src.tooling.inspection_service import INSPECTION_SERVICE, InspectionService
from src.tooling.runtime import TOOL_SERVICE, ToolExecutionService, ToolHook

if TYPE_CHECKING:
    from src.data.fetcher import SmartMarketDataFetcher


@dataclass(frozen=True)
class ProviderRuntime:
    """Long-lived provider/runtime dependencies owned by a process or run."""

    fetcher: SmartMarketDataFetcher
    rate_limiter: BaseRateLimiter


@dataclass(frozen=True)
class RuntimeServices:
    """Runtime-scoped services used by tool- and prompt-bound execution."""

    tool_service: ToolExecutionService
    inspection_service: InspectionService
    providers: ProviderRuntime | None = None

    def with_tool_service(self, tool_service: ToolExecutionService) -> RuntimeServices:
        return replace(self, tool_service=tool_service)

    def with_extra_tool_hooks(self, hooks: list[ToolHook]) -> RuntimeServices:
        return replace(
            self,
            tool_service=self.tool_service.with_extra_hooks(hooks),
        )


_CURRENT_RUNTIME_SERVICES: ContextVar[RuntimeServices | None] = ContextVar(
    "current_runtime_services",
    default=None,
)


def get_current_runtime_services() -> RuntimeServices | None:
    """Return the runtime services active for the current async/thread context."""
    return _CURRENT_RUNTIME_SERVICES.get()


@contextmanager
def use_runtime_services(services: RuntimeServices) -> Iterator[RuntimeServices]:
    """Bind *services* for the current async/thread context."""
    token: Token[RuntimeServices | None] = _CURRENT_RUNTIME_SERVICES.set(services)
    try:
        yield services
    finally:
        _CURRENT_RUNTIME_SERVICES.reset(token)


def get_current_tool_service() -> ToolExecutionService:
    services = get_current_runtime_services()
    return services.tool_service if services is not None else TOOL_SERVICE


def get_current_inspection_service() -> InspectionService:
    services = get_current_runtime_services()
    return services.inspection_service if services is not None else INSPECTION_SERVICE


def get_current_provider_runtime() -> ProviderRuntime | None:
    services = get_current_runtime_services()
    return services.providers if services is not None else None


def get_current_market_data_fetcher() -> SmartMarketDataFetcher:
    providers = get_current_provider_runtime()
    if providers is not None and providers.fetcher is not None:
        return providers.fetcher

    from src.data.fetcher import get_fetcher

    return get_fetcher()


def build_provider_runtime(
    *,
    fetcher: SmartMarketDataFetcher | None = None,
    rate_limiter: BaseRateLimiter | None = None,
    explicit: bool = False,
) -> ProviderRuntime:
    """Build a provider runtime for the current process.

    ``explicit=True`` creates process-owned instances instead of reusing the
    legacy fallback singletons. Worker/web processes should use explicit mode.
    """
    if fetcher is None:
        if explicit:
            from src.data.fetcher import SmartMarketDataFetcher

            fetcher = SmartMarketDataFetcher()
        else:
            from src.data.fetcher import get_fetcher

            fetcher = get_fetcher()

    if rate_limiter is None:
        if explicit:
            from src.llms import create_process_rate_limiter

            rate_limiter = create_process_rate_limiter()
        else:
            from src.llms import GLOBAL_RATE_LIMITER

            rate_limiter = GLOBAL_RATE_LIMITER

    return ProviderRuntime(fetcher=fetcher, rate_limiter=rate_limiter)


def build_runtime_services_from_config(
    config,
    *,
    enable_tool_audit: bool,
    provider_runtime: ProviderRuntime | None = None,
    logger=None,
) -> RuntimeServices:
    """Build runtime services from the active config object."""
    from src.tooling.audit import LoggingToolAuditHook
    from src.tooling.inspection_hook import ContentInspectionHook
    from src.tooling.inspector import NullInspector
    from src.tooling.runtime import ToolExecutionService
    from src.tooling.tool_argument_policy import ToolArgumentPolicyHook

    inspection_service = InspectionService()
    hooks = []

    if enable_tool_audit:
        hooks.append(LoggingToolAuditHook())

    if not config.untrusted_content_inspection_enabled:
        inspection_service.configure(
            NullInspector(),
            mode="warn",
            fail_policy="fail_open",
        )
        return RuntimeServices(
            tool_service=ToolExecutionService(hooks),
            inspection_service=inspection_service,
            providers=provider_runtime or build_provider_runtime(),
        )

    mode = config.untrusted_content_inspection_mode
    fail_policy = config.untrusted_content_fail_policy
    backend_name = config.untrusted_content_backend

    if backend_name == "null" or not backend_name:
        inspector = NullInspector()
    elif backend_name == "python":
        from src.tooling.heuristic_inspector import HeuristicInspector

        inspector = HeuristicInspector()
    elif backend_name == "composite":
        from src.tooling.escalating_inspector import EscalatingInspector
        from src.tooling.heuristic_inspector import HeuristicInspector
        from src.tooling.llm_judge_inspector import LLMJudgeInspector

        inspector = EscalatingInspector(
            heuristic=HeuristicInspector(),
            judge=LLMJudgeInspector(),
        )
    else:
        raise ValueError(
            f"UNTRUSTED_CONTENT_BACKEND={backend_name!r} is not implemented. "
            "Supported: null, python, composite."
        )

    inspection_service.configure(inspector, mode=mode, fail_policy=fail_policy)
    if logger is not None:
        logger.info(
            "content_inspection_configured",
            inspector=type(inspector).__name__,
            mode=mode,
            fail_policy=fail_policy,
        )
    arg_policy_mode = "block" if mode == "block" else "warn"
    hooks.append(ToolArgumentPolicyHook(mode=arg_policy_mode))
    hooks.append(ContentInspectionHook(inspection_service))
    if logger is not None:
        logger.info(
            "content_inspection_enabled",
            mode=mode,
            fail_policy=fail_policy,
            backend=backend_name,
        )
    return RuntimeServices(
        tool_service=ToolExecutionService(hooks),
        inspection_service=inspection_service,
        providers=provider_runtime or build_provider_runtime(),
    )
