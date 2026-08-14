"""Runtime-scoped service and provider ownership.

This module provides the process/run-scoped container used by the CLI, worker,
and dashboard app to keep tool execution, content inspection, and provider
ownership explicit without forcing broad signature churn through the codebase.
"""

from __future__ import annotations

import ipaddress
import re
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal
from urllib.parse import urlparse

from langchain_core.rate_limiters import BaseRateLimiter

from src.error_safety import redact_sensitive_text
from src.llm_runtime.rate_limits import create_process_rate_limiter
from src.tooling.inspection_service import INSPECTION_SERVICE, InspectionService
from src.tooling.inspector import ContentInspector
from src.tooling.runtime import TOOL_SERVICE, ToolExecutionService, ToolHook

if TYPE_CHECKING:
    from src.data.fetcher import SmartMarketDataFetcher
    from src.mcp.client import MCPRuntime
    from src.tooling.evidence_recorder import EvidenceRecord, EvidenceRecorder


@dataclass(frozen=True)
class ProviderRuntimeKey:
    """Throttle/failure identity; endpoint paths and credentials never participate."""

    vendor_id: str
    endpoint_host: str | None = None


@dataclass(frozen=True)
class ProviderRuntime:
    """Long-lived provider/runtime dependencies owned by a process or run.

    ``rate_limiter`` remains as the legacy Google/default limiter until all
    construction paths request a named binding. Named entries isolate vendors
    and endpoints without changing existing call behavior.
    """

    fetcher: SmartMarketDataFetcher
    rate_limiter: BaseRateLimiter
    rate_limiters: dict[ProviderRuntimeKey, BaseRateLimiter] = field(
        default_factory=dict
    )

    def limiter_for(
        self, vendor_id: str, endpoint_host: str | None = None
    ) -> BaseRateLimiter | None:
        key = ProviderRuntimeKey(vendor_id, endpoint_host)
        if key in self.rate_limiters:
            return self.rate_limiters[key]
        vendor_default = ProviderRuntimeKey(vendor_id, None)
        if vendor_default in self.rate_limiters:
            return self.rate_limiters[vendor_default]
        return self.rate_limiter if vendor_id == "google" else None


@dataclass
class IssuerAuthorityRegistry:
    """Run-scoped issuer hosts bound from structured company-profile identity."""

    _provenance_by_host: dict[str, str] = field(default_factory=dict)

    def register_url(self, url: str, *, provenance: str) -> bool:
        parsed = urlparse(url)
        if parsed.scheme != "https" or not parsed.hostname:
            return False
        try:
            port = parsed.port
        except ValueError:
            return False
        if parsed.username or parsed.password or port not in {None, 443}:
            return False
        host = parsed.hostname.rstrip(".").lower()
        if host.startswith("www."):
            host = host[4:]
        if (
            "." not in host
            or len(host) > 253
            or not re.fullmatch(r"[a-z0-9](?:[a-z0-9.-]{0,251}[a-z0-9])?", host)
        ):
            return False
        try:
            ipaddress.ip_address(host)
        except ValueError:
            pass
        else:
            return False
        self._provenance_by_host.setdefault(host, provenance)
        return True

    def hosts(self) -> tuple[str, ...]:
        return tuple(self._provenance_by_host)


@dataclass(frozen=True)
class RuntimeServices:
    """Runtime-scoped services used by tool- and prompt-bound execution."""

    tool_service: ToolExecutionService
    inspection_service: InspectionService
    providers: ProviderRuntime | None = None
    mcp_runtime: MCPRuntime | None = None
    evidence_recorder: EvidenceRecorder | None = None
    issuer_authority: IssuerAuthorityRegistry = field(
        default_factory=IssuerAuthorityRegistry
    )

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


def get_current_evidence_records(
    *, agent_key: str | None = None
) -> list[EvidenceRecord]:
    services = get_current_runtime_services()
    if services is None or services.evidence_recorder is None:
        return []
    return services.evidence_recorder.snapshot(agent_key=agent_key)


def register_current_issuer_url(url: str, *, provenance: str) -> bool:
    services = get_current_runtime_services()
    if services is None:
        return False
    return services.issuer_authority.register_url(url, provenance=provenance)


def get_current_issuer_hosts() -> tuple[str, ...]:
    services = get_current_runtime_services()
    return services.issuer_authority.hosts() if services is not None else ()


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
    rate_limiters: dict[ProviderRuntimeKey, BaseRateLimiter] | None = None,
    settings=None,
    google_rpm_override: int | None = None,
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

    if settings is None:
        from src.config import config as settings

    if rate_limiter is None:
        if explicit:
            configured_google_rpm = (
                settings.google_rpm_limit
                if settings.llm_base_provider is not None
                else settings.gemini_rpm_limit
            )
            rate_limiter = create_process_rate_limiter(
                rpm=google_rpm_override or configured_google_rpm
            )
        else:
            from src.llms import GLOBAL_RATE_LIMITER

            rate_limiter = GLOBAL_RATE_LIMITER

    resolved_limiters = dict(rate_limiters or {})
    resolved_limiters.setdefault(ProviderRuntimeKey("google"), rate_limiter)
    for vendor_id, field_name in (
        ("openai", "openai_rpm_limit"),
        ("anthropic", "anthropic_rpm_limit"),
        ("deepseek", "deepseek_rpm_limit"),
        ("zai", "zai_rpm_limit"),
        ("moonshot", "moonshot_rpm_limit"),
    ):
        rpm = getattr(settings, field_name, None)
        key = ProviderRuntimeKey(vendor_id)
        if rpm is not None and key not in resolved_limiters:
            resolved_limiters[key] = create_process_rate_limiter(rpm=int(rpm))

    return ProviderRuntime(
        fetcher=fetcher,
        rate_limiter=rate_limiter,
        rate_limiters=resolved_limiters,
    )


def validate_llm_bindings(settings) -> None:
    """Fail fast on an unusable LLM binding configuration.

    Raises ``BindingConfigurationError`` listing every problem at once. Non-
    ``Settings`` objects (test doubles, legacy config protocols) are skipped.
    """
    from src.config import Settings

    if not isinstance(settings, Settings):
        return
    from src.llm_runtime.bindings import resolve_binding_plan

    resolve_binding_plan(settings)


def build_runtime_services_from_config(
    config,
    *,
    enable_tool_audit: bool,
    provider_runtime: ProviderRuntime | None = None,
    logger=None,
) -> RuntimeServices:
    """Build runtime services from the active config object.

    Resolving the binding plan here makes an unusable LLM configuration a
    *startup* failure for every process that builds services (analyzer, dashboard,
    worker, smoke scripts) rather than a surprise at first model construction.
    ``resolve_binding_plan`` is pure and cheap; the plan is intentionally not
    cached, because ``Settings`` is mutable and a stale plan is worse than a
    re-resolve. Test doubles that are not real ``Settings`` are skipped.
    """
    from src.tooling.audit import LoggingToolAuditHook
    from src.tooling.evidence_recorder import EvidenceRecorder
    from src.tooling.inspection_hook import ContentInspectionHook
    from src.tooling.inspector import NullInspector
    from src.tooling.runtime import ToolExecutionService
    from src.tooling.tool_argument_policy import ToolArgumentPolicyHook

    validate_llm_bindings(config)

    inspection_service = InspectionService()
    evidence_recorder = EvidenceRecorder()
    # First in the chain means last in reverse after-hook order, so the ledger
    # receives the final inspected/sanitized result.
    hooks: list[ToolHook] = [evidence_recorder]
    providers = provider_runtime or build_provider_runtime()

    if enable_tool_audit:
        hooks.append(LoggingToolAuditHook())

    if not config.untrusted_content_inspection_enabled:
        inspection_service.configure(
            NullInspector(),
            mode="warn",
            fail_policy="fail_open",
        )
    else:
        mode = config.untrusted_content_inspection_mode
        fail_policy = config.untrusted_content_fail_policy
        backend_name = config.untrusted_content_backend

        inspector: ContentInspector
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
        arg_policy_mode: Literal["warn", "block"] = (
            "block" if mode == "block" else "warn"
        )
        hooks.append(ToolArgumentPolicyHook(mode=arg_policy_mode))
        hooks.append(ContentInspectionHook(inspection_service))
        if logger is not None:
            logger.info(
                "content_inspection_enabled",
                mode=mode,
                fail_policy=fail_policy,
                backend=backend_name,
            )

    # Build MCP runtime if enabled
    mcp_runtime = None
    if getattr(config, "mcp_enabled", False):
        try:
            from src.mcp.budget import MCPBudgetHook
            from src.mcp.client import MCPRuntime
            from src.mcp.config import load_registry

            servers_path = config.mcp_servers_path
            # MCP is optional: invalid registry config disables MCP for this run
            # with a warning instead of breaking the rest of the analysis.
            servers = load_registry(servers_path, required=True)
            mcp_runtime = MCPRuntime(
                servers=servers,
                budget_db_path=str(config.mcp_usage_db_path),
            )
            hooks.append(MCPBudgetHook(mcp_runtime))
            if logger is not None:
                logger.info(
                    "mcp_runtime_initialized",
                    server_count=len([s for s in servers if s.enabled]),
                )
        except Exception as exc:
            if logger is not None:
                logger.warning(
                    "mcp_runtime_init_failed",
                    reason=redact_sensitive_text(str(exc), max_chars=120),
                )
            mcp_runtime = None

    return RuntimeServices(
        tool_service=ToolExecutionService(hooks),
        inspection_service=inspection_service,
        providers=providers,
        mcp_runtime=mcp_runtime,
        evidence_recorder=evidence_recorder,
        issuer_authority=IssuerAuthorityRegistry(),
    )
