from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.runtime_services import (
    IssuerAuthorityRegistry,
    ProviderRuntime,
    RuntimeServices,
    get_current_issuer_hosts,
    get_current_runtime_services,
    register_current_issuer_url,
    use_runtime_services,
)
from src.tooling.inspection_service import InspectionService
from src.tooling.runtime import ToolExecutionService


def test_runtime_services_context_restores_after_exit():
    services = RuntimeServices(
        tool_service=ToolExecutionService(),
        inspection_service=InspectionService(),
    )

    assert get_current_runtime_services() is None
    with use_runtime_services(services):
        assert get_current_runtime_services() is services
    assert get_current_runtime_services() is None


@pytest.mark.asyncio
async def test_fetcher_proxy_uses_runtime_bound_provider():
    from src.data.fetcher import fetcher

    fake_fetcher = SimpleNamespace(
        get_financial_metrics=lambda *args, **kwargs: None,
        get_historical_prices=lambda *args, **kwargs: None,
    )

    async def _fake_financial_metrics(*args, **kwargs):
        return {"source": "runtime"}

    fake_fetcher.get_financial_metrics = _fake_financial_metrics
    services = RuntimeServices(
        tool_service=ToolExecutionService(),
        inspection_service=InspectionService(),
        providers=ProviderRuntime(
            fetcher=fake_fetcher,
            rate_limiter=SimpleNamespace(
                acquire=lambda **kwargs: True,
                aacquire=lambda **kwargs: True,
            ),
        ),
    )

    with use_runtime_services(services):
        result = await fetcher.get_financial_metrics("AAPL")

    assert result == {"source": "runtime"}


def test_with_extra_tool_hooks_returns_new_service_without_mutating_original():
    base_hook = object()
    extra_hook = object()
    services = RuntimeServices(
        tool_service=ToolExecutionService([base_hook]),
        inspection_service=InspectionService(),
    )

    scoped = services.with_extra_tool_hooks([extra_hook])

    assert services.tool_service.hooks == [base_hook]
    assert scoped.tool_service.hooks == [base_hook, extra_hook]


def test_issuer_authority_is_validated_and_run_scoped():
    services = RuntimeServices(
        tool_service=ToolExecutionService(),
        inspection_service=InspectionService(),
        issuer_authority=IssuerAuthorityRegistry(),
    )

    assert get_current_issuer_hosts() == ()
    with use_runtime_services(services):
        assert register_current_issuer_url(
            "https://www.issuer.example/investors",
            provenance="structured_profile",
        )
        assert not register_current_issuer_url(
            "http://issuer.example/investors",
            provenance="untrusted",
        )
        assert not register_current_issuer_url(
            "https://127.0.0.1/investors",
            provenance="untrusted",
        )
        assert get_current_issuer_hosts() == ("issuer.example",)
    assert get_current_issuer_hosts() == ()
