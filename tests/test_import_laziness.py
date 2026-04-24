"""Regression tests for import-time initialization and runtime seam hygiene."""

import importlib
import importlib.util
import sys
from pathlib import Path


def test_src_llms_import_does_not_construct_default_models(monkeypatch):
    import langchain_google_genai

    init_calls = []

    class StubChatGoogleGenerativeAI:
        marker = "ready"

        def __init__(self, **kwargs):
            init_calls.append(kwargs)

    monkeypatch.setattr(
        langchain_google_genai,
        "ChatGoogleGenerativeAI",
        StubChatGoogleGenerativeAI,
    )

    llms = importlib.import_module("src.llms")
    llms = importlib.reload(llms)

    assert init_calls == []
    assert llms.quick_thinking_llm.marker == "ready"
    assert len(init_calls) == 1


def test_src_llms_import_does_not_construct_global_rate_limiter(monkeypatch):
    rate_limiters = importlib.import_module("langchain_core.rate_limiters")
    init_calls = []

    class StubInMemoryRateLimiter:
        def __init__(self, **kwargs):
            init_calls.append(kwargs)
            self.requests_per_second = kwargs["requests_per_second"]
            self.check_every_n_seconds = kwargs["check_every_n_seconds"]
            self.max_bucket_size = kwargs["max_bucket_size"]

        def acquire(self, *, blocking=True):
            return True

        async def aacquire(self, *, blocking=True):
            return True

    monkeypatch.setattr(
        rate_limiters,
        "InMemoryRateLimiter",
        StubInMemoryRateLimiter,
    )

    llms = importlib.import_module("src.llms")
    llms = importlib.reload(llms)

    assert init_calls == []
    assert "lazy" in repr(llms.GLOBAL_RATE_LIMITER)

    _ = llms.GLOBAL_RATE_LIMITER.requests_per_second
    assert len(init_calls) == 1


def test_src_data_fetcher_import_does_not_construct_singleton(monkeypatch):
    init_calls = []

    av_fetcher = importlib.import_module("src.data.alpha_vantage_fetcher")
    eodhd_fetcher = importlib.import_module("src.data.eodhd_fetcher")
    fmp_fetcher = importlib.import_module("src.data.fmp_fetcher")
    fetcher_module = importlib.import_module("src.data.fetcher")

    monkeypatch.setattr(
        fmp_fetcher,
        "get_fmp_fetcher",
        lambda: init_calls.append("fmp") or object(),
    )
    monkeypatch.setattr(
        eodhd_fetcher,
        "get_eodhd_fetcher",
        lambda: init_calls.append("eodhd") or object(),
    )
    monkeypatch.setattr(
        av_fetcher,
        "get_av_fetcher",
        lambda: init_calls.append("alpha_vantage") or object(),
    )

    fetcher_module = importlib.reload(fetcher_module)

    assert init_calls == []
    assert "lazy" in repr(fetcher_module.fetcher)


def test_runtime_code_uses_tool_registry_instead_of_removed_toolkit_facade():
    graph_components = Path("src/graph/components.py").read_text(encoding="utf-8")
    consultant_tools = Path("src/consultant_tools.py").read_text(encoding="utf-8")

    assert "from src.toolkit import toolkit" not in graph_components
    assert "from src.tools.registry import toolkit" in graph_components
    assert "from src.toolkit import get_official_filings" not in consultant_tools


def test_legacy_toolkit_facade_file_is_removed():
    assert not Path("src/toolkit.py").exists()


def test_tools_package_no_longer_reexports_legacy_tool_surface():
    src = Path("src/tools/__init__.py").read_text(encoding="utf-8")
    assert "__getattr__" not in src
    assert "get_financial_metrics" not in src


def test_src_package_root_is_inert():
    src = Path("src/__init__.py").read_text(encoding="utf-8")
    assert "toolkit" not in src
    assert "from src." not in src


def test_tooling_package_init_is_inert():
    src = Path("src/tooling/__init__.py").read_text(encoding="utf-8")
    assert "from src.tooling." not in src


def test_portfolio_manager_cold_start_import_has_no_runtime_services_cycle():
    spec = importlib.util.spec_from_file_location(
        "portfolio_manager_import_test",
        Path("scripts/portfolio_manager.py"),
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
