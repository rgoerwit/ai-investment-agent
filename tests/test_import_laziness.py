"""Regression tests for import-time initialization safety."""

import importlib
import sys


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


def test_toolkit_facade_defers_market_tool_import():
    sys.modules.pop("src.toolkit", None)
    sys.modules.pop("src.tools.registry", None)
    sys.modules.pop("src.tools.market", None)

    import src.toolkit as toolkit_module

    assert "src.tools.market" not in sys.modules

    toolkit = toolkit_module.toolkit
    assert "src.tools.market" not in sys.modules

    market_tools = toolkit.get_market_tools()
    assert market_tools
    assert "src.tools.market" in sys.modules
