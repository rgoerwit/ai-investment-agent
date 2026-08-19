from types import SimpleNamespace

from src.runtime_services import ProviderRuntime, ProviderRuntimeKey


def test_same_vendor_endpoints_can_have_independent_buckets() -> None:
    default = SimpleNamespace(name="default")
    east = SimpleNamespace(name="east")
    west = SimpleNamespace(name="west")
    runtime = ProviderRuntime(
        fetcher=SimpleNamespace(),
        rate_limiter=default,
        rate_limiters={
            ProviderRuntimeKey("openai", "east.example"): east,
            ProviderRuntimeKey("openai", "west.example"): west,
        },
    )
    assert runtime.limiter_for("openai", "east.example") is east
    assert runtime.limiter_for("openai", "west.example") is west
    assert east is not west


def test_vendor_default_precedes_and_unknown_does_not_share_google_bucket() -> None:
    fallback = SimpleNamespace(name="legacy")
    vendor = SimpleNamespace(name="vendor")
    runtime = ProviderRuntime(
        fetcher=SimpleNamespace(),
        rate_limiter=fallback,
        rate_limiters={ProviderRuntimeKey("google"): vendor},
    )
    assert runtime.limiter_for("google", "generativelanguage.googleapis.com") is vendor
    assert runtime.limiter_for("unknown") is None
