"""The suite's offline boundary: refusal, recording, strict teardown, opt-in integration.

``_offline_boundary`` (``tests/conftest.py``) refuses every network path for a
non-``integration`` test and fails the test at teardown if it tried. The refusal
tests below install their own ``install_offline_network`` so the attempts they make
on purpose land in their own list rather than failing them at teardown.
"""

from __future__ import annotations

import asyncio
import socket
import threading
from pathlib import Path

import pytest
from curl_cffi import requests as curl_requests
from curl_cffi.requests.exceptions import ConnectionError as CurlConnectionError

from tests.helpers.offline_network import (
    EmbeddingProviderUnreachable,
    install_offline_network,
)

pytest_plugins = ["pytester"]

_TEST_NET = "192.0.2.1"
_REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def attempts(monkeypatch):
    return install_offline_network(monkeypatch)


def test_tcp_connect_is_refused_and_recorded(attempts):
    with pytest.raises(OSError, match="network disabled"):
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((_TEST_NET, 80))
    assert attempts == [f"socket:{socket.AF_INET}"]


def test_udp_sendto_is_refused_and_recorded(attempts):
    with pytest.raises(OSError, match="network disabled"):
        socket.socket(socket.AF_INET, socket.SOCK_DGRAM).sendto(b"x", (_TEST_NET, 53))
    assert attempts == [f"socket:{socket.AF_INET}"]


def test_ipv6_socket_is_refused(attempts):
    with pytest.raises(OSError):
        socket.socket(socket.AF_INET6)
    assert attempts == [f"socket:{socket.AF_INET6}"]


def test_dns_is_refused_and_records_the_host(attempts):
    with pytest.raises(socket.gaierror):
        socket.getaddrinfo("example.com", 443)
    assert attempts == ["dns:example.com"]


def test_create_connection_is_refused_at_name_resolution(attempts):
    with pytest.raises(OSError):
        socket.create_connection(("example.com", 443), timeout=1)
    assert attempts == ["dns:example.com"]


@pytest.mark.asyncio
async def test_asyncio_open_connection_is_refused(attempts):
    with pytest.raises(OSError):
        await asyncio.open_connection(_TEST_NET, 80)
    assert attempts == [f"socket:{socket.AF_INET}"]


def test_curl_sync_raises_the_libcurl_connection_error(attempts):
    with pytest.raises(CurlConnectionError):
        curl_requests.Session().get("https://example.com/quote")
    assert attempts == ["curl:https://example.com/quote"]


@pytest.mark.asyncio
async def test_curl_async_raises_the_libcurl_connection_error(attempts):
    async with curl_requests.AsyncSession() as session:
        with pytest.raises(CurlConnectionError):
            await session.get("https://example.com/async")
    assert attempts == ["curl:https://example.com/async"]


def test_curl_connection_error_is_an_oserror():
    assert issubclass(CurlConnectionError, OSError)


def test_unix_sockets_and_socketpair_stay_allowed(attempts, tmp_path):
    left, right = socket.socketpair()
    try:
        left.sendall(b"ping")
        assert right.recv(4) == b"ping"
    finally:
        left.close()
        right.close()
    unix = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    unix.close()
    assert attempts == []


@pytest.mark.asyncio
async def test_event_loop_self_pipe_is_not_an_attempt(attempts):
    await asyncio.sleep(0)
    assert attempts == []


def test_a_background_thread_attempt_is_recorded_with_its_thread_name(attempts):
    def reach() -> None:
        try:
            socket.getaddrinfo("worker.example", 80)
        except OSError:
            pass

    worker = threading.Thread(target=reach, name="egress-worker")
    worker.start()
    worker.join()

    assert attempts == ["dns:worker.example"]
    assert attempts.threads == ["egress-worker"]
    # The call site names the code that tried, which is what finds a probe that
    # only runs on another platform (the CI-only socket:2 attempts).
    assert attempts.describe().startswith(
        "dns:worker.example (thread egress-worker, at tests/test_offline_boundary.py:"
    )
    assert attempts.describe().endswith(" reach)")


def test_a_tests_own_transport_patch_wins(monkeypatch, attempts):
    sentinel = object()
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: sentinel)
    assert socket.getaddrinfo("example.com", 80) is sentinel
    assert attempts == []


def test_embedding_sdks_are_refused_for_a_unit_test():
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    client = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001", google_api_key="test-dummy"
    )
    with pytest.raises(EmbeddingProviderUnreachable):
        client.embed_query("x")


# --- the fixture itself, driven through an inner pytest session ------------------

_INNER_CONFTEST = """
from tests.conftest import (  # noqa: F401
    _offline_boundary,
    pytest_addoption,
    pytest_collection_modifyitems,
)
"""


@pytest.fixture(autouse=True)
def _inner_sessions_import_the_repo(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", str(_REPO_ROOT))


def _inner(pytester: pytest.Pytester, body: str) -> pytest.Pytester:
    """Build an inner project; it runs in a subprocess so this session's own
    boundary patches cannot mask what the inner session does."""
    pytester.makeconftest(_INNER_CONFTEST)
    pytester.makepyfile(test_inner=body)
    return pytester


def test_an_unmarked_test_that_touches_the_network_fails_at_teardown(
    pytester: pytest.Pytester,
):
    _inner(
        pytester,
        """
        import socket

        def test_leaks():
            try:
                socket.getaddrinfo("leak.example", 80)
            except OSError:
                pass  # swallowed, as production fallbacks do
        """,
    )
    result = pytester.runpytest_subprocess("-p", "no:cacheprovider")
    result.assert_outcomes(passed=1, errors=1)
    result.stdout.fnmatch_lines(
        [
            "*1 network attempt(s) in a unit test: dns:leak.example "
            "(thread MainThread, at *test_leaks). Inject a fake at the data "
            "source or mark the test integration.*",
        ]
    )


def test_a_clean_unmarked_test_passes(pytester: pytest.Pytester):
    _inner(pytester, "def test_clean():\n    assert True\n")
    pytester.runpytest_subprocess("-p", "no:cacheprovider").assert_outcomes(passed=1)


def test_integration_tests_are_skipped_without_the_option(
    pytester: pytest.Pytester,
):
    _inner(
        pytester,
        """
        import pytest

        @pytest.mark.integration
        def test_live():
            raise AssertionError("must not run")
        """,
    )
    result = pytester.runpytest_subprocess("-p", "no:cacheprovider", "-rs")
    result.assert_outcomes(skipped=1)
    result.stdout.fnmatch_lines(["*pass --run-integration to run*"])


def test_integration_tests_run_unrefused_with_the_option(
    pytester: pytest.Pytester,
):
    _inner(
        pytester,
        """
        import pytest, socket

        @pytest.mark.integration
        def test_live():
            # Not refused and not failed at teardown: the real resolver answers
            # or errors on its own terms, never with the boundary's message.
            try:
                socket.getaddrinfo("localhost", 80)
            except OSError as exc:
                assert "network disabled" not in str(exc)
        """,
    )
    result = pytester.runpytest_subprocess(
        "-p", "no:cacheprovider", "--run-integration"
    )
    result.assert_outcomes(passed=1)


def test_a_security_test_is_refused_but_its_embedding_sdk_is_not(
    pytester: pytest.Pytester,
):
    _inner(
        pytester,
        """
        import socket
        import pytest

        @pytest.mark.security
        def test_secure():
            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            # The SDK method is the real one: it is not replaced by the refusal.
            method = GoogleGenerativeAIEmbeddings.embed_query
            assert method.__qualname__ == "GoogleGenerativeAIEmbeddings.embed_query"
            with pytest.raises(socket.gaierror):
                socket.getaddrinfo("secure.example", 80)
        """,
    )
    result = pytester.runpytest_subprocess("-p", "no:cacheprovider")
    # Passes the body; the deliberate DNS attempt is then reported at teardown.
    result.assert_outcomes(passed=1, errors=1)
    result.stdout.fnmatch_lines(["*dns:secure.example*"])


def test_a_unit_test_inner_session_refuses_the_embedding_sdk(
    pytester: pytest.Pytester,
):
    _inner(
        pytester,
        """
        import pytest

        def test_unit():
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            from tests.helpers.offline_network import EmbeddingProviderUnreachable

            client = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001", google_api_key="test-dummy"
            )
            with pytest.raises(EmbeddingProviderUnreachable):
                client.embed_query("x")
        """,
    )
    pytester.runpytest_subprocess("-p", "no:cacheprovider").assert_outcomes(passed=1)
