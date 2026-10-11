"""Refuse and record every way a test could reach the network.

``install_offline_network`` is the one mechanism behind the suite's ``_offline_boundary``
fixture (``tests/conftest.py``) and the tests that prove a code path sends nothing.
It covers what pytest-socket never could: libcurl (``curl_cffi``, yfinance's
transport) does not touch Python sockets, so it is refused at the session methods.
``refuse_embedding_sdks`` is the sibling for the embedding SDKs, which are refused
before they build a request.
"""

from __future__ import annotations

import socket
import threading
from typing import Any

import pytest


class OfflineAttempts(list[str]):
    """Targets a test tried to reach, with the thread that made each attempt."""

    def __init__(self) -> None:
        super().__init__()
        self.threads: list[str] = []

    def record(self, target: str) -> None:
        self.append(target)
        self.threads.append(threading.current_thread().name)

    def describe(self) -> str:
        return ", ".join(
            f"{target} (thread {thread})"
            for target, thread in zip(self, self.threads, strict=True)
        )


class EmbeddingProviderUnreachable(ConnectionError):
    """What every embedding-SDK request raises in a unit test."""


_EMBEDDING_SDK_METHODS = (
    "embed_query",
    "embed_documents",
    "aembed_query",
    "aembed_documents",
)


def install_offline_network(monkeypatch: pytest.MonkeyPatch) -> OfflineAttempts:
    """Refuse TCP/UDP sockets, DNS and libcurl; record each attempt.

    AF_UNIX sockets and ``socketpair`` (a ``fileno`` is passed) stay allowed:
    asyncio's self-pipe and local IPC are not network access.
    """
    # urllib3 probes IPv6 once, at import, by binding an AF_INET6 socket to ::1
    # (no connection). Import it first so a lazy first import inside a test is not
    # recorded as an attempt.
    import urllib3.util.connection  # noqa: F401

    attempts = OfflineAttempts()
    real_socket = socket.socket

    class _OfflineSocket(real_socket):  # type: ignore[misc, valid-type]
        def __init__(
            self, family: Any = -1, type: Any = -1, proto: Any = -1, fileno: Any = None
        ) -> None:
            if fileno is None and family != socket.AF_UNIX:
                attempts.record(f"socket:{family}")
                raise OSError("network disabled in this test")
            super().__init__(family, type, proto, fileno)

    def offline_getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
        attempts.record(f"dns:{host}")
        raise socket.gaierror("network disabled in this test")

    monkeypatch.setattr(socket, "socket", _OfflineSocket)
    monkeypatch.setattr(socket, "getaddrinfo", offline_getaddrinfo)

    import curl_cffi.requests as curl_requests
    from curl_cffi.requests.exceptions import ConnectionError as CurlConnectionError

    def offline_curl(
        self: Any, method: Any, url: Any, *args: Any, **kwargs: Any
    ) -> Any:
        attempts.record(f"curl:{url}")
        raise CurlConnectionError("network disabled in this test")

    async def offline_curl_async(
        self: Any, method: Any, url: Any, *args: Any, **kwargs: Any
    ) -> Any:
        return offline_curl(self, method, url)

    monkeypatch.setattr(curl_requests.Session, "request", offline_curl)
    monkeypatch.setattr(curl_requests.AsyncSession, "request", offline_curl_async)
    return attempts


def refuse_embedding_sdks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every embedding-SDK request raise ``EmbeddingProviderUnreachable``.

    The session's dummy ``GOOGLE_API_KEY`` is enough for ``FinancialSituationMemory``
    to build a real embeddings client and run its startup probe. Refusing at the SDK
    method keeps the client real for construction assertions, fails the probe exactly
    as an unreachable provider would, and lets a test's own patch of the client or its
    transport win. Refused before sending, so it records no network attempt.
    """
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from langchain_openai import OpenAIEmbeddings

    def refuse(self: Any, *args: Any, **kwargs: Any) -> Any:
        raise EmbeddingProviderUnreachable(
            "embedding provider unreachable: unit tests run offline"
        )

    async def arefuse(self: Any, *args: Any, **kwargs: Any) -> Any:
        refuse(self)

    for client_class in (GoogleGenerativeAIEmbeddings, OpenAIEmbeddings):
        for method in _EMBEDDING_SDK_METHODS:
            replacement = arefuse if method.startswith("a") else refuse
            monkeypatch.setattr(client_class, method, replacement)
