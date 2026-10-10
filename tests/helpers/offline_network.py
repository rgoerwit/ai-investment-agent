"""Refuse and record every way a test could reach the network.

Covers what pytest-socket alone misses: libcurl (``curl_cffi``, yfinance's
transport) never touches Python sockets. Use it where a test must prove that
nothing left the process -- assert the returned list is empty.
"""

from __future__ import annotations

import socket
from typing import Any

import pytest


def install_offline_network(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    attempts: list[str] = []
    real_socket = socket.socket

    class _OfflineSocket(real_socket):  # type: ignore[misc, valid-type]
        def __init__(
            self, family: Any = -1, type: Any = -1, proto: Any = -1, fileno: Any = None
        ) -> None:
            if fileno is None and family != socket.AF_UNIX:
                attempts.append(f"socket:{family}")
                raise OSError("network disabled in this test")
            super().__init__(family, type, proto, fileno)

    def offline_getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
        attempts.append(f"dns:{host}")
        raise socket.gaierror("network disabled in this test")

    monkeypatch.setattr(socket, "socket", _OfflineSocket)
    monkeypatch.setattr(socket, "getaddrinfo", offline_getaddrinfo)

    import curl_cffi.requests as curl_requests

    def offline_curl(
        self: Any, method: Any, url: Any, *args: Any, **kwargs: Any
    ) -> Any:
        attempts.append(f"curl:{url}")
        raise OSError("network disabled in this test")

    async def offline_curl_async(
        self: Any, method: Any, url: Any, *args: Any, **kwargs: Any
    ) -> Any:
        return offline_curl(self, method, url)

    monkeypatch.setattr(curl_requests.Session, "request", offline_curl)
    monkeypatch.setattr(curl_requests.AsyncSession, "request", offline_curl_async)
    return attempts
