from __future__ import annotations

import socket

import pytest
from pytest_socket import SocketBlockedError


@pytest.mark.security
def test_security_marked_tests_block_socket_connect():
    with pytest.warns(UserWarning, match="A test tried to use socket\\.socket"):
        with pytest.raises(SocketBlockedError):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(("192.0.2.1", 80))


@pytest.mark.security
def test_security_marked_tests_block_dns_resolution():
    with pytest.raises(RuntimeError, match="network access blocked"):
        socket.getaddrinfo("example.com", 443)


@pytest.mark.security
def test_security_marked_tests_block_udp_socket_creation():
    with pytest.warns(UserWarning, match="A test tried to use socket\\.socket"):
        with pytest.raises(SocketBlockedError):
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.sendto(b"payload", ("192.0.2.1", 53))


@pytest.mark.security
@pytest.mark.asyncio
async def test_security_marked_tests_block_asyncio_connection():
    import asyncio

    with pytest.warns(UserWarning, match="A test tried to use socket\\.socket"):
        with pytest.raises(SocketBlockedError):
            await asyncio.open_connection("192.0.2.1", 80)
