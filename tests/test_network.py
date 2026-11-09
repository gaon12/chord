"""Socket-level destination checks, independent of HTTP response mocks."""

import socket
from unittest.mock import AsyncMock

import httpcore
import pytest

from chord.network import PublicNetworkBackend, public_addresses


@pytest.mark.parametrize(
    "address", ["127.0.0.1", "10.0.0.1", "169.254.169.254", "::1", "::ffff:127.0.0.1"]
)
def test_private_and_mapped_addresses_are_refused(address):
    with pytest.raises(ValueError):
        public_addresses(address)


def test_one_private_record_rejects_a_mixed_dns_answer(monkeypatch):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *a, **k: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 0))
            for address in ("93.184.216.34", "127.0.0.1")
        ],
    )
    with pytest.raises(ValueError):
        public_addresses("example.com")


async def test_socket_uses_checked_ip_instead_of_resolving_hostname_again():
    delegate = AsyncMock()
    await PublicNetworkBackend(delegate).connect_tcp("example.com", 443, timeout=5)
    delegate.connect_tcp.assert_awaited_once_with("93.184.216.34", 443, 5, None, None)


async def test_rebinding_between_preflight_and_connection_is_blocked(monkeypatch):
    assert public_addresses("example.com") == ["93.184.216.34"]
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 0))],
    )
    delegate = AsyncMock()
    with pytest.raises(httpcore.ConnectError):
        await PublicNetworkBackend(delegate).connect_tcp("example.com", 443)
    delegate.connect_tcp.assert_not_called()
