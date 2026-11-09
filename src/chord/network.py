"""Public-only connections for URLs supplied by chat users.

DNS checks must happen where a socket is opened. Checking a hostname and
then letting the HTTP client resolve it again allows DNS rebinding. This
backend resolves once, checks every address, and connects to a checked IP.
The connection pool still owns the original hostname, so HTTPS certificates,
TLS SNI, and the HTTP Host header continue to use the requested hostname.
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from collections.abc import AsyncIterator

import httpcore
import httpx


def public_addresses(host: str) -> list[str]:
    """Resolve a host and reject the entire answer if any address is private."""
    records = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    addresses = list(dict.fromkeys(record[4][0] for record in records))
    if not addresses:
        raise ValueError("The hostname has no public address.")
    for address in addresses:
        parsed = ipaddress.ip_address(address)
        # An IPv4-mapped IPv6 address must pass the IPv4 rules as well.
        mapped = getattr(parsed, "ipv4_mapped", None)
        if not parsed.is_global or (mapped is not None and not mapped.is_global):
            raise ValueError("The hostname is not a public address.")
    return addresses


class PublicNetworkBackend(httpcore.AsyncNetworkBackend):
    """Open TCP sockets only to numeric addresses that passed our checks."""

    def __init__(self, backend: httpcore.AsyncNetworkBackend | None = None) -> None:
        self._backend = backend if backend is not None else httpcore.AnyIOBackend()

    async def connect_tcp(
        self, host, port, timeout=None, local_address=None, socket_options=None
    ) -> httpcore.AsyncNetworkStream:
        try:
            addresses = await asyncio.wait_for(
                asyncio.to_thread(public_addresses, host), timeout=timeout
            )
        except TimeoutError as exc:
            raise httpcore.ConnectTimeout("Public hostname lookup timed out.") from exc
        except (OSError, ValueError) as exc:
            raise httpcore.ConnectError("The hostname has no allowed public address.") from exc

        # Passing a numeric IP prevents a second hostname lookup from choosing
        # an unchecked address. Use one address per attempt; the caller may retry.
        return await self._backend.connect_tcp(
            addresses[0], port, timeout, local_address, socket_options
        )

    async def connect_unix_socket(self, *args, **kwargs):
        raise httpcore.ConnectError("Local sockets are not allowed for fetched URLs.")

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


class _ResponseStream(httpx.AsyncByteStream):
    def __init__(self, stream) -> None:
        self._stream = stream

    async def __aiter__(self) -> AsyncIterator[bytes]:
        try:
            async for chunk in self._stream:
                yield chunk
        except httpcore.TimeoutException as exc:
            raise httpx.ReadTimeout("The public server took too long to respond.") from exc
        except (httpcore.NetworkError, httpcore.ProtocolError) as exc:
            raise httpx.ReadError("The public server connection failed.") from exc

    async def aclose(self) -> None:
        await self._stream.aclose()


class PublicTransport(httpx.AsyncBaseTransport):
    """Small HTTPX adapter around a pool with the guarded network backend.

    This pool has no proxy support. Environment proxy settings must not bypass
    the destination check or route a user-supplied link through a local proxy.
    """

    def __init__(self) -> None:
        self._pool = httpcore.AsyncConnectionPool(network_backend=PublicNetworkBackend())

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        core_request = httpcore.Request(
            method=request.method,
            url=httpcore.URL(
                scheme=request.url.raw_scheme,
                host=request.url.raw_host,
                port=request.url.port,
                target=request.url.raw_path,
            ),
            headers=request.headers.raw,
            content=request.stream,
            extensions=request.extensions,
        )
        try:
            response = await self._pool.handle_async_request(core_request)
        except httpcore.TimeoutException as exc:
            raise httpx.ConnectTimeout("The public server connection timed out.") from exc
        except (httpcore.NetworkError, httpcore.ProtocolError) as exc:
            raise httpx.ConnectError("Could not connect to an allowed public server.") from exc
        return httpx.Response(
            response.status,
            headers=response.headers,
            stream=_ResponseStream(response.stream),
            extensions=response.extensions,
        )

    async def aclose(self) -> None:
        await self._pool.aclose()
