"""Shared pytest fixtures.

The quota store fixture redirects every Settings instance created in
tests to a per-test JSON file, so quota counters never leak between
tests and never touch a real ``usage.json`` in the repo root.

DNS is stubbed for every test. The URL-fetching guard in
chord.skills._fetch calls ``socket.getaddrinfo`` directly, which respx
cannot intercept, so without this the suite would need a resolver - and
worse, a test asserting "this address is refused" could pass merely
because the name failed to resolve.

The font fixture does the same for the chart font: the cache goes to a
per-test directory, and the resolved path - which chord.fonts memoizes
for the whole process - is cleared around every test so no test inherits
another one's answer.
"""

from __future__ import annotations

import ipaddress
import socket

import pytest

from chord.fonts import forget_resolved_font
from chord.skills import _fetch
from chord.skills._http import close_shared_client


@pytest.fixture(autouse=True)
def _isolated_quota_store(tmp_path, monkeypatch):
    monkeypatch.setenv("QUOTA_STORE_PATH", str(tmp_path / "usage.json"))


@pytest.fixture(autouse=True)
def _isolated_font_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("FONT_CACHE_DIR", str(tmp_path / "fonts"))
    forget_resolved_font()
    yield
    forget_resolved_font()


#: Hostnames the fake resolver knows about. Anything else is NXDOMAIN.
FAKE_DNS = {
    "example.com": "93.184.216.34",
    "docs.example.org": "93.184.216.36",
    "docs.python.org": "151.101.128.223",
    "peps.python.org": "151.101.128.223",
    "evil.example": "93.184.216.35",
    "localhost": "127.0.0.1",
    "internal.example": "10.1.2.3",
}


@pytest.fixture(autouse=True)
def _fake_dns(monkeypatch):
    def getaddrinfo(host, *_args, **_kwargs):
        try:  # an IP literal resolves to itself
            address = str(ipaddress.ip_address(host))
        except ValueError:
            if host not in FAKE_DNS:
                raise socket.gaierror(f"no fake record for {host}") from None
            address = FAKE_DNS[host]
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 0))]

    monkeypatch.setattr(_fetch.socket, "getaddrinfo", getaddrinfo)


@pytest.fixture(autouse=True)
async def _fresh_http_client():
    """No shared HTTP client survives a test.

    The client is bound to the event loop that made it, and every test
    gets its own, so leaving one behind would hand the next test a
    client wired to a dead loop.
    """
    yield
    await close_shared_client()
