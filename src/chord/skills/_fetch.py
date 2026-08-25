"""Fetching a URL a chat user pasted, without becoming their proxy.

Every other skill talks to an endpoint chord chose. This one talks to
whatever address someone typed into a Discord channel, which is a
different thing entirely: without checks, anyone who can mention the bot
can make the machine it runs on issue requests on their behalf - at
``169.254.169.254`` for cloud credentials, at ``localhost:8080`` for the
admin panel next door, at anything inside the network the bot happens to
sit in.

So the rules here are about *where* a request may go, not about parsing:

* http and https only - no file://, no gopher://, no data:.
* The host must resolve to a public address. Every address it resolves
  to, in fact; one public A record does not make a hostname safe if the
  next one is 127.0.0.1.
* Redirects are followed by hand, re-checking each hop, because a public
  URL that 302s to ``http://10.0.0.1/`` would otherwise walk straight
  past the check at the front door.

This is not airtight - a name that passes the check and then resolves
differently when httpx connects (DNS rebinding) still gets through. It
raises the cost of the attack from "paste a link" to "run a DNS server",
which for a chat bot is the right place to stop.
"""

from __future__ import annotations

import ipaddress
import logging
import re
import socket
from dataclasses import dataclass
from urllib.parse import urlsplit

import httpx

from chord.skills._http import DEFAULT_HEADERS, SkillHTTPError

logger = logging.getLogger(__name__)

#: The only two schemes a chat user has any business pointing us at.
ALLOWED_SCHEMES = ("http", "https")

#: Stop reading past this. A news article is tens of kilobytes; anything
#: in megabytes is a download, and this skill does not do downloads.
MAX_RESPONSE_BYTES = 2 * 1024 * 1024

#: Redirect chains longer than this are a loop or a tracker maze.
MAX_REDIRECTS = 5

#: Long enough for a slow news site, short enough that a hung server
#: does not hold a chat turn open.
FETCH_TIMEOUT = 20.0

#: Content types worth handing to a language model as text. Anything
#: else - a PDF, a zip - is reported rather than silently mangled.
TEXTUAL_TYPES = ("text/", "application/json", "application/xml", "+json", "+xml")

#: Content types a decoder can look at. Kept separate from the textual
#: list because "can a model read this?" and "can Pillow open this?" are
#: different questions with different right answers.
IMAGE_TYPES = ("image/",)

#: Images get more room than pages: a phone screenshot of a QR code is
#: routinely bigger than an article, and it is one file rather than a
#: whole document tree.
MAX_IMAGE_BYTES = 8 * 1024 * 1024

#: Charset declared in the document rather than in the headers. Korean
#: pages still ship EUC-KR this way, and httpx assumes UTF-8 when the
#: header says nothing, which turns the whole article into mojibake.
_META_CHARSET_RE = re.compile(
    rb"""<meta[^>]+charset\s*=\s*["']?\s*([a-zA-Z0-9_\-]+)""",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class FetchedBytes:
    """What came back, before anyone decides what it means."""

    url: str
    content_type: str
    body: bytes
    truncated: bool


@dataclass(frozen=True)
class FetchedPage:
    """A fetched document, decoded but not yet interpreted."""

    url: str
    content_type: str
    text: str
    truncated: bool


def normalize_url(raw: str) -> str:
    """Tidy what a person actually pastes into a chat window.

    Discord users wrap links in angle brackets to suppress the embed,
    quote them, and leave the trailing bracket of a sentence attached.
    A bare ``example.com`` is a URL to everyone except a URL parser.
    """
    url = (raw or "").strip().strip("<>").strip().rstrip(".,;)")
    if not url:
        raise SkillHTTPError("No URL was given.")
    if "://" not in url:
        url = f"https://{url}"
    return url


def is_public_address(host: str) -> bool:
    """Whether every address ``host`` resolves to is on the public net."""
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False

    addresses = {info[4][0] for info in infos}
    if not addresses:
        return False

    for address in addresses:
        try:
            parsed = ipaddress.ip_address(address)
        except ValueError:
            return False
        if (
            parsed.is_private
            or parsed.is_loopback
            or parsed.is_link_local  # 169.254.0.0/16 - cloud metadata lives here
            or parsed.is_reserved
            or parsed.is_multicast
            or parsed.is_unspecified
        ):
            return False
    return True


def assert_fetchable(url: str) -> None:
    """Raise unless ``url`` is a public http(s) address we may request."""
    parts = urlsplit(url)
    if parts.scheme not in ALLOWED_SCHEMES:
        raise SkillHTTPError(f"Only http and https links can be opened, not '{parts.scheme}:'.")
    if not parts.hostname:
        raise SkillHTTPError(f"'{url}' has no host to connect to.")
    if not is_public_address(parts.hostname):
        logger.warning("Refusing to fetch %s: host is not a public address.", url)
        raise SkillHTTPError(f"'{parts.hostname}' is not a public address, so I will not open it.")


def is_textual(content_type: str) -> bool:
    return _type_matches(content_type, TEXTUAL_TYPES)


def decode_body(body: bytes, content_type: str) -> str:
    """Decode bytes using the charset the server or the document names."""
    charset = ""
    if "charset=" in content_type.lower():
        charset = content_type.lower().split("charset=", 1)[1].split(";")[0].strip().strip("\"'")
    if not charset:
        match = _META_CHARSET_RE.search(body[:4096])
        if match:
            charset = match.group(1).decode("ascii", "ignore")

    for candidate in (charset, "utf-8"):
        if not candidate:
            continue
        try:
            return body.decode(candidate)
        except (LookupError, UnicodeDecodeError):
            continue
    # Something is left of every page, even one in an encoding nobody
    # declared correctly - a summary of mostly-right text beats an error.
    return body.decode("utf-8", errors="replace")


async def fetch_page(url: str) -> FetchedPage:
    """GET a user-supplied URL and decode it as text.

    Raises:
        SkillHTTPError: for anything the model should tell the user
            about - a blocked address, a dead link, a PDF, a timeout.
    """
    fetched = await fetch_bytes(
        url,
        accept=TEXTUAL_TYPES,
        accept_header="text/html,application/xhtml+xml,text/plain;q=0.9",
        rejection="which I cannot read as text",
    )
    return FetchedPage(
        url=fetched.url,
        content_type=fetched.content_type,
        text=decode_body(fetched.body, fetched.content_type),
        truncated=fetched.truncated,
    )


async def fetch_image(url: str) -> FetchedBytes:
    """GET a user-supplied URL that is expected to be an image."""
    return await fetch_bytes(
        url,
        accept=IMAGE_TYPES,
        accept_header="image/*",
        rejection="which is not an image",
        max_bytes=MAX_IMAGE_BYTES,
    )


async def fetch_bytes(
    url: str,
    *,
    accept: tuple[str, ...],
    accept_header: str,
    rejection: str,
    max_bytes: int = MAX_RESPONSE_BYTES,
) -> FetchedBytes:
    """GET a user-supplied URL, following redirects the careful way.

    The address rules live here and nowhere else, so every caller that
    fetches something a chat user named - a page, an image - gets the
    same guard whether it remembers to ask for it or not.
    """
    current = normalize_url(url)
    headers = {**DEFAULT_HEADERS, "Accept": accept_header}

    async with httpx.AsyncClient(
        timeout=FETCH_TIMEOUT,
        follow_redirects=False,
        headers=headers,
    ) as client:
        for _hop in range(MAX_REDIRECTS + 1):
            assert_fetchable(current)
            try:
                async with client.stream("GET", current) as response:
                    if response.is_redirect:
                        current = _next_hop(current, response)
                        continue
                    if response.status_code >= 400:
                        raise SkillHTTPError(f"{current} answered HTTP {response.status_code}.")

                    content_type = response.headers.get("content-type", "")
                    if not _type_matches(content_type, accept):
                        raise SkillHTTPError(
                            f"{current} is {content_type or 'an unknown type'}, {rejection}."
                        )

                    body, truncated = await _read_capped(response, max_bytes)
            except httpx.RequestError as exc:
                logger.warning("Could not fetch %s: %s", current, exc)
                raise SkillHTTPError(f"Could not reach {current}.") from exc

            return FetchedBytes(
                url=current,
                content_type=content_type,
                body=body,
                truncated=truncated,
            )

    raise SkillHTTPError(f"{url} redirected more than {MAX_REDIRECTS} times; giving up.")


def _type_matches(content_type: str, accept: tuple[str, ...]) -> bool:
    lowered = content_type.lower()
    return any(marker in lowered for marker in accept)


def _next_hop(current: str, response: httpx.Response) -> str:
    location = response.headers.get("location")
    if not location:
        raise SkillHTTPError(f"{current} redirected without saying where to.")
    return str(httpx.URL(current).join(location))


async def _read_capped(response: httpx.Response, max_bytes: int) -> tuple[bytes, bool]:
    """Read a response body, stopping at ``max_bytes``."""
    chunks: list[bytes] = []
    size = 0
    async for chunk in response.aiter_bytes():
        chunks.append(chunk)
        size += len(chunk)
        if size >= max_bytes:
            return b"".join(chunks)[:max_bytes], True
    return b"".join(chunks), False
