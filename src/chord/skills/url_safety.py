"""URL safety-check skill combining three independent sources.

Verdicts are gathered from whichever sources are available and merged
into one answer:

1. **lrl.kr URL check v5** - Google Safe Browsing cache (needs the same
   ``LRL_API_KEY`` as the shortener; sent as a ``key`` parameter, not a
   header). Success is HTTP **201**, errors carry an ``ERR_*`` message.
2. **Cloudflare 1.1.1.2 for Families** - key-less DNS blocklist via
   DoH JSON: blocked domains resolve to ``0.0.0.0``.
3. **Cloudflare Radar URL Scanner** - live scan when
   ``CLOUDFLARE_API_KEY`` + ``CLOUDFLARE_ACCOUNT_ID`` are configured;
   best-effort and purely informational.

Final verdict: UNSAFE if ANY source flags the URL, SAFE if at least one
source actively cleared it, UNKNOWN when nothing could decide.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any, ClassVar
from urllib.parse import urlsplit

import httpx

from chord.config import Settings
from chord.skills._http import DEFAULT_HEADERS, TIMEOUT_SECONDS, SkillHTTPError, get_json
from chord.skills.base import Skill

LRL_CHECK_URL = "https://api.lrl.kr/v5/url/check"
CLOUDFLARE_DOH_URL = "https://security.cloudflare-dns.com/dns-query"  # 1.1.1.2
RADAR_SCANS_URL = "https://api.cloudflare.com/client/v4/accounts/{account}/radar/url_scanner/scans"

#: Human explanations for Safe Browsing threat codes.
THREAT_LABELS = {
    "MALWARE": "malware detected",
    "SOCIAL_ENGINEERING": "phishing / social engineering",
    "UNWANTED_SOFTWARE": "unwanted software",
    "POTENTIALLY_HARMFUL_APPLICATION": "potentially harmful application",
    "THREAT_TYPE_UNSPECIFIED": "unspecified threat",
}

URL_RE = re.compile(r"^https?://\S+$", re.I)


@dataclass
class SourceVerdict:
    """One source's opinion about the URL."""

    source: str
    status: str  # 'safe' | 'unsafe' | 'unknown' | 'skipped'
    detail: str = ""
    threats: list[str] = field(default_factory=list)


def extract_domain(url: str) -> str:
    """Hostname of a URL, for DNS-based checks."""
    host = urlsplit(url.strip()).hostname or ""
    return host.lower()


def threat_label(code: str) -> str:
    """Human explanation for a Safe Browsing threat code."""
    code = (code or "").strip().upper()
    if not code:
        return ""
    return THREAT_LABELS.get(code, code.lower().replace("_", " "))


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


async def check_lrl(api_key: str, url: str) -> SourceVerdict:
    """lrl.kr v5: Google Safe Browsing cache (key as query/body param)."""
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
            response = await client.get(
                LRL_CHECK_URL,
                params={"key": api_key, "url": url},
                headers=DEFAULT_HEADERS,
            )
        status_code = response.status_code
        try:
            body: dict[str, Any] = response.json()
        except ValueError:
            body = {}
    except httpx.RequestError as exc:
        raise SkillHTTPError("Could not reach lrl.kr URL checker.") from exc

    if status_code >= 400:
        message = body.get("message", "")
        detail = f" ({message})" if message else ""
        return SourceVerdict("lrl.kr", "unknown", f"HTTP {status_code}{detail}")

    result = body.get("result") or {}
    safe = str(result.get("safe", ""))
    threat = str(result.get("threat", "")).upper()
    if safe == "0":
        return SourceVerdict("lrl.kr", "unsafe", threat_label(threat) or "flagged unsafe", [threat])
    if safe == "1":
        return SourceVerdict("lrl.kr", "safe", "not on any blocklist")
    return SourceVerdict("lrl.kr", "unknown", "inconclusive response")


async def check_cloudflare_dns(url: str) -> SourceVerdict:
    """Cloudflare 1.1.1.2 for Families: malware domains resolve to 0.0.0.0."""
    domain = extract_domain(url)
    if not domain:
        raise SkillHTTPError(f"Could not read a domain from '{url}'.")
    data = await get_json(
        CLOUDFLARE_DOH_URL,
        params={"name": domain, "type": "A"},
        headers={"accept": "application/dns-json"},
    )

    answers = data.get("Answer") or []
    blocked = any(str(answer.get("data")) == "0.0.0.0" for answer in answers)
    # Blocked domains may also come back with Status 3 (NXDOMAIN).
    if blocked or data.get("Status") == 3:
        return SourceVerdict("Cloudflare 1.1.1.2", "unsafe", f"{domain} is blocked as malicious")
    if answers:
        return SourceVerdict("Cloudflare 1.1.1.2", "safe", f"{domain} resolves normally")
    return SourceVerdict("Cloudflare 1.1.1.2", "unknown", "no DNS answer")


async def scan_cloudflare_radar(api_key: str, account_id: str, url: str) -> SourceVerdict:
    """Best-effort live scan via Cloudflare Radar URL Scanner."""
    base_url = RADAR_SCANS_URL.format(account=account_id)
    auth_headers = {"Authorization": f"Bearer {api_key}", **DEFAULT_HEADERS}
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
            created = await client.post(base_url, json={"url": url}, headers=auth_headers)
            created.raise_for_status()
            scan_uuid = ((created.json().get("result") or {}).get("uuid")) or ""
            if not scan_uuid:
                raise SkillHTTPError("Radar did not return a scan id.")

            # Scanning is asynchronous; give it a moment then fetch once.
            await asyncio.sleep(3)
            fetched = await client.get(f"{base_url}/{scan_uuid}", headers=auth_headers)
            fetched.raise_for_status()
            result = fetched.json().get("result") or {}
    except (httpx.HTTPStatusError, httpx.RequestError, ValueError) as exc:
        return SourceVerdict("Cloudflare Radar", "unknown", f"scan failed: {exc}")

    verdicts = result.get("verdicts") or {}
    malicious = bool(verdicts.get("malicious"))
    status_text = str(result.get("status", ""))
    if malicious:
        return SourceVerdict("Cloudflare Radar", "unsafe", "live scan flagged this URL")
    if status_text in ("completed", "finished"):
        return SourceVerdict("Cloudflare Radar", "safe", "live scan found nothing")
    return SourceVerdict("Cloudflare Radar", "unknown", f"scan still {status_text or 'processing'}")


# ---------------------------------------------------------------------------
# Merging + skill
# ---------------------------------------------------------------------------


def merge_verdicts(verdicts: list[SourceVerdict]) -> tuple[str, list[str]]:
    """Combine source verdicts into (overall, reason list)."""
    reasons = [f"{v.source}: {v.detail}" for v in verdicts if v.status == "unsafe"]
    if reasons:
        return "UNSAFE", reasons
    cleared = [v.source for v in verdicts if v.status == "safe"]
    if cleared:
        return "SAFE", [f"cleared by {', '.join(cleared)}"]
    return "UNKNOWN", ["no source could decide"]


class CheckUrlSafetySkill(Skill):
    name = "check_url_safety"
    description = (
        "Check whether a URL is safe before visiting or sharing it: "
        "Google Safe Browsing cache (via lrl.kr), Cloudflare's malware "
        "blocklist and an optional Cloudflare Radar live scan."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The full URL to inspect, including https:// .",
            }
        },
        "required": ["url"],
    }

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def run(self, url: str) -> str:
        url = url.strip()
        if not URL_RE.match(url):
            raise SkillHTTPError(f"'{url}' does not look like a valid http(s) URL.")

        verdicts: list[SourceVerdict] = []

        # 1. lrl.kr / Google Safe Browsing cache.
        if self._settings.lrl_api_key:
            verdicts.append(await check_lrl(self._settings.lrl_api_key, url))
        else:
            verdicts.append(SourceVerdict("lrl.kr", "skipped", "no LRL_API_KEY"))

        # 2. Cloudflare 1.1.1.2 DNS blocklist (key-less).
        verdicts.append(await check_cloudflare_dns(url))

        # 3. Optional live scan via Cloudflare Radar.
        if self._settings.cloudflare_api_key and self._settings.cloudflare_account_id:
            verdicts.append(
                await scan_cloudflare_radar(
                    self._settings.cloudflare_api_key,
                    self._settings.cloudflare_account_id,
                    url,
                )
            )
        else:
            verdicts.append(SourceVerdict("Cloudflare Radar", "skipped", "no API token configured"))

        overall, reasons = merge_verdicts(verdicts)

        lines = [f"URL safety check for {url}", f"VERDICT: {overall}"]
        for verdict in verdicts:
            suffix = "" if verdict.status != "skipped" else " (skipped)"
            detail = f" - {verdict.detail}" if verdict.detail else ""
            lines.append(f"- {verdict.source}{suffix}{detail}")
        lines.extend(f"Reason: {reason}" for reason in reasons[:3])
        return "\n".join(lines)
