"""Tests for the URL safety-check skill (all sources mocked)."""

from __future__ import annotations

import pytest
import respx

import quota_helpers  # noqa: F401  (conftest already isolates the store)
from chord.config import Settings
from chord.skills._http import SkillHTTPError
from chord.skills.url_safety import (
    CLOUDFLARE_DOH_URL,
    LRL_CHECK_URL,
    RADAR_SCANS_URL,
    CheckUrlSafetySkill,
    SourceVerdict,
    check_cloudflare_dns,
    check_lrl,
    extract_domain,
    merge_verdicts,
    threat_label,
)


def _settings(**keys) -> Settings:
    defaults = {"lrl_api_key": "lrl-key"}
    defaults.update(keys)
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        discord_token="t",
        openai_api_key="k",
        **defaults,
    )


def _lrl_safe():
    return {"result": {"safe": "1"}, "message": "SUCCESS"}


def _lrl_malware():
    return {"result": {"safe": "0", "threat": "MALWARE"}, "message": "SUCCESS"}


def _doh_normal(domain="example.com"):
    return {
        "Status": 0,
        "Answer": [{"name": domain, "type": 1, "data": "93.184.216.34"}],
    }


def _doh_blocked(domain="evil.test"):
    return {"Status": 0, "Answer": [{"name": domain, "type": 1, "data": "0.0.0.0"}]}


# -- Sources ------------------------------------------------------------------------


@respx.mock
async def test_lrl_detects_malware():
    respx.get(LRL_CHECK_URL).respond(status_code=201, json=_lrl_malware())

    verdict = await check_lrl("key", "http://x.test/page")

    assert verdict.status == "unsafe"
    assert verdict.threats == ["MALWARE"]
    assert verdict.detail == "malware detected"


@respx.mock
async def test_lrl_error_message_passthrough():
    respx.get(LRL_CHECK_URL).respond(status_code=400, json={"message": "ERR_NO_URL"})

    verdict = await check_lrl("key", "")

    assert verdict.status == "unknown"
    assert "ERR_NO_URL" in verdict.detail


@respx.mock
async def test_cloudflare_dns_detects_block():
    respx.get(CLOUDFLARE_DOH_URL).respond(json=_doh_blocked("bad.test"))

    verdict = await check_cloudflare_dns("https://bad.test/x")

    assert verdict.status == "unsafe"
    assert "blocked as malicious" in verdict.detail


# -- Full-skill behavior ---------------------------------------------------------------


@respx.mock
async def test_full_unsafe_verdict_from_lrl_only():
    respx.get(LRL_CHECK_URL).respond(status_code=201, json=_lrl_malware())
    respx.get(CLOUDFLARE_DOH_URL).respond(json=_doh_normal())

    result = await CheckUrlSafetySkill(_settings()).run(url="http://x.test/page")

    assert "VERDICT: UNSAFE" in result
    assert "malware detected" in result
    assert "Cloudflare Radar (skipped)" in result


@respx.mock
async def test_safe_when_all_sources_clear():
    respx.get(LRL_CHECK_URL).respond(status_code=201, json=_lrl_safe())
    respx.get(CLOUDFLARE_DOH_URL).respond(json=_doh_normal())

    result = await CheckUrlSafetySkill(_settings()).run(url="https://example.com")

    assert "VERDICT: SAFE" in result
    assert "cleared by lrl.kr, Cloudflare 1.1.1.2" in result


@respx.mock
async def test_cloudflare_dns_block_overrides_lrl_clear():
    respx.get(LRL_CHECK_URL).respond(status_code=201, json=_lrl_safe())
    respx.get(CLOUDFLARE_DOH_URL).respond(json=_doh_blocked("bad.test"))

    result = await CheckUrlSafetySkill(_settings()).run(url="https://bad.test/x")

    assert "VERDICT: UNSAFE" in result
    assert "blocked as malicious" in result


@respx.mock
async def test_missing_lrl_key_skips_source_without_failing():
    with respx.mock:
        respx.get(CLOUDFLARE_DOH_URL).respond(json=_doh_normal())

        settings = _settings(lrl_api_key="")
        result = await CheckUrlSafetySkill(settings).run(url="https://ok.test")

    assert "VERDICT: SAFE" in result
    assert "lrl.kr (skipped) - no LRL_API_KEY" in result


@respx.mock
async def test_radar_configured_and_flags_malicious():
    respx.get(LRL_CHECK_URL).respond(status_code=201, json=_lrl_safe())
    respx.get(CLOUDFLARE_DOH_URL).respond(json=_doh_normal())
    respx.post(RADAR_SCANS_URL.format(account="acc123")).respond(
        json={"result": {"uuid": "scan-1"}}
    )
    respx.get(RADAR_SCANS_URL.format(account="acc123") + "/scan-1").respond(
        json={"result": {"status": "completed", "verdicts": {"malicious": True}}}
    )

    settings = _settings(cloudflare_api_key="cf", cloudflare_account_id="acc123")
    result = await CheckUrlSafetySkill(settings).run(url="https://maybe.test")

    assert "VERDICT: UNSAFE" in result
    assert "live scan flagged this URL" in result


async def test_invalid_url_fails_fast():
    with pytest.raises(SkillHTTPError, match="does not look like"):
        await CheckUrlSafetySkill(_settings()).run(url="not-a-url")


# -- Helpers --------------------------------------------------------------------------


def test_threat_labels():
    assert threat_label("SOCIAL_ENGINEERING") == "phishing / social engineering"
    assert threat_label("") == ""
    assert threat_label("weird_new") == "weird new"


def test_extract_domain():
    assert extract_domain("https://Example.COM/a?b=1") == "example.com"


def test_merge_prefers_unsafe_then_safe_then_unknown():
    unsafe = SourceVerdict("a", "unsafe", "bad")
    safe = SourceVerdict("b", "safe")
    unknown = SourceVerdict("c", "unknown")

    overall, _ = merge_verdicts([unknown])
    assert overall == "UNKNOWN"

    overall, reasons = merge_verdicts([safe, unknown])
    assert overall == "SAFE"
    assert reasons == ["cleared by b"]

    overall, reasons = merge_verdicts([safe, unsafe, unknown])
    assert overall == "UNSAFE" and reasons == ["a: bad"]
