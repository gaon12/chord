"""Header units, reset parsing, and stale observations must remain explicit."""

import pytest

from chord.groq_limits import GroqLimits, duration_seconds


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2m59.56s", 179.56),
        ("1h2m3s", 3723),
        ("7.66s", 7.66),
        ("500ms", 0.5),
        ("2", 2),
        (None, None),
        ("bad", None),
        ("-1", None),
        ("nan", None),
        ("inf", None),
        ("2s junk", None),
    ],
)
def test_duration_parser(raw, expected):
    assert duration_seconds(raw) == expected


def test_headers_use_daily_requests_and_minute_tokens():
    limits = GroqLimits()
    limits.update(
        {
            "X-RateLimit-Limit-Requests": "1000",
            "X-RateLimit-Remaining-Requests": "0",
            "X-RateLimit-Reset-Requests": "2h",
            "X-RateLimit-Limit-Tokens": "8000",
            "X-RateLimit-Remaining-Tokens": "10",
            "X-RateLimit-Reset-Tokens": "5s",
        },
        now=100,
    )
    assert not limits.ready(now=106)
    assert limits.ready(now=7300)
    text = limits.describe(now=101)
    assert "RPD: 0/1000" in text
    assert "TPM: 10/8000" in text
    assert "unknown" in limits.describe(now=7301)


def test_token_reset_does_not_hold_a_model_until_the_daily_reset():
    limits = GroqLimits()
    limits.update(
        {
            "x-ratelimit-remaining-requests": "12",
            "x-ratelimit-reset-requests": "2h",
            "x-ratelimit-remaining-tokens": "0",
            "x-ratelimit-reset-tokens": "5s",
        },
        now=100,
    )
    assert not limits.ready(now=104)
    assert limits.ready(now=105)


def test_missing_or_invalid_headers_do_not_invent_limits():
    limits = GroqLimits()
    limits.update({"x-ratelimit-remaining-tokens": "NaN"}, now=100)
    assert limits.tokens_remaining is None
    assert limits.ready(now=100)
    assert "unknown" in limits.describe(now=100)


def test_retry_after_is_respected_when_no_bucket_is_reported():
    limits = GroqLimits()
    limits.mark_rate_limited({"retry-after": "15"}, now=100)
    assert not limits.ready(now=114)
    assert limits.ready(now=115)
    assert limits.requests_remaining is None


def test_unreported_limit_uses_a_bounded_cooldown():
    limits = GroqLimits()
    limits.mark_rate_limited({}, now=100)
    assert not limits.ready(now=159)
    assert limits.ready(now=160)
