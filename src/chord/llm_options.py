"""Compatibility helpers for optional OpenAI-style request fields."""

from __future__ import annotations

import copy
import logging
import math
import re
from typing import Any

from chord.config import Settings

logger = logging.getLogger(__name__)

#: Substrings that identify a 400 caused by ``reasoning_effort`` rather
#: than by something else in the request (a bad tool schema, say).
#: Seen in the wild: "Thinking budget is not supported for this model",
#: "Thinking level is not supported for this model" (Gemini/Gemma),
#: "Unrecognized request argument supplied: reasoning_effort" (OpenAI).
_REASONING_REJECTION_MARKERS = (
    "reasoning_effort",
    "reasoning effort",
    "thinking budget",
    "thinking level",
    "thinking config",
    "thinking_config",
)


#: Substrings that identify a 400 caused by the provider extras we send
#: in ``extra_body`` rather than by the conversation itself.
_EXTRA_BODY_REJECTION_MARKERS = (
    "extra_body",
    "extrabody",
    "safety_settings",
    "safetysettings",
    "safety setting",
)

#: Gemini's harm categories, kept for whoever wires them up through
#: LLM_EXTRA_BODY against an endpoint that accepts them.
GEMINI_HARM_CATEGORIES = (
    "HARM_CATEGORY_HARASSMENT",
    "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "HARM_CATEGORY_DANGEROUS_CONTENT",
    "HARM_CATEGORY_CIVIC_INTEGRITY",
)


#: How long to wait out a rate limit before answering anyway. The SDK's
#: own retry ladder starts under a second, which is right for a
#: per-second limit and useless against a per-minute token quota - the
#: window it needs to clear is sixty seconds wide. Capped so a chat turn
#: cannot hang: past this, saying "try again shortly" is the better
#: answer than a reply nobody is still waiting for.
MAX_RATE_LIMIT_WAIT = 45.0

#: What to wait when the provider does not say. Long enough to cross
#: most of a per-minute window, short enough to still be a conversation.
DEFAULT_RATE_LIMIT_WAIT = 20.0

#: Gemini reports its own backoff inside the error body rather than in a
#: Retry-After header: "retryDelay": "38s".
_RETRY_DELAY_RE = re.compile(r"retry[-_ ]?delay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s?", re.I)


def error_diagnostics(error: Exception) -> str:
    """Log bounded error metadata without provider text, requests, or credentials."""
    body = getattr(error, "body", None)
    details = body.get("error", body) if isinstance(body, dict) else {}
    if not isinstance(details, dict):
        details = {}
    code = details.get("code")
    known_codes = (
        "rate_limit_exceeded",
        "model_not_found",
        "model_decommissioned",
        "tool_use_failed",
        "tool_call_validation_failed",
        "json_validate_failed",
        "context_length_exceeded",
        "invalid_request_error",
        "invalid_api_key",
        "insufficient_quota",
    )
    # Unknown fields may contain arbitrary provider data. Never echo them.
    code = code if code in known_codes else "unknown"
    message = details.get("message")
    message = message.lower() if isinstance(message, str) else ""
    quota = "unknown"
    for name, marker in (
        ("ITPM", "input tokens per minute"),
        ("OTPM", "output tokens per minute"),
        ("TPM", "tokens per minute"),
        ("TPD", "tokens per day"),
        ("RPM", "requests per minute"),
        ("RPD", "requests per day"),
    ):
        if marker in message:
            quota = name
            break
    status = getattr(error, "status_code", None)
    status = status if isinstance(status, int) else "unknown"
    return f"error={type(error).__name__} status={status} code={code} quota={quota}"


def rate_limit_delay(error: Exception) -> float | None:
    """How long the provider asked us to wait, if it said.

    Checks the Retry-After header first, then the retryDelay the Gemini
    API puts in the error body. Returns None when neither is present, so
    the caller can pick its own wait rather than inventing one here.
    """
    response = getattr(error, "response", None)
    header = getattr(response, "headers", None)
    if header is not None:
        raw = header.get("retry-after") or header.get("Retry-After")
        if raw:
            try:
                value = float(raw)
                if math.isfinite(value) and value >= 0:
                    return value
            except (TypeError, ValueError):
                pass

    match = _RETRY_DELAY_RE.search(str(error))
    value = float(match.group(1)) if match else None
    return value if value is not None and math.isfinite(value) else None


def is_history_rejection(message: str) -> bool:
    """Only clear history when the provider identifies an invalid tool exchange."""
    lowered = message.lower()
    return any(
        marker in lowered
        for marker in (
            "tool_call_id",
            "must be followed by",
            "must be a response to",
            "function_response.name",
            "messages with role 'tool'",
        )
    )


def is_reasoning_rejection(message: str) -> bool:
    """True when an error message blames the reasoning parameter."""
    lowered = message.lower()
    return any(marker in lowered for marker in _REASONING_REJECTION_MARKERS)


def is_extra_body_rejection(message: str) -> bool:
    """True when an error message blames our provider-extras payload."""
    lowered = message.lower()
    return any(marker in lowered for marker in _EXTRA_BODY_REJECTION_MARKERS)


def is_google_endpoint(base_url: str) -> bool:
    """Whether the base URL points at Gemini's OpenAI-compatible API."""
    return "generativelanguage.googleapis.com" in (base_url or "").lower()


def gemini_safety_settings(threshold: str = "BLOCK_NONE") -> dict[str, Any]:
    """The payload Google documents for lowering its safety filter.

    Kept because it is the correct shape *per the documentation*, and
    because ``LLM_EXTRA_BODY`` is how anyone would send it - but chord
    no longer sends it on its own. See :func:`build_extra_body`.
    """
    return {
        "extra_body": {
            "google": {
                "safety_settings": [
                    {"category": category, "threshold": threshold}
                    for category in GEMINI_HARM_CATEGORIES
                ]
            }
        }
    }


def merge_extra_body(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge two extra_body payloads; ``override`` wins on conflict.

    Shallow merging would be a trap here: everything Gemini-specific
    lives under the same ``extra_body.google`` key, so a hand-written
    thinking_config would silently drop the safety settings next to it.
    """
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_extra_body(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def build_extra_body(settings: Settings) -> dict[str, Any]:
    """Return explicitly configured extras without inventing safety overrides.

    There is no application content-moderation call in this request path.
    LLM_SAFETY_FILTERS is kept for configuration compatibility, but a local
    preference cannot change a model's training or undocumented server policy.
    Unsupported filter fields must not cause a 400 before every answer.
    """
    if settings.llm_safety_filters == "off":
        logger.info(
            "No application content moderation is enabled. "
            "LLM_SAFETY_FILTERS=off does not override provider or model policies."
        )
    return merge_extra_body({}, settings.extra_body)
