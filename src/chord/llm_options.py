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
    """Provider extras to send with every request, from settings.

    ``LLM_SAFETY_FILTERS=off`` produces nothing. It used to emit the
    payload Google documents, and that payload is rejected: the
    OpenAI-compatible endpoint answers

        Unknown name "safety_settings" at 'extra_body.google'

    for gemma-4-31b-it and gemini-2.5-flash alike, while
    ``thinking_config`` in the same object is accepted - so the nesting
    is right and the field simply is not there. Sending it bought one
    guaranteed 400 per process and no change in behaviour.

    The knob stays because the *question* is real and the answer may
    change; what it does now is say where refusals actually come from.
    ``LLM_EXTRA_BODY`` remains the way to send whatever a provider does
    accept.
    """
    if settings.llm_safety_filters == "off":
        logger.warning(
            "LLM_SAFETY_FILTERS=off has no effect: no OpenAI-compatible "
            "endpoint chord can reach exposes a content-filter threshold "
            "(Google's rejects safety_settings outright). Refusals come "
            "from the model's own training and from the Boundaries "
            "section of persona.md - persona.md is the one you can edit."
        )
    return merge_extra_body({}, settings.extra_body)
