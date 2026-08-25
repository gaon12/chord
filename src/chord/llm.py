"""Thin wrapper around the OpenAI SDK.

The bot never talks to the `openai` package directly; it goes through
:class:`LLMService` so that:

* the provider is configured in exactly one place (``Settings``), and
* switching to any OpenAI-compatible server (OpenRouter, Ollama, vLLM,
  Azure-style gateways, ...) is a pure configuration change.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import re
from collections.abc import Iterable
from typing import Any

from openai import AsyncOpenAI, BadRequestError, RateLimitError
from openai.types.chat import ChatCompletion, ChatCompletionToolParam

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
                return float(raw)
            except (TypeError, ValueError):
                pass

    match = _RETRY_DELAY_RE.search(str(error))
    return float(match.group(1)) if match else None


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


def create_client(settings: Settings) -> AsyncOpenAI:
    """Build an async OpenAI client from application settings.

    ``base_url`` is what makes the bot provider-agnostic: every
    OpenAI-compatible endpoint accepts the same wire protocol.
    """
    return AsyncOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        # Without an explicit timeout the SDK waits 10 minutes per try;
        # a chat bot that answers 10 minutes late has already failed.
        timeout=settings.llm_timeout_seconds,
    )


class LLMService:
    """Chat-completion facade over an OpenAI-compatible API."""

    def __init__(self, settings: Settings, client: AsyncOpenAI | None = None) -> None:
        # ``client`` is injectable mainly for testing: tests pass a fake
        # instead of hitting a real API.
        self._client = client if client is not None else create_client(settings)
        self.model = settings.openai_model
        # Public so /reasoning can retune a running bot; None = send nothing.
        self.reasoning_effort = settings.reasoning_effort
        #: Flipped to False the first time the provider rejects the
        #: parameter, so an incompatible model costs one 400 per process
        #: instead of one per message.
        self._reasoning_supported = True
        #: Provider-specific request extras (safety thresholds and the
        #: LLM_EXTRA_BODY escape hatch), with the same one-400 rule.
        self._extra_body = build_extra_body(settings)
        self._extra_body_supported = True

    def set_reasoning_effort(self, effort: str | None) -> None:
        """Change the effort at runtime, re-enabling the parameter.

        A deliberate change deserves a fresh attempt even if the model
        rejected the parameter earlier: the operator may have swapped
        models, or may simply want to see the rejection again.
        """
        self.reasoning_effort = effort
        self._reasoning_supported = True

    @property
    def reasoning_enabled(self) -> bool:
        """Whether the next request will carry ``reasoning_effort``."""
        return self.reasoning_effort is not None and self._reasoning_supported

    async def complete(
        self,
        messages: Iterable[dict],
        tools: list[ChatCompletionToolParam] | None = None,
    ) -> ChatCompletion:
        """Send a chat completion request and return the raw response.

        Args:
            messages: OpenAI-format message dicts (system/user/assistant/
                tool). The caller owns the conversation history.
            tools: Optional list of tool definitions in OpenAI format.
                Both built-in skills and MCP tools are passed here.

        Returns:
            The full :class:`ChatCompletion` so callers can inspect text,
            tool calls or usage freely.
        """
        kwargs: dict = {
            "model": self.model,
            "messages": list(messages),
        }
        if tools:
            # "auto" lets the model decide when a tool is useful; plain
            # questions are answered directly without forced tool use.
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if self.reasoning_enabled:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if self._extra_body and self._extra_body_supported:
            kwargs["extra_body"] = self._extra_body

        waited_out_a_limit = False
        while True:
            try:
                return await self._client.chat.completions.create(**kwargs)
            except BadRequestError as exc:
                # Optional knobs are worth one 400 each, never the reply:
                # drop whichever one the provider named and try again.
                # Each drop removes a key for good, so this ends.
                if not self._drop_rejected_option(str(exc), kwargs):
                    raise
            except RateLimitError as exc:
                # Once. A second wait would mean two minutes of a chat
                # turn spent on a quota that is clearly not ours today.
                if waited_out_a_limit:
                    raise
                waited_out_a_limit = True
                await self._wait_out_rate_limit(exc)

    async def _wait_out_rate_limit(self, error: RateLimitError) -> None:
        """Sleep long enough for a per-minute quota to reopen.

        The SDK already retried on its own ladder - roughly 0.4s then
        0.9s - and those are the right numbers for a per-second limit
        and the wrong ones for a token-per-minute quota, which is what a
        large tool catalog actually runs into. The window is sixty
        seconds wide; sub-second retries just spend the attempts.
        """
        asked = rate_limit_delay(error)
        wait = min(asked if asked is not None else DEFAULT_RATE_LIMIT_WAIT, MAX_RATE_LIMIT_WAIT)
        logger.warning(
            "Rate limited by %s; waiting %.0fs before one more try%s.",
            self.model,
            wait,
            " (provider asked for it)" if asked is not None else "",
        )
        await asyncio.sleep(wait)

    def _drop_rejected_option(self, error: str, kwargs: dict) -> bool:
        """Remove the optional parameter a 400 blamed; False if none did.

        Both knobs are conveniences that plenty of OpenAI-compatible
        servers have never heard of. Answering without them beats not
        answering, and remembering the rejection keeps an incompatible
        provider at one 400 per process instead of one per message.
        """
        if "reasoning_effort" in kwargs and is_reasoning_rejection(error):
            kwargs.pop("reasoning_effort")
            self._reasoning_supported = False
            logger.warning(
                "Model %s rejected reasoning_effort=%s (%s); "
                "continuing without it for the rest of this run.",
                self.model,
                self.reasoning_effort,
                error,
            )
            return True

        if "extra_body" in kwargs and is_extra_body_rejection(error):
            kwargs.pop("extra_body")
            self._extra_body_supported = False
            logger.warning(
                "Provider rejected the extra request body (%s); continuing "
                "without it. Check LLM_SAFETY_FILTERS / LLM_EXTRA_BODY - "
                "these fields are provider-specific.",
                error,
            )
            return True

        return False
