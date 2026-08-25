"""Thin wrapper around the OpenAI SDK.

The bot never talks to the `openai` package directly; it goes through
:class:`LLMService` so that:

* the provider is configured in exactly one place (``Settings``), and
* switching to any OpenAI-compatible server (OpenRouter, Ollama, vLLM,
  Azure-style gateways, ...) is a pure configuration change.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Iterable
from typing import Any

from openai import AsyncOpenAI, BadRequestError
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

#: Gemini's harm categories. Every one has to be listed explicitly - a
#: category left out keeps the provider default, so a partial list reads
#: as "off" but behaves as "mostly on".
GEMINI_HARM_CATEGORIES = (
    "HARM_CATEGORY_HARASSMENT",
    "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "HARM_CATEGORY_DANGEROUS_CONTENT",
    "HARM_CATEGORY_CIVIC_INTEGRITY",
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
    """Gemini safety thresholds, wrapped the way the compat layer wants.

    The OpenAI-compatible endpoint tunnels Gemini-only request fields
    through a nested ``extra_body.google`` object - the outer key really
    is named ``extra_body`` again, which looks like a typo and is not.

    ``BLOCK_NONE`` turns off *probability-based blocking* by the API's
    filter. It does not touch what the model itself was trained to
    decline, which is where most surprising refusals actually come from.
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

    Two sources, merged: the ``LLM_SAFETY_FILTERS`` shortcut and the raw
    ``LLM_EXTRA_BODY`` escape hatch, with the raw JSON winning so an
    operator can always override what the shortcut produced.
    """
    body: dict[str, Any] = {}
    if settings.llm_safety_filters == "off":
        if is_google_endpoint(settings.openai_base_url):
            body = gemini_safety_settings()
        else:
            logger.warning(
                "LLM_SAFETY_FILTERS=off has no effect on %s - only the Gemini "
                "API exposes a filter threshold over the OpenAI-compatible "
                "wire format. Refusals from other providers come from the "
                "model's own training and from persona.md, not from a knob.",
                settings.openai_base_url,
            )
    return merge_extra_body(body, settings.extra_body)


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

        while True:
            try:
                return await self._client.chat.completions.create(**kwargs)
            except BadRequestError as exc:
                # Optional knobs are worth one 400 each, never the reply:
                # drop whichever one the provider named and try again.
                # Each drop removes a key for good, so this ends.
                if not self._drop_rejected_option(str(exc), kwargs):
                    raise

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
