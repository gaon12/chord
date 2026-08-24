"""Thin wrapper around the OpenAI SDK.

The bot never talks to the `openai` package directly; it goes through
:class:`LLMService` so that:

* the provider is configured in exactly one place (``Settings``), and
* switching to any OpenAI-compatible server (OpenRouter, Ollama, vLLM,
  Azure-style gateways, ...) is a pure configuration change.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

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


def is_reasoning_rejection(message: str) -> bool:
    """True when an error message blames the reasoning parameter."""
    lowered = message.lower()
    return any(marker in lowered for marker in _REASONING_REJECTION_MARKERS)


def create_client(settings: Settings) -> AsyncOpenAI:
    """Build an async OpenAI client from application settings.

    ``base_url`` is what makes the bot provider-agnostic: every
    OpenAI-compatible endpoint accepts the same wire protocol.
    """
    return AsyncOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
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

        if not self.reasoning_enabled:
            return await self._client.chat.completions.create(**kwargs)

        try:
            return await self._client.chat.completions.create(
                reasoning_effort=self.reasoning_effort, **kwargs
            )
        except BadRequestError as exc:
            if not is_reasoning_rejection(str(exc)):
                raise
            # Plenty of OpenAI-compatible models simply have no notion of
            # reasoning effort. Losing the knob is fine; losing the reply
            # is not - so drop the parameter and answer anyway.
            self._reasoning_supported = False
            logger.warning(
                "Model %s rejected reasoning_effort=%s (%s); "
                "continuing without it for the rest of this run.",
                self.model,
                self.reasoning_effort,
                exc,
            )
            return await self._client.chat.completions.create(**kwargs)
