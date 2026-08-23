"""Thin wrapper around the OpenAI SDK.

The bot never talks to the `openai` package directly; it goes through
:class:`LLMService` so that:

* the provider is configured in exactly one place (``Settings``), and
* switching to any OpenAI-compatible server (OpenRouter, Ollama, vLLM,
  Azure-style gateways, ...) is a pure configuration change.
"""

from __future__ import annotations

from collections.abc import Iterable

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionToolParam

from chord.config import Settings


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
        return await self._client.chat.completions.create(**kwargs)
