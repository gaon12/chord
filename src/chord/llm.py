"""Thin wrapper around the OpenAI SDK.

The bot never talks to the `openai` package directly; it goes through
:class:`LLMService` so that:

* the provider is configured in exactly one place (``Settings``), and
* switching to any OpenAI-compatible server (OpenRouter, Ollama, vLLM,
  Azure-style gateways, ...) is a pure configuration change.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable

from openai import AsyncOpenAI, BadRequestError, RateLimitError
from openai.types.chat import ChatCompletion, ChatCompletionToolParam

from chord.config import Settings
from chord.groq_provider import GROQ_FREE_MODELS, GroqProvider, groq_reasoning_effort
from chord.llm_options import (
    DEFAULT_RATE_LIMIT_WAIT,
    MAX_RATE_LIMIT_WAIT,
    build_extra_body,
    is_extra_body_rejection,
    is_reasoning_rejection,
    rate_limit_delay,
)

logger = logging.getLogger(__name__)


def create_client(settings: Settings) -> AsyncOpenAI:
    """Build an async OpenAI client from application settings.

    ``base_url`` is what makes the bot provider-agnostic: every
    OpenAI-compatible endpoint accepts the same wire protocol.
    """
    return AsyncOpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        # The Groq router owns retries; SDK retries would delay model fallback.
        **({"max_retries": 0} if settings.use_groq else {}),
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
        self._groq = (
            GroqProvider(
                self._client,
                GROQ_FREE_MODELS if settings.groq_free_tier else (settings.groq_model,),
                settings.groq_max_completion_tokens,
            )
            if settings.use_groq
            else None
        )
        self._model = settings.openai_model
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
        if self._groq is not None:
            return groq_reasoning_effort(self.model, self.reasoning_effort) is not None
        return self.reasoning_effort is not None and self._reasoning_supported

    @property
    def model(self) -> str:
        return self._groq.model if self._groq is not None else self._model

    def describe_usage(self) -> str:
        return self._groq.describe_usage() if self._groq is not None else ""

    def describe_reasoning(self) -> str:
        return (
            self._groq.describe_reasoning(self.reasoning_effort) if self._groq is not None else ""
        )

    async def aclose(self) -> None:
        await self._client.close()

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
        if self._groq is not None:
            return await self._groq.complete(
                messages, tools, self.reasoning_effort, self._extra_body
            )

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
                "Model %s rejected reasoning_effort=%s; "
                "continuing without it for the rest of this run.",
                self.model,
                self.reasoning_effort,
            )
            return True

        if "extra_body" in kwargs and is_extra_body_rejection(error):
            kwargs.pop("extra_body")
            self._extra_body_supported = False
            logger.warning(
                "Provider rejected the extra request body; continuing "
                "without it. Check LLM_SAFETY_FILTERS / LLM_EXTRA_BODY - "
                "these fields are provider-specific.",
            )
            return True

        return False
