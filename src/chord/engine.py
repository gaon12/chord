"""The chat engine - runs one full user turn, including tool calling.

Flow of ``reply()``:

1. Build the message list: system prompt + channel history + new user text.
2. Ask the LLM for a completion.
3. If the model wants tools, run each requested skill, feed results back,
   and ask again (up to ``max_tool_rounds`` times).
4. Return the final text plus every generated message so the caller can
   update the stored history.
"""

from __future__ import annotations

import logging
from typing import Any

from openai.types.chat import (
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
)

from chord.llm import LLMService
from chord.skills.registry import SkillRegistry
from chord.tool_catalog import SELECTION_RULE, ToolCatalog

logger = logging.getLogger(__name__)

#: Safety valve so a confused model cannot loop forever.
DEFAULT_MAX_TOOL_ROUNDS = 6


class ChatEngine:
    """Turns one user message into one assistant answer."""

    def __init__(
        self,
        llm: LLMService,
        registry: SkillRegistry,
        system_prompt: str,
        max_tool_rounds: int = DEFAULT_MAX_TOOL_ROUNDS,
        on_demand_tools: bool = False,
    ) -> None:
        self._llm = llm
        self._registry = registry
        # Public and mutable so callers can swap the persona at runtime.
        self.system_prompt = system_prompt
        self._max_tool_rounds = max_tool_rounds
        self._on_demand_tools = on_demand_tools

    @property
    def llm(self) -> LLMService:
        """The underlying LLM service, exposed so /reasoning can retune it."""
        return self._llm

    async def reply(
        self,
        user_text: str,
        history: list[dict],
        *,
        system_prompt: str | None = None,
    ) -> tuple[str, list[dict]]:
        """Generate an answer for one user message.

        Args:
            user_text: What the user just said.
            history: Previous messages for this channel (oldest first).
            system_prompt: Optional prompt snapshot for this turn only. It
                never changes the shared engine's default prompt.

        Returns:
            ``(answer_text, new_messages)`` where ``new_messages`` holds
            everything that should be remembered from this turn (the
            user message, intermediate tool traffic and the answer).
        """
        catalog = ToolCatalog(self._registry, self._on_demand_tools)
        prompt = self.system_prompt if system_prompt is None else system_prompt
        if self._on_demand_tools:
            prompt += "\n\n" + SELECTION_RULE
        messages: list[dict] = [
            {"role": "system", "content": prompt},
            *history,
            {"role": "user", "content": user_text},
        ]
        # This turn owns everything from the new user message onwards.
        # The system prompt and the caller's own history must NOT be
        # returned: the caller already holds the history and appends what
        # comes back, so re-returning it duplicates the whole
        # conversation on every single turn.
        turn_start = len(messages) - 1

        for _round in range(self._max_tool_rounds):
            completion = await self._llm.complete(messages, catalog.definitions())
            log_token_usage(completion)
            message = completion.choices[0].message
            if completion.choices[0].finish_reason == "length":
                # A truncated tool call may have incomplete arguments. Never
                # execute it or store it as a valid assistant/tool exchange.
                note = (
                    "The model reached its output limit. Try a shorter request "
                    "or ask the operator to increase the completion token limit."
                )
                answer = f"{message.content}\n\n{note}" if message.content else note
                messages.append({"role": "assistant", "content": answer})
                return answer, messages[turn_start:]
            messages.append(_serialize_message(message))

            if not message.tool_calls:
                # Plain answer - the turn is done.
                content = message.content or ""
                return content, messages[turn_start:]

            logger.info("Model requested %d tool call(s)", len(message.tool_calls))
            results = await catalog.execute_calls(message.tool_calls)
            for call, result in zip(message.tool_calls, results, strict=True):
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": result,
                    }
                )

        logger.warning("Tool-call loop hit the %d-round limit", self._max_tool_rounds)
        fallback = "I got stuck going back and forth between tools - please try again."
        return fallback, [*messages[turn_start:], {"role": "assistant", "content": fallback}]


def log_token_usage(completion: Any) -> None:
    """Log what one request cost, so rate limits stop being a mystery.

    Input tokens are the scarce resource: providers meter them per
    minute, the tool catalog is re-sent on every round, and a turn that
    calls tools spends this several times over. Logged at DEBUG, and
    tolerant of providers that omit ``usage`` entirely.
    """
    usage = getattr(completion, "usage", None)
    if usage is None:
        return
    logger.debug(
        "Tokens: prompt=%s completion=%s total=%s",
        getattr(usage, "prompt_tokens", "?"),
        getattr(usage, "completion_tokens", "?"),
        getattr(usage, "total_tokens", "?"),
    )


def _serialize_message(message: ChatCompletionMessage) -> dict[str, Any]:
    """Convert the SDK message object into a plain OpenAI-format dict.

    Hand-built instead of ``model_dump()`` so we send exactly the fields
    chat APIs accept - different providers choke on extra keys like
    ``refusal`` or ``audio``.
    """
    serialized: dict[str, Any] = {
        "role": "assistant",
        "content": message.content,
    }
    if message.tool_calls:
        serialized["tool_calls"] = [_serialize_tool_call(call) for call in message.tool_calls]
    return serialized


def _serialize_tool_call(call: ChatCompletionMessageToolCall) -> dict[str, Any]:
    return {
        "id": call.id,
        "type": "function",
        "function": {"name": call.function.name, "arguments": call.function.arguments},
    }
