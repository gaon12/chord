"""Groq completions with ordered free-model fallback and header observations."""

from __future__ import annotations

import asyncio
import copy
import inspect
import logging
import time
from collections.abc import Iterable

import httpx
from openai import APIStatusError, AsyncOpenAI, BadRequestError, NotFoundError, RateLimitError
from openai.types.chat import ChatCompletion, ChatCompletionToolParam

from chord.groq_limits import GroqLimits
from chord.token_estimates import estimate_request_tokens

logger = logging.getLogger(__name__)

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_FREE_MODELS = (
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
)


def groq_reasoning_effort(model: str, effort: str | None) -> str | None:
    """Translate the application setting into values the selected model accepts.

    None means omit the parameter: it never sends the unsupported literal
    'auto'. GPT-OSS cannot disable reasoning; low is its minimum effort.
    """
    if effort is None:
        return None
    if model in GROQ_FREE_MODELS:
        if effort == "minimal":
            return "none" if model == GROQ_FREE_MODELS[0] else "low"
        return effort
    # A paid custom model may not support reasoning at all. Do not guess its
    # API vocabulary; leave it to the provider's model default.
    return None


class GroqProvider:
    def __init__(self, client: AsyncOpenAI, models: tuple[str, ...], max_tokens: int) -> None:
        self._client = client
        self.models = models
        self.model = models[0]
        self.max_tokens = max_tokens
        self.limits = {model: GroqLimits() for model in models}
        self._lock = asyncio.Lock()
        self._unavailable: set[str] = set()

    def describe_usage(self) -> str:
        lines = ["Groq (last observed headers; shared organization quotas):"]
        lines.extend(f"- {model}: {self.limits[model].describe()}" for model in self.models)
        lines.append("RPM and TPD remaining are not supplied by these headers.")
        return "\n".join(lines)

    def describe_reasoning(self, effort: str | None) -> str:
        effective = groq_reasoning_effort(self.model, effort)
        value = effective if effective is not None else "provider default (parameter omitted)"
        return f"Groq model: {self.model}; effective reasoning: {value}."

    async def complete(
        self,
        messages: Iterable[dict],
        tools: list[ChatCompletionToolParam] | None,
        effort: str | None,
        extra_body: dict,
    ) -> ChatCompletion:
        # Chat, compaction, and text skills share this provider. Serializing
        # requests prevents them from racing on a model's last header snapshot.
        async with self._lock:
            return await self._complete(list(messages), tools, effort, extra_body)

    async def _complete(self, messages, tools, effort, extra_body) -> ChatCompletion:
        last_error: Exception | None = None
        required_tokens = estimate_request_tokens(messages, tools, self.max_tokens)
        for model in self.models:
            limits = self.limits[model]
            if not limits.ready(required_tokens=required_tokens):
                logger.info(
                    "Skipping Groq model %s: cooldown or insufficient observed capacity "
                    "for an estimated %d-token request.",
                    model,
                    required_tokens,
                )
                continue
            kwargs = {
                "model": model,
                "messages": messages,
                "max_completion_tokens": self.max_tokens,
            }
            if tools:
                kwargs.update(tools=tools, tool_choice="auto")
            reasoning = groq_reasoning_effort(model, effort)
            if reasoning is not None:
                kwargs["reasoning_effort"] = reasoning
            body = copy.deepcopy(extra_body)
            # Keep reasoning out of visible content for the documented models.
            # Reject reserved extra-body keys before reaching this point.
            if model in GROQ_FREE_MODELS:
                body["reasoning_format"] = "hidden"
            if body:
                kwargs["extra_body"] = body
            try:
                raw = await self._client.chat.completions.with_raw_response.create(**kwargs)
                limits.update(raw.headers)
                completion = raw.parse()
                if inspect.isawaitable(completion):
                    completion = await completion
                self.model = model
                self._unavailable.discard(model)
                return completion
            except RateLimitError as exc:
                limits.mark_rate_limited(exc.response.headers)
                logger.info(
                    "Groq model %s is rate limited; checking the next configured model.", model
                )
                last_error = exc
            except (BadRequestError, NotFoundError) as exc:
                # Only an explicit missing/decommissioned model justifies a
                # fallback. Invalid schemas, credentials, and other requests
                # must remain visible errors instead of trying every model.
                code = _error_code(exc)
                if code not in {"model_not_found", "model_decommissioned"}:
                    raise
                limits.retry_at = time.monotonic() + 300
                self._unavailable.add(model)
                logger.warning("Groq model %s is unavailable (%s).", model, code)
                last_error = exc
            except APIStatusError as exc:
                # Groq also reports requests exceeding ITPM/TPM as 413 with
                # code=rate_limit_exceeded. The SDK does not classify this as
                # RateLimitError, so it needs an explicit routing path.
                if exc.status_code != 413 or _error_code(exc) != "rate_limit_exceeded":
                    raise
                limits.update(exc.response.headers)
                logger.info(
                    "Groq model %s cannot fit this request's token rate budget; "
                    "checking the next configured model.",
                    model,
                )
                # A smaller request may fit immediately. Do not put the model
                # into a global cooldown just because this prompt is too big.
                last_error = RateLimitError(
                    "The request exceeds the Groq model's token rate budget.",
                    response=exc.response,
                    body=exc.body,
                )

        if last_error is not None:
            raise last_error
        if all(model in self._unavailable for model in self.models):
            raise RuntimeError("All configured Groq models are unavailable; check the model IDs.")
        # No network call is needed when all models are in a known cooldown.
        # Use the existing rate-limit error path in the Discord layer.
        response = httpx.Response(
            429, request=httpx.Request("POST", GROQ_BASE_URL + "/chat/completions")
        )
        raise RateLimitError(
            "All configured Groq models are temporarily limited.", response=response, body=None
        )


def _error_code(error: Exception) -> str | None:
    body = getattr(error, "body", None)
    if not isinstance(body, dict):
        return None
    details = body.get("error", body)
    return details.get("code") if isinstance(details, dict) else None
