"""Tests for chord.llm - the OpenAI-compatible client wrapper.

The OpenAI SDK (>=3.x) ships its own vendored HTTP stack, which external
HTTP-mocking libraries cannot reliably intercept. Instead of mocking the
network, these tests inject a fake client object into ``LLMService`` and
verify the wrapper's actual job: passing model / messages / tools through
untouched and returning the completion as-is.
"""

from __future__ import annotations

import pytest
from openai import BadRequestError
from openai.types.chat import ChatCompletion

from chord.config import Settings
from chord.llm import (
    GEMINI_HARM_CATEGORIES,
    LLMService,
    build_extra_body,
    create_client,
    is_reasoning_rejection,
    merge_extra_body,
)

API_BASE = "http://llm.test/v1"

# Minimal chat.completions response bodies in OpenAI wire format.
CHAT_RESPONSE = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 1_700_000_000,
    "model": "test-model",
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "hello!"},
        }
    ],
}

TOOL_CALL_RESPONSE = {
    "id": "chatcmpl-2",
    "object": "chat.completion",
    "created": 1_700_000_001,
    "model": "test-model",
    "choices": [
        {
            "index": 0,
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Seoul"}',
                        },
                    }
                ],
            },
        }
    ],
}


class FakeCompletions:
    """Stands in for ``client.chat.completions`` and records calls."""

    def __init__(self, response: dict):
        self._response = response
        self.calls: list[dict] = []

    async def create(self, **kwargs) -> ChatCompletion:
        self.calls.append(kwargs)
        return ChatCompletion.model_validate(self._response)


class FakeChat:
    def __init__(self, response: dict):
        self.completions = FakeCompletions(response)


class FakeClient:
    def __init__(self, response: dict):
        self.chat = FakeChat(response)


def _settings() -> Settings:
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        discord_token="discord-token",
        openai_api_key="secret-key",
        openai_base_url=API_BASE,
        openai_model="test-model",
    )


def _service(response: dict) -> tuple[LLMService, FakeClient]:
    fake = FakeClient(response)
    return LLMService(_settings(), client=fake), fake


def test_create_client_uses_configured_base_url():
    """The provider endpoint comes from settings - this is what makes the
    bot work with OpenAI, OpenRouter, Ollama or anything compatible."""
    client = create_client(_settings())
    assert str(client.base_url).rstrip("/") == API_BASE


async def test_complete_passes_model_and_messages_through():
    service, fake = _service(CHAT_RESPONSE)

    messages = [
        {"role": "system", "content": "you are a bot"},
        {"role": "user", "content": "hi"},
    ]
    completion = await service.complete(messages=messages)

    call = fake.chat.completions.calls[0]
    assert call["model"] == "test-model"
    assert call["messages"] == messages
    assert "tools" not in call  # no tools -> no tool fields sent
    assert completion.choices[0].message.content == "hello!"


async def test_complete_passes_tools_with_auto_choice():
    service, fake = _service(TOOL_CALL_RESPONSE)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]

    completion = await service.complete(
        messages=[{"role": "user", "content": "weather in Seoul?"}],
        tools=tools,
    )

    call = fake.chat.completions.calls[0]
    assert call["tools"] == tools
    assert call["tool_choice"] == "auto"

    # The requested tool call is surfaced untouched for the caller.
    tool_calls = completion.choices[0].message.tool_calls
    assert tool_calls is not None
    assert tool_calls[0].function.name == "get_weather"
    assert '"city"' in tool_calls[0].function.arguments


async def test_complete_returns_full_completion_object():
    """Callers get the raw ChatCompletion so they can inspect usage etc."""
    service, _ = _service(CHAT_RESPONSE)

    completion = await service.complete(messages=[{"role": "user", "content": "?"}])

    assert isinstance(completion, ChatCompletion)
    assert completion.model == "test-model"


# -- Reasoning effort -----------------------------------------------------------------


def _bad_request(message: str) -> BadRequestError:
    """A BadRequestError carrying `message`, without a real HTTP response."""
    exc = BadRequestError.__new__(BadRequestError)
    Exception.__init__(exc, message)
    return exc


class RejectingCompletions(FakeCompletions):
    """Rejects any request carrying reasoning_effort, like Gemma does."""

    def __init__(self, response: dict, message: str):
        super().__init__(response)
        self._message = message

    async def create(self, **kwargs) -> ChatCompletion:
        if "reasoning_effort" in kwargs:
            self.calls.append(kwargs)
            raise _bad_request(self._message)
        return await super().create(**kwargs)


def _rejecting_service(
    message: str = "Thinking level is not supported for this model.",
) -> tuple[LLMService, FakeClient]:
    fake = FakeClient(CHAT_RESPONSE)
    fake.chat.completions = RejectingCompletions(CHAT_RESPONSE, message)
    return LLMService(_settings(), client=fake), fake


async def test_default_request_asks_for_minimal_reasoning():
    service, fake = _service(CHAT_RESPONSE)

    await service.complete(messages=[{"role": "user", "content": "hi"}])

    assert fake.chat.completions.calls[0]["reasoning_effort"] == "minimal"


async def test_reasoning_level_heavy_is_sent_as_high_effort():
    fake = FakeClient(CHAT_RESPONSE)
    settings = _settings().model_copy(update={"reasoning_level": "heavy"})
    service = LLMService(settings, client=fake)

    await service.complete(messages=[{"role": "user", "content": "hi"}])

    assert fake.chat.completions.calls[0]["reasoning_effort"] == "high"


async def test_reasoning_level_auto_sends_no_parameter():
    fake = FakeClient(CHAT_RESPONSE)
    settings = _settings().model_copy(update={"reasoning_level": "auto"})
    service = LLMService(settings, client=fake)

    await service.complete(messages=[{"role": "user", "content": "hi"}])

    assert "reasoning_effort" not in fake.chat.completions.calls[0]
    assert service.reasoning_enabled is False


async def test_provider_rejecting_reasoning_still_gets_an_answer():
    """Losing the knob is acceptable; losing the reply is not."""
    service, fake = _rejecting_service()

    completion = await service.complete(messages=[{"role": "user", "content": "hi"}])

    assert completion.choices[0].message.content == "hello!"
    assert "reasoning_effort" not in fake.chat.completions.calls[-1]


async def test_reasoning_rejection_is_remembered_for_later_requests():
    """An incompatible model costs one 400 per process, not one per message."""
    service, fake = _rejecting_service()

    await service.complete(messages=[{"role": "user", "content": "one"}])
    calls_after_first = len(fake.chat.completions.calls)
    await service.complete(messages=[{"role": "user", "content": "two"}])

    assert service.reasoning_enabled is False
    # Second turn is a single call - no repeated probe of the dead parameter.
    assert len(fake.chat.completions.calls) == calls_after_first + 1


async def test_gemini_thinking_budget_wording_is_recognized():
    service, _ = _rejecting_service("Thinking budget is not supported for this model.")

    completion = await service.complete(messages=[{"role": "user", "content": "hi"}])

    assert completion.choices[0].message.content == "hello!"


async def test_openai_unrecognized_argument_wording_is_recognized():
    service, _ = _rejecting_service("Unrecognized request argument supplied: reasoning_effort")

    completion = await service.complete(messages=[{"role": "user", "content": "hi"}])

    assert completion.choices[0].message.content == "hello!"


async def test_unrelated_bad_request_is_not_swallowed():
    """A broken tool schema must not be misread as a reasoning rejection."""
    fake = FakeClient(CHAT_RESPONSE)

    async def create(**kwargs):
        raise _bad_request("Invalid JSON schema for function 'get_weather'.")

    fake.chat.completions.create = create
    service = LLMService(_settings(), client=fake)

    with pytest.raises(BadRequestError):
        await service.complete(messages=[{"role": "user", "content": "hi"}])
    assert service.reasoning_enabled is True


def test_is_reasoning_rejection_ignores_unrelated_errors():
    assert is_reasoning_rejection("Thinking budget is not supported")
    assert not is_reasoning_rejection("rate limit exceeded")


def test_client_carries_the_configured_timeout():
    """A stalled provider must fail fast enough to still be answerable."""
    settings = _settings().model_copy(update={"llm_timeout_seconds": 45.0})
    assert create_client(settings).timeout == 45.0


# -- Provider extras (safety filters, escape hatch) -------------------------------------

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"


def _gemini_settings(**overrides) -> Settings:
    return _settings().model_copy(update={"openai_base_url": GEMINI_BASE, **overrides})


async def test_no_extra_body_is_sent_by_default():
    """Every provider understands a request that carries no extras."""
    service, fake = _service(CHAT_RESPONSE)

    await service.complete(messages=[{"role": "user", "content": "hi"}])

    assert "extra_body" not in fake.chat.completions.calls[0]


def test_safety_off_asks_gemini_to_stop_blocking_every_category():
    """A category left out keeps its default, so all of them are listed."""
    body = build_extra_body(_gemini_settings(llm_safety_filters="off"))

    settings_sent = body["extra_body"]["google"]["safety_settings"]
    assert {entry["category"] for entry in settings_sent} == set(GEMINI_HARM_CATEGORIES)
    assert {entry["threshold"] for entry in settings_sent} == {"BLOCK_NONE"}


def test_safety_off_is_a_no_op_on_providers_without_the_knob(caplog):
    """Silently sending Gemini fields to OpenAI would just 400."""
    with caplog.at_level("WARNING"):
        body = build_extra_body(_settings().model_copy(update={"llm_safety_filters": "off"}))

    assert body == {}
    assert "no effect" in caplog.text


async def test_safety_off_reaches_the_request():
    fake = FakeClient(CHAT_RESPONSE)
    service = LLMService(_gemini_settings(llm_safety_filters="off"), client=fake)

    await service.complete(messages=[{"role": "user", "content": "hi"}])

    sent = fake.chat.completions.calls[0]["extra_body"]
    assert sent["extra_body"]["google"]["safety_settings"]


def test_extra_body_json_is_merged_next_to_the_safety_settings():
    """A hand-written google.* field must not wipe the ones we generated."""
    body = build_extra_body(
        _gemini_settings(
            llm_safety_filters="off",
            llm_extra_body='{"extra_body": {"google": {"cached_content": "abc"}}}',
        )
    )

    google = body["extra_body"]["google"]
    assert google["cached_content"] == "abc"
    assert len(google["safety_settings"]) == len(GEMINI_HARM_CATEGORIES)


def test_hand_written_json_wins_over_the_shortcut():
    body = build_extra_body(
        _gemini_settings(
            llm_safety_filters="off",
            llm_extra_body='{"extra_body": {"google": {"safety_settings": []}}}',
        )
    )

    assert body["extra_body"]["google"]["safety_settings"] == []


def test_merge_extra_body_leaves_its_inputs_alone():
    base = {"a": {"b": 1}}
    merged = merge_extra_body(base, {"a": {"c": 2}})

    assert merged == {"a": {"b": 1, "c": 2}}
    assert base == {"a": {"b": 1}}


class ExtraBodyRejectingCompletions(FakeCompletions):
    """Stands in for a provider that has never heard of our extras."""

    async def create(self, **kwargs) -> ChatCompletion:
        if "extra_body" in kwargs:
            self.calls.append(kwargs)
            raise _bad_request("Unknown name \"safety_settings\" at 'extra_body'.")
        return await super().create(**kwargs)


def _extra_body_rejecting_service() -> tuple[LLMService, FakeClient]:
    fake = FakeClient(CHAT_RESPONSE)
    fake.chat.completions = ExtraBodyRejectingCompletions(CHAT_RESPONSE)
    settings = _gemini_settings(llm_safety_filters="off", reasoning_level="auto")
    return LLMService(settings, client=fake), fake


async def test_a_rejected_extra_body_costs_the_knob_not_the_reply():
    service, fake = _extra_body_rejecting_service()

    completion = await service.complete(messages=[{"role": "user", "content": "hi"}])

    assert completion.choices[0].message.content == "hello!"
    assert "extra_body" not in fake.chat.completions.calls[-1]


async def test_the_extra_body_rejection_is_remembered():
    service, fake = _extra_body_rejecting_service()

    await service.complete(messages=[{"role": "user", "content": "one"}])
    calls_after_first = len(fake.chat.completions.calls)
    await service.complete(messages=[{"role": "user", "content": "two"}])

    assert len(fake.chat.completions.calls) == calls_after_first + 1


async def test_a_reasoning_rejection_leaves_the_extra_body_in_place():
    """One 400 must not cost both knobs."""
    fake = FakeClient(CHAT_RESPONSE)
    fake.chat.completions = RejectingCompletions(
        CHAT_RESPONSE, "Thinking level is not supported for this model."
    )
    service = LLMService(_gemini_settings(llm_safety_filters="off"), client=fake)

    await service.complete(messages=[{"role": "user", "content": "hi"}])

    last = fake.chat.completions.calls[-1]
    assert "reasoning_effort" not in last
    assert "extra_body" in last
