"""Tests for chord.llm - the OpenAI-compatible client wrapper.

The OpenAI SDK (>=3.x) ships its own vendored HTTP stack, which external
HTTP-mocking libraries cannot reliably intercept. Instead of mocking the
network, these tests inject a fake client object into ``LLMService`` and
verify the wrapper's actual job: passing model / messages / tools through
untouched and returning the completion as-is.
"""

from __future__ import annotations

from openai.types.chat import ChatCompletion

from chord.config import Settings
from chord.llm import LLMService, create_client

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
