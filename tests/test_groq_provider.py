"""Ordered fallback, safe request fields, and per-model reasoning behavior."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from openai import AuthenticationError, BadRequestError, RateLimitError
from openai.types.chat import ChatCompletion

from chord.config import Settings, load_settings
from chord.groq_provider import GROQ_BASE_URL, GROQ_FREE_MODELS, groq_reasoning_effort
from chord.llm import LLMService, create_client

MODELS = GROQ_FREE_MODELS
MESSAGES = [{"role": "user", "content": "hello"}]


def settings(**overrides):
    return Settings(
        _env_file=None,
        discord_token="discord",
        groq_api_key="groq-key",
        llm_provider="groq",
        **overrides,
    )


def response(model=MODELS[0], **headers):
    completion = ChatCompletion.model_validate(
        {
            "id": "completion",
            "object": "chat.completion",
            "created": 0,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "hello"},
                }
            ],
        }
    )
    return SimpleNamespace(headers=headers, parse=lambda: completion)


def error(cls=RateLimitError, status=429, headers=None, code=None):
    result = httpx.Response(
        status, headers=headers or {}, request=httpx.Request("POST", GROQ_BASE_URL)
    )
    return cls("provider error", response=result, body={"code": code} if code else None)


def service(*results, **overrides):
    create = AsyncMock(side_effect=results)
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(with_raw_response=SimpleNamespace(create=create))
        ),
        close=AsyncMock(),
    )
    return LLMService(settings(**overrides), client=client), create


async def test_free_models_are_tried_in_the_requested_order_without_sdk_retries():
    llm, create = service(error(), error(), response(MODELS[2]))
    result = await llm.complete(MESSAGES)
    assert [call.kwargs["model"] for call in create.call_args_list] == list(MODELS)
    assert [call.kwargs["reasoning_effort"] for call in create.call_args_list] == [
        "none",
        "low",
        "low",
    ]
    assert result.model == MODELS[2]
    assert llm.model == MODELS[2]


async def test_known_cooldown_skips_a_model_and_restores_priority_after_reset():
    llm, create = service(
        error(headers={"retry-after": "600"}),
        response(MODELS[1]),
        response(MODELS[1]),
        response(MODELS[0]),
    )
    await llm.complete(MESSAGES)
    await llm.complete(MESSAGES)
    assert [c.kwargs["model"] for c in create.call_args_list] == [MODELS[0], MODELS[1], MODELS[1]]
    llm._groq.limits[MODELS[0]].retry_at = 0
    await llm.complete(MESSAGES)
    assert create.call_args.kwargs["model"] == MODELS[0]


async def test_success_headers_can_skip_an_exhausted_primary_model():
    llm, create = service(
        response(
            **{
                "x-ratelimit-remaining-tokens": "0",
                "x-ratelimit-reset-tokens": "60s",
            }
        ),
        response(MODELS[1]),
    )
    await llm.complete(MESSAGES)
    await llm.complete(MESSAGES)
    assert create.call_args.kwargs["model"] == MODELS[1]


async def test_all_models_limited_fail_quickly_without_extra_requests():
    llm, create = service(error(), error(), error())
    with pytest.raises(RateLimitError):
        await llm.complete(MESSAGES)
    with pytest.raises(RateLimitError):
        await llm.complete(MESSAGES)
    assert create.await_count == 3


@pytest.mark.parametrize(("cls", "status"), [(AuthenticationError, 401), (BadRequestError, 400)])
async def test_authentication_and_invalid_request_errors_do_not_trigger_fallback(cls, status):
    llm, create = service(error(cls, status))
    with pytest.raises(cls):
        await llm.complete(MESSAGES)
    assert create.await_count == 1


async def test_explicit_model_decommissioning_can_fall_back():
    llm, create = service(
        error(BadRequestError, 400, code="model_decommissioned"), response(MODELS[1])
    )
    await llm.complete(MESSAGES)
    assert create.await_count == 2


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize(
    ("effort", "expected"), [(None, None), ("low", "low"), ("medium", "medium"), ("high", "high")]
)
def test_supported_reasoning_values(model, effort, expected):
    assert groq_reasoning_effort(model, effort) == expected


async def test_auto_omits_reasoning_effort_and_hides_reasoning_content():
    llm, create = service(response(), reasoning_level="auto")
    await llm.complete(MESSAGES)
    kwargs = create.call_args.kwargs
    assert "reasoning_effort" not in kwargs
    assert kwargs["extra_body"]["reasoning_format"] == "hidden"
    assert kwargs["max_completion_tokens"] == 1024


async def test_tools_and_history_survive_a_model_switch():
    llm, create = service(error(), response(MODELS[1]))
    tools = [{"type": "function", "function": {"name": "get_weather"}}]
    await llm.complete(iter(MESSAGES), tools)
    for call in create.call_args_list:
        assert call.kwargs["messages"] == MESSAGES
        assert call.kwargs["tools"] == tools
        assert call.kwargs["tool_choice"] == "auto"


async def test_paid_mode_uses_only_the_explicit_model():
    llm, create = service(response(MODELS[2]), groq_free_tier=False, groq_model=MODELS[2])
    await llm.complete(MESSAGES)
    assert create.call_args.kwargs["model"] == MODELS[2]
    assert create.await_count == 1


async def test_client_uses_official_endpoint_and_closes():
    llm, _ = service(response())
    await llm.aclose()
    llm._client.close.assert_awaited_once()
    client = create_client(settings())
    assert str(client.base_url).rstrip("/") == GROQ_BASE_URL
    assert client.max_retries == 0
    await client.close()


def test_groq_only_environment_needs_no_openai_key(tmp_path):
    env = tmp_path / ".env"
    env.write_text("DISCORD_TOKEN=t\nLLM_PROVIDER=groq\nGROQ_API_KEY=gsk-test\n")
    loaded = load_settings(env)
    assert loaded.llm_api_key == "gsk-test"
    assert "gsk-test" not in repr(loaded)


def test_explicit_groq_does_not_forward_an_openai_key(tmp_path):
    env = tmp_path / ".env"
    env.write_text("DISCORD_TOKEN=t\nLLM_PROVIDER=groq\nOPENAI_API_KEY=unrelated-openai-key\n")
    with pytest.raises(SystemExit, match="GROQ_API_KEY is empty"):
        load_settings(env)


def test_existing_groq_compatible_url_is_recognized():
    loaded = Settings(
        _env_file=None,
        discord_token="t",
        openai_api_key="legacy-groq-key",
        openai_base_url=GROQ_BASE_URL,
    )
    assert loaded.use_groq
    assert loaded.llm_api_key == "legacy-groq-key"


@pytest.mark.parametrize(
    "key", ["model", "messages", "stream", "max_completion_tokens", "reasoning_effort"]
)
def test_extra_body_cannot_bypass_free_routing_or_request_limits(key):
    import json

    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="cannot override"):
        settings(llm_extra_body=json.dumps({key: "override"}))
