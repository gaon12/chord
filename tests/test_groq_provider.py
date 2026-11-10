"""Ordered fallback, safe request fields, and per-model reasoning behavior."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from openai import APIStatusError, AuthenticationError, BadRequestError, RateLimitError
from openai.types.chat import ChatCompletion

from chord import groq_limits, groq_provider
from chord.config import Settings, load_settings
from chord.engine import ChatEngine
from chord.groq_provider import GROQ_BASE_URL, GROQ_FREE_MODELS, groq_reasoning_effort
from chord.llm import LLMService, create_client
from chord.skills.base import Skill
from chord.skills.registry import SkillRegistry
from chord.token_estimates import estimate_request_tokens
from test_engine import _tool_call_completion

MODELS = GROQ_FREE_MODELS
MESSAGES = [{"role": "user", "content": "hello"}]


@pytest.fixture
def groq_clock(monkeypatch):
    clock, sleeps = [1000.0], []

    async def sleep(delay):
        sleeps.append(delay)
        clock[0] += delay

    timer = SimpleNamespace(monotonic=lambda: clock[0])
    monkeypatch.setattr(groq_provider, "time", timer)
    monkeypatch.setattr(groq_limits, "time", timer)
    monkeypatch.setattr(groq_provider, "asyncio", SimpleNamespace(Lock=asyncio.Lock, sleep=sleep))
    return clock, sleeps


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


async def test_nearly_exhausted_model_is_skipped_before_sending_another_request():
    llm, create = service(
        response(
            **{
                "x-ratelimit-remaining-tokens": "1200",
                "x-ratelimit-reset-tokens": "60s",
            }
        ),
        response(MODELS[1]),
        response(),
    )
    await llm.complete(MESSAGES)
    await llm.complete([{"role": "user", "content": "긴 질문입니다. " * 100}])
    assert create.call_args.kwargs["model"] == MODELS[1]
    # Skipping a large request must not globally disable smaller requests.
    await llm.complete(MESSAGES)
    assert create.call_args.kwargs["model"] == MODELS[0]


async def test_tool_schema_budget_can_trigger_preemptive_fallback():
    llm, create = service(response(MODELS[1]))
    tools = [{"type": "function", "function": {"name": "lookup", "description": "x" * 2000}}]
    llm._groq.limits[MODELS[0]].update(
        {
            "x-ratelimit-remaining-tokens": str(estimate_request_tokens(MESSAGES, None, 1024)),
            "x-ratelimit-reset-tokens": "60s",
        }
    )
    await llm.complete(MESSAGES, tools)
    assert create.await_count == 1
    assert create.call_args.kwargs["model"] == MODELS[1]


async def test_all_models_with_insufficient_remaining_tokens_fail_without_requests():
    llm, create = service()
    for limits in llm._groq.limits.values():
        limits.update({"x-ratelimit-remaining-tokens": "10", "x-ratelimit-reset-tokens": "60s"})
    with pytest.raises(RateLimitError):
        await llm.complete(MESSAGES)
    create.assert_not_awaited()


async def test_token_rate_budget_413_falls_back_without_a_global_cooldown():
    llm, create = service(
        error(APIStatusError, 413, code="rate_limit_exceeded"),
        response(MODELS[1]),
        response(),
    )
    await llm.complete(MESSAGES)
    assert [c.kwargs["model"] for c in create.call_args_list] == list(MODELS[:2])
    await llm.complete(MESSAGES)
    assert create.call_args.kwargs["model"] == MODELS[0]


async def test_all_models_exceeding_token_rate_budget_raise_a_usage_limit():
    llm, create = service(*(error(APIStatusError, 413, code="rate_limit_exceeded") for _ in MODELS))
    with pytest.raises(RateLimitError):
        await llm.complete(MESSAGES)
    assert create.await_count == len(MODELS)


@pytest.mark.parametrize("status", [400, 403, 413, 500])
async def test_unrelated_status_errors_do_not_trigger_fallback(status):
    llm, create = service(error(APIStatusError, status))
    with pytest.raises(APIStatusError):
        await llm.complete(MESSAGES)
    assert create.await_count == 1


async def test_all_models_limited_fail_quickly_without_extra_requests():
    llm, create = service(error(), error(), error())
    with pytest.raises(RateLimitError):
        await llm.complete(MESSAGES)
    with pytest.raises(RateLimitError):
        await llm.complete(MESSAGES)
    assert create.await_count == 3


async def test_all_limited_models_resume_after_the_earliest_short_cooldown(groq_clock):
    llm, create = service(
        error(headers={"retry-after": "2"}),
        error(headers={"retry-after": "5"}),
        error(headers={"retry-after": "6"}),
        response(),
    )
    await llm.complete(MESSAGES)
    assert groq_clock[1] == [2]
    assert [c.kwargs["model"] for c in create.call_args_list] == [*MODELS, MODELS[0]]


async def test_unavailable_fallbacks_do_not_mask_a_recoverable_primary_rate_limit(groq_clock):
    llm, create = service(
        error(headers={"retry-after": "2"}),
        error(BadRequestError, 400, code="model_decommissioned"),
        error(BadRequestError, 400, code="model_decommissioned"),
        response(),
    )
    await llm.complete(MESSAGES)
    assert groq_clock[1] == [2]
    assert [c.kwargs["model"] for c in create.call_args_list] == [*MODELS, MODELS[0]]


async def test_recovery_waits_only_once_if_quota_is_consumed_again(groq_clock):
    llm, create = service(
        error(headers={"retry-after": "2"}),
        error(headers={"retry-after": "5"}),
        error(headers={"retry-after": "6"}),
        error(headers={"retry-after": "2"}),
    )
    with pytest.raises(RateLimitError):
        await llm.complete(MESSAGES)
    assert groq_clock[1] == [2]
    assert create.await_count == 4


async def test_413_model_is_not_retried_while_recovering_another_models_quota(groq_clock):
    llm, create = service(
        error(APIStatusError, 413, code="rate_limit_exceeded"),
        error(headers={"retry-after": "2"}),
        error(headers={"retry-after": "600"}),
        response(MODELS[1]),
    )
    await llm.complete(MESSAGES)
    assert groq_clock[1] == [2]
    assert [c.kwargs["model"] for c in create.call_args_list] == [*MODELS, MODELS[1]]


async def test_daily_quota_prevents_a_short_retry_after_from_triggering_recovery(groq_clock):
    llm, create = service(
        *(
            error(
                headers={
                    "retry-after": "2",
                    "x-ratelimit-remaining-requests": "0",
                    "x-ratelimit-reset-requests": "24h",
                }
            )
            for _ in MODELS
        )
    )
    with pytest.raises(RateLimitError):
        await llm.complete(MESSAGES)
    assert groq_clock[1] == []
    assert create.await_count == 3


@pytest.mark.parametrize(("duration", "waits"), [("2.5s", [2.5]), ("1h2m", [])])
async def test_error_body_retry_duration_is_used_when_headers_are_missing(
    duration, waits, groq_clock
):
    limited = error()
    limited.body = {"error": {"message": f"Rate limit reached. Please try again in {duration}."}}
    llm, create = service(
        limited,
        error(headers={"retry-after": "600"}),
        error(headers={"retry-after": "600"}),
        response(),
    )
    if waits:
        await llm.complete(MESSAGES)
    else:
        with pytest.raises(RateLimitError):
            await llm.complete(MESSAGES)
        with pytest.raises(RateLimitError):
            await llm.complete(MESSAGES)
    assert groq_clock[1] == waits
    assert create.await_count == (4 if waits else 3)


async def test_preemptively_skipped_models_can_recover_without_failing_a_turn(groq_clock):
    llm, create = service(response())
    for index, model in enumerate(MODELS):
        llm._groq.limits[model].update(
            {"x-ratelimit-remaining-tokens": "10", "x-ratelimit-reset-tokens": f"{index + 2}s"}
        )
    await llm.complete(MESSAGES)
    assert groq_clock[1] == [2]
    assert create.await_count == 1


async def test_recovery_releases_provider_lock_so_smaller_requests_can_continue(
    monkeypatch, groq_clock
):
    llm, create = service(response(), response())
    for index, model in enumerate(MODELS):
        llm._groq.limits[model].update(
            {
                "x-ratelimit-remaining-tokens": "2000" if index == 0 else "0",
                "x-ratelimit-reset-tokens": "2s" if index == 0 else "60s",
            }
        )

    async def recover(delay):
        assert not llm._groq._lock.locked()
        await llm.complete(MESSAGES)
        groq_clock[0][0] += delay

    monkeypatch.setattr(groq_provider.asyncio, "sleep", recover)
    await asyncio.wait_for(llm.complete([{"role": "user", "content": "long request " * 1000}]), 2)
    assert create.await_count == 2
    assert create.call_args_list[0].kwargs["messages"] == MESSAGES


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


@pytest.mark.parametrize("limit", ["429", "413", "remaining_tokens", "recovery"])
async def test_mid_tool_turn_switches_models_without_repeating_the_tool(limit, groq_clock):
    class Lookup(Skill):
        name = "lookup"
        description = "Look up a result."

        async def run(self):
            return "결과" * 1000

    skill = Lookup()
    skill.run = AsyncMock(wraps=skill.run)
    registry = SkillRegistry()
    registry.register(skill)
    first = response(
        **{
            "x-ratelimit-remaining-tokens": "2000",
            "x-ratelimit-reset-tokens": "60s",
        }
        if limit == "remaining_tokens"
        else {}
    )
    first.parse = lambda: ChatCompletion.model_validate(
        _tool_call_completion(("lookup-1", "lookup", "{}"))
    )
    results = [first]
    if limit == "recovery":
        results.extend(error(headers={"retry-after": str(delay)}) for delay in [2, 600, 600])
    elif limit != "remaining_tokens":
        results.append(
            error() if limit == "429" else error(APIStatusError, 413, code="rate_limit_exceeded")
        )
    llm, create = service(*results, response(MODELS[0] if limit == "recovery" else MODELS[1]))
    engine = ChatEngine(llm, registry, "system")
    answer, turn = await engine.reply("look it up", [])
    assert answer == "hello"
    expected = [MODELS[0], MODELS[1]]
    if limit == "recovery":
        expected = [MODELS[0], *MODELS, MODELS[0]]
        assert groq_clock[1] == [2]
    elif limit != "remaining_tokens":
        expected.insert(1, MODELS[0])
    assert [call.kwargs["model"] for call in create.call_args_list] == expected
    skill.run.assert_awaited_once()
    assert [message["role"] for message in turn] == ["user", "assistant", "tool", "assistant"]
    assert create.call_args.kwargs["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "lookup-1",
        "content": "결과" * 1000,
    }


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
