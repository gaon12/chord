"""Exercise the installed SDK's raw-response API without a network request."""

import json

from openai import AsyncOpenAI

try:  # SDK 3.x uses httpx2; SDK 1.x and 2.x use httpx.
    import httpx2 as sdk_httpx
except ImportError:
    import httpx as sdk_httpx

from chord.groq_provider import GROQ_BASE_URL, GROQ_FREE_MODELS
from chord.llm import LLMService
from test_groq_provider import settings


async def test_real_sdk_fallback_reads_success_and_error_headers():
    calls = []

    def handle(request):
        payload = json.loads(request.content)
        calls.append(payload)
        if payload["model"] == GROQ_FREE_MODELS[0]:
            return sdk_httpx.Response(
                429,
                headers={"retry-after": "20"},
                json={
                    "error": {"message": "limited", "type": "rate_limit_error"},
                },
            )
        return sdk_httpx.Response(
            200,
            headers={
                "x-ratelimit-limit-requests": "1000",
                "x-ratelimit-remaining-requests": "999",
                "x-ratelimit-reset-requests": "1h",
                "x-ratelimit-limit-tokens": "8000",
                "x-ratelimit-remaining-tokens": "7000",
                "x-ratelimit-reset-tokens": "5s",
            },
            json={
                "id": "completion",
                "object": "chat.completion",
                "created": 0,
                "model": payload["model"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "hello"},
                    }
                ],
            },
        )

    client = AsyncOpenAI(
        api_key="fake-key",
        base_url=GROQ_BASE_URL,
        max_retries=0,
        http_client=sdk_httpx.AsyncClient(transport=sdk_httpx.MockTransport(handle)),
    )
    llm = LLMService(settings(), client=client)
    try:
        result = await llm.complete([{"role": "user", "content": "hello"}])
        assert result.choices[0].message.content == "hello"
        assert [call["model"] for call in calls] == list(GROQ_FREE_MODELS[:2])
        assert [call["reasoning_effort"] for call in calls] == ["none", "low"]
        assert all(call["reasoning_format"] == "hidden" for call in calls)
        assert "RPD: 999/1000" in llm.describe_usage()
        assert "TPM: 7000/8000" in llm.describe_usage()
        assert "effective reasoning: low" in llm.describe_reasoning()
    finally:
        await llm.aclose()
