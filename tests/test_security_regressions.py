"""Security boundaries found while reviewing the integrated provider changes."""

import ssl
from unittest.mock import AsyncMock

import httpcore
import pytest
from openai import BadRequestError

from chord.llm_options import rate_limit_delay
from chord.network import PublicNetworkBackend, public_addresses
from test_bot import FakeChannel, FakeMessage, _bad_request, _bot
from test_groq_provider import MODELS, error, response, service
from test_llm import _rate_limited


@pytest.mark.parametrize(
    "address",
    [
        "224.0.0.1",
        "ff02::1",
        "fec0::1",
        "2002:7f00:1::1",
        "64:ff9b::7f00:1",
    ],
)
def test_special_and_encapsulated_private_destinations_are_refused(address):
    with pytest.raises(ValueError):
        public_addresses(address)


async def test_https_keeps_original_hostname_and_certificate_verification():
    class Stream(httpcore.AsyncNetworkStream):
        def __init__(self):
            self.writes = []
            self.hostname = None
            self.sent = False

        async def read(self, max_bytes, timeout=None):
            if self.sent:
                return b""
            self.sent = True
            return b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok"

        async def write(self, buffer, timeout=None):
            self.writes.append(buffer)

        async def start_tls(self, ssl_context, server_hostname=None, timeout=None):
            assert ssl_context.verify_mode == ssl.CERT_REQUIRED
            assert ssl_context.check_hostname
            self.hostname = server_hostname
            return self

        async def aclose(self):
            pass

        def get_extra_info(self, info):
            return None

    stream = Stream()
    delegate = AsyncMock()
    delegate.connect_tcp.return_value = stream
    pool = httpcore.AsyncConnectionPool(network_backend=PublicNetworkBackend(delegate))
    try:
        result = await pool.request("GET", "https://example.com/page")
        assert result.content == b"ok"
        assert stream.hostname == "example.com"
        assert b"Host: example.com\r\n" in b"".join(stream.writes)
        assert delegate.connect_tcp.call_args.args[0] == "93.184.216.34"
    finally:
        await pool.aclose()


@pytest.mark.parametrize("value", ["nan", "inf", "-1"])
def test_generic_provider_cannot_supply_an_invalid_retry_delay(value):
    assert rate_limit_delay(_rate_limited(retry_after=value)) is None


async def test_custom_model_does_not_report_an_unsent_reasoning_parameter():
    llm, create = service(response("custom"), groq_free_tier=False, groq_model="custom")
    await llm.complete([{"role": "user", "content": "hello"}])
    assert not llm.reasoning_enabled
    assert "reasoning_effort" not in create.call_args.kwargs


async def test_unavailable_models_are_not_reported_as_exhausted_quotas():
    llm, create = service(
        *[error(BadRequestError, 400, code="model_decommissioned") for _ in MODELS]
    )
    with pytest.raises(BadRequestError):
        await llm.complete([])
    with pytest.raises(RuntimeError, match="models are unavailable"):
        await llm.complete([])
    assert create.await_count == 3


async def test_unrelated_request_error_preserves_history_and_private_text(caplog):
    bot, engine = _bot(history_token_budget=0)
    channel = FakeChannel(1)
    await bot.on_message(FakeMessage("old message", channel))
    history = bot._store.history(1)

    async def reject(text, history):
        raise _bad_request("Invalid tool schema; private-provider-detail-123")

    engine.reply = reject
    with caplog.at_level("DEBUG"):
        await bot.on_message(FakeMessage("private-user-message-123", channel))
    assert bot._store.history(1) == history
    assert "private-provider-detail-123" not in caplog.text
    assert "private-user-message-123" not in caplog.text


def _truncated_llm():
    from openai.types.chat import ChatCompletion

    from test_engine import _text_completion

    payload = _text_completion("partial output")
    payload["choices"][0]["finish_reason"] = "length"
    llm = AsyncMock()
    llm.complete.return_value = ChatCompletion.model_validate(payload)
    return llm


async def test_truncated_summary_never_replaces_original_history():
    from chord.compaction import HistoryCompactor

    history = [
        {"role": "user", "content": "old " * 100},
        {"role": "assistant", "content": "answer " * 100},
        {"role": "user", "content": "new question"},
    ]
    compactor = HistoryCompactor(_truncated_llm(), token_budget=1, keep_tokens=1)
    assert await compactor.compact(history) is None
    assert history[0]["content"] == "old " * 100


async def test_truncated_text_task_is_reported_as_incomplete():
    from chord.skills._errors import SkillError
    from chord.skills.summarize import SummarizeSkill

    with pytest.raises(SkillError, match="output limit"):
        await SummarizeSkill(_truncated_llm()).run("summarize this")
