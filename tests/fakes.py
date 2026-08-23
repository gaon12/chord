"""Shared test double for LLM-backed skills."""

from __future__ import annotations

from openai.types.chat import ChatCompletion


class FakeLLM:
    """Records complete() calls and returns a canned message."""

    def __init__(self, reply: str = "fake answer"):
        self.reply = reply
        self.calls: list[dict] = []

    async def complete(self, messages, tools=None) -> ChatCompletion:
        self.calls.append({"messages": messages, "tools": tools})
        return ChatCompletion.model_validate(
            {
                "id": "c",
                "object": "chat.completion",
                "created": 1,
                "model": "m",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": self.reply},
                    }
                ],
            }
        )
