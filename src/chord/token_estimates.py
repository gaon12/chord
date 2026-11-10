"""Approximate prompt sizes for compaction and quota-aware routing.

These are scheduling estimates, not provider token accounting. Non-ASCII
text needs a more conservative allowance than the usual ASCII JSON ratio.
"""

from __future__ import annotations

import json


def estimate_tokens(text: str) -> int:
    ascii_chars = sum(1 for char in text if char.isascii())
    return ascii_chars // 4 + (len(text) - ascii_chars)


def estimate_message_tokens(message: dict) -> int:
    return estimate_tokens(json.dumps(message, ensure_ascii=False)) + 4


def estimate_history_tokens(history: list[dict]) -> int:
    return sum(estimate_message_tokens(message) for message in history)


def estimate_request_tokens(messages: list[dict], tools: list | None, max_tokens: int) -> int:
    """Reserve prompt, tool schemas, and the full completion/reasoning cap."""
    prompt = estimate_history_tokens(messages)
    if tools:
        prompt += estimate_tokens(json.dumps(tools, ensure_ascii=False)) + 8
    # Leave room for provider-specific framing and estimation error.
    return prompt + max(64, (prompt + 9) // 10) + max_tokens
