"""Per-channel conversation history kept in memory.

The bot stays stateless across restarts on purpose: history lives in RAM
and disappears when the process exits. That keeps privacy simple and
avoids any database dependency.
"""

from __future__ import annotations

from collections import defaultdict


class ConversationStore:
    """Remembers recent messages for each Discord channel."""

    def __init__(self, max_messages: int = 40) -> None:
        self._max = max_messages
        self._channels: defaultdict[int, list[dict]] = defaultdict(list)

    def history(self, channel_id: int) -> list[dict]:
        """Messages so far for one channel (oldest first)."""
        return list(self._channels[channel_id])

    def append(self, channel_id: int, *messages: dict) -> None:
        """Add messages to a channel, trimming old ones beyond the limit.

        Trimming keeps memory bounded and requests small; the most
        recent context is what matters for a chat anyway.
        """
        channel_history = self._channels[channel_id]
        channel_history.extend(messages)
        if len(channel_history) > self._max:
            del channel_history[: len(channel_history) - self._max]

    def reset(self, channel_id: int) -> None:
        """Forget everything said in one channel."""
        self._channels.pop(channel_id, None)
