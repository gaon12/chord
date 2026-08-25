"""Per-channel conversation history kept in memory.

The bot stays stateless across restarts on purpose: history lives in RAM
and disappears when the process exits. That keeps privacy simple and
avoids any database dependency.
"""

from __future__ import annotations

from collections import defaultdict


def next_turn_start(history: list[dict], minimum: int) -> int:
    """First index at or after ``minimum`` that starts a user turn.

    Every turn begins with the user's message and may be followed by
    assistant/tool round-trips that only make sense together: a tool
    result without the assistant message that requested it is invalid to
    send back. Gemini rejects exactly that case with

        400 - contents[0].parts[0].function_response.name:
              Name cannot be empty

    so a trim must land on a user message and drop whole turns.

    Returns 0 - keep everything - when no boundary exists past
    ``minimum``, which means the tail is a single unusually long turn.
    Briefly exceeding the cap beats sending a broken conversation, and
    the overshoot is bounded because a turn is capped by the engine's
    tool-round limit.
    """
    for index in range(minimum, len(history)):
        if history[index].get("role") == "user":
            return index
    return 0


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
        recent context is what matters for a chat anyway. It always cuts
        on a turn boundary - see :func:`next_turn_start`.
        """
        channel_history = self._channels[channel_id]
        channel_history.extend(messages)
        excess = len(channel_history) - self._max
        if excess > 0:
            del channel_history[: next_turn_start(channel_history, excess)]

    def replace_prefix(
        self,
        channel_id: int,
        head: list[dict],
        replacement: list[dict],
    ) -> bool:
        """Swap the oldest messages of a channel for shorter ones.

        This is how compaction lands its summary. It runs after the
        answer has already been sent, so another message in the same
        channel may have been appended - or trimmed away - while the
        summary was being written. Comparing by identity makes that
        harmless: the swap applies only when the exact messages that
        were summarized are still at the front.

        Returns:
            True when the history was actually rewritten.
        """
        channel_history = self._channels.get(channel_id)
        if channel_history is None or len(channel_history) < len(head):
            return False
        if any(old is not seen for old, seen in zip(channel_history, head, strict=False)):
            return False
        channel_history[: len(head)] = replacement
        return True

    def reset(self, channel_id: int) -> None:
        """Forget everything said in one channel."""
        self._channels.pop(channel_id, None)
