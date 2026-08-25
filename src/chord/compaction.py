"""Folding old turns into a summary before the context window fills up.

Channel history grows every turn, and a long-running channel eventually
sends more prompt than the provider (or its per-minute input-token quota)
will take. The message-count cap in :mod:`chord.conversation` bounds that,
but bluntly: it *deletes* the oldest turns, so the bot forgets a decision
made twenty messages ago even though the conversation is still about it.

Compaction is the softer version. Once the stored history is estimated to
cost more than a token budget, the oldest whole turns are replaced by a
short digest of themselves, written by the same model. What was said
survives; the tokens it cost do not.

The digest re-enters history as a normal user/assistant pair rather than a
second system message, because OpenAI-compatible providers disagree about
system messages appearing anywhere but first.
"""

from __future__ import annotations

import json
import logging

from chord.llm import LLMService

logger = logging.getLogger(__name__)

#: Marks the digest inside the history so a later compaction can spot the
#: previous one and fold it in instead of nesting summaries of summaries.
SUMMARY_PREFIX = "[earlier conversation summary]"

#: What the assistant "says" after receiving a digest. Keeps the history
#: strictly alternating, which the stricter compatibility layers want.
SUMMARY_ACK = "Noted."

#: Tool results are the bulkiest thing in a history and the least worth
#: preserving verbatim - the digest keeps the finding, not the payload.
MAX_TOOL_RESULT_CHARS = 400

SUMMARY_SYSTEM_PROMPT = (
    "You compress a chat transcript so the assistant can keep the "
    "conversation going after the original messages are dropped.\n"
    "Write a compact digest of at most 200 words as short bullet lines. "
    "Keep: who said what (names matter - several people share the "
    "channel), decisions, stated preferences, unresolved questions, and "
    "facts that were looked up. Drop pleasantries and tool mechanics.\n"
    "Write it in the language the participants used. Output only the "
    "digest, with no preamble. If the transcript starts with an earlier "
    "digest, merge it into the new one."
)


def estimate_tokens(text: str) -> int:
    """Rough token count for mixed Korean/English chat text.

    Deliberately not the chars-per-token ratio used for tool schemas in
    :mod:`chord.bot`: that one is calibrated on ASCII JSON, while a
    Korean sentence costs roughly one token per character. Counting
    non-ASCII characters as a token each keeps the estimate from
    under-reading a Korean channel by a factor of three.

    Only ever used to decide *when* to compact, never for accounting.
    """
    ascii_chars = sum(1 for char in text if char.isascii())
    return ascii_chars // 4 + (len(text) - ascii_chars)


def estimate_message_tokens(message: dict) -> int:
    """Tokens one stored message adds, role framing included.

    Serializing to JSON covers tool calls and tool results as readily as
    plain content, and ``ensure_ascii=False`` keeps Korean text one
    character per character instead of six.
    """
    return estimate_tokens(json.dumps(message, ensure_ascii=False)) + 4


def estimate_history_tokens(history: list[dict]) -> int:
    return sum(estimate_message_tokens(message) for message in history)


def split_for_compaction(history: list[dict], keep_tokens: int) -> tuple[list[dict], list[dict]]:
    """Split history into ``(older, recent)`` on a turn boundary.

    ``recent`` is the newest run of *whole* turns that fits in
    ``keep_tokens``; everything before it is what gets summarized. The
    newest turn is always kept even when it alone busts the budget -
    answering the message just asked matters more than the budget, and
    cutting inside a turn would orphan a tool result (an instant 400).

    ``older`` comes back empty when there is nothing safe to cut.
    """
    boundary: int | None = None
    total = 0
    for index in range(len(history) - 1, -1, -1):
        total += estimate_message_tokens(history[index])
        if history[index].get("role") != "user":
            continue
        if boundary is not None and total > keep_tokens:
            break
        boundary = index

    if boundary is None or boundary == 0:
        return [], history
    return history[:boundary], history[boundary:]


def render_transcript(messages: list[dict]) -> str:
    """Turn stored messages into something readable to summarize."""
    lines: list[str] = []
    for message in messages:
        role = message.get("role")
        content = (message.get("content") or "").strip()
        if role == "tool":
            lines.append(f"tool result: {_truncate(content, MAX_TOOL_RESULT_CHARS)}")
            continue
        if role == "assistant":
            for call in message.get("tool_calls") or []:
                function = call.get("function", {})
                lines.append(
                    f"assistant called {function.get('name')}({function.get('arguments')})"
                )
            if content:
                lines.append(f"assistant: {content}")
            continue
        if content:
            # User messages already carry their "[name]: " label.
            lines.append(content)
    return "\n".join(lines)


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def summary_messages(summary: str) -> list[dict]:
    """The digest as history the provider will accept."""
    return [
        {"role": "user", "content": f"{SUMMARY_PREFIX}\n{summary}"},
        {"role": "assistant", "content": SUMMARY_ACK},
    ]


class HistoryCompactor:
    """Summarizes a channel's oldest turns once they get expensive."""

    def __init__(
        self,
        llm: LLMService | None,
        token_budget: int,
        keep_tokens: int | None = None,
    ) -> None:
        self._llm = llm
        self._budget = token_budget
        # Compacting down to the budget itself would trigger again on the
        # very next turn, paying for a summary every message. Keeping a
        # third leaves room for several turns before the next one.
        self._keep_tokens = keep_tokens if keep_tokens is not None else max(token_budget // 3, 1)

    @property
    def enabled(self) -> bool:
        return self._llm is not None and self._budget > 0

    async def compact(self, history: list[dict]) -> tuple[list[dict], list[dict]] | None:
        """Summarize the oldest turns of ``history`` if it got too big.

        Returns:
            ``(consumed, replacement)`` - the messages that were folded
            up, and the digest that should stand in for them - or None
            when nothing needs doing. The caller applies the swap, so a
            channel that moved on meanwhile can decline it.
        """
        if not self.enabled or not history:
            return None

        used = estimate_history_tokens(history)
        if used <= self._budget:
            return None

        older, _recent = split_for_compaction(history, self._keep_tokens)
        if not older:
            # One turn is the whole history. Nothing can be summarized
            # without orphaning a tool result; the message-count cap in
            # ConversationStore is the backstop for that case.
            logger.debug("History over budget (~%d tokens) but has no turn boundary to cut", used)
            return None

        summary = await self._summarize(older)
        if not summary:
            logger.warning("Summarizer returned nothing; keeping the full history")
            return None

        logger.info(
            "Compacting %d of %d message(s) (~%d tokens, budget %d)",
            len(older),
            len(history),
            used,
            self._budget,
        )
        return older, summary_messages(summary)

    async def _summarize(self, messages: list[dict]) -> str:
        """Ask the model for a digest - no tools, it only reads."""
        assert self._llm is not None  # guarded by .enabled
        completion = await self._llm.complete(
            [
                {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": render_transcript(messages)},
            ]
        )
        return (completion.choices[0].message.content or "").strip()
