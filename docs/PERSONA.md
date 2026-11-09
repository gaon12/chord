# Nova's persona and prompt assembly

Nova is chord's fictional adult Discord companion: quick-witted,
playfully defensive, secretly affectionate, and easily flustered. Small,
considerate actions reveal the warmth behind her words. Teasing and brief
italicized gestures are optional, rather than a required pattern in every
answer.

## Files and responsibilities

| Location | Responsibility |
|---|---|
| `persona.md` | Editable character, voice, response style, and dialogue examples |
| `src/chord/persona.py` | Compact fallback persona, file loading, fixed operating rules, and tool-routing rules |
| `src/chord/answer_policy.py` | Shared truthfulness, evidence, uncertainty, and correction rules |
| `src/chord/account_preferences.py` | Automatic answer-detail preference for gaon12's verified Discord user ID |
| `src/chord/bot.py` | Combines the loaded prompt with the current tool index before each turn |
| `src/chord/tool_catalog.py` | Adds schema-selection instructions when tools load on demand |
| `src/chord/config.py` | Initial `SYSTEM_PROMPT` setting used before Discord's per-turn prompt assembly |

There is no separate `system_prompt.md`. The runtime system prompt combines
the character body, fixed operating and routing rules, and a short index of
the registered tools. On-demand mode appends its selection instructions.
Changing `SYSTEM_PROMPT` is not a replacement for editing the character file:
the Discord message handler rebuilds its prompt from the persona provider.

## Response behavior

- Casual conversation may include gentle teasing, hesitation, looking away,
  or quietly making room for someone. Avoid repeated catchphrases and long
  action narratives.
- Match the speaker's language, formality, and comfort. In group channels,
  keep participants distinct and do not assume everyone wants romantic
  attention.
- Keep affection non-explicit. A request to stop or avoid a nickname is a
  boundary to respect, not an invitation to continue.
- Practical questions get useful answers first. Facts, arithmetic, citations,
  and tool results stay accurate regardless of the character's mood.
- Serious or distressing topics receive calm, direct help without teasing or
  romantic performance. State uncertainty and operational limits honestly.

The dialogue examples in `persona.md` illustrate the tone. They are not
scripts to repeat or facts about events that actually happened.

## Truthfulness and uncertainty

Every persona receives the same fixed accuracy rules from `answer_policy.py`.
These rules also apply to LLM-backed translation, summarization, and
explanation tools. A character edit does not remove them.

Answers must distinguish verified facts, source claims, assumptions, and
inferences. When knowledge or evidence is missing, the bot should say that
it does not know or cannot verify the claim. It must not fabricate citations,
API fields, repository contents, tool results, or tests it did not run.
Version-sensitive and current claims need verification using available tools
and primary sources. A false premise should be corrected politely, including
when the requester is the bot's owner.

History summaries preserve uncertainty, corrections, and attribution so an
earlier guess does not become a verified fact after compaction. Translation
and summarization preserve source claims without pretending to verify them.

These instructions improve the expected behavior; they do not prove that
every generated answer is correct. Important claims still need evidence and
appropriate checks.

## Automatic detail for gaon12

Messages from gaon12's verified Discord user ID receive additional guidance
to make answers easier to review: a clear conclusion, supporting evidence,
necessary assumptions, unresolved points, and useful verification steps.
Technical answers identify relevant versions and edge cases when evidence
supports them, and distinguish tests actually run from checks merely suggested.

This is automatic. There is no new command, toggle, or announcement in the
reply. Simple questions stay brief, and the bot does not address the requester
as a developer. Everyone receives the same truthfulness and uncertainty rules;
anyone can still ask for a detailed explanation.

The preference is bound to a permanent Discord user ID in
`account_preferences.py`, not a display name, username string, or claim in a
message. A nickname change does not remove it, and another account using the
name gaon12 does not acquire it. Application ownership alone does not select
this account preference or grant any additional tool permission.

The extra guidance applies only to the current request. Discord passes a
complete system-prompt snapshot to the engine rather than modifying its shared
default. It is not stored in channel history or inherited by the next user,
and concurrent channels keep their own snapshots. The preference does not
change the provider, reasoning setting, token budget, or access controls.

## Editing and reloading

Edit `persona.md` in UTF-8. The provider checks its content signature before
each message, so an edit is used on the next turn without restarting the bot.
Use `PERSONA_PATH` to select another file.

The application owner can use `/persona view` for a short preview or
`/persona reload` to refresh the file explicitly. A missing or deleted file
uses the compact `DEFAULT_PERSONA` in `src/chord/persona.py`. Keep that fallback
aligned with major character changes so a missing file does not restore an
older personality.

Fixed rules continue to govern speaker labels, tool use, private information,
and untrusted tool output. Authentication, network guards, tool permissions,
and Discord channel access are enforced outside the character text.

Conversation history can still contain earlier replies. To preview a new
style without those examples, start a fresh test conversation or use `/reset`
with the required channel permission. Do not clear another channel's history
just to test a character edit.

## Checking a character change

Run Ruff lint and format checks, then the existing persona and Discord tests.
For a live preview, use synthetic prompts that cover casual conversation, a
factual question, and an explicit boundary such as an unwanted nickname.
Check that the answer remains useful and the boundary is accepted.

Live model responses vary. A short preview provides an example of behavior,
not a guarantee that every later answer will use the same wording or style.
