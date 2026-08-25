"""Read-a-link skill - opens a URL and returns its readable text.

Pasting a link and asking "이거 요약해줘" was the one obvious thing chord
could not do: ``summarize_text`` wants the text handed to it, and
``web_search`` finds pages without ever opening one. So the model had
nothing to reach for, and either said so or - worse - summarized the
page it imagined from the URL.

Fetching lives in :mod:`chord.skills._fetch` (which is where the rules
about *where* a request may go are), and turning markup into text in
:mod:`chord.skills._readable`. What is left here is the skill itself:
how much text to hand back, and how to label it.

What it cannot do is worth being honest about: a page that renders its
content with JavaScript arrives empty, because this is an HTTP client
and not a browser.
"""

from __future__ import annotations

from typing import ClassVar

from chord.skills._fetch import fetch_page
from chord.skills._http import SkillHTTPError
from chord.skills._readable import extract_readable, fence_untrusted
from chord.skills.base import Skill

#: Characters handed back by default. Enough for a long article's worth
#: of substance, bounded because every one of them lands in the chat
#: history and is re-sent with every later message in the channel.
DEFAULT_MAX_CHARS = 5000

#: Ceiling on what the model may ask for, for the same reason.
MAX_CHARS_LIMIT = 15000


def _clamp_chars(max_chars: int | None) -> int:
    try:
        value = int(max_chars) if max_chars is not None else DEFAULT_MAX_CHARS
    except (TypeError, ValueError):
        return DEFAULT_MAX_CHARS
    return max(500, min(value, MAX_CHARS_LIMIT))


class ReadUrlSkill(Skill):
    name = "read_url"
    description = (
        "Open a web page and return its readable text, to summarize or "
        "quote. Use whenever someone pastes a link and asks what it says "
        "(요약해줘, 무슨 내용이야, 정리해줘) - there is no other way to see a "
        "page. Not PDFs, images, or JavaScript-rendered pages."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The link to open, e.g. 'https://example.com/article'.",
            },
            "max_chars": {
                "type": "integer",
                "description": (
                    f"Characters to return (default {DEFAULT_MAX_CHARS}, max "
                    f"{MAX_CHARS_LIMIT}). Raise only if the text was cut short."
                ),
            },
        },
        "required": ["url"],
    }

    async def run(self, url: str, max_chars: int | None = None) -> str:
        page = await fetch_page(url)
        title, text = extract_readable(page)
        if not text:
            raise SkillHTTPError(
                f"{page.url} came back with no readable text. It is probably "
                "rendered by JavaScript, which I cannot run."
            )

        limit = _clamp_chars(max_chars)
        body = text[:limit]
        cut = len(text) > limit or page.truncated

        header = [f"URL: {page.url}"]
        if title:
            header.insert(0, f"Title: {title}")
        header.append(
            f"{len(body):,} characters shown"
            + (f" of {len(text):,}+ - the rest was cut" if cut else "")
        )
        return "\n".join(header) + "\n\n" + fence_untrusted(body, page.url)
