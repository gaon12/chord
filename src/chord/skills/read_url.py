"""Read-a-link skill - opens a URL and returns its readable text.

Pasting a link and asking "이거 요약해줘" was the one obvious thing chord
could not do: ``summarize_text`` wants the text handed to it, and
``web_search`` finds pages without ever opening one. So the model had
nothing to reach for, and either said so or - worse - summarized the
page it imagined from the URL.

Extraction is deliberately plain: strip the furniture (scripts, nav
bars, footers), prefer ``<article>``/``<main>`` when the page marks it,
collapse the rest to text. No readability heuristics, no BeautifulSoup.
A language model is a very good salvager of slightly messy text, and the
alternative is a dependency tree bigger than the rest of the bot.

What it cannot do is worth being honest about: a page that renders its
content with JavaScript arrives empty, because this is an HTTP client
and not a browser.
"""

from __future__ import annotations

import json
from html.parser import HTMLParser
from typing import ClassVar

from chord.skills._fetch import FetchedPage, fetch_page
from chord.skills._http import SkillHTTPError
from chord.skills.base import Skill

#: Characters handed back by default. Enough for a long article's worth
#: of substance, bounded because every one of them lands in the chat
#: history and is re-sent with every later message in the channel.
DEFAULT_MAX_CHARS = 5000

#: Ceiling on what the model may ask for, for the same reason.
MAX_CHARS_LIMIT = 15000

#: Elements whose text is never content: page furniture and code.
SKIPPED_TAGS = frozenset(
    {
        "script",
        "style",
        "noscript",
        "template",
        "svg",
        "canvas",
        "iframe",
        "form",
        "button",
        "select",
        "nav",
        "header",
        "footer",
        "aside",
    }
)

#: Elements that mark the actual article on a well-built page.
MAIN_TAGS = frozenset({"article", "main"})

#: Elements that end a line, so paragraphs survive as paragraphs.
BLOCK_TAGS = frozenset(
    {
        "p",
        "br",
        "div",
        "section",
        "li",
        "tr",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "blockquote",
        "pre",
        "hr",
        "figcaption",
        "dt",
        "dd",
    }
)

#: Inline elements that sit flush against their neighbours in the
#: markup. Without a space at their boundaries, "Hacker News" followed
#: by a "new" link comes out as "Hacker Newsnew", which costs a token
#: and reads as a typo. A stray space inside a word is the cheaper
#: mistake of the two.
INLINE_TAGS = frozenset({"a", "td", "th", "span", "label", "option", "abbr", "cite"})

#: Below this, an <article> block is a teaser or a byline rather than
#: the story, and the whole-page text is the better answer.
MIN_MAIN_CHARS = 200


class ReadableText(HTMLParser):
    """Collects visible text, remembering what came from the article."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self._skip_depth = 0
        self._main_depth = 0
        self._in_title = False
        #: (came from <article>/<main>, text) in document order.
        self._parts: list[tuple[bool, str]] = []

    # -- HTMLParser hooks ---------------------------------------------------

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in SKIPPED_TAGS:
            self._skip_depth += 1
        elif tag in MAIN_TAGS:
            self._main_depth += 1
        elif tag == "title":
            self._in_title = True
        self._break_for(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag in SKIPPED_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag in MAIN_TAGS:
            self._main_depth = max(0, self._main_depth - 1)
        elif tag == "title":
            self._in_title = False
        self._break_for(tag)

    def _break_for(self, tag: str) -> None:
        """Record whatever whitespace this tag boundary implies."""
        if tag in BLOCK_TAGS:
            self._parts.append((self._main_depth > 0, "\n"))
        elif tag in INLINE_TAGS:
            self._parts.append((self._main_depth > 0, " "))

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
        elif self._skip_depth == 0:
            self._parts.append((self._main_depth > 0, data))

    # -- Results ------------------------------------------------------------

    def text(self) -> str:
        """The page's readable text, article-only when there is one."""
        main = _join([text for in_main, text in self._parts if in_main])
        if len(main) >= MIN_MAIN_CHARS:
            return main
        return _join([text for _in_main, text in self._parts])


def _join(parts: list[str]) -> str:
    """Turn collected fragments into tidy lines.

    Consecutive duplicate lines go too: a nav list rendered as `<li>`
    tends to reappear verbatim in the footer, and paying tokens for a
    site's menu twice helps nobody.
    """
    lines: list[str] = []
    for raw_line in "".join(parts).split("\n"):
        line = " ".join(raw_line.split())
        if line and line != (lines[-1] if lines else None):
            lines.append(line)
    return "\n".join(lines)


def extract_readable(page: FetchedPage) -> tuple[str, str]:
    """``(title, text)`` for a fetched document.

    JSON and plain text are already readable and are passed through:
    running an HTML parser over them would strip anything in angle
    brackets and quietly corrupt the content.
    """
    content_type = page.content_type.lower()
    if "json" in content_type:
        return "", _pretty_json(page.text)
    if "html" not in content_type and "xml" not in content_type:
        return "", page.text.strip()

    parser = ReadableText()
    try:
        parser.feed(page.text)
        parser.close()
    except (AssertionError, ValueError):
        # html.parser gives up on some deeply broken markup; whatever it
        # collected before that beats refusing to answer.
        pass
    return " ".join(parser.title.split()), parser.text()


def _pretty_json(text: str) -> str:
    try:
        return json.dumps(json.loads(text), ensure_ascii=False, indent=2)
    except ValueError:
        return text.strip()


def _clamp_chars(max_chars: int | None) -> int:
    try:
        value = int(max_chars) if max_chars is not None else DEFAULT_MAX_CHARS
    except (TypeError, ValueError):
        return DEFAULT_MAX_CHARS
    return max(500, min(value, MAX_CHARS_LIMIT))


class ReadUrlSkill(Skill):
    name = "read_url"
    description = (
        "Open a web page at a URL and return its readable text, so you "
        "can summarize it, quote it, or answer questions about it. Use "
        "this whenever someone pastes a link and asks what it says "
        "(요약해줘, 무슨 내용이야, 정리해줘) - you have no other way to see "
        "a page. Works on articles and plain text; not on PDFs, images, "
        "or pages that need JavaScript to render."
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
                    f"How much text to return (default {DEFAULT_MAX_CHARS}, "
                    f"max {MAX_CHARS_LIMIT}). Raise it only when the "
                    "default was cut short of what you need."
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
        return "\n".join(header) + "\n\n" + body
