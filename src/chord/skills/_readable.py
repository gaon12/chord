"""Turning fetched markup into text worth reading.

Shared by the skills that open pages - :mod:`chord.skills.read_url` for
a link someone pasted, :mod:`chord.skills.web_search` for the results it
found - because "what counts as the content of a page" is one question
and deserves one answer.

Deliberately plain: strip the furniture (scripts, nav bars, footers),
prefer ``<article>``/``<main>`` when the page marks it, collapse the
rest to lines. No readability heuristics, no BeautifulSoup. A language
model is a very good salvager of slightly messy text, and the
alternative is a dependency tree bigger than the rest of the bot.
"""

from __future__ import annotations

import json
from html.parser import HTMLParser

from chord.skills._fetch import FetchedPage

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
