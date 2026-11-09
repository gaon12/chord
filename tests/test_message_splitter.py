"""Long code samples must remain readable in every Discord message."""

import pytest

from chord.message_splitter import split_message


@pytest.mark.parametrize("marker", ["```", "~~~~", "````"])
@pytest.mark.parametrize("body", ["print('hello')\n" * 400, "x" * 5000])
def test_long_code_reopens_its_language_and_preserves_content(marker, body):
    text = f"{marker}python\n{body}\n{marker}"
    chunks = split_message(text)
    assert len(chunks) > 1
    recovered = []
    for chunk in chunks:
        assert len(chunk) <= 2000
        assert chunk.startswith(f"{marker}python\n")
        assert chunk.endswith(f"\n{marker}")
        recovered.append(chunk[len(marker) + len("python\n") : -len(marker) - 1])
    assert "".join(recovered) == body


def test_unclosed_code_fence_is_closed_for_display():
    assert split_message("```python\nprint(1)") == ["```python\nprint(1)\n```"]


def test_code_indentation_and_blank_lines_survive():
    text = "```python\ndef f():\n    return 1\n\n# next\n```"
    assert split_message(text) == [text]


def test_mixed_text_and_code_stays_bounded_and_balanced():
    text = "intro\n\n```python\n" + "x = 1\n" * 500 + "```\n\nend"
    chunks = split_message(text)
    assert chunks[0].startswith("intro")
    assert chunks[-1].endswith("end")
    assert all(len(chunk) <= 2000 and chunk.count("```") % 2 == 0 for chunk in chunks)
