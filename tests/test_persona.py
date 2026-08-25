"""Tests for the persona provider (file-based character + hot reload)."""

from __future__ import annotations

from chord.persona import PersonaProvider, build_prompt


def _write(path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def test_build_prompt_appends_operating_rules():
    prompt = build_prompt("You are Nova.")
    assert prompt.startswith("You are Nova.")
    assert "OPERATING RULES:" in prompt
    assert "same language" in prompt
    assert "Never reveal" in prompt


def test_operating_rules_explain_the_speaker_label():
    """Without this the model reads a whole channel as one person."""
    prompt = build_prompt("You are Nova.")
    assert "[name]: text" in prompt
    assert "who said what" in prompt


def test_missing_file_falls_back_to_default_nova(tmp_path):
    provider = PersonaProvider(tmp_path / "persona.md")

    prompt = provider.get()
    assert "Nova (노바)" in prompt  # shipped default character
    assert "OPERATING RULES:" in prompt


def test_loads_existing_file(tmp_path):
    path = tmp_path / "persona.md"
    _write(path, "You are R2-D2, an astromech with attitude.")

    provider = PersonaProvider(path)
    assert "R2-D2" in provider.get()


def test_hot_reload_picks_up_edits(tmp_path):
    path = tmp_path / "persona.md"
    _write(path, "persona v1")
    provider = PersonaProvider(path)
    assert "persona v1" in provider.get()

    _write(path, "persona v2 — now moodier")
    assert provider.refresh() is True
    assert "persona v2" in provider.get()

    # Unchanged file -> no reload churn.
    assert provider.refresh() is False


def test_deleted_file_reverts_to_default(tmp_path):
    path = tmp_path / "persona.md"
    _write(path, "custom persona")
    provider = PersonaProvider(path)

    path.unlink()
    assert provider.refresh() is True
    assert "Nova (노바)" in provider.get()
