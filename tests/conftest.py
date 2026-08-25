"""Shared pytest fixtures.

The quota store fixture redirects every Settings instance created in
tests to a per-test JSON file, so quota counters never leak between
tests and never touch a real ``usage.json`` in the repo root.

The font fixture does the same for the chart font: the cache goes to a
per-test directory, and the resolved path - which chord.fonts memoizes
for the whole process - is cleared around every test so no test inherits
another one's answer.
"""

from __future__ import annotations

import pytest

from chord.fonts import forget_resolved_font


@pytest.fixture(autouse=True)
def _isolated_quota_store(tmp_path, monkeypatch):
    monkeypatch.setenv("QUOTA_STORE_PATH", str(tmp_path / "usage.json"))


@pytest.fixture(autouse=True)
def _isolated_font_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("FONT_CACHE_DIR", str(tmp_path / "fonts"))
    forget_resolved_font()
    yield
    forget_resolved_font()
