"""Shared pytest fixtures.

The quota store fixture redirects every Settings instance created in
tests to a per-test JSON file, so quota counters never leak between
tests and never touch a real ``usage.json`` in the repo root.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolated_quota_store(tmp_path, monkeypatch):
    monkeypatch.setenv("QUOTA_STORE_PATH", str(tmp_path / "usage.json"))
