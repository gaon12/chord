"""Tests for chord.__main__ - logging setup around the entry point."""

from __future__ import annotations

import logging

import pytest

from chord.__main__ import NOISY_LOGGERS, apply_log_level


@pytest.fixture(autouse=True)
def _restore_levels():
    """Logging is global state; put every level back afterwards."""
    root = logging.getLogger()
    saved = [(root, root.level)] + [
        (logging.getLogger(name), logging.getLogger(name).level) for name in NOISY_LOGGERS
    ]
    yield
    for logger, level in saved:
        logger.setLevel(level)


def test_log_level_is_applied_to_the_root_logger():
    apply_log_level("WARNING")
    assert logging.getLogger().level == logging.WARNING


def test_debug_keeps_chatty_libraries_at_info():
    """DEBUG exists to read the bot's own diagnostics, not every heartbeat."""
    apply_log_level("DEBUG")

    assert logging.getLogger().level == logging.DEBUG
    assert logging.getLogger("chord.bot").getEffectiveLevel() == logging.DEBUG
    assert all(logging.getLogger(name).level == logging.INFO for name in NOISY_LOGGERS)


def test_info_leaves_library_loggers_alone():
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.NOTSET)

    apply_log_level("INFO")

    assert all(logging.getLogger(name).level == logging.NOTSET for name in NOISY_LOGGERS)
