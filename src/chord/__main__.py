"""Entry point so the bot starts with ``python -m chord`` or the
``chord`` console script."""

from __future__ import annotations

import logging

from chord.bot import build_bot
from chord.config import load_settings

#: Libraries that log every heartbeat, request and SSE frame at DEBUG.
#: Letting them through would bury the bot's own diagnostics, which are
#: the reason to turn DEBUG on in the first place.
NOISY_LOGGERS = ("discord", "mcp", "anyio")

# HTTP INFO logs include full URLs, and SDK DEBUG logs may include bodies.
# Keep them quiet even when the application's own diagnostics are enabled.
HTTP_LOGGERS = ("httpx", "httpx2", "httpcore", "httpcore2", "openai")


def apply_log_level(level: str) -> None:
    """Raise the root logger to ``level``, keeping third parties quiet."""
    logging.getLogger().setLevel(level)
    for name in HTTP_LOGGERS:
        logging.getLogger(name).setLevel(max(logging.WARNING, logging.getLogger().level))
    if logging.getLogger().getEffectiveLevel() < logging.INFO:
        for name in NOISY_LOGGERS:
            logging.getLogger(name).setLevel(logging.INFO)


def main() -> None:
    # INFO until settings are read, so configuration errors are visible
    # even when the file that sets LOG_LEVEL is the broken one.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    settings = load_settings()
    apply_log_level(settings.log_level)
    bot = build_bot(settings)
    # discord.py owns the event loop and blocks until Ctrl+C.
    bot.run(settings.discord_token)


if __name__ == "__main__":
    main()
