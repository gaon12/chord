"""Entry point so the bot starts with ``python -m chord`` or the
``chord`` console script."""

from __future__ import annotations

import logging

from chord.bot import build_bot
from chord.config import load_settings


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    settings = load_settings()
    bot = build_bot(settings)
    # discord.py owns the event loop and blocks until Ctrl+C.
    bot.run(settings.discord_token)


if __name__ == "__main__":
    main()
