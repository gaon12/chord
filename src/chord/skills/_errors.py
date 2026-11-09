"""Errors whose messages are intentionally safe to show in a chat."""


class SkillError(Exception):
    """An actionable message written by a skill, without provider secrets."""
