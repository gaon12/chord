"""Exchange-rate skill - ECB reference rates via Frankfurter (key-less).

Frankfurter (https://frankfurter.dev) republishes European Central Bank
reference rates for ~30 currencies, including KRW, JPY, EUR and USD.
"""

from __future__ import annotations

import re
from typing import ClassVar

from chord.skills._http import SkillHTTPError, get_json
from chord.skills.base import Skill

FRANKFURTER_URL = "https://api.frankfurter.dev/v1/latest"

#: ISO-4217 style check so obviously bad input fails before the request.
CURRENCY_RE = re.compile(r"^[A-Za-z]{3}$")


def format_amount(value: float) -> str:
    """Format a number with thousands separators, trimming .0."""
    if float(value).is_integer():
        return f"{int(value):,}"
    return f"{value:,.2f}"


class ExchangeRateSkill(Skill):
    name = "get_exchange_rate"
    description = (
        "Convert money between two currencies using the latest ECB "
        "reference exchange rates, e.g. USD to KRW."
    )
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "base": {
                "type": "string",
                "description": "Currency to convert from, 3-letter code (e.g. 'USD').",
            },
            "target": {
                "type": "string",
                "description": "Currency to convert to, 3-letter code (e.g. 'KRW').",
            },
            "amount": {
                "type": "number",
                "description": "Amount of base currency to convert (default 1).",
            },
        },
        "required": ["base", "target"],
    }

    async def run(self, base: str, target: str, amount: float = 1.0) -> str:
        base = base.strip().upper()
        target = target.strip().upper()
        for code in (base, target):
            if not CURRENCY_RE.match(code):
                raise SkillHTTPError(f"'{code}' is not a valid 3-letter currency code.")

        data = await get_json(
            FRANKFURTER_URL,
            params={"base": base, "symbols": target},
        )
        rates = data.get("rates") or {}
        if target not in rates:
            raise SkillHTTPError(f"No rate available from {base} to {target}.")

        rate = float(rates[target])
        converted = amount * rate
        rate_date = data.get("date", "latest")

        return (
            f"{format_amount(amount)} {base} = {format_amount(converted)} {target} "
            f"(1 {base} = {rate:,g} {target}, ECB reference rate of {rate_date})."
        )
