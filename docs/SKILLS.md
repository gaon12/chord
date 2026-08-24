# Adding a skill (SKILLS.md)

A *skill* is one tool the LLM can call. Skills are plugins: a single
file in `src/chord/skills/` is all it takes.

## Quick recipe

Create `src/chord/skills/my_skill.py`:

```python
from typing import ClassVar

from chord.skills.base import Skill


class MySkill(Skill):
    name = "my_tool"                      # the function name the model calls
    description = (                       # tell the model WHEN to use it
        "Do something useful with X. Use when the user asks about X."
    )
    parameters: ClassVar[dict] = {        # JSON Schema for arguments
        "type": "object",
        "properties": {"x": {"type": "string", "description": "..."}},
        "required": ["x"],
    }

    async def run(self, x: str) -> str:
        return f"did something with {x}"
```

That is it - `create_default_registry()` scans every public module in
`chord/skills/`, finds concrete classes ending in `Skill`, and registers
them automatically. No registration edits anywhere.

## Constructor injection

Declare what you need as constructor parameters; they are filled by name:

| Parameter | Receives |
|---|---|
| `settings` | shared `Settings` object (API keys, quota path, ...) |
| `llm` | one shared `LLMService` for LLM-backed skills |

Unknown required parameters fail at startup with a log line, and only
that plugin is skipped.

## HTTP helpers

Prefer the shared wrappers over raw httpx - they add timeouts, a
browser-ish User-Agent, and turn transport/status failures into one
exception type (`SkillHTTPError`) that the registry renders as readable
text and that triggers graceful provider fallbacks.

```python
from chord.skills._http import get_json, get_text, SkillHTTPError

data = await get_json(url, params={...}, headers={...})
```

City lookups: `from chord.skills._geo import geocode`.

## Quota integration

For metered upstreams add an entry to `LIMITS` in
`chord/skills/_quota.py`, then guard calls:

```python
quota.require("my_provider")   # raises QuotaExceededError when spent
... call ...
quota.record("my_provider")
```

Because `QuotaExceededError` is a `SkillHTTPError`, multi-source skills
fall back to their key-less providers automatically.

## Output guidelines

- Clean text, not JSON dumps.
- Short lines; name the source (`[via Open-Meteo]`).
- Raise `SkillHTTPError("human message")` instead of leaking traces.

## Testing

Mock all network with `respx`; inject fakes for the LLM.

```python
@respx.mock
async def test_my_skill():
    respx.get("https://api.example.com").respond(json={"ok": True})
    assert "done" in await MySkill().run()
```
