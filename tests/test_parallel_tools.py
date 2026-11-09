"""Parallel lookups stay bounded and must not reorder state-changing calls."""

import asyncio

from chord.skills.base import Skill
from chord.skills.registry import SkillRegistry
from test_tool_catalog import _calls


async def test_parallel_safe_lookups_are_bounded_and_results_keep_call_order():
    registry = SkillRegistry()
    release = asyncio.Event()
    four_running = asyncio.Event()
    active = 0
    maximum = 0

    class Lookup(Skill):
        name = "lookup"
        parallel_safe = True
        parameters = {"type": "object", "properties": {"number": {"type": "integer"}}}

        async def run(self, number):
            nonlocal active, maximum
            active += 1
            maximum = max(maximum, active)
            if active == 4:
                four_running.set()
            await release.wait()
            active -= 1
            return str(number)

    registry.register(Lookup())
    calls = _calls(*[(str(n), "lookup", '{"number":' + str(n) + "}") for n in range(10)])
    task = asyncio.create_task(registry.execute_calls(calls))
    await asyncio.wait_for(four_running.wait(), 2)
    assert maximum == 4
    release.set()
    assert await asyncio.wait_for(task, 2) == [str(n) for n in range(10)]


async def test_writes_are_barriers_between_lookup_groups():
    registry = SkillRegistry()
    events = []

    class Lookup(Skill):
        name = "lookup"
        parallel_safe = True

        async def run(self):
            events.append("lookup-start")
            await asyncio.sleep(0)
            events.append("lookup-end")
            return "read"

    class Write(Skill):
        name = "write"

        async def run(self):
            events.append("write")
            return "written"

    registry.register(Lookup())
    registry.register(Write())
    result = await registry.execute_calls(
        _calls(
            ("a", "lookup", "{}"),
            ("b", "lookup", "{}"),
            ("c", "write", "{}"),
            ("d", "lookup", "{}"),
        )
    )
    assert result == ["read", "read", "written", "read"]
    assert events == [
        "lookup-start",
        "lookup-start",
        "lookup-end",
        "lookup-end",
        "write",
        "lookup-start",
        "lookup-end",
    ]


async def test_one_failed_lookup_does_not_cancel_other_results():
    registry = SkillRegistry()

    class Lookup(Skill):
        name = "lookup"
        parallel_safe = True

        async def run(self, fail=False):
            if fail:
                raise RuntimeError("private error")
            return "ok"

    registry.register(Lookup())
    results = await registry.execute_calls(
        _calls(
            ("a", "lookup", '{"fail":true}'),
            ("b", "lookup", "{}"),
        )
    )
    assert results[0].startswith("Error")
    assert "private error" not in results[0]
    assert results[1] == "ok"
