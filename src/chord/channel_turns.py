"""Serialize turns within a channel without keeping idle locks forever."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from weakref import WeakValueDictionary


class ChannelTurns:
    def __init__(self) -> None:
        self._locks: WeakValueDictionary[int, asyncio.Lock] = WeakValueDictionary()

    @asynccontextmanager
    async def hold(self, channel_id: int) -> AsyncIterator[None]:
        lock = self._locks.get(channel_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[channel_id] = lock
        # The local reference keeps this lock alive for owners and waiters.
        # Once they all leave, the weak dictionary forgets the idle channel.
        async with lock:
            yield
