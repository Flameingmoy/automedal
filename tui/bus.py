"""Tiny async pub/sub used by sources → widgets.

One producer feeds many queues. Each subscriber owns a queue and consumes at its own rate.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

from tui.events import Event


class EventBus:
    def __init__(self) -> None:
        self._subscribers: list[asyncio.Queue[Event]] = []
        self._closed = False

    def subscribe(self, maxsize: int = 2048) -> asyncio.Queue[Event]:
        q: asyncio.Queue[Event] = asyncio.Queue(maxsize=maxsize)
        self._subscribers.append(q)
        return q

    async def publish(self, event: Event) -> None:
        if self._closed:
            return
        for q in self._subscribers:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                # Drop the oldest item to keep latency bounded.
                try:
                    q.get_nowait()
                    q.put_nowait(event)
                except Exception:
                    pass

    def publish_nowait(self, event: Event) -> None:
        if self._closed:
            return
        for q in self._subscribers:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                try:
                    q.get_nowait()
                    q.put_nowait(event)
                except Exception:
                    pass

    async def stream(self, q: asyncio.Queue[Event]) -> AsyncIterator[Event]:
        while not self._closed:
            yield await q.get()

    def close(self) -> None:
        self._closed = True
