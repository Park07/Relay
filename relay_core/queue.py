"""Queue abstraction shared by the batch former, the simulator, and tests.

The scheduler's batch former only needs four operations: ``size``, ``peek``,
``pop(n)``, ``push``. In production these map onto a **Redis Stream** consumer
group (``XADD`` to enqueue, ``XREADGROUP`` to pop with at-least-once delivery,
``XACK`` on completion, ``XAUTOCLAIM`` to reclaim a dead worker's unacked
entries — DESIGN.md ADR-6/§5.2). For local runs, unit tests, and the in-process
benchmark we use ``LocalQueue``, an ordinary in-memory FIFO with the same shape.

``services/scheduler/redis_stream_queue.py`` provides the Redis implementation
of this same Protocol; nothing above the queue layer changes between them.
"""

from __future__ import annotations

from collections import deque
from typing import Protocol

from relay_core.types import InferItem


class Queue(Protocol):
    def size(self) -> int: ...
    def peek(self) -> InferItem | None: ...
    def pop(self, n: int) -> list[InferItem]: ...
    def push(self, item: InferItem) -> None: ...


class LocalQueue:
    """In-memory FIFO. Oldest item is at the head (index 0), matching the
    oldest-first reprocessing order Redis Streams give via stream-ID ordering.
    """

    def __init__(self) -> None:
        self._dq: deque[InferItem] = deque()

    def size(self) -> int:
        return len(self._dq)

    def peek(self) -> InferItem | None:
        return self._dq[0] if self._dq else None

    def pop(self, n: int) -> list[InferItem]:
        n = max(0, min(n, len(self._dq)))
        return [self._dq.popleft() for _ in range(n)]

    def push(self, item: InferItem) -> None:
        self._dq.append(item)
