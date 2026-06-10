"""Redis Streams queue — the production implementation of relay_core.queue.Queue
(DESIGN.md §5.4, ADR-6).

The whole point of choosing Streams over a plain list is **at-least-once
delivery with crash recovery**, which a list cannot give you:

  * ``XADD``       enqueue,
  * ``XREADGROUP`` pop into a consumer group (entries become "pending" until
    acked) — this is the at-least-once read,
  * ``XACK``       acknowledge once the batch's results are durably recorded,
  * ``XAUTOCLAIM`` reclaim entries a dead worker/consumer left pending past the
    lease timeout, reprocessed oldest-first by stream ID (DESIGN.md §5.2).

Nothing above the queue layer changes between this and ``LocalQueue``; the batch
former and simulator use the same four-method shape. ``peek`` over a consumer
group is necessarily an approximation (Streams have no non-consuming head read
within a group), so the live former reads a 1-entry batch to inspect the head;
that detail is documented here and kept out of the pure helper API.

NEEDS REDIS: imports ``redis.asyncio``; excluded from the default import path.
"""

from __future__ import annotations

from typing import Optional

from relay_core.types import InferItem, InferParams, Priority


class RedisStreamQueue:
    def __init__(
        self,
        redis,
        model: str,
        priority: Priority,
        group: str = "scheduler",
        consumer: str = "former-1",
    ) -> None:
        self.redis = redis
        self.key = f"queue:{model}:{priority.value}"
        self.group = group
        self.consumer = consumer

    async def ensure_group(self) -> None:
        # MKSTREAM so the group can be created before the first XADD.
        try:
            await self.redis.xgroup_create(self.key, self.group, id="0", mkstream=True)
        except Exception:
            pass  # BUSYGROUP: already exists — fine.

    async def push(self, item: InferItem) -> None:
        await self.redis.xadd(self.key, _encode(item))

    async def size(self) -> int:
        return int(await self.redis.xlen(self.key))

    async def read(self, n: int) -> list[tuple[str, InferItem]]:
        """XREADGROUP up to ``n`` new entries; returns (entry_id, item) pairs that
        must later be ``ack``-ed. This is the at-least-once read path."""
        resp = await self.redis.xreadgroup(
            self.group, self.consumer, {self.key: ">"}, count=n, block=0
        )
        out: list[tuple[str, InferItem]] = []
        for _stream, entries in resp or []:
            for entry_id, fields in entries:
                out.append((entry_id, _decode(fields)))
        return out

    async def ack(self, *entry_ids: str) -> None:
        if entry_ids:
            await self.redis.xack(self.key, self.group, *entry_ids)

    async def reclaim(self, min_idle_ms: int = 30_000, count: int = 64):
        """XAUTOCLAIM entries pending longer than the lease timeout — recovers a
        dead worker's unacked work, oldest-first (DESIGN.md §5.2)."""
        return await self.redis.xautoclaim(
            self.key, self.group, self.consumer, min_idle_time=min_idle_ms, count=count
        )


def _encode(item: InferItem) -> dict:
    return {
        "request_id": item.request_id,
        "input": item.input,
        "prefix_hash": item.prefix_hash,
        "max_tokens": item.params.max_tokens,
        "temperature": item.params.temperature,
        "top_p": item.params.top_p,
        "stream": int(item.params.stream),
        "enqueue_ts": item.enqueue_ts,
        "priority": item.priority.value,
    }


def _decode(fields: dict) -> InferItem:
    return InferItem(
        request_id=fields["request_id"],
        input=fields.get("input", ""),
        params=InferParams(
            max_tokens=int(fields.get("max_tokens", 128)),
            temperature=float(fields.get("temperature", 0.7)),
            top_p=float(fields.get("top_p", 1.0)),
            stream=bool(int(fields.get("stream", 0))),
        ),
        prefix_hash=fields.get("prefix_hash", ""),
        enqueue_ts=float(fields.get("enqueue_ts", 0.0)),
        priority=Priority(fields.get("priority", "default")),
    )
